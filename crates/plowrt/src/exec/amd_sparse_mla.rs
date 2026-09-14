use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR, TENSOR_NONE16};

use packet::dev::PrefillSpan;

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

/// Every row of the fixed-width (2048-key) CSR must own all 2048 causal keys: a span's first
/// row sits at position `kv_row0`, so `kv_row0 >= SPAN_MIN_PRIOR` is the admission rule the
/// scheduler applies before it packs a span onto a sparse rung (`AmdEngine::packed_span_admissible`).
pub(crate) const SPAN_MIN_PRIOR: u32 = 2047;

/// `PLOW_NATIVE_LAUNCH_TIMING`: drain after each launch of a native route and report the split.
pub(super) struct SplitTimer {
    laps: Option<(std::time::Instant, Vec<(&'static str, f64)>)>,
}

impl SplitTimer {
    pub fn start(be: &HsaBackend) -> Result<Self> {
        let on = crate::config::RuntimeConfig::get().amd.native_launch_timing;
        if on {
            be.synchronize()?;
        }
        Ok(Self {
            laps: on.then(|| (std::time::Instant::now(), Vec::new())),
        })
    }

    pub fn lap(&mut self, be: &HsaBackend, name: &'static str) -> Result<()> {
        if let Some((t, laps)) = &mut self.laps {
            be.synchronize()?;
            laps.push((name, t.elapsed().as_secs_f64() * 1e6));
            *t = std::time::Instant::now();
        }
        Ok(())
    }

    pub fn report(&self, route: &str, rows: u32) {
        if let Some((_, laps)) = &self.laps {
            let parts: Vec<String> = laps.iter().map(|(n, us)| format!("{n}_us={us:.1}")).collect();
            eprintln!("PLOW_NATIVE_LAUNCH_TIMING route={route} rows={rows} {}", parts.join(" "));
        }
    }
}

const OBJECT_HASH: &str = "cd8fa62e18abada15beeeedd49357bcc1e9e2eee7353d038533f30cac93c3607";
const OBJECT: &str = "mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co";

/// PLOW_GLM_ROWSPLIT_ATTN's 16-head object (`rowsplit-attention-design.md`): four launches of
/// this, `mqa=16` each, cover the 64 heads a row-split rung's row band needs. Same 320-byte
/// `uint64_t[40]` kernarg ABI as [`OBJECT`] — `rowsplit_probe.cpp` launches both through one
/// `args[40]` buffer — so [`load_attention16`] mirrors [`load_attention`]'s checks exactly.
const OBJECT16_HASH: &str = "8c615201a6687d97ec2b00bbafd5377caadc6cfe01f1c6669135bd9eec08b5bb";
/// Also read by [`super::amd::AmdEngine::load_rank`] to gate `check_sparse_fp8_packet`'s
/// row-split qualification on whether this file is actually installed.
pub(super) const OBJECT16: &str = "mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co";

pub(super) fn union_handle(inst: &DevInst64) -> Option<u32> {
    if inst.op == DevOp::FlashMlaPrefill as u16 && inst.t[7] != TENSOR_NONE16 {
        Some(u32::from(inst.t[7]))
    } else if inst.op == DevOp::FlashMlaPrefillFp8 as u16 {
        inst.fj[1].checked_sub(1)
    } else {
        None
    }
}

/// Per segment: a sparse flash segment reading the table of the lone `IndexUnionPf` in it, when
/// every reader of that table is a natively routed sparse flash (a union also feeds the flashes of
/// the layers that reuse its indexer). All of a program's sparse routes rebase with the same
/// rows/prior, so while that route is active none reads the union and the host may skip its
/// segment. `routes` is the output of [`routes`].
pub(super) fn unread_unions(prog: &DevProg, routes: &[Option<Route>]) -> Vec<Option<usize>> {
    let n = prog.insts.len();
    let mut owner = vec![None::<usize>; n];
    let mut clean = vec![true; n];
    for e in prog.stream.iter().chain(&prog.gq_stream) {
        let (i, seg) = (e.inst as usize, e.seg as usize);
        if i >= n {
            continue;
        }
        if owner[i].is_some_and(|s| s != seg) || e.succ_len != 0 || e.flags & SE_XCTR != 0 {
            clean[i] = false;
        }
        owner[i] = Some(seg);
    }
    let mut seg_insts = vec![0usize; routes.len()];
    for s in owner.iter().flatten() {
        if let Some(c) = seg_insts.get_mut(*s) {
            *c += 1;
        }
    }
    let mut out = vec![None; routes.len()];
    for (ix, u) in prog.insts.iter().enumerate() {
        if u.op != DevOp::IndexUnionPf as u16 || !clean[ix] {
            continue;
        }
        let Some(seg) = owner[ix].filter(|&s| seg_insts.get(s) == Some(&1)) else {
            continue;
        };
        let table = u32::from(u.t[0]);
        let mut first = None;
        let all_routed = prog.insts[ix + 1..]
            .iter()
            .enumerate()
            .take_while(|(_, d)| !(d.op == DevOp::IndexUnionPf as u16 && d.t[0] == u.t[0]))
            .filter(|(_, d)| {
                d.t.contains(&u.t[0]) || d.t.contains(&u.t[1]) || union_handle(d) == Some(table)
            })
            .all(|(k, d)| {
                let routed = owner[ix + 1 + k].filter(|&f| routes.get(f).is_some_and(Option::is_some));
                first = first.or(routed);
                union_handle(d) == Some(table) && !d.t.contains(&u.t[1]) && routed.is_some()
            });
        if all_routed {
            out[seg] = first;
        }
    }
    out
}

/// The segments of `prog` whose union `PLOW_AMD_UNION_SKIP` skips while its sparse routes are
/// active: the load-time table, for tools that plan without an engine.
pub(crate) fn skippable_unions(prog: &DevProg, tensors: &[DevTensor]) -> Result<Vec<usize>> {
    let segments = prog
        .stream
        .iter()
        .chain(&prog.gq_stream)
        .map(|e| e.seg as usize + 1)
        .max()
        .unwrap_or(0);
    let routes = routes(prog, tensors, segments)?;
    Ok(unread_unions(prog, &routes)
        .iter()
        .enumerate()
        .filter_map(|(seg, flash)| flash.map(|_| seg))
        .collect())
}

/// [`unread_unions`] when `PLOW_AMD_UNION_SKIP` is on; otherwise no segment is skippable.
pub(super) fn union_skip_table(on: bool, prog: &DevProg, routes: &[Option<Route>]) -> Vec<Option<usize>> {
    if on {
        unread_unions(prog, routes)
    } else {
        vec![None; routes.len()]
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    index: u16,
    scale: Option<u16>,
    rows: u32,
    kv_len: u32,
    pub active: bool,
    ix: u32,
    union_ix: Option<u32>,
    pub split_row0: u32,
    pub native_lo: bool,
    /// The flash reads a local-band TP selection: this rank's rows start at row 0.
    local_index: bool,
}

impl Route {
    pub fn rebase(&mut self, rows: u32, prior: u32, row_split: bool, native_lo: bool) -> Result<()> {
        // Row-split (nh=64, a row-split sibling program): `rows` is the bucket-wide count, `i[4]`
        // this rank's band. The runtime selects the sibling only for a full chunk whose every band
        // has a key plan (`rowsplit_chunk_prog`, `band_keys`); anything else has no route here.
        if self.inst.i[1] == 64 {
            if rows != self.inst.i[4] * 8
                || !rowsplit_prior_ok(prior, self.inst.i[4])
                || prior.checked_add(rows).is_none_or(|n| n > self.inst.i[2])
            {
                return Err(RuntimeError::Device(format!(
                    "row-split sparse MLA needs a full {}-row chunk whose bands have a key plan \
                     (rows={rows}, prior={prior})",
                    self.inst.i[4] * 8
                )));
            }
            self.rows = self.inst.i[4];
            self.kv_len = prior + rows;
            self.active = true;
            self.split_row0 = 0;
            self.native_lo = false;
            return Ok(());
        }
        if rows > self.inst.i[4] || prior.checked_add(rows).is_none_or(|n| n > self.inst.i[2]) {
            return Err(RuntimeError::Device(
                "sparse MLA chunk exceeds its query/KV capacity".into(),
            ));
        }
        self.rows = rows;
        self.kv_len = prior + rows;
        // Fixed-width CSR is valid only when every row has all 2048 causal keys.
        self.active = rows != 0 && prior >= 2047;
        self.split_row0 = 0;
        self.native_lo = false;
        if row_split && !self.active && rows != 0 && self.union_ix.is_some() {
            // Interpreter rows: 8-aligned so it keeps exactly the unsplit per-8-query union tiles.
            // Native lower rows need no alignment.
            let s = if native_lo {
                SPAN_MIN_PRIOR - prior
            } else {
                (SPAN_MIN_PRIOR - prior).next_multiple_of(8)
            };
            if rows > s {
                self.split_row0 = s;
                self.native_lo = native_lo;
            }
        }
        Ok(())
    }

    /// A 64-head row-split instruction (`PLOW_GLM_ROWSPLIT_ATTN` sibling program).
    pub fn is_rowsplit(&self) -> bool {
        self.inst.i[1] == 64
    }

    /// AQL packets one active whole-sparse `SparseMla::enqueue` is counted as.
    pub fn active_launches(&self) -> usize {
        if self.is_rowsplit() {
            5
        } else {
            3
        }
    }

    /// `(flash, union, rows)`: the instructions whose row count the interpreter half runs with.
    pub fn split_patch(&self) -> Option<(usize, usize, u32)> {
        let union = self.union_ix?;
        (self.split_row0 != 0 && !self.native_lo).then_some((self.ix as usize, union as usize, self.split_row0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR"]
    fn sparse_mla_hsa_dispatch() {
        sparse_mla_hsa(true);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and a default PLOW_TEST_AITER_DIR adapter"]
    fn sparse_mla_legacy_hsa_dispatch() {
        sparse_mla_hsa(false);
    }

    /// Row-split (`nh=64`) dispatch, same uniform-value trick as [`sparse_mla_hsa`]: every
    /// query's 2048 selected keys land on one constant-valued half of the cache, so the softmax
    /// output is exactly that constant regardless of which of the 4 sixteen-head groups wrote
    /// it — this exercises the group-offset addressing in
    /// [`SparseMla::enqueue_window_rowsplit`], not just the reduce math [`sparse_mla_hsa`] covers.
    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the row-split object"]
    fn sparse_mla_rowsplit_hsa_dispatch() {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        const ROWS: u32 = 513;
        const NH: u64 = 64;
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        // `SparseMla`'s internal scratch (q/part/lse/qp/kp/last/splits) is sized off `load`'s
        // own `rows` at the SAME per-row-8-head layout the dense path uses; row-split's pack
        // and reduce calls address it through the `rows*(nh/8)` inflation (see
        // `enqueue_window_rowsplit`'s doc comment), which only fits when this constructor cap
        // is at least that inflated count -- in production this holds because `rows` there is
        // the packet's own T (== this rank's row band x nh/8 exactly). ROWS alone (the real
        // per-rank band) is 8x too small and was this test's own bug (a real device fault, not
        // a runtime regression): see `rowsplit_scratch_fits` below for the checked invariant.
        assert!(rowsplit_scratch_fits(u64::from(ROWS) * 8, u64::from(ROWS)));
        let kernel =
            SparseMla::load(&be, Path::new(&dir), ROWS * 8, 4096, true, &mut modules).unwrap();
        assert!(
            kernel.attention16.is_some(),
            "PLOW_TEST_AITER_DIR must carry the row-split 16-head object"
        );
        let sizes = [
            u64::from(ROWS) * NH * 512 * 4 + 1024,
            u64::from(ROWS) * NH * 2 * 4 + 1024,
            u64::from(ROWS) * NH * 512 * 2,
            u64::from(ROWS) * NH * 64 * 2,
            2 * 4096 * 512 * 2,
            2 * 4096 * 64 * 2,
            4,
            2 * 4096 * 4,
            u64::from(ROWS) * 2048 * 4,
            4,
        ];
        let mut buffers = Vec::new();
        for bytes in sizes {
            let buf = EngineDevice::alloc(&be, bytes).unwrap();
            EngineDevice::upload(&be, &buf, 0, &vec![0; bytes as usize]).unwrap();
            buffers.push(buf);
        }
        // FP8 e4m3 1.0 (0x38) and 2.0 (0x40) at unit scale.
        let mut ck = vec![0x38u8; 2 * 4096 * 512];
        ck[4096 * 512..].fill(0x40);
        EngineDevice::upload(&be, &buffers[4], 0, &ck).unwrap();
        let unit = vec![1f32; 2 * 4096];
        EngineDevice::upload(&be, &buffers[7], 0, bytemuck::cast_slice(&unit)).unwrap();
        let idx: Vec<u32> = (0..ROWS * 2048).map(|i| i % 2048).collect();
        EngineDevice::upload(&be, &buffers[8], 0, bytemuck::cast_slice(&idx)).unwrap();
        let mut route = Route {
            inst: DevInst64 {
                t: [0, 1, 2, 3, 4, 5, 6, 7],
                i: [1, 64, 4096, 0, ROWS, u32::MAX, 0, 0],
                fj: [0.0625f32.to_bits(), 0, 0],
                ..Default::default()
            },
            index: 8,
            scale: Some(7),
            rows: ROWS,
            kv_len: 0,
            active: false,
            ix: 0,
            union_ix: None,
            split_row0: 0,
            native_lo: false,
            local_index: false,
        };
        for slot in 0..2u64 {
            let mut table: Vec<u64> = buffers.iter().map(|m| m.base).collect();
            table[0] += 512;
            table[1] += 512;
            table[4] += slot * 4096 * 512;
            table[5] += slot * 4096 * 64 * 2;
            table[7] += slot * 4096 * 4;
            for rows in [ROWS] {
                for b in [0, 1] {
                    let poison = vec![f32::NAN; buffers[b].len as usize / 4];
                    EngineDevice::upload(&be, &buffers[b], 0, bytemuck::cast_slice(&poison))
                        .unwrap();
                }
                route.rebase(rows * 8, 2048, false, false).unwrap();
                kernel
                    .enqueue(&be, route, bytemuck::cast_slice(&table), 0)
                    .unwrap();
                be.synchronize().unwrap();
                let mut out = vec![0f32; rows as usize * 64 * 512];
                EngineDevice::download(&be, &buffers[0], 512, bytemuck::cast_slice_mut(&mut out))
                    .unwrap();
                assert!(
                    out.iter().all(|&x| (x - (slot + 1) as f32).abs() < 1e-5),
                    "rows={rows} slot={slot}"
                );
                let mut ml = vec![0f32; rows as usize * 64 * 2];
                EngineDevice::download(&be, &buffers[1], 512, bytemuck::cast_slice_mut(&mut ml))
                    .unwrap();
                assert!(ml.chunks_exact(2).all(|x| x == [0.0, 1.0]));
                for (b, width) in [(0, 512), (1, 2)] {
                    for offset in [0, 512 + u64::from(rows) * NH * width * 4] {
                        let mut guard = vec![0f32; 128];
                        EngineDevice::download(
                            &be,
                            &buffers[b],
                            offset,
                            bytemuck::cast_slice_mut(&mut guard),
                        )
                        .unwrap();
                        assert!(guard.iter().all(|x| x.is_nan()), "rows={rows} buffer={b}");
                    }
                }
            }
        }
    }

    fn sparse_mla_hsa(single_supported: bool) {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        const ROWS: u32 = 513;
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        for (fp8, single) in [(false, false), (true, false), (true, true)] {
            if single && !single_supported {
                continue;
            }
            let mut kernel =
                SparseMla::load(&be, Path::new(&dir), ROWS, 4096, fp8, &mut modules).unwrap();
            assert_eq!(kernel.pack_fp8_single.is_some(), fp8 && single_supported);
            if !single {
                kernel.pack_fp8_single = None;
            }
            let sizes = [
                u64::from(ROWS) * 8 * 512 * 4 + 1024,
                u64::from(ROWS) * 8 * 2 * 4 + 1024,
                u64::from(ROWS) * 8 * 512 * 2,
                u64::from(ROWS) * 8 * 64 * 2,
                2 * 4096 * 512 * 2,
                2 * 4096 * 64 * 2,
                4,
                4,
                u64::from(ROWS) * 2048 * 4,
                2 * 4096 * 4,
            ];
            let mut buffers = Vec::new();
            for bytes in sizes {
                let buf = EngineDevice::alloc(&be, bytes).unwrap();
                EngineDevice::upload(&be, &buf, 0, &vec![0; bytes as usize]).unwrap();
                buffers.push(buf);
            }
            let mut ck = vec![0x3f80u16; 2 * 4096 * 512];
            ck[4096 * 512..].fill(0x4000);
            if fp8 {
                let bytes: Vec<u8> = (0..2 * 4096 * 512)
                    .map(|i| if i % 2 == 0 { 0x38 } else { 0x40 })
                    .collect();
                EngineDevice::upload(&be, &buffers[4], 0, &bytes).unwrap();
                let scales: Vec<f32> = (0..2 * 4096)
                    .map(|i| (1 + i / 4096) as f32 * (1.0 + (i % 128) as f32 / 128.0))
                    .collect();
                EngineDevice::upload(&be, &buffers[9], 0, bytemuck::cast_slice(&scales)).unwrap();
            } else {
                EngineDevice::upload(&be, &buffers[4], 0, bytemuck::cast_slice(&ck)).unwrap();
            }
            let idx: Vec<u32> = (0..ROWS * 2048).map(|i| i % 2048).collect();
            EngineDevice::upload(&be, &buffers[8], 0, bytemuck::cast_slice(&idx)).unwrap();
            let mut route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, 5, 6, 7],
                    i: [1, 8, 4096, 0, ROWS, u32::MAX, 0, 0],
                    fj: [0.0625f32.to_bits(), 0, 0],
                    ..Default::default()
                },
                index: 8,
                scale: fp8.then_some(9),
                rows: ROWS,
                kv_len: 0,
                active: false,
                ix: 0,
                union_ix: None,
                split_row0: 0,
                native_lo: false,
                local_index: false,
            };
            for slot in 0..2u64 {
                let mut table: Vec<u64> = buffers.iter().map(|m| m.base).collect();
                table[0] += 512;
                table[1] += 512;
                table[4] += slot * 4096 * 512 * if fp8 { 1 } else { 2 };
                table[9] += slot * 4096 * 4;
                table[5] += slot * 4096 * 64 * 2;
                for rows in [1, 129, 511, 512, ROWS] {
                    for b in [0, 1] {
                        let poison = vec![f32::NAN; buffers[b].len as usize / 4];
                        EngineDevice::upload(&be, &buffers[b], 0, bytemuck::cast_slice(&poison))
                            .unwrap();
                    }
                    route.rebase(rows, 2048, false, false).unwrap();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table), 0)
                        .unwrap();
                    be.synchronize().unwrap();
                    let mut out = vec![0f32; rows as usize * 8 * 512];
                    EngineDevice::download(
                        &be,
                        &buffers[0],
                        512,
                        bytemuck::cast_slice_mut(&mut out),
                    )
                    .unwrap();
                    assert!(out.iter().enumerate().all(|(i, &x)| {
                        let factor = if fp8 {
                            1.49609375 * (1 + i % 2) as f32
                        } else {
                            1.0
                        };
                        (x - (slot + 1) as f32 * factor).abs() < 1e-5
                    }));
                    if fp8 {
                        let mut packed = vec![0u16; (2048 + rows) as usize * 576];
                        EngineDevice::download(
                            &be,
                            &kernel._scratch,
                            kernel.kv - kernel._scratch.base,
                            bytemuck::cast_slice_mut(&mut packed),
                        )
                        .unwrap();
                        for (i, &bits) in packed.iter().enumerate() {
                            let (row, col) = (i / 576, i % 576);
                            let expected = if col < 512 {
                                (slot + 1) as f32
                                    * (1.0 + (row % 128) as f32 / 128.0)
                                    * (1 + col % 2) as f32
                            } else {
                                0.0
                            };
                            assert_eq!(bits, (expected.to_bits() >> 16) as u16);
                        }
                    }
                    let mut ml = vec![0f32; rows as usize * 8 * 2];
                    EngineDevice::download(
                        &be,
                        &buffers[1],
                        512,
                        bytemuck::cast_slice_mut(&mut ml),
                    )
                    .unwrap();
                    assert!(ml.chunks_exact(2).all(|x| x == [0.0, 1.0]));
                    for (b, width) in [(0, 512), (1, 2)] {
                        for offset in [0, 512 + u64::from(rows) * 8 * width * 4] {
                            let mut guard = vec![0f32; 128];
                            EngineDevice::download(
                                &be,
                                &buffers[b],
                                offset,
                                bytemuck::cast_slice_mut(&mut guard),
                            )
                            .unwrap();
                            assert!(
                                guard.iter().all(|x| x.is_nan()),
                                "rows={rows} fp8={fp8} buffer={b}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let union = DevInst64 {
            op: DevOp::IndexUnionPf as u16,
            t: [
                7,
                9,
                8,
                6,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            i: [8192, 2048, 81920, 16384, 8, 0, 0, 0],
            ..Default::default()
        };
        let flash = DevInst64 {
            op: DevOp::FlashMlaPrefill as u16,
            t: [0, 1, 2, 3, 4, 5, 6, 7],
            i: [1, 8, 81920, 0, 8192, u32::MAX, 16384, 4],
            fj: [0.0625f32.to_bits(), 0, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: 8192,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows: 8192 },
            n_counter: 0,
            insts: vec![union, flash],
            stream: vec![StreamEnt {
                inst: 1,
                seg: 1,
                ..Default::default()
            }],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let tensors = (0..10)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 256 << 20,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn unread_unions_name_the_lone_union_segment_and_its_routed_reader() {
        let (mut prog, tensors) = fixture();
        prog.stream.push(StreamEnt {
            inst: 0,
            seg: 0,
            ..Default::default()
        });
        let routed = routes(&prog, &tensors, 2).unwrap();
        assert_eq!(unread_unions(&prog, &routed), vec![Some(1), None]);
        assert_eq!(union_skip_table(true, &prog, &routed), vec![Some(1), None]);
        assert_eq!(union_skip_table(false, &prog, &routed), vec![None, None]);
        assert_eq!(unread_unions(&prog, &[None, None]), vec![None, None]);
        prog.stream[1].seg = 1;
        assert_eq!(unread_unions(&prog, &[None, routed[1]]), vec![None, None]);
        prog.stream[1].seg = 0;
        prog.stream[1].succ_len = 1;
        assert_eq!(unread_unions(&prog, &routed), vec![None, None]);
        prog.stream[1].succ_len = 0;
        let mut reader = prog.insts[1];
        reader.op = DevOp::Residual as u16;
        prog.insts.push(reader);
        assert_eq!(unread_unions(&prog, &routed), vec![None, None]);
        prog.insts[2] = prog.insts[1];
        let mut routed3 = routed.clone();
        routed3.push(None);
        assert_eq!(unread_unions(&prog, &routed3), vec![None, None, None]);
        prog.stream.push(StreamEnt {
            inst: 2,
            seg: 2,
            ..Default::default()
        });
        routed3[2] = routed[1];
        assert_eq!(unread_unions(&prog, &routed3), vec![Some(1), None, None]);
    }

    #[test]
    fn sparse_mla_fp8_routes_require_scales_and_selected_union() {
        let (mut prog, mut tensors) = fixture();
        prog.insts[1].op = DevOp::FlashMlaPrefillFp8 as u16;
        prog.insts[1].fj[1] = 8;
        prog.insts[1].t[7] = 9;
        let route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        assert_eq!(route.scale, Some(9));
        assert_eq!(route.index, 8);
        tensors[9].bytes = 81920 * 4 - 1;
        assert!(routes(&prog, &tensors, 2).is_err());
        tensors[9].bytes = 81920 * 4;
        prog.insts[1].fj[1] = u32::MAX;
        assert!(routes(&prog, &tensors, 2).is_err());
    }

    /// A packed sibling / token-batch body carries no union: its sparse flash names the TP
    /// indexer's per-row selection, which the route gathers directly. An ordinary program may
    /// not take that form, and the geometry must still agree.
    #[test]
    fn sparse_mla_routes_take_the_tp_selection_in_packed_programs_only() {
        let (mut prog, tensors) = fixture();
        prog.insts[0] = DevInst64 {
            op: DevOp::IndexTpPf as u16,
            t: [8, 9, 2, 3, 4, 6, 7, TENSOR_NONE16],
            i: [8192, 81920, 2048, 8, 96 << 20, 0, 2, 0],
            ..Default::default()
        };
        prog.insts[1].op = DevOp::FlashMlaPrefillFp8 as u16;
        prog.insts[1].fj[1] = 9;
        prog.insts[1].t[7] = 9;
        for body in [false, true] {
            prog.role = if body {
                packet::devbuild::ProgramRole::TokenBatchBody {
                    band: 8,
                    rows: prog.t,
                }
            } else {
                packet::devbuild::ProgramRole::PackedSibling { of_rows: prog.t }
            };
            let route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
            assert_eq!((route.index, route.scale), (8, Some(9)));
        }
        prog.role = packet::devbuild::ProgramRole::PrefillBucket { rows: prog.t };
        assert!(routes(&prog, &tensors, 2).is_err(), "ordinary programs keep the union");
        prog.role = packet::devbuild::ProgramRole::PackedSibling { of_rows: prog.t };
        prog.insts[0].i[1] = 65536;
        assert!(routes(&prog, &tensors, 2).is_err(), "ctx must agree");
        prog.insts[0].i[1] = 81920;
        prog.insts[0].t[5] = 5;
        assert!(routes(&prog, &tensors, 2).is_err(), "kv_len operand must agree");
    }

    #[test]
    fn rowsplit_sibling_reads_its_local_band_selection_from_row_zero() {
        let (mut prog, tensors) = fixture();
        prog.insts[0] = DevInst64 {
            op: DevOp::IndexTpPf as u16,
            t: [8, 9, 2, 3, 4, 6, 7, TENSOR_NONE16],
            i: [8192, 81920, 2048, 8, 96 << 20, 0, 2, 1],
            ..Default::default()
        };
        prog.insts[1].op = DevOp::FlashMlaPrefillFp8 as u16;
        prog.insts[1].fj[1] = 9;
        prog.insts[1].t[7] = 9;
        prog.insts[1].i[1] = 64;
        prog.insts[1].i[4] = prog.t / 8;
        prog.role = packet::devbuild::ProgramRole::RowSplitSibling { of_rows: prog.t };
        let route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        assert!(route.local_index && route.is_rowsplit());
        prog.insts[0].i[7] = 0;
        assert!(routes(&prog, &tensors, 2).is_err(), "the sibling's selection must be local");
        prog.insts[0].i[7] = 1;
        prog.insts[1].i[1] = 8;
        prog.insts[1].i[4] = prog.t;
        assert!(routes(&prog, &tensors, 2).is_err(), "only the row-split flash reads a local band");
    }

    #[test]
    fn sparse_mla_rebases_ragged_rows_and_restores_early_fallback() {
        let (prog, tensors) = fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        route.rebase(129, 65536, false, false).unwrap();
        assert!(route.active);
        assert_eq!(route.rows, 129);
        route.rebase(8192, 0, false, false).unwrap();
        assert!(!route.active);
        route.rebase(1, 2046, false, false).unwrap();
        assert!(!route.active);
        route.rebase(1, 2047, false, false).unwrap();
        assert!(route.active);
        assert!(route.rebase(8193, 0, false, false).is_err());
        assert!(route.rebase(129, 81920, false, false).is_err());
    }

    /// Row-split's `i[4]` is the rank's OWN row band (prog.t/8, see `routes`'s `expect_i4`),
    /// never the bucket-wide row count `rebase`'s caller always passes -- comparing the two
    /// directly refused every chunk (tracker: rowsplit-t3b, "sparse MLA chunk exceeds its
    /// query/KV capacity" on chunks 2..9 of a 73728-token walk).
    #[test]
    fn sparse_mla_rebase_accepts_full_bucket_rows_on_row_split() {
        // c0 in {8192, 65536}: both >= SPAN_MIN_PRIOR (2047), the whole-sparse route
        // (`route.active`) -- the only case PLOW_GLM_ROWSPLIT_ATTN's 64-head instruction is
        // used for. c0=0 (prior < SPAN_MIN_PRIOR) is covered separately: that chunk must keep
        // today's tip path (split_row0/native_lo on the ordinary nh=8 instruction) unchanged.
        let (mut prog, tensors) = fixture();
        prog.insts[1].i[1] = 64; // nh=64
        prog.insts[1].i[4] = prog.t / 8; // this rank's row band, per `routes`
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        for c0 in [2047u32, 8192, 65536] {
            route.rebase(8192, c0, true, false).unwrap_or_else(|e| {
                panic!("row-split chunk at c0={c0} refused: {e}");
            });
            assert!(route.active, "c0={c0} must take the whole-sparse route");
            assert_eq!((route.rows, route.kv_len), (1024, c0 + 8192), "the rank's own band");
            assert_eq!((route.split_row0, route.split_patch()), (0, None));
        }
        // No interpreter or split fallback exists for the 64-head instruction: a chunk the
        // runtime would not select the sibling for is refused, never patched.
        for (rows, c0) in [(8192u32, 1u32), (8192, 2046), (4096, 65536), (8191, 65536)] {
            assert!(route.rebase(rows, c0, true, true).is_err(), "rows={rows} c0={c0}");
        }
    }

    /// A c0=0 chunk (prior < SPAN_MIN_PRIOR) with BOTH knobs on must rebase identically to
    /// PLOW_GLM_ROWSPLIT_ATTN off: it is entirely represented by an instruction's OWN `i[1]`
    /// (64 vs 8, see `routes`'s `rowsplit`/`expect_i4`), never by `rebase`'s `row_split` bool
    /// (that flag is PLOW_MLA_PF_ROW_SPLIT's -- an orthogonal, already-default lever that this
    /// test must not conflate: it legitimately changes `split_row0`/`native_lo` regardless of
    /// TP row-split). So an ORDINARY (i[1]=8) instruction's rebase at c0=0 -- the only shape
    /// PLOW_GLM_ROWSPLIT_ATTN's own rows-cap scaling (`i[1]==64`) can affect -- must come out
    /// identical whether or not that scaling exists in the code, since it never applies here.
    #[test]
    fn sparse_mla_rebase_c0_zero_is_unchanged_by_row_split() {
        let (prog, tensors) = fixture();
        let mut off = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        let mut on = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        off.rebase(8192, 0, false, false).unwrap();
        on.rebase(8192, 0, true, false).unwrap();
        assert_eq!(off.active, on.active);
        assert_eq!(off.rows, on.rows);
        assert_eq!(off.kv_len, on.kv_len);
    }

    /// Every packet in `PLOW_LT_PACKETS` that carries a row-split sibling: its 8192 bucket still
    /// skips its unions, and the sibling carries none (its flash names the local-band selection).
    #[test]
    #[ignore = "reads the packets named by PLOW_LT_PACKETS"]
    fn rowsplit_sibling_carries_no_union_and_its_bucket_skips_them() {
        let list = std::env::var("PLOW_LT_PACKETS").expect("PLOW_LT_PACKETS");
        let mut checked = 0;
        for path in list.split(',').filter(|s| !s.is_empty()) {
            let blob =
                crate::asset::devblob::DevBlob::parse_l2(&std::fs::read(path).unwrap(), true).unwrap();
            let Some(sibling) = blob.progs.iter().find(|p| p.role.is_rowsplit_sibling()) else {
                continue;
            };
            let bucket = blob
                .progs
                .iter()
                .find(|p| p.role == packet::devbuild::ProgramRole::PrefillBucket { rows: sibling.t })
                .expect("the sibling's bucket");
            let n = |p: &DevProg| skippable_unions(p, &blob.tensors).unwrap().len();
            eprintln!("{path}: skippable unions bucket={} sibling={}", n(bucket), n(sibling));
            assert!(n(bucket) > 0);
            assert!(sibling.insts.iter().all(|d| d.op != DevOp::IndexUnionPf as u16), "{path}");
            checked += 1;
        }
        assert!(checked > 0, "no packet in PLOW_LT_PACKETS carries a row-split sibling");
    }

    #[test]
    fn sparse_mla_rejects_counter_obligations_in_both_streams() {
        for global in [false, true] {
            for kind in 0..3 {
                let (mut prog, tensors) = fixture();
                prog.gq_stream = prog.stream.clone();
                let entry = if global {
                    &mut prog.gq_stream[0]
                } else {
                    &mut prog.stream[0]
                };
                match kind {
                    0 => entry.wait_len = 1,
                    1 => entry.succ_len = 1,
                    _ => entry.flags = SE_XCTR,
                }
                assert!(routes(&prog, &tensors, 2).is_err());
            }
        }
    }

    #[test]
    fn sparse_mla_row_split_cuts_at_the_first_8_aligned_full_key_row() {
        let (prog, tensors) = fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        for (rows, prior, split, active, want) in [
            (8192, 0, true, false, Some((1, 0, 2048))),
            (4096, 0, true, false, Some((1, 0, 2048))),
            (2048, 0, true, false, None),
            (8192, 100, true, false, Some((1, 0, 1952))),
            (8192, 0, false, false, None),
            (8192, 2047, true, true, None),
        ] {
            route.rebase(rows, prior, split, false).unwrap();
            assert_eq!(route.active, active);
            assert_eq!(route.split_patch(), want, "rows={rows} prior={prior}");
        }
    }

    #[test]
    fn sparse_mla_native_lo_cuts_at_the_first_full_key_row_without_interpreter_rows() {
        let (prog, tensors) = fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        for (rows, prior, split, lo, row0, native) in [
            (8192, 0, true, true, 2047, true),
            (4096, 0, true, true, 2047, true),
            (8192, 100, true, true, 1947, true),
            (2047, 0, true, true, 0, false),
            (8192, 0, false, true, 0, false),
            (8192, 2047, true, true, 0, false),
            (8192, 0, true, false, 2048, false),
        ] {
            route.rebase(rows, prior, split, lo).unwrap();
            assert_eq!((route.split_row0, route.native_lo), (row0, native), "rows={rows} prior={prior}");
            assert_eq!(route.split_patch().is_some(), row0 != 0 && !native);
        }
    }

    #[test]
    fn sparse_mla_native_lo_csr_is_causal_identity_within_its_buffer() {
        for prior in [0u32, 1, 100, 1535, 2046] {
            let rows = SPAN_MIN_PRIOR - prior;
            let (kp, idx) = lo_csr(prior, rows);
            assert_eq!(kp.len(), rows as usize + 1);
            for t in 0..rows as usize {
                let row = &idx[kp[t] as usize..kp[t + 1] as usize];
                assert!(row.iter().copied().eq(0..=prior + t as u32), "prior={prior} t={t}");
            }
            assert!(idx.len() as u64 <= LO_KEYS && (kp.len() as u64) * 4 <= LO_IDX_OFF);
        }
        assert_eq!(lo_csr(0, SPAN_MIN_PRIOR).1.len() as u64, LO_KEYS);
    }

    #[test]
    fn sparse_mla_rejects_mixed_segments_and_wrong_geometry() {
        for kind in 0..6 {
            let (mut prog, mut tensors) = fixture();
            match kind {
                0 => prog.stream.push(StreamEnt {
                    inst: 0,
                    seg: 1,
                    ..Default::default()
                }),
                1 => prog.stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
                2 => prog.insts[0].i[1] = 1024,
                3 => prog.insts[1].i[1] = 16,
                4 => tensors[4].bytes = 16,
                _ => prog.insts[1].t[2] = TENSOR_NONE16,
            }
            assert!(routes(&prog, &tensors, 2).is_err());
        }
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    if prog.t < 2048 {
        return Ok(routes);
    }
    for (ix, inst) in prog.insts.iter().enumerate() {
        let Some(union_handle) = union_handle(inst) else {
            continue;
        };
        let fp8 = inst.op == DevOp::FlashMlaPrefillFp8 as u16;
        let err = |s: &str| RuntimeError::Device(format!("sparse AITER MLA instruction {ix}: {s}"));
        // Row-split (PLOW_GLM_ROWSPLIT_ATTN): nh=64 over this rank's own T/8 row band instead
        // of nh=8 over the whole T. `i[4]` is that band's row count either way — T for the
        // dense qh8 arm, T/8 for row-split — never `prog.t` directly when row-split.
        let nh = inst.i[1];
        let rowsplit = nh == 64;
        let expect_i4 = if rowsplit { prog.t / 8 } else { prog.t };
        if inst.i[0] != 1
            || (nh != 8 && nh != 64)
            || inst.i[2] > 81920
            || inst.i[2] < 2048
            || inst.i[3] != 0
            || (rowsplit && prog.t % 8 != 0)
            || inst.i[4] != expect_i4
            || inst.i[5] != u32::MAX
            || prog.t > 8192
            || (!fp8 && inst.fj[1] != 0)
            || inst.fj[2] != 0
            || !f32::from_bits(inst.fj[0]).is_finite()
            || f32::from_bits(inst.fj[0]) <= 0.0
        {
            return Err(err(
                "requires BF16 QH{8,64} latent512/rope64, rows<=8192, ctx<=81920",
            ));
        }
        // The per-row selection this attention gathers: an ordinary program names the union
        // table (op 119) and the route reads the union's `iidx_pf` operand; a packed sibling /
        // token-batch body carries no union (its 8-query tiles could straddle request spans)
        // and names the TP indexer's per-row selection directly.
        let (producer_ix, producer) = prog.insts[..ix]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, d)| {
                (d.op == DevOp::IndexUnionPf as u16 || d.op == DevOp::IndexTpPf as u16)
                    && u32::from(d.t[0]) == union_handle
            })
            .ok_or_else(|| err("no preceding index union or TP selection"))?;
        let index = if producer.op == DevOp::IndexUnionPf as u16 {
            if producer.i[..5] != [prog.t, 2048, inst.i[2], inst.i[6], 8]
                || producer.t[3] != inst.t[6]
            {
                return Err(err(
                    "index union does not describe 2048 selected keys per query",
                ));
            }
            producer.t[2]
        } else {
            if !(prog.role.is_packed_sibling()
                || prog.role.is_token_batch_body()
                || prog.role.is_rowsplit_sibling())
                || producer.i[..3] != [prog.t, inst.i[2], 2048]
                || producer.t[5] != inst.t[6]
            {
                return Err(err(
                    "a direct TP selection feeds the flash only in a packed program with matching rows/ctx/top2048",
                ));
            }
            producer.t[0]
        };
        let local_index = producer.op == DevOp::IndexTpPf as u16 && producer.i[7] == 1;
        if local_index != (rowsplit && prog.role.is_rowsplit_sibling() && producer.op == DevOp::IndexTpPf as u16) {
            return Err(err("a local-band TP selection feeds exactly the row-split sibling's flash"));
        }
        let ctx = u64::from(inst.i[2]);
        let rows = u64::from(inst.i[4]);
        let nh_elems = u64::from(nh);
        // The CSR/selection table (`index`) is sized at the packet's FULL row count even under
        // row-split — every rank reads its own T/8 band out of the SAME table, at a row offset
        // `enqueue` applies at dispatch time (rank*rows), not a narrower table per rank.
        let index_rows = u64::from(prog.t);
        for (handle, bytes) in [
            (inst.t[0], rows * nh_elems * 512 * 4),
            (inst.t[1], rows * nh_elems * 2 * 4),
            (inst.t[2], rows * nh_elems * 512 * 2),
            (inst.t[3], rows * nh_elems * 64 * 2),
            (inst.t[4], ctx * 512 * if fp8 { 1 } else { 2 }),
            (inst.t[5], ctx * 64 * 2),
            (index, index_rows * 2048 * 4),
        ] {
            if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        if fp8
            && (inst.t[7] == TENSOR_NONE16
                || tensors
                    .get(inst.t[7] as usize)
                    .is_none_or(|t| t.bytes < ctx * 4))
        {
            return Err(err("FP8 scale capacity is insufficient"));
        }
        let mut owner = BTreeSet::new();
        for entry in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            if entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0 {
                return Err(err(
                    "counter obligations remain; re-emit with PLOW_MLA_PF_AITER=1",
                ));
            }
            owner.insert(entry.seg as usize);
        }
        if owner.len() != 1 {
            return Err(err("requires exactly one segment owner"));
        }
        let seg = *owner.first().unwrap();
        if seg >= segments
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            return Err(err("segment contains other interpreter work"));
        }
        routes[seg] = Some(Route {
            inst: *inst,
            index,
            scale: fp8.then_some(inst.t[7]),
            rows: inst.i[4],
            kv_len: 0,
            active: false,
            ix: ix as u32,
            union_ix: (producer.op == DevOp::IndexUnionPf as u16).then_some(producer_ix as u32),
            split_row0: 0,
            native_lo: false,
            local_index,
        });
    }
    Ok(routes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackArgs {
    q: u64,
    kv: u64,
    qa: u64,
    qr: u64,
    ck: u64,
    kr: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
    rows: u32,
    ctx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackFp8Args {
    base: PackArgs,
    scale: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackSingleArgs {
    base: PackFp8Args,
    ml: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceArgs {
    out: u64,
    ml: u64,
    part: u64,
    lse: u64,
    rows: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<PackArgs>() == 88);
const _: () = assert!(std::mem::size_of::<PackFp8Args>() == 96);
const _: () = assert!(std::mem::size_of::<PackSingleArgs>() == 104);
const _: () = assert!(std::mem::size_of::<ReduceArgs>() == 40);

/// Most key entries a native lower half holds: at prior 0, rows `[0, 2047)` with `t + 1` keys each
/// (fewer at any later prior).
const LO_KEYS: u64 = SPAN_MIN_PRIOR as u64 * (SPAN_MIN_PRIOR as u64 + 1) / 2;
const LO_IDX_OFF: u64 = (SPAN_MIN_PRIOR as u64 + 1) * 4;

/// `(kp, kv_indices)` of a native lower half at `prior`: row `t` attends `[0, prior + t]`.
fn lo_csr(prior: u32, rows: u32) -> (Vec<u32>, Vec<u32>) {
    let mut kp = Vec::with_capacity(rows as usize + 1);
    let mut idx = Vec::new();
    kp.push(0);
    for t in 0..rows {
        idx.extend(0..=prior + t);
        kp.push(idx.len() as u32);
    }
    (kp, idx)
}

struct LoCsr {
    mem: DeviceMem,
    prior: std::sync::atomic::AtomicU32,
    rows: std::sync::atomic::AtomicU32,
}

impl LoCsr {
    /// Drop the staged `(prior, rows)` so the next [`Self::stage`] re-uploads unconditionally.
    fn forget(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.prior.store(u32::MAX, Relaxed);
        self.rows.store(u32::MAX, Relaxed);
    }

    /// Upload the identity CSR of `rows` rows after `prior` unless it is already staged.
    fn stage(&self, be: &HsaBackend, prior: u32, rows: u32) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        if self.prior.load(Relaxed) != prior || self.rows.load(Relaxed) != rows {
            // Launches still queued may read the previous layout.
            be.synchronize()?;
            let (kp, idx) = lo_csr(prior, rows);
            EngineDevice::upload(be, &self.mem, 0, bytemuck::cast_slice(&kp))?;
            EngineDevice::upload(be, &self.mem, LO_IDX_OFF, bytemuck::cast_slice(&idx))?;
            self.prior.store(prior, Relaxed);
            self.rows.store(rows, Relaxed);
        }
        Ok(())
    }
}

pub(super) struct SparseMla {
    pack: HsaKernel,
    pack_fp8: Option<HsaKernel>,
    pack_fp8_single: Option<HsaKernel>,
    attention: HsaKernel,
    /// PLOW_GLM_ROWSPLIT_ATTN's 16-head object. `None` when the AITER install has no
    /// `OBJECT16` file — refused at dispatch, not at load, so a deployment that never emits
    /// the row-split arm is unaffected.
    attention16: Option<HsaKernel>,
    reduce: HsaKernel,
    _scratch: DeviceMem,
    /// The `rows` this workspace was sized for (`scratch_sizes`). Row-split addresses `q` /
    /// `part` / `lse` / `qp` / `kp` / `last` / `splits` through the `real_rows*(nh/8)`
    /// inflation `enqueue_window_rowsplit` documents; this is what that inflated count is
    /// checked against before any launch touches the scratch.
    rows_cap: u64,
    q: u64,
    kv: u64,
    part: u64,
    lse: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
    lo: Option<LoCsr>,
}

fn load_attention(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<HsaKernel> {
    let path = dir.join(OBJECT);
    let mut image = std::fs::read(&path)
        .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
    if plow_asset::decode_objects::image_sha256(&image) != OBJECT_HASH {
        return Err(RuntimeError::Device(
            "sparse AITER MLA object hash does not match qualified ABI".into(),
        ));
    }
    // This exact object declares 320 bytes in metadata but leaves KERNARG_SIZE
    // unspecified in its descriptor. ROCr reports the descriptor's zero, unlike HIP.
    // The hash fixes the descriptor at file offset 0x1000; only its size is normalized.
    image[0x1008..0x100c].copy_from_slice(&320u32.to_le_bytes());
    let module = EngineDevice::module_load(be, &image)?;
    let attention = EngineDevice::get_function(
        be,
        &module,
        "_ZN5aiter36mla_a16w16_qh8_qseqlen1_gqaratio8_v3E",
    )?;
    if attention.kernarg_size() != 320
        || attention.private_segment_size() != 0
        || HsaBackend::kernel_lds_bytes(&attention) != 65536
    {
        return Err(RuntimeError::Device(format!(
            "sparse AITER MLA resource ABI mismatch: kernarg={}, private={}, LDS={}",
            attention.kernarg_size(),
            attention.private_segment_size(),
            HsaBackend::kernel_lds_bytes(&attention)
        )));
    }
    modules.push(module);
    Ok(attention)
}

fn load_attention16(be: &HsaBackend, dir: &Path, modules: &mut Vec<Module>) -> Result<HsaKernel> {
    let path = dir.join(OBJECT16);
    let mut image = std::fs::read(&path)
        .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
    if plow_asset::decode_objects::image_sha256(&image) != OBJECT16_HASH {
        return Err(RuntimeError::Device(
            "row-split 16-head sparse AITER MLA object hash does not match qualified ABI".into(),
        ));
    }
    // Same AITER-toolchain quirk as [`OBJECT`] (`load_attention`'s comment): the kernel
    // descriptor's `kernarg_size` field is left zero in the file; ROCr reports that zero,
    // unlike HIP (which is why `rowsplit_probe.cpp`'s `hipModuleLaunchKernel` path never sees
    // this). This object's `.kd` symbol (`llvm-readelf -s`) sits at file offset 0xf40, not
    // 0x1000 like `OBJECT` — a different pinned file, a different offset — so `kernarg_size`
    // (descriptor offset +8) is at 0xf48. Checked zero, not just overwritten: a re-pin of
    // OBJECT16_HASH built by a fixed toolchain would carry a real value here, and blindly
    // stomping it would silently run the wrong kernarg layout.
    if image[0xf48..0xf4c] != [0, 0, 0, 0] {
        return Err(RuntimeError::Device(
            "row-split 16-head sparse AITER MLA object's kernarg_size field is no longer the \
             known-zero quirk this patch assumes — re-derive the offset before trusting it"
                .into(),
        ));
    }
    image[0xf48..0xf4c].copy_from_slice(&320u32.to_le_bytes());
    let module = EngineDevice::module_load(be, &image)?;
    let attention16 = EngineDevice::get_function(
        be,
        &module,
        "_ZN5aiter39mla_dec_stage1_bf16_a16w16_subQ16_mqa16E",
    )?;
    if attention16.kernarg_size() != 320
        || attention16.private_segment_size() != 0
        || HsaBackend::kernel_lds_bytes(&attention16) != 65536
    {
        return Err(RuntimeError::Device(format!(
            "row-split 16-head sparse AITER MLA resource ABI mismatch: kernarg={}, private={}, LDS={}",
            attention16.kernarg_size(),
            attention16.private_segment_size(),
            HsaBackend::kernel_lds_bytes(&attention16)
        )));
    }
    modules.push(module);
    Ok(attention16)
}

/// Byte size of each of `SparseMla`'s scratch regions (`q, kv, part, lse, qp, kp, last,
/// splits`, in that order) at `rows` rows / `ctx` context. Every field but `kv` (context-cache
/// sized, head-count-independent) is monotonic non-decreasing in `rows` — this is what makes
/// row-split's `rows*(nh/8)` inflation (`enqueue_window_rowsplit`) safe exactly when the
/// inflated count is `<=` the `rows` this was called with at construction (`rows_cap`).
fn scratch_sizes(rows: u64, ctx: u64) -> [u64; 8] {
    [
        rows * 8 * 576 * 2,
        ctx * 576 * 2,
        rows * 2 * 8 * 512 * 4,
        rows * 2 * 8 * 4,
        (rows + 1) * 4,
        (rows + 1) * 4,
        (rows + 1) * 4,
        (rows + 1) * 4,
    ]
}

/// Row-split touches every ROWS-indexed scratch region (all but `kv`, which is `ctx`-sized and
/// head-count-independent — the same at load time and at dispatch time either way) as if it had
/// `real_rows*8` rows (the inflation `enqueue_window_rowsplit` and its doc comment describe).
/// This holds exactly when that inflated count fits the capacity the workspace was built with.
fn rowsplit_scratch_fits(rows_cap: u64, real_rows: u64) -> bool {
    let touched = scratch_sizes(real_rows * 8, 0);
    let allocated = scratch_sizes(rows_cap, 0);
    touched
        .iter()
        .zip(&allocated)
        .enumerate()
        .all(|(i, (t, a))| i == 1 || t <= a)
}

#[cfg(test)]
mod rowsplit_scratch_tests {
    use super::*;

    #[test]
    fn tight_capacity_at_the_production_invariant_fits() {
        // Production's own invariant: `SparseMla::load`'s `rows` cap is the packet's T (the
        // MAX prefill row count across candidate programs), and row-split only runs at the
        // rung where `real_rows == T/8` exactly — so `real_rows*8 == rows_cap`, no slack.
        assert!(rowsplit_scratch_fits(8192, 1024));
        // One row short of that exact fit still fits (equality, not a hard boundary).
        assert!(rowsplit_scratch_fits(8192, 1023));
    }

    #[test]
    fn a_capacity_sized_for_the_real_row_count_alone_does_not_fit() {
        // The exact bug this test exists to catch: sizing the workspace for `real_rows`
        // instead of `real_rows*(nh/8)` undersizes every region 8x and faults on device.
        assert!(!rowsplit_scratch_fits(513, 513));
        assert!(!rowsplit_scratch_fits(1024, 1024));
    }

    #[test]
    fn one_row_over_capacity_does_not_fit() {
        assert!(!rowsplit_scratch_fits(8191, 1024));
    }
}

impl SparseMla {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        rows: u32,
        ctx: u32,
        fp8: bool,
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
        let attention = load_attention(be, dir, modules)?;
        let attention16 = match load_attention16(be, dir, modules) {
            Ok(k) => Some(k),
            Err(_) if !dir.join(OBJECT16).exists() => None,
            Err(e) => return Err(e),
        };
        let path = dir.join("mla_sparse_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&"plow_mla_sparse_adapter_abi_1") {
            return Err(RuntimeError::Device(
                "sparse MLA adapter lacks ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let pack = EngineDevice::get_function(be, &module, "plow_mla_sparse_pack")?;
        let pack_fp8 = if fp8 {
            if !super::amd::elf_symbol_names(&image).contains(&"plow_mla_sparse_adapter_fp8_abi_1")
            {
                return Err(RuntimeError::Device(
                    "sparse MLA adapter lacks FP8 ABI marker".into(),
                ));
            }
            let kernel = EngineDevice::get_function(be, &module, "plow_mla_sparse_pack_fp8")?;
            if ![96, 352].contains(&kernel.kernarg_size()) || kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(
                    "sparse MLA FP8 pack resource ABI mismatch".into(),
                ));
            }
            Some(kernel)
        } else {
            None
        };
        let reduce = EngineDevice::get_function(be, &module, "plow_mla_sparse_reduce")?;
        let pack_fp8_single = if fp8
            && super::amd::elf_symbol_names(&image).contains(&"plow_mla_sparse_single_abi_1")
        {
            let kernel =
                EngineDevice::get_function(be, &module, "plow_mla_sparse_pack_fp8_single")?;
            if ![104, 360].contains(&kernel.kernarg_size()) || kernel.private_segment_size() != 0 {
                return Err(RuntimeError::Device(
                    "sparse MLA single-pass pack resource ABI mismatch".into(),
                ));
            }
            Some(kernel)
        } else {
            None
        };
        for (kernel, size) in [(pack, 88), (reduce, 40)] {
            if ![size, size + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(
                    "sparse MLA adapter resource ABI mismatch".into(),
                ));
            }
        }
        modules.push(module);
        let rows = u64::from(rows);
        let sizes = scratch_sizes(rows, u64::from(ctx));
        let bytes = sizes.iter().map(|s| s.div_ceil(256) * 256).sum();
        let scratch = EngineDevice::alloc(be, bytes)?;
        let mut ptr = scratch.base;
        let offsets = sizes.map(|size| {
            let p = ptr;
            ptr += size.div_ceil(256) * 256;
            p
        });
        let [q, kv, part, lse, qp, kp, last, splits] = offsets;
        // The row-band sibling's bands below row 2047 read the same identity CSR.
        let lo = if crate::config::RuntimeConfig::get().amd.mla_pf_row_split_native_lo
            || attention16.is_some()
        {
            Some(LoCsr {
                mem: EngineDevice::alloc(be, (LO_IDX_OFF + LO_KEYS * 4).div_ceil(256) * 256)?,
                prior: std::sync::atomic::AtomicU32::new(u32::MAX),
                rows: std::sync::atomic::AtomicU32::new(u32::MAX),
            })
        } else {
            None
        };
        tracing::info!(
            bytes,
            single_pass = pack_fp8_single.is_some(),
            "allocated sparse AITER MLA workspace"
        );
        Ok(Self {
            pack,
            pack_fp8,
            pack_fp8_single,
            attention,
            attention16,
            reduce,
            _scratch: scratch,
            rows_cap: rows,
            q,
            kv,
            part,
            lse,
            qp,
            kp,
            last,
            splits,
            lo,
        })
    }

    /// DIAGNOSTIC (`PLOW_GLM_ROWBAND_CLEAR_WS`). Zero every byte of the sparse workspace and the
    /// lower-half CSR, and forget the CSR's staged key, so a row-band dispatch sees exactly the
    /// device state a freshly loaded server would give it.
    ///
    /// It exists to decide ONE question. The row-band wrong-answer defect is triggered by any
    /// ordinary (program 3) execution in the 8192 bucket: run one first on a fresh server and
    /// every later row-band chunk is wrong, permanently; run the row-band program first and
    /// everything after is right (`rbfault13`/`14`/`15`). This workspace is the only device
    /// memory the two paths share that no per-sequence state clear covers -- it is allocated in
    /// [`SparseMla::load`], not in the packet's tensor table. If clearing it here makes the
    /// poisoned sequence correct, the carrier is in this allocation and the next step is to
    /// bisect its regions; if it does not, the whole allocation is exonerated in one run and the
    /// carrier is elsewhere (interpreter activations, KV, or the peer slots).
    ///
    /// Deliberately slow and deliberately off by default: a full zero of a workspace sized for
    /// the packet's max rows is hundreds of MB of H2D per dispatch.
    fn clear_workspace(&self, be: &HsaBackend) -> Result<()> {
        const CHUNK: usize = 4 << 20;
        let zeros = vec![0u8; CHUNK];
        // Launches still queued would otherwise race the clear.
        be.synchronize()?;
        for mem in std::iter::once(&self._scratch).chain(self.lo.as_ref().map(|l| &l.mem)) {
            let mut off = 0u64;
            while off < mem.len {
                let n = CHUNK.min((mem.len - off) as usize);
                EngineDevice::upload(be, mem, off, &zeros[..n])?;
                off += n as u64;
            }
        }
        if let Some(lo) = self.lo.as_ref() {
            lo.forget();
        }
        Ok(())
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8], rank: u32) -> Result<usize> {
        let one = SpanWindow {
            row0: 0,
            rows: route.rows,
            kv_len: route.kv_len,
            kv_base: 0,
        };
        if route.is_rowsplit() {
            return self.enqueue_window_rowsplit(be, route, tensor_table, one, rank);
        }
        self.enqueue_window(be, route, tensor_table, one, None)
    }

    /// Native half of a row-split chunk: rows `[split_row0, rows)`, each with all 2048 causal keys.
    pub fn enqueue_split(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<usize> {
        let window = SpanWindow {
            row0: route.split_row0,
            rows: route.rows - route.split_row0,
            kv_len: route.kv_len,
            kv_base: 0,
        };
        self.enqueue_window(be, route, tensor_table, window, None)
    }

    /// Rows `[0, split_row0)` of a native-lo chunk, every causal key through the ragged identity
    /// CSR. Single split only: the pinned kernel mis-merges ragged rows at ns=2.
    pub fn enqueue_split_lo(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<usize> {
        let lo = self.lo.as_ref().ok_or_else(|| {
            RuntimeError::Device("native lower half CSR was not allocated".into())
        })?;
        let prior = route.kv_len - route.rows;
        let rows = route.split_row0;
        if !route.native_lo || prior + rows != SPAN_MIN_PRIOR {
            return Err(RuntimeError::Device(
                "native lower half must end at the first 2048-key row".into(),
            ));
        }
        lo.stage(be, prior, rows)?;
        let window = SpanWindow {
            row0: 0,
            rows,
            kv_len: SPAN_MIN_PRIOR,
            kv_base: 0,
        };
        let csr = (lo.mem.base, lo.mem.base + LO_IDX_OFF);
        self.enqueue_window(be, route, tensor_table, window, Some(csr))
    }

    /// Launches of [`Self::enqueue_split_lo`] plus [`Self::enqueue_split`].
    pub fn native_lo_launches(&self, route: Route) -> usize {
        let single = route.rows - route.split_row0 >= 512
            && route.scale.is_some()
            && self.pack_fp8_single.is_some();
        2 + if single { 2 } else { 3 }
    }

    /// The AQL packets [`Self::enqueue_spans`] emits for `spans` (each span is its own
    /// pack → attention [→ reduce] chain).
    pub fn span_launches(&self, route: Route, spans: &[PrefillSpan]) -> usize {
        spans
            .iter()
            .map(|s| {
                let single =
                    s.n_rows >= 512 && route.scale.is_some() && self.pack_fp8_single.is_some();
                if single { 2 } else { 3 }
            })
            .sum()
    }

    /// One packed sibling / token-batch body launch: every request span runs the isolated
    /// chain on its own row window and its own request's cache (`kv_base = slot * ctx` cache
    /// rows, converted to `[0, span.kv_len)`), so a span never reads another slot's keys and
    /// rows no span covers (the band, parked padding) are never attended. Returns the packets
    /// emitted. Refuses a span whose rows would lack the full 2048 causal keys the fixed-width
    /// CSR assumes — the scheduler admits spans by the same rule (`SPAN_MIN_PRIOR`).
    pub fn enqueue_spans(
        &self,
        be: &HsaBackend,
        route: Route,
        tensor_table: &[u8],
        spans: &[PrefillSpan],
    ) -> Result<usize> {
        let ctx = route.inst.i[2];
        let mut launches = 0;
        for s in spans {
            if s.n_rows == 0
                || s.kv_row0 < SPAN_MIN_PRIOR
                || s.kv_len != s.kv_row0 + s.n_rows
                || s.kv_len > ctx
                || s.row0 + s.n_rows > route.inst.i[4]
            {
                return Err(RuntimeError::Device(format!(
                    "sparse MLA span slot {} rows [{}, +{}) kv [{}, {}) is not admissible on a \
                     2048-key CSR (needs kv_row0 >= {SPAN_MIN_PRIOR}, kv_len <= ctx {ctx})",
                    s.slot, s.row0, s.n_rows, s.kv_row0, s.kv_len
                )));
            }
            let window = SpanWindow {
                row0: s.row0,
                rows: s.n_rows,
                kv_len: s.kv_len,
                kv_base: u64::from(s.slot) * u64::from(ctx),
            };
            launches += self.enqueue_window(be, route, tensor_table, window, None)?;
        }
        Ok(launches)
    }

    fn enqueue_window(
        &self,
        be: &HsaBackend,
        route: Route,
        tensor_table: &[u8],
        w: SpanWindow,
        csr: Option<(u64, u64)>,
    ) -> Result<usize> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route.inst.t;
        let fp8 = route.scale.is_some();
        let row0 = u64::from(w.row0);
        let mut timer = SplitTimer::start(be)?;
        let single = (w.rows >= 512 || csr.is_some()) && fp8 && self.pack_fp8_single.is_some();
        if csr.is_some() && !single {
            return Err(RuntimeError::Device(
                "sparse MLA ragged CSR requires the single-pass FP8 pack (ns=1)".into(),
            ));
        }
        let splits = if single { 1 } else { 2 };
        let pack = PackArgs {
            q: self.q,
            kv: self.kv,
            qa: addr(t[2]) + row0 * 8 * 512 * 2,
            qr: addr(t[3]) + row0 * 8 * 64 * 2,
            ck: addr(t[4]) + w.kv_base * 512 * if fp8 { 1 } else { 2 },
            kr: addr(t[5]) + w.kv_base * 64 * 2,
            qp: self.qp,
            kp: self.kp,
            last: self.last,
            splits: self.splits,
            rows: w.rows,
            // VMM may leave the capacity beyond the live prefix unmapped.
            ctx: w.kv_len,
        };
        if let Some(scale) = route.scale {
            let kernel = self
                .pack_fp8
                .ok_or_else(|| RuntimeError::Device("FP8 sparse MLA pack was not loaded".into()))?;
            let args = PackFp8Args {
                base: pack,
                scale: addr(scale) + w.kv_base * 4,
            };
            if single {
                let args = PackSingleArgs {
                    base: args,
                    ml: addr(t[1]) + row0 * 8 * 2 * 4,
                };
                be.launch(
                    self.pack_fp8_single.unwrap(),
                    304,
                    256,
                    0,
                    bytemuck::bytes_of(&args),
                )?;
            } else {
                be.launch(kernel, 304, 256, 0, bytemuck::bytes_of(&args))?;
            }
        } else {
            be.launch(self.pack, 304, 256, 0, bytemuck::bytes_of(&pack))?;
        }
        timer.lap(be, "pack")?;
        let mut args = [0u64; 40];
        args[0] = if single { addr(t[0]) + row0 * 8 * 512 * 4 } else { self.part };
        args[2] = self.lse;
        args[4] = self.q;
        args[6] = self.kv;
        args[8] = csr.map_or(self.kp, |c| c.0);
        args[10] = csr.map_or(addr(route.index) + row0 * 2048 * 4, |c| c.1);
        args[12] = self.last;
        args[14] = u64::from(route.inst.fj[0]);
        args[16] = 8;
        args[18] = splits;
        args[20] = 8 * 576 * 2;
        args[22] = 576 * 2;
        args[26] = self.qp;
        args[28] = if single { self.qp } else { self.splits };
        // The pinned kernel writes normalized FP32 at one split when out_16_nosplit stays zero.
        be.launch_3d(
            self.attention,
            [1, w.rows, splits as u32],
            256,
            bytemuck::cast_slice(&args),
        )?;
        timer.lap(be, "attention")?;
        if single {
            timer.report(if csr.is_some() { "sparse_mla_lo" } else { "sparse_mla_single" }, w.rows);
            return Ok(2);
        }
        let reduce = ReduceArgs {
            out: addr(t[0]) + row0 * 8 * 512 * 4,
            ml: addr(t[1]) + row0 * 8 * 2 * 4,
            part: self.part,
            lse: self.lse,
            rows: w.rows,
            pad: 0,
        };
        be.launch(self.reduce, 304, 256, 0, bytemuck::bytes_of(&reduce))?;
        timer.lap(be, "reduce")?;
        timer.report("sparse_mla", w.rows);
        Ok(3)
    }

    /// Row-split (`nh=64`): 16 heads/group x 4 groups over this rank's own `T/8` row band. The
    /// FP8 single-pass pack runs unmodified at `rows*(nh/8)`: a rank's band (`T/8` rows x 64
    /// heads) covers the same `T*8` (row, head) pairs as the dense band, so its loop bounds and the
    /// scratch sized at `prog.t` rows fit. Each 16-head launch then writes normalized output at
    /// ONE split straight into its group's block of the group-major `opart`. Two splits cannot be
    /// used: `plow_mla_sparse_reduce` pairs lse entries 8 apart, which at 16 heads per launch is
    /// two heads of one split, not two splits of one head (rowsplit_probe: rel-L2 0.883 at ns=2 vs
    /// 1.12e-3 at ns=1).
    fn enqueue_window_rowsplit(
        &self,
        be: &HsaBackend,
        route: Route,
        tensor_table: &[u8],
        w: SpanWindow,
        rank: u32,
    ) -> Result<usize> {
        let (Some(scale), Some(pack)) = (route.scale, self.pack_fp8_single) else {
            return Err(RuntimeError::Device(
                "row-split sparse MLA requires the FP8 latent cache and the single-pass pack".into(),
            ));
        };
        let attention16 = self.attention16.ok_or_else(|| {
            RuntimeError::Device("row-split sparse AITER MLA object was not loaded".into())
        })?;
        if crate::config::RuntimeConfig::get().amd.glm_rowband_clear_ws {
            self.clear_workspace(be)?;
        }
        if !rowsplit_scratch_fits(self.rows_cap, u64::from(w.rows)) {
            return Err(RuntimeError::Device(format!(
                "row-split sparse MLA: {} rows inflated 8x exceeds the workspace's {} row \
                 capacity",
                w.rows, self.rows_cap
            )));
        }
        const NH: u32 = 64;
        const HG: u32 = 16;
        const GROUPS: u32 = 4;
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route.inst.t;
        let mut timer = SplitTimer::start(be)?;
        let args = PackSingleArgs {
            base: PackFp8Args {
                base: PackArgs {
                    q: self.q,
                    kv: self.kv,
                    qa: addr(t[2]),
                    qr: addr(t[3]),
                    ck: addr(t[4]) + w.kv_base * 512,
                    kr: addr(t[5]) + w.kv_base * 64 * 2,
                    qp: self.qp,
                    kp: self.kp,
                    last: self.last,
                    splits: self.splits,
                    rows: w.rows * (NH / 8),
                    ctx: w.kv_len,
                },
                scale: addr(scale) + w.kv_base * 4,
            },
            ml: addr(t[1]),
        };
        be.launch(pack, 304, 256, 0, bytemuck::bytes_of(&args))?;
        timer.lap(be, "pack")?;
        let index_row0 = if route.local_index { 0 } else { u64::from(rank) * u64::from(route.rows) };
        let prior = w.kv_len - w.rows * (NH / 8);
        // Every quantity the row-band dispatch derives, per rank. `rbfault19` showed that on a
        // poisoned server ALL EIGHT bands are wrong, which rules out either arm of `band_keys`
        // on its own and points at what feeds them; none of these is printed anywhere else, so
        // a healthy and a poisoned run cannot currently be compared on the numbers they use.
        if tracing::enabled!(tracing::Level::DEBUG) {
            let band = band_keys(prior, rank, w.rows);
            tracing::debug!(
                rank,
                rows = w.rows,
                kv_len = w.kv_len,
                kv_base = w.kv_base,
                row0 = w.row0,
                prior,
                route_rows = route.rows,
                route_kv_len = route.kv_len,
                local_index = route.local_index,
                index_row0,
                band = match band {
                    Some(BandKeys::Identity { band_prior }) => format!("identity@{band_prior}"),
                    Some(BandKeys::Selection) => "selection".to_string(),
                    None => "STRADDLE".to_string(),
                },
                "row-band dispatch"
            );
        }
        let csr = match band_keys(prior, rank, w.rows) {
            Some(BandKeys::Selection) => None,
            Some(BandKeys::Identity { band_prior }) => {
                let lo = self.lo.as_ref().ok_or_else(|| {
                    RuntimeError::Device("row-band identity keys need the lower-half CSR buffer".into())
                })?;
                lo.stage(be, band_prior, w.rows)?;
                Some((lo.mem.base, lo.mem.base + LO_IDX_OFF))
            }
            None => {
                return Err(RuntimeError::Device(format!(
                    "row-split band of rank {rank} straddles the first 2048-key row (prior={prior})"
                )))
            }
        };
        for g in 0..GROUPS {
            let a = rowsplit_launch_args(g, HG, NH, w.rows, 512, 1);
            let mut args = [0u64; 40];
            args[0] = addr(t[0]) + a.out_off_elems * 4;
            args[2] = self.lse + a.lse_off_elems * 4;
            args[4] = self.q + a.q_off_vec16 * 16;
            args[6] = self.kv;
            args[8] = csr.map_or(self.kp, |c| c.0);
            args[10] = csr.map_or(addr(route.index) + index_row0 * 2048 * 4, |c| c.1);
            args[12] = self.last;
            args[14] = u64::from(route.inst.fj[0]);
            args[16] = u64::from(a.mqa);
            args[18] = 1;
            args[20] = a.q_row_stride_bytes;
            args[22] = a.q_head_stride_bytes;
            args[26] = self.qp;
            args[28] = self.qp;
            be.launch_3d(attention16, [1, w.rows, 1], 256, bytemuck::cast_slice(&args))?;
        }
        timer.lap(be, "attention16")?;
        timer.report("sparse_mla_rowsplit", w.rows);
        Ok(1 + GROUPS as usize)
    }
}

/// One group's kernarg addressing for `mla_dec_stage1_bf16_a16w16_subQ{16,128}_mqa{16,128}` (or
/// today's `qh8` object, `groups=1`), mirroring `rowsplit_probe.cpp`'s `launch_all` byte for
/// byte (`args[0]`, `args[4]`, `args[20]`, `args[22]`).
///
/// Q is read from a WIDE, head-interleaved row (`q_row_stride_bytes = total_h*576*2`, `mqa` is
/// just how many of `total_h` heads this launch consumes) — this is why the Q all-to-all's
/// gathered `[T/tp][64][d]` layout feeds it directly, no repacking. Attention OUTPUT has no
/// independent row stride in this ABI: each launch's `out`/`lse` land in their OWN dense
/// `[rows][mqa][v_head_dim]` block (`out_off` scales by `mqa`, not `total_h`), a layout
/// `plow_mla_sparse_reduce` consumes directly when given the SAME `rows*nh/8` inflation the
/// pack kernel uses (see [`SparseMla::enqueue_window_rowsplit`]).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct RowsplitLaunchArgs {
    pub out_off_elems: u64,
    pub lse_off_elems: u64,
    pub q_off_vec16: u64,
    pub q_row_stride_bytes: u64,
    pub q_head_stride_bytes: u64,
    pub mqa: u32,
}

pub(super) fn rowsplit_launch_args(
    group: u32,
    mqa: u32,
    total_h: u32,
    rows: u32,
    v_head_dim: u32,
    ns: u32,
) -> RowsplitLaunchArgs {
    debug_assert!(mqa > 0 && total_h % mqa == 0 && group < total_h / mqa);
    RowsplitLaunchArgs {
        out_off_elems: u64::from(group)
            * u64::from(rows)
            * u64::from(mqa)
            * u64::from(v_head_dim)
            * u64::from(ns),
        lse_off_elems: u64::from(group) * u64::from(rows) * u64::from(mqa) * u64::from(ns),
        q_off_vec16: u64::from(group) * u64::from(mqa) * 72,
        q_row_stride_bytes: u64::from(total_h) * 576 * 2,
        q_head_stride_bytes: 576 * 2,
        mqa,
    }
}

#[cfg(test)]
mod rowsplit_addressing_tests {
    use super::*;

    /// `qh8` today, one group: must reproduce `enqueue_window`'s literal args
    /// (`args[16]=8`, `args[20]=8*576*2=9216`, `args[22]=576*2=1152`, no group offset) exactly.
    #[test]
    fn degenerate_eight_head_group_matches_todays_single_launch() {
        let a = rowsplit_launch_args(0, 8, 8, 1024, 512, 1);
        assert_eq!(a.out_off_elems, 0);
        assert_eq!(a.lse_off_elems, 0);
        assert_eq!(a.q_off_vec16, 0);
        assert_eq!(a.q_row_stride_bytes, 9216);
        assert_eq!(a.q_head_stride_bytes, 1152);
        assert_eq!(a.mqa, 8);
    }

    /// `rowsplit_probe.cpp`'s four-launch case at its own constants
    /// (`ROWS=1024, TOTAL_H=64, MQA=16`): `out_g = d_out + g*rows*mqa*512`,
    /// `q_ptr = d_q + g*mqa*72`, `s_Q_Bs = total_h*576*2`, `s_Bs = 576*2`, every group ns=1.
    #[test]
    fn four_launches_of_sixteen_match_the_probe() {
        let (rows, total_h, mqa, vd) = (1024u32, 64u32, 16u32, 512u32);
        for g in 0..4u32 {
            let a = rowsplit_launch_args(g, mqa, total_h, rows, vd, 1);
            assert_eq!(a.out_off_elems, u64::from(g) * u64::from(rows * mqa * vd), "group {g}");
            assert_eq!(a.lse_off_elems, u64::from(g) * u64::from(rows * mqa), "group {g}");
            assert_eq!(a.q_off_vec16, u64::from(g) * u64::from(mqa) * 72, "group {g}");
            assert_eq!(a.q_row_stride_bytes, u64::from(total_h) * 576 * 2);
            assert_eq!(a.q_head_stride_bytes, 576 * 2);
            assert_eq!(a.mqa, 16);
        }
        // Groups are non-overlapping and exactly tile [0, rows*64*vd).
        let last = rowsplit_launch_args(3, mqa, total_h, rows, vd, 1);
        assert_eq!(
            last.out_off_elems + u64::from(rows * mqa * vd),
            u64::from(rows) * u64::from(total_h) * u64::from(vd)
        );
    }

    /// The guard is a `debug_assert!`, so it exists only where debug assertions are compiled in --
    /// `rowsplit_launch_args` is on the per-launch path and an unconditional check does not belong
    /// there. Without this gate the test fails under `--release`, which is how plowrt ships.
    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn group_out_of_range_is_a_bug_not_silent_wraparound() {
        rowsplit_launch_args(4, 16, 64, 1024, 512, 1);
    }
}

/// Which keys rank `rank`'s row band reads in a full row-split chunk at `prior`: every causal key
/// through the identity CSR while the band's last row holds at most 2048 keys, the TP selection
/// once its first row holds all 2048, and neither for a band straddling row 2047.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BandKeys {
    Identity { band_prior: u32 },
    Selection,
}

pub(super) fn band_keys(prior: u32, rank: u32, band_rows: u32) -> Option<BandKeys> {
    let band_prior = prior.checked_add(rank.checked_mul(band_rows)?)?;
    if band_prior >= SPAN_MIN_PRIOR {
        Some(BandKeys::Selection)
    } else if band_prior + band_rows <= SPAN_MIN_PRIOR + 1 {
        Some(BandKeys::Identity { band_prior })
    } else {
        None
    }
}

/// A full TP8 row-split chunk at `prior` has a key plan for every rank's band.
pub(crate) fn rowsplit_prior_ok(prior: u32, band_rows: u32) -> bool {
    (0..8).all(|rank| band_keys(prior, rank, band_rows).is_some())
}

#[cfg(test)]
mod band_key_tests {
    use super::*;

    #[test]
    fn prior_zero_bands_split_at_row_2047() {
        let plan: Vec<_> = (0..8).map(|r| band_keys(0, r, 1024).unwrap()).collect();
        assert_eq!(plan[0], BandKeys::Identity { band_prior: 0 });
        assert_eq!(plan[1], BandKeys::Identity { band_prior: 1024 });
        assert!(plan[2..].iter().all(|k| *k == BandKeys::Selection));
        assert!(rowsplit_prior_ok(0, 1024));
    }

    #[test]
    fn bands_straddling_row_2047_have_no_plan() {
        for prior in [1u32, 500, 1022, 1025, 2046] {
            assert!(!rowsplit_prior_ok(prior, 1024), "prior={prior}");
        }
        for prior in [0u32, 1023, 1024, 2047, 8192, 65536] {
            assert!(rowsplit_prior_ok(prior, 1024), "prior={prior}");
        }
        assert_eq!(band_keys(1024, 0, 1024), Some(BandKeys::Identity { band_prior: 1024 }));
        assert_eq!(band_keys(1023, 1, 1024), Some(BandKeys::Selection));
    }

    #[test]
    fn identity_band_csr_is_causal_and_fits_the_lower_half_buffer() {
        for (prior, rank) in [(0u32, 0u32), (0, 1), (1023, 0), (1024, 0)] {
            let Some(BandKeys::Identity { band_prior }) = band_keys(prior, rank, 1024) else {
                panic!("prior={prior} rank={rank} is an identity band");
            };
            let (kp, idx) = lo_csr(band_prior, 1024);
            for t in 0..1024usize {
                let row = &idx[kp[t] as usize..kp[t + 1] as usize];
                assert!(row.iter().copied().eq(0..=band_prior + t as u32));
                assert!(row.len() <= 2048);
            }
            assert!(idx.len() as u64 <= LO_KEYS && (kp.len() as u64) * 4 <= LO_IDX_OFF);
        }
    }
}

/// One request's rows inside a launch: `[row0, row0 + rows)` of the query/output tensors, its
/// cache converted over `[0, kv_len)` starting `kv_base` cache rows into the bound tensors.
#[derive(Clone, Copy)]
struct SpanWindow {
    row0: u32,
    rows: u32,
    kv_len: u32,
    kv_base: u64,
}

/// Decode twin of [`Route`]: one isolated sparse FP8 `FlashMlaDecodeFp8` per segment, every
/// row with its own 2048-key selection, `nsplit` partials per (row, head) for the merge.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecodeRoute {
    inst: DevInst64,
    index: u16,
    rows: u32,
    pub active: bool,
}

impl DecodeRoute {
    /// The fixed-width CSR needs every row to hold all 2048 keys; shorter rows run the
    /// interpreter arm for the whole step.
    pub fn arm(&mut self, kvlen: &[u32]) {
        let rows = self.rows as usize;
        self.active = kvlen.len() >= rows && kvlen[..rows].iter().all(|&k| k >= 2048);
    }
}

pub(super) fn sparse_decode_inst(inst: &DevInst64) -> bool {
    inst.op == DevOp::FlashMlaDecodeFp8 as u16 && inst.fj[1] != 0
}

pub(super) fn decode_routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<DecodeRoute>>> {
    let mut routes = vec![None; segments];
    for (ix, inst) in prog.insts.iter().enumerate() {
        if !sparse_decode_inst(inst) {
            continue;
        }
        // Only a pure, counter-free segment is a native boundary; the ordinary emit keeps the
        // op inside an interpreter segment and is not this route's business.
        let mut owner = BTreeSet::new();
        let mut obligations = false;
        for entry in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            obligations |= entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0;
            owner.insert(entry.seg as usize);
        }
        let Some(&seg) = owner.first().filter(|_| owner.len() == 1 && !obligations) else {
            continue;
        };
        if seg >= segments
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            continue;
        }
        let err = |s: &str| {
            RuntimeError::Device(format!("sparse AITER MLA decode instruction {ix}: {s}"))
        };
        let ns = inst.i[4];
        if inst.i[0] != prog.t
            || prog.t > 20
            || inst.i[1] != 8
            || !(2048..=81920).contains(&inst.i[2])
            || inst.i[3] != 0
            || !(1..=16).contains(&ns)
            || inst.i[5] != u32::MAX
            || inst.i[6] != 2048
            || inst.i[7] != 4
            || inst.fj[2] != 0
            || inst.fj[1] > u32::from(TENSOR_NONE16)
            || !f32::from_bits(inst.fj[0]).is_finite()
            || f32::from_bits(inst.fj[0]) <= 0.0
        {
            return Err(err(
                "requires QH8 latent512/rope64 GF4 top-2048 rows<=20, nsplit<=16, ctx 2048..81920",
            ));
        }
        let index = (inst.fj[1] - 1) as u16;
        let rows = u64::from(prog.t);
        let ctx = u64::from(inst.i[2]);
        let ns = u64::from(ns);
        for (handle, bytes) in [
            (inst.t[0], rows * 8 * ns * 512 * 4),
            (inst.t[1], rows * 8 * ns * 2 * 4),
            (inst.t[2], rows * 8 * 512 * 2),
            (inst.t[3], rows * 8 * 64 * 2),
            (inst.t[4], rows * ctx * 512),
            (inst.t[5], rows * ctx * 64 * 2),
            (inst.t[6], rows * 4),
            (inst.t[7], rows * ctx * 4),
            (index, rows * 2048 * 4),
        ] {
            if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        if inst.t[1] == inst.t[0] {
            return Err(err("partial outputs alias"));
        }
        routes[seg] = Some(DecodeRoute {
            inst: *inst,
            index,
            rows: prog.t,
            active: false,
        });
    }
    Ok(routes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecodePackArgs {
    q: u64,
    kv: u64,
    kvi: u64,
    qa: u64,
    qr: u64,
    ck: u64,
    kr: u64,
    idx: u64,
    kvlen: u64,
    scale: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
    rows: u32,
    ctx: u32,
    nsplit: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RelayoutArgs {
    opart: u64,
    ml: u64,
    part: u64,
    lse: u64,
    rows: u32,
    nsplit: u32,
}

const _: () = assert!(std::mem::size_of::<DecodePackArgs>() == 128);
const _: () = assert!(std::mem::size_of::<RelayoutArgs>() == 40);

pub(super) struct SparseMlaDecode {
    pack: HsaKernel,
    attention: HsaKernel,
    relayout: HsaKernel,
    _scratch: DeviceMem,
    q: u64,
    kv: u64,
    kvi: u64,
    part: u64,
    lse: u64,
    qp: u64,
    kp: u64,
    last: u64,
    splits: u64,
}

impl SparseMlaDecode {
    pub fn load(be: &HsaBackend, dir: &Path, rows: u32, modules: &mut Vec<Module>) -> Result<Self> {
        if !(1..=20).contains(&rows) {
            return Err(RuntimeError::Device(
                "sparse AITER MLA decode requires 1..20 rows".into(),
            ));
        }
        let attention = load_attention(be, dir, modules)?;
        let path = dir.join("mla_sparse_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        let syms = super::amd::elf_symbol_names(&image);
        if !syms.contains(&"plow_mla_sparse_adapter_abi_1")
            || !syms.contains(&"plow_mla_sparse_decode_abi_1")
        {
            return Err(RuntimeError::Device(
                "sparse MLA adapter lacks decode ABI marker".into(),
            ));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let pack = EngineDevice::get_function(be, &module, "plow_mla_sparse_decode_pack")?;
        let relayout = EngineDevice::get_function(be, &module, "plow_mla_sparse_decode_relayout")?;
        for (kernel, size) in [(pack, 128), (relayout, 40)] {
            if ![size, size + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(
                    "sparse MLA decode adapter resource ABI mismatch".into(),
                ));
            }
        }
        modules.push(module);
        let rows = u64::from(rows);
        let sizes = [
            rows * 8 * 576 * 2,
            rows * 2048 * 576 * 2,
            rows * 2048 * 4,
            rows * 16 * 8 * 512 * 4,
            rows * 16 * 8 * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
            (rows + 1) * 4,
        ];
        let bytes = sizes.iter().map(|s| s.div_ceil(256) * 256).sum();
        let scratch = EngineDevice::alloc(be, bytes)?;
        let mut ptr = scratch.base;
        let offsets = sizes.map(|size| {
            let p = ptr;
            ptr += size.div_ceil(256) * 256;
            p
        });
        let [q, kv, kvi, part, lse, qp, kp, last, splits] = offsets;
        tracing::info!(bytes, rows, "allocated sparse AITER MLA decode workspace");
        Ok(Self {
            pack,
            attention,
            relayout,
            _scratch: scratch,
            q,
            kv,
            kvi,
            part,
            lse,
            qp,
            kp,
            last,
            splits,
        })
    }

    pub fn enqueue(&self, be: &HsaBackend, route: DecodeRoute, tensor_table: &[u8]) -> Result<()> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route.inst.t;
        let ns = route.inst.i[4];
        let pack = DecodePackArgs {
            q: self.q,
            kv: self.kv,
            kvi: self.kvi,
            qa: addr(t[2]),
            qr: addr(t[3]),
            ck: addr(t[4]),
            kr: addr(t[5]),
            idx: addr(route.index),
            kvlen: addr(t[6]),
            scale: addr(t[7]),
            qp: self.qp,
            kp: self.kp,
            last: self.last,
            splits: self.splits,
            rows: route.rows,
            ctx: route.inst.i[2],
            nsplit: ns,
            pad: 0,
        };
        be.launch(self.pack, 304, 256, 0, bytemuck::bytes_of(&pack))?;
        let mut args = [0u64; 40];
        args[0] = self.part;
        args[2] = self.lse;
        args[4] = self.q;
        args[6] = self.kv;
        args[8] = self.kp;
        args[10] = self.kvi;
        args[12] = self.last;
        args[14] = u64::from(route.inst.fj[0]);
        args[16] = 8;
        args[18] = u64::from(ns);
        args[20] = 8 * 576 * 2;
        args[22] = 576 * 2;
        args[26] = self.qp;
        args[28] = self.splits;
        be.launch_3d(
            self.attention,
            [1, route.rows, ns],
            256,
            bytemuck::cast_slice(&args),
        )?;
        let relayout = RelayoutArgs {
            opart: addr(t[0]),
            ml: addr(t[1]),
            part: self.part,
            lse: self.lse,
            rows: route.rows,
            nsplit: ns,
        };
        be.launch(self.relayout, 304, 256, 0, bytemuck::bytes_of(&relayout))
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn fixture(rows: u32) -> (DevProg, Vec<DevTensor>) {
        let flash = DevInst64 {
            op: DevOp::FlashMlaDecodeFp8 as u16,
            t: [0, 1, 2, 3, 4, 5, 6, 7],
            i: [rows, 8, 4096, 0, 16, u32::MAX, 2048, 4],
            fj: [0.0625f32.to_bits(), 9, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: rows,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows: rows },
            n_counter: 0,
            insts: vec![
                DevInst64 {
                    op: DevOp::Nop as u16,
                    ..Default::default()
                },
                flash,
            ],
            stream: vec![
                StreamEnt {
                    inst: 0,
                    seg: 0,
                    ..Default::default()
                },
                StreamEnt {
                    inst: 1,
                    seg: 1,
                    ..Default::default()
                },
            ],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let tensors = (0..9)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 64 << 20,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    #[test]
    fn sparse_decode_routes_require_pure_segments_geometry_and_capacity() {
        let (prog, tensors) = fixture(20);
        let routes = decode_routes(&prog, &tensors, 2).unwrap();
        assert!(routes[0].is_none());
        let mut route = routes[1].unwrap();
        assert_eq!(route.index, 8);
        assert!(!route.active);
        route.arm(&[2048; 20]);
        assert!(route.active);
        route.arm(&[2048; 19]);
        assert!(!route.active);
        let mut short = [4096u32; 20];
        short[7] = 2047;
        route.arm(&short);
        assert!(!route.active);
        // Mixed or counter-bearing segments are the ordinary emit: no route, no error.
        let (mut mixed, tensors) = fixture(20);
        mixed.stream[1].seg = 0;
        assert!(decode_routes(&mixed, &tensors, 1).unwrap()[0].is_none());
        let (mut waits, tensors) = fixture(20);
        waits.stream[1].wait_len = 1;
        assert!(decode_routes(&waits, &tensors, 2).unwrap()[1].is_none());
        for bad in 0..9 {
            let (mut prog, mut tensors) = fixture(20);
            match bad {
                0 => prog.insts[1].i[0] = 8,
                1 => prog.insts[1].i[1] = 16,
                2 => prog.insts[1].i[4] = 17,
                3 => prog.insts[1].i[6] = 1024,
                4 => prog.insts[1].i[7] = 2,
                5 => prog.insts[1].fj[2] = 1,
                6 => tensors[8].bytes = 20 * 2048 * 4 - 1,
                7 => tensors[7].bytes = 20 * 4096 * 4 - 1,
                _ => prog.insts[1].t[7] = TENSOR_NONE16,
            }
            assert!(decode_routes(&prog, &tensors, 2).is_err(), "case {bad}");
        }
        let (mut wide, tensors) = fixture(21);
        wide.insts[1].i[0] = 21;
        assert!(decode_routes(&wide, &tensors, 2).is_err());
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the decode adapter"]
    fn sparse_mla_decode_hsa_dispatch() {
        const CTX: usize = 4096;
        const NS: usize = 16;
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let kernel = SparseMlaDecode::load(&be, Path::new(&dir), 20, &mut modules).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let bf16 = |v: f32| {
            let b = v.to_bits();
            ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
        };
        let unbf16 = |b: u16| f64::from(f32::from_bits(u32::from(b) << 16));
        let decode = |byte: u8| -> f64 {
            let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
            let exp = i32::from((byte >> 3) & 0xf);
            let man = f64::from(byte & 7) / 8.0;
            if exp == 0 {
                sign * man * 2f64.powi(-6)
            } else {
                sign * (1.0 + man) * 2f64.powi(exp - 7)
            }
        };
        for rows in [1usize, 8, 20] {
            // Random FP8 latent (magnitudes below 2, no NaN pattern), per-row scales, BF16 rope,
            // BF16 queries, and a distinct random 2048-key selection per row.
            let ck: Vec<u8> = (0..rows * CTX * 512)
                .map(|_| {
                    let r = next();
                    (((r >> 8) & 1) << 7) as u8 | (((r >> 4) % 8) << 3) as u8 | (r % 8) as u8
                })
                .collect();
            let scales: Vec<f32> = (0..rows * CTX)
                .map(|_| 0.5 + (next() % 1000) as f32 / 1000.0)
                .collect();
            let kr: Vec<u16> = (0..rows * CTX * 64)
                .map(|_| bf16((next() % 2001) as f32 / 1000.0 - 1.0))
                .collect();
            let qa: Vec<u16> = (0..rows * 8 * 512)
                .map(|_| bf16((next() % 2001) as f32 / 2000.0 - 0.5))
                .collect();
            let qr: Vec<u16> = (0..rows * 8 * 64)
                .map(|_| bf16((next() % 2001) as f32 / 2000.0 - 0.5))
                .collect();
            let kvlen: Vec<i32> = (0..rows).map(|b| (2048 + b * 97).min(CTX) as i32).collect();
            let mut idx = vec![0i32; rows * 2048];
            for b in 0..rows {
                let len = kvlen[b] as usize;
                let mut keys: Vec<i32> = (0..len as i32).collect();
                for i in (1..len).rev() {
                    keys.swap(i, (next() % (i as u64 + 1)) as usize);
                }
                idx[b * 2048..(b + 1) * 2048].copy_from_slice(&keys[..2048]);
            }
            let upload = |bytes: &[u8]| {
                let m = EngineDevice::alloc(&be, bytes.len() as u64 + 1024).unwrap();
                EngineDevice::upload(&be, &m, 0, bytes).unwrap();
                m
            };
            let opart_bytes = rows * 8 * NS * 512 * 4;
            let ml_bytes = rows * 8 * NS * 2 * 4;
            let opart = upload(&vec![0xa5u8; opart_bytes + 512]);
            let mlpart = upload(&vec![0xa5u8; ml_bytes + 512]);
            let d_qa = upload(bytemuck::cast_slice(&qa));
            let d_qr = upload(bytemuck::cast_slice(&qr));
            let d_ck = upload(&ck);
            let d_kr = upload(bytemuck::cast_slice(&kr));
            let d_kvlen = upload(bytemuck::cast_slice(&kvlen));
            let d_scale = upload(bytemuck::cast_slice(&scales));
            let d_idx = upload(bytemuck::cast_slice(&idx));
            let table = [
                opart.base + 256,
                mlpart.base + 256,
                d_qa.base,
                d_qr.base,
                d_ck.base,
                d_kr.base,
                d_kvlen.base,
                d_scale.base,
                d_idx.base,
            ];
            let (prog, tensors) = fixture(rows as u32);
            let mut route = decode_routes(&prog, &tensors, 2).unwrap()[1].unwrap();
            route.arm(&kvlen.iter().map(|&k| k as u32).collect::<Vec<_>>());
            assert!(route.active);
            kernel
                .enqueue(&be, route, bytemuck::cast_slice(&table))
                .unwrap();
            be.synchronize().unwrap();
            let mut ob = vec![0u8; opart_bytes + 512];
            let mut mb = vec![0u8; ml_bytes + 512];
            EngineDevice::download(&be, &opart, 0, &mut ob).unwrap();
            EngineDevice::download(&be, &mlpart, 0, &mut mb).unwrap();
            for (bytes, len) in [(&ob, opart_bytes), (&mb, ml_bytes)] {
                assert!(bytes[..256]
                    .iter()
                    .chain(&bytes[256 + len..])
                    .all(|&x| x == 0xa5));
            }
            let op: &[f32] = bytemuck::cast_slice(&ob[256..256 + opart_bytes]);
            let ml: &[f32] = bytemuck::cast_slice(&mb[256..256 + ml_bytes]);
            let (mut num, mut den) = (0f64, 0f64);
            for b in 0..rows {
                for h in 0..8 {
                    // Host copy of d_mla_merge_fold's combine over the NS partials.
                    let base = (b * 8 + h) * NS;
                    let gm = (0..NS)
                        .map(|s| ml[(base + s) * 2])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut merged = vec![0f64; 512];
                    let mut gl = 0f64;
                    for s in 0..NS {
                        let (m, l) = (ml[(base + s) * 2], ml[(base + s) * 2 + 1]);
                        assert!(
                            m.is_finite() && l == 1.0,
                            "rows={rows} b={b} h={h} s={s}: ml=({m},{l})"
                        );
                        let w = f64::from(l) * f64::from((m - gm).exp());
                        gl += w;
                        for d in 0..512 {
                            merged[d] += f64::from(op[(base + s) * 512 + d]) * w;
                        }
                    }
                    for v in &mut merged {
                        *v /= gl;
                    }
                    // FP64 reference over the row's selected keys on the BF16-packed values.
                    let q: Vec<f64> = (0..576)
                        .map(|d| {
                            if d < 512 {
                                unbf16(qa[(b * 8 + h) * 512 + d])
                            } else {
                                unbf16(qr[(b * 8 + h) * 64 + d - 512])
                            }
                        })
                        .collect();
                    let mut scores = Vec::with_capacity(2048);
                    let mut keys = Vec::with_capacity(2048);
                    for j in 0..2048 {
                        let key = idx[b * 2048 + j] as usize;
                        let src = b * CTX + key;
                        let scale = f64::from(scales[src]);
                        let mut k: Vec<f64> = (0..512)
                            .map(|d| {
                                let v = (decode(ck[src * 512 + d]) * scale) as f32;
                                unbf16(bf16(v))
                            })
                            .collect();
                        k.extend((0..64).map(|d| unbf16(kr[src * 64 + d])));
                        scores.push(0.0625 * q.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>());
                        keys.push(k);
                    }
                    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ws: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let total: f64 = ws.iter().sum();
                    for d in 0..512 {
                        let r: f64 =
                            ws.iter().zip(&keys).map(|(w, k)| w * k[d]).sum::<f64>() / total;
                        num += (merged[d] - r).powi(2);
                        den += r.powi(2);
                    }
                }
            }
            let rel = (num / den).sqrt();
            // Warm route latency (pack + attention + relayout, host-timed through a drain).
            for _ in 0..3 {
                kernel
                    .enqueue(&be, route, bytemuck::cast_slice(&table))
                    .unwrap();
            }
            be.synchronize().unwrap();
            let mut samples: Vec<f64> = (0..15)
                .map(|_| {
                    let start = std::time::Instant::now();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.synchronize().unwrap();
                    start.elapsed().as_secs_f64() * 1e6
                })
                .collect();
            samples.sort_by(|a, b| a.total_cmp(b));
            eprintln!(
                "rows={rows} rel_l2 vs fp64 = {rel:.3e}; route warm median {:.1}us",
                samples[samples.len() / 2]
            );
            assert!(rel < 5e-3, "rows={rows}: rel L2 {rel}");
        }
    }
}
