//! Heterogeneous prefill lanes (plans/apple-heterogeneous-emit.md, `plow_asset::hetero`).
//!
//! The compiler split every prefill bucket's rows into GPU / ANE / CPU blocks and emitted the
//! GPU packet over its own rows with a host segment at each join. This module runs the other
//! two lanes of a segment concurrently with the GPU's command buffer:
//!
//! * ANE lane: one CoreML program per layer (`pre` = norm+QKV, `mid` = o_proj..MLP of layer l
//!   plus norm+QKV of l+1, `post` = o_proj..MLP of the last layer), weights baked fp16 from the
//!   packet's quantized twins with the norm gammas folded into the consuming projections. The
//!   residual stream rides the ANE scaled by `RESID_SCALE` (o_proj/down weights carry the same
//!   factor) so the fp16 sum of squares inside the norm cannot overflow on Llama's massive
//!   activations; the norm is scale-invariant, so q/k/v come out at true scale.
//! * CPU lane: the segment's row-split GPU instructions re-based onto the CPU's row block
//!   (pointer offsets + M), run by a persistent thread pool on the NEON kernels.
//!
//! Every lane writes only its own rows of the shared (unified-memory) activation tensors, so
//! no lane needs a counter: the segment boundary is the join.

use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::Instant;

use packet::dev::{DevInst64, DevOp};
#[cfg(feature = "ane")]
use plow_asset::hetero::WeightEncoding;
use plow_asset::hetero::{AneLane, HeteroPlan, ProgPlan, SegPlan, FILE};

use crate::exec::cpu::engine::CpuModel;
use crate::exec::cpu::ffi;
use crate::exec::kvrow::{prefill_row_field, RowField};
use crate::{Result, RuntimeError};

/// Residual-stream scale on the ANE (see the module doc). 1/16 keeps `sum(x^2)` of a row with
/// entries up to ~4000 inside fp16.
pub const RESID_SCALE: f32 = 1.0 / 16.0;

/// Rows per ANE program call. A program's cost is dominated by streaming its layer's weights
/// (measured: 32-row programs are SLOWER than 64-row ones), so every call carries 64 rows; the
/// lane's block is walked in 64-row calls with the last one zero-padded in the staging buffer.
pub const ANE_ROWS: usize = 64;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct LaneProfile {
    pub input_convert_ms: f64,
    pub output_convert_ms: f64,
    pub residual_ms: f64,
    #[cfg(feature = "ane")]
    pub coreml: crate::exec::ane::RunTimings,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub segs: usize,
    pub ane_runs: usize,
    /// Whole host lane, including conversion and CoreML; not pure ANE execution time.
    pub ane_ms: f64,
    pub cpu_ops: usize,
    pub cpu_ms: f64,
    /// Host time spent waiting for the GPU after its own lanes finished (the GPU was the
    /// critical path) — summed over segments.
    pub gpu_wait_ms: f64,
    pub profile: LaneProfile,
}

#[derive(Clone, Copy)]
pub struct Handles {
    pub x: usize,
    pub at: usize,
    pub qg: usize,
    pub kg: usize,
    pub vg: usize,
}

pub struct Hetero {
    pub plan: HeteroPlan,
    by_prog: Vec<Option<usize>>,
    pub h: Handles,
    pub threads: usize,
    pool: Option<LanePool>,
    #[cfg(feature = "ane")]
    ane: Option<AneLanes>,
    pub stats: Stats,
    pub profile_enabled: bool,
    /// Rows of the current chunk that hold real tokens (`prepare_chunk`).
    pub clen: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_lanes_restore_all_gpu_rows() {
        let mut h = Hetero {
            plan: HeteroPlan {
                ane_pct: 50,
                programs: vec![ProgPlan {
                    t: 128,
                    rows_gpu: 64,
                    rows_ane: 64,
                    segments: vec![SegPlan {
                        cpu_insts: vec![0],
                        ..SegPlan::default()
                    }],
                    ..ProgPlan::default()
                }],
                ..HeteroPlan::default()
            },
            by_prog: vec![Some(0)],
            h: Handles {
                x: 0,
                at: 0,
                qg: 0,
                kg: 0,
                vg: 0,
            },
            threads: 0,
            pool: None,
            #[cfg(feature = "ane")]
            ane: None,
            stats: Stats::default(),
            profile_enabled: false,
            clen: 0,
        };
        let mut d = DevInst64 {
            op: 34,
            ..DevInst64::default()
        };
        d.i[0] = 64;
        let mut insts = [d];
        h.prepare_chunk(0, &mut insts, &[d], 8);
        assert!(!h.has_active_lanes(0));
        assert_eq!(insts[0].i[0], 8);
        h.prepare_chunk(0, &mut insts, &[d], 128);
        assert!(h.has_active_lanes(0));
        assert_eq!(insts[0].i[0], 64);
        h.plan.ane_pct = 0;
        h.prepare_chunk(0, &mut insts, &[d], 128);
        assert!(!h.has_active_lanes(0));
        assert_eq!(insts[0].i[0], 128);
        assert!(!h.has_active_lanes(1));
        h.plan.ane_pct = 50;
        for (rows, ane) in [
            (31, 8),
            (32, 16),
            (33, 16),
            (63, 24),
            (64, 32),
            (65, 32),
            (127, 56),
            (128, 64),
        ] {
            assert_eq!(h.rows_for_chunk(0, rows), Some((rows - ane, ane, 0)));
        }
        assert_eq!(h.rows_for_chunk(1, 64), None);
        h.profile_enabled = true;
        h.stats.profile.input_convert_ms = 3.0;
        h.reset_stats();
        assert!(h.profile_enabled);
        assert_eq!(h.stats.profile.input_convert_ms, 0.0);
    }
}

impl Hetero {
    /// Load `<blob dir>/hetero.json` if present.
    pub fn load(model: &CpuModel, blob: &Path) -> Result<Option<Hetero>> {
        let path = blob.with_file_name(FILE);
        let Ok(bytes) = std::fs::read(&path) else {
            return Ok(None);
        };
        let plan = match plow_asset::hetero::parse(&bytes)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?
        {
            plow_asset::hetero::Plan::Row(plan) => plan,
            plow_asset::hetero::Plan::Channel(plan) => {
                model
                    .blob
                    .with_packet_view(|p| plan.validate(p))
                    .map_err(RuntimeError::Device)?;
                if !crate::config::RuntimeConfig::get().apple.ane_mlp {
                    eprintln!("channel MLP: validated plan; offload disabled, using unsplit GPU");
                }
                return Ok(None);
            }
        };
        let handle =
            |name: &str| -> Result<usize> {
                model.names.iter().position(|n| n == name).ok_or_else(|| {
                    RuntimeError::Device(format!("hetero: tensor {name} not in blob"))
                })
            };
        let h = Handles {
            x: handle(&plan.tensors.x)?,
            at: handle(&plan.tensors.at)?,
            qg: handle(&plan.tensors.qg)?,
            kg: handle(&plan.tensors.kg)?,
            vg: handle(&plan.tensors.vg)?,
        };
        let mut by_prog = vec![None; model.blob.progs.len()];
        for (i, pp) in plan.programs.iter().enumerate() {
            let p = pp.prog as usize;
            if p >= by_prog.len() || model.blob.progs[p].t != pp.t {
                return Err(RuntimeError::Device(format!(
                    "hetero: program {p} (T={}) does not match the blob",
                    pp.t
                )));
            }
            for sp in &pp.segments {
                for &ix in &sp.cpu_insts {
                    let d = model.blob.progs[p].insts.get(ix as usize).ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "hetero: program {p} instruction {ix} missing"
                        ))
                    })?;
                    if cpu_rebase(d, &plan, 0, 1).is_none() {
                        return Err(RuntimeError::Device(format!(
                            "hetero: program {p} instruction {ix}: op {} has no CPU row re-base",
                            d.op
                        )));
                    }
                }
            }
            by_prog[p] = Some(i);
        }
        let threads = match crate::config::RuntimeConfig::get().cpu.threads {
            0 => 8,
            n => n as usize,
        };
        let any_cpu = plan.programs.iter().any(|p| p.rows_cpu > 0);
        let any_ane = plan.programs.iter().any(|p| p.rows_ane > 0);
        let pool = any_cpu.then(|| LanePool::new(threads));
        #[cfg(feature = "ane")]
        let ane = if any_ane {
            Some(AneLanes::new(model, &plan, blob)?)
        } else {
            None
        };
        #[cfg(not(feature = "ane"))]
        if any_ane {
            return Err(RuntimeError::Device(
                "hetero: the plan gives rows to the ANE; build with --features ane".into(),
            ));
        }
        tracing::info!(
            programs = plan.programs.len(),
            ane_pct = plan.ane_pct,
            cpu_pct = plan.cpu_pct,
            threads,
            "heterogeneous prefill plan loaded"
        );
        Ok(Some(Hetero {
            plan,
            by_prog,
            h,
            threads,
            pool,
            #[cfg(feature = "ane")]
            ane,
            stats: Stats::default(),
            profile_enabled: false,
            clen: 0,
        }))
    }

    pub fn prog_plan(&self, p: usize) -> Option<&ProgPlan> {
        self.by_prog
            .get(p)
            .copied()
            .flatten()
            .map(|i| &self.plan.programs[i])
    }

    pub fn has_active_lanes(&self, p: usize) -> bool {
        self.prog_plan(p).is_some_and(|pp| {
            let (_, a, c) = self.split(pp);
            a > 0 || c > 0
        })
    }

    /// Chunk staging for a planned program: the bucket-row shrink of `rebase_chunk_rows` must
    /// not touch the row-split ops (their `M` is the GPU block, not `T`); those shrink to
    /// `min(rows_gpu, clen)` here instead.
    pub fn prepare_chunk(
        &mut self,
        p: usize,
        insts: &mut [DevInst64],
        pristine: &[DevInst64],
        clen: u32,
    ) {
        self.clen = clen;
        let Some(pp) = self.prog_plan(p) else {
            return;
        };
        let eff = self.split(pp).0.max(1);
        for sp in &pp.segments {
            for &ix in &sp.cpu_insts {
                let ix = ix as usize;
                let mut d = pristine[ix];
                match prefill_row_field(d.op) {
                    Some(RowField::Rows(f)) if d.i[f] == pp.rows_gpu => d.i[f] = eff,
                    Some(RowField::RowsTimes(f)) if d.i[f] > 0 && d.i[f] % pp.rows_gpu == 0 => {
                        d.i[f] = d.i[f] / pp.rows_gpu * eff
                    }
                    _ => {}
                }
                insts[ix] = d;
            }
        }
    }

    /// Row policy for the current chunk: the ANE and CPU take their calibrated percentage of the
    /// REAL rows (multiples of 8), the GPU the rest — its `M` is a runtime operand, the ANE walks
    /// its block in [`ANE_ROWS`] calls, the CPU kernels take any row count. The compile-time
    /// blocks in the plan only fix the GPU tile choice and the segment structure.
    fn split(&self, pp: &ProgPlan) -> (u32, u32, u32) {
        self.split_rows(pp, self.clen)
    }

    pub fn rows_for_chunk(&self, p: usize, rows: u32) -> Option<(u32, u32, u32)> {
        self.prog_plan(p).map(|pp| self.split_rows(pp, rows))
    }

    fn split_rows(&self, pp: &ProgPlan, rows: u32) -> (u32, u32, u32) {
        let clen = rows.max(1).min(pp.t);
        let r8 = |pct: u32| (clen * pct / 100) / 8 * 8;
        let (mut a, mut c) = (
            if pp.rows_ane > 0 {
                r8(self.plan.ane_pct)
            } else {
                0
            },
            if pp.rows_cpu > 0 {
                r8(self.plan.cpu_pct)
            } else {
                0
            },
        );
        while a + c + 8 > clen {
            if a >= c && a > 0 {
                a -= 8;
            } else if c > 0 {
                c -= 8;
            } else {
                break;
            }
        }
        (clen - a - c, a, c)
    }

    /// Row blocks `(ane: (row0, rows), cpu: (row0, rows))` of program `p` for the current chunk.
    fn blocks(&self, pp: &ProgPlan) -> (Option<(u32, u32)>, Option<(u32, u32)>) {
        let (g, a, c) = self.split(pp);
        ((a > 0).then_some((g, a)), (c > 0).then_some((g + a, c)))
    }

    /// Start the CPU lane of `sp` (returns whether one was started) and run the ANE lane on the
    /// calling thread; `wait_cpu` joins the CPU lane.
    pub fn run_lanes(
        &mut self,
        p: usize,
        sp: &SegPlan,
        insts: &[DevInst64],
        table: &[*mut u8],
    ) -> Result<bool> {
        let Some(pp) = self.prog_plan(p) else {
            return Ok(false);
        };
        let (ane, cpu) = self.blocks(pp);
        let mut started = false;
        if let (Some((row0, rows)), Some(pool)) = (cpu, self.pool.as_ref()) {
            if !sp.cpu_insts.is_empty() {
                let mut ops = Vec::with_capacity(sp.cpu_insts.len());
                for &ix in &sp.cpu_insts {
                    let d = insts[ix as usize];
                    let (d, offs) =
                        cpu_rebase(&d, &self.plan, row0, rows).expect("validated at load");
                    let mut tab: Vec<usize> = table.iter().map(|&p| p as usize).collect();
                    // An in-place op names one tensor in two slots (Residual t0 == t1): offset each
                    // HANDLE once.
                    let mut done: Vec<usize> = Vec::with_capacity(offs.len());
                    for (slot, bytes) in offs {
                        let h = d.t[slot] as usize;
                        if h != packet::dev::TENSOR_NONE16 as usize && !done.contains(&h) {
                            tab[h] += bytes;
                            done.push(h);
                        }
                    }
                    ops.push((d, tab));
                }
                self.stats.cpu_ops += ops.len();
                pool.start(Job { ops });
                started = true;
            }
        }
        #[cfg(feature = "ane")]
        if let (Some(lane), Some((row0, rows))) = (&sp.ane, ane) {
            let t0 = Instant::now();
            let ane_l = self.ane.as_mut().expect("ANE lanes built at load");
            ane_l.profile = self.profile_enabled.then_some(self.stats.profile);
            ane_l.run(lane, row0 as usize, rows as usize, &self.h, table)?;
            if let Some(profile) = ane_l.profile {
                self.stats.profile = profile;
            }
            self.stats.ane_runs += 1;
            self.stats.ane_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        #[cfg(not(feature = "ane"))]
        let _ = ane;
        Ok(started)
    }

    pub fn wait_cpu(&mut self) {
        if let Some(pool) = &self.pool {
            let t0 = Instant::now();
            pool.wait();
            self.stats.cpu_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
    }

    pub fn reset_stats(&mut self) {
        self.stats = Stats::default();
    }
}

/// Re-base a row-split instruction onto rows `[row0, row0+rows)`: the returned instruction has
/// its row count replaced and the listed tensor slots need the byte offsets added.
fn cpu_rebase(
    d: &DevInst64,
    plan: &HeteroPlan,
    row0: u32,
    rows: u32,
) -> Option<(DevInst64, Vec<(usize, usize)>)> {
    let mut o = *d;
    let (row0, rows) = (row0 as usize, rows as usize);
    let op = DevOp::from_u16(d.op)?;
    let offs = match op {
        DevOp::RmsNorm => {
            let feat = d.i[1] as usize;
            o.i[0] = rows as u32;
            vec![(0, row0 * feat * 2), (1, row0 * feat * 2)]
        }
        DevOp::NormResidual => {
            let feat = d.i[1] as usize;
            o.i[0] = rows as u32;
            vec![
                (0, row0 * feat * 2),
                (1, row0 * feat * 2),
                (2, row0 * feat * 2),
            ]
        }
        DevOp::Residual => {
            let hidden = plan.hidden as usize;
            o.i[0] = (rows * hidden) as u32;
            vec![
                (0, row0 * hidden * 2),
                (1, row0 * hidden * 2),
                (2, row0 * hidden * 2),
            ]
        }
        DevOp::PerLayerInput => {
            let (h, stride) = (d.i[1] as usize, d.i[4] as usize);
            o.i[0] = rows as u32;
            vec![(0, row0 * h * 2), (4, row0 * stride * 2), (5, row0 * h * 2)]
        }
        DevOp::Glu => {
            let inter = plan.inter as usize;
            o.i[0] = (rows * inter) as u32;
            vec![
                (0, row0 * inter * 2),
                (1, row0 * inter * 2),
                (2, row0 * inter * 2),
            ]
        }
        DevOp::Gemm
        | DevOp::GemmSmall
        | DevOp::GemmMed
        | DevOp::GemmFp8
        | DevOp::GemmMedFp8
        | DevOp::GemmSmallFp8
        | DevOp::GemmMxfp4
        | DevOp::GemmMedMxfp4
        | DevOp::GemmSmallMxfp4
        | DevOp::GemmWideMxfp4
        | DevOp::GemmGluMxfp4
        | DevOp::GemmGlu
        | DevOp::GemmGluFp8 => {
            let (n, k) = (d.i[1] as usize, d.i[2] as usize);
            o.i[0] = rows as u32;
            vec![(0, row0 * n * 2), (1, row0 * k * 2)]
        }
        _ => return None,
    };
    Some((o, offs))
}

// ---- CPU lane pool -------------------------------------------------------------------------

struct Job {
    /// `(instruction, tensor table)` in dependency order; each op runs on every worker as slice
    /// `w` of `threads`.
    ops: Vec<(DevInst64, Vec<usize>)>,
}

struct State {
    gen: u64,
    job: Option<Arc<Job>>,
    pending: usize,
    quit: bool,
}

struct Shared {
    m: Mutex<State>,
    cv: Condvar,
    done: Condvar,
    barrier: Barrier,
    threads: usize,
}

/// Persistent worker threads for the CPU lane: one job per segment, a barrier between ops.
pub struct LanePool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl LanePool {
    fn new(threads: usize) -> LanePool {
        let shared = Arc::new(Shared {
            m: Mutex::new(State {
                gen: 0,
                job: None,
                pending: 0,
                quit: false,
            }),
            cv: Condvar::new(),
            done: Condvar::new(),
            barrier: Barrier::new(threads),
            threads,
        });
        let handles = (0..threads)
            .map(|w| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("plow-lane-{w}"))
                    .spawn(move || worker(sh, w))
                    .expect("spawn lane worker")
            })
            .collect();
        LanePool { shared, handles }
    }

    fn start(&self, job: Job) {
        let mut st = self.shared.m.lock().unwrap();
        debug_assert_eq!(st.pending, 0, "lane job already in flight");
        st.job = Some(Arc::new(job));
        st.pending = self.shared.threads;
        st.gen += 1;
        self.shared.cv.notify_all();
    }

    fn wait(&self) {
        let mut st = self.shared.m.lock().unwrap();
        while st.pending > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
    }
}

impl Drop for LanePool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.m.lock().unwrap();
            st.quit = true;
            self.shared.cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker(sh: Arc<Shared>, w: usize) {
    #[cfg(target_os = "macos")]
    // SAFETY: sets this thread's QoS class (keeps it on the performance cluster).
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let mut ctx = ffi::PlowCpuCtx::new(w as u32, 0);
    let scratch_bytes = ffi::scratch_bytes().max(64) as usize;
    let mut scratch = vec![0u64; scratch_bytes / 8 + 8];
    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch_bytes as u32;
    let _ = ffi::thread_init(&mut ctx);
    let mut seen = 0u64;
    loop {
        let job = {
            let mut st = sh.m.lock().unwrap();
            while st.gen == seen && !st.quit {
                st = sh.cv.wait(st).unwrap();
            }
            if st.quit {
                return;
            }
            seen = st.gen;
            st.job.clone().expect("job set with gen")
        };
        for (d, tab) in &job.ops {
            if let Some(f) = ffi::kernel(d.op) {
                // SAFETY: the kernel contract (validated handles; disjoint row slices per op,
                // ops ordered by dependency with a barrier between them).
                unsafe {
                    f(
                        d,
                        w as u32,
                        sh.threads as u32,
                        tab.as_ptr() as *const *mut c_void,
                        &mut ctx,
                    )
                };
            }
            sh.barrier.wait();
        }
        let mut st = sh.m.lock().unwrap();
        st.pending -= 1;
        if st.pending == 0 {
            st.job = None;
            sh.done.notify_all();
        }
    }
}

// ---- ANE lane -------------------------------------------------------------------------------

#[cfg(feature = "ane")]
use std::collections::HashMap;

#[cfg(feature = "ane")]
struct AneLanes {
    dir: std::path::PathBuf,
    units: objc2_core_ml::MLComputeUnits,
    /// Weight sources by layer (raw pointers into the model's tensors).
    weights: Vec<LayerSrc>,
    dims: Dims,
    nets: HashMap<(String, u32, usize), crate::exec::ane::AneNet>,
    xin: Vec<f32>,
    atin: Vec<f32>,
    xo: Vec<f32>,
    qkv: Vec<f32>,
    tmp: Vec<f32>,
    /// Three programs per layer (o_proj | norm+MLP | norm+QKV) with the residual adds on the host
    /// in f32 (default; measured rel-L2 vs the CPU engine 0.009), instead of one fused program
    /// carrying the residual in fp16 (`PLOW_ANE_RESID=fused`; 0.12, same speed).
    host_resid: bool,
    pub compile_ms: f64,
    profile: Option<LaneProfile>,
}

#[cfg(feature = "ane")]
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    inter: usize,
    qd: usize,
    kd: usize,
    eps: f32,
    gelu: bool,
    encoding: WeightEncoding,
}

/// One weight matrix in the plan's encoding; scale storage is f32 for FP8, E8M0 for MXFP4.
#[cfg(feature = "ane")]
#[derive(Clone, Copy)]
struct WSrc {
    w: *const u8,
    scale: *const f32,
    n: usize,
    k: usize,
}

#[cfg(feature = "ane")]
#[derive(Clone, Copy)]
struct LayerSrc {
    g_in: *const u16,
    g_pa: *const u16,
    wq: WSrc,
    wk: WSrc,
    wv: WSrc,
    wo: WSrc,
    wg: WSrc,
    wu: WSrc,
    wd: WSrc,
}

// SAFETY: the pointers target the model's tensors, which outlive the engine.
#[cfg(feature = "ane")]
unsafe impl Send for AneLanes {}

#[cfg(feature = "ane")]
/// `PLOW_ANE_W8=1`: ANE programs carry 8-bit linear-quantized weights (half the bytes).
pub fn ane_w8() -> bool {
    crate::config::RuntimeConfig::get().apple.ane_w8
}

fn e4m3_lut() -> [f32; 256] {
    let mut lut = [0f32; 256];
    for (b, v) in lut.iter_mut().enumerate() {
        let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = ((b >> 3) & 0xf) as i32;
        let m = (b & 7) as f32;
        *v = if e == 0 {
            s * m / 8.0 * 2f32.powi(-6)
        } else if e == 15 && m == 7.0 {
            f32::NAN
        } else {
            s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
        };
    }
    lut
}

#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    let x = v.to_bits();
    if (x & 0x7f80_0000) == 0x7f80_0000 {
        return (x >> 16) as u16 | if x & 0x7f_ffff != 0 { 0x40 } else { 0 };
    }
    let round = 0x7fff + ((x >> 16) & 1);
    ((x.wrapping_add(round)) >> 16) as u16
}

#[cfg(feature = "ane")]
impl AneLanes {
    fn new(model: &CpuModel, plan: &HeteroPlan, blob: &Path) -> Result<AneLanes> {
        let encoding = if plan.fp8 {
            if plan.weight_encoding == WeightEncoding::Mxfp4 {
                return Err(RuntimeError::Device(
                    "hetero: conflicting weight encodings".into(),
                ));
            }
            WeightEncoding::Fp8
        } else {
            plan.weight_encoding
        };
        if !(plan.arch == "llama" || plan.arch == "qwen3") || plan.mlp_act != 1 {
            return Err(RuntimeError::Device(format!(
                "hetero: the ANE lane implements Llama-style layers only (arch {:?}, mlp_act {})",
                plan.arch, plan.mlp_act
            )));
        }
        let ten = |name: &str| -> Result<(*const u8, usize)> {
            model
                .tensor_by_name(name)
                .map(|t| (t.as_ptr() as *const u8, t.bytes))
                .ok_or_else(|| RuntimeError::Device(format!("hetero: tensor {name} not in blob")))
        };
        let w = |name: &str, scale: &Option<String>, n: usize, k: usize| -> Result<WSrc> {
            let (p, bytes) = ten(name)?;
            let (need, scale_bytes) = match encoding {
                WeightEncoding::Bf16 => (n * k * 2, 0),
                WeightEncoding::Fp8 => (n * k, n * 4),
                WeightEncoding::Mxfp4 => {
                    if k % 2 != 0 {
                        return Err(RuntimeError::Device(format!(
                            "hetero: {name}: odd MXFP4 width {k}"
                        )));
                    }
                    (n * (k / 2), n * k.div_ceil(32))
                }
            };
            if bytes < need {
                return Err(RuntimeError::Device(format!(
                    "hetero: {name} has {bytes} bytes, need {need} for [{n}][{k}]"
                )));
            }
            let sc = match (scale_bytes, scale) {
                (0, _) => std::ptr::null(),
                (_, Some(s)) => {
                    let (p, bytes) = ten(s)?;
                    if bytes < scale_bytes {
                        return Err(RuntimeError::Device(format!(
                            "hetero: {s} has {bytes} bytes, need {scale_bytes}"
                        )));
                    }
                    p as *const f32
                }
                (_, None) => {
                    return Err(RuntimeError::Device(format!(
                        "hetero: {name}: quantized twin without scale"
                    )))
                }
            };
            Ok(WSrc {
                w: p,
                scale: sc,
                n,
                k,
            })
        };
        let (h, i, qd, kd) = (
            plan.hidden as usize,
            plan.inter as usize,
            plan.qd as usize,
            plan.kd as usize,
        );
        let mut weights = Vec::with_capacity(plan.layers.len());
        for l in &plan.layers {
            weights.push(LayerSrc {
                g_in: ten(&l.g_in)?.0 as *const u16,
                g_pa: ten(&l.g_pa)?.0 as *const u16,
                wq: w(&l.wq, &l.sq, qd, h)?,
                wk: w(&l.wk, &l.sk, kd, h)?,
                wv: w(&l.wv, &l.sv, kd, h)?,
                wo: w(&l.wo, &l.so, h, qd)?,
                wg: w(&l.wg, &l.sg, i, h)?,
                wu: w(&l.wu, &l.su, i, h)?,
                wd: w(&l.wd, &l.sd, h, i)?,
            });
        }
        let units = match crate::config::RuntimeConfig::get()
            .apple
            .ane_units
            .as_deref()
        {
            Some("cpu") => objc2_core_ml::MLComputeUnits::CPUOnly,
            Some("all") => objc2_core_ml::MLComputeUnits::All,
            _ => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
        };
        let max_rows = ANE_ROWS;
        Ok(AneLanes {
            dir: blob.with_file_name("ane"),
            units,
            weights,
            dims: Dims {
                hidden: h,
                inter: i,
                qd,
                kd,
                eps: plan.eps,
                gelu: plan.mlp_act == 0,
                encoding,
            },
            nets: HashMap::new(),
            xin: vec![0.0; max_rows * h],
            atin: vec![0.0; max_rows * qd],
            xo: vec![0.0; max_rows * h],
            qkv: vec![0.0; max_rows * (qd + 2 * kd)],
            tmp: vec![0.0; max_rows * h],
            host_resid: crate::config::RuntimeConfig::get()
                .apple
                .ane_resid
                .as_deref()
                != Some("fused"),
            compile_ms: 0.0,
            profile: None,
        })
    }

    /// fp16 `[n][k]` of `src` with `factor[k]` (a folded norm gamma) and a scalar `mul` applied.
    fn fold(&self, src: WSrc, factor: Option<*const u16>, mul: f32, out: &mut [u16]) {
        let lut = e4m3_lut();
        let (n, k, encoding) = (src.n, src.k, self.dims.encoding);
        let threads = 8usize;
        let chunk = n.div_ceil(threads);
        let f = factor.map(|p| p as usize);
        let (wp, sp) = (src.w as usize, src.scale as usize);
        std::thread::scope(|sc| {
            for (ti, o) in out.chunks_mut(chunk * k).enumerate() {
                let lut = &lut;
                sc.spawn(move || {
                    let n0 = ti * chunk;
                    for (r, row) in o.chunks_mut(k).enumerate() {
                        let nn = n0 + r;
                        // SAFETY: row `nn < n` of a validated `[n][k]` tensor; the gamma has k entries.
                        unsafe {
                            let s = if encoding == WeightEncoding::Fp8 {
                                *(sp as *const f32).add(nn)
                            } else {
                                1.0
                            } * mul;
                            for kk in 0..k {
                                let wv = match encoding {
                                    WeightEncoding::Fp8 => {
                                        lut[*(wp as *const u8).add(nn * k + kk) as usize]
                                    }
                                    WeightEncoding::Bf16 => {
                                        bf16_to_f32(*(wp as *const u16).add(nn * k + kk))
                                    }
                                    WeightEncoding::Mxfp4 => {
                                        const E2M1: [f32; 16] = [
                                            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5,
                                            -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
                                        ];
                                        let packed = *(wp as *const u8).add(nn * (k / 2) + kk / 2);
                                        let code = (packed >> ((kk % 2) * 4)) & 15;
                                        let scale =
                                            *(sp as *const u8).add(nn * k.div_ceil(32) + kk / 32);
                                        E2M1[code as usize] * f32::from_bits((scale as u32) << 23)
                                    }
                                };
                                let g = match f {
                                    Some(fp) => bf16_to_f32(*(fp as *const u16).add(kk)),
                                    None => 1.0,
                                };
                                row[kk] = crate::exec::ane::f32_to_f16(wv * s * g);
                            }
                        }
                    }
                });
            }
        });
    }

    fn spec(&self, lane: &AneLane, t: usize) -> crate::exec::ane::NetSpec {
        use crate::exec::ane::{Layer, NetSpec};
        let d = self.dims;
        let l = lane.layer as usize;
        let mut layers = Vec::new();
        let mut inputs = vec![("x".to_string(), d.hidden)];
        let mut outputs = Vec::new();
        let qkv_ip = |layers: &mut Vec<Layer>, src: &str, l: usize| {
            let ws = &self.weights[l];
            let n = d.qd + 2 * d.kd;
            let mut w16 = vec![0u16; n * d.hidden];
            self.fold(ws.wq, Some(ws.g_in), 1.0, &mut w16[..d.qd * d.hidden]);
            self.fold(
                ws.wk,
                Some(ws.g_in),
                1.0,
                &mut w16[d.qd * d.hidden..(d.qd + d.kd) * d.hidden],
            );
            self.fold(
                ws.wv,
                Some(ws.g_in),
                1.0,
                &mut w16[(d.qd + d.kd) * d.hidden..],
            );
            layers.push(Layer::RmsNorm {
                input: src.into(),
                output: format!("{src}_n"),
                c: d.hidden,
                eps: d.eps * RESID_SCALE * RESID_SCALE,
            });
            layers.push(Layer::InnerProduct {
                input: format!("{src}_n"),
                output: "qkv".into(),
                k: d.hidden,
                n,
                w_f16: w16,
            });
        };
        let post = |layers: &mut Vec<Layer>, l: usize| {
            let ws = &self.weights[l];
            let mut wo = vec![0u16; d.hidden * d.qd];
            self.fold(ws.wo, None, RESID_SCALE, &mut wo);
            layers.push(Layer::InnerProduct {
                input: "at".into(),
                output: "o".into(),
                k: d.qd,
                n: d.hidden,
                w_f16: wo,
            });
            layers.push(Layer::Add {
                a: "x".into(),
                b: "o".into(),
                output: "x1".into(),
            });
            layers.push(Layer::RmsNorm {
                input: "x1".into(),
                output: "hn1".into(),
                c: d.hidden,
                eps: d.eps * RESID_SCALE * RESID_SCALE,
            });
            let mut wg = vec![0u16; d.inter * d.hidden];
            self.fold(ws.wg, Some(ws.g_pa), 1.0, &mut wg);
            layers.push(Layer::InnerProduct {
                input: "hn1".into(),
                output: "g".into(),
                k: d.hidden,
                n: d.inter,
                w_f16: wg,
            });
            let mut wu = vec![0u16; d.inter * d.hidden];
            self.fold(ws.wu, Some(ws.g_pa), 1.0, &mut wu);
            layers.push(Layer::InnerProduct {
                input: "hn1".into(),
                output: "u".into(),
                k: d.hidden,
                n: d.inter,
                w_f16: wu,
            });
            if d.gelu {
                layers.push(Layer::Gelu {
                    input: "g".into(),
                    output: "act".into(),
                });
            } else {
                layers.push(Layer::Sigmoid {
                    input: "g".into(),
                    output: "sg".into(),
                });
                layers.push(Layer::Mul {
                    a: "g".into(),
                    b: "sg".into(),
                    output: "act".into(),
                });
            }
            layers.push(Layer::Mul {
                a: "act".into(),
                b: "u".into(),
                output: "h".into(),
            });
            let mut wd = vec![0u16; d.hidden * d.inter];
            self.fold(ws.wd, None, RESID_SCALE, &mut wd);
            layers.push(Layer::InnerProduct {
                input: "h".into(),
                output: "dn".into(),
                k: d.inter,
                n: d.hidden,
                w_f16: wd,
            });
            layers.push(Layer::Add {
                a: "x1".into(),
                b: "dn".into(),
                output: "xo".into(),
            });
        };
        match lane.kind.as_str() {
            "pre" => {
                qkv_ip(&mut layers, "x", l);
                outputs.push(("qkv".to_string(), d.qd + 2 * d.kd));
            }
            "post" => {
                inputs.push(("at".to_string(), d.qd));
                post(&mut layers, l);
                outputs.push(("xo".to_string(), d.hidden));
            }
            "mid" => {
                inputs.push(("at".to_string(), d.qd));
                post(&mut layers, l);
                qkv_ip(&mut layers, "xo", l + 1);
                outputs.push(("xo".to_string(), d.hidden));
                outputs.push(("qkv".to_string(), d.qd + 2 * d.kd));
            }
            // Host-residual programs (`host_resid`): plain o_proj / down outputs, no residual scale.
            "o" => {
                let ws = &self.weights[l];
                let mut wo = vec![0u16; d.hidden * d.qd];
                self.fold(ws.wo, None, 1.0, &mut wo);
                inputs = vec![("at".to_string(), d.qd)];
                layers.push(Layer::InnerProduct {
                    input: "at".into(),
                    output: "o".into(),
                    k: d.qd,
                    n: d.hidden,
                    w_f16: wo,
                });
                outputs.push(("o".to_string(), d.hidden));
            }
            "mlp" => {
                let ws = &self.weights[l];
                layers.push(Layer::RmsNorm {
                    input: "x".into(),
                    output: "hn1".into(),
                    c: d.hidden,
                    eps: d.eps * RESID_SCALE * RESID_SCALE,
                });
                let mut wg = vec![0u16; d.inter * d.hidden];
                self.fold(ws.wg, Some(ws.g_pa), 1.0, &mut wg);
                layers.push(Layer::InnerProduct {
                    input: "hn1".into(),
                    output: "g".into(),
                    k: d.hidden,
                    n: d.inter,
                    w_f16: wg,
                });
                let mut wu = vec![0u16; d.inter * d.hidden];
                self.fold(ws.wu, Some(ws.g_pa), 1.0, &mut wu);
                layers.push(Layer::InnerProduct {
                    input: "hn1".into(),
                    output: "u".into(),
                    k: d.hidden,
                    n: d.inter,
                    w_f16: wu,
                });
                if d.gelu {
                    layers.push(Layer::Gelu {
                        input: "g".into(),
                        output: "act".into(),
                    });
                } else {
                    layers.push(Layer::Sigmoid {
                        input: "g".into(),
                        output: "sg".into(),
                    });
                    layers.push(Layer::Mul {
                        a: "g".into(),
                        b: "sg".into(),
                        output: "act".into(),
                    });
                }
                layers.push(Layer::Mul {
                    a: "act".into(),
                    b: "u".into(),
                    output: "h".into(),
                });
                let mut wd = vec![0u16; d.hidden * d.inter];
                self.fold(ws.wd, None, 1.0, &mut wd);
                layers.push(Layer::InnerProduct {
                    input: "h".into(),
                    output: "dn".into(),
                    k: d.inter,
                    n: d.hidden,
                    w_f16: wd,
                });
                outputs.push(("dn".to_string(), d.hidden));
            }
            "qkv" => {
                qkv_ip(&mut layers, "x", l);
                outputs.push(("qkv".to_string(), d.qd + 2 * d.kd));
            }
            other => panic!("hetero: unknown ANE lane kind {other:?}"),
        }
        NetSpec {
            inputs,
            outputs,
            t_enum: vec![t],
            flex_outputs: false,
            range: false,
            out_range: crate::config::RuntimeConfig::get()
                .apple
                .out_range
                .is_some(),
            w8: ane_w8(),
            layers,
        }
    }

    fn io(&self, lane: &AneLane) -> (Vec<(String, usize)>, Vec<(String, usize)>) {
        let d = self.dims;
        let n = d.qd + 2 * d.kd;
        match lane.kind.as_str() {
            "pre" | "qkv" => (vec![("x".into(), d.hidden)], vec![("qkv".into(), n)]),
            "o" => (vec![("at".into(), d.qd)], vec![("o".into(), d.hidden)]),
            "mlp" => (vec![("x".into(), d.hidden)], vec![("dn".into(), d.hidden)]),
            "post" => (
                vec![("x".into(), d.hidden), ("at".into(), d.qd)],
                vec![("xo".into(), d.hidden)],
            ),
            _ => (
                vec![("x".into(), d.hidden), ("at".into(), d.qd)],
                vec![("xo".into(), d.hidden), ("qkv".into(), n)],
            ),
        }
    }

    fn net(&mut self, lane: &AneLane, t: usize) -> Result<&mut crate::exec::ane::AneNet> {
        let key = (lane.kind.clone(), lane.layer, t);
        if !self.nets.contains_key(&key) {
            let name = format!(
                "v1-l{}-{}-t{t}{}{}",
                lane.layer,
                lane.kind,
                if self.dims.encoding == WeightEncoding::Mxfp4 {
                    "-mx4"
                } else {
                    ""
                },
                if ane_w8() { "-w8" } else { "" }
            );
            let cached = crate::exec::ane::AneNet::cached(&self.dir, &name);
            let t0 = Instant::now();
            let io = self.io(lane);
            let net = crate::exec::ane::AneNet::new(
                &self.dir,
                &name,
                || self.spec(lane, t),
                io,
                self.units,
            )?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            self.compile_ms += ms;
            tracing::info!(program = %name, cached, ms = format!("{ms:.0}").as_str(), "ANE program ready");
            self.nets.insert(key.clone(), net);
        }
        Ok(self.nets.get_mut(&key).unwrap())
    }

    fn run_net(
        &mut self,
        lane: &AneLane,
        t: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> Result<()> {
        let mut timings = self
            .profile
            .map(|_| crate::exec::ane::RunTimings::default());
        self.net(lane, t)?
            .run_timed(t, inputs, outputs, timings.as_mut())?;
        if let (Some(profile), Some(timings)) = (&mut self.profile, timings) {
            profile.coreml.accumulate(timings);
        }
        Ok(())
    }

    fn profile_start(&self) -> Option<Instant> {
        self.profile.map(|_| Instant::now())
    }

    fn profile_host(&mut self, start: Option<Instant>, field: fn(&mut LaneProfile) -> &mut f64) {
        if let (Some(profile), Some(start)) = (&mut self.profile, start) {
            *field(profile) += start.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// Run `lane` on rows `[row0, row0+rows)` in [`ANE_ROWS`]-row calls (the last zero-padded).
    fn run(
        &mut self,
        lane: &AneLane,
        row0: usize,
        rows: usize,
        h: &Handles,
        table: &[*mut u8],
    ) -> Result<()> {
        let mut at = 0usize;
        while at < rows {
            let n = (rows - at).min(ANE_ROWS);
            self.run_call(lane, row0 + at, n, h, table)?;
            at += n;
        }
        Ok(())
    }

    /// The host-residual form of [`Self::run_call`]: o = P_o(at); x1 = x + o; d = P_mlp(x1);
    /// xo = x1 + d (f32 on the host); qkv = P_qkv(xo).
    fn run_call_host(
        &mut self,
        lane: &AneLane,
        row0: usize,
        rows: usize,
        h: &Handles,
        table: &[*mut u8],
    ) -> Result<()> {
        let d = self.dims;
        let (hid, qd, kd) = (d.hidden, d.qd, d.kd);
        let t = ANE_ROWS;
        let n = qd + 2 * kd;
        let l = lane.layer;
        let (mut xin, mut atin) = (
            std::mem::take(&mut self.xin),
            std::mem::take(&mut self.atin),
        );
        let (mut xo, mut qkv, mut tmp) = (
            std::mem::take(&mut self.xo),
            std::mem::take(&mut self.qkv),
            std::mem::take(&mut self.tmp),
        );
        let r = (|| -> Result<()> {
            let start = self.profile_start();
            // SAFETY: as in `run_call`.
            unsafe {
                let xp = (table[h.x] as *const u16).add(row0 * hid);
                for i in 0..rows * hid {
                    xo[i] = bf16_to_f32(*xp.add(i));
                }
                xo[rows * hid..t * hid].fill(0.0);
            }
            self.profile_host(start, |p| &mut p.input_convert_ms);
            if lane.kind != "pre" {
                let start = self.profile_start();
                unsafe {
                    let ap = (table[h.at] as *const u16).add(row0 * qd);
                    for i in 0..rows * qd {
                        atin[i] = bf16_to_f32(*ap.add(i));
                    }
                    atin[rows * qd..t * qd].fill(0.0);
                }
                self.profile_host(start, |p| &mut p.input_convert_ms);
                let key = AneLane {
                    kind: "o".into(),
                    layer: l,
                };
                self.run_net(&key, t, &[&atin[..t * qd]], &mut [&mut tmp[..t * hid]])?;
                let start = self.profile_start();
                for i in 0..t * hid {
                    xo[i] += tmp[i];
                    xin[i] = xo[i] * RESID_SCALE;
                }
                self.profile_host(start, |p| &mut p.residual_ms);
                let key = AneLane {
                    kind: "mlp".into(),
                    layer: l,
                };
                self.run_net(&key, t, &[&xin[..t * hid]], &mut [&mut tmp[..t * hid]])?;
                let start = self.profile_start();
                for i in 0..t * hid {
                    xo[i] += tmp[i];
                }
                self.profile_host(start, |p| &mut p.residual_ms);
                let start = self.profile_start();
                unsafe {
                    let xp = (table[h.x] as *mut u16).add(row0 * hid);
                    for i in 0..rows * hid {
                        *xp.add(i) = f32_to_bf16(xo[i]);
                    }
                }
                self.profile_host(start, |p| &mut p.output_convert_ms);
            }
            if lane.kind != "post" {
                let ql = if lane.kind == "pre" { l } else { l + 1 };
                let start = self.profile_start();
                for i in 0..t * hid {
                    xin[i] = xo[i] * RESID_SCALE;
                }
                self.profile_host(start, |p| &mut p.input_convert_ms);
                let key = AneLane {
                    kind: "qkv".into(),
                    layer: ql,
                };
                self.run_net(&key, t, &[&xin[..t * hid]], &mut [&mut qkv[..t * n]])?;
                let start = self.profile_start();
                unsafe {
                    let qp = (table[h.qg] as *mut u16).add(row0 * qd);
                    let kp = (table[h.kg] as *mut u16).add(row0 * kd);
                    let vp = (table[h.vg] as *mut u16).add(row0 * kd);
                    for r in 0..rows {
                        let src = &qkv[r * n..(r + 1) * n];
                        for j in 0..qd {
                            *qp.add(r * qd + j) = f32_to_bf16(src[j]);
                        }
                        for j in 0..kd {
                            *kp.add(r * kd + j) = f32_to_bf16(src[qd + j]);
                            *vp.add(r * kd + j) = f32_to_bf16(src[qd + kd + j]);
                        }
                    }
                }
                self.profile_host(start, |p| &mut p.output_convert_ms);
            }
            Ok(())
        })();
        self.xin = xin;
        self.atin = atin;
        self.xo = xo;
        self.qkv = qkv;
        self.tmp = tmp;
        r
    }

    /// One program call over `rows <= ANE_ROWS` real rows at `row0`.
    fn run_call(
        &mut self,
        lane: &AneLane,
        row0: usize,
        rows: usize,
        h: &Handles,
        table: &[*mut u8],
    ) -> Result<()> {
        if self.host_resid {
            return self.run_call_host(lane, row0, rows, h, table);
        }
        let d = self.dims;
        let (hid, qd, kd) = (d.hidden, d.qd, d.kd);
        let with_at = lane.kind != "pre";
        let with_qkv = lane.kind != "post";
        let t = ANE_ROWS;
        let start = self.profile_start();
        // SAFETY: the activation tensors are `[T][feat]` bf16 with T >= row0+rows (the bucket).
        unsafe {
            let xp = (table[h.x] as *const u16).add(row0 * hid);
            for i in 0..rows * hid {
                self.xin[i] = bf16_to_f32(*xp.add(i)) * RESID_SCALE;
            }
            self.xin[rows * hid..t * hid].fill(0.0);
            if with_at {
                let ap = (table[h.at] as *const u16).add(row0 * qd);
                for i in 0..rows * qd {
                    self.atin[i] = bf16_to_f32(*ap.add(i));
                }
                self.atin[rows * qd..t * qd].fill(0.0);
            }
        }
        self.profile_host(start, |p| &mut p.input_convert_ms);
        let n = qd + 2 * kd;
        let (xin, atin) = (
            std::mem::take(&mut self.xin),
            std::mem::take(&mut self.atin),
        );
        let (mut xo, mut qkv) = (std::mem::take(&mut self.xo), std::mem::take(&mut self.qkv));
        let r = {
            match lane.kind.as_str() {
                "pre" => self.run_net(lane, t, &[&xin[..t * hid]], &mut [&mut qkv[..t * n]]),
                "post" => self.run_net(
                    lane,
                    t,
                    &[&xin[..t * hid], &atin[..t * qd]],
                    &mut [&mut xo[..t * hid]],
                ),
                _ => self.run_net(
                    lane,
                    t,
                    &[&xin[..t * hid], &atin[..t * qd]],
                    &mut [&mut xo[..t * hid], &mut qkv[..t * n]],
                ),
            }
        };
        self.xin = xin;
        self.atin = atin;
        self.xo = xo;
        self.qkv = qkv;
        r?;
        let start = self.profile_start();
        // SAFETY: as above; this lane owns exactly these rows of x/qg/kg/vg in this segment.
        unsafe {
            if with_at {
                let xp = (table[h.x] as *mut u16).add(row0 * hid);
                for i in 0..rows * hid {
                    *xp.add(i) = f32_to_bf16(self.xo[i] / RESID_SCALE);
                }
            }
            if with_qkv {
                let qp = (table[h.qg] as *mut u16).add(row0 * qd);
                let kp = (table[h.kg] as *mut u16).add(row0 * kd);
                let vp = (table[h.vg] as *mut u16).add(row0 * kd);
                for r in 0..rows {
                    let src = &self.qkv[r * n..(r + 1) * n];
                    for j in 0..qd {
                        *qp.add(r * qd + j) = f32_to_bf16(src[j]);
                    }
                    for j in 0..kd {
                        *kp.add(r * kd + j) = f32_to_bf16(src[qd + j]);
                        *vp.add(r * kd + j) = f32_to_bf16(src[qd + kd + j]);
                    }
                }
            }
        }
        self.profile_host(start, |p| &mut p.output_convert_ms);
        Ok(())
    }
}
