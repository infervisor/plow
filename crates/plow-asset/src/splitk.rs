use crate::program::{Packet, Program};
use packet::dev::{DevOp, TENSOR_NONE16};
use std::collections::{BTreeMap, BTreeSet};
const Z: u16 = DevOp::ZeroF32 as u16;
const G: u16 = DevOp::GemmSplitK as u16;
const F: u16 = DevOp::CastF32Bf16 as u16;
type Result<T> = std::result::Result<T, String>;
fn check(ok: bool, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("splitK packet: {why}"))
    }
}
pub fn capability(marker: Option<u32>, block: Option<u32>, arena: u32) -> Result<()> {
    check(
        marker == Some(1) && block == Some(256) && arena >= 82944,
        "requires Hopper splitK ABI1, block256 and arena82944",
    )
}
pub fn dependencies(g: &Program<'_>) -> Result<Vec<BTreeSet<usize>>> {
    check(
        g.n_counter as usize == g.insts.len(),
        "only ordinary coarse counters supported",
    )?;
    check(
        g.gq_seg_ofs == [0, g.gq_stream.len() as u32] && g.l2_domains == 0,
        "requires one ordinary GQ window",
    )?;
    check(
        !g.insts.is_empty() && g.stream_ofs.len() == g.stream_len.len(),
        "invalid static stream tables",
    )?;
    let mut end = 0usize;
    for (&offset, &length) in g.stream_ofs.iter().zip(g.stream_len) {
        check(
            offset as usize == end,
            "noncontiguous static stream partition",
        )?;
        end = end
            .checked_add(length as usize)
            .ok_or("static partition overflow")?;
        check(end <= g.stream.len(), "static partition out of range")?;
    }
    check(end == g.stream.len(), "missing static partition")?;
    let mut lists: Vec<Option<BTreeSet<usize>>> = vec![None; g.insts.len()];
    let mut seen = BTreeSet::new();
    let mut entries = BTreeMap::new();
    for e in g.stream {
        let pc = e.inst as usize;
        check(pc < g.insts.len(), "bad instruction")?;
        check(
            e.flags == 0 && e.seg == 0 && e.slice < g.insts[pc].blocks as u32,
            "noncoarse or invalid slice",
        )?;
        check(seen.insert((pc, e.slice)), "duplicate static slice")?;
        entries.insert((pc, e.slice), e);
        let succ = g
            .succs
            .get(e.succ_ofs as usize..e.succ_ofs as usize + e.succ_len as usize)
            .ok_or_else(|| "splitK bad successor range".to_string())?;
        check(succ == [pc as u32], "coarse own counter required")?;
        let waits = g
            .waits
            .get(e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize)
            .ok_or_else(|| "splitK bad wait range".to_string())?;
        let mut set = BTreeSet::new();
        for w in waits {
            let id = w.id as usize;
            check(
                id < pc && w.threshold == g.insts[id].blocks as u32,
                "missing full producer threshold",
            )?;
            check(set.insert(id), "duplicate dependency")?;
        }
        if let Some(prior) = &lists[pc] {
            check(*prior == set, "slice dependencies differ")?;
        } else {
            lists[pc] = Some(set);
        }
    }
    for (pc, i) in g.insts.iter().enumerate() {
        check(
            i.blocks > 0 && (0..i.blocks as u32).all(|s| seen.contains(&(pc, s))),
            "missing slice",
        )?;
    }
    let lists: Vec<_> = lists.into_iter().map(|v| v.unwrap()).collect();
    let mut qseen = BTreeSet::new();
    let mut last = vec![0usize; g.insts.len()];
    for (q, e) in g.gq_stream.iter().enumerate() {
        let pc = e.inst as usize;
        check(
            seen.contains(&(pc, e.slice)) && qseen.insert((pc, e.slice)),
            "bad GQ coverage",
        )?;
        last[pc] = q;
    }
    check(qseen == seen, "missing GQ entries")?;
    for (q, e) in g.gq_stream.iter().enumerate() {
        let original = entries[&(e.inst as usize, e.slice)];
        check(
            e.wait_ofs == original.wait_ofs
                && e.wait_len == original.wait_len
                && e.succ_ofs == original.succ_ofs
                && e.succ_len == original.succ_len
                && e.flags == 0
                && e.seg == 0,
            "GQ gates differ",
        )?;
        check(
            lists[e.inst as usize].iter().all(|&id| last[id] < q),
            "GQ producer queued after consumer",
        )?;
    }
    Ok(lists)
}
fn write_slots(op: u16) -> Result<&'static [usize]> {
    Ok(
        match DevOp::from_u16(op).ok_or("unknown opcode in splitK program")? {
            DevOp::Nop => &[],
            DevOp::RmsNorm => &[0, 3, 4],
            DevOp::AddNorm | DevOp::NormResidualNorm | DevOp::FlashDecode => &[0, 1],
            DevOp::GemvQkv => &[0, 3, 5],
            DevOp::Gemv
            | DevOp::GemvGlu
            | DevOp::HeadNormRope
            | DevOp::FlashMerge
            | DevOp::NormResidual
            | DevOp::Embed
            | DevOp::Residual
            | DevOp::Glu
            | DevOp::SoftCap
            | DevOp::Argmax
            | DevOp::ArgmaxFin
            | DevOp::ZeroF32
            | DevOp::GemmSplitK
            | DevOp::CastF32Bf16 => &[0],
            _ => return Err("unaudited writer in splitK program".into()),
        },
    )
}
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionAccess {
    pub program: usize,
    pub zero_pc: usize,
    pub partial: u16,
    pub input: u16,
    pub weight: u16,
    pub output: u16,
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub split: u32,
}
#[derive(Debug)]
pub struct Validated {
    pub canonical: Vec<Canonical>,
    pub access: Vec<ProjectionAccess>,
}
pub fn validate(blob: &Packet<'_>) -> Result<Option<Validated>> {
    let has = blob
        .programs
        .iter()
        .any(|p| p.insts.iter().any(|i| matches!(i.op, Z | G | F)));
    if !has {
        return Ok(None);
    }
    check(
        blob.n_cu > 0 && blob.n_cu <= u16::MAX as u32 && !blob.tp,
        "requires a positive single-GPU grid",
    )?;
    let lo = blob.prefill_count;
    check(lo < blob.programs.len(), "missing decode programs")?;
    check(
        blob.programs[..lo]
            .iter()
            .all(|p| p.insts.iter().all(|i| !matches!(i.op, Z | G | F))),
        "prefill splitK unsupported",
    )?;
    let mut used = BTreeSet::new();
    for g in &blob.programs[lo..] {
        check(
            !g.packed_prefill_only && g.rows > 0,
            "packed or empty decode program",
        )?;
        check(
            g.stream_ofs.len() == blob.n_cu as usize,
            "static grid differs from packet",
        )?;
        validate_program(blob, g, &mut used)?;
    }
    for (pi, p) in blob.programs.iter().enumerate() {
        for i in p.insts {
            for (slot, &h) in i.t.iter().enumerate() {
                if used.contains(&h) {
                    check(
                        pi >= lo
                            && ((i.op == Z || i.op == G) && slot == 0 || i.op == F && slot == 1),
                        "scratch used outside partial operands",
                    )?;
                }
            }
        }
    }
    let mut access = Vec::new();
    for (program, p) in blob.programs.iter().enumerate().skip(lo) {
        for (pc, d) in p.insts.iter().enumerate().filter(|(_, d)| d.op == G) {
            access.push(ProjectionAccess {
                program,
                zero_pc: pc - 1,
                partial: d.t[0],
                input: d.t[1],
                weight: d.t[2],
                output: p.insts[pc + 1].t[0],
                rows: d.i[0],
                n: d.i[1],
                k: d.i[2],
                split: d.i[3],
            });
        }
    }
    let operands: BTreeSet<_> = access
        .iter()
        .flat_map(|a| [a.partial, a.input, a.weight, a.output])
        .collect();
    let widest = blob
        .programs
        .last()
        .ok_or("missing widest decode program")?;
    for &index in blob.kv_row_insts {
        let instruction = widest
            .insts
            .get(index as usize)
            .ok_or("KV row patch out of range")?;
        check(
            instruction.op == DevOp::HeadNormRope as u16
                && !operands.contains(&instruction.t[0])
                && !instruction.t.iter().any(|h| used.contains(h)),
            "KV row patch touches splitK storage or unsupported opcode",
        )?;
    }
    for generated in blob.generated {
        check(
            !operands.iter().any(|&h| u32::from(h) == generated.tensor),
            "generated target aliases splitK operand",
        )?;
        let mut handles = vec![generated.tensor];
        match generated.kind {
            packet::rope::GEN_TMAP_BF16 | packet::rope::GEN_TMAP_E4M3 => {
                handles.push(generated.aux)
            }
            packet::rope::GEN_TMAP_KV_PAIR => handles.extend([generated.aux, generated.scale]),
            _ => {}
        }
        check(
            !used.iter().any(|&h| handles.contains(&u32::from(h))),
            "scratch used by generated tensor",
        )?;
    }
    let canonical = blob.programs[lo..]
        .iter()
        .map(canonical)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(Validated { canonical, access }))
}
fn validate_program(blob: &Packet<'_>, g: &Program<'_>, used: &mut BTreeSet<u16>) -> Result<()> {
    let dep = dependencies(g)?;
    let mut previous: BTreeMap<u16, usize> = BTreeMap::new();
    let mut pc = 0;
    while pc < g.insts.len() {
        if !matches!(g.insts[pc].op, Z | G | F) {
            pc += 1;
            continue;
        }
        check(pc + 2 < g.insts.len(), "incomplete triple")?;
        let (z, c, f) = (&g.insts[pc], &g.insts[pc + 1], &g.insts[pc + 2]);
        check(
            z.op == Z && c.op == G && f.op == F,
            "requires adjacent zero/compute/cast",
        )?;
        let (m, n, k, s) = (c.i[0], c.i[1], c.i[2], c.i[3]);
        check(
            matches!(m, 4 | 8 | 16)
                && g.rows == m
                && n > 0
                && k > 0
                && k % 8 == 0
                && matches!(s, 1 | 2 | 4 | 8 | 16),
            "unsupported geometry or S",
        )?;
        check(
            n <= i32::MAX as u32 - 127
                && k <= i32::MAX as u32 - 63
                && n.div_ceil(128) as u64 * s as u64 <= i32::MAX as u64,
            "device integer overflow",
        )?;
        check(
            z.i[..2] == [m, n]
                && f.i[..2] == [m, n]
                && z.i[2..].iter().chain(&f.i[2..]).all(|&x| x == 0)
                && c.i[4..].iter().all(|&x| x == 0),
            "reserved immediates",
        )?;
        check(
            [z, c, f]
                .iter()
                .all(|i| u32::from(i.blocks) == blob.n_cu && i.fj == [0; 3]),
            "blocks or reserved float words",
        )?;
        check(
            z.t[1..]
                .iter()
                .chain(&c.t[3..])
                .chain(&f.t[2..])
                .all(|&x| x == TENSOR_NONE16),
            "unused tensor operands",
        )?;
        let (partial, a, w, out) = (c.t[0], c.t[1], c.t[2], f.t[0]);
        let handles = BTreeSet::from([partial, a, w, out]);
        check(
            handles.len() == 4 && !handles.contains(&TENSOR_NONE16),
            "aliases or absent operands",
        )?;
        check(
            z.t[0] == partial && f.t[1] == partial,
            "partial identity mismatch",
        )?;
        let tensor = |h: u16| {
            blob.tensors
                .get(h as usize)
                .ok_or_else(|| "splitK bad tensor handle".to_string())
        };
        let bytes = |x: u32, y: u32, e: u64| {
            (x as u64)
                .checked_mul(y as u64)
                .and_then(|v| v.checked_mul(e))
                .ok_or_else(|| "splitK size overflow".to_string())
        };
        check(
            tensor(partial)?.bytes >= bytes(m, n, 4)?
                && tensor(partial)?.name.starts_with("act.")
                && !tensor(partial)?.initialized,
            "partial allocation must contain runtime FP32 plane",
        )?;
        check(
            tensor(a)?.bytes >= bytes(m, k, 2)?
                && tensor(out)?.bytes >= bytes(m, n, 2)?
                && tensor(w)?.bytes == bytes(n, k, 2)?,
            "tensor extent",
        )?;
        check(
            packet::names::is_checkpoint_weight(&tensor(w)?.name)
                && tensor(a)?.name.starts_with("act.")
                && tensor(out)?.name.starts_with("act."),
            "BF16 activation/weight namespaces",
        )?;
        check(
            dep[pc + 1].contains(&pc) && dep[pc + 2] == BTreeSet::from([pc + 1]),
            "missing zero/producer edge",
        )?;
        check(
            dep[pc] == previous.get(&partial).copied().into_iter().collect(),
            "zero must wait exactly on prior scratch finalizer",
        )?;
        let ordered = |later: usize, earlier: usize| {
            let mut stack = vec![later];
            let mut visited = BTreeSet::new();
            while let Some(next) = stack.pop() {
                if next == earlier {
                    return true;
                }
                if next > earlier && visited.insert(next) {
                    stack.extend(dep[next].iter().copied());
                }
            }
            false
        };
        for (other_pc, instruction) in g.insts.iter().enumerate() {
            let writes = write_slots(instruction.op)?;
            check(
                !writes.iter().any(|&slot| instruction.t[slot] == w),
                "checkpoint weight is written by a packet",
            )?;
            if writes.iter().any(|&slot| instruction.t[slot] == a) {
                check(
                    if other_pc < pc {
                        ordered(pc + 1, other_pc)
                    } else {
                        ordered(other_pc, pc + 1)
                    },
                    "activation writer is unordered with splitK read",
                )?;
            }
            if !instruction.t.contains(&out) || (pc..pc + 3).contains(&other_pc) {
                continue;
            }
            check(
                if other_pc < pc {
                    ordered(pc + 1, other_pc)
                } else {
                    ordered(other_pc, pc + 2)
                },
                "output storage access is unordered with finalizer",
            )?;
        }
        previous.insert(partial, pc + 2);
        used.insert(partial);
        for (consumer, d) in dep.iter().enumerate() {
            check(
                !d.contains(&pc) || consumer == pc + 1,
                "zero counter escapes triple",
            )?;
            check(
                !d.contains(&(pc + 1)) || consumer == pc + 2,
                "partial producer counter escapes triple",
            )?;
        }
        pc += 3;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub struct Canonical {
    pub instructions: Vec<packet::dev::DevInst64>,
    pub dependencies: Vec<BTreeSet<usize>>,
}
fn canonical(g: &Program<'_>) -> Result<Canonical> {
    let dependencies = dependencies(g)?;
    let mut insts = Vec::new();
    let mut logical = vec![None; g.insts.len()];
    let mut origins = Vec::new();
    let mut pc = 0;
    while pc < g.insts.len() {
        let mut d = g.insts[pc];
        let index = insts.len();
        if d.op == Z {
            check(
                pc + 2 < g.insts.len() && g.insts[pc + 1].op == G && g.insts[pc + 2].op == F,
                "unvalidated canonical triple",
            )?;
            let c = g.insts[pc + 1];
            let f = g.insts[pc + 2];
            d = c;
            d.op = DevOp::Gemv as u16;
            d.t.fill(TENSOR_NONE16);
            d.t[..3].copy_from_slice(&[f.t[0], c.t[1], c.t[2]]);
            d.i[3] = 0;
            logical[pc + 1] = Some(index);
            logical[pc + 2] = Some(index);
            origins.push((pc + 1, Some(pc)));
            pc += 3;
        } else {
            check(!matches!(d.op, G | F), "orphan canonical compute/cast")?;
            if d.op == DevOp::Gemv as u16 && d.i[3] == 0 {
                d.fj[0] = 0;
            }
            logical[pc] = Some(index);
            origins.push((pc, None));
            pc += 1;
        }
        insts.push(d);
    }
    let mut graph = Vec::new();
    for (pc, zero) in origins {
        let mut set = BTreeSet::new();
        for &id in &dependencies[pc] {
            if Some(id) == zero {
                continue;
            }
            set.insert(
                logical[id].ok_or_else(|| "splitK dependency on internal zero".to_string())?,
            );
        }
        graph.push(set);
    }
    Ok(Canonical {
        instructions: insts,
        dependencies: graph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Builder, Model};
    fn fixture(s: u32, reuse: bool) -> Model {
        let mut b = Builder::new(132);
        b.force_uniseg();
        let a = b.tensor("act.a", 16 * 64 * 2);
        let w = b.tensor("model.layers.0.mlp.down_proj.weight", 128 * 64 * 2);
        let c = b.tensor("act.c", 16 * 128 * 2);
        let p = b.tensor("act.partial", 16 * 128 * 4);
        let mut prior = None;
        for _ in 0..if reuse { 2 } else { 1 } {
            let z = b.emit(
                DevOp::ZeroF32,
                (0..132).collect(),
                &prior.into_iter().collect::<Vec<_>>(),
                |i| {
                    i.t[0] = p;
                    i.i[..2].copy_from_slice(&[16, 128]);
                },
            );
            let g = b.emit(DevOp::GemmSplitK, (0..132).collect(), &[z], |i| {
                i.t[..3].copy_from_slice(&[p, a, w]);
                i.i[..4].copy_from_slice(&[16, 128, 64, s]);
            });
            prior = Some(b.emit(DevOp::CastF32Bf16, (0..132).collect(), &[g], |i| {
                i.t[..2].copy_from_slice(&[c, p]);
                i.i[..2].copy_from_slice(&[16, 128]);
            }));
        }
        let tensors = b.tensors();
        let prog = b.finish();
        let model = Model {
            n_cu: 132,
            target: 0,
            tensors,
            progs: vec![prog],
            kv_row_insts: vec![],
            prog_t: vec![16],
            gen: vec![],
        };
        model
    }
    #[test]
    fn valid_s_and_scratch_reuse() {
        for s in [1, 2, 4, 8, 16] {
            for reuse in [false, true] {
                assert!(crate::program::with_model(&fixture(s, reuse), validate)
                    .unwrap()
                    .is_some());
            }
        }
    }
    #[test]
    fn bad_fields_and_geometry() {
        for case in 0..12 {
            let mut b = fixture(8, false);
            match case {
                0 => b.progs[0].insts[1].i[3] = 3,
                1 => b.progs[0].insts[1].i[2] = 63,
                2 => b.progs[0].insts[1].i[0] = 32,
                3 => b.progs[0].insts[1].i[4] = 1,
                4 => b.progs[0].insts[1].f[0] = 1.0,
                5 => b.progs[0].insts[0].op = F,
                6 => b.tensors[3].bytes -= 4,
                7 => b.progs[0].insts[1].t[1] = b.progs[0].insts[1].t[0],
                8 => b.progs[0].insts[1].t[7] = 1,
                9 => b.progs[0].insts[2].t[1] = 0,
                10 => b.progs[0].insts[1].blocks = 131,
                11 => b.progs[0].insts[1].i[1] = u32::MAX,
                _ => unreachable!(),
            };
            assert!(
                crate::program::with_model(&b, validate).is_err(),
                "case {case}"
            );
        }
    }
    #[test]
    fn missing_edges_thresholds_slices_and_reordered_queue() {
        for case in 0..7 {
            let mut b = fixture(8, true);
            let g = &mut b.progs[0];
            match case {
                0 | 1 | 2 => {
                    let pc = [1, 2, 3][case];
                    for e in g
                        .stream
                        .iter_mut()
                        .chain(&mut g.gq_stream)
                        .filter(|e| e.inst == pc)
                    {
                        e.wait_len = 0;
                    }
                }
                3 => g.waits[0].threshold = 131,
                4 => {
                    g.gq_stream.remove(0);
                }
                5 => {
                    g.gq_stream.swap(0, 132);
                }
                6 => {
                    for e in g
                        .stream
                        .iter_mut()
                        .chain(&mut g.gq_stream)
                        .filter(|e| e.inst == 1)
                    {
                        e.flags = packet::dev::SE_XCTR;
                    }
                }
                _ => unreachable!(),
            };
            assert!(
                crate::program::with_model(&b, validate).is_err(),
                "case {case}"
            );
        }
    }
    #[test]
    fn scratch_cannot_be_read_as_activation() {
        let mut b = fixture(8, true);
        b.progs[0].insts[4].t[1] = 3;
        assert!(crate::program::with_model(&b, validate).is_err());
    }
    fn with_input_writer(after: bool, ordered: bool, target: u32) -> Model {
        let mut model = fixture(8, false);
        let old = model.progs.remove(0);
        let mut b = Builder::new(132);
        b.force_uniseg();
        b.adopt_tensors(model.tensors.clone());
        let writer = |b: &mut Builder, deps: &[u32]| {
            b.emit(DevOp::Residual, (0..132).collect(), deps, |d| {
                d.t[0] = target
            })
        };
        let before = if after {
            None
        } else {
            Some(writer(&mut b, &[]))
        };
        let z = b.emit(DevOp::ZeroF32, (0..132).collect(), &[], |d| {
            *d = old.insts[0]
        });
        let mut deps = vec![z];
        if ordered {
            deps.extend(before);
        }
        let g = b.emit(DevOp::GemmSplitK, (0..132).collect(), &deps, |d| {
            *d = old.insts[1]
        });
        let f = b.emit(DevOp::CastF32Bf16, (0..132).collect(), &[g], |d| {
            *d = old.insts[2]
        });
        if after {
            writer(
                &mut b,
                if ordered {
                    std::slice::from_ref(&f)
                } else {
                    &[]
                },
            );
        }
        model.progs.push(b.finish());
        model
    }
    #[test]
    fn input_writers_must_be_ordered_and_weights_immutable() {
        for after in [false, true] {
            for ordered in [false, true] {
                let m = with_input_writer(after, ordered, 0);
                assert_eq!(crate::program::with_model(&m, validate).is_ok(), ordered);
                let m = with_input_writer(after, ordered, 1);
                assert!(crate::program::with_model(&m, validate).is_err());
            }
        }
    }
    #[test]
    fn generated_targets_and_kv_patches_cannot_alias_projection_storage() {
        for target in 0..4 {
            let mut m = fixture(8, false);
            m.gen.push(packet::rope::GenTensor {
                tensor: target,
                kind: packet::rope::GEN_TMAP_BF16,
                ctx: 1,
                hd: 1,
                aux: 0,
                scale: 0,
                theta: 0.,
                frac: 0.,
                factor: 0.,
                low: 0.,
                high: 0.,
                orig: 0.,
            });
            assert!(crate::program::with_model(&m, validate).is_err());
        }
        let mut m = with_input_writer(true, true, 0);
        m.progs[0].insts[3].op = DevOp::HeadNormRope as u16;
        m.kv_row_insts.push(3);
        assert!(crate::program::with_model(&m, validate).is_err());
        let m = fixture(8, false);
        crate::program::with_model(&m, |p| {
            let mut programs = p.programs.to_vec();
            programs[0].packed_prefill_only = true;
            let packet = Packet {
                programs: &programs,
                ..*p
            };
            assert!(validate(&packet).is_err());
        });
    }
    #[test]
    fn old_object_capability_rejected() {
        assert!(capability(None, Some(256), 82944).is_err());
        assert!(capability(Some(1), Some(512), 82944).is_err());
        assert!(capability(Some(1), Some(256), 16448).is_err());
        assert!(capability(Some(1), Some(256), 82944).is_ok());
    }
    #[test]
    fn all_schedule_orders_preserve_partial_lifetime() {
        // Slice completions are symmetric. Enumerating counters covers every interleaving.
        fn walk(c: [u8; 6], seen: &mut BTreeSet<[u8; 6]>) {
            if !seen.insert(c) {
                return;
            }
            let totals = [2, 3, 2, 2, 3, 2];
            for op in 0..6 {
                if c[op] == totals[op] || op > 0 && c[op - 1] != totals[op - 1] {
                    continue;
                }
                let mut n = c;
                n[op] += 1;
                if op == 1 {
                    assert_eq!(c[0], 2)
                }
                if op == 2 {
                    assert_eq!(c[1], 3)
                }
                if op == 3 {
                    assert_eq!(c[2], 2)
                }
                if op == 4 {
                    assert_eq!(c[3], 2)
                }
                if op == 5 {
                    assert_eq!(c[4], 3)
                }
                walk(n, seen);
            }
        }
        let mut seen = BTreeSet::new();
        walk([0; 6], &mut seen);
        assert!(seen.contains(&[2, 3, 2, 2, 3, 2]));
        assert_eq!(seen.len(), 15);
    }
}
