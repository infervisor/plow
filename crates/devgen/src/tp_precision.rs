use packet::dev::{DevOp, SE_XR_WAVE_RS, TENSOR_NONE, XREDUCE_PARTIAL_F32_MARK};
use packet::devbuild::{decode_rung_lo, Model};

pub(crate) fn apply(
    model: &mut Model,
    enabled: bool,
    arch: &str,
    slots: [(u32, u32); 2],
) -> Result<(), String> {
    if !enabled {
        return Ok(());
    }
    let nvidia = arch == "sm_120a";
    if (arch != "gfx1201" && !nvidia) || slots[0].0 == slots[1].0 {
        return Err(
            "FP32 TP partials require gfx1201 or sm_120a and two distinct peer slots".into(),
        );
    }
    let mut old_bytes = None;
    for &(partial, output) in &slots {
        let tensor = model
            .tensors
            .get(partial as usize)
            .ok_or("missing peer partial")?;
        if partial == output
            || tensor.bytes == 0
            || tensor.bytes % 2 != 0
            || tensor.init.is_some()
            || model
                .tensors
                .iter()
                .any(|t| t.name.starts_with(&format!("{}@", tensor.name)))
        {
            return Err("invalid peer partial storage or alias".into());
        }
        if old_bytes
            .replace(tensor.bytes)
            .is_some_and(|n| n != tensor.bytes)
        {
            return Err("peer partial slots have different capacities".into());
        }
    }
    let old_bytes = old_bytes.unwrap();
    let new_bytes = old_bytes
        .checked_mul(2)
        .filter(|&n| n <= u32::MAX as u64)
        .ok_or("peer partial byte overflow")?;
    let mut ranks = None;
    let mut marked = 0;
    let decode = decode_rung_lo(&model.prog_t);
    for (pi, prog) in model.progs.iter().enumerate() {
        let fp32_slot = if pi < decode { 0 } else { 1 };
        if prog.stream.iter().any(|e| e.flags & SE_XR_WAVE_RS != 0) {
            return Err("FP32 peer partials do not support wave-RS objects".into());
        }
        let mut produced = [None; 2];
        for (si, &(partial, _)) in slots.iter().enumerate() {
            let tensor = &model.tensors[partial as usize];
            let declared = prog
                .tensors
                .get(partial as usize)
                .ok_or("missing program partial")?;
            if declared.bytes != old_bytes || declared.name != tensor.name {
                return Err("program peer layout differs from model".into());
            }
            for inst in &prog.insts {
                if !inst.t.contains(&partial) {
                    continue;
                }
                if inst.t[0] != partial
                    || inst.t[1..].contains(&partial)
                    || !matches!(
                        DevOp::from_u16(inst.op),
                        Some(
                            DevOp::Gemv
                                | DevOp::GemmSmall
                                | DevOp::GemmMed
                                | DevOp::Gemm
                                | DevOp::GemmWide
                        )
                    )
                    || inst.t[1..3].contains(&TENSOR_NONE)
                    || inst.t[3..].iter().any(|&t| t != TENSOR_NONE)
                    || inst.i[..3].contains(&0)
                    || inst.i[3..].iter().any(|&v| v != 0)
                    || inst.f[1].to_bits() != 0
                    || inst.j != [0, 0]
                    || u64::from(inst.i[0]) * u64::from(inst.i[1]) > old_bytes / 2
                {
                    return Err(format!(
                        "unsupported peer partial producer or alias: op={} t={:?} i={:?}",
                        inst.op, inst.t, inst.i
                    ));
                }
                if nvidia && (si == fp32_slot || pi < decode) {
                    let (m, n, k) = (inst.i[0], inst.i[1], inst.i[2]);
                    if k % 8 != 0
                        || [
                            (inst.t[1], u64::from(m) * u64::from(k)),
                            (inst.t[2], u64::from(n) * u64::from(k)),
                        ]
                        .into_iter()
                        .any(|(id, elements)| {
                            prog.tensors
                                .get(id as usize)
                                .is_none_or(|t| t.bytes < elements * 2)
                        })
                    {
                        return Err(
                            "NVIDIA FP32 peer producer needs aligned BF16 operand extents".into(),
                        );
                    }
                }
            }
        }
        for inst in &prog.insts {
            if let Some(si) = slots.iter().position(|&(p, _)| inst.t[0] == p) {
                if produced[si]
                    .replace(u64::from(inst.i[0]) * u64::from(inst.i[1]))
                    .is_some()
                {
                    return Err("peer partial overwritten before reduction".into());
                }
            }
            if !matches!(
                DevOp::from_u16(inst.op),
                Some(DevOp::XReduce | DevOp::XReduceTwoShot)
            ) {
                continue;
            }
            let si = if inst.i[2] == 0 {
                0
            } else if u64::from(inst.i[2]) == old_bytes {
                1
            } else {
                return Err("unsupported peer slot".into());
            };
            let plain = inst.op == DevOp::XReduce as u16;
            if produced[si].take() != Some(u64::from(inst.i[0]))
                || inst.t[0] != slots[si].1
                || inst.t[1..].iter().any(|&t| t != TENSOR_NONE)
                || inst.i[0] == 0
                || u64::from(inst.i[0]) * 2 > old_bytes
                || inst.i[1] < 2
                || (nvidia && inst.i[1] != 2)
                || ranks.replace(inst.i[1]).is_some_and(|n| n != inst.i[1])
                || inst.i[if plain { 4 } else { 5 }..].iter().any(|&v| v != 0)
                || inst.f.iter().any(|v| v.to_bits() != 0)
                || inst.j != [0, 0]
            {
                return Err("unsupported peer collective geometry or fusion".into());
            }
            if si == fp32_slot || (nvidia && pi < decode) {
                marked += 1;
            }
        }
        if produced.iter().any(Option::is_some) {
            return Err("peer partial has no collective consumer".into());
        }
        if prog.insts.iter().any(|d| {
            matches!(
                DevOp::from_u16(d.op),
                Some(
                    DevOp::XReduceAddNorm
                        | DevOp::XReduceScatter
                        | DevOp::XAllGather
                        | DevOp::XFlashMerge
                        | DevOp::XArgmaxFin
                )
            )
        }) {
            return Err(
                "FP32 peer partials do not support fused or sequence-parallel collectives".into(),
            );
        }
    }
    if marked == 0 {
        return Err("no selected peer collectives".into());
    }
    for &(partial, _) in &slots {
        model.tensors[partial as usize].bytes = new_bytes;
    }
    for (pi, prog) in model.progs.iter_mut().enumerate() {
        let fp32_slot = if pi < decode { 0 } else { 1 };
        for &(partial, _) in &slots {
            prog.tensors[partial as usize].bytes = new_bytes;
        }
        for inst in &mut prog.insts {
            if matches!(
                DevOp::from_u16(inst.op),
                Some(DevOp::XReduce | DevOp::XReduceTwoShot)
            ) {
                let slot = if inst.i[2] == 0 { 0 } else { 1 };
                inst.i[2] *= 2;
                if slot == fp32_slot || (nvidia && pi < decode) {
                    inst.op = DevOp::XReduce as u16;
                    inst.i[4] = 0;
                    inst.f[0] = f32::from_bits(XREDUCE_PARTIAL_F32_MARK);
                }
            } else if slots[fp32_slot].0 == inst.t[0]
                || (nvidia && pi < decode && slots.iter().any(|&(partial, _)| partial == inst.t[0]))
            {
                inst.op = if nvidia && pi >= decode {
                    DevOp::GemvF32Out
                } else {
                    DevOp::GemmF32
                } as u16;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::Builder;

    fn single(rows: u32) -> Model {
        let mut b = Builder::new(2);
        b.force_uniseg();
        for name in [
            "act.og_tp",
            "act.og",
            "act.dg_tp",
            "act.dg",
            "input",
            "weight",
        ] {
            b.tensor(name, 128);
        }
        for (p, o, offset) in [(0, 1, 0), (2, 3, 128)] {
            let producer = b.emit(DevOp::Gemv, vec![0, 1], &[], |d| {
                d.t[..3].copy_from_slice(&[p, 4, 5]);
                d.i[..3].copy_from_slice(&[rows, 8, 8]);
            });
            b.emit(DevOp::XReduce, vec![0], &[producer], |d| {
                d.t[0] = o;
                d.i[..4].copy_from_slice(&[rows * 8, 2, offset, 0]);
            });
        }
        Model {
            n_cu: 2,
            target: 0,
            tensors: b.tensors(),
            progs: vec![b.finish()],
            prog_t: vec![rows],
            gen: vec![],
            kv_row_insts: vec![],
        }
    }
    fn model() -> Model {
        let mut m = single(1);
        m.progs.insert(0, single(4).progs.remove(0));
        m.prog_t.insert(0, 4);
        m
    }
    fn snapshot(m: &Model) -> String {
        format!(
            "{:?}{:?}",
            m.tensors,
            m.progs.iter().map(|p| &p.insts).collect::<Vec<_>>()
        )
    }
    #[test]
    fn tp_partial_f32_default_preserves_model() {
        let mut m = model();
        let before = snapshot(&m);
        apply(&mut m, false, "sm_120", [(0, 1), (2, 3)]).unwrap();
        assert_eq!(before, snapshot(&m));
    }
    #[test]
    fn tp_partial_f32_rewrites_storage_producers_and_collectives() {
        let mut m = model();
        apply(&mut m, true, "gfx1201", [(0, 1), (2, 3)]).unwrap();
        assert_eq!(
            m.tensors.iter().map(|t| t.bytes).collect::<Vec<_>>(),
            [256, 128, 256, 128, 128, 128]
        );
        for (i, p) in m.progs.iter().enumerate() {
            assert_eq!(p.tensors[0].bytes, 256);
            assert_eq!(
                p.insts[0].op,
                if i == 0 { DevOp::GemmF32 } else { DevOp::Gemv } as u16
            );
            assert_eq!(
                p.insts[2].op,
                if i == 0 { DevOp::Gemv } else { DevOp::GemmF32 } as u16
            );
            assert_eq!(
                p.insts[1].f[0].to_bits(),
                if i == 0 { XREDUCE_PARTIAL_F32_MARK } else { 0 }
            );
            assert_eq!(
                p.insts[3].f[0].to_bits(),
                if i == 0 { 0 } else { XREDUCE_PARTIAL_F32_MARK }
            );
            assert_eq!(p.insts[3].i[2], 256);
        }
    }
    #[test]
    fn nv_tp_partial_f32_uses_native_phase_producers() {
        let mut m = model();
        apply(&mut m, true, "sm_120a", [(0, 1), (2, 3)]).unwrap();
        assert_eq!(m.progs[0].insts[0].op, DevOp::GemmF32 as u16);
        assert_eq!(m.progs[0].insts[2].op, DevOp::GemmF32 as u16);
        assert_eq!(m.progs[0].insts[3].f[0].to_bits(), XREDUCE_PARTIAL_F32_MARK);
        assert_eq!(m.progs[1].insts[0].op, DevOp::Gemv as u16);
        assert_eq!(m.progs[1].insts[2].op, DevOp::GemvF32Out as u16);
        for p in &m.progs {
            assert_eq!(p.insts[3].i[2], 256);
        }
    }
    #[test]
    fn nv_tp_partial_f32_rejects_bad_operand_extents_and_rank_count_atomically() {
        for case in 0..7 {
            let mut m = model();
            match case {
                0 => m.progs[0].insts[0].i[2] = 7,
                1 => m.progs[0].tensors[4].bytes = 1,
                2 => m.progs[1].tensors[5].bytes = 1,
                3 => m.progs[0].insts[1].i[1] = 4,
                5 => m.progs[0].insts[2].i[2] = 7,
                6 => m.progs[0].insts[2].i[2] = 32,
                _ => {}
            }
            let before = snapshot(&m);
            assert!(apply(
                &mut m,
                true,
                if case == 4 { "sm_90a" } else { "sm_120a" },
                [(0, 1), (2, 3)]
            )
            .is_err());
            assert_eq!(before, snapshot(&m));
        }
    }

    #[test]
    fn tp_partial_f32_uses_one_round_prefill_and_moves_its_peer_slot() {
        let mut m = single(1);
        let mut prefill = single(4).progs.remove(0);
        for inst in &mut prefill.insts {
            if inst.op == DevOp::Gemv as u16 {
                inst.op = DevOp::GemmWide as u16;
            } else {
                inst.op = DevOp::XReduceTwoShot as u16;
                inst.i[4] = 1;
            }
        }
        m.progs.insert(0, prefill);
        m.prog_t.insert(0, 4);
        apply(&mut m, true, "gfx1201", [(0, 1), (2, 3)]).unwrap();
        assert_eq!(m.progs[0].insts[0].op, DevOp::GemmF32 as u16);
        assert_eq!(m.progs[0].insts[1].op, DevOp::XReduce as u16);
        assert_eq!(m.progs[0].insts[1].i[4], 0);
        assert_eq!(m.progs[0].insts[1].f[0].to_bits(), XREDUCE_PARTIAL_F32_MARK);
        assert_eq!(m.progs[0].insts[2].op, DevOp::GemmWide as u16);
        assert_eq!(m.progs[0].insts[3].op, DevOp::XReduceTwoShot as u16);
        assert_eq!(m.progs[0].insts[3].i[4], 1);
        assert_eq!(m.progs[0].insts[3].f[0].to_bits(), 0);
        assert_eq!(m.progs[0].insts[3].i[2], 256);
        assert_eq!(m.progs[1].insts[1].f[0].to_bits(), 0);
        assert_eq!(m.progs[1].insts[3].f[0].to_bits(), XREDUCE_PARTIAL_F32_MARK);
    }
    #[test]
    fn tp_partial_f32_rejects_unsupported_contracts_atomically() {
        for case in 0..10 {
            let mut m = model();
            match case {
                0 => m.progs[0].insts[0].op = DevOp::GemvFp8 as u16,
                1 => m.progs[0].insts[1].i[5] = 8,
                2 => m.progs[0].insts[0].t[3] = 2,
                3 => m.progs[0].insts[3].i[2] = 64,
                4 => m.progs[0].tensors[0].bytes = 64,
                5 => m.progs[0].insts[1].f[0] = 1.0,
                6 => {
                    for i in [0, 2] {
                        m.tensors[i].bytes = u32::MAX as u64 + 1;
                    }
                }
                8 => m.progs[0].insts[1].i[0] = 64,
                9 => m.progs[0].insts[0].i[..2].fill(u32::MAX),
                _ => {}
            }
            let before = snapshot(&m);
            assert!(apply(
                &mut m,
                true,
                if case == 7 { "gfx950" } else { "gfx1201" },
                [(0, 1), (2, 3)]
            )
            .is_err());
            assert_eq!(before, snapshot(&m));
        }
    }
}
