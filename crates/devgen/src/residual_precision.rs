use packet::dev::{DevOp, RESIDUAL_F32_FLAG, TENSOR_NONE};
use packet::devbuild::Model;

pub(crate) fn apply(model: &mut Model, enabled: bool, arch: &str) -> Result<(), String> {
    if !enabled {
        return Ok(());
    }
    if arch != "gfx1201" {
        return Err(format!("FP32 residual objects are unavailable for {arch}"));
    }
    let state = model
        .tensors
        .iter()
        .position(|t| t.name == "act.x")
        .ok_or("FP32 residual packet has no act.x")?;
    let tensor = &model.tensors[state];
    if tensor.bytes == 0 || tensor.bytes % 2 != 0 || tensor.init.is_some() {
        return Err("FP32 residual needs uninitialized BF16 state".into());
    }
    if model.tensors.iter().any(|t| t.name.starts_with("act.x@")) {
        return Err("FP32 residual does not support residual views".into());
    }
    let bytes = tensor
        .bytes
        .checked_mul(2)
        .ok_or("residual byte overflow")?;
    let state_id = state as u32;
    let mut embedded = false;
    for prog in &model.progs {
        let declared = prog.tensors.get(state).ok_or("missing program residual")?;
        if declared.name != tensor.name || declared.bytes != tensor.bytes {
            return Err("program residual layout differs from the model".into());
        }
        for inst in &prog.insts {
            if !inst.t.contains(&state_id) {
                continue;
            }
            let allowed: &[usize] = match DevOp::from_u16(inst.op) {
                Some(DevOp::Embed) if inst.t[0] == state_id => {
                    embedded = true;
                    &[0]
                }
                Some(DevOp::RmsNorm)
                    if inst.t[1] == state_id
                        && inst.t[3..].iter().all(|&t| t == TENSOR_NONE)
                        && inst.i[2..7].iter().all(|&v| v == 0) =>
                {
                    &[1]
                }
                Some(DevOp::NormResidual) if inst.t[0] == state_id && inst.t[1] == state_id => {
                    &[0, 1]
                }
                Some(DevOp::NormResidualNorm) if inst.t[1] == state_id && inst.t[2] == state_id => {
                    &[1, 2]
                }
                _ => {
                    return Err(format!(
                        "unsupported FP32 residual use in opcode {}",
                        inst.op
                    ))
                }
            };
            if inst.i[6] != 0
                || inst.i[7] != 0
                || inst
                    .t
                    .iter()
                    .enumerate()
                    .any(|(slot, &t)| t == state_id && !allowed.contains(&slot))
            {
                return Err("FP32 residual has incompatible flags or aliases".into());
            }
        }
    }
    if !embedded {
        return Err("FP32 residual needs a token embedding initializer".into());
    }
    model.tensors[state].bytes = bytes;
    for prog in &mut model.progs {
        prog.tensors[state].bytes = bytes;
        for inst in &mut prog.insts {
            if inst.t.contains(&state_id) {
                inst.i[6] = RESIDUAL_F32_FLAG;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::Builder;

    fn model() -> Model {
        let mut b = Builder::new(2);
        b.force_uniseg();
        let x = b.tensor("act.x", 16);
        let hn = b.tensor("act.hn", 16);
        let weight = b.tensor("model.norm.weight", 16);
        let ids = b.tensor("in.ids", 4);
        let branch = b.tensor("act.branch", 16);
        for (op, tensors) in [
            (DevOp::Embed, vec![x, weight, ids]),
            (DevOp::RmsNorm, vec![hn, x, weight]),
            (DevOp::NormResidual, vec![x, x, branch, weight]),
            (
                DevOp::NormResidualNorm,
                vec![hn, x, x, branch, weight, weight],
            ),
        ] {
            b.emit(op, vec![0, 1], &[], |d| {
                d.t[..tensors.len()].copy_from_slice(&tensors);
                d.i[..2].copy_from_slice(&[1, 8]);
            });
        }
        let tensors = b.tensors();
        Model {
            n_cu: 2,
            target: 0,
            tensors,
            progs: vec![b.finish()],
            prog_t: vec![1],
            gen: vec![],
            kv_row_insts: vec![],
        }
    }

    fn snapshot(m: &Model) -> String {
        format!("{:?}{:?}", m.tensors, m.progs[0].insts)
    }

    #[test]
    fn residual_f32_default_preserves_model() {
        let mut m = model();
        let before = snapshot(&m);
        apply(&mut m, false, "sm_120").unwrap();
        assert_eq!(snapshot(&m), before);
    }

    #[test]
    fn residual_f32_marks_only_state_storage() {
        let mut m = model();
        apply(&mut m, true, "gfx1201").unwrap();
        assert_eq!(
            m.tensors.iter().map(|t| t.bytes).collect::<Vec<_>>(),
            [32, 16, 16, 4, 16]
        );
        assert_eq!(m.progs[0].tensors[0].bytes, 32);
        assert!(m.progs[0].insts.iter().all(|d| d.i[6] == RESIDUAL_F32_FLAG));
    }

    #[test]
    fn residual_f32_rejects_incompatible_contracts_atomically() {
        for mutation in 0..7 {
            let mut m = model();
            match mutation {
                0 => m.progs[0].insts[3].op = DevOp::Gemv as u16,
                1 => m.progs[0].insts[1].i[7] = 1,
                2 => m.progs[0].insts[1].t[3] = 4,
                3 => m.progs[0].insts[1].t[0] = 0,
                4 => {
                    m.tensors[0].bytes = u64::MAX - 1;
                    m.progs[0].tensors[0].bytes = u64::MAX - 1;
                }
                5 => m.progs[0].insts[1].i[6] = 4,
                _ => {}
            }
            let before = snapshot(&m);
            assert!(apply(
                &mut m,
                true,
                if mutation == 6 { "gfx942" } else { "gfx1201" }
            )
            .is_err());
            assert_eq!(snapshot(&m), before);
        }
    }
}
