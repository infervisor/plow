use packet::dev::{DevOp, SE_FINE, SE_XCTR};
use packet::devbuild::{Model, SectionData, SECT_METADATA};

pub(crate) fn apply(model: &mut Model) -> SectionData {
    let index = model.progs.len() - 1;
    assert_eq!(model.prog_t[index], 1, "GEMV role requires M1 decode");
    assert!(model.prog_t[..index].iter().all(|&t| t > 1));
    let p = &mut model.progs[index];
    assert!(p.l2_domains == 0 && p.hier_base == 0 && p.gq_seg_ofs.len() == 2);
    assert!(p
        .stream
        .iter()
        .all(|e| e.seg == 0 && e.flags & (SE_FINE | SE_XCTR) == 0));
    assert!(p.gq_stream.windows(2).all(|w| w[0].inst <= w[1].inst));
    let mut roles = Vec::new();
    let mut segments = Vec::new();
    for d in &p.insts {
        let eligible = matches!(
            DevOp::from_u16(d.op),
            Some(DevOp::Gemv | DevOp::GemvQkv | DevOp::GemvGlu)
        ) && d.i[0] == 1
            && d.i[2] > 0
            && d.i[2] <= 32768
            && d.i[2] % 8 == 0
            && u32::from(d.blocks) == model.n_cu
            && (d.op != DevOp::Gemv as u16 || d.i[3] == 0);
        let role = if eligible { 3 } else { 0 };
        if roles.is_empty() || role != 0 || roles.last() != Some(&0) {
            roles.push(role);
        }
        segments.push(u16::try_from(roles.len() - 1).expect("too many decode segments"));
    }
    assert!(roles.contains(&3), "no eligible BF16 M1 GEMV instructions");
    for e in p.stream.iter_mut().chain(&mut p.gq_stream) {
        e.seg = segments[e.inst as usize];
    }
    p.gq_seg_ofs = vec![0];
    for (i, pair) in p.gq_stream.windows(2).enumerate() {
        if pair[0].seg != pair[1].seg {
            p.gq_seg_ofs.push((i + 1) as u32);
        }
    }
    p.gq_seg_ofs.push(p.gq_stream.len() as u32);
    assert_eq!(p.gq_seg_ofs.len(), roles.len() + 1);
    SectionData {
        kind: SECT_METADATA,
        name: "segment_roles.json".into(),
        data: serde_json::to_vec(&serde_json::json!({
            "version":1,
            "objects":{"3":{"abi":"gemv_sm90_cta512_v1","file":"interp_sm90a_gemv512.cubin"}},
            "programs":[{"index":index,"roles":roles}]
        }))
        .unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::Builder;

    fn model() -> Model {
        let mut b = Builder::new(132);
        b.force_uniseg();
        let a = b.emit(DevOp::Nop, b.all(), &[], |_| {});
        let g = b.emit(DevOp::Gemv, b.all(), &[a], |d| {
            d.i = [1, 4096, 4096, 0, 0, 0, 0, 0];
        });
        let q = b.emit(DevOp::GemvQkv, b.all(), &[g], |d| {
            d.i = [1, 4096, 4096, 256, 256, 0, 0, 0];
        });
        b.emit(DevOp::Nop, b.all(), &[q], |_| {});
        Model {
            n_cu: 132,
            target: 0,
            tensors: vec![],
            progs: vec![b.finish()],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        }
    }

    #[test]
    fn segmentation_preserves_instruction_counter_and_slice_identity() {
        let mut m = model();
        let old = model().progs.remove(0);
        let section = apply(&mut m);
        let p = &m.progs[0];
        assert_eq!(p.insts, old.insts);
        assert_eq!(p.waits, old.waits);
        assert_eq!(p.succs, old.succs);
        assert_eq!(p.n_counter, old.n_counter);
        assert_eq!(p.stream_ofs, old.stream_ofs);
        assert_eq!(p.stream_len, old.stream_len);
        for (a, b) in p
            .stream
            .iter()
            .zip(&old.stream)
            .chain(p.gq_stream.iter().zip(&old.gq_stream))
        {
            let mut a = *a;
            a.seg = 0;
            assert_eq!(a, *b);
        }
        let metadata: serde_json::Value = serde_json::from_slice(&section.data).unwrap();
        assert_eq!(
            metadata["programs"][0]["roles"],
            serde_json::json!([0, 3, 3, 0])
        );
        assert_eq!(p.gq_seg_ofs, [0, 132, 264, 396, 528]);
    }

    #[test]
    fn no_eligible_shape_is_rejected() {
        let mut m = model();
        for d in &mut m.progs[0].insts {
            d.i[2] = 32776;
        }
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| apply(&mut m))).is_err());
    }

    #[test]
    fn larger_or_multiple_decode_rungs_are_rejected() {
        let mut m = model();
        m.prog_t[0] = 2;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| apply(&mut m))).is_err());
        let mut m = model();
        m.progs.push(model().progs.remove(0));
        m.prog_t.push(1);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| apply(&mut m))).is_err());
    }
}
