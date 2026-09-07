use super::*;
use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};

fn fixture() -> (DevBlob, SegmentRoles) {
    const E: u64 = 32;
    const H: u64 = 2880;
    const I: u64 = 2880;
    let mut b = Builder::new(132);
    b.force_uniseg();
    let x = b.tensor("x", H * 2);
    let table = b.tensor("table", 4 * 8);
    let fu = b.tensor("fu", 4 * I * 2);
    let glu_w = b.tensor("glu_w", E * 2 * I * H.div_ceil(2));
    let glu_s = b.tensor("glu_s", E * 2 * I * H.div_ceil(32));
    let part = b.tensor("part", 4 * H * 4);
    let down_w = b.tensor("down_w", E * H * I.div_ceil(2));
    let down_s = b.tensor("down_s", E * H * I.div_ceil(32));
    let before = b.emit(DevOp::Nop, b.all(), &[], |_| {});
    let glu = b.emit(DevOp::MoeGluMx, b.all(), &[before], |d| {
        d.t[..5].copy_from_slice(&[fu, x, table, glu_w, glu_s]);
        d.i = [4, I as u32, H as u32, E as u32, 0, 3, 1, 0];
    });
    b.isolate(glu);
    let down = b.emit(DevOp::MoeDownMx, b.all(), &[glu], |d| {
        d.t[..5].copy_from_slice(&[part, fu, table, down_w, down_s]);
        d.i = [4, H as u32, I as u32, E as u32, 0, 0, 1, 0];
    });
    b.isolate(down);
    b.emit(DevOp::Nop, b.all(), &[down], |_| {});
    let program = b.finish();
    let model = Model {
        n_cu: 132,
        target: 0,
        tensors: program.tensors.clone(),
        progs: vec![program],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen: vec![],
    };
    let section = SectionData {
        kind: SECT_METADATA,
        name: plow_asset::segment_roles::SECTION.into(),
        data: serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "objects": {"7": {
                "abi": plow_asset::segment_roles::MXFP4_MOE_ABI,
                "file": "moe.cubin",
                "sha256": "a".repeat(64)
            }},
            "programs": [{"index": 0, "roles": [0, 7, 7, 0]}]
        }))
        .unwrap(),
    };
    let raw = model.to_blob_v6(&[section]);
    let blob = DevBlob::parse(&raw).unwrap();
    let roles = SegmentRoles::parse(
        blob.section_data_named(&raw, SECT_METADATA, plow_asset::segment_roles::SECTION)
            .unwrap(),
        &blob,
    )
    .unwrap();
    (blob, roles)
}

#[test]
fn mxfp4_moe_role_accepts_only_complete_m1_projection_segments() {
    let (blob, roles) = fixture();
    assert!(roles.validate(&blob.progs, &[], &blob.tensors).is_ok());
    let mutations: &[fn(&mut DevBlob)] = &[
        |b| b.progs[0].t = 2,
        |b| b.progs[0].insts[1].i[6] = 2,
        |b| b.progs[0].insts[1].i[5] = 4,
        |b| b.progs[0].insts[2].op = DevOp::MoeCombine as u16,
        |b| b.tensors[2].bytes -= 2,
        |b| b.progs[0].gq_seg_ofs[2] += 1,
        |b| {
            let program = &mut b.progs[0];
            for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
                if entry.seg == 1 {
                    entry.flags |= packet::dev::SE_FINE;
                }
            }
        },
    ];
    for (index, mutation) in mutations.iter().enumerate() {
        let (mut blob, roles) = fixture();
        mutation(&mut blob);
        assert!(
            roles.validate(&blob.progs, &[], &blob.tensors).is_err(),
            "mutation {index}"
        );
    }
}
