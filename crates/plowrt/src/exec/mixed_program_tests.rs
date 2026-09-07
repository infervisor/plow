use super::*;

// TEST_MIXED_ASSETS points at an ordinary compiled model; this test uses no GPU.
#[test]
#[ignore = "requires an ordinary dense BF16 model asset"]
fn ordinary_asset_synthesis() {
    let path = std::path::PathBuf::from(std::env::var_os("TEST_MIXED_ASSETS").unwrap());
    let raw = std::fs::read(path.join("model.pkt")).unwrap();
    let mut blob = DevBlob::parse_l2(&raw, true).unwrap();
    let expected: BTreeSet<_> = blob
        .progs
        .iter()
        .filter(|p| {
            !p.packed_prefill_only && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16)
        })
        .map(|p| p.t)
        .collect();
    let kvlen = tensor(&blob, "in.kvlen").unwrap() as usize;
    let batch = (blob.tensors[kvlen].bytes / 4) as usize;
    {
        let mixed = synthesize(&blob, batch).unwrap();
        assert_eq!(
            mixed
                .programs
                .iter()
                .map(|p| p.program.rows)
                .collect::<BTreeSet<_>>(),
            expected
        );
        for spec in &mixed.programs {
            assert_eq!(
                spec.decode_rows,
                (batch as u32 - 1).min(spec.program.rows - 1)
            );
            assert_eq!(spec.program.gq_seg_ofs.len(), 2);
            assert!(spec
                .program
                .insts
                .iter()
                .all(|i| i.blocks as u32 == blob.n_cu));
            assert!(spec
                .program
                .insts
                .iter()
                .any(|i| i.op == DevOp::FlashDecode as u16));
            assert!(spec
                .program
                .insts
                .iter()
                .any(|i| i.op == DevOp::GemmGlu as u16));
        }
        for name in ["in.ids", "in.pos", "in.kvlen"] {
            assert!(mixed.tensors.iter().any(|t| t.name == name));
        }
        assert!(mixed
            .tensors
            .iter()
            .all(|t| t.name.starts_with("in.") || t.name.starts_with("act.")));
        eprintln!(
            "runtime synthesis batch={batch}: capacities={expected:?}, private tensors={}",
            mixed.tensors.len()
        );
    }
    assert!(synthesize(&blob, 1).is_err());
    assert!(synthesize(&blob, batch + 1).is_err());
    if batch > 2 {
        assert!(synthesize(&blob, batch - 1).is_err());
    }
    blob.tensors[kvlen].bytes += 4;
    assert!(synthesize(&blob, batch).is_err());
    blob.tensors[kvlen].bytes -= 4;
    let k = blob
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .find(|i| i.op == DevOp::FlashDecode as u16)
        .unwrap()
        .t[3] as usize;
    let size = blob.tensors[k].bytes;
    blob.tensors[k].bytes = 2;
    assert!(synthesize(&blob, batch).is_err());
    blob.tensors[k].bytes = size;
    let prefill = blob
        .progs
        .iter_mut()
        .find(|p| p.t > 1 && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16))
        .unwrap();
    let first = prefill.insts[0].op;
    prefill.insts[0].op = u16::MAX;
    assert!(synthesize(&blob, batch).is_err());
    let prefill = blob
        .progs
        .iter_mut()
        .find(|p| p.t > 1 && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16))
        .unwrap();
    prefill.insts[0].op = first;
    let flash = prefill
        .insts
        .iter_mut()
        .find(|i| i.op == DevOp::FlashPrefill as u16)
        .unwrap();
    flash.i[6] = 128;
    assert!(synthesize(&blob, batch).is_err());
}
