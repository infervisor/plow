use super::*;
use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};

fn fixture() -> (DevBlob, SegmentRoles) {
    let mut b = Builder::new(132);
    b.force_uniseg();
    let out = b.tensor("out", 4096 * 2);
    let x = b.tensor("x", 4096 * 2);
    let w = b.tensor("w", 4096 * 4096 * 2);
    let a = b.emit(DevOp::Nop, b.all(), &[], |_| {});
    let c = b.emit(DevOp::Gemv, b.all(), &[a], |d| {
        d.t[..3].copy_from_slice(&[out, x, w]);
        d.i = [1, 4096, 4096, 0, 0, 0, 0, 0];
    });
    b.isolate(c);
    b.emit(DevOp::Nop, b.all(), &[c], |_| {});
    let p = b.finish();
    let m = Model {
        n_cu: 132,
        target: 0,
        tensors: p.tensors.clone(),
        progs: vec![p],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen: vec![],
    };
    let section = SectionData {
        kind: SECT_METADATA,
        name: "segment_roles.json".into(),
        data: serde_json::to_vec(&serde_json::json!({"version":1,
            "objects":{"3":{"abi":"gemv_sm90_cta512_v1","file":"interp_sm90a_gemv512.cubin"}},
            "programs":[{"index":0,"roles":[0,3,0]}]}))
        .unwrap(),
    };
    let raw = m.to_blob_v6(&[section]);
    let blob = DevBlob::parse(&raw).unwrap();
    let roles = SegmentRoles::parse(
        blob.section_data_named(&raw, SECT_METADATA, "segment_roles.json")
            .unwrap(),
        &blob,
    )
    .unwrap();
    (blob, roles)
}

#[test]
fn gemv_role_rejects_unsafe_geometry_and_counter_windows() {
    let (blob, roles) = fixture();
    assert!(roles.validate(&blob.progs, &[], &blob.tensors).is_ok());
    let mutations: &[fn(&mut DevBlob)] = &[
        |b| b.progs[0].t = 2,
        |b| b.progs[0].insts[1].i[0] = 2,
        |b| b.progs[0].insts[1].i[2] = 32776,
        |b| b.progs[0].insts[1].i[2] = 4097,
        |b| b.progs[0].insts[1].i[3] = 1,
        |b| b.progs[0].insts[1].i[4] = 1,
        |b| b.progs[0].insts[1].op = DevOp::GemvFp8 as u16,
        |b| b.tensors[2].bytes -= 2,
        |b| b.progs[0].gq_seg_ofs[1] += 1,
        |b| b.progs[0].l2_domains = 1,
        |b| {
            let g = &mut b.progs[0];
            for e in g.stream.iter_mut().chain(&mut g.gq_stream) {
                e.flags |= packet::dev::SE_XCTR;
            }
        },
        |b| {
            let g = &mut b.progs[0];
            for e in g.stream.iter_mut().chain(&mut g.gq_stream) {
                e.flags |= packet::dev::SE_FINE;
            }
        },
    ];
    for (i, mutation) in mutations.iter().enumerate() {
        let (mut b, roles) = fixture();
        mutation(&mut b);
        assert!(
            roles.validate(&b.progs, &[], &b.tensors).is_err(),
            "mutation {i}"
        );
    }
    let (mut b, mut roles) = fixture();
    let g = &mut b.progs[0];
    g.gq_stream[..264].rotate_left(132);
    for e in g.stream.iter_mut().chain(&mut g.gq_stream) {
        if e.seg < 2 {
            e.seg = 1 - e.seg;
        }
    }
    roles.programs[0].roles.swap(0, 1);
    assert!(validate_segment_windows(g).is_ok());
    assert!(roles.validate(&b.progs, &[], &b.tensors).is_err());
}

#[test]
fn gemv_role_rejects_larger_rungs_and_prefill_routing() {
    let (mut b, mut roles) = fixture();
    assert!(roles.validate(&b.progs, &[0], &b.tensors).is_err());
    b.progs.push(fixture().0.progs.remove(0));
    roles.programs[0].index = 1;
    assert!(roles.validate(&b.progs, &[], &b.tensors).is_err());
}

#[test]
#[ignore = "CPU actual pair gate; set TEST_GEMV_ROLE_ASSETS and TEST_GEMV_ROLE_BASELINE"]
fn gemv_role_actual_packet_preserves_work() {
    let load = |key| {
        let dir = std::path::PathBuf::from(std::env::var(key).unwrap());
        let raw = std::fs::read(DevBlob::find_in_dir(&dir).unwrap().unwrap()).unwrap();
        (DevBlob::parse(&raw).unwrap(), raw)
    };
    let (base, _) = load("TEST_GEMV_ROLE_BASELINE");
    let (candidate, raw) = load("TEST_GEMV_ROLE_ASSETS");
    let roles = SegmentRoles::parse(
        candidate
            .section_data_named(&raw, SECT_METADATA, "segment_roles.json")
            .unwrap(),
        &candidate,
    )
    .unwrap();
    assert_eq!(candidate.decode_rungs(), [1]);
    assert_eq!(base.decode_rungs(), [1]);
    assert!(base
        .tensors
        .iter()
        .map(|t| (&t.name, t.bytes))
        .eq(candidate.tensors.iter().map(|t| (&t.name, t.bytes))));
    for (a, b) in base.progs.iter().zip(&candidate.progs) {
        assert_eq!(a.insts, b.insts);
        assert_eq!(a.n_counter, b.n_counter);
        assert_eq!(a.waits, b.waits);
        assert_eq!(a.succs, b.succs);
        assert_eq!(a.stream_ofs, b.stream_ofs);
        assert_eq!(a.stream_len, b.stream_len);
        assert_eq!(a.stream.len(), b.stream.len());
        assert_eq!(a.gq_stream.len(), b.gq_stream.len());
        for (e, f) in a
            .stream
            .iter()
            .zip(&b.stream)
            .chain(a.gq_stream.iter().zip(&b.gq_stream))
        {
            let mut f = *f;
            if a.t == 1 {
                f.seg = 0;
            }
            assert_eq!(*e, f);
        }
        if a.t != 1 {
            assert_eq!(a.gq_seg_ofs, b.gq_seg_ofs);
        }
    }
    let r = roles.program(candidate.progs.len() - 1).unwrap();
    eprintln!("validated {} segments, {} GEMV512 instructions; all instructions/counters/slices and prefill unchanged",r.roles.len(),r.roles.iter().filter(|&&r|r==3).count());
}

#[test]
#[ignore = "GPU full-logit and graph gate; set TEST_GEMV_ROLE_GPU=1 plus pair paths"]
fn gpu_gemv_role_graph_matches_coarse_full_logits() {
    assert_eq!(std::env::var("TEST_GEMV_ROLE_GPU").as_deref(), Ok("1"));
    gemv_role_actual_packet_preserves_work();
    let baseline = std::path::PathBuf::from(std::env::var("TEST_GEMV_ROLE_BASELINE").unwrap());
    let candidate = std::path::PathBuf::from(std::env::var("TEST_GEMV_ROLE_ASSETS").unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut references: Vec<(u32, Vec<u32>)> = Vec::new();
    for (is_role, assets) in [(false, baseline), (true, candidate)] {
        let mut e = GpuEngine::load(Arc::clone(&be), &assets, &assets.join("checkpoint")).unwrap();
        assert_eq!(e.batch, 1);
        assert!(e.multistep.is_none() && e.lt_decode.is_empty() && e.decode_rungs.is_empty());
        assert_eq!(e.lt_decode_graph.is_some(), is_role);
        assert_eq!(!e.decode_packet_roles.is_empty(), is_role);
        assert!(!e.prefill.is_empty(), "prefill required");
        let prefixes: Vec<usize> = std::env::var("TEST_GEMV_ROLE_PREFIXES")
            .unwrap_or_else(|_| "1024,8193,32768".into())
            .split(',')
            .map(|n| n.parse().unwrap())
            .collect();
        assert!(!prefixes.is_empty());
        let mut checked = 0;
        let mut compare = |e: &mut GpuEngine, token: u32| {
            let mut logits = Vec::new();
            e.logits_row(0, &mut logits).unwrap();
            assert!(!logits.is_empty() && logits.iter().all(|v| v.is_finite()));
            let bits: Vec<_> = logits.iter().map(|v| v.to_bits()).collect();
            if is_role {
                assert_eq!(
                    (token, &bits),
                    (references[checked].0, &references[checked].1),
                    "snapshot {checked}"
                );
            } else {
                references.push((token, bits));
            }
            checked += 1;
        };
        let mut decoded = Vec::new();
        for (phase, rows) in prefixes.into_iter().enumerate() {
            let prompt: Vec<_> = (0..rows)
                .map(|i| 100 + ((i * 13 + phase * 79) % 1000) as u32)
                .collect();
            e.begin_slot(0, prompt.len() + 64).unwrap();
            let mut token = e.prefill_slot(0, &prompt).unwrap();
            compare(&mut e, token);
            let before = e.pos.clone();
            assert!(e.step_slots(&[(1, token)], &mut decoded).is_err());
            assert_eq!(e.pos, before, "invalid sparse slot changed sequence state");
            for _ in 0..8 {
                e.step_slots(&[(0, token)], &mut decoded).unwrap();
                token = decoded[0];
                compare(&mut e, token);
            }
            token = e.consume_prompt(0, &[341, 617, 829], &mut decoded).unwrap();
            compare(&mut e, token);
            e.step_slots(&[(0, token)], &mut decoded).unwrap();
            compare(&mut e, decoded[0]);
        }
        assert_eq!(e.lt_decode_graph.is_some(), is_role);
        eprintln!("role3={is_role}: {checked} full-logit snapshots; repeated graph replay, reset, padded prefill, continuation and sparse-slot rejection passed");
    }
}
