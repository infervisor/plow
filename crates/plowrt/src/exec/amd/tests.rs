use super::object::*;
use super::*;
use crate::exec::kvrow::KDA_ROW_COUNT_OPS;
use packet::dev::PREFILL_SPAN_RESET_STATE;

#[test]
fn materialized_mla_flat_grid_covers_every_qblock_head_and_batch_once() {
    for &(n, heads, batches) in &[
        (1u32, 1u32, 1u32),
        (256, 12, 1),
        (257, 12, 1),
        (1025, 12, 1),
        (8192, 12, 1),
        (1025, 12, 3),
    ] {
        let q_grid = n.div_ceil(MLA_MATERIALIZED_Q_BLOCK);
        let grid = mla_materialized_flat_grid(n, heads, batches).unwrap();
        let mut seen = vec![0u8; grid as usize];
        for flat_id in 0..grid {
            let q_block = flat_id % q_grid;
            let rest = flat_id / q_grid;
            let head = rest % heads;
            let batch = rest / heads;
            assert!(q_block < q_grid && head < heads && batch < batches);
            let logical = ((batch * heads + head) * q_grid + q_block) as usize;
            seen[logical] += 1;
        }
        assert!(seen.into_iter().all(|n| n == 1), "T={n}");
    }
}

fn materialized_mla_route_probe(t: u32, heads: u32) -> (DevProg, Vec<DeviceMem>) {
    let pack = DevInst64 {
        op: DevOp::MlaMaterializePack as u16,
        blocks: 64,
        t: [0, 1, 2, 3, 0, 0, 0, 0],
        i: [t, heads, 128, 64, 128, 0, 0, 0],
        ..Default::default()
    };
    let attention = DevInst64 {
        op: DevOp::FlashMlaMaterializedPrefill as u16,
        blocks: 1,
        t: [0, 1, 2, 3, 0, 0, 0, 0],
        i: [t, heads, heads, 192, 128, 1, 0, 0],
        fj: [0.07216878f32.to_bits(), 0, 0],
        ..Default::default()
    };
    let stream = vec![
        packet::dev::StreamEnt {
            inst: 0,
            seg: 0,
            ..Default::default()
        },
        packet::dev::StreamEnt {
            inst: 1,
            seg: 1,
            ..Default::default()
        },
    ];
    let prog = DevProg {
        t,
        packed_prefill_only: false,
        token_batch_body: false,
        n_counter: 0,
        insts: vec![pack, attention],
        stream,
        stream_ofs: vec![0],
        stream_len: vec![2],
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: Vec::new(),
        gq_seg_ofs: Vec::new(),
        l2_domains: 0,
    };
    // K/V transients sized at a 16384-row cache capacity; `K_rope` is the krot cache.
    let rows = 16384 * u64::from(heads) * 2;
    let devp = [rows * 192, rows * 128, rows * 256, 16384 * 64 * 2]
        .iter()
        .enumerate()
        .map(|(i, &len)| DeviceMem::view(0x1000 + (i as u64) * 0x1000_0000, len))
        .collect();
    (prog, devp)
}

#[test]
fn materialized_mla_routes_follow_the_chunk() {
    let (mut prog, devp) = materialized_mla_route_probe(256, 12);
    // The kv_b projection over the latent cache, `M = T` at emit.
    prog.insts.push(DevInst64 {
        op: DevOp::Gemm as u16,
        blocks: 256,
        t: [4, 5, 6, 0, 0, 0, 0, 0],
        i: [256, 12 * 256, 512, 0, 0, 0, 0, 0],
        ..Default::default()
    });
    let names: Vec<String> = [
        "act.pf.k_materialized",
        "act.pf.v_materialized",
        "act.pf.kv_materialized",
        "kv.l3.krot",
        "act.pf.kv_materialized",
        "kv.l3.ckv",
        "l3.kv_b",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    mla_materialized_routes(&prog, &devp, &mut routes).unwrap();
    assert!(matches!(
        routes[0],
        PrefillSegmentRoute::MlaMaterializePack {
            k_rope_ten: 3,
            kv_rows_cap: 16384,
            ..
        }
    ));

    // Continuation chunk: 8192 cached rows, 208 real of a 256 bucket under RAGGED-M.
    let mut insts = prog.insts.clone();
    rebase_mla_materialized_routes(&mut insts, &names, &mut routes, 208, 8400).unwrap();
    assert_eq!(insts[2].i[0], 8400, "projection spans every cached row");
    assert_eq!(
        insts[0].i[0], 256,
        "raw instruction words are the route's, not patched"
    );
    let PrefillSegmentRoute::MlaMaterializePack { args, .. } = routes[0] else {
        panic!()
    };
    assert_eq!(args.t, 8400);
    let PrefillSegmentRoute::MlaMaterializedPrefill { args, grid } = routes[1] else {
        panic!()
    };
    assert_eq!((args.n, args.n_kv), (208, 8400));
    assert_eq!(grid, 12);
    assert_eq!(args.stride_k_b, 8400 * 12 * 192);
    assert_eq!(args.stride_q_b, 208 * 12 * 192);

    // Back to an exact initial chunk: every patch is re-derived, none accumulates.
    rebase_mla_materialized_routes(&mut insts, &names, &mut routes, 256, 256).unwrap();
    let PrefillSegmentRoute::MlaMaterializedPrefill { args, grid } = routes[1] else {
        panic!()
    };
    assert_eq!((args.n, args.n_kv, grid), (256, 256, 12));

    // Past the transient capacity: refuse rather than overrun.
    assert!(
        rebase_mla_materialized_routes(&mut insts, &names, &mut routes, 256, 16385)
            .unwrap_err()
            .to_string()
            .contains("hold 16384")
    );
}

#[test]
fn materialized_mla_raw_routes_use_exact_abi_and_flat_grid() {
    let (prog, devp) = materialized_mla_route_probe(1025, 12);
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    mla_materialized_routes(&prog, &devp, &mut routes).unwrap();
    assert!(matches!(
        routes[0],
        PrefillSegmentRoute::MlaMaterializePack { grid: 64, .. }
    ));
    let PrefillSegmentRoute::MlaMaterializedPrefill { args, grid } = routes[1] else {
        panic!("attention did not take the standalone route")
    };
    assert_eq!(grid, 5 * 12);
    assert_eq!((args.b, args.n, args.h, args.h_kv), (1, 1025, 12, 12));
    assert_eq!((args.d_qk, args.d_v), (192, 128));
    assert_eq!((args.stride_q_n, args.stride_k_n), (12 * 192, 12 * 192));
    assert_eq!((args.stride_o_n, args.stride_v_n), (12 * 128, 12 * 128));
}

#[test]
fn materialized_mla_raw_route_rejects_mixed_or_counter_segments() {
    let (mut prog, devp) = materialized_mla_route_probe(1025, 12);
    prog.insts.push(DevInst64::default());
    prog.stream.push(packet::dev::StreamEnt {
        inst: 2,
        seg: 0,
        ..Default::default()
    });
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    assert!(mla_materialized_routes(&prog, &devp, &mut routes)
        .unwrap_err()
        .to_string()
        .contains("mixes"));

    let (mut prog, devp) = materialized_mla_route_probe(1025, 12);
    prog.stream[1].wait_len = 1;
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    assert!(mla_materialized_routes(&prog, &devp, &mut routes)
        .unwrap_err()
        .to_string()
        .contains("counter obligations"));
}

#[test]
fn grouped_moe_decode_pair_routes_with_exact_mxfp4_abi() {
    let glu = DevInst64 {
        op: DevOp::MoeGroupGluFp8Blk as u16,
        blocks: 256,
        t: [0, 1, 2, 3, 4, 0, 0, 0],
        i: [16, 384, 3584, 896, 0, 2, 2, 0],
        fj: [4.0f32.to_bits(), 25.0f32.to_bits(), 0],
        ..Default::default()
    };
    let down = DevInst64 {
        op: DevOp::MoeGroupDownFp8Blk as u16,
        blocks: 256,
        t: [5, 0, 2, 3, 4, 0, 0, 0],
        i: [16, 3584, 384, 896, 0, 0, 2, 0],
        ..Default::default()
    };
    let prog = DevProg {
        t: 1,
        packed_prefill_only: false,
        token_batch_body: false,
        n_counter: 0,
        insts: vec![glu, down],
        stream: vec![
            packet::dev::StreamEnt {
                inst: 0,
                ..Default::default()
            },
            packet::dev::StreamEnt {
                inst: 1,
                ..Default::default()
            },
        ],
        stream_ofs: vec![0],
        stream_len: vec![2],
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: Vec::new(),
        gq_seg_ofs: Vec::new(),
        l2_domains: 0,
    };
    assert_eq!(
        decode_segment_kinds(&prog).unwrap(),
        vec![DecodeSegmentKind::GroupedMoeMxfp4 { glu: 0, down: 1 }]
    );
    let devp: Vec<_> = (0..6)
        .map(|i| DeviceMem::view(0x1000 + i * 0x1000, 0x1000))
        .collect();
    let DecodeSegmentRoute::GroupedMoeMxfp4 { glu, down, grid } =
        decode_segment_routes(&prog, &[], &[], &devp).unwrap()[0]
    else {
        panic!("grouped MoE pair did not take standalone route")
    };
    assert_eq!(grid, 768);
    assert_eq!(
        (glu.topk, glu.intermediate, glu.hidden, glu.enc),
        (16, 384, 3584, 2)
    );
    assert_eq!(
        (down.topk, down.hidden, down.intermediate, down.enc),
        (16, 3584, 384, 2)
    );
    assert_eq!((glu.beta, glu.linear_beta), (4.0, 25.0));
}

#[test]
fn specialised_amd_object_pairing_is_fail_closed() {
    let object = Path::new("interp_decode.elf");
    assert!(validate_packet_pairing_stamp(None, None, None, object).is_ok());
    assert!(validate_packet_pairing_stamp(Some(1), None, Some(1), object).is_err());
    assert!(validate_packet_pairing_stamp(Some(1), Some(2), None, object).is_err());
    assert!(
        validate_packet_pairing_stamp(Some(1), Some(2), Some(0x0000_0002_0000_0001), object)
            .is_ok()
    );
    let err = validate_packet_pairing_stamp(Some(1), Some(2), Some(0x0000_0003_0000_0001), object)
        .unwrap_err()
        .to_string();
    assert!(err.contains("packet/object MISMATCH"));
}

#[test]
fn f32mix_attn_res_route_requires_marker_geometry_and_a_pure_segment() {
    let mut prog = segmented_prog(
        &[DevOp::RmsNorm, DevOp::AttnRes, DevOp::RmsNorm],
        &[0, 1, 2],
    );
    prog.t = 8192;
    let d = &mut prog.insts[1];
    d.blocks = 256;
    d.t = [0, 1, 2, 3, packet::dev::TENSOR_NONE16, 4, 5, 6];
    d.i = [8192, 7168, 4, 4, 8, packet::dev::TENSOR_NONE_I, 0, 0];
    d.fj = [1e-5f32.to_bits(), 1e-5f32.to_bits(), 0];
    let tensors: Vec<_> = (0..7)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..7)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();

    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::AttnResF32Mix { args, grid } = routes[1] else {
        panic!("a marked, pure f32-mix AttnRes packet must select the object")
    };
    assert_eq!(
        (args.t, args.hid, args.nb, args.nbcap, grid),
        (8192, 7168, 4, 8, 768)
    );
    assert_eq!(
        (args.out, args.gamma, args.res_a, args.res_b),
        (0x1000, 0x1400, 0x1500, 0x1600)
    );
    assert_eq!((args.push_src, args.res_pre, args.reserved), (0, 0, 0));
    assert_eq!(args.out_eps, 1e-5);

    // The interpreter's contract (no output-norm epsilon) stays on the interpreter.
    prog.insts[1].fj[1] = 0;
    let fallback = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(fallback[1], PrefillSegmentRoute::Interpreter));
    prog.insts[1].fj[1] = 1e-5f32.to_bits();

    // Decode width and a foreign hidden size are outside the object's contract.
    prog.insts[1].i[0] = 1;
    assert!(matches!(
        moe_mxfp4_routes(&prog, &tensors, &devp).unwrap()[1],
        PrefillSegmentRoute::Interpreter
    ));
    prog.insts[1].i[0] = 8192;
    prog.insts[1].i[1] = 4096;
    assert!(matches!(
        moe_mxfp4_routes(&prog, &tensors, &devp).unwrap()[1],
        PrefillSegmentRoute::Interpreter
    ));
    prog.insts[1].i[1] = 7168;

    // A marked packet sharing a segment with other work is refused, never silently run.
    prog.stream[2].seg = 1;
    let err = moe_mxfp4_routes(&prog, &tensors, &devp)
        .expect_err("mixed f32-mix segment must fail closed")
        .to_string();
    assert!(err.contains("f32-mix AttnRes packet in a mixed segment"));
}

#[test]
fn f32mix_attn_res_object_gates_markers_and_pairing() {
    let path = Path::new("attn_res_f32mix_gfx950.elf");
    let mut syms = ATTN_RES_F32MIX_MARKERS.to_vec();
    syms.pop();
    let err = check_attn_res_f32mix_symbols(&syms, path)
        .expect_err("a stale resource contract must be rejected")
        .to_string();
    assert!(err.contains("lacks required ABI/resource marker"));
    let err = check_attn_res_f32mix_symbols(&ATTN_RES_F32MIX_MARKERS, path)
        .expect_err("an unstamped object must be rejected")
        .to_string();
    assert!(err.contains("no packet-pairing stamp"));
    let missing = std::env::temp_dir().join(format!(
        "plow-attn-res-f32mix-missing-{}.elf",
        std::process::id()
    ));
    let err = read_attn_res_f32mix_object(&missing)
        .expect_err("a marked packet must not silently use the interpreter")
        .to_string();
    assert!(err.contains("f32-mix AttnRes packets require"));
}

#[test]
fn marked_kda_wave_items_requires_its_object() {
    let path = std::env::temp_dir().join(format!(
        "plow-kda-wave-items-missing-{}-{}.elf",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let err = read_kda_intra_wave_items_object(&path)
        .expect_err("a marked packet must not silently use the interpreter")
        .to_string();
    assert!(err.contains("marked KDA-intra wave-item segments require"));
}

#[test]
fn marked_kda_wave_items_rejects_missing_markers_and_pairing() {
    let path = Path::new("kda_chunk_intra_wave_items_gfx950.elf");
    let mut syms = KDA_INTRA_WAVE_ITEMS_MARKERS.to_vec();
    syms.pop();
    let err = check_kda_intra_wave_items_symbols(&syms, path)
        .expect_err("a stale resource contract must be rejected")
        .to_string();
    assert!(err.contains("lacks required ABI/resource marker"));

    let err = check_kda_intra_wave_items_symbols(&KDA_INTRA_WAVE_ITEMS_MARKERS, path)
        .expect_err("an unstamped object must be rejected")
        .to_string();
    assert!(err.contains("no packet-pairing stamp"));
}

#[test]
fn marked_kda_carry_regstate_requires_its_object() {
    let path = std::env::temp_dir().join(format!(
        "plow-kda-carry-regstate-missing-{}-{}.elf",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let err = read_kda_carry_regstate_object(&path)
        .expect_err("a marked packet must not silently use the interpreter")
        .to_string();
    assert!(err.contains("marked KDA carry regstate segments require"));

    let path = Path::new("kda_chunk_carry_regstate_gfx950.elf");
    let mut syms = KDA_CARRY_REGSTATE_MARKERS.to_vec();
    syms.pop();
    let err = check_kda_carry_regstate_symbols(&syms, path)
        .expect_err("a stale resource contract must be rejected")
        .to_string();
    assert!(err.contains("lacks required ABI/resource marker"));
    let err = check_kda_carry_regstate_symbols(&KDA_CARRY_REGSTATE_MARKERS, path)
        .expect_err("an unstamped object must be rejected")
        .to_string();
    assert!(err.contains("no packet-pairing stamp"));
}

#[test]
fn expert_parallel_loader_requires_every_abi_and_resource_marker() {
    let required = [
        "plow_moe_ep_filter_align_abi_1",
        "plow_moe_ep_filter_align_wave64_1",
        "plow_moe_ep_filter_align_stable_1",
        "plow_moe_ep_filter_align_no_spill_1",
    ];
    let path = Path::new("moe_ep_align_gfx950.elf");
    assert!(check_moe_ep_symbols(&required, path, &required).is_ok());
    let mut stale = required.to_vec();
    stale.pop();
    let err = check_moe_ep_symbols(&stale, path, &required)
        .expect_err("a stale specialist object must fail closed")
        .to_string();
    assert!(err.contains("plow_moe_ep_filter_align_no_spill_1"));
    assert_eq!(std::mem::size_of::<MoeEpAlignArgs>(), 80);
    assert_eq!(std::mem::size_of::<MoeEpCombineArgs>(), 48);
}

#[test]
fn pairing_stamp_is_read_from_elf_data() {
    let mut elf = vec![0u8; 0x304];
    elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
    elf[0x28..0x30].copy_from_slice(&0x100u64.to_le_bytes());
    elf[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
    elf[0x3c..0x3e].copy_from_slice(&4u16.to_le_bytes());

    let symtab = 0x100 + 64;
    elf[symtab + 4..symtab + 8].copy_from_slice(&2u32.to_le_bytes());
    elf[symtab + 24..symtab + 32].copy_from_slice(&0x200u64.to_le_bytes());
    elf[symtab + 32..symtab + 40].copy_from_slice(&48u64.to_le_bytes());
    elf[symtab + 40..symtab + 44].copy_from_slice(&2u32.to_le_bytes());
    elf[symtab + 56..symtab + 64].copy_from_slice(&24u64.to_le_bytes());

    let strtab = 0x100 + 128;
    elf[strtab + 4..strtab + 8].copy_from_slice(&3u32.to_le_bytes());
    elf[strtab + 24..strtab + 32].copy_from_slice(&0x280u64.to_le_bytes());
    elf[strtab + 32..strtab + 40].copy_from_slice(&32u64.to_le_bytes());

    let data = 0x100 + 192;
    elf[data + 4..data + 8].copy_from_slice(&1u32.to_le_bytes());
    elf[data + 16..data + 24].copy_from_slice(&0x1000u64.to_le_bytes());
    elf[data + 24..data + 32].copy_from_slice(&0x300u64.to_le_bytes());
    elf[data + 32..data + 40].copy_from_slice(&4u64.to_le_bytes());

    let symbol = 0x200 + 24;
    elf[symbol..symbol + 4].copy_from_slice(&1u32.to_le_bytes());
    elf[symbol + 6..symbol + 8].copy_from_slice(&3u16.to_le_bytes());
    elf[symbol + 8..symbol + 16].copy_from_slice(&0x1000u64.to_le_bytes());
    elf[symbol + 16..symbol + 24].copy_from_slice(&4u64.to_le_bytes());
    elf[0x281..0x295].copy_from_slice(b"plow_packet_hash_lo\0");
    elf[0x300..0x304].copy_from_slice(&0x1234_5678u32.to_le_bytes());

    assert_eq!(
        elf_symbol_u32(&elf, "plow_packet_hash_lo"),
        Some(0x1234_5678)
    );
    assert_eq!(elf_symbol_u32(&elf, "plow_packet_hash_hi"), None);
}

#[test]
fn required_compiled_opcode_markers_are_fail_closed() {
    let path = Path::new("interp_decode_k3.elf");
    assert!(
        check_compiled_opcode_marker_set(&["plow_opcode_attn_res_1"], path, [DevOp::AttnRes])
            .is_ok()
    );
    for &(op, marker) in COMPILED_OPCODE_MARKERS {
        let err = check_compiled_opcode_marker_set(&[], path, [op])
            .unwrap_err()
            .to_string();
        assert!(err.contains(marker), "{op:?} must require {marker}: {err}");
        assert!(check_compiled_opcode_marker_set(&[marker], path, [op]).is_ok());
    }
}

#[test]
fn interpreter_wave_geometry_rejects_missing_or_swapped_phase_objects() {
    for phase in [Phase::Prefill, Phase::Decode, Phase::Flash] {
        let expected = phase.interpreter_threads() / 64;
        assert!(check_interpreter_waves(Some(expected), phase, Path::new("phase.elf")).is_ok());
        for waves in [None, Some(0), Some(2), Some(if expected == 4 { 8 } else { 4 })] {
            let error = check_interpreter_waves(waves, phase, Path::new("phase.elf"))
                .expect_err("a filename cannot establish a compatible launch geometry");
            assert!(error.to_string().contains("plow_geom_PLOW_WG_WAVES"));
        }
    }
}

#[test]
fn specialised_decode_arms_require_a_build_manifest() {
    let mut plain = segmented_prog(&[DevOp::KdaStateStepG], &[0]);
    assert!(packet_decode_arm_requirements(std::slice::from_ref(&plain)).is_empty());

    plain.insts[0].i[4] = 4;
    assert_eq!(
        packet_decode_arm_requirements(std::slice::from_ref(&plain)),
        ["PLOW_KDA_FB_FOLD=1"]
    );

    let mut xr = segmented_prog(&[DevOp::XReduce], &[0]);
    xr.insts[0].i[7] = 1;
    assert_eq!(
        packet_decode_arm_requirements(std::slice::from_ref(&xr)),
        ["PLOW_XR_COMBINE_FOLD=1"]
    );

    let err = check_decode_object(
        &[],
        Path::new("stripped.elf"),
        &packet_decode_arm_requirements(std::slice::from_ref(&plain)),
        false,
        false,
    )
    .expect_err("a specialised packet cannot use an unverifiable object");
    assert!(err.to_string().contains("no ELF symbol table"));
}

#[test]
fn specialised_prefill_arms_are_derived_without_a_manifest() {
    let mut down = segmented_prog(&[DevOp::MoeGroupDownPf], &[0]);
    down.insts[0].i[4] = 3;
    down.insts[0].i[7] = 1;
    assert_eq!(
        packet_prefill_arm_requirements(std::slice::from_ref(&down)),
        [
            "PLOW_MOE_PREFILL=1",
            "PLOW_MOE_PF_PART16=1",
            "PLOW_MOE_PF_ATOMIC=1"
        ]
    );
    let err = check_prefill_object(
        &[],
        Path::new("stripped-prefill.elf"),
        &packet_prefill_arm_requirements(std::slice::from_ref(&down)),
    )
    .expect_err("a specialised packet cannot use an unverifiable prefill object");
    assert!(err.to_string().contains("no ELF symbol table"));
}

#[test]
fn specialised_amd_manifest_hash_must_be_valid() {
    let path = Path::new("build.json");
    assert_eq!(
        manifest_pairing_hash(br#"{"pairing":{"hash":"0x1234"}}"#, path).unwrap(),
        0x1234
    );
    assert!(manifest_pairing_hash(br#"{"pairing":{}}"#, path).is_err());
}

fn segmented_decode_probe() -> DevProg {
    let insts = vec![
        DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::KdaDecodeFused as u16,
            blocks: 12,
            i: [1, 12, 128, 8, 4, 1, 1, 2],
            fj: [
                0.08838835f32.to_bits(),
                (-5.0f32).to_bits(),
                1.0e-5f32.to_bits(),
            ],
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        },
    ];
    DevProg {
        t: 1,
        packed_prefill_only: false,
        token_batch_body: false,
        n_counter: 0,
        insts,
        stream: (0..3)
            .map(|seg| packet::dev::StreamEnt {
                inst: seg,
                seg: seg as u16,
                ..Default::default()
            })
            .collect(),
        stream_ofs: vec![0],
        stream_len: vec![3],
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: Vec::new(),
        gq_seg_ofs: Vec::new(),
        l2_domains: 0,
    }
}

#[test]
fn isolated_sparse_mla_decode_is_a_native_segment_only_when_pure() {
    let sparse = DevInst64 {
        op: DevOp::FlashMlaDecodeFp8 as u16,
        blocks: 256,
        t: [0, 1, 2, 3, 4, 5, 6, 7],
        i: [1, 8, 4096, 0, 16, u32::MAX, 2048, 4],
        fj: [0.0625f32.to_bits(), 9, 0],
        ..Default::default()
    };
    let make = || {
        let mut p = segmented_decode_probe();
        p.insts[1] = sparse;
        p
    };
    let p = make();
    assert_eq!(
        decode_segment_kinds(&p).unwrap(),
        [
            DecodeSegmentKind::Interpreter,
            DecodeSegmentKind::SparseMlaDecode(1),
            DecodeSegmentKind::Interpreter,
        ]
    );
    validate_decode_dispatch(std::slice::from_ref(&p), 0).unwrap();
    // The ordinary emit: counter obligations or a shared segment keep it on the interpreter.
    let mut waits = make();
    waits.stream[1].wait_len = 1;
    assert!(decode_segment_kinds(&waits)
        .unwrap()
        .iter()
        .all(|k| matches!(k, DecodeSegmentKind::Interpreter)));
    let mut shared = make();
    shared.stream[0].seg = 1;
    assert!(decode_segment_kinds(&shared)
        .unwrap()
        .iter()
        .all(|k| matches!(k, DecodeSegmentKind::Interpreter)));
    // A dense FP8 decode op (no selection handle) is never a native boundary.
    let mut dense = make();
    dense.insts[1].fj[1] = 0;
    assert!(decode_segment_kinds(&dense)
        .unwrap()
        .iter()
        .all(|k| matches!(k, DecodeSegmentKind::Interpreter)));
}

#[test]
fn fused_decode_segment_is_a_pure_ordered_single_rank_route() {
    let p = segmented_decode_probe();
    assert_eq!(
        decode_segment_kinds(&p).unwrap(),
        [
            DecodeSegmentKind::Interpreter,
            DecodeSegmentKind::KdaDecodeFused(1),
            DecodeSegmentKind::Interpreter,
        ]
    );
    validate_decode_dispatch(&[p], 0).unwrap();
}

#[test]
fn decode_mla_pair_routes_as_one_specialist_segment() {
    let mut p = segmented_decode_probe();
    p.insts = vec![
        DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::FlashMlaDecode as u16,
            blocks: 256,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::MlaMergeFold as u16,
            blocks: 128,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::Gemv as u16,
            blocks: 256,
            ..Default::default()
        },
    ];
    p.stream = [0u32, 1, 2, 3]
        .into_iter()
        .zip([0u16, 1, 1, 2])
        .map(|(inst, seg)| packet::dev::StreamEnt {
            inst,
            seg,
            ..Default::default()
        })
        .collect();
    p.stream_len[0] = 4;
    assert_eq!(
        decode_segment_kinds(&p).unwrap(),
        [
            DecodeSegmentKind::Interpreter,
            DecodeSegmentKind::MlaAttention,
            DecodeSegmentKind::Interpreter,
        ]
    );
    assert!(matches!(
        decode_segment_routes(&p, &[], &[], &[]).unwrap()[1],
        DecodeSegmentRoute::MlaAttention
    ));
}

#[test]
fn mixed_decode_mla_ops_remain_on_the_ordinary_interpreter() {
    let mut p = segmented_decode_probe();
    p.insts = vec![
        DevInst64 {
            op: DevOp::FlashMlaDecode as u16,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::MlaMergeFold as u16,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::Gemv as u16,
            ..Default::default()
        },
    ];
    p.stream = (0..3)
        .map(|inst| packet::dev::StreamEnt {
            inst,
            seg: 0,
            ..Default::default()
        })
        .collect();
    assert_eq!(
        decode_segment_kinds(&p).unwrap(),
        [DecodeSegmentKind::Interpreter]
    );
    assert!(!requires_segmented_decode(
        &decode_segment_routes(&p, &[], &[], &[]).unwrap()
    ));
}

#[test]
fn fused_decode_rejects_one_raw_instruction_in_multiple_segments() {
    let mut p = segmented_decode_probe();
    p.stream.push(packet::dev::StreamEnt {
        inst: 1,
        seg: 3,
        ..Default::default()
    });
    p.stream_len[0] += 1;
    let err = decode_segment_kinds(&p)
        .expect_err("a stateful raw instruction must execute in one segment")
        .to_string();
    assert!(err.contains("segments 1 and 3"), "{err}");
}

#[test]
fn fused_decode_v1_descriptor_fails_closed() {
    let mut p = segmented_decode_probe();
    p.insts[1].i[7] = 1;
    let err = decode_segment_routes(&p, &[], &[], &[])
        .expect_err("KDA fused ABI v1 must not route to the v2 object")
        .to_string();
    assert!(err.contains("version=1"), "{err}");
}

#[test]
fn fused_decode_descriptor_builds_the_exact_raw_kernarg() {
    let mut p = segmented_decode_probe();
    p.insts[1].t = [0, 1, 2, 3, 4, 5, 6, 7];
    let handles = [
        8u32,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        packet::dev::TENSOR_NONE_I,
    ];
    let init: Vec<u8> = handles.iter().flat_map(|h| h.to_le_bytes()).collect();
    let tensors: Vec<crate::asset::devblob::DevTensor> = (0..18)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: if i == 7 { 44 } else { 1 << 20 },
            init: (i == 7).then_some(0..44),
        })
        .collect();
    let devp: Vec<DeviceMem> = (0..18)
        .map(|i| DeviceMem::view(0x1000 + i * 0x1000, 1 << 20))
        .collect();
    let routes = decode_segment_routes(&p, &tensors, &init, &devp).unwrap();
    let DecodeSegmentRoute::KdaDecodeFused(a) = routes[1] else {
        panic!("segment 1 must route to the raw object")
    };
    assert_eq!(
        (a.y, a.q_raw, a.k_raw, a.v_raw),
        (0x1000, 0x2000, 0x3000, 0x4000)
    );
    assert_eq!((a.wq, a.wk, a.wv), (0x9000, 0xa000, 0xb000));
    assert_eq!((a.csq, a.csk, a.csv), (0xc000, 0xd000, 0xe000));
    assert_eq!(
        (a.forget_raw, a.beta_raw, a.state),
        (0x5000, 0x6000, 0x7000)
    );
    assert_eq!((a.a_log, a.dt_bias, a.norm_w), (0xf000, 0x10000, 0x11000));
    assert_eq!(a.output_gate_raw, 0x12000);
    assert_eq!(a.parked, 0);
    assert_eq!((a.rows, a.heads, a.dim, a.bv, a.conv_w), (1, 12, 128, 8, 4));
    assert_eq!((a.flags, a.gate_mode), (1, 1));
    assert_eq!(
        (a.lower_bound, a.scale, a.norm_eps),
        (-5.0, 0.08838835, 1.0e-5)
    );
    assert_eq!(as_bytes(std::slice::from_ref(&a)).len(), 184);
}

#[test]
fn fused_decode_refuses_mixed_countered_and_l2_segments() {
    let mut mixed = segmented_decode_probe();
    mixed.stream.push(packet::dev::StreamEnt {
        inst: 0,
        seg: 1,
        ..Default::default()
    });
    mixed.stream_len[0] += 1;
    assert!(decode_segment_kinds(&mixed)
        .unwrap_err()
        .to_string()
        .contains("mixes KdaDecodeFused"));

    let mut countered = segmented_decode_probe();
    countered.stream[1].succ_len = 1;
    countered.succs.push(0);
    assert!(decode_segment_kinds(&countered)
        .unwrap_err()
        .to_string()
        .contains("counter obligations"));

    let mut crossing = segmented_decode_probe();
    crossing.stream[0].succ_len = 1;
    crossing.succs.push(7);
    crossing.stream[2].wait_len = 1;
    crossing.waits.push(packet::dev::Wait {
        id: 7,
        threshold: 1,
    });
    assert!(decode_segment_kinds(&crossing)
        .unwrap_err()
        .to_string()
        .contains("crosses into segment"));

    let mut placed = segmented_decode_probe();
    placed.l2_domains = 8;
    assert!(validate_decode_dispatch(&[placed], 0)
        .unwrap_err()
        .to_string()
        .contains("L2-domain placement"));
}

#[test]
fn ordinary_l2_decode_allows_cross_domain_counters() {
    let mut placed = segmented_prog(&[DevOp::Gemv, DevOp::RmsNorm], &[0, 1]);
    placed.l2_domains = 2;
    placed.stream[0].succ_len = 1;
    placed.succs.push(7);
    placed.stream[1].wait_len = 1;
    placed.waits.push(packet::dev::Wait {
        id: 7,
        threshold: 1,
    });

    assert!(decode_segment_kinds(&placed).is_ok());
    assert!(validate_decode_dispatch(&[placed], 0).is_ok());
}

#[test]
fn legacy_l2_prefill_forces_primary_interpreter() {
    let legacy = ProgramDispatch::classify(8, 8, 8);
    assert_eq!(legacy, ProgramDispatch::L2Domains(8));
    assert!(!prefill_segment_specialization_allowed(legacy));

    let current = ProgramDispatch::classify(8, 3, 24);
    assert_eq!(
        current,
        ProgramDispatch::L2Segments {
            domains: 8,
            segments: 3,
        }
    );
    assert!(prefill_segment_specialization_allowed(current));
    assert!(prefill_segment_specialization_allowed(
        ProgramDispatch::WaveSegments(3)
    ));
}

fn overlap_range(name: &str, start: u64, end: u64) -> AmdOwnedRange {
    AmdOwnedRange {
        name: name.into(),
        address_space: "device",
        start,
        end,
    }
}

#[test]
fn overlap_capability_fails_closed_for_shared_or_missing_evidence() {
    let shared = overlap_range("shared", 0x1000, 0x2000);
    let evidence = [AmdOverlapRankEvidence {
        rank: 0,
        queue_count: 1,
        prefill_ranges: vec![shared.clone()],
        decode_ranges: vec![shared],
        prefill_queue_ids: vec![7],
        decode_queue_ids: vec![7],
    }];
    assert_eq!(
        derive_overlap_capability(&evidence),
        AmdOverlapCapability {
            scratch_isolated: false,
            queue_isolated: false,
            overlap_safe: false,
            queue_scope: "global_per_rank",
            queue_count: 1,
            per_xcd_queues: false,
            ranks: 1,
        }
    );
    let missing = derive_overlap_capability(&[]);
    assert!(!missing.scratch_isolated);
    assert!(!missing.queue_isolated);
    assert!(!missing.overlap_safe);
    assert_eq!(missing.queue_count, 0);
}

#[test]
fn overlap_capability_requires_both_disjoint_intervals_and_queues() {
    let evidence = [AmdOverlapRankEvidence {
        rank: 0,
        queue_count: 2,
        prefill_ranges: vec![overlap_range("prefill", 0x1000, 0x2000)],
        decode_ranges: vec![overlap_range("decode", 0x2000, 0x3000)],
        prefill_queue_ids: vec![7],
        decode_queue_ids: vec![8],
    }];
    let capability = derive_overlap_capability(&evidence);
    assert!(capability.scratch_isolated);
    assert!(capability.queue_isolated);
    assert!(capability.overlap_safe);
    assert_eq!(capability.queue_count, 2);
}

fn packed_spans() -> [PrefillSpan; 2] {
    [
        PrefillSpan {
            row0: 0,
            n_rows: 3,
            slot: 0,
            flags: PREFILL_SPAN_RESET_STATE,
            kv_row0: 0,
            kv_len: 3,
            state_slot: 0,
            program: 2,
        },
        PrefillSpan {
            row0: 3,
            n_rows: 2,
            slot: 1,
            flags: 0,
            kv_row0: 5,
            kv_len: 7,
            state_slot: 1,
            program: 2,
        },
    ]
}

#[test]
fn packed_dispatch_selects_tagged_sibling_without_changing_ordinary_rung() {
    let roles = [
        (128, false),
        (512, false),
        (128, true),
        (512, true),
        (1, false),
    ];
    let select = |requested| packed_prefill_topology_index(requested, 4, |i| roles.get(i).copied());
    assert_eq!(select(0), Some(2));
    assert_eq!(select(1), Some(3));
    assert_eq!(select(2), Some(2));
    assert_eq!(select(4), None);

    let legacy = [(128, false), (1, false)];
    assert_eq!(
        packed_prefill_topology_index(0, 1, |i| legacy.get(i).copied()),
        Some(0)
    );
}

#[test]
fn packed_prefill_accepts_dense_ragged_rows_and_parked_padding() {
    let binding =
        validate_packed_prefill(2, 8, 2, &packed_spans(), &[0, 0, 0, 0, 0, 1, 1, 1]).unwrap();
    assert_eq!(
        binding,
        PackedPrefillBinding {
            prog: 2,
            n_spans: 2,
            n_rows: 8,
            token_batch: false,
        }
    );
}

#[test]
fn packed_prefill_rejects_malformed_spans_and_masks() {
    let good_mask = [0, 0, 0, 0, 0, 1, 1, 1];
    let reject = |spans: &[PrefillSpan], mask: &[u32]| {
        assert!(validate_packed_prefill(2, 8, 2, spans, mask).is_err());
    };

    let mut spans = packed_spans();
    spans[1].row0 = 4;
    reject(&spans, &good_mask);
    let mut spans = packed_spans();
    spans[1].kv_len = 8;
    reject(&spans, &good_mask);
    let mut spans = packed_spans();
    spans[1].flags = PREFILL_SPAN_RESET_STATE;
    reject(&spans, &good_mask);
    let mut spans = packed_spans();
    spans[1].slot = 0;
    spans[1].state_slot = 0;
    reject(&spans, &good_mask);
    let mut spans = packed_spans();
    spans[1].state_slot = 0;
    reject(&spans, &good_mask);
    let mut spans = packed_spans();
    spans[1].program = 3;
    reject(&spans, &good_mask);
    reject(&packed_spans(), &[0, 0, 0, 0, 0]);
    reject(&packed_spans(), &[0, 1, 0, 0, 0, 1, 1, 1]);
    reject(&packed_spans(), &[0, 0, 0, 0, 0, 0, 1, 1]);
}

#[test]
fn packed_prefill_prompt_slices_must_match_spans_exactly() {
    let a = [10, 11, 12];
    let b = [20, 21];
    assert_eq!(
        validate_packed_prompt_slices(8, &packed_spans(), &[&a, &b]).unwrap(),
        5
    );
    assert!(validate_packed_prompt_slices(8, &packed_spans(), &[&a]).is_err());
    assert!(validate_packed_prompt_slices(8, &packed_spans(), &[&a, &[20]]).is_err());
    assert!(validate_packed_prompt_slices(4, &packed_spans(), &[&a, &b]).is_err());
}

#[test]
fn packed_prefill_stages_concatenated_ids_and_absolute_positions() {
    let a = [10, 11, 12];
    let b = [20, 21];
    let mut stage = [0xff; 64];
    assert_eq!(
        stage_packed_prompt_rows(&mut stage, 8, &packed_spans(), &[&a, &b]).unwrap(),
        5
    );
    let words = |half: usize| {
        (0..8)
            .map(|i| {
                let off = half * 32 + i * 4;
                u32::from_le_bytes(stage[off..off + 4].try_into().unwrap())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(words(0), [10, 11, 12, 20, 21, 0, 0, 0]);
    assert_eq!(words(1), [0, 1, 2, 5, 6, 0, 0, 0]);
}

#[test]
fn packed_prefill_patches_generic_work_to_dense_total_rows() {
    let mut inst = DevInst64 {
        op: DevOp::Embed as u16,
        i: [8, 0, 0, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    rebase_chunk_rows(std::slice::from_mut(&mut inst), &[], 0, 5, 8, Some(8));
    assert_eq!(inst.i[0], 5);
}

#[test]
fn packed_prefill_kernarg_is_legacy_null_or_exact_program_only() {
    assert_eq!(
        packed_prefill_kernarg(None, 2, 0x1000, 0x2000),
        (0, 0, 0, 0)
    );
    let binding = PackedPrefillBinding {
        prog: 2,
        n_spans: 2,
        n_rows: 8,
        token_batch: false,
    };
    assert_eq!(
        packed_prefill_kernarg(Some(binding), 2, 0x1000, 0x2000),
        (0x1000, 0x2000, 2, 8)
    );
    assert_eq!(
        packed_prefill_kernarg(Some(binding), 1, 0x1000, 0x2000),
        (0, 0, 0, 0)
    );
    assert!(check_packed_prefill_dispatch(Some(binding), 2).is_ok());
    assert!(check_packed_prefill_dispatch(Some(binding), 1).is_err());
}

#[test]
fn packed_prefill_requires_abi_on_every_routed_object() {
    assert!(check_packed_prefill_abi(true, false, false).is_ok());
    assert!(check_packed_prefill_abi(true, true, true).is_ok());

    let missing_prefill = check_packed_prefill_abi(false, false, false)
        .unwrap_err()
        .to_string();
    assert!(missing_prefill.contains(PACKED_PREFILL_ABI_SYM));

    let missing_flash = check_packed_prefill_abi(true, true, false)
        .unwrap_err()
        .to_string();
    assert!(missing_flash.contains("flash object"));
    assert!(missing_flash.contains(PACKED_PREFILL_ABI_SYM));
}

#[test]
fn hierarchical_gate_marker_requires_decode_gq_and_l2_capability() {
    let object = Path::new("interp_decode_gq.elf");
    let valid = [GATE_HIER_SYM, L2_DISPATCH_SYM];
    assert!(check_gate_hier_object(&valid, object, Phase::Decode, Sched::GlobalQueue).is_ok());
    assert!(
        check_gate_hier_object(&[L2_DISPATCH_SYM], object, Phase::Prefill, Sched::Static).is_ok()
    );

    for (syms, phase, sched) in [
        (&[GATE_HIER_SYM][..], Phase::Decode, Sched::GlobalQueue),
        (
            &[GATE_HIER_SYM, L2_DISPATCH_SYM][..],
            Phase::Decode,
            Sched::Static,
        ),
        (
            &[GATE_HIER_SYM, L2_DISPATCH_SYM][..],
            Phase::Prefill,
            Sched::GlobalQueue,
        ),
    ] {
        let message = check_gate_hier_object(syms, object, phase, sched)
            .expect_err("invalid hierarchical-gate object must be refused")
            .to_string();
        assert!(message.contains(GATE_HIER_SYM));
        assert!(message.contains(L2_DISPATCH_SYM));
    }
}

/// One decode program whose queue entries carry `nper` slices per domain, or none.
fn gate_hier_probe(l2_domains: u32, nper: u16) -> DevProg {
    DevProg {
        t: 1,
        packed_prefill_only: false,
        token_batch_body: false,
        n_counter: 0,
        insts: vec![DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        }],
        stream: Vec::new(),
        stream_ofs: vec![0],
        stream_len: vec![0],
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: (0..4)
            .map(|_| packet::dev::StreamEnt {
                inst: 0,
                seg: 0,
                flags: nper << packet::dev::SE_NPER_SHIFT,
                ..Default::default()
            })
            .collect(),
        gq_seg_ofs: vec![0, 4],
        l2_domains,
    }
}

/// ARMED IS NOT FIRING. The object flag alone licenses nothing: `interp.hip` takes the
/// hierarchy only when the blob is placed AND a packet has more than one slice on a domain.
#[test]
fn gate_hier_status_separates_armed_from_firing() {
    let placed = [gate_hier_probe(8, 38)];
    let unplaced = [gate_hier_probe(0, 0)];

    let firing = GateHierStatus::of(&placed, 0, true);
    assert!(firing.armed && firing.firing);
    assert_eq!(
        (firing.domains, firing.rendezvous, firing.entries),
        (8, 4, 4)
    );
    assert!(firing.verdict().contains("FIRING"));

    // The shipped-but-inert pairing this whole campaign exists to make visible.
    let inert = GateHierStatus::of(&unplaced, 0, true);
    assert!(inert.armed && !inert.firing);
    assert!(inert.verdict().contains("ARMED BUT INERT"));
    assert!(inert.verdict().contains("PLOW_L2_PLACE=1"));

    // Placed blob, object without the gate: legal, and the placement half still runs.
    let no_gate = GateHierStatus::of(&placed, 0, false);
    assert!(!no_gate.armed && !no_gate.firing);
    assert!(no_gate.verdict().contains("PLOW_L2HIER=1"));

    // A single slice per (packet, domain) has nobody to rendezvous with; the emitter
    // leaves `nper` at 0 and the interpreter reads that as "no hierarchy".
    let alone = GateHierStatus::of(&[gate_hier_probe(8, 0)], 0, true);
    assert!(alone.armed && !alone.firing);
    assert_eq!(alone.rendezvous, 0);

    assert!(!GateHierStatus::of(&unplaced, 0, false).firing);
}

/// The refusal has to name BOTH halves and the two ways out — the emit-side flag and the
/// object-side flag are different names in different files.
#[test]
fn l2_pairing_refusal_names_both_halves_and_the_fix() {
    let decode = l2_pairing_refusal(Path::new("interp_decode_gq.elf"), Phase::Decode);
    assert!(decode.contains("interp_decode_gq.elf"));
    assert!(decode.contains("PLOW_L2_PLACE_DISPATCH"));
    assert!(decode.contains("PLOW_L2_PLACE=0"));
    assert!(decode.contains("PLOW_L2HIER=1"));

    for phase in [Phase::Prefill, Phase::Flash] {
        let prefill = l2_pairing_refusal(Path::new("interp_prefill_gq.elf"), phase);
        assert!(prefill.contains("interp_prefill_gq.elf"));
        assert!(prefill.contains("PLOW_L2_PLACE_DISPATCH"));
        // The prefill half has its OWN two flags; naming the decode ones would send the
        // reader to a build that does not move this object.
        assert!(prefill.contains("PLOW_L2HIER_PF=1"));
        assert!(prefill.contains("PLOW_L2_PLACE_PREFILL=1"));
    }
}

#[test]
fn prefill_only_accumulation_does_not_require_a_decode_arm() {
    let obj = Path::new("interp_decode_k3.elf");
    let mut prefill = segmented_prog(&[DevOp::MoeGroupDownPf], &[0]);
    prefill.insts[0].i[4] = 4;
    let b1_decode = segmented_prog(&[DevOp::Gemv], &[0]);
    assert!(required_moe_pf_accum(std::slice::from_ref(&prefill), 4));
    assert!(!required_moe_pf_accum(std::slice::from_ref(&b1_decode), 4));
    let requires = vec![
        "PLOW_MOE_PF_ATOMIC=1".to_owned(),
        "PLOW_MOE_PF_DET=1".to_owned(),
    ];
    assert!(
        check_decode_object(&["plow_interp_dec_gfx950"], obj, &requires, false, false,).is_ok()
    );
}

#[test]
fn grouped_decode_requires_its_accumulation_arm() {
    let obj = Path::new("interp_decode_k3.elf");
    for (flag, marker) in [
        ("PLOW_MOE_PF_ATOMIC", "plow_moe_pf_atomic_arm"),
        ("PLOW_MOE_PF_DET", "plow_moe_pf_det_arm"),
    ] {
        let requires = vec![format!("{flag}=1")];
        let mut prog = segmented_prog(&[DevOp::MoeGroupDownPf], &[0]);
        prog.insts[0].i[if flag == "PLOW_MOE_PF_ATOMIC" { 4 } else { 5 }] = 4;
        let need_atomic = required_moe_pf_accum(std::slice::from_ref(&prog), 4);
        let need_det = required_moe_pf_accum(std::slice::from_ref(&prog), 5);
        assert!(check_decode_object(&[marker], obj, &requires, need_atomic, need_det,).is_ok());
        let err = check_decode_object(
            &["plow_interp_dec_gfx950"],
            obj,
            &requires,
            need_atomic,
            need_det,
        )
        .expect_err("a grouped decode object without its accumulation arm must refuse");
        assert!(err.to_string().contains(flag));
    }
}

fn segmented_prog(ops: &[DevOp], segs: &[u16]) -> DevProg {
    assert_eq!(ops.len(), segs.len());
    DevProg {
        t: 2048,
        packed_prefill_only: false,
        token_batch_body: false,
        n_counter: 0,
        insts: ops
            .iter()
            .map(|&op| DevInst64 {
                op: op as u16,
                ..Default::default()
            })
            .collect(),
        stream: segs
            .iter()
            .enumerate()
            .map(|(inst, &seg)| packet::dev::StreamEnt {
                inst: inst as u32,
                seg,
                ..Default::default()
            })
            .collect(),
        stream_ofs: Vec::new(),
        stream_len: Vec::new(),
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: Vec::new(),
        gq_seg_ofs: Vec::new(),
        l2_domains: 0,
    }
}

#[test]
fn packed_dense_contract_accepts_split_attention_and_refuses_other_state() {
    let mut p = segmented_prog(
        &[
            DevOp::RmsNorm,
            DevOp::GemmWide,
            DevOp::HeadNormRope,
            DevOp::FlashPrefill,
            DevOp::FlashMerge,
        ],
        &[0, 0, 0, 1, 2],
    );
    p.insts[3].i[6] = 512;
    p.insts[3].i[7] = 4;
    p.insts[3].t[5] = packet::dev::TENSOR_NONE16;
    assert!(super::check_packed_dense_program(&p.insts).is_ok());
    p.insts[0].op = DevOp::NormResidualNorm as u16;
    assert!(super::check_packed_dense_program(&p.insts).is_ok());
    p.insts[0].op = DevOp::RmsNorm as u16;
    p.insts[3].i[6] = 64;
    assert!(super::check_packed_dense_program(&p.insts).is_err());
    p.insts[3].i[6] = 512;
    p.insts[0].i[2] = 1;
    assert!(super::check_packed_dense_program(&p.insts).is_err());
    p.insts[0].i[2] = 0;
    for op in [
        DevOp::FlashPrefillFp8,
        DevOp::FlashMlaPrefill,
        DevOp::KdaStateStep,
    ] {
        p.insts[3].op = op as u16;
        assert!(super::check_packed_dense_program(&p.insts).is_err());
    }
}

#[test]
fn prefill_sandwich_norm_requires_dispatch_marker() {
    let plain = segmented_prog(&[DevOp::RmsNorm], &[0]);
    let fused = segmented_prog(&[DevOp::NormResidualNorm], &[0]);
    let path = Path::new("interp_prefill_gq.elf");
    let legacy = [
        "plow_packed_prefill_dense_consumers_1",
        "d_norm_residual_norm",
    ];
    let plain_requires = packet_prefill_arm_requirements(&[plain]);
    assert!(plain_requires.is_empty());
    assert!(check_prefill_object(&legacy, path, &plain_requires).is_ok());
    let requires = packet_prefill_arm_requirements(&[fused]);
    assert_eq!(requires, ["PLOW_HAS_NORM_RESIDUAL_NORM=1"]);
    for syms in [&legacy[..], &[][..]] {
        assert!(check_prefill_object(syms, path, &requires).is_err());
    }
    assert!(check_prefill_object(&["plow_prefill_nrn_consumer_1"], path, &requires).is_ok());
}

fn phase_chain_manifest() -> serde_json::Value {
    serde_json::json!({
        "dispatch_chains": [{
            "program": 0,
            "kind": "prefill",
            "topology": "ordinary",
            "segments": [
                {"segment": 0, "families": ["elementwise"], "arms": ["Nop"]},
                {"segment": 1, "families": ["collective"], "arms": ["XReduceTwoShot"]},
                {"segment": 2, "families": ["elementwise"], "arms": ["Nop"]}
            ],
            "phases": [
                {
                    "first_segment": 0,
                    "last_segment": 0,
                    "segments": 1,
                    "families": ["elementwise"],
                    "arms": ["Nop"],
                    "object_class": "ordinary"
                },
                {
                    "first_segment": 1,
                    "last_segment": 1,
                    "segments": 1,
                    "families": ["collective"],
                    "arms": ["XReduceTwoShot"],
                    "object_class": "ordinary",
                    "resource_contract": {
                        "policy": "refuse",
                        "wavefront_size": 64,
                        "min_occupancy_waves_per_simd": 2,
                        "max_private_segment_bytes_delta": 0,
                        "max_vgpr_spill_delta": 0,
                        "max_sgpr_spill_delta": 0
                    }
                },
                {
                    "first_segment": 2,
                    "last_segment": 2,
                    "segments": 1,
                    "families": ["elementwise"],
                    "arms": ["Nop"],
                    "object_class": "ordinary"
                }
            ]
        }]
    })
}

#[test]
fn graph_phase_selector_requires_matching_contiguous_packet_inventory() {
    let prog = segmented_prog(&[DevOp::Nop, DevOp::XReduceTwoShot, DevOp::Nop], &[0, 1, 2]);
    let selected = graph_phase_xreduce_segments_from_manifest(
        &phase_chain_manifest(),
        std::slice::from_ref(&prog),
        1,
        Path::new("build.json"),
    )
    .unwrap();
    assert_eq!(selected[0], BTreeSet::from([1]));

    let mut bad = phase_chain_manifest();
    bad["dispatch_chains"][0]["segments"][1]["arms"] = serde_json::json!(["Gemm"]);
    let err = graph_phase_xreduce_segments_from_manifest(
        &bad,
        std::slice::from_ref(&prog),
        1,
        Path::new("build.json"),
    )
    .expect_err("phase and segment inventories must agree");
    assert!(err.to_string().contains("segment inventory differs"));
}

#[test]
fn graph_phase_selector_rejects_topology_and_resource_drift() {
    let prog = segmented_prog(&[DevOp::Nop, DevOp::XReduceTwoShot, DevOp::Nop], &[0, 1, 2]);
    for (pointer, value, needle) in [
        (
            "/dispatch_chains/0/topology",
            serde_json::json!("packed"),
            "topology mismatch",
        ),
        (
            "/dispatch_chains/0/phases/1/resource_contract/wavefront_size",
            serde_json::json!(32),
            "incompatible resource contract",
        ),
    ] {
        let mut bad = phase_chain_manifest();
        *bad.pointer_mut(pointer).unwrap() = value;
        let err = graph_phase_xreduce_segments_from_manifest(
            &bad,
            std::slice::from_ref(&prog),
            1,
            Path::new("build.json"),
        )
        .expect_err("manifest drift must fail closed");
        assert!(err.to_string().contains(needle), "{err}");
    }
}

#[test]
fn lean_moe_stage1_route_requires_exact_shape_and_align() {
    let mut prog = segmented_prog(&[DevOp::MoeAlignPf, DevOp::MoeGroupGluPf], &[0, 1]);
    prog.t = 8192;
    prog.insts[0].t[0] = 4;
    prog.insts[0].i = [8192, 896, 16, 0, 0, 0, 0, 0];
    prog.insts[1].blocks = 256;
    prog.insts[1].t = [0, 1, 2, 3, 4, 5, 6, 7];
    prog.insts[1].i = [384, 3584, 896, MOE_ENC_MXFP4, 0, 2, 0, 0];
    prog.insts[1].fj = [4.0f32.to_bits(), 25.0f32.to_bits(), 0];
    let tensors: Vec<_> = (0..8)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..tensors.len())
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();

    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::MoeStage1Mxfp4(route) = routes[1] else {
        panic!("exact stage-1 packet must route to the BK256 object")
    };
    assert_eq!(route.grid, 256);
    assert_eq!((route.args.inter_dim, route.args.model_dim), (384, 3584));
    assert_eq!((route.args.beta, route.args.linear_beta), (4.0, 25.0));

    prog.insts[1].i[0] = 224;
    let fallback = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(fallback[1], PrefillSegmentRoute::Interpreter));
}

#[test]
fn lean_moe_stage1_a4_reuse_is_geometry_gated_and_scratch_bounded() {
    let mut prog = segmented_prog(&[DevOp::MoeAlignPf, DevOp::MoeGroupGluPf], &[0, 1]);
    prog.t = 8192;
    prog.insts[0].t[0] = 4;
    prog.insts[0].i = [8192, 896, 16, 0, 0, 0, 0, 0];
    prog.insts[1].blocks = 256;
    prog.insts[1].t = [0, 1, 2, 3, 4, 5, 6, 7];
    prog.insts[1].i = [384, 3584, 896, MOE_ENC_MXFP4, 0, 2, 0, 0];
    prog.insts[1].fj = [4.0f32.to_bits(), 25.0f32.to_bits(), 0];
    let rows = 151_232u64;
    let mut tensors: Vec<_> = (0..8)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    tensors[2].name = "moe.x.expert_weight_table".into();
    tensors[3].name = "moe.x.expert_scale_table".into();
    tensors[5].bytes = rows * 4;
    tensors[6].bytes = rows * 4;
    let devp: Vec<_> = tensors
        .iter()
        .enumerate()
        .map(|(i, t)| DeviceMem::view(0x1000 + i as u64 * 0x10_0000, t.bytes))
        .collect();
    let (payload, scales) =
        moe_stage1_a4_scratch_bytes(std::slice::from_ref(&prog), &tensors).unwrap();
    assert_eq!((payload, scales), (rows * 3584 / 2, rows * 3584 / 32));
    let routes = moe_mxfp4_routes_with_scratch(
        &prog,
        &tensors,
        &devp,
        Some((0x8000_0000, 0x9000_0000)),
        None,
    )
    .unwrap();
    let PrefillSegmentRoute::MoeStage1A4Reuse(route) = routes[1] else {
        panic!("profitable three-N-tile shape must select A4 reuse")
    };
    assert_eq!(route.quant_args.row_capacity, rows as u32);
    assert_eq!(
        (route.quant_args.out, route.quant_args.out_scale),
        (0x8000_0000, 0x9000_0000)
    );
    assert_eq!(
        (route.args.weight_table, route.args.weight_scale_table),
        (0x201000, 0x301000)
    );
    assert_eq!(route.grid, rows.div_ceil(64) as u32 * 3);

    prog.insts[1].i[0] = 256;
    let fallback = moe_mxfp4_routes_with_scratch(
        &prog,
        &tensors,
        &devp,
        Some((0x8000_0000, 0x9000_0000)),
        None,
    )
    .unwrap();
    assert!(matches!(
        fallback[1],
        PrefillSegmentRoute::MoeStage1Mxfp4(_)
    ));
}

#[test]
fn replicated_prefill_ep_routes_full_i_and_balanced_whole_experts() {
    let mut prog = segmented_prog(
        &[
            DevOp::MoeAlignPf,
            DevOp::MoeGroupGluPf,
            DevOp::MoeGroupDownPf,
            DevOp::MoeCombinePf,
        ],
        &[0, 1, 2, 3],
    );
    prog.t = 8192;
    prog.insts[0].t = [0, 1, 2, 3, 4, 0, 0, 0];
    prog.insts[0].i = [8192, 896, 16, 0, 0, 8, 0, 0];
    prog.insts[1].t = [8, 5, 6, 7, 0, 2, 3, 9];
    prog.insts[1].i = [3072, 3584, 896, MOE_ENC_MXFP4, 0, 2, 8, 0];
    prog.insts[1].fj = [4.0f32.to_bits(), 25.0f32.to_bits(), 0];
    prog.insts[2].t = [10, 8, 6, 7, 0, 9, 3, 4];
    prog.insts[2].i = [3584, 3072, 896, MOE_ENC_MXFP4, 0, 0, 8, 0];
    prog.insts[3].t = [11, 0, 0, 10, 1, 0, 0, 0];
    prog.insts[3].i = [3584, 16, 8192, 0, 0, 8, 896, 0];

    let rows = 8192u64 * 16;
    let mut tensors: Vec<_> = (0..14)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    tensors[0].bytes = (67 * 896 + 1) * 4;
    tensors[2].bytes = rows * 4;
    tensors[3].bytes = rows * 4;
    tensors[6].name = "m.expert_weight_table_ep".into();
    tensors[7].name = "m.expert_scale_table_ep".into();
    tensors[12].name = "m.expert_weight_table_moe2_ep".into();
    tensors[13].name = "m.expert_scale_table_moe2_ep".into();
    let devp: Vec<_> = tensors
        .iter()
        .enumerate()
        .map(|(i, t)| DeviceMem::view(0x1000 + i as u64 * 0x10_0000, t.bytes))
        .collect();

    let routes = moe_mxfp4_routes_with_scratch(
        &prog,
        &tensors,
        &devp,
        Some((0x8000_0000, 0x9000_0000)),
        Some((3, 8)),
    )
    .unwrap();
    let PrefillSegmentRoute::MoeEpAlign(align) = routes[0] else {
        panic!("EP align did not take its specialist route")
    };
    assert_eq!((align.args.expert_begin, align.args.expert_end), (336, 448));
    assert_eq!(align.args.experts, 896);
    assert_eq!(align.args.row_capacity, rows as u32);
    let PrefillSegmentRoute::MoeStage1A4Reuse(stage1) = routes[1] else {
        panic!("EP stage-1 did not reuse the qualified A4 object")
    };
    assert_eq!((stage1.args.inter, stage1.args.experts), (3072, 896));
    let PrefillSegmentRoute::MoeEpStage2(stage2) = routes[2] else {
        panic!("EP stage-2 did not take the full-I object")
    };
    assert_eq!((stage2.args.inter_dim, stage2.args.experts), (3072, 896));
    assert_eq!(stage2.args.weight_table, devp[12].base);
    let PrefillSegmentRoute::MoeEpCombine(combine) = routes[3] else {
        panic!("EP combine did not take the fixed-slot object")
    };
    assert_eq!(
        (combine.args.expert_begin, combine.args.expert_end),
        (336, 448)
    );
    let matrix_payload = 3584u64 * 3072 / 2;
    let matrix_scales = 3584u64 * (3072 / 32);
    assert_eq!(
        moe_prefill_ep_extra_bytes(std::slice::from_ref(&prog), &tensors, 8).unwrap(),
        112 * (3 * (matrix_payload + matrix_scales) + matrix_payload + matrix_scales)
    );

    let err = moe_mxfp4_routes_with_scratch(
        &prog,
        &tensors,
        &devp,
        Some((0x8000_0000, 0x9000_0000)),
        None,
    )
    .unwrap_err();
    assert!(err.to_string().contains("without a TP binding"));
    let err = moe_mxfp4_routes_with_scratch(
        &prog,
        &tensors,
        &devp,
        Some((0x8000_0000, 0x9000_0000)),
        Some((0, 4)),
    )
    .unwrap_err();
    assert!(err.to_string().contains("topology-mismatched"));
}

#[test]
fn lean_moe_combine_route_is_generic_and_exact_contract_only() {
    let mut prog = segmented_prog(&[DevOp::MoeCombinePf], &[0]);
    let d = &mut prog.insts[0];
    d.t = [
        0,
        packet::dev::TENSOR_NONE16,
        2,
        3,
        packet::dev::TENSOR_NONE16,
        packet::dev::TENSOR_NONE16,
        packet::dev::TENSOR_NONE16,
        packet::dev::TENSOR_NONE16,
    ];
    d.i = [2816, 16, 1024, 0, 0, 0, 0, 0];
    let tensors: Vec<_> = (0..4)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..4)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();

    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::MoeCombine(route) = routes[0] else {
        panic!("generic exact combine packet must route to the lean object")
    };
    assert_eq!((route.args.hidden, route.args.tokens), (2816, 1024));
    assert_eq!((route.args.residual, route.args.shared), (0, 0x1200));
    assert_eq!((route.args.out, route.args.part), (0x1000, 0x1300));
    assert_eq!(route.grid, 512);

    prog.insts[0].i[2] = 128;
    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::MoeCombine(route) = routes[0] else {
        panic!("short-token exact combine packet must remain eligible")
    };
    assert_eq!(route.grid, 128);

    for (integer, value) in [(1, 8), (4, 1), (7, 1)] {
        prog.insts[0].i = [2816, 16, 1024, 0, 0, 0, 0, 0];
        prog.insts[0].i[integer] = value;
        let fallback = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
        assert!(matches!(fallback[0], PrefillSegmentRoute::Interpreter));
    }
    prog.insts[0].i = [2816, 16, 1024, 0, 0, 0, 0, 0];
    prog.insts[0].fj[0] = 1;
    let fallback = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(fallback[0], PrefillSegmentRoute::Interpreter));
}

#[test]
fn lean_moe_combine_route_rejects_mixed_segments() {
    let mut prog = segmented_prog(&[DevOp::MoeCombinePf, DevOp::RmsNorm], &[0, 0]);
    prog.insts[0].t[0] = 0;
    prog.insts[0].t[3] = 1;
    prog.insts[0].i = [4096, 16, 256, 0, 0, 0, 0, 0];
    let tensors: Vec<_> = (0..2)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..2)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();
    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(routes[0], PrefillSegmentRoute::Interpreter));
}

#[test]
fn cached_kda_intra_route_requires_a_pure_bt64_d128_segment() {
    let mut prog = segmented_prog(
        &[
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
        ],
        &[0, 1, 2],
    );
    prog.t = 8192;
    prog.insts[1].blocks = 256;
    prog.insts[1].t = [packet::dev::TENSOR_NONE16; 8];
    prog.insts[1].t[..6].copy_from_slice(&[0, 1, 2, 3, 4, 5]);
    prog.insts[1].i = [8192, 12, 128, 0, 0, 0, 0, 0];
    prog.insts[1].fj[0] = (1.0 / 128.0f32.sqrt()).to_bits();
    let tensors: Vec<_> = (0..6)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..6)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();

    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::KdaChunkIntraCached { args, grid } = routes[1] else {
        panic!("pure BT64/D128 intra packet must select the cached object")
    };
    assert_eq!((args.t, args.heads, args.dim, grid), (8192, 12, 128, 256));
    assert_eq!((args.aqk, args.beta), (0x1000, 0x1500));

    for e in prog.stream.iter_mut().filter(|e| e.seg == 1) {
        e.flags |= packet::dev::SE_KDA_INTRA_WAVE_ITEMS;
    }
    let classes = derive_segments_for(&prog, false).unwrap();
    let mut routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    promote_kda_intra_wave_items_routes(&prog, &classes, &mut routes).unwrap();
    assert!(matches!(
        routes[1],
        PrefillSegmentRoute::KdaChunkIntraWaveItems { grid: 256, .. }
    ));

    prog.insts[1].i[2] = 64;
    let fallback = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(fallback[1], PrefillSegmentRoute::Interpreter));

    prog.insts[1].i[2] = 128;
    prog.stream[2].seg = 1;
    let mixed = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(mixed[1], PrefillSegmentRoute::Interpreter));
}

#[test]
fn kda_key_factor_pair_routes_share_one_scratch_pair() {
    let mut prog = segmented_prog(&[DevOp::KdaChunkWu, DevOp::KdaChunkCarry], &[0, 1]);
    prog.t = 8192;
    let scale = (1.0 / 128.0f32.sqrt()).to_bits();
    prog.insts[0].blocks = 256;
    prog.insts[0].t = [0, 1, 2, 3, 4, 5, 6, 7];
    prog.insts[0].i = [8192, 12, 128, 128, 1, 0, 0, 0];
    prog.insts[0].fj[0] = scale;
    prog.insts[1].blocks = 96;
    prog.insts[1].t = [8, 9, 7, 3, 0, 1, 10, 5];
    prog.insts[1].i = [8192, 12, 128, 128, 1, 0, 0, 0];
    prog.insts[1].fj[0] = scale;
    let tensors: Vec<_> = (0..11)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..11)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();
    let half = 8192 * 12 * 128 * 2;
    assert_eq!(
        kda_key_factor_scratch_half_bytes(std::slice::from_ref(&prog)).unwrap(),
        half
    );
    let mut routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    add_kda_key_factor_routes(&prog, &devp, &mut routes, Some((0x8000_0000, half))).unwrap();
    let PrefillSegmentRoute::KdaChunkKeyFactorWu {
        args: wu,
        grid: 256,
    } = routes[0]
    else {
        panic!("eligible Wu segment did not select key-factor producer")
    };
    let PrefillSegmentRoute::KdaChunkKeyFactorCarry {
        args: carry,
        grid: 96,
    } = routes[1]
    else {
        panic!("eligible carry segment did not select key-factor consumer")
    };
    assert_eq!((wu.key_hi, carry.key_hi), (0x8000_0000, 0x8000_0000));
    assert_eq!(
        (wu.key_lo, carry.key_lo),
        (0x8000_0000 + half, 0x8000_0000 + half)
    );
    assert_eq!(
        (wu.w, carry.w, wu.u, carry.u),
        (0x1000, 0x1000, 0x1100, 0x1100)
    );

    rebase_kda_key_factor_routes(&mut routes, 777);
    let PrefillSegmentRoute::KdaChunkKeyFactorWu { args: wu, .. } = routes[0] else {
        unreachable!()
    };
    let PrefillSegmentRoute::KdaChunkKeyFactorCarry { args: carry, .. } = routes[1] else {
        unreachable!()
    };
    assert_eq!((wu.t, carry.t), (777, 777));

    prog.insts[1].i[4] = 0;
    assert!(kda_key_factor_segment_pairs(&prog).is_empty());
    assert_eq!(kda_key_factor_scratch_half_bytes(&[prog]).unwrap(), 0);
}

#[test]
fn xreduce_attnres_route_requires_the_exact_row_partition_contract() {
    let mut prog = segmented_prog(&[DevOp::XReduceTwoShot], &[0]);
    let d = &mut prog.insts[0];
    d.blocks = 256;
    d.t = [
        0,
        packet::dev::TENSOR_NONE16,
        2,
        3,
        4,
        5,
        6,
        packet::dev::TENSOR_NONE16,
    ];
    d.i = [8192 * 7168, 8, 0, 1, 2, 7168, 4, 8];
    d.fj[0] = 1e-5f32.to_bits();
    patch_tp_xaudit(&mut prog.insts, 464);
    let tensors: Vec<_> = (0..7)
        .map(|i| crate::asset::devblob::DevTensor {
            name: format!("t{i}"),
            bytes: 0x100,
            init: None,
        })
        .collect();
    let devp: Vec<_> = (0..7)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();
    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::XReduceAttnRes {
        args,
        device_args,
        grid,
    } = routes[0]
    else {
        panic!("exact packet must use the raw XR+AttnRes route")
    };
    assert_eq!(grid, 256, "raw launch grid comes from packet blocks");
    assert_eq!((args.n, args.row_w, args.nranks), (8192 * 7168, 7168, 0));
    assert_eq!(
        args.status, 465,
        "route must consume the runtime TP audit id"
    );
    assert_eq!(device_args, 0, "load-time upload fills the stable pointer");

    prog.insts[0].i[0] = 8191 * 7168;
    let err = moe_mxfp4_routes(&prog, &tensors, &devp)
        .expect_err("an encoded packet with an invalid row contract must fail closed");
    assert!(err.to_string().contains("invalid or mixed"));

    prog.insts[0].i[0] = 8192 * 7168;
    prog.insts.push(DevInst64 {
        op: DevOp::RmsNorm as u16,
        ..Default::default()
    });
    prog.stream.push(packet::dev::StreamEnt {
        inst: 1,
        seg: 0,
        ..Default::default()
    });
    let err = moe_mxfp4_routes(&prog, &tensors, &devp)
        .expect_err("the fused encoding must never fall through to the mega interpreter");
    assert!(err.to_string().contains("invalid or mixed"));
}

#[test]
fn lean_moe_stage2_route_requires_the_exact_pure_down_segment() {
    let mut prog = segmented_prog(
        &[DevOp::MoeGroupDownPf, DevOp::MoeCombinePf, DevOp::RmsNorm],
        &[0, 1, 2],
    );
    let (down, rest) = prog.insts.split_at_mut(1);
    let (d, c) = (&mut down[0], &mut rest[0]);
    d.t = [0, 1, 2, 3, 4, 5, 6, 7];
    d.i = [3584, 384, 896, MOE_ENC_MXFP4, 0, 0, 0, 0];
    c.t[0] = 0;
    c.t[1] = packet::dev::TENSOR_NONE16;
    c.t[2] = packet::dev::TENSOR_NONE16;
    c.t[3] = d.t[0];
    c.i = [3584, 16, 1024, 0, 0, 0, 0, 0];
    let tensors: Vec<_> = [
        "t0",
        "t1",
        "moe.mlp.expert_weight_table",
        "moe.mlp.expert_scale_table",
        "t4",
        "t5",
        "t6",
        "t7",
        "moe.mlp.expert_weight_table_moe2",
        "moe.mlp.expert_scale_table_moe2",
    ]
    .into_iter()
    .map(|name| crate::asset::devblob::DevTensor {
        name: name.into(),
        bytes: 0x100,
        init: None,
    })
    .collect();
    let devp: Vec<_> = (0..tensors.len())
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();
    let routes = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    let PrefillSegmentRoute::MoeStage2Mxfp4(args) = routes[0] else {
        panic!("pure supported Down must route to the lean object")
    };
    assert_eq!((args.args.model_dim, args.args.inter_dim), (3584, 384));
    assert_eq!(
        (args.args.weight_table, args.args.weight_scale_table),
        (0x1800, 0x1900)
    );
    assert_eq!(args.grid, 32256);
    // The pure combine segment takes the exact lean combine object (edffb73).
    assert!(
        matches!(routes[1], PrefillSegmentRoute::MoeCombine(_)),
        "pure combine segment must route to the lean combine object, got {:?}",
        routes[1]
    );

    let no_companion = moe_mxfp4_routes(&prog, &tensors[..8], &devp[..8]).unwrap();
    assert!(matches!(no_companion[0], PrefillSegmentRoute::Interpreter));

    prog.stream[2].seg = 0;
    let fallback = moe_mxfp4_routes(&prog, &tensors, &devp).unwrap();
    assert!(matches!(fallback[0], PrefillSegmentRoute::Interpreter));
}

#[test]
fn lean_moe_stage2_companion_layout_matches_the_declared_permutations() {
    let rows = 32usize;
    let kbytes = 64usize;
    let weight: Vec<u8> = (0..rows * kbytes).map(|i| (i % 251) as u8).collect();
    let shuffled = shuffle_moe_weight_16x32(&weight, rows, kbytes).unwrap();
    for nb in 0..rows / 16 {
        for kb in 0..kbytes / 32 {
            for kh in 0..2 {
                for nr in 0..16 {
                    for b in 0..16 {
                        let dst = ((((nb * (kbytes / 32) + kb) * 2 + kh) * 16 + nr) * 16) + b;
                        let src = (nb * 16 + nr) * kbytes + kb * 32 + kh * 16 + b;
                        assert_eq!(shuffled[dst], weight[src]);
                    }
                }
            }
        }
    }

    let (rows, groups) = (33usize, 12usize);
    let scales: Vec<u8> = (0..rows * groups).map(|i| (i % 113) as u8).collect();
    let shuffled = shuffle_mxfp4_moe2_scale(&scales, rows, groups).unwrap();
    let (padded_rows, padded_groups) = (256usize, 16usize);
    assert_eq!(shuffled.len(), padded_rows * padded_groups);
    for nb in 0..padded_rows / 32 {
        for gb in 0..padded_groups / 8 {
            for gi in 0..4 {
                for nr in 0..16 {
                    for gh in 0..2 {
                        for nh in 0..2 {
                            let dst =
                                (((((nb * (padded_groups / 8) + gb) * 4 + gi) * 16 + nr) * 2 + gh)
                                    * 2)
                                    + nh;
                            let row = nb * 32 + nh * 16 + nr;
                            let group = gb * 8 + gh * 4 + gi;
                            let expected = if row < rows && group < groups {
                                scales[row * groups + group]
                            } else {
                                127
                            };
                            assert_eq!(shuffled[dst], expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn decode_dispatch_rejects_multiple_wave_launches() {
    let one = segmented_prog(&[DevOp::Gemv], &[0]);
    assert!(validate_decode_dispatch(std::slice::from_ref(&one), 0).is_ok());

    let split = segmented_prog(&[DevOp::Gemv, DevOp::RmsNorm], &[0, 1]);
    let err = validate_decode_dispatch(std::slice::from_ref(&split), 0)
        .expect_err("decode cannot execute multiple host wave launches");
    assert!(err.to_string().contains("has 2 wave segments"));

    let mut placed = split;
    placed.l2_domains = 2;
    assert!(validate_decode_dispatch(std::slice::from_ref(&placed), 0).is_ok());
}

#[test]
fn fp8_mla_v2_routes_only_a_pure_segment_to_four_waves() {
    let pure = segmented_prog(&[DevOp::FlashMlaPrefillFp8], &[0]);
    assert_eq!(derive_segments_for(&pure, true).unwrap(), [4]);
    assert_eq!(derive_segments_for(&pure, false).unwrap(), [8]);

    let mixed = segmented_prog(&[DevOp::FlashMlaPrefillFp8, DevOp::Gemv], &[0, 0]);
    assert_eq!(derive_segments_for(&mixed, true).unwrap(), [8]);
}

#[test]
fn small_mla_family_requires_pure_dense_ordinary_segments() {
    for op in [DevOp::FlashMlaPrefill, DevOp::FlashMlaPrefillFp8] {
        let mut prog = segmented_prog(&[DevOp::Gemv, op, DevOp::MlaMergeFold], &[0, 1, 2]);
        prog.insts[1].t[7] = packet::dev::TENSOR_NONE16;
        for rows in [1, 4, 8, 16, 20, 128, 512, 1024] {
            prog.t = rows;
            assert_eq!(segment::small_mla_segments(&prog, 4), [false, true, false, false]);
            assert_eq!(derive_segments_for(&prog, false).unwrap(), [8, 8, 8]);
        }
        prog.packed_prefill_only = true;
        assert_eq!(segment::small_mla_segments(&prog, 3), [false; 3]);
        prog.packed_prefill_only = false;
        prog.t = 2048;
        assert_eq!(segment::small_mla_segments(&prog, 3), [false; 3]);
        prog.t = 128;
        prog.insts[1].i[3] = 1 << 31;
        assert_eq!(segment::small_mla_segments(&prog, 3), [false; 3]);
        prog.insts[1].i[3] = 0;
        prog.stream[0].seg = 1;
        assert_eq!(segment::small_mla_segments(&prog, 3), [false; 3]);
        prog.stream[0].seg = 0;
        if op == DevOp::FlashMlaPrefill {
            prog.insts[1].i[6] = 2;
            assert_eq!(segment::small_mla_segments(&prog, 3), [false; 3]);
            prog.insts[1].i[6] = 0;
            prog.insts[1].t[7] = 10;
        } else {
            prog.insts[1].fj[1] = 11;
        }
        assert_eq!(segment::small_mla_segments(&prog, 3), [false; 3]);
    }
}

fn small_mla_split_fixture() -> (DevProg, Vec<crate::asset::devblob::DevTensor>) {
    let mut prog = segmented_prog(&[DevOp::FlashMlaPrefillFp8, DevOp::MlaMergeFold], &[0, 1]);
    prog.t = 128;
    prog.insts[0].t = [0, 1, 2, 3, 4, 5, 6, 7];
    prog.insts[0].i = [1, 8, 81920, 0, 128, u32::MAX, 0, 4];
    prog.insts[0].fj[2] = 16;
    prog.insts[1].t = [8, 0, 1, 9, 0, 0, 0, 0];
    prog.insts[1].i = [128, 8, 256, 0, 16, 0, 0, 0];
    let tensors = [128*8*16*512*4, 128*8*16*2*4, 128*8*512*2, 128*8*64*2,
        81920*512, 81920*64*2, 4, 81920*4].into_iter().enumerate().map(|(i, bytes)| {
            crate::asset::devblob::DevTensor { name: format!("test.{i}"), bytes, init: None }
        }).collect();
    (prog, tensors)
}

#[test]
fn small_mla_splits_follow_live_context_and_patch_both_sides() {
    let (mut prog, tensors) = small_mla_split_fixture();
    let (sites, segments) = mla_prefill::split_sites(&prog, &tensors).unwrap();
    assert_eq!(segments, [true, false]);
    assert_eq!(derive_segments_for(&prog, true).unwrap(), [4, 8]);
    assert!(derive_segments_for(&prog, false).is_err());
    assert_eq!(segment::small_mla_segments(&prog, 2), [false, false]);
    for (rows, ctx, ns) in [(1,1,1), (4,32,1), (20,33,2), (128,128,4),
        (128,1024,16), (128,70000,16), (4,70000,32), (128,128,4)] {
        prog.insts[0].i[4] = rows;
        prog.insts[1].i[0] = rows;
        mla_prefill::rebase(&sites, &mut prog.insts, ctx).unwrap();
        assert_eq!((prog.insts[0].fj[2], prog.insts[1].i[4]), (ns, ns));
    }
    for (rows, ctx) in [(0,128), (129,70000), (128,127)] {
        prog.insts[0].i[4] = rows;
        assert!(mla_prefill::rebase(&sites, &mut prog.insts, ctx).is_err());
    }
}

#[test]
fn small_mla_append_context_matrix_stays_inside_partial_storage() {
    for capacity in [1u32,2,4,8,16,20,32,64,128,256,512,1024] {
        let split_cap = (304 / (capacity.div_ceil(64) * 8)).clamp(1,32);
        let partial_rows = capacity * (1 << split_cap.ilog2());
        for rows in [1,2,4,8,16,20,31,32,33,63,64,65,127,128,129,256,512,1024] {
            if rows > capacity { continue; }
            for ctx in [32,33,128,129,512,1024,4096,16384,32768,65536,70000,81920] {
                if ctx < rows { continue; }
                let ns = mla_prefill::live_splits(partial_rows, rows, ctx);
                assert!(ns.is_power_of_two() && ns <= 32);
                assert!(rows * ns <= partial_rows, "capacity={capacity} rows={rows} ctx={ctx}");
                assert!(ns <= ctx.div_ceil(32));
                assert!(rows.div_ceil(64) * 8 * ns <= 304);
            }
        }
    }
}

#[test]
fn small_mla_split_layout_refuses_unsafe_pairings() {
    for kind in 0..12 {
        let (mut prog, mut tensors) = small_mla_split_fixture();
        match kind {
            0 => prog.insts[0].fj[2] = 3,
            1 => prog.insts[1].i[4] = 1,
            2 => prog.insts[1].t[1] = 4,
            3 => tensors[0].bytes -= 4,
            4 => tensors[1].bytes -= 4,
            5 => tensors[7].bytes -= 4,
            6 => prog.insts[0].fj[1] = 1,
            7 => prog.insts[0].i[3] = 0x80000000,
            8 => prog.packed_prefill_only = true,
            9 => prog.stream[1].seg = 0,
            10 => prog.insts[0].i[6] = 1,
            _ => prog.insts[1].i[5] = 1,
        }
        assert!(mla_prefill::split_sites(&prog, &tensors).is_err(), "case {kind}");
    }
}

/// NoPE MLA prefill now ROUTES to the four-wave object instead of being refused there.
///
/// This test asserted the opposite until the `d_flash_mla_prefill_v2<512, 0>` arm landed,
/// and the change is deliberate rather than a relaxation. The old refusal was correct when
/// no zero-rope instantiation existed anywhere: routing a DR=0 segment to class 4 would have
/// staged a `Krope` half the model does not have. Now one exists, so the question is not
/// what the kernel FAMILY can do but whether THIS OBJECT carries the arm — which is a
/// symbol-table fact and is not decidable in the segment-class pass, where no object is in
/// hand. Deciding it here would either refuse every object forever or trust every object
/// blindly.
///
/// The gate therefore moved to load (`check_mla_nope_arm`, keyed on
/// `plow_mla_pf2_nope_arm`), which has its own test. What this one still pins is that the
/// segment CLASSIFICATION is unchanged in every other respect: a NoPE segment is class 4
/// exactly where a roped one is, it is still not packed-MLA compatible, and the 8-wave
/// fallback and the ns>1 refusal below still behave as before.
#[test]
fn nope_mla_prefill_routes_to_four_waves_and_is_gated_at_load() {
    for op in [DevOp::FlashMlaPrefill, DevOp::FlashMlaPrefillFp8] {
        let mut prog = segmented_prog(&[op], &[0]);
        prog.insts[0].i[3] = 0x8000_0040;
        prog.insts[0].t[7] = packet::dev::TENSOR_NONE16;
        assert!(!packed_mla_compatible(&prog));
        assert_eq!(
            derive_segments_for(&prog, true).unwrap(),
            [4],
            "{op:?}: a NoPE segment routes to the four-wave object like any other MLA \
             prefill; whether the object can run it is check_mla_nope_arm's question"
        );
        prog.insts[0].i[3] = 64;
        assert_eq!(derive_segments_for(&prog, true).unwrap(), [4]);
        assert!(packed_mla_compatible(&prog));
        prog.insts[0].t[7] = 7;
        assert_eq!(
            packed_mla_compatible(&prog),
            op == DevOp::FlashMlaPrefillFp8
        );
    }
    let mut plain = segmented_prog(&[DevOp::FlashMlaPrefill], &[0]);
    plain.insts[0].i[3] = 0x8000_0000;
    plain.insts[0].t[7] = packet::dev::TENSOR_NONE16;
    assert_eq!(derive_segments_for(&plain, false).unwrap(), [8]);
    assert_eq!(derive_raw_mla_v2_segments(&plain).unwrap(), [false]);
    plain.insts[0].i[6] = 2;
    assert!(derive_segments_for(&plain, false).is_err());

    // The second way a NoPE MLA prefill reaches the four-wave object, and the reason the old
    // refusal was not redundant with the `v2` flag: a segment carrying a plain FlashPrefill
    // is class 4 by ITS OWN rule, regardless of `v2` and regardless of MLA purity. So an
    // MLA prefill sharing that segment lands on the flash object even with V2 routing off.
    // That used to be refused; with the zero-rope arm it is simply correct, and an object
    // without the arm is caught at load rather than here.
    let mut mixed = segmented_prog(&[DevOp::FlashPrefill, DevOp::FlashMlaPrefill], &[0, 0]);
    mixed.insts[1].i[3] = 0x8000_0000;
    assert_eq!(
        derive_segments_for(&mixed, false).unwrap(),
        [4],
        "a FlashPrefill segment is class 4 on its own; the NoPE MLA riding it is now servable"
    );
}

#[test]
fn xreduce_wave_rs_routes_only_a_marked_pure_segment() {
    let mut pure = segmented_prog(&[DevOp::XReduceTwoShot], &[0]);
    pure.stream[0].flags |= packet::dev::SE_XR_WAVE_RS;
    assert_eq!(derive_segments_for(&pure, false).unwrap(), [19]);

    let mut mixed = segmented_prog(&[DevOp::XReduceTwoShot, DevOp::RmsNorm], &[0, 0]);
    for e in &mut mixed.stream {
        e.flags |= packet::dev::SE_XR_WAVE_RS;
    }
    let err = derive_segments_for(&mixed, false)
        .expect_err("a marked mixed segment must not reach the specialist object");
    assert!(err.to_string().contains("not pure XReduceTwoShot"));

    mixed.stream[1].flags &= !packet::dev::SE_XR_WAVE_RS;
    let err = derive_segments_for(&mixed, false)
        .expect_err("an incompletely marked segment must not reach the specialist object");
    assert!(err.to_string().contains("not pure XReduceTwoShot"));
}

#[test]
fn kda_wave_items_routes_only_a_marked_pure_segment() {
    let mut pure = segmented_prog(&[DevOp::KdaChunkIntra], &[0]);
    pure.stream[0].flags |= packet::dev::SE_KDA_INTRA_WAVE_ITEMS;
    assert_eq!(derive_segments_for(&pure, false).unwrap(), [20]);

    let mut mixed = segmented_prog(&[DevOp::KdaChunkIntra, DevOp::RmsNorm], &[0, 0]);
    for e in &mut mixed.stream {
        e.flags |= packet::dev::SE_KDA_INTRA_WAVE_ITEMS;
    }
    let err = derive_segments_for(&mixed, false)
        .expect_err("a marked mixed segment must not reach the wave-item object");
    assert!(err.to_string().contains("not pure KdaChunkIntra"));

    mixed.stream[1].flags &= !packet::dev::SE_KDA_INTRA_WAVE_ITEMS;
    let err = derive_segments_for(&mixed, false)
        .expect_err("an incompletely marked segment must not reach the wave-item object");
    assert!(err.to_string().contains("not pure KdaChunkIntra"));
}

#[test]
fn kda_carry_regstate_routes_only_a_marked_exact_singleton() {
    let mut prog = segmented_prog(&[DevOp::KdaChunkWu, DevOp::KdaChunkCarry], &[0, 1]);
    prog.t = 8192;
    prog.insts[1].blocks = 256;
    prog.insts[1].t = [0, 1, 2, 3, 4, 5, 6, 7];
    prog.insts[1].i = [8192, 12, 128, 128, 1, 0, 0, 0];
    prog.insts[1].fj[0] = (1.0 / 128.0f32.sqrt()).to_bits();
    prog.stream[1].flags |= packet::dev::SE_KDA_CARRY_REGSTATE;
    let devp: Vec<_> = (0..8)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();
    let classes = derive_segments_for(&prog, false).unwrap();
    assert_eq!(classes, [8, 23]);
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    promote_kda_carry_regstate_routes(&prog, &classes, &devp, &mut routes).unwrap();
    let PrefillSegmentRoute::KdaChunkCarryRegstate { args, grid, .. } = routes[1] else {
        panic!("a marked exact carry must select the regstate object")
    };
    assert_eq!(
        (args.t, args.heads, args.dim, args.value_dim, grid),
        (8192, 12, 128, 128, 96)
    );
    assert_eq!((args.out, args.g), (0x1000, 0x1700));
    assert!(matches!(routes[0], PrefillSegmentRoute::Interpreter));

    prog.insts[1].i[4] = 0;
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    let err = promote_kda_carry_regstate_routes(&prog, &classes, &devp, &mut routes)
        .expect_err("a non-qpre carry must not reach the regstate object");
    assert!(err.to_string().contains("not an exact qpre"));

    let mut mixed = segmented_prog(&[DevOp::KdaChunkCarry, DevOp::RmsNorm], &[0, 0]);
    for e in &mut mixed.stream {
        e.flags |= packet::dev::SE_KDA_CARRY_REGSTATE;
    }
    assert!(derive_segments_for(&mixed, false).is_err());
}

#[test]
fn kda_wu_lean_routes_the_marked_pair_and_feeds_the_regstate_carry() {
    let mut prog = segmented_prog(&[DevOp::KdaChunkWu, DevOp::KdaChunkCarry], &[0, 1]);
    prog.t = 8192;
    for d in &mut prog.insts {
        d.blocks = 256;
        d.i = [8192, 12, 128, 128, 1, 0, 0, 0];
        d.fj[0] = (1.0 / 128.0f32.sqrt()).to_bits();
    }
    prog.insts[0].t = [0, 1, 2, 3, 4, 5, 6, 7];
    prog.insts[0].i[5] = 1;
    prog.insts[1].t = [8, 9, 7, 3, 0, 1, 10, 5];
    prog.stream[0].flags |= packet::dev::SE_KDA_WU_LEAN;
    prog.stream[1].flags |= packet::dev::SE_KDA_CARRY_REGSTATE;
    let devp: Vec<_> = (0..11)
        .map(|i| DeviceMem::view(0x1000 + i as u64 * 0x100, 0x100))
        .collect();
    let classes = derive_segments_for(&prog, false).unwrap();
    assert_eq!(classes, [25, 23]);
    let half = kda_keyfeed_scratch_half_bytes(std::slice::from_ref(&prog)).unwrap();
    assert_eq!(half, 8192 * 12 * 128 * 2);

    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    promote_kda_carry_regstate_routes(&prog, &classes, &devp, &mut routes).unwrap();
    promote_kda_wu_lean_routes(
        &prog,
        &classes,
        &devp,
        &mut routes,
        Some((0x8000_0000, half)),
    )
    .unwrap();
    let PrefillSegmentRoute::KdaChunkWuLean { args: wu, grid, .. } = routes[0] else {
        panic!("a marked exact Wu must select the lean object")
    };
    assert_eq!(
        (wu.w, wu.q, wu.beta, wu.key_hi, wu.key_lo, wu.t, grid),
        (
            0x1000,
            0x1700,
            0x1600,
            0x8000_0000,
            0x8000_0000 + half,
            8192,
            768
        )
    );
    let PrefillSegmentRoute::KdaChunkCarryKeyfeed { args: c, grid, .. } = routes[1] else {
        panic!("a key-emitting Wu must convert its regstate carry into the key-fed carry")
    };
    assert_eq!(
        (c.out, c.g, c.key_hi, c.key_lo, c.t, grid),
        (0x1800, 0x1500, 0x8000_0000, 0x8000_0000 + half, 8192, 96)
    );
    rebase_kda_key_factor_routes(&mut routes, 777);
    assert!(matches!(routes[0], PrefillSegmentRoute::KdaChunkWuLean { args, .. } if args.t == 777));
    assert!(
        matches!(routes[1], PrefillSegmentRoute::KdaChunkCarryKeyfeed { args, .. } if args.t == 777)
    );

    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    promote_kda_carry_regstate_routes(&prog, &classes, &devp, &mut routes).unwrap();
    let err = promote_kda_wu_lean_routes(&prog, &classes, &devp, &mut routes, None)
        .expect_err("a key-emitting Wu without a scratch pair must refuse");
    assert!(err.to_string().contains("no key-factor scratch"));

    prog.insts[0].i[5] = 0;
    assert_eq!(
        kda_keyfeed_scratch_half_bytes(std::slice::from_ref(&prog)).unwrap(),
        0
    );
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    promote_kda_carry_regstate_routes(&prog, &classes, &devp, &mut routes).unwrap();
    promote_kda_wu_lean_routes(&prog, &classes, &devp, &mut routes, None).unwrap();
    assert!(
        matches!(routes[0], PrefillSegmentRoute::KdaChunkWuLean { args, .. } if args.key_hi == 0)
    );
    assert!(matches!(
        routes[1],
        PrefillSegmentRoute::KdaChunkCarryRegstate { .. }
    ));

    prog.insts[0].i[4] = 0;
    let mut routes = vec![PrefillSegmentRoute::Interpreter; 2];
    let err = promote_kda_wu_lean_routes(&prog, &classes, &devp, &mut routes, None)
        .expect_err("a non-qpre Wu must not reach the lean object");
    assert!(err.to_string().contains("not an exact qpre"));
}

#[test]
fn raw_mla_v2_object_requires_exact_capabilities() {
    let object = Path::new("interp_mla_v2_sv_gq.elf");
    let exact = [MLA_PF_V2_SYM, MLA_PF_V2_SV_RAW_SYM, L2_DISPATCH_SYM];
    assert!(check_mla_v2_sv_raw_symbols(&exact, object, true).is_ok());

    for bad in [
        vec![MLA_PF_V2_SYM, L2_DISPATCH_SYM],
        vec![MLA_PF_V2_SYM, MLA_PF_V2_SV_RAW_SYM],
        vec![
            MLA_PF_V2_SYM,
            MLA_PF_V2_SV_RAW_SYM,
            L2_DISPATCH_SYM,
            MLA_PF_V2_FP8_SYM,
        ],
        vec![
            MLA_PF_V2_SYM,
            MLA_PF_V2_SV_RAW_SYM,
            L2_DISPATCH_SYM,
            PACKED_PREFILL_MLA_FLASH_SEG_SYM,
        ],
    ] {
        assert!(check_mla_v2_sv_raw_symbols(&bad, object, true).is_err());
    }
}

#[test]
fn required_flash_object_load_errors_are_not_swallowed() {
    let required = resolve_flash_object_load::<()>(
        Err(RuntimeError::Device("missing V2 marker".into())),
        true,
    )
    .expect_err("required V2 flash object must fail closed");
    assert!(required.to_string().contains("missing V2 marker"));

    let optional = resolve_flash_object_load::<()>(
        Err(RuntimeError::Device(
            "optional flash object unavailable".into(),
        )),
        false,
    )
    .expect("ordinary flash object may fall back");
    assert!(optional.is_none());
}

#[test]
fn raw_mla_v2_route_is_dense_bf16_pure_and_machine_filling() {
    let mut pure = segmented_prog(&[DevOp::FlashMlaPrefill], &[0]);
    pure.insts[0].t[7] = packet::dev::TENSOR_NONE16;
    assert_eq!(derive_raw_mla_v2_segments(&pure).unwrap(), [true]);

    let mut gathered = segmented_prog(&[DevOp::FlashMlaPrefill], &[0]);
    gathered.insts[0].t[7] = 7;
    assert_eq!(derive_raw_mla_v2_segments(&gathered).unwrap(), [false]);

    let mut fp8 = segmented_prog(&[DevOp::FlashMlaPrefillFp8], &[0]);
    fp8.insts[0].t[7] = packet::dev::TENSOR_NONE16;
    assert_eq!(derive_raw_mla_v2_segments(&fp8).unwrap(), [false]);

    let mut mixed = segmented_prog(&[DevOp::FlashMlaPrefill, DevOp::Gemv], &[0, 0]);
    mixed.insts[0].t[7] = packet::dev::TENSOR_NONE16;
    assert_eq!(derive_raw_mla_v2_segments(&mixed).unwrap(), [false]);

    pure.t = 1024;
    assert_eq!(derive_raw_mla_v2_segments(&pure).unwrap(), [false]);
}

#[test]
fn packed_operator_families_require_pure_segments() {
    let pure = segmented_prog(
        &[
            DevOp::RmsNorm,
            DevOp::HeadNormRope,
            DevOp::FlashMlaPrefill,
            DevOp::KdaConv3,
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ],
        &[0, 0, 1, 2, 2, 2, 2, 2],
    );
    assert_eq!(derive_packed_segment_families(&pure).unwrap(), [5, 6, 7]);
    let routes: Vec<_> = derive_packed_segment_families(&pure)
        .unwrap()
        .into_iter()
        .map(|family| packed_segment_route(true, family, true, true, true).unwrap())
        .collect();
    assert_eq!(
        routes,
        [
            PackedSegmentRoute::MlaNorm,
            PackedSegmentRoute::MlaFlash,
            PackedSegmentRoute::Kda,
        ]
    );
    assert!(packed_kda_compatible(&pure));
    assert_eq!(derive_segments_for(&pure, false).unwrap(), [8, 8, 8]);
    assert_eq!(derive_segments_for(&pure, true).unwrap(), [8, 4, 8]);

    let mixed = segmented_prog(
        &[DevOp::RmsNorm, DevOp::Gemv, DevOp::KdaStateStepG],
        &[0, 0, 1],
    );
    assert_eq!(derive_packed_segment_families(&mixed).unwrap(), [0, 7]);
    assert!(!packed_family_segments_cover(&mixed, &[0, 7], &[5]));
    assert!(packed_family_segments_cover(&mixed, &[0, 7], &[7]));
    let none = segmented_prog(&[DevOp::Gemv], &[0]);
    assert!(!packed_family_segments_cover(&none, &[0], &[5, 6]));
}

#[test]
fn packed_chunk_kda_requires_a_complete_operator_group() {
    let complete = segmented_prog(
        &[
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ],
        &[0, 0, 0, 0],
    );
    assert!(packed_kda_compatible(&complete));

    let partial = segmented_prog(&[DevOp::KdaChunkPrepare, DevOp::KdaChunkCarry], &[0, 0]);
    assert!(!packed_kda_compatible(&partial));

    let serial = segmented_prog(&[DevOp::KdaConv3, DevOp::KdaStateStepG], &[0, 0]);
    assert!(packed_kda_compatible(&serial));
    let double_buffered = segmented_prog(&[DevOp::KdaConvStateStepG], &[0]);
    assert!(!packed_kda_compatible(&double_buffered));
}

#[test]
fn packed_family_route_never_falls_back_for_a_consumer_segment() {
    assert_eq!(
        packed_segment_route(false, 5, false, false, false).unwrap(),
        PackedSegmentRoute::Primary
    );
    assert_eq!(
        packed_segment_route(true, 0, false, false, false).unwrap(),
        PackedSegmentRoute::Primary
    );
    assert_eq!(
        packed_segment_route(true, 5, true, false, false).unwrap(),
        PackedSegmentRoute::MlaNorm
    );
    assert_eq!(
        packed_segment_route(true, 6, false, true, false).unwrap(),
        PackedSegmentRoute::MlaFlash
    );
    assert_eq!(
        packed_segment_route(true, 7, false, false, true).unwrap(),
        PackedSegmentRoute::Kda
    );
    assert_eq!(
        packed_segment_route(false, 7, false, false, true).unwrap(),
        PackedSegmentRoute::Kda
    );
    assert_eq!(
        packed_segment_route(false, 7, false, false, false).unwrap(),
        PackedSegmentRoute::Primary
    );
    assert!(packed_segment_route(true, 5, false, true, true).is_err());
    assert!(packed_segment_route(true, 9, true, true, true).is_err());
    assert!(check_packed_family_kv_encoding(false, false).is_ok());
    assert!(check_packed_family_kv_encoding(true, true).is_ok());
    assert!(check_packed_family_kv_encoding(false, true).is_err());
    assert!(check_packed_family_kv_encoding(true, false).is_err());
}

#[test]
fn state_clear_ranges_cover_each_slot_stride_once() {
    let devp = vec![
        DeviceMem::view(0x1000, 3 * STATE_CLEAR_CHUNK),
        DeviceMem::view(0x20_0000, 24 * 1024),
    ];
    let ranges = state_clear_ranges(&devp, &[(0, 3 * STATE_CLEAR_CHUNK), (1, 24 * 1024)]).unwrap();
    assert_eq!(ranges.len(), 4);
    assert_eq!(ranges[0].base, 0x1000);
    assert_eq!(ranges[2].base, 0x1000 + 2 * STATE_CLEAR_CHUNK);
    assert!(ranges[..3]
        .iter()
        .all(|r| r.slot_stride == 3 * STATE_CLEAR_CHUNK));
    assert_eq!(
        ranges.iter().map(|r| r.words as u64 * 4).sum::<u64>(),
        3 * STATE_CLEAR_CHUNK + 24 * 1024
    );
}

#[test]
fn tp_counter_banks_are_clean_on_first_use_and_alternate() {
    let state = CounterBankState::new();
    assert_eq!(state.current(), 0);
    assert!(state.inactive_ready());

    let mut used = Vec::new();
    for _ in 0..3 {
        assert_eq!(state.begin_tp(true), Ok(true));
        used.push(state.current());
        assert!(!state.inactive_ready());
        let executed = state.current();
        state.mark_inactive_ready();
        assert_eq!(state.current(), executed, "re-arm must not move snapshots");
    }
    assert_eq!(used, [1, 0, 1]);
}

#[test]
fn tp_counter_bank_state_is_per_program() {
    let rung8 = CounterBankState::new();
    let rung32 = CounterBankState::new();

    assert_eq!(rung8.begin_tp(true), Ok(true));
    rung8.mark_inactive_ready();
    assert_eq!(rung8.current(), 1);
    assert_eq!(rung32.current(), 0);

    assert_eq!(rung32.begin_tp(true), Ok(true));
    assert_eq!(rung32.current(), 1);
    assert_eq!(rung8.current(), 1);
}

#[test]
fn tp_counter_single_bank_mode_requires_synchronous_rearm() {
    let state = CounterBankState::new();
    assert_eq!(state.begin_tp(false), Ok(false));
    assert_eq!(state.current(), 0);
    assert!(state.inactive_ready());
}

#[test]
fn tp_counter_bank_refuses_stale_inactive_bank() {
    let state = CounterBankState::new();
    assert_eq!(state.begin_tp(true), Ok(true));
    assert_eq!(state.begin_tp(true), Err(()));
    assert_eq!(state.current(), 1, "failed selection must not change banks");

    state.mark_inactive_ready();
    assert_eq!(state.begin_tp(true), Ok(true));
    assert_eq!(state.current(), 0);
}

/// The kernarg slice must cover the WHOLE struct.
///
/// It was a literal `128`, and appending `seg_ofs` made the struct 136: the
/// launcher copied 128 bytes and then wrote the COv5 implicit block at
/// `(args_size + 7) & !7 == 128`, ON TOP of the missing field. The
/// interpreter dereferenced the grid-dimension word as a device pointer and
/// every static-scheduler prefill died with `Memory access fault`. Nothing
/// caught it at compile time because the literal is still a valid length.
#[test]
fn the_kernarg_slice_is_the_whole_struct() {
    // SAFETY: `DevProgram` is a `#[repr(C)]` POD of integers and raw
    // pointers, so all-zeroes is a valid value (null pointers included —
    // this instance is only ever measured, never dispatched).
    let p: DevProgram = unsafe { std::mem::zeroed() };
    assert_eq!(kernarg_bytes(&p).len(), std::mem::size_of::<DevProgram>());
}

#[test]
fn compact_audit_patches_only_tp_collectives() {
    let mut insts = [
        DevInst64 {
            op: DevOp::XReduce as u16,
            i: [0, 0, 0, 0, 0, 0, 0, 11],
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::XReduceTwoShot as u16,
            i: [0, 0, 0, 0, 0, 0, 23, 29],
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::XReduceAddNorm as u16,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::XArgmaxFin as u16,
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::Gemm as u16,
            ..Default::default()
        },
    ];
    patch_tp_xaudit(&mut insts, 464);
    assert!(insts[..4].iter().all(|d| d.fj[2] == 465));
    assert_eq!(insts[4].fj[2], 0);
    assert_eq!(insts[0].i[7], 11);
    assert_eq!(&insts[1].i[6..=7], &[23, 29]);
}

/// The flash object follows the PREFILL scheduler: a flash segment IS a
/// prefill segment. Pairing it with the decode choice loads an object whose
/// scheduling loop does not match the stream it is handed.
#[test]
fn object_names_match_the_shipped_set() {
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::None,
            Sched::GlobalQueue
        ),
        "interp_prefill_gq.elf"
    );
    assert_eq!(
        object_name(
            Phase::Decode,
            Variant::Bf16,
            PrefillArm::None,
            Sched::Static
        ),
        "interp_decode.elf"
    );
    assert_eq!(
        object_name(
            Phase::Decode,
            Variant::Fp8Kv,
            PrefillArm::None,
            Sched::GlobalQueue
        ),
        "interp_decode_fp8kv_gq.elf"
    );
    assert_eq!(
        object_name(Phase::Decode, Variant::Fp8, PrefillArm::None, Sched::Static),
        "interp_decode_fp8.elf"
    );
    // There is no fp8-WEIGHT flash object — flash only varies on KV — so an
    // fp8 packet must fall back to the bf16 flash object rather than ask
    // for a file that was never built.
    assert_eq!(
        object_name(Phase::Flash, Variant::Fp8, PrefillArm::None, Sched::Static),
        "interp_flash.elf"
    );
    assert_eq!(
        object_name(
            Phase::Flash,
            Variant::Fp8Kv,
            PrefillArm::None,
            Sched::GlobalQueue
        ),
        "interp_flash_fp8kv_gq.elf"
    );
    // The MLA/MoE-prefill axis is PREFILL-only — no decode or flash twin
    // exists (`scripts/build_gfx950.sh` never builds one) — so a non-None
    // arm on those phases must not leak into the filename.
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::Mla,
            Sched::GlobalQueue
        ),
        "interp_prefill_mla_gq.elf"
    );
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::Mla,
            Sched::Static
        ),
        "interp_prefill_mla.elf"
    );
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::MlaMoe,
            Sched::GlobalQueue
        ),
        "interp_prefill_mla_moe_gq.elf"
    );
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::MlaMoe,
            Sched::Static
        ),
        "interp_prefill_mla_moe.elf"
    );
    assert_eq!(
        object_name(
            Phase::Decode,
            Variant::Bf16,
            PrefillArm::MlaMoe,
            Sched::Static
        ),
        "interp_decode.elf"
    );
    assert_eq!(
        object_name(
            Phase::Flash,
            Variant::Bf16,
            PrefillArm::MlaMoe,
            Sched::Static
        ),
        "interp_flash.elf"
    );
    // KIMI-K3 is the exception, and deliberately: `PLOW_K3` is a MODEL axis, not a
    // prefill-kernel one, and `interp_decode_k3.elf` is a real row in
    // `runtime/CMakeLists.txt`. A K3 decode packet handed the plain `interp_decode.elf` has no
    // `case` for AttnRes (104), SituGlu (105), MlaOutGate (106) or the KDA mixer, and this
    // interpreter's dispatch `default:` writes NOTHING.
    assert_eq!(
        object_name(Phase::Prefill, Variant::Bf16, PrefillArm::K3, Sched::Static),
        "interp_prefill_k3.elf"
    );
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::K3Moe,
            Sched::Static
        ),
        "interp_prefill_k3_moe.elf"
    );
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Bf16,
            PrefillArm::K3MoeA4w4,
            Sched::GlobalQueue
        ),
        "interp_prefill_k3_moe_a4w4_gq.elf"
    );
    // Decode collapses ALL THREE K3 arms onto one object: the grouped-MoE ops are
    // prefill-only, so there is nothing for a `_k3_moe` decode object to contain that `_k3`
    // does not, and no such row exists in runtime/CMakeLists.txt.
    for a in [PrefillArm::K3, PrefillArm::K3Moe, PrefillArm::K3MoeA4w4] {
        assert_eq!(
            object_name(Phase::Decode, Variant::Bf16, a, Sched::Static),
            "interp_decode_k3.elf"
        );
        // No K3 flash object, and no packet can ask for one: K3 is NoPE MLA + KDA and emits
        // no `FlashPrefill` at any head dim.
        assert_eq!(
            object_name(Phase::Flash, Variant::Bf16, a, Sched::Static),
            "interp_flash.elf"
        );
    }
    assert_eq!(
        object_name(
            Phase::Decode,
            Variant::Fp8Kv,
            PrefillArm::K3,
            Sched::GlobalQueue
        ),
        "interp_decode_fp8kv_k3_gq.elf"
    );
    assert_eq!(
        object_name(
            Phase::Prefill,
            Variant::Fp8Kv,
            PrefillArm::K3MoeA4w4,
            Sched::GlobalQueue,
        ),
        "interp_prefill_fp8kv_k3_moe_a4w4_gq.elf"
    );
}

/// Synthetic packets exercising `PrefillArm::detect` — the axis whose
/// absence let a GLM-5.2 prefill packet load `interp_prefill_gq.elf`
/// (no MLA/MoE arms) and silently produce all-zero activations.
#[test]
fn prefill_arm_detect_selects_the_right_variant() {
    fn prog_with_ops(ops: &[DevOp]) -> DevProg {
        let insts = ops
            .iter()
            .map(|&op| DevInst64 {
                op: op as u16,
                ..Default::default()
            })
            .collect();
        DevProg {
            t: 1,
            packed_prefill_only: false,
            token_batch_body: false,
            n_counter: 0,
            insts,
            stream: Vec::new(),
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains: 0,
        }
    }

    // MoE-prefill opcodes present (even alongside MLA ones) => MlaMoe.
    let moe_progs = vec![prog_with_ops(&[
        DevOp::FlashMlaPrefill,
        DevOp::MoeRouterTopkPf,
        DevOp::MoeAlignPf,
        DevOp::MoeGroupGluPf,
        DevOp::MoeGroupDownPf,
        DevOp::MoeCombinePf,
    ])];
    assert_eq!(PrefillArm::detect(&moe_progs), PrefillArm::MlaMoe);

    // Only MLA-prefill opcodes => Mla, not MlaMoe.
    let mla_progs = vec![prog_with_ops(&[
        DevOp::FlashMlaPrefill,
        DevOp::FlashGatherPrefill,
        DevOp::MlaMergeFold,
    ])];
    assert_eq!(PrefillArm::detect(&mla_progs), PrefillArm::Mla);

    // Decode-only packet (no prefill opcodes at all) => None, unchanged.
    let decode_only = vec![prog_with_ops(&[DevOp::Embed, DevOp::Gemv, DevOp::RmsNorm])];
    assert_eq!(PrefillArm::detect(&decode_only), PrefillArm::None);

    // The opcodes live ONLY in an earlier program (a prefill bucket ahead
    // of the decode program) — detect must scan every program, not just
    // `progs.last()`, or this regresses to the exact bug being fixed.
    let bucket_then_decode = vec![
        prog_with_ops(&[
            DevOp::FlashMlaPrefill,
            DevOp::MoeRouterTopkPf,
            DevOp::MoeAlignPf,
            DevOp::MoeGroupGluPf,
            DevOp::MoeGroupDownPf,
            DevOp::MoeCombinePf,
        ]),
        prog_with_ops(&[DevOp::Embed, DevOp::Gemv, DevOp::RmsNorm]),
    ];
    assert_eq!(PrefillArm::detect(&bucket_then_decode), PrefillArm::MlaMoe);

    // KIMI-K3. The block ops SUPERSEDE both — `_hs_ax_mla_k3` composes PLOW_MLA_PREFILL with
    // PLOW_K3, because K3's full-attention layers are MLA — and without this axis a K3 blob
    // resolves to `interp_prefill_mla_moe.elf`, which has no `case` for any of them.
    //
    // The grouped GEMMs carry the expert ENCODING, and it selects an OBJECT: MXFP4 needs the
    // A4W4 body, which is compiled only under `PLOW_MOE_PF_A4W4`. Wrong in both directions —
    // an mxfp4 packet on the plain object takes `moe_pf_refuse`, a bf16 packet on the a4w4
    // object gets 140 KB of arms it never runs — so the field is read, not assumed.
    let k3_moe = |enc: u32| {
        let grouped = |op: DevOp| {
            let mut i = [0u32; 8];
            i[MOE_PF_ENC_SLOT] = enc;
            DevInst64 {
                op: op as u16,
                i,
                ..Default::default()
            }
        };
        let mut first = prog_with_ops(&[
            DevOp::AttnRes,
            DevOp::SituGlu,
            DevOp::KdaStateStepG,
            DevOp::FlashMlaPrefill,
        ]);
        first.insts.push(grouped(DevOp::MoeGroupGluPf));
        first.insts.push(grouped(DevOp::MoeGroupDownPf));
        vec![
            first,
            prog_with_ops(&[DevOp::AttnRes, DevOp::SituGlu, DevOp::Gemv]),
        ]
    };
    assert_eq!(
        PrefillArm::detect(&k3_moe(2)),
        PrefillArm::K3MoeA4w4,
        "PLOW_MOE_ENC_MXFP4"
    );
    assert_eq!(
        PrefillArm::detect(&k3_moe(0)),
        PrefillArm::K3Moe,
        "bf16 experts"
    );
    assert_eq!(
        PrefillArm::detect(&k3_moe(1)),
        PrefillArm::K3Moe,
        "block-fp8 experts"
    );

    // A DECODE-ONLY K3 blob still selects the K3 objects. This is what makes
    // `interp_decode_k3` reachable at all, and it is not a corner: K3 emitted decode-only for
    // its whole bring-up, and `K3_PREFILL=0` still does.
    let k3_decode = vec![prog_with_ops(&[
        DevOp::Embed,
        DevOp::AttnRes,
        DevOp::KdaConv3,
        DevOp::MlaOutGate,
        DevOp::Gemv,
    ])];
    assert_eq!(PrefillArm::detect(&k3_decode), PrefillArm::K3);

    // The MLA-only K3 bucket (attention emitted, FFN still on the decode ops) is `K3`, not
    // `K3MoeA4w4`: the grouped chain is what `PLOW_MOE_PREFILL` builds and it is absent here.
    let k3_attn = vec![prog_with_ops(&[
        DevOp::AttnRes,
        DevOp::FlashMlaPrefill,
        DevOp::MlaMergeFold,
    ])];
    assert_eq!(PrefillArm::detect(&k3_attn), PrefillArm::K3);
}

/// A prog carrying `ops`, each instruction asking for `m` GEMV rows.
fn prog_gemv(ops: &[DevOp], m: u32) -> DevProg {
    let insts = ops
        .iter()
        .map(|&op| {
            let mut i = [0u32; 8];
            i[0] = m;
            DevInst64 {
                op: op as u16,
                i,
                ..Default::default()
            }
        })
        .collect();
    DevProg {
        t: m,
        packed_prefill_only: false,
        token_batch_body: false,
        n_counter: 0,
        insts,
        stream: Vec::new(),
        stream_ofs: Vec::new(),
        stream_len: Vec::new(),
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: Vec::new(),
        gq_seg_ofs: Vec::new(),
        l2_domains: 0,
    }
}

/// A packet dispatching a K3/KDA op against an object built without `PLOW_K3` is REFUSED.
///
/// The gating of those seven arms (interp.hip) is what makes this necessary: AMD's dispatch
/// `default:` writes nothing, so without this check the op would silently leave its output
/// untouched and the run would finish fluently on uninitialised memory — the exact failure
/// class `GFX950_DISPATCHED` was introduced to end.
#[test]
fn a_k3_packet_against_an_object_without_the_arms_is_refused() {
    let obj = Path::new("interp_decode.elf");
    let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950"];
    let with_k3 = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950", K3_ARMS_SYM];

    for &op in K3_ARM_OPS {
        let pkt = vec![prog_gemv(&[op], 1)];
        assert_eq!(
            required_k3_op(&pkt),
            Some(op),
            "{op:?} must be recognised as a K3 arm"
        );

        let e = check_k3_arms(&bare, obj, required_k3_op(&pkt))
            .expect_err("a K3 op against an object with no K3 arms must be refused");
        let msg = e.to_string();
        // The refusal has to name the op, the marker, and the remedy — the object is not on a
        // device yet, so this message is the only thing the operator gets.
        assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
        assert!(
            msg.contains(K3_ARMS_SYM),
            "must name the missing marker: {msg}"
        );
        assert!(
            msg.contains("PLOW_K3"),
            "must name the flag to rebuild with: {msg}"
        );

        // The same packet against an object that advertises the arms is fine.
        assert!(check_k3_arms(&with_k3, obj, required_k3_op(&pkt)).is_ok());
    }

    // A packet with no K3 op at all is untouched on either object — gating must not refuse
    // the Gemma/GLM packets that are the whole reason the arms were taken out.
    let plain = vec![prog_gemv(&[DevOp::Gemv, DevOp::RmsNorm], 1)];
    assert_eq!(required_k3_op(&plain), None);
    assert!(check_k3_arms(&bare, obj, required_k3_op(&plain)).is_ok());
    assert!(check_k3_arms(&with_k3, obj, required_k3_op(&plain)).is_ok());
}

/// A packet dispatching a Qwen GDN op against an object built without `PLOW_QWEN_GDN` is
/// REFUSED, for exactly the reason the K3 twin above is: the family is compiled out by default
/// and the dispatch `default:` writes nothing, so the alternative to this refusal is a run that
/// completes over untouched Gated DeltaNet outputs.
///
/// `QwenGdnPrefill` is checked NOT to trip it: no object symbol can answer for a host-side
/// dispatch, and the emit refuses that capability by name instead.
#[test]
fn a_qwen_gdn_packet_against_an_object_without_the_arms_is_refused() {
    let obj = Path::new("interp_decode.elf");
    let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950"];
    let armed = vec![
        "plow_gemv_mm_cap_1",
        "plow_interp_dec_gfx950",
        QWEN_GDN_ARMS_SYM,
    ];

    for &op in QWEN_GDN_ARM_OPS {
        let pkt = vec![prog_gemv(&[op], 1)];
        assert_eq!(
            required_qwen_gdn_op(&pkt),
            Some(op),
            "{op:?} must be recognised as a Qwen GDN arm"
        );
        let e = check_qwen_gdn_arms(&bare, obj, required_qwen_gdn_op(&pkt))
            .expect_err("a Qwen GDN op against an unarmed object must be refused");
        let msg = e.to_string();
        assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
        assert!(
            msg.contains(QWEN_GDN_ARMS_SYM),
            "must name the missing marker: {msg}"
        );
        assert!(
            msg.contains("PLOW_QWEN_GDN"),
            "must name the flag to rebuild with: {msg}"
        );
        assert!(check_qwen_gdn_arms(&armed, obj, required_qwen_gdn_op(&pkt)).is_ok());
    }

    // The host-side prefill op has no arm to advertise, so it must not be claimed by this gate.
    let pf = vec![prog_gemv(&[DevOp::QwenGdnPrefill], 1)];
    assert_eq!(required_qwen_gdn_op(&pf), None);

    let plain = vec![prog_gemv(&[DevOp::Gemv, DevOp::RmsNorm], 1)];
    assert_eq!(required_qwen_gdn_op(&plain), None);
    assert!(check_qwen_gdn_arms(&bare, obj, required_qwen_gdn_op(&plain)).is_ok());
}

/// A sparse (DSA) MLA-prefill blob is refused unless BOTH the V2 routing and the gathered
/// arm are present.
///
/// This is the strongest refusal in the flash-object set because its fallback is the least
/// visible one in the tree. A missing K3 arm leaves a buffer untouched; a missing ofold arm
/// feeds bf16 garbage to a GEMM. A missing gathered arm does neither — the dense body runs
/// to completion, ignores `t[7]`, and hands back full causal attention for a model trained
/// sparse. Nothing about the output looks wrong.
///
/// Both halves are tested independently because they fail for different reasons and an
/// operator has a different remedy for each: the routing is a serve-time env var, the arm is
/// a rebuild.
#[test]
fn dsa_decode_score_requires_the_cdna3_lds_fix() {
    let prog = segmented_prog(&[DevOp::IndexScore], &[0]);
    assert!(
        check_dsa_decode_batch(&[], Path::new("old.elf"), std::slice::from_ref(&prog), true)
            .is_err()
    );
    assert!(check_dsa_decode_batch(
        &[],
        Path::new("old.elf"),
        std::slice::from_ref(&prog),
        false
    )
    .is_ok());
    assert!(check_dsa_decode_batch(
        &["plow_dsa_decode_batch_arm"],
        Path::new("new.elf"),
        &[prog],
        true
    )
    .is_ok());
}

#[test]
fn sparse_fp8_rejects_stale_objects_and_invalid_handles() {
    let mut p = segmented_prog(&[DevOp::FlashMlaDecodeFp8], &[0]);
    p.insts[0].i = [16, 8, 81920, 0, 16, u32::MAX, 2048, 4];
    p.insts[0].t = [0; 8];
    p.insts[0].fj = [0.0625f32.to_bits(), 1, 0];
    let tensors = vec![crate::asset::devblob::DevTensor {
        name: "large".into(),
        bytes: 16 * 81920 * 512,
        init: None,
    }];
    assert!(check_sparse_fp8_packet(std::slice::from_ref(&p), &tensors, "gfx942").is_ok());
    assert!(check_sparse_fp8_packet(std::slice::from_ref(&p), &tensors, "gfx950").is_err());
    assert!(
        check_sparse_fp8_object(&[], Path::new("old"), std::slice::from_ref(&p), true).is_err()
    );
    assert!(check_sparse_fp8_object(
        &["plow_mla_sparse_fp8_decode_arm"],
        Path::new("new"),
        std::slice::from_ref(&p),
        true
    )
    .is_ok());
    p.insts[0].fj[1] = u32::MAX;
    assert!(check_sparse_fp8_packet(std::slice::from_ref(&p), &tensors, "gfx942").is_err());
    p.insts[0].fj[1] = 0;
    assert!(check_sparse_fp8_object(&[], Path::new("old"), &[p], true).is_ok());
}

#[test]
fn local_dsa_selection_checks_rows_operands_and_object() {
    let mut prog = segmented_prog(&[DevOp::IndexSelect], &[0]);
    prog.t = 8;
    let inst = &mut prog.insts[0];
    inst.blocks = 8;
    inst.t = [0, 1, 65535, 65535, 2, 65535, 65535, 65535];
    inst.i = [81920, 2048, 0, 0, 1, 0, 0, 0];
    let mut tensors: Vec<_> = [8 * 2048 * 4, 8 * 81920 * 4, 8 * 4]
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| crate::asset::devblob::DevTensor {
            name: format!("act.{i}"),
            bytes,
            init: None,
        })
        .collect();
    for (arch, tp8, dec_ix) in [
        ("gfx942", true, 0),
        ("gfx950", true, 0),
        ("gfx942", false, 0),
        ("gfx942", true, 1),
    ] {
        assert_eq!(
            check_dsa_select_local(std::slice::from_ref(&prog), &tensors, dec_ix, arch, tp8)
                .is_ok(),
            arch == "gfx942" && tp8 && dec_ix == 0
        );
    }
    for rows in [1, 2, 4, 8, 16, 20, 32] {
        prog.t = rows;
        prog.insts[0].blocks = rows as u16;
        assert_eq!(
            check_dsa_select_local(std::slice::from_ref(&prog), &tensors, 0, "gfx942", true)
                .is_ok(),
            matches!(rows, 2 | 4 | 8)
        );
    }
    let mut wide: Vec<_> = [20 * 2048 * 4, 20 * 81920 * 4, 20 * 4]
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| crate::asset::devblob::DevTensor {
            name: format!("act.{i}"),
            bytes,
            init: None,
        })
        .collect();
    for rows in [16, 20, 21, 32] {
        prog.t = rows;
        prog.insts[0].blocks = rows as u16;
        assert_eq!(
            check_dsa_select_local(std::slice::from_ref(&prog), &wide, 0, "gfx942", true).is_ok(),
            matches!(rows, 16 | 20)
        );
    }
    prog.t = 20;
    prog.insts[0].blocks = 20;
    for operand in 0..3 {
        wide[operand].bytes -= 1;
        assert!(
            check_dsa_select_local(std::slice::from_ref(&prog), &wide, 0, "gfx942", true).is_err()
        );
        wide[operand].bytes += 1;
    }
    prog.t = 8;
    prog.insts[0].blocks = 8;
    tensors[2].bytes -= 1;
    assert!(
        check_dsa_select_local(std::slice::from_ref(&prog), &tensors, 0, "gfx942", true).is_err()
    );
    tensors[2].bytes += 1;
    for (slot, bad) in [(0, 2047), (1, 1024), (2, 2), (3, 1), (4, 2)] {
        let good = prog.insts[0].i[slot];
        prog.insts[0].i[slot] = bad;
        assert!(
            check_dsa_select_local(std::slice::from_ref(&prog), &tensors, 0, "gfx942", true)
                .is_err()
        );
        prog.insts[0].i[slot] = good;
    }
    let requires = packet_decode_arm_requirements(&[prog]);
    assert_eq!(requires, ["PLOW_DSA_SELECT_LOCAL=1"]);
    assert!(check_decode_object(
        &["plow_dsa_decode_batch_arm"],
        Path::new("old"),
        &requires,
        false,
        false
    )
    .is_err());
    assert!(check_decode_object(
        &["plow_dsa_select_local_arm"],
        Path::new("new"),
        &requires,
        false,
        false
    )
    .is_ok());
}

#[test]
fn dsa_decode_batch_refuses_an_object_without_row_offsets() {
    let mut prog = segmented_prog(&[DevOp::IndexSelect], &[0]);
    assert!(
        check_dsa_decode_batch(&[], Path::new("old.elf"), std::slice::from_ref(&prog), true)
            .is_ok()
    );
    prog.insts[0].i[3] = 1;
    assert!(
        check_dsa_decode_batch(&[], Path::new("old.elf"), std::slice::from_ref(&prog), true)
            .is_err()
    );
    assert!(check_dsa_decode_batch(
        &["plow_dsa_decode_batch_arm"],
        Path::new("new.elf"),
        &[prog],
        true
    )
    .is_ok());
}

#[test]
fn a_sparse_mla_prefill_blob_needs_both_the_v2_routing_and_the_gathered_arm() {
    let obj = Path::new("interp_flash.elf");
    let bare = ["plow_glm_ofold_arm"];
    let armed = ["plow_glm_ofold_arm", DSA_PF_ARM_SYM];
    let need = vec!["PLOW_DSA_PF_ARM=1".to_string()];
    let dense = vec!["PLOW_MLA_PF_NS=1".to_string()];

    // Routing off: refused even on an object that HAS the arm, because without the V2 split
    // the segment never reaches it.
    let msg = check_dsa_pf_arm(&armed, obj, &need, false)
        .expect_err("sparse without V2 routing must be refused")
        .to_string();
    assert!(
        msg.contains("PLOW_MLA_PF_V2=1"),
        "must name the remedy: {msg}"
    );
    assert!(msg.contains("DENSE"), "must say what goes wrong: {msg}");

    // Routing on, arm missing: refused, naming the marker and the build flag.
    let msg = check_dsa_pf_arm(&bare, obj, &need, true)
        .expect_err("sparse against an object with no gathered arm must be refused")
        .to_string();
    assert!(
        msg.contains(DSA_PF_ARM_SYM),
        "must name the missing marker: {msg}"
    );
    assert!(
        msg.contains("PLOW_DSA_PF_ARM=1"),
        "must name the flag to rebuild with: {msg}"
    );
    assert!(
        msg.contains(&obj.display().to_string()),
        "must name the object: {msg}"
    );

    // Both present: accepted.
    assert!(check_dsa_pf_arm(&armed, obj, &need, true).is_ok());

    // A blob that does not ask for the arm is untouched in every combination — the gate must
    // not refuse the dense GLM prefill that is the whole reason the arm is off by default.
    for syms in [bare.as_slice(), armed.as_slice()] {
        for v2 in [false, true] {
            assert!(check_dsa_pf_arm(syms, obj, &dense, v2).is_ok());
            assert!(check_dsa_pf_arm(syms, obj, &[], v2).is_ok());
        }
    }
}

#[test]
fn chunk_kda_requires_its_exact_object_marker() {
    let obj = Path::new("interp_prefill_k3.elf");
    let bare = [K3_ARMS_SYM];
    let armed = [K3_ARMS_SYM, KDA_CHUNK_SYM];
    let ops = [
        DevOp::KdaChunkPrepare,
        DevOp::KdaChunkIntra,
        DevOp::KdaChunkWu,
        DevOp::KdaChunkCarry,
    ];
    let pkt = vec![prog_gemv(&ops, 512)];
    assert_eq!(required_kda_chunk(&pkt), Some(DevOp::KdaChunkPrepare));
    let err = check_kda_chunk(&bare, obj, required_kda_chunk(&pkt)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains(KDA_CHUNK_SYM));
    assert!(msg.contains("PLOW_KDA_CHUNK"));
    assert!(check_kda_chunk(&armed, obj, required_kda_chunk(&pkt)).is_ok());

    let serial = vec![prog_gemv(&[DevOp::KdaStateStepG], 512)];
    assert_eq!(required_kda_chunk(&serial), None);
    assert!(check_kda_chunk(&bare, obj, None).is_ok());
}

#[test]
fn batched_mxfp4_moe_requires_a4w4_in_the_decode_object() {
    let obj = Path::new("interp_decode_k3.elf");
    let mut pkt = vec![prog_gemv(&[DevOp::MoeGroupGluPf, DevOp::MoeGroupDownPf], 4)];
    for i in &mut pkt[0].insts {
        i.i[MOE_PF_ENC_SLOT] = MOE_ENC_MXFP4;
    }
    let need = required_moe_pf_a4w4(&pkt);
    assert_eq!(need, Some(DevOp::MoeGroupGluPf));
    let bare = ["plow_k3_arms_1"];
    let armed = ["plow_k3_arms_1", MOE_PF_A4W4_SYM];
    let e = check_moe_pf_a4w4(&bare, obj, need).unwrap_err();
    assert!(e.to_string().contains("PLOW_MOE_PF_A4W4"));
    assert!(check_moe_pf_a4w4(&armed, obj, need).is_ok());

    pkt[0].insts[0].i[MOE_PF_ENC_SLOT] = 1;
    pkt[0].insts[1].i[MOE_PF_ENC_SLOT] = 0;
    assert_eq!(required_moe_pf_a4w4(&pkt), None);
    assert!(check_moe_pf_a4w4(&bare, obj, None).is_ok());
}

#[test]
fn kda_conv_step_db_requires_exact_packet_object_pairing() {
    let obj = Path::new("interp_decode_fp8kv_k3.elf");
    let bare = [K3_ARMS_SYM];
    let armed = [K3_ARMS_SYM, KDA_CONV_STEP_DB_SYM];
    let fused = vec![prog_gemv(&[DevOp::KdaConvStateStepG], 1)];
    let legacy = vec![prog_gemv(&[DevOp::KdaConv3, DevOp::KdaStateStepG], 1)];

    assert!(check_kda_conv_step_db(
        &bare,
        obj,
        required_kda_conv_step_db(&fused),
        first_op_in(&fused, KDA_CONV_STEP_DB_REPLACED_OPS),
    )
    .is_err());
    assert!(check_kda_conv_step_db(
        &armed,
        obj,
        required_kda_conv_step_db(&fused),
        first_op_in(&fused, KDA_CONV_STEP_DB_REPLACED_OPS),
    )
    .is_ok());
    assert!(check_kda_conv_step_db(
        &armed,
        obj,
        required_kda_conv_step_db(&legacy),
        first_op_in(&legacy, KDA_CONV_STEP_DB_REPLACED_OPS),
    )
    .is_err());
    assert!(check_kda_conv_step_db(
        &bare,
        obj,
        required_kda_conv_step_db(&legacy),
        first_op_in(&legacy, KDA_CONV_STEP_DB_REPLACED_OPS),
    )
    .is_ok());
}

/// A packet dispatching a Gemma-4 MoE op against an object built without the matching axis is
/// REFUSED — the same argument as the K3 gate, and for a family that until the AMD port had
/// NO arm at all, so its silent-NOP failure was not hypothetical. [GEMMA4-MOE-AMD]
#[test]
fn a_gemma_moe_packet_against_an_object_without_the_arms_is_refused() {
    let obj = Path::new("interp_decode.elf");
    let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx942"];
    let with_dec = vec![
        "plow_gemv_mm_cap_1",
        "plow_interp_dec_gfx942",
        MOE_GEMMA_SYM,
    ];
    let with_pf = vec!["plow_gemv_mm_cap_1", "plow_interp_gfx942", MOE_GEMMA_PF_SYM];

    for (set, sym, flag, good) in [
        (MOE_GEMMA_OPS, MOE_GEMMA_SYM, "PLOW_MOE_GEMMA", &with_dec),
        (
            MOE_GEMMA_PF_OPS,
            MOE_GEMMA_PF_SYM,
            "PLOW_MOE_GEMMA_PF",
            &with_pf,
        ),
    ] {
        for &op in set {
            let pkt = vec![prog_gemv(&[op], 1)];
            let need_dec = first_op_in(&pkt, MOE_GEMMA_OPS);
            let need_pf = first_op_in(&pkt, MOE_GEMMA_PF_OPS);
            assert!(
                need_dec == Some(op) || need_pf == Some(op),
                "{op:?} must be recognised as a Gemma MoE arm"
            );
            let e = check_moe_gemma_arms(&bare, obj, need_dec, need_pf)
                .expect_err("a Gemma MoE op against a bare object must be refused");
            let msg = e.to_string();
            assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
            assert!(msg.contains(sym), "must name the missing marker: {msg}");
            assert!(msg.contains(flag), "must name the flag: {msg}");
            assert!(check_moe_gemma_arms(good, obj, need_dec, need_pf).is_ok());
        }
    }

    // A packet with no Gemma MoE op is untouched — the gate must not refuse every other model.
    let plain = vec![prog_gemv(&[DevOp::Gemv, DevOp::RmsNorm], 1)];
    assert_eq!(first_op_in(&plain, MOE_GEMMA_OPS), None);
    assert_eq!(first_op_in(&plain, MOE_GEMMA_PF_OPS), None);
    assert!(check_moe_gemma_arms(&bare, obj, None, None).is_ok());
}

/// The two Gemma-MoE opcode lists must match the `#if PLOW_MOE_GEMMA` / `#if
/// PLOW_MOE_GEMMA_PF` regions of `interp.hip` — PARSED, not restated, for the reason
/// `k3_arm_ops_match_the_interpreter` gives: a twentieth arm added inside the guard and not
/// listed here would be dispatched by an object that does not advertise the axis, with no
/// refusal and no fault.
#[test]
fn gemma_moe_arm_ops_match_the_interpreter() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("runtime/amd/interp.hip"));
    let Some(path) = path.filter(|p| p.exists()) else {
        eprintln!("interp.hip not found — skipping (source checkout only)");
        return;
    };
    let src = std::fs::read_to_string(&path).unwrap();
    // Walk the guard nesting and collect `case PLOW_DOP_*` inside each Gemma region.
    let mut dec: Vec<String> = Vec::new();
    let mut pf: Vec<String> = Vec::new();
    let mut stack: Vec<&'static str> = Vec::new();
    let mut depth: Vec<bool> = Vec::new(); // is this #if one of ours?
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("#if") {
            let which = if t.contains("PLOW_MOE_GEMMA_PF") {
                Some("pf")
            } else if t.contains("PLOW_MOE_GEMMA") {
                Some("dec")
            } else {
                None
            };
            depth.push(which.is_some());
            if let Some(w) = which {
                stack.push(w);
            }
            continue;
        }
        if t.starts_with("#endif") {
            if depth.pop().unwrap_or(false) {
                stack.pop();
            }
            continue;
        }
        let Some(r) = t.strip_prefix("case PLOW_DOP_") else {
            continue;
        };
        let name: String = r
            .chars()
            .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
            .collect();
        match stack.last() {
            Some(&"dec") => dec.push(format!("PLOW_DOP_{name}")),
            Some(&"pf") => pf.push(format!("PLOW_DOP_{name}")),
            _ => {}
        }
    }
    for (found, listed, what) in [
        (&dec, MOE_GEMMA_OPS, "PLOW_MOE_GEMMA"),
        (&pf, MOE_GEMMA_PF_OPS, "PLOW_MOE_GEMMA_PF"),
    ] {
        let mut want: Vec<String> = listed.iter().map(|o| o.c_name().to_string()).collect();
        want.sort();
        let mut got = found.clone();
        got.sort();
        got.dedup();
        assert_eq!(
            got, want,
            "{what}: interp.hip's guarded `case` labels disagree with the Rust list"
        );
    }
}

/// The KV-encoding SWAP is refused in BOTH directions, which is what makes it different from
/// the K3 gate.
///
/// The bf16 direction is the one that had no check at all and is not hypothetical: the K3 MLA
/// gate's bf16 packet run against the fp8 object reports "all packets executed on every
/// slice: YES" and scores rel 1.000e+00 at the attention output.
#[test]
fn a_kv_encoding_mismatch_is_refused_in_both_directions() {
    let obj = Path::new("interp_decode.elf");
    let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950"];
    let with_fp8 = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950", FP8_KV_SYM];

    let check = |syms: &[&str], pkt: &[DevProg]| {
        check_kv_encoding(
            syms,
            obj,
            required_kv_op(pkt, FP8_KV_OPS),
            required_kv_op(pkt, BF16_KV_OPS),
        )
    };

    for &op in FP8_KV_OPS {
        let pkt = vec![prog_gemv(&[op], 1)];
        let e =
            check(&bare, &pkt).expect_err("an fp8-KV op against a bf16-KV object must be refused");
        let msg = e.to_string();
        assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
        assert!(msg.contains(FP8_KV_SYM), "must name the marker: {msg}");
        assert!(msg.contains("PLOW_FP8_KV"), "must name the flag: {msg}");
        assert!(
            check(&with_fp8, &pkt).is_ok(),
            "{op:?} belongs on the fp8 object"
        );
    }
    for &op in BF16_KV_OPS {
        let pkt = vec![prog_gemv(&[op], 1)];
        let e = check(&with_fp8, &pkt)
            .expect_err("a bf16-KV op against an fp8-KV object must be refused");
        let msg = e.to_string();
        assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
        assert!(
            check(&bare, &pkt).is_ok(),
            "{op:?} belongs on the bf16 object"
        );
    }

    // The ops that are in BOTH objects must be refused by NEITHER. `HeadNormRope` is the one
    // that matters: an fp8-KV packet still uses it for the QUERY norm, so listing it as bf16
    // would refuse every fp8 packet ever emitted. The gathered MLA ops keep their bf16 arm in
    // both objects for want of a free tensor slot.
    for &op in &[
        DevOp::HeadNormRope,
        DevOp::FlashGatherDecode,
        DevOp::FlashGatherPrefill,
        DevOp::Gemv,
        DevOp::MlaMergeFold,
    ] {
        let pkt = vec![prog_gemv(&[op], 1)];
        assert!(
            check(&bare, &pkt).is_ok(),
            "{op:?} must not be refused on a bf16 object"
        );
        assert!(
            check(&with_fp8, &pkt).is_ok(),
            "{op:?} must not be refused on an fp8 object"
        );
    }
}

/// [`K3_ARM_OPS`] is exactly the set of `case` labels inside the `#if PLOW_K3` region of
/// `runtime/amd/interp.hip`.
///
/// The two halves of this contract sit in different languages in different files, and the
/// consequence of them drifting is asymmetric: an arm added inside the guard but missing from
/// this list is an op that silently NOPs on a non-K3 object with no refusal, which is the bug
/// the guard was supposed to make impossible. Read out of the file rather than restated, by
/// the same discipline `dispatched_list_matches_the_amd_interpreter` applies in devgen.
#[test]
fn k3_arm_ops_match_the_interpreter() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("runtime/amd/interp.hip"));
    let Some(path) = path.filter(|p| p.exists()) else {
        eprintln!("interp.hip not found — skipping (source checkout only)");
        return;
    };
    let src = std::fs::read_to_string(&path).unwrap();

    // The LAST `#if PLOW_K3` is the dispatch region; the earlier ones guard the includes and
    // the marker. Scan to its matching `#endif`.
    let mut in_region = false;
    let mut nested = 0usize;
    let mut found: Vec<String> = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if t == "#if PLOW_K3" {
            in_region = true;
            nested = 0;
            found.clear(); // a later region supersedes an earlier one
            continue;
        }
        if in_region && t.starts_with("#if") {
            nested += 1;
            continue;
        }
        if in_region && t.starts_with("#endif") {
            if nested == 0 {
                in_region = false;
            } else {
                nested -= 1;
            }
            continue;
        }
        if in_region {
            if let Some(r) = t.strip_prefix("case PLOW_DOP_") {
                let n: String = r
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                found.push(format!("PLOW_DOP_{n}"));
            }
        }
    }
    found.sort();

    let mut want: Vec<String> = K3_ARM_OPS.iter().map(|o| o.c_name().to_string()).collect();
    want.sort();

    assert_eq!(
        found, want,
        "K3_ARM_OPS disagrees with the `#if PLOW_K3` region of interp.hip.\n  interp has: \
         {found:?}\n  K3_ARM_OPS has: {want:?}\nAn arm inside the guard but missing from the \
         list NOPs silently on an object built without PLOW_K3, with no load-time refusal."
    );
}

/// THE 14th SILENT-CORRUPTION BUG, CONSTRUCTED. A packet whose GEMVs ask for
/// more rows than the object was compiled for must be REFUSED, not clamped.
///
/// Every expectation is DERIVED from the constants, never written as a
/// literal: the object's advertised capacity is built by concatenating onto
/// [`GEMV_CAP_SYM_PREFIX`], and the packet's demand is that capacity `+ 1`.
/// A sibling test that spelled its bound out inverted its own meaning when
/// the bound moved underneath it; move `PLOW_GEMV_MM`'s default or the
/// marker's spelling and this test follows rather than lying.
#[test]
fn a_gemv_wider_than_the_objects_bucket_is_refused() {
    let obj = Path::new("interp_decode.elf");
    for cap in [1u32, 2, 4, 8, 16] {
        let marked = format!("{GEMV_CAP_SYM_PREFIX}{cap}");
        let syms = vec![marked.as_str(), "plow_interp_dec_gfx950"];

        // Exactly at the bucket: accepted, for every op that reads it.
        let fits = vec![prog_gemv(GEMV_BUCKET_OPS, cap)];
        assert_eq!(required_gemv_m(&fits), cap);
        assert!(check_gemv_capacity(&syms, obj, required_gemv_m(&fits)).is_ok());

        // One row past it: refused. `gemv_rows<MM>` would write rows
        // 0..cap and leave the last one holding whatever was there.
        let over = vec![prog_gemv(GEMV_BUCKET_OPS, cap + 1)];
        assert_eq!(required_gemv_m(&over), cap + 1);
        let e = check_gemv_capacity(&syms, obj, required_gemv_m(&over))
            .expect_err("an M past the object's bucket must be refused");
        let msg = e.to_string();
        // The refusal must name all three: what the packet needs, what the
        // object has, and how to rebuild.
        assert!(msg.contains(&format!("M={} rows", cap + 1)), "{msg}");
        assert!(msg.contains(&format!("PLOW_GEMV_MM={cap}")), "{msg}");
        // Past `PLOW_GEMV_MAXM` there is nothing to rebuild, and the
        // remedy has to say so instead of naming an impossible bucket.
        if cap + 1 > GEMV_MAXM {
            assert!(msg.contains("No object can serve this"), "{msg}");
            assert!(
                msg.contains(&format!("PLOW_DECODE_BATCH <= {GEMV_MAXM}")),
                "{msg}"
            );
        } else {
            assert!(
                msg.contains(&format!("PLOW_DECODE_BATCH={}", cap + 1)),
                "{msg}"
            );
        }
    }
}

#[test]
fn b128_argmax_requires_an_object_capacity_marker() {
    let obj = Path::new("interp_decode_k3.elf");
    assert!(check_xargmax_capacity(&[], obj, 32).is_ok());
    assert!(check_xargmax_capacity(&[], obj, 64).is_err());
    assert!(check_xargmax_capacity(&[XARGMAX_B128_SYM], obj, 128).is_ok());
}

/// An object that does not advertise a bucket is refused above M=1 and
/// accepted at M=1.
///
/// Silence is not consent: every object built before the marker compiled at
/// `op_gemm.h`'s default of 1, which is the bug. But MM >= 1 always, so an
/// M=1 packet — the whole batch-1 world, including every TP asset, which
/// `load` refuses to batch at all — must stay loadable against the objects
/// already on disk.
#[test]
fn an_object_that_advertises_nothing_is_refused_only_above_one_row() {
    let obj = Path::new("interp_decode.elf");
    let bare = vec!["plow_interp_dec_gfx950", "__hip_cuid_3ec2926410ebcf30"];
    assert_eq!(object_gemv_cap(&bare), None);

    let one = vec![prog_gemv(GEMV_BUCKET_OPS, 1)];
    assert_eq!(required_gemv_m(&one), 1);
    assert!(check_gemv_capacity(&bare, obj, required_gemv_m(&one)).is_ok());
    // A packet with no GEMV at all is 0, which is likewise never refused.
    assert!(check_gemv_capacity(&bare, obj, 0).is_ok());

    let two = vec![prog_gemv(&[DevOp::GemvQkv], 2)];
    let e = check_gemv_capacity(&bare, obj, required_gemv_m(&two))
        .expect_err("an unmarked object must not silently serve M>1");
    let msg = e.to_string();
    assert!(msg.contains(GEMV_CAP_SYM_PREFIX), "{msg}");
    assert!(msg.contains("PLOW_DECODE_BATCH=2"), "{msg}");
}

/// `required_gemv_m` reads the INSTRUCTIONS, and only the ops that actually
/// reach a `<PLOW_GEMV_MM>` instantiation.
///
/// The MoE expert arms live in `op_moe.h` and do their own row handling, so
/// counting them would refuse packets the bucket cannot hurt — the mirror
/// image of the bug, and just as wrong.
#[test]
fn only_the_bucketed_ops_set_the_requirement() {
    // A wide MoE expert GEMV next to a one-row bucketed GEMV: still 1.
    let mixed = vec![prog_gemv(&[DevOp::MoeExpertGlu, DevOp::MoeExpertDown], 64)];
    assert_eq!(required_gemv_m(&mixed), 0);

    let mut p = prog_gemv(&[DevOp::MoeExpertGlu], 64);
    p.insts.extend(prog_gemv(&[DevOp::Gemv], 1).insts);
    assert_eq!(required_gemv_m(std::slice::from_ref(&p)), 1);

    // The maximum is taken over EVERY program and every instruction, not
    // the first one found.
    let progs = vec![
        prog_gemv(&[DevOp::Gemv], 1),
        prog_gemv(&[DevOp::GemvGlu], 8),
    ];
    assert_eq!(required_gemv_m(&progs), 8);
}

/// A REAL object from the shipped tree carries no bucket marker, and the
/// parser says so rather than guessing.
///
/// `build-amd/hsaco-abi144/interp_decode.elf` is the current ABI-144 decode
/// object, built before this marker existed and — like every gfx950 decode
/// object built without `PLOW_DECODE_BATCH` — compiled at the `op_gemm.h`
/// default of 1. It is the exact input that produced the bug. Skipped when
/// the tree is not on this machine.
#[test]
fn a_shipped_object_without_the_marker_reads_as_unknown() {
    // Fixture: a decode object built before the marker existed. Skipped unless
    // `PLOW_TEST_ABI144_DECODE_ELF` points at one.
    let Some(p) = std::env::var_os("PLOW_TEST_ABI144_DECODE_ELF").map(std::path::PathBuf::from)
    else {
        return;
    };
    let Ok(img) = std::fs::read(&p) else { return };
    let syms = elf_symbol_names(&img);
    // The reader works on this file (sanity: the interpreter body is there),
    // so `None` below means "no marker", never "no symbol table".
    assert!(syms.iter().any(|s| s.contains("plow_exec")), "{syms:?}");
    assert_eq!(object_gemv_cap(&syms), None);
    assert!(check_gemv_capacity(&syms, &p, 1).is_ok());
    assert!(check_gemv_capacity(&syms, &p, 2).is_err());
}

/// The marker is a contract between `op_gemm.h` and this file, written in
/// two languages. Read the C side rather than restating it.
///
/// The concatenation `plow_gemv_mm_cap_##n` cannot be grepped for its
/// expanded form, so this asserts on the token the macro pastes onto — which
/// is what [`GEMV_CAP_SYM_PREFIX`] has to match — and on the fact that the
/// value pasted is `PLOW_GEMV_MM` itself. If the symbol were named for
/// anything else it could disagree with what was compiled, which is the
/// entire failure this check exists to end.
#[test]
fn op_gemm_h_emits_the_capacity_marker() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/amd/op_gemm.h");
    let src = std::fs::read_to_string(&p).expect("runtime/amd/op_gemm.h");
    assert!(
        src.contains(&format!("{GEMV_CAP_SYM_PREFIX}##n")),
        "op_gemm.h no longer pastes onto `{GEMV_CAP_SYM_PREFIX}` — the loader's \
         capacity check would silently stop finding any object's bucket"
    );
    assert!(
        src.contains("PLOW_GEMV_CAP_SYM(PLOW_GEMV_MM) = PLOW_GEMV_MM"),
        "the capacity marker must be named for, and hold, PLOW_GEMV_MM itself"
    );
    // The ceiling too. [`GEMV_MAXM`] decides both the `load` refusal and
    // which remedy `check_gemv_capacity` prints, and a stale copy of it
    // would send someone to build a bucket the header's static assert
    // rejects.
    assert!(
        src.contains(&format!("#define PLOW_GEMV_MAXM {GEMV_MAXM}")),
        "op_gemm.h's PLOW_GEMV_MAXM is no longer {GEMV_MAXM}"
    );
    // The walk marker is the OTHER half of the same contract, and it fails
    // more dangerously than the capacity one: if op_gemm.h stops emitting it,
    // a walking object silently becomes a hard-capacity object to the loader
    // and every M > MM packet is REFUSED (loud, recoverable). If the loader
    // stops looking for it, an MM=8 object serving M=16 is accepted with no
    // walk and rows 8..15 come back STALE (silent, fluent, wrong).
    assert!(
        src.contains(&format!("unsigned {GEMV_WALK_SYM} = 1")),
        "op_gemm.h no longer emits `{GEMV_WALK_SYM}` under PLOW_GEMV_WALK — \
         `check_gemv_capacity` would refuse every walking object it should serve"
    );
    assert!(
        src.contains("#if PLOW_GEMV_WALK"),
        "the walk marker must be gated on PLOW_GEMV_WALK itself, or a \
         non-walking object would advertise a capacity it does not have"
    );
}

/// The walk marker turns the capacity check off, and nothing else does.
///
/// Guards both directions of the failure above: a walking MM=8 object must
/// serve M=16, and a NON-walking one must still refuse it. The second half
/// is the silent-corruption case (§6g-BATCH slots 13/14/15), so it is the
/// one worth a test.
#[test]
fn the_walk_marker_lifts_the_capacity_refusal() {
    let p = Path::new("/tmp/interp_decode.elf");
    let hard = ["plow_gemv_mm_cap_8"];
    let walk = ["plow_gemv_mm_cap_8", GEMV_WALK_SYM];
    assert!(check_gemv_capacity(&hard, p, 8).is_ok(), "MM=8 covers M=8");
    assert!(
        check_gemv_capacity(&hard, p, 16).is_err(),
        "a NON-walking MM=8 object must still refuse M=16 — rows 8..15 would be stale"
    );
    assert!(
        check_gemv_capacity(&walk, p, 16).is_ok(),
        "a walking MM=8 object serves M=16 in two row blocks"
    );
    // An unmarked object is still refused, walk or no walk: silence is not consent.
    assert!(check_gemv_capacity(&[], p, 2).is_err());
}

#[test]
fn symbols_carry_the_isa_and_the_scheduler() {
    assert_eq!(
        symbol_name(Phase::Decode, Sched::Static, "gfx950"),
        "plow_interp_dec_gfx950"
    );
    assert_eq!(
        symbol_name(Phase::Prefill, Sched::GlobalQueue, "gfx950"),
        "plow_interp_gfx950_gq"
    );
    assert_eq!(
        symbol_name(Phase::Flash, Sched::GlobalQueue, "gfx950"),
        "plow_interp_flash_gfx950_gq"
    );
}

/// The ladder is mixed, not repeated: covering 1536 with 1024+512 beats two
/// 1024s, which would compute 512 padded rows at full cost.
#[test]
fn chunk_plan_mixes_the_ladder_and_puts_the_ragged_chunk_last() {
    let bkt = [128, 512, 1024];
    // `plan_chunks_cfg(.., false)`, not `plan_chunks`: this test is about the
    // PADDED DP, and `plan_chunks` reads the process-wide default — which is
    // now ragged, so the wrapper would silently stop exercising the DP.
    let dp = |n| plan_chunks_cfg(&bkt, n, LAUNCH_ROWS, false).unwrap();
    assert_eq!(dp(1536), vec![1024, 512]);
    assert_eq!(dp(1024), vec![1024]);
    assert_eq!(dp(1), vec![128]);
    assert_eq!(dp(0), Vec::<u32>::new());

    // Descending, so the ragged chunk is the LAST one — padding lands in
    // the tail, where a padded row writes KV that `n_kv` bounds out.
    for n in [200u32, 700, 1300, 4000, 9000] {
        let plan = dp(n);
        assert!(
            plan.windows(2).all(|w| w[0] >= w[1]),
            "plan for {n} is not largest-first: {plan:?}"
        );
        let covered: u32 = plan.iter().sum();
        assert!(covered >= n, "plan for {n} covers only {covered}: {plan:?}");
    }
}

/// A blob with no prefill bucket at all is a decode-only blob, not a silent
/// zero-chunk prefill.
///
/// And the cap is the PACKET's, not a constant: a 16384 rung is USED, not
/// filtered. This is the whole of the `MAX_CHUNK = 16384` change — the runtime
/// used to hold its own 8192 and quietly serve a wider blob as if the rung
/// were absent.
#[test]
fn the_ladder_is_its_own_cap() {
    const B16: &[u32] = &[128, 512, 1024, 2048, 4096, 8192, 16384];
    const B8: &[u32] = &[128, 512, 1024, 2048, 4096, 8192];
    let ragged16 = |n| plan_chunks_cfg(B16, n, LAUNCH_ROWS, true).unwrap();

    assert!(plan_chunks(&[], 10).is_err());
    // A 16384-rung packet plans ON the 16384 rung.
    assert_eq!(ragged16(16384), vec![16384]);
    assert_eq!(ragged16(8193), vec![16384]);
    // ...and the SAME prompt on an 8192-ladder packet still takes two chunks,
    // so the cap follows the blob rather than the binary.
    let ragged8 = |n| plan_chunks_cfg(B8, n, LAUNCH_ROWS, true).unwrap();
    assert_eq!(ragged8(8193), vec![8192, 128]);
    // Under the PADDED DP the wide rung is correctly declined below its width
    // — 8191 rows of dead compute cost more than a second launch — which is
    // why the rung is worth nothing without ragged-M.
    let padded16 = |n| plan_chunks_cfg(B16, n, LAUNCH_ROWS, false).unwrap();
    assert_eq!(padded16(8193), vec![8192, 128]);
    assert_eq!(
        plan_chunks_capped(B8, 8192, 4096).unwrap(),
        vec![4096, 4096]
    );
    assert!(plan_chunks_capped(B8, 128, 64).is_err());
}

/// One contiguous h2d, not `n_kvrow` scattered ones: the sites straddle
/// almost the whole instruction array (Gemma-31B: [4,664] of 676), and
/// submission overhead dominates bytes.
#[test]
fn kvrow_span_covers_every_site() {
    assert_eq!(kvrow_span(&[4, 300, 664, 12]), Some((4, 664)));
    assert_eq!(kvrow_span(&[7]), Some((7, 7)));
    assert_eq!(kvrow_span(&[]), None);
}

/// The live split rule reproduces `devgen::mla::glm_nsplit`'s measured ladder
/// (`ctx/256`, floored at 16) and never exceeds what the emitter baked.
#[test]
fn live_nsplit_walks_the_measured_ladder_and_never_grows() {
    // A 32768-max-ctx TP4 blob bakes 64. The optima the ladder measured are
    // 16 / 16 / 16 / 32 / 64 / 64 at 1k / 2k / 4k / 8k / 16k / 32k.
    for (kv, want) in [
        (1024u32, 16u32),
        (2048, 16),
        (4096, 16),
        (8192, 32),
        (16384, 64),
        (32768, 64),
    ] {
        assert_eq!(mla_live_nsplit(64, kv), want, "kv_len {kv}");
    }
    // At `kv_len == max_ctx` the live value IS the baked one, so the top of
    // the served range keeps the shipped dispatch.
    assert_eq!(mla_live_nsplit(32, 8192), 32);
    assert_eq!(
        mla_live_nsplit(16, 8192),
        16,
        "baked is a ceiling, not a target"
    );
    // Below 16 KV tiles there are not 16 splits to make.
    assert_eq!(mla_live_nsplit(64, 512), 16);
    assert_eq!(mla_live_nsplit(64, 256), 8);
    assert_eq!(mla_live_nsplit(64, 1), 1);
}

/// The flash and its merge move together, and anything that is not a plain
/// dense MLA decode is left alone.
#[test]
fn mla_nsplit_sites_pair_the_flash_with_its_merge() {
    let inst = |op: DevOp, ns: u32| {
        let mut d = DevInst64 {
            op: op as u16,
            ..Default::default()
        };
        d.i[4] = ns;
        d
    };
    let two_layers = vec![
        inst(DevOp::RmsNorm, 0),
        inst(DevOp::FlashMlaDecode, 64),
        inst(DevOp::MlaMergeFold, 64),
        inst(DevOp::Gemv, 0),
        inst(DevOp::FlashMlaDecode, 64),
        inst(DevOp::MlaMergeFold, 64),
    ];
    assert_eq!(
        derive_mla_nsplit(&two_layers),
        Some((vec![1, 2, 4, 5], 64)),
        "both fields of both layers, and the baked count"
    );

    // The fp8-latent twin is the same packet with a scale strip.
    let fp8 = [
        inst(DevOp::FlashMlaDecodeFp8, 32),
        inst(DevOp::MlaMergeFold, 32),
    ];
    assert_eq!(derive_mla_nsplit(&fp8), Some((vec![0, 1], 32)));

    // DSA: the gather splits over selected rows, not the KV window.
    let gathered = [
        inst(DevOp::FlashGatherDecode, 64),
        inst(DevOp::MlaMergeFold, 64),
    ];
    assert_eq!(derive_mla_nsplit(&gathered), None);

    // A merge with no flash (or the reverse) is not a pair.
    assert_eq!(derive_mla_nsplit(&[inst(DevOp::MlaMergeFold, 64)]), None);
    assert_eq!(derive_mla_nsplit(&[inst(DevOp::Gemv, 0)]), None);
    // Disagreeing sites would desynchronise a flash/merge pair if half-patched.
    assert_eq!(
        derive_mla_nsplit(&[
            inst(DevOp::FlashMlaDecode, 64),
            inst(DevOp::MlaMergeFold, 32),
        ]),
        None
    );
}

/// A GLM-5.2 MLA prefill chunk: the two KV-write sites move, the ordinary
/// norms and the query rope do NOT, and the flash keeps every operand it
/// was emitted with.
///
/// This is the shape the old `HeadNormRope && fj[1] != 0` test got wrong.
/// MLA's k_rope sets `j[1] = KV_MASK_NONE`, which packs into `fj[2]`, and
/// leaves `f[1]`/`j[0]` — the two halves of `fj[1]` — at zero, so the test
/// matched NOTHING on an MLA packet: every chunk's latent and rope rows were
/// written at row 0, with no error anywhere.
#[test]
fn rebase_chunk_moves_only_the_kv_write_rows() {
    let names: Vec<String> = [
        "kv.0.ckv",
        "act.xn",
        "kv.0.krot",
        "act.qr",
        "act.opart",
        "kv.0.kidx_pool",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let inst = |op: DevOp, dst: u16| DevInst64 {
        op: op as u16,
        t: [dst, 0, 0, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    // 0 kv_a_layernorm -> kv.0.ckv (row in i[2]); 1 the input norm, same
    // opcode, i[2] means nothing there; 2 k_rope -> kv.0.krot (row in i[3]);
    // 3 the QUERY rope, same opcode, must be left alone; 4 the flash;
    // 5 GLM-5.3's pooled indexer cache (pool-row base in i[3]).
    let mut insts = vec![
        inst(DevOp::RmsNorm, 0),
        inst(DevOp::RmsNorm, 1),
        inst(DevOp::HeadNormRope, 2),
        inst(DevOp::HeadNormRope, 3),
        inst(DevOp::FlashMlaPrefill, 4),
        inst(DevOp::DsaPoolCompress, 5),
    ];
    insts[4].i = [1, 64, 8192, 0, 128, u32::MAX, 0, 7];
    let before = insts.clone();

    rebase_chunk_rows(&mut insts, &names, 512, 128, 128, None);

    assert_eq!(insts[0].i[2], 512, "kv.0.ckv out_row0 was not rebased");
    assert_eq!(insts[2].i[3], 512, "kv.0.krot out_row was not rebased");
    assert_eq!(
        insts[5].i[3], 512,
        "kv.0.kidx_pool out-pool base was not rebased"
    );
    // Everything else, field for field.
    assert_eq!(insts[0].i[..2], before[0].i[..2]);
    assert_eq!(insts[0].i[3..], before[0].i[3..]);
    assert_eq!(insts[1].i, before[1].i, "the ordinary RmsNorm moved");
    assert_eq!(insts[2].i[..3], before[2].i[..3]);
    assert_eq!(insts[2].i[4..], before[2].i[4..]);
    assert_eq!(insts[3].i, before[3].i, "the QUERY rope moved");
    // FlashMlaPrefill takes its query base from `in.kvlen`, not from an
    // immediate; every `i[]` here is a live operand and none may be touched.
    assert_eq!(insts[4].i, before[4].i, "FlashMlaPrefill was patched");
}

/// The RAGGED-M row shrink rewrites the row count of every family that
/// carries one, RESCALES the element-count families, and leaves alone both
/// the lm_head (whose `M` is 1, not `T`) and a row-BANDED GEMM (whose `M` is
/// `T/kb`). The last two are the whole reason the shrink is guarded on the
/// field already equalling the bucket width.
#[test]
fn ragged_rows_shrink_only_the_fields_that_hold_the_bucket_width() {
    const T: u32 = 8192;
    const CLEN: u32 = 4097;
    const H: u32 = 6144;
    let names: Vec<String> = ["act.xn", "act.logits"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let inst = |op: DevOp, i: [u32; 8]| DevInst64 {
        op: op as u16,
        t: [0, 0, 0, 0, 0, 0, 0, 0],
        i,
        ..Default::default()
    };
    let mut insts = vec![
        inst(DevOp::Embed, [T, H, 0, 0, 0, 0, 0, 0]),
        inst(DevOp::RmsNorm, [T, H, 0, 0, 0, 0, 0, 0]),
        inst(DevOp::GemmSmall, [T, 2048, H, 0, 0, 0, 0, 0]),
        inst(DevOp::FlashMlaPrefill, [1, 8, 73728, 0, T, u32::MAX, 2, 7]),
        inst(DevOp::MlaMergeFold, [T, 8, 256, 0, 2, 0, 0, 0]),
        inst(DevOp::XReduceTwoShot, [T * H, 8, 0, 0, 1, 0, 0, 0]),
        inst(DevOp::Residual, [T * H, 0, 0, 0, 0, 0, 0, 0]),
        inst(DevOp::MoeRouterTopkPf, [0, 256, 8, 0, T, 0, 1, 1]),
        inst(DevOp::MoeAlignPf, [T, 256, 8, 0, 0, 0, 0, 0]),
        inst(DevOp::MoeCombinePf, [H, 8, T, 0, 0, 0, 0, 0]),
        // The lm_head: M = 1 over `a_row0`, NOT a row count.
        inst(DevOp::Gemv, [1, 154880, H, 0, T - 1, 0, 0, 0]),
        // A PLOW_GLM_XR_BAND row-band GEMM: M = T/2 at a_row0 = T/2.
        inst(DevOp::Gemm, [T / 2, H, 2048, 0, T / 2, T / 2, 0, 0]),
    ];
    let before = insts.clone();

    rebase_chunk_rows(&mut insts, &names, 0, CLEN, T, Some(T));

    assert_eq!(insts[0].i[0], CLEN, "Embed ntok");
    assert_eq!(insts[0].i[1], H, "Embed hidden must not move");
    assert_eq!(insts[1].i[0], CLEN, "RmsNorm rows");
    assert_eq!(insts[2].i[0], CLEN, "GEMM M");
    assert_eq!(insts[2].i[1..3], before[2].i[1..3], "GEMM N/K moved");
    assert_eq!(insts[3].i[4], CLEN, "flash n_tok");
    assert_eq!(insts[3].i[..4], before[3].i[..4], "flash operands moved");
    assert_eq!(insts[4].i[0], CLEN, "merge token count");
    assert_eq!(insts[5].i[0], CLEN * H, "two-shot element count");
    assert_eq!(insts[6].i[0], CLEN * H, "residual element count");
    assert_eq!(insts[7].i[4], CLEN, "router T");
    assert_eq!(insts[8].i[0], CLEN, "align T");
    assert_eq!(insts[9].i[2], CLEN, "combine T");
    assert_eq!(insts[10].i, before[10].i, "the lm_head GEMV was rewritten");
    assert_eq!(insts[11].i, before[11].i, "a banded GEMM was rewritten");
}

#[test]
fn ragged_sparse_prefill_keeps_selection_and_flash_layouts_equal() {
    const T: u32 = 8192;
    let inst = |op: DevOp, i: [u32; 8]| DevInst64 {
        op: op as u16,
        t: [u16::MAX; 8],
        i,
        ..Default::default()
    };
    let chain = vec![
        inst(DevOp::LayerNorm, [T, 128, 0, 0, 0, 0, 0, 0]),
        inst(DevOp::IndexScorePf, [T, 32, 81920, 128, 0, 0, 0, 0]),
        inst(DevOp::IndexSelectPf, [T, 2048, 81920, 0, 0, 0, 0, 0]),
        inst(DevOp::IndexUnionPf, [T, 2048, 81920, 16384, 8, 0, 0, 0]),
        inst(
            DevOp::FlashMlaPrefill,
            [1, 8, 81920, 0, T, u32::MAX, 16384, 0],
        ),
    ];
    for rows in [1, 2049, 4097, T] {
        let mut insts = chain.clone();
        rebase_chunk_rows(&mut insts, &[], T, rows, T, Some(T));
        for d in &insts[..4] {
            assert_eq!(d.i[0], insts[4].i[4], "op {} retained padded rows", d.op);
        }
        assert_eq!(insts[4].i[4], rows);
        for (before, after) in chain[..4].iter().zip(&insts[..4]) {
            assert_eq!(before.i[1..], after.i[1..], "index geometry changed");
        }
        let union_header = (insts[3].i[0].div_ceil(8) * 4).div_ceil(256) * 256;
        let flash_header = (insts[4].i[4].div_ceil(8) * 4).div_ceil(256) * 256;
        assert_eq!(union_header, flash_header);
    }
}

/// A FULL last chunk (`clen == T`) must leave every instruction alone — the
/// shrink is a no-op on an exactly-covered prompt, which is what keeps
/// 1024/4096/8192/16384 byte-identical to the padded path.
#[test]
fn ragged_rows_are_a_no_op_when_the_chunk_is_full() {
    const T: u32 = 4096;
    let names: Vec<String> = ["act.xn"].iter().map(|s| s.to_string()).collect();
    let mut insts = vec![
        DevInst64 {
            op: DevOp::RmsNorm as u16,
            i: [T, 6144, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::XReduceTwoShot as u16,
            i: [T * 6144, 8, 0, 0, 1, 0, 0, 0],
            ..Default::default()
        },
    ];
    let before = insts.clone();
    rebase_chunk_rows(&mut insts, &names, 0, T, T, Some(T));
    assert_eq!(insts[0].i, before[0].i);
    assert_eq!(insts[1].i, before[1].i);
}

/// The RAGGED cover is the MINIMUM number of launches, and the tail rung is
/// the smallest one that HOLDS the remainder — not the cheapest padded cover
/// of it, because under the row shrink the padding is free.
///
/// The paired padded plans are the shipped DP's answers, so this test is also
/// the record of which lengths the axis changes and which it does not.
#[test]
fn the_ragged_cover_takes_the_fewest_launches() {
    const B: &[u32] = &[128, 512, 1024, 2048, 4096, 8192];
    let ragged = |n| plan_chunks_cfg(B, n, LAUNCH_ROWS, true).unwrap();
    let padded = |n| plan_chunks_cfg(B, n, LAUNCH_ROWS, false).unwrap();

    // Exactly on a rung, or an exact multiple of the widest one: IDENTICAL.
    for n in [128u32, 1024, 4096, 8192, 16384, 24576] {
        assert_eq!(
            ragged(n),
            padded(n),
            "the cover moved at an exact length {n}"
        );
    }
    // One token over a rung: one launch instead of two.
    assert_eq!(padded(1025), vec![1024, 128]);
    assert_eq!(ragged(1025), vec![2048]);
    assert_eq!(padded(4097), vec![4096, 128]);
    assert_eq!(ragged(4097), vec![8192]);
    // Past the widest rung the second launch is STRUCTURAL (no bucket is
    // wider than MAX_CHUNK), so the plan is the same and only the tail's row
    // count shrinks.
    assert_eq!(padded(8193), vec![8192, 128]);
    assert_eq!(ragged(8193), vec![8192, 128]);
    // A deeply ragged length: three launches become two.
    assert_eq!(ragged(12345), vec![8192, 8192]);
    assert!(
        padded(12345).len() > 2,
        "expected the DP to add a tail chunk"
    );
    // Every ragged cover is the arithmetic minimum, and covers the prompt.
    for n in [1u32, 127, 129, 1025, 4097, 8193, 12345, 16385, 65536, 73728] {
        let c = ragged(n);
        assert_eq!(
            c.len(),
            n.div_ceil(8192) as usize,
            "not the fewest launches at {n}"
        );
        assert!(c.iter().sum::<u32>() >= n, "cover short at {n}");
    }
}

/// The dense-GQA rules are unchanged, and the two KV tests are a UNION: a
/// `kv.*` destination and a non-zero `fj[1]` both mark the same site.
#[test]
fn rebase_chunk_still_patches_the_dense_gqa_families() {
    let names: Vec<String> = ["kv.0.k", "act.q"].iter().map(|s| s.to_string()).collect();
    let mut insts = vec![
        // k norm: `kv.*` AND j[0] = ring stride, both tests fire, one field.
        DevInst64 {
            op: DevOp::HeadNormRopeFp8 as u16,
            t: [0; 8],
            fj: [0, 4096, 0],
            ..Default::default()
        },
        // q norm: neither test fires.
        DevInst64 {
            op: DevOp::HeadNormRope as u16,
            t: [1, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        },
        DevInst64 {
            op: DevOp::FlashPrefillFp8 as u16,
            t: [1, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        },
    ];
    rebase_chunk_rows(&mut insts, &names, 1024, 512, 512, None);
    assert_eq!(insts[0].i[3], 1024);
    assert_eq!(insts[1].i, [0; 8], "the query norm was patched");
    assert_eq!(insts[2].i[4], 1024, "q_pos0");
    assert_eq!(insts[2].i[1], 1536, "n_kv is everything written so far");
}

/// EVERY KDA OP'S ROW COUNT BECOMES `clen`, AND THE BUG IS WHAT THIS ASSERTS.
///
/// This covers both a compiled 512-row rung shortened to 511 and the production
/// 8192-row rung shortened to 8191. Before KDA rebasing, every stateful arm ran
/// through the padded tail and advanced the carried state past the real final token.
///
/// Asserted as `!= t` rather than only `== clen` so that a future change which
/// re-bakes `T` somewhere else still fails here: the property is "the KDA row
/// count is the REAL row count", not one pinned tail size.
#[test]
fn rebase_chunk_shortens_every_kda_arm_to_the_real_row_count() {
    let names: Vec<String> = ["kv.0.state", "act.q"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (t, clen) in [(512, 511), (8192, 8191)] {
        let kda = |op: DevOp| DevInst64 {
            op: op as u16,
            i: [t, 96, 128, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        let mut insts: Vec<DevInst64> = KDA_ROW_COUNT_OPS.iter().map(|&o| kda(o)).collect();
        // A non-KDA neighbour that must NOT be touched by the new arm.
        insts.push(DevInst64 {
            op: DevOp::RmsNorm as u16,
            i: [t, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        });
        rebase_chunk_rows(&mut insts, &names, 1024, clen, t, None);

        for (d, &op) in insts.iter().zip(KDA_ROW_COUNT_OPS) {
            assert_eq!(d.i[0], clen, "{op:?} still runs the padded bucket width");
            assert_ne!(d.i[0], t, "{op:?} kept the compiler's baked T");
            assert_eq!((d.i[1], d.i[2]), (96, 128), "{op:?}: a live operand moved");
        }
        assert_eq!(insts.last().unwrap().i[0], t, "a non-KDA op was shortened");
    }
}

/// GLM-5.3's pooled DSA indexer, non-pool-aligned tail: `DsaPoolCompress` must stop
/// at the last COMPLETE pool `clen` covers, and `DsaPoolStash`'s prefill tail-seed
/// instructions must fire on exactly the trailing real rows a padded chunk's
/// compress call now excludes — neither more (a stray write into a slot an
/// already-cached pool owns) nor fewer (a real trailing token silently never
/// becomes searchable).
///
/// `T=512, CLEN=502, pool_size=4`: `complete = (502/4)*4 = 500`, so exactly the
/// last TWO real rows (500, 501) are the genuine trailing remainder and the third
/// baked stash (row 509 -> rebased 499, which is `< complete`, i.e. already inside
/// a pool `DsaPoolCompress` itself just compressed) must `Nop`.
#[test]
fn dsa_pool_tail_seed_shrinks_compress_and_rebases_exactly_the_real_remainder() {
    let names: Vec<String> = Vec::new();
    const T: u32 = 512;
    const CLEN: u32 = 502;
    const POOL_SIZE: u32 = 4;
    let compress = DevInst64 {
        op: DevOp::DsaPoolCompress as u16,
        i: [T / POOL_SIZE, POOL_SIZE, 128, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    // Baked exactly as `emit_glm_dsa_prefill_select` bakes them: rows T-1, T-2, T-3.
    let stash = |row: u32| DevInst64 {
        op: DevOp::DsaPoolStash as u16,
        i: [POOL_SIZE, 128, row, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    let mut insts = vec![compress, stash(T - 1), stash(T - 2), stash(T - 3)];
    rebase_chunk_rows(&mut insts, &names, 0, CLEN, T, None);

    assert_eq!(
        insts[0].i[0],
        CLEN / POOL_SIZE,
        "compress did not shrink to clen's pools"
    );
    assert_eq!(
        (insts[0].i[1], insts[0].i[2]),
        (POOL_SIZE, 128),
        "a live compress operand moved"
    );

    assert_eq!(
        insts[1].op,
        DevOp::DsaPoolStash as u16,
        "genuine trailing row #1 was dropped"
    );
    assert_eq!(
        insts[1].i[2],
        CLEN - 1,
        "row was not rebased from t's end to clen's end"
    );
    assert_eq!(
        insts[2].op,
        DevOp::DsaPoolStash as u16,
        "genuine trailing row #2 was dropped"
    );
    assert_eq!(insts[2].i[2], CLEN - 2);
    assert_eq!(
        insts[3].op,
        DevOp::Nop as u16,
        "surplus stash for an already-compressed row was left live -- it can race a \
         genuine trailing write for the same ring slot (pos % pool_size repeats every \
         pool_size rows) and clobber it with stale already-cached data"
    );
}

/// A full/aligned chunk (`clen == t`, `t` always a multiple of `pool_size`) must
/// `Nop` every tail-seed stash: `complete == clen`, so every baked row is `<
/// complete` by construction — `DsaPoolCompress` itself already compressed the
/// chunk's last pool directly, and the ring has nothing left to carry.
#[test]
fn dsa_pool_tail_seed_is_a_full_nop_on_an_aligned_chunk() {
    let names: Vec<String> = Vec::new();
    const T: u32 = 512;
    const POOL_SIZE: u32 = 4;
    let stash = |row: u32| DevInst64 {
        op: DevOp::DsaPoolStash as u16,
        i: [POOL_SIZE, 128, row, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    let mut insts = vec![stash(T - 1), stash(T - 2), stash(T - 3)];
    rebase_chunk_rows(&mut insts, &names, 0, T, T, None);
    for d in &insts {
        assert_eq!(
            d.op,
            DevOp::Nop as u16,
            "an aligned chunk's tail-seed stash must not fire"
        );
    }
}

/// THE ATTNRES SCORE WEIGHT IS DERIVED, AND THE FOLD IS CHECKED AGAINST THE
/// REAL CHECKPOINT RATHER THAN A FIXTURE.
///
/// `models--moonshotai--Kimi-K3` ships `*_res_norm.weight` [7168] bf16 and
/// `*_res_proj.weight` [1, 7168] bf16, 93 of each at both the attention and the
/// MLP site; the packet declares one f32 [7168] per site. Without the fold all
/// 186 resolve to MISSING WEIGHT and no real-weight K3 run can start.
///
/// Skipped when the checkpoint is not on this machine — the same convention
/// `prefill_object_without_mla_arms_is_refused` uses for its fixture.
#[test]
fn the_attn_res_score_weight_folds_from_the_pair_the_checkpoint_ships() {
    let Some(dir) = k3_snapshot_dir() else { return };
    let Ok(c) = crate::asset::checkpoint::Checkpoint::open(&dir) else {
        return;
    };
    const H: usize = 7168;
    for site in ["self_attention", "mlp"] {
        let base = format!("language_model.model.layers.1.{site}");
        let out = fold_res_score(&c, &format!("{base}_res_score.weight"))
            .expect("name matches the derived pattern")
            .expect("both sources present in the checkpoint");
        assert_eq!(out.len(), H * 4, "{site}: f32 [hidden]");

        // Recompute element 0 and a middle element from the two sources, so
        // this pins the RELATION and not merely the length.
        let bf = |n: &str, i: usize| -> f32 {
            let (raw, _) = c.tensor_ex(n).expect("source tensor");
            let b = &raw[i * 2..i * 2 + 2];
            f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
        };
        for i in [0usize, H / 2, H - 1] {
            let want = bf(&format!("{base}_res_norm.weight"), i)
                * bf(&format!("{base}_res_proj.weight"), i);
            let got = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(got, want, "{site}[{i}] is not norm * proj");
        }
    }
    // A name that is not a score weight must not be intercepted at all —
    // otherwise every ordinary weight would take the derived path.
    assert!(fold_res_score(&c, "language_model.lm_head.weight").is_none());
}

/// The K3 snapshot directory, or `None` on a machine without it.
fn k3_snapshot_dir() -> Option<std::path::PathBuf> {
    // `PLOW_TEST_K3_SNAPSHOTS` names the HF hub `snapshots/` dir of a Kimi-K3 checkout.
    let root = std::env::var_os("PLOW_TEST_K3_SNAPSHOTS")?;
    std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("model.safetensors.index.json").exists())
}

/// The two questions about a carried-state tensor must have ONE answer.
///
/// `kv_skips_zeroing` (may LOAD skip the memset?) and `begin_slot` (what must a
/// NEW SEQUENCE clear?) are the same set for the same reason — these tensors are
/// read before they are written. They were two separate pieces of knowledge and
/// only the first existed, which is how the state came to be zeroed exactly once
/// per process and never again between requests.
#[test]
fn carried_state_is_never_skippable_and_the_kv_cache_always_is() {
    for n in [
        "kv.0.state",
        "kv.12.conv_state.q",
        "kv.12.conv_state_alt.q",
        "kv.blkres",
        "kv.3.state.v",
    ] {
        assert!(is_carried_state(n), "{n} is carried state");
        assert!(
            !kv_skips_zeroing(n),
            "{n} would keep stale bytes across a load"
        );
    }
    // The append-only cache: skippable at load, and nothing for begin_slot to do.
    for n in ["kv.0.k", "kv.31.v", "kv.7.latent"] {
        assert!(!is_carried_state(n), "{n} is append-only, not carried");
        assert!(kv_skips_zeroing(n), "{n} lost the 11.5 GiB memset skip");
    }
    // Outside the namespace entirely: neither question applies.
    for n in ["act.x", "model.layers.0.mlp.down_proj.weight", "in.pos"] {
        assert!(!is_carried_state(n));
        assert!(!kv_skips_zeroing(n), "{n} is not a kv. tensor");
    }
}

/// The negative fixture is real: a GLM-5.2 object set whose prefill object was
/// built WITHOUT `PLOW_MLA_PREFILL`, and pairing it with a GLM packet is exactly
/// the silent-garbage run this check exists to refuse. Skipped unless
/// `PLOW_TEST_GLM52_PREFILL_ELF` names that object.
#[test]
fn prefill_object_without_mla_arms_is_refused() {
    let Some(p) = std::env::var_os("PLOW_TEST_GLM52_PREFILL_ELF").map(std::path::PathBuf::from)
    else {
        return;
    };
    let Ok(img) = std::fs::read(&p) else { return };
    let syms = elf_symbol_names(&img);
    // The reader works on this file at all (it is where the rule was
    // derived): the interpreter body and the norms are there.
    assert!(syms.iter().any(|s| s.contains("plow_exec")), "{syms:?}");
    assert!(syms.iter().any(|s| s.contains("d_rmsnorm")), "{syms:?}");

    let e = check_prefill_object(&syms, &p, &["PLOW_MLA_PREFILL=1".into()])
        .expect_err("an object with no MLA-prefill symbol must be refused");
    let msg = e.to_string();
    assert!(msg.contains("PLOW_MLA_PREFILL"), "{msg}");
    assert!(msg.contains("interp_prefill.elf"), "{msg}");
    assert!(check_prefill_object(&syms, &p, &["PLOW_MOE_PREFILL=1".into()]).is_err());
    // `=0` and the filename-selected flags are not refusals.
    assert!(check_prefill_object(
        &syms,
        &p,
        &["PLOW_BUCKET_DECODE=0".into(), "PLOW_FP8=1".into()]
    )
    .is_ok());
}

/// A minimal ELF64 with one `.data` word per `(name, value)`, laid out the way
/// `pairing_stamp_is_read_from_elf_data` does it, so both [`elf_symbol_u32`] and
/// [`elf_symbol_names`] read it. Section 1 `.symtab`, 2 `.strtab`, 3 `.data`.
fn synthetic_object(symbols: &[(&str, u32)]) -> Vec<u8> {
    let n = symbols.len();
    let shoff = 0x40usize;
    let symtab_off = shoff + 4 * 64;
    let symtab_size = 24 * (n + 1);
    let strtab_off = symtab_off + symtab_size;
    let mut strtab = vec![0u8];
    let mut name_ofs = Vec::new();
    for (name, _) in symbols {
        name_ofs.push(strtab.len());
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
    }
    let data_off = (strtab_off + strtab.len() + 15) & !15;
    let data_size = 4 * n;
    let mut elf = vec![0u8; data_off + data_size];
    elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
    elf[0x28..0x30].copy_from_slice(&(shoff as u64).to_le_bytes());
    elf[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
    elf[0x3c..0x3e].copy_from_slice(&4u16.to_le_bytes());
    let s1 = shoff + 64;
    elf[s1 + 4..s1 + 8].copy_from_slice(&2u32.to_le_bytes());
    elf[s1 + 24..s1 + 32].copy_from_slice(&(symtab_off as u64).to_le_bytes());
    elf[s1 + 32..s1 + 40].copy_from_slice(&(symtab_size as u64).to_le_bytes());
    elf[s1 + 40..s1 + 44].copy_from_slice(&2u32.to_le_bytes());
    elf[s1 + 56..s1 + 64].copy_from_slice(&24u64.to_le_bytes());
    let s2 = shoff + 2 * 64;
    elf[s2 + 4..s2 + 8].copy_from_slice(&3u32.to_le_bytes());
    elf[s2 + 24..s2 + 32].copy_from_slice(&(strtab_off as u64).to_le_bytes());
    elf[s2 + 32..s2 + 40].copy_from_slice(&(strtab.len() as u64).to_le_bytes());
    let s3 = shoff + 3 * 64;
    elf[s3 + 4..s3 + 8].copy_from_slice(&1u32.to_le_bytes());
    elf[s3 + 16..s3 + 24].copy_from_slice(&0x1000u64.to_le_bytes());
    elf[s3 + 24..s3 + 32].copy_from_slice(&(data_off as u64).to_le_bytes());
    elf[s3 + 32..s3 + 40].copy_from_slice(&(data_size as u64).to_le_bytes());
    elf[strtab_off..strtab_off + strtab.len()].copy_from_slice(&strtab);
    for (k, (_, value)) in symbols.iter().enumerate() {
        let sym = symtab_off + 24 * (k + 1);
        elf[sym..sym + 4].copy_from_slice(&(name_ofs[k] as u32).to_le_bytes());
        elf[sym + 6..sym + 8].copy_from_slice(&3u16.to_le_bytes());
        elf[sym + 8..sym + 16].copy_from_slice(&(0x1000 + 4 * k as u64).to_le_bytes());
        elf[sym + 16..sym + 24].copy_from_slice(&4u64.to_le_bytes());
        let d = data_off + 4 * k;
        elf[d..d + 4].copy_from_slice(&value.to_le_bytes());
    }
    elf
}

#[test]
fn synthetic_object_is_read_by_both_elf_readers() {
    let elf = synthetic_object(&[("plow_geom_GM_BM", 192), ("plow_fp8_weights_1", 1)]);
    assert_eq!(elf_symbol_u32(&elf, "plow_geom_GM_BM"), Some(192));
    assert_eq!(elf_symbol_u32(&elf, "plow_fp8_weights_1"), Some(1));
    assert_eq!(elf_symbol_u32(&elf, "plow_geom_GM_BN"), None);
    let names = elf_symbol_names(&elf);
    assert!(names.contains(&"plow_geom_GM_BM") && names.contains(&FP8_WEIGHT_SYM));
}

/// The fp8 WEIGHT axis is refused on the OPCODE the phase dispatches, never on the
/// blob-wide `requires` entry: block-fp8 (`*Fp8Blk`) is outside `PLOW_FP8`.
#[test]
fn fp8_weight_object_is_refused_by_dispatched_opcode() {
    let path = Path::new("interp_prefill_mla_moe.elf");
    let bf16 = ["plow_interp_gfx942", FP8_KV_SYM];
    assert!(check_fp8_weight_arms(&bf16, path, None).is_ok());
    let err = check_fp8_weight_arms(&bf16, path, Some(DevOp::GemmFp8))
        .unwrap_err()
        .to_string();
    assert!(err.contains("packet/object FP8-WEIGHT MISMATCH"), "{err}");
    assert!(err.contains(FP8_WEIGHT_SYM), "{err}");
    assert!(
        err.contains("GemmFp8") && err.contains("interp_prefill_mla_moe.elf"),
        "{err}"
    );
    let fp8 = ["plow_interp_gfx942", FP8_WEIGHT_SYM];
    assert!(check_fp8_weight_arms(&fp8, path, Some(DevOp::GemmFp8)).is_ok());

    let glm = segmented_prog(&[DevOp::GemvFp8Blk, DevOp::DenseGluFp8Blk], &[0, 0]);
    assert_eq!(
        first_op_in(std::slice::from_ref(&glm), FP8_WEIGHT_OPS),
        None
    );
    let gemma = segmented_prog(&[DevOp::RmsNorm, DevOp::GemvFp8], &[0, 0]);
    assert_eq!(
        first_op_in(std::slice::from_ref(&gemma), FP8_WEIGHT_OPS),
        Some(DevOp::GemvFp8)
    );
    // Every op in the set has a `case` inside an `#if PLOW_FP8` block of interp.hip, and
    // the marker is emitted under the same guard.
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/amd/interp.hip");
    let src = std::fs::read_to_string(&p).expect("runtime/amd/interp.hip");
    assert!(src.contains(&format!(
        "#if PLOW_FP8\nextern \"C\" __device__ unsigned {FP8_WEIGHT_SYM} = 1;"
    )));
    for op in FP8_WEIGHT_OPS {
        assert!(
            src.contains(&format!("case {}:", op.c_name())),
            "{op:?} has no dispatch case"
        );
    }
}

/// The prefill GEMM tile is checked BY VALUE from the `geom_contract.h` markers.
#[test]
fn prefill_geometry_is_checked_by_value() {
    let path = Path::new("interp_prefill_fp8kv_mla_moe.elf");
    let requires: Vec<String> = [
        "PLOW_WG_WAVES=8",
        "GM_DBUF=1",
        "GM_BM=192",
        "GM_BN=256",
        "PLOW_BUCKET_DECODE=0",
        "PLOW_FP8=1",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let cdna3 = synthetic_object(&[
        ("plow_geom_PLOW_WG_WAVES", 8),
        ("plow_geom_GM_BM", 192),
        ("plow_geom_GM_BN", 256),
        ("plow_geom_GM_DBUF", 1),
    ]);
    assert!(check_prefill_geometry(&cdna3, path, &requires).is_ok());

    let recut = synthetic_object(&[
        ("plow_geom_PLOW_WG_WAVES", 8),
        ("plow_geom_GM_BM", 64),
        ("plow_geom_GM_BN", 256),
        ("plow_geom_GM_DBUF", 1),
    ]);
    let err = check_prefill_geometry(&recut, path, &requires)
        .unwrap_err()
        .to_string();
    assert!(err.contains("packet/object GEOMETRY MISMATCH"), "{err}");
    assert!(
        err.contains("GM_BM=192") && err.contains("GM_BM=64"),
        "{err}"
    );
    assert!(
        err.contains("plow_geom_GM_BM") && err.contains(&path.display().to_string()),
        "{err}"
    );

    // The 4-wave flash tile handed in as the prefill object.
    let flash = synthetic_object(&[
        ("plow_geom_PLOW_WG_WAVES", 4),
        ("plow_geom_GM_BM", 64),
        ("plow_geom_GM_BN", 128),
        ("plow_geom_GM_DBUF", 1),
    ]);
    let err = check_prefill_geometry(&flash, path, &requires)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("PLOW_WG_WAVES=8") && err.contains("PLOW_WG_WAVES=4"),
        "{err}"
    );

    // An object from before geom_contract.h: nothing to compare, refused by name.
    let unmarked = synthetic_object(&[("plow_geom_PLOW_WG_WAVES", 8)]);
    let err = check_prefill_geometry(&unmarked, path, &requires)
        .unwrap_err()
        .to_string();
    assert!(err.contains("packet/object GEOMETRY UNVERIFIABLE"), "{err}");
    assert!(err.contains("plow_geom_GM_DBUF"), "{err}");

    // No geometry named: nothing to check, whatever the image is.
    let arms_only = vec!["PLOW_MLA_PREFILL=1".to_string()];
    assert!(check_prefill_geometry(&[0u8; 64], path, &arms_only).is_ok());
    assert!(check_prefill_geometry(&unmarked, path, &[]).is_ok());

    // The geometry keys are owned here, so `check_prefill_object` neither refuses
    // nor warns about them; the same holds for the opcode-checked encodings.
    for (key, _) in PREFILL_GEOMETRY_MARKERS {
        assert!(VERIFIED_BY_OTHER_CHECKS.contains(key), "{key}");
        assert!(PREFILL_ARM_MARKERS.iter().all(|(f, _)| f != key));
    }
    for flag in VERIFIED_BY_OTHER_CHECKS {
        assert!(PREFILL_ARM_MARKERS.iter().all(|(f, _)| f != flag), "{flag}");
        assert!(DECODE_ARM_MARKERS.iter().all(|(f, _)| f != flag), "{flag}");
    }
    assert!(check_prefill_object(&["plow_interp_gfx942"], path, &requires).is_ok());
}

/// The pairing stamp end to end: read from the ELF, compared with the `build.json` beside
/// the packet. Unstamped objects stay accepted (the shipped state); a stamped object whose
/// hash disagrees, or that has no manifest to agree with, is refused.
#[test]
fn pairing_stamp_is_checked_against_the_manifest_beside_the_packet() {
    let dir = std::env::temp_dir().join(format!(
        "plow-pairing-stamp-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let blob = dir.join("model.pkt");
    let object = Path::new("interp_prefill_fp8kv_mla_moe.elf");
    let stamped = synthetic_object(&[
        ("plow_packet_hash_lo", 0x01fe_d7ac),
        ("plow_packet_hash_hi", 0x7843_2b97),
    ]);
    std::fs::write(
        dir.join("build.json"),
        br#"{"pairing":{"hash":"0x78432b9701fed7ac"}}"#,
    )
    .unwrap();
    assert!(check_packet_pairing_stamp(&stamped, &blob, object).is_ok());

    std::fs::write(
        dir.join("build.json"),
        br#"{"pairing":{"hash":"0x78432b9701fed7ad"}}"#,
    )
    .unwrap();
    let err = check_packet_pairing_stamp(&stamped, &blob, object)
        .unwrap_err()
        .to_string();
    assert!(err.contains("packet/object MISMATCH"), "{err}");
    assert!(
        err.contains("0x78432b9701fed7ac") && err.contains("0x78432b9701fed7ad"),
        "{err}"
    );

    // General (unstamped) object: accepted against any manifest, and with none.
    let general = synthetic_object(&[("plow_geom_GM_BM", 192)]);
    assert!(check_packet_pairing_stamp(&general, &blob, object).is_ok());
    std::fs::remove_file(dir.join("build.json")).unwrap();
    assert!(check_packet_pairing_stamp(&general, &blob, object).is_ok());
    // A stamped object with no manifest to agree with is not a general object.
    let err = check_packet_pairing_stamp(&stamped, &blob, object)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("0x78432b9701fed7ac") && err.contains("build.json"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Junk stays bounded and cannot establish a required prefill capability.
#[test]
fn elf_reader_is_bounded_on_junk() {
    assert!(elf_symbol_names(b"").is_empty());
    assert!(elf_symbol_names(b"\x7fELF\x02\x01").is_empty());
    assert!(elf_symbol_names(&[0xffu8; 4096]).is_empty());
    let junk = elf_symbol_names(&[0u8; 128]);
    let err = check_prefill_object(&junk, Path::new("x.elf"), &["PLOW_MLA_PREFILL=1".into()])
        .expect_err("a required arm needs a verifiable symbol table");
    assert!(err.to_string().contains("no ELF symbol table"));
    assert!(check_prefill_object(&junk, Path::new("x.elf"), &[]).is_ok());
    assert!(
        check_prefill_object(&junk, Path::new("x.elf"), &["PLOW_MLA_PREFILL=0".into()]).is_ok()
    );
    // Missing GEMV capacity metadata admits only the legacy B1 contract.
    assert!(check_gemv_capacity(&junk, Path::new("x.elf"), 1).is_ok());
    assert!(check_gemv_capacity(&junk, Path::new("x.elf"), 2).is_err());
}

/// The load-time zeroing skip is about WRITE-BEFORE-READ, not about the
/// `kv.` prefix. K3 names two read-modify-write things `kv.` so the loader
/// does not demand them of the checkpoint, and they must still be zeroed:
/// uninitialised HBM in a recurrence is garbage that never washes out, and
/// it neither faults nor reports a missing weight.
#[test]
fn only_append_only_kv_caches_skip_zeroing() {
    // Append-only caches: written before read, so the skip is sound.
    for n in ["kv.0.k", "kv.0.v", "kv.3.ckv", "kv.3.krot", "kv.7.kidx"] {
        assert!(kv_skips_zeroing(n), "`{n}` is an append-only cache");
    }
    // Read-modify-write state, and the AttnRes snapshot ring.
    for n in [
        "kv.2.state",
        "kv.2.conv_state.q",
        "kv.2.conv_state.k",
        "kv.2.conv_state.v",
        "kv.blkres",
    ] {
        assert!(!kv_skips_zeroing(n), "`{n}` is READ before it is written");
    }
    // Everything outside the namespace was always zeroed and still is.
    for n in ["act.x", "in.ids", "moe.expert_weight_table"] {
        assert!(!kv_skips_zeroing(n));
    }
}

#[test]
fn decode_tier_discovery_matches_variant_scheduler_and_orders_all_widths() {
    let root = std::env::temp_dir().join(format!(
        "plow-decode-tier-discovery-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    for (dir, file) in [
        ("lowrung20", "interp_decode_fp8kv_gq.elf"),
        ("lowrung16", "interp_decode_fp8kv_gq.elf"),
        ("lowrung1", "interp_decode_fp8kv_gq.elf"),
        ("lowrung2", "interp_decode_gq.elf"),
        ("lowrung4", "interp_decode_fp8kv.elf"),
        ("lowrung0", "interp_decode_fp8kv_gq.elf"),
        ("lowrungbad", "interp_decode_fp8kv_gq.elf"),
        ("lowrung8", "unfinished.tmp"),
    ] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        std::fs::write(root.join(dir).join(file), []).unwrap();
    }
    for (variant, sched, widths) in [
        (Variant::Fp8Kv, Sched::GlobalQueue, vec![1, 16, 20]),
        (Variant::Fp8Kv, Sched::Static, vec![4]),
        (Variant::Bf16, Sched::GlobalQueue, vec![2]),
        (Variant::Fp8, Sched::GlobalQueue, vec![]),
    ] {
        let name = object_name(Phase::Decode, variant, PrefillArm::MlaMoe, sched);
        let want = (!widths.is_empty()).then(|| {
            widths
                .iter()
                .map(|w| format!("{}:{w}", root.join(format!("lowrung{w}")).display()))
                .collect::<Vec<_>>()
                .join(",")
        });
        assert_eq!(discover_lowrung_tiers(&root, &name), want);
    }
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(discover_lowrung_tiers(&root, "interp_decode_gq.elf"), None);
}

#[test]
fn native_moe_decode_keeps_ordered_xcd_boundaries() {
    let make = || {
        let mut p = segmented_decode_probe();
        p.t = 8;
        p.insts[1].op = DevOp::MoeAiterFp8Pf as u16;
        p
    };
    let mut p = make();
    assert_eq!(
        decode_segment_kinds(&p).unwrap()[1],
        DecodeSegmentKind::MoeAiter
    );
    validate_decode_dispatch(std::slice::from_ref(&p), 0).unwrap();
    p.l2_domains = 8;
    assert!(validate_decode_dispatch(std::slice::from_ref(&p), 0).is_err());
    p.gq_stream = p.stream.clone();
    p.gq_seg_ofs = std::iter::once(0)
        .chain((1..=3).flat_map(|n| std::iter::repeat_n(n, 8)))
        .collect();
    validate_decode_dispatch(std::slice::from_ref(&p), 0).unwrap();
    for bad in 0..4 {
        let mut p = make();
        match bad {
            0 => p.stream[1].wait_len = 1,
            1 => p.stream[1].succ_len = 1,
            2 => p.stream[1].flags |= SE_XCTR,
            _ => p.stream[0].seg = 1,
        }
        assert!(decode_segment_kinds(&p).is_err(), "case {bad}");
    }
}

#[test]
fn native_gemm_decode_keeps_ordered_xcd_boundaries() {
    let make = || {
        let mut p = segmented_decode_probe();
        p.t = 20;
        p.insts[1].op = DevOp::GemmLtPf as u16;
        p
    };
    let mut p = make();
    assert_eq!(
        decode_segment_kinds(&p).unwrap()[1],
        DecodeSegmentKind::GemmLt
    );
    validate_decode_dispatch(std::slice::from_ref(&p), 0).unwrap();
    p.l2_domains = 8;
    assert!(validate_decode_dispatch(std::slice::from_ref(&p), 0).is_err());
    p.gq_stream = p.stream.clone();
    p.gq_seg_ofs = std::iter::once(0)
        .chain((1..=3).flat_map(|n| std::iter::repeat_n(n, 8)))
        .collect();
    validate_decode_dispatch(std::slice::from_ref(&p), 0).unwrap();
    for bad in 0..4 {
        let mut p = make();
        match bad {
            0 => p.stream[1].wait_len = 1,
            1 => p.stream[1].succ_len = 1,
            2 => p.stream[1].flags |= SE_XCTR,
            _ => p.stream[0].seg = 1,
        }
        assert!(decode_segment_kinds(&p).is_err(), "case {bad}");
    }
}

#[test]
fn resident_moe_weight_layout_matches_gpu_vector_packing() {
    for (rows, k) in [(256usize, 6144usize), (6144, 256)] {
        let src: Vec<u8> = (0..rows * k)
            .map(|i| ((i * 31 + i / 251) % 256) as u8)
            .collect();
        let actual = shuffle_moe_weight_16x32(&src, rows, k).unwrap();
        for v in 0..rows * k / 16 {
            let row = (v / (32 * (k / 32))) * 16 + v % 16;
            let col = (v / 32) % (k / 32) * 2 + (v / 16) % 2;
            let from = row * k + col * 16;
            assert_eq!(actual[v * 16..v * 16 + 16], src[from..from + 16]);
        }
    }
    let scales: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.0125).collect();
    let actual = resident_moe_scales(bytemuck::cast_slice(&scales)).unwrap();
    let expected: Vec<f32> = scales.iter().map(|s| s * 2.0).collect();
    assert_eq!(actual, bytemuck::cast_slice::<f32, u8>(&expected));
    for value in [f32::NAN, f32::INFINITY, f32::MAX] {
        assert!(resident_moe_scales(bytemuck::cast_slice(&vec![value; 96])).is_err());
    }
}
