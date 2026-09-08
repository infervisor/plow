//! Contract tests for the unified token batch's planner and row resolver.
//!
//! The §9 "Planner" row: D=0,1,2,4; multiple intermediate/final spans; one-token prompts;
//! ragged positions; S=0 — with exact row/owner/frontier maps and no padded output. Plus the
//! "Row resolver" row's malformed tables, checked against the Rust twin here and against the
//! shared C header in `crates/plowrt/tests/token_batch_resolver.rs`.

use super::*;

const CAP: usize = 16;

fn decode(id: u32, slot: u32, prompt_len: u32, token: &'static [u32]) -> Request<'static> {
    Request {
        id,
        slot,
        state_slot: slot,
        generation: 1,
        phase: Phase::Decode,
        tokens: token,
        prompt_len,
        selection: Selection::default(),
    }
}

fn prefill(id: u32, slot: u32, prompt_len: u32, tokens: &'static [u32]) -> Request<'static> {
    Request {
        id,
        slot,
        state_slot: slot,
        generation: 1,
        phase: Phase::Prefill,
        tokens,
        prompt_len,
        selection: Selection::default(),
    }
}

fn tables(frontiers: &[u32]) -> (Vec<u32>, Vec<u32>) {
    (frontiers.to_vec(), vec![1u32; frontiers.len()])
}

/// §6.1's worked example, which the plan calls "the acceptance case for every family on every
/// backend": two ongoing decodes, a prompt that finishes, a prompt that needs another chunk.
#[test]
fn the_section_six_example_produces_its_exact_row_and_sample_maps() {
    static A: [u32; 1] = [7];
    static B: [u32; 1] = [9];
    static C: [u32; 50] = [3; 50];
    static D: [u32; 80] = [4; 80];
    let (frontiers, generations) = tables(&[100, 900, 70, 0]);
    let requests = [
        decode(10, 0, 100, &A),
        decode(11, 1, 900, &B),
        prefill(12, 2, 120, &C),
        prefill(13, 3, 4096, &D),
    ];
    let plan = plan(&requests, &frontiers, &generations, 132, 8192, 0).unwrap();

    assert_eq!(plan.real_rows, 132, "M");
    assert_eq!(plan.sample_rows, 3, "S = 2 decodes + 1 completing prompt");
    assert_eq!(plan.sample_input_rows, vec![0, 1, 51]);
    assert_eq!(
        plan.sample_owners
            .iter()
            .map(|o| (o.request, o.slot))
            .collect::<Vec<_>>(),
        vec![(10, 0), (11, 1), (12, 2)],
    );
    // C's selected row is its LAST scheduled prompt token (position 119). Its first generated
    // token is at 120, which is not yet in KV and becomes the input of C's next decode step.
    assert_eq!(plan.positions[51], 119);
    assert_eq!(plan.positions[2], 70, "C's first row is its frontier");
    assert_eq!(plan.positions[131], 79, "D's last intermediate row");
    // D contributes tokens to M and zero to S.
    assert!(plan.pending[3].sample_index.is_none());
    assert_eq!(
        plan.pending
            .iter()
            .map(|p| (p.slot, p.expected_frontier, p.new_frontier, p.completes_prompt))
            .collect::<Vec<_>>(),
        vec![
            (0, 100, 101, true),
            (1, 900, 901, true),
            (2, 70, 120, true),
            (3, 0, 80, false),
        ],
    );
    assert!(plan.active[..132].iter().all(|&a| a == 1));
    assert_eq!(plan.row_capacity, 132, "no padding in this example");
}

/// D = 0, 1, 2, 4 decode requests, each with the same three prefill spans behind them. The
/// decode rows always occupy the prefix, the spans always cover [0, M) exactly.
#[test]
fn decode_counts_zero_through_four_keep_a_dense_cover() {
    static ONE: [u32; 1] = [1];
    static CHUNK: [u32; 3] = [5, 6, 7];
    for d in [0usize, 1, 2, 4] {
        let mut requests = Vec::new();
        for i in 0..d {
            requests.push(decode(i as u32, i as u32, 64, &ONE));
        }
        requests.push(prefill(100, 8, 3, &CHUNK));
        let (frontiers, generations) = tables(&[64, 64, 64, 64, 64, 64, 64, 64, 0]);
        let plan = plan(&requests, &frontiers, &generations, 16, 4096, 0).unwrap();
        assert_eq!(plan.real_rows as usize, d + 3);
        assert_eq!(plan.spans.len(), d + 1);
        assert_eq!(plan.sample_rows as usize, d + 1, "every decode plus the prompt");
        // Decode spans first, in caller order, then prefill.
        for i in 0..d {
            assert_eq!(plan.spans[i].row0, i as u32);
            assert_eq!(plan.spans[i].n_rows, 1);
            assert_eq!(plan.phases[i], Phase::Decode);
        }
        assert_eq!(plan.spans[d].row0, d as u32);
        assert_eq!(plan.phases[d], Phase::Prefill);
        validate(&plan.batch()).unwrap();
    }
}

/// A one-token prompt is ONE ordinary span with ONE selected output row — not a decode span,
/// and not a prompt that needs a replayed final token.
#[test]
fn a_one_token_prompt_is_one_span_with_one_sample_row() {
    static ONE: [u32; 1] = [42];
    let (frontiers, generations) = tables(&[0]);
    let plan = plan(&[prefill(1, 0, 1, &ONE)], &frontiers, &generations, 8, 4096, 0).unwrap();
    assert_eq!((plan.real_rows, plan.sample_rows), (1, 1));
    assert_eq!(plan.phases, vec![Phase::Prefill]);
    assert_eq!(plan.sample_input_rows, vec![0]);
    assert_eq!(plan.positions[0], 0);
    assert!(plan.pending[0].completes_prompt);
}

/// A FINAL prefill chunk of length one is still a prefill span, and it still samples. Length
/// is never what decides a span's phase.
#[test]
fn a_final_chunk_of_length_one_is_prefill_and_samples() {
    static ONE: [u32; 1] = [42];
    let (frontiers, generations) = tables(&[99]);
    let plan = plan(&[prefill(1, 0, 100, &ONE)], &frontiers, &generations, 8, 4096, 0).unwrap();
    assert_eq!(plan.phases, vec![Phase::Prefill]);
    assert_eq!(plan.sample_rows, 1);
    assert_eq!(plan.positions[0], 99);
    assert_eq!(plan.pending[0].new_frontier, 100);
}

/// Every span is an intermediate chunk, so S = 0 and there is no output segment at all. Zero
/// is zero: the legacy `n_batch == 0 means one row` reading would sample row 0, which belongs
/// to a request that asked for nothing.
#[test]
fn all_intermediate_chunks_give_s_zero() {
    static CHUNK: [u32; 4] = [1, 2, 3, 4];
    let (frontiers, generations) = tables(&[0, 8]);
    let plan = plan(
        &[prefill(1, 0, 64, &CHUNK), prefill(2, 1, 64, &CHUNK)],
        &frontiers,
        &generations,
        8,
        4096,
        0,
    )
    .unwrap();
    assert_eq!(plan.sample_rows, 0);
    assert!(plan.sample_input_rows.is_empty() && plan.sample_owners.is_empty());
    assert!(plan.pending.iter().all(|p| p.sample_index.is_none()));
    assert_eq!(sample_row(&plan.batch(), 0), Err(Refusal::Sample));
}

/// Ragged positions: three requests at completely different frontiers in one batch. Each row's
/// position comes from its OWN span, never from the packed row index.
#[test]
fn ragged_positions_come_from_each_span_not_the_row_index() {
    static ONE: [u32; 1] = [1];
    static TWO: [u32; 2] = [1, 2];
    static THREE: [u32; 3] = [1, 2, 3];
    let (frontiers, generations) = tables(&[5, 1000, 0]);
    let plan = plan(
        &[
            decode(1, 0, 5, &ONE),
            prefill(2, 1, 1002, &TWO),
            prefill(3, 2, 900, &THREE),
        ],
        &frontiers,
        &generations,
        8,
        4096,
        0,
    )
    .unwrap();
    assert_eq!(&plan.positions[..6], &[5, 1000, 1001, 0, 1, 2]);
    for row in 0..plan.real_rows {
        let r = resolve_row(&plan.batch(), row).unwrap();
        assert_eq!(r.position, plan.positions[row as usize]);
        assert!(r.active);
    }
}

/// Padding belongs to no span, carries no request's coordinates, and is inert. This is the
/// invariant the mask exists for: `active[row] == 0` rows never write KV or produce output.
#[test]
fn padding_is_inert_and_owned_by_nobody() {
    static ONE: [u32; 1] = [42];
    let (frontiers, generations) = tables(&[7]);
    let plan = plan(&[decode(1, 0, 7, &ONE)], &frontiers, &generations, 8, 4096, 0).unwrap();
    assert_eq!((plan.real_rows, plan.row_capacity), (1, 8));
    assert_eq!(&plan.active, &[1, 0, 0, 0, 0, 0, 0, 0]);
    assert!(plan.input_ids[1..].iter().all(|&t| t == 0));
    assert!(plan.positions[1..].iter().all(|&p| p == 0));
    for row in 1..8 {
        let r = resolve_row(&plan.batch(), row).unwrap();
        assert_eq!(r.span, None);
        assert!(!r.active);
    }
    assert_eq!(resolve_row(&plan.batch(), 8), Err(Refusal::Row));
}

/// Context limits are checked against the ADMITTED maximum, before any device sees the plan.
#[test]
fn a_span_past_the_context_limit_is_refused() {
    static CHUNK: [u32; 4] = [1, 2, 3, 4];
    let (frontiers, generations) = tables(&[125]);
    let err = plan(&[prefill(1, 0, 200, &CHUNK)], &frontiers, &generations, 8, 128, 0).unwrap_err();
    assert!(err.contains("context limit"), "{err}");
    assert!(plan(&[prefill(1, 0, 200, &CHUNK)], &frontiers, &generations, 8, 129, 0).is_ok());
}

/// Ownership: a span must start at its slot's committed frontier and carry the generation the
/// host currently holds. Both are how a recycled slot is stopped from receiving another
/// request's work.
#[test]
fn invalid_ownership_is_refused_at_plan_time() {
    static ONE: [u32; 1] = [1];
    let (frontiers, generations) = tables(&[10, 10]);

    let mut stale = decode(1, 0, 10, &ONE);
    stale.generation = 2;
    let err = plan(&[stale], &frontiers, &generations, 8, 4096, 0).unwrap_err();
    assert!(err.contains("generation"), "{err}");

    // A decode row before the prompt end is a request that has not finished prefilling.
    let err = plan(&[decode(1, 0, 64, &ONE)], &frontiers, &generations, 8, 4096, 0).unwrap_err();
    assert!(err.contains("decode span"), "{err}");

    // Two spans on one physical slot would both start at its frontier and both advance it.
    let err = plan(
        &[decode(1, 0, 10, &ONE), decode(2, 0, 10, &ONE)],
        &frontiers,
        &generations,
        8,
        4096,
        0,
    )
    .unwrap_err();
    assert!(err.contains("duplicate"), "{err}");

    // A prefill span that overruns its own prompt.
    static FIVE: [u32; 5] = [1, 2, 3, 4, 5];
    let err = plan(&[prefill(1, 1, 12, &FIVE)], &frontiers, &generations, 8, 4096, 0).unwrap_err();
    assert!(err.contains("overruns"), "{err}");
}

/// A refused plan leaves NO partially filled state: a half-built plan is worse than none
/// because it still looks stageable.
#[test]
fn a_refused_plan_leaves_no_residue() {
    static ONE: [u32; 1] = [1];
    static BIG: [u32; 9] = [1; 9];
    let (frontiers, generations) = tables(&[0, 0]);
    let mut plan = Plan::with_capacity(CAP, 4);
    assert!(plan_into(
        &[prefill(1, 0, 9, &BIG)],
        &frontiers,
        &generations,
        8,
        4096,
        0,
        &mut plan
    )
    .is_err());
    assert_eq!(plan, Plan::with_capacity(CAP, 4));
    // And the same storage plans correctly afterwards.
    plan_into(
        &[prefill(1, 0, 1, &ONE)],
        &frontiers,
        &generations,
        8,
        4096,
        0,
        &mut plan,
    )
    .unwrap();
    assert_eq!(plan.real_rows, 1);
}

/// Every malformed table the §9 row names, against the Rust twin. The C header is checked
/// against these same shapes in `plowrt`.
#[test]
fn malformed_tables_are_refused_by_the_named_invariant() {
    let good = |spans: Vec<PrefillSpan>, m: u32| {
        let mut positions = vec![0u32; 8];
        let mut active = vec![0u32; 8];
        // Fill in REVERSE span order so the earliest span wins a contested row. Otherwise an
        // overlap perturbs positions[] too and is reported as a position mismatch, which
        // hides which invariant the malformed table actually broke.
        for span in spans.iter().rev() {
            for j in 0..span.n_rows {
                positions[(span.row0 + j) as usize] = span.kv_row0 + j;
                active[(span.row0 + j) as usize] = 1;
            }
        }
        (spans, positions, active, m)
    };
    let span = |row0, n_rows, kv_row0| PrefillSpan {
        row0,
        n_rows,
        slot: 0,
        flags: 0,
        kv_row0,
        kv_len: kv_row0 + n_rows,
        state_slot: 0,
        program: 0,
    };
    let run = |(spans, positions, active, m): (Vec<PrefillSpan>, Vec<u32>, Vec<u32>, u32)| {
        validate(&HostBatch {
            row_capacity: 8,
            real_rows: m,
            sample_rows: 0,
            spans: &spans,
            positions: &positions,
            active: &active,
            sample_input_rows: &[],
        })
    };

    // Baseline: two spans covering [0, 4) exactly.
    assert_eq!(run(good(vec![span(0, 2, 0), span(2, 2, 10)], 4)), Ok(()));

    // Gap: the second span starts one row late.
    assert_eq!(
        run(good(vec![span(0, 2, 0), span(3, 1, 10)], 4)),
        Err(Refusal::Cover)
    );
    // Overlap.
    assert_eq!(
        run(good(vec![span(0, 3, 0), span(2, 2, 10)], 4)),
        Err(Refusal::Cover)
    );
    // Zero length.
    assert_eq!(
        run(good(vec![span(0, 0, 0), span(0, 4, 0)], 4)),
        Err(Refusal::Cover)
    );
    // Short cover: spans stop before M.
    assert_eq!(
        run(good(vec![span(0, 2, 0)], 4)),
        Err(Refusal::Cover)
    );
    // Position mismatch.
    let (spans, mut positions, active, m) = good(vec![span(0, 2, 0), span(2, 2, 10)], 4);
    positions[3] = 99;
    assert_eq!(
        run((spans, positions, active, m)),
        Err(Refusal::Position)
    );
    // A live mask bit past M: the filler and the planner disagree about who owns the row.
    let (spans, positions, mut active, m) = good(vec![span(0, 4, 0)], 4);
    active[5] = 1;
    assert_eq!(run((spans, positions, active, m)), Err(Refusal::Active));
    // kv_len that does not match its own span.
    let (mut spans, positions, active, m) = good(vec![span(0, 4, 0)], 4);
    spans[0].kv_len = 7;
    assert_eq!(run((spans, positions, active, m)), Err(Refusal::KvLen));
    // A sample index outside the live rows.
    let (spans, positions, active, _) = good(vec![span(0, 4, 0)], 4);
    assert_eq!(
        validate(&HostBatch {
            row_capacity: 8,
            real_rows: 4,
            sample_rows: 1,
            spans: &spans,
            positions: &positions,
            active: &active,
            sample_input_rows: &[4],
        }),
        Err(Refusal::Sample)
    );
}

/// The load-time capability check names the capability it refused — never a silent dense or
/// no-op fallback.
#[test]
fn refusals_name_the_capability() {
    let caps = |aware: Vec<u16>, converted: Vec<u16>| Capabilities {
        target: "gfx942".into(),
        descriptor_version: TOKEN_BATCH_VERSION,
        row_capacity: 256,
        sample_capacity: 4,
        max_prefill_spans: 1,
        descriptor_aware: aware,
        converted_c: converted,
    };
    let gemm = DevOp::Gemm as u16;
    let prefill_op = DevOp::FlashPrefill as u16;
    let gdn = DevOp::QwenGdnPrefill as u16;
    let gather = DevOp::RowGather as u16;

    // A-class opcode with an arm: admitted.
    let entries = caps(vec![gemm], vec![])
        .refuse_program([gemm].into_iter())
        .unwrap();
    assert_eq!(entries.len(), 1);
    assert!(entries[0].admitted() && entries[0].class == RowClass::A);

    // A-class opcode with NO arm: refused, named. This is the AMD `default:` hazard.
    let err = caps(vec![], vec![])
        .refuse_program([gemm].into_iter())
        .unwrap_err();
    assert_eq!(err.capability, "gfx942.PLOW_DOP_GEMM");
    assert!(err.detail.contains("writes nothing"), "{}", err.detail);

    // C-class with an arm but no conversion: refused, and the message says why C is the hazard.
    let err = caps(vec![prefill_op], vec![])
        .refuse_program([prefill_op].into_iter())
        .unwrap_err();
    assert_eq!(err.capability, "gfx942.PLOW_DOP_FLASH_PREFILL");
    assert!(err.detail.contains("class C"), "{}", err.detail);
    // Declared converted: admitted.
    assert!(caps(vec![prefill_op], vec![prefill_op])
        .refuse_program([prefill_op].into_iter())
        .is_ok());

    // D-class: refused with the per-sequence-state reason.
    let err = caps(vec![gdn], vec![])
        .refuse_program([gdn].into_iter())
        .unwrap_err();
    assert!(err.detail.contains("class D"), "{}", err.detail);

    // An unknown opcode is treated as C and refused.
    let err = caps(vec![9999], vec![])
        .refuse_program([9999u16].into_iter())
        .unwrap_err();
    assert_eq!(err.capability, "gfx942.opcode_9999");
    assert!(err.detail.contains("class C"), "{}", err.detail);

    // Descriptor version mismatch is refused before any opcode is looked at.
    let mut wrong = caps(vec![gemm], vec![]);
    wrong.descriptor_version = TOKEN_BATCH_VERSION + 1;
    assert_eq!(
        wrong.refuse_program([gemm].into_iter()).unwrap_err().capability,
        format!("gfx942.{OBJECT_CAPABILITY}"),
    );

    // "Armed" and "can fire" are different claims.
    assert!(caps(vec![gemm], vec![]).can_run_output().is_err());
    assert!(caps(vec![gemm, gather], vec![]).can_run_output().is_ok());
}

/// Capacities and the D-class prefill-span limit are refused at admission, not truncated.
#[test]
fn plan_capacities_and_the_d_class_limit_are_refused_not_truncated() {
    static ONE: [u32; 1] = [1];
    static TWO: [u32; 2] = [1, 2];
    let caps = Capabilities {
        target: "cpu".into(),
        descriptor_version: TOKEN_BATCH_VERSION,
        row_capacity: 8,
        sample_capacity: 1,
        max_prefill_spans: 1,
        descriptor_aware: vec![],
        converted_c: vec![],
    };
    let (frontiers, generations) = tables(&[0, 0, 5]);

    let one_span = plan(&[prefill(1, 0, 2, &TWO)], &frontiers, &generations, 8, 64, 0).unwrap();
    assert!(caps.refuse_plan(&one_span).is_ok());

    // Two INTERMEDIATE chunks, so S = 0 and only the D-class limit can fire. Isolating it
    // matters: the refusal has to name the span limit, not whatever check runs first.
    let two_spans = plan(
        &[prefill(1, 0, 64, &TWO), prefill(2, 1, 64, &TWO)],
        &frontiers,
        &generations,
        8,
        64,
        0,
    )
    .unwrap();
    assert_eq!(two_spans.sample_rows, 0);
    let err = caps.refuse_plan(&two_spans).unwrap_err();
    assert_eq!(err.capability, "cpu.d_class_prefill_spans");
    assert!(err.detail.contains("no request axis"), "{}", err.detail);

    // Decode spans do not count against the D-class prefill limit: the decode side is class B
    // and packs through the `active[]` path.
    let decodes = plan(
        &[decode(3, 2, 5, &ONE), prefill(1, 0, 2, &TWO)],
        &frontiers,
        &generations,
        8,
        64,
        0,
    )
    .unwrap();
    assert_eq!(
        caps.refuse_plan(&decodes).unwrap_err().capability,
        "cpu.sample_capacity",
        "S=2 exceeds the compiled sample capacity, and that is what should be named",
    );

    let wide = plan(&[prefill(1, 0, 2, &TWO)], &frontiers, &generations, 16, 64, 0).unwrap();
    assert_eq!(
        caps.refuse_plan(&wide).unwrap_err().capability,
        "cpu.row_capacity"
    );
}

/// The descriptor refuses a null row array and an address/count disagreement, because a
/// descriptor whose counts and pointers disagree is exactly the shape that reads unrelated
/// memory without faulting.
#[test]
fn the_descriptor_requires_addresses_that_match_its_counts() {
    static ONE: [u32; 1] = [1];
    let (frontiers, generations) = tables(&[3]);
    let p = plan(&[decode(1, 0, 3, &ONE)], &frontiers, &generations, 4, 64, 0).unwrap();
    let d = descriptor(&p, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000).unwrap();
    assert_eq!(d.version, TOKEN_BATCH_VERSION);
    assert_eq!((d.real_rows, d.sample_rows, d.n_spans), (1, 1, 1));
    assert_eq!((d.flags, d._pad0, d._pad1), (0, 0, 0));
    assert!(descriptor(&p, 0x1000, 0, 0x3000, 0x4000, 0x5000).is_err());
    assert!(descriptor(&p, 0, 0x2000, 0x3000, 0x4000, 0x5000).is_err());
    assert!(
        descriptor(&p, 0x1000, 0x2000, 0x3000, 0x4000, 0).is_err(),
        "S > 0 with no sample-row address"
    );

    // S = 0 must NOT carry a sample-row address: a stale pointer beside a zero count is how a
    // zero-output step still reads somebody's row list.
    static CHUNK: [u32; 3] = [1, 2, 3];
    let intermediate = plan(
        &[prefill(1, 0, 64, &CHUNK)],
        &tables(&[0]).0,
        &tables(&[0]).1,
        4,
        64,
        0,
    )
    .unwrap();
    assert_eq!(intermediate.sample_rows, 0);
    assert!(descriptor(&intermediate, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000).is_err());
    assert!(descriptor(&intermediate, 0x1000, 0x2000, 0x3000, 0x4000, 0).is_ok());
}
