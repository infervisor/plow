//! §9 "Row resolver": the shared C header vs the Rust twin, over identical span tables.
//!
//! `runtime/common/token_batch.h` is compiled by HIP and CUDA as device code and by the host
//! through `runtime/cpu/dev/token_batch.c`; `plow_asset::token_batch` carries a Rust twin
//! because the planner has to validate a plan before any device sees it. Two implementations
//! that are never compared are two implementations — so this runs both over the same tables,
//! including every malformed one the plan names (gap, overlap, zero length, position mismatch),
//! and compares the acceptance, the exact refusal code, and every resolved field.
#![cfg(feature = "cpu")]

use packet::dev::{PrefillSpan, TokenBatch, TOKEN_BATCH_VERSION};
use plow_asset::token_batch::{self, HostBatch, Phase, Refusal, Request, Selection};
use plowrt::exec::cpu::ffi;

/// Host image whose arrays outlive the descriptor built from them.
struct Image {
    spans: Vec<PrefillSpan>,
    input_ids: Vec<u32>,
    positions: Vec<u32>,
    active: Vec<u32>,
    sample_rows: Vec<u32>,
    row_capacity: u32,
    real_rows: u32,
    n_sample: u32,
}

impl Image {
    fn batch(&self) -> HostBatch<'_> {
        HostBatch {
            row_capacity: self.row_capacity,
            real_rows: self.real_rows,
            sample_rows: self.n_sample,
            spans: &self.spans,
            positions: &self.positions,
            active: &self.active,
            sample_input_rows: &self.sample_rows,
        }
    }

    fn descriptor(&self) -> TokenBatch {
        TokenBatch {
            version: TOKEN_BATCH_VERSION,
            row_capacity: self.row_capacity,
            real_rows: self.real_rows,
            sample_rows: self.n_sample,
            n_spans: self.spans.len() as u32,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
            spans: self.spans.as_ptr() as u64,
            input_ids: self.input_ids.as_ptr() as u64,
            positions: self.positions.as_ptr() as u64,
            active: self.active.as_ptr() as u64,
            sample_rows_idx: if self.n_sample == 0 {
                0
            } else {
                self.sample_rows.as_ptr() as u64
            },
        }
    }
}

fn span(row0: u32, n_rows: u32, kv_row0: u32, slot: u32) -> PrefillSpan {
    PrefillSpan {
        row0,
        n_rows,
        slot,
        flags: 0,
        kv_row0,
        kv_len: kv_row0 + n_rows,
        state_slot: slot + 100,
        program: 0,
    }
}

/// Build a well-formed image, then let the caller break exactly one thing.
fn image(spans: Vec<PrefillSpan>, row_capacity: u32, samples: Vec<u32>) -> Image {
    let real_rows = spans.iter().map(|s| s.row0 + s.n_rows).max().unwrap_or(0);
    let mut positions = vec![0u32; row_capacity as usize];
    let mut active = vec![0u32; row_capacity as usize];
    // Reverse order so the earliest span owns a contested row; an overlap then breaks the
    // COVER invariant rather than incidentally breaking positions[] too.
    for s in spans.iter().rev() {
        for j in 0..s.n_rows {
            positions[(s.row0 + j) as usize] = s.kv_row0 + j;
            active[(s.row0 + j) as usize] = 1;
        }
    }
    let n_sample = samples.len() as u32;
    Image {
        spans,
        input_ids: vec![7; row_capacity as usize],
        positions,
        active,
        sample_rows: samples,
        row_capacity,
        real_rows,
        n_sample,
    }
}

/// Both resolvers agree on acceptance and on the exact refusal code.
fn agree(image: &Image) -> Result<(), i32> {
    let rust = token_batch::validate(&image.batch());
    let descriptor = image.descriptor();
    // SAFETY: every pointer in `descriptor` borrows `image`'s live host arrays, each sized for
    // the counts the descriptor declares.
    let c = unsafe { ffi::token_batch_validate(&descriptor) };
    match (rust, c) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(r), Err(c)) => {
            assert_eq!(r.code(), c.0, "twins disagree on WHICH invariant failed");
            Err(c.0)
        }
        (r, c) => panic!("twins disagree on acceptance: rust={r:?} c={c:?}"),
    }
}

#[test]
fn the_descriptor_version_matches_across_the_language_boundary() {
    assert_eq!(ffi::token_batch_descriptor_version(), TOKEN_BATCH_VERSION);
}

#[test]
fn both_resolvers_resolve_every_row_identically() {
    // Two decodes at different frontiers, a completing prompt, an intermediate chunk, and a
    // padded tail — the §6.1 example's shape, shrunk so every row can be checked.
    let img = image(
        vec![
            span(0, 1, 100, 0),
            span(1, 1, 900, 1),
            span(2, 4, 70, 2),
            span(6, 3, 0, 3),
        ],
        12,
        vec![0, 1, 5],
    );
    agree(&img).unwrap();
    let descriptor = img.descriptor();
    for row in 0..img.row_capacity {
        let rust = token_batch::resolve_row(&img.batch(), row).unwrap();
        // SAFETY: as in `agree`.
        let c = unsafe { ffi::token_row(&descriptor, row) }.unwrap();
        assert_eq!(
            rust.span.unwrap_or(ffi::TokenRowFlat::SPAN_NONE),
            c.span,
            "row {row} span"
        );
        assert_eq!(rust.local_row, c.local_row, "row {row} local");
        assert_eq!(rust.position, c.position, "row {row} position");
        assert_eq!(u32::from(rust.active), c.active, "row {row} active");
        if rust.active {
            assert_eq!(rust.slot, c.slot, "row {row} slot");
            assert_eq!(rust.state_slot, c.state_slot, "row {row} state slot");
            assert_eq!(
                rust.slot + 100,
                rust.state_slot,
                "state slot is not the KV slot"
            );
        }
    }
    // Padding rows resolve as padding on both sides, and a row past capacity is refused, not
    // clamped.
    assert!(!token_batch::resolve_row(&img.batch(), 9).unwrap().active);
    assert_eq!(
        token_batch::resolve_row(&img.batch(), 12).unwrap_err(),
        Refusal::Row
    );
    // SAFETY: as in `agree`.
    assert_eq!(
        unsafe { ffi::token_row(&descriptor, 12) }.unwrap_err().0,
        Refusal::Row.code()
    );

    for s in 0..img.n_sample {
        let rust = token_batch::sample_row(&img.batch(), s).unwrap();
        // SAFETY: as in `agree`.
        let c = unsafe { ffi::token_sample_row(&descriptor, s) }.unwrap();
        assert_eq!(rust, c);
    }
}

#[test]
fn both_resolvers_refuse_the_same_malformed_tables() {
    // Gap.
    let mut img = image(vec![span(0, 2, 0, 0), span(3, 1, 10, 1)], 8, vec![]);
    img.real_rows = 4;
    assert_eq!(agree(&img), Err(Refusal::Cover.code()));

    // Overlap.
    let mut img = image(vec![span(0, 3, 0, 0), span(2, 2, 10, 1)], 8, vec![]);
    img.real_rows = 4;
    assert_eq!(agree(&img), Err(Refusal::Cover.code()));

    // Zero length.
    let img = image(vec![span(0, 0, 0, 0), span(0, 4, 0, 1)], 8, vec![]);
    assert_eq!(agree(&img), Err(Refusal::Cover.code()));

    // Short cover: the spans stop before M.
    let mut img = image(vec![span(0, 2, 0, 0)], 8, vec![]);
    img.real_rows = 4;
    assert_eq!(agree(&img), Err(Refusal::Cover.code()));

    // Position mismatch — the one that matters most: a row whose position disagrees with its
    // span is a row attributed to the wrong request's history.
    let mut img = image(vec![span(0, 4, 30, 0)], 8, vec![]);
    img.positions[2] = 999;
    assert_eq!(agree(&img), Err(Refusal::Position.code()));

    // A live mask bit past M.
    let mut img = image(vec![span(0, 4, 0, 0)], 8, vec![]);
    img.active[5] = 1;
    assert_eq!(agree(&img), Err(Refusal::Active.code()));

    // A dead mask bit inside a span.
    let mut img = image(vec![span(0, 4, 0, 0)], 8, vec![]);
    img.active[2] = 0;
    assert_eq!(agree(&img), Err(Refusal::Active.code()));

    // kv_len that does not match its own span.
    let mut img = image(vec![span(0, 4, 0, 0)], 8, vec![]);
    img.spans[0].kv_len = 9;
    assert_eq!(agree(&img), Err(Refusal::KvLen.code()));

    // A sample index outside the live rows.
    let img = image(vec![span(0, 4, 0, 0)], 8, vec![4]);
    assert_eq!(agree(&img), Err(Refusal::Sample.code()));

    // M beyond the compiled capacity.
    let mut img = image(vec![span(0, 4, 0, 0)], 8, vec![]);
    img.real_rows = 9;
    assert_eq!(agree(&img), Err(Refusal::Capacity.code()));

    // Live rows with no spans, and spans with no live rows: both are the same disagreement
    // between the counts and the table, seen from either side.
    let mut img = image(vec![span(0, 4, 0, 0)], 8, vec![]);
    img.spans.clear();
    assert_eq!(agree(&img), Err(Refusal::Spans.code()));
    let mut img = image(vec![span(0, 4, 0, 0)], 8, vec![]);
    img.real_rows = 0;
    assert_eq!(agree(&img), Err(Refusal::Spans.code()));
}

#[test]
fn a_wrong_descriptor_version_is_refused_rather_than_reinterpreted() {
    let img = image(vec![span(0, 2, 0, 0)], 4, vec![]);
    let mut descriptor = img.descriptor();
    descriptor.version = TOKEN_BATCH_VERSION + 1;
    // SAFETY: the arrays are live; only the version field is wrong.
    assert_eq!(
        unsafe { ffi::token_batch_validate(&descriptor) }
            .unwrap_err()
            .0,
        Refusal::Version.code()
    );
    // Reserved flags and the explicit padding are checked too: a nonzero pad is a descriptor
    // built by something this build does not understand, not a harmless byte. The padding is
    // declared rather than implicit precisely so it can be asserted zero.
    for set in [
        (|d: &mut TokenBatch| d.flags = 1) as fn(&mut TokenBatch),
        |d: &mut TokenBatch| d._pad0 = 1,
        |d: &mut TokenBatch| d._pad1 = 1,
    ] {
        let mut d = img.descriptor();
        set(&mut d);
        // SAFETY: the arrays are live; only a header word is wrong.
        assert_eq!(
            unsafe { ffi::token_batch_validate(&d) }.unwrap_err().0,
            Refusal::Flags.code()
        );
    }
}

/// The planner's own output resolves identically through the C header — which is the claim
/// that matters, because the planner is what every backend will submit.
#[test]
fn a_planned_batch_resolves_identically_through_the_shared_header() {
    let tokens: Vec<u32> = (0..50).collect();
    let one = [9u32];
    let requests = [
        Request {
            id: 10,
            slot: 0,
            state_slot: 4,
            generation: 3,
            phase: Phase::Decode,
            tokens: &one,
            prompt_len: 100,
            selection: Selection::default(),
        },
        Request {
            id: 11,
            slot: 1,
            state_slot: 5,
            generation: 3,
            phase: Phase::Prefill,
            tokens: &tokens,
            prompt_len: 120,
            selection: Selection::default(),
        },
    ];
    let frontiers = [100, 70, 0, 0, 0, 0];
    let generations = [3, 3, 0, 0, 0, 0];
    let plan = token_batch::plan(&requests, &frontiers, &generations, 64, 8192, 0).unwrap();
    assert_eq!((plan.real_rows, plan.sample_rows), (51, 2));

    let descriptor = token_batch::descriptor(
        &plan,
        plan.spans.as_ptr() as u64,
        plan.input_ids.as_ptr() as u64,
        plan.positions.as_ptr() as u64,
        plan.active.as_ptr() as u64,
        plan.sample_input_rows.as_ptr() as u64,
    )
    .unwrap();
    // SAFETY: every pointer borrows `plan`'s live host arrays.
    unsafe { ffi::token_batch_validate(&descriptor) }.unwrap();
    for row in 0..plan.row_capacity {
        let rust = token_batch::resolve_row(&plan.batch(), row).unwrap();
        // SAFETY: as above.
        let c = unsafe { ffi::token_row(&descriptor, row) }.unwrap();
        assert_eq!(rust.position, c.position);
        assert_eq!(u32::from(rust.active), c.active);
        assert_eq!(
            rust.span.unwrap_or(ffi::TokenRowFlat::SPAN_NONE),
            c.span,
            "row {row}"
        );
    }
    // The decode row's state slot is NOT its KV slot, and the resolver reports the span's, not
    // the row's index.
    // SAFETY: as above.
    let decode_row = unsafe { ffi::token_row(&descriptor, 0) }.unwrap();
    assert_eq!((decode_row.slot, decode_row.state_slot), (0, 4));
    // SAFETY: as above.
    let last_prompt_row = unsafe { ffi::token_row(&descriptor, 50) }.unwrap();
    assert_eq!((last_prompt_row.slot, last_prompt_row.position), (1, 119));
}
