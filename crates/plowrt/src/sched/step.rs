//! Backend-neutral per-tick step planner — the shape of vLLM's `Scheduler.schedule()`.
//!
//! One token budget per step. Decodes are admitted first (one row each); then the requests
//! already mid-prefill get spans — several requests sharing one launch when the backend can
//! pack them, else one request's planned chunk at a time, oldest first — and finally fresh
//! requests are admitted while the budget is untouched. Every backend runs this same function
//! and only LOWERS the plan to its own step: the AMD/CPU tick runs each [`Launch`] as one
//! packed or isolated prefill and its decodes as a separate batched dispatch; the CUDA tick
//! runs the single launch as its batched (optionally unified, decode-carrying) pass. Backends
//! describe themselves through [`Backend`]; nothing here names a device.
//!
//! What the planner does NOT do is re-plan a request narrower to fit a remainder: under
//! [`Backend::split_spans`] = false a chunk wider than what is left waits for the next tick,
//! which is what keeps a request on the compiled rungs its cursor was planned against.

use super::prefill::{admit, SpanPolicy};
use packet::dev::PrefillSpan;

/// What a backend declares about the steps it can run. Static per engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backend {
    /// Prefill rows one tick may carry before decode runs when the runtime sets no cap:
    /// the widest compiled prefill rung. `u32::MAX` = no ladder to budget against (one launch
    /// per tick, as a whole-prompt engine behaves).
    pub step_budget: u32,
    /// Several requests' spans may share one launch (the runtime's `pf_batch` still gates it).
    pub packing: bool,
    /// A request's offered span may be cut to fit the launch; otherwise it runs whole or waits.
    pub split_spans: bool,
    /// Decode rows ride in the prefill launch and are charged to the same budget.
    pub decode_rows_join_prefill: bool,
}

impl Default for Backend {
    fn default() -> Self {
        Self {
            step_budget: u32::MAX,
            packing: false,
            split_spans: false,
            decode_rows_join_prefill: false,
        }
    }
}

/// The runtime's per-tick inputs.
#[derive(Clone, Copy, Debug)]
pub struct Tick {
    /// The tick's prefill row cap before the backend's budget (`u32::MAX` = uncapped).
    pub cap_rows: u32,
    /// Whether cross-request packing is requested this tick.
    pub packing: bool,
    /// Rotate admission by slot from `turn` instead of oldest-first.
    pub rotate: bool,
    pub turn: usize,
    /// Slot table capacity.
    pub slots: usize,
}

/// One request that wants prefill rows this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The rows it offers next: `slot`, `kv_row0`, `n_rows`, and the compiled `program` a pack
    /// of it would run (`0` when the backend has no packed program for it).
    pub span: PrefillSpan,
    /// Arrival order key; smaller is older.
    pub arrival: u64,
    /// May share a launch with other candidates (not a final chunk, not a snapshot boundary,
    /// a capable packed program exists).
    pub packable: bool,
    /// Its cursor is planned (`n_rows` is a real chunk). A fresh request's first chunk is
    /// unknown until admission plans it, so it is admitted only into an untouched budget.
    pub planned: bool,
}

/// One prefill launch: two or more spans is a pack, one is an isolated chunk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Launch {
    pub spans: Vec<PrefillSpan>,
}

impl Launch {
    pub fn rows(&self) -> u32 {
        self.spans.iter().map(|span| span.n_rows).sum()
    }
    pub fn is_pack(&self) -> bool {
        self.spans.len() >= 2
    }
}

/// The plan for one tick.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Decode slots, in the order given — every live decode row is admitted (one row each).
    pub decodes: Vec<u32>,
    /// Prefill launches in execution order.
    pub launches: Vec<Launch>,
    /// Rows of the prefill budget left unspent.
    pub budget_left: u32,
}

impl Plan {
    pub fn prefill_rows(&self) -> u32 {
        self.launches.iter().map(Launch::rows).sum()
    }
    pub fn slots_advanced(&self) -> impl Iterator<Item = u32> + '_ {
        self.launches
            .iter()
            .flat_map(|launch| launch.spans.iter().map(|span| span.slot))
    }
}

/// Plan one tick.
///
/// `program_rows(program)` is the compiled row count of a packed program and
/// `program_span_limit(program)` its D-class span limit, both backend-supplied and consulted
/// only for packs. Every slot advances at most once per tick.
pub fn plan(
    backend: Backend,
    tick: Tick,
    decodes: impl IntoIterator<Item = u32>,
    candidates: &[Candidate],
    program_rows: impl Fn(u32) -> Option<u32>,
    program_span_limit: impl Fn(u32) -> u32,
) -> Plan {
    let decodes: Vec<u32> = decodes.into_iter().collect();
    let mut budget = tick.cap_rows.min(backend.step_budget);
    if backend.decode_rows_join_prefill {
        budget = budget.saturating_sub(u32::try_from(decodes.len()).unwrap_or(u32::MAX));
    }
    let full_budget = budget;
    // No ladder to budget against: one launch per tick, as before the budget existed.
    let max_launches = if backend.step_budget == u32::MAX { 1 } else { usize::MAX };
    let slots = tick.slots.min(u128::BITS as usize);
    let mut advanced = 0u128;
    let taken = |advanced: u128, span: &PrefillSpan| {
        (span.slot as usize) >= slots || advanced & (1u128 << span.slot) != 0
    };
    let mut launches: Vec<Launch> = Vec::new();
    while budget > 0 && launches.len() < max_launches {
        if backend.packing && tick.packing {
            let policy = if backend.split_spans {
                SpanPolicy::FairSplit
            } else {
                SpanPolicy::Whole
            };
            let mut pack = admit(
                candidates
                    .iter()
                    .filter(|c| c.packable && !taken(advanced, &c.span))
                    .map(|c| c.span),
                budget,
                tick.turn,
                slots,
                policy,
                &program_rows,
            );
            if let Some(program) = pack.spans().first().map(|span| span.program) {
                pack.limit_spans(program_span_limit(program) as usize);
            }
            // A pack of one is only a launch when the backend runs every launch as a pack
            // (split spans): a whole-span backend serves a lone request isolated instead.
            let min_spans = if backend.split_spans { 1 } else { 2 };
            if pack.spans().len() >= min_spans {
                let launch = Launch {
                    spans: pack.spans().to_vec(),
                };
                for span in &launch.spans {
                    advanced |= 1u128 << span.slot;
                }
                budget = budget.saturating_sub(launch.rows().max(1));
                launches.push(launch);
                continue;
            }
        }
        // Isolated: the oldest (or, rotating, the next slot from `turn`) whose chunk fits.
        let fits = |c: &Candidate| {
            if backend.split_spans {
                true
            } else if c.planned {
                c.span.n_rows <= budget
            } else {
                budget == full_budget
            }
        };
        let pick = candidates
            .iter()
            .filter(|c| c.span.n_rows > 0 && !taken(advanced, &c.span) && fits(c))
            .min_by_key(|c| {
                if tick.rotate {
                    let slot = c.span.slot as usize;
                    ((slot + slots - tick.turn % slots.max(1)) % slots.max(1), 0, slot)
                } else {
                    (0, c.arrival as usize, c.span.slot as usize)
                }
            });
        let Some(c) = pick else { break };
        let mut span = c.span;
        span.row0 = 0;
        span.n_rows = span.n_rows.min(budget);
        span.kv_len = span.kv_row0 + span.n_rows;
        advanced |= 1u128 << span.slot;
        budget = budget.saturating_sub(span.n_rows.max(1));
        launches.push(Launch { spans: vec![span] });
    }
    Plan {
        decodes,
        launches,
        budget_left: budget,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(slot: u32, arrival: u64, kv_row0: u32, rows: u32, packable: bool, planned: bool) -> Candidate {
        Candidate {
            span: PrefillSpan {
                row0: 0,
                n_rows: rows,
                slot,
                flags: 0,
                kv_row0,
                kv_len: kv_row0 + rows,
                state_slot: slot,
                program: 3,
            },
            arrival,
            packable,
            planned,
        }
    }

    fn amd() -> Backend {
        Backend {
            step_budget: 8192,
            packing: true,
            split_spans: false,
            decode_rows_join_prefill: false,
        }
    }

    fn tick(packing: bool) -> Tick {
        Tick {
            cap_rows: u32::MAX,
            packing,
            rotate: false,
            turn: 0,
            slots: 20,
        }
    }

    fn rung_2048(program: u32) -> Option<u32> {
        (program == 3).then_some(2048)
    }

    #[test]
    fn decodes_are_admitted_first_and_never_touch_a_separate_prefill_budget() {
        for d in 0..=20u32 {
            let decodes: Vec<u32> = (0..d).collect();
            let got = plan(amd(), tick(true), decodes.clone(), &[cand(19, 0, 0, 8192, false, true)], rung_2048, |_| u32::MAX);
            assert_eq!(got.decodes, decodes);
            assert_eq!(got.prefill_rows(), 8192, "d={d}");
            assert_eq!(got.budget_left, 0);
        }
    }

    #[test]
    fn joined_decode_rows_are_charged_to_the_budget() {
        let backend = Backend {
            step_budget: 2048,
            packing: true,
            split_spans: true,
            decode_rows_join_prefill: true,
        };
        let got = plan(backend, tick(true), 0..20, &[cand(0, 0, 0, 100_000, true, true)], |_| Some(2048), |_| u32::MAX);
        assert_eq!(got.decodes.len(), 20);
        assert_eq!(got.prefill_rows(), 2048 - 20);
        assert_eq!(got.launches.len(), 1);
    }

    /// The default shape on the GLM-5.3 TP8 packet: 8192-row planned chunks (unpackable — the
    /// rung is a sparse bucket), budget = widest rung. Exactly one request advances per tick,
    /// the oldest, and nobody is re-planned narrower.
    #[test]
    fn one_widest_chunk_per_tick_oldest_first() {
        let candidates: Vec<Candidate> = (0..20).map(|s| cand(s, 100 - s as u64, 8192, 8192, false, true)).collect();
        let got = plan(amd(), tick(true), 0..20, &candidates, rung_2048, |_| u32::MAX);
        assert_eq!(got.launches.len(), 1);
        assert_eq!(got.launches[0].spans[0].slot, 19, "arrival 81 is the oldest");
        assert_eq!(got.launches[0].rows(), 8192);
        assert_eq!(got.budget_left, 0);
    }

    /// `PLOW_PF_CHUNK=2048` shape: planned 2048-row chunks on an 8192 budget. Packing fills the
    /// 2048 packed rung first with whole spans, then the remaining 6144 rows go to isolated
    /// chunks oldest-first; a slot is never launched twice in one tick.
    #[test]
    fn budget_is_filled_by_a_pack_then_isolated_chunks_without_repeating_a_slot() {
        let candidates = vec![
            cand(0, 5, 2048, 1024, true, true),
            cand(1, 6, 2048, 1024, true, true),
            cand(2, 1, 4096, 2048, false, true),
            cand(3, 2, 4096, 2048, false, true),
            cand(4, 3, 4096, 2048, false, true),
            cand(5, 4, 4096, 2048, false, true),
        ];
        let got = plan(amd(), tick(true), 0..6, &candidates, rung_2048, |_| u32::MAX);
        assert!(got.launches[0].is_pack());
        assert_eq!(got.launches[0].spans.iter().map(|s| s.slot).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(got.launches[0].spans.iter().map(|s| s.row0).collect::<Vec<_>>(), [0, 1024]);
        let isolated: Vec<u32> = got.launches[1..].iter().map(|l| l.spans[0].slot).collect();
        assert_eq!(isolated, [2, 3, 4], "oldest first, three 2048 chunks fill the 6144 left");
        assert_eq!(got.prefill_rows(), 8192);
        let mut seen: Vec<u32> = got.slots_advanced().collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn a_chunk_wider_than_the_remainder_waits_instead_of_being_split() {
        let candidates = vec![
            cand(0, 1, 0, 2048, true, true),
            cand(1, 2, 0, 2048, true, true),
            cand(2, 0, 8192, 8192, false, true),
        ];
        // The pack (2048 rung: one 2048 span fits, so no pack) → the oldest isolated: slot 2's
        // 8192 fills the budget; slots 0 and 1 wait.
        let got = plan(amd(), tick(true), [], &candidates, rung_2048, |_| u32::MAX);
        assert_eq!(got.launches.len(), 1);
        assert_eq!(got.launches[0].spans[0].slot, 2);
        // With a 4096 cap the 8192 chunk does not fit at all and is NOT cut: the 2048s run.
        let capped = Tick { cap_rows: 4096, ..tick(true) };
        let got = plan(amd(), capped, [], &candidates, rung_2048, |_| u32::MAX);
        assert_eq!(got.launches.iter().map(|l| l.spans[0].slot).collect::<Vec<_>>(), [0, 1]);
        assert!(got.launches.iter().all(|l| l.rows() == 2048));
    }

    #[test]
    fn fresh_requests_are_admitted_only_into_an_untouched_budget() {
        let fresh = cand(7, 0, 0, u32::MAX, false, false);
        let planned = cand(3, 1, 2048, 2048, false, true);
        let got = plan(amd(), tick(false), [], &[fresh, planned], rung_2048, |_| u32::MAX);
        // The fresh request is the oldest and the budget is untouched: it goes first, charged
        // the whole budget (its plan is built against the full tick cap by the backend).
        assert_eq!(got.launches.len(), 1);
        assert_eq!(got.launches[0].spans[0].slot, 7);
        assert_eq!(got.launches[0].rows(), 8192);
        let got = plan(amd(), tick(false), [], &[cand(7, 2, 0, u32::MAX, false, false), planned], rung_2048, |_| u32::MAX);
        assert_eq!(got.launches.iter().map(|l| l.spans[0].slot).collect::<Vec<_>>(), [3], "a partial budget admits no fresh request");
    }

    #[test]
    fn rotation_starts_at_the_turn_slot_and_oldest_first_ignores_slot_order() {
        let candidates: Vec<Candidate> = (0..4).map(|s| cand(s, 10 - s as u64, 0, 8192, false, true)).collect();
        let by_arrival = plan(amd(), tick(false), [], &candidates, rung_2048, |_| u32::MAX);
        assert_eq!(by_arrival.launches[0].spans[0].slot, 3);
        let rotating = Tick { rotate: true, turn: 2, slots: 4, ..tick(false) };
        let by_turn = plan(amd(), rotating, [], &candidates, rung_2048, |_| u32::MAX);
        assert_eq!(by_turn.launches[0].spans[0].slot, 2);
    }

    #[test]
    fn packing_off_or_unsupported_serves_everything_isolated() {
        let candidates = vec![cand(0, 0, 0, 512, true, true), cand(1, 1, 0, 512, true, true)];
        let got = plan(amd(), tick(false), [], &candidates, rung_2048, |_| u32::MAX);
        assert!(got.launches.iter().all(|l| !l.is_pack()));
        assert_eq!(got.launches.len(), 2);
        let no_packing = Backend { packing: false, ..amd() };
        let got = plan(no_packing, tick(true), [], &candidates, rung_2048, |_| u32::MAX);
        assert!(got.launches.iter().all(|l| !l.is_pack()));
    }

    #[test]
    fn the_d_class_span_limit_shrinks_a_pack_before_anything_moves() {
        let candidates: Vec<Candidate> = (0..4).map(|s| cand(s, s as u64, 0, 256, true, true)).collect();
        let got = plan(amd(), tick(true), [], &candidates, rung_2048, |_| 2);
        assert_eq!(got.launches[0].spans.len(), 2);
        // The two dropped spans are not lost: they form the next pack within the same budget.
        assert_eq!(got.launches.len(), 2);
        assert!(got.launches.iter().all(|l| l.spans.len() == 2));
        assert_eq!(got.slots_advanced().count(), 4);
        let limited_to_one = plan(amd(), tick(true), [], &candidates, rung_2048, |_| 1);
        assert!(limited_to_one.launches.iter().all(|l| !l.is_pack()));
    }

    /// The CUDA shape: one fair-split pass per tick, a lone request still forms the launch,
    /// decode rows join it and are charged.
    #[test]
    fn split_span_backends_run_one_fair_split_launch_per_tick() {
        let backend = Backend {
            step_budget: 4096,
            packing: true,
            split_spans: true,
            decode_rows_join_prefill: true,
        };
        let candidates = vec![cand(0, 0, 0, 100_000, true, true), cand(1, 1, 0, 100, true, true)];
        let got = plan(backend, tick(true), 0..4, &candidates, |_| Some(4096), |_| u32::MAX);
        assert_eq!(got.launches.len(), 1);
        assert_eq!(got.launches[0].rows(), 4096 - 4);
        assert_eq!(got.launches[0].spans.iter().map(|s| (s.slot, s.n_rows)).collect::<Vec<_>>(), [(0, 4096 - 4 - 100), (1, 100)]);
        let lone = plan(backend, tick(true), [], &candidates[..1], |_| Some(4096), |_| u32::MAX);
        assert_eq!(lone.launches.len(), 1);
        assert_eq!(lone.launches[0].spans.len(), 1);
    }

    #[test]
    fn a_whole_prompt_backend_runs_one_launch_per_tick() {
        let candidates: Vec<Candidate> = (0..3).map(|s| cand(s, s as u64, 0, 300, false, false)).collect();
        let got = plan(Backend::default(), tick(true), 0..2, &candidates, |_| None, |_| u32::MAX);
        assert_eq!(got.launches.len(), 1);
        assert_eq!(got.launches[0].spans[0].slot, 0);
    }

    #[test]
    fn budget_accounting_holds_over_ragged_mixes() {
        for seed in 0..200u64 {
            let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut next = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let d = (next() % 21) as u32;
            let n = (next() % 8) as u32;
            let candidates: Vec<Candidate> = (0..n)
                .map(|i| {
                    let rows = [128, 512, 1024, 2048, 8192][(next() % 5) as usize];
                    let planned = next() % 4 != 0;
                    cand(d + i, next() % 50, 0, if planned { rows } else { u32::MAX }, next() % 2 == 0, planned)
                })
                .collect();
            let cap = [u32::MAX, 8192, 4096, 2048][(next() % 4) as usize];
            let t = Tick { cap_rows: cap, packing: true, rotate: next() % 2 == 0, turn: (next() % 20) as usize, slots: 32 };
            let got = plan(amd(), t, 0..d, &candidates, rung_2048, |_| u32::MAX);
            let budget = cap.min(8192);
            assert!(got.prefill_rows() <= budget, "seed={seed}");
            assert_eq!(got.prefill_rows() + got.budget_left, budget, "seed={seed}");
            let mut slots: Vec<u32> = got.slots_advanced().collect();
            let count = slots.len();
            slots.sort_unstable();
            slots.dedup();
            assert_eq!(slots.len(), count, "seed={seed}: a slot advanced twice");
            for launch in &got.launches {
                let mut row = 0;
                for span in &launch.spans {
                    assert_eq!(span.row0, row);
                    assert_eq!(span.kv_len, span.kv_row0 + span.n_rows);
                    row += span.n_rows;
                    let c = candidates.iter().find(|c| c.span.slot == span.slot).unwrap();
                    assert!(span.n_rows <= c.span.n_rows);
                    if !launch.is_pack() && c.planned {
                        assert_eq!(span.n_rows, c.span.n_rows, "seed={seed}: a whole-span backend never cuts a planned chunk");
                    }
                }
            }
        }
    }
}
