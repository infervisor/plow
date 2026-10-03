//! Turn deadlines for the deadline co-scheduler: when each stage of a voice turn must be done,
//! back-scheduled from `speech_end + budget` with device costs from [`crate::sched::cost`]
//! (design: `plans/voice-turn-scheduling.md` section 2).
//!
//! | work | deadline |
//! |---|---|
//! | ASR final | `speech_end + B - est(LLM first token) - est(TTS first chunk)` |
//! | LLM first token | `speech_end + B - est(TTS first chunk)` |
//! | TTS first chunk | `speech_end + B` |
//! | TTS next window | playback clock: `now + buffered - PLAYBACK_MARGIN` |
//! | LLM decode, voice turn | TTS not started: `last + TBT` (the TTS starts on this text); TTS playing: `last + LOOSE * TBT` |
//! | LLM / TTS LM decode, no turn stage ahead | `last + TBT` |
//! | ASR partial | `arrival + PARTIAL_TARGET` (soft) |
//!
//! A first output's back-scheduled deadline is clamped to `[arrival + T/2, arrival + T]`, `T` its
//! stage target: a turn with budget to spare does not defer its first outputs past their targets
//! (that meets them late instead of minimizing them), one already behind is pulled in at most to
//! half its target.
//!
//! Every deadline is capped at `since + max_wait()`, the starvation bound. Work without a turn
//! keeps today's class mapping ([`Due::from_urgency`]). A `Due`'s cost is the work's own
//! remaining device time, so its slack is `deadline - now - cost`.
//!
//! The est() terms are another model's costs: the model serving each stage registers itself the
//! first time it asks for a deadline of that stage (no model names are configured).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::sched::cost::{self, Op};
use crate::serve::cosched::{max_wait, Band, Due, Urgency};
use crate::serve::turns::{Kind, Stage, TurnKey, TurnTimes};

/// Turn budget when the turn carries none (`PLOW_TURN_BUDGET_MS` normally fills it).
const DEFAULT_BUDGET: Duration = Duration::from_millis(1500);
/// Inter-token target when `PLOW_TBT_SLO_MS` is unset.
const DEFAULT_TBT: Duration = Duration::from_millis(100);
/// A voice turn's decode once its TTS is playing: speech consumes text far slower than decode.
const LOOSE: u32 = 4;
/// A revisable partial transcript's soft target from its arrival.
pub const PARTIAL_TARGET: Duration = Duration::from_millis(1000);
/// A stream window renders this much ahead of the point its playback runs dry.
pub const PLAYBACK_MARGIN: Duration = Duration::from_millis(50);
/// Speech-LM ticks to a stream's first chunk (its `first` tokens over a typical quantum).
const FIRST_CHUNK_TICKS: u32 = 4;

const NONE: usize = usize::MAX;
/// Cost-model ids of the models serving a turn's LLM, TTS speech LM and TTS vocoder.
static LLM: AtomicUsize = AtomicUsize::new(NONE);
static TTS_LM: AtomicUsize = AtomicUsize::new(NONE);
static VOCODER: AtomicUsize = AtomicUsize::new(NONE);

fn claim(role: &AtomicUsize, id: usize) {
    if role.load(Ordering::Relaxed) != id {
        role.store(id, Ordering::Relaxed);
    }
}

fn role_cost(role: &AtomicUsize, op: Op) -> Duration {
    match role.load(Ordering::Relaxed) {
        NONE => Duration::ZERO,
        id => cost::estimate_id(id, op),
    }
}

/// The downstream device costs back-scheduling subtracts, read once per decision.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ests {
    /// A prompt's admission to its first token on the turn's LLM.
    pub llm_first: Duration,
    /// A TTS request to its first audio: speech-LM prefill and first-chunk ticks, then a
    /// one-stream render.
    pub tts_first: Duration,
    /// A one-stream render.
    pub render1: Duration,
}

pub fn ests() -> Ests {
    let render1 = role_cost(&VOCODER, Op::Render { streams: 1 });
    Ests {
        llm_first: role_cost(&LLM, Op::Prefill { rows: 0 }) + role_cost(&LLM, Op::DecodeTick { width: 0 }),
        tts_first: role_cost(&TTS_LM, Op::Prefill { rows: 0 })
            + role_cost(&TTS_LM, Op::DecodeTick { width: 0 }) * FIRST_CHUNK_TICKS
            + render1,
        render1,
    }
}

fn tbt() -> Duration {
    static TBT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *TBT.get_or_init(|| {
        crate::config::RuntimeConfig::get()
            .slo_targets()
            .tbt_ms
            .map_or(DEFAULT_TBT, |ms| Duration::from_secs_f64(ms / 1e3))
    })
}

/// The class a stage had before turns, for work outside any turn; also every stage's band.
fn urgency(stage: Stage) -> Urgency {
    match stage {
        Stage::AsrFinal => Urgency::Final,
        Stage::LlmFirst | Stage::TtsFirst => Urgency::Deadline,
        Stage::LlmDecode | Stage::TtsStream => Urgency::Normal,
    }
}

fn turn_end(t: &TurnTimes, since: Instant) -> Instant {
    let budget = if t.budget.is_zero() { DEFAULT_BUDGET } else { t.budget };
    t.speech_end.unwrap_or(since) + budget
}

fn before(at: Instant, d: Duration) -> Instant {
    at.checked_sub(d).unwrap_or(at)
}

/// When `stage` of the turn must be done. `since` is when the work became pending (arrival for a
/// first output, the last token for decode). A first output is due by its own stage target from
/// arrival, and never before half of it: a client that sends TTS only after the whole reply misses
/// the end-to-end budget at any load, and back-scheduling alone would hand those turns unbounded
/// negative slack.
fn stage_deadline(stage: Stage, t: &TurnTimes, e: &Ests, since: Instant) -> Instant {
    let end = turn_end(t, since);
    let own = |at: Instant, kind: Kind| at.clamp(since + kind.target() / 2, since + kind.target());
    match stage {
        Stage::AsrFinal => own(before(end, e.llm_first + e.tts_first), Kind::Asr),
        Stage::LlmFirst => own(before(end, e.tts_first), Kind::Llm),
        Stage::TtsFirst => own(end, Kind::Tts),
        Stage::LlmDecode if t.tts_first_audio.is_some() => since + tbt() * LOOSE,
        Stage::LlmDecode | Stage::TtsStream => since + tbt(),
    }
}

/// Register `model` as the turns' model for `stage` (its costs feed the upstream deadlines).
pub fn claim_stage(stage: Stage, model: usize) {
    match stage {
        Stage::LlmFirst | Stage::LlmDecode => claim(&LLM, model),
        Stage::TtsFirst | Stage::TtsStream => claim(&TTS_LM, model),
        Stage::AsrFinal => {}
    }
}

/// [`due`] for a caller holding the turn's times (cached per job) and the estimates; `cost` is
/// the work's own remaining device time.
pub fn stage_due(stage: Stage, t: &TurnTimes, e: &Ests, cost: Duration, since: Instant) -> Due {
    Due { deadline: stage_deadline(stage, t, e, since).min(since + max_wait()), cost, band: urgency(stage).into() }
}

/// [`stage_due`] for `model`, registering it for the stage.
pub fn turn_due(stage: Stage, t: &TurnTimes, model: usize, cost: Duration, since: Instant) -> Due {
    claim_stage(stage, model);
    stage_due(stage, t, &ests(), cost, since)
}

/// Remaining device time of `stage` work: `prefill` still to run, then its ticks of `tick` each.
pub fn stage_cost(stage: Stage, prefill: Duration, tick: Duration, e: &Ests) -> Duration {
    match stage {
        Stage::AsrFinal | Stage::LlmFirst => prefill + tick,
        Stage::TtsFirst => prefill + tick * FIRST_CHUNK_TICKS + e.render1,
        Stage::LlmDecode | Stage::TtsStream => tick,
    }
}

/// When `stage` of the turn `key` is due on the device for `model`. Without a turn (or once the
/// turn has expired) it is today's class deadline from `since`.
pub fn due(stage: Stage, key: Option<&TurnKey>, model: &str, since: Instant, _now: Instant) -> Due {
    match key.and_then(|k| crate::serve::turns::table().times(k)) {
        Some(t) => {
            let id = cost::id(model);
            claim_stage(stage, id);
            let e = ests();
            let tick = cost::estimate_id(id, Op::DecodeTick { width: 0 });
            let cost = stage_cost(stage, cost::estimate_id(id, Op::Prefill { rows: 0 }), tick, &e);
            stage_due(stage, &t, &e, cost, since)
        }
        None => Due::from_urgency(urgency(stage), since),
    }
}

/// A partial transcript pending since `since`: soft, the first work to give up under load.
pub fn partial(since: Instant, cost: Duration) -> Due {
    Due { deadline: since + PARTIAL_TARGET.min(max_wait()), cost, band: Band::Bulk }
}

/// A render of `streams` on the vocoder `model`, registering it as the turns' vocoder.
pub fn render_cost(model: usize, streams: usize) -> Duration {
    claim(&VOCODER, model);
    cost::estimate_id(model, Op::Render { streams })
}

/// A render of a turn's first audio (or whole reply), pending since `since`.
pub fn first_audio(t: &TurnTimes, since: Instant, cost: Duration) -> Due {
    stage_due(Stage::TtsFirst, t, &ests(), cost, since)
}

/// A started stream's next window: due before its playback runs dry (`buffered` from now).
pub fn playback(now: Instant, buffered: Duration, cost: Duration) -> Due {
    Due { deadline: now + buffered.saturating_sub(PLAYBACK_MARGIN), cost, band: Band::Stream }
}

/// Overload level 2+: work this far ahead of its deadline (over two TBT targets of slack, i.e. a
/// voice turn's decode while its TTS plays) runs one decode step per device turn.
pub fn ahead_under_overload(due: Due, now: Instant) -> bool {
    crate::serve::overload::level() >= 2 && due.slack(now) > (tbt() * 2).as_nanos() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn times(speech_end: Instant, budget_ms: u64) -> TurnTimes {
        TurnTimes { speech_end: Some(speech_end), budget: Duration::from_millis(budget_ms), ..Default::default() }
    }

    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    #[test]
    fn first_outputs_are_due_between_half_and_all_of_their_target() {
        let t0 = Instant::now();
        let late = t0 + Duration::from_secs(3);
        let t = times(t0, 1500);
        let long = times(t0, 60_000);
        let e = Ests::default();
        let zero = Duration::ZERO;
        for (stage, kind) in [(Stage::AsrFinal, Kind::Asr), (Stage::LlmFirst, Kind::Llm), (Stage::TtsFirst, Kind::Tts)] {
            assert_eq!(stage_due(stage, &t, &e, zero, late).deadline, late + kind.target() / 2, "{stage:?} past the budget");
            assert_eq!(stage_due(stage, &long, &e, zero, t0).deadline, t0 + kind.target(), "{stage:?} budget to spare");
        }
    }

    /// Roles are process-wide: only this test claims them, so the arithmetic is exact.
    #[test]
    fn back_scheduling_subtracts_downstream_costs() {
        let (llm, lm, voc) = (cost::id("dl-test-llm"), cost::id("dl-test-tts-lm"), cost::id("dl-test-vocoder"));
        cost::record_id(llm, Op::Prefill { rows: 512 }, Duration::from_millis(30));
        cost::record_id(llm, Op::DecodeTick { width: 8 }, Duration::from_millis(10));
        cost::record_id(lm, Op::Prefill { rows: 300 }, Duration::from_millis(20));
        cost::record_id(lm, Op::DecodeTick { width: 16 }, Duration::from_millis(15));
        cost::record_id(voc, Op::Render { streams: 1 }, Duration::from_millis(80));
        let t0 = Instant::now();
        let t = times(t0, 1500);
        let zero = Duration::ZERO;
        render_cost(voc, 1);
        // Arriving 900 ms into the turn, every stage's back-scheduled deadline lies within its
        // [T/2, T] clamp.
        let at = t0 + Duration::from_millis(900);
        let tts = turn_due(Stage::TtsFirst, &t, lm, zero, at);
        let llm_first = turn_due(Stage::LlmFirst, &t, llm, zero, at);
        let asr = turn_due(Stage::AsrFinal, &t, cost::id("dl-test-asr"), zero, at);
        let end = t0 + Duration::from_millis(1500);
        assert_eq!(tts.deadline, end);
        // est(TTS first) = 20 + 4 * 15 + 80 = 160 ms; est(LLM first) = 30 + 10 = 40 ms.
        assert!((ms(end - llm_first.deadline) - 160.0).abs() < 1e-3);
        assert!((ms(end - asr.deadline) - 200.0).abs() < 1e-3, "{:?}", end - asr.deadline);
        assert!(asr.deadline <= llm_first.deadline && llm_first.deadline <= tts.deadline);

        // A turn with no budget gets the default; a speech end in the past is honoured.
        let early = before(t0, Duration::from_millis(400));
        let d = turn_due(Stage::TtsFirst, &times(early, 0), lm, zero, t0 + Duration::from_millis(500));
        assert_eq!(d.deadline, early + DEFAULT_BUDGET);

        // Decode: TBT pace until the TTS plays, loose after.
        let mut t = t;
        let dec = turn_due(Stage::LlmDecode, &t, llm, zero, t0);
        assert_eq!(dec.deadline, t0 + tbt());
        t.tts_first_audio = Some(t0);
        let dec = turn_due(Stage::LlmDecode, &t, llm, zero, t0);
        assert_eq!(dec.deadline, t0 + tbt() * LOOSE);

        // The starvation bound caps every turn deadline, however long the budget.
        let long = times(t0, 60_000);
        for (stage, id) in [(Stage::AsrFinal, llm), (Stage::LlmFirst, llm), (Stage::LlmDecode, llm), (Stage::TtsFirst, lm), (Stage::TtsStream, lm)] {
            assert!(turn_due(stage, &long, id, zero, t0).deadline <= t0 + max_wait(), "{stage:?}");
        }
        assert!(partial(t0, zero).deadline <= t0 + max_wait());
    }

    #[test]
    fn playback_clock_and_slack() {
        let now = Instant::now();
        let d = playback(now, Duration::from_millis(500), Duration::from_millis(100));
        assert_eq!(d.deadline, now + Duration::from_millis(450));
        assert_eq!(d.slack(now), Duration::from_millis(350).as_nanos() as i64);
        // A drained buffer is due now, and its cost puts it behind already.
        let dry = playback(now, Duration::ZERO, Duration::from_millis(100));
        assert_eq!(dry.deadline, now);
        assert!(dry.slack(now) < 0);
    }

    #[test]
    fn no_turn_falls_back_to_class_order() {
        let t0 = Instant::now();
        let at = |stage| due(stage, None, "dl-test-noturn", t0, t0);
        for stage in [Stage::AsrFinal, Stage::LlmFirst, Stage::LlmDecode, Stage::TtsFirst, Stage::TtsStream] {
            assert_eq!(at(stage).deadline, Due::from_urgency(urgency(stage), t0).deadline);
        }
        assert!(at(Stage::AsrFinal).slack(t0) < at(Stage::LlmFirst).slack(t0));
        assert!(at(Stage::LlmFirst).slack(t0) < at(Stage::LlmDecode).slack(t0));
        // An unknown turn key is no turn.
        let key = TurnKey { session: "dl-test-none".into(), turn: "1".into() };
        assert_eq!(due(Stage::LlmFirst, Some(&key), "dl-test-noturn", t0, t0).deadline, at(Stage::LlmFirst).deadline);
    }
}
