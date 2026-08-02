//! §P HostExecutor — runs host-class token packets (sample / tokenize) as part
//! of the counter-gated walk.
//!
//! A host packet (`Body::Token`, `ResourceKind::Host`) is gated by counters like
//! any other: it fires the instant its `wait` counters are satisfied (e.g. the
//! logits producer's counter), does its host work, and its `succ` bump unblocks
//! any consumer. The elegant fit with the packet design: the `HostExecutor` is a
//! [`StepObserver`], so it runs the host op in `on_fire` — which the interpreter
//! calls the moment a packet fires and *before* its successors increment. No new
//! execution path; the same run_streams walk drives host and device packets.
//!
//! Per-request sampling params and the concrete logits/tokens buffers are set on
//! the executor before each iteration (the runtime resolves them from the
//! address map by RequestIo semantic + the indirection table); the handler reads
//! logits and writes the chosen token id.

use packet::{Body, Inst, Opcode};

use crate::device::cpu::StepObserver;
use crate::text::sample::{self, SamplingParams};
use crate::text::tokenizer::{ByteTokenizer, Tokenize};

/// Runs host token ops against the current request's buffers. Holds the logits
/// row(s) and receives the sampled token(s); a real deployment resolves these
/// from the arena by semantic, this models the contract for the reference
/// path/tests.
///
/// **Batched sample.** For `TOKEN_SAMPLE_BATCH` the mux fills `logits` with a
/// `B×vocab` row-major buffer, sets one entry in `slot_params` / `slot_rng01`
/// per row, and the executor writes one produced id into `slot_tokens[b]` for
/// each row. The scalar `params`/`rng01` and `tokens` are kept for the single
/// SAMPLE path so today's tests are unchanged.
#[derive(Default)]
pub struct HostExecutor {
    /// Current logits: for scalar SAMPLE this is one vocab-wide row; for
    /// SAMPLE_BATCH it is a `B×vocab` row-major tile (row `b` at
    /// `[b*vocab .. (b+1)*vocab]`).
    pub logits: Vec<f32>,
    /// Per-request sampling params (from the API request via the indirection
    /// table). Greedy when temperature == 0. Ignored by SAMPLE_BATCH when
    /// `slot_params` is populated.
    pub params: SamplingParams,
    /// Deterministic draw in `[0,1)` for stochastic sampling (tests pass fixed).
    /// Ignored by SAMPLE_BATCH when `slot_rng01` is populated.
    pub rng01: f32,
    /// Tokens produced by scalar SAMPLE, in fire order.
    pub tokens: Vec<u32>,
    /// Per-row sampling params for SAMPLE_BATCH (mux writes B entries per
    /// tick; empty ⇒ fall back to `params`).
    pub slot_params: Vec<SamplingParams>,
    /// Per-row `[0,1)` draws for SAMPLE_BATCH (empty ⇒ fall back to `rng01`).
    pub slot_rng01: Vec<f32>,
    /// Produced token per row for SAMPLE_BATCH. Sized to B by the mux before
    /// each tick; the executor writes each row's sampled id in place.
    pub slot_tokens: Vec<u32>,
    /// Input text a TOKENIZE packet encodes into `token_ids` (when the host owns
    /// tokenization). Empty for the `--net` path (input is already `tokens`).
    pub input_text: Option<String>,
    /// Token ids a TOKENIZE packet produced from `input_text`.
    pub token_ids: Vec<u32>,
    /// Count of host ops actually run (for assertions/telemetry).
    pub ran: usize,
}

impl HostExecutor {
    pub fn new() -> Self {
        HostExecutor::default()
    }

    /// Load the logits row a subsequent SAMPLE packet will consume.
    pub fn set_logits(&mut self, logits: Vec<f32>) {
        self.logits = logits;
    }

    /// Run one host token op. Returns the produced token id for scalar SAMPLE
    /// ops; for SAMPLE_BATCH the produced ids land in `slot_tokens` (return None).
    fn run_token(&mut self, kind: u8, batch: u32, vocab: u32) -> Option<u32> {
        self.ran += 1;
        match kind {
            Opcode::TOKEN_SAMPLE_GREEDY | Opcode::TOKEN_SAMPLE_STOCHASTIC => {
                if self.logits.is_empty() {
                    return None;
                }
                let tok = sample::sample(&self.logits, &self.params, None, self.rng01);
                self.tokens.push(tok);
                Some(tok)
            }
            Opcode::TOKEN_SAMPLE_BATCH => {
                // `TokenBody.arg` carries B; `TokenBody.vocab` carries V. Row
                // `b` is `logits[b*V..(b+1)*V]`. Per-row params/rng fall back
                // to the scalar `params`/`rng01` when the slot vecs are empty
                // (e.g. tests calling into a pre-populated executor).
                let b = batch as usize;
                let v = vocab as usize;
                if v == 0 || self.logits.len() < b * v {
                    return None;
                }
                self.slot_tokens.resize(b, 0);
                for row in 0..b {
                    let logits = &self.logits[row * v..(row + 1) * v];
                    let params = self.slot_params.get(row).unwrap_or(&self.params);
                    let rng01 = self.slot_rng01.get(row).copied().unwrap_or(self.rng01);
                    let tok = sample::sample(logits, params, None, rng01);
                    self.slot_tokens[row] = tok;
                    // Mirror into `tokens` too so a `.last()` fallback in the
                    // serving path still works for slot 0.
                    self.tokens.push(tok);
                }
                None
            }
            // TOKENIZE: host text → ids. Encodes `input_text` into `token_ids`
            // when the host owns tokenization; a no-op when the `--net` input is
            // already the `tokens` buffer.
            Opcode::TOKEN_TOKENIZE => {
                if let Some(text) = &self.input_text {
                    self.token_ids = ByteTokenizer.encode(text);
                }
                None
            }
            // DETOKENIZE and any future host kind: no token draw.
            _ => None,
        }
    }
}

impl StepObserver for HostExecutor {
    /// Host ops are not "math" — they always run regardless of dry/golden mode.
    #[inline]
    fn run_math(&self) -> bool {
        false
    }

    fn on_fire(&mut self, _packet_index: usize, inst: &Inst, _t_start: u64, _t_end: u64) {
        if let Body::Token {
            kind, arg, vocab, ..
        } = inst.body
        {
            // For SAMPLE_BATCH `arg` is the batch width; for scalar SAMPLE the
            // batch is implicit 1 (arg is ignored).
            let batch = if kind == Opcode::TOKEN_SAMPLE_BATCH {
                arg
            } else {
                1
            };
            self.run_token(kind, batch, vocab);
        }
    }
}
