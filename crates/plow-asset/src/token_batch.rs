//! Unified token batches: the backend-neutral host plan, its device descriptor, the row
//! resolver's Rust twin, and the load-time capability check.
//!
//! One packed activation matrix per step containing all scheduled input tokens — decode and
//! prefill, from any number of requests — and one compact terminal segment producing exactly
//! the next-token distributions the step owes. See `docs/arch/17-unified-token-batch.md`.
//!
//! This module is deliberately backend-neutral, for the same reason
//! [`crate::mixed_step`] is: `PlowProgram` is one kernarg struct for HIP, CUDA and CPU, and
//! `runtime/common/token_batch.h` is one row resolver for all three. A second planner is how
//! two device layouts drift apart while both look tested.
//!
//! # The three counts
//!
//! * `row_capacity` — compiled/padded allocation capacity.
//! * `real_rows = M` — scheduled input tokens this step.
//! * `sample_rows = S` — hidden rows whose next-token distributions are needed.
//!
//! `S = decode_requests + prompts_completed_this_step`. An intermediate prompt chunk
//! contributes tokens to `M` and zero to `S`. **`S` may be zero**, and zero means "no output
//! segment" — not the legacy `n_batch == 0 means one row`, which reads zero as one and would
//! sample a row nobody asked for.
//!
//! # What is deliberately NOT here
//!
//! No decode prefix. [`crate::mixed_step`]'s plan splits rows into a decode band and a prefill
//! band, and its device resolver has a matching special case for rows below the first span.
//! Here every row belongs to a span, decode spans included, so there is one row source and one
//! search. A decode request is a span of length one — and note the converse is NOT true: a
//! FINAL PREFILL CHUNK can also have length one, so length is never what decides a span's
//! phase.

use packet::dev::{DevOp, PrefillSpan, TokenBatch, PREFILL_SPAN_RESET_STATE, TOKEN_BATCH_VERSION};
use packet::rowclass::{class_of_op, RowClass};

use crate::aux_program;

type Result<T> = std::result::Result<T, String>;

pub const SECTION: &str = "token_batch";
pub const VERSION: u32 = 1;

/// Object capability proving the code object was built against the `PlowProgram` that carries
/// [`packet::dev::DevProgram::token_batch`]. It proves the ABI and NOTHING about which
/// descriptor-aware math arms the object has — those are separate markers, because an object
/// that parses the descriptor and then runs a packet-scalar kernel is exactly the silent
/// wrong answer this contract exists to prevent.
pub const OBJECT_CAPABILITY: &str = "plow_token_batch_abi";

/// Object capability proving [`DevOp::RowGather`] has a real dispatch arm. On AMD the
/// interpreter's `default:` neither writes nor traps, so a missing arm leaves the terminal
/// segment's gathered rows at whatever the buffer held.
pub const ROW_GATHER_CAPABILITY: &str = "plow_row_gather";

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("token batch: {reason}"))
    }
}

// ---------------------------------------------------------------------------------------------
// Host plan
// ---------------------------------------------------------------------------------------------

/// Which serving phase produced a span. Kernel selection derives query geometry from span
/// LENGTHS; this tag exists for the consumers that need an attention partition, and class-A
/// operators never read it.
///
/// A span is never classified as decode because its length is one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Decode,
    Prefill,
}

/// Per-row selection policy, attached to the LOGICAL REQUEST and therefore independent of row
/// order and padding.
///
/// Sampling policy is per backend and this contract neither widens nor narrows it: AMD's device
/// argmax path implements greedy only, and NVIDIA's `sample_sm120.cu` already implements
/// on-device temperature/top-k/top-p/min-p specifically so `temperature > 0` does not download
/// the whole vocabulary. Compacting the logits must feed that sampler, not replace it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    pub greedy: bool,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    /// RNG state carried with the request, so a fixed seed reproduces across row orders.
    pub rng: u64,
}

impl Default for Selection {
    fn default() -> Self {
        Selection {
            greedy: true,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            rng: 0,
        }
    }
}

impl Selection {
    fn validate(&self) -> Result<()> {
        if self.greedy {
            return require(
                self.temperature == 0.0,
                "greedy selection with a nonzero temperature",
            );
        }
        require(
            self.temperature > 0.0
                && self.temperature.is_finite()
                && (0.0..=1.0).contains(&self.top_p)
                && (0.0..=1.0).contains(&self.min_p),
            "stochastic selection bounds",
        )
    }
}

/// One request's contribution to a step. Decode and prefill are the same shape on purpose:
/// the difference is the phase tag and how many tokens are scheduled, not the representation.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Logical request id, carried through to [`SampleOwner`] so delivery does not depend on
    /// row order.
    pub id: u32,
    /// Physical KV slot.
    pub slot: u32,
    /// Carried-state slot. Explicit because KV and recurrent-state layouts may diverge, and
    /// because the D-class families index their state by it.
    pub state_slot: u32,
    /// Slot generation at admission. Checked against the host's table before the plan is built
    /// AND again before the commit, so a recycled slot cannot receive a prior step's output.
    pub generation: u32,
    pub phase: Phase,
    /// The tokens scheduled this step, in order.
    pub tokens: &'a [u32],
    /// Total prompt length. A prefill span whose end reaches this completes the prompt and
    /// therefore contributes a sample row; a decode request must pass its own frontier + 1.
    pub prompt_len: u32,
    pub selection: Selection,
}

/// A hidden row the step owes a next-token distribution for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleOwner {
    pub request: u32,
    pub slot: u32,
    pub generation: u32,
    pub selection: Selection,
}

/// What the host will publish if — and only if — the whole chain succeeds.
///
/// Frontiers are NOT advanced for sampled tokens: a generated token advances the frontier when
/// it is processed as an INPUT, not when it is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingCommit {
    pub request: u32,
    pub slot: u32,
    pub generation: u32,
    /// The frontier this plan was built against. Re-checked before any frontier is mutated.
    pub expected_frontier: u32,
    /// The frontier after this step's rows are committed.
    pub new_frontier: u32,
    /// This span finished the prompt, so the request becomes ready for decode.
    pub completes_prompt: bool,
    /// Index into [`Plan::sample_input_rows`] / [`Plan::sample_owners`], or `None` for an
    /// intermediate chunk that produces no output.
    pub sample_index: Option<u32>,
}

/// Backend-neutral host plan for one unified token batch.
///
/// `input_ids`, `positions` and `active` are `row_capacity` long; the spans cover exactly
/// `[0, real_rows)`. Padding rows carry `active = 0`, belong to no span, and read zero for
/// both id and position — they are not another request's rows extended past its frontier.
#[derive(Debug, PartialEq, Default)]
pub struct Plan {
    pub row_capacity: u32,
    pub real_rows: u32,
    pub sample_rows: u32,
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    /// Park mask in the ISA's `active[B]` sense: `1` = live row.
    pub active: Vec<u32>,
    pub spans: Vec<PrefillSpan>,
    pub phases: Vec<Phase>,
    pub sample_input_rows: Vec<u32>,
    pub sample_owners: Vec<SampleOwner>,
    pub pending: Vec<PendingCommit>,
}

impl Plan {
    /// Allocate persistent storage for repeated [`plan_into`] calls. With adequate capacities
    /// the successful path performs no heap allocation and never grows an output vector.
    pub fn with_capacity(row_capacity: usize, request_capacity: usize) -> Self {
        Plan {
            row_capacity: 0,
            real_rows: 0,
            sample_rows: 0,
            input_ids: Vec::with_capacity(row_capacity),
            positions: Vec::with_capacity(row_capacity),
            active: Vec::with_capacity(row_capacity),
            spans: Vec::with_capacity(request_capacity),
            phases: Vec::with_capacity(request_capacity),
            sample_input_rows: Vec::with_capacity(request_capacity),
            sample_owners: Vec::with_capacity(request_capacity),
            pending: Vec::with_capacity(request_capacity),
        }
    }

    fn clear(&mut self) {
        self.row_capacity = 0;
        self.real_rows = 0;
        self.sample_rows = 0;
        self.input_ids.clear();
        self.positions.clear();
        self.active.clear();
        self.spans.clear();
        self.phases.clear();
        self.sample_input_rows.clear();
        self.sample_owners.clear();
        self.pending.clear();
    }

    /// The borrowed view the resolver twin and the descriptor filler read.
    pub fn batch(&self) -> HostBatch<'_> {
        HostBatch {
            row_capacity: self.row_capacity,
            real_rows: self.real_rows,
            sample_rows: self.sample_rows,
            spans: &self.spans,
            positions: &self.positions,
            active: &self.active,
            sample_input_rows: &self.sample_input_rows,
        }
    }
}

/// Build a plan in caller-owned storage.
///
/// `frontiers[slot]` is the committed KV frontier and `generations[slot]` the slot generation;
/// neither is mutated here. A span must start EXACTLY at its slot's committed frontier and
/// carry the generation the host currently holds, so a plan built against a recycled slot is
/// refused at planning time rather than committed against the wrong request.
#[allow(clippy::too_many_arguments)]
pub fn plan_into(
    requests: &[Request<'_>],
    frontiers: &[u32],
    generations: &[u32],
    row_capacity: u32,
    max_ctx: u32,
    program: u32,
    out: &mut Plan,
) -> Result<()> {
    out.clear();
    let result = plan_inner(
        requests,
        frontiers,
        generations,
        row_capacity,
        max_ctx,
        program,
        out,
    );
    if result.is_err() {
        // A partially filled plan is worse than none: it looks stageable.
        out.clear();
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn plan_inner(
    requests: &[Request<'_>],
    frontiers: &[u32],
    generations: &[u32],
    row_capacity: u32,
    max_ctx: u32,
    program: u32,
    out: &mut Plan,
) -> Result<()> {
    require(row_capacity > 0 && max_ctx > 0, "zero capacity")?;
    require(
        frontiers.len() == generations.len(),
        "frontier and generation tables disagree on slot capacity",
    )?;
    let capacity = usize::try_from(row_capacity).map_err(|_| "token batch: row capacity")?;
    require(
        out.input_ids.capacity() >= capacity
            && out.positions.capacity() >= capacity
            && out.active.capacity() >= capacity
            && out.spans.capacity() >= requests.len()
            && out.phases.capacity() >= requests.len()
            && out.sample_input_rows.capacity() >= requests.len()
            && out.sample_owners.capacity() >= requests.len()
            && out.pending.capacity() >= requests.len(),
        "output buffer capacity",
    )?;

    // Decode requests, then prefill requests, each in caller order. Deterministic ordering is
    // for reproducibility and simple metadata construction — it is NOT a projection boundary,
    // and it is identical on every backend so a CPU-computed reference is comparable row for
    // row with a GPU result.
    let mut order: Vec<usize> = Vec::with_capacity(requests.len());
    order.extend(
        requests
            .iter()
            .enumerate()
            .filter(|(_, r)| r.phase == Phase::Decode)
            .map(|(i, _)| i),
    );
    order.extend(
        requests
            .iter()
            .enumerate()
            .filter(|(_, r)| r.phase == Phase::Prefill)
            .map(|(i, _)| i),
    );

    for (position, &index) in order.iter().enumerate() {
        let request = &requests[index];
        request.selection.validate()?;
        let slot = request.slot as usize;
        let state_slot = request.state_slot as usize;
        require(
            slot < frontiers.len() && state_slot < frontiers.len(),
            "physical or carried-state slot outside capacity",
        )?;
        // A request contributes AT MOST ONE span in this mode (§4.4). Duplicate slots are the
        // interesting failure: two spans on one slot would both start at its frontier and both
        // claim to advance it.
        let prior = &order[..position];
        require(
            !prior.iter().any(|&p| requests[p].id == request.id)
                && !prior.iter().any(|&p| requests[p].slot == request.slot)
                && !prior
                    .iter()
                    .any(|&p| requests[p].state_slot == request.state_slot),
            "duplicate request id, physical slot or carried-state slot",
        )?;
        require(
            generations[slot] == request.generation,
            "slot generation does not match the admitted request",
        )?;

        let n_rows = u32::try_from(request.tokens.len()).map_err(|_| "token batch: row count")?;
        require(n_rows > 0, "span with no scheduled tokens")?;
        let start = frontiers[slot];
        let end = start
            .checked_add(n_rows)
            .ok_or("token batch: span extent overflow")?;
        require(end <= max_ctx, "span exceeds the admitted context limit")?;
        require(
            out.input_ids.len().saturating_add(request.tokens.len()) <= capacity,
            "scheduled tokens exceed the row capacity",
        )?;

        let completes_prompt = match request.phase {
            // A decode row extends a request that already consumed its whole prompt, so its
            // position must be at or past the prompt's end.
            Phase::Decode => {
                require(
                    n_rows == 1 && start >= request.prompt_len && request.prompt_len > 0,
                    "decode span must be one token at or past the prompt end",
                )?;
                true
            }
            // A prefill span completes the prompt exactly when it reaches `prompt_len`. Length
            // one does NOT make it a decode span, and an intermediate chunk contributes zero
            // to S.
            Phase::Prefill => {
                require(end <= request.prompt_len, "prefill span overruns the prompt")?;
                end == request.prompt_len
            }
        };

        let row0 = u32::try_from(out.input_ids.len()).map_err(|_| "token batch: row offset")?;
        out.spans.push(PrefillSpan {
            row0,
            n_rows,
            slot: request.slot,
            flags: u32::from(start == 0) * PREFILL_SPAN_RESET_STATE,
            kv_row0: start,
            kv_len: end,
            state_slot: request.state_slot,
            program,
        });
        out.phases.push(request.phase);
        for (offset, &token) in request.tokens.iter().enumerate() {
            out.input_ids.push(token);
            out.positions.push(start + offset as u32);
            out.active.push(1);
        }

        let sample_index = completes_prompt.then(|| {
            let s = out.sample_input_rows.len() as u32;
            // "exactly its owner's last scheduled row" — the row whose hidden state predicts
            // the next token. For a completing prompt that is the prompt's FINAL input token,
            // included in the body normally and sampled here, rather than replayed through a
            // decode pass.
            out.sample_input_rows.push(row0 + n_rows - 1);
            out.sample_owners.push(SampleOwner {
                request: request.id,
                slot: request.slot,
                generation: request.generation,
                selection: request.selection,
            });
            s
        });
        out.pending.push(PendingCommit {
            request: request.id,
            slot: request.slot,
            generation: request.generation,
            expected_frontier: start,
            new_frontier: end,
            completes_prompt,
            sample_index,
        });
    }

    let real_rows = u32::try_from(out.input_ids.len()).map_err(|_| "token batch: real rows")?;
    // Padding belongs to NO span and carries no request's coordinates. `mixed_step`'s plan
    // extends the last owner's positions across the pad and has to bound that against
    // `max_ctx`; here padding is simply inert, which removes that check and the class of bug
    // it guards.
    out.input_ids.resize(capacity, 0);
    out.positions.resize(capacity, 0);
    out.active.resize(capacity, 0);
    out.row_capacity = row_capacity;
    out.real_rows = real_rows;
    out.sample_rows = out.sample_input_rows.len() as u32;

    // The plan is only complete if the resolver agrees with it. Running the twin here means a
    // planner bug is caught at plan time, not as a device trap.
    validate(&out.batch()).map_err(|refusal| format!("token batch: {refusal}"))?;
    Ok(())
}

/// Allocate an owned plan and delegate to [`plan_into`].
pub fn plan(
    requests: &[Request<'_>],
    frontiers: &[u32],
    generations: &[u32],
    row_capacity: u32,
    max_ctx: u32,
    program: u32,
) -> Result<Plan> {
    let capacity = usize::try_from(row_capacity).map_err(|_| "token batch: row capacity")?;
    let mut out = Plan::with_capacity(capacity, requests.len());
    plan_into(
        requests,
        frontiers,
        generations,
        row_capacity,
        max_ctx,
        program,
        &mut out,
    )?;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Row resolver — the Rust twin of runtime/common/token_batch.h
// ---------------------------------------------------------------------------------------------

/// Borrowed host image of a [`TokenBatch`], as the resolver reads it.
#[derive(Clone, Copy, Debug)]
pub struct HostBatch<'a> {
    pub row_capacity: u32,
    pub real_rows: u32,
    pub sample_rows: u32,
    pub spans: &'a [PrefillSpan],
    pub positions: &'a [u32],
    pub active: &'a [u32],
    pub sample_input_rows: &'a [u32],
}

/// One refusal, mirroring the `PLOW_TB_E_*` codes in `runtime/common/token_batch.h`.
///
/// The numeric [`Refusal::code`] is the contract: `plowrt`'s resolver test runs this twin and
/// the C header over identical span tables — including the malformed ones — and compares both
/// the acceptance and the exact code. Two resolvers that are never compared are two resolvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Null,
    Version,
    Flags,
    Capacity,
    Spans,
    Cover,
    Position,
    Active,
    KvLen,
    Sample,
    Row,
}

impl Refusal {
    pub fn code(self) -> i32 {
        match self {
            Refusal::Null => -1,
            Refusal::Version => -2,
            Refusal::Flags => -3,
            Refusal::Capacity => -4,
            Refusal::Spans => -5,
            Refusal::Cover => -6,
            Refusal::Position => -7,
            Refusal::Active => -8,
            Refusal::KvLen => -9,
            Refusal::Sample => -10,
            Refusal::Row => -11,
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Refusal::Null => "descriptor or a required array is missing",
            Refusal::Version => "descriptor version this build does not implement",
            Refusal::Flags => "reserved flags set",
            Refusal::Capacity => "real_rows exceeds row_capacity, or capacity is zero",
            Refusal::Spans => "span count disagrees with the live row count",
            Refusal::Cover => "spans do not cover [0, M) exactly (gap, overlap or zero length)",
            Refusal::Position => "positions[] disagrees with the span's arithmetic",
            Refusal::Active => "park mask disagrees with the span cover",
            Refusal::KvLen => "kv_len != kv_row0 + n_rows",
            Refusal::Sample => "a sample index is not a live row",
            Refusal::Row => "row index outside row_capacity",
        })
    }
}

/// One resolved packed row. `span == None` is a padding row, which is also the only case with
/// `active == false`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedRow {
    pub span: Option<u32>,
    pub local_row: u32,
    pub slot: u32,
    pub state_slot: u32,
    pub position: u32,
    pub active: bool,
}

/// Structural checks that touch no row array — the twin of `plow_token_batch_check`.
fn check(b: &HostBatch<'_>) -> std::result::Result<(), Refusal> {
    if b.row_capacity == 0 || b.real_rows > b.row_capacity {
        return Err(Refusal::Capacity);
    }
    if b.positions.len() < b.row_capacity as usize || b.active.len() < b.row_capacity as usize {
        return Err(Refusal::Null);
    }
    if b.real_rows > 0 && b.spans.is_empty() {
        return Err(Refusal::Spans);
    }
    if b.real_rows == 0 && !b.spans.is_empty() {
        return Err(Refusal::Spans);
    }
    if b.spans.len() as u64 > b.real_rows as u64 {
        return Err(Refusal::Cover);
    }
    if b.sample_input_rows.len() < b.sample_rows as usize {
        return Err(Refusal::Null);
    }
    if b.sample_rows > b.real_rows {
        return Err(Refusal::Sample);
    }
    Ok(())
}

/// Full `O(M + R)` sweep — the twin of `plow_token_batch_validate`. The defensive checks stay
/// on in release on both sides: they turn a mis-planned batch into a refusal instead of a
/// plausible wrong answer, and on AMD "plausible wrong answer" is the actual failure mode.
pub fn validate(b: &HostBatch<'_>) -> std::result::Result<(), Refusal> {
    check(b)?;
    let mut next = 0u32;
    for span in b.spans {
        let end = span
            .row0
            .checked_add(span.n_rows)
            .ok_or(Refusal::Cover)?;
        if span.n_rows == 0 || end > b.real_rows || span.row0 != next {
            return Err(Refusal::Cover);
        }
        if span.kv_len != span.kv_row0.checked_add(span.n_rows).ok_or(Refusal::KvLen)? {
            return Err(Refusal::KvLen);
        }
        for j in 0..span.n_rows {
            let row = (span.row0 + j) as usize;
            if b.active[row] == 0 {
                return Err(Refusal::Active);
            }
            if b.positions[row] != span.kv_row0 + j {
                return Err(Refusal::Position);
            }
        }
        next = end;
    }
    if next != b.real_rows {
        return Err(Refusal::Cover);
    }
    for row in b.real_rows..b.row_capacity {
        if b.active[row as usize] != 0 {
            return Err(Refusal::Active);
        }
    }
    for &row in &b.sample_input_rows[..b.sample_rows as usize] {
        if row >= b.real_rows || b.active[row as usize] == 0 {
            return Err(Refusal::Sample);
        }
    }
    Ok(())
}

/// Resolve one packed row — the twin of `plow_token_row`, reporting instead of trapping.
pub fn resolve_row(b: &HostBatch<'_>, row: u32) -> std::result::Result<ResolvedRow, Refusal> {
    validate(b)?;
    if row >= b.row_capacity {
        return Err(Refusal::Row);
    }
    if row >= b.real_rows {
        return Ok(ResolvedRow {
            span: None,
            local_row: 0,
            slot: 0,
            state_slot: 0,
            position: 0,
            active: false,
        });
    }
    // Same binary search as the device: last span whose `row0 <= row`.
    let index = b.spans.partition_point(|s| s.row0 <= row);
    if index == 0 {
        return Err(Refusal::Cover);
    }
    let span = &b.spans[index - 1];
    let local = row - span.row0;
    if local >= span.n_rows {
        return Err(Refusal::Cover);
    }
    Ok(ResolvedRow {
        span: Some((index - 1) as u32),
        local_row: local,
        slot: span.slot,
        state_slot: span.state_slot,
        position: span.kv_row0 + local,
        active: true,
    })
}

/// The `s`-th selected hidden row — the twin of `plow_token_sample_row`.
pub fn sample_row(b: &HostBatch<'_>, s: u32) -> std::result::Result<u32, Refusal> {
    validate(b)?;
    if s >= b.sample_rows {
        return Err(Refusal::Sample);
    }
    Ok(b.sample_input_rows[s as usize])
}

/// Build the device descriptor from device base addresses for the four row arrays, the span
/// table and the sample-row list. The counts come from the plan; the pointers come from
/// whatever the backend uploaded them into, which is why this takes addresses rather than
/// allocating.
pub fn descriptor(
    plan: &Plan,
    spans: u64,
    input_ids: u64,
    positions: u64,
    active: u64,
    sample_rows_idx: u64,
) -> Result<TokenBatch> {
    validate(&plan.batch()).map_err(|refusal| format!("token batch: {refusal}"))?;
    require(
        input_ids != 0 && positions != 0 && active != 0,
        "descriptor requires the row arrays",
    )?;
    require(
        plan.spans.is_empty() == (spans == 0),
        "span table address disagrees with the span count",
    )?;
    require(
        (plan.sample_rows == 0) == (sample_rows_idx == 0),
        "sample row address disagrees with S",
    )?;
    Ok(TokenBatch {
        version: TOKEN_BATCH_VERSION,
        row_capacity: plan.row_capacity,
        real_rows: plan.real_rows,
        sample_rows: plan.sample_rows,
        n_spans: plan.spans.len() as u32,
        flags: 0,
        _pad0: 0,
        _pad1: 0,
        spans,
        input_ids,
        positions,
        active,
        sample_rows_idx,
    })
}

// ---------------------------------------------------------------------------------------------
// Program roles and the load-time capability check
// ---------------------------------------------------------------------------------------------

/// What a program in this contract IS. A new program kind is explicitly body or output; it is
/// never an ordinary sampled decode rung, and must not be picked up by a scan that looks for
/// one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramRole {
    /// The packed body: `M` rows through every layer, no row selection, no head.
    Body,
    /// The terminal segment: gather -> final norm -> [softcap] -> LM head -> selection.
    Output,
}

/// Derive a program's role from its instruction stream, so a declared role can be checked
/// against what the program actually contains rather than trusted.
///
/// The discriminator is [`DevOp::RowGather`]: the output segment begins with exactly one, at
/// instruction zero, because gathering AFTER any tail stage would normalize or project rows
/// nobody asked for. A body program contains none — a gather in the body would mean the row
/// selection happened before the final residual, which is the "earlier alias" wrong-model bug.
pub fn role_of(program: &aux_program::Program) -> Result<ProgramRole> {
    let gathers = program
        .insts
        .iter()
        .filter(|i| i.op == DevOp::RowGather as u16)
        .count();
    match gathers {
        0 => Ok(ProgramRole::Body),
        1 => {
            require(
                program.insts[0].op == DevOp::RowGather as u16,
                "the output segment's row gather must be its first instruction",
            )?;
            Ok(ProgramRole::Output)
        }
        _ => Err("token batch: more than one row gather in one program".into()),
    }
}

/// What one (backend, family) pair declares it can execute on this route.
///
/// Everything absent from `descriptor_aware` is refused at load WITH THE CAPABILITY NAMED. On
/// AMD the interpreter's dispatch `default:` writes nothing and does not trap, so a missing arm
/// is a silent-wrong-answer hazard rather than a crash; the CPU's `plow_cpu_has` probe and the
/// CUDA module load carry the same duty.
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Backend/arch this describes. Named in every refusal, because a result on gfx942 with one
    /// model is evidence about gfx942 and that model.
    pub target: String,
    pub descriptor_version: u32,
    pub row_capacity: u32,
    pub sample_capacity: u32,
    /// Maximum prefill spans one step may carry — the D-class limit (§5.4). `u32::MAX` when the
    /// program has no D-class layer.
    pub max_prefill_spans: u32,
    /// Opcodes with a descriptor-aware dispatch arm on this backend.
    pub descriptor_aware: Vec<u16>,
    /// C-class opcodes whose descriptor-aware form has been CONVERTED and qualified for this
    /// target. A C-class opcode not listed here is refused even if it has an arm: having an arm
    /// says the backend can execute the packet, not that the packet reads `positions[]`.
    pub converted_c: Vec<u16>,
}

/// A refusal that names the capability, so it can never read as a silent fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityRefusal {
    pub capability: String,
    pub detail: String,
}

impl std::fmt::Display for CapabilityRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "token batch refused: {} — {}", self.capability, self.detail)
    }
}

/// One opcode's audit line: what it is, what class it is, and what was decided about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuditEntry {
    pub op: u16,
    pub class: RowClass,
    pub descriptor_aware: bool,
    pub converted: bool,
}

impl AuditEntry {
    /// Whether this opcode may run on the token-batch route as declared.
    pub fn admitted(self) -> bool {
        self.descriptor_aware && (self.class.packs() || self.converted)
    }
}

impl Capabilities {
    fn names(&self, op: u16) -> String {
        DevOp::from_u16(op).map_or_else(
            || format!("{}.opcode_{op}", self.target),
            |o| format!("{}.{}", self.target, o.c_name()),
        )
    }

    /// Classify every opcode a program contains and report the audit lines, ascending by
    /// opcode. This is the input to [`Self::refuse_program`] and the data the operator-audit
    /// tool prints; both read the same classification rather than restating it.
    pub fn audit(&self, ops: impl Iterator<Item = u16>) -> Vec<AuditEntry> {
        let mut seen: Vec<u16> = ops.collect();
        seen.sort_unstable();
        seen.dedup();
        seen.into_iter()
            .map(|op| AuditEntry {
                op,
                class: class_of_op(op),
                descriptor_aware: self.descriptor_aware.contains(&op),
                converted: self.converted_c.contains(&op),
            })
            .collect()
    }

    /// Refuse a program this target cannot execute on the token-batch route, naming the first
    /// capability that fails. Opcodes are checked in ascending numeric order so the refusal is
    /// deterministic and diffable across runs.
    pub fn refuse_program(
        &self,
        ops: impl Iterator<Item = u16>,
    ) -> std::result::Result<Vec<AuditEntry>, CapabilityRefusal> {
        if self.descriptor_version != TOKEN_BATCH_VERSION {
            return Err(CapabilityRefusal {
                capability: format!("{}.{OBJECT_CAPABILITY}", self.target),
                detail: format!(
                    "target declares descriptor version {}, this build implements {TOKEN_BATCH_VERSION}",
                    self.descriptor_version
                ),
            });
        }
        let entries = self.audit(ops);
        for entry in &entries {
            if !entry.descriptor_aware {
                return Err(CapabilityRefusal {
                    capability: self.names(entry.op),
                    detail: format!(
                        "class {} opcode has no descriptor-aware arm on this target; a missing \
                         arm does not trap, it writes nothing",
                        entry.class.letter()
                    ),
                });
            }
            if !entry.class.packs() && !entry.converted {
                return Err(CapabilityRefusal {
                    capability: self.names(entry.op),
                    detail: match entry.class {
                        RowClass::C => "class C: derives every row's position from a packet \
                                        scalar, and no converted form is declared for this \
                                        target"
                            .into(),
                        RowClass::D => "class D: carries per-sequence state with no request \
                                        axis, and no batched variable-length form is declared \
                                        for this target"
                            .into(),
                        _ => unreachable!("A and B pack"),
                    },
                });
            }
        }
        Ok(entries)
    }

    /// Refuse a plan this target cannot admit. Capacities and the D-class span limit are
    /// checked BEFORE any device work, because a plan that exceeds the limit must be refused,
    /// not truncated.
    pub fn refuse_plan(&self, plan: &Plan) -> std::result::Result<(), CapabilityRefusal> {
        if plan.row_capacity > self.row_capacity {
            return Err(CapabilityRefusal {
                capability: format!("{}.row_capacity", self.target),
                detail: format!(
                    "plan needs {} rows, target compiled for {}",
                    plan.row_capacity, self.row_capacity
                ),
            });
        }
        if plan.sample_rows > self.sample_capacity {
            return Err(CapabilityRefusal {
                capability: format!("{}.sample_capacity", self.target),
                detail: format!(
                    "plan needs S={}, target compiled for {}",
                    plan.sample_rows, self.sample_capacity
                ),
            });
        }
        let prefill_spans = plan
            .phases
            .iter()
            .filter(|p| **p == Phase::Prefill)
            .count() as u32;
        if prefill_spans > self.max_prefill_spans {
            return Err(CapabilityRefusal {
                capability: format!("{}.d_class_prefill_spans", self.target),
                detail: format!(
                    "plan carries {prefill_spans} prefill spans, target admits {} (a D-class \
                     layer's state has no request axis)",
                    self.max_prefill_spans
                ),
            });
        }
        Ok(())
    }

    /// The output segment additionally needs a real [`DevOp::RowGather`] arm. Stated separately
    /// from [`Self::refuse_program`] so the claim "the route is ARMED" and the claim "the route
    /// CAN FIRE" stay distinguishable: only the second licenses a measurement.
    pub fn can_run_output(&self) -> std::result::Result<(), CapabilityRefusal> {
        if self.descriptor_aware.contains(&(DevOp::RowGather as u16)) {
            return Ok(());
        }
        Err(CapabilityRefusal {
            capability: format!("{}.{ROW_GATHER_CAPABILITY}", self.target),
            detail: "no RowGather arm: the terminal segment would leave its gathered rows at \
                     whatever the buffer held"
                .into(),
        })
    }
}

#[cfg(test)]
#[path = "token_batch_tests.rs"]
mod tests;
