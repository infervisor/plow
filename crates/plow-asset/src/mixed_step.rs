use crate::aux_program;
use packet::dev::{PrefillSpan, PREFILL_SPAN_RESET_STATE};
use packet::devbuild::{SECT_CUBIN, SECT_HSACO, SECT_NAME_LEN, SECT_PROGRAMS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, String>;

pub const SECTION: &str = "mixed_step";
pub const VERSION: u32 = 1;
pub const PROGRAM_CAPABILITY: &str = aux_program::CAPABILITY;
pub const OBJECT_CAPABILITY: &str = "plow.mixed.interpreter";

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("mixed step: {reason}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RowPhase {
    Decode = 0,
    Prefill = 1,
    Parked = 2,
}

/// One dense activation row. Device adapters may transpose these fields into
/// their existing input tensors, but may not infer a physical slot from `row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub token: u32,
    pub slot: u32,
    pub state_slot: u32,
    pub position: u32,
    pub kv_len: u32,
    pub phase: RowPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeRequest {
    pub slot: u32,
    pub state_slot: u32,
    pub token: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillRequest<'a> {
    pub slot: u32,
    pub state_slot: u32,
    pub start: u32,
    pub tokens: &'a [u32],
    pub prompt_len: u32,
}

/// Backend-neutral host plan for one combined dispatch.
///
/// Decode rows occupy `[0, decode_rows)`. The canonical [`PrefillSpan`] values
/// cover `[decode_rows, real_rows)` densely. `parked` is the compiled-row binary
/// mask: zero for `[0, real_rows)`, one for the padded suffix. In this contract `PrefillSpan::program`
/// names the selected auxiliary program, rather than an ordinary prefill rung.
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub decode_rows: u32,
    pub real_rows: u32,
    pub rows: Vec<Row>,
    pub prefill_spans: Vec<PrefillSpan>,
    pub parked: Vec<u32>,
    /// Largest mapped KV end per active physical slot, including bounded
    /// parked padding for adapters whose fixed-width kernels still address it.
    pub mapped_ends: Vec<(u32, u32)>,
}

impl Plan {
    /// Allocate persistent storage for repeated [`plan_into`] calls.
    pub fn with_capacity(
        row_capacity: usize,
        prefill_capacity: usize,
        active_capacity: usize,
    ) -> Self {
        Self {
            decode_rows: 0,
            real_rows: 0,
            rows: Vec::with_capacity(row_capacity),
            prefill_spans: Vec::with_capacity(prefill_capacity),
            parked: Vec::with_capacity(row_capacity),
            mapped_ends: Vec::with_capacity(active_capacity),
        }
    }

    fn clear(&mut self) {
        self.decode_rows = 0;
        self.real_rows = 0;
        self.rows.clear();
        self.prefill_spans.clear();
        self.parked.clear();
        self.mapped_ends.clear();
    }
}

/// Build a mixed plan in caller-owned storage. With sufficient capacities, the
/// successful path performs no heap allocation and never grows an output vector.
pub fn plan_into(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    out: &mut Plan,
) -> Result<()> {
    out.clear();
    let result = plan_into_inner(
        decode,
        prefill,
        frontiers,
        rows,
        max_ctx,
        auxiliary_program,
        out,
    );
    if result.is_err() {
        out.clear();
    }
    result
}

fn plan_into_inner(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
    out: &mut Plan,
) -> Result<()> {
    require(
        (!decode.is_empty() || !prefill.is_empty()) && rows > 0 && max_ctx > 0,
        "empty work or capacity",
    )?;
    require(
        decode.len().saturating_add(prefill.len()) <= frontiers.len(),
        "active request capacity",
    )?;

    let capacity = usize::try_from(rows).map_err(|_| "mixed step: row capacity")?;
    let active = decode
        .len()
        .checked_add(prefill.len())
        .ok_or("mixed step: active request overflow")?;
    require(
        out.rows.capacity() >= capacity
            && out.prefill_spans.capacity() >= prefill.len()
            && out.parked.capacity() >= capacity
            && out.mapped_ends.capacity() >= active,
        "output buffer capacity",
    )?;

    for (index, request) in decode.iter().enumerate() {
        let slot = request.slot as usize;
        let state_slot = request.state_slot as usize;
        require(
            slot < frontiers.len()
                && state_slot < frontiers.len()
                && !decode[..index]
                    .iter()
                    .any(|prior| prior.slot == request.slot)
                && !decode[..index]
                    .iter()
                    .any(|prior| prior.state_slot == request.state_slot),
            "physical/state slot or duplicate",
        )?;
        let position = frontiers[slot];
        let kv_len = position
            .checked_add(1)
            .ok_or("mixed step: decode position overflow")?;
        require(
            kv_len <= max_ctx && out.rows.len() < capacity,
            "decode extent",
        )?;
        out.rows.push(Row {
            token: request.token,
            slot: request.slot,
            state_slot: request.state_slot,
            position,
            kv_len,
            phase: RowPhase::Decode,
        });
        out.mapped_ends.push((request.slot, kv_len));
    }
    let decode_rows = u32::try_from(out.rows.len()).map_err(|_| "mixed step: decode rows")?;

    for (index, request) in prefill.iter().enumerate() {
        let slot = request.slot as usize;
        let state_slot = request.state_slot as usize;
        let duplicate_physical = decode.iter().any(|prior| prior.slot == request.slot)
            || prefill[..index]
                .iter()
                .any(|prior| prior.slot == request.slot);
        let duplicate_state = decode
            .iter()
            .any(|prior| prior.state_slot == request.state_slot)
            || prefill[..index]
                .iter()
                .any(|prior| prior.state_slot == request.state_slot);
        require(
            slot < frontiers.len()
                && state_slot < frontiers.len()
                && !duplicate_physical
                && !duplicate_state,
            "physical/state slot or duplicate",
        )?;
        let n_rows =
            u32::try_from(request.tokens.len()).map_err(|_| "mixed step: prefill row count")?;
        let end = request
            .start
            .checked_add(n_rows)
            .ok_or("mixed step: prefill extent overflow")?;
        require(
            n_rows > 0
                && request.start == frontiers[slot]
                && end <= request.prompt_len
                && end <= max_ctx
                && out.rows.len().saturating_add(request.tokens.len()) <= capacity,
            "prefill frontier or extent",
        )?;
        let row0 = u32::try_from(out.rows.len()).map_err(|_| "mixed step: prefill row offset")?;
        out.prefill_spans.push(PrefillSpan {
            row0,
            n_rows,
            slot: request.slot,
            flags: u32::from(request.start == 0) * PREFILL_SPAN_RESET_STATE,
            kv_row0: request.start,
            kv_len: end,
            state_slot: request.state_slot,
            program: auxiliary_program,
        });
        for (offset, &token) in request.tokens.iter().enumerate() {
            let position = request.start + offset as u32;
            out.rows.push(Row {
                token,
                slot: request.slot,
                state_slot: request.state_slot,
                position,
                kv_len: position + 1,
                phase: RowPhase::Prefill,
            });
        }
        out.mapped_ends.push((request.slot, end));
    }

    let real_rows = u32::try_from(out.rows.len()).map_err(|_| "mixed step: real rows")?;
    let owner = out
        .rows
        .last()
        .copied()
        .ok_or("mixed step: padding owner")?;
    let pad = capacity - out.rows.len();
    let padded_end = owner
        .position
        .checked_add(1)
        .and_then(|end| end.checked_add(pad as u32))
        .ok_or("mixed step: padding extent overflow")?;
    require(padded_end <= max_ctx, "padding exceeds physical context")?;
    out.parked.resize(capacity, 0);
    out.parked[real_rows as usize..].fill(1);
    for offset in 0..pad {
        let position = owner.position + 1 + offset as u32;
        out.rows.push(Row {
            token: 0,
            slot: owner.slot,
            state_slot: owner.state_slot,
            position,
            kv_len: position + 1,
            phase: RowPhase::Parked,
        });
    }
    if pad > 0 {
        let end = out
            .mapped_ends
            .iter_mut()
            .find(|(slot, _)| *slot == owner.slot)
            .ok_or("mixed step: padding owner mapping")?;
        end.1 = padded_end;
    }
    out.decode_rows = decode_rows;
    out.real_rows = real_rows;
    Ok(())
}

/// Allocate an owned reference plan and delegate to [`plan_into`].
pub fn plan(
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
    rows: u32,
    max_ctx: u32,
    auxiliary_program: u32,
) -> Result<Plan> {
    let row_capacity = usize::try_from(rows).map_err(|_| "mixed step: row capacity")?;
    let active_capacity = decode
        .len()
        .checked_add(prefill.len())
        .ok_or("mixed step: active request overflow")?;
    let mut out = Plan::with_capacity(row_capacity, prefill.len(), active_capacity);
    plan_into(
        decode,
        prefill,
        frontiers,
        rows,
        max_ctx,
        auxiliary_program,
        &mut out,
    )?;
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    pub name: String,
    pub version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadKind {
    Programs,
    Cubin,
    Hsaco,
}

impl PayloadKind {
    pub fn section_kind(self) -> u32 {
        match self {
            Self::Programs => SECT_PROGRAMS,
            Self::Cubin => SECT_CUBIN,
            Self::Hsaco => SECT_HSACO,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadBinding {
    pub section: String,
    pub kind: PayloadKind,
    pub version: u32,
    pub sha256: String,
    pub capability: Capability,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramBinding {
    pub index: u32,
    pub payload: PayloadBinding,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub rows: u32,
    pub decode_rows: u32,
    pub program: ProgramBinding,
    #[serde(default)]
    pub objects: Vec<PayloadBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub n_cu: u32,
    /// Maximum number of decode rows plus prefill spans in one step.
    pub max_active_requests: u32,
    /// Size of the persistent KV/state slot namespace. Active requests may use
    /// sparse indices anywhere in `[0, physical_slot_capacity)`.
    pub physical_slot_capacity: u32,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Copy)]
pub struct ValidatedManifest<'a> {
    manifest: &'a Manifest,
}

#[derive(Clone, Copy)]
pub struct Payload<'a> {
    pub section: &'a str,
    pub kind: PayloadKind,
    pub version: u32,
    pub n_cu: u32,
    pub bytes: &'a [u8],
}

fn identifier(s: &str, max: usize) -> bool {
    !s.is_empty()
        && s.len() <= max
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub fn payload_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl PayloadBinding {
    fn validate(&self) -> Result<()> {
        require(
            identifier(&self.section, SECT_NAME_LEN - 1)
                && self.section != SECTION
                && self.version > 0
                && digest(&self.sha256)
                && identifier(&self.capability.name, 63)
                && self.capability.version > 0,
            "payload identity",
        )
    }

    fn bind_identity(&self, expected_n_cu: u32, payload: &Payload<'_>) -> Result<()> {
        self.validate()?;
        require(
            payload.section == self.section
                && payload.kind == self.kind
                && payload.version == self.version
                && payload.n_cu == expected_n_cu
                && payload_sha256(payload.bytes) == self.sha256,
            "payload binding",
        )
    }

    fn bind_object_with(
        &self,
        expected_kind: PayloadKind,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        mut read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        require(self.kind == expected_kind, "object backend kind")?;
        self.bind_identity(expected_n_cu, payload)?;
        require(
            read_capability(&self.capability.name) == Some(self.capability.version),
            "object capability",
        )
    }
}

impl Manifest {
    /// Validate packet metadata once at load time. Per-step callers retain the
    /// returned wrapper, whose plan checks do not allocate.
    pub fn validate(&self) -> Result<ValidatedManifest<'_>> {
        require(
            self.version == VERSION
                && self.n_cu > 0
                && self.max_active_requests > 0
                && self.max_active_requests <= self.physical_slot_capacity
                && !self.variants.is_empty(),
            "version or capacity",
        )?;
        let mut shapes = std::collections::BTreeSet::new();
        let mut payloads = std::collections::BTreeMap::new();
        for variant in &self.variants {
            require(
                variant.rows > 1
                    && variant.decode_rows > 0
                    && variant.decode_rows < variant.rows
                    && variant.decode_rows <= self.max_active_requests
                    && shapes.insert((variant.rows, variant.decode_rows))
                    && variant.program.payload.kind == PayloadKind::Programs
                    && variant.program.payload.version == VERSION
                    && variant.program.payload.capability.name == PROGRAM_CAPABILITY
                    && variant.program.payload.capability.version == VERSION,
                "variant geometry",
            )?;
            variant.program.payload.validate()?;
            record_payload(&mut payloads, &variant.program.payload)?;
            require(!variant.objects.is_empty(), "variant has no backend object")?;
            let mut objects = std::collections::BTreeSet::new();
            let mut backends = std::collections::BTreeSet::new();
            for object in &variant.objects {
                require(
                    matches!(object.kind, PayloadKind::Cubin | PayloadKind::Hsaco)
                        && object.version == VERSION
                        && object.capability.name == OBJECT_CAPABILITY
                        && object.capability.version == VERSION
                        && backends.insert(object.kind.section_kind())
                        && objects.insert((object.kind.section_kind(), object.section.as_str())),
                    "object kind or duplicate",
                )?;
                object.validate()?;
                record_payload(&mut payloads, object)?;
            }
        }
        Ok(ValidatedManifest { manifest: self })
    }
}

impl ValidatedManifest<'_> {
    pub fn variant(&self, rows: u32, decode_rows: u32) -> Option<&Variant> {
        self.manifest
            .variants
            .iter()
            .find(|v| v.rows == rows && v.decode_rows == decode_rows)
    }

    pub fn validate_plan(&self, plan: &Plan) -> Result<&Variant> {
        let rows = u32::try_from(plan.rows.len()).map_err(|_| "mixed step: plan rows")?;
        let variant = self
            .variant(rows, plan.decode_rows)
            .ok_or("mixed step: missing plan variant")?;
        variant.validate_plan(
            plan,
            self.manifest.max_active_requests,
            self.manifest.physical_slot_capacity,
        )?;
        Ok(variant)
    }
}

impl Variant {
    fn validate_plan(
        &self,
        plan: &Plan,
        max_active_requests: u32,
        physical_slot_capacity: u32,
    ) -> Result<()> {
        require(
            plan.rows.len() == self.rows as usize
                && plan.decode_rows == self.decode_rows
                && plan.decode_rows <= plan.real_rows
                && plan.real_rows <= self.rows
                && plan
                    .prefill_spans
                    .len()
                    .saturating_add(plan.decode_rows as usize)
                    <= max_active_requests as usize
                && plan
                    .prefill_spans
                    .iter()
                    .all(|span| span.program == self.program.index),
            "plan does not match variant",
        )?;

        let decode_rows = plan.decode_rows as usize;
        let real_rows = plan.real_rows as usize;
        let owner = plan.rows[real_rows - 1];
        let parked_end = (real_rows < plan.rows.len()).then(|| plan.rows.last().unwrap().kv_len);
        require(
            plan.mapped_ends.len() == decode_rows + plan.prefill_spans.len(),
            "mapped slot count",
        )?;
        require(
            plan.rows[..decode_rows].iter().all(|row| {
                row.phase == RowPhase::Decode
                    && row.kv_len == row.position.checked_add(1).unwrap_or(0)
            }),
            "decode row coordinates",
        )?;

        let mut next = decode_rows;
        for (index, row) in plan.rows[..decode_rows].iter().enumerate() {
            let prior = &plan.rows[..index];
            let mapped_end = parked_end
                .filter(|_| row.slot == owner.slot)
                .unwrap_or(row.kv_len);
            require(
                row.slot < physical_slot_capacity
                    && row.state_slot < physical_slot_capacity
                    && !prior.iter().any(|p| p.slot == row.slot)
                    && !prior.iter().any(|p| p.state_slot == row.state_slot)
                    && plan.mapped_ends[index] == (row.slot, mapped_end),
                "duplicate or out-of-range active slot",
            )?;
        }
        for (span_index, span) in plan.prefill_spans.iter().enumerate() {
            let end = span
                .row0
                .checked_add(span.n_rows)
                .ok_or("mixed step: prefill row overflow")?;
            let kv_end = span
                .kv_row0
                .checked_add(span.n_rows)
                .ok_or("mixed step: prefill KV overflow")?;
            let prior_span = &plan.prefill_spans[..span_index];
            let prior_physical = plan.rows[..decode_rows]
                .iter()
                .any(|row| row.slot == span.slot)
                || prior_span.iter().any(|prior| prior.slot == span.slot);
            let prior_state = plan.rows[..decode_rows]
                .iter()
                .any(|row| row.state_slot == span.state_slot)
                || prior_span
                    .iter()
                    .any(|prior| prior.state_slot == span.state_slot);
            let mapped_end = parked_end
                .filter(|_| span.slot == owner.slot)
                .unwrap_or(span.kv_len);
            require(
                span.row0 as usize == next
                    && span.n_rows > 0
                    && end as usize <= real_rows
                    && span.kv_len == kv_end
                    && span.flags == u32::from(span.kv_row0 == 0) * PREFILL_SPAN_RESET_STATE
                    && span.slot < physical_slot_capacity
                    && span.state_slot < physical_slot_capacity
                    && !prior_physical
                    && !prior_state
                    && plan.mapped_ends[decode_rows + span_index] == (span.slot, mapped_end),
                "prefill span coverage",
            )?;
            for (offset, row) in plan.rows[span.row0 as usize..end as usize]
                .iter()
                .enumerate()
            {
                let position = span.kv_row0 + offset as u32;
                require(
                    row.phase == RowPhase::Prefill
                        && row.slot == span.slot
                        && row.state_slot == span.state_slot
                        && row.position == position
                        && row.kv_len == position + 1,
                    "prefill row coordinates",
                )?;
            }
            next = end as usize;
        }
        require(next == real_rows, "real row coverage")?;
        for (offset, row) in plan.rows[real_rows..].iter().enumerate() {
            let position = owner
                .position
                .checked_add(1)
                .and_then(|p| p.checked_add(offset as u32))
                .ok_or("mixed step: parked row overflow")?;
            let kv_len = position
                .checked_add(1)
                .ok_or("mixed step: parked KV overflow")?;
            require(
                row.phase == RowPhase::Parked
                    && row.token == 0
                    && row.slot == owner.slot
                    && row.state_slot == owner.state_slot
                    && row.position == position
                    && row.kv_len == kv_len,
                "parked row coordinates",
            )?;
        }
        require(
            plan.parked.len() == plan.rows.len()
                && plan.parked[..real_rows].iter().all(|&value| value == 0)
                && plan.parked[real_rows..].iter().all(|&value| value == 1),
            "parked suffix or mapping",
        )
    }

    pub fn bind_program(
        &self,
        expected_n_cu: u32,
        tensor_count: usize,
        program: &Payload<'_>,
    ) -> Result<aux_program::Section> {
        self.program.payload.bind_identity(expected_n_cu, program)?;
        require(
            self.program.payload.kind == PayloadKind::Programs
                && self.program.payload.capability.name == aux_program::CAPABILITY
                && self.program.payload.capability.version == aux_program::VERSION,
            "program capability",
        )?;
        let parsed = aux_program::parse(program.bytes, expected_n_cu, tensor_count)?;
        require(
            parsed
                .programs
                .get(self.program.index as usize)
                .is_some_and(|p| p.rows == self.rows),
            "auxiliary program index",
        )?;
        Ok(parsed)
    }

    /// Bind a CUDA object after its adapter has read an actual module or ELF
    /// capability symbol. The callback maps the backend-neutral capability
    /// name to the object's initialized u32 value.
    pub fn bind_cubin_with(
        &self,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        self.bind_object_with(PayloadKind::Cubin, expected_n_cu, payload, read_capability)
    }

    /// HSACO twin of [`Self::bind_cubin_with`]. HSA module symbol lookup stays
    /// in the runtime adapter; shared policy only compares the proven value.
    pub fn bind_hsaco_with(
        &self,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        self.bind_object_with(PayloadKind::Hsaco, expected_n_cu, payload, read_capability)
    }

    fn bind_object_with(
        &self,
        kind: PayloadKind,
        expected_n_cu: u32,
        payload: &Payload<'_>,
        read_capability: impl FnMut(&str) -> Option<u32>,
    ) -> Result<()> {
        self.objects
            .iter()
            .find(|binding| binding.section == payload.section && binding.kind == kind)
            .ok_or_else(|| "mixed step: undeclared object".to_string())?
            .bind_object_with(kind, expected_n_cu, payload, read_capability)
    }
}

fn record_payload(
    payloads: &mut std::collections::BTreeMap<(u32, String), (u32, String, Capability)>,
    binding: &PayloadBinding,
) -> Result<()> {
    let key = (binding.kind.section_kind(), binding.section.clone());
    let identity = (
        binding.version,
        binding.sha256.clone(),
        binding.capability.clone(),
    );
    require(
        payloads
            .insert(key, identity.clone())
            .is_none_or(|existing| existing == identity),
        "conflicting payload identity",
    )
}

#[cfg(test)]
#[path = "mixed_step_tests.rs"]
mod tests;
