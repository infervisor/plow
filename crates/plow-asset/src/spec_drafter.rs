//! External speculative drafters: a drafter directory that attaches to an already-emitted target
//! packet without changing it.
//!
//! The directory holds [`FILE`] (this manifest), [`PROGRAMS_FILE`] (an [`crate::aux_program`]
//! section whose operands index the drafter's own tensor table), an optional [`INIT_FILE`]
//! (bytes of `init` tensors) and the drafter's weight shards. The runtime builds a separate
//! pointer table for the drafter programs: `import` entries point at the target's tensors of the
//! same name, the others are allocated (weights bound from the shards by name). Target programs
//! and bytes are never touched.
//!
//! Compatibility is fail-closed: every field in [`Requires`] must be proven by the target
//! ([`TargetView`]) or the attach is refused, naming the first field that differs. A target
//! without a `spec.target.json` proves only facts the runtime derives itself (packet digest,
//! tensor table, tokenizer, object symbols), so an external drafter for such a packet pins
//! `target_packet_sha256`.
use std::collections::BTreeMap;

use packet::dev::{DevInst64, DevOp};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::serve_manifest::WeightPin;
use crate::spec_draft;

pub const FILE: &str = "drafter.json";
pub const PROGRAMS_FILE: &str = "drafter.programs";
pub const INIT_FILE: &str = "drafter.init";
pub const CONTRACT: &str = "plow.spec.drafter.v1";

/// Data-flow shape of the drafter; the runtime never branches on model names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Reads the target's committed KV and final hidden row; owns no KV (Gemma assistant MTP).
    ChainSharedKv,
    /// Autoregressive over its own KV layers (EAGLE-3, nextn MTP layers).
    ChainOwnKv,
    /// One non-causal block per sequence over its own KV layers (DSpark, DFlash).
    BlockOwnKv,
}

impl Kind {
    pub fn owns_kv(self) -> bool {
        !matches!(self, Kind::ChainSharedKv)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    ReadWrite,
}

/// Where a drafter tensor's memory comes from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    /// The target tensor of the same name (exact byte size).
    Import { access: Access },
    /// Bound by name from the drafter's weight shards.
    Weight,
    /// `len` bytes at `offset` of [`INIT_FILE`].
    Init { offset: u64, len: u64 },
    /// Device scratch the drafter programs own (activations, staged inputs).
    Scratch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub name: String,
    pub bytes: u64,
    pub source: Source,
}

/// The tokenizer contract: identical base vocabulary, and every drafter added token present in
/// the target under the same id (the target may add more, e.g. media tokens).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerReq {
    /// [`vocab_digest`] of the drafter's `tokenizer.json`.
    pub vocab_sha256: String,
    /// The drafter's added tokens as `(id, content)`.
    pub added_tokens: Vec<(u32, String)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requires {
    /// sha256 of the target's packet file; required when the target has no `spec.target.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_packet_sha256: Option<String>,
    /// Target model identity, proven only by a `spec.target.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_sha256: Option<String>,
    pub tokenizer: Option<TokenizerReq>,
    /// Symbols the decode object running the drafter programs must export with these values.
    #[serde(default)]
    pub object_symbols: BTreeMap<String, u32>,
    /// Target geometry facts (`spec.target.json` keys) the drafter math assumes.
    #[serde(default)]
    pub geometry: BTreeMap<String, String>,
    /// Target layers whose post-layer residual the drafter reads.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub taps: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub contract: String,
    pub kind: Kind,
    /// Free-form label for logs, e.g. `gemma4_assistant`.
    pub family: String,
    /// Interpreter grid the programs were scheduled for.
    pub n_cu: u32,
    pub requires: Requires,
    pub tensors: Vec<Tensor>,
    /// Draft programs; its handles index [`Manifest::tensors`].
    pub draft: spec_draft::Manifest,
    /// The drafter's weight shards, pinned like a packet's checkpoint.
    pub weights: Vec<WeightPin>,
}

/// The opcodes an external drafter's programs may use, with the `t` slots each one writes
/// (`None`: not allowed). The runtime refuses a program that writes a read-only import, so
/// every allowed opcode's writes must be listed here exactly; an opcode joins only after its
/// interpreter body was audited (integer-field handles included, see
/// [`crate::aux_program::integer_tensor_handles`]).
pub fn vetted_writes(op: DevOp, inst: &DevInst64) -> Option<&'static [usize]> {
    Some(match op {
        DevOp::Embed
        | DevOp::RowGather
        | DevOp::HeadNormRope
        | DevOp::Residual
        | DevOp::FlashMerge
        | DevOp::Argmax
        | DevOp::ArgmaxFin => &[0],
        DevOp::RmsNorm => &[0, 3, 4],
        // Folded prologues (`i[3] != 0`) write their staging operands too: not vetted.
        DevOp::Gemv | DevOp::GemvGlu if inst.i[3] == 0 => &[0],
        DevOp::FlashDecodeFp8 => &[0, 1],
        DevOp::NormResidualNorm => &[0, 1],
        _ => return None,
    })
}

/// What a target proves about itself. Built by the runtime from the loaded packet.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TargetView {
    pub packet_sha256: String,
    /// From `spec.target.json` only.
    pub model: Option<String>,
    pub revision: Option<String>,
    pub checkpoint_sha256: Option<String>,
    pub geometry: BTreeMap<String, String>,
    pub taps: Vec<u32>,
    /// `(name, bytes)` in tensor-table order.
    pub tensors: Vec<(String, u64)>,
    /// `None` when the bundle has no `tokenizer.json`.
    pub tokenizer: Option<TokenizerFacts>,
    pub object_symbols: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenizerFacts {
    pub vocab_sha256: String,
    pub added_tokens: BTreeMap<u32, String>,
}

impl TokenizerFacts {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("tokenizer.json: {e}"))?;
        let mut added = BTreeMap::new();
        for t in v["added_tokens"].as_array().into_iter().flatten() {
            let (Some(id), Some(content)) = (t["id"].as_u64(), t["content"].as_str()) else {
                return Err("tokenizer.json: added token without id/content".into());
            };
            added.insert(id as u32, content.to_string());
        }
        Ok(Self {
            vocab_sha256: vocab_digest(&v)?,
            added_tokens: added,
        })
    }
}

/// sha256 of the canonical (key-sorted, compact) JSON of a tokenizer's `model` object.
pub fn vocab_digest(tokenizer: &Value) -> Result<String, String> {
    let model = tokenizer
        .get("model")
        .ok_or("tokenizer.json: no `model` object")?;
    let mut buf = Vec::new();
    canonical(model, &mut buf);
    Ok(format!("{:x}", Sha256::digest(&buf)))
}

fn canonical(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push(b'{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(k).unwrap_or_default());
                out.push(b':');
                canonical(&m[*k], out);
            }
            out.push(b'}');
        }
        Value::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonical(x, out);
            }
            out.push(b']');
        }
        _ => out.extend(serde_json::to_vec(v).unwrap_or_default()),
    }
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let m: Self = serde_json::from_slice(bytes).map_err(|e| format!("{FILE}: {e}"))?;
        m.validate()?;
        Ok(m)
    }

    /// Self-consistency, independent of any target.
    pub fn validate(&self) -> Result<(), String> {
        if self.contract != CONTRACT {
            return Err(format!(
                "{FILE}: contract {:?}, this runtime reads {CONTRACT:?}",
                self.contract
            ));
        }
        if self.n_cu == 0 || self.tensors.is_empty() {
            return Err(format!("{FILE}: empty grid or tensor table"));
        }
        let mut names: Vec<&str> = self.tensors.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return Err(format!("{FILE}: duplicate tensor name"));
        }
        if self.tensors.len() > usize::from(u16::MAX) {
            return Err(format!("{FILE}: tensor table exceeds the u16 handle range"));
        }
        for t in &self.tensors {
            if t.bytes == 0 {
                return Err(format!("{FILE}: tensor {} has no bytes", t.name));
            }
            if let Source::Init { len, .. } = t.source {
                if len != t.bytes {
                    return Err(format!("{FILE}: init of {} is {len} bytes", t.name));
                }
            }
        }
        if (self.kind == Kind::BlockOwnKv) != self.draft.block.is_some() {
            return Err(format!(
                "{FILE}: kind {:?} vs draft program shape",
                self.kind
            ));
        }
        let r = &self.requires;
        let pins = [&r.target_packet_sha256, &r.checkpoint_sha256];
        if pins
            .iter()
            .any(|p| p.as_deref().is_some_and(|s| !is_sha256(s)))
        {
            return Err(format!(
                "{FILE}: requires: a sha256 pin is not 64 lowercase hex"
            ));
        }
        if r.tokenizer.is_none() {
            return Err(format!("{FILE}: requires.tokenizer is mandatory"));
        }
        // Handles are checked against the drafter table, not the target's.
        let programs = self.draft.programs.iter().map(|p| p.index).max();
        self.draft
            .validate(programs.map_or(0, |i| i as usize + 1), self.tensors.len())
    }

    /// Identity of this drafter for cache namespaces and logs: sha256 of the manifest's
    /// canonical JSON (which pins programs only through the caller adding their digest).
    pub fn digest(&self, programs_sha256: &str) -> String {
        let v = serde_json::to_value(self).unwrap_or(Value::Null);
        let mut buf = Vec::new();
        canonical(&v, &mut buf);
        buf.extend(programs_sha256.as_bytes());
        format!("{:x}", Sha256::digest(&buf))
    }

    /// Check the drafter against what the target proves. The error names the first mismatch.
    pub fn check(&self, t: &TargetView) -> Result<(), String> {
        if self.kind != Kind::ChainSharedKv {
            return Err(format!(
                "kind {:?}: this runtime attaches only chain_shared_kv drafters externally",
                self.kind
            ));
        }
        let r = &self.requires;
        let fail = |field: &str, want: &dyn std::fmt::Debug, have: &dyn std::fmt::Debug| {
            Err(format!(
                "drafter requires {field} = {want:?}, the target has {have:?}"
            ))
        };
        if r.target_packet_sha256.is_none() {
            if t.model.is_none() {
                return Err(
                    "the target packet has no spec.target.json, so the drafter must pin \
                     requires.target_packet_sha256"
                        .into(),
                );
            }
            if r.model.is_none() || r.revision.is_none() || r.checkpoint_sha256.is_none() {
                return Err(
                    "a drafter without requires.target_packet_sha256 must pin model, revision \
                     and checkpoint_sha256"
                        .into(),
                );
            }
        }
        if let Some(want) = &r.target_packet_sha256 {
            if *want != t.packet_sha256 {
                return fail("target_packet_sha256", want, &t.packet_sha256);
            }
        }
        for (field, want, have) in [
            ("model", &r.model, &t.model),
            ("revision", &r.revision, &t.revision),
            (
                "checkpoint_sha256",
                &r.checkpoint_sha256,
                &t.checkpoint_sha256,
            ),
        ] {
            if let Some(want) = want {
                if have.as_ref() != Some(want) {
                    return fail(field, want, have);
                }
            }
        }
        if let Some(tok) = &r.tokenizer {
            let Some(have) = &t.tokenizer else {
                return fail("tokenizer", &tok.vocab_sha256, &"no tokenizer.json");
            };
            if tok.vocab_sha256 != have.vocab_sha256 {
                return fail(
                    "tokenizer.vocab_sha256",
                    &tok.vocab_sha256,
                    &have.vocab_sha256,
                );
            }
            for (id, content) in &tok.added_tokens {
                if have.added_tokens.get(id) != Some(content) {
                    return fail(
                        &format!("tokenizer.added_tokens[{id}]"),
                        content,
                        &have.added_tokens.get(id),
                    );
                }
            }
        }
        for (key, want) in &r.geometry {
            if t.geometry.get(key) != Some(want) {
                return fail(&format!("geometry.{key}"), want, &t.geometry.get(key));
            }
        }
        for tap in &r.taps {
            if !t.taps.contains(tap) {
                return fail("taps", &r.taps, &t.taps);
            }
        }
        for (sym, want) in &r.object_symbols {
            if t.object_symbols.get(sym) != Some(want) {
                return fail(
                    &format!("object symbol {sym}"),
                    want,
                    &t.object_symbols.get(sym),
                );
            }
        }
        for d in &self.tensors {
            if let Source::Import { .. } = d.source {
                match t.tensors.iter().find(|(n, _)| *n == d.name) {
                    Some((_, b)) if *b == d.bytes => {}
                    have => {
                        return fail(
                            &format!("import {}", d.name),
                            &d.bytes,
                            &have.map(|(_, b)| b),
                        )
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "spec_drafter_tests.rs"]
mod tests;
