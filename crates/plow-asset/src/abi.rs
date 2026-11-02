//! `abi.json`: what a packet requires of the runtime that loads it, carried in the packet so a
//! missing or stale `build.json` cannot drop it. Every block is optional; a packet without the
//! section loads exactly as before.
//!
//! * `contract` — the minimum runtime contract ([`crate::RUNTIME_CONTRACT`]).
//! * `must_understand` — metadata sections in the packet that a runtime must implement to run
//!   it correctly ([`MUST_UNDERSTAND`] at the writer's commit). A reader refuses a name it does
//!   not know, or one the packet does not carry.
//! * `target` — the instruction set the programs were compiled for. A reader refuses only a real
//!   instruction-set mismatch; the product the packet was tuned on, its memory and its L2 size
//!   never refuse it (the CU count is checked by each engine against the packet header).
//! * `bindings` — the runtime role of each fixed binding tensor and every KV-cache write site the
//!   write-row rule ([`kv_write_row_field`]) finds.
//!   A reader checks them against the tensor names they are derived from today.
use std::collections::BTreeMap;

use packet::dev::DevOp;
use serde::{Deserialize, Serialize};

pub const SECTION: &str = "abi.json";
pub const VERSION: u32 = 1;

/// Metadata sections whose semantics a runtime must implement to run a packet carrying them.
/// Append a section here when a runtime that skipped it would run the packet wrongly.
pub const MUST_UNDERSTAND: &[&str] = &[
    crate::live_kv::SECTION,
    crate::kv_shared_tail::SECTION,
    crate::recurrent_state::SECTION,
    crate::derived_weights::SECTION,
    crate::object_requirements::SECTION,
    crate::segment_roles::SECTION,
    crate::decode_objects::SECTION,
    crate::weight_shards::SECTION,
];

/// Runtime roles bound by fixed tensor name: (role, tensor name).
pub const ROLES: &[(&str, &str)] = &[
    ("ids", "in.ids"),
    ("pos", "in.pos"),
    ("kvlen", "in.kvlen"),
    ("pos_base", "in.pos_base"),
    ("active", "in.active"),
    ("parked", "in.parked"),
    ("values", "in.values"),
    ("encoder_overlay", "in.encoder_overlay"),
    ("encoder_overlay_index", "in.encoder_overlay_index"),
    ("cos", "in.cos"),
    ("sin", "in.sin"),
    ("x", "act.x"),
    ("logits", "act.logits"),
];

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Abi {
    pub version: u32,
    pub contract: u32,
    #[serde(default)]
    pub must_understand: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<Target>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bindings: Option<Bindings>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    /// `sm_90a`, `sm_89`, `gfx942`, ...: the instruction set of the objects the programs need.
    pub isa: String,
    /// The CU count the programs' streams are laid out for (equals the header's `n_cu`).
    pub n_cu: u32,
    /// The device spec the packet was tuned on; informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tuned_on: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bindings {
    /// Role ([`ROLES`]) -> tensor handle, for every role the packet declares.
    pub roles: BTreeMap<String, u16>,
    /// Every KV-cache write site [`kv_write_row_field`] finds: `[program, instruction, i-field
    /// holding the write row]`.
    pub kv_writes: Vec<[u32; 3]>,
}

impl Bindings {
    /// The bindings a packet's tensor names and programs imply; each program is given as its
    /// instructions' `(opcode, t[0] handle)`.
    pub fn derive<P, I>(names: &[String], programs: P) -> Bindings
    where
        P: IntoIterator<Item = I>,
        I: IntoIterator<Item = (u16, u32)>,
    {
        let roles = ROLES
            .iter()
            .filter_map(|(role, name)| {
                let h = names.iter().position(|n| n == name)?;
                Some((role.to_string(), u16::try_from(h).ok()?))
            })
            .collect();
        let mut kv_writes = Vec::new();
        for (p, insts) in programs.into_iter().enumerate() {
            for (i, (op, t0)) in insts.into_iter().enumerate() {
                if let Some(f) = kv_write_row_field(op, names.get(t0 as usize)) {
                    kv_writes.push([p as u32, i as u32, f as u32]);
                }
            }
        }
        Bindings { roles, kv_writes }
    }
}

impl Abi {
    /// The checks that need only the section and the packet's section names.
    pub fn validate(&self, n_cu: u32, sections: &[&str]) -> Result<(), String> {
        if self.version > VERSION {
            return Err(format!(
                "{SECTION} version {} is newer than this runtime reads ({VERSION})",
                self.version
            ));
        }
        if self.contract > crate::RUNTIME_CONTRACT {
            return Err(format!(
                "the packet requires runtime contract {} and this runtime implements {}",
                self.contract,
                crate::RUNTIME_CONTRACT
            ));
        }
        for name in &self.must_understand {
            if !MUST_UNDERSTAND.contains(&name.as_str()) {
                return Err(format!(
                    "the packet requires section `{name}`, which this runtime does not implement"
                ));
            }
            if !sections.contains(&name.as_str()) {
                return Err(format!(
                    "the packet lists section `{name}` as required but does not carry it"
                ));
            }
        }
        if let Some(t) = &self.target {
            if t.n_cu != n_cu {
                return Err(format!(
                    "target n_cu {} disagrees with the header's {n_cu}",
                    t.n_cu
                ));
            }
        }
        Ok(())
    }
}

impl Target {
    /// Refuses a CUDA device the packet's objects cannot run on; otherwise a note when the
    /// packet was tuned on a different device (never a refusal).
    pub fn check_cuda(&self, cc: (u32, u32), device: &str) -> Result<Option<String>, String> {
        cuda_isa_runs_on(&self.isa, cc)?;
        Ok(self.tuned_note(device))
    }

    pub fn check_hsa(&self, arch: &str) -> Result<(), String> {
        hsa_isa_runs_on(&self.isa, arch)
    }

    fn tuned_note(&self, device: &str) -> Option<String> {
        self.tuned_on
            .as_ref()
            .filter(|t| !device.contains(t.as_str()))
            .map(|t| format!("packet tuned on {t}, running on {device}"))
    }
}

/// Whether objects compiled for `isa` run on a CUDA device of compute capability `cc`. An
/// architecture-specific target (`sm_90a`) runs only on its exact capability; a family target
/// (`sm_100f`) on its major with an equal or higher minor; a plain one (`sm_89`) likewise.
pub fn cuda_isa_runs_on(isa: &str, cc: (u32, u32)) -> Result<(), String> {
    let refuse = || {
        format!(
            "the packet was compiled for {isa}, which does not run on compute capability {}.{}",
            cc.0, cc.1
        )
    };
    let digits = isa.strip_prefix("sm_").ok_or_else(refuse)?;
    let (num, exact) = match digits.as_bytes().last() {
        Some(b'a') => (&digits[..digits.len() - 1], true),
        Some(b'f') => (&digits[..digits.len() - 1], false),
        _ => (digits, false),
    };
    if num.len() < 2 || !num.bytes().all(|b| b.is_ascii_digit()) {
        return Err(refuse());
    }
    let major: u32 = num[..num.len() - 1].parse().map_err(|_| refuse())?;
    let minor: u32 = num[num.len() - 1..].parse().map_err(|_| refuse())?;
    let ok = major == cc.0 && if exact { minor == cc.1 } else { minor <= cc.1 };
    ok.then_some(()).ok_or_else(refuse)
}

/// Whether objects compiled for `isa` run on an HSA agent named `arch` (`gfx942`, with optional
/// `:feature+`/`:feature-` suffixes): same processor, and no feature set one way on one side and
/// the other way on the other.
pub fn hsa_isa_runs_on(isa: &str, arch: &str) -> Result<(), String> {
    let features = |s: &'_ str| -> Vec<(String, bool)> {
        s.split(':')
            .skip(1)
            .map(|f| (f.trim_end_matches(['+', '-']).to_string(), f.ends_with('+')))
            .collect()
    };
    let (fa, fb) = (features(isa), features(arch));
    let conflict = fa
        .iter()
        .any(|(n, on)| fb.iter().any(|(m, other)| n == m && on != other));
    let same = isa.split(':').next() == arch.split(':').next();
    (same && !conflict)
        .then_some(())
        .ok_or_else(|| format!("the packet was compiled for {isa}, which does not run on {arch}"))
}

/// Which `i[]` field of an instruction carries the KV-cache WRITE ROW, or `None`
/// when the instruction does not write the cache at all.
///
/// ONE rule, consulted by BOTH phases — the runtime's `derive_kvrow` for the decode step's
/// per-token row and `rebase_chunk` for a prefill chunk's base row. They used
/// to disagree: decode found the sites by destination NAME, prefill by
/// `HeadNormRope` + `fj[1] != 0`, and the second test matches nothing on GLM's
/// MLA (its k_rope sets `j[1]`, which packs into `fj[2]`, and leaves `f[1]`/`j[0]`
/// — the two halves of `fj[1]` — at zero). So a GLM prefill chunk wrote every
/// latent row of every chunk at row 0 and only the LAST chunk's tail survived,
/// with no error anywhere. Two phases deriving "what is a KV write" by two rules
/// is the bug; this is the one rule.
///
/// The destination NAME is the discriminator, and it has to be: the field alone
/// cannot tell GLM's `kv_a_layernorm` (an `RmsNorm` whose `i[2]` is the latent
/// out-row) from the input/post-attention norms, which are the same opcode with
/// `i[2]` meaning nothing. Keying on the opcode alone would rebase those two and
/// corrupt the block input.
///
/// The `HeadNormRopeFp8` twin is included for the same reason the flash test
/// includes its own: an fp8-KV packet emits the fp8 opcode, and a bf16-only test
/// silently matches nothing there — the class of miss `derive_segments` records.
pub fn kv_write_row_field(op: u16, dst: Option<&String>) -> Option<usize> {
    if !dst.is_some_and(|n| n.starts_with("kv.")) {
        return None;
    }
    if op == DevOp::RmsNorm as u16 {
        // GLM/DeepSeek MLA: `kv_a_layernorm` -> `kv.{L}.ckv`, out_row0 in i[2]
        // (`runtime/amd/interp.hip`, PLOW_DOP_RMSNORM).
        Some(2)
    } else if op == DevOp::DsaPoolCompress as u16 {
        // GLM-5.3 pooled indexer: prefill pool-cache base in i[3].
        Some(3)
    } else if op == DevOp::HeadNormRope as u16 || op == DevOp::HeadNormRopeFp8 as u16 {
        // Dense GQA k/v norm -> `kv.{L}.k`/`.v`, and MLA's k_rope -> `kv.{L}.krot`.
        // Both carry the write row in i[3].
        Some(3)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> Target {
        Target {
            isa: "sm_90a".into(),
            n_cu: 132,
            tuned_on: Some("H100 SXM".into()),
        }
    }

    #[test]
    fn an_h100_packet_runs_on_h200_and_other_hopper_parts() {
        let abi = Abi {
            version: VERSION,
            contract: crate::RUNTIME_CONTRACT_BASE,
            target: Some(h100()),
            ..Abi::default()
        };
        // H200 SXM: compute capability 9.0, 132 SMs, 141 GB; memory and L2 are no input.
        abi.validate(132, &[]).unwrap();
        let note = h100().check_cuda((9, 0), "NVIDIA H200").unwrap();
        assert!(note.unwrap().contains("H100 SXM"));
        h100().check_cuda((9, 0), "NVIDIA H100 NVL").unwrap();
    }

    #[test]
    fn only_a_real_instruction_set_mismatch_refuses() {
        assert!(cuda_isa_runs_on("sm_90a", (8, 9)).is_err());
        assert!(cuda_isa_runs_on("sm_90a", (10, 0)).is_err());
        assert!(cuda_isa_runs_on("sm_90a", (12, 0)).is_err());
        assert!(cuda_isa_runs_on("sm_120a", (12, 0)).is_ok());
        assert!(cuda_isa_runs_on("sm_89", (8, 9)).is_ok());
        assert!(cuda_isa_runs_on("sm_86", (8, 9)).is_ok());
        assert!(cuda_isa_runs_on("sm_89", (8, 6)).is_err());
        assert!(cuda_isa_runs_on("sm_89", (9, 0)).is_err());
        assert!(cuda_isa_runs_on("sm_100f", (10, 3)).is_ok());
        assert!(cuda_isa_runs_on("gfx942", (9, 0)).is_err());
        assert!(cuda_isa_runs_on("sm_", (9, 0)).is_err());
        assert!(hsa_isa_runs_on("gfx942", "gfx942:sramecc+:xnack-").is_ok());
        assert!(hsa_isa_runs_on("gfx942", "gfx950").is_err());
        assert!(hsa_isa_runs_on("gfx90a:xnack+", "gfx90a:xnack-").is_err());
        assert!(hsa_isa_runs_on("gfx90a:xnack+", "gfx90a").is_ok());
    }

    #[test]
    fn requirements_refuse_what_this_runtime_cannot_honour() {
        let mut abi = Abi {
            version: VERSION,
            contract: crate::RUNTIME_CONTRACT,
            must_understand: vec![crate::live_kv::SECTION.into()],
            ..Abi::default()
        };
        abi.validate(8, &[crate::live_kv::SECTION]).unwrap();
        assert!(abi.validate(8, &[]).unwrap_err().contains("does not carry"));
        abi.must_understand = vec!["future.json".into()];
        assert!(abi
            .validate(8, &["future.json"])
            .unwrap_err()
            .contains("does not implement"));
        abi.must_understand.clear();
        abi.contract = crate::RUNTIME_CONTRACT + 1;
        assert!(abi
            .validate(8, &[])
            .unwrap_err()
            .contains("runtime contract"));
        abi.contract = crate::RUNTIME_CONTRACT;
        abi.version = VERSION + 1;
        assert!(abi.validate(8, &[]).is_err());
        abi.version = VERSION;
        abi.target = Some(h100());
        assert!(abi.validate(8, &[]).unwrap_err().contains("n_cu"));
    }

    #[test]
    fn bindings_follow_the_names() {
        let names: Vec<String> = ["in.ids", "x", "kv.0.k", "act.logits"]
            .map(String::from)
            .to_vec();
        let (write, other) = (
            (DevOp::HeadNormRope as u16, 2),
            (DevOp::HeadNormRope as u16, 1),
        );
        let b = Bindings::derive(&names, [vec![other, write], vec![write]]);
        assert_eq!(b.roles.get("ids"), Some(&0));
        assert_eq!(b.roles.get("logits"), Some(&3));
        assert_eq!(b.roles.len(), 2);
        assert_eq!(b.kv_writes, vec![[0, 1, 3], [1, 0, 3]]);
    }
}
