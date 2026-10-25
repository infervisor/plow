//! OBJECT REQUIREMENTS: the compiled-capability symbols a packet needs from the interpreter
//! objects that run its programs, declared by the compiler from its emission facts.
//!
//! An interpreter object exports `extern "C" __device__ unsigned` symbols that attest compiled
//! arms (`plow_w8a8_arms = 1`, `plow_w8a8_vllm_decode_rows = 128`, ...). The runtime reads every
//! symbol a list names and refuses the load on a missing symbol or a different value; it does not
//! decide from opcodes or tensor names which symbols matter. Every module the packet executes has
//! its own complete list: `prefill` binds the main prefill module, `decode` the main decode
//! module, `objects` each `decode_objects` ladder object by id. A list is explicit, so an empty
//! one states that the object's other contracts (pairing stamp, role ABIs) cover its programs.
//! A packet carrying this section declares `runtime_requires.plowrt_contract >= 2`.
use packet::dev::DevOp;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SECTION: &str = "object_requirements";
pub const VERSION: u32 = 1;

/// Opcodes the NVIDIA interpreter compiles only under `PLOW_NV_W8A8`: their presence makes
/// the packet a W8A8 one (the compiler defines the arms, `W8A8_ARMS` attests them).
pub fn w8a8_only_op(op: u16) -> bool {
    matches!(
        DevOp::from_u16(op),
        Some(DevOp::QuantFp8 | DevOp::MoeGroupGluGemmaPfW8a8)
    )
}

/// The FP8 opcode family a W8A8 object's `W8A8_ARMS` covers. Outside a W8A8 packet the GEMM
/// members run the W8A16 arms, which need no declaration.
pub fn fp8_arm_op(op: u16) -> bool {
    matches!(
        DevOp::from_u16(op),
        Some(
            DevOp::GemmFp8
                | DevOp::GemmMedFp8
                | DevOp::GemmSmallFp8
                | DevOp::GemmGluFp8
                | DevOp::QuantFp8
                | DevOp::MoeGroupGluGemmaPfW8a8
        )
    )
}

/// The gated-delta-net chain's opcode family (`PLOW_NV_QWEN_GDN` arms). `QwenHeadNormRope` is
/// a shared rotary opcode other chains emit and is not part of it.
pub fn gdn_op(op: u16) -> bool {
    matches!(
        DevOp::from_u16(op),
        Some(
            DevOp::QwenGdnConv
                | DevOp::QwenGdnStep
                | DevOp::QwenGatedNorm
                | DevOp::QwenQGateSplit
                | DevOp::QwenSigmoidGate
                | DevOp::QwenRmsNorm
                | DevOp::QwenGdnConvPrefill
                | DevOp::QwenGdnQkvPrep
                | DevOp::QwenGdnGatePrep
                | DevOp::QwenGdnPrefill
        )
    )
}

/// Why a packet without this section is refused: its programs run an opcode family whose arms
/// an object attests only through it. `None` for a packet the older contracts fully cover.
pub fn undeclared_reason(ops: impl IntoIterator<Item = u16>) -> Option<&'static str> {
    let mut reason = None;
    for op in ops {
        if w8a8_only_op(op) {
            return Some("W8A8 opcode arms");
        }
        if gdn_op(op) {
            reason = Some("gated-delta-net opcode arms");
        }
    }
    reason
}

/// Symbols the NVIDIA interpreter exports (runtime/nvidia/interp_sm120.cu), each meaning fixed
/// at the value `1` unless stated.
/// The object compiles the FP8 opcode arms (`GemmFp8`, `QuantFp8`, `GemvFp8`, ...).
pub const W8A8_ARMS: &str = "plow_w8a8_arms";
/// Prefill object: the W8A8 route with per-token activation scales and the weight-scale-first
/// epilogue, on the TMA GEMM.
pub const W8A8_VLLM_PREFILL_ABI: &str = "plow_w8a8_vllm_prefill_abi";
/// Decode object: the W8A8 route's single-row GEMV arms and per-token activation quant.
pub const W8A8_VLLM_DECODE_ABI: &str = "plow_w8a8_vllm_decode_abi";
/// Decode object: rows per launch its W8A8 GEMM arm serves as one M tile (rung 1 on the GEMV
/// arms); the compiler requires the exact ceiling its ladder was built against.
pub const W8A8_VLLM_DECODE_ROWS: &str = "plow_w8a8_vllm_decode_rows";
/// Prefill object: the gated-delta-net prefill opcode arms under the global-queue scheduler.
pub const GDN_PREFILL_ABI: &str = "plow_gdn_prefill_abi";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirement {
    pub symbol: String,
    pub value: u32,
}

impl Requirement {
    pub fn new(symbol: &str, value: u32) -> Self {
        Self {
            symbol: symbol.into(),
            value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub prefill: Vec<Requirement>,
    pub decode: Vec<Requirement>,
    #[serde(deserialize_with = "unique_objects")]
    pub objects: BTreeMap<u32, Vec<Requirement>>,
}

/// A JSON object keeps the last of two equal keys, which would drop a list silently (a
/// non-canonical spelling such as `"03"` is already a malformed number to the parser).
fn unique_objects<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<u32, Vec<Requirement>>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<u32, Vec<Requirement>>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique numeric decode object ids")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut objects = BTreeMap::new();
            while let Some((id, list)) = map.next_entry::<u32, Vec<Requirement>>()? {
                if objects.insert(id, list).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "object requirements list decode object {id} twice"
                    )));
                }
            }
            Ok(objects)
        }
    }
    deserializer.deserialize_map(Unique)
}

fn symbol_ok(symbol: &str) -> bool {
    !symbol.is_empty()
        && symbol.len() <= 96
        && symbol
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && !symbol.as_bytes()[0].is_ascii_digit()
}

fn list_ok(list: &[Requirement]) -> Result<(), String> {
    for (i, r) in list.iter().enumerate() {
        if !symbol_ok(&r.symbol) {
            return Err(format!(
                "object requirement symbol {:?} malformed",
                r.symbol
            ));
        }
        if list[..i].iter().any(|o| o.symbol == r.symbol) {
            return Err(format!("object requirement {} repeated", r.symbol));
        }
    }
    Ok(())
}

impl Manifest {
    pub fn new() -> Self {
        Self {
            version: VERSION,
            prefill: Vec::new(),
            decode: Vec::new(),
            objects: BTreeMap::new(),
        }
    }

    /// `decode_objects`: the ids the packet's `decode_objects` section declares (none without
    /// it). Every id has exactly one list and every list names a declared id.
    pub fn validate(&self, decode_objects: &[u32]) -> Result<(), String> {
        if self.version != VERSION {
            return Err(format!(
                "object requirements version {} unsupported",
                self.version
            ));
        }
        list_ok(&self.prefill)?;
        list_ok(&self.decode)?;
        for (id, list) in &self.objects {
            if !decode_objects.contains(id) {
                return Err(format!(
                    "object requirements name unknown decode object {id}"
                ));
            }
            list_ok(list)?;
        }
        if let Some(id) = decode_objects
            .iter()
            .find(|id| !self.objects.contains_key(id))
        {
            return Err(format!("object requirements omit decode object {id}"));
        }
        Ok(())
    }

    pub fn object(&self, id: u32) -> &[Requirement] {
        self.objects.get(&id).map_or(&[], Vec::as_slice)
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

/// Every requirement in `list` holds on the loaded object: `read` is the module's symbol
/// reader (`None` = absent). `what` names the object in the error.
pub fn check(
    list: &[Requirement],
    mut read: impl FnMut(&str) -> Option<u32>,
    what: &str,
) -> Result<(), String> {
    for r in list {
        match read(&r.symbol) {
            Some(v) if v == r.value => {}
            Some(v) => {
                return Err(format!(
                    "{what}: {} = {v}, packet requires {}",
                    r.symbol, r.value
                ))
            }
            None => {
                return Err(format!(
                    "{what}: {} absent, packet requires {}",
                    r.symbol, r.value
                ))
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader<'a>(table: &'a [(&'a str, u32)]) -> impl FnMut(&str) -> Option<u32> + 'a {
        move |s| table.iter().find(|(n, _)| *n == s).map(|(_, v)| *v)
    }

    #[test]
    fn check_requires_every_symbol_at_its_value() {
        let list = vec![
            Requirement::new(W8A8_VLLM_DECODE_ABI, 1),
            Requirement::new(W8A8_VLLM_DECODE_ROWS, 128),
        ];
        let ok = [
            ("plow_w8a8_vllm_decode_abi", 1),
            ("plow_w8a8_vllm_decode_rows", 128),
        ];
        assert!(check(&list, reader(&ok), "x").is_ok());
        let unequal = [
            ("plow_w8a8_vllm_decode_abi", 1),
            ("plow_w8a8_vllm_decode_rows", 1),
        ];
        assert!(check(&list, reader(&unequal), "decode")
            .unwrap_err()
            .contains("plow_w8a8_vllm_decode_rows = 1"));
        let absent = [("plow_w8a8_vllm_decode_abi", 1)];
        assert!(check(&list, reader(&absent), "decode")
            .unwrap_err()
            .contains("absent"));
        assert!(check(&[], reader(&[]), "x").is_ok());
    }

    #[test]
    fn undeclared_packets_are_named_by_their_arm_family() {
        let ops = |list: &[DevOp]| list.iter().map(|&op| op as u16).collect::<Vec<_>>();
        assert_eq!(undeclared_reason(ops(&[DevOp::Gemm, DevOp::Gemv])), None);
        // W8A16: FP8 weights through the plain GEMM arms.
        assert_eq!(
            undeclared_reason(ops(&[DevOp::GemmFp8, DevOp::GemvFp8])),
            None
        );
        assert_eq!(
            undeclared_reason(ops(&[DevOp::GemmFp8, DevOp::QuantFp8])),
            Some("W8A8 opcode arms")
        );
        assert_eq!(
            undeclared_reason(ops(&[DevOp::QwenRmsNorm, DevOp::Gemm])),
            Some("gated-delta-net opcode arms")
        );
        assert_eq!(
            undeclared_reason(ops(&[DevOp::QwenRmsNorm, DevOp::QuantFp8])),
            Some("W8A8 opcode arms")
        );
    }

    #[test]
    fn manifest_validates_version_symbols_duplicates_and_object_coverage() {
        let mut m = Manifest::new();
        m.prefill.push(Requirement::new(GDN_PREFILL_ABI, 1));
        m.objects.insert(3, vec![Requirement::new(W8A8_ARMS, 1)]);
        assert!(m.validate(&[3]).is_ok());
        assert_eq!(m.object(3).len(), 1);
        assert!(m.object(4).is_empty());
        assert!(m
            .validate(&[3, 4])
            .unwrap_err()
            .contains("omit decode object 4"));
        assert!(m
            .validate(&[])
            .unwrap_err()
            .contains("unknown decode object 3"));
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
        let mut dup = m.clone();
        dup.prefill.push(Requirement::new(GDN_PREFILL_ABI, 0));
        assert!(dup.validate(&[3]).unwrap_err().contains("repeated"));
        let mut bad = m.clone();
        bad.decode.push(Requirement::new("plow bad", 1));
        assert!(bad.validate(&[3]).unwrap_err().contains("malformed"));
        let mut v = m;
        v.version = 2;
        assert!(v.validate(&[3]).unwrap_err().contains("version"));
        // Two lists for one object are malformed, as is a non-canonical id spelling.
        let twice = r#"{"version":1,"prefill":[],"decode":[],"objects":{"3":[{"symbol":"plow_w8a8_arms","value":1}],"3":[]}}"#;
        assert!(
            serde_json::from_str::<Manifest>(&twice.replace(r#""3":[]"#, r#""03":[]"#)).is_err()
        );
        assert!(serde_json::from_str::<Manifest>(twice)
            .unwrap_err()
            .to_string()
            .contains("twice"));
        // Every list is explicit: a section without one is malformed, not empty.
        assert!(
            serde_json::from_str::<Manifest>(r#"{"version":1,"prefill":[],"decode":[]}"#).is_err()
        );
        assert!(serde_json::from_str::<Manifest>(
            r#"{"version":1,"prefill":[],"decode":[],"objects":{},"x":1}"#
        )
        .is_err());
    }
}
