use std::collections::{BTreeMap, BTreeSet};

pub const SECTION: &str = "segment_roles.json";
pub const INTERPRETER: u8 = 0;
pub const FP8_PREFILL_GEMM: u8 = 1;
pub const PREFILL_ATTENTION: u8 = 2;
pub const GEMV_CTA512: u8 = 3;
pub const FP8_M1: u8 = 4;
pub const CUBLASLT: u8 = 5;
pub const PREFILL_ATTENTION_HD512_WG32: u8 = 6;
pub const MAX_ROLE: u8 = PREFILL_ATTENTION_HD512_WG32;

pub const PREFILL_ATTENTION_HD512_WG32_ABI: &str = "attention_sm90_hd512_wg32_v1";

pub fn requires_object(role: u8) -> bool {
    matches!(
        role,
        FP8_PREFILL_GEMM | PREFILL_ATTENTION | GEMV_CTA512 | FP8_M1 | PREFILL_ATTENTION_HD512_WG32
    )
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentRoles {
    pub version: u32,
    #[serde(deserialize_with = "unique_segment_objects")]
    pub objects: std::collections::BTreeMap<u8, SegmentObject>,
    pub programs: Vec<ProgramRoles>,
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentObject {
    pub abi: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promote_k512: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionCapability>,
}
#[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionCapability {
    pub profile: String,
    pub dtype: String,
    pub head_dim: u32,
    pub query_tile: u32,
    pub kv_tile: u32,
    pub warps: u32,
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramRoles {
    pub index: usize,
    pub roles: Vec<u8>,
}

fn unique_segment_objects<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<u8, SegmentObject>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<u8, SegmentObject>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique segment object IDs")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = a.next_entry()? {
                if out.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate segment object ID"));
                }
            }
            Ok(out)
        }
    }
    d.deserialize_map(Unique)
}

impl SegmentRoles {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let value: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        value.validate_schema()?;
        Ok(value)
    }
    pub fn validate_schema(&self) -> Result<(), String> {
        if self.version != 1
            || self.programs.is_empty()
            || self.objects.keys().any(|&id| !requires_object(id))
        {
            return Err("unsupported packet segment roles".into());
        }
        for (&id, object) in &self.objects {
            let abi = match id {
                FP8_PREFILL_GEMM => "fp8_gemm_tma128_v1",
                PREFILL_ATTENTION => "attention_sm90_hd256_v1",
                GEMV_CTA512 => "gemv_sm90_cta512_v1",
                FP8_M1 => crate::fp8_m1_role::ABI,
                PREFILL_ATTENTION_HD512_WG32 => PREFILL_ATTENTION_HD512_WG32_ABI,
                _ => return Err("invalid packet segment object role".into()),
            };
            let valid_hash = |hash: Option<&str>| {
                hash.is_some_and(|s| {
                    s.len() == 64
                        && s.bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
            };
            let hd512_wg = AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: 512,
                query_tile: 64,
                kv_tile: 32,
                warps: 8,
            };
            let hd512_px4 = AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: 512,
                query_tile: 32,
                kv_tile: 16,
                warps: 8,
            };
            if object.abi != abi
                || object.file.is_empty()
                || std::path::Path::new(&object.file)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
                || (id == FP8_M1
                    && (object.promote_k512.is_none_or(|v| v > 1)
                        || !valid_hash(object.sha256.as_deref())
                        || object.attention.is_some()))
                || (id == PREFILL_ATTENTION_HD512_WG32
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object
                            .attention
                            .as_ref()
                            .is_none_or(|a| a != &hd512_wg && a != &hd512_px4)))
                || (!matches!(id, FP8_M1 | PREFILL_ATTENTION_HD512_WG32)
                    && (object.sha256.is_some()
                        || object.promote_k512.is_some()
                        || object.attention.is_some()))
            {
                return Err("invalid packet segment object".into());
            }
        }
        let mut programs = BTreeSet::new();
        let mut used = BTreeSet::new();
        for program in &self.programs {
            if !programs.insert(program.index)
                || program.roles.is_empty()
                || program.roles.iter().any(|&r| r > MAX_ROLE)
            {
                return Err("invalid packet segment program".into());
            }
            used.extend(
                program
                    .roles
                    .iter()
                    .copied()
                    .filter(|&role| requires_object(role)),
            );
        }
        if used != self.objects.keys().copied().collect() {
            return Err("packet segment declarations do not match use".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_rejects_duplicate_alias_ids_and_unknown_fields() {
        let object = r#"{"abi":"fp8_gemm_tma128_v1","file":"role.cubin"}"#;
        let raw = format!(
            r#"{{"version":1,"objects":{{"1":{object}}},"programs":[{{"index":0,"roles":[1]}}]}}"#
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for key in ["1", "01"] {
            let bad = raw.replace(
                &format!(r#""1":{object}"#),
                &format!(r#""1":{object},"{key}":{object}"#),
            );
            let error = SegmentRoles::from_bytes(bad.as_bytes()).unwrap_err();
            if key == "1" {
                assert!(error.contains("duplicate"), "{error}");
            }
        }
        for (needle, replacement) in [
            (r#""version":1"#, r#""version":1,"extra":0"#),
            (r#""abi":"#, r#""extra":0,"abi":"#),
            (r#""index":0"#, r#""index":0,"extra":0"#),
            (r#""version":1"#, r#""version":1,"version":1"#),
        ] {
            assert!(SegmentRoles::from_bytes(raw.replace(needle, replacement).as_bytes()).is_err());
        }
    }

    #[test]
    fn cublaslt_role_is_packet_only() {
        let raw = br#"{"version":1,"objects":{},"programs":[{"index":0,"roles":[0,5]}]}"#;
        SegmentRoles::from_bytes(raw).unwrap();
        for bad in [
            br#"{"version":1,"objects":{"5":{"abi":"x","file":"x"}},"programs":[{"index":0,"roles":[5]}]}"#.as_slice(),
            br#"{"version":1,"objects":{},"programs":[{"index":0,"roles":[6]}]}"#.as_slice(),
        ] {
            assert!(SegmentRoles::from_bytes(bad).is_err());
        }
    }

    #[test]
    fn hd512_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"6":{{"abi":"attention_sm90_hd512_wg32_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":512,"query_tile":64,"kv_tile":32,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,6,6,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        SegmentRoles::from_bytes(
            raw.replace(
                "\"query_tile\":64,\"kv_tile\":32",
                "\"query_tile\":32,\"kv_tile\":16",
            )
            .as_bytes(),
        )
        .unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":512", "\"head_dim\":256"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":16"),
            raw.replace("\"warps\":8", "\"warps\":4"),
            raw.replace("\"profile\":\"sm90a\"", "\"profile\":\"sm120\""),
            raw.replace("\"dtype\":\"bf16\"", "\"dtype\":\"fp8\""),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }
}
