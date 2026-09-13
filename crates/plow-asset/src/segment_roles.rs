use std::collections::{BTreeMap, BTreeSet};

pub const SECTION: &str = "segment_roles.json";
pub const INTERPRETER: u8 = 0;
pub const FP8_PREFILL_GEMM: u8 = 1;
pub const PREFILL_ATTENTION: u8 = 2;
pub const GEMV_CTA512: u8 = 3;
pub const FP8_M1: u8 = 4;
pub const CUBLASLT: u8 = 5;
pub const PREFILL_ATTENTION_HD512_WG32: u8 = 6;
pub const MXFP4_MOE: u8 = 7;
pub const NATIVE_DECODE_TC: u8 = 8;
pub const W8A16_PREFILL_M1: u8 = 9;
pub const PREFILL_ATTENTION_HD256_BKV64: u8 = 10;
pub const PREFILL_ATTENTION_HD256_BKV32: u8 = 11;
pub const BF16_PREFILL_GEMM_GLU_GEMMA4: u8 = 12;
pub const W8A8_PREFILL_GEMM_GLU_GEMMA4: u8 = 13;
pub const PREFILL_ATTENTION_HD256_GQA2_BKV32: u8 = 14;
pub const MAX_ROLE: u8 = PREFILL_ATTENTION_HD256_GQA2_BKV32;

pub fn is_projection(role: u8) -> bool {
    matches!(role, CUBLASLT | NATIVE_DECODE_TC)
}

pub const CUBLASLT_PREFILL_MAX_ROWS: u32 = 8192;
pub const CUBLASLT_PREFILL_ROWS: [u32; 3] = [128, 256, 512];
pub const CUBLASLT_PREFILL_WIDE_ROWS: [u32; 4] = [1024, 2048, 4096, 8192];
pub const CUBLASLT_PREFILL_GEMMA4_SHAPES: [(u32, u32); 8] = [
    (15360, 3840),
    (2048, 3840),
    (3840, 15360),
    (4096, 3840),
    (3840, 4096),
    (8192, 3840),
    (512, 3840),
    (3840, 8192),
];

pub fn cublaslt_prefill_bf16(profile: &str, m: u32, n: u32, k: u32) -> bool {
    matches!(profile, "sm90a" | "sm_90a")
        && ((CUBLASLT_PREFILL_ROWS.contains(&m)
            && matches!((n, k), (3840, 15360) | (3840, 8192)))
            || (CUBLASLT_PREFILL_WIDE_ROWS.contains(&m)
                && CUBLASLT_PREFILL_GEMMA4_SHAPES.contains(&(n, k))))
}

pub const PREFILL_ATTENTION_HD512_WG32_ABI: &str = "attention_sm90_hd512_wg32_v1";
pub const PREFILL_ATTENTION_HD256_BKV64_ABI: &str = "attention_sm90_hd256_bkv64_v1";
pub const PREFILL_ATTENTION_HD256_BKV32_ABI: &str = "attention_sm90_hd256_bkv32_v1";
pub const PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI: &str =
    "attention_sm90_hd256_gqa2_bkv32_v1";
pub const MXFP4_MOE_ABI: &str = "mxfp4_moe_sm90_v1";
pub const W8A16_PREFILL_M1_ABI: &str = "w8a16_prefill_m1_sm90_v1";
pub const BF16_PREFILL_GEMM_GLU_GEMMA4_ABI: &str = "gemm_glu_sm90_gemma4_4k8k_v1";
pub const W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI: &str =
    "gemm_glu_w8a8_sm90_gemma4_4k8k_v1";

pub fn requires_object(role: u8) -> bool {
    matches!(
        role,
        FP8_PREFILL_GEMM
            | PREFILL_ATTENTION
            | GEMV_CTA512
            | FP8_M1
            | PREFILL_ATTENTION_HD512_WG32
            | MXFP4_MOE
            | NATIVE_DECODE_TC
            | W8A16_PREFILL_M1
            | PREFILL_ATTENTION_HD256_BKV64
            | PREFILL_ATTENTION_HD256_BKV32
            | BF16_PREFILL_GEMM_GLU_GEMMA4
            | W8A8_PREFILL_GEMM_GLU_GEMMA4
            | PREFILL_ATTENTION_HD256_GQA2_BKV32
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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
                MXFP4_MOE => MXFP4_MOE_ABI,
                NATIVE_DECODE_TC => "gemv_transposed_sm90_bf16_v1",
                W8A16_PREFILL_M1 => W8A16_PREFILL_M1_ABI,
                PREFILL_ATTENTION_HD256_BKV64 => PREFILL_ATTENTION_HD256_BKV64_ABI,
                PREFILL_ATTENTION_HD256_BKV32 => PREFILL_ATTENTION_HD256_BKV32_ABI,
                BF16_PREFILL_GEMM_GLU_GEMMA4 => BF16_PREFILL_GEMM_GLU_GEMMA4_ABI,
                W8A8_PREFILL_GEMM_GLU_GEMMA4 => W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI,
                PREFILL_ATTENTION_HD256_GQA2_BKV32 => {
                    PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI
                }
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
            let hd512_wg16 = AttentionCapability {
                kv_tile: 16,
                ..hd512_wg.clone()
            };
            let hd512_wg64 = AttentionCapability {
                kv_tile: 64,
                ..hd512_wg.clone()
            };
            let hd256_bkv64 = AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: 256,
                query_tile: 64,
                kv_tile: 64,
                warps: 8,
            };
            let hd256_bkv32 = AttentionCapability {
                kv_tile: 32,
                ..hd256_bkv64.clone()
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
                            .is_none_or(|a| {
                                a != &hd512_wg
                                    && a != &hd512_wg16
                                    && a != &hd512_wg64
                                    && a != &hd512_px4
                            })))
                || (id == PREFILL_ATTENTION_HD256_BKV64
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.as_ref() != Some(&hd256_bkv64)))
                || (id == PREFILL_ATTENTION_HD256_BKV32
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.as_ref() != Some(&hd256_bkv32)))
                || (id == PREFILL_ATTENTION_HD256_GQA2_BKV32
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.as_ref() != Some(&hd256_bkv32)))
                || (matches!(
                    id,
                    MXFP4_MOE
                        | NATIVE_DECODE_TC
                        | W8A16_PREFILL_M1
                        | BF16_PREFILL_GEMM_GLU_GEMMA4
                        | W8A8_PREFILL_GEMM_GLU_GEMMA4
                )
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.is_some()))
                || (!matches!(
                    id,
                    FP8_M1
                        | PREFILL_ATTENTION_HD512_WG32
                        | MXFP4_MOE
                        | NATIVE_DECODE_TC
                        | W8A16_PREFILL_M1
                        | PREFILL_ATTENTION_HD256_BKV64
                        | PREFILL_ATTENTION_HD256_BKV32
                        | BF16_PREFILL_GEMM_GLU_GEMMA4
                        | W8A8_PREFILL_GEMM_GLU_GEMMA4
                        | PREFILL_ATTENTION_HD256_GQA2_BKV32
                ) && (object.sha256.is_some()
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
                || (program.roles.contains(&CUBLASLT) && program.roles.contains(&NATIVE_DECODE_TC))
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
    fn cublaslt_prefill_policy_is_exactly_the_measured_sm90_bf16_cells() {
        for profile in ["sm90a", "sm_90a"] {
            for m in CUBLASLT_PREFILL_ROWS {
                assert!(cublaslt_prefill_bf16(profile, m, 3840, 15360));
                assert!(cublaslt_prefill_bf16(profile, m, 3840, 8192));
            }
            for m in CUBLASLT_PREFILL_WIDE_ROWS {
                for (n, k) in CUBLASLT_PREFILL_GEMMA4_SHAPES {
                    assert!(cublaslt_prefill_bf16(profile, m, n, k));
                }
            }
        }
        for (profile, m, n, k) in [
            ("sm120", 128, 3840, 15360),
            ("gfx942", 128, 3840, 8192),
            ("sm90a", 0, 3840, 15360),
            ("sm90a", 64, 3840, 15360),
            ("sm90a", 1024, 3840, 3840),
            ("sm90a", 1024, 15360, 8192),
            ("sm90a", 16384, 3840, 15360),
            ("sm90a", 128, 15360, 3840),
            ("sm90a", 128, 3840, 4096),
        ] {
            assert!(!cublaslt_prefill_bf16(profile, m, n, k));
        }
    }

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
        SegmentRoles::from_bytes(raw.replace("\"kv_tile\":32", "\"kv_tile\":64").as_bytes())
            .unwrap();
        SegmentRoles::from_bytes(
            raw.replace(
                "\"query_tile\":64,\"kv_tile\":32",
                "\"query_tile\":32,\"kv_tile\":16",
            )
            .as_bytes(),
        )
        .unwrap();
        SegmentRoles::from_bytes(raw.replace("\"kv_tile\":32", "\"kv_tile\":16").as_bytes())
            .unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":512", "\"head_dim\":256"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":128"),
            raw.replace("\"warps\":8", "\"warps\":4"),
            raw.replace("\"profile\":\"sm90a\"", "\"profile\":\"sm120\""),
            raw.replace("\"dtype\":\"bf16\"", "\"dtype\":\"fp8\""),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn hd256_bkv64_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"10":{{"abi":"attention_sm90_hd256_bkv64_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":64,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,10,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":256", "\"head_dim\":512"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":64", "\"kv_tile\":32"),
            raw.replace("\"warps\":8", "\"warps\":4"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn hd256_bkv32_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"11":{{"abi":"attention_sm90_hd256_bkv32_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":32,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,11,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":256", "\"head_dim\":512"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":64"),
            raw.replace("\"warps\":8", "\"warps\":4"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn hd256_gqa2_bkv32_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"14":{{"abi":"attention_sm90_hd256_gqa2_bkv32_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":32,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,14,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace(
                PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI,
                PREFILL_ATTENTION_HD256_BKV32_ABI,
            ),
            raw.replace("\"head_dim\":256", "\"head_dim\":512"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":64"),
            raw.replace("\"warps\":8", "\"warps\":4"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn mxfp4_moe_role_requires_exact_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"7":{{"abi":"mxfp4_moe_sm90_v1","file":"moe.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,7,7,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&format!(r#","sha256":"{}""#, "a".repeat(64)), ""),
            raw.replace(&"a".repeat(64), "bad"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn native_decode_requires_hash_and_one_projection_backend() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"8":{{"abi":"gemv_transposed_sm90_bf16_v1","file":"native.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,8,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("[0,8,0]", "[5,8,0]"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn native_w8a16_m1_requires_exact_abi_and_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"9":{{"abi":"w8a16_prefill_m1_sm90_v1","file":"m1.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,9,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("w8a16_prefill_m1_sm90_v1", "w8a16_prefill_small_sm90_v1"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn gemma4_bf16_gemm_glu_requires_exact_abi_and_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"12":{{"abi":"gemm_glu_sm90_gemma4_4k8k_v1","file":"glu.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,12,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace(
                BF16_PREFILL_GEMM_GLU_GEMMA4_ABI,
                "gemm_glu_sm90_gemma4_v0",
            ),
            raw.replace("glu.cubin", "../glu.cubin"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn gemma4_w8a8_gemm_glu_requires_exact_abi_and_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"13":{{"abi":"gemm_glu_w8a8_sm90_gemma4_4k8k_v1","file":"glu.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,13,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace(
                W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI,
                "gemm_glu_w8a8_sm90_gemma4_v0",
            ),
            raw.replace("glu.cubin", "../glu.cubin"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }
}
