use std::collections::{BTreeMap, BTreeSet};

pub const SECTION: &str = "segment_roles.json";

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
            || self.objects.keys().any(|id| !(1..=4).contains(id))
        {
            return Err("unsupported packet segment roles".into());
        }
        for (&id, object) in &self.objects {
            let abi = match id {
                1 => "fp8_gemm_tma128_v1",
                2 => "attention_sm90_hd256_v1",
                3 => "gemv_sm90_cta512_v1",
                _ => crate::fp8_m1_role::ABI,
            };
            if object.abi != abi
                || object.file.is_empty()
                || std::path::Path::new(&object.file)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
                || (id == 4
                    && (object.promote_k512.is_none_or(|v| v > 1)
                        || object.sha256.as_ref().is_none_or(|s| {
                            s.len() != 64
                                || !s
                                    .bytes()
                                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                        })))
                || (id != 4 && (object.sha256.is_some() || object.promote_k512.is_some()))
            {
                return Err("invalid packet segment object".into());
            }
        }
        let mut programs = BTreeSet::new();
        let mut used = BTreeSet::new();
        for program in &self.programs {
            if !programs.insert(program.index)
                || program.roles.is_empty()
                || program.roles.iter().any(|&r| r > 4)
            {
                return Err("invalid packet segment program".into());
            }
            used.extend(program.roles.iter().copied().filter(|&r| r != 0));
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
}
