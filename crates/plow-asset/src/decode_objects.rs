use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SECTION: &str = "decode_objects.json";
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeObjects {
    pub version: u32,
    pub kernarg_bytes: usize,
    #[serde(deserialize_with = "unique_objects")]
    pub objects: BTreeMap<u32, DecodeObject>,
    pub programs: Vec<DecodeProgramObject>,
}
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeObject {
    pub file: String,
    pub sha256: String,
    pub profile: String,
    pub entry: String,
    pub threads: u32,
    pub arena_bytes: u32,
    pub grid: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeProgramObject {
    pub index: usize,
    pub rows: u32,
    pub object: u32,
}
fn unique_objects<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<u32, DecodeObject>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<u32, DecodeObject>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique numeric decode object IDs")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut objects = BTreeMap::new();
            while let Some((id, object)) = map.next_entry::<u32, DecodeObject>()? {
                if objects.insert(id, object).is_some() {
                    return Err(serde::de::Error::custom("duplicate decode object ID"));
                }
            }
            Ok(objects)
        }
    }
    deserializer.deserialize_map(Unique)
}
pub fn image_sha256(image: &[u8]) -> String {
    format!("{:x}", Sha256::digest(image))
}

impl DecodeObject {
    pub fn validate(&self, grid: u32) -> Result<(), String> {
        let mut components = std::path::Path::new(&self.file).components();
        if !matches!(components.next(), Some(std::path::Component::Normal(_)))
            || components.next().is_some()
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || self.profile.is_empty()
            || self.entry.is_empty()
            || self.entry.as_bytes().contains(&0)
            || self.threads == 0
            || self.threads > 1024
            || self.threads % 32 != 0
            || self.grid != grid
            || self.arena_bytes == 0
            || self.arena_bytes > i32::MAX as u32
        {
            return Err("invalid decode object identity or resource declaration".into());
        }
        Ok(())
    }
    pub fn matches_image(&self, image: &[u8]) -> bool {
        self.sha256 == image_sha256(image)
    }
}
impl DecodeObjects {
    pub fn validate(
        &self,
        programs: &[(usize, u32)],
        grid: u32,
        kernarg_bytes: usize,
    ) -> Result<(), String> {
        if self.version != 1
            || self.kernarg_bytes != kernarg_bytes
            || programs.is_empty()
            || self.programs.len() != programs.len()
            || self.objects.is_empty()
            || grid == 0
        {
            return Err("unsupported decode object metadata or program coverage".into());
        }
        if programs
            .windows(2)
            .any(|w| w[0].0 >= w[1].0 || w[0].1 >= w[1].1)
        {
            return Err("decode programs must be strictly increasing".into());
        }
        let mut used = BTreeSet::new();
        for (binding, &(index, rows)) in self.programs.iter().zip(programs) {
            if binding.index != index
                || binding.rows != rows
                || !self.objects.contains_key(&binding.object)
            {
                return Err("decode object binding does not cover exact programs".into());
            }
            used.insert(binding.object);
        }
        if used.len() != self.objects.len() {
            return Err("unused decode object".into());
        }
        for object in self.objects.values() {
            object.validate(grid)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> DecodeObjects {
        DecodeObjects {
            version: 1,
            kernarg_bytes: 192,
            objects: BTreeMap::from([(
                0,
                DecodeObject {
                    file: "old.cubin".into(),
                    sha256: "a".repeat(64),
                    profile: "sm90a".into(),
                    entry: "_Z12interp_sm90a11PlowProgram".into(),
                    threads: 256,
                    arena_bytes: 16448,
                    grid: 132,
                },
            )]),
            programs: vec![
                DecodeProgramObject {
                    index: 3,
                    rows: 1,
                    object: 0,
                },
                DecodeProgramObject {
                    index: 4,
                    rows: 2,
                    object: 0,
                },
            ],
        }
    }
    fn valid(m: &DecodeObjects) -> bool {
        m.validate(&[(3, 1), (4, 2)], 132, 192).is_ok()
    }
    #[test]
    fn rejects_malformed_metadata_and_partial_coverage() {
        let base = fixture();
        assert!(valid(&base));
        let mut cases = vec![];
        let mut m = base.clone();
        m.version = 2;
        cases.push(m);
        let mut m = base.clone();
        m.kernarg_bytes += 8;
        cases.push(m);
        let mut m = base.clone();
        m.programs.pop();
        cases.push(m);
        let mut m = base.clone();
        m.programs.swap(0, 1);
        cases.push(m);
        let mut m = base.clone();
        m.programs[0].index = 0;
        cases.push(m);
        let mut m = base.clone();
        m.programs[0].rows = 2;
        cases.push(m);
        let mut m = base.clone();
        m.programs[1].object = 4;
        cases.push(m);
        let mut m = base.clone();
        m.objects.insert(1, m.objects[&0].clone());
        cases.push(m);
        for m in cases {
            assert!(!valid(&m), "accepted {m:?}");
        }
        assert!(base.validate(&[(3, 2), (4, 2)], 132, 192).is_err());
        let raw = serde_json::to_value(base).unwrap();
        let mut bad = raw;
        bad["objects"]["0"]["unknown"] = serde_json::json!(1);
        assert!(serde_json::from_value::<DecodeObjects>(bad).is_err());
    }
    #[test]
    fn accepts_one_complete_decode_program() {
        let mut metadata = fixture();
        metadata.programs.truncate(1);
        assert!(metadata.validate(&[(3, 1)], 132, 192).is_ok());
        assert!(metadata.validate(&[], 132, 192).is_err());
    }
    #[test]
    fn rejects_object_identity_resource_and_path_mutations() {
        let base = fixture();
        for file in ["", "../old.cubin", "/old.cubin", "dir/old.cubin"] {
            let mut m = base.clone();
            m.objects.get_mut(&0).unwrap().file = file.into();
            assert!(!valid(&m));
        }
        for (key, value) in [
            ("profile", serde_json::json!("")),
            ("entry", serde_json::json!("")),
            ("threads", serde_json::json!(33)),
            ("grid", serde_json::json!(264)),
            ("arena_bytes", serde_json::json!(0)),
            ("sha256", serde_json::json!("bad")),
        ] {
            let mut m = serde_json::to_value(&base).unwrap();
            m["objects"]["0"][key] = value;
            assert!(!valid(&serde_json::from_value(m).unwrap()));
        }
    }
    #[test]
    fn raw_json_rejects_duplicate_object_ids_and_numeric_aliases() {
        let base = fixture();
        let object = serde_json::to_string(&base.objects[&0]).unwrap();
        let programs = serde_json::to_string(&base.programs).unwrap();
        for duplicate in ["0", "00"] {
            let raw = format!(
                "{{\"version\":1,\"kernarg_bytes\":192,\"objects\":{{\"0\":{object},\"{duplicate}\":{object}}},\"programs\":{programs}}}"
            );
            assert!(
                serde_json::from_str::<DecodeObjects>(&raw).is_err(),
                "duplicate key {duplicate} accepted"
            );
        }
        assert!(
            serde_json::from_str::<DecodeObjects>(&serde_json::to_string(&base).unwrap()).is_ok()
        );
    }
    #[test]
    fn binds_exact_object_content() {
        let mut object = fixture().objects.remove(&0).unwrap();
        object.sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into();
        assert!(object.matches_image(b"abc"));
        assert!(!object.matches_image(b"abd"));
    }
}
