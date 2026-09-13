use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "packet_pipeline.json";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacketPipelines {
    pub version: u32,
    pub pipelines: Vec<PacketPipeline>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacketPipeline {
    pub name: String,
    pub driver: String,
    pub programs: BTreeMap<String, u32>,
    pub tensors: BTreeMap<String, PipelineTensor>,
    #[serde(default)]
    pub parameters: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineTensor {
    pub name: String,
    pub dtype: PipelineDType,
    pub shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PipelineDType {
    U8,
    U32,
    F16,
    Bf16,
    F32,
}

impl PipelineDType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U32 | Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

impl PacketPipelines {
    pub fn validate(
        &self,
        program_count: usize,
        tensors: impl Fn(&str) -> Option<u64>,
    ) -> Result<(), String> {
        if self.version != VERSION || self.pipelines.is_empty() || self.pipelines.len() > 64 {
            return Err("invalid packet pipeline version or count".into());
        }
        let mut names = BTreeSet::new();
        for pipeline in &self.pipelines {
            if !identifier(&pipeline.name, 64)
                || !identifier(&pipeline.driver, 128)
                || !names.insert(&pipeline.name)
                || pipeline.programs.is_empty()
                || pipeline.programs.len() > 4096
                || pipeline.tensors.len() > 128
                || pipeline.parameters.len() > 64
            {
                return Err("invalid packet pipeline descriptor".into());
            }
            for (role, &program) in &pipeline.programs {
                if !identifier(role, 64) || program as usize >= program_count {
                    return Err(format!("invalid program role {role:?}"));
                }
            }
            for (role, tensor) in &pipeline.tensors {
                if !identifier(role, 64)
                    || !identifier(&tensor.name, packet::devbuild::NAME_LEN - 1)
                    || tensor.shape.is_empty()
                    || tensor.shape.len() > 8
                    || tensor.shape.contains(&0)
                {
                    return Err(format!("invalid tensor role {role:?}"));
                }
                let expected = tensor
                    .shape
                    .iter()
                    .try_fold(tensor.dtype.bytes(), |bytes, &dim| bytes.checked_mul(dim));
                if expected.is_none() || tensors(&tensor.name) != expected {
                    return Err(format!("tensor role {role:?} has the wrong size"));
                }
            }
            if pipeline.parameters.keys().any(|name| !identifier(name, 64)) {
                return Err("invalid packet pipeline parameter".into());
            }
        }
        Ok(())
    }
}

fn identifier(value: &str, limit: usize) -> bool {
    !value.is_empty()
        && value.len() <= limit
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_programs_and_tensor_geometry() {
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![PacketPipeline {
                name: "transcribe".into(),
                driver: "rnnt.greedy.v1".into(),
                programs: BTreeMap::from([("encoder".into(), 0)]),
                tensors: BTreeMap::from([(
                    "encoder.input".into(),
                    PipelineTensor {
                        name: "in.audio".into(),
                        dtype: PipelineDType::F32,
                        shape: vec![3, 4],
                    },
                )]),
                parameters: BTreeMap::new(),
            }],
        };
        assert!(metadata
            .validate(1, |name| (name == "in.audio").then_some(48))
            .is_ok());
        assert!(metadata.validate(0, |_| Some(48)).is_err());
        assert!(metadata.validate(1, |_| Some(44)).is_err());

        let mut long = metadata;
        long.pipelines[0].programs = (0..65)
            .map(|index| (format!("forward.{index}"), 0))
            .collect();
        assert!(long
            .validate(1, |name| (name == "in.audio").then_some(48))
            .is_ok());

        long.pipelines[0]
            .tensors
            .get_mut("encoder.input")
            .unwrap()
            .shape = vec![u64::MAX, 2];
        assert!(long.validate(1, |_| None).is_err());
    }
}
