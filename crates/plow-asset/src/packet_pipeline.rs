use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "packet_pipeline.json";
pub const VERSION: u32 = 1;

#[path = "packet_pipeline_handoff.rs"]
mod handoff;
pub use handoff::*;

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
    /// Text the host side of a generic driver needs (prompt templates, output patterns,
    /// marker tokens), so the runtime never names a model.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub strings: BTreeMap<String, String>,
}

impl PacketPipeline {
    pub fn speaker_encoder_plan(&self) -> Result<SpeakerEncoderPlan, String> {
        let exact_rows = match self.driver.as_str() {
            "speaker.encode.v1" => false,
            "speaker.encode.v2" | "speaker.encode.v3" => true,
            _ => {
                return Err(format!(
                    "unsupported speaker encoder driver {:?}",
                    self.driver
                ))
            }
        };
        let parameter = |name: &str| {
            self.parameters
                .get(name)
                .and_then(|&v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("invalid speaker parameter {name}"))
        };
        let min_rows = parameter("input.min_rows")?;
        let max_rows = parameter("input.max_rows")?;
        let input_width = parameter("input.width")?;
        let output_width = parameter("output.width")?;
        let reflect_pad = if self.driver == "speaker.encode.v3" {
            parameter("input.reflect_pad")?
        } else if self.parameters.contains_key("input.reflect_pad") {
            return Err("speaker input reflection requires driver v3".into());
        } else {
            0
        };
        if min_rows > max_rows || max_rows > u32::MAX as usize || reflect_pad >= min_rows {
            return Err("invalid speaker row bounds or reflection".into());
        }
        let input_rows = reflect_pad
            .checked_mul(2)
            .and_then(|pad| max_rows.checked_add(pad))
            .ok_or("speaker input dimensions overflow")?;
        for (role, dtype, shape) in [
            (
                "mel",
                PipelineDType::F32,
                vec![input_rows as u64, input_width as u64],
            ),
            ("length", PipelineDType::U32, vec![1]),
            ("embedding", PipelineDType::F32, vec![output_width as u64]),
        ] {
            if !self
                .tensors
                .get(role)
                .is_some_and(|t| t.dtype == dtype && t.shape == shape)
            {
                return Err(format!("invalid speaker tensor role {role}"));
            }
        }
        let mut capacities = Vec::new();
        let mut rows = BTreeMap::<usize, (usize, usize)>::new();
        for (role, &program) in &self.programs {
            let bad = || format!("invalid speaker program role {role}");
            let suffix = role.strip_prefix("encode.r").ok_or_else(bad)?;
            let (cap, row) = match suffix.split_once(".n") {
                Some((cap, row)) => (cap, Some(row.parse::<usize>().map_err(|_| bad())?)),
                None => (suffix, None),
            };
            let cap = cap.parse::<usize>().map_err(|_| bad())?;
            let canonical = match row {
                Some(row) => format!("encode.r{cap}.n{row}"),
                None => format!("encode.r{cap}"),
            };
            if role != &canonical || cap < min_rows || cap > max_rows || row.is_some() != exact_rows
            {
                return Err(bad());
            }
            if let Some(row) = row {
                if row < min_rows || row > cap {
                    return Err(bad());
                }
                let choice = rows.entry(row).or_insert((cap, program as usize));
                if cap < choice.0 {
                    *choice = (cap, program as usize);
                }
            } else {
                capacities.push((cap, program as usize));
            }
        }
        let programs = if exact_rows {
            if rows.len() != max_rows - min_rows + 1 {
                return Err("speaker programs do not cover every input row count".into());
            }
            SpeakerEncoderPrograms::Rows(rows.into_iter().map(|(n, (_, p))| (n, p)).collect())
        } else {
            capacities.sort_unstable();
            if capacities.last().map(|&(cap, _)| cap) != Some(max_rows) {
                return Err("speaker programs do not cover the maximum row count".into());
            }
            SpeakerEncoderPrograms::Capacities(capacities)
        };
        Ok(SpeakerEncoderPlan {
            min_rows,
            max_rows,
            input_width,
            output_width,
            reflect_pad,
            programs,
        })
    }

    pub fn codec_sample_input(&self) -> Result<bool, String> {
        match self.driver.as_str() {
            "codec.encode.v1" => Ok(false),
            "codec.encode.v2" => match self.tensors.get("samples") {
                Some(t) if t.dtype == PipelineDType::U32 && t.shape == [1] => Ok(true),
                _ => Err("codec.encode.v2 requires samples: u32[1]".into()),
            },
            _ => Err(format!(
                "unsupported codec encoder driver {:?}",
                self.driver
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeakerEncoderPlan {
    pub min_rows: usize,
    pub max_rows: usize,
    pub input_width: usize,
    pub output_width: usize,
    pub reflect_pad: usize,
    programs: SpeakerEncoderPrograms,
}

#[derive(Clone, Debug)]
enum SpeakerEncoderPrograms {
    Capacities(Vec<(usize, usize)>),
    Rows(BTreeMap<usize, usize>),
}

impl SpeakerEncoderPlan {
    pub fn program(&self, rows: usize) -> Option<usize> {
        if !(self.min_rows..=self.max_rows).contains(&rows) {
            return None;
        }
        match &self.programs {
            SpeakerEncoderPrograms::Capacities(capacities) => capacities
                .iter()
                .find_map(|&(cap, program)| (cap >= rows).then_some(program)),
            SpeakerEncoderPrograms::Rows(programs) => programs.get(&rows).copied(),
        }
    }
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
    U64,
    I16,
    F16,
    Bf16,
    F32,
}

impl PipelineDType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U32 | Self::F32 => 4,
            Self::U64 => 8,
            Self::I16 | Self::F16 | Self::Bf16 => 2,
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
            if pipeline.strings.len() > 64
                || pipeline
                    .strings
                    .iter()
                    .any(|(name, value)| !identifier(name, 64) || value.len() > 16384)
            {
                return Err("invalid packet pipeline string".into());
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

    fn speaker(version: u32) -> PacketPipeline {
        let pad = usize::from(version == 3);
        let mut p = PacketPipeline {
            name: "speaker.encode".into(),
            driver: format!("speaker.encode.v{version}"),
            programs: BTreeMap::new(),
            tensors: [
                ("mel", PipelineDType::F32, vec![(8 + 2 * pad) as u64, 2]),
                ("length", PipelineDType::U32, vec![1]),
                ("embedding", PipelineDType::F32, vec![3]),
            ]
            .into_iter()
            .map(|(role, dtype, shape)| {
                (
                    role.into(),
                    PipelineTensor {
                        name: format!("in.{role}"),
                        dtype,
                        shape,
                    },
                )
            })
            .collect(),
            parameters: BTreeMap::from([
                ("input.min_rows".into(), 2),
                ("input.max_rows".into(), 8),
                ("input.width".into(), 2),
                ("output.width".into(), 3),
            ]),
            strings: BTreeMap::new(),
        };
        if pad > 0 {
            p.parameters.insert("input.reflect_pad".into(), pad as u64);
        }
        if version == 1 {
            p.programs = BTreeMap::from([("encode.r4".into(), 0), ("encode.r8".into(), 1)]);
        } else {
            for n in 2..=8 {
                p.programs.insert(format!("encode.r8.n{n}"), 100 + n);
                if n <= 4 {
                    p.programs.insert(format!("encode.r4.n{n}"), n);
                }
            }
        }
        p
    }

    #[test]
    fn speaker_programs_select_smallest_capacity_for_every_valid_length() {
        for version in [1, 2, 3] {
            let plan = speaker(version).speaker_encoder_plan().unwrap();
            assert_eq!(plan.reflect_pad, usize::from(version == 3));
            assert_eq!((plan.input_width, plan.output_width), (2, 3));
            assert_eq!(plan.program(1), None);
            assert_eq!(plan.program(9), None);
            for n in 2..=8 {
                assert_eq!(
                    plan.program(n),
                    Some(if version == 1 {
                        usize::from(n > 4)
                    } else if n <= 4 {
                        n
                    } else {
                        100 + n
                    })
                );
            }
        }
    }

    #[test]
    fn speaker_plan_rejects_missing_shapes_lengths_and_program_coverage() {
        for mutation in 0..13 {
            let mut p = speaker(3);
            match mutation {
                0 => p.driver = "speaker.encode.v4".into(),
                1 => {
                    p.tensors.remove("length");
                }
                2 => p.tensors.get_mut("length").unwrap().dtype = PipelineDType::F32,
                3 => p.tensors.get_mut("mel").unwrap().shape = vec![8, 2],
                4 => p.tensors.get_mut("embedding").unwrap().shape = vec![1, 3],
                5 => {
                    p.parameters.insert("input.min_rows".into(), 9);
                }
                6 => {
                    p.parameters.insert("input.max_rows".into(), u64::MAX);
                }
                7 => {
                    p.parameters.remove("input.reflect_pad");
                }
                8 => {
                    p.parameters.insert("input.reflect_pad".into(), 2);
                }
                9 => {
                    p.programs.remove("encode.r8.n5");
                }
                10 => {
                    p.programs.insert("encode.r4.n5".into(), 10);
                }
                11 => {
                    p.programs.insert("encode.r08.n5".into(), 10);
                }
                12 => {
                    p.programs.insert("encode.r8".into(), 10);
                }
                _ => unreachable!(),
            }
            assert!(p.speaker_encoder_plan().is_err(), "mutation {mutation}");
        }
        let mut p = speaker(1);
        p.programs.remove("encode.r8");
        assert!(p.speaker_encoder_plan().is_err());
        let mut p = speaker(2);
        p.parameters.insert("input.reflect_pad".into(), 1);
        assert!(p.speaker_encoder_plan().is_err());
    }

    #[test]
    fn codec_sample_count_requires_versioned_scalar_u32() {
        let mut p = PacketPipeline {
            name: "codec.encode".into(),
            driver: "codec.encode.v1".into(),
            programs: BTreeMap::new(),
            tensors: BTreeMap::new(),
            parameters: BTreeMap::new(),
            strings: BTreeMap::new(),
        };
        assert_eq!(p.codec_sample_input(), Ok(false));
        p.driver = "codec.encode.v2".into();
        assert!(p.codec_sample_input().is_err());
        for (dtype, shape, valid) in [
            (PipelineDType::U32, vec![1], true),
            (PipelineDType::F32, vec![1], false),
            (PipelineDType::U32, vec![2], false),
            (PipelineDType::U32, vec![1, 1], false),
        ] {
            p.tensors.insert(
                "samples".into(),
                PipelineTensor {
                    name: "in.samples".into(),
                    dtype,
                    shape,
                },
            );
            assert_eq!(p.codec_sample_input().is_ok(), valid);
        }
        p.driver = "codec.encode.v3".into();
        assert!(p.codec_sample_input().is_err());
    }

    #[test]
    fn validates_programs_and_tensor_geometry() {
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![PacketPipeline {
                strings: Default::default(),
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
