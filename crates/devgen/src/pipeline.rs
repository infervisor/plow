use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

#[derive(Clone, Copy)]
pub struct CausalPipelineTensors {
    pub tokens: u32,
    pub positions: u32,
    pub kv_lengths: u32,
    pub overlay: Option<u32>,
    pub overlay_index: Option<u32>,
}

#[derive(Clone, Copy)]
pub struct CausalPipelineSpec<'a> {
    pub name: &'a str,
    pub max_context: u32,
    pub hidden: u32,
    pub decode_capacity: u32,
    pub overlay_rows: u32,
    /// Execute dependency-bearing programs in compiler order. Backends whose normal
    /// execution is already ordered may ignore this hint.
    pub ordered_dispatch: bool,
    pub tensors: CausalPipelineTensors,
}

pub fn causal_pipeline_section(
    model: &Model,
    spec: CausalPipelineSpec<'_>,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    use std::collections::BTreeMap;

    if spec.max_context == 0 || spec.hidden == 0 || spec.decode_capacity == 0 {
        return Err("invalid causal pipeline geometry".into());
    }
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    if prefill_count == 0 || prefill_count == model.progs.len() {
        return Err("causal pipeline requires prefill and decode programs".into());
    }
    let mut programs = BTreeMap::new();
    for (program, &rows) in model.prog_t.iter().enumerate() {
        let phase = if program < prefill_count {
            "prefill"
        } else {
            "decode"
        };
        if programs
            .insert(format!("{phase}.{rows}"), program as u32)
            .is_some()
        {
            return Err(format!("duplicate {phase} program for {rows} rows"));
        }
    }
    let binding = |handle: u32, dtype, shape: Vec<u64>| -> Result<PipelineTensor, String> {
        let tensor = model
            .tensors
            .get(handle as usize)
            .ok_or_else(|| format!("causal pipeline tensor handle {handle} is missing"))?;
        Ok(PipelineTensor {
            name: tensor.name.clone(),
            dtype,
            shape,
        })
    };
    let mut tensors = BTreeMap::from([
        (
            "tokens".into(),
            binding(
                spec.tensors.tokens,
                PipelineDType::U32,
                vec![u64::from(spec.max_context)],
            )?,
        ),
        (
            "positions".into(),
            binding(
                spec.tensors.positions,
                PipelineDType::U32,
                vec![u64::from(spec.max_context)],
            )?,
        ),
        (
            "kv_lengths".into(),
            binding(
                spec.tensors.kv_lengths,
                PipelineDType::U32,
                vec![u64::from(spec.decode_capacity)],
            )?,
        ),
    ]);
    match (spec.tensors.overlay, spec.tensors.overlay_index) {
        (Some(overlay), Some(index)) if spec.overlay_rows > 0 => {
            tensors.insert(
                "overlay".into(),
                binding(
                    overlay,
                    PipelineDType::F32,
                    vec![u64::from(spec.overlay_rows), u64::from(spec.hidden)],
                )?,
            );
            tensors.insert(
                "overlay_index".into(),
                binding(index, PipelineDType::U32, vec![u64::from(spec.max_context)])?,
            );
        }
        (None, None) if spec.overlay_rows == 0 => {}
        _ => return Err("causal overlay tensors and geometry disagree".into()),
    }
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: spec.name.into(),
            driver: "causal.v1".into(),
            programs,
            tensors,
            parameters: BTreeMap::from([
                ("max_context".into(), u64::from(spec.max_context)),
                ("hidden".into(), u64::from(spec.hidden)),
                ("decode_capacity".into(), u64::from(spec.decode_capacity)),
                ("overlay_rows".into(), u64::from(spec.overlay_rows)),
                (
                    "ordered_dispatch".into(),
                    u64::from(spec.ordered_dispatch),
                ),
                ("prefill_program_count".into(), prefill_count as u64),
                (
                    "decode_program_count".into(),
                    (model.progs.len() - prefill_count) as u64,
                ),
            ]),
        }],
    };
    metadata.validate(model.progs.len(), |name| {
        model
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| tensor.bytes)
    })?;
    Ok(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
    })
}

pub struct PacketPrefix {
    pub model: Model,
    pub programs: Vec<usize>,
    pub input: u32,
    pub output: u32,
    pub input_shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseActivation {
    None,
    Relu,
    GeluErfBf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseWeight {
    F32,
    Bf16,
}

#[derive(Clone, Copy)]
pub struct DenseF32Stage<'a> {
    pub output: &'a str,
    pub weight: &'a str,
    pub bias: Option<&'a str>,
    pub input_width: u32,
    pub output_width: u32,
    pub activation: DenseActivation,
    pub round_bf16: bool,
    pub weight_type: DenseWeight,
}

pub struct InitializedAddF32Stage<'a> {
    pub output: &'a str,
    pub addend: &'a str,
    pub values: &'a [f32],
    pub scale: f32,
    pub round_bf16: bool,
}

#[derive(Clone, Copy)]
pub struct ScaledAddF32Stage<'a> {
    pub output: &'a str,
    pub scale: f32,
    pub round_bf16: bool,
}

#[derive(Clone, Copy)]
pub struct LayerNormF32Stage<'a> {
    pub output: &'a str,
    pub gamma: Option<&'a str>,
    pub beta: Option<&'a str>,
    pub width: u32,
    pub epsilon: f32,
    pub round_bf16: bool,
    pub ordered_statistics: bool,
}

#[derive(Clone, Copy)]
pub struct GroupedAttentionF32Stage<'a> {
    pub output: &'a str,
    pub valid_rows: Option<u32>,
    pub width: u32,
    pub head_width: u32,
    pub group_rows: u32,
    pub round_score_bf16: bool,
    pub round_probability_bf16: bool,
    pub round_output_bf16: bool,
}

#[derive(Clone, Copy)]
pub struct EmbedOverlayBf16Stage<'a> {
    pub output: &'a str,
    pub table: &'a str,
    pub tokens: &'a str,
    pub overlay_index: &'a str,
    pub rows: u32,
    pub width: u32,
    pub vocabulary: u32,
    pub overlay_rows: u32,
}

impl PacketPrefix {
    pub fn forward_pipeline_section(
        &self,
        name: &str,
        input_dtype: plow_asset::packet_pipeline::PipelineDType,
        output_dtype: plow_asset::packet_pipeline::PipelineDType,
        output_shape: Vec<u64>,
    ) -> Result<packet::devbuild::SectionData, String> {
        use plow_asset::packet_pipeline::{
            PacketPipeline, PacketPipelines, PipelineTensor, SECTION, VERSION,
        };
        use std::collections::BTreeMap;

        let input = self
            .model
            .tensors
            .get(self.input as usize)
            .ok_or("pipeline input tensor is missing")?;
        let output = self
            .model
            .tensors
            .get(self.output as usize)
            .ok_or("pipeline output tensor is missing")?;
        let programs = self
            .programs
            .iter()
            .enumerate()
            .map(|(stage, &program)| (format!("forward.{stage}"), program as u32))
            .collect();
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![PacketPipeline {
                name: name.into(),
                driver: "forward.v1".into(),
                programs,
                tensors: BTreeMap::from([
                    (
                        "input".into(),
                        PipelineTensor {
                            name: input.name.clone(),
                            dtype: input_dtype,
                            shape: self.input_shape.clone(),
                        },
                    ),
                    (
                        "output".into(),
                        PipelineTensor {
                            name: output.name.clone(),
                            dtype: output_dtype,
                            shape: output_shape,
                        },
                    ),
                ]),
                parameters: BTreeMap::from([(
                    "forward_program_count".into(),
                    self.programs.len() as u64,
                )]),
            }],
        };
        metadata.validate(self.model.progs.len(), |tensor_name| {
            self.model
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
                .map(|tensor| tensor.bytes)
        })?;
        Ok(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
        })
    }

    pub fn append_dense_f32(self, rows: u32, stage: DenseF32Stage<'_>) -> Result<Self, String> {
        let source = self.output;
        self.append_dense_f32_from(source, rows, stage)
    }

    pub fn append_dense_f32_from(
        mut self,
        source: u32,
        rows: u32,
        stage: DenseF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0 || rows == 0 || stage.input_width == 0 || stage.output_width == 0 {
            return Err("invalid dense geometry".into());
        }
        if source as usize >= self.model.tensors.len() {
            return Err("dense source tensor is missing".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(rows, stage.output_width, 4)?);
        let weight = builder.tensor(
            stage.weight,
            tensor_bytes(
                stage.output_width,
                stage.input_width,
                match stage.weight_type {
                    DenseWeight::F32 => 4,
                    DenseWeight::Bf16 => 2,
                },
            )?,
        );
        let bias = stage
            .bias
            .map(|name| builder.tensor(name, u64::from(stage.output_width) * 4))
            .unwrap_or(packet::dev::TENSOR_NONE);
        let activation = u32::from(stage.activation == DenseActivation::Relu);
        let flags = u32::from(stage.round_bf16)
            | (u32::from(stage.activation == DenseActivation::GeluErfBf16) << 1)
            | (u32::from(stage.weight_type == DenseWeight::Bf16) << 2);
        let blocks = rows
            .div_ceil(128)
            .checked_mul(stage.output_width.div_ceil(64))
            .ok_or("dense dispatch geometry overflows")?;
        builder.emit(
            DevOp::DenseGemmF32,
            repeated(self.model.n_cu, blocks),
            &[],
            |instruction| {
                instruction.t[..4].copy_from_slice(&[output, source, weight, bias]);
                instruction.i[..4].copy_from_slice(&[
                    rows,
                    stage.output_width,
                    stage.input_width,
                    activation,
                ]);
                instruction.i[7] = flags;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_initialized_add_f32(
        mut self,
        stage: InitializedAddF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0 || stage.values.is_empty() || !stage.scale.is_finite() {
            return Err("invalid scaled-add stage".into());
        }
        let elements: u32 = stage
            .values
            .len()
            .try_into()
            .map_err(|_| "scaled-add tensor is too large")?;
        let mut init = Vec::with_capacity(stage.values.len() * 4);
        for value in stage.values {
            init.extend_from_slice(&value.to_le_bytes());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, u64::from(elements) * 4);
        let addend = builder.tensor_init(stage.addend, init);
        builder.emit(
            DevOp::ScaledAddF32,
            repeated(self.model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..3].copy_from_slice(&[output, self.output, addend]);
                instruction.i[..2].copy_from_slice(&[elements, u32::from(stage.round_bf16)]);
                instruction.f[0] = stage.scale;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(elements);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_scaled_add_f32(
        mut self,
        a: u32,
        b: u32,
        elements: u32,
        stage: ScaledAddF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0 || elements == 0 || !stage.scale.is_finite() {
            return Err("invalid scaled-add stage".into());
        }
        if [a, b]
            .into_iter()
            .any(|handle| handle as usize >= self.model.tensors.len())
        {
            return Err("scaled-add source tensor is missing".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, u64::from(elements) * 4);
        builder.emit(
            DevOp::ScaledAddF32,
            repeated(self.model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..3].copy_from_slice(&[output, a, b]);
                instruction.i[..2].copy_from_slice(&[elements, u32::from(stage.round_bf16)]);
                instruction.f[0] = stage.scale;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(elements);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_layer_norm_f32(
        mut self,
        rows: u32,
        stage: LayerNormF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0
            || rows == 0
            || stage.width == 0
            || !stage.epsilon.is_finite()
            || stage.epsilon <= 0.0
        {
            return Err("invalid layer-normalization stage".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(rows, stage.width, 4)?);
        let gamma = stage
            .gamma
            .map(|name| builder.tensor(name, u64::from(stage.width) * 4))
            .unwrap_or(packet::dev::TENSOR_NONE);
        let beta = stage
            .beta
            .map(|name| builder.tensor(name, u64::from(stage.width) * 4))
            .unwrap_or(packet::dev::TENSOR_NONE);
        builder.emit(
            DevOp::LayerNormF32,
            repeated(self.model.n_cu, self.model.n_cu),
            &[],
            |instruction| {
                instruction.t[..4].copy_from_slice(&[output, self.output, gamma, beta]);
                let flags =
                    u32::from(stage.round_bf16) | (u32::from(stage.ordered_statistics) << 1);
                instruction.i[..3].copy_from_slice(&[rows, stage.width, flags]);
                instruction.f[0] = stage.epsilon;
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_grouped_attention_f32(
        mut self,
        query: u32,
        key: u32,
        value: u32,
        rows: u32,
        stage: GroupedAttentionF32Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0
            || rows == 0
            || stage.width == 0
            || stage.head_width == 0
            || stage.width % stage.head_width != 0
            || !(1..=256).contains(&stage.group_rows)
        {
            return Err("invalid grouped-attention geometry".into());
        }
        if [Some(query), Some(key), Some(value), stage.valid_rows]
            .into_iter()
            .flatten()
            .any(|handle| handle as usize >= self.model.tensors.len())
        {
            return Err("grouped-attention source tensor is missing".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(rows, stage.width, 4)?);
        let flags = u32::from(stage.round_score_bf16)
            | (u32::from(stage.round_probability_bf16) << 1)
            | (u32::from(stage.round_output_bf16) << 2);
        let heads = stage.width / stage.head_width;
        let blocks = rows
            .checked_mul(heads)
            .ok_or("grouped-attention dispatch geometry overflows")?
            .div_ceil(32);
        builder.emit(
            DevOp::GroupedAttentionF32,
            repeated(self.model.n_cu, blocks),
            &[],
            |instruction| {
                instruction.t[..4].copy_from_slice(&[output, query, key, value]);
                instruction.t[4] = stage.valid_rows.unwrap_or(packet::dev::TENSOR_NONE);
                instruction.i[..5].copy_from_slice(&[
                    rows,
                    stage.width,
                    stage.head_width,
                    stage.group_rows,
                    flags,
                ]);
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }

    pub fn append_embed_overlay_bf16(
        mut self,
        stage: EmbedOverlayBf16Stage<'_>,
    ) -> Result<Self, String> {
        if self.model.n_cu == 0
            || stage.rows == 0
            || stage.width == 0
            || stage.vocabulary == 0
            || stage.overlay_rows == 0
        {
            return Err("invalid embedding-overlay geometry".into());
        }
        let overlay_bytes = tensor_bytes(stage.overlay_rows, stage.width, 4)?;
        if self
            .model
            .tensors
            .get(self.output as usize)
            .is_none_or(|source| source.bytes != overlay_bytes)
        {
            return Err("embedding-overlay source has the wrong size".into());
        }
        let mut builder = Builder::new(self.model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(std::mem::take(&mut self.model.tensors));
        let output = builder.tensor(stage.output, tensor_bytes(stage.rows, stage.width, 2)?);
        let table = builder.tensor(stage.table, tensor_bytes(stage.vocabulary, stage.width, 2)?);
        let tokens = builder.tensor(stage.tokens, u64::from(stage.rows) * 4);
        let overlay_index = builder.tensor(stage.overlay_index, u64::from(stage.rows) * 4);
        let elements = stage
            .rows
            .checked_mul(stage.width)
            .ok_or("embedding-overlay dispatch geometry overflows")?;
        builder.emit(
            DevOp::EmbedOverlayBf16,
            repeated(self.model.n_cu, elements.div_ceil(1024)),
            &[],
            |instruction| {
                instruction.t[..5].copy_from_slice(&[
                    output,
                    table,
                    tokens,
                    self.output,
                    overlay_index,
                ]);
                instruction.i[..4].copy_from_slice(&[
                    stage.rows,
                    stage.width,
                    stage.vocabulary,
                    stage.overlay_rows,
                ]);
            },
        );
        let program = self.model.progs.len();
        self.model.tensors = builder.tensors();
        self.model.progs.push(builder.finish());
        self.model.prog_t.push(stage.rows);
        self.programs.push(program);
        self.output = output;
        Ok(self)
    }
}

fn tensor_bytes(rows: u32, columns: u32, element_bytes: u32) -> Result<u64, String> {
    u64::from(rows)
        .checked_mul(u64::from(columns))
        .and_then(|count| count.checked_mul(u64::from(element_bytes)))
        .ok_or_else(|| "tensor size overflows".into())
}

fn repeated(n_cu: u32, blocks: u32) -> Vec<u32> {
    (0..blocks.max(1)).map(|index| index % n_cu).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_backend_neutral_causal_pipeline() {
        let mut tensors = Builder::new(2);
        let tokens = tensors.tensor("in.ids", 512 * 4);
        let positions = tensors.tensor("in.pos", 512 * 4);
        let kv_lengths = tensors.tensor("in.kvlen", 4 * 4);
        let overlay = tensors.tensor("in.encoder", 8 * 16 * 4);
        let overlay_index = tensors.tensor("in.encoder_index", 512 * 4);
        let model = Model {
            n_cu: 2,
            target: 0,
            tensors: tensors.tensors(),
            progs: (0..4).map(|_| Builder::new(2).finish()).collect(),
            kv_row_insts: Vec::new(),
            prog_t: vec![128, 512, 1, 4],
            gen: Vec::new(),
        };
        let section = causal_pipeline_section(
            &model,
            CausalPipelineSpec {
                name: "decode",
                max_context: 512,
                hidden: 16,
                decode_capacity: 4,
                overlay_rows: 8,
                ordered_dispatch: false,
                tensors: CausalPipelineTensors {
                    tokens,
                    positions,
                    kv_lengths,
                    overlay: Some(overlay),
                    overlay_index: Some(overlay_index),
                },
            },
        )
        .unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let pipeline = &metadata.pipelines[0];
        assert_eq!(pipeline.driver, "causal.v1");
        assert_eq!(pipeline.programs["prefill.128"], 0);
        assert_eq!(pipeline.programs["prefill.512"], 1);
        assert_eq!(pipeline.programs["decode.1"], 2);
        assert_eq!(pipeline.programs["decode.4"], 3);
        assert_eq!(pipeline.tensors["overlay"].shape, [8, 16]);
        assert_eq!(pipeline.parameters["decode_capacity"], 4);
        assert_eq!(pipeline.parameters["ordered_dispatch"], 0);
    }

    #[test]
    fn appends_dense_program_to_prefix() {
        let mut builder = Builder::new(4);
        let input = builder.tensor("input", 3 * 8 * 4);
        let model = Model {
            n_cu: 4,
            target: 0,
            tensors: builder.tensors(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        };
        let prefix = PacketPrefix {
            model,
            programs: Vec::new(),
            input,
            output: input,
            input_shape: vec![3, 8],
        }
        .append_dense_f32(
            3,
            DenseF32Stage {
                output: "output",
                weight: "weight",
                bias: None,
                input_width: 8,
                output_width: 5,
                activation: DenseActivation::GeluErfBf16,
                round_bf16: false,
                weight_type: DenseWeight::F32,
            },
        )
        .unwrap();
        assert_eq!(prefix.programs, [0]);
        let instruction = &prefix.model.progs[0].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::DenseGemmF32));
        assert_eq!(&instruction.i[..4], &[3, 5, 8, 0]);
        assert_eq!(instruction.i[7], 2);
        assert_eq!(instruction.t[3], packet::dev::TENSOR_NONE);
        assert!(!prefix.model.to_blob().is_empty());
        let values = [0.0f32; 15];
        let prefix = prefix
            .append_initialized_add_f32(InitializedAddF32Stage {
                output: "positioned",
                addend: "position",
                values: &values,
                scale: 1.0,
                round_bf16: true,
            })
            .unwrap();
        assert_eq!(prefix.programs, [0, 1]);
        let instruction = &prefix.model.progs[1].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::ScaledAddF32));
        assert_eq!(&instruction.i[..2], &[15, 1]);
        let prefix = prefix
            .append_layer_norm_f32(
                3,
                LayerNormF32Stage {
                    output: "normalized",
                    gamma: Some("gamma"),
                    beta: Some("beta"),
                    width: 5,
                    epsilon: 1e-5,
                    round_bf16: true,
                    ordered_statistics: true,
                },
            )
            .unwrap();
        assert_eq!(prefix.programs, [0, 1, 2]);
        let instruction = &prefix.model.progs[2].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::LayerNormF32));
        assert_eq!(&instruction.i[..3], &[3, 5, 3]);

        let normalized = prefix.output;
        let prefix = prefix
            .append_dense_f32_from(
                normalized,
                3,
                DenseF32Stage {
                    output: "query",
                    weight: "query.weight",
                    bias: Some("query.bias"),
                    input_width: 5,
                    output_width: 5,
                    activation: DenseActivation::None,
                    round_bf16: true,
                    weight_type: DenseWeight::F32,
                },
            )
            .unwrap();
        let query = prefix.output;
        let prefix = prefix
            .append_dense_f32_from(
                normalized,
                3,
                DenseF32Stage {
                    output: "key",
                    weight: "key.weight",
                    bias: Some("key.bias"),
                    input_width: 5,
                    output_width: 5,
                    activation: DenseActivation::None,
                    round_bf16: true,
                    weight_type: DenseWeight::F32,
                },
            )
            .unwrap();
        let key = prefix.output;
        assert_eq!(prefix.programs, [0, 1, 2, 3, 4]);
        assert_eq!(prefix.model.progs[3].insts[0].t[1], normalized);
        assert_eq!(prefix.model.progs[4].insts[0].t[1], normalized);
        let prefix = prefix
            .append_grouped_attention_f32(
                query,
                key,
                query,
                3,
                GroupedAttentionF32Stage {
                    output: "attention",
                    valid_rows: None,
                    width: 5,
                    head_width: 5,
                    group_rows: 3,
                    round_score_bf16: true,
                    round_probability_bf16: true,
                    round_output_bf16: true,
                },
            )
            .unwrap();
        let instruction = &prefix.model.progs[5].insts[0];
        assert_eq!(
            DevOp::from_u16(instruction.op),
            Some(DevOp::GroupedAttentionF32)
        );
        assert_eq!(&instruction.t[1..4], &[query, key, query]);
        assert_eq!(instruction.t[4], packet::dev::TENSOR_NONE);
        assert_eq!(&instruction.i[..5], &[3, 5, 5, 3, 7]);
        let prefix = prefix
            .append_scaled_add_f32(
                query,
                key,
                15,
                ScaledAddF32Stage {
                    output: "residual",
                    scale: 0.5,
                    round_bf16: true,
                },
            )
            .unwrap();
        let instruction = &prefix.model.progs[6].insts[0];
        assert_eq!(DevOp::from_u16(instruction.op), Some(DevOp::ScaledAddF32));
        assert_eq!(&instruction.t[1..3], &[query, key]);
        assert_eq!(&instruction.i[..2], &[15, 1]);
        assert_eq!(instruction.f[0], 0.5);
        let prefix = prefix
            .append_embed_overlay_bf16(EmbedOverlayBf16Stage {
                output: "act.embeddings",
                table: "embedding.weight",
                tokens: "in.tokens",
                overlay_index: "in.overlay_index",
                rows: 3,
                width: 5,
                vocabulary: 11,
                overlay_rows: 3,
            })
            .unwrap();
        let instruction = &prefix.model.progs[7].insts[0];
        assert_eq!(
            DevOp::from_u16(instruction.op),
            Some(DevOp::EmbedOverlayBf16)
        );
        assert_eq!(&instruction.i[..4], &[3, 5, 11, 3]);
        assert_eq!(prefix.model.tensors[prefix.output as usize].bytes, 30);
        let section = prefix
            .forward_pipeline_section(
                "encode",
                plow_asset::packet_pipeline::PipelineDType::F32,
                plow_asset::packet_pipeline::PipelineDType::Bf16,
                vec![3, 5],
            )
            .unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        assert_eq!(metadata.pipelines[0].programs.len(), 8);
        assert_eq!(metadata.pipelines[0].programs["forward.7"], 7);
        assert_eq!(metadata.pipelines[0].tensors["input"].shape, [3, 8]);
        assert_eq!(metadata.pipelines[0].tensors["output"].shape, [3, 5]);
    }
}
