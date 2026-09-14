use crate::conv2d::{
    self, Conv2dSpec, Conv2dStage, ConvActivation, ConvKind, ConvLayout, ConvWeight,
};
use crate::pipeline::{
    DenseActivation, DenseF32Stage, DenseWeight, GroupedAttentionF32Stage, InitializedAddF32Stage,
    LayerNormF32Stage, PacketPrefix, ScaledAddF32Stage,
};
use std::collections::BTreeMap;

pub struct AudioEncoderPackets {
    pub prefix: PacketPrefix,
    pub rows: u32,
    pub input: u32,
    pub convolution: u32,
    pub projection: u32,
    pub positioned: u32,
    pub first_norm: u32,
    pub first_qkv: [u32; 3],
    pub first_attention: u32,
    pub first_output: u32,
    pub transformer_output: u32,
    pub output_shape: [u32; 4],
    feature_frames: u32,
    valid_rows: u32,
    capacity_programs: BTreeMap<u32, Vec<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioWeightDType {
    Bf16,
    F32,
}

impl AudioEncoderPackets {
    pub fn merge_capacity(&mut self, mut bucket: AudioEncoderPackets) -> Result<(), String> {
        let capacity = bucket.feature_frames;
        if capacity == 0
            || capacity >= self.feature_frames
            || self.capacity_programs.contains_key(&capacity)
            || bucket.rows > self.rows
            || bucket.prefix.model.n_cu != self.prefix.model.n_cu
            || bucket.prefix.model.target != self.prefix.model.target
            || !bucket.prefix.model.gen.is_empty()
            || bucket.prefix.model.tensors.len() != self.prefix.model.tensors.len()
            || !bucket.capacity_programs.is_empty()
        {
            return Err("incompatible Qwen audio encoder capacity".into());
        }
        for (base, candidate) in self
            .prefix
            .model
            .tensors
            .iter_mut()
            .zip(&bucket.prefix.model.tensors)
        {
            if base.name != candidate.name || base.bytes < candidate.bytes {
                return Err(format!(
                    "Qwen audio capacity tensor {:?} is incompatible",
                    candidate.name
                ));
            }
            if let Some(init) = &candidate.init {
                match &base.init {
                    Some(existing) if existing.starts_with(init) => {}
                    None if base.bytes == candidate.bytes => base.init = Some(init.clone()),
                    _ => {
                        return Err(format!(
                            "Qwen audio capacity tensor {:?} has different initialization",
                            candidate.name
                        ))
                    }
                }
            }
        }
        let mut programs: Vec<_> = bucket.prefix.model.progs.drain(..).map(Some).collect();
        let mut merged = Vec::with_capacity(bucket.prefix.programs.len());
        for program in bucket.prefix.programs {
            let value = programs
                .get_mut(program)
                .and_then(Option::take)
                .ok_or("Qwen audio capacity program is missing")?;
            let index = self.prefix.model.progs.len();
            self.prefix.model.progs.push(value);
            self.prefix.model.prog_t.push(
                *bucket
                    .prefix
                    .model
                    .prog_t
                    .get(program)
                    .ok_or("Qwen audio capacity program shape is missing")?,
            );
            merged.push(index);
        }
        self.capacity_programs.insert(capacity, merged);
        Ok(())
    }

    pub fn embed_checkpoint(&mut self, dir: &std::path::Path) -> Result<(), String> {
        let checkpoint = crate::checkpoint::TensorReader::open(dir)?;
        self.embed_weights(|name| {
            let (dtype, bytes) = checkpoint.read(name)?;
            let dtype = match dtype {
                "BF16" => AudioWeightDType::Bf16,
                "F32" => AudioWeightDType::F32,
                _ => return Err(format!("{name} has unsupported dtype {dtype}")),
            };
            Ok((dtype, bytes))
        })
    }

    pub fn embed_weights(
        &mut self,
        mut resolve: impl FnMut(&str) -> Result<(AudioWeightDType, Vec<u8>), String>,
    ) -> Result<(), String> {
        for tensor in &mut self.prefix.model.tensors {
            if !tensor.name.starts_with("thinker.audio_tower.") || tensor.init.is_some() {
                continue;
            }
            let (dtype, bytes) = resolve(&tensor.name)?;
            let converted = match dtype {
                AudioWeightDType::Bf16 if bytes.len() as u64 == tensor.bytes => bytes,
                AudioWeightDType::Bf16 if (bytes.len() as u64) * 2 == tensor.bytes => bytes
                    .chunks_exact(2)
                    .flat_map(|bytes| {
                        let bits = u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16;
                        bits.to_le_bytes()
                    })
                    .collect(),
                AudioWeightDType::F32 if bytes.len() as u64 == tensor.bytes => bytes,
                _ => {
                    return Err(format!(
                        "tensor {} has dtype {dtype:?} and {} bytes, expected {} bytes",
                        tensor.name,
                        bytes.len(),
                        tensor.bytes
                    ))
                }
            };
            tensor.init = Some(converted);
        }
        Ok(())
    }

    pub fn pipeline_section(
        &self,
        feature_frames: u32,
    ) -> Result<packet::devbuild::SectionData, String> {
        use plow_asset::packet_pipeline::{PacketPipelines, PipelineDType};

        if qwen_audio_rows(feature_frames) != self.rows {
            return Err("Qwen audio pipeline capacity does not match its output rows".into());
        }
        let mut section = self.prefix.forward_pipeline_section(
            "audio.encode",
            PipelineDType::F32,
            PipelineDType::F32,
            vec![u64::from(self.rows), 2048],
        )?;
        let mut metadata: PacketPipelines =
            serde_json::from_slice(&section.data).map_err(|error| error.to_string())?;
        let pipeline = metadata
            .pipelines
            .first_mut()
            .ok_or("Qwen audio packet has no pipeline")?;
        for (stage, &program) in self.prefix.programs.iter().enumerate() {
            pipeline
                .programs
                .insert(format!("forward.{feature_frames}.{stage}"), program as u32);
        }
        for (&capacity, programs) in &self.capacity_programs {
            for (stage, &program) in programs.iter().enumerate() {
                pipeline
                    .programs
                    .insert(format!("forward.{capacity}.{stage}"), program as u32);
            }
        }
        pipeline
            .parameters
            .insert("input_frames".into(), u64::from(feature_frames));
        pipeline
            .parameters
            .insert("output_rows".into(), u64::from(self.rows));
        pipeline.parameters.insert("feature_bins".into(), 128);
        pipeline.parameters.insert("output_width".into(), 2048);
        pipeline.parameters.insert("qwen_audio_graph_v1".into(), 1);
        pipeline.tensors.insert(
            "valid_rows".into(),
            plow_asset::packet_pipeline::PipelineTensor {
                name: self.prefix.model.tensors[self.valid_rows as usize]
                    .name
                    .clone(),
                dtype: PipelineDType::U32,
                shape: vec![1],
            },
        );
        section.data = serde_json::to_vec(&metadata).map_err(|error| error.to_string())?;
        Ok(section)
    }
}

pub fn lower_audio_encoder(feature_frames: u32, n_cu: u32) -> Result<AudioEncoderPackets, String> {
    if !(50..=3000).contains(&feature_frames) || n_cu == 0 {
        return Err("invalid Qwen audio encoder geometry".into());
    }
    let chunks = feature_frames.div_ceil(100);
    let chunk_frames = feature_frames.min(100);
    let names: Vec<_> = (1..=3)
        .map(|index| {
            (
                format!("thinker.audio_tower.conv2d{index}.weight"),
                format!("thinker.audio_tower.conv2d{index}.bias"),
            )
        })
        .collect();
    let stages: Vec<_> = names
        .iter()
        .enumerate()
        .map(|(index, (weight, bias))| Conv2dStage {
            weight,
            bias,
            kernel: 3,
            stride: 2,
            pad_before: 1,
            pad_after: 1,
            input_channels: if index == 0 { 1 } else { 480 },
            output_channels: 480,
            kind: ConvKind::Standard,
            activation: ConvActivation::GeluErfBf16,
            output_layout: ConvLayout::ChannelsFramesWidth,
            weight_type: ConvWeight::F32,
        })
        .collect();
    let packets = conv2d::lower(
        Conv2dSpec {
            batches: chunks,
            input_frames: 128,
            input_width: chunk_frames,
            input_channels: 1,
            input_layout: ConvLayout::ChannelsFramesWidth,
        },
        &stages,
        n_cu,
    )?;
    let input = packets.input;
    let convolution = packets.output;
    let output_shape = [
        chunks,
        packets.output_channels,
        packets.output_frames,
        packets.output_width,
    ];
    let rows = qwen_audio_rows(feature_frames);
    let mut prefix = packets.pack_ncfw_rows(rows)?.append_dense_f32(
        rows,
        DenseF32Stage {
            output: "act.qwen.conv_projection",
            weight: "thinker.audio_tower.conv_out.weight",
            bias: None,
            input_width: 7680,
            output_width: 1024,
            activation: DenseActivation::None,
            round_bf16: true,
            weight_type: DenseWeight::Bf16,
        },
    )?;
    let mut builder = packet::devbuild::Builder::new(prefix.model.n_cu);
    builder.set_tensor_dedup(true);
    builder.adopt_tensors(std::mem::take(&mut prefix.model.tensors));
    let valid_rows = builder.tensor("in.qwen.audio_valid_rows", 4);
    prefix.model.tensors = builder.tensors();
    let projection = prefix.output;
    let positions = qwen_positions(rows as usize, 1024, output_shape[3] as usize);
    prefix = prefix.append_initialized_add_f32(InitializedAddF32Stage {
        output: "act.qwen.positioned",
        addend: "const.qwen.audio_position",
        values: &positions,
        scale: 1.0,
        round_bf16: true,
    })?;
    let positioned = prefix.output;
    prefix = prefix.append_layer_norm_f32(
        rows,
        LayerNormF32Stage {
            output: "act.qwen.layers.0.self_attn_norm",
            gamma: Some("thinker.audio_tower.layers.0.self_attn_layer_norm.weight"),
            beta: Some("thinker.audio_tower.layers.0.self_attn_layer_norm.bias"),
            width: 1024,
            epsilon: 1e-5,
            round_bf16: true,
            ordered_statistics: true,
        },
    )?;
    let first_norm = prefix.output;
    let mut first_qkv = [0; 3];
    for (index, name) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
        prefix = prefix.append_dense_f32_from(
            first_norm,
            rows,
            DenseF32Stage {
                output: match index {
                    0 => "act.qwen.layers.0.query",
                    1 => "act.qwen.layers.0.key",
                    _ => "act.qwen.layers.0.value",
                },
                weight: &format!("thinker.audio_tower.layers.0.self_attn.{name}.weight"),
                bias: Some(&format!(
                    "thinker.audio_tower.layers.0.self_attn.{name}.bias"
                )),
                input_width: 1024,
                output_width: 1024,
                activation: DenseActivation::None,
                round_bf16: true,
                weight_type: DenseWeight::Bf16,
            },
        )?;
        first_qkv[index] = prefix.output;
    }
    prefix = prefix.append_grouped_attention_f32(
        first_qkv[0],
        first_qkv[1],
        first_qkv[2],
        rows,
        GroupedAttentionF32Stage {
            output: "act.qwen.layers.0.attention",
            valid_rows: Some(valid_rows),
            width: 1024,
            head_width: 64,
            group_rows: output_shape[3] * 8,
            round_score_bf16: true,
            round_probability_bf16: true,
            round_output_bf16: true,
        },
    )?;
    let first_attention = prefix.output;
    prefix = prefix.append_dense_f32(
        rows,
        DenseF32Stage {
            output: "act.qwen.layers.0.attention_projection",
            weight: "thinker.audio_tower.layers.0.self_attn.out_proj.weight",
            bias: Some("thinker.audio_tower.layers.0.self_attn.out_proj.bias"),
            input_width: 1024,
            output_width: 1024,
            activation: DenseActivation::None,
            round_bf16: true,
            weight_type: DenseWeight::Bf16,
        },
    )?;
    let attention_projection = prefix.output;
    prefix = prefix.append_scaled_add_f32(
        positioned,
        attention_projection,
        rows * 1024,
        ScaledAddF32Stage {
            output: "act.qwen.layers.0.attention_residual",
            scale: 1.0,
            round_bf16: true,
        },
    )?;
    let attention_residual = prefix.output;
    prefix = prefix.append_layer_norm_f32(
        rows,
        LayerNormF32Stage {
            output: "act.qwen.layers.0.final_norm",
            gamma: Some("thinker.audio_tower.layers.0.final_layer_norm.weight"),
            beta: Some("thinker.audio_tower.layers.0.final_layer_norm.bias"),
            width: 1024,
            epsilon: 1e-5,
            round_bf16: true,
            ordered_statistics: true,
        },
    )?;
    prefix = prefix.append_dense_f32(
        rows,
        DenseF32Stage {
            output: "act.qwen.layers.0.fc1",
            weight: "thinker.audio_tower.layers.0.fc1.weight",
            bias: Some("thinker.audio_tower.layers.0.fc1.bias"),
            input_width: 1024,
            output_width: 4096,
            activation: DenseActivation::GeluErfBf16,
            round_bf16: true,
            weight_type: DenseWeight::Bf16,
        },
    )?;
    prefix = prefix.append_dense_f32(
        rows,
        DenseF32Stage {
            output: "act.qwen.layers.0.fc2",
            weight: "thinker.audio_tower.layers.0.fc2.weight",
            bias: Some("thinker.audio_tower.layers.0.fc2.bias"),
            input_width: 4096,
            output_width: 1024,
            activation: DenseActivation::None,
            round_bf16: true,
            weight_type: DenseWeight::Bf16,
        },
    )?;
    let fc2 = prefix.output;
    prefix = prefix.append_scaled_add_f32(
        attention_residual,
        fc2,
        rows * 1024,
        ScaledAddF32Stage {
            output: "act.qwen.layers.0.output",
            scale: 1.0,
            round_bf16: true,
        },
    )?;
    let first_output = prefix.output;
    prefix = append_audio_transformer_layers(
        prefix,
        AudioTransformerSpec {
            rows,
            width: 1024,
            ffn_width: 4096,
            head_width: 64,
            group_rows: output_shape[3] * 8,
            valid_rows: Some(valid_rows),
            first_layer: 1,
            layers: 23,
            weight_prefix: "thinker.audio_tower",
            activation_prefix: "act.qwen",
        },
    )?;
    let transformer_output = prefix.output;
    prefix = prefix.append_layer_norm_f32(
        rows,
        LayerNormF32Stage {
            output: "act.qwen.ln_post",
            gamma: Some("thinker.audio_tower.ln_post.weight"),
            beta: Some("thinker.audio_tower.ln_post.bias"),
            width: 1024,
            epsilon: 1e-5,
            round_bf16: true,
            ordered_statistics: true,
        },
    )?;
    prefix = prefix.append_dense_f32(
        rows,
        DenseF32Stage {
            output: "act.qwen.proj1",
            weight: "thinker.audio_tower.proj1.weight",
            bias: Some("thinker.audio_tower.proj1.bias"),
            input_width: 1024,
            output_width: 1024,
            activation: DenseActivation::GeluErfBf16,
            round_bf16: true,
            weight_type: DenseWeight::Bf16,
        },
    )?;
    prefix = prefix.append_dense_f32(
        rows,
        DenseF32Stage {
            output: "act.qwen.audio_features",
            weight: "thinker.audio_tower.proj2.weight",
            bias: Some("thinker.audio_tower.proj2.bias"),
            input_width: 1024,
            output_width: 2048,
            activation: DenseActivation::None,
            round_bf16: true,
            weight_type: DenseWeight::Bf16,
        },
    )?;
    Ok(AudioEncoderPackets {
        rows,
        input,
        convolution,
        projection,
        positioned,
        first_norm,
        first_qkv,
        first_attention,
        first_output,
        transformer_output,
        output_shape,
        feature_frames,
        valid_rows,
        capacity_programs: BTreeMap::new(),
        prefix,
    })
}

pub fn qwen_audio_rows(frames: u32) -> u32 {
    13 * (frames / 100) + (frames % 100).div_ceil(8)
}

fn qwen_positions(rows: usize, width: usize, time: usize) -> Vec<f32> {
    let half = width / 2;
    let mut values = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let position = (row % time) as f32;
        for column in 0..width {
            let exponent = -10000.0f32.ln() * (column % half) as f32 / (half - 1) as f32;
            let angle = position * exponent.exp();
            values.push(round_bf16(if column < half {
                angle.sin()
            } else {
                angle.cos()
            }));
        }
    }
    values
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

#[derive(Clone, Copy)]
pub struct AudioTransformerSpec<'a> {
    pub rows: u32,
    pub width: u32,
    pub ffn_width: u32,
    pub head_width: u32,
    pub group_rows: u32,
    pub valid_rows: Option<u32>,
    pub first_layer: u32,
    pub layers: u32,
    pub weight_prefix: &'a str,
    pub activation_prefix: &'a str,
}

pub fn append_audio_transformer_layers(
    mut prefix: PacketPrefix,
    spec: AudioTransformerSpec<'_>,
) -> Result<PacketPrefix, String> {
    if spec.layers == 0
        || spec.width == 0
        || spec.ffn_width == 0
        || spec.first_layer.checked_add(spec.layers).is_none()
    {
        return Err("invalid Qwen audio transformer geometry".into());
    }
    let elements = spec
        .rows
        .checked_mul(spec.width)
        .ok_or("Qwen audio transformer tensor size overflows")?;
    for layer in spec.first_layer..spec.first_layer + spec.layers {
        let weights = format!("{}.layers.{layer}", spec.weight_prefix);
        let activations = format!("{}.layers.{layer}", spec.activation_prefix);
        let residual = prefix.output;
        prefix = prefix.append_layer_norm_f32(
            spec.rows,
            LayerNormF32Stage {
                output: &format!("{activations}.self_attn_norm"),
                gamma: Some(&format!("{weights}.self_attn_layer_norm.weight")),
                beta: Some(&format!("{weights}.self_attn_layer_norm.bias")),
                width: spec.width,
                epsilon: 1e-5,
                round_bf16: true,
                ordered_statistics: true,
            },
        )?;
        let normalized = prefix.output;
        let mut qkv = [0; 3];
        for (index, name) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
            prefix = prefix.append_dense_f32_from(
                normalized,
                spec.rows,
                DenseF32Stage {
                    output: &format!("{activations}.{}", ["query", "key", "value"][index]),
                    weight: &format!("{weights}.self_attn.{name}.weight"),
                    bias: Some(&format!("{weights}.self_attn.{name}.bias")),
                    input_width: spec.width,
                    output_width: spec.width,
                    activation: DenseActivation::None,
                    round_bf16: true,
                    weight_type: DenseWeight::Bf16,
                },
            )?;
            qkv[index] = prefix.output;
        }
        prefix = prefix.append_grouped_attention_f32(
            qkv[0],
            qkv[1],
            qkv[2],
            spec.rows,
            GroupedAttentionF32Stage {
                output: &format!("{activations}.attention"),
                valid_rows: spec.valid_rows,
                width: spec.width,
                head_width: spec.head_width,
                group_rows: spec.group_rows,
                round_score_bf16: true,
                round_probability_bf16: true,
                round_output_bf16: true,
            },
        )?;
        prefix = prefix.append_dense_f32(
            spec.rows,
            DenseF32Stage {
                output: &format!("{activations}.attention_projection"),
                weight: &format!("{weights}.self_attn.out_proj.weight"),
                bias: Some(&format!("{weights}.self_attn.out_proj.bias")),
                input_width: spec.width,
                output_width: spec.width,
                activation: DenseActivation::None,
                round_bf16: true,
                weight_type: DenseWeight::Bf16,
            },
        )?;
        let attention_projection = prefix.output;
        prefix = prefix.append_scaled_add_f32(
            residual,
            attention_projection,
            elements,
            ScaledAddF32Stage {
                output: &format!("{activations}.attention_residual"),
                scale: 1.0,
                round_bf16: true,
            },
        )?;
        let attention_residual = prefix.output;
        prefix = prefix.append_layer_norm_f32(
            spec.rows,
            LayerNormF32Stage {
                output: &format!("{activations}.final_norm"),
                gamma: Some(&format!("{weights}.final_layer_norm.weight")),
                beta: Some(&format!("{weights}.final_layer_norm.bias")),
                width: spec.width,
                epsilon: 1e-5,
                round_bf16: true,
                ordered_statistics: true,
            },
        )?;
        prefix = prefix.append_dense_f32(
            spec.rows,
            DenseF32Stage {
                output: &format!("{activations}.fc1"),
                weight: &format!("{weights}.fc1.weight"),
                bias: Some(&format!("{weights}.fc1.bias")),
                input_width: spec.width,
                output_width: spec.ffn_width,
                activation: DenseActivation::GeluErfBf16,
                round_bf16: true,
                weight_type: DenseWeight::Bf16,
            },
        )?;
        prefix = prefix.append_dense_f32(
            spec.rows,
            DenseF32Stage {
                output: &format!("{activations}.fc2"),
                weight: &format!("{weights}.fc2.weight"),
                bias: Some(&format!("{weights}.fc2.bias")),
                input_width: spec.ffn_width,
                output_width: spec.width,
                activation: DenseActivation::None,
                round_bf16: true,
                weight_type: DenseWeight::Bf16,
            },
        )?;
        let ffn = prefix.output;
        prefix = prefix.append_scaled_add_f32(
            attention_residual,
            ffn,
            elements,
            ScaledAddF32Stage {
                output: &format!("{activations}.output"),
                scale: 1.0,
                round_bf16: true,
            },
        )?;
    }
    Ok(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevOp;
    use packet::devbuild::{Builder, Model};

    #[test]
    fn emits_ordered_packet_layers() {
        let mut builder = Builder::new(4);
        let input = builder.tensor("positioned", 2 * 4 * 4);
        let prefix = PacketPrefix {
            model: Model {
                n_cu: 4,
                target: 0,
                tensors: builder.tensors(),
                progs: Vec::new(),
                kv_row_insts: Vec::new(),
                prog_t: Vec::new(),
                gen: Vec::new(),
            },
            programs: Vec::new(),
            input,
            output: input,
            input_shape: vec![2, 4],
        };
        let prefix = append_audio_transformer_layers(
            prefix,
            AudioTransformerSpec {
                rows: 2,
                width: 4,
                ffn_width: 8,
                head_width: 2,
                group_rows: 2,
                valid_rows: None,
                first_layer: 3,
                layers: 2,
                weight_prefix: "tower",
                activation_prefix: "act.audio",
            },
        )
        .unwrap();
        assert_eq!(prefix.programs.len(), 22);
        assert_eq!(prefix.model.progs.len(), 22);
        let weight = prefix
            .model
            .tensors
            .iter()
            .find(|tensor| tensor.name == "tower.layers.4.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(weight.bytes, 4 * 4 * 2);
        assert_eq!(prefix.model.progs[1].insts[0].i[7], 5);
        let layer_three = prefix
            .model
            .tensors
            .iter()
            .position(|tensor| tensor.name == "act.audio.layers.3.output")
            .unwrap() as u32;
        let layer_four_norm = &prefix.model.progs[11].insts[0];
        assert_eq!(
            DevOp::from_u16(layer_four_norm.op),
            Some(DevOp::LayerNormF32)
        );
        assert_eq!(layer_four_norm.t[1], layer_three);
        assert_eq!(
            prefix.model.tensors[prefix.output as usize].name,
            "act.audio.layers.4.output"
        );
    }

    #[test]
    fn lowers_complete_audio_encoder_as_backend_neutral_packets() {
        let packets = lower_audio_encoder(50, 4).unwrap();
        assert_eq!(packets.rows, 7);
        assert_eq!(packets.output_shape, [1, 480, 16, 7]);
        assert_eq!(packets.prefix.input_shape, [1, 128, 50]);
        assert_eq!(packets.prefix.programs.len(), 271);
        assert_eq!(
            packets.prefix.model.tensors[packets.input as usize].name,
            "in.conv2d"
        );
        assert_eq!(
            packets.prefix.model.tensors[packets.prefix.output as usize].name,
            "act.qwen.audio_features"
        );
        assert!(packets
            .prefix
            .model
            .tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with("thinker.audio_tower."))
            .all(|tensor| tensor.init.is_none()));

        let section = packets.pipeline_section(50).unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let pipeline = &metadata.pipelines[0];
        assert_eq!(pipeline.name, "audio.encode");
        assert_eq!(pipeline.driver, "forward.v1");
        assert_eq!(pipeline.parameters["input_frames"], 50);
        assert_eq!(pipeline.parameters["output_rows"], 7);
        assert_eq!(pipeline.parameters["feature_bins"], 128);
        assert_eq!(pipeline.parameters["output_width"], 2048);
        assert_eq!(pipeline.tensors["valid_rows"].shape, [1]);
    }

    #[test]
    fn merges_audio_capacities_with_one_tensor_table() {
        let mut packets = lower_audio_encoder(200, 4).unwrap();
        packets
            .merge_capacity(lower_audio_encoder(100, 4).unwrap())
            .unwrap();
        assert_eq!(packets.prefix.model.progs.len(), 542);
        let section = packets.pipeline_section(200).unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let programs = &metadata.pipelines[0].programs;
        assert!(programs.contains_key("forward.100.270"));
        assert!(programs.contains_key("forward.200.270"));
    }
}
