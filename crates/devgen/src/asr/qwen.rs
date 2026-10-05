use crate::conv2d::{
    self, Conv2dSpec, Conv2dStage, ConvActivation, ConvKind, ConvLayout, ConvWeight,
};
use crate::pipeline::{
    DenseActivation, DenseF32Stage, DenseLayerNorm, DenseSplit, DenseWeight, GroupedAttentionF32Stage,
    InitializedAddF32Stage, LayerNormF32Stage, PacketPrefix, RowStatsF32Stage, ScaledAddF32Stage,
    TensorRef,
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
    groups: u32,
    split_rows: u32,
    /// Utterances packed chunk by chunk (group-table attention) instead of one per capacity.
    packed: bool,
    capacity_programs: BTreeMap<u32, Vec<usize>>,
    packed_programs: BTreeMap<u32, Vec<usize>>,
    frontend: Option<(u32, BTreeMap<String, u64>, [u64; 2])>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioWeightDType {
    Bf16,
    F32,
}

impl AudioEncoderPackets {
    pub fn merge_capacity(&mut self, mut bucket: AudioEncoderPackets) -> Result<(), String> {
        let capacity = bucket.feature_frames;
        let (programs, own) = if bucket.packed {
            (&self.packed_programs, self.packed)
        } else {
            (&self.capacity_programs, !self.packed)
        };
        if capacity == 0
            || (own && capacity >= self.feature_frames)
            || programs.contains_key(&capacity)
            || bucket.rows > self.rows
            || bucket.prefix.model.n_cu != self.prefix.model.n_cu
            || bucket.prefix.model.target != self.prefix.model.target
            || !bucket.prefix.model.gen.is_empty()
            || bucket.prefix.model.tensors.len() != self.prefix.model.tensors.len()
            || !bucket.capacity_programs.is_empty()
            || !bucket.packed_programs.is_empty()
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
        if bucket.packed {
            self.packed_programs.insert(capacity / 100, merged);
        } else {
            self.capacity_programs.insert(capacity, merged);
        }
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
        let own = if self.packed { format!("packed.{}", feature_frames / 100) } else { format!("forward.{feature_frames}") };
        for (stage, &program) in self.prefix.programs.iter().enumerate() {
            pipeline.programs.insert(format!("{own}.{stage}"), program as u32);
        }
        for (&capacity, programs) in &self.capacity_programs {
            for (stage, &program) in programs.iter().enumerate() {
                pipeline
                    .programs
                    .insert(format!("forward.{capacity}.{stage}"), program as u32);
            }
        }
        // `packed.{chunks}.{stage}`: utterances packed chunk by chunk, with the `groups` table.
        for (&chunks, programs) in &self.packed_programs {
            for (stage, &program) in programs.iter().enumerate() {
                pipeline
                    .programs
                    .insert(format!("packed.{chunks}.{stage}"), program as u32);
            }
        }
        if self.packed {
            // The unbucketed forward sequence is the largest single-utterance capacity.
            let (_, largest) = self
                .capacity_programs
                .last_key_value()
                .ok_or("packed Qwen audio packet has no single-utterance capacity")?;
            for (stage, &program) in largest.iter().enumerate() {
                pipeline.programs.insert(format!("forward.{stage}"), program as u32);
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
        pipeline
            .parameters
            .insert("attention.window_rows".into(), u64::from(self.output_shape[3] * 8));
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
        pipeline.tensors.insert(
            "split_rows".into(),
            plow_asset::packet_pipeline::PipelineTensor {
                name: self.prefix.model.tensors[self.split_rows as usize].name.clone(),
                dtype: PipelineDType::U32,
                shape: vec![self.prefix.model.tensors[self.split_rows as usize].bytes / 4],
            },
        );
        pipeline.tensors.insert(
            "groups".into(),
            plow_asset::packet_pipeline::PipelineTensor {
                name: self.prefix.model.tensors[self.groups as usize].name.clone(),
                dtype: PipelineDType::U32,
                shape: vec![self.prefix.model.tensors[self.groups as usize].bytes / 4],
            },
        );
        if let Some((tensor, parameters, shape)) = &self.frontend {
            pipeline.parameters.extend(parameters.iter().map(|(k, v)| (k.clone(), *v)));
            pipeline.tensors.insert(
                "audio.frontend.filterbank".into(),
                plow_asset::packet_pipeline::PipelineTensor {
                    name: self.prefix.model.tensors[*tensor as usize].name.clone(),
                    dtype: PipelineDType::F32,
                    shape: shape.to_vec(),
                },
            );
        }
        section.data = serde_json::to_vec(&metadata).map_err(|error| error.to_string())?;
        Ok(section)
    }

    /// Embed the host frontend (its parameters and filterbank) in the encoder packet.
    pub fn embed_frontend(&mut self, frontend: WhisperFrontend) {
        let bytes: Vec<u8> = frontend.filterbank.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = self.prefix.model.tensors.len() as u32;
        self.prefix.model.tensors.push(packet::devbuild::TensorDecl {
            name: "const.audio.log_mel_filterbank".into(),
            bytes: bytes.len() as u64,
            init: Some(bytes),
        });
        self.frontend = Some((tensor, frontend.parameters, [u64::from(frontend.bins), u64::from(frontend.spectrum_bins)]));
    }
}

pub fn lower_audio_encoder(feature_frames: u32, n_cu: u32) -> Result<AudioEncoderPackets, String> {
    if !(50..=3000).contains(&feature_frames) {
        return Err("invalid Qwen audio encoder geometry".into());
    }
    lower(feature_frames, n_cu, false)
}

/// `chunks` 100-frame chunks of any number of utterances, each starting on a chunk; the `groups`
/// table lists every utterance's attention windows `(first row, valid rows)`.
pub fn lower_packed_audio_encoder(chunks: u32, n_cu: u32) -> Result<AudioEncoderPackets, String> {
    if chunks == 0 || chunks > 1024 {
        return Err("invalid packed Qwen audio encoder geometry".into());
    }
    lower(chunks * 100, n_cu, true)
}

fn lower(feature_frames: u32, n_cu: u32, packed: bool) -> Result<AudioEncoderPackets, String> {
    if n_cu == 0 {
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
            weight_type: ConvWeight::Bf16InF32,
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
    let mut prefix = packets.pack_ncfw_rows(rows)?;
    let mut builder = packet::devbuild::Builder::new(prefix.model.n_cu);
    builder.set_tensor_dedup(true);
    builder.adopt_tensors(std::mem::take(&mut prefix.model.tensors));
    let valid_rows = builder.tensor("in.qwen.audio_valid_rows", 4);
    // At most one window per chunk (an utterance takes at least one).
    let groups = builder.tensor("in.qwen.audio_groups", 4 * (1 + 2 * u64::from(chunks)));
    // Per row: the rows of its utterance's single-utterance capacity, whose split-K it reproduces.
    let split_rows = builder.tensor("in.qwen.audio_split_rows", 4 * u64::from(rows));
    let attention_rows = if packed { groups } else { valid_rows };
    let split = if packed { DenseSplit::Reference(split_rows) } else { DenseSplit::Parallel };
    prefix.model.tensors = builder.tensors();
    prefix = prefix.append_dense_f32(
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
            input_bf16_exact: true,
            layer_norm: None,
            split,
        },
    )?;
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
                input_bf16_exact: true,
                layer_norm: None,
                split,
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
            valid_rows: Some(attention_rows),
            group_table: packed,
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
            input_bf16_exact: true,
            layer_norm: None,
            split,
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
            input_bf16_exact: true,
            layer_norm: None,
            split,
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
            input_bf16_exact: true,
            layer_norm: None,
            split,
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
            valid_rows: Some(attention_rows),
            group_table: packed,
            split,
            first_layer: 1,
            layers: 23,
            weight_prefix: "thinker.audio_tower",
            activation_prefix: "act.qwen",
            fuse_layer_norm: false,
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
            input_bf16_exact: true,
            layer_norm: None,
            split,
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
            input_bf16_exact: true,
            layer_norm: None,
            split,
        },
    )?;
    fuse_programs(&mut prefix, FUSED_OPS)?;
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
        groups,
        split_rows,
        packed,
        capacity_programs: BTreeMap::new(),
        packed_programs: BTreeMap::new(),
        frontend: None,
        prefix,
    })
}

/// Ops per fused encoder program: a sequence pays a launch per program rather than per op, and
/// at the largest bucket one program still ends every few ms, so other streams' kernels (the
/// decoder's) interleave.
const FUSED_OPS: usize = 12;

/// Fold runs of `ops` consecutive single-segment programs into one program each, every op waiting
/// for the previous one to retire: the order separate launches gave.
fn fuse_programs(prefix: &mut PacketPrefix, ops: usize) -> Result<(), String> {
    use packet::dev::DevOp;
    let model = &mut prefix.model;
    let n_cu = model.n_cu as usize;
    let mut progs = Vec::new();
    let mut prog_t = Vec::new();
    let mut programs = Vec::new();
    let mut run: Vec<usize> = Vec::new();
    let mut flush = |run: &mut Vec<usize>, model: &packet::devbuild::Model| -> Result<(), String> {
        if run.is_empty() {
            return Ok(());
        }
        let mut builder = packet::devbuild::Builder::new(model.n_cu);
        builder.set_tensor_dedup(true);
        builder.adopt_tensors(model.tensors.clone());
        let mut previous: Option<u32> = None;
        for &p in run.iter() {
            let program = &model.progs[p];
            if program.gq_seg_ofs.len() != 2 || program.l2_domains != 0 || program.stream_ofs.len() < n_cu {
                return Err("Qwen audio program is not a single unplaced segment".into());
            }
            for (k, inst) in program.insts.iter().enumerate() {
                let mut cus = vec![u32::MAX; usize::from(inst.blocks)];
                for cu in 0..n_cu {
                    let start = program.stream_ofs[cu] as usize;
                    for entry in &program.stream[start..start + program.stream_len[cu] as usize] {
                        if entry.inst as usize == k {
                            *cus.get_mut(entry.slice as usize).ok_or("Qwen audio slice out of range")? = cu as u32;
                        }
                    }
                }
                if cus.contains(&u32::MAX) {
                    return Err("Qwen audio op has an unplaced slice".into());
                }
                let op = DevOp::from_u16(inst.op).ok_or("Qwen audio op is unknown")?;
                let deps: Vec<u32> = previous.into_iter().collect();
                previous = Some(builder.emit(op, cus, &deps, |fused| {
                    fused.t = inst.t;
                    fused.i = inst.i;
                    fused.f = inst.f;
                    fused.j = inst.j;
                }));
            }
        }
        let program = builder.finish();
        if program.gq_seg_ofs.len() != 2 {
            return Err("fused Qwen audio program is segmented".into());
        }
        programs.push(progs.len());
        prog_t.push(model.prog_t[run[0]]);
        progs.push(program);
        run.clear();
        Ok(())
    };
    for &p in &prefix.programs {
        run.push(p);
        if run.len() == ops {
            flush(&mut run, model)?;
        }
    }
    flush(&mut run, model)?;
    model.progs = progs;
    model.prog_t = prog_t;
    prefix.programs = programs;
    Ok(())
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
    pub group_table: bool,
    pub split: DenseSplit,
    pub first_layer: u32,
    pub layers: u32,
    pub weight_prefix: &'a str,
    pub activation_prefix: &'a str,
    /// RowStatsF32 + DenseGemmF32 LayerNorm prologue instead of LayerNormF32 -> DenseGemmF32
    /// (bit-identical). Off by default: a packet runtime must implement both (op 204, dense
    /// flag bits 5/6); the Metal interpreter does not yet.
    pub fuse_layer_norm: bool,
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
        let attn_gamma = format!("{weights}.self_attn_layer_norm.weight");
        let attn_beta = format!("{weights}.self_attn_layer_norm.bias");
        let (normalized, qkv_norm) = if spec.fuse_layer_norm {
            prefix = prefix.append_row_stats_f32(
                residual,
                RowStatsF32Stage {
                    output: TensorRef::Named(&format!("{activations}.self_attn_norm_stats")),
                    rows: spec.rows,
                    width: spec.width,
                    epsilon: 1e-5,
                    ordered_statistics: true,
                },
            )?;
            let stats = prefix.output;
            (
                residual,
                Some(DenseLayerNorm {
                    stats,
                    gamma: Some(TensorRef::Named(&attn_gamma)),
                    beta: Some(TensorRef::Named(&attn_beta)),
                    round_bf16: true,
                }),
            )
        } else {
            prefix = prefix.append_layer_norm_f32(
                spec.rows,
                LayerNormF32Stage {
                    output: &format!("{activations}.self_attn_norm"),
                    gamma: Some(&attn_gamma),
                    beta: Some(&attn_beta),
                    width: spec.width,
                    epsilon: 1e-5,
                    round_bf16: true,
                    ordered_statistics: true,
                },
            )?;
            (prefix.output, None)
        };
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
                    input_bf16_exact: true,
                    layer_norm: qkv_norm,
                    split: spec.split,
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
                group_table: spec.group_table,
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
                input_bf16_exact: true,
                layer_norm: None,
                split: spec.split,
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
        let ffn_gamma = format!("{weights}.final_layer_norm.weight");
        let ffn_beta = format!("{weights}.final_layer_norm.bias");
        let fc1_norm = if spec.fuse_layer_norm {
            prefix = prefix.append_row_stats_f32(
                attention_residual,
                RowStatsF32Stage {
                    output: TensorRef::Named(&format!("{activations}.final_norm_stats")),
                    rows: spec.rows,
                    width: spec.width,
                    epsilon: 1e-5,
                    ordered_statistics: true,
                },
            )?;
            let stats = prefix.output;
            prefix.output = attention_residual;
            Some(DenseLayerNorm {
                stats,
                gamma: Some(TensorRef::Named(&ffn_gamma)),
                beta: Some(TensorRef::Named(&ffn_beta)),
                round_bf16: true,
            })
        } else {
            prefix = prefix.append_layer_norm_f32(
                spec.rows,
                LayerNormF32Stage {
                    output: &format!("{activations}.final_norm"),
                    gamma: Some(&ffn_gamma),
                    beta: Some(&ffn_beta),
                    width: spec.width,
                    epsilon: 1e-5,
                    round_bf16: true,
                    ordered_statistics: true,
                },
            )?;
            None
        };
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
                input_bf16_exact: true,
                layer_norm: fc1_norm,
                split: spec.split,
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
                input_bf16_exact: true,
                layer_norm: None,
                split: spec.split,
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
    fn fused_layer_norm_emits_row_stats_and_dense_prologue() {
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
                group_table: false,
                split: DenseSplit::Parallel,
                first_layer: 0,
                layers: 1,
                weight_prefix: "tower",
                activation_prefix: "act.audio",
                fuse_layer_norm: true,
            },
        )
        .unwrap();
        let ops: Vec<_> =
            prefix.model.progs.iter().map(|p| DevOp::from_u16(p.insts[0].op).unwrap()).collect();
        assert_eq!(ops[0], DevOp::RowStatsF32);
        assert!(!ops.contains(&DevOp::LayerNormF32));
        let stats = prefix.model.progs[0].insts[0].t[0];
        let q = &prefix.model.progs[1].insts[0];
        assert_eq!(DevOp::from_u16(q.op), Some(DevOp::DenseGemmF32));
        assert_eq!(q.i[7], 1 | 4 | 8 | 32 | 64);
        assert_eq!(q.t[1], input);
        assert_eq!(q.t[5], stats);
        assert_eq!(prefix.model.progs[0].insts[0].i[2], 2);
    }

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
                group_table: false,
                split: DenseSplit::Parallel,
                first_layer: 3,
                layers: 2,
                weight_prefix: "tower",
                activation_prefix: "act.audio",
                fuse_layer_norm: false,
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
        assert_eq!(prefix.model.progs[1].insts[0].i[7], 13);
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
        assert_eq!(packets.prefix.programs.len(), 271usize.div_ceil(FUSED_OPS));
        let ops: usize = packets.prefix.programs.iter().map(|&p| packets.prefix.model.progs[p].insts.len()).sum();
        assert_eq!(ops, 273);
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
        assert_eq!(packets.prefix.model.progs.len(), 2 * 271usize.div_ceil(FUSED_OPS));
        let section = packets.pipeline_section(200).unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let programs = &metadata.pipelines[0].programs;
        let last = 271usize.div_ceil(FUSED_OPS) - 1;
        assert!(programs.contains_key(&format!("forward.100.{last}")));
        assert!(programs.contains_key(&format!("forward.200.{last}")));
    }

    #[test]
    fn packed_buckets_share_the_largest_tensor_table() {
        let mut packets = lower_packed_audio_encoder(4, 4).unwrap();
        packets.merge_capacity(lower_packed_audio_encoder(2, 4).unwrap()).unwrap();
        packets.merge_capacity(lower_audio_encoder(300, 4).unwrap()).unwrap();
        packets.merge_capacity(lower_audio_encoder(100, 4).unwrap()).unwrap();
        assert!(packets.merge_capacity(lower_audio_encoder(100, 4).unwrap()).is_err());
        let attention = packets
            .prefix
            .model
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .find(|i| DevOp::from_u16(i.op) == Some(DevOp::GroupedAttentionF32))
            .unwrap();
        assert_eq!(attention.i[4], 7 | 8);
        assert_eq!(attention.t[4], packets.groups);
        let dense: Vec<_> = packets.packed_programs[&2]
            .iter()
            .flat_map(|&p| &packets.prefix.model.progs[p].insts)
            .filter(|i| DevOp::from_u16(i.op) == Some(DevOp::DenseGemmF32))
            .collect();
        assert!(dense.iter().all(|i| i.i[7] & 128 != 0 && i.t[4] == packets.split_rows && i.j[0] == 4));
        let single = &packets.prefix.model.progs[packets.capacity_programs[&300][0]].insts[4];
        assert_eq!(DevOp::from_u16(single.op), Some(DevOp::DenseGemmF32));
        assert_eq!(single.i[7] & 128, 0);
        let section = packets.pipeline_section(400).unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        let pipeline = &metadata.pipelines[0];
        let last = 271usize.div_ceil(FUSED_OPS) - 1;
        for role in ["packed.4", "packed.2", "forward.300", "forward.100"] {
            assert!(pipeline.programs.contains_key(&format!("{role}.{last}")), "{role}");
        }
        assert_eq!(pipeline.programs["forward.0"], pipeline.programs["forward.300.0"]);
        assert_eq!(pipeline.tensors["groups"].shape, [9]);
        assert_eq!(pipeline.parameters["output_rows"], 52);
    }
}

/// Whisper feature extraction as the generic packet log-mel frontend: parameters plus the
/// Slaney mel filterbank (`[bins][n_fft/2+1]`), from `preprocessor_config.json`.
pub struct WhisperFrontend {
    pub parameters: BTreeMap<String, u64>,
    pub filterbank: Vec<f32>,
    pub bins: u32,
    pub spectrum_bins: u32,
}

pub fn whisper_frontend(checkpoint: &std::path::Path) -> Result<WhisperFrontend, String> {
    let text = std::fs::read(checkpoint.join("preprocessor_config.json")).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
    if v["feature_extractor_type"] != "WhisperFeatureExtractor" || v["dither"].as_f64().unwrap_or(0.0) != 0.0 {
        return Err("audio frontend is not a dither-free WhisperFeatureExtractor".into());
    }
    let get = |k: &str| v[k].as_u64().ok_or(format!("preprocessor_config.json: {k} missing"));
    let (bins, hop, fft, samples) = (get("feature_size")?, get("hop_length")?, get("n_fft")?, get("n_samples")?);
    let rate = v["sampling_rate"].as_u64().unwrap_or(16_000);
    let spectrum = fft / 2 + 1;
    let hz_to_mel = |hz: f64| if hz < 1000.0 { hz * 3.0 / 200.0 } else { 15.0 + (hz / 1000.0).ln() * 27.0 / 6.4f64.ln() };
    let mel_to_hz = |mel: f64| if mel < 15.0 { mel * 200.0 / 3.0 } else { 1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp() };
    let top = hz_to_mel(rate as f64 / 2.0);
    let points: Vec<f64> = (0..bins + 2).map(|i| mel_to_hz(top * i as f64 / (bins + 1) as f64)).collect();
    let mut filterbank = vec![0f32; (bins * spectrum) as usize];
    for m in 0..bins as usize {
        for k in 0..spectrum as usize {
            let hz = k as f64 * rate as f64 / fft as f64;
            let rise = (hz - points[m]) / (points[m + 1] - points[m]);
            let fall = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
            filterbank[m * spectrum as usize + k] = (rise.min(fall).max(0.0) * 2.0 / (points[m + 2] - points[m])) as f32;
        }
    }
    let f = |x: f32| u64::from(x.to_bits());
    let parameters = BTreeMap::from([
        ("audio.frontend.kind".into(), 1),
        ("audio.sample_rate".into(), rate),
        ("audio.frontend.fft".into(), fft),
        ("audio.frontend.window".into(), fft),
        ("audio.frontend.hop".into(), hop),
        ("audio.frontend.bins".into(), bins),
        ("audio.frontend.preemphasis_f32".into(), f(0.0)),
        ("audio.frontend.center_window".into(), 0),
        ("audio.frontend.periodic_hann".into(), 1),
        ("audio.frontend.normalize_per_feature".into(), 0),
        ("audio.frontend.mask_invalid_frames".into(), 0),
        ("audio.frontend.log_guard_f32".into(), f(1e-10)),
        ("audio.frontend.pad_reflect".into(), 1),
        ("audio.frontend.drop_last_frame".into(), 1),
        ("audio.frontend.log10".into(), 1),
        ("audio.frontend.log_floor".into(), 1),
        ("audio.frontend.dynamic_range_f32".into(), f(8.0)),
        ("audio.frontend.scale_f32".into(), f(0.25)),
        ("audio.frontend.shift_f32".into(), f(4.0)),
        ("audio.min_samples".into(), rate / 2),
        ("audio.max_samples".into(), samples),
        // Encoder input: 100-frame chunks, each ceil(len / 8) output rows (three stride-2 convs).
        ("input.chunk_frames".into(), 100),
        ("input.round_bf16".into(), 1),
        ("encoder.frame_stride".into(), 8),
    ]);
    Ok(WhisperFrontend { parameters, filterbank, bins: bins as u32, spectrum_bins: spectrum as u32 })
}

pub const ENCODER_PACKET: &str = "encoder.pkt";

/// Host contract of the Qwen3-ASR decoder for the generic audio-LM driver: prompt layout,
/// audio marker, output markers, languages and stop ids, as packet strings and parameters.
pub fn audio_lm_contract(
    checkpoint: &std::path::Path,
) -> Result<(BTreeMap<String, u64>, BTreeMap<String, String>), String> {
    let json = |file: &str| -> Result<serde_json::Value, String> {
        serde_json::from_slice(&std::fs::read(checkpoint.join(file)).map_err(|e| format!("{file}: {e}"))?)
            .map_err(|e| format!("{file}: {e}"))
    };
    let config = json("config.json")?;
    let thinker = &config["thinker_config"];
    let audio_token = thinker["audio_token_id"].as_u64().ok_or("audio_token_id missing")?;
    let mut stops = Vec::new();
    for file in ["generation_config.json", "config.json"] {
        let Ok(v) = json(file) else { continue };
        match &v["eos_token_id"] {
            serde_json::Value::Number(n) => stops.extend(n.as_u64()),
            serde_json::Value::Array(a) => stops.extend(a.iter().filter_map(|x| x.as_u64())),
            _ => {}
        }
        if !stops.is_empty() {
            break;
        }
    }
    if stops.is_empty() {
        return Err("no EOS ids".into());
    }
    let languages: Vec<String> =
        serde_json::from_value(config["support_languages"].clone()).map_err(|e| format!("support_languages: {e}"))?;
    const ALIASES: &[(&str, &str)] = &[
        ("zh", "Chinese"), ("en", "English"), ("yue", "Cantonese"), ("ar", "Arabic"), ("de", "German"),
        ("fr", "French"), ("es", "Spanish"), ("pt", "Portuguese"), ("id", "Indonesian"), ("it", "Italian"),
        ("ko", "Korean"), ("ru", "Russian"), ("th", "Thai"), ("vi", "Vietnamese"), ("ja", "Japanese"),
        ("tr", "Turkish"), ("hi", "Hindi"), ("ms", "Malay"), ("nl", "Dutch"), ("sv", "Swedish"),
        ("da", "Danish"), ("fi", "Finnish"), ("pl", "Polish"), ("cs", "Czech"), ("fil", "Filipino"),
        ("fa", "Persian"), ("el", "Greek"), ("hu", "Hungarian"), ("mk", "Macedonian"), ("ro", "Romanian"),
    ];
    let pre = json("preprocessor_config.json").unwrap_or_default();
    let mut parameters = BTreeMap::from([
        ("audio.token_id".into(), audio_token),
        ("audio.sample_rate".into(), pre["sampling_rate"].as_u64().unwrap_or(16_000)),
        ("audio.max_seconds".into(), pre["chunk_length"].as_u64().unwrap_or(30)),
        ("output.max_tokens".into(), 1024),
        ("prompt.context_max_tokens".into(), 256),
        ("stop.count".into(), stops.len() as u64),
    ]);
    for (i, id) in stops.iter().enumerate() {
        parameters.insert(format!("stop.{i}"), *id);
    }
    let strings = BTreeMap::from([
        (
            "prompt.messages".into(),
            r#"[{"role":"system","content":"{context}"},{"role":"user","content":[{"type":"audio"}]}]"#.into(),
        ),
        ("audio.marker".into(), "<|audio_pad|>".into()),
        ("encoder.packet".into(), ENCODER_PACKET.into()),
        ("prompt.language_suffix".into(), "language {language}<asr_text>".into()),
        ("prompt.context_forbidden".into(), "<|\n<asr_text>".into()),
        ("output.text_marker".into(), "<asr_text>".into()),
        ("output.language_prefix".into(), "language ".into()),
        ("output.language_none".into(), "none".into()),
        ("languages".into(), languages.join("\n")),
        ("language.aliases".into(), ALIASES.iter().map(|(a, b)| format!("{a}={b}")).collect::<Vec<_>>().join("\n")),
    ]);
    Ok((parameters, strings))
}

/// Merge extra parameters and strings into every pipeline of a metadata section.
pub fn extend_pipeline_section(
    section: &mut packet::devbuild::SectionData,
    parameters: &BTreeMap<String, u64>,
    strings: &BTreeMap<String, String>,
) -> Result<(), String> {
    let mut metadata: plow_asset::packet_pipeline::PacketPipelines =
        serde_json::from_slice(&section.data).map_err(|e| e.to_string())?;
    for pipeline in &mut metadata.pipelines {
        pipeline.parameters.extend(parameters.iter().map(|(k, v)| (k.clone(), *v)));
        pipeline.strings.extend(strings.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    section.data = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
    Ok(())
}
