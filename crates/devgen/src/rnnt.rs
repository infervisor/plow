//! Backend-neutral lowering for recurrent neural network transducer heads.

use packet::dev::DevOp;
use packet::devbuild::{Builder, Model, TensorDecl};
use std::collections::BTreeMap;

use crate::asr::frontend::LogMelSpec;
use crate::pipeline::PacketPrefix;

#[derive(Clone, Copy)]
pub struct LinearWeights<'a> {
    pub weight: &'a str,
    pub bias: &'a str,
}

#[derive(Clone, Copy)]
pub struct LstmWeights<'a> {
    pub input: LinearWeights<'a>,
    pub recurrent: LinearWeights<'a>,
}

pub struct RnntWeights<'a> {
    pub prompt_in: LinearWeights<'a>,
    pub prompt_out: LinearWeights<'a>,
    pub encoder: LinearWeights<'a>,
    pub embedding: &'a str,
    pub lstm: &'a [LstmWeights<'a>],
    pub predictor: LinearWeights<'a>,
    pub output: LinearWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct RnntSpec {
    pub frames: u32,
    pub encoder_width: u32,
    pub prompt_count: u32,
    pub prompt_width: u32,
    pub vocabulary: u32,
    pub predictor_width: u32,
    pub joint_width: u32,
    pub joint_batch: u32,
}

struct EmbeddedLogMelFrontend {
    spec: LogMelSpec,
    tensor: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RnntJointPrograms {
    pub frames: u32,
    pub banks: [usize; 2],
}

pub struct RnntPackets {
    pub model: Model,
    pub encoder_programs: Vec<usize>,
    pub predictor_programs: [usize; 2],
    pub joint_programs: Vec<RnntJointPrograms>,
    spec: RnntSpec,
    input: u32,
    input_shape: Vec<u64>,
    host_prompt: bool,
    compiled_prompt_index: Option<u32>,
    frontend: Option<EmbeddedLogMelFrontend>,
    encoder_buckets: BTreeMap<u32, Vec<usize>>,
}

impl RnntPackets {
    pub fn merge_encoder_bucket(
        &mut self,
        input_frames: u32,
        mut bucket: RnntPackets,
    ) -> Result<(), String> {
        if input_frames == 0
            || self.encoder_buckets.contains_key(&input_frames)
            || bucket.spec.frames > self.spec.frames
            || bucket.spec.encoder_width != self.spec.encoder_width
            || bucket.spec.prompt_count != self.spec.prompt_count
            || bucket.spec.prompt_width != self.spec.prompt_width
            || bucket.spec.vocabulary != self.spec.vocabulary
            || bucket.spec.predictor_width != self.spec.predictor_width
            || bucket.spec.joint_width != self.spec.joint_width
            || bucket.spec.joint_batch != self.spec.joint_batch
            || bucket.host_prompt != self.host_prompt
            || bucket.compiled_prompt_index != self.compiled_prompt_index
            || bucket.frontend.is_some()
            || self.frontend.is_some()
            || bucket.model.n_cu != self.model.n_cu
            || bucket.model.target != self.model.target
            || !bucket.model.gen.is_empty()
            || bucket.model.tensors.len() != self.model.tensors.len()
        {
            return Err("incompatible RNNT encoder bucket".into());
        }
        for (base, candidate) in self.model.tensors.iter_mut().zip(&bucket.model.tensors) {
            if base.name != candidate.name || base.bytes < candidate.bytes {
                return Err(format!(
                    "RNNT encoder bucket tensor {:?} is incompatible",
                    candidate.name
                ));
            }
            if let Some(init) = &candidate.init {
                match &base.init {
                    Some(existing) if existing == init => {}
                    None if base.bytes == candidate.bytes => base.init = Some(init.clone()),
                    _ => {
                        return Err(format!(
                            "RNNT encoder bucket tensor {:?} has different initialization",
                            candidate.name
                        ))
                    }
                }
            }
        }
        let mut programs: Vec<_> = bucket.model.progs.drain(..).map(Some).collect();
        let mut merged = Vec::with_capacity(bucket.encoder_programs.len());
        for program in bucket.encoder_programs {
            let value = programs
                .get_mut(program)
                .and_then(Option::take)
                .ok_or("RNNT encoder bucket program is missing")?;
            let index = self.model.progs.len();
            self.model.progs.push(value);
            self.model.prog_t.push(
                *bucket
                    .model
                    .prog_t
                    .get(program)
                    .ok_or("RNNT encoder bucket program shape is missing")?,
            );
            merged.push(index);
        }
        self.encoder_buckets.insert(input_frames, merged);
        Ok(())
    }

    pub fn embed_log_mel_frontend(
        &mut self,
        spec: LogMelSpec,
        filterbank: &[f32],
    ) -> Result<(), String> {
        let spectrum_bins = spec.fft / 2 + 1;
        let elements = u64::from(spec.bins)
            .checked_mul(u64::from(spectrum_bins))
            .ok_or("log-mel filterbank size overflows")?;
        if self.frontend.is_some()
            || spec.sample_rate == 0
            || !spec.fft.is_power_of_two()
            || spec.window < 2
            || spec.window > spec.fft
            || spec.hop == 0
            || spec.bins == 0
            || spec.min_samples == 0
            || spec.min_samples > spec.max_samples
            || !spec.preemphasis.is_finite()
            || !spec.log_guard.is_finite()
            || spec.log_guard <= 0.0
            || filterbank.len() as u64 != elements
            || filterbank
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("invalid embedded log-mel frontend".into());
        }
        let bytes: Vec<u8> = filterbank
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensor = self.model.tensors.len() as u32;
        self.model.tensors.push(TensorDecl {
            name: "const.audio.log_mel_filterbank".into(),
            bytes: bytes.len() as u64,
            init: Some(bytes),
        });
        self.frontend = Some(EmbeddedLogMelFrontend { spec, tensor });
        Ok(())
    }

    pub fn embed_weights(
        &mut self,
        mut resolve: impl FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<(), String> {
        for tensor in &mut self.model.tensors {
            if packet::names::is_checkpoint_weight(&tensor.name) && tensor.init.is_none() {
                let bytes = resolve(&tensor.name)?;
                if bytes.len() as u64 != tensor.bytes {
                    return Err(format!(
                        "tensor {} has {} bytes, expected {}",
                        tensor.name,
                        bytes.len(),
                        tensor.bytes
                    ));
                }
                tensor.init = Some(bytes);
            }
        }
        Ok(())
    }

    pub fn pipeline_section(
        &self,
        blank_id: u32,
        max_symbols_per_frame: u32,
        prompt_index: u32,
    ) -> Result<packet::devbuild::SectionData, String> {
        self.pipeline_section_with_frame_transform(
            blank_id,
            max_symbols_per_frame,
            prompt_index,
            0,
            &[],
            0,
        )
    }

    pub fn pipeline_section_with_frame_transform(
        &self,
        blank_id: u32,
        max_symbols_per_frame: u32,
        prompt_index: u32,
        input_frames: u32,
        stages: &[[u32; 4]],
        trailing_frames: u32,
    ) -> Result<packet::devbuild::SectionData, String> {
        use plow_asset::packet_pipeline::{
            PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
        };
        use std::collections::BTreeMap;

        if blank_id >= self.spec.vocabulary
            || max_symbols_per_frame == 0
            || prompt_index >= self.spec.prompt_count
        {
            return Err("invalid RNNT decode parameters".into());
        }
        if self
            .compiled_prompt_index
            .is_some_and(|compiled| compiled != prompt_index)
        {
            return Err("RNNT prompt differs from the compiled packet prompt".into());
        }
        let mut programs = BTreeMap::from([
            ("predictor.0".into(), self.predictor_programs[0] as u32),
            ("predictor.1".into(), self.predictor_programs[1] as u32),
        ]);
        for (stage, &program) in self.encoder_programs.iter().enumerate() {
            programs.insert(format!("encoder.{stage}"), program as u32);
            if input_frames != 0 {
                programs.insert(format!("encoder.{input_frames}.{stage}"), program as u32);
            }
        }
        for (&capacity, sequence) in &self.encoder_buckets {
            if capacity >= input_frames {
                return Err("RNNT encoder bucket exceeds primary capacity".into());
            }
            for (stage, &program) in sequence.iter().enumerate() {
                programs.insert(format!("encoder.{capacity}.{stage}"), program as u32);
            }
        }
        for rung in &self.joint_programs {
            programs.insert(format!("joint.0.{}", rung.frames), rung.banks[0] as u32);
            programs.insert(format!("joint.1.{}", rung.frames), rung.banks[1] as u32);
        }
        let mut tensors = BTreeMap::from([
            (
                "input".into(),
                PipelineTensor {
                    name: self.model.tensors[self.input as usize].name.clone(),
                    dtype: PipelineDType::F32,
                    shape: self.input_shape.clone(),
                },
            ),
            (
                "encoder.joint".into(),
                PipelineTensor {
                    name: "act.rnnt.encoder_joint".into(),
                    dtype: PipelineDType::F32,
                    shape: vec![self.spec.frames as u64, self.spec.joint_width as u64],
                },
            ),
            (
                "predictor.token".into(),
                PipelineTensor {
                    name: "in.rnnt.token".into(),
                    dtype: PipelineDType::U32,
                    shape: vec![1],
                },
            ),
            (
                "joint.encoder_window".into(),
                PipelineTensor {
                    name: "act.rnnt.encoder_window".into(),
                    dtype: PipelineDType::F32,
                    shape: vec![self.spec.joint_batch as u64, self.spec.joint_width as u64],
                },
            ),
            (
                "joint.ids".into(),
                PipelineTensor {
                    name: "act.rnnt.ids".into(),
                    dtype: PipelineDType::U32,
                    shape: vec![self.spec.joint_batch as u64],
                },
            ),
        ]);
        for (index, state) in self
            .model
            .tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with("state.rnnt."))
            .enumerate()
        {
            if !state.bytes.is_multiple_of(4) {
                return Err(format!("RNNT state tensor {} is not FP32", state.name));
            }
            tensors.insert(
                format!("predictor.state.{index}"),
                PipelineTensor {
                    name: state.name.clone(),
                    dtype: PipelineDType::F32,
                    shape: vec![state.bytes / 4],
                },
            );
        }
        let mut parameters = BTreeMap::from([
            ("blank_id".into(), blank_id as u64),
            ("frames".into(), self.spec.frames as u64),
            ("encoder_width".into(), self.spec.encoder_width as u64),
            ("host_prompt".into(), self.host_prompt as u64),
            ("prompt_count".into(), self.spec.prompt_count as u64),
            ("prompt_index".into(), prompt_index as u64),
            ("joint_batch_max".into(), self.spec.joint_batch as u64),
            ("max_symbols_per_frame".into(), max_symbols_per_frame as u64),
        ]);
        if let Some(frontend) = &self.frontend {
            let spec = frontend.spec;
            let tensor = &self.model.tensors[frontend.tensor as usize];
            tensors.insert(
                "audio.frontend.filterbank".into(),
                PipelineTensor {
                    name: tensor.name.clone(),
                    dtype: PipelineDType::F32,
                    shape: vec![u64::from(spec.bins), u64::from(spec.fft / 2 + 1)],
                },
            );
            for (name, value) in [
                ("audio.frontend.kind", 1),
                ("audio.sample_rate", u64::from(spec.sample_rate)),
                ("audio.min_samples", u64::from(spec.min_samples)),
                ("audio.max_samples", u64::from(spec.max_samples)),
                ("audio.frontend.fft", u64::from(spec.fft)),
                ("audio.frontend.window", u64::from(spec.window)),
                ("audio.frontend.hop", u64::from(spec.hop)),
                ("audio.frontend.bins", u64::from(spec.bins)),
                (
                    "audio.frontend.preemphasis_f32",
                    u64::from(spec.preemphasis.to_bits()),
                ),
                ("audio.frontend.center_window", spec.center_window as u64),
                ("audio.frontend.periodic_hann", spec.periodic_hann as u64),
                (
                    "audio.frontend.normalize_per_feature",
                    spec.normalize_per_feature as u64,
                ),
                (
                    "audio.frontend.mask_invalid_frames",
                    spec.mask_invalid_frames as u64,
                ),
                (
                    "audio.frontend.log_guard_f32",
                    u64::from(spec.log_guard.to_bits()),
                ),
            ] {
                parameters.insert(name.into(), value);
            }
        }
        if !stages.is_empty() {
            if input_frames == 0 {
                return Err("RNNT frame transform input is empty".into());
            }
            let mut frames = input_frames;
            parameters.insert("input_frames".into(), input_frames as u64);
            parameters.insert("frame_transform.count".into(), stages.len() as u64);
            parameters.insert(
                "frame_transform.trailing_frames".into(),
                trailing_frames as u64,
            );
            for (index, &[kernel, stride, pad_before, pad_after]) in stages.iter().enumerate() {
                frames =
                    crate::conv2d::output_extent(frames, kernel, stride, pad_before, pad_after)?;
                for (field, value) in [
                    ("kernel", kernel),
                    ("stride", stride),
                    ("pad_before", pad_before),
                    ("pad_after", pad_after),
                ] {
                    parameters.insert(format!("frame_transform.{index}.{field}"), value as u64);
                }
            }
            if frames != self.spec.frames {
                return Err(format!(
                    "RNNT frame transform produces {frames}, expected {}",
                    self.spec.frames
                ));
            }
        } else if input_frames != 0 || trailing_frames != 0 {
            return Err("RNNT frame transform stages are empty".into());
        }
        let pipeline = PacketPipeline {
            name: "transcribe".into(),
            driver: "rnnt.greedy.v1".into(),
            programs,
            tensors,
            parameters,
        };
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![pipeline],
        };
        metadata.validate(self.model.progs.len(), |name| {
            self.model
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
}

pub fn lower(spec: RnntSpec, weights: &RnntWeights<'_>, n_cu: u32) -> Result<RnntPackets, String> {
    lower_inner(spec, weights, n_cu, None, None)
}

pub fn append(
    spec: RnntSpec,
    weights: &RnntWeights<'_>,
    prefix: PacketPrefix,
    prompt_index: u32,
) -> Result<RnntPackets, String> {
    let n_cu = prefix.model.n_cu;
    lower_inner(spec, weights, n_cu, Some(prefix), Some(prompt_index))
}

fn lower_inner(
    spec: RnntSpec,
    weights: &RnntWeights<'_>,
    n_cu: u32,
    prefix: Option<PacketPrefix>,
    compiled_prompt_index: Option<u32>,
) -> Result<RnntPackets, String> {
    if n_cu == 0
        || spec.frames == 0
        || spec.encoder_width == 0
        || spec.prompt_count == 0
        || spec.prompt_width == 0
        || spec.vocabulary == 0
        || spec.predictor_width == 0
        || spec.joint_width == 0
        || spec.joint_batch == 0
        || spec.joint_batch > spec.frames
        || weights.lstm.is_empty()
        || spec.encoder_width % 32 != 0
        || spec.predictor_width % 32 != 0
        || spec.joint_width % 32 != 0
    {
        return Err("invalid RNNT geometry".into());
    }
    if compiled_prompt_index.is_some_and(|prompt| prompt >= spec.prompt_count) {
        return Err("invalid compiled RNNT prompt".into());
    }
    let (
        mut tensors,
        mut programs,
        mut program_t,
        mut encoder_programs,
        input,
        input_shape,
        source_output,
        target,
        kv_row_insts,
        gen,
    ) = if let Some(prefix) = prefix {
        let input_bytes = prefix
            .input_shape
            .iter()
            .try_fold(4u64, |bytes, &dim| bytes.checked_mul(dim));
        if prefix.model.n_cu != n_cu
            || prefix.programs.is_empty()
            || prefix
                .programs
                .iter()
                .any(|&program| program >= prefix.model.progs.len())
            || prefix.input as usize >= prefix.model.tensors.len()
            || prefix.output as usize >= prefix.model.tensors.len()
            || prefix.model.tensors[prefix.output as usize].bytes
                != bytes3(spec.frames, spec.encoder_width, 4)
            || prefix.input_shape.is_empty()
            || prefix.input_shape.contains(&0)
            || input_bytes != Some(prefix.model.tensors[prefix.input as usize].bytes)
        {
            return Err("invalid RNNT packet prefix".into());
        }
        let input = prefix.input;
        let output = prefix.output;
        let input_shape = prefix.input_shape;
        let Model {
            n_cu: _,
            target,
            tensors,
            progs,
            prog_t,
            kv_row_insts,
            gen,
        } = prefix.model;
        (
            tensors,
            progs,
            prog_t,
            prefix.programs,
            input,
            input_shape,
            output,
            target,
            kv_row_insts,
            gen,
        )
    } else {
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            u32::MAX,
            vec![
                spec.frames as u64,
                (spec.encoder_width + spec.prompt_count) as u64,
            ],
            u32::MAX,
            0,
            Vec::new(),
            Vec::new(),
        )
    };

    let mut encoder = builder(n_cu, tensors);
    let encoder_input = if let Some(prefix_prompt) = compiled_prompt_index {
        EncoderInput::Existing {
            tensor: source_output,
            prompt_index: prefix_prompt,
        }
    } else {
        EncoderInput::Prompted
    };
    let encoder_input_handle = emit_encoder(&mut encoder, spec, weights, encoder_input);
    let program = encoder.finish();
    tensors = program.tensors.clone();
    let encoder_program = programs.len();
    programs.push(program);
    program_t.push(spec.frames);
    encoder_programs.push(encoder_program);
    let input = if input == u32::MAX {
        encoder_input_handle
    } else {
        input
    };

    let predictor_programs = [programs.len(), programs.len() + 1];
    for source_bank in 0..2 {
        let mut predictor = builder(n_cu, tensors);
        emit_predictor(&mut predictor, spec, weights, source_bank);
        let program = predictor.finish();
        tensors = program.tensors.clone();
        programs.push(program);
        program_t.push(1);
    }

    let mut joint_programs = Vec::new();
    for frames in joint_batches(spec.joint_batch) {
        let banks = [programs.len(), programs.len() + 1];
        for bank in 0..2 {
            let mut joint = builder(n_cu, tensors);
            emit_joint(&mut joint, spec, weights, bank, frames);
            let program = joint.finish();
            tensors = program.tensors.clone();
            programs.push(program);
            program_t.push(frames);
        }
        joint_programs.push(RnntJointPrograms { frames, banks });
    }
    initialize_states(&mut tensors);
    Ok(RnntPackets {
        model: Model {
            n_cu,
            target,
            tensors,
            progs: programs,
            prog_t: program_t,
            kv_row_insts,
            gen,
        },
        encoder_programs,
        predictor_programs,
        joint_programs,
        spec,
        input,
        input_shape,
        host_prompt: compiled_prompt_index.is_none(),
        compiled_prompt_index,
        frontend: None,
        encoder_buckets: BTreeMap::new(),
    })
}

fn builder(n_cu: u32, tensors: Vec<TensorDecl>) -> Builder {
    let mut builder = Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    builder.adopt_tensors(tensors);
    builder
}

#[derive(Clone, Copy)]
enum EncoderInput {
    Prompted,
    Existing { tensor: u32, prompt_index: u32 },
}

fn emit_encoder(b: &mut Builder, s: RnntSpec, w: &RnntWeights<'_>, source: EncoderInput) -> u32 {
    let full_input_width = s.encoder_width + s.prompt_count;
    let (input, input_width, implicit_onehot) = match source {
        EncoderInput::Prompted => (
            b.tensor("in.rnnt.prompted", bytes3(s.frames, full_input_width, 4)),
            full_input_width,
            None,
        ),
        EncoderInput::Existing {
            tensor,
            prompt_index,
        } => (
            tensor,
            s.encoder_width,
            Some(s.encoder_width + prompt_index),
        ),
    };
    let hidden = b.tensor(
        "act.rnnt.prompt_hidden",
        bytes3(s.frames, s.prompt_width, 4),
    );
    let encoded = b.tensor("act.rnnt.encoded", bytes3(s.frames, s.encoder_width, 4));
    let joint = b.tensor("act.rnnt.encoder_joint", bytes3(s.frames, s.joint_width, 4));
    let p0 = dense(
        b,
        hidden,
        input,
        w.prompt_in,
        s.frames,
        s.prompt_width,
        input_width,
        1,
        implicit_onehot.map(|column| (full_input_width, column)),
        None,
    );
    let p2 = dense(
        b,
        encoded,
        hidden,
        w.prompt_out,
        s.frames,
        s.encoder_width,
        s.prompt_width,
        0,
        None,
        Some(p0),
    );
    q8(
        b,
        joint,
        encoded,
        w.encoder,
        s.frames,
        s.joint_width,
        s.encoder_width,
        Some(p2),
    );
    input
}

fn emit_predictor(b: &mut Builder, s: RnntSpec, w: &RnntWeights<'_>, source: usize) {
    let target = source ^ 1;
    let token = b.tensor("in.rnnt.token", 4);
    let embedding_table = b.tensor(
        w.embedding,
        u64::from(s.vocabulary) * u64::from(s.predictor_width) * 2,
    );
    let embedding = b.tensor("act.rnnt.embedding", u64::from(s.predictor_width) * 4);
    let mut dep = b.emit(DevOp::EmbedF16F32, b.all(), &[], |d| {
        d.t[..3].copy_from_slice(&[embedding, embedding_table, token]);
        d.i[..2].copy_from_slice(&[s.vocabulary, s.predictor_width]);
    });
    let mut layer_input = embedding;
    for (layer, weights) in w.lstm.iter().enumerate() {
        let gates = b.tensor("act.rnnt.gates", u64::from(s.predictor_width) * 16);
        let recurrent = b.tensor("act.rnnt.recurrent", u64::from(s.predictor_width) * 16);
        let h_source = state(b, "h", source, layer, s.predictor_width);
        let c_source = state(b, "c", source, layer, s.predictor_width);
        let h_target = state(b, "h", target, layer, s.predictor_width);
        let c_target = state(b, "c", target, layer, s.predictor_width);
        let ih = q8(
            b,
            gates,
            layer_input,
            weights.input,
            1,
            s.predictor_width * 4,
            s.predictor_width,
            Some(dep),
        );
        let hh = q8(
            b,
            recurrent,
            h_source,
            weights.recurrent,
            1,
            s.predictor_width * 4,
            s.predictor_width,
            Some(dep),
        );
        let sum = b.emit(DevOp::ScaledAddF32, b.all(), &[ih, hh], |d| {
            d.t[..3].copy_from_slice(&[gates, gates, recurrent]);
            d.i[0] = s.predictor_width * 4;
            d.f[0] = 1.0;
        });
        dep = b.emit(DevOp::LstmCellF32, b.all(), &[sum], |d| {
            d.t[..4].copy_from_slice(&[h_target, c_target, gates, c_source]);
            d.i[0] = s.predictor_width;
        });
        layer_input = h_target;
    }
    let projected = b.tensor(
        &format!("act.rnnt.predictor_joint.{target}"),
        u64::from(s.joint_width) * 4,
    );
    q8(
        b,
        projected,
        layer_input,
        w.predictor,
        1,
        s.joint_width,
        s.predictor_width,
        Some(dep),
    );
}

fn emit_joint(b: &mut Builder, s: RnntSpec, w: &RnntWeights<'_>, bank: usize, frames: u32) {
    let encoded = b.tensor(
        "act.rnnt.encoder_window",
        bytes3(s.joint_batch, s.joint_width, 4),
    );
    let predictor = b.tensor(
        &format!("act.rnnt.predictor_joint.{bank}"),
        u64::from(s.joint_width) * 4,
    );
    let activation = b.tensor("act.rnnt.joint", bytes3(s.joint_batch, s.joint_width, 4));
    let logits = b.tensor("act.rnnt.logits", bytes3(s.joint_batch, s.vocabulary, 4));
    let ids = b.tensor("act.rnnt.ids", u64::from(s.joint_batch) * 4);
    let add = b.emit(DevOp::BroadcastAddF32, b.all(), &[], |d| {
        d.t[..3].copy_from_slice(&[activation, encoded, predictor]);
        d.i[..2].copy_from_slice(&[frames, s.joint_width]);
    });
    let relu = b.emit(DevOp::ReluF32, b.all(), &[add], |d| {
        d.t[..2].copy_from_slice(&[activation, activation]);
        d.i[0] = frames * s.joint_width;
    });
    let output = q8(
        b,
        logits,
        activation,
        w.output,
        frames,
        s.vocabulary,
        s.joint_width,
        Some(relu),
    );
    b.emit(
        DevOp::ArgmaxF32,
        repeated(n_cu(b), frames),
        &[output],
        |d| {
            d.t[..2].copy_from_slice(&[ids, logits]);
            d.i[..2].copy_from_slice(&[frames, s.vocabulary]);
        },
    );
}

fn dense(
    b: &mut Builder,
    out: u32,
    input: u32,
    w: LinearWeights<'_>,
    m: u32,
    n: u32,
    k: u32,
    act: u32,
    implicit_onehot: Option<(u32, u32)>,
    dep: Option<u32>,
) -> u32 {
    let stored_k = implicit_onehot.map_or(k, |(weight_stride, _)| weight_stride);
    let weight = b.tensor(w.weight, bytes3(n, stored_k, 4));
    let bias = b.tensor(w.bias, u64::from(n) * 4);
    b.emit(
        DevOp::DenseGemmF32,
        repeated(n_cu(b), m.div_ceil(128) * n.div_ceil(64)),
        &dep.into_iter().collect::<Vec<_>>(),
        |d| {
            d.t[..4].copy_from_slice(&[out, input, weight, bias]);
            d.i[..4].copy_from_slice(&[m, n, k, act]);
            if let Some((weight_stride, column)) = implicit_onehot {
                d.i[5] = weight_stride;
                d.i[6] = column;
            }
        },
    )
}

fn q8(
    b: &mut Builder,
    out: u32,
    input: u32,
    w: LinearWeights<'_>,
    m: u32,
    n: u32,
    k: u32,
    dep: Option<u32>,
) -> u32 {
    let weight = b.tensor(w.weight, u64::from(n) * u64::from(k / 32) * 34);
    let bias = b.tensor(w.bias, u64::from(n) * 4);
    b.emit(
        DevOp::Q8GemmF32,
        repeated(n_cu(b), m.div_ceil(128) * n.div_ceil(64)),
        &dep.into_iter().collect::<Vec<_>>(),
        |d| {
            d.t[..4].copy_from_slice(&[out, input, weight, bias]);
            d.i[..3].copy_from_slice(&[m, n, k]);
        },
    )
}

fn state(b: &mut Builder, kind: &str, bank: usize, layer: usize, width: u32) -> u32 {
    b.tensor(
        &format!("state.rnnt.{kind}.{bank}.{layer}"),
        u64::from(width) * 4,
    )
}
fn initialize_states(tensors: &mut [TensorDecl]) {
    for t in tensors
        .iter_mut()
        .filter(|t| t.name.starts_with("state.rnnt."))
    {
        t.init = Some(vec![0; t.bytes as usize]);
    }
}
fn bytes3(a: u32, b: u32, c: u32) -> u64 {
    u64::from(a) * u64::from(b) * u64::from(c)
}
fn n_cu(b: &Builder) -> u32 {
    b.n_cu()
}
fn repeated(n_cu: u32, blocks: u32) -> Vec<u32> {
    (0..blocks.max(1)).map(|i| i % n_cu).collect()
}

fn joint_batches(maximum: u32) -> Vec<u32> {
    let mut batches = Vec::new();
    let mut batch = 1;
    while batch <= maximum {
        batches.push(batch);
        let Some(next) = batch.checked_mul(2) else {
            break;
        };
        batch = next;
    }
    if batches.last().copied() != Some(maximum) {
        batches.push(maximum);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowering_builds_encoder_two_bank_predictor_and_joint_programs() {
        let lstm = [
            LstmWeights {
                input: LinearWeights {
                    weight: "lstm.ih.0",
                    bias: "lstm.ib.0",
                },
                recurrent: LinearWeights {
                    weight: "lstm.hh.0",
                    bias: "lstm.hb.0",
                },
            },
            LstmWeights {
                input: LinearWeights {
                    weight: "lstm.ih.1",
                    bias: "lstm.ib.1",
                },
                recurrent: LinearWeights {
                    weight: "lstm.hh.1",
                    bias: "lstm.hb.1",
                },
            },
        ];
        let weights = RnntWeights {
            prompt_in: LinearWeights {
                weight: "prompt.in.weight",
                bias: "prompt.in.bias",
            },
            prompt_out: LinearWeights {
                weight: "prompt.out.weight",
                bias: "prompt.out.bias",
            },
            encoder: LinearWeights {
                weight: "joint.enc.weight",
                bias: "joint.enc.bias",
            },
            embedding: "embed.weight",
            lstm: &lstm,
            predictor: LinearWeights {
                weight: "joint.pred.weight",
                bias: "joint.pred.bias",
            },
            output: LinearWeights {
                weight: "joint.out.weight",
                bias: "joint.out.bias",
            },
        };
        let mut packets = lower(
            RnntSpec {
                frames: 3,
                encoder_width: 32,
                prompt_count: 4,
                prompt_width: 64,
                vocabulary: 64,
                predictor_width: 32,
                joint_width: 32,
                joint_batch: 3,
            },
            &weights,
            4,
        )
        .unwrap();
        assert_eq!(packets.model.progs.len(), 9);
        assert_eq!(packets.predictor_programs, [1, 2]);
        assert_eq!(
            packets.joint_programs,
            [
                RnntJointPrograms {
                    frames: 1,
                    banks: [3, 4]
                },
                RnntJointPrograms {
                    frames: 2,
                    banks: [5, 6]
                },
                RnntJointPrograms {
                    frames: 3,
                    banks: [7, 8]
                }
            ]
        );
        assert_eq!(
            packets
                .model
                .tensors
                .iter()
                .filter(|tensor| tensor.name.starts_with("state.rnnt."))
                .count(),
            8
        );
        assert!(packets
            .model
            .tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with("state.rnnt."))
            .all(|tensor| tensor
                .init
                .as_ref()
                .is_some_and(|bytes| bytes.iter().all(|&b| b == 0))));
        assert!(packets
            .model
            .progs
            .iter()
            .flat_map(|program| &program.insts)
            .all(|inst| {
                matches!(
                    DevOp::from_u16(inst.op),
                    Some(
                        DevOp::DenseGemmF32
                            | DevOp::Q8GemmF32
                            | DevOp::EmbedF16F32
                            | DevOp::ScaledAddF32
                            | DevOp::LstmCellF32
                            | DevOp::BroadcastAddF32
                            | DevOp::ReluF32
                            | DevOp::ArgmaxF32
                    )
                )
            }));
        let bucket = lower(
            RnntSpec {
                frames: 3,
                encoder_width: 32,
                prompt_count: 4,
                prompt_width: 64,
                vocabulary: 64,
                predictor_width: 32,
                joint_width: 32,
                joint_batch: 3,
            },
            &weights,
            4,
        )
        .unwrap();
        let tensor_count = packets.model.tensors.len();
        packets.merge_encoder_bucket(6, bucket).unwrap();
        assert_eq!(packets.model.tensors.len(), tensor_count);
        assert_eq!(packets.model.progs.len(), 10);
        packets
            .embed_log_mel_frontend(
                LogMelSpec {
                    sample_rate: 16_000,
                    fft: 8,
                    window: 6,
                    hop: 4,
                    bins: 2,
                    preemphasis: 0.97,
                    center_window: true,
                    periodic_hann: false,
                    normalize_per_feature: true,
                    mask_invalid_frames: true,
                    log_guard: 0.25,
                    min_samples: 8,
                    max_samples: 39,
                },
                &[1.0; 10],
            )
            .unwrap();
        let section = packets
            .pipeline_section_with_frame_transform(63, 10, 0, 10, &[[3, 2, 1, 0], [3, 2, 1, 1]], 1)
            .unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        assert_eq!(metadata.pipelines[0].driver, "rnnt.greedy.v1");
        assert_eq!(metadata.pipelines[0].programs["encoder.0"], 0);
        assert_eq!(metadata.pipelines[0].programs["joint.1.3"], 8);
        assert_eq!(metadata.pipelines[0].parameters["frames"], 3);
        assert_eq!(metadata.pipelines[0].parameters["audio.frontend.kind"], 1);
        assert_eq!(
            metadata.pipelines[0].tensors["audio.frontend.filterbank"].shape,
            [2, 5]
        );
        assert_eq!(
            metadata.pipelines[0]
                .tensors
                .keys()
                .filter(|role| role.starts_with("predictor.state."))
                .count(),
            8
        );
        assert_eq!(metadata.pipelines[0].parameters["input_frames"], 10);
        assert_eq!(metadata.pipelines[0].programs["encoder.6.0"], 9);
        assert_eq!(metadata.pipelines[0].programs["encoder.10.0"], 0);
        assert_eq!(metadata.pipelines[0].parameters["frame_transform.count"], 2);
        assert_eq!(
            metadata.pipelines[0].parameters["frame_transform.1.stride"],
            2
        );
        assert_eq!(
            metadata.pipelines[0].parameters["frame_transform.trailing_frames"],
            1
        );

        let mut prefix_builder = Builder::new(4);
        let prefix_io = prefix_builder.tensor("prefix.io", bytes3(3, 32, 4));
        let prefix_program = prefix_builder.finish();
        let prefix_tensors = prefix_program.tensors.clone();
        let appended = append(
            RnntSpec {
                frames: 3,
                encoder_width: 32,
                prompt_count: 4,
                prompt_width: 64,
                vocabulary: 64,
                predictor_width: 32,
                joint_width: 32,
                joint_batch: 3,
            },
            &weights,
            PacketPrefix {
                model: Model {
                    n_cu: 4,
                    target: 0,
                    tensors: prefix_tensors,
                    progs: vec![prefix_program],
                    prog_t: vec![3],
                    kv_row_insts: vec![],
                    gen: vec![],
                },
                programs: vec![0],
                input: prefix_io,
                output: prefix_io,
                input_shape: vec![3, 32],
            },
            2,
        )
        .unwrap();
        assert_eq!(appended.encoder_programs, [0, 1]);
        let fused = appended.model.progs[1]
            .insts
            .iter()
            .find(|inst| DevOp::from_u16(inst.op) == Some(DevOp::DenseGemmF32))
            .unwrap();
        assert_eq!(fused.i[2], 32);
        assert_eq!(fused.i[5], 36);
        assert_eq!(fused.i[6], 34);
        let section = appended.pipeline_section(63, 10, 2).unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&section.data).unwrap();
        assert_eq!(metadata.pipelines[0].programs["encoder.0"], 0);
        assert_eq!(metadata.pipelines[0].programs["encoder.1"], 1);
        assert_eq!(metadata.pipelines[0].tensors["input"].name, "prefix.io");
        assert_eq!(metadata.pipelines[0].parameters["host_prompt"], 0);
        assert!(appended.pipeline_section(63, 10, 1).is_err());
    }
}
