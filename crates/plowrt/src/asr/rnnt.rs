//! Model-independent greedy RNNT control over packetized predictor and joint programs.

use std::path::Path;
use std::time::Instant;

use crate::asr::frontend::PacketLogMelFrontend;
use crate::exec::packet_runtime::{
    load_packet_runtime, BoundPacketPipeline, PacketAsset, PacketRuntime, PacketTensor,
};
use crate::{Result, RuntimeError};

/// Device-side RNNT data path. Implementations dispatch packet programs and retain encoder and
/// predictor tensors on their selected backend.
pub trait RnntExecution: Send {
    fn frames(&self) -> usize;
    fn predict(&mut self, previous_token: u32) -> Result<()>;
    fn joint_argmax(&mut self, first_frame: usize, output: &mut [u32]) -> Result<usize>;
    fn commit_prediction(&mut self);
}

pub struct GreedyRnnt {
    blank_id: u32,
    max_symbols_per_frame: usize,
    previous_token: u32,
    predictor_valid: bool,
}

pub struct PacketRnnt {
    backend: &'static str,
    runtime: Box<dyn PacketRuntime>,
    pipeline: BoundPacketPipeline,
    input: PacketTensor,
    encoder_joint: PacketTensor,
    encoder_window: PacketTensor,
    token: PacketTensor,
    ids: PacketTensor,
    states: Vec<PacketTensor>,
    encoder_programs: Vec<(usize, Vec<usize>)>,
    predictor_programs: [usize; 2],
    frames: usize,
    row_bytes: usize,
    joint_batch_max: usize,
    blank_id: u32,
    max_symbols_per_frame: usize,
    state_zeros: Vec<u8>,
    input_frames: Option<usize>,
    frame_transform: Vec<[usize; 4]>,
    trailing_frames: usize,
    profiling: bool,
    last_profile: Option<RnntProfile>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct RnntProfile {
    pub encoder_runtime_us: f64,
    pub predictor_joint_runtime_us: f64,
    pub joint_runtime_us: f64,
    pub transfer_us: f64,
    pub predictor_joint_runs: usize,
    pub joint_runs: usize,
}

impl PacketRnnt {
    pub fn load(path: &Path, requested_backend: &str) -> Result<Self> {
        let asset = PacketAsset::load(path)?;
        let loaded = load_packet_runtime(path, requested_backend)?;
        let backend = loaded.backend;
        let mut runtime = loaded.runtime;
        let pipeline = asset.bind_driver("rnnt.greedy.v1", &*runtime)?;
        if pipeline.driver() != "rnnt.greedy.v1" {
            return Err(RuntimeError::Rejected(format!(
                "unsupported packet pipeline driver {:?}",
                pipeline.driver()
            )));
        }
        let frames = usize_parameter(&pipeline, "frames")?;
        let joint_batch_max = usize_parameter(&pipeline, "joint_batch_max")?;
        let blank_id = u32::try_from(pipeline.parameter("blank_id")?)
            .map_err(|_| RuntimeError::Rejected("RNNT blank token overflows".into()))?;
        let max_symbols_per_frame = usize_parameter(&pipeline, "max_symbols_per_frame")?;
        let (input_frames, frame_transform, trailing_frames) = frame_transform(&pipeline)?;
        if input_frames
            .map(|capacity| transformed_frame_count(capacity, &frame_transform))
            .transpose()?
            .is_some_and(|transformed| transformed != frames)
        {
            return Err(RuntimeError::Rejected(
                "packet frame transform does not match encoder capacity".into(),
            ));
        }
        let input = pipeline.tensor("input")?;
        let encoder_joint = pipeline.tensor("encoder.joint")?;
        let encoder_window = pipeline.tensor("joint.encoder_window")?;
        let token = pipeline.tensor("predictor.token")?;
        let ids = pipeline.tensor("joint.ids")?;
        let states = pipeline.tensor_sequence("predictor.state")?;
        let default_encoder = pipeline.program_sequence("encoder")?;
        let capacity_encoders = pipeline.program_capacity_sequences("encoder")?;
        let encoder_programs = if capacity_encoders.is_empty() {
            vec![(input_frames.unwrap_or(usize::MAX), default_encoder)]
        } else {
            let converted: Vec<_> = capacity_encoders
                .into_iter()
                .map(|(capacity, programs)| {
                    usize::try_from(capacity)
                        .map(|capacity| (capacity, programs))
                        .map_err(|_| {
                            RuntimeError::Rejected("encoder capacity overflows usize".into())
                        })
                })
                .collect::<Result<_>>()?;
            if input_frames != converted.last().map(|entry| entry.0) {
                return Err(RuntimeError::Rejected(
                    "packet encoder capacities do not cover its input capacity".into(),
                ));
            }
            converted
        };
        let predictor_programs = [
            pipeline.program("predictor.0")?,
            pipeline.program("predictor.1")?,
        ];
        if frames == 0
            || joint_batch_max == 0
            || joint_batch_max > frames
            || !encoder_joint.bytes.is_multiple_of(frames)
        {
            return Err(RuntimeError::Rejected(
                "packet RNNT frame geometry is invalid".into(),
            ));
        }
        let row_bytes = encoder_joint.bytes / frames;
        if encoder_window.bytes != joint_batch_max * row_bytes
            || ids.bytes != joint_batch_max * std::mem::size_of::<u32>()
            || token.bytes != std::mem::size_of::<u32>()
        {
            return Err(RuntimeError::Rejected(
                "packet RNNT tensor geometry is inconsistent".into(),
            ));
        }
        let state_zeros = vec![0; states.iter().map(|state| state.bytes).max().unwrap_or(0)];
        runtime.end_execution()?;
        Ok(Self {
            backend,
            runtime,
            pipeline,
            input,
            encoder_joint,
            encoder_window,
            token,
            ids,
            states,
            encoder_programs,
            predictor_programs,
            frames,
            row_bytes,
            joint_batch_max,
            blank_id,
            max_symbols_per_frame,
            state_zeros,
            input_frames,
            frame_transform,
            trailing_frames,
            profiling: false,
            last_profile: None,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn input_elements(&self) -> usize {
        self.input.bytes / std::mem::size_of::<f32>()
    }

    pub fn parameter(&self, name: &str) -> Result<u64> {
        self.pipeline.parameter(name)
    }

    pub fn log_mel_frontend(&self) -> Result<PacketLogMelFrontend> {
        PacketLogMelFrontend::bind(&self.pipeline, &*self.runtime)
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if !enabled {
            self.last_profile = None;
        }
    }

    pub fn last_profile(&self) -> Option<RnntProfile> {
        self.last_profile
    }

    pub fn transcribe_input(&mut self, input: &[f32]) -> Result<Vec<u32>> {
        self.transcribe_input_with_encoder_frames(
            input,
            self.frames,
            self.encoder_programs.len() - 1,
        )
    }

    pub fn transcribe_input_frames(
        &mut self,
        input: &[f32],
        valid_input_frames: usize,
    ) -> Result<Vec<u32>> {
        let encoder = self
            .encoder_programs
            .iter()
            .position(|&(capacity, _)| capacity >= valid_input_frames)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "packet has no encoder capacity for {valid_input_frames} frames"
                ))
            })?;
        let frames =
            self.valid_encoder_frames(valid_input_frames, self.encoder_programs[encoder].0)?;
        self.transcribe_input_with_encoder_frames(input, frames, encoder)
    }

    fn transcribe_input_with_encoder_frames(
        &mut self,
        input: &[f32],
        frames: usize,
        encoder: usize,
    ) -> Result<Vec<u32>> {
        self.runtime.begin_execution()?;
        let result = self.transcribe_active(input, frames, encoder);
        let ended = self.runtime.end_execution();
        result.and_then(|tokens| ended.map(|()| tokens))
    }

    fn transcribe_active(
        &mut self,
        input: &[f32],
        frames: usize,
        encoder: usize,
    ) -> Result<Vec<u32>> {
        if input.len() != self.input_elements() {
            return Err(RuntimeError::Rejected(format!(
                "packet RNNT input has {} elements, expected {}",
                input.len(),
                self.input_elements()
            )));
        }
        for &state in &self.states {
            self.runtime
                .write_tensor(state, &self.state_zeros[..state.bytes])?;
        }
        self.runtime
            .write_tensor(self.input, bytemuck::cast_slice(input))?;
        self.runtime
            .run_sequence(&self.encoder_programs[encoder].1)?;
        let mut profile = self.profiling.then(|| RnntProfile {
            encoder_runtime_us: self.runtime.last_run_us(),
            ..RnntProfile::default()
        });
        let mut execution = PacketRnntExecution {
            runtime: &mut *self.runtime,
            pipeline: &self.pipeline,
            frames,
            token: self.token,
            ids: self.ids,
            encoder_joint: self.encoder_joint,
            encoder_window: self.encoder_window,
            row_bytes: self.row_bytes,
            predictor_programs: self.predictor_programs,
            active_bank: 0,
            pending_predictor: None,
            all_ids: vec![0; self.joint_batch_max],
            profile: profile.as_mut(),
        };
        let result =
            GreedyRnnt::new(self.blank_id, self.max_symbols_per_frame)?.decode(&mut execution);
        self.last_profile = profile;
        result
    }

    fn valid_encoder_frames(
        &self,
        valid_input_frames: usize,
        encoder_capacity: usize,
    ) -> Result<usize> {
        let capacity = self.input_frames.ok_or_else(|| {
            RuntimeError::Rejected("packet does not declare an input frame transform".into())
        })?;
        if valid_input_frames == 0 || valid_input_frames > capacity {
            return Err(RuntimeError::Rejected(format!(
                "packet input has {valid_input_frames} valid frames, capacity is {capacity}"
            )));
        }
        let frames = bounded_transformed_frame_count(
            valid_input_frames,
            encoder_capacity,
            &self.frame_transform,
            self.trailing_frames,
        )?;
        if frames == 0 || frames > self.frames {
            return Err(RuntimeError::Rejected(format!(
                "packet frame transform produced {frames}, capacity is {}",
                self.frames
            )));
        }
        Ok(frames)
    }
}

struct PacketRnntExecution<'a> {
    runtime: &'a mut dyn PacketRuntime,
    pipeline: &'a BoundPacketPipeline,
    frames: usize,
    token: PacketTensor,
    ids: PacketTensor,
    encoder_joint: PacketTensor,
    encoder_window: PacketTensor,
    row_bytes: usize,
    predictor_programs: [usize; 2],
    active_bank: usize,
    pending_predictor: Option<usize>,
    all_ids: Vec<u32>,
    profile: Option<&'a mut RnntProfile>,
}

impl RnntExecution for PacketRnntExecution<'_> {
    fn frames(&self) -> usize {
        self.frames
    }

    fn predict(&mut self, previous_token: u32) -> Result<()> {
        let started = self.profile.is_some().then(Instant::now);
        self.runtime
            .write_tensor(self.token, &previous_token.to_ne_bytes())?;
        if let (Some(profile), Some(started)) = (&mut self.profile, started) {
            profile.transfer_us += started.elapsed().as_secs_f64() * 1e6;
        }
        self.pending_predictor = Some(self.predictor_programs[self.active_bank]);
        Ok(())
    }

    fn joint_argmax(&mut self, first_frame: usize, output: &mut [u32]) -> Result<usize> {
        let requested_rows = u32::try_from(output.len())
            .map_err(|_| RuntimeError::Rejected("RNNT joint batch overflows".into()))?;
        let (_, rows) = self
            .pipeline
            .program_for_rows_at_most("joint.0", requested_rows)?;
        let rows = usize::try_from(rows)
            .map_err(|_| RuntimeError::Rejected("RNNT joint capacity overflows".into()))?;
        let program = self
            .pipeline
            .program(&format!("joint.{}.{rows}", self.active_bank ^ 1))?;
        let transfer = self.profile.is_some().then(Instant::now);
        let source_offset = first_frame
            .checked_mul(self.row_bytes)
            .ok_or_else(|| RuntimeError::Rejected("RNNT encoder offset overflows".into()))?;
        let copy_bytes = rows
            .checked_mul(self.row_bytes)
            .ok_or_else(|| RuntimeError::Rejected("RNNT encoder window overflows".into()))?;
        self.runtime.copy_tensor(
            self.encoder_joint,
            source_offset,
            self.encoder_window,
            0,
            copy_bytes,
        )?;
        if let (Some(profile), Some(transfer)) = (&mut self.profile, transfer) {
            profile.transfer_us += transfer.elapsed().as_secs_f64() * 1e6;
        }
        if let Some(predictor) = self.pending_predictor.take() {
            self.runtime.run_sequence(&[predictor, program])?;
            if let Some(profile) = &mut self.profile {
                profile.predictor_joint_runtime_us += self.runtime.last_run_us();
                profile.predictor_joint_runs += 1;
            }
        } else {
            self.runtime.run(program)?;
            if let Some(profile) = &mut self.profile {
                profile.joint_runtime_us += self.runtime.last_run_us();
                profile.joint_runs += 1;
            }
        }
        let transfer = self.profile.is_some().then(Instant::now);
        self.runtime
            .read_tensor(self.ids, bytemuck::cast_slice_mut(&mut self.all_ids))?;
        if let (Some(profile), Some(transfer)) = (&mut self.profile, transfer) {
            profile.transfer_us += transfer.elapsed().as_secs_f64() * 1e6;
        }
        output[..rows].copy_from_slice(&self.all_ids[..rows]);
        Ok(rows)
    }

    fn commit_prediction(&mut self) {
        self.active_bank ^= 1;
    }
}

fn usize_parameter(pipeline: &BoundPacketPipeline, name: &str) -> Result<usize> {
    usize::try_from(pipeline.parameter(name)?)
        .map_err(|_| RuntimeError::Rejected(format!("packet parameter {name:?} overflows")))
}

fn frame_transform(
    pipeline: &BoundPacketPipeline,
) -> Result<(Option<usize>, Vec<[usize; 4]>, usize)> {
    let Some(input_frames) = pipeline.optional_parameter("input_frames") else {
        return Ok((None, Vec::new(), 0));
    };
    let input_frames = usize::try_from(input_frames)
        .map_err(|_| RuntimeError::Rejected("packet input frame capacity overflows".into()))?;
    let count = usize_parameter(pipeline, "frame_transform.count")?;
    let trailing_frames = usize::try_from(
        pipeline
            .optional_parameter("frame_transform.trailing_frames")
            .unwrap_or(0),
    )
    .map_err(|_| RuntimeError::Rejected("packet trailing frame count overflows".into()))?;
    if input_frames == 0 || count == 0 || count > 32 || trailing_frames > 32 {
        return Err(RuntimeError::Rejected(
            "packet frame transform geometry is invalid".into(),
        ));
    }
    let mut stages = Vec::with_capacity(count);
    for index in 0..count {
        let mut stage = [0; 4];
        for (slot, field) in ["kernel", "stride", "pad_before", "pad_after"]
            .into_iter()
            .enumerate()
        {
            stage[slot] = usize_parameter(pipeline, &format!("frame_transform.{index}.{field}"))?;
        }
        stages.push(stage);
    }
    Ok((Some(input_frames), stages, trailing_frames))
}

fn transformed_frame_count(mut frames: usize, stages: &[[usize; 4]]) -> Result<usize> {
    for &[kernel, stride, pad_before, pad_after] in stages {
        let padded = frames
            .checked_add(pad_before)
            .and_then(|value| value.checked_add(pad_after))
            .ok_or_else(|| RuntimeError::Rejected("packet frame transform overflows".into()))?;
        if kernel == 0 || stride == 0 || padded < kernel {
            return Err(RuntimeError::Rejected(
                "packet frame transform is invalid".into(),
            ));
        }
        frames = (padded - kernel) / stride + 1;
    }
    Ok(frames)
}

fn bounded_transformed_frame_count(
    valid_frames: usize,
    capacity: usize,
    stages: &[[usize; 4]],
    trailing_frames: usize,
) -> Result<usize> {
    Ok(transformed_frame_count(valid_frames, stages)?
        .saturating_add(trailing_frames)
        .min(transformed_frame_count(capacity, stages)?))
}

impl GreedyRnnt {
    pub fn new(blank_id: u32, max_symbols_per_frame: usize) -> Result<Self> {
        if max_symbols_per_frame == 0 {
            return Err(RuntimeError::Rejected(
                "RNNT max symbols per frame must be positive".into(),
            ));
        }
        Ok(Self {
            blank_id,
            max_symbols_per_frame,
            previous_token: blank_id,
            predictor_valid: false,
        })
    }

    pub fn reset(&mut self) {
        self.previous_token = self.blank_id;
        self.predictor_valid = false;
    }

    /// Decode one encoder chunk. A fixed predictor state evaluates every remaining frame in one
    /// packet dispatch; the first nonblank token invalidates that speculative tail.
    pub fn decode(&mut self, execution: &mut dyn RnntExecution) -> Result<Vec<u32>> {
        let frames = execution.frames();
        let mut emitted = Vec::new();
        let mut ids = vec![self.blank_id; frames];
        let mut frame = 0;
        let mut symbols_at_frame = 0;
        while frame < frames {
            if !self.predictor_valid {
                execution.predict(self.previous_token)?;
                self.predictor_valid = true;
            }
            let evaluated = execution.joint_argmax(frame, &mut ids[..frames - frame])?;
            if evaluated == 0 || evaluated > frames - frame {
                return Err(RuntimeError::Device(
                    "RNNT joint returned an invalid frame count".into(),
                ));
            }
            let Some(offset) = ids[..evaluated]
                .iter()
                .position(|&token| token != self.blank_id)
            else {
                frame += evaluated;
                symbols_at_frame = 0;
                continue;
            };
            if offset != 0 {
                symbols_at_frame = 0;
            }
            frame += offset;
            let token = ids[offset];
            emitted.push(token);
            execution.commit_prediction();
            self.previous_token = token;
            self.predictor_valid = false;
            symbols_at_frame += 1;
            if symbols_at_frame == self.max_symbols_per_frame {
                frame += 1;
                symbols_at_frame = 0;
            }
        }
        Ok(emitted)
    }
}

pub fn detokenize_sentencepiece(vocabulary: &[String], ids: &[u32]) -> String {
    let mut text = String::new();
    for &id in ids {
        let Some(piece) = vocabulary.get(id as usize) else {
            continue;
        };
        if piece.starts_with('<') && piece.ends_with('>') {
            continue;
        }
        let value = piece.strip_prefix('▁').unwrap_or(piece);
        if piece.starts_with('▁') && !matches!(value, "." | "?" | "!" | "।" | "॥") {
            text.push(' ');
        }
        text.push_str(&value.replace('▁', " "));
    }
    text.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scripted {
        frames: usize,
        prediction: usize,
        calls: Vec<(usize, usize)>,
        commits: usize,
        always_emit: bool,
    }

    impl RnntExecution for Scripted {
        fn frames(&self) -> usize {
            self.frames
        }

        fn predict(&mut self, _previous_token: u32) -> Result<()> {
            self.prediction += 1;
            Ok(())
        }

        fn joint_argmax(&mut self, first_frame: usize, output: &mut [u32]) -> Result<usize> {
            self.calls.push((first_frame, output.len()));
            output.fill(9);
            if !self.always_emit {
                output.fill(13);
                if self.prediction == 1 && first_frame == 0 {
                    output.copy_from_slice(&[13, 7, 13]);
                }
            }
            Ok(output.len())
        }

        fn commit_prediction(&mut self) {
            self.commits += 1;
        }
    }

    #[test]
    fn batches_a_blank_run_and_restarts_at_the_emitting_frame() {
        let mut execution = Scripted {
            frames: 3,
            prediction: 0,
            calls: Vec::new(),
            commits: 0,
            always_emit: false,
        };
        let mut decoder = GreedyRnnt::new(13, 10).unwrap();
        assert_eq!(decoder.decode(&mut execution).unwrap(), [7]);
        assert_eq!(execution.calls, [(0, 3), (1, 2)]);
        assert_eq!(execution.commits, 1);
    }

    #[test]
    fn symbol_limit_advances_a_frame() {
        let mut execution = Scripted {
            frames: 2,
            prediction: 0,
            calls: Vec::new(),
            commits: 0,
            always_emit: true,
        };
        let mut decoder = GreedyRnnt::new(13, 2).unwrap();
        assert_eq!(decoder.decode(&mut execution).unwrap(), [9, 9, 9, 9]);
        assert_eq!(execution.commits, 4);
    }

    #[test]
    fn detokenizes_sentencepiece_boundaries_and_punctuation() {
        let vocabulary = vec![
            "▁hello".into(),
            "▁world".into(),
            "▁!".into(),
            "<en-US>".into(),
        ];
        assert_eq!(
            detokenize_sentencepiece(&vocabulary, &[0, 1, 2, 3]),
            "hello world!"
        );
    }

    #[test]
    fn applies_compiled_frame_transform_to_partial_input() {
        let stages = [[3, 2, 2, 1], [3, 2, 2, 1], [1, 1, 0, 0], [3, 2, 2, 1]];
        assert_eq!(transformed_frame_count(1600, &stages).unwrap(), 201);
        assert_eq!(transformed_frame_count(433, &stages).unwrap(), 55);
        assert!(transformed_frame_count(1, &[[3, 1, 0, 0]]).is_err());
        assert_eq!(
            bounded_transformed_frame_count(799, 800, &[[1, 1, 0, 0]], 2).unwrap(),
            800
        );
    }
}
