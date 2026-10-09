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
    /// TDT: the duration-logit argmax of each row the last [`Self::joint_argmax`] evaluated.
    fn joint_durations(&self) -> &[u32] {
        &[]
    }
    fn commit_prediction(&mut self);
}

pub struct GreedyRnnt {
    blank_id: u32,
    max_symbols_per_frame: usize,
    previous_token: u32,
    predictor_valid: bool,
    /// TDT: frames each duration logit advances; empty for plain RNNT.
    tdt_durations: Vec<u32>,
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
    /// TDT packets: `joint.duration_ids` and the frames each duration logit advances.
    duration_ids: Option<PacketTensor>,
    tdt_durations: Vec<u32>,
    /// Full-context encoders: the valid frame count of the padded bucket, written per run.
    valid_rows: Option<PacketTensor>,
    state_zeros: Vec<u8>,
    input_frames: Option<usize>,
    frame_transform: Vec<[usize; 4]>,
    trailing_frames: usize,
    profiling: bool,
    last_profile: Option<RnntProfile>,
    stream: Option<StreamBinding>,
}

/// A cache-aware encoder stream (`stream.*` pipeline roles): each step turns one mel window into
/// `rows` encoder frames whose joint rows the ordinary greedy loop decodes.
struct StreamBinding {
    first: Vec<usize>,
    step: Vec<usize>,
    input: PacketTensor,
    key_start: PacketTensor,
    /// The packet's live stream caches, then the predictor banks: what a session saves.
    states: Vec<PacketTensor>,
    rows: usize,
    left_rows: usize,
    first_input_frames: usize,
    step_input_frames: usize,
    history_input_frames: usize,
    bins: usize,
    /// Released sessions' state copies, reused by the next open.
    pool: Vec<Vec<PacketTensor>>,
}

/// One open stream: a device copy of its state and the greedy decoder's position.
pub struct RnntStream {
    saved: Vec<PacketTensor>,
    previous_token: u32,
    active_bank: usize,
    steps: usize,
}

/// Mel frames the stream reads per step: the first window, later windows, and how many of a later
/// window's leading frames repeat the previous step's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamWindows {
    pub first: usize,
    pub step: usize,
    pub history: usize,
    pub bins: usize,
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
        let (duration_ids, tdt_durations) = match pipeline.optional_parameter("tdt.durations") {
            None => (None, Vec::new()),
            Some(count) => {
                let durations = (0..count)
                    .map(|index| {
                        u32::try_from(pipeline.parameter(&format!("tdt.duration.{index}"))?)
                            .map_err(|_| RuntimeError::Rejected("TDT duration overflows".into()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let tensor = pipeline.tensor("joint.duration_ids")?;
                if durations.is_empty() || tensor.bytes != ids.bytes {
                    return Err(RuntimeError::Rejected("packet TDT geometry is inconsistent".into()));
                }
                (Some(tensor), durations)
            }
        };
        let valid_rows = pipeline.tensor("encoder.valid_rows").ok();
        if valid_rows.is_some_and(|tensor| tensor.bytes != std::mem::size_of::<u32>()) {
            return Err(RuntimeError::Rejected("packet encoder valid-rows tensor is not a u32".into()));
        }
        let stream = if backend == "cuda" && pipeline.program("stream.init").is_ok() {
            let usize_param = |name| usize_parameter(&pipeline, name);
            let input_tensor = pipeline.tensor("stream.input")?;
            let bins = usize_param("audio.frontend.bins")?;
            let mut stream_states = pipeline.tensor_sequence("stream.state")?;
            stream_states.extend(states.iter().copied());
            let binding = StreamBinding {
                first: pipeline.program_sequence("stream.first")?,
                step: pipeline.program_sequence("stream.step")?,
                input: input_tensor,
                key_start: pipeline.tensor("stream.key_start")?,
                states: stream_states,
                rows: usize_param("stream.rows")?,
                left_rows: usize_param("stream.left_rows")?,
                first_input_frames: usize_param("stream.first_input_frames")?,
                step_input_frames: usize_param("stream.step_input_frames")?,
                history_input_frames: usize_param("stream.history_input_frames")?,
                bins,
                pool: Vec::new(),
            };
            if binding.rows == 0
                || binding.rows > frames
                || bins == 0
                || binding.step_input_frames.max(binding.first_input_frames) * bins * 4 > input_tensor.bytes
                || binding.key_start.bytes != 4
            {
                return Err(RuntimeError::Rejected("packet stream geometry is invalid".into()));
            }
            // The per-layer position projections are input independent: fill them once.
            runtime.run(pipeline.program("stream.init")?)?;
            Some(binding)
        } else {
            None
        };
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
            duration_ids,
            tdt_durations,
            valid_rows,
            state_zeros,
            input_frames,
            frame_transform,
            trailing_frames,
            profiling: false,
            last_profile: None,
            stream,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// The mel windows of the packet's encoder stream, if it carries one this backend can run.
    pub fn stream_windows(&self) -> Option<StreamWindows> {
        self.stream.as_ref().map(|s| StreamWindows {
            first: s.first_input_frames,
            step: s.step_input_frames,
            history: s.history_input_frames,
            bins: s.bins,
        })
    }

    /// Open a stream: zeroed caches and predictor, the decoder at the blank token.
    pub fn stream_open(&mut self) -> Result<RnntStream> {
        let binding = self.stream.as_mut().ok_or_else(|| RuntimeError::Rejected("packet has no encoder stream".into()))?;
        let saved = match binding.pool.pop() {
            Some(saved) => {
                for &tensor in &saved {
                    self.runtime.write_tensor(tensor, &vec![0; tensor.bytes])?;
                }
                saved
            }
            None => binding
                .states
                .iter()
                .map(|state| self.runtime.create_tensor(state.bytes))
                .collect::<Result<_>>()?,
        };
        Ok(RnntStream { saved, previous_token: self.blank_id, active_bank: 0, steps: 0 })
    }

    pub fn stream_close(&mut self, stream: RnntStream) {
        if let Some(binding) = &mut self.stream {
            binding.pool.push(stream.saved);
        }
    }

    /// One stream step: `window` holds `StreamWindows::first` mel frames on the stream's first step
    /// and `StreamWindows::step` after (its leading `history` frames repeat the previous window's
    /// last ones). Returns the tokens the step's encoder rows decode to, reporting each through
    /// `on_emit` as it is emitted (with this step's tokens so far).
    pub fn stream_step(
        &mut self,
        stream: &mut RnntStream,
        window: &[f32],
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
        let binding = self.stream.take().ok_or_else(|| RuntimeError::Rejected("packet has no encoder stream".into()))?;
        self.runtime.begin_execution()?;
        let result = self.stream_step_active(&binding, stream, window, on_emit);
        let ended = self.runtime.end_execution();
        self.stream = Some(binding);
        let tokens = result?;
        ended?;
        stream.steps += 1;
        Ok(tokens)
    }

    fn stream_step_active(
        &mut self,
        binding: &StreamBinding,
        stream: &mut RnntStream,
        window: &[f32],
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
        let first = stream.steps == 0;
        let frames = if first { binding.first_input_frames } else { binding.step_input_frames };
        if window.len() != frames * binding.bins || stream.saved.len() != binding.states.len() {
            return Err(RuntimeError::Rejected(format!(
                "stream window has {} values, expected {}",
                window.len(),
                frames * binding.bins
            )));
        }
        self.runtime.write_tensor_at(binding.input, 0, bytemuck::cast_slice(window))?;
        for (&live, &saved) in binding.states.iter().zip(&stream.saved) {
            self.runtime.copy_tensor(saved, 0, live, 0, live.bytes)?;
        }
        // Keys before this window row are not filled yet: the stream's first chunks.
        let key_start = binding.left_rows.saturating_sub(stream.steps.saturating_mul(binding.rows)) as u32;
        self.runtime.write_tensor(binding.key_start, &key_start.to_ne_bytes())?;
        let started = Instant::now();
        self.runtime.run_sequence(if first { &binding.first } else { &binding.step })?;
        let encoder_us = self.runtime.last_run_us();
        let encoder_wall = started.elapsed();
        let mut execution = PacketRnntExecution {
            runtime: &mut *self.runtime,
            pipeline: &self.pipeline,
            frames: binding.rows,
            token: self.token,
            ids: self.ids,
            encoder_joint: self.encoder_joint,
            encoder_window: self.encoder_window,
            row_bytes: self.row_bytes,
            predictor_programs: self.predictor_programs,
            active_bank: stream.active_bank,
            pending_predictor: None,
            all_ids: vec![0; self.joint_batch_max],
            duration_ids: self.duration_ids,
            all_durations: vec![0; self.joint_batch_max],
            profile: None,
        };
        // The predictor reruns from the saved bank each step (idempotent), so only its banks persist.
        let mut decoder = GreedyRnnt::new(self.blank_id, self.max_symbols_per_frame)?.with_tdt_durations(&self.tdt_durations);
        decoder.previous_token = stream.previous_token;
        let tokens = decoder.decode_with(&mut execution, on_emit)?;
        stream.active_bank = execution.active_bank;
        stream.previous_token = decoder.previous_token;
        tracing::debug!(
            step = stream.steps,
            encoder_us,
            encoder_wall_us = encoder_wall.as_micros() as u64,
            decode_us = (started.elapsed() - encoder_wall).as_micros() as u64,
            tokens = tokens.len(),
            "rnnt stream step"
        );
        for (&live, &saved) in binding.states.iter().zip(&stream.saved) {
            self.runtime.copy_tensor(live, 0, saved, 0, live.bytes)?;
        }
        Ok(tokens)
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
            &mut |_| {},
        )
    }

    pub fn transcribe_input_frames(&mut self, input: &[f32], valid_input_frames: usize) -> Result<Vec<u32>> {
        self.transcribe_input_frames_with(input, valid_input_frames, &mut |_| {})
    }

    /// [`Self::transcribe_input_frames`], reporting the tokens emitted so far after each one.
    pub fn transcribe_input_frames_with(
        &mut self,
        input: &[f32],
        valid_input_frames: usize,
        on_emit: &mut dyn FnMut(&[u32]),
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
        self.transcribe_input_with_encoder_frames(input, frames, encoder, on_emit)
    }

    fn transcribe_input_with_encoder_frames(
        &mut self,
        input: &[f32],
        frames: usize,
        encoder: usize,
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
        self.runtime.begin_execution()?;
        let result = self.transcribe_active(input, frames, encoder, on_emit);
        let ended = self.runtime.end_execution();
        result.and_then(|tokens| ended.map(|()| tokens))
    }

    fn transcribe_active(
        &mut self,
        input: &[f32],
        frames: usize,
        encoder: usize,
        on_emit: &mut dyn FnMut(&[u32]),
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
        if let Some(valid_rows) = self.valid_rows {
            let valid = u32::try_from(frames)
                .map_err(|_| RuntimeError::Rejected("encoder frame count overflows".into()))?;
            self.runtime.write_tensor(valid_rows, &valid.to_ne_bytes())?;
        }
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
            duration_ids: self.duration_ids,
            all_durations: vec![0; self.joint_batch_max],
            profile: profile.as_mut(),
        };
        let result =
            GreedyRnnt::new(self.blank_id, self.max_symbols_per_frame)?.with_tdt_durations(&self.tdt_durations).decode_with(&mut execution, on_emit);
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
    /// TDT: `joint.duration_ids`, read beside the token ids.
    duration_ids: Option<PacketTensor>,
    all_durations: Vec<u32>,
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
        if let Some(durations) = self.duration_ids {
            self.runtime
                .read_tensor(durations, bytemuck::cast_slice_mut(&mut self.all_durations))?;
        }
        if let (Some(profile), Some(transfer)) = (&mut self.profile, transfer) {
            profile.transfer_us += transfer.elapsed().as_secs_f64() * 1e6;
        }
        output[..rows].copy_from_slice(&self.all_ids[..rows]);
        Ok(rows)
    }

    fn joint_durations(&self) -> &[u32] {
        &self.all_durations
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
            tdt_durations: Vec::new(),
        })
    }

    /// Token-and-duration transducer decoding: each joint row also picks how many frames to
    /// advance, `durations[argmax]`.
    pub fn with_tdt_durations(mut self, durations: &[u32]) -> Self {
        self.tdt_durations = durations.to_vec();
        self
    }

    pub fn reset(&mut self) {
        self.previous_token = self.blank_id;
        self.predictor_valid = false;
    }

    /// Decode one encoder chunk. A fixed predictor state evaluates every remaining frame in one
    /// packet dispatch; the first nonblank token invalidates that speculative tail.
    pub fn decode(&mut self, execution: &mut dyn RnntExecution) -> Result<Vec<u32>> {
        self.decode_with(execution, &mut |_| {})
    }

    /// [`Self::decode`], calling `on_emit` with every token emitted so far after each one.
    pub fn decode_with(
        &mut self,
        execution: &mut dyn RnntExecution,
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
        if !self.tdt_durations.is_empty() {
            return self.decode_tdt(execution, on_emit);
        }
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
            on_emit(&emitted);
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

    /// NeMo's greedy TDT loop, walked over one joint evaluation per predictor state: the rows
    /// past `frame` share the predictor, so blanks jump by their duration without another
    /// dispatch. A blank that predicts 0 frames would repeat unchanged until the symbol limit,
    /// so it advances one frame; a token that predicts 0 stays on its frame up to that limit.
    fn decode_tdt(
        &mut self,
        execution: &mut dyn RnntExecution,
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
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
            let window = frame;
            let evaluated = execution.joint_argmax(window, &mut ids[..frames - window])?;
            if evaluated == 0 || evaluated > frames - window || execution.joint_durations().len() < evaluated {
                return Err(RuntimeError::Device("TDT joint returned an invalid frame count".into()));
            }
            while frame < window + evaluated {
                let row = frame - window;
                let skip = *self
                    .tdt_durations
                    .get(execution.joint_durations()[row] as usize)
                    .ok_or_else(|| RuntimeError::Device("TDT duration index is out of range".into()))?
                    as usize;
                let token = ids[row];
                // NeMo counts every step at a frame, blanks included; a step that lands on the
                // limit advances one frame more than its duration.
                symbols_at_frame += 1;
                let at_limit = usize::from(symbols_at_frame == self.max_symbols_per_frame);
                if token == self.blank_id {
                    frame += if skip == 0 { 1 } else { skip + at_limit };
                    symbols_at_frame = 0;
                    continue;
                }
                emitted.push(token);
                on_emit(&emitted);
                execution.commit_prediction();
                self.previous_token = token;
                self.predictor_valid = false;
                if skip > 0 || at_limit == 1 {
                    frame += skip + at_limit;
                    symbols_at_frame = 0;
                }
                break;
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

    /// Per predictor state (0, 1, ...): `(token, duration index)` for every frame.
    struct ScriptedTdt {
        rows: Vec<Vec<(u32, u32)>>,
        prediction: usize,
        calls: Vec<usize>,
        durations: Vec<u32>,
    }

    impl RnntExecution for ScriptedTdt {
        fn frames(&self) -> usize {
            self.rows[0].len()
        }

        fn predict(&mut self, _previous_token: u32) -> Result<()> {
            self.prediction += 1;
            Ok(())
        }

        fn joint_argmax(&mut self, first_frame: usize, output: &mut [u32]) -> Result<usize> {
            self.calls.push(first_frame);
            let rows = &self.rows[self.prediction - 1][first_frame..];
            for (out, &(token, _)) in output.iter_mut().zip(rows) {
                *out = token;
            }
            self.durations = rows.iter().map(|&(_, d)| d).collect();
            Ok(rows.len())
        }

        fn joint_durations(&self) -> &[u32] {
            &self.durations
        }

        fn commit_prediction(&mut self) {}
    }

    #[test]
    fn tdt_jumps_blanks_by_duration_and_stays_on_zero_duration_tokens() {
        const B: u32 = 13;
        // Durations [0, 1, 2, 3, 4]: index = frames advanced.
        let mut execution = ScriptedTdt {
            rows: vec![
                // Blank skips 2 to frame 2, which emits 7 and stays (duration 0).
                vec![(B, 2), (9, 1), (7, 0), (B, 1), (B, 1), (B, 1)],
                // Same frame: 8 advances 1 to frame 3; frame 3's blank with duration 0 moves 1.
                vec![(B, 1), (B, 1), (8, 1), (B, 0), (B, 4), (5, 1)],
                // Frame 4 blank jumps 4, past the end.
                vec![(B, 1), (B, 1), (B, 1), (B, 1), (B, 4), (5, 1)],
            ],
            prediction: 0,
            calls: Vec::new(),
            durations: Vec::new(),
        };
        let mut decoder = GreedyRnnt::new(B, 10).unwrap().with_tdt_durations(&[0, 1, 2, 3, 4]);
        assert_eq!(decoder.decode(&mut execution).unwrap(), [7, 8]);
        assert_eq!(execution.calls, [0, 2, 3]);

        // A zero-duration token at the symbol limit moves one frame on.
        let mut limited = ScriptedTdt {
            rows: vec![vec![(4, 0), (B, 4)], vec![(4, 0), (B, 4)], vec![(4, 0), (B, 4)]],
            prediction: 0,
            calls: Vec::new(),
            durations: Vec::new(),
        };
        let mut decoder = GreedyRnnt::new(B, 2).unwrap().with_tdt_durations(&[0, 1, 2, 3, 4]);
        assert_eq!(decoder.decode(&mut limited).unwrap(), [4, 4]);
        assert_eq!(limited.calls, [0, 0, 1]);
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
