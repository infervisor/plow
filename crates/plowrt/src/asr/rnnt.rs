//! Model-independent greedy RNNT control over packetized predictor and joint programs.

use std::path::Path;
use std::time::Instant;

use crate::asr::frontend::PacketLogMelFrontend;
use crate::exec::packet_runtime::{
    load_packet_runtime, BoundPacketPipeline, PacketAsset, PacketRuntime, PacketTensor,
};
use crate::{Result, RuntimeError};
use plow_asset::speech_contract::{self, TokenOutput};

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
    batch: Option<BatchBinding>,
    /// Batched streams; preferred over `stream` (CUDA, with the batched decode).
    chunk: Option<ChunkBinding>,
    /// Upload streams (`upload.*` roles): batched streams of more rows per step in `chunk`'s
    /// cache slots.
    upload: Option<ChunkBinding>,
    /// Cache rows no open batched stream holds.
    free: Vec<u32>,
    /// Vocabulary, detokenizer and language (ASR contract >= 1).
    output: Option<TokenOutput>,
}

/// The batched decode programs (`batch.*` pipeline roles): slot-indexed predictor state and a
/// table of encoder rows the joint reads by index.
struct BatchBinding {
    /// (recordings, program) and (joint rows, program), ascending.
    predict: Vec<(usize, usize)>,
    joint: Vec<(usize, usize)>,
    token: PacketTensor,
    state_index: PacketTensor,
    slot: PacketTensor,
    row_index: PacketTensor,
    row_slot: PacketTensor,
    ids: PacketTensor,
    encoder: PacketTensor,
    /// Predictor state rows; the last is the scratch slot of padding lanes.
    slots: u32,
    encoder_rows: u32,
}

impl BatchBinding {
    fn bind(pipeline: &BoundPacketPipeline, slots: u64, row_bytes: usize) -> Result<Self> {
        let rungs = |prefix: &str| -> Vec<(usize, usize)> {
            let mut rungs: Vec<_> = pipeline
                .programs()
                .filter_map(|(role, program)| {
                    Some((role.strip_prefix(prefix)?.parse().ok()?, program))
                })
                .collect();
            rungs.sort_unstable();
            rungs
        };
        let invalid = || RuntimeError::Rejected("packet batched RNNT geometry is invalid".into());
        let predict = rungs("batch.predict.");
        let joint = rungs("batch.joint.");
        let slots = u32::try_from(slots).map_err(|_| invalid())?;
        let encoder_rows =
            u32::try_from(pipeline.parameter("batch.encoder_rows")?).map_err(|_| invalid())?;
        let binding = Self {
            token: pipeline.tensor("batch.token")?,
            state_index: pipeline.tensor("batch.state_index")?,
            slot: pipeline.tensor("batch.slot")?,
            row_index: pipeline.tensor("batch.row_index")?,
            row_slot: pipeline.tensor("batch.row_slot")?,
            ids: pipeline.tensor("batch.ids")?,
            encoder: pipeline.tensor("batch.encoder")?,
            predict,
            joint,
            slots,
            encoder_rows,
        };
        let (Some(&(items, _)), Some(&(rows, _))) = (binding.predict.last(), binding.joint.last())
        else {
            return Err(invalid());
        };
        let lanes = [binding.token, binding.state_index, binding.slot];
        let row_lanes = [binding.row_index, binding.row_slot, binding.ids];
        if slots < 2
            || lanes.iter().any(|t| t.bytes != items * 4)
            || row_lanes.iter().any(|t| t.bytes != rows * 4)
            || binding.encoder.bytes != encoder_rows as usize * row_bytes
        {
            return Err(invalid());
        }
        Ok(binding)
    }
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

/// Batched cache-aware streams (`chunk.*` pipeline roles, or an upload geometry's `upload.*`): a
/// launch steps many streams, each with its caches in a row of slot-indexed tables, and the
/// batched decode takes their rows.
struct ChunkBinding {
    /// (streams, first-window, later-window and step program sequences), ascending.
    rungs: Vec<(usize, [Vec<usize>; 3])>,
    first: PacketTensor,
    next: PacketTensor,
    index: PacketTensor,
    slot: PacketTensor,
    pos: PacketTensor,
    key_start: PacketTensor,
    rows: usize,
    left_rows: usize,
    first_input_frames: usize,
    step_input_frames: usize,
    history_input_frames: usize,
    bins: usize,
    next_row0: usize,
    encoder_row0: usize,
    predictor_slot0: u32,
    /// The scratch cache row of padding items (the tables' last).
    scratch: u32,
    /// Host staging of a launch's windows and per-item tables.
    windows: [Vec<f32>; 2],
    lanes: [Vec<u32>; 4],
}

impl ChunkBinding {
    /// `role`: `chunk`, or `upload` for the upload geometry (the cache slots, encoder-table rows
    /// and predictor slots are the `chunk.*` parameters' either way).
    fn bind(
        pipeline: &BoundPacketPipeline,
        batch: &BatchBinding,
        bins: usize,
        role: &str,
    ) -> Result<Self> {
        let invalid = || RuntimeError::Rejected("packet batched stream geometry is invalid".into());
        let param = |name| usize_parameter(pipeline, name);
        let step = format!("{role}.step.");
        let mut items: Vec<usize> = pipeline
            .programs()
            .filter_map(|(name, _)| name.strip_prefix(&step)?.split('.').next()?.parse().ok())
            .collect();
        items.sort_unstable();
        items.dedup();
        let rungs = items
            .into_iter()
            .map(|n| {
                let sequence = |kind| pipeline.program_sequence(&format!("{role}.{kind}.{n}"));
                Ok((
                    n,
                    [sequence("first")?, sequence("next")?, sequence("step")?],
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let slots = u32::try_from(param("chunk.slots")?).map_err(|_| invalid())?;
        let tensor = |name| pipeline.tensor(&format!("{role}.{name}"));
        let (first, next) = (tensor("input.first")?, tensor("input.next")?);
        let (index, slot) = (tensor("index")?, tensor("slot")?);
        // Rows per step and window frames follow from the per-stream tensors' sizes.
        let max_items = slot.bytes / 4;
        let per_stream = |bytes: usize, unit: usize| {
            (max_items != 0 && unit != 0 && bytes % (max_items * unit) == 0)
                .then(|| bytes / (max_items * unit))
                .ok_or_else(invalid)
        };
        let rows = per_stream(index.bytes, 4)?;
        let first_input_frames = per_stream(first.bytes, bins * 4)?;
        let step_input_frames = per_stream(next.bytes, bins * 4)?;
        let binding = Self {
            first,
            next,
            index,
            slot,
            pos: tensor("pos")?,
            key_start: tensor("key_start")?,
            rows,
            left_rows: param("chunk.left_rows")?,
            first_input_frames,
            step_input_frames,
            history_input_frames: step_input_frames
                .checked_sub(first_input_frames)
                .ok_or_else(invalid)?,
            bins,
            // An upload geometry's later windows start past its widest step's first ones.
            next_row0: match role {
                "chunk" => param("chunk.next_row0")?,
                _ => max_items * rows,
            },
            encoder_row0: param("chunk.encoder_row0")?,
            predictor_slot0: u32::try_from(param("chunk.predictor_slot0")?)
                .map_err(|_| invalid())?,
            scratch: slots.saturating_sub(1),
            windows: Default::default(),
            lanes: Default::default(),
            rungs,
        };
        let streams = slots.saturating_sub(1);
        if binding.rungs.last().is_none_or(|rung| rung.0 != max_items)
            || streams == 0
            || rows == 0
            || [binding.pos, binding.key_start]
                .iter()
                .any(|t| t.bytes != max_items * 4)
            || binding.next_row0 < max_items * rows
            || binding.encoder_row0 + max_items * rows > batch.encoder_rows as usize
            || binding.predictor_slot0 + streams >= batch.slots
            || batch
                .predict
                .last()
                .is_none_or(|&(lanes, _)| lanes < max_items)
            || batch
                .joint
                .last()
                .is_none_or(|&(rows, _)| rows < max_items * binding.rows)
        {
            return Err(invalid());
        }
        Ok(binding)
    }
}

impl ChunkBinding {
    fn windows(&self) -> StreamWindows {
        StreamWindows {
            first: self.first_input_frames,
            step: self.step_input_frames,
            history: self.history_input_frames,
            bins: self.bins,
            rows: self.rows,
        }
    }
}

/// One open stream: its state (a device copy, or a batched stream's cache row) and the greedy
/// decoder's position.
pub struct RnntStream {
    saved: Vec<PacketTensor>,
    previous_token: u32,
    active_bank: usize,
    steps: usize,
    /// A batched stream's cache row; its predictor slot holds the output for `previous_token`
    /// (`predicted`), or no state yet (`fresh`).
    slot: Option<u32>,
    predicted: bool,
    fresh: bool,
    /// An upload stream ([`PacketRnnt::upload_open`]): it steps in the upload geometry.
    upload: bool,
}

impl RnntStream {
    pub fn is_upload(&self) -> bool {
        self.upload
    }
}

/// Mel frames the stream reads per step: the first window, later windows, and how many of a later
/// window's leading frames repeat the previous step's; the encoder rows a step adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamWindows {
    pub first: usize,
    pub step: usize,
    pub history: usize,
    pub bins: usize,
    pub rows: usize,
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
                    return Err(RuntimeError::Rejected(
                        "packet TDT geometry is inconsistent".into(),
                    ));
                }
                (Some(tensor), durations)
            }
        };
        let valid_rows = pipeline.tensor("encoder.valid_rows").ok();
        if valid_rows.is_some_and(|tensor| tensor.bytes != std::mem::size_of::<u32>()) {
            return Err(RuntimeError::Rejected(
                "packet encoder valid-rows tensor is not a u32".into(),
            ));
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
                || binding.step_input_frames.max(binding.first_input_frames) * bins * 4
                    > input_tensor.bytes
                || binding.key_start.bytes != 4
            {
                return Err(RuntimeError::Rejected(
                    "packet stream geometry is invalid".into(),
                ));
            }
            // The per-layer position projections are input independent: fill them once.
            runtime.run(pipeline.program("stream.init")?)?;
            Some(binding)
        } else {
            None
        };
        let batch = pipeline
            .optional_parameter("batch.slots")
            .map(|slots| BatchBinding::bind(&pipeline, slots, row_bytes))
            .transpose()?;
        let mut chunks = |role: &str| -> Result<Option<ChunkBinding>> {
            match &batch {
                Some(batch)
                    if backend == "cuda" && pipeline.program(&format!("{role}.init")).is_ok() =>
                {
                    let bins = usize_parameter(&pipeline, "audio.frontend.bins")?;
                    let binding = ChunkBinding::bind(&pipeline, batch, bins, role)?;
                    runtime.run(pipeline.program(&format!("{role}.init"))?)?;
                    Ok(Some(binding))
                }
                _ => Ok(None),
            }
        };
        let chunk = chunks("chunk")?;
        let upload = match chunk {
            Some(_) => chunks("upload")?,
            None => None,
        };
        let free = chunk
            .as_ref()
            .map_or(Vec::new(), |c| (0..c.scratch).rev().collect());
        let state_zeros = vec![0; states.iter().map(|state| state.bytes).max().unwrap_or(0)];
        let output = match pipeline.optional_parameter(speech_contract::CONTRACT) {
            None => None,
            Some(_) => {
                let vocabulary = asset
                    .metadata(speech_contract::VOCABULARY_SECTION)
                    .ok_or_else(|| {
                        RuntimeError::Rejected(format!(
                            "packet is missing {}",
                            speech_contract::VOCABULARY_SECTION
                        ))
                    })?;
                let output = TokenOutput::from_pipeline(
                    |name| pipeline.optional_parameter(name),
                    |name| pipeline.optional_string(name).map(str::to_owned),
                    vocabulary,
                )
                .map_err(RuntimeError::Rejected)?;
                if output.word_boundary.is_empty() {
                    return Err(RuntimeError::Rejected(
                        "ASR output word boundary is empty".into(),
                    ));
                }
                Some(output)
            }
        };
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
            batch,
            chunk,
            upload,
            free,
            output,
        })
    }

    /// The packet's output contract; `None` for a contract-0 packet.
    pub fn output(&self) -> Option<&TokenOutput> {
        self.output.as_ref()
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// The mel windows of the packet's encoder stream, if it carries one this backend can run.
    pub fn stream_windows(&self) -> Option<StreamWindows> {
        if let Some(c) = &self.chunk {
            return Some(c.windows());
        }
        self.stream.as_ref().map(|s| StreamWindows {
            first: s.first_input_frames,
            step: s.step_input_frames,
            history: s.history_input_frames,
            bins: s.bins,
            rows: s.rows,
        })
    }

    /// An upload stream's windows ([`Self::upload_open`]); `None` without upload streams.
    pub fn upload_windows(&self) -> Option<StreamWindows> {
        self.upload.as_ref().map(ChunkBinding::windows)
    }

    /// Streams one [`Self::stream_step_batch`] steps at most: 1 without batched streams.
    pub fn stream_batch(&self) -> usize {
        self.chunk
            .as_ref()
            .and_then(|c| c.rungs.last())
            .map_or(1, |rung| rung.0)
    }

    /// Upload streams one [`Self::stream_step_batch`] steps at most: 0 without upload streams.
    pub fn upload_batch(&self) -> usize {
        self.upload
            .as_ref()
            .and_then(|c| c.rungs.last())
            .map_or(0, |rung| rung.0)
    }

    /// Batched streams that can be open at once; `None`: unbounded (one state copy each).
    pub fn stream_slots(&self) -> Option<usize> {
        self.chunk.as_ref().map(|c| c.scratch as usize)
    }

    /// Encoder rows the offline transcription of `valid_input_frames` mel frames decodes: a
    /// stream's final steps decode up to the same row.
    pub fn final_rows(&self, valid_input_frames: usize) -> Result<usize> {
        match self.encoder_for(valid_input_frames) {
            Ok((_, frames)) => Ok(frames),
            Err(_) => Ok(
                transformed_frame_count(valid_input_frames, &self.frame_transform)?
                    + self.trailing_frames,
            ),
        }
    }

    /// Open a stream: zeroed caches and predictor, the decoder at the blank token.
    pub fn stream_open(&mut self) -> Result<RnntStream> {
        let mut stream = RnntStream {
            saved: Vec::new(),
            previous_token: self.blank_id,
            active_bank: 0,
            steps: 0,
            slot: None,
            predicted: false,
            fresh: true,
            upload: false,
        };
        if self.chunk.is_some() {
            // Caches read as zeros at a stream's first step, so a reused row needs no reset.
            stream.slot = Some(self.free.pop().ok_or_else(|| {
                RuntimeError::Overloaded("every ASR stream slot is open".into())
            })?);
            return Ok(stream);
        }
        let binding = self
            .stream
            .as_mut()
            .ok_or_else(|| RuntimeError::Rejected("packet has no encoder stream".into()))?;
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
        stream.saved = saved;
        Ok(stream)
    }

    /// Open an upload stream: a batched stream that steps [`Self::upload_windows`] in the
    /// streams' cache slots (a whole recording at a time, its windows pushed at once).
    pub fn upload_open(&mut self) -> Result<RnntStream> {
        if self.upload.is_none() {
            return Err(RuntimeError::Rejected(
                "packet has no upload streams".into(),
            ));
        }
        let mut stream = self.stream_open()?;
        stream.upload = true;
        Ok(stream)
    }

    pub fn stream_close(&mut self, stream: RnntStream) {
        match (stream.slot, &mut self.stream) {
            (Some(slot), _) => self.free.push(slot),
            (None, Some(binding)) => binding.pool.push(stream.saved),
            _ => {}
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
        if stream.slot.is_some() {
            let chunk = if stream.upload { &self.upload } else { &self.chunk };
            let rows = chunk.as_ref().map_or(0, |c| c.rows);
            let tokens = self
                .stream_step_batch(&mut [(stream, window, rows)])?
                .pop()
                .unwrap_or_default();
            if !tokens.is_empty() {
                on_emit(&tokens);
            }
            return Ok(tokens);
        }
        let binding = self
            .stream
            .take()
            .ok_or_else(|| RuntimeError::Rejected("packet has no encoder stream".into()))?;
        self.runtime.begin_execution()?;
        let result = self.stream_step_active(&binding, stream, window, on_emit);
        let ended = self.runtime.end_execution();
        self.stream = Some(binding);
        let tokens = result?;
        ended?;
        stream.steps += 1;
        Ok(tokens)
    }

    /// One step of several batched streams in one launch (at most [`Self::stream_batch`], or
    /// [`Self::upload_batch`] upload streams; one kind per launch): each `(stream, window, rows)`
    /// takes `window` as [`Self::stream_step`] does and decodes the first `rows` of the step's
    /// encoder rows (all of them but on a stream's final step). Returns each stream's tokens, as
    /// its steps alone decode them.
    pub fn stream_step_batch(
        &mut self,
        steps: &mut [(&mut RnntStream, &[f32], usize)],
    ) -> Result<Vec<Vec<u32>>> {
        let upload = steps.first().is_some_and(|(stream, ..)| stream.upload);
        if steps.iter().any(|(stream, ..)| stream.upload != upload) {
            return Err(RuntimeError::Rejected(
                "a batched stream step takes live or upload streams, not both".into(),
            ));
        }
        let binding = if upload {
            &mut self.upload
        } else {
            &mut self.chunk
        };
        let (Some(mut chunk), Some(batch)) = (binding.take(), self.batch.take()) else {
            return Err(RuntimeError::Rejected(
                "packet has no batched encoder stream".into(),
            ));
        };
        self.runtime.begin_execution()?;
        let result = self.chunk_step_active(&mut chunk, &batch, steps);
        let ended = self.runtime.end_execution();
        *(if upload {
            &mut self.upload
        } else {
            &mut self.chunk
        }) = Some(chunk);
        self.batch = Some(batch);
        result.and_then(|tokens| ended.map(|()| tokens))
    }

    fn chunk_step_active(
        &mut self,
        chunk: &mut ChunkBinding,
        batch: &BatchBinding,
        steps: &mut [(&mut RnntStream, &[f32], usize)],
    ) -> Result<Vec<Vec<u32>>> {
        let rung = |count: usize| chunk.rungs.iter().position(|rung| rung.0 >= count);
        let Some(step_rung) = rung(steps.len()).filter(|_| !steps.is_empty()) else {
            return Err(RuntimeError::Rejected(format!(
                "a batched stream step takes 1..={} streams",
                chunk.rungs.last().map_or(0, |rung| rung.0)
            )));
        };
        let (r, left) = (chunk.rows, chunk.left_rows);
        let [first, next] = &mut chunk.windows;
        let [index, slot, pos, key_start] = &mut chunk.lanes;
        for buffer in [&mut *index, &mut *slot, &mut *pos, &mut *key_start] {
            buffer.clear();
        }
        first.clear();
        next.clear();
        let (mut firsts, mut nexts) = (0, 0);
        for (stream, window, rows) in steps.iter() {
            let (Some(cache_row), true) = (stream.slot, *rows <= r) else {
                return Err(RuntimeError::Rejected(
                    "batched stream step needs open batched streams".into(),
                ));
            };
            let (frames, row0) = if stream.steps == 0 {
                firsts += 1;
                (chunk.first_input_frames, (firsts - 1) * r)
            } else {
                nexts += 1;
                (chunk.step_input_frames, chunk.next_row0 + (nexts - 1) * r)
            };
            if window.len() != frames * chunk.bins {
                return Err(RuntimeError::Rejected(format!(
                    "stream window has {} values, expected {}",
                    window.len(),
                    frames * chunk.bins
                )));
            }
            if stream.steps == 0 {
                &mut *first
            } else {
                &mut *next
            }
            .extend_from_slice(window);
            index.extend((row0..row0 + r).map(|row| row as u32));
            slot.push(cache_row);
            pos.push(stream.steps.min(u32::MAX as usize) as u32);
            // Keys before this window row are not filled yet: the stream's first chunks.
            key_start.push(left.saturating_sub(stream.steps.saturating_mul(r)) as u32);
        }
        // Padding items read zeros (past the staging table, at a first step) into the scratch row.
        let lanes = chunk.rungs[step_rung].0;
        index.resize(lanes * r, u32::MAX);
        slot.resize(lanes, chunk.scratch);
        pos.resize(lanes, 0);
        key_start.resize(lanes, left as u32);
        let mut programs = Vec::new();
        for (count, kind, tensor, values) in [
            (firsts, 0, chunk.first, &*first),
            (nexts, 1, chunk.next, &*next),
        ] {
            if count != 0 {
                self.runtime
                    .write_tensor_at(tensor, 0, bytemuck::cast_slice(values))?;
                let rung = rung(count).expect("within the step rung");
                programs.extend_from_slice(&chunk.rungs[rung].1[kind]);
            }
        }
        for (tensor, values) in [
            (chunk.index, &*index),
            (chunk.slot, &*slot),
            (chunk.pos, &*pos),
            (chunk.key_start, &*key_start),
        ] {
            self.runtime
                .write_tensor_at(tensor, 0, bytemuck::cast_slice(values))?;
        }
        programs.extend_from_slice(&chunk.rungs[step_rung].1[2]);
        let started = Instant::now();
        self.runtime.run_sequence(&programs)?;
        let encoder_us = self.runtime.last_run_us();
        let encoder_wall = started.elapsed();
        let mut items: Vec<_> = steps
            .iter()
            .enumerate()
            .map(|(i, (stream, _, rows))| BatchItem {
                slot: chunk.predictor_slot0 + stream.slot.expect("checked"),
                base: chunk.encoder_row0 + i * r,
                frames: *rows,
                frame: 0,
                symbols_at_frame: 0,
                previous_token: stream.previous_token,
                predicted: stream.predicted,
                fresh: stream.fresh,
                tokens: Vec::new(),
            })
            .collect();
        let mut execution = PacketBatchExecution {
            runtime: &mut *self.runtime,
            batch,
            blank_id: self.blank_id,
            lanes: Vec::new(),
            ids: Vec::new(),
        };
        decode_batch(
            &mut execution,
            &mut items,
            self.blank_id,
            self.max_symbols_per_frame,
            self.joint_batch_max,
        )?;
        tracing::debug!(
            streams = steps.len(),
            lanes,
            encoder_us,
            encoder_wall_us = encoder_wall.as_micros() as u64,
            decode_us = (started.elapsed() - encoder_wall).as_micros() as u64,
            "rnnt batched stream step"
        );
        Ok(steps
            .iter_mut()
            .zip(items)
            .map(|((stream, _, _), item)| {
                stream.previous_token = item.previous_token;
                stream.predicted = item.predicted;
                stream.fresh = item.fresh;
                stream.steps += 1;
                item.tokens
            })
            .collect())
    }

    fn stream_step_active(
        &mut self,
        binding: &StreamBinding,
        stream: &mut RnntStream,
        window: &[f32],
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
        let first = stream.steps == 0;
        let frames = if first {
            binding.first_input_frames
        } else {
            binding.step_input_frames
        };
        if window.len() != frames * binding.bins || stream.saved.len() != binding.states.len() {
            return Err(RuntimeError::Rejected(format!(
                "stream window has {} values, expected {}",
                window.len(),
                frames * binding.bins
            )));
        }
        self.runtime
            .write_tensor_at(binding.input, 0, bytemuck::cast_slice(window))?;
        for (&live, &saved) in binding.states.iter().zip(&stream.saved) {
            self.runtime.copy_tensor(saved, 0, live, 0, live.bytes)?;
        }
        // Keys before this window row are not filled yet: the stream's first chunks.
        let key_start = binding
            .left_rows
            .saturating_sub(stream.steps.saturating_mul(binding.rows))
            as u32;
        self.runtime
            .write_tensor(binding.key_start, &key_start.to_ne_bytes())?;
        let started = Instant::now();
        self.runtime
            .run_sequence(if first { &binding.first } else { &binding.step })?;
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
        let mut decoder = GreedyRnnt::new(self.blank_id, self.max_symbols_per_frame)?
            .with_tdt_durations(&self.tdt_durations);
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

    pub fn transcribe_input_frames(
        &mut self,
        input: &[f32],
        valid_input_frames: usize,
    ) -> Result<Vec<u32>> {
        self.transcribe_input_frames_with(input, valid_input_frames, &mut |_| {})
    }

    /// [`Self::transcribe_input_frames`], reporting the tokens emitted so far after each one.
    pub fn transcribe_input_frames_with(
        &mut self,
        input: &[f32],
        valid_input_frames: usize,
        on_emit: &mut dyn FnMut(&[u32]),
    ) -> Result<Vec<u32>> {
        let (encoder, frames) = self.encoder_for(valid_input_frames)?;
        self.transcribe_input_with_encoder_frames(input, frames, encoder, on_emit)
    }

    /// Recordings one [`Self::transcribe_batch_input_frames`] call takes; 0 without the packet's
    /// batched decode.
    pub fn batch_capacity(&self) -> usize {
        self.batch.as_ref().map_or(0, |batch| match &self.chunk {
            // The slots and rows past these are the batched streams'.
            Some(chunk) => (chunk.predictor_slot0 as usize).min(chunk.encoder_row0 / self.frames),
            None => (batch.slots as usize - 1).min(batch.encoder_rows as usize / self.frames),
        })
    }

    /// [`Self::transcribe_input_frames`] for several recordings (input, valid input frames): each
    /// encoder runs alone, then one greedy decode steps them all together. Each recording's
    /// tokens equal its own [`Self::transcribe_input_frames`].
    pub fn transcribe_batch_input_frames(
        &mut self,
        inputs: &[(&[f32], usize)],
    ) -> Result<Vec<Vec<u32>>> {
        if inputs.is_empty() || inputs.len() > self.batch_capacity() {
            return Err(RuntimeError::Rejected(format!(
                "batched RNNT decode needs 1..={} recordings",
                self.batch_capacity()
            )));
        }
        let batch = self.batch.take().expect("checked by batch_capacity");
        self.runtime.begin_execution()?;
        let result = self.transcribe_batch_active(&batch, inputs);
        let ended = self.runtime.end_execution();
        self.batch = Some(batch);
        result.and_then(|tokens| ended.map(|()| tokens))
    }

    fn transcribe_batch_active(
        &mut self,
        batch: &BatchBinding,
        inputs: &[(&[f32], usize)],
    ) -> Result<Vec<Vec<u32>>> {
        let mut items = Vec::with_capacity(inputs.len());
        for (index, &(input, valid_input_frames)) in inputs.iter().enumerate() {
            let (encoder, frames) = self.encoder_for(valid_input_frames)?;
            self.encode(input, frames, encoder)?;
            let base = index * self.frames;
            self.runtime.copy_tensor(
                self.encoder_joint,
                0,
                batch.encoder,
                base * self.row_bytes,
                frames * self.row_bytes,
            )?;
            items.push(BatchItem::new(index as u32, base, frames, self.blank_id));
        }
        let mut execution = PacketBatchExecution {
            runtime: &mut *self.runtime,
            batch,
            blank_id: self.blank_id,
            lanes: Vec::new(),
            ids: Vec::new(),
        };
        decode_batch(
            &mut execution,
            &mut items,
            self.blank_id,
            self.max_symbols_per_frame,
            self.joint_batch_max,
        )?;
        Ok(items.into_iter().map(|item| item.tokens).collect())
    }

    /// The smallest encoder capacity holding `valid_input_frames` (with a stream, its decoded rows
    /// in whole stream steps), and its valid encoder frames.
    fn encoder_for(&self, valid_input_frames: usize) -> Result<(usize, usize)> {
        let capacities: Vec<usize> = self.encoder_programs.iter().map(|e| e.0).collect();
        let encoder = encoder_index(
            &capacities,
            valid_input_frames,
            &self.frame_transform,
            self.trailing_frames,
            self.stream_windows().map(|w| w.rows),
        )?
        .ok_or_else(|| {
            RuntimeError::Rejected(format!(
                "packet has no encoder capacity for {valid_input_frames} frames"
            ))
        })?;
        let frames = self.valid_encoder_frames(valid_input_frames, capacities[encoder])?;
        Ok((encoder, frames))
    }

    /// Run encoder `encoder` on `input`, leaving its rows in `encoder.joint`.
    fn encode(&mut self, input: &[f32], frames: usize, encoder: usize) -> Result<()> {
        if input.len() != self.input_elements() {
            return Err(RuntimeError::Rejected(format!(
                "packet RNNT input has {} elements, expected {}",
                input.len(),
                self.input_elements()
            )));
        }
        self.runtime
            .write_tensor(self.input, bytemuck::cast_slice(input))?;
        if let Some(valid_rows) = self.valid_rows {
            let valid = u32::try_from(frames)
                .map_err(|_| RuntimeError::Rejected("encoder frame count overflows".into()))?;
            self.runtime
                .write_tensor(valid_rows, &valid.to_ne_bytes())?;
        }
        self.runtime.run_sequence(&self.encoder_programs[encoder].1)
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
        for &state in &self.states {
            self.runtime
                .write_tensor(state, &self.state_zeros[..state.bytes])?;
        }
        self.encode(input, frames, encoder)?;
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
        let result = GreedyRnnt::new(self.blank_id, self.max_symbols_per_frame)?
            .with_tdt_durations(&self.tdt_durations)
            .decode_with(&mut execution, on_emit);
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

/// One recording of a [`decode_batch`]: encoder rows `base..base + frames` of the batch encoder
/// table, decoded with the predictor state of `slot`.
pub struct BatchItem {
    slot: u32,
    base: usize,
    frames: usize,
    frame: usize,
    symbols_at_frame: usize,
    previous_token: u32,
    /// The slot's predictor output is current (computed for `previous_token`).
    predicted: bool,
    /// The slot's state is not this recording's yet: the predictor starts from zeros.
    fresh: bool,
    tokens: Vec<u32>,
}

impl BatchItem {
    pub fn new(slot: u32, base: usize, frames: usize, blank_id: u32) -> Self {
        Self {
            slot,
            base,
            frames,
            frame: 0,
            symbols_at_frame: 0,
            previous_token: blank_id,
            predicted: false,
            fresh: true,
            tokens: Vec::new(),
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// A predictor lane of a batched step: embed `token` into the state of `slot` (zeros when `fresh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchPredict {
    pub slot: u32,
    pub token: u32,
    pub fresh: bool,
}

/// Device side of [`decode_batch`].
pub trait BatchedRnntExecution {
    /// Joint rows one step evaluates at most.
    fn max_rows(&self) -> usize;
    /// Run the predictor lanes, then the joint for `rows` (slot, encoder row), writing each row's
    /// argmax to `ids`.
    fn step(
        &mut self,
        predict: &[BatchPredict],
        rows: &[(u32, u32)],
        ids: &mut [u32],
    ) -> Result<()>;
}

/// [`GreedyRnnt::decode`] for many recordings per launch. Each step predicts for the recordings
/// that emitted (or start) and evaluates up to `window` frames of every unfinished one; a joint
/// row depends only on its frame and its recording's tokens, so each recording decodes exactly
/// as it would alone.
pub fn decode_batch(
    execution: &mut dyn BatchedRnntExecution,
    items: &mut [BatchItem],
    blank_id: u32,
    max_symbols_per_frame: usize,
    window: usize,
) -> Result<()> {
    if max_symbols_per_frame == 0 || window == 0 || items.len() > execution.max_rows() {
        return Err(RuntimeError::Rejected(
            "invalid batched RNNT decode geometry".into(),
        ));
    }
    let (mut predict, mut rows, mut spans, mut ids) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    loop {
        let active = items.iter().filter(|item| item.frame < item.frames).count();
        if active == 0 {
            return Ok(());
        }
        let width = window.min(execution.max_rows() / active);
        predict.clear();
        rows.clear();
        spans.clear();
        for (index, item) in items.iter().enumerate() {
            if item.frame >= item.frames {
                continue;
            }
            if !item.predicted {
                predict.push(BatchPredict {
                    slot: item.slot,
                    token: item.previous_token,
                    fresh: item.fresh,
                });
            }
            let count = width.min(item.frames - item.frame);
            spans.push((index, rows.len(), count));
            let first = item.base + item.frame;
            rows.extend((first..first + count).map(|row| (item.slot, row as u32)));
        }
        ids.clear();
        ids.resize(rows.len(), blank_id);
        execution.step(&predict, &rows, &mut ids)?;
        for &(index, start, count) in &spans {
            let item = &mut items[index];
            item.predicted = true;
            item.fresh = false;
            let evaluated = &ids[start..start + count];
            let Some(offset) = evaluated.iter().position(|&token| token != blank_id) else {
                item.frame += count;
                item.symbols_at_frame = 0;
                continue;
            };
            if offset != 0 {
                item.symbols_at_frame = 0;
            }
            item.frame += offset;
            let token = evaluated[offset];
            item.tokens.push(token);
            item.previous_token = token;
            item.predicted = false;
            item.symbols_at_frame += 1;
            if item.symbols_at_frame == max_symbols_per_frame {
                item.frame += 1;
                item.symbols_at_frame = 0;
            }
        }
    }
}

struct PacketBatchExecution<'a> {
    runtime: &'a mut dyn PacketRuntime,
    batch: &'a BatchBinding,
    blank_id: u32,
    lanes: Vec<u32>,
    ids: Vec<u32>,
}

impl PacketBatchExecution<'_> {
    /// The smallest rung holding `count`.
    fn rung(rungs: &[(usize, usize)], count: usize) -> Result<(usize, usize)> {
        rungs
            .iter()
            .copied()
            .find(|&(capacity, _)| capacity >= count)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!("no batched RNNT program holds {count} lanes"))
            })
    }

    /// Write `values` then `pad` up to `count` lanes.
    fn write(
        &mut self,
        tensor: PacketTensor,
        values: impl Iterator<Item = u32>,
        pad: u32,
        count: usize,
    ) -> Result<()> {
        self.lanes.clear();
        self.lanes.extend(values);
        self.lanes.resize(count, pad);
        self.runtime
            .write_tensor_at(tensor, 0, bytemuck::cast_slice(&self.lanes))
    }
}

impl BatchedRnntExecution for PacketBatchExecution<'_> {
    fn max_rows(&self) -> usize {
        self.batch.joint.last().map_or(0, |&(rows, _)| rows)
    }

    fn step(
        &mut self,
        predict: &[BatchPredict],
        rows: &[(u32, u32)],
        ids: &mut [u32],
    ) -> Result<()> {
        let batch = self.batch;
        let mut programs = Vec::with_capacity(2);
        if !predict.is_empty() {
            let (lanes, program) = Self::rung(&batch.predict, predict.len())?;
            // Padding lanes embed blank from zeros into the scratch slot.
            let (beyond, scratch) = (batch.slots, batch.slots - 1);
            self.write(
                batch.token,
                predict.iter().map(|p| p.token),
                self.blank_id,
                lanes,
            )?;
            let state = |p: &BatchPredict| if p.fresh { beyond } else { p.slot };
            self.write(batch.state_index, predict.iter().map(state), beyond, lanes)?;
            self.write(batch.slot, predict.iter().map(|p| p.slot), scratch, lanes)?;
            programs.push(program);
        }
        let (count, program) = Self::rung(&batch.joint, rows.len())?;
        // Padding rows read zeros: past the encoder table and past the slots.
        self.write(
            batch.row_index,
            rows.iter().map(|r| r.1),
            batch.encoder_rows,
            count,
        )?;
        self.write(batch.row_slot, rows.iter().map(|r| r.0), batch.slots, count)?;
        programs.push(program);
        self.runtime.run_sequence(&programs)?;
        self.ids.resize(count, 0);
        self.runtime
            .read_tensor_at(batch.ids, 0, bytemuck::cast_slice_mut(&mut self.ids))?;
        ids.copy_from_slice(&self.ids[..rows.len()]);
        Ok(())
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

/// The smallest of `capacities` (input frames, ascending) holding `valid_frames` and, with
/// `step_rows`, their decoded rows rounded up to whole steps. A stream final computes whole steps
/// and no row attends past its step, so an encoder holding them decodes the stream's rows. Past
/// the largest, the smallest holding the frames.
fn encoder_index(
    capacities: &[usize],
    valid_frames: usize,
    stages: &[[usize; 4]],
    trailing_frames: usize,
    step_rows: Option<usize>,
) -> Result<Option<usize>> {
    let Some(holding) = capacities.iter().position(|&c| c >= valid_frames) else {
        return Ok(None);
    };
    let Some(step_rows) = step_rows else {
        return Ok(Some(holding));
    };
    let rows = (transformed_frame_count(valid_frames, stages)? + trailing_frames)
        .next_multiple_of(step_rows);
    let whole = capacities[holding..]
        .iter()
        .position(|&c| transformed_frame_count(c, stages).is_ok_and(|frames| frames >= rows));
    Ok(Some(holding + whole.unwrap_or(0)))
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
            if evaluated == 0
                || evaluated > frames - window
                || execution.joint_durations().len() < evaluated
            {
                return Err(RuntimeError::Device(
                    "TDT joint returned an invalid frame count".into(),
                ));
            }
            while frame < window + evaluated {
                let row = frame - window;
                let skip = *self
                    .tdt_durations
                    .get(execution.joint_durations()[row] as usize)
                    .ok_or_else(|| {
                        RuntimeError::Device("TDT duration index is out of range".into())
                    })? as usize;
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

/// Text of `ids` under the packet's [`TokenOutput`] rules (`output.detokenizer` sentencepiece).
pub fn detokenize(output: &TokenOutput, ids: &[u32]) -> String {
    let boundary = output.word_boundary.as_str();
    let mut text = String::new();
    for &id in ids {
        let Some(piece) = output.pieces.get(id as usize) else {
            continue;
        };
        if output.skip_bracketed && piece.starts_with('<') && piece.ends_with('>') {
            continue;
        }
        let value = piece.strip_prefix(boundary).unwrap_or(piece);
        if piece.starts_with(boundary) && !output.no_space_before.iter().any(|p| p == value) {
            text.push(' ');
        }
        text.push_str(&value.replace(boundary, " "));
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
        let mut decoder = GreedyRnnt::new(B, 10)
            .unwrap()
            .with_tdt_durations(&[0, 1, 2, 3, 4]);
        assert_eq!(decoder.decode(&mut execution).unwrap(), [7, 8]);
        assert_eq!(execution.calls, [0, 2, 3]);

        // A zero-duration token at the symbol limit moves one frame on.
        let mut limited = ScriptedTdt {
            rows: vec![
                vec![(4, 0), (B, 4)],
                vec![(4, 0), (B, 4)],
                vec![(4, 0), (B, 4)],
            ],
            prediction: 0,
            calls: Vec::new(),
            durations: Vec::new(),
        };
        let mut decoder = GreedyRnnt::new(B, 2)
            .unwrap()
            .with_tdt_durations(&[0, 1, 2, 3, 4]);
        assert_eq!(decoder.decode(&mut limited).unwrap(), [4, 4]);
        assert_eq!(limited.calls, [0, 0, 1]);
    }

    /// Mostly blank (13); now and then a token, sometimes several at one frame.
    fn scripted(history: &[u32], frame: usize) -> u32 {
        let h = history
            .iter()
            .fold(frame as u64 * 0x9e37_79b9 + 7, |h, &t| {
                h.wrapping_mul(31).wrapping_add(u64::from(t) + 1)
            });
        let h = (h ^ (h >> 13)).wrapping_mul(0x5bd1_e995);
        let h = h ^ (h >> 29);
        if h % 3 == 0 {
            1 + (h >> 8) as u32 % 7
        } else {
            13
        }
    }

    /// One recording through the batch-1 interface: the joint sees the committed tokens and the
    /// pending prediction's token.
    struct ScriptedRecording {
        frames: usize,
        window: usize,
        committed: Vec<u32>,
        pending: Vec<u32>,
    }

    impl RnntExecution for ScriptedRecording {
        fn frames(&self) -> usize {
            self.frames
        }

        fn predict(&mut self, previous_token: u32) -> Result<()> {
            self.pending = self.committed.clone();
            self.pending.push(previous_token);
            Ok(())
        }

        fn joint_argmax(&mut self, first_frame: usize, output: &mut [u32]) -> Result<usize> {
            let rows = output.len().min(self.window);
            for (row, out) in output[..rows].iter_mut().enumerate() {
                *out = scripted(&self.pending, first_frame + row);
            }
            Ok(rows)
        }

        fn commit_prediction(&mut self) {
            self.committed = self.pending.clone();
        }
    }

    /// Slot states as token histories; rows are `base[slot] + frame`.
    struct ScriptedBatch {
        history: Vec<Vec<u32>>,
        base: Vec<usize>,
        max_rows: usize,
        steps: usize,
    }

    impl BatchedRnntExecution for ScriptedBatch {
        fn max_rows(&self) -> usize {
            self.max_rows
        }

        fn step(
            &mut self,
            predict: &[BatchPredict],
            rows: &[(u32, u32)],
            ids: &mut [u32],
        ) -> Result<()> {
            assert!(rows.len() <= self.max_rows && rows.len() == ids.len());
            self.steps += 1;
            for p in predict {
                let history = &mut self.history[p.slot as usize];
                if p.fresh {
                    history.clear();
                }
                history.push(p.token);
            }
            for (id, &(slot, row)) in ids.iter_mut().zip(rows) {
                let slot = slot as usize;
                *id = scripted(&self.history[slot], row as usize - self.base[slot]);
            }
            Ok(())
        }
    }

    #[test]
    fn batched_decode_matches_each_recording_alone() {
        let frames = [0, 1, 5, 17, 40, 3, 64, 9];
        for (window, max_rows, max_symbols) in [(16, 64, 10), (1, 8, 2), (4, 12, 1), (16, 8, 3)] {
            let mut alone = Vec::new();
            for &n in &frames {
                let mut recording = ScriptedRecording {
                    frames: n,
                    window: 3,
                    committed: Vec::new(),
                    pending: Vec::new(),
                };
                alone.push(
                    GreedyRnnt::new(13, max_symbols)
                        .unwrap()
                        .decode(&mut recording)
                        .unwrap(),
                );
            }
            let base: Vec<usize> = frames
                .iter()
                .scan(0, |next, &n| {
                    let base = *next;
                    *next += n + 2;
                    Some(base)
                })
                .collect();
            let mut items: Vec<_> = frames
                .iter()
                .zip(&base)
                .enumerate()
                .map(|(slot, (&n, &base))| BatchItem::new(slot as u32, base, n, 13))
                .collect();
            let mut execution = ScriptedBatch {
                history: vec![vec![99]; frames.len()],
                base,
                max_rows,
                steps: 0,
            };
            decode_batch(&mut execution, &mut items, 13, max_symbols, window).unwrap();
            for (item, expected) in items.iter().zip(&alone) {
                assert_eq!(
                    item.tokens(),
                    &expected[..],
                    "window {window} max_rows {max_rows}"
                );
            }
            assert!(
                alone.iter().map(Vec::len).sum::<usize>() > 10,
                "the script emits"
            );
        }
        let mut execution = ScriptedBatch {
            history: vec![Vec::new(); 3],
            base: vec![0; 3],
            max_rows: 2,
            steps: 0,
        };
        let mut items: Vec<_> = (0..3).map(|slot| BatchItem::new(slot, 0, 4, 13)).collect();
        assert!(decode_batch(&mut execution, &mut items, 13, 10, 4).is_err());
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
            detokenize(
                &TokenOutput::sentencepiece(vocabulary, &[".", "?", "!"]),
                &[0, 1, 2, 3]
            ),
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

    #[test]
    fn encoder_holds_a_stream_final_in_whole_steps() {
        let stages = [[3, 2, 1, 1]; 3];
        let capacities = [200, 400, 3000];
        // 185 frames: 24 rows + 6 trailing, 32 in whole steps: more than 200 frames' 25 rows.
        assert_eq!(
            encoder_index(&capacities, 185, &stages, 6, Some(4)).unwrap(),
            Some(1)
        );
        assert_eq!(
            encoder_index(&capacities, 185, &stages, 6, None).unwrap(),
            Some(0)
        );
        // 100 frames: 13 + 6 rows, 20 in whole steps.
        assert_eq!(
            encoder_index(&capacities, 100, &stages, 6, Some(4)).unwrap(),
            Some(0)
        );
        // 2990 frames: 374 + 6 rows outgrow 3000 frames' 375; it still holds the frames.
        assert_eq!(
            encoder_index(&capacities, 2990, &stages, 6, Some(4)).unwrap(),
            Some(2)
        );
        assert_eq!(
            encoder_index(&capacities, 3001, &stages, 6, Some(4)).unwrap(),
            None
        );
    }
}
