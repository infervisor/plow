//! Host executor for `vad.frame.v1` packets (`plow_asset::speech_contract`): the packet's
//! state-bank programs interpreted in plain Rust over a per-stream FP32 arena, with weights shared
//! read-only. The segment policy (thresholds, hysteresis, durations) is packet data. A [`Vad`] is
//! `Sync`; each [`VadStream`] is independent, so streams run on any number of threads.

use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use plow_asset::speech_contract::{self as contract, VadContract, VadPolicy};

use crate::asset::devblob::DevBlob;
use crate::exec::packet_runtime::PacketAsset;
use crate::{Result, RuntimeError};

pub const DRIVER: &str = contract::VAD_DRIVER;

#[derive(Clone, Copy)]
enum Loc {
    Weight(usize, usize),
    Arena(usize, usize),
}

pub struct Vad {
    /// One program per state bank, run in rotation.
    programs: Vec<Vec<DevInst64>>,
    locs: Vec<Loc>,
    weights: Vec<f32>,
    /// Arena image at stream open: zeros, with the initialized runtime tensors (states) filled.
    arena: Vec<f32>,
    input: usize,
    probability: usize,
    scratch: usize,
    fma: bool,
    pub sample_rate: u32,
    pub frame: usize,
    pub context: usize,
    pub contract: VadContract,
}

pub struct VadStream {
    arena: Vec<f32>,
    scratch: Vec<f32>,
    bank: usize,
    tail: Vec<f32>,
}

fn reject(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(message.into())
}

/// The `--asr-vad-packet` VAD, loaded once; `Err` when it is configured but does not load.
pub fn configured() -> Result<Option<Arc<Vad>>> {
    static VAD: OnceLock<std::result::Result<Option<Arc<Vad>>, String>> = OnceLock::new();
    VAD.get_or_init(|| match &crate::config::RuntimeConfig::get().asr_vad_packet {
        None => Ok(None),
        Some(path) => Vad::load(path).map(|vad| Some(Arc::new(vad))).map_err(|e| format!("{}: {e}", path.display())),
    })
    .clone()
    .map_err(RuntimeError::Rejected)
}

/// Total detected speech in `segments`, in samples.
pub fn speech_samples(segments: &[Range<usize>]) -> usize {
    segments.iter().map(ExactSizeIterator::len).sum()
}

impl Vad {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).map_err(|source| RuntimeError::Io { path: path.to_owned(), source })?;
        let blob = DevBlob::parse(&raw)?;
        let asset = PacketAsset::load(path)?;
        if asset.pipelines().iter().any(|p| p.driver == contract::VAD_DRIVER_V0) {
            return Err(reject(format!(
                "{} is a contract-0 VAD packet; re-emit it (scripts/asr/silero_vad_build.sh) or upgrade it (asr_packet_upgrade vad)",
                contract::VAD_DRIVER_V0
            )));
        }
        let pipeline = asset
            .pipelines()
            .iter()
            .find(|p| p.driver == DRIVER)
            .ok_or_else(|| reject(format!("packet pipeline driver {DRIVER:?} is missing")))?;
        let parameter = |name: &str| {
            pipeline.parameters.get(name).copied().ok_or_else(|| reject(format!("VAD parameter {name} is missing")))
        };
        let version = parameter(contract::CONTRACT)?;
        if version > contract::VAD_CONTRACT {
            return Err(reject(format!("VAD packet contract {version}; this plowrt implements {}", contract::VAD_CONTRACT)));
        }
        if parameter("executor")? != contract::EXECUTOR_HOST {
            return Err(reject("VAD packet does not declare the host executor; this plowrt runs VAD on the host only"));
        }
        let banks = parameter("state_banks")?;
        if !(1..=16).contains(&banks) {
            return Err(reject("VAD state bank count is invalid"));
        }
        let programs = (0..banks)
            .map(|bank| {
                let role = format!("step.{bank}");
                let index = *pipeline.programs.get(&role).ok_or_else(|| reject(format!("VAD program {role} is missing")))?;
                Ok(blob.progs.get(index as usize).ok_or_else(|| reject("VAD program index is invalid"))?.insts.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let (sample_rate, frame, context) =
            (parameter("sample_rate")? as u32, parameter("frame_samples")? as usize, parameter("context_samples")? as usize);
        // Serving decodes and resamples every input to the canonical rate before the VAD sees it.
        if sample_rate != crate::asr::frontend::SAMPLE_RATE || frame == 0 || context > frame {
            return Err(reject(format!(
                "VAD packet frames {frame} + {context} samples at {sample_rate} Hz; plowrt feeds {} Hz frames",
                crate::asr::frontend::SAMPLE_RATE
            )));
        }
        let contract = VadContract::from_parameters(|name| pipeline.parameters.get(name).copied()).map_err(reject)?;

        let written: std::collections::BTreeSet<u16> =
            programs.iter().flatten().flat_map(|inst| outputs(inst).iter().map(move |&slot| inst.t[slot])).collect();
        let (mut locs, mut weights, mut arena) = (Vec::with_capacity(blob.tensors.len()), Vec::new(), Vec::new());
        for (index, tensor) in blob.tensors.iter().enumerate() {
            if tensor.bytes % 4 != 0 {
                return Err(reject(format!("VAD tensor {} is not FP32", tensor.name)));
            }
            let elements = tensor.bytes as usize / 4;
            let init = tensor.init.clone().map(|range| &blob.init[range]);
            let (store, loc): (&mut Vec<f32>, fn(usize, usize) -> Loc) = match init {
                Some(_) if !written.contains(&(index as u16)) => (&mut weights, Loc::Weight),
                _ => (&mut arena, Loc::Arena),
            };
            locs.push(loc(store.len(), elements));
            match init {
                Some(bytes) => store.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))),
                None => store.resize(store.len() + elements, 0.0),
            }
        }
        let role = |name: &str, elements: usize| -> Result<usize> {
            let tensor = pipeline.tensors.get(name).ok_or_else(|| reject(format!("VAD tensor {name} is missing")))?;
            let index = blob.tensors.iter().position(|t| t.name == tensor.name).ok_or_else(|| reject("VAD tensor is undeclared"))?;
            match locs[index] {
                Loc::Arena(offset, len) if len == elements => Ok(offset),
                _ => Err(reject(format!("VAD tensor {name} has the wrong geometry"))),
            }
        };
        let input = role("input", context + frame)?;
        let probability = role("probability", 1)?;
        let mut scratch = 0;
        for inst in programs.iter().flatten() {
            scratch = scratch.max(validate(inst, &locs)?);
        }
        #[cfg(target_arch = "x86_64")]
        let fma = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let fma = false;
        Ok(Self { programs, locs, weights, arena, input, probability, scratch, fma, sample_rate, frame, context, contract })
    }

    pub fn open(&self) -> VadStream {
        VadStream { arena: self.arena.clone(), scratch: vec![0.0; self.scratch], bank: 0, tail: vec![0.0; self.context] }
    }

    /// The speech probability of the next `frame` samples of the stream.
    pub fn step(&self, stream: &mut VadStream, frame: &[f32]) -> f32 {
        assert_eq!(frame.len(), self.frame, "a VAD step takes exactly one frame");
        let window = &mut stream.arena[self.input..self.input + self.context + self.frame];
        window[..self.context].copy_from_slice(&stream.tail);
        window[self.context..].copy_from_slice(frame);
        stream.tail.copy_from_slice(&frame[self.frame - self.context..]);
        #[cfg(target_arch = "x86_64")]
        if self.fma {
            // SAFETY: the CPU supports the enabled features (checked at load).
            unsafe { self.run_fma(stream) };
        } else {
            self.run::<false>(stream);
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.run::<false>(stream);
        stream.bank = (stream.bank + 1) % self.programs.len();
        stream.arena[self.probability]
    }

    /// Per-frame probabilities of a whole signal; a short last frame is zero padded.
    pub fn probabilities(&self, audio: &[f32]) -> Vec<f32> {
        let mut stream = self.open();
        let mut padded = vec![0.0; self.frame];
        audio
            .chunks(self.frame)
            .map(|chunk| {
                let frame = if chunk.len() == self.frame {
                    chunk
                } else {
                    padded[..chunk.len()].copy_from_slice(chunk);
                    &padded[..]
                };
                self.step(&mut stream, frame)
            })
            .collect()
    }

    fn weight(&self, inst: &DevInst64, slot: usize) -> Option<&[f32]> {
        match self.locs.get(inst.t[slot] as usize)? {
            Loc::Weight(offset, len) => Some(&self.weights[*offset..offset + len]),
            Loc::Arena(..) => None,
        }
    }

    fn arena(&self, inst: &DevInst64, slot: usize) -> Range<usize> {
        match self.locs[inst.t[slot] as usize] {
            Loc::Arena(offset, len) => offset..offset + len,
            Loc::Weight(..) => unreachable!("validated: operand {slot} is a runtime tensor"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_fma(&self, stream: &mut VadStream) {
        self.run::<true>(stream)
    }

    #[inline(always)]
    fn run<const FMA: bool>(&self, stream: &mut VadStream) {
        for inst in &self.programs[stream.bank] {
            self.execute::<FMA>(inst, &mut stream.arena, &mut stream.scratch);
        }
    }

    #[inline(always)]
    fn execute<const FMA: bool>(&self, inst: &DevInst64, arena: &mut [f32], scratch: &mut [f32]) {
        let i = inst.i;
        match DevOp::from_u16(inst.op) {
            Some(DevOp::Conv1dF32) => {
                let (out, x) = disjoint(arena, self.arena(inst, 0), self.arena(inst, 1));
                conv1d::<FMA>(out, x, self.weight(inst, 2).unwrap(), self.weight(inst, 3), &i, inst.fj, scratch);
            }
            Some(DevOp::DenseGemmF32) => {
                let (out, a) = disjoint(arena, self.arena(inst, 0), self.arena(inst, 1));
                let (w, bias) = (self.weight(inst, 2).unwrap(), self.weight(inst, 3));
                let (m, n, k) = (i[0] as usize, i[1] as usize, i[2] as usize);
                for row in 0..m {
                    for col in 0..n {
                        let v = dot::<FMA>(&a[row * k..][..k], &w[col * k..][..k]) + bias.map_or(0.0, |b| b[col]);
                        out[row * n + col] = if i[3] == 1 { v.max(0.0) } else { v };
                    }
                }
            }
            Some(DevOp::CopyColsF32) => {
                let (out, x) = disjoint(arena, self.arena(inst, 0), self.arena(inst, 1));
                let (items, rows, cols) = (i[0] as usize, i[1] as usize, i[2] as usize);
                let (in_items, out_items) = (inst.fj[1] as usize, inst.fj[2] as usize);
                for item in 0..items {
                    for row in 0..rows {
                        let src = item * in_items + row * i[3] as usize + i[4] as usize;
                        let dst = item * out_items + row * i[5] as usize + i[6] as usize;
                        out[dst..dst + cols].copy_from_slice(&x[src..src + cols]);
                    }
                }
            }
            Some(DevOp::BinaryF32) => {
                let (o, a, b) = (self.arena(inst, 0).start, self.arena(inst, 1).start, self.arena(inst, 2).start);
                let (items, rows, width) = (i[0] as usize, i[1] as usize, i[2] as usize);
                let scale = if i[7] & 1 != 0 { f32::from_bits(inst.fj[0]) } else { 1.0 };
                for item in 0..items {
                    for row in 0..rows {
                        for col in 0..width {
                            let at = (item * rows + row) * width + col;
                            let y = arena[b + item * i[4] as usize + row * i[5] as usize + col * i[6] as usize];
                            let x = arena[a + at];
                            arena[o + at] = scale
                                * match i[3] {
                                    0 => x + y,
                                    1 => x - y,
                                    2 => x * y,
                                    3 => x / y,
                                    4 => x.max(y),
                                    _ => x.min(y),
                                };
                        }
                    }
                }
            }
            Some(DevOp::UnaryF32) => {
                let (o, x) = (self.arena(inst, 0).start, self.arena(inst, 1).start);
                let (rows, width, stride, col0) = (i[0] as usize, i[1] as usize, i[3] as usize, i[4] as usize);
                let stride = if stride == 0 { width } else { stride };
                for row in 0..rows {
                    for col in col0..col0 + width {
                        let at = row * stride + col;
                        arena[o + at] = activation(i[2], arena[x + at]);
                    }
                }
            }
            Some(DevOp::ScaledAddF32) => {
                let (o, a, b) = (self.arena(inst, 0).start, self.arena(inst, 1).start, self.arena(inst, 2).start);
                let scale = f32::from_bits(inst.fj[0]);
                for at in 0..i[0] as usize {
                    arena[o + at] = arena[a + at] + scale * arena[b + at];
                }
            }
            Some(DevOp::LstmCellF32) => {
                let (h, c, gates, previous) =
                    (self.arena(inst, 0).start, self.arena(inst, 1).start, self.arena(inst, 2).start, self.arena(inst, 3).start);
                let width = i[0] as usize;
                let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                for j in 0..width {
                    let gate = |g: usize| arena[gates + g * width + j];
                    let cell = sigmoid(gate(1)) * arena[previous + j] + sigmoid(gate(0)) * gate(2).tanh();
                    let hidden = sigmoid(gate(3)) * cell.tanh();
                    arena[c + j] = cell;
                    arena[h + j] = hidden;
                }
            }
            _ => unreachable!("validated op"),
        }
    }
}

/// Output operand slots of each supported op.
fn outputs(inst: &DevInst64) -> &'static [usize] {
    match DevOp::from_u16(inst.op) {
        Some(DevOp::LstmCellF32) => &[0, 1],
        _ => &[0],
    }
}

/// Check one instruction is a form this executor runs; returns the scratch it needs.
fn validate(inst: &DevInst64, locs: &[Loc]) -> Result<usize> {
    let op = DevOp::from_u16(inst.op).ok_or_else(|| reject(format!("unknown VAD op {}", inst.op)))?;
    let i = inst.i;
    let loc = |slot: usize| locs.get(inst.t[slot] as usize).copied();
    let arena = |slot: usize| matches!(loc(slot), Some(Loc::Arena(..)));
    let weight = |slot: usize| matches!(loc(slot), Some(Loc::Weight(..)));
    let len = |slot: usize| match loc(slot) {
        Some(Loc::Arena(_, len) | Loc::Weight(_, len)) => len,
        None => 0,
    };
    let none = |slots: Range<usize>| slots.into_iter().all(|slot| inst.t[slot] == TENSOR_NONE16);
    let distinct = |a: usize, b: usize| inst.t[a] != inst.t[b];
    let (ok, scratch) = match op {
        DevOp::Conv1dF32 => {
            let (batch, rows, cin, cout, kernel, stride, dilation, groups) =
                (i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
            let (before, after) = (inst.fj[1] & 0xffff, inst.fj[1] >> 16);
            let (flags, span) = (inst.fj[2], dilation * kernel.saturating_sub(1) + 1);
            let padded = rows + before + after;
            let out_rows = if padded >= span && stride > 0 { (padded - span) / stride + 1 } else { 0 };
            let ok = batch == 1
                && kernel > 0
                && dilation > 0
                && groups > 0
                && cin % groups == 0
                && cout % groups == 0
                && flags & !0xf03 == 0
                && flags & 3 <= 1
                && matches!((flags >> 8) & 15, 0 | 15)
                && out_rows > 0
                && (flags & 3 == 0 || (before < rows && after < rows))
                && arena(0)
                && arena(1)
                && distinct(0, 1)
                && weight(2)
                && (inst.t[3] == TENSOR_NONE16 || (weight(3) && len(3) == cout as usize))
                && none(4..8)
                && len(0) >= (out_rows * cout) as usize
                && len(1) >= (rows * cin) as usize
                && len(2) == (cout * cin / groups * kernel) as usize;
            (ok, (cin / groups.max(1) * kernel) as usize)
        }
        DevOp::DenseGemmF32 => {
            let (m, n, k) = (i[0] as usize, i[1] as usize, i[2] as usize);
            let ok = i[3] <= 1
                && i[4..].iter().all(|&v| v == 0)
                && arena(0)
                && arena(1)
                && distinct(0, 1)
                && weight(2)
                && (inst.t[3] == TENSOR_NONE16 || (weight(3) && len(3) == n))
                && none(4..8)
                && len(0) >= m * n
                && len(1) >= m * k
                && len(2) == n * k;
            (ok, 0)
        }
        DevOp::CopyColsF32 => {
            let last = |items: u32, item_stride: u32, rows: u32, stride: u32, offset: u32| {
                (items.saturating_sub(1) * item_stride + rows.saturating_sub(1) * stride + offset + i[2]) as usize
            };
            let ok = i[7] == 0
                && arena(0)
                && arena(1)
                && distinct(0, 1)
                && none(2..8)
                && len(1) >= last(i[0], inst.fj[1], i[1], i[3], i[4])
                && len(0) >= last(i[0], inst.fj[2], i[1], i[5], i[6]);
            (ok, 0)
        }
        DevOp::BinaryF32 => {
            let n = (i[0] * i[1] * i[2]) as usize;
            let b_last = (i[0].saturating_sub(1) * i[4] + i[1].saturating_sub(1) * i[5] + i[2].saturating_sub(1) * i[6]) as usize;
            let ok = i[3] <= 5
                && i[7] & !1 == 0
                && (0..3).all(arena)
                && none(3..8)
                && len(0) >= n
                && len(1) >= n
                && len(2) > b_last
                && distinct(0, 2);
            (ok, 0)
        }
        DevOp::UnaryF32 => {
            let stride = if i[3] == 0 { i[1] } else { i[3] };
            let need = (i[0].saturating_sub(1) * stride + i[4] + i[1]) as usize;
            let ok = matches!(i[2], 0 | 1 | 4 | 5 | 6 | 15 | 16)
                && arena(0)
                && arena(1)
                && none(2..8)
                && len(0) >= need
                && len(1) >= need;
            (ok, 0)
        }
        DevOp::ScaledAddF32 => {
            let n = i[0] as usize;
            let ok = i[1] == 0 && (0..3).all(arena) && none(3..8) && (0..3).all(|slot| len(slot) >= n);
            (ok, 0)
        }
        DevOp::LstmCellF32 => {
            let w = i[0] as usize;
            let ok = (0..4).all(arena)
                && none(4..8)
                && [0, 1, 3].iter().all(|&slot| len(slot) >= w)
                && len(2) >= 4 * w
                && distinct(0, 1);
            (ok, 0)
        }
        _ => (false, 0),
    };
    if !ok {
        return Err(reject(format!("VAD executor does not support this {op:?} form")));
    }
    Ok(scratch)
}

fn activation(kind: u32, x: f32) -> f32 {
    match kind {
        1 => x.tanh(),
        4 => x.exp(),
        5 => x.abs(),
        6 => 1.0 / (1.0 + (-x).exp()),
        15 => x.max(0.0),
        16 => x.sqrt(),
        _ => x,
    }
}

/// Mutable `out` and shared `input` views of the arena; validated non-overlapping.
fn disjoint(arena: &mut [f32], out: Range<usize>, input: Range<usize>) -> (&mut [f32], &[f32]) {
    if out.end <= input.start {
        let (low, high) = arena.split_at_mut(input.start);
        (&mut low[out], &high[..input.len()])
    } else {
        assert!(input.end <= out.start, "VAD operands overlap");
        let (low, high) = arena.split_at_mut(out.start);
        (&mut high[..out.len()], &low[input])
    }
}

/// `Conv1dF32` for one item: gather each output row's receptive field into `scratch` in the
/// weight's `[in/groups][kernel]` order, then one dot product per output channel.
#[inline(always)]
fn conv1d<const FMA: bool>(out: &mut [f32], x: &[f32], weight: &[f32], bias: Option<&[f32]>, i: &[u32; 8], fj: [u32; 3], scratch: &mut [f32]) {
    let (rows, cin, cout, kernel, stride, dilation, groups) =
        (i[1] as i64, i[2] as usize, i[3] as usize, i[4] as usize, i[5] as i64, i[6] as i64, i[7] as usize);
    let (before, after) = ((fj[1] & 0xffff) as i64, (fj[1] >> 16) as i64);
    let (reflect, relu) = (fj[2] & 3 == 1, (fj[2] >> 8) & 15 == 15);
    let span = dilation * (kernel as i64 - 1) + 1;
    let out_rows = ((rows + before + after - span) / stride + 1) as usize;
    let (cg, ng) = (cin / groups, cout / groups);
    let field = &mut scratch[..cg * kernel];
    for t in 0..out_rows {
        for g in 0..groups {
            for k in 0..kernel {
                let mut u = t as i64 * stride + k as i64 * dilation - before;
                if reflect && !(0..rows).contains(&u) {
                    u = if u < 0 { -u } else { 2 * (rows - 1) - u };
                }
                for c in 0..cg {
                    field[c * kernel + k] = if (0..rows).contains(&u) { x[u as usize * cin + g * cg + c] } else { 0.0 };
                }
            }
            for o in g * ng..(g + 1) * ng {
                let v = dot::<FMA>(field, &weight[o * cg * kernel..][..cg * kernel]) + bias.map_or(0.0, |b| b[o]);
                out[t * cout + o] = if relu { v.max(0.0) } else { v };
            }
        }
    }
}

/// Independent accumulators so the loop vectorizes; `FMA` only inside an `fma` target feature
/// (elsewhere `mul_add` is a libm call).
#[inline(always)]
fn dot<const FMA: bool>(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 16];
    let (ac, bc) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = ac.remainder().iter().zip(bc.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ac.zip(bc) {
        for lane in 0..16 {
            lanes[lane] = if FMA { x[lane].mul_add(y[lane], lanes[lane]) } else { lanes[lane] + x[lane] * y[lane] };
        }
    }
    lanes.iter().sum::<f32>() + tail
}

/// Segment policy of one request: the packet's [`VadPolicy`], optionally overridden.
#[derive(Clone, Copy, Debug)]
pub struct SegmentOptions {
    pub threshold: f32,
    pub release_offset: f32,
    pub release_floor: f32,
    pub min_speech_ms: u32,
    pub max_speech_s: f32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    pub min_silence_at_max_speech_ms: u32,
}

impl SegmentOptions {
    pub fn from_policy(p: &VadPolicy) -> Self {
        Self {
            threshold: p.threshold,
            release_offset: p.release_offset,
            release_floor: p.release_floor,
            min_speech_ms: p.min_speech_ms,
            max_speech_s: if p.max_speech_ms == 0 { f32::INFINITY } else { p.max_speech_ms as f32 / 1000.0 },
            min_silence_ms: p.min_silence_ms,
            speech_pad_ms: p.speech_pad_ms,
            min_silence_at_max_speech_ms: p.min_silence_at_max_speech_ms,
        }
    }

    /// Below this a speaking stream turns silent (hysteresis).
    pub fn release_threshold(&self) -> f32 {
        (self.threshold - self.release_offset).max(self.release_floor)
    }
}

impl Vad {
    /// The packet's default segment policy (`/v1/audio/vad`, turn detection).
    pub fn segment_options(&self) -> SegmentOptions {
        SegmentOptions::from_policy(&self.contract.policy)
    }

    /// The no-speech upload gate's policy and the least speech (samples) an upload needs.
    pub fn gate(&self) -> (SegmentOptions, usize) {
        let g = &self.contract.gate;
        let options = SegmentOptions {
            min_speech_ms: g.min_speech_ms,
            min_silence_ms: g.min_silence_ms,
            speech_pad_ms: g.speech_pad_ms,
            ..self.segment_options()
        };
        (options, self.sample_rate as usize * g.min_total_speech_ms as usize / 1000)
    }

    /// Longest duration (ms) a request override may set.
    pub fn max_override_ms(&self) -> u32 {
        self.contract.bounds.max_duration_ms
    }
}

/// Speech `[start, end)` sample ranges from per-frame probabilities: a port of Silero's
/// `get_speech_timestamps_from_probs` (longest-silence cut at the maximum speech length).
pub fn speech_segments(probabilities: &[f32], frame: usize, sample_rate: u32, audio_samples: usize, o: SegmentOptions) -> Vec<Range<usize>> {
    let rate = sample_rate as f64;
    let ms = |v: u32| rate * f64::from(v) / 1000.0;
    let (min_speech, pad, min_silence, min_silence_at_max) =
        (ms(o.min_speech_ms), ms(o.speech_pad_ms), ms(o.min_silence_ms), ms(o.min_silence_at_max_speech_ms));
    let max_speech = rate * f64::from(o.max_speech_s) - frame as f64 - 2.0 * pad;
    let threshold = o.threshold;
    let neg_threshold = o.release_threshold();
    let audio = audio_samples as i64;

    let mut speeches: Vec<(i64, i64)> = Vec::new();
    let mut current: Option<i64> = None;
    let (mut temp_end, mut prev_end, mut next_start) = (0i64, 0i64, 0i64);
    let mut possible_ends: Vec<(i64, i64)> = Vec::new();
    for (index, &p) in probabilities.iter().enumerate() {
        let sample = (frame * index) as i64;
        if p >= threshold && temp_end != 0 {
            let silence = sample - temp_end;
            if silence as f64 > min_silence_at_max {
                possible_ends.push((temp_end, silence));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = sample;
            }
        }
        let Some(start) = current else {
            if p >= threshold {
                current = Some(sample);
            }
            continue;
        };
        if (sample - start) as f64 > max_speech {
            // Silero keeps `possible_ends` in order; the first of the longest silences wins.
            if let Some(&(end, silence)) = possible_ends.iter().rev().max_by_key(|e| e.1) {
                speeches.push((start, end));
                next_start = end + silence;
                current = (next_start < end + sample).then_some(next_start);
            } else {
                speeches.push((start, sample));
                current = None;
                (prev_end, next_start, temp_end) = (0, 0, 0);
                possible_ends.clear();
                continue;
            }
            (prev_end, next_start, temp_end) = (0, 0, 0);
            possible_ends.clear();
        }
        let Some(start) = current else { continue };
        if p < neg_threshold {
            if temp_end == 0 {
                temp_end = sample;
            }
            if ((sample - temp_end) as f64) < min_silence {
                continue;
            }
            if (temp_end - start) as f64 > min_speech {
                speeches.push((start, temp_end));
            }
            current = None;
            (prev_end, next_start, temp_end) = (0, 0, 0);
            possible_ends.clear();
        }
    }
    if let Some(start) = current {
        if (audio - start) as f64 > min_speech {
            speeches.push((start, audio));
        }
    }
    let pad_samples = pad as i64;
    for index in 0..speeches.len() {
        if index == 0 {
            speeches[0].0 = (speeches[0].0 as f64 - pad).max(0.0) as i64;
        }
        if index + 1 < speeches.len() {
            let silence = speeches[index + 1].0 - speeches[index].1;
            if (silence as f64) < 2.0 * pad {
                speeches[index].1 += silence / 2;
                speeches[index + 1].0 = (speeches[index + 1].0 - silence / 2).max(0);
            } else {
                speeches[index].1 = (speeches[index].1 + pad_samples).min(audio);
                speeches[index + 1].0 = (speeches[index + 1].0 - pad_samples).max(0);
            }
        } else {
            speeches[index].1 = (speeches[index].1 + pad_samples).min(audio);
        }
    }
    speeches.into_iter().map(|(s, e)| s as usize..e as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scripts/asr/silero_export.py` output compiled by `asr_silero_vad_compile`; skipped when absent.
    const MODEL: &str = "/home/ssm-user/models/silero-vad-v5";

    #[test]
    fn packet_matches_the_reference_probabilities() {
        let dir = Path::new(MODEL);
        let Ok(vad) = Vad::load(&dir.join("silero_vad.pkt")) else { return };
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
        let values = |key: &str| -> Vec<f32> {
            reference[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
        };
        let (signal, expected) = (values("signal"), values("probabilities"));
        let frames = signal.len() / vad.frame;
        let actual = vad.probabilities(&signal[..frames * vad.frame]);
        assert_eq!(actual.len(), expected.len());
        let worst = actual.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "max |p - p_torch| = {worst}");        assert!(expected.iter().any(|&p| p > 0.5) && expected.iter().any(|&p| p < 0.1));
    }

    #[test]
    fn segments_join_short_pauses_drop_short_speech_and_pad() {
        let p = |speech: &[Range<usize>]| -> Vec<f32> {
            (0..100).map(|f| if speech.iter().any(|r| r.contains(&f)) { 0.9 } else { 0.05 }).collect()
        };
        let segment = |speech: &[Range<usize>], options| speech_segments(&p(speech), 512, 16_000, 100 * 512, options);
        // A 64 ms pause (< 100 ms) joins; 128 ms of speech (< 250 ms) is dropped; 30 ms pads.
        assert_eq!(segment(&[10..40, 42..60, 80..84], SegmentOptions::from_policy(&devgen::vad::POLICY.policy)), vec![10 * 512 - 480..60 * 512 + 480]);
        // A pause shorter than both pads is split down the middle.
        let wide = SegmentOptions { speech_pad_ms: 100, ..SegmentOptions::from_policy(&devgen::vad::POLICY.policy) };
        let split = segment(&[10..40, 45..60], wide);
        assert_eq!(split, vec![10 * 512 - 1600..40 * 512 + 1280, 40 * 512 + 1280..60 * 512 + 1600]);
    }
}
