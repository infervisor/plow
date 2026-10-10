use std::io::Cursor;
use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::exec::packet_runtime::{BoundPacketPipeline, PacketRuntime};
use crate::{Result, RuntimeError};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 128;
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
const FFT: usize = 400;
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
const HOP: usize = 160;
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
const BINS: usize = FFT / 2 + 1;
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

fn invalid(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(message.into())
}

/// `f32::log10` links `log10f@GLIBC_2.43` when built on glibc >= 2.43, the only symbol that keeps
/// the release binary off glibc 2.34-2.42 hosts (Ubuntu 22.04/24.04, RHEL 9). The f64 route links
/// `log10@GLIBC_2.2.5`; rounded to f32 it is the correctly rounded value 2.43's `log10f` returns,
/// short of a double-rounding tie.
fn log10_f32(x: f32) -> f32 {
    (x as f64).log10() as f32
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("audio exceeds {0} seconds")]
    TooLong(usize),
}

/// Input sample rates accepted (WAV and the WebSocket stream); audio is resampled to 16 kHz.
pub const MIN_INPUT_RATE: u32 = 8_000;
pub const MAX_INPUT_RATE: u32 = 48_000;

pub fn decode_wav(bytes: &[u8]) -> std::result::Result<Vec<f32>, AudioError> {
    decode_wav_within(bytes, MAX_SAMPLES)
}

/// [`decode_wav`] of audio up to `max_samples` (16 kHz) long.
pub fn decode_wav_within(bytes: &[u8], max_samples: usize) -> std::result::Result<Vec<f32>, AudioError> {
    let (samples, rate) = read_wav(bytes, max_samples)?;
    let samples = if rate == SAMPLE_RATE {
        samples
    } else {
        let mut resampler = Resampler::new(rate).map_err(AudioError::Unsupported)?;
        let mut out = resampler.push(&samples);
        out.extend(resampler.finish());
        out
    };
    if samples.is_empty() {
        return Err(AudioError::Invalid("audio is empty".into()));
    }
    // Shorter clips (a recording's tail) are padded with silence to the models' minimum.
    let mut samples = samples;
    samples.resize(samples.len().max(SAMPLE_RATE as usize / 2), 0.0);
    Ok(samples)
}

/// [`decode_wav`] for a piece of a longer recording: any length up to 30 seconds, 16 kHz only
/// (resampling each piece on its own would put seams at the piece boundaries).
pub fn decode_wav_chunk(bytes: &[u8]) -> std::result::Result<Vec<f32>, AudioError> {
    let (samples, rate) = read_wav(bytes, MAX_SAMPLES)?;
    if rate != SAMPLE_RATE {
        return Err(AudioError::Unsupported(
            "appended recording pieces must be 16 kHz WAV; the WebSocket stream resamples".into(),
        ));
    }
    Ok(samples)
}

/// Mono samples (stereo averaged) and their rate, at most `max_samples` at 16 kHz.
fn read_wav(bytes: &[u8], max_samples: usize) -> std::result::Result<(Vec<f32>, u32), AudioError> {
    let invalid = |message| AudioError::Invalid(message);
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| invalid(format!("invalid WAV: {e}")))?;
    let spec = reader.spec();
    if !(MIN_INPUT_RATE..=MAX_INPUT_RATE).contains(&spec.sample_rate) || !(1..=2).contains(&spec.channels) {
        return Err(AudioError::Unsupported(
            "ASR requires 8-48 kHz mono/stereo WAV".into(),
        ));
    }
    if reader.duration() as u64 * u64::from(SAMPLE_RATE) > max_samples as u64 * u64::from(spec.sample_rate) {
        return Err(AudioError::TooLong(max_samples / SAMPLE_RATE as usize));
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float if spec.bits_per_sample == 32 => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| invalid(format!("invalid WAV samples: {e}")))?,
        hound::SampleFormat::Int if matches!(spec.bits_per_sample, 8 | 16 | 24 | 32) => {
            let scale = 2f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| invalid(format!("invalid WAV samples: {e}")))?
        }
        _ => {
            return Err(AudioError::Unsupported(
                "unsupported WAV sample format".into(),
            ))
        }
    };
    if samples.iter().any(|v| !v.is_finite()) || samples.len() % spec.channels as usize != 0 {
        return Err(invalid("invalid PCM samples".into()));
    }
    if spec.channels == 1 {
        return Ok((samples, spec.sample_rate));
    }
    Ok((samples
        .chunks_exact(2)
        .map(|s| s[0] * 0.5 + s[1] * 0.5)
        .collect(), spec.sample_rate))
}

/// Windowed-sinc resampling to 16 kHz, fed in pieces. Polyphase over the exact rational ratio
/// `L/M`: output `n` sits at input position `n M / L` and sums the `2K` inputs around it, so an
/// output depends only on its inputs and is the same whatever the piece boundaries. Inputs before
/// the start and after [`finish`](Self::finish) are zeros. 16 kHz input passes through untouched.
pub struct Resampler {
    up: u64,
    down: u64,
    half: usize,
    /// `[phase][tap]`, taps for input offsets `-half+1 ..= half`.
    taps: Vec<f32>,
    /// Inputs from absolute index `base` on.
    input: Vec<f32>,
    base: u64,
    received: u64,
    produced: u64,
}

impl Resampler {
    /// Zero crossings each side at the narrower of the two Nyquist bands.
    const ZEROS: f64 = 16.0;
    /// Cutoff as a share of that Nyquist band (the transition band below it).
    const ROLLOFF: f64 = 0.94;

    pub fn new(rate: u32) -> std::result::Result<Self, String> {
        if !(MIN_INPUT_RATE..=MAX_INPUT_RATE).contains(&rate) {
            return Err(format!("sample rate {rate} is outside {MIN_INPUT_RATE}..={MAX_INPUT_RATE} Hz"));
        }
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let g = gcd(u64::from(rate), u64::from(SAMPLE_RATE));
        let (up, down) = (u64::from(SAMPLE_RATE) / g, u64::from(rate) / g);
        if up == down {
            return Ok(Self { up, down, half: 0, taps: Vec::new(), input: Vec::new(), base: 0, received: 0, produced: 0 });
        }
        // Cutoff in cycles per input sample (Nyquist = 0.5), below both rates' Nyquist.
        let cutoff = 0.5 * (up as f64 / down as f64).min(1.0) * Self::ROLLOFF;
        let half = (Self::ZEROS / (2.0 * cutoff)).ceil() as usize;
        let mut taps = Vec::with_capacity(up as usize * 2 * half);
        for phase in 0..up {
            let frac = phase as f64 / up as f64;
            let row: Vec<f64> = (0..2 * half)
                .map(|i| {
                    let t = (i as f64 - half as f64 + 1.0) - frac;
                    let x = 2.0 * cutoff * t;
                    let sinc = if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                    let u = t / half as f64;
                    let window = if u.abs() >= 1.0 {
                        0.0
                    } else {
                        0.42 + 0.5 * (std::f64::consts::PI * u).cos() + 0.08 * (2.0 * std::f64::consts::PI * u).cos()
                    };
                    sinc * window
                })
                .collect();
            // Unit DC gain on every phase.
            let sum: f64 = row.iter().sum();
            taps.extend(row.iter().map(|v| (v / sum) as f32));
        }
        Ok(Self { up, down, half, taps, input: Vec::new(), base: 0, received: 0, produced: 0 })
    }

    /// Outputs `samples` completes: every one whose right context has arrived.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.half == 0 {
            return samples.to_vec();
        }
        self.input.extend_from_slice(samples);
        self.received += samples.len() as u64;
        self.drain(false)
    }

    /// The remaining outputs, the input end padded with zeros.
    pub fn finish(&mut self) -> Vec<f32> {
        if self.half == 0 {
            return Vec::new();
        }
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> Vec<f32> {
        // Outputs n with n M / L < received; without flush, also all `half` right taps present.
        let total = (self.received * self.up).div_ceil(self.down);
        let mut out = Vec::with_capacity(total.saturating_sub(self.produced) as usize);
        let width = 2 * self.half;
        // (center input index, tap phase) of output `n`.
        let at = |n: u64| {
            let pos = n * self.down;
            (pos / self.up, (pos % self.up) as usize)
        };
        while self.produced < total {
            // LANES outputs whose windows all lie inside the buffered input run together: each
            // keeps its own accumulator and adds its taps in order, so every sum is the one the
            // single-output loop forms, while the LANES dependency chains overlap.
            const LANES: usize = 8;
            let last = self.produced + LANES as u64 - 1;
            if last < total {
                let (c0, cl) = (at(self.produced).0, at(last).0);
                let ready = flush || cl + (self.half as u64) < self.received;
                if ready && c0 + 1 >= self.half as u64 + self.base && cl + 1 + self.half as u64 <= self.received {
                    let mut acc = [0f32; LANES];
                    let lanes: [(&[f32], &[f32]); LANES] = std::array::from_fn(|j| {
                        let (center, phase) = at(self.produced + j as u64);
                        let start = (center + 1 - self.half as u64 - self.base) as usize;
                        (&self.taps[phase * width..][..width], &self.input[start..][..width])
                    });
                    for i in 0..width {
                        for (a, (taps, x)) in acc.iter_mut().zip(&lanes) {
                            // SAFETY: both slices were cut to exactly `width` elements above.
                            *a += unsafe { taps.get_unchecked(i) * x.get_unchecked(i) };
                        }
                    }
                    out.extend_from_slice(&acc);
                    self.produced += LANES as u64;
                    continue;
                }
            }
            let (center, phase) = at(self.produced);
            if !flush && center + self.half as u64 >= self.received {
                break;
            }
            let taps = &self.taps[phase * width..(phase + 1) * width];
            let first = center as i64 - self.half as i64 + 1;
            let mut acc = 0f32;
            for (i, &w) in taps.iter().enumerate() {
                let index = first + i as i64;
                if index >= self.base as i64 && (index as u64) < self.received {
                    acc += w * self.input[(index as u64 - self.base) as usize];
                }
            }
            out.push(acc);
            self.produced += 1;
        }
        // Keep the inputs the next output can still reach.
        let next = (self.produced * self.down / self.up).saturating_sub(self.half as u64);
        if next > self.base {
            let drop = ((next - self.base) as usize).min(self.input.len());
            self.input.drain(..drop);
            self.base += drop as u64;
        }
        out
    }
}

pub struct MelFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

#[derive(Clone, Copy)]
pub struct LogMelConfig {
    pub sample_rate: usize,
    pub fft: usize,
    pub window: usize,
    pub hop: usize,
    pub bins: usize,
    pub preemphasis: f32,
    pub center_window: bool,
    pub periodic_hann: bool,
    pub normalize_per_feature: bool,
    pub mask_invalid_frames: bool,
    pub log_guard: f32,
    pub shaping: LogMelShaping,
}

/// Framing and log-compression variants beyond the defaults (zero-padded centred frames,
/// `samples / hop + 1` of them, `ln(energy + guard)`).
#[derive(Clone, Copy, Debug)]
pub struct LogMelShaping {
    pub pad_reflect: bool,
    pub drop_last_frame: bool,
    pub log10: bool,
    /// `log(max(energy, guard))` instead of `log(energy + guard)`.
    pub log_floor: bool,
    /// When positive, clamp every value to at least `max - dynamic_range` over the utterance.
    pub dynamic_range: f32,
    pub scale: f32,
    pub shift: f32,
}

impl Default for LogMelShaping {
    fn default() -> Self {
        Self { pad_reflect: false, drop_last_frame: false, log10: false, log_floor: false, dynamic_range: 0.0, scale: 1.0, shift: 0.0 }
    }
}

pub struct LogMelFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
    pub bins: usize,
}

pub struct LogMelFrontend {
    config: LogMelConfig,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filters: Vec<f32>,
    filter_ranges: Vec<std::ops::Range<usize>>,
}

pub struct PacketLogMelFrontend {
    inner: LogMelFrontend,
    sample_rate: usize,
    min_samples: usize,
    max_samples: usize,
}

impl PacketLogMelFrontend {
    pub fn bind(pipeline: &BoundPacketPipeline, runtime: &dyn PacketRuntime) -> Result<Self> {
        let tensor = pipeline.tensor("audio.frontend.filterbank")?;
        let mut bytes = vec![0; tensor.bytes];
        runtime.read_tensor(tensor, &mut bytes)?;
        Self::from_parameters(|name| pipeline.optional_parameter(name), &bytes)
    }

    /// From packet parameters and the raw FP32 filterbank bytes.
    pub fn from_parameters(parameter: impl Fn(&str) -> Option<u64>, filterbank: &[u8]) -> Result<Self> {
        let required = |name: &str| {
            parameter(name).ok_or_else(|| invalid(format!("packet parameter {name:?} is missing")))
        };
        if required("audio.frontend.kind")? != 1 {
            return Err(invalid("packet audio frontend kind is unsupported"));
        }
        let usize_param = |name| {
            usize::try_from(required(name)?)
                .map_err(|_| invalid(format!("packet parameter {name:?} overflows")))
        };
        let bool_param = |name| match required(name)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid(format!("packet parameter {name:?} is not boolean"))),
        };
        let f32_param = |name| {
            let bits = u32::try_from(required(name)?)
                .map_err(|_| invalid(format!("packet parameter {name:?} is not f32")))?;
            let value = f32::from_bits(bits);
            value
                .is_finite()
                .then_some(value)
                .ok_or_else(|| invalid(format!("packet parameter {name:?} is not finite")))
        };
        let config = LogMelConfig {
            sample_rate: usize_param("audio.sample_rate")?,
            fft: usize_param("audio.frontend.fft")?,
            window: usize_param("audio.frontend.window")?,
            hop: usize_param("audio.frontend.hop")?,
            bins: usize_param("audio.frontend.bins")?,
            preemphasis: f32_param("audio.frontend.preemphasis_f32")?,
            center_window: bool_param("audio.frontend.center_window")?,
            periodic_hann: bool_param("audio.frontend.periodic_hann")?,
            normalize_per_feature: bool_param("audio.frontend.normalize_per_feature")?,
            mask_invalid_frames: bool_param("audio.frontend.mask_invalid_frames")?,
            log_guard: f32_param("audio.frontend.log_guard_f32")?,
            shaping: {
                let flag = |name| parameter(name).unwrap_or(0) == 1;
                let float = |name, default: f32| parameter(name).map_or(Ok(default), |_| f32_param(name));
                LogMelShaping {
                    pad_reflect: flag("audio.frontend.pad_reflect"),
                    drop_last_frame: flag("audio.frontend.drop_last_frame"),
                    log10: flag("audio.frontend.log10"),
                    log_floor: flag("audio.frontend.log_floor"),
                    dynamic_range: float("audio.frontend.dynamic_range_f32", 0.0)?,
                    scale: float("audio.frontend.scale_f32", 1.0)?,
                    shift: float("audio.frontend.shift_f32", 0.0)?,
                }
            },
        };
        if !filterbank.len().is_multiple_of(std::mem::size_of::<f32>()) {
            return Err(invalid("packet log-mel filterbank is not FP32"));
        }
        let filters = filterbank
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let min_samples = usize_param("audio.min_samples")?;
        let max_samples = usize_param("audio.max_samples")?;
        if min_samples == 0 || min_samples > max_samples {
            return Err(invalid("packet audio sample limits are invalid"));
        }
        Ok(Self {
            inner: LogMelFrontend::new(config, filters)?,
            sample_rate: config.sample_rate,
            min_samples,
            max_samples,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Microseconds of audio per feature frame (the packet's hop).
    pub fn frame_us(&self) -> u32 {
        (self.inner.config.hop as u64 * 1_000_000 / self.sample_rate as u64) as u32
    }

    /// Frames past a frame its analysis window reaches (centred, `fft / 2` samples), plus two of
    /// slack: earlier frames no longer change as audio arrives.
    pub fn reach_frames(&self) -> usize {
        (self.inner.config.fft / 2).div_ceil(self.inner.config.hop) + 2
    }

    pub fn extract(&self, samples: &[f32]) -> Result<LogMelFeatures> {
        if samples.len() < self.min_samples || samples.len() > self.max_samples {
            return Err(invalid(format!(
                "audio has {} samples; packet accepts {}..={}",
                samples.len(),
                self.min_samples,
                self.max_samples
            )));
        }
        self.inner.extract(samples, true)
    }

    /// Mel frames `from_frame..` of a growing recording and how many leading frames (counted from
    /// 0) are final: their analysis window has arrived, so later audio cannot change them (no
    /// per-utterance normalization). Frames are computed from a slice starting `MARGIN` frames
    /// earlier, whose edge (padding, pre-emphasis) only the dropped frames see.
    pub fn extract_partial(&self, samples: &[f32], from_frame: usize) -> Result<(LogMelFeatures, usize)> {
        const MARGIN: usize = 2;
        let config = self.inner.config;
        let window_offset = if config.center_window { (config.fft - config.window) / 2 } else { 0 };
        let reach = window_offset + config.window;
        let pad = config.fft / 2;
        if samples.len() > self.max_samples
            || config.normalize_per_feature
            || pad > MARGIN * config.hop + window_offset
        {
            return Err(invalid("audio cannot be streamed through this frontend"));
        }
        let complete = (samples.len() + pad).checked_sub(reach).map_or(0, |span| span / config.hop + 1);
        let skip = from_frame.saturating_sub(MARGIN);
        let tail = samples.get(skip * config.hop..).filter(|t| !t.is_empty()).ok_or_else(|| invalid("no audio past the requested frame"))?;
        let mut features = self.inner.extract(tail, true)?;
        let drop = from_frame - skip;
        features.values.drain(..(drop * features.bins).min(features.values.len()));
        features.frames = features.frames.saturating_sub(drop);
        Ok((features, complete))
    }
}

impl LogMelFrontend {
    pub fn new(config: LogMelConfig, filters: Vec<f32>) -> Result<Self> {
        if config.sample_rate == 0
            || config.fft < 2
            || config.window < 2
            || config.window > config.fft
            || config.hop == 0
            || config.bins == 0
            || !config.preemphasis.is_finite()
            || !config.log_guard.is_finite()
            || config.log_guard <= 0.0
        {
            return Err(invalid("invalid log-mel configuration"));
        }
        let spectrum_bins = config.fft / 2 + 1;
        if filters.len() != config.bins * spectrum_bins
            || filters
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid("invalid log-mel filterbank"));
        }
        let denominator = if config.periodic_hann {
            config.window
        } else {
            config.window - 1
        } as f64;
        let window = (0..config.window)
            .map(|index| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * index as f64 / denominator).cos()))
                    as f32
            })
            .collect();
        let filter_ranges = filters
            .chunks_exact(spectrum_bins)
            .map(|filter| {
                let start = filter.iter().position(|&value| value != 0.0).unwrap_or(0);
                let end = filter
                    .iter()
                    .rposition(|&value| value != 0.0)
                    .map_or(0, |index| index + 1);
                start..end
            })
            .collect();
        Ok(Self {
            config,
            fft: FftPlanner::new().plan_fft_forward(config.fft),
            window,
            filters,
            filter_ranges,
        })
    }

    pub fn extract(&self, samples: &[f32], normalize: bool) -> Result<LogMelFeatures> {
        if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
            return Err(invalid("audio must contain finite PCM"));
        }
        let config = self.config;
        let shaping = config.shaping;
        let frames = samples.len() / config.hop + usize::from(!shaping.drop_last_frame);
        let valid_frames = (samples.len() / config.hop).min(frames);
        let spectrum_bins = config.fft / 2 + 1;
        let mut preemphasized = samples.to_vec();
        if config.preemphasis != 0.0 {
            for index in (1..preemphasized.len()).rev() {
                preemphasized[index] -= config.preemphasis * preemphasized[index - 1];
            }
        }
        let mut values = vec![0.0; frames * config.bins];
        let window_offset = if config.center_window {
            (config.fft - config.window) / 2
        } else {
            0
        };
        let center_pad = config.fft / 2;
        let buffers = || {
            (
                vec![Complex32::default(); config.fft],
                vec![Complex32::default(); self.fft.get_inplace_scratch_len()],
                vec![0.0f32; spectrum_bins],
            )
        };
        let frame_values = |(spectrum, scratch, power): &mut (Vec<Complex32>, Vec<Complex32>, Vec<f32>),
                            frame: usize,
                            out: &mut [f32]| {
            spectrum.fill(Complex32::default());
            for index in 0..config.window {
                let padded_index = frame * config.hop + window_offset + index;
                let sample = if shaping.pad_reflect {
                    let n = preemphasized.len() as isize;
                    let mut i = padded_index as isize - center_pad as isize;
                    if i < 0 {
                        i = -i;
                    }
                    if i >= n {
                        i = 2 * n - 2 - i;
                    }
                    preemphasized.get(i as usize).copied().unwrap_or(0.0)
                } else {
                    padded_index
                        .checked_sub(center_pad)
                        .and_then(|index| preemphasized.get(index))
                        .copied()
                        .unwrap_or(0.0)
                };
                spectrum[window_offset + index].re = sample * self.window[index];
            }
            self.fft.process_with_scratch(spectrum, scratch);
            for (power, value) in power.iter_mut().zip(spectrum.iter()) {
                *power = value.norm_sqr();
            }
            for (bin, out) in out.iter_mut().enumerate() {
                let range = self.filter_ranges[bin].clone();
                let energy = self.filters
                    [bin * spectrum_bins + range.start..bin * spectrum_bins + range.end]
                    .iter()
                    .zip(&power[range])
                    .map(|(filter, power)| filter * power)
                    .sum::<f32>();
                let energy = if shaping.log_floor { energy.max(config.log_guard) } else { energy + config.log_guard };
                *out = if shaping.log10 { log10_f32(energy) } else { energy.ln() };
            }
        };
        // Frames are independent: a request's frontend spreads over the rayon pool (c1 latency).
        #[cfg(feature = "hf-tokenizer")]
        {
            use rayon::prelude::*;
            const FRAMES_PER_TASK: usize = 64;
            values.par_chunks_mut(config.bins * FRAMES_PER_TASK).enumerate().for_each_init(buffers, |b, (task, out)| {
                for (i, out) in out.chunks_exact_mut(config.bins).enumerate() {
                    frame_values(b, task * FRAMES_PER_TASK + i, out);
                }
            });
        }
        #[cfg(not(feature = "hf-tokenizer"))]
        {
            let mut b = buffers();
            for (frame, out) in values.chunks_exact_mut(config.bins).enumerate() {
                frame_values(&mut b, frame, out);
            }
        }
        if shaping.dynamic_range > 0.0 {
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for value in &mut values {
                *value = value.max(maximum - shaping.dynamic_range);
            }
        }
        if shaping.scale != 1.0 || shaping.shift != 0.0 {
            for value in &mut values {
                *value = (*value + shaping.shift) * shaping.scale;
            }
        }
        if config.mask_invalid_frames {
            values[valid_frames * config.bins..].fill(0.0);
        }
        if normalize && config.normalize_per_feature {
            for bin in 0..config.bins {
                let mean = if valid_frames == 0 {
                    0.0
                } else {
                    (0..valid_frames)
                        .map(|frame| f64::from(values[frame * config.bins + bin]))
                        .sum::<f64>()
                        / valid_frames as f64
                };
                let variance = if valid_frames > 1 {
                    (0..valid_frames)
                        .map(|frame| (f64::from(values[frame * config.bins + bin]) - mean).powi(2))
                        .sum::<f64>()
                        / (valid_frames - 1) as f64
                } else {
                    0.0
                };
                let inverse_stddev = 1.0 / (variance.sqrt() + 1e-5);
                for frame in 0..valid_frames {
                    let index = frame * config.bins + bin;
                    values[index] = ((f64::from(values[index]) - mean) * inverse_stddev) as f32;
                }
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("PCM amplitude exceeds log-mel range"));
        }
        Ok(LogMelFeatures {
            values,
            frames,
            bins: config.bins,
        })
    }
}

/// Qwen3-ASR's Whisper log-mel, hand-written: the Metal reference path and the oracle the
/// packet-driven [`LogMelFrontend`] is tested against. Serving uses the packet's frontend.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub struct QwenFrontend {
    fft: Arc<dyn Fft<f32>>,
    window: [f32; FFT],
    filters: Vec<f32>,
    filter_ranges: [std::ops::Range<usize>; MEL_BINS],
}

#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
impl Default for QwenFrontend {
    fn default() -> Self {
        let hz_to_mel = |hz: f64| {
            if hz < 1000.0 {
                hz * 3.0 / 200.0
            } else {
                15.0 + (hz / 1000.0).ln() * 27.0 / 6.4f64.ln()
            }
        };
        let mel_to_hz = |mel: f64| {
            if mel < 15.0 {
                mel * 200.0 / 3.0
            } else {
                1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp()
            }
        };
        let points: Vec<_> = (0..MEL_BINS + 2)
            .map(|i| mel_to_hz(hz_to_mel(8000.0) * i as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let mut filters = vec![0.0; MEL_BINS * BINS];
        for m in 0..MEL_BINS {
            for k in 0..BINS {
                let hz = k as f64 * SAMPLE_RATE as f64 / FFT as f64;
                let rise = (hz - points[m]) / (points[m + 1] - points[m]);
                let fall = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
                filters[m * BINS + k] =
                    (rise.min(fall).max(0.0) * 2.0 / (points[m + 2] - points[m])) as f32;
            }
        }
        let filter_ranges = std::array::from_fn(|m| {
            let row = &filters[m * BINS..(m + 1) * BINS];
            let start = row.iter().position(|&v| v != 0.0).unwrap_or(0);
            let end = row.iter().rposition(|&v| v != 0.0).map_or(0, |k| k + 1);
            start..end
        });
        Self {
            fft: FftPlanner::new().plan_fft_forward(FFT),
            window: std::array::from_fn(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos()) as f32
            }),
            filters,
            filter_ranges,
        }
    }
}

#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
impl QwenFrontend {
    pub fn extract(&self, samples: &[f32]) -> Result<MelFeatures> {
        if samples.len() < SAMPLE_RATE as usize / 2
            || samples.len() > MAX_SAMPLES
            || samples.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "audio must be finite PCM between 0.5 and 30 seconds",
            ));
        }
        let frames = samples.len() / HOP;
        let mut values = vec![0.0; MEL_BINS * frames];
        let mut input = vec![Complex32::default(); FFT];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        let mut power = [0.0; BINS];
        let mut maximum = f32::NEG_INFINITY;
        for frame in 0..frames {
            for (i, value) in input.iter_mut().enumerate() {
                let pos = frame as isize * HOP as isize + i as isize - (FFT / 2) as isize;
                let pos = if pos < 0 {
                    -pos
                } else if pos >= samples.len() as isize {
                    2 * samples.len() as isize - 2 - pos
                } else {
                    pos
                } as usize;
                *value = Complex32::new(samples[pos] * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut input, &mut scratch);
            for k in 0..BINS {
                power[k] = input[k].norm_sqr();
            }
            let finite_power = power.iter().all(|v| v.is_finite());
            for m in 0..MEL_BINS {
                // Preserve 0 * non-finite behavior outside the normal PCM range.
                let range = if finite_power {
                    self.filter_ranges[m].clone()
                } else {
                    0..BINS
                };
                let energy: f32 = self.filters[m * BINS + range.start..m * BINS + range.end]
                    .iter()
                    .zip(&power[range])
                    .map(|(a, b)| a * b)
                    .sum();
                let value = log10_f32(energy.max(1e-10));
                values[m * frames + frame] = value;
                maximum = maximum.max(value);
            }
        }
        for value in &mut values {
            *value = (value.max(maximum - 8.0) + 4.0) / 4.0;
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("PCM amplitude exceeds frontend range"));
        }
        Ok(MelFeatures { values, frames })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resample(rate: u32, input: &[f32]) -> Vec<f32> {
        let mut r = Resampler::new(rate).unwrap();
        let mut out = r.push(input);
        out.extend(r.finish());
        out
    }

    #[test]
    fn resampler_keeps_dc_tone_and_length() {
        for rate in [8_000, 22_050, 24_000, 32_000, 44_100, 48_000] {
            let out = resample(rate, &vec![0.5; rate as usize]);
            assert_eq!(out.len(), 16_000, "{rate}");
            assert!(out[2_000..14_000].iter().all(|v| (v - 0.5).abs() < 1e-4), "{rate}: DC");
            let tone: Vec<f32> = (0..rate)
                .map(|i| (2.0 * std::f64::consts::PI * 1_000.0 * i as f64 / rate as f64).sin() as f32)
                .collect();
            let out = resample(rate, &tone);
            let rms = (out[2_000..14_000].iter().map(|v| v * v).sum::<f32>() / 12_000.0).sqrt();
            assert!((rms - std::f32::consts::FRAC_1_SQRT_2).abs() < 5e-3, "{rate}: 1 kHz rms {rms}");
            let expected: Vec<f32> = (0..16_000)
                .map(|n| (2.0 * std::f64::consts::PI * 1_000.0 * n as f64 / 16_000.0).sin() as f32)
                .collect();
            let err = out[2_000..14_000].iter().zip(&expected[2_000..14_000]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(err < 5e-3, "{rate}: phase-aligned error {err}");
        }
    }

    #[test]
    fn resampler_pieces_match_one_buffer() {
        let input: Vec<f32> = (0..30_011u32).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
        for rate in [8_000, 22_050, 44_100, 48_000] {
            let whole = resample(rate, &input);
            let mut r = Resampler::new(rate).unwrap();
            let mut pieces = Vec::new();
            let mut at = 0;
            for size in [1usize, 317, 4_000, 29, 16_000].iter().cycle() {
                if at >= input.len() {
                    break;
                }
                let end = (at + size).min(input.len());
                pieces.extend(r.push(&input[at..end]));
                at = end;
            }
            pieces.extend(r.finish());
            assert_eq!(pieces, whole, "{rate}");
        }
        assert_eq!(resample(16_000, &input), input);
        assert!(Resampler::new(7_999).is_err() && Resampler::new(48_001).is_err());
    }

    /// The resampler's sums as first written: every tap bounds-checked, over the whole input.
    fn resample_per_tap(rate: u32, input: &[f32]) -> Vec<f32> {
        let r = Resampler::new(rate).unwrap();
        let total = (input.len() as u64 * r.up).div_ceil(r.down);
        (0..total)
            .map(|n| {
                let pos = n * r.down;
                let (center, phase) = (pos / r.up, (pos % r.up) as usize);
                let taps = &r.taps[phase * 2 * r.half..(phase + 1) * 2 * r.half];
                let first = center as i64 - r.half as i64 + 1;
                let mut acc = 0f32;
                for (i, &w) in taps.iter().enumerate() {
                    let index = first + i as i64;
                    if index >= 0 && (index as usize) < input.len() {
                        acc += w * input[index as usize];
                    }
                }
                acc
            })
            .collect()
    }

    #[test]
    fn resampler_matches_the_per_tap_reference() {
        let input: Vec<f32> = (0..50_021u32).map(|i| ((i * 7919) % 2003) as f32 / 1001.0 - 1.0).collect();
        for rate in [8_000, 11_025, 22_050, 24_000, 44_100, 48_000] {
            let (got, want) = (resample(rate, &input), resample_per_tap(rate, &input));
            assert!(got.len() == want.len() && got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{rate}");
        }
    }

    /// Host cost of the ASR request frontend (release, `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn frontend_microbench() {
        let seconds = 5;
        for rate in [24_000u32, 44_100, 48_000] {
            let input: Vec<f32> = (0..rate * seconds).map(|i| ((i * 7919) % 2003) as f32 / 1001.0 - 1.0).collect();
            let t = std::time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(resample(rate, &input));
            }
            let fast = t.elapsed().as_secs_f64() * 1e3 / 10.0;
            let t = std::time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(resample_per_tap(rate, &input));
            }
            let per_tap = t.elapsed().as_secs_f64() * 1e3 / 10.0;
            println!("FRONTEND resample {rate}->16k {seconds}s: {fast:.3} ms (per-tap bounds checks {per_tap:.3})");
        }
        let samples: Vec<f32> = (0..SAMPLE_RATE * seconds).map(|i| ((i * 17 % 101) as i32 - 50) as f32 / 500.0).collect();
        let qwen = QwenFrontend::default();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            std::hint::black_box(qwen.extract(&samples).unwrap());
        }
        println!("FRONTEND qwen log-mel {seconds}s: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 10.0);
    }

    #[test]
    fn sparse_filters_match_dense_projection() {
        let sparse = QwenFrontend::default();
        let mut dense = QwenFrontend::default();
        dense.filter_ranges = std::array::from_fn(|_| 0..BINS);
        for n in [8000, 8001, 15999, MAX_SAMPLES] {
            for amplitude in [0.0, 1.0, 1000.0, f32::MAX] {
                let samples: Vec<_> = (0..n)
                    .map(|i| ((i * 17 % 101) as i32 - 50) as f32 / 50.0 * amplitude)
                    .collect();
                match (sparse.extract(&samples), dense.extract(&samples)) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a.frames, b.frames);
                        assert_eq!(a.values, b.values, "samples={n} amplitude={amplitude}");
                    }
                    (Err(_), Err(_)) => {}
                    _ => panic!("sparse/dense result mismatch"),
                }
            }
        }
    }

    /// The configurable frontend with Whisper shaping reproduces the Whisper feature extractor.
    #[test]
    fn configurable_log_mel_matches_whisper_shaping() {
        let reference = QwenFrontend::default();
        let config = LogMelConfig {
            sample_rate: 16_000,
            fft: FFT,
            window: FFT,
            hop: HOP,
            bins: MEL_BINS,
            preemphasis: 0.0,
            center_window: false,
            periodic_hann: true,
            normalize_per_feature: false,
            mask_invalid_frames: false,
            log_guard: 1e-10,
            shaping: LogMelShaping { pad_reflect: true, drop_last_frame: true, log10: true, log_floor: true, dynamic_range: 8.0, scale: 0.25, shift: 4.0 },
        };
        let generic = LogMelFrontend::new(config, reference.filters.clone()).unwrap();
        let samples: Vec<f32> = (0..16_000 * 3 + 123).map(|i| ((i as f32 * 0.013).sin() * 0.3 + (i as f32 * 0.0007).cos() * 0.1)).collect();
        let want = reference.extract(&samples).unwrap();
        let got = generic.extract(&samples, false).unwrap();
        assert_eq!(got.frames, want.frames);
        let mut worst = 0f32;
        for frame in 0..want.frames {
            for bin in 0..MEL_BINS {
                worst = worst.max((got.values[frame * MEL_BINS + bin] - want.values[bin * want.frames + frame]).abs());
            }
        }
        assert!(worst <= 1e-6, "max abs diff {worst}");
    }

    #[test]
    fn configurable_log_mel_masks_centered_tail() {
        let config = LogMelConfig {
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
            shaping: LogMelShaping::default(),
        };
        let frontend = LogMelFrontend::new(config, vec![1.0; 10]).unwrap();
        let raw = frontend.extract(&[0.0; 8], false).unwrap();
        assert_eq!(raw.frames, 3);
        assert_eq!(raw.bins, 2);
        assert_eq!(&raw.values[..4], &[0.25f32.ln(); 4]);
        assert_eq!(&raw.values[4..], &[0.0; 2]);
        let normalized = frontend.extract(&[0.0; 8], true).unwrap();
        assert_eq!(normalized.values, vec![0.0; 6]);
    }

    #[test]
    fn silence_and_invalid_samples() {
        let frontend = QwenFrontend::default();
        let features = frontend.extract(&vec![0.0; 8000]).unwrap();
        assert_eq!(features.frames, 50);
        assert!(features.values.iter().all(|&v| v == -1.5));
        assert!(frontend.extract(&[0.0; 100]).is_err());
        assert!(frontend.extract(&vec![f32::NAN; 8000]).is_err());
    }

    #[test]
    fn matches_transformers_frontend_boundaries() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/qwen-asr-mel.json")).unwrap();
        let frontend = QwenFrontend::default();
        for case in fixtures["cases"].as_array().unwrap() {
            let n = case["samples"].as_u64().unwrap() as usize;
            let samples: Vec<_> = (0..n)
                .map(|i| ((i * 17 % 101) as i32 - 50) as f32 / 50.0)
                .collect();
            let mel = frontend.extract(&samples).unwrap();
            assert_eq!(mel.frames, case["frames"].as_u64().unwrap() as usize);
            for point in case["points"].as_array().unwrap() {
                let bin = point[0].as_u64().unwrap() as usize;
                let frame = point[1].as_u64().unwrap() as usize;
                let expected = point[2].as_f64().unwrap() as f32;
                let actual = mel.values[bin * mel.frames + frame];
                assert!(
                    (expected - actual).abs() < 2e-5,
                    "samples={n} bin={bin} frame={frame}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn wav_downmix_and_truncation() {
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut bytes,
                hound::WavSpec {
                    channels: 2,
                    sample_rate: SAMPLE_RATE,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for _ in 0..8000 {
                writer.write_sample(8192i16).unwrap();
                writer.write_sample(-8192i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        assert_eq!(decode_wav(bytes.get_ref()).unwrap(), vec![0.0; 8000]);
        bytes.get_mut().truncate(50);
        assert!(decode_wav(bytes.get_ref()).is_err());
    }
}
