use std::io::Cursor;
use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::exec::packet_runtime::{BoundPacketPipeline, PacketRuntime};
use crate::{Result, RuntimeError};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 128;
const FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = FFT / 2 + 1;
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

fn invalid(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(message.into())
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("audio exceeds 30 seconds")]
    TooLong,
}

pub fn decode_wav(bytes: &[u8]) -> std::result::Result<Vec<f32>, AudioError> {
    let invalid = |message| AudioError::Invalid(message);
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| invalid(format!("invalid WAV: {e}")))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE || !(1..=2).contains(&spec.channels) {
        return Err(AudioError::Unsupported(
            "ASR requires 16 kHz mono/stereo WAV".into(),
        ));
    }
    if reader.duration() as usize > MAX_SAMPLES {
        return Err(AudioError::TooLong);
    }
    if reader.duration() < SAMPLE_RATE / 2 {
        return Err(invalid("audio must contain at least 0.5 seconds".into()));
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
        return Ok(samples);
    }
    Ok(samples
        .chunks_exact(2)
        .map(|s| s[0] * 0.5 + s[1] * 0.5)
        .collect())
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
        if pipeline.parameter("audio.frontend.kind")? != 1 {
            return Err(invalid("packet audio frontend kind is unsupported"));
        }
        let usize_param = |name| {
            usize::try_from(pipeline.parameter(name)?)
                .map_err(|_| invalid(format!("packet parameter {name:?} overflows")))
        };
        let bool_param = |name| match pipeline.parameter(name)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid(format!("packet parameter {name:?} is not boolean"))),
        };
        let f32_param = |name| {
            let bits = u32::try_from(pipeline.parameter(name)?)
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
        };
        let tensor = pipeline.tensor("audio.frontend.filterbank")?;
        let mut bytes = vec![0; tensor.bytes];
        runtime.read_tensor(tensor, &mut bytes)?;
        if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
            return Err(invalid("packet log-mel filterbank is not FP32"));
        }
        let filters = bytes
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
}

impl LogMelFrontend {
    pub fn new(config: LogMelConfig, filters: Vec<f32>) -> Result<Self> {
        if config.sample_rate == 0
            || !config.fft.is_power_of_two()
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
        let frames = samples.len() / config.hop + 1;
        let valid_frames = (samples.len() / config.hop).min(frames);
        let spectrum_bins = config.fft / 2 + 1;
        let mut preemphasized = samples.to_vec();
        if config.preemphasis != 0.0 {
            for index in (1..preemphasized.len()).rev() {
                preemphasized[index] -= config.preemphasis * preemphasized[index - 1];
            }
        }
        let mut values = vec![0.0; frames * config.bins];
        let mut spectrum = vec![Complex32::default(); config.fft];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; spectrum_bins];
        let window_offset = if config.center_window {
            (config.fft - config.window) / 2
        } else {
            0
        };
        let center_pad = config.fft / 2;
        for frame in 0..frames {
            spectrum.fill(Complex32::default());
            for index in 0..config.window {
                let padded_index = frame * config.hop + window_offset + index;
                let sample = padded_index
                    .checked_sub(center_pad)
                    .and_then(|index| preemphasized.get(index))
                    .copied()
                    .unwrap_or(0.0);
                spectrum[window_offset + index].re = sample * self.window[index];
            }
            self.fft.process_with_scratch(&mut spectrum, &mut scratch);
            for (power, value) in power.iter_mut().zip(&spectrum) {
                *power = value.norm_sqr();
            }
            for bin in 0..config.bins {
                let range = self.filter_ranges[bin].clone();
                let energy = self.filters
                    [bin * spectrum_bins + range.start..bin * spectrum_bins + range.end]
                    .iter()
                    .zip(&power[range])
                    .map(|(filter, power)| filter * power)
                    .sum::<f32>();
                values[frame * config.bins + bin] = (energy + config.log_guard).ln();
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

pub struct QwenFrontend {
    fft: Arc<dyn Fft<f32>>,
    window: [f32; FFT],
    filters: Vec<f32>,
    filter_ranges: [std::ops::Range<usize>; MEL_BINS],
}

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
                let value = energy.max(1e-10).log10();
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
