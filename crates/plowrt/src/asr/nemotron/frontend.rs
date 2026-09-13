use gguf_rs_lib::format::metadata::Metadata;

use crate::asr::frontend::{LogMelConfig, LogMelFeatures, LogMelFrontend};
use crate::asset::gguf::GgufFile;
use crate::{Result, RuntimeError};

pub struct NemotronFrontend {
    inner: LogMelFrontend,
}

impl NemotronFrontend {
    pub fn load(model: &GgufFile) -> Result<Self> {
        let (config, filters) = Self::components(model)?;
        Ok(Self {
            inner: LogMelFrontend::new(config, filters)?,
        })
    }

    pub fn components(model: &GgufFile) -> Result<(LogMelConfig, Vec<f32>)> {
        let metadata = model.metadata();
        let sample_rate = integer(metadata, "asr.preprocessor.sample_rate")?;
        if sample_rate != crate::asr::frontend::SAMPLE_RATE as usize {
            return Err(rejected(format!(
                "sample rate {sample_rate} requires resampling support"
            )));
        }
        let fft = integer(metadata, "asr.preprocessor.n_fft")?;
        let bins = integer(metadata, "asr.preprocessor.features")?;
        let window = seconds(metadata, "asr.preprocessor.window_size", sample_rate)?;
        let hop = seconds(metadata, "asr.preprocessor.window_stride", sample_rate)?;
        let normalize_per_feature =
            string(metadata, "asr.preprocessor.normalize")? == "per_feature";
        let filter = model.tensor("preprocessor.fb")?;
        let expected_dimensions = [
            u64::try_from(fft / 2 + 1).unwrap(),
            u64::try_from(bins).unwrap(),
        ];
        if filter.dimensions != expected_dimensions {
            return Err(rejected(format!(
                "preprocessor.fb shape {:?}, expected {:?}",
                filter.dimensions, expected_dimensions
            )));
        }
        let config = LogMelConfig {
            sample_rate,
            fft,
            window,
            hop,
            bins,
            preemphasis: number(metadata, "asr.preprocessor.preemph")?,
            center_window: boolean(metadata, "asr.preprocessor.stft_center_window")?,
            periodic_hann: boolean(metadata, "asr.preprocessor.hann_periodic")?,
            normalize_per_feature,
            mask_invalid_frames: boolean(metadata, "asr.preprocessor.mask_invalid_frames")?,
            log_guard: 1.0 / 16_777_216.0,
        };
        Ok((config, filter.f32_values()?))
    }

    pub fn extract(&self, samples: &[f32]) -> Result<LogMelFeatures> {
        self.inner.extract(samples, true)
    }
}

fn integer(metadata: &Metadata, key: &str) -> Result<usize> {
    metadata
        .get_u64(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| rejected(format!("missing or invalid {key}")))
}

fn number(metadata: &Metadata, key: &str) -> Result<f32> {
    metadata
        .get_f64(key)
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
        .ok_or_else(|| rejected(format!("missing or invalid {key}")))
}

fn seconds(metadata: &Metadata, key: &str, sample_rate: usize) -> Result<usize> {
    let samples = f64::from(number(metadata, key)?) * sample_rate as f64;
    if samples <= 0.0 || samples > usize::MAX as f64 {
        return Err(rejected(format!("invalid {key}")));
    }
    Ok(samples.round() as usize)
}

fn string<'a>(metadata: &'a Metadata, key: &str) -> Result<&'a str> {
    metadata
        .get_string(key)
        .ok_or_else(|| rejected(format!("missing or invalid {key}")))
}

fn boolean(metadata: &Metadata, key: &str) -> Result<bool> {
    metadata
        .get_bool(key)
        .ok_or_else(|| rejected(format!("missing or invalid {key}")))
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("invalid Nemotron frontend: {}", message.into()))
}
