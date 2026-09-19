//! Per-model serving configuration: what the CHECKPOINT says about how it
//! wants to be sampled and how it frames a reasoning trace.
//!
//! WHY THIS EXISTS. Every request was built from one process-wide
//! `SamplingParams::default()` — temperature 1.0, top_p 1.0, top_k off — for
//! every model, while `generation_config.json` (which this crate already reads,
//! for the eos set) carries the values the checkpoint's authors chose. Qwen3
//! ships 0.6/0.95/20; serving it at 1.0/1.0 is a quality regression against
//! vLLM, which applies the checkpoint's generation config by default, and it is
//! per-model by nature so there is nowhere else to put it.

use std::path::Path;

use crate::text::sample::SamplingParams;

/// Everything per-model the request path needs that is not the weights.
#[derive(Clone, Debug)]
pub struct ServingConfig {
    /// The checkpoint's own sampling defaults. A request field overrides one of
    /// these; a request that omits it gets the checkpoint's value, NOT the
    /// process-wide default.
    pub default_sampling: SamplingParams,
}

impl Default for ServingConfig {
    fn default() -> Self {
        ServingConfig {
            default_sampling: SamplingParams::default(),
        }
    }
}

impl ServingConfig {
    /// Read `generation_config.json` from `dir` or `dir/checkpoint` — the same
    /// two places [`crate::asset::checkpoint`] already looks for the eos set.
    ///
    /// Absent file, absent key or unparseable value each leave the
    /// corresponding default alone; a checkpoint that says nothing about
    /// sampling is served exactly as before.
    pub fn load(dir: &Path) -> Self {
        let mut cfg = ServingConfig::default();
        let Some(v) = [dir.join("generation_config.json"), dir.join("checkpoint").join("generation_config.json")]
            .iter()
            .find_map(|p| {
                std::fs::read_to_string(p)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            })
        else {
            return cfg;
        };
        let p = &mut cfg.default_sampling;
        if let Some(t) = v.get("temperature").and_then(serde_json::Value::as_f64) {
            p.temperature = t as f32;
        }
        if let Some(t) = v.get("top_p").and_then(serde_json::Value::as_f64) {
            p.top_p = t as f32;
        }
        // HF spells "no top-k" as 0 or as a missing key; the sampler spells it 0.
        if let Some(k) = v.get("top_k").and_then(serde_json::Value::as_u64) {
            p.top_k = k as usize;
        }
        if let Some(r) = v
            .get("repetition_penalty")
            .and_then(serde_json::Value::as_f64)
        {
            p.repetition_penalty = r as f32;
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;





    #[test]
    fn a_checkpoint_without_a_generation_config_keeps_the_stock_defaults() {
        let cfg = ServingConfig::load(Path::new("/nonexistent"));
        assert_eq!(cfg.default_sampling.temperature, 1.0);
        assert_eq!(cfg.default_sampling.top_p, 1.0);
    }

    #[test]
    fn the_checkpoints_sampling_defaults_are_read() {
        let dir = std::env::temp_dir().join("plowrt-servingconfig-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.6, "top_p": 0.95, "top_k": 20}"#,
        )
        .expect("write");
        let cfg = ServingConfig::load(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(cfg.default_sampling.temperature, 0.6);
        assert_eq!(cfg.default_sampling.top_p, 0.95);
        assert_eq!(cfg.default_sampling.top_k, 20);
    }
}
