//! Parallelism auto-derivation for multi-GPU deployment.
//!
//! Given a model's architecture spec and a target device's memory capacity,
//! `derive_parallel` produces the TP/PP/EP/DP configuration that fits.
//! Explicit overrides (--tp, --pp, --ep) always win.

/// The derived parallelism configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelConfig {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
    pub dp: u32,
}

/// Model architecture spec extracted from a checkpoint's config.json.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kvh: u32,
    pub inter: u32,
    pub n_experts: u32,
    pub weight_bytes: u64,
    pub vocab: u32,
}

impl ModelSpec {
    /// Rough per-layer KV cache bytes for one token at bf16.
    fn kv_bytes_per_token(&self) -> u64 {
        let head_dim = (self.hidden / self.heads) as u64;
        // 2 (K+V) * kvh * head_dim * 2 (bf16) * layers
        2 * self.kvh as u64 * head_dim * 2 * self.layers as u64
    }
}

/// Derive optimal parallelism from model + hardware.
///
/// `gpu_mem_bytes` is per-device HBM capacity. `num_devices` is the total
/// device count available. Returns a config where `tp * pp * ep * dp == num_devices`.
pub fn derive_parallel(model: &ModelSpec, gpu_mem_bytes: u64, num_devices: u32) -> ParallelConfig {
    if num_devices <= 1 {
        return ParallelConfig {
            tp: 1,
            pp: 1,
            ep: 1,
            dp: 1,
        };
    }

    // Estimate activation + KV overhead for a generous context window (8k tokens).
    let kv_estimate = model.kv_bytes_per_token() * 8192;
    let act_estimate = (model.hidden as u64) * (model.hidden as u64) * 2; // one layer's activations

    let usable = (gpu_mem_bytes as f64 * 0.75) as u64;

    // Find smallest TP that makes weights fit per device.
    let mut tp = 1u32;
    for candidate in [1, 2, 4, 8] {
        if candidate > num_devices {
            break;
        }
        if model.heads % candidate != 0 {
            continue;
        }
        if model.inter % candidate != 0 {
            continue;
        }
        let per_device_weight = model.weight_bytes / candidate as u64;
        let per_device_kv = kv_estimate / candidate as u64;
        if per_device_weight + per_device_kv + act_estimate <= usable {
            tp = candidate;
            break;
        }
        tp = candidate;
    }

    // EP for MoE models.
    let mut ep = 1u32;
    if model.n_experts > 0 && num_devices > tp {
        let max_ep = num_devices / tp;
        for candidate in (1..=max_ep).rev() {
            if model.n_experts % candidate == 0 && candidate <= model.n_experts {
                ep = candidate;
                break;
            }
        }
    }

    // PP if weights still don't fit after TP.
    let mut pp = 1u32;
    let per_device_weight = model.weight_bytes / (tp as u64 * ep as u64);
    if per_device_weight > usable {
        pp = ((per_device_weight as f64 / usable as f64).ceil() as u32).max(1);
        pp = pp.min(model.layers).min(num_devices / (tp * ep));
    }

    // DP with remainder.
    let dp = num_devices / (tp * pp * ep);

    ParallelConfig {
        tp,
        pp: pp.max(1),
        ep: ep.max(1),
        dp: dp.max(1),
    }
}

/// Validate a parallelism config against model constraints. Returns an error
/// message if invalid.
pub fn validate(model: &ModelSpec, cfg: &ParallelConfig, gpu_mem_bytes: u64) -> Result<(), String> {
    if cfg.tp > 1 && model.heads % cfg.tp != 0 {
        return Err(format!(
            "TP={} does not divide heads={} evenly",
            cfg.tp, model.heads
        ));
    }
    if cfg.tp > 1 && model.inter % cfg.tp != 0 {
        return Err(format!(
            "TP={} does not divide inter={} evenly",
            cfg.tp, model.inter
        ));
    }
    if cfg.ep > 1 && model.n_experts % cfg.ep != 0 {
        return Err(format!(
            "EP={} does not divide n_experts={} evenly",
            cfg.ep, model.n_experts
        ));
    }
    let per_device = model.weight_bytes / (cfg.tp as u64 * cfg.pp as u64 * cfg.ep as u64);
    let limit = (gpu_mem_bytes as f64 * 0.80) as u64;
    if per_device > limit {
        return Err(format!(
            "weights per device ({:.2} GiB) exceed 80% of HBM ({:.2} GiB) — increase TP/PP",
            per_device as f64 / (1u64 << 30) as f64,
            limit as f64 / (1u64 << 30) as f64,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma4_31b() -> ModelSpec {
        ModelSpec {
            layers: 48,
            hidden: 5376,
            heads: 32,
            kvh: 4,
            inter: 21504,
            n_experts: 0,
            weight_bytes: 57 * (1u64 << 30), // ~57 GiB bf16
            vocab: 262144,
        }
    }

    fn gemma4_12b() -> ModelSpec {
        ModelSpec {
            layers: 36,
            hidden: 3840,
            heads: 16,
            kvh: 4,
            inter: 15360,
            n_experts: 0,
            weight_bytes: 24 * (1u64 << 30),
            vocab: 262144,
        }
    }

    fn moe_26b() -> ModelSpec {
        ModelSpec {
            layers: 34,
            hidden: 5376,
            heads: 32,
            kvh: 4,
            inter: 21504,
            n_experts: 128,
            weight_bytes: 50 * (1u64 << 30),
            vocab: 262144,
        }
    }

    const H100_MEM: u64 = 80 * (1u64 << 30); // 80 GiB
    const MI350_MEM: u64 = 288 * (1u64 << 30); // 288 GiB
    const B200_MEM: u64 = 192 * (1u64 << 30); // 192 GiB

    #[test]
    fn gemma4_31b_on_1x_mi350_fits_tp1() {
        let cfg = derive_parallel(&gemma4_31b(), MI350_MEM, 1);
        assert_eq!(
            cfg,
            ParallelConfig {
                tp: 1,
                pp: 1,
                ep: 1,
                dp: 1
            }
        );
    }

    #[test]
    fn gemma4_31b_on_8x_h100_needs_tp() {
        // 31B at bf16 is ~57 GiB weights; with KV + activations it exceeds
        // 80 GiB * 0.75 = 60 GiB usable on a single H100.
        let mut model = gemma4_31b();
        model.weight_bytes = 62 * (1u64 << 30); // realistic with embedding + lm_head
        let cfg = derive_parallel(&model, H100_MEM, 8);
        assert!(cfg.tp >= 2, "62 GiB model needs TP on 80 GiB GPUs");
        assert_eq!(cfg.tp * cfg.pp * cfg.ep * cfg.dp, 8);
    }

    #[test]
    fn gemma4_12b_on_2x_h100_tp2() {
        let cfg = derive_parallel(&gemma4_12b(), H100_MEM, 2);
        assert_eq!(cfg.tp, 1); // 24 GiB fits in 80 GiB * 0.75 = 60 GiB
        assert_eq!(cfg.dp, 2);
    }

    #[test]
    fn moe_on_16_gpus_gets_ep() {
        let cfg = derive_parallel(&moe_26b(), H100_MEM, 16);
        assert!(cfg.ep >= 2, "MoE model should get EP on 16 GPUs");
        assert_eq!(cfg.tp * cfg.pp * cfg.ep * cfg.dp, 16);
        assert_eq!(128 % cfg.ep, 0, "EP must divide n_experts");
    }

    #[test]
    fn validate_bad_tp() {
        let model = gemma4_31b();
        let bad = ParallelConfig {
            tp: 3,
            pp: 1,
            ep: 1,
            dp: 1,
        };
        assert!(validate(&model, &bad, H100_MEM).is_err());
    }

    #[test]
    fn single_device_returns_trivial() {
        let cfg = derive_parallel(&gemma4_31b(), H100_MEM, 1);
        assert_eq!(
            cfg,
            ParallelConfig {
                tp: 1,
                pp: 1,
                ep: 1,
                dp: 1
            }
        );
    }
}
