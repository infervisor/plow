//! GPU Memory Headroom and Ring Park Sizing.
//!
//! Sizes sliding-ring reservations and emergency free-memory eviction floors at engine load.
//! Parked sliding rings and the block store dynamically share usable headroom above the floor,
//! with ring parking bounded to at most half the headroom and clamped to admitted concurrency.

use crate::config::RuntimeConfig;

#[derive(Clone, Copy, Debug)]
pub struct GpuMemoryBudgetParams<'a> {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub packet_batch: u32,
    pub ring_slot_charge: u64,
    pub config: &'a RuntimeConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMemoryBudget {
    /// Minimum emergency free memory floor (typically 2-4% of total).
    pub floor_bytes: u64,
    /// Usable headroom above the emergency floor (`free_bytes.saturating_sub(floor_bytes)`).
    pub usable_room_bytes: u64,
    /// Concurrency cap: bounded by `decode_max_rung` if specified, else `packet_batch`.
    pub concurrency_cap: u32,
    /// Per-slot sliding ring memory charge in bytes (0 if full attention only).
    pub ring_slot_charge: u64,
    /// Maximum memory committed by parked sliding rings in bytes.
    pub ring_park_cap_bytes: u64,
    /// Max slots allowed to park sliding rings.
    pub ring_park_slots: u32,
}

impl GpuMemoryBudget {
    /// Compute the memory headroom and sliding ring park limits.
    pub fn compute(params: GpuMemoryBudgetParams<'_>) -> Self {
        // 1. Minimum emergency floor: derived from config or sensible default (typically 2-4% of total).
        let floor_bytes = params
            .config
            .vmm_cache_min_free_bytes(params.total_bytes, Some(params.free_bytes))
            .unwrap_or(0);

        // 2. Usable headroom above the emergency floor.
        let usable_room_bytes = params.free_bytes.saturating_sub(floor_bytes);

        // 3. Concurrency cap: bounded by decode_max_rung if specified, else packet batch.
        let concurrency_cap = params
            .config
            .decode_max_rung
            .or(params.config.decode_elastic_rung)
            .or(params.config.decode_nominal_rung)
            .unwrap_or(params.packet_batch)
            .min(params.packet_batch);

        // 4. Sliding ring park cap:
        // When ring_slot_charge == 0 (e.g. full-attention LLMs, TTS models), park cap is 0.
        // When ring_slot_charge > 0 (Gemma hybrid), cap to nominal concurrency (preserving prefix cache headroom)
        // and at most half of usable room.
        let park_nominal = params
            .config
            .decode_nominal_rung
            .unwrap_or_else(|| {
                if params.config.decode_max_rung.is_some()
                    || params.config.decode_elastic_rung.is_some()
                {
                    32.min(concurrency_cap)
                } else {
                    concurrency_cap
                }
            })
            .min(concurrency_cap);
        let (ring_park_slots, ring_park_cap_bytes) = if params.ring_slot_charge > 0 {
            let max_slots_in_half_room = (usable_room_bytes / 2) / params.ring_slot_charge.max(1);
            let slots = max_slots_in_half_room
                .min(params.packet_batch as u64)
                .min(park_nominal as u64) as u32;
            (slots, slots as u64 * params.ring_slot_charge)
        } else {
            (0, 0)
        };

        Self {
            floor_bytes,
            usable_room_bytes,
            concurrency_cap,
            ring_slot_charge: params.ring_slot_charge,
            ring_park_cap_bytes,
            ring_park_slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Args, FromArgMatches};

    fn test_config() -> RuntimeConfig {
        let command = RuntimeConfig::augment_args(clap::Command::new("test"));
        RuntimeConfig::from_arg_matches(&command.try_get_matches_from(["test"]).unwrap()).unwrap()
    }

    #[test]
    fn test_budget_hybrid_gemma26b() {
        let total = 81559u64 << 20; // 80 GB H100
        let free = 51000u64 << 20;
        let ring_charge = 200u64 << 20;
        let batch = 128;
        let mut cfg = test_config();
        cfg.decode_max_rung = Some(32);

        let b = GpuMemoryBudget::compute(GpuMemoryBudgetParams {
            total_bytes: total,
            free_bytes: free,
            packet_batch: batch,
            ring_slot_charge: ring_charge,
            config: &cfg,
        });
        assert_eq!(b.concurrency_cap, 32);
        assert_eq!(b.ring_park_slots, 32);
        assert_eq!(b.ring_park_cap_bytes, 32 * ring_charge);
        assert!(b.usable_room_bytes > 45u64 << 30);
    }

    #[test]
    fn test_budget_full_attention_tts_or_qwen() {
        let total = 81559u64 << 20;
        let free = 52000u64 << 20;
        let ring_charge = 0u64; // No sliding layers
        let batch = 128;
        let cfg = test_config();

        let b = GpuMemoryBudget::compute(GpuMemoryBudgetParams {
            total_bytes: total,
            free_bytes: free,
            packet_batch: batch,
            ring_slot_charge: ring_charge,
            config: &cfg,
        });
        assert_eq!(b.ring_park_slots, 0);
        assert_eq!(b.ring_park_cap_bytes, 0);
        assert_eq!(b.usable_room_bytes, free - b.floor_bytes);
    }

    #[test]
    fn test_budget_constrained_memory_guards() {
        // High pressure: only 4 GB free remaining.
        let total = 81559u64 << 20;
        let free = 4000u64 << 20;
        let ring_charge = 200u64 << 20;
        let batch = 128;
        let cfg = test_config();

        let b = GpuMemoryBudget::compute(GpuMemoryBudgetParams {
            total_bytes: total,
            free_bytes: free,
            packet_batch: batch,
            ring_slot_charge: ring_charge,
            config: &cfg,
        });
        // Floor is preserved (free/8 = 500 MiB)
        assert!(b.floor_bytes > 0);
        // Ring park is strictly capped to at most half of usable room
        assert!(b.ring_park_cap_bytes <= b.usable_room_bytes / 2);
    }

    #[test]
    fn test_budget_tiny_ring_charge_no_wrap() {
        // Guard against u32 wrap when ring charge is tiny (e.g. 1 byte)
        let total = 81559u64 << 20;
        let free = 51000u64 << 20;
        let ring_charge = 1u64;
        let batch = 64;
        let cfg = test_config();

        let b = GpuMemoryBudget::compute(GpuMemoryBudgetParams {
            total_bytes: total,
            free_bytes: free,
            packet_batch: batch,
            ring_slot_charge: ring_charge,
            config: &cfg,
        });
        assert_eq!(b.ring_park_slots, 64);
        assert_eq!(b.ring_park_cap_bytes, 64);
    }
}
