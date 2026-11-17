//! Unified GPU Memory Budget Ledger.
//!
//! Replaces fragmented memory calculations across `prefix.rs`, `gpu.rs`, and `vmm.rs`
//! with a single deterministic VRAM ledger computed at engine load.
//!
//! Partition formula:
//! Total VRAM = Weights + Workspaces + Sliding Rings + KV Cache + Min Free Floor

use crate::config::RuntimeConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMemoryBudget {
    /// Total VRAM on device in bytes.
    pub total_bytes: u64,
    /// Free VRAM after weights and runtime load in bytes.
    pub free_after_load_bytes: u64,
    /// Static weights size in bytes.
    pub weights_bytes: u64,
    /// Minimum emergency free memory floor (typically 2-4% of total).
    pub floor_bytes: u64,
    /// Max admitted concurrency (slots).
    pub concurrency_cap: u32,
    /// Per-slot sliding ring memory charge (bytes, 0 if full attention only).
    pub ring_slot_charge: u64,
    /// Maximum memory committed by parked sliding rings in bytes.
    pub ring_park_cap_bytes: u64,
    /// Max slots allowed to park sliding rings.
    pub ring_park_slots: u32,
    /// Memory assigned to the dynamic KV block cache / prefix store in bytes.
    pub kv_cache_cap_bytes: u64,
}

impl GpuMemoryBudget {
    /// Compute the unified memory budget for any model family (LLM, Sliding, Recurrent, TTS)
    /// across single-model or multi-model co-tenant deployments.
    pub fn compute(
        total_bytes: u64,
        free_bytes: u64,
        weights_bytes: u64,
        packet_batch: u32,
        ring_slot_charge: u64,
        config: &RuntimeConfig,
    ) -> Self {
        // 1. Minimum emergency floor: derived from config or sensible default (typically 2-4% of total).
        let floor_bytes = match config.vmm_cache_min_free_bytes(total_bytes, Some(free_bytes)) {
            Some(f) => f,
            None => 0,
        };

        // 2. Usable headroom above the emergency floor.
        let usable_room = free_bytes.saturating_sub(floor_bytes);

        // 3. Concurrency cap: bounded by decode_max_rung if specified, else packet batch.
        let concurrency_cap = config
            .decode_max_rung
            .unwrap_or(packet_batch)
            .min(packet_batch);

        // 4. Sliding ring park cap:
        // When ring_slot_charge == 0 (e.g. Qwen full-attention, TTS models), park cap is 0.
        // When ring_slot_charge > 0 (Gemma hybrid), cap to concurrency_cap and half of usable room.
        let (ring_park_slots, ring_park_cap_bytes) = if ring_slot_charge > 0 {
            let max_slots_in_half_room = (usable_room / 2) / ring_slot_charge.max(1);
            let slots = (max_slots_in_half_room as u32)
                .min(packet_batch)
                .min(concurrency_cap);
            (slots, slots as u64 * ring_slot_charge)
        } else {
            (0, 0)
        };

        // 5. Dynamic KV cache / prefix block store budget.
        // Takes all remaining usable memory after the ring reservation.
        // If an explicit cap is configured, it is respected but clamped to available usable room
        // so that misconfigured large values never trigger out-of-memory errors.
        let remaining_room = usable_room.saturating_sub(ring_park_cap_bytes);
        let explicit_cap = config.prefix_cache_cap_bytes(total_bytes);
        let kv_cache_cap_bytes = if explicit_cap > 0 {
            explicit_cap.min(remaining_room)
        } else {
            remaining_room
        };

        Self {
            total_bytes,
            free_after_load_bytes: free_bytes,
            weights_bytes,
            floor_bytes,
            concurrency_cap,
            ring_slot_charge,
            ring_park_cap_bytes,
            ring_park_slots,
            kv_cache_cap_bytes,
        }
    }

    #[inline]
    pub fn usable_room_bytes(&self) -> u64 {
        self.free_after_load_bytes.saturating_sub(self.floor_bytes)
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
        let weights = 25000u64 << 20;
        let ring_charge = 200u64 << 20;
        let batch = 128;
        let mut cfg = test_config();
        cfg.decode_max_rung = Some(32);

        let b = GpuMemoryBudget::compute(total, free, weights, batch, ring_charge, &cfg);
        assert_eq!(b.concurrency_cap, 32);
        assert_eq!(b.ring_park_slots, 32);
        assert_eq!(b.ring_park_cap_bytes, 32 * ring_charge);
        assert!(b.kv_cache_cap_bytes > 40u64 << 30);
    }

    #[test]
    fn test_budget_full_attention_tts_or_qwen() {
        let total = 81559u64 << 20;
        let free = 52000u64 << 20;
        let weights = 15000u64 << 20;
        let ring_charge = 0u64; // No sliding layers
        let batch = 128;
        let cfg = test_config();

        let b = GpuMemoryBudget::compute(total, free, weights, batch, ring_charge, &cfg);
        assert_eq!(b.ring_park_slots, 0);
        assert_eq!(b.ring_park_cap_bytes, 0);
        assert!(b.kv_cache_cap_bytes > 45u64 << 30);
    }

    #[test]
    fn test_budget_explicit_prefix_cap() {
        let total = 81559u64 << 20;
        let free = 51000u64 << 20;
        let weights = 25000u64 << 20;
        let ring_charge = 200u64 << 20;
        let batch = 128;
        let mut cfg = test_config();
        cfg.decode_max_rung = Some(32);
        cfg.vmm_cache_mib = Some(16384); // 16 GiB

        let b = GpuMemoryBudget::compute(total, free, weights, batch, ring_charge, &cfg);
        assert_eq!(b.kv_cache_cap_bytes, 16384u64 << 20);
        assert_eq!(b.ring_park_cap_bytes, 32 * ring_charge);
    }

    #[test]
    fn test_budget_multi_model_co_tenancy() {
        // Co-tenancy: Model A already holds 40 GB on an 80 GB device.
        // Model B (TTS) loads with only 25 GB free.
        let total = 81559u64 << 20;
        let free = 25000u64 << 20;
        let weights = 5000u64 << 20;
        let ring_charge = 0u64;
        let batch = 64;
        let cfg = test_config();

        let b = GpuMemoryBudget::compute(total, free, weights, batch, ring_charge, &cfg);
        assert_eq!(b.ring_park_slots, 0);
        assert_eq!(b.ring_park_cap_bytes, 0);
        // All usable headroom above floor goes to KV/audio token cache
        assert_eq!(b.kv_cache_cap_bytes, b.usable_room_bytes());
        assert!(b.kv_cache_cap_bytes > 20u64 << 30);
    }

    #[test]
    fn test_budget_constrained_memory_guards() {
        // High co-tenant pressure: only 4 GB free remaining.
        let total = 81559u64 << 20;
        let free = 4000u64 << 20;
        let weights = 2000u64 << 20;
        let ring_charge = 200u64 << 20;
        let batch = 128;
        let cfg = test_config();

        let b = GpuMemoryBudget::compute(total, free, weights, batch, ring_charge, &cfg);
        // Floor is preserved (free/8 = 500 MiB)
        assert!(b.floor_bytes > 0);
        // Ring park is strictly capped to at most half of usable room
        assert!(b.ring_park_cap_bytes <= b.usable_room_bytes() / 2);
        // KV cache cap plus ring park cap never exceeds usable room
        assert!(b.ring_park_cap_bytes + b.kv_cache_cap_bytes <= b.usable_room_bytes());
    }

    #[test]
    fn test_budget_explicit_cap_clamped_to_headroom() {
        // Operator specifies an oversized 60 GiB cache cap, but only 30 GiB is usable.
        let total = 81559u64 << 20;
        let free = 32000u64 << 20;
        let weights = 25000u64 << 20;
        let ring_charge = 200u64 << 20;
        let batch = 128;
        let mut cfg = test_config();
        cfg.decode_max_rung = Some(32);
        cfg.vmm_cache_mib = Some(61440); // 60 GiB

        let b = GpuMemoryBudget::compute(total, free, weights, batch, ring_charge, &cfg);
        // Clamped to remaining usable room: never overcommits
        let max_kv = b.usable_room_bytes().saturating_sub(b.ring_park_cap_bytes);
        assert_eq!(b.kv_cache_cap_bytes, max_kv);
        assert!(b.ring_park_cap_bytes + b.kv_cache_cap_bytes <= b.usable_room_bytes());
    }
}
