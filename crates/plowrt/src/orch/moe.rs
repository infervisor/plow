//! §L MoE expert-parallel dispatch — consumes the `experts.json` sidecar.
//!
//! EP dispatch is SM-local (host out of the loop): the SM reads the per-request
//! routing table and remaps the weight base for the selected expert, or returns
//! early on an unused slot. The host's job is to resolve the compile-time expert
//! weight names into the flat base-pointer table the SM indexes, at model load.

use plow_asset::Experts;

use crate::memory::AddressSpace;

/// Resolve each routed layer's expert weight names to physical base addresses,
/// producing the flat per-expert base table the SM indexes by routed expert id.
///
/// Order (the load-time half of the common-expert-segment mechanism,
/// `moe-plow-design.md §4b`, `moe-ep-kernels.md §3a`): the shared experts first (kept from
/// the original skeleton), then, per routed layer, each expert's `{gate, up, down}` bases in
/// id order — the exact `[num_experts][3]` layout the `MoeExpertGlu`/`MoeExpertDown` weight
/// prologue indexes as `expert_weight_table[expert_id * 3 + {0,1,2}]` in `op_moe.h`.
pub fn resolve_expert_tables(experts: &Experts, space: &AddressSpace, device: u8) -> Vec<u64> {
    build_expert_table(experts, |name| space.addr_of(name, device).ok())
}

/// The pure resolution — testable without an [`AddressSpace`]. `resolve` maps a weight name
/// to its physical base (or `None` if absent). A missing name is skipped, matching the
/// skeleton's tolerance for a partial address map.
pub fn build_expert_table<F>(experts: &Experts, mut resolve: F) -> Vec<u64>
where
    F: FnMut(&str) -> Option<u64>,
{
    let mut bases = Vec::new();
    for shared in &experts.shared {
        for name in [shared.gate_up_weight.as_ref(), shared.down_weight.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Some(addr) = resolve(name) {
                bases.push(addr);
            }
        }
    }
    // Routed experts: per layer, per expert, {gate, up, down} — the SM's two-level lookup
    // `expert_weight_table[expert_id]` reads exactly this triple.
    for layer in &experts.layers {
        for e in &layer.routed_experts {
            for name in [&e.gate, &e.up, &e.down] {
                if let Some(addr) = resolve(name) {
                    bases.push(addr);
                }
            }
        }
    }
    bases
}

/// The sentinel a router writes for an unused expert slot; the SM skips compute
/// (signals the completion counter only) when `expert_id >= num_experts`.
pub fn unused_sentinel(experts: &Experts) -> u32 {
    experts.expert_unused_sentinel
}

/// The `[E][3]` pointer table for ONE **packed** MoE layer (GLM-5.2 / DeepSeek).
///
/// GLM's 256 routed experts are deliberately NOT declared packet tensors
/// (`crates/devgen/src/mla.rs`: 75 layers x 256 x 6 handles for zero emit
/// benefit) — the ops only ever index the table. So the host packs each layer's
/// experts into ONE buffer and fills `mlp.expert_weight_table` /
/// `mlp.expert_scale_table` with addresses INTO it. This is the address
/// arithmetic for that, and it is the single source of truth the packer copies
/// against: slot `k` of the buffer lives at `base + k*stride`, and the table
/// entry for `(expert, proj)` is the address of the slot that was filled.
///
/// Same `[E][3] = {gate, up, down}` order as [`build_expert_table`] — what
/// `op_moe.h` reads as `wtab[eid*3 + {0,1,2}]`.
///
/// `owned` is the half-open expert range THIS rank packed:
/// * **TP** — every rank holds a `1/N` slice of every expert, so `0..n_exp`;
/// * **EP** — every rank holds `n_exp/N` WHOLE experts, so a contiguous block.
///
/// An expert outside `owned` keeps a **zero** entry, which is not an omission
/// but the interface: `d_moe_expert_glu` bails on `wtab[eid*3] == 0` and
/// `d_moe_expert_down` zeroes that slot's partial, so a remote expert costs the
/// rank nothing and the combine still sums a deterministic zero.
pub fn packed_expert_table(
    base: u64,
    stride: u64,
    n_exp: u32,
    owned: std::ops::Range<u32>,
) -> Vec<u64> {
    let mut table = vec![0u64; n_exp as usize * 3];
    for (slot, e) in owned.filter(|e| *e < n_exp).enumerate() {
        for j in 0..3 {
            table[e as usize * 3 + j] = base + (slot as u64 * 3 + j as u64) * stride;
        }
    }
    table
}

/// Offset-based expert-table resolution for **FUSED 3-D expert tensors** (Gemma-4 26B-A4B).
/// Unlike GLM/DeepSeek — where each expert is a separately
/// named `{gate, up, down}` tensor resolved by [`build_expert_table`] — Gemma stores ONE
/// `experts.gate_up_proj [E, 2·I, H]` and ONE `experts.down_proj [E, H, I]` per layer. The SM's
/// two-level lookup therefore indexes `expert_weight_table[eid*2 + {0,1}] = {gate_up base, down
/// base}`, with the per-expert base a byte offset into the fused tensor: `base + eid·stride`.
///
/// `gate_up_base`/`down_base` are the two fused tensors' device bases; the strides are their
/// per-expert byte pitches (`2·I·H·2` and `H·I·2` for bf16). Returns the flat `[E][2]` u64 table.
/// The name-based [`build_expert_table`] path (GLM, `[E][3]`) is unchanged.
pub fn build_fused_expert_table(
    gate_up_base: u64,
    down_base: u64,
    num_experts: u32,
    gate_up_stride: u64,
    down_stride: u64,
) -> Vec<u64> {
    let mut bases = Vec::with_capacity(num_experts as usize * 2);
    for e in 0..num_experts as u64 {
        bases.push(gate_up_base + e * gate_up_stride);
        bases.push(down_base + e * down_stride);
    }
    bases
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::{ExpertLayer, RoutedExpertWeights, SharedExpert};
    use std::collections::HashMap;

    /// The routed table resolves in `[num_experts][3]` = `{gate, up, down}` order (after the
    /// shared experts), which is exactly what the `op_moe.h` weight prologue indexes.
    #[test]
    fn resolve_routed_experts_in_gate_up_down_order() {
        let mk = |g: &str, u: &str, d: &str| RoutedExpertWeights {
            gate: g.into(),
            up: u.into(),
            down: d.into(),
        };
        let experts = Experts {
            layers: vec![ExpertLayer {
                block: 1,
                layer_label: "l1".into(),
                num_experts: 2,
                top_k: 2,
                router_op_name: "moe_router_1".into(),
                routing_table_slot: "rt_1".into(),
                expert_weight_table_slot: "ewt_1".into(),
                routed_experts: vec![
                    mk("l1.e0.gate", "l1.e0.up", "l1.e0.down"),
                    mk("l1.e1.gate", "l1.e1.up", "l1.e1.down"),
                ],
            }],
            shared: vec![SharedExpert {
                block: 1,
                layer_label: "l1".into(),
                gate_up_weight: Some("l1.shared.gate_up".into()),
                down_weight: Some("l1.shared.down".into()),
                replicated_across_gpus: false,
            }],
            expert_unused_sentinel: u32::MAX,
            complete: true,
        };
        // A resolver that hands each name a distinct fake base address.
        let addrs: HashMap<&str, u64> = [
            ("l1.shared.gate_up", 0x1000),
            ("l1.shared.down", 0x2000),
            ("l1.e0.gate", 0x3000),
            ("l1.e0.up", 0x3100),
            ("l1.e0.down", 0x3200),
            ("l1.e1.gate", 0x4000),
            ("l1.e1.up", 0x4100),
            ("l1.e1.down", 0x4200),
        ]
        .into_iter()
        .collect();
        let table = build_expert_table(&experts, |n| addrs.get(n).copied());
        assert_eq!(
            table,
            vec![0x1000, 0x2000, 0x3000, 0x3100, 0x3200, 0x4000, 0x4100, 0x4200],
            "shared first, then routed experts in {{gate, up, down}} × id order"
        );
    }

    /// A sidecar with no routed names (pre-resolution) still yields just the shared bases —
    /// backward-compatible with the skeleton.
    #[test]
    fn empty_routed_experts_is_shared_only() {
        let experts = Experts {
            layers: vec![],
            shared: vec![SharedExpert {
                block: 0,
                layer_label: "l0".into(),
                gate_up_weight: Some("s.gu".into()),
                down_weight: Some("s.d".into()),
                replicated_across_gpus: false,
            }],
            expert_unused_sentinel: u32::MAX,
            complete: false,
        };
        let table = build_expert_table(&experts, |n| match n {
            "s.gu" => Some(7),
            "s.d" => Some(8),
            _ => None,
        });
        assert_eq!(table, vec![7, 8]);
    }

    /// TP: every rank packs a slice of EVERY expert, so the table is dense and
    /// walks the buffer in `[E][3]` order with no holes.
    #[test]
    fn packed_table_under_tp_is_dense() {
        let t = packed_expert_table(0x1000, 0x10, 3, 0..3);
        assert_eq!(
            t,
            vec![0x1000, 0x1010, 0x1020, 0x1030, 0x1040, 0x1050, 0x1060, 0x1070, 0x1080]
        );
    }

    /// EP: a rank packs only its contiguous block of WHOLE experts. The block is
    /// dense in the BUFFER (slot 0 is the first local expert) but sparse in the
    /// TABLE, and every remote expert must read back as a null base — that zero
    /// is what makes the kernel skip it instead of dereferencing a stale address.
    #[test]
    fn packed_table_under_ep_is_null_outside_the_local_block() {
        // 4 experts, 2 ranks: rank 1 owns {2,3} and packs them at slots 0,1.
        let t = packed_expert_table(0x2000, 0x100, 4, 2..4);
        assert_eq!(t[..6], [0u64; 6], "remote experts stay NULL");
        assert_eq!(
            t[6..],
            [0x2000, 0x2100, 0x2200, 0x2300, 0x2400, 0x2500],
            "local experts pack from slot 0 of this rank's buffer"
        );
    }

    /// Fused (Gemma-4) resolution: `[E][2] = {gate_up base + e·stride, down base + e·stride}`,
    /// the offset-based twin of the name-based GLU path.
    #[test]
    fn fused_expert_table_is_base_plus_stride() {
        // E=3, gate_up base 0x1000 stride 0x100, down base 0x9000 stride 0x40.
        let t = build_fused_expert_table(0x1000, 0x9000, 3, 0x100, 0x40);
        assert_eq!(
            t,
            vec![0x1000, 0x9000, 0x1100, 0x9040, 0x1200, 0x9080],
            "interleaved {{gate_up, down}} per expert, each base + e·stride"
        );
    }
}
