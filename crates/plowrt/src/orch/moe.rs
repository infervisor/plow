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

/// How ONE routed expert is spelled in the checkpoint on disk.
///
/// Three spellings reach this loader and they disagree on all four axes:
///
/// | checkpoint | sub-namespace | projections | payload | scale |
/// |---|---|---|---|---|
/// | GLM-5.2 / DeepSeek block-fp8 | `…mlp.` | `gate_proj`/`up_proj`/`down_proj` | `.weight` | `.weight_scale_inv` |
/// | Kimi-K2.7-Code MXFP4 | `…mlp.` | the same three | `.weight` | `.weight_scale` |
/// | Kimi-K3 (compressed-tensors mxfp4) | `…block_sparse_moe.` | `w1`/`w3`/`w2` | `.weight_packed` | `.weight_scale` |
///
/// The middle row is the reason this is RESOLVED and not switched on a flag: a
/// K2.7 checkpoint is the standard projection names with an E8M0 scale, so
/// "mxfp4" and "Mixtral-spelled" are independent facts and no single boolean
/// carries both. A flag that disagrees with the bytes is the failure this file
/// keeps finding; the bytes are the only thing that cannot disagree with itself.
///
/// `proj` is in `expert_weight_table` slot order — gate, up, down — which is why
/// the Mixtral row reads `w1`/`w3`/`w2` and not `w1`/`w2`/`w3`.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpertNames {
    /// Everything up to and including `experts.`; an expert index follows.
    pub(crate) ns: String,
    /// gate, up, down.
    pub(crate) proj: [&'static str; 3],
    /// `.weight` or `.weight_packed`.
    pub(crate) payload: &'static str,
    /// `.weight_scale_inv` (block-fp8 f32 grid) or `.weight_scale` (E8M0 row).
    pub(crate) scale: &'static str,
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
impl ExpertNames {
    /// `shared`: name the SHARED expert instead of routed expert `e`.
    ///
    /// GLM spells it `…mlp.shared_experts.{gate,up,down}_proj.*` beside
    /// `…mlp.experts.{e}.{gate,up,down}_proj.*`, so the substitution is exactly `experts.{e}.`
    /// -> `shared_experts.` and everything after it — projection name, payload suffix, scale
    /// suffix — is the routed spelling unchanged. Only the shared-expert fold passes `true`,
    /// and only for the last table entry.
    fn projection(&self, e: u32, j: usize, shared: bool, suffix: &str) -> String {
        let ns = match self.ns.strip_suffix("experts.") {
            Some(base) if shared => format!("{base}shared_experts."),
            _ => format!("{}{e}.", self.ns),
        };
        format!("{ns}{}{suffix}", self.proj[j])
    }

    pub(crate) fn weight_of(&self, e: u32, j: usize, shared: bool) -> String {
        self.projection(e, j, shared, self.payload)
    }

    pub(crate) fn scale_of(&self, e: u32, j: usize, shared: bool) -> String {
        self.projection(e, j, shared, self.scale)
    }

    /// Is the scale an MX microscaling row (one E8M0 byte per 32 elements along
    /// K) rather than a block-fp8 `[N/128][K/128]` f32 grid?
    ///
    /// Keyed on the SCALE's spelling, and both MX spellings are listed. `.weight_scale` is the
    /// compressed-tensors one that rides `.weight_packed`; `.scale` is DeepSeek-V4.1's, which
    /// rides a plain `.weight`. Only `.weight_scale_inv` -- block-fp8's -- is not MX, and
    /// enumerating the MX side rather than excluding that one means a spelling nobody has taught
    /// this function is read as block-fp8 and caught by `check_expert_geometry`'s grid arithmetic,
    /// rather than read as MX and accepted because the byte counts happened to line up.
    pub(crate) fn microscaled(&self) -> bool {
        self.scale == ".weight_scale" || self.scale == ".scale"
    }
}

/// Which spelling THIS checkpoint uses, decided by probing it.
///
/// `pfx` is what is left of the packet's `…expert_weight_table` after the suffix
/// is stripped, and it is not always a checkpoint prefix: the GLM emitter
/// declares the table under the model prefix (`model.layers.{l}.mlp.`), the K3
/// emitter under its own `moe.` namespace (`moe.language_model.model.layers.{l}.`)
/// because `packet::names` classifies compiler-owned tensors by that prefix. So
/// `moe.` is stripped and the MoE sub-namespace is probed rather than assumed.
///
/// ORDER IS THE COMPATIBILITY GUARANTEE. The first candidate is `{pfx}experts.0.
/// gate_proj.weight` + `.weight_scale_inv` — character for character the two
/// names this function replaced hardcoded — so a block-fp8 packet resolves on
/// probe one and every name built downstream is the name it was built before.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
pub(crate) fn resolve_expert_names(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    pfx: &str,
) -> crate::Result<ExpertNames> {
    const TEMPLATES: [([&str; 3], &str); 3] = [
        (["gate_proj", "up_proj", "down_proj"], ".weight"),
        (["w1", "w3", "w2"], ".weight_packed"),
        // DeepSeek-V4.1-Flash: `w1`/`w3`/`w2` with a PLAIN `.weight` payload and a `.scale` grid,
        // which is neither of the two above. LAST, so it can only be reached once the other two
        // have missed -- a checkpoint carrying `gate_proj.weight` still resolves as it always did,
        // and one carrying `w1.weight_packed` still prefers the packed spelling over this one.
        (["w1", "w3", "w2"], ".weight"),
    ];
    const SCALES: [&str; 3] = [".weight_scale_inv", ".weight_scale", ".scale"];
    let base = pfx.strip_prefix("moe.").unwrap_or(pfx);
    let mut tried: Vec<String> = Vec::new();
    for sub in ["", "mlp.", "block_sparse_moe."] {
        for (proj, payload) in TEMPLATES {
            let ns = format!("{base}{sub}experts.");
            let probe = format!("{ns}0.{}{payload}", proj[0]);
            if ckpt.tensor_ex(&probe).is_none() {
                tried.push(probe);
                continue;
            }
            // The payload is there, so this IS the layout — a missing scale is
            // now a broken checkpoint and not a wrong guess, and saying so beats
            // falling through to a spelling that cannot be right.
            for scale in SCALES {
                if ckpt
                    .tensor_ex(&format!("{ns}0.{}{scale}", proj[0]))
                    .is_some()
                {
                    return Ok(ExpertNames {
                        ns,
                        proj,
                        payload,
                        scale,
                    });
                }
            }
            return Err(crate::RuntimeError::Device(format!(
                "MISSING EXPERT SCALE: `{probe}` is in the checkpoint but neither \
                 `{ns}0.{}{}` nor `{ns}0.{}{}` is. A quantized expert without its scale \
                 cannot be dequantized, and binding the payload alone would decode from \
                 4-bit or 8-bit mantissas read as if they were already scaled.",
                proj[0], SCALES[0], proj[0], SCALES[1]
            )));
        }
    }
    Err(crate::RuntimeError::Device(format!(
        "MISSING EXPERT WEIGHT: the packet declares `{pfx}expert_weight_table` but the \
         checkpoint has no routed experts under any spelling this loader knows. Probed: \
         {tried:?}"
    )))
}

/// Fail unless expert 0's three scale twins are the right SIZE for the weights
/// they scale.
///
/// Every expert in a layer is the same shape, and `slice_for` re-checks each one
/// against the stride derived here — so this is the only place the WEIGHT and its
/// SCALE are compared to each other at all. Getting it wrong is silent in the
/// worst way: an E8M0 row and a block-fp8 grid can be the same number of bytes
/// for some geometries, so a size that merely "looks plausible" is exactly the
/// thing that must not be accepted.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
pub(crate) fn check_expert_geometry(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    n: &ExpertNames,
) -> crate::Result<()> {
    let miss = |name: &str| {
        crate::RuntimeError::Device(format!(
            "MISSING EXPERT WEIGHT: {name} (expert 0 resolved to the `{}` + `{}` layout \
             under `{}`, so every projection must be present in it)",
            n.payload, n.scale, n.ns
        ))
    };
    for j in 0..3 {
        let (wn, sn) = (n.weight_of(0, j, false), n.scale_of(0, j, false));
        let (w, ws) = ckpt.tensor_ex(&wn).ok_or_else(|| miss(&wn))?;
        let (s, ss) = ckpt.tensor_ex(&sn).ok_or_else(|| miss(&sn))?;
        let bad = |m: String| {
            Err(crate::RuntimeError::Device(format!(
                "EXPERT SCALE GEOMETRY: `{sn}` {ss:?} ({} B) cannot be the scale of `{wn}` \
                 {ws:?} ({} B): {m}",
                s.len(),
                w.len()
            )))
        };
        if ws.len() != 2 || ss.len() != 2 {
            return bad(
                "both must be 2-D — a routed expert is a matrix and its scale is \
                        a grid or a per-group row, never a vector"
                    .into(),
            );
        }
        let (wn0, wn1, sn0, sn1) = (ws[0], ws[1], ss[0], ss[1]);
        if n.microscaled() {
            // MX: payload is [N, K/2] (two fp4 per byte), scale is [N, K/32]
            // (one E8M0 byte per group of 32 along K). Both are u8, so the byte
            // count IS the element count.
            if w.len() != wn0 * wn1 || s.len() != sn0 * sn1 {
                return bad("an mxfp4 payload and its E8M0 scale are both u8, so each \
                            must be exactly the product of its shape"
                    .into());
            }
            if sn0 != wn0 {
                return bad(format!("the output dim disagrees: {wn0} vs {sn0}"));
            }
            if wn1 * 2 != sn1 * 32 {
                return bad(format!(
                    "K disagrees: the payload packs {} elements per row, the scale covers {}",
                    wn1 * 2,
                    sn1 * 32
                ));
            }
        } else {
            // Block-fp8: payload is [N, K] e4m3 (1 B/element), scale is
            // [ceil(N/128), ceil(K/128)] f32. Verified against
            // zai-org/GLM-5.2-FP8: [2048, 6144] -> [16, 48].
            const B: usize = 128;
            if w.len() != wn0 * wn1 {
                return bad("an fp8 e4m3 payload is 1 B/element, so it must be exactly \
                            the product of its shape"
                    .into());
            }
            let (gn, gk) = (wn0.div_ceil(B), wn1.div_ceil(B));
            if (sn0, sn1) != (gn, gk) || s.len() != gn * gk * 4 {
                return bad(format!(
                    "a block-fp8 scale grid must be [{gn}, {gk}] f32 ({} B)",
                    gn * gk * 4
                ));
            }
        }
    }
    Ok(())
}
