//! Graph-derived expert-parallel prefill boundary planning.
//!
//! The planner deliberately knows no model names.  Expert ownership comes from
//! `(experts, ranks)`, while transport comes from the placement of the tensor
//! entering the routed FFN.  A replicated input needs no dispatch copy: every
//! rank can filter the common route table to its locally-owned experts.  A
//! token-sharded input sends one activation per `(token, destination rank)`, not
//! once per selected expert.

use crate::moe::RouteEntry;

pub const MOE_EP_ABI_VERSION: u32 = 1;

pub const MOE_EP_INPUT_REPLICATED: u32 = 1 << 0;
pub const MOE_EP_FIXED_SLOT_COMBINE: u32 = 1 << 1;

/// One-resident 2D expert layout. Expert groups partition experts; ranks within
/// a group partition each expert's intermediate dimension. `expert_degree *
/// intra_expert_tp == world_size`, so every checkpoint byte is resident once
/// per rank-equivalent, independent of the factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moe2dLayout {
    pub world_size: u32,
    pub expert_degree: u32,
}

impl Moe2dLayout {
    pub fn validate(self, experts: u32, full_intermediate: u32) -> Result<(), &'static str> {
        if self.world_size == 0
            || self.expert_degree == 0
            || !self.world_size.is_multiple_of(self.expert_degree)
        {
            return Err("expert degree must divide world size");
        }
        if experts < self.expert_degree {
            return Err("experts must cover every expert group");
        }
        if !full_intermediate.is_multiple_of(self.intra_expert_tp()) {
            return Err("intermediate width must divide intra-expert TP");
        }
        Ok(())
    }

    pub const fn intra_expert_tp(self) -> u32 {
        self.world_size / self.expert_degree
    }

    pub fn expert_group(self, rank: u32) -> u32 {
        assert!(rank < self.world_size);
        rank / self.intra_expert_tp()
    }

    pub fn intra_rank(self, rank: u32) -> u32 {
        assert!(rank < self.world_size);
        rank % self.intra_expert_tp()
    }

    pub fn expert_range(self, experts: u32, rank: u32) -> core::ops::Range<u32> {
        balanced_expert_range(experts, self.expert_degree, self.expert_group(rank))
    }

    pub fn local_intermediate(self, full_intermediate: u32) -> u32 {
        assert!(full_intermediate.is_multiple_of(self.intra_expert_tp()));
        full_intermediate / self.intra_expert_tp()
    }

    /// Resident MXFP4 bytes for one rank. Payload is two values/byte and E8M0
    /// scales are one byte per block of 32 along K. The optional stage-2 view
    /// has the same down payload plus its pad256x8 scale slab.
    pub fn mxfp4_resident_bytes(
        self,
        experts: u32,
        hidden: u32,
        full_intermediate: u32,
        layers: u32,
        stage2_view: bool,
    ) -> Result<Moe2dResidentBytes, &'static str> {
        self.validate(experts, full_intermediate)?;
        if hidden == 0 || layers == 0 || !full_intermediate.is_multiple_of(32) {
            return Err("MXFP4 geometry must have non-zero H/layers and I divisible by 32");
        }
        let local_experts = self.expert_range(experts, 0).len() as u64;
        let local_i = u64::from(self.local_intermediate(full_intermediate));
        let h = u64::from(hidden);
        let matrix_payload = h
            .checked_mul(local_i)
            .and_then(|n| n.checked_div(2))
            .ok_or("payload bytes overflow")?;
        let matrix_scales = h.checked_mul(local_i / 32).ok_or("scale bytes overflow")?;
        let primary_per_layer = local_experts
            .checked_mul(
                3u64.checked_mul(matrix_payload + matrix_scales)
                    .ok_or("primary bytes overflow")?,
            )
            .ok_or("primary bytes overflow")?;
        let stage2_per_layer = if stage2_view {
            let padded_scales = h.div_ceil(256) * 256 * ((local_i / 32).div_ceil(8) * 8);
            local_experts
                .checked_mul(matrix_payload + padded_scales)
                .ok_or("stage-2 bytes overflow")?
        } else {
            0
        };
        let layers = u64::from(layers);
        Ok(Moe2dResidentBytes {
            primary: primary_per_layer
                .checked_mul(layers)
                .ok_or("primary bytes overflow")?,
            stage2_view: stage2_per_layer
                .checked_mul(layers)
                .ok_or("stage-2 bytes overflow")?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moe2dResidentBytes {
    pub primary: u64,
    pub stage2_view: u64,
}

impl Moe2dResidentBytes {
    pub const fn total(self) -> u64 {
        self.primary + self.stage2_view
    }
}

pub fn balanced_expert_range(experts: u32, ranks: u32, rank: u32) -> core::ops::Range<u32> {
    assert!(ranks > 0 && rank < ranks && experts >= ranks);
    let q = experts / ranks;
    let rem = experts % ranks;
    let begin = rank * q + rank.min(rem);
    let count = q + u32::from(rank < rem);
    begin..begin + count
}

pub fn expert_owner(experts: u32, ranks: u32, expert: u32) -> u32 {
    assert!(ranks > 0 && experts >= ranks && expert < experts);
    let q = experts / ranks;
    let rem = experts % ranks;
    let wide = (q + 1) * rem;
    if expert < wide {
        expert / (q + 1)
    } else {
        rem + (expert - wide) / q
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeEpGeometry {
    pub tokens: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub experts: u32,
    pub top_k: u32,
    pub ranks: u32,
    pub block_m: u32,
    pub element_bytes: u32,
}

impl MoeEpGeometry {
    pub fn validate(self) -> Result<(), &'static str> {
        if self.tokens == 0 || self.hidden == 0 || self.intermediate == 0 {
            return Err("tensor dimensions must be non-zero");
        }
        if self.ranks == 0 || self.experts < self.ranks {
            return Err("experts must cover every rank");
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return Err("top_k must be in 1..=experts");
        }
        if self.block_m == 0 || self.element_bytes == 0 {
            return Err("block_m and element_bytes must be non-zero");
        }
        Ok(())
    }

    /// Contiguous, balanced ownership.  It handles non-divisible expert counts
    /// without a model-specific table: earlier ranks own one extra expert.
    pub fn expert_range(self, rank: u32) -> core::ops::Range<u32> {
        balanced_expert_range(self.experts, self.ranks, rank)
    }

    pub fn owner(self, expert: u32) -> u32 {
        expert_owner(self.experts, self.ranks, expert)
    }
}

/// Static packet descriptor.  Offsets are bytes into peer-visible slabs; the
/// runtime may materialize only the windows used by the current route table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeEpBoundaryDesc {
    pub abi_version: u32,
    pub flags: u32,
    pub tokens: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub experts: u32,
    pub top_k: u32,
    pub ranks: u32,
    pub rank: u32,
    pub expert_begin: u32,
    pub expert_end: u32,
    pub route_entry_bytes: u32,
    pub activation_element_bytes: u32,
    pub output_element_bytes: u32,
    pub reserved: [u32; 2],
}

impl MoeEpBoundaryDesc {
    pub fn new(geometry: MoeEpGeometry, rank: u32, replicated: bool) -> Self {
        geometry
            .validate()
            .expect("invalid expert-parallel geometry");
        let owned = geometry.expert_range(rank);
        Self {
            abi_version: MOE_EP_ABI_VERSION,
            flags: (if replicated {
                MOE_EP_INPUT_REPLICATED
            } else {
                0
            }) | MOE_EP_FIXED_SLOT_COMBINE,
            tokens: geometry.tokens,
            hidden: geometry.hidden,
            intermediate: geometry.intermediate,
            experts: geometry.experts,
            top_k: geometry.top_k,
            ranks: geometry.ranks,
            rank,
            expert_begin: owned.start,
            expert_end: owned.end,
            route_entry_bytes: core::mem::size_of::<MoeEpRouteRecord>() as u32,
            activation_element_bytes: geometry.element_bytes,
            output_element_bytes: geometry.element_bytes,
            reserved: [0; 2],
        }
    }
}

/// Compacted route metadata. `token` and `slot` preserve the original
/// token-major fixed-slot combine order across transport and expert sorting.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoeEpRouteRecord {
    pub token: u32,
    pub slot: u32,
    pub expert: u32,
    pub gate: f32,
}

/// One peer's compacted window. `activation_rows` counts unique tokens, while
/// `route_entries` counts selected experts; this distinction prevents top-k
/// activation amplification in the all-to-all path.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeEpPeerWindow {
    pub peer: u32,
    pub activation_rows: u32,
    pub route_entries: u32,
    pub activation_offset: u64,
    pub route_offset: u64,
    pub return_offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeEpTransferPlan {
    pub source_rank: u32,
    pub windows: Vec<MoeEpPeerWindow>,
    pub activation_bytes: u64,
    pub route_bytes: u64,
    pub return_bytes: u64,
    pub local_bytes: u64,
}

/// Build the exact source-rank send plan for a token-sharded input.  Routes are
/// token-major and retain their original slot index; consumers can therefore
/// return partials in deterministic `(token, slot)` order.
pub fn token_sharded_transfer_plan(
    geometry: MoeEpGeometry,
    source_rank: u32,
    token_source: &[u32],
    routes: &[RouteEntry],
) -> MoeEpTransferPlan {
    geometry
        .validate()
        .expect("invalid expert-parallel geometry");
    assert!(source_rank < geometry.ranks);
    assert_eq!(token_source.len(), geometry.tokens as usize);
    assert!(token_source.iter().all(|&rank| rank < geometry.ranks));
    assert_eq!(
        routes.len(),
        geometry.tokens as usize * geometry.top_k as usize
    );

    let mut activation_rows = vec![0u32; geometry.ranks as usize];
    let mut route_counts = vec![0u32; geometry.ranks as usize];
    let mut seen_at = vec![u32::MAX; geometry.ranks as usize];
    for token in 0..geometry.tokens {
        if token_source[token as usize] != source_rank {
            continue;
        }
        let row = token as usize * geometry.top_k as usize;
        for route in &routes[row..row + geometry.top_k as usize] {
            if route.expert_id >= geometry.experts {
                continue;
            }
            let peer = geometry.owner(route.expert_id) as usize;
            if seen_at[peer] != token {
                seen_at[peer] = token;
                activation_rows[peer] += 1;
            }
            route_counts[peer] += 1;
        }
    }

    let row_bytes = geometry.hidden as u64 * geometry.element_bytes as u64;
    let mut activation_offset = 0u64;
    let mut route_offset = 0u64;
    let mut return_offset = 0u64;
    let mut windows = Vec::with_capacity(geometry.ranks as usize);
    for peer in 0..geometry.ranks {
        let activation_rows = activation_rows[peer as usize];
        let route_entries = route_counts[peer as usize];
        let activation_len = activation_rows as u64 * row_bytes;
        let route_len = route_entries as u64 * core::mem::size_of::<MoeEpRouteRecord>() as u64;
        // The destination combines all of its experts for a token before return.
        let return_len = activation_len;
        windows.push(MoeEpPeerWindow {
            peer,
            activation_rows,
            route_entries,
            activation_offset,
            route_offset,
            return_offset,
        });
        activation_offset += activation_len;
        route_offset += route_len;
        return_offset += return_len;
    }
    let local = windows[source_rank as usize];
    let local_bytes = local.activation_rows as u64 * row_bytes * 2
        + local.route_entries as u64 * core::mem::size_of::<MoeEpRouteRecord>() as u64;
    MoeEpTransferPlan {
        source_rank,
        windows,
        activation_bytes: activation_offset,
        route_bytes: route_offset,
        return_bytes: return_offset,
        local_bytes,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeEpCost {
    pub current_ms: f64,
    pub required_target_ms: f64,
    pub fabric_bytes_per_rank: u64,
    pub fabric_ms: f64,
    pub remaining_compute_sort_ms: f64,
}

/// Cost the boundary without assuming a kernel speedup.  The result states the
/// compute+sort budget that an isolated EP implementation must beat after its
/// measured transport charge. `fabric_gb_s` is per-rank bidirectional effective
/// bandwidth from the same-rank-placement P2P gate.
pub fn cost_boundary(
    current_ms: f64,
    layers: u32,
    plan: &MoeEpTransferPlan,
    fabric_gb_s: f64,
) -> MoeEpCost {
    assert!(current_ms > 0.0 && layers > 0 && fabric_gb_s > 0.0);
    let total = plan.activation_bytes + plan.route_bytes + plan.return_bytes;
    let fabric_bytes_per_rank = total.saturating_sub(plan.local_bytes);
    let fabric_ms = fabric_bytes_per_rank as f64 / (fabric_gb_s * 1.0e6);
    let required_target_ms = (current_ms * 0.85).max(current_ms - 35.0);
    MoeEpCost {
        current_ms,
        required_target_ms,
        fabric_bytes_per_rank,
        fabric_ms: fabric_ms * layers as f64,
        remaining_compute_sort_ms: required_target_ms - fabric_ms * layers as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> MoeEpGeometry {
        MoeEpGeometry {
            tokens: 16,
            hidden: 8,
            intermediate: 32,
            experts: 14,
            top_k: 4,
            ranks: 3,
            block_m: 64,
            element_bytes: 2,
        }
    }

    #[test]
    fn ownership_is_total_balanced_and_invertible() {
        let g = geometry();
        let ranges: Vec<_> = (0..g.ranks).map(|r| g.expert_range(r)).collect();
        assert_eq!(ranges, vec![0..5, 5..10, 10..14]);
        for r in 0..g.ranks {
            for e in ranges[r as usize].clone() {
                assert_eq!(g.owner(e), r);
            }
        }
    }

    #[test]
    fn dispatch_deduplicates_activation_per_token_and_owner() {
        let g = geometry();
        let mut routes = Vec::new();
        for _ in 0..g.tokens {
            // Two experts on rank 0, one each on ranks 1 and 2.
            for expert_id in [0, 1, 5, 10] {
                routes.push(RouteEntry {
                    expert_id,
                    gate: 0.25,
                });
            }
        }
        let source: Vec<u32> = (0..g.tokens).map(|t| t % g.ranks).collect();
        let p = token_sharded_transfer_plan(g, 0, &source, &routes);
        let source_tokens = source.iter().filter(|&&rank| rank == 0).count() as u32;
        assert_eq!(p.windows[0].activation_rows, source_tokens);
        assert_eq!(p.windows[0].route_entries, source_tokens * 2);
        assert_eq!(p.windows[1].activation_rows, source_tokens);
        assert_eq!(p.windows[2].activation_rows, source_tokens);
        assert_eq!(
            p.activation_bytes,
            source_tokens as u64 * 3 * g.hidden as u64 * 2
        );
        assert_eq!(p.return_bytes, p.activation_bytes);
    }

    #[test]
    fn descriptor_is_stable_and_model_agnostic() {
        let d = MoeEpBoundaryDesc::new(geometry(), 2, true);
        assert_eq!(d.abi_version, MOE_EP_ABI_VERSION);
        assert_eq!((d.expert_begin, d.expert_end), (10, 14));
        assert_ne!(d.flags & MOE_EP_INPUT_REPLICATED, 0);
        assert_eq!(core::mem::size_of::<MoeEpBoundaryDesc>(), 64);
        assert_eq!(core::mem::size_of::<MoeEpPeerWindow>(), 40);
        assert_eq!(core::mem::size_of::<MoeEpRouteRecord>(), 16);
    }

    #[test]
    fn two_dimensional_layout_is_single_resident_for_every_factorization() {
        let mut totals = Vec::new();
        for expert_degree in [1, 2, 4, 8] {
            let layout = Moe2dLayout {
                world_size: 8,
                expert_degree,
            };
            assert_eq!(layout.local_intermediate(3072), 384 * expert_degree);
            let resident = layout
                .mxfp4_resident_bytes(896, 3584, 3072, 92, true)
                .unwrap();
            assert_eq!(resident.primary, 180_807_008_256);
            assert_eq!(
                resident.stage2_view,
                if expert_degree == 1 {
                    61_450_747_904
                } else {
                    60_269_002_752
                }
            );
            totals.push(resident.total());
        }
        assert!(totals[1..].windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(totals[1], 241_076_011_008);
    }

    #[test]
    fn two_dimensional_rank_mapping_repeats_owners_across_intra_tp() {
        let layout = Moe2dLayout {
            world_size: 8,
            expert_degree: 4,
        };
        assert_eq!(layout.intra_expert_tp(), 2);
        assert_eq!(layout.expert_range(896, 0), 0..224);
        assert_eq!(layout.expert_range(896, 1), 0..224);
        assert_eq!(layout.expert_range(896, 2), 224..448);
        assert_eq!(layout.intra_rank(0), 0);
        assert_eq!(layout.intra_rank(1), 1);
    }
}
