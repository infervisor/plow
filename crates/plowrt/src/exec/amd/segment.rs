use super::{
    packed_segment_route, prefill_segment_specialization_allowed, AmdEngine, HsaKernel,
    PackedSegmentRoute, PrefillSegmentRoute, WG_THREADS_4, WG_THREADS_8,
};
use crate::Result;

impl AmdEngine {
    /// Diagnostic label for the kernel object that will execute one prefill segment.
    /// Kept out of the launch path unless segment timing is explicitly enabled.
    pub(crate) fn prefill_segment_family(&self, p: usize, seg: usize) -> &'static str {
        let route = self.progs[p]
            .prefill_routes
            .get(seg)
            .copied()
            .unwrap_or(PrefillSegmentRoute::Interpreter);
        match route {
            PrefillSegmentRoute::XReduceAttnRes { .. } => return "xreduce_attnres",
            PrefillSegmentRoute::XReduceWaveRs => return "xreduce_wave_rs",
            PrefillSegmentRoute::GraphPhaseXReduceWaveRs => return "graph_phase_xreduce_wave_rs",
            _ => {}
        }
        if !prefill_segment_specialization_allowed(self.prog_dispatch(p)) {
            return "interpreter";
        }
        let active = self.packed_prefill.is_some_and(|b| b.prog == p);
        if !active {
            match route {
                PrefillSegmentRoute::SparseMla(route) if route.active => return "mla_sparse_aiter",
                PrefillSegmentRoute::MoeAiter(_) => return "moe_aiter_fp8",
                PrefillSegmentRoute::IndexTp(_) => return "index_tp",
                PrefillSegmentRoute::GemmLt(_) => return "gemm_lt",
                PrefillSegmentRoute::MlaFold(_) => return "mla_fold",
                PrefillSegmentRoute::MlaMaterializePack { .. } => return "mla_materialize_pack",
                PrefillSegmentRoute::MlaMaterializedPrefill { .. } => {
                    return "mla_materialized_prefill";
                }
                PrefillSegmentRoute::KdaChunkIntraCached { .. }
                    if self.k_kda_chunk_intra_cached.is_some() =>
                {
                    return "kda_intra_cached";
                }
                PrefillSegmentRoute::KdaChunkIntraWaveItems { .. } => {
                    return "kda_intra_wave_items";
                }
                PrefillSegmentRoute::AttnResF32Mix { .. } => return "attn_res_f32mix",
                PrefillSegmentRoute::KdaChunkCarryRegstate { .. } => {
                    return "kda_carry_regstate";
                }
                PrefillSegmentRoute::KdaChunkWuLean { args, .. } => {
                    return if args.key_hi != 0 {
                        "kda_wu_lean_keys"
                    } else {
                        "kda_wu_lean"
                    };
                }
                PrefillSegmentRoute::KdaChunkCarryKeyfeed { .. } => {
                    return "kda_carry_keyfeed";
                }
                PrefillSegmentRoute::KdaChunkKeyFactorWu { .. }
                    if self.k_kda_key_factor_wu.is_some() =>
                {
                    return "kda_key_factor_wu";
                }
                PrefillSegmentRoute::KdaChunkKeyFactorCarry { .. }
                    if self.k_kda_key_factor_carry.is_some() =>
                {
                    return "kda_key_factor_carry";
                }
                PrefillSegmentRoute::MoeStage1Mxfp4(_) if self.k_moe_stage1_mxfp4.is_some() => {
                    return "moe_stage1_mxfp4";
                }
                PrefillSegmentRoute::MoeStage1A4Reuse(_)
                    if self.k_moe_stage1_a4_reuse.is_some() =>
                {
                    return "moe_stage1_a4_reuse";
                }
                PrefillSegmentRoute::MoeStage2Mxfp4(_) if self.k_moe_stage2_mxfp4.is_some() => {
                    return "moe_stage2_mxfp4";
                }
                PrefillSegmentRoute::MoeCombine(_) if self.k_moe_combine.is_some() => {
                    return "moe_combine";
                }
                PrefillSegmentRoute::MoeEpAlign(_) if self.k_moe_ep_align.is_some() => {
                    return "moe_ep_align";
                }
                PrefillSegmentRoute::MoeEpStage2(_) if self.k_moe_ep_stage2.is_some() => {
                    return "moe_ep_stage2";
                }
                PrefillSegmentRoute::MoeEpCombine(_) if self.k_moe_ep_combine.is_some() => {
                    return "moe_ep_combine";
                }
                _ => {}
            }
        }
        self.prefill_interpreter_kernel(p, seg)
            .map(|(_, _, family)| family)
            .unwrap_or("invalid_interpreter")
    }

    pub(super) fn prefill_interpreter_kernel(
        &self,
        p: usize,
        seg: usize,
    ) -> Result<(HsaKernel, u32, &'static str)> {
        let active = self.packed_prefill.is_some_and(|b| b.prog == p);
        let family = self.progs[p].packed_seg_family[seg];
        let route = packed_segment_route(
            active && !self.progs[p].packed_dense,
            family,
            self.k_packed_mla_norm.is_some(),
            self.k_packed_mla_flash.is_some(),
            self.k_packed_kda.is_some(),
        )?;
        Ok(match route {
            PackedSegmentRoute::MlaNorm => (
                self.k_packed_mla_norm.unwrap(),
                WG_THREADS_8,
                "packed_mla_norm",
            ),
            PackedSegmentRoute::MlaFlash => (
                self.k_packed_mla_flash.unwrap(),
                WG_THREADS_4,
                "packed_mla_flash",
            ),
            PackedSegmentRoute::Kda => (self.k_packed_kda.unwrap(), WG_THREADS_8, "kda_family_raw"),
            PackedSegmentRoute::Primary => {
                let raw_mla =
                    self.progs[p].raw_mla_v2_segment[seg] && self.k_mla_v2_sv_raw.is_some();
                if let (true, Some(k)) = (raw_mla, self.k_mla_v2_sv_raw) {
                    (k, WG_THREADS_4, "mla_v2_raw")
                } else {
                    let use4 = self.k_flash.is_some() && self.progs[p].seg_class[seg] == 4;
                    match (use4, self.k_flash) {
                        (true, Some(kf)) => (kf, WG_THREADS_4, "flash_interpreter"),
                        _ => (self.k_prefill, WG_THREADS_8, "interpreter"),
                    }
                }
            }
        })
    }
}
