use super::{
    packed_segment_route, prefill_segment_specialization_allowed, AmdEngine, HsaKernel,
    PackedSegmentRoute, PrefillSegmentRoute, WG_THREADS_4, WG_THREADS_8,
};
use crate::{Result, RuntimeError};

pub(super) fn small_mla_segments(prog: &super::DevProg, segments: usize) -> Vec<bool> {
    use packet::dev::{DevOp, TENSOR_NONE16};
    let mut pure = vec![true; segments];
    let mut any = vec![false; segments];
    if prog.packed_prefill_only || prog.t == 0 || prog.t >= 2048 {
        return any;
    }
    for entry in &prog.stream {
        let inst = &prog.insts[entry.inst as usize];
        let eligible = inst.i[3] & (1 << 31) == 0
            && if inst.op == DevOp::FlashMlaPrefill as u16 {
                inst.t[7] == TENSOR_NONE16 && inst.i[6] <= 1
            } else if inst.op == DevOp::FlashMlaPrefillFp8 as u16 {
                inst.fj[1] == 0 && inst.fj[2] == 0
            } else {
                false
            };
        pure[entry.seg as usize] &= eligible;
        any[entry.seg as usize] = true;
    }
    for (pure, any) in pure.iter_mut().zip(any) {
        *pure &= any;
    }
    pure
}

pub(super) fn load_small_mla(
    be: &super::HsaBackend,
    dir: &std::path::Path,
    blob_path: &std::path::Path,
    arch: &str,
    sched: super::Sched,
    fp8: bool,
    split: bool,
    modules: &mut Vec<crate::device::Module>,
) -> Result<HsaKernel> {
    use super::object::{
        check_interpreter_waves, check_packet_pairing_stamp, elf_symbol_names, elf_symbol_u32,
        FP8_KV_SYM, L2_DISPATCH_SYM,
    };
    use crate::exec::device_api::EngineDevice;
    use crate::RuntimeError;
    let infix = if fp8 { "_fp8kv" } else { "" };
    let (family, marker, phase) = if split {
        ("split", "plow_mla_prefill_fp8_split_1", super::Phase::Flash)
    } else {
        ("small", "plow_mla_prefill_small_1", super::Phase::Prefill)
    };
    let path = dir.join(format!("interp_mla_{family}{infix}{}.elf", sched.suffix()));
    let image = std::fs::read(&path).map_err(|e| {
        RuntimeError::Device(format!(
            "small MLA segments require {}: {e}",
            path.display()
        ))
    })?;
    let syms = elf_symbol_names(&image);
    if !syms.contains(&marker)
        || !syms.contains(&L2_DISPATCH_SYM)
        || syms.contains(&FP8_KV_SYM) != fp8
    {
        return Err(RuntimeError::Device(format!(
            "{}: wrong small MLA family, KV encoding or L2 dispatch contract",
            path.display()
        )));
    }
    check_interpreter_waves(
        elf_symbol_u32(&image, "plow_geom_PLOW_WG_WAVES"),
        phase,
        &path,
    )?;
    check_packet_pairing_stamp(&image, blob_path, &path)?;
    let module = EngineDevice::module_load(be, &image)?;
    let symbol = format!("plow_interp_mla_{family}_{arch}{}", sched.suffix());
    let result = (|| {
        let kernel = EngineDevice::get_function(be, &module, &symbol)?;
        let want = (std::mem::size_of::<super::DevProgram>() as u32 + 7) & !7;
        if kernel.kernarg_size() != want && kernel.kernarg_size() != want + 256 {
            return Err(RuntimeError::Device(format!(
                "{}: wrong interpreter kernarg size",
                path.display()
            )));
        }
        Ok(kernel)
    })();
    match result {
        Ok(kernel) => {
            modules.push(module);
            Ok(kernel)
        }
        Err(error) => {
            if let Err(unload) = EngineDevice::module_unload(be, &module) {
                tracing::warn!(%unload, object = %path.display(), "failed to unload rejected small MLA object");
            }
            Err(error)
        }
    }
}

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
            // The widest decode rung's object: the band is exactly that wide.
            PackedSegmentRoute::Band => (self.k_decode, WG_THREADS_8, "token_batch_band"),
            PackedSegmentRoute::Primary => {
                if self.progs[p].small_mla_split_segments[seg] {
                    if active {
                        return Err(RuntimeError::Device(
                            "dense MLA split segment cannot consume packed request metadata".into(),
                        ));
                    }
                    let kernel = self.k_mla_split.ok_or_else(|| {
                        crate::RuntimeError::Device(
                            "small MLA split segment has no matching object".into(),
                        )
                    })?;
                    return Ok((kernel, WG_THREADS_4, "mla_split_interpreter"));
                }
                if !active && self.progs[p].small_mla_segment[seg] {
                    if let Some(kernel) = self.k_mla_small {
                        return Ok((kernel, WG_THREADS_8, "mla_small_interpreter"));
                    }
                }
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
