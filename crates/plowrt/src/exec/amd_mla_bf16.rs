use std::{cell::Cell, path::Path};

use packet::dev::{DevOp, SE_XCTR, TENSOR_NONE16};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

const STAGE: &str = "mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co";
const METADATA: &str = "mla_metadata_gfx950.elf";
const REDUCE: &str = "mla_reduce_gfx950.elf";
const ADAPTER: &str = "mla_sparse_adapter_gfx950.elf";

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase { Pack, Metadata, Stage, Reduce, Unpad }

const PHASES: [Phase; 5] = [Phase::Pack, Phase::Metadata, Phase::Stage, Phase::Reduce, Phase::Unpad];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(super) struct DispatchSpec {
    pub phase: Phase,
    pub grid: [u32; 3],
    pub threads: u16,
    pub dynamic_lds: u32,
    pub explicit_bytes: u32,
    pub kernarg_bytes: u32,
    pub static_lds: u32,
    pub private_bytes: u32,
}

impl Phase {
    #[inline(always)]
    fn spec(self, rows: u32) -> DispatchSpec {
        let (grid, threads, dynamic_lds, explicit_bytes, kernarg_bytes, static_lds) = match self {
            Self::Pack => ([256, 1, 1], 256, 0, 112, 368, 260),
            Self::Metadata => ([1, 1, 1], 512, 163840, 136, 392, 0),
            Self::Stage => ([256, 1, 1], 256, 0, 384, 384, 163840),
            Self::Reduce => ([16, 1, rows], 128, 2048, 84, 84, 0),
            Self::Unpad => ([16, 1, 1], 256, 0, 24, 280, 0),
        };
        DispatchSpec { phase: self, grid, threads, dynamic_lds, explicit_bytes, kernarg_bytes,
            static_lds, private_bytes: 0 }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct NativeRouteContract {
    pub rows: u32,
    pub context_capacity: u32,
    pub heads: u32,
    pub latent: u32,
    pub rope: u32,
    pub selected_keys: Option<u32>,
    pub min_live: u32,
    pub inactive_rows: bool,
    pub output_handle: u16,
    pub output_head_stride: u32,
    pub metadata_hoist_enabled: bool,
    pub reuse_metadata: bool,
    pub condition: &'static str,
    pub rows_capacity: u32,
    pub scratch_split_capacity: u32,
    pub split_hint_rule: &'static str,
    pub caller_source_sha256: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct NativeLaunchEvidence {
    pub object: crate::device::hsa::SelectedKernelEvidence,
    pub dispatch: DispatchSpec,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct NativeVariantEvidence {
    pub contract: NativeRouteContract,
    pub launches: Vec<NativeLaunchEvidence>,
}

fn invalid(message: &str) -> RuntimeError {
    RuntimeError::Device(format!("BF16 persistent MLA: {message}"))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    handles: [u16; 7],
    rows: u32,
    ctx: u32,
    max_live: u32,
    metadata_key: MetadataKey,
    reuse_metadata: bool,
    padded_output: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MetadataKey {
    selected: [u16; 31],
    rows: u32,
}

impl Route {
    pub fn arm(&mut self, lengths: &[u32]) {
        self.reuse_metadata = false;
        self.metadata_key = MetadataKey::default();
        self.max_live = lengths
            .get(..self.rows as usize)
            .filter(|v| v.iter().all(|&n| n > 0 && n <= self.ctx))
            .and_then(|v| v.iter().max().copied())
            .unwrap_or(0);
        if self.max_live != 0 {
            self.metadata_key.rows = self.rows;
            for (dst, &length) in self.metadata_key.selected.iter_mut().zip(lengths)
                .take(self.rows as usize) {
                *dst = length.min(2048) as u16;
            }
        }
    }

    pub fn plan_metadata(&mut self, previous: &mut Option<MetadataKey>, enabled: bool) {
        self.reuse_metadata = enabled && self.max_live != 0 && *previous == Some(self.metadata_key);
        *previous = (self.max_live != 0).then_some(self.metadata_key);
    }

    pub fn launches(self) -> usize {
        3 + usize::from(self.includes(Phase::Metadata)) + usize::from(self.includes(Phase::Unpad))
    }

    #[inline(always)]
    fn includes(self, phase: Phase) -> bool {
        match phase {
            Phase::Metadata => !self.reuse_metadata,
            Phase::Unpad => !self.padded_output,
            _ => true,
        }
    }

    fn evidence(
        self, rows_capacity: u32, hoist: bool,
        mut resolve: impl FnMut(Phase) -> Result<crate::device::hsa::SelectedKernelEvidence>,
    ) -> Result<Vec<NativeVariantEvidence>> {
        if !(1..32).contains(&self.rows) || !(1..32).contains(&rows_capacity) || self.rows > rows_capacity {
            return Err(invalid("evidence outside allocated row domain"));
        }
        let mut variants = Vec::new();
        for reuse in [false, true].into_iter().filter(|reuse| !reuse || hoist) {
            let planned = Route { reuse_metadata: reuse, ..self };
            let mut launches = Vec::new();
            for phase in PHASES.into_iter().filter(|&phase| planned.includes(phase)) {
                let dispatch = phase.spec(self.rows);
                let object = resolve(phase)?;
                if object.symbol.kernarg_bytes != dispatch.kernarg_bytes
                    || object.symbol.static_lds_bytes != dispatch.static_lds
                    || object.symbol.private_bytes_per_workitem != dispatch.private_bytes {
                    return Err(invalid("selected native object differs from caller ABI"));
                }
                launches.push(NativeLaunchEvidence { object, dispatch });
            }
            debug_assert_eq!(launches.len(), planned.launches());
            variants.push(NativeVariantEvidence {
                contract: NativeRouteContract {
                    rows: self.rows, context_capacity: self.ctx, heads: 8, latent: 512, rope: 64,
                    selected_keys: (self.handles[6] != TENSOR_NONE16).then_some(2048),
                    min_live: 1, inactive_rows: false, output_handle: self.handles[0],
                    output_head_stride: if self.padded_output { 1024 } else { 512 },
                    metadata_hoist_enabled: hoist, reuse_metadata: reuse,
                    condition: if reuse { "positive live rows plus exact prior metadata key in this replay/allocation" }
                        else { "positive live rows; build metadata before stage" },
                    rows_capacity, scratch_split_capacity: 256 + rows_capacity - 1,
                    split_hint_rule: "min(256, rows * next_pow2(max(1, min(max_live,2048)/128))); actual ragged splits are GPU-produced, not this hint",
                    caller_source_sha256: plow_asset::decode_objects::image_sha256(include_bytes!("amd_mla_bf16.rs")),
                },
                launches,
            });
        }
        Ok(variants)
    }
}

pub(super) fn pair_index(prog: &DevProg, seg: usize) -> Result<Option<[usize; 2]>> {
    let mut pair = [usize::MAX; 2];
    for e in prog
        .stream
        .iter()
        .chain(&prog.gq_stream)
        .filter(|e| e.seg as usize == seg)
    {
        let Some(inst) = prog.insts.get(e.inst as usize) else {
            return Err(invalid("instruction index"));
        };
        let slot = match DevOp::from_u16(inst.op) {
            Some(DevOp::FlashMlaDecode | DevOp::FlashGatherDecode) => 0,
            Some(DevOp::FlashMerge) => 1,
            _ => return Ok(None),
        };
        if pair[slot] != usize::MAX && pair[slot] != e.inst as usize {
            return Ok(None);
        }
        pair[slot] = e.inst as usize;
    }
    if pair.contains(&usize::MAX) {
        return Ok(None);
    }
    if pair[1] != pair[0] + 1 {
        return Err(invalid("nonadjacent flash/merge"));
    }
    for e in prog
        .stream
        .iter()
        .chain(&prog.gq_stream)
        .filter(|e| pair.contains(&(e.inst as usize)))
    {
        if e.seg as usize != seg || e.wait_len != 0 || e.succ_len != 0 || e.flags & SE_XCTR != 0 {
            return Err(invalid(
                "raw pair has counter obligations or multiple owners",
            ));
        }
    }
    Ok(Some(pair))
}

pub(super) fn route(prog: &DevProg, tensors: &[DevTensor], seg: usize) -> Result<Option<Route>> {
    let Some(pair) = pair_index(prog, seg)? else {
        return Ok(None);
    };
    let f = &prog.insts[pair[0]];
    let merge = &prog.insts[pair[1]];
    let sparse = f.op == DevOp::FlashGatherDecode as u16;
    if !(1..32).contains(&prog.t)
        || f.i[0] != prog.t
        || f.i[1] != 8
        || !(1..=131072).contains(&f.i[2])
        || f.i[3] != 0
        || f.i[4] == 0
        || f.i[5] != u32::MAX
        || f.fj != [0.0625f32.to_bits(), 0, 0]
        || (sparse && (f.i[6] != 2048 || f.t[7] == TENSOR_NONE16))
        || (!sparse && (f.i[2] > 2048 || f.t[7] != TENSOR_NONE16 || f.i[6] != 0))
        || merge.i[..4] != [prog.t, 8, f.i[4], 512]
        || merge.t[1..3] != f.t[..2]
        || !matches!(merge.i[4], 0 | 1024)
        || merge.i[5..].iter().any(|&v| v != 0)
    {
        return Err(invalid(
            "requires QH8 latent512/rope64, BF16, top2048 or short dense, M<32",
        ));
    }
    let handles = [merge.t[0], f.t[2], f.t[3], f.t[4], f.t[5], f.t[6], f.t[7]];
    let padded_output = merge.i[4] == 1024;
    if !padded_output && prog.insts.iter().any(|inst|
        inst.op == DevOp::MlaBmmFp8 as u16 && inst.t[1] == handles[0] && inst.i[5] != 0)
    {
        return Err(invalid("strided WV input requires a padded producer"));
    }
    if padded_output {
        let consumer_idx = pair[1] + 1;
        let consumer = prog.insts.get(consumer_idx)
            .ok_or_else(|| invalid("padded output has no WV consumer"))?;
        if consumer.op != DevOp::MlaBmmFp8 as u16
            || consumer.i != [prog.t, 8, 256, 512, 0, 1024, 0, 0]
            || consumer.t[1] != handles[0]
            || consumer.t.iter().enumerate().any(|(slot, &handle)| slot != 1 && handle == handles[0])
            || prog.insts.iter().enumerate().any(|(i, inst)|
                i != pair[1] && i != consumer_idx && may_reference(inst, handles[0]))
        {
            return Err(invalid("padded output requires one exact stride-aware WV consumer"));
        }
    }
    let rows = u64::from(prog.t);
    let sizes = [
        rows * if padded_output { 16 } else { 8 } * 512 * 2,
        rows * 8 * 512 * 2,
        rows * 8 * 64 * 2,
        rows * u64::from(f.i[2]) * 512 * 2,
        rows * u64::from(f.i[2]) * 64 * 2,
        rows * 4,
        if sparse { rows * 2048 * 4 } else { 0 },
    ];
    for (&handle, bytes) in handles.iter().zip(sizes) {
        if bytes != 0
            && (handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes))
        {
            return Err(invalid("operand capacity"));
        }
    }
    if handles[1..7].contains(&handles[0]) {
        return Err(invalid("output aliases an input"));
    }
    Ok(Some(Route {
        handles,
        rows: prog.t,
        ctx: f.i[2],
        max_live: 0,
        metadata_key: MetadataKey::default(),
        reuse_metadata: false,
        padded_output,
    }))
}

fn may_reference(inst: &packet::dev::DevInst64, handle: u16) -> bool {
    if inst.t.contains(&handle) { return true; }
    let h = u32::from(handle);
    let extra: &[usize] = match DevOp::from_u16(inst.op) {
        Some(DevOp::GemvQkv) if h != 0 => &[5, 6, 7],
        Some(DevOp::GemvQkvg) => &[6],
        Some(DevOp::GemvQkvFp8 | DevOp::GemvQkvMxfp4) => &[5, 6, 7],
        Some(DevOp::GemvGluFp8) if inst.fj[2] != 0 => &[3, 4, 6, 7],
        Some(DevOp::AttnRes) => &[5, 6],
        Some(DevOp::FlashMlaDecode) if inst.t[7] != TENSOR_NONE16 => &[6],
        Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8) if inst.i[1] & 0x10000 != 0 => return true,
        Some(DevOp::MlaBmmFp8) if inst.i[4] != 0 => return inst.fj[1..].contains(&h),
        Some(DevOp::FlashMlaDecodeFp8 | DevOp::FlashMlaPrefillFp8) =>
            return inst.fj[1] != 0 && inst.fj[1] - 1 == h,
        // These carry additional state/weight pointers; not in the qualified GLM attention domain.
        Some(DevOp::KdaConv3 | DevOp::KdaStateStep | DevOp::KdaStateStepG
            | DevOp::KdaConvStateStepG | DevOp::KdaDecodeFused) | None => return true,
        _ => &[],
    };
    extra.iter().any(|&slot| inst.i[slot] == h)
}

fn load<'a>(
    be: &HsaBackend,
    dir: &Path,
    file: &str,
    hash: &str,
    modules: &'a mut Vec<Module>,
) -> Result<&'a Module> {
    let path = dir.join(file);
    let mut image =
        std::fs::read(&path).map_err(|e| invalid(&format!("{}: {e}", path.display())))?;
    if plow_asset::decode_objects::image_sha256(&image) != hash {
        return Err(invalid(&format!("unqualified object {file}")));
    }
    if file == STAGE {
        if image.get(0x1708..0x170c) != Some(&[0, 0, 0, 0]) {
            return Err(invalid("stage descriptor"));
        }
        image[0x1708..0x170c].copy_from_slice(&384u32.to_le_bytes());
    }
    let module = EngineDevice::module_load(be, &image)?;
    modules.push(module);
    Ok(modules.last().unwrap())
}

fn kernel(
    be: &HsaBackend,
    module: &Module,
    symbol: &str,
    phase: Phase,
) -> Result<HsaKernel> {
    let k = EngineDevice::get_function(be, module, symbol)?;
    let spec = phase.spec(1);
    if k.kernarg_size() != spec.kernarg_bytes || k.private_segment_size() != spec.private_bytes
        || k.group_segment_size() != spec.static_lds {
        return Err(invalid(&format!("resource ABI for {symbol}")));
    }
    Ok(k)
}

pub(super) struct PersistentMla {
    pack: HsaKernel,
    metadata: HsaKernel,
    stage: HsaKernel,
    reduce: HsaKernel,
    unpad: HsaKernel,
    _scratch: DeviceMem,
    ptr: [u64; 14],
    rows_cap: u32,
    metadata_key: Cell<Option<MetadataKey>>,
}

impl PersistentMla {
    #[inline(always)]
    fn phase_kernel(&self, phase: Phase) -> HsaKernel {
        match phase {
            Phase::Pack => self.pack, Phase::Metadata => self.metadata, Phase::Stage => self.stage,
            Phase::Reduce => self.reduce, Phase::Unpad => self.unpad,
        }
    }

    #[inline(always)]
    fn launch_phase(&self, be: &HsaBackend, route: Route, phase: Phase, args: &[u8]) -> Result<()> {
        let spec = phase.spec(route.rows);
        debug_assert_eq!(args.len(), spec.explicit_bytes as usize);
        match phase {
            Phase::Reduce => be.launch_3d_lds(self.phase_kernel(phase), spec.grid, spec.threads, spec.dynamic_lds, args),
            _ => be.launch(self.phase_kernel(phase), spec.grid[0], u32::from(spec.threads), spec.dynamic_lds, args),
        }
    }

    pub fn selected_evidence(&self, be: &HsaBackend, route: Route, hoist: bool) -> Result<Vec<NativeVariantEvidence>> {
        route.evidence(self.rows_cap, hoist, |phase| be.selected_kernel_evidence(self.phase_kernel(phase)))
    }

    pub fn load(be: &HsaBackend, dir: &Path, rows: u32, modules: &mut Vec<Module>) -> Result<Self> {
        if be.arch() != "gfx950"
            || be.sm_count() != 256
            || be.lds_bytes() != 163840
            || !(1..32).contains(&rows)
        {
            return Err(invalid("requires gfx950/256CU/160KiB LDS and M<32"));
        }
        let m = load(
            be,
            dir,
            STAGE,
            "b6d4181c3ed19750b22a02dc0d290727272ed53091678c5e75c39f45d9832cfd",
            modules,
        )?;
        let stage = kernel(
            be,
            &m,
            "_ZN5aiter41mla_a16w16_qh64_qseqlen1_gqaratio64_v3_psE",
            Phase::Stage,
        )?;
        let m = load(
            be,
            dir,
            ADAPTER,
            "ed1426279c1b8228334315570830fac9bd57d4c43adc3066466afcda0649a35b",
            modules,
        )?;
        let pack = kernel(be, &m, "plow_mla_bf16_pack", Phase::Pack)?;
        let unpad = kernel(be, &m, "plow_mla_bf16_unpad", Phase::Unpad)?;
        let m = load(
            be,
            dir,
            METADATA,
            "3ade36a825eb249bc7dd8168e2338119173c595f1a1a427622733b73d9bd4081",
            modules,
        )?;
        let metadata = kernel(be, &m, "_Z33kn_get_mla_metadata_v1_2_parallelI20MlaMetadataV12TraitsILi128ELb0ELi1ELb1ELb0EEEv28MlaMetadataV1KernelParameter", Phase::Metadata)?;
        let m = load(
            be,
            dir,
            REDUCE,
            "401e7dd9c9714650361a87bba36b216e5b491d90fa10e8fc9cda72712e62f383",
            modules,
        )?;
        let reduce = kernel(be, &m, "_Z16kn_mla_reduce_v1I23MlaReduceKernelV1TraitsILi512ELi16ELi1EEfDF16bEv23MlaReduceKernelV1Params24MlaReduceKernelV1Configs", Phase::Reduce)?;
        let r = u64::from(rows);
        let cap = 256 + r - 1;
        let sizes = [
            r * 16 * 576 * 2,
            r * 2048 * 576 * 2,
            r * 2048 * 4,
            (r + 1) * 4,
            (r + 1) * 4,
            r * 4,
            80,
            257 * 4,
            cap * 8 * 4,
            (r + 1) * 4,
            r * 2 * 4,
            cap * 4,
            cap * 16 * 513 * 4,
            r * 16 * 512 * 2,
        ];
        let scratch = EngineDevice::alloc(be, sizes.iter().map(|n| n.div_ceil(256) * 256).sum())?;
        let mut at = scratch.base;
        let ptr = sizes.map(|n| {
            let p = at;
            at += n.div_ceil(256) * 256;
            p
        });
        Ok(Self {
            pack,
            metadata,
            stage,
            reduce,
            unpad,
            _scratch: scratch,
            ptr,
            rows_cap: rows,
            metadata_key: Cell::new(None),
        })
    }

    pub fn reset_metadata(&self) {
        self.metadata_key.set(None);
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, table: &[u8]) -> Result<()> {
        if route.rows > self.rows_cap || route.max_live == 0 {
            return Err(invalid("unarmed or invalid live rows"));
        }
        if route.reuse_metadata && self.metadata_key.get() != Some(route.metadata_key) {
            return Err(invalid("metadata reuse precedes its producer or crosses a replay"));
        }
        let mut a = [0u64; 7];
        for (dst, &handle) in a.iter_mut().zip(&route.handles) {
            if handle == TENSOR_NONE16 {
                continue;
            }
            let offset = usize::from(handle) * 8;
            *dst = u64::from_le_bytes(
                table
                    .get(offset..offset + 8)
                    .ok_or_else(|| invalid("tensor table"))?
                    .try_into()
                    .unwrap(),
            );
        }
        let [q, kv, kvi, qp, kp, last, metadata, workptr, info, reduceptr, finalmap, partialmap, part, out] =
            self.ptr;
        let out = if route.padded_output { a[0] } else { out };
        let lse = part + u64::from(256 + self.rows_cap - 1) * 16 * 512 * 4;
        let r = route.rows;
        let mut pack = [
            q, kv, kvi, a[1], a[2], a[3], a[4], a[6], a[5], qp, kp, last, 0, 0,
        ];
        pack[12] = u64::from(r) | u64::from(route.ctx) << 32;
        pack[13] = if route.handles[6] == TENSOR_NONE16 {
            0
        } else {
            2048
        };
        self.launch_phase(be, route, Phase::Pack, bytemuck::cast_slice(&pack))?;
        let mut meta = [0u32; 34];
        for (slot, ptr) in meta.chunks_exact_mut(2).zip([
            metadata, workptr, info, reduceptr, finalmap, partialmap, qp, kp, last,
        ]) {
            slot[0] = ptr as u32;
            slot[1] = (ptr >> 32) as u32;
        }
        let splits = ((route.max_live.min(2048) / 128).max(1).next_power_of_two() * r).min(256);
        meta[18..31].copy_from_slice(&[r, 0, 16, 256, r + 1, 1, 16, 4, 1, 1, u32::MAX, 1, splits]);
        meta[31..].copy_from_slice(&[1, 16, 1]);
        if route.includes(Phase::Metadata) {
            self.metadata_key.set(None);
            self.launch_phase(be, route, Phase::Metadata, bytemuck::cast_slice(&meta))?;
            self.metadata_key.set(Some(route.metadata_key));
        }
        let mut stage = [0u64; 48];
        for (i, value) in [
            (0, part),
            (2, lse),
            (4, q),
            (6, kv),
            (8, kp),
            (10, kvi),
            (12, last),
            (14, u64::from(0.0625f32.to_bits())),
            (16, 16),
            (18, 1),
            (20, 16 * 576 * 2),
            (22, 576 * 2),
            (26, qp),
            (28, metadata),
            (30, out),
            (36, 1),
            (42, 1),
        ] {
            stage[i] = value;
        }
        self.launch_phase(be, route, Phase::Stage, bytemuck::cast_slice(&stage))?;
        let mut reduce = [0u32; 21];
        for (slot, ptr) in reduce
            .chunks_exact_mut(2)
            .zip([reduceptr, finalmap, partialmap, 0, out, lse, part])
        {
            slot[0] = ptr as u32;
            slot[1] = (ptr >> 32) as u32;
        }
        reduce[14..].copy_from_slice(&[16 * 512, 512, 256, r, 256, 0, 256]);
        self.launch_phase(be, route, Phase::Reduce, bytemuck::cast_slice(&reduce))?;
        if !route.includes(Phase::Unpad) {
            return Ok(());
        }
        self.launch_phase(be, route, Phase::Unpad, bytemuck::cast_slice(&[a[0], out, u64::from(r)]))
    }
}

#[cfg(test)]
pub(super) fn test_native_evidence(padded_output: bool, hoist: bool) -> Vec<NativeVariantEvidence> {
    let route = Route { handles: [0,1,2,3,4,5,6], rows: 16, ctx: 8192, max_live: 0,
        metadata_key: MetadataKey::default(), reuse_metadata: false, padded_output };
    route.evidence(16, hoist, |phase| {
        let spec = phase.spec(16);
        Ok(crate::device::hsa::SelectedKernelEvidence {
            object_sha256: "a".repeat(64), object_bytes: 4096,
            symbol: crate::device::hsa::ResolvedKernelEvidence {
                entry: format!("{phase:?}"), kernarg_bytes: spec.kernarg_bytes,
                static_lds_bytes: spec.static_lds, private_bytes_per_workitem: spec.private_bytes,
            },
        })
    }).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};

    #[test]
    fn native_evidence_uses_launch_predicates_and_loaded_abi_without_resource_defaults() {
        for padded in [false, true] {
            for hoist in [false, true] {
                let evidence = test_native_evidence(padded, hoist);
                assert_eq!(evidence.len(), if hoist { 2 } else { 1 });
                for variant in evidence {
                    let mut route = Route { handles: [0,1,2,3,4,5,6], rows: 16, ctx: 8192, max_live: 0,
                        metadata_key: MetadataKey::default(), reuse_metadata: false, padded_output: padded };
                    route.arm(&[512; 16]);
                    let mut previous = variant.contract.reuse_metadata.then_some(route.metadata_key);
                    route.plan_metadata(&mut previous, hoist);
                    assert_eq!(route.launches(), variant.launches.len());
                    let phases: Vec<_> = variant.launches.iter().map(|l| l.dispatch.phase).collect();
                    assert_eq!(phases.contains(&Phase::Metadata), !route.reuse_metadata);
                    assert_eq!(phases.contains(&Phase::Unpad), !padded);
                    assert_eq!(variant.contract.scratch_split_capacity, 271);
                    assert_eq!(variant.contract.min_live, 1);
                    assert!(!variant.contract.inactive_rows);
                    let reduce = variant.launches.iter().find(|l| l.dispatch.phase == Phase::Reduce).unwrap();
                    assert_eq!(reduce.dispatch.grid, [16, 1, 16]);
                    assert_eq!(reduce.dispatch.dynamic_lds, 2048);
                    let bad = route.evidence(16, hoist, |phase| {
                        let spec = phase.spec(16);
                        Ok(crate::device::hsa::SelectedKernelEvidence {
                            object_sha256: "a".repeat(64), object_bytes: 4096,
                            symbol: crate::device::hsa::ResolvedKernelEvidence {
                                entry: format!("{phase:?}"), kernarg_bytes: spec.kernarg_bytes + 8,
                                static_lds_bytes: spec.static_lds, private_bytes_per_workitem: 0,
                            },
                        })
                    });
                    assert!(bad.is_err());
                    assert!(route.evidence(15, hoist, |_| panic!("invalid capacity")).is_err());
                }
            }
        }
    }

    fn fixture(rows: u32, ctx: u32) -> (DevProg, Vec<DevTensor>) {
        let f = DevInst64 {
            op: DevOp::FlashGatherDecode as u16,
            t: [7, 8, 1, 2, 3, 4, 5, 6],
            i: [rows, 8, ctx, 0, 8, u32::MAX, 2048, 4],
            fj: [0.0625f32.to_bits(), 0, 0],
            ..Default::default()
        };
        let merge = DevInst64 {
            op: DevOp::FlashMerge as u16,
            t: [
                0,
                7,
                8,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            i: [rows, 8, 8, 512, 0, 0, 0, 0],
            ..Default::default()
        };
        (
            DevProg {
                t: rows,
                role: packet::devbuild::ProgramRole::DecodeRung { rows },
                n_counter: 0,
                insts: vec![f, merge],
                stream: vec![
                    StreamEnt {
                        inst: 0,
                        ..Default::default()
                    },
                    StreamEnt {
                        inst: 1,
                        ..Default::default()
                    },
                ],
                stream_ofs: vec![],
                stream_len: vec![],
                waits: vec![],
                succs: vec![],
                gq_stream: vec![],
                gq_seg_ofs: vec![],
                l2_domains: 0,
            },
            (0..9)
                .map(|i| DevTensor {
                    name: i.to_string(),
                    bytes: 4 << 30,
                    init: None,
                })
                .collect(),
        )
    }

    #[test]
    fn padded_output_requires_capacity_and_exclusive_strided_consumer() {
        let (mut p, mut tensors) = fixture(16, 8192);
        p.insts[1].i[4] = 1024;
        assert!(route(&p, &tensors, 0).is_err());
        p.insts.push(DevInst64 {
            op: DevOp::MlaBmmFp8 as u16,
            t: [9, 0, 10, 11, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
            i: [16, 8, 256, 512, 0, 1024, 0, 0],
            ..Default::default()
        });
        let r = route(&p, &tensors, 0).unwrap().unwrap();
        assert_eq!(r.launches(), 4);
        p.insts[1].i[4] = 0;
        assert!(route(&p, &tensors, 0).is_err());
        p.insts[1].i[4] = 1024;
        tensors[0].bytes = 16 * 8 * 512 * 2;
        assert!(route(&p, &tensors, 0).is_err());
        tensors[0].bytes *= 2;
        assert!(route(&p, &tensors, 0).is_ok());
        p.insts[2].i[5] = 0;
        assert!(route(&p, &tensors, 0).is_err());
        p.insts[2].i[5] = 1024;
        p.insts.push(p.insts[2]);
        assert!(route(&p, &tensors, 0).is_err());
        p.insts.pop();
        p.insts[2].t[2] = 0;
        assert!(route(&p, &tensors, 0).is_err());
        p.insts[2].t[2] = 10;
        let hidden = DevInst64 {
            op: DevOp::GemvQkvg as u16,
            t: [TENSOR_NONE16; 8],
            i: [16, 256, 512, 256, 256, 256, 0, 0],
            ..Default::default()
        };
        p.insts.push(hidden);
        assert!(route(&p, &tensors, 0).is_err(), "demoted weight handle aliases padded output");
        p.insts.pop();
        for op in [DevOp::AttnRes, DevOp::KdaStateStep, DevOp::KdaConvStateStepG,
            DevOp::KdaDecodeFused] {
            p.insts.push(DevInst64 { op: op as u16, t: [TENSOR_NONE16; 8],
                ..Default::default() });
            assert!(route(&p, &tensors, 0).is_err(), "hidden state/descriptor references: {op:?}");
            p.insts.pop();
        }
    }

    #[test]
    fn metadata_reuse_requires_exact_selected_lengths_and_replay_producer() {
        let (p, tensors) = fixture(8, 8192);
        let mut first = route(&p, &tensors, 0).unwrap().unwrap();
        first.arm(&[1, 16, 127, 128, 129, 512, 2048, 8192]);
        let mut next = first;
        let mut previous = None;
        first.plan_metadata(&mut previous, true);
        next.plan_metadata(&mut previous, true);
        assert_eq!((first.launches(), next.launches()), (5, 4));
        next.arm(&[1, 16, 127, 128, 129, 512, 2048, 4096]);
        next.plan_metadata(&mut previous, true);
        assert_eq!(next.launches(), 4);
        next.arm(&[1, 16, 127, 128, 130, 512, 2048, 4096]);
        next.plan_metadata(&mut previous, true);
        assert_eq!(next.launches(), 5);
        next.plan_metadata(&mut None, true);
        assert_eq!(next.launches(), 5);
        next.plan_metadata(&mut previous, false);
        assert_eq!(next.launches(), 5);
        next.arm(&[0; 8]);
        next.plan_metadata(&mut previous, true);
        assert_eq!(next.launches(), 5);
        assert_eq!(previous, None);
    }

    #[test]
    fn bf16_route_guards_geometry_counters_capacity_and_live_lengths() {
        for rows in [1, 8, 16, 31] {
            let (p, t) = fixture(rows, 71680);
            let mut r = route(&p, &t, 0).unwrap().unwrap();
            r.arm(&vec![512; rows as usize]);
            assert_eq!(r.max_live, 512);
            for invalid in [0, 71681] {
                r.arm(&vec![invalid; rows as usize]);
                assert_eq!(r.max_live, 0);
            }
        }
        for bad in 0..10 {
            let (mut p, mut t) = fixture(16, 8192);
            match bad {
                0 => p.t = 32,
                1 => p.insts[0].i[1] = 16,
                2 => p.insts[0].i[6] = 1024,
                3 => p.insts[0].fj[0] = 0.125f32.to_bits(),
                4 => p.insts[1].t[1] = 0,
                5 => t[3].bytes = 1,
                6 => p.stream[0].wait_len = 1,
                7 => {
                    p.gq_stream = p.stream.clone();
                    p.gq_stream[0].flags = SE_XCTR;
                }
                8 => {
                    p.gq_stream = p.stream.clone();
                    p.gq_stream[1].seg = 1;
                }
                9 => p.insts[1].t[0] = 1,
                _ => unreachable!(),
            }
            assert!(route(&p, &t, 0).is_err(), "case {bad}");
        }
        let (mut p, t) = fixture(8, 512);
        p.insts[0].op = DevOp::FlashMlaDecode as u16;
        p.insts[0].t[7] = TENSOR_NONE16;
        p.insts[0].i[6] = 0;
        assert!(route(&p, &t, 0).unwrap().is_some());
        p.insts[0].i[2] = 8192;
        assert!(route(&p, &t, 0).is_err());
        p.insts.push(DevInst64::default());
        p.stream.push(StreamEnt {
            inst: 2,
            ..Default::default()
        });
        assert!(route(&p, &t, 0).unwrap().is_none());
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with pinned objects/reference fixtures"]
    fn bf16_persistent_hsa_reference_replay() {
        replay(false);
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with stride-WV object"]
    fn bf16_persistent_strided_wv_replay() {
        replay(true);
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with captured actual-weight WV fixtures"]
    fn bf16_strided_wv_captured_weights() {
        captured_weights(false);
    }

    #[test]
    #[ignore = "requires queued gfx950 GPU and frozen captured attention/WV operands and reports"]
    fn bf16_attention_strided_wv_captured_chain() {
        captured_weights(true);
    }

    fn captured_weights(connected: bool) {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let dir = Path::new(&dir);
        let report_bytes = std::fs::read(dir.join("captured/mla-comparison.json")).unwrap();
        assert_eq!(plow_asset::decode_objects::image_sha256(&report_bytes),
            "31eeafdb8fa74e0a05e1cb355aef0eb22efb5dfdb5e52d8b4116aa2679f3acf8");
        let report: serde_json::Value = serde_json::from_slice(&report_bytes).unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(report["vllm_version"], "0.29.0");
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 8);
        let be = HsaBackend::new(0).unwrap();
        let image = std::fs::read(dir.join("test_kernels.elf")).unwrap();
        assert_eq!(plow_asset::decode_objects::image_sha256(&image),
            "652f9021da05451f5b06bdf3d44ccb043c317ff8e0d5d583ca186dc8ee7f2c94");
        let module = EngineDevice::module_load(&be, &image).unwrap();
        let kernel = EngineDevice::get_function(&be, &module, "mla_bmm_fp8_stride").unwrap();
        let mut modules = Vec::new();
        let attention = connected.then(|| PersistentMla::load(&be, dir, 31, &mut modules).unwrap());
        let attention_report = connected.then(|| {
            let bytes = std::fs::read(dir.join("captured/attention-comparison.json")).unwrap();
            assert_eq!(plow_asset::decode_objects::image_sha256(&bytes),
                "7636e6606b199edc6315aab6f7683446125a83a174878b449a4475ce02c48fbf");
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["passed"], true);
            assert_eq!(value["vllm_version"], "0.29.0");
            assert_eq!(value["shape"], serde_json::json!([16,512,8,512,64]));
            value
        });
        let mut previous_attention_inputs = None;
        for (rank, case) in cases.iter().enumerate() {
            assert_eq!(case["rank"], rank);
            assert_eq!(case["shape"], serde_json::json!([16, 8]));
            let bounds = &case["boundaries"];
            assert_eq!(bounds["act.oat"]["bitwise"], true);
            assert_eq!(bounds["act.oat"]["reference_repeat_bitwise"], true);
            let read = |name: &str, expected: &serde_json::Value| {
                let bytes = std::fs::read(dir.join(format!("captured/rank{rank}.{name}.bin"))).unwrap();
                assert_eq!(plow_asset::decode_objects::image_sha256(&bytes), expected.as_str().unwrap());
                bytes
            };
            let input = read("act.olat", &bounds["act.oat"]["input_sha256"]);
            let expected = read("act.oat", &bounds["act.oat"]["reference_sha256"]);
            let weight_name = "model.layers.3.self_attn.derived.mla_fp8_tp8.wv.weight";
            let scale_name = "model.layers.3.self_attn.derived.mla_fp8_tp8.wv.weight_scale";
            let weights = read(weight_name, &bounds[weight_name]["reference_sha256"]);
            let scale = read(scale_name, &bounds[scale_name]["reference_sha256"]);
            assert_eq!((input.len(), expected.len(), weights.len(), scale.len()),
                (16 * 8 * 512 * 2, 16 * 8 * 256 * 2, 8 * 256 * 512, 4));
            let mut padded = vec![255; input.len() * 2];
            for (source, target) in input.chunks_exact(1024).zip(padded.chunks_exact_mut(2048)) {
                target[..1024].copy_from_slice(source);
            }
            let buffers: Vec<_> = [&input, &padded, &weights, &scale].into_iter().map(|bytes| {
                let buffer = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
                be.memcpy_htod(buffer.base, bytes).unwrap();
                buffer
            }).collect();
            let output = EngineDevice::alloc(&be, expected.len() as u64).unwrap();
            for poison in [255, 85] {
                for (source, stride) in [(0usize, 0u64), (1, 1024)] {
                    be.memcpy_htod(output.base, &vec![poison; expected.len()]).unwrap();
                    let args = [output.base, buffers[source].base, buffers[2].base, buffers[3].base,
                        16u64 | 8u64 << 32, 256u64 | 512u64 << 32, stride];
                    be.begin_dispatch_chain(1).unwrap();
                    be.launch(kernel, 256, 512, 0, bytemuck::cast_slice(&args)).unwrap();
                    be.commit_dispatch_chain().unwrap();
                    be.synchronize().unwrap();
                    let mut actual = vec![0; expected.len()];
                    EngineDevice::download(&be, &output, 0, &mut actual).unwrap();
                    assert_eq!(actual, expected, "captured rank{rank} WV stride{stride} poison{poison}");
                }
            }
            if let Some(attention) = &attention {
                let attn_report = attention_report.as_ref().unwrap();
                let case = &attn_report["cases"][rank];
                assert_eq!(case["rank"], rank);
                assert_eq!(case["passed"], true);
                assert_eq!(case["attention"]["reference_repeat_bitwise"], true);
                assert_eq!(plow_asset::decode_objects::image_sha256(&input),
                    case["attention"]["reference_sha256"].as_str().unwrap());
                let raw = |name: &str| std::fs::read(dir.join(format!("captured/rank{rank}.{name}.bin"))).unwrap();
                let qa = raw("act.qa");
                let qr = raw("act.qr");
                assert_eq!((qa.len(),qr.len()),(16*8*512*2,16*8*64*2));
                let query: Vec<_> = qa.chunks_exact(1024).zip(qr.chunks_exact(128))
                    .flat_map(|(a,b)| a.iter().chain(b).copied()).collect();
                assert_eq!(plow_asset::decode_objects::image_sha256(&query),case["query_sha256"].as_str().unwrap());
                let ckv: Vec<_> = (0..16).flat_map(|row| raw(&format!("slot{row}.kv.3.ckv"))).collect();
                let krot: Vec<_> = (0..16).flat_map(|row| raw(&format!("slot{row}.kv.3.krot"))).collect();
                assert_eq!((ckv.len(),krot.len()),(16*512*512*2,16*512*64*2));
                let kv: Vec<_> = ckv.chunks_exact(1024).zip(krot.chunks_exact(128))
                    .flat_map(|(a,b)| a.iter().chain(b).copied()).collect();
                assert_eq!(plow_asset::decode_objects::image_sha256(&kv),case["kv_sha256"].as_str().unwrap());
                let indices = std::fs::read(dir.join("captured/act.iidx.bin")).unwrap();
                assert_eq!(indices.len(),16*2048*4);
                assert_eq!(plow_asset::decode_objects::image_sha256(&indices),
                    attn_report["selected_indices_sha256"].as_str().unwrap());
                let mapped: Vec<u32> = indices.chunks_exact(2048*4).enumerate().flat_map(|(row,bytes)| {
                    bytes[..512*4].chunks_exact(4).map(move |v| {
                        let index = u32::from_le_bytes(v.try_into().unwrap());
                        assert!(index<512); index + row as u32*512
                    })
                }).collect();
                assert_eq!(plow_asset::decode_objects::image_sha256(bytemuck::cast_slice(&mapped)),
                    case["mapped_indices_sha256"].as_str().unwrap());
                let values = [vec![255;input.len()*2],qa,qr,ckv,krot,
                    bytemuck::cast_slice(&[512u32;16]).to_vec(),indices];
                let devices: Vec<_> = values.iter().map(|bytes| {
                    let device = EngineDevice::alloc(&be,bytes.len() as u64).unwrap();
                    be.memcpy_htod(device.base,bytes).unwrap(); device
                }).collect();
                let table: Vec<u64> = devices.iter().map(|d|d.base).collect();
                let prior = previous_attention_inputs.as_ref().unwrap_or(&devices);
                let prior_table: Vec<u64> = prior.iter().map(|d|d.base).collect();
                for stride in [0u32,1024] {
                    let (mut program,tensors) = fixture(16,512);
                    if stride != 0 {
                        program.insts[1].i[4]=stride;
                        program.insts.push(DevInst64 {
                            op:DevOp::MlaBmmFp8 as u16,
                            t:[9,0,10,11,TENSOR_NONE16,TENSOR_NONE16,TENSOR_NONE16,TENSOR_NONE16],
                            i:[16,8,256,512,0,stride,0,0],..Default::default()
                        });
                    }
                    let mut route = route(&program,&tensors,0).unwrap().unwrap();
                    route.arm(&[512;16]);
                    for hoist in [false,true] {
                        let evidence = attention.selected_evidence(&be, route, hoist).unwrap();
                        assert_eq!(evidence.len(), if hoist { 2 } else { 1 });
                        println!("BF16_NATIVE_ROUTE_EVIDENCE rank={rank} stride={stride} hoist={hoist} {}",
                            serde_json::to_string(&evidence).unwrap());
                        let mut previous = None;
                        let mut producer = route;
                        producer.plan_metadata(&mut previous,hoist);
                        let mut consumer = route;
                        consumer.plan_metadata(&mut previous,hoist);
                        assert_eq!(evidence.last().unwrap().launches.len(), consumer.launches());
                        for poison in [255,85] {
                            attention.reset_metadata();
                            be.memcpy_htod(attention._scratch.base,&vec![poison;attention._scratch.len as usize]).unwrap();
                            be.memcpy_htod(devices[0].base,&vec![poison;values[0].len()]).unwrap();
                            be.memcpy_htod(output.base,&vec![poison;expected.len()]).unwrap();
                            let launches = consumer.launches()+1+if hoist { producer.launches() } else {0};
                            be.begin_dispatch_chain(launches).unwrap();
                            if hoist { attention.enqueue(&be,producer,bytemuck::cast_slice(&prior_table)).unwrap(); }
                            attention.enqueue(&be,consumer,bytemuck::cast_slice(&table)).unwrap();
                            let args = [output.base,devices[0].base,buffers[2].base,buffers[3].base,
                                16u64|8u64<<32,256u64|512u64<<32,u64::from(stride)];
                            be.launch(kernel,256,512,0,bytemuck::cast_slice(&args)).unwrap();
                            be.commit_dispatch_chain().unwrap();
                            be.synchronize().unwrap();
                            let mut latent = vec![0;input.len()*if stride==0 {1} else {2}];
                            EngineDevice::download(&be,&devices[0],0,&mut latent).unwrap();
                            let latent = if stride==0 {latent} else {
                                latent.chunks_exact(2048).flat_map(|row|row[..1024].iter().copied()).collect()
                            };
                            assert_eq!(latent,input,"rank{rank} attention stride{stride} hoist{hoist} poison{poison}");
                            let mut actual = vec![0;expected.len()];
                            EngineDevice::download(&be,&output,0,&mut actual).unwrap();
                            assert_eq!(actual,expected,"rank{rank} attention->WV stride{stride} hoist{hoist} poison{poison}");
                        }
                        println!("captured attention->WV chain bitwise PASS rank{rank} B16 ctx512 stride{stride} hoist{hoist}; no packet/full-model/T4 claim");
                    }
                }
                previous_attention_inputs=Some(devices);
            } else {
                println!("actual-weight WV bitwise PASS rank{rank} B16 ctx512; conditioned input, no serving/performance claim");
            }
        }
    }

    fn replay(strided_wv: bool) {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let dir = Path::new(&dir);
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let kernel = PersistentMla::load(&be, dir, 31, &mut modules).unwrap();
        let wv = if strided_wv {
            let image = std::fs::read(dir.join("test_kernels.elf")).unwrap();
            let module = EngineDevice::module_load(&be, &image).unwrap();
            let function = EngineDevice::get_function(&be, &module, "mla_bmm_fp8_stride").unwrap();
            modules.push(module);
            Some(function)
        } else {
            None
        };
        let report: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("reference/attention-comparison.json")).unwrap(),
        )
        .unwrap();
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 16);
        for case in cases {
            let bytes =
                std::fs::read(dir.join("reference").join(case["file"].as_str().unwrap())).unwrap();
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&bytes),
                case["sha256"].as_str().unwrap()
            );
            let word =
                |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
            let (m, ctx, nw, cap) = (word(1), word(2), word(4), word(6));
            let ragged = word(0) == 0x41505332;
            assert!(ragged || word(0) == 0x41505331);
            let lengths: Vec<u32> = (0..m)
                .map(|b| {
                    if ragged {
                        word(7 + b) as u32
                    } else {
                        ctx as u32
                    }
                })
                .collect();
            let mut at = 28 + if ragged { m * 4 } else { 0 };
            let q = &bytes[at..at + m * 16 * 576 * 2];
            at += q.len();
            let kv = &bytes[at..at + m * ctx * 576 * 2];
            at += kv.len();
            let total: usize = lengths.iter().map(|&n| n.min(2048) as usize).sum();
            let indices = &bytes[at..at + total * 4];
            at += indices.len() + 257 * 4 + cap * 8 * 4;
            let part = &bytes[at..at + nw * 16 * 512 * 4];
            at += part.len();
            let lse = &bytes[at..];
            assert_eq!(lse.len(), nw * 16 * 4);
            let mut inputs = [
                vec![255; m * 8 * 512 * 2],
                vec![0; m * 8 * 512 * 2],
                vec![0; m * 8 * 64 * 2],
                vec![0; m * ctx * 512 * 2],
                vec![0; m * ctx * 64 * 2],
                bytemuck::cast_slice(&lengths).to_vec(),
                vec![255; m * 2048 * 4],
            ];
            for row in 0..m * 8 {
                let src = row * 2 * 576 * 2;
                assert_eq!(&q[src..src + 1152], &q[src + 1152..src + 2304]);
                inputs[1][row * 1024..(row + 1) * 1024].copy_from_slice(&q[src..src + 1024]);
                inputs[2][row * 128..(row + 1) * 128].copy_from_slice(&q[src + 1024..src + 1152]);
            }
            for (row, src) in kv.chunks_exact(1152).enumerate() {
                inputs[3][row * 1024..(row + 1) * 1024].copy_from_slice(&src[..1024]);
                inputs[4][row * 128..(row + 1) * 128].copy_from_slice(&src[1024..]);
            }
            let mut at = 0;
            for (b, &len) in lengths.iter().enumerate() {
                for j in 0..len.min(2048) as usize {
                    let idx = u32::from_le_bytes(indices[at..at + 4].try_into().unwrap())
                        - (b * ctx) as u32;
                    assert!(idx < len);
                    let off = (b * 2048 + j) * 4;
                    inputs[6][off..off + 4].copy_from_slice(&idx.to_le_bytes());
                    at += 4;
                }
            }
            let reduction = std::fs::read(
                dir.join("reference")
                    .join(case["reduce_file"].as_str().unwrap()),
            )
            .unwrap();
            assert_eq!(
                plow_asset::decode_objects::image_sha256(&reduction),
                case["reduce_sha256"].as_str().unwrap()
            );
            let padded = &reduction[reduction.len() - m * 16 * 512 * 2..];
            let expected: Vec<u8> = padded
                .chunks_exact(2048)
                .flat_map(|v| v[..1024].iter().copied())
                .collect();
            let devices: Vec<_> = inputs
                .iter()
                .map(|v| {
                    let d = EngineDevice::alloc(&be, v.len() as u64).unwrap();
                    be.memcpy_htod(d.base, v).unwrap();
                    d
                })
                .collect();
            let table: Vec<u64> = devices.iter().map(|d| d.base).collect();
            let poison = vec![255; kernel._scratch.len as usize];
            for dense in [false, true] {
                if dense && ctx > 2048 {
                    continue;
                }
                let (mut p, t) = fixture(m as u32, ctx as u32);
                if dense {
                    // Preserve the reference's key order while replacing selection with identity.
                    for (handle, width) in [(3, 1024), (4, 128)] {
                        let mut reordered = inputs[handle].clone();
                        for (b, &len) in lengths.iter().enumerate() {
                            for j in 0..len as usize {
                                let off = (b * 2048 + j) * 4;
                                let key =
                                    u32::from_le_bytes(inputs[6][off..off + 4].try_into().unwrap())
                                        as usize;
                                let src = (b * ctx + key) * width;
                                let dst = (b * ctx + j) * width;
                                reordered[dst..dst + width]
                                    .copy_from_slice(&inputs[handle][src..src + width]);
                            }
                        }
                        be.memcpy_htod(devices[handle].base, &reordered).unwrap();
                    }
                    p.insts[0].op = DevOp::FlashMlaDecode as u16;
                    p.insts[0].t[7] = TENSOR_NONE16;
                    p.insts[0].i[6] = 0;
                }
                let mut r = route(&p, &t, 0).unwrap().unwrap();
                r.arm(&lengths);
                for _ in 0..2 {
                    be.memcpy_htod(kernel._scratch.base, &poison).unwrap();
                    be.memcpy_htod(devices[0].base, &inputs[0]).unwrap();
                    be.begin_dispatch_chain(5).unwrap();
                    kernel
                        .enqueue(&be, r, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.commit_dispatch_chain().unwrap();
                    be.synchronize().unwrap();
                    let mut output = vec![0; expected.len()];
                    EngineDevice::download(&be, &devices[0], 0, &mut output).unwrap();
                    assert_eq!(
                        output, expected,
                        "{} dense={dense} final BF16",
                        case["file"]
                    );
                    let offset = kernel.ptr[12] - kernel._scratch.base;
                    let mut actual = vec![0; part.len()];
                    EngineDevice::download(&be, &kernel._scratch, offset, &mut actual).unwrap();
                    assert_eq!(actual, part, "stage partials");
                    let mut actual = vec![0; lse.len()];
                    EngineDevice::download(
                        &be,
                        &kernel._scratch,
                        offset + (256 + 31 - 1) * 16 * 512 * 4,
                        &mut actual,
                    )
                    .unwrap();
                    assert_eq!(actual, lse, "stage LSE");
                }
                println!(
                    "BF16 Rust AQL chain bitwise PASS {} dense={dense}",
                    case["file"]
                );
                let mut changed = inputs.clone();
                for handle in [1, 2, 3, 4] {
                    EngineDevice::download(&be, &devices[handle], 0, &mut changed[handle]).unwrap();
                    for value in changed[handle].chunks_exact_mut(2) {
                        value[1] ^= 0x80;
                    }
                }
                if !dense {
                    for (row, &length) in lengths.iter().enumerate() {
                        let begin = row * 2048 * 4;
                        let end = begin + length.min(2048) as usize * 4;
                        changed[6][begin..end].rotate_left(4);
                    }
                }
                let changed_devices: Vec<_> = changed.iter().map(|data| {
                    let device = EngineDevice::alloc(&be, data.len() as u64).unwrap();
                    be.memcpy_htod(device.base, data).unwrap();
                    device
                }).collect();
                let changed_table: Vec<u64> = changed_devices.iter().map(|d| d.base).collect();
                let snapshot = || {
                    let mut output = vec![0; expected.len()];
                    EngineDevice::download(&be, &changed_devices[0], 0, &mut output).unwrap();
                    let mut partials = vec![0; part.len()];
                    EngineDevice::download(&be, &kernel._scratch,
                        kernel.ptr[12] - kernel._scratch.base, &mut partials).unwrap();
                    let mut ls = vec![0; lse.len()];
                    EngineDevice::download(&be, &kernel._scratch,
                        kernel.ptr[12] - kernel._scratch.base + (256 + 31 - 1) * 16 * 512 * 4,
                        &mut ls).unwrap();
                    (output, partials, ls)
                };
                kernel.reset_metadata();
                be.memcpy_htod(kernel._scratch.base, &poison).unwrap();
                be.begin_dispatch_chain(5).unwrap();
                kernel.enqueue(&be, r, bytemuck::cast_slice(&changed_table)).unwrap();
                be.commit_dispatch_chain().unwrap();
                be.synchronize().unwrap();
                let fresh = snapshot();
                let mut previous = None;
                let mut producer = r;
                producer.plan_metadata(&mut previous, true);
                let mut consumer = r;
                consumer.plan_metadata(&mut previous, true);
                assert_eq!((producer.launches(), consumer.launches()), (5, 4));
                for _ in 0..2 {
                    kernel.reset_metadata();
                    assert!(kernel.enqueue(&be, consumer, bytemuck::cast_slice(&changed_table)).is_err());
                    be.memcpy_htod(kernel._scratch.base, &poison).unwrap();
                    be.memcpy_htod(changed_devices[0].base, &changed[0]).unwrap();
                    be.begin_dispatch_chain(9).unwrap();
                    kernel.enqueue(&be, producer, bytemuck::cast_slice(&table)).unwrap();
                    kernel.enqueue(&be, consumer, bytemuck::cast_slice(&changed_table)).unwrap();
                    be.commit_dispatch_chain().unwrap();
                    be.synchronize().unwrap();
                    assert_eq!(snapshot(), fresh, "metadata reuse changes layer-local data");
                }
                kernel.reset_metadata();
                assert!(kernel.enqueue(&be, consumer, bytemuck::cast_slice(&changed_table)).is_err());
                println!("BF16 metadata hoist bitwise PASS {} dense={dense}", case["file"]);
                if let Some(wv) = wv {
                    let output = EngineDevice::alloc(&be, padded.len() as u64).unwrap();
                    let values: Vec<u8> = (0..8 * 256 * 512)
                        .map(|i| ((i * 17 + i / 128) % 110 + 1) as u8 ^ ((i % 3 == 0) as u8 * 128))
                        .collect();
                    let scales: Vec<f32> = (0..8 * 2 * 4).map(|i| 2f32.powi(i % 5 - 8)).collect();
                    let weights = EngineDevice::alloc(&be, values.len() as u64).unwrap();
                    be.memcpy_htod(weights.base, &values).unwrap();
                    let scale = EngineDevice::alloc(&be, (scales.len() * 4) as u64).unwrap();
                    be.memcpy_htod(scale.base, bytemuck::cast_slice(&scales)).unwrap();
                    let projected = EngineDevice::alloc(&be, (m * 8 * 256 * 2) as u64).unwrap();
                    let project = |input: u64, stride: u32| {
                        let args = [projected.base, input, weights.base, scale.base,
                            m as u64 | 8u64 << 32, 256u64 | 512u64 << 32, u64::from(stride)];
                        be.launch(wv, 256, 512, 0, bytemuck::cast_slice(&args)).unwrap();
                    };
                    be.begin_dispatch_chain(1).unwrap();
                    project(changed_devices[0].base, 0);
                    be.commit_dispatch_chain().unwrap();
                    be.synchronize().unwrap();
                    let mut reference = vec![0; projected.len as usize];
                    EngineDevice::download(&be, &projected, 0, &mut reference).unwrap();
                    let mut direct = r;
                    direct.padded_output = true;
                    let mut direct_table = changed_table.clone();
                    direct_table[0] = output.base;
                    for reuse in [false, true] {
                        kernel.reset_metadata();
                        let mut previous = None;
                        let mut producer = direct;
                        producer.plan_metadata(&mut previous, reuse);
                        direct.plan_metadata(&mut previous, reuse);
                        let launches = direct.launches() + 1 + if reuse { producer.launches() } else { 0 };
                        be.memcpy_htod(kernel._scratch.base, &poison).unwrap();
                        be.memcpy_htod(output.base, &vec![255; padded.len()]).unwrap();
                        be.begin_dispatch_chain(launches).unwrap();
                        if reuse {
                            let mut producer_table = table.clone();
                            producer_table[0] = output.base;
                            kernel.enqueue(&be, producer, bytemuck::cast_slice(&producer_table)).unwrap();
                        }
                        kernel.enqueue(&be, direct, bytemuck::cast_slice(&direct_table)).unwrap();
                        project(output.base, 1024);
                        be.commit_dispatch_chain().unwrap();
                        be.synchronize().unwrap();
                        let mut actual = vec![0; reference.len()];
                        EngineDevice::download(&be, &projected, 0, &mut actual).unwrap();
                        assert_eq!(actual, reference, "stride WV changed projection; reuse={reuse}");
                        let mut padded_actual = vec![0; padded.len()];
                        EngineDevice::download(&be, &output, 0, &mut padded_actual).unwrap();
                        let unpadded: Vec<u8> = padded_actual.chunks_exact(2048)
                            .flat_map(|v| v[..1024].iter().copied()).collect();
                        assert_eq!(unpadded, fresh.0, "direct padded output changed BF16 values");
                        for heads in padded_actual.chunks_exact_mut(2048) {
                            heads[1024..].fill(255);
                        }
                        be.memcpy_htod(output.base, &padded_actual).unwrap();
                        be.begin_dispatch_chain(1).unwrap();
                        project(output.base, 1024);
                        be.commit_dispatch_chain().unwrap();
                        be.synchronize().unwrap();
                        EngineDevice::download(&be, &projected, 0, &mut actual).unwrap();
                        assert_eq!(actual, reference, "stride WV consumed an odd padded head");
                    }
                    println!("BF16 stride WV bitwise PASS {} dense={dense}", case["file"]);
                }
            }
        }
    }
}
