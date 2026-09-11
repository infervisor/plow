use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR, TENSOR_NONE16};

use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::hsa::{HsaBackend, HsaKernel};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

const OBJECT: &str = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co";
const OBJECT_HASH: &str = "65b4c0a0b290dd83039047c18e0bb86f4253790e926dce324ddb6b45a7b28650";

const FLAT_OBJECT: &str = "fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co";
const FLAT_OBJECT_HASH: &str = "be7052284094e7cedeb266afb24d4d6723bdf4234ac391b2e5b29473d8ee8f06";

// AITER's GLM-5 gfx942/304-CU tuning (a8w8_blockscale_tuned_fmoe_glm5_1.csv) selects this
// 64-row persistent tile for the 6144/256/256/8 shape from 2048 tokens up; 1024 takes the
// same tile non-persistent, below that the 32x256 objects. Same 448-byte ABI and kd layout.
const TILE64_OBJECT: &str = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_psx_64x256.co";
const TILE64_OBJECT_HASH: &str = "f8efb79a4ecfd80c7c4d6e20c797c7bdcdac26f1c22c825c640b239e07e88779";
const TILE64_MIN_ROWS: u32 = 1024;

/// The unset-flag default: the 64-row tile runs when the object dir carries the pinned object
/// and a tile64-marked adapter (both from `scripts/build_moe_aiter.sh` with its 4th argument).
/// Qualified on GLM-5.3 TP8 gfx942: 18/18 retrieval, 49.79 -> 50.57 tok/s at C20/70k.
pub(super) fn tile64_available(dir: &Path) -> bool {
    let object = std::fs::read(dir.join(TILE64_OBJECT))
        .is_ok_and(|image| plow_asset::decode_objects::image_sha256(&image) == TILE64_OBJECT_HASH);
    let adapter = std::fs::read(dir.join("moe_aiter_adapter_gfx942.elf")).is_ok_and(|image| {
        super::amd::elf_symbol_names(&image).contains(&"plow_moe_aiter_tile64_abi_1")
    });
    if !(object && adapter) {
        tracing::info!(
            object,
            adapter,
            "64-row MoE prefill tile off: object dir lacks the pinned object or a tile64 adapter"
        );
    }
    object && adapter
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Sorted,
    SortedBf16,
    Flat,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    inst: DevInst64,
    rows: u32,
    mode: Mode,
}

impl Route {
    pub fn launches(self) -> u64 {
        match self.mode {
            Mode::Sorted => {
                if self.inst.i[7] == 1 {
                    3
                } else {
                    4
                }
            }
            Mode::Flat => 2,
            Mode::SortedBf16 => {
                if self.inst.i[7] == 1 {
                    2
                } else {
                    3
                }
            }
        }
    }

    pub fn rebase(&mut self, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.inst.i[0] {
            return Err(RuntimeError::Device(
                "AITER MoE chunk exceeds row capacity".into(),
            ));
        }
        self.rows = rows;
        Ok(())
    }
}

pub(super) fn routes(
    prog: &DevProg,
    tensors: &[DevTensor],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut routes = vec![None; segments];
    for (ix, inst) in prog.insts.iter().enumerate() {
        if inst.op != DevOp::MoeAiterFp8Pf as u16 {
            continue;
        }
        let err = |s: &str| RuntimeError::Device(format!("AITER MoE instruction {ix}: {s}"));
        let mode = match inst.i[6] {
            0 => Mode::Sorted,
            1 => Mode::Flat,
            2 => Mode::SortedBf16,
            _ => return Err(err("unknown output mode")),
        };
        let flat = mode == Mode::Flat;
        let resident = inst.i[7] == 1;
        let geometry = if flat {
            [prog.t, 6144, 256, 256, 8, 0, 1, u32::from(resident)]
        } else {
            [prog.t, 6144, 256, 256, 8, 64, inst.i[6], u32::from(resident)]
        };
        // A packed-prefill sibling carries this instruction unchanged: the kernel is
        // row-agnostic over the dense live rows (the align maps sort real tokens; parked rows
        // route to nothing a live row reads), which is what lets the packet pack requests.
        if !(if flat {
                matches!(prog.t, 2 | 4 | 8) || (resident && matches!(prog.t, 1 | 16 | 20))
            } else {
                (1..=8192).contains(&prog.t)
            })
            || inst.i != geometry
            || inst.fj != [0; 3]
        {
            return Err(err("requires H6144/I256/E256/top8; sorted prefill rows1..8192 or flat decode rows2/4/8 (resident:1/2/4/8/16/20)"));
        }
        if flat {
            let router = prog.insts[..ix]
                .iter()
                .rev()
                .find(|d| d.op == DevOp::MoeRouterTopkPf as u16 && d.t[0] == inst.t[4])
                .ok_or_else(|| err("no preceding raw routing table"))?;
            if router.i[1..3] != [256, 8] || router.i[4] != prog.t {
                return Err(err("raw routing geometry does not match"));
            }
            let combine = prog.insts[ix + 1..]
                .iter()
                .find(|d| d.op == DevOp::MoeCombinePf as u16 && d.t[3] == inst.t[0])
                .ok_or_else(|| err("flat output has no BF16 combine consumer"))?;
            if combine.i[..5] != [6144, 1, prog.t, 0, 0] || combine.i[7] != 1 {
                return Err(err("flat output requires one BF16 partial per row"));
            }
        } else {
            let align = prog.insts[..ix]
                .iter()
                .rev()
                .find(|d| d.op == DevOp::MoeAlignPf as u16 && d.t[0] == inst.t[4])
                .ok_or_else(|| err("no preceding aligned routing table"))?;
            if align.i[..3] != [prog.t, 256, 8]
                || ![0, 4].contains(&align.i[3])
                || align.t[2..5] != inst.t[5..8]
            {
                return Err(err("aligned routing geometry does not match"));
            }
        }
        if mode == Mode::SortedBf16 {
            if inst.t[1..].contains(&inst.t[0]) {
                return Err(err("direct BF16 output aliases an input"));
            }
            let mut covered = 0;
            // Stop at the next producer: subsequent layers can reuse this buffer.
            for combine in prog.insts[ix + 1..]
                .iter()
                .take_while(|d| d.t[0] != inst.t[0])
                .filter(|d| d.op == DevOp::MoeCombinePf as u16 && d.t[3] == inst.t[0])
            {
                if combine.i[..2] != [6144, 1]
                    || combine.i[2] == 0
                    || combine.i[2] > prog.t - covered
                    || combine.i[3] != covered
                    || combine.i[4..7] != [0; 3]
                    || combine.i[7] != 1
                {
                    return Err(err(
                        "sorted BF16 output requires contiguous BF16 combine bands",
                    ));
                }
                covered += combine.i[2];
            }
            if covered != prog.t {
                return Err(err("sorted BF16 output has incomplete combine coverage"));
            }
        }
        let rows = u64::from(prog.t);
        let capacity = rows * 8 + 256 * 63;
        let sizes = if flat {
            [
                rows * 6144 * 2 + 8,
                rows * 6144 * 2,
                256 * 3 * 8,
                256 * 3 * 8,
                rows * 8 * 8,
                0,
                0,
                0,
            ]
        } else {
            [
                rows * 6144 * if mode == Mode::SortedBf16 { 2 } else { 4 },
                rows * 6144 * 2,
                256 * 3 * 8,
                256 * 3 * 8,
                (3 * 256 + 1) * 4,
                capacity * 4,
                capacity * 4,
                capacity * 4,
            ]
        };
        for (handle, bytes) in inst.t.into_iter().zip(sizes) {
            if bytes == 0 {
                if handle != TENSOR_NONE16 {
                    return Err(err("unused flat operand must be absent"));
                }
            } else if handle == TENSOR_NONE16
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(err("operand capacity is insufficient"));
            }
        }
        let mut owners = BTreeSet::new();
        for entry in prog
            .stream
            .iter()
            .chain(&prog.gq_stream)
            .filter(|e| e.inst as usize == ix)
        {
            if entry.wait_len != 0 || entry.succ_len != 0 || entry.flags & SE_XCTR != 0 {
                return Err(err(
                    "native segment retains interpreter counter obligations",
                ));
            }
            owners.insert(entry.seg as usize);
        }
        if owners.len() != 1 {
            return Err(err("requires exactly one segment owner"));
        }
        let seg = *owners.first().unwrap();
        if seg >= segments
            || prog
                .stream
                .iter()
                .chain(&prog.gq_stream)
                .any(|e| e.seg as usize == seg && e.inst as usize != ix)
        {
            return Err(err("segment contains other interpreter work"));
        }
        routes[seg] = Some(Route {
            inst: *inst,
            rows: prog.t,
            mode,
        });
    }
    Ok(routes)
}

pub(super) fn resident_tables(
    progs: &[DevProg],
    tensors: &[DevTensor],
) -> Result<Vec<Option<u16>>> {
    let mut tables = vec![None; tensors.len()];
    let mut staged = false;
    for inst in progs
        .iter()
        .flat_map(|p| &p.insts)
        .filter(|d| d.op == DevOp::MoeAiterFp8Pf as u16)
    {
        match inst.i[7] {
            0 => {
                staged = true;
                continue;
            }
            1 => {}
            _ => {
                return Err(RuntimeError::Device(
                    "unknown native MoE weight layout".into(),
                ))
            }
        }
        let invalid = || {
            RuntimeError::Device("resident MoE requires matching H6144/I256/E256/top8 expert tables without companions".into())
        };
        let wt = tensors.get(inst.t[2] as usize).ok_or_else(invalid)?;
        let st = tensors.get(inst.t[3] as usize).ok_or_else(invalid)?;
        let prefix = wt
            .name
            .strip_suffix("expert_weight_table")
            .ok_or_else(invalid)?;
        if inst.i[1..5] != [6144, 256, 256, 8]
            || wt.bytes != 256 * 24
            || st.bytes != 256 * 24
            || st.name != format!("{prefix}expert_scale_table")
            || tensors.iter().any(|t| {
                t.name.starts_with(&format!("{prefix}expert_weight_table_"))
                    || t.name.starts_with(&format!("{prefix}expert_scale_table_"))
            })
        {
            return Err(invalid());
        }
        if tables[inst.t[2] as usize]
            .replace(inst.t[3])
            .is_some_and(|old| old != inst.t[3])
        {
            return Err(invalid());
        }
    }
    if tables.iter().all(Option::is_none) {
        return Ok(tables);
    }
    if staged {
        return Err(RuntimeError::Device(
            "resident MoE cannot mix staged native weights".into(),
        ));
    }
    for (weight, scale) in tables
        .iter()
        .enumerate()
        .filter_map(|(w, s)| s.map(|s| (w as u16, s)))
    {
        for inst in progs.iter().flat_map(|p| &p.insts) {
            if inst.t.contains(&weight) || inst.t.contains(&scale) {
                if inst.op != DevOp::MoeAiterFp8Pf as u16
                    || inst.i[7] != 1
                    || inst.t[2..4] != [weight, scale]
                    || inst
                        .t
                        .iter()
                        .enumerate()
                        .any(|(slot, &t)| (t == weight || t == scale) && slot != 2 && slot != 3)
                {
                    return Err(RuntimeError::Device(
                        "resident MoE table has an incompatible consumer".into(),
                    ));
                }
            }
        }
    }
    Ok(tables)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResidentWeights {
    pub gu: u64,
    pub down: u64,
    pub gs: u64,
    pub ds: u64,
}

impl ResidentWeights {
    pub fn new(weights: u64, scales: u64) -> Self {
        Self {
            gu: weights,
            down: weights + 512 * 256 * 6144,
            gs: scales,
            ds: scales + 512 * 96 * 4,
        }
    }
}

pub(super) fn resident_expert_table(base: u64, stride: u64) -> Vec<u64> {
    (0..256u64)
        .flat_map(|e| {
            [
                base + 2 * e * stride,
                base + (2 * e + 1) * stride,
                base + (512 + e) * stride,
            ]
        })
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlatRouteArgs {
    pointers: [u64; 3],
    rows: u32,
    pad: u32,
}
const _: () = assert!(std::mem::size_of::<FlatRouteArgs>() == 32);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepareArgs {
    pointers: [u64; 12],
    rows: u32,
    block_m: u32,
    swizzle: u32,
    pad: u32,
}

/// Kernarg bytes of `plow_moe_aiter_prepare` before (no `swizzle`) and after the XCD field.
const PREPARE_ARGS_LEGACY: u32 = 104;
const PREPARE_ARGS: u32 = 112;

/// Bit i of a sorted-route stage mask (`enqueue_sorted`): weight pack, prepare, fmoe, store.
pub(super) const STAGE_PACK: u8 = 1;
pub(super) const STAGE_PREPARE: u8 = 2;
pub(super) const STAGE_MOE: u8 = 4;
pub(super) const STAGE_STORE: u8 = 8;

/// Host mirror of the adapter's `swizzled_block`: the sorted block at position `position` of
/// the launch order. Blocks come expert-major from the align pass; with K = 256 / block_m
/// (mean rows per expert per rank at top-8 of 256) the K consecutive blocks of one expert are
/// placed 8 positions apart, so on gfx942 (workgroup w -> XCD w % 8) they run concurrently on
/// one XCD and its L2 serves the expert's weights once instead of K times. A partial tail
/// chunk keeps the identity order. Single MI300X, 8192 rows, random top-8: the ps_32x256
/// fmoe launch 2628 -> 1500 us (2048 rows: 851 -> 547). Only for the ps object — see `enqueue`.
pub(super) fn xcd_swizzled_block(position: u32, block_m: u32, blocks: u32) -> u32 {
    let k = 256 / block_m;
    let chunk = 8 * k;
    let (c, r) = (position / chunk, position % chunk);
    if (c + 1) * chunk > blocks {
        return position;
    }
    c * chunk + (r % 8) * k + r / 8
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlatPackArgs {
    pointers: [u64; 9],
    rows: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<FlatPackArgs>() == 80);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StoreArgs {
    out: u64,
    src: u64,
    rows: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<PrepareArgs>() == PREPARE_ARGS as usize);
const _: () = assert!(std::mem::size_of::<StoreArgs>() == 24);

pub(super) struct MoeAiter {
    pack: HsaKernel,
    flat: Option<(HsaKernel, HsaKernel)>,
    prepare: HsaKernel,
    moe: HsaKernel,
    tile64: Option<HsaKernel>,
    store: HsaKernel,
    _scratch: DeviceMem,
    buffers: [u64; 11],
    resident: bool,
    resident_weights: Vec<Option<ResidentWeights>>,
    prepare_args: u32,
    xcd_swizzle: bool,
}

impl MoeAiter {
    pub fn load(
        be: &HsaBackend,
        dir: &Path,
        rows: u32,
        flat_decode: bool,
        resident: bool,
        tile64: bool,
        modules: &mut Vec<Module>,
    ) -> Result<Self> {
        if !(128..=8192).contains(&rows) {
            return Err(RuntimeError::Device(
                "AITER MoE workspace requires rows 128..8192".into(),
            ));
        }
        let path = dir.join(OBJECT);
        let mut image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if plow_asset::decode_objects::image_sha256(&image) != OBJECT_HASH {
            return Err(RuntimeError::Device(
                "AITER MoE object hash does not match qualified ABI".into(),
            ));
        }
        // The pinned descriptor at 0x1d00 omits KERNARG_SIZE; HIP uses metadata,
        // but ROCr reads the descriptor. Normalize only this field after hashing.
        image[0x1d08..0x1d0c].copy_from_slice(&448u32.to_le_bytes());
        let module = EngineDevice::module_load(be, &image)?;
        let moe = EngineDevice::get_function(
            be,
            &module,
            "_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E",
        )?;
        if moe.kernarg_size() != 448
            || moe.private_segment_size() != 0
            || HsaBackend::kernel_lds_bytes(&moe) != 65536
        {
            return Err(RuntimeError::Device(
                "AITER MoE resource ABI mismatch".into(),
            ));
        }
        modules.push(module);
        let tile64 = if tile64 {
            let path = dir.join(TILE64_OBJECT);
            let mut image = std::fs::read(&path)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            if plow_asset::decode_objects::image_sha256(&image) != TILE64_OBJECT_HASH {
                return Err(RuntimeError::Device(
                    "AITER 64-row MoE object hash does not match qualified ABI".into(),
                ));
            }
            // Same descriptor layout as the 32x256 object: KERNARG_SIZE is left zero at 0x1d08.
            image[0x1d08..0x1d0c].copy_from_slice(&448u32.to_le_bytes());
            let module = EngineDevice::module_load(be, &image)?;
            let moe = EngineDevice::get_function(
                be,
                &module,
                "_ZN5aiter46fmoe_bf16_blockscaleFp8_g1u1_vs_ps_silu_64x256E",
            )?;
            if moe.kernarg_size() != 448
                || moe.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&moe) != 65536
            {
                return Err(RuntimeError::Device(
                    "AITER 64-row MoE resource ABI mismatch".into(),
                ));
            }
            modules.push(module);
            tracing::info!(object = %path.display(), sha256 = TILE64_OBJECT_HASH, "64-row MoE prefill kernel loaded");
            Some(moe)
        } else {
            None
        };
        let path = dir.join("moe_aiter_adapter_gfx942.elf");
        let image = std::fs::read(&path)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
        if !super::amd::elf_symbol_names(&image).contains(&"plow_moe_aiter_fp8_abi_1") {
            return Err(RuntimeError::Device(
                "AITER MoE adapter lacks ABI marker".into(),
            ));
        }
        if tile64.is_some()
            && !super::amd::elf_symbol_names(&image).contains(&"plow_moe_aiter_tile64_abi_1")
        {
            return Err(RuntimeError::Device(
                "AITER MoE adapter lacks 64-row block ABI marker".into(),
            ));
        }
        if resident
            && !super::amd::elf_symbol_names(&image).contains(&"plow_moe_resident_fp8_abi_1")
        {
            return Err(RuntimeError::Device(
                "resident MoE adapter lacks ABI marker".into(),
            ));
        }
        let flat = if flat_decode {
            if !super::amd::elf_symbol_names(&image).contains(&"plow_moe_flat_fp8_abi_1") {
                return Err(RuntimeError::Device(
                    "flat MoE adapter lacks ABI marker".into(),
                ));
            }
            let path = dir.join(FLAT_OBJECT);
            let mut image = std::fs::read(&path)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            if plow_asset::decode_objects::image_sha256(&image) != FLAT_OBJECT_HASH {
                return Err(RuntimeError::Device(
                    "flat MoE object hash does not match qualified ABI".into(),
                ));
            }
            // The pinned descriptor omits KERNARG_SIZE although metadata declares 448 bytes.
            image[0x1208..0x120c].copy_from_slice(&448u32.to_le_bytes());
            let module = EngineDevice::module_load(be, &image)?;
            let moe = EngineDevice::get_function(
                be,
                &module,
                "_ZN5aiter60fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3E",
            )?;
            if moe.kernarg_size() != 448
                || moe.private_segment_size() != 0
                || HsaBackend::kernel_lds_bytes(&moe) != 65536
            {
                return Err(RuntimeError::Device(
                    "flat MoE resource ABI mismatch".into(),
                ));
            }
            modules.push(module);
            tracing::info!(object = %path.display(), sha256 = FLAT_OBJECT_HASH, "flat MoE decode kernels loaded");
            Some(moe)
        } else {
            None
        };
        let module = EngineDevice::module_load(be, &image)?;
        let flat = flat
            .map(|moe| -> Result<_> {
                let name = if resident {
                    "plow_moe_aiter_flat_route"
                } else {
                    "plow_moe_aiter_flat_pack"
                };
                let size = if resident { 32 } else { 80 };
                let pack = EngineDevice::get_function(be, &module, name)?;
                if ![size, size + 256].contains(&pack.kernarg_size())
                    || pack.private_segment_size() != 0
                {
                    return Err(RuntimeError::Device("flat MoE packing ABI mismatch".into()));
                }
                Ok((pack, moe))
            })
            .transpose()?;
        let pack = EngineDevice::get_function(be, &module, "pack_moe")?;
        let prepare = EngineDevice::get_function(be, &module, "plow_moe_aiter_prepare")?;
        let store = EngineDevice::get_function(be, &module, "plow_moe_aiter_store")?;
        let prepare_args = if super::amd::elf_symbol_names(&image)
            .contains(&"plow_moe_aiter_swizzle_abi_1")
        {
            PREPARE_ARGS
        } else {
            PREPARE_ARGS_LEGACY
        };
        for (kernel, size) in [(pack, 48), (prepare, prepare_args), (store, 24)] {
            if ![size, size + 256].contains(&kernel.kernarg_size())
                || kernel.private_segment_size() != 0
            {
                return Err(RuntimeError::Device(format!("AITER MoE adapter resource ABI mismatch: kernarg={} expected={size}, private={}",
                    kernel.kernarg_size(), kernel.private_segment_size())));
            }
        }
        modules.push(module);
        let rows = u64::from(rows);
        let capacity = rows * 8 + 256 * 63;
        let mut sizes = [
            256 * 512 * 6144,
            256 * 6144 * 256,
            256 * 192 * 4,
            256 * 96 * 4,
            rows * 6144,
            rows * 48 * 4,
            rows * 6144 * 2,
            capacity * 4,
            capacity * 4,
            capacity.div_ceil(32) * 4,
            8,
        ];
        if resident {
            sizes[..4].fill(0);
        }
        let bytes = sizes.iter().map(|s| s.div_ceil(256) * 256).sum();
        let scratch = EngineDevice::alloc(be, bytes)?;
        let mut ptr = scratch.base;
        let buffers = sizes.map(|size| {
            let p = ptr;
            ptr += size.div_ceil(256) * 256;
            p
        });
        tracing::info!(bytes, "allocated reusable AITER MoE workspace");
        Ok(Self {
            pack,
            flat,
            prepare,
            moe,
            tile64,
            store,
            _scratch: scratch,
            buffers,
            resident,
            resident_weights: Vec::new(),
            prepare_args,
            xcd_swizzle: false,
        })
    }

    /// Launch the sorted blocks of the 32x256 object in XCD-swizzled order
    /// (`xcd_swizzled_block`). `None` = on when the adapter carries the
    /// `plow_moe_aiter_swizzle_abi_1` field (qualified: 18/18 retrieval, 49.89 -> 50.54 tok/s
    /// at C20/70k with the 32-row object); `Some(true)` is refused without it.
    pub fn set_xcd_swizzle(&mut self, on: Option<bool>) -> Result<()> {
        let supported = self.prepare_args == PREPARE_ARGS;
        self.xcd_swizzle = match on {
            Some(true) if !supported => {
                return Err(RuntimeError::Device(
                    "AITER MoE adapter lacks the XCD swizzle ABI marker".into(),
                ));
            }
            Some(on) => on,
            None => {
                if !supported {
                    tracing::info!("XCD-swizzled MoE block order off: adapter lacks plow_moe_aiter_swizzle_abi_1");
                }
                supported
            }
        };
        Ok(())
    }

    pub fn bind_resident(&mut self, tables: Vec<(u16, ResidentWeights)>) {
        for (handle, weights) in tables {
            self.resident_weights.resize(
                self.resident_weights.len().max(usize::from(handle) + 1),
                None,
            );
            self.resident_weights[usize::from(handle)] = Some(weights);
        }
    }

    pub fn enqueue(&self, be: &HsaBackend, route: Route, tensor_table: &[u8]) -> Result<()> {
        self.enqueue_stages(be, route, tensor_table, !0)
    }

    /// `stages` masks the sorted route's launches (`STAGE_*`) so a measurement can time one
    /// stage at a time; the flat route ignores it.
    pub(super) fn enqueue_stages(
        &self,
        be: &HsaBackend,
        route: Route,
        tensor_table: &[u8],
        stages: u8,
    ) -> Result<()> {
        let addr = |handle: u16| {
            let at = usize::from(handle) * 8;
            u64::from_le_bytes(tensor_table[at..at + 8].try_into().unwrap())
        };
        let t = route
            .inst
            .t
            .map(|h| if h == TENSOR_NONE16 { 0 } else { addr(h) });
        let [mut gu, mut down, mut gs, mut ds, q, qs, workspace_out, ids, weights, experts, valid] =
            self.buffers;
        if (route.inst.i[7] == 1) != self.resident {
            return Err(RuntimeError::Device(
                "MoE route and workspace weight layouts differ".into(),
            ));
        }
        if self.resident {
            let w = self
                .resident_weights
                .get(route.inst.t[2] as usize)
                .and_then(|w| *w)
                .ok_or_else(|| {
                    RuntimeError::Device("resident MoE weights were not bound".into())
                })?;
            (gu, down, gs, ds) = (w.gu, w.down, w.gs, w.ds);
        }
        if route.mode == Mode::Flat {
            let (pack, moe) = self.flat.ok_or_else(|| {
                RuntimeError::Device("flat MoE route has no loaded kernels".into())
            })?;
            if self.resident {
                let args = FlatRouteArgs {
                    pointers: [ids, weights, t[4]],
                    rows: route.rows,
                    pad: 0,
                };
                be.launch(pack, 1, 256, 0, bytemuck::bytes_of(&args))?;
            } else {
                let pack_args = FlatPackArgs {
                    pointers: [gu, down, gs, ds, t[2], t[3], t[4], ids, weights],
                    rows: route.rows,
                    pad: 0,
                };
                be.launch(pack, 2048, 256, 0, bytemuck::bytes_of(&pack_args))?;
            }
            let args = flat_moe_args(t[0], t[1], gu, down, gs, ds, ids, weights, route.rows);
            return be.launch_3d(moe, [2, 8, route.rows], 256, bytemuck::cast_slice(&args));
        }
        if !self.resident && stages & STAGE_PACK != 0 {
            be.launch(
                self.pack,
                2048,
                256,
                0,
                bytemuck::cast_slice(&[gu, down, gs, ds, t[2], t[3]]),
            )?;
        }
        let out = if route.mode == Mode::SortedBf16 {
            t[0]
        } else {
            workspace_out
        };
        let tile64 = self.tile64.filter(|_| route.rows >= TILE64_MIN_ROWS);
        let prepare = PrepareArgs {
            pointers: [
                q, qs, out, ids, weights, experts, valid, t[1], t[4], t[5], t[6], t[7],
            ],
            rows: route.rows,
            block_m: if tile64.is_some() { 64 } else { 32 },
            // The psx object remaps workgroup w to tile (w % 8) * 38 + w / 8 itself (its prologue:
            // s_and_b32 7 / s_lshr_b32 3 on the workgroup id), so consecutive blocks already share
            // an XCD there; a second permutation on top measured +8%. The ps object has no remap.
            swizzle: u32::from(self.xcd_swizzle && tile64.is_none()),
            pad: 0,
        };
        if stages & STAGE_PREPARE != 0 {
            be.launch(
                self.prepare,
                304 * 4,
                256,
                0,
                &bytemuck::bytes_of(&prepare)[..self.prepare_args as usize],
            )?;
        }
        let mut args = [0u64; 56];
        for (i, value) in [
            out,
            q,
            gu,
            valid,
            down,
            qs,
            gs,
            ds,
            0,
            ids,
            weights,
            experts,
            6144,
            256,
            u64::from(route.rows),
            256,
            6144,
            6144,
            256,
            12288,
            512 * 6144,
            256 * 6144,
            192 * 4,
            96 * 4,
            256 * 4,
            8,
            304,
            1,
        ]
        .into_iter()
        .enumerate()
        {
            args[i * 2] = value;
        }
        if stages & STAGE_MOE != 0 {
            be.launch(
                tile64.unwrap_or(self.moe),
                304,
                256,
                0,
                bytemuck::cast_slice(&args),
            )?;
        }
        if route.mode == Mode::SortedBf16 || stages & STAGE_STORE == 0 {
            return Ok(());
        }
        let store = StoreArgs {
            out: t[0],
            src: out,
            rows: route.rows,
            pad: 0,
        };
        be.launch(self.store, 304, 256, 0, bytemuck::bytes_of(&store))
    }
}

#[allow(clippy::too_many_arguments)]
fn flat_moe_args(
    out: u64,
    x: u64,
    gu: u64,
    down: u64,
    gs: u64,
    ds: u64,
    ids: u64,
    weights: u64,
    rows: u32,
) -> [u64; 56] {
    let mut args = [0; 56];
    for (i, value) in [
        out,
        x,
        gu,
        ids,
        down,
        0,
        gs,
        ds,
        0,
        ids,
        weights,
        ids,
        6144,
        256,
        u64::from(rows),
        256,
        12288,
        6144,
        256,
        12288,
        512 * 6144,
        256 * 6144,
        192 * 4,
        96 * 4,
        256 * 4,
        8,
        0,
        2,
    ]
    .into_iter()
    .enumerate()
    {
        args[i * 2] = value;
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    #[test]
    fn tile64_auto_requires_pinned_object_and_marked_adapter() {
        let dir = std::env::temp_dir().join(format!("plow-tile64-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!tile64_available(&dir));
        std::fs::write(dir.join(TILE64_OBJECT), b"not the pinned object").unwrap();
        assert!(!tile64_available(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sorted_prefill_routes_cover_the_full_append_ladder() {
        for rows in [1,2,4,8,16,20,32,64,128,256,512,1024,2048,4096,8192] {
            let (mut prog, tensors) = fixture();
            prog.t = rows;
            prog.insts[0].i[0] = rows;
            prog.insts[1].i[0] = rows;
            let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
            route.rebase(1).unwrap();
            route.rebase(rows).unwrap();
            assert!(route.rebase(rows + 1).is_err());
        }
    }

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let align = DevInst64 {
            op: DevOp::MoeAlignPf as u16,
            t: [4, 8, 5, 6, 7, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
            i: [8192, 256, 8, 4, 64, 0, 0, 0],
            ..Default::default()
        };
        let moe = DevInst64 {
            op: DevOp::MoeAiterFp8Pf as u16,
            t: [0, 1, 2, 3, 4, 5, 6, 7],
            i: [8192, 6144, 256, 256, 8, 64, 0, 0],
            ..Default::default()
        };
        let prog = DevProg {
            t: 8192,
            role: packet::devbuild::ProgramRole::PrefillBucket { rows: 8192 },
            n_counter: 0,
            insts: vec![align, moe],
            stream: vec![StreamEnt {
                inst: 1,
                seg: 1,
                ..Default::default()
            }],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_domains: 0,
        };
        let tensors = (0..9)
            .map(|i| DevTensor {
                name: i.to_string(),
                bytes: 256 << 20,
                init: None,
            })
            .collect();
        (prog, tensors)
    }

    fn sorted_bf16_fixture(rows: u32, bands: u32) -> (DevProg, Vec<DevTensor>) {
        let (mut prog, mut tensors) = fixture();
        prog.t = rows;
        prog.insts[0].i[0] = rows;
        prog.insts[1].i[0] = rows;
        prog.insts[1].i[6] = 2;
        tensors[0].bytes = u64::from(rows) * 6144 * 2;
        for band in 0..bands {
            prog.insts.push(DevInst64 {
                op: DevOp::MoeCombinePf as u16,
                t: [
                    8,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    0,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [6144, 1, rows / bands, band * (rows / bands), 0, 0, 0, 1],
                ..Default::default()
            });
        }
        (prog, tensors)
    }

    #[test]
    fn sorted_bf16_routes_cover_ladder_bands_and_resident_layouts() {
        for rows in [
            1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ] {
            for bands in [1, 2].into_iter().filter(|&b| b <= rows) {
                for resident in [false, true] {
                    let (mut prog, tensors) = sorted_bf16_fixture(rows, bands);
                    prog.insts[1].i[7] = u32::from(resident);
                    let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
                    assert_eq!(route.mode, Mode::SortedBf16);
                    assert_eq!(route.launches(), if resident { 2 } else { 3 });
                    route.rebase(1).unwrap();
                    route.rebase(rows).unwrap();
                    assert!(route.rebase(rows + 1).is_err());
                }
            }
        }
    }

    #[test]
    fn sorted_bf16_rejects_invalid_consumers_aliases_and_capacity() {
        for bad in 0..11 {
            let (mut prog, mut tensors) = sorted_bf16_fixture(128, 2);
            match bad {
                0 => prog.insts.truncate(2),
                1 => prog.insts[2].i[7] = 0,
                2 => prog.insts[3].i[7] = 0,
                3 => prog.insts[3].i[3] = 0,
                4 => prog.insts[3].i[2] = 65,
                5 => prog.insts[3].i[2] = 0,
                6 => prog.insts[2].i[4] = 1,
                7 => prog.insts[2].i[1] = 8,
                8 => prog.insts[1].t[0] = prog.insts[1].t[1],
                9 => tensors[0].bytes -= 1,
                _ => prog.insts[1].i[6] = 3,
            }
            assert!(routes(&prog, &tensors, 2).is_err(), "bad={bad}");
        }
        let (mut prog, tensors) = sorted_bf16_fixture(128, 1);
        let mut next_producer = prog.insts[1];
        next_producer.op = DevOp::MoeGroupDownPf as u16;
        prog.insts.push(next_producer);
        let mut next_combine = prog.insts[2];
        next_combine.i[7] = 0;
        prog.insts.push(next_combine);
        assert!(routes(&prog, &tensors, 2).is_ok());
    }

    #[test]
    fn moe_aiter_routes_require_geometry_capacity_and_isolation() {
        let (prog, tensors) = fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        route.rebase(1).unwrap();
        route.rebase(8192).unwrap();
        assert!(route.rebase(0).is_err());
        assert!(route.rebase(8193).is_err());
        for bad in 0..10 {
            let (mut p, mut t) = fixture();
            match bad {
                0 => p.t = 0,
                1 => p.insts[1].i[2] = 512,
                2 => p.insts[0].t[2] = 6,
                3 => p.insts[0].i[3] = 1,
                4 => t[0].bytes = 16,
                5 => p.stream[0].wait_len = 1,
                6 => p.stream[0].succ_len = 1,
                7 => p.stream[0].flags |= SE_XCTR,
                8 => p.stream.push(StreamEnt {
                    inst: 0,
                    seg: 1,
                    ..Default::default()
                }),
                _ => p.gq_stream.push(StreamEnt {
                    inst: 1,
                    seg: 0,
                    ..Default::default()
                }),
            }
            assert!(routes(&p, &t, 2).is_err(), "case {bad}");
        }
    }

    fn flat_fixture() -> (DevProg, Vec<DevTensor>) {
        let (mut prog, tensors) = fixture();
        prog.t = 8;
        prog.insts = vec![
            DevInst64 {
                op: DevOp::MoeRouterTopkPf as u16,
                t: [
                    4,
                    8,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [0, 256, 8, 0, 8, 0, 0, 0],
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::MoeAiterFp8Pf as u16,
                t: [0, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
                i: [8, 6144, 256, 256, 8, 0, 1, 0],
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::MoeCombinePf as u16,
                t: [
                    8,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    0,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                i: [6144, 1, 8, 0, 0, 0, 0, 1],
                ..Default::default()
            },
        ];
        (prog, tensors)
    }

    #[test]
    fn flat_route_requires_raw_routing_bf16_consumer_and_protocol_space() {
        let (prog, tensors) = flat_fixture();
        let mut route = routes(&prog, &tensors, 2).unwrap()[1].unwrap();
        assert_eq!(route.mode, Mode::Flat);
        assert_eq!(route.launches(), 2);
        route.rebase(3).unwrap();
        assert!(route.rebase(9).is_err());
        for bad in 0..8 {
            let (mut p, mut t) = flat_fixture();
            match bad {
                0 => p.insts[0].i[4] = 7,
                1 => p.insts[2].i[7] = 0,
                2 => p.insts[2].i[1] = 8,
                3 => p.insts[2].i[4] = 1,
                4 => t[0].bytes = 8 * 6144 * 2 + 7,
                5 => p.insts[1].t[7] = 0,
                6 => p.insts[1].i[6] = 0,
                _ => p.stream[0].succ_len = 1,
            }
            assert!(routes(&p, &t, 2).is_err(), "case {bad}");
        }
    }

    #[test]
    fn resident_tables_reject_incompatible_consumers_across_programs() {
        let (mut prog, mut tensors) = flat_fixture();
        prog.insts[1].i[7] = 1;
        tensors[2].name = "layer.3.expert_weight_table".into();
        tensors[3].name = "layer.3.expert_scale_table".into();
        tensors[2].bytes = 256 * 24;
        tensors[3].bytes = 256 * 24;
        let (mut other, _) = flat_fixture();
        other.insts.clear();
        let mut progs = vec![prog, other];
        assert_eq!(resident_tables(&progs, &tensors).unwrap()[2], Some(3));
        let compatible = progs[0].insts[1];
        progs[0].insts.push(compatible);
        assert!(resident_tables(&progs, &tensors).is_ok());
        for bad in 0..5 {
            let mut inst = compatible;
            match bad {
                0 => inst.i[7] = 0,
                1 => inst.op = DevOp::MoeGroupGluPf as u16,
                2 => inst.t[3] = 4,
                3 => inst.t[0] = 2,
                _ => inst.i[7] = 2,
            }
            progs[1].insts.push(inst);
            assert!(resident_tables(&progs, &tensors).is_err(), "case {bad}");
            progs[1].insts.pop();
        }
        tensors[4].name = "layer.3.expert_weight_table_pf".into();
        assert!(resident_tables(&progs, &tensors).is_err());
    }

    #[test]
    fn resident_routes_cover_all_decode_rungs() {
        for rows in [1, 2, 4, 8, 16, 20] {
            let (mut p, mut t) = flat_fixture();
            p.t = rows;
            p.insts[0].i[4] = rows;
            p.insts[1].i[0] = rows;
            p.insts[1].i[7] = 1;
            p.insts[2].i[2] = rows;
            t[0].bytes = u64::from(rows) * 6144 * 2 + 8;
            t[1].bytes = u64::from(rows) * 6144 * 2;
            t[4].bytes = u64::from(rows) * 8 * 8;
            assert_eq!(routes(&p, &t, 2).unwrap()[1].unwrap().launches(), 2);
        }
        let (mut p, t) = fixture();
        p.insts[1].i[7] = 1;
        assert_eq!(routes(&p, &t, 2).unwrap()[1].unwrap().launches(), 3);
    }

    #[test]
    fn resident_table_addresses_fill_one_allocation() {
        let base = 4096;
        let stride = 256 * 6144;
        let table = resident_expert_table(base, stride);
        let mut offsets = table
            .iter()
            .map(|p| (*p - base) / stride)
            .collect::<Vec<_>>();
        offsets.sort_unstable();
        assert_eq!(offsets, (0..768).collect::<Vec<_>>());
        let w = ResidentWeights::new(base, 8192);
        assert_eq!(table[0], w.gu);
        assert_eq!(table[2], w.down);
        for e in 0..256 {
            assert_eq!(table[e * 3], w.gu + e as u64 * 2 * stride);
            assert_eq!(table[e * 3 + 1], table[e * 3] + stride);
            assert_eq!(table[e * 3 + 2], w.down + e as u64 * stride);
        }
        assert_eq!(resident_expert_table(8192, 384)[2], w.ds);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with flat objects"]
    fn moe_aiter_flat_hsa_dispatch() {
        check_flat_hsa(false);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with resident adapter"]
    fn moe_aiter_resident_flat_hsa_dispatch() {
        check_flat_hsa(true);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR"]
    fn moe_aiter_resident_packing_matches_gpu() {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let kernel = MoeAiter::load(&be, Path::new(&dir), 128, false, false, false, &mut modules).unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let mut projections = Vec::new();
        let mut expected = Vec::new();
        let mut scale_buffers = Vec::new();
        let mut scale_expected = Vec::new();
        for (j, (rows, k)) in [(256usize, 6144usize), (256, 6144), (6144, 256)]
            .into_iter()
            .enumerate()
        {
            let src: Vec<u8> = (0..rows * k)
                .map(|i| {
                    let b = ((i * 31 + i / 251 + j * 71) % 256) as u8;
                    if b == 0x80 {
                        0
                    } else {
                        b
                    }
                })
                .collect();
            expected.push(super::super::amd::shuffle_moe_weight_16x32(&src, rows, k).unwrap());
            projections.push(upload(&src));
            let scales: Vec<f32> = (0..96).map(|i| (i + j * 96 + 1) as f32 * 0.001).collect();
            scale_expected.push(
                super::super::amd::resident_moe_scales(bytemuck::cast_slice(&scales)).unwrap(),
            );
            scale_buffers.push(upload(bytemuck::cast_slice(&scales)));
        }
        let wt: Vec<u64> = (0..768).map(|i| projections[i % 3].base).collect();
        let st: Vec<u64> = (0..768).map(|i| scale_buffers[i % 3].base).collect();
        let wt = upload(bytemuck::cast_slice(&wt));
        let st = upload(bytemuck::cast_slice(&st));
        let [gu, down, gs, ds, ..] = kernel.buffers;
        be.launch(
            kernel.pack,
            2048,
            256,
            0,
            bytemuck::cast_slice(&[gu, down, gs, ds, wt.base, st.base]),
        )
        .unwrap();
        be.synchronize().unwrap();
        let wtab = resident_expert_table(gu, 256 * 6144);
        let stab = resident_expert_table(gs, 96 * 4);
        assert_eq!(wtab[2], down);
        assert_eq!(stab[2], ds);
        let mut actual = vec![0u8; 256 * 6144];
        for e in 0..256 {
            for j in 0..3 {
                let offset = wtab[e * 3 + j] - kernel._scratch.base;
                EngineDevice::download(&be, &kernel._scratch, offset, &mut actual).unwrap();
                assert!(
                    actual == expected[j],
                    "expert {e} projection {j} weight mismatch"
                );
                let mut scales = vec![0u8; 96 * 4];
                let offset = stab[e * 3 + j] - kernel._scratch.base;
                EngineDevice::download(&be, &kernel._scratch, offset, &mut scales).unwrap();
                assert!(
                    scales == scale_expected[j],
                    "expert {e} projection {j} scale mismatch"
                );
            }
        }
    }

    fn bind_uniform_resident(be: &HsaBackend, kernel: &mut MoeAiter) -> Vec<DeviceMem> {
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(be, bytes.len() as u64).unwrap();
            EngineDevice::upload(be, &m, 0, bytes).unwrap();
            m
        };
        let weights = upload(&vec![0x38; 256 * 3 * 256 * 6144]);
        let mut native_scales = vec![0f32; 256 * 3 * 96];
        for e in 0..256 {
            let scale = 2.0 * (e % 8 + 1) as f32 * 0.001;
            native_scales[e * 192..(e + 1) * 192].fill(scale);
            native_scales[256 * 192 + e * 96..256 * 192 + (e + 1) * 96].fill(scale);
        }
        let scales = upload(bytemuck::cast_slice(&native_scales));
        kernel.bind_resident(vec![(2, ResidentWeights::new(weights.base, scales.base))]);
        vec![weights, scales]
    }

    fn check_flat_hsa(resident: bool) {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let mut kernel =
            MoeAiter::load(&be, Path::new(&dir), 128, true, resident, false, &mut modules).unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let weight = upload(&vec![0x38; 256 * 6144]);
        let scales: Vec<f32> = (0..256 * 96)
            .map(|i| (1 + (i / 96) % 8) as f32 * 0.001)
            .collect();
        let scale = upload(bytemuck::cast_slice(&scales));
        let wt = upload(bytemuck::cast_slice(&vec![weight.base; 256 * 3]));
        let scale_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| scale.base + (i / 3 * 96 * 4) as u64)
            .collect();
        let st = upload(bytemuck::cast_slice(&scale_ptrs));
        let _resident_buffers = if resident {
            bind_uniform_resident(&be, &mut kernel)
        } else {
            vec![]
        };
        let out = upload(&vec![0xa5; 20 * 6144 * 2 + 8 + 512]);
        let rungs: &[u32] = if resident {
            &[20, 16, 8, 4, 2, 1, 20, 3]
        } else {
            &[8, 3, 2, 1, 4, 8]
        };
        for (iteration, &rows) in rungs.iter().enumerate() {
            let x: Vec<u16> = (0..rows as usize * 6144)
                .map(|i| [0x3f80, 0x3f00, 0x3e80][i / 6144 % 3])
                .collect();
            let mut routes = Vec::new();
            let mut expected = Vec::new();
            for row in 0..rows as usize {
                let activation = f32::from_bits(u32::from(x[row * 6144]) << 16);
                let mut sum = 0.0f32;
                for slot in 0..8 {
                    let expert = (iteration * 37 + row * 11 + slot) % 256;
                    let gate = (slot + 1) as f32 / 36.0;
                    routes.extend([expert as u32, gate.to_bits()]);
                    let scale = (expert % 8 + 1) as f32 * 0.001;
                    let gu = 6144.0 * activation * scale;
                    sum += gu * gu / (1.0 + (-gu).exp()) * 256.0 * scale * gate;
                }
                expected.push(sum);
            }
            let x = upload(bytemuck::cast_slice(&x));
            let routes = upload(bytemuck::cast_slice(&routes));
            let poison = vec![0xa5; out.len as usize];
            EngineDevice::upload(&be, &out, 0, &poison).unwrap();
            let table = [out.base, x.base, wt.base, st.base, routes.base];
            let route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
                    i: [rows, 6144, 256, 256, 8, 0, 1, u32::from(resident)],
                    ..Default::default()
                },
                rows,
                mode: Mode::Flat,
            };
            kernel
                .enqueue(&be, route, bytemuck::cast_slice(&table))
                .unwrap();
            be.synchronize().unwrap();
            let mut actual = poison.clone();
            EngineDevice::download(&be, &out, 0, &mut actual).unwrap();
            let end = rows as usize * 6144 * 2;
            assert_eq!(&actual[end + 8..], &poison[end + 8..]);
            for (i, bits) in actual[..end].chunks_exact(2).enumerate() {
                let value = f32::from_bits(u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16);
                let want = expected[i / 6144];
                assert!(
                    value.is_finite() && (value / want - 1.0).abs() < 0.04,
                    "iteration={iteration} rows={rows} element={i} expected={want} actual={value}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR"]
    fn moe_aiter_hsa_dispatch() {
        check_sorted_hsa(false, false, false);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with resident adapter"]
    fn moe_aiter_resident_sorted_hsa_dispatch() {
        check_sorted_hsa(true, false, false);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR"]
    fn moe_aiter_sorted_bf16_hsa_dispatch() {
        check_sorted_hsa(false, true, false);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with resident adapter"]
    fn moe_aiter_resident_sorted_bf16_hsa_dispatch() {
        check_sorted_hsa(true, true, false);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the 64-row object"]
    fn moe_aiter_tile64_hsa_dispatch() {
        check_sorted_hsa(false, false, true);
    }

    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the 64-row object"]
    fn moe_aiter_tile64_bf16_hsa_dispatch() {
        check_sorted_hsa(false, true, true);
    }

    /// Random top-8 routing over 256 experts in the align pass's layout: meta (row offset /
    /// count / 64-row group per expert, total groups at 768), padded row_token / row_part /
    /// row_weight, and each row's (expert, gate) list. Every row picks 8 distinct experts
    /// spread over all 256; group counts vary.
    type Routing = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<Vec<(usize, f32)>>);
    fn random_routing(rows: u32, next: &mut impl FnMut() -> u64) -> Routing {
        let mut per_expert: Vec<Vec<(u32, u32, f32)>> = vec![Vec::new(); 256];
        let mut row_experts: Vec<Vec<(usize, f32)>> = vec![Vec::new(); rows as usize];
        for row in 0..rows {
            let base = (next() % 256) as u32;
            for k in 0..8u32 {
                let e = (base + k * 37 + (next() % 5) as u32 * 3) % 256;
                let e = (0..256u32)
                    .map(|d| (e + d) % 256)
                    .find(|c| !per_expert[*c as usize].iter().any(|(r, _, _)| *r == row))
                    .unwrap();
                let gate = 0.05 + (k as f32) * 0.02;
                per_expert[e as usize].push((row, row * 8 + k, gate));
                row_experts[row as usize].push((e as usize, gate));
            }
        }
        let mut meta = vec![0u32; 769];
        let mut rt = Vec::new();
        let mut rp = Vec::new();
        let mut rg = Vec::new();
        let mut groups = 0u32;
        for e in 0..256 {
            meta[e] = rt.len() as u32;
            meta[e + 256] = per_expert[e].len() as u32;
            meta[e + 512] = groups;
            let padded = (per_expert[e].len() as u32).div_ceil(64) * 64;
            groups += padded / 64;
            for i in 0..padded as usize {
                let (t, p, g) = per_expert[e]
                    .get(i)
                    .copied()
                    .unwrap_or((u32::MAX, u32::MAX, 0.0));
                rt.push(t);
                rp.push(p);
                rg.push(g);
            }
        }
        meta[768] = groups;
        (meta, rt, rp, rg, row_experts)
    }

    #[test]
    fn xcd_swizzle_is_a_same_xcd_block_permutation() {
        for (block_m, blocks) in [(64u32, 1024u32), (64, 1030), (32, 2048), (32, 2059), (64, 7)] {
            let k = 256 / block_m;
            let mut seen = vec![false; blocks as usize];
            for p in 0..blocks {
                let b = xcd_swizzled_block(p, block_m, blocks);
                assert!(!std::mem::replace(&mut seen[b as usize], true), "{p} -> {b} twice");
                // The k consecutive blocks starting at a k-aligned block share p % 8 and lie
                // within one 8k chunk, unless the chunk is the partial tail (identity).
                let chunk = 8 * k;
                if (b / chunk + 1) * chunk <= blocks {
                    let first = (0..blocks)
                        .find(|q| xcd_swizzled_block(*q, block_m, blocks) == b / k * k)
                        .unwrap();
                    assert_eq!(first % 8, p % 8, "block {b} left XCD {}", first % 8);
                    assert!(p.abs_diff(first) < chunk);
                } else {
                    assert_eq!(b, p);
                }
            }
            assert!(seen.iter().all(|s| *s), "not a permutation at {block_m}/{blocks}");
        }
    }

    /// The swizzled launch order is the host mirror's permutation of the same blocks (ids and
    /// expert per position), its output matches the unswizzled order up to accumulation
    /// order, and the route's stages are timed one at a time (warm host medians; the split
    /// the serving trace cannot see inside a native segment). Not a serving measurement.
    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the 64-row and swizzle adapter"]
    fn moe_aiter_xcd_swizzle_matches_and_splits() {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let dir = Path::new(&dir);
        let mut k32 = MoeAiter::load(&be, dir, 8192, false, false, false, &mut modules).unwrap();
        let mut k64 = MoeAiter::load(&be, dir, 8192, false, false, true, &mut modules).unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let weight_bytes: Vec<u8> = (0..3 * 256 * 256 * 6144usize)
            .map(|_| [0x38u8, 0xb8, 0x40, 0xc0, 0x44, 0xc4, 0x00, 0x48][(next() % 8) as usize])
            .collect();
        let weight = upload(&weight_bytes);
        let scales: Vec<f32> = (0..3 * 256 * 96)
            .map(|_| 0.002 * (1.0 + (next() % 1000) as f32 / 1000.0))
            .collect();
        let scale = upload(bytemuck::cast_slice(&scales));
        let wt_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| weight.base + (i * 256 * 6144) as u64)
            .collect();
        let wt = upload(bytemuck::cast_slice(&wt_ptrs));
        let scale_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| scale.base + (i * 96 * 4) as u64)
            .collect();
        let st = upload(bytemuck::cast_slice(&scale_ptrs));
        let rows_list: Vec<u32> = std::env::var("PLOW_TEST_MOE_ROWS")
            .ok()
            .map(|s| s.split(',').map(|r| r.parse().unwrap()).collect())
            .unwrap_or_else(|| vec![2048, 8192]);
        for rows in rows_list {
            let (meta, rt, rp, rg, _) = random_routing(rows, &mut next);
            let x_bits: Vec<u16> = (0..rows as usize * 6144)
                .map(|_| {
                    let v = (next() % 2001) as f32 / 1000.0 - 1.0;
                    (v.to_bits() >> 16) as u16
                })
                .collect();
            let x = upload(bytemuck::cast_slice(&x_bits));
            let output_bytes = rows as usize * 6144 * 4;
            let out = upload(&vec![0xa5; output_bytes + 512]);
            let meta_dev = upload(bytemuck::cast_slice(&meta));
            let rt_dev = upload(bytemuck::cast_slice(&rt));
            let rp_dev = upload(bytemuck::cast_slice(&rp));
            let rg_dev = upload(bytemuck::cast_slice(&rg));
            let table = [
                out.base + 256,
                x.base,
                wt.base,
                st.base,
                meta_dev.base,
                rt_dev.base,
                rp_dev.base,
                rg_dev.base,
            ];
            let route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, 5, 6, 7],
                    i: [rows, 6144, 256, 256, 8, 64, 0, 0],
                    ..Default::default()
                },
                rows,
                mode: Mode::Sorted,
            };
            let padded = meta[768] as usize * 64;
            for (kernel, block_m) in [(&mut k32, 32u32), (&mut k64, 64)] {
                let blocks = padded as u32 / block_m;
                let mut outputs = Vec::new();
                for swizzle in [false, true] {
                    kernel.set_xcd_swizzle(Some(swizzle)).unwrap();
                    EngineDevice::upload(&be, &out, 0, &vec![0xa5; output_bytes + 512]).unwrap();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.synchronize().unwrap();
                    // Launch order: position p carries block xcd_swizzled_block(p)'s rows and
                    // expert; the psx (64-row) object is never swizzled.
                    let swizzled = swizzle && block_m == 32;
                    let mut ids = vec![0u8; padded * 4];
                    let mut experts = vec![0u8; blocks as usize * 4];
                    let scratch = &kernel._scratch;
                    EngineDevice::download(&be, scratch, kernel.buffers[7] - scratch.base, &mut ids)
                        .unwrap();
                    EngineDevice::download(
                        &be,
                        scratch,
                        kernel.buffers[9] - scratch.base,
                        &mut experts,
                    )
                    .unwrap();
                    let ids: &[u32] = bytemuck::cast_slice(&ids);
                    let experts: &[u32] = bytemuck::cast_slice(&experts);
                    for p in 0..blocks {
                        let b = if swizzled { xcd_swizzled_block(p, block_m, blocks) } else { p };
                        let group = (b * block_m / 64) as usize;
                        let expert = (0..256).find(|e| meta[512 + e + 1] as usize > group).unwrap();
                        assert_eq!(experts[p as usize], expert as u32, "block_m={block_m} p={p}");
                        for r in 0..block_m as usize {
                            let (pos, src) = (p as usize * block_m as usize + r, b as usize * block_m as usize + r);
                            let expect = if rt[src] < rows {
                                rt[src] | ((rp[src] % 8) << 24)
                            } else {
                                rows
                            };
                            assert_eq!(ids[pos], expect, "block_m={block_m} p={p} r={r}");
                        }
                    }
                    let mut bytes = vec![0u8; output_bytes + 512];
                    EngineDevice::download(&be, &out, 0, &mut bytes).unwrap();
                    assert!(bytes[..256]
                        .iter()
                        .chain(&bytes[256 + output_bytes..])
                        .all(|&x| x == 0xa5));
                    let values: Vec<f32> =
                        bytemuck::cast_slice(&bytes[256..256 + output_bytes]).to_vec();
                    assert!(values.iter().all(|v| v.is_finite()));
                    let median = |stages: u8| -> f64 {
                        for _ in 0..3 {
                            kernel
                                .enqueue_stages(&be, route, bytemuck::cast_slice(&table), stages)
                                .unwrap();
                        }
                        be.synchronize().unwrap();
                        let mut samples = Vec::new();
                        for _ in 0..15 {
                            let start = std::time::Instant::now();
                            kernel
                                .enqueue_stages(&be, route, bytemuck::cast_slice(&table), stages)
                                .unwrap();
                            be.synchronize().unwrap();
                            samples.push(start.elapsed().as_secs_f64() * 1e6);
                        }
                        samples.sort_by(|a, b| a.total_cmp(b));
                        samples[samples.len() / 2]
                    };
                    let (all, pack, prepare, moe, store) = (
                        median(!0),
                        median(STAGE_PACK),
                        median(STAGE_PREPARE),
                        median(STAGE_MOE),
                        median(STAGE_STORE),
                    );
                    eprintln!(
                        "rows={rows} tile{block_m} swizzle={swizzle}: route={all:.1}us pack={pack:.1} prepare={prepare:.1} fmoe={moe:.1} store={store:.1} (serving route = prepare + fmoe)"
                    );
                    outputs.push(values);
                }
                let (mut num, mut den) = (0f64, 0f64);
                for (a, b) in outputs[0].iter().zip(&outputs[1]) {
                    num += (f64::from(*a) - f64::from(*b)).powi(2);
                    den += f64::from(*a).powi(2);
                }
                let rel = (num / den).sqrt();
                eprintln!("rows={rows} tile{block_m}: swizzled vs unswizzled rel_l2={rel:.3e}");
                assert!(rel < 5e-3, "rows={rows} tile{block_m}: swizzle changed the result");
            }
        }
    }

    /// Same sorted inputs through the 32x256 and the 64x256 objects: both consume the identical
    /// A8 quantization from `plow_moe_aiter_prepare`, so they may differ only by accumulation
    /// order. Also prints warm medians for the single-GPU A/B; not a serving measurement.
    #[test]
    #[ignore = "requires a gfx942 GPU lease and PLOW_TEST_AITER_DIR with the 64-row object"]
    fn moe_aiter_tile64_matches_tile32() {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let dir = Path::new(&dir);
        let k32 = MoeAiter::load(&be, dir, 8192, false, false, false, &mut modules).unwrap();
        let k64 = MoeAiter::load(&be, dir, 8192, false, false, true, &mut modules).unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Signed FP8 integers in [-8, 8] (exact in e4m3) and per-block scales that differ.
        let fp8 = |v: i32| -> u8 {
            let magnitude: u8 = match v.unsigned_abs() {
                0 => 0,
                1 => 0x38,
                2 => 0x40,
                3 => 0x44,
                4 => 0x48,
                5 => 0x4a,
                6 => 0x4c,
                7 => 0x4e,
                _ => 0x50,
            };
            magnitude | if v < 0 { 0x80 } else { 0 }
        };
        let weight_bytes: Vec<u8> = (0..3 * 256 * 256 * 6144usize)
            .map(|_| fp8((next() % 17) as i32 - 8))
            .collect();
        let weight = upload(&weight_bytes);
        let scales: Vec<f32> = (0..3 * 256 * 96)
            .map(|_| 0.002 * (1.0 + (next() % 1000) as f32 / 1000.0))
            .collect();
        let scale = upload(bytemuck::cast_slice(&scales));
        let wt_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| weight.base + (i * 256 * 6144) as u64)
            .collect();
        let wt = upload(bytemuck::cast_slice(&wt_ptrs));
        let scale_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| scale.base + (i * 96 * 4) as u64)
            .collect();
        let st = upload(bytemuck::cast_slice(&scale_ptrs));
        for rows in [1024u32, 2048, 4464, 8192] {
            let (meta, rt, rp, rg, row_experts) = random_routing(rows, &mut next);
            let x_bits: Vec<u16> = (0..rows as usize * 6144)
                .map(|_| {
                    let v = (next() % 2001) as f32 / 1000.0 - 1.0;
                    (v.to_bits() >> 16) as u16
                })
                .collect();
            let x = upload(bytemuck::cast_slice(&x_bits));
            let output_bytes = rows as usize * 6144 * 4;
            let out = upload(&vec![0xa5; output_bytes + 512]);
            let meta = upload(bytemuck::cast_slice(&meta));
            let rt = upload(bytemuck::cast_slice(&rt));
            let rp = upload(bytemuck::cast_slice(&rp));
            let rg = upload(bytemuck::cast_slice(&rg));
            let table = [
                out.base + 256,
                x.base,
                wt.base,
                st.base,
                meta.base,
                rt.base,
                rp.base,
                rg.base,
            ];
            let route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, 5, 6, 7],
                    i: [rows, 6144, 256, 256, 8, 64, 0, 0],
                    ..Default::default()
                },
                rows,
                mode: Mode::Sorted,
            };
            let mut results = Vec::new();
            for (kernel, tile) in [(&k32, 32u32), (&k64, 64)] {
                EngineDevice::upload(&be, &out, 0, &vec![0xa5; output_bytes + 512]).unwrap();
                kernel
                    .enqueue(&be, route, bytemuck::cast_slice(&table))
                    .unwrap();
                be.synchronize().unwrap();
                let mut bytes = vec![0u8; output_bytes + 512];
                EngineDevice::download(&be, &out, 0, &mut bytes).unwrap();
                assert!(
                    bytes[..256]
                        .iter()
                        .chain(&bytes[256 + output_bytes..])
                        .all(|&x| x == 0xa5),
                    "tile{tile} rows={rows}: guard bytes overwritten"
                );
                let values: Vec<f32> =
                    bytemuck::cast_slice(&bytes[256..256 + output_bytes]).to_vec();
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "tile{tile} rows={rows}: non-finite"
                );
                let mut samples = Vec::new();
                for _ in 0..3 {
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                }
                be.synchronize().unwrap();
                for _ in 0..15 {
                    let start = std::time::Instant::now();
                    kernel
                        .enqueue(&be, route, bytemuck::cast_slice(&table))
                        .unwrap();
                    be.synchronize().unwrap();
                    samples.push(start.elapsed().as_secs_f64() * 1e6);
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                results.push((tile, values, samples[samples.len() / 2]));
            }
            // FP64 reference on sampled rows from the dequantized weights and the BF16 input;
            // the A8 activation quantization inside both objects is the only unmodelled term.
            let decode = |byte: u8| -> f64 {
                let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
                let exp = i32::from((byte >> 3) & 0xf);
                let man = f64::from(byte & 7) / 8.0;
                if exp == 0 {
                    sign * man * 2f64.powi(-6)
                } else {
                    sign * (1.0 + man) * 2f64.powi(exp - 7)
                }
            };
            let picks = [0usize, rows as usize / 2, rows as usize - 1];
            let mut reference = vec![0f64; picks.len() * 6144];
            for (j, &row) in picks.iter().enumerate() {
                let xr: Vec<f64> = x_bits[row * 6144..(row + 1) * 6144]
                    .iter()
                    .map(|b| f64::from(f32::from_bits(u32::from(*b) << 16)))
                    .collect();
                for &(e, gate_w) in &row_experts[row] {
                    let mut act = [0f64; 256];
                    for n in 0..256 {
                        let mut gate = 0f64;
                        let mut up = 0f64;
                        for k in 0..6144 {
                            let sg = f64::from(scales[(e * 3) * 96 + (n / 128) * 48 + k / 128]);
                            let su = f64::from(scales[(e * 3 + 1) * 96 + (n / 128) * 48 + k / 128]);
                            gate += decode(weight_bytes[(e * 3) * 256 * 6144 + n * 6144 + k])
                                * sg
                                * xr[k];
                            up += decode(weight_bytes[(e * 3 + 1) * 256 * 6144 + n * 6144 + k])
                                * su
                                * xr[k];
                        }
                        act[n] = gate / (1.0 + (-gate).exp()) * up;
                    }
                    for n in 0..6144 {
                        let mut acc = 0f64;
                        for k in 0..256 {
                            let sd = f64::from(scales[(e * 3 + 2) * 96 + (n / 128) * 2 + k / 128]);
                            acc += decode(weight_bytes[(e * 3 + 2) * 256 * 6144 + n * 256 + k])
                                * sd
                                * act[k];
                        }
                        reference[j * 6144 + n] += acc * f64::from(gate_w);
                    }
                }
            }
            let rel = |values: &[f32]| -> f64 {
                let (mut num, mut den) = (0f64, 0f64);
                for (j, &row) in picks.iter().enumerate() {
                    for n in 0..6144 {
                        let r = reference[j * 6144 + n];
                        num += (f64::from(values[row * 6144 + n]) - r).powi(2);
                        den += r.powi(2);
                    }
                }
                (num / den).sqrt()
            };
            let (a, b) = (&results[0].1, &results[1].1);
            let (mut num, mut den) = (0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                num += (f64::from(*x) - f64::from(*y)).powi(2);
                den += f64::from(*x).powi(2);
            }
            let (rel32, rel64, rel_pair) = (rel(a), rel(b), (num / den).sqrt());
            eprintln!(
                "rows={rows} tile32={:.1}us tile64={:.1}us rel_l2 vs fp64: tile32={rel32:.3e} tile64={rel64:.3e}; tile64 vs tile32={rel_pair:.3e}",
                results[0].2, results[1].2
            );
            assert!(
                rel32 < 0.1 && rel64 < 0.1,
                "rows={rows}: A8 screening threshold exceeded"
            );
            assert!(
                rel64 <= rel32 * 1.25 + 1e-3,
                "rows={rows}: tile64 is less accurate than tile32"
            );
        }
    }

    fn check_sorted_hsa(resident: bool, part16: bool, tile64: bool) {
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let capacity = if tile64 { 2048 } else { 129 };
        let mut kernel = MoeAiter::load(
            &be,
            Path::new(&dir),
            capacity,
            false,
            resident,
            tile64,
            &mut modules,
        )
        .unwrap();
        let upload = |bytes: &[u8]| {
            let m = EngineDevice::alloc(&be, bytes.len() as u64).unwrap();
            EngineDevice::upload(&be, &m, 0, bytes).unwrap();
            m
        };
        let weight = upload(&vec![0x38; 256 * 6144]);
        let scales: Vec<f32> = (0..256 * 96)
            .map(|i| (1 + (i / 96) % 8) as f32 * 0.001)
            .collect();
        let scale = upload(bytemuck::cast_slice(&scales));
        let wt = upload(bytemuck::cast_slice(&vec![weight.base; 256 * 3]));
        let scale_ptrs: Vec<u64> = (0..256 * 3)
            .map(|i| scale.base + (i / 3 * 96 * 4) as u64)
            .collect();
        let st = upload(bytemuck::cast_slice(&scale_ptrs));
        let _resident_buffers = if resident {
            bind_uniform_resident(&be, &mut kernel)
        } else {
            vec![]
        };
        let cases: &[u32] = if tile64 {
            &[1024, 1025, 2048, 129]
        } else {
            &[1, 129, 1]
        };
        for &rows in cases {
            let stride = rows.div_ceil(64) * 64;
            let mut meta = vec![0u32; 769];
            let mut rt = vec![u32::MAX; 8 * stride as usize];
            let mut rp = rt.clone();
            let mut rg = vec![0f32; rt.len()];
            for e in 0..256 {
                meta[e] = e.min(8) as u32 * stride;
                meta[e + 256] = if e < 8 { rows } else { 0 };
                meta[e + 512] = e.min(8) as u32 * stride / 64;
                if e < 8 {
                    for row in 0..rows as usize {
                        let i = e * stride as usize + row;
                        rt[i] = row as u32;
                        rp[i] = row as u32 * 8 + e as u32;
                        rg[i] = (e + 1) as f32 / 36.0;
                    }
                }
            }
            meta[768] = 8 * stride / 64;
            let x = upload(bytemuck::cast_slice(&vec![0x3f80u16; rows as usize * 6144]));
            let element_bytes = if part16 { 2 } else { 4 };
            let output_bytes = rows as usize * 6144 * element_bytes;
            let out = upload(&vec![0xa5; output_bytes + 512]);
            let meta = upload(bytemuck::cast_slice(&meta));
            let rt = upload(bytemuck::cast_slice(&rt));
            let rp = upload(bytemuck::cast_slice(&rp));
            let rg = upload(bytemuck::cast_slice(&rg));
            let table = [
                out.base + 256,
                x.base,
                wt.base,
                st.base,
                meta.base,
                rt.base,
                rp.base,
                rg.base,
            ];
            let route = Route {
                inst: DevInst64 {
                    t: [0, 1, 2, 3, 4, 5, 6, 7],
                    i: [
                        rows,
                        6144,
                        256,
                        256,
                        8,
                        64,
                        if part16 { 2 } else { 0 },
                        u32::from(resident),
                    ],
                    ..Default::default()
                },
                rows,
                mode: if part16 {
                    Mode::SortedBf16
                } else {
                    Mode::Sorted
                },
            };
            kernel
                .enqueue(&be, route, bytemuck::cast_slice(&table))
                .unwrap();
            be.synchronize().unwrap();
            let expected: f32 = (1..=8)
                .map(|e| {
                    let scale = e as f32 * 0.001;
                    let gu = 6144.0 * scale;
                    gu * gu / (1.0 + (-gu).exp()) * 256.0 * scale * e as f32 / 36.0
                })
                .sum();
            let mut bytes = vec![0u8; output_bytes + 512];
            EngineDevice::download(&be, &out, 0, &mut bytes).unwrap();
            assert!(bytes[..256]
                .iter()
                .chain(&bytes[256 + output_bytes..])
                .all(|&x| x == 0xa5));
            let actual: Vec<f32> = bytes[256..256 + output_bytes]
                .chunks_exact(element_bytes)
                .map(|x| {
                    if part16 {
                        f32::from_bits(u32::from(u16::from_le_bytes(x.try_into().unwrap())) << 16)
                    } else {
                        f32::from_le_bytes(x.try_into().unwrap())
                    }
                })
                .collect();
            assert!(
                actual
                    .iter()
                    .all(|x| x.is_finite() && (x / expected - 1.0).abs() < 0.04),
                "rows={rows} expected={expected} first={}",
                actual[0]
            );
        }
    }
}
