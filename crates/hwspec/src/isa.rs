//! ISA level and hardware fingerprint.
//!
//! [`Arch`](crate::Arch) answers "which microarchitecture generation", which is
//! the granularity the cost model's tile families need. It is **too coarse to
//! select a kernel**: `Arch::Blackwell` covers both B200 (`sm_100a`, `tcgen05`,
//! TMEM) and RTX 5090 / RTX PRO 6000 (`sm_120a`, `mma.sync`, no TMEM). Its own
//! doc comment claims "SM 10.0, wgmma" for all three, which is wrong for two of
//! them. A selector that trusts `Arch` alone will offer a consumer part
//! instructions its silicon does not have.
//!
//! [`IsaLevel`] is that missing distinction, and [`IsaCaps`] states what the
//! level can actually execute. Selection predicates are written against
//! capabilities, never against a SKU name — so adding hardware means adding a
//! level and its capabilities, not editing selection logic.
//!
//! [`HardwareFingerprint`] is the key a measurement is stored under. It
//! deliberately carries more than the silicon: the same GPU under a different
//! driver, toolchain, or clock policy is a different measurement population,
//! and mixing them is how a tuning database starts lying.

use crate::spec::{GpuSpec, MmaDtype, Vendor};

/// The instruction-set level a kernel is compiled for.
///
/// This is the `-arch=` / `--offload-arch=` the build actually passes, not a
/// marketing generation. Two SKUs share a level only if a cubin/hsaco built for
/// one runs on the other with the same instruction availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IsaLevel {
    /// NVIDIA Ada. `mma.sync m16n8k16`, `cp.async`. No TMA, no wgmma.
    Sm89,
    /// NVIDIA Hopper. `wgmma`, TMA (`cp.async.bulk` + mbarrier), DSM clusters.
    Sm90a,
    /// NVIDIA Blackwell datacenter (B200). `tcgen05`, TMEM, block-scaled MMA.
    Sm100a,
    /// NVIDIA Blackwell consumer (RTX 5090, RTX PRO 6000). 5th-gen tensor cores
    /// via `mma.sync`; **no** `tcgen05`, **no** TMEM. Sharing the "Blackwell"
    /// name with [`Self::Sm100a`] does not share its instructions.
    Sm120a,
    /// AMD CDNA 3 (MI300X/MI325X). 64 KiB LDS, `v_mfma_f32_32x32x8_bf16`.
    /// See [`IsaLevel::geometry`] for the tile and stage-buffer policy that
    /// budget forces.
    Gfx942,
    /// AMD CDNA 4 (MI350X/MI355X). 160 KiB LDS, double-K bf16 MFMA.
    Gfx950,
    /// Portable scalar reference. Always available; never fast.
    CpuRef,
}

impl IsaLevel {
    /// The `-arch` / `--offload-arch` string a compiler expects.
    pub fn arch_flag(self) -> &'static str {
        match self {
            IsaLevel::Sm89 => "sm_89",
            IsaLevel::Sm90a => "sm_90a",
            IsaLevel::Sm100a => "sm_100a",
            IsaLevel::Sm120a => "sm_120a",
            IsaLevel::Gfx942 => "gfx942",
            IsaLevel::Gfx950 => "gfx950",
            IsaLevel::CpuRef => "cpu",
        }
    }

    pub fn vendor(self) -> Vendor {
        match self {
            IsaLevel::Sm89 | IsaLevel::Sm90a | IsaLevel::Sm100a | IsaLevel::Sm120a => {
                Vendor::Nvidia
            }
            IsaLevel::Gfx942 | IsaLevel::Gfx950 => Vendor::Amd,
            // The CPU reference has no vendor in the GPU sense; it is reported as
            // NVIDIA only because `Vendor` has no third variant. Callers should
            // branch on `IsaLevel`, which is why this is the one lossy mapping.
            IsaLevel::CpuRef => Vendor::Nvidia,
        }
    }

    /// Derive the level from a spec. This is where the `Arch::Blackwell`
    /// conflation is resolved: the compute capability, not the generation name,
    /// decides.
    pub fn from_spec(spec: &GpuSpec) -> Option<Self> {
        Some(match (spec.vendor, spec.compute_cap) {
            (Vendor::Nvidia, (8, 9)) => IsaLevel::Sm89,
            (Vendor::Nvidia, (9, 0)) => IsaLevel::Sm90a,
            (Vendor::Nvidia, (10, 0)) => IsaLevel::Sm100a,
            (Vendor::Nvidia, (12, 0)) => IsaLevel::Sm120a,
            (Vendor::Amd, (9, 4)) => IsaLevel::Gfx942,
            (Vendor::Amd, (9, 5)) => IsaLevel::Gfx950,
            _ => return None,
        })
    }

    /// What this level can execute.
    pub fn caps(self) -> IsaCaps {
        // Written out per level rather than derived from `Arch` so that adding a
        // level is a local edit and a wrong inheritance is impossible.
        match self {
            IsaLevel::Sm89 => IsaCaps {
                mma_sync: true,
                wgmma: false,
                tcgen05: false,
                tmem: false,
                tma: false,
                dsm_cluster: false,
                mfma: false,
                block_scale_mma: false,
                mx_scale_cvt: false,
                warp_lanes: 32,
                mma_dtypes: &[
                    MmaDtype::Fp16,
                    MmaDtype::Bf16,
                    MmaDtype::Fp8,
                    MmaDtype::Int8,
                ],
            },
            IsaLevel::Sm90a => IsaCaps {
                mma_sync: true,
                wgmma: true,
                tcgen05: false,
                tmem: false,
                tma: true,
                dsm_cluster: true,
                mfma: false,
                block_scale_mma: false,
                mx_scale_cvt: false,
                warp_lanes: 32,
                mma_dtypes: &[
                    MmaDtype::Fp16,
                    MmaDtype::Bf16,
                    MmaDtype::Fp8,
                    MmaDtype::Int8,
                ],
            },
            IsaLevel::Sm100a => IsaCaps {
                mma_sync: true,
                wgmma: true,
                tcgen05: true,
                tmem: true,
                tma: true,
                dsm_cluster: true,
                mfma: false,
                block_scale_mma: true,
                mx_scale_cvt: true,
                warp_lanes: 32,
                mma_dtypes: &[
                    MmaDtype::Fp16,
                    MmaDtype::Bf16,
                    MmaDtype::Fp8,
                    MmaDtype::Fp4,
                    MmaDtype::Int8,
                ],
            },
            // The load-bearing row. Consumer Blackwell gets fp4 tensor cores but
            // NOT tcgen05/TMEM, and reaches them through `mma.sync`.
            IsaLevel::Sm120a => IsaCaps {
                mma_sync: true,
                wgmma: false,
                tcgen05: false,
                tmem: false,
                tma: false,
                dsm_cluster: false,
                mfma: false,
                block_scale_mma: false,
                mx_scale_cvt: false,
                warp_lanes: 32,
                mma_dtypes: &[
                    MmaDtype::Fp16,
                    MmaDtype::Bf16,
                    MmaDtype::Fp8,
                    MmaDtype::Fp4,
                    MmaDtype::Int8,
                ],
            },
            IsaLevel::Gfx942 => IsaCaps {
                mma_sync: false,
                wgmma: false,
                tcgen05: false,
                tmem: false,
                tma: false,
                dsm_cluster: false,
                mfma: true,
                block_scale_mma: false,
                mx_scale_cvt: false,
                warp_lanes: 64,
                mma_dtypes: &[
                    MmaDtype::Fp16,
                    MmaDtype::Bf16,
                    MmaDtype::Fp8,
                    MmaDtype::Int8,
                ],
            },
            IsaLevel::Gfx950 => IsaCaps {
                mma_sync: false,
                wgmma: false,
                tcgen05: false,
                tmem: false,
                tma: false,
                dsm_cluster: false,
                mfma: true,
                block_scale_mma: false,
                mx_scale_cvt: true,
                warp_lanes: 64,
                // CDNA4 adds fp4 matrix cores (mi350.rs:58, "fp4 is new on this
                // generation"), which CDNA3 does not have. Combined with the
                // E8M0-only scale conversion above, gfx950 is a first-class
                // MXFP4 target -- arguably the best one in this registry.
                mma_dtypes: &[
                    MmaDtype::Fp16,
                    MmaDtype::Bf16,
                    MmaDtype::Fp8,
                    MmaDtype::Fp4,
                    MmaDtype::Int8,
                ],
            },
            IsaLevel::CpuRef => IsaCaps {
                mma_sync: false,
                wgmma: false,
                tcgen05: false,
                tmem: false,
                tma: false,
                dsm_cluster: false,
                mfma: false,
                block_scale_mma: false,
                mx_scale_cvt: false,
                warp_lanes: 1,
                mma_dtypes: &[],
            },
        }
    }

    /// Device geometry for this level, or `None` where no interpreter object is
    /// built (`CpuRef`) or where the host does not yet mirror one.
    ///
    /// AMD only today, and deliberately: the NVIDIA GEMM's tile is one
    /// object-wide macro triple that `kernelcaps` probes out of `op_gemm.cuh`
    /// per build, so there is no second copy for a table to disagree with. The
    /// CDNA numbers are per-arch DEFAULTS chosen inside the header (`PLOW_CDNA4`)
    /// and re-cut by a build script, which is exactly the shape that drifts.
    pub fn geometry(self) -> Option<ArchGeometry> {
        Some(match self {
            // CDNA3, 64 KiB/workgroup. The default 256x256x64 stage is 147,456 B
            // — 2.25x over budget — so `op_gemm.h` defaults this part to a
            // SINGLE-buffered 192x256x64 (64,512 B, fits). Decode re-cuts again
            // under `PLOW_OCC4=1` to 128x256x32 (30,720 B), the smallest arena
            // the fused-GLU epilogue's `SN == 2` assert allows.
            IsaLevel::Gfx942 => ArchGeometry {
                lds_bytes: 64 * 1024,
                gemm_stage_buffers: 1,
                gemm_tile: GemmTile {
                    bm: 192,
                    bn: 256,
                    bk: 64,
                },
                decode_gemm_tile: GemmTile {
                    bm: 128,
                    bn: 256,
                    bk: 32,
                },
            },
            // CDNA4, 160 KiB/workgroup. 256x256x64 double-buffered is 144 KiB and
            // fits, so prefill and decode are built at the same tile.
            IsaLevel::Gfx950 => ArchGeometry {
                lds_bytes: 160 * 1024,
                gemm_stage_buffers: 2,
                gemm_tile: GemmTile {
                    bm: 256,
                    bn: 256,
                    bk: 64,
                },
                decode_gemm_tile: GemmTile {
                    bm: 256,
                    bn: 256,
                    bk: 64,
                },
            },
            _ => return None,
        })
    }
}

/// One GEMM stage tile, `BM x BN x BK`, exactly as `runtime/amd/op_gemm.h` spells
/// it (`GM_BM` / `GM_BN` / `GM_BK`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmTile {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
}

impl GemmTile {
    /// `GM_STRIDE`'s row padding, in halves. A fragment read walks 32 consecutive
    /// rows at one k, so an unpadded `BK*2`-byte stride lands every lane on the
    /// same bank; +8 halves shifts each row by 16 B and breaks the conflict.
    pub const PAD_HALVES: u32 = 8;

    /// `GM_LDS_HALVES_T(BM, BN, BK)` at `buffers` stage buffers.
    pub const fn stage_halves(self, buffers: u32) -> u64 {
        buffers as u64 * (self.bm + self.bn) as u64 * (self.bk + Self::PAD_HALVES) as u64
    }

    /// The same arena in bytes — the number that has to fit
    /// [`ArchGeometry::lds_bytes`].
    pub const fn stage_bytes(self, buffers: u32) -> u64 {
        self.stage_halves(buffers) * 2
    }
}

/// Device geometry the HOST has to know, because it decides what the emitter is
/// allowed to emit.
///
/// # Why this is a queryable table and not a doc comment
///
/// These are compile-time constants of the interpreter object, and the host
/// duplicates them: `devgen` picks a FUSED decode GEMV opcode only when the
/// staged rows fit the GEMM arena (`op_gemm.h`: *"x is ALWAYS staged in LDS
/// here: plowc emits this op only when M*K fits GM_LDS_HALVES"*), and the cost
/// model offers a tile only when its stage fits the LDS budget. A host copy that
/// is right for one part and wrong for another does not fail loudly — it emits
/// an opcode the object cannot run.
///
/// That is not hypothetical. `devgen`'s `GM_LDS_HALVES` mirror was the CDNA4
/// value (73,728 halves) on every part, while gfx942's shipped occ4 decode
/// profile holds **15,360** — so a Gemma-4-12B batch of up to 19 rows was fused
/// onto an arena holding four, and every row past the fourth was written past
/// the end of `plow_smem`. No batched-decode blob was ever correct on gfx942.
///
/// `crates/hwspec/tests/device_header_agreement.rs` asserts this table against
/// `runtime/amd/op_gemm.h` and `scripts/build_gfx942.sh` directly, so drift
/// fails a test rather than a serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchGeometry {
    /// LDS available to ONE workgroup. The interpreter is persistent — one
    /// workgroup per CU — so this is the whole per-CU budget.
    pub lds_bytes: u64,
    /// `GM_DBUF`: stage buffers the GEMM mainloop trades between. 2 hides the
    /// commit behind the MFMA clusters; 1 exists because on a 64 KiB part
    /// double-buffering IS the tile ceiling — no tile wider than 64x128 fits
    /// with it, and 64x128 measures 2.8x worse than a single-buffered 192x256.
    pub gemm_stage_buffers: u32,
    /// The default `GM_BM`/`GM_BN`/`GM_BK` — what `d_gemm` instantiates, and
    /// what the PREFILL object is built at.
    pub gemm_tile: GemmTile,
    /// The tile the SHIPPED DECODE object is built at, which is not the default
    /// on every part.
    ///
    /// gfx942 decode ships `PLOW_OCC4=1` (`scripts/build_gfx942.sh`), which
    /// re-cuts the arena to 128x256x32 to reach 4 waves/SIMD on both the LDS and
    /// the VGPR axis. The emitter cannot know which profile the objects will be
    /// built with — that is chosen when the OBJECTS are built, long after the
    /// blob is emitted — so the smaller of the two is the only value that is
    /// right for both.
    pub decode_gemm_tile: GemmTile,
}

impl ArchGeometry {
    /// `GM_LDS_HALVES` in a PREFILL object built for this arch.
    pub const fn gemm_arena_halves(&self) -> u64 {
        self.gemm_tile.stage_halves(self.gemm_stage_buffers)
    }

    /// `GM_LDS_HALVES` in the DECODE object — the arena a fused GEMV stages its
    /// `x` rows through, and therefore the bound the emitter's fusion gate owes.
    pub const fn decode_arena_halves(&self) -> u64 {
        self.decode_gemm_tile.stage_halves(self.gemm_stage_buffers)
    }
}

/// What an [`IsaLevel`] can execute. Kernel predicates are written against
/// these, never against a SKU name or an [`Arch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsaCaps {
    /// Warp-synchronous MMA (`mma.sync.*`).
    pub mma_sync: bool,
    /// Warpgroup MMA (`wgmma.*`) — Hopper and datacenter Blackwell.
    pub wgmma: bool,
    /// 5th-gen tensor-core MMA (`tcgen05.*`) — datacenter Blackwell only.
    pub tcgen05: bool,
    /// Tensor Memory accumulators, addressable separately from shared memory.
    pub tmem: bool,
    /// Tensor Memory Accelerator bulk copies (`cp.async.bulk` + mbarrier).
    pub tma: bool,
    /// Thread-block clusters with distributed shared memory.
    pub dsm_cluster: bool,
    /// AMD matrix cores (`v_mfma_*`).
    pub mfma: bool,
    /// Block-scaled MMA: the shared exponent of an MX block is applied *inside*
    /// the matrix instruction (`mma.sync...kind::mxf8f6f4.block_scale`, or
    /// tcgen05's block-scale forms).
    ///
    /// This is the capability that decides whether MXFP8/MXFP4 is a native path
    /// or a software dequant, and it was settled by measurement rather than by
    /// reading a marketing table: `runtime/nvidia/experiments/mxfp8_probe.cu`
    /// found that ptxas rejects the instruction on sm_120 outright --
    /// *"Instruction 'mma with block scale' not supported on .target 'sm_120'"*
    /// -- and that plain `m16n8k32` e4m3 is bit-exact there at 933 TFLOPS. So
    /// consumer Blackwell must dequant the UE8M0 scale in software and feed
    /// plain e4m3, while datacenter Blackwell does it in the instruction.
    pub block_scale_mma: bool,
    /// A conversion instruction whose scale operand is E8M0 (power-of-two) only.
    ///
    /// gfx950's `v_cvt_scalef32_pk_bf16_fp8` is exactly this shape, which makes
    /// it a natural fit for MX block scales and a *poor* one for DeepSeek/GLM's
    /// arbitrary-f32 `weight_scale_inv` -- see `runtime/amd/amd_common.h:302`,
    /// where folding 0.01 lands 22% low because only the exponent survives.
    /// The two block-scaled formats are not interchangeable on this hardware.
    pub mx_scale_cvt: bool,
    /// Lanes per warp/wave. 32 NVIDIA, 64 CDNA.
    pub warp_lanes: u32,
    /// Matrix-engine operand dtypes this level accelerates.
    pub mma_dtypes: &'static [MmaDtype],
}

impl IsaCaps {
    pub fn accelerates(&self, dt: MmaDtype) -> bool {
        self.mma_dtypes.contains(&dt)
    }
}

/// Everything that makes one measurement population distinct from another.
///
/// Silicon alone is not enough: the same card under a new driver, a new
/// compiler, or a different clock/power policy produces a different latency
/// distribution. Storing those together is how a tuning database quietly starts
/// reporting a number that was never measured on the machine being compiled for.
///
/// Construct with [`HardwareFingerprint::from_spec`] and fill in the observed
/// fields from the live device/toolchain where they are known. Fields that are
/// genuinely unknown stay `None`, and a record carrying `None` can only ever
/// satisfy a *looser* query than one carrying the real value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HardwareFingerprint {
    /// Instruction set the kernel is built for. The primary selection key.
    pub isa: IsaLevel,
    /// SKU name, for provenance and human reports. **Never** a selection key —
    /// that is what `isa` plus the resource fields are for.
    pub sku: String,
    /// Enabled SMs/CUs on this part. Two SKUs at one ISA level differ here, and
    /// it changes tile/grid choices, so it is part of the key.
    pub units: u32,
    /// Configurable shared memory / LDS per SM/CU.
    pub shared_mem_bytes: u64,
    /// 32-bit register file slots per SM/CU.
    pub regs_32bit: u32,
    /// Device memory capacity class.
    pub mem_bytes: u64,
    /// Driver version, when observed from a live device.
    pub driver: Option<String>,
    /// Compiler/toolkit version (`nvcc`/`hipcc`), when observed.
    pub toolchain: Option<String>,
    /// Clock and power policy label, when the harness pins one. An unpinned
    /// card is a different population from a locked one.
    pub clock_policy: Option<String>,
}

impl HardwareFingerprint {
    /// Build from a static spec. Observed fields are left `None` — a caller
    /// that has a live device should fill them in before publishing a
    /// measurement.
    pub fn from_spec(spec: &GpuSpec) -> Option<Self> {
        Some(HardwareFingerprint {
            isa: IsaLevel::from_spec(spec)?,
            sku: spec.name.to_string(),
            units: spec.sm_count,
            shared_mem_bytes: spec.sm.shared_mem.0,
            regs_32bit: spec.sm.regs_32bit,
            mem_bytes: spec.mem.capacity.0,
            driver: None,
            toolchain: None,
            clock_policy: None,
        })
    }

    pub fn caps(&self) -> IsaCaps {
        self.isa.caps()
    }

    /// A stable path segment for the tuning tree, e.g. `nvidia/sm_90a/h100-nvl`.
    /// Lowercased and punctuation-folded so it is filesystem-safe.
    pub fn tuning_path(&self) -> String {
        let vendor = match self.isa.vendor() {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
        };
        let sku: String = self
            .sku
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let sku = sku.trim_matches('-').to_string();
        let mut folded = String::with_capacity(sku.len());
        let mut last_dash = false;
        for c in sku.chars() {
            if c == '-' {
                if !last_dash {
                    folded.push(c);
                }
                last_dash = true;
            } else {
                folded.push(c);
                last_dash = false;
            }
        }
        format!("{vendor}/{}/{folded}", self.isa.arch_flag())
    }

    /// Whether a record measured under `self` may be used to answer a query for
    /// `want`. Same ISA level and unit count is the minimum; observed fields
    /// must match when *both* sides state them, since a `None` means "not
    /// recorded", not "matches anything measured".
    pub fn satisfies(&self, want: &HardwareFingerprint) -> bool {
        fn agree(a: &Option<String>, b: &Option<String>) -> bool {
            match (a, b) {
                (Some(x), Some(y)) => x == y,
                _ => true,
            }
        }
        self.isa == want.isa
            && self.units == want.units
            && self.shared_mem_bytes == want.shared_mem_bytes
            && agree(&self.driver, &want.driver)
            && agree(&self.toolchain, &want.toolchain)
            && agree(&self.clock_policy, &want.clock_policy)
    }
}

/// How closely a stored measurement matches the hardware being compiled for.
/// Mirrors the calibration tiers in the tuning architecture: a build that had to
/// fall back records the tier it actually used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CalibrationTier {
    /// Analytical model plus reference kernels. No measurement.
    Portable,
    /// Measured on some SKU at this ISA level.
    ArchitectureSeed,
    /// Measured on this exact SKU.
    SkuCalibrated,
    /// Measured on this deployment under its own clock/power policy.
    DeploymentCalibrated,
}

impl CalibrationTier {
    pub fn label(self) -> &'static str {
        match self {
            CalibrationTier::Portable => "portable",
            CalibrationTier::ArchitectureSeed => "architecture-seed",
            CalibrationTier::SkuCalibrated => "sku-calibrated",
            CalibrationTier::DeploymentCalibrated => "deployment-calibrated",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry;

    /// The bug this module exists to prevent: B200 and RTX 5090 are both
    /// `Arch::Blackwell`, and offering one the other's instructions is a
    /// miscompile, not a slowdown.
    #[test]
    fn blackwell_datacenter_and_consumer_are_different_isas() {
        let b200 = registry::lookup("B200").expect("B200");
        let rtx = registry::lookup("RTX 5090").expect("RTX 5090");
        assert_eq!(
            b200.arch, rtx.arch,
            "precondition: Arch really does conflate them"
        );

        let b200 = IsaLevel::from_spec(b200).unwrap();
        let rtx = IsaLevel::from_spec(rtx).unwrap();
        assert_ne!(b200, rtx);
        assert_eq!(b200, IsaLevel::Sm100a);
        assert_eq!(rtx, IsaLevel::Sm120a);

        assert!(b200.caps().tcgen05 && b200.caps().tmem);
        assert!(!rtx.caps().tcgen05, "consumer Blackwell has no tcgen05");
        assert!(!rtx.caps().tmem, "consumer Blackwell has no TMEM");
    }

    /// Hopper has wgmma; consumer Blackwell does not, despite being newer.
    /// Capability is not ordered by release date, which is why predicates read
    /// capabilities rather than comparing levels.
    #[test]
    fn capability_is_not_monotonic_in_release_order() {
        assert!(IsaLevel::Sm90a.caps().wgmma);
        assert!(!IsaLevel::Sm120a.caps().wgmma);
        assert!(IsaLevel::Sm90a.caps().tma);
        assert!(!IsaLevel::Sm120a.caps().tma);
    }

    #[test]
    fn every_registered_sku_resolves_to_an_isa_level() {
        for spec in registry::ALL {
            let isa = IsaLevel::from_spec(spec)
                .unwrap_or_else(|| panic!("{} has no IsaLevel", spec.name));
            assert_eq!(isa.vendor(), spec.vendor, "{} vendor", spec.name);
            assert_eq!(
                isa.caps().warp_lanes,
                spec.sm.warp_lanes,
                "{} warp lanes disagree between IsaCaps and SmSpec",
                spec.name
            );
        }
    }

    /// TMEM is recorded in two places; they must not disagree.
    #[test]
    fn tmem_capability_matches_the_spec_field() {
        for spec in registry::ALL {
            let isa = IsaLevel::from_spec(spec).unwrap();
            assert_eq!(
                isa.caps().tmem,
                spec.sm.tmem.0 > 0,
                "{}: IsaCaps.tmem disagrees with SmSpec.tmem",
                spec.name
            );
        }
    }

    /// `IsaCaps::mma_dtypes` is hand-written per level; `MatrixThroughput` is
    /// hand-written per SKU. They describe the same fact and must agree, or one
    /// of them is a belief rather than a datum.
    #[test]
    fn mma_dtypes_agree_with_the_spec_throughput() {
        use crate::spec::MmaDtype::*;
        for spec in registry::ALL {
            let isa = IsaLevel::from_spec(spec).unwrap();
            for dt in [Fp16, Bf16, Fp8, Fp4, Int8] {
                assert_eq!(
                    isa.caps().accelerates(dt),
                    spec.sm.mma.of(dt).is_some(),
                    "{}: IsaCaps and MatrixThroughput disagree about {dt:?}",
                    spec.name
                );
            }
        }
    }

    /// In-MMA block scaling is a tcgen05 feature. Measured, not assumed:
    /// `runtime/nvidia/experiments/mxfp8_probe.cu` records ptxas rejecting
    /// `.kind::mxf8f6f4` / `.block_scale` on sm_120 outright, so consumer
    /// Blackwell must software-dequant the UE8M0 scale.
    #[test]
    fn block_scale_mma_tracks_tcgen05_and_excludes_consumer_blackwell() {
        for lvl in [
            IsaLevel::Sm89,
            IsaLevel::Sm90a,
            IsaLevel::Sm100a,
            IsaLevel::Sm120a,
            IsaLevel::Gfx942,
            IsaLevel::Gfx950,
            IsaLevel::CpuRef,
        ] {
            assert_eq!(
                lvl.caps().block_scale_mma,
                lvl.caps().tcgen05,
                "{lvl:?}: in-MMA block scale is a tcgen05 feature"
            );
        }
        assert!(IsaLevel::Sm100a.caps().block_scale_mma);
        assert!(
            !IsaLevel::Sm120a.caps().block_scale_mma,
            "ptxas rejects it on sm_120"
        );
        assert!(!IsaLevel::Sm90a.caps().block_scale_mma);
    }

    /// gfx950's scale-converting instruction takes an E8M0 operand only, which
    /// suits MX and does NOT suit DeepSeek/GLM's arbitrary-f32 scales -- folding
    /// 0.01 there lands 22% low (`runtime/amd/amd_common.h:302`). Recording the
    /// capability keeps those two block-scaled formats from being conflated.
    #[test]
    fn e8m0_conversion_is_recorded_where_the_hardware_has_it() {
        assert!(IsaLevel::Gfx950.caps().mx_scale_cvt);
        assert!(IsaLevel::Sm100a.caps().mx_scale_cvt);
        assert!(!IsaLevel::Sm90a.caps().mx_scale_cvt);
        assert!(!IsaLevel::Gfx942.caps().mx_scale_cvt);
    }

    #[test]
    fn fingerprint_path_is_filesystem_safe() {
        let h100 = registry::lookup("H100 NVL").expect("H100 NVL");
        let fp = HardwareFingerprint::from_spec(h100).unwrap();
        assert_eq!(fp.tuning_path(), "nvidia/sm_90a/h100-nvl");
        assert!(!fp.tuning_path().contains(' '));
    }

    /// An unrecorded field must not silently match a recorded one in the
    /// direction that matters: two *stated* values that differ never match.
    #[test]
    fn observed_fields_must_agree_when_both_are_known() {
        let h100 = registry::lookup("H100 NVL").unwrap();
        let base = HardwareFingerprint::from_spec(h100).unwrap();

        let mut cuda13 = base.clone();
        cuda13.toolchain = Some("cuda-13.0".into());
        let mut cuda12 = base.clone();
        cuda12.toolchain = Some("cuda-12.4".into());

        assert!(
            base.satisfies(&cuda13),
            "unrecorded toolchain answers a looser query"
        );
        assert!(
            !cuda13.satisfies(&cuda12),
            "two known, differing toolchains must not match"
        );
        assert!(cuda13.satisfies(&cuda13));
    }

    #[test]
    fn a_different_sku_at_the_same_isa_is_not_interchangeable() {
        let nvl = HardwareFingerprint::from_spec(registry::lookup("H100 NVL").unwrap()).unwrap();
        let sxm = HardwareFingerprint::from_spec(registry::lookup("H100 SXM5").unwrap()).unwrap();
        assert_eq!(nvl.isa, sxm.isa);
        // Same ISA, different memory capacity class — they are not one population.
        assert_ne!(nvl.mem_bytes, sxm.mem_bytes);
    }
}
