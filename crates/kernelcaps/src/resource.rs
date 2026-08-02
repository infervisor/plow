//! Interpreter resource envelopes, and the gate that rejects a poisoned one.
//!
//! The persistent interpreter inlines every dispatch arm, so its register
//! allocation is the **worst case over all inlined code**. One expensive arm
//! therefore sets the register count, the shared-memory union, and the
//! occupancy for every other op in the same object. An isolated kernel win that
//! pushes the object over a cliff makes the model slower.
//!
//! AMD already enforces this at build time: `scripts/build_gfx950.sh:118` parses
//! `-Rpass-analysis=kernel-resource-usage` and **fails the build** when
//! `VGPR+AGPR > 256` or occupancy drops below 2 waves/SIMD. The comment there is
//! blunt about why — past the cliff the dispatch is rejected outright with
//! `HSA_STATUS_ERROR_INVALID_ISA`.
//!
//! NVIDIA has no equivalent. `runtime/nvidia/interp_sm120.cu:21` states the gate
//! in prose — "must show 0 bytes spill and >= 1 block/SM" — but no script or
//! CMake target runs it, and the numbers survive only as code comments. They
//! have drifted: on CUDA 13.0 the sm90a decode object reports 208 registers
//! against a documented 150, and the sm90a prefill object reports 255 registers
//! with 180 B of spill stores and 644 B of spill loads against a documented 236
//! and an explicit zero-spill requirement.
//!
//! This module parses both vendors' reports into one type and applies the gate
//! the comments describe, so that drift is a test failure rather than an
//! archaeology exercise.

use hwspec::{HardwareFingerprint, IsaLevel};

use crate::spec::ProfileId;

/// What one built interpreter object costs.
///
/// Per *object*, not per kernel: the numbers are a property of everything
/// inlined together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceEnvelope {
    pub profile: ProfileId,
    pub isa: IsaLevel,
    /// Mangled kernel symbol the numbers were read from.
    pub symbol: String,
    /// NVIDIA: registers/thread. AMD: VGPR + AGPR, which share a budget.
    pub registers: u32,
    /// Bytes spilled to local memory. The gate wants this at zero.
    pub spill_stores: u32,
    pub spill_loads: u32,
    /// Statically allocated shared memory / LDS.
    pub static_smem_bytes: u32,
    /// Dynamic shared memory the launch requests, when known. On NVIDIA this is
    /// the arena, embedded in the cubin as `plow_arena_bytes`.
    pub dynamic_smem_bytes: Option<u32>,
    /// Resident blocks per SM (NVIDIA) or waves per SIMD (AMD).
    pub occupancy: Option<u32>,
    /// Toolchain that produced these numbers. The same source under a different
    /// compiler is a different envelope, which is exactly how the documented
    /// figures went stale.
    pub toolchain: Option<String>,
}

impl ResourceEnvelope {
    pub fn spills(&self) -> bool {
        self.spill_stores > 0 || self.spill_loads > 0
    }
}

/// The limits an envelope must satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceGate {
    /// Inclusive ceiling on registers.
    pub max_registers: u32,
    /// Whether any spill at all fails the gate.
    pub require_zero_spill: bool,
    /// Minimum occupancy, in the vendor's unit.
    pub min_occupancy: u32,
}

impl ResourceGate {
    /// The gate `runtime/nvidia/interp_sm120.cu:21` describes: zero spill and at
    /// least one resident block per SM. 255 is ptxas's own ceiling, so a kernel
    /// reporting it has saturated the allocator and is spilling by definition.
    pub const NVIDIA: ResourceGate = ResourceGate {
        max_registers: 255,
        require_zero_spill: true,
        min_occupancy: 1,
    };

    /// The gate `scripts/build_gfx950.sh:118` enforces for the prefill and
    /// decode objects: VGPR+AGPR within 256 and at least 2 waves/SIMD.
    pub const AMD_INTERP: ResourceGate = ResourceGate {
        max_registers: 256,
        require_zero_spill: false,
        min_occupancy: 2,
    };

    /// The relaxed AMD budget the flash object is allowed
    /// (`build_gfx950.sh` checks it at 512 registers / 1 wave).
    pub const AMD_FLASH: ResourceGate = ResourceGate {
        max_registers: 512,
        require_zero_spill: false,
        min_occupancy: 1,
    };

    pub fn for_isa(isa: IsaLevel) -> Self {
        match isa {
            IsaLevel::Gfx942 | IsaLevel::Gfx950 => ResourceGate::AMD_INTERP,
            _ => ResourceGate::NVIDIA,
        }
    }

    pub fn check(&self, env: &ResourceEnvelope) -> GateVerdict {
        let mut failures = Vec::new();
        if env.registers > self.max_registers {
            failures.push(format!(
                "{} registers exceeds the {} ceiling",
                env.registers, self.max_registers
            ));
        }
        if self.require_zero_spill && env.spills() {
            failures.push(format!(
                "spills {} B stores / {} B loads; the gate requires zero",
                env.spill_stores, env.spill_loads
            ));
        }
        if let Some(occ) = env.occupancy {
            if occ < self.min_occupancy {
                failures.push(format!(
                    "occupancy {} is below the required {}",
                    occ, self.min_occupancy
                ));
            }
        }
        if failures.is_empty() {
            GateVerdict::Pass
        } else {
            GateVerdict::Fail(failures)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateVerdict {
    Pass,
    Fail(Vec<String>),
}

impl GateVerdict {
    pub fn is_pass(&self) -> bool {
        matches!(self, GateVerdict::Pass)
    }
}

/// Parse `nvcc -Xptxas -v` output for one entry function.
///
/// ptxas reports every `__global__` in the translation unit, so the caller names
/// the symbol it cares about — for an interpreter object that is the megakernel,
/// and the surrounding helper kernels are noise.
pub fn parse_ptxas(
    text: &str,
    symbol: &str,
    profile: ProfileId,
    isa: IsaLevel,
) -> Option<ResourceEnvelope> {
    let mut lines = text.lines();
    // Find the entry-function banner, then read the block that follows it.
    loop {
        let line = lines.next()?;
        if line.contains("Compiling entry function") && line.contains(symbol) {
            break;
        }
    }

    let mut env = ResourceEnvelope {
        profile,
        isa,
        symbol: symbol.to_string(),
        registers: 0,
        spill_stores: 0,
        spill_loads: 0,
        static_smem_bytes: 0,
        dynamic_smem_bytes: None,
        occupancy: None,
        toolchain: None,
    };
    let mut saw_usage = false;

    for line in lines {
        // The next entry function ends this block.
        if line.contains("Compiling entry function") {
            break;
        }
        if let Some(v) = field_before(line, "bytes spill stores") {
            env.spill_stores = v;
        }
        if let Some(v) = field_before(line, "bytes spill loads") {
            env.spill_loads = v;
        }
        if let Some(v) = field_after(line, "Used ", " registers") {
            env.registers = v;
            saw_usage = true;
        }
        if let Some(v) = field_before(line, "bytes smem") {
            env.static_smem_bytes = v;
        }
    }

    saw_usage.then_some(env)
}

/// Parse `hipcc -Rpass-analysis=kernel-resource-usage` output.
///
/// AMD's VGPR and AGPR files share one budget, so the gate is applied to their
/// sum — the same arithmetic `build_gfx950.sh` does.
pub fn parse_rocm_resource_usage(
    text: &str,
    profile: ProfileId,
    isa: IsaLevel,
) -> Option<ResourceEnvelope> {
    let mut vgpr = None;
    let mut agpr = None;
    let mut occ = None;
    let mut spill = 0u32;
    let mut lds = 0u32;

    for line in text.lines() {
        if let Some(v) = field_after(line, "VGPRs: ", "") {
            vgpr.get_or_insert(v);
        }
        if let Some(v) = field_after(line, "AGPRs: ", "") {
            agpr.get_or_insert(v);
        }
        if let Some(v) = field_after(line, "Occupancy [waves/SIMD]: ", "") {
            occ.get_or_insert(v);
        }
        if let Some(v) = field_after(line, "VGPRs Spill: ", "") {
            spill = spill.max(v);
        }
        if let Some(v) = field_after(line, "LDS Size [bytes/block]: ", "") {
            lds = lds.max(v);
        }
    }

    let registers = vgpr? + agpr.unwrap_or(0);
    Some(ResourceEnvelope {
        profile,
        isa,
        symbol: String::new(),
        registers,
        spill_stores: spill,
        spill_loads: 0,
        static_smem_bytes: lds,
        dynamic_smem_bytes: None,
        occupancy: occ,
        toolchain: None,
    })
}

/// `"... 180 bytes spill stores ..."` -> `180`.
fn field_before(line: &str, tail: &str) -> Option<u32> {
    let idx = line.find(tail)?;
    line[..idx].split_whitespace().next_back()?.parse().ok()
}

/// `"Used 208 registers"` -> `208`. An empty `tail` reads to the first
/// non-numeric character, which is how the AMD report is shaped.
fn field_after(line: &str, head: &str, tail: &str) -> Option<u32> {
    let idx = line.find(head)? + head.len();
    let rest = &line[idx..];
    let rest = if tail.is_empty() {
        rest
    } else {
        let end = rest.find(tail)?;
        &rest[..end]
    };
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Attach the observed toolchain to a fingerprint, so a measurement taken under
/// one compiler cannot answer a query about another.
pub fn stamp_toolchain(fp: &mut HardwareFingerprint, toolchain: impl Into<String>) {
    fp.toolchain = Some(toolchain.into());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real `nvcc -Xptxas -v` output for the sm90a **decode** object built from
    /// this tree on CUDA 13.0. It passes the gate, but note the register count
    /// against the 150 documented in `scripts/build_sm90a_cubin.sh`.
    const DECODE_SM90A: &str = "\
ptxas info    : 8 bytes gmem
ptxas info    : Compiling entry function '_Z12interp_sm90a11PlowProgram' for 'sm_90a'
ptxas info    : Function properties for _Z12interp_sm90a11PlowProgram
    1024 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 208 registers, used 1 barriers, 1024 bytes cumulative stack size, 2192 bytes smem
ptxas info    : Compile time = 43664.770 ms
ptxas info    : Compiling entry function '_Z25plow_moe_slot_glu_fp8_blkP13__nv_bfloat16' for 'sm_90a'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers
";

    /// Real output for the sm90a **prefill** object, same tree and toolchain.
    /// This one violates the zero-spill gate the interpreter's own header
    /// states, and nothing in the build catches it today.
    const PREFILL_SM90A: &str = "\
ptxas info    : Compiling entry function '_Z15interp_sm90a_pf11PlowProgram' for 'sm_90a'
ptxas info    : Function properties for _Z15interp_sm90a_pf11PlowProgram
    1216 bytes stack frame, 180 bytes spill stores, 644 bytes spill loads
ptxas info    : Used 255 registers, used 1 barriers, 1216 bytes cumulative stack size, 2320 bytes smem
ptxas info    : Compile time = 51221.010 ms
";

    #[test]
    fn parses_the_megakernel_and_ignores_helper_kernels() {
        let env = parse_ptxas(
            DECODE_SM90A,
            "_Z12interp_sm90a11PlowProgram",
            ProfileId::DecodeDense,
            IsaLevel::Sm90a,
        )
        .expect("decode envelope");

        assert_eq!(
            env.registers, 208,
            "must not pick up the 40-register helper"
        );
        assert_eq!(env.spill_stores, 0);
        assert_eq!(env.spill_loads, 0);
        assert_eq!(env.static_smem_bytes, 2192);
        assert!(!env.spills());
    }

    #[test]
    fn decode_object_passes_the_nvidia_gate() {
        let env = parse_ptxas(
            DECODE_SM90A,
            "_Z12interp_sm90a11PlowProgram",
            ProfileId::DecodeDense,
            IsaLevel::Sm90a,
        )
        .unwrap();
        assert!(ResourceGate::NVIDIA.check(&env).is_pass());
    }

    /// The finding this module exists to make enforceable: the Hopper prefill
    /// interpreter spills on CUDA 13.0, against an explicit zero-spill
    /// requirement, and no build step notices.
    #[test]
    fn prefill_object_fails_the_documented_zero_spill_gate() {
        let env = parse_ptxas(
            PREFILL_SM90A,
            "_Z15interp_sm90a_pf11PlowProgram",
            ProfileId::PrefillDense,
            IsaLevel::Sm90a,
        )
        .expect("prefill envelope");

        assert_eq!(env.registers, 255, "saturated the ptxas ceiling");
        assert_eq!(env.spill_stores, 180);
        assert_eq!(env.spill_loads, 644);
        assert!(env.spills());

        match ResourceGate::NVIDIA.check(&env) {
            GateVerdict::Pass => panic!("a spilling interpreter must not pass the gate"),
            GateVerdict::Fail(reasons) => {
                assert!(
                    reasons.iter().any(|r| r.contains("spill")),
                    "the spill must be named in the failure: {reasons:?}"
                );
            }
        }
    }

    /// AMD's own format, as `build_gfx950.sh` greps it. VGPR and AGPR sum.
    #[test]
    fn parses_amd_resource_usage_and_sums_the_register_files() {
        let text = "\
remark: interp.hip:872:0: Function Name: plow_interp_gfx950
remark: interp.hip:872:0:     VGPRs: 160
remark: interp.hip:872:0:     AGPRs: 128
remark: interp.hip:872:0:     Occupancy [waves/SIMD]: 1
remark: interp.hip:872:0:     VGPRs Spill: 0
remark: interp.hip:872:0:     LDS Size [bytes/block]: 65536
";
        let env =
            parse_rocm_resource_usage(text, ProfileId::PrefillDense, IsaLevel::Gfx950).unwrap();
        assert_eq!(env.registers, 288, "160 VGPR + 128 AGPR share one budget");
        assert_eq!(env.occupancy, Some(1));
        assert_eq!(env.static_smem_bytes, 65536);

        // 288 > 256 and occupancy 1 < 2: this is the register cliff that makes
        // the HSA dispatch fail outright, so both must be reported.
        match ResourceGate::AMD_INTERP.check(&env) {
            GateVerdict::Pass => panic!("288 registers at occupancy 1 must fail"),
            GateVerdict::Fail(reasons) => assert_eq!(reasons.len(), 2, "{reasons:?}"),
        }
    }

    /// The flash object is allowed a larger budget; the same numbers that fail
    /// the interpreter gate pass this one.
    #[test]
    fn the_relaxed_flash_budget_admits_a_heavier_object() {
        let env = ResourceEnvelope {
            profile: ProfileId::PrefillDense,
            isa: IsaLevel::Gfx950,
            symbol: String::new(),
            registers: 288,
            spill_stores: 0,
            spill_loads: 0,
            static_smem_bytes: 0,
            dynamic_smem_bytes: None,
            occupancy: Some(1),
            toolchain: None,
        };
        assert!(!ResourceGate::AMD_INTERP.check(&env).is_pass());
        assert!(ResourceGate::AMD_FLASH.check(&env).is_pass());
    }

    #[test]
    fn a_missing_symbol_yields_no_envelope() {
        assert!(parse_ptxas(
            DECODE_SM90A,
            "_Z9not_there",
            ProfileId::DecodeDense,
            IsaLevel::Sm90a
        )
        .is_none());
    }

    #[test]
    fn amd_gate_applies_to_amd_isas_and_nvidia_gate_otherwise() {
        assert_eq!(
            ResourceGate::for_isa(IsaLevel::Gfx950),
            ResourceGate::AMD_INTERP
        );
        assert_eq!(ResourceGate::for_isa(IsaLevel::Sm90a), ResourceGate::NVIDIA);
        assert_eq!(
            ResourceGate::for_isa(IsaLevel::Sm120a),
            ResourceGate::NVIDIA
        );
    }
}
