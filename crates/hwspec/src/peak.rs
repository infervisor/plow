//! Peak dense matrix throughput: the compute half of every roofline floor (costmodel's
//! `BlockContract`, `Roofline`, kernelcaps' certification gate, tunedb's records). One table, so
//! a floor never depends on which crate computed it.
use crate::spec::{GpuSpec, MmaDtype};

/// Peak dense TFLOP/s for `gpu` at `dtype`: a calibrated (datasheet or measured) number for the
/// parts listed, else analytical from the SM count, tensor cores, MACs per core and boost clock.
pub fn peak_compute_tflops(gpu: &GpuSpec, dtype: MmaDtype) -> f64 {
    let name = gpu.name.to_uppercase();
    if name.contains("H100") || name.contains("H200") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 989.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1979.0,
            MmaDtype::Fp4 => 1979.0,
        }
    } else if name.contains("MI300") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 1307.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 2614.0,
            MmaDtype::Fp4 => 2614.0,
        }
    } else if name.contains("MI350") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 2300.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 4600.0,
            MmaDtype::Fp4 => 9200.0,
        }
    } else if name.contains("5090") {
        match dtype {
            // Measured, fp32 accumulate (GeForce runs it at half the fp16-accumulate rate).
            MmaDtype::Fp16 | MmaDtype::Bf16 => 235.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 1670.0,
            MmaDtype::Fp4 => 3340.0,
        }
    } else if name.contains("4090") {
        match dtype {
            MmaDtype::Fp16 | MmaDtype::Bf16 => 330.0,
            MmaDtype::Fp8 | MmaDtype::Int8 => 660.0,
            MmaDtype::Fp4 => 660.0,
        }
    } else if name == "GB10" {
        match dtype {
            // Measured (calibrate_ceilings, 2026-11-21, driver 595): 8192^3 bf16 matmul / e4m3
            // _scaled_mm, median of 30. scripts/campaign/roofline.py `gb10` carries the same
            // numbers (tests::gb10_python_registry_matches).
            MmaDtype::Fp16 | MmaDtype::Bf16 => GB10_BF16_TFLOPS,
            MmaDtype::Fp8 | MmaDtype::Int8 => GB10_FP8_TFLOPS,
            MmaDtype::Fp4 => GB10_FP4_TFLOPS,
        }
    } else {
        analytical(gpu, dtype)
    }
}

/// GB10 measured dense BF16 / FP8 TFLOP/s (see [`peak_compute_tflops`]).
pub const GB10_BF16_TFLOPS: f64 = 93.9;
pub const GB10_FP8_TFLOPS: f64 = 181.1;
/// GB10 measured dense NVFP4 TFLOP/s: cuBLASLt via torch._scaled_mm, 8192^3, e2m1 with e4m3 scales
/// per 16 (calibrate_ceilings, 2026-11-24; e4m3 184.7 in the same lease).
pub const GB10_FP4_TFLOPS: f64 = 378.6;

/// Ceilings of ONE SM running alone (gridDim = 1): the roofline of a kernel's inner loop on its
/// per-SM work slice. The whole-GPU numbers above stay the floors of a full launch; a single SM
/// is bounded by its own issue rate and its own load path, not by its 1/n_sm share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmCeilings {
    /// SM clock the rates were measured at (GHz).
    pub clock_ghz: f64,
    /// `mma.sync` m16n8k16 bf16, f32 accumulate.
    pub bf16_tflops: f64,
    /// `mma.sync` m16n8k32 e4m3 (plain and `kind::f8f6f4`; MXFP8 block-scaled is the same rate).
    pub fp8_tflops: f64,
    /// `mma.sync` m16n8k64 `kind::mxf4nvf4` ue4m3 4X (NVFP4; MXFP4 is the same rate).
    pub nvfp4_tflops: f64,
    pub fp32_tflops: f64,
    /// Streaming DRAM read (best of ld.global.v4/v8, cp.async, cp.async.bulk; within 6%).
    pub dram_gbps: f64,
    /// L2-resident streaming read.
    pub l2_gbps: f64,
    /// `ld.shared.v4`, conflict-free.
    pub smem_gbps: f64,
}

/// GB10 single-SM ceilings: runtime/ubench `ubench_cu_roofline_nv --ceilings` (2026-11-24, jg02,
/// driver 595). Whole-GPU cross-check (gridDim 48, one block per SM) = 48x these for the matrix
/// rates (126.7 / 253.4 / 506.4 TFLOP/s of `mma.sync` issue, above the cuBLAS peaks above), 236-239
/// GB/s DRAM (4.9 GB/s per SM). One SM reads DRAM at ~30 B/clk, a third of the chip's bandwidth,
/// and L2 hits are no faster (32 B/clk): the per-SM load path, not DRAM, is its memory ceiling.
pub const GB10_SM_CEILINGS: SmCeilings = SmCeilings {
    clock_ghz: 2.579,
    bf16_tflops: 2.640,
    fp8_tflops: 5.283,
    nvfp4_tflops: 10.556,
    fp32_tflops: 0.645,
    dram_gbps: 78.1,
    l2_gbps: 82.6,
    smem_gbps: 329.1,
};

/// The measured single-SM ceilings of `gpu`, where there are any.
pub fn sm_ceilings(gpu: &GpuSpec) -> Option<SmCeilings> {
    (gpu.name == "GB10").then_some(GB10_SM_CEILINGS)
}

/// Whether `gpu`'s FP4 peak is measured. W4A4 floors use it only then (costmodel `peak_dtype`); an
/// analytical FP4 peak never prices a floor.
pub fn fp4_peak_measured(gpu: &GpuSpec) -> bool {
    gpu.name == "GB10"
}

fn analytical(gpu: &GpuSpec, dtype: MmaDtype) -> f64 {
    let cores = gpu.sm_count as f64 * gpu.sm.tensor_cores as f64;
    let macs_core = gpu.sm.mma.of(dtype).unwrap_or(256) as f64;
    let clock_hz = gpu.clock_boost.0 as f64;
    // 2 FLOPs per MAC
    (cores * macs_core * 2.0 * clock_hz) / 1e12
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One SM's issue rates are bf16 : fp8 : fp4 = 1 : 2 : 4, n_sm of them bound the whole-GPU
    /// GEMM peaks from above, and one SM alone cannot pull the chip's DRAM bandwidth.
    #[test]
    fn gb10_sm_ceilings_are_consistent_with_the_whole_gpu() {
        let gb10 = crate::registry::lookup("GB10").unwrap();
        let sm = sm_ceilings(gb10).expect("GB10 single-SM ceilings");
        let n = gb10.sm_count as f64;
        assert!((sm.fp8_tflops / sm.bf16_tflops - 2.0).abs() < 0.05);
        assert!((sm.nvfp4_tflops / sm.fp8_tflops - 2.0).abs() < 0.05);
        assert!(sm.bf16_tflops * n >= GB10_BF16_TFLOPS);
        assert!(sm.fp8_tflops * n >= GB10_FP8_TFLOPS);
        assert!(sm.nvfp4_tflops * n >= GB10_FP4_TFLOPS);
        let bw = gb10.mem.bandwidth_measured.unwrap().0;
        assert!(sm.dram_gbps < bw && sm.dram_gbps > bw / n);
        // bf16 m16n8k16: 1024 FLOP per SM clock (4 tensor cores x 128 FMA)
        assert!((sm.bf16_tflops * 1e3 / sm.clock_ghz - 1024.0).abs() < 16.0);
        assert!(sm_ceilings(crate::registry::lookup("RTX 5090").unwrap()).is_none());
    }

    /// The GB10 entry of the Python roofline registry (op_roof.py, block_roofline.py,
    /// gen_bench ceilings) states the bandwidth and peaks this crate holds.
    #[test]
    fn gb10_python_registry_matches() {
        let src = include_str!("../../../scripts/campaign/roofline.py");
        let start = src
            .find("\"gb10\": HardwareSpec(")
            .expect("roofline.py gb10 entry");
        let entry = &src[start..start + src[start..].find("),").expect("entry end")];
        let field = |name: &str| -> String {
            let at = entry
                .find(&format!("{name}="))
                .unwrap_or_else(|| panic!("gb10 {name}"));
            let v = &entry[at + name.len() + 1..];
            v[..v.find(',').unwrap()].trim().to_string()
        };
        let gb10 = crate::registry::lookup("GB10").unwrap();
        let num = |name: &str| {
            field(name)
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("gb10 {name}"))
        };
        assert_eq!(num("bandwidth_datasheet_gbps"), gb10.mem.bandwidth.0);
        assert_eq!(
            num("bandwidth_measured_gbps"),
            gb10.mem
                .bandwidth_measured
                .expect("GB10 measured bandwidth")
                .0
        );
        assert_eq!(
            num("bf16_tflops_dense"),
            peak_compute_tflops(gb10, MmaDtype::Bf16)
        );
        assert_eq!(
            num("fp16_tflops_dense"),
            peak_compute_tflops(gb10, MmaDtype::Fp16)
        );
        assert_eq!(
            num("fp8_tflops_dense"),
            peak_compute_tflops(gb10, MmaDtype::Fp8)
        );
        assert_eq!(
            num("fp4_tflops_dense"),
            peak_compute_tflops(gb10, MmaDtype::Fp4)
        );
        assert!(fp4_peak_measured(gb10));
        assert_eq!(num("sm_or_cu_count"), gb10.sm_count as f64);
        assert_eq!(num("clock_boost_mhz"), gb10.clock_boost.0 as f64 / 1e6);
    }
}
