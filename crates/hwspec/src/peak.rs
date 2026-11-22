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
            // Not measured on this part.
            MmaDtype::Fp4 => analytical(gpu, dtype),
        }
    } else {
        analytical(gpu, dtype)
    }
}

/// GB10 measured dense BF16 / FP8 TFLOP/s (see [`peak_compute_tflops`]).
pub const GB10_BF16_TFLOPS: f64 = 93.9;
pub const GB10_FP8_TFLOPS: f64 = 181.1;

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
        assert_eq!(num("sm_or_cu_count"), gb10.sm_count as f64);
        assert_eq!(num("clock_boost_mhz"), gb10.clock_boost.0 as f64 / 1e6);
    }
}
