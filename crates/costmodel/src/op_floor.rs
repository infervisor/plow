//! Per-instruction floors for the sm_120 Nemotron-H / NVFP4 / unit-scale FP8 ops (RowColScale
//! 231, the NVFP4 MoE 232-239 with its combine 87, Mamba-2 240-245): compulsory bytes and FLOPs
//! of one instruction, and the [`BlockContract`] they make on a GPU.
//!
//! The same formulas price these ops in `scripts/bench/op_roof.py` (per-op roofline tables) and
//! back `scripts/campaign/certify_kernel.py`'s gate; `data/op_floor_cases.json` holds rows both
//! must reproduce (this module's test and `test_op_roof.py`), so a kernel certified here and a
//! floor printed there are the same number. Span-verify forms of an op (operands in its high
//! halves) price as the op itself: snapshot writes are not in the floor.
//!
//! Operands are read by slot name as the disassembler prints them (`packet::slots`).
use hwspec::{GpuSpec, MmaDtype};

use crate::contract::{BlockContract, BoundCategory};
use crate::roofline::peak_compute_tflops;

/// Why an instruction has no floor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FloorError {
    /// Not one of this module's ops.
    NotPriced,
    /// The floor needs an operand the instruction does not carry (`experts`: the touched-expert
    /// count of a routed expert op, from the router table or a routing model).
    Missing(&'static str),
}

/// `(bytes, flops)` of one `op` instruction in a program of `rows` rows. `p` reads an operand by
/// slot name.
pub fn op_floor(
    op: &str,
    rows: f64,
    p: &dyn Fn(&str) -> Option<f64>,
) -> Result<(f64, f64), FloorError> {
    let g = |k: &'static str, default: f64| p(k).unwrap_or(default);
    let need = |k: &'static str| p(k).ok_or(FloorError::Missing(k));
    let fdiv = |a: f64, b: f64| (a / b).floor();
    Ok(match op {
        // Unit-scale FP8 GEMM outer scaling: bf16 x read, bf16 out written, the two vectors.
        "RowColScale" => {
            let (m, n) = (g("M", rows), need("N")?);
            (4.0 * m * n + 4.0 * (m + n), 0.0)
        }
        // Mamba-2 (op_mamba.cuh): bf16 activations, SSM state [H][P][N] per slot of
        // `state_bytes` per element (0 = f32, packets before the immediate).
        "Mamba2Conv" => {
            let (n, c, w) = (g("rows", rows), need("C")?, need("W")?);
            let history = (w - 1.0)
                * c
                * 2.0
                * if g("prefill", 0.0) != 0.0 {
                    1.0
                } else {
                    2.0 * n
                };
            (4.0 * n * c + history + 2.0 * c * (w + 1.0), 2.0 * n * c * w)
        }
        "Mamba2Step" => {
            let (n, h, pp, nn, gg) = (g("B", rows), need("H")?, need("P")?, need("N")?, need("G")?);
            let sb = state_bytes(p);
            let bytes =
                n * (2.0 * sb * h * pp * nn + 2.0 * (h * pp + 2.0 * gg * nn) + 2.0 * h * pp);
            (bytes, 6.0 * n * h * pp * nn)
        }
        "Mamba2GatedNorm" => (
            6.0 * g("rows", rows) * need("groups")? * need("group")?,
            0.0,
        ),
        "Mamba2ChunkState" | "Mamba2ChunkPass" | "Mamba2ChunkOut" => {
            let (r, h, pp, nn) = (g("R", rows), need("H")?, need("P")?, need("N")?);
            let chunks = (r / 128.0).ceil();
            let states = 4.0 * chunks * h * pp * nn;
            if op == "Mamba2ChunkPass" {
                let bytes = 2.0 * states + 2.0 * state_bytes(p) * h * pp * nn + 4.0 * h * r;
                return Ok((bytes, 2.0 * chunks * h * pp * nn));
            }
            let gg = need("G")?;
            let xbc = 2.0 * r * (h * pp + 2.0 * gg * nn);
            if op == "Mamba2ChunkState" {
                (xbc + states + 8.0 * h * r, 2.0 * r * h * pp * nn)
            } else {
                let flops = 2.0 * r * (128.0 * nn * gg + 128.0 * pp * h + nn * pp * h);
                (xbc + states + 8.0 * h * r + 2.0 * r * h * pp, flops)
            }
        }
        // NVFP4 MoE (op_moe_nvfp4.cuh): e2m1 weights + e4m3 scale per 16 + f32 gs per expert.
        "MoeRouterLogitsF32" => {
            let (t, e, h) = (g("T", rows), need("E")?, need("H")?);
            let ks = g("KS", 1.0).max(1.0);
            (
                2.0 * e * h + 2.0 * t * h + 4.0 * ks * t * e,
                2.0 * t * e * h,
            )
        }
        "MoeRouterSigmoidTopk" => {
            let (t, e, k) = (g("T", rows), need("E")?, need("k")?);
            (
                4.0 * g("KS", 1.0).max(1.0) * t * e + 4.0 * e + 8.0 * t * k,
                0.0,
            )
        }
        "MoeAlignNvfp4" => (20.0 * g("T", rows) * need("k")? + 12.0 * need("E")?, 0.0),
        "MoeCombinePf" => {
            let (t, h, k) = (g("T", rows), need("H")?, need("k")?);
            if g("bf16", 0.0) != 0.0 {
                // bf16 partials + the routing table's gates
                (t * h * (2.0 * k + 2.0) + 8.0 * t * k, 0.0)
            } else {
                (t * h * (4.0 * k + 2.0), 0.0)
            }
        }
        // bf16 rows in, e2m1 + e4m3 per 16 out.
        "MoeQuantNvfp4" => {
            let (r, k) = (need("R")?, need("K")?);
            (
                2.0 * r * k + fdiv(r * k * 9.0, 16.0) + 4.0 * need("n")?,
                0.0,
            )
        }
        "MoeUpNvfp4" | "MoeDownNvfp4" | "MoeUpNvfp4Pf" | "MoeDownNvfp4Pf" => {
            let (t, e) = (g("T", rows), need("E")?);
            let up = op.starts_with("MoeUp");
            let k = if e > 1.0 { need("k")? } else { 1.0 };
            let (n, kin, mut out) = if up {
                (need("I")?, need("K")?, 2.0)
            } else {
                let out = if e > 1.0 && g("mode", 0.0) != 2.0 {
                    4.0
                } else {
                    2.0
                };
                (need("H")?, need("I")?, out)
            };
            let experts = if e > 1.0 { need("experts")? } else { 1.0 };
            let weight = experts * (fdiv(n * kin, 2.0) + fdiv(n * kin, 16.0) + 4.0);
            let mut act_in = 2.0 * t * if up { kin } else { k * kin };
            let rows_out = if e > 1.0 || up { t * k } else { t };
            if g("w4a4", 0.0) != 0.0 {
                // NVFP4 q tensors (9/16 B per element) in, and out of the up projection.
                act_in = fdiv(act_in * 9.0, 32.0);
                if up {
                    out = 9.0 / 16.0;
                }
            }
            (
                weight + act_in + (out * rows_out * n).trunc(),
                2.0 * t * k * n * kin,
            )
        }
        _ => return Err(FloorError::NotPriced),
    })
}

fn state_bytes(p: &dyn Fn(&str) -> Option<f64>) -> f64 {
    match p("state_bytes") {
        Some(b) if b != 0.0 => b,
        _ => 4.0,
    }
}

/// The matrix peak an op's FLOPs are priced at: W4A4 experts at FP8 (FP4 is not calibrated; its
/// peak is at least FP8's, so the floor stays a bound), everything else at BF16.
pub fn peak_dtype(op: &str, p: &dyn Fn(&str) -> Option<f64>) -> MmaDtype {
    let w4a4 = p("w4a4").is_some_and(|v| v != 0.0);
    let expert = matches!(
        op,
        "MoeUpNvfp4" | "MoeDownNvfp4" | "MoeUpNvfp4Pf" | "MoeDownNvfp4Pf"
    );
    if expert && w4a4 {
        MmaDtype::Fp8
    } else {
        MmaDtype::Bf16
    }
}

impl BlockContract {
    /// The contract of one `op` instruction on `gpu`: its floor, bounded by whichever of
    /// `bytes / bandwidth` and `flops / peak` is larger there.
    pub fn for_op(
        op: &str,
        rows: f64,
        p: &dyn Fn(&str) -> Option<f64>,
        gpu: &GpuSpec,
    ) -> Result<Self, FloorError> {
        let (bytes, flops) = op_floor(op, rows, p)?;
        let dtype = peak_dtype(op, p);
        let t_mem = bytes / (gpu.mem.bandwidth_for_bound().0 * 1e9);
        let t_mat = flops / (peak_compute_tflops(gpu, dtype) * 1e12);
        let bound = if t_mat > t_mem {
            BoundCategory::ComputeBoundTensorCore
        } else {
            BoundCategory::MemoryBoundHbm
        };
        Ok(Self::new(op, bound, dtype, flops as u64, bytes as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{CERTIFY_COMPUTE_BOUND, CERTIFY_MEMORY_BOUND};

    fn cases() -> serde_json::Value {
        serde_json::from_str(include_str!("../data/op_floor_cases.json")).unwrap()
    }

    #[test]
    fn reproduces_the_shared_floor_table() {
        let doc = cases();
        let mut wrong = Vec::new();
        for c in doc["cases"].as_array().unwrap() {
            let op = c["op"].as_str().unwrap();
            let params = c["params"].as_object().unwrap();
            let p = |k: &str| params.get(k).and_then(|v| v.as_f64());
            let got = op_floor(op, c["rows"].as_f64().unwrap(), &p).unwrap();
            let want = (c["bytes"].as_f64().unwrap(), c["flops"].as_f64().unwrap());
            if got != want {
                wrong.push(format!("{op} {params:?}: {got:?} vs {want:?}"));
            }
        }
        assert!(
            wrong.is_empty(),
            "op floors drifted from op_floor_cases.json:\n{}",
            wrong.join("\n")
        );
        let gate = &doc["certification"];
        assert_eq!(gate["memory_bound"].as_f64(), Some(CERTIFY_MEMORY_BOUND));
        assert_eq!(gate["compute_bound"].as_f64(), Some(CERTIFY_COMPUTE_BOUND));
    }

    #[test]
    fn routed_experts_need_the_touched_count() {
        let p = |k: &str| match k {
            "T" => Some(4.0),
            "k" => Some(22.0),
            "I" => Some(2688.0),
            "K" => Some(1024.0),
            "E" => Some(512.0),
            _ => None,
        };
        assert_eq!(
            op_floor("MoeUpNvfp4", 4.0, &p),
            Err(FloorError::Missing("experts"))
        );
        assert_eq!(op_floor("Gemv", 4.0, &p), Err(FloorError::NotPriced));
    }

    #[test]
    fn contract_bound_follows_the_hardware() {
        let gb10 = hwspec::registry::lookup("GB10").unwrap();
        let step = |k: &str| match k {
            "H" => Some(128.0),
            "P" => Some(64.0),
            "N" => Some(128.0),
            "G" => Some(8.0),
            "B" => Some(4.0),
            "state_bytes" => Some(2.0),
            _ => None,
        };
        let c = BlockContract::for_op("Mamba2Step", 4.0, &step, gb10).unwrap();
        assert_eq!(c.bound_category, BoundCategory::MemoryBoundHbm);
        let up = |k: &str| match k {
            "T" => Some(8192.0),
            "k" => Some(22.0),
            "I" => Some(2688.0),
            "K" => Some(1024.0),
            "E" => Some(512.0),
            "w4a4" => Some(1.0),
            "experts" => Some(512.0),
            _ => None,
        };
        let c = BlockContract::for_op("MoeUpNvfp4Pf", 8192.0, &up, gb10).unwrap();
        assert_eq!(
            (c.bound_category, c.dtype),
            (BoundCategory::ComputeBoundTensorCore, MmaDtype::Fp8)
        );
    }
}
