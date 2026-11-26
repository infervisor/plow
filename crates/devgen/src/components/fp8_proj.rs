//! A checkpoint FP8 per-tensor linear (`[n, k]` e4m3 + rank-0 f32 `weight_scale`, ModelOpt FP8).
//! The rank-0 scale is tiled to the per-channel vector the FP8 GEMV / native GEMM read
//! (`derived_weights` `repeat`).
//!
//! - decode: `GemvFp8` on bf16 activations (W8A16);
//! - prefill, where `segment_roles::cublaslt_prefill_fp8_tensor` holds the shape: the peer's
//!   static per-tensor activation quant with the checkpoint `input_scale` and one cuBLASLt matmul
//!   with scalar scales (`FP8_TENSOR_SCALE`);
//! - prefill elsewhere: per-row dynamic `QuantFp8` + the native `GemmFp8`.
//!
//! Projections of one input share its quant ([`Quant`]); the next norm (or any other input) ends
//! its lifetime.
use packet::dev::FP8_TENSOR_SCALE;
use packet::typed_abi::{GemmFp8Args, GemvFp8Args, QuantFp8Args};

use super::Program;

/// The live activation quant: `(input, width, scale tensor)`.
pub(crate) type Quant = (u32, u32, u32);

/// The static per-tensor route (cuBLASLt, scalar scales) holds this prefill shape on `isa`.
pub(crate) fn static_route(isa: Option<hwspec::IsaLevel>, rows: u32, n: u32, k: u32) -> bool {
    let profile = isa.map_or("", hwspec::IsaLevel::arch_flag);
    plow_asset::segment_roles::cublaslt_prefill_fp8_tensor(profile, rows, n, k)
}

/// A prefill rung the static route takes at other rows only: the native GEMM would quantize the
/// activations per row instead of at the checkpoint's `input_scale` (a different arithmetic), so
/// the rung is refused. Below the route's smallest rung (verify spans) the native GEMM stands.
fn static_route_gap(isa: Option<hwspec::IsaLevel>, rows: u32, n: u32, k: u32) -> Option<String> {
    let profile = isa.map_or("", hwspec::IsaLevel::arch_flag);
    let routed = plow_asset::segment_roles::cublaslt_prefill_fp8_tensor_rows(profile, n, k)?;
    (rows >= routed[0] && !routed.contains(&rows)).then(|| {
        format!(
            "prefill rung {rows} ({n}x{k}) is outside the per-tensor FP8 cuBLASLt route's rows \
             {routed:?} on {profile}: the native GEMM would quantize per row, not at input_scale"
        )
    })
}

/// `out[rows, n] = src[rows, k] . module.weight^T`.
pub(crate) fn emit(
    p: &mut Program,
    out: u32,
    src: u32,
    module: &str,
    n: u32,
    k: u32,
    dep: u32,
) -> u32 {
    let w =
        p.b.tensor(&format!("{module}.weight"), u64::from(n) * u64::from(k));
    let scale_name = crate::derived::repeat(&format!("{module}.weight_scale"), n);
    let scale = p.b.tensor(&scale_name, u64::from(n) * 4);
    let rows = p.rows;
    if !p.prefill {
        return p.emit(
            GemvFp8Args {
                C: out,
                x: src,
                W: w,
                w_scale: scale,
                M: rows,
                N: n,
                K: k,
                ..Default::default()
            },
            &[dep],
        );
    }
    let arms = kernelcaps::arms::SM120;
    for (axis, v) in [("N", n), ("K", k)] {
        arms.require(packet::dev::DevOp::GemmFp8, axis, v)
            .unwrap_or_else(|e| panic!("{module}: {e}"));
    }
    let xq =
        p.b.tensor(&format!("act.fp8.x{k}"), u64::from(rows) * u64::from(k));
    if static_route(p.isa, rows, n, k) {
        let wscale = p.b.tensor(&format!("{module}.weight_scale"), 4);
        let input_scale = p.b.tensor(&format!("{module}.input_scale"), 4);
        // q/k/v carry one fused input_scale in the checkpoint, as vLLM's QKV linear takes their max.
        let (dq, a_scale) = match p.quant {
            Some((s, w, a)) if (s, w) == (src, k) => (dep, a),
            _ => {
                p.quant = Some((src, k, input_scale));
                let q = p.emit(
                    QuantFp8Args {
                        xq,
                        x: src,
                        a_scale: input_scale,
                        M: rows,
                        K: k,
                        scale_mode: FP8_TENSOR_SCALE,
                        ..Default::default()
                    },
                    &[dep],
                );
                (q, input_scale)
            }
        };
        return p.emit(
            GemmFp8Args {
                C: out,
                A: xq,
                B: w,
                a_scale,
                w_scale: wscale,
                M: rows,
                N: n,
                K: k,
                scale_mode: FP8_TENSOR_SCALE,
                ..Default::default()
            },
            &[dq],
        );
    }
    if let Some(gap) = static_route_gap(p.isa, rows, n, k) {
        panic!("{module}: {gap}");
    }
    let a_scale = p.b.tensor("act.fp8.scale", u64::from(rows) * 4);
    let dq = if p.quant.is_some_and(|(s, w, _)| (s, w) == (src, k)) {
        dep
    } else {
        p.quant = Some((src, k, a_scale));
        p.emit(
            QuantFp8Args {
                xq,
                x: src,
                a_scale,
                M: rows,
                K: k,
                ..Default::default()
            },
            &[dep],
        )
    };
    p.emit(
        GemmFp8Args {
            C: out,
            A: xq,
            B: w,
            a_scale,
            w_scale: scale,
            M: rows,
            N: n,
            K: k,
            ..Default::default()
        },
        &[dq],
    )
}

#[cfg(test)]
mod tests {
    use super::static_route_gap;

    #[test]
    fn off_route_prefill_rungs_are_refused() {
        let gb10 = hwspec::IsaLevel::from_arch_flag("sm_121a");
        for rows in [16, 128, 1024, 1152, 1536, 2048, 2176, 4096, 8192] {
            assert_eq!(static_route_gap(gb10, rows, 4096, 4096), None);
        }
        let gap = static_route_gap(gb10, 3072, 4096, 4096).expect("3072 is off the route");
        assert!(gap.contains("prefill rung 3072"), "{gap}");
        let hopper = hwspec::IsaLevel::from_arch_flag("sm_90a");
        assert_eq!(static_route_gap(hopper, 3072, 4096, 4096), None);
    }
}
