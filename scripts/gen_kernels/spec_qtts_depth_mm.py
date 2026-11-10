"""gen_search block spec: Qwen3-TTS code-predictor depth stage, the plain small-M GEMMs
(position projection, q|k|v, o_proj, down) at B = 1 / 8 / 16 / 32. Bodies: qtts_depth_gemm.py."""
import qtts_depth_gemm as g

# bf16 output: the rounding alone is ~1.1e-3 rel. L2 against fp64.
SPEC = g.spec("qtts_depth_mm", "mm", "depth GEMV family (proj / qkv / o / down), M = B rows",
              "bf16 x bf16 -> f32 acc -> bf16", 3e-3)
