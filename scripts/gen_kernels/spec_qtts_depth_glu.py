"""gen_search block spec: Qwen3-TTS code-predictor depth stage, the fused gate|up GEMM with the
SiLU-GLU epilogue (K = 1024, N = 3072 per weight) at B = 1 / 8 / 16 / 32. Bodies: qtts_depth_gemm.py."""
import qtts_depth_gemm as g

SPEC = g.spec("qtts_depth_glu", "glu", "depth GemvGlu (gate|up, silu(gate) * up), M = B rows",
              "bf16 x bf16 -> f32 acc -> bf16", 3e-3)
