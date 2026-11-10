"""gen_search block spec: Qwen3-TTS code-predictor depth stage, the residual-codebook head
(GemvF32Out, K = 1024, N = 2048 classes, f32 logits) at B = 1 / 8 / 16 / 32. Bodies:
qtts_depth_gemm.py. The sampler that follows (SampleSmallF32) is not a spec: its pick is keyed by
the exact descending order of the top-k (u64 radix select + rank-keyed Gumbel noise), which neither
generator expresses."""
import qtts_depth_gemm as g

SPEC = g.spec("qtts_depth_head", "f32", "depth head GemvF32Out (1024 -> 2048), M = B rows",
              "bf16 x bf16 -> f32 acc -> f32", 1e-5)
