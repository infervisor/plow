# Vendored kernel reference sources

The performant kernel tier (opcode variants `0x01+`) is extracted/adapted from
these repositories. Add each as a git submodule (preferred — avoids
redistribution and records the commit) or a pinned copy, and record the source
commit + license here.

| Dir | Repo | License | Used for |
|-----|------|---------|----------|
| `fast.cu/` | github.com/pranjalssh/fast.cu | MIT | Hopper bf16 GEMM (wgmma + TMA) |
| `ThunderKittens/` | github.com/HazyResearch/ThunderKittens | (verify) | Hopper+Blackwell GEMM/attention; HipKittens = MI300 |
| `DeepGEMM/` | github.com/deepseek-ai/DeepGEMM | MIT | FP8 GEMM (SM90/SM100 tcgen05), grouped/MoE |
| `LiquidGEMM/` | (W4A8 GEMM, Hopper) | (verify) | W4A8 quantized GEMM |

Adapters that map our `Body` structs onto each launcher live in
`runtime/nvidia/*.cu` and `runtime/amd/*.hip` (the `*_bf16_*`, `*_fp8`, `*_w4a8`,
`*_mfma` entry points). Verify license compatibility before committing any
vendored source into this tree.

Suggested submodule setup:

```sh
git submodule add https://github.com/pranjalssh/fast.cu          runtime/extern/fast.cu
git submodule add https://github.com/HazyResearch/ThunderKittens runtime/extern/ThunderKittens
git submodule add https://github.com/deepseek-ai/DeepGEMM         runtime/extern/DeepGEMM
```
