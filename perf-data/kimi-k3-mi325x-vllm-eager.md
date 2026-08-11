# Kimi-K3 vLLM eager baseline attempt on MI325X

Date: 2026-08-11. Hardware: 8 leased MI325X GPUs, gfx942. Client:
flake-pinned vLLM 0.27.0. Server image: official Kimi-K3 AMD image pinned at
`sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b`.

The official recipe requires a nightly vLLM build and lists MI355X as the
verified AMD target. The image advertises gfx942 code, so a TP8 MI325X attempt
was still made with native inferred MXFP4, bf16, `--enforce-eager`, model length
1024, 32 sequences, and the recipe's AITER environment.

No benchmark number is valid. All eight workers segfaulted in RCCL 2.27.7
`ncclCommInitRank` before weight loading. The failure reproduced in the full
image namespace and with `NCCL_IB_DISABLE=1`; RCCL also warned that the host
kernel lacks `iommu=pt`. The endpoint never became healthy, so `vllm bench
serve` was intentionally not run and throughput is N/A rather than zero.

Evidence:

- final debug log: `/tmp/vllm-k3-eager-20260811/server.log`
- native reproduction: `/tmp/vllm-k3-eager-20260811/server-attempt3-native-rccl-segfault.log`
- full-image reproduction: `/tmp/vllm-k3-eager-20260811/server-attempt4-proot-rccl-segfault.log`

The comparison remains blocked on a working gfx942 RCCL/vLLM environment. It
must not use the recipe's gfx950 result as an MI325X same-box baseline.
