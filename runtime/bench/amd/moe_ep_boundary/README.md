# Routed-MoE expert-parallel boundary

This probe measures the concurrent peer-SDMA transport primitive needed when a
routed FFN receives token-sharded activations. It submits every directed peer
copy before waiting; a loop over `plow_hsa_copy_p2p` is intentionally invalid
because that serializes ranks.

The default 6,473,415-byte peer window is the expected BF16 activation
payload per directed peer for `T=8192, H=3584, top-k=16, E=896, EP=8` after
deduplicating multiple experts on the same destination. The return phase has the
same payload and is costed as a second invocation.

```bash
nix develop -c cc -O3 -std=c11 \
  -Iruntime/amd runtime/bench/amd/moe_ep_boundary/p2p_batch.c \
  runtime/amd/hsa_backend.c -lhsa-runtime64 -lpthread -o /tmp/moe-ep-p2p
GPU_LEASE_DIR=/tmp/gpulease perf-data/tools/gpulease -n 8 moe-ep-p2p \
  /tmp/moe-ep-p2p
```

The planner and stable packet/tensor descriptors are in
`packet::moe_ep`. The production graph should select its transport from tensor
placement: replicated inputs filter routes locally and reuse the existing
`[T,H]` reduction; token-sharded inputs use these compact peer windows.

## Full-intermediate isolated gate

The full-I experiment keeps aggregate expert work constant: one EP rank owns
`E/8=112` whole experts and about `top-k/8=2` routes per token. Stage 1 changes
from `N=384` to `N=3072`; stage 2 changes from `K=384` to `K=3072`.

```bash
nix develop -c runtime/bench/amd/moe_ep_boundary/build_full_i.sh /tmp/moe-ep-full-i
GPU_LEASE_DIR=/tmp/gpulease perf-data/tools/gpulease -n 1 moe-ep-full-i-stage1 \
  /tmp/moe-ep-full-i/stage1/full_i_compare \
  /tmp/moe-ep-full-i/stage1/shipping_full_i.elf \
  /tmp/moe-ep-full-i/stage1/reuse.elf --run
```

`filter_gate.py`, `combine_gate.py`, and the stage-2 reference gate require a
ROCm PyTorch environment. They verify stable expert/token/slot ordering,
fixed-slot combine, and the full-I matrix result before timing. All four
specialists are wave64 and the build rejects private memory or spills.
