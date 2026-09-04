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
