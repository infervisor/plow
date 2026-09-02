# Kimi-K3 Plow production-path smoke — MI355X TP8

These are real-weight correctness and coverage gates for the generalized AMD
harness. They are not the matched 8192→1024 performance baseline. MTP,
speculative decoding, and FP8 KV were disabled.

## Environment

- Hardware: 8× AMD Instinct MI355X (`gfx950`), TP8.
- Checkpoint: `/home/shaswot/models/Kimi-K3`, 97 safetensor shards,
  1,565,513,142,816 bytes.
- Packet: 334,207,612 bytes,
  `fnv1a64:2015a4eef0a5c508`.
- Runtime: production `plowrt bench` → `ModelMux` → `AmdServe`; BF16 KV,
  greedy decode, `PLOW_MLA_PF_V2=1`, full per-token TP agreement unless noted.
- Objects: pinned Nix ROCm 7.14 builds for B1, B32, and B128. All carry L2
  placement dispatch; B32/B128 carry GEMV walk; the primary flash object carries
  the V2 MLA prefill arm.

## Results

| Gate | Result | TTFT p50 | TPOT p50 | Output throughput | Checksum |
|---|---:|---:|---:|---:|---|
| C1, 64→2, B128 object only | 1/1, no errors | 275.587 ms | 100.716 ms | 5.31 tok/s | `fnv1a64:cc35e253e915e0a5` |
| C1, 64→2, B1 low-rung tier | 1/1, no errors | 275.745 ms | 54.283 ms | 6.06 tok/s | `fnv1a64:cc35e253e915e0a5` |
| C128, 64→2, B128 | 128/128, no errors/sheds | 37,070.728 ms | 395.804 ms | 2.53 tok/s | `fnv1a64:335688db92d85157` |

The C128 run reached decode rung 128. Its 101.25 s measured duration is a
coverage result, not a competitive throughput result: recurrence-safe prefill
chunks are scheduled fairly but remain one request per packet.

## Findings

- Low-rung object tiers are required for latency. Selecting the B1 object cut
  the measured inter-token interval 46.1% with identical output.
- B128 decode capacity is functional, but it does not compensate for serialized
  prompt execution. True prompt co-packing requires per-row owner, KV base,
  position, KV length, and recurrence-boundary fields in the packet ABI.
- A C2 9000→2 fairness/interleave gate completed both requests with no
  rejects/sheds and reached decode rung 2. The chunk order alternated slots
  `0,1,0,1`; pending chunks are now re-split if a newly decode-live request
  lowers the per-tick interleave cap.
- The corrected K1/K4 gate produced identical eight-token output
  (`fnv1a64:affdb8f705469ce9`) and 7 vs 2 decode scheduler ticks. End-to-end
  duration was 655.69 vs 653.60 ms, so deferred readback is correct but showed
  no resolved speedup in this short cell. It remains opt-in through a TP
  agreement cadence above one; true launch-level multi-step still needs
  device-side counter audit/rearm or a bounded segment packet.
- The current gfx950 tuning database was stale against the emitted build
  fingerprint. Analytical fallback selected the asset recipes, so the kernels
  are not all already optimized for this exact build.
