# Kimi-K3 DSpark five-layer TP8 block gate

Date: 2026-08-11. Hardware: 8x MI325X (`gfx942`, 304 CUs/rank). Plow
toolchain: repository Nix ROCm 7.14. The independent oracle used one GPU from
the official Kimi-K3 vLLM image rootfs: Python 3.12.13, torch
2.11.0+gitd0c8b1f, vLLM 0.1.dev19253+g5f76ae224.d20260727, HIP 7.2.53211.

## Decision

The complete five-layer, seven-query DSpark block advances to target-verifier
integration. It is numerically consistent with the independent implementation
and costs 5.208 ms per TP8 dispatch at a 32-row context. This is a draft-body
measurement, not speculative served throughput: context projection/KV
precompute, final norm, shared target head, rank-256 Markov head, eight-row
target verification, acceptance, and recurrent-state commit/rollback are not
yet in this packet.

## Numerical gate

The packet has 95 instructions and ten TP collectives. Its five MLA layers use
seven independent query rows over one committed context. The runtime derives
the query width from the packet and writes consecutive positions to `in.pos`.

An asymmetric matched negative control proves all seven positions are consumed:

- positions `[32..38]` vs `[33..39]` change 56,790/100,352 final bytes;
- every 14,336-byte query row changes independently;
- with position-dependent RoPE replaced by zero RoPE, the same position change
  produces zero final-byte differences.

The official single-GPU PyTorch/vLLM implementation was driven from the exact
same BF16 activation, five per-layer BF16 context caches, weights, positions,
and non-causal attention rule. Against Plow TP8:

| output | relative RMS | cosine | max abs |
|---|---:|---:|---:|
| layer 0 `xnext` | 0.00211080 | 0.9999977728 | 0.25 |
| layer 4/final `xnext` | 0.00200780 | 0.9999979844 | 0.50 |

The oracle report SHA256 is
`c7960b843d627dcc2b57000fbd4e7d482986f3a62e077b3ed5a206343b423787`.

## Timing gate

`amd-block --warmup 2 --reps 21` restores only `act.x` before each dispatch,
outside the timed interval. This is required because the five-layer packet
ping-pongs and overwrites the input activation. The timed region includes
position/KV-length preparation, local/cross-rank counter handling, all eight
interpreter launches, GPU drain, compact exact TP audit, and rank agreement.

| samples | median | p90 | mean |
|---:|---:|---:|---:|
| 21 | **5.208414 ms** | 5.231244 ms | 5.210209 ms |

The final output is finite, nonzero, byte-identical on all eight ranks, and
matches the one-dispatch control SHA256
`d7d36ce4026fe0664566d706c939a658209770d8053e9904f858d9602e519646`.
The post-run GPU audit is clean.

Artifacts:

- packet SHA256: `661ad6192059050d9076dad654f8ea3fa34eae01b2b4e82caec7cd0cf4dca3e6`;
- DSpark GQ object SHA256:
  `0188a609c0d152be89ba62cb6742d1762045fbb210f1550b989ce2feb92fdcf3`;
- timing log SHA256:
  `33073b715b1e3a83df9a5f46f2102eef171d251c504a5d6c64aa6f15541ea527`.

## Reproduction

```bash
nix develop --command env \
  PLOW_TP_AUDIT_COMPACT=1 GPU_LEASE_DIR=/tmp/plow-gpulease-shared \
  perf-data/tools/gpulease -n 8 dspark-five-timing \
  ./target/release/plowrt amd-block \
    --blob /tmp/plow-k3-dspark-five-fixed/model.pkt \
    --hsaco /tmp/plow-dspark-hsaco.EQW6EQ \
    --checkpoint /home/lava/models/Kimi-K3-DSpark-plow \
    --load-dir /tmp/plow-k3-dspark-five-fixed-fixtures \
    --decode-pos 32 --decode-kvlen 32 --tp 8 \
    --warmup 2 --reps 21 --inspect act.xnext
```

The eight-row target verifier and versioned KDA/conv state are implemented,
but the full-depth serial-equivalence gate fails. Speculative serving and
acceptance measurement remain blocked; see
`perf-data/kimi-k3-mi325x-dspark-verifier.md`.
