# V1-vmm-prefix — VMM-backed KV prefix sharing (HBM dedup)

Campaign **V1-vmm-prefix**, 2026-07-20. NVIDIA RTX PRO 6000 Blackwell Server
Edition (188 SMs, 95.0 GiB), driver 580.82.07, CUDA 13.0. plowrt
`--features cuda`, release. Implementation: `crates/plowrt/src/memory/vmm.rs`
(+ `device/cuda.rs` VMM driver surface, `exec/gpu.rs` KV binding), flag-gated
`PLOW_VMM_PREFIX=1`, default **off**. Design + measured feasibility review:
the design notes.

## What is measured

Full-attention-layer KV for a shared prompt prefix is held once in HBM and
`cuMemMap`'d (same physical handles) into every sharing sequence's VA window.
Sliding layers stay on cudaMalloc rings; their last `window=1024` rows are
restored from a boundary snapshot. The win is **HBM dedup per sharer**
(10 GiB at 31B@128k), not admission latency.

## 1. Attach vs D2D copy (pool-level, 31B@128k-class geometry)

10 full layers × 4 kv heads × {K,V} × 1 KiB rows (80 KiB/token), max_ctx
128k, measured through the production `VmmOps` entrypoints
(`gpu_vmm_prefix::attach_latency_vs_copy_baseline`):

| prefix rows | block | owner build ms | ATTACH ms | detach ms | D2D copy ms | deduped |
|---|---|---|---|---|---|---|
| 4096   | 2 MiB  | 14.8  | 12.2  | 6.3   | 0.47  | 0.31 GiB |
| 32768  | 2 MiB  | 122.1 | 99.0  | 49.8  | 3.68  | 2.5 GiB |
| 32768  | 16 MiB | 14.3  | 12.0  | 6.2   | 3.68  | 2.5 GiB |
| 131072 | 2 MiB  | 497.6 | 395.6 | 203.1 | 14.69 | 10 GiB |
| 131072 | 16 MiB | 55.8  | 46.2  | 24.3  | 14.69 | 10 GiB |
| 131072 | 64 MiB | 14.7  | 11.6  | 5.9   | 14.69 | **10 GiB** |

- Reproduces the feasibility review within noise: attach ≈ 0.07 ms/block
  (`cuMemSetAccess`-bound), copy at ~730 GB/s.
- **64 MiB blocks: attach 11.6 ms beats the 14.7 ms copy at 128k** while
  deduping the full 10 GiB — the production default block size.
- Owner-side build (create+map+setaccess) is paid at prefill, amortized.

## 2. Shared-vs-independent correctness + dedup + TPOT (Gemma-4-12B, ctx 8k)

`gpu_vmm_prefix::shared_prefix_token_identity_and_dedup`: prompt = 4200-token
shared prefix + divergent suffixes, greedy, 32 new tokens, batch-4 engine
(`gemma4-12b-ctx8k-b4.pkt`), sharing blocks 2 MiB (2048 tokens — the 8k-ctx
test geometry; dedup mechanics are block-size-independent).

- **Greedy token identity** (the correctness bar): sequence A (prefix+sufA),
  sequence B (prefix+sufB, attached), and a 7400-token sharer all produce
  token streams **identical** to the same prompts served independently on the
  default cudaMalloc engine — no cross-sequence bleed, prefix reads byte-exact.
- **Dedup ledger (exact):** A (publisher) created 64 blocks; B (sharer)
  created 48 and multi-mapped **32 shared blocks (64 MiB of prefix KV not
  re-created)**; `created_B + shared == created_A + tracks` held exactly
  (the +16 is B's displaced row-0 blocks). VRAM after A 36098 MiB → after B
  36130 MiB (+32 MiB for B's tail+lookahead blocks only).
- **TPOT neutrality (gate d):**

  | ctx | default (cudaMalloc) | PLOW_VMM_PREFIX=1 | delta |
  |---|---|---|---|
  | ~4.2k | 20.344 ms | 20.342 ms | −0.0% |
  | ~7.4k | 20.490 ms | 20.487 ms | −0.0% |

## 3. Leak cycle

`gpu_vmm_prefix::vmm_leak_cycle`: load → publish+attach serves → drop, twice;
VRAM back to baseline within 64 MiB each cycle.

Both cycles: baseline 558 MiB → engine + shared serves → after unload
**558 MiB** (exact return; pool unmaps windows, releases every block and
snapshot, frees the VA reservations on engine drop).

## Remaining for production default

`PLOW_VMM_PREFIX` stays **off** by default: flip needs the mux/multi-model
merge coordination, an end-to-end 128k-ctx asset run at the 64 MiB default
block (e2e here validated at ctx 8k with 2 MiB blocks; pool-level 128k
measured above), fp8-KV backing (falls back to cudaMalloc today), and a
block-size policy call for attach TTFT vs sub-block tail recompute
(≤ block_rows tokens re-prefilled per admission).
