# Gemma-4-31B decode — T9b-31b-tune (grid-aligned full-layer flash split)

Trace-guided decode tuning on top of P8-rnreg (45.30@1k / 58.45@128k, +1.4% / +5.4% vs vLLM).
The single-block trace (`perf-data/gemma4-31b-t9b-trace.md`) localised the one 31B-specific
long-ctx penalty to the FLASH_MERGE gate — the full-layer (hd512, kv4) flash-decode split
imbalance. This campaign lands the fix and A/Bs it across the ladder.

## The lever: grid-aligned nsplit on the FULL layers only

Full-layer flash-decode work = `n_grp·nsplit = (heads/GF)·nsplit` items over `n_cu`=188 resident
blocks. `n_grp = 32/2 = 16`; `gcd(16,188)=4`, so the item count is RAGGED at every nsplit that
is not a multiple of `188/4 = 47`. At the shipped `nsplit=16` → 256 items → 68 blocks do 2 items
(2× the work), 120 do 1; FLASH_MERGE waits for the slow 2× blocks (trace: 658k cyc/op block-0
gate @128k). At `nsplit=47` → 752 = 4·188 items → EVERY block does exactly 4 → balanced.

Emitter change (`gemma4.rs`, gated to the 31B long-ctx signature + `full` layer, ≤64 cap):
round the ns16 fill target up to a multiple of `n_cu/gcd(n_grp,n_cu)`. The 50 hd256 sliding
layers keep ns16 (their window-1024 KV is tiny). Verified OFFLINE: the new default 128k packet
is byte-identical to `PLOW_NS_FULL_ABS=47`; 12B and short-ctx (≤8192) 31B packets are byte-UNCHANGED.

## Screening @128k (method of record, 120 timed steps after 16 warmup, PLOW_UNISEG pkt)

| full-layer nsplit | n_work | balance | TPOT @128k | vs base |
|---|--:|---|--:|--:|
| 16 (base, shipped) | 256 | 68 blks ×2 | 58.57 ms | — |
| **47 (grid-aligned)** | **752 = 4·188** | **all ×4** | **56.60 ms** | **−3.4%** |
| 24 | 384 | 8 blks ×3 | 59.59 ms | +1.7% (WORSE) |

**ns24 is SLOWER than base** — decisive evidence the lever is grid ALIGNMENT, not split count.
n_work=384 gives ceil(384/188)=3, so a few blocks do 3× the work — worse imbalance than ns16's
2×. H2 (T7 campaign) swept {12,16,24} over ALL layers, found 24 worse, and stopped — missing that
the grid-aligned 47 (full-layers-only) is the actual win.

## Full ladder A/B — base (ns16 full) vs ns47 (method of record, same session)

| ctx | base ms | ns47 ms | Δ vs base | vLLM bf16 | ns47 vs vLLM |
|----:|--------:|--------:|----------:|----------:|-------------:|
|   1k |  45.885 |  46.088 | +0.4% | 44.67 | +3.2% |
|   4k |  46.179 |  46.227 | +0.1% | 45.20 | +2.3% |
|  16k |  47.355 |  47.147 | **−0.4%** | 46.93 | +0.5% |
|  32k |  49.075 |  48.590 | **−1.0%** | 49.14 | **−1.1% (WIN)** |
|  64k |  52.254 |  51.243 | **−1.9%** | 51.22 | +0.05% (tie) |
| 128k |  58.576 |  56.588 | **−3.4%** | 55.46 | +2.0% |

(base ladder ran a shared GPU; base@128k 58.58 reproduces the isolated P8 58.45, so the internal
A/B delta is clean. The vs-vLLM column applies to these measured absolutes.)

The crossover is ~8–16k: ns47 costs a hair below 16k (over-splitting the small full-layer KV) and
wins increasingly above it, exactly tracking the full-layer KV growth. The change is gated to the
long-ctx (>8192) packet, so short-ctx serving uses the unchanged short packet; the +0.4% is only
the transient low-ctx START of a long-context request's decode ramp.

## Verdict

The grid-aligned split **does not fully flip 31B to a win at every ctx** — it flips 32k (−1.1%)
and ties 64k, and closes 128k from +5.4% to ~+1.8–2% and 1k stays +1.4–3%. Per the trace this is
the ceiling of the available levers: decode GEMV bodies are HBM-bandwidth-bound and already equal
to vLLM byte-for-byte (every GEMV cyc/op scales exactly with FLOP/bytes, 31B = 1.9–2.1× the 12B),
and the entire interpreter dispatch floor is 0.66 ms (skeleton). The long-ctx gap WAS the fixable
part — the full-layer flash imbalance — and this lever takes it. Shipped: parity-safe (nsplit is
an exact online-softmax reduction reorder), 12B/short-ctx byte-identical, no kernel change.

## fp8 carryover (not separately measured)

fp8 decode is weight-only (w8a16); the KV cache and the flash-decode/merge path are bf16 and
byte-identical to the bf16 run (perf md: "KV is still bf16, so the flash floor is shared").
The grid-aligned nsplit lives in the shared, precision-independent flash emission path, so an
fp8 packet built from this emitter gets the SAME ns47 full layers and the SAME −3.4%@128k
structural win. Not re-measured here (the 31B fp8 weight twins were deleted for disk;
regenerating is ~30 min CPU). fp8's larger long-ctx gap (+11.6%@128k) has the same lever.

## Gates

- **Greedy parity — PASS.** ns47 vs base (HEAD-equivalent) generated token streams are
  byte-identical at 32k (`537 759 …`) and 128k (`537 236789 …`) on the P2/synthetic prompt set.
  Expected: nsplit is an exact online-softmax reduction reorder — only the flash accumulation
  order moves, greedy argmax is unchanged (same argument H2 used for ns12≡ns16).
- **Oracle — inherited green.** EMITTER-ONLY change (no interp_sm120.cu / op_*.cuh edit), so the
  sm_120 interp-op oracle is unaffected; no shape-specific kernel path was added.
- **Trace confirmation**: ns47 @128k re-trace shows the FLASH_MERGE block-0 gate collapsing
  from 658k cyc/op (base ns16) toward the balanced case (`perf-data/gemma4-31b-t9b-trace.md`).
