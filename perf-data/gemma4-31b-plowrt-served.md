# Gemma-4-31B served through `plowrt serve` — first end-to-end 31B serving (sm_120)

**Campaign S6-31b**, 2026-07-20, `main` @ `a058492`. The 12B serving path
(gemma4-12b-plowrt-served-tpot.md) applied unchanged to 31B — the engine is
model-agnostic; only the assets differ. Data: `gemma4-31b-plowrt-served.json`.

## Assets (`/root/gpu-assets-31b/b1`, kept in place for the final-numbers pass)

- **Packet:** `PLOW_UNISEG=1 gemma4 /workspace/models/gemma-4-31B-it 132096
  gemma4-31b-ctx132096-b1.pkt 188` at HEAD (430 MiB). The ns47 grid-aligned
  full-layer nsplit (T9b) is the emitter default for the 31B signature. ctx
  132096 is a shipped pow2-ring shape (`kv_ring` asserts non-pow2 rings,
  leak-audit #6; sliding layers ring at `KV_RING`, full layers linear).
- **Cubins:** HEAD `scripts/build_sm120_cubin.sh` pair (same objects as 12B):
  decode 212 regs / prefill `_pf` 240 regs, smem_pf 81664, 0 spills.
- Plus `tokenizer.json` (copy), `checkpoint` →
  `/workspace/models/gemma-4-31B-it`, `weights.json` (network
  `gemma-4-31b-it`, buckets []).
- **Engine load: 57.2 GiB weights + 22.6 GiB KV @132k, 21.6 s warm.** With
  activations ≈ 82.5 GiB of the 96 GiB card — one 132k sequence, as the
  capacity report's KV-wall arithmetic predicted.

## Gates (all pass)

- "capital of France" (greedy) → **"The capital of France is Paris."**,
  `finish stop`.
- **Token parity, served vs standalone harness: 48/48.** Two-phase
  capture-replay: the 82.5 GiB harness cannot co-reside with the server, so
  phase A captures the served prompt ids + greedy stream from the mux debug
  log, the server exits, and phase B replays the identical ids through
  `gemma4_sm120_chat` `PLOW_PREFILL=1` on the same pkt
  (`/root/gpu-assets-s6/scripts/parity_offline.sh`).

## Served rows, single user, greedy (vs vLLM bf16 sm_120 baseline)

| target | n_prompt | served TTFT | served TPOT | vLLM TTFT | vLLM TPOT | TTFT ratio | TPOT delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4k  | 4137  | **1.65 s**  | **46.36 ms** | 0.706 s | 45.20 ms | 2.3× | +2.6% |
| 32k | 32837 | **14.12 s** | **48.78 ms** | 10.73 s | 49.14 ms | 1.3× | **−0.7% (plow wins)** |

- **TPOT lands on the committed raw-decode gap** (+1.4–5.4% behind at short
  ctx, T9 win at 32k): serving adds nothing measurable on a 46 ms step.
- **Served TTFT@32k is +0.5% over the raw prefill wall** (14.045 s cover-policy
  harness, same lease). The 4k served point (1.65 s) beats the raw cover
  number (2.69 s) via [4096,128] chunk decomposition — same mechanics as 12B.
- Server run with `--slo-ms 100000000` — the admission-shed regression
  workaround documented in the 12B S6-refresh notes.
- Batch is 1 on this blob; B>1 dense blobs exist for 12B only so far.
