# Gemma-4-26B-A4B (MoE) served through `plowrt serve` — first measured rows (sm_120)

**Campaign S6-26b**, 2026-07-20, `main` @ `a058492`. Completes the P9
serving bring-up (plans/p9-26b-campaign.md logged smoke + word-approximate
ctx only) with token-exact prompts, gates, and committed rows. Data:
`gemma4-26b-plowrt-served.json`.

## Servable verdict

**26B serves end-to-end with no engine gap.** The serve engine's name-driven
fused expert-table fill (`moe.ewt.{l}`/`moe.est.{l}` in `exec/gpu.rs`, the
port of `orch::moe::build_fused_expert_table`) resolves the MoE tables from
the checkpoint at load — no sidecar, no serve-side wiring missing. The one
structural limit stands: the 26B decode blob is **B=1** (flat expert kernels
index one token row), so concurrency >1 needs the batched-MoE blob work
(agent A of the batching campaign), not serving work.

## Assets (`/root/gpu-assets-26b/b1`, kept in place)

- **Packet:** `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 gemma4
  /workspace/models/gemma-4-26B-A4B-it 132096 … 188` — the committed P9
  recipe; re-emitting at HEAD reproduces the P9 pkt **md5-identical**
  (04c807bd…), so the existing blob was kept.
- **Cubins:** rebuilt at HEAD with the P9 GF4 recipe (the
  `build_sm120_cubin.sh` flags + `-DPLOW_NV_FA_GF_FULL=4`): decode **219
  regs**, prefill `_pf` **240 regs**, 0 spills, arena sizes embedded
  (`PLOW_NV_EMBED_SMEM` — the 3c32db9 GF4-arena fix depends on it).
- **Engine load: 47.0 GiB weights + 5.6 GiB KV @132k, 21.0 s warm.**

## Gates (all pass)

- "capital of France" (greedy) → **"The capital of France is **Paris**."**,
  `finish stop`.
- **Token parity, served vs standalone `build-gf4` harness: 48/48**
  (two-phase capture-replay, same protocol as 31B —
  `/root/gpu-assets-s6/scripts/parity_offline.sh`).

## Served rows, single user, greedy (vs vLLM bf16 sm_120 baseline)

| target | n_prompt | served TTFT | served TPOT | vLLM TTFT | vLLM TPOT | TTFT ratio | TPOT delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4k  | 4137  | **0.30 s** | **8.48 ms** | 0.169 s | 7.90 ms | 1.8× | +7.4% |
| 32k | 32837 | **2.72 s** | **9.22 ms** | 1.544 s | 9.57 ms | 1.8× | **−3.6% (plow wins)** |

- The 32k TPOT win through the full serving stack matches the P9/T9 raw
  ladder (26B wins 16k–128k after P9 + router-split).
- Raw prefill same lease (cover-policy harness): 451 ms @4137, 2626 ms
  @32837 → served 32k TTFT +3.6% over raw (includes tokenizing a 32.8k-token
  prompt on the handler task).
- Server run with `--slo-ms 100000000` — admission-shed workaround, see the
  12B S6-refresh notes.
