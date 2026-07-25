# M1-multimodel — S1 model switching + co-residency on sm_120 (2026-07-20)

Source data: `m1-multimodel-sm120.json`. Harness: `crates/plowrt/tests/gpu_multimodel.rs`
(the serve path minus HTTP: manager → mux → engine), release build, under `gpulease`.
Models: gemma-4-12B and gemma-4-26B-A4B, both ctx-132096 b1 bf16 serve blobs.

## What S1 costs now (the honest re-measure of wave-4 item 7b)

The 5e880f6 "switch ≈ 0" was a **residency** switch (both models resident, pass a
different kernarg) at ctx=512. S1 under a VRAM budget is a different animal: the
target's engine does not exist and must be built — and the cost is **entirely the
target's checkpoint H2D**, not anything about the outgoing model:

| direction | outgoing live KV | drain | unload | load | first token | TOTAL |
|---|---|---|---|---|---|---|
| 12B→26B | 4k  | 0.0–0.1 ms | 122–149 ms | 8.55–10.10 s | 43–44 ms | **8.7–10.3 s** |
| 26B→12B | 4k  | 0.0–0.1 ms | 181–185 ms | 4.89–5.44 s | 56 ms | **5.1–5.7 s** |
| 12B→26B | 32k | 0.1 ms | 137 ms | 10.14 s | 44 ms | **10.3 s** |
| 26B→12B | 32k | 0.1 ms | 191 ms | 4.88 s | 55 ms | **5.1 s** |

- Live KV depth on the victim (4k vs 32k) changes nothing measurable: eviction is
  driver frees (~120–190 ms flat) — KV dies with the engine either way.
- Load ≈ weights / ~4.6 GiB/s effective (26B 47.0 GiB, 12B 22.2 GiB through the
  64 MiB pinned staging loop + blob tables + module loads). The 8.55 vs 10.1 s
  spread on 26B is page-cache state of the checkpoint on disk.
- First token after the switch is normal TTFT (43–56 ms on a short prompt) — the
  fresh engine has no warm-up penalty.

## Correctness gates (all green)

- A(Paris) → switch → B answers Paris → switch back → A **token-identical** to its
  first run, at both outgoing-KV depths.
- VRAM back to plan at every state: A resident 32882 MiB, B resident 56964 MiB,
  bit-for-bit repeatable across 4 switch rounds — no leak (planner bounds
  `resident weights ≤ used ≤ Σrequired + 2 GiB` assert each state).
- Long-prefill no-shed gate: a live 64-token decode stream + a ~6k-token prompt
  arriving mid-decode at the default `--slo-ms 250` both complete (the admission
  EWMA fix; pre-fix the 32k prime shed itself with "arrival-rate admission shed").

## Co-residency (planner-driven, measured)

12B + 26B (both at compiled ctx 132096, bf16) **co-fit 96 GB**: 89 286 MiB used,
both models stream concurrently, zero switches. Requests interleave via the
per-model dispatchers; GPU execution serializes structurally — one CUDA context,
cooperative launches on the NULL stream, each engine syncs under its own mutex
before readback — so ticks from different models can never interleave within a
launch and no global lock is needed.

Matrix vs 96 GB (per-model plans blob-verified: 12B 31.2 / 26B 54.9 / 31B 82.8 GiB
at ctx 132096; fp8 rows computed from measured fp8 weight bytes, no fp8 blobs on
disk): 12B+26B bf16 FITS (measured); 12B+31B and 26B+31B bf16 do NOT at 132k;
fp8 makes every pair fit (12B+31B fp8 ≈ 74 GiB — the wave-4 target combo), and
even 26B+31B fp8 ≈ 86 GiB fits tight; all three fp8 (~106 GiB) needs reduced ctx.

## What S1 cannot do (that S2 would)

1. Every non-resident hit pays seconds (weight H2D): 5–10 s here. S2 keeps both
   resident and pays ~0 (the measured 5e880f6 residency switch).
2. No preemption: the switch drains the victim — an in-flight long generation
   delays the switch for its full remaining duration.
3. Victim KV is destroyed: evicted conversations re-prefill from scratch on
   return (no KV save/restore).
4. Requests racing the victim's teardown get 503 (dispatcher removed), not queued.
5. Thrash: alternating A/B traffic pays a full reload per alternation (LRU has no
   hysteresis).
6. KV arenas are compiled into the blob — the planner accounts them but cannot
   resize at admission time; "KV squeezing" a pair into VRAM means registering a
   smaller-ctx blob, not a runtime split.
