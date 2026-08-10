# MoE-PF activation-side arms (Phase D2): NEAR-NULL — both traffic models now falsified

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC** — two traffic models falsified at this box's rates.

2026-08-07, fork branch (base worktree-glm52-bringup @ 6ad92f8). Objects
`hsaco_actv` (arms are runtime packet flags — one object set serves all arms;
resource table byte-identical to the shipped rows: 256 VGPR / 64,560 LDS /
spill 2). Blobs `glm_actv_{ctl,p16,full}` in the jobs tmp dir; assets
`glm52-tp8-actv-{ctl,p16,full}`. Control emit (envs unset) verified
byte-identical to the shipped V2 blob.

## What landed (all opt-in, default off)

- **`PLOW_MOE_PF_PART16=1`** — DOWN's `part[T*k,H]` scatter and the combine's
  readback in bf16 (packet flag `i[7]` on ops 86/87; `act.part` declared at
  half width; decode untouched — its expert ops keep f32 in the same, larger
  buffer).
- **`PLOW_MOE_PF_A8=1`** — fp8 gathered activations for the grouped GLU: the
  post-attention RmsNorm's EXISTING fused-quant epilogue (t3/t4) writes
  `act.xn2q` + per-token scales; op 85 reads them (t6/t7, `i[7]=1`), maskless
  6-VALU dequant (the encoder never emits 0x80), per-token scale × the fp8
  decode's ×2 applied per output row BEFORE the GLU nonlinearity.
- Marker symbols `plow_moe_pf_{part16,a8}_arm` + manifest `requires` derivation
  from the packet fields, so an old object REFUSES a flagged blob.

## Measured (interleaved 2 rounds, all arms served PLOW_MLA_PF_V2=1, gates PASS)

| | ctl | p16 | p16+a8 |
|---|---|---|---|
| TTFT @4k | 1192.4 / 1082.2 | 1076.4 / 1069.7 | 1069.2 / 1299.0* |
| TTFT @8k | 1995.1 / 2021.5 | 1990.3 / 1983.7 | 1979.6 / 2133.6* |
| GluPf span @8k (layer 40) | 2313–2375 (preshuffle record) | — | 2244 (−4%) |
| DownPf span @8k | 3421–3427 | — | 3378 (−1.4%) |
| combine span @8k | ~1360 | — | 1357 (wash) |

(*) round-2 full rows are the documented DVFS-ordering outlier class. Best
honest read: **−5..−10 ms @8k (−0.3..0.5%), noise-adjacent.**

## Numerics (the gate class this lever family requires)

Fixed 2048-token prompt vs ctl: **p16 FLIPS top-1** (first token 8004 vs 168;
max|Δ| 0.82 on logit amax 6.22). p16+a8 happens to match top-1 (max|Δ| 0.50) —
same error class, chaotic amplification through 78 layers; the match is not
robustness. The 3-diverse-prompt serve comparison errored (curl payload; not
re-run — the perf verdict below makes it moot). **FAILS the ship gate.**

## The finding that matters: the pair is ISSUE-bound

Halving the part stream (−1.6 GB/layer @8k) and the gathered-A stream
(−0.8 GB) moved the kernel spans ≤4%. With the preshuffle null (B stream) this
closes ALL THREE stream hypotheses: at ~430 GB/s effective and 93% busy the
grouped GEMM is bound by its serialized issue pattern (gather addressing, the
DBUF=1 commit, scatter address math), not by any memory stream. aiter's 2.8×
comes from instruction-level structure. Remaining route for the pair:
asm-class/ISA-scheduled restructure of `d_moe_group_pf_t` (16×16 fragments,
register-resident staging, wave-group role split), not more bandwidth dieting.

## Bonus fix (the real keeper): the gfx942 arm-check was INERT

`build_requires` (plowrt exec/amd.rs) read `/backends/gfx950/requires`
hardcoded, while gfx942 blobs write `backends.gfx942` — so the packet/object
arm refusal has been a no-op for EVERY gfx942 blob (a PART16 blob ran on a
pre-arm object silently: the exact heap-overrun class the check exists for).
Fixed to probe both keys; refusal now measured firing
("packet/object MISMATCH … plow_moe_pf_part16_arm") and the unflagged control
still loads on old objects (no false refusals; marker-less flags warn as
before).

Verdict: keep both arms opt-in default-off (they cost nothing unset —
byte-identical emit, same object resources); do NOT serve them. Phase D
continues at the instruction level, not the byte level.
