# fp8 (e4m3) latent KV for the MLA family — measurement                      [MLA-FP8-KV]

`FlashMlaDecodeFp8(109)` / `FlashMlaPrefillFp8(110)`. Built, validated, measured, and
**deliberately not routed by any emitter** — both are in `GFX950_UNEMITTED` with this file's
numbers as the reason. What follows is why that is a numerics decision and not a wiring omission.

Hardware: MI355X (gfx950), `ROCR_VISIBLE_DEVICES=1` (= KFD node 3 = rocm-smi device 0), ROCm 7.x.
Gate: `runtime/tests/k3_mla_block_gfx950_test.c` on **real Kimi-K3 layer-3 weights**, 21 scored
rows at TOL 1.5e-2, `K3_MLA_FP8=0|1|2`, contexts 64 / 512 / 4096.

    ./scripts/build_k3_mla.sh                          <out-bf16>
    PLOW_EXTRA_DEFS=-DPLOW_FP8_KV=1 ./scripts/build_k3_mla.sh <out-fp8>
    K3_MLA_CTX=<c> python3 runtime/tests/k3_mla_oracle.py <fixture>
    K3_MLA_FP8=<0|1|2> ./k3_mla_test interp_decode.elf <fixture>

## 1. The gap it closes

`PLOW_FP8_KV=1` swapped fp8 KV in for the DENSE GQA family only (ops 37/38/39). The MLA family had
no fp8 twin, so DeepSeek, Kimi-K2.7, GLM-5.2 and Kimi-K3 — every model whose KV is the largest —
could not use it. K3's latent is `ckv`(512) + `krot`(64) bf16 over 24 MLA layers = **1152
B/token/layer = 27.0 KiB/token = 3.38 GiB at 128k**.

## 2. Accuracy — rms relative error vs the model's own `k_pass`/`value` reference

| row | ctx 64 | | | ctx 512 | | | ctx 4096 | | |
|---|---|---|---|---|---|---|---|---|---|
| | bf16 | ckv fp8 | +krot fp8 | bf16 | ckv fp8 | +krot fp8 | bf16 | ckv fp8 | +krot fp8 |
| K0 `kv.ckv` write | 0 | 2.659e-2 | 2.659e-2 | 0 | 2.659e-2 | 2.659e-2 | 0 | 2.659e-2 | 2.659e-2 |
| K1 `kv.krot` write | 0 | 0 | 2.564e-2 | 0 | 0 | 2.564e-2 | 0 | 0 | 2.564e-2 |
| **M2 MLA output** | **1.069e-3** | **2.690e-2** | **2.777e-2** | **1.918e-3** | **2.855e-2** | **2.933e-2** | **2.124e-3** | **2.961e-2** | **2.856e-2** |
| G1 output gate | 3.47e-5 | 2.728e-2 | 2.811e-2 | 0 | 2.957e-2 | 3.033e-2 | 2.49e-7 | 3.107e-2 | 2.991e-2 |
| A2 o_proj | 2.75e-4 | 2.748e-2 | 2.836e-2 | 1.88e-7 | 3.010e-2 | 3.093e-2 | 1.97e-7 | 3.075e-2 | 2.966e-2 |
| M9 block out | 7.32e-4 | 3.552e-3 | 3.647e-3 | 2.66e-4 | 2.727e-3 | 2.804e-3 | 3.19e-4 | 1.924e-3 | 1.885e-3 |

All 21 rows and the bitwise NoPE checks pass in all three modes (in mode 2 the k-side check becomes
"the stored bytes are the exact e4m3 of `krr`" — still a byte comparison, not a tolerance).

Rows Q1–Q3, A0/A1 and the MoE tail are unchanged by the axis and are omitted; the harness prints
all 21.

## 3. The three things this had to establish, and what it found

**3a. The write is EXACT.** Row K0 = 2.659e-2 against an e4m3 round-trip floor of **2.658e-2**
measured on the same reference row by the harness. `d_headnorm_rope_fp8<512>` adds nothing to what
e4m3 costs.

**3b. The shared latent does NOT amplify.** This was the MLA-specific risk: one latent row is read
by all 96 query heads and consumed through `q_absorb` and `MlaMergeFold`, so a quantization error
here is common-mode where a dense-GQA one is confined to a head.

    M2 / e4m3-floor  =  1.012  1.074  1.114   (ckv fp8,   ctx 64 / 512 / 4096)
                        1.045  1.103  1.075   (+krot fp8, ctx 64 / 512 / 4096)

The attention output carries the cache error and no more. This is now a hard gate condition
(`amp < 1.3`), so a future change that DOES amplify fails rather than drifts.

**3c. It does not average down with context either.** 2.69e-2 @64 → 2.96e-2 @4096. Attending over
more rows does not help: `W_uv` cancels signal faster than noise on the way out.

## 4. Scale placement is NOT a lever

Same reference row, re-quantized with a **per-128-element block** scale instead of the per-row one:

    per-row scale   2.658e-2
    per-128 block   2.552e-2      (4% better)

The error is e4m3's 3 mantissa bits, and every RMS-significant element is already in a normal
binade — the row scale maps amax onto 448, so nothing that carries weight lands in a subnormal. No
scale placement (per tensor, per row, per block) can move this. Per-row is therefore the right
choice on its own merits (one f32 per 512 stored bytes, written once at the row's own step, no
second pass), not merely by inheritance from `HeadNormRopeFp8`.

## 5. `ckv` only vs `ckv` + `krot`

| | ckv fp8 only | ckv + krot fp8 |
|---|---|---|
| bytes/token/layer | 1152 → **644** (44.1% saved) | 1152 → **584** (49.3% saved) |
| K3 @128k, 24 MLA layers | 3.38 → **1.89 GiB** | 3.38 → **1.71 GiB** |
| M2, mean over 3 contexts | 2.84e-2 | 2.86e-2 |
| NoPE bit-exactness | **kept** | **lost** |

**Ship `ckv` only.** The extra 5.2 pp of saving costs an M2 delta inside run-to-run noise (mode 2 is
*better* than mode 1 at ctx 4096), so on the numbers alone it is nearly free — but it destroys a
property K3 has today. With `mla_use_nope` the rope table is the identity, so `krot` is a bit-exact
copy and the gate checks it BITWISE; quantizing it is a strictly new error source on a path that
currently has none, and it converts a byte comparison into a tolerance. 11% more of the cache is
not worth that. The arm exists (`i6 = krot_fp8`, a runtime bit on ONE object so the A/B is not
confounded by object size) and the shipped form sets it to 0.

## 6. Object cost

| object | VGPR | AGPR | occ | VGPR spill | SGPR spill | LDS | size |
|---|---|---|---|---|---|---|---|
| decode bf16, before | 248 | 0 | 2 | 0 | 64 | 147464 | 354,952 |
| decode bf16, after | 248 | 0 | 2 | 0 | 64 | 147464 | 354,952 |
| decode `-DPLOW_FP8_KV=1` | 248 | 0 | 2 | 0 | 67 | 147464 | 382,432 |

The bf16 object is unchanged in every number **including size** — the fp8 body is behind
`if constexpr (FP8)` on a template only instantiated with `FP8=true` under the axis. The fp8-KV
object grows 7.8%; a *dense-GQA* fp8-KV object (Gemma, Qwen) pays that for nothing, since
`exec_flash_mla_decode` is unconditional in the decode bucket. The fix if it ever matters is a
`PLOW_MLA_FP8_KV` sub-axis, deliberately NOT taken: a second KV axis re-opens the pairing problem
§7 just closed.

## 7. A silence this found and closed, worth more than the feature

`PLOW_FP8_KV` is a **SWAP**, so it is silent in BOTH directions, where `PLOW_K3` (additive) is
silent in only one. Measured, not argued — the K3 MLA gate's **bf16** packet against the **fp8**
object:

    all packets executed on every slice: YES
    M2  FLASH_MLA + MERGE_FOLD oat    1.000e+00   1.000e+00
    M9  BLOCK out                     9.447e-02   1.250e-01

A completely untouched `Opart` read as a result, with the packet graph reporting full success —
`GFX950_DISPATCHED`'s failure class reached through a build axis rather than a missing case label.
This pairing had no check before this commit, for the DENSE family either.

Closed the way `PLOW_K3` is: `interp.hip` emits a `plow_fp8_kv_1` marker iff the axis is on, and
`plowrt`'s `check_kv_encoding` refuses a packet whose KV half does not match the object's — in both
directions. The manifest (`fp8_kv -> PLOW_FP8_KV=1`) SELECTS the right object; this REFUSES a wrong
pair that was selected anyway.

## 8. Verdict

fp8 latent KV works, is exact to the e4m3 floor, and does not amplify through the shared latent —
the three things that could have gone wrong for MLA specifically did not. It halves the largest KV
in the fleet: **44.1%, 3.38 → 1.89 GiB for K3 at 128k**.

It also costs **~25x the bf16-KV error at the MLA output** (1.1e-3 → 2.7e-2), irreducibly (§4) and
without decaying with context (§3c). At the *block* output that is 3.6e-3 against bf16's 7.3e-4 —
still under half a percent, which is why vLLM ships `--kv-cache-dtype fp8` for K3 and why this is a
deployment trade rather than a defect.

Routing it is therefore a numerics decision to be taken with these numbers in hand, not defaulted
into every MLA blob. What routing would take is listed in `GFX950_UNEMITTED`'s entry and in
`plans/mla-fp8-kv.md` §5.

## 9. Not measured here

* **Throughput.** No ITL/decode-latency A/B was taken. The fp8 latent halves the decode KV stream,
  which is HBM-bound, so the expectation is the same ~2x roofline the dense fp8 KV gets — but that
  is an expectation, not a number, and nothing here measured it.
* **The prefill twin.** `FlashMlaPrefillFp8` is no longer a six-line wrapper over the decode body and
  is no longer ungated. It runs `d_flash_mla_prefill_mfma`, which tiles the query axis onto the MFMA
  (2.25-2.79x over the wrapper at K3's TP8 shape), and `mla_gfx950_test.c` phase 3 checks it against
  the scalar fp8 body on synthesized e4m3 — max rel err 3.3e-03 over n_head 12/64, ctx 512-2048,
  n_tok 64-512. Phase 2 ties the tiling itself to the validated decode oracle in bf16 (7.2e-05).
  The tiled kernel does NOT dequantize into LDS: e4m3 is exact in bf16, so the staged tile is raw and
  the row scale is applied per kv column after the score MFMA and folded into P for the PV — the same
  association the scalar body uses (`dotc * cs + dotr * rs`), and the reason the two agree at all.
* **Gathered (DSA) MLA with an fp8 latent.** Not implementable through the current ABI: `t7` is the
  `idx` table and `t7` is where the scales live. The gathered arm stays bf16 in both objects.
