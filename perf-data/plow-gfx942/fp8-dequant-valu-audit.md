# gfx942 fp8-dequant VALU audit (staging paths)

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC** — counts VALU for the OCP-e4m3 -> bf16 software chain CDNA3 needs. CDNA4 has `cvt_scalef32_pk_bf16_fp8` and does not run this chain at all.

2026-08-07. ASM probe (`clang++ -x hip -S --offload-arch=gfx942 -O3`, isolated
kernels around the exact shipping sequence), counting VALU per 4 OCP-e4m3 bytes
decoded to packed bf16 — the chain every fp8 staging path runs on CDNA3
(`plow_fp8x4_ocp_to_bf16` in runtime/amd/amd_arch.h, reached from
`fp8_to_bf16v8` / `fp8v8_to_bf16v8` / `mpf_ld_w8`):

| variant | VALU /4B | notes |
|---|---|---|
| shipping (`mask + cvt + ×2 + shift-pack`) | **21** | compiler already folds the four ×2 muls into two `v_pk_add_f32`; pack compiles to lshr+and+or_sdwa — LLVM does **not** synthesize `v_perm` |
| fold OCP ×2 into the block scale | 16 | delete the pk_adds; caller multiplies its f32 scale by 2 once |
| + `v_perm_b32` pack (`__builtin_amdgcn_perm(b, a, 0x03020706)`) | **14 (−33%)** | mask kept — safe for raw checkpoint weight bytes; **bit-identical** (halved e4m3 exact in bf16; ×2 distributes exactly over f32 sums; 2·bs exact) |
| + drop the neg-0 SWAR mask (source guaranteed 0x80-free) | **6 (−71%)** | 2 cvt + 2 perm; needs a scrubbed source (below) |

Bit-identity of the ×2 fold: every e4m3 value has ≤3 mantissa bits, so /2 is an
exponent shift, exact in bf16; MFMA computes sum(wᵢ/2·aᵢ) = sum(wᵢ·aᵢ)/2
exactly (power-of-two scaling commutes with f32 addition); acc×(2·bs) restores
it and 2·bs is exact. NaN weight bytes 0x7f/0xff decode to 240→480 after the
fold — the same documented divergence as today.

## Where each lands

- **MoE-PF grouped kernel (ops 85/86)** — hottest consumer (64 fp8 B-bytes per
  thread per k-tile → 336 VALU/thread/k-tile shipping, ~224 with the 14-VALU
  variant). Handed to the MoE-PF rate agent as a lever recipe (halved decode in
  MPF_FETCH_B + `bs*2` at every promotion site); its logit byte-compare gate is
  the arbiter. Composes with wider 16-byte B loads.
- **fp8-KV flash (opt-in `PLOW_GLM_FP8_KV`, branch glm52-fp8kv)** — the arm
  measured +0.4..+3.1% TTFT partly because of in-loop dequant VALU on a
  compute-bound body. The 6-VALU variant applies here **without any loader
  work**: the KV bytes are self-produced by `plow_f32_to_fp8_ocp`, which today
  emits 0x80 for negative values whose magnitude rounds to 0 — a one-line
  value-identical fix (emit 0x00 when mag==0) makes the cache 0x80-free by
  construction. Queued for after the agent branches merge.
- **Decode fp8 GEMVs** — already audited: the raw-f32 rewire was tried and
  REVERTED (+1% — latency-bound at 2 waves/SIMD hides the VALU; see the
  `plow_fp8x16_to_f32` header). The ×2-fold/perm-pack variants were NOT part of
  that experiment, but the same latency-bound verdict likely applies; only
  worth retrying if a staging-path win materializes first.
- **Weight scrub to reach 6 VALU for weights**: a loader-side pass rewriting
  0x80→0x00 in fp8 weight buffers at bind (value-identical, −0 ≡ +0 in every
  product). Not landed; contract change (kernels must know bytes are scrubbed).

Probe source: /root/.claude/jobs/b09a4bcc/tmp/fp8probe/probe.hip.

## Landed (hsaco_glm8 objects + scrub-aware plowrt), measured

The 6-VALU maskless staging decode is IN for ops 85/86, backed by a loader-side
0x80→0x00 scrub of **every F8_E4M3 checkpoint payload** at stage time
(`scrub_fp8_neg0` SWAR inside the pinned-slab copy; dtype from the safetensors
header; both the routed-expert packing loop and the generic named-weight path).
The encoder `plow_f32_to_fp8_ocp` now never emits −0, so CDNA3 device-produced
fp8 streams are 0x80-free by construction too.

**Lesson bought with one failed run:** scrubbing only the routed-expert path
was NOT enough — GLM's three dense layers run the same grouped ops through
`dense_weight_table`, whose projections are ordinary declared tensors on the
generic upload path. First attempt → FNUZ NaN in layers 0–2, token 0, prefill
2.2× slower. The dtype-driven rule (scrub every e4m3 payload wherever it
stages) is the correct contract width; GLM expert shards really do carry 0x80
bytes (~3 ppm, measured).

Verification: prefill logits BYTE-IDENTICAL to the masked hsaco_glm7 arm on a
fixed 2048-token prompt; 5/5 serve coherence gates PASS (GLM ×4, Gemma ×1);
Gemma 11.02 ms @4k unaffected. Interleaved 2-round TTFT (median):
358.4/1385.5/2969.8 → **355.2/1374.4/2950.3 @1k/4k/8k** (−0.7..−0.9%); TPOT
wash. Pair spans @2k: GluPf 973 µs (was 1003), DownPf 1141 (was ~1120-1176,
run-to-run noise) — consistent with a weight-stream-bound kernel where VALU was
a minor term. Canonical objects: **hsaco_glm8**; pfwide / wg152-l2 /
g12b-showdown repointed. Unit test: `scrubs_exactly_neg_zero` (hsa.rs).
