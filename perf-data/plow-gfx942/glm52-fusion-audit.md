# GLM-5.2 prefill block: fusion-completeness + rearrangement audit (gfx942 TP8)

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL** — a seam-by-seam completeness audit of the emitted chain. The verdicts turn on op availability and data layout, not on silicon.

2026-08-08, branch `fusion-audit` (base aa26a6c). Priced against the fresh
single-block trace @T=8192 (hsaco_glm13 + PLOW_MLA_PF_V2=1, 29.4 ms/layer);
spans in the table below are that trace's.

## Seam-by-seam verdict table

| # | seam (spans µs) | verdict | why |
|---|---|---|---|
| 1 | MlaMergeFold 1574 (+540 gap) → o_proj 672 | **FUSABLE — designed, unlanded (the largest single win left, ~−0.9 ms/layer)** | At prefill nsplit==1 the merge is exactly `oat = normalize(opart) × W_uv`, so the pair collapses to ONE GEMM `og_tp = opart × W_ofold` with a PREP-DERIVED weight `W_ofold[n, h*512+d] = Σ_v Wuv[h,d,v]·Wo[n, h*256+v]` ([6144, nh_l*4096/8... per-rank K = nh_l·512]). Needs: (a) v2 flash epilogue writes NORMALIZED bf16 rows (each wave holds its rows' `l`; `i[6]` free in the dense arm), (b) additive prep shard (`derived.o_fold`, computable from the EXISTING lite checkpoint's bf16 v_absorb + o_proj — no re-prep), (c) `shard_of` ROW entry, (d) the PART16-class marker/`requires` refusal chain (an ofold blob on a pre-arm object is silently wrong). 1.84× more o-GEMM FLOPs (412 vs 224 GFLOP) but deletes the fold packet, the 134 MB oat round trip, and the mlpart write; at o_proj's measured ~300 TF/s the fused GEMM ≈ 1.4 ms vs 2.25 ms split. Numerics: better, not identical (f32 carried through K=4096 instead of a bf16 hop) → logit-gate class. NOT landed here: the refusal chain + prep + kernel epilogue exceeded this audit's budget alongside seam 4; full spec above is implementation-ready. |
| 2 | RmsNorm 137/136 → head GEMMs | **NO** | GemmNorm (op 9) has NO AMD arm (op-coverage: never dispatched on AMD); post-widening the norms are 137/136 µs — a new MFMA-object arm plus per-GEMM re-normalization (3 consumers re-deriving the same norm) cannot pay. |
| 3 | router GemmSmall 185 → RouterTopkPf 327 → AlignPf 241 | **NO (loses)** | The [T,256] logit round trip is 8 MB (~4 µs at HBM); the chain already hides partially under the shared-expert GEMMs (measured −283 µs gap). A fused score+topk saves ≤10 µs and welds a 304-wide GEMM to a block-per-token tail with mismatched shapes. Align must stay b=1 (global scan). |
| 4 | XR 1054 → Residual 125 (both TP seams + dense) | **LANDED — `PLOW_GLM_XR_RES=1`** (this commit) | The two-shot's all-gather writes `out2 = bf16(resid + reduced)` directly (t1/t2 on XREDUCE2, dispatch-mapped; every pre-fold blob carries TENSOR_NONE). Deletes the Residual packet AND the AG's own `out` write per seam: −2 packets, ~−600 MB/layer traffic. BIT-IDENTICAL (the gathered value is already bf16; add order is d_residual's; verified logits byte-equal on GPU). **Trap found and fixed**: the first cut interleaved a dependent local load+add+store per element and served 4.7× SLOWER — the uncached peer stream lives on independent requests in flight; the landed loop hand-batches 8 remote loads ahead (see the kernel comment). |
| 5 | HeadNormRope 63/66 → flash | **NO (price only)** | 129 µs combined, already overlapped to ~1 µs gaps; the Gemma hnr→flash fold does not transfer to the v2 flash (Q rides pre-formed A-fragments; a rope epilogue inside the flash would re-derive per-position angles per wave). |
| 6 | rearrangement | **NO further wins found** | Measured overlaps already present (ckv/krr under q-chain −356/−256 µs, shared expert under router −283 µs, q_rope under q_absorb −164 µs). The chain is machine-full: apparent "late starts" (kv_ln +631 µs after its dep) are CU saturation, not dependency slack. The one real hole — MlaMergeFold's 540 µs flash-straggler gap — is absorbed by seam 1's fused GEMM, not by reordering. |
| 7 | XR → RmsNorm (post-Residual norm) | **NO** | A fused rmsnorm needs row-complete data; the AG writes elements slice-major with no row locality on any single workgroup. Extending the fold past the residual would need a second pass = exactly the packet it would delete. (Decode's op 116 XRN is one-shot/row-local — different shape; its decode null stands.) |

## Landed: XR+Residual fold (seam 4)

Emit: `PLOW_GLM_XR_RES=1` (default off; unset = byte-identical blob, verified by
cmp against the shipped V2 blob). Applied at the attn seam, the MoE combine
seam, and the dense-FFN seam; composes with kb==1 banding only (banded seams
keep the split Residual). Kernel: batched fused AG in `d_xreduce_twoshot_mega`
(op_collective.h); dispatch maps t1/t2 with the band offset e0. 156 Residual
packets/program → 0.

Numbers (hsaco_fuse2 objects, both arms PLOW_MLA_PF_V2=1):

- **Prefill logits BYTE-IDENTICAL** to the split emit (fixed 2048-token prompt,
  cmp on dumped logits; amd-bench walls 542.5 ctl vs 532.4 fold).
- Served, clean interleaved round (gates PASS both arms, TPOT unchanged
  32.0/32.2): ctl 1075.9 / 2000.0 → fold **1068.6 / 1966.0 @4k/8k**
  (−0.7% / −1.7%).
- Round 1's fold serve died on VRAM co-tenancy (a co-resident workload's model was
  resident; `hsa_amd_memory_pool_allocate(1.2 GB): 4104`) — infrastructure, not
  the fold; round 2 ran clean. Re-run a second clean round at consolidation.
- Perf trap recorded in the kernel: the naive fused loop (dependent local
  load+add+store per element) served 4.7× SLOWER — the uncached peer stream
  needs its independent-request depth preserved; the landed loop batches 8
  remote loads ahead. This is a reusable rule for ANY future epilogue on the
  peer path.

## PLOW_GLM_XR_RES — ADOPTED into the canonical recipe (2026-08-08)

The fold was landed long ago, measured −0.7%/−1.7% TTFT with **byte-identical prefill logits**,
and then simply never added to the emit recipe — found by the history-leverage review, which
flagged it as a free item sitting unused.

Re-gated on the current config (objects `hsaco_glm18` both arms; emit-side only, so the blob
differs and the objects are identical). 3 interleaved rounds, port 8196:

| ctx | final2 | +XR_RES | Δ | control's own spread |
|---|---|---|---|---|
| 1024 | 343.5 | **331.4** | **−3.54%** | 0.15% → **23×** |
| 4096 | 972.1 | **964.6** | −0.77% | 0.37% → 2× |
| 8192 | 1676.3 | 1672.3 | −0.24% | 0.42% → inside noise |
| 16384 | 3617.9 | 3634.2 | +0.45% | 0.35% → inside noise |

Quality: **5/5 character-identical**, as a byte-identical fold must be.

**Read:** the win is concentrated at 1k and decays to nothing by 8k. That is the expected shape —
the fold removes a per-collective-seam packet, and the seam count is fixed per layer while the
work per seam grows with T, so the saving is a fixed cost that matters most where fixed costs
dominate. 1k is exactly the regime where plow is furthest behind vLLM (4.97×) and where the
decomposition found 164 ms of the 343 ms is per-chunk intercept.

**Adopted**: `PLOW_GLM_XR_RES=1` joins the canonical blob recipe. It is free at worst (the 16k
+0.45% is half the control spread and the 8k −0.24% likewise) and worth 12 ms where we most need
it. No object change, no arm requirement.
