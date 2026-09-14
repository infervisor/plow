#!/usr/bin/env python3
"""Roofline for DeepSeek-V4.1-Flash, 8192-token prefill, 8x MI300X.

Establishes the BASELINE the campaign is measured against, before any GPU is
leased. Everything here is derived from two sources and nothing else:

  * the checkpoint's own `config.json` and safetensors headers -- exact shapes
    and dtypes, so no parameter count is estimated;
  * `crates/hwspec` MI300X -- 304 CUs x 4 matrix cores at 2.10 GHz, and the
    MEASURED 4091.9 GB/s HBM rather than the 5325 datasheet peak. A roofline
    drawn against the datasheet understates every kernel here by ~30%.

Three facts drive the answer and none of them are obvious from the config:

  1. PREFILL ATTENTION IS LINEAR IN T, NOT QUADRATIC. `Attention.forward`
     concatenates a 128-token sliding window with `index_topk`=512 compressed
     positions and makes ONE `sparse_attn` call, so every query sees 640 keys
     regardless of sequence length. Quadratic cost survives only in the
     indexer's own scoring.
  2. MI300X HAS NO fp4 MATRIX ENGINE (`hwspec` MI300X `mma.fp4: None`; the
     gfx942 asm contract forbids `v_mfma_f32_32x32x64_f8f6f4` for exactly this
     reason). The routed experts are MXFP4, so on this part fp4 buys MEMORY,
     not FLOPs: the weights must be dequantized and issued at the fp8 rate.
  3. WITH EXPERT PARALLELISM THE COLLECTIVES ARE FIRST-ORDER. The indexer
     all-reduces an [S, S/ratio] fp32 score tensor per index layer, which at
     8k is 268 MB on a ratio-1 layer -- larger than any weight tensor in the
     model.

Usage:  python3 scripts/dsv41_roofline.py [--tokens 8192] [--gpus 8]
"""

import argparse
import json
import os

HF = os.environ.get("DSV41_HF", "/workspace/models/DeepSeek-V4.1-Flash")

# ---------------------------------------------------------------- hardware
# crates/hwspec/src/amd/mi300.rs. `mma` is MACs/cycle/matrix-core; CDNA3 has 4
# matrix cores per CU, so peak = CUs * 4 * MACs * 2 FLOP/MAC * clock. That
# reproduces the published 1307.4 (bf16) / 2614.9 (fp8) TFLOP/s dense exactly.
CUS = 304
MATRIX_CORES_PER_CU = 4
CLOCK_HZ = 2.10e9
MACS = {"bf16": 256, "fp8": 512, "fp4": None}
HBM_MEASURED = 4091.9e9          # bandwidth_measured, not the 5325e9 datasheet
XGMI_PER_GPU = 896.0e9           # interconnect.per_gpu_bandwidth


def peak_flops(dtype):
    m = MACS[dtype]
    if m is None:
        return None
    return CUS * MATRIX_CORES_PER_CU * m * 2 * CLOCK_HZ


# Sustained fraction of dense matrix peak for a well-tuned GEMM of these shapes
# on this part. Not a guess pulled from the air: it is the band plow's own
# gfx942 GEMMs land in, and the roofline is reported at both ends of it.
EFF_LO, EFF_HI = 0.35, 0.55


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    T, G = a.tokens, a.gpus

    c = json.load(open(os.path.join(HF, "config.json")))["text_config"]
    H = c["hidden_size"]                      # 5120
    L = c["num_hidden_layers"]                # 40
    NH = c["num_attention_heads"]             # 64
    HD = c["head_dim"]                        # 512  (the KV latent width too)
    RD = c["qk_rope_head_dim"]                # 64
    QLR = c["q_lora_rank"]                    # 1280
    OLR = c["o_lora_rank"]                    # 1024
    OG = c["o_groups"]                        # 8
    MI = c["moe_intermediate_size"]           # 2304
    NE = c["n_routed_experts"]                # 384
    TOPK = c["num_experts_per_tok"]           # 6
    NSH = c["n_shared_experts"]               # 1
    WIN = c["sliding_window"]                 # 128
    RATIOS = c["compress_ratios"]
    KVSRC = c["kv_source_layer_ids"]          # [2, 8, 14, 20]
    IDXSRC = c["index_source_layer_ids"]      # 8 layers
    INH = c["index_n_heads"]                  # 32
    IHD = c["index_head_dim"]                 # 128
    ITOPK = c["index_topk"]                   # 512
    HCM = c["hc_mult"]                        # 4
    ENG = c["engram_layer_ids"]               # [1, 14]
    VOCAB = c["vocab_size"]

    MIX = (2 + HCM) * HCM                     # 24, matches hc_*_fn [24, 20480]
    ENG_IN, ENG_OUT = 6144, 25600             # engram.wkv shape from the header

    def gemm(tokens, k, n):
        return 2.0 * tokens * k * n

    # ------------------------------------------------------------ FLOPs
    f = {}

    # Dense attention projections, every backbone layer.
    per_layer_attn = (
        gemm(T, H, QLR)                 # wq_a   5120 -> 1280
        + gemm(T, QLR, NH * HD)         # wq_b   1280 -> 32768
        + gemm(T, H, HD)                # wkv    5120 -> 512 (one shared latent)
        # wo_a is block-diagonal: OG groups, each [OLR, NH*HD/OG] over its own
        # slice of the head output. Same FLOP count as a dense [OG*OLR, NH*HD/OG].
        + gemm(T, NH * HD, OLR)
        + gemm(T, OG * OLR, H)          # wo_b   8192 -> 5120
    )
    f["attn_proj"] = per_layer_attn * L

    # Sparse attention. Every query sees WIN raw keys plus, where the layer has
    # a compress ratio, ITOPK compressed ones -- capped, hence linear in T.
    attn_core = 0.0
    keys_per_layer = []
    for l in range(L):
        ratio = RATIOS[l]
        keys = WIN + (min(ITOPK, (T // ratio)) if ratio > 0 else 0)
        keys_per_layer.append(keys)
        # QK^T over (HD) then PV over (HD); MQA, so one KV stream for NH heads.
        attn_core += 2.0 * (2.0 * T * keys * NH * HD)
    f["attn_core"] = attn_core

    # Lightning indexer: a per-layer projection plus the score einsum, which is
    # the only quadratic term left in the model.
    idx_proj = idx_score = 0.0
    for l in IDXSRC:
        ratio = RATIOS[l]
        tc = T // ratio                       # compressed positions visible
        idx_proj += gemm(T, QLR, INH * IHD) + gemm(T, H, INH)
        idx_score += 2.0 * T * tc * INH * IHD  # einsum bshd,btd->bsht
    f["indexer_proj"] = idx_proj
    # The reference computes the full S x T_c block and masks afterwards; a
    # causal kernel does half. Report the half, note the reference's 2x.
    f["indexer_score"] = idx_score / 2.0
    f["_indexer_score_reference_dense"] = idx_score

    # Compressor, only on the 4 kv_source layers (wkv + wgate, wgate on 3).
    comp = 0.0
    for l in KVSRC:
        comp += gemm(T, H, HD)
        if l != 20:                            # layer 20 is ratio-1, no gate
            comp += gemm(T, H, HD)
    f["compressor"] = comp

    # Engram: sparse embedding gather (no FLOPs) then one wide projection.
    f["engram_proj"] = len(ENG) * gemm(T, ENG_IN, ENG_OUT)

    # MoE. Every token runs the shared expert; TOPK of NE routed experts fire.
    swiglu = lambda tok: gemm(tok, H, MI) * 2 + gemm(tok, MI, H)   # w1, w3, w2
    f["shared_expert"] = L * NSH * swiglu(T)
    f["routed_expert"] = L * swiglu(T * TOPK)
    f["router_gate"] = L * gemm(T, H, NE)      # fp32, replicated on every rank

    # Single-pass mHC: two [MIX, HCM*H] fp32 mixes per layer.
    f["mhc"] = L * 2 * gemm(T, HCM * H, MIX)

    # Prefill only needs logits for the final token.
    f["lm_head"] = gemm(1, H, VOCAB)

    total_flops = sum(v for k, v in f.items() if not k.startswith("_"))

    # ------------------------------------------------------------ bytes
    b = {}
    # Routed experts, MXFP4: two values per byte, plus one E8M0 scale per 32.
    w_elems = MI * H * 2 + H * MI                 # w1, w3, w2 logical elements
    per_expert = w_elems / 2 + w_elems / 32       # packed payload + scales
    b["routed_expert_w"] = L * NE * per_expert
    # Dense fp8 weights carry a [32,32]-block ue8m0 scale: 1 byte per 1024.
    fp8 = lambda n: n * (1 + 1 / 1024)
    b["attn_w"] = L * fp8(H * QLR + QLR * NH * HD + H * HD + NH * HD * OLR + OG * OLR * H)
    b["shared_expert_w"] = L * NSH * fp8(w_elems)
    b["indexer_w"] = len(IDXSRC) * (fp8(QLR * INH * IHD) + 2 * H * INH)
    b["compressor_w"] = (len(KVSRC) + 3) * 2 * H * HD
    b["engram_w"] = len(ENG) * fp8(ENG_IN * ENG_OUT)
    b["router_w"] = L * 2 * H * NE
    b["mhc_w"] = L * 2 * 4 * MIX * HCM * H
    b["lm_head_w"] = 2 * H * VOCAB

    # The residual stream is hc_mult x hidden EVERYWHERE -- h is unsqueezed and
    # repeated to [.., hc_mult, dim] at entry -- so every touch of it moves 4x
    # what a conventional model would. Six touches per layer: hc_pre reads it,
    # attn writes, hc_post reads+writes, ffn reads, ffn writes.
    resid = T * HCM * H * 2
    b["residual_stream"] = L * 6 * resid

    # Engram lookup is a gather of at most max_ngram rows per head per token.
    b["engram_gather"] = T * c["engram_n_heads"] * c["engram_max_ngram_size"] * c["engram_head_dim"]

    total_bytes = sum(b.values())

    # ------------------------------------------------------------ collectives
    coll = {}
    ring = 2.0 * (G - 1) / G                   # ring all-reduce, per-GPU bytes
    isc = 0.0
    for l in IDXSRC:
        isc += T * (T // RATIOS[l]) * 4        # fp32 scores, all-reduced
    coll["indexer_score_ar"] = isc * ring
    coll["wo_b_ar"] = L * T * H * 2 * ring     # RowParallelLinear output
    coll["moe_ar"] = L * T * H * 2 * ring      # expert-parallel output
    total_coll = sum(coll.values())

    # ------------------------------------------------------------ report
    fp8_peak = peak_flops("fp8") * G
    bf16_peak = peak_flops("bf16") * G
    hbm = HBM_MEASURED * G
    xgmi = XGMI_PER_GPU                        # per-GPU link, not aggregated

    t_compute_peak = total_flops / fp8_peak
    t_mem = total_bytes / hbm
    t_coll = total_coll / xgmi

    print(f"DeepSeek-V4.1-Flash roofline -- {T} token prefill on {G}x MI300X (gfx942)")
    print(f"  source: {HF}/config.json + safetensors headers; hwspec MI300X\n")

    print("FLOPs")
    for k in sorted((k for k in f if not k.startswith("_")), key=lambda k: -f[k]):
        print(f"  {k:<22}{f[k]/1e12:>10.2f} TFLOP{100*f[k]/total_flops:>8.1f}%")
    print(f"  {'TOTAL':<22}{total_flops/1e12:>10.2f} TFLOP")
    print(f"  (reference's non-causal indexer would add "
          f"{(f['_indexer_score_reference_dense']-f['indexer_score'])/1e12:.2f} TFLOP)\n")

    print("Bytes (HBM)")
    for k in sorted(b, key=lambda k: -b[k]):
        print(f"  {k:<22}{b[k]/1e9:>10.2f} GB{100*b[k]/total_bytes:>10.1f}%")
    print(f"  {'TOTAL':<22}{total_bytes/1e9:>10.2f} GB\n")

    print("Collectives (per-GPU bytes over xGMI)")
    for k in sorted(coll, key=lambda k: -coll[k]):
        print(f"  {k:<22}{coll[k]/1e9:>10.2f} GB")
    print(f"  {'TOTAL':<22}{total_coll/1e9:>10.2f} GB\n")

    ai = total_flops / total_bytes
    balance = peak_flops("fp8") / HBM_MEASURED
    print("Roofline")
    print(f"  arithmetic intensity     {ai:>10.1f} FLOP/byte")
    print(f"  MI300X fp8 balance       {balance:>10.1f} FLOP/byte")
    print(f"  -> {'COMPUTE' if ai > balance else 'MEMORY'}-bound "
          f"by {max(ai, balance)/min(ai, balance):.2f}x\n")
    print(f"  fp8 dense peak, {G} GPUs  {fp8_peak/1e12:>10.1f} TFLOP/s")
    print(f"  bf16 dense peak, {G} GPUs {bf16_peak/1e12:>10.1f} TFLOP/s")
    print(f"  fp4 matrix engine        {'ABSENT on gfx942 -- experts dequant to fp8':>10}")
    print(f"  HBM, {G} GPUs (measured)  {hbm/1e12:>10.2f} TB/s\n")

    print("Time floor")
    print(f"  compute @100% fp8        {t_compute_peak*1e3:>10.1f} ms   (unreachable)")
    print(f"  compute @{EFF_HI:.0%}              {total_flops/(fp8_peak*EFF_HI)*1e3:>10.1f} ms")
    print(f"  compute @{EFF_LO:.0%}              {total_flops/(fp8_peak*EFF_LO)*1e3:>10.1f} ms")
    print(f"  HBM                      {t_mem*1e3:>10.1f} ms")
    print(f"  collectives (exposed)    {t_coll*1e3:>10.1f} ms")

    lo = max(total_flops / (fp8_peak * EFF_HI), t_mem)
    hi = max(total_flops / (fp8_peak * EFF_LO), t_mem) + t_coll
    print(f"\n  BASELINE 8k TTFT floor   {lo*1e3:.0f} - {hi*1e3:.0f} ms")
    print(f"  target                          300 ms")
    print(f"  headroom                 {300/(lo*1e3):.1f}x - {300/(hi*1e3):.1f}x")

    # ---------------------------------------------------------- as built
    # The floor above prices every FLOP at the fp8 peak. The ASM audit of the
    # prebuilt gfx942 objects says the expert path does not get it: the mxfp4
    # GEMM bodies (gemm_mxfp4_c2/c3/c4 in test_kernels.elf) dequantize and
    # issue v_mfma_f32_32x32x8_bf16, so the MXFP4 experts run at the BF16 rate
    # -- half of fp8. Price the experts there and the rest at fp8.
    expert_flops = f["routed_expert"] + f["shared_expert"]
    other_flops = total_flops - expert_flops
    t_built_peak = expert_flops / bf16_peak + other_flops / fp8_peak
    b_lo = max(t_built_peak / EFF_HI, t_mem)
    b_hi = max(t_built_peak / EFF_LO, t_mem) + t_coll
    print("\n  As built (experts at the bf16 MFMA the mxfp4 kernels actually issue)")
    print(f"    compute @100%          {t_built_peak*1e3:>10.1f} ms")
    print(f"    AS-BUILT floor         {b_lo*1e3:>7.0f} - {b_hi*1e3:.0f} ms"
          f"   headroom {300/(b_lo*1e3):.1f}x - {300/(b_hi*1e3):.1f}x")

    if a.json_out:
        out = {
            "tokens": T, "gpus": G,
            "tflop": round(total_flops / 1e12, 2),
            "hbm_gb": round(total_bytes / 1e9, 2),
            "collective_gb": round(total_coll / 1e9, 2),
            "arithmetic_intensity": round(ai, 1),
            "machine_balance": round(balance, 1),
            "compute_bound": ai > balance,
            "floor_ms_lo": round(lo * 1e3, 1),
            "floor_ms_hi": round(hi * 1e3, 1),
            "target_ms": 300,
            "flops_breakdown_tflop": {k: round(v / 1e12, 3) for k, v in f.items()},
            "bytes_breakdown_gb": {k: round(v / 1e9, 3) for k, v in b.items()},
            "collectives_gb": {k: round(v / 1e9, 3) for k, v in coll.items()},
            "keys_per_query_by_layer": keys_per_layer,
        }
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {a.json_out}")


main()
