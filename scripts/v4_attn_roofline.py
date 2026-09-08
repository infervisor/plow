#!/usr/bin/env python3
"""Attention roofline for DeepSeek-V4-Flash on MI300X, per rank, at TP2/TP4/TP8.

Extends the GLM-5.3 / Gemma-4 / Kimi-K3 calculator (attn_roofline.py, same ceilings and the
same Layer/Geo shape) with V4-Flash's three attention regimes and its lightning indexer, and
re-emits the GLM/K3 rows so the comparison table comes out of one program.

Ceilings: docs/amd/glm53-roofline-and-trace.md (measured read 4112 GB/s -> 4100 used).
Geometry: /workspace/models/DeepSeek-V4-Flash-0731/{config.json,inference/model.py} and the
checkpoint's own safetensors headers (scripts/_v4_dump.py).
"""
import math

# ---------------------------------------------------------------- ceilings (measured, MI300X)
BW = 4100e9            # HBM read, measured 8 GiB reduction (4112 rounded); spec 5325 unused
F = 1307e12            # bf16 dense MFMA peak
RIDGE = F / BW         # 318.8 FLOP/B
VALU = 304 * 64 * 2.1e9  # 40.9 T fp32 lane-ops/s
IF_BW = 896e9          # Infinity Fabric per GPU (all-reduce path)
L2_XCD = 4 << 20       # 4 MiB per XCD

CTX = [4096, 32768, 131072, 262144, 1048576]

# ---------------------------------------------------------------- V4 geometry
DIM = 4096
HEADS = 64
HD = 512               # head_dim; last 64 dims are the rope strip
RD = 64
WIN = 128              # sliding_window
IDX_HEADS = 64
IDX_D = 128
IDX_TOPK = 512
Q_LORA = 1024
O_LORA = 1024
O_GROUPS = 8
N_EXPERT = 256
N_ACT = 6
MOE_INTER = 2048

# compress_ratios[0:43] of config.json -> layer kinds
RATIOS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 43)]
assert len(RATIOS) == 43
L_W = RATIOS.count(0)      # 2   pure sliding-window
L_C4 = RATIOS.count(4)     # 21  window + indexer top-512 over the ratio-4 compressed stream
L_C128 = RATIOS.count(128)  # 20  window + the whole ratio-128 compressed stream
assert (L_W, L_C4, L_C128) == (2, 21, 20), (L_W, L_C4, L_C128)

# KV row bytes per position per layer. K == V (one 512-wide row serves both), replicated on
# every rank because Attention.wkv / Compressor.wkv are plain Linear, not ColumnParallel.
ROW_BF16 = HD * 2                                  # 1024
ROW_FP8 = (HD - RD) * 1 + (HD - RD) // 64 + RD * 2  # 583: 448 e4m3 + 7 ue8m0 + 64 bf16 rope
IROW_BF16 = IDX_D * 2                              # 256
IROW_FP4 = IDX_D // 2 + IDX_D // 32                # 68: fp4 + ue8m0 per 32


def rows_W(T):
    return min(T, WIN)


def rows_C4(T):
    return min(T, WIN) + min(IDX_TOPK, T // 4)


def rows_C128(T):
    return min(T, WIN) + T // 128


def rows_idx(T):
    return T // 4


KINDS = [("W", L_W, rows_W), ("C4", L_C4, rows_C4), ("C128", L_C128, rows_C128)]


# -------- prefill pair counts (sum over query positions of the rows that query attends to)
def pairs_W(T):
    if T <= WIN:
        return T * (T + 1) // 2
    return WIN * (WIN + 1) // 2 + (T - WIN) * WIN


def _sum_floor_div(T, r):
    """Sum_{s=1..T} floor(s/r) = r*q*(q-1)/2 + (rem+1)*q  with q=T//r, rem=T%r."""
    q, rem = T // r, T % r
    return r * q * (q - 1) // 2 + (rem + 1) * q


def _sum_min_floor_div(T, r, cap):
    """Sum_{s=1..T} min(cap, floor(s/r)). floor(s/r) >= cap first at s = cap*r."""
    s0 = cap * r
    if T < s0:
        return _sum_floor_div(T, r)
    return _sum_floor_div(s0 - 1, r) + (T - s0 + 1) * cap


def pairs_C4(T):
    return pairs_W(T) + _sum_min_floor_div(T, 4, IDX_TOPK)


def pairs_C128(T):
    return pairs_W(T) + _sum_floor_div(T, 128)


def pairs_idx(T):
    return _sum_floor_div(T, 4)


PAIRS = {"W": pairs_W, "C4": pairs_C4, "C128": pairs_C128}


# ---------------------------------------------------------------- weight stream, per rank
# From the checkpoint headers. Sharding follows the reference module classes:
#   ColumnParallelLinear / RowParallelLinear / MoE routed experts -> /P
#   plain Linear (wq_a, wkv, compressor, shared expert), gate, hyper-connection fns -> replicated
def _b(shape, dt):
    n = 1
    for s in shape:
        n *= s
    return n * {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}[dt]


EXPERT_B = (_b([2048, 2048], "I8") + _b([2048, 128], "F8_E8M0")) * 2 + \
           _b([4096, 1024], "I8") + _b([4096, 64], "F8_E8M0")          # 14.02 MB fp4+scales
SHARED_B = _b([2048, 4096], "F8_E4M3") * 2 + _b([4096, 2048], "F8_E4M3") + 1536  # 25.17 MB

# replicated per rank, per token, all 43 layers
REP = 43 * (_b([1024, 4096], "F8_E4M3") + 256                 # wq_a
            + _b([512, 4096], "F8_E4M3") + 128                # wkv
            + _b([24, 16384], "F32") * 2                      # hc_attn_fn + hc_ffn_fn
            + _b([256, 4096], "BF16"))                        # ffn.gate
REP += L_C4 * _b([1024, 4096], "BF16") * 2                    # ratio-4 compressor wkv+wgate
REP += L_C128 * _b([512, 4096], "BF16") * 2                   # ratio-128 compressor
REP += L_C4 * _b([256, 4096], "BF16") * 2                     # indexer compressor

# sharded (divide by P), per token
SHARD = 43 * (_b([32768, 1024], "F8_E4M3") + 2048             # wq_b
              + _b([8192, 4096], "F8_E4M3") + 2048            # wo_a
              + _b([4096, 8192], "F8_E4M3") + 2048)           # wo_b
SHARD += L_C4 * (_b([8192, 1024], "F8_E4M3") + 512 + _b([64, 4096], "BF16"))  # indexer wq_b, w_proj
SHARD += 43 * N_ACT * EXPERT_B                                # 6 of 256 routed experts
SHARD_SHARED = 43 * SHARED_B                                  # shared expert, if plow shards it


def weight_stream(P, shared_sharded=True):
    w = REP + SHARD / P
    w += SHARD_SHARED / P if shared_sharded else SHARD_SHARED
    return w


def active_flops_per_token(P):
    """2*params touched per token on this rank (dense-GEMM FLOP count)."""
    # count params, not bytes: fp8=1 B/param, fp4=0.5 B/param, bf16=2, f32=4 (hc is elementwise)
    p_rep = 43 * (1024 * 4096 + 512 * 4096 + 256 * 4096)
    p_rep += L_C4 * 1024 * 4096 * 2 + L_C128 * 512 * 4096 * 2 + L_C4 * 256 * 4096 * 2
    p_sh = 43 * (32768 * 1024 + 8192 * 4096 + 4096 * 8192)
    p_sh += L_C4 * (8192 * 1024 + 64 * 4096)
    p_sh += 43 * (N_ACT + 1) * 3 * DIM * MOE_INTER
    return 2 * (p_rep + p_sh / P)


def chunk_weight_bytes(P, C, shared_sharded=True):
    """Bytes of weights a prefill chunk of C tokens streams on this rank: all 256/P routed
    experts are hit once C is a few hundred tokens, everything else once."""
    all_experts = 43 * (N_EXPERT / P) * EXPERT_B
    non_expert = weight_stream(P, shared_sharded) - 43 * N_ACT * EXPERT_B / P
    return non_expert + all_experts


def lin_pf_us(P, C, shared_sharded=True):
    b = chunk_weight_bytes(P, C, shared_sharded)
    f = active_flops_per_token(P) * C
    return max(b / BW, f / F) / C * 1e6


# ---------------------------------------------------------------- decode
def dec_attn(T, P, fp8=False, idx_fp4=False, idx_sharded=True):
    row = ROW_FP8 if fp8 else ROW_BF16
    irow = IROW_FP4 if idx_fp4 else IROW_BF16
    h = HEADS / P
    hi = IDX_HEADS / P if idx_sharded else IDX_HEADS
    per = {}
    b_tot = f_tot = 0.0
    for name, L, rf in KINDS:
        b = L * rf(T) * row
        f = L * 2 * h * rf(T) * (HD + HD)
        per[name] = (b, f)
        b_tot += b
        f_tot += f
    bi = L_C4 * rows_idx(T) * irow + L_C4 * rows_idx(T) * 8   # K matrix + score write/read
    fi = L_C4 * 2 * hi * IDX_D * rows_idx(T)
    per["IDX"] = (bi, fi)
    per["TOTAL"] = (b_tot + bi, f_tot + fi)
    return per


def dec_write(T):
    """cache writes amortized per decode token (window ring + compressed rows)."""
    w = 43 * ROW_BF16                                   # every layer writes its window row
    w += L_C4 * ROW_BF16 / 4 + L_C128 * ROW_BF16 / 128  # compressed rows
    w += L_C4 * IROW_BF16 / 4                           # indexer compressed rows
    return w


# ---------------------------------------------------------------- prefill
def pf_attn(T, P, C, fp8=False, idx_fp4=False, idx_sharded=True):
    row = ROW_FP8 if fp8 else ROW_BF16
    irow = IROW_FP4 if idx_fp4 else IROW_BF16
    h = HEADS / P
    hi = IDX_HEADS / P if idx_sharded else IDX_HEADS
    per = {}
    q_o = h * HD * 2 * 2                                 # Q read + O write per token per layer
    b_tot = f_tot = 0.0
    for name, L, _rf in KINDS:
        f = L * 2 * h * (HD + HD) * PAIRS[name](T)
        b = 0.0
        pos = 0
        while pos < T:
            c = min(C, T - pos)
            prefix = pos + c
            r = min(prefix, c + WIN)                      # window rows this chunk can touch
            if name == "C4":
                r += prefix // 4                          # union of the tile top-512 sets
            elif name == "C128":
                r += prefix // 128
            b += r * row + c * q_o
            pos += c
        b *= L
        per[name] = (b, f)
        b_tot += b
        f_tot += f
    fi = L_C4 * 2 * hi * IDX_D * pairs_idx(T)
    bi = 0.0
    pos = 0
    while pos < T:
        c = min(C, T - pos)
        prefix = pos + c
        bi += (prefix // 4) * irow + c * (hi * IDX_D * 2 + 4 * (prefix // 4))
        pos += c
    bi *= L_C4
    per["IDX"] = (bi, fi)
    per["TOTAL"] = (b_tot + bi, f_tot + fi)
    return per


# ---------------------------------------------------------------- reference geometries
MLA_ROW = (512 + 64) * 2
MLA_ROW_FP8 = 512 + 64 * 2 + 4


def mla_dec(L, h, T, row=MLA_ROW):
    return L * T * row, L * 2 * h * T * (576 + 512)


def mla_pf(L, h, T, C, row=MLA_ROW):
    f = L * 2 * h * (576 + 512) * (T * (T + 1) / 2)
    b = 0.0
    pos = 0
    while pos < T:
        c = min(C, T - pos)
        b += (pos + c) * row + c * (h * 576 * 2 + h * 512 * 2)
        pos += c
    return L * b, f


# ---------------------------------------------------------------- crossovers
def pf_crossover(P, C, fp8=False):
    """smallest T where the whole prefill attention path (incl. indexer) has AI >= ridge."""
    T = 128
    while T <= 1 << 21:
        p = pf_attn(T, P, C, fp8)["TOTAL"]
        if p[1] / p[0] >= RIDGE:
            return T
        T += 128
    return None


def pf_crossover_kind(name, L, P, C, fp8=False):
    row = ROW_FP8 if fp8 else ROW_BF16
    h = HEADS / P
    q_o = h * HD * 2 * 2
    T = 128
    while T <= 1 << 21:
        f = 2 * h * (HD + HD) * PAIRS[name](T)
        b = 0.0
        pos = 0
        while pos < T:
            c = min(C, T - pos)
            prefix = pos + c
            r = min(prefix, c + WIN)
            if name == "C4":
                r += prefix // 4
            elif name == "C128":
                r += prefix // 128
            b += r * row + c * q_o
            pos += c
        if f / b >= RIDGE:
            return T
        T += 128
    return None


def dec_overtake(P, fp8=False, idx_fp4=False, shared_sharded=True):
    """T where decode attention bytes == the weight stream on this rank."""
    w = weight_stream(P, shared_sharded)
    T = 1024
    while T <= 1 << 22:
        if dec_attn(T, P, fp8, idx_fp4)["TOTAL"][0] >= w:
            return T
        T = int(T * 1.02) + 1
    return None


def idx_crossover(P, fp8=False, idx_fp4=False):
    """T at which the indexer + top-512 gather costs fewer bytes than reading the whole
    ratio-4 compressed stream densely (the sparse-selection win on bytes alone)."""
    row = ROW_FP8 if fp8 else ROW_BF16
    irow = IROW_FP4 if idx_fp4 else IROW_BF16
    T = 512
    while T <= 1 << 22:
        dense = L_C4 * (T // 4) * row
        sparse = L_C4 * min(IDX_TOPK, T // 4) * row + L_C4 * (T // 4) * (irow + 8)
        if sparse < dense:
            return T
        T += 512
    return None


def pf_overtake(P, C=8192):
    """T where the prefill attention roof equals the linear-term roof on this rank."""
    lin = lin_pf_us(P, C) * 1e-6
    T = 1024
    while T <= 1 << 22:
        b, f = pf_attn(T, P, C)["TOTAL"]
        if max(b / BW, f / F) >= lin * T:
            return T
        T = int(T * 1.03) + 1
    return None


def batch_crit(T, P, fp8=False, idx_fp4=False):
    """decode batch size at which attention bytes equal the weight stream."""
    return weight_stream(P) / dec_attn(T, P, fp8, idx_fp4)["TOTAL"][0]


FA_PAD_H = 8   # FA_PAD, op_attention.h:51 -- LDS row padding in halves


def gb(x):
    return x / 1e9


def us(x):
    return x * 1e6


# ---------------------------------------------------------------- report
def main():
    print(f"ridge {RIDGE:.1f} FLOP/B   BW {BW/1e9:.0f} GB/s   F {F/1e12:.0f} TFLOP/s")
    print(f"layers: W(ratio 0)={L_W}  C4={L_C4}  C128={L_C128}   +3 MTP DSpark blocks (ratio 0)")
    print(f"row_B bf16 {ROW_BF16}  fp8 {ROW_FP8}   idx row bf16 {IROW_BF16}  fp4 {IROW_FP4}")
    print(f"expert {EXPERT_B/1e6:.2f} MB fp4+scale   shared expert {SHARED_B/1e6:.2f} MB fp8")
    print(f"replicated weight stream {REP/1e9:.3f} GB/rank/token; sharded pool "
          f"{(SHARD+SHARD_SHARED)/1e9:.3f} GB")
    for P in (2, 4, 8):
        print(f"  TP{P}: weight stream {weight_stream(P)/1e9:.3f} GB/rank/token "
              f"(shared expert replicated instead: {weight_stream(P, False)/1e9:.3f}); "
              f"active FLOP/token {active_flops_per_token(P)/1e9:.2f} G; "
              f"lin_pf 8192-chunk {lin_pf_us(P, 8192):.2f} us/token")

    for P in (2, 4, 8):
        w = weight_stream(P)
        print(f"\n================ V4-Flash TP{P}  (h={HEADS//P}/rank, weight stream "
              f"{w/1e9:.3f} GB/token/rank) ================")
        print("\n--- decode, per token per rank, bf16 cache ---")
        print("| T | kind | rows/layer | bytes MB | GFLOP | AI FLOP/B | mem us | cmp us | bound |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for T in CTX:
            per = dec_attn(T, P)
            for name, L, rf in KINDS:
                b, f = per[name]
                print(f"| {T} | {name} x{L} | {rf(T)} | {b/1e6:.2f} | {f/1e9:.2f} | "
                      f"{f/b:.1f} | {us(b/BW):.1f} | {us(f/F):.2f} | "
                      f"{'MEM' if f/b < RIDGE else 'CMP'} |")
            b, f = per["IDX"]
            print(f"| {T} | IDX x{L_C4} | {rows_idx(T)} | {b/1e6:.2f} | {f/1e9:.2f} | "
                  f"{f/b:.1f} | {us(b/BW):.1f} | {us(f/F):.2f} | "
                  f"{'MEM' if f/b < RIDGE else 'CMP'} |")
            b, f = per["TOTAL"]
            b8 = dec_attn(T, P, True, True)["TOTAL"][0]
            print(f"| {T} | **TOTAL** | | **{b/1e6:.2f}** | **{f/1e9:.2f}** | {f/b:.1f} | "
                  f"**{us(b/BW):.1f}** | {us(f/F):.2f} | "
                  f"{'MEM' if f/b < RIDGE else 'CMP'} | fp8+fp4 {us(b8/BW):.1f} us "
                  f"| attn/weight {b/w*100:.1f}% |")
        print(f"decode writes {dec_write(0)/1e3:.1f} KB/token ({us(dec_write(0)/BW):.2f} us)")
        print(f"decode attention bytes == weight stream at T = {dec_overtake(P)} (bf16), "
              f"{dec_overtake(P, True, True)} (fp8 kv + fp4 idx)")
        ar = L_C4 * (CTX[-1] // 4) * 4
        print(f"indexer all-reduce at 1M, TP{P}: {ar/1e6:.1f} MB/token/rank -> "
              f"{us(ar * 2 * (P-1) / P / IF_BW):.1f} us over IF; replicating the indexer "
              f"instead costs {L_C4*(8192*1024+64*4096)*(1-1/P)/1e6:.1f} MB/token of extra weights")

        print("\n--- prefill, chunked at 8192, per rank ---")
        print("| T | kind | TFLOP | bytes GB | AI FLOP/B | mem ms | cmp ms | roof ms | bound |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for T in CTX:
            per = pf_attn(T, P, 8192)
            for name in ("W", "C4", "C128", "IDX", "TOTAL"):
                b, f = per[name]
                m, c = b / BW * 1e3, f / F * 1e3
                print(f"| {T} | {name} | {f/1e12:.3f} | {gb(b):.3f} | {f/b:.0f} | {m:.3f} | "
                      f"{c:.3f} | {max(m,c):.3f} | {'MEM' if f/b < RIDGE else 'CMP'} |")
            lin = lin_pf_us(P, 8192) * T * 1e-3
            b, f = per["TOTAL"]
            print(f"| {T} | (linear-term roof {lin:.1f} ms) | | | | | | "
                  f"attn/linear {max(b/BW,f/F)*1e3/lin:.3f}x | |")
        for name, L, _ in KINDS:
            print(f"prefill mem->compute crossover, {name} alone: "
                  f"T = {pf_crossover_kind(name, L, P, 8192)}")
        print(f"prefill mem->compute crossover, whole attention path: T = {pf_crossover(P, 8192)}")
        print(f"prefill attention roof overtakes the linear-term roof "
              f"({lin_pf_us(P,8192):.2f} us/token) at T = {pf_overtake(P)}")
        print("decode batch at which attention bytes == weight stream:")
        for T in CTX:
            print(f"   T={T}: B = {batch_crit(T,P):.1f} (bf16), "
                  f"{batch_crit(T,P,True,True):.1f} (fp8 kv + fp4 idx)")

    print("\n================ indexer vs dense/compressed paths ================")
    print(f"sparse-selection byte crossover (top-512 gather + index scan < dense ratio-4 "
          f"stream): T = {idx_crossover(2)} (bf16 idx), {idx_crossover(2, False, True)} (fp4 idx)")
    print("| T | C4 dense-compressed GB | C4 top512 gather GB | indexer scan GB | sparse total GB | dense/sparse |")
    print("|---:|---:|---:|---:|---:|---:|")
    for T in CTX:
        dense = L_C4 * (T // 4) * ROW_BF16
        gath = L_C4 * min(IDX_TOPK, T // 4) * ROW_BF16
        scan = L_C4 * (T // 4) * (IROW_BF16 + 8)
        print(f"| {T} | {gb(dense):.4f} | {gb(gath):.4f} | {gb(scan):.4f} | "
              f"{gb(gath+scan):.4f} | {dense/(gath+scan):.2f}x |")

    print("\n================ comparison, per rank, decode at 128k ================")
    print("| model | ranks | h/rank | attn bytes/token/rank GB | AI | mem roof us | "
          "prefill attn TFLOP @32k | prefill roof ms @32k | weight stream GB |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for P in (2, 4, 8):
        d = dec_attn(131072, P)["TOTAL"]
        pf = pf_attn(32768, P, 8192)["TOTAL"]
        print(f"| DeepSeek-V4-Flash | TP{P} | {HEADS//P} | {gb(d[0]):.4f} | {d[1]/d[0]:.1f} | "
              f"{us(d[0]/BW):.1f} | {pf[1]/1e12:.2f} | "
              f"{max(pf[0]/BW, pf[1]/F)*1e3:.1f} | {weight_stream(P)/1e9:.2f} |")
    for L, h, P, nm, w in ((78, 16, 4, "GLM-5.3", 17.70), (78, 8, 8, "GLM-5.3", 10.23),
                           (24, 12, 8, "Kimi-K3 (MLA layers)", 16.78)):
        d = mla_dec(L, h, 131072)
        pf = mla_pf(L, h, 32768, 8192)
        print(f"| {nm} | TP{P} | {h} | {gb(d[0]):.4f} | {d[1]/d[0]:.1f} | {us(d[0]/BW):.1f} | "
              f"{pf[1]/1e12:.2f} | {max(pf[0]/BW, pf[1]/F)*1e3:.1f} | {w:.2f} |")

    print("\n================ kernel-level re-stream AI (prefill) ================")
    print("  convention of the GLM report: FLOPs of one (BQ x BKV) slab / the LDS slab bytes")
    for P in (2, 4, 8):
        h = HEADS / P
        for DK, DR, BQ, BKV in ((512, 0, 64, 32), (512, 0, 64, 48), (512, 0, 32, 32)):
            slab = BKV * (DK + DR + FA_PAD_H) * 2
            ai = 2 * h * BQ * BKV * (HD + HD) / slab
            print(f"  TP{P} <{DK},{DR}> BQ={BQ} BKV={BKV} slab {slab:,} B: {ai:.0f} FLOP/B "
                  f"({'above' if ai > RIDGE else 'BELOW'} ridge by {ai/RIDGE:.1f}x)")
    print(f"  GLM-5.3 TP4 reference: 2*16*64*32*1088/36864 = "
          f"{2*16*64*32*1088/36864:.0f} FLOP/B")
    print("  indexer op-117 shipped arm re-fetch AI: "
          f"{2*32*32*IDX_D/((32*IDX_D + 32*IDX_D)*2):.1f} FLOP/B (both operands from global); "
          f"row-resident arm (K slab in LDS, Q hoisted to VGPR): "
          f"{2*32*32*IDX_D/(32*IDX_D*2):.0f} FLOP/B")

    print("\n================ LDS budgets (gfx942, 64 KiB cap; flash `fa` arena 58,368 B) ==========")
    def pf2_lds(DK, DR, BKV, swz=0):
        return (BKV * (DK + DR + 8) + 3 * swz + 4 * 16 * BKV) * 2
    for DK, DR in ((512, 64), (512, 0)):
        for BKV in (32, 48, 64):
            v = pf2_lds(DK, DR, BKV)
            print(f"  d_flash_mla_prefill_v2<{DK},{DR}> BKV={BKV}: {v:,} B "
                  f"{'FITS 58,368' if v <= 58368 else 'over'}")
    def pf_lds_floats(DK, DR, BQ, ksplit, np_):
        return (BQ * (DK + DR + 8) + 32 * (DK // ksplit + DR + 8) + np_ * 32 * 32 + 1) // 2
    for DK, DR in ((512, 64), (512, 0)):
        for BQ, ksplit, np_ in ((32, 2, 1), (64, 2, 2), (64, 1, 2), (32, 1, 1)):
            v = pf_lds_floats(DK, DR, BQ, ksplit, np_) * 4
            print(f"  d_flash_mla_prefill<{DK},{DR}> BQ={BQ} KSPLIT={ksplit} NP={np_}: "
                  f"{v:,} B {'FITS 65,536' if v <= 65536 else 'over'}")




def extra():
    print("\n================ chunk ladder: prefill attention bytes vs chunk size ==========")
    print("| T | TP | C=2048 GB | C=8192 GB | ratio | C=2048 roof ms | C=8192 roof ms |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for T in (32768, 131072, 262144):
        for P in (4, 8):
            a = pf_attn(T, P, 2048)["TOTAL"]
            b = pf_attn(T, P, 8192)["TOTAL"]
            print(f"| {T} | TP{P} | {gb(a[0]):.1f} | {gb(b[0]):.1f} | {a[0]/b[0]:.2f}x | "
                  f"{max(a[0]/BW,a[1]/F)*1e3:.1f} | {max(b[0]/BW,b[1]/F)*1e3:.1f} |")

    print("\n================ per-sequence cache footprint (replicated on EVERY rank) =======")
    print("| T | window MB | C4 compressed MB | C128 compressed MB | indexer MB | total GB | fp8+fp4 GB |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for T in CTX:
        win = 43 * WIN * ROW_BF16
        c4 = L_C4 * (T // 4) * ROW_BF16
        c128 = L_C128 * (T // 128) * ROW_BF16
        ix = L_C4 * (T // 4) * IROW_BF16
        w8 = 43 * WIN * ROW_FP8 + L_C4 * (T // 4) * ROW_FP8 + \
            L_C128 * (T // 128) * ROW_FP8 + L_C4 * (T // 4) * IROW_FP4
        print(f"| {T} | {win/1e6:.2f} | {c4/1e6:.1f} | {c128/1e6:.1f} | {ix/1e6:.1f} | "
              f"{(win+c4+c128+ix)/1e9:.3f} | {w8/1e9:.3f} |")

    print("\n================ decode share of the byte roof vs T (solve) ====================")
    slope = L_C128 * ROW_BF16 / 128 + L_C4 * (IROW_BF16 + 8) / 4
    const = 43 * WIN * ROW_BF16 + L_C4 * IDX_TOPK * ROW_BF16
    print(f"  large-T decode bytes/token/rank = {slope:.0f}*T + {const/1e6:.1f} MB")
    for P in (2, 4, 8):
        w = weight_stream(P)
        print(f"  TP{P}: attention = 50% of the byte roof at T = {(0.5*w-const)/slope/1e3:.0f}k, "
              f"100% at T = {(w-const)/slope/1e3:.0f}k")

    print("\n================ derived prefill FLOP crossovers (TP-independent) ==============")
    print("  indexer T^2 coefficient  = 21*2*h*128/8   = 672*h")
    print("  C128    T^2 coefficient  = 20*2*h*1024/256 = 160*h  -> indexer is 4.2x")
    print("  C4      linear coeff     = 21*2*h*1024*640 = 27.5e6*h")
    print(f"  indexer overtakes C4 at T = {21*2*1024*640/(21*2*128/8):.0f}")
    print(f"  C128 overtakes C4 at T = {256*21*640/20:.0f}")


if __name__ == "__main__":
    main()
    extra()
