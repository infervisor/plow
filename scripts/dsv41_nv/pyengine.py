"""Python prototype of the plowrt DeepSeek-V4.1 sm_90a engine: one layer's forward as a sequence of
cubin kernel launches over explicit device buffers and per-slot caches.

It exists to pin the launch schedule and every numeric detail against the reference model.py before
the same schedule is written in Rust (crates/plowrt/src/dsv41). Shapes follow the checkpoint config.
"""
import json
import math
import os
import struct

import torch

from cudrv import Cubin, f32, i32, i64

H = 5120
HC = 4
NH = 64
HD = 512
RD = 64
QR = 1280
OG = 8
OR = 1024
WIN = 128
E = 384
TOPK = 6
MI = 2304
IXH = 32
IXD = 128
IXK = 512
EPS = 1e-20
IX_SMEM = (4 * IXH * (IXD + 8) + 64 * (IXD + 8)) * 2 + 2 * 4 * 64 * 4
def MOE_SMEM(k, bm=64):
    return 3 * bm * 160 + 3 * 128 * 80 + (128 + bm) * (k // 32) + bm * 4 + 16


SA_SMEM = (64 * 520 + 64 * 520 + 64 * 72) * 2 + 4 * 64 * 4 * 2 + 64 * 4


class Ckpt:
    """Tensor access over the HF safetensors shards (header-parsed, read on demand)."""

    def __init__(self, path):
        self.path = path
        self.index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.cfg = json.load(open(os.path.join(path, "config.json")))["text_config"]
        self.headers = {}

    def _hdr(self, f):
        if f not in self.headers:
            with open(os.path.join(self.path, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[f] = (json.loads(fh.read(n)), 8 + n)
        return self.headers[f]

    DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.uint8, "F8_E8M0": torch.uint8, "I8": torch.uint8}

    def get(self, name, rows=None):
        f = self.index[name]
        hdr, base = self._hdr(f)
        h = hdr[name]
        dt = self.DT[h["dtype"]]
        shape = h["shape"]
        s, e = h["data_offsets"]
        if rows is not None:  # a contiguous row range only
            r0, r1 = rows
            row_bytes = (e - s) // shape[0]
            s, e = s + r0 * row_bytes, s + r1 * row_bytes
            shape = [r1 - r0] + shape[1:]
        with open(os.path.join(self.path, f), "rb") as fh:
            fh.seek(base + s)
            buf = bytearray(fh.read(e - s))
        return torch.frombuffer(buf, dtype=dt).reshape(shape)

    def has(self, name):
        return name in self.index


def pos_tables(cfg, max_pos, compressed, dev):
    """cos/sin [max_pos][rd/2] f32, as model.py precompute_freqs_cis (YaRN on compressed layers)."""
    dim = RD
    if compressed:
        base, orig = float(cfg["compress_rope_theta"]), cfg["rope_scaling"]["original_max_position_embeddings"]
    else:
        base, orig = float(cfg["rope_theta"]), 0
    factor = cfg["rope_scaling"]["factor"]
    beta_fast, beta_slow = cfg["rope_scaling"]["beta_fast"], cfg["rope_scaling"]["beta_slow"]
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=dev) / dim))
    if orig > 0:
        def corrected_dim(rot):
            return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=dev) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    f = torch.outer(torch.arange(max_pos, device=dev), freqs)
    cis = torch.polar(torch.ones_like(f), f)
    return cis.real.contiguous(), cis.imag.contiguous()


class Layer:
    """One layer's weights in the engine's device formats."""

    def __init__(self, ck: Ckpt, L: int, dev):
        c = ck.cfg
        p = f"layers.{L}."
        g = lambda n: ck.get(p + n).to(dev)
        self.L = L
        self.ratio = c["compress_ratios"][L]
        self.kv_source = L in c["kv_source_layer_ids"]
        self.index_source = L in c["index_source_layer_ids"]
        self.cand_source = L == c["candidate_source_layer_id"]
        self.uses_cand = 0 <= c["candidate_source_layer_id"] < L
        self.attn_norm, self.ffn_norm = g("attn_norm.weight"), g("ffn_norm.weight")
        for k in ("attn", "ffn"):
            setattr(self, f"hc_{k}_fn", g(f"hc_{k}_fn"))
            setattr(self, f"hc_{k}_base", g(f"hc_{k}_base"))
            setattr(self, f"hc_{k}_scale", g(f"hc_{k}_scale"))
        a = "attn."
        self.wq_a, self.wq_a_s = g(a + "wq_a.weight"), g(a + "wq_a.scale")
        self.q_norm = g(a + "q_norm.weight")
        self.wq_b, self.wq_b_s = g(a + "wq_b.weight"), g(a + "wq_b.scale")
        self.wkv, self.wkv_s = g(a + "wkv.weight"), g(a + "wkv.scale")
        self.kv_norm = g(a + "kv_norm.weight")
        self.wo_a, self.wo_a_s = g(a + "wo_a.weight"), g(a + "wo_a.scale")
        self.wo_b, self.wo_b_s = g(a + "wo_b.weight"), g(a + "wo_b.scale")
        self.sink = g(a + "attn_sink")
        if self.kv_source:
            self.c_wkv = g(a + "compressor.wkv.weight")
            self.c_norm = g(a + "compressor.norm.weight")
            self.c_wgate = g(a + "compressor.wgate.weight") if self.ratio > 1 else None
        if self.index_source:
            self.i_wq_b, self.i_wq_b_s = g(a + "indexer.wq_b.weight"), g(a + "indexer.wq_b.scale")
            self.i_wproj = g(a + "indexer.weights_proj.weight")
            if self.kv_source:
                self.i_wk, self.i_knorm = g(a + "indexer.wk.weight"), g(a + "indexer.k_norm.weight")
        f = "ffn."
        self.gate_w, self.gate_b = g(f + "gate.weight"), g(f + "gate.bias")
        se = f + "shared_experts."
        self.sh_w13 = torch.cat([g(se + "w1.weight"), g(se + "w3.weight")])
        self.sh_w13_s = torch.cat([g(se + "w1.scale"), g(se + "w3.scale")])
        self.sh_w2, self.sh_w2_s = g(se + "w2.weight"), g(se + "w2.scale")
        # routed experts: W13 [E][2*MI][H/2] fp4, W2 [E][H][MI/2]
        self.w13 = torch.empty(E, 2 * MI, H // 2, dtype=torch.uint8, device=dev)
        self.w13_s = torch.empty(E, 2 * MI, H // 32, dtype=torch.uint8, device=dev)
        self.w2 = torch.empty(E, H, MI // 2, dtype=torch.uint8, device=dev)
        self.w2_s = torch.empty(E, H, MI // 32, dtype=torch.uint8, device=dev)
        for e in range(E):
            ep = f"{p}{f}experts.{e}."
            self.w13[e, :MI] = ck.get(ep + "w1.weight").to(dev)
            self.w13[e, MI:] = ck.get(ep + "w3.weight").to(dev)
            self.w13_s[e, :MI] = ck.get(ep + "w1.scale").to(dev)
            self.w13_s[e, MI:] = ck.get(ep + "w3.scale").to(dev)
            self.w2[e] = ck.get(ep + "w2.weight").to(dev)
            self.w2_s[e] = ck.get(ep + "w2.scale").to(dev)
        self.engram = ck.has(p + "engram.wkv.weight")
        if self.engram:
            self.e_wkv, self.e_wkv_s = g("engram.wkv.weight"), g("engram.wkv.scale")
            self.e_qw, self.e_kw = g("engram.q_weight"), g("engram.k_weight")


class SeqCache:
    """Per-sequence caches for one layer (the engine allocates these per slot)."""

    def __init__(self, layer: Layer, max_len, dev):
        self.win = torch.zeros(WIN, HD, dtype=torch.bfloat16, device=dev)
        if layer.kv_source:
            n = max_len // layer.ratio + 64
            self.cmp = torch.zeros(n, HD, dtype=torch.bfloat16, device=dev)
            self.idx_k = torch.zeros(n, IXD, dtype=torch.bfloat16, device=dev)
            if layer.ratio > 1:
                self.st_kv = torch.zeros(layer.ratio, HD, dtype=torch.float32, device=dev)
                self.st_sc = torch.full((layer.ratio, HD), -float("inf"), dtype=torch.float32, device=dev)


class Shared:
    """model.py SharedAttentionRuntime for one forward: the latest source caches and index picks."""

    def __init__(self):
        self.cmp = None
        self.idx_k = None
        self.topk = None  # [T][kout] int32, offset applied
        self.keep = None


class Engine:
    def __init__(self, cubin, cfg, max_pos, dev="cuda"):
        self.K = Cubin(cubin)
        self.cfg = cfg
        self.dev = dev
        self.cos_w, self.sin_w = pos_tables(cfg, max_pos, False, dev)
        self.cos_c, self.sin_c = pos_tables(cfg, max_pos, True, dev)

    # ------------------------------------------------------------------ primitives
    def rmsnorm(self, x, w):
        M, D = x.shape
        y = torch.empty_like(x)
        self.K.launch("dsv_rmsnorm", (M,), (256,), [y, x, w, i32(D), i64(D), i64(D), f32(EPS)])
        return y

    def quant(self, x):
        M, Kd = x.shape
        q = torch.empty(M, Kd, dtype=torch.uint8, device=self.dev)
        s = torch.empty(M, Kd // 32, dtype=torch.uint8, device=self.dev)
        self.K.launch("dsv_act_quant_fp8", ((M * Kd // 32 + 7) // 8,), (256,), [q, s, None, x, i32(M), i32(Kd), i64(Kd)])
        return q, s

    def fq8(self, x):
        M, Kd = x.shape
        self.K.launch("dsv_act_quant_fp8", ((M * Kd // 32 + 7) // 8,), (256,), [None, None, x, x, i32(M), i32(Kd), i64(Kd)])

    def fq4(self, x, gs, e4m3):
        n = x.numel()
        self.K.launch("dsv_fp4_fakequant", ((n + 255) // 256,), (256,), [x, i64(n), i32(gs), i32(e4m3)])

    def w8a8(self, qs, w, ws, out_f32=False):
        q, s = qs
        M, Kd = q.shape
        N = w.shape[0]
        c = torch.empty(M, N, dtype=torch.float32 if out_f32 else torch.bfloat16, device=self.dev)
        self.K.launch("dsv_gemm_w8a8", ((N + 127) // 128, (M + 63) // 64), (128,),
                      [c, q, s, w, ws, i32(M), i32(N), i32(Kd), i64(N), i32(int(out_f32)), i32(1), None], smem=3 * (64 + 128) * 144)
        return c

    def bf16w(self, a, w, ws=None, out_f32=False):
        M, Kd = a.shape
        N = w.shape[0]
        c = torch.empty(M, N, dtype=torch.float32 if out_f32 else torch.bfloat16, device=self.dev)
        self.K.launch("dsv_gemm_bf16w", ((N + 127) // 128, (M + 63) // 64, 1), (128,),
                      [c, a, w, ws, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(int(ws is not None)), i32(int(out_f32)),
                       i64(0), i64(0), i64(0), i64(0), i32(1), None])
        return c

    def f32gemm(self, a, w):
        M, Kd = a.shape
        N = w.shape[0]
        c = torch.empty(M, N, dtype=torch.float32, device=self.dev)
        self.K.launch("dsv_gemm_f32", ((N + 63) // 64, (M + 63) // 64), (256,),
                      [c, a, w, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(int(a.dtype == torch.bfloat16)),
                       i32(int(w.dtype == torch.bfloat16))])
        return c

    def rope(self, x, n_tok, n_head, tok_stride, head_stride, width, pos, compressed, inverse=False, pos_mul=1):
        """Rotate the last RD dims of each `width`-wide head row."""
        cosb, sinb = (self.cos_c, self.sin_c) if compressed else (self.cos_w, self.sin_w)
        total = n_tok * n_head * (RD // 2)
        self.K.launch("dsv_rope", ((total + 255) // 256,), (256,),
                      [x, pos, cosb, sinb, i32(n_tok), i32(n_head), i64(tok_stride), i64(head_stride), i32(width - RD),
                       i32(RD), i32(pos_mul), i32(0), i32(int(inverse))])

    def hc_mixes(self, x, fn, scale, base, T):
        rsq = torch.empty(T, dtype=torch.float32, device=self.dev)
        self.K.launch("dsv_row_rsqrt", (T,), (256,), [rsq, x, i32(HC * H), f32(EPS)])
        mixes = self.f32gemm(x.view(T, HC * H), fn)
        pre = torch.empty(T, HC, dtype=torch.float32, device=self.dev)
        post = torch.empty(T, HC, dtype=torch.float32, device=self.dev)
        comb = torch.empty(T, HC, HC, dtype=torch.float32, device=self.dev)
        self.K.launch("dsv_hc_sinkhorn", ((T + 127) // 128,), (128,),
                      [pre, post, comb, mixes, rsq, scale, base, i32(T), i32(self.cfg["hc_sinkhorn_iters"]), f32(self.cfg["hc_eps"])])
        return pre, post, comb

    def hc_pre(self, x, pre, T):
        y = torch.empty(T, H, dtype=torch.bfloat16, device=self.dev)
        self.K.launch("dsv_hc_pre", ((T * H + 255) // 256,), (256,), [y, x, pre, i32(T), i32(H)])
        return y

    def hc_post(self, y, res, post, comb, T):
        out = torch.empty(T, HC, H, dtype=torch.bfloat16, device=self.dev)
        self.K.launch("dsv_hc_post", ((T * H + 255) // 256,), (256,), [out, y, res, post, comb, i32(T), i32(H)])
        return out

    # ------------------------------------------------------------------ sublayers
    def attention(self, ly: Layer, hn, cache: SeqCache, sh: Shared, start_pos):
        """One sequence: T = hn rows at positions start_pos .. start_pos+T-1 (prefill: start_pos 0;
        decode: T = 1)."""
        T = hn.shape[0]
        dev = self.dev
        decode = start_pos > 0
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.int32, device=dev)
        comp = ly.ratio > 0
        hq = self.quant(hn)
        qr = self.rmsnorm(self.w8a8(hq, ly.wq_a, ly.wq_a_s), ly.q_norm)
        qrq = self.quant(qr)
        q = self.w8a8(qrq, ly.wq_b, ly.wq_b_s)  # [T][64*512]
        self.rope(q, T, NH, NH * HD, HD, HD, pos, comp)
        kv = self.rmsnorm(self.w8a8(hq, ly.wkv, ly.wkv_s), ly.kv_norm)
        self.rope(kv, T, 1, HD, HD, HD, pos, comp)
        self.fq8(kv)
        # window ring: position p lives at slot p % WIN
        if not decode:
            n = min(T, WIN)
            slots = torch.arange(T - n, T, device=dev) % WIN
            cache.win[slots] = kv[T - n:]
            win_src, off = kv, T
        else:
            cache.win[start_pos % WIN] = kv[0]
            win_src, off = cache.win, WIN
        cmp_idx, n_cmp = None, 0
        if comp:
            r = ly.ratio
            end = start_pos + T
            clen_end = end // r
            latent = None
            G = 0
            if ly.kv_source:
                if r > 1:
                    kvf = self.f32gemm(hn, ly.c_wkv)
                    sc = self.f32gemm(hn, ly.c_wgate)
                    if not decode:
                        G = T // r
                        pooled = torch.empty(max(G, 1), HD, dtype=torch.bfloat16, device=dev)
                        n_thr = G * HD + (T % r) * HD
                        if n_thr:
                            self.K.launch("dsv_compress_pool_prefill", ((n_thr + 255) // 256,), (256,),
                                          [pooled, kvf, sc, i32(T), i32(HD), i32(r), cache.st_kv, cache.st_sc])
                    else:
                        pooled = torch.empty(1, HD, dtype=torch.bfloat16, device=dev)
                        pp = torch.tensor([start_pos], dtype=torch.int32, device=dev)
                        sk = torch.tensor([cache.st_kv.data_ptr()], dtype=torch.uint64, device=dev)
                        ss = torch.tensor([cache.st_sc.data_ptr()], dtype=torch.uint64, device=dev)
                        self.K.launch("dsv_compress_pool_decode", (1,), (256,), [pooled, kvf, sc, i32(1), i32(HD), i32(r), pp, sk, ss])
                        G = 1 if (start_pos + 1) % r == 0 else 0
                    if G:
                        latent = self.rmsnorm(pooled[:G].contiguous(), ly.c_norm)
                else:
                    latent = self.rmsnorm(self.bf16w(hn, ly.c_wkv), ly.c_norm)
                    G = T
                sh.cmp, sh.idx_k = cache.cmp, cache.idx_k
            # a latent stands for the first token of its group: group j of this call is at position
            # (base_group + j) * r, base_group = start_pos // r (decode: start_pos + 1 - r)
            gbase = start_pos // r
            if ly.index_source:
                if latent is not None and ly.kv_source:
                    k = self.rmsnorm(self.bf16w(latent, ly.i_wk), ly.i_knorm)
                    gp = torch.arange(gbase, gbase + G, dtype=torch.int32, device=dev)
                    self.rope(k, G, 1, IXD, IXD, IXD, gp, True, pos_mul=r)
                    self.fq4(k, 32, 0)
                    cache.idx_k[gbase:gbase + G] = k
                if clen_end > 0:
                    qi = self.w8a8(qrq, ly.i_wq_b, ly.i_wq_b_s)  # [T][32*128]
                    self.rope(qi, T, IXH, IXH * IXD, IXD, IXD, pos, True)
                    self.fq4(qi, 32, 0)
                    w = self.bf16w(hn, ly.i_wproj)
                    n = w.numel()
                    self.K.launch("dsv_scale_bf16", ((n + 255) // 256,), (256,), [w, i64(n), f32(IXD ** -0.5 * IXH ** -0.5)])
                    clen = ((pos + 1) // r) if not decode else torch.full((T,), clen_end, dtype=torch.int32, device=dev)
                    clen = clen.to(torch.int32)
                    S = clen_end
                    s_ld = (S + 63) // 64 * 64
                    score = torch.empty(T, s_ld, dtype=torch.bfloat16, device=dev)
                    kp = torch.tensor([sh.idx_k.data_ptr()], dtype=torch.uint64, device=dev)
                    self.K.launch("dsv_index_score", (s_ld // 64, (T + 3) // 4), (256,),
                                  [score, i64(s_ld), qi, w, kp, i32(T), clen, i32(T)], smem=IX_SMEM)
                    keep = None
                    cb = self.cfg["candidate_block_size"]
                    if ly.cand_source:
                        nb_ld = (S + cb - 1) // cb
                        bs = torch.empty(T, nb_ld, dtype=torch.bfloat16, device=dev)
                        self.K.launch("dsv_cand_block_scores", ((nb_ld + 255) // 256, T), (256,),
                                      [bs, i64(nb_ld), score, i64(s_ld), clen, i32(cb)])
                        nbl = ((clen + cb - 1) // cb).to(torch.int32)
                        kb = min(self.cfg["candidate_topk_blocks"], nb_ld)
                        bidx = torch.empty(T, kb, dtype=torch.int32, device=dev)
                        self.K.launch("dsv_topk_select", (T,), (1024,),
                                      [bidx, i32(kb), bs, i64(nb_ld), nbl, i32(kb), i32(0), None, i64(0), i32(1)])
                        keep = torch.empty(T, nb_ld, dtype=torch.uint8, device=dev)
                        self.K.launch("dsv_keep_from_idx", (T,), (256,), [keep, i64(nb_ld), i32(nb_ld), bidx, i32(kb)])
                        sh.keep = (keep, nb_ld)
                    elif ly.uses_cand and sh.keep is not None:
                        keep = sh.keep
                    kout = min(IXK, S)
                    idx = torch.empty(T, kout, dtype=torch.int32, device=dev)
                    kk, kld = (keep if isinstance(keep, tuple) else (keep, keep.shape[1] if keep is not None else 0))
                    self.K.launch("dsv_topk_select", (T,), (1024,),
                                  [idx, i32(kout), score, i64(s_ld), clen, i32(IXK), i32(off), kk, i64(kld), i32(cb)])
                    sh.topk = idx
                else:
                    sh.topk = None
            if latent is not None:
                gp = torch.arange(gbase, gbase + G, dtype=torch.int32, device=dev)
                self.rope(latent, G, 1, HD, HD, HD, gp, True, pos_mul=r)
                self.fq4(latent, 16, 1)
                cache.cmp[gbase:gbase + G] = latent
            if sh.topk is not None:
                cmp_idx, n_cmp = sh.topk, sh.topk.shape[1]
        # full index table and the attention itself
        n_idx = WIN + n_cmp
        table = torch.empty(T, n_idx, dtype=torch.int32, device=dev)
        self.K.launch("dsv_attn_index", (T,), (256,),
                      [table, i32(T), i32(T if not decode else 1), i32(WIN), i32(int(decode)), torch.tensor([start_pos], dtype=torch.int32, device=dev),
                       cmp_idx, i32(n_cmp)])
        o = torch.empty(T, NH * HD, dtype=torch.bfloat16, device=dev)
        wp = torch.tensor([win_src.data_ptr()], dtype=torch.uint64, device=dev)
        cp = torch.tensor([sh.cmp.data_ptr() if (comp and sh.cmp is not None) else 0], dtype=torch.uint64, device=dev)
        self.K.launch("dsv_sparse_attn", (T,), (512,),
                      [o, q, table, i32(n_idx), wp, cp, i32(off), i32(T if not decode else 1), ly.sink, f32(HD ** -0.5), None],
                      smem=SA_SMEM)
        self.rope(o, T, NH, NH * HD, HD, HD, pos, comp, inverse=True)
        # wo_a: 8 groups, o[t][g*4096 ..] x wo_a[g*1024 ..]^T -> ga[t][g*1024 ..]
        ga = torch.empty(T, OG * OR, dtype=torch.bfloat16, device=dev)
        KG = NH * HD // OG
        self.K.launch("dsv_gemm_bf16w", ((OR + 127) // 128, (T + 63) // 64, OG), (128,),
                      [ga, o, ly.wo_a, ly.wo_a_s, i32(T), i32(OR), i32(KG), i64(NH * HD), i64(OG * OR), i32(1), i32(0),
                       i64(KG), i64(OR * KG), i64((OR // 32) * (KG // 32)), i64(OR), i32(1), None])
        return self.w8a8(self.quant(ga), ly.wo_b, ly.wo_b_s)


    def moe(self, ly: Layer, hn):
        T = hn.shape[0]
        dev = self.dev
        logits = self.f32gemm(hn, ly.gate_w)
        idx = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        wt = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
        self.K.launch("dsv_moe_route", ((T + 7) // 8,), (256,),
                      [idx, wt, logits, ly.gate_b, i32(T), i32(E), i32(TOPK), i32(1), f32(self.cfg["routed_scaling_factor"])])
        self.dbg_route = (idx, wt)
        n = T * TOPK
        counts = torch.zeros(E, dtype=torch.int32, device=dev)
        self.K.launch("dsv_moe_count", ((n + 255) // 256,), (256,), [counts, idx, i32(n)])
        BM = 64
        max_tiles = (n + BM - 1) // BM + min(n, E)
        offs = torch.empty(E + 1, dtype=torch.int32, device=dev)
        tiles = torch.empty(max_tiles * 2, dtype=torch.int32, device=dev)
        meta = torch.empty(1, dtype=torch.int32, device=dev)
        ctr = torch.empty(E, dtype=torch.int32, device=dev)
        self.K.launch("dsv_moe_offsets", (1,), (512,), [offs, tiles, meta, ctr, counts, i32(E), i32(BM)])
        rows = torch.empty(n, dtype=torch.int32, device=dev)
        rowpos = torch.empty(n, dtype=torch.int32, device=dev)
        row_w = torch.empty(n, dtype=torch.float32, device=dev)
        self.K.launch("dsv_moe_fill", ((n + 255) // 256,), (256,), [rows, rowpos, row_w, ctr, offs, idx, wt, i32(n), i32(TOPK)])
        xq, xs = self.quant(hn)
        gu = torch.empty(n, 2 * MI, dtype=torch.bfloat16, device=dev)
        self.K.launch("dsv_moe_gemm_fp4", ((2 * MI + 127) // 128, max_tiles), (128,),
                      [gu, xq, xs, ly.w13, ly.w13_s, tiles, meta, offs, rows, i32(0), i32(2 * MI), i32(H),
                       i64(2 * MI * H // 2), i64(2 * MI * H // 32)], smem=MOE_SMEM(H))
        hq = torch.empty(n, MI, dtype=torch.uint8, device=dev)
        hs = torch.empty(n, MI // 32, dtype=torch.uint8, device=dev)
        lim = float(self.cfg["swiglu_limit"])
        self.K.launch("dsv_swiglu_quant", ((n * MI // 32 + 7) // 8,), (256,),
                      [hq, hs, gu, gu[:, MI:], i64(2 * MI), row_w, i32(n), i32(MI), f32(lim), None, None])
        down = torch.empty(n, H, dtype=torch.bfloat16, device=dev)
        self.K.launch("dsv_moe_gemm_fp4", ((H + 127) // 128, max_tiles), (128,),
                      [down, hq, hs, ly.w2, ly.w2_s, tiles, meta, offs, rows, i32(1), i32(H), i32(MI),
                       i64(H * MI // 2), i64(H * MI // 32)], smem=MOE_SMEM(MI))
        # shared expert (fp8 weights)
        sgu = self.w8a8((xq, xs), ly.sh_w13, ly.sh_w13_s)
        shq = torch.empty(T, MI, dtype=torch.uint8, device=dev)
        shs = torch.empty(T, MI // 32, dtype=torch.uint8, device=dev)
        self.K.launch("dsv_swiglu_quant", ((T * MI // 32 + 7) // 8,), (256,),
                      [shq, shs, sgu, sgu[:, MI:], i64(2 * MI), None, i32(T), i32(MI), f32(lim), None, None])
        shared = self.w8a8((shq, shs), ly.sh_w2, ly.sh_w2_s)
        y = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        self.K.launch("dsv_moe_combine", (T,), (256,), [y, down, shared, idx, rowpos, i32(T), i32(H), i32(TOPK)])
        return y

    def engram(self, ly: Layer, x, emb):
        """x [T][4][H] in place; emb [T][6144] bf16 (host-gathered, dequantized rows)."""
        T = x.shape[0]
        kv = self.w8a8(self.quant(emb), ly.e_wkv, ly.e_wkv_s)  # [T][25600]
        self.K.launch("dsv_engram_gate", (T, HC), (256,), [x, kv, ly.e_qw, ly.e_kw, None, i32(H), f32(EPS)])

    def layer(self, ly: Layer, x, pre_mix, cache: SeqCache, sh: Shared, start_pos, emb=None, do_engram=True):
        T = x.shape[0]
        if ly.engram and do_engram:
            self.engram(ly, x, emb)
        a_pre, a_post, a_comb = self.hc_mixes(x, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, T)
        hn = self.rmsnorm(self.hc_pre(x, pre_mix, T), ly.attn_norm)
        ao = self.attention(ly, hn, cache, sh, start_pos)
        x2 = self.hc_post(ao, x, a_post, a_comb, T)
        f_pre, f_post, f_comb = self.hc_mixes(x2, ly.hc_ffn_fn, ly.hc_ffn_scale, ly.hc_ffn_base, T)
        hn2 = self.rmsnorm(self.hc_pre(x2, a_pre, T), ly.ffn_norm)
        y = self.moe(ly, hn2)
        x3 = self.hc_post(y, x2, f_post, f_comb, T)
        self.dbg = {"hn": hn, "ao": ao, "x2": x2, "hn2": hn2, "y": y}
        return x3, f_pre
