#!/usr/bin/env python3
"""Export Chatterbox S3Gen (ResembleAI/chatterbox s3gen.safetensors, English, non-meanflow)
weights for libplow_s3gen.so, plus the builtin voice (conds.pt `gen` dict).

Usage:
  HF_HOME=/root/tts-work/hf /root/tts-work/venv-ref/bin/python scripts/tts/s3gen_prep.py \
      out/s3gen.bin [--voice-out out/voice_default.bin] [--conds path/to/conds.pt]

Runs on CPU. Weight norm is folded (the parametrization's computed .weight). Everything is
exported in fp32 in the layout the native GEMMs consume; at create the library pads rows / K
(to 128) and converts GEMM weights to fp16 (round to nearest; the CFM ResNet / final convs also
get an fp16 residual plane w - fp16(w)).

File format (both files; all little-endian):
  offset 0  : 8 bytes magic  "S3GENW01" (weights) or "S3GENV01" (voice)
  offset 8  : u32 format version (=1)
  offset 12 : u32 tensor count N
  then N records:
      u32 name_len, name bytes (utf-8, no NUL)
      u32 dtype (0 = fp32, 1 = int32)
      u32 ndim, u32 dims[ndim]
      u64 byte offset of the data from the START OF THE FILE (64-byte aligned)
      u64 byte size (= 4 * prod(dims))
  then the data: each tensor a contiguous C-order array at its offset.

GEMM weight convention: every Linear / Conv1d is a matrix W[N][K] (N = output channels) with
K = tap-major im2col order, k = tap * Cin + ci:   W[n][tap*Cin + ci] = torch_w[n][ci][tap].
ConvTranspose1d(Cin, Cout, k, stride u, padding p) is exported per output phase r (0..u-1)
as a stride-1 conv over the input:  W[r][co][m*Cin + ci] = torch_w[ci][co][r + u*m]
(m = 0..ceil(k/u)-1, zero where r + u*m >= k); output row t = q*u + r - p takes input rows
q - m.

Weights file tensors (fp32 unless noted):
  enc.emb [6561][512]                       flow.input_embedding
  enc.pe  [9999][512]                       EspnetRelPositionalEncoding table; row m <-> relative
                                            position (4999 - m) (identical to the module's pe)
  enc.embed.{w [512][512], b, ln.g, ln.b}   encoder.embed (Linear + LayerNorm eps 1e-5), x*sqrt(512)
  enc.la1.{w [512][4*512], b}  enc.la2.{w [512][3*512], b}     pre_lookahead_layer conv1/conv2
  enc.L{i}.* i=0..9 (0..5 = encoders, 6..9 = up_encoders), conformer layer (LN eps 1e-12):
      ln_mha.{g,b}, qkv.{w [1536][512], b [1536]} (rows q|k|v), pos.w [512][512],
      pos_u [8][64], pos_v [8][64], out.{w [512][512], b}, ln_ff.{g,b},
      ff1.{w [2048][512], b}, ff2.{w [512][2048], b}          (FFN activation: SiLU)
  enc.up.{w [512][5*512], b}                up_layer (nearest x2, left pad 4, conv k5)
  enc.up_embed.{w, b, ln.g, ln.b}           as enc.embed
  enc.after_ln.{g, b}                       after_norm (eps 1e-5)
  enc.proj.{w [80][512], b [80]}            flow.encoder_proj
  spk.{w [80][192], b [80]}                 flow.spk_embed_affine_layer (applied to normalize(emb))
  cfm.t_span [11]                           cosine schedule, fp32 exactly as torch computes it
  cfm.tvec [10][14][256]                    mlp(time_mlp(time_embeddings(t_k))) per Euler step k and
                                            ResNet block (0 = down, 1..12 = mid, 13 = up)
  cfm.r0.{c1x.w [256][3*80], c1c.w [256][3*240], c1.b, rx.w [256][80], rc.w [256][240], r.b}
        first ResNet split into the x channels (0..79) and the per-utterance constant context
        [mu | spks | cond] (80..319)
  cfm.r{j}.{c1.w [256][3*Cin], c1.b, r.w [256][Cin], r.b}   j = 1..13 (Cin 256, 512 for j = 13)
  cfm.r{j}.{ln1.g, ln1.b, c2.w [256][3*256], c2.b, ln2.g, ln2.b}   j = 0..13
  cfm.tb{k}.* k=0..55 (4 per block: down, mid0..11, up), BasicTransformerBlock (LN eps 1e-5):
      ln1.{g,b}, qkv.w [1536][256] (no bias), out.{w [256][512], b}, ln3.{g,b},
      ff1.{w [1024][256], b} (GELU, erf), ff2.{w [256][1024], b}
  cfm.down.{w [256][3*256], b}  cfm.upc.{w, b}   causal k3 convs after the down / up block
  cfm.fin.{w [256][3*256], b, ln.g, ln.b}  cfm.proj.{w [80][256], b}   final_block + final_proj
  hift.f0.c{i}.{w [512][3*Cin], b} i=0..4 (Cin 80, 512...), hift.f0.cls.{w [1][512], b [1]}
  hift.src.{w [9], b [1]}                   m_source.l_linear (9 harmonics -> 1)
  hift.pre.{w [512][7*80], b}
  hift.up{i}.{w [u][Cout][m*Cin], b [Cout]} i=0..2, (u, k) = (8,16), (5,11), (3,7)
  hift.sd{i}.{w [Cout][k*20], b}            source_downs; STFT input channels padded 18 -> 20
                                            (channels 18, 19 zero); (k, stride, pad) = (30,15,7),
                                            (6,3,1), (1,1,0)
  hift.sr{i}.d{j}.{a1, c1.w [C][k*C], c1.b, a2, c2.w [C][k*C], c2.b}   source_resblocks, j = 0..2
  hift.rb{i}.d{j}.{...}                     resblocks (i = stage*3 + kernel idx), as above
  hift.post.{w [18][7*64], b [18]}
  hift.window [16]                          periodic Hann

Voice file tensors:
  prompt_token [P] int32, prompt_feat [Pf][80] fp32 (24 kHz log-mel), embedding [192] fp32
  (x-vector, raw; the library applies normalize + spk affine).
"""
import argparse
import glob
import math
import os
import struct

import numpy as np
import torch


def write_blob(path, magic, tensors):
    hdr = bytearray(magic + struct.pack("<II", 1, len(tensors)))
    hlen = len(hdr) + sum(4 + len(n) + 4 + 4 + 4 * a.ndim + 16 for n, a in tensors.items())
    off = (hlen + 63) // 64 * 64
    layout = []
    for n, a in tensors.items():
        dt = 1 if a.dtype == np.int32 else 0
        hdr += struct.pack("<I", len(n)) + n.encode() + struct.pack("<II", dt, a.ndim)
        hdr += struct.pack(f"<{a.ndim}I", *a.shape) + struct.pack("<QQ", off, a.nbytes)
        layout.append((off, a))
        off = (off + a.nbytes + 63) // 64 * 64
    assert len(hdr) == hlen
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        for o, a in layout:
            f.write(b"\0" * (o - f.tell()))
            f.write(a.tobytes())
    print(f"wrote {path}: {len(tensors)} tensors, {off} bytes")


def snapshot():
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    c = glob.glob(os.path.join(hf, "hub/models--ResembleAI--chatterbox/snapshots/*/"))
    if not c:
        raise SystemExit("ResembleAI/chatterbox not found under $HF_HOME")
    return c[0]


def load_s3gen():
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from safetensors.torch import load_file
    from chatterbox.models.s3gen import S3Gen
    m = S3Gen()
    m.load_state_dict(load_file(os.path.join(snapshot(), "s3gen.safetensors")), strict=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--voice-out")
    ap.add_argument("--conds", help="conds.pt (default: the builtin voice of the snapshot)")
    args = ap.parse_args()

    T = {}

    def put(name, t):
        a = t.detach().double().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float64)
        T[name] = np.ascontiguousarray(a.astype("<f4"))

    def conv(t):  # Conv1d weight [Cout][Cin][k] -> [Cout][k*Cin]
        w = t.detach().double()
        if w.dim() == 2:
            return w
        return w.permute(0, 2, 1).reshape(w.shape[0], -1)

    def convT(t, u):  # ConvTranspose1d weight [Cin][Cout][k] -> [u][Cout][m*Cin]
        w = t.detach().double()
        ci, co, k = w.shape
        m = (k + u - 1) // u
        out = torch.zeros(u, co, m, ci, dtype=torch.float64)
        for r in range(u):
            for j in range(m):
                if r + u * j < k:
                    out[r, :, j, :] = w[:, :, r + u * j].t()
        return out.reshape(u, co, m * ci)

    m = load_s3gen()
    fl = m.flow
    enc = fl.encoder
    est = fl.decoder.estimator
    h = m.mel2wav
    with torch.no_grad():
        # ---------------- encoder ----------------
        put("enc.emb", fl.input_embedding.weight)
        put("enc.pe", enc.embed.pos_enc.pe[0])
        assert torch.equal(enc.embed.pos_enc.pe, enc.up_embed.pos_enc.pe)
        for nm, e in (("enc.embed", enc.embed), ("enc.up_embed", enc.up_embed)):
            put(nm + ".w", e.out[0].weight)
            put(nm + ".b", e.out[0].bias)
            put(nm + ".ln.g", e.out[1].weight)
            put(nm + ".ln.b", e.out[1].bias)
            assert e.out[1].eps == 1e-5
        la = enc.pre_lookahead_layer
        put("enc.la1.w", conv(la.conv1.weight))
        put("enc.la1.b", la.conv1.bias)
        put("enc.la2.w", conv(la.conv2.weight))
        put("enc.la2.b", la.conv2.bias)
        layers = list(enc.encoders) + list(enc.up_encoders)
        assert len(layers) == 10
        for i, L in enumerate(layers):
            p = f"enc.L{i}."
            a = L.self_attn
            put(p + "ln_mha.g", L.norm_mha.weight)
            put(p + "ln_mha.b", L.norm_mha.bias)
            put(p + "qkv.w", torch.cat([a.linear_q.weight, a.linear_k.weight, a.linear_v.weight], 0))
            put(p + "qkv.b", torch.cat([a.linear_q.bias, a.linear_k.bias, a.linear_v.bias], 0))
            put(p + "pos.w", a.linear_pos.weight)
            put(p + "pos_u", a.pos_bias_u)
            put(p + "pos_v", a.pos_bias_v)
            put(p + "out.w", a.linear_out.weight)
            put(p + "out.b", a.linear_out.bias)
            put(p + "ln_ff.g", L.norm_ff.weight)
            put(p + "ln_ff.b", L.norm_ff.bias)
            put(p + "ff1.w", L.feed_forward.w_1.weight)
            put(p + "ff1.b", L.feed_forward.w_1.bias)
            put(p + "ff2.w", L.feed_forward.w_2.weight)
            put(p + "ff2.b", L.feed_forward.w_2.bias)
            assert L.feed_forward_macaron is None and L.conv_module is None
        put("enc.up.w", conv(enc.up_layer.conv.weight))
        put("enc.up.b", enc.up_layer.conv.bias)
        put("enc.after_ln.g", enc.after_norm.weight)
        put("enc.after_ln.b", enc.after_norm.bias)
        put("enc.proj.w", fl.encoder_proj.weight)
        put("enc.proj.b", fl.encoder_proj.bias)
        put("spk.w", fl.spk_embed_affine_layer.weight)
        put("spk.b", fl.spk_embed_affine_layer.bias)

        # ---------------- CFM estimator ----------------
        dec = fl.decoder
        assert abs(dec.inference_cfg_rate - 0.7) < 1e-9 and dec.t_scheduler == "cosine"
        n_steps = 10
        t_span = torch.linspace(0, 1, n_steps + 1, dtype=torch.float32)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        T["cfm.t_span"] = t_span.numpy().astype("<f4")
        resnets = [est.down_blocks[0][0]] + [b[0] for b in est.mid_blocks] + [est.up_blocks[0][0]]
        tblocks = list(est.down_blocks[0][1]) + [t for b in est.mid_blocks for t in b[1]] + list(est.up_blocks[0][1])
        assert len(resnets) == 14 and len(tblocks) == 56
        tv = torch.zeros(n_steps, 14, 256, dtype=torch.float32)
        for k in range(n_steps):
            t = t_span[k].unsqueeze(0)
            te = est.time_mlp(est.time_embeddings(t).to(t.dtype))
            for j, r in enumerate(resnets):
                tv[k, j] = r.mlp(te)[0]
        T["cfm.tvec"] = tv.numpy().astype("<f4")
        for j, r in enumerate(resnets):
            p = f"cfm.r{j}."
            c1 = r.block1.block[0]
            w1 = c1.weight.double()  # [256][Cin][3]
            wr = r.res_conv.weight.double()[:, :, 0]
            if j == 0:
                put(p + "c1x.w", conv(w1[:, :80]))
                put(p + "c1c.w", conv(w1[:, 80:]))
                put(p + "rx.w", wr[:, :80])
                put(p + "rc.w", wr[:, 80:])
            else:
                put(p + "c1.w", conv(w1))
                put(p + "r.w", wr)
            put(p + "c1.b", c1.bias)
            put(p + "r.b", r.res_conv.bias)
            put(p + "ln1.g", r.block1.block[2].weight)
            put(p + "ln1.b", r.block1.block[2].bias)
            put(p + "c2.w", conv(r.block2.block[0].weight))
            put(p + "c2.b", r.block2.block[0].bias)
            put(p + "ln2.g", r.block2.block[2].weight)
            put(p + "ln2.b", r.block2.block[2].bias)
        for k, tb in enumerate(tblocks):
            p = f"cfm.tb{k}."
            a = tb.attn1
            assert a.heads == 8 and tb.attn2 is None
            put(p + "ln1.g", tb.norm1.weight)
            put(p + "ln1.b", tb.norm1.bias)
            put(p + "qkv.w", torch.cat([a.to_q.weight, a.to_k.weight, a.to_v.weight], 0))
            assert a.to_q.bias is None
            put(p + "out.w", a.to_out[0].weight)
            put(p + "out.b", a.to_out[0].bias)
            put(p + "ln3.g", tb.norm3.weight)
            put(p + "ln3.b", tb.norm3.bias)
            put(p + "ff1.w", tb.ff.net[0].proj.weight)
            put(p + "ff1.b", tb.ff.net[0].proj.bias)
            assert tb.ff.net[0].approximate == "none"
            put(p + "ff2.w", tb.ff.net[2].weight)
            put(p + "ff2.b", tb.ff.net[2].bias)
        put("cfm.down.w", conv(est.down_blocks[0][2].weight))
        put("cfm.down.b", est.down_blocks[0][2].bias)
        put("cfm.upc.w", conv(est.up_blocks[0][2].weight))
        put("cfm.upc.b", est.up_blocks[0][2].bias)
        put("cfm.fin.w", conv(est.final_block.block[0].weight))
        put("cfm.fin.b", est.final_block.block[0].bias)
        put("cfm.fin.ln.g", est.final_block.block[2].weight)
        put("cfm.fin.ln.b", est.final_block.block[2].bias)
        put("cfm.proj.w", conv(est.final_proj.weight))
        put("cfm.proj.b", est.final_proj.bias)

        # ---------------- HiFT ----------------
        f0p = h.f0_predictor
        for i in range(5):
            c = f0p.condnet[2 * i]
            put(f"hift.f0.c{i}.w", conv(c.weight))
            put(f"hift.f0.c{i}.b", c.bias)
        put("hift.f0.cls.w", f0p.classifier.weight)
        put("hift.f0.cls.b", f0p.classifier.bias)
        put("hift.src.w", h.m_source.l_linear.weight[0])
        put("hift.src.b", h.m_source.l_linear.bias)
        put("hift.pre.w", conv(h.conv_pre.weight))
        put("hift.pre.b", h.conv_pre.bias)
        for i, (u, k) in enumerate(((8, 16), (5, 11), (3, 7))):
            up = h.ups[i]
            assert up.stride[0] == u and up.kernel_size[0] == k and up.padding[0] == (k - u) // 2
            put(f"hift.up{i}.w", convT(up.weight, u))
            put(f"hift.up{i}.b", up.bias)
        for i, sd in enumerate(h.source_downs):
            w = sd.weight.double()  # [Cout][18][k]
            wp = torch.zeros(w.shape[0], 20, w.shape[2], dtype=torch.float64)
            wp[:, :18] = w
            put(f"hift.sd{i}.w", conv(wp))
            put(f"hift.sd{i}.b", sd.bias)

        def resblock(p, rb):
            for j in range(3):
                q = f"{p}.d{j}."
                put(q + "a1", rb.activations1[j].alpha)
                put(q + "c1.w", conv(rb.convs1[j].weight))
                put(q + "c1.b", rb.convs1[j].bias)
                put(q + "a2", rb.activations2[j].alpha)
                put(q + "c2.w", conv(rb.convs2[j].weight))
                put(q + "c2.b", rb.convs2[j].bias)
                assert not rb.activations1[j].alpha_logscale

        for i, rb in enumerate(h.source_resblocks):
            resblock(f"hift.sr{i}", rb)
        for i, rb in enumerate(h.resblocks):
            resblock(f"hift.rb{i}", rb)
        put("hift.post.w", conv(h.conv_post.weight))
        put("hift.post.b", h.conv_post.bias)
        put("hift.window", h.stft_window)
        assert h.m_source.l_sin_gen.harmonic_num == 8 and h.m_source.l_sin_gen.voiced_threshold == 10
    write_blob(args.out, b"S3GENW01", T)

    if args.voice_out:
        cp = args.conds or os.path.join(snapshot(), "conds.pt")
        c = torch.load(cp, map_location="cpu", weights_only=True)
        g = c["gen"]
        V = {
            "prompt_token": np.ascontiguousarray(g["prompt_token"].detach().reshape(-1).numpy().astype("<i4")),
            "prompt_feat": np.ascontiguousarray(g["prompt_feat"].detach().reshape(-1, 80).float().numpy().astype("<f4")),
            "embedding": np.ascontiguousarray(g["embedding"].detach().reshape(-1).float().numpy().astype("<f4")),
        }
        assert V["embedding"].shape == (192,)
        write_blob(args.voice_out, b"S3GENV01", V)


if __name__ == "__main__":
    main()
