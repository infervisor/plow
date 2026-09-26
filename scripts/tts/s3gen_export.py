#!/usr/bin/env python3
"""Export Chatterbox S3Gen (ResembleAI/chatterbox s3gen.safetensors, English, non-meanflow) for
plowc's s3gen.pkt lowering (crates/devgen/src/s3gen.rs, PLOW_TTS_VOCODER_DIR).

Writes OUT_DIR/model.safetensors (fp32, torch layouts, weight norm folded) and OUT_DIR/config.json.

  PYTHONPATH= /opt/dlami/nvme/lava-tts/venv-cbx/bin/python scripts/tts/s3gen_export.py OUT_DIR \
      [--voice NAME=conds.pt ...] [--max-tokens 1000]

Folded / precomputed (all in fp64, stored fp32):
  enc.{embed,up_embed}.ln.{g,b}   LayerNorm affine times the positional xscale sqrt(512)
  enc.L{i}.q_u.b                  linear_q bias + pos_bias_u (the query every score term uses)
  enc.L{i}.pos_vu                 pos_bias_v - pos_bias_u (the positional term's query offset)
  enc.L{i}.relpos [2R-1][512]     linear_pos(pe(r)) / sqrt(64), row m <-> relative position R-1-m;
                                  R = relpos_rows[0] (layers 0..5) or relpos_rows[1] (6..9)
  cfm.tvec [steps][14][256]       ResNet time vectors mlp(time_mlp(time_embeddings(t_k)))
  hift.stft.w [18][1][16]         analysis conv: Hann * (cos | -sin), channels Re 0..8, Im 9..17
  hift.istft.w [18][1][16]        synthesis conv-transpose: Hann * inverse real DFT of (Re | Im)
  hift.env.w [1][1][16]           Hann^2 (the iSTFT window envelope)
  hift.fade [960]                 trim_fade
  voice.prompt_token [V][P] i32, voice.prompt_feat [V][Pf*80], voice.spks [V][80] (spk affine of
  the normalized x-vector). Every voice must have the same P and Pf = 2 P.
"""
import argparse
import glob
import json
import math
import os

import torch


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
    ap.add_argument("--voice", action="append", default=[], help="NAME=conds.pt (default: the builtin voice)")
    ap.add_argument("--max-tokens", type=int, default=1000)
    args = ap.parse_args()
    from safetensors.torch import save_file

    T = {}

    def put(name, t):
        T[name] = torch.as_tensor(t).detach().double().float().contiguous()

    m = load_s3gen()
    fl, enc, h = m.flow, m.flow.encoder, m.mel2wav
    est = fl.decoder.estimator
    voices = args.voice or [f"default={os.path.join(snapshot(), 'conds.pt')}"]
    V = []
    for v in voices:
        name, path = v.split("=", 1)
        g = torch.load(path, map_location="cpu", weights_only=True)["gen"]
        V.append((name, g["prompt_token"].reshape(-1).long(), g["prompt_feat"].reshape(-1, 80).double(),
                  g["embedding"].reshape(-1).double()))
    P = len(V[0][1])
    if any(len(v[1]) != P or v[2].shape[0] != 2 * P for v in V):
        raise SystemExit("every voice needs the same prompt length P and 2 P prompt mel frames")
    rows = [P + args.max_tokens, 2 * (P + args.max_tokens)]

    with torch.no_grad():
        # ---------------- voices ----------------
        T["voice.prompt_token"] = torch.stack([v[1] for v in V]).to(torch.int32).contiguous()
        put("voice.prompt_feat", torch.stack([v[2].reshape(-1) for v in V]))
        sw, sb = fl.spk_embed_affine_layer.weight.double(), fl.spk_embed_affine_layer.bias.double()
        put("voice.spks", torch.stack([sw @ (v[3] / v[3].norm().clamp_min(1e-12)) + sb for v in V]))

        # ---------------- encoder ----------------
        xscale = math.sqrt(512)
        put("emb", fl.input_embedding.weight)
        for nm, e in (("enc.embed", enc.embed), ("enc.up_embed", enc.up_embed)):
            assert e.out[1].eps == 1e-5 and e.pos_enc.xscale == xscale
            put(nm + ".w", e.out[0].weight)
            put(nm + ".b", e.out[0].bias)
            put(nm + ".ln.g", e.out[1].weight.double() * xscale)
            put(nm + ".ln.b", e.out[1].bias.double() * xscale)
        la = enc.pre_lookahead_layer
        put("enc.la1.w", la.conv1.weight)
        put("enc.la1.b", la.conv1.bias)
        put("enc.la2.w", la.conv2.weight)
        put("enc.la2.b", la.conv2.bias)
        pe = enc.embed.pos_enc.pe[0].double()  # row m <-> relative position center - m
        assert torch.equal(enc.embed.pos_enc.pe, enc.up_embed.pos_enc.pe)
        center = (pe.shape[0] - 1) // 2
        layers = list(enc.encoders) + list(enc.up_encoders)
        assert len(enc.encoders) == 6 and len(enc.up_encoders) == 4
        for i, L in enumerate(layers):
            p = f"enc.L{i}."
            a = L.self_attn
            assert a.h == 8 and a.d_k == 64 and L.feed_forward_macaron is None and L.conv_module is None
            assert L.norm_mha.eps == 1e-12 and L.norm_ff.eps == 1e-12 and a.linear_pos.bias is None
            put(p + "ln_mha.g", L.norm_mha.weight)
            put(p + "ln_mha.b", L.norm_mha.bias)
            put(p + "q.w", a.linear_q.weight)
            put(p + "q_u.b", a.linear_q.bias.double() + a.pos_bias_u.double().reshape(-1))
            put(p + "k.w", a.linear_k.weight)
            put(p + "k.b", a.linear_k.bias)
            put(p + "v.w", a.linear_v.weight)
            put(p + "v.b", a.linear_v.bias)
            put(p + "pos_vu", (a.pos_bias_v.double() - a.pos_bias_u.double()).reshape(-1))
            R = rows[0] if i < 6 else rows[1]
            assert R - 1 <= center
            put(p + "relpos", pe[center - (R - 1): center + R] @ a.linear_pos.weight.double().t() / 8.0)
            put(p + "out.w", a.linear_out.weight)
            put(p + "out.b", a.linear_out.bias)
            put(p + "ln_ff.g", L.norm_ff.weight)
            put(p + "ln_ff.b", L.norm_ff.bias)
            put(p + "ff1.w", L.feed_forward.w_1.weight)
            put(p + "ff1.b", L.feed_forward.w_1.bias)
            put(p + "ff2.w", L.feed_forward.w_2.weight)
            put(p + "ff2.b", L.feed_forward.w_2.bias)
        assert enc.up_layer.stride == 2 and enc.up_layer.conv.kernel_size[0] == 5
        put("enc.up.w", enc.up_layer.conv.weight)
        put("enc.up.b", enc.up_layer.conv.bias)
        assert enc.after_norm.eps == 1e-5
        put("enc.after_ln.g", enc.after_norm.weight)
        put("enc.after_ln.b", enc.after_norm.bias)
        put("enc.proj.w", fl.encoder_proj.weight)
        put("enc.proj.b", fl.encoder_proj.bias)

        # ---------------- CFM estimator ----------------
        dec = fl.decoder
        assert dec.t_scheduler == "cosine"
        steps = 10
        t_span = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        resnets = [est.down_blocks[0][0]] + [b[0] for b in est.mid_blocks] + [est.up_blocks[0][0]]
        tblocks = list(est.down_blocks[0][1]) + [t for b in est.mid_blocks for t in b[1]] + list(est.up_blocks[0][1])
        assert len(resnets) == 14 and len(tblocks) == 56
        tv = torch.zeros(steps, 14, 256, dtype=torch.float32)
        for k in range(steps):
            t = t_span[k].unsqueeze(0)
            te = est.time_mlp(est.time_embeddings(t).to(t.dtype))
            for j, r in enumerate(resnets):
                tv[k, j] = r.mlp(te)[0]
        put("cfm.tvec", tv)
        for j, r in enumerate(resnets):
            p = f"cfm.r{j}."
            put(p + "c1.w", r.block1.block[0].weight)
            put(p + "c1.b", r.block1.block[0].bias)
            put(p + "ln1.g", r.block1.block[2].weight)
            put(p + "ln1.b", r.block1.block[2].bias)
            put(p + "c2.w", r.block2.block[0].weight)
            put(p + "c2.b", r.block2.block[0].bias)
            put(p + "ln2.g", r.block2.block[2].weight)
            put(p + "ln2.b", r.block2.block[2].bias)
            put(p + "res.w", r.res_conv.weight)
            put(p + "res.b", r.res_conv.bias)
        for k, tb in enumerate(tblocks):
            p = f"cfm.tb{k}."
            a = tb.attn1
            assert a.heads == 8 and tb.attn2 is None and a.to_q.bias is None
            assert tb.ff.net[0].approximate == "none"
            put(p + "ln1.g", tb.norm1.weight)
            put(p + "ln1.b", tb.norm1.bias)
            put(p + "qkv.w", torch.cat([a.to_q.weight, a.to_k.weight, a.to_v.weight], 0))
            put(p + "out.w", a.to_out[0].weight)
            put(p + "out.b", a.to_out[0].bias)
            put(p + "ln3.g", tb.norm3.weight)
            put(p + "ln3.b", tb.norm3.bias)
            put(p + "ff1.w", tb.ff.net[0].proj.weight)
            put(p + "ff1.b", tb.ff.net[0].proj.bias)
            put(p + "ff2.w", tb.ff.net[2].weight)
            put(p + "ff2.b", tb.ff.net[2].bias)
        put("cfm.down.w", est.down_blocks[0][2].weight)
        put("cfm.down.b", est.down_blocks[0][2].bias)
        put("cfm.upc.w", est.up_blocks[0][2].weight)
        put("cfm.upc.b", est.up_blocks[0][2].bias)
        put("cfm.fin.w", est.final_block.block[0].weight)
        put("cfm.fin.b", est.final_block.block[0].bias)
        put("cfm.fin.ln.g", est.final_block.block[2].weight)
        put("cfm.fin.ln.b", est.final_block.block[2].bias)
        put("cfm.proj.w", est.final_proj.weight)
        put("cfm.proj.b", est.final_proj.bias)

        # ---------------- HiFT ----------------
        f0p = h.f0_predictor
        for i in range(5):
            put(f"hift.f0.c{i}.w", f0p.condnet[2 * i].weight)
            put(f"hift.f0.c{i}.b", f0p.condnet[2 * i].bias)
        put("hift.f0.cls.w", f0p.classifier.weight)
        put("hift.f0.cls.b", f0p.classifier.bias)
        sg = h.m_source.l_sin_gen
        assert sg.harmonic_num == 8 and sg.voiced_threshold == 10 and sg.sampling_rate == 24000
        put("hift.src.w", h.m_source.l_linear.weight)
        put("hift.src.b", h.m_source.l_linear.bias)
        put("hift.harm", torch.arange(1, 10, dtype=torch.float64))
        put("hift.pre.w", h.conv_pre.weight)
        put("hift.pre.b", h.conv_pre.bias)
        ups = []
        for i, up in enumerate(h.ups):
            u, k = up.stride[0], up.kernel_size[0]
            assert up.padding[0] == (k - u) // 2
            ups.append([int(u), int(k), int(up.padding[0])])
            put(f"hift.up{i}.w", up.weight)
            put(f"hift.up{i}.b", up.bias)
        downs = []
        for i, sd in enumerate(h.source_downs):
            downs.append([int(sd.kernel_size[0]), int(sd.stride[0]), int(sd.padding[0])])
            put(f"hift.sd{i}.w", sd.weight)
            put(f"hift.sd{i}.b", sd.bias)

        def resblock(p, rb):
            dil = []
            for j in range(len(rb.convs1)):
                q = f"{p}.d{j}."
                assert not rb.activations1[j].alpha_logscale
                assert rb.convs2[j].dilation[0] == 1
                dil.append(int(rb.convs1[j].dilation[0]))
                put(q + "a1", rb.activations1[j].alpha)
                put(q + "c1.w", rb.convs1[j].weight)
                put(q + "c1.b", rb.convs1[j].bias)
                put(q + "a2", rb.activations2[j].alpha)
                put(q + "c2.w", rb.convs2[j].weight)
                put(q + "c2.b", rb.convs2[j].bias)
            return [int(rb.convs1[0].kernel_size[0]), dil]

        source_rbs = [resblock(f"hift.sr{i}", rb) for i, rb in enumerate(h.source_resblocks)]
        rbs = [resblock(f"hift.rb{i}", rb) for i, rb in enumerate(h.resblocks)]
        put("hift.post.w", h.conv_post.weight)
        put("hift.post.b", h.conv_post.bias)
        n_fft, hop = h.istft_params["n_fft"], h.istft_params["hop_len"]
        assert n_fft == 16 and hop == 4
        win = h.stft_window.double()
        nb = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float64)
        kk = torch.arange(nb, dtype=torch.float64)
        ang = 2 * math.pi * kk[:, None] * n[None, :] / n_fft  # [9][16]
        put("hift.stft.w", torch.cat([win * torch.cos(ang), -win * torch.sin(ang)], 0)[:, None, :])
        # irfft: x[n] = (Re0 + (-1)^n Re8 + 2 sum_{k=1..7} (Re_k cos - Im_k sin)) / N (Im0, Im8 ignored)
        wgt = torch.full((nb,), 2.0, dtype=torch.float64)
        wgt[0] = wgt[-1] = 1.0
        re = wgt[:, None] * torch.cos(ang) / n_fft
        im = -wgt[:, None] * torch.sin(ang) / n_fft
        im[0] = im[-1] = 0.0
        put("hift.istft.w", (torch.cat([re, im], 0) * win[None, :])[:, None, :])
        put("hift.env.w", (win * win)[None, None, :])
        put("hift.fade", m.trim_fade)
        assert h.lrelu_slope == 0.1 and h.audio_limit == 0.99

    os.makedirs(args.out, exist_ok=True)
    save_file(T, os.path.join(args.out, "model.safetensors"))
    config = dict(
        model_type="chatterbox_s3gen", sampling_rate=24000, vocab=int(fl.input_embedding.num_embeddings),
        voices=[v[0] for v in V], prompt_tokens=P, max_tokens=args.max_tokens, relpos_rows=rows,
        t_span=[float(x) for x in t_span], cfg_rate=float(dec.inference_cfg_rate),
        upsample=ups, source_downs=downs, source_resblocks=source_rbs, resblocks=rbs,
        n_fft=n_fft, hop=hop, harmonics=int(sg.harmonic_num) + 1, sine_amp=float(sg.sine_amp),
        noise_std=float(sg.noise_std), voiced_threshold=float(sg.voiced_threshold),
        trim=len(m.trim_fade), audio_limit=float(h.audio_limit))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {args.out}: {len(T)} tensors, P={P}, voices={config['voices']}")


if __name__ == "__main__":
    main()
