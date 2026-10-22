#!/usr/bin/env python3
"""Export Chatterbox's voice conditioning (prepare_conditionals) for plowc's voice_cond.pkt:
24 kHz reference PCM -> T3 speaker embedding + prompt speech tokens, s3gen prompt tokens / mels /
speaker vector, T3CondEnc rows.

Front end (every per-frame transform before |X|^2 is linear, so each STFT is one strided conv
against a packet basis, the device pattern of crates/devgen/src/vad.rs):
  fe.rs16.w     [2][1][K]     torchaudio Resample(24000 -> 16000) sinc taps as a stride-3 conv
                              with 2 output channels (= the interleaved 16 kHz stream)
  fe.rs16s.w    [2][1][K]     the same shape approximating librosa soxr_hq (T3 / VE path)
  fe.stft400.w  [402][1][400] periodic hann * [cos; -sin] (16 kHz, n_fft 400)
  fe.mel40      [40][201]     VoiceEncoder power mel (slaney)
  fe.mel128     [128][201]    S3Tokenizer log-mel filters
  fe.stft1920.w [1922][1][1920] periodic hann * [cos; -sin] (24 kHz prompt mels)
  fe.mel80      [80][961]     Matcha mel (fmax 8000)
  fe.kaldi.w    [514][1][400] DFT512 . povey . pre-emphasis(0.97) . DC removal (Kaldi fbank)
  fe.kaldimel   [80][257]     Kaldi mel banks (20 Hz .. Nyquist)
  fe.{mel40,mel128,kaldimel}x2  [M | M]: the filterbank over [re^2 | im^2] rows
  fe.pair961    [961][1922]   [I | I]: |X|^2 of the 1920-point STFT

S3Tokenizer V2 encoder (shared by the T3 and s3gen prompt tokens):
  tok.conv{1,2}.{w,b}         k3 s2 p1 convs (GELU after each)
  tok.b{k}.ln{1,2}.{w,b}      pre-attention / pre-MLP LayerNorm (eps 1e-5)
  tok.b{k}.qkv.{w,b}          [q; k; v] [3840][1280], bias [bq; 0; bv]
  tok.b{k}.fsmn.w             depthwise k31 [1280][1][31]
  tok.b{k}.out.{w,b}, tok.b{k}.mlp{1,2}.{w,b}
  tok.fsq.{w,b}               project_down [8][1280]
  tok.fsq.E / tok.fsq.Eb      product codebook [6561][8] (digit_i(c) - 1) and -|E_c|^2 / 2: the
                              argmax of h . E_c + Eb_c is the per-dimension round of h

CAMPPlus x-vector (eval BatchNorms folded; Conv2d in torch layout [cout][cin][freq][time]):
  camp.fcm.c0.{w,b}, camp.fcm.l{1,2}.{0,1}.c{1,2}.{w,b} (+ .sc.{w,b} shortcut), camp.fcm.c2.{w,b}
  camp.tdnn.{w,b}             conv k5 s2 p2 (+BN, ReLU)
  camp.d{blk}.{i}.bn1.{s,t}   pre-activation BN as scale / shift (then ReLU), zero past the
                              layer's input channels up to the block's concat width
  camp.d{blk}.{i}.l1.{w,b}    1x1 (+BN2 folded, ReLU), input zero-padded the same way
  camp.d{blk}.{i}.local.w     k3 dilation d, no bias
  camp.d{blk}.{i}.cam{1,2}.{w,b}  context MLP (ReLU, sigmoid)
  camp.t{blk}.bn.{s,t} / camp.t{blk}.w   transit BN-ReLU, 1x1 no bias
  camp.out.bn.{s,t}           final BN-ReLU
  camp.dense.{w,b}            1x1 (+ non-affine BN folded) -> 192
  spks.{w,b}                  s3gen spk_embed_affine_layer (192 -> 80, after L2 normalization)

VoiceEncoder (T3 speaker embedding):
  ve.l{k}.w [1024][in + 256]  [W_ih | W_hh] per LSTM layer, gates i, f, g, o; ve.l{k}.b = b_ih + b_hh
  ve.proj.{w,b}               256 -> 256, ReLU, L2 norm per partial; mean of partials, L2 norm

T3CondEnc (per T3 checkpoint, --t3):
  cond.spkr.{w,b}             256 -> 1024 (row 0)
  cond.emo.w [1024]           emotion row = exaggeration * w (row 33)
  cond.query [32][1024]       perceiver queries
  cond.ln.{w,b}, cond.{q,k,v,o}.{w,b}, cond.qkv.{w,b} ([q; k; v])   the perceiver's one
                              AttentionBlock2 (4 heads x 256,
                              SDPA scale 1/16), applied as cross- then self-attention
  cond.speech_emb [vocab][1024], cond.speech_pos [150][1024]   prompt token embedding + position

  HF_HOME=... venv-cbx/bin/python scripts/tts/chatterbox_voice_export.py OUT_DIR [--golden DIR]

--golden replays every front-end basis the way the device convs run it against cbx_golden.py's
arrays and prints rel-L2 (gates G1-G5 of docs: voice-cond plan).
"""
import argparse
import json
import math
import os

import numpy as np
import torch

S3GEN_SR, S3_SR = 24000, 16000


def hann(n, periodic=True):
    return torch.hann_window(n, periodic=periodic, dtype=torch.float64).numpy()


def dft_basis(n_fft, window, width=None):
    """[2 * (n_fft/2 + 1)][width] rows re | im of rfft(window * frame) over `width` taps (frames
    zero-padded to n_fft when width < n_fft)."""
    width = width or n_fft
    k = np.arange(n_fft // 2 + 1)[:, None]
    t = np.arange(width)[None, :]
    ang = 2 * np.pi * k * t / n_fft
    return np.concatenate([np.cos(ang) * window, -np.sin(ang) * window], 0)


def kaldi_basis():
    """Kaldi fbank's per-frame linear chain (remove DC, pre-emphasis 0.97 with x[-1] = x[0],
    povey window, zero-pad to 512) folded into the DFT: [514][400]."""
    n = 400
    a = np.eye(n) - np.full((n, n), 1.0 / n)
    pre = np.eye(n)
    pre[np.arange(1, n), np.arange(n - 1)] = -0.97
    pre[0, 0] = 1.0 - 0.97
    win = hann(n, periodic=False) ** 0.85
    return dft_basis(512, 1.0, n) @ (win[:, None] * pre @ a)


def sinc_taps():
    """torchaudio Resample(24000, 16000) kernel [2][K] and its left pad."""
    import torchaudio
    r = torchaudio.transforms.Resample(S3GEN_SR, S3_SR)
    k = r.kernel.double().numpy()[:, 0, :]
    return k, int(r.width)


def soxr_taps(width=128, rolloff=0.955, beta=8.0):
    """A 3:2 Kaiser-windowed sinc approximating librosa's soxr_hq (no exact multistage clone;
    parameters swept for the least rel-L2 against librosa on a speech clip: 1.5e-3)."""
    orig, new = 3, 2
    base = min(orig, new) * rolloff
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = (-np.arange(new, dtype=np.float64)[:, None] / new + idx) * base
    w = np.kaiser(2 * width + orig, beta)[None, :]
    k = np.where(t == 0, 1.0, np.sin(np.pi * t) / np.where(t == 0, 1.0, np.pi * t))
    return k * w * base / orig, width


def conv_frames(x, k, stride, pad_l, pad_r, mode="zero"):
    """Device Conv1dF32 with one input channel: out[t][o] = sum_j xpad[t*stride + j] * k[o][j]."""
    if mode == "reflect":
        xp = np.pad(x, (pad_l, pad_r), mode="reflect")
    else:
        xp = np.pad(x, (pad_l, pad_r))
    K = k.shape[1]
    T = (len(xp) - K) // stride + 1
    fr = np.lib.stride_tricks.as_strided(xp, (T, K), (xp.strides[0] * stride, xp.strides[0]))
    return fr @ k.T


def resample16(x, k, width):
    y = conv_frames(x.astype(np.float64), k, 3, width, width + 3)
    n = math.ceil(2 * len(x) / 3)
    return y.reshape(-1)[:n]


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def front_end(T):
    import librosa
    import torchaudio
    rs, width = sinc_taps()
    T["fe.rs16.w"] = rs[:, None, :]
    sx, swidth = soxr_taps()
    T["fe.rs16s.w"] = sx[:, None, :]
    T["fe.stft400.w"] = dft_basis(400, hann(400))[:, None, :]
    T["fe.mel40"] = librosa.filters.mel(sr=S3_SR, n_fft=400, n_mels=40, fmin=0, fmax=8000)
    T["fe.mel128"] = librosa.filters.mel(sr=S3_SR, n_fft=400, n_mels=128)
    T["fe.stft1920.w"] = dft_basis(1920, hann(1920))[:, None, :]
    T["fe.mel80"] = librosa.filters.mel(sr=S3GEN_SR, n_fft=1920, n_mels=80, fmin=0, fmax=8000)
    T["fe.kaldi.w"] = kaldi_basis()[:, None, :]
    banks, _ = torchaudio.compliance.kaldi.get_mel_banks(80, 512, float(S3_SR), 20.0, 0.0, 100.0, -500.0, 1.0)
    T["fe.kaldimel"] = torch.nn.functional.pad(banks, (0, 1)).double().numpy()
    # Power spectra stay [frames][re^2 | im^2]: a duplicated filterbank sums the halves.
    for k in ("mel40", "mel128", "kaldimel"):
        T[f"fe.{k}x2"] = np.concatenate([T[f"fe.{k}"], T[f"fe.{k}"]], 1)
    T["fe.pair961"] = np.concatenate([np.eye(961), np.eye(961)], 1)
    return dict(rs16_width=width, rs16s_width=swidth)


def s3tok(T, s3gen):
    sd = {k: v.double().numpy() for k, v in s3gen.tokenizer.state_dict().items()}
    for c in ("conv1", "conv2"):
        T[f"tok.{c}.w"], T[f"tok.{c}.b"] = sd[f"encoder.{c}.weight"], sd[f"encoder.{c}.bias"]
    nb = len(s3gen.tokenizer.encoder.blocks)
    for k in range(nb):
        p, o = f"encoder.blocks.{k}", f"tok.b{k}"
        a = f"{p}.attn"
        T[f"{o}.ln1.w"], T[f"{o}.ln1.b"] = sd[f"{p}.attn_ln.weight"], sd[f"{p}.attn_ln.bias"]
        T[f"{o}.qkv.w"] = np.concatenate([sd[f"{a}.query.weight"], sd[f"{a}.key.weight"], sd[f"{a}.value.weight"]])
        T[f"{o}.qkv.b"] = np.concatenate([sd[f"{a}.query.bias"], np.zeros(1280), sd[f"{a}.value.bias"]])
        T[f"{o}.fsmn.w"] = sd[f"{a}.fsmn_block.weight"]
        T[f"{o}.out.w"], T[f"{o}.out.b"] = sd[f"{a}.out.weight"], sd[f"{a}.out.bias"]
        T[f"{o}.ln2.w"], T[f"{o}.ln2.b"] = sd[f"{p}.mlp_ln.weight"], sd[f"{p}.mlp_ln.bias"]
        for i, j in ((1, 0), (2, 2)):
            T[f"{o}.mlp{i}.w"], T[f"{o}.mlp{i}.b"] = sd[f"{p}.mlp.{j}.weight"], sd[f"{p}.mlp.{j}.bias"]
    q = "quantizer._codebook.project_down"
    T["tok.fsq.w"], T["tok.fsq.b"] = sd[f"{q}.weight"], sd[f"{q}.bias"]
    codes = np.arange(3 ** 8)
    E = np.stack([(codes // 3 ** i) % 3 for i in range(8)], 1).astype(np.float64) - 1.0
    T["tok.fsq.E"], T["tok.fsq.Eb"] = E, -0.5 * (E * E).sum(1)
    return dict(tok_blocks=nb, tok_heads=20, tok_dim=1280, tok_rope_theta=10000.0, tok_fsq_scale=0.9990000128746033)


def bn_affine(sd, p):
    s = sd[f"{p}.weight"] / np.sqrt(sd[f"{p}.running_var"] + 1e-5) if f"{p}.weight" in sd else \
        1.0 / np.sqrt(sd[f"{p}.running_var"] + 1e-5)
    t = (sd[f"{p}.bias"] if f"{p}.bias" in sd else 0.0) - sd[f"{p}.running_mean"] * s
    return s, t


def fold(w, sd, bn, b=None):
    """Conv weight `w` [cout][...] followed by BatchNorm `bn` -> (w', b')."""
    s, t = bn_affine(sd, bn)
    return w * s.reshape(-1, *([1] * (w.ndim - 1))), (0.0 if b is None else b * s) + t


def camp(T, s3gen):
    sd = {k: v.double().numpy() for k, v in s3gen.speaker_encoder.state_dict().items()}
    h = "head"
    T["camp.fcm.c0.w"], T["camp.fcm.c0.b"] = fold(sd[f"{h}.conv1.weight"], sd, f"{h}.bn1")
    for l in (1, 2):
        for i in (0, 1):
            p, o = f"{h}.layer{l}.{i}", f"camp.fcm.l{l}.{i}"
            T[f"{o}.c1.w"], T[f"{o}.c1.b"] = fold(sd[f"{p}.conv1.weight"], sd, f"{p}.bn1")
            T[f"{o}.c2.w"], T[f"{o}.c2.b"] = fold(sd[f"{p}.conv2.weight"], sd, f"{p}.bn2")
            if f"{p}.shortcut.0.weight" in sd:
                T[f"{o}.sc.w"], T[f"{o}.sc.b"] = fold(sd[f"{p}.shortcut.0.weight"], sd, f"{p}.shortcut.1")
    T["camp.fcm.c2.w"], T["camp.fcm.c2.b"] = fold(sd[f"{h}.conv2.weight"], sd, f"{h}.bn2")
    x = "xvector"
    T["camp.tdnn.w"], T["camp.tdnn.b"] = fold(sd[f"{x}.tdnn.linear.weight"], sd, f"{x}.tdnn.nonlinear.batchnorm")
    layers = []
    for blk in (1, 2, 3):
        n = 0
        while f"{x}.block{blk}.tdnnd{n + 1}.linear1.weight" in sd:
            p, o = f"{x}.block{blk}.tdnnd{n + 1}", f"camp.d{blk}.{n}"
            T[f"{o}.bn1.s"], T[f"{o}.bn1.t"] = bn_affine(sd, f"{p}.nonlinear1.batchnorm")
            T[f"{o}.l1.w"], T[f"{o}.l1.b"] = fold(sd[f"{p}.linear1.weight"], sd, f"{p}.nonlinear2.batchnorm")
            cin = T[f"{o}.l1.w"].shape[1]
            c = f"{p}.cam_layer"
            T[f"{o}.local.w"] = sd[f"{c}.linear_local.weight"]
            for j in (1, 2):
                T[f"{o}.cam{j}.w"], T[f"{o}.cam{j}.b"] = sd[f"{c}.linear{j}.weight"], sd[f"{c}.linear{j}.bias"]
            n += 1
        layers.append(n)
        width = cin + 32
        for i in range(n):
            o = f"camp.d{blk}.{i}"
            pad = width - T[f"{o}.l1.w"].shape[1]
            T[f"{o}.bn1.s"], T[f"{o}.bn1.t"] = (np.pad(T[f"{o}.bn1.{k}"], (0, pad)) for k in "st")
            T[f"{o}.l1.w"] = np.pad(T[f"{o}.l1.w"], ((0, 0), (0, pad), (0, 0)))
        T[f"camp.t{blk}.bn.s"], T[f"camp.t{blk}.bn.t"] = bn_affine(sd, f"{x}.transit{blk}.nonlinear.batchnorm")
        T[f"camp.t{blk}.w"] = sd[f"{x}.transit{blk}.linear.weight"]
    T["camp.out.bn.s"], T["camp.out.bn.t"] = bn_affine(sd, f"{x}.out_nonlinear.batchnorm")
    T["camp.dense.w"], T["camp.dense.b"] = fold(sd[f"{x}.dense.linear.weight"], sd, f"{x}.dense.nonlinear.batchnorm")
    return dict(camp_layers=layers, camp_dilations=[1, 2, 2], camp_growth=32, camp_seg=100)


def camp_ref(T, cfg, fb, G=None):
    """CAMPPlus on a mean-normalized fbank [T][80] (all rows valid) in numpy f64 -> [192]."""
    relu = lambda v: np.maximum(v, 0)

    def conv2(x, w, b, sf):
        """x [C][F][T]; 3x3 (pad 1) or 1x1 (pad 0) conv with stride (sf, 1)."""
        k = w.shape[2]
        pd = k // 2
        xp = np.pad(x, ((0, 0), (pd, pd), (pd, pd)))
        F = (x.shape[1] + 2 * pd - k) // sf + 1
        out = np.zeros((w.shape[0], F, x.shape[2]))
        for i in range(k):
            for j in range(k):
                out += np.einsum("oc,cft->oft", w[:, :, i, j], xp[:, i: i + sf * F: sf, j: j + x.shape[2]])
        return out + b[:, None, None]

    def conv1(x, w, b=0.0, stride=1, dil=1):
        """x [T][C]."""
        k = w.shape[2]
        pd = (k - 1) // 2 * dil
        xp = np.pad(x, ((pd, pd), (0, 0)))
        t = (len(xp) - dil * (k - 1) - 1) // stride + 1
        return sum(xp[j * dil: j * dil + stride * t: stride] @ w[:, :, j].T for j in range(k)) + b

    x = relu(conv2(fb.T[None], T["camp.fcm.c0.w"], T["camp.fcm.c0.b"], 1))
    for l in (1, 2):
        for i in (0, 1):
            o, st = f"camp.fcm.l{l}.{i}", 2 if i == 0 else 1
            y = relu(conv2(x, T[f"{o}.c1.w"], T[f"{o}.c1.b"], st))
            y = conv2(y, T[f"{o}.c2.w"], T[f"{o}.c2.b"], 1)
            sc = conv2(x, T[f"{o}.sc.w"], T[f"{o}.sc.b"], st) if f"{o}.sc.w" in T else x
            x = relu(y + sc)
    x = relu(conv2(x, T["camp.fcm.c2.w"], T["camp.fcm.c2.b"], 2))
    x = x.reshape(-1, x.shape[2]).T                            # [T][320], channel c * 10 + f
    if G is not None:
        print(f"   camp FCM rel {rel(x, G('camp_fcm')[0].T):.2e}")
    x = relu(conv1(x, T["camp.tdnn.w"], T["camp.tdnn.b"], stride=2))
    for blk, (n, d) in enumerate(zip(cfg["camp_layers"], cfg["camp_dilations"]), 1):
        for i in range(n):
            o = lambda s: T[f"camp.d{blk}.{i}.{s}"]
            c = x.shape[1]
            h = relu(x * o("bn1.s")[:c] + o("bn1.t")[:c])
            h = relu(conv1(h, o("l1.w")[:, :c], o("l1.b")))
            y = conv1(h, o("local.w"), dil=d)
            t = len(h)
            seg = np.stack([h[s: s + cfg["camp_seg"]].mean(0) for s in range(0, t, cfg["camp_seg"])])
            ctx = h.mean(0) + seg
            m = 1 / (1 + np.exp(-(relu(ctx @ o("cam1.w")[:, :, 0].T + o("cam1.b")) @ o("cam2.w")[:, :, 0].T + o("cam2.b"))))
            x = np.concatenate([x, y * np.repeat(m, cfg["camp_seg"], 0)[:t]], 1)
        x = relu(x * T[f"camp.t{blk}.bn.s"] + T[f"camp.t{blk}.bn.t"]) @ T[f"camp.t{blk}.w"][:, :, 0].T
        if G is not None:
            print(f"   camp transit{blk} rel {rel(x, G(f'camp_transit{blk}')[0].T):.2e}")
    x = relu(x * T["camp.out.bn.s"] + T["camp.out.bn.t"])
    st = np.concatenate([x.mean(0), x.std(0, ddof=1)])
    return st @ T["camp.dense.w"][:, :, 0].T + T["camp.dense.b"]


def ve(T, vmod):
    sd = {k: v.double().numpy() for k, v in vmod.state_dict().items()}
    n = vmod.lstm.num_layers
    for k in range(n):
        T[f"ve.l{k}.w"] = np.concatenate([sd[f"lstm.weight_ih_l{k}"], sd[f"lstm.weight_hh_l{k}"]], 1)
        T[f"ve.l{k}.b"] = sd[f"lstm.bias_ih_l{k}"] + sd[f"lstm.bias_hh_l{k}"]
    T["ve.proj.w"], T["ve.proj.b"] = sd["proj.weight"], sd["proj.bias"]
    return dict(ve_layers=n, ve_hidden=vmod.lstm.hidden_size, ve_partial_frames=160, ve_frame_step=77,
                ve_trim_top_db=20.0, ve_trim_frame=2048, ve_trim_hop=512)


def ve_ref(T, cfg, partials):
    """partials [n][160][40] -> speaker embedding [256]."""
    sig = lambda v: 1 / (1 + np.exp(-v))
    x = partials
    H = cfg["ve_hidden"]
    for k in range(cfg["ve_layers"]):
        w, b = T[f"ve.l{k}.w"], T[f"ve.l{k}.b"]
        h, c = np.zeros((len(x), H)), np.zeros((len(x), H))
        out = []
        for t in range(x.shape[1]):
            g = np.concatenate([x[:, t], h], 1) @ w.T + b
            i, f, gg, o = sig(g[:, :H]), sig(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]), sig(g[:, 3 * H:])
            c = f * c + i * gg
            h = o * np.tanh(c)
            out.append(h)
        x = np.stack(out, 1)
    e = np.maximum(h @ T["ve.proj.w"].T + T["ve.proj.b"], 0)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    e = e.mean(0)
    return e / np.linalg.norm(e)


def cond(T, t3, s3gen):
    sd = {k: v.double().numpy() for k, v in t3.state_dict().items()}
    c = "cond_enc"
    T["cond.spkr.w"], T["cond.spkr.b"] = sd[f"{c}.spkr_enc.weight"], sd[f"{c}.spkr_enc.bias"]
    T["cond.emo.w"] = sd[f"{c}.emotion_adv_fc.weight"][:, 0]
    a = f"{c}.perceiver.attn"
    T["cond.query"] = sd[f"{c}.perceiver.pre_attention_query"][0]
    T["cond.ln.w"], T["cond.ln.b"] = sd[f"{a}.norm.weight"], sd[f"{a}.norm.bias"]
    for o, n in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "proj_out")):
        T[f"cond.{o}.w"], T[f"cond.{o}.b"] = sd[f"{a}.{n}.weight"], sd[f"{a}.{n}.bias"]
    T["cond.qkv.w"] = np.concatenate([T[f"cond.{o}.w"] for o in "qkv"])
    T["cond.qkv.b"] = np.concatenate([T[f"cond.{o}.b"] for o in "qkv"])
    plen = t3.hp.speech_cond_prompt_len
    T["cond.speech_emb"] = sd["speech_emb.weight"]
    T["cond.speech_pos"] = sd["speech_pos_emb.emb.weight"][:plen]
    aff = {k: v.double().numpy() for k, v in s3gen.flow.spk_embed_affine_layer.state_dict().items()}
    T["spks.w"], T["spks.b"] = aff["weight"], aff["bias"]
    return dict(cond_heads=t3.cond_enc.perceiver.attn.num_heads, cond_prompt_tokens=plen,
                cond_rows=1 + T["cond.query"].shape[0] + 1, speech_vocab=T["cond.speech_emb"].shape[0])


def cond_ref(T, cfg, spk, tokens, exaggeration):
    def ln(x):
        m = x.mean(1, keepdims=True)
        return (x - m) / np.sqrt(((x - m) ** 2).mean(1, keepdims=True) + 1e-5) * T["cond.ln.w"] + T["cond.ln.b"]

    def block(x1, x2):
        h = cfg["cond_heads"]
        q, k, v = (ln(x) @ T[f"cond.{n}.w"].T + T[f"cond.{n}.b"] for x, n in ((x1, "q"), (x2, "k"), (x2, "v")))
        d = q.shape[1] // h
        out = []
        for i in range(h):
            sl = slice(i * d, (i + 1) * d)
            sc = q[:, sl] @ k[:, sl].T / np.sqrt(d)
            sc = np.exp(sc - sc.max(1, keepdims=True))
            out.append(sc / sc.sum(1, keepdims=True) @ v[:, sl])
        return x1 + np.concatenate(out, 1) @ T["cond.o.w"].T + T["cond.o.b"]

    e = T["cond.speech_emb"][tokens] + T["cond.speech_pos"][: len(tokens)]
    pre = block(T["cond.query"], e)
    return np.concatenate([(spk @ T["cond.spkr.w"].T + T["cond.spkr.b"])[None], block(pre, pre),
                           exaggeration * T["cond.emo.w"][None]])


def s3tok_ref(T, cfg, mel):
    """The device graph of the S3Tokenizer in numpy f64: mel [T][128] (all rows valid) ->
    hidden [T2][1280], tokens [T2]."""
    from scipy.special import erf
    gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def conv(x, w, b, stride, pad):
        xp = np.pad(x, ((pad, pad), (0, 0)))
        k = w.shape[2]
        t = (len(xp) - k) // stride + 1
        return sum(xp[j: j + stride * t: stride] @ w[:, :, j].T for j in range(k)) + b

    def ln(x, w, b):
        m = x.mean(1, keepdims=True)
        v = ((x - m) ** 2).mean(1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * w + b

    x = gelu(conv(mel, T["tok.conv1.w"], T["tok.conv1.b"], 2, 1))
    x = gelu(conv(x, T["tok.conv2.w"], T["tok.conv2.b"], 2, 1))
    t, d, h = len(x), cfg["tok_dim"], cfg["tok_heads"]
    hd = d // h
    inv = 1.0 / cfg["tok_rope_theta"] ** (np.arange(0, hd, 2) / hd)
    ang = np.arange(t)[:, None] * inv[None]
    cos, sin = np.cos(np.concatenate([ang, ang], 1)), np.sin(np.concatenate([ang, ang], 1))
    rot = lambda q: q * cos[:, None] + np.concatenate([-q[..., hd // 2:], q[..., : hd // 2]], -1) * sin[:, None]
    for k in range(cfg["tok_blocks"]):
        o = lambda s: T[f"tok.b{k}.{s}"]
        a = ln(x, o("ln1.w"), o("ln1.b"))
        qkv = a @ o("qkv.w").T + o("qkv.b")
        q, kk, v = (rot(qkv[:, i * d:(i + 1) * d].reshape(t, h, hd)) if i < 2 else qkv[:, 2 * d:].reshape(t, h, hd)
                    for i in range(3))
        s = np.einsum("qhd,khd->hqk", q, kk) / 8.0
        s = np.exp(s - s.max(-1, keepdims=True))
        att = np.einsum("hqk,khd->qhd", s / s.sum(-1, keepdims=True), v).reshape(t, d)
        vv = v.reshape(t, d)
        vp = np.pad(vv, ((15, 15), (0, 0)))
        fsmn = sum(vp[j: j + t] * o("fsmn.w")[:, 0, j] for j in range(31)) + vv
        x = x + att @ o("out.w").T + o("out.b") + fsmn
        b = ln(x, o("ln2.w"), o("ln2.b"))
        x = x + gelu(b @ o("mlp1.w").T + o("mlp1.b")) @ o("mlp2.w").T + o("mlp2.b")
    z = np.tanh(x @ T["tok.fsq.w"].T + T["tok.fsq.b"]) * cfg["tok_fsq_scale"]
    return x, (z @ T["tok.fsq.E"].T + T["tok.fsq.Eb"]).argmax(1)


def power(spec):
    h = spec.shape[1] // 2
    return spec[:, :h] ** 2 + spec[:, h:] ** 2


def check_front_end(T, cfg, g):
    G = lambda k: np.load(os.path.join(g, k + ".npy"))
    w24 = G("wav24_dec").astype(np.float64)
    w16 = resample16(w24, T["fe.rs16.w"][:, 0], cfg["rs16_width"])
    print(f"G1 torchaudio 24k->16k       rel {rel(w16, G('s3gen_wav16_torchaudio')):.2e}  len {len(w16)}")
    w16s = resample16(G("wav24_full").astype(np.float64), T["fe.rs16s.w"][:, 0], cfg["rs16s_width"])
    print(f"   soxr approximation         rel {rel(w16s, G('wav16_librosa')):.2e}  len {len(w16s)} vs {len(G('wav16_librosa'))}")

    def s3tok_mel(x, frames=None):
        p = power(conv_frames(x, T["fe.stft400.w"][:, 0], 160, 200, 200, "reflect"))[:-1]
        m = p @ T["fe.mel128"].T
        if frames:
            m = m[:frames]
        lg = np.log(np.maximum(m, 1e-10)) / np.log(10)
        lg = np.maximum(lg, lg.max() - 8.0)
        return ((lg + 4.0) / 4.0).T
    print(f"G2 s3gen-path S3Tok log-mel  rel {rel(s3tok_mel(G('s3gen_wav16_torchaudio').astype(np.float64)), G('s3gen_tok_logmel')):.2e}")
    t3m = G("t3_tok_logmel")
    print(f"   T3-path S3Tok log-mel      rel {rel(s3tok_mel(G('wav16_librosa')[:96000].astype(np.float64), t3m.shape[1]), t3m):.2e}")
    p = conv_frames(w24, T["fe.stft1920.w"][:, 0], 480, 720, 720, "reflect")
    mel = np.log(np.maximum(np.sqrt(power(p) + 1e-9) @ T["fe.mel80"].T, 1e-5))
    pf = G("s3gen_prompt_feat")[0]
    print(f"G3 prompt mel                rel {rel(mel, pf):.2e}  rows {len(mel)} vs {len(pf)}")
    x16 = G("s3gen_wav16_torchaudio").astype(np.float64)
    fb = np.log(np.maximum(power(conv_frames(x16, T["fe.kaldi.w"][:, 0], 160, 0, 0)) @ T["fe.kaldimel"].T,
                           np.finfo(np.float32).eps))
    fb -= fb.mean(0, keepdims=True)
    cf = G("campplus_fbank")
    print(f"G4 Kaldi fbank (mean-norm)   rel {rel(fb, cf):.2e}  maxabs {np.abs(fb - cf).max():.2e}  rows {len(fb)} vs {len(cf)}")
    vm = power(conv_frames(G("ve_wav16_trimmed").astype(np.float64), T["fe.stft400.w"][:, 0], 160, 200, 200,
                           "reflect")) @ T["fe.mel40"].T
    print(f"G5 VE power mel              rel {rel(vm, G('ve_mel')):.2e}  rows {len(vm)} vs {len(G('ve_mel'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--golden")
    ap.add_argument("--s3gen", default="s3gen.safetensors")
    ap.add_argument("--t3", default="t3_cfg.safetensors")
    ap.add_argument("--exaggeration", type=float, default=0.5, help="--golden's T3 cond rows")
    ap.add_argument("--revision", default="5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18")
    a = ap.parse_args()
    T = {}
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chatterbox.models.s3gen import S3Gen
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", a.s3gen, revision=a.revision)), strict=False)
    cfg = front_end(T)
    cfg.update(s3tok(T, s3gen))
    cfg.update(camp(T, s3gen))
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    vmod = VoiceEncoder()
    vmod.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors", revision=a.revision)))
    cfg.update(ve(T, vmod))
    t3 = T3(T3Config.multilingual() if "mtl" in a.t3 else T3Config.english_only())
    t3.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", a.t3, revision=a.revision)))
    cfg.update(cond(T, t3, s3gen))
    if a.golden:
        check_front_end(T, cfg, a.golden)
        G = lambda k: np.load(os.path.join(a.golden, k + ".npy"))
        for mel, hid, tok in (("t3_tok_logmel", "t3_tok_hidden", "t3_cond_prompt_speech_tokens"),
                              ("s3gen_tok_logmel", "s3gen_tok_hidden", None)):
            x, ids = s3tok_ref(T, cfg, G(mel).T.astype(np.float64))
            line = f"G6 {mel[:-7]:<10} hidden rel {rel(x, G(hid)):.2e}"
            if tok:
                ref = G(tok).reshape(-1)
                line += f"  tokens {(ids[:len(ref)] == ref).sum()}/{len(ref)}"
            print(line)
        e = camp_ref(T, cfg, G("campplus_fbank").astype(np.float64), G)
        r = G("s3gen_embedding").reshape(-1)
        print(f"G8 CAMPPlus embedding        rel {rel(e, r):.2e}  cos {e @ r / np.linalg.norm(e) / np.linalg.norm(r):.7f}")
        sp = (e / np.linalg.norm(e)) @ T["spks.w"].T + T["spks.b"]
        print(f"   s3gen spks80              rel {rel(sp, G('s3gen_spks80').reshape(-1)):.2e}")
        v = ve_ref(T, cfg, G("ve_partials").astype(np.float64))
        r = G("t3_speaker_emb").reshape(-1)
        print(f"G7 VE speaker embedding      rel {rel(v, r):.2e}  cos {v @ r / np.linalg.norm(r):.7f}")
        rows = cond_ref(T, cfg, r.astype(np.float64), G("t3_cond_prompt_speech_tokens").reshape(-1), a.exaggeration)
        print(f"G9 T3 cond rows              rel {rel(rows, G('t3_cond_rows')):.2e}  rows {len(rows)}")
    from safetensors.torch import save_file
    os.makedirs(a.out, exist_ok=True)
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in T.items()},
              os.path.join(a.out, "model.safetensors"))
    json.dump(dict(model_type="chatterbox_voice_cond", **cfg), open(os.path.join(a.out, "config.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
