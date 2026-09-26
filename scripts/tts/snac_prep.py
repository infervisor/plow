#!/usr/bin/env python3
"""Export SNAC-24kHz (hubertsiuzdak/snac_24khz) decoder weights for libplow_snac.so.

Usage:
  HF_HOME=/root/tts-work/hf /root/tts-work/venv-ref/bin/python scripts/tts/snac_prep.py out.bin

Weight norm is folded (w = g * v / ||v||, via the parametrization's computed
.weight). Only the decode path is exported. The quantizer's codebook lookup and
out_proj 1x1 conv (with bias) are pre-multiplied into per-level tables.

File format (all little-endian):
  offset 0   : 8 bytes  magic "SNAC24K1"
  offset 8   : u32      format version (=2)
  offset 12  : u32      tensor count N
  then N records:
      u32 name_len, name bytes (utf-8, no NUL)
      u32 ndim, u32 dims[ndim]
      u64 byte offset of the tensor data from the START OF THE FILE (64-byte aligned)
      u64 byte size (= 4 * prod(dims))
  then the data: each tensor a contiguous fp32 C-order array at its offset.

Tensors (fp32; C = channels at that stage, K = 2*stride):
  q.P{0,1,2}          [4096][768]    codebook_l @ out_proj_l^T + bias_l  (row = code id)
  pre.dw.w [768][7], pre.dw.b [768]  depthwise k7 conv on the latent
  pre.pw.w [1024][768], pre.pw.b [1024]   1x1 conv (torch [Cout][Cin] order)
  blk{i}.snake [Cin]                 i = 0..3
  blk{i}.up.w  [s][Cout][2*Cin]      ConvTranspose1d weight split by output phase r:
                                     up.w[r][co][h*Cin+ci] = torch_w[ci][co][r + h*s]
  blk{i}.up.b  [Cout]
  blk{i}.noise.w [Cout][Cout]        NoiseBlock 1x1 (no bias)
  blk{i}.ru{j}.a1 [C], .dw.w [C][7], .dw.b [C], .a2 [C], .pw.w [C][C], .pw.b [C]   j = 0..2
  out.snake [64], out.w [64][7], out.b [1]
"""
import struct
import sys

import numpy as np
import torch


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    from snac import SNAC

    m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    T = {}

    def put(name, t):
        T[name] = np.ascontiguousarray(t.detach().double().cpu().numpy().astype(np.float32))

    with torch.no_grad():
        for l, q in enumerate(m.quantizer.quantizers):
            cb = q.codebook.weight.double()  # [4096, 8]
            w = q.out_proj.weight.double()[:, :, 0]  # [768, 8]
            b = q.out_proj.bias.double()
            put(f"q.P{l}", cb @ w.t() + b[None, :])
        d = m.decoder.model
        put("pre.dw.w", d[0].weight[:, 0, :])
        put("pre.dw.b", d[0].bias)
        put("pre.pw.w", d[1].weight[:, :, 0])
        put("pre.pw.b", d[1].bias)
        for i in range(4):
            blk = d[2 + i].block
            put(f"blk{i}.snake", blk[0].alpha.reshape(-1))
            w = blk[1].weight  # [Cin, Cout, 2s]
            ci, co, k = w.shape
            s = k // 2
            put(f"blk{i}.up.w", w.reshape(ci, co, 2, s).permute(3, 1, 2, 0).reshape(s, co, 2 * ci))
            put(f"blk{i}.up.b", blk[1].bias)
            put(f"blk{i}.noise.w", blk[2].linear.weight[:, :, 0])
            for j in range(3):
                ru = blk[3 + j].block
                p = f"blk{i}.ru{j}"
                put(p + ".a1", ru[0].alpha.reshape(-1))
                put(p + ".dw.w", ru[1].weight[:, 0, :])
                put(p + ".dw.b", ru[1].bias)
                put(p + ".a2", ru[2].alpha.reshape(-1))
                put(p + ".pw.w", ru[3].weight[:, :, 0])
                put(p + ".pw.b", ru[3].bias)
        put("out.snake", d[6].alpha.reshape(-1))
        put("out.w", d[7].weight[0])
        put("out.b", d[7].bias)

    hdr = bytearray(b"SNAC24K1" + struct.pack("<II", 2, len(T)))
    hlen = len(hdr) + sum(4 + len(n) + 4 + 4 * a.ndim + 16 for n, a in T.items())
    off = (hlen + 63) // 64 * 64
    layout = []
    for n, a in T.items():
        hdr += struct.pack("<I", len(n)) + n.encode() + struct.pack("<I", a.ndim)
        hdr += struct.pack(f"<{a.ndim}I", *a.shape) + struct.pack("<QQ", off, a.nbytes)
        layout.append((off, a))
        off = (off + a.nbytes + 63) // 64 * 64
    assert len(hdr) == hlen
    with open(sys.argv[1], "wb") as f:
        f.write(hdr)
        for o, a in layout:
            f.write(b"\0" * (o - f.tell()))
            f.write(a.astype("<f4").tobytes())
    print(f"wrote {sys.argv[1]}: {len(T)} tensors, {off} bytes")


if __name__ == "__main__":
    main()
