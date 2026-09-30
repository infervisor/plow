#!/usr/bin/env python3
"""Export the SNAC-24kHz decoder (hubertsiuzdak/snac_24khz) for plowc's codec lowering.

Writes OUT_DIR/model.safetensors (fp32, torch layouts, weight norm folded) and OUT_DIR/config.json.
The quantizer's codebook lookup and out_proj 1x1 conv (with bias) are pre-multiplied into per-level
tables q.P{l} [4096][768] (row = code id).

  python scripts/tts/snac_export.py OUT_DIR
"""
import json
import os
import sys

import torch


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    from safetensors.torch import save_file
    from snac import SNAC

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    T = {}

    def put(name, t):
        T[name] = t.detach().double().float().contiguous()

    with torch.no_grad():
        for l, q in enumerate(m.quantizer.quantizers):
            cb = q.codebook.weight.double()
            w = q.out_proj.weight.double()[:, :, 0]
            put(f"q.P{l}", cb @ w.t() + q.out_proj.bias.double()[None, :])
        d = m.decoder.model
        put("pre.dw.w", d[0].weight)
        put("pre.dw.b", d[0].bias)
        put("pre.pw.w", d[1].weight)
        put("pre.pw.b", d[1].bias)
        blocks = []
        for i in range(4):
            blk = d[2 + i].block
            up = blk[1]
            blocks.append(dict(stride=up.stride[0], kernel=up.kernel_size[0], padding=up.padding[0],
                               output_padding=up.output_padding[0], cin=up.in_channels, cout=up.out_channels))
            put(f"blk{i}.snake", blk[0].alpha.reshape(-1))
            put(f"blk{i}.up.w", up.weight)
            put(f"blk{i}.up.b", up.bias)
            put(f"blk{i}.noise.w", blk[2].linear.weight)
            for j in range(3):
                ru = blk[3 + j].block
                p = f"blk{i}.ru{j}"
                put(p + ".a1", ru[0].alpha.reshape(-1))
                put(p + ".dw.w", ru[1].weight)
                put(p + ".dw.b", ru[1].bias)
                put(p + ".a2", ru[2].alpha.reshape(-1))
                put(p + ".pw.w", ru[3].weight)
                put(p + ".pw.b", ru[3].bias)
                blocks[-1].setdefault("dilations", []).append(ru[1].dilation[0])
        put("out.snake", d[6].alpha.reshape(-1))
        put("out.w", d[7].weight)
        put("out.b", d[7].bias)

    save_file(T, os.path.join(out, "model.safetensors"))
    config = dict(model_type="snac", sampling_rate=m.sampling_rate, codebook_size=m.codebook_size,
                  vq_strides=list(m.vq_strides), latent_dim=m.latent_dim if hasattr(m, "latent_dim") else 768,
                  blocks=blocks, frame_codes=7, frame_samples=2048,
                  # Orpheus/Veena 7-code frame: (level, slot) per code position.
                  frame_layout=[[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [2, 2], [2, 3]])
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {out}: {len(T)} tensors")


if __name__ == "__main__":
    main()
