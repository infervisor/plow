#!/usr/bin/env python3
"""Export Silero VAD v5 (16 kHz branch) for plowc's VAD lowering (crates/devgen/src/vad.rs).

Reads the TorchScript model of the `silero-vad` pip package (snakers4/silero-vad, MIT) and writes
OUT_DIR/model.safetensors (fp32, torch layouts) and OUT_DIR/config.json (model_type silero_vad,
geometry, source provenance). `plowc --hf-dir OUT_DIR --emit devblob+cubin` then emits vad.pkt.

  python scripts/asr/silero_vad_export.py SILERO_VAD_JIT OUT_DIR
"""
import hashlib
import json
import os
import sys

import torch


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    from safetensors.torch import save_file

    jit, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    m = torch.jit.load(jit, map_location="cpu").eval()
    v = m._model
    T = {}

    def put(name, t):
        T[name] = t.detach().float().contiguous()

    stft = v.stft
    put("stft.basis", stft.forward_basis_buffer)
    encoder = []
    for i in range(4):
        conv = getattr(v.encoder, str(i)).reparam_conv
        put(f"enc{i}.w", conv.weight)
        put(f"enc{i}.b", conv.bias)
        encoder.append(dict(cin=conv.in_channels, cout=conv.out_channels, kernel=conv.kernel_size[0],
                            stride=conv.stride[0], padding=conv.padding[0]))
    rnn = v.decoder.rnn
    put("lstm.wih", rnn.weight_ih)
    put("lstm.whh", rnn.weight_hh)
    put("lstm.bih", rnn.bias_ih)
    put("lstm.bhh", rnn.bias_hh)
    head = getattr(v.decoder.decoder, "2")
    put("head.w", head.weight)
    put("head.b", head.bias)
    save_file(T, os.path.join(out, "model.safetensors"))

    pad = stft.padding.code
    if '[0, 64], "reflect"' not in pad:
        sys.exit(f"unexpected STFT padding: {pad}")
    jit_sha = hashlib.sha256(open(jit, "rb").read()).hexdigest()
    cfg = dict(
        model_type="silero_vad",
        version="5.1.2",
        source="pip:silero-vad==5.1.2 (github.com/snakers4/silero-vad, MIT): silero_vad/data/silero_vad.jit",
        source_sha256=jit_sha,
        sampling_rate=16000,
        frame_samples=512,
        context_samples=int(v.context_size_samples),
        filter_length=int(stft.filter_length),
        hop_length=int(stft.hop_length),
        pad_after=64,
        encoder=encoder,
        lstm_width=int(rnn.weight_hh.shape[1]),
    )
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
