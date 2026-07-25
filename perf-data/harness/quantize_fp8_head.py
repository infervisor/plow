#!/usr/bin/env python3
"""Quantize ONLY the tied embed/lm_head tensor to the per-output-channel (per-vocab-row)
e4m3 twin the sm_120 decode path expects for PLOW_FP8_HEAD=1 (rtx-19 E5). Emitted as a
SEPARATE safetensors file so the stock projection twin (fp8/model.safetensors) is untouched
— drop it in the same checkpoint directory and the loader mmaps both.

Method matches quantize_fp8.py exactly (per-row amax/448 e4m3, f32 scale). The lm_head GEMV
is logits = x @ W^T with W = embed_tokens [vocab, hidden]; row n = output logit n, so the
per-row scale is per-vocab-token — identical convention to the projection twins.

Usage: quantize_fp8_head.py <src-model-dir> <out.safetensors> [prefix]
  prefix default "model.language_model." (Gemma-4 multimodal re-export).
"""
import sys, os, struct, json, mmap, ctypes
import torch

E4M3_MAX = 448.0
ROW_CHUNK = 4096


def raw_bytes(t):
    t = t.contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())


def read_header(path):
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hn))
    return hdr, 8 + hn


def find_source(src_dir, name):
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            wm = json.load(f)["weight_map"]
        fn = wm[name]
    else:
        fn = "model.safetensors"
    path = os.path.join(src_dir, fn)
    hdr, data0 = read_header(path)
    if name not in hdr:
        raise KeyError(f"{name} not in {path}")
    return path, hdr, data0


def main():
    src_dir, out_path = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "model.language_model."
    name = f"{prefix}embed_tokens.weight"
    path, hdr, data0 = find_source(src_dir, name)
    shape = list(hdr[name]["shape"])           # [vocab, hidden]
    N, K = shape[0], shape[1]
    a, e = hdr[name]["data_offsets"]
    if e - a != N * K * 2:
        raise ValueError(f"{name}: expected {N*K*2} BF16 bytes, got {e-a}")
    print(f"src {path}: {name} [{N},{K}] bf16 -> per-row e4m3", flush=True)

    wname = f"fp8/{name}"
    sname = f"fp8/{name}_scale"
    meta = {
        wname: {"dtype": "F8_E4M3", "shape": shape, "data_offsets": [0, N * K]},
        sname: {"dtype": "F32", "shape": [N], "data_offsets": [N * K, N * K + N * 4]},
    }
    hdr_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    hdr_bytes += b" " * ((-len(hdr_bytes)) % 8)

    backing = open(path, "rb")
    mm = mmap.mmap(backing.fileno(), 0, prot=mmap.PROT_READ)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scale_parts = []
    with open(out_path, "wb") as o:
        o.write(struct.pack("<Q", len(hdr_bytes)))
        o.write(hdr_bytes)
        for row0 in range(0, N, ROW_CHUNK):
            nr = min(ROW_CHUNK, N - row0)
            b0 = data0 + a + row0 * K * 2
            raw = bytearray(mm[b0:b0 + nr * K * 2])
            w = torch.frombuffer(raw, dtype=torch.bfloat16).view(nr, K).float()
            amax = w.abs().amax(dim=1)
            scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
            q = (w / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            o.write(raw_bytes(q.view(torch.uint8)))
            scale_parts.append(raw_bytes(scale.to(torch.float32)))
        for sb in scale_parts:
            o.write(sb)
    mm.close(); backing.close()
    print(f"done: {out_path}  ({(N*K + N*4)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
