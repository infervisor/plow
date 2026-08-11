#!/usr/bin/env python3
"""Quantize a Gemma-4 bf16 checkpoint to the per-output-channel e4m3 fp8 weight
twins the sm_120 decode path (crates/plowc/src/bin/gemma4.rs, GemvFp8 / GemvGluFp8)
expects.

METHOD (settled — matches the AMD campaign, runtime/amd/op_gemm.h:1440 and
amd_common.h:364): weight-only w8a16, PER-OUTPUT-CHANNEL (per-row) scale.
  W is [N, K] row-major (HF nn.Linear, row n = output channel n).
  scale[n] = amax(|W[n,:]|) / 448          (448 = e4m3fn max finite, PLOW_FP8_E4M3_MAX)
  W8[n,k] = round_e4m3(W[n,k] / scale[n])   (torch.float8_e4m3fn, RTN + saturate)
  dequant  W[n,k] ~= float(W8[n,k]) * scale[n]   (device: scale ONCE in the epilogue)

Output: <out-dir>/model.safetensors, one file, keyed EXACTLY as the emitter declares
the twins — `fp8/<name>` for the uint8 weight, `fp8/<name>_scale` for the f32 scale.
The runtime loader (runtime/common/safetensors.h st_find) ignores the dtype string and
returns the raw byte range by name, so F8_E4M3 / F32 here are documentation only.

The 7 dense projections per layer (q/k/v/o/gate/up/down) are quantized. For Gemma-4
26B-A4B the fused `experts.gate_up_proj [E,2I,H]` and `experts.down_proj [E,H,I]`
are also quantized, with one scale per output row (`[E,2I]` and `[E,H]`). Gemma full
layers (k_eq_v) have no v_proj, so those are skipped — exactly the set the pkt declares.
Router/norm/embedding/lm_head tensors stay bf16. No numpy dependency (torch only).

Both single-file checkpoints (`model.safetensors`) and sharded checkpoints
(`model.safetensors.index.json`) are accepted.

Usage: quantize_fp8.py <src-model-dir> <out-dir> [prefix]
  prefix default "model.language_model." (Gemma-4 multimodal re-export).
"""
import sys, os, struct, json, mmap, ctypes
import torch


def raw_bytes(t):
    """Raw little-endian bytes of a contiguous tensor via a single memcpy.
    (bytes(untyped_storage()) is O(n) in Python and ~2700x slower; numpy is absent.)"""
    t = t.contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())

E4M3_MAX = 448.0
PROJS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
         "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
EXPERT_PROJS = ["experts.gate_up_proj", "experts.down_proj"]
ROW_CHUNK = 1024


def read_src_header(path):
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hn))
    return hdr, 8 + hn


def open_sources(src_dir):
    """Return (weight_map, shard state) for single-file or indexed checkpoints."""
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = "model.safetensors"
        single_path = os.path.join(src_dir, single)
        hdr, _ = read_src_header(single_path)
        weight_map = {name: single for name in hdr if name != "__metadata__"}

    shards = {}
    for filename in sorted(set(weight_map.values())):
        path = os.path.join(src_dir, filename)
        hdr, data0 = read_src_header(path)
        backing = open(path, "rb")
        mapped = mmap.mmap(backing.fileno(), 0, prot=mmap.PROT_READ)
        shards[filename] = (hdr, data0, mapped, backing)
    return weight_map, shards


def build_plan(weight_map, shards, prefix, layers):
    """Return fp8 twins as (out_w, out_s, source, weight_shape, scale_shape, N, K).

    Safetensors preserves the fused experts' 3-D shape for documentation, while the device
    consumes them as N contiguous rows of K elements. The row-scale shape is every leading
    dimension (`shape[:-1]`), which is byte-equivalent to the packet's flat declaration.
    """
    plan = []
    for l in range(layers):
        for proj in PROJS + EXPERT_PROJS:
            name = f"{prefix}layers.{l}.{proj}"
            if name not in weight_map:
                continue
            hdr, _, _, _ = shards[weight_map[name]]
            shape = list(hdr[name]["shape"])
            if len(shape) < 2:
                raise ValueError(f"{name}: expected matrix-like weight, got shape {shape}")
            K = shape[-1]
            N = 1
            for d in shape[:-1]:
                N *= d
            plan.append((f"fp8/{name}", f"fp8/{name}_scale", name,
                         shape, shape[:-1], N, K))
    return plan


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "model.language_model."
    weight_map, shards = open_sources(src_dir)
    layers = 1 + max(int(k.split("layers.")[1].split(".")[0])
                     for k in weight_map if "layers." in k)
    print(f"src {src_dir}: {len(weight_map)} tensors, {layers} layers, "
          f"{len(shards)} shard(s)")

    # Dense matrices are 2-D; fused experts are 3-D but flatten to contiguous [N,K] rows.
    plan = build_plan(weight_map, shards, prefix, layers)

    meta, off = {}, 0
    for wname, sname, _src, wshape, sshape, N, K in plan:
        wb = N * K
        meta[wname] = {"dtype": "F8_E4M3", "shape": wshape, "data_offsets": [off, off + wb]}
        off += wb
        sb = N * 4
        meta[sname] = {"dtype": "F32", "shape": sshape, "data_offsets": [off, off + sb]}
        off += sb
    hdr_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    hdr_bytes += b" " * ((-len(hdr_bytes)) % 8)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "model.safetensors")
    written = 0
    with open(out, "wb") as o:
        o.write(struct.pack("<Q", len(hdr_bytes)))
        o.write(hdr_bytes)
        assert o.tell() == 8 + len(hdr_bytes)
        for i, (wname, sname, src_name, _wshape, _sshape, N, K) in enumerate(plan):
            hdr, data0, mm, _ = shards[weight_map[src_name]]
            a, e = hdr[src_name]["data_offsets"]
            if e - a != N * K * 2:
                raise ValueError(f"{src_name}: expected {N*K*2} BF16 bytes, got {e-a}")
            # Experts are ~1.4 GiB BF16 per fused tensor. Quantize bounded row chunks and retain
            # only the small f32 scale vector until all fp8 rows have been written.
            scale_parts = []
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
            for scale_bytes in scale_parts:
                o.write(scale_bytes)
            written += N * K + N * 4
            if i % 40 == 0:
                print(f"  [{i+1}/{len(plan)}] {src_name}  N={N} K={K}  "
                      f"rows/chunk<={ROW_CHUNK}", flush=True)
    for _, _, mm, backing in shards.values():
        mm.close()
        backing.close()
    print(f"done: {out}  ({written/1e9:.2f} GB over {len(plan)} projections)")


if __name__ == "__main__":
    main()
