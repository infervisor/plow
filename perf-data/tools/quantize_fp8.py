#!/usr/bin/env python3
"""Quantize a Gemma-4 or Qwen3.5-family bf16 checkpoint to the per-output-channel e4m3 fp8 weight
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
Qwen linear-attention QKV/Z/A/B/output projections are also quantized.
Convolution, A_log, dt_bias, router/norm/embedding/lm_head tensors stay bf16. No numpy dependency (torch only).

Both single-file checkpoints (`model.safetensors`) and sharded checkpoints
(`model.safetensors.index.json`) are accepted.

Usage: quantize_fp8.py <src-model-dir> <out-dir> [prefix]
  prefix default "model.language_model." (Gemma-4 multimodal re-export).
  --scale-mode packed-tensor uses one scalar per vLLM packed dense matrix, broadcast
  into the existing scale vectors. Output tensors remain separate row-major matrices.
  This exports weights only; it does not enable FP8 activations or claim W8A8 parity.
"""
import sys, os, struct, json, mmap, ctypes, argparse, hashlib
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
GDN_PROJS = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
             "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
             "linear_attn.out_proj.weight"]
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
        for proj in PROJS + GDN_PROJS + EXPERT_PROJS:
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


PACKED_SUFFIXES = {
    "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", 0),
    "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", 1),
    "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", 2),
    "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0),
    "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", 1),
    "linear_attn.in_proj_qkv.weight": ("linear_attn.in_proj_qkvz.weight", 0),
    "linear_attn.in_proj_z.weight": ("linear_attn.in_proj_qkvz.weight", 1),
    "linear_attn.in_proj_b.weight": ("linear_attn.in_proj_ba.weight", 0),
    "linear_attn.in_proj_a.weight": ("linear_attn.in_proj_ba.weight", 1),
}


def packed_group(name):
    for suffix, (packed, order) in PACKED_SUFFIXES.items():
        if name.endswith(suffix):
            return name[:-len(suffix)] + packed, order
    return name, 0


def source_chunks(name, N, K, weight_map, shards):
    hdr, data0, mm, _ = shards[weight_map[name]]
    a, e = hdr[name]["data_offsets"]
    if hdr[name]["dtype"] != "BF16" or e - a != N * K * 2:
        raise ValueError(f"{name}: expected {N*K*2} BF16 bytes")
    for row0 in range(0, N, ROW_CHUNK):
        nr = min(ROW_CHUNK, N - row0)
        b0 = data0 + a + row0 * K * 2
        raw = bytearray(mm[b0:b0 + nr * K * 2])
        yield torch.frombuffer(raw, dtype=torch.bfloat16).view(nr, K).float(), raw


def packed_scales(plan, weight_map, shards):
    groups = {}
    for _wn, _sn, name, shape, _ss, N, K in plan:
        if len(shape) != 2:
            raise ValueError("packed-tensor mode currently supports dense 2-D weights only")
        group, order = packed_group(name)
        item = groups.setdefault(group, {"members": [], "amax": 0.0, "input_width": K})
        if item["input_width"] != K:
            raise ValueError(f"{group}: packed members have different input widths")
        digest = hashlib.sha256()
        amax = 0.0
        for w, raw in source_chunks(name, N, K, weight_map, shards):
            lo, hi = w.aminmax()
            value = torch.maximum(lo.abs(), hi.abs()).item()
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name}: nonfinite source weight")
            amax = max(amax, value)
            digest.update(raw)
        item["amax"] = max(item["amax"], amax)
        item["members"].append({"source": name, "shape": shape, "order": order,
                                "source_sha256": digest.hexdigest()})
    scales = {}
    for group, item in groups.items():
        if item["amax"] == 0:
            raise ValueError(f"{group}: zero packed amax is not supported by this matching profile")
        scale = (torch.tensor(item["amax"], dtype=torch.float32) / E4M3_MAX).item()
        item["scale_f32"] = scale
        item["scale_f32_bits"] = struct.unpack("<I", struct.pack("<f", scale))[0]
        row = 0
        item["members"].sort(key=lambda m: m["order"])
        for member in item["members"]:
            member["packed_row_offset"] = row
            row += member["shape"][0]
            scales[member["source"]] = scale
        item["packed_shape"] = [row, item["input_width"]]
    return scales, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("prefix", nargs="?", default="model.language_model.")
    parser.add_argument("--scale-mode", choices=["per-channel", "packed-tensor"], default="per-channel")
    args = parser.parse_args()
    src_dir, out_dir, prefix = args.src_dir, args.out_dir, args.prefix
    if args.scale_mode == "packed-tensor" and os.path.exists(os.path.join(out_dir, "model.safetensors")):
        raise FileExistsError("packed-tensor export requires a new output file")
    weight_map, shards = open_sources(src_dir)
    layers = 1 + max(int(k.split("layers.")[1].split(".")[0])
                     for k in weight_map if "layers." in k)
    print(f"src {src_dir}: {len(weight_map)} tensors, {layers} layers, "
          f"{len(shards)} shard(s)")

    # Dense matrices are 2-D; fused experts are 3-D but flatten to contiguous [N,K] rows.
    plan = build_plan(weight_map, shards, prefix, layers)
    if not plan:
        raise ValueError("no supported projection weights found")
    scales, groups = ({}, {})
    if args.scale_mode == "packed-tensor":
        scales, groups = packed_scales(plan, weight_map, shards)
        print(f"packed groups: {len(groups)}", flush=True)

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
    with open(out, "xb" if args.scale_mode == "packed-tensor" else "wb") as o:
        o.write(struct.pack("<Q", len(hdr_bytes)))
        o.write(hdr_bytes)
        assert o.tell() == 8 + len(hdr_bytes)
        for i, (wname, sname, src_name, _wshape, _sshape, N, K) in enumerate(plan):
            scale_parts = []
            for w, _raw in source_chunks(src_name, N, K, weight_map, shards):
                if args.scale_mode == "packed-tensor":
                    scale = torch.full((w.shape[0],), scales[src_name], dtype=torch.float32)
                    q = (w * scale.unsqueeze(1).reciprocal()).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
                else:
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
    if args.scale_mode == "packed-tensor":
        with open(os.path.join(src_dir, "config.json"), "rb") as f:
            config_sha256 = hashlib.sha256(f.read()).hexdigest()
        with open(__file__, "rb") as f:
            exporter_sha256 = hashlib.sha256(f.read()).hexdigest()
        metadata = {
            "schema": 1, "complete": True, "scale_mode": args.scale_mode,
            "source": os.path.realpath(src_dir), "config_sha256": config_sha256,
            "exporter_sha256": exporter_sha256,
            "torch_version": torch.__version__, "tensor_parallel_size": 1,
            "weight_dtype": "float8_e4m3fn", "scale_dtype": "float32",
            "layout": "original row-major [N,K] tensors; packed rows are not physically concatenated",
            "scale_storage": "packed scalar broadcast to each member's per-output-channel scale vector",
            "weight_rule": "FP32 packed amax/448; FP32 reciprocal multiply, saturate [-448,448], E4M3FN round-to-nearest",
            "reference": "vLLM0.28 Fp8PerTensorOnlineLinearMethod",
            "activation_status": "not implemented by this weight exporter; no W8A8 equivalence claim",
            "reference_activation": "H100 CUTLASS: dynamic per-token; other backends may select per-tensor",
            "activation_gap": "Plow QuantFp8 floor1e-12 differs from vLLM native fallback1/(448*512); CUDA parity unverified",
            "groups": groups,
        }
        with open(os.path.join(out_dir, "quantization.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    print(f"done: {out}  ({written/1e9:.2f} GB over {len(plan)} projections)")


if __name__ == "__main__":
    main()
