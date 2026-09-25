#!/usr/bin/env python3
"""(a) prep: publish the block-fp8 form of the GLM-5.2 projections the existing prep DEQUANTISES.

`scripts/glm52_prep.py` writes `o_proj` and the shared expert as bf16 (`p_bf16(dequant_blockfp8(...))`).
Both are block-fp8 WHOLE TENSORS on disk, so their fp8 bytes and their [128,128] `weight_scale_inv`
grids can be republished VERBATIM — no dequant, no requant, no numeric change of any kind.
`--qkva` additionally concatenates original Q-A and KV-A bytes/scales at their block-aligned boundary.
`--mla-tp N` derives TP-specific MLA FP8 weights with BF16 preparation rounding.

This writes them ADDITIVELY into a new weight dir that SYMLINKS the existing 79 prepped shards, so
nothing already on disk is touched and no other agent's run is disturbed. The fp8 weight takes a
`.weight_fp8` name (the bf16 `.weight` still lives in the base shard and must not collide); the scale
grid keeps its checkpoint name `.weight_scale_inv`, which does not exist in the base shards.

Both names keep the projection substring the harness's `glm_col`/`glm_row` predicates match on, so
the TP slicing of the weight AND of its scale grid is already correct with no host change.

  python3 prep_fp8_linear.py --src <GLM-5.2-FP8 snapshot> --base <GLM-5.2-plow> --out <dir>
"""
import argparse, json, mmap, os, struct, sys, time

DT_ELT = {"F8_E4M3": 1, "F32": 4, "BF16": 2, "F16": 2}


def index_shards(model_dir):
    idx = {}
    for fn in sorted(os.listdir(model_dir)):
        if not (fn.startswith("model-") and fn.endswith(".safetensors")):
            continue
        path = os.path.join(model_dir, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            idx[k] = (path, base + v["data_offsets"][0], base + v["data_offsets"][1],
                      v["dtype"], v["shape"])
    return idx


_MM = {}
def mm(path):
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return _MM[path]


def shard_header(entries):
    hdr, off = {}, 0
    for name, rec in entries:
        parts = rec if isinstance(rec, list) else [rec]
        dt, shape = parts[0][3], list(parts[0][4])
        shape[0] = sum(part[4][0] for part in parts)
        size = 0
        for _p, a, b, pdt, dims in parts:
            if pdt != dt or dims[1:] != shape[1:]:
                raise ValueError(f"incompatible concatenation for {name}")
            size += b - a
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + size]}
        off += size
    return hdr, off


def write_shard(path, entries):
    """Stream original records, optionally concatenated along their first dimension."""
    hdr, off = shard_header(entries)
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * ((-((8 + len(blob)) % 8)) % 8)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _name, rec in entries:
            for p, a, b, _dt, _shape in rec if isinstance(rec, list) else [rec]:
                src = mm(p)
                o = a
                while o < b:
                    n = min(1 << 24, b - o)
                    f.write(src[o:o + n])
                    o += n
    os.replace(tmp, path)
    return off


def shard_ok(path, entries):
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
    except Exception:
        return False
    expected, end = shard_header(entries)
    if hdr != expected:
        return False
    return sz == 8 + n + end


def qkva_entries(src, config, layer):
    h, q = config["hidden_size"], config["q_lora_rank"]
    kv = config["kv_lora_rank"] + config["qk_rope_head_dim"]
    if min(h, q, kv) < 1 or h % 128 or q % 128:
        raise ValueError("QKV-A requires block-aligned K and Q/KV concatenation boundary")
    prefix = f"model.layers.{layer}.self_attn."
    result = []
    for suffix, dtype, shape in (("", "F8_E4M3", lambda n: [n, h]),
                                 ("_scale_inv", "F32", lambda n: [(n + 127) // 128, h // 128])):
        parts = []
        for proj, n in (("q_a_proj", q), ("kv_a_proj_with_mqa", kv)):
            rec = src[prefix + proj + ".weight" + suffix]
            dims = shape(n)
            if rec[3] != dtype or rec[4] != dims or rec[2] - rec[1] != dims[0] * dims[1] * DT_ELT[dtype]:
                raise ValueError(f"unexpected QKV-A dtype/shape: {proj}{suffix}")
            parts.append(rec)
        result.append((prefix + "fused_qkv_a_proj.weight" + (suffix or "_fp8"), parts))
    return result


def mla_tensors(src, config, layer, tp):
    import torch

    nh, ql, dk = config["num_attention_heads"], config["q_lora_rank"], config["kv_lora_rank"]
    qn, dr, vd = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
    if (tp < 1 or nh % tp or min(nh, ql, dk, qn, dr, vd) < 1 or ql % 128 or dk % 128
            or (nh // tp * (qn + dr)) % 128 or (nh // tp * (qn + vd)) % 128):
        raise ValueError("MLA FP8 requires block-aligned TP head shards and latent dimensions")
    prefix = f"model.layers.{layer}.self_attn."
    def tensor(name, dtype, shape):
        path, lo, hi, stored_dtype, stored_shape = src[prefix + name]
        expected_dtype = "F32" if dtype == torch.float32 else "F8_E4M3"
        if stored_dtype != expected_dtype or stored_shape != shape or hi - lo != shape[0] * shape[1] * DT_ELT[expected_dtype]:
            raise ValueError(f"unexpected MLA original dtype/shape: {name}")
        value = torch.frombuffer(bytearray(mm(path)[lo:hi]), dtype=dtype).reshape(shape)
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"nonfinite MLA original tensor: {name}")
        return value
    kv = tensor("kv_b_proj.weight", torch.float8_e4m3fn, [nh * (qn + vd), dk])
    kvs = tensor("kv_b_proj.weight_scale_inv", torch.float32, [nh * (qn + vd) // 128, dk // 128])
    result = {}
    matrices, scales = {"wk": [], "wv": []}, {"wk": [], "wv": []}
    rows, heads = nh // tp * (qn + vd), nh // tp
    for rank in range(tp):
        w = kv[rank * rows:(rank + 1) * rows].float()
        ws = kvs[rank * rows // 128:(rank + 1) * rows // 128].repeat_interleave(128, 0).repeat_interleave(128, 1)
        bf16 = (w * ws).to(torch.bfloat16).T.reshape(dk, heads, qn + vd)
        uk, uv = bf16.split([qn, vd], dim=-1)
        for tag, value in (("wk", uk.transpose(0, 1)), ("wv", uv.permute(1, 2, 0))):
            # Pinned vLLM keeps amax, multiplier and scaled weights BF16; the saved inverse is F32.
            low, high = value.aminmax()
            amax = torch.maximum(low.abs(), high.abs()).clamp(min=1e-10)
            multiplier = 448.0 / amax
            quant = (value * multiplier).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            inverse = multiplier.float().reciprocal()
            if not bool(torch.isfinite(quant.float()).all() and torch.isfinite(inverse) and inverse > 0):
                raise ValueError("invalid derived MLA FP8 tensor or scale")
            matrices[tag].append(quant)
            scales[tag].append(inverse)
    for tag in matrices:
        name = prefix + f"derived.mla_fp8_tp{tp}.{tag}"
        result[name + ".weight"] = torch.cat(matrices[tag], dim=0)
        result[name + ".weight_scale"] = torch.stack(scales[tag]).reshape(tp, 1)
    return result


def write_mla_shard(path, tensors):
    import torch
    from safetensors import safe_open, SafetensorError
    from safetensors.torch import save_file

    if os.path.exists(path):
        try:
            with safe_open(path, framework="pt", device="cpu") as shard:
                if set(shard.keys()) == set(tensors) and all(
                    (actual := shard.get_tensor(name)).dtype == want.dtype and actual.shape == want.shape
                    and torch.equal(actual.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
                    for name, want in tensors.items()
                ):
                    if os.stat(path).st_mode & 0o444 != 0o444:
                        os.chmod(path, 0o644)
                    return os.path.getsize(path)
        except (OSError, ValueError, SafetensorError):
            pass
    temporary = f"{path}.{os.getpid()}.tmp"
    save_file(tensors, temporary)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zai-org/GLM-5.2-FP8 checkpoint (block-fp8 on disk)")
    ap.add_argument("--base", default="/home/lava/models/GLM-5.2-plow", help="existing prepped dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--first-k-dense", type=int, default=3)
    ap.add_argument("--qkva", action="store_true", help="also preserve original fused QKV-A FP8 weights/scales")
    ap.add_argument("--mla-tp", type=int, help="derive MLA FP8 tensors for this TP degree (CPU PyTorch); original Q-B remains in the base")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # symlink the base dir's shards + config so the harness sees ONE complete weight dir
    for fn in sorted(os.listdir(a.base)):
        dst = os.path.join(a.out, fn)
        if os.path.lexists(dst):
            continue
        if fn.endswith(".safetensors"):
            os.symlink(os.path.join(a.base, fn), dst)
        elif fn in ("config.json", "model.safetensors.index.json"):
            os.symlink(os.path.join(a.base, fn), dst)

    src = index_shards(a.src)
    if a.qkva or a.mla_tp is not None:
        with open(os.path.join(a.src, "config.json")) as f:
            config = json.load(f)
    t0, tot = time.time(), 0
    for L in range(a.layers):
        if a.mla_tp is not None:
            tensors = mla_tensors(src, config, L, a.mla_tp)
            path = os.path.join(a.out, f"model-mla-tp{a.mla_tp}-{L:05d}.safetensors")
            tot += write_mla_shard(path, tensors)
            print(f"[mla-fp8] layer {L:2d}: TP{a.mla_tp}, {len(tensors)} tensors", flush=True)
        if a.qkva:
            entries = qkva_entries(src, config, L)
            path = os.path.join(a.out, f"model-qkva-{L:05d}.safetensors")
            if not shard_ok(path, entries):
                write_shard(path, entries)
            tot += os.path.getsize(path)
        p = f"model.layers.{L}."
        want = [(p + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight_fp8")]
        if L >= a.first_k_dense:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                want.append((p + f"mlp.shared_experts.{proj}.weight",
                             p + f"mlp.shared_experts.{proj}.weight_fp8"))
        entries = []
        for sname, oname in want:
            rec = src[sname]
            assert rec[3] == "F8_E4M3", (sname, rec[3])
            entries.append((oname, rec))
            srec = src[sname + "_scale_inv"]
            assert srec[3] == "F32", (sname, srec[3])
            entries.append((oname.replace("_fp8", "_scale_inv"), srec))
        path = os.path.join(a.out, f"model-idx-{L:05d}-of-idx.safetensors")
        if shard_ok(path, entries):
            tot += os.path.getsize(path)
            continue
        tot += write_shard(path, entries)
        print(f"[fp8lin] layer {L:2d}: {len(entries)} tensors, {tot/1e9:.2f} GB cum, "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[fp8lin] DONE {tot/1e9:.2f} GB in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
