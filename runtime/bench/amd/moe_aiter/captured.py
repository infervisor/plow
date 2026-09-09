#!/usr/bin/env python3
"""Compare grouped MoE on captured GLM activations and checkpoint TP8 weights."""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import struct

import aiter
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from safetensors import safe_open
import torch

from compare import timing


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fnuz_weight(weight):
    # Reinterpreting OCP as FNUZ halves finite values; compensate in block scales.
    raw = weight.view(torch.uint8).clone()
    raw[raw == 128] = 0
    converted = raw.view(torch.float8_e4m3fnuz)
    assert torch.equal(converted.float() * 2, weight.float())
    return converted


def checkpoint_weights(directory, layer, rank):
    prefix = f"model.layers.{layer}.mlp.experts."
    tensors = {}
    headers = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as f:
            header = f.read(struct.unpack("<Q", f.read(8))[0])
        entries = json.loads(header)
        selected = [name for name in entries if name.startswith(prefix)]
        if selected:
            headers[path.name] = hashlib.sha256(header).hexdigest()
            for name in selected:
                assert name not in tensors, name
                tensors[name] = path
    weights = [torch.empty(shape, device="cuda", dtype=dtype) for shape, dtype in [
        ((256, 512, 6144), torch.float8_e4m3fn),
        ((256, 6144, 256), torch.float8_e4m3fn),
        ((256, 4, 48), torch.float32),
        ((256, 48, 2), torch.float32),
    ]]
    digest = hashlib.sha256()
    negative_zeros = 0
    for expert in range(256):
        for projection, row_shard in [("gate", True), ("up", True), ("down", False)]:
            for scale in [False, True]:
                name = f"{prefix}{expert}.{projection}_proj.weight" + ("_scale_inv" if scale else "")
                width = 2 if scale else 256
                with safe_open(tensors[name], framework="pt", device="cpu") as f:
                    view = f.get_slice(name)
                    expected = ([16, 48] if row_shard else [48, 16]) if scale else (
                        [2048, 6144] if row_shard else [6144, 2048])
                    assert view.get_shape() == expected, (name, view.get_shape())
                    value = (view[rank*width:(rank+1)*width, :] if row_shard else
                             view[:, rank*width:(rank+1)*width]).contiguous()
                assert value.dtype == (torch.float32 if scale else torch.float8_e4m3fn)
                digest.update(name.encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
                if not scale:
                    # Match the runtime's scrub during upload to the gfx942 weight slab.
                    raw = value.view(torch.uint8)
                    negative_zeros += int((raw == 128).sum())
                    raw[raw == 128] = 0
                target = weights[(2 if scale else 0) + (0 if row_shard else 1)][expert]
                if row_shard:
                    offset = width if projection == "up" else 0
                    target = target[offset:offset+width]
                target.copy_(value)
    return weights, {"header_sha256": headers, "selected_tp_weight_sha256": digest.hexdigest(),
                     "negative_zeros_canonicalized": negative_zeros}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["plow", "aiter", "ck", "aiter-repack"], required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4464)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert "gfx942" in torch.cuda.get_device_properties(0).gcnArchName
    assert 0 <= args.rank < 8 and 0 < args.rows <= 8192
    rows, hidden, intermediate, experts, topk = args.rows, 6144, 256, 256, 8

    def read(name, dtype):
        return torch.frombuffer(bytearray((args.capture / name).read_bytes()), dtype=dtype).cuda()

    x = read("x.bin", torch.bfloat16).reshape(-1, hidden)[:rows].contiguous()
    table = read("tab.bin", torch.int32).reshape(-1, topk, 2)[:rows].contiguous()
    ids = table[:, :, 0].contiguous()
    weights = table[:, :, 1].contiguous().view(torch.float32)
    assert torch.isfinite(x).all() and torch.isfinite(weights).all()
    assert ((ids >= 0) & (ids < experts)).all() and (weights >= 0).all()
    assert (ids.sort(dim=1).values[:, 1:] != ids.sort(dim=1).values[:, :-1]).all()
    (w1, w2, s1, s2), checkpoint_meta = checkpoint_weights(args.checkpoint, args.layer, args.rank)
    assert torch.isfinite(w1.float()).all() and torch.isfinite(w2.float()).all()
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all() and (s1 > 0).all() and (s2 > 0).all()
    capture_check = None
    pack_ms = None
    lib = ctypes.CDLL(str(args.library.resolve()))
    wt = torch.tensor([
        [w1.data_ptr() + e*512*hidden, w1.data_ptr() + (e*2+1)*256*hidden,
         w2.data_ptr() + e*hidden*256] for e in range(experts)
    ], device="cuda", dtype=torch.int64)
    scale_stride = 2 * 48 * 4
    st = torch.tensor([
        [s1.data_ptr() + e*2*scale_stride, s1.data_ptr() + (e*2+1)*scale_stride,
         s2.data_ptr() + e*scale_stride] for e in range(experts)
    ], device="cuda", dtype=torch.int64)
    if args.arm == "plow":
        lib.plow_moe.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_uint, ctypes.c_void_p]
        lib.plow_moe.restype = ctypes.c_int
        capacity = rows*topk + experts*63
        meta = torch.empty(3*experts+1+64*experts, device="cuda", dtype=torch.int32)
        rt = torch.empty(capacity, device="cuda", dtype=torch.int32)
        rp, rg = torch.empty_like(rt), torch.empty(capacity, device="cuda")
        fu = torch.empty(capacity, intermediate, device="cuda", dtype=torch.bfloat16)
        part = torch.empty(rows, hidden, device="cuda", dtype=torch.float64)
        out = torch.empty_like(x)

        def run():
            rc = lib.plow_moe(*(t.data_ptr() for t in (out, x, wt, st, table, meta, rt, rp, rg, fu, part)),
                              rows, torch.cuda.current_stream().cuda_stream)
            assert rc == 0, rc
            return out
    else:
        if args.arm == "ck":
            importlib.import_module("aiter.fused_moe").fused_moe_1stage_dict["gfx942"] = {}
        wp1 = shuffle_weight(fnuz_weight(w1), (16, 16))
        wp2 = shuffle_weight(fnuz_weight(w2), (16, 16))
        sp1, sp2 = s1 * 2, s2 * 2
        if args.arm == "aiter-repack":
            lib.plow_moe_pack.argtypes = [ctypes.c_void_p] * 7
            lib.plow_moe_pack.restype = ctypes.c_int

            def pack():
                rc = lib.plow_moe_pack(*(t.data_ptr() for t in (wp1, wp2, sp1, sp2, wt, st)),
                                       torch.cuda.current_stream().cuda_stream)
                assert rc == 0, rc

            expected = [t.view(torch.uint8).clone() for t in (wp1, wp2, sp1, sp2)]
            for t in (wp1, wp2, sp1, sp2):
                t.view(torch.uint8).fill_(0xff)
            pack()
            assert all(torch.equal(t.view(torch.uint8), e)
                       for t, e in zip((wp1, wp2, sp1, sp2), expected))
            saved_wt, saved_st = wt.clone(), st.clone()
            wt.copy_(saved_wt.flip(0))
            st.copy_(saved_st.flip(0))
            pack()
            assert all(torch.equal(t.view(torch.uint8), e.flip(0))
                       for t, e in zip((wp1, wp2, sp1, sp2), expected))
            wt.copy_(saved_wt)
            st.copy_(saved_st)
            pack()
            assert all(torch.equal(t.view(torch.uint8), e)
                       for t, e in zip((wp1, wp2, sp1, sp2), expected))
            del expected
            pack_ms = timing.bench(pack)

        def run():
            if args.arm == "aiter-repack":
                pack()
            return fused_moe(x, wp1, wp2, weights, ids, quant_type=aiter.QuantType.per_1x128,
                             activation=aiter.ActivationType.Silu, w1_scale=sp1, w2_scale=sp2)

    out = run()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    if args.arm == "plow":
        old_fu = read("fu.bin", torch.bfloat16).reshape(-1, intermediate)
        old_rp = read("rp.bin", torch.int32)
        old_meta = read("meta.bin", torch.int32)
        # Only aligned rows covered by meta are initialized; capacity tails may be stale.
        old_live = torch.zeros_like(old_rp, dtype=torch.bool)
        live = torch.zeros_like(rp, dtype=torch.bool)
        for e in range(experts):
            old_live[int(old_meta[e]):int(old_meta[e] + old_meta[e+experts])] = True
            live[int(meta[e]):int(meta[e] + meta[e+experts])] = True
        old_live &= (old_rp >= 0) & (old_rp < rows*topk)
        live &= (rp >= 0) & (rp < rows*topk)
        assert int(old_live.sum()) == int(live.sum()) == rows*topk
        before = old_fu[old_live][old_rp[old_live].argsort()].float()
        after = fu[live][rp[live].argsort()].float()
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        assert old_rp[old_live].unique().numel() == rp[live].unique().numel() == rows*topk
        rel = ((before-after).square().sum() / before.square().sum()).sqrt().item()
        capture_check = {"glu_relative_l2": rel, "elements": before.numel(),
                         "exact_fraction": (before == after).float().mean().item()}
        assert rel < 0.01, capture_check

    picks = list(dict.fromkeys([0, rows//2, rows-1]))
    gold = torch.zeros(len(picks), hidden, device="cuda")
    for j, row in enumerate(picks):
        for slot in range(topk):
            expert = int(ids[row, slot])
            a = w1[expert].float() * s1[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
            b = w2[expert].float() * s2[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
            gate, up = (a @ x[row].float()).chunk(2)
            gold[j] += (b @ (torch.nn.functional.silu(gate) * up)) * weights[row, slot]
    rel = ((out[picks].float()-gold).square().sum() / gold.square().sum()).sqrt().item()
    assert rel < (0.01 if args.arm == "plow" else 0.1), rel
    record = {
        "arm": args.arm, "layer": args.layer, "rank": args.rank, "tp": 8, "rows": rows,
        "hidden": hidden, "intermediate": intermediate, "experts": experts, "topk": topk,
        "torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0),
        "aiter": importlib.metadata.version("amd-aiter"), "oracle_rows": picks,
        "relative_l2": rel, "ms": timing.bench(run), "capture_check": capture_check,
        "weight_pack_ms": pack_ms,
        "reusable_weight_pack_bytes": (w1.numel() + w2.numel() + (s1.numel()+s2.numel())*4)
        if pack_ms is not None else None,
        "library_sha256": sha(args.library), "source_sha256": sha(Path(__file__)),
        "capture_sha256": {p.name: sha(p) for p in sorted(args.capture.glob("*.bin"))},
        "checkpoint": checkpoint_meta,
        "boundary": "align, activation quantization if applicable, expert GLU/down, routing-weight combine; aiter-repack also includes weight shuffle and scale conversion on every dispatch; excludes checkpoint loading, router top-k, shared expert, residual and TP communication",
    }
    torch.save(out.cpu(), args.out.with_suffix(".pt"))
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ["arm", "rows", "relative_l2", "ms", "capture_check"]}), flush=True)


if __name__ == "__main__":
    main()
