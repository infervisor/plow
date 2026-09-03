#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import torch


FP4 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def weight_layout(src):
    e, n, kbytes = src.shape
    if n % 16 or kbytes % 32:
        raise ValueError("weight layout requires N%16=0 and Kbytes%32=0")
    return src.view(e, n // 16, 16, kbytes // 32, 2, 16).permute(
        0, 1, 3, 4, 2, 5
    ).contiguous().view(e, n, kbytes)


def scale_layout(src):
    m, n = src.shape
    sm, sn = math.ceil(m / 256) * 256, math.ceil(n / 8) * 8
    padded = torch.full((sm, sn), 127, dtype=src.dtype, device=src.device)
    padded[:m, :n] = src
    return padded.view(sm // 32, 2, 16, sn // 8, 2, 4).permute(
        0, 3, 5, 2, 4, 1
    ).contiguous().view(sm, sn)


def unpack_fp4(src):
    lut = FP4.to(src.device)
    indices = torch.stack((src & 15, src >> 4), dim=-1).reshape(*src.shape[:-1], -1)
    return lut[indices.long()]


def check_manifest(obj, manifest):
    required = {
        "status": "gate-only-not-routed",
        "capability.arch": "gfx950",
        "capability.wavefront": 64,
        "capability.workgroup": 256,
        "geometry.tile_m": 32,
        "geometry.tile_n": 256,
        "geometry.tile_k": 128,
        "geometry.sort_block_m": 32,
        "geometry.tokens": 1024,
        "geometry.topk": 16,
        "geometry.model_dim": 3584,
        "geometry.inter_dim": 384,
        "geometry.experts": 896,
        "abi.kernarg_bytes": 96,
        "encoding.activation": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
        "encoding.weight": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
        "encoding.output": "bf16-atomic-accumulate",
        "encoding.weight_layout": "E,N/16,Kbytes/32,2,16,16-permute-0,1,3,4,2,5",
        "encoding.scale_layout": "pad256x8-view-sm/32,2,16,sn/8,2,4-permute-0,3,5,2,4,1",
    }
    for key, expected in required.items():
        value = manifest
        for component in key.split("."):
            value = value[component]
        if value != expected:
            raise SystemExit(f"manifest gate failed: {key}={value!r}, expected {expected!r}")
    digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    if digest != manifest["object"]["sha256"]:
        raise SystemExit("object SHA-256 does not match manifest")
    expected_args = [
        "out*", "activation*", "weight*", "activation_scale*", "weight_scale*",
        "sorted_token_ids*", "expert_ids*", "sorted_weights*", "num_valid_ids*",
        "bias*", "tokens:i32", "model_dim:i32", "inter_dim:i32", "expert_blocks:i32",
    ]
    if manifest["abi"]["arguments"] != expected_args:
        raise SystemExit("manifest ABI argument order is not the phase-1 contract")
    implementation = manifest.get("implementation", "reference-aiter-generated")
    if implementation not in ("reference-aiter-generated", "native-hip"):
        raise SystemExit(f"unsupported implementation {implementation!r}")

    readelf = shutil.which("llvm-readelf") or "/opt/rocm/llvm/bin/llvm-readelf"
    notes = subprocess.check_output([readelf, "-n", str(obj)], text=True)
    symbol = manifest["object"]["symbol"]
    resources = manifest["resources"]
    for needle in (
        "amdgcn-amd-amdhsa--gfx950", f".name:           {symbol}",
        ".kernarg_segment_size: 96",
        f".group_segment_fixed_size: {resources['fixed_lds_bytes']}",
        f".private_segment_fixed_size: {resources['private_bytes']}",
        f".sgpr_count:     {resources['sgpr']}", f".vgpr_count:     {resources['vgpr']}",
        f".sgpr_spill_count: {resources['sgpr_spills']}",
        f".vgpr_spill_count: {resources['vgpr_spills']}", ".wavefront_size: 64",
    ):
        if needle not in notes:
            raise SystemExit(f"ELF metadata gate failed: missing {needle!r}")


class HipModule:
    def __init__(self, path, symbol):
        self.lib = ctypes.CDLL("libamdhip64.so")
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self._call("hipModuleLoad", ctypes.byref(self.module), str(path).encode())
        self._call("hipModuleGetFunction", ctypes.byref(self.function), self.module, symbol.encode())

    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with hipError_t {status}")

    def launch(self, grid, stream, shared_bytes, tensors, integers):
        holders = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
        holders.extend(ctypes.c_int32(v) for v in integers)
        params = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in holders)
        )
        self._call(
            "hipModuleLaunchKernel", self.function,
            ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(1),
            ctypes.c_uint(256), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(shared_bytes), ctypes.c_void_p(stream), params, ctypes.c_void_p(),
        )


def launch(module, manifest, out, act, weight, act_scale, weight_scale,
           ids, expert_ids, gates, num_valid):
    g = manifest["geometry"]
    bias = torch.empty(0, dtype=torch.float32, device="cuda")
    module.launch(
        (math.ceil(g["model_dim"] / g["tile_n"]), expert_ids.numel()),
        torch.cuda.current_stream().cuda_stream,
        manifest["resources"].get("dynamic_lds_bytes", 0),
        [out, act, weight, act_scale, weight_scale, ids, expert_ids, gates, num_valid, bias],
        [g["tokens"], g["model_dim"], g["inter_dim"], expert_ids.numel()],
    )


def oracle(module, manifest):
    g = manifest["geometry"]
    t, topk, n, k, e = g["tokens"], g["topk"], g["model_dim"], g["inter_dim"], g["experts"]
    rows = 32
    gen = torch.Generator().manual_seed(9301)
    act_focus = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=gen)
    act = torch.zeros((t, topk, k // 2), dtype=torch.uint8, device="cuda")
    act[:rows, 0].copy_(act_focus.to("cuda"))
    weight_focus = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    weight_raw = torch.zeros((e, n, k // 2), dtype=torch.uint8, device="cuda")
    weight_raw[0].copy_(weight_focus.to("cuda"))
    weight = weight_layout(weight_raw)
    del weight_raw
    act_scale_raw = torch.randint(126, 129, (rows, k // 32), dtype=torch.uint8, generator=gen)
    weight_scale_focus = torch.randint(126, 129, (n, k // 32), dtype=torch.uint8, generator=gen)
    act_scale = scale_layout(act_scale_raw.to("cuda"))
    weight_scale_raw = torch.full((e * n, k // 32), 127, dtype=torch.uint8, device="cuda")
    weight_scale_raw[:n].copy_(weight_scale_focus.to("cuda"))
    weight_scale = scale_layout(weight_scale_raw)
    del weight_scale_raw
    ids = torch.arange(rows, dtype=torch.int32, device="cuda")
    expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    gates = torch.linspace(0.125, 0.875, rows, dtype=torch.float32, device="cuda")
    num_valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    out = torch.zeros((t, n), dtype=torch.bfloat16, device="cuda")
    launch(module, manifest, out, act, weight, act_scale, weight_scale, ids, expert_ids, gates, num_valid)
    torch.cuda.synchronize()

    av = unpack_fp4(act_focus).reshape(rows, k // 32, 32)
    av *= torch.ldexp(torch.ones_like(av), act_scale_raw.to(torch.int32).unsqueeze(-1) - 127)
    bv = unpack_fp4(weight_focus).reshape(n, k // 32, 32)
    bv *= torch.ldexp(torch.ones_like(bv), weight_scale_focus.to(torch.int32).unsqueeze(-1) - 127)
    ref = ((av.reshape(rows, k) @ bv.reshape(n, k).T) * gates.cpu()[:, None]).to(torch.bfloat16).float()
    got = out[:rows].float().cpu()
    diff = (got - ref).abs()
    bad = int((diff > 0.02 * ref.abs().clamp_min(1.0)).sum())
    print(f"oracle rows={rows} values={got.numel()} bad={bad} max_abs={diff.max().item():.6g} max_rel={(diff / ref.abs().clamp_min(1)).max().item():.6g}")
    if bad:
        raise SystemExit(2)


def xorshift(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def timing(module, manifest):
    g = manifest["geometry"]
    t, topk, n, k, e, bm = g["tokens"], g["topk"], g["model_dim"], g["inter_dim"], g["experts"], g["sort_block_m"]
    state = 12345
    buckets = [[] for _ in range(e)]
    for routed in range(t * topk):
        state = xorshift(state); expert = state % e
        state = xorshift(state)
        if state & 3 == 0:
            state = xorshift(state); expert = state % (e // 8)
        token, slot = divmod(routed, topk)
        buckets[expert].append(token | (slot << 24))
    ids_host, experts_host = [], []
    for expert, bucket in enumerate(buckets):
        if not bucket:
            continue
        padded = math.ceil(len(bucket) / bm) * bm
        ids_host.extend(bucket + [-1] * (padded - len(bucket)))
        experts_host.extend([expert] * (padded // bm))
    padded = len(ids_host)
    act = torch.full((t, topk, k // 2), 0x53, dtype=torch.uint8, device="cuda")
    weight = weight_layout(torch.full((e, n, k // 2), 0x43, dtype=torch.uint8, device="cuda"))
    act_scale = scale_layout(torch.full((padded, k // 32), 127, dtype=torch.uint8, device="cuda"))
    weight_scale = scale_layout(torch.full((e * n, k // 32), 127, dtype=torch.uint8, device="cuda"))
    ids = torch.tensor(ids_host, dtype=torch.int32, device="cuda")
    expert_ids = torch.tensor(experts_host, dtype=torch.int32, device="cuda")
    gates = torch.full((padded,), 0.25, dtype=torch.float32, device="cuda")
    gates[ids == -1] = 0
    num_valid = torch.tensor([padded], dtype=torch.int32, device="cuda")
    out = torch.zeros((t, n), dtype=torch.bfloat16, device="cuda")

    for _ in range(5):
        out.zero_(); launch(module, manifest, out, act, weight, act_scale, weight_scale, ids, expert_ids, gates, num_valid)
    torch.cuda.synchronize()
    samples = {"forward_kernel": [], "forward_boundary": [], "reverse_kernel": [], "reverse_boundary": []}

    def measured(key, zero):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        if zero: out.zero_()
        launch(module, manifest, out, act, weight, act_scale, weight_scale, ids, expert_ids, gates, num_valid)
        end.record(); end.synchronize()
        samples[key].append(start.elapsed_time(end))

    for _ in range(31):
        out.zero_(); torch.cuda.synchronize()
        measured("forward_kernel", False); measured("forward_boundary", True)
        measured("reverse_boundary", True); out.zero_(); measured("reverse_kernel", False)
    med = {key: sorted(values)[len(values) // 2] for key, values in samples.items()}
    print(
        f"exact T={t} H={n} K={k} E={e} topk={topk} pad={padded} blocks={len(experts_host)} "
        f"forward_kernel_ms={med['forward_kernel']:.6f} forward_zero_kernel_ms={med['forward_boundary']:.6f} "
        f"reverse_kernel_ms={med['reverse_kernel']:.6f} reverse_zero_kernel_ms={med['reverse_boundary']:.6f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("object", type=Path)
    p.add_argument("manifest", type=Path)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    check_manifest(args.object, manifest)
    module = HipModule(args.object, manifest["object"]["symbol"])
    oracle(module, manifest)
    timing(module, manifest)


if __name__ == "__main__":
    main()
