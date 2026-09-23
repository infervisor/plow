#!/usr/bin/env python3
"""Compare block_fp8_gfx950_test's prequantized captures with installed vLLM/AITER."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import torch


def expected_shapes():
    return {(m, n, k) for m in (1, 8, 16, 32, 64) for n, k in ((256, 6144), (6144, 256))} | {
        (8, 2048, 6144), (8, 6144, 2048), (3, 130, 260), (65, 129, 129),
    }


def load_case(path, glu=False, weighted=False):
    raw = bytearray(path.read_bytes())
    m, n, k = struct.unpack_from("<III", raw)
    if not min(m, n, k) or sys.byteorder != "little" or (glu and weighted):
        raise ValueError("capture requires positive dimensions and a little-endian host")
    kb, nb = (k + 127) // 128, (n + 127) // 128
    branches = 2 if glu else 1
    expected = 12 + m * k + branches * n * k + kb * m * 4 + branches * nb * kb * 4 + m * n * 2 + (m * 4 if weighted else 0)
    if len(raw) != expected:
        raise ValueError("unexpected capture size")
    offset = 12

    def take(dtype, shape):
        nonlocal offset
        count = 1
        for dim in shape:
            count *= dim
        value = torch.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape).clone()
        offset += count * value.element_size()
        return value

    a = take(torch.uint8, (m, k)).view(torch.float8_e4m3fn)
    w = take(torch.uint8, (branches * n, k)).view(torch.float8_e4m3fn)
    asc = take(torch.float32, (kb, m)).T.contiguous()
    wsc = take(torch.float32, (branches * nb, kb))
    out = take(torch.bfloat16, (m, n))
    gates = (take(torch.float32, (m,)),) if weighted else ()
    if offset != len(raw):
        raise ValueError("unexpected capture size")
    return (m, n, k), (a, w, asc, wsc) + gates, out, hashlib.sha256(raw).hexdigest()


def load_quant_case(path):
    raw = bytearray(path.read_bytes())
    m, k = struct.unpack_from("<II", raw)
    if not m or not k or k % 128 or sys.byteorder != "little":
        raise ValueError("quant capture requires positive aligned dimensions and little endian")
    count, groups = m * k, m * k // 128
    if len(raw) != 8 + count * 3 + groups * 4:
        raise ValueError("unexpected quant capture size")
    x = torch.frombuffer(raw, dtype=torch.bfloat16, count=count, offset=8).reshape(m, k).clone()
    q = torch.frombuffer(raw, dtype=torch.uint8, count=count, offset=8 + count * 2).reshape(m, k).clone()
    scale = torch.frombuffer(raw, dtype=torch.float32, count=groups, offset=8 + count * 3)
    scale = scale.reshape(k // 128, m).T.contiguous().clone()
    return (m, k), x, q, scale, hashlib.sha256(raw).hexdigest()


def write_case(path, a, w, asc, wsc, expected, glu=False, row_weights=None):
    m, k = a.shape
    n, wk = w.shape
    branches = 2 if glu else 1
    if glu:
        if n % 256:
            raise ValueError("GLU replay requires two aligned output halves")
        n //= 2
    if (not min(m, n, k) or wk != k or a.dtype != torch.uint8 or w.dtype != torch.float8_e4m3fn
            or asc.dtype != torch.float32 or wsc.dtype != torch.float32 or expected.dtype != torch.bfloat16
            or tuple(asc.shape) != ((k + 127) // 128, m)
            or tuple(wsc.shape) != (branches * ((n + 127) // 128), (k + 127) // 128)
            or tuple(expected.shape) != (m, n) or sys.byteorder != "little"):
        raise ValueError("unsupported replay capture geometry or dtype")
    if row_weights is not None and (glu or row_weights.dtype != torch.float32
            or row_weights.shape != (m,) or not bool(torch.isfinite(row_weights).all())):
        raise ValueError("weighted replay requires one finite FP32 gate per row")
    with path.open("xb") as f:
        f.write(struct.pack("<III", m, n, k))
        for value in (a, w, asc, wsc, expected):
            f.write(value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        if row_weights is not None:
            f.write(row_weights.cpu().contiguous().numpy().tobytes())


def load_tensor(path, dtype, shape):
    raw = bytearray(path.read_bytes())
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("capture dimensions must be positive")
        count *= dim
    if sys.byteorder != "little" or len(raw) != count * torch.empty((), dtype=dtype).element_size():
        raise ValueError(f"{path}: unexpected tensor size or byte order")
    value = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
    return value, hashlib.sha256(raw).hexdigest()


def load_routes(path, batch, topk, experts):
    table, digest = load_tensor(path, torch.int32, (batch, topk, 2))
    ids = table[:, :, 0].contiguous()
    weights = table[:, :, 1].contiguous().view(torch.float32)
    if (bool((ids < 0).any() or (ids >= experts).any())
            or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or any(len(set(row)) != topk for row in ids.tolist())):
        raise ValueError("invalid captured expert routes")
    return ids, weights, digest


def routed_weights(checkpoint, layer, rank, tp):
    from safetensors import safe_open

    config = json.loads((checkpoint / "config.json").read_text())
    h, full_i, experts = (config[key] for key in
                          ("hidden_size", "moe_intermediate_size", "n_routed_experts"))
    if tp < 1 or not 0 <= rank < tp or full_i % tp or full_i // tp % 128 or h % 128:
        raise ValueError("unsupported routed block-FP8 TP geometry")
    i = full_i // tp
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    result = (torch.empty((experts, 2 * i, h), dtype=torch.float8_e4m3fn),
              torch.empty((experts, h, i), dtype=torch.float8_e4m3fn),
              torch.empty((experts, 2 * i // 128, h // 128), dtype=torch.float32),
              torch.empty((experts, h // 128, i // 128), dtype=torch.float32))
    for expert in range(experts):
        for proj in ("gate", "up", "down"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight"
            down = proj == "down"
            for scaled in (False, True):
                key = name + ("_scale_inv" if scaled else "")
                unit = 128 if scaled else 1
                dtype = torch.float32 if scaled else torch.float8_e4m3fn
                with safe_open(checkpoint / index[key], framework="pt", device="cpu") as shard:
                    view = shard.get_slice(key)
                    shape = [h // unit, full_i // unit] if down else [full_i // unit, h // unit]
                    if view.get_shape() != shape:
                        raise ValueError(f"{key}: unexpected checkpoint geometry")
                    lo, hi = rank * i // unit, (rank + 1) * i // unit
                    value = view[:, lo:hi] if down else view[lo:hi, :]
                if value.dtype != dtype:
                    raise ValueError(f"{key}: unexpected checkpoint dtype")
                dest = result[(1 if down else 0) + (2 if scaled else 0)][expert]
                if not down:
                    start = i // unit if proj == "up" else 0
                    dest = dest[start:start + i // unit]
                dest.copy_(value)
    return result


def qkva_weights(checkpoint, layer):
    from safetensors import safe_open

    config = json.loads((checkpoint / "config.json").read_text())
    h, ql, kv = config["hidden_size"], config["q_lora_rank"], config["kv_lora_rank"] + config["qk_rope_head_dim"]
    if min(h, ql, kv) < 1 or h % 128 or ql % 128:
        raise ValueError("fused QKV-A requires aligned input and Q/KV concatenation boundary")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights, scales = [], []
    for proj, n in (("q_a_proj", ql), ("kv_a_proj_with_mqa", kv)):
        name = f"model.layers.{layer}.self_attn.{proj}.weight"
        for suffix, dtype, shape, dest in (("", torch.float8_e4m3fn, (n, h), weights),
                                          ("_scale_inv", torch.float32, ((n + 127) // 128, h // 128), scales)):
            key = name + suffix
            with safe_open(checkpoint / index[key], framework="pt", device="cpu") as shard:
                value = shard.get_tensor(key)
            if value.dtype != dtype or tuple(value.shape) != shape:
                raise ValueError(f"{key}: unexpected dtype or shape")
            dest.append(value)
    name = f"model.layers.{layer}.input_layernorm.weight"
    with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
        gamma = shard.get_tensor(name)
    if gamma.dtype != torch.bfloat16 or gamma.shape != (h,):
        raise ValueError("unexpected input norm weight")
    return torch.cat(weights), torch.cat(scales), gamma


def qb_weights(checkpoint, layer, rank, tp):
    from safetensors import safe_open

    cfg = json.loads((checkpoint / "config.json").read_text())
    heads, k = cfg["num_attention_heads"], cfg["q_lora_rank"]
    width = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    if tp < 1 or not 0 <= rank < tp or heads % tp or (heads // tp * width) % 128 or k % 128:
        raise ValueError("unsupported Q-B TP/block geometry")
    n = heads // tp * width
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    name = f"model.layers.{layer}.self_attn.q_b_proj.weight"
    result = []
    for suffix, dtype, unit in (("", torch.float8_e4m3fn, 1), ("_scale_inv", torch.float32, 128)):
        key = name + suffix
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as shard:
            view = shard.get_slice(key)
            if view.get_shape() != [heads * width // unit, k // unit]:
                raise ValueError(f"{key}: unexpected original weight geometry")
            value = view[rank * n // unit:(rank + 1) * n // unit, :]
        if value.dtype != dtype:
            raise ValueError(f"{key}: unexpected original weight dtype")
        result.append(value)
    name = f"model.layers.{layer}.self_attn.q_a_layernorm.weight"
    with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
        gamma = shard.get_tensor(name)
    if gamma.dtype != torch.bfloat16 or gamma.shape != (k,):
        raise ValueError("unexpected Q-A norm weight")
    return (*result, gamma)


def export_qb(args, rocm_aiter_ops, version):
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS, gemm_a8w8_blockscale_ck

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None or args.reference_json is None:
        raise ValueError("requires pinned vLLM, original checkpoint, loaded inventory and QKV-A reference")
    source = json.loads(args.reference_json.read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    layer, k = source["layer"], cfg["q_lora_rank"]
    rungs = (1, 8, 16, 32, 64, 128)
    if (not source["audit_complete"] or source["vllm_version"] != version or args.tp < 1
            or len(inventory["ranks"]) != args.tp or len(source["cases"]) != len(rungs)
            or {c["shape"][0] for c in source["cases"]} != set(rungs)):
        raise ValueError("incomplete independent QKV-A reference or loaded TP mismatch")
    inputs = {}
    for case in source["cases"]:
        shape, _, out, digest = load_case(args.reference_json.parent / case["file"])
        if (list(shape) != case["shape"] or shape[1:] != (k + cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], cfg["hidden_size"])
                or digest != case["sha256"] or tensor_digest(out) != case["reference_sha256"]
                or not all(case[key] for key in ("finite", "norm_quant_scales_repeat_bitwise", "output_repeat_bitwise"))):
            raise ValueError("QKV-A source hash/geometry/stability mismatch")
        inputs[shape[0]] = out[:, :k].contiguous()
    cases, weights = [], []
    for rank in range(args.tp):
        w, ws, gamma = qb_weights(args.checkpoint, layer, rank, args.tp)
        n, _ = w.shape
        module = inventory["ranks"][rank]["modules"][f"model.layers.{layer}.self_attn.q_b_proj"]
        tensors = module["tensors"]
        selected = module["attributes"]["quant_method"]["fields"]["fp8_linear"]
        stride = tensors["weight"]["stride"]
        if (tensors["weight"]["shape"] != [n, k] or tensors["weight"]["dtype"] != "torch.float8_e4m3fn"
                or tensors["weight_scale_inv"]["shape"] != list(ws.shape)
                or tensors["weight_scale_inv"]["dtype"] != "torch.float32"
                or len(stride) != 2 or stride[1] != 1 or stride[0] < k
                or selected["class_name"].rsplit(".", 1)[-1] != "AiterFp8BlockScaledMMKernel"
                or selected["fields"]["use_triton"] is not False
                or selected["fields"]["quant_fp8"]["fields"]["use_ue8m0"] is not False):
            raise ValueError("Q-B differs from loaded pinned backend/geometry")
        gpu_w = torch.empty((n, stride[0]), dtype=w.dtype, device="cuda")[:, :k]
        gpu_w.copy_(w)
        gpu_ws, gpu_gamma = ws.cuda(), gamma.cuda()
        weights.append(dict(rank=rank, weight_sha256=tensor_digest(w), scale_sha256=tensor_digest(ws),
                            norm_weight_sha256=tensor_digest(gamma), loaded_weight_stride=stride))
        for m in rungs:
            x = inputs[m].cuda()
            def run():
                norm = rocm_aiter_ops.rms_norm(x, gpu_gamma, cfg["rms_norm_eps"])
                quant, asc = rocm_aiter_ops.group_fp8_quant(norm, 128)
                out = rocm_aiter_ops.gemm_a8w8_blockscale(quant, gpu_w, asc, gpu_ws,
                                                       [128, 128], output_dtype=torch.bfloat16)
                return tuple(t.cpu() for t in (norm, quant, asc, out))
            values, repeats = run(), run()
            stable = all(tensor_digest(a) == tensor_digest(b) for a, b in zip(values, repeats))
            finite = all(bool(torch.isfinite(t.float()).all()) for t in (*values, *repeats))
            dest = args.output.parent / f"rank{rank}.m{m}.qb.bin"
            write_case(dest, values[1].view(torch.uint8), w, values[2].T.contiguous(), ws, values[3])
            with (args.output.parent / f"rank{rank}.m{m}.norm.bf16").open("xb") as f:
                f.write(values[0].contiguous().view(torch.uint8).numpy().tobytes())
            tuned = get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
            splits = 1 << int(tuned["splitK"]) if tuned is not None else 1
            if splits > 8 or k % (128 * splits):
                raise ValueError("Q-B split-K partition is outside the qualified block-aligned contract")
            parts, part_cases = [], []
            parts_stable = True
            for part in range(splits):
                lo, hi = part * (k // splits), (part + 1) * (k // splits)
                a = values[1][:, lo:hi].contiguous().cuda()
                weight = w[:, lo:hi].contiguous().cuda()
                asc = values[2][:, lo // 128:hi // 128].contiguous().cuda()
                wsc = ws[:, lo // 128:hi // 128].contiguous().cuda()
                def isolated():
                    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                    return gemm_a8w8_blockscale_ck(a, weight, asc, wsc, y, splitK=0,
                        kernelName="" if tuned is None else str(tuned["kernelName"])).cpu()
                value, repeat = isolated(), isolated()
                parts_stable &= tensor_digest(value) == tensor_digest(repeat)
                finite &= bool(torch.isfinite(value).all() and torch.isfinite(repeat).all())
                parts.append(value)
                part_path = args.output.parent / f"rank{rank}.m{m}.part{part}.qb.bin"
                write_case(part_path, a.cpu().view(torch.uint8), weight.cpu(), asc.cpu().T.contiguous(), wsc.cpu(), value)
                part_cases.append(dict(part=part, k_start=lo, k_end=hi, file=part_path.name,
                    sha256=hashlib.sha256(part_path.read_bytes()).hexdigest(), reference_sha256=tensor_digest(value)))
            lower, upper = bf16_order_bounds(torch.stack(parts, dim=1))
            bounds = {tag: bool(((y >= lower) & (y <= upper)).all())
                      for tag, y in (("reference", values[3]), ("repeat", repeats[3]))}
            with (args.output.parent / f"rank{rank}.m{m}.repeat.bf16").open("xb") as f:
                f.write(repeats[3].contiguous().view(torch.uint8).numpy().tobytes())
            row = dict(rank=rank, shape=[m, n, k], file=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                input_sha256=tensor_digest(x), norm_sha256=tensor_digest(values[0]),
                reference_sha256=tensor_digest(values[3]), repeat_sha256=tensor_digest(repeats[3]),
                finite=finite, all_boundaries_repeat_bitwise=stable,
                norm_quant_scales_repeat_bitwise=all(tensor_digest(a) == tensor_digest(b) for a, b in zip(values[:3], repeats[:3])),
                split_count=splits, isolated_parts_repeat_bitwise=parts_stable, parts=part_cases,
                bf16_addition_order_bounds=bounds,
                selected_config=None if tuned is None else {key: str(value) for key, value in tuned.items()})
            cases.append(row)
            print(json.dumps(row), flush=True)
    complete = all(c["finite"] and c["norm_quant_scales_repeat_bitwise"] and c["isolated_parts_repeat_bitwise"]
                   and all(c["bf16_addition_order_bounds"].values()) for c in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="Q-A norm and original FP8 Q-B projection from independent QKV-A component outputs; not full block or serving",
            audit_complete=complete, precision_qualified=False, vllm_version=version, layer=layer, tp=args.tp,
            source_reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
            weights=weights, cases=cases), f, indent=2, allow_nan=False)
    return 0 if complete else 1


def check_qb(args):
    if args.reference_json is None:
        raise ValueError("requires independent Q-B reference")
    audit = json.loads(args.reference_json.read_text())
    expected = {(r, m) for r in range(audit["tp"]) for m in (1, 8, 16, 32, 64, 128)}
    keys = [(c["rank"], c["shape"][0]) for c in audit["cases"]]
    if (not audit["audit_complete"] or audit["vllm_version"] != "0.29.0" or audit["tp"] < 1
            or set(keys) != expected or len(keys) != len(expected)):
        raise ValueError("Q-B reference must cover the pinned TP/rung set exactly once")
    records = []
    for case in audit["cases"]:
        shape, (a, w, asc, wsc), ref, digest = load_case(args.reference_json.parent / case["file"])
        m, n, k = shape
        splits = case["split_count"]
        if (list(shape) != case["shape"] or digest != case["sha256"] or tensor_digest(ref) != case["reference_sha256"]
                or splits not in (1, 4, 8) or k % (128 * splits) or len(case["parts"]) != splits
                or not case["finite"] or not case["norm_quant_scales_repeat_bitwise"]
                or not case["isolated_parts_repeat_bitwise"]):
            raise ValueError("Q-B reference artifact hash/geometry/stability mismatch")
        parts = []
        for part, item in enumerate(case["parts"]):
            ps, (pa, pw, pas, pws), value, phash = load_case(args.reference_json.parent / item["file"])
            lo, hi = part * (k // splits), (part + 1) * (k // splits)
            if (ps != (m, n, k // splits) or phash != item["sha256"] or tensor_digest(value) != item["reference_sha256"]
                    or (item["part"], item["k_start"], item["k_end"]) != (part, lo, hi)
                    or any(tensor_digest(x) != tensor_digest(y) for x, y in
                           ((pa, a[:, lo:hi]), (pw, w[:, lo:hi]),
                            (pas, asc[:, lo // 128:hi // 128]), (pws, wsc[:, lo // 128:hi // 128])))):
                raise ValueError("isolated Q-B part differs from original full operands or partition")
            parts.append(value)
        want = torch.stack(parts)
        stem = f"rank{case['rank']}.m{m}"
        got, got_hash = load_tensor(args.capture / "outputs" / f"{stem}.parts.bf16", torch.bfloat16, want.shape)
        part_bitwise = tensor_digest(want) == got_hash
        lower, upper = bf16_order_bounds(want.permute(1, 0, 2))
        repeat, repeat_hash = load_tensor(args.reference_json.parent / f"{stem}.repeat.bf16", torch.bfloat16, ref.shape)
        if repeat_hash != case["repeat_sha256"]:
            raise ValueError("Q-B repeat hash mismatch")
        bounds = {tag: bool(torch.isfinite(y).all() and ((y >= lower) & (y <= upper)).all())
                  for tag, y in (("reference", ref), ("repeat", repeat))}
        hashes = {}
        for run in range(4):
            value, hashes[str(run)] = load_tensor(args.capture / "outputs" / f"{stem}.atomic{run}.bf16", torch.bfloat16, ref.shape)
            bounds[f"native{run}"] = bool(torch.isfinite(value).all() and ((value >= lower) & (value <= upper)).all())
        records.append(dict(rank=case["rank"], shape=list(shape), split_count=splits,
            parts_bitwise=part_bitwise, parts_sha256=got_hash, atomic_sha256=hashes,
            bf16_addition_order_bounds=bounds, passed=part_bitwise and all(bounds.values())))
    passed = all(r["passed"] for r in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="Q-B native split-K parts and BF16 atomic reduction against pinned CK; not full-model parity",
            precision_qualified=False, passed=passed, cases=records,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), f, indent=2, allow_nan=False)
    print(f"validated {len(records)} Q-B split-K cases; passed={passed}", flush=True)
    return 0 if passed else 1


def mla_kvb_weights(checkpoint, layer, rank, tp):
    from safetensors import safe_open

    cfg = json.loads((checkpoint / "config.json").read_text())
    heads, k = cfg["num_attention_heads"], cfg["kv_lora_rank"]
    width = cfg["qk_nope_head_dim"] + cfg["v_head_dim"]
    if tp < 1 or not 0 <= rank < tp or heads % tp or (heads // tp * width) % 128 or k % 128:
        raise ValueError("unsupported MLA TP/block geometry")
    n = heads // tp * width
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
    result = []
    for suffix, dtype, unit in (("", torch.float8_e4m3fn, 1), ("_scale_inv", torch.float32, 128)):
        key = name + suffix
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as shard:
            view = shard.get_slice(key)
            if view.get_shape() != [heads * width // unit, k // unit]:
                raise ValueError(f"{key}: unexpected original weight geometry")
            value = view[rank * n // unit:(rank + 1) * n // unit, :]
        if value.dtype != dtype:
            raise ValueError(f"{key}: unexpected original weight dtype")
        result.append(value)
    return tuple(result)


def write_mla_case(path, x, w, scale, out):
    if x.ndim != 3 or w.ndim != 3:
        raise ValueError("MLA replay requires rank-three input and weight")
    m, heads, k = x.shape
    wh, n, wk = w.shape
    if (min(m, heads, n, k) < 1 or (wh, wk) != (heads, k)
            or x.dtype != torch.bfloat16 or w.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float32 or scale.shape != ()
            or out.dtype != torch.bfloat16 or out.shape != (m, heads, n)
            or sys.byteorder != "little"
            or not all(bool(torch.isfinite(t.float()).all()) for t in (x, w, scale, out))
            or not bool(scale > 0)):
        raise ValueError("unsupported MLA replay geometry, dtype or values")
    with path.open("xb") as f:
        f.write(struct.pack("<IIII", m, heads, n, k))
        for t in (x, w, scale, out):
            f.write(t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def load_mla_case(path):
    raw = bytearray(path.read_bytes())
    if len(raw) < 16:
        raise ValueError("truncated MLA replay header")
    m, heads, n, k = struct.unpack_from("<IIII", raw)
    if (not min(m, heads, n, k) or sys.byteorder != "little"
            or len(raw) != 16 + 2 * m * heads * k + heads * n * k + 4 + 2 * m * heads * n):
        raise ValueError("unexpected MLA replay geometry or size")
    offset = 16
    tensors = []
    for dtype, shape in ((torch.bfloat16, (m, heads, k)), (torch.float8_e4m3fn, (heads, n, k)),
                         (torch.float32, ()), (torch.bfloat16, (m, heads, n))):
        count = 1
        for dim in shape:
            count *= dim
        t = torch.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape).clone()
        offset += count * t.element_size()
        tensors.append(t)
    if not all(bool(torch.isfinite(t.float()).all()) for t in tensors) or not bool(tensors[2] > 0):
        raise ValueError("nonfinite MLA replay or nonpositive scale")
    return (m, heads, n, k), tuple(tensors), hashlib.sha256(raw).hexdigest()


def check_mla_weights(args, version):
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
    from vllm.model_executor.layers.attention.mla_attention import dynamic_per_batched_tensor_quant

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None or args.tp < 1:
        raise ValueError("requires pinned vLLM, original checkpoint and loaded TP inventory")
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    if len(inventory["ranks"]) != args.tp or cfg["num_attention_heads"] % args.tp:
        raise ValueError("loaded inventory TP mismatch")
    heads = cfg["num_attention_heads"] // args.tp
    dk, qn, vd = cfg["kv_lora_rank"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    rows = []
    for layer in range(cfg["num_hidden_layers"]):
        path = args.capture / f"model-mla-tp{args.tp}-{layer:05d}.safetensors"
        prefix = f"model.layers.{layer}.self_attn.derived.mla_fp8_tp{args.tp}."
        with safe_open(path, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != {prefix + p + s for p in ("wk", "wv") for s in (".weight", ".weight_scale")}:
                raise ValueError("MLA overlay must contain exactly the two weights and TP scales")
            for rank in range(args.tp):
                original, scales = mla_kvb_weights(args.checkpoint, layer, rank, args.tp)
                dequant = scaled_dequantize(original.cuda(), scales.cuda(), group_shape=[128, 128], out_dtype=torch.bfloat16)
                uk, uv = dequant.T.reshape(dk, heads, qn + vd).split([qn, vd], dim=-1)
                loaded = inventory["ranks"][rank]["modules"][f"model.layers.{layer}.self_attn.mla_attn.mla_attn"]["tensors"]
                for tag, value, ref_tag in (("wk", uk.transpose(0, 1), "W_K"), ("wv", uv.permute(1, 2, 0), "W_V")):
                    want, scale = dynamic_per_batched_tensor_quant(value, dtype=torch.float8_e4m3fn)
                    view = shard.get_slice(prefix + tag + ".weight")
                    if (view.get_shape() != [cfg["num_attention_heads"], *want.shape[1:]]
                            or loaded[ref_tag]["shape"] != list(want.shape)
                            or loaded[ref_tag]["dtype"] != str(want.dtype)
                            or loaded[ref_tag + "_scale"]["shape"] != []):
                        raise ValueError("MLA overlay or loaded reference geometry mismatch")
                    got = view[rank * heads:(rank + 1) * heads]
                    stored_scale = shard.get_tensor(prefix + tag + ".weight_scale")
                    if (got.dtype != torch.float8_e4m3fn or stored_scale.dtype != torch.float32
                            or stored_scale.shape != (args.tp, 1)):
                        raise ValueError("MLA overlay dtype or scale geometry mismatch")
                    scale_got = stored_scale[rank, 0]
                    wh, sh = tensor_digest(want), tensor_digest(scale)
                    gh, gsh = tensor_digest(got), tensor_digest(scale_got)
                    rows.append(dict(layer=layer, rank=rank, projection=tag, weight_sha256=gh, scale_sha256=gsh,
                        reference_weight_sha256=wh, reference_scale_sha256=sh,
                        bitwise=wh == gh and sh == gsh,
                        finite=bool(torch.isfinite(got.float()).all() and torch.isfinite(scale_got) and scale_got > 0)))
        print(f"MLA weight layer {layer}: bitwise={all(r['bitwise'] and r['finite'] for r in rows[-2 * args.tp:])}", flush=True)
    passed = all(r["bitwise"] and r["finite"] for r in rows)
    with args.output.open("x") as f:
        json.dump(dict(scope="full-model derived MLA weight preparation against installed pinned GPU helpers; not serving parity",
            passed=passed, precision_qualified=False, vllm_version=version, tp=args.tp, cases=rows,
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def export_mla(args, rocm_aiter_ops, version):
    from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
    from vllm.model_executor.layers.attention.mla_attention import dynamic_per_batched_tensor_quant
    from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import _get_config

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires pinned vLLM, original checkpoint and loaded precision inventory")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    if args.tp < 1 or len(inventory["ranks"]) != args.tp or cfg["num_attention_heads"] % args.tp:
        raise ValueError("loaded inventory TP mismatch")
    layer, heads = meta["layer"], cfg["num_attention_heads"] // args.tp
    latent, nope, value_dim = cfg["kv_lora_rank"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    qb_reference = None
    if args.reference_json is not None:
        qb_reference = json.loads(args.reference_json.read_text())
        if (not qb_reference["audit_complete"] or qb_reference["vllm_version"] != version
                or qb_reference["layer"] != layer or qb_reference["tp"] != args.tp):
            raise ValueError("Q-B source must match pinned MLA layer/TP")
    cases, weights = [], []
    for rank in range(args.tp):
        original, scales = mla_kvb_weights(args.checkpoint, layer, rank, args.tp)
        dequant = scaled_dequantize(original.cuda(), scales.cuda(), group_shape=[128, 128], out_dtype=torch.bfloat16)
        uk, uv = dequant.T.reshape(latent, heads, nope + value_dim).split([nope, value_dim], dim=-1)
        tensors = inventory["ranks"][rank]["modules"][f"model.layers.{layer}.self_attn.mla_attn.mla_attn"]["tensors"]
        for tag, bf16 in (("W_K", uk.transpose(0, 1)), ("W_V", uv.permute(1, 2, 0))):
            w, scale = dynamic_per_batched_tensor_quant(bf16, dtype=torch.float8_e4m3fn)
            w2, scale2 = dynamic_per_batched_tensor_quant(bf16, dtype=torch.float8_e4m3fn)
            layout, slayout = tensors[tag], tensors[tag + "_scale"]
            if (layout["shape"] != list(w.shape) or layout["stride"] != list(w.stride())
                    or layout["dtype"] != str(w.dtype) or slayout["shape"] != []
                    or slayout["stride"] != [] or slayout["dtype"] != "torch.float32"
                    or tensor_digest(w) != tensor_digest(w2) or tensor_digest(scale) != tensor_digest(scale2)):
                raise ValueError("MLA weight preparation differs from loaded layout or is not repeatable")
            weights.append(dict(rank=rank, projection=tag, original_sha256=tensor_digest(original),
                original_scale_sha256=tensor_digest(scales), dequant_sha256=tensor_digest(bf16),
                weight_sha256=tensor_digest(w), scalar_sha256=tensor_digest(scale), scalar=float(scale)))
            _, n, k = w.shape
            generator = torch.Generator().manual_seed(5300 + rank * 2 + (tag == "W_V"))
            base = torch.randn((128, heads, k), generator=generator).to(torch.bfloat16)
            base[0, 0] = 0
            if heads > 1:
                base[0, 1] *= 1e-12
            for m in (1, 8, 16, 32, 64, 128):
                if qb_reference is not None and tag == "W_K":
                    rows = [c for c in qb_reference["cases"] if c["rank"] == rank and c["shape"][0] == m]
                    if len(rows) != 1:
                        raise ValueError("Q-B source must contain every requested rank/rung exactly once")
                    case = rows[0]
                    shape, _, q, digest = load_case(args.reference_json.parent / case["file"])
                    if (shape != (m, heads * (nope + cfg["qk_rope_head_dim"]), cfg["q_lora_rank"])
                            or digest != case["sha256"] or tensor_digest(q) != case["reference_sha256"]
                            or not case["finite"] or not case["norm_quant_scales_repeat_bitwise"]
                            or not case["isolated_parts_repeat_bitwise"] or not all(case["bf16_addition_order_bounds"].values())):
                        raise ValueError("Q-B source hash/geometry/stability mismatch")
                    x = q.cuda().reshape(m, heads, nope + cfg["qk_rope_head_dim"])[..., :nope]
                else:
                    x = base[:m].contiguous().cuda()
                y = rocm_aiter_ops.triton_fp8_bmm(x.transpose(0, 1), w, scale, group_size=128, transpose_bm=True)
                repeat = torch.empty_like(y)
                rocm_aiter_ops.triton_fp8_bmm(x.transpose(0, 1), w, scale, group_size=128,
                                             transpose_bm=True, YQ=repeat)
                y, repeat = y.cpu(), repeat.cpu()
                dest = args.output.parent / f"rank{rank}.m{m}.{tag}.mla.bin"
                write_mla_case(dest, x, w, scale, y)
                selected, _ = _get_config(m, n, k)
                selected = dict(selected, BLOCK_SIZE_K=128, kpack=1)
                row = dict(rank=rank, projection=tag, shape=[m, heads, n, k], file=dest.name,
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), input_sha256=tensor_digest(x),
                    input_stride=list(x.transpose(0, 1).stride()), output_stride=list(y.stride()),
                    reference_sha256=tensor_digest(y), repeat_sha256=tensor_digest(repeat),
                    output_repeat_bitwise=tensor_digest(y) == tensor_digest(repeat),
                    finite=bool(torch.isfinite(y.float()).all() and torch.isfinite(repeat.float()).all()),
                    selected_config=selected)
                cases.append(row)
                print(json.dumps(row), flush=True)
    complete = all(row["finite"] and row["output_repeat_bitwise"] for row in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="MLA FP8 BMM components; W_K input from independent Q-B when supplied, otherwise seeded; W_V seeded; not full block or serving",
            audit_complete=complete, precision_qualified=False, vllm_version=version, layer=layer, tp=args.tp,
            qb_reference_sha256=None if args.reference_json is None else hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
            weights=weights, cases=cases), f, indent=2, allow_nan=False)
    return 0 if complete else 1


def check_mla(args):
    if args.reference_json is None:
        raise ValueError("requires independent MLA reference")
    audit = json.loads(args.reference_json.read_text())
    if not audit["audit_complete"] or audit["vllm_version"] != "0.29.0" or audit["tp"] < 1:
        raise ValueError("MLA reference is incomplete or not pinned")
    expected = {(r, m, p) for r in range(audit["tp"]) for m in (1, 8, 16, 32, 64, 128) for p in ("W_K", "W_V")}
    keys = [(c["rank"], c["shape"][0], c["projection"]) for c in audit["cases"]]
    weight_keys = [(w["rank"], w["projection"]) for w in audit["weights"]]
    if (set(keys) != expected or len(keys) != len(expected)
            or set(weight_keys) != {(r, p) for r in range(audit["tp"]) for p in ("W_K", "W_V")}
            or len(weight_keys) != audit["tp"] * 2):
        raise ValueError("MLA reference must cover every rank/projection/rung exactly once")
    weights = dict(zip(weight_keys, audit["weights"]))
    qb = None
    qb_path = getattr(args, "qb_reference", None)
    if qb_path is not None:
        if hashlib.sha256(qb_path.read_bytes()).hexdigest() != audit.get("qb_reference_sha256"):
            raise ValueError("Q-B reference does not match MLA input provenance")
        qb = json.loads(qb_path.read_text())
        if not qb["audit_complete"] or qb["tp"] != audit["tp"]:
            raise ValueError("Q-B reference is incomplete or has different TP")
    records = []
    for case in audit["cases"]:
        shape, (x, w, scale, want), digest = load_mla_case(args.reference_json.parent / case["file"])
        weight = weights[(case["rank"], case["projection"])]
        if (list(shape) != case["shape"] or digest != case["sha256"]
                or tensor_digest(x) != case["input_sha256"] or tensor_digest(want) != case["reference_sha256"]
                or case["reference_sha256"] != case["repeat_sha256"]
                or tensor_digest(w) != weight["weight_sha256"] or tensor_digest(scale) != weight["scalar_sha256"]
                or not case["finite"] or not case["output_repeat_bitwise"]):
            raise ValueError("MLA reference artifact hash/geometry/stability mismatch")
        path = args.capture / "outputs" / f"rank{case['rank']}.m{shape[0]}.{case['projection']}.bf16"
        got, got_hash = load_tensor(path, torch.bfloat16, want.shape)
        finite = bool(torch.isfinite(got.float()).all())
        error = (got.double() - want.double()).norm(dim=-1) / want.double().norm(dim=-1).clamp_min(1e-30)
        row = dict(rank=case["rank"], projection=case["projection"], shape=list(shape), sha256=got_hash,
            finite=finite, bitwise=got_hash == case["reference_sha256"],
            max_row_rel_l2=float(error.max()) if finite else None)
        if qb is not None and case["projection"] == "W_K":
            sources = [c for c in qb["cases"] if c["rank"] == case["rank"] and c["shape"][0] == shape[0]]
            if len(sources) != 1:
                raise ValueError("Q-B provenance must contain exactly one matching rank/rung")
            source = sources[0]
            qshape, _, q, qhash = load_case(qb_path.parent / source["file"])
            m, heads, _, k = shape
            if (qhash != source["sha256"] or tensor_digest(q) != source["reference_sha256"]
                    or qshape[:2] != (m, heads * (k + 64))):
                raise ValueError("Q-B provenance hash or geometry mismatch")
            q = q.reshape(m, heads, k + 64)
            if tensor_digest(q[..., :k]) != case["input_sha256"]:
                raise ValueError("MLA input is not the captured Q-B no-RoPE slice")
            rope, rope_hash = load_tensor(Path(str(path) + ".rope"), torch.bfloat16, (m, heads, 64))
            row["rope_sha256"] = rope_hash
            row["rope_bitwise"] = rope_hash == tensor_digest(q[..., k:])
        records.append(row)
    passed = all(r["finite"] and r["bitwise"] and r.get("rope_bitwise", True) for r in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="native MLA BMM replay against pinned vLLM; seeded component inputs, not full-model parity",
            precision_qualified=False, passed=passed, cases=records,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), f, indent=2, allow_nan=False)
    print(f"validated {len(records)} MLA component cases; bitwise={passed}", flush=True)
    return 0 if passed else 1


def export_qkva(args, rocm_aiter_ops, version):
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires pinned vLLM, original checkpoint and loaded precision inventory")
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    config = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    layer = metadata["layer"]
    weight, scale, gamma = qkva_weights(args.checkpoint, layer)
    n, k = weight.shape
    name = f"model.layers.{layer}.self_attn.fused_qkv_a_proj"
    layouts = [rank["modules"][name] for rank in inventory["ranks"]]
    if len(layouts) != args.tp:
        raise ValueError("loaded inventory TP mismatch")
    for module in layouts:
        tensors = module["tensors"]
        selected = module["attributes"]["quant_method"]["fields"]["fp8_linear"]
        if (tensors["weight"]["shape"] != [n, k] or tensors["weight"]["dtype"] != "torch.float8_e4m3fn"
                or tensors["weight_scale_inv"]["shape"] != list(scale.shape)
                or tensors["weight_scale_inv"]["dtype"] != "torch.float32"
                or selected["class_name"].rsplit(".", 1)[-1] != "AiterFp8BlockScaledMMKernel"
                or selected["fields"]["use_triton"] is not False
                or tensors["weight"]["stride"] != layouts[0]["tensors"]["weight"]["stride"]):
            raise ValueError("fused projection differs from loaded pinned backend/geometry")
    stride = layouts[0]["tensors"]["weight"]["stride"]
    if stride[1] != 1 or stride[0] < k:
        raise ValueError("unsupported loaded weight stride")
    gpu_weight = torch.empty((n, stride[0]), dtype=weight.dtype, device="cuda")[:, :k]
    gpu_weight.copy_(weight)
    gpu_scale, gpu_gamma = scale.cuda(), gamma.cuda()
    x, input_hash = load_tensor(args.capture / "inputs/act.x.bin", torch.bfloat16, (metadata["batch"], k))
    cases = []
    for m in (1, 8, 16, 32, 64, 128):
        source = x.repeat(((m + x.shape[0] - 1) // x.shape[0], 1))[:m].contiguous().cuda()
        def run():
            norm = rocm_aiter_ops.rms_norm(source, gpu_gamma, config["rms_norm_eps"])
            quant, asc = rocm_aiter_ops.group_fp8_quant(norm, 128)
            out = rocm_aiter_ops.gemm_a8w8_blockscale(quant, gpu_weight, asc, gpu_scale,
                                                    [128, 128], output_dtype=torch.bfloat16)
            return tuple(t.cpu() for t in (norm, quant, asc, out))
        values, repeats = run(), run()
        stable = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(values, repeats)]
        finite = all(bool(torch.isfinite(t.float()).all()) for t in (*values, *repeats))
        tuned = get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
        dest = args.output.parent / f"m{m}.qkva.bin"
        write_case(dest, values[1].view(torch.uint8), weight, values[2].T.contiguous(), scale, values[3])
        for tag, value in (("norm", values[0]), ("repeat", repeats[3])):
            with (args.output.parent / f"m{m}.{tag}.bf16").open("xb") as f:
                f.write(value.contiguous().view(torch.uint8).numpy().tobytes())
        row = dict(shape=[m, n, k], file=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            finite=finite, norm_quant_scales_repeat_bitwise=all(stable[:3]), output_repeat_bitwise=stable[3],
            reference_sha256=tensor_digest(values[3]), repeat_sha256=tensor_digest(repeats[3]),
            reference_repeat_max_row_rel_l2=float(((values[3].double() - repeats[3].double()).norm(dim=1)
                / values[3].double().norm(dim=1).clamp_min(1e-30)).max()),
            selected_config=None if tuned is None else {key: str(value) for key, value in tuned.items()},
            loaded_weight_stride=stride, normalized_input_sha256=tensor_digest(values[0]))
        cases.append(row)
        print(json.dumps(row), flush=True)
    complete = all(row["finite"] and row["norm_quant_scales_repeat_bitwise"] for row in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="fused QKV-A component reference; fixture rows repeated above original B; not full block or serving",
            audit_complete=complete, precision_qualified=False, vllm_version=version, layer=layer,
            input_sha256=input_hash, source_batch=metadata["batch"], weight_sha256=tensor_digest(weight),
            scale_sha256=tensor_digest(scale), norm_weight_sha256=tensor_digest(gamma),
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(), cases=cases), f, indent=2)
    return 0 if complete else 1


def check_qkva(args):
    if args.reference_json is None or args.checkpoint is None or args.tp < 1:
        raise ValueError("requires independent QKV-A reference, checkpoint geometry and positive TP")
    audit = json.loads(args.reference_json.read_text())
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    m, layer = meta["batch"], meta["layer"]
    ql, dk, dr = cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
    n, k = ql + dk + dr, cfg["hidden_size"]
    cases = [c for c in audit["cases"] if c["shape"] == [m, n, k]]
    if (not audit["audit_complete"] or audit["vllm_version"] != "0.29.0" or len(cases) != 1
            or audit["layer"] != layer or audit["source_batch"] != m
            or hashlib.sha256((args.capture / "inputs/act.x.bin").read_bytes()).hexdigest() != audit["input_sha256"]):
        raise ValueError("independent QKV-A reference input/geometry does not match this block")
    case = cases[0]
    if not all(case[key] for key in ("finite", "norm_quant_scales_repeat_bitwise", "output_repeat_bitwise")):
        raise ValueError("QKV-A reference boundaries are not finite and stable")
    shape, (xq, weight, xs, ws), reference, digest = load_case(args.reference_json.parent / case["file"])
    norm, norm_hash = load_tensor(args.reference_json.parent / f"m{m}.norm.bf16", torch.bfloat16, (m, k))
    if (digest != case["sha256"] or norm_hash != case["normalized_input_sha256"]
            or tensor_digest(reference) != case["reference_sha256"]
            or tensor_digest(weight) != audit["weight_sha256"] or tensor_digest(ws) != audit["scale_sha256"]):
        raise ValueError("QKV-A reference artifact hash mismatch")
    weights = f"model.layers.{layer}.self_attn.fused_qkv_a_proj"
    expected = {"act.xn": norm, "act.qkva_xq": xq, "act.qkva_xs": xs.T.contiguous(),
        weights + ".weight_fp8": weight, weights + ".weight_scale_inv": ws,
        "act.qlr": reference[:, :ql], "act.ckvraw": reference[:, ql:ql + dk],
        "act.krr": reference[:, ql + dk:]}
    records = []
    for rank in range(args.tp):
        boundaries = {}
        for name, want in expected.items():
            value, value_hash = load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", want.dtype, want.shape)
            finite = bool(torch.isfinite(value.float()).all() and torch.isfinite(want.float()).all())
            bitwise = torch.equal(value.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
            rel = ((value.double() - want.double()).norm(dim=1) / want.double().norm(dim=1).clamp_min(1e-30)).max()
            boundaries[name] = dict(finite=finite, bitwise=bitwise, sha256=value_hash,
                                   max_row_rel_l2=float(rel))
        record = dict(rank=rank, boundaries=boundaries,
                      passed=all(b["finite"] and b["bitwise"] for b in boundaries.values()))
        records.append(record)
        print(json.dumps(record), flush=True)
    passed = all(record["passed"] for record in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="independent QKV-A input-norm/quant/GEMM block boundaries; not full-model parity",
            precision_qualified=False, passed=passed, cases=records,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def compare_packet_mla(args, rocm_aiter_ops, version):
    from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
    from vllm.model_executor.layers.attention.mla_attention import dynamic_per_batched_tensor_quant

    if (version != "0.29.0" or args.tp != 8 or args.checkpoint is None
            or args.precision_inventory is None or args.qb_reference is None):
        raise ValueError("requires pinned TP8 MLA inventory, original checkpoint and independent Q-B reference")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    audit = json.loads(args.qb_reference.read_text())
    m, layer = meta["batch"], meta["layer"]
    if (not audit["audit_complete"] or audit["vllm_version"] != version
            or audit["layer"] != layer or audit["tp"] != args.tp or len(inventory["ranks"]) != args.tp
            or audit["loaded_inventory_sha256"] != hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest()
            or (cfg["num_attention_heads"], cfg["kv_lora_rank"], cfg["qk_nope_head_dim"],
                cfg["qk_rope_head_dim"], cfg["v_head_dim"], cfg["q_lora_rank"]) != (64, 512, 192, 64, 256, 2048)):
        raise ValueError("independent Q-B reference or MLA geometry mismatch")
    records = []
    for rank in range(args.tp):
        prefix = f"model.layers.{layer}.self_attn."
        boundaries = {}
        def captured(name, dtype, shape):
            return load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape)[0]
        def exact(name, want):
            got = captured(name, want.dtype, want.shape)
            boundaries[name] = dict(sha256=tensor_digest(got), reference_sha256=tensor_digest(want),
                finite=bool(torch.isfinite(got.float()).all() and torch.isfinite(want.float()).all()),
                bitwise=tensor_digest(got) == tensor_digest(want))
            return got
        cases = [c for c in audit["cases"] if c["rank"] == rank and c["shape"] == [m, 2048, 2048]]
        if len(cases) != 1:
            raise ValueError("Q-B reference must contain this rank/rung exactly once")
        case = cases[0]
        shape, (xq, weight, xs, ws), ref_qb, digest = load_case(args.qb_reference.parent / case["file"])
        norm, norm_hash = load_tensor(args.qb_reference.parent / f"rank{rank}.m{m}.norm.bf16",
                                      torch.bfloat16, (m, 2048))
        if (digest != case["sha256"] or norm_hash != case["norm_sha256"]
                or tensor_digest(ref_qb) != case["reference_sha256"]
                or not case["finite"] or not case["norm_quant_scales_repeat_bitwise"]
                or not case["isolated_parts_repeat_bitwise"] or not all(case["bf16_addition_order_bounds"].values())):
            raise ValueError("Q-B reference hash/stability mismatch")
        original, original_s, gamma = qb_weights(args.checkpoint, layer, rank, args.tp)
        if tensor_digest(original) != tensor_digest(weight) or tensor_digest(original_s) != tensor_digest(ws):
            raise ValueError("Q-B reference is not the original checkpoint")
        for name, want in [("act.qlat", norm), ("act.qb_xq", xq), ("act.qb_xs", xs.T.contiguous()),
            (prefix + "q_b_proj.weight", weight), (prefix + "q_b_proj.weight_scale_inv", ws),
            (prefix + "q_a_layernorm.weight", gamma)]:
            exact(name, want)
        parts = []
        for part in case["parts"]:
            _, (aq, wq, asc, wsc), value, part_hash = load_case(args.qb_reference.parent / part["file"])
            lo, hi = part["k_start"], part["k_end"]
            if (part_hash != part["sha256"] or tensor_digest(value) != part["reference_sha256"]
                    or tensor_digest(aq) != tensor_digest(xq[:, lo:hi])
                    or tensor_digest(wq) != tensor_digest(weight[:, lo:hi])
                    or tensor_digest(asc) != tensor_digest(xs[:, lo // 128:hi // 128])
                    or tensor_digest(wsc) != tensor_digest(ws[:, lo // 128:hi // 128])):
                raise ValueError("isolated Q-B partition hash/operands mismatch")
            parts.append(value)
        if len(parts) != case["split_count"]:
            raise ValueError("incomplete Q-B partition set")
        lower, upper = bf16_order_bounds(torch.stack(parts, dim=1))
        qb = captured("act.qb", torch.bfloat16, (m, 2048))
        bounded = bool(torch.isfinite(qb).all() and ((qb >= lower) & (qb <= upper)).all())
        boundaries["act.qb"] = dict(sha256=tensor_digest(qb), finite=bool(torch.isfinite(qb).all()),
            exact_bf16_addition_order_bounds=bounded, reference_bitwise=tensor_digest(qb) == tensor_digest(ref_qb))
        q = qb.reshape(m, 8, 256).cuda()
        exact("act.qrr", qb.reshape(m, 8, 256)[..., 192:])
        original, scales = mla_kvb_weights(args.checkpoint, layer, rank, args.tp)
        dequant = scaled_dequantize(original.cuda(), scales.cuda(), group_shape=[128, 128], out_dtype=torch.bfloat16)
        uk, uv = dequant.T.reshape(512, 8, 448).split([192, 256], dim=-1)
        loaded = inventory["ranks"][rank]["modules"][prefix + "mla_attn.mla_attn"]["tensors"]
        for tag, name, bf16, x in (("W_K", "qa", uk.transpose(0, 1), q[..., :192]),
            ("W_V", "oat", uv.permute(1, 2, 0), captured("act.olat", torch.bfloat16, (m, 8, 512)).cuda())):
            w, scale = dynamic_per_batched_tensor_quant(bf16, dtype=torch.float8_e4m3fn)
            if (loaded[tag]["shape"] != list(w.shape) or loaded[tag]["stride"] != list(w.stride())
                    or loaded[tag]["dtype"] != str(w.dtype) or loaded[tag + "_scale"]["shape"] != []
                    or loaded[tag + "_scale"]["dtype"] != "torch.float32"):
                raise ValueError("MLA weight contract differs from loaded pinned inventory")
            stem = prefix + "derived.mla_fp8_tp8." + ("wk" if tag == "W_K" else "wv")
            exact(stem + ".weight", w.cpu())
            exact(stem + ".weight_scale", scale.cpu().reshape(1))
            expected = rocm_aiter_ops.triton_fp8_bmm(x.transpose(0, 1), w, scale, group_size=128, transpose_bm=True)
            repeat = rocm_aiter_ops.triton_fp8_bmm(x.transpose(0, 1), w, scale, group_size=128, transpose_bm=True)
            exact("act." + name, expected.cpu())
            boundaries["act." + name].update(reference_repeat_bitwise=tensor_digest(expected) == tensor_digest(repeat),
                input_sha256=tensor_digest(x), input_finite=bool(torch.isfinite(x).all()))
        passed = all(v["finite"] and v.get("bitwise", v.get("exact_bf16_addition_order_bounds", False))
                     and v.get("reference_repeat_bitwise", True) and v.get("input_finite", True)
                     for v in boundaries.values())
        record = dict(rank=rank, shape=[m, 8], passed=passed, boundaries=boundaries)
        records.append(record)
        print(json.dumps(record), flush=True)
    passed = all(row["passed"] for row in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="connected Q-A norm/quant/Q-B exact atomic bounds; query/value BMM conditioned on captured BF16 inputs; attention merge and full-model parity not qualified",
            passed=passed, audit_complete=passed, precision_qualified=False, vllm_version=version,
            qb_reference_sha256=hashlib.sha256(args.qb_reference.read_bytes()).hexdigest(),
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
            cases=records), f, indent=2, allow_nan=False)
    return 0 if passed else 1


def sparse_decode_indices(indices, ctx):
    if (indices.dtype != torch.int32 or indices.ndim != 2 or not indices.shape[0]
            or indices.shape[1] != 2048 or ctx <= 0):
        raise ValueError("requires nonempty int32 decode indices with topk2048")
    count = min(ctx, indices.shape[1])
    selected = indices[:, :count]
    if (bool((selected < 0).any() or (selected >= ctx).any())
            or bool((indices[:, count:] != -1).any())
            or any(len(set(row)) != count for row in selected.tolist())):
        raise ValueError("invalid selected-key prefix or padding")
    return (selected + torch.arange(indices.shape[0], dtype=torch.int32)[:, None] * ctx).flatten()


def boundary_difference(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.ndim < 2:
        raise ValueError("boundary shape/dtype mismatch")
    a, b = actual.float().cpu().flatten(1), expected.float().cpu().flatten(1)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    difference = (a - b).double()
    norms = b.double().norm(dim=1)
    relative = difference.norm(dim=1) / norms.clamp_min(torch.finfo(torch.float64).tiny)
    return dict(sha256=tensor_digest(actual), reference_sha256=tensor_digest(expected), finite=finite,
                bitwise=tensor_digest(actual) == tensor_digest(expected),
                mismatch_count=int((a != b).sum()), elements=a.numel(),
                max_abs=float(difference.abs().max()) if finite else None,
                max_row_rel_l2=float(relative.max()) if finite and bool(torch.isfinite(relative).all()) else None)


def split_attention_rounding_model(query, kv, indices, splits, scale, bf16_probability,
                                   split_ends=None, tile=None):
    if (query.ndim != 3 or kv.ndim != 3 or indices.ndim != 2
            or query.shape[0] != kv.shape[0] or indices.shape[0] != kv.shape[0]
            or query.shape[-1] != kv.shape[-1] or query.shape[-1] <= 64
            or not 1 <= splits <= indices.shape[1] or not 0 < scale <= 1
            or bool((indices < 0).any() or (indices >= kv.shape[1]).any())):
        raise ValueError("invalid attention rounding-model geometry")
    selected = kv.double().gather(1, indices.long()[..., None].expand(-1, -1, kv.shape[-1]))
    scores = query.double() @ selected.transpose(1, 2) * scale
    global_max = scores.amax(dim=-1, keepdim=True)
    numerator = torch.zeros((*query.shape[:2], query.shape[-1] - 64), dtype=torch.float64, device=query.device)
    denominator = torch.zeros((*query.shape[:2], 1), dtype=torch.float64, device=query.device)
    width = (indices.shape[1] + splits - 1) // splits
    ends = list(split_ends) if split_ends is not None else list(range(width, indices.shape[1], width)) + [indices.shape[1]]
    if (not ends or ends[-1] != indices.shape[1] or any(a >= b for a, b in zip([0] + ends, ends))
            or (tile is not None and tile <= 0)):
        raise ValueError("invalid attention rounding-model partitions")
    start = 0
    for end in ends:
        maximum = torch.full_like(global_max, -torch.inf)
        part_num = torch.zeros_like(numerator)
        part_den = torch.zeros_like(denominator)
        step = tile or (end - start)
        for lo in range(start, end, step):
            hi = min(end, lo + step)
            score = scores[..., lo:hi]
            new_maximum = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
            correction = (maximum - new_maximum).exp()
            probability = (score - new_maximum).exp()
            part_den = part_den * correction + probability.sum(dim=-1, keepdim=True)
            if bf16_probability:
                probability = probability.bfloat16().double()
            part_num = part_num * correction + probability @ selected[:, lo:hi, :-64]
            maximum = new_maximum
        correction = (maximum - global_max).exp()
        denominator += part_den * correction
        numerator += part_num * correction
        start = end
    return (numerator / denominator).bfloat16()


def export_attention_ps_case(path, query, kv, md, reference, scale, kv_lengths=None):
    import aiter
    from aiter.mla import _use_persistent_mla_decode
    from aiter.ops.attention import get_mla_decode_fwd_max_splits

    m, heads, width = query.shape
    if (heads != 16 or width != 576 or query.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16
            or not _use_persistent_mla_decode(m, heads, 1, query.dtype, kv.dtype)):
        raise ValueError("persistent replay must match the installed serving route")
    info = md.work_info_set.cpu()
    workptr = md.work_indptr.cpu()
    nwork = int(workptr[-1])
    lengths = torch.tensor(kv_lengths if kv_lengths is not None else [md.max_seq_len] * m, dtype=torch.int32)
    if lengths.shape != (m,) or bool((lengths <= 0).any() or (lengths > md.max_seq_len).any()):
        raise ValueError("invalid persistent replay KV lengths")
    csr = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.clamp_max(md.topk_tokens).cumsum(0))).int()
    if not torch.equal(md.paged_kv_indptr.cpu(), csr):
        raise ValueError("persistent replay KV lengths differ from serving CSR")
    if (nwork <= 0 or nwork > info.shape[0] or info.shape[1] != 8
            or md.reduce_partial_map.numel() != info.shape[0]):
        raise ValueError("invalid persistent replay work geometry")
    partial_slots = info[:nwork, 1]
    partial_slots = partial_slots[partial_slots >= 0]
    npartial = partial_slots.numel()
    if bool((info[:nwork, 1] < -1).any()) or not torch.equal(partial_slots, torch.arange(npartial)):
        raise ValueError("persistent replay requires contiguous split-output slots")
    part = torch.empty((info.shape[0], 1, heads, 512), dtype=torch.float32, device="cuda")
    lse = torch.empty((info.shape[0], 1, heads, 1), dtype=torch.float32, device="cuda")
    out = torch.empty((m, heads, 512), dtype=torch.bfloat16, device="cuda")
    max_splits = get_mla_decode_fwd_max_splits(heads, 1, query.dtype, kv.dtype)
    def run():
        part.view(torch.int32).fill_(-1)
        lse.view(torch.int32).fill_(-1)
        out.view(torch.int16).fill_(-1)
        aiter.mla_decode_stage1_asm_fwd(query, kv.reshape(-1, 1, 1, 576), md.qo_indptr,
            md.paged_kv_indptr, md.paged_kv_indices, md.paged_kv_last_page_len,
            None, md.work_meta_data, md.work_indptr, md.work_info_set, 1, 1, 1, scale,
            part, lse, out, None, None, None, None, 1, 0, None, 0)
        aiter.mla_reduce_v1(part, lse, md.reduce_indptr, md.reduce_final_map,
            md.reduce_partial_map, 1, max_splits, out, None)
        return part[:nwork].cpu().clone(), lse[:nwork].cpu().clone(), out.cpu().clone()
    actual_part, actual_lse, actual_out = run()
    again_part, again_lse, again_out = run()
    finite = all(bool(torch.isfinite(t).all()) for t in (actual_part[:npartial], actual_lse[:npartial], actual_out))
    stable = all(tensor_digest(a) == tensor_digest(b) for a, b in
                 ((actual_part, again_part), (actual_lse, again_lse), (actual_out, again_out)))
    matches = tensor_digest(actual_out) == tensor_digest(reference)
    if not finite or not stable or not matches:
        raise ValueError("direct persistent stage1/reduce must match serving reference and repeat bitwise")
    with path.open("xb") as f:
        f.write(struct.pack("<7I", 0x41505332 if kv_lengths is not None else 0x41505331, m, md.max_seq_len, md.topk_tokens,
                            nwork, workptr.numel() - 1, info.shape[0]))
        if kv_lengths is not None:
            f.write(lengths.numpy().tobytes())
        for tensor in (query, kv, md.paged_kv_indices[:int(csr[-1])], workptr, info,
                       actual_part, actual_lse):
            f.write(tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    reduce_path = path.with_suffix(".reduce.bin")
    with reduce_path.open("xb") as f:
        f.write(struct.pack("<6I", 0x41505231, m, nwork, info.shape[0],
                            md.reduce_indptr.numel() - 1, max_splits))
        for tensor in (md.reduce_indptr, md.reduce_final_map, md.reduce_partial_map, actual_out):
            f.write(tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reduce_file=reduce_path.name, reduce_sha256=hashlib.sha256(reduce_path.read_bytes()).hexdigest(),
        shape=[m, md.max_seq_len, heads, nwork], finite=finite, repeat_bitwise=stable,
        serving_reference_bitwise=matches, part_sha256=tensor_digest(actual_part),
        lse_sha256=tensor_digest(actual_lse), partial_slots=npartial, kv_lengths=lengths.tolist())


def export_attention_sweep(args, version):
    import inspect
    from types import SimpleNamespace
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl, ROCMAiterMLASparseMetadata, ROCMAiterMLASparseMetadataBuilder)

    if version != "0.29.0" or args.tp != 8 or args.checkpoint is None:
        raise ValueError("requires pinned vLLM 0.29 TP8 model geometry")
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    if tuple(cfg[k] for k in ("num_attention_heads", "kv_lora_rank", "qk_nope_head_dim",
                             "qk_rope_head_dim", "index_topk")) != (64, 512, 192, 64, 2048):
        raise ValueError("unexpected GLM sparse attention geometry")
    source = Path(inspect.getfile(ROCMAiterMLASparseImpl))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != "187d72a2ecbf0c845950454dbc3cda2c7eff022568a035ebb57ab5b8b98b132c":
        raise ValueError("installed sparse backend differs from pinned reference")
    units = current_platform.num_compute_units()
    if units != 256 or AiterMLAHelper.get_actual_mla_num_heads(8) != 16:
        raise ValueError("requires gfx950 256-CU padded16-head reference")
    cases = []
    shapes = [(1, 512, None), (31, 512, None)] + [(m, ctx, None) for m in (8, 16) for ctx in (2048, 8192, 71680)]
    shapes += [(m, ctx, None) for m, ctx in ((1, 16), (8, 16), (16, 128), (31, 128), (8, 256))]
    shapes += [(8, 2048, [1, 15, 16, 127, 128, 129, 512, 2048])]
    shapes += [(16, ctx, [1, 16, 127, 128, 129, 512, 2048, ctx] * 2) for ctx in (8192, 71680)]
    for case, (m, ctx, ragged) in enumerate(shapes):
        generator = torch.Generator(device="cuda").manual_seed(57300 + case)
        qa = (torch.randn((m, 8, 512), generator=generator, device="cuda") * 0.25).bfloat16()
        qr = (torch.randn((m, 8, 64), generator=generator, device="cuda") * 0.25).bfloat16()
        kv = (torch.randn((m * ctx // 16, 16, 576), generator=generator, device="cuda") * 0.25).bfloat16()
        lengths = ragged or [ctx] * m
        live = torch.tensor([min(length, 2048) for length in lengths], dtype=torch.int32, device="cuda")
        indices = torch.full((m, 2048), -1, dtype=torch.int32, device="cuda")
        for b in range(m):
            selected = min(lengths[b], 2048)
            indices[b, :selected] = torch.randperm(lengths[b], generator=generator, device="cuda")[:selected].int()
        qo = torch.arange(m + 1, dtype=torch.int32, device="cuda")
        indptr = torch.cat((torch.zeros(1, dtype=torch.int32, device="cuda"), live.cumsum(0))).int()
        last = torch.ones(m, dtype=torch.int32, device="cuda")
        work, workptr, info, reduceptr, final, partial = [
            torch.empty(size, dtype=dtype, device="cuda") for size, dtype in
            get_mla_metadata_info_v1(m, 1, 16, torch.bfloat16, torch.bfloat16, is_sparse=True, fast_mode=True)]
        split_cap = ROCMAiterMLASparseMetadataBuilder._sparse_decode_max_split(
            SimpleNamespace(topk_tokens=2048, _num_compute_units=units), ctx)
        get_mla_metadata_v1(qo, indptr, last, 16, 1, True, work, info, workptr,
            reduceptr, final, partial, page_size=1, kv_granularity=16, max_seqlen_qo=1,
            uni_seqlen_qo=1, fast_mode=True, max_split_per_batch=split_cap)
        impl = ROCMAiterMLASparseImpl.__new__(ROCMAiterMLASparseImpl)
        impl.num_heads, impl.kv_lora_rank, impl.kv_cache_dtype = 8, 512, "auto"
        impl.scale = 0.0625
        impl.q_concat_buffer = torch.empty((m, 8, 576), dtype=torch.bfloat16, device="cuda")
        impl.topk_indices_buffer = indices
        md = ROCMAiterMLASparseMetadata(num_reqs=m, max_query_len=1, max_seq_len=ctx, num_actual_tokens=m,
            query_start_loc=qo, slot_mapping=torch.arange(m, device="cuda") * ctx + torch.tensor(lengths, device="cuda") - 1,
            block_table=torch.arange(m * ctx // 16, dtype=torch.int32, device="cuda").reshape(m, -1),
            req_id_per_token=torch.arange(m, dtype=torch.int32, device="cuda"),
            qo_indptr=qo, paged_kv_last_page_len=last,
            paged_kv_indices=torch.zeros(m * 2048, dtype=torch.int32, device="cuda"), paged_kv_indptr=indptr,
            attn_out_dtype=torch.bfloat16, block_size=16, topk_tokens=2048, num_decodes=m,
            num_decode_tokens=m, work_meta_data=work, work_indptr=workptr, work_info_set=info,
            reduce_indptr=reduceptr, reduce_final_map=final, reduce_partial_map=partial)
        layer = SimpleNamespace(_q_scale=torch.ones((), device="cuda"), _k_scale=torch.ones((), device="cuda"))
        expected, _ = impl.forward_mqa((qa, qr), kv, md, layer)
        repeat, _ = impl.forward_mqa((qa, qr), kv, md, layer)
        if tensor_digest(expected) != tensor_digest(repeat):
            raise ValueError("serving attention reference is not repeat-bitwise")
        padded = AiterMLAHelper.get_mla_padded_q(8, torch.cat((qa, qr), dim=-1))
        suffix = ".ragged" if ragged else ""
        exported = export_attention_ps_case(args.output.parent / f"m{m}.ctx{ctx}{suffix}.attention-ps.bin",
            padded, kv, md, expected.repeat_interleave(2, dim=1), impl.scale, kv_lengths=ragged)
        exported.update(seed=57300 + case, max_split_per_batch=split_cap,
                        actual_splits=(reduceptr[1:] - reduceptr[:-1]).cpu().tolist())
        cases.append(exported)
        print(json.dumps(exported), flush=True)
    with args.output.open("x") as f:
        json.dump(dict(scope="synthetic BF16 Q/KV and unique permuted selected indices; installed serving attention only; no model quality or performance qualification",
            precision_qualified=False, audit_complete=True, persistent_exports_complete=True,
            vllm_version=version, backend_sha256=source_hash, cases=cases), f, indent=2, allow_nan=False)
    return 0


def compare_packet_attention(args, rocm_aiter_ops, version):
    import inspect
    from types import SimpleNamespace
    from safetensors import safe_open
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl, ROCMAiterMLASparseMetadata, ROCMAiterMLASparseMetadataBuilder)

    if version != "0.29.0" or args.tp != 8 or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires pinned TP8 sparse-MLA inventory and original checkpoint")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    m, ctx, layer = meta["batch"], meta["ctx"], meta["layer"]
    if (not 1 <= m <= 64 or ctx <= 0 or len(inventory["ranks"]) != 8
            or (cfg["num_attention_heads"], cfg["kv_lora_rank"], cfg["qk_nope_head_dim"],
                cfg["qk_rope_head_dim"], cfg["index_topk"]) != (64, 512, 192, 64, 2048)
            or cfg["rope_parameters"]["rope_type"] != "default"):
        raise ValueError("unsupported sparse decode geometry or scale")
    sources = {}
    for obj in (ROCMAiterMLASparseImpl, AiterMLAHelper, get_mla_metadata_info_v1,
                get_mla_metadata_v1, rocm_aiter_ops.mla_decode_fwd):
        path = Path(inspect.getfile(obj))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if sources[inspect.getfile(ROCMAiterMLASparseImpl)] != "187d72a2ecbf0c845950454dbc3cda2c7eff022568a035ebb57ab5b8b98b132c":
        raise ValueError("installed sparse backend differs from pinned loaded inventory")
    indices, indices_hash = load_tensor(args.capture / "inputs/act.iidx.bin", torch.int32, (m, 2048))
    expected_indices = sparse_decode_indices(indices, ctx)
    heads = AiterMLAHelper.get_actual_mla_num_heads(8)
    units = current_platform.num_compute_units()
    splits = ROCMAiterMLASparseMetadataBuilder._sparse_decode_max_split(
        SimpleNamespace(topk_tokens=2048, _num_compute_units=units), ctx)
    qo = torch.arange(m + 1, dtype=torch.int32, device="cuda")
    indptr = qo * min(ctx, 2048)
    lastpage = torch.ones(m, dtype=torch.int32, device="cuda")
    work, workptr, info, reduceptr, final, partial = [
        torch.empty(size, dtype=dtype, device="cuda") for size, dtype in
        get_mla_metadata_info_v1(m, 1, heads, torch.bfloat16, torch.bfloat16, is_sparse=True, fast_mode=True)]
    get_mla_metadata_v1(qo, indptr, lastpage, heads, 1, True, work, info, workptr,
                        reduceptr, final, partial, page_size=1, kv_granularity=16,
                        max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=True, max_split_per_batch=splits)
    torch.cuda.synchronize()
    valid_work = int(workptr[-1].item())
    work_rows = info.cpu()[:valid_work].tolist()
    selected_count = min(ctx, 2048)
    work_ranges = []
    for row in range(m):
        ranges = sorted((entry[4] - row * selected_count, entry[5] - row * selected_count)
                        for entry in work_rows if entry[0] == row)
        if (not ranges or ranges[0][0] != 0 or ranges[-1][1] != selected_count
                or any(lo >= hi for lo, hi in ranges)
                or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))):
            raise ValueError("persistent work metadata does not partition the selected keys")
        work_ranges.append(ranges)
    metadata_buffers = {}
    for name, tensor in (("work_indptr", workptr), ("work_info", info), ("reduce_indptr", reduceptr),
                         ("reduce_final_map", final), ("reduce_partial_map", partial)):
        path = args.output.parent / f"{name}.bin"
        raw_metadata = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        with path.open("xb") as f:
            f.write(raw_metadata)
        metadata_buffers[name] = dict(file=path.name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                     sha256=hashlib.sha256(raw_metadata).hexdigest())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    gamma_name = f"model.layers.{layer}.self_attn.kv_a_layernorm.weight"
    with safe_open(args.checkpoint / index[gamma_name], framework="pt", device="cpu") as shard:
        gamma = shard.get_tensor(gamma_name)
    if gamma.dtype != torch.bfloat16 or gamma.shape != (512,):
        raise ValueError("unexpected original KV norm weight")
    records = []
    for rank in range(8):
        loaded = inventory["ranks"][rank]["modules"][f"model.layers.{layer}.self_attn.mla_attn.mla_attn"]
        block_size = loaded["tensors"]["kv_cache"]["shape"][1]
        if (loaded["attributes"]["impl"]["class_name"] != ROCMAiterMLASparseImpl.__module__ + "." + ROCMAiterMLASparseImpl.__name__
                or loaded["attributes"]["kv_cache_dtype"] != "auto"
                or loaded["tensors"]["kv_cache"]["dtype"] != "torch.bfloat16"
                or ctx % block_size):
            raise ValueError("loaded backend/cache layout differs or context not page-aligned")
        def captured(name, dtype, shape):
            return load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape)[0]
        ckv = torch.stack([captured(f"slot{row}.kv.{layer}.ckv", torch.bfloat16, (ctx, 512)) for row in range(m)])
        krot = torch.stack([captured(f"slot{row}.kv.{layer}.krot", torch.bfloat16, (ctx, 64)) for row in range(m)])
        carried = {}
        for name, value, width in (("ckv", ckv, 512), ("krot", krot, 64)):
            original, _ = load_tensor(args.capture / f"inputs/kv.{layer}.{name}.bin", torch.bfloat16, (m, ctx, width))
            carried[name] = tensor_digest(value[:, :-1]) == tensor_digest(original[:, :-1])
        got_gamma = captured(gamma_name, torch.bfloat16, (512,))
        if tensor_digest(gamma) != tensor_digest(got_gamma):
            raise ValueError("captured KV norm weight differs from checkpoint")
        raw = captured("act.ckvraw", torch.bfloat16, (m, 512)).cuda()
        norm = rocm_aiter_ops.rms_norm(raw, gamma.cuda(), cfg["rms_norm_eps"])
        norm_repeat = rocm_aiter_ops.rms_norm(raw, gamma.cuda(), cfg["rms_norm_eps"])
        norm_difference = boundary_difference(ckv[:, -1], norm.cpu())
        norm_difference["reference_repeat_bitwise"] = tensor_digest(norm) == tensor_digest(norm_repeat)
        qa = captured("act.qa", torch.bfloat16, (m, 8, 512)).cuda()
        qr = captured("act.qr", torch.bfloat16, (m, 8, 64)).cuda()
        kv = torch.cat((ckv, krot), dim=-1).reshape(-1, block_size, 576).cuda()
        # Only replace serving allocation/config plumbing; execute the installed forward unchanged.
        impl = ROCMAiterMLASparseImpl.__new__(ROCMAiterMLASparseImpl)
        impl.num_heads, impl.kv_lora_rank, impl.kv_cache_dtype = 8, 512, "auto"
        impl.scale = (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) ** -0.5
        impl.q_concat_buffer = torch.empty((m, 8, 576), dtype=torch.bfloat16, device="cuda")
        impl.topk_indices_buffer = indices.cuda()
        attention_layer = SimpleNamespace(_q_scale=torch.ones((), device="cuda"), _k_scale=torch.ones((), device="cuda"))
        md = ROCMAiterMLASparseMetadata(num_reqs=m, max_query_len=1, max_seq_len=ctx, num_actual_tokens=m,
            query_start_loc=qo, slot_mapping=torch.arange(m, device="cuda") * ctx + ctx - 1,
            block_table=torch.arange(m * ctx // block_size, dtype=torch.int32, device="cuda").reshape(m, -1),
            req_id_per_token=torch.arange(m, dtype=torch.int32, device="cuda"),
            qo_indptr=qo, paged_kv_last_page_len=lastpage,
            paged_kv_indices=torch.zeros(m * 2048, dtype=torch.int32, device="cuda"), paged_kv_indptr=indptr,
            attn_out_dtype=torch.bfloat16, block_size=block_size, topk_tokens=2048, num_decodes=m,
            num_decode_tokens=m, work_meta_data=work, work_indptr=workptr, work_info_set=info,
            reduce_indptr=reduceptr, reduce_final_map=final, reduce_partial_map=partial)
        expected, _ = impl.forward_mqa((qa, qr), kv, md, attention_layer)
        repeat, _ = impl.forward_mqa((qa, qr), kv, md, attention_layer)
        exported = None
        if args.export_attention_ps:
            padded_query = AiterMLAHelper.get_mla_padded_q(8, torch.cat((qa, qr), dim=-1))
            exported = export_attention_ps_case(args.output.parent / f"rank{rank}.attention-ps.bin",
                padded_query, kv, md, expected.repeat_interleave(2, dim=1), impl.scale)
        mapped = md.paged_kv_indices[:expected_indices.numel()].cpu()
        if not torch.equal(mapped, expected_indices):
            raise ValueError("installed selected-key mapping differs from captured logical selection")
        actual = captured("act.olat", torch.bfloat16, (m, 8, 512))
        difference = boundary_difference(actual, expected.cpu())
        difference["reference_repeat_bitwise"] = tensor_digest(expected) == tensor_digest(repeat)
        models = []
        for model_splits in (4, 8):
            if min(ctx, 2048) < model_splits:
                continue
            for rounded in (False, True):
                model = split_attention_rounding_model(torch.cat((qa, qr), dim=-1),
                    kv.reshape(m, ctx, 576), indices[:, :min(ctx, 2048)].cuda(),
                    model_splits, impl.scale, rounded).cpu()
                models.append(dict(splits=model_splits, bf16_probability=rounded,
                    versus_plow=boundary_difference(model, actual),
                    versus_pinned=boundary_difference(model, expected.cpu())))
        if all(ranges == work_ranges[0] for ranges in work_ranges):
            ends = [end for _, end in work_ranges[0]]
            for model_tile in (None, 32):
                model = split_attention_rounding_model(torch.cat((qa, qr), dim=-1),
                    kv.reshape(m, ctx, 576), indices[:, :selected_count].cuda(), len(ends),
                    impl.scale, True, split_ends=ends, tile=model_tile).cpu()
                models.append(dict(split_ends=ends, tile=model_tile, bf16_probability=True,
                    versus_plow=boundary_difference(model, actual),
                    versus_pinned=boundary_difference(model, expected.cpu())))
        inputs_finite = all(bool(torch.isfinite(t).all()) for t in (qa, qr, kv, raw))
        for tag, value in (("attention", expected), ("attention-repeat", repeat), ("kv-norm", norm)):
            with (args.output.parent / f"rank{rank}.{tag}.bf16").open("xb") as f:
                f.write(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        passed = (inputs_finite and all(carried.values()) and difference["finite"] and difference["bitwise"]
                  and difference["reference_repeat_bitwise"] and norm_difference["finite"]
                  and norm_difference["bitwise"] and norm_difference["reference_repeat_bitwise"])
        record = dict(rank=rank, passed=passed, carried_rows_bitwise=carried, inputs_finite=inputs_finite,
            query_sha256=tensor_digest(torch.cat((qa, qr), dim=-1)), kv_sha256=tensor_digest(kv),
            mapped_indices_sha256=tensor_digest(mapped), attention=difference, kv_norm=norm_difference,
            rounding_models=models, persistent_export=exported)
        records.append(record)
        print(json.dumps(record), flush=True)
    passed = all(r["passed"] for r in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="installed persistent sparse-decode attention/merge conditioned on captured BF16 Q and post-write KV; KV norm conditioned on captured raw projection; RoPE/indexer/full-model parity not qualified",
            passed=passed, audit_complete=True, precision_qualified=False, vllm_version=version,
            persistent_exports_complete=all(r["persistent_export"] is not None for r in records),
            shape=[m, ctx, 8, 512, 64], padded_heads=heads, max_split_per_batch=splits, compute_units=units,
            metadata_buffers=metadata_buffers,
            work_ranges=work_ranges,
            rounding_model_scope="diagnostic only: FP64 QK/exp/accumulation; uniform splits plus captured work boundaries with whole-partition or 32-key online tiles; not exact kernel reduction or a qualification oracle",
            selected_indices_sha256=indices_hash, sources=sources,
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(), cases=records),
            f, indent=2, allow_nan=False)
    return 0 if passed else 1


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def describe_call(call):
    import functools

    if isinstance(call, functools.partial):
        return dict(function=describe_call(call.func), keywords={key: describe_value(value)
                    for key, value in call.keywords.items()})
    return getattr(call, "__module__", "") + "." + getattr(call, "__name__", str(call))


def describe_value(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def routed_stage_snapshots(calls, run_1stage):
    if run_1stage:
        return {}
    if [name for name, _ in calls] != ["stage1", "stage2"]:
        raise ValueError("expected exactly two recorded MoE stages")
    result = {}
    for name, call in calls:
        for key, value in (("input", call.args[0]), ("output", call.args[6]),
                           ("scale", call.keywords["a1_scale" if name == "stage1" else "a2_scale"])):
            if not isinstance(value, torch.Tensor):
                raise ValueError("missing recorded MoE boundary tensor")
            result[name + "." + key] = value.detach().cpu().contiguous().clone()
    return result


def unsort_routed_hidden(meta, tokens, slots, gates, hidden, ids, weights):
    m, topk = ids.shape
    experts = (meta.numel() - 1) // 3
    if meta.numel() != 3 * experts + 1 or experts < 1:
        raise ValueError("invalid routed metadata")
    counts = torch.bincount(ids.flatten().long(), minlength=experts)
    tiles = (counts + 63) // 64
    prefix = torch.cat((torch.zeros(1, dtype=torch.int64), tiles.cumsum(0)))
    if (not torch.equal(meta[:experts].long(), prefix[:-1] * 64)
            or not torch.equal(meta[experts:2 * experts].long(), counts)
            or not torch.equal(meta[2 * experts:].long(), prefix)):
        raise ValueError("routed counts or padded offsets disagree with routes")
    end = int(prefix[-1]) * 64
    if end > min(tokens.numel(), slots.numel(), gates.numel(), hidden.shape[0]):
        raise ValueError("routed padded extent exceeds capture")
    result = torch.empty((m * topk, hidden.shape[1]), dtype=hidden.dtype)
    seen = torch.zeros(m * topk, dtype=torch.bool)
    for expert in range(experts):
        begin, count = int(prefix[expert]) * 64, int(counts[expert])
        stop = int(prefix[expert + 1]) * 64
        part = slots[begin:begin + count].long()
        token = tokens[begin:begin + count].long()
        if (bool((part < 0).any() or (part >= m * topk).any())
                or part.unique().numel() != count or bool(seen[part].any())
                or not torch.equal(token, part // topk)
                or not bool((ids.flatten()[part] == expert).all())
                or not torch.equal(gates[begin:begin + count].view(torch.int32),
                                   weights.flatten()[part].view(torch.int32))):
            raise ValueError("routed row maps or gate bits disagree with routes")
        if (not bool((slots[begin + count:stop] == -1).all())
                or not bool((tokens[begin + count:stop] == -1).all())
                or not bool((gates[begin + count:stop] == 0).all())
                or not bool((hidden[begin + count:stop] == 0).all())):
            raise ValueError("routed padding is not initialized")
        result[part] = hidden[begin:begin + count]
        seen[part] = True
    if not bool(seen.all()):
        raise ValueError("incomplete routed row coverage")
    return result.reshape(m, topk, hidden.shape[1])


def native_routed_boundaries(prefix, h, i, experts, ids, gates):
    m, topk = ids.shape
    capacity = Path(str(prefix) + ".act.moe_rowtok.bin").stat().st_size // 4
    if capacity < m * topk:
        raise ValueError("routed scratch cannot hold the active slots")
    tensors, hashes = {}, {}
    for name, dtype, shape in (
        ("routed_xq", torch.uint8, (m, h)),
        ("routed_xs", torch.float32, (h // 128, m)),
        ("routed_hq", torch.uint8, (m, topk, i)),
        ("routed_hs", torch.float32, (i // 128, m * topk)),
        ("moe_meta", torch.int32, (3 * experts + 1,)),
        ("moe_rowtok", torch.int32, (capacity,)),
        ("moe_rowpart", torch.int32, (capacity,)),
        ("moe_rowgate", torch.float32, (capacity,)),
        ("moe_fug", torch.bfloat16, (capacity, i)),
    ):
        tensors[name], hashes[name] = load_tensor(Path(str(prefix) + f".act.{name}.bin"), dtype, shape)
    hidden = unsort_routed_hidden(tensors["moe_meta"], tensors["moe_rowtok"],
        tensors["moe_rowpart"], tensors["moe_rowgate"], tensors["moe_fug"], ids, gates)
    path = Path(str(prefix) + ".act.part.bin")
    part, hashes["part"] = load_tensor(path, torch.bfloat16, (path.stat().st_size // 2,))
    if part.numel() < m * h:
        raise ValueError("routed BF16 output capture is too small")
    return {"stage1.input": tensors["routed_xq"].view(torch.float8_e4m3fn),
            "stage1.scale": tensors["routed_xs"].T.contiguous(), "stage1.output": hidden,
            "stage2.input": tensors["routed_hq"].view(torch.float8_e4m3fn),
            "stage2.scale": tensors["routed_hs"].T.reshape(m, topk, i // 128).contiguous(),
            "stage2.output": part[:m * h].reshape(m, h)}, hashes


def bf16_order_bounds(parts):
    if (parts.dtype != torch.bfloat16 or parts.ndim != 3 or not 1 <= parts.shape[1] <= 8
            or not bool(torch.isfinite(parts).all())):
        raise ValueError("requires finite BF16 parts with topk1..8")
    lo, hi = [torch.zeros_like(parts[:, 0])], [torch.zeros_like(parts[:, 0])]
    for mask in range(1, 1 << parts.shape[1]):
        low, high = None, None
        for slot in range(parts.shape[1]):
            if mask & (1 << slot):
                prev = mask ^ (1 << slot)
                a, b = lo[prev] + parts[:, slot], hi[prev] + parts[:, slot]
                low = a if low is None else torch.minimum(low, a)
                high = b if high is None else torch.maximum(high, b)
        lo.append(low); hi.append(high)
    return lo[-1], hi[-1]


def check_routed_ab(args):
    if args.reference_json is None:
        raise ValueError("routed A/B validation requires a pinned connected audit")
    audit = json.loads(args.reference_json.read_text())
    if (audit.get("passed") is not True or audit.get("audit_complete") is not True
            or audit.get("vllm_version") != "0.29.0" or len(audit["cases"]) != args.tp
            or {r["rank"] for r in audit["cases"]} != set(range(args.tp))):
        raise ValueError("requires a complete passed pinned audit")
    arms = ("ctl", "treat", "ctl2", "treat2")
    partials = {arm: [] for arm in arms}
    rank_checks = []
    for row in sorted(audit["cases"], key=lambda r: r["rank"]):
        rank = row["rank"]
        m, h, i, experts, topk = row["shape"]
        expected = {}
        for key, dtype, shape in (("stage1.input", torch.float8_e4m3fn, (m, h)),
                                 ("stage1.scale", torch.float32, (m, h // 128)),
                                 ("stage1.output", torch.bfloat16, (m, topk, i)),
                                 ("stage2.input", torch.float8_e4m3fn, (m, topk, i)),
                                 ("stage2.scale", torch.float32, (m, topk, i // 128))):
            boundary = row["boundaries"][key]
            path = args.reference_json.parent / f"{args.reference_json.stem}.rank{rank}.{key}.bin"
            value, digest = load_tensor(path, dtype, shape)
            if (digest != boundary["sha256"] or boundary["repeat_bitwise"] is not True
                    or boundary["plow_bitwise"] is not True or boundary["finite"] is not True):
                raise ValueError("pinned stable boundary is not qualified or changed")
            expected[key] = value
        parts = torch.empty((m, topk, h), dtype=torch.bfloat16)
        seen = set()
        for case in row["isolated_weighted_down"]:
            path = args.reference_json.parent / case["file"]
            shape, _, value, digest = load_case(path, weighted=True)
            if (digest != case["sha256"] or case["finite"] is not True or case["repeat_bitwise"] is not True
                    or shape != (len(case["tokens"]), h, i)):
                raise ValueError("isolated weighted-down reference changed")
            for at, (token, slot) in enumerate(zip(case["tokens"], case["slots"], strict=True)):
                if not (0 <= token < m and 0 <= slot < topk) or (token, slot) in seen:
                    raise ValueError("invalid isolated reference route coverage")
                seen.add((token, slot)); parts[token, slot] = value[at]
        if len(seen) != m * topk:
            raise ValueError("incomplete isolated reference route coverage")
        lo, hi = bf16_order_bounds(parts)
        for arm in arms:
            prefix = args.capture / arm / "outputs" / f"rank{rank}"
            _, xhash = load_tensor(Path(str(prefix) + ".act.xn2.bin"), torch.bfloat16, (m, h))
            ids, gates, routehash = load_routes(Path(str(prefix) + ".act.tab.bin"), m, topk, experts)
            if xhash != row["input_sha256"] or routehash != row["routes_sha256"]:
                raise ValueError("A/B changed routed input or routes; needs a new pinned audit")
            native, _ = native_routed_boundaries(prefix, h, i, experts, ids, gates)
            for key, value in expected.items():
                if not torch.equal(native[key].view(torch.uint8), value.view(torch.uint8)):
                    raise ValueError(f"{arm}/rank{rank}/{key}: nonbitwise stable boundary")
            out = native["stage2.output"]
            if not bool(torch.isfinite(out).all() and ((out >= lo) & (out <= hi)).all()):
                raise ValueError(f"{arm}/rank{rank}: BF16 atomic output outside addition-order bounds")
            shared, _ = load_tensor(Path(str(prefix) + ".act.shared.bin"), torch.bfloat16, (m, h))
            partials[arm].append((out.float() + shared.float()).bfloat16().float())
            rank_checks.append(dict(arm=arm, rank=rank, stable_boundaries_bitwise=True, reduction_in_bounds=True))
    for arm in arms:
        ffn = torch.stack(partials[arm]).sum(0).bfloat16()
        for rank in range(args.tp):
            prefix = args.capture / arm / "outputs" / f"rank{rank}"
            actual, _ = load_tensor(Path(str(prefix) + ".act.attn.bin"), torch.bfloat16, ffn.shape)
            xmid, _ = load_tensor(Path(str(prefix) + ".act.xmid.bin"), torch.bfloat16, ffn.shape)
            xnext, _ = load_tensor(Path(str(prefix) + ".act.xnext.bin"), torch.bfloat16, ffn.shape)
            if (not torch.equal(actual.view(torch.uint8), ffn.view(torch.uint8))
                    or not torch.equal(xnext.view(torch.uint8), (xmid.float() + ffn.float()).bfloat16().view(torch.uint8))):
                raise ValueError(f"{arm}/rank{rank}: shared/TP/residual rounding mismatch")
    outputs = {arm: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((args.capture / arm / "outputs").glob("*.bin"))} for arm in arms}
    with args.output.open("x") as stream:
        json.dump(dict(scope="routed-BF16-atomic-repeat-v1", passed=True, tp=args.tp,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            output_sha256=outputs, checks=rank_checks,
            limitations="addition-order extrema do not establish reachability of every interior value; not full-model qualification"), stream, indent=2)
    print(f"validated {len(rank_checks)} arm/rank chains and shared/TP/residual outputs")
    return 0


def write_routed_group_case(path, a, asc, ids, gates, w, ws, expected, *, down=False):
    m, topk = ids.shape
    experts, wn, wk = w.shape
    i, h = (wk, wn) if down else (wn // 2, wk)
    input_rows, input_width, output_width = (m * topk, i, h) if down else (m, h, i)
    if (min(m, h, experts, i, topk) < 1 or i % 128 or h % 128 or (not down and wn != i * 2)
            or a.dtype != torch.uint8 or a.shape != (input_rows, input_width)
            or asc.dtype != torch.float32 or asc.shape != (input_width // 128, input_rows)
            or ids.dtype != torch.int32 or ids.shape != (m, topk)
            or gates.dtype != torch.float32 or gates.shape != ids.shape
            or w.dtype != torch.float8_e4m3fn or ws.dtype != torch.float32
            or ws.shape != (experts, wn // 128, wk // 128)
            or expected.dtype != torch.bfloat16 or expected.shape != (m, topk, output_width)
            or bool((ids < 0).any() or (ids >= experts).any())
            or not bool(torch.isfinite(gates).all())):
        raise ValueError("unsupported grouped routed replay geometry or dtype")
    active = sorted(set(ids.flatten().tolist()))
    with path.open("xb") as f:
        f.write(struct.pack("<6I", m, i, h, experts, topk, len(active)))
        def tensor(value):
            f.write(value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        for value in (a, asc, ids, gates, expected):
            tensor(value)
        for expert in active:
            f.write(struct.pack("<I", expert))
            tensor(w[expert])
            tensor(ws[expert])


def export_routed(args):
    if args.checkpoint is None or args.reference_json is None:
        raise ValueError("routed export requires original checkpoint and boundary audit")
    audit = json.loads(args.reference_json.read_text())
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    rows = audit["cases"]
    if (audit.get("audit_complete") is not True or audit.get("vllm_version") != "0.29.0"
            or args.tp < 1 or len(rows) != args.tp or {r["rank"] for r in rows} != set(range(args.tp))):
        raise ValueError("requires complete pinned routed boundary audit")
    exported = []
    for row in rows:
        rank = row["rank"]
        m, h, i, experts, topk = row["shape"]
        if m != metadata["batch"] or row["selected"]["run_1stage"] or not row["finite"]:
            raise ValueError("requires finite two-stage routed capture")
        captured = {}
        down = getattr(args, "export_routed_down_grouped", False)
        boundaries = (("stage1.output", torch.bfloat16, (m, topk, i)),
                      ("stage2.input", torch.uint8, (m, topk, i)),
                      ("stage2.scale", torch.float32, (m, topk, i // 128))) if down else (
                                 ("stage1.input", torch.uint8, (m, h)),
                                  ("stage1.scale", torch.float32, (m, h // 128)),
                                  ("stage1.output", torch.bfloat16, (m, topk, i)))
        for key, dtype, shape in boundaries:
            boundary = row["boundaries"][key]
            path = args.reference_json.parent / f"{args.reference_json.stem}.rank{rank}.{key}.bin"
            value, digest = load_tensor(path, dtype, shape)
            repeated, repeat_digest = load_tensor(path.with_name(path.stem + ".repeat.bin"), dtype, shape)
            if (boundary["finite"] is not True or boundary["repeat_bitwise"] is not True
                    or digest != boundary["sha256"] or repeat_digest != boundary["repeat_sha256"]
                    or not torch.equal(value.view(torch.uint8), repeated.view(torch.uint8))):
                raise ValueError("routed boundary hash or repeat mismatch")
            captured[key] = value
        ids, gates, route_digest = load_routes(args.capture / "outputs" / f"rank{rank}.act.tab.bin", m, topk, experts)
        if route_digest != row["routes_sha256"]:
            raise ValueError("routed table changed since reference audit")
        weights = routed_weights(args.checkpoint, metadata["layer"], rank, args.tp)
        for name, value in zip(("gate_up", "down", "gate_up_scale", "down_scale"), weights):
            if tensor_digest(value) != row["checkpoint_shard_sha256"][name]:
                raise ValueError("original checkpoint shard changed since reference audit")
        if getattr(args, "export_routed_grouped", False) or down:
            copies = args.grouped_repeat
            if not 1 <= copies <= 16:
                raise ValueError("grouped repeat must be in [1,16]")
            path = args.output.parent / f"rank{rank}.grouped.bin"
            if down:
                expected = torch.empty((m, topk, h), dtype=torch.bfloat16)
                seen = torch.zeros((m, topk), dtype=torch.bool)
                for case in row["isolated_weighted_down"]:
                    shape, operands, result, digest = load_case(args.reference_json.parent / case["file"], weighted=True)
                    token, slot = (ids == case["expert"]).nonzero(as_tuple=True)
                    a, w, asc, ws, route_weights = operands
                    if (not case["finite"] or not case["repeat_bitwise"] or digest != case["sha256"]
                            or tuple(shape) != (token.numel(), h, i)
                            or token.tolist() != case["tokens"] or slot.tolist() != case["slots"]
                            or bool(seen[token, slot].any())
                            or not torch.equal(a.view(torch.uint8), captured["stage2.input"][token, slot])
                            or not torch.equal(asc.view(torch.int32), captured["stage2.scale"][token, slot].view(torch.int32))
                            or not torch.equal(w.view(torch.uint8), weights[1][case["expert"]].view(torch.uint8))
                            or not torch.equal(ws.view(torch.int32), weights[3][case["expert"]].view(torch.int32))
                            or not torch.equal(route_weights.view(torch.int32), gates[token, slot].view(torch.int32))):
                        raise ValueError("isolated weighted-down operand or hash mismatch")
                    expected[token, slot] = result
                    seen[token, slot] = True
                if not bool(seen.all()):
                    raise ValueError("missing isolated weighted-down routes")
                write_routed_group_case(path, captured["stage2.input"].reshape(-1, i).repeat(copies, 1),
                    captured["stage2.scale"].reshape(-1, i // 128).repeat(copies, 1).T.contiguous(),
                    ids.repeat(copies, 1), gates.repeat(copies, 1), weights[1], weights[3],
                    expected.repeat(copies, 1, 1), down=True)
                hidden_path = path.with_suffix(".hidden.bf16")
                with hidden_path.open("xb") as f:
                    f.write(captured["stage1.output"].repeat(copies, 1, 1).contiguous().view(torch.uint8).numpy().tobytes())
            else:
                write_routed_group_case(path, captured["stage1.input"].repeat(copies, 1),
                    captured["stage1.scale"].repeat(copies, 1).T.contiguous(), ids.repeat(copies, 1),
                    gates.repeat(copies, 1), weights[0], weights[2], captured["stage1.output"].repeat(copies, 1, 1))
            exported.append(dict(rank=rank, file=path.name, shape=[m * copies, i, h, experts, topk],
                captured_batch=m, replicated_rows=copies != 1,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if down:
                exported[-1].update(hidden_file=hidden_path.name,
                    hidden_sha256=hashlib.sha256(hidden_path.read_bytes()).hexdigest())
            del weights
            continue
        for expert in sorted(set(ids.flatten().tolist())):
            token, slot = (ids == expert).nonzero(as_tuple=True)
            path = args.output.parent / f"rank{rank}.expert{expert}.glu.bin"
            write_case(path, captured["stage1.input"][token], weights[0][expert],
                captured["stage1.scale"][token].T.contiguous(), weights[2][expert],
                captured["stage1.output"][token, slot], glu=True)
            exported.append(dict(rank=rank, expert=expert, tokens=token.tolist(), slots=slot.tolist(),
                shape=[token.numel(), i, h], file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del weights
    with args.output.open("x") as f:
        json.dump(dict(scope="routed captured operands/CK outputs (stage2 uses isolated weighted parts); replicated rows test tiling, not active reference dispatch at expanded batch; not serving",
            precision_qualified=False, reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            cases=exported), f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"exported {len(exported)} routed replay cases", flush=True)
    return 0


def isolated_route_weights(ids, sorted_ids, sorted_weights, valid_ids, expert):
    token = sorted_ids.long() & 0xffffff
    slot = (sorted_ids.long() >> 24) & 0xff
    valid = (token < ids.shape[0]) & (slot < ids.shape[1])
    valid &= torch.arange(sorted_ids.numel(), device=sorted_ids.device) < valid_ids
    matched = ids[token.clamp_max(ids.shape[0] - 1), slot.clamp_max(ids.shape[1] - 1)] == expert
    return torch.where(valid & matched, sorted_weights, torch.zeros_like(sorted_weights))


def isolate_routed_down(args, rank, call, ids, gates, down, down_scale):
    a = call.args[0].detach().cpu()
    asc = call.keywords["a2_scale"].detach().cpu()
    if a.dtype != torch.float8_e4m3fn or asc.dtype != torch.float32:
        raise ValueError("weighted-down isolation requires block FP8 stage2")
    gpu_ids = ids.to(call.args[3].device)
    valid_ids = int(call.args[5][0].item())
    records = []
    parts = torch.empty((*ids.shape, down.shape[1]), dtype=torch.bfloat16)
    for expert in sorted(set(ids.flatten().tolist())):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        kw = dict(call.keywords)
        kw["sorted_weights"] = isolated_route_weights(gpu_ids, call.args[3],
            call.keywords["sorted_weights"], valid_ids, expert)
        def run():
            out = torch.zeros_like(call.args[6])
            operands = call.args[:6] + (out,) + call.args[7:]
            call.func(*operands, **kw)
            return out.cpu()[token].contiguous()
        reference, repeat = run(), run()
        parts[token, slot] = reference
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(repeat).all())
        stable = torch.equal(reference.view(torch.uint8), repeat.view(torch.uint8))
        record = dict(rank=rank, expert=expert, tokens=token.tolist(), slots=slot.tolist(),
            shape=[token.numel(), down.shape[1], down.shape[2]], finite=finite, repeat_bitwise=stable)
        if not finite or not stable:
            raise ValueError(f"isolated weighted-down reference is not stable: {record}")
        dest = args.output.parent / f"rank{rank}.expert{expert}.weighted-down.bin"
        write_case(dest, a.view(torch.uint8)[token, slot], down[expert],
            asc[token, slot].T.contiguous(), down_scale[expert], reference,
            row_weights=gates[token, slot])
        record.update(file=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        records.append(record)
    return records, parts


def compare_packet_routed(args, rocm_aiter_ops, version):
    import importlib
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import QuantMethod

    if version != "0.29.0" or args.checkpoint is None or not rocm_aiter_ops.is_fused_moe_enabled():
        raise ValueError("requires pinned vLLM 0.29.0, original checkpoint and active AITER fused MoE")
    fm = importlib.import_module("aiter.fused_moe")
    measurement = json.loads((args.capture / "measurement.json").read_text())
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    config = json.loads((args.checkpoint / "config.json").read_text())
    if (args.tp < 1 or measurement.get("tp") != args.tp
            or measurement.get("scope") != "single-block-decode"
            or measurement.get("batch") != metadata.get("batch")):
        raise ValueError("requires single-block capture with matching TP and batch")
    m, layer = metadata["batch"], metadata["layer"]
    h, topk, experts = (config[key] for key in
                        ("hidden_size", "num_experts_per_tok", "n_routed_experts"))
    i = config["moe_intermediate_size"] // args.tp
    rows = []
    for rank in range(args.tp):
        prefix = args.capture / "outputs" / f"rank{rank}"
        x, xhash = load_tensor(Path(str(prefix) + ".act.xn2.bin"), torch.bfloat16, (m, h))
        ids, gates, routehash = load_routes(Path(str(prefix) + ".act.tab.bin"), m, topk, experts)
        native = None
        if args.routed_w8a8:
            native, native_hashes = native_routed_boundaries(prefix, h, i, experts, ids, gates)
            part, parthash = native["stage2.output"], native_hashes["part"]
        else:
            part, parthash = load_tensor(Path(str(prefix) + ".act.part.bin"), torch.float32, (m, topk, h))
        weights = routed_weights(args.checkpoint, layer, rank, args.tp)
        weight_hashes = {name: tensor_digest(value) for name, value in
                         zip(("gate_up", "down", "gate_up_scale", "down_scale"), weights)}
        w1, w2 = rocm_aiter_ops.shuffle_weights(weights[0].cuda(), weights[1].cuda())
        w1.is_shuffled = w2.is_shuffled = True
        ws1, ws2 = weights[2].cuda(), weights[3].cuda()
        down_original = (weights[1], weights[3]) if args.routed_down_isolate else None
        del weights
        selected = fm.get_2stage_cfgs(fm.get_padded_M(m), h, i, experts, topk,
            torch.bfloat16, w1.dtype, w2.dtype, fm.QuantType.per_1x128, True,
            fm.ActivationType.Silu, False, 0, 0, True, fm.GateMode.SEPARATED)
        selection = {key: describe_call(getattr(selected, key)) if key in ("stage1", "stage2")
                     else describe_value(getattr(selected, key)) for key in
                     ("stage1", "stage2", "block_m", "ksplit", "run_1stage", "fuse_quant", "prequant", "flat")}
        gx, gg, gi = x.cuda(), gates.cuda(), ids.cuda()
        def run():
            return rocm_aiter_ops.fused_moe(gx, w1, w2, gg, gi,
                quant_method=QuantMethod.BLOCK_128x128.value,
                w1_scale=ws1, w2_scale=ws2, doweight_stage1=False,
                output_dtype=torch.bfloat16,
                moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy())

        previous = fm.kernel_bench_callable
        try:
            fm.kernel_bench_callable = []
            reference = run().cpu()
            calls = fm.kernel_bench_callable
            snapshots = routed_stage_snapshots(calls, selected.run_1stage)
            stages = []
            for name, call in calls:
                stages.append(dict(stage=name, selected=describe_call(call),
                                   operands=[describe_value(v) for v in call.args]))
            fm.kernel_bench_callable = []
            repeat = run().cpu()
            repeated = routed_stage_snapshots(fm.kernel_bench_callable, selected.run_1stage)
            if args.routed_down_isolate:
                if selected.run_1stage or [name for name, _ in fm.kernel_bench_callable] != ["stage1", "stage2"]:
                    raise ValueError("weighted-down isolation requires two-stage reference")
                down_cases, down_parts = isolate_routed_down(args, rank, fm.kernel_bench_callable[1][1],
                    ids, gates, *down_original)
            else:
                down_cases = []
        finally:
            fm.kernel_bench_callable = previous
        del calls
        boundaries = {}
        for key, value in snapshots.items():
            other = repeated[key]
            boundaries[key] = dict(**describe_value(value), sha256=tensor_digest(value),
                repeat_sha256=tensor_digest(other),
                finite=bool(torch.isfinite(value.float()).all() and torch.isfinite(other.float()).all()),
                repeat_bitwise=torch.equal(value.view(torch.uint8), other.view(torch.uint8)))
            if native is not None:
                candidate = native[key]
                if candidate.shape != value.shape or candidate.dtype != value.dtype:
                    raise ValueError(f"native routed boundary geometry/dtype mismatch: {key}")
                boundaries[key].update(plow_sha256=tensor_digest(candidate),
                    plow_bitwise=torch.equal(candidate.view(torch.uint8), value.view(torch.uint8)),
                    plow_finite=bool(torch.isfinite(candidate.float()).all()),
                    plow_changed_bytes=int((candidate.contiguous().reshape(-1).view(torch.int8)
                        != value.contiguous().reshape(-1).view(torch.int8)).sum()))
            for suffix, tensor in (("", value), (".repeat", other)):
                dest = args.output.parent / f"{args.output.stem}.rank{rank}.{key}{suffix}.bin"
                with dest.open("xb") as f:
                    f.write(tensor.view(torch.uint8).numpy().tobytes())
        if native is not None:
            plow = part
        else:
            plow_sum = torch.zeros((m, h), dtype=torch.float32)
            for slot in range(topk):
                plow_sum += part[:, slot]
            plow = plow_sum.bfloat16()
        denom = reference.double().norm(dim=1).clamp_min(1e-30)
        rel = (plow.double() - reference.double()).norm(dim=1) / denom
        repeat_rel = (repeat.double() - reference.double()).norm(dim=1) / denom
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(repeat).all()
                      and torch.isfinite(part).all() and torch.isfinite(x).all()
                      and all(boundary["finite"] for boundary in boundaries.values()))
        row = dict(rank=rank, shape=[m, h, i, experts, topk], selected=selection, stages=stages,
            boundaries=boundaries,
            isolated_weighted_down=down_cases,
            input_sha256=xhash, routes_sha256=routehash, part_sha256=parthash,
            checkpoint_shard_sha256=weight_hashes, finite=finite,
            reference_repeat_bitwise=torch.equal(reference.view(torch.int16), repeat.view(torch.int16)),
            reference_repeat_max_row_rel_l2=float(repeat_rel.max()),
            plow_routed_max_row_rel_l2=float(rel.max()),
            changed_elements=int((plow.view(torch.int16) != reference.view(torch.int16)).sum()),
            block_oracle_verified=measurement.get("oracle_verified") is True)
        if native is not None:
            row["native_capture_sha256"] = native_hashes
            row["stable_boundaries_bitwise"] = len(boundaries) == 6 and all(
                v["repeat_bitwise"] and v["plow_bitwise"] and v["plow_finite"]
                for key, v in boundaries.items() if key != "stage2.output")
            if args.routed_down_isolate:
                lo, hi = bf16_order_bounds(down_parts)
                row["bf16_addition_order_bounds"] = {
                    tag: bool(torch.isfinite(value).all() and ((value >= lo) & (value <= hi)).all())
                    for tag, value in (("plow", plow), ("reference", reference), ("repeat", repeat))}
        for name, value in (("reference", reference), ("repeat", repeat), ("plow-routed", plow)):
            with (args.output.parent / f"{args.output.stem}.rank{rank}.{name}.bf16").open("xb") as f:
                f.write(value.contiguous().view(torch.uint8).numpy().tobytes())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del w1, w2, ws1, ws2, selected
    complete = len(rows) == args.tp and all(row["finite"] and row["stages"] for row in rows)
    passed = complete and args.routed_w8a8 and args.routed_down_isolate and all(
        row["stable_boundaries_bitwise"] and all(row["bf16_addition_order_bounds"].values()) for row in rows)
    with args.output.open("x") as f:
        json.dump(dict(scope="routed MoE diagnostic conditioned on Plow input and routes; excludes router, shared addition and TP reduction",
            vllm_version=version, checkpoint=str(args.checkpoint), precision_qualified=False,
            passed=passed, audit_complete=complete,
            plow_reduction=("captured BF16 atomic output; exact addition-order extrema, not bitwise reduction or reachability of interior values"
                if args.routed_w8a8 else "FP32 captured weighted parts summed in slot order, then BF16; not packet shared-add boundary"),
            aiter_source_sha256=hashlib.sha256(Path(fm.__file__).read_bytes()).hexdigest(),
            cases=rows), f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if (passed if args.routed_w8a8 else complete) else 1


def packet_oproj_cases(capture, checkpoint, tp):
    from safetensors import safe_open

    measurement = json.loads((capture / "measurement.json").read_text())
    metadata = json.loads((capture / "inputs/reference.json").read_text())
    if (tp < 1 or measurement.get("tp") != tp or measurement.get("scope") != "single-block-decode"
            or measurement.get("oracle_verified") is not True
            or measurement.get("batch") != metadata.get("batch")):
        raise ValueError("requires a passed single-block capture with matching TP and batch")
    m, layer = metadata["batch"], metadata["layer"]
    weight_name = f"model.layers.{layer}.self_attn.o_proj.weight"
    scale_name = weight_name + "_scale_inv"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    originals = []
    for name in (weight_name, scale_name):
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
            originals.append(shard.get_tensor(name))
    weight, scale = originals
    n, full_k = weight.shape
    if (weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or full_k % tp
            or (full_k // tp) % 128 or n % 128 or tuple(scale.shape) != (n // 128, full_k // 128)):
        raise ValueError("unsupported original checkpoint block-FP8 geometry")
    k = full_k // tp
    for rank in range(tp):
        tensors, hashes = {}, {}
        for name, dtype, shape in (
            ("act.oat", torch.bfloat16, (m, k)),
            ("act.blk_xq", torch.uint8, (m, k)),
            ("act.blk_xs", torch.float32, (k // 128, m)),
            ("act.og_tp", torch.bfloat16, (m, n)),
            (weight_name + "_fp8", torch.uint8, (n, k)),
            (scale_name, torch.float32, (n // 128, k // 128)),
        ):
            path = capture / "outputs" / f"rank{rank}.{name}.bin"
            tensors[name], hashes[name] = load_tensor(path, dtype, shape)
        w = weight[:, rank * k:(rank + 1) * k].contiguous()
        ws = scale[:, rank * (k // 128):(rank + 1) * (k // 128)].contiguous()
        weight_equal = torch.equal(tensors[weight_name + "_fp8"], w.view(torch.uint8))
        scale_equal = torch.equal(tensors[scale_name].view(torch.int32), ws.view(torch.int32))
        yield dict(rank=rank, shape=(m, n, k), hashes=hashes,
                   checkpoint_weight_bitwise=weight_equal, checkpoint_scale_bitwise=scale_equal), (
            tensors["act.oat"], tensors["act.blk_xq"], tensors["act.blk_xs"].T.contiguous(),
            w, ws, tensors["act.og_tp"])


def ordered_bf16_sum(partials):
    if len(partials) != 4 or any(p.dtype != torch.bfloat16 or p.shape != partials[0].shape
                                  for p in partials):
        raise ValueError("requires four equally shaped BF16 partials")
    result = partials[0].clone()
    for partial in partials[1:]:
        result = result + partial
    return result


def packet_shared_cases(capture, checkpoint, tp):
    from safetensors import safe_open

    measurement = json.loads((capture / "measurement.json").read_text())
    metadata = json.loads((capture / "inputs/reference.json").read_text())
    if (tp < 1 or measurement.get("tp") != tp or measurement.get("scope") != "single-block-decode"
            or not isinstance(measurement.get("oracle_verified"), bool)
            or measurement.get("batch") != metadata.get("batch")):
        raise ValueError("requires an explicitly validated single-block capture with matching TP and batch")
    m, layer = metadata["batch"], metadata["layer"]
    prefix = f"model.layers.{layer}.mlp.shared_experts."
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    originals = {}
    for proj in ("gate", "up", "down"):
        for suffix in ("weight", "weight_scale_inv"):
            name = f"{prefix}{proj}_proj.{suffix}"
            with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
                originals[proj, suffix] = shard.get_tensor(name)
    full_inter, h = originals["gate", "weight"].shape
    if full_inter % tp or (full_inter // tp) % 128 or h % 128:
        raise ValueError("unsupported shared-expert TP block geometry")
    inter = full_inter // tp
    for proj in ("gate", "up", "down"):
        shape = (h, full_inter) if proj == "down" else (full_inter, h)
        w, s = originals[proj, "weight"], originals[proj, "weight_scale_inv"]
        if (w.dtype != torch.float8_e4m3fn or tuple(w.shape) != shape or s.dtype != torch.float32
                or tuple(s.shape) != tuple(dim // 128 for dim in shape)):
            raise ValueError("unsupported shared-expert checkpoint dtype or scale shape")
    for rank in range(tp):
        tensors, hashes = {}, {}
        def read(name, dtype, shape):
            value, hashes[name] = load_tensor(capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape)
            return value
        for name, dtype, shape in (
            ("xn2", torch.bfloat16, (m, h)), ("sh_gate", torch.bfloat16, (m, inter)),
            ("shfu_up", torch.bfloat16, (m, inter)), ("shfu", torch.bfloat16, (m, inter)),
            ("sh_xq", torch.uint8, (m, h)), ("sh_xs", torch.float32, (h // 128, m)),
            ("sh_hq", torch.uint8, (m, inter)), ("sh_hs", torch.float32, (inter // 128, m)),
            ("shared", torch.bfloat16, (m, h)),
        ):
            tensors[name] = read("act." + name, dtype, shape)
        matched = {}
        for proj in ("gate", "up", "down"):
            for suffix in ("weight", "weight_scale_inv"):
                width = inter if suffix == "weight" else inter // 128
                whole = originals[proj, suffix]
                part = (whole[:, rank * width:(rank + 1) * width] if proj == "down"
                        else whole[rank * width:(rank + 1) * width]).contiguous()
                name = f"{prefix}{proj}_proj.{suffix}" + ("_fp8" if suffix == "weight" else "")
                captured = read(name, torch.uint8 if suffix == "weight" else torch.float32, part.shape)
                matched[proj + "." + suffix] = torch.equal(captured.view(torch.uint8), part.view(torch.uint8))
                tensors[proj + "." + suffix] = part
        yield dict(rank=rank, shape=(m, h, inter), hashes=hashes, checkpoint_bitwise=matched,
                   block_oracle_verified=measurement["oracle_verified"]), tensors


def shared_qualification(rows, tp):
    complete = tp > 0 and len(rows) == tp and {row["rank"] for row in rows} == set(range(tp))
    boundaries_passed = complete and all(row["passed"] for row in rows)
    return boundaries_passed, boundaries_passed and all(row["block_oracle_verified"] is True for row in rows)


def compare_packet_norm(args, version):
    from safetensors import safe_open
    from aiter import rms_norm
    from vllm.kernels.aiter_ops import fused_add_rms_norm

    if args.checkpoint is None:
        raise ValueError("normalization comparison requires original checkpoint")
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    config = json.loads((args.checkpoint / "config.json").read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    name = f"model.layers.{metadata['layer']}.post_attention_layernorm.weight"
    with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as shard:
        weight = shard.get_tensor(name)
    epsilon = config["rms_norm_eps"]
    rows = []
    residual = attention = None
    for row, tensors in packet_shared_cases(args.capture, args.checkpoint, args.tp):
        m, h, _ = row["shape"]
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (h,):
            raise ValueError("unexpected post-attention norm weight")
        x, digest = load_tensor(args.capture / "outputs" / f"rank{row['rank']}.act.xmid.bin",
                                torch.bfloat16, (m, h))
        actual = tensors["xn2"]
        if residual is None:
            residual, _ = load_tensor(args.capture / "inputs/act.x.bin", torch.bfloat16, (m, h))
            summed = torch.zeros((m, h), dtype=torch.float32)
            for rank in range(args.tp):
                partial, _ = load_tensor(args.capture / "outputs" / f"rank{rank}.act.og_tp.bin",
                                          torch.bfloat16, (m, h))
                summed += partial.float()
            attention = summed.bfloat16()
        fused, fused_residual = fused_add_rms_norm.impl_fn(attention.cuda(), residual.cuda(), weight.cuda(), epsilon)
        fused_repeat, _ = fused_add_rms_norm.impl_fn(attention.cuda(), residual.cuda(), weight.cuda(), epsilon)
        fused, fused_repeat = fused.cpu(), fused_repeat.cpu()
        residual_equal = torch.equal(fused_residual.cpu().view(torch.int16), x.view(torch.int16))
        reference = rms_norm(x.cuda(), weight.cuda(), epsilon).cpu()
        repeat = rms_norm(x.cuda(), weight.cuda(), epsilon).cpu()
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon)
        comparisons = {}
        for label, value in (("aiter", reference), ("fused_aiter", fused),
                             ("fp32_multiply", (normalized * weight.float()).bfloat16()),
                             ("bf16_intermediate", normalized.bfloat16() * weight)):
            rel = (actual.double() - value.double()).norm(dim=1) / value.double().norm(dim=1).clamp_min(1e-30)
            comparisons[label] = dict(bitwise=torch.equal(actual.view(torch.int16), value.view(torch.int16)),
                changed_elements=int((actual.view(torch.int16) != value.view(torch.int16)).sum()),
                max_row_rel_l2=float(rel.max()))
        finite = all(bool(torch.isfinite(t).all()) for t in (actual, reference, repeat, fused, fused_repeat))
        stable = (torch.equal(reference.view(torch.int16), repeat.view(torch.int16))
                  and torch.equal(fused.view(torch.int16), fused_repeat.view(torch.int16)))
        rows.append(dict(rank=row["rank"], xmid_sha256=digest, comparisons=comparisons,
                         finite=finite, reference_repeat_bitwise=stable, reconstructed_residual_bitwise=residual_equal,
                         passed=finite and stable and residual_equal and comparisons["fused_aiter"]["max_row_rel_l2"] < 0.004))
    passed = len(rows) == args.tp and all(row["passed"] for row in rows)
    with args.output.open("x") as f:
        json.dump(dict(scope="post-attention norm; fused input reconstructed from captured ordered TP partials; not whole-block qualification",
                       vllm_version=version, backend="aiter", precision_qualified=False,
                       epsilon=epsilon, passed=passed, cases=rows), f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if passed else 1


def export_shared(args):
    if args.checkpoint is None or args.reference_json is None:
        raise ValueError("shared replay export requires checkpoint and reference JSON")
    reference = json.loads(args.reference_json.read_text())
    refs = {row["rank"]: row for row in reference["cases"]}
    if len(reference["cases"]) != args.tp or set(refs) != set(range(args.tp)):
        raise ValueError("reference must contain every rank exactly once")
    rows = []
    for row, t in packet_shared_cases(args.capture, args.checkpoint, args.tp):
        rank = row["rank"]
        ref = refs[rank]
        if (row["hashes"] != ref["hashes"] or not all(row["checkpoint_bitwise"].values())
                or not all(ref["quantization"].get(stage, {}).get(check) is True
                           for stage in ("input", "hidden")
                           for check in ("fp8_bitwise", "scales_bitwise", "repeat_bitwise"))
                or not all(ref["stages"][p]["reference_repeat_bitwise"] for p in ("gate_up", "down"))):
            raise ValueError("reference operands or repetition check do not match")
        m, h, inter = row["shape"]
        def ref_tensor(stage, shape):
            path = args.reference_json.with_name(f"{args.reference_json.stem}.rank{rank}.{stage}.reference.bf16")
            value = load_tensor(path, torch.bfloat16, shape)[0]
            repeat_path = path.with_name(path.name.replace(".reference.bf16", ".repeat.bf16"))
            repeat = load_tensor(repeat_path, torch.bfloat16, shape)[0]
            if not torch.equal(value.view(torch.int16), repeat.view(torch.int16)):
                raise ValueError("saved reference repetition is not bitwise stable")
            return value
        gate_up = ref_tensor("gate_up", (m, 2 * inter))
        down = ref_tensor("down", (m, h))
        files = []
        for proj, q, s, expected in (
            ("gate", "sh_xq", "sh_xs", gate_up[:, :inter]),
            ("up", "sh_xq", "sh_xs", gate_up[:, inter:]),
            ("down", "sh_hq", "sh_hs", down),
        ):
            path = args.output.parent / f"rank{rank}.{proj}.bin"
            write_case(path, t[q], t[proj + ".weight"], t[s], t[proj + ".weight_scale_inv"], expected)
            files.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows.append(dict(**row, files=files))
    with args.output.open("x") as f:
        json.dump(dict(scope="shared GEMM replay inputs; not qualification", precision_qualified=False,
                       reference=str(args.reference_json), cases=rows), f, indent=2)
        f.write("\n")
    return 0


def compare_packet_shared(args, rocm_aiter_ops, version):
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS

    if args.checkpoint is None or not rocm_aiter_ops.is_linear_fp8_enabled():
        raise ValueError("shared comparison requires original checkpoint and active AITER FP8")
    rows = []
    for row, t in packet_shared_cases(args.capture, args.checkpoint, args.tp):
        m, h, inter = row["shape"]
        row["quantization"] = {}
        quantized = {}
        for stage, src, qname, sname in (("input", "xn2", "sh_xq", "sh_xs"),
                                       ("hidden", "shfu", "sh_hq", "sh_hs")):
            q, s = rocm_aiter_ops.group_fp8_quant(t[src].cuda(), 128)
            qr, sr = rocm_aiter_ops.group_fp8_quant(t[src].cuda(), 128)
            row["quantization"][stage] = dict(
                fp8_bitwise=torch.equal(t[qname], q.cpu().view(torch.uint8)),
                scales_bitwise=torch.equal(t[sname].T.contiguous().view(torch.int32), s.cpu().view(torch.int32)),
                repeat_bitwise=torch.equal(q.view(torch.uint8), qr.view(torch.uint8))
                and torch.equal(s.view(torch.int32), sr.view(torch.int32)))
            quantized[stage] = (q, s)
        row["gemm_routes"] = {}
        def gemm(stage, operands, weight, scales):
            n, k = weight.shape
            use_triton = rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
            op = (rocm_aiter_ops.triton_gemm_a8w8_blockscale if use_triton
                  else rocm_aiter_ops.gemm_a8w8_blockscale)
            config = None if use_triton else get_CKGEMM_config(
                m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
            row["gemm_routes"][stage] = dict(shape=(m, n, k), backend="triton" if use_triton else "aiter",
                selected_config=None if config is None else {key: str(value) for key, value in config.items()})
            return op(operands[0], weight, operands[1], scales, [128, 128], output_dtype=torch.bfloat16)
        guw = torch.cat([t[p + ".weight"].view(torch.uint8) for p in ("gate", "up")]).view(torch.float8_e4m3fn).cuda()
        gus = torch.cat([t[p + ".weight_scale_inv"] for p in ("gate", "up")]).cuda()
        dw, ds = t["down.weight"].cuda(), t["down.weight_scale_inv"].cuda()
        gu = gemm("gate_up", quantized["input"], guw, gus)
        gur = gemm("gate_up", quantized["input"], guw, gus)
        local_gu = torch.cat([t["sh_gate"], t["shfu_up"]], dim=1).cuda()
        def silu(value):
            out = torch.empty((m, inter), dtype=torch.bfloat16, device=value.device)
            torch.ops._C.silu_and_mul(out, value)
            return out
        act, actr = silu(local_gu), silu(local_gu)
        down = gemm("down", quantized["hidden"], dw, ds)
        downr = gemm("down", quantized["hidden"], dw, ds)
        def chain(value):
            actq, acts = rocm_aiter_ops.group_fp8_quant(silu(value), 128)
            return gemm("down", (actq, acts), dw, ds)
        full, fullr = chain(gu), chain(gur)
        row["stages"] = {}
        for name, plow, ref, repeat in (("gate_up", local_gu, gu, gur), ("silu", t["shfu"], act, actr),
                                       ("down", t["shared"], down, downr), ("chain", t["shared"], full, fullr)):
            plow, ref, repeat = plow.cpu(), ref.cpu(), repeat.cpu()
            finite = bool(torch.isfinite(plow).all() and torch.isfinite(ref).all() and torch.isfinite(repeat).all())
            rel = (plow.double() - ref.double()).norm(dim=1) / ref.double().norm(dim=1).clamp_min(1e-30)
            stable = torch.equal(ref.view(torch.int16), repeat.view(torch.int16))
            row["stages"][name] = dict(finite=finite, reference_repeat_bitwise=stable,
                max_row_rel_l2=float(rel.max()), bitwise=torch.equal(plow.view(torch.int16), ref.view(torch.int16)),
                passed=finite and stable and bool((rel < 0.004).all()))
            for tag, value in (("reference", ref), ("repeat", repeat)):
                dest = args.output.parent / f"{args.output.stem}.rank{row['rank']}.{name}.{tag}.bf16"
                with dest.open("xb") as f:
                    f.write(value.contiguous().view(torch.uint16).numpy().tobytes())
        row["passed"] = (all(row["checkpoint_bitwise"].values())
            and all(all(v.values()) for v in row["quantization"].values())
            and all(v["passed"] for v in row["stages"].values()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    boundaries_passed, passed = shared_qualification(rows, args.tp)
    with args.output.open("x") as f:
        json.dump(dict(scope="captured shared-expert boundaries; not full-model or serving qualification",
            vllm_version=version, checkpoint=str(args.checkpoint), precision_qualified=False,
            passed=passed, boundaries_passed=boundaries_passed,
            max_row_rel_l2_limit=0.004, cases=rows), f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if passed else 1


def compare_packet_oproj(args, rocm_aiter_ops, version):
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS

    if args.checkpoint is None or not rocm_aiter_ops.is_linear_fp8_enabled():
        raise ValueError("packet comparison requires original checkpoint and active AITER FP8")
    rows = []
    for row, (x, q, scales, w, ws, plow) in packet_oproj_cases(args.capture, args.checkpoint, args.tp):
        m, n, k = row["shape"]
        reference_q, reference_s = rocm_aiter_ops.group_fp8_quant(x.cuda(), 128)
        repeat_q, repeat_s = rocm_aiter_ops.group_fp8_quant(x.cuda(), 128)
        quant_equal = torch.equal(q, reference_q.cpu().view(torch.uint8))
        scales_equal = torch.equal(scales.view(torch.int32), reference_s.cpu().view(torch.int32))
        quant_stable = torch.equal(reference_q.view(torch.uint8), repeat_q.view(torch.uint8)) and torch.equal(
            reference_s.view(torch.int32), repeat_s.view(torch.int32))
        use_triton = rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
        op = (rocm_aiter_ops.triton_gemm_a8w8_blockscale if use_triton
              else rocm_aiter_ops.gemm_a8w8_blockscale)
        gpu_w, gpu_ws = w.cuda(), ws.cuda()
        reference = op(reference_q, gpu_w, reference_s, gpu_ws, [128, 128], output_dtype=torch.bfloat16).cpu()
        repeat = op(reference_q, gpu_w, reference_s, gpu_ws, [128, 128], output_dtype=torch.bfloat16).cpu()
        a, b = plow.double(), reference.double()
        rel = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
        repeat_rel = (repeat.double() - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(repeat).all())
        stable = torch.equal(reference.view(torch.int16), repeat.view(torch.int16))
        row.update(quant_fp8_bitwise=quant_equal, quant_scale_bitwise=scales_equal,
                   quant_repeat_bitwise=quant_stable, reference_repeat_bitwise=stable,
                   backend="triton" if use_triton else "aiter", finite=finite,
                   max_row_rel_l2=float(rel.max()), reference_repeat_max_row_rel_l2=float(repeat_rel.max()),
                   changed_elements=int((plow.view(torch.int16) != reference.view(torch.int16)).sum()),
                   passed=finite and stable and quant_equal and scales_equal and quant_stable
                   and row["checkpoint_weight_bitwise"] and row["checkpoint_scale_bitwise"]
                   and bool((rel < 0.004).all()))
        if not use_triton:
            config = get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
            row["selected_config"] = None if config is None else {key: str(value) for key, value in config.items()}
            if config is not None and str(config.get("libtype")) == "ck":
                row["effective_k_partitions"] = 1 << int(config["splitK"])
                row["reference_rounding_source"] = ("CK FP32 accumulator -> BF16 partial -> BF16 atomic add"
                    if row["effective_k_partitions"] > 1 else "CK FP32 accumulator -> BF16 output")
                if row["effective_k_partitions"] == 4 and k % 512 == 0:
                    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck

                    partials = []
                    for part in range(4):
                        lo, hi = part * (k // 4), (part + 1) * (k // 4)
                        out = torch.empty((m, n), dtype=torch.bfloat16, device=reference_q.device)
                        gemm_a8w8_blockscale_ck(
                            reference_q[:, lo:hi].contiguous(), gpu_w[:, lo:hi].contiguous(),
                            reference_s[:, lo // 128:hi // 128].contiguous(),
                            gpu_ws[:, lo // 128:hi // 128].contiguous(), out,
                            splitK=0, kernelName=str(config["kernelName"]))
                        partials.append(out.cpu())
                    ordered = ordered_bf16_sum(partials)
                    ordered_rel = (a - ordered.double()).norm(dim=1) / ordered.double().norm(dim=1).clamp_min(1e-30)
                    row["ordered_split4_audit"] = dict(
                        scope="same CK kernel on four K/4 slices, fixed-order BF16 adds; not the active atomic route",
                        finite=bool(torch.isfinite(ordered).all()),
                        max_row_rel_l2=float(ordered_rel.max()),
                        bitwise=torch.equal(plow.view(torch.int16), ordered.view(torch.int16)))
                    for part, value in enumerate(partials + [ordered]):
                        dest = args.output.parent / f"{args.output.stem}.rank{row['rank']}.ordered{part}.bf16"
                        with dest.open("xb") as f:
                            f.write(value.contiguous().view(torch.uint16).numpy().tobytes())
        for name, value in (("reference", reference), ("repeat", repeat)):
            dest = args.output.parent / f"{args.output.stem}.rank{row['rank']}.{name}.bf16"
            with dest.open("xb") as f:
                f.write(value.contiguous().view(torch.uint16).numpy().tobytes())
        rows.append(row)
        print(json.dumps(row), flush=True)
    passed = len(rows) == args.tp and all(row["passed"] for row in rows)
    with args.output.open("x") as f:
        json.dump(dict(scope="captured block o_proj boundary; synthetic upstream activations; not serving",
                       vllm_version=version, precision_qualified=False, passed=passed,
                       checkpoint=str(args.checkpoint), max_row_rel_l2_limit=0.004, cases=rows),
                  f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if passed else 1


def compare_quant(args, rocm_aiter_ops, version):
    if not rocm_aiter_ops.is_linear_fp8_enabled():
        raise ValueError("reference requires the active AITER FP8 linear path")
    files = sorted(args.capture.glob("*.quant"))
    shapes = [load_quant_case(path)[0] for path in files]
    expected = {(m, k) for m in (1, 8, 16, 32, 64) for k in (256, 2048, 6144)}
    if len(shapes) != len(set(shapes)) or set(shapes) != expected:
        raise ValueError("quant capture must contain every expected shape exactly once")
    rows = []
    for path in files:
        shape, x, q, scale, digest = load_quant_case(path)
        gpu = x.cuda()
        reference, ref_scale = rocm_aiter_ops.group_fp8_quant(gpu, 128)
        repeat, repeat_scale = rocm_aiter_ops.group_fp8_quant(gpu, 128)
        reference, ref_scale = reference.cpu(), ref_scale.cpu()
        repeat, repeat_scale = repeat.cpu(), repeat_scale.cpu()
        changed = int((q != reference.view(torch.uint8)).sum())
        scale_changed = int((scale.view(torch.int32) != ref_scale.view(torch.int32)).sum())
        stable = torch.equal(reference.view(torch.uint8), repeat.view(torch.uint8)) and torch.equal(
            ref_scale.view(torch.int32), repeat_scale.view(torch.int32))
        row = dict(shape=shape, sha256=digest, changed_fp8_bytes=changed,
                   changed_fp32_scales=scale_changed, reference_repeat_bitwise=stable,
                   passed=changed == 0 and scale_changed == 0 and stable)
        rows.append(row)
        print(json.dumps(row), flush=True)
        for name, value in (("fp8", reference.view(torch.uint8)), ("scales", ref_scale)):
            with (args.output.parent / f"{args.output.stem}.{path.stem}.{name}").open("xb") as f:
                f.write(value.contiguous().numpy().tobytes())
    passed = all(row["passed"] for row in rows)
    with args.output.open("x") as f:
        json.dump(dict(scope="synthetic BF16 to FP8 group128 quantization; not serving",
                       vllm_version=version, precision_qualified=False, passed=passed, cases=rows),
                  f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quant128", action="store_true")
    modes.add_argument("--block-oproj", action="store_true")
    modes.add_argument("--block-shared", action="store_true")
    modes.add_argument("--export-shared", action="store_true")
    modes.add_argument("--block-norm", action="store_true")
    modes.add_argument("--block-routed", action="store_true")
    modes.add_argument("--check-routed-ab", action="store_true")
    modes.add_argument("--export-qkva", action="store_true")
    modes.add_argument("--export-qb", action="store_true")
    modes.add_argument("--check-qb", action="store_true")
    modes.add_argument("--export-mla", action="store_true")
    modes.add_argument("--check-mla", action="store_true")
    modes.add_argument("--check-mla-weights", action="store_true")
    modes.add_argument("--block-mla", action="store_true")
    modes.add_argument("--block-attention", action="store_true")
    modes.add_argument("--check-qkva", action="store_true")
    modes.add_argument("--export-routed", action="store_true")
    modes.add_argument("--export-routed-grouped", action="store_true")
    modes.add_argument("--export-routed-down-grouped", action="store_true")
    parser.add_argument("--grouped-repeat", type=int, default=1)
    parser.add_argument("--routed-down-isolate", action="store_true")
    parser.add_argument("--routed-w8a8", action="store_true")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--qb-reference", type=Path, help="also verify strided MLA input provenance and copied raw RoPE output")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision-inventory", type=Path)
    parser.add_argument("--export-attention-ps", action="store_true")
    parser.add_argument("--export-attention-sweep", action="store_true")
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()
    if args.export_attention_ps and not args.block_attention:
        parser.error("--export-attention-ps requires --block-attention")
    if args.routed_w8a8 and not (args.block_routed and args.routed_down_isolate):
        parser.error("--routed-w8a8 requires --block-routed and --routed-down-isolate")
    if args.check_routed_ab:
        return check_routed_ab(args)
    if args.check_qkva:
        return check_qkva(args)
    if args.check_qb:
        return check_qb(args)
    if args.check_mla:
        return check_mla(args)
    if args.export_shared:
        return export_shared(args)
    if args.export_routed or args.export_routed_grouped or args.export_routed_down_grouped:
        return export_routed(args)
    from vllm import __version__
    if args.export_attention_sweep:
        return export_attention_sweep(args, __version__)
    from vllm._aiter_ops import rocm_aiter_ops
    if args.check_mla_weights:
        return check_mla_weights(args, __version__)
    if args.block_mla:
        return compare_packet_mla(args, rocm_aiter_ops, __version__)
    if args.block_attention:
        return compare_packet_attention(args, rocm_aiter_ops, __version__)
    if args.export_qb:
        return export_qb(args, rocm_aiter_ops, __version__)
    if args.export_mla:
        return export_mla(args, rocm_aiter_ops, __version__)
    if args.export_qkva:
        return export_qkva(args, rocm_aiter_ops, __version__)
    if args.block_routed:
        return compare_packet_routed(args, rocm_aiter_ops, __version__)
    if args.block_norm:
        return compare_packet_norm(args, __version__)
    if args.quant128:
        return compare_quant(args, rocm_aiter_ops, __version__)
    if args.block_oproj:
        return compare_packet_oproj(args, rocm_aiter_ops, __version__)
    if args.block_shared:
        return compare_packet_shared(args, rocm_aiter_ops, __version__)
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS

    files = sorted(args.capture.glob("*.bin"))
    shapes = [load_case(path)[0] for path in files]
    if len(shapes) != len(set(shapes)) or set(shapes) != expected_shapes():
        raise ValueError("capture must contain every expected shape exactly once")
    rows = []
    for path in files:
        shape, operands, plow, digest = load_case(path)
        m, n, k = shape
        row = dict(file=path.name, sha256=digest, shape=shape)
        if n % 128 or k % 128:
            row.update(skipped="ragged dimensions are a Plow-only f64 test")
            rows.append(row)
            continue
        use_triton = rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
        op = (rocm_aiter_ops.triton_gemm_a8w8_blockscale if use_triton
              else rocm_aiter_ops.gemm_a8w8_blockscale)
        gpu = [v.cuda() for v in operands]
        reference = op(*gpu, [128, 128], output_dtype=torch.bfloat16).cpu()
        repeat = op(*gpu, [128, 128], output_dtype=torch.bfloat16).cpu()
        for name, value in (("reference", reference), ("repeat", repeat)):
            dest = args.output.parent / f"{args.output.stem}.{path.stem}.{name}.bf16"
            with dest.open("xb") as f:
                f.write(value.contiguous().view(torch.uint16).numpy().tobytes())
        a, b = plow.double(), reference.double()
        relative = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
        repeat_relative = (repeat.double() - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        repeat_equal = torch.equal(reference.view(torch.int16), repeat.view(torch.int16))
        row.update(backend="triton" if use_triton else "aiter", finite=finite,
                   reference_repeat_bitwise=repeat_equal,
                   reference_repeat_max_row_rel_l2=float(repeat_relative.max()),
                   changed_elements=int((plow.view(torch.int16) != reference.view(torch.int16)).sum()),
                   max_row_rel_l2=float(relative.max()), max_abs=float((a - b).abs().max()),
                   passed=finite and repeat_equal and bool((relative < 0.004).all()))
        if not use_triton:
            config = get_CKGEMM_config(m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
            row["selected_config"] = None if config is None else {key: str(value) for key, value in config.items()}
            if config is not None and str(config.get("libtype")) == "ck":
                row["effective_k_partitions"] = 1 << int(config["splitK"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    checked = [r for r in rows if "passed" in r]
    passed = bool(checked) and all(r["passed"] for r in checked)
    result = dict(scope="prequantized synthetic operands; excludes activation quantization and serving",
                  vllm_version=__version__, precision_qualified=False,
                  max_row_rel_l2_limit=0.004, passed=passed, cases=rows)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
