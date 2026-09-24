#!/usr/bin/env python3
"""Compare block_fp8_gfx950_test's prequantized captures with installed vLLM/AITER."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys

import torch


def expected_shapes():
    return {(m, n, k) for m in (1, 8, 16, 32, 64) for n, k in ((256, 6144), (6144, 256))} | {
        (8, 2048, 6144), (8, 6144, 2048), (3, 130, 260), (65, 129, 129),
    }


def load_case(path, glu=False, weighted=False, fp16=False):
    raw = bytearray(path.read_bytes())
    m, n, k = struct.unpack_from("<III", raw)
    if not min(m, n, k) or sys.byteorder != "little" or (glu and weighted) or (fp16 and (glu or weighted)):
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
    out = take(torch.float16 if fp16 else torch.bfloat16, (m, n))
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


def write_case(path, a, w, asc, wsc, expected, glu=False, row_weights=None, fp16=False):
    m, k = a.shape
    n, wk = w.shape
    branches = 2 if glu else 1
    if glu:
        if n % 256:
            raise ValueError("GLU replay requires two aligned output halves")
        n //= 2
    if (not min(m, n, k) or wk != k or a.dtype != torch.uint8 or w.dtype != torch.float8_e4m3fn
            or (fp16 and (glu or row_weights is not None))
            or asc.dtype != torch.float32 or wsc.dtype != torch.float32
            or expected.dtype != (torch.float16 if fp16 else torch.bfloat16)
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


def load_tensor(path, dtype, shape, *, live_prefix=False):
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("capture dimensions must be positive")
        count *= dim
    with path.open("rb") as stream:
        raw = bytearray(stream.read(count * torch.empty((), dtype=dtype).element_size()) if live_prefix else stream.read())
    if sys.byteorder != "little" or len(raw) != count * torch.empty((), dtype=dtype).element_size():
        raise ValueError(f"{path}: unexpected tensor size or byte order")
    value = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
    return value, hashlib.sha256(raw).hexdigest()


def load_routes(path, batch, topk, experts, *, live_prefix=False):
    table, digest = load_tensor(path, torch.int32, (batch, topk, 2), live_prefix=live_prefix)
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


def indexer_contract(cfg, inventory, tp, layer):
    if (tp != 8 or len(inventory["ranks"]) != tp or layer < 0
            or layer >= len(cfg["indexer_types"]) or cfg["indexer_types"][layer] != "full"
            or tuple(cfg[k] for k in ("hidden_size", "q_lora_rank", "index_n_heads",
                                      "index_head_dim", "qk_rope_head_dim", "index_topk"))
            != (6144, 2048, 32, 128, 64, 2048)
            or cfg.get("indexer_rope_interleave") is not True
            or cfg["rope_parameters"] != dict(rope_theta=8000000, rope_type="default")):
        raise ValueError("requires GLM full indexer, TP8, 32x128 heads and interleaved plain RoPE")
    prefix = f"model.layers.{layer}.self_attn.indexer"
    for rank in inventory["ranks"]:
        modules = rank["modules"]
        for name in (prefix, prefix + ".indexer_op"):
            attrs = modules[name]["attributes"]
            if attrs.get("scale_fmt") != "ue8m0" or attrs.get("quant_block_size") != 128:
                raise ValueError("indexer query/cache quantization differs from loaded reference")
        q = modules[prefix + ".wq_b"]
        kernel = q["attributes"]["quant_method"]["fields"]["fp8_linear"]
        quant = kernel["fields"]["quant_fp8"]["fields"]
        if (not kernel["class_name"].endswith(".AiterFp8BlockScaledMMKernel")
                or kernel["fields"]["use_triton"] is not False
                or quant["use_ue8m0"] is not False or quant["group_shape"] != [1, 128]):
            raise ValueError("query projection requires loaded CK W8A8 block128 backend")
        expected = [(".wq_b", "weight", "torch.float8_e4m3fn", [4096, 2048], [2304, 1]),
                    (".wq_b", "weight_scale_inv", "torch.float32", [32, 16], [16, 1]),
                    (".wk_weights_proj", "weight", "torch.bfloat16", [160, 6144], [6144, 1]),
                    (".k_norm", "weight", "torch.float32", [128], [1]),
                    (".k_norm", "bias", "torch.float32", [128], [1])]
        for suffix, name, dtype, shape, stride in expected:
            t = modules[prefix + suffix]["tensors"][name]
            if (t["dtype"], t["shape"], t["stride"], t["device_type"]) != (dtype, shape, stride, "cuda"):
                raise ValueError("indexer loaded tensor contract mismatch")
        cache = modules[prefix + ".k_cache"]["tensors"]["kv_cache"]
        if (cache["dtype"] != "torch.uint8" or cache["device_type"] != "cuda"
                or len(cache["shape"]) != 3 or cache["shape"][0] < 1
                or cache["shape"][1:] != [16, 132] or cache["stride"] != [2112, 132, 1]):
            raise ValueError("requires loaded block16 packed FP8 + FP32-scale indexer cache")
    return prefix


def indexer_weights(checkpoint, layer):
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    spec = [("wq_b.weight", (4096, 2048), (torch.float8_e4m3fn,)),
            ("wq_b.weight_scale_inv", (32, 16), (torch.float32,)),
            ("wk.weight", (128, 6144), (torch.float8_e4m3fn,)),
            ("wk.weight_scale_inv", (1, 48), (torch.float32,)),
            ("weights_proj.weight", (32, 6144), (torch.bfloat16,)),
            ("k_norm.weight", (128,), (torch.bfloat16, torch.float32)),
            ("k_norm.bias", (128,), (torch.bfloat16, torch.float32))]
    result = {}
    for suffix, shape, dtypes in spec:
        key = f"model.layers.{layer}.self_attn.indexer.{suffix}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(key)
        if value.shape != shape or value.dtype not in dtypes or not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"{key}: unsupported original weight")
        result[suffix] = value
    return result


def write_indexer_quant_case(path, mode, x, aux, expected, blocks=0):
    rows = x.shape[0]
    if (x.dtype != torch.bfloat16 or x.shape != (rows, 128) or not 0 < rows <= 2097152
            or not bool(torch.isfinite(x).all()) or mode not in (0, 1)
            or not 0 <= blocks <= 131072 or (mode == 0) != (blocks == 0)):
        raise ValueError("invalid indexer quant geometry or input")
    if aux.shape != (rows,) or aux.dtype != (torch.int64 if mode else torch.bfloat16):
        raise ValueError("invalid indexer quant auxiliary input")
    if mode:
        live = aux[aux >= 0]
        if bool((live >= blocks * 16).any()) or live.unique().numel() != live.numel():
            raise ValueError("invalid or duplicate indexer cache slots")
    elif not bool(torch.isfinite(aux).all()):
        raise ValueError("nonfinite indexer weights")
    length = blocks * 2112 if mode else rows * 136
    if expected.dtype != torch.uint8 or expected.shape != (length,):
        raise ValueError("invalid indexer quant expected output")
    with path.open("xb") as f:
        f.write(struct.pack("<4I", 0x49515131, mode, rows, blocks))
        for value in (x, aux, expected):
            f.write(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return dict(file=path.name, mode=mode, rows=rows, blocks=blocks,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_sha256=tensor_digest(expected), input_sha256=tensor_digest(x),
                auxiliary_sha256=tensor_digest(aux))


def export_indexer_quant(args, meta, harness, rope, sources, version):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import indexer_k_quant_and_cache_triton
    if meta["layer"] != args.indexer_layer:
        raise ValueError("indexer quant export requires captured full layer")
    source_batch, ctx = meta["batch"], meta["ctx"]
    def captured(name, shape):
        return load_tensor(args.capture / "outputs" / f"rank0.{name}.bin", torch.bfloat16, shape)[0]
    raw_query = captured("act.qidx", (source_batch, 32, 128))
    raw_weight = captured("act.widx", (source_batch, 32))
    hidden = captured("act.xn", (source_batch, 6144))
    latent = captured("act.qlat", (source_batch, 2048))
    def rows(x, m):
        return x.repeat((math.ceil(m / source_batch),) + (1,) * (x.ndim - 1))[:m].contiguous().cuda()
    cases = []
    def query_case(name, x, weight, composed=None):
        q, scale = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
        again, scale_again = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
        scaled = weight.float() * scale.flatten() * harness.softmax_scale * harness.n_head_scale
        scaled_again = weight.float() * scale_again.flatten() * harness.softmax_scale * harness.n_head_scale
        expected = torch.cat([v.contiguous().view(torch.uint8).flatten() for v in (q, scale, scaled)])
        record = write_indexer_quant_case(args.output.parent / f"{name}.iqq", 0, x, weight, expected)
        record["reference_repeat_bitwise"] = all(tensor_digest(a) == tensor_digest(b)
            for a, b in ((q, again), (scale, scale_again), (scaled, scaled_again)))
        record["finite"] = all(bool(torch.isfinite(v.float()).all()) for v in (q, scale, scaled))
        record["indexer_forward_bitwise"] = (composed is None or
            (tensor_digest(q.reshape_as(composed[0])) == tensor_digest(composed[0]) and
             tensor_digest(scaled.reshape_as(composed[1])) == tensor_digest(composed[1])))
        cases.append(record)
    def key_case(name, key, slots, blocks):
        cache = torch.full((blocks, 16, 132), 0xa5, dtype=torch.uint8, device="cuda")
        repeated = torch.full_like(cache, 0xa5)
        for out in (cache, repeated):
            indexer_k_quant_and_cache_triton(key, out, slots, 128, "ue8m0")
        record = write_indexer_quant_case(args.output.parent / f"{name}.iqq", 1,
                                          key, slots, cache.flatten(), blocks)
        record.update(reference_repeat_bitwise=tensor_digest(cache) == tensor_digest(repeated),
                      finite=bool(torch.isfinite(key).all()), indexer_forward_bitwise=True)
        cases.append(record)
    from vllm.model_executor.models.deepseek_v2 import Indexer
    for m in (1, 8, 16, 32, 64):
        positions = torch.full((m,), ctx - 1, dtype=torch.int64, device="cuda")
        q, key, weight = Indexer.forward(harness, rows(hidden, m), rows(latent, m), positions, rope)
        query_case(f"m{m}.query", rows(raw_query, m).reshape(-1, 128),
                   rows(raw_weight, m).flatten(), (q, weight))
        blocks = math.ceil(m / 16) + 2
        slots = torch.arange(blocks * 16 - 1, blocks * 16 - m - 1, -1, device="cuda", dtype=torch.int64)
        if m > 1:
            slots[::7] = -1
        key_case(f"m{m}.key", key.contiguous(), slots, blocks)
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    adversarial = (values.float()[:, None] * torch.linspace(-1, 1, 128)[None]).to(torch.bfloat16).cuda()
    n = adversarial.shape[0]
    weight = torch.linspace(-1, 1, n, device="cuda").to(torch.bfloat16)
    query_case("finite-bf16.query", adversarial, weight)
    slots = torch.arange(n - 1, -1, -1, device="cuda", dtype=torch.int64)
    slots[::17] = -1
    key_case("finite-bf16.key", adversarial, slots, math.ceil(n / 16) + 1)
    passed = all(r["reference_repeat_bitwise"] and r["finite"] and r["indexer_forward_bitwise"] for r in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="installed non-pooled indexer UE8M0 quantization, weight scaling and block16 shuffled key insertion; conditioned captured operands plus finite-BF16 adversarial inputs, not native or serving qualification",
            vllm_version=version, passed=passed, audit_complete=True, precision_qualified=False,
            source_capture=str(args.capture), layer=meta["layer"], source_ctx=ctx, sources=sources,
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
            cases=cases), f, indent=2, allow_nan=False)
    return 0 if passed else 1


def write_indexer_decode_case(path, query, weights, key, positions, lengths, ctx, expected):
    m = query.shape[0]
    if (m not in (1, 8, 16, 32, 64) or ctx < 16 or ctx > 131072 or ctx % 16
            or query.dtype != torch.bfloat16 or query.shape != (m, 32, 128)
            or weights.dtype != torch.bfloat16 or weights.shape != (m, 32)
            or key.dtype != torch.bfloat16 or key.shape != (m, 128)
            or positions.dtype != torch.int32 or lengths.dtype != torch.int32
            or positions.shape != (m,) or lengths.shape != (m,)
            or bool((lengths < 0).any() or (lengths > ctx).any())
            or bool(((lengths > 0) & (positions != lengths - 1)).any())):
        raise ValueError("invalid indexer decode fixture geometry/positions")
    live = lengths > 0
    if not all(bool(torch.isfinite(v[live]).all()) for v in (query, weights, key)):
        raise ValueError("nonfinite active indexer decode input")
    expected_bytes = m * 32 * 136 + m * ctx // 16 * 2112
    if expected.dtype != torch.uint8 or expected.shape != (expected_bytes,):
        raise ValueError("invalid indexer decode expected output")
    with path.open("xb") as f:
        f.write(struct.pack("<4I", 0x49445031, m, ctx, 0))
        for value in (query, weights, key, positions, lengths, expected):
            f.write(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return dict(file=path.name, rows=m, ctx=ctx, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_sha256=tensor_digest(expected), live_rows=int(live.sum()))


def export_indexer_decode(args, version):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import indexer_k_quant_and_cache_triton
    if version != "0.29.0" or args.reference_json is None:
        raise ValueError("requires pinned vLLM and quantization reference")
    source = json.loads(args.reference_json.read_text())
    if not source["passed"] or not source["audit_complete"] or source["vllm_version"] != version:
        raise ValueError("unqualified source quantization reference")
    files = {c["file"]: c for c in source["cases"]}
    cases = []
    for m in (1, 8, 16, 32, 64):
        operands = []
        for mode, kind in enumerate(("query", "key")):
            spec = files[f"m{m}.{kind}.iqq"]
            raw = bytearray((args.reference_json.parent / spec["file"]).read_bytes())
            magic, got_mode, n, blocks = struct.unpack_from("<4I", raw)
            if (hashlib.sha256(raw).hexdigest() != spec["sha256"] or magic != 0x49515131
                    or got_mode != mode or n != m * (1 if mode else 32)
                    or not all(spec[k] for k in ("finite", "reference_repeat_bitwise", "indexer_forward_bitwise"))):
                raise ValueError("source indexer operand identity mismatch")
            x = torch.frombuffer(raw, dtype=torch.bfloat16, count=n * 128, offset=16).clone()
            aux = None if mode else torch.frombuffer(raw, dtype=torch.bfloat16, count=n, offset=16 + n * 256).clone()
            operands.append((x, aux))
        query = operands[0][0].reshape(m, 32, 128)
        weights = operands[0][1].reshape(m, 32)
        key = operands[1][0].reshape(m, 128)
        for ctx in (64, 71680):
            for mixed in (False, True):
                pos = torch.tensor(([0, 15, 16, 31, ctx - 1] * math.ceil(m / 5))[:m], dtype=torch.int32)
                lengths = pos + 1
                x, w, k = query.clone(), weights.clone(), key.clone()
                if mixed:
                    lengths[::3] = 0
                    pos[::3] = -1
                    x[::3] = float("nan")
                    w[::3] = float("nan")
                    k[::3] = float("nan")
                live = lengths > 0
                masked_q, masked_w = x.clone(), w.clone()
                masked_q[~live] = 0
                masked_w[~live] = 0
                masked_q, masked_w = masked_q.cuda().reshape(-1, 128), masked_w.cuda().flatten()
                slots = torch.arange(m, dtype=torch.int64) * ctx + pos.long()
                slots[~live] = -1
                slots, gpu_key = slots.cuda(), k.cuda()
                outputs = []
                for _ in range(2):
                    q, scale = per_token_group_quant_fp8(masked_q, 128, use_ue8m0=True)
                    scaled = masked_w.float() * scale.flatten() * (128 ** -0.5) * (32 ** -0.5)
                    cache = torch.full((m * ctx // 16, 16, 132), 0xa5, dtype=torch.uint8, device="cuda")
                    indexer_k_quant_and_cache_triton(gpu_key, cache, slots, 128, "ue8m0")
                    outputs.append(torch.cat([v.contiguous().view(torch.uint8).flatten()
                        for v in (q, scale, scaled, cache)]).cpu())
                name = f"m{m}.ctx{ctx}.{'mixed' if mixed else 'live'}.idp"
                record = write_indexer_decode_case(args.output.parent / name, x, w, k, pos, lengths, ctx, outputs[0])
                record["repeat_bitwise"] = tensor_digest(outputs[0]) == tensor_digest(outputs[1])
                cases.append(record)
    passed = all(c["repeat_bitwise"] for c in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="conditioned post-RoPE decode preparation: UE8M0 query/weights and one current-key append per active slot; inactive inputs poisoned, untouched cache bytes included; not packet/serving qualification",
            passed=passed, precision_qualified=False, vllm_version=version, cases=cases,
            source_reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def indexer_native_lengths(m, ctx, profile):
    if m not in (1, 8, 16, 32, 64) or ctx not in (8192, 71680, 81920, 131072):
        raise ValueError("unsupported native indexer fixture geometry")
    if profile == "full":
        return torch.full((m,), ctx, dtype=torch.int32)
    if profile == "inactive":
        return torch.zeros(m, dtype=torch.int32)
    if profile == "headroom" and ctx >= 81920:
        boundaries = [8192, 8193, 8447, 8448, 71680, 71681, 71935, 71936]
        return torch.tensor((boundaries * math.ceil(m / len(boundaries)))[:m], dtype=torch.int32)
    if profile not in ("ragged", "mixed"):
        raise ValueError("unsupported native indexer profile")
    boundaries = [ctx - 1, 1, 15, 16, 17, 255, 256, 257, 2047, 2048, 2049, ctx // 2]
    lengths = torch.tensor((boundaries * math.ceil(m / len(boundaries)))[:m], dtype=torch.int32)
    if profile == "mixed":
        lengths[::3] = 0
    return lengths


def poison_indexer_tail(cache, lengths):
    m = lengths.numel()
    if (m == 0 or lengths.dtype != torch.int32 or lengths.shape != (m,)
            or cache.dtype != torch.uint8 or cache.ndim != 3 or cache.shape[1:] != (16, 132)
            or cache.shape[0] % m):
        raise ValueError("invalid packed indexer tail geometry")
    blocks = cache.shape[0] // m
    if bool((lengths < 0).any() or (lengths > blocks * 16).any()):
        raise ValueError("invalid packed indexer lengths")
    out = cache.clone().view(m, blocks, 2112)
    tail = torch.arange(blocks * 16, device=cache.device).view(1, blocks, 16) >= lengths[:, None, None]
    out[:, :, :2048].view(m, blocks, 8, 16, 16).masked_fill_(tail[:, :, None, :, None], 0xff)
    out[:, :, 2048:].view(m, blocks, 16, 4).masked_fill_(tail[:, :, :, None], 0xff)
    return out.view_as(cache)


def indexer_prefill_cases(live_query_rows=False):
    cases = [(rows, base, rows - int(base != 0))
             for rows in (128, 512, 1024, 2048, 4096, 8192) for base in (0, 71680)]
    cases += [(128, 15, 125), (512, 8191, 511), (128, 0, 1),
              (128, 15, 0), (128, 131071, 1), (128, 131056, 16)]
    if live_query_rows:
        cases += [(8192, 71680, live) for live in (1, 4096, 4097)]
    return cases


def indexer_prefill_geometry(rows, ctx, base, live, slots, slot):
    if (rows not in (128, 512, 1024, 2048, 4096, 8192) or ctx != 131072
            or not 0 <= live <= rows or not 0 <= base <= ctx
            or not 0 < base + live <= ctx or not 1 <= slots <= 64 or not 0 <= slot < slots):
        raise ValueError("unsupported prefill indexer fixture geometry")
    n = base + live
    stride = (n + 255) // 256 * 256
    inputs = [rows * 8192, rows * 64, rows * 256, slots * ctx * 132]
    outputs = [rows * 4096, rows * 128, rows * 128, slots * ctx * 132,
               n * 128, n * 4, rows * 4, rows * 4, rows * stride * 4]
    return stride, inputs, outputs


def digest_segment(stream, size):
    digest = hashlib.sha256()
    while size:
        chunk = stream.read(min(size, 8 * 1024 * 1024))
        if not chunk:
            raise ValueError("truncated fixture segment")
        size -= len(chunk)
        digest.update(chunk)
    return digest.hexdigest()


def check_indexer_prefill(args):
    if args.reference_json is None:
        raise ValueError("requires frozen prefill reference")
    audit = json.loads(args.reference_json.read_text())
    live_query_rows = audit.get("live_query_rows", False)
    wanted = {f"t{t}.base{b}.live{l}.prefill.bin": (t, b, l)
              for t, b, l in indexer_prefill_cases(live_query_rows)}
    if (audit["vllm_version"] != "0.29.0" or not audit["passed"]
            or len(audit["cases"]) != len(wanted) or {c["file"] for c in audit["cases"]} != set(wanted)):
        raise ValueError("incomplete prefill reference coverage")
    prep_hash = hashlib.sha256((args.capture / "indexer_prefill_gfx950.elf").read_bytes()).hexdigest()
    if prep_hash != "38a75b92793716ff023d9bee7011aa680697a5e76563897cfbebb3ce1921c803":
        raise ValueError("prefill preparation code object identity mismatch")
    records = []
    for case in audit["cases"]:
        t, base, live = wanted[case["file"]]
        if ((case["rows"], case["base"], case["live"], case["ctx"], case["slots"], case["slot"])
                != (t, base, live, 131072, 3, 1) or not case["repeat_bitwise"] or not case["repeat_kernel_identical"]):
            raise ValueError("prefill reference geometry/stability mismatch")
        stride, inputs, outputs = indexer_prefill_geometry(t, 131072, base, live, 3, 1)
        score_rows = live if live_query_rows else t
        use_buffer = score_rows * (base + live) * 4 < 2**31
        score_hash = ("279e8359e2f5002ab46bd13388598c194f7dbc1984a7babc60f624553ad4cd55" if use_buffer
                      else "87862e1204f8c16c533917f2e6ab9070467a3192f1e1c2264acb7795f19699f1")
        kernel = case["kernel"]
        if score_rows == 0:
            if kernel is not None:
                raise ValueError("empty prefill must not dispatch scoring")
            score_hash = None
        elif (kernel["sha256"] != score_hash or kernel["file"] != f"prefill-score-{score_hash}.elf"
                or hashlib.sha256((args.reference_json.parent / kernel["file"]).read_bytes()).hexdigest() != score_hash
                or kernel["symbol"] != "_gluon_fp8_mqa_logits_kernel" or kernel["shared"] != 12416
                or kernel["num_warps"] != 1 or kernel["chains"] != 0 or not kernel["padded_shared"]
                or not kernel["buffer_load"] or kernel["buffer_store"] != use_buffer or kernel["grid"] != [score_rows]):
            raise ValueError("prefill score code object/launch identity mismatch")
        path = args.reference_json.parent / case["file"]
        if path.stat().st_size != 32 + sum(inputs) + sum(outputs):
            raise ValueError("prefill fixture size mismatch")
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != case["sha256"]:
                raise ValueError("prefill fixture identity mismatch")
            f.seek(0)
            if struct.unpack("<8I", f.read(32)) != (0x49504631, t, 131072, base, live, 3, 1, stride):
                raise ValueError("prefill fixture header mismatch")
            f.seek(sum(inputs), 1)
            expected = [digest_segment(f, size) for size in outputs]
        if expected != case["output_sha256"]:
            raise ValueError("prefill reference output identity mismatch")
        stem = Path(case["file"]).stem
        with (args.capture / f"{stem}.out").open("rb") as f:
            actual = [digest_segment(f, size) for size in outputs]
            if f.read(1):
                raise ValueError("prefill replay has trailing output")
        log = (args.capture / f"{stem}.replay.log").read_text()
        guard_lines = [line for line in log.splitlines() if line.startswith("indexer prefill ")]
        guards = guard_lines == [f"indexer prefill T={t} base={base} live={live} slot=1/3 run={run} mismatched_bytes=0 guard_failures=0" for run in range(2)]
        records.append(dict(file=case["file"], bitwise=actual == expected, guards=guards,
            output_sha256=actual, replay_log_sha256=hashlib.sha256(log.encode()).hexdigest(),
            score_object_sha256=score_hash))
    passed = all(r["bitwise"] and r["guards"] for r in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="native fused query/key quantization, cache insertion/history gather and frozen installed prefill scorer; conditioned operands, causal score prefixes, not model/packet/top-k/serving qualification",
            passed=passed, precision_qualified=False, cases=records,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            preparation_sha256=prep_hash,
            binary_sha256=hashlib.sha256((args.capture / "block_fp8_test").read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def export_indexer_prefill(args, version):
    import importlib
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.v1.worker.workspace import init_workspace_manager
    if version != "0.29.0" or args.reference_json is None:
        raise ValueError("requires pinned decode reference")
    init_workspace_manager(torch.device("cuda"))
    source = json.loads(args.reference_json.read_text())
    history_report = json.loads((args.capture / "reference.json").read_text())
    if (not source["passed"] or source["vllm_version"] != version
            or history_report["vllm_version"] != version):
        raise ValueError("unqualified indexer source")
    pipeline = next(p for p in history_report["pipelines"] if p["shape"] == [16, 71680, 32, 128])
    hs = pipeline["artifacts"]["history"]
    history, digest = load_tensor(args.capture / hs["file"], torch.bfloat16, (16, 71680, 128))
    if digest != hs["sha256"] or not pipeline["finite"] or not pipeline["backend_repeat"]["cache"]:
        raise ValueError("history identity mismatch")
    history = history[0].clone().cuda()
    spec = next(c for c in source["cases"] if c["file"] == "m64.ctx64.live.idp")
    raw = bytearray((args.reference_json.parent / spec["file"]).read_bytes())
    if (hashlib.sha256(raw).hexdigest() != spec["sha256"] or not spec["repeat_bitwise"]
            or struct.unpack_from("<4I", raw) != (0x49445031, 64, 64, 0)):
        raise ValueError("decode operand identity mismatch")
    query = torch.frombuffer(raw, dtype=torch.bfloat16, count=64 * 4096, offset=16).clone().cuda().reshape(64, 32, 128)
    weight = torch.frombuffer(raw, dtype=torch.bfloat16, count=64 * 32, offset=16 + 64 * 8192).clone().cuda().reshape(64, 32)
    key = torch.frombuffer(raw, dtype=torch.bfloat16, count=64 * 128, offset=16 + 64 * 8256).clone().cuda().reshape(64, 128)
    module = importlib.import_module("aiter.ops.triton.attention.fp8_mqa_logits")
    original = module._gluon_fp8_mqa_logits_kernel
    if original is None or module.arch != "gfx950":
        raise ValueError("requires installed gfx950 prefill scorer")
    captured = []

    class CaptureKernel:
        def __getitem__(self, grid):
            def launch(*values, **kwargs):
                kernel = original[grid](*values, **kwargs)
                image = kernel.asm["hsaco"]
                sha = hashlib.sha256(image).hexdigest()
                path = args.output.parent / f"prefill-score-{sha}.elf"
                if not path.exists():
                    with path.open("xb") as f:
                        f.write(image)
                record = dict(file=path.name, sha256=sha, symbol=kernel.metadata.name,
                    shared=kernel.metadata.shared, num_warps=kernel.metadata.num_warps,
                    buffer_load=kwargs["USE_BUFFER_LOAD"], buffer_store=kwargs["USE_BUFFER_STORE"],
                    chains=kwargs["NUM_CHAINS"], padded_shared=kwargs["USE_PADDED_SHARED_LAYOUT"],
                    grid=list(grid), signature={str(k): str(v) for k, v in kernel.src.signature.items()},
                    constants={str(k): str(v) for k, v in kernel.src.constants.items()})
                captured.append(record)
                return kernel
            return launch

    ctx, slots, slot = 131072, 3, 1
    records = []
    module._gluon_fp8_mqa_logits_kernel = CaptureKernel()
    try:
        for rows, base, live in indexer_prefill_cases(live_query_rows=True):
            stride, input_sizes, output_sizes = indexer_prefill_geometry(rows, ctx, base, live, slots, slot)
            n = base + live
            x, w, k = query.repeat(rows // 64, 1, 1), weight.repeat(rows // 64, 1), key.repeat(rows // 64, 1)
            for v in (x, w, k):
                v[live:] = float("nan")
            masked_q, masked_w = x.clone(), w.clone()
            masked_q[live:] = 0
            masked_w[live:] = 0
            initial = torch.full((slots * ctx // 16, 16, 132), 0xff, dtype=torch.uint8, device="cuda")
            initial[:ctx // 16].fill_(0x5a)
            initial[2 * ctx // 16:].fill_(0xa5)
            if base:
                old_keys = history[torch.arange(base, device="cuda") % history.shape[0]]
                old_slots = torch.arange(base, device="cuda", dtype=torch.int64) + slot * ctx
                ops.indexer_k_quant_and_cache_triton(old_keys, initial, old_slots, 128, "ue8m0")
            current = torch.arange(rows, device="cuda", dtype=torch.int64) + slot * ctx + base
            current[live:] = -1
            table = (torch.arange(ctx // 16, dtype=torch.int32, device="cuda") + slot * ctx // 16).reshape(1, -1)
            cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
            token_to_seq = torch.zeros(n, dtype=torch.int32, device="cuda")
            starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
            ends = torch.arange(rows, dtype=torch.int32, device="cuda") + base + 1
            ends[live:] = 0
            output_hashes = []
            score_kernels = []
            path = args.output.parent / f"t{rows}.base{base}.live{live}.prefill.bin"
            with path.open("xb") as f:
                f.write(struct.pack("<8I", 0x49504631, rows, ctx, base, live, slots, slot, stride))
                for v, size in zip((x, w, k, initial), input_sizes):
                    value = v.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    if len(value) != size:
                        raise ValueError("prefill input size mismatch")
                    f.write(value)
                for repeat in range(2):
                    q, scale = per_token_group_quant_fp8(masked_q.reshape(-1, 128), 128, use_ue8m0=True)
                    scaled = masked_w.float().flatten() * scale.flatten() * (128 ** -0.5) * (32 ** -0.5)
                    cache = initial.clone()
                    ops.indexer_k_quant_and_cache_triton(k, cache, current, 128, "ue8m0")
                    keys = torch.empty((n, 128), dtype=torch.float8_e4m3fn, device="cuda")
                    key_scales = torch.empty((n, 4), dtype=torch.uint8, device="cuda")
                    ops.cp_gather_indexer_k_quant_cache_triton(cache, keys, key_scales, table, cu, token_to_seq)
                    before = len(captured)
                    canonical = torch.zeros((rows, stride), dtype=torch.float32, device="cuda")
                    if live:
                        scores = ops.rocm_fp8_mqa_logits(q.reshape(rows, 32, 128)[:live],
                            (keys, key_scales.view(torch.float32).flatten()),
                            scaled.reshape(rows, 32)[:live], starts[:live], ends[:live])
                        if len(captured) != before + 1 or scores.stride() != (stride, 1):
                            raise ValueError("installed prefill kernel/stride mismatch")
                        valid = torch.arange(n, device="cuda")[None] < ends[:live, None]
                        canonical[:live, :n] = scores.masked_fill(~valid, 0)
                    score_kernels.append(captured[-1] if live else None)
                    values = (q, scale, scaled, cache, keys, key_scales, starts, ends, canonical)
                    if not all(bool(torch.isfinite(v.float()).all()) for v in (q, scale, scaled, keys, key_scales.view(torch.float32), canonical)):
                        raise ValueError("nonfinite prefill oracle")
                    hashes = []
                    for v, size in zip(values, output_sizes):
                        value = v.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        if len(value) != size:
                            raise ValueError("prefill output size mismatch")
                        hashes.append(hashlib.sha256(value).hexdigest())
                        if not repeat:
                            f.write(value)
                    output_hashes.append(hashes)
            with path.open("rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            record = dict(file=path.name, sha256=sha, rows=rows, base=base, live=live, ctx=ctx,
                slots=slots, slot=slot, stride=stride, output_sha256=output_hashes[0],
                repeat_bitwise=output_hashes[0] == output_hashes[1], kernel=score_kernels[0],
                repeat_kernel_identical=score_kernels[0] == score_kernels[1])
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        module._gluon_fp8_mqa_logits_kernel = original
    passed = all(c["repeat_bitwise"] and c["repeat_kernel_identical"] for c in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="conditioned repeated post-RoPE decode operands used as prefill inputs; installed key append/gather/prefill scorer with causal prefixes and isolated cache slot; not actual prefill projections/top-k/packet/serving qualification",
            passed=passed, precision_qualified=False, vllm_version=version, cases=records, live_query_rows=True,
            history_sha256=digest, source_operand_sha256=spec["sha256"],
            source_reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            history_reference_sha256=hashlib.sha256((args.capture / "reference.json").read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def indexer_selection_evidence(scores, starts, ends, indices):
    rows, stride = scores.shape
    if (scores.dtype != torch.float32 or indices.dtype != torch.int32 or indices.ndim != 2
            or indices.shape[0] != rows or starts.shape != (rows,) or ends.shape != (rows,)
            or starts.dtype != torch.int32 or ends.dtype != torch.int32
            or bool(((starts < 0) | (ends < starts) | (ends > stride)).any())):
        raise ValueError("invalid selection geometry")
    k = indices.shape[1]
    if not k:
        raise ValueError("empty selection width")
    lengths = ends - starts
    positions = torch.arange(stride, device=scores.device)[None]
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    finite = bool((torch.isfinite(scores) | ~valid).all())
    selected = indices >= 0
    in_range = bool(((indices == -1) | (selected & (indices < lengths[:, None]))).all())
    counts = bool((selected.sum(dim=1) == lengths.clamp(max=k)).all())
    ordered = indices.sort(dim=1).values
    unique = bool(((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all())
    if not (finite and in_range and counts and unique):
        return dict(passed=False, finite=finite, in_range=in_range, counts=counts, unique=unique)
    absolute = (indices.long().clamp_min(0) + starts[:, None]).clamp(max=stride - 1)
    chosen = scores.gather(1, absolute).masked_fill(~selected, float("inf"))
    threshold = chosen.min(dim=1).values
    above = valid & (scores > threshold[:, None])
    equal = valid & (scores == threshold[:, None])
    chosen_above = selected & (chosen > threshold[:, None])
    chosen_equal = selected & (chosen == threshold[:, None])
    exact = above.sum(dim=1) == chosen_above.sum(dim=1)
    ambiguous = (equal.sum(dim=1) > chosen_equal.sum(dim=1)) & (lengths > k)
    last_tie = absolute.masked_fill(~chosen_equal, -1).max(dim=1).values
    lowest = (equal & (positions <= last_tie[:, None])).sum(dim=1) == chosen_equal.sum(dim=1)
    want_short = torch.arange(k, device=scores.device)[None].expand(rows, -1)
    want_short = want_short.masked_fill(want_short >= lengths[:, None], -1)
    short_identity = bool(((indices == want_short) | (lengths[:, None] > k)).all())
    return dict(passed=bool(exact.all()) and short_identity, finite=finite, in_range=in_range,
        counts=counts, unique=unique, short_identity=short_identity,
        exact_rows=int(exact.sum()), ambiguous_rows=int(ambiguous.sum()),
        ambiguous_mask=ambiguous.cpu().tolist(),
        lowest_index_ties_rows=int(lowest.sum()), rows=rows)


def indexer_selection_scores(rows, stride, profile, device="cpu"):
    positions = torch.arange(stride, device=device, dtype=torch.int64)[None]
    row = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    if profile == "unique":
        return ((positions * 65537 + row * 17) % 262139).to(torch.float32)
    if profile == "equal":
        return torch.ones((rows, stride), dtype=torch.float32, device=device)
    if profile == "boundary-tie":
        return (1 + (positions < 1024).to(torch.float32)).expand(rows, -1).contiguous()
    if profile == "signed-zero":
        return torch.where(positions % 2 == 0, 0.0, -0.0).expand(rows, -1).contiguous()
    raise ValueError("unsupported selection score profile")


class IndexerSelectionModule:
    def __init__(self, path, symbol=b"plow_indexer_select_bounds"):
        import ctypes as c
        prop = torch.cuda.get_device_properties(0)
        if not prop.gcnArchName.startswith("gfx950") or prop.multi_processor_count != 256:
            raise ValueError("native selection requires gfx950/256CU")
        self.c = c
        self.lib = c.CDLL("libamdhip64.so")
        ptr = c.c_void_p
        self.lib.hipModuleLoad.argtypes = [c.POINTER(ptr), c.c_char_p]
        self.lib.hipModuleGetFunction.argtypes = [c.POINTER(ptr), ptr, c.c_char_p]
        self.lib.hipModuleLaunchKernel.argtypes = [ptr] + [c.c_uint] * 7 + [ptr] * 3
        self.lib.hipModuleUnload.argtypes = [ptr]
        self.module, self.function = ptr(), ptr()
        self.call("hipModuleLoad", c.byref(self.module), str(path).encode())
        self.call("hipModuleGetFunction", c.byref(self.function), self.module, symbol)

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name}: hipError_t {status}")

    def launch(self, indices, scores, lengths):
        c = self.c
        rows, stride = scores.shape
        if (scores.dtype != torch.float32 or indices.dtype != torch.int32
                or lengths.dtype != torch.int32 or lengths.shape != (rows,)
                or indices.shape != (rows, 2048) or not scores.is_contiguous()
                or not indices.is_contiguous() or not lengths.is_contiguous()
                or not scores.is_cuda or scores.device != indices.device or scores.device != lengths.device
                or rows < 1 or stride not in (8192, 71680, 81920, 131072)
                or bool(((lengths < 0) | (lengths > stride)).any())):
            raise ValueError("invalid native selection inputs")
        values = [c.c_void_p(v.data_ptr()) for v in (indices, scores, lengths)]
        values += [c.c_uint(rows), c.c_uint(stride)]
        params = (c.c_void_p * len(values))(*(c.cast(c.byref(v), c.c_void_p) for v in values))
        self.call("hipModuleLaunchKernel", self.function, rows, 1, 1, 512, 1, 1, 0,
            c.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)

    def close(self):
        torch.cuda.synchronize()
        self.call("hipModuleUnload", self.module)


def export_indexer_selection(args, version):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    if version != "0.29.0":
        raise ValueError("requires installed vLLM0.29")
    source_path = args.capture / "reference.json"
    source = json.loads(source_path.read_text())
    if source["vllm_version"] != version:
        raise ValueError("wrong selection source version")
    real = {}
    for ctx in (8192, 71680):
        case = next(c for c in source["pipelines"] if c["shape"] == [16, ctx, 32, 128])
        spec = case["artifacts"]["logits"]
        value, digest = load_tensor(args.capture / spec["file"], torch.float32, (16, ctx))
        if digest != spec["sha256"] or not case["backend_repeat"]["logits"] or not case["finite"]:
            raise ValueError("unqualified selection score source")
        real[ctx] = value.cuda()
    cases = [("decode", rows, ctx, profile, 0) for rows in (1, 8, 16, 32, 64)
             for ctx in (8192, 71680, 81920, 131072) for profile in ("real", "ragged")]
    cases += [("decode", 8, ctx, profile, 0) for ctx in (8192, 71680, 81920, 131072)
              for profile in ("unique", "equal", "boundary-tie", "signed-zero")]
    cases += [("prefill", rows, 131072, "unique", base)
              for rows in (128, 512, 1024, 2048, 4096, 8192) for base in (0, 71680)]
    native = None
    if args.native_indexer_selection is not None:
        if args.reference_json is None:
            raise ValueError("native selection requires frozen reference")
        frozen_bytes = args.reference_json.read_bytes()
        if hashlib.sha256(frozen_bytes).hexdigest() != "13012cc41d87b05bbf0a37146f6779ca9c569f0b3cc131065d2641eccc79112e":
            raise ValueError("selection reference identity mismatch")
        frozen = json.loads(frozen_bytes)
        frozen_cases = {(c["mode"], c["rows"], c["stride"], c["profile"], c["base"]): c
                        for c in frozen["cases"]}
        if not frozen["passed"] or set(frozen_cases) != set(cases):
            raise ValueError("selection reference coverage mismatch")
        native = IndexerSelectionModule(args.native_indexer_selection)
    records = []
    for mode, rows, stride, profile, base in cases:
        if profile in ("real", "ragged"):
            ctx = min(stride, 71680)
            scores = real[ctx][torch.arange(rows, device="cuda") % 16][:, torch.arange(stride, device="cuda") % ctx]
        else:
            scores = indexer_selection_scores(rows, stride, profile, "cuda")
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        if mode == "prefill":
            ends = torch.arange(rows, dtype=torch.int32, device="cuda") + base + 1
            ends[-1] = 0
        elif profile == "ragged":
            ends = indexer_native_lengths(rows, stride, "mixed").cuda()
        else:
            ends = torch.full((rows,), stride, dtype=torch.int32, device="cuda")
        scores.masked_fill_(torch.arange(stride, device="cuda")[None] >= ends[:, None], float("nan"))
        inputs_hash = tensor_digest(scores)
        bounds_hash = [tensor_digest(v) for v in (starts, ends)]
        outputs, evidence = [], []
        native_outputs, native_evidence = [], []
        for poison in (-7, -19):
            guarded = torch.full((rows + 2, 2048), poison, dtype=torch.int32, device="cuda")
            indices = guarded[1:-1]
            if ops._get_aiter_top_k_kernel(is_prefill=mode == "prefill", compress_ratio=1,
                    num_rows=rows, max_valid_seq_len=stride) is not None:
                raise ValueError("unexpected compressed selection dispatch")
            if mode == "prefill":
                torch.ops._C.top_k_per_row_prefill(scores, starts, ends, indices, rows,
                    scores.stride(0), scores.stride(1), 2048)
            else:
                torch.ops._C.top_k_per_row_decode(scores, 1, ends, indices, rows,
                    scores.stride(0), scores.stride(1), 2048)
            proof = indexer_selection_evidence(scores, starts, ends, indices)
            proof["guards"] = bool((guarded[0] == poison).all() and (guarded[-1] == poison).all())
            evidence.append(proof)
            outputs.append(indices.cpu().clone())
            if native is not None:
                guarded.fill_(poison)
                native.launch(indices, scores, ends)
                proof = indexer_selection_evidence(scores, starts, ends, indices)
                proof["guards"] = bool((guarded[0] == poison).all() and (guarded[-1] == poison).all())
                native_evidence.append(proof)
                native_outputs.append(indices.cpu().clone())
        name = f"{mode}.m{rows}.ctx{stride}.base{base}.{profile}.indices.bin"
        with (args.output.parent / name).open("xb") as f:
            for value in outputs:
                f.write(value.numpy().tobytes())
        unchanged = (tensor_digest(scores) == inputs_hash
            and [tensor_digest(v) for v in (starts, ends)] == bounds_hash)
        record = dict(mode=mode, rows=rows, stride=stride, profile=profile, base=base,
            starts=starts.cpu().tolist(), ends=ends.cpu().tolist(), scores_sha256=inputs_hash,
            indices_file=name, indices_sha256=[tensor_digest(v) for v in outputs], evidence=evidence,
            repeat_bitwise=torch.equal(*outputs),
            repeat_set=torch.equal(*(v.sort(dim=1).values for v in outputs)), inputs_unchanged=unchanged)
        record["passed"] = unchanged and all(e["passed"] and e["guards"] for e in evidence)
        if native is not None:
            old = frozen_cases[(mode, rows, stride, profile, base)]
            if (old["scores_sha256"] != inputs_hash or old["starts"] != record["starts"]
                    or old["ends"] != record["ends"]):
                raise ValueError("native selection operands differ from frozen reference")
            same_sets = [(a.sort(dim=1).values == b.sort(dim=1).values).all(dim=1)
                         for a, b in zip(outputs, native_outputs)]
            unambiguous = all(bool((same | torch.tensor(e["ambiguous_mask"])).all())
                              for same, e in zip(same_sets, evidence))
            name = name.replace(".indices.bin", ".native.indices.bin")
            with (args.output.parent / name).open("xb") as f:
                for value in native_outputs:
                    f.write(value.numpy().tobytes())
            record.update(native_indices_file=name, native_indices_sha256=[tensor_digest(v) for v in native_outputs],
                native_evidence=native_evidence, native_same_set_rows=[int(v.sum()) for v in same_sets],
                native_unambiguous_sets_equal=unambiguous,
                native_repeat_set=torch.equal(*(v.sort(dim=1).values for v in native_outputs)))
            record["passed"] &= (unambiguous and record["native_repeat_set"]
                and all(e["passed"] and e["guards"] for e in native_evidence))
        records.append(record)
        print(json.dumps({k: record[k] for k in ("mode", "rows", "stride", "profile", "base", "passed", "repeat_set")}), flush=True)
    passed = all(c["passed"] for c in records)
    if native is not None:
        native.close()
    with args.output.open("x") as f:
        json.dump(dict(scope="installed top-k and optional standalone Plow radix selector on conditioned/adversarial scores; not runtime selection/attention or full-model qualification",
            passed=passed, precision_qualified=False, vllm_version=version, cases=records,
            native_object_sha256=None if native is None else hashlib.sha256(args.native_indexer_selection.read_bytes()).hexdigest(),
            frozen_reference_sha256=None if native is None else hashlib.sha256(frozen_bytes).hexdigest(),
            source_reference_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def export_indexer_model_selection(args, version):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    if version != "0.29.0" or args.reference_json is None or args.native_indexer_selection is None:
        raise ValueError("requires pinned model-native reference and native selector")
    source_bytes = args.reference_json.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != "22f7033058f2496d303161b2f9d85c9d7846e2918cae73ca4ac646176123735d":
        raise ValueError("wrong model-native reference")
    source = json.loads(source_bytes)
    captured = {}
    for path in (args.capture / "tensors").glob("*.json"):
        spec = json.loads(path.read_text())
        if spec["semantic"] == "indexer.selected":
            captured[spec["invocation_index"]] = spec
    native = IndexerSelectionModule(args.native_indexer_selection)
    records = []
    try:
        for case in source["cases"]:
            raw = (args.reference_json.parent / case["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != case["sha256"]:
                raise ValueError("model score fixture hash mismatch")
            stride, length = case["ctx"], case["lengths"][0]
            scores = torch.frombuffer(bytearray(raw[-stride * 4:]), dtype=torch.float32).reshape(1, stride).cuda()
            scores[:, length:] = float("nan")
            starts = torch.zeros(1, dtype=torch.int32, device="cuda")
            ends = torch.tensor([length], dtype=torch.int32, device="cuda")
            spec = captured[case["invocation"]]
            if spec["rank"] != 0 or spec["layer"] != 6 or spec["context"]["max_seq_len"] != length:
                raise ValueError("wrong captured selection identity")
            model = load_indexer_model_tensor(args.capture, spec, torch.int32, spec["source_shape"])[:1].cuda()
            model_proof = indexer_selection_evidence(scores, starts, ends, model)
            expected_set = model.sort(dim=1).values
            inputs_hash = [tensor_digest(v) for v in (scores, starts, ends)]
            runs = []
            for poison in (-7, -19):
                for implementation in ("vllm", "plow"):
                    guarded = torch.full((3, 2048), poison, dtype=torch.int32, device="cuda")
                    indices = guarded[1:2]
                    if implementation == "vllm":
                        if ops._get_aiter_top_k_kernel(is_prefill=False, compress_ratio=1,
                                num_rows=1, max_valid_seq_len=stride) is not None:
                            raise ValueError("unexpected selection dispatch")
                        torch.ops._C.top_k_per_row_decode(scores, 1, ends, indices, 1,
                                                          scores.stride(0), scores.stride(1), 2048)
                    else:
                        native.launch(indices, scores, ends)
                    proof = indexer_selection_evidence(scores, starts, ends, indices)
                    proof["guards"] = bool((guarded[0] == poison).all() and (guarded[-1] == poison).all())
                    filename = f"invocation{case['invocation']}.{implementation}.{abs(poison)}.indices.bin"
                    with (args.output.parent / filename).open("xb") as stream:
                        stream.write(indices.cpu().numpy().tobytes())
                    chosen_set = indices.sort(dim=1).values
                    runs.append(dict(implementation=implementation, poison=poison, evidence=proof,
                        same_order_as_model=torch.equal(indices, model), same_set_as_model=torch.equal(chosen_set, expected_set),
                        model_overlap=int(torch.isin(indices, model).sum()), file=filename, sha256=tensor_digest(indices)))
            unchanged = inputs_hash == [tensor_digest(v) for v in (scores, starts, ends)]
            passed = model_proof["passed"] and unchanged and all(r["evidence"]["passed"] and r["evidence"]["guards"] for r in runs)
            record = dict(invocation=case["invocation"], length=length, model_evidence=model_proof,
                          inputs_unchanged=unchanged, passed=passed, runs=runs)
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        native.close()
    passed = all(c["passed"] for c in records)
    with args.output.open("x") as stream:
        json.dump(dict(passed=passed, precision_qualified=False, vllm_version=version,
            scope="standalone Plow selector on exact model-native scores; set/order comparisons, not attention qualification",
            cases=records, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            native_object_sha256=hashlib.sha256(args.native_indexer_selection.read_bytes()).hexdigest()), stream, indent=2)
    return 0 if passed else 1


def compact_indexer_model_cache(cache, table, length, capacity):
    if (cache.dtype != torch.uint8 or cache.ndim != 3 or tuple(cache.shape[1:]) != (16, 132)
            or table.dtype != torch.int32 or table.ndim != 2 or table.shape[0] != 1
            or not 0 < length <= capacity or capacity % 16):
        raise ValueError("invalid model cache geometry")
    count = (length + 15) // 16
    pages = table[0, :count].long()
    if (pages.numel() != count or pages.unique().numel() != count
            or bool((pages < 0).any() or (pages >= cache.shape[0]).any())):
        raise ValueError("invalid model cache pages")
    compact = torch.full((capacity // 16, 16, 132), 255, dtype=torch.uint8)
    compact[:count] = cache[pages]
    return compact


def export_indexer_model_native(args, version):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.v1.worker.workspace import init_workspace_manager

    if version != "0.29.0" or args.reference_json is None:
        raise ValueError("requires original-input projection replay")
    source = json.loads(args.reference_json.read_text())
    manifest = json.loads((args.capture / "reference/manifest.json").read_text())
    if (not source["passed"] or source["vllm_version"] != version
            or Path(source["capture"]).resolve() != args.capture.resolve()
            or source["audit_sha256"] != hashlib.sha256((args.capture / "cache-audit.json").read_bytes()).hexdigest()):
        raise ValueError("unqualified model replay identity")
    init_workspace_manager(torch.device("cuda"))
    records = {}
    for path in (args.capture / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        records[record["invocation_index"], record["semantic"]] = record
    cases = []
    ctx = 131072
    for case in source["cases"]:
        if case["rows"] != 1:
            continue
        invocation, length = case["invocation"], case["length"]
        if not case["passed"] or not 8193 <= length <= 8196:
            raise ValueError("unexpected model decode case")

        def operand(name):
            spec = case["artifacts"][name]
            dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
            value, digest = load_tensor(args.reference_json.parent / spec["file"], dtype, spec["shape"])
            if digest != spec["sha256"]:
                raise ValueError("model operand hash mismatch")
            return value.cuda()

        def captured(name):
            spec = records[invocation, "indexer." + name]
            dtype = getattr(torch, spec["source_dtype"])
            return load_indexer_model_tensor(args.capture, spec, dtype, spec["source_shape"])

        query, weight, key = (operand(name) for name in ("query_bf16", "weights_bf16", "key_bf16"))
        pos = operand("positions").to(torch.int32)
        if pos.tolist() != [length - 1]:
            raise ValueError("model decode position mismatch")
        table = captured("decode.block_table")
        before = compact_indexer_model_cache(captured("cache.before"), table, length, ctx).cuda()
        original_after = captured("cache.after")
        after = compact_indexer_model_cache(original_after, table, length, ctx).cuda()
        expected_q, expected_w = captured("q_fp8").cuda(), captured("weights").cuda()
        if tensor_digest(key) != tensor_digest(captured("key_bf16")):
            raise ValueError("model key differs from replay")
        lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
        identity = torch.arange(ctx // 16, dtype=torch.int32, device="cuda")[None]
        schedule = captured("decode.schedule_metadata").cuda()
        original_scores = ops.rocm_fp8_paged_mqa_logits(expected_q[:, None], original_after.cuda().unsqueeze(2),
            expected_w, lengths, table.cuda(), schedule, max_model_len=manifest["max_model_len"])[:, :length].clone()
        if not bool(torch.isfinite(original_scores).all()):
            raise ValueError("nonfinite model indexer scores")
        outputs = []
        for _ in range(2):
            q, scale = per_token_group_quant_fp8(query.reshape(-1, 128), 128, use_ue8m0=True)
            scaled = weight.float().flatten() * scale.flatten() * (128 ** -0.5) * (32 ** -0.5)
            if tensor_digest(q.reshape(1, 32, 128)) != tensor_digest(expected_q) or tensor_digest(scaled) != tensor_digest(expected_w):
                raise ValueError("preparation reference differs from model capture")
            cache = before.clone()
            ops.indexer_k_quant_and_cache_triton(key, cache, pos.long(), 128, "ue8m0")
            if tensor_digest(cache) != tensor_digest(after):
                raise ValueError("cache append differs from actual model")
            scores = ops.rocm_fp8_paged_mqa_logits(q.reshape(1, 1, 32, 128), cache.unsqueeze(2),
                scaled.reshape(1, 32), lengths, identity, schedule, max_model_len=ctx).clone()
            if tensor_digest(scores[:, :length]) != tensor_digest(original_scores):
                raise ValueError("compacted cache scoring differs from physical model cache")
            scores[:, length:] = 0
            outputs.append(torch.cat([v.contiguous().view(torch.uint8).flatten()
                                      for v in (q, scale, scaled, cache, scores)]).cpu())
        if tensor_digest(outputs[0]) != tensor_digest(outputs[1]):
            raise ValueError("model native reference not repeatable")
        path = args.output.parent / f"model-{length}.native.bin"
        with path.open("xb") as stream:
            stream.write(struct.pack("<4I", 0x494e4431, 1, ctx, 0))
            for value in (query, weight, key, pos, lengths, before, outputs[0]):
                stream.write(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        cases.append(dict(file=path.name, rows=1, ctx=ctx, profile=f"model-{length}", lengths=[length],
                          invocation=invocation, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          expected_sha256=tensor_digest(outputs[0]), repeat_bitwise=True))
        print(json.dumps(cases[-1]), flush=True)
    if sorted(c["lengths"][0] for c in cases) != [8193, 8194, 8195, 8196]:
        raise ValueError("incomplete model decode coverage")
    with args.output.open("x") as stream:
        json.dump(dict(passed=True, vllm_version=version, model_capture=True, cases=cases,
                       scope="actual model decode operands/cache, compacted pages; installed score reference, not native replay",
                       source_reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), stream, indent=2)
    return 0


def export_indexer_native(args, version, sweep=False, headroom=False):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.v1.worker.workspace import init_workspace_manager
    if version != "0.29.0" or args.reference_json is None:
        raise ValueError("requires pinned decode reference")
    init_workspace_manager(torch.device("cuda"))
    source = json.loads(args.reference_json.read_text())
    history_report = json.loads((args.capture / "reference.json").read_text())
    if (not source["passed"] or source["vllm_version"] != version
            or history_report["vllm_version"] != version):
        raise ValueError("unqualified indexer source")
    cases = []
    for ctx in ((81920, 131072) if headroom else (8192, 71680)):
        history_ctx = min(ctx, 71680)
        pipeline = next(p for p in history_report["pipelines"] if p["shape"] == [16, history_ctx, 32, 128])
        hs = pipeline["artifacts"]["history"]
        history, digest = load_tensor(args.capture / hs["file"], torch.bfloat16, (16, history_ctx, 128))
        if digest != hs["sha256"] or not pipeline["finite"] or not pipeline["backend_repeat"]["cache"]:
            raise ValueError("history identity mismatch")
        for m in ((1, 8, 16, 32, 64) if sweep else (16,)):
            spec = next(c for c in source["cases"] if c["file"] == f"m{m}.ctx64.live.idp")
            raw = bytearray((args.reference_json.parent / spec["file"]).read_bytes())
            if (hashlib.sha256(raw).hexdigest() != spec["sha256"] or not spec["repeat_bitwise"]
                    or struct.unpack_from("<4I", raw) != (0x49445031, m, 64, 0)):
                raise ValueError("decode operand identity mismatch")
            query = torch.frombuffer(raw, dtype=torch.bfloat16, count=m * 4096, offset=16).clone().cuda().reshape(m, 32, 128)
            weight = torch.frombuffer(raw, dtype=torch.bfloat16, count=m * 32, offset=16 + m * 8192).clone().cuda().reshape(m, 32)
            key = torch.frombuffer(raw, dtype=torch.bfloat16, count=m * 128, offset=16 + m * 8256).clone().cuda().reshape(m, 128)
            initial = torch.empty((m * ctx // 16, 16, 132), dtype=torch.uint8, device="cuda")
            slots = torch.arange(m * ctx, dtype=torch.int64, device="cuda")
            h = history[torch.arange(m) % 16][:, torch.arange(ctx) % history_ctx].reshape(m * ctx, 128).cuda()
            ops.indexer_k_quant_and_cache_triton(h, initial, slots, 128, "ue8m0")
            table = torch.arange(m * ctx // 16, dtype=torch.int32, device="cuda").reshape(m, -1)
            schedule = torch.empty(0, device="cuda", dtype=torch.int32)
            profiles = ("full", "ragged", "mixed", "inactive", "headroom") if headroom else (
                ("full", "ragged", "mixed", "inactive") if sweep else ("full",))
            for profile in profiles:
                lengths = indexer_native_lengths(m, ctx, profile).cuda()
                profile_cache = poison_indexer_tail(initial, lengths)
                pos = lengths - 1
                live = lengths > 0
                x, w, k = query.clone(), weight.clone(), key.clone()
                for v in (x, w, k):
                    v[~live] = float("nan")
                masked_q, masked_w = x.clone(), w.clone()
                masked_q[~live] = 0
                masked_w[~live] = 0
                current = torch.arange(m, dtype=torch.int64, device="cuda") * ctx + pos.long()
                current[~live] = -1
                valid = torch.arange(ctx, device="cuda")[None] < lengths[:, None]
                outputs = []
                for _ in range(2):
                    q, scale = per_token_group_quant_fp8(masked_q.reshape(-1, 128), 128, use_ue8m0=True)
                    scaled = masked_w.float().flatten() * scale.flatten() * (128 ** -0.5) * (32 ** -0.5)
                    cache = profile_cache.clone()
                    ops.indexer_k_quant_and_cache_triton(k, cache, current, 128, "ue8m0")
                    scores = ops.rocm_fp8_paged_mqa_logits(q.reshape(m, 1, 32, 128), cache.unsqueeze(2),
                        scaled.reshape(m, 32), lengths, table, schedule, max_model_len=ctx).clone()
                    # The installed wrapper leaves out-of-prefix values unspecified; top-k uses lengths.
                    scores[~valid] = 0
                    if not all(bool(torch.isfinite(v.float()).all()) for v in (q, scale, scaled, scores)):
                        raise ValueError("nonfinite native-chain oracle")
                    outputs.append(torch.cat([v.contiguous().view(torch.uint8).flatten()
                                              for v in (q, scale, scaled, cache, scores)]).cpu())
                suffix = f".{profile}" if sweep else ""
                path = args.output.parent / f"m{m}.ctx{ctx}{suffix}.native.bin"
                with path.open("xb") as f:
                    f.write(struct.pack("<4I", 0x494e4431, m, ctx, 0))
                    for v in (x, w, k, pos, lengths, profile_cache, outputs[0]):
                        f.write(v.cpu().contiguous().view(torch.uint8).numpy().tobytes())
                with path.open("rb") as f:
                    fixture_hash = hashlib.file_digest(f, "sha256").hexdigest()
                cases.append(dict(file=path.name, rows=m, ctx=ctx, profile=profile,
                    lengths=lengths.cpu().tolist(), sha256=fixture_hash,
                    expected_sha256=tensor_digest(outputs[0]), history_sha256=digest, source_operand_sha256=spec["sha256"],
                    repeat_bitwise=tensor_digest(outputs[0]) == tensor_digest(outputs[1])))
                print(json.dumps({k: cases[-1][k] for k in ("file", "repeat_bitwise")}), flush=True)
    passed = all(c["repeat_bitwise"] for c in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="conditioned post-RoPE operands plus repeated history, identity pages and current-key append; scores compared only within live prefixes, inactive operands poisoned; not new RoPE/prefill/selection/serving qualification",
            passed=passed, vllm_version=version, precision_qualified=False, cases=cases, sweep=sweep, headroom=headroom,
            unused_cache_poison=255,
            source_reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            history_reference_sha256=hashlib.sha256((args.capture / "reference.json").read_bytes()).hexdigest()), f, indent=2)
    return 0 if passed else 1


def check_indexer_quant(args):
    if args.reference_json is None:
        raise ValueError("requires frozen indexer quant reference")
    audit = json.loads(args.reference_json.read_text())
    shapes = {f"m{m}.query.iqq": (0, m * 32, 0) for m in (1, 8, 16, 32, 64)}
    shapes.update({f"m{m}.key.iqq": (1, m, math.ceil(m / 16) + 2) for m in (1, 8, 16, 32, 64)})
    shapes.update({"finite-bf16.query.iqq": (0, 65280, 0), "finite-bf16.key.iqq": (1, 65280, 4081)})
    if (audit["vllm_version"] != "0.29.0" or not audit["passed"] or not audit["audit_complete"]
            or len(audit["cases"]) != len(shapes) or {c["file"] for c in audit["cases"]} != set(shapes)):
        raise ValueError("incomplete indexer quant reference coverage")
    records = []
    for case in audit["cases"]:
        raw = bytearray((args.reference_json.parent / case["file"]).read_bytes())
        if (hashlib.sha256(raw).hexdigest() != case["sha256"]
                or not all(case[k] for k in ("finite", "reference_repeat_bitwise", "indexer_forward_bitwise"))):
            raise ValueError("indexer quant reference identity/stability mismatch")
        magic, mode, rows, blocks = struct.unpack_from("<4I", raw)
        expected_size = blocks * 2112 if mode else rows * 136
        offset = 16 + rows * (256 + (8 if mode else 2))
        if (magic != 0x49515131 or [mode, rows, blocks] != [case[k] for k in ("mode", "rows", "blocks")]
                or (mode, rows, blocks) != shapes[case["file"]] or len(raw) != offset + expected_size):
            raise ValueError("indexer quant fixture header/size mismatch")
        expected = torch.frombuffer(raw, dtype=torch.uint8, offset=offset)
        if tensor_digest(expected) != case["expected_sha256"]:
            raise ValueError("indexer quant expected digest mismatch")
        actual, digest = load_tensor(args.capture / (Path(case["file"]).stem + ".out"),
                                     torch.uint8, (expected_size,))
        if mode:
            boundaries = dict(packed_cache=boundary_difference(actual.reshape(blocks, 2112), expected.reshape(blocks, 2112)))
        else:
            qend, send = rows * 128, rows * 132
            boundaries = dict(query=boundary_difference(actual[:qend].reshape(rows, 128), expected[:qend].reshape(rows, 128)),
                scales=boundary_difference(actual[qend:send].view(torch.float32).reshape(rows, 1), expected[qend:send].view(torch.float32).reshape(rows, 1)),
                weights=boundary_difference(actual[send:].view(torch.float32).reshape(rows, 1), expected[send:].view(torch.float32).reshape(rows, 1)))
        records.append(dict(file=case["file"], rows=rows, mode=mode, output_sha256=digest, boundaries=boundaries))
    passed = all(b["finite"] and b["bitwise"] for r in records for b in r["boundaries"].values())
    with args.output.open("x") as f:
        json.dump(dict(scope="native conditioned indexer UE8M0 query/weights and packed key insertion only; not score/selection/packet/serving qualification",
            passed=passed, precision_qualified=False, cases=records,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            object_sha256=hashlib.sha256((args.capture / "indexer_quant.elf").read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256((args.capture / "block_fp8_test").read_bytes()).hexdigest()), f, indent=2, allow_nan=False)
    return 0 if passed else 1


def check_indexer_decode(args):
    if args.reference_json is None:
        raise ValueError("requires frozen decode preparation reference")
    audit = json.loads(args.reference_json.read_text())
    shapes = {f"m{m}.ctx{ctx}.{profile}.idp": (m, ctx) for m in (1, 8, 16, 32, 64)
              for ctx in (64, 71680) for profile in ("live", "mixed")}
    if (audit["vllm_version"] != "0.29.0" or not audit["passed"] or len(audit["cases"]) != len(shapes)
            or {c["file"] for c in audit["cases"]} != set(shapes)):
        raise ValueError("incomplete decode preparation coverage")
    object_hash = hashlib.sha256((args.capture / "indexer_decode.elf").read_bytes()).hexdigest()
    if object_hash != "280c6a04bb9029e3ce3e6a9cf6f31578d7be3b2de470eef34dfe6f212aad3920":
        raise ValueError("decode preparation code object differs")
    cases = []
    for case in audit["cases"]:
        raw = bytearray((args.reference_json.parent / case["file"]).read_bytes())
        magic, m, ctx, reserved = struct.unpack_from("<4I", raw)
        offset = 16 + m * 8520
        size = m * 4352 + m * ctx * 132
        if (not case["repeat_bitwise"] or hashlib.sha256(raw).hexdigest() != case["sha256"]
                or magic != 0x49445031 or reserved or (m, ctx) != shapes[case["file"]]
                or [m, ctx] != [case["rows"], case["ctx"]] or len(raw) != offset + size):
            raise ValueError("decode preparation fixture identity/geometry differs")
        expected = torch.frombuffer(raw, dtype=torch.uint8, offset=offset)
        if tensor_digest(expected) != case["expected_sha256"]:
            raise ValueError("decode preparation expected output differs")
        actual, digest = load_tensor(args.capture / (Path(case["file"]).stem + ".out"), torch.uint8, (size,))
        qend, send, wend = m * 4096, m * 4224, m * 4352
        boundaries = dict(query=boundary_difference(actual[:qend].reshape(m * 32, 128), expected[:qend].reshape(m * 32, 128)),
            scales=boundary_difference(actual[qend:send].view(torch.float32).reshape(m, 32), expected[qend:send].view(torch.float32).reshape(m, 32)),
            weights=boundary_difference(actual[send:wend].view(torch.float32).reshape(m, 32), expected[send:wend].view(torch.float32).reshape(m, 32)))
        cache_hash, reference_cache_hash = tensor_digest(actual[wend:]), tensor_digest(expected[wend:])
        cases.append(dict(file=case["file"], rows=m, ctx=ctx, output_sha256=digest, boundaries=boundaries,
            cache_sha256=cache_hash, reference_cache_sha256=reference_cache_hash, cache_bitwise=cache_hash == reference_cache_hash))
    passed = all(c["cache_bitwise"] and all(b["finite"] and b["bitwise"] for b in c["boundaries"].values()) for c in cases)
    with args.output.open("x") as f:
        json.dump(dict(scope="native fused decode preparation only: query quantization, weight scaling and current-key append including inactive/cache-preservation cases; not score, runtime packet or serving qualification",
            passed=passed, precision_qualified=False, cases=cases, object_sha256=object_hash,
            binary_sha256=hashlib.sha256((args.capture / "block_fp8_test").read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), f, indent=2, allow_nan=False)
    return 0 if passed else 1


def check_indexer_score(args, cache_score=False):
    if args.reference_json is None:
        raise ValueError("requires frozen indexer score reference")
    audit = json.loads(args.reference_json.read_text())
    if (audit["vllm_version"] != "0.29.0" or len(audit["pipelines"]) != 2
            or {tuple(p["shape"]) for p in audit["pipelines"]} != {(16, 8192, 32, 128), (16, 71680, 32, 128)}):
        raise ValueError("incomplete indexer score reference coverage")
    object_hash = hashlib.sha256((args.capture / "indexer_score.elf").read_bytes()).hexdigest()
    if object_hash != "b8dfada77eb90e79572ba057c31245dfa2b3cf86f6dbb5620d7c6d0f15f24182":
        raise ValueError("indexer score code object differs from pinned reference")
    key_hash = None
    if cache_score:
        key_hash = hashlib.sha256((args.capture / "indexer_quant.elf").read_bytes()).hexdigest()
        if key_hash != "35d69f1db89beeca99e0c6fc5d06c992fec457aa5698d8f9fd82000afb10d37f":
            raise ValueError("indexer key-cache object differs from qualified native object")
    records = []
    for p in audit["pipelines"]:
        m, ctx, _, _ = p["shape"]
        if (not p["finite"] or not all(p["forward_repeat"].values())
                or not p["backend_repeat"]["cache"] or not p["backend_repeat"]["logits"]):
            raise ValueError("indexer score reference not finite/repeat-bitwise")
        for name in ("query", "cache", "weights", "block-table", "logits") + (("history",) if cache_score else ()):
            spec = p["artifacts"][name]
            if spec["file"] != f"ctx{ctx}.{name}.bin" or hashlib.sha256(
                    (args.reference_json.parent / spec["file"]).read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError("indexer score reference artifact identity mismatch")
        expected, _ = load_tensor(args.reference_json.parent / f"ctx{ctx}.logits.bin", torch.float32, (m, ctx))
        actual, digest = load_tensor(args.capture / f"ctx{ctx}.f32", torch.float32, (m, ctx))
        row = dict(shape=[m, ctx], output_sha256=digest, difference=boundary_difference(actual, expected))
        if cache_score:
            expected_cache, _ = load_tensor(args.reference_json.parent / f"ctx{ctx}.cache.bin", torch.uint8, (m * ctx // 16, 2112))
            actual_cache, _ = load_tensor(args.capture / f"ctx{ctx}.f32.cache", torch.uint8, expected_cache.shape)
            row["cache_difference"] = boundary_difference(actual_cache, expected_cache)
        records.append(row)
    passed = all(r[name]["finite"] and r[name]["bitwise"] for r in records
                 for name in (("difference", "cache_difference") if cache_score else ("difference",)))
    scope = ("native BF16 history to packed FP8 cache then pinned AOT score directly on HSA; two ordered dispatches, no intervening host wait; excludes query quantization/selection/packet/serving qualification"
             if cache_score else "prequantized FP8 indexer scoring via pinned AOT object directly on HSA; no quantization/selection/packet/serving qualification")
    with args.output.open("x") as f:
        json.dump(dict(scope=scope, passed=passed, precision_qualified=False, cases=records, object_sha256=object_hash,
            key_object_sha256=key_hash,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256((args.capture / "block_fp8_test").read_bytes()).hexdigest()), f, indent=2, allow_nan=False)
    return 0 if passed else 1


def compare_indexer_boundaries(args, meta, weights, harness, rope, sources, version):
    m, ctx, layer = meta["batch"], meta["ctx"], meta["layer"]
    if layer != args.indexer_layer:
        raise ValueError("connected indexer capture must use the audited full layer")
    positions = torch.full((m,), ctx - 1, dtype=torch.int64, device="cuda")
    records = []
    for rank in range(args.tp):
        hashes = {}
        def captured(name, dtype, shape):
            value, digest = load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape)
            hashes[name] = digest
            return value
        x = captured("act.xn", torch.bfloat16, (m, 6144))
        latent = captured("act.qlat", torch.bfloat16, (m, 2048))
        if not bool(torch.isfinite(x).all() and torch.isfinite(latent).all()):
            raise ValueError("nonfinite connected indexer inputs")
        bound_weights = {}
        for suffix in ("wq_b.weight", "wq_b.weight_scale_inv", "weights_proj.weight", "k_norm.weight", "k_norm.bias"):
            want = weights[suffix]
            bound = captured(f"model.layers.{layer}.self_attn.indexer.{suffix}", want.dtype, want.shape)
            bound_weights[suffix] = tensor_digest(bound) == tensor_digest(want)
        raw = harness.wq_b(latent.cuda())[0].reshape(m, 32, 128)
        raw_repeat = harness.wq_b(latent.cuda())[0].reshape_as(raw)
        from vllm._aiter_ops import rocm_aiter_ops
        q, scales = rocm_aiter_ops.group_fp8_quant(latent.cuda(), 128)
        q_repeat, scales_repeat = rocm_aiter_ops.group_fp8_quant(latent.cuda(), 128)
        kw = harness.wk_weights_proj(x.cuda())[0]
        kw_repeat = harness.wk_weights_proj(x.cuda())[0]
        key_raw = captured("act.kidx_raw", torch.bfloat16, (m, 128))
        norm = harness.k_norm(key_raw.cuda())
        norm_repeat = harness.k_norm(key_raw.cuda())
        query_prefix, key_prefix = rope(positions, raw[..., :64].contiguous().reshape(m, -1),
                              norm[..., :64].contiguous())
        expected_query = torch.cat((query_prefix.reshape(m, 32, 64), raw[..., 64:]), dim=-1)
        expected_key = torch.cat((key_prefix.reshape(m, 64), norm[..., 64:]), dim=-1)
        query_repeat, key_repeat = rope(positions, raw_repeat[..., :64].contiguous().reshape(m, -1),
                              norm_repeat[..., :64].contiguous())
        original_key_cache, original_cache_hash = load_tensor(
            args.capture / "inputs" / f"kv.{layer}.kidx.bin", torch.bfloat16, (m, ctx, 128))
        hashes["input_key_cache"] = original_cache_hash
        written_key_cache = torch.stack([captured(f"slot{row}.kv.{layer}.kidx",
            torch.bfloat16, (ctx, 128)) for row in range(m)])
        key_cache_checks = indexer_cache_boundaries(written_key_cache, original_key_cache, expected_key.cpu())
        cos = captured("in.icos", torch.float32, (ctx, 64))
        sin = captured("in.isin", torch.float32, (ctx, 64))
        if not (bool(torch.isfinite(cos).all() and torch.isfinite(sin).all())
                and bool((cos[:, 32:] == 1).all() and (sin[:, 32:] == 0).all())):
            raise ValueError("invalid loaded indexer identity-tail RoPE tables")
        loaded_query = raw.cpu().clone()
        loaded_query[..., :64] = rotate_interleaved(raw.cpu()[..., :64],
            torch.cat((cos[-1, :32], sin[-1, :32])))
        cache = rope.cos_sin_cache[:ctx].cpu().float()
        if cache.shape != (ctx, 64):
            raise ValueError("unexpected reference rotary cache shape")
        actual_query = captured("act.qidx", torch.bfloat16, (m, 32, 128))
        boundaries = {
            "query_quantization": boundary_difference(captured("act.qb_xq", torch.uint8, (m, 2048)), q.cpu().view(torch.uint8)),
            "query_scales": boundary_difference(captured("act.qb_xs", torch.float32, (16, m)).T.contiguous(), scales.cpu()),
            "key_projection": boundary_difference(key_raw, kw[:, :128].cpu()),
            "weights_projection": boundary_difference(captured("act.widx", torch.bfloat16, (m, 32)), kw[:, 128:].cpu()),
            "conditioned_key_norm": boundary_difference(captured("act.kidx_normed", torch.bfloat16, (m, 128)), norm.cpu()),
            "query_reference_rope": boundary_difference(actual_query, expected_query.cpu()),
            "rope_cos_coefficients": boundary_difference(cos[:, :32].contiguous(), cache[:, :32].contiguous()),
            "rope_sin_coefficients": boundary_difference(sin[:, :32].contiguous(), cache[:, 32:].contiguous()),
            "query_loaded_rope_model": boundary_difference(actual_query, loaded_query),
            **key_cache_checks,
        }
        stable = all(tensor_digest(a) == tensor_digest(b) for a, b in
                     ((raw, raw_repeat), (kw, kw_repeat), (norm, norm_repeat),
                      (q, q_repeat), (scales, scales_repeat), (query_prefix, query_repeat),
                      (key_prefix, key_repeat)))
        records.append(dict(rank=rank, input_hashes=hashes, checkpoint_bitwise=bound_weights,
                            reference_repeat_bitwise=stable, boundaries=boundaries))
        print(json.dumps(records[-1]), flush=True)
    complete = all(r["reference_repeat_bitwise"] and all(r["checkpoint_bitwise"].values())
                   and all(b["finite"] for b in r["boundaries"].values()) for r in records)
    passed = complete and all(b["bitwise"] for r in records for key, b in r["boundaries"].items()
                             if key != "query_loaded_rope_model")
    with args.output.open("x") as f:
        json.dump(dict(scope="connected captured indexer projections, query/key RoPE, BF16 key-cache write and unchanged carried history, loaded coefficients and conditioned key norm; loaded-RoPE composition model is diagnostic, not raw-WQ, FP8 cache or complete indexer/serving qualification",
            audit_complete=complete, passed=passed, precision_qualified=False, vllm_version=version,
            layer=layer, shape=[m, ctx], sources=sources,
            weights={name: dict(dtype=str(value.dtype), sha256=tensor_digest(value)) for name, value in weights.items()},
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(), cases=records),
            f, indent=2, allow_nan=False)
    return 0 if passed else 1


def indexer_cache_boundaries(actual, original, key):
    if (actual.dtype != torch.bfloat16 or original.dtype != actual.dtype or key.dtype != actual.dtype
            or actual.ndim != 3 or actual.shape != original.shape or actual.shape[1] < 2
            or actual.shape[2] != 128 or key.shape != (actual.shape[0], 128)):
        raise ValueError("invalid captured indexer key-cache geometry")
    return dict(key_reference_rope=boundary_difference(actual[:, -1].contiguous(), key),
                key_cache_carried=boundary_difference(actual[:, :-1].contiguous(), original[:, :-1].contiguous()))


def load_indexer_model_tensor(root, record, dtype, shape):
    if (record["stored_dtype"] != "raw" or record["source_dtype"] != str(dtype).removeprefix("torch.")
            or record["source_shape"] != list(shape)):
        raise ValueError("model indexer tensor contract mismatch")
    value, digest = load_tensor(root / "tensors" / record["file"], dtype, shape)
    if digest != record["sha256"]:
        raise ValueError("model indexer tensor hash mismatch")
    return value


def rebind_attention_work_metadata(work_indptr, work_info_set):
    if (work_indptr.dtype != torch.int32 or work_info_set.dtype != torch.int32
            or work_indptr.shape != (257,) or work_info_set.ndim != 2
            or work_info_set.shape[1] != 8 or not work_indptr.is_contiguous()
            or not work_info_set.is_contiguous() or work_indptr.device != work_info_set.device):
        raise ValueError("invalid captured attention work buffers")
    # AITER v1_2_pa stores these two addresses, not portable scheduling values.
    return torch.tensor([work_indptr.data_ptr(), work_info_set.data_ptr()],
                        dtype=torch.uint64, device=work_indptr.device)


def attention_physical_indices(selected, table, length, cache_tokens):
    if (selected.dtype != torch.int32 or selected.shape != (1, 2048)
            or table.dtype != torch.int32 or table.ndim != 2 or table.shape[0] != 1
            or length <= 0 or table.shape[1] * 16 < length):
        raise ValueError("invalid attention selection geometry")
    logical = selected[0].long()
    if (logical.unique().numel() != 2048
            or not bool(((logical >= 0) & (logical < length)).all())):
        raise ValueError("invalid attention selection tokens")
    physical = table[0][logical // 16].long() * 16 + logical % 16
    if not bool(((physical >= 0) & (physical < cache_tokens)).all()):
        raise ValueError("invalid attention selection pages")
    return physical.to(torch.int32)


def batched_attention_physical_indices(selected, table, length, cache_tokens):
    if (selected.ndim != 2 or not selected.shape[0] or selected.shape[1] != 2048
            or table.ndim != 2 or table.shape[0] != selected.shape[0]):
        raise ValueError("attention selection batch mismatch")
    return torch.cat([attention_physical_indices(selected[row:row+1], table[row:row+1], length, cache_tokens)
                      for row in range(selected.shape[0])])


def model_attention_selection_orders(args, manifest, audit, length, expected):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    cases = [c for c in audit["invocations"] if c["length"] == length and c["live"] == 1]
    if len(cases) != 1:
        raise ValueError("ambiguous attention/indexer invocation binding")
    invocation = cases[0]["invocation"]
    records = {}
    for path in (args.capture / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        if record["invocation_index"] == invocation and record["semantic"].startswith("indexer."):
            name = record["semantic"].removeprefix("indexer.")
            if name in records:
                raise ValueError("duplicate bound indexer input")
            records[name] = record
    hashes = {}

    def load(name):
        record = records[name]
        context = record["context"]
        if (record["rank"] != 0 or record["layer"] != 6
                or context["max_seq_len"] != length
                or record["prompt_sha256_u32le"] != manifest["requests"][0]["prompt_sha256_u32le"]
                or record["context_sha256"] != hashlib.sha256(
                    json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
            raise ValueError("bound indexer input identity mismatch")
        hashes[name] = record["sha256"]
        return load_indexer_model_tensor(args.capture, record, getattr(torch, record["source_dtype"]),
                                         record["source_shape"])

    model = load("selected")[:1].contiguous()
    if not torch.equal(model, expected):
        raise ValueError("indexer order differs from attention capture")
    query, weights, cache, table, schedule = (load(name).cuda() for name in (
        "q_fp8", "weights", "cache.after", "decode.block_table", "decode.schedule_metadata"))
    ends = load("decode.seq_lens")
    if ends.shape != (1, 1) or ends.dtype != torch.int32:
        raise ValueError("bound indexer length geometry mismatch")
    ends = ends.reshape(1).cuda()
    if query.shape != (1, 32, 128) or weights.shape != (1, 32) or ends.tolist() != [length]:
        raise ValueError("bound indexer decode geometry mismatch")
    score_inputs = [tensor_digest(v) for v in (query, weights, cache, table, schedule, ends)]
    scores = torch.full((1, 131072), float("nan"), device="cuda")
    for repeat in range(2):
        live = ops.rocm_fp8_paged_mqa_logits(query[:, None], cache.unsqueeze(2), weights,
            ends, table, schedule, max_model_len=manifest["max_model_len"])[:, :length].clone()
        if repeat == 0:
            scores[:, :length] = live
        elif tensor_digest(scores[:, :length]) != tensor_digest(live):
            raise ValueError("bound indexer scores are not repeatable")
    if score_inputs != [tensor_digest(v) for v in (query, weights, cache, table, schedule, ends)]:
        raise ValueError("bound indexer scoring changed inputs")
    starts = torch.zeros_like(ends)
    model_evidence = indexer_selection_evidence(scores, starts, ends, model.cuda())
    if not model_evidence["passed"]:
        raise ValueError("captured selection does not satisfy regenerated scores")
    if ops._get_aiter_top_k_kernel(is_prefill=False, compress_ratio=1,
            num_rows=1, max_valid_seq_len=131072) is not None:
        raise ValueError("unexpected selection dispatch")
    input_hashes = [tensor_digest(v) for v in (scores, starts, ends)]
    variants, reports = [], []
    native = IndexerSelectionModule(args.native_indexer_selection)
    try:
        for implementation in ("vllm", "plow"):
            for repeat, poison in enumerate((-7, -19)):
                guarded = torch.full((3, 2048), poison, dtype=torch.int32, device="cuda")
                indices = guarded[1:2]
                if implementation == "vllm":
                    torch.ops._C.top_k_per_row_decode(scores, 1, ends, indices, 1,
                                                      scores.stride(0), scores.stride(1), 2048)
                else:
                    native.launch(indices, scores, ends)
                proof = indexer_selection_evidence(scores, starts, ends, indices)
                if (not proof["passed"] or not bool((guarded[0] == poison).all())
                        or not bool((guarded[-1] == poison).all())):
                    raise ValueError("bound selector evidence or guard failure")
                result = indices.cpu().contiguous()
                name = f"{implementation}{repeat}"
                path = args.output.parent / f"length{length}.{name}.indices.bin"
                with path.open("xb") as stream:
                    stream.write(result.numpy().tobytes())
                variants.append((name, result))
                reports.append(dict(name=name, evidence=proof, file=path.name, sha256=tensor_digest(result),
                    same_order_as_model=torch.equal(result, model),
                    same_set_as_model=torch.equal(result.sort().values, model.sort().values)))
    finally:
        native.close()
    if input_hashes != [tensor_digest(v) for v in (scores, starts, ends)]:
        raise ValueError("bound selection changed scores or bounds")
    return variants, dict(invocation=invocation, captured_tensor_sha256=hashes,
        score_sha256=input_hashes[0], model_evidence=model_evidence, runs=reports, inputs_unchanged=True)


def replay_model_query(args, rocm_aiter_ops, version):
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS, gemm_a8w8_blockscale_ck
    from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
    from vllm.model_executor.layers.attention.mla_attention import dynamic_per_batched_tensor_quant

    if version != "0.29.0" or args.checkpoint is None or args.tp != 8:
        raise ValueError("requires pinned model capture and connected TP8 run record")
    run_bytes = args.reference_json.read_bytes() if args.reference_json else None
    native_root = args.reference_json.parent if args.reference_json else None
    manifest_bytes = (args.capture / "reference/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    m = len(manifest["requests"])
    if (manifest["vllm_version"] != version or manifest["invalid_cases"]
            or manifest["tensor_parallel_size"] != 8 or m not in (1, 8)):
        raise ValueError("requires valid original-model TP8 capture")
    if native_root:
        metadata = json.loads((native_root / "inputs/reference.json").read_text())
        if (metadata["capture_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
                or metadata["batch"] != m or metadata["ctx"] != 8193):
            raise ValueError("connected run is not bound to this model capture")
        for name, digest in json.loads(run_bytes)["inputs"].items():
            if hashlib.sha256((native_root / "inputs" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("connected input changed")
    records = {}
    for path in (args.capture / "tensors").glob("*.json"):
        r = json.loads(path.read_text())
        if r["context"]["max_seq_len"] == 8193:
            if r["semantic"] in records:
                raise ValueError("ambiguous query capture")
            records[r["semantic"]] = r

    def captured(name):
        r = records[name]
        if (r["rank"] != 0 or r["layer"] != 6
                or r["prompt_sha256_u32le"] != manifest["requests"][0]["prompt_sha256_u32le"]
                or r["context_sha256"] != hashlib.sha256(json.dumps(
                    r["context"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
            raise ValueError("wrong query capture rank/layer")
        return load_indexer_model_tensor(args.capture, r, getattr(torch, r["source_dtype"]), r["source_shape"])

    identities = {}

    def native(name, dtype, shape, rank=0):
        count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        path = native_root / "outputs" / f"rank{rank}.{name}.bin"
        with path.open("rb") as stream:
            raw = stream.read(count)
        if len(raw) != count:
            raise ValueError("short native query operand")
        identities[name if rank == 0 else f"rank{rank}.{name}"] = dict(prefix_bytes=count, allocation_bytes=path.stat().st_size,
                                prefix_sha256=hashlib.sha256(raw).hexdigest())
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)

    latent = captured("indexer.input.qr")
    if latent.shape != (m, 2048) or (native_root and not torch.equal(latent, native("act.qlat", torch.bfloat16, (m, 2048)))):
        raise ValueError("query latent is not exact model input")
    inventory_bytes = (args.capture / "reference/precision.json").read_bytes()
    inventory = json.loads(inventory_bytes)
    prefix = "model.layers.6.self_attn."
    module = inventory["ranks"][0]["modules"][prefix + "q_b_proj"]
    selected = module["attributes"]["quant_method"]["fields"]["fp8_linear"]
    if (selected["class_name"].rsplit(".", 1)[-1] != "AiterFp8BlockScaledMMKernel"
            or selected["fields"]["use_triton"] is not False
            or selected["fields"]["quant_fp8"]["fields"]["use_ue8m0"] is not False):
        raise ValueError("unexpected installed query projection backend")
    weight, scale, _ = qb_weights(args.checkpoint, 6, 0, 8)
    boundaries = {}
    for name, value in ((prefix + "q_b_proj.weight", weight), (prefix + "q_b_proj.weight_scale_inv", scale)):
        if native_root:
            boundaries[name] = boundary_difference(native(name, value.dtype, value.shape), value)
    stride = module["tensors"]["weight"]["stride"]
    gpu_weight = torch.empty_strided(weight.shape, stride, dtype=weight.dtype, device="cuda")
    gpu_weight.copy_(weight)
    xq, xs = rocm_aiter_ops.group_fp8_quant(latent.cuda(), 128)
    for name, value in (("act.qb_xq", xq), ("act.qb_xs", xs.T.contiguous())):
        if native_root:
            boundaries[name] = boundary_difference(native(name, value.dtype, value.shape), value.cpu())
    projected = [rocm_aiter_ops.gemm_a8w8_blockscale(xq, gpu_weight, xs, scale.cuda(),
        [128, 128], output_dtype=torch.bfloat16).cpu() for _ in range(2)]
    native_qb = native("act.qb", torch.bfloat16, (m, 2048)) if native_root else None
    model_qb = captured("block.qb") if "block.qb" in records else None
    if native_qb is None and model_qb is None:
        raise ValueError("query replay requires native or captured model Q-B output")
    operands = [("installed_qb0", projected[0]), ("installed_qb1", projected[1])]
    for label, qb in (("native_qb", native_qb), ("model_qb", model_qb)):
        if qb is not None:
            boundaries[label + "_vs_installed"] = [boundary_difference(qb, value) for value in projected]
            operands.append((label, qb))
    tuned = get_CKGEMM_config(m, 2048, 2048, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
    splits = 1 << int(tuned["splitK"]) if tuned is not None else 1
    if splits > 8 or 2048 % (128 * splits):
        raise ValueError("query split-K geometry is outside the audited contract")
    parts, part_records = [], []
    for part in range(splits):
        lo, hi = part * (2048 // splits), (part + 1) * (2048 // splits)
        a = xq[:, lo:hi].contiguous()
        w = gpu_weight[:, lo:hi].contiguous()
        asc = xs[:, lo // 128:hi // 128].contiguous()
        wsc = scale[:, lo // 128:hi // 128].contiguous().cuda()
        def isolated():
            y = torch.empty((m, 2048), dtype=torch.bfloat16, device="cuda")
            return gemm_a8w8_blockscale_ck(a, w, asc, wsc, y, splitK=0,
                kernelName="" if tuned is None else str(tuned["kernelName"])).cpu()
        value, repeat = isolated(), isolated()
        path = args.output.parent / f"query.part{part}.bin"
        write_case(path, a.cpu().view(torch.uint8), w.cpu(), asc.cpu().T.contiguous(), wsc.cpu(), value)
        parts.append(value)
        part_records.append(dict(part=part, k_start=lo, k_end=hi, file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            repeat_bitwise=tensor_digest(value) == tensor_digest(repeat)))
    lower, upper = bf16_order_bounds(torch.stack(parts, dim=1))
    order_bounds = {}
    for label, value in operands:
        outside = (value < lower) | (value > upper) | ~torch.isfinite(value)
        order_bounds[label] = dict(within_bounds=not bool(outside.any()), outside_count=int(outside.sum()),
            reachability=bf16_order_reachability(torch.stack(parts, dim=1), value))
    accumulation = dict(split_count=splits, parts=part_records, order_bounds=order_bounds,
        selected_config=None if tuned is None else {key: str(value) for key, value in tuned.items()})
    original, scales = mla_kvb_weights(args.checkpoint, 6, 0, 8)
    dequant = scaled_dequantize(original.cuda(), scales.cuda(), group_shape=[128, 128], out_dtype=torch.bfloat16)
    uk, uv = dequant.T.reshape(512, 8, 448).split([192, 256], dim=-1)
    wk, wk_scale = dynamic_per_batched_tensor_quant(uk.transpose(0, 1), dtype=torch.float8_e4m3fn)
    for name, value in ((prefix + "derived.mla_fp8_tp8.wk.weight", wk.cpu()),
                        (prefix + "derived.mla_fp8_tp8.wk.weight_scale", wk_scale.cpu().reshape(1))):
        if native_root:
            operand = native(name, value.dtype, value.shape)
            boundaries[name] = boundary_difference(operand.reshape(1, -1), value.reshape(1, -1))
    full_query = captured("attention.query")
    if full_query.shape != (m, 16, 576) or not torch.equal(full_query[:, ::2], full_query[:, 1::2]):
        raise ValueError("unexpected duplicated model query geometry")
    model_query = full_query[:, ::2, :512].contiguous()
    actual_query = native("act.qa", torch.bfloat16, (m, 8, 512)) if native_root else None
    transforms = []
    for label, qb in operands:
        x = qb.cuda().reshape(m, 8, 256)[..., :192]
        results = [rocm_aiter_ops.triton_fp8_bmm(x.transpose(0, 1), wk, wk_scale,
            group_size=128, transpose_bm=True).cpu() for _ in range(2)]
        transforms.append(dict(input=label, repeat_bitwise=tensor_digest(results[0]) == tensor_digest(results[1]),
            vs_native=boundary_difference(results[0], actual_query) if actual_query is not None else None,
            vs_model=boundary_difference(results[0], model_query)))
    downstream = {}
    if native_root:
        from safetensors import safe_open
        wv, wv_scale = dynamic_per_batched_tensor_quant(uv.permute(1, 2, 0), dtype=torch.float8_e4m3fn)
        for name, value in ((prefix + "derived.mla_fp8_tp8.wv.weight", wv.cpu()),
                            (prefix + "derived.mla_fp8_tp8.wv.weight_scale", wv_scale.cpu().reshape(1))):
            operand = native(name, value.dtype, value.shape)
            downstream[name] = boundary_difference(operand.reshape(1, -1), value.reshape(1, -1))
        olat = native("act.olat", torch.bfloat16, (m, 8, 512))
        oat = native("act.oat", torch.bfloat16, (m, 2048))
        value_results = [rocm_aiter_ops.triton_fp8_bmm(olat.cuda().transpose(0, 1), wv, wv_scale,
            group_size=128, transpose_bm=True).cpu().reshape(m, 2048) for _ in range(2)]
        downstream["value_transform"] = [boundary_difference(oat, value) for value in value_results]
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        projection = []
        for suffix, width in (("weight", 2048), ("weight_scale_inv", 16)):
            name = prefix + "o_proj." + suffix
            with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as shard:
                value = shard.get_slice(name)[:, :width].contiguous()
            loaded_name = name + "_fp8" if suffix == "weight" else name
            downstream[name] = boundary_difference(native(loaded_name, value.dtype, value.shape), value)
            projection.append(value.cuda())
        oq, os = rocm_aiter_ops.group_fp8_quant(oat.cuda(), 128)
        partial = native("act.og_tp", torch.bfloat16, (m, 6144))
        projection_results = [rocm_aiter_ops.gemm_a8w8_blockscale(oq, projection[0], os, projection[1],
            [128, 128], output_dtype=torch.bfloat16).cpu() for _ in range(2)]
        downstream["output_projection"] = [boundary_difference(partial, value) for value in projection_results]
        downstream["output_projection_repeat_bitwise"] = tensor_digest(projection_results[0]) == tensor_digest(projection_results[1])
        config = get_CKGEMM_config(m, 6144, 2048, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
        count = 1 << int(config["splitK"]) if config is not None else 1
        if count > 8 or 2048 % (128 * count):
            raise ValueError("output projection partition is outside the audited contract")
        output_parts, output_records = [], []
        for part in range(count):
            lo, hi = part * (2048 // count), (part + 1) * (2048 // count)
            a, w = oq[:, lo:hi].contiguous(), projection[0][:, lo:hi].contiguous()
            asc, wsc = os[:, lo // 128:hi // 128].contiguous(), projection[1][:, lo // 128:hi // 128].contiguous()
            def isolated_output():
                y = torch.empty((m, 6144), dtype=torch.bfloat16, device="cuda")
                return gemm_a8w8_blockscale_ck(a, w, asc, wsc, y, splitK=0,
                    kernelName="" if config is None else str(config["kernelName"])).cpu()
            value, repeat = isolated_output(), isolated_output()
            path = args.output.parent / f"output.part{part}.bin"
            write_case(path, a.cpu().view(torch.uint8), w.cpu(), asc.cpu().T.contiguous(), wsc.cpu(), value)
            output_parts.append(value)
            output_records.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                repeat_bitwise=tensor_digest(value) == tensor_digest(repeat)))
        downstream["output_accumulation"] = dict(split_count=count, parts=output_records,
            selected_config=None if config is None else {key: str(value) for key, value in config.items()},
            reachability={label: bf16_order_reachability(torch.stack(output_parts, dim=1), value)
                for label, value in (("native", partial), ("installed0", projection_results[0]),
                                     ("installed1", projection_results[1]))})
        from vllm.kernels.aiter_ops import fused_add_rms_norm
        norm_name = "model.layers.6.post_attention_layernorm.weight"
        with safe_open(args.checkpoint / index[norm_name], framework="pt", device="cpu") as shard:
            gamma = shard.get_tensor(norm_name)
        epsilon = json.loads((args.checkpoint / "config.json").read_text())["rms_norm_eps"]
        summed = torch.zeros((m, 6144), dtype=torch.float32)
        for rank in range(8):
            summed += native("act.og_tp", torch.bfloat16, (m, 6144), rank).float()
        residual, _ = load_tensor(native_root / "inputs/act.x.bin", torch.bfloat16, (m, 6144))
        norm_results = [fused_add_rms_norm.impl_fn(summed.bfloat16().cuda(), residual.cuda(), gamma.cuda(), epsilon)
                        for _ in range(2)]
        norm, updated = (value.cpu() for value in norm_results[0])
        downstream["post_attention"] = dict(
            scope="installed fused norm on ordered native TP partials and captured residual; not reference TP collective qualification",
            reference_weight_sha256=tensor_digest(gamma), native_norm_weight_captured=False,
            repeat_bitwise=all(tensor_digest(a) == tensor_digest(b) for a, b in zip(norm_results[0], norm_results[1])),
            ranks=[dict(rank=rank,
                norm=boundary_difference(native("act.xn2", torch.bfloat16, (m, 6144), rank), norm),
                residual=boundary_difference(native("act.xmid", torch.bfloat16, (m, 6144), rank), updated))
                for rank in range(8)])
    report = dict(scope="actual rank0/layer6 query projection and absorbed transform localization, not precision qualification",
        precision_qualified=False, batch=m, run_record_sha256=hashlib.sha256(run_bytes).hexdigest() if run_bytes else None,
        capture_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(), native_prefixes=identities,
        boundaries=boundaries, transforms=transforms, accumulation=accumulation, downstream=downstream,
        qb_repeat_bitwise=tensor_digest(projected[0]) == tensor_digest(projected[1]))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


def check_model_decode_boundaries(args):
    if args.reference_json is None:
        raise ValueError("requires connected run-record.json")
    root = args.reference_json.parent
    run_bytes = args.reference_json.read_bytes()
    run = json.loads(run_bytes)
    meta = json.loads((root / "inputs/reference.json").read_text())
    manifest_bytes = (args.capture / "reference/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    m, ctx = meta["batch"], meta["ctx"]
    if (m not in (1, 8) or ctx not in (8193, 8194, 8195, 8196)
            or len(manifest["requests"]) != m or manifest["invalid_cases"]
            or manifest["vllm_version"] != "0.29.0" or manifest["tensor_parallel_size"] != 8
            or meta["capture_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("connected decode does not match model capture")
    if any(run["inputs"].get(name) != digest for name, digest in meta["files"].items()):
        raise ValueError("packed model operands differ from connected run inputs")
    for name, digest in run["inputs"].items():
        with (root / "inputs" / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("connected input changed")
    records = {}
    for path in (args.capture / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        if record["context"]["max_seq_len"] != ctx:
            continue
        if record["semantic"] in records:
            raise ValueError("ambiguous captured boundary")
        records[record["semantic"]] = record

    def model(name):
        record = records[name]
        if (record["rank"] != 0 or record["layer"] != 6
                or record["prompt_sha256_u32le"] != manifest["requests"][0]["prompt_sha256_u32le"]
                or record["context_sha256"] != hashlib.sha256(json.dumps(
                    record["context"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
            raise ValueError("wrong model boundary identity")
        return load_indexer_model_tensor(args.capture, record,
            getattr(torch, record["source_dtype"]), record["source_shape"])

    def native(name, dtype, shape):
        return load_tensor(root / "outputs" / f"rank0.{name}.bin", dtype, shape,
                           live_prefix=name.startswith("act."))[0]

    boundaries = {}
    for source, target in (("block.xn", "act.xn"), ("indexer.input.qr", "act.qlat"),
                           ("block.qb", "act.qb"), ("attention.output", "act.olat")):
        expected = model(source)
        boundaries[target] = boundary_difference(native(target, expected.dtype, expected.shape), expected)
    query = model("attention.query")
    if query.shape != (m, 16, 576) or not torch.equal(query[:, ::2], query[:, 1::2]):
        raise ValueError("unexpected duplicated model query heads")
    for name, expected in (("act.qa", query[:, ::2, :512]), ("act.qr", query[:, ::2, 512:])):
        boundaries[name] = boundary_difference(native(name, expected.dtype, expected.shape), expected)
    for name, width in (("ckv", 512), ("krot", 64)):
        expected, _ = load_tensor(root / "inputs" / f"kv.6.{name}.bin", torch.bfloat16, (m, ctx, width))
        actual = torch.stack([native(f"slot{slot}.kv.6.{name}", torch.bfloat16, (ctx, width))
                              for slot in range(m)])
        boundaries["kv." + name + ".new_row"] = boundary_difference(actual[:, -1], expected[:, -1])
        boundaries["kv." + name + ".history"] = dict(bitwise=torch.equal(actual[:, :-1], expected[:, :-1]),
            sha256=tensor_digest(actual[:, :-1]), reference_sha256=tensor_digest(expected[:, :-1]))
    selected = model("indexer.selected")[:m].contiguous()
    actual = native("act.iidx", torch.int32, (m, 2048))
    for value in (actual, selected):
        sparse_decode_indices(value, ctx)
    selection = dict(sha256=tensor_digest(actual), reference_sha256=tensor_digest(selected),
        order_bitwise=torch.equal(actual, selected),
        overlap_per_row=[len(set(a) & set(b)) for a, b in zip(actual.tolist(), selected.tolist())])
    report = dict(scope="rank0 actual-model decode boundary localization; not precision qualification",
        precision_qualified=False, batch=m, ctx=ctx, boundaries=boundaries, selection=selection,
        run_record_sha256=hashlib.sha256(run_bytes).hexdigest(),
        capture_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)
    return 0


def pack_model_block(args):
    manifest_bytes = (args.capture / "reference/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    audit_path = args.reference_json or (args.capture / "block-cache-audit.json")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    batch = len(manifest["requests"])
    if (manifest["vllm_version"] != "0.29.0" or manifest["invalid_cases"]
            or manifest["tensor_parallel_size"] != 8 or batch not in (1, 8)
            or [c["length"] for c in audit["block_invocations"]] != [8193, 8194, 8195, 8196]):
        raise ValueError("requires complete audited original-model block capture")
    if batch > 1 and (audit.get("batch") != batch
            or audit.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
            or [c["length"] for c in audit.get("cache_invocations", [])] != [8193, 8194, 8195, 8196]
            or any(c["batch"] != batch for c in audit["cache_invocations"])):
        raise ValueError("requires manifest-bound batched cache audit")
    records = {}
    for path in (args.capture / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        length = record["context"]["max_seq_len"]
        key = length, record["semantic"]
        if length < 8193:
            continue
        if key in records:
            raise ValueError("ambiguous model block tensor")
        records[key] = record
    cases = []
    for length in (8193, 8194, 8195, 8196):
        identities = {}

        def load(name):
            record = records[length, name]
            if (record["rank"] != 0 or record["layer"] != 6
                    or record["prompt_sha256_u32le"] != manifest["requests"][0]["prompt_sha256_u32le"]):
                raise ValueError("wrong model block identity")
            identities[name] = record["sha256"]
            if batch > 1:
                category = "block_invocations" if name.startswith("block.") else "cache_invocations"
                bound = next(row for row in audit[category] if row["length"] == length)["tensor_sha256"]
                key = name.removeprefix("block.") if category == "block_invocations" else name
                if bound.get(key) != record["sha256"]:
                    raise ValueError("block packing tensor differs from batch audit")
            return load_indexer_model_tensor(args.capture, record,
                getattr(torch, record["source_dtype"]), record["source_shape"])

        directory = args.output.parent / f"length{length}"
        directory.mkdir(parents=True, exist_ok=False)
        outputs = {}

        def save(name, value):
            data = value.contiguous().view(torch.uint8).numpy().tobytes()
            with (directory / name).open("xb") as stream:
                stream.write(data)
            outputs[name] = hashlib.sha256(data).hexdigest()

        save("act.x.bin", load("block.x"))
        for source, target in (("xn", "act.xn"), ("xmid", "act.xmid"), ("xn2", "act.xn2")):
            save(target + ".reference.bf16", load("block." + source))
        hidden, residual = load("block.output.hidden"), load("block.output.residual")
        save("reference.bf16", (hidden.float() + residual.float()).bfloat16())
        save("input.hidden.reference.bf16", load("block.input.hidden"))
        save("input.residual.reference.bf16", load("block.input.residual"))
        selected = load("indexer.selected")[:batch]
        save("act.iidx.bin", selected)
        cache, table = load("attention.cache"), load("attention.block_table")
        logical = torch.arange(length, dtype=torch.int64)
        compact = []
        for row in range(batch):
            attention_physical_indices(selected[row:row+1], table[row:row+1], length, cache.shape[0] * 16)
            pages = table[row][logical // 16].long()
            if not bool(((pages >= 0) & (pages < cache.shape[0])).all()):
                raise ValueError("invalid full block main-cache pages")
            compact.append(cache[pages, logical % 16])
        compact = torch.stack(compact)
        save("kv.6.ckv.bin", compact[:, :, :512])
        save("kv.6.krot.bin", compact[:, :, 512:])
        del cache, compact
        cache, table = load("indexer.cache.before"), load("indexer.decode.block_table")
        compact = torch.stack([compact_indexer_model_cache(cache, table[row:row+1], length,
            ((length + 15) // 16) * 16) for row in range(batch)])
        save("kv.6.kidx_fp8.bin", compact)
        del cache, compact
        metadata = dict(batch=batch, ctx=length, tolerance_rel_l2=0.0,
            stages=["act.xn", "act.xmid", "act.xn2"],
            scope="original model block; input is captured rounded residual, raw fused-norm operands retained; strict diagnostic, not qualified",
            captured_tensor_sha256=identities, files=outputs,
            capture_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            capture_audit_sha256=hashlib.sha256(audit_bytes).hexdigest())
        with (directory / "reference.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
        cases.append(dict(length=length, directory=directory.name, files=outputs))
    with args.output.open("x") as stream:
        json.dump(dict(scope="CPU block operand packing, not native execution", cases=cases), stream, indent=2)
    return 0


def export_model_attention_fixture(path, values, indices, reference, length):
    from types import SimpleNamespace

    if values["query"].shape != (1, 16, 576) or reference.shape != (1, 8, 512):
        raise ValueError("invalid model attention fixture geometry")
    query = values["query"].cuda()
    if not torch.equal(query[:, ::2], query[:, 1::2]):
        raise ValueError("model attention query heads are not duplicated")
    logical = torch.arange(length, dtype=torch.int64)
    pages = values["block_table"][0][logical // 16].long()
    if not bool(((pages >= 0) & (pages < values["cache"].shape[0])).all()):
        raise ValueError("invalid model attention cache pages")
    kv = values["cache"][pages, logical % 16].unsqueeze(0).contiguous().cuda()
    cap, nwork = 257, int(values["work_indptr"][-1])
    if not 0 < nwork <= cap or values["reduce_indptr"][:2].tolist() != [0, nwork]:
        raise ValueError("invalid model attention live reduction")
    metadata = {name: values[name].cuda() for name in (
        "qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len", "work_indptr")}
    metadata.update(work_info_set=values["work_info_set"][:cap].contiguous().cuda(),
        reduce_indptr=values["reduce_indptr"][:2].contiguous().cuda(),
        reduce_final_map=values["reduce_final_map"][:1].contiguous().cuda(),
        reduce_partial_map=values["reduce_partial_map"][:cap].contiguous().cuda(),
        paged_kv_indices=indices.flatten().contiguous().cuda())
    metadata["work_meta_data"] = rebind_attention_work_metadata(
        metadata["work_indptr"], metadata["work_info_set"])
    md = SimpleNamespace(max_seq_len=length, topk_tokens=2048, **metadata)
    return export_attention_ps_case(path, query, kv, md,
        reference.repeat_interleave(2, dim=1).cuda(), 0.0625)


def replay_model_attention(args, version):
    import inspect
    from types import SimpleNamespace
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import ROCMAiterMLASparseImpl

    source = Path(inspect.getfile(ROCMAiterMLASparseImpl))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (version != "0.29.0" or args.tp != 8 or args.indexer_layer != 6
            or current_platform.num_compute_units() != 256
            or source_hash != "187d72a2ecbf0c845950454dbc3cda2c7eff022568a035ebb57ab5b8b98b132c"):
        raise ValueError("requires pinned vLLM 0.29 TP8 gfx950 attention backend")
    manifest_bytes = (args.capture / "reference/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    batch = len(manifest["requests"])
    audit_path = args.capture / ("cache-audit.json" if batch == 1 else "batch-cache-audit-v2.json")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if (manifest["vllm_version"] != version or manifest["invalid_cases"]
            or manifest["tensor_parallel_size"] != 8 or batch not in (1, 8)):
        raise ValueError("invalid original-model attention capture")
    request = manifest["requests"][0]
    if batch > 1 and (audit["batch"] != batch
            or audit["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or args.export_attention_ps or args.native_indexer_selection is not None):
        raise ValueError("requires bound batch audit; batched fixture export/selector replay unsupported")
    cases = audit["attention_invocations"] if batch == 1 else audit["cache_invocations"]
    if (len(request["prompt_token_ids"]) != 8192
            or [c["length"] for c in cases] != [8193, 8194, 8195, 8196]
            or (batch == 1 and not all(c["mapping_matches_indexer"] for c in cases))):
        raise ValueError("incomplete original-model attention history")
    native_root = args.reference_json.parent if args.reference_json else None
    if native_root:
        run_bytes = args.reference_json.read_bytes()
        run = json.loads(run_bytes)
        meta = json.loads((native_root / "inputs/reference.json").read_text())
        if (meta["batch"] != batch or meta["capture_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
                or meta["capture_audit_sha256"] != hashlib.sha256(audit_bytes).hexdigest()):
            raise ValueError("native attention run differs from audited capture")
        for name, digest in run["inputs"].items():
            with (native_root / "inputs" / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("native attention input changed")
        cases = [case for case in cases if case["length"] == meta["ctx"]]
        if len(cases) != 1:
            raise ValueError("native attention context missing from capture")
    records = {}
    for path in (args.capture / "tensors").glob("attention.*.json"):
        record = json.loads(path.read_text())
        key = record["invocation_index"], record["semantic"]
        if key in records:
            raise ValueError("duplicate attention capture record")
        records[key] = record
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.native_indexer_selection is not None:
        from vllm.v1.worker.workspace import init_workspace_manager
        init_workspace_manager(torch.device("cuda"))
    for case in cases:
        values, identities = {}, {}
        context = None
        for (invocation, semantic), record in records.items():
            if (batch == 1 and invocation != case["invocation"]) or (batch > 1
                    and record["context"]["max_seq_len"] != case["length"]):
                continue
            ctx = record["context"]
            if (record["rank"] != 0 or record["layer"] != 6
                    or record["prompt_sha256_u32le"] != request["prompt_sha256_u32le"]
                    or ctx != dict(num_actual_tokens=batch, max_seq_len=case["length"], block_size=16,
                                   topk_tokens=2048, layer_name="model.layers.6.self_attn.attn", scale=0.0625)
                    or hashlib.sha256(json.dumps(ctx, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    != record["context_sha256"]):
                raise ValueError("attention capture identity mismatch")
            context = ctx
            name = semantic.removeprefix("attention.")
            if name in values or (batch > 1 and semantic in case["tensor_sha256"]
                    and case["tensor_sha256"][semantic] != record["sha256"]):
                raise ValueError("ambiguous attention tensor or changed audited tensor")
            dtype = {"bfloat16": torch.bfloat16, "int32": torch.int32, "uint64": torch.uint64}[record["source_dtype"]]
            values[name] = load_indexer_model_tensor(args.capture, record, dtype, record["source_shape"])
            identities[name] = record["sha256"]
        if (context is None or values["query"].shape != (batch, 16, 576)
                or values["output"].shape != (batch, 8, 512) or values["cache"].shape[1:] != (16, 576)
                or values["work_meta_data"].shape != (2,) or values["qo_indptr"].tolist() != list(range(batch + 1))
                or values["paged_kv_indptr"].tolist() != [i * 2048 for i in range(batch + 1)]
                or values["paged_kv_last_page_len"].tolist() != [1] * batch):
            raise ValueError("attention capture geometry mismatch")
        def physical_indices(indices):
            if indices.shape != (batch, 2048):
                raise ValueError("attention selection batch mismatch")
            return batched_attention_physical_indices(indices, values["block_table"],
                case["length"], values["cache"].shape[0] * 16)
        physical = physical_indices(values["selected"][:batch])
        if (not torch.equal(physical, values["paged_kv_indices"].to(torch.int64))
                or not bool(((physical >= 0) & (physical < values["cache"].shape[0] * 16)).all())):
            raise ValueError("attention page mapping mismatch")
        variants = []
        selection = None
        if args.native_indexer_selection is not None:
            variants, selection = model_attention_selection_orders(
                args, manifest, audit, case["length"], values["selected"][:1])
        query, cache = values["query"].cuda(), values["cache"].cuda()
        variants = [(name, indices, query) for name, indices in variants]
        native_output = None
        if native_root:
            def native(name, dtype, shape):
                return load_tensor(native_root / "outputs" / f"rank0.{name}.bin", dtype, shape, live_prefix=True)[0]
            nq = torch.cat((native("act.qa", torch.bfloat16, (batch, 8, 512)),
                            native("act.qr", torch.bfloat16, (batch, 8, 64))), dim=-1).repeat_interleave(2, dim=1).cuda()
            ni = native("act.iidx", torch.int32, (batch, 2048))
            if not torch.equal(ni.sort(-1).values, values["selected"][:batch].sort(-1).values):
                raise ValueError("counterfactual requires identical selected sets")
            native_output = native("act.olat", torch.bfloat16, (batch, 8, 512))
            variants += [("native-query-model-order", values["selected"][:batch], nq),
                         ("model-query-native-order", ni, query), ("native-query-native-order", ni, nq)]
        impl = SimpleNamespace(num_heads=8, kv_lora_rank=512, scale=context["scale"])
        layer = SimpleNamespace(_q_scale=torch.ones((), device="cuda"), _k_scale=torch.ones((), device="cuda"))
        repeats, outputs = [], []
        for repeat in range(2):
            metadata = {name: values[name].cuda() for name in (
                "qo_indptr", "paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len",
                "work_indptr", "work_info_set", "reduce_indptr", "reduce_final_map", "reduce_partial_map")}
            metadata["work_meta_data"] = rebind_attention_work_metadata(metadata["work_indptr"], metadata["work_info_set"])
            md = SimpleNamespace(attn_out_dtype=torch.bfloat16, **metadata)
            result = ROCMAiterMLASparseImpl._forward_mla(impl, layer, query, cache, md).cpu().contiguous()
            comparison = boundary_difference(result, values["output"])
            path = args.output.parent / f"length{case['length']}.repeat{repeat}.bf16.bin"
            with path.open("xb") as stream:
                stream.write(result.view(torch.uint8).numpy().tobytes())
            comparison.update(file=path.name, sha256=tensor_digest(result))
            repeats.append(comparison)
            outputs.append(result)
        stable = tensor_digest(outputs[0]) == tensor_digest(outputs[1])
        passed = stable and all(r["bitwise"] and r["finite"] for r in repeats)
        exports = []
        if args.export_attention_ps:
            exports.append(dict(order="model", **export_model_attention_fixture(
                args.output.parent / f"length{case['length']}.model.attention-ps.bin",
                values, values["selected"][:1], outputs[0], case["length"])))
        order_results = []
        for name, indices, variant_query in variants:
            physical = physical_indices(indices)
            variant_outputs = []
            for repeat in range(2):
                metadata = {key: values[key].cuda() for key in (
                    "qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len", "work_indptr",
                    "work_info_set", "reduce_indptr", "reduce_final_map", "reduce_partial_map")}
                metadata["paged_kv_indices"] = physical.cuda()
                metadata["work_meta_data"] = rebind_attention_work_metadata(
                    metadata["work_indptr"], metadata["work_info_set"])
                md = SimpleNamespace(attn_out_dtype=torch.bfloat16, **metadata)
                result = ROCMAiterMLASparseImpl._forward_mla(impl, layer, variant_query, cache, md).cpu().contiguous()
                comparison = boundary_difference(result, values["output"])
                if native_output is not None:
                    comparison["vs_native"] = boundary_difference(result, native_output)
                path = args.output.parent / f"length{case['length']}.{name}.repeat{repeat}.bf16.bin"
                with path.open("xb") as stream:
                    stream.write(result.view(torch.uint8).numpy().tobytes())
                comparison.update(file=path.name)
                variant_outputs.append(comparison)
                if args.export_attention_ps and repeat == 0:
                    exports.append(dict(order=name, **export_model_attention_fixture(
                        args.output.parent / f"length{case['length']}.{name}.attention-ps.bin",
                        values, indices, result, case["length"])))
            repeat_bitwise = variant_outputs[0]["sha256"] == variant_outputs[1]["sha256"]
            passed = passed and repeat_bitwise and all(r["finite"] for r in variant_outputs)
            order_results.append(dict(name=name, repeats=variant_outputs, repeat_bitwise=repeat_bitwise,
                query_sha256=tensor_digest(variant_query), indices_sha256=tensor_digest(indices)))
        results.append(dict(**case, captured_tensor_sha256=identities, repeats=repeats,
                            repeat_bitwise=stable, selection=selection, order_results=order_results,
                            native_fixtures=exports, passed=passed))
        print(json.dumps(results[-1], allow_nan=False), flush=True)
        del query, cache, values, metadata, md
    report = dict(scope="actual model rank0 layer6 decode attention; bitwise control and optional query/selection-order diagnostics, fixed work partition; not serving parity",
        precision_qualified=False,
        native_selector_sha256=(hashlib.sha256(args.native_indexer_selection.read_bytes()).hexdigest()
                                if args.native_indexer_selection is not None else None),
        vllm_version=version, backend_sha256=source_hash,
        run_record_sha256=hashlib.sha256(run_bytes).hexdigest() if native_root else None,
        capture_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        pointer_rebinding="work_meta_data=[work_indptr.data_ptr(),work_info_set.data_ptr()]; all scheduling arrays captured unchanged",
        cases=results, passed=all(c["passed"] for c in results))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if report["passed"] else 1


def export_indexer_model(args, harness, rope, snapshots, sources, version):
    from vllm.model_executor.models.deepseek_v2 import Indexer

    manifest = json.loads((args.capture / "reference/manifest.json").read_text())
    audit = json.loads((args.capture / "cache-audit.json").read_text())
    if (manifest["vllm_version"] != version or manifest["invalid_cases"]
            or len(manifest["requests"]) != 1 or not audit["invocations"]
            or not all(c["projection_inputs_bound"] for c in audit["invocations"])):
        raise ValueError("requires audited original model input capture")
    request = manifest["requests"][0]
    records = {}
    for path in (args.capture / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        key = (record["invocation_index"], record["semantic"])
        if key in records:
            raise ValueError("duplicate model capture record")
        records[key] = record
    cases = []
    for case in audit["invocations"]:
        invocation, rows = case["invocation"], case["live"]

        def load(name, dtype, shape):
            record = records[invocation, "indexer." + name]
            if (record["rank"] != 0 or record["layer"] != args.indexer_layer
                    or record["context"]["max_seq_len"] != case["length"]
                    or record["prompt_sha256_u32le"] != request["prompt_sha256_u32le"]):
                raise ValueError("model capture identity mismatch")
            return load_indexer_model_tensor(args.capture, record, dtype, shape).cuda()

        hidden = load("input.hidden", torch.bfloat16, (rows, 6144))
        latent = load("input.qr", torch.bfloat16, (rows, 2048))
        positions = load("input.positions", torch.int64, (rows,))
        expected = [load(name, dtype, shape) for name, dtype, shape in (
            ("q_fp8", torch.float8_e4m3fn, (rows, 32, 128)),
            ("key_bf16", torch.bfloat16, (rows, 128)),
            ("weights", torch.float32, (rows, 32)),
        )]
        runs = []
        for _ in range(2):
            actual = Indexer.forward(harness, hidden, latent, positions, rope)
            runs.append([v.clone() for v in actual])
        comparisons = {}
        for name, wanted, first, second in zip(("query", "key", "weights"), expected, *runs):
            comparisons[name] = dict(bitwise=tensor_digest(wanted) == tensor_digest(first),
                                     repeat_bitwise=tensor_digest(first) == tensor_digest(second),
                                     max_abs=float((wanted.float() - first.float()).abs().max()))
        passed = all(c["bitwise"] and c["repeat_bitwise"] for c in comparisons.values())
        artifacts = {}
        if passed:
            for name, value in (("query_bf16", snapshots["q_live"].reshape(rows, 32, 128)),
                                ("weights_bf16", snapshots["kw_raw"][:, 128:]),
                                ("key_bf16", runs[0][1]), ("positions", positions)):
                value = value.detach().cpu().contiguous()
                filename = f"invocation{invocation}.{name}.bin"
                with (args.output.parent / filename).open("xb") as stream:
                    stream.write(value.view(torch.uint8).numpy().tobytes())
                artifacts[name] = dict(file=filename, dtype=str(value.dtype), shape=list(value.shape),
                                       sha256=tensor_digest(value))
        cases.append(dict(invocation=invocation, rows=rows, length=case["length"],
                          passed=passed, comparisons=comparisons, artifacts=artifacts))
        print(json.dumps(cases[-1]), flush=True)
    passed = all(c["passed"] for c in cases)
    with args.output.open("x") as stream:
        json.dump(dict(scope="original model inputs replayed through installed indexer projections/RoPE; not native Plow or serving qualification",
                       passed=passed, vllm_version=version, cases=cases, sources=sources,
                       capture=str(args.capture), audit_sha256=hashlib.sha256(
                           (args.capture / "cache-audit.json").read_bytes()).hexdigest(),
                       inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest()),
                  stream, indent=2)
    return 0 if passed else 1


def export_indexer(args, rocm_aiter_ops, version):
    import inspect
    from types import SimpleNamespace
    from vllm.config import VllmConfig, CompilationConfig, set_current_vllm_config
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.layernorm import LayerNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize, GroupShape
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.v1.worker.workspace import init_workspace_manager
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, AITER_CONFIGS

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires original checkpoint and pinned vLLM0.29 inventory")
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    prefix = indexer_contract(cfg, inventory, args.tp, args.indexer_layer)
    model_capture = getattr(args, "export_indexer_model", False)
    if not model_capture:
        meta = json.loads((args.capture / "inputs/reference.json").read_text())
        rope_capture_contract(meta, cfg, inventory, args.tp)
    invocation = json.loads((args.precision_inventory.parent / "invocation.json").read_text())
    compilation = invocation["effective_compilation_config"]
    if compilation["mode"] != "NONE" or "all" not in compilation["custom_ops"]:
        raise ValueError("requires captured custom-op-enabled eager reference")
    sources = {}
    for cls in (Indexer, UnquantizedLinearMethod, LayerNorm):
        name = cls.__module__ + "." + cls.__name__
        digest = hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
        if any(r["sources"][name]["sha256"] != digest for r in inventory["ranks"]):
            raise ValueError("installed indexer source differs from loaded inventory")
        sources[name] = digest
    sources[ops.__name__] = hashlib.sha256(Path(inspect.getfile(ops)).read_bytes()).hexdigest()
    weights = indexer_weights(args.checkpoint, args.indexer_layer)
    if not model_capture:
        source_batch = meta["batch"]
        hidden, hidden_hash = load_tensor(args.capture / "outputs/rank0.act.xn.bin", torch.bfloat16, (source_batch, 6144))
        latent, latent_hash = load_tensor(args.capture / "outputs/rank0.act.qlat.bin", torch.bfloat16, (source_batch, 2048))
        if not bool(torch.isfinite(hidden).all() and torch.isfinite(latent).all()):
            raise ValueError("nonfinite conditioned inputs")
    config = VllmConfig(compilation_config=CompilationConfig(mode=0, custom_ops=compilation["custom_ops"]))
    init_workspace_manager(torch.device("cuda"))
    with set_current_vllm_config(config), torch.device("cuda"):
        rope = get_rope(64, max_position=cfg["max_position_embeddings"], is_neox_style=False,
                        rope_parameters=cfg["rope_parameters"], dtype=torch.bfloat16)
        norm = LayerNorm(128, eps=1e-6)
        linear = UnquantizedLinearMethod()
    if rope._forward_method.__name__ != "forward_hip" or not ops._ON_GFX950:
        raise ValueError("requires gfx950 and installed HIP rotary custom op")
    norm.weight.data.copy_(weights["k_norm.weight"].float())
    norm.bias.data.copy_(weights["k_norm.bias"].float())
    qw = torch.empty((4096, 2304), dtype=torch.float8_e4m3fn, device="cuda")[:, :2048]
    qw.copy_(weights["wq_b.weight"])
    qs = weights["wq_b.weight_scale_inv"].cuda()
    kw = scaled_dequantize(weights["wk.weight"].cuda(), weights["wk.weight_scale_inv"].cuda(),
                           group_shape=GroupShape(128, 128), out_dtype=torch.bfloat16)
    merged = SimpleNamespace(weight=torch.cat((kw, weights["weights_proj.weight"].cuda())))
    snapshots = {}
    def q_project(x):
        xq, xs = rocm_aiter_ops.group_fp8_quant(x, 128)
        y = rocm_aiter_ops.gemm_a8w8_blockscale(xq, qw, xs, qs, [128, 128], output_dtype=torch.bfloat16)
        snapshots.update(xq=xq.clone(), xs=xs.clone(), q_raw=y.clone())
        if model_capture:
            snapshots["q_live"] = y
        return y, None
    def kw_project(x):
        y = linear.apply(merged, x)
        snapshots["kw_raw"] = y.clone()
        return y, None
    harness = SimpleNamespace(wq_b=q_project, wk_weights_proj=kw_project, k_norm=norm,
        n_head=32, head_dim=128, rope_dim=64, is_inplace_rope=True, use_fused_indexer_q=False,
        quant_block_size=128, scale_fmt="ue8m0", softmax_scale=128 ** -0.5,
        n_head_scale=32 ** -0.5, indexer_op=lambda h, q, k, w: (q, k, w))
    if model_capture:
        with torch.no_grad(), set_current_vllm_config(config):
            return export_indexer_model(args, harness, rope, snapshots, sources, version)
    if args.block_indexer:
        with torch.no_grad(), set_current_vllm_config(config):
            return compare_indexer_boundaries(args, meta, weights, harness, rope, sources, version)
    if args.export_indexer_quant:
        with torch.no_grad(), set_current_vllm_config(config):
            return export_indexer_quant(args, meta, harness, rope, sources, version)
    def rows(x, m):
        return x.repeat(((m + source_batch - 1) // source_batch, 1))[:m].contiguous().cuda()
    def save(name, value):
        value = value.detach().cpu().contiguous()
        path = args.output.parent / name
        with path.open("xb") as f:
            f.write(value.view(torch.uint8).numpy().tobytes())
        return dict(file=name, shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_digest(value))
    cases = []
    with torch.no_grad(), set_current_vllm_config(config):
        for m in (1, 8, 16, 32, 64):
            x, qr = rows(hidden, m), rows(latent, m)
            q_project(qr)
            first = {k: v.clone() for k, v in snapshots.items()}
            q_project(qr)
            stable = all(tensor_digest(first[k]) == tensor_digest(snapshots[k]) for k in first)
            finite = all(bool(torch.isfinite(v.float()).all()) for v in snapshots.values())
            path = args.output.parent / f"m{m}.indexer-wq.bin"
            write_case(path, snapshots["xq"].view(torch.uint8), weights["wq_b.weight"],
                       snapshots["xs"].T.contiguous(), weights["wq_b.weight_scale_inv"], snapshots["q_raw"])
            selected = get_CKGEMM_config(m, 4096, 2048, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
            cases.append(dict(shape=[m, 4096, 2048], file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), repeat_bitwise=stable, finite=finite,
                selected_config=None if selected is None else {k: str(v) for k, v in selected.items()}))
        pipelines = []
        m = 16
        x, qr = rows(hidden, m), rows(latent, m)
        for ctx in (8192, 71680):
            positions = torch.full((m,), ctx - 1, dtype=torch.int64, device="cuda")
            q, k, w = Indexer.forward(harness, x, qr, positions, rope)
            first = [v.clone() for v in (q, k, w)]
            again = Indexer.forward(harness, x, qr, positions, rope)
            forward_repeat = {name: tensor_digest(a) == tensor_digest(b)
                              for name, a, b in zip(("query", "key", "weights"), first, again)}
            stable = all(forward_repeat.values())
            base = norm(snapshots["kw_raw"][:, :128]).repeat((ctx // m, 1))
            p = torch.arange(ctx, device="cuda", dtype=torch.int64)
            dummy = torch.zeros((ctx, 1, 64), device="cuda", dtype=torch.bfloat16)
            rope(p, dummy, base[:, :64].unsqueeze(1))
            history = base.repeat((m, 1)).contiguous()
            history.view(m, ctx, 128)[:, -1].copy_(k)
            blocks = torch.arange(m * ctx // 16 - 1, -1, -1, device="cuda", dtype=torch.int32)
            table = blocks.reshape(m, ctx // 16).contiguous()
            slots = (blocks.long()[:, None] * 16 + torch.arange(16, device="cuda")).flatten()
            cache = torch.empty((m * ctx // 16, 16, 132), device="cuda", dtype=torch.uint8)
            lengths = torch.full((m,), ctx, device="cuda", dtype=torch.int32)
            schedule = torch.empty(0, device="cuda", dtype=torch.int32)
            outputs = []
            for poison in (0x55, 0xaa):
                cache.fill_(poison)
                ops.indexer_k_quant_and_cache_triton(history, cache, slots, 128, "ue8m0")
                logits = ops.rocm_fp8_paged_mqa_logits(q[:, None], cache.unsqueeze(2), w, lengths,
                                                     table, schedule, max_model_len=ctx).clone()
                indices = torch.full((m, 2048), -1, device="cuda", dtype=torch.int32)
                if ops._get_aiter_top_k_kernel(is_prefill=False, compress_ratio=1,
                                              num_rows=m, max_valid_seq_len=ctx) is not None:
                    raise ValueError("unexpected non-compressed top-k dispatch")
                torch.ops._C.top_k_per_row_decode(logits, 1, lengths, indices, m,
                                                  logits.stride(0), logits.stride(1), 2048)
                outputs.append([v.clone() for v in (cache, logits, indices)])
            backend_repeat = {name: tensor_digest(a) == tensor_digest(b)
                              for name, a, b in zip(("cache", "logits", "indices"), *outputs)}
            selection_set_repeat = torch.equal(outputs[0][2].sort(dim=1).values, outputs[1][2].sort(dim=1).values)
            stable &= all(backend_repeat.values())
            finite = all(bool(torch.isfinite(v.float()).all()) for v in (q, k, w, history, logits))
            valid = bool(((indices >= 0) & (indices < ctx)).all()) and all(
                len(set(row)) == 2048 for row in indices.cpu().tolist())
            artifacts = {name: save(f"ctx{ctx}.{name}.bin", value) for name, value in
                (("query", q), ("key", k), ("weights", w), ("history", history), ("cache", cache),
                 ("block-table", table), ("logits", logits), ("indices", indices),
                 ("indices-first", outputs[0][2]))}
            pipelines.append(dict(shape=[m, ctx, 32, 128], finite=finite, repeat_bitwise=stable,
                                  forward_repeat=forward_repeat, backend_repeat=backend_repeat,
                                  selection_set_repeat=selection_set_repeat,
                                  indices_valid_unique=valid, artifacts=artifacts))
            print(json.dumps({k: v for k, v in pipelines[-1].items() if k != "artifacts"}), flush=True)
    complete = all(c["finite"] and c["repeat_bitwise"] for c in cases + pipelines)
    complete &= all(c["indices_valid_unique"] for c in pipelines)
    with args.output.open("x") as f:
        json.dump(dict(scope="installed indexer operators with layer-specific real weights, conditioned on another block's captured inputs; repeated normalized-key history, reversed page map; not prefill/full-model or Plow qualification",
            audit_complete=complete, precision_qualified=False, vllm_version=version, layer=args.indexer_layer,
            source_layer=meta["layer"], source_batch=source_batch, hidden_sha256=hidden_hash, latent_sha256=latent_hash,
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
            sources=sources, weights={k: dict(dtype=str(v.dtype), sha256=tensor_digest(v)) for k, v in weights.items()},
            cases=cases, pipelines=pipelines), f, indent=2, allow_nan=False)
    return 0 if complete else 1


def check_indexer_wq(args):
    if args.reference_json is None:
        raise ValueError("requires frozen indexer reference")
    audit = json.loads(args.reference_json.read_text())
    expected = {(m, 4096, 2048) for m in (1, 8, 16, 32, 64)}
    if (audit["vllm_version"] != "0.29.0" or len(audit["cases"]) != len(expected)
            or {tuple(c["shape"]) for c in audit["cases"]} != expected):
        raise ValueError("incomplete indexer query projection rung coverage")
    records = []
    for case in audit["cases"]:
        shape, operands, reference, digest = load_case(args.reference_json.parent / case["file"])
        if (list(shape) != case["shape"] or digest != case["sha256"]
                or not case["finite"] or not case["repeat_bitwise"]):
            raise ValueError("indexer query reference hash/shape/stability mismatch")
        if (tensor_digest(operands[1]) != audit["weights"]["wq_b.weight"]["sha256"]
                or tensor_digest(operands[3]) != audit["weights"]["wq_b.weight_scale_inv"]["sha256"]):
            raise ValueError("indexer query fixture weight/scale identity mismatch")
        actual, actual_hash = load_tensor(args.capture / f"m{shape[0]}.bf16", torch.bfloat16, reference.shape)
        row = boundary_difference(actual, reference)
        row.update(shape=list(shape), output_sha256=actual_hash)
        records.append(row)
        print(json.dumps(row), flush=True)
    passed = all(r["finite"] and r["bitwise"] for r in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="prequantized indexer query projection only; not packet, cache, selection or serving qualification",
            passed=passed, precision_qualified=False, cases=records,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest()), f, indent=2, allow_nan=False)
    return 0 if passed else 1


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
    layer = captured_block_layer(metadata)
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
    m, layer = meta["batch"], captured_block_layer(meta)
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
            value, value_hash = load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", want.dtype, want.shape,
                live_prefix=getattr(args, "live_prefix", False) and name.startswith("act."))
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


def mla_bmm_boundaries(args, rocm_aiter_ops, inventory, layer, rank, m, captured, exact, boundaries):
    from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
    from vllm.model_executor.layers.attention.mla_attention import dynamic_per_batched_tensor_quant

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    if tuple(cfg[key] for key in ("num_attention_heads", "kv_lora_rank", "qk_nope_head_dim",
            "qk_rope_head_dim", "v_head_dim", "q_lora_rank")) != (64, 512, 192, 64, 256, 2048):
        raise ValueError("unsupported MLA geometry")
    prefix = f"model.layers.{layer}.self_attn."
    q = captured("act.qb", torch.bfloat16, (m, 8, 256)).cuda()
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


def native_norm_diagnostic(native, x, gamma, eps):
    m, k = x.shape
    if (x.dtype != torch.bfloat16 or gamma.dtype != x.dtype or not x.is_cuda
            or gamma.device != x.device or not x.is_contiguous() or not gamma.is_contiguous()
            or gamma.shape != (k,) or k != 2048 or not 1 <= m <= 128):
        raise ValueError("native norm diagnostic requires contiguous GPU BF16 M1..128 K2048")
    out = torch.empty_like(x)
    raw = torch.empty((4, m, k), dtype=torch.float32, device=x.device)
    stats = torch.empty((m, 8), dtype=torch.float32, device=x.device)
    c = native.c
    values = [c.c_void_p(v.data_ptr()) for v in (out, raw, stats, x, gamma)]
    values += [c.c_uint(m), c.c_uint(k), c.c_float(eps), c.c_double(eps)]
    params = (c.c_void_p * len(values))(*(c.cast(c.byref(v), c.c_void_p) for v in values))
    native.call("hipModuleLaunchKernel", native.function, m, 1, 1, 512, 1, 1, 0,
        c.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)
    return out.cpu(), raw.cpu(), stats.cpu()


def qanorm_sweep_inputs(captured):
    if captured.dtype != torch.bfloat16 or captured.ndim != 2 or captured.shape[1] != 2048 or not captured.shape[0]:
        raise ValueError("requires nonempty BF16 Q-A rows")
    generator = torch.Generator().manual_seed(53029)
    for m in (1, 8, 16, 32, 64, 128):
        yield m, "captured-row-resize", captured.cpu().repeat(((m + captured.shape[0] - 1) // captured.shape[0], 1))[:m].contiguous()
        yield m, "zero", torch.zeros((m, 2048), dtype=torch.bfloat16)
        for scale in (0.0001, 0.1, 1.0, 100.0):
            yield m, f"seeded-gaussian-{scale}", (torch.randn((m, 2048), generator=generator) * scale).bfloat16()


def native_qanorm(native, x, gamma):
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or not x.shape[0] or x.shape[1] != 2048
            or not x.is_cuda or not x.is_contiguous() or gamma.dtype != x.dtype
            or gamma.device != x.device or gamma.shape != (2048,) or not gamma.is_contiguous()):
        raise ValueError("requires contiguous GPU BF16 Q-A operands")
    out = torch.empty_like(x)
    c = native.c
    values = [c.c_void_p(v.data_ptr()) for v in (out, x, gamma)] + [c.c_uint(x.shape[0])]
    params = (c.c_void_p * len(values))(*(c.cast(c.byref(v), c.c_void_p) for v in values))
    native.call("hipModuleLaunchKernel", native.function, min(x.shape[0], 8), 1, 1, 512, 1, 1, 0,
        c.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)
    return out.cpu()


def compare_packet_qanorm(args, rocm_aiter_ops, version):
    if version != "0.29.0" or args.checkpoint is None or args.native_rmsnorm is None:
        raise ValueError("requires pinned reference, original checkpoint and native norm object")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    m, layer = meta["batch"], captured_block_layer(meta)
    if not 1 <= m <= 128:
        raise ValueError("unsupported norm batch")
    _, _, gamma = qb_weights(args.checkpoint, layer, 0, args.tp)
    k = gamma.numel()
    if k != 2048:
        raise ValueError("requires GLM Q-A width2048")
    def captured(name, dtype, shape):
        return load_tensor(args.capture / "outputs" / f"rank0.{name}.bin", dtype, shape,
            live_prefix=args.live_prefix and name.startswith("act."))[0]
    x = captured("act.qlr", torch.bfloat16, (m, k)).cuda()
    actual = captured("act.qlat", torch.bfloat16, (m, k))
    weight = captured(f"model.layers.{layer}.self_attn.q_a_layernorm.weight", torch.bfloat16, (k,))
    if tensor_digest(weight) != tensor_digest(gamma):
        raise ValueError("native Q-A norm weight differs from checkpoint")
    gamma = gamma.cuda()
    eps = json.loads((args.checkpoint / "config.json").read_text())["rms_norm_eps"]
    if eps != 1e-5:
        raise ValueError("Q-A production profile requires original epsilon1e-5")
    reference = rocm_aiter_ops.rms_norm(x, gamma, eps)
    repeat = rocm_aiter_ops.rms_norm(x, gamma, eps)
    native = IndexerSelectionModule(args.native_rmsnorm, b"glm_rmsnorm_diagnostic_v2")
    production = IndexerSelectionModule(args.native_rmsnorm, b"glm_rmsnorm_qa")
    sweep = []
    try:
        production_out = native_qanorm(production, x, gamma)
        production_repeat = native_qanorm(production, x, gamma)
        first, first_raw, first_stats = native_norm_diagnostic(native, x, gamma, eps)
        second, raw_cpu, stats_cpu = native_norm_diagnostic(native, x, gamma, eps)
        raw, stats = raw_cpu.cuda(), stats_cpu.cuda()
        if args.qanorm_sweep:
            for rows, profile, source in qanorm_sweep_inputs(x):
                gpu = source.cuda()
                ref = rocm_aiter_ops.rms_norm(gpu, gamma, eps).cpu()
                ref2 = rocm_aiter_ops.rms_norm(gpu, gamma, eps).cpu()
                got = native_norm_diagnostic(native, gpu, gamma, eps)
                again = native_norm_diagnostic(native, gpu, gamma, eps)
                prod = native_qanorm(production, gpu, gamma)
                prod2 = native_qanorm(production, gpu, gamma)
                # AITER HIP takes double epsilon, but divides the FP32 sum first.
                hip_mean = ((got[2][:, 0] / k).double() + eps).float()
                hip_inv = torch.rsqrt(hip_mean.double()).float()
                hip_epsilon = (source.float() * hip_inv[:, None] * gamma.cpu().float()).bfloat16()
                finite = all(bool(torch.isfinite(v).all()) for v in (*got, *again, ref, ref2))
                sweep.append(dict(rows=rows, profile=profile, input_sha256=tensor_digest(source),
                    finite=finite, repeat_bitwise=all(tensor_digest(a) == tensor_digest(b) for a, b in zip(got, again)),
                    reference_repeat_bitwise=tensor_digest(ref) == tensor_digest(ref2),
                    native_changed=int((got[0] != ref).sum()),
                    refined_changed=int((got[1][1].bfloat16() != ref).sum()),
                    fp64_inverse_changed=int((got[1][2].bfloat16() != ref).sum()),
                    reference_tree_changed=int((got[1][3].bfloat16() != ref).sum()),
                    production_changed=int((prod != ref).sum()),
                    production_repeat_bitwise=tensor_digest(prod) == tensor_digest(prod2),
                    reference_tree_sum_changed=int((got[2][:, 0] != got[2][:, 5]).sum()),
                    double_epsilon_precise_inverse_changed=int((hip_epsilon != ref).sum()),
                    double_epsilon_mean_changed=int((hip_mean != got[2][:, 1]).sum()),
                    refined_vs_fp64_changed=int((got[1][1].bfloat16() != got[1][2].bfloat16()).sum())))
    finally:
        native.close()
        production.close()
    variants = {"native": first, "native_raw_rounded": raw[0].bfloat16(),
        "production": production_out,
        "refined_raw_rounded": raw[1].bfloat16(), "fp64_inverse_raw_rounded": raw[2].bfloat16(),
        "reference_tree_raw_rounded": raw[3].bfloat16(),
        "torch_fp32": (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * gamma.float()).bfloat16(),
        "native_inv": (x.float() * stats[:, 2:3] * gamma.float()).bfloat16()}
    torch_ss = x.float().square().sum(-1)
    torch_mean = torch_ss / k + eps
    torch_inv = torch.rsqrt(torch_mean)
    native_mean_inv = torch.rsqrt(stats[:, 1])
    records = {}
    for name, value in variants.items():
        value = value.cpu()
        records[name] = dict(sha256=tensor_digest(value), reference_changed=int((value != reference.cpu()).sum()),
            captured_changed=int((value != actual).sum()))
    indices = (actual != reference.cpu()).nonzero()
    samples = [dict(row=int(i), column=int(j), captured=float(actual[i, j]),
        reference=float(reference[i, j]), native=float(first[i, j]), raw=raw[:, i, j].cpu().tolist(),
        stats=stats[i].cpu().tolist(), torch_stats=[float(torch_ss[i]), float(torch_mean[i]), float(torch_inv[i])],
        torch_rsqrt_native_mean=float(native_mean_inv[i])) for i, j in indices[:32].tolist()]
    report = dict(scope="rank0 Q-A normalization diagnostic; no full-block or performance qualification",
        precision_qualified=False, vllm_version=version, shape=[m, k],
        input_sha256=tensor_digest(x), gamma_sha256=tensor_digest(gamma),
        native_object_sha256=hashlib.sha256(args.native_rmsnorm.read_bytes()).hexdigest(),
        reference_repeat_bitwise=tensor_digest(reference) == tensor_digest(repeat),
        native_repeat_bitwise=tensor_digest(first) == tensor_digest(second),
        production_repeat_bitwise=tensor_digest(production_out) == tensor_digest(production_repeat),
        diagnostic_repeat_bitwise=(tensor_digest(first_raw) == tensor_digest(raw)
                                   and tensor_digest(first_stats) == tensor_digest(stats)),
        variants=records, mismatch_samples=samples, synthetic_sweep=sweep)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)
    return 0 if (all(report[key] for key in ("reference_repeat_bitwise", "native_repeat_bitwise", "production_repeat_bitwise",
                                           "diagnostic_repeat_bitwise"))
        and records["production"]["reference_changed"] == 0
        and all(r["finite"] and r["repeat_bitwise"] and r["reference_repeat_bitwise"]
                and r["production_repeat_bitwise"] and r["production_changed"] == 0 for r in sweep)) else 1


def compare_packet_mla_bmm(args, rocm_aiter_ops, version):
    if version != "0.29.0" or args.tp != 8 or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires pinned TP8 MLA inventory and original checkpoint")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    measurement = json.loads((args.capture / "measurement.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    m, layer = meta["batch"], captured_block_layer(meta)
    if (m < 1 or measurement.get("batch") != m or measurement.get("tp") != args.tp
            or measurement.get("scope") != "single-block-decode"
            or [r["rank"] for r in inventory["ranks"]] != list(range(args.tp))):
        raise ValueError("requires matching block capture and ordered TP inventory")
    records = []
    for rank in range(args.tp):
        boundaries = {}
        def captured(name, dtype, shape):
            return load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape,
                live_prefix=args.live_prefix and name.startswith("act."))[0]
        def exact(name, want):
            got = captured(name, want.dtype, want.shape)
            boundaries[name] = dict(sha256=tensor_digest(got), reference_sha256=tensor_digest(want),
                finite=bool(torch.isfinite(got.float()).all() and torch.isfinite(want.float()).all()),
                bitwise=tensor_digest(got) == tensor_digest(want))
        mla_bmm_boundaries(args, rocm_aiter_ops, inventory, layer, rank, m, captured, exact, boundaries)
        passed = all(v["finite"] and v["bitwise"] and v.get("reference_repeat_bitwise", True)
                     and v.get("input_finite", True) for v in boundaries.values())
        records.append(dict(rank=rank, shape=[m, 8], passed=passed, boundaries=boundaries))
        print(json.dumps(records[-1]), flush=True)
    passed = len(records) == args.tp and all(r["passed"] for r in records)
    with args.output.open("x") as f:
        json.dump(dict(scope="query/value BMM conditioned on captured BF16 inputs; Q-A/Q-B, attention and full-block parity not qualified",
            passed=passed, precision_qualified=False, vllm_version=version,
            loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
            cases=records), f, indent=2, allow_nan=False)
    return 0 if passed else 1


def compare_packet_mla(args, rocm_aiter_ops, version):
    if (version != "0.29.0" or args.tp != 8 or args.checkpoint is None
            or args.precision_inventory is None or args.qb_reference is None):
        raise ValueError("requires pinned TP8 MLA inventory, original checkpoint and independent Q-B reference")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    audit = json.loads(args.qb_reference.read_text())
    m, layer = meta["batch"], captured_block_layer(meta)
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
            return load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape,
                live_prefix=args.live_prefix and name.startswith("act."))[0]
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
        exact("act.qrr", qb.reshape(m, 8, 256)[..., 192:])
        mla_bmm_boundaries(args, rocm_aiter_ops, inventory, layer, rank, m, captured, exact, boundaries)
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


def sparse_decode_indices(indices, ctx, capacity=None):
    capacity = ctx if capacity is None else capacity
    if (indices.dtype != torch.int32 or indices.ndim != 2 or not indices.shape[0]
            or indices.shape[1] != 2048 or ctx <= 0 or capacity < ctx):
        raise ValueError("requires nonempty int32 decode indices with topk2048")
    count = min(ctx, indices.shape[1])
    selected = indices[:, :count]
    if (bool((selected < 0).any() or (selected >= ctx).any())
            or bool((indices[:, count:] != -1).any())
            or any(len(set(row)) != count for row in selected.tolist())):
        raise ValueError("invalid selected-key prefix or padding")
    return (selected + torch.arange(indices.shape[0], dtype=torch.int32)[:, None] * capacity).flatten()


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


def load_selection_replays(root, case):
    values = []
    for prefix in ("", "native_"):
        value, _ = load_tensor(root / case[prefix + "indices_file"], torch.int32,
                               (2, case["rows"], 2048))
        if [tensor_digest(v) for v in value] != case[prefix + "indices_sha256"]:
            raise ValueError("selection replay hash mismatch")
        for indices in value:
            sparse_decode_indices(indices, case["stride"])
        values.extend(value.unbind(0))
    return values


def pack_attention_split(args):
    import shutil
    raw = args.reference_json.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "cf2b6b35ced1cc69c443544ae35421e5eef3cc718b011aa71cc1c0a28ea164a0":
        raise ValueError("split attention reference identity mismatch")
    audit = json.loads(raw)
    records = []
    for case in audit["cases"]:
        path = args.reference_json.parent / case["file"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != case["sha256"] or not case["repeat_bitwise"]:
            raise ValueError("split case identity mismatch")
        spec = json.loads(data)
        values = spec["stage1"]
        m, ctx, splits = case["rows"], case["ctx"], values[14]["shape"][1]
        if m not in (32, 64) or splits != 256 // m or ctx not in (512, 8192, 71680):
            raise ValueError("unsupported split replay geometry")
        dest = args.output.parent / path.with_suffix(".bin").name
        indices = (0, 1, 2, 3, 4, 5, 6, 16, 23, 14, 15)
        with dest.open("xb") as out:
            out.write(struct.pack("<5I", 0x41535031, m, ctx, splits, values[4]["shape"][0]))
            for index in indices:
                item = values[index]
                source = args.reference_json.parent / item["file"]
                with source.open("rb") as inp:
                    size = source.stat().st_size
                    if digest_segment(inp, size) != item["sha256"]:
                        raise ValueError("split tensor identity mismatch")
                    inp.seek(0)
                    shutil.copyfileobj(inp, out)
        with dest.open("rb") as inp:
            digest = digest_segment(inp, dest.stat().st_size)
        records.append(dict(file=dest.name, sha256=digest, rows=m, ctx=ctx, splits=splits))
    with args.output.open("x") as out:
        json.dump(dict(scope="native HSA split replay inputs", cases=records,
            reference_sha256=hashlib.sha256(raw).hexdigest()), out, indent=2)
    return 0


def check_attention_addressing(args, version):
    import aiter
    import importlib
    module = importlib.import_module("aiter.mla")
    raw = args.reference_json.read_bytes()
    if version != "0.29.0" or hashlib.sha256(raw).hexdigest() != "cf2b6b35ced1cc69c443544ae35421e5eef3cc718b011aa71cc1c0a28ea164a0":
        raise ValueError("requires pinned split reference and vLLM 0.29")
    records = []
    for case in json.loads(raw)["cases"]:
        if case["rows"] != 64 or case["ctx"] != 71680:
            continue
        path = args.reference_json.parent / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("case identity mismatch")
        spec = json.loads(path.read_bytes())
        cpu, gpu = {}, {}
        def load(value, device):
            if not isinstance(value, dict):
                return value
            name = value["file"]
            if name not in cpu:
                source = path.parent / name
                size = source.stat().st_size
                with source.open("rb") as inp:
                    if digest_segment(inp, size) != value["sha256"]:
                        raise ValueError("tensor identity mismatch")
                tensor = torch.from_file(str(source), shared=False, size=size, dtype=torch.uint8)
                tensor = tensor.view(getattr(torch, value["dtype"].removeprefix("torch."))).reshape(value["shape"])
                if list(tensor.stride()) != value["stride"]:
                    raise ValueError("requires contiguous captured tensor")
                cpu[name] = tensor
            if device and name not in gpu:
                gpu[name] = cpu[name].cuda()
            return gpu[name] if device else cpu[name]
        values = spec["stage1"]
        kv = load(values[1], False).reshape(-1, 576)
        ptr = load(values[3], False)
        indices = load(values[4], False)[:int(ptr[-1])].long()
        compact = kv.index_select(0, indices).reshape(-1, 1, 1, 576).cuda()
        stage = [None if i == 1 else load(v, True) for i, v in enumerate(values)]
        stage[1] = compact
        stage[4] = torch.arange(indices.numel(), dtype=torch.int32, device="cuda")
        reduce = [load(v, True) for v in spec["reducer"]]
        expected = load(values[16], False).clone()
        def run():
            stage[14].fill_(float("nan"))
            stage[15].fill_(float("nan"))
            stage[16].fill_(float("nan"))
            aiter.mla_decode_stage1_asm_fwd(*stage)
            module._fwd_kernel_stage2_asm[(64, 16)](*reduce,
                page_size=1, KV_INDPTR_IS_PAGE_LEVEL=False, MAYBE_FINAL_OUT=False,
                HAS_FINAL_LSE=False, USE_VALID_SPLIT_COUNT_REDUCE=1, BATCH_NUM=64,
                BLOCK_DV=512, Lv=512, mgc=64, num_warps=1)
            return stage[16].clone()
        actual = run()
        stable = torch.equal(actual, run())
        mathematical = torch.empty_like(actual, dtype=torch.float64)
        for b in range(64):
            k = compact[int(ptr[b]):int(ptr[b + 1])].reshape(-1, 576).double()
            q = stage[0][b].double()
            mathematical[b] = torch.softmax((q @ k.T) * 0.0625, dim=-1) @ k[:, :512]
        actual_cpu, math_cpu = actual.cpu(), mathematical.cpu()
        row_errors = []
        for b in range(64):
            selected = indices[int(ptr[b]):int(ptr[b + 1])]
            wrapped = bool(((selected * 1152 + 1151) >= (1 << 32)).any())
            denominator = float(torch.linalg.vector_norm(math_cpu[b]))
            row_errors.append(dict(row=b, wrapped=wrapped,
                reference_rel_l2=float(torch.linalg.vector_norm(expected[b].double() - math_cpu[b])) / denominator,
                compact_rel_l2=float(torch.linalg.vector_norm(actual_cpu[b].double() - math_cpu[b])) / denominator,
                compact_vs_reference_bitwise=torch.equal(actual_cpu[b], expected[b])))
        record = dict(case=case["file"], repeat_bitwise=stable,
            finite=bool(torch.isfinite(actual).all()), rows=row_errors,
            unchanged_below_boundary=all(r["compact_vs_reference_bitwise"] for r in row_errors if not r["wrapped"]),
            compact_closer_above_boundary=all(r["compact_rel_l2"] < r["reference_rel_l2"] for r in row_errors if r["wrapped"]))
        for name, tensor in (("output", actual_cpu), ("partial", stage[14].cpu()), ("lse", stage[15].cpu())):
            dest = args.output.parent / f"{path.stem}.compact.{name}.bin"
            data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            with dest.open("xb") as out:
                out.write(data)
            record[name] = dict(file=dest.name, sha256=hashlib.sha256(data).hexdigest())
        records.append(record)
        print(json.dumps(record), flush=True)
        del stage, reduce, compact, actual, mathematical, gpu, cpu
    passed = len(records) == 2 and all(r["repeat_bitwise"] and r["finite"]
        and r["unchanged_below_boundary"] and r["compact_closer_above_boundary"] for r in records)
    with args.output.open("x") as out:
        json.dump(dict(scope="captured synthetic split attention address diagnosis, not serving qualification",
            passed=passed, cases=records), out, indent=2, allow_nan=False)
    return 0 if passed else 1


def capture_attention_split(call, path):
    import aiter
    import importlib
    module = importlib.import_module("aiter.mla")
    original_stage1 = aiter.mla_decode_stage1_asm_fwd
    original_reduce = module._fwd_kernel_stage2_asm
    stages = []
    reducers = []

    def stage1(*values, **kwargs):
        if kwargs or len(values) != 25 or any(values[i] is not None for i in (7, 8, 9)):
            raise ValueError("unexpected nonpersistent stage1 ABI")
        result = original_stage1(*values)
        stages.append(values)
        return result

    class CaptureReduce:
        def __getitem__(self, grid):
            def launch(*values, **kwargs):
                kernel = original_reduce[grid](*values, **kwargs)
                raw = kernel.asm["hsaco"]
                digest = hashlib.sha256(raw).hexdigest()
                dest = path.parent / f"attention-reduce-{digest}.elf"
                if not dest.exists():
                    with dest.open("xb") as f:
                        f.write(raw)
                reducers.append((values, dict(file=dest.name, sha256=digest,
                    symbol=kernel.metadata.name, shared=kernel.metadata.shared,
                    num_warps=kernel.metadata.num_warps, grid=list(grid),
                    signature={str(k): str(v) for k, v in kernel.src.signature.items()},
                    constants={str(k): str(v) for k, v in kernel.src.constants.items()})))
                return kernel
            return launch

    aiter.mla_decode_stage1_asm_fwd = stage1
    module._fwd_kernel_stage2_asm = CaptureReduce()
    try:
        result = call().clone()
    finally:
        aiter.mla_decode_stage1_asm_fwd = original_stage1
        module._fwd_kernel_stage2_asm = original_reduce
    if len(stages) != 1 or len(reducers) != 1:
        raise ValueError("expected exactly one split stage1 and reducer")
    tensors = {}
    def save(values, phase):
        records = []
        for i, value in enumerate(values):
            if not isinstance(value, torch.Tensor):
                records.append(value)
                continue
            key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
            if key not in tensors:
                raw = value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                dest = path.parent / f"{path.stem}.{phase}.arg{i}.bin"
                with dest.open("xb") as f:
                    f.write(raw)
                tensors[key] = dict(file=dest.name, sha256=hashlib.sha256(raw).hexdigest(),
                    shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
            records.append(tensors[key])
        return records
    reference = result.cpu()
    repeated = call().cpu()
    stable = tensor_digest(reference) == tensor_digest(repeated)
    if not stable or not bool(torch.isfinite(reference).all()):
        raise ValueError("split attention reference must be finite and repeat-bitwise")
    record = dict(stage1=save(stages[0], "stage1"), reducer=save(reducers[0][0], "reduce"),
        reducer_kernel=reducers[0][1], reference_sha256=tensor_digest(reference),
        repeat_bitwise=stable, finite=True,
        scope="installed nonpersistent stage1/reducer operands; unused scratch is not numerical evidence")
    with path.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference_sha256=record["reference_sha256"], repeat_bitwise=stable)


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
    if args.export_attention_split:
        shapes = [(m, ctx, None) for m in (32, 64) for ctx in (512, 8192, 71680)]
        shapes += [(m, 71680, [1, 16, 127, 128, 129, 512, 2048, 71680] * (m // 8)) for m in (32, 64)]
    selection_cases = None
    if args.selection_attention:
        if args.reference_json is None:
            raise ValueError("selection attention requires native selection reference")
        raw = args.reference_json.read_bytes()
        if hashlib.sha256(raw).hexdigest() != "3a5f235cadbf1140ec7243d262b27cce8a5cd7379b4b7bfa67aff538c40a6033":
            raise ValueError("native selection reference identity mismatch")
        selection = json.loads(raw)
        selection_cases = [c for c in selection["cases"] if c["mode"] == "decode" and c["profile"] != "ragged"]
        if not selection["passed"] or len(selection_cases) != 36 or not all(c["passed"] for c in selection_cases):
            raise ValueError("native selection coverage mismatch")
        shapes = [(c["rows"], c["stride"], None) for c in selection_cases]
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
        if args.export_attention_split:
            suffix = ".ragged" if ragged else ""
            record = capture_attention_split(lambda: impl.forward_mqa((qa, qr), kv, md, layer)[0],
                args.output.parent / f"m{m}.ctx{ctx}{suffix}.split.json")
            record.update(rows=m, ctx=ctx, kv_lengths=lengths, seed=57300 + case)
            cases.append(record)
            print(json.dumps(record), flush=True)
            continue
        if selection_cases is not None:
            selected_case = selection_cases[case]
            variants = load_selection_replays(args.reference_json.parent, selected_case)
            outputs, stable = [], []
            for selected in variants:
                impl.topk_indices_buffer.copy_(selected)
                value, _ = impl.forward_mqa((qa, qr), kv, md, layer)
                value = value.clone()
                repeat, _ = impl.forward_mqa((qa, qr), kv, md, layer)
                stable.append(tensor_digest(value) == tensor_digest(repeat))
                outputs.append(value.cpu())
            record = dict(rows=m, stride=ctx, profile=selected_case["profile"], seed=57300 + case,
                query_sha256=[tensor_digest(v) for v in (qa, qr)], kv_sha256=tensor_digest(kv),
                indices_sha256=[tensor_digest(v) for v in variants],
                reference_repeat=boundary_difference(outputs[1], outputs[0]),
                native_repeat=boundary_difference(outputs[3], outputs[2]),
                native_vs_reference=[boundary_difference(outputs[i + 2], outputs[i]) for i in range(2)],
                attention_repeat_bitwise=stable,
                same_set_rows=[int((variants[i].sort(1).values == variants[i + 2].sort(1).values).all(1).sum())
                               for i in range(2)])
            record["finite"] = all(bool(torch.isfinite(value).all()) for value in outputs)
            for i, value in enumerate(outputs):
                dest = args.output.parent / f"case{case}.variant{i}.bf16"
                with dest.open("xb") as f:
                    f.write(value.contiguous().view(torch.uint8).numpy().tobytes())
            cases.append(record)
            print(json.dumps(record), flush=True)
            continue
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
    if args.export_attention_split:
        with args.output.open("x") as f:
            json.dump(dict(scope="installed split-KV decode attention reference; synthetic BF16 Q/KV and selected indices",
                precision_qualified=False, audit_complete=True, vllm_version=version,
                backend_sha256=source_hash, cases=cases), f, indent=2, allow_nan=False)
        return 0
    if selection_cases is not None:
        complete = all(c["finite"] and all(c["attention_repeat_bitwise"]) for c in cases)
        with args.output.open("x") as f:
            json.dump(dict(scope="frozen native/reference top-k into installed decode attention; synthetic BF16 query/KV; full active rows only",
                precision_qualified=False, audit_complete=complete, vllm_version=version,
                backend_sha256=source_hash, reference_sha256=hashlib.sha256(raw).hexdigest(), cases=cases),
                f, indent=2, allow_nan=False)
        return 0 if complete else 1
    with args.output.open("x") as f:
        json.dump(dict(scope="synthetic BF16 Q/KV and unique permuted selected indices; installed serving attention only; no model quality or performance qualification",
            precision_qualified=False, audit_complete=True, persistent_exports_complete=True,
            vllm_version=version, backend_sha256=source_hash, cases=cases), f, indent=2, allow_nan=False)
    return 0


def rope_capture_contract(meta, cfg, inventory, tp):
    m, ctx, layer = meta["batch"], meta["ctx"], captured_block_layer(meta)
    if (tp != 8 or not 1 <= m <= 64 or not 1 <= ctx <= cfg["max_position_embeddings"]
            or cfg["qk_rope_head_dim"] != 64 or cfg["num_attention_heads"] != 64
            or cfg["rope_parameters"] != {"rope_theta": 8000000, "rope_type": "default"}
            or len(inventory["ranks"]) != tp):
        raise ValueError("unsupported pinned GLM RoPE contract")
    # get_rope caches one shared module; named_modules inventories its first owner only.
    name = "model.layers.0.self_attn.rotary_emb"
    cls = "vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding"
    for rank in inventory["ranks"]:
        module = rank["modules"][name]
        cache = module["tensors"]["cos_sin_cache"]
        if (module["class_name"] != cls or module["attributes"]["dtype"] != "torch.bfloat16"
                or cache["dtype"] != "torch.bfloat16" or cache["device_type"] != "cuda"
                or cache["shape"] != [cfg["max_position_embeddings"], 64]
                or cache["stride"] != [64, 1]):
            raise ValueError("loaded RoPE cache differs from pinned BF16 contract")
    return m, ctx, layer


def compare_packet_rope(args, version):
    import inspect
    from vllm.config import VllmConfig, CompilationConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires pinned vLLM inventory and original checkpoint")
    meta = json.loads((args.capture / "inputs/reference.json").read_text())
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    m, ctx, layer = rope_capture_contract(meta, cfg, inventory, args.tp)
    invocation_path = args.precision_inventory.parent / "invocation.json"
    compilation = json.loads(invocation_path.read_text())["effective_compilation_config"]
    if compilation["mode"] != "NONE" or "all" not in compilation["custom_ops"]:
        raise ValueError("requires the captured custom-op-enabled eager oracle")
    cls = RotaryEmbedding.__module__ + "." + RotaryEmbedding.__name__
    source = Path(inspect.getfile(RotaryEmbedding))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if any(rank["sources"][cls]["sha256"] != source_hash for rank in inventory["ranks"]):
        raise ValueError("installed rotary implementation differs from loaded inventory")
    config = VllmConfig(compilation_config=CompilationConfig(mode=0, custom_ops=compilation["custom_ops"]))
    with set_current_vllm_config(config), torch.device("cuda"):
        rope = get_rope(64, max_position=cfg["max_position_embeddings"], is_neox_style=False,
                        rope_parameters=cfg["rope_parameters"], dtype=torch.bfloat16)
    if type(rope) is not RotaryEmbedding or rope._forward_method.__name__ != "forward_hip":
        raise ValueError("unexpected rotary dispatch")
    positions = torch.full((m,), ctx - 1, dtype=torch.int64, device="cuda")
    records = []
    for rank in range(args.tp):
        def captured(name, shape):
            return load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", torch.bfloat16, shape,
                live_prefix=args.live_prefix and name.startswith("act."))[0]
        query = captured("act.qrr", (m, 8, 64)).cuda()
        key = captured("act.krr", (m, 1, 64)).cuda()
        expected = rope(positions, query.clone(), key.clone())
        repeated = rope(positions, query.clone(), key.clone())
        actual_q = captured("act.qr", (m, 8, 64))
        actual_k = torch.stack([captured(f"slot{row}.kv.{layer}.krot", (ctx, 64))[-1]
                                for row in range(m)]).unsqueeze(1)
        boundaries = {}
        for name, actual, want, repeat in zip(("query", "key"), (actual_q, actual_k), expected, repeated):
            boundary = boundary_difference(actual, want.cpu())
            boundary["reference_repeat_bitwise"] = tensor_digest(want) == tensor_digest(repeat)
            boundaries[name] = boundary
            with (args.output.parent / f"rank{rank}.rope-{name}.bf16").open("xb") as f:
                f.write(want.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        finite = all(bool(torch.isfinite(t).all()) for t in (query, key))
        passed = finite and all(b["finite"] and b["bitwise"] and b["reference_repeat_bitwise"]
                                for b in boundaries.values())
        records.append(dict(rank=rank, passed=passed, inputs_finite=finite,
                            query_sha256=tensor_digest(query), key_sha256=tensor_digest(key), boundaries=boundaries))
        print(json.dumps(records[-1]), flush=True)
    cache = rope.cos_sin_cache[:131072].cpu().contiguous()
    with (args.output.parent / "rope-cache.bf16").open("xb") as f:
        f.write(cache.view(torch.uint8).numpy().tobytes())
    report = dict(scope="installed rotary custom op conditioned on captured BF16 projection outputs; not full-model qualification",
        passed=all(r["passed"] for r in records), audit_complete=True, precision_qualified=False,
        vllm_version=version, shape=[m, ctx, 8, 64], use_aiter=rope.use_aiter,
        cache_inventory_owner="model.layers.0.self_attn.rotary_emb",
        forward_method=describe_call(rope._forward_method), source_sha256=source_hash,
        loaded_inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
        invocation_sha256=hashlib.sha256(invocation_path.read_bytes()).hexdigest(),
        cache=dict(file="rope-cache.bf16", shape=list(cache.shape), dtype=str(cache.dtype), sha256=tensor_digest(cache)),
        cases=records)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return 0 if report["passed"] else 1


def rotate_interleaved(query, cache):
    if query.shape[-1] != 64 or cache.shape != (64,):
        raise ValueError("requires 64-d interleaved rotary inputs")
    cos, sin = cache.float().chunk(2)
    even, odd = query.float()[..., ::2], query.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).bfloat16()


def check_rope(args):
    if args.reference_json is None:
        raise ValueError("requires a completed pinned rotary comparison")
    ref = json.loads(args.reference_json.read_text())
    if not ref["audit_complete"] or ref["vllm_version"] != "0.29.0" or len(ref["cases"]) != 8:
        raise ValueError("incomplete pinned rotary comparison")
    cache_meta = ref["cache"]
    cache, cache_hash = load_tensor(args.reference_json.parent / cache_meta["file"],
                                   torch.bfloat16, tuple(cache_meta["shape"]))
    if cache_hash != cache_meta["sha256"]:
        raise ValueError("rotary cache hash differs")
    m, ctx, heads, width = ref["shape"]
    if (heads, width) != (8, 64) or not 1 <= ctx <= len(cache):
        raise ValueError("unsupported captured rotary shape")
    angles = torch.arange(len(cache), dtype=torch.float64)[:, None] / (
        8000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64)[None, :] / 64))
    host = torch.cat((angles.cos(), angles.sin()), dim=1).float()
    cast_host = host.bfloat16()
    records = []
    for case in ref["cases"]:
        rank = case["rank"]
        for name, tensor, h in (("query", "act.qrr", 8), ("key", "act.krr", 1)):
            raw, digest = load_tensor(args.capture / "outputs" / f"rank{rank}.{tensor}.bin", torch.bfloat16, (m, h, 64))
            if digest != case[name + "_sha256"]:
                raise ValueError("rotary diagnostic input changed")
            want, digest = load_tensor(args.reference_json.parent / f"rank{rank}.rope-{name}.bf16", torch.bfloat16, raw.shape)
            if digest != case["boundaries"][name]["reference_sha256"]:
                raise ValueError("rotary diagnostic reference changed")
            records.append(dict(rank=rank, boundary=name,
                reference_coefficients=boundary_difference(rotate_interleaved(raw, cache[ctx - 1]), want),
                host_f64_to_f32_coefficients=boundary_difference(rotate_interleaved(raw, host[ctx - 1]), want),
                host_f64_to_bf16_coefficients=boundary_difference(rotate_interleaved(raw, cast_host[ctx - 1]), want)))
    differences = cast_host.view(torch.int16) != cache.view(torch.int16)
    prefix = {str(n): int(differences[:n].sum()) for n in (512, 8192, 71680, len(cache))}
    report = dict(scope="CPU diagnostic rotation with frozen reference coefficients; host F64 model is not an exact Rust or GPU implementation proof",
        reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(),
        cache_sha256=cache_hash, host_bf16_cache_mismatches_by_context=prefix,
        host_cache_mismatches_at_capture_position=int(differences[ctx - 1].sum()), cases=records)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(host_bf16_cache_mismatches_by_context=prefix,
        reference_coefficient_rotation_bitwise=all(r["reference_coefficients"]["bitwise"] for r in records),
        rounded_host_rotation_bitwise=all(r["host_f64_to_bf16_coefficients"]["bitwise"] for r in records))), flush=True)
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
    m, ctx, layer = meta["batch"], meta["ctx"], captured_block_layer(meta)
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
                or loaded["tensors"]["kv_cache"]["dtype"] != "torch.bfloat16"):
            raise ValueError("loaded backend/cache layout differs")
        capacity = ((ctx + block_size - 1) // block_size) * block_size
        def captured(name, dtype, shape):
            return load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape,
                live_prefix=args.live_prefix)[0]
        indices = captured("act.iidx", torch.int32, (m, 2048))
        indices_hash = tensor_digest(indices)
        expected_indices = sparse_decode_indices(indices, ctx, capacity)
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
        logical_kv = torch.cat((ckv, krot), dim=-1).cuda()
        padded_kv = torch.zeros((m, capacity, 576), dtype=torch.bfloat16, device="cuda")
        padded_kv[:, :ctx] = logical_kv
        kv = padded_kv.reshape(-1, block_size, 576)
        # Only replace serving allocation/config plumbing; execute the installed forward unchanged.
        impl = ROCMAiterMLASparseImpl.__new__(ROCMAiterMLASparseImpl)
        impl.num_heads, impl.kv_lora_rank, impl.kv_cache_dtype = 8, 512, "auto"
        impl.scale = (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) ** -0.5
        impl.q_concat_buffer = torch.empty((m, 8, 576), dtype=torch.bfloat16, device="cuda")
        impl.topk_indices_buffer = indices.cuda()
        attention_layer = SimpleNamespace(_q_scale=torch.ones((), device="cuda"), _k_scale=torch.ones((), device="cuda"))
        md = ROCMAiterMLASparseMetadata(num_reqs=m, max_query_len=1, max_seq_len=ctx, num_actual_tokens=m,
            query_start_loc=qo, slot_mapping=torch.arange(m, device="cuda") * capacity + ctx - 1,
            block_table=torch.arange(m * capacity // block_size, dtype=torch.int32, device="cuda").reshape(m, -1),
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
                    logical_kv, indices[:, :min(ctx, 2048)].cuda(),
                    model_splits, impl.scale, rounded).cpu()
                models.append(dict(splits=model_splits, bf16_probability=rounded,
                    versus_plow=boundary_difference(model, actual),
                    versus_pinned=boundary_difference(model, expected.cpu())))
        if all(ranges == work_ranges[0] for ranges in work_ranges):
            ends = [end for _, end in work_ranges[0]]
            for model_tile in (None, 32):
                model = split_attention_rounding_model(torch.cat((qa, qr), dim=-1),
                    logical_kv, indices[:, :selected_count].cuda(), len(ends),
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
            selected_indices_sha256=indices_hash,
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
            selected_indices_sha256={str(r["rank"]): r["selected_indices_sha256"] for r in records}, sources=sources,
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


def native_routed_boundaries(prefix, h, i, experts, ids, gates, *, live_prefix=False):
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
        tensors[name], hashes[name] = load_tensor(Path(str(prefix) + f".act.{name}.bin"), dtype, shape,
                                                 live_prefix=live_prefix and (name.startswith("routed_") or name == "moe_meta"))
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


def bf16_order_reachability(parts, output):
    if (parts.dtype != torch.bfloat16 or parts.ndim != 3 or not 1 <= parts.shape[1] <= 8
            or output.dtype != parts.dtype or output.shape != (parts.shape[0], parts.shape[2])
            or not bool(torch.isfinite(parts).all() and torch.isfinite(output).all())):
        raise ValueError("requires finite BF16 parts and matching output")
    parts, output = parts.cpu(), output.cpu()
    reached = torch.zeros_like(output, dtype=torch.bool)
    checked = 0
    for order in itertools.permutations(range(parts.shape[1])):
        value = torch.zeros_like(output)
        for slot in order:
            value = value + parts[:, slot]
        reached |= value.view(torch.int16) == output.view(torch.int16)
        checked += 1
        if bool(reached.all()):
            break
    return dict(elementwise_reachable=bool(reached.all()), unreachable_count=int((~reached).sum()),
                elements=output.numel(), permutations_checked=checked,
                scope="per-element BF16 serial addition orders, not one common order or execution-schedule proof")


def check_routed_reachability(args):
    audit = json.loads(args.reference_json.read_text())
    if (audit.get("passed") is not True or audit.get("audit_complete") is not True
            or audit.get("vllm_version") != "0.29.0" or len(audit["cases"]) != args.tp
            or {row["rank"] for row in audit["cases"]} != set(range(args.tp))):
        raise ValueError("requires complete passed pinned routed audit")
    records = []
    for row in audit["cases"]:
        rank = row["rank"]
        m, h, i, _, topk = row["shape"]
        if not row["stable_boundaries_bitwise"]:
            raise ValueError("stable routed boundaries must match before reduction analysis")
        parts = torch.empty((m, topk, h), dtype=torch.bfloat16)
        seen = torch.zeros((m, topk), dtype=torch.bool)
        for case in row["isolated_weighted_down"]:
            shape, _, value, digest = load_case(args.reference_json.parent / case["file"], weighted=True)
            tokens, slots = torch.tensor(case["tokens"], dtype=torch.long), torch.tensor(case["slots"], dtype=torch.long)
            if (digest != case["sha256"] or not case["finite"] or not case["repeat_bitwise"]
                    or shape != (len(tokens), h, i) or tokens.shape != slots.shape
                    or bool((tokens < 0).any() or (tokens >= m).any() or (slots < 0).any() or (slots >= topk).any())
                    or (tokens * topk + slots).unique().numel() != tokens.numel()
                    or bool(seen[tokens, slots].any())):
                raise ValueError("isolated routed partial provenance/coverage mismatch")
            parts[tokens, slots] = value
            seen[tokens, slots] = True
        if not bool(seen.all()):
            raise ValueError("missing isolated routed partials")
        boundary = row["boundaries"]["stage2.output"]
        results = {}
        for name, expected_hash in (("plow-routed", boundary["plow_sha256"]),
                                    ("reference", boundary["sha256"]), ("repeat", boundary["repeat_sha256"])):
            path = args.reference_json.parent / f"{args.reference_json.stem}.rank{rank}.{name}.bf16"
            value, digest = load_tensor(path, torch.bfloat16, (m, h))
            if digest != expected_hash:
                raise ValueError("routed output provenance mismatch")
            results[name] = dict(sha256=digest, **bf16_order_reachability(parts, value))
        record = dict(rank=rank, parts_sha256=tensor_digest(parts), outputs=results)
        records.append(record)
        print(json.dumps(record), flush=True)
    passed = all(value["elementwise_reachable"] for row in records for value in row["outputs"].values())
    with args.output.open("x") as f:
        json.dump(dict(scope="per-element BF16 serial expert-addition reachability; not a common order, schedule proof, router or full-model qualification",
            passed=passed, precision_qualified=False,
            reference_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(), cases=records), f, indent=2)
        f.write("\n")
    return 0 if passed else 1


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
        ids, gates, route_digest = load_routes(args.capture / "outputs" / f"rank{rank}.act.tab.bin", m, topk, experts,
                                              live_prefix=getattr(args, "live_prefix", False))
        if route_digest != row["routes_sha256"]:
            raise ValueError("routed table changed since reference audit")
        weights = routed_weights(args.checkpoint, captured_block_layer(metadata), rank, args.tp)
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


def router_reference_weights(report, rank, xhash, routehash, ids):
    rows = [row for row in report["cases"] if row["rank"] == rank]
    if len(rows) != 1 or report.get("vllm_version") != "0.29.0":
        raise ValueError("requires unique pinned router reference rank")
    row = rows[0]
    if row["input_sha256"] != xhash or row["native_routes_sha256"] != routehash:
        raise ValueError("router input/routes provenance mismatch")
    routed = row["comparisons"]["reference_logits"]
    other_ids = torch.tensor(routed["ids"], dtype=torch.int32)
    weights = torch.tensor(routed["weights"], dtype=torch.float32)
    if (not routed["repeat_bitwise"] or not row["logits_repeat_bitwise"]
            or not torch.equal(ids, other_ids) or weights.shape != ids.shape
            or not torch.isfinite(weights).all() or bool((weights < 0).any())
            or tensor_digest(weights) != routed["weights_sha256"]):
        raise ValueError("requires stable identical ordered expert routes and hashed weights")
    return weights


def routed_router_impact(call, ids, gates, alternative, native_parts):
    sorted_ids = call.args[3]
    token = sorted_ids.long() & 0xffffff
    slot = (sorted_ids.long() >> 24) & 0xff
    valid_count = int(call.args[5][0].item())
    valid = ((token < ids.shape[0]) & (slot < ids.shape[1])
        & (torch.arange(sorted_ids.numel(), device=sorted_ids.device) < valid_count))
    token, slot = token.clamp_max(ids.shape[0] - 1), slot.clamp_max(ids.shape[1] - 1)
    gpu_gates, gpu_ids = gates.to(sorted_ids.device), ids.to(sorted_ids.device)
    if not torch.equal(call.keywords["sorted_weights"][valid].view(torch.int32),
                       gpu_gates[token[valid], slot[valid]].view(torch.int32)):
        raise ValueError("sorted native route weights differ from captured routes")
    new_sorted = torch.where(valid, alternative.to(sorted_ids.device)[token, slot],
                             torch.zeros_like(call.keywords["sorted_weights"]))
    parts = torch.empty_like(native_parts)
    for expert in sorted(set(ids.flatten().tolist())):
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        kw = dict(call.keywords)
        kw["sorted_weights"] = isolated_route_weights(gpu_ids, sorted_ids, new_sorted, valid_count, expert)
        values = []
        for _ in range(2):
            out = torch.zeros_like(call.args[6])
            call.func(*(call.args[:6] + (out,) + call.args[7:]), **kw)
            values.append(out.cpu()[tokens].contiguous())
        if (not torch.isfinite(values[0]).all()
                or not torch.equal(values[0].view(torch.int16), values[1].view(torch.int16))):
            raise ValueError("router impact isolated output is nonfinite or unstable")
        parts[tokens, slots] = values[0]
    return dict(scope="same installed stage2 operands and selected experts; only route weights replaced; isolated contributions avoid atomic-order noise",
        changed_elements=int((parts.view(torch.int16) != native_parts.view(torch.int16)).sum()),
        elements=parts.numel(), max_abs=float((parts.float()-native_parts.float()).abs().max()),
        native_parts_sha256=tensor_digest(native_parts), reference_parts_sha256=tensor_digest(parts),
        reference_weights_sha256=tensor_digest(alternative), repeat_bitwise=True)


def routed_stage1_partials(a, asc, w, ws, ids, splitk):
    m, k = a.shape
    experts, n, wk = w.shape
    if (splitk < 1 or a.device.type != "cpu" or w.device.type != "cpu" or k != wk or k % (128 * splitk)
            or tuple(asc.shape) != (m, k // 128) or tuple(ws.shape) != (experts, n // 128, k // 128)
            or n % 128 or ids.shape[0] != m or ids.min() < 0 or ids.max() >= experts):
        raise ValueError("unsupported CPU routed accumulation geometry")
    selected = w.view(torch.uint8)[ids.flatten()].view(w.dtype).double()
    scales = ws[ids.flatten()].repeat_interleave(128, dim=1)
    inputs = a.double().repeat_interleave(ids.shape[1], dim=0)
    input_scales = asc.repeat_interleave(ids.shape[1], dim=0)
    parts = []
    width = k // splitk // 128
    for part in range(splitk):
        acc = torch.zeros((ids.numel(), n), dtype=torch.float32)
        for group in range(part * width, (part + 1) * width):
            lo = group * 128
            dot = (selected[:, :, lo:lo + 128] * inputs[:, None, lo:lo + 128]).sum(dim=-1).float()
            scale = input_scales[:, group, None] * scales[:, :, group]
            # FP64 intermediates model FP32 FMA; this is not an MFMA instruction oracle.
            acc = (dot.double() * scale.double() + acc.double()).float()
        parts.append(acc)
    return torch.stack(parts)


def model_routed_stage1(args):
    report = json.loads(args.reference_json.read_text())
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    config = json.loads((args.checkpoint / "config.json").read_text())
    m, h, topk = metadata["batch"], config["hidden_size"], config["num_experts_per_tok"]
    layer, rows = captured_block_layer(metadata), []
    for case in report["cases"]:
        rank = case["rank"]
        ids, _, routehash = load_routes(args.capture / "outputs" / f"rank{rank}.act.tab.bin",
            m, topk, config["n_routed_experts"], live_prefix=True)
        if routehash != case["routes_sha256"]:
            raise ValueError("route provenance mismatch")
        weights = routed_weights(args.checkpoint, layer, rank, args.tp)
        for key, value in (("gate_up", weights[0]), ("gate_up_scale", weights[2])):
            if tensor_digest(value) != case["checkpoint_shard_sha256"][key]:
                raise ValueError("checkpoint provenance mismatch")
        def captured(name, dtype, shape):
            value, digest = load_tensor(args.reference_json.parent / f"{args.reference_json.stem}.rank{rank}.{name}.bin", dtype, shape)
            expected = (case["stage1_diagnostic"]["tensors"][name.split(".")[-1]]["sha256"]
                        if name.startswith("stage1-diagnostic.") else case["boundaries"][name]["sha256"])
            if digest != expected:
                raise ValueError("stage1 boundary provenance mismatch")
            return value
        a = captured("stage1.input", torch.float8_e4m3fn, (m, h))
        asc = captured("stage1.scale", torch.float32, (m, h // 128))
        pre = captured("stage1-diagnostic.preactivation", torch.float32, (m * topk, weights[0].shape[1]))
        if args.export_routed_stage1_partials:
            diagnostic = case["stage1_diagnostic"]
            splitk = case["selected"]["ksplit"]
            width = h // splitk
            if (h % splitk or width % 128 or not diagnostic["partials_repeat_bitwise"]
                    or not diagnostic["partials_sum"]["bitwise"]):
                raise ValueError("requires stable, reconstructing block-aligned partials")
            isolated = captured("stage1-diagnostic.partials", torch.float32, (splitk, m * topk, weights[0].shape[1]))
            repeated = captured("stage1-diagnostic.partials_repeat", torch.float32, tuple(isolated.shape))
            if not torch.equal(isolated, isolated.half().float()) or not torch.equal(isolated, repeated):
                raise ValueError("partial values are not stable exact FP16")
            exports = []
            for expert in sorted(set(ids.flatten().tolist())):
                token, slot = (ids == expert).nonzero(as_tuple=True)
                for part in range(splitk):
                    lo, hi = part * width, (part + 1) * width
                    path = args.output.parent / f"rank{rank}.expert{expert}.part{part}.fp16.bin"
                    write_case(path, a.view(torch.uint8)[token, lo:hi], weights[0][expert, :, lo:hi],
                        asc[token, lo // 128:hi // 128].T.contiguous(), weights[2][expert, :, lo // 128:hi // 128],
                        isolated[part, token * topk + slot].half(), fp16=True)
                    exports.append(dict(expert=expert, part=part, file=path.name,
                        shape=[token.numel(), weights[0].shape[1], width], output_dtype="float16",
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows.append(dict(rank=rank, exports=exports))
            continue
        parts = routed_stage1_partials(a, asc, weights[0], weights[2], ids, case["selected"]["ksplit"])
        isolated = (captured("stage1-diagnostic.partials", torch.float32, tuple(parts.shape))
                    if "partials" in case["stage1_diagnostic"]["tensors"] else None)
        variants = {}
        half = parts.half()
        half_rtz = torch.where(half.float().abs() > parts.abs(), torch.nextafter(half, torch.zeros_like(half)), half)
        for name, rounded in (("float32", parts), ("float16_rne", half.float()),
                              ("float16_rtz", half_rtz.float()), ("bfloat16_rne", parts.bfloat16().float())):
            total = torch.zeros_like(pre)
            for part in rounded:
                total += part
            variants[name] = boundary_difference(total, pre)
            if isolated is not None:
                variants[name]["isolated_partials"] = boundary_difference(rounded.flatten(0, 1), isolated.flatten(0, 1))
        row = dict(rank=rank, partials_sha256=tensor_digest(parts), variants=variants)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with args.output.open("x") as f:
        json.dump(dict(scope=("installed isolated stage1 FP16 partial fixtures; not serving qualification" if args.export_routed_stage1_partials
            else "CPU rounding diagnostic; FP64 block dot then FP32 FMA model, not native MFMA qualification"),
            source_report_sha256=hashlib.sha256(args.reference_json.read_bytes()).hexdigest(), cases=rows), f, indent=2, allow_nan=False)
        f.write("\n")
    return 0


def diagnose_routed_stage1(fm, call, native, reference, output, rank):
    if call.func != fm.ck_moe_stage1 or call.keywords.get("splitk", 0) <= 1:
        raise ValueError("stage1 diagnostic requires installed CK split-K wrapper")
    captures = []
    original = fm.aiter.ck_moe_stage1_fwd

    def capture(*operands, **keywords):
        result = original(*operands, **keywords)
        tmp = operands[6]
        if tmp.dtype != torch.float32 or tmp.ndim != 2:
            raise ValueError("expected split-K FP32 preactivation")
        captures.append(tmp[:reference.shape[0] * reference.shape[1]].detach().cpu().clone())
        return result

    def replay(splitk, inputs=None):
        out = torch.zeros_like(call.args[6])
        keywords = dict(call.keywords, splitk=splitk)
        operands = call.args[:6] + (out,) + call.args[7:]
        if inputs is not None:
            operands = (inputs,) + operands[1:]
        call.func(*operands, **keywords)
        return out.detach().cpu().clone()

    try:
        fm.aiter.ck_moe_stage1_fwd = capture
        split = replay(call.keywords["splitk"])
        split_repeat = replay(call.keywords["splitk"])
        splitk = call.keywords["splitk"]
        k = call.args[0].shape[-1]
        if k % splitk:
            raise ValueError("stage1 isolation requires equally sized K partitions")
        for _ in range(2):
            for part in range(splitk):
                masked = torch.zeros_like(call.args[0])
                lo, hi = part * (k // splitk), (part + 1) * (k // splitk)
                masked[:, lo:hi] = call.args[0][:, lo:hi]
                replay(splitk, masked)
    finally:
        fm.aiter.ck_moe_stage1_fwd = original
    if len(captures) != 2 + 2 * splitk or captures[0].shape != (reference.numel() // reference.shape[-1], 2 * reference.shape[-1]):
        raise ValueError("unexpected split-K preactivation geometry/count")
    unsplit, unsplit_repeat = replay(0), replay(0)
    values = dict(split=split, split_repeat=split_repeat, unsplit=unsplit,
                  unsplit_repeat=unsplit_repeat, preactivation=captures[0],
                  preactivation_repeat=captures[1], partials=torch.stack(captures[2:2 + splitk]),
                  partials_repeat=torch.stack(captures[2 + splitk:]))
    records = {}
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite stage1 diagnostic: {name}")
        dest = output.parent / f"{output.stem}.rank{rank}.stage1-diagnostic.{name}.bin"
        with dest.open("xb") as f:
            f.write(value.contiguous().view(torch.uint8).numpy().tobytes())
        records[name] = dict(file=dest.name, shape=list(value.shape), dtype=str(value.dtype),
                             sha256=tensor_digest(value))
        if value.shape == reference.shape and value.dtype == reference.dtype:
            records[name].update(
                reference_changed_elements=int((value.view(torch.int16) != reference.view(torch.int16)).sum()),
                native_changed_elements=int((value.view(torch.int16) != native.view(torch.int16)).sum()))
    total = torch.zeros_like(captures[0])
    for part in values["partials"]:
        total += part
    return dict(scope="diagnostic only; splitk=0 is not the selected reference configuration; isolated parts zero other prequantized K ranges with original scales",
                tensors=records, partials_sum=boundary_difference(total, captures[0]),
                partials_repeat_bitwise=tensor_digest(values["partials"]) == tensor_digest(values["partials_repeat"]),
                preactivation_repeat_bitwise=torch.equal(captures[0].view(torch.int32), captures[1].view(torch.int32)))


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
    m, layer = metadata["batch"], captured_block_layer(metadata)
    h, topk, experts = (config[key] for key in
                        ("hidden_size", "num_experts_per_tok", "n_routed_experts"))
    i = config["moe_intermediate_size"] // args.tp
    router_reference = json.loads(args.router_reference.read_text()) if args.router_reference else None
    rows = []
    for rank in range(args.tp):
        prefix = args.capture / "outputs" / f"rank{rank}"
        x, xhash = load_tensor(Path(str(prefix) + ".act.xn2.bin"), torch.bfloat16, (m, h), live_prefix=args.live_prefix)
        ids, gates, routehash = load_routes(Path(str(prefix) + ".act.tab.bin"), m, topk, experts, live_prefix=args.live_prefix)
        alternative_gates = (router_reference_weights(router_reference, rank, xhash, routehash, ids)
            if router_reference is not None else None)
        native = None
        if args.routed_w8a8:
            native, native_hashes = native_routed_boundaries(prefix, h, i, experts, ids, gates, live_prefix=args.live_prefix)
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
            stage1_diagnostic = None
            if args.routed_stage1_diagnostic:
                if selected.run_1stage or native is None or [name for name, _ in calls] != ["stage1", "stage2"]:
                    raise ValueError("stage1 diagnostic requires native two-stage reference")
                stage1_diagnostic = diagnose_routed_stage1(fm, calls[0][1], native["stage1.output"],
                    snapshots["stage1.output"], args.output, rank)
            if args.routed_down_isolate:
                if selected.run_1stage or [name for name, _ in fm.kernel_bench_callable] != ["stage1", "stage2"]:
                    raise ValueError("weighted-down isolation requires two-stage reference")
                down_cases, down_parts = isolate_routed_down(args, rank, fm.kernel_bench_callable[1][1],
                    ids, gates, *down_original)
                router_impact = (routed_router_impact(fm.kernel_bench_callable[1][1], ids, gates,
                    alternative_gates, down_parts) if alternative_gates is not None else None)
            else:
                down_cases = []
                router_impact = None
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
            router_weight_impact=router_impact,
            stage1_diagnostic=stage1_diagnostic,
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
            activation_hash_scope="live_prefix_for_row_tensors" if args.live_prefix else "whole_file",
            vllm_version=version, checkpoint=str(args.checkpoint), precision_qualified=False,
            passed=passed, audit_complete=complete,
            plow_reduction=("captured BF16 atomic output; exact addition-order extrema, not bitwise reduction or reachability of interior values"
                if args.routed_w8a8 else "FP32 captured weighted parts summed in slot order, then BF16; not packet shared-add boundary"),
            aiter_source_sha256=hashlib.sha256(Path(fm.__file__).read_bytes()).hexdigest(),
            router_reference_sha256=(hashlib.sha256(args.router_reference.read_bytes()).hexdigest()
                if args.router_reference else None),
            cases=rows), f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if (passed if args.routed_w8a8 else complete) else 1


def packet_oproj_cases(capture, checkpoint, tp, *, live_prefix=False, diagnostic=False):
    from safetensors import safe_open

    measurement = json.loads((capture / "measurement.json").read_text())
    metadata = json.loads((capture / "inputs/reference.json").read_text())
    if (tp < 1 or measurement.get("tp") != tp or measurement.get("scope") != "single-block-decode"
            or (measurement.get("oracle_verified") is not True and not diagnostic)
            or not isinstance(measurement.get("oracle_verified"), bool)
            or measurement.get("batch") != metadata.get("batch")):
        raise ValueError("requires a passed single-block capture with matching TP and batch")
    m, layer = metadata["batch"], captured_block_layer(metadata)
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
            tensors[name], hashes[name] = load_tensor(path, dtype, shape,
                live_prefix=live_prefix and name.startswith("act."))
        w = weight[:, rank * k:(rank + 1) * k].contiguous()
        ws = scale[:, rank * (k // 128):(rank + 1) * (k // 128)].contiguous()
        weight_equal = torch.equal(tensors[weight_name + "_fp8"], w.view(torch.uint8))
        scale_equal = torch.equal(tensors[scale_name].view(torch.int32), ws.view(torch.int32))
        yield dict(rank=rank, shape=(m, n, k), hashes=hashes,
                   block_oracle_verified=measurement["oracle_verified"],
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


def router_contract(config, inventory, tp, layer):
    prefix = f"model.layers.{layer}.mlp.gate"
    if (config.get("architectures") != ["GlmMoeDsaForCausalLM"]
            or config.get("scoring_func") != "sigmoid" or config.get("n_group") != 1
            or config.get("topk_group") != 1 or config.get("num_experts_per_tok") != 8
            or config.get("n_routed_experts") != 256 or config.get("hidden_size") != 6144
            or config.get("norm_topk_prob") is not True
            or config.get("routed_scaling_factor") != 2.5):
        raise ValueError("unsupported GLM router contract")
    if sorted(r["rank"] for r in inventory["ranks"]) != list(range(tp)):
        raise ValueError("router inventory ranks differ from TP")
    for rank in inventory["ranks"]:
        gate = rank["modules"][prefix]
        if (gate["class_name"] != "vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear"
                or gate["attributes"]["out_dtype"] != "torch.float32"
                or gate["tensors"]["weight"]["dtype"] != "torch.bfloat16"
                or gate["tensors"]["weight"]["shape"] != [256, 6144]
                or gate["tensors"]["e_score_correction_bias"]["dtype"] != "torch.float32"):
            raise ValueError("loaded router precision differs from replay")
    return prefix


def compare_packet_router(args, version):
    import inspect
    from types import SimpleNamespace
    from safetensors import safe_open
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
    from vllm._aiter_ops import rocm_aiter_ops

    if version != "0.29.0" or args.checkpoint is None or args.precision_inventory is None:
        raise ValueError("requires pinned vLLM0.29, original checkpoint and loaded inventory")
    config = json.loads((args.checkpoint / "config.json").read_text())
    inventory = json.loads(args.precision_inventory.read_text())
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    measurement = json.loads((args.capture / "measurement.json").read_text())
    if (measurement.get("scope") != "single-block-decode" or measurement.get("tp") != args.tp
            or measurement.get("batch") != metadata["batch"]):
        raise ValueError("requires matching single-block TP capture")
    prefix = router_contract(config, inventory, args.tp, captured_block_layer(metadata))
    gate_name = GateLinear.__module__ + "." + GateLinear.__name__
    source_digest = hashlib.sha256(Path(inspect.getfile(GateLinear)).read_bytes()).hexdigest()
    if any(r["sources"][gate_name]["sha256"] != source_digest for r in inventory["ranks"]):
        raise ValueError("installed gate source differs from loaded inventory")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in ("weight", "e_score_correction_bias"):
        key = prefix + "." + name
        with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as shard:
            weights[name] = shard.get_tensor(key)
    weight, bias = weights["weight"], weights["e_score_correction_bias"]
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.float32 or tuple(bias.shape) != (256,):
        raise ValueError("original router weight precision mismatch")
    gate = SimpleNamespace(weight=weight.cuda(), out_dtype=torch.float32,
        allow_ll_bf16_gemm=False, allow_fp32_router_gemm=False,
        allow_bf16x3_router_gemm=False, allow_cublas_router_gemm=True)
    router = create_fused_moe_router(top_k=8, global_num_experts=256, renormalize=True,
        use_grouped_topk=True, num_expert_group=1, topk_group=1, scoring_func="sigmoid",
        routed_scaling_factor=2.5, e_score_correction_bias=bias.cuda())
    if not rocm_aiter_ops.is_fused_moe_enabled():
        raise ValueError("captured AITER routing backend is not enabled")
    records = []
    for rank in range(args.tp):
        base = args.capture / "outputs"
        m = metadata["batch"]
        x, xhash = load_tensor(base / f"rank{rank}.act.xn2.bin", torch.bfloat16,
            (m, 6144), live_prefix=args.live_prefix)
        native_logits, lhash = load_tensor(base / f"rank{rank}.act.rlogit.bin", torch.float32,
            (m, 256), live_prefix=args.live_prefix)
        native_ids, native_weights, thash = load_routes(base / f"rank{rank}.act.tab.bin",
            m, 8, 256, live_prefix=args.live_prefix)
        logits, _ = GateLinear.forward(gate, x.cuda())
        logits_repeat, _ = GateLinear.forward(gate, x.cuda())
        comparisons = {}
        for label, values in (("native_logits", native_logits.cuda()), ("reference_logits", logits)):
            gates, ids = router._compute_routing(x.cuda(), values, torch.int32)
            gates2, ids2 = router._compute_routing(x.cuda(), values, torch.int32)
            gates, ids, gates2, ids2 = [v.cpu() for v in (gates, ids, gates2, ids2)]
            order = ids.argsort(dim=1)
            native_order = native_ids.argsort(dim=1)
            aligned, native_aligned = gates.gather(1, order), native_weights.gather(1, native_order)
            comparisons[label] = dict(ids_equal=torch.equal(ids, native_ids),
                sets_equal=torch.equal(ids.gather(1, order), native_ids.gather(1, native_order)),
                weights_bitwise=torch.equal(aligned.view(torch.int32), native_aligned.view(torch.int32)),
                weights_max_abs=float((aligned - native_aligned).abs().max()),
                repeat_bitwise=torch.equal(gates.view(torch.int32), gates2.view(torch.int32)) and torch.equal(ids, ids2),
                ids=ids.tolist(), weights=gates.tolist(), weights_sha256=tensor_digest(gates))
        ref = logits.cpu()
        records.append(dict(rank=rank, input_sha256=xhash, native_logits_sha256=lhash,
            native_routes_sha256=thash, reference_logits_sha256=tensor_digest(ref),
            logits_bitwise=torch.equal(ref.view(torch.int32), native_logits.view(torch.int32)),
            logits_max_abs=float((ref-native_logits).abs().max()),
            logits_repeat_bitwise=torch.equal(ref.view(torch.int32), logits_repeat.cpu().view(torch.int32)),
            comparisons=comparisons))
    passed = all(r["logits_bitwise"] and r["logits_repeat_bitwise"] and all(
        c["sets_equal"] and c["weights_bitwise"] and c["repeat_bitwise"]
        for c in r["comparisons"].values()) for r in records)
    sources = {str(Path(inspect.getfile(obj))): hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
        for obj in (GateLinear, create_fused_moe_router, type(router))}
    report = dict(passed=passed, precision_qualified=False, vllm_version=version,
        scope="native-input router diagnostic: installed ROCm GateLinear tier4 and factory-selected routing; not full-model qualification",
        router_class=type(router).__name__, sources=sources,
        inventory_sha256=hashlib.sha256(args.precision_inventory.read_bytes()).hexdigest(),
        weights={k: tensor_digest(v) for k, v in weights.items()}, cases=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


def check_block_rows(args):
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    measurement = json.loads((args.capture / "measurement.json").read_text())
    m = metadata["batch"]
    if (m < 1 or measurement.get("batch") != m or measurement.get("tp") != args.tp
            or measurement.get("scope") != "single-block-decode"):
        raise ValueError("requires matching batched block capture")
    cases = []
    for name in [*metadata["stages"], "act.xnext"]:
        path = args.capture / "inputs" / ("reference.bf16" if name == "act.xnext" else name + ".reference.bf16")
        if path.stat().st_size % (2 * m):
            raise ValueError("invalid block reference row size")
        shape = (m, path.stat().st_size // (2 * m))
        reference, digest = load_tensor(path, torch.bfloat16, shape)
        for rank in range(args.tp):
            value, actual_digest = load_tensor(args.capture / "outputs" / f"rank{rank}.{name}.bin",
                torch.bfloat16, shape, live_prefix=args.live_prefix)
            if not torch.isfinite(reference).all() or not torch.isfinite(value).all():
                raise ValueError("nonfinite block row")
            delta = value.double() - reference.double()
            relative = delta.norm(dim=1) / reference.double().norm(dim=1).clamp_min(1e-30)
            cases.append(dict(stage=name, rank=rank, shape=shape, reference_sha256=digest,
                actual_sha256=actual_digest, row_relative_l2=relative.tolist(),
                row_changed_elements=(value.view(torch.int16) != reference.view(torch.int16)).sum(dim=1).tolist(),
                max_row_relative_l2=float(relative.max())))
    report = dict(scope="all live block rows; strict numerical diagnostic, no tolerance promotion",
        precision_qualified=False, passed=all(not any(c["row_changed_elements"]) for c in cases), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def mlp_combine_replay(shared, routed):
    if (not shared or len(shared) != len(routed)
            or any(x.dtype != torch.bfloat16 or x.shape != shared[0].shape
                   or not torch.isfinite(x).all() for x in shared + routed)):
        raise ValueError("requires finite equally shaped BF16 shared/routed rank pairs")
    partials = [(s.float() + r.float()).bfloat16() for s, r in zip(shared, routed)]
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        total += partial.float()
    return total.bfloat16()


def check_mlp_combine(args):
    measurement = json.loads((args.capture / "measurement.json").read_text())
    metadata = json.loads((args.capture / "inputs/reference.json").read_text())
    m = metadata.get("batch")
    if (measurement.get("scope") != "single-block-decode" or args.tp < 2
            or measurement.get("tp") != args.tp or measurement.get("batch") != m
            or not isinstance(m, int) or m < 1):
        raise ValueError("requires matching single-block decode TP and batch")
    output_path = args.capture / "inputs/reference.bf16"
    if output_path.stat().st_size % (m * 2):
        raise ValueError("invalid live block output size")
    shape = (m, output_path.stat().st_size // (m * 2))
    tensors, digests = {}, {}
    for rank in range(args.tp):
        for name in ("shared", "part", "attn", "xmid", "xnext"):
            key = f"rank{rank}.act.{name}.bin"
            tensors[key], digests[key] = load_tensor(args.capture / "outputs" / key,
                torch.bfloat16, shape, live_prefix=args.live_prefix)
            if not torch.isfinite(tensors[key]).all():
                raise ValueError(f"non-finite MLP boundary: {key}")
    reduced = mlp_combine_replay(
        [tensors[f"rank{rank}.act.shared.bin"] for rank in range(args.tp)],
        [tensors[f"rank{rank}.act.part.bin"] for rank in range(args.tp)])
    cases = []
    for rank in range(args.tp):
        residual = tensors[f"rank{rank}.act.xmid.bin"]
        expected = (residual.float() + reduced.float()).bfloat16()
        cases.append(dict(rank=rank, reduced_bitwise=torch.equal(reduced.view(torch.int16),
            tensors[f"rank{rank}.act.attn.bin"].view(torch.int16)),
            output_bitwise=torch.equal(expected.view(torch.int16),
            tensors[f"rank{rank}.act.xnext.bin"].view(torch.int16))))
    passed = all(row["reduced_bitwise"] and row["output_bitwise"] for row in cases)
    report = dict(passed=passed, precision_qualified=False, shape=shape, tp=args.tp,
        scope="native-conditioned BF16 shared+routed combine, rank-ordered FP32 TP sum, residual; not vLLM collective equivalence",
        capture=str(args.capture), live_prefix=args.live_prefix, input_sha256=digests,
        measurement_sha256=hashlib.sha256((args.capture / "measurement.json").read_bytes()).hexdigest(),
        cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


def captured_block_layer(metadata):
    if "layer" in metadata:
        return metadata["layer"]
    layers = {int(name.split(".")[1]) for name in metadata.get("files", {})
              if name.startswith("kv.") and name.endswith(".ckv.bin") and name.split(".")[1].isdigit()}
    if len(layers) != 1:
        raise ValueError("capture must identify exactly one block layer")
    return layers.pop()


def packet_shared_cases(capture, checkpoint, tp, *, live_prefix=False):
    from safetensors import safe_open

    measurement = json.loads((capture / "measurement.json").read_text())
    metadata = json.loads((capture / "inputs/reference.json").read_text())
    if (tp < 1 or measurement.get("tp") != tp or measurement.get("scope") != "single-block-decode"
            or not isinstance(measurement.get("oracle_verified"), bool)
            or measurement.get("batch") != metadata.get("batch")):
        raise ValueError("requires an explicitly validated single-block capture with matching TP and batch")
    m, layer = metadata["batch"], captured_block_layer(metadata)
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
            value, hashes[name] = load_tensor(capture / "outputs" / f"rank{rank}.{name}.bin", dtype, shape,
                                             live_prefix=live_prefix and name.startswith("act."))
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
                   activation_hash_scope="live_prefix" if live_prefix else "whole_file",
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
    for row, t in packet_shared_cases(args.capture, args.checkpoint, args.tp, live_prefix=args.live_prefix):
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
    for row, (x, q, scales, w, ws, plow) in packet_oproj_cases(args.capture, args.checkpoint, args.tp,
            live_prefix=args.live_prefix, diagnostic=args.oproj_diagnostic):
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

                    partials, exports = [], []
                    for part in range(4):
                        lo, hi = part * (k // 4), (part + 1) * (k // 4)
                        operands = (reference_q[:, lo:hi].contiguous(), gpu_w[:, lo:hi].contiguous(),
                                    reference_s[:, lo // 128:hi // 128].contiguous(),
                                    gpu_ws[:, lo // 128:hi // 128].contiguous())
                        results = []
                        for _ in range(2):
                            out = torch.empty((m, n), dtype=torch.bfloat16, device=reference_q.device)
                            gemm_a8w8_blockscale_ck(*operands, out,
                                splitK=0, kernelName=str(config["kernelName"]))
                            results.append(out.cpu())
                        partials.append(results[0])
                        dest = args.output.parent / f"{args.output.stem}.rank{row['rank']}.partial{part}.bin"
                        pa, pw, ps, pws = (v.cpu() for v in operands)
                        write_case(dest, pa.view(torch.uint8), pw, ps.T.contiguous(), pws, results[0])
                        exports.append(dict(file=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                            k_begin=lo, k_end=hi, finite=bool(torch.isfinite(results[0]).all()),
                            repeat_bitwise=torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))))
                    row["isolated_partials"] = exports
                    ordered = ordered_bf16_sum(partials)
                    ordered_rel = (a - ordered.double()).norm(dim=1) / ordered.double().norm(dim=1).clamp_min(1e-30)
                    row["ordered_split4_audit"] = dict(
                        scope="same CK kernel on four K/4 slices, fixed-order BF16 adds; not the active atomic route",
                        finite=bool(torch.isfinite(ordered).all()),
                        max_row_rel_l2=float(ordered_rel.max()),
                        bitwise=torch.equal(plow.view(torch.int16), ordered.view(torch.int16)))
                    lo, hi = bf16_order_bounds(torch.stack(partials, dim=1))
                    row["split4_addition_order_bounds"] = {tag: bool(torch.isfinite(value).all()
                        and ((value >= lo) & (value <= hi)).all())
                        for tag, value in (("plow", plow), ("reference", reference), ("repeat", repeat))}
                    if all(item["finite"] and item["repeat_bitwise"] for item in exports):
                        row["split4_addition_order_reachability"] = {
                            tag: bf16_order_reachability(torch.stack(partials, dim=1), value)
                            for tag, value in (("plow", plow), ("reference", reference), ("repeat", repeat))}
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
        json.dump(dict(scope="captured block o_proj conditioned on native inputs; not serving or full-block qualification",
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
    parser.add_argument("--live-prefix", action="store_true", help="compare live activation prefixes in expert dumps")
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quant128", action="store_true")
    modes.add_argument("--block-oproj", action="store_true")
    parser.add_argument("--oproj-diagnostic", action="store_true")
    modes.add_argument("--block-shared", action="store_true")
    modes.add_argument("--export-shared", action="store_true")
    modes.add_argument("--block-norm", action="store_true")
    modes.add_argument("--block-routed", action="store_true")
    modes.add_argument("--block-router", action="store_true")
    modes.add_argument("--check-routed-ab", action="store_true")
    modes.add_argument("--export-qkva", action="store_true")
    modes.add_argument("--export-qb", action="store_true")
    modes.add_argument("--check-qb", action="store_true")
    modes.add_argument("--export-mla", action="store_true")
    modes.add_argument("--check-mla", action="store_true")
    modes.add_argument("--check-mla-weights", action="store_true")
    modes.add_argument("--block-mla", action="store_true")
    modes.add_argument("--block-mla-bmm", action="store_true")
    modes.add_argument("--block-qanorm", action="store_true")
    parser.add_argument("--native-rmsnorm", type=Path)
    parser.add_argument("--qanorm-sweep", action="store_true")
    modes.add_argument("--block-attention", action="store_true")
    modes.add_argument("--block-rope", action="store_true")
    modes.add_argument("--export-indexer", action="store_true")
    modes.add_argument("--export-indexer-model", action="store_true")
    modes.add_argument("--replay-model-attention", action="store_true")
    modes.add_argument("--pack-model-block", action="store_true")
    modes.add_argument("--check-model-decode-boundaries", action="store_true")
    modes.add_argument("--replay-model-query", action="store_true")
    modes.add_argument("--block-indexer", action="store_true")
    modes.add_argument("--export-indexer-quant", action="store_true")
    modes.add_argument("--export-indexer-decode", action="store_true")
    modes.add_argument("--export-indexer-native", action="store_true")
    modes.add_argument("--export-indexer-model-native", action="store_true")
    modes.add_argument("--export-indexer-native-sweep", action="store_true")
    modes.add_argument("--export-indexer-headroom", action="store_true")
    modes.add_argument("--export-indexer-prefill", action="store_true")
    modes.add_argument("--export-indexer-selection", action="store_true")
    modes.add_argument("--export-indexer-model-selection", action="store_true")
    modes.add_argument("--pack-attention-split", action="store_true")
    modes.add_argument("--check-attention-addressing", action="store_true")
    parser.add_argument("--native-indexer-selection", type=Path)
    modes.add_argument("--check-indexer-prefill", action="store_true")
    modes.add_argument("--check-indexer-decode", action="store_true")
    modes.add_argument("--check-indexer-quant", action="store_true")
    modes.add_argument("--check-indexer-score", action="store_true")
    modes.add_argument("--check-indexer-cache-score", action="store_true")
    modes.add_argument("--check-indexer-wq", action="store_true")
    modes.add_argument("--check-rope", action="store_true")
    modes.add_argument("--check-qkva", action="store_true")
    modes.add_argument("--export-routed", action="store_true")
    modes.add_argument("--model-routed-stage1", action="store_true")
    modes.add_argument("--export-routed-stage1-partials", action="store_true")
    modes.add_argument("--check-routed-reachability", action="store_true")
    modes.add_argument("--check-mlp-combine", action="store_true")
    modes.add_argument("--check-block-rows", action="store_true")
    modes.add_argument("--export-routed-grouped", action="store_true")
    modes.add_argument("--export-routed-down-grouped", action="store_true")
    parser.add_argument("--grouped-repeat", type=int, default=1)
    parser.add_argument("--routed-down-isolate", action="store_true")
    parser.add_argument("--routed-stage1-diagnostic", action="store_true")
    parser.add_argument("--routed-w8a8", action="store_true")
    parser.add_argument("--router-reference", type=Path)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--qb-reference", type=Path, help="also verify strided MLA input provenance and copied raw RoPE output")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision-inventory", type=Path)
    parser.add_argument("--export-attention-ps", action="store_true")
    parser.add_argument("--export-attention-sweep", action="store_true")
    parser.add_argument("--selection-attention", action="store_true")
    parser.add_argument("--export-attention-split", action="store_true")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--indexer-layer", type=int, default=6)
    args = parser.parse_args()
    if args.check_block_rows:
        return check_block_rows(args)
    if args.router_reference is not None and not (args.block_routed and args.routed_down_isolate):
        parser.error("--router-reference requires --block-routed and --routed-down-isolate")
    if args.check_mlp_combine:
        return check_mlp_combine(args)
    if args.pack_model_block:
        return pack_model_block(args)
    if args.check_model_decode_boundaries:
        return check_model_decode_boundaries(args)
    if args.check_routed_reachability:
        if args.reference_json is None:
            parser.error("--check-routed-reachability requires --reference-json")
        return check_routed_reachability(args)
    if args.model_routed_stage1 or args.export_routed_stage1_partials:
        if args.reference_json is None or args.checkpoint is None:
            parser.error("--model-routed-stage1 requires --reference-json and --checkpoint")
        return model_routed_stage1(args)
    if args.selection_attention and not args.export_attention_sweep:
        parser.error("--selection-attention requires --export-attention-sweep")
    if args.export_attention_split and (not args.export_attention_sweep or args.selection_attention):
        parser.error("--export-attention-split requires attention sweep without selection replay")
    if args.native_indexer_selection is not None and not (args.export_indexer_selection
            or args.export_indexer_model_selection or args.replay_model_attention):
        parser.error("--native-indexer-selection requires a selection export or model attention replay")
    if args.export_attention_ps and not (args.block_attention or args.replay_model_attention):
        parser.error("--export-attention-ps requires --block-attention or --replay-model-attention")
    if args.routed_w8a8 and not (args.block_routed and args.routed_down_isolate):
        parser.error("--routed-w8a8 requires --block-routed and --routed-down-isolate")
    if args.routed_stage1_diagnostic and not args.routed_w8a8:
        parser.error("--routed-stage1-diagnostic requires --routed-w8a8")
    if args.check_routed_ab:
        return check_routed_ab(args)
    if args.pack_attention_split:
        if args.reference_json is None:
            parser.error("--pack-attention-split requires --reference-json")
        return pack_attention_split(args)
    if args.check_rope:
        return check_rope(args)
    if args.check_indexer_wq:
        return check_indexer_wq(args)
    if args.check_indexer_quant:
        return check_indexer_quant(args)
    if args.check_indexer_decode:
        return check_indexer_decode(args)
    if args.check_indexer_prefill:
        return check_indexer_prefill(args)
    if args.check_indexer_score:
        return check_indexer_score(args)
    if args.check_indexer_cache_score:
        return check_indexer_score(args, cache_score=True)
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
    if args.block_router:
        return compare_packet_router(args, __version__)
    if args.replay_model_query:
        from vllm._aiter_ops import rocm_aiter_ops
        return replay_model_query(args, rocm_aiter_ops, __version__)
    if args.replay_model_attention:
        return replay_model_attention(args, __version__)
    if args.check_attention_addressing:
        if args.reference_json is None:
            parser.error("--check-attention-addressing requires --reference-json")
        return check_attention_addressing(args, __version__)
    if args.export_indexer_selection:
        return export_indexer_selection(args, __version__)
    if args.export_indexer_model_selection:
        return export_indexer_model_selection(args, __version__)
    if args.export_indexer_prefill:
        return export_indexer_prefill(args, __version__)
    if args.export_indexer_native or args.export_indexer_native_sweep or args.export_indexer_headroom:
        return export_indexer_native(args, __version__,
            sweep=args.export_indexer_native_sweep or args.export_indexer_headroom,
            headroom=args.export_indexer_headroom)
    if args.export_indexer_model_native:
        return export_indexer_model_native(args, __version__)
    if args.export_indexer_decode:
        return export_indexer_decode(args, __version__)
    if args.block_rope:
        return compare_packet_rope(args, __version__)
    if args.export_attention_sweep:
        return export_attention_sweep(args, __version__)
    from vllm._aiter_ops import rocm_aiter_ops
    if args.export_indexer or args.export_indexer_model or args.block_indexer or args.export_indexer_quant:
        return export_indexer(args, rocm_aiter_ops, __version__)
    if args.check_mla_weights:
        return check_mla_weights(args, __version__)
    if args.block_qanorm:
        return compare_packet_qanorm(args, rocm_aiter_ops, __version__)
    if args.block_mla_bmm:
        return compare_packet_mla_bmm(args, rocm_aiter_ops, __version__)
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
