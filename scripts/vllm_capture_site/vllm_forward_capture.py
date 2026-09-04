"""Config-driven PyTorch module-boundary capture.

This is deliberately independent of vLLM model classes.  Selectors match
``named_modules()`` paths and describe how to extract a tensor from ordinary
forward inputs or outputs.  The hook is installed through ``sitecustomize`` so
spawned tensor-parallel workers receive it without patching vLLM.
"""

import hashlib
import json
import os
import re
import threading
from pathlib import Path


_installed = False


def _rank():
    for name in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def _descend(value, path):
    for key in path:
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(key, int):
            value = value[key]
        else:
            value = getattr(value, key)
    return value


def _extract(spec, args, kwargs, output):
    import torch

    if "first_tensor" in spec:
        for candidate in spec["first_tensor"]:
            try:
                value = _extract(candidate, args, kwargs, output)
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            if isinstance(value, torch.Tensor):
                return value
        return None
    if "add" in spec:
        values = [_extract(x, args, kwargs, output).float() for x in spec["add"]]
        value = values[0]
        for other in values[1:]:
            value = value + other
        if spec.get("round_bf16"):
            value = value.to(torch.bfloat16)
        return value
    source = spec.get("source", "output")
    if source == "output":
        value = output
    elif source == "args":
        value = args
    elif source == "kwargs":
        value = kwargs
    else:
        raise ValueError(f"unknown capture source {source!r}")
    return _descend(value, spec.get("path", []))


def install(config):
    global _installed
    if _installed:
        return

    import torch

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selectors = []
    for raw in config["selectors"]:
        item = dict(raw)
        item["regex"] = re.compile(item.pop("module_regex"))
        selectors.append(item)
    prompt_hash = config["prompt_sha256_u32le"]
    history_id = config.get("history_id", prompt_hash[:16])
    wanted_rank = config.get("rank", 0)
    rank = _rank()
    original = torch.nn.Module._call_impl
    names = {}
    largest = {}
    sequences = {}
    lock = threading.Lock()

    def capture(module_name, item, match, args, kwargs, output):
        if wanted_rank is not None and rank != wanted_rank:
            return
        value = _extract(item["extract"], args, kwargs, output)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            if item.get("on_missing", "error") == "skip":
                return
            raise TypeError(f"{module_name}: capture expression did not return a tensor")
        rows = int(value.shape[0]) if value.ndim else 1
        source_dtype = str(value.dtype).removeprefix("torch.")
        source_shape = list(value.shape)
        source_stride = list(value.stride())
        minimum_rows = int(item.get("minimum_rows", 1))
        if rows < minimum_rows:
            return
        if item.get("row", "last") == "last" and value.ndim:
            value = value[-1]
        elif item.get("row") not in (None, "all"):
            value = value[int(item["row"])]
        fields = match.groupdict()
        layer = fields.get("layer", item.get("layer"))
        semantic = item["semantic"].format(**fields)
        key = (semantic, layer, rank)
        with lock:
            if rows < largest.get(key, 0):
                return
            if rows > largest.get(key, 0):
                sequences[key] = 0
            largest[key] = rows
            sequence = sequences.get(key, 0)
            sequences[key] = sequence + 1
        array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
        retain = int(item.get("retain", 1))
        sample = sequence % retain
        stem = (
            f"{_safe(semantic)}.layer-{_safe(layer)}.rank-{rank}.sample-{sample}"
            if retain > 1
            else f"{_safe(semantic)}.layer-{_safe(layer)}.rank-{rank}"
        )
        data_path = output_dir / f"{stem}.f32"
        tmp_path = output_dir / f".{stem}.{os.getpid()}.tmp"
        array.tofile(tmp_path)
        os.replace(tmp_path, data_path)
        meta = {
            "schema": 1,
            "producer": "pytorch-forward-hook",
            "semantic": semantic,
            "layer": int(layer) if layer is not None and str(layer).isdigit() else layer,
            "module": module_name,
            "rank": rank,
            "prompt_sha256_u32le": prompt_hash,
            "history_id": history_id,
            "source_dtype": source_dtype,
            "stored_dtype": "float32",
            "source_shape": source_shape,
            "stored_shape": list(array.shape),
            "source_stride": source_stride,
            "forward_rows": rows,
            "call_sequence": sequence,
            "file": data_path.name,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        meta_tmp = output_dir / f".{stem}.{os.getpid()}.json.tmp"
        meta_tmp.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(meta_tmp, output_dir / f"{stem}.json")

    def wrapped(module, *args, **kwargs):
        # The first outer model call exposes the complete qualified module tree.
        # Caching by object identity keeps subsequent calls at one dict lookup.
        if id(module) not in names:
            for module_name, child in module.named_modules():
                names.setdefault(id(child), module_name)
        output = original(module, *args, **kwargs)
        module_name = names.get(id(module), "")
        for item in selectors:
            match = item["regex"].fullmatch(module_name)
            if match:
                capture(module_name, item, match, args, kwargs, output)
        return output

    torch.nn.Module._call_impl = wrapped
    _installed = True


def install_from_env():
    path = Path(os.environ["PLOW_VLLM_CAPTURE_CONFIG"])
    install(json.loads(path.read_text()))
