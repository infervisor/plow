"""Capture the pinned vLLM 0.29 ROCm indexer boundary for connected replay."""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct


def audit_capture(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "reference/manifest.json").read_text())
    if manifest["vllm_version"] != "0.29.0" or len(manifest["requests"]) != 1:
        raise ValueError("audit requires one request from vLLM 0.29.0")
    if manifest["invalid_cases"]:
        raise ValueError("reference contains invalid logits")
    request = manifest["requests"][0]
    prompt_length = len(request["prompt_token_ids"])
    groups = {}
    attention_groups = {}
    block_groups = {}
    for path in (directory / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        context_bytes = json.dumps(record["context"], sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(context_bytes).hexdigest() != record["context_sha256"]:
            raise ValueError("capture context hash mismatch")
        with (path.parent / record["file"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"tensor hash mismatch: {path}")
        if not record["semantic"].startswith("indexer."):
            if record["semantic"].startswith("attention."):
                attention_groups.setdefault(record["invocation_index"], {})[record["semantic"]] = record
            elif record["semantic"].startswith("block."):
                key = record["context"]["max_seq_len"]
                group = block_groups.setdefault(key, {})
                if record["semantic"] in group:
                    raise ValueError("duplicate block boundary at one context")
                group[record["semantic"]] = record
            continue
        key = record["invocation_index"]
        group = groups.setdefault(key, {})
        if record["semantic"] in group:
            raise ValueError("duplicate invocation semantic")
        group[record["semantic"]] = record

    def integers(record, count=None):
        dtype = record["source_dtype"]
        code = {"int32": "i", "int64": "q"}[dtype]
        with (directory / "tensors" / record["file"]).open("rb") as stream:
            data = stream.read() if count is None else stream.read(count * struct.calcsize(code))
        return [v[0] for v in struct.iter_unpack("<" + code, data)]

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    slots = []
    previous = None
    results = []
    excluded = []
    for invocation, group in sorted(groups.items()):
        records = list(group.values())
        require(len({r["context_sha256"] for r in records}) == 1, "mixed invocation context")
        context = records[0]["context"]
        if context["max_seq_len"] < prompt_length:
            excluded.append(invocation)
            continue
        require(all(r["rank"] == 0 and r["layer"] == 6 for r in records), "wrong rank/layer")
        require(all(r["prompt_sha256_u32le"] == request["prompt_sha256_u32le"]
                    for r in records), "wrong prompt hash")
        live = context["num_prefill_tokens"] + context["num_decode_tokens"]
        require((not slots and live == prompt_length and context["num_prefills"] == 1)
                or (slots and live == 1 and context["num_decodes"] == 1), "unexpected history")
        inputs_bound = "indexer.input.hidden" in group
        if inputs_bound:
            require(group["indexer.input.hidden"]["sha256"] == group["indexer.hidden"]["sha256"],
                    "outer/inner hidden mismatch")
            for name, width in (("hidden", 6144), ("qr", 2048)):
                record = group[f"indexer.input.{name}"]
                require(record["source_dtype"] == "bfloat16"
                        and record["source_shape"] == [live, width], "projection input geometry mismatch")
            require(integers(group["indexer.input.positions"]) == list(range(len(slots), len(slots) + live)),
                    "projection positions mismatch")
        new_slots = integers(group["indexer.slot_mapping"])
        require(len(new_slots) == live, "slot count mismatch")
        before, after = group["indexer.cache.before"], group["indexer.cache.after"]
        shape = before["source_shape"]
        require(shape == after["source_shape"] and shape[1:] == [16, 132], "cache layout mismatch")
        require(before["source_dtype"] == after["source_dtype"] == "uint8", "cache dtype mismatch")
        require(all(0 <= slot < shape[0] * 16 for slot in new_slots), "slot out of bounds")
        if previous is not None:
            require(previous == before["sha256"], "cache history discontinuity")
        slots.extend(new_slots)
        require(len(set(slots)) == len(slots), "live slots alias")
        require(len(slots) == context["max_seq_len"], "sequence length mismatch")
        if context["num_decodes"]:
            table = integers(group["indexer.decode.block_table"])
            require(integers(group["indexer.decode.seq_lens"]) == [len(slots)], "decode length mismatch")
            require(all(table[i // 16] * 16 + i % 16 == slot
                        for i, slot in enumerate(slots)), "page table does not map history")
        changed_blocks = []
        allowed_blocks = {slot // 16 for slot in new_slots}
        with (directory / "tensors" / before["file"]).open("rb") as left, \
             (directory / "tensors" / after["file"]).open("rb") as right, \
             mmap.mmap(left.fileno(), 0, access=mmap.ACCESS_READ) as a, \
             mmap.mmap(right.fileno(), 0, access=mmap.ACCESS_READ) as b:
            require(len(a) == len(b) == shape[0] * 2112, "cache byte count mismatch")
            for block in range(shape[0]):
                start = block * 2112
                if a[start:start + 2112] != b[start:start + 2112]:
                    require(block in allowed_blocks, "cache changed outside inserted blocks")
                    changed_blocks.append(block)
                    positions = {slot % 16 for slot in new_slots if slot // 16 == block}
                    for pos in set(range(16)) - positions:
                        offsets = [start + head * 256 + pos * 16 for head in range(8)]
                        require(all(a[o:o + 16] == b[o:o + 16] for o in offsets)
                                and a[start + 2048 + pos * 4:start + 2052 + pos * 4]
                                == b[start + 2048 + pos * 4:start + 2052 + pos * 4],
                                "cache changed an uninserted token")
        previous = after["sha256"]
        results.append({"invocation": invocation, "live": live, "length": len(slots),
                        "changed_blocks": len(changed_blocks), "projection_inputs_bound": inputs_bound})
    require(len(results) == len(request["generated_token_ids"]), "incomplete request history")
    attention_results = []
    for invocation, group in sorted(attention_groups.items()):
        records = list(group.values())
        require(len({r["context_sha256"] for r in records}) == 1, "mixed attention context")
        context = records[0]["context"]
        length = context["max_seq_len"]
        if length < prompt_length:
            continue
        require(context["num_actual_tokens"] == 1 and context["topk_tokens"] == 2048,
                "unexpected attention geometry")
        require(all(r["rank"] == 0 and r["layer"] == 6 for r in records), "wrong attention rank/layer")
        for name, shape in (("query", [1, 16, 576]), ("output", [1, 8, 512])):
            record = group["attention." + name]
            require(record["source_dtype"] == "bfloat16" and record["source_shape"] == shape,
                    "attention tensor geometry mismatch")
            data = (directory / "tensors" / record["file"]).read_bytes()
            require(all((v[0] & 0x7f80) != 0x7f80 for v in struct.iter_unpack("<H", data)),
                    "nonfinite attention tensor")
        cache = group["attention.cache"]
        block_size = context["block_size"]
        require(cache["source_dtype"] == "bfloat16"
                and cache["source_shape"][1:] == [block_size, 576], "attention cache geometry mismatch")
        require(integers(group["attention.req_id_per_token"]) == [0], "unexpected attention request")
        require(integers(group["attention.qo_indptr"], 2) == [0, 1], "unexpected attention query pointers")
        count = min(length, 2048)
        require(integers(group["attention.paged_kv_indptr"], 2) == [0, count], "unexpected attention KV pointers")
        selected = integers(group["attention.selected"], count)
        indexer_group = next(g for g in groups.values()
                             if next(iter(g.values()))["context"]["max_seq_len"] == length)
        require(selected == integers(indexer_group["indexer.selected"], count), "attention/indexer selection mismatch")
        table = integers(group["attention.block_table"])
        mapped = integers(group["attention.paged_kv_indices"], count)
        require(all(0 <= token < length for token in selected), "attention selection out of range")
        require(mapped == [table[token // block_size] * block_size + token % block_size for token in selected],
                "attention page mapping mismatch")
        require(all(0 <= token < cache["source_shape"][0] * block_size for token in mapped),
                "attention cache index out of range")
        attention_results.append(dict(invocation=invocation, length=length, selected=count,
                                      mapping_matches_indexer=True))
    if attention_groups:
        require(sorted(r["length"] for r in attention_results) ==
                list(range(prompt_length + 1, prompt_length + len(request["generated_token_ids"]))),
                "incomplete attention capture")
    block_results = []
    for length, group in sorted(block_groups.items()):
        if length < prompt_length:
            continue
        names = {"positions", "input.hidden", "input.residual", "output.hidden", "output.residual",
                 "xn", "x", "attn", "xn2", "xmid", "mlp"}
        if "block.qb" in group:
            names.add("qb")
        require(set(group) == {"block." + name for name in names}, "incomplete block boundaries")
        require(len({r["context_sha256"] for r in group.values()}) == 1, "mixed block contexts")
        require(all(r["rank"] == 0 and r["layer"] == 6
                    and r["prompt_sha256_u32le"] == request["prompt_sha256_u32le"]
                    for r in group.values()), "wrong block identity")
        require(all(r["source_shape"] == [1, 2048 if name == "block.qb" else 6144] and r["source_dtype"] == "bfloat16"
                    for name, r in group.items() if name != "block.positions"), "wrong block geometry")
        require(integers(group["block.positions"]) == [length - 1], "wrong block position")
        indexers = [g for g in groups.values()
                    if next(iter(g.values()))["context"]["max_seq_len"] == length
                    and "indexer.input.hidden" in g]
        require(len(indexers) == 1, "ambiguous block/indexer binding")
        for left, right in ((group["block.xn"], indexers[0]["indexer.input.hidden"]),
                            (group["block.mlp"], group["block.output.hidden"]),
                            (group["block.xmid"], group["block.output.residual"])):
            require(left["sha256"] == right["sha256"], "block stage identity differs")
        block_results.append(dict(length=length, boundaries=len(group), indexer_input_bound=True))
    if block_groups:
        require([r["length"] for r in block_results] == [r["length"] for r in results if r["live"] == 1],
                "incomplete block decode history")
    return {"scope": "cache addressing and write isolation; not numerical parity",
            "excluded_warmup_invocations": excluded, "invocations": results,
            "attention_invocations": attention_results, "block_invocations": block_results}


def check_batch_pages(tables, lengths, slots, selected, mapped, cache_blocks, block_size=16):
    batch = len(lengths)
    if (batch < 1 or len(tables) != batch or len(slots) != batch or len(selected) != batch
            or len(mapped) != batch or block_size < 1):
        raise ValueError("batch page geometry mismatch")
    occupied = set()
    for table, length, slot, chosen, physical in zip(tables, lengths, slots, selected, mapped):
        pages = table[:(length + block_size - 1) // block_size]
        if (length < 1 or len(pages) * block_size < length
                or len(set(pages)) != len(pages) or occupied.intersection(pages)
                or any(page < 0 or page >= cache_blocks for page in pages)):
            raise ValueError("invalid or aliased per-request cache pages")
        occupied.update(pages)
        if slot != table[(length - 1) // block_size] * block_size + (length - 1) % block_size:
            raise ValueError("inserted slot does not map request position")
        if (len(chosen) != min(2048, length) or len(set(chosen)) != len(chosen)
                or any(token < 0 or token >= length for token in chosen)
                or physical != [table[token // block_size] * block_size + token % block_size for token in chosen]):
            raise ValueError("selected tokens do not map this request")


def audit_batch_cache(directory):
    directory = Path(directory)
    boundary = audit_batch_boundaries(directory)
    batch = boundary["batch"]
    lengths = [row["length"] for row in boundary["block_invocations"]]
    groups = {length: {} for length in lengths}
    for path in (directory / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        length = record["context"]["max_seq_len"]
        if length not in groups:
            continue
        name = record["semantic"]
        if name in groups[length]:
            raise ValueError("ambiguous batched capture semantic")
        groups[length][name] = record
    results = []
    previous = None
    for length, records in sorted(groups.items()):
        hashes = {}
        def load(name, dtype, shape=None, count=None):
            record = records[name]
            context = record["context"]
            context_bytes = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
            if (record["rank"] != 0 or record["layer"] != 6 or record["source_dtype"] != dtype
                    or (shape is not None and record["source_shape"] != shape)
                    or hashlib.sha256(context_bytes).hexdigest() != record["context_sha256"]):
                raise ValueError("batch cache identity/geometry mismatch")
            if ((name.startswith("indexer.") and (context["num_decodes"] != batch
                    or context["num_decode_tokens"] != batch or context["num_prefill_tokens"] != 0))
                    or (name.startswith("attention.") and (context["num_actual_tokens"] != batch
                    or context["block_size"] != 16 or context["topk_tokens"] != 2048))
                    or record["prompt_sha256_u32le"] != records["block.xn"]["prompt_sha256_u32le"]):
                raise ValueError("batch cache context differs from block")
            path = directory / "tensors" / record["file"]
            elements = 1
            for dim in record["source_shape"]:
                elements *= dim
            if path.stat().st_size != elements * {"int32": 4, "int64": 8, "uint8": 1, "bfloat16": 2}[dtype]:
                raise ValueError("batch cache byte size mismatch")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record["sha256"]:
                raise ValueError("batch cache tensor hash mismatch")
            hashes[name] = digest
            if dtype not in ("int32", "int64"):
                return record
            code = "i" if dtype == "int32" else "q"
            with path.open("rb") as stream:
                data = stream.read() if count is None else stream.read(count * struct.calcsize(code))
            return [v[0] for v in struct.iter_unpack("<" + code, data)]
        hidden = load("indexer.input.hidden", "bfloat16", [batch, 6144])
        if hidden["sha256"] != records["block.xn"]["sha256"]:
            raise ValueError("batch indexer input differs from block norm")
        positions = load("indexer.input.positions", "int64", [batch])
        seq = load("indexer.decode.seq_lens", "int32", [batch, 1])
        slots = load("indexer.slot_mapping", "int64", [batch])
        if positions != [length - 1] * batch or seq != [length] * batch:
            raise ValueError("batch cache positions/lengths mismatch")
        chosen = load("indexer.selected", "int32", count=batch * 2048)
        if chosen != load("attention.selected", "int32", count=batch * 2048):
            raise ValueError("batch attention/indexer selected tokens differ")
        if (load("attention.req_id_per_token", "int32", [batch]) != list(range(batch))
                or load("attention.qo_indptr", "int32", [batch + 1]) != list(range(batch + 1))
                or load("attention.paged_kv_indptr", "int32", [batch + 1]) != [2048 * i for i in range(batch + 1)]):
            raise ValueError("batch attention request pointers mismatch")
        mapped = load("attention.paged_kv_indices", "int32", [batch * 2048])
        before = load("indexer.cache.before", "uint8")
        after = load("indexer.cache.after", "uint8", before["source_shape"])
        cache = load("attention.cache", "bfloat16")
        if (before["source_shape"][1:] != [16, 132] or cache["source_shape"][1:] != [16, 576]
                or (previous is not None and previous != before["sha256"])):
            raise ValueError("batch cache layout/history mismatch")
        previous = after["sha256"]
        for name, blocks, physical in (("attention.block_table", cache["source_shape"][0], mapped),
                                      ("indexer.decode.block_table", before["source_shape"][0], None)):
            shape = records[name]["source_shape"]
            if len(shape) != 2 or shape[0] != batch:
                raise ValueError("batch page table geometry mismatch")
            values = load(name, "int32")
            tables = [values[row * shape[1]:(row + 1) * shape[1]] for row in range(batch)]
            selected = [chosen[row * 2048:(row + 1) * 2048] for row in range(batch)]
            if physical is None:
                physical_rows = [[table[t // 16] * 16 + t % 16 for t in row]
                                 for table, row in zip(tables, selected)]
                insert = slots
            else:
                physical_rows = [physical[row * 2048:(row + 1) * 2048] for row in range(batch)]
                insert = [table[(length - 1) // 16] * 16 + (length - 1) % 16 for table in tables]
            check_batch_pages(tables, seq, insert, selected, physical_rows, blocks)
        results.append(dict(length=length, batch=batch, tensor_sha256=hashes))
    return dict(scope="C8-style homogeneous decode cache page mappings and captured history continuity; not prefill history, write-isolation or numerical qualification",
        precision_qualified=False, batch=batch, manifest_sha256=boundary["manifest_sha256"],
        block_invocations=boundary["block_invocations"], cache_invocations=results)


def audit_batch_boundaries(directory):
    directory = Path(directory)
    manifest_path = directory / "reference/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    requests = manifest["requests"]
    batch = manifest.get("request_batch_size")
    if (manifest["vllm_version"] != "0.29.0" or manifest["invalid_cases"]
            or not isinstance(batch, int) or batch < 2 or len(requests) != batch
            or manifest.get("max_num_seqs") != batch or manifest.get("enable_prefix_caching") is not False):
        raise ValueError("requires one complete concurrent batch with prefix caching disabled")
    prompt = requests[0]["prompt_token_ids"]
    digest = hashlib.sha256(struct.pack(f"<{len(prompt)}I", *prompt)).hexdigest()
    steps = len(requests[0]["generated_token_ids"])
    if (not prompt or steps < 2 or any(r["prompt_token_ids"] != prompt
            or r["prompt_sha256_u32le"] != digest or len(r["generated_token_ids"]) != steps
            for r in requests) or len({r["request_id"] for r in requests}) != batch):
        raise ValueError("requires distinct requests with identical prompts and complete decode histories")
    groups = {}
    for path in (directory / "tensors").glob("*.json"):
        record = json.loads(path.read_text())
        if not record["semantic"].startswith("block."):
            continue
        context = record["context"]
        length = context["max_seq_len"]
        if length <= len(prompt):
            continue
        if (context["num_decodes"] != batch or context["num_decode_tokens"] != batch
                or context["num_prefills"] != 0 or context["num_prefill_tokens"] != 0
                or record["rank"] != 0 or record["layer"] != 6
                or record["prompt_sha256_u32le"] != digest):
            raise ValueError("wrong live decode batch or block identity")
        context_bytes = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(context_bytes).hexdigest() != record["context_sha256"]:
            raise ValueError("batch context hash mismatch")
        data = (directory / "tensors" / record["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("batch tensor hash mismatch")
        name = record["semantic"].removeprefix("block.")
        if name == "positions":
            if (record["source_dtype"] != "int64" or record["source_shape"] != [batch]
                    or len(data) != batch * 8
                    or list(struct.unpack(f"<{batch}q", data)) != [length - 1] * batch):
                raise ValueError("wrong batch decode positions")
        else:
            width = 2048 if name == "qb" else 6144
            if (record["source_dtype"] != "bfloat16" or record["source_shape"] != [batch, width]
                    or len(data) != batch * width * 2
                    or any((v[0] & 0x7f80) == 0x7f80 for v in struct.iter_unpack("<H", data))):
                raise ValueError("wrong/nonfinite batch block geometry")
        group = groups.setdefault(length, {})
        if name in group:
            raise ValueError("duplicate batch block boundary")
        group[name] = record
    names = {"positions", "input.hidden", "input.residual", "output.hidden", "output.residual",
             "xn", "x", "attn", "qb", "xn2", "xmid", "mlp"}
    if sorted(groups) != list(range(len(prompt) + 1, len(prompt) + steps)):
        raise ValueError("incomplete actual-batch decode history")
    for group in groups.values():
        if set(group) != names or len({r["context_sha256"] for r in group.values()}) != 1:
            raise ValueError("incomplete/mixed batch block boundaries")
        if (group["mlp"]["sha256"] != group["output.hidden"]["sha256"]
                or group["xmid"]["sha256"] != group["output.residual"]["sha256"]):
            raise ValueError("batch block output identity mismatch")
    return dict(scope="actual homogeneous decode batch, block shapes/positions/hashes; NOT cache-addressing or numerical qualification",
        precision_qualified=False, batch=batch,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        block_invocations=[dict(length=length, boundaries=len(group),
            tensor_sha256={name: r["sha256"] for name, r in group.items()}) for length, group in sorted(groups.items())])


def capture_config(output_dir, prompt_hash, layer=6, rank=0, retain=8, attention=False, block=False,
                   decode_batch=1, block_prefill_tokens=0, dense=False):
    if layer < 0 or retain < 1 or decode_batch < 1 or block_prefill_tokens < 0:
        raise ValueError("layer must be nonnegative and retain/decode batch must be positive")
    if len(prompt_hash) != 64 or any(c not in "0123456789abcdef" for c in prompt_hash):
        raise ValueError("prompt hash must be lowercase SHA256 of uint32 little-endian tokens")
    prefix = f"model.layers.{layer}.self_attn.indexer.k_cache"

    def metadata(*path):
        return {"call": "vllm.forward_context.get_forward_context",
                "path": ["attn_metadata", prefix, *path]}

    when = [
        {"extract": {"source": "module", "path": ["k_cache", "prefix"]}, "equals": prefix},
        {"extract": {"call": "vllm.forward_context.get_forward_context",
                     "path": ["attn_metadata"]}, "not_none": True},
    ]
    context = {field: metadata(field) for field in (
        "num_decodes", "num_decode_tokens", "num_prefills", "num_prefill_tokens", "max_seq_len"
    )}
    context["cache_prefix"] = {"source": "module", "path": ["k_cache", "prefix"]}
    selectors = []

    def add(name, extract, phase="before", extra_when=(),
            target="vllm.model_executor.layers.sparse_attn_indexer.SparseAttnIndexer.forward_hip"):
        selectors.append({
            "target": target,
            "semantic": name, "layer": layer, "phase": phase, "extract": extract,
            "when": [*when, *extra_when], "context": context,
            "storage_dtype": "raw", "row": "all", "row_policy": "all", "retain": retain,
        })

    for index, name in enumerate(("hidden", "qr", "positions")):
        add(f"indexer.input.{name}", {"source": "args", "path": [index]},
            target="vllm.model_executor.models.deepseek_v2.Indexer.forward")
    for index, name in enumerate(("hidden", "q_fp8", "key_bf16", "weights")):
        add(f"indexer.{name}", {"source": "args", "path": [index]})
    for phase in ("before", "after"):
        add(f"indexer.cache.{phase}", {"source": "module", "path": ["k_cache", "kv_cache"]}, phase)
    add("indexer.selected", {"source": "output"}, "after")
    for field in ("slot_mapping", "seq_lens"):
        add(f"indexer.{field}", metadata(field))
    for field in ("block_table", "seq_lens", "decode_lens", "schedule_metadata"):
        add(f"indexer.decode.{field}", metadata("decode", field), extra_when=[
            {"extract": metadata("decode"), "not_none": True},
            {"extract": metadata("decode", field), "not_none": True},
        ])
    if attention:
        attention_context = {field: {"source": "args", "path": [3, field]}
                             for field in ("num_actual_tokens", "max_seq_len", "block_size", "topk_tokens")}
        attention_context["layer_name"] = {"source": "args", "path": [0, "layer_name"]}
        attention_context["scale"] = {"source": "module", "path": ["scale"]}
        fields = [("query", {"source": "args", "path": [1]}, "before"),
                  ("cache", {"source": "args", "path": [2]}, "before"),
                  ("selected", {"source": "module", "path": ["topk_indices_buffer"]}, "before"),
                  ("output", {"source": "output"}, "after")]
        metadata_fields = ("block_table", "req_id_per_token", "qo_indptr", "paged_kv_indptr",
                           "paged_kv_indices", "paged_kv_last_page_len", "work_meta_data", "work_indptr",
                           "work_info_set", "reduce_indptr", "reduce_final_map", "reduce_partial_map")
        fields.extend((name, {"source": "args", "path": [3, name]}, "before")
                      for name in metadata_fields)
        for name, extract, phase in fields:
            selectors.append({
                "target": "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseImpl._forward_mla",
                "semantic": "attention." + name, "layer": layer, "phase": phase, "extract": extract,
                "when": [
                    {"extract": attention_context["layer_name"], "equals": f"model.layers.{layer}.self_attn.attn"},
                    {"extract": attention_context["num_actual_tokens"], "equals": decode_batch},
                ],
                "context": attention_context, "on_missing": "skip", "storage_dtype": "raw",
                "row": "all", "row_policy": "all", "retain": retain,
            })
    module_selectors = []
    if block:
        live_tokens = block_prefill_tokens or decode_batch
        live_field = "num_prefill_tokens" if block_prefill_tokens else "num_decode_tokens"
        idle_field = "num_decode_tokens" if block_prefill_tokens else "num_prefill_tokens"
        live_when = [
            {"extract": {"call": "vllm.forward_context.get_forward_context",
                     "path": ["attn_metadata"]}, "not_none": True},
            {"extract": metadata(live_field), "equals": live_tokens},
            {"extract": metadata(idle_field), "equals": 0},
        ]
        block_context = {field: metadata(field) for field in (
            "num_decodes", "num_decode_tokens", "num_prefills", "num_prefill_tokens", "max_seq_len")}
        common = dict(layer=layer, context=block_context, storage_dtype="raw",
                      row="all", row_policy="all", retain=retain)
        for name, extract, phase in (
            ("positions", {"source": "args", "path": [0]}, "before"),
            ("input.hidden", {"source": "args", "path": [1]}, "before"),
            ("input.residual", {"source": "args", "path": [2]}, "before"),
            ("output.hidden", {"source": "output", "path": [0]}, "after"),
            ("output.residual", {"source": "output", "path": [1]}, "after"),
        ):
            optional = block_prefill_tokens and layer == 0 and name == "input.residual"
            selectors.append(dict(common, **({"on_missing": "skip"} if optional else {}),
                target="vllm.model_executor.models.deepseek_v2.DeepseekV2DecoderLayer.forward",
                semantic="block." + name, extract=extract, phase=phase,
                when=[{"extract": {"source": "module", "path": ["layer_idx"]}, "equals": layer},
                      *live_when]))
        for module, name, extract in (
            ("input_layernorm", "xn", {"source": "output", "path": [0]}),
            ("input_layernorm", "x", {"source": "output", "path": [1]}),
            ("self_attn", "attn", {"source": "output"}),
            (r"self_attn\.q_b_proj", "qb", {"source": "output", "path": [0]}),
            ("post_attention_layernorm", "xn2", {"source": "output", "path": [0]}),
            ("post_attention_layernorm", "xmid", {"source": "output", "path": [1]}),
            ("mlp", "mlp", {"source": "output"}),
        ):
            module_selectors.append(dict(common,
                module_regex=rf"model\.layers\.{layer}\.{module}", semantic="block." + name,
                extract=extract, when=live_when))
        if dense:
            for module, name, extract in (
                (r"mlp\.gate_up_proj", "mlp.gate_up", {"source": "output", "path": [0]}),
                (r"mlp\.act_fn", "mlp.activation", {"source": "output"}),
            ):
                module_selectors.append(dict(common,
                    module_regex=rf"model\.layers\.{layer}\.{module}", semantic="block." + name,
                    extract=extract, when=live_when))
    return {"output_dir": str(output_dir), "prompt_sha256_u32le": prompt_hash,
            "rank": rank, "selectors": module_selectors, "method_selectors": selectors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir")
    mode.add_argument("--audit", type=Path)
    mode.add_argument("--audit-batch-boundaries", type=Path)
    mode.add_argument("--audit-batch-cache", type=Path)
    parser.add_argument("--prompt-sha256-u32le")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--retain", type=int, default=8)
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--block-prefill-tokens", type=int, default=0)
    parser.add_argument("--attention", action="store_true", help="also capture the selected decode-batch MLA boundary")
    parser.add_argument("--block", action="store_true", help="also capture selected decode-batch layer residuals and shared stages")
    parser.add_argument("--dense", action="store_true", help="with --block, capture dense gate/up and activation outputs")
    args = parser.parse_args()
    if args.audit_batch_cache:
        print(json.dumps(audit_batch_cache(args.audit_batch_cache), indent=2))
        return
    if args.audit_batch_boundaries:
        print(json.dumps(audit_batch_boundaries(args.audit_batch_boundaries), indent=2))
        return
    if args.audit:
        print(json.dumps(audit_capture(args.audit), indent=2))
        return
    if args.prompt_sha256_u32le is None:
        parser.error("--output-dir requires --prompt-sha256-u32le")
    print(json.dumps(capture_config(args.output_dir, args.prompt_sha256_u32le,
                                    args.layer, args.rank, args.retain, args.attention, args.block,
                                    args.decode_batch, args.block_prefill_tokens, args.dense), indent=2))


if __name__ == "__main__":
    main()
