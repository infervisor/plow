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
    return {"scope": "cache addressing and write isolation; not numerical parity",
            "excluded_warmup_invocations": excluded, "invocations": results,
            "attention_invocations": attention_results}


def capture_config(output_dir, prompt_hash, layer=6, rank=0, retain=8, attention=False):
    if layer < 0 or retain < 1:
        raise ValueError("layer must be nonnegative and retain must be positive")
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
                    {"extract": attention_context["num_actual_tokens"], "equals": 1},
                ],
                "context": attention_context, "on_missing": "skip", "storage_dtype": "raw",
                "row": "all", "row_policy": "all", "retain": retain,
            })
    return {"output_dir": str(output_dir), "prompt_sha256_u32le": prompt_hash,
            "rank": rank, "selectors": [], "method_selectors": selectors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir")
    mode.add_argument("--audit", type=Path)
    parser.add_argument("--prompt-sha256-u32le")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--retain", type=int, default=8)
    parser.add_argument("--attention", action="store_true", help="also capture the one-token MLA boundary")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_capture(args.audit), indent=2))
        return
    if args.prompt_sha256_u32le is None:
        parser.error("--output-dir requires --prompt-sha256-u32le")
    print(json.dumps(capture_config(args.output_dir, args.prompt_sha256_u32le,
                                    args.layer, args.rank, args.retain, args.attention), indent=2))


if __name__ == "__main__":
    main()
