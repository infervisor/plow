#!/usr/bin/env python3
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch


_context = {"metadata": None}


def forward_context():
    return _context


class Dummy(torch.nn.Module):
    def boundary(self, q, output):
        output.copy_(q + torch.ones_like(q))

    def raw_boundary(self, value):
        return value

    def history_boundary(self, value):
        return value

    def cache_boundary(self, value):
        self.cache.add_(1)
        return value


def main():
    from vllm_forward_capture import install

    with tempfile.TemporaryDirectory() as tmp:
        raw_cases = [
            torch.tensor([[0, 1, 127], [128, 254, 255]], dtype=torch.uint8),
            torch.tensor([[0, -1], [2147483647, -2147483648]], dtype=torch.int32),
            torch.tensor([0, -1, 2**40], dtype=torch.int64),
            torch.tensor([0, 128, 127, 255], dtype=torch.uint8).view(torch.float8_e4m3fnuz),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16).T,
            torch.tensor(1.5, dtype=torch.bfloat16),
        ]
        config = {
            "output_dir": tmp,
            "prompt_sha256_u32le": "history",
            "rank": 0,
            "selectors": [],
            "method_selectors": [
                {"target": "__main__.Dummy.boundary", "semantic": "q", "layer": 0,
                 "phase": "before", "extract": {"source": "args", "path": [0]},
                 "storage_dtype": "bf16", "row": "all"},
                {"target": "__main__.Dummy.boundary", "semantic": "attention.output", "layer": 0,
                 "phase": "after", "extract": {"source": "args", "path": [1]},
                 "storage_dtype": "bf16", "row": "all"},
                {"target": "__main__.Dummy.history_boundary", "semantic": "history", "layer": 0,
                 "phase": "before", "extract": {"source": "args", "path": [0]},
                 "storage_dtype": "raw", "row": "all", "row_policy": "all", "retain": 3},
                {"target": "__main__.Dummy.history_boundary", "semantic": "largest", "layer": 0,
                 "phase": "before", "extract": {"source": "args", "path": [0]},
                 "storage_dtype": "raw", "row": "all"},
            ],
        }
        config["method_selectors"].extend(
            {"target": "__main__.Dummy.raw_boundary", "semantic": f"raw-{i}", "layer": 0,
             "phase": "before", "extract": {"source": "args", "path": [0]},
             "storage_dtype": "raw", "row": "all", "call_index": i}
            for i in range(len(raw_cases))
        )
        for semantic, phase, extract in (
            ("cache.before", "before", {"source": "module", "path": ["cache"]}),
            ("cache.after", "after", {"source": "module", "path": ["cache"]}),
            ("cache.map", "before", {"call": "__main__.forward_context",
                                      "path": ["metadata", "block_table"]}),
        ):
            config["method_selectors"].append({
                "target": "__main__.Dummy.cache_boundary", "semantic": semantic,
                "layer": 6, "phase": phase, "extract": extract,
                "storage_dtype": "raw", "row": "all", "row_policy": "all", "retain": 3,
                "when": [
                    {"extract": {"source": "module", "path": ["prefix"]}, "equals": "layer6"},
                    {"extract": {"call": "__main__.forward_context", "path": ["metadata"]},
                     "not_none": True},
                ],
                "context": {"prefix": {"source": "module", "path": ["prefix"]}},
            })
        install(config)
        q = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        output = torch.empty_like(q)
        Dummy().boundary(q, output)
        for semantic, expected in (("q", q), ("attention.output", output)):
            meta = json.loads(next(Path(tmp).glob(f"{semantic}.*.json")).read_text())
            raw = (Path(tmp) / meta["file"]).read_bytes()
            assert raw == expected.view(torch.uint16).cpu().numpy().astype("<u2").tobytes()
            assert meta["stored_dtype"] == "bf16"
        dummy = Dummy()
        for i, expected in enumerate(raw_cases):
            assert dummy.raw_boundary(expected) is expected
            meta = json.loads(next(Path(tmp).glob(f"raw-{i}.*.json")).read_text())
            raw = (Path(tmp) / meta["file"]).read_bytes()
            assert raw == expected.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            assert meta["stored_dtype"] == "raw"
            assert meta["source_dtype"] == str(expected.dtype).removeprefix("torch.")
            assert meta["source_shape"] == list(expected.shape)
            assert meta["source_stride"] == list(expected.stride())
            assert meta["stored_shape"] == [expected.numel() * expected.element_size()]
        for sequence, rows in enumerate((8, 1, 2, 1)):
            value = torch.full((rows, 4), sequence, dtype=torch.int32)
            assert dummy.history_boundary(value) is value
            meta_path = Path(tmp) / f"history.layer-0.rank-0.sample-{sequence % 3}.json"
            meta = json.loads(meta_path.read_text())
            assert meta["call_sequence"] == sequence
            assert meta["forward_rows"] == rows
            assert meta["row_policy"] == "all"
            assert (Path(tmp) / meta["file"]).read_bytes() == value.view(torch.uint8).numpy().tobytes()
        assert len(list(Path(tmp).glob("history.*.json"))) == 3
        largest = json.loads((Path(tmp) / "largest.layer-0.rank-0.json").read_text())
        assert largest["forward_rows"] == 8
        assert largest["call_sequence"] == 0
        assert largest["row_policy"] == "largest"
        dummy.prefix = "layer6"
        dummy.cache = torch.zeros((2, 16, 132), dtype=torch.uint8)
        dummy.cache_boundary(q)
        assert not list(Path(tmp).glob("cache.*.json"))
        _context["metadata"] = {"block_table": torch.tensor([[1, 0]], dtype=torch.int32)}
        dummy.prefix = "layer5"
        dummy.cache_boundary(q)
        assert not list(Path(tmp).glob("cache.*.json"))
        dummy.prefix = "layer6"
        expected_before = dummy.cache.clone()
        dummy.cache_boundary(q)
        context_hashes = set()
        for semantic, phase, expected in (
            ("cache.before", "before", expected_before),
            ("cache.after", "after", dummy.cache),
            ("cache.map", "before", _context["metadata"]["block_table"]),
        ):
            meta = json.loads((Path(tmp) / f"{semantic}.layer-6.rank-0.sample-0.json").read_text())
            assert meta["invocation_index"] == 2
            assert meta["call_sequence"] == 0
            assert meta["phase"] == phase
            assert meta["context"] == {"prefix": "layer6"}
            context_hashes.add(meta["context_sha256"])
            assert (Path(tmp) / meta["file"]).read_bytes() == expected.view(torch.uint8).numpy().tobytes()
        assert len(context_hashes) == 1
        with patch.object(torch.distributed, "is_available", return_value=True), \
             patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_rank", return_value=1):
            dummy.cache_boundary(q)
        assert not list(Path(tmp).glob("*.rank-1.*"))
        assert len(list(Path(tmp).glob("cache.before.*.json"))) == 1


if __name__ == "__main__":
    main()
