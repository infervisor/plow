#!/usr/bin/env python3
import json
import os
from pathlib import Path
import tempfile

import torch


class Dummy(torch.nn.Module):
    def boundary(self, q, output):
        output.copy_(q + torch.ones_like(q))


def main():
    from vllm_forward_capture import install

    with tempfile.TemporaryDirectory() as tmp:
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
            ],
        }
        install(config)
        q = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        output = torch.empty_like(q)
        Dummy().boundary(q, output)
        for semantic, expected in (("q", q), ("attention.output", output)):
            meta = json.loads(next(Path(tmp).glob(f"{semantic}.*.json")).read_text())
            raw = (Path(tmp) / meta["file"]).read_bytes()
            assert raw == expected.view(torch.uint16).cpu().numpy().astype("<u2").tobytes()
            assert meta["stored_dtype"] == "bf16"


if __name__ == "__main__":
    main()
