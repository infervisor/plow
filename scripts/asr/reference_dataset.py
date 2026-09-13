#!/usr/bin/env python3
"""Run the pinned PyTorch BF16 reference with explicit packed encoder windows."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel
from reference import packed_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    for row in rows:
        if "excluded_reason" not in row and hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest() != row["audio_sha256"]:
            raise ValueError(f"audio checksum mismatch: {row['id']}")
    torch.set_num_threads(8)
    model = Qwen3ASRModel.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager", max_new_tokens=1024, local_files_only=True)
    packed_windows(model.model.thinker.audio_tower)
    print(json.dumps({"kind": "metadata", "device": "cpu", "dtype": "bfloat16", "packed_windows": True,
                      "max_new_tokens": 1024, "language_policy": "automatic",
                      "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      "versions": {p: importlib.metadata.version(p) for p in ["torch", "qwen-asr", "transformers"]}}), flush=True)
    warmed = False
    failed = False
    for row in rows:
        if "excluded_reason" in row:
            print(json.dumps({"kind": "excluded", "id": row["id"], "reason": row["excluded_reason"]}), flush=True)
            continue
        samples, rate = sf.read(row["audio"], dtype="float32")
        if rate != 16000 or samples.ndim != 1:
            raise ValueError("manifest audio must be 16 kHz mono")
        with torch.inference_mode():
            if not warmed:
                model.transcribe(audio=(samples, rate))
                warmed = True
            start = time.perf_counter()
            output = {"kind": "result", "id": row["id"], "backend": "torch-bf16-eager-masked-cpu",
                      "reference": row["reference"], "duration_seconds": len(samples)/rate}
            try:
                result = model.transcribe(audio=(samples, rate))[0]
                output.update(text=result.text, language=result.language)
            except Exception as error:
                failed = True
                output.update(text="", error=str(error))
            output["seconds"] = time.perf_counter()-start
            print(json.dumps(output, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit("transcription failures retained in results")


if __name__ == "__main__":
    main()
