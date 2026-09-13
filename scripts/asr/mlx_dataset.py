#!/usr/bin/env python3
"""Run a pinned local MLX comparison on an ASR JSONL manifest."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx_qwen3_asr import Session, load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    quant_path = Path(args.checkpoint) / "quantization_config.json"
    quantization = json.loads(quant_path.read_text()) if quant_path.exists() else None
    backend = (f"mlx-q{quantization['bits']}-g{quantization['group_size']}"
               if quantization else "mlx-bf16")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    for row in rows:
        if "excluded_reason" not in row:
            if hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest() != row["audio_sha256"]:
                raise ValueError(f"audio checksum mismatch: {row['id']}")
    session = Session(model=args.checkpoint, dtype=mx.bfloat16)
    print(json.dumps({"kind": "metadata", "dtype": "checkpoint" if quantization else "bfloat16",
                      "requested_dtype": "bfloat16", "weight_quantization": quantization,
                      "backend": backend, "max_new_tokens": 1024,
                      "language_policy": "automatic", "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      "versions": {p: importlib.metadata.version(p) for p in ["mlx", "mlx-metal", "mlx-qwen3-asr"]}}), flush=True)
    warmed = False
    failed = False
    for row in rows:
        if "excluded_reason" in row:
            print(json.dumps({"kind": "excluded", "id": row["id"], "reason": row["excluded_reason"]}), flush=True)
            continue
        samples = load_audio(row["audio"])
        if not warmed:
            session.transcribe(samples, max_new_tokens=1024)
            mx.synchronize()
            warmed = True
        mx.synchronize()
        start = time.perf_counter()
        output = {"kind": "result", "id": row["id"], "backend": backend,
                  "reference": row["reference"], "duration_seconds": len(samples)/16000}
        try:
            result = session.transcribe(samples, max_new_tokens=1024)
            mx.synchronize()
            output.update(text=result.text, language=result.language)
        except Exception as error:
            failed = True
            output.update(text="", error=str(error))
        output.update(seconds=time.perf_counter()-start, peak_memory_bytes=mx.get_peak_memory())
        print(json.dumps(output, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit("transcription failures retained in results")


if __name__ == "__main__":
    main()
