#!/usr/bin/env python3
"""Run the pinned Nemotron GGUF with the official external Metal reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


MODEL_REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
MODEL_SHA256 = "3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1"
MODEL_BYTES = 742090464


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--right-context", type=int, choices=(0, 3, 6, 13), default=3)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    runtime, model, audio = (p.absolute() for p in (args.runtime, args.model, args.audio))
    if model.stat().st_size != MODEL_BYTES or sha256(model) != MODEL_SHA256:
        parser.error("model does not match the pinned NVIDIA Nemotron 3.5 Q8_0 GGUF")
    if not runtime.is_file() or not audio.is_file():
        parser.error("runtime and audio must be existing files")
    args.output.mkdir(parents=True, exist_ok=False)
    command = [str(runtime), "transcribe", str(audio), "--model", str(model),
               "--device", "metal", "--language", args.language, "--json", "--verbose",
               "--no-batching", "--asr.streaming.rnnt_right_context",
               str(args.right_context)]
    if args.stream:
        command.append("--stream")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NEMO_SPEECH_")}
    metadata = {"backend": "nvidia-nemo-speech-cpp-reference", "plow_packets": False,
                "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
                "runtime_sha256": sha256(runtime), "audio_sha256": sha256(audio),
                "platform": platform.platform(), "command": command,
                "stream": args.stream, "input_paced_realtime": False,
                "timing_scope": "process including load, warmup and transcription"}
    started = time.perf_counter()
    try:
        with (args.output / "stdout.json").open("w") as out, \
             (args.output / "stderr.log").open("w") as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err,
                                    timeout=args.timeout, check=False)
        metadata["returncode"] = result.returncode
    except subprocess.TimeoutExpired:
        metadata["error"] = "timeout"
        raise
    finally:
        metadata["process_seconds"] = time.perf_counter() - started
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    output = json.loads((args.output / "stdout.json").read_text())
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
