#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request


ROOT = pathlib.Path("plans/gemma4-12b-roofline")
URL = "http://127.0.0.1:8013"
MODEL = "checkpoint-fp8"


def post(body):
    request = urllib.request.Request(
        URL + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=600)


def prompt(length, seed):
    return (" " + ["hello", "world", "test", "data"][seed % 4]) * length


def generate(length, output, seed):
    body = {
        "model": MODEL,
        "prompt": prompt(length, seed),
        "add_special_tokens": False,
        "temperature": 0,
        "max_tokens": output,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text, usage, finish, done = [], None, None, False
    with post(body) as response:
        for line in response:
            if line.strip() == b"data: [DONE]":
                done = True
                break
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            assert "error" not in event, event
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text.append(choice.get("text", ""))
                finish = choice.get("finish_reason") or finish
    assert done and usage, (done, usage)
    assert usage["prompt_tokens"] == length, usage
    assert usage["completion_tokens"] == output, usage
    assert finish == "length", finish
    return "".join(text)


def cancel_and_recover():
    body = {
        "model": MODEL,
        "prompt": prompt(2305, 3),
        "add_special_tokens": False,
        "temperature": 0,
        "max_tokens": 256,
        "ignore_eos": True,
        "stream": True,
    }
    with post(body) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                break
    return generate(777, 3, 0)


def context_rejection():
    try:
        with post({"model": MODEL, "prompt": prompt(20481, 0), "max_tokens": 1}) as response:
            raise AssertionError(f"oversized prompt accepted: {response.read()!r}")
    except urllib.error.HTTPError as error:
        assert error.code in (400, 429), error.code


def wait_ready(server):
    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server readiness timeout")


def run(label, chunk, budget):
    env = dict(
        os.environ,
        GEMMA_NATIVE_ASSETS=str(ROOT / "w8a16-b1-all-prefill-rungs"),
        GEMMA_NATIVE_OBJECTS=str(ROOT / "cubin-w8a16-async-1"),
        GEMMA_PF_CHUNK=str(chunk),
        GEMMA_PF_BUDGET=str(budget),
    )
    server_log = ROOT / f"w8a16-b1-ladder-{label}-server.log"
    with server_log.open("w") as log:
        server = subprocess.Popen(
            ["/usr/bin/bash", str(ROOT / "serve_native.sh"), "fp8"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(server)
            outputs = {
                f"{length}:{output}:{seed}": generate(length, output, seed)
                for length, output, seed in [(777, 3, 0), (2305, 7, 1), (8192, 8, 2), (16384, 8, 3)]
            }
            assert cancel_and_recover() == outputs["777:3:0"]
            context_rejection()
            bench_log = ROOT / f"w8a16-b1-ladder-{label}.log"
            bench_json = ROOT / f"w8a16-b1-ladder-{label}.json"
            with bench_log.open("w") as bench:
                subprocess.run(
                    [
                        "/usr/bin/python3",
                        "scripts/bench_packed_serve.py",
                        "--url",
                        URL,
                        "--out",
                        str(bench_json),
                        "--label",
                        f"w8a16-b1-ladder-{label}",
                        "--inputs",
                        "1024",
                        "16384",
                        "--outputs",
                        "128",
                        "--concurrency",
                        "1",
                        "--repeats",
                        "3",
                        "--warmups",
                        "1",
                    ],
                    stdout=bench,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=env,
                )
            (ROOT / f"w8a16-b1-ladder-{label}-outputs.json").write_text(
                json.dumps(outputs, indent=2, sort_keys=True) + "\n"
            )
        finally:
            server.terminate()
            server.wait(timeout=30)


def main():
    run("chunk1024", 1024, 2048)
    run("chunk8192", 8192, 8192)
    control = json.loads((ROOT / "w8a16-b1-ladder-chunk1024-outputs.json").read_text())
    candidate = json.loads((ROOT / "w8a16-b1-ladder-chunk8192-outputs.json").read_text())
    assert candidate == control, (control, candidate)
    print("PASS sequential outputs, cancellation recovery, context rejection, and chunk parity")


if __name__ == "__main__":
    main()
