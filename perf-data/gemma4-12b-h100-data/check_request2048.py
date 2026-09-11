import os
import pathlib
import subprocess
import time
import urllib.request

root = pathlib.Path("plans/gemma4-12b-roofline")
for label, asset in [("request2048", "bf16-b32-request2048")]:
    env = dict(os.environ, GEMMA_NATIVE_ASSETS=str(root / asset),
               GEMMA_NATIVE_OBJECTS=str(root / "cubin-bf16-b32-smepi"),
               GEMMA_PF_CHUNK="2048", GEMMA_PF_BUDGET="8192")
    with open(root / f"bf16-chunk-serving-{label}-server.log", "w") as log:
        server = subprocess.Popen(["bash", str(root / "serve_native.sh"), "bf16"],
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"{label}: server PID {server.pid}", flush=True)
        try:
            for _ in range(90):
                if server.poll() is not None:
                    raise RuntimeError(f"{label} server exited")
                try:
                    urllib.request.urlopen("http://127.0.0.1:8013/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("readiness timeout")
            with open(root / f"bf16-chunk-serving-{label}-verify.log", "w") as verify:
                subprocess.run(["python3", "scripts/verify_packed_serve.py", "--control",
                                "http://127.0.0.1:8013", "--candidate", "http://127.0.0.1:8013",
                                "--model", "checkpoint", "--max-ctx", "20480"],
                               stdout=verify, stderr=subprocess.STDOUT, check=True)
            with open(root / f"bf16-chunk-serving-{label}.log", "w") as bench:
                subprocess.run(["python3", "scripts/bench_packed_serve.py", "--url",
                                "http://127.0.0.1:8013", "--out", str(root / f"bf16-chunk-serving-{label}.json"),
                                "--label", f"bf16-chunk-serving-{label}", "--inputs", "1024", "16384",
                                "--outputs", "128", "--concurrency", "1", "32", "--repeats", "1",
                                "--warmups", "0"], stdout=bench, stderr=subprocess.STDOUT, check=True)
        finally:
            server.terminate()
            server.wait(timeout=30)
        print(f"{label}: complete", flush=True)
