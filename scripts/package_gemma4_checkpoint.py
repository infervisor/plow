#!/usr/bin/env python3
"""Freeze an existing Gemma 4 BF16/FP8-KV H100 build without pruning its programs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--runtime", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--evidence", action="append", default=[], type=Path)
    args = ap.parse_args()
    build = json.loads((args.assets / "build.json").read_text())
    assert build["arch"] == "sm_90a" and build["n_cu"] == 132
    assert build["precision"]["weight_enc"] == "bf16"
    assert set(build["shapes"]["kv_dtype"].values()) == {"e4m3"}
    decode = sorted({p["batch"] for p in build["programs"] if p["kind"] == "decode"})
    prefill = sorted({p["bucket"] for p in build["programs"] if p["kind"] == "prefill"})
    assert decode == [1, 2, 4, 8, 16] and prefill == [128, 512, 1024]
    packet = json.loads(subprocess.check_output(
        [str(args.runtime.resolve()), "disasm", str(args.assets / "model.pkt"),
         "--format", "json", "--tensors", "--no-analysis", "--range", "0..0"],
        env={**os.environ, "RUST_LOG": "off"}))
    assert [p["t"] for p in packet["programs"]] == prefill + decode
    assert [t["bytes"] for t in packet["tensors"] if t["name"] == "in.pos"] == [32768 * 4]
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ["bin", "lib", "assets", "evidence"]:
        (args.out / name).mkdir()
    shutil.copy2(args.runtime, args.out / "bin/plowrt")
    for source in args.assets.iterdir():
        if source.is_file():
            shutil.copy2(source, args.out / "assets" / source.name)
    checkpoint = args.out / "assets/checkpoint"
    checkpoint.mkdir()
    subprocess.run(["cp", "--reflink=auto", "-L", "-R",
                    str(args.assets / "checkpoint") + "/.", str(checkpoint)], check=True)
    linked = subprocess.check_output(["ldd", str(args.runtime)], text=True)
    for name in re.findall(r"(/[^\s()]+)", linked):
        source = Path(name)
        target = args.out / "lib" / source.name
        if target.exists():
            assert digest(target) == digest(source), "conflicting library names"
        else:
            shutil.copy2(source, target)
        if source.name == "ld-linux-x86-64.so.2":
            # The NVIDIA driver loads these compatibility libraries at runtime.
            for name in ["libdl.so.2", "libpthread.so.0", "librt.so.1"]:
                shutil.copy2(source.resolve().parent / name, args.out / "lib" / name)
    assert (args.out / "lib/ld-linux-x86-64.so.2").is_file()
    for source in args.evidence:
        target = args.out / "evidence" / source.name
        assert not target.exists(), target
        shutil.copy2(source, target)
    scripts = Path(__file__).resolve().parent
    shutil.copy2(scripts / "gemma4_checkpoint_workloads.py", args.out / "workloads.py")
    (args.out / "serve.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
checkpoint_root="$(cd -- "$(dirname -- "$0")" && pwd)"
export PLOW_LIBCUDA="${PLOW_LIBCUDA:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}"
export PLOW_NV_CUBIN="$checkpoint_root/assets/interp_sm90a_fp8kv.cubin"
export PLOW_NV_CUBIN_PF="$checkpoint_root/assets/interp_sm90a_pf_fp8kv.cubin"
export PLOW_VMM_PREFIX=1
export PLOW_PF_BATCH=0
export PLOW_MULTISTEP=0
export PLOW_TOKEN_BATCH=1
exec "$checkpoint_root/lib/ld-linux-x86-64.so.2" \\
  --library-path "$checkpoint_root/lib:/usr/local/cuda/lib64" \\
  "$checkpoint_root/bin/plowrt" serve --assets "$checkpoint_root/assets" \\
  --port "${PORT:-8080}" --max-hold-ms 0 --slo-ms "${SLO_MS:-600000}" \\
  --max-queued-requests "${MAX_QUEUED_REQUESTS:-64}" "$@"
''')
    (args.out / "serve.sh").chmod(0o755)
    (args.out / "verify.py").write_text('''#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for name, expected in manifest["files"].items():
    path = root / name
    assert path.is_file() and not path.is_symlink(), name
    assert path.stat().st_size == expected["bytes"], name
    with path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected["sha256"], name
build = json.loads((root / "assets/build.json").read_text())
assert sorted({p["batch"] for p in build["programs"] if p["kind"] == "decode"}) == manifest["decode_rungs"]
assert sorted({p["bucket"] for p in build["programs"] if p["kind"] == "prefill"}) == manifest["prefill_buckets"]
print("Verified", len(manifest["files"]), "files; decode", manifest["decode_rungs"], "prefill", manifest["prefill_buckets"])
''')
    (args.out / "README.md").write_text('''# Gemma 4 31B IT / plowrt checkpoint

BF16 weights and activations; E4M3 FP8 KV with FP32 per-row scales.
Target: one H100 SXM5 80GB, sm_90a, 132 SMs. Context limit: 32768 tokens
including output. Physical slots: 16. Decode rungs: 1, 2, 4, 8, 16.
Prefill buckets: 128, 512, 1024. All emitted programs and cubins are retained.

```sh
python3 verify.py
./serve.sh
# From another terminal:
python3 workloads.py --concurrency 4
python3 workloads.py --concurrency 16
```

`PORT`, `SLO_MS`, and `MAX_QUEUED_REQUESTS` configure the launcher.
The sample SLO is ten minutes to allow long workloads to queue. Set it to the
deployment's latency budget. Queued requests do not increase physical slots.
The API listens on the runtime's default TCP interface; route it through your
deployment's access-controlled ingress. `/metrics` exposes runtime metrics.

Prefix reuse is explicitly enabled. FP8 KV uses ordinary single-segment
prefill; packed prefill is unavailable. The token-batch selector is enabled,
but CUDA currently uses the serving fallback executor. Multi-step decode is
disabled in this qualified configuration. No vLLM performance win is claimed.
Application-specific quality and sustained-load qualification remain required.
FP8-KV logits can differ when cold and warm requests use different prefill
buckets. The workloads report cold/warm text agreement separately from cache
reuse and concurrent warm replay. A cache hit does not guarantee identical text.

Weights, tokenizer, compiled assets, and runtime are local regular files.
Bundled ELF libraries avoid a dependency on the build machine's Nix store.
A compatible Linux x86_64 NVIDIA driver is required; `PLOW_LIBCUDA` can
override its path. This checkpoint is stored on this instance's local disk;
it is not an off-instance backup. Preserve the whole directory when copying.
`manifest.json` records source, precision, rungs, and SHA-256 hashes.
`evidence/` contains the validation logs supplied during packaging.
''')
    repo = scripts.parent
    manifest = {
        "schema": 1, "model": "google/gemma-4-31B-it",
        "source_revision": (args.assets / "checkpoint").resolve().name,
        "runtime_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "runtime_source_diff": subprocess.check_output(
            ["git", "diff", "HEAD", "--", "crates/plowrt"], cwd=repo, text=True),
        "precision": build["precision"], "arch": build["arch"],
        "decode_rungs": decode, "prefill_buckets": prefill,
        "max_context": 32768, "physical_slots": 16, "files": {},
    }
    for path in sorted(args.out.rglob("*")):
        if path.is_file():
            assert not path.is_symlink(), path
            manifest["files"][str(path.relative_to(args.out))] = {
                "bytes": path.stat().st_size, "sha256": digest(path)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
