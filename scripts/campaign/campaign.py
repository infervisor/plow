#!/usr/bin/env python3
"""Perf-campaign driver: one recipe file per measured cell, one command per stage.

    campaign.py build   <recipe.toml> --out DIR        # base emit -> objects -> role emit
    campaign.py bench   <recipe.toml> --assets DIR --out DIR [--concs "1 4"] [--in-lens ...]
    campaign.py compare <results.csv> [reference.csv]   # ratio table, cell by cell
    campaign.py ledger  <results.csv> --cell NAME --note TEXT [--provisional]

The recipe pins everything that decides a number: checkpoint revision, precision, emit
knobs and flags, object-build gates, serve-side mirrors, and the client protocol. Every
GPU stage runs under `perf-data/tools/gpulease`; a run the lease audits as contended is
recorded as such and refused by `ledger` unless `--provisional` is given. Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPULEASE = REPO / "perf-data" / "tools" / "gpulease"
BENCH = REPO / "scripts" / "bench_plowrt_serve.sh"
CSV_HEADER = (
    "input_len,concurrency,ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,itl_p99,"
    "out_tok_s,req_per_s,ok_reqs,gen_toks"
)


def die(msg: str) -> None:
    print(f"campaign: {msg}", file=sys.stderr)
    sys.exit(2)


def load(path: str) -> dict:
    with open(path, "rb") as f:
        r = tomllib.load(f)
    for k in ("cell", "emit", "bench"):
        if k not in r:
            die(f"{path}: missing [{k}]")
    return r


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def run(cmd: list[str], env: dict, log: Path) -> int:
    print("  $", " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(cmd) + "\n").encode())
        f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode


def nix(cmd: list[str]) -> list[str]:
    return ["nix", "develop", "--command", *cmd]


def env_with(base: dict, extra: dict) -> dict:
    e = dict(base)
    e.update({k: str(v) for k, v in extra.items()})
    return e


# ---------------------------------------------------------------- build
def cmd_build(a: argparse.Namespace) -> None:
    r = load(a.recipe)
    cell, emit = r["cell"], r["emit"]
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        die(f"{out} exists and is not empty; a build is reproducible only into a fresh dir")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "build.log"
    plowc = REPO / "target" / "release" / "plowc"
    if not plowc.exists():
        die("target/release/plowc missing: nix develop -c cargo build -p plowc --release")

    base_args = [
        str(plowc),
        "--hf-dir", cell["hf_dir"],
        "--gpu", cell["gpu"], "--arch", cell["arch"], "--n-cu", str(cell["n_cu"]),
        "--max-ctx", str(cell["max_ctx"]),
        "--emit", emit.get("emit", "devblob+cubin"),
        *emit.get("args", []),
    ]
    common = env_with(os.environ, emit.get("env", {}))
    # The one emit-side variable of an A/B, named on the command line so build-record carries it.
    overrides = dict(kv.split("=", 1) for kv in (a.env or []))
    common.update(overrides)

    roles = r.get("emit_roles")
    objects = r.get("objects")
    if roles:
        # Role objects are looked up in the emit --out dir, so: base emit -> objects -> role emit.
        base_dir = out / "base"
        base_dir.mkdir()
        print("== base emit", file=sys.stderr)
        if run(nix([*base_args, "--out", str(base_dir)]), common, log):
            die("base emit failed; see build.log")
        obj_dir = out / "objects"
        if objects:
            print("== objects", file=sys.stderr)
            oenv = env_with(os.environ, objects.get("env", {}))
            oenv["PLOW_CUBIN_CONFIG"] = str(base_dir / "plow_config.h")
            if run(["bash", str(REPO / objects["script"]), str(base_dir), str(obj_dir)], oenv, log):
                die("object build failed; see build.log")
        assets = out / "assets"
        assets.mkdir()
        for f in objects.get("role_files", []) if objects else []:
            (assets / f).write_bytes((obj_dir / f).read_bytes())
        print("== role emit", file=sys.stderr)
        if run(nix([*base_args, "--out", str(assets)]), env_with(common, roles.get("env", {})), log):
            die("role emit failed; see build.log")
    else:
        assets = out / "assets"
        print("== emit", file=sys.stderr)
        if run(nix([*base_args, "--out", str(assets)]), common, log):
            die("emit failed; see build.log")

    ck = cell.get("checkpoint_dir")
    if ck:
        # A composed checkpoint (BF16 shards + fp8/ twins) replaces the emit's snapshot link.
        link = assets / "checkpoint"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ck)

    rec = {
        "recipe": str(Path(a.recipe).resolve()),
        "cell": cell,
        "overrides": overrides,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hashes": {p.name: sha(p) for p in sorted(assets.glob("*")) if p.is_file() and p.suffix in (".pkt", ".cubin", ".elf", ".co")},
        "objects": {p.name: sha(p) for p in sorted((out / "objects").glob("*.cubin"))} if (out / "objects").exists() else {},
    }
    (out / "build-record.json").write_text(json.dumps(rec, indent=1))
    print(f"built {assets}\nrecord {out / 'build-record.json'}", file=sys.stderr)


# ---------------------------------------------------------------- bench
def gpu_header() -> dict:
    q = "name,uuid,driver_version,clocks.sm,clocks.mem,power.limit,memory.used"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(zip(q.split(","), [x.strip() for x in out.split(",")]))
    except Exception:  # AMD or no SMI: the lease log still records the vendor
        return {}


def cmd_bench(a: argparse.Namespace) -> None:
    r = load(a.recipe)
    cell, bench, serve = r["cell"], dict(r["bench"]), dict(r.get("serve", {}))
    # A profile is a named workload on the same cell: `realtime` owns C1-C4 latency, `throughput`
    # owns C4-C16 output tok/s at long context. Its keys override [bench]; its `serve_env`
    # merges over [serve].env, so the two profiles can differ in multistep, ladder use, etc.
    profile = None
    if a.profile:
        profiles = bench.get("profiles", {})
        if a.profile not in profiles:
            die(f"recipe has no [bench.profiles.{a.profile}]; have {sorted(profiles)}")
        profile = dict(profiles[a.profile])
        serve_env = dict(serve.get("env", {}))
        serve_env.update(profile.pop("serve_env", {}))
        serve["env"] = serve_env
        bench.update(profile)
    assets = Path(a.assets).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (assets / "model.pkt").exists():
        die(f"{assets}/model.pkt missing")
    plowrt = Path(serve.get("plowrt", str(REPO / "target" / "release" / "plowrt"))).resolve()
    # A private copy: the shared target/release binary can be rebuilt by another agent mid-run.
    private = out / "plowrt"
    private.write_bytes(plowrt.read_bytes())
    private.chmod(0o755)

    env = env_with(os.environ, serve.get("env", {}))
    # The one variable of an A/B, named on the command line so the record carries it.
    overrides = dict(kv.split("=", 1) for kv in (a.env or []))
    env.update(overrides)
    # A `build` places the segment/role objects beside the assets; the serve-side mirror of
    # the emit classing needs that directory and must not be typed by hand.
    objects = assets.parent / "objects"
    if "objects" in r and "PLOW_PF_SEG_DIR" not in env and objects.is_dir():
        env["PLOW_PF_SEG_DIR"] = str(objects)
    env.update({
        "VLLM_VENV": bench.get("vllm_venv", "/opt/pytorch"),
        "HF_HOME": str(out / "hf-home"),
        "IN_LENS": a.in_lens or bench.get("in_lens", "128 1024 4096"),
        "CONCS": a.concs or bench.get("concs", "1"),
        "NPROMPT": str(bench.get("nprompt", 32)),
        "OUTLEN": str(bench.get("outlen", 128)),
        "BENCH_BACKEND": bench.get("backend", "openai"),
        "BENCH_EXTRA_ARGS": f"--num-warmups {bench.get('warmups', 16)} --seed {bench.get('seed', 42)}",
        "GATE_PROMPT": bench["gate_prompt"],
        "PLOWRT_BIN": str(private),
        "OUTDIR": str(out / "client"),
        "LOG": str(out / "server.log"),
        "SERVE_EXTRA_ARGS": serve.get("extra_args", ""),
    })
    (out / "hf-home").mkdir(exist_ok=True)
    model_id = bench.get("model_id") or json.loads((assets / "build.json").read_text()).get("slug") or cell["revision"]
    label = a.label or f"{cell['name']}-{Path(a.recipe).stem}" + (f"-{a.profile}" if a.profile else "")
    # The env is exported INSIDE the leased child, not passed through gpulease: gpulease assigns
    # its own `LOG=` and an exported LOG would keep that value, sending the server log into the
    # lease log (which is exactly what happened before this wrapper existed).
    wrapper = out / "run.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for k, v in sorted(env.items()):
        if k in os.environ and os.environ[k] == v and k not in overrides:
            continue  # inherited, unchanged
        lines.append(f"export {k}={shlex.quote(v)}")
    lines.append("exec " + " ".join(shlex.quote(x) for x in [
        str(BENCH), str(assets), str(bench.get("port", 8765)), model_id, bench["tokenizer"], str(bench.get("ready_s", 1200))]))
    wrapper.write_text("\n".join(lines) + "\n")
    wrapper.chmod(0o755)
    cmd = [str(GPULEASE), "-n", str(cell.get("n_gpu", 1)), label, str(wrapper)]
    log = out / "run.log"
    log.write_bytes(b"")
    rc = run(cmd, dict(os.environ), log)
    text = log.read_text(errors="replace")
    rows = [ln for ln in text.splitlines() if ln[:1].isdigit() and ln.count(",") == 12]
    (out / "results.csv").write_text(CSV_HEADER + "\n" + "\n".join(rows) + ("\n" if rows else ""))
    rec = {
        "recipe": str(Path(a.recipe).resolve()),
        "cell": cell,
        "label": label,
        "profile": a.profile,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "gpu": gpu_header(),
        "contended": "CONTENDED" in text,
        "gate": "coherence gate: PASS" in text,
        "protocol": {k: env[k] for k in ("IN_LENS", "CONCS", "NPROMPT", "OUTLEN", "BENCH_BACKEND", "BENCH_EXTRA_ARGS")},
        "serve_env": serve.get("env", {}),
        "overrides": overrides,
        "hashes": {p.name: sha(p) for p in sorted(assets.glob("*")) if p.is_file() and p.suffix in (".pkt", ".cubin", ".elf", ".co")},
        "rows": len(rows),
        "bench_rc": rc,
    }
    (out / "run-record.json").write_text(json.dumps(rec, indent=1))
    print(CSV_HEADER)
    print("\n".join(rows))
    if not rec["gate"]:
        die("coherence gate did not pass; numbers above are not evidence")
    if rec["contended"]:
        print("campaign: GPU was contended -- recorded as provisional", file=sys.stderr)
    if a.reference or r.get("reference"):
        compare(out / "results.csv", Path(a.reference or REPO / r["reference"]["csv"]))


# ---------------------------------------------------------------- compare
def read_rows(p: Path) -> dict:
    with open(p) as f:
        return {(int(x["input_len"]), int(x["concurrency"])): x for x in csv.DictReader(f)}


def compare(res: Path, ref: Path) -> None:
    a, b = read_rows(res), read_rows(ref)
    print(f"\n{'in':>5} {'C':>2} | {'TTFT':>8} {'ref':>8} {'x':>5} | {'TPOT':>7} {'ref':>7} {'x':>5} | {'tok/s':>7} {'ref':>7}")
    for k in sorted(a):
        x = a[k]
        y = b.get(k)
        if not y:
            print(f"{k[0]:>5} {k[1]:>2} | {float(x['ttft_ms']):8.2f} {'-':>8} {'-':>5} | {float(x['tpot_ms']):7.2f} {'-':>7} {'-':>5} | {float(x['out_tok_s']):7.1f} {'-':>7}")
            continue
        print(f"{k[0]:>5} {k[1]:>2} | {float(x['ttft_ms']):8.2f} {float(y['ttft_ms']):8.2f} {float(x['ttft_ms'])/float(y['ttft_ms']):5.2f} | "
              f"{float(x['tpot_ms']):7.2f} {float(y['tpot_ms']):7.2f} {float(x['tpot_ms'])/float(y['tpot_ms']):5.2f} | "
              f"{float(x['out_tok_s']):7.1f} {float(y['out_tok_s']):7.1f}")


def cmd_compare(a: argparse.Namespace) -> None:
    compare(Path(a.results), Path(a.reference))


# ---------------------------------------------------------------- ledger
def cmd_ledger(a: argparse.Namespace) -> None:
    res = Path(a.results).resolve()
    rec_path = res.parent / "run-record.json"
    if not rec_path.exists():
        die("ledger needs the run-record.json beside results.csv (a `bench` output)")
    rec = json.loads(rec_path.read_text())
    if rec.get("contended") and not a.provisional:
        die("run was contended; pass --provisional to record it as such")
    if not rec.get("gate"):
        die("run did not pass the coherence gate")
    ledger = REPO / "perf-data" / "campaign" / f"{a.cell}.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    with open(ledger, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "commit", "label", "provisional", "pkt_sha", "note", *CSV_HEADER.split(",")])
        for row in read_rows(res).values():
            w.writerow([rec["utc"], rec["commit"][:12], rec["label"], int(bool(rec.get("contended"))),
                        rec["hashes"].get("model.pkt", "")[:16], a.note, *[row[c] for c in CSV_HEADER.split(",")]])
    print(f"appended {len(read_rows(res))} row(s) to {ledger}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("recipe"); b.add_argument("--out", required=True)
    b.add_argument("--env", action="append", metavar="K=V", help="one-variable override for the emit env; recorded")
    b.set_defaults(f=cmd_build)
    n = sp.add_parser("bench"); n.add_argument("recipe"); n.add_argument("--assets", required=True); n.add_argument("--out", required=True)
    n.add_argument("--concs"); n.add_argument("--in-lens"); n.add_argument("--label"); n.add_argument("--reference")
    n.add_argument("--env", action="append", metavar="K=V", help="one-variable override for the server env; recorded")
    n.add_argument("--profile", help="named workload from [bench.profiles.*] (e.g. realtime, throughput)")
    n.set_defaults(f=cmd_bench)
    c = sp.add_parser("compare"); c.add_argument("results"); c.add_argument("reference"); c.set_defaults(f=cmd_compare)
    l = sp.add_parser("ledger"); l.add_argument("results"); l.add_argument("--cell", required=True); l.add_argument("--note", required=True)
    l.add_argument("--provisional", action="store_true"); l.set_defaults(f=cmd_ledger)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
