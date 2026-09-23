#!/usr/bin/env python3
"""Perf-campaign driver: one recipe file per measured cell, one command per stage.

    campaign.py build   <recipe.toml> --out DIR [--hf-dir SNAPSHOT]  # base emit -> objects -> role emit
    campaign.py serve   <recipe.toml> --assets DIR --profile P [--port N]  # production plowrt serve
    campaign.py bench   <recipe.toml> --assets DIR --out DIR [--concs "1 4"] [--in-lens ...]
    campaign.py compare <results.csv> <reference.csv> [--roofline] [--recipe <recipe.toml>]
    campaign.py roofline <recipe.toml> [--results results.csv]
    campaign.py loop    <recipe.toml> [--out DIR] [--profile realtime]
    campaign.py sweep   <recipe.toml> --param KNOB --values V1,V2 [--out DIR]
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
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

# Add scripts/campaign to path for roofline module
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from roofline import generate_roofline_report
except Exception:
    generate_roofline_report = None

REPO = Path(__file__).resolve().parents[2]
GPULEASE = REPO / "perf-data" / "tools" / "gpulease"
BENCH = REPO / "scripts" / "bench_plowrt_serve.sh"
CSV_HEADER = (
    "input_len,concurrency,ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,itl_p99,"
    "out_tok_s,req_per_s,ok_reqs,gen_toks"
)
# Appended by `bench` from the bench script's `peak_mem_mib,<in>,<c>,<MiB>` lines; empty when the
# cell was not sampled (no nvidia-smi, or a results.csv from before the column existed).
MEM_COL = "peak_mem_mib"
VLLM_REFERENCE_IMAGE = "vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1"


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


def cmd_block_roofline(a: argparse.Namespace) -> None:
    from packet_roofline import analyze

    with open(a.recipe, "rb") as f:
        recipe = tomllib.load(f)
    roof = recipe["roofline"]
    runtime = Path(a.plowrt).resolve()
    packet = Path(a.packet).resolve()
    command = [str(runtime), "disasm", str(packet), "--program", str(a.program)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    if result.returncode:
        die(result.stderr or result.stdout)
    router = Path(a.router_table).resolve() if a.router_table else None
    record = analyze(result.stdout, a.ctx, roof["bandwidth_gbps"], roof["bf16_tflops"],
                     router.read_bytes() if router else None, fp8_tflops=roof.get("fp8_tflops"))
    if router:
        record.update(router_table=str(router), router_table_sha256=sha(router))
    record.update(recipe=str(Path(a.recipe).resolve()), recipe_sha256=sha(Path(a.recipe)),
                  packet=str(packet), packet_sha256=sha(packet), runtime_sha256=sha(runtime),
                  commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "disasm.txt").write_text(result.stdout)
    (out / "roofline.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def prepare_block_bench(a: argparse.Namespace) -> dict:
    with open(a.recipe, "rb") as f:
        recipe = tomllib.load(f)
    cell = recipe["cell"]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    assets = out / "assets"
    assets.mkdir()
    packet, objects, inputs = Path(a.packet).resolve(), Path(a.objects).resolve(), Path(a.inputs).resolve()
    checkpoint = Path(a.checkpoint).resolve()
    if not (inputs / "reference.bf16").is_file() or not (inputs / "reference.json").is_file():
        die("block inputs require reference.bf16 and reference.json from the numerical oracle")
    shutil.copy2(packet, assets / "model.pkt")
    build_record = packet.parent.parent / "build-record.json"
    if packet.name == "model.pkt" and build_record.is_file():
        built = json.loads(build_record.read_text())
        for name in ("model.pkt", "build.json", "plow_config.h"):
            source = packet.parent / name
            if built["hashes"].get(name) != sha(source):
                die(f"{source}: differs from the campaign build record")
            if name != "model.pkt":
                shutil.copy2(source, assets / name)
        shutil.copy2(build_record, out / "build-record.json")
    shutil.copytree(objects, out / "objects")
    shutil.copytree(inputs, out / "inputs")
    shutil.copy2(a.plowrt, out / "plowrt")
    private = out / "plowrt"
    private.chmod(0o755)
    command = [str(private), "amd-block", "--blob", str(assets / "model.pkt"),
               "--hsaco", str(out / "objects"), "--checkpoint", str(checkpoint),
               "--input-dir", str(out / "inputs"), "--ctx", str(a.ctx),
               "--tp", str(cell["n_gpu"]), "--repeat", str(a.repeat), "--warmup", str(a.warmup),
               "--dump", str(out / "outputs"), "--report", str(out / "measurement.json")]
    check = [str(REPO / "scripts/bench/plowbench-doctor.sh"), str(assets), str(out / "objects"),
             str(private), cell["arch"], "block"]
    rc = run(check, dict(os.environ), out / "doctor.log")
    if rc not in (0, 2):
        die(f"block preflight failed; see {out / 'doctor.log'}")
    env = {str(k): str(v) for k, v in recipe.get("block", {}).get("env", {}).items()}
    if a.trace:
        env.update(PLOW_TRACE_RAW=str(out / "trace.bin"), PLOW_TRACE_ALLRANKS="1")
    if getattr(a, "dstep_log", False):
        env.update(PLOW_DSTEP_LOG="1", PLOW_DSTEP_EVERY="32")
    record = dict(recipe=str(Path(a.recipe).resolve()), recipe_sha256=sha(Path(a.recipe)),
                  cell=cell, commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                  command=command, env=env, packet_sha256=sha(assets / "model.pkt"),
                  runtime_sha256=sha(private),
                  objects={p.name: sha(p) for p in sorted((out / "objects").iterdir()) if p.is_file()},
                  inputs={p.name: sha(p) for p in sorted((out / "inputs").iterdir()) if p.is_file()})
    # The queue exports visibility inside gpulease; never capture a parent's GPU mask.
    wrapper = out / "run.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             "source " + shlex.quote(str(REPO / "scripts/bench/plowbench.sh")), "pb_require_nix", "pb_hazard_env"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())]
    lines += ["exec " + shlex.join(command)]
    wrapper.write_text("\n".join(lines) + "\n")
    (out / "run-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def cmd_block_bench(a: argparse.Namespace) -> None:
    record = prepare_block_bench(a)
    out = Path(a.out).resolve()
    cell = record["cell"]
    submitted = subprocess.run([sys.executable, str(REPO / "scripts/bench/gpuq.py"),
                               "--root", str(Path(a.queue).resolve()), "submit", cell["name"],
                               str(cell["n_gpu"]), "bash", str(out / "run.sh")], check=True, capture_output=True, text=True)
    record.update(queue=str(Path(a.queue).resolve()), job=submitted.stdout.strip())
    (out / "run-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"queued {record['job']}; results: {out}")


def cmd_block_ab(a: argparse.Namespace) -> None:
    if a.routed_reference and a.require_bitwise:
        die("routed atomic repeat validation cannot claim bitwise reduction")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    arms = {}
    for name, source in (("ctl", a.control_build), ("treat", a.treatment_build),
                         ("ctl2", a.control_build), ("treat2", a.treatment_build)):
        build = Path(source).resolve()
        args = argparse.Namespace(**vars(a))
        args.out, args.packet, args.objects = str(out / name), str(build / "assets/model.pkt"), str(build / "objects")
        args.trace = False
        arms[name] = prepare_block_bench(args)
    scorer = out / "block_ab.py"
    shutil.copy2(REPO / "scripts/campaign/block_ab.py", scorer)
    wrapper = out / "run.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines += [f"bash {shlex.quote(str(out / name / 'run.sh'))} >{shlex.quote(str(out / name / 'run.log'))} 2>&1"
              for name in arms]
    score_command = [sys.executable, str(scorer), str(out)]
    audit_record = {}
    if a.routed_reference:
        reference = Path(a.routed_reference).resolve()
        audit = json.loads(reference.read_text())
        if audit.get("passed") is not True or audit.get("vllm_version") != "0.29.0":
            die("routed A/B requires a passed pinned connected reference audit")
        reference_dir = out / "routed-reference"
        shutil.copytree(reference.parent, reference_dir)
        checker = out / "block_fp8_aiter_compare.py"
        shutil.copy2(REPO / "runtime/tests/block_fp8_aiter_compare.py", checker)
        certificate = out / "routed-repeat.json"
        lines += [shlex.join(["sudo", "-n", "docker", "run", "--rm", "--network", "none",
            "-e", "OMP_NUM_THREADS=8", "-e", "MKL_NUM_THREADS=8", "-v", f"{out}:{out}",
            "--entrypoint", "python3", VLLM_REFERENCE_IMAGE, str(checker), str(out),
            "--check-routed-ab", "--reference-json", str(reference_dir / reference.name),
            "--tp", str(arms["ctl"]["cell"]["n_gpu"]), "--output", str(certificate)])]
        score_command += ["--routed-repeat-audit", str(certificate)]
        audit_record = dict(routed_reference_sha256=sha(reference_dir / reference.name),
                            routed_checker_sha256=sha(checker), routed_checker_image=VLLM_REFERENCE_IMAGE)
    if a.require_bitwise:
        score_command.append("--require-bitwise")
    lines += [shlex.join(score_command)]
    wrapper.write_text("\n".join(lines) + "\n")
    command = [sys.executable, str(REPO / "scripts/bench/gpuq.py"), "--root", a.queue,
               "submit", arms["ctl"]["cell"]["name"] + "-ab", str(arms["ctl"]["cell"]["n_gpu"]),
               "bash", str(REPO / "scripts/bench/quietx.sh"), str(Path(a.quiet_lock).resolve()),
               "bash", str(wrapper)]
    submitted = subprocess.run(command, check=True, capture_output=True, text=True)
    record = dict(scope="single-block-four-arm; not serving qualification", note=a.note,
                  require_bitwise=a.require_bitwise,
                  arms={name: sha(out / name / "run-record.json") for name in arms},
                  scorer_sha256=sha(scorer), job=submitted.stdout.strip(), queue=a.queue,
                  quiet_lock=str(Path(a.quiet_lock).resolve()), **audit_record)
    (out / "run-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"queued {record['job']}; results: {out}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def cmd_serve_bench(a: argparse.Namespace) -> None:
    r = load(a.recipe)
    cell, bench = r["cell"], r["bench"]
    raw = Path(cell["hf_dir"]).resolve()
    index = json.loads((raw / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values()) if not (raw / name).is_file()]
    if missing:
        die(f"full checkpoint incomplete: {len(missing)} missing shards")
    assets = Path(a.assets).resolve()
    objects = Path(a.objects).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(assets, out / "assets", symlinks=True)
    shutil.copytree(objects, out / "objects")
    shutil.copy2(a.plowrt, out / "plowrt")
    for name in ("plowbench.sh", "vllm029-client.sh"):
        shutil.copy2(REPO / "scripts/bench" / name, out / name)
    quality_lens = getattr(a, "quality_lens", None)
    if quality_lens:
        if any(int(n) < 1 or int(n) + 32 > cell["max_ctx"] for n in quality_lens.split(",")):
            die("quality context plus output exceeds the compiled capacity")
        shutil.copy2(REPO / "scripts/glm53_needle_probe.py", out / "needle_probe.py")
    env = {str(k): str(v) for k, v in r.get("serve", {}).get("env", {}).items()}
    env.update(dict(item.split("=", 1) for item in (a.env or [])))
    env.update(PB_VLLM=str(out / "vllm029-client.sh"), PB_TOKENIZER=str(raw),
               PB_SEED=str(bench.get("seed", 8193)), HSA_DISABLE_COREDUMP_ON_EXCEPTION="1")
    doctor_env = dict(os.environ, **env, PB_VLLM_ROCM_LIB=os.environ["ROCM_PATH"] + "/lib")
    rc = run([str(REPO / "scripts/bench/plowbench-doctor.sh"), str(out / "assets"),
              str(out / "objects"), str(out / "plowrt"), cell["arch"]], doctor_env, out / "doctor.log")
    if rc not in (0, 2):
        die(f"preflight failed: {out / 'doctor.log'}")
    in_lens = [int(n) for n in (a.in_lens or bench["in_lens"]).split()]
    concs = [int(n) for n in (a.concs or bench["concs"]).split()]
    prompts = a.nprompt or bench.get("nprompt", 128)
    output_len = bench.get("outlen", 128)
    if not in_lens or not concs or min(*in_lens, *concs, prompts, output_len) < 1:
        die("benchmark dimensions must be positive")
    if prompts < max(concs):
        die("prompt count must reach every requested concurrency")
    if max(in_lens) + output_len > cell["max_ctx"]:
        die("input plus output exceeds the compiled context capacity")
    image = VLLM_REFERENCE_IMAGE
    reference_env = {str(k): str(v) for k, v in r.get("reference", {}).get("env", {}).items()}
    reference_env_args = [arg for key, value in sorted(reference_env.items()) for arg in ("-e", f"{key}={value}")]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             "source " + shlex.quote(str(out / "plowbench.sh")), "pb_require_nix", "pb_hazard_env"]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items())]
    lines += ["PB_SERVER_PORT=$(pb_free_port)",
              "PB_SERVER_LOG=" + shlex.quote(str(out / "server.log"))]
    if a.server == "plow":
        lines += ["trap pb_serve_stop EXIT",
                  "pb_serve_start " + shlex.join([str(out / "plowrt"), str(out / "assets"),
                                                  str(out / "objects")]) +
                  ' "$PB_SERVER_PORT" "$PB_SERVER_LOG" 86400']
    else:
        cidfile = out / "container.id"
        lines += ["cidfile=" + shlex.quote(str(cidfile)),
                  'cleanup() { if test -s "$cidfile"; then sudo -n docker stop --time 30 "$(<"$cidfile")" >/dev/null 2>&1 || true; fi; if test -n "${PB_SERVER_PID:-}"; then wait "$PB_SERVER_PID" 2>/dev/null || true; fi; }',
                  "trap cleanup EXIT",
                  "visibility=()",
                  'if test -n "${ROCR_VISIBLE_DEVICES:-}"; then visibility+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"); fi',
                  'if test -n "${HIP_VISIBLE_DEVICES:-}"; then visibility+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"); fi',
                  "sudo -n docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri "
                  '--cidfile "$cidfile" "${visibility[@]}" '
                  "-e HF_HUB_OFFLINE=1 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 "
                  "-v /opt/models:/opt/models:ro --entrypoint vllm " + shlex.join([
                      *reference_env_args, image, "serve", str(raw), "--host", "127.0.0.1",
                      "--served-model-name", bench.get("model_id", "glm-5.3"),
                      "--tensor-parallel-size", str(cell["n_gpu"]), "--max-model-len", str(cell["max_ctx"]),
                      "--max-num-seqs", str(max(concs)), "--gpu-memory-utilization", "0.95",
                      "--no-enable-prefix-caching", "--trust-remote-code",
                      *r.get("reference", {}).get("args", []),
                  ]) + ' --port "$PB_SERVER_PORT" > "$PB_SERVER_LOG" 2>&1 &',
                  "PB_SERVER_PID=$!"]
    smoke_code = "import json,sys; print(json.dumps(dict(model=sys.argv[1], prompt='What is the capital of France? Answer:', max_tokens=64, temperature=0)))"
    lines += [f"pb_serve_wait {bench.get('ready_s', 1800)}", "model=$(pb_model_id)",
              "smoke_payload=$(python3 -c " + shlex.quote(smoke_code) + ' "$model")',
              'curl -fsS --max-time 600 "http://127.0.0.1:$PB_SERVER_PORT/v1/completions" '
              '-H "Content-Type: application/json" --data "$smoke_payload" > ' + shlex.quote(str(out / "smoke.json")),
              "python3 -c " + shlex.quote("import json,sys; d=json.load(open(sys.argv[1])); assert 'paris' in d['choices'][0]['text'].lower(), d; print('coherence smoke passed; not a full numerics gate')") + " " + shlex.quote(str(out / "smoke.json"))]
    if quality_lens:
        lines += ["python3 " + shlex.quote(str(out / "needle_probe.py")) +
                  ' --url "http://127.0.0.1:$PB_SERVER_PORT" ' + shlex.join([
                      "--arm", a.server, "--out", str(out / "needle.json"),
                      "--lens", quality_lens, "--exact-lengths",
                  ])]
    for context in in_lens:
        for concurrency in concs:
            tag = f"in{context}_c{concurrency}"
            lines += ["pb_bench " + shlex.join([str(out / "client"), tag]) + ' "$model" ' + shlex.join([
                str(concurrency), str(prompts), str(context), str(output_len),
                "--backend", "openai", "--endpoint", "/v1/completions", "--num-warmups",
                str(bench.get("warmups", 2)), "--temperature", "0", "--percentile-metrics", "ttft,tpot,itl,e2el",
            ]), "result=$(pb_result " + shlex.join([str(out / "client"), tag]) + ")",
                f'pb_validate_result "$result" {prompts} {output_len}']
    wrapper = out / "run.sh"
    wrapper.write_text("\n".join(lines) + "\n")
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    record = dict(server=a.server, recipe_sha256=sha(Path(a.recipe)), cell=cell,
                  image=image, env=env, reference_env=reference_env, contexts=in_lens, concurrencies=concs, prompts=prompts,
                  output_len=output_len, raw_index_sha256=sha(raw / "model.safetensors.index.json"),
                  runtime_sha256=sha(out / "plowrt"), packet_sha256=sha(out / "assets/model.pkt"),
                  objects={p.name: sha(p) for p in sorted((out / "objects").iterdir()) if p.is_file()},
                  status="prepared", numerics_qualified=False, precision_qualified=False,
                  comparison_scope="diagnostic; per-operation dtype parity and numerics not qualified")
    manifest = out / "assets/build.json"
    if manifest.is_file():
        record["plow_declared_precision"] = json.loads(manifest.read_text()).get("precision")
    if quality_lens:
        record.update(quality_lens=quality_lens, quality_probe_sha256=sha(out / "needle_probe.py"))
    if not a.dry_run:
        submitted = subprocess.run([sys.executable, str(REPO / "scripts/bench/gpuq.py"), "--root", a.queue,
                                    "submit", cell["name"] + "-" + a.server, str(cell["n_gpu"]),
                                    "bash", str(wrapper)], check=True, capture_output=True, text=True)
        record.update(status="queued", job=submitted.stdout.strip())
    (out / "run-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{record['status']}: {out}")


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
    if getattr(a, "hf_dir", None):
        cell["hf_dir"] = a.hf_dir
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
        "--emit", emit.get("emit", "devblob+cubin"),
        *emit.get("args", []),
    ]
    if "max_ctx" in cell and cell["max_ctx"]:
        base_args.extend(["--max-ctx", str(cell["max_ctx"])])
    common = env_with(os.environ, emit.get("env", {}))
    # The one emit-side variable of an A/B, named on the command line so build-record carries it.
    overrides = dict(kv.split("=", 1) for kv in (a.env or []))
    object_overrides = dict(kv.split("=", 1) for kv in (getattr(a, "object_env", None) or []))
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
            oenv.update(object_overrides)
            oenv["PLOW_CUBIN_CONFIG"] = str(base_dir / "plow_config.h")
            if run(["bash", str(REPO / objects["script"]), str(base_dir), str(obj_dir)], oenv, log):
                die("object build failed; see build.log")
        assets = out / "assets"
        assets.mkdir()
        for f in objects.get("role_files", []) if objects else []:
            (assets / f).write_bytes((obj_dir / f).read_bytes())
        print("== role emit", file=sys.stderr)
        # CLI overrides win over the recipe's role env too, so an A/B can switch a role off.
        if run(nix([*base_args, "--out", str(assets)]), env_with(env_with(common, roles.get("env", {})), overrides), log):
            die("role emit failed; see build.log")
    else:
        assets = out / "assets"
        print("== emit", file=sys.stderr)
        if run(nix([*base_args, "--out", str(assets)]), common, log):
            die("emit failed; see build.log")
        if objects and cell["arch"].startswith("gfx"):
            print("== AMD objects", file=sys.stderr)
            oenv = env_with(os.environ, objects.get("env", {}))
            oenv.update(object_overrides)
            oenv["PLOW_HSACO_CONFIG"] = str(assets / "plow_config.h")
            if run(nix(["bash", str(REPO / objects["script"]), str(out / "objects")]), oenv, log):
                die("object build failed; see build.log")

    ck = cell.get("checkpoint_dir")
    if ck:
        # A composed checkpoint (BF16 shards + fp8/ twins) replaces the emit's snapshot link.
        link = assets / "checkpoint"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ck)

    rec = {
        "recipe": str(Path(a.recipe).resolve()),
        "recipe_sha256": sha(Path(a.recipe)),
        "compiler_sha256": sha(plowc),
        "cell": cell,
        "overrides": overrides,
        "object_overrides": object_overrides,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hashes": {p.name: sha(p) for p in sorted(assets.glob("*")) if p.is_file() and p.suffix in (".pkt", ".cubin", ".elf", ".co", ".json", ".h")},
        "objects": {p.name: sha(p) for p in sorted((out / "objects").glob("*")) if p.is_file() and p.suffix in (".cubin", ".elf", ".co")} if (out / "objects").exists() else {},
    }
    (out / "build-record.json").write_text(json.dumps(rec, indent=1))
    print(f"built {assets}\nrecord {out / 'build-record.json'}", file=sys.stderr)
    # With the GPU on this box, select the exact-shape cuBLASLt algorithms now and packetize
    # them (leased); without it, plowc has already packetized the tune store's rows.
    if not a.no_probe and (assets / "build.json").exists() and gpu_matches(cell.get("gpu", "")):
        if (assets / "cublaslt_algos.jsonl").exists():
            print("probe: table already packetized from the tune store; skipping", file=sys.stderr)
        else:
            a.assets = str(assets)
            a.store_cell = a.store_cell or "h100"
            a.force = False
            a.label = None
            cmd_probe(a)


def gpu_matches(recipe_gpu: str) -> bool:
    """Whether nvidia-smi reports a GPU of the recipe's family (first word, e.g. `H100`)."""
    family = recipe_gpu.split()[0].upper() if recipe_gpu else ""
    if not family:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    return any(family in ln.upper() for ln in out.splitlines())


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
    packet_env(r, assets, env)
    lt_table = assets / "cublaslt_algos.jsonl"
    env.update({
        "VLLM_VENV": bench.get("vllm_venv", "/opt/pytorch"),
        "HF_HOME": str(out / "hf-home"),
        "IN_LENS": a.in_lens or bench.get("in_lens", "128 1024 4096"),
        "CONCS": a.concs or bench.get("concs", "1"),
        "NPROMPT": str(getattr(a, "nprompt", None) or bench.get("nprompt", 32)),
        "OUTLEN": str(bench.get("outlen", 128)),
        "BENCH_BACKEND": bench.get("backend", "openai"),
        "BENCH_EXTRA_ARGS": f"--num-warmups {bench.get('warmups', 16)} --seed {bench.get('seed', 42)}",
        "GATE_PROMPT": bench["gate_prompt"],
        "PLOWRT_BIN": str(private),
        "OUTDIR": str(out / "client"),
        "LOG": str(out / "server.log"),
        "SERVE_EXTRA_ARGS": serve.get("extra_args", ""),
        "DATASET_ARGS": getattr(a, "dataset_args", None) or bench.get("dataset_args", ""),
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
    # `--quiet-lock`: the bench holds this file lock exclusively for its whole session, INSIDE the
    # GPU lease (lease first, then lock, everywhere — the other order deadlocks against a run that
    # already holds the GPU). Builds take the same lock shared through scripts/bench/quiets.sh, so
    # no compile overlaps a measurement; quietx.sh gates new builds while this session waits.
    quiet = [str(REPO / "scripts" / "bench" / "quietx.sh"), a.quiet_lock] if getattr(a, "quiet_lock", None) else []
    lines.append("exec " + " ".join(shlex.quote(x) for x in [*quiet,
        str(BENCH), str(assets), str(bench.get("port", 8765)), model_id, bench["tokenizer"], str(bench.get("ready_s", 1200))]))
    wrapper.write_text("\n".join(lines) + "\n")
    wrapper.chmod(0o755)
    cmd = [str(GPULEASE), "-n", str(cell.get("n_gpu", 1)), label, str(wrapper)]
    log = out / "run.log"
    log.write_bytes(b"")
    rc = run(cmd, dict(os.environ), log)
    text = log.read_text(errors="replace")
    peak = {}
    for ln in text.splitlines():
        f = ln.split(",")
        if f[0] == MEM_COL and len(f) == 4:
            peak[(f[1], f[2])] = f[3]
    rows = [ln for ln in text.splitlines() if ln[:1].isdigit() and ln.count(",") == 12]
    rows = [ln + "," + peak.get(tuple(ln.split(",")[:2]), "") for ln in rows]
    (out / "results.csv").write_text(f"{CSV_HEADER},{MEM_COL}\n" + "\n".join(rows) + ("\n" if rows else ""))
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
        "protocol": {k: env[k] for k in ("IN_LENS", "CONCS", "NPROMPT", "OUTLEN", "BENCH_BACKEND", "BENCH_EXTRA_ARGS", "DATASET_ARGS")},
        "quiet_lock": getattr(a, "quiet_lock", None),
        "serve_env": serve.get("env", {}),
        "overrides": overrides,
        "lt_algos": {
            "pinned": env.get("PLOW_LT_ALGOS"),
            "rows": sum(1 for ln in lt_table.read_text().splitlines() if ln.strip()) if lt_table.is_file() else 0,
            "written": env.get("PLOW_LT_ALGOS_WRITE"),
        },
        "hashes": {p.name: sha(p) for p in sorted(assets.glob("*")) if p.is_file() and p.suffix in (".pkt", ".cubin", ".elf", ".co")},
        "rows": len(rows),
        "bench_rc": rc,
    }
    (out / "run-record.json").write_text(json.dumps(rec, indent=1))
    print(f"{CSV_HEADER},{MEM_COL}")
    print("\n".join(rows))
    if not rec["gate"]:
        die("coherence gate did not pass; numbers above are not evidence")
    if rec["contended"]:
        print("campaign: GPU was contended -- recorded as provisional", file=sys.stderr)
    if a.reference or r.get("reference"):
        compare(out / "results.csv", Path(a.reference or REPO / r["reference"]["csv"]))
    if generate_roofline_report is not None:
        try:
            print("\n" + generate_roofline_report(Path(a.recipe), out / "results.csv"))
        except Exception as e:
            print(f"campaign: roofline report skipped: {e}", file=sys.stderr)


def packet_env(r: dict, assets: Path, env: dict) -> None:
    # A `build` places the segment/role objects beside the assets; the serve-side mirror of
    # the emit classing needs that directory and must not be typed by hand.
    objects = assets.parent / "objects"
    if "objects" in r and "PLOW_PF_SEG_DIR" not in env and objects.is_dir():
        env["PLOW_PF_SEG_DIR"] = str(objects)
    # A `probe` (or a prior write) leaves the exact-shape cuBLASLt algorithm table beside the
    # packet; serving with it pins every Lt shape after AlgoCheck instead of re-timing at load.
    lt_table = assets / "cublaslt_algos.jsonl"
    if lt_table.is_file() and "PLOW_LT_ALGOS" not in env and "PLOW_LT_ALGOS_WRITE" not in env:
        env["PLOW_LT_ALGOS"] = str(lt_table)


# ---------------------------------------------------------------- serve
def cmd_serve(a: argparse.Namespace) -> None:
    """Production `plowrt serve` of a recipe-built packet, in the foreground: the env `bench`
    serves the profile with, minus the matched ladder's prefix-cache pin. No lease, no client."""
    r = load(a.recipe)
    serve = dict(r.get("serve", {}))
    profiles = r.get("bench", {}).get("profiles", {})
    if a.profile not in profiles:
        die(f"recipe has no [bench.profiles.{a.profile}]; have {sorted(profiles)}")
    env = {k: str(v) for k, v in serve.get("env", {}).items()}
    # [serve.env] pins the cache off so the ladder matches vLLM's --no-enable-prefix-caching;
    # production takes plowrt's default (on) unless --env sets it.
    env.pop("PLOW_PREFIX_CACHE", None)
    env.update({k: str(v) for k, v in profiles[a.profile].get("serve_env", {}).items()})
    env.update(dict(kv.split("=", 1) for kv in (a.env or [])))
    assets = Path(a.assets).resolve()
    if not (assets / "model.pkt").exists():
        die(f"{assets}/model.pkt missing")
    # As in `bench`: the shell's exports are visible to packet_env's guards.
    env = env_with(os.environ, env)
    packet_env(r, assets, env)
    plowrt = Path(a.plowrt or serve.get("plowrt", str(REPO / "target" / "release" / "plowrt"))).resolve()
    cmd = [str(plowrt), "serve", "--assets", str(assets), "--port", str(a.port), *shlex.split(serve.get("extra_args", ""))]
    shown = sorted((k, v) for k, v in env.items() if os.environ.get(k) != v)
    print(" ".join(f"{k}={shlex.quote(v)}" for k, v in shown) + " " + shlex.join(cmd), file=sys.stderr)
    if a.dry_run:
        return
    os.execvpe(cmd[0], cmd, env)


# ---------------------------------------------------------------- probe
def cmd_probe(a: argparse.Namespace) -> None:
    """Serve the packet once under the lease with `--lt-algos-write`, answer the coherence
    gate, and stop. Leaves `<assets>/cublaslt_algos.jsonl` (consumed by `bench`) and copies it
    into the tune store, so an emit on a GPU-less host can packetize the same selection."""
    r = load(a.recipe)
    cell, bench, serve = r["cell"], r["bench"], dict(r.get("serve", {}))
    assets = Path(a.assets).resolve()
    if not (assets / "model.pkt").exists():
        die(f"{assets}/model.pkt missing")
    out = assets.parent / "probe"
    out.mkdir(exist_ok=True)
    table = assets / "cublaslt_algos.jsonl"
    if table.exists() and not a.force:
        die(f"{table} exists; pass --force to re-probe")
    plowrt = Path(serve.get("plowrt", str(REPO / "target" / "release" / "plowrt"))).resolve()
    private = out / "plowrt"
    private.write_bytes(plowrt.read_bytes())
    private.chmod(0o755)
    env = env_with(os.environ, serve.get("env", {}))
    objects = assets.parent / "objects"
    if "objects" in r and "PLOW_LT_ALGOS_WRITE" not in env and objects.is_dir():
        env.setdefault("PLOW_PF_SEG_DIR", str(objects))
    env.update(dict(kv.split("=", 1) for kv in (a.env or [])))
    env["PLOW_LT_ALGOS_WRITE"] = str(table)
    env.pop("PLOW_LT_ALGOS", None)
    port = str(bench.get("port", 8765))
    model_id = bench.get("model_id") or cell["revision"]
    gate = json.dumps({"model": model_id, "prompt": bench["gate_prompt"], "max_tokens": 16, "temperature": 0})
    script = out / "probe.sh"
    lines = ["#!/usr/bin/env bash", "set -uo pipefail"]
    for k, v in sorted(env.items()):
        if k in os.environ and os.environ[k] == v:
            continue
        lines.append(f"export {k}={shlex.quote(v)}")
    lines += [
        f"setsid {shlex.quote(str(private))} serve --assets {shlex.quote(str(assets))} --port {port} >{shlex.quote(str(out / 'server.log'))} 2>&1 &",
        "SRV=$!",
        "trap 'kill -TERM -\"$SRV\" 2>/dev/null; sleep 2; kill -KILL -\"$SRV\" 2>/dev/null' EXIT",
        f"for i in $(seq 1 {bench.get('ready_s', 1200)}); do kill -0 $SRV 2>/dev/null || {{ echo 'server died'; tail -20 {shlex.quote(str(out / 'server.log'))}; exit 1; }}; "
        f"curl -sf --max-time 2 http://127.0.0.1:{port}/v1/models >/dev/null 2>&1 && break; sleep 1; done",
        f"curl -s --max-time 300 http://127.0.0.1:{port}/v1/completions -H 'Content-Type: application/json' --data-binary {shlex.quote(gate)} | grep -qi paris || {{ echo 'gate FAIL'; exit 1; }}",
        "echo 'gate PASS'",
    ]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    log = out / "probe.log"
    log.write_bytes(b"")
    label = a.label or f"{cell['name']}-probe"
    rc = run([str(GPULEASE), "-n", str(cell.get("n_gpu", 1)), label, str(script)], dict(os.environ), log)
    text = log.read_text(errors="replace")
    if rc != 0 or "gate PASS" not in text or not table.is_file():
        die("probe failed; see probe/probe.log and probe/server.log")
    rows = [ln for ln in table.read_text().splitlines() if ln.strip()]
    store = REPO / "tuning" / "nvidia" / cell["arch"].replace("_", "") / a.store_cell / "cublaslt_algos.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if store.exists():
        for ln in store.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                seen.add((d["m"], d["n"], d["k"], d["dtype"], d["gpu"]))
    added = 0
    with open(store, "a") as f:
        for ln in rows:
            d = json.loads(ln)
            key = (d["m"], d["n"], d["k"], d["dtype"], d["gpu"])
            if key in seen:
                continue
            seen.add(key)
            f.write(ln + "\n")
            added += 1
    print(f"probe: {len(rows)} shape(s) selected -> {table}\n       {added} new row(s) -> {store}", file=sys.stderr)


# ---------------------------------------------------------------- cert
def _samples(run_dir: Path) -> dict:
    """Per-request samples per (input_len, concurrency, metric) from a bench run's client JSONs."""
    out = {}
    for f in sorted((run_dir / "client").glob("in*_c*.json")):
        d = json.loads(f.read_text())
        key = (int(d["input_lens"][0]), int(d["max_concurrency"] or 1))
        ttft = [x * 1e3 for x in d["ttfts"]]
        tpot = [sum(i) / len(i) * 1e3 for i in d["itls"] if i]
        out[(*key, "ttft_ms")] = ttft
        out[(*key, "tpot_ms")] = tpot
    return out


def _stats(xs: list) -> dict:
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    dev = sorted(abs(x - med) for x in s)
    mad = dev[n // 2] if n % 2 else (dev[n // 2 - 1] + dev[n // 2]) / 2
    return {"median": med, "mad": mad, "n": n, "p95": s[min(n - 1, int(0.95 * n))]}


def cmd_cert(a: argparse.Namespace) -> None:
    """Checkpoint P certificate for a default flip from four bench runs (ctrl, ctrl2, treat,
    treat2): the ledger entries are the runs' per-request samples, the request names every cell
    as a touched serving rung, and `scripts/perf_cert.py make` runs the verifier."""
    runs = {arm: Path(getattr(a, arm)).resolve() for arm in ("ctrl", "ctrl2", "treat", "treat2")}
    recs = {arm: json.loads((p / "run-record.json").read_text()) for arm, p in runs.items()}
    samples = {arm: _samples(p) for arm, p in runs.items()}
    cells = sorted(set.intersection(*(set(s) for s in samples.values())))
    if not cells:
        die("the four runs share no (input_len, concurrency, metric) cell")
    delta = dict(kv.split("=", 1) for kv in (a.knob_delta or []))
    gpu = recs["treat"].get("gpu", {})
    hardware = {"box": f"1x{gpu.get('name', 'GPU')}", "driver": gpu.get("driver_version"), "firmware": None, "cuda": None}
    work = Path(a.out).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ledger, touched, serving = [], [], []
    for (L, C, metric) in cells:
        rung = {"digest": f"{a.cell}/serve/in{L}-c{C}-out128", "prior": 0, "role": "serve", "rows": L, "topology": f"C{C}"}
        ids = {arm: f"{a.job}:in{L}-c{C}:{metric}:{arm}" for arm in runs}
        for arm in ("ctrl", "ctrl2", "treat", "treat2"):
            xs = samples[arm][(L, C, metric)]
            e = {
                "id": ids[arm], "job": a.job, "metric": metric, "better": "lower",
                "rung": rung, "samples": xs, "stats": _stats(xs),
                "knob_delta": delta if arm.startswith("treat") else {},
                "recipe_digest": recs[arm]["hashes"].get("model.pkt", "")[:16],
                "hardware": hardware, "harness": "campaign-bench",
                "date": recs[arm]["utc"][:10],
            }
            if arm.startswith("treat"):
                e["control_of"] = ids["ctrl"]
                e["repeat_control_of"] = ids["ctrl2"]
            ledger.append(e)
        t = {"rung": rung["digest"], "treat": ids["treat"]}
        # `--neutral tpot_ms` (every rung) or `--neutral ttft_ms@in128` (one input length).
        neutral = any(
            spec == metric or spec == f"{metric}@in{L}" for spec in (a.neutral or [])
        )
        if neutral:
            t["neutral_evidence"] = [a.neutral_evidence or "unchanged by the flip"]
        touched.append(t)
        serving += [ids["treat"], ids["treat2"]]
    gates = all(recs[arm].get("gate") for arm in runs)
    request = {
        "touched": touched,
        "untouched": [],
        "tier4": True,
        "serving": serving,
        "numeric": bool(a.numeric),
        "facts": [{"kind": "gate", "pass": gates,
                   "evidence": "coherence gate PASS on ctrl, ctrl2, treat, treat2 (run-record.json of each)"}]
                 + [{"kind": "note", "pass": True, "evidence": f} for f in (a.fact or [])],
    }
    (work / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    (work / "request.json").write_text(json.dumps(request, indent=1))
    cert = REPO / "perf-certs" / f"{a.knob}.json"
    cmd = ["python3", str(REPO / "scripts" / "perf_cert.py"), "make", "--knob", a.knob,
           "--request", str(work / "request.json"), "--ledger", str(work / "ledger.jsonl"), "--out", str(cert)]
    print("  $", " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"cert: ledger {len(ledger)} entries, {len(touched)} touched rungs -> {cert} (rc={rc})", file=sys.stderr)
    if rc != 0:
        sys.exit(rc)


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
    if getattr(a, "roofline", False):
        recipe_path = None
        if getattr(a, "recipe", None):
            recipe_path = Path(a.recipe)
        else:
            rec_file = Path(a.results).parent / "run-record.json"
            if rec_file.is_file():
                try:
                    rec = json.loads(rec_file.read_text())
                    if "recipe" in rec:
                        recipe_path = Path(rec["recipe"])
                except Exception:
                    pass
        if recipe_path and recipe_path.is_file():
            print("\n" + generate_roofline_report(recipe_path, Path(a.results)))
        else:
            print("\n(roofline report: pass --recipe <recipe.toml> or place run-record.json beside results to calculate % roofline achieved)")


def cmd_roofline(a: argparse.Namespace) -> None:
    print(generate_roofline_report(Path(a.recipe), Path(a.results) if a.results else None))


# ---------------------------------------------------------------- loop (closed-loop bring-up)
def cmd_loop(a: argparse.Namespace) -> None:
    """Closed-loop bring-up and optimization:
    Doctor check -> Build/Emit -> Probe BLAS -> Lease & Bench -> Roofline & Baseline Compare -> Bottleneck Identification.
    """
    r = load(a.recipe)
    cell = r["cell"]
    recipe_path = Path(a.recipe).resolve()
    base_out = Path(a.out or f"/tmp/plow-campaign/{cell['name']}").resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    build_dir = base_out / "build"
    bench_dir = base_out / "bench"

    print(f"=== Starting Optimization Loop: {cell['name']} ===", file=sys.stderr)

    # 1. Build
    print(f"[1/5] Building {cell['name']}...", file=sys.stderr)
    build_args = argparse.Namespace(
        recipe=str(recipe_path),
        out=str(build_dir),
        env=a.env or [],
        no_probe=a.no_probe,
        store_cell=getattr(a, "store_cell", None),
    )
    if not (build_dir / "assets" / "model.pkt").exists():
        cmd_build(build_args)
    else:
        print(f"  Reusing existing build at {build_dir}", file=sys.stderr)

    assets_dir = build_dir / "assets"
    objects_dir = build_dir / "objects"

    # 2. Pre-flight doctor check
    if not a.skip_doctor:
        print("[2/5] Pre-flight doctor check...", file=sys.stderr)
        doc_cmd = [
            "bash",
            str(REPO / "scripts" / "bench" / "plowbench-doctor.sh"),
            str(assets_dir),
            str(objects_dir) if objects_dir.is_dir() else "",
            "",
            cell.get("arch", ""),
        ]
        rc = subprocess.run(nix(doc_cmd), cwd=REPO).returncode
        if rc != 0:
            die("pre-flight doctor check FAILED. Fix environment or artifacts before continuing.")
        print("  Doctor check passed.", file=sys.stderr)

    # 3. Probe (if applicable and not skipped)
    if not a.no_probe and not (assets_dir / "cublaslt_algos.jsonl").exists():
        print("[3/5] Probing cuBLASLt algorithms...", file=sys.stderr)
        probe_args = argparse.Namespace(
            recipe=str(recipe_path),
            assets=str(assets_dir),
            store_cell=getattr(a, "store_cell", "h100") or "h100",
            label=f"{cell['name']}-probe",
            force=False,
            env=a.env or [],
        )
        try:
            cmd_probe(probe_args)
        except Exception as e:
            print(f"  Probe skipped / failed: {e}", file=sys.stderr)
    else:
        print("[3/5] Probe step skipped or already present.", file=sys.stderr)

    # 4. Bench
    print("[4/5] Leasing GPU & running benchmark...", file=sys.stderr)
    bench_args = argparse.Namespace(
        recipe=str(recipe_path),
        assets=str(assets_dir),
        out=str(bench_dir),
        concs=a.concs,
        in_lens=a.in_lens,
        label=a.label,
        reference=a.reference,
        env=a.env or [],
        profile=a.profile,
    )
    cmd_bench(bench_args)

    # 5. Roofline & Bottleneck Diagnosis
    print("\n[5/5] Roofline Analysis & Bottleneck Identification:", file=sys.stderr)
    results_csv = bench_dir / "results.csv"
    if results_csv.is_file() and generate_roofline_report is not None:
        try:
            print(generate_roofline_report(recipe_path, results_csv))
        except Exception as e:
            print(f"campaign: roofline report skipped: {e}", file=sys.stderr)
    print(f"\nOptimization loop complete. Artifacts in: {base_out}")


# ---------------------------------------------------------------- sweep
def cmd_sweep(a: argparse.Namespace) -> None:
    """Sweep a parameter across candidate values, benchmark each, compare against baseline, and rank."""
    r = load(a.recipe)
    cell = r["cell"]
    recipe_path = Path(a.recipe).resolve()
    base_out = Path(a.out or f"/tmp/plow-campaign/sweep-{cell['name']}-{a.param}").resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    values = [v.strip() for v in a.values.split(",") if v.strip()]
    if not values:
        die("no values specified for sweep")

    print(f"=== Parameter Sweep: {a.param} across {values} ===", file=sys.stderr)
    sweep_results = {}
    for val in values:
        run_label = f"{a.param}_{val}"
        run_out = base_out / run_label
        print(f"\n--- Arm: {a.param}={val} ---", file=sys.stderr)
        build_dir = run_out / "build"
        bench_dir = run_out / "bench"
        env_override = [f"{a.param}={val}", *(a.env or [])]

        build_args = argparse.Namespace(
            recipe=str(recipe_path),
            out=str(build_dir),
            env=env_override,
            no_probe=True,
            store_cell=None,
        )
        if not (build_dir / "assets" / "model.pkt").exists():
            cmd_build(build_args)

        bench_args = argparse.Namespace(
            recipe=str(recipe_path),
            assets=str(build_dir / "assets"),
            out=str(bench_dir),
            concs=a.concs,
            in_lens=a.in_lens,
            label=f"{cell['name']}-{run_label}",
            reference=a.reference,
            env=env_override,
            profile=a.profile,
        )
        cmd_bench(bench_args)
        if (bench_dir / "results.csv").is_file():
            sweep_results[val] = read_rows(bench_dir / "results.csv")

    # Summary table across sweep values
    print(f"\n=== Sweep Summary: {a.param} ===")
    ref_rows = read_rows(Path(a.reference or REPO / r["reference"]["csv"])) if (a.reference or r.get("reference")) else {}
    for val, rows in sweep_results.items():
        print(f"\nConfiguration: {a.param}={val}")
        for k in sorted(rows):
            cur = rows[k]
            ref = ref_rows.get(k)
            tpot_s = f"{float(cur['tpot_ms']):.2f} ms"
            ttft_s = f"{float(cur['ttft_ms']):.2f} ms"
            vs_ref = ""
            if ref:
                ratio_tpot = float(cur['tpot_ms']) / float(ref['tpot_ms'])
                ratio_ttft = float(cur['ttft_ms']) / float(ref['ttft_ms'])
                vs_ref = f"(vs ref: TTFT {ratio_ttft:.2f}x, TPOT {ratio_tpot:.2f}x)"
            print(f"  in={k[0]:<5} C={k[1]:<2} | TTFT: {ttft_s:>9} | TPOT: {tpot_s:>8} | {vs_ref}")


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
    cols = [*CSV_HEADER.split(","), MEM_COL]
    new = not ledger.exists()
    if not new:
        # A ledger from before the memory column: widen it in place, old rows get an empty field.
        with open(ledger, newline="") as f:
            old_rows = list(csv.reader(f))
        if old_rows and MEM_COL not in old_rows[0]:
            with open(ledger, "w", newline="") as f:
                csv.writer(f).writerows([old_rows[0] + [MEM_COL], *[r + [""] for r in old_rows[1:]]])
    with open(ledger, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "commit", "label", "provisional", "pkt_sha", "note", *cols])
        for row in read_rows(res).values():
            w.writerow([rec["utc"], rec["commit"][:12], rec["label"], int(bool(rec.get("contended"))),
                        rec["hashes"].get("model.pkt", "")[:16], a.note, *[row.get(c) or "" for c in cols]])
    print(f"appended {len(read_rows(res))} row(s) to {ledger}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    br = sp.add_parser("block-roofline", help="CPU-only floor from one modular decode packet")
    br.add_argument("recipe"); br.add_argument("--packet", required=True)
    br.add_argument("--program", type=int, default=1, help="compiled row width, as in plowrt disasm")
    br.add_argument("--ctx", type=int, required=True); br.add_argument("--out", required=True)
    br.add_argument("--plowrt", default=str(REPO / "target/release/plowrt"))
    br.add_argument("--router-table", help="captured rank act.tab.bin; uses actual selected-expert union")
    br.set_defaults(f=cmd_block_roofline)
    bb = sp.add_parser("block-bench", help="freeze, preflight and queue a numerically gated modular block")
    bb.add_argument("recipe"); bb.add_argument("--packet", required=True); bb.add_argument("--objects", required=True)
    bb.add_argument("--inputs", required=True); bb.add_argument("--checkpoint", required=True)
    bb.add_argument("--ctx", type=int, required=True); bb.add_argument("--out", required=True)
    bb.add_argument("--repeat", type=int, default=30); bb.add_argument("--warmup", type=int, default=5)
    bb.add_argument("--plowrt", default=str(REPO / "target/release/plowrt"))
    bb.add_argument("--queue", default="/tmp/plow-gpuq"); bb.set_defaults(f=cmd_block_bench)
    bb.add_argument("--trace", action="store_true", help="record device traces; instrumented timings are diagnostic only")
    bb.add_argument("--dstep-log", action="store_true", help="log host phase timings; diagnostic only, not A/B evidence")
    ba = sp.add_parser("block-ab", help="freeze and compare four block arms in one quiet GPU lease")
    ba.add_argument("recipe"); ba.add_argument("--control-build", required=True)
    ba.add_argument("--treatment-build", required=True); ba.add_argument("--inputs", required=True)
    ba.add_argument("--checkpoint", required=True); ba.add_argument("--ctx", type=int, required=True)
    ba.add_argument("--out", required=True); ba.add_argument("--note", required=True)
    ba.add_argument("--repeat", type=int, default=300); ba.add_argument("--warmup", type=int, default=30)
    ba.add_argument("--plowrt", default=str(REPO / "target/release/plowrt"))
    ba.add_argument("--queue", default="/tmp/plow-gpuq")
    ba.add_argument("--quiet-lock", default="/tmp/plow-campaign-quiet.lock")
    ba.add_argument("--require-bitwise", action="store_true", help="reject any control/candidate output difference")
    ba.add_argument("--routed-reference", help="passed pinned routed audit; validate atomic repeats before scoring")
    ba.set_defaults(f=cmd_block_ab)
    sb = sp.add_parser("serve-bench", help="freeze and queue plow or vLLM 0.29 with the same Docker client")
    sb.add_argument("recipe"); sb.add_argument("--server", choices=["plow", "vllm"], required=True)
    sb.add_argument("--assets", required=True); sb.add_argument("--objects", required=True)
    sb.add_argument("--out", required=True); sb.add_argument("--in-lens"); sb.add_argument("--concs")
    sb.add_argument("--nprompt", type=int); sb.add_argument("--env", action="append", metavar="K=V")
    sb.add_argument("--plowrt", default=str(REPO / "target/release/plowrt"))
    sb.add_argument("--queue", default="/tmp/plow-gpuq"); sb.add_argument("--dry-run", action="store_true")
    sb.add_argument("--quality-lens", help="comma-separated exact needle contexts, run before timing")
    sb.set_defaults(f=cmd_serve_bench)
    b = sp.add_parser("build"); b.add_argument("recipe"); b.add_argument("--out", required=True)
    b.add_argument("--env", action="append", metavar="K=V", help="one-variable override for the emit env; recorded")
    b.add_argument("--object-env", action="append", metavar="K=V", help="object-build-only override; recorded")
    b.add_argument("--no-probe", action="store_true", help="skip the leased cuBLASLt algorithm probe even with the GPU present")
    b.add_argument("--store-cell", help="tune-store cell for the probe (default h100)")
    b.add_argument("--hf-dir", help="checkpoint snapshot on this host, replacing [cell].hf_dir; recorded")
    b.set_defaults(f=cmd_build)
    s = sp.add_parser("serve"); s.add_argument("recipe"); s.add_argument("--assets", required=True)
    s.add_argument("--profile", required=True, help="serving policy from [bench.profiles.*] (e.g. realtime, high_concurrency)")
    s.add_argument("--port", type=int, default=8080); s.add_argument("--plowrt", help="plowrt binary (default target/release/plowrt)")
    s.add_argument("--env", action="append", metavar="K=V", help="host-specific or policy override, e.g. PLOW_LIBCUDA=...")
    s.add_argument("--dry-run", action="store_true", help="print the env and command, do not start")
    s.set_defaults(f=cmd_serve)
    n = sp.add_parser("bench"); n.add_argument("recipe"); n.add_argument("--assets", required=True); n.add_argument("--out", required=True)
    n.add_argument("--concs"); n.add_argument("--in-lens"); n.add_argument("--label"); n.add_argument("--reference")
    n.add_argument("--nprompt", type=int, help="prompts per cell, overriding the recipe/profile")
    n.add_argument("--dataset-args", help="replaces the client's random-dataset block (see bench_plowrt_serve.sh DATASET_ARGS); recorded")
    n.add_argument("--quiet-lock", metavar="FILE", help="hold this lock exclusively for the bench session, inside the GPU lease (scripts/bench/quietx.sh; builds take it shared via quiets.sh)")
    n.add_argument("--env", action="append", metavar="K=V", help="one-variable override for the server env; recorded")
    n.add_argument("--profile", help="named workload from [bench.profiles.*] (e.g. realtime, throughput)")
    n.set_defaults(f=cmd_bench)
    pr = sp.add_parser("probe"); pr.add_argument("recipe"); pr.add_argument("--assets", required=True)
    pr.add_argument("--store-cell", default="h100", help="tune-store cell under tuning/nvidia/<arch>/")
    pr.add_argument("--label"); pr.add_argument("--force", action="store_true")
    pr.add_argument("--env", action="append", metavar="K=V"); pr.set_defaults(f=cmd_probe)
    ce = sp.add_parser("cert"); ce.add_argument("--knob", required=True); ce.add_argument("--job", required=True)
    ce.add_argument("--cell", required=True, help="cell name used in rung digests, e.g. gemma4-12b.h100.bf16")
    for arm in ("ctrl", "ctrl2", "treat", "treat2"):
        ce.add_argument(f"--{arm}", required=True, help=f"bench run dir of the {arm} arm")
    ce.add_argument("--knob-delta", action="append", metavar="K=V"); ce.add_argument("--numeric", action="store_true")
    ce.add_argument("--neutral", action="append", metavar="METRIC", help="metric expected unchanged (e.g. tpot_ms)")
    ce.add_argument("--neutral-evidence"); ce.add_argument("--fact", action="append")
    ce.add_argument("--out", required=True, help="work dir for ledger.jsonl and request.json"); ce.set_defaults(f=cmd_cert)
    c = sp.add_parser("compare"); c.add_argument("results"); c.add_argument("reference")
    c.add_argument("--roofline", action="store_true", help="display roofline analysis alongside comparison")
    c.add_argument("--recipe", help="optional recipe path to use for roofline geometry")
    c.set_defaults(f=cmd_compare)
    rf = sp.add_parser("roofline"); rf.add_argument("recipe"); rf.add_argument("--results"); rf.set_defaults(f=cmd_roofline)
    lp = sp.add_parser("loop"); lp.add_argument("recipe"); lp.add_argument("--out"); lp.add_argument("--profile")
    lp.add_argument("--concs"); lp.add_argument("--in-lens"); lp.add_argument("--label"); lp.add_argument("--reference")
    lp.add_argument("--env", action="append", metavar="K=V"); lp.add_argument("--no-probe", action="store_true")
    lp.add_argument("--skip-doctor", action="store_true"); lp.set_defaults(f=cmd_loop)
    sw = sp.add_parser("sweep"); sw.add_argument("recipe"); sw.add_argument("--param", required=True)
    sw.add_argument("--values", required=True, help="comma-separated list of parameter values")
    sw.add_argument("--out"); sw.add_argument("--profile"); sw.add_argument("--concs"); sw.add_argument("--in-lens")
    sw.add_argument("--reference"); sw.add_argument("--env", action="append", metavar="K=V")
    sw.set_defaults(f=cmd_sweep)
    l = sp.add_parser("ledger"); l.add_argument("results"); l.add_argument("--cell", required=True); l.add_argument("--note", required=True)
    l.add_argument("--provisional", action="store_true"); l.set_defaults(f=cmd_ledger)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
