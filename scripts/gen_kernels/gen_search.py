#!/usr/bin/env python3
"""Kernel-generator search for one block: build generator candidates (TileLang, Triton) over a
schedule grid, time them and the current plow kernel standalone, check numerics against the
block's reference, and print the block-vs-roofline table with the best generated candidate as a
reference row (docs/bringup/agent-tools.md HARD RULE, docs/runtime/kernel-generators.md).

A block is a spec file `spec_<name>.py` beside this driver exporting SPEC (fields: SPEC_FIELDS
below, example: spec_s3gen_cfm_attn.py). SPEC arguments take the name or a path.

  gen_search.py list
  gen_search.py plan SPEC [--classes a,b] [--gens tilelang,triton]       CPU: grid, floors
  gen_search.py build-current SPEC --out DIR                             CPU: the plow probe lib
  gen_search.py submit SPEC --out DIR [run options]                      queues `run` (gpuq.py)
  gen_search.py run SPEC --out DIR [--classes ..] [--gens ..] [--budget-s 780] [--force]
  gen_search.py report SPEC --out DIR [--ceilings ceilings.json] [--all] CPU: the table

`run` needs one leased GPU and a Python with torch and the spec's generators (`PYREF`, the
catalog's interpreter); go through `submit`, which queues it under a 15 min timeout and forwards
the compiler environment (PATH, CUDA_HOME, NVCC_PREPEND_FLAGS, LD_LIBRARY_PATH, TL_*). Rows land
in DIR/<name>.results.jsonl as they are measured; a rerun resumes after the rows already there,
so a grid larger than one job's budget is a second `submit`. DIR is campaign scratch, never the
repo. Timing: a CUDA graph over rotating cold inputs larger than L2, median us per call.
"""
import argparse
import ast
import importlib.metadata
import importlib.util
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import gen_bench  # noqa: E402

# name: (required, what)
SPEC_FIELDS = {
    "name": (True, "the file's <name>"),
    "op": (True, "op family and geometry, free text for the table"),
    "dtype": (True, "arithmetic as served, free text"),
    "classes": (True, "{class name: shape dict}; optional 'step': {'us', 'calls'} per class "
                      "(full step time and this block's calls in it) for the in-model verdict"),
    "cost": (True, "cost(cls) -> {'bytes', 'flops', 'peak': bf16|fp8|tf32|fp32, 'passes'=1}"),
    "inputs": (True, "inputs(cls, seed) -> {name: cuda tensor}"),
    "rotate": (True, "input names regenerated per cold set (the streamed operands)"),
    "out": (True, "out(cls) -> (shape, torch dtype name)"),
    "reference": (True, "reference(cls, inputs) -> fp64 tensor of the output's shape"),
    "tol": (True, "{'rel_l2': bound}; 'pcm': True adds the int16-RMS PCM bound (output is "
                  "full-scale float PCM, cases on dim 0)"),
    "generators": (True, "{generator: {'grid': {key: [values]} or 'sweep': [cfg], "
                         "'build'(cfg, cls) -> kernel, 'bind'(kernel, cls, inputs, out) -> "
                         "zero-arg launch; optional 'valid'(cfg, cls), 'static_shapes': True "
                         "(build per class)}}"),
    "current": (False, "the plow kernel: {'label', 'lib': {'src', 'flags'}, "
                       "'bind'(ctypes lib, cls, inputs, out) -> launch} or {'label', "
                       "'measured_us': {class: us}, 'source'}"),
}
FORWARD_ENV = ("PATH", "CUDA_HOME", "NVCC_PREPEND_FLAGS", "LD_LIBRARY_PATH")
JOB_SECONDS = 900


def load_spec(ref):
    path = Path(ref) if ref.endswith(".py") else HERE / f"spec_{ref}.py"
    if not path.is_file():
        raise SystemExit(f"no spec {path}")
    mod_spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return check_spec(getattr(mod, "SPEC", None), path)


def check_spec(spec, path):
    def bad(msg):
        raise SystemExit(f"{path}: {msg}")
    if not isinstance(spec, dict):
        bad("no SPEC dict")
    for key, (required, what) in SPEC_FIELDS.items():
        if required and key not in spec:
            bad(f"SPEC lacks '{key}' ({what})")
    for key in spec:
        if key not in SPEC_FIELDS:
            bad(f"unknown SPEC field '{key}'")
    if f"spec_{spec['name']}.py" != Path(path).name:
        bad(f"name '{spec['name']}' does not match the file name")
    if not spec["classes"] or not all(isinstance(c, dict) for c in spec["classes"].values()):
        bad("'classes' must be a non-empty {name: dict}")
    if "rel_l2" not in spec["tol"]:
        bad("'tol' needs 'rel_l2'")
    if not spec["generators"]:
        bad("'generators' is empty")
    for gen, g in spec["generators"].items():
        if ("grid" in g) == ("sweep" in g):
            bad(f"generator '{gen}': exactly one of 'grid', 'sweep'")
        for fn in ("build", "bind"):
            if not callable(g.get(fn)):
                bad(f"generator '{gen}': '{fn}' must be callable")
    cur = spec.get("current")
    if cur is not None and ("lib" in cur) == ("measured_us" in cur):
        bad("'current': exactly one of 'lib' (+ 'bind'), 'measured_us' (+ 'source')")
    for name, cls in spec["classes"].items():
        cost = spec["cost"](cls)
        if not {"bytes", "flops", "peak"} <= set(cost):
            bad(f"cost({name}) needs bytes, flops, peak")
    return spec


def expand(gen):
    """Config dicts of a generator: the cartesian product of 'grid', or 'sweep' as written."""
    if "sweep" in gen:
        return [dict(c) for c in gen["sweep"]]
    keys = list(gen["grid"])
    return [dict(zip(keys, vals)) for vals in itertools.product(*(gen["grid"][k] for k in keys))]


def cfg_label(cfg):
    return " ".join(f"{k}{v}" for k, v in cfg.items())


def candidates(spec, classes, gens):
    """(generator, cfg, [class names]) in run order: every class of one config before the next."""
    out = []
    for gen in gens:
        g = spec["generators"][gen]
        for cfg in expand(g):
            names = [n for n in classes if g.get("valid", lambda *_: True)(cfg, spec["classes"][n])]
            if names:
                out.append((gen, cfg, names))
    return out


def pick(spec, a):
    classes = a.classes.split(",") if a.classes else list(spec["classes"])
    gens = a.gens.split(",") if a.gens else list(spec["generators"])
    for name, known in ((classes, spec["classes"]), (gens, spec["generators"])):
        for n in name:
            if n not in known:
                raise SystemExit(f"{spec['name']}: unknown '{n}' (have {', '.join(known)})")
    return classes, gens


def results_path(spec, out):
    return Path(out) / f"{spec['name']}.results.jsonl"


def read_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(r):
    return (r["class"], r["arm"], json.dumps(r["cfg"], sort_keys=True))


def summarize(spec, rows, ceil, classes=None):
    """One table row per class: current, best passing generated candidate, floor, verdict. A later
    row for the same (class, arm, cfg) replaces an earlier one."""
    last = {row_key(r): r for r in rows}
    cur = spec.get("current") or {}
    out = []
    for name in classes or spec["classes"]:
        cls = spec["classes"][name]
        mine = [r for r in last.values() if r["class"] == name]
        cur_us, note = None, ""
        if "measured_us" in cur:
            cur_us, note = cur["measured_us"].get(name), " (recorded)"
        else:
            got = [r for r in mine if r["arm"] == "current" and r.get("ok")]
            cur_us = got[0]["us"] if got else None
        ok = [r for r in mine if r["arm"] != "current" and r.get("ok")]
        best = min(ok, key=lambda r: r["us"]) if ok else None
        out.append({
            "block": f"{spec['op']}, {name}",
            "floor_us": gen_bench.floor_us(spec["cost"](cls), ceil),
            "cur_us": cur_us, "cur_note": note if cur_us is not None else "",
            "gen_us": best and best["us"],
            "gen_label": best and f"{best['arm']} {cfg_label(best['cfg'])}",
            "rel_l2": best and best["rel_l2"],
            "verdict": gen_bench.verdict(cur_us, best and best["us"], cls.get("step")),
            "tried": len([r for r in mine if r["arm"] != "current"]),
            "passed": len(ok),
        })
    return out


def report(spec, rows, ceil, classes=None, every=False):
    summary = summarize(spec, rows, ceil, classes)
    cur = spec.get("current") or {}
    lines = [f"{spec['name']}: {spec['op']}, {spec['dtype']}", "", gen_bench.table(summary), "",
             f"floor = max(bytes / {ceil['bw'] / 1e9:.0f} GB/s, passes x FLOPs / peak); peaks TF/s "
             + ", ".join(f"{k} {v / 1e12:g}" for k, v in sorted(ceil["peak"].items()))
             + f"; ceilings: {ceil['source']}",
             f"current: {cur.get('label', 'none declared')}"
             + (f"; recorded, not same-lease: {cur['source']}" if "measured_us" in cur else ""),
             "candidates passing numerics / tried: "
             + ", ".join(f"{s['block'].rsplit(', ', 1)[1]} {s['passed']}/{s['tried']}"
                         for s in summary)]
    if every:
        lines += ["", "| class | arm | config | us | rel-L2 | max/rms | pcm rms | status |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted({row_key(r): r for r in rows}.values(),
                        key=lambda r: (r["class"], r.get("us") or 1e30)):
            f = lambda v, p: "—" if v is None else format(v, p)
            lines.append(f"| {r['class']} | {r['arm']} | {cfg_label(r['cfg'])} | "
                         f"{f(r.get('us'), '.1f')} | {f(r.get('rel_l2'), '.1e')} | "
                         f"{f(r.get('max_rms'), '.1e')} | {f(r.get('pcm_rms'), '.1e')} | "
                         f"{'ok' if r.get('ok') else r.get('error', 'FAIL')[:60]} |")
    return "\n".join(lines)


def versions(gens):
    out = {}
    for name in ("torch", *gens):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def measure(spec, cls, bind, base, sets, out):
    """Numerics on the base inputs, then the cold-rotation time. Returns the row's fields."""
    import torch
    tol = spec["tol"]
    out.fill_(float("nan")) if out.is_floating_point() else out.zero_()
    bind(base, out)()
    torch.cuda.synchronize()
    ref = spec["reference"](cls, base)
    mx, l2 = gen_bench.rel_err(out, ref)
    row = {"rel_l2": l2, "max_rms": mx}
    ok = l2 == l2 and l2 <= tol["rel_l2"]
    if tol.get("pcm"):
        row["pcm_rms"] = gen_bench.pcm_rms(out, ref)
        ok = ok and row["pcm_rms"] <= gen_bench.PCM_RMS_BOUND
    row["us"] = gen_bench.graph_time([bind(s, out) for s in sets])
    row["ok"] = bool(ok)
    if not ok:
        row["error"] = "numerics over tol"
    return row


def run(spec, a):
    import torch
    classes, gens = pick(spec, a)
    path = results_path(spec, a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if a.force else {row_key(r) for r in read_rows(path)}
    t0 = time.monotonic()
    stamp = {"spec": spec["name"], "device": torch.cuda.get_device_name(), "versions": versions(gens)}
    data = {}

    def class_data(name):
        if name not in data:
            data.clear()  # one class's cold sets on the device at a time
            torch.cuda.empty_cache()
            cls = spec["classes"][name]
            base = spec["inputs"](cls, 0)
            nbytes = sum(base[k].numel() * base[k].element_size() for k in spec["rotate"])
            sets = [dict(base, **{k: v for k, v in spec["inputs"](cls, s + 1).items()
                                  if k in spec["rotate"]})
                    for s in range(gen_bench.cold_sets(nbytes))]
            shape, dtype = spec["out"](cls)
            data[name] = (base, sets, torch.empty(shape, device="cuda", dtype=getattr(torch, dtype)))
        return data[name]

    def emit(name, arm, cfg, fn=None, err=None):
        row = dict(stamp, **{"class": name, "arm": arm, "cfg": cfg})
        try:
            if err is not None:
                raise err
            row.update(fn())
        except Exception as e:  # a candidate that does not build or launch is a result
            row.update(ok=False, error=f"{type(e).__name__}: {e}".splitlines()[0][:200])
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    todo = []
    cur = spec.get("current") or {}
    if "lib" in cur:
        names = [n for n in classes if (n, "current", "{}") not in done]
        if names:
            todo.append(("current", {}, names))
    for gen, cfg, names in candidates(spec, classes, gens):
        names = [n for n in names if (n, gen, json.dumps(cfg, sort_keys=True)) not in done]
        if names:
            todo.append((gen, cfg, names))
    left = 0
    for i, (arm, cfg, names) in enumerate(todo):
        if time.monotonic() - t0 > a.budget_s:
            left = len(todo) - i
            break
        if arm == "current":
            import ctypes
            lib_path = a.current_lib or Path(a.out) / f"{spec['name']}.current.so"
            if not Path(lib_path).is_file():
                raise SystemExit(f"{lib_path}: run build-current first (CPU, outside the queue)")
            lib = ctypes.CDLL(str(lib_path))
            for name in names:
                cls = spec["classes"][name]
                base, sets, out = class_data(name)
                bind = lambda inp, o, cls=cls: cur["bind"](lib, cls, inp, o)
                emit(name, arm, cfg, lambda: measure(spec, cls, bind, base, sets, out))
            continue
        g = spec["generators"][arm]
        kern, kern_err = None, None
        for name in names:
            cls = spec["classes"][name]
            if g.get("static_shapes") or (kern is None and kern_err is None):
                try:
                    kern, kern_err = g["build"](cfg, cls), None
                except Exception as e:
                    kern, kern_err = None, e
            if kern_err is not None:
                emit(name, arm, cfg, err=kern_err)
                continue
            base, sets, out = class_data(name)
            bind = lambda inp, o, cls=cls, kern=kern: g["bind"](kern, cls, inp, o)
            emit(name, arm, cfg, lambda: measure(spec, cls, bind, base, sets, out))
    print()
    print(report(spec, read_rows(path), gen_bench.load_ceilings(a.ceilings), classes))
    if left:
        print(f"\nbudget {a.budget_s}s reached: {left} of {len(todo)} configs not run; "
              f"submit again to resume")


def plan(spec, a):
    classes, gens = pick(spec, a)
    ceil = gen_bench.load_ceilings(a.ceilings)
    cands = candidates(spec, classes, gens)
    print(f"{spec['name']}: {spec['op']}, {spec['dtype']}")
    for gen in gens:
        mine = [c for c in cands if c[0] == gen]
        print(f"  {gen}: {len(mine)} configs, {sum(len(c[2]) for c in mine)} runs")
        for _, cfg, names in mine:
            print(f"    {cfg_label(cfg)}" + ("" if len(names) == len(classes) else f"  [{','.join(names)}]"))
    for name in classes:
        cost = spec["cost"](spec["classes"][name])
        print(f"  {name}: {cost['bytes'] / 1e6:.1f} MB, "
              f"{cost['flops'] * cost.get('passes', 1) / 1e9:.1f} GFLOP {cost['peak']}, "
              f"floor {gen_bench.floor_us(cost, ceil):.1f} us")
    cur = spec.get("current") or {}
    print(f"  current: {cur.get('label', 'none declared')}")


def build_current(spec, a):
    lib = (spec.get("current") or {}).get("lib")
    if not lib:
        raise SystemExit(f"{spec['name']}: no current.lib to build")
    out = Path(a.out).resolve() / f"{spec['name']}.current.so"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [os.environ.get("PLOW_NVCC", "nvcc"), "-shared", "-Xcompiler", "-fPIC", *lib["flags"],
           lib["src"], "-o", str(out)]
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
    print(out)


def queued_search_jobs(gpuq):
    """Labels of gen_search jobs still queued or running (one job at a time per the GPU budget)."""
    text = subprocess.run([sys.executable, str(gpuq), "status"], capture_output=True, text=True,
                          check=True).stdout
    busy = []
    for line in text.splitlines():
        job = ast.literal_eval(line.split(" ", 1)[1])
        if job["state"] in ("queued", "running") and job["label"].startswith("gs-"):
            busy.append(job["label"])
    return busy


def submit_command(spec, a, env, argv):
    """The queued command: timeout + forwarded compiler env + the same arguments as `run`."""
    fwd = [f"{k}={env[k]}" for k in FORWARD_ENV if env.get(k)]
    fwd += [f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("TL_")]
    return ["timeout", "--kill-after=60s", str(JOB_SECONDS), "env", *fwd,
            env.get("PYREF", "python3"), str(HERE / "gen_search.py"), "run", *argv]


def submit(spec, a, argv):
    if a.budget_s > JOB_SECONDS - 120:
        raise SystemExit(f"--budget-s over {JOB_SECONDS - 120}: a queue job is 15 min")
    gpuq = REPO / "scripts" / "bench" / "gpuq.py"
    busy = queued_search_jobs(gpuq)
    if busy:
        raise SystemExit(f"gen_search job already in the queue: {', '.join(busy)}")
    cmd = submit_command(spec, a, os.environ, argv)
    label = "gs-" + spec["name"].replace("_", "-")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(gpuq), "submit", label, "1", *cmd], check=True)
    print(f"queued {label}; rows -> {results_path(spec, a.out)}; job log in the queue root")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("plan", "build-current", "submit", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("spec")
        p.add_argument("--classes")
        p.add_argument("--gens")
        p.add_argument("--ceilings", help="calibrate_ceilings.py JSON (default: calibrated H100)")
        if name != "plan":
            p.add_argument("--out", required=True, help="campaign scratch directory")
        if name in ("submit", "run"):
            p.add_argument("--budget-s", type=float, default=780.0)
            p.add_argument("--current-lib")
            p.add_argument("--force", action="store_true", help="re-measure rows already recorded")
        if name == "report":
            p.add_argument("--all", action="store_true", help="also list every candidate")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for path in sorted(HERE.glob("spec_*.py")):
            spec = load_spec(str(path))
            print(f"{spec['name']}: {spec['op']}, {spec['dtype']}; classes {', '.join(spec['classes'])}; "
                  f"generators {', '.join(spec['generators'])}")
        return
    spec = load_spec(a.spec)
    if a.cmd != "plan" and REPO in Path(a.out).resolve().parents:
        raise SystemExit("--out is campaign scratch outside the repo")
    if a.cmd == "plan":
        plan(spec, a)
    elif a.cmd == "build-current":
        build_current(spec, a)
    elif a.cmd == "submit":
        submit(spec, a, argv[1:])
    elif a.cmd == "run":
        run(spec, a)
    else:
        classes, _ = pick(spec, a)
        print(report(spec, read_rows(results_path(spec, a.out)), gen_bench.load_ceilings(a.ceilings),
                     classes, a.all))


if __name__ == "__main__":
    main()
