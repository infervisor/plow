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
  gen_search.py run SPEC --out DIR [--classes ..] [--gens ..] [--budget-s 720] [--force]
  gen_search.py report SPEC --out DIR [--ceilings ceilings.json] [--all] CPU: the table

`run` needs one leased GPU and a Python with torch and the spec's generators (`PYREF`, the
catalog's interpreter); go through `submit`, which queues it under a 15 min timeout and forwards
the compiler environment (PATH, CUDA_HOME, NVCC_PREPEND_FLAGS, LD_LIBRARY_PATH, TL_*). Rows land
in DIR/<name>.results.jsonl as they are measured; a rerun resumes after the rows already there,
so a grid larger than one job's budget is a second `submit`. A row counts while its class
geometry, tolerance and config are still the spec's (and, for resume, device, generator versions
and TL_* match); after changing a kernel body use a fresh DIR or --force. DIR is campaign
scratch, never the repo. Timing: a CUDA graph over rotating cold inputs larger than L2, median us
per call; a class whose rotating inputs stay under that at the set cap is reported cache-warm.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import shlex
import subprocess
import sys
import tempfile
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
# A queue job is 15 min: TERM at JOB_TERM_S, KILL JOB_KILL_S later, and `run` stops starting
# candidates BUDGET_MARGIN_S before the TERM.
JOB_TERM_S, JOB_KILL_S, BUDGET_MARGIN_S = 840, 30, 120


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


def class_fp(spec, name):
    """Identity of what a row measured: the class geometry and the tolerance it was judged by."""
    cls = {k: v for k, v in spec["classes"][name].items() if k != "step"}
    text = json.dumps([cls, spec["tol"]], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def live_rows(spec, rows, classes, gens):
    """Latest row per (class, arm, cfg), restricted to the spec as it is now: known class with
    the recorded geometry and tolerance, and a config still in the selected generators' grids."""
    keep = {(n, "current", "{}") for n in classes}
    for gen, cfg, names in candidates(spec, classes, gens):
        keep |= {(n, gen, json.dumps(cfg, sort_keys=True)) for n in names}
    last = {}
    for r in rows:
        if row_key(r) in keep and r.get("fp") == class_fp(spec, r["class"]):
            last[row_key(r)] = r
    return last


def summarize(spec, rows, ceil, classes=None, gens=None):
    """One table row per class: current, best passing generated candidate, floor, verdict."""
    classes = classes or list(spec["classes"])
    last = live_rows(spec, rows, classes, gens or list(spec["generators"]))
    cur = spec.get("current") or {}
    out = []
    for name in classes:
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
            "class": name,
            "block": f"{spec['op']}, {name}",
            "floor_us": gen_bench.floor_us(spec["cost"](cls), ceil),
            "cur_us": cur_us, "cur_note": note if cur_us is not None else "",
            "gen_us": best and best["us"],
            "gen_label": best and f"{best['arm']} {cfg_label(best['cfg'])}",
            "rel_l2": best and best["rel_l2"],
            "verdict": gen_bench.verdict(cur_us, best and best["us"], cls.get("step")),
            "tried": len([r for r in mine if r["arm"] != "current"]),
            "passed": len(ok),
            "warm": any(r.get("cold") is False for r in mine),
        })
    return out


def report(spec, rows, ceil, classes=None, gens=None, every=False):
    classes = classes or list(spec["classes"])
    gens = gens or list(spec["generators"])
    summary = summarize(spec, rows, ceil, classes, gens)
    last = live_rows(spec, rows, classes, gens)
    cur = spec.get("current") or {}
    where = sorted({f"{r.get('device')} {r.get('versions')} {r.get('tl_env') or ''}".strip()
                    for r in last.values()})
    lines = [f"{spec['name']}: {spec['op']}, {spec['dtype']}", "", gen_bench.table(summary), "",
             f"floor = max(bytes / {ceil['bw'] / 1e9:.0f} GB/s, passes x FLOPs / peak); peaks TF/s "
             + ", ".join(f"{k} {v / 1e12:g}" for k, v in sorted(ceil["peak"].items()))
             + f"; ceilings: {ceil['source']}",
             f"current: {cur.get('label', 'none declared')}"
             + (f"; recorded, not same-lease: {cur['source']}" if "measured_us" in cur else ""),
             "candidates passing numerics / tried: "
             + ", ".join(f"{s['class']} {s['passed']}/{s['tried']}" for s in summary),
             "measured on: " + ("; ".join(where) or "nothing measured")]
    if len(gens) > 1:
        lines += ["", "| class | generator | best us | config | rel-L2 | passing / tried |",
                  "|---|---|---|---|---|---|"]
        for name in classes:
            for gen in gens:
                mine = [r for r in last.values() if r["class"] == name and r["arm"] == gen]
                ok = [r for r in mine if r.get("ok")]
                best = min(ok, key=lambda r: r["us"]) if ok else None
                lines.append(f"| {name} | {gen} | " + (f"{best['us']:.1f} | {cfg_label(best['cfg'])} | "
                             f"{best['rel_l2']:.1e}" if best else "— | — | —")
                             + f" | {len(ok)} / {len(mine)} |")
        lines.append("")
    warm = [s["class"] for s in summary if s["warm"]]
    if warm:
        lines.append(f"cache-warm timing (rotating inputs under {gen_bench.COLD_BYTES / 1e6:.0f} MB "
                     f"at the set cap): {', '.join(warm)}")
    if every:
        lines += ["", "| class | arm | config | us | rel-L2 | max/rms | pcm rms | status |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(last.values(), key=lambda r: (r["class"], r.get("us") or 1e30)):
            f = lambda v, p: "—" if v is None else format(v, p)
            lines.append(f"| {r['class']} | {r['arm']} | {cfg_label(r['cfg'])} | "
                         f"{f(r.get('us'), '.1f')} | {f(r.get('rel_l2'), '.1e')} | "
                         f"{f(r.get('max_rms'), '.1e')} | {f(r.get('pcm_rms'), '.1e')} | "
                         f"{'ok' if r.get('ok') else r.get('error', 'FAIL')[:60]} |")
    return "\n".join(lines)


def versions(arm):
    """Package versions a row depends on: torch and the row's own generator."""
    out = {}
    for name in ("torch", arm):
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


def error_text(e):
    return f"{type(e).__name__}: {e}".splitlines()[0][:200]


def tl_env(env):
    return {k: v for k, v in sorted(env.items()) if k.startswith("TL_")}


def run_class(spec, name, arms, kernels, lib, emit, expired):
    """Measure `arms` ((arm, cfg) pairs) on one class; its inputs live only in this scope.
    Returns how many arms the budget left unmeasured."""
    import torch
    cls = spec["classes"][name]
    base = spec["inputs"](cls, 0)
    nbytes = sum(base[k].numel() * base[k].element_size() for k in spec["rotate"])
    nsets = gen_bench.cold_sets(nbytes)
    sets = [dict(base, **{k: v for k, v in spec["inputs"](cls, s + 1).items() if k in spec["rotate"]})
            for s in range(nsets)]
    shape, dtype = spec["out"](cls)
    out = torch.empty(shape, device="cuda", dtype=getattr(torch, dtype))
    extra = {"cold": nbytes * nsets >= gen_bench.COLD_BYTES}
    for i, (arm, cfg) in enumerate(arms):
        if expired():
            return len(arms) - i
        if arm == "current":
            bind = lambda inp, o: spec["current"]["bind"](lib, cls, inp, o)
        else:
            g = spec["generators"][arm]
            key = (arm, json.dumps(cfg, sort_keys=True), name if g.get("static_shapes") else None)
            if key not in kernels:
                try:
                    kernels[key] = (g["build"](cfg, cls), None)
                except Exception as e:  # recorded for every class that would have used it
                    kernels[key] = (None, error_text(e))
            kern, err = kernels[key]
            if err is not None:
                emit(name, arm, cfg, extra, err=err)
                continue
            bind = lambda inp, o, g=g, kern=kern: g["bind"](kern, cls, inp, o)
        emit(name, arm, cfg, extra, lambda: measure(spec, cls, bind, base, sets, out))
    return 0


def run(spec, a):
    import torch
    classes, gens = pick(spec, a)
    path = results_path(spec, a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = os.environ.get("GEN_SEARCH_TL")
    if keep is not None:  # the queue runner's own TL_* toggles are not the submitter's
        for k in [k for k in os.environ if k.startswith("TL_") and k not in keep.split(",")]:
            del os.environ[k]
    stamp = {"spec": spec["name"], "device": torch.cuda.get_device_name(), "tl_env": tl_env(os.environ)}
    same = lambda r: (all(r.get(k) == v for k, v in stamp.items())
                      and r.get("versions") == versions(r["arm"]))
    if a.force and path.is_file():
        # drop the selection's recorded rows, so a forced sweep cut short by the budget cannot
        # leave the old measurements selectable or resumable
        arms = {"current", *gens}
        kept = [r for r in read_rows(path) if not (r["class"] in classes and r["arm"] in arms)]
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    done = {k for k, r in live_rows(spec, read_rows(path), classes, gens).items() if same(r)}
    t0 = time.monotonic()

    def emit(name, arm, cfg, extra, fn=None, err=None):
        row = dict(stamp, **{"class": name, "arm": arm, "cfg": cfg, "fp": class_fp(spec, name),
                             "versions": versions(arm)}, **extra)
        fatal = False
        if err is None:
            try:
                row.update(fn())
            except Exception as e:  # a candidate that does not launch is a result
                err = error_text(e)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    fatal = True
        if err is not None:
            row.update(ok=False, error=err)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if fatal:  # a device fault poisons the context: the remaining arms need a new process
            raise SystemExit(f"{arm} {cfg_label(cfg)} on {name} left the CUDA context unusable; "
                             f"the remaining arms are pending, submit again")

    lib = None
    if "lib" in (spec.get("current") or {}):
        import ctypes
        lib_path = Path(a.current_lib or Path(a.out) / f"{spec['name']}.current.so").resolve()
        if not lib_path.is_file():
            raise SystemExit(f"{lib_path}: run build-current first (CPU, outside the queue)")
        lib = ctypes.CDLL(str(lib_path))
    cands = candidates(spec, classes, gens)
    kernels, left, total = {}, 0, 0
    for name in classes:
        arms = [("current", {})] if lib is not None else []
        arms += [(gen, cfg) for gen, cfg, names in cands if name in names]
        arms = [(arm, cfg) for arm, cfg in arms if (name, arm, json.dumps(cfg, sort_keys=True)) not in done]
        total += len(arms)
        if not arms:
            continue
        if time.monotonic() - t0 > a.budget_s:
            left += len(arms)
            continue
        torch.cuda.empty_cache()
        left += run_class(spec, name, arms, kernels, lib, emit,
                          lambda: time.monotonic() - t0 > a.budget_s)
    print()
    print(report(spec, read_rows(path), gen_bench.load_ceilings(a.ceilings), classes, gens))
    if left:
        print(f"\nbudget {a.budget_s:g}s reached: {left} of {total} runs not measured; "
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
    subprocess.run(cmd, cwd=REPO, env={**os.environ, "TMPDIR": str(out.parent)}, check=True)
    print(out)


def queued_jobs(gpuq, label):
    """Queue ids of this spec's search jobs still queued or running. The guard is per spec:
    workstreams own different specs, and the one-queued-lease-per-agent rule is the agent's."""
    text = subprocess.run([sys.executable, str(gpuq), "status"], capture_output=True, text=True,
                          check=True).stdout
    busy = []
    for line in text.splitlines():
        job_id, job = line.split(" ", 1)
        job = ast.literal_eval(job)
        if job["state"] in ("queued", "running") and job["label"] == label:
            busy.append(job_id)
    return busy


def submit_command(env, argv):
    """The queued command: a 15 min bound, the submitter's compiler environment in place of the
    queue runner's (set or explicitly unset), then the same arguments as `run`."""
    tl = tl_env(env)
    fwd = [x for k in FORWARD_ENV if not env.get(k) for x in ("-u", k)]
    fwd += [f"{k}={env[k]}" for k in FORWARD_ENV if env.get(k)]
    fwd += [f"{k}={v}" for k, v in tl.items()] + ["GEN_SEARCH_TL=" + ",".join(tl)]
    return ["timeout", f"--kill-after={JOB_KILL_S}s", str(JOB_TERM_S), "env", *fwd,
            env.get("PYREF", "python3"), str(HERE / "gen_search.py"), "run", *argv]


def submit(spec, a, argv):
    if a.budget_s > JOB_TERM_S - BUDGET_MARGIN_S:
        raise SystemExit(f"--budget-s over {JOB_TERM_S - BUDGET_MARGIN_S}: a queue job is 15 min")
    gpuq = REPO / "scripts" / "bench" / "gpuq.py"
    label = "gs-" + spec["name"].replace("_", "-")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(tempfile.gettempdir()) / "plow-gen-search.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        busy = queued_jobs(gpuq, label)
        if busy:
            raise SystemExit(f"{label} is already in the queue ({', '.join(busy)}); it resumes "
                             f"from the recorded rows, wait for it")
        subprocess.run([sys.executable, str(gpuq), "submit", label, "1",
                        *submit_command(os.environ, argv)], check=True)
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
            p.add_argument("--budget-s", type=float, default=float(JOB_TERM_S - BUDGET_MARGIN_S))
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
    if a.cmd != "plan" and Path(a.out).resolve().is_relative_to(REPO):
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
        classes, gens = pick(spec, a)
        print(report(spec, read_rows(results_path(spec, a.out)), gen_bench.load_ceilings(a.ceilings),
                     classes, gens, a.all))


if __name__ == "__main__":
    main()
