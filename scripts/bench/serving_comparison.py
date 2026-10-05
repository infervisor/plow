#!/usr/bin/env python3
"""Strict Infervisor (plow) vs baseline serving report: the only accepted format for a final
performance comparison (docs/bringup/agent-tools.md, "Final performance report (strict)").

  serving_comparison.py record <resdir> --side plow|vllm --hf DIR --reps N [--server-args S]
      [--sampled S] [--pyref PY] [--assets DIR] [--plowrt BIN] [--repo DIR]
      Called by llm_grid.sh inside the lease: writes <resdir>/provenance.json (model version,
      precision, GPU name/count from nvidia-smi, stack version + flags, plowrt git sha, packet
      sha256). Env overrides: MODEL_VERSION, PRECISION, KV_DTYPE (plow side: required unless
      PRECISION is set), PLOWRT_GIT_SHA (required when --plowrt is outside a git checkout).

  serving_comparison.py render --baseline DIR --infervisor DIR --gate gates.json --out DIR
      [--cells g128,s128,a64.g] [--baseline-provenance P] [--infervisor-provenance P]
      One 12-row table per cell (single-turn g<c>/s<c>, agentic a<c>.g/a<c>.s, open-loop production
      mix q<1000*rate>.g/.s with goodput etc. in a supplementary note under the table) into
      comparison.md / comparison.json / comparison.csv. --gate is the `campaign.py gate --only
      llm_fp32_ref` gates.json for the served packet.
      Exit 2: rule violation (missing data, gate for another packet, < 2 repeats, unpaired cells);
      nothing is written. Exit 1: written, but a cell is NOT MATCHED or NOT EQUIVALENT.
      Exit 0: every cell matched and equivalent.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import waterfall  # noqa: E402

LABELS = ("Model / version", "Precision / quantization", "Input / output length",
          "Traffic / concurrency", "GPU type & count", "Serving stack",
          "Output quality / correctness", "Peak GPU memory", "Total throughput",
          "Throughput / GPU", "TTFT P99", "TPOT P99")
MATCHED_ROWS = LABELS[:5]
SAME = "Same as baseline"
SPREAD_MAX_PCT = 10.0
# (key, label, format); ratios are Infervisor / Baseline.
MEASURED = (("peak_gib", "Peak GPU memory", "{:,.1f} GiB"),
            ("total_tok_s", "Total throughput", "{:,.0f} tok/s"),
            ("total_tok_s_gpu", "Throughput / GPU", "{:,.0f} tok/s/GPU"),
            ("ttft_p99_ms", "TTFT P99", "{:,.1f} ms"),
            ("tpot_p99_ms", "TPOT P99", "{:,.2f} ms"))


class ReportError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- record
def model_version(hf):
    if os.environ.get("MODEL_VERSION"):
        return os.environ["MODEL_VERSION"]
    p = Path(hf).resolve()
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([^/]+)", str(p))
    name = f"{m[1]}/{m[2]}@{m[3][:12]}" if m else p.name
    h = hashlib.sha256((p / "config.json").read_bytes())
    for f in sorted(p.glob("*.safetensors")):
        h.update(f"{f.name}:{f.stat().st_size}\n".encode())
    return f"{name} (config+weights {h.hexdigest()[:12]})"


def describe_quant(q):
    method = q.get("quant_method", "quantized")
    groups = list((q.get("config_groups") or {}).values())
    if groups:
        w, a = groups[0].get("weights") or {}, groups[0].get("input_activations")
        s = f"{method} W{w.get('num_bits')} {w.get('type')} per-{w.get('strategy')}"
        if a:
            s += f", A{a.get('num_bits')} {a.get('type')} {'dynamic ' if a.get('dynamic') else ''}per-{a.get('strategy')}"
        return s
    extras = [f"{k}={q[k]}" for k in ("fmt", "activation_scheme", "weight_block_size") if k in q]
    return method + (f" ({', '.join(extras)})" if extras else "")


def flag_value(args, name):
    toks = shlex.split(args or "")
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None


def precision(hf, side, server_args):
    if os.environ.get("PRECISION"):
        return os.environ["PRECISION"]
    cfg = json.loads((Path(hf) / "config.json").read_text())
    text = cfg.get("text_config") or {}
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype") or text.get("dtype")
    q = cfg.get("quantization_config")
    weights = describe_quant(q) if q else dtype
    if side == "vllm":
        kv = flag_value(server_args, "--kv-cache-dtype") or "auto"
        if kv == "auto":
            kv = dtype
    else:
        kv = os.environ.get("KV_DTYPE")
    return f"weights {weights}; KV cache {kv}" if weights and kv else None


def gpus():
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        cmd += ["-i", os.environ["CUDA_VISIBLE_DEVICES"]]
    try:
        names = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return None, None
    names = [n.strip() for n in names if n.strip()]
    return (" + ".join(sorted(set(names))), len(names)) if names else (None, None)


def git_sha(directory):
    def g(*a):
        r = subprocess.run(["git", "-C", str(directory), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    head = g("rev-parse", "HEAD")
    if head is None:
        return None
    return head[:12] + ("+dirty" if g("status", "--porcelain", "--untracked-files=no") else "")


def record(a):
    res = Path(a.resdir)
    name, count = gpus()
    prov = dict(side=a.side, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), hf=str(Path(a.hf).resolve()),
                model_version=model_version(a.hf), precision=precision(a.hf, a.side, a.server_args),
                gpu_name=name, gpu_count=count, repeats=a.reps, sampled=a.sampled,
                server_args=a.server_args, recorder_commit=git_sha(Path(__file__).resolve().parent))
    if a.side == "vllm":
        version = None
        if a.pyref:
            r = subprocess.run([a.pyref, "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
                               capture_output=True, text=True)
            version = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
        prov.update(stack_version=version,
                    stack=f"vLLM {version} ({a.server_args.strip() or 'default flags'})" if version else None)
    else:
        if os.environ.get("PLOWRT_GIT_SHA"):
            sha = os.environ["PLOWRT_GIT_SHA"]
        elif a.plowrt:
            sha = git_sha(Path(a.plowrt).resolve().parent)
        else:
            sha = git_sha(a.repo or Path(__file__).resolve().parents[2])
        pkt = Path(a.assets or "") / "model.pkt"
        binary = res / "plowrt"
        prov.update(plowrt_git_sha=sha, packet_sha256=sha256(pkt) if a.assets and pkt.is_file() else None,
                    plowrt_sha256=sha256(binary) if binary.is_file() else None, assets=a.assets,
                    plow_env={k: v for k, v in sorted(os.environ.items()) if k.startswith("PLOW_")})
    res.mkdir(parents=True, exist_ok=True)
    (res / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    missing = [k for k, v in prov.items() if v is None and k not in ("recorder_commit",)]
    if missing:
        print(f"provenance: {', '.join(missing)} unrecorded in {res}/provenance.json "
              "(render refuses an arm missing a table field; see --help)", file=sys.stderr)


# ---------------------------------------------------------------- load
def read_peak(root, tag, errors):
    p = root / f"{tag}.peak_gpu_memory_mib.txt"
    text = p.read_text().strip() if p.is_file() else ""
    if not text:
        errors.append(f"{root}: {tag} peak GPU memory missing ({p.name})")
        return None
    return float(text) / 1024


def temp_label(temperature, top_p):
    if temperature == 0:
        return "greedy (T=0)"
    return f"sampled (T={temperature:g}" + (f", top_p={top_p:g}" if top_p is not None else "") + ")"


def single_run(root, tag, prov, errors):
    d = waterfall.bench(str(root), tag)
    if d is None:
        errors.append(f"{root}: {tag} has no bench.json")
        return None
    if d.get("failed") or d.get("completed") != d.get("num_prompts"):
        errors.append(f"{root}: {tag} completed {d.get('completed')}/{d.get('num_prompts')}, failed {d.get('failed')}")
        return None
    ins, outs = d.get("input_lens"), d.get("output_lens")
    if ins and outs and len(set(ins)) == 1 and len(set(outs)) == 1:
        io = f"{ins[0]} / {outs[0]} tokens"
    else:
        io = f"{d['total_input_tokens'] / d['completed']:.0f} / {d['total_output_tokens'] / d['completed']:.0f} tokens (mean)"
    kind = "greedy (T=0)" if tag[0] == "g" else f"sampled ({prov.get('sampled') or 'unrecorded'})"
    traffic = (f"{kind}, {d['num_prompts']} prompts, request rate {d['request_rate']} / "
               f"concurrency {d['max_concurrency']}")
    return dict(io=io, io_key=io + json.dumps([ins, outs]), traffic=traffic,
                total_tok_s=(d["total_input_tokens"] + d["total_output_tokens"]) / d["duration"],
                ttft_p99_ms=d.get("p99_ttft_ms"), tpot_p99_ms=d.get("p99_tpot_ms"),
                peak_gib=read_peak(root, tag, errors), source=str(root / tag))


def agentic_run(root, tag, errors):
    path = root / f"{tag}.json"
    d = json.loads(path.read_text())
    c, o = d["config"], d["overall"]
    if o.get("errors"):
        errors.append(f"{path}: {o['errors']} failed requests")
        return None
    ok = [r for r in d["requests"] if r.get("error") is None]
    io = (f"agentic {c['turns']} turns, system {c['system_tokens']}, prompt grows to "
          f"{c['target_tokens']} / {c['max_tokens']} tokens per turn")
    traffic = (f"{temp_label(c['temperature'], c.get('top_p'))}, {c['api']} API, closed-loop sessions, "
               f"session header {'on' if c.get('session_header', True) else 'off'} / "
               f"concurrency {c['sessions'] * c.get('sessions_per_worker', 1)} sessions")
    return dict(io=io, io_key=io, traffic=traffic,
                total_tok_s=sum(r["prompt_tokens"] + r["completion_tokens"] for r in ok) / o["wall_s"],
                ttft_p99_ms=o.get("ttft_p99_ms"), tpot_p99_ms=o.get("tpot_p99_ms"),
                peak_gib=read_peak(root, tag, errors), source=str(path))


# Open-loop cells: values under the strict table, means over repeats.
SUPPLEMENTARY = (("goodput_req_s", "goodput", "{:.3f} req/s"), ("slo_attainment", "SLO met", "{:.1%}"),
                 ("request_s", "requests", "{:.3f} req/s"), ("mean_inflight", "mean in-flight", "{:.1f}"),
                 ("mean_sessions", "mean live sessions", "{:.1f}"), ("ttft_p50_ms", "TTFT P50", "{:,.1f} ms"),
                 ("tpot_p50_ms", "TPOT P50", "{:.2f} ms"), ("cached_fraction", "cached prompt tokens", "{:.1%}"))


def prod_run(root, tag, errors):
    path = root / f"{tag}.json"
    d = json.loads(path.read_text())
    c, o = d["config"], d["overall"]
    if o["errors_total"]:
        errors.append(f"{path}: {o['errors_total']} failed requests")
        return None
    io = (f"open-loop mix: {c['apps']} system prompts (lognormal median {c['system_median']:g}), turns geometric "
          f"mean {c['turns_mean']:g} max {c['turns_max']}, first message lognormal({c['first_median']:g}, "
          f"{c['first_sigma']:g}), tool output lognormal({c['tool_median']:g}, {c['tool_sigma']:g}), context cap "
          f"{c['max_model_len']} / output lognormal({c['out_median']:g}, {c['out_sigma']:g}) in "
          f"[{c['out_min']}, {c['out_max']}] tokens, ignore_eos")
    traffic = (f"{temp_label(c['temperature'], c.get('top_p'))}, {c['api']} API, Poisson {c['rate']:g} sessions/s, "
               f"think lognormal({c['think_median_s']:g} s, {c['think_sigma']:g}) <= {c['think_max_s']:g} s, "
               f"{c['duration']:g} s (measured {c['warmup']:g}-{c['duration'] - c['cooldown']:g} s), "
               f"session header {'on' if c.get('session_header', True) else 'off'} / open loop (achieved "
               f"concurrency in the supplementary note)")
    extra = {k: o.get(k) for k, _, _ in SUPPLEMENTARY}
    extra["seed"] = c["seed"]
    extra["slo"] = f"TTFT <= {o['slo_ttft_ms']:g} ms and TPOT <= {o['slo_tpot_ms']:g} ms"
    # Pair on everything that shapes the plan or the goodput score, not only the described subset.
    # Per-run fields (endpoint, output path, served name) and the seed (compared across arms per
    # repeat) are left out; the rate stays in, so how it was derived does not matter.
    plan = {k: v for k, v in c.items()
            if k not in ("url", "out", "model", "seed", "timeout", "est_e2e_s", "target_concurrency")}
    plan["slo"] = [o["slo_ttft_ms"], o["slo_tpot_ms"]]
    io_key = io + " #" + hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    return dict(io=io, io_key=io_key, traffic=traffic, total_tok_s=o["total_tok_s"],
                ttft_p99_ms=o.get("ttft_p99_ms"), tpot_p99_ms=o.get("tpot_p99_ms"),
                peak_gib=read_peak(root, tag, errors), source=str(path), extra=extra)


def discover(root):
    """{cell: {repeat: tag}} for single-turn g<c>/s<c>, agentic a<c>.g/a<c>.s and open-loop q<r>.g/q<r>.s results."""
    cells = {}
    for p in root.iterdir():
        m = re.fullmatch(r"([gs]\d+)\.r(\d+)", p.name) if p.is_dir() else re.fullmatch(r"([aq]\d+\.[gs])\.r(\d+)\.json", p.name)
        if m:
            cells.setdefault(m[1], {})[int(m[2])] = p.name.removesuffix(".json")
    return cells


def cell_order(cell):
    return ("gsaq".index(cell[0]) // 2 + (cell[0] == "q"), int(re.search(r"\d+", cell)[0]), cell)


def arm(root, prov, cell, tags, errors):
    runs = []
    for rep in sorted(tags):
        try:
            r = (agentic_run(root, tags[rep], errors) if cell[0] == "a"
                 else prod_run(root, tags[rep], errors) if cell[0] == "q"
                 else single_run(root, tags[rep], prov, errors))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            errors.append(f"{root}: {tags[rep]} malformed result ({type(e).__name__}: {e})")
            r = None
        if r is not None:
            runs.append(r)
    if len(runs) != len(tags):
        return None
    for k in ("io_key", "traffic"):
        if len({r[k] for r in runs}) > 1:
            errors.append(f"{root}: {cell} repeats differ in {k}")
    traffic = runs[0]["traffic"]
    if "extra" in runs[0]:  # one seed per repeat; the arms must replay the same plans
        traffic += ", seeds " + "/".join(str(r["extra"]["seed"]) for r in runs)
    out = dict(io=runs[0]["io"], io_key=runs[0]["io_key"], traffic=traffic, repeats=len(runs),
               sources=[r["source"] for r in runs], per_repeat={}, mean={}, spread_pct={})
    gpu_count = prov.get("gpu_count") or 1
    for r in runs:
        r["total_tok_s_gpu"] = r["total_tok_s"] / gpu_count
    for key, label, _ in MEASURED:
        values = [r[key] for r in runs]
        if any(v is None for v in values):
            errors.append(f"{root}: {cell} {label} missing in a repeat")
            continue
        mean = sum(values) / len(values)
        if mean <= 0:  # the Infervisor / Baseline ratio needs a positive mean
            errors.append(f"{root}: {cell} {label} mean {mean} is not positive")
            continue
        out["per_repeat"][key] = values
        out["mean"][key] = mean
        out["spread_pct"][key] = (max(values) - min(values)) / mean * 100 if mean else 0.0
    if "extra" in runs[0]:
        out["supplementary"] = {k: None if any(r["extra"][k] is None for r in runs)
                                else sum(r["extra"][k] for r in runs) / len(runs) for k, _, _ in SUPPLEMENTARY}
        out["supplementary"]["slo"] = runs[0]["extra"]["slo"]
    return out


# ---------------------------------------------------------------- render
def load_gate(path, packet_sha, errors):
    if path is None or not Path(path).is_file():
        errors.append(f"FP32-reference gate result missing ({path}); run campaign.py gate --only llm_fp32_ref")
        return None
    try:
        g = json.loads(Path(path).read_text())
        res = (g.get("gates") or {}).get("llm_fp32_ref")
    except (ValueError, AttributeError) as e:
        errors.append(f"{path}: malformed gate record ({e})")
        return None
    if res is None:
        errors.append(f"{path}: no llm_fp32_ref gate result")
        return None
    if not g.get("packet_sha256"):
        errors.append(f"{path}: gate record has no packet_sha256; re-score with campaign.py gate --score-only")
        return None
    if packet_sha and g["packet_sha256"] != packet_sha:
        errors.append(f"{path}: gate scored packet {g['packet_sha256'][:12]}, Infervisor served {packet_sha[:12]}")
        return None
    return dict(path=str(path), sha256=sha256(path), packet_sha256=g["packet_sha256"], passed=bool(res.get("pass")),
                why=list(res.get("why") or []), metrics={k: v for k, v in res.items() if k not in ("pass", "why")})


def load_prov(path, side, errors):
    if not Path(path).is_file():
        errors.append(f"{path}: provenance missing (llm_grid.sh records it; or serving_comparison.py record)")
        return {}
    try:
        prov = json.loads(Path(path).read_text())
    except ValueError as e:
        errors.append(f"{path}: malformed provenance ({e})")
        return {}
    need = ["model_version", "precision", "gpu_name", "gpu_count", "repeats"]
    need += ["stack"] if side == "baseline" else ["plowrt_git_sha", "packet_sha256"]
    for k in need:
        if prov.get(k) in (None, ""):
            errors.append(f"{path}: {k} unrecorded")
    return prov


def quality_cells(gate):
    m = gate["metrics"]
    def fmt(prefix):
        parts = [f"{k} {m[prefix + k]:.4g}" for k in ("kl_mean", "top1_decisive", "needle_acc")
                 if isinstance(m.get(prefix + k), (int, float))]
        return ", ".join(parts)
    base = "FP32-reference gate peer" + (f" ({fmt('vllm_')})" if fmt("vllm_") else "")
    infer = "Equivalent" if gate["passed"] else "Not equivalent (" + ("; ".join(gate["why"]) or "gate FAIL") + ")"
    return base, infer


def build_cell(cell, b, i, bprov, iprov, gate):
    gpu = lambda p: f"{p.get('gpu_count')} x {p.get('gpu_name')}"
    config = [("Model / version", bprov["model_version"], iprov["model_version"]),
              ("Precision / quantization", bprov["precision"], iprov["precision"]),
              ("Input / output length", (b["io"], b["io_key"]), (i["io"], i["io_key"])),
              ("Traffic / concurrency", b["traffic"], i["traffic"]),
              ("GPU type & count", gpu(bprov), gpu(iprov))]
    rows, mismatched = [], []
    for label, bv, iv in config:
        (bshow, bkey), (ishow, ikey) = (bv if isinstance(bv, tuple) else (bv, bv)), (iv if isinstance(iv, tuple) else (iv, iv))
        if bkey == ikey:
            rows.append((label, bshow, SAME))
        else:
            mismatched.append(label)
            rows.append((label, bshow, ishow + (" (per-request lengths differ)" if ishow == bshow else "")))
    rows.append(("Serving stack", bprov["stack"],
                 f"Infervisor (plowrt {iprov['plowrt_git_sha']}, packet {iprov['packet_sha256'][:12]})"))
    rows.append(("Output quality / correctness", *quality_cells(gate)))
    flagged = []
    for key, label, fmt in MEASURED:
        bm, im = b["mean"][key], i["mean"][key]
        bs, is_ = fmt.format(bm), fmt.format(im) + f" ({im / bm:.2f}x)"
        if b["spread_pct"][key] > SPREAD_MAX_PCT:
            flagged.append(f"Baseline {label} {b['spread_pct'][key]:.1f}%")
            bs += " *"
        if i["spread_pct"][key] > SPREAD_MAX_PCT:
            flagged.append(f"Infervisor {label} {i['spread_pct'][key]:.1f}%")
            is_ += " *"
        rows.append((label, bs, is_))
    assert tuple(r[0] for r in rows) == LABELS
    extra = {}
    if "supplementary" in b:
        extra["supplementary"] = dict(baseline=b["supplementary"], infervisor=i["supplementary"])
    return dict(extra, cell=cell, matched=not mismatched, mismatched=mismatched, equivalent=gate["passed"],
                spread_flagged=flagged, rows=rows, repeats=b["repeats"],
                ratios={k: i["mean"][k] / b["mean"][k] for k, _, _ in MEASURED},
                baseline={k: b[k] for k in ("mean", "per_repeat", "spread_pct", "sources")},
                infervisor={k: i[k] for k in ("mean", "per_repeat", "spread_pct", "sources")})


def markdown(report):
    md = ["# Serving comparison: Infervisor vs baseline", "",
          f"**Status: {report['status']}**", "",
          f"- Baseline results: `{report['baseline']['dir']}`",
          f"- Infervisor results: `{report['infervisor']['dir']}`",
          f"- FP32-reference gate: `{report['gate']['path']}` (sha256 {report['gate']['sha256'][:12]}, "
          f"packet {report['gate']['packet_sha256'][:12]}, {'PASS' if report['gate']['passed'] else 'FAIL'})",
          "- Values are means over repeats; ratios are Infervisor / Baseline. Total throughput = "
          "(input + output) tok/s. `*` = repeat spread > 10%.", ""]
    for c in report["cells"]:
        status = "MATCHED" if c["matched"] else "NOT MATCHED (" + ", ".join(c["mismatched"]) + ")"
        md += [f"## {c['cell']}", "", f"Comparison: **{status}**; quality: "
               f"**{'EQUIVALENT' if c['equivalent'] else 'NOT EQUIVALENT'}**", "",
               "| Metric | Baseline | Infervisor |", "|---|---|---|"]
        md += [f"| {label} | {bv} | {iv} |" for label, bv, iv in c["rows"]]
        spreads = "; ".join(f"{label} {c['baseline']['spread_pct'][k]:.1f}% / {c['infervisor']['spread_pct'][k]:.1f}%"
                            for k, label, _ in MEASURED)
        md += ["", f"Spread over {c['repeats']} repeats, (max - min) / mean, Baseline / Infervisor: {spreads}."]
        if c["spread_flagged"]:
            md += ["", "**FLAGGED: spread > 10%: " + "; ".join(c["spread_flagged"]) + ".**"]
        if "supplementary" in c:
            b, i = c["supplementary"]["baseline"], c["supplementary"]["infervisor"]
            show = lambda v, f: "-" if v is None else f.format(v)
            md += ["", f"Supplementary (outside the strict table; goodput = requests meeting {b['slo']} per "
                   "second, measured window), Baseline / Infervisor: "
                   + "; ".join(f"{label} {show(b[k], f)} / {show(i[k], f)}" for k, label, f in SUPPLEMENTARY) + "."]
        md.append("")
    return "\n".join(md)


def render(a):
    errors = []
    roots = dict(baseline=Path(a.baseline), infervisor=Path(a.infervisor))
    provs = {side: load_prov(getattr(a, f"{side}_provenance") or roots[side] / "provenance.json", side, errors)
             for side in roots}
    gate = load_gate(a.gate, provs["infervisor"].get("packet_sha256"), errors)
    found = {side: discover(root) if root.is_dir() else {} for side, root in roots.items()}
    if a.cells:
        cells = a.cells.split(",")
        for side in roots:
            errors += [f"{roots[side]}: cell {c} missing" for c in cells if c not in found[side]]
    else:
        cells = sorted(set(found["baseline"]) | set(found["infervisor"]), key=cell_order)
        for side, other in (("baseline", "infervisor"), ("infervisor", "baseline")):
            errors += [f"{roots[side]}: cell {c} missing (present in {other}; pass --cells to select)"
                       for c in cells if c not in found[side]]
        if not cells:
            errors.append("no result cells found")
    built = []
    for cell in cells:
        if not all(cell in found[s] for s in roots):
            continue
        reps = {s: sorted(found[s][cell]) for s in roots}
        for s in roots:
            want = provs[s].get("repeats")
            if len(reps[s]) < 2:
                errors.append(f"{roots[s]}: {cell} has {len(reps[s])} repeat(s); at least 2 required")
            if want is not None and reps[s] != list(range(1, int(want) + 1)):
                errors.append(f"{roots[s]}: {cell} repeats {reps[s]} != recorded {want}")
        if reps["baseline"] != reps["infervisor"]:
            errors.append(f"{cell}: repeats differ, baseline {reps['baseline']} vs infervisor {reps['infervisor']}")
        b = arm(roots["baseline"], provs["baseline"], cell, found["baseline"][cell], errors)
        i = arm(roots["infervisor"], provs["infervisor"], cell, found["infervisor"][cell], errors)
        if not errors:
            built.append(build_cell(cell, b, i, provs["baseline"], provs["infervisor"], gate))
    if errors:
        raise ReportError("\n".join(errors))
    matched = all(c["matched"] for c in built)
    equivalent = gate["passed"]
    status = ("MATCHED, EQUIVALENT" if matched and equivalent else
              ", ".join(s for s, bad in (("NOT MATCHED", not matched), ("NOT EQUIVALENT", not equivalent)) if bad)
              + " - not a valid final comparison")
    report = dict(status=status, matched=matched, equivalent=equivalent, gate=gate, cells=built,
                  baseline=dict(dir=str(roots["baseline"].resolve()), provenance=provs["baseline"]),
                  infervisor=dict(dir=str(roots["infervisor"].resolve()), provenance=provs["infervisor"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(markdown(report))
    (out / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")
    with (out / "comparison.csv").open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cell", "matched", "equivalent", "spread_flagged"]
                   + [f"{side}: {label}" for label in LABELS for side in ("Baseline", "Infervisor")]
                   + [f"{side}_{k}" for side in ("baseline", "infervisor") for k, _, _ in MEASURED]
                   + [f"ratio_{k}" for k, _, _ in MEASURED])
        for c in built:
            w.writerow([c["cell"], int(c["matched"]), int(c["equivalent"]), "; ".join(c["spread_flagged"])]
                       + [v for _, bv, iv in c["rows"] for v in (bv, iv)]
                       + [c[side]["mean"][k] for side in ("baseline", "infervisor") for k, _, _ in MEASURED]
                       + [c["ratios"][k] for k, _, _ in MEASURED])
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record")
    r.add_argument("resdir")
    r.add_argument("--side", choices=("plow", "vllm"), required=True)
    r.add_argument("--hf", required=True)
    r.add_argument("--reps", type=int, required=True)
    r.add_argument("--server-args", default="")
    r.add_argument("--sampled", default="")
    r.add_argument("--pyref")
    r.add_argument("--assets")
    r.add_argument("--plowrt", help="plowrt binary as given (empty: <repo>/target/release/plowrt)")
    r.add_argument("--repo")
    v = sp.add_parser("render")
    v.add_argument("--baseline", required=True, help="baseline (vLLM) llm_grid result dir")
    v.add_argument("--infervisor", required=True, help="Infervisor (plow) llm_grid result dir")
    v.add_argument("--gate", required=True, help="campaign.py gate gates.json with llm_fp32_ref")
    v.add_argument("--out", required=True)
    v.add_argument("--cells", help="comma-separated cells (default: all; both arms must have the same set)")
    v.add_argument("--baseline-provenance")
    v.add_argument("--infervisor-provenance")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        record(a)
        return 0
    try:
        report = render(a)
    except ReportError as e:
        print(f"serving_comparison: refused, no report written:\n{e}", file=sys.stderr)
        return 2
    print(f"{report['status']}: {len(report['cells'])} cells -> {a.out}/comparison.md")
    return 0 if report["matched"] and report["equivalent"] else 1


if __name__ == "__main__":
    sys.exit(main())
