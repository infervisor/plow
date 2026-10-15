"""Per-rung sweep of one rung-eligible emit knob (`campaign.py rung-sweep`).

Each value arm is the recipe with the knob set to that value on EVERY prefill rung through
`PLOW_EMIT_RUNG_ENV`, so a rung's program in arm v is exactly the program a mixed recipe gets at
that rung with value v (`verify` checks that on the mixed packet). One GPU job times every swept
rung in every arm with step_bench packed prefill (graph on), blocks in ABBA order, and captures
the compact-terminal logits of the same launches; `score` picks the per-rung winner and writes the
recipe snippet. Stdlib only.

Timing metric: host wall ms of one packed prefill launch at exactly the rung's rows (compact
terminal included), rep 0 (graph capture) dropped, median per block. Scored like agent-tools §7:
saving against max(control drift, treatment spread); spread > 3x drift is not convictable.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import re
import shlex
import statistics
import subprocess
from pathlib import Path

import rung_env

LAUNCH = re.compile(r"packed prefill: rep=(\d+) prefix=(\d+) chunk_rows=(\d+) requests=(\d+) rows=(\d+) ms=([0-9.]+)")
REQUIRES = ["mixed-packet build + per-rung program digest check (rung-sweep --stage verify)",
            "re-time the switched rungs on the mixed packet (verify-run.sh)",
            "llm_fp32_ref gate on the mixed packet when any switched rung is not bitwise"]


def arm_name(value: str) -> str:
    return "v" + re.sub(r"[^A-Za-z0-9]+", "_", value)


def build_json(build_dir: Path) -> dict:
    return json.loads((Path(build_dir) / "assets" / "build.json").read_text())


def vocab_size(build_dir: Path) -> int | None:
    """The model's vocabulary from the build's checkpoint config (each captured case is vocab bf16)."""
    try:
        c = json.loads((Path(build_dir) / "assets" / "checkpoint" / "config.json").read_text())
    except (OSError, ValueError):
        return None
    v = c.get("vocab_size") or (c.get("text_config") or {}).get("vocab_size")
    return v if isinstance(v, int) and v > 0 else None


def knob_id(knob: str) -> str:
    ids, rungs = rung_env.registry()
    if knob not in ids:
        raise ValueError(f"{knob} is not a registered emit knob")
    if "prefill" not in rungs.get(knob, ()):
        raise ValueError(f"{knob} ({ids[knob]}) is packet-wide: knob_spec declares no per-rung prefill override")
    return ids[knob]


def resolved(bj: dict, knob: str, rows: int) -> str | None:
    """The build's value of `knob` at prefill rung `rows`: `rung_knobs` (per program), else the
    packet-wide `emit_config` record (a packet from before per-rung overrides)."""
    kid = knob_id(knob)
    for p in (bj.get("rung_knobs") or {}).get("programs", []):
        if p.get("phase") == "prefill" and p.get("rows") == rows and not p.get("packed"):
            return (p["knobs"].get(kid) or {}).get("value")
    for k in (bj.get("emit_config") or {}).get("knobs", []):
        if k.get("env") == knob:
            return rung_env.boolish(str(k.get("value"))) or str(k.get("value"))
    return None


def program_digest(bj: dict, rows: int) -> str:
    """sha256 over the rung program's instructions (kernel_cases) and its segment chain."""
    cases = [p for p in bj["kernel_cases"]["programs"]
             if p["kind"] == "prefill" and p["rows"] == rows and not p.get("packed_only")]
    if len(cases) != 1:
        raise ValueError(f"expected one ordinary prefill program at rows {rows}, found {len(cases)}")
    index = cases[0]["program"]
    chain = [c for c in bj.get("dispatch_chains", []) if c.get("program") == index]
    blob = json.dumps({"cases": cases[0]["cases"], "chain": chain}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def rung_arms(bj: dict, rows: int) -> list[str]:
    return sorted({a for p in bj.get("programs", []) if p.get("kind") == "prefill" and p.get("bucket") == rows
                   for a in p.get("arms", [])})


def plan_rungs(bj: dict, rungs: list[int]) -> list[dict]:
    """slots x ctx per rung so one packed launch carries exactly `rows` rows."""
    shapes = bj["shapes"]
    buckets = shapes["prefill_buckets"]
    knobs = {k["id"]: k["value"] for k in (bj.get("emit_config") or {}).get("knobs", [])}
    chunk = int(knobs.get("max_request_chunk") or 0) or int(shapes["max_chunk"])
    batch = int(shapes["decode_batch"])
    out = []
    for rows in rungs:
        if rows not in buckets:
            raise ValueError(f"rung {rows} is not a prefill bucket of this packet ({buckets})")
        slots = -(-rows // chunk)
        if rows % slots or rows // slots > chunk or slots > batch:
            raise ValueError(f"rung {rows}: no slots x ctx fits request chunk {chunk} and batch {batch}")
        out.append({"rows": rows, "slots": slots, "ctx": rows // slots})
    return out


def block_orders(arms: list[str], blocks: int) -> list[list[str]]:
    """ABBA...: even blocks in arm order, odd blocks reversed."""
    return [list(arms) if b % 2 == 0 else list(reversed(arms)) for b in range(blocks)]


def run_script(plan: dict, step_bench: Path, run_env: dict[str, str]) -> str:
    out = Path(plan["out"])
    lines = ["#!/usr/bin/env bash",
             "# generated by `campaign.py rung-sweep --stage run`; queue it through scripts/bench/gpuq.py",
             "set -u", f"O={shlex.quote(str(out))}", f"SB=$O/bin/{step_bench.name}", "mkdir -p $O/runs"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(run_env.items())]
    lines += ["nvidia-smi -L > $O/runs/gpu.txt 2>&1; sha256sum $SB > $O/runs/step_bench.sha256",
              "run() {  # run <tag> <assets> <slots> <ctx>",
              "  local d=$O/runs/$1; rm -rf $d.logits",
              f"  timeout {int(plan['timeout_s'])} $SB \"$2\" \"$3\" \"$4\" 1 --warmup 0 --spread --packed-prefill "
              f"--pf-reps {int(plan['reps'])} --dump-prefill-logits $d.logits > $d.log 2>&1",
              "  echo \"rc=$?\" >> $d.log; echo \"$1 $(tail -1 $d.log)\"",
              "}"]
    for r in plan["rungs"]:
        for b, order in enumerate(block_orders(plan["arms"], plan["blocks"])):
            for arm in order:
                assets = Path(plan["builds"][arm]) / "assets"
                lines.append(f"run r{r['rows']}-b{b}-{arm} {shlex.quote(str(assets))} {r['slots']} {r['ctx']}")
    lines.append("echo RUNG_SWEEP_DONE")
    return "\n".join(lines) + "\n"


def parse_log(path: Path, rung: dict, reps: int) -> dict:
    """{ok, ms (median of reps >= 1), samples, why} for one run log: exactly one launch per planned
    rep, each at exactly the rung's width."""
    text = path.read_text(errors="replace") if path.is_file() else ""
    rc = re.findall(r"^rc=(\d+)$", text, re.M)
    launches = [tuple(map(float, m)) for m in LAUNCH.findall(text)]
    why = []
    if not rc or rc[-1] != "0":
        why.append(f"rc={rc[-1] if rc else 'missing'}")
    if sorted(int(x[0]) for x in launches) != list(range(reps)):
        why.append(f"launch reps {[int(x[0]) for x in launches]}, planned 0..{reps - 1} once each")
    samples = []
    for rep, prefix, chunk, reqs, rows, ms in launches:
        if (prefix, chunk, reqs, rows) != (0, rung["ctx"], rung["slots"], rung["rows"]):
            why.append(f"launch rep {int(rep)} ran {int(reqs)}x{int(chunk)} rows={int(rows)} prefix={int(prefix)}, "
                       f"planned {rung['slots']}x{rung['ctx']}")
        elif rep >= 1:
            samples.append(ms)
    if not samples:
        why.append("no timed launches after rep 0")
    return {"ok": not why, "why": why, "samples": samples,
            "ms": statistics.median(samples) if samples and not why else None}


def load_logits(path: Path) -> array.array:
    raw = path.read_bytes()
    if len(raw) % 2:
        raise ValueError(f"{path}: odd byte count for bf16")
    wide = bytearray(len(raw) * 2)
    wide[2::4], wide[3::4] = raw[0::2], raw[1::2]
    out = array.array("f")
    out.frombytes(bytes(wide))
    return out


def logits_cases(d: Path, slots: int) -> dict[str, tuple[str, Path]]:
    """The capture's cases, which must be exactly one per slot, each with a file."""
    man = json.loads((d / "manifest.json").read_text())
    cases = man.get("cases") if isinstance(man, dict) else None
    if not isinstance(cases, list) or not all(isinstance(c, dict) for c in cases):
        raise ValueError(f"{d}: malformed manifest")
    ids = [c["id"] for c in cases]
    want = [f"prefill-s{b:03}" for b in range(slots)]
    if sorted(ids) != want:
        raise ValueError(f"{d}: cases {ids}, expected {want}")
    out = {c["id"]: (c["prompt_sha256_u32le"], d / c["file"]) for c in cases}
    for cid, (_, f) in out.items():
        if not f.is_file():
            raise ValueError(f"{d}: {cid} has no {f.name}")
    return out


def compare_logits(ref_dir: Path, cand_dir: Path, slots: int, vocab: int | None) -> dict:
    """bitwise, finite, top1 agreement, KL(ref || cand) mean/max and max |dlogit| over the slots.
    Any missing, malformed or mis-sized capture is `complete: False` with the reason."""
    try:
        if vocab is None:
            raise ValueError("vocabulary size unknown (no checkpoint config in the reference build)")
        ref, cand = logits_cases(ref_dir, slots), logits_cases(cand_dir, slots)
        kls, top1, dmax, bitwise, finite = [], 0, 0.0, True, True
        for cid, (sha, rp) in sorted(ref.items()):
            csha, cp = cand[cid]
            if sha != csha:
                raise ValueError(f"{cid}: different prompts")
            rb, cb = rp.read_bytes(), cp.read_bytes()
            if len(rb) != 2 * vocab or len(cb) != 2 * vocab:
                raise ValueError(f"{cid}: logit buffers of {len(rb)} vs {len(cb)} bytes, want {2 * vocab}")
            bitwise &= rb == cb
            a, b = load_logits(rp), load_logits(cp)
            if not all(map(math.isfinite, a)) or not all(map(math.isfinite, b)):
                finite = False
                continue
            ma, mb = max(a), max(b)
            ea = [math.exp(x - ma) for x in a]
            eb = [math.exp(x - mb) for x in b]
            za, zb = sum(ea), sum(eb)
            lza, lzb = math.log(za), math.log(zb)
            kl = sum(e / za * ((x - ma - lza) - (y - mb - lzb)) for e, x, y in zip(ea, a, b))
            kls.append(max(kl, 0.0))
            top1 += a.index(ma) == b.index(mb)
            dmax = max(dmax, max(abs(x - y) for x, y in zip(a, b)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return {"complete": False, "why": str(e)}
    return {"complete": True, "finite": finite, "bitwise": bitwise and finite, "cases": slots,
            "top1": top1 / slots if finite else None, "kl_mean": statistics.fmean(kls) if finite else None,
            "kl_max": max(kls) if finite else None, "max_dlogit": dmax if finite else None}


def worst(per_block: list[dict]) -> dict:
    """One verdict over every block's comparison: incomplete or non-finite anywhere wins."""
    bad = next((x for x in per_block if not x.get("complete") or not x.get("finite")), None)
    if bad is not None:
        return {"complete": bad.get("complete", False), "finite": False,
                "why": bad.get("why", "non-finite logits"), "blocks": len(per_block)}
    return {"complete": True, "finite": True, "bitwise": all(x["bitwise"] for x in per_block),
            "cases": per_block[0]["cases"], "blocks": len(per_block),
            "top1": min(x["top1"] for x in per_block), "kl_mean": max(x["kl_mean"] for x in per_block),
            "kl_max": max(x["kl_max"] for x in per_block), "max_dlogit": max(x["max_dlogit"] for x in per_block)}


def resources(assets: Path, cuobjdump: str) -> dict[str, dict]:
    """Per cubin: sha256 and the max REG / STACK / SHARED / LOCAL over its kernels."""
    out = {}
    for cubin in sorted(list(assets.glob("*.cubin")) + list((assets / "objects").glob("*.cubin"))):
        rel = str(cubin.relative_to(assets))
        res = {"sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()}
        try:
            text = subprocess.run([cuobjdump, "--dump-resource-usage", str(cubin)], capture_output=True,
                                  text=True, check=False, timeout=120).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            res["error"] = str(e)
            out[rel] = res
            continue
        for field in ("REG", "STACK", "SHARED", "LOCAL"):
            vals = [int(v) for v in re.findall(rf"\b{field}:(\d+)", text)]
            res[field.lower()] = max(vals) if vals else None
        out[rel] = res
    return out


def noise_rule(ctl: list[float], trt: list[float]) -> dict:
    """agent-tools §7 over every block median: saving against max(control drift, treatment
    spread); spread > 3x drift is not convictable."""
    c, t = statistics.fmean(ctl), statistics.fmean(trt)
    drift = max(ctl) - min(ctl)
    spread = max(trt) - min(trt)
    noise = max(drift, spread)
    convictable = spread <= 3 * drift
    return {"control_ms": c, "treatment_ms": t, "saving_ms": c - t, "saving_pct": 100 * (c - t) / c,
            "drift_ms": drift, "spread_ms": spread, "noise_ms": noise, "convictable": convictable,
            "faster": convictable and c - t > noise}


def score(plan: dict, runs: Path, ref_build: dict, arm_builds: dict[str, dict],
          kl_max: float | None = None, top1_min: float | None = None, vocab: int | None = None) -> dict:
    """Per rung: the reference arm (the reference build's value), and another arm only if its
    program differs, every run and every block's logits are complete and finite (reference too),
    and it is faster beyond §7 noise. Accuracy is ACCEPTED only when bitwise, or when both
    --kl-max and --top1-min are given and pass; otherwise the choice stays experimental."""
    if kl_max is not None and not (math.isfinite(kl_max) and kl_max >= 0):
        raise ValueError(f"--kl-max {kl_max}: must be finite and >= 0")
    if top1_min is not None and not (math.isfinite(top1_min) and 0 <= top1_min <= 1):
        raise ValueError(f"--top1-min {top1_min}: must be in [0, 1]")
    knob, values, blocks = plan["knob"], plan["values"], plan["blocks"]
    by_value = {rung_env.boolish(v) or v: arm_name(v) for v in values}
    out_rungs = []
    for r in plan["rungs"]:
        rows = r["rows"]
        ref_value = resolved(ref_build, knob, rows)
        ref_arm = by_value.get(ref_value)
        row = {"rows": rows, "slots": r["slots"], "ctx": r["ctx"], "reference_value": ref_value, "arms": {}}
        for arm, bj in arm_builds.items():
            runs_ok, why, block_ms = True, [], []
            for b in range(blocks):
                p = parse_log(runs / f"r{rows}-b{b}-{arm}.log", r, plan["reps"])
                runs_ok &= p["ok"]
                why += [f"b{b}: {w}" for w in p["why"]]
                block_ms.append(p["ms"])
            row["arms"][arm] = {"value": next(v for v in values if arm_name(v) == arm), "ok": runs_ok, "why": why,
                                "block_ms": block_ms, "program": program_digest(bj, rows), "ops": rung_arms(bj, rows)}
        if ref_arm is None:
            row.update(winner=None, status=f"reference value {ref_value!r} was not swept")
            out_rungs.append(row)
            continue
        ref_row = row["arms"][ref_arm]
        row["reference_arm"] = ref_arm
        row["reference_program_matches"] = ref_row["program"] == program_digest(ref_build, rows)
        ref_row["repeat"] = worst([compare_logits(runs / f"r{rows}-b0-{ref_arm}.logits",
                                                  runs / f"r{rows}-b{b}-{ref_arm}.logits", r["slots"], vocab)
                                   for b in range(blocks)])
        ref_bad = list(ref_row["why"])
        if not row["reference_program_matches"]:
            ref_bad.append("reference arm's program differs from the reference build's")
        rp = ref_row["repeat"]
        if not rp.get("finite"):
            ref_bad.append(f"reference logits: {rp.get('why')}")
        elif not rp["bitwise"] and (kl_max is None or top1_min is None):
            ref_bad.append("reference logits not repeatable across blocks (not bitwise; give --kl-max and --top1-min)")
        elif not rp["bitwise"] and (rp["kl_max"] > kl_max or rp["top1"] < top1_min):
            ref_bad.append(f"reference logits not repeatable: KL max {rp['kl_max']:.3g}, top1 {rp['top1']:.2f}")
        if ref_bad:
            row.update(winner=None, status="reference failed: " + "; ".join(ref_bad))
            out_rungs.append(row)
            continue
        best = None
        for arm, a in row["arms"].items():
            if arm == ref_arm:
                continue
            a["logits"] = worst([compare_logits(runs / f"r{rows}-b{b}-{ref_arm}.logits",
                                                runs / f"r{rows}-b{b}-{arm}.logits", r["slots"], vocab) for b in range(blocks)])
            if a["program"] == ref_row["program"]:
                a["verdict"] = "no effect (same program as the reference)"
                continue
            lg = a["logits"]
            bad = list(a["why"])
            if not lg.get("finite"):
                bad.append(f"accuracy incomplete: {lg.get('why')}")
            elif kl_max is not None and not lg["bitwise"] and lg["kl_max"] > kl_max:
                bad.append(f"kl_max {lg['kl_max']:.3g} > {kl_max}")
            elif top1_min is not None and not lg["bitwise"] and lg["top1"] < top1_min:
                bad.append(f"top1 {lg['top1']:.2f} < {top1_min}")
            if bad:
                a["verdict"] = "failed: " + "; ".join(bad)
                continue
            a["accuracy_accepted"] = lg["bitwise"] or (kl_max is not None and top1_min is not None)
            a["timing"] = noise_rule(ref_row["block_ms"], a["block_ms"])
            a["verdict"] = "faster" if a["timing"]["faster"] else "not faster beyond noise"
            if a["timing"]["faster"] and (best is None or a["timing"]["saving_ms"] > row["arms"][best]["timing"]["saving_ms"]):
                best = arm
        if best is None:
            row.update(winner=ref_row["value"], winner_arm=ref_arm, status="keep", accuracy_accepted=True)
        else:
            w = row["arms"][best]
            row.update(winner=w["value"], winner_arm=best, accuracy_accepted=w["accuracy_accepted"],
                       status="switch (bitwise)" if w["logits"]["bitwise"]
                       else "switch (numerics changed: FP32 gate on the mixed packet)")
        out_rungs.append(row)
    return {"knob": knob, "values": values, "rungs": out_rungs, "promotable": False, "requires": REQUIRES,
            "metric": "host wall ms per packed prefill launch at the rung's rows, graph on, rep 0 dropped, "
                      "median per block, ABBA blocks"}


def qualify(choice: dict, res: dict[str, dict[str, dict]], roofline: str | None, library: str | None) -> dict:
    """`qualified` only with every rung decided, accuracy accepted, no spilling object in a chosen
    arm that differs from the reference arm, a calibrated roofline record and a library-kernel
    resource record (cuBLASLt kernels are not in the packet's cubins); else experimental."""
    why = []
    for r in choice["rungs"]:
        if r.get("winner") is None:
            why.append(f"{r['rows']}: {r.get('status')}")
            continue
        if not r.get("accuracy_accepted"):
            why.append(f"{r['rows']}: accuracy not accepted (non-bitwise; give --kl-max and --top1-min)")
        arm, ref = r.get("winner_arm"), r.get("reference_arm")
        if arm == ref:
            continue
        # Every object of a switched arm, identical bytes included: the override can route to
        # a different kernel inside an unchanged cubin.
        if not res.get(arm):
            why.append(f"{r['rows']}: no resource record for {arm}")
        for name, x in res.get(arm, {}).items():
            if x.get("error") or any(x.get(f) is None for f in ("reg", "stack", "shared", "local")):
                why.append(f"{r['rows']}: {arm} object {name} resources unknown ({x.get('error', 'cuobjdump gave none')})")
            elif x["local"] > 0:
                why.append(f"{r['rows']}: {arm} object {name} spills (LOCAL {x['local']})")
    if not roofline:
        why.append("no calibrated roofline record (--roofline; op_roof.py / packet_roofline.py)")
    if not library:
        why.append("no library-kernel (cuBLASLt) resource record (--library-resources; gemma4_route_matrix.sh)")
    for path in (roofline, library):
        if path and (not Path(path).is_file() or Path(path).stat().st_size == 0):
            why.append(f"{path}: missing or empty")
    return {"label": "experimental" if why else "qualified", "why": why, "roofline": roofline,
            "library_resources": library}


def switches(choice: dict, recipe: dict) -> dict[int, str]:
    """rows -> winning value, for the rungs whose winner differs from the recipe's packet-wide value."""
    packet = (recipe.get("emit") or {}).get("env", {}).get(choice["knob"])
    canon = lambda v: (rung_env.boolish(v) or v) if v is not None else None
    return {r["rows"]: r["winner"] for r in choice["rungs"]
            if r.get("winner") is not None and canon(r["winner"]) != canon(packet)}


def snippet(choice: dict, recipe: dict) -> str:
    """[emit.rung_env] entries for the swept knob: one per rung whose winner differs from the
    packet-wide value, and the recipe's existing entries for it that must go."""
    knob = choice["knob"]
    packet = (recipe.get("emit") or {}).get("env", {}).get(knob)
    want = {("prefill", rows): {knob: v} for rows, v in switches(choice, recipe).items()}
    swept = {r["rows"] for r in choice["rungs"] if r.get("winner") is not None}
    stale = sorted(rows for (phase, rows), knobs in rung_env.from_recipe(recipe).items()
                   if phase == "prefill" and knob in knobs and rows in swept and ("prefill", rows) not in want)
    q = choice.get("qualification") or {"label": "experimental", "why": ["not qualified"]}
    head = [f"# rung-sweep {knob}: packet-wide value {packet!r} ([emit.env]); per-rung winners below.",
            f"# {q['label'].upper()}: " + "; ".join(q["why"]) + ".",
            "# NOT promotable as is: " + "; ".join(choice["requires"]) + "."]
    head += [f"# delete {knob} from [emit.rung_env.prefill.{rows}] (now the packet-wide value)" for rows in stale]
    return "\n".join(head) + "\n\n" + rung_env.toml_snippet(want)


def table(choice: dict) -> str:
    knob = choice["knob"]
    arms = [arm_name(v) for v in choice["values"]]
    q = choice.get("qualification") or {"label": "experimental", "why": []}
    out = [f"# rung-sweep: {knob} ({q['label']})", "", choice["metric"] + ".", "",
           "| rows | ref | " + " | ".join(f"{a} ms (blocks)" for a in arms) + " | saving | noise | logits vs ref | winner | status |",
           "|---|---|" + "---|" * len(arms) + "---|---|---|---|---|"]
    for r in choice["rungs"]:
        cells = []
        for a in arms:
            x = r["arms"].get(a, {})
            ms = "/".join(f"{m:.2f}" if m is not None else "-" for m in x.get("block_ms", []))
            cells.append(ms + ("" if x.get("ok", True) else " FAIL"))
        others = [x for a, x in r["arms"].items() if a != r.get("reference_arm")]
        t = next((x["timing"] for x in others if "timing" in x), None)
        lg = next((x.get("logits") for x in others if x.get("logits")), {}) or {}
        acc = ("bitwise" if lg.get("bitwise") else
               f"top1 {lg['top1']:.2f} KL {lg['kl_mean']:.3g}/{lg['kl_max']:.3g}" if lg.get("complete") and lg.get("finite") else
               "incomplete: " + str(lg.get("why", "-"))[:60])
        sav = f"{t['saving_ms']:+.2f} ms ({t['saving_pct']:+.1f}%)" if t else "-"
        noise = f"{t['noise_ms']:.2f}" + ("" if t["convictable"] else " (spread)") if t else "-"
        out.append(f"| {r['rows']} | {r.get('reference_value')} | " + " | ".join(cells)
                   + f" | {sav} | {noise} | {acc} | {r.get('winner')} | {r.get('status')} |")
    out += ["", "Saving = reference arm minus the other arm (positive: the other arm is faster). "
            "Logits: worst over every block, same launches as the timing.",
            f"Label {q['label']}: " + "; ".join(q["why"] or ["all checks passed"]) + ".",
            "Not promotable until: " + "; ".join(choice["requires"]) + "."]
    return "\n".join(out) + "\n"


def resource_table(res: dict[str, dict[str, dict]]) -> str:
    """Cubins whose bytes differ between arms, with REG/STACK/SHARED/LOCAL; spills flagged."""
    arms = sorted(res)
    names = sorted({n for a in arms for n in res[a]})
    out = ["| object | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    same = 0
    for n in names:
        shas = {res[a].get(n, {}).get("sha256") for a in arms}
        if len(shas) == 1:
            same += 1
            continue
        cell = lambda r: ("absent" if not r else f"REG {r.get('reg')} STACK {r.get('stack')} SHARED {r.get('shared')} "
                          f"LOCAL {r.get('local')}" + (" SPILL" if (r.get("local") or 0) > 0 else ""))
        out.append(f"| {n} | " + " | ".join(cell(res[a].get(n)) for a in arms) + " |")
    spills = sorted({f"{a}:{n}" for a in arms for n, r in res[a].items() if (r.get("local") or 0) > 0})
    out += ["", f"{same} object(s) byte-identical across arms (not listed).",
            "Objects with LOCAL > 0 (spills): " + (", ".join(spills) if spills else "none") + "."]
    return "\n".join(out) + "\n"
