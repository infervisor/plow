#!/usr/bin/env python3
"""Host resources of a serving process tree: memory, CPU, threads, mapped libraries, footprint.

    host_res.py sample --root-pid PID --json FILE [--tsv FILE] [--interval-ms 2000] [--metrics-url URL]
    host_res.py cells RESDIR             # RESDIR/host.tsv split by RESDIR/cells.log -> <tag>.host.json
    host_res.py footprint RESDIR --side plow|vllm [--plowrt BIN] [--assets DIR] [--pyref PY] [--hf DIR]
    host_res.py report PLOW_RES VLLM_RES [--cell TAG]   # markdown table, plow / vLLM ratios

`sample` walks the root pid's live descendants every interval (vLLM's EngineCore and workers are
children of the API server) until SIGTERM/SIGINT, writing one TSV row per sample and a summary
JSON: RSS sum (and Pss with --pss), VmLck/VmPin, threads, processes, cores busy (utime+stime
delta per pid), system MemAvailable, every mapped shared library (linked or dlopened) with its
size, and, with --metrics-url, the plowrt KV store's pinned host tier bytes. Peak = max over
samples; steady = median. CUDA pinned host memory (cuMemHostAlloc) counts in RSS, not VmPin.
Pss is opt-in: smaps_rollup walks every page table entry under the target's mmap lock, 0.3 s per
16 GiB resident (1.6 s on a plowrt holding an 80 GiB host tier), which stalls the server's own
mmap/fault path and moved plowrt's TTFT P99 by seconds. /proc/<pid>/status and stat are O(1).
`footprint` sizes what a deploy carries: the plowrt binary, the packet dir, and for vLLM the
interpreter's site-packages (largest components) and the JIT/compile caches.
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_peak_mem import pss_kib, tree  # noqa: E402

COLS = ["t", "procs", "threads", "rss_kib", "pss_kib", "anon_kib", "lck_kib", "pin_kib",
        "cpu_s", "dt_s", "avail_kib", "tier_host_bytes"]
TICK = os.sysconf("SC_CLK_TCK")
LIB = re.compile(r"\.so(\.[0-9.]+)?$")


def status(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "RssAnon", "VmLck", "VmPin", "Threads"):
                    out[k] = int(v.split()[0])
    except OSError:
        return None
    return out


def cpu_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            f = f.read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])  # utime, stime (fields 14, 15)
    except (OSError, IndexError, ValueError):
        return None


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:160]
    except OSError:
        return ""


def comm(pid):
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return ""


def mapped_libs(pid):
    libs = set()
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                p = line.split(None, 5)
                if len(p) == 6 and LIB.search(p[5].strip()):
                    libs.add(p[5].strip())
    except OSError:
        pass
    return libs


def mem_available_kib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


def tier_host_bytes(url):
    """Sum of plowrt_kv_store_host_bytes over models, or None (no tier, not plowrt, unreachable)."""
    try:
        text = urllib.request.urlopen(url, timeout=1).read().decode(errors="replace")
    except OSError:
        return None
    vals = [float(ln.rsplit(None, 1)[1]) for ln in text.splitlines() if ln.startswith("plowrt_kv_store_host_bytes{")]
    return int(sum(vals)) if vals else None


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"peak": max(values), "min": min(values), "steady": statistics.median(values)}


def summarize(rows):
    """Summary of TSV rows (dicts of COLS); memory in MiB, CPU in cores."""
    if not rows:
        return None
    mib = lambda k: stat([r[k] / 1024 for r in rows if r[k] is not None])  # noqa: E731
    busy = [r["cpu_s"] / r["dt_s"] for r in rows if r["dt_s"] > 0]
    span = sum(r["dt_s"] for r in rows)
    tier = [r["tier_host_bytes"] / 2**20 for r in rows if r["tier_host_bytes"] is not None]
    out = {
        "samples": len(rows), "t0": rows[0]["t"], "t1": rows[-1]["t"],
        "rss_mib": mib("rss_kib"), "pss_mib": mib("pss_kib"), "anon_mib": mib("anon_kib"),
        "locked_mib": mib("lck_kib"), "pinned_mib": mib("pin_kib"),
        "threads": stat([r["threads"] for r in rows]), "procs": stat([r["procs"] for r in rows]),
        "cores": {"avg": sum(r["cpu_s"] for r in rows) / span if span else None,
                  "peak": max(busy) if busy else None},
        "mem_available_mib": mib("avail_kib"),
        "tier_host_mib": stat(tier),
    }
    return {k: v for k, v in out.items() if v is not None}


def read_tsv(path):
    rows = []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            r = dict(zip(head, vals))
            rows.append({k: (None if r.get(k, "") == "" else float(r[k])) for k in COLS})
    return rows


def sample(a):
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    tsv = open(a.tsv, "w") if a.tsv else None
    if tsv:
        tsv.write("\t".join(COLS) + "\n")
    rows, procs, libs = [], {}, set()
    prev, prev_t = {}, None
    while not stop:
        t = time.time()
        pids = tree(a.root_pid)
        agg = dict(procs=0, threads=0, rss_kib=0, pss_kib=0, anon_kib=0, lck_kib=0, pin_kib=0)
        ticks = {}
        for pid in pids:
            st = status(pid)
            if st is None or "VmRSS" not in st:
                continue  # exited, or a kernel thread
            agg["procs"] += 1
            agg["threads"] += st.get("Threads", 0)
            agg["rss_kib"] += st["VmRSS"]
            agg["anon_kib"] += st.get("RssAnon", 0)
            agg["lck_kib"] += st.get("VmLck", 0)
            agg["pin_kib"] += st.get("VmPin", 0)
            if a.pss:
                agg["pss_kib"] += pss_kib(pid) or 0
            tk = cpu_ticks(pid)
            if tk is not None:
                ticks[pid] = tk
            p = procs.setdefault(pid, {"pid": pid, "comm": comm(pid), "cmd": cmdline(pid), "rss_mib_peak": 0})
            p["rss_mib_peak"] = max(p["rss_mib_peak"], round(st["VmRSS"] / 1024))
            libs |= mapped_libs(pid)
        if agg["procs"]:
            # A pid new since the last sample started inside the interval: all its ticks count.
            cpu = sum(v - prev.get(p, 0) for p, v in ticks.items()) / TICK
            if not a.pss:
                agg["pss_kib"] = None
            r = dict(agg, t=t, cpu_s=cpu if prev_t else 0.0, dt_s=(t - prev_t) if prev_t else 0.0,
                     avail_kib=mem_available_kib(),
                     tier_host_bytes=tier_host_bytes(a.metrics_url) if a.metrics_url else None)
            rows.append(r)
            if tsv:
                tsv.write("\t".join("" if r[k] is None else (f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]))
                                    for k in COLS) + "\n")
                tsv.flush()
            prev, prev_t = ticks, t
        time.sleep(max(0.0, a.interval_ms / 1000 - (time.time() - t)))
    out = summarize(rows)
    if out is None:
        return
    sizes = {}
    for lib in sorted(libs):
        try:
            sizes[lib] = os.stat(lib).st_size
        except OSError:
            pass
    out.update(interval_ms=a.interval_ms, processes=sorted(procs.values(), key=lambda p: p["pid"]),
               libs=sizes, libs_bytes=sum(sizes.values()))
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


def cells(a):
    res = Path(a.resdir)
    rows = read_tsv(res / "host.tsv")
    begin = {}
    for line in (res / "cells.log").read_text().splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "CELL_BEGIN":
            begin[f[1]] = float(f[2])
        elif len(f) == 3 and f[0] == "CELL_END" and f[1] in begin:
            s = summarize([r for r in rows if begin[f[1]] <= r["t"] <= float(f[2])])
            if s:
                (res / f"{f[1]}.host.json").write_text(json.dumps(s, indent=1) + "\n")


def du(path, follow=False):
    """Bytes of regular files under path; symlinks are skipped (a packet links its checkpoint) unless
    follow (an HF snapshot is all links into its blobs)."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                f = os.path.join(root, f)
                if follow or not os.path.islink(f):
                    total += os.stat(f).st_size
            except OSError:
                continue
    return total


def site_packages(pyref):
    r = subprocess.run([pyref, "-c", "import json, site, sys; print(json.dumps([p for p in sys.path if "
                        "p.endswith(('site-packages', 'dist-packages'))]))"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout) if r.returncode == 0 else []
    except ValueError:
        return []


def footprint(a):
    res = Path(a.resdir)
    out = {"side": a.side}
    if a.hf:
        out["checkpoint_bytes"] = du(Path(a.hf).resolve(), follow=True)
    if a.side == "plow":
        binary = Path(a.plowrt or res / "plowrt")
        out["plowrt_bytes"] = binary.stat().st_size if binary.is_file() else None
        if a.assets:
            out["packet_dir_bytes"] = du(a.assets)
            pkt = Path(a.assets) / "model.pkt"
            out["packet_bytes"] = pkt.stat().st_size if pkt.is_file() else None
    else:
        dirs = {}
        for sp in site_packages(a.pyref) if a.pyref else []:
            if not os.path.isdir(sp):
                continue
            items = {e.name: du(e.path) for e in os.scandir(sp)}
            dirs[sp] = {"bytes": sum(items.values()),
                        "top": dict(sorted(items.items(), key=lambda kv: -kv[1])[:12])}
        out["site_packages"] = dirs
        out["site_packages_bytes"] = sum(d["bytes"] for d in dirs.values())
        home = Path.home()
        xdg = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
        caches = {
            "vllm": os.environ.get("VLLM_CACHE_ROOT", str(xdg / "vllm")),
            "flashinfer": str(xdg / "flashinfer"), "deep_gemm": str(home / ".deep_gemm"),
            "triton": os.environ.get("TRITON_CACHE_DIR", str(home / ".triton" / "cache")),
            "torchinductor": os.environ.get("TORCHINDUCTOR_CACHE_DIR",
                                            f"{os.environ.get('TMPDIR', '/tmp')}/torchinductor_{os.environ.get('USER', '')}"),
        }
        out["caches"] = {k: {"path": v, "bytes": du(v)} for k, v in caches.items() if os.path.isdir(v)}
    host = res / "host.json"
    if host.is_file():
        h = json.loads(host.read_text())
        out["mapped_libs_bytes"] = h.get("libs_bytes")
    ready = res / "ready_s.txt"
    if ready.is_file():
        out["ready_s"] = float(ready.read_text())
    (res / "footprint.json").write_text(json.dumps(out, indent=1) + "\n")


def load(res, cell):
    res = Path(res)
    r = {"fp": json.loads((res / "footprint.json").read_text()) if (res / "footprint.json").is_file() else {}}
    r["session"] = json.loads((res / "host.json").read_text()) if (res / "host.json").is_file() else {}
    r["cell"] = json.loads((res / f"{cell}.host.json").read_text()) if (res / f"{cell}.host.json").is_file() else {}
    p = res / f"{cell}.peak_gpu_memory_mib.txt"
    r["gpu_mib"] = float(p.read_text()) if p.is_file() and p.read_text().strip() else None
    j = res / f"{cell}.json"
    if j.is_file():
        d = json.loads(j.read_text())
        if "overall" in d:
            o = d["overall"]
            r["serve"] = dict(tok_s=o.get("total_tok_s"), ttft_p99=o.get("ttft_p99_ms"), tpot_p99=o.get("tpot_p99_ms"))
    else:
        for j in (res / cell / "bench.json", res / cell / "main" / "bench.json"):
            if j.is_file():
                d = json.loads(j.read_text())
                r["serve"] = dict(tok_s=d.get("total_token_throughput"), ttft_p99=d.get("p99_ttft_ms"),
                                  tpot_p99=d.get("p99_tpot_ms"))
    t = res / f"{cell}.client_time.txt"
    if t.is_file():
        f = t.read_text().split()
        if len(f) >= 3:
            r["client_cores"] = (float(f[1]) + float(f[2])) / float(f[0]) if float(f[0]) else None
    return r


def report(a):
    cell = a.cell
    if cell is None:
        tags = sorted(p.name[:-len(".host.json")] for p in Path(a.plow).glob("*.host.json"))
        cell = tags[0] if tags else ""
    p, v = load(a.plow, cell), load(a.vllm, cell)

    def g(d, *path):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    def fmt(x, unit, nd=0):
        if x is None:
            return "-"
        return f"{x:,.{nd}f} {unit}".strip()

    gib = lambda b: b / 2**30 if b is not None else None  # noqa: E731
    rows = [
        ("Host RSS peak (cell)", g(p, "cell", "rss_mib", "peak"), g(v, "cell", "rss_mib", "peak"), "MiB", 0),
        ("Host RSS steady (cell median)", g(p, "cell", "rss_mib", "steady"), g(v, "cell", "rss_mib", "steady"), "MiB", 0),
        ("Host PSS peak (cell)", g(p, "cell", "pss_mib", "peak"), g(v, "cell", "pss_mib", "peak"), "MiB", 0),
        ("Host PSS steady (cell median)", g(p, "cell", "pss_mib", "steady"), g(v, "cell", "pss_mib", "steady"), "MiB", 0),
        ("Host RSS peak (whole session)", g(p, "session", "rss_mib", "peak"), g(v, "session", "rss_mib", "peak"), "MiB", 0),
        ("  of which pinned KV host tier (peak)", g(p, "session", "tier_host_mib", "peak"),
         g(v, "session", "tier_host_mib", "peak"), "MiB", 0),
        ("Anon RSS peak (session)", g(p, "session", "anon_mib", "peak"), g(v, "session", "anon_mib", "peak"), "MiB", 0),
        ("VmLck peak (mlock)", g(p, "session", "locked_mib", "peak"), g(v, "session", "locked_mib", "peak"), "MiB", 0),
        ("VmPin peak (pinned pages)", g(p, "session", "pinned_mib", "peak"), g(v, "session", "pinned_mib", "peak"), "MiB", 0),
        ("System MemAvailable min", g(p, "session", "mem_available_mib", "min"),
         g(v, "session", "mem_available_mib", "min"), "MiB", 0),
        ("CPU cores busy avg (cell)", g(p, "cell", "cores", "avg"), g(v, "cell", "cores", "avg"), "cores", 2),
        ("CPU cores busy peak (cell, per interval)", g(p, "cell", "cores", "peak"), g(v, "cell", "cores", "peak"), "cores", 2),
        ("Client CPU cores avg", p.get("client_cores"), v.get("client_cores"), "cores", 2),
        ("Threads (cell peak)", g(p, "cell", "threads", "peak"), g(v, "cell", "threads", "peak"), "", 0),
        ("Processes (cell peak)", g(p, "cell", "procs", "peak"), g(v, "cell", "procs", "peak"), "", 0),
        ("Server binary / site-packages", gib(g(p, "fp", "plowrt_bytes")), gib(g(v, "fp", "site_packages_bytes")), "GiB", 2),
        ("Mapped shared libraries", gib(g(p, "fp", "mapped_libs_bytes")), gib(g(v, "fp", "mapped_libs_bytes")), "GiB", 2),
        ("Packet dir / compile caches", gib(g(p, "fp", "packet_dir_bytes")),
         gib(sum(c["bytes"] for c in (g(v, "fp", "caches") or {}).values()) or None), "GiB", 2),
        ("GPU memory peak (cell)", p.get("gpu_mib"), v.get("gpu_mib"), "MiB", 0),
        ("Server ready time", g(p, "fp", "ready_s"), g(v, "fp", "ready_s"), "s", 1),
        ("Total throughput", g(p, "serve", "tok_s"), g(v, "serve", "tok_s"), "tok/s", 0),
        ("TTFT P99", g(p, "serve", "ttft_p99"), g(v, "serve", "ttft_p99"), "ms", 1),
        ("TPOT P99", g(p, "serve", "tpot_p99"), g(v, "serve", "tpot_p99"), "ms", 2),
    ]
    print(f"cell {cell}\n")
    print("| metric | plowrt | vLLM | plowrt / vLLM |\n|---|---|---|---|")
    for name, x, y, unit, nd in rows:
        ratio = f"{x / y:.2f}x" if x is not None and y not in (None, 0) else "-"
        print(f"| {name} | {fmt(x, unit, nd)} | {fmt(y, unit, nd)} | {ratio} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--root-pid", type=int, required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--tsv")
    s.add_argument("--interval-ms", type=int, default=2000)
    s.add_argument("--metrics-url")
    s.add_argument("--pss", action="store_true", help="also sum Pss (smaps_rollup; stalls large servers)")
    c = sub.add_parser("cells")
    c.add_argument("resdir")
    f = sub.add_parser("footprint")
    f.add_argument("resdir")
    f.add_argument("--side", required=True, choices=("plow", "vllm", "llamacpp"))
    f.add_argument("--plowrt")
    f.add_argument("--assets")
    f.add_argument("--pyref")
    f.add_argument("--hf")
    r = sub.add_parser("report")
    r.add_argument("plow")
    r.add_argument("vllm")
    r.add_argument("--cell")
    a = ap.parse_args(argv)
    {"sample": sample, "cells": cells, "footprint": footprint, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
