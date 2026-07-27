#!/usr/bin/env python3
"""px13_emit_tuning.py — assemble the sm_120a prefill-tile cell from the run logs.

Writes tuning/nvidia/sm_120a/rtx-5090/prefill_tile_measurement.jsonl.

Why a NEW record kind and not tunedb::KernelMeasurement: on NVIDIA the GEMM tile is a
compile-time macro of the interpreter OBJECT, and `plowc tune` says so itself ("the real tuning
axis here is which object is built, not which opcode is emitted").  All three dense-GEMM opcodes
alias to one body, so a KernelMeasurement — keyed by op_case + kernel_id — has no field that can
distinguish BN=128 from BN=64.  The axis this cell measures is the define set, so the record
carries it explicitly.

The SCORE is the end-to-end conc-1 127k prefill wall, per tuning/README-decode-tuner.md §2.
The isolated bench numbers are carried too, but as data, not as the ranking key — PX-13 measured
them disagreeing in SIGN on PGM_GLU_BN.
"""
import json, os, re, subprocess, sys, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = os.path.join(ROOT, "tuning", "nvidia", "sm_120a", "rtx-5090")
LOGS = "/root/px13/logs"
MICRO = "/tmp/px13-out"
CUBIN = "/root/px13/cubin"

# arm -> the -D set that builds it (empty = the shipped default object)
ARMS = {
    "base":       [],
    "glubn64":    ["-DPGM_GLU_BN=64"],
    "stages2":    ["-DPGM_STAGES=2"],
    "stages4":    ["-DPGM_STAGES=4"],
    "stages5":    ["-DPGM_STAGES=5"],
    "glustages3": ["-DPGM_GLU_STAGES=3"],
    "bn64":       ["-DPGM_BN=64", "-DPGM_GLU_BN=64"],
    "bn256":      ["-DPGM_BN=256", "-DPGM_GLU_BN=128"],
    "bm64":       ["-DPGM_BM=64"],
    "bm256":      ["-DPGM_BM=256"],
    "nolds64":    ["-DPGM_W8A8_LDS64=0"],
    "nosw8v2":    ["-DPGM_SW8_V2=0"],
    "st4glu3":    ["-DPGM_STAGES=4", "-DPGM_GLU_STAGES=3"],
}


def res_usage(cub):
    try:
        out = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-res-usage", cub],
                             capture_output=True, text=True).stdout
    except OSError:
        return None, None
    m = re.search(r"interp_sm120_pf.*?\n\s*REG:(\d+) STACK:(\d+)", out, re.S)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def md5(p):
    return subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0]


def walls(arm, extra):
    """Every conc-1 wall recorded for this arm, from the sweep stdout captures."""
    out = []
    for line in extra:
        f = line.split()
        if len(f) >= 3 and f[1] == arm and f[2].startswith("duration_s="):
            v = f[2].split("=", 1)[1]
            if v != "DIED":
                out.append(float(v))
    return out


def micro(arm):
    """full-grid TFLOP/s per shape from the isolated bench at M=1024."""
    p = os.path.join(MICRO, f"e8_micro_{arm}.txt")
    if not os.path.exists(p):
        return None
    tf, shape = {}, None
    for line in open(p):
        f = line.split()
        if len(f) >= 9 and f[0] in ("gate|up", "down", "q_full", "o_full"):
            shape = f[0]
        elif shape and "fullG" in line:
            # ... M T 170 - fullG <ms> <TFLOP/s> <pct> <ratio>
            tf[shape] = float(f[f.index("fullG") + 2])
            shape = None
    return tf or None


def main(extra_paths):
    extra = []
    for p in extra_paths:
        if os.path.exists(p):
            extra += open(p).read().splitlines()

    rows = []
    for arm, defs in ARMS.items():
        cub = os.path.join(CUBIN, f"pf_{arm}.cubin")
        if not os.path.exists(cub):
            continue
        reg, stack = res_usage(cub)
        w = walls(arm, extra)
        serve = os.path.join(LOGS, f"serve-{arm}.log")
        rejected = None
        # plowrt's tracing writes ANSI colour between a field name and its `=`, so the raw
        # text never contains the literal "prefill_buckets=0" — strip the escapes first.
        txt = re.sub(r"\x1b\[[0-9;]*m", "", open(serve, errors="ignore").read()) \
            if os.path.exists(serve) else ""
        if "prefill_buckets=0" in txt:
            rejected = ("arena over the 101376 B dynamic-smem opt-in: plowrt drops the packet's "
                        "prefill buckets and consumes the prompt one decode step at a time")
        rec = {
            "kind": "prefill_tile_measurement",
            "campaign": "px13",
            "hardware": "nvidia/sm_120a/rtx-5090",
            "sku": "RTX 5090", "isa": "sm_120a", "units": 170,
            "driver": "580.159.03", "toolchain": "cuda-13.0",
            "model": "gemma-4-12B-it", "dtype": "fp8_w8a8", "phase": "prefill",
            "ctx": 126976, "prefill_buckets": [128, 512, 1024],
            "arm": arm, "defines": defs,
            "object": {"pf_cubin_md5": md5(cub), "registers": reg, "stack_bytes": stack},
            "score": {
                "metric": "conc1_prefill_wall_s",
                "protocol": ("vllm bench serve, 1 prompt x 126976 in / 8 out, conc 1, "
                             "--ignore-eos --seed 0; wall == serial prefill (PX-12 sec 2)"),
                "samples": w,
                "median_s": round(statistics.median(w), 3) if w else None,
            },
            "microbench": {
                "bench": "perf-data/px9_gemm_body_bench.cu",
                "note": ("recorded, NOT the ranking key. PX-13 measured microbench and runtime "
                         "disagreeing in SIGN on PGM_GLU_BN."),
                "m": 1024, "grid": "full", "l2": "cold", "arena_pad_bytes": 89104,
                "tflops_full_grid": micro(arm),
            },
            "correctness": "bit_exact_vs_base" if arm in ("glubn64", "bn64", "nolds64", "nosw8v2")
                           else "unchecked",
            "state": "rejected" if rejected else ("qualified" if w else "provisional"),
        }
        if rejected:
            rec["reason"] = rejected
        rows.append(rec)

    base = next((r for r in rows if r["arm"] == "base" and r["score"]["median_s"]), None)
    for r in rows:
        m = r["score"]["median_s"]
        r["score"]["vs_base"] = round(m / base["score"]["median_s"], 4) if (m and base) else None

    os.makedirs(CELL, exist_ok=True)
    out = os.path.join(CELL, "prefill_tile_measurement.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    for r in sorted(rows, key=lambda r: (r["score"]["median_s"] or 1e9)):
        print(f"  {r['arm']:<12} {str(r['score']['median_s']):>7} s  "
              f"{str(r['score']['vs_base']):>6}x  reg={r['object']['registers']}  {r['state']}")


if __name__ == "__main__":
    main(sys.argv[1:])
