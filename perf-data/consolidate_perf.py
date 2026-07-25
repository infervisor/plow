#!/usr/bin/env python3
"""
consolidate_perf.py — regenerate the consolidated perf index from the per-campaign
source files in this directory.

Emits (into perf-data/):
  - all-perf-data.json  : {metadata, rows:[...]}   one row = one measured value
  - all-perf-data.csv   : the same rows, flat

Design
------
Every measurement (across heterogeneous JSON + markdown-table sources) is flattened
to ONE schema:

  model      canonical model name  (gemma-4-31B, gemma-4-12B, llama-3.1-8B,
                                     qwen3-4B, qwen3-1.7B)
  engine     "plow" | "vllm"
  precision  "bf16" | "fp8" (weight-only) | "fp8kv" (fp8 weights + fp8 KV)
  phase      "decode" | "prefill"
  tp         tensor-parallel degree (int)
  ctx        input context length in tokens (int)
  metric     tpot_ms | ttft_ms | itl_ms | prefill_ms | prefill_tok_s |
             decode_tok_s | decode_ms_per_token | gemv_TBps |
             decode_step_mean_ms | decode_kernel_<name>_ms   (unit is in the name)
  value      float (transcribed verbatim from the source; never interpolated)
  source_file  originating file in perf-data/
  campaign   short label for the measurement run
  version    engine build/version (vLLM version string, or plow branch/source)
  git_commit short commit the source file was committed under ("(uncommitted)"
             for files not yet committed when this index was generated)
  date       measurement date (YYYY-MM-DD)
  notes      per-entry caveats / provenance

FAITHFULNESS: only values literally present in the sources are transcribed. Derived
analysis columns in the sources (ratios, crossover estimates, scaling multipliers,
speedup %) are NOT emitted as rows. The four summary markdown files
(decode-only-sweep.md, gemma4-31b-longctx-sweep.md, vllm-docker-baseline.md,
vllm-fp8-baseline.md) mirror numbers already carried by their JSON siblings and are
NOT re-transcribed. Markdown files that carry data found in NO json
(gemma4-31b-tp-prefill*.md, plow-vs-vllm-baseline.md, vllm-tp-baseline.md) are
transcribed below as literals.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Latest git-commit date across the perf-data sources (used as the generation stamp;
# passed as a literal so this script never depends on `date`/network).
GENERATION_DATE = "2026-07-17"

# short commit each source file was committed under (git log -1 -- <file>)
COMMIT = {
    "decode-only-sweep.json": "30b0cd3",
    "gemma4-31b-longctx-sweep.json": "3a2749a",
    "gemma4-31b-tp-prefill.md": "293d9b0",
    "gemma4-31b-tp-prefill-2shot.md": "7ee6dfc",
    "gemma4-vllm-fp8.json": "e70d6d1",
    "llama8b-vllm-fp8.json": "e70d6d1",
    "qwen3-4b-vllm-fp8.json": "e70d6d1",
    "gemma4-vllm-perf.json": "7659947",
    "llama-3.1-8b-vllm-docker.json": "d69c9d3",
    "llama-3.1-8b-vllm-perf.json": "7659947",
    "qwen3-1.7b-vllm-perf.json": "a109dcf",
    "qwen3-4b-vllm-docker.json": "d69c9d3",
    "qwen3-4b-vllm-perf.json": "a109dcf",
    "plow-vs-vllm-baseline.md": "d69c9d3",
    # untracked when this index was generated:
    "gemma4-31b-vllm-tp.json": "(uncommitted)",
    "vllm_decode_breakdown.json": "(uncommitted)",
    "glm52-vllm-decode.json": "9e3d3d1",
    "glm52-plow-decode.json": "48452bd",
    "glm52-plow-decode-tuned.json": "41573bb",
    "glm52-plow-decode-256k.json": "e65ac0c",
    "vllm-tp-baseline.md": "(uncommitted)",
    "gemma4-12b-plowrt-served-ttft.json": "(uncommitted)",
}

# --- model-name normalisation -------------------------------------------------
def canon_model(raw):
    r = raw.rsplit("/", 1)[-1].lower()
    if "gemma-4-31b" in r or "gemma-4-31" in r:
        return "gemma-4-31B"
    if "gemma-4-26b-a4b" in r:
        return "gemma-4-26B-A4B"
    if "gemma-4-12b" in r:
        return "gemma-4-12B"
    if "llama-3.1-8b" in r or "llama" in r:
        return "llama-3.1-8B"
    if "qwen3-4b" in r:
        return "qwen3-4B"
    if "qwen3-1.7b" in r:
        return "qwen3-1.7B"
    return raw

ROWS = []

def add(model, engine, precision, phase, tp, ctx, metric, value,
        source_file, campaign, version, date, notes=""):
    if value is None:
        return
    ROWS.append({
        "model": canon_model(model),
        "engine": engine,
        "precision": precision,
        "phase": phase,
        "tp": tp,
        "ctx": ctx,
        "metric": metric,
        "value": value,
        "source_file": source_file,
        "campaign": campaign,
        "version": version,
        "git_commit": COMMIT.get(source_file, "(uncommitted)"),
        "date": date,
        "notes": notes,
    })

def load(fn):
    with open(os.path.join(HERE, fn)) as f:
        return json.load(f)

# =============================================================================
# 1. decode-only-sweep.json  — DEFINITIVE Gemma-4-31B bf16 decode sweep
# =============================================================================
def do_decode_only_sweep():
    fn = "decode-only-sweep.json"
    d = load(fn)
    date = d["date"]
    plow_ver = "branch " + d["plow"]["branch"]
    vllm_ver = d["vllm"]["version"]
    for c in d["grid"]:
        tp, ctx = c["tp"], c["ctx"]
        add("gemma-4-31B", "plow", "bf16", "decode", tp, ctx, "tpot_ms",
            c["plow_ms_tok"], fn, "decode-only-sweep", plow_ver, date,
            "definitive single-build idle-node sweep; monotonic in ctx; "
            "device==host verified; supersedes longctx-sweep plow decode.")
        vsrc = ("vLLM TP1 fresh this session; TP4/TP8 1k-32k from gemma4-31b-vllm-tp.json; "
                "TP4/TP8 48k-128k from gemma4-31b-longctx-sweep.json") if tp in (4, 8) \
               else "vLLM TP1 fresh this session (single GPU)"
        add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "tpot_ms",
            c["vllm_ms_tok"], fn, "decode-only-sweep", vllm_ver, date,
            vsrc + "; TRITON_ATTN, cudagraphs, not bit-exact.")

# =============================================================================
# 2. gemma4-31b-longctx-sweep.json — plow prefill + long-ctx decode vs vLLM
# =============================================================================
def do_longctx_sweep():
    fn = "gemma4-31b-longctx-sweep.json"
    d = load(fn)
    date = d["date"]
    vllm_ver = d["vllm"]["version"]
    plow_ver = "branch " + d["plow"]["branch"]
    seen_prefill = set()
    seen_tp1 = set()
    for p in d["points"]:
        ctx, tp = p["ctx"], p["tp"]
        # plow prefill is single-GPU (TP1), identical across the TP4/TP8 columns
        if ctx not in seen_prefill:
            seen_prefill.add(ctx)
            note = ("plow single-GPU (TP1) prefill; plow had NO TP prefill in this "
                    "campaign, same value fills the TP4 & TP8 columns.")
            add("gemma-4-31B", "plow", "bf16", "prefill", 1, ctx, "prefill_ms",
                p["plow_prefill_ms"], fn, "longctx-sweep", plow_ver, date, note)
            add("gemma-4-31B", "plow", "bf16", "prefill", 1, ctx, "prefill_tok_s",
                p["plow_prefill_tok_s"], fn, "longctx-sweep", plow_ver, date, note)
        if ctx not in seen_tp1:
            seen_tp1.add(ctx)
            add("gemma-4-31B", "plow", "bf16", "decode", 1, ctx, "tpot_ms",
                p["plow_tp1_decode_ms"], fn, "longctx-sweep", plow_ver, date,
                "TP1 decode reference recorded alongside this longctx campaign.")
        # plow sharded decode (superseded by decode-only-sweep except 72k)
        supnote = ("SUPERSEDED for overlapping ctx by decode-only-sweep.json "
                   "(~0.5-1.0 ms/tok faster here, non-monotonic when stitched); "
                   "72k point is UNIQUE to this file.")
        add("gemma-4-31B", "plow", "bf16", "decode", tp, ctx, "tpot_ms",
            p["plow_decode_ms_tok"], fn, "longctx-sweep", plow_ver, date, supnote)
        # vLLM
        tp8note = " vLLM TP8 long-ctx run under node contention (see md)." if tp == 8 else ""
        add("gemma-4-31B", "vllm", "bf16", "prefill", tp, ctx, "ttft_ms",
            p["vllm_ttft_ms"], fn, "longctx-sweep", vllm_ver, date,
            "TRITON_ATTN, cudagraphs, not bit-exact." + tp8note)
        add("gemma-4-31B", "vllm", "bf16", "prefill", tp, ctx, "prefill_tok_s",
            p["vllm_prefill_tok_s"], fn, "longctx-sweep", vllm_ver, date,
            "derived L/TTFT; vLLM TTFT at 64k/72k anomalously low (torch.compile "
            "specialisation) so tok/s there is optimistic." + tp8note)
        add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "tpot_ms",
            p["vllm_decode_ms_tok"], fn, "longctx-sweep", vllm_ver, date,
            "TRITON_ATTN, cudagraphs, not bit-exact." + tp8note)

# =============================================================================
# 3. gemma4-31b-vllm-tp.json — vLLM TP2/4/8 decode+prefill, Gemma-4-31B bf16
# =============================================================================
def do_vllm_tp():
    fn = "gemma4-31b-vllm-tp.json"
    d = load(fn)
    date = d["date"]
    ver = d["vllm_version"]
    for cfg in d["configs"]:
        tp = cfg["tp"]
        for r in cfg["results"]:
            ctx = r["ctx"]
            warm = " ctx=1024 TTFT warm-up-contaminated (first-request torch.compile/cudagraph)." \
                   if ctx == 1024 else ""
            add("gemma-4-31B", "vllm", "bf16", "prefill", tp, ctx, "ttft_ms",
                r["ttft_ms"], fn, "vllm-tp-baseline", ver, date,
                "TRITON_ATTN, cudagraphs, not bit-exact." + warm)
            add("gemma-4-31B", "vllm", "bf16", "prefill", tp, ctx, "prefill_tok_s",
                r["prefill_tok_s"], fn, "vllm-tp-baseline", ver, date, "")
            add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "tpot_ms",
                r["tpot_ms"], fn, "vllm-tp-baseline", ver, date, "")
            add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "itl_ms",
                r["itl_ms"], fn, "vllm-tp-baseline", ver, date, "")
            add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "decode_tok_s",
                r["decode_tok_s"], fn, "vllm-tp-baseline", ver, date, "")

# =============================================================================
# 4. {gemma4,llama8b,qwen3-4b}-vllm-fp8.json — vLLM bf16/fp8/fp8kv decode+prefill
# =============================================================================
def do_vllm_fp8(fn):
    d = load(fn)
    date = d["date"]
    ver = d["vllm_version"]
    model = d["model"]
    for cfg in d["configs"]:
        prec = cfg["config"]  # bf16 | fp8 | fp8kv
        for r in cfg["results"]:
            ctx = r.get("ctx", r.get("input_len"))
            warm = " ctx=1024 TTFT warm-up-contaminated (HIP-graph capture); 4k+ clean." \
                   if ctx == 1024 else ""
            add(model, "vllm", prec, "prefill", 1, ctx, "ttft_ms",
                r["ttft_ms"], fn, "vllm-fp8-baseline", ver, date, warm.strip())
            add(model, "vllm", prec, "prefill", 1, ctx, "prefill_tok_s",
                r["prefill_tok_s"], fn, "vllm-fp8-baseline", ver, date, "")
            add(model, "vllm", prec, "decode", 1, ctx, "tpot_ms",
                r["tpot_ms"], fn, "vllm-fp8-baseline", ver, date,
                "decode metric == itl_ms; TPOT unaffected by 1k warm-up artifact.")
            add(model, "vllm", prec, "decode", 1, ctx, "itl_ms",
                r["itl_ms"], fn, "vllm-fp8-baseline", ver, date, "")
            add(model, "vllm", prec, "decode", 1, ctx, "decode_tok_s",
                r["decode_tok_s"], fn, "vllm-fp8-baseline", ver, date, "")

# =============================================================================
# 4b. gemma4-12b-vllm-sm120.json — vLLM 0.25.1 SERVED sm_120 (RTX PRO 6000
#     Blackwell) baseline, campaign B1: Gemma-4-12B bf16/fp8/fp8kv to 128k.
#     Same served-harness shape as do_vllm_fp8 (config.results rows), but its
#     own campaign label + sm_120/TRITON_ATTN provenance.
# =============================================================================
def do_gemma12b_sm120_b1():
    fn = "gemma4-12b-vllm-sm120.json"
    d = load(fn)
    date = d["date"]
    ver = d["vllm_version"]
    model = d["model"]
    prov = "RTX PRO 6000 Blackwell (sm_120); TRITON_ATTN (forced, Gemma4 heterogeneous head dims); cudagraphs ON; not bit-exact."
    for cfg in d["configs"]:
        prec = cfg["config"]  # bf16 | fp8 | fp8kv
        for r in cfg["results"]:
            ctx = r.get("ctx", r.get("input_len"))
            warm = " ctx=1024 TTFT warm-up-contaminated (first-shape torch.compile/cudagraph capture); 4k+ clean." \
                   if ctx == 1024 else ""
            add(model, "vllm", prec, "prefill", 1, ctx, "ttft_ms",
                r["ttft_ms"], fn, "B1-sm120", ver, date, (prov + warm).strip())
            add(model, "vllm", prec, "prefill", 1, ctx, "prefill_tok_s",
                r["prefill_tok_s"], fn, "B1-sm120", ver, date,
                "DERIVED input_len/TTFT (TTFT includes 1st decode token); under-reports true prefill tok/s. " + prov)
            add(model, "vllm", prec, "decode", 1, ctx, "tpot_ms",
                r["tpot_ms"], fn, "B1-sm120", ver, date,
                "decode metric == itl_ms; TPOT unaffected by 1k warm-up. " + prov)
            add(model, "vllm", prec, "decode", 1, ctx, "itl_ms",
                r["itl_ms"], fn, "B1-sm120", ver, date, prov)
            add(model, "vllm", prec, "decode", 1, ctx, "decode_tok_s",
                r["decode_tok_s"], fn, "B1-sm120", ver, date, prov)

# =============================================================================
# 4c. gemma4-31b-vllm-sm120.json — same B1 sm_120 served baseline for the dense
#     Gemma-4-31B (identical harness/schema to 4b; bf16 fits 128k at batch 1
#     only — 1.45x concurrency headroom; fp8kv decode regresses past fp8 at
#     128k, see source notes).
# =============================================================================
def do_gemma31b_sm120_b1():
    fn = "gemma4-31b-vllm-sm120.json"
    d = load(fn)
    date = d["date"]
    ver = d["vllm_version"]
    model = d["model"]
    prov = "RTX PRO 6000 Blackwell (sm_120); TRITON_ATTN (forced, Gemma4 heterogeneous head dims); cudagraphs ON; not bit-exact."
    for cfg in d["configs"]:
        prec = cfg["config"]  # bf16 | fp8 | fp8kv
        for r in cfg["results"]:
            ctx = r.get("ctx", r.get("input_len"))
            warm = " ctx=1024 TTFT warm-up-contaminated (first-shape torch.compile/cudagraph capture); 4k+ clean." \
                   if ctx == 1024 else ""
            anom = " fp8kv 128k decode anomaly: TPOT regresses past plain fp8 (steady-state, no preemption; see source notes)." \
                   if prec == "fp8kv" and ctx == 131072 else ""
            add(model, "vllm", prec, "prefill", 1, ctx, "ttft_ms",
                r["ttft_ms"], fn, "B1-sm120", ver, date, (prov + warm).strip())
            add(model, "vllm", prec, "prefill", 1, ctx, "prefill_tok_s",
                r["prefill_tok_s"], fn, "B1-sm120", ver, date,
                "DERIVED input_len/TTFT (TTFT includes 1st decode token); under-reports true prefill tok/s. " + prov)
            add(model, "vllm", prec, "decode", 1, ctx, "tpot_ms",
                r["tpot_ms"], fn, "B1-sm120", ver, date,
                ("decode metric == itl_ms; TPOT unaffected by 1k warm-up. " + prov + anom).strip())
            add(model, "vllm", prec, "decode", 1, ctx, "itl_ms",
                r["itl_ms"], fn, "B1-sm120", ver, date, prov)
            add(model, "vllm", prec, "decode", 1, ctx, "decode_tok_s",
                r["decode_tok_s"], fn, "B1-sm120", ver, date, prov)

# =============================================================================
# 4b. gemma4-12b-plow-sm120-decode.json — plow Gemma-4-12B bf16 DECODE on sm_120
#     (P1-sm120-decode; the like-for-like counterpart to do_gemma12b_sm120_b1)
# =============================================================================
def do_gemma12b_plow_sm120_decode():
    fn = "gemma4-12b-plow-sm120-decode.json"
    d = load(fn)
    date = d["date"]
    ver = d["version"]
    model = d["model"]
    prec = d["precision"]  # bf16
    prov = ("RTX PRO 6000 Blackwell (sm_120); Gemma interp -DPLOW_NV_GEMMA=1 "
            "-DPLOW_NV_FA_GF=2 (hd 256/512, GF=2), global-queue scheduler; batch 1, "
            "single seq; 112 timed steps after 16 warmup; not bit-exact but token-parity "
            "gated vs HF greedy (Phase-0 32cc434). Standalone decode harness, no cudagraph.")
    for r in d["results"]:
        ctx = r.get("ctx", r.get("input_len"))
        add(model, "plow", prec, "decode", 1, ctx, "tpot_ms",
            r["tpot_ms"], fn, "P1-sm120-decode", ver, date, prov)
        # itl == tpot at concurrency 1 (mirrors the vLLM B1 convention).
        add(model, "plow", prec, "decode", 1, ctx, "itl_ms",
            r["tpot_ms"], fn, "P1-sm120-decode", ver, date,
            "itl == tpot_ms at concurrency 1. " + prov)
        add(model, "plow", prec, "decode", 1, ctx, "decode_tok_s",
            r["decode_tok_s"], fn, "P1-sm120-decode", ver, date, prov)

# =============================================================================
# 4b'. gemma4-26b-plow-sm120.json — plow Gemma-4-26B-A4B (MoE) bf16 DECODE on sm_120
#      (P3-26b; the like-for-like counterpart to do_gemma26b_sm120_b1). campaigns[].
# =============================================================================
def do_gemma26b_plow_sm120_decode():
    fn = "gemma4-26b-plow-sm120.json"
    d = load(fn)
    # Schema history: originally {model, version, precision, campaigns[]}; the P9
    # batched-MoE campaign (2026-07-20) rewrote it as a FLAT LIST of records
    # {model, config, batch, ctx, decode_tpot_ms, prefill_ms, campaign?, ...}.
    # Handle both; trace/verdict records (no decode_tpot_ms) live in the .md only.
    if isinstance(d, dict):
        model, ver, prec = d["model"], d["version"], d["precision"]
        for camp in d["campaigns"]:
            cname, date = camp["campaign"], camp["date"]
            for r in camp["results"]:
                ctx = r.get("ctx", r.get("input_len"))
                add(model, "plow", prec, "decode", 1, ctx, "tpot_ms",
                    r.get("tpot_ms"), fn, cname, ver, date,
                    "26B MoE decode, batch 1 (dict-era schema).")
        return
    for r in d:
        if not isinstance(r, dict) or "ctx" not in r:
            continue
        model = r.get("model", "gemma-4-26B-A4B-it")
        prec = r.get("config", "bf16")
        cname = r.get("campaign", "P9-26b-batch")
        date = r.get("date", "2026-07-20")
        ver = "plow @ " + r.get("commit", "(uncommitted)")
        b = r.get("batch", 1)
        prov = (f"batch {b}; {r.get('binary','')} {r.get('pkt_env','')}".strip())
        if r.get("decode_tpot_ms") is not None:
            metric = "tpot_ms" if b == 1 else f"tpot_ms_served_b{b}"
            add(model, "plow", prec, "decode", 1, r["ctx"], metric,
                r["decode_tpot_ms"], fn, cname, ver, date, prov)
        if r.get("prefill_ms") is not None and b == 1:
            add(model, "plow", prec, "prefill", 1, r["ctx"], "prefill_ms",
                r["prefill_ms"], fn, cname, ver, date, prov)

# =============================================================================
# 4c. gemma4-12b-plow-prefill-sm120.json — plow Gemma-4-12B bf16 PREFILL on sm_120
#     (chunked prefill through the persistent interpreter; multiple campaigns kept,
#      newest last). ctx 4k..128k, prefill_ms + prefill_tok_s.
# =============================================================================
def do_gemma12b_plow_prefill():
    fn = "gemma4-12b-plow-prefill-sm120.json"
    d = load(fn)
    model = d["model"]
    for camp in d["campaigns"]:
        cname = camp["campaign"]
        date = camp["date"]
        note = (camp.get("change", "") + " " + camp.get("notes", "")).strip() or \
               "plow Gemma-4-12B bf16 chunked prefill on sm_120 (RTX PRO 6000)."
        for r in camp["results"]:
            ctx = r["ctx"]
            # Per-campaign key shapes: bf16 campaigns carry prefill_ms; the fp8
            # campaigns carry fp8_prefill_ms (T6, w8a16 dequant) or
            # w8a8_prefill_ms (T8, true fp8 mma). Transcribe under the right
            # precision; never both from one row.
            if "prefill_ms" in r:
                add(model, "plow", "bf16", "prefill", 1, ctx, "prefill_ms",
                    r["prefill_ms"], fn, cname, "(uncommitted)", date, note)
                add(model, "plow", "bf16", "prefill", 1, ctx, "prefill_tok_s",
                    r.get("prefill_tok_s"), fn, cname, "(uncommitted)", date, note)
            if "fp8_prefill_ms" in r:
                add(model, "plow", "fp8", "prefill", 1, ctx, "prefill_ms",
                    r["fp8_prefill_ms"], fn, cname, "(uncommitted)", date,
                    (note + " w8a16 dequant-in-smem (measured-negative vs bf16).").strip())
                add(model, "plow", "fp8", "prefill", 1, ctx, "prefill_tok_s",
                    r.get("fp8_prefill_tok_s"), fn, cname, "(uncommitted)", date, note)
            if "w8a8_prefill_ms" in r:
                add(model, "plow", "fp8", "prefill", 1, ctx, "prefill_ms",
                    r["w8a8_prefill_ms"], fn, cname, "(uncommitted)", date,
                    (note + " true w8a8 mma.sync.m16n8k32; bf16 control in bf16_prefill_ms.").strip())
                add(model, "plow", "fp8", "prefill", 1, ctx, "prefill_tok_s",
                    r.get("w8a8_prefill_tok_s"), fn, cname, "(uncommitted)", date, note)


# =============================================================================
# 4c'. gemma4-31b-plow-sm120.json — plow Gemma-4-31B DECODE (bf16+fp8) + PREFILL
#      (bf16) on sm_120 (P2-31b-*; the like-for-like counterpart to
#      do_gemma31b_sm120_b1). Kernels = HEAD 7fe44b8; pkts carry an uncommitted
#      emitter argmax fix required for 31B prefill (see source md).
# =============================================================================
def do_gemma31b_plow_sm120():
    fn = "gemma4-31b-plow-sm120.json"
    d = load(fn)
    model = d["model"]
    date = d["date"]
    ver = d["git_commit_measured"]
    prov = ("RTX PRO 6000 Blackwell (sm_120); Gemma interp -DPLOW_NV_GEMMA=1 "
            "-DPLOW_NV_FA_GF=2 (full-layer GQA-8 fused GF=2), global-queue scheduler; "
            "batch 1, single seq; 112 timed steps after 16 warmup; standalone harness, "
            "no cudagraph; HF-greedy parity PASS (p2 48/48, p3 31/31 to EOS, p1 "
            "reconverging bf16 near-tie). PLOW_UNISEG=1 pkt; emitter argmax fix "
            "(uncommitted, crates/plowc) required for 31B prefill.")
    for camp in d["campaigns"]:
        cname = camp["campaign"]
        prec = camp["precision"]
        for r in camp["results"]:
            if "tpot_ms" not in r and "prefill_ms" not in r:
                continue  # ablation grid rows (ns/UN columns) live in the .md only
            if "tpot_ms" in r:  # decode campaign (bf16 or fp8)
                ctx = r.get("ctx", r.get("input_len"))
                add(model, "plow", prec, "decode", 1, ctx, "tpot_ms",
                    r["tpot_ms"], fn, cname, ver, date, prov)
                add(model, "plow", prec, "decode", 1, ctx, "itl_ms",
                    r["tpot_ms"], fn, cname, ver, date,
                    "itl == tpot_ms at concurrency 1. " + prov)
                add(model, "plow", prec, "decode", 1, ctx, "decode_tok_s",
                    r.get("decode_tok_s"), fn, cname, ver, date, prov)
            else:  # prefill campaign
                ctx = r.get("ctx", r.get("input_len"))
                add(model, "plow", prec, "prefill", 1, ctx, "prefill_ms",
                    r["prefill_ms"], fn, cname, ver, date, prov)
                add(model, "plow", prec, "prefill", 1, ctx, "prefill_tok_s",
                    r.get("prefill_tok_s"), fn, cname, ver, date, prov)

# =============================================================================
# 4d. gemma4-26b-a4b-vllm-sm120.json — same B1 sm_120 served baseline for the
#     MoE Gemma-4-26B-A4B (128 experts top-8, ~4B active). Identical harness/
#     schema to 4b/4c. MoE paths: bf16 = FlashInfer CUTLASS (autotuned at
#     warmup); fp8/fp8kv = TRITON fp8 MoE on a DEFAULT (untuned) config for
#     this GPU — vLLM's own sub-optimal-performance warning; see source notes.
# =============================================================================
def do_gemma26b_sm120_b1():
    fn = "gemma4-26b-a4b-vllm-sm120.json"
    d = load(fn)
    date = d["date"]
    ver = d["vllm_version"]
    model = d["model"]
    prov = "RTX PRO 6000 Blackwell (sm_120); TRITON_ATTN (forced, Gemma4 heterogeneous head dims); cudagraphs ON; not bit-exact."
    moe = {
        "bf16": " MoE: FlashInfer CUTLASS unquantized (autotuned).",
        "fp8": " MoE: TRITON fp8, DEFAULT untuned config for this GPU (vLLM warns sub-optimal).",
        "fp8kv": " MoE: TRITON fp8, DEFAULT untuned config for this GPU (vLLM warns sub-optimal).",
    }
    for cfg in d["configs"]:
        prec = cfg["config"]  # bf16 | fp8 | fp8kv
        for r in cfg["results"]:
            ctx = r.get("ctx", r.get("input_len"))
            warm = " ctx=1024 TTFT warm-up-contaminated (first-shape torch.compile/cudagraph capture); 4k+ clean." \
                   if ctx == 1024 else ""
            p = prov + moe[prec]
            add(model, "vllm", prec, "prefill", 1, ctx, "ttft_ms",
                r["ttft_ms"], fn, "B1-sm120", ver, date, (p + warm).strip())
            add(model, "vllm", prec, "prefill", 1, ctx, "prefill_tok_s",
                r["prefill_tok_s"], fn, "B1-sm120", ver, date,
                "DERIVED input_len/TTFT (TTFT includes 1st decode token); under-reports true prefill tok/s. " + p)
            add(model, "vllm", prec, "decode", 1, ctx, "tpot_ms",
                r["tpot_ms"], fn, "B1-sm120", ver, date,
                "decode metric == itl_ms; TPOT unaffected by 1k warm-up. " + p)
            add(model, "vllm", prec, "decode", 1, ctx, "itl_ms",
                r["itl_ms"], fn, "B1-sm120", ver, date, p)
            add(model, "vllm", prec, "decode", 1, ctx, "decode_tok_s",
                r["decode_tok_s"], fn, "B1-sm120", ver, date, p)

# =============================================================================
# 4d. gemma4-12b-plowrt-served-ttft.json — plow Gemma-4-12B bf16 SERVED TTFT/TPOT
#     (plowrt serve OpenAI endpoint; in-serve prefill). ctx 1k..64k.
# =============================================================================
def do_gemma12b_plowrt_ttft():
    fn = "gemma4-12b-plowrt-served-ttft.json"
    d = load(fn)
    model = d["model"]
    for camp in d["campaigns"]:
        cname = camp["campaign"]
        date = camp["date"]
        note = camp.get("notes", "")
        for r in camp.get("results", []):
            ctx = r["ctx"]
            add(model, "plow", "bf16", "prefill", 1, ctx, "ttft_ms",
                r["ttft_ms"], fn, cname, "plowrt-serve", date, note)
            add(model, "plow", "bf16", "decode", 1, ctx, "tpot_ms",
                r["tpot_ms"], fn, cname, "plowrt-serve", date, note)
        # Batch campaigns (S4-served-batch): B concurrent served sequences.
        # The flat schema has no batch column, so the metric name carries it
        # (same convention as decode_kernel_<name>_ms). Only full-occupancy
        # rows (users == blob batch) are emitted.
        for r in camp.get("batch_results", []):
            if r["users"] != r["blob_batch"]:
                continue
            b = r["blob_batch"]
            add(model, "plow", "bf16", "decode", 1, r["ctx"],
                f"tpot_ms_served_b{b}", r["per_user_tpot_ms"],
                fn, cname, "plowrt-serve", date, note)
            add(model, "plow", "bf16", "decode", 1, r["ctx"],
                f"decode_tok_s_served_agg_b{b}", r["aggregate_tok_s"],
                fn, cname, "plowrt-serve", date, note)

# =============================================================================
# 4e. gemma4-{31b,26b}-plowrt-served.json — S6: 31B + 26B served TTFT/TPOT rows
#     (same campaigns[].results[] shape as the 12B served file). Guarded: the
#     files land with the S6 campaign; absent files are skipped.
# =============================================================================
def do_plowrt_served(fn):
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        return
    d = load(fn)
    model = d["model"]
    for camp in d["campaigns"]:
        cname, date, note = camp["campaign"], camp["date"], camp.get("notes", "")
        for r in camp.get("results", []):
            add(model, "plow", "bf16", "prefill", 1, r["ctx"], "ttft_ms",
                r["ttft_ms"], fn, cname, "plowrt-serve", date, note)
            add(model, "plow", "bf16", "decode", 1, r["ctx"], "tpot_ms",
                r["tpot_ms"], fn, cname, "plowrt-serve", date, note)

# =============================================================================
# 5. gemma4-vllm-perf.json — vLLM 0.25.1 in-proc, 12B+31B, + plow reference
# =============================================================================
def do_gemma12b_fp8_longctx():
    # fp8-decode-gemma-sm120.json — plow Gemma-4-12B fp8 (w8a16) weight-only DECODE
    # on sm_120 (RTX PRO 6000). Only the plow-fp8 tpot_ms rows are emitted here; the
    # vLLM-fp8 and plow-bf16 comparison columns carried in the JSON are reference
    # baselines sourced elsewhere and are NOT re-transcribed (faithfulness rule).
    fn = "fp8-decode-gemma-sm120.json"
    d = load(fn)
    model = d["model"]
    for camp in d["campaigns"]:
        cname = camp["campaign"]
        date = camp["date"]
        note = camp.get("note", "") or camp.get("notes", "")
        for r in camp["results"]:
            add(model, "plow", "fp8", "decode", 1, r["ctx"], "tpot_ms",
                r["plow_fp8_tpot_ms"], fn, cname, "(uncommitted)", date, note)

def do_gemma_vllm_perf():
    fn = "gemma4-vllm-perf.json"
    d = load(fn)
    date = d["generated"]
    ver = d["common_setup"]["engine"]
    for mname, m in d["models"].items():
        if not m.get("results"):
            continue
        model = mname
        for r in m["results"]:
            ctx = r["ctx"]
            add(model, "vllm", "bf16", "decode", 1, ctx, "decode_tok_s",
                r["decode_tok_s"], fn, "vllm-gemma4-bench", ver, date,
                "in-process vllm.LLM (not served); text-only stripped checkpoint.")
            add(model, "vllm", "bf16", "decode", 1, ctx, "decode_ms_per_token",
                r["decode_ms_per_token"], fn, "vllm-gemma4-bench", ver, date, "")
            add(model, "vllm", "bf16", "prefill", 1, ctx, "prefill_ms",
                r["prefill_ms"], fn, "vllm-gemma4-bench", ver, date,
                "prefill_ms is TTFT (prefill + 1 decode token).")
            add(model, "vllm", "bf16", "prefill", 1, ctx, "prefill_tok_s",
                r["prefill_tok_s"], fn, "vllm-gemma4-bench", ver, date, "")
    # embedded plow reference (gemma4-mi350x-sprint.md section 17)
    pr = d["plow_reference"]
    pnote = "plow reference from gemma4-mi350x-sprint.md §17; hand-written CDNA4 kernels."
    for r in pr["results"]:
        add("gemma-4-31B", "plow", "bf16", "decode", 1, r["ctx"], "decode_tok_s",
            r["decode_tok_s"], fn, "plow-sprint-ref", "plow sprint main", date, pnote)
        add("gemma-4-31B", "plow", "bf16", "decode", 1, r["ctx"], "decode_ms_per_token",
            r["decode_ms_per_token"], fn, "plow-sprint-ref", "plow sprint main", date, pnote)
    pf = pr["prefill"]
    add("gemma-4-31B", "plow", "bf16", "prefill", 1, pf["tokens"], "prefill_ms",
        pf["time_ms"], fn, "plow-sprint-ref", "plow sprint main", date,
        pnote + " pure prefill.")
    add("gemma-4-31B", "plow", "bf16", "prefill", 1, pf["tokens"], "prefill_tok_s",
        pf["tok_s"], fn, "plow-sprint-ref", "plow sprint main", date, pnote)

# =============================================================================
# 6. {llama-3.1-8b,qwen3-4b}-vllm-docker.json — vLLM 0.11.2 docker served, bf16
# =============================================================================
def do_vllm_docker(fn):
    d = load(fn)
    date = d["date"]
    ver = d["vllm_version"]
    model = d["model"]
    for r in d["results"]:
        ctx = r["ctx"]
        add(model, "vllm", "bf16", "prefill", 1, ctx, "ttft_ms",
            r["ttft_ms"], fn, "vllm-docker-baseline", ver, date,
            "docker-served (rocm/vllm:latest); TTFT = prefill + 1st token.")
        add(model, "vllm", "bf16", "prefill", 1, ctx, "prefill_tok_s",
            r["prefill_tok_s"], fn, "vllm-docker-baseline", ver, date, "")
        add(model, "vllm", "bf16", "decode", 1, ctx, "tpot_ms",
            r["tpot_ms"], fn, "vllm-docker-baseline", ver, date, "TPOT == ITL here.")
        add(model, "vllm", "bf16", "decode", 1, ctx, "decode_tok_s",
            r["decode_tok_s"], fn, "vllm-docker-baseline", ver, date, "")

# =============================================================================
# 7. {llama-3.1-8b,qwen3-1.7b,qwen3-4b}-vllm-perf.json — vLLM 0.25.1 in-proc, bf16
# =============================================================================
def do_vllm_inproc(fn):
    d = load(fn)
    ver = "vLLM " + d["vllm_version"] + " (in-process)"
    model = d["model"]
    impl_note = ('source file\'s "impl" field reads "native Gemma4ForCausalLM" — a '
                 "copy-paste artifact; the model is actually " + canon_model(model) + ".")
    for r in d["results"]:
        ctx = r["ctx"]
        add(model, "vllm", "bf16", "prefill", 1, ctx, "prefill_ms",
            r["prefill_ms"], fn, "vllm-inproc-perf", ver, "2026-07-15",
            "pure prefill timer (in-process). " + impl_note)
        add(model, "vllm", "bf16", "prefill", 1, ctx, "prefill_tok_s",
            r["prefill_tok_s"], fn, "vllm-inproc-perf", ver, "2026-07-15", "")
        add(model, "vllm", "bf16", "decode", 1, ctx, "decode_ms_per_token",
            r["decode_ms_per_token"], fn, "vllm-inproc-perf", ver, "2026-07-15", "")
        add(model, "vllm", "bf16", "decode", 1, ctx, "decode_tok_s",
            r["decode_tok_s"], fn, "vllm-inproc-perf", ver, "2026-07-15", "")

# =============================================================================
# 8. vllm_decode_breakdown.json — per-kernel decode-step profile, Gemma-4-31B
# =============================================================================
def do_decode_breakdown():
    fn = "vllm_decode_breakdown.json"
    d = load(fn)
    date = "2026-07-17"
    ver = "0.25.1+rocm723"
    ctx = 1024
    note = "torch-profiler per-decode-step GPU-kernel time, batch1 ctx~1k, MI350X."
    for tpkey, tp in (("tp1", 1), ("tp4", 4)):
        blk = d[tpkey]
        if "measured_tpot_ms" in blk:
            add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "tpot_ms",
                blk["measured_tpot_ms"], fn, "vllm-decode-breakdown", ver, date, note)
        add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "decode_step_mean_ms",
            blk.get("profiled_mean_step_ms"), fn, "vllm-decode-breakdown", ver, date,
            "profiler mean step time. " + note)
        add("gemma-4-31B", "vllm", "bf16", "prefill", tp, ctx, "prefill_ms",
            blk.get("prefill_1024_ms"), fn, "vllm-decode-breakdown", ver, date, note)
        if "gemv_effective_TBps" in blk:
            add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx, "gemv_TBps",
                blk["gemv_effective_TBps"], fn, "vllm-decode-breakdown", ver, date,
                "effective GEMV bandwidth. " + note)
        for kname, kval in blk["per_step_ms"].items():
            add("gemma-4-31B", "vllm", "bf16", "decode", tp, ctx,
                "decode_kernel_" + kname + "_ms", kval, fn, "vllm-decode-breakdown",
                ver, date, "per-kernel share of the decode step. " + note)
    t8 = d["tp8_not_profiled"]
    add("gemma-4-31B", "vllm", "bf16", "decode", 8, ctx, "tpot_ms",
        t8["documented_tpot_ms"], fn, "vllm-decode-breakdown", ver, date,
        "documented value, not profiled (" + t8["reason"] + ").")
    # embedded plow reference
    pr = d["plow_reference"]
    pnote = "plow reference: " + pr["source"]
    pver = "branch tp"
    add("gemma-4-31B", "plow", "bf16", "decode", 1, ctx, "tpot_ms",
        pr["tp1_1k_ms"], fn, "plow-tp-design-ref", pver, date, pnote)
    add("gemma-4-31B", "plow", "bf16", "decode", 4, ctx, "tpot_ms",
        pr["tp4_best_1k_ms"], fn, "plow-tp-design-ref", pver, date, pnote + " (best xr-tuned)")
    add("gemma-4-31B", "plow", "bf16", "decode", 8, ctx, "tpot_ms",
        pr["tp8_best_1k_ms"], fn, "plow-tp-design-ref", pver, date, pnote + " (best)")
    add("gemma-4-31B", "plow", "bf16", "decode", 1, ctx, "decode_kernel_gemv_ms",
        pr["decode_gemv_ms"], fn, "plow-tp-design-ref", pver, date, pnote)
    add("gemma-4-31B", "plow", "bf16", "decode", 1, ctx, "gemv_TBps",
        pr["gemv_TBps"], fn, "plow-tp-design-ref", pver, date, pnote)

# =============================================================================
# 9-11. MARKDOWN-ONLY tables (transcribed as literals — data in NO json)
# =============================================================================
# plow TP prefill sweeps. Each cell: (prefill_ms, prefill_tok_s). Grid ctx x tp.
_PF_CTX = [8192, 32768, 65536, 131072]
_PF_TP = [1, 2, 4, 8]

def _emit_prefill_grid(fn, campaign, version, date, note, grid):
    # grid[ci][ti] = (ms, toks)
    for ci, ctx in enumerate(_PF_CTX):
        for ti, tp in enumerate(_PF_TP):
            ms, toks = grid[ci][ti]
            add("gemma-4-31B", "plow", "bf16", "prefill", tp, ctx, "prefill_ms",
                ms, fn, campaign, version, date, note)
            add("gemma-4-31B", "plow", "bf16", "prefill", tp, ctx, "prefill_tok_s",
                toks, fn, campaign, version, date, note)

def do_tp_prefill_oneshot():
    fn = "gemma4-31b-tp-prefill.md"
    grid = [
        [(1174.5, 6975), (810.7, 10104), (566.7, 14455), (547.7, 14958)],   # 8k
        [(7586.2, 4319), (4679.4, 7003), (2998.3, 10929), (2570.7, 12746)], # 32k
        [(22773.4, 2878), (13174.8, 4974), (7881.0, 8316), (6099.2, 10745)],# 64k
        [(75850.6, 1728), (42019.8, 3119), (23321.4, 5620), (16010.7, 8186)],# 128k
    ]
    _emit_prefill_grid(fn, "tp-prefill-oneshot", "branch tp-prefill", "2026-07-17",
        "one-shot [T,hidden] XReduce TP prefill; bit-exact TP1=2=4=8 "
        "(tok0 identical); TP1 reproduces shipped single-GPU baseline.", grid)

def do_tp_prefill_twoshot():
    fn = "gemma4-31b-tp-prefill-2shot.md"
    grid = [
        [(1166.1, 7025), (827.4, 9901), (533.0, 15369), (415.5, 19716)],    # 8k
        [(7540.4, 4346), (4766.1, 6875), (2845.2, 11517), (2030.5, 16138)], # 32k
        [(22654.9, 2893), (13409.9, 4887), (7605.2, 8617), (5017.3, 13062)],# 64k
        [(76971.1, 1703), (42161.6, 3109), (22959.5, 5709), (13831.1, 9477)],# 128k
    ]
    _emit_prefill_grid(fn, "tp-prefill-twoshot", "branch tp-prefill-2shot", "2026-07-17",
        "two-shot (reduce-scatter + all-gather) TP prefill; bit-exact, tok0 "
        "identical to one-shot; 14-24% faster TP8 than one-shot (N>=4 only).", grid)

def do_plow_vs_vllm_baseline():
    fn = "plow-vs-vllm-baseline.md"
    date = "2026-07-15"
    cmn = ("plow prefill is PURE prefill (not TTFT). 'campaign' = verified wins on "
           "branches qwen-prefill-perf/qwen-decode-perf/llama-decode-perf, NOT merged to main.")
    # (model, ctx, plow_main, plow_campaign)  prefill_ms
    prefill = [
        ("qwen3-4B", 4096, 222, 148), ("qwen3-4B", 8192, 651, 356), ("qwen3-4B", 16384, 2178, None),
        ("llama-3.1-8B", 4096, 237, 173), ("llama-3.1-8B", 8192, 649, 393), ("llama-3.1-8B", 16384, 2087, None),
    ]
    for model, ctx, main, camp in prefill:
        add(model, "plow", "bf16", "prefill", 1, ctx, "prefill_ms", main, fn,
            "plow-main-baseline", "plow main (a109dcf)", date, cmn)
        add(model, "plow", "bf16", "prefill", 1, ctx, "prefill_ms", camp, fn,
            "plow-campaign", "plow campaign (unmerged branches)", date, cmn)
    # decode ms/token
    decode = [
        ("qwen3-4B", 4096, 4.8, 4.7), ("qwen3-4B", 8192, 5.2, None),
        ("llama-3.1-8B", 4096, 5.5, 5.2), ("llama-3.1-8B", 8192, 5.9, 5.6),
        ("llama-3.1-8B", 16384, 6.3, 6.2),
    ]
    for model, ctx, main, camp in decode:
        add(model, "plow", "bf16", "decode", 1, ctx, "tpot_ms", main, fn,
            "plow-main-baseline", "plow main (a109dcf)", date, cmn)
        add(model, "plow", "bf16", "decode", 1, ctx, "tpot_ms", camp, fn,
            "plow-campaign", "plow campaign (unmerged branches)", date, cmn)

def do_vllm_tp_baseline_plow():
    # plow bit-exact xr-tuned TP decode figures (from plans/tp-design.md §14), Gemma-4-31B.
    # vLLM columns in this md are already captured from gemma4-31b-vllm-tp.json.
    fn = "vllm-tp-baseline.md"
    date = "2026-07-16"
    ver = "branch tp (plans/tp-design.md §14)"
    sup = ("bit-exact xr-tuned TP decode from plans/tp-design.md §14; SUPERSEDED by "
           "decode-only-sweep.json for TP1/4/8 (fresh unified build measured faster); "
           "TP2 is unique to this file.")
    # (tp, ctx, tpot_ms, extra_note)
    cells = [
        (1, 1024, 19.1, "TP1 single-GPU reference."),
        (1, 65536, 22.9, "TP1 single-GPU reference."),
        (2, 1024, 15.43, ""),
        (2, 65536, 18.08, "GQ-default value (no xr-tuned 64k point recorded)."),
        (4, 1024, 13.75, "best xr32 config."),
        (4, 65536, 14.73, "only cell where plow leads vLLM here."),
        (8, 1024, 15.81, "TP8 regresses vs TP4 (all-reduce crosses NUMA boundary)."),
        (8, 65536, 16.47, ""),
    ]
    for tp, ctx, val, extra in cells:
        add("gemma-4-31B", "plow", "bf16", "decode", tp, ctx, "tpot_ms", val, fn,
            "tp-design-xrtuned", ver, date, (sup + " " + extra).strip())

# --- run all -----------------------------------------------------------------
# =============================================================================
# glm52-vllm-decode.json — vLLM 0.25.1 GLM-5.2-FP8 (GlmMoeDsa) decode sweep, TP4+TP8
# =============================================================================
def do_glm52_vllm():
    fn = "glm52-vllm-decode.json"
    d = load(fn)
    ver = d["vllm_version"]
    date = d["date"]
    caveat = ("vLLM 0.25.1 GlmMoeDsa (DeepSeek-V3.2-DSA class; AITER required); block-fp8 "
              "auto-detected; UNTUNED aiter kernels (no tuned config for GLM shapes) => "
              "out-of-box, not tuned ceiling; decode near ctx-independent (MLA latent KV + DSA top-2048).")
    for r in d["results"]:
        tp, ctx = r["tp"], r["ctx"]
        note = (r.get("notes", "") + "; " + caveat).strip("; ")
        add("GLM-5.2-FP8", "vllm", "fp8", "decode", tp, ctx, "tpot_ms",
            r["tpot_ms"], fn, "glm52-vllm", ver, date, note)
        add("GLM-5.2-FP8", "vllm", "fp8", "decode", tp, ctx, "decode_tok_s",
            r["decode_tok_s"], fn, "glm52-vllm", ver, date, note)
        add("GLM-5.2-FP8", "vllm", "fp8", "prefill", tp, ctx, "ttft_ms",
            r["ttft_ms"], fn, "glm52-vllm", ver, date, note)
        add("GLM-5.2-FP8", "vllm", "fp8", "prefill", tp, ctx, "prefill_tok_s",
            r["prefill_tok_s"], fn, "glm52-vllm", ver, date, note)

# =============================================================================
# glm52-plow-decode.json — plow full 78-layer GLM-5.2-FP8 TP4 decode (honest baseline)
# =============================================================================
def do_glm52_plow():
    fn = "glm52-plow-decode.json"
    d = load(fn)
    ver = d["version"]
    date = d["date"]
    caveat = ("plow full 78-layer GLM-5.2 TP4, block-fp8, UNFUSED/TP-shard baseline; op-overhead + M=1 "
              "expert-GEMV work bound (~7-10x vLLM's ~20ms); grows with ctx (MLA latent-cache attention). "
              "Honest baseline before EP/fusion. Fusion (grouped experts + A/G/B1) separately reached "
              "~129ms extrapolated on the subset.")
    for r in d["results"]:
        note = (r.get("notes", "") + "; " + caveat).strip("; ")
        add("GLM-5.2-FP8", "plow", "fp8", "decode", r["tp"], r["ctx"], "tpot_ms",
            r["tpot_ms"], fn, "glm52-plow", ver, date, note)

# =============================================================================
# glm52-plow-decode-tuned.json — plow tuned GLM-5.2-FP8 decode, full TP/EP x ctx sweep
# =============================================================================
def do_glm52_plow_tuned():
    fn = "glm52-plow-decode-tuned.json"
    d = load(fn)
    ver = d["version"]
    date = d["date"]
    caveat = ("plow TUNED (router-split + wave-parallel topk + co-resident experts "
              "GLM_MOE_CORESIDENT=1 + grouped block-fp8 + K-adaptive GEMV UN + per-rank "
              "ctx-scaled nsplit + fused MlaMergeFold + block-fp8 buffer-waterfall + per-pkt "
              "MLA GF); coherent decode byte-identical to pre-tuning baseline. TP beats EP at "
              "every ctx (co-residency collapses all 8 experts/rank under TP; EP replicates MLA). "
              "TP8 is ship config. Turned GLM decode 146->228 (pre-tune TP4) into 40->51 (tuned "
              "TP8), 3.6x@1k/4.5x@128k; still ~2.1-2.5x above vLLM (M=1 replicated-MLA/expert-GEMV "
              "floor: VALU fdot2 vs AITER MFMA). DSA-gather indexer (long-ctx) not yet in these numbers.")
    for r in d["results"]:
        mode = r["mode"]                       # "tp" or "ep"
        campaign = "glm52-plow-tuned-" + mode  # distinguishes TP4 from EP4 at same degree
        note = (mode.upper() + str(r["tp"]) + "; " + (r.get("notes", "") or "") +
                "; " + caveat).strip("; ")
        add("GLM-5.2-FP8", "plow", "fp8", "decode", r["tp"], r["ctx"], "tpot_ms",
            r["tpot_ms"], fn, campaign, ver, date, note)

# =============================================================================
# glm52-plow-decode-256k.json — plow full-78-layer DSA gather TP4 sweep to 256k
# =============================================================================
def do_glm52_plow_256k():
    fn = "glm52-plow-decode-256k.json"
    d = load(fn)
    ver = d["version"]
    date = d["date"]
    caveat = ("plow FULL 78-layer real-weight TP4 (consolidated glm-dsa, fast-indexer interp path); "
              "first true end-to-end run. DENSE grows with ctx (full-cache MLA flash); GATHER near-"
              "constant ~51-54ms (DSA top-2048), CROSSOVER ~64k, 1.41x vs dense @256k. Gather bit-"
              "identical to dense (kv_len<=top_k). Measured on a 256k-max pkt (fixed buffers inflate "
              "low-ctx vs a per-ctx pkt). Still ~2.2-2.5x above vLLM TP4 (M=1 floor); gather CAPS the "
              "long-ctx growth (dense 75.8ms vs gather 53.8ms @256k) rather than reaching vLLM parity. "
              "The ~52ms floor is MoE/projection/indexer-dominated, not flash — the earlier ~17ms flash "
              "projection over-credited the flash fraction.")
    for r in d["results"]:
        mode = r["mode"]                          # "dense" or "gather"
        campaign = "glm52-plow-256k-" + mode
        note = (mode + "; " + (r.get("notes", "") or "") + "; " + caveat).strip("; ")
        add("GLM-5.2-FP8", "plow", "fp8", "decode", r["tp"], r["ctx"], "tpot_ms",
            r["tpot_ms"], fn, campaign, ver, date, note)

def do_b2_concurrency():
    # B2-ib final-numbers v2 — multi-user concurrency/capacity head-to-head
    # through the pinned huggingface/inference-benchmarker binary (rev in the
    # source jsons), whole Gemma-4 family + the MM1 co-resident scenario.
    # Only the fixed-VU (ConstantVUs) points are transcribed here; the full
    # sweep-mode rate grid and all percentiles live in the source jsons. The
    # flat schema has no users/batch column, so the metric name carries both
    # (serve_vu<N>[_<tag-suffix>]_<metric>); MM1 rows get serve_mm_ prefix.
    # TTFT here INCLUDES server-side queueing — capacity-benchmark convention.
    # Rows flagged "valid": false (gate-failed configs) are SKIPPED.
    for fn in ("b2-concurrency-12b.json", "b2-concurrency-31b.json",
               "b2-concurrency-26b.json"):
        if not os.path.exists(os.path.join(HERE, fn)):
            continue
        d = load(fn)
        date = d["date"]
        for r in d["runs"]:
            if r["executor"] != "ConstantVUs" or r["bench_id"] != "throughput":
                continue
            if r.get("valid") is False:
                continue
            vus = r["max_vus"]
            tag = r["tag"]              # vllm[-*] | plow-b8[-24k] | mm-12b ...
            engine = r["engine"]
            mm = tag.startswith("mm-")
            if engine == "vllm":
                suffix = f"vu{vus}"
            elif mm:
                suffix = f"mm_vu{vus}"
            else:
                suffix = f"vu{vus}_" + "_".join(tag.split("-")[1:])
            ver = "vllm-0.25.1" if engine == "vllm" else "plowrt-serve"
            note = (f"{r['run_id']}; {r['successful_requests']} ok / "
                    f"{r['failed_requests']} failed; ttft includes queueing")
            if mm:
                note += "; 12B+26B co-resident, simultaneous load"
            ctx = r.get("prompt_tokens") or 4000
            camp = r.get("campaign") or d["campaign"]
            add(d["model"], engine, "bf16", "decode", 1, ctx,
                f"serve_{suffix}_agg_tok_s", r["aggregate_tok_s"],
                fn, camp, ver, date, note)
            add(d["model"], engine, "bf16", "decode", 1, ctx,
                f"serve_{suffix}_itl_avg_ms", r["itl_ms"]["avg"],
                fn, camp, ver, date, note)
            add(d["model"], engine, "bf16", "decode", 1, ctx,
                f"serve_{suffix}_itl_p99_ms", r["itl_ms"]["p99"],
                fn, camp, ver, date, note)
            add(d["model"], engine, "bf16", "prefill", 1, ctx,
                f"serve_{suffix}_ttft_avg_ms", r["ttft_ms"]["avg"],
                fn, camp, ver, date, note)
            add(d["model"], engine, "bf16", "prefill", 1, ctx,
                f"serve_{suffix}_ttft_p99_ms", r["ttft_ms"]["p99"],
                fn, camp, ver, date, note)

def do_m1_multimodel():
    if not os.path.exists(os.path.join(HERE, "m1-multimodel-sm120.json")):
        return  # loader activates when the campaign file lands
    # M1-multimodel — S1 switch cost (evict LRU resident + load target +
    # first token) between gemma-4-12B and gemma-4-26B-A4B, and the measured
    # co-residency point. The flat schema has no switch column, so the
    # metric names carry the phase breakdown (switch_*_ms); `ctx` is the
    # LIVE KV depth on the OUTGOING model at eviction (the wave-4 item 7b
    # variable — measured to not matter). phase=decode by convention (the
    # switch serves a decode request); direction is in the notes.
    fn = "m1-multimodel-sm120.json"
    d = load(fn)
    tgt_model = {"12B": "gemma-4-12B", "26B": "gemma-4-26B-A4B"}
    for camp in d["campaigns"]:
        cname, date = camp["campaign"], camp["date"]
        for r in camp.get("switch_rows", []):
            model = tgt_model[r["direction"].split("->")[1]]
            note = (f"S1 switch {r['direction']} run{r['run']}; outgoing live "
                    f"KV ctx {r['outgoing_kv_ctx']}; unload is the victim, "
                    f"load/first-token are the target")
            for metric, key in [
                ("switch_total_ms", "total_ms"),
                ("switch_load_ms", "load_ms"),
                ("switch_unload_ms", "unload_ms"),
                ("switch_first_token_ms", "first_token_ms"),
            ]:
                add(model, "plow", "bf16", "decode", 1, r["outgoing_kv_ctx"],
                    metric, r[key], fn, cname, "plowrt-serve", date, note)
        co = camp.get("co_residency_measured")
        if co:
            add("gemma-4-12B", "plow", "bf16", "decode", 1, 132096,
                "coresident_pair_vram_used_mib", co["vram_used_mib"],
                fn, cname, "plowrt-serve", date,
                f"pair: {co['pair']}; both stream concurrently, zero switches")

def do_vmm_prefix():
    # V1-vmm-prefix — VMM-backed KV prefix sharing (plans/rtx-09 V1).
    # attach_vs_copy is pool-level 31B-class full-layer geometry (80 KiB/token);
    # the flat schema has no block-size column, so the metric name carries it
    # (prefix_attach_b{MiB}mib_ms). The e2e section (12B ctx-8k engine) lands
    # as tpot_ms twin rows (vmm vs default) under the same campaign.
    fn = "vmm-prefix-v1.json"
    d = load(fn)
    c = d["campaigns"][0]
    date = c["date"]
    camp = c["campaign"]
    for r in c["attach_vs_copy"]:
        blk = r["block_mib"]
        note = f"block={blk} MiB; {c['metric_convention'][:80]}"
        for metric, key in [
            (f"prefix_attach_b{blk}mib_ms", "attach_ms"),
            (f"prefix_owner_build_b{blk}mib_ms", "owner_build_ms"),
            (f"prefix_detach_b{blk}mib_ms", "detach_ms"),
            (f"prefix_dedup_b{blk}mib_gib", "dedup_gib"),
        ]:
            add("gemma-4-31B", "plow", "bf16", "prefill", 1, r["prefix_rows"],
                metric, r[key], fn, camp, "plowrt-vmm-v1", date, note)
        add("gemma-4-31B", "plow", "bf16", "prefill", 1, r["prefix_rows"],
            "prefix_d2d_copy_ms", r["d2d_copy_ms"], fn, camp,
            "plowrt-vmm-v1", date, "cuMemcpyDtoD of the same bytes (status-quo blit)")
    for r in c.get("e2e_12b_ctx8k", []):
        add("gemma-4-12B", "plow", "bf16", "decode", 1, r["ctx"],
            r["metric"], r["value"], fn, camp, "plowrt-vmm-v1", date, r["notes"])


def do_ws_batched_gemv():
    # Weight-stationary wide GEMV rungs (GV_MM_MAX=16/32). Microbench decode
    # tok/s + tpot per batch, and the served b16-mm16 concurrency points.
    fn = "ws-batched-gemv.json"
    d = load(fn)
    camp, date = d["campaign"], d["date"]
    for r in d["microbench_decode"]:
        note = "B=%d GV_MM_MAX=%d (%d weight pass%s)" % (
            r["batch"], r["gv_mm_max"], r["weight_passes"],
            "" if r["weight_passes"] == 1 else "es")
        add("gemma-4-12B", "plow", "bf16", "decode", 1, 4096, "decode_tok_s",
            r["agg_tok_s"], fn, camp, "eafea6c", date, note)
        add("gemma-4-12B", "plow", "bf16", "decode", 1, 4096, "tpot_ms",
            r["tpot_ms"], fn, camp, "eafea6c", date, note)
    for p in d["serving_b16_mm16"]["points"]:
        if "agg_tok_s" not in p:
            continue
        note = "served b16-mm16 VU=%d (%s)" % (
            p["vu"], "SLO pass" if p.get("slo_pass") else "SLO fail")
        add("gemma-4-12B", "plow", "bf16", "decode", 1, 4000, "tok_s",
            p["agg_tok_s"], fn, camp, "eafea6c", date, note)
        add("gemma-4-12B", "plow", "bf16", "decode", 1, 4000, "itl_ms",
            p["itl_p99_ms"], fn, camp, "eafea6c", date, note + " ITL p99")
        add("gemma-4-12B", "plow", "bf16", "decode", 1, 4000, "ttft_ms",
            p["ttft_p99_ms"], fn, camp, "eafea6c", date, note + " TTFT p99")


def main():
    do_ws_batched_gemv()
    do_vmm_prefix()
    do_decode_only_sweep()
    do_longctx_sweep()
    do_vllm_tp()
    for fn in ("gemma4-vllm-fp8.json", "llama8b-vllm-fp8.json", "qwen3-4b-vllm-fp8.json"):
        do_vllm_fp8(fn)
    do_gemma12b_sm120_b1()
    do_gemma12b_plow_sm120_decode()
    do_gemma12b_plow_prefill()
    do_gemma12b_fp8_longctx()
    do_gemma12b_plowrt_ttft()
    for fn in ("gemma4-31b-plowrt-served.json", "gemma4-26b-plowrt-served.json"):
        do_plowrt_served(fn)
    do_gemma31b_sm120_b1()
    do_gemma31b_plow_sm120()
    do_gemma26b_sm120_b1()
    do_gemma26b_plow_sm120_decode()
    do_gemma_vllm_perf()
    for fn in ("llama-3.1-8b-vllm-docker.json", "qwen3-4b-vllm-docker.json"):
        do_vllm_docker(fn)
    for fn in ("llama-3.1-8b-vllm-perf.json", "qwen3-1.7b-vllm-perf.json", "qwen3-4b-vllm-perf.json"):
        do_vllm_inproc(fn)
    do_decode_breakdown()
    do_tp_prefill_oneshot()
    do_tp_prefill_twoshot()
    do_plow_vs_vllm_baseline()
    do_vllm_tp_baseline_plow()
    do_glm52_vllm()
    do_glm52_plow()
    do_glm52_plow_tuned()
    do_glm52_plow_256k()
    do_b2_concurrency()
    do_m1_multimodel()

    source_files = sorted({r["source_file"] for r in ROWS})
    metadata = {
        "title": "Consolidated perf-data index — plow vs vLLM on MI350X / gfx950 (CDNA4)",
        "generated": GENERATION_DATE,
        "generated_note": ("generation date is the latest git-commit date among the "
                           "sources; regenerate with `python3 perf-data/consolidate_perf.py`."),
        "row_count": len(ROWS),
        "hardware": "AMD MI350X / gfx950 (CDNA4), 8-GPU node, batch 1, single-user unless noted",
        "schema": {
            "model": "canonical model (gemma-4-31B|gemma-4-12B|llama-3.1-8B|qwen3-4B|qwen3-1.7B)",
            "engine": "plow | vllm",
            "precision": "bf16 | fp8 (weight-only) | fp8kv (fp8 weights + fp8 KV)",
            "phase": "decode | prefill",
            "tp": "tensor-parallel degree (int)",
            "ctx": "input context length in tokens (int)",
            "metric": "tpot_ms|ttft_ms|itl_ms|prefill_ms|prefill_tok_s|decode_tok_s|"
                      "decode_ms_per_token|gemv_TBps|decode_step_mean_ms|decode_kernel_<name>_ms",
            "value": "float transcribed verbatim; never interpolated",
            "source_file": "originating file in perf-data/",
            "campaign": "measurement-run label",
            "version": "engine build/version (vLLM version, or plow branch/source)",
            "git_commit": "commit the source was committed under, or (uncommitted)",
            "date": "measurement date (YYYY-MM-DD)",
            "notes": "per-entry caveats / provenance",
        },
        "source_files_transcribed": source_files,
        "summary_docs_not_re_transcribed": [
            "decode-only-sweep.md", "gemma4-31b-longctx-sweep.md",
            "vllm-docker-baseline.md", "vllm-fp8-baseline.md",
        ],
        "excluded": ("derived analysis in sources (ratios, crossover estimates, "
                     "scaling multipliers, speedup %) is intentionally NOT emitted as rows."),
    }
    with open(os.path.join(HERE, "all-perf-data.json"), "w") as f:
        json.dump({"metadata": metadata, "rows": ROWS}, f, indent=2)

    cols = ["model", "engine", "precision", "phase", "tp", "ctx", "metric", "value",
            "source_file", "campaign", "version", "git_commit", "date", "notes"]
    with open(os.path.join(HERE, "all-perf-data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)

    # console summary
    print("total rows:", len(ROWS))
    from collections import Counter
    by_model = Counter(r["model"] for r in ROWS)
    by_engine = Counter(r["engine"] for r in ROWS)
    by_phase = Counter(r["phase"] for r in ROWS)
    by_prec = Counter(r["precision"] for r in ROWS)
    print("by model:", dict(by_model))
    print("by engine:", dict(by_engine))
    print("by phase:", dict(by_phase))
    print("by precision:", dict(by_prec))
    print("source files:", len(source_files))

if __name__ == "__main__":
    main()
