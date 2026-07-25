#!/usr/bin/env python3
# glm52_sweep_json.py — assemble perf-data/glm52-plow-decode.json from the GLM-5.2 plow decode
# sweep. Reads parsed "<tp> <ctx> <tpot_ms>" rows on stdin (one per line; ctx/tpot as measured)
# and emits the JSON in the SAME schema as perf-data/glm52-vllm-decode.json (engine=plow) so the
# two files line up row-for-row and consolidate_perf.py can flatten both with one handler.
#
#   glm52_sweep.sh pipes its parsed rows here:  ... | glm52_sweep_json.py --out perf-data/glm52-plow-decode.json
#   manual:  printf '4 4096 6.1\n8 4096 3.4\n' | python3 scripts/glm52_sweep_json.py --out /tmp/x.json
import sys, json, argparse, datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="plow @ glm52-prep (full-model decode)")
    ap.add_argument("--notes", default="", help="global note applied to every row's empty note")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args()

    results = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 3:
            continue
        tp, ctx, tpot = int(f[0]), int(f[1]), float(f[2])
        note = " ".join(f[3:]) if len(f) > 3 else args.notes
        results.append({"tp": tp, "ctx": ctx, "tpot_ms": round(tpot, 3), "notes": note})
    results.sort(key=lambda r: (r["tp"], r["ctx"]))

    doc = {
        "model": "GLM-5.2-FP8",
        "model_arch": ("GlmMoeDsaForCausalLM (glm_moe_dsa; DeepSeek-V3.2-DSA-class: MLA + sparse-"
                       "attention indexer + fine-grained block-fp8 MoE, 78 layers, 256 experts "
                       "top-8 + 1 shared, hidden 6144, 1M max ctx)"),
        "engine": "plow",
        "version": args.version,
        "gpu": "AMD gfx950 (MI350X-class, CDNA4), 8-GPU node; HIP_VISIBLE_DEVICES pins shard GPUs",
        "date": args.date,
        "precision": "fp8",
        "quantization": ("fp8 block (e4m3, weight_block_size [128,128], dynamic acts) — experts + "
                         "attn/dense projections block-fp8 verbatim; norms/router/embed/lm_head + "
                         "MLA-derived weights bf16 (matches the frozen declare_glm prep contract)"),
        "dtype": "bfloat16 (compute; MLA latent KV bf16)",
        "batch": 1,
        "concurrency": 1,
        "bit_exact": False,
        "metric": "tpot_ms (median 1-token decode step, ms/tok)",
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[sweep-json] wrote {args.out}: {len(results)} rows "
          f"(tp {sorted({r['tp'] for r in results})}, "
          f"ctx {sorted({r['ctx'] for r in results})})")


if __name__ == "__main__":
    main()
