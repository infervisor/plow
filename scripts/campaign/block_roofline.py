#!/usr/bin/env python3
"""Per-BLOCK roofline: one decoder layer, per layer kind, per rung.

    block_roofline.py <hf_dir> --gpu "H100 SXM5" --kind sliding|full \
        [--plow sweep.json] [--vllm layer.json] [--ctx 1024,4096,8192,16384] [--batch 1,4]

`roofline.py` prices the whole model; kernel work happens one block at a time
(`plowc --block L` + `block_run bench`, and `block_layer_bench.py` for vLLM's own
layer), so this prices ONE layer the same way and splits it by component, which is
what says where a rung's time can still go.

PREFILL of T rows is priced per component as max(FLOP time, byte time):
  linear      FLOPs = 2 * params * T          bytes = the weights, streamed once
  experts     FLOPs = 2 * k * 3*H*I * T       bytes = the UNION of touched experts
  attention   FLOPs = 4 * heads * hd * sum_t kv(t), kv(t) = min(t, window) when sliding
DECODE of B rows at context T is bandwidth: dense weights + the expert union of B rows + the
layer's KV read (window-capped when sliding). Sweep rows are joined on (batch, ctx).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from roofline import lookup_gpu  # noqa: E402


def layer_components(text: dict, kind: str) -> dict:
    h = text["hidden_size"]
    heads = text["num_attention_heads"]
    if kind == "full":
        hd = text.get("global_head_dim") or text["head_dim"]
        kvh = text.get("num_global_key_value_heads") or text["num_key_value_heads"]
        # Gemma-4 full-attention layers share K and V (attention_k_eq_v): no v_proj.
        n_kv_proj = 1 if text.get("attention_k_eq_v") else 2
        window = 0
    else:
        hd, kvh, n_kv_proj = text["head_dim"], text["num_key_value_heads"], 2
        window = text.get("sliding_window") or 0
    inter = text["intermediate_size"]
    n_exp = text.get("num_experts") or 0
    top_k = text.get("top_k_experts") or text.get("num_experts_per_tok") or 0
    i_moe = text.get("moe_intermediate_size") or text.get("expert_intermediate_size") or 0
    return {
        "h": h, "heads": heads, "hd": hd, "kvh": kvh, "window": window, "n_exp": n_exp,
        "top_k": top_k, "i_moe": i_moe,
        "attn_proj": h * heads * hd + n_kv_proj * h * kvh * hd + heads * hd * h,
        "dense_mlp": 3 * h * inter,
        "router": h * n_exp,
        "expert": 3 * h * i_moe,  # params of ONE expert
        "kv_per_tok": 2 * kvh * hd,  # K and V elements per token (cache is kept either way)
    }


def expert_union(n: int, k: int, rows: int) -> float:
    return n * (1.0 - (1.0 - k / n) ** rows) if n else 0.0


def attn_keys(t: int, window: int) -> float:
    if not window or t <= window:
        return t * (t + 1) / 2.0
    return window * (window + 1) / 2.0 + (t - window) * window


def prefill_roof(c: dict, t: int, flops: float, gbps: float, eb: int = 2) -> dict:
    out = {}
    lin = {"attn_proj": c["attn_proj"], "dense_mlp": c["dense_mlp"], "router": c["router"]}
    for name, params in lin.items():
        out[name] = max(2.0 * params * t / flops, params * eb / (gbps * 1e9)) * 1e3
    if c["n_exp"]:
        f = 2.0 * c["top_k"] * c["expert"] * t
        b = expert_union(c["n_exp"], c["top_k"], t) * c["expert"] * eb
        out["experts"] = max(f / flops, b / (gbps * 1e9)) * 1e3
    out["attention"] = 4.0 * c["heads"] * c["hd"] * attn_keys(t, c["window"]) / flops * 1e3
    return out


def decode_roof(c: dict, b: int, t: int, gbps: float, eb: int = 2) -> dict:
    kv = min(t, c["window"]) if c["window"] else t
    parts = {
        "dense": (c["attn_proj"] + c["dense_mlp"] + c["router"]) * eb,
        "experts": expert_union(c["n_exp"], c["top_k"], b) * c["expert"] * eb,
        "kv": b * kv * c["kv_per_tok"] * eb,
    }
    return {k: v / (gbps * 1e9) * 1e3 for k, v in parts.items()}


def load(path):
    if not path:
        return {}
    rows = json.load(open(path))
    rows = (rows.get("sweep") or rows.get("rows") or []) if isinstance(rows, dict) else rows
    return {(r["batch"], r["ctx"]): r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--gpu", default="H100 SXM5")
    ap.add_argument("--kind", choices=["sliding", "full"], required=True)
    ap.add_argument("--plow")
    ap.add_argument("--vllm")
    ap.add_argument("--ctx", default="1024,4096,8192,16384")
    ap.add_argument("--batch", default="1,4")
    a = ap.parse_args()

    cfg = json.load(open(Path(a.hf_dir) / "config.json"))
    c = layer_components(cfg.get("text_config", cfg), a.kind)
    gpu = lookup_gpu(a.gpu)
    flops, gbps = gpu.peak_tflops("bf16") * 1e12, gpu.bandwidth_for_bound_gbps
    plow, vllm = load(a.plow), load(a.vllm)
    ctxs = [int(x) for x in a.ctx.split(",")]

    print(f"{a.kind} layer on {a.gpu}: {flops / 1e12:.0f} TFLOP/s, {gbps:.0f} GB/s | heads {c['heads']} hd {c['hd']} "
          f"kvh {c['kvh']} window {c['window'] or '-'} | experts {c['n_exp']} top-{c['top_k']}")
    print("\nPREFILL, B=1, ms per layer")
    print(f"{'T':>6} | {'proj':>6} {'mlp':>6} {'router':>6} {'experts':>7} {'attn':>6} {'ROOF':>7} | "
          f"{'plow':>7} {'%roof':>6} | {'vllm':>7} {'plow/vllm':>9}")
    for t in ctxs:
        r = prefill_roof(c, t, flops, gbps)
        roof = sum(r.values())
        p = plow.get((1, t), {}).get("prefill_ms_median")
        v = vllm.get((1, t), {}).get("prefill_ms_median")
        print(f"{t:6d} | {r['attn_proj']:6.3f} {r['dense_mlp']:6.3f} {r['router']:6.3f} {r.get('experts', 0):7.3f} "
              f"{r['attention']:6.3f} {roof:7.3f} | "
              f"{p if p is not None else float('nan'):7.2f} {100 * roof / p if p else float('nan'):5.1f}% | "
              f"{v if v is not None else float('nan'):7.2f} {p / v if p and v else float('nan'):9.2f}")

    print("\nDECODE step, ms per layer (bandwidth roof)")
    print(f"{'B':>3} {'T':>6} | {'dense':>6} {'experts':>7} {'kv':>6} {'ROOF':>6} | {'plow':>7} {'%roof':>6} | "
          f"{'vllm':>7} {'plow/vllm':>9}")
    for b in [int(x) for x in a.batch.split(",")]:
        for t in ctxs:
            r = decode_roof(c, b, t, gbps)
            roof = sum(r.values())
            p = plow.get((b, t), {}).get("latency_us_median")
            v = vllm.get((b, t), {}).get("latency_us_median")
            p, v = (p / 1e3 if p else None), (v / 1e3 if v else None)
            print(f"{b:3d} {t:6d} | {r['dense']:6.3f} {r['experts']:7.3f} {r['kv']:6.3f} {roof:6.3f} | "
                  f"{p if p is not None else float('nan'):7.3f} {100 * roof / p if p else float('nan'):5.1f}% | "
                  f"{v if v is not None else float('nan'):7.3f} {p / v if p and v else float('nan'):9.2f}")


if __name__ == "__main__":
    main()
