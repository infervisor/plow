#!/usr/bin/env python3
"""Practical GEMM ceiling on THIS box at YOUR model's exact prefill shapes
(docs/bringup/07-perf-campaign.md). cuBLASLt via torch._scaled_mm (fp8) and torch.matmul (bf16).

Knowing the real ceiling is what separates "the kernel is slow" from "the box
is the wall": on GH200/Gemma-12B it measured fp8 1324-1468 / bf16 804-861 TF/s
and directly justified the 384-thread warp-specialized object.

Shapes are (M=chunk_rows, N=out_features, K=in_features, name), per RANK: a TP
rank runs the sharded GEMM, so the ceiling that bounds a served token is the
sharded one. Pick a model's set with `--model`, and the TP degree with `--tp`:

  perf-data/tools/gpulease -n 1 ceil \\
    perf-data/tools/bringup_ceiling.py --model glm53 --tp 8

`--model gemma12b` (the default) reproduces the original hardcoded set at tp=1
byte for byte, so every number this tool has already reported still stands.
"""
import argparse
import time

import torch

# Per-model prefill GEMM shapes. Each entry is a callable taking the TP degree
# and returning the per-rank (M, N, K, name) list, because which dimension the
# shard falls on is a property of the projection, not of the model:
#   * a column-parallel projection (q/kv/gate/up, expert glu) shards N
#   * a row-parallel one (o_proj, down) shards K
# Getting that backwards yields a ceiling for a GEMM the rank never runs.


def _gemma12b(tp: int):
    # The original hardcoded set. Dense GQA, tp=1 — kept verbatim as the
    # regression anchor for the numbers already published against it.
    assert tp == 1, "the gemma12b anchor set is only defined at tp=1"
    return [
        (4096, 4096, 3840, "q_proj"),
        (4096, 512, 3840, "kv_proj"),
        (4096, 3840, 4096, "o_proj"),
        (4096, 15360, 3840, "gate/up"),
        (4096, 3840, 15360, "down"),
    ]


def _glm53(tp: int, m: int = 4096):
    # GLM-5.3 (glm_moe_dsa): MLA with q_lora_rank 2048 / kv_lora_rank 512,
    # 64 heads x qk_head_dim 256, v_head_dim 256, hidden 6144. The two LoRA
    # DOWN-projections (q_a, kv_a) are NOT sharded — every rank computes the
    # full latent, which is why they carry no `/tp` and why their cost does not
    # fall as TP rises.
    h, heads, qk, v = 6144, 64, 256, 256
    q_lora, kv_lora, rope = 2048, 512, 64
    nope = 192
    hd = heads // tp
    return [
        (m, q_lora, h, "q_a_proj"),
        (m, hd * qk, q_lora, "q_b_proj"),
        (m, kv_lora + rope, h, "kv_a_proj"),
        (m, hd * (nope + v), kv_lora, "kv_b_proj"),
        (m, h, hd * v, "o_proj"),
        # Dense layers (first_k_dense_replace = 3) keep intermediate_size 12288.
        (m, 2 * (12288 // tp), h, "dense gate/up"),
        (m, h, 12288 // tp, "dense down"),
        # MoE: 256 experts, top-8, moe_intermediate_size 2048. The per-expert M
        # is the routed share of the chunk, not the chunk: top_k/n_exp of it.
        (max(1, m * 8 // 256), 2 * 2048, h, "expert gate/up"),
        (max(1, m * 8 // 256), h, 2048, "expert down"),
    ]


def _k27(tp: int, m: int = 4096):
    # Kimi-K2.7-Code text tower (kimi_k2): MLA, hidden 7168, q_lora_rank 1536,
    # kv_lora_rank 512, 64 heads, qk_nope 128 + qk_rope 64, v_head_dim 128,
    # 384 experts top-8, moe_intermediate_size 2048.
    h, heads = 7168, 64
    q_lora, kv_lora, rope, nope, v = 1536, 512, 64, 128, 128
    hd = heads // tp
    return [
        (m, q_lora, h, "q_a_proj"),
        (m, hd * (nope + rope), q_lora, "q_b_proj"),
        (m, kv_lora + rope, h, "kv_a_proj"),
        (m, hd * (nope + v), kv_lora, "kv_b_proj"),
        (m, h, hd * v, "o_proj"),
        (max(1, m * 8 // 384), 2 * 2048, h, "expert gate/up"),
        (max(1, m * 8 // 384), h, 2048, "expert down"),
    ]


def _dsv4(tp: int, m: int = 4096):
    # DeepSeek-V4-Flash (deepseek_v4). plow cannot serve this yet — these shapes bound what
    # it could reach on this part, they do not describe a running model. Geometry from the
    # shipped config.json: hidden 4096, 43 layers, 64 heads at head_dim 512 with
    # num_key_value_heads=1 (the DK=512 DR=0 MLA), q_lora_rank 1024, 256 experts top-6 at
    # moe_intermediate_size 2048, and a GROUPED output LoRA (o_groups 8 x o_lora_rank 1024)
    # that has no analogue in the other two models — it is priced here as its two factors.
    h, heads, hd = 4096, 64, 512
    q_lora, rope = 1024, 64
    o_groups, o_lora = 8, 1024
    hdl = heads // tp
    return [
        (m, q_lora, h, "q_a_proj"),
        (m, hdl * hd, q_lora, "q_b_proj"),
        # kv is num_key_value_heads=1: one head's worth, NOT sharded by tp.
        (m, hd + rope, h, "kv_a_proj"),
        # Grouped output LoRA: down into the per-group rank, then back out to hidden.
        (m, o_groups * o_lora // tp, hdl * hd, "o_lora_down"),
        (m, h, o_groups * o_lora // tp, "o_lora_up"),
        # 256 experts top-6 -> a smaller routed share per expert than GLM's top-8.
        (max(1, m * 6 // 256), 2 * 2048, h, "expert gate/up"),
        (max(1, m * 6 // 256), h, 2048, "expert down"),
    ]


MODELS = {"gemma12b": _gemma12b, "glm53": _glm53, "k27": _k27, "dsv4": _dsv4}


def fp8_dtype():
    """The e4m3 encoding THIS backend's scaled GEMM accepts.

    Not a preference: CDNA3 (gfx942/MI300X) implements the `fnuz` variant — no
    negative zero, exponent bias 8 — and `torch._scaled_mm` hard-refuses the OCP
    `e4m3fn` operand there, while Hopper/Blackwell and gfx950 take `e4m3fn`.
    Probed rather than derived from `torch.version.hip`, because the split is per
    ISA, not per vendor. The two encodings differ in bias, so a number measured
    under one is not the other's number — the header row prints which ran.
    """
    for dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        try:
            a = torch.zeros(16, 32, device="cuda", dtype=torch.float16).to(dt)
            s = torch.ones(1, 1, device="cuda")
            torch._scaled_mm(a, a.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
            return dt
        except (ValueError, RuntimeError, TypeError):
            continue
    raise SystemExit("no e4m3 encoding accepted by torch._scaled_mm on this device")


def bench(f, it=20, warm=5):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(MODELS))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rows", type=int, default=4096, help="prefill chunk rows (M)")
    a = ap.parse_args()
    shapes = MODELS[a.model](a.tp) if a.model == "gemma12b" else MODELS[a.model](a.tp, a.rows)

    torch.cuda.init()
    f8 = fp8_dtype()
    print(f"# model={a.model} tp={a.tp} rows={a.rows} dev={torch.cuda.get_device_name(0)} fp8={str(f8).split('.')[-1]}")
    print(f"{'shape':16s} {'M':>6s} {'N':>6s} {'K':>6s} {'fp8 ms':>8s} {'fp8 TF/s':>9s} {'bf16 ms':>8s} {'bf16 TF/s':>10s}")
    for m, n, k, name in shapes:
        fl = 2.0 * m * n * k
        a8 = torch.randn(m, k, device="cuda", dtype=torch.float16).to(f8)
        b8 = torch.randn(n, k, device="cuda", dtype=torch.float16).to(f8)
        sa = torch.ones(m, 1, device="cuda")
        sb = torch.ones(1, n, device="cuda")
        dt8 = bench(lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb,
                                             out_dtype=torch.bfloat16))
        ab = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        bb = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        dtb = bench(lambda: ab @ bb)
        print(f"{name:16s} {m:6d} {n:6d} {k:6d} {dt8*1e3:8.3f} {fl/dt8/1e12:9.1f} {dtb*1e3:8.3f} {fl/dtb/1e12:10.1f}")


if __name__ == "__main__":
    main()
