#!/usr/bin/env python3
"""block_op_bench.py — PER-OP single-block breakdown of a vLLM decoder layer.

Companion to block_layer_bench.py (which times the WHOLE block). This times each
op of the block INDIVIDUALLY, for DECODE (M=1) and PREFILL (M=T), across a ctx
sweep -- so onboarding/tuning a NEW model needs only a single-block run, never a
full-model load. The plow kernel agent reads the per-op table to see which op
dominates and what bandwidth / FLOP rate to beat.

Method — L2-flushed CUDA-graph-per-op microbench:
  Each op is captured into a CUDA graph, so replay carries ~no launch/python
  overhead -- what remains is device kernel time, the same regime real decode
  runs in (plow's packet and vLLM's cudagraph both erase launch overhead). An L2
  flush (>50 MB write) between replays forces a COLD weight read from HBM,
  matching real decode where consecutive ops touch different weights whose
  combined footprint far exceeds the 50 MB L2. Two event-bracketed loops
  (flush;op) and (flush) are subtracted:  op_us = (T_both - T_flush)/N.
  A kernel that refuses capture falls back to eager (launch-inclusive), marked.

Metrics per op:
  us      device time
  GB/s    read_bytes / us   (weights + activations; the decode GEMV target)
  TFLOP/s 2*M*N*K / us      (the prefill tensor-core GEMM target; NA for norms/attn)

Imports block_layer_bench.py from the block-baseline-harness worktree so layer
construction cannot drift. Run under gpulease; needs the vLLM venv.

  PLOW_PY=/workspace/venvs/vllm-blk/bin/python \
    gpulease blkop python block_op_bench.py <cfg> --phases decode,prefill --ctx 1024,4096
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
HARNESS = os.environ.get("BLB_DIR", "/root/plow/.claude/worktrees/block-baseline-harness/scripts")
sys.path.insert(0, HARNESS)
import torch
import block_layer_bench as blb

_CFG_CM = None  # holds the set_current_vllm_config context for the program lifetime


def flushed_us(op, flush, iters, warmup=30):
    """(median device-us, graphed?) for `op` with cold weights (L2 flushed)."""
    for _ in range(warmup):
        flush.zero_(); op()
    torch.cuda.synchronize()
    graphed = True
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                op()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            op()
        replay = g.replay
    except Exception:
        graphed = False
        replay = op
    for _ in range(10):
        flush.zero_(); replay()
    torch.cuda.synchronize()

    def loop(n, with_op):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(n):
            flush.zero_()
            if with_op:
                replay()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3
    both, only = [], []
    for _ in range(5):
        both.append(loop(iters, True)); only.append(loop(iters, False))
    both.sort(); only.sort()
    return max((both[2] - only[2]) / iters, 0.0), graphed


def build_layer(spec, cfg_d, arch, layer_idx, dt, dev, quant=None):
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    LayerCls = blb.resolve_layer_class(arch, spec.get("layer_class"))
    quant_config = None
    if quant == "fp8":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        quant_config = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    cache_config = CacheConfig(block_size=blb.BLOCK_SIZE, gpu_memory_utilization=0.1, cache_dtype="auto")
    vllm_config = VllmConfig(cache_config=cache_config)
    global _CFG_CM
    _CFG_CM = set_current_vllm_config(vllm_config)
    _CFG_CM.__enter__()  # held for program lifetime (must not be GC'd)
    init_distributed_environment(world_size=1, rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{spec.get('port',29601)}", local_rank=0, backend="nccl")
    ensure_model_parallel_initialized(1, 1)
    try:
        from vllm.v1.worker.workspace import init_workspace_manager, is_workspace_manager_initialized
        if not is_workspace_manager_initialized():
            init_workspace_manager(dev)
    except ImportError:
        pass
    prefix = blb.PREFIX_TMPL.format(i=layer_idx)
    cfg = blb.DuckConfig(cfg_d)
    if quant_config is not None and vllm_config.model_config is None:
        # online fp8 method + attention backend selection need a real ModelConfig
        # (dtype, head counts). Build one from the local HF dir (config only, no weights).
        from vllm.config import ModelConfig
        mpath = spec.get("model_path", "/workspace/models/gemma-4-26B-A4B-it")
        vllm_config.model_config = ModelConfig(model=mpath, tokenizer=mpath, dtype="bfloat16",
                                               trust_remote_code=True, seed=0, enforce_eager=True)
    with torch.device(dev):
        old = torch.get_default_dtype(); torch.set_default_dtype(dt)
        try:
            layer = LayerCls(config=cfg, cache_config=cache_config, quant_config=quant_config, prefix=prefix)
        finally:
            torch.set_default_dtype(old)
    # quant methods create weights on meta (materialized at load time); to_empty places
    # real storage on-device so the random-init below can fill them.
    if any(p.is_meta for p in layer.parameters()):
        layer = layer.to_empty(device=dev).eval()
    else:
        layer = layer.to(dev).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.02) if p.dim() >= 2 else p.fill_(1.0)
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    for mod in layer.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, QuantizeMethodBase):
            qm.process_weights_after_loading(mod)
    return layer, vllm_config


def run_phase(layer, cfg_d, vllm_config, phase, t, iters, flush, dt, dev):
    from vllm.forward_context import set_forward_context
    attn = layer.self_attn.attn
    attn_name = attn.layer_name
    hidden = int(cfg_d["hidden_size"])
    n_kv = int(cfg_d.get("num_key_value_heads") or cfg_d["num_attention_heads"])
    n_q = int(cfg_d["num_attention_heads"])
    head_dim = int(cfg_d.get("head_dim") or hidden // n_q)
    bpe = 2
    decode = (phase == "decode")
    M = 1 if decode else t

    per_req = (t + 1 + blb.BLOCK_SIZE - 1) // blb.BLOCK_SIZE
    kv_shape = attn.attn_backend.get_kv_cache_shape(per_req, blb.BLOCK_SIZE, n_kv, head_dim)
    kv_cache = torch.zeros(kv_shape, dtype=dt, device=dev)
    attn.kv_cache = kv_cache
    block_table = torch.arange(per_req, dtype=torch.int32).view(1, per_req).contiguous()
    if decode:
        md, slots = blb.make_metadata([1], [t + 1], block_table, dev)
        pos = torch.full((1,), t, dtype=torch.long, device=dev)
    else:
        md, slots = blb.make_metadata([t], [t], block_table, dev)
        pos = torch.arange(t, dtype=torch.long, device=dev)

    results = []

    def rec(label, op, rd_bytes, flops=0):
        us, gr = flushed_us(op, flush, iters)
        gbs = rd_bytes / us / 1e3 if us > 0 else 0.0
        tfs = flops / us / 1e6 if (us > 0 and flops) else 0.0
        row = {"op": label, "us": round(us, 3), "GBps": round(gbs, 1)}
        if flops:
            row["TFLOPs"] = round(tfs, 1)
        if not gr:
            row["eager"] = True
        results.append(row)
        extra = f"{tfs:8.1f} TF/s" if flops else " " * 13
        print(f"  {label:<20} {us:9.3f} us  {gbs:8.1f} GB/s  {extra}{'' if gr else '  [eager]'}")

    with set_forward_context(md, vllm_config, num_tokens=M, slot_mapping={attn_name: slots}), \
         torch.inference_mode():
        sa = layer.self_attn
        x = torch.randn(M, hidden, dtype=dt, device=dev) * 0.1
        q_size = n_q * head_dim; kv_size = n_kv * head_dim
        qkvN = q_size + 2 * kv_size
        rec("qkv_proj", lambda: sa.qkv_proj(x), (hidden * qkvN + M * (hidden + qkvN)) * bpe, 2 * M * qkvN * hidden)
        qkv, _ = sa.qkv_proj(x)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        qn = sa.q_norm(q.unflatten(-1, (n_q, head_dim))).flatten(-2, -1)
        kn = sa.k_norm(k.unflatten(-1, (n_kv, head_dim))).flatten(-2, -1)
        vn = sa.v_norm(v.unflatten(-1, (n_kv, head_dim))).flatten(-2, -1)
        qr, kr = sa.rotary_emb(pos, qn, kn)
        # attention read bytes: decode reads full KV (ctx); prefill streams M q x M kv
        attn_bytes = (t * n_kv * head_dim * 2 * bpe) if decode else (M * n_q * head_dim * 2 * bpe)
        try:
            attn_out = sa.attn(qr, kr, vn)
            rec("attn", lambda: sa.attn(qr, kr, vn), attn_bytes)
        except Exception as e:  # backend/metadata mismatch (attn is unquantized → == bf16)
            print(f"  attn                     SKIPPED ({type(e).__name__}); using q_size zeros")
            attn_out = torch.zeros(M, q_size, dtype=dt, device=dev)
        rec("o_proj", lambda: sa.o_proj(attn_out), (q_size * hidden + M * (q_size + hidden)) * bpe, 2 * M * hidden * q_size)
        mlp = layer.mlp
        inter = int(cfg_d["intermediate_size"])
        rec("mlp_gate_up", lambda: mlp.gate_up_proj(x), (hidden * 2 * inter + M * (hidden + 2 * inter)) * bpe, 2 * M * 2 * inter * hidden)
        gu, _ = mlp.gate_up_proj(x)
        act = mlp.act_fn(gu)
        rec("mlp_down", lambda: mlp.down_proj(act), (inter * hidden + M * (inter + hidden)) * bpe, 2 * M * hidden * inter)
        if getattr(layer, "enable_moe_block", False) and layer.moe is not None:
            router = layer.router
            rl = router(x); rl_t = rl[0] if isinstance(rl, tuple) else rl
            n_exp = int(cfg_d["num_experts"])
            rec("moe_router", lambda: router(x), (hidden * n_exp + M * (hidden + n_exp)) * bpe, 2 * M * n_exp * hidden)
            topk = int(cfg_d["top_k_experts"]); moe_int = int(cfg_d["moe_intermediate_size"])
            # decode: each of M tokens gathers top_k experts (weight-bound). prefill:
            # all experts likely resident, act-bound; weight read bounded by 3*E*mi*h.
            if decode:
                moe_bytes = M * topk * (hidden * moe_int * 2 + moe_int * hidden) * bpe
            else:
                moe_bytes = min(M * topk, n_exp) * (hidden * moe_int * 2 + moe_int * hidden) * bpe
            moe_flops = 2 * M * topk * (2 * moe_int * hidden + hidden * moe_int)
            rec("moe_experts", lambda: layer.moe(x, rl_t), moe_bytes, moe_flops)
        rec("rmsnorm(one)", lambda: layer.input_layernorm(x), M * hidden * bpe)

        # whole-block cross-check (skipped if attn backend mismatched)
        def whole():
            layer(positions=pos, hidden_states=x, residual=None)
        try:
            for _ in range(20):
                whole()
            wl = []
            for _ in range(60 if decode else 20):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); whole(); e1.record(); torch.cuda.synchronize()
                wl.append(e0.elapsed_time(e1) * 1e3)
            wl.sort(); whole_us = wl[len(wl) // 2]
        except Exception as e:
            print(f"  whole-block SKIPPED ({type(e).__name__})")
            whole_us = 0.0

    op_sum = sum(r["us"] for r in results)
    print(f"  {'SUM of ops':<20} {op_sum:9.3f} us     whole-block(eager) {whole_us:9.3f} us")
    del kv_cache; torch.cuda.empty_cache()
    return {"phase": phase, "ctx": t, "M": M, "ops": results,
            "op_sum_us": round(op_sum, 3), "whole_block_eager_us": round(whole_us, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--phases", default="decode")
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--flush-mb", type=int, default=96)
    ap.add_argument("--out", default="/dev/shm/block-op/opbench.json")
    ap.add_argument("--quant", default=None, help="None (bf16) or fp8")
    args = ap.parse_args()

    dev = torch.device("cuda:0"); dt = torch.bfloat16
    torch.cuda.set_device(dev); torch.manual_seed(0)
    spec = json.loads(Path(args.config).read_text())
    cfg_d = spec["config"]; arch = spec.get("arch") or cfg_d["architectures"][0]
    name = spec.get("name", Path(args.config).stem)
    layer_idx = int(spec.get("layer_index", 0))

    layer, vllm_config = build_layer(spec, cfg_d, arch, layer_idx, dt, dev, quant=args.quant)
    print(f"block {name}  hidden={cfg_d['hidden_size']} quant={args.quant} dev={torch.cuda.get_device_name(0)}")
    flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device=dev)

    runs = []
    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        for t in [int(x) for x in args.ctx.split(",") if x.strip()]:
            print(f"\n=== {phase.upper()}  ctx={t} ===")
            runs.append(run_phase(layer, cfg_d, vllm_config, phase, t, args.iters, flush, dt, dev))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"block": name, "device": torch.cuda.get_device_name(0),
        "method": "L2-flushed cudagraph-per-op microbench", "runs": runs}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
