#!/usr/bin/env python3
"""scripts/block_layer_bench.py — FRAMEWORK single-block baseline via vLLM's OWN decoder layers.

Where `block_baseline.py` hand-writes a block per architecture family, this
drives **vLLM's real decoder layer classes** directly — no engine, no server, no
HTTP, no checkpoint. Adding a model is adding a JSON config, not code.

    HF config .json  ->  architectures[0]  ->  vLLM model module  ->  *DecoderLayer

Verified resolution (vLLM 0.25.1):

| architecture                | module        | layer class              | runs here |
|-----------------------------|---------------|--------------------------|-----------|
| Gemma4ForCausalLM           | gemma4        | Gemma4DecoderLayer       | YES       |
| Glm4MoeForCausalLM          | glm4_moe      | Glm4MoeDecoderLayer      | not yet   |
| GlmMoeDsaForCausalLM (5.2)  | deepseek_v2   | DeepseekV2DecoderLayer   | not yet   |
| DeepseekV3/V32ForCausalLM   | deepseek_v2   | DeepseekV2DecoderLayer   | not yet   |
| KimiLinearForCausalLM       | kimi_linear   | KimiDecoderLayer         | not yet   |

GLM-5.2, DeepSeek-V3/V3.2 and **Kimi-K2** collapse onto vLLM's shared MLA+MoE
`DeepseekV2DecoderLayer` (K2 ships `DeepseekV3ForCausalLM` in its config -- there
is no KimiK2 class in the registry; `kimi_linear` is the separate Kimi *Linear*
model, NOT K2).

SCOPE TODAY: the Gemma family runs end to end. The MLA family resolves correctly
but does NOT yet run through this harness, because it needs three things this
script does not do yet:
  * a DIFFERENT constructor signature -- DeepseekV2DecoderLayer is
    (vllm_config, prefix, config=None, topk_indices_buffer=None), not the
    Gemma-style (config=, cache_config=, quant_config=, prefix=)
  * a REAL ModelConfig -- it does `model_config.use_mla` unconditionally, so the
    omit-model_config trick used here raises AttributeError on None. Needs a
    synthesized config.json on disk fed to ModelConfig(...)
  * MLA attention plumbing -- the module is at self_attn.mla_attn.mla_attn (not
    self_attn.attn), the backend is FLASH_ATTN_MLA with head_size 576 and
    num_kv_heads 1, the KV cache is 3-D (blocks, block_size, 576), and metadata
    comes from FlashAttnMLAMetadataBuilder fed a CommonAttentionMetadata -- the
    hand-built FlashAttentionMetadata below does not apply.
Also note kv_lora_rank + qk_rope_head_dim must be 320 or 576 (the only MLA head
sizes vLLM supports), so those dims cannot be shrunk arbitrarily.

WHY THIS IS THE RIGHT BASELINE
------------------------------
It is vLLM's actual kernel path for that architecture — its attention backend,
its fused MoE, its norms — on exactly one block. No reimplementation to be wrong
about, and no full-model serving overhead to subtract out.

MEASURES BOTH PHASES over a B x T grid, same as `block_baseline.py`:
  prefill — B sequences of T tokens, causal, one full-block pass
  decode  — B sequences of 1 token against the T-token paged KV cache
Both timed with CUDA events; median / p95; emits `block_run`'s sweep.json schema.

Weights are random: per-step kernel time is data-independent (see block_run.rs),
so no checkpoint or HF download is needed — which also sidesteps gated repos.

Usage:
  python3 scripts/block_layer_bench.py perf-data/block-configs/gemma4-12b.json \
      --batch 1,4 --ctx 128,1024 --out /dev/shm/bb/12b-layer.json

Needs a vLLM venv: PLOW_PY=/workspace/venvs/vllm-blk/bin/python
Prefer scripts/block_layer_bench.sh (wraps in gpulease for multi-agent GPU safety).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch

BLOCK_SIZE = 16  # FLASH_ATTN requires a multiple of 16
PREFIX_TMPL = "model.layers.{i}"


# --------------------------------------------------------------------------- #
# Architecture -> vLLM decoder layer class (generic; no per-model code)
# --------------------------------------------------------------------------- #
def resolve_layer_class(arch: str, override: str | None = None):
    """architectures[0] -> vLLM's decoder layer class for that model."""
    from vllm.model_executor.models.registry import (
        _MULTIMODAL_MODELS,
        _TEXT_GENERATION_MODELS,
    )

    table = {}
    for d in (_TEXT_GENERATION_MODELS, _MULTIMODAL_MODELS):
        table.update(d)
    entry = table.get(arch)
    if entry is None:
        raise SystemExit(f"architecture {arch!r} not in vLLM's registry")
    mod_name = entry[0] if isinstance(entry, (tuple, list)) else entry
    mod = importlib.import_module(f"vllm.model_executor.models.{mod_name}")

    if override:
        return getattr(mod, override)
    cands = [
        n
        for n, c in inspect.getmembers(mod, inspect.isclass)
        if n.endswith("DecoderLayer") and c.__module__ == mod.__name__
    ]
    if not cands:
        raise SystemExit(
            f"{arch} -> module {mod_name} exposes no *DecoderLayer; "
            f"set \"layer_class\" in the config to name one explicitly"
        )
    # Prefer the plainest name when a module ships several variants.
    cands.sort(key=len)
    return getattr(mod, cands[0])


class DuckConfig:
    """Duck-typed HF config: every key in the JSON becomes an attribute.

    vLLM layers read plain attributes, so a real PretrainedConfig is unnecessary
    (and would drag in a gated HF download).
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def __getattr__(self, name):  # unknown attrs -> None, like a sparse config
        return None


# --------------------------------------------------------------------------- #
# Paged KV cache + attention metadata (hand-built: the vLLM metadata BUILDER
# needs a real ModelConfig, which we deliberately do not have)
# --------------------------------------------------------------------------- #
def make_metadata(query_lens, seq_lens, block_table, device):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    qsl = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    slots = []
    for req, (q, s) in enumerate(zip(query_lens, seq_lens)):
        for pos in range(s - q, s):
            blk = int(block_table[req, pos // BLOCK_SIZE])
            slots.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    md = FlashAttentionMetadata(
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        query_start_loc=qsl,
        max_seq_len=max(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_table=block_table.to(device),
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=None,
        max_num_splits=0,
        causal=True,
        # None => the impl uses its OWN per-layer window; a tuple would override
        # and silently break sliding-attention layers.
        sliding_window=None,
    )
    return md, slot_mapping


def time_gpu(fn, iters):
    us = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        us.append(e0.elapsed_time(e1) * 1e3)
    us.sort()
    return us[len(us) // 2], us[min(int(len(us) * 0.95), len(us) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM-native single decoder-layer baseline")
    ap.add_argument("config", help="block config json (arch + HF config attrs)")
    ap.add_argument("--batch", default="1,4")
    ap.add_argument("--ctx", default="128,1024")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--prefill-iters", type=int, default=10)
    ap.add_argument("--no-cudagraph", action="store_true",
                    help="eager decode (exposes kernel-launch overhead)")
    ap.add_argument("--out", default="/dev/shm/block-baseline/layer_sweep.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    spec = json.loads(Path(args.config).read_text())
    cfg_d = spec["config"]
    arch = spec.get("arch") or cfg_d.get("architectures", [None])[0]
    layer_idx = int(spec.get("layer_index", 0))
    name = spec.get("name", Path(args.config).stem)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.forward_context import set_forward_context

    LayerCls = resolve_layer_class(arch, spec.get("layer_class"))
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, gpu_memory_utilization=0.1, cache_dtype="auto"
    )
    # model_config intentionally omitted (pydantic rejects an explicit None).
    vllm_config = VllmConfig(cache_config=cache_config)

    rows = []
    with set_current_vllm_config(vllm_config):
        # initialize_model_parallel() itself calls get_current_vllm_config(),
        # so distributed init must happen INSIDE this context.
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{spec.get('port', 29597)}",
            local_rank=0,
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1)

        # GPUModelRunner.__init__ does this; fused-MoE kernels allocate their
        # scratch from it, so without it MoE layers die with
        # "WorkspaceManager not initialized".
        try:
            from vllm.v1.worker.workspace import (
                init_workspace_manager,
                is_workspace_manager_initialized,
            )

            if not is_workspace_manager_initialized():
                init_workspace_manager(device)
        except ImportError:
            pass  # older vLLM without a workspace manager

        prefix = PREFIX_TMPL.format(i=layer_idx)
        cfg = DuckConfig(cfg_d)
        # Default dtype at construction drives vLLM's attention-backend choice.
        with torch.device(device):
            old = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                layer = LayerCls(
                    config=cfg, cache_config=cache_config, quant_config=None, prefix=prefix
                )
            finally:
                torch.set_default_dtype(old)
        # Move device only — do NOT blanket-cast dtype. Construction already ran
        # under bf16 as the default dtype, and some layers deliberately keep a
        # component in fp32: Glm4MoE.gate is nn.Linear(..., dtype=torch.float32)
        # and calls it with hidden_states.to(float32); casting it to bf16 gives
        # "expected mat1 and mat2 to have the same dtype: float != BFloat16".
        layer = layer.to(device).eval()

        with torch.no_grad():  # random weights; no checkpoint exists
            for p in layer.parameters():
                p.normal_(0.0, 0.02) if p.dim() >= 2 else p.fill_(1.0)

        # The real model runner runs this pass after weight loading; skipping it
        # leaves FusedMoE without its selected kernel and MoE layers die with
        # "assert self.moe_kernel is not None". Mirror what vLLM does.
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

        for mod in layer.modules():
            qm = getattr(mod, "quant_method", None)
            if isinstance(qm, QuantizeMethodBase):
                qm.process_weights_after_loading(mod)

        attn = layer.self_attn.attn
        attn_name = attn.layer_name
        hidden = int(cfg_d["hidden_size"])
        n_kv = int(cfg_d.get("num_key_value_heads") or cfg_d["num_attention_heads"])
        head_dim = int(cfg_d.get("head_dim") or hidden // int(cfg_d["num_attention_heads"]))

        print(f"block: {name}  arch={arch}  layer={LayerCls.__name__}  idx={layer_idx}")
        print(
            f"  backend={attn.attn_backend.get_name()} impl={type(attn.impl).__name__} "
            f"hidden={hidden} kv_heads={n_kv} head_dim={head_dim} dtype=bf16 "
            f"device={torch.cuda.get_device_name(0)}"
        )

        for t in [int(x) for x in args.ctx.split(",") if x.strip()]:
            for b in [int(x) for x in args.batch.split(",") if x.strip()]:
                # Paged KV cache sized for this (B, T) point (+1 decode token).
                per_req = (t + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
                nblocks = b * per_req
                kv_shape = attn.attn_backend.get_kv_cache_shape(
                    nblocks, BLOCK_SIZE, n_kv, head_dim
                )
                kv_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
                attn.kv_cache = kv_cache  # what vllm.v1.worker.utils.bind_kv_cache does
                block_table = (
                    torch.arange(nblocks, dtype=torch.int32).view(b, per_req).contiguous()
                )

                # ---- prefill: B sequences of T tokens, causal ----
                q_lens, s_lens = [t] * b, [t] * b
                md, slots = make_metadata(q_lens, s_lens, block_table, device)
                pos = torch.cat([torch.arange(t, device=device) for _ in range(b)]).long()
                hs = (torch.randn(b * t, hidden, dtype=dtype, device=device) * 0.1)

                # NOTE: set_forward_context is entered ONCE, OUTSIDE the timed
                # region. In real vLLM it is entered once per model forward and
                # amortized over all N layers; timing it per layer call charges
                # the whole Python context-manager cost to a single block and
                # overstates decode ~5x. Only the layer call itself is timed.
                with set_forward_context(
                    md, vllm_config, num_tokens=b * t, slot_mapping={attn_name: slots}
                ), torch.inference_mode():
                    def prefill():
                        layer(positions=pos, hidden_states=hs, residual=None)

                    for _ in range(min(args.warmup, 3)):
                        prefill()
                    pf_med_us, pf_p95_us = time_gpu(prefill, max(args.prefill_iters, 1))

                # ---- decode: B sequences of 1 token against the T-token cache ----
                dq, ds = [1] * b, [t + 1] * b
                dmd, dslots = make_metadata(dq, ds, block_table, device)
                dpos = torch.full((b,), t, dtype=torch.long, device=device)
                dhs = (torch.randn(b, hidden, dtype=dtype, device=device) * 0.1)

                with set_forward_context(
                    dmd, vllm_config, num_tokens=b, slot_mapping={attn_name: dslots}
                ), torch.inference_mode():
                    def decode():
                        layer(positions=dpos, hidden_states=dhs, residual=None)

                    for _ in range(args.warmup):
                        decode()
                    graphed = False
                    if not args.no_cudagraph:
                        try:
                            s = torch.cuda.Stream()
                            s.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s):
                                for _ in range(3):
                                    decode()
                            torch.cuda.current_stream().wait_stream(s)
                            gr = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(gr):
                                decode()
                            for _ in range(args.warmup):
                                gr.replay()
                            d_med, d_p95 = time_gpu(gr.replay, args.iters)
                            graphed = True
                        except Exception as e:
                            print(f"    (cudagraph capture failed B={b} T={t}: {type(e).__name__}; eager)")
                    if not graphed:
                        d_med, d_p95 = time_gpu(decode, args.iters)

                pf_med, pf_p95 = pf_med_us / 1e3, pf_p95_us / 1e3
                tok_s = 1e6 / d_med * b
                pf_tok_s = b * t / (pf_med / 1e3)
                print(
                    f"  B={b:>2} T={t:>5}  decode median={d_med:>9.2f} us p95={d_p95:>9.2f} us "
                    f"tok/s={tok_s:>8.1f} | prefill median={pf_med:>8.2f} ms tok/s={pf_tok_s:>9.1f}"
                )
                rows.append(
                    {
                        "batch": b,
                        "ctx": t,
                        "latency_us_median": round(d_med, 2),
                        "latency_us_p95": round(d_p95, 2),
                        "tok_s": round(tok_s, 1),
                        "prefill_ms_median": round(pf_med, 3),
                        "prefill_ms_p95": round(pf_p95, 3),
                        "prefill_tok_s": round(pf_tok_s, 1),
                    }
                )
                del kv_cache
                torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "baseline": "vllm-layer",
                "cudagraph": not args.no_cudagraph,
                "block": name,
                "arch": arch,
                "layer_class": LayerCls.__name__,
                "dtype": "bf16",
                "device": torch.cuda.get_device_name(0),
                "sweep": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
