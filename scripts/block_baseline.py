#!/usr/bin/env python3
"""scripts/block_baseline.py — PyTorch single-block BASELINE for the plow block harness.

Companion to `crates/plowrt/examples/block_run.rs` (`block_run <asset> bench`).
`block_run` measures ONE compiled transformer block on plow's runtime and writes
`sweep.json`.  This script measures the SAME single block on the SAME (B,T) grid
through PyTorch's tuned path and writes a `sweep.json` with the identical row
schema, so the two files diff directly.

WHY EAGER PYTORCH IS THE WRONG BASELINE (and how this fixes it)
--------------------------------------------------------------
At decode (M=1) a single block is ~15-20 tiny ops.  In eager mode each is a
separate kernel launch (~5-8 us), so ~100-150 us/step is pure LAUNCH OVERHEAD,
not compute.  vLLM erases this with **CUDA graphs** (`cudagraph_mode
FULL_AND_PIECEWISE` in the repo's vLLM baselines); plow erases it with its fused
packet.  An eager baseline therefore overstates block latency 2-5x and makes
plow look artificially good — a dishonest floor.

So this harness captures the decode step into a **CUDA graph** by default (the
same technique vLLM uses), which makes the measured number compute-bound and a
fair tuned-path floor.  `--no-cudagraph` reveals the eager number for contrast.

MAKING IT "RIGHT" AGAINST vLLM
------------------------------
A single block cannot be *served* by vLLM in isolation, so the vLLM-grounded
per-block floor is derived from the full-model decode latency:

    per_block_us  ~=  (TPOT_ms - fixed_overhead_ms) * 1000 / num_layers

where fixed_overhead is embed + final norm + lm_head + sampling (context- and
layer-independent).  Pass `--vllm-tpot-ms` and `--layers` and this prints the
harness per-block number next to that implied vLLM floor for cross-validation.
If harness*layers ~= measured TPOT, the isolated baseline is trustworthy.

CAVEATS THE NUMBERS DEPEND ON (read before comparing)
-----------------------------------------------------
* GPU must match.  The repo's vLLM baselines are on RTX PRO 6000 Blackwell
  (sm_120); this harness reports on whatever CUDA device it runs.  Cross-GPU
  block numbers are not comparable — re-measure vLLM on the same box.
* MoE geometry.  The plow descriptor `moe_gemma4_26b_a4b.json` is a linear-chain
  APPROXIMATION (8 experts / top-2, hidden 2560).  The real Gemma-4-26B-A4B that
  vLLM serves is 128 experts / top-8, hidden 2816, FlashInfer CUTLASS fused MoE.
  This harness runs `top_k` full expert FFNs / token (matching the descriptor's
  per-token matmul volume), which is a faithful FLOP proxy but NOT vLLM's fused
  MoE kernel.  To compare against vLLM's MoE, run the real geometry.

WHY RANDOM WEIGHTS ARE VALID
----------------------------
`block_run.rs`: the isolated block has no upstream, `act.x` is never refreshed,
"the tokens are meaningless, but the per-step KERNEL time … is data-independent,
which is the point."  A latency baseline needs shapes + kernel path, not real
weights — no checkpoint / HF download, runs anywhere a CUDA torch does.

Usage:
  python3 scripts/block_baseline.py crates/plowc/examples/transformer_block_gemma4_12b.json \
      --batch 1,2,4,8 --ctx 128,512,1024,4096 --dtype bf16 \
      --layers 48 --vllm-tpot-ms 19.78 --out /dev/shm/block-baseline/gemma12b.json

Prefer `scripts/block_baseline.sh` (wraps this in `gpulease` for multi-agent GPU
serialization).  No numpy dependency — torch + json only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Descriptor parsing: derive block geometry from the plow-native ops list.
# --------------------------------------------------------------------------- #
class Geometry:
    """Block dims recovered from a plowc example descriptor.

    Assumes the Gemma-family conventions the descriptors encode: attention_k_eq_v
    (one shared K/V projection), attn scale 1.0, GeGLU MLP with a single fused up
    projection to 2*intermediate.  MoE is detected by a router GEMM (tiny n)
    followed by repeated gemm-act-gemm expert groups.
    """

    def __init__(self, desc: dict):
        self.name: str = desc.get("name", "block")
        self.hidden: int = int(desc["hidden"])
        ops = desc["ops"]

        flash = next((o for o in ops if o.get("op") == "flash"), None)
        if flash is None:
            raise ValueError(f"{self.name}: no flash op — attention geometry unknown")
        self.q_heads: int = int(flash["heads"])
        self.head_dim: int = int(flash["head_dim"])

        gemms = [o for o in ops if o.get("op") == "gemm"]
        if not gemms:
            raise ValueError(f"{self.name}: no gemm ops")
        qkv_n = int(gemms[0]["n"])  # fused QKV: q_heads*hd + kv_heads*hd (shared K/V)
        q_dim = self.q_heads * self.head_dim
        kv_extra = qkv_n - q_dim
        if kv_extra <= 0 or kv_extra % self.head_dim != 0:
            raise ValueError(
                f"{self.name}: fused QKV n={qkv_n} inconsistent with "
                f"q_heads*head_dim={q_dim} (expected k_eq_v shared K/V)"
            )
        self.kv_heads: int = kv_extra // self.head_dim

        fi = ops.index(flash)
        post = ops[fi + 1 :]
        mlp_gemms = [o for o in post if o.get("op") == "gemm"][1:]  # drop o_proj

        self.is_moe = False
        self.num_experts = 0
        self.top_k = 0
        if mlp_gemms and int(mlp_gemms[0]["n"]) <= 64:
            self.is_moe = True
            self.num_experts = int(mlp_gemms[0]["n"])
            expert_gemms = mlp_gemms[1:]
            self.top_k = len(expert_gemms) // 2  # each expert = (up, down)
            up_n = int(expert_gemms[0]["n"])
        else:
            up_n = int(mlp_gemms[0]["n"])
        self.intermediate: int = up_n // 2  # fused up -> 2*intermediate (GeGLU)

    def summary(self) -> str:
        base = (
            f"{self.name}: hidden={self.hidden} q_heads={self.q_heads} "
            f"kv_heads={self.kv_heads} head_dim={self.head_dim} inter={self.intermediate}"
        )
        if self.is_moe:
            base += f" MoE(experts={self.num_experts} top_k={self.top_k})"
        return base


# --------------------------------------------------------------------------- #
# The block module.  One decoder layer, Gemma conventions, STATIC shapes so the
# decode step is CUDA-graph capturable (the vLLM decode technique).
# --------------------------------------------------------------------------- #
def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(dt)


class GeGLU(nn.Module):
    """Gemma GeGLU FFN: fused up -> [gate|up], gelu_tanh(gate)*up, down."""

    def __init__(self, hidden: int, inter: int, dtype, device):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inter, bias=False, dtype=dtype, device=device)
        self.down = nn.Linear(inter, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.up(x).chunk(2, dim=-1)
        return self.down(F.gelu(gate, approximate="tanh") * up)


class Block(nn.Module):
    def __init__(self, g: Geometry, dtype, device):
        super().__init__()
        self.g = g
        H, hd = g.hidden, g.head_dim
        self.q_dim = g.q_heads * hd
        self.kv_dim = g.kv_heads * hd
        self.scale = 1.0  # Gemma: q_norm absorbs 1/sqrt(head_dim)

        self.in_norm_w = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.post_norm_w = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.qkv = nn.Linear(H, self.q_dim + self.kv_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(self.q_dim, H, bias=False, dtype=dtype, device=device)
        self.q_norm_w = nn.Parameter(torch.ones(hd, dtype=dtype, device=device))
        self.k_norm_w = nn.Parameter(torch.ones(hd, dtype=dtype, device=device))

        if g.is_moe:
            self.router = nn.Linear(H, g.num_experts, bias=False, dtype=dtype, device=device)
            # Only top_k experts run per token; a fixed top_k set gives the same
            # per-token GEMM volume with static, graph-capturable shapes.
            self.experts = nn.ModuleList(
                [GeGLU(H, g.intermediate, dtype, device) for _ in range(g.top_k)]
            )
        else:
            self.mlp = GeGLU(H, g.intermediate, dtype, device)

    def _project_qkv(self, x: torch.Tensor):
        B, S, _ = x.shape
        g = self.g
        q, kv = self.qkv(x).split([self.q_dim, self.kv_dim], dim=-1)
        q = _rmsnorm(q.view(B, S, g.q_heads, g.head_dim), self.q_norm_w)
        kv = kv.view(B, S, g.kv_heads, g.head_dim)
        k = _rmsnorm(kv, self.k_norm_w)  # qk_norm; K and V both from `kv` (k_eq_v)
        return q, k, kv  # v == raw kv

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        g = self.g
        if not g.is_moe:
            return self.mlp(x)
        # Router GEMM (volume) + top_k fixed expert FFNs, summed. Static shapes.
        _ = self.router(x)
        out = self.experts[0](x)
        for e in range(1, g.top_k):
            out = out + self.experts[e](x)
        return out

    @torch.inference_mode()
    def prefill(self, x: torch.Tensor):
        """x:[B,T,H] -> (last hidden [B,1,H], kv cache k,v [B,Hkv,T,hd])."""
        h = _rmsnorm(x, self.in_norm_w)
        q, k, v = self._project_qkv(h)
        qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qh, kh, vh, scale=self.scale, is_causal=True, enable_gqa=True)
        attn = self.o(o.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.q_dim))
        x = x + _rmsnorm(attn, self.post_norm_w)
        ff = self._ffn(_rmsnorm(x, self.in_norm_w))
        x = x + _rmsnorm(ff, self.post_norm_w)
        return x[:, -1:, :].contiguous(), kh.contiguous(), vh.contiguous()

    def configure(self, b: int, t: int) -> None:  # torch backend: nothing to set up
        pass

    @torch.inference_mode()
    def decode_step(self, x1: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
        """One decode token. x1:[B,1,H]; attends over the FIXED [B,Hkv,T,hd] cache.

        Cache is not grown (numerics are meaningless; the +iters length delta is
        immaterial and, crucially, keeping T fixed makes shapes static so the
        step is CUDA-graph capturable). Per-step GEMM+attention volume — the
        sweep metric — is exactly what plow's block_run times.
        """
        h = _rmsnorm(x1, self.in_norm_w)
        q, k, v = self._project_qkv(h)  # S=1
        qh = q.transpose(1, 2)  # [B,Hq,1,hd]
        o = F.scaled_dot_product_attention(qh, kc, vc, scale=self.scale, is_causal=False, enable_gqa=True)
        attn = self.o(o.transpose(1, 2).reshape(x1.shape[0], 1, self.q_dim))
        x1 = x1 + _rmsnorm(attn, self.post_norm_w)
        ff = self._ffn(_rmsnorm(x1, self.in_norm_w))
        return x1 + _rmsnorm(ff, self.post_norm_w)


# --------------------------------------------------------------------------- #
# vLLM backend: the SAME block through vLLM's shipped kernels (fused MoE,
# flash-attn varlen decode, GeGLU act) instead of the server. This is what the
# user asked for — vLLM's tuned kernels, no HTTP/serving overhead. Attention KV
# cache is laid out for flash_attn_varlen_func ([total_tokens, Hkv, hd]); the
# GEMMs are plain cuBLAS (same as vLLM's UnquantizedLinear). Guarded import so
# the torch backend runs without vLLM installed.
# --------------------------------------------------------------------------- #
_VLLM_ERR = None
try:
    from vllm.model_executor.layers.fused_moe import fused_experts as _v_fused_experts
    from vllm.model_executor.layers.fused_moe import fused_topk as _v_fused_topk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation as _VMoEAct
    from vllm.vllm_flash_attn import flash_attn_varlen_func as _v_flash
except Exception as _e:  # pragma: no cover - only when vLLM absent
    _VLLM_ERR = _e


def _geglu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Dense GeGLU activation: gelu_tanh(gate) * up over a fused [.., 2I] tensor."""
    gate, up = x.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


class VllmBlock(nn.Module):
    def __init__(self, g: Geometry, dtype, device):
        super().__init__()
        if _VLLM_ERR is not None:
            raise RuntimeError(
                f"--backend vllm needs vLLM importable (use the vLLM venv): {_VLLM_ERR}"
            )
        self.g = g
        H, hd = g.hidden, g.head_dim
        self.q_dim = g.q_heads * hd
        self.kv_dim = g.kv_heads * hd
        self.scale = 1.0

        self.in_norm_w = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.post_norm_w = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.q_norm_w = nn.Parameter(torch.ones(hd, dtype=dtype, device=device))
        self.k_norm_w = nn.Parameter(torch.ones(hd, dtype=dtype, device=device))
        self.qkv = nn.Linear(H, self.q_dim + self.kv_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(self.q_dim, H, bias=False, dtype=dtype, device=device)

        if g.is_moe:
            self.router = nn.Linear(H, g.num_experts, bias=False, dtype=dtype, device=device)
            E, I = g.num_experts, g.intermediate
            # vLLM fused_experts layout: w1=[E,2I,H] (gate|up), w2=[E,H,I] (down).
            self.w1 = nn.Parameter(torch.randn(E, 2 * I, H, dtype=dtype, device=device) * 0.02)
            self.w2 = nn.Parameter(torch.randn(E, H, I, dtype=dtype, device=device) * 0.02)
        else:
            self.up = nn.Linear(H, 2 * g.intermediate, bias=False, dtype=dtype, device=device)
            self.down = nn.Linear(g.intermediate, H, bias=False, dtype=dtype, device=device)

        # Per-point flash varlen metadata (set by configure()).
        self.register_buffer("cu_q", torch.zeros(1, dtype=torch.int32, device=device))
        self.register_buffer("cu_k", torch.zeros(1, dtype=torch.int32, device=device))
        self._b = 0
        self._t = 0

    def configure(self, b: int, t: int) -> None:
        dev = self.in_norm_w.device
        self._b, self._t = b, t
        self.cu_q = torch.arange(0, b + 1, dtype=torch.int32, device=dev)
        self.cu_k = torch.arange(0, (b + 1) * t, t, dtype=torch.int32, device=dev)

    def _project_qkv(self, x):
        # x:[B,S,H] -> q:[B*S,Hq,hd], k/v:[B*S,Hkv,hd]
        B, S, _ = x.shape
        g = self.g
        q, kv = self.qkv(x).split([self.q_dim, self.kv_dim], dim=-1)
        q = _rmsnorm(q.reshape(-1, g.q_heads, g.head_dim), self.q_norm_w)
        kv = kv.reshape(-1, g.kv_heads, g.head_dim)
        k = _rmsnorm(kv, self.k_norm_w)
        return q, k, kv  # v == raw kv

    def _ffn(self, x1: torch.Tensor) -> torch.Tensor:
        g = self.g
        x2 = x1.reshape(-1, g.hidden)  # [B,H]
        if not g.is_moe:
            return self.down(_geglu_tanh(self.up(x2))).reshape(x1.shape)
        logits = self.router(x2)
        tw, tid = _v_fused_topk(x2, logits, g.top_k, renormalize=True)[:2]
        out = _v_fused_experts(
            x2, self.w1, self.w2, tw, tid,
            activation=_VMoEAct.GELU_TANH, global_num_experts=g.num_experts,
        )
        return out.reshape(x1.shape)

    @torch.inference_mode()
    def prefill(self, x: torch.Tensor):
        """FULL block forward over [B,T,H] — the prefill phase, same work as the
        torch backend's prefill (causal attention + o_proj + FFN), so the two are
        comparable. Returns (last hidden [B,1,H], kc/vc [B*T,Hkv,hd] for varlen).
        """
        B, T, _ = x.shape
        g = self.g
        h = _rmsnorm(x, self.in_norm_w)
        q, k, v = self._project_qkv(h)  # [B*T, H*, hd]
        kc = k.reshape(B * T, g.kv_heads, g.head_dim).contiguous()
        vc = v.reshape(B * T, g.kv_heads, g.head_dim).contiguous()
        # Causal varlen attention over the B sequences of length T.
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=x.device)
        o = _v_flash(
            q.reshape(B * T, g.q_heads, g.head_dim), kc, vc,
            max_seqlen_q=T, cu_seqlens_q=cu, max_seqlen_k=T, cu_seqlens_k=cu,
            causal=True, softmax_scale=self.scale,
        )
        o = o if torch.is_tensor(o) else o[0]
        attn = self.o(o.reshape(B, T, self.q_dim))
        x = x + _rmsnorm(attn, self.post_norm_w)
        ff = self._ffn(_rmsnorm(x, self.in_norm_w))
        x = x + _rmsnorm(ff, self.post_norm_w)
        return x[:, -1:, :].contiguous(), kc, vc

    @torch.inference_mode()
    def decode_step(self, x1: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
        g = self.g
        h = _rmsnorm(x1, self.in_norm_w)
        q, _, _ = self._project_qkv(h)  # q:[B,Hq,hd]
        o = _v_flash(
            q, kc, vc, max_seqlen_q=1, cu_seqlens_q=self.cu_q,
            max_seqlen_k=self._t, cu_seqlens_k=self.cu_k, causal=False, softmax_scale=self.scale,
        )
        o = o if torch.is_tensor(o) else o[0]  # [B,Hq,hd]
        attn = self.o(o.reshape(self._b, 1, self.q_dim))
        x1 = x1 + _rmsnorm(attn, self.post_norm_w)
        ff = self._ffn(_rmsnorm(x1, self.in_norm_w))
        return x1 + _rmsnorm(ff, self.post_norm_w)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def parse_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _time_steps(step, iters):
    """Time `iters` calls, each with CUDA events (pure GPU time, us)."""
    us = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        step()
        e1.record()
        torch.cuda.synchronize()
        us.append(e0.elapsed_time(e1) * 1e3)
    return us


def bench_point(block, g, b, t, dtype, dev, iters, warmup, use_graph, pf_iters=10):
    """Return (decode_median_us, decode_p95_us, prefill_stats, graphed) for one (B,T).

    BOTH phases are measured on the SAME full-block forward on both backends:
      prefill — one [B,T,H] pass (causal attention over T, o_proj, FFN)
      decode  — one [B,1,H] step against the T-row KV cache
    prefill is timed with the same warmup/median/p95 treatment as decode so the
    two backends' prefill numbers are comparable.
    """
    block.configure(b, t)
    x = torch.randn(b, t, g.hidden, dtype=dtype, device=dev) * 0.1

    # --- prefill phase (timed properly, not a single sample) ---
    for _ in range(min(warmup, 3)):
        block.prefill(x)
    pf_us = _time_steps(lambda: block.prefill(x), max(pf_iters, 1))
    pf_us.sort()
    pf_med = pf_us[len(pf_us) // 2]
    pf_p95 = pf_us[min(int(len(pf_us) * 0.95), len(pf_us) - 1)]
    prefill_stats = (pf_med / 1e3, pf_p95 / 1e3, b * t / (pf_med / 1e6))  # ms, ms, tok/s

    x1, kc, vc = block.prefill(x)
    torch.cuda.synchronize()

    xbuf = x1.clone()  # static input buffer for the decode step
    graphed = False

    if use_graph:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    block.decode_step(xbuf, kc, vc)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                block.decode_step(xbuf, kc, vc)
            for _ in range(warmup):
                graph.replay()
            us = _time_steps(graph.replay, iters)
            graphed = True
        except Exception as e:
            # Some kernels (e.g. Triton fused-MoE) don't capture cleanly; fall
            # back to eager for this point and say so.
            print(f"    (cudagraph capture failed at B={b} T={t}: {type(e).__name__}; eager) ")
            use_graph = False

    if not graphed:
        def step():
            return block.decode_step(xbuf, kc, vc)

        for _ in range(warmup):
            step()
        us = _time_steps(step, iters)

    us.sort()
    median = us[len(us) // 2]
    p95 = us[min(int(len(us) * 0.95), len(us) - 1)]
    return median, p95, prefill_stats, graphed


def main() -> int:
    ap = argparse.ArgumentParser(description="PyTorch single-block baseline")
    ap.add_argument("descriptor", help="plowc example block descriptor .json")
    ap.add_argument("--batch", default="1,2,4,8")
    ap.add_argument("--ctx", default="128,512,1024,4096")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prefill-iters", type=int, default=10,
                    help="timed prefill passes per (B,T) point")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--backend", default="torch", choices=["torch", "vllm"],
                    help="torch (SDPA+cuBLAS) or vllm (fused MoE + flash-attn + GeGLU kernels)")
    ap.add_argument("--no-cudagraph", action="store_true", help="eager decode (exposes launch overhead)")
    ap.add_argument("--layers", type=int, default=0, help="full-model layer count (vLLM anchor)")
    ap.add_argument("--vllm-tpot-ms", type=float, default=0.0, help="measured full-model TPOT ms/token (vLLM anchor)")
    ap.add_argument("--vllm-overhead-ms", type=float, default=0.0, help="fixed embed+lm_head+sampling ms to subtract")
    ap.add_argument("--out", default="/dev/shm/block-baseline/sweep.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = DTYPES[args.dtype]
    dev = "cuda"
    use_graph = not args.no_cudagraph

    g = Geometry(json.loads(Path(args.descriptor).read_text()))
    block = (VllmBlock if args.backend == "vllm" else Block)(g, dtype, dev).eval()
    dev_name = torch.cuda.get_device_name(0)
    print(f"baseline: {g.summary()}")
    print(f"  backend={args.backend} dtype={args.dtype} device={dev_name} "
          f"cudagraph={use_graph} iters={args.iters} warmup={args.warmup}")

    # vLLM-anchored per-block floor, if provided.
    anchor_us = 0.0
    if args.layers and args.vllm_tpot_ms:
        anchor_us = (args.vllm_tpot_ms - args.vllm_overhead_ms) * 1e3 / args.layers
        print(
            f"  vLLM anchor: TPOT {args.vllm_tpot_ms} ms - overhead {args.vllm_overhead_ms} ms "
            f"over {args.layers} layers => implied per-block floor ~= {anchor_us:.2f} us"
        )

    # Absorb one-time init (lazy CUDA context, cuBLAS handle, Triton autotune)
    # so the first sweep point's prefill timing is not inflated.
    with torch.inference_mode():
        block.configure(1, 64)
        wx1, wkc, wvc = block.prefill(torch.randn(1, 64, g.hidden, dtype=dtype, device=dev) * 0.1)
        block.decode_step(wx1, wkc, wvc)
    torch.cuda.synchronize()

    rows = []
    for t in parse_list(args.ctx):
        for b in parse_list(args.batch):
            median, p95, pf, graphed = bench_point(
                block, g, b, t, dtype, dev, args.iters, args.warmup, use_graph,
                args.prefill_iters,
            )
            pf_med, pf_p95, pf_tok_s = pf
            tok_s = 1e6 / median * b
            anchor_str = ""
            if anchor_us:
                anchor_str = f"  vs vLLM/block {anchor_us:.1f}us ({median / anchor_us:.2f}x)"
            gflag = "" if graphed or not use_graph else " [eager-fallback]"
            print(
                f"  B={b:>2} T={t:>5}  decode median={median:>9.2f} us p95={p95:>9.2f} us "
                f"tok/s={tok_s:>8.1f} | prefill median={pf_med:>8.2f} ms "
                f"tok/s={pf_tok_s:>9.1f}{anchor_str}{gflag}"
            )
            rows.append(
                {
                    "batch": b,
                    "ctx": t,
                    "latency_us_median": round(median, 2),
                    "latency_us_p95": round(p95, 2),
                    "tok_s": round(tok_s, 1),
                    "prefill_ms_median": round(pf_med, 3),
                    "prefill_ms_p95": round(pf_p95, 3),
                    "prefill_tok_s": round(pf_tok_s, 1),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": args.backend,
        "block": g.name,
        "dtype": args.dtype,
        "device": dev_name,
        "cudagraph": use_graph,
        "geometry": {
            "hidden": g.hidden,
            "q_heads": g.q_heads,
            "kv_heads": g.kv_heads,
            "head_dim": g.head_dim,
            "intermediate": g.intermediate,
            "is_moe": g.is_moe,
            "num_experts": g.num_experts,
            "top_k": g.top_k,
        },
        "sweep": rows,
    }
    if anchor_us:
        payload["vllm_anchor"] = {
            "tpot_ms": args.vllm_tpot_ms,
            "overhead_ms": args.vllm_overhead_ms,
            "layers": args.layers,
            "implied_per_block_us": round(anchor_us, 2),
        }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
