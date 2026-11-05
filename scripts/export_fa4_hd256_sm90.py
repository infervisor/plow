"""AOT-export FlashAttention-4's SM90 forward (vLLM 0.28 vllm_flash_attn CuTe DSL) for causal
full-attention layers of head dim 256 without accessing a GPU: bf16, varlen Q, paged K/V of 64
rows over a row-interleaved cache ([slot][ctx][kv head][256]: a page is 64 rows of every KV head,
row stride kvh*256), H query heads over KVH KV heads with GQA packing. Config = vLLM's own SM90
hd256 causal choice (tile 128x80, two stages, 384 threads, P in registers, intra-WG overlap; the
64-row pages are read through the non-TMA paged path, which loads no row at or past seqused_k).

  export_fa4_hd256_sm90.py OUT_DIR H KVH          -> fa4_hd256_h<H>kv<KVH>{,_split}.{h,o}
  export_fa4_hd256_sm90.py OUT_DIR H combine      -> fa4_hd256_h<H>_combine.{h,o}

`_split` writes f32 partials [splits][total_q][H][256] and LSE [splits][H][total_q]; the combine
entry (vLLM's combine at hd256: tile_m 8, k block 128, up to 128 splits) folds them into bf16.
"""
import pathlib
import sys
import torch


def forbidden(*args, **kwargs):
    raise RuntimeError("GPU initialization forbidden during AOT export")


torch.cuda._lazy_init = forbidden
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)  # quack's import-time probe
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda_driver
from cutlass.cute.runtime import make_fake_tensor, make_fake_stream
from vllm.vllm_flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from vllm.vllm_flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
H, D, PAGE = int(sys.argv[2]), 256, 64
S = cute.sym_int
f32, i32, bf = cutlass.Float32, cutlass.Int32, cutlass.BFloat16


def export(compiled, name):
    compiled.export_to_c(str(out), name, "plow_" + name)
    print("exported", sorted(p.name for p in out.glob(name + ".*")), flush=True)


if sys.argv[3] == "combine":
    combine = FlashAttentionForwardCombine(dtype=bf, dtype_partial=f32, head_dim=D, tile_m=8,
                                           k_block_size=128, log_max_splits=7,
                                           output_quant_key=None)
    tq = cute.sym_int64(divisibility=1)
    o_part = make_fake_tensor(f32, (S(), S(), S(), S()),
                              (cute.sym_int64(divisibility=4), H * D, D, 1), assumed_align=16)
    lse_part = make_fake_tensor(f32, (S(), S(), S()), (cute.sym_int64(divisibility=1), 1, tq),
                                assumed_align=4)
    o = make_fake_tensor(bf, (S(), S(), S()), (H * D, D, 1), assumed_align=16)
    cuq = make_fake_tensor(i32, (S(),), (1,), assumed_align=4)

    @cute.jit
    def entry(o_part: cute.Tensor, lse_part: cute.Tensor, o: cute.Tensor, cuq: cute.Tensor,
              stream: cuda_driver.CUstream):
        combine(o_part, lse_part, o, None, cuq, None, None, None, None, None, stream=stream)

    export(cute.compile(entry, o_part, lse_part, o, cuq, make_fake_stream(),
                        options="--gpu-arch=sm_90a"), f"fa4_hd256_h{H}_combine")
    sys.exit(0)

KVH = int(sys.argv[3])
q = make_fake_tensor(bf, (S(), S(), S()), (H * D, D, 1), assumed_align=16)
k = make_fake_tensor(bf, (S(), S(), S(), S()), (PAGE * KVH * D, KVH * D, D, 1), assumed_align=16)
v = make_fake_tensor(bf, (S(), S(), S(), S()), (PAGE * KVH * D, KVH * D, D, 1), assumed_align=16)
cuq = make_fake_tensor(i32, (S(),), (1,), assumed_align=4)
usedk = make_fake_tensor(i32, (S(),), (1,), assumed_align=4)
pt = make_fake_tensor(i32, (S(), S()), (S(), 1), assumed_align=4)
for split in (False, True):
    fa = FlashAttentionForwardSm90(bf, D, D, H // KVH, is_causal=True, is_local=False,
                                   is_split_kv=split, pack_gqa=H > KVH, tile_m=128, tile_n=80,
                                   num_stages=2, num_threads=384, Q_in_regs=False,
                                   intra_wg_overlap=True, mma_pv_is_rs=True,
                                   paged_kv_non_tma=True)
    if split:
        tq = cute.sym_int64(divisibility=1)
        o = make_fake_tensor(f32, (S(), S(), S(), S()),
                             (cute.sym_int64(divisibility=4), H * D, D, 1), assumed_align=16)
        lse = make_fake_tensor(f32, (S(), S(), S()), (cute.sym_int64(divisibility=1), tq, 1),
                               assumed_align=4)
    else:
        o = make_fake_tensor(bf, (S(), S(), S()), (H * D, D, 1), assumed_align=16)
        lse = None

    @cute.jit
    def entry(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor, lse: cute.Tensor,
              cuq: cute.Tensor, usedk: cute.Tensor, pt: cute.Tensor, scale: cutlass.Float32,
              stream: cuda_driver.CUstream):
        fa(q, k, v, o, lse, scale, cuq, None, None, usedk, None, pt, None, None, None,
           stream=stream)

    export(cute.compile(entry, q, k, v, o, lse, cuq, usedk, pt, cutlass.Float32(1.0),
                        make_fake_stream(), options="--gpu-arch=sm_90a"),
           f"fa4_hd256_h{H}kv{KVH}" + ("_split" if split else ""))
