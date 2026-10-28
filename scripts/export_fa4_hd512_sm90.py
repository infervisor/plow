"""AOT-export FlashAttention-4's SM90 forward (vLLM 0.29 vllm_flash_attn CuTe DSL) for Gemma-4
global layers without accessing a GPU: head dim 512, bf16, causal, varlen Q, paged K/V of 64 rows
(= the hd512 tile_n, so the TMA paged path), 16 query heads over KVH KV heads (GQA packing at 2
KV heads as vLLM runs it; the 16-way pack crashes the DSL compiler), symbolic KV head stride (the
page pool's per-head extent). Config = vLLM's own SM90 hd512 choice (tile 64x64, one stage, 384
threads, P from smem).

  export_fa4_hd512_sm90.py OUT_DIR KVH       -> fa4_hd512_h16kv<KVH>{,_split}.{h,o}
  export_fa4_hd512_sm90.py OUT_DIR combine   -> fa4_hd512_combine.{h,o}

`_split` writes f32 partials [splits][total_q][16][512] and LSE [splits][16][total_q] (split
count = the partial tensor's first extent); the combine entry (vLLM's combine kernel at hd512:
tile_m 8, k block 128, up to 128 splits) folds them into the bf16 output.
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
H, D, PAGE = 16, 512, 64
S = cute.sym_int
bf, f32, i32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32


def export(compiled, name):
    compiled.export_to_c(str(out), name, "plow_" + name)
    print("exported", sorted(p.name for p in out.glob(name + ".*")), flush=True)


if sys.argv[2] == "combine":
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
                        options="--gpu-arch=sm_90a"), "fa4_hd512_combine")
    sys.exit(0)

KVH = int(sys.argv[2])
q = make_fake_tensor(bf, (S(), S(), S()), (H * D, D, 1), assumed_align=16)
hs = cute.sym_int64(divisibility=8)
k = make_fake_tensor(bf, (S(), S(), S(), S()), (PAGE * D, D, hs, 1), assumed_align=16)
v = make_fake_tensor(bf, (S(), S(), S(), S()), (PAGE * D, D, hs, 1), assumed_align=16)
cuq = make_fake_tensor(i32, (S(),), (1,), assumed_align=4)
usedk = make_fake_tensor(i32, (S(),), (1,), assumed_align=4)
pt = make_fake_tensor(i32, (S(), S()), (S(), 1), assumed_align=4)
for split in (False, True):
    fa = FlashAttentionForwardSm90(bf, D, D, H // KVH, is_causal=True, is_local=False,
                                   is_split_kv=split, pack_gqa=KVH == 2, tile_m=64, tile_n=PAGE,
                                   num_stages=1, num_threads=384, Q_in_regs=False,
                                   intra_wg_overlap=True, mma_pv_is_rs=False,
                                   paged_kv_non_tma=False)
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
           f"fa4_hd512_h16kv{KVH}" + ("_split" if split else ""))
