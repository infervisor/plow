"""AOT-export FlashAttention-4's SM90 forward (vLLM 0.29 vllm_flash_attn CuTe DSL) for Gemma-4
global layers without accessing a GPU: head dim 512, causal, varlen Q, paged K/V of 64 rows (=
the hd512 tile_n, so the TMA paged path), 16 query heads over KVH KV heads, symbolic KV head
stride (a head-major slot's kv_stride * 512). Config = vLLM's own SM90 hd512 choice (tile 64x64,
one stage, 384 threads, P from smem).

  export_fa4_hd512_sm90.py OUT_DIR KVH [bf16|f16]   -> OUT_DIR/fa4_hd512_h16kv<KVH>.{h,o}
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

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
H, KVH, D, PAGE = 16, int(sys.argv[2]), 512, 64
dt = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16}[sys.argv[3] if len(sys.argv) > 3 else "bf16"]
S = cute.sym_int
q = make_fake_tensor(dt, (S(), S(), S()), (H * D, D, 1), assumed_align=16)
o = make_fake_tensor(dt, (S(), S(), S()), (H * D, D, 1), assumed_align=16)
hs = cute.sym_int64(divisibility=8)
k = make_fake_tensor(dt, (S(), S(), S(), S()), (PAGE * D, D, hs, 1), assumed_align=16)
v = make_fake_tensor(dt, (S(), S(), S(), S()), (PAGE * D, D, hs, 1), assumed_align=16)
cuq = make_fake_tensor(cutlass.Int32, (S(),), (1,), assumed_align=4)
usedk = make_fake_tensor(cutlass.Int32, (S(),), (1,), assumed_align=4)
pt = make_fake_tensor(cutlass.Int32, (S(), S()), (S(), 1), assumed_align=4)
fa = FlashAttentionForwardSm90(dt, D, D, H // KVH, is_causal=True, is_local=False,
                               is_split_kv=False, pack_gqa=False, tile_m=64, tile_n=PAGE,
                               num_stages=1, num_threads=384, Q_in_regs=False,
                               intra_wg_overlap=True, mma_pv_is_rs=False, paged_kv_non_tma=False)


@cute.jit
def entry(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor, cuq: cute.Tensor,
          usedk: cute.Tensor, pt: cute.Tensor, scale: cutlass.Float32,
          stream: cuda_driver.CUstream):
    fa(q, k, v, o, None, scale, cuq, None, None, usedk, None, pt, None, None, None,
       stream=stream)


compiled = cute.compile(entry, q, k, v, o, cuq, usedk, pt, cutlass.Float32(1.0),
                        make_fake_stream(), options="--gpu-arch=sm_90a")
name = f"fa4_hd512_h{H}kv{KVH}"
compiled.export_to_c(str(out), name, "plow_" + name)
print("exported", sorted(p.name for p in out.glob(name + ".*")), flush=True)
