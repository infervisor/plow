"""Compile the installed FlashInfer SM90 GDN kernel without accessing a GPU."""
import pathlib
import sys
import torch

def forbidden(*args, **kwargs):
    raise RuntimeError("GPU initialization/allocation forbidden during AOT export")

torch.cuda._lazy_init = forbidden
import cutlass
import cuda.bindings.driver as cuda_driver
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor, make_fake_stream
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm90 import _FullyFusedDeltaRuleSm90

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
length = cute.sym_int64(symbol="tokens")
state_size = cute.sym_int64(symbol="state_elements")
gate_size = cute.sym_int64(symbol="gate_elements")
offset_size = cute.sym_int64(symbol="sequence_offsets")

def tensor(dtype, shape, stride, align=16):
    return make_fake_tensor(dtype, shape, stride, assumed_align=align)

q = tensor(cutlass.BFloat16, (length, 128, 16), (2048, 1, 128))
k = tensor(cutlass.BFloat16, (128, length, 16), (1, 2048, 128))
v = tensor(cutlass.BFloat16, (128, length, 48), (1, 6144, 128))
o = tensor(cutlass.BFloat16, (128, length, 48), (1, 6144, 128))
alpha = tensor(cutlass.Float32, (gate_size,), (1,))
beta = tensor(cutlass.Float32, (gate_size,), (1,))
state = tensor(cutlass.Float32, (state_size,), (1,))
initial = tensor(cutlass.Float32, (state_size,), (1,))
maps = tensor(cutlass.Uint8, (132 * 128,), (1,), 128)
offsets = tensor(cutlass.Int64, (offset_size,), (1,), 8)
kernel = _FullyFusedDeltaRuleSm90(True, True, True, False, cutlass.BFloat16)
args = (q, k, v, o, alpha, beta, state, initial, None, None, maps, offsets,
        cutlass.Float32(128 ** -0.5), cutlass.Int32(16), cutlass.Int32(16),
        cutlass.Int32(48), cutlass.Int32(48), cutlass.Int32(1), cutlass.Int32(1),
        cutlass.Int32(0), 48, make_fake_stream())
@cute.jit
def entry(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor,
          alpha: cute.Tensor, beta: cute.Tensor, state: cute.Tensor,
          initial: cute.Tensor, maps: cute.Tensor, offsets: cute.Tensor,
          scale: cutlass.Float32, stream: cuda_driver.CUstream):
    kernel(q, k, v, o, alpha, beta, state, initial, None, None, maps, offsets,
           scale, cutlass.Int32(16), cutlass.Int32(16), cutlass.Int32(48),
           cutlass.Int32(48), cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(0),
           48, stream)

args = (q, k, v, o, alpha, beta, state, initial, maps, offsets,
        cutlass.Float32(128 ** -0.5), make_fake_stream())
print("Compiling SM90a with fake tensors; no kernel invocation", flush=True)
compiled = cute.compile(entry, *args, options="--gpu-arch=sm_90a")
compiled.export_to_c(str(out), "gdn_sm90", "plow_gdn_sm90")
print("Exported", sorted(p.name for p in out.glob("gdn_sm90.*")), flush=True)
