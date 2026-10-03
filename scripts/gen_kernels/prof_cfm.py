"""Phase clocks of a PROF build of cfm_attn.cu (sp_attention_tc64m): prof_cfm.py LIB ITEMS T WHICH."""
import sys, os, ctypes, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_cfm import inputs, H, HW, PRE, dev

lib, B, Tq, which = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
so = ctypes.CDLL(lib)
so.cfm_make.restype = ctypes.c_void_p
so.cfm_make.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 3
so.cfm_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
qkv, prefix, pidx, klen = inputs(B, Tq)
out = torch.empty(B, Tq, H * HW, device=dev)
h = so.cfm_make(out.data_ptr(), qkv.data_ptr(), prefix.data_ptr(), pidx.data_ptr(), klen.data_ptr(), B, Tq, PRE)
buf = (ctypes.c_ulonglong * 16)()
for _ in range(3):
    so.cfm_run(h, which, 132, None)
torch.cuda.synchronize()
so.cfm_prof(buf, 1)
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
so.cfm_run(h, which, 132, None)
e1.record()
torch.cuda.synchronize()
so.cfm_prof(buf, 1)
us = e0.elapsed_time(e1) * 1000
names = ["item+loop", "load issue", "wait full", "QK", "softmax", "PV", "store", "empty wait", "store pfx", "store own"]
tot = sum(buf[i] for i in range(10))
warps = 132 * 8
print(f"{us:.1f} us; per-warp clocks (avg over {warps} warps):")
for i in range(10):
    if buf[i]:
        print(f"  {names[i]:14s} {buf[i] / warps:10.0f}  {100 * buf[i] / tot:5.1f}%")
