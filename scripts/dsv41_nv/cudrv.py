"""Launch the DeepSeek-V4.1 sm_90a cubin's kernels on torch tensors through the CUDA driver API.

Test harness only: the plowrt engine launches the same cubin from Rust (crates/plowrt/src/dsv41).
"""
import ctypes
import torch

_cu = ctypes.CDLL("libcuda.so.1")


def _chk(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        _cu.cuGetErrorName(rc, ctypes.byref(name))
        raise RuntimeError(f"{what}: CUDA error {rc} {name.value}")


class Cubin:
    def __init__(self, path):
        torch.cuda.init()
        torch.empty(1, device="cuda")  # make torch's primary context current
        data = open(path, "rb").read()
        self._buf = ctypes.create_string_buffer(data)
        self.mod = ctypes.c_void_p()
        _chk(_cu.cuModuleLoadData(ctypes.byref(self.mod), self._buf), "cuModuleLoadData")
        self.fns = {}

    def fn(self, name):
        if name not in self.fns:
            f = ctypes.c_void_p()
            _chk(_cu.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()), f"get {name}")
            self.fns[name] = f
        return self.fns[name]

    def launch(self, name, grid, block, args, smem=0):
        """args: list of (ctype, value). torch tensors pass as their data_ptr (or 0 for None)."""
        f = self.fn(name)
        if smem > 48 * 1024:
            _chk(_cu.cuFuncSetAttribute(f, 8, smem), "max dyn smem")  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        store = []
        for a in args:
            if isinstance(a, torch.Tensor):
                store.append(ctypes.c_uint64(a.data_ptr()))
            elif a is None:
                store.append(ctypes.c_uint64(0))
            else:
                store.append(a)
        ptrs = (ctypes.c_void_p * len(store))(*[ctypes.cast(ctypes.pointer(s), ctypes.c_void_p) for s in store])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _chk(_cu.cuLaunchKernel(f, *g, *b, smem, stream, ptrs, None), f"launch {name}")


i32 = ctypes.c_int
i64 = ctypes.c_longlong
f32 = ctypes.c_float
