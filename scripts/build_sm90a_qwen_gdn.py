#!/usr/bin/env python3
"""Build <out>/libplow_gdn_prefill.so, the host-side Qwen3.5 GDN prefill library plowrt loads
beside a qwen3_5 packet (exec/gpu.rs NativeGdn). Two steps, both CPU only:

  1. scripts/export_gdn_prefill_sm90.py <out>/gdn   (FlashInfer CuTe DSL kernel -> gdn_sm90.{h,o};
     needs the interpreter running this script to have flashinfer 0.6.16 + cutlass 4.6.2)
  2. nvcc -shared runtime/nvidia/gdn_prefill.cpp gdn_sm90.o -> <out>/libplow_gdn_prefill.so

Toolchain as in build_sm90a_cubin.sh: PLOW_NVCC (default /usr/local/cuda/bin/nvcc),
NVCC_PREPEND_FLAGS (e.g. "-ccbin /usr/bin/g++-14" where the host gcc is newer than nvcc accepts).
Recipe use: a `[[prep]]` step with `python = "{env:PYREF}"`, `args = ["{out}/assets"]`.
"""
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    gdn = out / "gdn"
    gdn.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO / "scripts/export_gdn_prefill_sm90.py"), str(gdn)], check=True)
    nvcc = os.environ.get("PLOW_NVCC", "/usr/local/cuda/bin/nvcc")
    # The exported object calls the DSL's host shims (_cudaLaunchKernelEx, _cudaFuncSetAttribute, ...),
    # defined in the cutlass package's libcuda_dialect_runtime_static.a (cu12 for a CUDA 12 nvcc).
    import cutlass
    dsl_root = pathlib.Path(cutlass.__file__).resolve().parents[2]
    cu = "cu13" if "release 13." in subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout else "cu12"
    archive = dsl_root / cu / "lib" / "libcuda_dialect_runtime_static.a"
    if not archive.is_file():
        raise SystemExit(f"CuTe DSL host runtime archive missing: {archive}")
    # Static cudart and libstdc++: plowrt (a nix binary whose loader does not search the system
    # library path) dlopens this with a bare environment, and customer boxes carry no CUDA toolkit,
    # so the library may depend on libc only (libcuda is resolved by the static cudart at runtime).
    cmd = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-std=c++17", "-O2", "-cudart", "static",
           "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc", "-Wno-deprecated-gpu-targets",
           "-I", str(gdn), "-o", str(out / "libplow_gdn_prefill.so"),
           str(REPO / "runtime/nvidia/gdn_prefill.cpp"), str(gdn / "gdn_sm90.o"), str(archive), "-lcuda"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
