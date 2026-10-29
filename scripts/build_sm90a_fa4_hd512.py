#!/usr/bin/env python3
"""Build <out>/libplow_fa4_hd512.so, the host-side FlashAttention-4 hd512 prefill attention
(Gemma-4 global layers) plowrt can load beside a packet. CPU only:

  1. scripts/export_fa4_hd512_sm90.py <out>/fa4 {1,2,combine}  (CuTe DSL ->
     fa4_hd512_h16kv{1,2}{,_split}.{h,o} and fa4_hd512_combine.{h,o};
     needs an interpreter with vLLM's vllm_flash_attn (0.28.0 built) + cutlass DSL 4.6, e.g. /opt/pytorch
     with PYTHONPATH=<site-packages>/nvidia_cutlass_dsl/dsl_packages)
  2. nvcc -shared runtime/nvidia/fa4_hd512.cpp + the objects -> <out>/libplow_fa4_hd512.so

Same toolchain and static-link rules as build_sm90a_qwen_gdn.py.
"""
import os
import pathlib
import subprocess
import sys
import sysconfig

REPO = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    obj = out / "fa4"
    obj.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUTE_DSL_ARCH="sm_90a")
    # The CuTe DSL wheel keeps its Python package under nvidia_cutlass_dsl/dsl_packages.
    dsl = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia_cutlass_dsl" / "dsl_packages"
    if dsl.is_dir():
        sys.path.insert(0, str(dsl))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(dsl), env.get("PYTHONPATH")]))
    for kvh in ("1", "2", "combine"):
        subprocess.run([sys.executable, str(REPO / "scripts/export_fa4_hd512_sm90.py"), str(obj), kvh],
                       check=True, env=env)
    nvcc = os.environ.get("PLOW_NVCC", "/usr/local/cuda/bin/nvcc")
    import cutlass
    dsl_root = pathlib.Path(cutlass.__file__).resolve().parents[2]
    cu = "cu13" if "release 13." in subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout else "cu12"
    archive = dsl_root / cu / "lib" / "libcuda_dialect_runtime_static.a"
    if not archive.is_file():
        raise SystemExit(f"CuTe DSL host runtime archive missing: {archive}")
    cmd = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-std=c++17", "-O2", "-cudart", "static",
           "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc", "-Wno-deprecated-gpu-targets",
           "-I", str(obj), "-o", str(out / "libplow_fa4_hd512.so"),
           str(REPO / "runtime/nvidia/fa4_hd512.cpp"),
           *sorted(str(p) for p in obj.glob("fa4_hd512_*.o")), str(archive), "-lcuda"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
