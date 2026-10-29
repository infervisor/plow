#!/usr/bin/env python3
"""Build <out>/libplow_fa4_hd256.so, the host-side FlashAttention-4 hd256 causal prefill attention
over a row-interleaved KV cache (H query heads over KVH KV heads) plowrt can load beside a packet.
CPU only:

  build_sm90a_fa4_hd256.py OUT_DIR H KVH

  1. scripts/export_fa4_hd256_sm90.py <out>/fa4hd256 H {KVH,combine}  (CuTe DSL ->
     fa4_hd256_h<H>kv<KVH>{,_split}.{h,o} and fa4_hd256_h<H>_combine.{h,o}; needs an interpreter
     with vLLM's vllm_flash_attn (0.28.0 built) + cutlass DSL 4.6, e.g. /opt/pytorch)
  2. nvcc -shared runtime/nvidia/fa4_hd256.cpp + runtime/nvidia/kv_dequant_sm90.cu (the KV
     adapters, PLOW_FA4_HOST) + the objects -> <out>/libplow_fa4_hd256.so

Same toolchain and static-link rules as build_sm90a_fa4_hd512.py.
"""
import os
import pathlib
import subprocess
import sys
import sysconfig

REPO = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) != 4 or not all(a.isdigit() for a in sys.argv[2:]):
        print(__doc__, file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    heads, kv_heads = sys.argv[2], sys.argv[3]
    obj = out / "fa4hd256"
    obj.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUTE_DSL_ARCH="sm_90a")
    dsl = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia_cutlass_dsl" / "dsl_packages"
    if dsl.is_dir():
        sys.path.insert(0, str(dsl))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(dsl), env.get("PYTHONPATH")]))
    for arg in (kv_heads, "combine"):
        subprocess.run([sys.executable, str(REPO / "scripts/export_fa4_hd256_sm90.py"), str(obj), heads, arg],
                       check=True, env=env)
    nvcc = os.environ.get("PLOW_NVCC", "/usr/local/cuda/bin/nvcc")
    import cutlass
    dsl_root = pathlib.Path(cutlass.__file__).resolve().parents[2]
    cu = "cu13" if "release 13." in subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout else "cu12"
    archive = dsl_root / cu / "lib" / "libcuda_dialect_runtime_static.a"
    if not archive.is_file():
        raise SystemExit(f"CuTe DSL host runtime archive missing: {archive}")
    adapters = obj / "kv_adapters.o"
    subprocess.run([nvcc, "-c", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-arch=sm_90a", "-DPLOW_FA4_HOST",
                    "-o", str(adapters), str(REPO / "runtime/nvidia/kv_dequant_sm90.cu")], check=True)
    whole = f"fa4_hd256_h{heads}kv{kv_heads}"
    entries = [whole, whole + "_split", f"fa4_hd256_h{heads}_combine"]
    lib = out / "libplow_fa4_hd256.so"
    cmd = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-std=c++17", "-O2", "-cudart", "static",
           "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc", "-Wno-deprecated-gpu-targets",
           f"-DPLOW_FA4_H={heads}", f"-DPLOW_FA4_KVH={kv_heads}", "-I", str(obj),
           *[x for e in entries for x in ("-include", f"{e}.h")], "-o", str(lib),
           str(REPO / "runtime/nvidia/fa4_hd256.cpp"), str(adapters),
           *[str(obj / f"{e}.o") for e in entries], str(archive), "-lcuda"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    # plowrt dlopens it with nothing else loaded: only libc symbols may stay unresolved.
    undefined = subprocess.run(["nm", "-D", "--undefined-only", str(lib)],
                               capture_output=True, text=True, check=True).stdout.splitlines()
    unresolved = [l.split()[-1] for l in undefined if l.split()[0] == "U" and "@GLIBC" not in l]
    if unresolved:
        raise SystemExit(f"{lib.name} leaves {unresolved} unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
