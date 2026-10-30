#!/usr/bin/env python3
"""Build a packet object holding FlashAttention-4's SM90 forward as cubins plus the record plowrt
needs to launch them itself (exec/gpu/fa4_sm90.rs): no host library. CPU only:

  build.py OUT_DIR hd512 [f16]      -> OUT_DIR/plow_fa4_hd512.aot (16 heads over 1 or 2 KV heads)
  build.py OUT_DIR hd256 H KVH      -> OUT_DIR/plow_fa4_hd256.aot (H heads over KVH, row-interleaved)

  1. scripts/export_fa4_hd{512,256}_sm90.py (CuTe DSL AOT export; needs an interpreter with vLLM's
     vllm_flash_attn (0.28.0 built) + cutlass DSL 4.6, e.g. /opt/pytorch).
  2. Each export's embedded fatbin -> its sm_90a cubin (byte-identical to the one the export's own
     host code loads); kernel parameter layout from the cubin's KPARAM_INFO (cuobjdump).
  3. The export's own host code, linked against capture stubs (capture_stubs.c), is run on
     exemplar shapes; its launches (grid, block, smem, attributes, parameter bytes with padding
     masked) are recorded. plowrt rebuilds every exemplar at load and refuses the object on any
     difference, so a cubin whose host logic differs from plowrt's stub never runs.
  4. runtime/nvidia/kv_dequant_sm90.cu -> the KV adapter cubin (FP8 page gather, tail zeroing,
     bf16 <-> fp16 conversion).

Container: b"PLOWAOT1", u32 manifest bytes, manifest JSON, then the blobs it indexes.
"""
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
HOST_STUB = "fa4_sm90_fwd"
HOST_STUB_VERSION = 1
# Exemplar launches: (total_q, requests, page-table width, pages, KV head stride, splits).
EXEMPLARS = [(301, 3, 67, 4099, 1 << 23, 5), (6, 1, 180, 16384, 16384 * 512, 52),
             (4096, 8, 256, 2048, 1 << 20, 1)]


def upstream():
    import vllm
    cute = pathlib.Path(vllm.__file__).parent / "vllm_flash_attn" / "cute"
    h = hashlib.sha256()
    for p in sorted(cute.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return {"vllm": vllm.__version__, "vllm_flash_attn_cute_sha256": h.hexdigest(),
            "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl")}


def elf_size(b):
    phoff, shoff = struct.unpack_from("<QQ", b, 0x20)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", b, 0x36)
    size = max(shoff + shentsize * shnum, phoff + phentsize * phnum)
    for i in range(shnum):
        typ, = struct.unpack_from("<I", b, shoff + i * shentsize + 4)
        off, sz = struct.unpack_from("<QQ", b, shoff + i * shentsize + 0x18)
        if typ != 8:
            size = max(size, off + sz)
    for i in range(phnum):
        off, = struct.unpack_from("<Q", b, phoff + i * phentsize + 8)
        sz, = struct.unpack_from("<Q", b, phoff + i * phentsize + 0x20)
        size = max(size, off + sz)
    return size


def cubin_of(obj):
    """The export object's embedded fatbin (symbol kernels_binary) -> (cubin bytes, kernel symbol)."""
    nm = subprocess.run(["nm", "-S", str(obj)], capture_output=True, text=True, check=True).stdout
    off, size = next((int(l.split()[0], 16), int(l.split()[1], 16)) for l in nm.splitlines()
                     if l.endswith(" kernels_binary"))
    kernel = [l.split()[-1][len("kernels_"):] for l in nm.splitlines()
              if " b kernels_kernel_" in l]
    assert len(kernel) == 1, kernel
    hdr = subprocess.run(["readelf", "-S", "-W", str(obj)], capture_output=True, text=True,
                         check=True).stdout
    sec = next(int(l.split("]")[1].split()[3], 16) for l in hdr.splitlines() if " .lrodata " in l)
    blob = obj.read_bytes()[sec + off:sec + off + size]
    at = blob.find(b"\x7fELF")
    return blob[at:at + elf_size(blob[at:])], kernel[0]


def kparams(cubin_path):
    t = subprocess.run([os.environ.get("PLOW_CUOBJDUMP", "cuobjdump"), "-elf", str(cubin_path)],
                       capture_output=True, text=True, check=True).stdout
    found = re.findall(r"Ordinal : (0x[0-9a-f]+)\s+Offset\s+: (0x[0-9a-f]+)\s+Size\s+: (0x[0-9a-f]+)", t)
    p = {int(o, 16): [int(off, 16), int(sz, 16)] for o, off, sz in found}
    assert sorted(p) == list(range(len(p))), p
    return [p[k] for k in sorted(p)]


def capture(exe, sizes, args, fill):
    env = dict(os.environ, PLOW_CAP_SIZES=",".join(str(s) for _, s in sizes), PLOW_CAP_FILL=str(fill))
    out = subprocess.run([str(exe), *map(str, args)], env=env, capture_output=True, text=True,
                         check=True).stdout
    r = {"params": [], "attrs": [], "kernel_attrs": []}
    for l in out.splitlines():
        f = l.split()
        if f[0] == "launch":
            m = re.match(r"launch grid=(\d+),(\d+),(\d+) block=(\d+),(\d+),(\d+) smem=(\d+)", l)
            g = [int(x) for x in m.groups()]
            r["grid"], r["block"], r["smem"] = g[:3], g[3:6], g[6]
        elif f[0] == "param":
            r["params"].append(f[3] if len(f) > 3 else "")
        elif f[0] == "attr":
            # Every attribute these entries set holds an int at offset 0 of the value union.
            r["attrs"].append([int(f[1]), int.from_bytes(bytes.fromhex(f[2])[:4], "little")])
        elif f[0] == "kernel_attr":
            r["kernel_attrs"].append([int(f[1]), int(f[2])])
    return r


def exemplar(exe, sizes, args):
    """A launch with never-written bytes masked: a byte is kept iff every capture agrees. The
    captures run in separate processes (ASLR moves the stack residue a host frame leaves in
    padding) over differently filled stacks; residue that survives every run is the same in
    every build, so the container stays byte-reproducible."""
    runs = [capture(exe, sizes, args, fill) for fill in (0x00, 0xff, 0x5a, 0xa5, 0x00, 0xff)]
    a = runs[0]
    for r in runs[1:]:
        assert {k: a[k] for k in a if k != "params"} == {k: r[k] for k in r if k != "params"}
    a["params"] = ["".join(x[i:i + 2] if all(r["params"][j][i:i + 2] == x[i:i + 2] for r in runs) else ".."
                           for i in range(0, len(x), 2))
                   for j, x in enumerate(a["params"])]
    a["args"] = list(args)
    return a


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[1] not in ("hd512", "hd256"):
        print(__doc__, file=sys.stderr)
        return 2
    out = pathlib.Path(argv[0])
    hd512 = argv[1] == "hd512"
    f16 = hd512 and argv[2:] == ["f16"]
    heads, kvhs = (16, [1, 2]) if hd512 else (int(argv[2]), [int(argv[3])])
    head_dim = 512 if hd512 else 256
    work = pathlib.Path(tempfile.mkdtemp(prefix="fa4aot.", dir=out))
    env = dict(os.environ, CUTE_DSL_ARCH="sm_90a")
    dsl = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia_cutlass_dsl" / "dsl_packages"
    if dsl.is_dir():
        sys.path.insert(0, str(dsl))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(dsl), env.get("PYTHONPATH")]))
    exporter = REPO / ("scripts/export_fa4_hd512_sm90.py" if hd512 else "scripts/export_fa4_hd256_sm90.py")
    jobs = [[str(k)] for k in kvhs] + [["combine"]]
    for j in jobs:
        extra = (["f16"] if f16 else []) if hd512 else []
        lead = [] if hd512 else [str(heads)]
        subprocess.run([sys.executable, str(exporter), str(work), *lead, *j, *extra], check=True, env=env)
    stem = "fa4_hd512" if hd512 else "fa4_hd256"
    names = []
    for k in kvhs:
        whole = f"{stem}_h{heads}kv{k}"
        names += [(whole, "whole", k), (whole + "_split", "split", k)]
    names.append((f"{stem}_combine" if hd512 else f"{stem}_h{heads}_combine", "combine", 0))
    cuda_inc = os.environ.get("PLOW_CUDA_INCLUDE", "/usr/local/cuda/include")
    blobs, entries = [], []
    for name, role, kvh in names:
        cubin, symbol = cubin_of(work / f"{name}.o")
        path = work / f"{name}.cubin"
        path.write_bytes(cubin)
        layout = kparams(path)
        exe = work / f"cap_{name}"
        kind = {"whole": 0, "split": 1, "combine": 2}[role]
        subprocess.run(["gcc", "-O1", f"-I{work}", f"-I{cuda_inc}", f"-DE=plow_{name}",
                        f"-DHDR={work / (name + '.h')}", f"-DKIND={kind}", f"-DH={heads}",
                        f"-DKVH={max(kvh, 1)}", f"-DD={head_dim}", f"-DHM={int(hd512)}",
                        "-o", str(exe), str(HERE / "capture_main.c"), str(HERE / "capture_stubs.c"),
                        str(work / f"{name}.o"), "-lm"], check=True)
        cases = [exemplar(exe, layout, e) for e in EXEMPLARS]
        entries.append({"name": name, "role": role, "kv_heads": kvh, "symbol": symbol,
                        "params": layout, "sha256": hashlib.sha256(cubin).hexdigest(),
                        "exemplars": cases})
        blobs.append(cubin)
    nvcc = os.environ.get("PLOW_NVCC", "/usr/local/cuda/bin/nvcc")
    adapters = work / "kv_adapters.cubin"
    subprocess.run([nvcc, "-cubin", "-std=c++17", "-O3", "-arch=sm_90a", "-o", str(adapters),
                    str(REPO / "runtime/nvidia/kv_dequant_sm90.cu")], check=True)
    blob = adapters.read_bytes()
    entries.append({"name": "kv_adapters", "role": "adapters", "kv_heads": 0, "symbol": "",
                    "params": [], "sha256": hashlib.sha256(blob).hexdigest(), "exemplars": []})
    blobs.append(blob)
    at = 0
    for e, b in zip(entries, blobs):
        e["offset"], e["bytes"] = at, len(b)
        at += len(b)
    layout = "head_major" if hd512 else "ilv"
    kv = ",".join(str(k) for k in kvhs)
    manifest = {"format": 1, "host_stub": HOST_STUB, "host_stub_version": HOST_STUB_VERSION,
                "upstream": upstream(),
                "abi": f"lib_flash_prefill_v2:fa4:hd={head_dim}:heads={heads}:kv_heads={kv}:"
                       f"layout={layout}:dtype={'f16' if f16 else 'bf16'}",
                "entries": entries}
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    target = out / f"plow_fa4_hd{head_dim}.aot"
    target.write_bytes(b"PLOWAOT1" + struct.pack("<I", len(text)) + text + b"".join(blobs))
    shutil.rmtree(work)
    print(target, target.stat().st_size, manifest["abi"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
