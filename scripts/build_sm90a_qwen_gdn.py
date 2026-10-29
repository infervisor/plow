#!/usr/bin/env python3
"""Build <out>/plow_gdn_prefill.aot, the Qwen3.5 GDN prefill kernel as a packet cubin plus the
record plowrt needs to launch it itself (device/cuda/qwen_gdn.rs): no host library. CPU only:

  1. scripts/export_gdn_prefill_sm90.py <work>  (FlashInfer CuTe DSL kernel -> gdn_sm90.{h,o};
     needs the interpreter running this script to have flashinfer 0.6.16 + cutlass DSL 4.6.2)
  2. The export's embedded fatbin -> its sm_90a cubin and parameter layout (scripts/fa4_aot/build.py).
  3. The export's own host code, linked against the capture stubs (scripts/fa4_aot/capture_stubs.c)
     through scripts/gdn_aot/capture_main.c, is run on exemplar shapes; plowrt rebuilds every
     exemplar at load and refuses the object on any difference.

Container: b"PLOWAOT1", u32 manifest bytes, manifest JSON, then the cubin (device/cuda/aot.rs).
Recipe use: a `[[prep]]` step with `python = "{env:PYREF}"`, `args = ["{out}/assets"]`.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
HOST_STUB = "gdn_sm90_prefill"
HOST_STUB_VERSION = 1
ABI = "plow_gdn_prefill:v1:hk=16:hv=48:d=128:dtype=bf16:state=f32_vk"
# (tokens, nseq) exemplar launches; the addresses are distinct, 128-aligned and never dereferenced.
EXEMPLARS = [(1, 1), (301, 3), (8192, 64)]
ADDRS = [(0x7f - i) << 40 for i in range(10)]
# Bytes per parameter the export's host code defines: {ptr, i32 extent} tensors 12, tensor maps 64,
# i64 shapes it writes as i32 4. The rest is stack garbage the capture cannot always mask (the same
# in both fills); `unread()` proves the kernel never loads it.
DEFINED = [12, 12, 64, 4, 64, 4, 64, 4, 64, 4, 12, 12, 8, 12] + [4] * 8
TMAPS = {2, 4, 6, 8}
# sm_90: kernel parameters start at c[0x0][PARAM_BASE]; c[0x0][PARAM_ADDR] holds their generic address.
PARAM_BASE = 0x210
PARAM_ADDR = 0x198


class Unproven(Exception):
    pass


def width(op):
    if ".128" in op:
        return 16
    if ".64" in op:
        return 8
    if ".U8" in op or ".S8" in op:
        return 1
    if ".U16" in op or ".S16" in op:
        return 2
    return 4


def unread(cubin_path, layout):
    """Proves the kernel never reads a masked scalar byte. Every c[0x0] operand must be an
    immediate, its access width covers no masked byte, and each load of the generic parameter base
    (c[0x0][PARAM_ADDR]) is used only to form a tensor-map parameter's address."""
    sass = subprocess.run([os.environ.get("PLOW_CUOBJDUMP", "cuobjdump"), "-sass", str(cubin_path)],
                          capture_output=True, text=True, check=True).stdout
    lines = [re.sub(r"/\*[0-9a-f]+\*/|;.*", "", l).strip() for l in sass.splitlines()]
    lines = [l for l in lines if l]
    masked = {PARAM_BASE + off + b for i, ((off, size), d) in enumerate(zip(layout, DEFINED))
              if i not in TMAPS for b in range(d, size)}
    tmap_offsets = {layout[i][0] for i in TMAPS}
    reads = 0
    for n, l in enumerate(lines):
        for idx in re.findall(r"c\[0x0\]\[([^]]*)\]", l):
            if not re.fullmatch(r"0x[0-9a-f]+", idx):
                raise Unproven(f"register-indexed parameter read: {l}")
            at, op = int(idx, 16), l.split()[1] if l.startswith("@") else l.split()[0]
            reads += 1
            if masked & set(range(at, at + width(op))):
                raise Unproven(f"masked parameter byte read: {l}")
            if at == PARAM_ADDR:
                dst = re.match(r"(?:@\S+\s+)?ULDC\.64\s+UR(\d+),", l)
                if not dst:
                    raise Unproven(f"parameter base not loaded into a uniform pair: {l}")
                lo, hi = f"UR{dst.group(1)}", f"UR{int(dst.group(1)) + 1}"
                for u in lines[n + 1:]:
                    ops = [t.strip() for t in u.split(None, 1)[-1].split(",")] if " " in u else []
                    srcs, dsts = ops[1:], ops[:1]
                    if any(t in (lo, hi) for t in srcs):
                        m = re.fullmatch(rf"UIADD3 (UR\d+), UP\d, {lo}, (0x[0-9a-f]+), URZ", u)
                        x = re.fullmatch(rf"UIADD3\.X UR\d+, URZ, {hi}, URZ, !?UP\d, !UPT", u)
                        if not (m and int(m.group(2), 16) in tmap_offsets) and not x:
                            raise Unproven(f"parameter base used beyond tensor-map addressing: {u}")
                    if lo in dsts or hi in dsts:
                        break
    if not reads:
        raise Unproven("no parameter reads found in the SASS")


def mask(case):
    for i, (h, d) in enumerate(zip(case["params"], DEFINED)):
        if ".." in h[:2 * d]:
            raise Unproven(f"parameter {i}: a defined byte differs between captures")
    case["params"] = [h[:2 * d] + ".." * (len(h) // 2 - d) for h, d in zip(case["params"], DEFINED)]
    return case


def aot_helpers():
    spec = importlib.util.spec_from_file_location("fa4_aot_build", REPO / "scripts/fa4_aot/build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def upstream():
    import flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm90 as src
    return {"flashinfer": importlib.metadata.version("flashinfer-python"),
            "delta_rule_sm90_sha256": hashlib.sha256(pathlib.Path(src.__file__).read_bytes()).hexdigest(),
            "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl")}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    aot = aot_helpers()
    work = pathlib.Path(tempfile.mkdtemp(prefix="gdnaot.", dir=out))
    subprocess.run([sys.executable, str(REPO / "scripts/export_gdn_prefill_sm90.py"), str(work)], check=True)
    obj = work / "gdn_sm90.o"
    cubin, symbol = aot.cubin_of(obj)
    path = work / "gdn_sm90.cubin"
    path.write_bytes(cubin)
    layout = aot.kparams(path)
    if len(layout) != len(DEFINED):
        raise Unproven(f"layout {layout}")
    unread(path, layout)
    exe = work / "cap_gdn"
    cuda_inc = os.environ.get("PLOW_CUDA_INCLUDE", "/usr/local/cuda/include")
    subprocess.run(["gcc", "-O1", f"-I{work}", f"-I{cuda_inc}", "-o", str(exe),
                    str(REPO / "scripts/gdn_aot/capture_main.c"),
                    str(REPO / "scripts/fa4_aot/capture_stubs.c"), str(obj), "-lm"], check=True)
    cases = [mask(aot.exemplar(exe, layout, [t, n, *ADDRS])) for t, n in EXEMPLARS]
    entry = {"name": "gdn_prefill", "role": "gdn_prefill", "kv_heads": 0, "symbol": symbol,
             "params": layout, "sha256": hashlib.sha256(cubin).hexdigest(), "offset": 0,
             "bytes": len(cubin), "exemplars": cases}
    manifest = {"format": 1, "host_stub": HOST_STUB, "host_stub_version": HOST_STUB_VERSION,
                "upstream": upstream(), "abi": ABI, "entries": [entry]}
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    target = out / "plow_gdn_prefill.aot"
    target.write_bytes(b"PLOWAOT1" + struct.pack("<I", len(text)) + text + cubin)
    shutil.rmtree(work)
    print(target, target.stat().st_size, ABI, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
