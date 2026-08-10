#!/usr/bin/env python3
# glm52_prep_ofold.py — ADDITIVE prep shard for the MlaMergeFold+o_proj fusion
# (perf-data/plow-gfx942/glm52-fusion-audit.md seam 1, emit knob PLOW_GLM_OFOLD=1).
#
# Per layer:
#   W_ofold[n, h*DK+d] = Σ_v  Wuv[h*DK+d, v] · Wo[n, h*VD+v]
#
# i.e. the per-head product of the lite prep's OWN bf16 derived tensors — v_absorb
# ([NH*DK, VD]) and the zz-shadowed bf16 o_proj ([H, NH*VD]) — so this reads the lite
# OUT dir, needs no re-prep and never touches the fp8 originals. Product computed in
# f32, stored bf16 (the fused GEMM is bf16; the reassociation is the fusion's
# documented numerics change, logit-gate class).
#
# Output: zz2-ofold-{L:05d}.safetensors in the SAME dir — a new tensor name
# (`…derived.o_fold.weight`), so nothing is shadowed; `zz2-` merely keeps the naming
# convention's sort-after-raw property. Resumable via the full prep's shard check.
#
# Usage:
#   python3 scripts/glm52_prep_ofold.py --lite /workspace/models/GLM-5.2-plow-lite \
#       [--layers 0-77]
import os, sys, json, struct, argparse, time, mmap

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import torch
from glm52_prep import STWriter, p_bf16, _DT
from glm52_prep_full import _shard_ok, _parse_layers, N_LAYERS

NH, DK, VD, H = 64, 512, 256, 6144


def _index_all(d):
    """Name -> (path, a, b, dtype, shape) over EVERY *.safetensors, sorted order so a later
    file wins — the same later-file-wins contract the plowrt loader applies, which is what
    makes the zz-derived bf16 o_proj shadow the raw fp8 one here too."""
    idx = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(d, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            idx[k] = (path, base + v["data_offsets"][0], base + v["data_offsets"][1],
                      v["dtype"], v["shape"])
    return idx


_MM = {}


def _load(idx, name):
    path, a, b, dt, shape = idx[name]
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    buf = _MM[path][a:b]
    return torch.frombuffer(bytearray(buf), dtype=_DT[dt]).view(*shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", required=True, help="the GLM-5.2 lite prep dir (read + write)")
    ap.add_argument("--layers", default=None)
    args = ap.parse_args()
    layers = _parse_layers(args.layers) if args.layers else list(range(N_LAYERS))
    idx = _index_all(args.lite)
    for L in layers:
        path = os.path.join(args.lite, f"zz2-ofold-{L:05d}.safetensors")
        ok, _ = _shard_ok(path)
        if ok:
            print(f"[ofold] layer {L}: shard complete, skipped")
            continue
        t0 = time.time()
        A = f"model.layers.{L}.self_attn."
        wuv = _load(idx, A + "derived.v_absorb.weight").view(NH, DK, VD).float()
        wo = _load(idx, A + "o_proj.weight")
        assert wo.dtype == torch.bfloat16, (
            f"layer {L}: o_proj resolved to {wo.dtype}, not the zz-derived bf16 copy — "
            "run glm52_prep_lite.py first")
        wo = wo.view(H, NH, VD).float()
        # per-head: block[h] = Wo[:, h, :] @ Wuv[h].T  ->  [H, DK]; head-major columns.
        of = torch.bmm(wo.permute(1, 0, 2), wuv.transpose(1, 2))  # [NH, H, DK]
        of = of.permute(1, 0, 2).reshape(H, NH * DK).contiguous()
        w = STWriter()
        w.add(A + "derived.o_fold.weight", "BF16", [H, NH * DK], H * NH * DK * 2, p_bf16(of))
        tmp = f"{path}.{os.getpid()}.tmp"
        w.flush(tmp)
        os.replace(tmp, path)
        print(f"[ofold] layer {L}: [{H},{NH*DK}] in {time.time()-t0:.1f}s")
    print("[ofold] done")


if __name__ == "__main__":
    main()
