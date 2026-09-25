"""glm53_mxfp4_overlay.py OUT — runtime checkpoint for the GLM-5.3 MXFP4 W8A8/MHA recipe
(docs/amd/glm53-mxfp4-mi350x.md). New directory of symlinks over the prepped checkpoint
(scripts/glm53_prep_quark.py) plus:
  * model-mla-tp8-* : derived self_attn.derived.mla_fp8_tp8.{wk,wv} from the FP8-original MLA overlay,
    valid only because kv_b/q_b bytes are identical between Quark and the FP8 original (checked);
  * model-plow-qb-scale-inv / model-plow-kvb-scale-inv : sidecar copies of Quark's
    {q_b,kv_b}_proj.weight_scale under Plow's FP8 scale name (weight_scale_inv);
  * model-plow-oproj-fp8 : o_proj FP8 weight/scale aliases (PLOW_GLM_OPROJ_W8A8), all layers.
Existing files are never modified."""
import hashlib, json, os, struct, sys

SRC = os.environ.get("GLM_PREPPED", "/opt/models/plow-glm53-mxfp4-prepped-20260923-1aS85M")
QUARK = os.environ.get("GLM_QUARK", "/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b")
FP8 = os.environ.get("GLM_FP8", "/opt/models/GLM-5.3-full-aca966e4")
MLA = os.environ.get("GLM_MLA_TP8", "/opt/models/GLM-5.3-full-aca966e4-plow-mla-fp8-tp8")
OUT = sys.argv[1]
LAYERS = 78


def scan(d):
    idx = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                idx[k] = (p, 8 + n, v)
    return idx


def raw(idx, k):
    p, base, v = idx[k]
    a, b = v["data_offsets"]
    with open(p, "rb") as fh:
        fh.seek(base + a)
        return fh.read(b - a), v


def sidecar(name, proj, note):
    header, blob = {}, bytearray()
    for l in range(LAYERS):
        data, v = raw(q, f"model.layers.{l}.self_attn.{proj}.weight_scale")
        header[f"model.layers.{l}.self_attn.{proj}.weight_scale_inv"] = {
            "dtype": v["dtype"], "shape": v["shape"], "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    header["__metadata__"] = {"format": "pt", "source": note}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path = os.path.join(OUT, name)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)) + hj + blob)
    print(name, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])


def oproj_sidecar():
    """o_proj.weight_fp8 (+ .weight_scale_inv) for every layer: raw Quark FP8 bytes and F32 block scales
    (glm53_prep_quark.py build_oproj_fp8_alias, all layers), streamed tensor by tensor."""
    plan, off, header = [], 0, {}
    for l in range(LAYERS):
        for src, dst in (("o_proj.weight", "o_proj.weight_fp8"), ("o_proj.weight_scale", "o_proj.weight_scale_inv")):
            p, base, v = q[f"model.layers.{l}.self_attn.{src}"]
            a, b = v["data_offsets"]
            assert (src.endswith("scale") and v["dtype"] == "F32") or v["dtype"] == "F8_E4M3", (l, src, v["dtype"])
            header[f"model.layers.{l}.self_attn.{dst}"] = {"dtype": v["dtype"], "shape": v["shape"],
                                                            "data_offsets": [off, off + (b - a)]}
            plan.append((p, base + a, b - a))
            off += b - a
    header["__metadata__"] = {"format": "pt", "source": "raw copy of Quark o_proj FP8 weight/scale"}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path = os.path.join(OUT, "model-plow-oproj-fp8.safetensors")
    with open(path, "xb") as out:
        out.write(struct.pack("<Q", len(hj)) + hj)
        for p, start, n in plan:
            with open(p, "rb") as fh:
                fh.seek(start)
                while n:
                    chunk = fh.read(min(n, 64 << 20))
                    out.write(chunk)
                    n -= len(chunk)
    print(os.path.basename(path), os.path.getsize(path) / 1e9, "GB")


q, f = scan(QUARK), scan(FP8)
for l in range(LAYERS):
    pre = f"model.layers.{l}.self_attn."
    for name, fname in (("kv_b_proj.weight", "kv_b_proj.weight"), ("kv_b_proj.weight_scale", "kv_b_proj.weight_scale_inv"),
                        ("q_b_proj.weight", "q_b_proj.weight"), ("q_b_proj.weight_scale", "q_b_proj.weight_scale_inv")):
        if raw(q, pre + name)[0] != raw(f, pre + fname)[0]:
            sys.exit(f"MISMATCH layer {l} {name}: derived FP8 MLA tensors are not valid for this checkpoint")
print("kv_b/q_b bytes identical in all", LAYERS, "layers")

os.makedirs(OUT)
for e in sorted(os.listdir(SRC)):
    os.symlink(os.path.realpath(os.path.join(SRC, e)), os.path.join(OUT, e))
for e in sorted(os.listdir(MLA)):
    if e.startswith("model-mla-tp8-"):
        os.symlink(os.path.realpath(os.path.join(MLA, e)), os.path.join(OUT, e))
sidecar("model-plow-qb-scale-inv.safetensors", "q_b_proj", "copy of Quark q_b_proj.weight_scale (Plow scale name)")
sidecar("model-plow-kvb-scale-inv.safetensors", "kv_b_proj", "copy of Quark kv_b_proj.weight_scale (Plow scale name)")
oproj_sidecar()
print("overlay", OUT)
