#!/usr/bin/env python3
# glm52_prep_full.py — FULL-MODEL host weight-prep for GLM-5.2 (GlmMoeDsa) plow serving.
#
# Generalizes the emitter's SINGLE-layer prep (scripts/glm52_prep.py) to all 78 decoder layers
# (0-77) + the 3 model-level tensors, writing a plow-ready NAMED weight dir whose tensor names match
# the FROZEN full-model bound-tensor contract that gemma4.rs::declare_glm emits (verified against it).
#
# It does NOT reimplement the per-layer logic — it IMPORTS and CALLS the emitter's `prep_layer`
# (dense layers 0-2 + MoE layers 3-77, both branches) and reuses its STWriter/producers, so the
# per-layer output is byte-identical to what the ms1 gate already validates. The only new machinery
# here is:
#   1. SHARDING — one safetensors shard per layer (+ one globals shard) instead of a single 750 GB
#      file, so a fresh STWriter is flushed and freed each layer (RAM bounded to ~1 layer) and the
#      output matches the HF sharded convention (model-%05d-of-%05d.safetensors + index.json).
#   2. RESUMABILITY — a completed, size-consistent shard is skipped, so this multi-hour ~750 GB IO
#      job can be interrupted/resumed. (Layer 78 = MTP head is SKIPPED — not used for decode.)
#   3. VERIFY — names vs the contract, per-layer shapes, and a spot-check dequant vs the raw ckpt.
#
# Usage:
#   nix develop -c python3 scripts/glm52_prep_full.py \
#       --model /home/lava/models/GLM-5.2-FP8 --out /home/lava/models/GLM-5.2-FP8-plow
#   nix develop -c python3 scripts/glm52_prep_full.py --out <dir> --verify-only   # re-check an out dir
#
# No GPU needed (CPU/IO weight processing). ~750 GB out; run `df -h /home/lava` first.
import os, sys, json, struct, argparse, time

# Import the emitter's single-layer prep for reuse (do NOT modify it).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import glm52_prep as P                 # prep_layer, STWriter, p_bf16, load_tensor, dequant_blockfp8, _index_shards
import numpy as np
import torch
from transformers import AutoConfig

N_LAYERS = 78                          # decoder layers 0-77; index 78 (MTP head) is SKIPPED
SHARD_FMT = "model-{:05d}-of-{:05d}.safetensors"


# ---------------------------------------------------------------- shard integrity (resume check)
def _shard_ok(path):
    """True if `path` is a complete safetensors shard: header parses and the file size equals
    8 + header + the max declared data offset (a truncated/interrupted flush fails this)."""
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
    except Exception:
        return False, {}
    data_end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
    return sz == 8 + n + data_end, {k: v for k, v in hdr.items() if k != "__metadata__"}


def _write_shard(build, path, tmp_tag=""):
    """build(w) populates STWriter w; flush to `path`. Returns {name: (dtype, shape, nbytes)}.
    `tmp_tag` disambiguates the scratch file so a concurrent writer of the same shard can't collide."""
    w = P.STWriter()
    build(w)
    names = {name: (dt, shape, nb) for (name, dt, shape, nb, _) in w.entries}
    tmp = f"{path}.{os.getpid()}{tmp_tag}.tmp"
    w.flush(tmp)
    os.replace(tmp, path)              # atomic publish so a crash never leaves a "valid-looking" partial
    return names


def _globals_builder(idx, cfg):
    """Producer for the model-level bf16 tensors (embed_tokens / final norm / lm_head)."""
    H, V = cfg.hidden_size, cfg.vocab_size

    def build(w):
        w.add("model.embed_tokens.weight", "BF16", [V, H], V * H * 2,
              P.p_bf16(P.load_tensor(idx, "model.embed_tokens.weight")))
        w.add("model.norm.weight", "BF16", [H], H * 2,
              P.p_bf16(P.load_tensor(idx, "model.norm.weight")))
        w.add("lm_head.weight", "BF16", [V, H], V * H * 2,
              P.p_bf16(P.load_tensor(idx, "lm_head.weight")))
    return build


def _scan_index(out_dir):
    """Scan out_dir for complete shard files; return (weight_map, total_size) over present tensors."""
    weight_map, total_size = {}, 0
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("model-") and fn.endswith(".safetensors")):
            continue
        ok, hdr = _shard_ok(os.path.join(out_dir, fn))
        if not ok:
            continue
        for name, v in hdr.items():
            weight_map[name] = fn
            total_size += v["data_offsets"][1] - v["data_offsets"][0]
    return weight_map, total_size


# ---------------------------------------------------------------- full-model prep
def prep_full(model_dir, out_dir, layers=None):
    """Prep `layers` (default = all 78) + globals into `out_dir`. A restricted list produces a
    standalone slice dir (its own index.json + total_shards) — e.g. layers 0-5 for the decode-loop
    smoke-test. Shard `i+1` holds `layers[i]`; the globals shard is `len(layers)+1`."""
    if layers is None:
        layers = list(range(N_LAYERS))
    idx = P._index_shards(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfgj = f.read()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(cfgj)

    total_shards = len(layers) + 1     # one shard per layer (1..n) + a globals shard (n+1)
    weight_map, total_size = {}, 0
    t0 = time.time()

    def record(shard_no, names):
        nonlocal total_size
        fn = SHARD_FMT.format(shard_no, total_shards)
        for name, (_dt, _shape, nb) in names.items():
            weight_map[name] = fn
            total_size += nb

    for i, L in enumerate(layers):
        shard_no = i + 1
        path = os.path.join(out_dir, SHARD_FMT.format(shard_no, total_shards))
        ok, hdr = _shard_ok(path)
        if ok:
            names = {k: (v["dtype"], v["shape"],
                         v["data_offsets"][1] - v["data_offsets"][0]) for k, v in hdr.items()}
            record(shard_no, names)
            print(f"[prep] layer {L:2d}: shard {shard_no}/{total_shards} exists ({len(names)} tensors) — skip")
            continue
        # RoPE is applied dynamically on device; L/pos are signature-only in prep_layer (unused for bytes).
        names = _write_shard(lambda w: P.prep_layer(idx, cfg, w, 512, L, 511), path)
        gb = os.path.getsize(path) / 1e9
        record(shard_no, names)
        kind = "dense" if L < cfg.first_k_dense_replace else "MoE  "
        print(f"[prep] layer {L:2d}: {kind} shard {shard_no}/{total_shards} wrote {len(names)} tensors "
              f"({gb:.2f} GB, {total_size/1e12:.3f} TB cum, {time.time()-t0:.0f}s)")

    # --- globals shard: embed_tokens / final norm / lm_head (all bf16), reusing the emitter producers ---
    gshard = total_shards
    gpath = os.path.join(out_dir, SHARD_FMT.format(gshard, total_shards))
    ok, hdr = _shard_ok(gpath)
    if ok:
        names = {k: (v["dtype"], v["shape"],
                     v["data_offsets"][1] - v["data_offsets"][0]) for k, v in hdr.items()}
        record(gshard, names)
        print(f"[prep] globals: shard {gshard}/{total_shards} exists — skip")
    else:
        names = _write_shard(_globals_builder(idx, cfg), gpath)
        record(gshard, names)
        print(f"[prep] globals: shard {gshard}/{total_shards} wrote {len(names)} tensors")

    # --- HF-standard index (weight_map + total_size) so the full-model loader binds by name ---
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=1)
    print(f"[prep] DONE — {len(weight_map)} tensors across {total_shards} shards, "
          f"{total_size/1e12:.3f} TB, {time.time()-t0:.0f}s -> {out_dir}")
    return weight_map, cfg, layers


# ---------------------------------------------------------------- early-unblock: globals + partial index
def refresh_index(model_dir, out_dir):
    """Publish a PARTIAL model.safetensors.index.json for whatever shards are present so far, and
    ensure the GLOBALS shard (embed/norm/lm_head) exists — so a consumer (the emitter's decode-loop
    smoke-test) can bind the already-written early layers + globals BEFORE the full prep finishes.
    Safe to run concurrently with prep_full: it only writes the globals shard (if absent, to the same
    n+1 name prep_full will later resume-skip) and rewrites index.json (prep_full overwrites it with
    the complete map at the end). The globals shard uses the FULL total_shards (N_LAYERS+1) naming."""
    cfg = AutoConfig.from_pretrained(out_dir)
    total_shards = N_LAYERS + 1
    gpath = os.path.join(out_dir, SHARD_FMT.format(total_shards, total_shards))
    ok, _ = _shard_ok(gpath)
    if not ok:
        idx = P._index_shards(model_dir)
        _write_shard(_globals_builder(idx, cfg), gpath, tmp_tag="-glob")
        print(f"[refresh] wrote globals shard {SHARD_FMT.format(total_shards, total_shards)}")
    else:
        print("[refresh] globals shard already present")
    weight_map, total_size = _scan_index(out_dir)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size, "partial": True},
                   "weight_map": weight_map}, f, indent=1)
    layers = sorted({int(m.group(1)) for k in weight_map
                     if (m := __import__("re").match(r"model\.layers\.(\d+)\.", k))})
    contig = layers == list(range(len(layers)))
    print(f"[refresh] partial index: {len(weight_map)} tensors, {total_size/1e12:.3f} TB, "
          f"layers present {layers[0] if layers else '-'}..{layers[-1] if layers else '-'} "
          f"({'contiguous' if contig else 'GAPS'}), globals "
          f"{'yes' if 'lm_head.weight' in weight_map else 'NO'} -> {out_dir}")
    return weight_map, layers


# ---------------------------------------------------------------- verification
def expected_names(cfg, layers=None):
    """The exact per-layer + model-level tensor NAMES the frozen declare_glm contract binds.
    Mirrors gemma4.rs::declare_glm (verified). Used to assert the prepped dir is name-complete."""
    if layers is None:
        layers = list(range(N_LAYERS))
    H, NH, DK, DR, VD, QL, E, IMOE = (cfg.hidden_size, cfg.num_attention_heads, cfg.kv_lora_rank,
        cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.q_lora_rank, cfg.n_routed_experts, cfg.moe_intermediate_size)
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for L in layers:
        p = f"model.layers.{L}."
        a = p + "self_attn."
        names |= {
            p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            a + "q_a_proj.weight", a + "q_a_layernorm.weight", a + "kv_a_layernorm.weight",
            a + "derived.q_absorb.weight", a + "derived.q_rope.weight", a + "derived.kv_a_latent.weight",
            a + "derived.k_rope.weight", a + "derived.v_absorb.weight", a + "o_proj.weight",
        }
        if cfg.indexer_types[L] == "full":     # DSA lightning indexer (full layers only) — G1
            ix = a + "indexer."
            names |= {ix + "wq_b.weight", ix + "wq_b.weight_scale_inv",
                      ix + "wk.weight", ix + "wk.weight_scale_inv",
                      ix + "k_norm.weight", ix + "k_norm.bias", ix + "weights_proj.weight"}
        if L < cfg.first_k_dense_replace:                      # dense FFN (fp8 verbatim)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names |= {p + f"mlp.{proj}.weight", p + f"mlp.{proj}.weight_scale_inv"}
        else:                                                  # MoE
            names |= {p + "mlp.gate.weight", p + "mlp.gate.e_score_correction_bias"}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.add(p + f"mlp.shared_experts.{proj}.weight")
            for e in range(E):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    names |= {p + f"mlp.experts.{e}.{proj}.weight",
                              p + f"mlp.experts.{e}.{proj}.weight_scale_inv"}
    return names


def verify(model_dir, out_dir, weight_map=None, cfg=None, layers=None):
    if cfg is None:
        cfg = AutoConfig.from_pretrained(out_dir)
    if weight_map is None:
        with open(os.path.join(out_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
    if layers is None:                 # verify exactly the layers present (works for slice/partial dirs too)
        import re
        layers = sorted({int(m.group(1)) for k in weight_map
                         if (m := re.match(r"model\.layers\.(\d+)\.", k))})

    # (a) NAME contract: prepped set == expected set (no missing, no extras — extras would mean an
    #     emitted-but-forbidden tensor like expert tables / cos-sin / kv caches leaked in).
    want, got = expected_names(cfg, layers), set(weight_map)
    missing, extra = want - got, got - want
    print(f"[verify] names: expected {len(want)}, present {len(got)}, "
          f"missing {len(missing)}, extra {len(extra)}")
    if missing:
        print("  MISSING (sample):", sorted(missing)[:6])
    if extra:
        print("  EXTRA (sample):  ", sorted(extra)[:6])

    # (b) SHAPES + DTYPES spot-check on the first dense layer, first MoE layer, and globals, by
    #     reading each owning shard's header.
    def hdr_of(shard_fn):
        path = os.path.join(out_dir, shard_fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n))
    H, NH, DK, DR, VD, QL, E, IMOE, V = (cfg.hidden_size, cfg.num_attention_heads, cfg.kv_lora_rank,
        cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.q_lora_rank, cfg.n_routed_experts,
        cfg.moe_intermediate_size, cfg.vocab_size)
    DI = cfg.intermediate_size
    IB, HB = -(-IMOE // 128), -(-H // 128)
    checks = [  # (tensor, dtype, shape)
        ("model.embed_tokens.weight", "BF16", [V, H]),
        ("lm_head.weight", "BF16", [V, H]),
        ("model.layers.0.self_attn.derived.q_absorb.weight", "BF16", [NH * DK, QL]),
        ("model.layers.0.self_attn.derived.v_absorb.weight", "BF16", [NH * DK, VD]),
        ("model.layers.0.self_attn.o_proj.weight", "BF16", [H, NH * VD]),
        ("model.layers.0.mlp.gate_proj.weight", "F8_E4M3", [DI, H]),
        ("model.layers.0.mlp.gate_proj.weight_scale_inv", "F32", [-(-DI // 128), HB]),
        ("model.layers.3.mlp.gate.weight", "BF16", [E, H]),
        ("model.layers.3.mlp.gate.e_score_correction_bias", "F32", [E]),
        ("model.layers.3.mlp.shared_experts.gate_proj.weight", "BF16", [IMOE, H]),
        ("model.layers.3.mlp.experts.0.gate_proj.weight", "F8_E4M3", [IMOE, H]),
        ("model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv", "F32", [IB, HB]),
        ("model.layers.3.mlp.experts.255.down_proj.weight", "F8_E4M3", [H, IMOE]),
    ]
    shape_ok = True
    for name, dt, shape in checks:
        if name not in weight_map:
            print(f"  SHAPE FAIL {name}: absent from index"); shape_ok = False; continue
        h = hdr_of(weight_map[name]).get(name)
        if h is None or h["dtype"] != dt or h["shape"] != shape:
            print(f"  SHAPE FAIL {name}: got {h}, want dtype={dt} shape={shape}"); shape_ok = False
    print(f"[verify] shapes/dtypes: {'OK' if shape_ok else '*** MISMATCH ***'} ({len(checks)} spot-checks)")

    # (c) DEQUANT spot-check vs the RAW checkpoint (only meaningful with the original model present):
    #     experts are copied VERBATIM so their fp8 bytes must be bit-identical; q_a_proj is dequantized
    #     so it must equal dequant_blockfp8 of the raw. Read the prepped tensor straight from its shard.
    deq_ok = None
    if model_dir and os.path.isdir(model_dir):
        idx = P._index_shards(model_dir)

        def read_prepped(name):
            fn = weight_map[name]
            path = os.path.join(out_dir, fn)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
                base = 8 + n
                a, b = hdr[name]["data_offsets"]
                fh.seek(base + a)
                raw = fh.read(b - a)
            dt = hdr[name]["dtype"]
            arr = np.frombuffer(raw, dtype=np.uint8) if dt == "F8_E4M3" else (
                  torch.frombuffer(bytearray(raw), dtype=P._DT[dt]).view(*hdr[name]["shape"]))
            return arr, dt

        deq_ok = True
        # VERBATIM expert: prepped fp8 bytes == raw fp8 bytes.
        en = "model.layers.3.mlp.experts.0.gate_proj.weight"
        pre, _ = read_prepped(en)
        rawbuf, _, _ = P.raw_bytes(idx, en)
        vb = np.frombuffer(bytes(rawbuf), dtype=np.uint8)
        if not (pre.shape == vb.shape and (pre == vb).all()):
            print(f"  DEQUANT FAIL {en}: verbatim fp8 bytes differ from raw ckpt"); deq_ok = False
        else:
            print(f"  verbatim fp8 expert bytes bit-identical to raw ({pre.size} bytes) — OK")
        # DEQUANTIZED q_a_proj: prepped bf16 == dequant_blockfp8(raw), within bf16 rounding.
        qn = "model.layers.3.self_attn.q_a_proj.weight"
        preq, _ = read_prepped(qn)
        ref = P.dequant_blockfp8(idx, qn).to(torch.bfloat16)
        got = preq.to(torch.float32)
        rel = (got - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9)
        print(f"  dequant q_a_proj prepped-vs-raw relerr={rel:.2e} {'OK' if rel < 1e-2 else '*** FAIL ***'}")
        deq_ok &= rel < 1e-2

    ok = (not missing) and (not extra) and shape_ok and (deq_ok is not False)
    print(f"\n[verify] {'ALL CHECKS PASSED' if ok else '*** VERIFY FAILED ***'}")
    return ok


def _parse_layers(s):
    """'0-5' | '0,1,2' | '0-2,7' -> sorted unique int list."""
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL_DIR", "/home/lava/models/GLM-5.2-FP8"))
    ap.add_argument("--out", default="/home/lava/models/GLM-5.2-FP8-plow")
    ap.add_argument("--layers", default=None,
                    help="restrict to a layer subset (e.g. '0-5') -> standalone slice dir; default all 78")
    ap.add_argument("--verify-only", action="store_true", help="check an existing out dir, don't write")
    ap.add_argument("--refresh-index", action="store_true",
                    help="write the globals shard early + publish a PARTIAL index for present shards, "
                         "then exit (unblocks a consumer before the full prep finishes)")
    args = ap.parse_args()

    if args.refresh_index:
        refresh_index(args.model, args.out)
        return
    if args.verify_only:
        ok = verify(args.model, args.out)
        sys.exit(0 if ok else 1)

    layers = _parse_layers(args.layers) if args.layers else None
    weight_map, cfg, layers = prep_full(args.model, args.out, layers)
    ok = verify(args.model, args.out, weight_map, cfg, layers)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
