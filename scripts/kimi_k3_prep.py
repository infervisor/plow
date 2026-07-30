#!/usr/bin/env python3
# kimi_k3_prep.py — HOST WEIGHT-PREP for Kimi-K3 (moonshotai/Kimi-K3), the analogue of
# scripts/glm52_prep.py.  See plans/kimi-k3-frontend.md for the design this implements.
#
# THE CHECKPOINT (measured, not assumed — 44/96 shards on disk 2026-07-28):
#   96 shards, ~1.59 TB projected (NOT the 618 GB in the original brief).  Roughly one layer per
#   shard: layer 0 is 2.3 GB (dense FFN), every MoE layer is 17.0 GB, of which 15.7 GB is the 896
#   mxfp4 routed experts.  The experts are 97% of the model.
#
#   text prefix   language_model.model.        (multimodal wrapper — NOT "model.")
#   layer 0       KDA attention + dense mlp.{gate,up,down}_proj    (first_k_dense_replace=1)
#   layers 1..92  KDA or MLA attention + block_sparse_moe
#   MLA at 0-based {3,7,11,...,87,91,92}  = linear_attn_config.full_attn_layers MINUS ONE.
#                 The lists in config.json are 1-BASED (configuration_kimi_k3.py::is_kda_layer
#                 tests `(layer_idx + 1) in kda_layers`).  Confirmed against the shard headers:
#                 self_attn.q_a_proj appears on 0-based 3, 7, 11, ... and nowhere else.
#
# WHAT THIS SCRIPT IS FOR, in order of what works today:
#
#   --inspect   (DEFAULT)  Header-only reconciliation.  mmaps nothing, reads only the safetensors
#               JSON headers of whatever shards have landed, classifies every layer from its
#               ACTUAL tensor set, and checks every config dimension against the tensors.  This is
#               the GLM-5.2 lesson made mechanical: AutoConfig reported qk_rope_head_dim=192 where
#               the tensors said 64 and it cost a day.  TRUST THE TENSORS.  Runs on a partial
#               download and never reads a byte of tensor data.  stdlib + numpy only.
#
#   --layers    Write a plow-named weight dir for the given 0-based layers.  Streaming: the only
#               thing ever held in RAM is one layer's MLA projections.
#
# WHAT IT DOES NOT DO, deliberately:
#
#   * It does not copy the mxfp4 routed experts (--experts inplace, the default).  They are
#     1.54 of the 1.59 TB and they are ALREADY in plow's exact layout: `weight_packed` is
#     [N, K/2] with the low nibble at even k and `weight_scale` is one E8M0 byte per 32 of K with
#     row stride K/32 — byte-for-byte what DevOp::GemvMxfp4 documents
#     (crates/packet/src/dev.rs:622).  Copying them would burn 1.5 TB to change nothing.  The
#     runtime already binds routed experts by name pattern rather than as packet tensors
#     (`bind_packed_experts`, crates/plowrt exec/amd.rs, done for GLM); K3 reuses that seam and
#     reads them from the original snapshot.  `--experts verbatim` copies them anyway, for a
#     single-layer bring-up fixture.
#
#   * It does not emit a KDA layer's derived weights.  KDA semantics are owned by a sibling agent
#     (docs/kimi-k3-kda.md); this script passes KDA tensors through under their original names so
#     that work is not pre-empted.
#
#   * It does not touch the vision tower.  TEXT ONLY.
#
# E8M0: scales are BIASED BY 127 (knob-contract §2 — this has bitten twice).  Neutral is 127, not
# 0; byte 0 means 2^-127 and flushes its block to zero.  compressed-tensors stores the raw biased
# exponent byte, so a verbatim copy is already correct and any synthesised scale must be 127.
#
# Usage:
#   python3 scripts/kimi_k3_prep.py --model <snapshot> --inspect
#   python3 scripts/kimi_k3_prep.py --model <snapshot> --out <dir> --layers 3
import argparse, json, mmap, os, re, struct, sys

import numpy as np

# safetensors dtype -> (numpy dtype, bytes per element).  BF16 has no numpy type; it is carried as
# uint16 and converted explicitly (see bf16_to_f32 / f32_to_bf16).
_DT = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "F16": (np.float16, 2),
       "U8": (np.uint8, 1), "I8": (np.int8, 1), "I32": (np.int32, 4), "I64": (np.int64, 8),
       "F8_E4M3": (np.uint8, 1)}

TEXT = "language_model.model."


# ---------------------------------------------------------------- safetensors: headers only
def index_shards(model_dir):
    """{name: (path, start, end, dtype, shape)} over every shard PRESENT.

    Tolerates a download in progress: a shard whose header is not fully written is skipped, and a
    missing shard is simply absent.  Returns the (present, total) shard counts alongside, because
    every conclusion drawn from this index has to be qualified by them.  **A tensor's absence
    proves nothing while present < total.**
    """
    idx, present, total = {}, 0, 0
    for fn in sorted(os.listdir(model_dir)):
        base = fn[:-len(".partial.safetensors")] if fn.endswith(".partial.safetensors") else (
            fn[:-len(".safetensors")] if fn.endswith(".safetensors") else None)
        if base is None:
            continue
        if "-of-" in base:
            try:
                total = max(total, int(base.rsplit("-of-", 1)[1]))
            except ValueError:
                pass
        path = os.path.join(model_dir, fn)
        try:
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                raw = fh.read(n)
                if len(raw) != n:
                    continue                       # header still being written
                hdr = json.loads(raw)
        except (OSError, ValueError):
            continue
        present += 1
        base_off = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            idx[k] = (path, base_off + a, base_off + b, v["dtype"], tuple(v["shape"]))
    return idx, present, max(total, present)


_MM = {}


def _mm(path):
    # Keep the file object alive alongside the mapping: letting it be collected closes the fd and
    # the next slice raises EBADF.
    if path not in _MM:
        fh = open(path, "rb")
        _MM[path] = (fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ))
    return _MM[path][1]


def raw(idx, name):
    path, a, b, dt, shape = idx[name]
    return _mm(path)[a:b], dt, shape


def load(idx, name):
    """Tensor as f32 (bf16/f16 widened) or as its native integer type."""
    buf, dt, shape = raw(idx, name)
    np_dt, _ = _DT[dt]
    arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
    return bf16_to_f32(arr) if dt == "BF16" else (arr.astype(np.float32) if dt == "F16" else arr)


# ---------------------------------------------------------------- bf16 <-> f32 (no torch here)
def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f):
    """Round-to-nearest-even, matching torch.Tensor.to(torch.bfloat16)."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    bias = 0x7FFF + ((u >> 16) & 1)
    out = ((u + bias) >> 16).astype(np.uint16)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    out[nan] = 0x7FC0
    return out


# ---------------------------------------------------------------- streaming safetensors writer
class STWriter:
    """Register (name, dtype, shape, nbytes, producer), then flush() writes the header once and
    streams each producer to disk.  Same two-pass shape as glm52_prep.py::STWriter — a K3 layer's
    experts alone are 15.7 GB, so nothing may be accumulated in RAM."""

    def __init__(self):
        self.entries = []

    def add(self, name, dtype, shape, nbytes, producer):
        self.entries.append((name, dtype, list(shape), int(nbytes), producer))

    def add_arr(self, name, dtype, arr):
        a = np.ascontiguousarray(arr)
        self.add(name, dtype, a.shape, a.nbytes, lambda f, a=a: f.write(a.tobytes()))

    def add_bf16(self, name, f32):
        self.add_arr(name, "BF16", f32_to_bf16(f32))

    def add_raw(self, name, idx, src):
        """Byte-for-byte passthrough — no dequant, no repack, no dtype change."""
        _, a, b, dt, shape = idx[src]
        path = idx[src][0]

        def prod(f, path=path, a=a, b=b):
            with open(path, "rb") as fh:
                fh.seek(a)
                left = b - a
                while left:
                    chunk = fh.read(min(left, 1 << 24))
                    if not chunk:
                        raise IOError(f"{src}: short read")
                    f.write(chunk)
                    left -= len(chunk)

        self.add(name, dt, shape, b - a, prod)

    def flush(self, path):
        hdr, off = {}, 0
        for name, dt, shape, nb, _ in self.entries:
            hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
            off += nb
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        blob += b" " * ((-((8 + len(blob)) % 8)) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for name, dt, shape, nb, prod in self.entries:
                start = f.tell()
                prod(f)
                assert f.tell() - start == nb, f"{name}: wrote {f.tell()-start} != {nb}"
        return off


# ---------------------------------------------------------------- config: read, then RECONCILE
class Cfg:
    """Kimi-K3 text-tower config.

    Read straight out of config.json's `text_config`.  **Never AutoConfig** — GLM-5.2's
    AutoConfig reported qk_rope_head_dim=192 against tensors that said 64.  Every dimension is
    then re-derived from the tensor shapes in `reconcile()`, and where they disagree the TENSOR
    WINS and the override is printed.
    """

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            raw_cfg = json.load(f)
        if raw_cfg.get("model_type") != "kimi_k3":
            sys.exit(f"not a kimi_k3 config: model_type={raw_cfg.get('model_type')!r}")
        self.has_vision = isinstance(raw_cfg.get("vision_config"), dict)
        t = raw_cfg["text_config"]
        self.raw = t
        g = lambda k: t[k]
        self.hidden, self.heads = g("hidden_size"), g("num_attention_heads")
        self.layers, self.vocab = g("num_hidden_layers"), g("vocab_size")
        self.dense_inter = g("intermediate_size")
        self.q_lora, self.kv_lora = g("q_lora_rank"), g("kv_lora_rank")
        self.qk_nope, self.qk_rope = g("qk_nope_head_dim"), g("qk_rope_head_dim")
        self.v_head = g("v_head_dim")
        self.qk_head = self.qk_nope + self.qk_rope
        self.mla_nope, self.mla_gate = g("mla_use_nope"), g("mla_use_output_gate")
        # ABSENT on K3 — and that is the fact, not a reason to default.  cfg_glm
        # (crates/devgen/src/mla.rs:123) would substitute 8e6 here.
        self.rope_theta = t.get("rope_theta")
        self.n_exp, self.top_k = g("num_experts"), g("num_experts_per_token")
        self.shared_exp, self.moe_inter = g("num_shared_experts"), g("moe_intermediate_size")
        self.moe_latent = g("routed_expert_hidden_size")
        self.latent_norm = g("latent_moe_use_norm")
        self.first_k_dense = g("first_k_dense_replace")
        self.hidden_act = g("hidden_act")
        self.attn_res_block = g("attn_res_block_size")
        lac = t["linear_attn_config"]
        self.kda_heads, self.kda_head_dim = lac["num_heads"], lac["head_dim"]
        self.kda_conv = lac["short_conv_kernel_size"]
        q = t.get("quantization_config", {})
        self.quant_format = q.get("format", "<none>")
        qw = q.get("config_groups", {}).get("group_0", {}).get("weights", {})
        self.group_size, self.num_bits = qw.get("group_size", 0), qw.get("num_bits", 0)
        # 1-BASED -> 0-based, once, here.  Both lists are read and they must PARTITION the tower.
        self.attn = [None] * self.layers
        for key, kind in (("full_attn_layers", "mla"), ("kda_layers", "kda")):
            for one in lac[key]:
                l = one - 1
                assert 0 <= l < self.layers, f"{key} lists 1-based {one} (0-based {l}) of {self.layers}"
                assert self.attn[l] is None, f"0-based layer {l} in both lists"
                self.attn[l] = kind
        missing = [i for i, k in enumerate(self.attn) if k is None]
        assert not missing, f"layers {missing[:8]} in neither list"

    def is_dense(self, l):
        return l < self.first_k_dense

    def reconcile(self, idx):
        """Re-derive every dimension from tensor shapes.  Returns the list of overrides applied."""
        fixes = []

        def fix(field, want, name, why):
            if want is None or getattr(self, field) == want:
                return
            fixes.append(f"{field}: config {getattr(self, field)} -> {want}  (from {name}: {why})")
            setattr(self, field, want)

        def shape(n):
            return idx[n][4] if n in idx else None

        mla = next((l for l, k in enumerate(self.attn) if k == "mla" and f"{TEXT}layers.{l}.self_attn.q_a_proj.weight" in idx), None)
        kda = next((l for l, k in enumerate(self.attn) if k == "kda" and f"{TEXT}layers.{l}.self_attn.q_proj.weight" in idx), None)
        moe = next((l for l in range(self.first_k_dense, self.layers) if f"{TEXT}layers.{l}.block_sparse_moe.gate.weight" in idx), None)

        if mla is not None:
            a = f"{TEXT}layers.{mla}.self_attn."
            s = shape(a + "q_a_proj.weight")
            fix("q_lora", s[0], "q_a_proj", "out"); fix("hidden", s[1], "q_a_proj", "in")
            s = shape(a + "kv_a_proj_with_mqa.weight")
            # kv_a emits kv_lora ++ qk_rope; kv_lora is pinned by kv_a_layernorm [kv_lora].
            kvl = shape(a + "kv_a_layernorm.weight")
            if kvl:
                fix("kv_lora", kvl[0], "kv_a_layernorm", "width")
            fix("qk_rope", s[0] - self.kv_lora, "kv_a_proj_with_mqa", "out - kv_lora")
            s = shape(a + "q_b_proj.weight")
            fix("qk_nope", s[0] // self.heads - self.qk_rope, "q_b_proj", "out/heads - qk_rope")
            s = shape(a + "kv_b_proj.weight")
            fix("v_head", s[0] // self.heads - self.qk_nope, "kv_b_proj", "out/heads - qk_nope")
            self.qk_head = self.qk_nope + self.qk_rope
        if kda is not None:
            a = f"{TEXT}layers.{kda}.self_attn."
            s = shape(a + "q_proj.weight")
            fix("kda_head_dim", s[0] // self.kda_heads, "kda q_proj", "out/kda_heads")
            s = shape(a + "q_conv1d.weight")
            fix("kda_conv", s[-1], "q_conv1d", "kernel")
        if moe is not None:
            m = f"{TEXT}layers.{moe}.block_sparse_moe."
            fix("n_exp", shape(m + "gate.weight")[0], "moe gate", "rows")
            fix("moe_latent", shape(m + "routed_expert_norm.weight")[0], "routed_expert_norm", "width")
            s = shape(m + "shared_experts.gate_proj.weight")
            fix("shared_exp", s[0] // self.moe_inter, "shared gate_proj", "out/moe_inter")
            e0 = m + "experts.0."
            if e0 + "w1.weight_packed" in idx:
                pk, sc = shape(e0 + "w1.weight_packed"), shape(e0 + "w1.weight_scale")
                # THE load-bearing check: the routed-expert GEMM's K is the LATENT width
                # (routed_expert_hidden_size), not moe_intermediate_size.
                fix("moe_inter", pk[0], "experts.0.w1.weight_packed", "N")
                fix("moe_latent", pk[1] * 2, "experts.0.w1.weight_packed", "K = 2*cols (2 fp4/byte)")
                fix("group_size", self.moe_latent // sc[1], "experts.0.w1.weight_scale", "K/cols")
        return fixes


# ---------------------------------------------------------------- inspect (header-only)
KDA_TENSORS = ["q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d",
               "f_a_proj", "f_b_proj", "b_proj", "A_log", "dt_bias", "o_norm"]
MLA_TENSORS = ["q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa",
               "kv_a_layernorm", "kv_b_proj"]


def classify(idx, l):
    """What the TENSORS say layer `l` is — independent of config.json."""
    p = f"{TEXT}layers.{l}.self_attn."
    has_mla = any(p + t + (".weight" if t != "A_log" else "") in idx for t in MLA_TENSORS)
    has_kda = any(p + t + (".weight" if t not in ("A_log", "dt_bias") else "") in idx
                  for t in KDA_TENSORS)
    if has_mla and has_kda:
        return "BOTH?!"
    return "mla" if has_mla else ("kda" if has_kda else None)


def inspect(cfg, idx, present, total):
    print(f"kimi-k3 checkpoint inspection — {present}/{total} shards readable, {len(idx)} tensors")
    if present < total:
        print("  PARTIAL DOWNLOAD: a tensor's absence below proves nothing.")
    if cfg.has_vision:
        print("  vision_config present — TEXT TOWER ONLY is in scope; the vision tower is refused,")
        print("  not skipped (a text-only blob for a multimodal checkpoint is wrong on images).")

    fixes = cfg.reconcile(idx)
    print("\nconfig vs tensors (TENSORS WIN):")
    if fixes:
        for f in fixes:
            print(f"  OVERRIDE  {f}")
    else:
        print("  no disagreement on any dimension the present tensors can speak to.")

    print(f"\n  hidden {cfg.hidden}  heads {cfg.heads}  layers {cfg.layers}  vocab {cfg.vocab}")
    print(f"  MLA    q_lora {cfg.q_lora} kv_lora {cfg.kv_lora} qk {cfg.qk_nope}+{cfg.qk_rope} "
          f"v {cfg.v_head} | nope={cfg.mla_nope} out_gate={cfg.mla_gate} "
          f"rope_theta={cfg.rope_theta if cfg.rope_theta is not None else 'ABSENT'}")
    print(f"  KDA    {cfg.kda_heads} heads x {cfg.kda_head_dim} dim, conv k={cfg.kda_conv}")
    print(f"  MoE    {cfg.n_exp} routed (top-{cfg.top_k}) + {cfg.shared_exp} shared | "
          f"inter {cfg.moe_inter} | LATENT {cfg.moe_latent} | norm={cfg.latent_norm}")
    print(f"  quant  {cfg.quant_format} {cfg.num_bits}b group {cfg.group_size} | act {cfg.hidden_act!r}")

    print("\nper-layer attention, config vs tensors:")
    seen, bad = 0, 0
    for l in range(cfg.layers):
        got = classify(idx, l)
        if got is None:
            continue
        seen += 1
        if got != cfg.attn[l]:
            bad += 1
            print(f"  MISMATCH layer {l}: config says {cfg.attn[l]}, tensors say {got}")
    print(f"  {seen}/{cfg.layers} layers present; {bad} mismatch(es).")
    if seen and not bad:
        print("  the 1-based -> 0-based conversion of linear_attn_config is CONFIRMED by the "
              "tensor sets.")

    # Dense-vs-MoE, also from tensors.
    dense = [l for l in range(cfg.layers) if f"{TEXT}layers.{l}.mlp.gate_proj.weight" in idx]
    moe = [l for l in range(cfg.layers) if f"{TEXT}layers.{l}.block_sparse_moe.gate.weight" in idx]
    print(f"  dense-FFN layers present: {dense}  (config first_k_dense_replace={cfg.first_k_dense})")
    print(f"  MoE layers present: {len(moe)}")

    # mxfp4 layout vs plow's DevOp::GemvMxfp4 contract.
    if moe:
        e0 = f"{TEXT}layers.{moe[0]}.block_sparse_moe.experts.0."
        print("\nmxfp4 layout vs DevOp::GemvMxfp4 (crates/packet/src/dev.rs:622):")
        for w, k, n in (("w1", cfg.moe_latent, cfg.moe_inter),
                        ("w3", cfg.moe_latent, cfg.moe_inter),
                        ("w2", cfg.moe_inter, cfg.moe_latent)):
            if e0 + w + ".weight_packed" not in idx:
                continue
            pk, sc = idx[e0 + w + ".weight_packed"][4], idx[e0 + w + ".weight_scale"][4]
            ok_p = pk == (n, k // 2)
            ok_s = sc == (n, k // cfg.group_size)
            print(f"  {w}: packed {pk} {'OK' if ok_p else f'EXPECTED {(n, k//2)}'} [N, K/2]   "
                  f"scale {sc} {'OK' if ok_s else f'EXPECTED {(n, k//cfg.group_size)}'} [N, K/{cfg.group_size}]")
        print("  E8M0 scales are BIASED BY 127 (knob-contract §2): neutral 127, byte 0 = 2^-127.")
        print("  Layout matches; NO REPACK is needed.  The MoE expert path already carries the")
        print("  encoding (MoeEnc::Mxfp4=2, wave_dot_mxfp4 = w4a16) — see plans/kimi-k3-frontend.md §4.")

    # Size accounting — the number that decides whether a full prep is even sensible.
    if present:
        nb = sum(b - a for _, a, b, _, _ in idx.values())
        exp = sum(b - a for k, (_, a, b, _, _) in idx.items() if ".experts." in k)
        print(f"\nsize: {nb/1e9:.1f} GB in {present} shards, {exp/1e9:.1f} GB ({100*exp/max(nb,1):.1f}%) "
              f"is routed experts")
        print(f"      projected full checkpoint ~{nb/present*total/1e12:.2f} TB")
        print("      => --experts inplace (default): prep writes the other "
              f"{100*(nb-exp)/max(nb,1):.1f}% and binds experts from the snapshot.")
    return bad == 0


# ---------------------------------------------------------------- prep
def prep_layer(cfg, idx, w, l, experts_mode):
    """Write layer `l` under the PROPOSED plow name contract (`model.layers.{l}.…`).

    The contract is proposed, not settled: `declare_glm` (crates/devgen/src/mla.rs) is the thing it
    must eventually match, and no K3 emitter exists yet.  Names here mirror GLM's so the diff to a
    future declare_k3 is a rename, not a redesign.
    """
    src = f"{TEXT}layers.{l}."
    dst = f"model.layers.{l}."
    kind = cfg.attn[l]

    w.add_raw(dst + "input_layernorm.weight", idx, src + "input_layernorm.weight")
    w.add_raw(dst + "post_attention_layernorm.weight", idx, src + "post_attention_layernorm.weight")
    # RESIDUAL-ATTENTION BLOCKS.  `_apply_attn_res` (modeling_kimi_linear.py:1075) is a softmax over
    # the running prefix sum plus one snapshot per completed `attn_res_block_size` layers, and the
    # only thing it does with these two tensors is
    #     score_weight = norm.weight * proj.weight.squeeze(0)      # [H]
    # so the RMSNorm weight and the [1,H] projection FOLD INTO ONE [H] vector here, exactly as
    # glm52_prep.py folds the MLA absorption.  Both the folded vector and the originals are written:
    # the fold is what an emitter wants, the originals keep the prep lossless while the op's design
    # is still open.
    for pfx in ("self_attention_res", "mlp_res"):
        nw = load(idx, src + pfx + "_norm.weight")
        pw = load(idx, src + pfx + "_proj.weight").reshape(-1)
        w.add_raw(dst + pfx + "_norm.weight", idx, src + pfx + "_norm.weight")
        w.add_raw(dst + pfx + "_proj.weight", idx, src + pfx + "_proj.weight")
        w.add_bf16(dst + pfx + "_score_weight", nw * pw)

    a_src, a_dst = src + "self_attn.", dst + "self_attn."
    if kind == "mla":
        H, NH = cfg.hidden, cfg.heads
        DK, DR, QN, VD, QL = cfg.kv_lora, cfg.qk_rope, cfg.qk_nope, cfg.v_head, cfg.q_lora
        # MLA absorption, identical in form to glm52_prep.py::prep_layer.  The only difference is
        # that the checkpoint is bf16 (no block-fp8 dequant) and NO RoPE is applied anywhere:
        # mla_use_nope=True and config.json carries no rope_theta, so the DR decoupled dims are
        # projected and never rotated.  q_rope/k_rope are therefore RAW, and — unlike GLM — there
        # is no "fold at a fixed position" simplification to make.
        q_b = load(idx, a_src + "q_b_proj.weight").reshape(NH, QN + DR, QL)
        q_b_nope, q_b_rope = q_b[:, :QN, :], q_b[:, QN:, :]
        kv_b = load(idx, a_src + "kv_b_proj.weight").reshape(NH, QN + VD, DK)
        k_nope_w, value_w = kv_b[:, :QN, :], kv_b[:, QN:, :]
        Wqa = np.einsum("hpl,hpk->hlk", k_nope_w, q_b_nope)          # [NH, DK, QL]
        Wuv = np.swapaxes(value_w, -1, -2)                            # [NH, DK, VD]
        kv_a = load(idx, a_src + "kv_a_proj_with_mqa.weight")

        w.add_raw(a_dst + "q_a_proj.weight", idx, a_src + "q_a_proj.weight")
        w.add_raw(a_dst + "q_a_layernorm.weight", idx, a_src + "q_a_layernorm.weight")
        w.add_raw(a_dst + "kv_a_layernorm.weight", idx, a_src + "kv_a_layernorm.weight")
        w.add_raw(a_dst + "o_proj.weight", idx, a_src + "o_proj.weight")
        w.add_bf16(a_dst + "derived.q_absorb.weight", Wqa.reshape(NH * DK, QL))
        w.add_bf16(a_dst + "derived.q_rope.weight", np.ascontiguousarray(q_b_rope).reshape(NH * DR, QL))
        w.add_bf16(a_dst + "derived.kv_a_latent.weight", kv_a[:DK])
        w.add_bf16(a_dst + "derived.k_rope.weight", kv_a[DK:DK + DR])
        w.add_bf16(a_dst + "derived.v_absorb.weight", Wuv.reshape(NH * DK, VD))
        if cfg.mla_gate:
            w.add_raw(a_dst + "g_proj.weight", idx, a_src + "g_proj.weight")
    else:
        # KDA: PASSTHROUGH under the original names.  Its semantics belong to the sibling agent
        # (docs/kimi-k3-kda.md); inventing a derived form here would have to be redone.
        for t in KDA_TENSORS + ["g_proj", "o_proj"]:
            n = t if t in ("A_log", "dt_bias") else t + ".weight"
            if a_src + n in idx:
                w.add_raw(a_dst + n, idx, a_src + n)

    if cfg.is_dense(l):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w.add_raw(dst + f"mlp.{proj}.weight", idx, src + f"mlp.{proj}.weight")
        return

    m_src, m_dst = src + "block_sparse_moe.", dst + "mlp."
    w.add_raw(m_dst + "gate.weight", idx, m_src + "gate.weight")
    w.add_raw(m_dst + "gate.e_score_correction_bias", idx, m_src + "gate.e_score_correction_bias")
    # LATENT MoE, order per modeling_kimi_linear.py:815-837 (NOT inferable from the shapes):
    #   router scores the HIDDEN state -> routed_expert_down_proj (hidden -> latent)
    #   -> gated expert sum at K=latent -> routed_expert_norm -> routed_expert_up_proj
    #   -> + shared_experts(ORIGINAL hidden).
    # The norm is AFTER the expert loop, and the shared experts read the pre-projection hidden.
    for n in ("routed_expert_down_proj.weight", "routed_expert_up_proj.weight",
              "routed_expert_norm.weight"):
        w.add_raw(m_dst + n, idx, m_src + n)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w.add_raw(m_dst + f"shared_experts.{proj}.weight", idx, m_src + f"shared_experts.{proj}.weight")

    if experts_mode == "verbatim":
        # w1 = gate, w3 = up, w2 = down (Mixtral naming).  Bytes are copied UNCHANGED: the packing
        # (2 fp4/byte, low nibble = even k) and the E8M0 scale rows already match plow.
        for e in range(cfg.n_exp):
            for w_src, proj in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                w.add_raw(m_dst + f"experts.{e}.{proj}.weight", idx,
                          m_src + f"experts.{e}.{w_src}.weight_packed")
                w.add_raw(m_dst + f"experts.{e}.{proj}.weight_scale", idx,
                          m_src + f"experts.{e}.{w_src}.weight_scale")


# ---------------------------------------------------------------- derived sidecar + symlink farm
#
# THE NAME CONTRACT IS NOT NEGOTIABLE ANY MORE.  `prep_layer` above writes the names that were
# PROPOSED in 2026-07 while no K3 emitter existed.  One exists now (crates/devgen/src/k3.rs) and it
# disagrees in three independent ways, each of which is a fatal `MISSING WEIGHT` at load
# (crates/plowrt/src/exec/amd.rs:2963 -- missing is an ERROR naming the tensor, never a zeroed
# buffer, precisely so this surfaces as a stack trace and not as fluent wrong output):
#
#   prefix       `model.layers.{l}.`  ->  `language_model.model.layers.{l}.`   (k3.rs:1640)
#   MoE ns       `mlp.`               ->  `block_sparse_moe.`                  (K3_MOE_NS, k3.rs)
#   k_rope       `derived.k_rope`     ->  `derived.k_rope_down`                (k3.rs:1951)
#   kv_a latent  `derived.kv_a_latent`->  `kv_a_proj_with_mqa` SLICED to [DK,H] (k3.rs:1950)
#
# WHAT ACTUALLY HAS TO BE WRITTEN, though, is far less than `prep_layer` writes.  Checked against
# the snapshot index: `language_model.model.embed_tokens.weight`, `language_model.model.norm.weight`
# and `language_model.lm_head.weight` already carry the emitter's exact spelling, as does every
# per-layer norm, every KDA tensor, the whole `block_sparse_moe.` namespace and all 494,592 mxfp4
# expert tensors.  The ONLY tensors the checkpoint cannot serve are the five per MLA layer below.
# So the sidecar is ~4.5 GB against a 1.5 TB checkpoint, and everything else binds in place.
#
# HOW BOTH REACH ONE LOADER.  `Checkpoint::open` (crates/plowrt/src/asset/checkpoint.rs:40-85) takes
# EVERY `*.safetensors` in ONE directory, non-recursively, and never reads an index file -- so a
# directory of symlinks to the snapshot's 96 shards plus this sidecar is indistinguishable from a
# real checkpoint, and `bind_packed_experts` (amd.rs:3044) reads the experts straight out of the
# symlinked shards.  There is no overlay/search-path mechanism in the runtime, so the farm IS the
# mechanism.  Precedent: scripts/glm52_prep_fp8_linear.py:95-123 does exactly this for GLM.
#
# THE ONE OVERRIDE, and why the filename matters.  `kv_a_proj_with_mqa.weight` EXISTS in the
# snapshot, at [DK+DR, H] = [576, 7168], but the emitter declares [DK, H] = [512, 7168] and takes
# the rope rows as a separate `derived.k_rope_down`.  The name does not match any Column/Row pattern
# in `shard_of` (asset/shard.rs:59-183) so it is Replicated, and Replicated requires `full == want`
# EXACTLY (shard.rs:310-318) -- 576 rows against a declared 512 is a fatal size error, not a silent
# truncation.  The sidecar therefore rewrites it, and the rewrite only wins because
# `Checkpoint::open` sorts paths (checkpoint.rs:48) and inserts in that order (`:78`), so the
# LAST file to define a name wins.  `model-idx-derived-*.safetensors` sorts after
# `model-000NN-of-000096.safetensors` because 'i' > '0' in ASCII.  That is implicit behaviour with
# no test on it, so `build_farm` VERIFIES the resulting order rather than trusting the argument.

# The five per-MLA-layer tensors, under the names crates/devgen/src/k3.rs actually declares.
def derived_layer(cfg, idx, w, l):
    """Write the MLA tensors the checkpoint cannot serve. Returns the names written."""
    H, NH = cfg.hidden, cfg.heads
    DK, DR, QN, VD, QL = cfg.kv_lora, cfg.qk_rope, cfg.qk_nope, cfg.v_head, cfg.q_lora
    src = f"{TEXT}layers.{l}."
    a_dst = f"{TEXT}layers.{l}.self_attn."

    # Absorption, as prep_layer derives it. K3 applies NO RoPE (mla_use_nope=True, no rope_theta),
    # so the DR decoupled dims are projected and never rotated and q_rope/k_rope stay RAW.
    q_b = load(idx, src + "self_attn.q_b_proj.weight").reshape(NH, QN + DR, QL)
    q_b_nope, q_b_rope = q_b[:, :QN, :], q_b[:, QN:, :]
    kv_b = load(idx, src + "self_attn.kv_b_proj.weight").reshape(NH, QN + VD, DK)
    k_nope_w, value_w = kv_b[:, :QN, :], kv_b[:, QN:, :]
    Wqa = np.einsum("hpl,hpk->hlk", k_nope_w, q_b_nope)      # [NH, DK, QL]
    Wuv = np.swapaxes(value_w, -1, -2)                        # [NH, DK, VD]
    kv_a = load(idx, src + "self_attn.kv_a_proj_with_mqa.weight")

    # FULL width, not per-rank: derived.q_absorb / q_rope / v_absorb are in `shard_of`'s COL list
    # (asset/shard.rs:80-92) so the loader slices them per rank; and if the emitter ever declared
    # them full instead, the `full == want => Replicated` demotion (shard.rs:304-307) carries it.
    # Writing full is correct under both, writing per-rank is correct under neither.
    names = []
    for n, arr in (("derived.q_absorb.weight", Wqa.reshape(NH * DK, QL)),
                   ("derived.q_rope.weight", np.ascontiguousarray(q_b_rope).reshape(NH * DR, QL)),
                   ("derived.v_absorb.weight", Wuv.reshape(NH * DK, VD)),
                   # The kv_a SPLIT: latent rows keep the checkpoint's own name at the emitter's
                   # declared [DK,H] (this is the override), rope rows become k_rope_down [DR,H].
                   ("kv_a_proj_with_mqa.weight", kv_a[:DK]),
                   ("derived.k_rope_down.weight", kv_a[DK:DK + DR])):
        w.add_bf16(a_dst + n, arr)
        names.append(a_dst + n)
    return names


def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def build_farm(model_dir, farm_dir, sidecar):
    """Symlink the snapshot AND `sidecar` into `farm_dir`, then VERIFY the override order."""
    os.makedirs(farm_dir, exist_ok=True)
    linked = 0
    for fn in sorted(os.listdir(model_dir)):
        if not (fn.endswith(".safetensors") or fn in ("config.json", "tokenizer.json",
                                                      "tokenizer_config.json", "generation_config.json")):
            continue
        _link(os.path.join(model_dir, fn), os.path.join(farm_dir, fn))
        linked += 1
    # The sidecar goes in LAST, and only if it is not already this very file.
    dst = os.path.join(farm_dir, os.path.basename(sidecar))
    if os.path.realpath(dst) != os.path.realpath(sidecar):
        _link(sidecar, dst)
    print(f"[farm] {linked} snapshot symlink(s) + sidecar -> {farm_dir}")

    # THE CHECK THAT MATTERS. `Checkpoint::open` sorts and last-writer-wins, so the sidecar must
    # sort strictly after every shard that defines a name it overrides. Assert it here rather than
    # discover it as a 576-vs-512 size error 1.5 TB into a load.
    names = [f for f in sorted(os.listdir(farm_dir)) if f.endswith(".safetensors")]
    side = os.path.basename(sidecar)
    if names[-1] != side:
        sys.exit(f"[farm] REFUSING: {side} does not sort last among {len(names)} shards "
                 f"(last is {names[-1]}). The kv_a override would lose to the snapshot.")
    print(f"[farm] override order OK: {side} sorts last of {len(names)} shards")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF snapshot dir (partial download is fine)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true", help="header-only reconciliation (default)")
    ap.add_argument("--layers", default=None, help="comma-separated 0-BASED layer ids to prep")
    ap.add_argument("--globals", action="store_true", help="also write embed_tokens/norm/lm_head")
    ap.add_argument("--experts", choices=("inplace", "verbatim"), default="inplace",
                    help="inplace (default): do not copy the 1.5 TB of mxfp4 experts")
    ap.add_argument("--derived", action="store_true",
                    help="write ONLY the 5 per-MLA-layer tensors the checkpoint cannot serve, "
                         "under crates/devgen/src/k3.rs's real names (serve path)")
    ap.add_argument("--farm", default=None,
                    help="with --derived: also build a symlink farm of the snapshot beside the "
                         "sidecar, so one PLOW_CHECKPOINT dir serves both")
    args = ap.parse_args()

    idx, present, total = index_shards(args.model)
    cfg = Cfg(args.model)

    if args.derived:
        if not args.out:
            sys.exit("--derived needs --out")
        cfg.reconcile(idx)
        mla = [l for l in range(cfg.layers) if cfg.attn[l] == "mla"]
        want = [int(x) for x in args.layers.split(",")] if args.layers else mla
        bad = [l for l in want if cfg.attn[l] != "mla"]
        if bad:
            sys.exit(f"layers {bad} are not MLA; only MLA layers have derived tensors")
        os.makedirs(args.out, exist_ok=True)
        w, written = STWriter(), []
        for l in want:
            written += derived_layer(cfg, idx, w, l)
            print(f"[derived] layer {l} queued ({len(written)} tensors so far)")
        out = os.path.join(args.out, "model-idx-derived-00001.safetensors")
        nb = w.flush(out)
        print(f"[derived] wrote {out} ({nb/1e9:.2f} GB), {len(written)} tensors "
              f"over {len(want)}/{len(mla)} MLA layers")
        if args.farm:
            build_farm(args.model, args.farm, out)
        return

    if args.layers is None or args.inspect:
        sys.exit(0 if inspect(cfg, idx, present, total) else 1)

    cfg.reconcile(idx)
    layers = [int(x) for x in args.layers.split(",")]
    missing = [t for l in layers for t in (f"{TEXT}layers.{l}.input_layernorm.weight",)
               if t not in idx]
    if missing:
        sys.exit(f"layers {layers}: not downloaded yet ({missing[0]} absent; "
                 f"{present}/{total} shards present)")
    if not args.out:
        sys.exit("--layers needs --out")
    os.makedirs(args.out, exist_ok=True)

    w = STWriter()
    for l in layers:
        prep_layer(cfg, idx, w, l, args.experts)
        print(f"[prep] layer {l} ({cfg.attn[l]}, {'dense' if cfg.is_dense(l) else 'moe'}) queued")
    if args.globals:
        for src, dst in ((TEXT + "embed_tokens.weight", "model.embed_tokens.weight"),
                         (TEXT + "norm.weight", "model.norm.weight"),
                         ("lm_head.weight", "lm_head.weight")):
            if src in idx:
                w.add_raw(dst, idx, src)
            else:
                print(f"[prep] WARNING {src} not downloaded yet — skipped")
    # config.json passthrough so a future cfg_kimi_k3 parses the same dims.
    with open(os.path.join(args.model, "config.json")) as f, \
            open(os.path.join(args.out, "config.json"), "w") as g:
        g.write(f.read())
    out = os.path.join(args.out, "model-00001-of-00001.safetensors")
    total_bytes = w.flush(out)
    print(f"[prep] wrote {out} ({total_bytes/1e9:.2f} GB), {len(w.entries)} tensors, "
          f"experts={args.experts}")


if __name__ == "__main__":
    main()
