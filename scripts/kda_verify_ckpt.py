#!/usr/bin/env python3
"""Verify the KDA checkpoint invariants that silently corrupt if wrong.

Reads safetensors HEADERS for all 93 layers plus the 512 B of `A_log` and the
48 KiB of `dt_bias` per KDA layer. Never touches a projection matrix.

Checks, per `docs/kimi-k3-kda.md`:
  1. `kda_layers` / `full_attn_layers` are 1-based and partition 1..93.
  2. The layer classification implied by the tensors matches the config list
     (KDA <=> `self_attn.A_log` present; MLA <=> `self_attn.q_a_proj` present).
  3. Every KDA layer ships the same 14 tensors with the shapes of section 6.1.
  4. `A_log` is `[128]` with exactly indices 0..95 non-zero (per-head [96],
     zero-padded to head_dim) -- in EVERY KDA layer, not a sample.
  5. `dt_bias` is `[12288]` with no zero padding.
"""
import json
import os
import struct
import sys

SNAP = os.environ.get(
    "K3_SNAP",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)

KDA_TENSORS = {
    "q_proj.weight": ("BF16", [12288, 7168]),
    "k_proj.weight": ("BF16", [12288, 7168]),
    "v_proj.weight": ("BF16", [12288, 7168]),
    "g_proj.weight": ("BF16", [12288, 7168]),
    "o_proj.weight": ("BF16", [7168, 12288]),
    "f_a_proj.weight": ("BF16", [128, 7168]),
    "f_b_proj.weight": ("BF16", [12288, 128]),
    "b_proj.weight": ("BF16", [96, 7168]),
    "q_conv1d.weight": ("F32", [12288, 1, 4]),
    "k_conv1d.weight": ("F32", [12288, 1, 4]),
    "v_conv1d.weight": ("F32", [12288, 1, 4]),
    "A_log": ("F32", [128]),
    "dt_bias": ("F32", [12288]),
    "o_norm.weight": ("F32", [128]),
}


class Shards:
    def __init__(self, snap):
        self.snap = snap
        self.hdr = {}
        self.fh = {}

    def head(self, shard):
        if shard not in self.hdr:
            f = open(os.path.join(self.snap, shard), "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr[shard] = (json.loads(f.read(n)), 8 + n)
            self.fh[shard] = f
        return self.hdr[shard]

    def f32(self, shard, key):
        h, base = self.head(shard)
        m = h[key]
        assert m["dtype"] == "F32", m["dtype"]
        lo, hi = m["data_offsets"]
        f = self.fh[shard]
        f.seek(base + lo)
        raw = f.read(hi - lo)
        return struct.unpack(f"<{(hi - lo) // 4}f", raw)


def main():
    fail = []
    with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    with open(os.path.join(SNAP, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    la = cfg["linear_attn_config"]
    nl = cfg["num_hidden_layers"]
    kda1, full1 = set(la["kda_layers"]), set(la["full_attn_layers"])

    # 1. 1-based partition.
    if kda1 | full1 != set(range(1, nl + 1)) or kda1 & full1:
        fail.append(f"lists do not partition 1..{nl}")
    print(f"[1] 1-based lists partition 1..{nl}: KDA {len(kda1)}, MLA {len(full1)}")
    kda0 = {i - 1 for i in kda1}
    tail = "".join("K" if i in kda0 else "M" for i in range(nl - 5, nl))
    print(f"[1] 0-based tail (layers {nl-5}..{nl-1}) = {tail}"
          f"  {'OK: KKK MM, NOT i%4==3' if tail == 'KKKMM' else 'UNEXPECTED'}")
    if tail != "KKKMM":
        fail.append("tail is not KKKMM")
    bad_mod = sorted(i for i in range(nl) if (i % 4 == 3) != (i not in kda0))
    print(f"[1] layers where `i%4==3` disagrees with the config: {bad_mod}")

    sh = Shards(SNAP)
    n_alog = n_dt = 0
    for li in range(nl):
        pre = f"language_model.model.layers.{li}.self_attn."
        keys = {k[len(pre):]: wm[k] for k in wm if k.startswith(pre)}
        is_kda_t = "A_log" in keys
        is_mla_t = "q_a_proj.weight" in keys
        # 2. tensors vs config
        if is_kda_t != (li in kda0):
            fail.append(f"layer {li}: A_log presence {is_kda_t} != config KDA {li in kda0}")
        if is_mla_t == is_kda_t:
            fail.append(f"layer {li}: ambiguous (A_log={is_kda_t}, q_a_proj={is_mla_t})")
        if not is_kda_t:
            continue
        # 3. shapes
        if set(keys) != set(KDA_TENSORS):
            fail.append(f"layer {li}: tensor set {sorted(set(keys) ^ set(KDA_TENSORS))}")
        for name, (dt, shape) in KDA_TENSORS.items():
            h, _ = sh.head(keys[name])
            m = h[pre + name]
            if m["dtype"] != dt or m["shape"] != shape:
                fail.append(f"layer {li}.{name}: {m['dtype']} {m['shape']} != {dt} {shape}")
        # 4. A_log padding
        a = sh.f32(keys["A_log"], pre + "A_log")
        nz = [i for i, x in enumerate(a) if x != 0.0]
        if len(a) != 128 or nz != list(range(96)):
            fail.append(f"layer {li}: A_log nonzero pattern {nz[:3]}..{nz[-3:]} len {len(nz)}")
        else:
            n_alog += 1
        # 5. dt_bias no padding
        d = sh.f32(keys["dt_bias"], pre + "dt_bias")
        if len(d) != 12288 or any(x == 0.0 for x in d):
            fail.append(f"layer {li}: dt_bias has zeros or wrong length {len(d)}")
        else:
            n_dt += 1

    print(f"[2] tensor-implied classification matches the config for all {nl} layers")
    print(f"[3] all {n_alog} KDA layers ship the 14 tensors of section 6.1 at the spec shapes")
    print(f"[4] A_log = [128] f32, nonzero exactly on 0..95, in {n_alog}/{len(kda0)} KDA layers")
    print(f"[5] dt_bias = [12288] f32, zero zeros,      in {n_dt}/{len(kda0)} KDA layers")
    if fail:
        print(f"\nFAIL ({len(fail)}):")
        for m in fail[:40]:
            print("  " + m)
        return 1
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
