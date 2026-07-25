#!/usr/bin/env python3
"""Reference SplitZip codec: encode/decode a tile, verify BIT-EXACT (memcmp).

Also serves as the negative control: corrupting the code table or an escape
entry must break the memcmp. If corruption does NOT break it, the test is
worthless and we say so.

Layout per tile (BN*BK elements, tile-local table):
  code_table : K entries x EXP_BITS      (compile-time constant in the packet)
  codes      : BN*BK * CODE_BITS         (packed, fixed-length, no branching)
  payload    : BN*BK * PAYLOAD_BITS      (sign+mantissa, raw)
  escapes    : n_esc * (POS_BITS + EXP_BITS)
Decode: payload | (table[code] << shift), then scatter escape exponents. The
scatter is the ONLY data-dependent step and it is a sparse overwrite, not a
per-element branch.
"""
import sys
import numpy as np
import torch
from safetensors import safe_open

POS_BITS = 10


class Fmt:
    def __init__(self, kind):
        if kind == "bf16":
            self.bits, self.exp_bits, self.exp_shift = 16, 8, 7
        else:
            self.bits, self.exp_bits, self.exp_shift = 8, 4, 3
        self.payload_bits = self.bits - self.exp_bits
        self.exp_mask = (1 << self.exp_bits) - 1
        self.payload_mask = (1 << self.payload_bits) - 1
        self.udt = np.uint16 if self.bits == 16 else np.uint8

    def split(self, u):
        e = (u.astype(np.uint32) >> self.exp_shift) & self.exp_mask
        # payload = sign bit (top) + mantissa (low exp_shift bits)
        sign = (u.astype(np.uint32) >> (self.bits - 1)) & 1
        mant = u.astype(np.uint32) & ((1 << self.exp_shift) - 1)
        p = (sign << self.exp_shift) | mant
        return e.astype(np.uint8), p.astype(np.uint32)

    def join(self, e, p):
        sign = (p >> self.exp_shift) & 1
        mant = p & ((1 << self.exp_shift) - 1)
        u = (sign << (self.bits - 1)) | (e.astype(np.uint32) << self.exp_shift) | mant
        return u.astype(self.udt)


def encode_tile(u, fmt, k):
    """u: 1-D array of raw uint elements for one tile."""
    e, p = fmt.split(u)
    h = np.bincount(e, minlength=1 << fmt.exp_bits)
    table = np.sort(np.argsort(h, kind="stable")[::-1][:k]).astype(np.uint8)
    lut = np.full(1 << fmt.exp_bits, 0, dtype=np.uint8)   # default code 0
    inv = np.full(1 << fmt.exp_bits, False)
    for c, v in enumerate(table):
        lut[v] = c
        inv[v] = True
    codes = lut[e]
    esc_pos = np.nonzero(~inv[e])[0].astype(np.uint32)
    esc_val = e[esc_pos]
    return dict(table=table, codes=codes, payload=p, esc_pos=esc_pos,
                esc_val=esc_val, n=u.size)


def decode_tile(enc, fmt):
    # 1. uniform, divergence-free table lookup for EVERY element
    e = enc["table"][enc["codes"]].astype(np.uint8)
    # 2. sparse escape overwrite
    e[enc["esc_pos"]] = enc["esc_val"]
    return fmt.join(e, enc["payload"])


def stored_bits(enc, fmt, k, code_bits):
    return (k * fmt.exp_bits + 16
            + enc["n"] * (code_bits + fmt.payload_bits)
            + enc["esc_pos"].size * (POS_BITS + fmt.exp_bits))


def run(path, tname, kind, ntiles=64):
    fmt = Fmt(kind)
    with safe_open(path, framework="pt") as sf:
        t = sf.get_tensor(tname)
    u = t.view(torch.int16 if fmt.bits == 16 else torch.int8).numpy().view(fmt.udt)
    BN, BK = 128, 64
    N, K = u.shape
    tv = u.reshape(N // BN, BN, K // BK, BK).transpose(0, 2, 1, 3).reshape(-1, BN * BK)
    tv = tv[:ntiles]
    print(f"\n{kind}  {tname}  shape={u.shape}  testing {tv.shape[0]} tiles of {BN*BK}")

    for k, cb in ((8, 3), (16, 4)):
        tot_raw = tot_cmp = 0
        bad = 0
        for i in range(tv.shape[0]):
            orig = np.ascontiguousarray(tv[i])
            enc = encode_tile(orig, fmt, k)
            dec = decode_tile(enc, fmt)
            if orig.tobytes() != dec.tobytes():
                bad += 1
            tot_raw += orig.size * fmt.bits
            tot_cmp += stored_bits(enc, fmt, k, cb)
        print(f"  top-{k:<3}/{cb}b: bit-exact {tv.shape[0]-bad}/{tv.shape[0]} tiles"
              f"   ratio={tot_raw/tot_cmp:.4f}x")
        if bad:
            print(f"  !!! {bad} MISMATCHES")

    # ---------- negative control ----------
    print("  negative control (must all report MISMATCH):")
    orig = np.ascontiguousarray(tv[0])
    enc = encode_tile(orig, fmt, 16)
    assert orig.tobytes() == decode_tile(enc, fmt).tobytes(), "clean roundtrip broken"

    import copy
    e1 = copy.deepcopy(enc); e1["table"][3] ^= 1
    print(f"    corrupt code table entry -> {'MISMATCH' if decode_tile(e1,fmt).tobytes()!=orig.tobytes() else 'PASSED (TEST IS BROKEN)'}")
    if enc["esc_pos"].size:
        e2 = copy.deepcopy(enc); e2["esc_val"][0] ^= 1
        print(f"    corrupt escape value     -> {'MISMATCH' if decode_tile(e2,fmt).tobytes()!=orig.tobytes() else 'PASSED (TEST IS BROKEN)'}")
        e3 = copy.deepcopy(enc); e3["esc_pos"] = e3["esc_pos"][1:]; e3["esc_val"] = e3["esc_val"][1:]
        print(f"    drop one escape entry    -> {'MISMATCH' if decode_tile(e3,fmt).tobytes()!=orig.tobytes() else 'PASSED (TEST IS BROKEN)'}")
    else:
        print("    (tile 0 has zero escapes; forcing one)")
        e2 = copy.deepcopy(enc); e2["table"][0] ^= 2
        print(f"    corrupt table entry 0    -> {'MISMATCH' if decode_tile(e2,fmt).tobytes()!=orig.tobytes() else 'PASSED (TEST IS BROKEN)'}")
    e4 = copy.deepcopy(enc); e4["codes"][7] = (e4["codes"][7] + 1) % 16
    print(f"    corrupt one code         -> {'MISMATCH' if decode_tile(e4,fmt).tobytes()!=orig.tobytes() else 'PASSED (TEST IS BROKEN)'}")
    e5 = copy.deepcopy(enc); e5["payload"][11] ^= 1
    print(f"    corrupt one payload bit  -> {'MISMATCH' if decode_tile(e5,fmt).tobytes()!=orig.tobytes() else 'PASSED (TEST IS BROKEN)'}")


if __name__ == "__main__":
    run("/root/models/Qwen3-4B/model-00001-of-00003.safetensors",
        "model.layers.2.mlp.gate_proj.weight", "bf16")
    run("/root/models/Qwen3-4B/model-00001-of-00003.safetensors",
        "model.layers.5.self_attn.q_proj.weight", "bf16")
    run("/workspace/models/gemma-4-31B-it-fp8/model-00001-of-00001.safetensors",
        "model.language_model.layers.1.mlp.gate_proj.weight", "e4m3")
