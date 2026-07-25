#!/usr/bin/env python3
"""KV-1 host oracle for the frozen KVZIP-SZ12 v1.2 row blob (plans/p10-kv-zip.md).

Layout (per kv-head, per ring row, hd elems; row_bytes = hd/2 + hd + 32):
  [0        .. hd/2)     code plane: 4-bit exponent codes, 2/byte,
                         LOW nibble = even dim, HIGH nibble = odd dim
  [hd/2     .. hd/2+hd)  lo plane: 1 B/dim = sign<<7 | mant7
  [hd/2+hd  .. +4)       header u32 LE = base | nesc<<8   (u8 each, rsv u16)
  [hd/2+hd+4.. +31)      9 escape slots, 3 B each = {exp u8, dim_idx u16 LE},
                         ascending dim order, unused = FF FF FF; [+31..+32) pad
  code c in [0,14]: exp = base + c.  code 15: escape -> exp from slot.
  base: PER-ROW, chosen by the encoder as the start of the 15-value window
  covering the most of that row's exponents (v1's per-tensor base FAILED the
  oracle: rows dominated by small values cluster escapes — max 13/row on
  sliding.V; per-row optimal windows measure max 4-5/row, no model-side
  constants, no cross-domain dependence). v1.1's 7 u32 slots overflowed on
  ONE row of 10.5 GB (kv.33.k code dump, 8 escapes) -> v1.2 packs 9 3-byte
  slots in the same 32 B tail (measured worst row anywhere: 8).
  hd512 -> 800 B/row (1.280x). hd256 -> 416 B/row (1.2308x). 16 B-aligned.
  Only valid ring rows are ever encoded (encode at append). >9 escapes/row
  is unencodable -> oracle FATAL.

Checks, over every kv.* tensor of a dump:
  1. encode -> serialized blob -> decode == original bytes (bit-exact).
  2. escape count per row <= 9.
  3. negative controls (first tensor): corrupt lo byte / code nibble /
     used escape slot / header base -> decode must NOT round-trip.

Usage: kvzip_oracle.py <dump-dir> [--out PREFIX]
"""
import argparse, json, os, time
import torch

torch.set_num_threads(max(8, os.cpu_count() // 4))
E_SLOTS = 9
UNUSED = -1  # 0xFFFFFFFF as int32


def row_best_base(exp):
    """exp [N, hd] -> per-row start of the best 15-value window, [N, 1]."""
    N = exp.shape[0]
    hist = torch.zeros(N, 256, dtype=torch.int16)
    hist.scatter_add_(1, exp.to(torch.int64), torch.ones_like(exp, dtype=torch.int16))
    c = hist.to(torch.int32).cumsum(dim=1)
    win = c[:, 14:] - torch.nn.functional.pad(c, (1, 0))[:, :242]
    return win.argmax(dim=1, keepdim=True).to(torch.int32)


def encode(x_u16):
    """[N, hd] u16 -> ([N, row_bytes] u8 blob, max escapes/row)."""
    N, hd = x_u16.shape
    xi = x_u16.to(torch.int32)
    exp = (xi >> 7) & 0xFF
    lo = (((xi >> 8) & 0x80) | (xi & 0x7F)).to(torch.uint8)
    base = row_best_base(exp)
    c = exp - base
    esc = (c < 0) | (c > 14)
    codes = torch.where(esc, 15, c)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)

    nesc = esc.sum(dim=1, dtype=torch.int32)
    max_esc = int(nesc.max()) if N else 0
    if max_esc > E_SLOTS:
        return None, max_esc
    hdr = (base.reshape(N) | (nesc << 8)).to(torch.int32).reshape(N, 1)
    slots = torch.full((N, E_SLOTS, 3), 0xFF, dtype=torch.uint8)
    nz = esc.nonzero()
    if nz.numel():
        pos = (esc.cumsum(1) - 1)[nz[:, 0], nz[:, 1]]
        slots[nz[:, 0], pos, 0] = exp[nz[:, 0], nz[:, 1]].to(torch.uint8)
        slots[nz[:, 0], pos, 1] = (nz[:, 1] & 0xFF).to(torch.uint8)
        slots[nz[:, 0], pos, 2] = (nz[:, 1] >> 8).to(torch.uint8)
    pad = torch.zeros(N, 1, dtype=torch.uint8)
    blob = torch.cat([packed, lo, hdr.view(torch.uint8).reshape(N, 4),
                      slots.reshape(N, 3 * E_SLOTS), pad], dim=1)
    return blob, max_esc


def decode(blob, hd):
    N = blob.shape[0]
    packed = blob[:, : hd // 2].to(torch.int32)
    lo = blob[:, hd // 2: hd // 2 + hd].to(torch.int32)
    tail = blob[:, hd // 2 + hd:]
    base = tail[:, :4].contiguous().view(torch.int32).reshape(N, 1) & 0xFF
    slots = tail[:, 4: 4 + 3 * E_SLOTS].reshape(N, E_SLOTS, 3).to(torch.int32)
    codes = torch.empty(N, hd, dtype=torch.int32)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    exp = base + codes
    idx = slots[:, :, 1] | (slots[:, :, 2] << 8)
    used = idx != 0xFFFF
    if used.any():
        nz = used.nonzero()
        exp[nz[:, 0], idx[nz[:, 0], nz[:, 1]]] = slots[nz[:, 0], nz[:, 1], 0]
    return (((lo & 0x80) << 8) | (exp << 7) | (lo & 0x7F)).to(torch.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="perf-data/kv1-oracle")
    ap.add_argument("--full-geom", default="1,32768,512")
    ap.add_argument("--slide-geom", default="8,16384,256")
    args = ap.parse_args()

    fk, fr, fh = map(int, args.full_geom.split(","))
    sk, sr, sh = map(int, args.slide_geom.split(","))
    sizes = {fk * fr * fh * 2: ("full", fk, fr, fh),
             sk * sr * sh * 2: ("sliding", sk, sr, sh)}

    man, ctx = {}, None
    for line in open(os.path.join(args.dump, "manifest.txt")):
        p = line.split()
        if p[0] == "ctx":
            ctx = int(p[1])
        else:
            man[p[0]] = int(p[1])

    res = {"dump": args.dump, "ctx": ctx, "layout": "KVZIP-SZ12 v1.2 (per-row base, 9x3B slots)",
           "tensors": 0, "roundtrip_fail": [], "fatal_overflow": [],
           "max_esc_per_row": 0, "bytes_raw": 0, "bytes_blob": 0,
           "per_class": {}, "controls": None}
    t0 = time.time()
    controls_done = False
    for name, nbytes in sorted(man.items()):
        if nbytes not in sizes:
            continue
        ltype, kvh, ring, hd = sizes[nbytes]
        rows = min(ctx, ring)
        x = torch.from_file(os.path.join(args.dump, name + ".raw"),
                            dtype=torch.uint16, size=kvh * ring * hd)
        x = x.reshape(kvh, ring, hd)[:, :rows, :].reshape(-1, hd).contiguous()

        blob, max_esc = encode(x)
        if blob is None:
            res["fatal_overflow"].append({"tensor": name, "max_esc": max_esc})
            print(f"  {name}: FATAL escape overflow {max_esc} > {E_SLOTS}", flush=True)
            continue
        ok = torch.equal(decode(blob, hd), x)
        if not ok:
            res["roundtrip_fail"].append(name)
        res["tensors"] += 1
        res["max_esc_per_row"] = max(res["max_esc_per_row"], max_esc)
        res["bytes_raw"] += x.numel() * 2
        res["bytes_blob"] += blob.numel()
        pc = res["per_class"].setdefault(ltype, {"raw": 0, "blob": 0, "max_esc": 0})
        pc["raw"] += x.numel() * 2
        pc["blob"] += blob.numel()
        pc["max_esc"] = max(pc["max_esc"], max_esc)
        print(f"  {name} rt={'OK' if ok else 'FAIL'} max_esc={max_esc} "
              f"t={time.time()-t0:.0f}s", flush=True)

        if not controls_done:
            hd2 = hd // 2
            ctl = {}
            b2 = blob.clone(); b2[0, hd2 + 3] ^= 0x40                # lo byte
            ctl["lo_corrupt_detected"] = not torch.equal(decode(b2, hd), x)
            b3 = blob.clone()
            cand = (b3[:, :hd2].to(torch.int32) & 0xF < 14) \
                 & ((b3[:, :hd2].to(torch.int32) >> 4) & 0xF < 14)
            r, dcol = cand.nonzero()[0]
            b3[r, dcol] ^= 0x01                                      # code nibble
            ctl["code_corrupt_detected"] = not torch.equal(decode(b3, hd), x)
            b5 = blob.clone(); b5[0, hd2 + hd] ^= 0x01               # header base
            ctl["hdr_corrupt_detected"] = not torch.equal(decode(b5, hd), x)
            sl = blob[:, hd2 + hd + 4: hd2 + hd + 4 + 3 * E_SLOTS].reshape(-1, E_SLOTS, 3)
            urows = ((sl[:, :, 1].to(torch.int32) | (sl[:, :, 2].to(torch.int32) << 8))
                     != 0xFFFF).any(dim=1).nonzero()
            if urows.numel():
                r = int(urows[0]); b4 = blob.clone()
                b4[r, hd2 + hd + 4] ^= 0x01                          # slot exp byte
                ctl["slot_corrupt_detected"] = not torch.equal(decode(b4, hd), x)
            else:
                ctl["slot_corrupt_detected"] = "no-escape-row-in-tensor"
            res["controls"] = ctl
            controls_done = True

    for pc in res["per_class"].values():
        pc["net_ratio"] = pc["raw"] / pc["blob"]
    res["net_ratio_total"] = res["bytes_raw"] / max(1, res["bytes_blob"])
    res["PASS"] = (not res["roundtrip_fail"] and not res["fatal_overflow"]
                   and res["controls"] and
                   all(v is True for v in res["controls"].values()
                       if isinstance(v, bool)))
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
