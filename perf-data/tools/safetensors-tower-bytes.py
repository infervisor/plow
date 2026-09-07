import json, collections, struct, os

CK = "/home/lava/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52"

DT = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
      "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}


def header(path):
    """Read only the JSON header of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


bytes_by = collections.Counter()
count_by = collections.Counter()

files = sorted(f for f in os.listdir(CK) if f.endswith(".safetensors"))
for fn in files:
    h = header(os.path.join(CK, fn))
    for k, meta in h.items():
        if k == "__metadata__":
            continue
        shp = meta["shape"]
        n = 1
        for d in shp:
            n *= d
        b = n * DT.get(meta["dtype"], 2)
        if "vision" in k:
            g = "vision_tower"
        elif "audio" in k:
            g = "audio_tower"
        elif "embed_tokens" in k:
            g = "embed_tokens (tied lm_head)"
        elif "experts" in k:
            g = "moe experts"
        else:
            g = "language_model other"
        bytes_by[g] += b
        count_by[g] += 1

tot = sum(bytes_by.values())
print(f"{'group':32s} {'tensors':>8s} {'GiB':>9s}")
for g, b in bytes_by.most_common():
    print(f"{g:32s} {count_by[g]:8d} {b / 2**30:9.2f}")
print(f"{'TOTAL':32s} {sum(count_by.values()):8d} {tot / 2**30:9.2f}")

vis = bytes_by["vision_tower"] + bytes_by["audio_tower"]
print(f"\ntext-only (no vision/audio): {(tot - vis) / 2**30:.2f} GiB")
print(f"vision+audio towers        : {vis / 2**30:.2f} GiB")
