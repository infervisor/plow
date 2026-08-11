#!/usr/bin/env python3
"""Small CPU regression for sharded + fused-3D Gemma expert FP8 quantization."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
QUANTIZER = os.path.join(HERE, "quantize_fp8.py")
PREFIX = "model.language_model.layers.0."


def raw_bytes(t):
    return bytes(t.contiguous().view(torch.uint8).flatten().tolist())


def write_shard(path, tensors):
    meta, payload, off = {}, bytearray(), 0
    for name, tensor in tensors.items():
        data = raw_bytes(tensor)
        meta[name] = {"dtype": "BF16", "shape": list(tensor.shape),
                      "data_offsets": [off, off + len(data)]}
        payload.extend(data)
        off += len(data)
    header = json.dumps(meta, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(payload)


def read_st(path):
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hn))
        data0 = 8 + hn
        f.seek(0)
        raw = f.read()
    return meta, raw, data0


class QuantizeFp8Test(unittest.TestCase):
    def test_sharded_fused_experts_keep_shapes_and_row_scales(self):
        torch.manual_seed(7)
        q = (torch.randn(3, 8) * 0.3).to(torch.bfloat16)
        gu = (torch.randn(2, 4, 8) * 0.3).to(torch.bfloat16)
        dn = (torch.randn(2, 8, 4) * 0.3).to(torch.bfloat16)
        names = {
            PREFIX + "self_attn.q_proj.weight": q,
            PREFIX + "experts.gate_up_proj": gu,
            PREFIX + "experts.down_proj": dn,
        }
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
            write_shard(os.path.join(src, s1), dict(list(names.items())[:2]))
            write_shard(os.path.join(src, s2), dict(list(names.items())[2:]))
            weight_map = {name: (s1 if i < 2 else s2) for i, name in enumerate(names)}
            with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)
            subprocess.check_call([sys.executable, QUANTIZER, src, out])

            meta, raw, data0 = read_st(os.path.join(out, "model.safetensors"))
            gu_name = "fp8/" + PREFIX + "experts.gate_up_proj"
            dn_name = "fp8/" + PREFIX + "experts.down_proj"
            self.assertEqual(meta[gu_name]["shape"], [2, 4, 8])
            self.assertEqual(meta[gu_name + "_scale"]["shape"], [2, 4])
            self.assertEqual(meta[dn_name]["shape"], [2, 8, 4])
            self.assertEqual(meta[dn_name + "_scale"]["shape"], [2, 8])
            self.assertEqual(len(meta), 6)

            # Decode the emitted expert bytes and bound each row's error by half of the largest
            # e4m3 bin (16 normalized units) times that row's scale.
            for src_tensor, out_name in [(gu, gu_name), (dn, dn_name)]:
                wa, we = meta[out_name]["data_offsets"]
                sa, se = meta[out_name + "_scale"]["data_offsets"]
                q8 = torch.frombuffer(bytearray(raw[data0 + wa:data0 + we]), dtype=torch.uint8)
                q8 = q8.view(torch.float8_e4m3fn).float().view(*src_tensor.shape)
                scale = torch.frombuffer(bytearray(raw[data0 + sa:data0 + se]),
                                         dtype=torch.float32).view(*src_tensor.shape[:-1])
                dequant = q8 * scale.unsqueeze(-1)
                err = (dequant - src_tensor.float()).abs().amax(dim=-1)
                self.assertTrue(torch.all(err <= scale * 16.01 + 1e-7), (err, scale))


if __name__ == "__main__":
    unittest.main()
