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



import contextlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch
from safetensors.torch import load_file, save_file

spec = importlib.util.spec_from_file_location("quantize_fp8", Path(__file__).with_name("quantize_fp8.py"))
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)


class PackedQuantizationTests(unittest.TestCase):
    def export(self, mode):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source, out = root / "source", root / "out"
        source.mkdir()
        (source / "config.json").write_text("{}")
        prefix = "model.language_model.layers.0."
        weights = {
            prefix + "linear_attn.in_proj_qkv.weight": torch.tensor([[1., -2.], [0.25, 0.5]], dtype=torch.bfloat16),
            prefix + "linear_attn.in_proj_z.weight": torch.tensor([[7., -8.]], dtype=torch.bfloat16),
            prefix + "linear_attn.in_proj_a.weight": torch.tensor([[2., 0.]], dtype=torch.bfloat16),
            prefix + "linear_attn.in_proj_b.weight": torch.tensor([[0.125, -0.25]], dtype=torch.bfloat16),
        }
        entries = list(weights.items())
        save_file(dict(entries[::2]), source / "part1.safetensors")
        save_file(dict(entries[1::2]), source / "part2.safetensors")
        mapping = {k: "part1.safetensors" if i % 2 == 0 else "part2.safetensors" for i, (k, _) in enumerate(entries)}
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        args = ["quantize_fp8.py", str(source), str(out)]
        if mode is not None:
            args += ["--scale-mode", mode]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            quant.main()
        return source, out, weights, load_file(out / "model.safetensors")

    def test_default_preserves_per_channel_quantization(self):
        _, out, weights, actual = self.export(None)
        self.assertFalse((out / "quantization.json").exists())
        for name, weight in weights.items():
            w = weight.float()
            scale = w.abs().amax(dim=1) / 448
            expected = (w / scale[:, None]).to(torch.float8_e4m3fn)
            self.assertTrue(torch.equal(actual["fp8/" + name].view(torch.uint8), expected.view(torch.uint8)))
            self.assertTrue(torch.equal(actual["fp8/" + name + "_scale"], scale))

    def test_packed_scales_match_concatenated_matrix(self):
        source, out, weights, actual = self.export("packed-tensor")
        meta = json.loads((out / "quantization.json").read_text())
        self.assertEqual(len(meta["groups"]), 2)
        for group, info in meta["groups"].items():
            names = [m["source"] for m in info["members"]]
            combined = torch.cat([weights[n].float() for n in names])
            scale = combined.abs().max() / 448
            expected = (combined * scale.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
            observed = torch.cat([actual["fp8/" + n] for n in names])
            self.assertTrue(torch.equal(observed.view(torch.uint8), expected.view(torch.uint8)))
            for name in names:
                self.assertTrue(torch.all(actual["fp8/" + name + "_scale"] == scale))
            if group.endswith("in_proj_ba.weight"):
                self.assertTrue(names[0].endswith("in_proj_b.weight"))
                self.assertEqual(info["members"][1]["packed_row_offset"], 1)
        prior = (out / "model.safetensors").read_bytes()
        with patch("sys.argv", ["q", str(source), str(out), "--scale-mode", "packed-tensor"]):
            with self.assertRaises(FileExistsError):
                quant.main()
        self.assertEqual((out / "model.safetensors").read_bytes(), prior)

    def test_packed_groups_do_not_cross_layers_or_projection_families(self):
        group, order = quant.packed_group("model.layers.7.self_attn.k_proj.weight")
        self.assertEqual((group, order), ("model.layers.7.self_attn.qkv_proj.weight", 1))
        self.assertNotEqual(group, quant.packed_group("model.layers.8.self_attn.k_proj.weight")[0])
        self.assertEqual(quant.packed_group("model.layers.7.self_attn.o_proj.weight")[0],
                         "model.layers.7.self_attn.o_proj.weight")


if __name__ == "__main__":
    unittest.main()
