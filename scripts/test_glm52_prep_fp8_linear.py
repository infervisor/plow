import json
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import glm52_prep_fp8_linear as prep


class QkvaPrepTests(unittest.TestCase):
    def test_original_bytes_and_scale_tail(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "original.bin"
            q = bytes([128, 0, 127, 255]) * (128 * 128 // 4)
            kv = bytes([1, 2, 3, 4]) * (192 * 128 // 4)
            qs, ks = struct.pack("<f", 0.25), struct.pack("<ff", 0.5, 0.75)
            path.write_bytes(q + kv + qs + ks)
            prefix = "model.layers.3.self_attn."
            source = {}
            offset = 0
            for name, data, dtype, shape in [
                ("q_a_proj.weight", q, "F8_E4M3", [128, 128]),
                ("kv_a_proj_with_mqa.weight", kv, "F8_E4M3", [192, 128]),
                ("q_a_proj.weight_scale_inv", qs, "F32", [1, 1]),
                ("kv_a_proj_with_mqa.weight_scale_inv", ks, "F32", [2, 1]),
            ]:
                source[prefix + name] = (str(path), offset, offset + len(data), dtype, shape)
                offset += len(data)
            config = dict(hidden_size=128, q_lora_rank=128, kv_lora_rank=128, qk_rope_head_dim=64)
            entries = prep.qkva_entries(source, config, 3)
            output = Path(root) / "model-qkva-00003.safetensors"
            prep.write_shard(output, entries)
            self.assertTrue(prep.shard_ok(output, entries))
            blob = output.read_bytes()
            size = struct.unpack("<Q", blob[:8])[0]
            header = json.loads(blob[8:8 + size])
            self.assertEqual(header[prefix + "fused_qkv_a_proj.weight_fp8"]["shape"], [320, 128])
            self.assertEqual(header[prefix + "fused_qkv_a_proj.weight_scale_inv"]["shape"], [3, 1])
            self.assertEqual(blob[8 + size:], q + kv + qs + ks)
            self.assertFalse(prep.shard_ok(output, entries[:1]))
            for field, value in (("q_lora_rank", 64), ("hidden_size", 129)):
                with self.assertRaises(ValueError):
                    prep.qkva_entries(source, dict(config, **{field: value}), 3)
            key = prefix + "q_a_proj.weight"
            original = source[key]
            source[key] = (*original[:3], "BF16", original[4])
            with self.assertRaises(ValueError):
                prep.qkva_entries(source, config, 3)

    def test_legacy_single_record(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.bin"
            path.write_bytes(struct.pack("<ff", 1.0, 2.0))
            entries = [("scale", (str(path), 0, 8, "F32", [1, 2]))]
            output = Path(root) / "out.safetensors"
            prep.write_shard(output, entries)
            self.assertTrue(prep.shard_ok(output, entries))
            self.assertFalse(prep.shard_ok(output, [("scale", (str(path), 0, 8, "F32", [2, 1]))]))


@unittest.skipUnless(importlib.util.find_spec("torch"), "MLA prep requires CPU PyTorch")
class MlaPrepTests(unittest.TestCase):
    def test_tp_quantization_bytes_and_resume(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = "model.layers.0.self_attn."
            cfg = dict(num_attention_heads=4, q_lora_rank=128, kv_lora_rank=128,
                       qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=64)
            qb = torch.full((1024, 128), 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
            kv = torch.ones((1024, 128), dtype=torch.float8_e4m3fn)
            scales = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.]).reshape(8, 1)
            source = {prefix + "q_b_proj.weight": qb, prefix + "q_b_proj.weight_scale_inv": scales,
                      prefix + "kv_b_proj.weight": kv, prefix + "kv_b_proj.weight_scale_inv": scales.clone()}
            save_file(source, root / "model-source.safetensors")
            records = prep.index_shards(root)
            result = prep.mla_tensors(records, cfg, 0, 2)
            self.assertEqual(len(result), 4)
            self.assertFalse(any("q_b_proj" in name for name in result))
            base = prefix + "derived.mla_fp8_tp2."
            self.assertEqual(result[base + "wk.weight"].shape, (4, 128, 192))
            self.assertEqual(result[base + "wv.weight"].shape, (4, 64, 128))
            self.assertEqual(result[base + "wk.weight_scale"].shape, (2, 1))
            self.assertNotEqual(float(result[base + "wk.weight_scale"][0]), float(result[base + "wk.weight_scale"][1]))
            full = prep.mla_tensors(records, cfg, 0, 1)
            self.assertNotEqual(float(result[base + "wk.weight_scale"][0]),
                                float(full[prefix + "derived.mla_fp8_tp1.wk.weight_scale"][0]))
            path = root / "out.safetensors"
            prep.write_mla_shard(path, result)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            before = path.read_bytes(), path.stat().st_mtime_ns
            path.chmod(0o600)
            prep.write_mla_shard(path, result)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            path.write_bytes(b"broken")
            prep.write_mla_shard(path, result)
            with safe_open(path, framework="pt", device="cpu") as shard:
                self.assertEqual(set(shard.keys()), set(result))
            for tp in (0, 3, 8):
                with self.assertRaises(ValueError):
                    prep.mla_tensors(records, cfg, 0, tp)


if __name__ == "__main__":
    unittest.main()
