import struct
import tempfile
import unittest
from pathlib import Path

import glm52_prep_fp8_linear as F
import glm53_prep_quark as Q


class QuarkQkvaAliasTests(unittest.TestCase):
    def test_raw_fp8_weights_and_quark_scales_are_concatenated(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            output = Path(directory) / "qkva.safetensors"
            q = bytes([128, 0, 127, 255]) * (128 * 128 // 4)
            kv = bytes([1, 2, 3, 4]) * (192 * 128 // 4)
            qs, ks = struct.pack("<f", 0.25), struct.pack("<ff", 0.5, 0.75)
            source.write_bytes(q + kv + qs + ks)
            prefix = "model.layers.3.self_attn."
            raw = {}
            offset = 0
            for name, data, dtype, shape in [
                ("q_a_proj.weight", q, "F8_E4M3", [128, 128]),
                ("kv_a_proj_with_mqa.weight", kv, "F8_E4M3", [192, 128]),
                ("q_a_proj.weight_scale", qs, "F32", [1, 1]),
                ("kv_a_proj_with_mqa.weight_scale", ks, "F32", [2, 1]),
            ]:
                raw[prefix + name] = (str(source), offset, offset + len(data), dtype, shape)
                offset += len(data)
            cfg = {
                "hidden_size": 128, "q_lora_rank": 128, "kv_lora_rank": 128,
                "qk_rope_head_dim": 64,
                "quantization_config": {"layer_quant_config": {
                    prefix + proj: {"weight": {"dtype": "fp8_e4m3", "block_size": [128, 128]}}
                    for proj in ("q_a_proj", "kv_a_proj_with_mqa")
                }},
            }
            entries = Q.qkva_fp8_entries(raw, cfg, 3)
            F.write_shard(output, entries)
            self.assertTrue(F.shard_ok(output, entries))
            Q.verify_qkva_values(output, entries)
            with output.open("r+b") as shard:
                shard.seek(-1, 2)
                shard.write(b"\0")
            with self.assertRaises(AssertionError):
                Q.verify_qkva_values(output, entries)


if __name__ == "__main__":
    unittest.main()
