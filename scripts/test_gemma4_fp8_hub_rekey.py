import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


def write_shard(path, tensors):
    header, data = {}, bytearray()
    for name, dtype, shape, payload in tensors:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(data), len(data) + len(payload)])
        data.extend(payload)
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def read_shard(path):
    data = path.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    payload = data[8 + size:]
    return {name: (info["dtype"], info["shape"], payload[slice(*info["data_offsets"])])
            for name, info in header.items() if name != "__metadata__"}


class HubRekeyTests(unittest.TestCase):
    def run_rekey(self, shards, check=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source, output = root / "source", root / "output"
        source.mkdir()
        names = {}
        for i, tensors in enumerate(shards):
            filename = "model.safetensors" if len(shards) == 1 else f"model-{i}.safetensors"
            write_shard(source / filename, tensors)
            names.update({row[0]: filename for row in tensors})
        if len(shards) > 1:
            (source / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=names)))
        (source / "config.json").write_text('{"model_type":"fixture"}')
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("gemma4_fp8_hub_rekey.py")),
                                 str(source), str(output)], capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
            return read_shard(output / "model.safetensors"), output
        return result, output

    def test_fused_gate_up_keeps_expert_axis(self):
        weight = bytes(range(48))
        scales = struct.pack("<16H", *range(16))
        tensors, _ = self.run_rekey([[
            ("m.experts.gate_up_proj", "F8_E4M3", [2, 8, 3], weight),
            ("m.experts.gate_up_proj.weight_scale", "BF16", [2, 8, 1], scales),
        ]])
        self.assertEqual(tensors["fp8/m.experts.gate_up_proj"], ("F8_E4M3", [2, 8, 3], weight))
        self.assertEqual(tensors["fp8/m.experts.gate_up_proj_scale"],
                         ("F32", [2, 8], struct.pack("<16I", *(x << 16 for x in range(16)))))

    def test_fused_down_keeps_expert_axis(self):
        tensors, _ = self.run_rekey([[
            ("m.experts.down_proj", "F8_E4M3", [3, 2, 4], bytes(range(24))),
            ("m.experts.down_proj.weight_scale", "BF16", [3, 2, 1], bytes(12)),
        ]])
        self.assertEqual(tensors["fp8/m.experts.down_proj_scale"], ("F32", [3, 2], bytes(24)))

    def test_dense_widening_preserves_all_bf16_bits(self):
        bits = [0, 0x8000, 0x3F80, 0x7F80, 0x7FC1, 0xFF80]
        tensors, _ = self.run_rekey([[
            ("m.q_proj.weight", "F8_E4M3", [6, 2], bytes(range(12))),
            ("m.q_proj.weight_scale", "BF16", [6, 1], struct.pack("<6H", *bits)),
        ]])
        self.assertEqual(tensors["fp8/m.q_proj.weight_scale"],
                         ("F32", [6], struct.pack("<6I", *(x << 16 for x in bits))))

    def test_unquantized_payload_and_config_are_preserved(self):
        payload = bytes([255, 128, 1, 0])
        tensors, output = self.run_rekey([[('m.norm.weight', "BF16", [2], payload)]])
        self.assertEqual(tensors, {'m.norm.weight': ("BF16", [2], payload)})
        self.assertTrue((output / "config.json").is_symlink())
        self.assertEqual(json.loads((output / "model.safetensors.index.json").read_text())["weight_map"],
                         {'m.norm.weight': 'model.safetensors'})

    def test_weight_and_scale_can_live_in_different_shards(self):
        tensors, _ = self.run_rekey([
            [("m.q_proj.weight", "F8_E4M3", [2, 3], bytes(range(6)))],
            [("m.q_proj.weight_scale", "BF16", [2, 1], struct.pack("<2H", 0x3F80, 0x4000))],
        ])
        self.assertEqual(tensors["fp8/m.q_proj.weight"], ("F8_E4M3", [2, 3], bytes(range(6))))
        self.assertEqual(tensors["fp8/m.q_proj.weight_scale"], ("F32", [2], struct.pack("<2f", 1., 2.)))

    def test_malformed_scale_is_refused_before_output(self):
        for dtype, shape in [("F32", [2, 1]), ("BF16", [1, 2])]:
            with self.subTest(dtype=dtype, shape=shape):
                result, output = self.run_rekey([[
                    ("m.q_proj.weight", "F8_E4M3", [2, 2], bytes(4)),
                    ("m.q_proj.weight_scale", dtype, shape, bytes(8 if dtype == "F32" else 4)),
                ]], check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((output / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
