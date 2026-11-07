import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import header_only  # noqa: E402

TOOL = str(Path(__file__).with_name("header_only.py"))
EXPORTER = """
import sys, torch
from safetensors.torch import load_file, save_file
src, out = sys.argv[1:]
sd = load_file(src)
w = sd["w"].double() @ sd["v"].double().T
save_file({"folded": w.float(), "pad": torch.zeros(3, 2, dtype=torch.bfloat16)}, out)
pth = torch.load(src + ".pth", map_location="cpu", weights_only=True)
print(tuple(pth["state_dict"]["conv.weight"].shape), pth["state_dict"]["conv.weight"].device)
"""


class HeaderOnlyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def header_only_copy(self, real, out):
        with open(real, "rb") as f:
            header = header_only.read_safetensors_header(f)
        out.write_bytes(header_only.safetensors_header(header))

    def test_exporter_runs_on_shapes_and_writes_header_only_outputs(self):
        real = self.root / "real.safetensors"
        save_file({"w": torch.ones(4, 8), "v": torch.ones(6, 8)}, str(real))
        src = self.root / "src.safetensors"
        self.header_only_copy(real, src)
        buffer = io.BytesIO()
        torch.save({"state_dict": {"conv.weight": torch.ones(2, 1, 7)}}, buffer)
        buffer.seek(0)
        tree = header_only.torch_tree(buffer)
        (self.root / "src.safetensors.pth").write_text(json.dumps({header_only.MARKER: 1, "tree": tree}))
        script = self.root / "exporter.py"
        script.write_text(EXPORTER)
        out = self.root / "out.safetensors"
        result = subprocess.run([sys.executable, TOOL, "run", str(script), str(src), str(out)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("(2, 1, 7) meta", result.stdout)
        marked, header = header_only.is_header_only_safetensors(out)
        self.assertTrue(marked)
        self.assertEqual(header["folded"], {"dtype": "F32", "shape": [4, 6], "data_offsets": [0, 96]})
        self.assertEqual(header["pad"], {"dtype": "BF16", "shape": [3, 2], "data_offsets": [96, 108]})
        self.assertEqual(os.path.getsize(out), 8 + int.from_bytes(out.read_bytes()[:8], "little"))

    def test_real_weights_are_refused_and_marked_files_must_be_empty(self):
        real = self.root / "real.safetensors"
        save_file({"w": torch.ones(4, 8), "v": torch.ones(6, 8)}, str(real))
        script = self.root / "exporter.py"
        script.write_text(EXPORTER)
        result = subprocess.run([sys.executable, TOOL, "run", str(script), str(real), str(self.root / "o")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("real weights", result.stderr)
        forged = self.root / "forged.safetensors"
        with open(real, "rb") as f:
            header = header_only.read_safetensors_header(f)
            data = f.read()
        forged.write_bytes(header_only.safetensors_header(header) + data)
        with self.assertRaisesRegex(ValueError, "carries tensor data"):
            header_only.is_header_only_safetensors(forged)

    def test_torch_tree_never_reads_storage_records(self):
        buffer = io.BytesIO()
        torch.save({"a": torch.zeros(3, 5, dtype=torch.bfloat16), "n": 7,
                    "state": torch.nn.Linear(2, 3).state_dict()}, buffer)
        import zipfile
        raw = bytearray(buffer.getvalue())
        storages = [i for i in zipfile.ZipFile(io.BytesIO(bytes(raw))).infolist() if "/data/" in i.filename]
        self.assertTrue(storages)
        for info in storages:
            local = raw[info.header_offset:info.header_offset + 30]
            start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
            raw[start] ^= 0xFF
        with self.assertRaises(zipfile.BadZipFile):
            zipfile.ZipFile(io.BytesIO(bytes(raw))).read(storages[0].filename)
        tree = header_only.torch_tree(io.BytesIO(bytes(raw)))
        self.assertEqual(tree, {"a": {"__tensor__": ["bfloat16", [3, 5]]}, "n": 7, "state": {
            "weight": {"__tensor__": ["float32", [3, 2]]}, "bias": {"__tensor__": ["float32", [3]]}}})


if __name__ == "__main__":
    unittest.main()
