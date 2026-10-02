import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from gemma4_fp8_glu_oracle import glu, main, metrics, projection, read_rows


class GluOracleTest(unittest.TestCase):
    def test_projection_scales_and_bf16_boundary(self):
        x = torch.tensor([[1., 2.], [-1., 3.]])
        weight = torch.tensor([[2., -1.], [1., 1.]])
        result = projection(x, weight, torch.tensor([.5, 2.]), torch.tensor([2., .25]))
        self.assertTrue(torch.equal(result, torch.tensor([[0., .375], [-20., 1.]])))
        rounded = projection(torch.tensor([[1.003]]), torch.ones(1, 1), torch.ones(1), torch.ones(1))
        self.assertEqual(rounded.item(), 1.)

    def test_glu_zero_and_positive(self):
        result = glu(torch.tensor([[0., 8.]]), torch.tensor([[7., 2.]]))
        self.assertTrue(torch.equal(result, torch.tensor([[0., 16.]])))

    def test_fp8_rows_and_padding(self):
        values = torch.tensor([[1., -2.], [3., 4.], [5., 6.]]).to(torch.float8_e4m3fn)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tensor.bin"
            path.write_bytes(values.view(torch.uint8).numpy().tobytes())
            item = {"file": path.name, "bytes": 6}
            result = read_rows(Path(tmp), item, torch.float8_e4m3fn, 2, [1, 0], 2)
            self.assertTrue(torch.equal(result, torch.tensor([[3., 4.], [1., -2.]])))
            item["bytes"] = 5
            with self.assertRaises(ValueError):
                read_rows(Path(tmp), item, torch.float8_e4m3fn, 2, [0], 2)

    def test_metrics_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            metrics(torch.tensor([float("nan")]), torch.ones(1))
        self.assertEqual(metrics(torch.ones(2), torch.ones(2))["rel_l2"], 0.)

    def test_capture_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "fp8/model.language_model.layers.0.mlp."
            weights = {}
            for part in ("gate", "up"):
                weights[prefix + part + "_proj.weight"] = torch.ones(2, 2).to(torch.float8_e4m3fn)
                weights[prefix + part + "_proj.weight_scale"] = torch.ones(2)
            save_file(weights, root / "weights.safetensors")
            values = {
                "act.xqh": torch.full((2, 2), 4.).to(torch.float8_e4m3fn),
                "act.ash": torch.ones(2),
                "act.fu": torch.full((2, 2), 64.).to(torch.bfloat16),
            }
            items = []
            for i, (name, value) in enumerate(values.items()):
                raw = value.view(torch.uint8).numpy().tobytes()
                file = f"tensor-{i}.bin"
                (root / file).write_bytes(raw)
                items.append({"name": name, "file": file, "bytes": len(raw)})
            (root / "manifest.json").write_text(json.dumps({"input_rows": 2, "tensors": items}))
            output = root / "result.json"
            with patch("sys.argv", ["oracle", "--capture", str(root), "--checkpoint", str(root),
                                    "--rows", "1,0", "--output", str(output),
                                    "--export-probe-inputs", str(root / "replay")]):
                main()
            result = json.loads(output.read_text())
            self.assertEqual(result["glu"]["rel_l2"], 0.)
            self.assertEqual(set(result["captured_rows"]), set(values))
            exported = json.loads((root / "replay/manifest.json").read_text())
            self.assertEqual((exported["m"], exported["n"], exported["k"]), (2, 2, 2))
            expected = {"activation": values["act.xqh"], "activation_scale": values["act.ash"]}
            for part in ("gate", "up"):
                expected[part] = weights[prefix + part + "_proj.weight"]
                expected[part + "_scale"] = weights[prefix + part + "_proj.weight_scale"]
            self.assertEqual(set(exported["files"]), set(expected))
            for name, tensor in expected.items():
                raw = (root / "replay" / f"{name}.bin").read_bytes()
                self.assertEqual(raw, tensor.view(torch.uint8).numpy().tobytes())
                self.assertEqual(exported["files"][name]["sha256"], hashlib.sha256(raw).hexdigest())



if __name__ == "__main__":
    unittest.main()
