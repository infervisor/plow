#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from certify_kernel import certify_data, evaluate_efficiency


class TestCertifyKernel(unittest.TestCase):
    def test_gate_is_costmodels(self):
        import certify_kernel
        self.assertEqual((certify_kernel.MIN_MEMORY, certify_kernel.MIN_COMPUTE), (0.80, 0.75))

    def test_evaluate_memory_pass(self):
        row = {"phase": "decode", "bound": "memory", "efficiency": 0.84, "point": "decode:1:0"}
        res = evaluate_efficiency(row)
        self.assertTrue(res["passed"])
        self.assertEqual(res["threshold"], 0.80)

    def test_evaluate_memory_fail(self):
        row = {"phase": "decode", "bound": "memory", "efficiency": 0.76, "point": "decode:1:0"}
        res = evaluate_efficiency(row)
        self.assertFalse(res["passed"])
        self.assertIn("below required threshold", res["reason"])

    def test_evaluate_compute_pass(self):
        row = {"phase": "prefill", "bound": "compute", "efficiency": 0.78, "point": "prefill:1:2048:0"}
        res = evaluate_efficiency(row)
        self.assertTrue(res["passed"])
        self.assertEqual(res["threshold"], 0.75)

    def test_evaluate_compute_fail(self):
        row = {"phase": "prefill", "bound": "compute", "efficiency": 0.71, "point": "prefill:1:2048:0"}
        res = evaluate_efficiency(row)
        self.assertFalse(res["passed"])

    def test_certify_dataset(self):
        dataset = {
            "rows": [
                {"phase": "decode", "bound": "memory", "efficiency": 0.85, "point": "decode:1:0"},
                {"phase": "decode", "bound": "memory", "efficiency": 0.82, "point": "decode:8:0"},
                {"phase": "prefill", "bound": "compute", "efficiency": 0.79, "point": "prefill:1:4096:0"},
            ]
        }
        all_passed, results = certify_data(dataset)
        self.assertTrue(all_passed)
        self.assertEqual(len(results), 3)

    def test_cli_execution(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({
                "rows": [
                    {"phase": "decode", "bound": "memory", "efficiency": 0.90, "point": "decode:1:0"}
                ]
            }, f)
            f.flush()

            cmd = [sys.executable, str(Path(__file__).parent / "certify_kernel.py"), "--input", f.name]
            out = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0)
            self.assertIn("QUALIFIED", out.stdout)


if __name__ == "__main__":
    import sys
    unittest.main()
