import copy
import json
from pathlib import Path
import tempfile
import unittest

from block_ab import ARMS, compare, score, output_hashes


def measurements(values=(100, 80, 102, 81)):
    return {name: dict(scope="single-block-decode", batch=8, ctx=512, tp=8,
                       clock="host-prepare-dispatch-drain-audit", cache_policy="warm",
                       warmup=30, checkpoint="weights", trace_instrumented=False,
                       oracle_verified=True, rank_identity=True, finiteness=True,
                       samples_us=[value] * 30, latency_us_median=value)
            for name, value in zip(ARMS, values)}


class BlockAbTests(unittest.TestCase):
    def test_atomic_repeat_requires_complete_hash_bound_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(inputs={"x": "same"}, runtime_sha256="runtime", env={}, cell={},
                          packet_sha256="packet", objects={"object": "hash"})
            for arm, row in measurements().items():
                path = root / arm
                (path / "outputs").mkdir(parents=True)
                for rank in range(8):
                    for name in ("moe_fug", "moe_rowtok", "moe_rowpart", "moe_rowgate", "part", "attn", "xnext"):
                        (path / "outputs" / f"rank{rank}.act.{name}.bin").write_bytes(arm.encode())
                (path / "outputs/stable.bin").write_bytes(b"same")
                (path / "run-record.json").write_text(json.dumps(record))
                (path / "measurement.json").write_text(json.dumps(row))
            audit = dict(scope="routed-BF16-atomic-repeat-v1", passed=True, tp=8,
                output_sha256={arm: output_hashes(root / arm / "outputs") for arm in ARMS},
                checks=[dict(arm=arm, rank=rank, stable_boundaries_bitwise=True, reduction_in_bounds=True)
                        for arm in ARMS for rank in range(8)])
            report = root / "audit.json"
            report.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "nonrepeatable"):
                compare(root)
            self.assertTrue(compare(root, routed_repeat_audit=report)["gate_pass"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                compare(root, require_bitwise=True, routed_repeat_audit=report)
            for field, value in (("passed", False), ("tp", 4), ("checks", audit["checks"][:-1])):
                bad = {**audit, field: value}
                report.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    compare(root, routed_repeat_audit=report)
            report.write_text(json.dumps(audit))
            (root / "treat/outputs/stable.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "stale"):
                compare(root, routed_repeat_audit=report)
            audit["output_sha256"]["treat"] = output_hashes(root / "treat/outputs")
            report.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "unaudited"):
                compare(root, routed_repeat_audit=report)

    def test_win_and_control_control(self):
        result = score(measurements())
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["noise_floor_us"], 2)
        self.assertFalse(score(measurements((100, 100, 100, 100)))["gate_pass"])

    def test_noise_and_treatment_instability(self):
        self.assertFalse(score(measurements((100, 80, 101, 90)))["gate_pass"])
        self.assertFalse(score(measurements((100, 99, 104, 100)))["gate_pass"])
        data = measurements()
        for row in data.values():
            median = row["latency_us_median"]
            row["samples_us"] = [median - 20] * 15 + [median] + [median + 20] * 14
        self.assertFalse(score(data)["gate_pass"])

    def test_invalid_or_mixed_cells(self):
        for field, value in (("oracle_verified", False), ("batch", 16), ("ctx", 8192),
                             ("clock", "device"), ("checkpoint", "other"),
                             ("trace_instrumented", True), ("host_timing_instrumented", True),
                             ("latency_us_median", 1),
                             ("samples_us", [float("nan")] * 30), ("samples_us", [80] * 29)):
            with self.subTest(field=field):
                data = measurements()
                data["treat"][field] = value
                with self.assertRaises(ValueError):
                    score(data)

    def test_frozen_identity_and_repeat_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(inputs={"x": "same"}, runtime_sha256="runtime", env={}, cell={"batch": 8},
                          packet_sha256="packet", objects={"object": "hash"})
            for name, row in measurements().items():
                arm = root / name
                (arm / "outputs").mkdir(parents=True)
                (arm / "outputs/rank0.bin").write_bytes(b"candidate" if name.startswith("treat") else b"control")
                (arm / "run-record.json").write_text(json.dumps(record))
                (arm / "measurement.json").write_text(json.dumps(row))
            self.assertTrue(compare(root)["gate_pass"])
            self.assertEqual(compare(root)["changed_output_files"], 1)
            self.assertFalse(compare(root, require_bitwise=True)["gate_pass"])
            changed = copy.deepcopy(record)
            changed["inputs"]["x"] = "different"
            (root / "treat/run-record.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "inputs"):
                compare(root)
            (root / "treat/run-record.json").write_text(json.dumps(record))
            (root / "treat2/outputs/rank0.bin").write_bytes(b"unstable")
            with self.assertRaisesRegex(ValueError, "nonrepeatable"):
                compare(root)


if __name__ == "__main__":
    unittest.main()
