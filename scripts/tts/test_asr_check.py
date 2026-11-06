import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from asr_check import cer_report


class CerReportTests(unittest.TestCase):
    def test_zero_median_does_not_hide_a_single_severe_failure(self):
        rows = [dict(cer=0.0, hyp="correct") for _ in range(31)]
        rows.append(dict(cer=41.3125, hyp="ha " * 400))
        report = cer_report(rows, "pinned-model", 0.15)
        self.assertTrue(report["median_gate"]["passed"])
        self.assertEqual(report["median_cer"], 0.0)
        self.assertEqual(report["p95_cer"], 0.0)
        self.assertEqual(report["max_cer"], 41.3125)
        self.assertEqual(report["over_limit_count"], 1)
        self.assertEqual(report["mean_cer"], 41.3125 / 32)
        self.assertEqual(json.loads(json.dumps(report))["rows"][-1]["hyp"], rows[-1]["hyp"])

    def test_percentile_and_limit_boundary_are_explicit(self):
        rows = [dict(cer=c) for c in [0.0, 0.1, 0.15, 0.4]]
        report = cer_report(rows, "model", 0.15)
        self.assertEqual(report["p95_cer"], 0.4)
        self.assertEqual(report["over_limit_count"], 1)
        self.assertEqual(report["median_gate"]["limit"], 0.15)


if __name__ == "__main__":
    unittest.main()
