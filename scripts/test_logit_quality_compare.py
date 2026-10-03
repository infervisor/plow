import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from logit_quality_compare import main, metrics


class LogitQualityTests(unittest.TestCase):
    def test_nonfinite_rows_cannot_pass(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                metrics(np.array([1, bad]), np.array([1, 2]), [1])

    def test_declared_suppression_preserves_token_ids(self):
        row = metrics(np.array([1., 999., 2., 3.]), np.array([1., -np.inf, 2., 3.]), [1], [1])
        self.assertEqual(row["token"], {"candidate": 3, "reference": 3})
        self.assertEqual(row["vocab_compared"], 3)
        self.assertEqual(row["full_row_centered_rel_l2"], 0)
        self.assertEqual(row["excluded_token_ids"], [1])

    def test_suppression_does_not_hide_other_invalid_values(self):
        for row in (np.array([1., np.nan, 2.]), np.array([1., np.inf, 2.]),
                    np.array([-np.inf, -np.inf, 2.])):
            with self.assertRaises(ValueError):
                metrics(row, np.array([1., -np.inf, 2.]), [1], [1])

    def run_comparison(self, missing=False, wrong_phase=False, repeated_bad_first=False, suppressed_sample=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = root / "row.f32"
            np.array([1, 2, 3], dtype="<f4").tofile(row)
            case = dict(id="a", prompt_sha256_u32le="a", prompt_len=8, file=str(row),
                        execution_phase="decode_output")
            reference = dict(cases=[case], repeat_checks=[dict(
                prompt_sha256_u32le="a", same_argmax=True, full_row_centered_rel_l2=.01,
                reference_head64_centered_rel_l2=.01, centered_max_abs=.01, top64_overlap=1)])
            candidate = dict(cases=[dict(case, execution_phase="prefill_output" if wrong_phase else "decode_output")])
            if suppressed_sample:
                reference["suppression"] = {"token_ids": [0]}
                candidate["cases"][0]["sampled_token_id"] = 0
            if repeated_bad_first:
                candidate["cases"].insert(0, dict(case, id="earlier-slot", execution_phase="prefill_output"))
            if missing:
                candidate["cases"].append(dict(case, id="missing", prompt_sha256_u32le="missing"))
            for name, data in (("ref", reference), ("cand", candidate)):
                (root / f"{name}.json").write_text(json.dumps(data))
            args = ["compare", "--reference", str(root / "ref.json"), "--candidate", str(root / "cand.json"),
                    "--output", str(root / "result.json"), "--require-same-phase", "--require-pass"]
            with patch("sys.argv", args):
                if missing or wrong_phase or repeated_bad_first or suppressed_sample:
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 1)
                else:
                    main()
            return json.loads((root / "result.json").read_text())["comparisons"][0]

    def test_selecting_suppressed_token_cannot_pass(self):
        row = self.run_comparison(suppressed_sample=True)
        self.assertFalse(row["quality_gate_pass"])
        self.assertEqual(row["candidate_sampled_suppressed_rows"], 1)

    def test_exact_complete_same_phase_passes(self):
        self.assertTrue(self.run_comparison()["quality_gate_pass"])

    def test_missing_history_is_not_silently_dropped(self):
        row = self.run_comparison(missing=True)
        self.assertFalse(row["quality_gate_pass"])
        self.assertEqual(row["unmatched_candidate_cases"], ["missing"])

    def test_cross_phase_is_diagnostic_not_qualification(self):
        row = self.run_comparison(wrong_phase=True)
        self.assertFalse(row["quality_gate_pass"])
        self.assertEqual(row["phase_mismatch_or_unmeasured_rows"], 1)

    def test_repeated_history_does_not_hide_a_bad_earlier_slot(self):
        row = self.run_comparison(repeated_bad_first=True)
        self.assertFalse(row["quality_gate_pass"])
        self.assertEqual(row["matched_histories"], 1)
        self.assertEqual(row["matched_rows"], 2)
        self.assertEqual(row["phase_mismatch_or_unmeasured_rows"], 1)


if __name__ == "__main__":
    unittest.main()
