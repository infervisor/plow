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

    def run_comparison(self, missing=False, wrong_phase=False):
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
            if missing:
                candidate["cases"].append(dict(case, id="missing", prompt_sha256_u32le="missing"))
            for name, data in (("ref", reference), ("cand", candidate)):
                (root / f"{name}.json").write_text(json.dumps(data))
            args = ["compare", "--reference", str(root / "ref.json"), "--candidate", str(root / "cand.json"),
                    "--output", str(root / "result.json"), "--require-same-phase", "--require-pass"]
            with patch("sys.argv", args):
                if missing or wrong_phase:
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 1)
                else:
                    main()
            return json.loads((root / "result.json").read_text())["comparisons"][0]

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


if __name__ == "__main__":
    unittest.main()
