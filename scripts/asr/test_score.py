import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from score import counts, errors, normalize, paired_interval, main


class ScoreTests(unittest.TestCase):
    def test_normalization_and_word_errors(self):
        self.assertEqual(normalize("  DON’T—stop, 42!  "), "don't stop 42")
        result = counts(normalize("ONE TWO THREE"), normalize("one four"))
        self.assertEqual(result["reference_units"], 3)
        self.assertEqual(errors(result), 2)

    def test_failure_is_all_deletions(self):
        self.assertEqual(counts("one two", ""), {
            "substitutions": 0, "deletions": 2, "insertions": 0, "reference_units": 2})

    def test_character_score_and_identical_pair(self):
        self.assertEqual(errors(counts("你好", "您好", True)), 1)
        rows = {"a": {"normalized_words": counts("one two", "one"), "seconds": 1.0}}
        result = paired_interval(rows, rows, {"a": {"speaker": "s"}}, 20)
        self.assertEqual(result["normalized_wer_delta_95_interval"], [0.0, 0.0])
        self.assertEqual(result["inference_time_ratio_95_interval"], [1.0, 1.0])

    def test_paired_time_ratio_uses_total_time(self):
        left = {u: {"normalized_words": counts("one", "one"), "seconds": t}
                for u, t in [("a", 1.0), ("b", 9.0)]}
        right = {u: {**row, "seconds": 2.0} for u, row in left.items()}
        result = paired_interval(left, right, {u: {"speaker": "s"} for u in left}, 20)
        self.assertEqual(result["inference_time_ratio_95_interval"], [2.5, 2.5])

    def test_incomplete_run_cannot_produce_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = [{"id": u, "speaker": "s", "reference": "one", "duration_seconds": 1.0} for u in ["a", "b"]]
            (root / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in manifest))
            (root / "results.jsonl").write_text(json.dumps({"kind": "result", "backend": "test", "id": "a",
                "reference": "one", "duration_seconds": 1.0, "seconds": .1, "text": "one"})+"\n")
            with patch("sys.argv", ["score.py", str(root / "manifest.jsonl"), str(root / "results.jsonl"), "--out", str(root / "report.json")]):
                with self.assertRaisesRegex(ValueError, "incomplete run"):
                    main()
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
