from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from plow_logit_manifest import assemble_sharded_row, main


class ShardedLogitTests(unittest.TestCase):
    def test_generation_histories_and_execution_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt").write_text("1,2,3")
            (root / "stdout").write_text("prefill: 3 tokens -> 4\n[5, 6]\n")
            for tag in ("prefill", "000", "001"):
                (root / f"logits_{tag}.bin").write_bytes(b"\0\0")
            args = ["manifest", "--name", "test", "--prompt", str(root / "prompt"),
                    "--stdout", str(root / "stdout"), "--logits-dir", str(root),
                    "--output", str(root / "manifest.json")]
            with patch("sys.argv", args):
                main()
            rows = json.loads((root / "manifest.json").read_text())["cases"]
            self.assertEqual([row["prompt_token_ids"] for row in rows], [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
            self.assertEqual([row["execution_phase"] for row in rows], ["prefill_output", "decode_output", "decode_output"])
            self.assertEqual([row["generation_step"] for row in rows], [0, 1, 2])

    def test_rank_order_assembles_full_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank, values in enumerate((b"abcd", b"efgh")):
                (root / f"logits.rk{rank}.prefill.bin").write_bytes(values)
            path = assemble_sharded_row(root, "prefill", 2, 4)
            self.assertEqual(path.read_bytes(), b"abcdefgh")

    def test_missing_partial_and_extra_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                assemble_sharded_row(root, "000", 2, 4)
            for size in (0, 2, 8):
                (root / "logits.rk0.000.bin").write_bytes(bytes(size))
                with self.assertRaises(ValueError):
                    assemble_sharded_row(root, "000", 2, 4)
            with self.assertRaises(ValueError):
                assemble_sharded_row(root, "000", 2, 5)

    def test_nonfinite_logits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (b"\x80\x7f", b"\xc0\x7f", b"\x80\xff"):
                (root / "logits.rk0.000.bin").write_bytes(value + b"\x00\x00")
                with self.assertRaises(ValueError):
                    assemble_sharded_row(root, "000", 2, 4)


if __name__ == "__main__":
    unittest.main()
