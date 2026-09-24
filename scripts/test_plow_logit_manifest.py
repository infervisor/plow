from pathlib import Path
import json
import tempfile
import argparse
import unittest
from unittest.mock import patch

from plow_logit_manifest import assemble_replicated_row, assemble_sharded_row, batched_cases, main


class ShardedLogitTests(unittest.TestCase):
    def test_batched_histories_and_rank_row_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt").write_text("1,2,3;7,8")
            text = ("batched TP decode: 2 active of 2 slots per dispatch, 2 ranks\n"
                    "  slot 0: prefill 3 tokens -> sampled 4\n"
                    "  slot 1: prefill 2 tokens -> sampled 9\n  [5, 6]\n  [10, 11]\n")
            args = argparse.Namespace(batch_size=2, prompt=root / "prompt", name="test",
                                      logits_dir=root, tp_shards=2, vocab=4, replicated_vocab=False)
            for tag in ("b000", "b001"):
                for rank, data in enumerate((b"abcdefgh", b"ijklmnop")):
                    (root / f"logits.rk{rank}.{tag}.bin").write_bytes(data)
            rows = batched_cases(args, text)
            self.assertEqual([r["prompt_token_ids"] for r in rows],
                             [[1, 2, 3, 4], [7, 8, 9], [1, 2, 3, 4, 5], [7, 8, 9, 10]])
            self.assertEqual([r["sampled_token_id"] for r in rows], [5, 10, 6, 11])
            self.assertEqual([r["generation_step"] for r in rows], [1, 1, 2, 2])
            self.assertTrue(all(r["execution_phase"] == "decode_output" for r in rows))
            self.assertEqual(Path(rows[0]["file"]).read_bytes(), b"abcdijkl")
            self.assertEqual(Path(rows[1]["file"]).read_bytes(), b"efghmnop")
            for invalid in (text.replace("2 active", "8 active"),
                            text.replace("slot 1", "slot 0"),
                            text.replace("prefill 2", "prefill 3"),
                            text.replace("[10, 11]", "[10]")):
                with self.assertRaises(ValueError):
                    batched_cases(args, invalid)
            (root / "logits.rk1.b000.bin").write_bytes(b"ijkl")
            with self.assertRaises(ValueError):
                batched_cases(args, text)

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

    def test_replicated_full_vocabulary_rows_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "logits.b000.bin"
            source.write_bytes(b"abcdefgh" + b"ijklmnop")
            self.assertEqual(assemble_replicated_row(root, "b000", 4, 2, 0).read_bytes(), b"abcdefgh")
            self.assertEqual(assemble_replicated_row(root, "b000", 4, 2, 1).read_bytes(), b"ijklmnop")
            source.write_bytes(b"abcdefgh")
            with self.assertRaises(ValueError):
                assemble_replicated_row(root, "b000", 4, 2, 0)
            source.write_bytes(b"\x80\x7f" + b"\x00\x00" * 3)
            with self.assertRaises(ValueError):
                assemble_replicated_row(root, "b000", 4)

    def test_replicated_manifest_uses_one_full_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt").write_text("1,2,3")
            (root / "stdout").write_text("prefill: 3 tokens -> 4\n[5]\n")
            for tag in ("prefill", "000"):
                (root / f"logits.{tag}.bin").write_bytes(b"abcdefgh")
            args = ["manifest", "--name", "test", "--prompt", str(root / "prompt"),
                    "--stdout", str(root / "stdout"), "--logits-dir", str(root),
                    "--output", str(root / "manifest.json"), "--tp-shards", "8",
                    "--vocab", "4", "--replicated-vocab"]
            with patch("sys.argv", args):
                main()
            result = json.loads((root / "manifest.json").read_text())
            self.assertEqual(result["vocabulary_layout"], "replicated")
            self.assertEqual(result["tensor_parallel_ranks"], 8)
            self.assertEqual([Path(case["file"]).read_bytes() for case in result["cases"]],
                             [b"abcdefgh", b"abcdefgh"])

    def test_replicated_batched_rows_follow_active_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt").write_text("1,2,3;7,8")
            text = ("batched TP decode: 2 active of 2 slots per dispatch, 8 ranks\n"
                    "  slot 0: prefill 3 tokens -> sampled 4\n"
                    "  slot 1: prefill 2 tokens -> sampled 9\n  [5]\n  [10]\n")
            args = argparse.Namespace(batch_size=2, prompt=root / "prompt", name="test",
                                      logits_dir=root, tp_shards=8, vocab=4, replicated_vocab=True)
            (root / "logits.b000.bin").write_bytes(b"abcdefgh" + b"ijklmnop")
            rows = batched_cases(args, text)
            self.assertEqual([Path(row["file"]).read_bytes() for row in rows],
                             [b"abcdefgh", b"ijklmnop"])
            self.assertEqual([row["sampled_token_id"] for row in rows], [5, 10])

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
