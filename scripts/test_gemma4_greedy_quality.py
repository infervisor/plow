import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gemma4_greedy_quality as quality


class ConcurrentCaptureTests(unittest.TestCase):
    def test_parallel_capture_keeps_prompt_order_and_all_results(self):
        prompts = [dict(length=128, index=i, text=str(i), sha256=str(i)) for i in range(6)]
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        active = maximum = 0

        def post(url, path, body):
            nonlocal active, maximum
            index = int(body["messages"][0]["content"].splitlines()[0])
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait(timeout=3)
            time.sleep(0.01 * (3 - index % 3))
            with lock:
                active -= 1
            return {"choices": [{"message": {"content": f"result {index}"}}],
                    "usage": {"prompt_tokens": 150, "completion_tokens": 3}}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "capture.jsonl"
            args = SimpleNamespace(concurrency=3, mode="chat", tokenizer="unused", corpus=[],
                                   lengths=[128], per_length=6, url="http://test", out=out,
                                   label="test", max_tokens=64)
            with patch.object(quality, "build_prompts", return_value=prompts), \
                    patch.object(quality, "post", side_effect=post), \
                    patch.object(quality.urllib.request, "urlopen",
                                 return_value=io.BytesIO(b'{"data":[{"id":"model"}]}')), \
                    contextlib.redirect_stdout(io.StringIO()):
                quality.cmd_capture(args)
            records = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(maximum, 3)
        self.assertEqual([r["index"] for r in records], list(range(6)))
        self.assertEqual([r["completion"] for r in records], [f"result {i}" for i in range(6)])

    def test_nonpositive_concurrency_fails_before_requests(self):
        with self.assertRaisesRegex(SystemExit, "concurrency must be positive"):
            quality.cmd_capture(SimpleNamespace(concurrency=0))


class CompareCoverageTests(unittest.TestCase):
    def test_rejects_missing_duplicate_and_empty_captures(self):
        a = dict(length=128, index=0)
        b = dict(length=128, index=1)
        for left, right, error in [([a, b], [a], "prompt sets differ"),
                                    ([a], [a, b], "prompt sets differ"),
                                    ([a, a], [a], "duplicate prompt"),
                                    ([a], [a, a], "duplicate prompt"),
                                    ([], [], "contain no prompts")]:
            with self.subTest(left=left, right=right), tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / name for name in ("left.jsonl", "right.jsonl")]
                for path, rows in zip(paths, (left, right)):
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                args = SimpleNamespace(left=paths[0], right=paths[1], tokenizer="unused")
                with self.assertRaisesRegex(SystemExit, error):
                    quality.cmd_compare(args)


if __name__ == "__main__":
    unittest.main()
