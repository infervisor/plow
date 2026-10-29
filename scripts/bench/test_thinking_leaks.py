import tempfile
import types
import unittest
from unittest.mock import patch

import thinking_leaks as audit


class ThinkingLeaksTests(unittest.TestCase):
    def args(self, **extra):
        return types.SimpleNamespace(
            model="test", temperature=0, url="http://unused", raw=None, **extra)

    def test_all_http_errors_fail_the_audit(self):
        with patch.object(audit, "PROMPTS", ["Hi"]), patch.object(audit, "post", return_value=(
                400, {"error": {"message": "unsupported"}})):
            rows, fails, no_room = audit.audit(self.args(seeds=1))
        self.assertEqual(fails, len(audit.ARMS))
        self.assertTrue(all(r["status"] == 400 for r in rows))
        self.assertFalse(audit.report("test", rows, fails, no_room).endswith("none\n"))

    def test_http_errors_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as out, patch.object(audit, "PROMPTS", ["Hi"]), \
                patch.object(audit, "post", return_value=(400, {"error": {"message": "unsupported"}})), \
                patch("sys.argv", ["thinking_leaks.py", "--url", "http://unused", "--name", "test",
                                   "--model", "test", "--out", out]), patch("builtins.print"):
            self.assertEqual(audit.main(), 1)

    def test_empty_coverage_fails(self):
        with patch.object(audit, "post", return_value=(400, {"error": {"message": "no room"}})):
            rows, fails, _ = audit.audit(self.args(seeds=0))
        self.assertEqual(rows, [])
        self.assertGreater(fails, 0)

    def test_successful_coverage_passes_but_stream_errors_fail(self):
        for stream_status in [200, 400]:
            def post(url, body):
                if body.get("thinking_token_budget") == 300:
                    return 400, {"error": {"message": "no room"}}
                if body.get("stream"):
                    if stream_status != 200:
                        return stream_status, {"error": {"message": "stream refused"}}
                    return 200, [{"choices": [{"delta": {"content": "Hi"}}]}]
                return 200, {"choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]}

            with patch.object(audit, "PROMPTS", ["Hi"]), patch.object(audit, "post", side_effect=post):
                rows, fails, _ = audit.audit(self.args(seeds=1))
            self.assertEqual(fails, 0 if stream_status == 200 else len(audit.ARMS))
            if stream_status != 200:
                self.assertTrue(all("stream_error" in r["checks"] for r in rows))


if __name__ == "__main__":
    unittest.main()
