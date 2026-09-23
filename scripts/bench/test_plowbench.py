import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HELPERS = Path(__file__).resolve().with_name("plowbench.sh")


class ServerLifecycleTests(unittest.TestCase):
    def check_stop(self, ignore_term):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "server"
            server.write_text(
                f"#!{sys.executable}\n"
                "import os, signal, time\n"
                "from pathlib import Path\n"
                "if os.environ['TEST_IGNORE_TERM'] == '1':\n"
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(os.environ['TEST_PIDFILE']).write_text(str(os.getpid()))\n"
                "time.sleep(600)\n"
            )
            server.chmod(0o755)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            try:
                result = subprocess.run([
                    "bash", "-c",
                    'source "$1"; pb_serve_start "$2" assets objects 12345 "$3" 600; '
                    'for i in {1..100}; do test -s "$TEST_PIDFILE" && break; sleep .05; done; '
                    'test -s "$TEST_PIDFILE" || exit 2; pb_serve_stop',
                    "test", str(HELPERS), str(server), str(root / "server.log"),
                ], env=dict(os.environ, TEST_PIDFILE=str(root / "pid"),
                            TEST_IGNORE_TERM=str(int(ignore_term))), timeout=25)
                self.assertEqual(result.returncode, 0)
                child_pid = int((root / "pid").read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_stop_only_owned_server(self):
        self.check_stop(False)

    def test_timeout_escalates_only_owned_server(self):
        self.check_stop(True)


class ResultTests(unittest.TestCase):
    def test_rejects_partial_tokens_missing_metrics_and_nonfinite(self):
        result = dict(completed=8, total_output_tokens=1024, output_throughput=3000, request_throughput=10)
        result.update({f"{stat}_{metric}_ms": 1.0 for stat in ("mean", "median", "p99")
                       for metric in ("ttft", "tpot", "itl", "e2el")})
        cases = [(result, 0), (dict(result, completed=7), 1),
                 (dict(result, total_output_tokens=8), 1),
                 (dict(result, mean_ttft_ms=float("nan")), 1),
                 ({k: v for k, v in result.items() if k != "p99_e2el_ms"}, 1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for value, rc in cases:
                path.write_text(json.dumps(value))
                check = subprocess.run(["bash", "-c", 'source "$1"; pb_validate_result "$2" 8 128',
                                        "test", str(HELPERS), str(path)], capture_output=True)
                self.assertEqual(check.returncode, rc, check.stderr)


if __name__ == "__main__":
    unittest.main()
