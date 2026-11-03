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


class ObjectTests(unittest.TestCase):
    native = ("interp_decode.elf", "interp_prefill.elf", "interp_flash.elf")

    def check_objects(self, objects, arch, assets):
        return subprocess.run([
            "bash", "-c", 'source "$1"; pb_check_objects "$2" "$3" "$4"',
            "test", str(HELPERS), str(objects), arch, str(assets),
        ], capture_output=True, text=True)

    def test_native_architectures_require_their_interpreters_not_cdna_vendor_kernels(self):
        for arch in ("gfx950", "gfx1201"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in self.native:
                    (root / name).touch()
                self.assertEqual(self.check_objects(root, arch, root).returncode, 0)
                for name in self.native:
                    (root / name).unlink()
                    result = self.check_objects(root, arch, root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(name, result.stdout)
                    (root / name).touch()

    def test_gfx942_still_requires_pinned_vendor_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.native:
                (root / name).touch()
            result = self.check_objects(root, "gfx942", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pinned vendor kernel", result.stdout)

    def test_gfx950_mixed_precision_still_requires_specialized_interpreters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.native:
                (root / name).touch()
            (root / "build.json").write_text(json.dumps({
                "features": {k: True for k in ("fp8_kv", "a4w4", "mla", "moe")},
            }))
            result = self.check_objects(root, "gfx950", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mixed FP8-KV/MXFP4", result.stdout)
            for name in ("interp_decode_fp8kv.elf", "interp_decode_fp8kv_gq.elf",
                         "interp_flash_fp8kv.elf", "interp_flash_fp8kv_gq.elf",
                         "interp_prefill_fp8kv_mla_moe_a4w4_full.elf",
                         "interp_prefill_fp8kv_mla_moe_a4w4_full_gq.elf"):
                (root / name).touch()
            self.assertEqual(self.check_objects(root, "gfx950", root).returncode, 0)


class BenchClientTests(unittest.TestCase):
    def test_console_script_and_python_launcher(self):
        for name, prefix in (("vllm", ["bench", "serve"]),
                             ("vllm-python", ["-m", "vllm.entrypoints.cli.main", "bench", "serve"])):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                client = root / name
                client.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                                  "open(os.environ['TEST_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n")
                client.chmod(0o755)
                subprocess.run(["bash", "-c", 'source "$1"; PB_SERVER_PORT=1 pb_bench "$2" t m 1 1 8 8',
                                "test", str(HELPERS), str(root / "res")], check=True,
                               env=dict(os.environ, PB_VLLM=str(client), TEST_ARGV=str(root / "argv")))
                argv = json.loads((root / "argv").read_text())
                self.assertEqual(argv[:len(prefix)], prefix)


class HostSamplerTests(unittest.TestCase):
    def test_host_start_stop_writes_session_summary(self):
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with tempfile.TemporaryDirectory() as directory:
                subprocess.run(["bash", "-c", 'source "$1"; pb_host_start "$2" "$3"; sleep 1; pb_host_stop',
                                "test", str(HELPERS), directory, str(server.pid)],
                               check=True, env=dict(os.environ, PB_HOST_MS="200"), timeout=20)
                summary = json.loads((Path(directory) / "host.json").read_text())
                self.assertEqual(summary["procs"]["peak"], 1)
                self.assertGreater(summary["rss_mib"]["peak"], 0)
                self.assertNotIn("pss_mib", summary)
                self.assertEqual(summary["processes"][0]["pid"], server.pid)
                self.assertTrue((Path(directory) / "host.tsv").read_text().startswith("t\tprocs"))
        finally:
            server.kill()
            server.wait()


if __name__ == "__main__":
    unittest.main()
