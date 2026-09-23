import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import client_latency
import evidence


class ClientLatencyTests(unittest.TestCase):
    def test_wrapper_requires_detailed_result_artifacts(self):
        wrapper = Path(__file__).resolve().parents[1] / "bench/vllm029-client.sh"
        result = subprocess.run(["bash", str(wrapper), "--plow-exact-latencies"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("requires --result-dir", result.stderr)

    def test_unpinned_source_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "client-overlay"
            with self.assertRaisesRegex(ValueError, "source identity"):
                client_latency.write_overlay(output, "# changed client", client_latency.ENDPOINT_SHA256)
            self.assertFalse(output.exists())

    @unittest.skipUnless(os.environ.get("PB_TEST_PINNED_CLIENT") == "1", "CPU Docker source replay opt-in")
    def test_pinned_overlay_changes_only_post_timing_exports_and_replays_fields(self):
        source = client_latency.fetch_source()
        original = source["source"]
        overlay = client_latency.overlay_source(original, source["endpoint_sha256"])
        changed = ast.parse(overlay)
        result = next(node for node in ast.walk(changed) if isinstance(node, ast.Dict)
                      and any(isinstance(k, ast.Constant) and k.value == "client_latency_export" for k in node.keys))
        added = copy.deepcopy(result)
        exports = {"request_latencies", "request_success", "client_latency_export"}
        keep = [i for i, k in enumerate(result.keys) if not isinstance(k, ast.Constant) or k.value not in exports]
        result.keys = [result.keys[i] for i in keep]
        result.values = [result.values[i] for i in keep]
        self.assertEqual(ast.dump(changed), ast.dump(ast.parse(original)))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "client-overlay"
            identity = client_latency.write_overlay(directory, original, source["endpoint_sha256"])
            self.assertEqual(identity, client_latency.export_identity())
            self.assertEqual(client_latency.write_overlay(directory, original, source["endpoint_sha256"]), identity)
            self.assertEqual((directory / "serve.py").read_text(), overlay)
            # Evaluate only the added post-timing fields using actual source AST.
            keep = [i for i, k in enumerate(added.keys) if isinstance(k, ast.Constant) and k.value in exports]
            added.keys = [added.keys[i] for i in keep]
            added.values = [added.values[i] for i in keep]
            outputs = [SimpleNamespace(latency=.15, success=True), SimpleNamespace(latency=.3, success=False)]
            exported = eval(compile(ast.Expression(added), "overlay-exports", "eval"),
                            {"outputs": outputs, "__file__": str(directory / "serve.py")})
            self.assertEqual(exported["request_latencies"], [.15, .3])
            self.assertEqual(exported["request_success"], [True, False])
            self.assertEqual(exported["client_latency_export"], identity)
            (directory / "serve.py").chmod(0o644)
            (directory / "serve.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                client_latency.write_overlay(directory, original, source["endpoint_sha256"])
        with self.assertRaises(ValueError):
            client_latency.overlay_source(original, "0" * 64)

    @unittest.skipUnless(os.environ.get("PB_TEST_PINNED_CLIENT") == "1", "CPU Docker source replay opt-in")
    def test_wrapper_mount_and_actual_client_metrics_match_exported_fields(self):
        wrapper = Path(__file__).resolve().parents[1] / "bench/vllm029-client.sh"
        code = '''
import ast, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import vllm.benchmarks.serve as serve
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
serve.TERM_PLOTLIB_AVAILABLE = False
outputs = [RequestFuncOutput(success=True, latency=latency, ttft=.1,
    output_tokens=n, itl=intervals, prompt_len=8192)
    for n, latency, intervals in [(5, .1501, [.02, .03]), (3, .1601, [.06]),
                                  (1, .1, []), (0, .1, [])]]
metrics, actual_output_lens = serve.calculate_metrics([], outputs, 1.0,
    lambda *a, **k: SimpleNamespace(input_ids=[]), [], {})
tree = ast.parse(Path(serve.__file__).read_text())
fields = {"output_lens", "ttfts", "itls", "request_latencies", "request_success", "client_latency_export"}
node = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict)
    and any(isinstance(k, ast.Constant) and k.value == "request_latencies" for k in n.keys))
keep = [i for i, k in enumerate(node.keys) if isinstance(k, ast.Constant) and k.value in fields]
node.keys = [node.keys[i] for i in keep]
node.values = [node.values[i] for i in keep]
client = eval(compile(ast.Expression(node), "actual-client-exports", "eval"),
    dict(outputs=outputs, actual_output_lens=actual_output_lens, __file__=serve.__file__))
client.update(input_lens=[8192] * 4, max_concurrency=16, completed=metrics.completed)
for metric in ("ttft_ms", "tpot_ms"):
    for stat in ("mean", "median"):
        key = f"{stat}_{metric}"
        client[key] = getattr(metrics, key)
print("CLIENT_FIXTURE=" + json.dumps(client))
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(["bash", str(wrapper), "-c", code, "--plow-exact-latencies",
                                     "--result-dir", tmp, "--save-result", "--save-detailed"],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            raw = next(line.removeprefix("CLIENT_FIXTURE=") for line in result.stdout.splitlines()
                       if line.startswith("CLIENT_FIXTURE="))
            client = json.loads(raw)
            self.assertEqual(client["client_latency_export"], client_latency.export_identity())
            self.assertEqual(client["output_lens"], [5, 3, 1, 0])
            self.assertEqual(client["request_latencies"], [.1501, .1601, .1, .1])
            _, metrics = evidence.samples(client)
            self.assertEqual(len(metrics["tpot_ms"]), 2)
            evidence.check_reported_aggregates(client, metrics)


if __name__ == "__main__":
    unittest.main()
