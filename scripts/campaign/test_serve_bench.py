import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import campaign
import client_latency


class ServeBenchTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        (raw / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
        (raw / "part.safetensors").write_bytes(b"fixture")
        assets, objects = root / "assets", root / "objects"
        assets.mkdir()
        objects.mkdir()
        (assets / "model.pkt").write_bytes(b"packet")
        (assets / "build.json").write_text(json.dumps({"precision": {
            "weight_enc": "fp8", "act_enc": "bf16", "kv_enc": "bf16", "expert_enc": "fp8blk"}}))
        (objects / "decode.elf").write_bytes(b"object")
        runtime = root / "plowrt"
        runtime.write_bytes(b"runtime")
        runtime.chmod(0o755)
        recipe = root / "recipe.toml"
        recipe.write_text(f'''[cell]
name="test"
hf_dir="{raw}"
arch="gfx950"
n_gpu=8
max_ctx=73728
[emit]
[bench]
in_lens="8192 71680"
concs="8 64"
nprompt=128
outlen=128
[reference.env]
VLLM_ROCM_USE_AITER="1"
''')
        return argparse.Namespace(recipe=str(recipe), assets=str(assets), objects=str(objects),
                                  out=str(root / "run"), plowrt=str(runtime), env=None,
                                  in_lens=None, concs=None, nprompt=None, server="plow",
                                  queue=str(root / "queue"), dry_run=True)

    def test_both_servers_share_client_protocol_and_dry_run_does_not_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            args.quality_lens = "8192,71680"
            scripts = []
            for server in ("plow", "vllm"):
                args.server = server
                args.out = str(root / server)
                with patch.object(campaign, "run", return_value=0), patch.dict(os.environ, ROCM_PATH="/rocm"):
                    campaign.cmd_serve_bench(args)
                text = (Path(args.out) / "run.sh").read_text()
                scripts.append([line[line.index("in8192"):] for line in text.splitlines() if line.startswith("pb_bench ") and "in8192" in line])
                record = json.loads((Path(args.out) / "run-record.json").read_text())
                self.assertEqual(record["status"], "prepared")
                self.assertFalse(record["exact_request_latencies"])
                self.assertIsNone(record["expected_client_identity"])
                self.assertNotIn("--plow-exact-latencies", text)
                self.assertFalse(record["numerics_qualified"])
                self.assertFalse(record["precision_qualified"])
                self.assertEqual(record["plow_declared_precision"]["act_enc"], "bf16")
                self.assertIn("diagnostic", record["comparison_scope"])
                self.assertNotIn("job", record)
                self.assertIn("--backend openai --endpoint /v1/completions", text)
                self.assertIn("--lens 8192,71680 --exact-lengths", text)
                self.assertEqual(record["quality_probe_sha256"], campaign.sha(Path(args.out) / "needle_probe.py"))
                self.assertIn('"$smoke_payload"', text)
                self.assertNotIn("kill -", text)
                if server == "vllm":
                    self.assertIn("-e VLLM_ROCM_USE_AITER=1", text)
                    self.assertIn("--host 127.0.0.1", text)
                    self.assertEqual(record["reference_env"], {"VLLM_ROCM_USE_AITER": "1"})
                    self.assertIn('"${visibility[@]}"', text)
                    self.assertNotIn("queue visibility missing", text)
                    visibility = "\n".join(line for line in text.splitlines()
                                           if line.startswith(("visibility=", "if test -n")))
                    for mask in (None, "2,3"):
                        env = dict(os.environ)
                        for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
                            env.pop(key, None)
                        if mask is not None:
                            env["ROCR_VISIBLE_DEVICES"] = mask
                        result = subprocess.run(["bash", "-c", visibility + '\nprintf "%s\\n" "${visibility[@]}"'],
                                                env=env, check=True, capture_output=True, text=True)
                        self.assertEqual(result.stdout.strip(), "" if mask is None else "-e\nROCR_VISIBLE_DEVICES=2,3")
            self.assertEqual(scripts[0], scripts[1])
            self.assertFalse(Path(args.queue).exists())

    def test_exact_export_is_same_opt_in_for_both_servers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            recipe = Path(args.recipe)
            recipe.write_text(recipe.read_text().replace("[bench]", "[bench]\nexact_request_latencies=true"))
            protocols = []
            for server in ("plow", "vllm"):
                args.server = server
                args.out = str(root / server)
                with patch.object(campaign, "run", return_value=0), patch.dict(os.environ, ROCM_PATH="/rocm"):
                    campaign.cmd_serve_bench(args)
                out = Path(args.out)
                record = json.loads((out / "run-record.json").read_text())
                self.assertTrue(record["exact_request_latencies"])
                self.assertEqual(record["expected_client_identity"], client_latency.export_identity())
                self.assertEqual(record["client_exporter_sha256"], campaign.sha(out / "client_latency.py"))
                lines = [line.split('"$model"', 1)[1] for line in (out / "run.sh").read_text().splitlines()
                         if line.startswith("pb_bench ")]
                self.assertEqual(len(lines), 4)
                self.assertTrue(all(line.endswith("--plow-exact-latencies") for line in lines))
                protocols.append(lines)
            self.assertEqual(*protocols)

    def test_partial_checkpoint_refused_before_preparing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            (root / "raw/part.safetensors").unlink()
            with self.assertRaises(SystemExit):
                campaign.cmd_serve_bench(args)
            self.assertFalse(Path(args.out).exists())


if __name__ == "__main__":
    unittest.main()
