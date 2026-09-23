import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import evidence
import campaign


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        runs = {}
        for arm in evidence.ARMS:
            run = root / arm
            (run / "client").mkdir(parents=True)
            runs[arm] = run
            record = {"utc": arm, "gate": True, "contended": False, "bench_rc": 0,
                      "cell": {"revision": "model-revision", "precision": "bf16"},
                      "gpu": {"name": "gfx950", "driver": "pinned"},
                      "protocol": {"OUTLEN": "128", "NPROMPT": "32"},
                      "hashes": {"model.pkt": ("b" if arm.startswith("treat") else "a") * 64},
                      "serve_env": {}, "overrides": {}}
            record["execution_artifacts_unchanged"] = True
            record["execution_artifacts"] = {"assets": dict(record["hashes"]), "objects": {},
                **{field: "a" * 64 for field in ("runtime_sha256", "recipe_sha256",
                                               "runtime_environment_sha256", "serve_args_sha256")}}
            (run / "run-record.json").write_text(json.dumps(record))
            base = 0.001 if arm.startswith("treat") else 0.002
            client = {"input_lens": [8192] * 32, "max_concurrency": 16,
                      "output_lens": [128] * 32,
                      "completed": 32, "ttfts": [base] * 32, "itls": [[base] * 127] * 32}
            (run / "client/in8192_c16.json").write_text(json.dumps(client))
        bundle = evidence.capture(runs)
        ledger = []
        for arm in evidence.ARMS:
            item = bundle["arms"][arm]["clients"][0]
            cell, metrics = evidence.samples(evidence.document(item))
            for metric, xs in metrics.items():
                entry = {"id": f"{arm}:{metric}", "metric": metric, "samples": xs, "stats": evidence.stats(xs),
                               "recipe_digest": evidence.document(bundle["arms"][arm]["record"])["hashes"]["model.pkt"],
                               "sample_source": {"arm": arm, "input_len": cell[0], "concurrency": cell[1],
                                                 "client_sha256": item["sha256"]}}
                if arm.startswith("treat"):
                    entry.update(control_of=f"ctrl:{metric}", repeat_control_of=f"ctrl2:{metric}",
                                 repeat_treatment_of=f"{'treat2' if arm == 'treat' else 'treat'}:{metric}")
                ledger.append(entry)
        return runs, bundle, {"ledger": ledger}

    def test_portable_evidence_replays_all_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, bundle, request = self.fixture(Path(tmp))
            evidence.validate(json.loads(json.dumps(bundle)), request)

    def test_execution_fingerprints_observe_actual_files_and_effective_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets, objects = root / "assets", root / "objects"
            assets.mkdir()
            objects.mkdir()
            runtime, recipe = root / "plowrt", root / "recipe.toml"
            runtime.write_bytes(b"runtime")
            recipe.write_text("[cell]\n")
            (assets / "model.pkt").write_bytes(b"packet")
            (assets / "build.json").write_text("{}")
            (objects / "decoder.elf").write_bytes(b"object")
            env = {"PLOW_TEST": "0", "SERVE_EXTRA_ARGS": "--tp 8"}
            before = campaign.execution_artifacts(runtime, assets, recipe, env)
            self.assertEqual(before, campaign.execution_artifacts(runtime, assets, recipe, env))
            for file in (runtime, recipe, assets / "model.pkt", assets / "build.json", objects / "decoder.elf"):
                old = file.read_bytes()
                file.write_bytes(old + b"mutation")
                self.assertNotEqual(before, campaign.execution_artifacts(runtime, assets, recipe, env))
                file.write_bytes(old)
            for key in env:
                self.assertNotEqual(before, campaign.execution_artifacts(runtime, assets, recipe, {**env, key: "changed"}))

    def test_tamper_and_omission_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, original, request = self.fixture(Path(tmp))
            for mutation in range(15):
                bundle, req = copy.deepcopy(original), copy.deepcopy(request)
                if mutation < 7 or mutation >= 13:
                    item = bundle["arms"]["ctrl2"]["record"]
                    record = evidence.document(item)
                    if mutation == 0:
                        record["gate"] = "false"
                    elif mutation == 1:
                        record["contended"] = True
                    elif mutation == 2:
                        record["protocol"]["OUTLEN"] = "256"
                    elif mutation == 3:
                        record["hashes"]["model.pkt"] = "c" * 64
                    elif mutation == 4:
                        record["gpu"]["driver"] = "changed"
                    elif mutation == 5:
                        record["cell"]["precision"] = "fp8"
                    elif mutation == 6:
                        record["bench_rc"] = 1
                    elif mutation == 13:
                        record["execution_artifacts_unchanged"] = False
                    else:
                        record["execution_artifacts"]["runtime_sha256"] = "b" * 64
                    item["text"] = json.dumps(record)
                    item["sha256"] = evidence.digest(item["text"].encode())
                elif mutation == 7:
                    bundle["arms"]["ctrl"]["clients"][0]["text"] += " "
                elif mutation == 8:
                    req["ledger"][0]["samples"][0] *= 0.5
                elif mutation == 9:
                    req["ledger"][0]["stats"]["mad"] = 1
                elif mutation == 10:
                    req["ledger"].pop()
                elif mutation == 11:
                    bundle["arms"]["ctrl2"] = bundle["arms"]["ctrl"]
                else:
                    req["ledger"][2]["control_of"] = req["ledger"][2]["repeat_treatment_of"]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    evidence.validate(bundle, req)

    def test_duplicate_run_and_partial_samples_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, bundle, _ = self.fixture(Path(tmp))
            runs["ctrl2"] = runs["ctrl"]
            with self.assertRaises(ValueError):
                evidence.capture(runs)
            for field, value in (("completed", 31), ("itls", [[]] * 32),
                                 ("input_lens", [8192, 4096] * 16), ("ttfts", [float("nan")] * 32)):
                client = evidence.document(bundle["arms"]["ctrl"]["clients"][0])
                client[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    evidence.samples(client)

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / "lean-plow/.lake/build/bin/plow_verify").is_file(),
                         "requires built Lean verifier")
    def test_actual_perf_certificate_make_and_replay_bind_raw_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, bundle, request = self.fixture(root)
            rung = "glm/serve/in8192-c16-out128"
            for entry in request["ledger"]:
                entry.update(job="fixture", harness="campaign-bench", better="lower",
                             hardware={"box": "8xgfx950", "rocm": "7.14", "driver": None, "firmware": None},
                             rung={"digest": rung, "prior": 0, "role": "serve", "rows": 8192, "topology": "C16"})
            request.update(touched=[{"rung": rung, "treat": "treat:ttft_ms"},
                                    {"rung": rung, "treat": "treat:tpot_ms"}], untouched=[], tier4=True,
                           serving=[f"{arm}:{metric}" for arm in ("treat", "treat2") for metric in ("ttft_ms", "tpot_ms")],
                           numeric=False, facts=[{"kind": "gate", "pass": True, "evidence": "synthetic fixture"}])
            (root / "request.json").write_text(json.dumps(request))
            (root / "evidence.json").write_text(json.dumps(bundle))
            script = Path(__file__).resolve().parents[1] / "perf_cert.py"
            cert = root / "cert.json"
            command = [sys.executable, str(script)]
            make = subprocess.run(command + ["make", "--knob", "fixture", "--request", str(root / "request.json"),
                                            "--evidence", str(root / "evidence.json"), "--out", str(cert)],
                                  capture_output=True, text=True)
            self.assertEqual(make.returncode, 0, make.stdout + make.stderr)
            replay = subprocess.run(command + ["verify", str(cert)], capture_output=True, text=True)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            doc = json.loads(cert.read_text())
            self.assertEqual(doc["schema"], 2)
            doc["request"]["ledger"][0]["samples"][0] *= 0.1
            cert.write_text(json.dumps(doc))
            reject = subprocess.run(command + ["verify", str(cert)], capture_output=True, text=True)
            self.assertEqual(reject.returncode, 1, reject.stdout + reject.stderr)


if __name__ == "__main__":
    unittest.main()
