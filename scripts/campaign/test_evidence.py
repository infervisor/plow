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
    def fixture(self, root, lengths=(8192,), concurrencies=(16,)):
        runs = {}
        def entry_id(arm, metric, cell):
            suffix = f":in{cell[0]}-c{cell[1]}" if len(lengths) * len(concurrencies) > 1 else ""
            return f"{arm}:{metric}{suffix}"

        for arm in evidence.ARMS:
            run = root / arm
            (run / "client").mkdir(parents=True)
            runs[arm] = run
            record = {"utc": arm, "gate": True, "contended": False, "bench_rc": 0,
                      "cell": {"revision": "model-revision", "precision": "bf16"},
                      "gpu": {"name": "gfx950", "driver": "pinned"},
                      "protocol": {"OUTLEN": "128", "NPROMPT": "32",
                                   "IN_LENS": " ".join(map(str, lengths)),
                                   "CONCS": " ".join(map(str, concurrencies))},
                      "hashes": {"model.pkt": ("b" if arm.startswith("treat") else "a") * 64},
                      "serve_env": {}, "overrides": {}}
            record["execution_artifacts_unchanged"] = True
            record["execution_artifacts"] = {"assets": dict(record["hashes"]), "objects": {},
                **{field: "a" * 64 for field in ("runtime_sha256", "recipe_sha256",
                                               "runtime_environment_sha256", "serve_args_sha256")}}
            (run / "run-record.json").write_text(json.dumps(record))
            base = 0.001 if arm.startswith("treat") else 0.002
            for length in lengths:
                for concurrency in concurrencies:
                    client = {"input_lens": [length] * 32, "max_concurrency": concurrency,
                              "output_lens": [128] * 32,
                              "completed": 32, "ttfts": [base] * 32, "itls": [[base] * 127] * 32}
                    (run / f"client/in{length}_c{concurrency}.json").write_text(json.dumps(client))
        bundle = evidence.capture(runs)
        ledger = []
        for arm in evidence.ARMS:
            for item in bundle["arms"][arm]["clients"]:
                cell, metrics = evidence.samples(evidence.document(item))
                for metric, xs in metrics.items():
                    entry = {"id": entry_id(arm, metric, cell), "metric": metric,
                               "samples": xs, "stats": evidence.stats(xs), "better": "lower",
                               "rung": {"digest": f"glm/serve/in{cell[0]}-c{cell[1]}-out128",
                                        "prior": 0, "role": "serve", "rows": cell[0], "topology": f"C{cell[1]}"},
                               "recipe_digest": evidence.document(bundle["arms"][arm]["record"])["hashes"]["model.pkt"],
                               "sample_source": {"arm": arm, "input_len": cell[0], "concurrency": cell[1],
                                                 "client_sha256": item["sha256"]}}
                    if arm.startswith("treat"):
                        entry.update(control_of=entry_id("ctrl", metric, cell),
                                     repeat_control_of=entry_id("ctrl2", metric, cell),
                                     repeat_treatment_of=entry_id("treat2" if arm == "treat" else "treat", metric, cell))
                    ledger.append(entry)
        return runs, bundle, {"ledger": ledger, "tier4": True, "untouched": [],
            "touched": [{"rung": e["rung"]["digest"], "treat": e["id"]}
                        for e in ledger if e["sample_source"]["arm"] == "treat"],
            "serving": [e["id"] for e in ledger if e["sample_source"]["arm"].startswith("treat")]}

    def replace_document(self, item, document):
        item["text"] = json.dumps(document)
        item["sha256"] = evidence.digest(item["text"].encode())

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
            for field, value in (("completed", 31), ("itls", [[]] * 31),
                                 ("input_lens", [8192, 4096] * 16), ("ttfts", [float("nan")] * 32)):
                client = evidence.document(bundle["arms"]["ctrl"]["clients"][0])
                client[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    evidence.samples(client)

    def test_declared_matrix_requires_exact_coverage_even_when_every_arm_agrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, original, request = self.fixture(Path(tmp), (8192, 71680), (8, 16, 64))
            evidence.validate(original, request)
            for mutation in ("omitted", "additional"):
                bundle, req = copy.deepcopy(original), copy.deepcopy(request)
                for arm in evidence.ARMS:
                    if mutation == "omitted":
                        bundle["arms"][arm]["clients"] = [item for item in bundle["arms"][arm]["clients"]
                            if evidence.document(item)["input_lens"][0] == 8192]
                    else:
                        item = copy.deepcopy(bundle["arms"][arm]["clients"][0])
                        client = evidence.document(item)
                        client["input_lens"] = [4096] * 32
                        self.replace_document(item, client)
                        bundle["arms"][arm]["clients"].append(item)
                if mutation == "omitted":
                    req["ledger"] = [e for e in req["ledger"] if e["sample_source"]["input_len"] == 8192]
                    ids = {e["id"] for e in req["ledger"]}
                    req["touched"] = [t for t in req["touched"] if t["treat"] in ids]
                    req["serving"] = [i for i in req["serving"] if i in ids]
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "declared IN_LENS"):
                    evidence.validate(bundle, req)

    def test_malformed_duplicate_or_missing_declarations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, original, request = self.fixture(Path(tmp))
            for field in ("IN_LENS", "CONCS", "NPROMPT", "OUTLEN"):
                for value in (None, "", True, 8, [], "0", "-8", "8.0", "8,16", "8 08", "８"):
                    bundle = copy.deepcopy(original)
                    for arm in evidence.ARMS:
                        item = bundle["arms"][arm]["record"]
                        record = evidence.document(item)
                        if value is None:
                            record["protocol"].pop(field)
                        else:
                            record["protocol"][field] = value
                        self.replace_document(item, record)
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        evidence.validate(bundle, request)

    def test_metric_request_scope_and_relabelled_rungs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, original, request = self.fixture(Path(tmp))
            for mutation in range(12):
                bundle, req = copy.deepcopy(original), copy.deepcopy(request)
                if mutation == 0:
                    req["ledger"] = [e for e in req["ledger"] if e["metric"] != "tpot_ms"]
                elif mutation == 1:
                    req["touched"].pop()
                elif mutation == 2:
                    req["serving"].pop()
                elif mutation == 3:
                    req["touched"].append(req["touched"][0])
                elif mutation == 4:
                    req["serving"].append(req["serving"][0])
                elif mutation == 5:
                    req["tier4"] = False
                elif mutation == 6:
                    req["untouched"] = [{"rung": "unknown", "base": "same", "variant": "same"}]
                elif mutation == 7:
                    req["touched"][0]["rung"] = "glm/serve/in71680-c64-out128"
                elif mutation == 8:
                    for e in req["ledger"]:
                        e["rung"].update(digest="glm/serve/in71680-c64-out128", rows=71680, topology="C64")
                elif mutation == 9:
                    req["ledger"][0]["rung"]["role"] = "decode"
                elif mutation == 10:
                    req["ledger"][0]["better"] = "higher"
                else:
                    bundle["required_metrics"] = ["ttft_ms"]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    evidence.validate(bundle, req)

    def test_stream_chunks_use_output_tokens_and_exclude_zero_one_outputs(self):
        client = {"input_lens": [8192] * 4, "max_concurrency": 16, "completed": 4,
                  "ttfts": [.1] * 4, "output_lens": [5, 3, 1, 0],
                  "itls": [[.01, .03], [.02], [], []],
                  "mean_tpot_ms": 9.999, "median_tpot_ms": 9.999}
        cell, metrics = evidence.samples(client)
        self.assertEqual(cell, (8192, 16))
        self.assertEqual(metrics, {"ttft_ms": [100.0] * 4, "tpot_ms": [10.0, 10.0]})
        self.assertEqual(client["mean_tpot_ms"], 9.999)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "client").mkdir()
            (root / "client/in8192_c16.json").write_text(json.dumps(client))
            self.assertEqual(campaign._samples(root)[(8192, 16, "tpot_ms")], metrics["tpot_ms"])
        for field, values in (("output_lens", [5, 3, 1]), ("output_lens", [5, 3, 1, -1]),
                              ("output_lens", [5, 3, 1, True]), ("output_lens", [5, 3, 1, 2.0]),
                              ("itls", [[.01], [.02], [], [float("nan")]])):
            bad = {**client, field: values}
            with self.subTest(field=field, values=values), self.assertRaises(ValueError):
                evidence.samples(bad)

    def test_legacy_semantics_do_not_inherit_new_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, bundle, request = self.fixture(Path(tmp))
            self.assertEqual(bundle["schema"], 2)
            self.assertEqual(bundle["sample_semantics"], evidence.SAMPLE_SEMANTICS)
            bundle["schema"] = 1
            with self.assertRaisesRegex(ValueError, "historical mean-ITL"):
                evidence.validate(bundle, request)

    def exact_fixture(self, root):
        runs, bundle, request = self.fixture(root)
        for arm in evidence.ARMS:
            item = bundle["arms"][arm]["clients"][0]
            client = evidence.document(item)
            client.update(client_latency_export=evidence.export_identity(),
                          request_success=[True] * 32,
                          request_latencies=[first + sum(row) + 0.0001
                              for first, row in zip(client["ttfts"], client["itls"])])
            _, metrics = evidence.samples(client)
            for metric, xs in metrics.items():
                client[f"mean_{metric}"] = sum(xs) / len(xs)
                client[f"median_{metric}"] = evidence.stats(xs)["median"]
            self.replace_document(item, client)
            (runs[arm] / "client/in8192_c16.json").write_text(item["text"])
            for entry in request["ledger"]:
                if entry["sample_source"]["arm"] == arm:
                    xs = metrics[entry["metric"]]
                    entry.update(samples=xs, stats=evidence.stats(xs))
                    entry["sample_source"]["client_sha256"] = item["sha256"]
        return runs, evidence.capture(runs), request

    def test_exact_latencies_preferred_and_pinned_aggregates_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, bundle, request = self.exact_fixture(Path(tmp))
            self.assertEqual(bundle["schema"], 3)
            self.assertEqual(bundle["sample_semantics"], evidence.EXACT_SAMPLE_SEMANTICS)
            evidence.validate(bundle, request)
            client = evidence.document(bundle["arms"]["ctrl"]["clients"][0])
            _, metrics = evidence.samples(client)
            expected = (client["request_latencies"][0] - client["ttfts"][0]) / 127 * 1000
            reconstructed = sum(client["itls"][0]) / 127 * 1000
            self.assertEqual(metrics["tpot_ms"][0], expected)
            self.assertNotEqual(expected, reconstructed)
            self.assertEqual(campaign._samples(runs["ctrl"])[(8192, 16, "tpot_ms")], metrics["tpot_ms"])
            for mutation in range(8):
                bad = copy.deepcopy(bundle)
                item = bad["arms"]["ctrl"]["clients"][0]
                client = evidence.document(item)
                if mutation == 0:
                    client["request_latencies"].pop()
                elif mutation == 1:
                    client["request_success"][0] = False
                elif mutation == 2:
                    client["client_latency_export"]["overlay_sha256"] = "0" * 64
                elif mutation == 3:
                    client["mean_tpot_ms"] += .001
                elif mutation == 4:
                    client.pop("median_ttft_ms")
                elif mutation == 5:
                    client["request_latencies"][0] = float("nan")
                elif mutation == 6:
                    bad.update(schema=2, sample_semantics=evidence.SAMPLE_SEMANTICS)
                else:
                    bad["client_identity"]["endpoint_sha256"] = "0" * 64
                self.replace_document(item, client)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    evidence.validate(bad, request)

    def test_mixed_client_exports_and_false_exact_claims_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, bundle, request = self.exact_fixture(Path(tmp))
            item = bundle["arms"]["ctrl"]["clients"][0]
            client = evidence.document(item)
            for field in ("request_latencies", "request_success", "client_latency_export"):
                client.pop(field)
            (runs["ctrl"] / "client/in8192_c16.json").write_text(json.dumps(client))
            with self.assertRaisesRegex(ValueError, "cannot mix"):
                evidence.capture(runs)
            self.replace_document(item, client)
            with self.assertRaises(ValueError):
                evidence.validate(bundle, request)

    def test_exact_zero_one_outputs_follow_client_tpot_exclusion(self):
        client = {"input_lens": [8192] * 4, "max_concurrency": 16, "completed": 4,
                  "ttfts": [.1] * 4, "output_lens": [5, 3, 1, 0],
                  "itls": [[.01, .03], [.02], [], []],
                  "request_latencies": [.15, .16, .1, .1], "request_success": [True] * 4,
                  "client_latency_export": evidence.export_identity()}
        _, metrics = evidence.samples(client)
        self.assertEqual(metrics["tpot_ms"], [(.15 - .1) / 4 * 1000, (.16 - .1) / 2 * 1000])

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / "lean-plow/.lake/build/bin/plow_verify").is_file(),
                         "requires built Lean verifier")
    def test_actual_perf_certificate_make_and_replay_bind_raw_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, bundle, request = self.exact_fixture(root)
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
