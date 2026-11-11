import copy
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from tag_gate import make_cases, score, native_grid, digest
import recipe_common as rc


def config(mode="gate"):
    return dict(policy=dict(version=1, profile="test", map=rc.default_speech_tag_map()),
                settings=dict(languages=["en"], cer_max=.05), mode=mode, packet_sha256="a" * 64)


def capture_fixture(directory, cfg):
    captures, asr = [], []
    for i, case in enumerate(make_cases(cfg)):
        row = {**case, "case_id": case["id"], "status": case["expected_status"], "response_headers": {}}
        del row["id"]
        if row["status"] != 200:
            row["expected_error"] = True
            captures.append(row)
            continue
        if case["gate_kind"] == "headers":
            row["response_headers"] = {k: ",".join(v) for k, v in case["expected_headers"].items()}
        pcm = [.2] * 100 + [.3] * 100
        if case["gate_kind"] == "pause":
            zeros = [0.] * case["pause_ms"]
            pcm = {"leading": zeros + pcm, "internal": pcm[:100] + zeros + pcm[100:], "trailing": pcm + zeros}[case["pause_position"]]
        row.update(wav=f"{i:04d}.wav", reference_language=case["reference_language"])
        with wave.open(str(directory / row["wav"]), "wb") as wav:
            wav.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<" + "h" * len(pcm), *(int(s * 32768) for s in pcm)))
        captures.append(row)
        asr.append(dict(case_id=row["case_id"], text=row["reference"], language=row["reference_language"], hyp=row["reference"], cer=0))
    return captures, dict(rows=asr)


class TagGateTests(unittest.TestCase):
    def test_cases_cover_fixture_literals_leniency_and_signal_pairs(self):
        cases = make_cases(config())
        self.assertEqual(len(cases), 66)
        self.assertEqual(sum(r["gate_kind"] == "lenient" for r in cases), 14)
        self.assertEqual(sum(r["gate_kind"] == "literal" for r in cases), 8)
        self.assertEqual(sum(r["gate_kind"] == "identity" for r in cases), 4)
        self.assertEqual(sum(r["gate_kind"] == "pause" for r in cases), 6)
        cfg = config()
        cfg["settings"].update(languages=["zh"], texts={"zh": ["今天有小雨。", "大家在家中喝水。"]}, request_languages={"zh": "chinese"})
        rows = make_cases(cfg)
        self.assertTrue(all(r["language"] == "chinese" for r in rows))
        self.assertTrue(any("[pause]" in r["text"] and "今天" in r["text"] for r in rows))
        cfg["settings"].update(languages=["en"], request_languages={"en": ""})
        self.assertTrue(all(r["language"] == "" and r["reference_language"] == "en" for r in make_cases(cfg)))

    def test_full_score_and_cpu_rescore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, asr = capture_fixture(root, config())
            report = score(config(), rows, asr, root)
            self.assertTrue(report["pass"], report["why"])
            self.assertEqual(len(report["pauses"]), 3)
            self.assertEqual(report["config_sha256"], digest(config()))

    def test_each_contract_failure_fails_gate(self):
        mutations = [
            lambda rows, asr: rows.pop(),
            lambda rows, asr: asr["rows"].pop(),
            lambda rows, asr: rows[0].update(status=500),
            lambda rows, asr: next(r for r in rows if r["gate_kind"] == "lenient").update(status=400),
            lambda rows, asr: next(r for r in rows if r["gate_kind"] == "headers").update(response_headers={}),
            lambda rows, asr: next(r for r in asr["rows"] if r["case_id"].endswith("br_calm")).update(hyp="calm was spoken"),
            lambda rows, asr: next(r for r in asr["rows"] if r["case_id"].endswith("br_sighs")).update(cer=.051),
            lambda rows, asr: asr["rows"][0].update(cer=.051),
            lambda rows, asr: rows[0].update(seed=999),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows, asr = capture_fixture(root, config())
                mutate(rows, asr)
                self.assertFalse(score(config(), rows, asr, root)["pass"])

    def test_literal_and_pause_audio_failures(self):
        for kind in ("literal", "pause"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows, asr = capture_fixture(root, config())
                target = next(r for r in rows if r["gate_kind"] == kind and (kind != "pause" or r["pause_ms"] == 600))
                (root / target["wav"]).write_bytes((root / rows[0]["wav"]).read_bytes())
                if kind == "literal":
                    with wave.open(str(root / target["wav"]), "wb") as wav:
                        wav.setparams((1, 2, 1000, 0, "NONE", "not compressed")); wav.writeframes(b"\0\0" * 200)
                self.assertFalse(score(config(), rows, asr, root)["pass"])

    def test_reject_policy_and_measured_native_sweep(self):
        cfg = config()
        cfg["policy"]["map"]["tags"]["laugh"] = dict(reject=True)
        with tempfile.TemporaryDirectory() as tmp:
            rows, asr = capture_fixture(Path(tmp), cfg)
            self.assertTrue(score(cfg, rows, asr, Path(tmp))["pass"])
        cfg = config("native-sweep")
        cfg["policy"]["map"]["tags"]["laugh"] = dict(text="<laugh>", langs=["en"])
        cfg["settings"].update(seeds=[1, 2, 3, 4, 5], native_metric="cer")
        self.assertEqual(len(native_grid(cfg)), 4)
        with tempfile.TemporaryDirectory() as tmp:
            rows, asr = capture_fixture(Path(tmp), cfg)
            for row in asr["rows"]:
                row["cer"] = .04 if row["case_id"].endswith("baseline") else .01
            report = score(cfg, rows, asr, Path(tmp))
            self.assertTrue(report["pass"], report["why"])
            self.assertEqual(len(report["native"]["cells"]), 4)
            asr["rows"][-1]["cer"] = .3
            self.assertFalse(score(cfg, rows, asr, Path(tmp))["pass"])
            cfg["settings"]["seeds"] = [1, 2, 3, 4]
            with self.assertRaisesRegex(ValueError, ">=5"):
                make_cases(cfg)

    def test_native_receipt_is_bound_and_zero_improvement_cannot_pass(self):
        sweep = config("native-sweep")
        sweep["policy"]["map"]["tags"]["laugh"] = dict(text="<laugh>")
        sweep["settings"].update(seeds=[1, 2, 3, 4, 5], native_metric="cer")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, asr = capture_fixture(root, sweep)
            self.assertFalse(score(sweep, rows, asr, root)["pass"])
            for row in asr["rows"]:
                row["cer"] = .04 if row["case_id"].endswith("baseline") else .01
            receipt = score(sweep, rows, asr, root)
            self.assertTrue(receipt["pass"])
            saved = root / "native.json"
            saved.write_text(json.dumps(receipt))
            gate = copy.deepcopy(sweep)
            gate.update(mode="gate", native_sweep=str(saved))
            rows, asr = capture_fixture(root, gate)
            self.assertTrue(score(gate, rows, asr, root)["pass"])
            for mutate in (lambda cfg: cfg.update(packet_sha256="different"),
                           lambda cfg: cfg.update(runtime_sha256="different"),
                           lambda cfg: cfg["settings"].update(voice="different")):
                changed = copy.deepcopy(gate)
                mutate(changed)
                rows, asr = capture_fixture(root, changed)
                report = score(changed, rows, asr, root)
                self.assertFalse(report["pass"])
                self.assertIn("different map/packet", ";".join(report["why"]))

    def test_threshold_and_native_evidence_refusals(self):
        cfg = config()
        cfg["settings"]["cer_margin"] = .1
        with self.assertRaisesRegex(ValueError, "owner decision"):
            make_cases(cfg)
        cfg = config()
        cfg["policy"]["map"]["tags"]["laugh"] = dict(text="<laugh>")
        with tempfile.TemporaryDirectory() as tmp:
            rows, asr = capture_fixture(Path(tmp), cfg)
            report = score(cfg, rows, asr, Path(tmp))
            self.assertFalse(report["pass"])
            self.assertIn("native map requires", ";".join(report["why"]))


if __name__ == "__main__":
    unittest.main()
