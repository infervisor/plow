import argparse
import json
from pathlib import Path
import tempfile
import unittest

import campaign


class GatePacketTests(unittest.TestCase):
    def fixture(self, root):
        assets = root / "assets"
        assets.mkdir()
        (assets / "model.pkt").write_bytes(b"packet")
        runtime = root / "plowrt"
        runtime.write_bytes(b"runtime")
        recipe = root / "recipe.toml"
        recipe.write_text('[gates.asr_wer]\nmanifest = "{repo}/m.jsonl"\nwer_max = 0.1\n')
        return argparse.Namespace(recipe=str(recipe), assets=str(assets), out=str(root / "gate"), only=None,
                                  plowrt=str(runtime), env=None, label=None, timeout=None, dry_run=False,
                                  score_only=False)

    def capture(self, a):
        a.dry_run = True
        campaign.cmd_gate(a)
        a.dry_run, a.score_only = False, True
        d = Path(a.out) / "asr_wer"
        (d / "served.jsonl").write_text(json.dumps({"wer": 0.01, "errors": 0}) + "\n")

    def test_score_only_uses_captured_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.capture(a)
            campaign.cmd_gate(a)
            rec = json.loads((Path(a.out) / "gates.json").read_text())
            self.assertEqual(rec["packet_sha256"], campaign.sha(Path(a.assets) / "model.pkt"))
            self.assertTrue(rec["pass"])

    def test_score_only_refuses_other_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.capture(a)
            (Path(a.assets) / "model.pkt").write_bytes(b"rebuilt packet")
            with self.assertRaises(SystemExit):
                campaign.cmd_gate(a)
            self.assertFalse((Path(a.out) / "gates.json").exists())

    def test_score_only_refuses_unrecorded_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.capture(a)
            (Path(a.out) / "packet.sha256").unlink()
            with self.assertRaises(SystemExit):
                campaign.cmd_gate(a)

    def test_suffixed_table_is_a_second_gate_of_its_kind(self):
        self.assertEqual(campaign.gate_kind("llm_fp32_ref_long"), "llm_fp32_ref")
        self.assertEqual(campaign.gate_kind("llm_fp32_ref"), "llm_fp32_ref")
        self.assertIsNone(campaign.gate_kind("llm_fp32"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            Path(a.recipe).write_text('[gates.asr_wer]\nmanifest = "m"\n[gates.asr_wer_long]\nmanifest = "l"\n')
            a.dry_run = True
            campaign.cmd_gate(a)
            run = (Path(a.out) / "run.sh").read_text()
            self.assertIn(str(Path(a.out) / "asr_wer_long"), run)
            self.assertIn("--manifest l", run)

    def test_lenient_expand_keeps_unset_env(self):
        out = Path("/x")
        self.assertEqual(campaign.expand("{env:GATE_TEST_UNSET}/{out}", out, lenient=True), "{env:GATE_TEST_UNSET}//x")
        with self.assertRaises(SystemExit):
            campaign.expand("{env:GATE_TEST_UNSET}", out)

    def mm_fixture(self, root, ref_exists):
        a = self.fixture(root)
        ref = root / "ref"
        ref.mkdir()
        if ref_exists:
            (ref / "ref.json").write_text("{}")
        runner = root / "mm_check"
        runner.write_text("")
        Path(a.recipe).write_text(
            f'[gates.mm_parity]\nreference = "{ref}"\nrunner = "{runner}"\nhf_dir = "/ckpt"\ncases = "scripts/mm/cases/audio.json"\n'
            f'audio = ["/a/ls00.wav"]\nrefusal_audio = "/a/ls00.wav"\ntie_margin = 0.25\n')
        return a

    def test_mm_parity_steps_and_reference_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.mm_fixture(root, ref_exists=False)
            a.dry_run = True
            campaign.cmd_gate(a)
            run = (Path(a.out) / "run.sh").read_text()
            hf, serve, gate, refusals, check = (run.index(s) for s in
                                                ("hf_ref.py all", " serve --assets", "gate.py", "refusals.py", "mm_check"))
            self.assertTrue(hf < serve < gate < refusals < check, run)
            self.assertIn("--tie-margin 0.25", run)
            self.assertIn("--audio /a/ls00.wav", run)
            self.assertIn(str(campaign.REPO / "scripts/mm/cases/audio.json"), run)
        with tempfile.TemporaryDirectory() as tmp:
            a = self.mm_fixture(Path(tmp), ref_exists=True)
            a.dry_run = True
            campaign.cmd_gate(a)
            self.assertNotIn("hf_ref.py", (Path(a.out) / "run.sh").read_text())

    def test_mm_parity_score(self):
        g = {"refusal_audio": "/a.wav"}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            case = lambda name, ok: json.dumps({"case": name, "pass": ok, "exact": ok, "prompt_tokens": 9, "hf_prompt_tokens": 9,
                                                "repeat_stable": True})
            (d / "gate.jsonl").write_text("\n".join([case("asr0", True), case("asr1", True), '{"gate": "pass"}']) + "\n")
            (d / "refusals.jsonl").write_text(json.dumps({"case": "opus_format", "pass": True}) + "\nrefusals pass\n")
            enc = {"pass": True, "cosine": 0.9999}
            (d / "mm_check.jsonl").write_text(json.dumps({"item": 0, "preprocess": {"pass": True}, "encode_hf_input": enc}) + "\n")
            s = campaign.gate_score("mm_parity", g, d)
            self.assertTrue(s["pass"], s)
            self.assertEqual((s["cases_pass"], s["refusals_pass"], s["encoder_checks_pass"]), (2, 1, 2))
            (d / "gate.jsonl").write_text(case("asr0", False) + "\n" + json.dumps({"media_collision": ["a", "b"]}) + "\n")
            (d / "mm_check.jsonl").write_text("")
            s = campaign.gate_score("mm_parity", g, d)
            self.assertFalse(s["pass"])
            self.assertEqual(len(s["why"]), 3, s["why"])


class PrivateRuntimeTests(unittest.TestCase):
    def test_sibling_libraries_travel_with_plowrt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = root / "rt"
            rt.mkdir()
            (rt / "plowrt").write_bytes(b"bin")
            (rt / "libcublasLt.so.13").write_bytes(b"lt")
            (rt / "notes.txt").write_text("x")
            out = root / "gate"
            out.mkdir()
            private = campaign.private_runtime(rt / "plowrt", out)
            self.assertEqual(private.read_bytes(), b"bin")
            self.assertEqual((out / "libcublasLt.so.13").read_bytes(), b"lt")
            self.assertFalse((out / "notes.txt").exists())
            self.assertEqual(campaign.private_runtime(private, out), private, "re-running from the copy is a no-op")


if __name__ == "__main__":
    unittest.main()
