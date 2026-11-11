import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import campaign

sys.path.insert(0, str(campaign.REPO / "scripts"))
from test_packet_sections import write_packet
from check_recipe import to_toml
sys.path.insert(0, str(campaign.REPO / "scripts/tts"))
import recipe_common as rc
from tag_gate import make_cases


class SpeechCampaignTests(unittest.TestCase):
    def fixture(self, root):
        assets = root / "assets"
        assets.mkdir()
        policy = dict(version=1, profile="test", map=rc.default_speech_tag_map())
        write_packet(assets / "model.pkt", [dict(name="speech", driver="tts.guided_lm.v1", parameters={"text.tags_version": 1},
                                               strings={"text.tags": json.dumps(policy["map"])})])
        runtime = root / "plowrt"
        runtime.write_text("runtime")
        recipe = root / "recipe.toml"
        recipe.write_text(to_toml(dict(speech_tags=policy, bench={"kind": "speech"}, gates={"tts_cer": {"arms": [], "cer_median_max": .05}})))
        return argparse.Namespace(recipe=str(recipe), assets=str(assets), out=str(root / "gate"), only="tts_cer", plowrt=str(runtime),
                                  env=None, label=None, timeout=None, dry_run=True, score_only=False, tag_sweep=False)

    def run_quiet(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.cmd_gate(args)

    def test_auto_gate_and_queued_shell_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.run_quiet(a)
            run = (Path(a.out) / "run.sh").read_text()
            self.assertIn("plowbench.sh", run)
            self.assertIn("pb_free_port", run)
            self.assertIn("pb_serve_wait", run)
            self.assertIn("--tts-tag-config", run)
            self.assertIn("--tag-capture", run)
            self.assertLess(run.index("--tts-tag-config"), run.index("\npb_serve_stop"))
            self.assertGreater(run.index("--tag-capture"), run.index("\npb_serve_stop"))
            cfg = json.loads((Path(a.out) / "tts_tags/config.json").read_text())
            self.assertEqual(cfg["settings"]["cer_max"], .05)
            self.assertEqual(cfg["runtime_sha256"], campaign.sha(root / "plowrt"))
            self.assertEqual(len(make_cases(cfg)), 66)
            self.assertFalse(campaign.gate_score("tts_tags", {"config": cfg}, Path(a.out) / "tts_tags")["pass"])

    def test_legacy_gate_preserves_cer_and_skips_tag_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            recipe = root / "recipes/infervisor/maya2-native/sm90a-h100-tp1.toml"
            recipe.parent.mkdir(parents=True)
            r = rc.load_toml(Path(a.recipe))
            del r["speech_tags"]
            r["gates"]["tts_tags"] = {}
            a.recipe = str(recipe)
            for changed in (False, True):
                if changed:
                    r["plow_git"] = "a" * 40
                    r["gates"]["tts_cer"]["cer_median_max"] = .07
                recipe.write_text(to_toml(r))
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.run_quiet(a)
                self.assertEqual(stderr.getvalue().count("speech_tags: legacy recipe without tag policy (grandfathered 2026-11-11)"), 1)
                run = (Path(a.out) / "run.sh").read_text()
                self.assertIn("asr_check.py", run)
                self.assertNotIn("--tts-tag-config", run)
                self.assertNotIn("--tag-capture", run)
                self.assertFalse((Path(a.out) / "tts_tags").exists())
            a.recipe = str(recipe.with_name("new.toml"))
            Path(a.recipe).write_text(to_toml(r))
            with self.assertRaisesRegex(rc.Fail, "missing policy"):
                self.run_quiet(a)
            a.recipe = str(recipe)
            r["speech_tags"] = dict(version=1, profile="test", map=rc.default_speech_tag_map())
            recipe.write_text(to_toml(r))
            write_packet(Path(a.assets) / "model.pkt", [dict(name="speech", driver="tts.codec_lm.v2")])
            with self.assertRaisesRegex(rc.Fail, "emitted text.tags"):
                self.run_quiet(a)

    def test_suffixed_tag_gate_keeps_existing_campaign_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            r = rc.load_toml(Path(a.recipe))
            r["gates"]["tts_tags_hi"] = {"languages": ["hi"]}
            Path(a.recipe).write_text(to_toml(r))
            a.only = "tts_tags_hi"
            self.run_quiet(a)
            cfg = json.loads((Path(a.out) / "tts_tags_hi/config.json").read_text())
            self.assertEqual(cfg["settings"]["languages"], ["hi"])
            self.assertEqual(len(make_cases(cfg)), 66)

    def test_packet_mismatch_and_missing_declaration_fail_before_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            Path(a.recipe).write_text('[bench]\nkind = "speech"\n')
            with self.assertRaises(rc.Fail):
                self.run_quiet(a)
            self.assertFalse(Path(a.out).exists())

    def test_fresh_capture_required_and_score_only_rejects_changed_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            self.run_quiet(a)
            (Path(a.out) / "tts_tags/capture").mkdir()
            with self.assertRaises(SystemExit):
                self.run_quiet(a)
            a.dry_run, a.score_only = False, True
            Path(a.recipe).write_text(Path(a.recipe).read_text() + "\n# changed after capture\n")
            with self.assertRaises(SystemExit):
                self.run_quiet(a)

    def test_native_sweep_is_same_gate_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            r = rc.load_toml(Path(a.recipe))
            r["speech_tags"]["map"]["tags"]["laugh"] = {"text": "<laugh>"}
            r["gates"]["tts_tags"] = {"seeds": [1, 2, 3, 4, 5], "native_metric": "cer"}
            Path(a.recipe).write_text(to_toml(r))
            write_packet(Path(a.assets) / "model.pkt", [dict(name="speech", driver="tts.codec_lm.v2", parameters={"text.tags_version": 1},
                                                           strings={"text.tags": json.dumps(r["speech_tags"]["map"])})])
            a.only, a.tag_sweep = "tts_tags", True
            self.run_quiet(a)
            cfg = json.loads((Path(a.out) / "tts_tags/config.json").read_text())
            self.assertEqual(cfg["mode"], "native-sweep")
            self.assertEqual(len(make_cases(cfg)), 25)

    def test_waiver_is_reported_and_cannot_be_a_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            Path(a.recipe).write_text(to_toml(dict(speech_tags=dict(version=1, waiver="CER 0.2 vs 0.01 at seed 7", evidence=["ab.json"]), bench={"kind": "speech"})))
            a.only = "tts_tags"
            self.run_quiet(a)
            a.dry_run, a.score_only = False, True
            self.run_quiet(a)
            report = json.loads((Path(a.out) / "gates.json").read_text())
            self.assertTrue(report["gates"]["tts_tags"]["waived"])
            a.score_only, a.tag_sweep = False, True
            with self.assertRaises(SystemExit):
                self.run_quiet(a)


if __name__ == "__main__":
    unittest.main()
