import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recipe_common as rc
from test_packet_sections import write_packet


class DoctorSpeechTests(unittest.TestCase):
    def test_speech_map_fail_closed_and_waiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkt"
            recipe = dict(speech_tags=dict(version=1, profile="test", map=rc.default_speech_tag_map()))
            pipeline = dict(name="speech", driver="tts.guided_lm.v1", strings={}, parameters={})
            write_packet(path, [pipeline])
            for r in (None, {}, recipe):
                with self.assertRaises(rc.Fail):
                    rc.check_packet_speech_tags(path, r)
            pipeline["strings"]["text.tags"] = json.dumps(recipe["speech_tags"]["map"])
            pipeline["parameters"]["text.tags_version"] = 1
            write_packet(path, [pipeline])
            self.assertEqual(rc.check_packet_speech_tags(path, recipe)[0][0], "ok")
            pipeline["parameters"]["text.tags_version"] = True
            write_packet(path, [pipeline])
            with self.assertRaisesRegex(rc.Fail, "unsupported"):
                rc.check_packet_speech_tags(path, recipe)
            pipeline["parameters"]["text.tags_version"] = 1
            write_packet(path, [pipeline])
            recipe["speech_tags"]["map"]["literal_rewrite"].append(["[", "("])
            with self.assertRaisesRegex(rc.Fail, "differs"):
                rc.check_packet_speech_tags(path, recipe)
            waiver = dict(speech_tags=dict(version=1, waiver="CER 0.2 vs 0.01 at seed 7", evidence=["ab.json"]))
            self.assertEqual(rc.check_packet_speech_tags(path, waiver)[0][0], "WARN")
            waiver["speech_tags"]["evidence"] = []
            with self.assertRaises(rc.Fail):
                rc.check_packet_speech_tags(path, waiver)

    def test_legacy_notice_and_declared_policy_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipes/infervisor/maya2-native/sm90a-h100-tp1.toml"
            recipe_path.parent.mkdir(parents=True)
            recipe_path.write_text('[bench]\nkind = "speech"\n')
            recipe = rc.load_toml(recipe_path)
            packet = root / "model.pkt"
            write_packet(packet, [dict(name="speech", driver="tts.codec_lm.v2")])
            notice = "speech_tags: legacy recipe without tag policy (grandfathered 2026-11-11)"
            self.assertEqual(rc.check_packet_speech_tags(packet, recipe, recipe_path), [("NOTICE", notice)])
            recipe["plow_git"] = "a" * 40
            self.assertEqual(rc.check_packet_speech_tags(packet, recipe, recipe_path), [("NOTICE", notice)])
            with self.assertRaisesRegex(rc.Fail, "missing policy"):
                rc.check_packet_speech_tags(packet, recipe, recipe_path.with_name("new.toml"))
            snapshot = root / "recipe.toml"
            snapshot.write_text(recipe_path.read_text())
            (root / "build-record.json").write_text(json.dumps(dict(recipe=str(root / "ri/infervisor/maya2-native/sm90a-h100-tp1.toml"), recipe_snapshot="recipe.toml")))
            self.assertEqual(rc.check_packet_speech_tags(packet, recipe, snapshot), [("NOTICE", notice)])
            recipe["speech_tags"] = dict(version=1, profile="test", map=rc.default_speech_tag_map())
            with self.assertRaisesRegex(rc.Fail, "emitted text.tags"):
                rc.check_packet_speech_tags(packet, recipe, recipe_path)
            pipeline = dict(name="speech", driver="tts.codec_lm.v2", parameters={"text.tags_version": 1},
                            strings={"text.tags": json.dumps(recipe["speech_tags"]["map"])})
            write_packet(packet, [pipeline])
            self.assertEqual(rc.check_packet_speech_tags(packet, recipe, recipe_path)[0][0], "ok")
        shell = Path(__file__).with_name("plowbench-doctor.sh").read_text()
        self.assertIn("NOTICE) printf", shell)

    def test_non_speech_and_shell_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkt"
            write_packet(path, [dict(name="other", driver="asr.encoder.v1")])
            self.assertEqual(rc.check_packet_speech_tags(path, None)[0][0], "ok")
            with self.assertRaisesRegex(rc.Fail, "no speech pipeline"):
                rc.check_packet_speech_tags(path, {"bench": {"kind": "speech"}})
        shell = Path(__file__).with_name("plowbench-doctor.sh").read_text()
        self.assertIn('--speech-packet "$ASSETS/model.pkt"', shell)
        self.assertIn('*) pb_bad "speech tag check:', shell)


if __name__ == "__main__":
    unittest.main()
