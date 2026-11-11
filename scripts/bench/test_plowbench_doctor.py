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
            recipe["speech_tags"]["map"]["literal_rewrite"].append(["[", "("])
            with self.assertRaisesRegex(rc.Fail, "differs"):
                rc.check_packet_speech_tags(path, recipe)
            waiver = dict(speech_tags=dict(version=1, waiver="CER 0.2 vs 0.01 at seed 7", evidence=["ab.json"]))
            self.assertEqual(rc.check_packet_speech_tags(path, waiver)[0][0], "WARN")
            waiver["speech_tags"]["evidence"] = []
            with self.assertRaises(rc.Fail):
                rc.check_packet_speech_tags(path, waiver)

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
