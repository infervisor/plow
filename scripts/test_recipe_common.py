import copy
import tempfile
import unittest
from pathlib import Path

import recipe_common as rc


class SpeechPolicyTests(unittest.TestCase):
    def policy(self):
        return dict(speech_tags=dict(version=1, profile="strip-silence", map=rc.default_speech_tag_map()))

    def test_complete_map(self):
        self.assertEqual(len(rc.speech_tag_policy(self.policy())["map"]["tags"]), 26)

    def test_missing_policy_for_speech_only(self):
        self.assertIsNone(rc.speech_tag_policy({"bench": {"kind": "llm"}}))
        for r in ({"bench": {"kind": "speech"}}, {"gates": {"tts_cer": {}}}):
            with self.assertRaisesRegex(rc.Fail, "missing policy"):
                rc.speech_tag_policy(r)

    def test_waiver(self):
        r = dict(speech_tags=dict(version=1, waiver="CER 0.23 vs 0.01 at seed 7; frontend correction required", evidence=["tag-ab.json sha256:abc"]))
        rc.speech_tag_policy(r)
        for change in ({"evidence": []}, {"waiver": "not measured"}, {"map": rc.default_speech_tag_map()}):
            with self.assertRaises(rc.Fail):
                rc.speech_tag_policy(dict(speech_tags={**r["speech_tags"], **change}))

    def test_invalid_maps(self):
        for mutate in (
            lambda m: m["tags"].pop("laugh"),
            lambda m: m["tags"]["laugh"].update(alias="sigh"),
            lambda m: m["tags"].update(laugh={"alias": "sigh"}, sigh={"alias": "laugh"}),
            lambda m: m["syntax"].update(markers="neutralize"),
            lambda m: m["tags"]["pause"]["silence_ms"].update(default=600),
            lambda m: m.update(literal_rewrite=[["[", "[["]]),
            lambda m: m["tags"].update(sigh={"drop": False}),
            lambda m: m["tags"]["sigh"].update(carry=True),
        ):
            r = self.policy()
            mutate(r["speech_tags"]["map"])
            with self.assertRaises(rc.Fail):
                rc.speech_tag_policy(r)

    def test_validated_native_policy_needs_a_sweep_reference(self):
        r = self.policy()
        r["status"] = "validated"
        r["speech_tags"]["map"]["tags"]["laugh"] = dict(text="<laugh>")
        with self.assertRaisesRegex(rc.Fail, "native_sweep"):
            rc.speech_tag_policy(r)
        r["speech_tags"]["native_sweep"] = "qualification/native.json"
        rc.speech_tag_policy(r)

    def test_native_instruction_alias_and_language_defaults(self):
        m = self.policy()["speech_tags"]["map"]
        m["tags"].update(laugh=dict(text="(laugh)", langs={"en": "(laugh)"}),
                         giggle=dict(alias="laugh"), calm=dict(instruction="Speak calmly.", langs=["en"]))
        self.assertEqual(rc.speech_tag_map(m)["tags"]["giggle"]["carry"], False)


if __name__ == "__main__":
    unittest.main()
