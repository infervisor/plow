import copy
import json
import tempfile
import unittest
from pathlib import Path

import recipe_common as rc
from test_packet_sections import write_packet


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

    def test_tag_policy_reads_both_pipeline_envelopes(self):
        recipe = self.policy()
        pipeline = dict(name="speech", driver="tts.codec_lm.v2",
                        parameters={"text.tags_version": 1},
                        strings={"text.tags": json.dumps(recipe["speech_tags"]["map"])})
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / "model.pkt"
            for version in (1, 2):
                envelope = {"handoff_plans": {"tick": {}}} if version == 2 else {}
                with self.subTest(version=version):
                    write_packet(packet, [pipeline], version=version, **envelope)
                    self.assertEqual(rc.speech_pipelines(packet), [pipeline])
                    self.assertEqual(rc.check_packet_speech_tags(packet, recipe)[0][0], "ok")
                    for mutation in ("map", "tag-version"):
                        changed = copy.deepcopy(pipeline)
                        if mutation == "map":
                            changed["strings"]["text.tags"] = json.dumps({
                                **recipe["speech_tags"]["map"], "unknown": "reject"})
                        else:
                            changed["parameters"]["text.tags_version"] = 2
                        write_packet(packet, [changed], version=version, **envelope)
                        with self.assertRaises(rc.Fail):
                            rc.check_packet_speech_tags(packet, recipe)

    def test_tag_reader_refuses_unknown_versions_and_malformed_v2_containers(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / "model.pkt"
            for version in (0, 3, "2", True, None):
                with self.subTest(version=version):
                    write_packet(packet, [], version=version)
                    with self.assertRaises(rc.Fail):
                        rc.speech_pipelines(packet)
            for plans in (None, [], {}, {"tick": []}):
                with self.subTest(plans=plans):
                    write_packet(packet, [], version=2, handoff_plans=plans)
                    with self.assertRaises(rc.Fail):
                        rc.speech_pipelines(packet)


if __name__ == "__main__":
    unittest.main()
