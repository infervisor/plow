import copy
import unittest

from qwen3_tts_prep import prompt_contract


def config(kind="base"):
    return dict(
        tts_model_type=kind, tts_pad_token_id=151671, tts_bos_token_id=151672,
        tts_eos_token_id=151673, speaker_encoder_config=dict(enc_dim=2048, sample_rate=24000),
        talker_config=dict(
            hidden_size=2048, num_code_groups=16, codec_think_id=2154,
            codec_think_bos_id=2156, codec_think_eos_id=2157, codec_nothink_id=2155,
            codec_pad_id=2148, codec_bos_id=2149, codec_language_id=dict(english=2050),
            spk_id={} if kind == "base" else dict(ryan=3061),
            spk_is_dialect={} if kind == "base" else dict(ryan=False)))


class PromptContractTests(unittest.TestCase):
    def test_base_declares_external_conditioning_without_preset_or_instructions(self):
        prompt, conditioning = prompt_contract(config())
        self.assertEqual(prompt["voices"], {})
        self.assertIsNone(prompt["instructions_template"])
        self.assertEqual(conditioning, dict(version=1, speaker_packet="speaker_encoder.pkt",
                                           reference_packet="ref_encoder.pkt"))
        self.assertEqual(prompt["frame_codes"], 16)
        self.assertEqual(prompt["text_pad"], 151671)
        self.assertEqual(prompt["codec_language"], [2154, 2156, 2157])

    def test_absent_base_speaker_maps_are_supported(self):
        cfg = config()
        expected = prompt_contract(cfg)
        for value in (None, {}):
            cfg["talker_config"].update(spk_id=value, spk_is_dialect=value)
            self.assertEqual(prompt_contract(cfg), expected)
        del cfg["talker_config"]["spk_id"]
        del cfg["talker_config"]["spk_is_dialect"]
        self.assertEqual(prompt_contract(cfg), expected)

    def test_custom_voice_keeps_preset_dialect_and_instructions(self):
        cfg = config("custom_voice")
        cfg["talker_config"]["spk_is_dialect"]["ryan"] = "english"
        prompt, conditioning = prompt_contract(cfg)
        self.assertIsNone(conditioning)
        self.assertEqual(prompt["voices"], dict(ryan=dict(code=3061, dialect="english")))
        self.assertEqual(prompt["instructions_template"],
                         "<|im_start|>user\n{instructions}<|im_end|>\n")

    def test_unsupported_and_ambiguous_modes_are_rejected(self):
        bad = [config("voice_design"), config("unknown"), config("custom_voice")]
        bad[-1]["talker_config"]["spk_id"] = {}
        for cfg in bad:
            with self.subTest(kind=cfg["tts_model_type"]), self.assertRaises(ValueError):
                prompt_contract(cfg)
        cfg = config()
        cfg["talker_config"].update(spk_id=dict(ryan=3061), spk_is_dialect=dict(ryan=False))
        with self.assertRaisesRegex(ValueError, "preset voices"):
            prompt_contract(cfg)

    def test_invalid_speaker_geometry_and_dialect_maps_are_rejected(self):
        cfg = config()
        cfg["speaker_encoder_config"]["enc_dim"] = 1024
        with self.assertRaises(ValueError):
            prompt_contract(cfg)
        cfg = config("custom_voice")
        cfg["talker_config"]["spk_is_dialect"] = {}
        with self.assertRaisesRegex(ValueError, "dialects"):
            prompt_contract(cfg)

    def test_prompt_preparation_does_not_mutate_source_config(self):
        for kind in ("base", "custom_voice"):
            cfg = config(kind)
            before = copy.deepcopy(cfg)
            prompt_contract(cfg)
            self.assertEqual(cfg, before)


if __name__ == "__main__":
    unittest.main()
