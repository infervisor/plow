import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from asr_check import cer_report, main, norm_seed_en, seed_en_asr, seed_wer_report


class CerReportTests(unittest.TestCase):
    def test_zero_median_does_not_hide_a_single_severe_failure(self):
        rows = [dict(cer=0.0, hyp="correct") for _ in range(31)]
        rows.append(dict(cer=41.3125, hyp="ha " * 400))
        report = cer_report(rows, "pinned-model", 0.15)
        self.assertTrue(report["median_gate"]["passed"])
        self.assertEqual(report["median_cer"], 0.0)
        self.assertEqual(report["p95_cer"], 0.0)
        self.assertEqual(report["max_cer"], 41.3125)
        self.assertEqual(report["over_limit_count"], 1)
        self.assertEqual(report["mean_cer"], 41.3125 / 32)
        self.assertEqual(json.loads(json.dumps(report))["rows"][-1]["hyp"], rows[-1]["hyp"])

    def test_percentile_and_limit_boundary_are_explicit(self):
        rows = [dict(cer=c) for c in [0.0, 0.1, 0.15, 0.4]]
        report = cer_report(rows, "model", 0.15)
        self.assertEqual(report["p95_cer"], 0.4)
        self.assertEqual(report["over_limit_count"], 1)
        self.assertEqual(report["median_gate"]["limit"], 0.15)


class SeedWerTests(unittest.TestCase):
    def test_seed_asr_rejects_turbo_before_loading_weights(self):
        torch, transformers = mock.MagicMock(), mock.MagicMock()
        with mock.patch.dict(sys.modules, torch=torch, transformers=transformers):
            with self.assertRaisesRegex(ValueError, "requires Whisper large-v3"):
                seed_en_asr("openai/whisper-large-v3-turbo", "cpu")
        transformers.WhisperProcessor.from_pretrained.assert_not_called()
        transformers.WhisperForConditionalGeneration.from_pretrained.assert_not_called()

    def test_seed_asr_keeps_full_model_protocol_and_rejects_local_turbo(self):
        for decoder_layers in (32, 4):
            with self.subTest(decoder_layers=decoder_layers):
                torch, transformers = mock.MagicMock(), mock.MagicMock()
                torch.__version__, transformers.__version__ = "2", "4"
                model = transformers.WhisperForConditionalGeneration.from_pretrained.return_value.to.return_value
                model.config.encoder_layers = 32
                model.config.decoder_layers = decoder_layers
                model.config.num_mel_bins = 128
                model.config.d_model = 1280
                model.parameters.side_effect = lambda: iter([mock.Mock(dtype="float32", device="cpu")])
                torch.inference_mode.return_value = lambda f: f
                with mock.patch.dict(sys.modules, torch=torch, transformers=transformers):
                    if decoder_layers == 4:
                        with self.assertRaisesRegex(ValueError, "requires Whisper large-v3"):
                            seed_en_asr("local-whisper", "cpu")
                        continue
                    transcribe, details = seed_en_asr("openai/whisper-large-v3", "cpu")
                self.assertTrue(callable(transcribe))
                self.assertEqual(details["dtype"], "float32")
                self.assertEqual(details["resampling"], "scipy.signal.resample")
                transformers.WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
                    "openai/whisper-large-v3", torch_dtype=torch.float32)
                transformers.WhisperProcessor.from_pretrained.return_value.get_decoder_prompt_ids.assert_called_once_with(
                    language="english", task="transcribe")

    def test_official_punctuation_removes_without_word_splitting(self):
        self.assertEqual(norm_seed_en("I'M “ready”—now!"), "i'm readynow")
        self.assertEqual(norm_seed_en("doesn’t"), "doesnt")
        self.assertNotEqual(norm_seed_en("doesn’t"), norm_seed_en("doesn't"))
        self.assertEqual(norm_seed_en("ＦＩＦＴＹ fifty 50"), "ｆｉｆｔｙ fifty 50")

    def test_sentence_mean_is_uncapped_and_not_word_weighted(self):
        rows = [dict(text="one two three four five six seven eight nine ten", hyp="one two three four five six seven eight nine ten", language="en"),
                dict(text="one", hyp="two three four", language="en")]
        report = seed_wer_report(rows, "model", 1.0)
        self.assertEqual(report["mean_wer"], 1.5)
        self.assertEqual(report["max_wer"], 3.0)
        self.assertFalse(report["mean_gate"]["passed"])
        self.assertEqual(report["over_limit_count"], 1)
        self.assertEqual(report["rows"][1]["insertions"], 2)
        self.assertEqual(report["rows"][0]["reference_words"], 10)
        self.assertIsNone(seed_wer_report(rows, "model")["mean_gate"])

    def test_no_rows_are_silently_dropped(self):
        for rows in ([], [dict(text="!", hyp="audio", language="en")],
                     [dict(text="text", hyp="audio", language="zh")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                seed_wer_report(rows, "model")

    def test_transcript_cli_does_not_load_asr_or_claim_asr_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "source.json", Path(directory) / "score.json"
            source.write_text(json.dumps(dict(model="saved", rows=[
                dict(text="That's fine.", hyp="that's fine", language="en", wav="saved.wav")
            ])))
            args = ["asr_check.py", "--metric", "seed-tts-wer", "--transcripts", str(source),
                    "--out", str(out), "--max-wer", "0"]
            with mock.patch.object(sys, "argv", args), mock.patch("asr_check.seed_en_asr") as asr:
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 0)
                asr.assert_not_called()
            report = json.loads(out.read_text())
            self.assertEqual(report["mean_wer"], 0)
            self.assertEqual(report["rows"][0]["wav"], "saved.wav")
            self.assertIn("ASR protocol not verified", report["scope"])
            self.assertEqual(len(report["transcripts_sha256"]), 64)

    def test_invalid_gate_stops_before_asr_load(self):
        for limit in ("nan", "inf", "-1"):
            args = ["asr_check.py", "--metric", "seed-tts-wer", "--max-wer", limit]
            with self.subTest(limit=limit), mock.patch.object(sys, "argv", args):
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 2)

    def test_invalid_language_stops_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            texts = Path(directory) / "texts.json"
            texts.write_text(json.dumps({"input.wav": dict(text="text", language="zh")}))
            args = ["asr_check.py", "input.wav", "--metric", "seed-tts-wer", "--texts", str(texts)]
            with mock.patch.object(sys, "argv", args), mock.patch("asr_check.seed_en_asr") as asr:
                with self.assertRaises(ValueError):
                    main()
                asr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
