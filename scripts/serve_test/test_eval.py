import argparse
import base64
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock
import wave


spec = importlib.util.spec_from_file_location("serve_eval", Path(__file__).with_name("eval.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def wav(pcm):
    output = io.BytesIO()
    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return output.getvalue()


class TtsCaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "requests.json"
        self.args = argparse.Namespace(tts_manifest=str(self.manifest), tts_concurrency=4,
            tts_stream=True, tts_capture_only=True, voice=None, limit=0, tts_prompts="cbx")

    def cases(self, count):
        cases = [dict(id=f"case-{i}", text=f"Speak phrase {i}.", voice="voice", seed=i + 1)
                 for i in range(count)]
        self.manifest.write_text(json.dumps(cases))
        return cases

    def test_concurrent_stream_capture_preserves_case_order_seed_and_pcm(self):
        cases = self.cases(8)
        barrier = threading.Barrier(4)
        seen, lock = [], threading.Lock()

        def call(method, endpoint, body):
            self.assertEqual((method, endpoint), ("POST", "/v1/audio/speech"))
            self.assertTrue(body["stream"])
            with lock:
                seen.append(body)
            barrier.wait(timeout=5)
            return 200, wav(struct.pack("<4h", *([body["seed"]] * 4)))

        with mock.patch.object(evaluation, "call", side_effect=call):
            directory, rows = evaluation.capture_tts(self.args, "model", self.root)
        self.assertEqual(len(seen), 8)
        self.assertEqual([r["case_id"] for r in rows], [c["id"] for c in cases])
        self.assertEqual([r["seed"] for r in rows], [c["seed"] for c in cases])
        self.assertFalse(any("error" in r for r in rows))
        for row in rows:
            expected = struct.pack("<4h", *([row["seed"]] * 4))
            with wave.open(str(Path(directory) / row["wav"])) as w:
                self.assertEqual(w.getnframes(), 4)
                self.assertEqual(w.readframes(4), expected)
            self.assertEqual(row["pcm_sha256"], hashlib.sha256(expected).hexdigest())
        texts = json.loads((Path(directory) / "texts.json").read_text())
        self.assertEqual(list(texts.values()), [c["text"] for c in cases])

    def test_request_and_invalid_audio_errors_remain_in_capture_only_result(self):
        self.cases(4)

        def call(method, endpoint, body):
            if body["seed"] == 1:
                return 429, b"busy"
            if body["seed"] == 2:
                return 200, b"invalid WAV"
            if body["seed"] == 3:
                return 200, wav(b"")
            raise TimeoutError("request timed out")

        with mock.patch.object(evaluation, "call", side_effect=call), \
                mock.patch.dict("sys.modules", {"torch": None, "transformers": None}):
            result = evaluation.eval_tts(self.args, ["model"], self.root)["model"]
        self.assertEqual(result["errors"], 4)
        self.assertEqual(result["n"], 4)
        self.assertIsNone(result["cer_median"])
        self.assertTrue(all("error" in r for r in result["rows"]))
        self.assertEqual(json.loads((Path(result["wavs"]) / "texts.json").read_text()), {})

    def test_streaming_unknown_length_wav_is_saved_with_actual_frame_count(self):
        self.cases(1)
        pcm = struct.pack("<3h", -12, 0, 123)
        data = bytearray(wav(pcm))
        data[4:8] = data[40:44] = b"\xff" * 4
        with mock.patch.object(evaluation, "call", return_value=(200, bytes(data))):
            directory, rows = evaluation.capture_tts(self.args, "model", self.root)
        self.assertNotIn("error", rows[0])
        with wave.open(str(Path(directory) / rows[0]["wav"])) as w:
            self.assertEqual(w.getnframes(), 3)
            self.assertEqual(w.readframes(3), pcm)

    def test_limit_and_voice_override_keep_fixed_manifest_seeds(self):
        self.cases(8)
        self.args.limit, self.args.voice = 2, "another-voice"
        rows = evaluation.tts_cases(self.args, "model")
        self.assertEqual([r["seed"] for r in rows], [1, 2])
        self.assertEqual([r["voice"] for r in rows], ["another-voice"] * 2)

    def test_reference_requests_keep_exact_audio_and_optional_transcript_without_payload_in_capture(self):
        cases = self.cases(2)
        audio = wav(struct.pack("<3h", -100, 0, 100))
        (self.root / "reference.wav").write_bytes(audio)
        for i, case in enumerate(cases):
            case.update(ref_audio_path="reference.wav", non_streaming_mode=bool(i))
        cases[1]["ref_text"] = "Reference transcript."
        self.manifest.write_text(json.dumps(cases))
        seen = []

        def call(method, endpoint, body):
            self.assertEqual(base64.b64decode(body["ref_audio"], validate=True), audio)
            seen.append(body)
            return 200, wav(b"\0\0" * 4)

        with mock.patch.object(evaluation, "call", side_effect=call):
            directory, rows = evaluation.capture_tts(self.args, "model", self.root)
        seen.sort(key=lambda b: b["seed"])
        self.assertNotIn("ref_text", seen[0])
        self.assertEqual(seen[1]["ref_text"], cases[1]["ref_text"])
        self.assertEqual([b["non_streaming_mode"] for b in seen], [False, True])
        for row in rows:
            self.assertNotIn("ref_audio", row)
            self.assertEqual(row["ref_audio_sha256"], hashlib.sha256(audio).hexdigest())
            self.assertEqual(row["ref_audio_path"], str(self.root / "reference.wav"))
        saved = json.loads((Path(directory) / "capture.json").read_text())
        self.assertFalse(any("ref_audio" in row for row in saved))

    def test_bad_reference_inputs_fail_before_any_request(self):
        for extra in [dict(ref_audio_path="missing.wav"), dict(ref_audio_path=""),
                      dict(ref_audio_path=42), dict(ref_text=""),
                      dict(non_streaming_mode="false"), dict(reference_language=""),
                      dict(reference_language=42)]:
            cases = self.cases(1)
            cases[0].update(extra)
            self.manifest.write_text(json.dumps(cases))
            with mock.patch.object(evaluation, "call") as call:
                with self.assertRaises((OSError, ValueError)):
                    evaluation.capture_tts(self.args, "model", self.root)
                call.assert_not_called()

    def test_asr_language_is_independent_of_request_language(self):
        cases = self.cases(2)
        cases[0].update(reference_language="hi")
        cases[1].update(language="english", reference_language="en")
        self.manifest.write_text(json.dumps(cases))
        with mock.patch.object(evaluation, "call", return_value=(200, wav(b"\0\0" * 4))) as call:
            directory, rows = evaluation.capture_tts(self.args, "model", self.root)
        bodies = [c.args[2] for c in call.call_args_list]
        by_seed = {b["seed"]: b for b in bodies}
        self.assertNotIn("language", by_seed[1])
        self.assertEqual(by_seed[2]["language"], "english")
        self.assertTrue(all("reference_language" not in b for b in bodies))
        texts = json.loads((Path(directory) / "texts.json").read_text())
        self.assertEqual([texts[f"{i:04d}.wav"]["language"] for i in range(2)], ["hi", "en"])

    def test_tag_controls_clean_reference_and_expected_errors(self):
        self.args.voice = "override"
        cases = self.cases(2)
        cases[0].update(text="[laughs] Speak phrase.", reference="Speak phrase.", tags="auto",
                        forbid=["laughs"], baseline="base", pauses=[], speed=1.14, sample_rate=8000)
        cases[1].update(voice=None, expected_status=400)
        self.manifest.write_text(json.dumps(cases))

        def call(method, endpoint, body):
            if body["seed"] == 2:
                self.assertNotIn("voice", body)
                return 400, b"missing voice"
            self.assertEqual(body["tags"], "auto")
            self.assertEqual(body["sample_rate"], 8000)
            self.assertNotIn("reference", body)
            return 200, wav(b"\0\0" * 4)

        with mock.patch.object(evaluation, "call", side_effect=call):
            directory, rows = evaluation.capture_tts(self.args, "model", self.root)
        self.assertFalse(any("error" in r for r in rows))
        self.assertTrue(rows[1]["expected_error"])
        text = json.loads((Path(directory) / "texts.json").read_text())["0000.wav"]
        self.assertEqual(text["text"], "Speak phrase.")
        self.assertEqual(text["forbid"], ["laughs"])
        self.assertEqual(text["baseline"], "base")
        with mock.patch.object(evaluation, "call", return_value=(200, wav(b"\0\0" * 4))):
            _, rows = evaluation.capture_tts(self.args, "model", self.root)
        self.assertIn("error", rows[1])

    def test_empty_or_malformed_cases_fail_before_requests(self):
        for cases in [[], {}, [dict(text="")], [None]]:
            self.manifest.write_text(json.dumps(cases))
            with self.assertRaises(ValueError):
                evaluation.tts_cases(self.args, "model")
        with self.assertRaises(ValueError):
            evaluation.eval_tts(self.args, [], self.root)


class CapabilityTests(unittest.TestCase):
    def test_all_skips_unadvertised_capabilities_and_writes_reports(self):
        for asr, llm in [(["asr"], []), ([], ["llm"]), (["asr"], ["llm"])]:
            with self.subTest(asr=asr, llm=llm), tempfile.TemporaryDirectory() as directory, \
                    mock.patch("sys.argv", ["eval.py", "all", "--out", directory]), \
                    mock.patch.object(evaluation, "models_by_kind", return_value=(asr, [], llm, {})), \
                    mock.patch.object(evaluation, "eval_asr", return_value={}) as run_asr, \
                    mock.patch.object(evaluation, "eval_llm", return_value=dict(
                        model="llm", passed=1, total=1, checks=[])) as run_llm, \
                    mock.patch.object(evaluation, "capture_tts") as capture:
                with self.assertRaises(SystemExit) as result:
                    evaluation.main()
                self.assertEqual(result.exception.code, 0)
                self.assertEqual(run_asr.call_count, bool(asr))
                self.assertEqual(run_llm.call_count, bool(llm))
                capture.assert_not_called()
                report = json.loads((Path(directory) / "eval.json").read_text())
                self.assertEqual("asr" in report, bool(asr))
                self.assertEqual("llm" in report, bool(llm))
                self.assertNotIn("tts", report)
                self.assertTrue((Path(directory) / "eval.md").is_file())

    def test_explicit_tts_still_rejects_missing_models(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("sys.argv", ["eval.py", "tts", "--out", directory]), \
                mock.patch.object(evaluation, "models_by_kind", return_value=([], [], [], {})):
            with self.assertRaisesRegex(ValueError, "no requested TTS model"):
                evaluation.main()


if __name__ == "__main__":
    unittest.main()


class TagGateWireTests(unittest.TestCase):
    def test_local_http_capture_headers_lenient_values_and_cpu_rescore(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import sys
        from tag_gate import score
        from test_tag_gate import config
        import asr_check

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(body)
                text = body["input"]
                marker = re.search(r"\[pause (\d+)ms\]", text)
                samples = [6500] * 100 + [9800] * 100
                if marker:
                    zeros = [0] * int(marker[1])
                    if marker.start() == 0:
                        samples = zeros + samples
                    elif marker.end() == len(text):
                        samples += zeros
                    else:
                        samples[100:100] = zeros
                output = io.BytesIO()
                with wave.open(output, "wb") as wav:
                    wav.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
                    wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
                self.send_response(200)
                if marker and body.get("tags") != "off":
                    self.send_header("x-plow-tags-applied", "pause")
                    self.send_header("x-plow-tags-dropped", "laugh" if "[laugh]" in text else "")
                self.end_headers()
                self.wfile.write(output.getvalue())

        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.shutdown)
            root = Path(tmp)
            cfg = config()
            cfg["settings"]["sample_rate"] = 1000
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            args = argparse.Namespace(tts_tag_config=str(path), tts_manifest=None, tts_capture_only=True,
                                      tts_concurrency=1, tts_stream=True, limit=0, voice=None, tts_prompts="cbx")
            with mock.patch.object(evaluation, "URL", f"http://127.0.0.1:{server.server_port}"):
                result = evaluation.eval_tts(args, ["model"], root)["model"]
            server.shutdown()
            thread.join(timeout=5)
            self.assertEqual(result["errors"], 0)
            self.assertEqual(len(seen), 66)
            self.assertTrue(any(body.get("tags") == {"mode": "off"} for body in seen))
            self.assertTrue(any(body.get("tags") is None for body in seen))
            rows = result["rows"]
            asr = dict(rows=[dict(case_id=r["case_id"], text=r["reference"], language=r["reference_language"], hyp=r["reference"], cer=0)
                             for r in rows])
            raw = root / "asr-raw.json"
            raw.write_text(json.dumps(asr))
            report = root / "tags.json"
            with mock.patch.object(sys, "argv", ["asr_check.py", "--tag-config", str(path), "--tag-capture", str(root),
                                                "--tag-report", str(raw), "--out", str(report)]), \
                    mock.patch.dict(sys.modules, torch=None, transformers=None), self.assertRaises(SystemExit) as result:
                asr_check.main()
            self.assertEqual(result.exception.code, 0)
            self.assertTrue(json.loads(report.read_text())["pass"])
