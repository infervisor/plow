import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Qwen3Config, Qwen3Model

from qwen3_tts_ref import decoder_reference, load_prepared_predictor, load_reference_decoder, prepared_predictor_logits


class PreparedPredictorTests(unittest.TestCase):
    def test_full_prefix_matches_incremental_predictor(self):
        torch.manual_seed(71)
        config = Qwen3Config(vocab_size=1, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
                            rope_theta=1000000, max_position_embeddings=16)
        config._attn_implementation = "sdpa"
        original = Qwen3Model(config).float().eval()
        state = {k: v.bfloat16().float() for k, v in original.state_dict().items()}
        original.load_state_dict(state)
        projection = torch.randn(8, 12).bfloat16().float() / 4
        bias = torch.randn(8).bfloat16().float()
        embeddings = torch.randn(4 * 7, 12).bfloat16().float()
        heads = [torch.randn(5, 8).bfloat16().float() for _ in range(3)]
        hidden = torch.randn(12).bfloat16().float()
        codes = [2, 4, 1]
        case = dict(prefix_codes=codes,
                    hidden_bf16=(hidden.numpy().view(np.uint32) >> 16).tolist())
        tensors = {f"depth.{k}": v for k, v in state.items() if k != "embed_tokens.weight"}
        tensors.update({"depth.proj.weight": projection, "depth.proj.bias": bias,
                        "frames": embeddings})
        tensors.update({f"depth.head.{j}.weight": head for j, head in enumerate(heads)})
        meta = dict(table="frames", prefix="depth", codebooks=4, codebook_size=7, classes=5,
                    depth=dict(hidden=8, intermediate=16, layers=2, heads=2, kv_heads=1,
                               head_dim=8, eps=1e-6, rope_theta=1000000, rope_scaling=None,
                               positions=16, qk_norm=True, proj_bias=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(json.dumps(dict(plow_frame_embed=meta)))
            save_file(tensors, path / "model.safetensors")
            model, weights, loaded_meta = load_prepared_predictor(path, 3, "cpu")
            actual = prepared_predictor_logits(model, weights, loaded_meta, case, 3)
            with torch.inference_mode():
                cache = DynamicCache()
                x = torch.stack((hidden, embeddings[codes[0]]))[None]
                for j in range(3):
                    projected = torch.nn.functional.linear(x, projection, bias)
                    result = original(inputs_embeds=projected, past_key_values=cache,
                                      use_cache=True).last_hidden_state[:, -1]
                    expected = torch.nn.functional.linear(result, heads[j])[0]
                    if j < 2:
                        x = embeddings[(j + 1) * 7 + codes[j + 1]][None, None]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            with self.assertRaises(AssertionError):
                prepared_predictor_logits(model, weights, loaded_meta,
                                          dict(case, prefix_codes=codes[:2]), 3)


class DecoderReferenceTests(unittest.TestCase):
    def test_streaming_source_receives_frame_major_codes_and_resets_each_decode(self):
        class Config:
            num_quantizers = 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.decoder_transformer = torch.nn.Module()
                self.decoder_transformer.config = self.config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(dict(
                model_type="mimi", sampling_rate=24, frame_rate=12, codebook_size=4)))
            source = root / "streaming.py"
            source.write_text('''import torch
class StaticMimi:
    def __init__(self, model, max_slots, batch_buckets):
        assert max_slots == 1 and batch_buckets == (1,)
        self.window = 250
    def reset(self, slot, zero_kv):
        assert slot == 0 and zero_kv
        self.position = 0
    def _forward(self, slots, codes):
        assert slots.tolist() == [0] and tuple(codes.shape) == (1, 2, 1)
        value = codes.sum().float() + self.position
        self.position += 1
        return value.expand(1, 1, 2)
''')
            with patch("transformers.MimiModel.from_pretrained", return_value=(Model(), {})):
                decode, meta = load_reference_decoder(root, "cpu", source)
            codes = np.array([[1, 2], [2, 3]], dtype=np.int64)
            for _ in range(2):
                np.testing.assert_array_equal(decode(codes), [3, 3, 6, 6])
            self.assertEqual(meta["reference_kind"], "upstream_static_streaming")
            self.assertEqual(meta["attention_window_rows"], 250)
            self.assertIn(str(source.resolve()), meta["source_files"])

    def test_rejects_incomplete_official_weight_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(dict(model_type="mimi")))
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                with self.subTest(key=key), patch("transformers.MimiModel.from_pretrained",
                                                  return_value=(None, {key: ["weight"]})):
                    with self.assertRaisesRegex(AssertionError, "incomplete reference weights"):
                        load_reference_decoder(root, "cpu")

    def fixture(self, root):
        source = root / "model"
        source.mkdir()
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").write_bytes(b"mock decoder weights")
        codes = np.arange(12, dtype=np.uint32).reshape(6, 2)
        codes.tofile(root / "codes.u32")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
        reference = dict(frame_codes=2, frame_samples=3, sample_rate=24000,
                         source_sha256=hashes, cases=[dict(case="clip", frames=6, codes=str(root / "codes.u32"))])
        (root / "reference.json").write_text(json.dumps(reference))
        expected = np.repeat(codes.sum(1).astype(np.float32), 3)[6:12]
        item = dict(width=8, item=0, source="clip", reference_frames=2, fresh_frames=2,
                    samples=6, output_offset_f32=0)
        capture = {"pass": True, "whole_relative_l2_limit": 1e-5, "checks": [item]}
        (root / "native.json").write_text(json.dumps(capture))
        np.tile(expected, 3).astype("<f4").tofile(root / "native.f32")
        args = SimpleNamespace(model=str(source), texts=str(root / "reference.json"), device="cpu",
                               decoder_result=[str(root / "native.json")], out=str(root / "out"))
        meta = dict(frame_codes=2, frame_samples=3, sample_rate=24000,
                    codebook_sizes=[64, 64], source_files=[])
        calls = []

        def decode(values):
            calls.append(values.copy())
            return np.repeat(values.sum(1).astype(np.float32), 3)
        return args, meta, decode, calls

    def test_uses_frame_major_codes_exact_crop_and_caches_repeated_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, meta, decode, calls = self.fixture(root)
            args.decoder_result *= 2
            with patch("qwen3_tts_ref.load_reference_decoder", return_value=(decode, meta)):
                report = decoder_reference(args)
            self.assertTrue(report["pass_all"])
            self.assertEqual(len(calls), 2)
            np.testing.assert_array_equal(calls[0], np.arange(8).reshape(4, 2))
            self.assertEqual(len(report["captures"]), 2)
            self.assertTrue((root / "out/report.json").is_file())
            self.assertEqual(report["captures"][0]["checks"][0]["metrics"]["primed"]["max_abs"], 0)

    def test_rejects_nonrepeatable_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, meta, decode, calls = self.fixture(Path(tmp))

            def changing(values):
                return decode(values) + len(calls)
            with patch("qwen3_tts_ref.load_reference_decoder", return_value=(changing, meta)):
                with self.assertRaisesRegex(AssertionError, "nonrepeatable"):
                    decoder_reference(args)

    def test_retains_each_pcm_failure_and_internal_gate_failure(self):
        for failure in ("primed", "whole", "stateful", "nonfinite", "internal"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, meta, decode, _ = self.fixture(root)
                pcm = np.fromfile(root / "native.f32", dtype="<f4")
                if failure == "internal":
                    capture = json.loads((root / "native.json").read_text())
                    capture["pass"] = False
                    (root / "native.json").write_text(json.dumps(capture))
                elif failure == "nonfinite":
                    pcm[-1] = np.nan
                else:
                    pcm[{"whole": 0, "stateful": 6, "primed": 12}[failure]] += 1
                pcm.tofile(root / "native.f32")
                with patch("qwen3_tts_ref.load_reference_decoder", return_value=(decode, meta)):
                    report = decoder_reference(args)
                self.assertFalse(report["pass_all"])
                result = report["captures"][0]
                if failure == "internal":
                    self.assertTrue(result["independent_reference_pass"])
                    self.assertFalse(result["internal_gate_pass"])
                else:
                    metrics = result["checks"][0]["metrics"]
                    failed = "primed" if failure == "nonfinite" else failure
                    self.assertFalse(metrics[failed]["passed"])
                    self.assertEqual(sum(not m["passed"] for m in metrics.values()), 1)

    def test_rejects_bad_capture_geometry_and_reference_codes(self):
        for failure in ("offset", "short", "trailing", "codes", "bounds", "samples", "limit", "source"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, meta, decode, _ = self.fixture(root)
                capture = json.loads((root / "native.json").read_text())
                if failure == "offset":
                    capture["checks"][0]["output_offset_f32"] = 1
                elif failure in ("short", "trailing"):
                    values = np.fromfile(root / "native.f32", dtype="<f4")
                    (values[:-1] if failure == "short" else np.append(values, np.float32(0))).tofile(root / "native.f32")
                elif failure == "codes":
                    (root / "codes.u32").write_bytes(b"\0" * 4)
                elif failure == "bounds":
                    values = np.fromfile(root / "codes.u32", dtype="<u4")
                    values[0] = 64
                    values.tofile(root / "codes.u32")
                elif failure == "samples":
                    capture["checks"][0]["samples"] = 5
                elif failure == "limit":
                    capture["whole_relative_l2_limit"] = 1e-4
                else:
                    reference = json.loads(Path(args.texts).read_text())
                    reference["source_sha256"] = {"config.json": "changed"}
                    Path(args.texts).write_text(json.dumps(reference))
                (root / "native.json").write_text(json.dumps(capture))
                with patch("qwen3_tts_ref.load_reference_decoder", return_value=(decode, meta)):
                    with self.assertRaises(AssertionError):
                        decoder_reference(args)


if __name__ == "__main__":
    unittest.main()
