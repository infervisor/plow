import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ecapa_export import export_qwen, qwen_config, qwen_tensors


def checkpoint_config():
    return dict(tts_model_type="base", talker_config=dict(hidden_size=6),
                speaker_encoder_config=dict(enc_dim=6, enc_channels=[8, 8, 8, 16],
                                            enc_kernel_sizes=[5, 3, 3, 1],
                                            enc_dilations=[1, 2, 3, 1],
                                            enc_res2net_scale=2, enc_se_channels=3,
                                            enc_attention_channels=4))


def weights():
    state = {}

    def conv(name, cin, cout, kernel):
        state[f"{name}.weight"] = torch.arange(cout * cin * kernel).reshape(
            cout, cin, kernel).float().div(1024).bfloat16()
        state[f"{name}.bias"] = torch.arange(cout).float().div(32).bfloat16()

    conv("blocks.0.conv", 128, 8, 5)
    for b in (1, 2):
        conv(f"blocks.{b}.tdnn1.conv", 8, 8, 1)
        conv(f"blocks.{b}.res2net_block.blocks.0.conv", 4, 4, 3)
        conv(f"blocks.{b}.tdnn2.conv", 8, 8, 1)
        conv(f"blocks.{b}.se_block.conv1", 8, 3, 1)
        conv(f"blocks.{b}.se_block.conv2", 3, 8, 1)
    conv("mfa.conv", 16, 16, 1)
    conv("asp.tdnn.conv", 48, 4, 1)
    conv("asp.conv", 4, 16, 1)
    conv("fc", 32, 6, 1)
    return state


class QwenSpeakerExportTests(unittest.TestCase):
    def test_default_geometry_and_numerical_contract(self):
        source = dict(tts_model_type="base", talker_config=dict(hidden_size=2048),
                      speaker_encoder_config=dict(enc_dim=2048, sample_rate=24000))
        cfg = qwen_config(source)
        self.assertEqual(cfg["channels"], [512, 512, 512, 512, 1536])
        self.assertEqual(cfg["compute_dtype"], "bfloat16")
        self.assertEqual(cfg["conv_bias_round"], "before_add")
        self.assertEqual(cfg["frontend"]["compute_dtype"], "float32")
        self.assertFalse(cfg["batch_norm"])
        self.assertFalse(cfg["l2_normalize"])
        self.assertFalse(cfg["frontend"]["sentence_normalize"])
        self.assertEqual(cfg["frontend"]["pad"], 384)
        self.assertEqual(cfg["statistics_eps"], 1e-12)

    def test_convolution_layout_keeps_bias_and_all_attention_columns(self):
        state = weights()
        out = qwen_tensors(state, qwen_config(checkpoint_config()))
        torch.testing.assert_close(out["blocks.0.conv.weight"].permute(0, 2, 1),
                                   state["blocks.0.conv.weight"].float(), rtol=0, atol=0)
        torch.testing.assert_close(out["blocks.1.res2net.0.conv.weight"].permute(0, 2, 1),
                                   state["blocks.1.res2net_block.blocks.0.conv.weight"].float(),
                                   rtol=0, atol=0)
        torch.testing.assert_close(out["asp.tdnn.conv.weight"].permute(0, 2, 1),
                                   state["asp.tdnn.conv.weight"].float(), rtol=0, atol=0)
        torch.testing.assert_close(out["fc.bias"], state["fc.bias"].float(), rtol=0, atol=0)
        self.assertEqual(len(out), len(state))
        self.assertTrue(all(t.dtype == torch.float32 and t.is_contiguous() for t in out.values()))

    def test_float_source_rounds_to_declared_compute_precision(self):
        state = weights()
        state["fc.bias"] = torch.tensor([1.001, -1.001, 0.1, 0.2, 0.3, 0.4])
        out = qwen_tensors(state, qwen_config(checkpoint_config()))
        torch.testing.assert_close(out["fc.bias"], state["fc.bias"].bfloat16().float(),
                                   rtol=0, atol=0)
        self.assertFalse(torch.equal(out["fc.bias"], state["fc.bias"]))

    def test_unsupported_geometries_fail(self):
        mutations = [dict(enc_dim=7), dict(enc_channels=[8, 8, 8, 24]),
                     dict(enc_res2net_scale=3), dict(enc_kernel_sizes=[4, 3, 3, 1]),
                     dict(enc_dilations=[1, 2]), dict(mel_dim=80), dict(sample_rate=16000)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                config = checkpoint_config()
                config["speaker_encoder_config"].update(mutation)
                qwen_config(config)
        config = checkpoint_config()
        config["tts_model_type"] = "custom_voice"
        with self.assertRaises(ValueError):
            qwen_config(config)

    def test_missing_extra_or_malformed_weights_fail(self):
        original = weights()
        variants = []
        missing = dict(original)
        del missing["fc.bias"]
        variants.append(missing)
        variants.append(dict(original, unexpected=torch.zeros(1)))
        variants.append(dict(original, **{"fc.weight": torch.zeros(6, 31, 1)}))
        variants.append(dict(original, **{"fc.bias": torch.full((6,), float("nan"))}))
        variants.append(dict(original, **{"fc.bias": torch.full((6,), torch.finfo(torch.float32).max)}))
        for state in variants:
            with self.subTest(keys=len(state)), self.assertRaises((ValueError, KeyError)):
                qwen_tensors(state, qwen_config(checkpoint_config()))

    def test_snapshot_export_selects_speaker_and_exact_frontend(self):
        import librosa

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "source", Path(tmp) / "export"
            src.mkdir()
            (src / "config.json").write_text(json.dumps(checkpoint_config()))
            state = {f"speaker_encoder.{k}": v for k, v in weights().items()}
            state["talker.other.weight"] = torch.zeros(1)
            save_file(state, src / "model.safetensors")
            export_qwen(src, dst)
            cfg = json.loads((dst / "config.json").read_text())
            tensors = load_file(dst / "model.safetensors")
            torch.testing.assert_close(tensors["frontend.window"], torch.hann_window(1024),
                                       rtol=0, atol=0)
            expected = torch.from_numpy(librosa.filters.mel(
                sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000))
            torch.testing.assert_close(tensors["frontend.mel"], expected, rtol=0, atol=0)
            self.assertEqual(cfg, qwen_config(checkpoint_config()))
            self.assertNotIn(str(src), (dst / "config.json").read_text())
            self.assertEqual(len(tensors), len(weights()) + 2)


if __name__ == "__main__":
    unittest.main()
