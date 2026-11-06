import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Qwen3Config, Qwen3Model

from qwen3_tts_ref import load_prepared_predictor, prepared_predictor_logits


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


if __name__ == "__main__":
    unittest.main()
