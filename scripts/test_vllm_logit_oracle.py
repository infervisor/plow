import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from vllm_logit_oracle import (dense_scores, repeat_metrics, suppression_metadata,
                               generation_rows, required_model_length, prompt_digest,
                               engine_overrides, model_precision_inventory, precision_report)


class RequestBatchTests(unittest.TestCase):
    def test_batches_preserve_case_order_and_tail(self):
        from vllm_logit_oracle import generate_requests
        calls = []
        def generate(prompts, sampling, use_tqdm):
            calls.append(prompts)
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"]) for p in prompts]
        cases = [dict(id=str(i), prompt_token_ids=[i + 1]) for i in range(5)]
        rows = list(generate_requests(SimpleNamespace(generate=generate), cases, None, 2))
        self.assertEqual([len(x) for x in calls], [2, 2, 1])
        self.assertEqual([case for case, _ in rows], cases)

    def test_single_default_and_invalid_batch_results(self):
        from vllm_logit_oracle import generate_requests
        cases = [dict(id="one", prompt_token_ids=[1])]
        def generate(prompt, sampling, use_tqdm):
            self.assertIsInstance(prompt, dict)
            return [SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"])]
        llm = SimpleNamespace(generate=generate)
        self.assertEqual(len(list(generate_requests(llm, cases, None, 1))), 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            list(generate_requests(llm, cases, None, 0))
        llm.generate = lambda *a, **kw: []
        with self.assertRaisesRegex(ValueError, "count"):
            list(generate_requests(llm, cases, None, 2))
        llm.generate = lambda *a, **kw: [SimpleNamespace(prompt_token_ids=[2])]
        with self.assertRaisesRegex(ValueError, "order"):
            list(generate_requests(llm, cases, None, 2))


class PrecisionInventoryTests(unittest.TestCase):
    def test_loaded_tensors_and_quantization_flags_without_readback(self):
        import torch

        model = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16, device="meta")
        model.kv_cache = torch.empty(3, 132, dtype=torch.uint8, device="meta")
        model.is_aiter_triton_fp8_bmm_enabled = True
        model.is_aiter_triton_fp4_bmm_enabled = False
        model.quant_method = SimpleNamespace(activation_quant_key="dynamic-1x128")
        row = model_precision_inventory(model)
        module = row["modules"][""]
        self.assertEqual(module["tensors"]["weight"]["dtype"], "torch.bfloat16")
        self.assertEqual(module["tensors"]["weight"]["shape"], [2, 4])
        self.assertEqual(module["tensors"]["kv_cache"]["dtype"], "torch.uint8")
        self.assertTrue(module["attributes"]["is_aiter_triton_fp8_bmm_enabled"])
        self.assertFalse(module["attributes"]["is_aiter_triton_fp4_bmm_enabled"])
        self.assertEqual(module["attributes"]["quant_method"]["fields"]["activation_quant_key"],
                         "dynamic-1x128")
        report = precision_report([row], 1)
        self.assertFalse(report["precision_qualified"])
        self.assertIn("not a runtime arithmetic trace", report["scope"])

    def test_missing_duplicate_and_empty_rank_reports_fail(self):
        row = dict(rank=0, modules={"layer": {}}, sources={})
        for ranks, expected in (([], 1), ([row], 2), ([row, row], 2),
                                ([dict(row, modules={})], 1)):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                precision_report(ranks, expected)


class EngineOverridesTests(unittest.TestCase):
    def test_defaults_do_not_override_engine_policy(self):
        self.assertEqual(engine_overrides(), {})

    def test_provider_control_preserves_compile_and_other_providers(self):
        overrides = engine_overrides(rms_norm_provider="vllm_c")
        self.assertNotIn("compilation_config", overrides)
        self.assertEqual(overrides["kernel_config"]["ir_op_priority"],
                         {"rms_norm": ["vllm_c"]})
        combined = engine_overrides(True, "vllm_c")
        self.assertEqual(combined["kernel_config"], overrides["kernel_config"])
        self.assertEqual(combined["compilation_config"], {"cudagraph_mode": "NONE"})

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            engine_overrides(rms_norm_provider="all")

    def test_precision_inventory_uses_named_worker_extension(self):
        self.assertEqual(engine_overrides(precision_inventory=True), {
            "worker_extension_cls": "vllm_logit_oracle.PrecisionInventoryWorker"})


class SuppressionTests(unittest.TestCase):
    def row(self, values, ids=None):
        return SimpleNamespace(token_ids=list(range(len(values))) if ids is None else ids, logprobs=values)

    def test_declared_negative_infinity_is_retained(self):
        row = dense_scores(self.row([1.0, -np.inf, 3.0]), 3, suppressed_ids=[1])
        self.assertTrue(np.isneginf(row[1]))
        self.assertEqual(row.shape, (3,))
        metrics = repeat_metrics(row, row, [1])
        self.assertEqual(metrics["full_row_centered_rel_l2"], 0.0)
        self.assertTrue(metrics["same_argmax"])

    def test_other_nonfinite_and_missing_are_rejected(self):
        for values, allowed in [([1.0, -np.inf], []), ([1.0, np.inf], [1]), ([1.0, np.nan], [1]), ([-np.inf, 1.0], [1])]:
            with self.subTest(values=values, allowed=allowed):
                with self.assertRaises(RuntimeError):
                    dense_scores(self.row(values), 2, suppressed_ids=allowed)
        with self.assertRaises(RuntimeError):
            dense_scores(self.row([1.0], [0]), 2, suppressed_ids=[1])

    def test_config_validation_and_finite_default(self):
        for ids in ([True], [-1], [2], [1.0], "1"):
            config = SimpleNamespace(try_get_generation_config=lambda: {"suppress_tokens": ids})
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                suppression_metadata(config, 2)
        config = SimpleNamespace(try_get_generation_config=lambda: {"suppress_tokens": [1, 1]})
        self.assertEqual(suppression_metadata(config, 2)["token_ids"], [1])
        np.testing.assert_array_equal(dense_scores(self.row([1.0, 2.0]), 2), [1.0, 2.0])

    def test_invalid_decode_rows_keep_distinct_diagnostics(self):
        with TemporaryDirectory() as directory:
            for step, value in enumerate([np.nan, np.inf]):
                prefix = Path(directory) / f"request.step{step:04d}"
                with self.assertRaises(RuntimeError):
                    dense_scores(self.row([value]), 1, prefix)
            first = np.fromfile(Path(directory) / "request.step0000.invalid.f32", dtype="<f4")
            second = np.fromfile(Path(directory) / "request.step0001.invalid.f32", dtype="<f4")
            self.assertTrue(np.isnan(first[0]))
            self.assertTrue(np.isposinf(second[0]))


class GenerationRowsTests(unittest.TestCase):
    def test_actual_generated_history_and_stable_row_ids(self):
        rows = generation_rows("base", [2, 10], [31, 44, 55], 3)
        self.assertEqual([r["id"] for r in rows], ["base.step0000", "base.step0001", "base.step0002"])
        self.assertEqual([r["prompt_len"] for r in rows], [2, 3, 4])
        self.assertEqual([r["prompt_sha256_u32le"] for r in rows],
                         [prompt_digest(x) for x in ([2, 10], [2, 10, 31], [2, 10, 31, 44])])
        self.assertEqual([r["sampled_token_id"] for r in rows], [31, 44, 55])
        self.assertEqual([r["execution_phase"] for r in rows],
                         ["prefill_output", "decode_output", "decode_output"])
        self.assertEqual(generation_rows("legacy", [2, 10], [31], 1)[0]["id"], "legacy")

    def test_generation_length_must_match_request(self):
        for count, tokens in [(0, []), (3, [1, 2]), (1, [1, 2])]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                generation_rows("x", [2], tokens, count)

    def test_context_bound_includes_all_output_tokens(self):
        cases = [{"prompt_token_ids": [2] * 8192}]
        self.assertEqual(required_model_length(cases, 3), 8195)
        self.assertEqual(required_model_length(cases, 3, 8192), 8195)
        self.assertEqual(required_model_length(cases, 3, 16384), 16384)
        with self.assertRaises(ValueError):
            required_model_length(cases, 0)
        with self.assertRaises(ValueError):
            required_model_length([{"prompt_token_ids": []}], 3)


if __name__ == "__main__":
    unittest.main()
