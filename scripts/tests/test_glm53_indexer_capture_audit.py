import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_capture_site"))
from glm53_indexer_capture import audit_capture, audit_batch_boundaries, capture_config, check_batch_pages


class CaptureConfigTests(unittest.TestCase):
    def test_batch_cache_pages_cannot_cross_requests(self):
        tables, lengths, slots = [[1], [2]], [2, 2], [17, 33]
        selected, mapped = [[0, 1], [1, 0]], [[16, 17], [33, 32]]
        check_batch_pages(tables, lengths, slots, selected, mapped, 3)
        with self.assertRaisesRegex(ValueError, "aliased"):
            check_batch_pages([[1], [1]], lengths, [17, 17], selected, [[16, 17], [17, 16]], 3)
        with self.assertRaisesRegex(ValueError, "slot"):
            check_batch_pages(tables, lengths, slots[::-1], selected, mapped, 3)
        with self.assertRaisesRegex(ValueError, "selected"):
            check_batch_pages(tables, lengths, slots, selected, mapped[::-1], 3)
        with self.assertRaisesRegex(ValueError, "pages"):
            check_batch_pages(tables, lengths, slots, selected, mapped, 2)

    def test_actual_batch_boundary_audit_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference").mkdir()
            (root / "tensors").mkdir()
            digest = hashlib.sha256(struct.pack("<I", 42)).hexdigest()
            manifest = dict(vllm_version="0.29.0", invalid_cases=[], request_batch_size=2,
                max_num_seqs=2, enable_prefix_caching=False, requests=[dict(request_id=str(i),
                prompt_token_ids=[42], prompt_sha256_u32le=digest, generated_token_ids=[1, 2])
                for i in range(2)])
            (root / "reference/manifest.json").write_text(json.dumps(manifest))
            context = dict(max_seq_len=2, num_decodes=2, num_decode_tokens=2,
                num_prefills=0, num_prefill_tokens=0)
            chash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for name in ("positions", "input.hidden", "input.residual", "output.hidden", "output.residual",
                         "xn", "x", "attn", "qb", "xn2", "xmid", "mlp"):
                positions = name == "positions"
                shape = [2] if positions else [2, 2048 if name == "qb" else 6144]
                data = struct.pack("<2q", 1, 1) if positions else bytes(2 * shape[0] * shape[1])
                (root / "tensors" / (name + ".bin")).write_bytes(data)
                record = dict(semantic="block." + name, context=context, context_sha256=chash,
                    rank=0, layer=6, prompt_sha256_u32le=digest, source_shape=shape,
                    source_dtype="int64" if positions else "bfloat16", file=name + ".bin",
                    sha256=hashlib.sha256(data).hexdigest())
                (root / "tensors" / (name + ".json")).write_text(json.dumps(record))
            self.assertEqual(audit_batch_boundaries(root)["batch"], 2)
            path = root / "tensors/positions.json"
            record = json.loads(path.read_text())
            record["context"]["num_decode_tokens"] = 1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "live decode batch"):
                audit_batch_boundaries(root)
            record["context"]["num_decode_tokens"] = 2
            path.write_text(json.dumps(record))
            (root / "tensors/positions.bin").write_bytes(bytes(16))
            with self.assertRaisesRegex(ValueError, "tensor hash"):
                audit_batch_boundaries(root)

    def test_decode_batch_filters_both_block_and_attention(self):
        config = capture_config("out", "a" * 64, attention=True, block=True, decode_batch=8)
        for item in config["selectors"] + config["method_selectors"]:
            if item["semantic"].startswith(("attention.", "block.")):
                conditions = [condition for condition in item["when"]
                    if condition["extract"].get("path", [])[-1:] in
                    (["num_actual_tokens"], ["num_decode_tokens"])]
                self.assertEqual(len(conditions), 1)
                self.assertEqual(conditions[0]["equals"], 8)
        with self.assertRaises(ValueError):
            capture_config("out", "a" * 64, decode_batch=0)

    def test_block_prefill_filter_keeps_decode_capture_unchanged(self):
        config = capture_config("out", "a" * 64, layer=0, block=True, block_prefill_tokens=3)
        for item in config["selectors"] + config["method_selectors"]:
            if item["semantic"].startswith("block."):
                fields = {condition["extract"].get("path", [])[-1]: condition["equals"]
                          for condition in item["when"] if "equals" in condition}
                self.assertEqual(fields["num_prefill_tokens"], 3)
                self.assertEqual(fields["num_decode_tokens"], 0)
                if item["semantic"] == "block.input.residual":
                    self.assertEqual(item["on_missing"], "skip")
        with self.assertRaises(ValueError):
            capture_config("out", "a" * 64, block_prefill_tokens=-1)

    def test_dense_prefill_capture_is_opt_in(self):
        base = capture_config("out", "a" * 64, layer=0, block=True, block_prefill_tokens=3)
        dense = capture_config("out", "a" * 64, layer=0, block=True,
                               block_prefill_tokens=3, dense=True)
        names = {item["semantic"] for item in dense["selectors"]}
        self.assertNotIn("block.mlp.activation", {item["semantic"] for item in base["selectors"]})
        self.assertIn("block.mlp.gate_up", names)
        self.assertIn("block.mlp.activation", names)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "reference").mkdir()
        (self.root / "tensors").mkdir()
        (self.root / "reference/manifest.json").write_text(json.dumps({
            "vllm_version": "0.29.0", "invalid_cases": [], "requests": [{
                "prompt_token_ids": list(range(16)), "prompt_sha256_u32le": "p",
                "generated_token_ids": [1, 2],
            }],
        }))
        before = bytes(3 * 2112)
        prefill = before[:2112] + bytes([1]) * 2112 + before[4224:]
        decode = bytearray(prefill)
        for head in range(8):
            decode[head * 256:head * 256 + 16] = bytes([2]) * 16
        decode[2048:2052] = bytes([2]) * 4
        for invocation, left, right, slots in ((1, before, prefill, list(range(16, 32))),
                                              (2, prefill, bytes(decode), [0])):
            self.context = dict(num_prefill_tokens=16 if invocation == 1 else 0,
                                num_decode_tokens=0 if invocation == 1 else 1,
                                num_prefills=int(invocation == 1), num_decodes=int(invocation == 2),
                                max_seq_len=15 + invocation)
            self.record(invocation, "indexer.cache.before", left, [3, 16, 132], "uint8")
            self.record(invocation, "indexer.cache.after", right, [3, 16, 132], "uint8")
            self.record(invocation, "indexer.slot_mapping", struct.pack(f"<{len(slots)}q", *slots),
                        [len(slots)], "int64")
        self.record(2, "indexer.decode.block_table", struct.pack("<2i", 1, 0), [1, 2], "int32")
        self.record(2, "indexer.decode.seq_lens", struct.pack("<i", 17), [1, 1], "int32")

    def record(self, invocation, semantic, data, shape, dtype):
        stem = f"{invocation}-{semantic}"
        record = dict(invocation_index=invocation, semantic=semantic, context=self.context,
                      context_sha256=hashlib.sha256(json.dumps(self.context, sort_keys=True,
                          separators=(",", ":")).encode()).hexdigest(), rank=0, layer=6,
                      prompt_sha256_u32le="p", source_shape=shape, source_dtype=dtype,
                      file=stem + ".bin", sha256=hashlib.sha256(data).hexdigest())
        (self.root / "tensors" / record["file"]).write_bytes(data)
        (self.root / "tensors" / (stem + ".json")).write_text(json.dumps(record))

    def mutate(self, semantic, offset):
        path = self.root / "tensors" / f"2-{semantic}.json"
        record = json.loads(path.read_text())
        binary = path.parent / record["file"]
        data = bytearray(binary.read_bytes())
        data[offset] ^= 1
        binary.write_bytes(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
        path.write_text(json.dumps(record))

    def test_permuted_pages(self):
        result = audit_capture(self.root)
        self.assertEqual([r["changed_blocks"] for r in result["invocations"]], [1, 1])

    def test_wrong_page_table(self):
        self.mutate("indexer.decode.block_table", 0)
        with self.assertRaisesRegex(ValueError, "page table"):
            audit_capture(self.root)

    def test_uninserted_token(self):
        self.mutate("indexer.cache.after", 16)
        with self.assertRaisesRegex(ValueError, "uninserted token"):
            audit_capture(self.root)

    def test_unrelated_block(self):
        self.mutate("indexer.cache.after", 4224)
        with self.assertRaisesRegex(ValueError, "outside inserted blocks"):
            audit_capture(self.root)

    def test_history_discontinuity(self):
        self.mutate("indexer.cache.before", 4224)
        with self.assertRaisesRegex(ValueError, "history discontinuity"):
            audit_capture(self.root)

    def bind_inputs(self):
        for name in ("indexer.hidden", "indexer.input.hidden"):
            self.record(2, name, bytes(6144 * 2), [1, 6144], "bfloat16")
        self.record(2, "indexer.input.qr", bytes(2048 * 2), [1, 2048], "bfloat16")
        self.record(2, "indexer.input.positions", struct.pack("<q", 16), [1], "int64")

    def test_projection_binding(self):
        self.bind_inputs()
        self.assertTrue(audit_capture(self.root)["invocations"][-1]["projection_inputs_bound"])

    def test_projection_position_mismatch(self):
        self.bind_inputs()
        self.mutate("indexer.input.positions", 0)
        with self.assertRaisesRegex(ValueError, "positions mismatch"):
            audit_capture(self.root)

    def test_projection_hidden_mismatch(self):
        self.bind_inputs()
        self.mutate("indexer.input.hidden", 0)
        with self.assertRaisesRegex(ValueError, "hidden mismatch"):
            audit_capture(self.root)

    def bind_block(self):
        self.bind_inputs()
        for name in ("input.hidden", "input.residual", "output.hidden", "output.residual",
                     "xn", "x", "attn", "xn2", "xmid", "mlp"):
            self.record(3, "block." + name, bytes(6144 * 2), [1, 6144], "bfloat16")
        self.record(3, "block.positions", struct.pack("<q", 16), [1], "int64")

    def test_block_stage_binding(self):
        self.bind_block()
        self.assertEqual(audit_capture(self.root)["block_invocations"],
                         [dict(length=17, boundaries=11, indexer_input_bound=True)])

    def test_block_stage_mismatch(self):
        self.bind_block()
        self.record(3, "block.xn", bytes([1]) + bytes(6144 * 2 - 1), [1, 6144], "bfloat16")
        with self.assertRaisesRegex(ValueError, "block stage identity differs"):
            audit_capture(self.root)

    def test_block_query_projection_geometry(self):
        self.bind_block()
        self.record(3, "block.qb", bytes(2048 * 2), [1, 2048], "bfloat16")
        self.assertEqual(audit_capture(self.root)["block_invocations"][0]["boundaries"], 12)
        self.record(3, "block.qb", bytes(6144 * 2), [1, 6144], "bfloat16")
        with self.assertRaisesRegex(ValueError, "wrong block geometry"):
            audit_capture(self.root)

    def test_block_missing_boundary(self):
        self.bind_block()
        (self.root / "tensors/3-block.x.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete block boundaries"):
            audit_capture(self.root)


if __name__ == "__main__":
    unittest.main()
