import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_capture_site"))
from glm53_indexer_capture import audit_capture


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


if __name__ == "__main__":
    unittest.main()
