import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import block_roofline
import nv_block_ab

DISASM = """===== program T=128  4 insts
#0     RmsNorm   b=128  out<-act.hn x<-act.x gamma<-l.g | rows=128 feat=2816 eps=0.000001
#1     Gemm      b=132  C<-act.q A<-act.hn B<-model.layers.0.q.weight | M=128 N=4096 K=2816
#2     FlashPrefill b=132  Q<-act.q K<-kv.0.k V<-kv.0.v O_final<-act.at | n_q=128 n_kv=128 n_head=16 n_kv_head=8 q_pos0=0 window=0 hd=256 nsplit=1
#3     Gemm      b=132  C<-act.o A<-act.at B<-model.layers.0.o.weight | M=128 N=2816 K=4096
===== program T=1  3 insts
#0     Gemv      b=132  C<-act.q A<-act.x B<-model.layers.0.q.weight | M=1 N=4096 K=2816
#1     FlashDecode b=132  Q<-act.q K<-kv.0.k V<-kv.0.v | n_batch=1 n_head=16 n_kv_head=8 hd=256 window=0
#2     Gemv      b=132  C<-act.o A<-act.at B<-model.layers.0.o.weight | M=1 N=2816 K=4096
"""
CEIL = (3000.0, 900.0, 1800.0)


class FloorTests(unittest.TestCase):
    def setUp(self):
        self.progs = block_roofline.op_roof.parse(DISASM)

    def test_decode_floor_is_the_sum_of_priced_ops(self):
        floor, priced = block_roofline.decode_floor(self.progs, 1, 4096, CEIL, 2, 2, None)
        self.assertAlmostEqual(floor, sum(e[3] for e in priced.values()))
        with self.assertRaisesRegex(ValueError, "no decode rung B=8"):
            block_roofline.decode_floor(self.progs, 8, 4096, CEIL, 2, 2, None)

    def test_packed_attention_is_charged_per_request_at_its_prefix(self):
        one, _, _ = block_roofline.packed_floor(self.progs, 1, 64, 4096, 0, 0, CEIL, 2, 2, None)
        two, _, _ = block_roofline.packed_floor(self.progs, 2, 64, 4096, 0, 0, CEIL, 2, 2, None)
        prog = self.progs[0]
        p = dict(prog[2][2][3], n_q=64, q_pos0=4096, n_kv=4160)
        nbytes, flops = block_roofline.op_roof.cost("FlashPrefill", p, 64, 4096, 2, 2)
        attn = max(nbytes / (CEIL[0] * 1e6), flops / (CEIL[1] * 1e9))
        linear = lambda rows: sum(block_roofline.op_roof.cost(op, dict(q, rows=rows, M=rows), rows, 0, 2, 2)[0]
                                  / (CEIL[0] * 1e6) for _, op, _, q in prog[2] if op != "FlashPrefill")
        # Two 64-row requests fill the 128-row bucket; one leaves 64 padding rows unpriced.
        self.assertAlmostEqual(two - one, attn + linear(128) - linear(64))
        with self.assertRaisesRegex(ValueError, "no prefill bucket holds 129 rows"):
            block_roofline.packed_floor(self.progs, 1, 128, 0, 1, 0, CEIL, 2, 2, None)

    def test_padding_rows_are_reported_not_priced(self):
        full, full_pad, _ = block_roofline.packed_floor(self.progs, 1, 128, 0, 0, 0, CEIL, 2, 2, None)
        half, half_pad, priced = block_roofline.packed_floor(self.progs, 1, 64, 0, 0, 0, CEIL, 2, 2, None)
        self.assertEqual(full_pad, 0.0)
        self.assertGreater(half_pad, 0.0)
        self.assertEqual(priced[0][1], 2 * 2 * 64 * 2816)  # RmsNorm at the 64 real rows
        rows = block_roofline.build_rows(self.progs, {"layer": 0, "kind": ["full_attn"]},
                                         {("prefill", 1, 64, 0, 0, 0): 1.0}, {}, {}, CEIL, 2, 2, None)
        self.assertIn("(bucket 128: 64 pad rows, +", rows[0]["point"])
        self.assertAlmostEqual(rows[0]["pad_floor_ms"], half_pad)
        self.assertLess(half, full)

    def test_rows_join_measurements_and_render_delta(self):
        plow = block_roofline.measured_points({"sweep": [
            {"batch": 1, "ctx": 4096, "latency_us_median": 100.0},
            {"phase": "prefill", "requests": 1, "query_rows": 64, "prior": 0, "riders": 0,
             "rider_ctx": 0, "ms_median": 2.0}]})
        current = {("decode", 1, 4096): 0.125}
        rows = block_roofline.build_rows(self.progs, {"layer": 3, "kind": ["full_attn", "dense_ffn"]},
                                         plow, current, {}, CEIL, 2, 2, None)
        self.assertEqual([r["point"] for r in rows], ["decode B=1, ctx 4096",
                                                  "prefill 1x64 rows, prior 0 (bucket 128: 64 pad rows, +0.001 ms floor)"])
        text = block_roofline.table(rows)
        self.assertIn("| L3 full_attn+dense_ffn, decode B=1, ctx 4096 | 0.100 ms |", text)
        self.assertIn("-0.025 ms (-20.0%)", text)


class AbTests(unittest.TestCase):
    def test_verdict_follows_the_spread_rule(self):
        self.assertEqual(nv_block_ab.verdict(1.0, 1.01, 0.9, 0.905)[0], "PASS")
        self.assertEqual(nv_block_ab.verdict(1.0, 1.01, 1.1, 1.105)[0], "REGRESSION")
        self.assertEqual(nv_block_ab.verdict(1.0, 1.01, 0.995, 1.0)[0], "NOISE")
        self.assertEqual(nv_block_ab.verdict(1.0, 1.001, 0.8, 0.9)[0], "UNCONVICTABLE")

    def test_score_refuses_unpaired_packets_and_writes_the_table(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ("control", "treatment"):
                (root / name).mkdir()
                (root / name / "model.pkt").write_bytes(name.encode())
                (root / name / "block.json").write_text(json.dumps({"layer": 0, "kind": ["full_attn"]}))
            disasm = root / "disasm.txt"
            disasm.write_text(DISASM)
            fake = root / "plowrt"
            fake.write_text(f"#!/bin/sh\ncat {disasm}\n")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            (root / "block_run").write_bytes(b"bin")
            times = {"ctl": 100.0, "ctl2": 101.0, "treat": 80.0, "treat2": 80.5}
            for arm, us in times.items():
                (root / arm).mkdir()
                pair = nv_block_ab.PAIRS[arm]
                prov = {"packet_sha256": nv_block_ab.sha(root / pair / "model.pkt"),
                        "binary_sha256": nv_block_ab.sha(root / "block_run"), "device": "H100"}
                (root / arm / "sweep.json").write_text(json.dumps(
                    {"sweep": [{"batch": 1, "ctx": 4096, "latency_us_median": us}], "provenance": prov}))
            rows, report = nv_block_ab.score(root)
            self.assertEqual(rows[0]["verdict"], "PASS")
            self.assertIn("delta vs current", report)
            self.assertTrue((root / "comparison.md").is_file())
            bad = json.loads((root / "treat2" / "sweep.json").read_text())
            bad["provenance"]["packet_sha256"] = "0" * 64
            (root / "treat2" / "sweep.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "treat2: measured packet"):
                nv_block_ab.score(root)


if __name__ == "__main__":
    unittest.main()
