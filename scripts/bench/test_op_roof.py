import json
import sqlite3
import unittest
import tempfile
from pathlib import Path

import op_roof


class SweepTests(unittest.TestCase):
    def test_roofline_rejects_unpriced_instructions(self):
        program = ("prefill", 128, [(1, "HeadNormRopeFp8", "", {})])
        priced = op_roof.price(program, 128, 2, 2, 3173, 989)
        with self.assertRaisesRegex(ValueError, "prefill T=128 has unpriced ops: HeadNormRopeFp8"):
            op_roof.require_priced(program, priced)

    def test_empty_sweep_is_not_treated_as_a_measurement(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sweep.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "no instruction caps"):
                op_roof.sweep_deltas(path)

    def test_missing_cap_or_full_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sweep.jsonl"
            for rows in (
                [{"cap": 0, "ms": 1}, {"cap": 2, "ms": 3}, {"cap": -1, "ms": 3}],
                [{"cap": 0, "ms": 1}, {"cap": 1, "ms": 2}],
            ):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "incomplete instruction caps"):
                    op_roof.sweep_deltas(path)


class QuantizedDecodeCostTests(unittest.TestCase):
    def test_fp8_weights_do_not_quantize_activations_or_scales(self):
        shape = {"M": 32, "N": 3840, "K": 8192}
        expected = (31457280 + 15360 + 524288 + 245760, 2013265920)
        for weight_bytes in (1, 2):
            self.assertEqual(op_roof.cost("GemvFp8", shape, 32, 128, weight_bytes, 2), expected)

    def test_fused_glu_reads_two_weights_but_one_input(self):
        shape = {"M": 32, "N": 15360, "K": 3840}
        self.assertEqual(op_roof.cost("GemvGluFp8", shape, 32, 128, 2, 2),
                         (117964800 + 122880 + 245760 + 983040, 7549747200))

    def test_fp8_gemm_reads_quantized_input_and_both_scale_vectors(self):
        shape = {"M": 128, "N": 4096, "K": 3840}
        expected = (15728640 + 491520 + 16896 + 1048576, 4026531840)
        for op in ("GemmFp8", "GemmMedFp8", "GemmSmallFp8"):
            for weight_bytes in (1, 2):
                self.assertEqual(op_roof.cost(op, shape, 128, 128, weight_bytes, 2), expected)

    def test_compute_bound_fp8_uses_its_own_peak(self):
        shape = {"M": 8192, "N": 4096, "K": 3840}
        prog = ("prefill", 8192, [(0, "GemmFp8", "", shape), (1, "Gemm", "", shape)])
        priced = op_roof.price(prog, 128, 2, 2, 3000, 989, 1979)
        self.assertEqual(priced[0][4], "matrix")
        self.assertAlmostEqual(priced[0][3], 2 * 8192 * 4096 * 3840 / (1979 * 1e9))
        self.assertAlmostEqual(priced[1][3], 2 * 8192 * 4096 * 3840 / (989 * 1e9))

    def test_quant_and_gemm_include_intermediate_write_and_read(self):
        shape = {"M": 128, "N": 4096, "K": 3840}
        quant = op_roof.cost("QuantFp8", shape, 128, 128, 2, 2)
        gemm = op_roof.cost("GemmFp8", shape, 128, 128, 2, 2)
        self.assertEqual(quant, (1475072, 0))
        self.assertEqual(quant[0] + gemm[0], 18760704)

    def test_quant_disasm_distinguishes_fused_glu(self):
        for gate, expected in (("—", 1475072), ("act.gate", 3441152)):
            text = ("===== program T=128 =====\n"
                    f"#2 QuantFp8 b=128 xq<-act.xqh x<-act.hn a_scale<-act.ash "
                    f"gate<-{gate} up<-{gate} | M=128 K=3840 act=0")
            prog = op_roof.parse(text)[0]
            priced = op_roof.price(prog, 128, 2, 2, 3000, 989)
            self.assertEqual(priced[2][1], expected)
            self.assertEqual(priced[2][2], 0)

    def test_fused_fp8_glu_shares_input_and_selects_activation_precision(self):
        for scale, fp8 in (("a_scale<-act.ash", True), ("a_scale<-—", False), ("", False)):
            text = ("===== program T=4096 =====\n"
                    f"#15 GemmGluFp8 b=132 fu<-act.fu A<-act.x Wg<-fp8/layers.0.gate {scale} "
                    "Wu<-fp8/layers.0.up | M=4096 N=15360 K=3840 act=0")
            prog = op_roof.parse(text)[0]
            priced = op_roof.price(prog, 2048, 2, 2, 3000, 989, 1979)[15]
            self.assertEqual(priced[0], "GemmGluFp8:fp8/gate")
            activation = 4096 * 3840 * (1 if fp8 else 2) + (4 * 4096 if fp8 else 0)
            self.assertEqual(priced[1], 2 * (15360 * 3840 + 4 * 15360) + activation + 2 * 4096 * 15360)
            flops = 4 * 4096 * 15360 * 3840
            self.assertEqual(priced[2], flops)
            self.assertEqual(priced[4], "matrix")
            self.assertAlmostEqual(priced[3], flops / ((1979 if fp8 else 989) * 1e9))

    def test_lm_head_stays_bf16_with_fp8_projections(self):
        shape = {"N": 262144, "K": 3840}
        self.assertEqual(op_roof.cost("GemvArgmax", shape, 32, 128, 1, 2),
                         (2013265920 + 245760 + 16777216, 64424509440))


class PrefillPrefixCostTests(unittest.TestCase):
    def test_fp8_kv_attention_uses_one_byte_cache_and_prefix_override(self):
        decode = dict(n_batch=64, n_head=16, n_kv_head=8, hd=256, window=1024)
        bf16 = op_roof.cost("FlashDecode", decode, 64, 4096, 2, 2)
        fp8 = op_roof.cost("FlashDecodeFp8", decode, 64, 4096, 2, 1)
        self.assertEqual(fp8[0], bf16[0] - 64 * 1024 * 8 * (256 * 2 - 8))
        self.assertEqual(fp8[1], bf16[1])
        self.assertEqual(op_roof.cost("FlashDecodeFp8", decode, 64, 4096, 2, 2), fp8)
        prefill = ("prefill", 128, [(0, "FlashPrefillFp8", "", dict(
            n_q=128, n_kv=128, n_head=16, n_kv_head=8, hd=256, window=1024, q_pos0=0))])
        initial = op_roof.price(prefill, 4096, 2, 1, 3173, 989)
        later = op_roof.price(prefill, 4096, 2, 1, 3173, 989, prefill_past=512)
        self.assertEqual(initial[0][1], 4 * 128 * 16 * 256 + 128 * 8 * (256 * 2 + 8))
        self.assertGreater(later[0][2], initial[0][2])

    def test_fp8_kv_writers_are_priced_in_prefill_and_decode_programs(self):
        text = ("===== program T=128 =====\n"
                "#1 HeadNormRopeFp8 b=132 out<-kv.0.k scale<-kv.0.k_scale\n"
                "#2 HeadNormRopeFp8 b=132 out<-kv.0.v scale<-kv.0.v_scale\n"
                "#3 FlashPrefillFp8 b=132 K<-kv.0.k V<-kv.0.v | "
                "n_q=128 n_kv=128 n_head=16 n_kv_head=8 hd=256 window=1024 q_pos0=0\n"
                "===== program T=64 =====\n"
                "#1 HeadNormRopeFp8 b=132 out<-kv.0.k scale<-kv.0.k_scale\n"
                "#2 HeadNormRopeFp8 b=132 out<-kv.0.v scale<-kv.0.v_scale\n"
                "#3 FlashDecodeFp8 b=132 K<-kv.0.k V<-kv.0.v | "
                "n_batch=64 n_head=16 n_kv_head=1 hd=512 window=0\n")
        prefill, decode = op_roof.parse(text)
        self.assertEqual((prefill[0], decode[0]), ("prefill", "decode"))
        for prog, rows, kv_heads, hd in ((prefill, 128, 8, 256), (decode, 64, 1, 512)):
            priced_default = op_roof.price(prog, 4096, 2, 2, 3173, 989)
            priced_fp8 = op_roof.price(prog, 4096, 2, 1, 3173, 989)
            self.assertEqual(priced_default, priced_fp8)
            writer_bytes = rows * kv_heads * (3 * hd + 4)
            for idx in (1, 2):
                self.assertEqual(priced_default[idx][1:3], (writer_bytes, 0))
                self.assertEqual(priced_default[idx][4], "hbm")
            self.assertTrue(all(entry[4] != "unpriced" for entry in priced_default.values()))

    def test_causal_pairs_include_diagonal_and_window_transition(self):
        for window in (0, 8):
            for past in (0, 5, 7, 16):
                shape = dict(n_q=4, n_kv=past + 4, n_head=16,
                             n_kv_head=8, hd=256, window=window, q_pos0=past)
                by, fl = op_roof.cost("FlashPrefill", shape, 4, 15000, 2, 2)
                lengths = [min(past + j, window) if window else past + j for j in range(1, 5)]
                self.assertEqual(fl, 4 * 16 * 256 * sum(lengths))
                first = max(0, past + 1 - window) if window else 0
                self.assertEqual(by, 4 * 4 * 16 * 256 + (past + 4 - first) * 8 * 256 * 4)

    def test_prefix_override_does_not_mutate_packet_or_decode(self):
        shape = dict(n_q=128, n_kv=128, n_head=16, n_kv_head=1, hd=512, q_pos0=0)
        prefill = ("prefill", 128, [(0, "FlashPrefill", "", shape)])
        initial = op_roof.price(prefill, 15000, 2, 2, 3000, 989)
        later = op_roof.price(prefill, 15000, 2, 2, 3000, 989, prefill_past=14000)
        self.assertGreater(later[0][1], initial[0][1])
        self.assertEqual(later[0][2] - initial[0][2], 4 * 16 * 512 * 128 * 14000)
        self.assertEqual(shape["q_pos0"], 0)
        decode = ("decode", 128, [(0, "FlashDecode", "", dict(n_batch=128, n_head=16,
                   n_kv_head=1, hd=512, window=0))])
        self.assertEqual(op_roof.price(decode, 15000, 2, 2, 3000, 989),
                         op_roof.price(decode, 15000, 2, 2, 3000, 989, prefill_past=14000))


class SegtimeChunkTests(unittest.TestCase):
    def test_selects_whole_chunk_without_adjacent_sites(self):
        lines = ["startup", "seg-class wall time (chunk) first", "site0",
                 "seg-class wall time (chunk) second", "site1", "site2"]
        self.assertEqual(op_roof.select_segtime_chunk(lines, 0), lines[1:3])
        self.assertEqual(op_roof.select_segtime_chunk(lines, 1), lines[3:])
        self.assertIs(op_roof.select_segtime_chunk(lines, None), lines)

    def test_rejects_missing_chunk(self):
        for index in (-1, 0, 1):
            with self.assertRaises(ValueError):
                op_roof.select_segtime_chunk([], index)
        with self.assertRaises(ValueError):
            op_roof.select_segtime_chunk(["seg-class wall time (chunk)"], 1)


class SegmentLaunchTests(unittest.TestCase):
    def setUp(self):
        self.program = {"stream": [{"seg": 0, "inst": 0}, {"seg": 0, "inst": 1},
                                    {"seg": 1, "inst": 2}],
                        "insts": [{"idx": 0, "op_name": "RmsNorm"},
                                  {"idx": 1, "op_name": "QuantFp8"},
                                  {"idx": 2, "op_name": "GemmFp8", "tensors": [
                                      {"name": "B", "tensor": "layers.3.q.weight"}]}]}
        self.launches = [{"start": 0, "end": 1000, "name": "interp_sm90a_gw(PlowProgram)"},
                         {"start": 1000, "end": 4000, "name": "nvjet_fp8"}]

    def test_fused_segment_keeps_one_measured_duration(self):
        rows = op_roof.segment_launches(self.program, self.launches)
        self.assertEqual(rows[0]["group"], "RmsNorm+QuantFp8")
        self.assertEqual(rows[0]["ms"], .001)
        self.assertEqual(rows[1]["group"], "q.weight")
        self.assertEqual(sum(r["ms"] for r in rows), .004)

    def test_main_decode_interpreter_maps_like_wide_sibling(self):
        self.launches[0]["name"] = "interp_sm90a(PlowProgram)"
        rows = op_roof.segment_launches(self.program, self.launches)
        self.assertEqual(rows[0]["kernel"], "interp_sm90a(PlowProgram)")

    def test_direct_norm_quant_maps_one_launch(self):
        program = {"stream": [{"seg": 0, "inst": 0}, {"seg": 0, "inst": 1}],
                   "insts": [{"idx": 0, "op_name": "NormResidualNorm"},
                             {"idx": 1, "op_name": "QuantFp8"}]}
        launches = [{"start": 0, "end": 1000, "name": "plow_sm90a_light_norm_quant"}]
        rows = op_roof.segment_launches(program, launches)
        self.assertEqual(rows[0]["group"], "NormResidualNorm+QuantFp8 [direct]")
        self.assertEqual(rows[0]["ms"], .001)
        launches[0]["name"] = "plow_sm90a_light"
        with self.assertRaises(ValueError):
            op_roof.segment_launches(program, launches)

    def test_light_head_tail_counts_both_kernels_without_launch_gap(self):
        program = {"stream": [{"seg": 0, "inst": i} for i in range(3)],
                   "insts": [{"idx": i, "op_name": name} for i, name in
                             enumerate(["SoftCap", "Argmax", "ArgmaxFin"])]}
        launches = [{"start": 0, "end": 1000, "name": "plow_sm90a_light_capmax"},
                    {"start": 2000, "end": 3000, "name": "plow_sm90a_light_tail"}]
        rows = op_roof.segment_launches(program, launches)
        self.assertEqual(rows[0]["ms"], .002)
        self.assertEqual(len(rows[0]["kernels"]), 2)
        with self.assertRaises(ValueError):
            op_roof.segment_launches(program, launches[:1])
        with self.assertRaises(ValueError):
            op_roof.segment_launches(program, launches + [dict(start=3000, end=4000, name="extra")])

    def test_light_attention_segment_counts_six_kernels(self):
        for flash in ("FlashDecode", "FlashDecodeFp8"):
            program = {"stream": [{"seg": 0, "inst": i} for i in range(3)],
                       "insts": [{"idx": i, "op_name": name} for i, name in
                                 enumerate(["HeadNormRope", flash, "FlashMerge"])]}
            for attention in ("plow_sm90a_light_attn_s", "plow_sm90a_light_attn"):
                variants = [[attention] * 4]
                if attention.endswith("_s"):
                    variants.append([attention] + ["plow_sm90a_light_attn"] * 3)
                for variant in variants:
                    names = variant + ["plow_sm90a_light"] * 2
                    launches = [dict(start=i * 1000, end=(i + 1) * 1000, name=name)
                                for i, name in enumerate(names)]
                    rows = op_roof.segment_launches(program, launches)
                    self.assertEqual(len(rows[0]["kernels"]), 6)
                    self.assertEqual(rows[0]["ms"], .006)
                    with self.assertRaises(ValueError):
                        op_roof.segment_launches(program, launches[:-1])

    def test_bf16_head_route_requires_head_weight(self):
        weight = {"slot": "t2", "tensor": "model.embed_tokens.weight"}
        program = {"stream": [{"seg": 0, "inst": 0}],
                   "insts": [{"idx": 0, "op_name": "Gemv", "tensors": [weight]}]}
        launches = [{"start": 0, "end": 1000, "name": "nvjet_tst_head"}]
        self.assertEqual(op_roof.segment_launches(program, launches)[0]["group"], "BF16 Lt head")
        weight["tensor"] = "model.layers.0.q_proj.weight"
        with self.assertRaises(ValueError):
            op_roof.segment_launches(program, launches)

    def test_reject_partial_graph(self):
        with self.assertRaises(ValueError):
            op_roof.segment_launches(self.program, self.launches[:1])

    def test_reject_swapped_routes(self):
        self.launches[0]["name"] = "nvjet_fp8"
        with self.assertRaises(ValueError):
            op_roof.segment_launches(self.program, self.launches)

    def test_reject_overlap(self):
        self.launches[1]["start"] = 999
        with self.assertRaises(ValueError):
            op_roof.segment_launches(self.program, self.launches)

    def test_prefill_routes_match_packet_order(self):
        names = ["FlashPrefillFp8", "GemmFp8", "QuantFp8", "Gemm"]
        program = {"stream": [{"seg": i, "inst": i} for i in range(4)],
                   "insts": [{"idx": 0, "op_name": names[0], "ints": [
                       {"name": "hd", "value": 256}, {"name": "n_head", "value": 16},
                       {"name": "n_kv_head", "value": 8}, {"name": "window", "value": 1024}]},
                             {"idx": 1, "op_name": names[1], "tensors": [
                                 {"name": "B", "tensor": "layers.0.q_proj.weight"}]},
                             {"idx": 2, "op_name": names[2], "tensors": [
                                 {"name": "gate", "present": True}, {"name": "up", "present": True}]},
                             {"idx": 3, "op_name": names[3]}]}
        kernels = ["interp_sm90a_pfpackedseg(PlowProgram)", "nvjet_fp8",
                   "plow_glu_quant_cached_pfpackedseg", "interp_sm90a_pfpackedgemm(PlowProgram)"]
        launches = [dict(start=i * 1000, end=(i + 1) * 1000, name=name)
                    for i, name in enumerate(kernels)]
        rows = op_roof.segment_launches(program, launches)
        self.assertEqual([row["group"] for row in rows],
                         ["FlashPrefillFp8 [hd=256, n_head=16, n_kv_head=8, window=1024]",
                          "q_proj.weight", "QuantFp8 [cached GLU]", "Gemm"])
        launches[2]["name"] = kernels[0]
        program["insts"][2]["tensors"][0]["present"] = False
        with self.assertRaisesRegex(ValueError, "cached GLU route requires gate and up"):
            op_roof.segment_launches(program, [*launches[:2], dict(launches[2], name=kernels[2]), launches[3]])

    def test_explicit_nsys_correlation_selects_prefill_graph(self):
        program = {"t": 128, "stream": [{"seg": 0, "inst": 0}],
                   "insts": [{"idx": 0, "op_name": "FlashPrefillFp8"}]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("create table StringIds(id integer, value text)")
                db.execute("create table CUPTI_ACTIVITY_KIND_KERNEL("
                           "start integer,end integer,demangledName integer,correlationId integer)")
                db.execute("insert into StringIds values(1,'interp_sm90a_pfpackedseg(PlowProgram)')")
                db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,1,?)",
                               [(0, 1000, 17), (2000, 5000, 19)])
            result = op_roof.nsys_segments({"programs": [program]}, 0, path, correlations=[17])
            self.assertEqual(result["graphs"][0]["correlation_id"], 17)
            self.assertEqual(result["groups"]["FlashPrefillFp8"]["mean_ms_per_graph"], .001)
            with self.assertRaisesRegex(ValueError, "trace lacks correlation"):
                op_roof.nsys_segments({"programs": [program]}, 0, path, correlations=[18])


if __name__ == "__main__":
    unittest.main()
