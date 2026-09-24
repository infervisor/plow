import unittest
import struct

from packet_roofline import analyze, decode_cost, routed_experts, trace_priorities


class PacketRooflineTests(unittest.TestCase):
    def test_native_indexer_uses_live_length_fp8_ceiling_and_packed_key_scales(self):
        text = ("===== program T=64  1 insts\n"
                "#0 IndexFp8Decode b=1 score<-q | M=64 ctx=71680\n")
        result = analyze(text, 8192, 6200, 2300, fp8_tflops=4600)
        self.assertEqual(result["bytes"], 64 * 8192 * (128 + 4))
        self.assertEqual(result["flops"], 2 * 64 * 8192 * 32 * 128)
        self.assertEqual(result["components"]["IndexFp8Decode"]["matrix_tflops"], 4600)
        self.assertTrue(any("physical HBM" in s for s in result["limitations"]))
        self.assertEqual(result["excluded_ops"], {})
        with self.assertRaises(ValueError):
            analyze(text, 8192, 6200, 2300)
        for bad in (text.replace("M=64", "M=32"), text.replace("ctx=71680", "ctx=8191")):
            with self.assertRaises(ValueError):
                analyze(bad, 8192, 6200, 2300, fp8_tflops=4600)
        with self.assertRaises(ValueError):
            analyze(text, 71681, 6200, 2300, fp8_tflops=4600)

    def test_observed_counter_chain_does_not_serialize_independent_packets(self):
        program = {"n_inst":3,"n_counter":3,"insts":[
            {"idx":i,"op":4,"op_name":"Residual","blocks":1} for i in range(3)],
            "counters":{"per_counter":[
                {"id":0,"producer":0,"threshold":1,"consumers":[2]},
                {"id":1,"producer":1,"threshold":1,"consumers":[2]},
                {"id":2,"producer":2,"threshold":1,"consumers":[]}]}}
        def record(inst, ready, end):
            return struct.pack("<IIIHHQQQ",0,inst,inst,4,0,ready,ready,end)
        trace = record(0,1,11) + record(1,1,21) + record(2,21,26)
        result = trace_priorities(trace,program,1e9)
        self.assertEqual(result["counter_chain_ns"],25)
        self.assertEqual(result["observed_wall_ns"],25)
        self.assertEqual(result["chain"],[1,2])
        self.assertEqual(result["priorities"][0]["instruction"],1)
        self.assertEqual(result["priorities"][-1]["chain_slack_ns"],10)
        self.assertFalse(result["performance_qualified"])
        import copy
        for mutation in range(14):
            bad = copy.deepcopy(program)
            raw = trace
            hz = 1e9
            if mutation == 0: raw = raw[:-1]
            elif mutation == 1: raw = raw[:80]
            elif mutation == 2: raw += raw[:40]
            elif mutation == 3: bad["insts"][0]["op"] = 5
            elif mutation == 4: bad["counters"]["per_counter"][0]["threshold"] = 2
            elif mutation == 5: bad["counters"]["per_counter"][2]["consumers"] = [0]
            elif mutation == 6: bad["n_counter"] = 4
            elif mutation == 7: hz = float("nan")
            elif mutation == 8: hz = True
            elif mutation == 9: bad["counters"]["per_counter"][0]["id"] = -1
            elif mutation == 10: bad["counters"]["per_counter"][0]["id"] = 3
            elif mutation == 11: bad["counters"]["per_counter"][0]["threshold"] = True
            elif mutation == 12: bad["n_counter"] = True
            else: bad["insts"][0]["idx"] = False
            with self.assertRaises(ValueError, msg=f"mutation {mutation}"):
                trace_priorities(raw,bad,hz)

    def test_roofline_is_explicitly_conditional_not_physical_traffic(self):
        text = "===== program T=1 1 insts\n#0 Gemv b=256 C<-x | M=1 N=128 K=256\n"
        result = analyze(text,512,6200,2300)
        self.assertIsNone(result["physical_hbm_bytes"])
        self.assertFalse(result["performance_qualified"])
        self.assertEqual(result["logical_operand_bytes"],result["bytes"])
        for value in (float("nan"),float("inf"),-1,0):
            with self.assertRaises(ValueError): analyze(text,512,value,2300)
            with self.assertRaises(ValueError): analyze(text,512,6200,value)

    def test_native_mla_fp8_counts_all_heads_and_one_scalar_scale(self):
        text = ("===== program T=16  2 insts\n"
                "#0 MlaBmmFp8 b=256 C<-x | M=16 heads=8 N=512 K=192 copy_rope64=1\n"
                "#1 MlaBmmFp8 b=256 C<-x | M=16 heads=8 N=256 K=512 copy_rope64=0\n")
        with self.assertRaises(ValueError):
            analyze(text, 512, 6200, 2300)
        result = analyze(text, 512, 6200, 2300, fp8_tflops=4600)
        self.assertEqual(result["bytes"], 8 * (512 * 192 + 256 * 512) + 8)
        self.assertEqual(result["flops"], 2 * 16 * 8 * (512 * 192 + 256 * 512))
        self.assertEqual(result["components"]["MlaBmmFp8"]["matrix_tflops"], 4600)
        self.assertEqual(result["excluded_ops"], {})

    def test_native_routed_fp8_costs_both_projections_and_uses_captured_union(self):
        text = ("===== program T=2  4 insts\n"
                "#0 MoeAlignPf b=1 meta<-routes | T=2 n_exp=4 k=2\n"
                "#1 MoeGluFp8Block128 b=256 fu<-x | I=256 H=6144 E=4 T=2\n"
                "#2 MoeQuantFp8Block128 b=256 q<-fu | I=256 E=4 topk=2 T=2 H=6144\n"
                "#3 MoeDownFp8Block128 b=256 out<-q | I=256 H=6144 E=4 topk=2 T=2\n")
        table = b"".join(struct.pack("<If", expert, 0.5) for expert in (1, 2, 2, 3))
        result = analyze(text, 512, 6200, 2300, table, fp8_tflops=4600)
        self.assertEqual(result["routed_experts"], 3)
        self.assertEqual(result["bytes"], 3 * 3 * (256 * 6144 + 2 * 48 * 4))
        self.assertEqual(result["flops"], 6 * 2 * 2 * 256 * 6144)
        self.assertEqual(result["components"]["MoeGluFp8Block128"]["matrix_tflops"], 4600)
        self.assertEqual(result["excluded_ops"], {"MoeAlignPf": 1, "MoeQuantFp8Block128": 1})
        with self.assertRaises(ValueError):
            analyze(text, 512, 6200, 2300, table)
        for bad in (text.replace("E=4 T=2", "E=3 T=2"), text.replace("E=4 T=2", "E=4 T=1"),
                    text.replace("topk=2 T=2", "topk=1 T=2"), text.replace("MoeAlignPf", "Nop")):
            with self.assertRaises(ValueError):
                analyze(bad, 512, 6200, 2300, table, fp8_tflops=4600)

    def test_native_fp8_uses_its_own_compute_ceiling(self):
        text = ("===== program T=16  3 insts\n"
                "#0 GemmFp8Block128 b=256 C<-x | M=16 N=256 K=6144\n"
                "#1 GemmFp8Block128Split4 b=256 C<-x | M=16 N=6144 K=2048\n"
                "#2 GemvFp8Blk b=256 C<-x | M=16 N=256 K=6144\n")
        with self.assertRaises(ValueError):
            analyze(text, 512, 6200, 2300)
        result = analyze(text, 512, 6200, 2300, fp8_tflops=4600)
        parts = result["components"]
        self.assertEqual(parts["GemmFp8Block128"]["compute_floor_us"] * 2,
                         parts["GemvFp8Blk"]["compute_floor_us"])
        self.assertEqual(parts["GemmFp8Block128Split4"]["flops"], 2 * 16 * 6144 * 2048)
        self.assertEqual(result["excluded_ops"], {})

    def test_mxfp4_packet_ops_use_packed_weights_and_explicit_ceiling(self):
        text = ("===== program T=16  5 insts\n"
                "#0 GemvMxfp4 b=256 C<-x | M=16 N=256 K=6144\n"
                "#1 GemmSmallMxfp4 b=256 C<-x | M=16 N=256 K=6144\n"
                "#2 GemmGluMxfp4 b=256 C<-x | M=16 N=256 K=6144\n"
                "#3 MoeAlignPf b=1 meta<-routes | T=16 n_exp=256 k=8\n"
                "#4 MoeGroupGluPf b=256 fu<-x | I_moe=256 H=6144 n_exp=256 fp8=2\n")
        with self.assertRaisesRegex(ValueError, "MXFP4 compute ceiling"):
            analyze(text, 8192, 6200, 2300)
        result = analyze(text, 8192, 6200, 2300, mxfp4_tflops=9200)
        weight = 256 * (6144 // 2 + 6144 // 32)
        self.assertEqual(result["components"]["GemvMxfp4"]["bytes"], weight)
        self.assertEqual(result["components"]["GemmGluMxfp4"]["bytes"], 2 * weight)
        self.assertEqual(result["components"]["MoeGroupGluPf"]["bytes"], 2 * 8 * weight)
        self.assertEqual(result["components"]["MoeGroupGluPf"]["matrix_tflops"], 9200)
        self.assertEqual(result["excluded_ops"], {"MoeAlignPf": 1})
        for op in ("GemmMedMxfp4", "GemmWideMxfp4", "GemmMxfp4", "GemvGluMxfp4"):
            size, flops = decode_cost(op, dict(M=16, N=256, K=6144), 8192)
            factor = 2 if op == "GemvGluMxfp4" else 1
            self.assertEqual(size, factor * weight)
            self.assertEqual(flops, factor * 2 * 16 * 256 * 6144)

    def test_mxfp4_expert_opcode_name_does_not_imply_fp8_bytes(self):
        p = dict(I_moe=256, H=6144, enc=2)
        weight = 256 * (6144 // 2 + 6144 // 32)
        self.assertEqual(decode_cost("MoeExpertGluFp8Blk", p, 8192)[0], 2 * weight)
        text = ("===== program T=1  1 insts\n"
                "#0 MoeExpertGluFp8Blk b=32 fu<-x | I_moe=256 H=6144 enc=2\n")
        with self.assertRaisesRegex(ValueError, "MXFP4 compute ceiling"):
            analyze(text, 8192, 6200, 2300)
        self.assertEqual(analyze(text, 8192, 6200, 2300, mxfp4_tflops=9200)
                         ["components"]["MoeExpertGluFp8Blk"]["bytes"], 2 * weight)

    def test_captured_expert_union_and_rejections(self):
        table = b"".join(struct.pack("<If", expert, 0.5) for expert in (1, 2, 2, 3))
        self.assertEqual(routed_experts(table, 2, 2, 4), 3)
        for bad in (table[:-1], table + table, struct.pack("<If", 4, 0.5) * 4,
                    struct.pack("<If", 1, float("nan")) * 4, struct.pack("<If", 1, 0.5) * 4):
            with self.assertRaises(ValueError):
                routed_experts(bad, 2, 2, 4)
        text = ("===== program T=2  2 insts\n"
                "#0 MoeAlignPf b=1 meta<-act.meta | T=2 n_exp=4 k=2\n"
                "#1 MoeGroupGluPf b=256 fu<-act.fu | I_moe=256 H=6144 n_exp=4 fp8=1 act=1\n")
        actual = analyze(text, 512, 6200, 2300, table)
        optimistic = analyze(text, 512, 6200, 2300)
        self.assertEqual(actual["routed_experts"], 3)
        self.assertEqual(actual["bytes"], optimistic["bytes"] * 3 // 2)
        self.assertEqual(actual["flops"], optimistic["flops"])
        with self.assertRaises(ValueError):
            analyze(text.replace("program T=2", "program T=4"), 512, 6200, 2300, table)

    def test_sparse_attention_uses_live_context_and_selection_cap(self):
        p = dict(n_batch=8, n_head=8, kv_stride=131072, top_k=2048)
        self.assertEqual(decode_cost("FlashGatherDecode", p, 70000)[0], 8 * 2048 * 576 * 2)
        self.assertEqual(decode_cost("FlashGatherDecode", p, 512)[0], 8 * 512 * 576 * 2)

    def test_batched_projection_reuses_weights_but_scales_flops(self):
        p = dict(M=16, N=6144, K=2048)
        memory, flops = decode_cost("Gemv", p, 8192)
        self.assertEqual(memory, 6144 * 2048 * 2)
        self.assertEqual(flops, memory * 16)

    def test_grouped_experts_do_not_charge_every_expert_per_decode(self):
        p = dict(T=16, k=8, n_exp=256, I_moe=256, H=6144)
        memory, flops = decode_cost("MoeGroupDownPf", p, 8192)
        self.assertEqual(memory, 8 * (256 * 6144 + 2 * 48 * 4))
        self.assertEqual(flops, 2 * 16 * 8 * 256 * 6144)

    def test_unmodeled_ops_remain_visible(self):
        text = "===== program T=1  2 insts\n#0 Gemv b=256 C<-x | M=1 N=128 K=256\n#1 XReduce b=12 x<-a | H=128\n"
        result = analyze(text, 8192, 6200, 2300)
        self.assertEqual(result["excluded_ops"], {"XReduce": 1})
        self.assertEqual(result["bytes"], 65536)
        with self.assertRaises(ValueError):
            analyze(text + text, 8192, 6200, 2300)

    def test_grouped_shape_comes_from_align_not_gemm_fields(self):
        text = ("===== program T=8  2 insts\n"
                "#0 MoeAlignPf b=1 meta<-act.meta | T=8 n_exp=256 k=8\n"
                "#1 MoeGroupGluPf b=256 fu<-act.fu | I_moe=256 H=6144 n_exp=256 fp8=1 act=1\n")
        result = analyze(text, 512, 6200, 2300)
        self.assertEqual(result["flops"], 4 * 8 * 8 * 256 * 6144)
        self.assertEqual(result["bytes"], 2 * 8 * (256 * 6144 + 2 * 48 * 4))
        with self.assertRaises(ValueError):
            analyze(text.replace("MoeAlignPf", "Nop"), 512, 6200, 2300)


if __name__ == "__main__":
    unittest.main()
