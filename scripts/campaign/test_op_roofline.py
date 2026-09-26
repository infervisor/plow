import struct
import unittest

from op_roofline import analyze, attach_trace, attn_keys

CEIL = {"bf16": 2300.0, "fp8": 4600.0, "mxfp4": 9200.0, "f32": 2300.0, "i32": 2300.0}
TEXT = """===== program T=1024  3 insts
#0     FlashMlaPrefill          b=256
#1     MlaMergeFold             b=256  O<-act.oat Wuv<-model.layers.0.self_attn.derived.v_absorb.weight | n_batch=1024 n_head=8 V=256 nsplit=1
#2     XReduceTwoShot           b=256  out<-act.attn | n=6291456 n_gpu=8
===== program T=1  2 insts
#0     XReduce                  b=12   out<-act.attn | H=6144 n_gpu=8 slot=0
#1     GemvMxfp4                b=256  C<-act.y x<-act.x W<-model.layers.3.mlp.shared_experts.down_proj.weight | M=1 N=6144 K=256
===== program T=8  1 insts
#0     Residual                 b=1    out<-act.x | n=49152
"""


class OpRooflineTests(unittest.TestCase):
    def setUp(self):
        self.r = analyze(TEXT, 8192, 6200, CEIL, 1075)["programs"]

    def test_phases_follow_rung_order(self):
        self.assertEqual(list(self.r), ["prefill-1024", "decode-1", "decode-8"])

    def test_bare_operand_instruction_is_priced(self):
        ops = self.r["prefill-1024"]["ops"]
        flash = next(v for k, v in ops.items() if k.startswith("FlashMlaPrefill"))
        self.assertEqual(flash["flops"], 2 * 8 * attn_keys(1024, 0) * (576 + 512))

    def test_collectives_are_fabric_bound_on_total_elements(self):
        x = next(v for k, v in self.r["decode-1"]["ops"].items() if k.startswith("XReduce"))
        self.assertEqual(x["fabric_bytes"], 2 * 2 * 6144 * 7 / 8)
        self.assertEqual(x["bound"], "fabric")

    def test_shared_expert_priced_as_a4w4(self):
        g = next(v for k, v in self.r["decode-1"]["ops"].items() if k.startswith("GemvMxfp4"))
        self.assertEqual(g["dtype"], "mxfp4")
        self.assertEqual(g["hbm_bytes"], 6144 * (128 + 8) + (128 + 8) + 2 * 6144)

    def test_sparse_topk_caps_keys(self):
        self.assertEqual(attn_keys(4096, 2048), 2048 * 2049 // 2 + 2048 * 2048)
        self.assertEqual(attn_keys(100, 2048), 100 * 101 // 2)

    def test_trace_attributes_body_envelope_per_op(self):
        prog = self.r["prefill-1024"]
        rec = lambda inst, part, a, ready, end: struct.pack("<IIIHHQQQ", 0, 0, inst, 0, part, a, ready, end)
        data = rec(0, 0, 1, 5, 105) + rec(0, 1, 2, 10, 205) + rec(2, 0, 300, 300, 400)
        attach_trace(prog, data, 100e6)
        flash = next(v for k, v in prog["ops"].items() if k.startswith("FlashMlaPrefill"))
        self.assertAlmostEqual(flash["measured_us"], 2.0)
        self.assertAlmostEqual(prog["trace_wall_us"], 3.99)
        self.assertEqual(prog["traced_insts"], 2)

    def test_unparsed_instruction_is_refused(self):
        with self.assertRaises(ValueError):
            analyze("===== program T=1  1 insts\n#0 ???\n", 1, 6200, CEIL, 1075)


if __name__ == "__main__":
    unittest.main()
