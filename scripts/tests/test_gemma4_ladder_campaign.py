import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gemma4_ladder_campaign", ROOT / "scripts" / "gemma4_ladder_campaign.py"
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def audit_fixture():
    programs = []
    for rung in campaign.RUNGS:
        cases = []
        pc = 0
        for (n, k), count in campaign.GEMM_COUNTS.items():
            m = 1 if n == 262144 else rung
            cases.append({"op": "Gemm", "i": [m, n, k, 0, 0, 0, 0, 0], "pcs": list(range(pc, pc + count))})
            pc += count
        for (qh, kvh, hd, window, splits), count in campaign.ATTENTION_COUNTS.items():
            stride, mask = (16384, 16383) if window else (20480, 0xffffffff)
            cases.append({
                "op": "FlashPrefill",
                "i": [rung, rung, qh, kvh, 0, window, hd, splits],
                "fj_bits": [0, stride, mask],
                "pcs": list(range(pc, pc + count)),
            })
            pc += count
        programs.append({"phase": "prefill", "rows": rung, "kernel_cases": cases})
    for rung in campaign.DECODE_RUNGS:
        cases = []
        pc = 0
        for (op, n, k, fused_n0, fused_n1), count in campaign.DECODE_GEMM_COUNTS.items():
            cases.append({
                "op": op, "i": [rung, n, k, fused_n0, fused_n1, 0, 0, 0],
                "pcs": list(range(pc, pc + count)),
            })
            pc += count
        for dims, count in campaign.DECODE_ATTENTION_COUNTS.items():
            kv_dtype, qh, kvh, stride, window, split_cap, hd, mask = dims
            cases.append({
                "op": "FlashDecodeFp8" if kv_dtype == "fp8_kv" else "FlashDecode",
                "i": [rung, qh, kvh, stride, window, split_cap, hd, mask],
                "pcs": list(range(pc, pc + count)),
            })
            pc += count
        programs.append({"phase": "decode", "rows": rung, "kernel_cases": cases})
    return {"packet_sha256": "a" * 64, "programs": programs}


class Gemma4LadderCampaignTests(unittest.TestCase):
    def test_inventory_has_every_shape_mode_and_stable_keys(self):
        profiles, rungs, decode_rungs = campaign.audit_inventory(
            audit_fixture(), "sm90a", "bf16", "packed-varlen", 128
        )
        self.assertEqual([x["rung"] for x in rungs], list(campaign.RUNGS))
        self.assertEqual([x["rung"] for x in decode_rungs], list(campaign.DECODE_RUNGS))
        self.assertEqual(len(profiles), 3456)
        self.assertEqual(sum(x["phase"] == "prefill" for x in profiles), 2753)
        self.assertEqual(sum(x["phase"] == "decode" for x in profiles), 703)
        self.assertEqual(len({x["profile_key"] for x in profiles}), len(profiles))
        local = next(
            x for x in profiles
            if x["phase"] == "prefill" and x["family"] == "attention"
            and x["rung"] == 8192 and x["head_dim"] == 256
            and x["request_topology"] == "single" and x["kv_length"] == 16384
        )
        self.assertEqual((local["q_heads"], local["kv_heads"], local["gqa"], local["window"]), (16, 8, 2, 1024))
        self.assertEqual(local["kv_dtype"], "bf16_kv")
        self.assertEqual((local["packet_kv_rows"], local["kv_length"], local["effective_kv_rows"]), (8192, 16384, 1024))

    def test_kv_growth_has_boundary_neighbors_and_request_topologies(self):
        profiles, _, _ = campaign.audit_inventory(
            audit_fixture(), "sm90a", "bf16", "packed-varlen", 128
        )
        self.assertTrue({1023, 1024, 1025, 16383, 16384}.issubset(campaign.LOCAL_KV_LENGTHS))
        self.assertTrue({127, 128, 129, 8191, 8192, 8193}.issubset(campaign.GLOBAL_KV_LENGTHS))
        cells = [x for x in profiles if x["phase"] == "prefill"
                 and x["family"] == "attention" and x["rung"] == 128
                 and x["head_dim"] == 512 and x["kv_length"] == 4097]
        self.assertEqual({x["request_topology"] for x in cells},
                         {"single", "packed_homogeneous", "packed_ragged"})
        ragged = next(x for x in cells if x["history_layout"] == "ragged")
        self.assertLess(ragged["kv_length_min"], ragged["kv_length"])

    def test_decode_expands_homogeneous_and_ragged_live_histories(self):
        profiles, _, _ = campaign.audit_inventory(
            audit_fixture(), "sm90a", "bf16", "packed-varlen", 128
        )
        cells = [x for x in profiles if x["phase"] == "decode"
                 and x["family"] == "attention" and x["rung"] == 16
                 and x["head_dim"] == 256 and x["kv_length"] == 2049]
        self.assertEqual({x["request_topology"] for x in cells},
                         {"decode_homogeneous", "decode_ragged"})
        self.assertTrue(all(x["query_rows"] == 1 and x["active_requests"] == 16 for x in cells))

    def test_inventory_rejects_one_missing_rung_shape(self):
        audit = audit_fixture()
        audit["programs"][3]["kernel_cases"][0]["pcs"].pop()
        with self.assertRaisesRegex(campaign.CampaignError, "GEMM shapes/counts"):
            campaign.audit_inventory(audit, "sm90a", "bf16", "packed", 128)

    def test_weighted_gate_rejects_a_hot_shape_regression(self):
        base = {
            "arch": "sm90a", "dtype": "bf16", "phase": "prefill",
            "family": "gemm", "rung": 1, "request_topology": "single",
            "concurrency": 1, "packed_topology": "packed", "m": 1, "n": 1,
            "k": 1, "occurrences": 100, "profile_key": "hot",
            "control_us": 10.0, "candidate_us": 11.0,
            "weighted_control_us": 1000.0, "weighted_candidate_us": 1100.0,
        }
        gate, ranked = campaign.gate_kernels([base], {
            "kernel_regression_tolerance": 1.02, "minimum_weighted_speedup": 1.0
        })
        self.assertFalse(gate["pass"])
        self.assertEqual(ranked[0]["cost_rank"], 1)

    def test_serving_gate_separates_latency_and_max_throughput(self):
        cells = []
        for arm, ttft, throughput in (("candidate", 9.0, 101.0), ("control", 10.0, 100.0)):
            for concurrency in (1, 128):
                cells.append({"arm": arm, "context": 16384, "concurrency": concurrency,
                              "output_tok_s": throughput, "ttft_ms": ttft,
                              "latency_ms": ttft + 10, "tpot_ms": 1.0})
        self.assertTrue(campaign.gate_serving(cells, 128, {}, "control")["pass"])
        cells[0]["ttft_ms"] = 11.0
        self.assertFalse(campaign.gate_serving(cells, 128, {}, "control")["pass"])

    def test_vllm_goal_requires_an_actual_latency_win(self):
        cells = []
        for arm, latency in (("candidate", 10.01), ("vllm", 10.0)):
            for concurrency in (1, 128):
                cells.append({"arm": arm, "context": 16384, "concurrency": concurrency,
                              "output_tok_s": 100.0, "ttft_ms": latency,
                              "latency_ms": latency, "tpot_ms": latency})
        self.assertFalse(campaign.gate_serving(cells, 128, {}, "vllm")["pass"])

    def test_kernel_runner_is_control_then_candidate_and_requires_compile_axes(self):
        profile = {
            "arch": "sm90a", "dtype": "bf16", "phase": "prefill",
            "family": "gemm", "rung": 1, "request_topology": "single",
            "concurrency": 1, "packed_topology": "single", "m": 1, "n": 2,
            "k": 3, "occurrences": 1, "dispatch_arm": "Gemm",
        }
        profile["profile_key"] = campaign.canonical_key(profile)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "measure.py"
            order = root / "order"
            helper.write_text(
                "import json,sys\n"
                "arm,key,path=sys.argv[1:]\n"
                "open(path,'a').write(arm+'\\n')\n"
                "r={'profile_key':key,'correct':True,'samples_us':[10]*5}\n"
                "if arm=='candidate': r['compiled_profile']={'object_sha256':'a'*64,'kernel_symbol':'k','threads':128,'warps':4,'registers':64,'smem_bytes':0,'tile':[64,64,32],'stages':2,'tma':True,'swizzle':'128b','spills':0,'segment_mode':'direct'}\n"
                "print(json.dumps(r))\n"
            )
            template = ["python3", str(helper), "{arm}", "{profile_key}", str(order)]
            spec = {"kernel_commands": {"gemm": {"control": template, "candidate": template}}}
            rows = campaign.run_kernels(spec, [profile], str(root), dict(__import__('os').environ))
            self.assertEqual(order.read_text().splitlines(), ["control", "candidate"])
            self.assertEqual(rows[0]["speedup"], 1.0)

    def test_tracked_summary_path_is_rejected(self):
        with self.assertRaisesRegex(campaign.CampaignError, "outside"):
            campaign.external_output(ROOT / "summary.json")


if __name__ == "__main__":
    unittest.main()
