import importlib.util
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gemma4_ladder_campaign", ROOT / "scripts" / "gemma4_ladder_campaign.py"
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
PREFIX_SPEC = importlib.util.spec_from_file_location(
    "gemma4_prefix_copack_gate", ROOT / "scripts" / "gemma4_prefix_copack_gate.py"
)
prefix_gate = importlib.util.module_from_spec(PREFIX_SPEC)
PREFIX_SPEC.loader.exec_module(prefix_gate)


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


def serving_metrics(value):
    return {
        f"{family}_{stat}_ms": value
        for family in ("ttft", "tpot", "itl", "e2el")
        for stat in ("mean", "median", "p99")
    }


def serving_cell(arm, concurrency, latency=10.0, throughput=100.0):
    return {
        "arm": arm,
        "context": 16384,
        "concurrency": concurrency,
        "output_tok_s": throughput,
        "request_s": throughput / 128,
        "prefill_bucket": 1024,
        "prefill_chunks": 16,
        "decode_bucket": concurrency,
        **serving_metrics(latency),
    }


class Gemma4LadderCampaignTests(unittest.TestCase):
    def test_gfx942_plan_maps_4k_8k_to_repeated_1024_prefill(self):
        audit = audit_fixture()
        audit["programs"] = [
            program for program in audit["programs"]
            if (program["phase"] == "prefill" and program["rows"] in (128, 512, 1024))
            or program["phase"] == "decode"
        ]
        decode_32 = next(
            program for program in audit["programs"]
            if program["phase"] == "decode" and program["rows"] == 32
        )
        for rung in (64, 128):
            program = copy.deepcopy(decode_32)
            program["rows"] = rung
            for case in program["kernel_cases"]:
                case["i"][0] = rung
            audit["programs"].append(program)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(audit))
            spec = {
                "audit": str(path), "arch": "gfx942", "dtype": "w8a8",
                "packed_topology": "unified-token-batch", "max_concurrency": 128,
                "production_features": {
                    "prefix_cache": True, "continuous_batching": True,
                    "unified_token_batch": True, "suffix_copack": True,
                },
            }
            campaign.validate_spec(spec)
            plan = campaign.campaign_plan(spec)
        self.assertEqual(plan["axes"]["prefill"], [128, 512, 1024])
        self.assertEqual(plan["axes"]["decode"], [1, 2, 4, 8, 16, 32, 64, 128])
        self.assertEqual(plan["axes"]["concurrency"], [1, 2, 4, 8, 16, 32, 64, 128])
        self.assertEqual(plan["schema_version"], 5)
        self.assertEqual(len(plan["serving_rungs"]), 40)
        self.assertEqual(plan["live_kv_buckets"], [128, 1024, 4096, 8192, 16384])
        full_attention = {
            profile["kv_length"]: profile["live_kv_bucket"]
            for profile in plan["profiles"]
            if profile["family"] == "attention"
            and profile["rung"] == 128
            and profile["request_topology"] == "single"
            and profile["window"] == 0
        }
        self.assertEqual(
            {length: full_attention[length] for length in (4095, 4096, 4097)},
            {4095: 4096, 4096: 4096, 4097: 8192},
        )
        rung = next(row for row in plan["serving_rungs"] if row["rung_key"] == "ctx8192/c128")
        self.assertEqual((rung["prefill_bucket"], rung["prefill_chunks"]), (1024, 8))
        self.assertEqual(rung["decode_bucket"], 128)
        self.assertEqual((rung["order"], rung["decode_steps"]), (32, 127))
        gemm = next(
            profile for profile in plan["profiles"]
            if profile["phase"] == "prefill" and profile["family"] == "gemm"
            and profile["rung"] == 1024 and profile["concurrency"] == 16
            and profile["request_topology"] == "packed_homogeneous"
            and (profile["n"], profile["k"]) == (15360, 3840)
        )
        self.assertEqual(
            gemm["serving_impact"],
            [
                {"rung_key": "ctx1024/c16", "executions": 1},
                {"rung_key": "ctx4096/c16", "executions": 4},
                {"rung_key": "ctx8192/c16", "executions": 8},
                {"rung_key": "ctx16384/c16", "executions": 16},
            ],
        )
        self.assertEqual(gemm["campaign_occurrences"], 96 * 29)
        self.assertEqual(
            [profile["optimization_rank"] for profile in plan["profiles"]],
            list(range(1, len(plan["profiles"]) + 1)),
        )
        self.assertEqual(
            [profile["campaign_occurrences"] for profile in plan["profiles"]],
            sorted(
                (profile["campaign_occurrences"] for profile in plan["profiles"]),
                reverse=True,
            ),
        )

    def test_focused_prefill_topology_has_only_selected_concurrency(self):
        topologies = campaign._prefill_topologies(1024, "unified", (16,))
        self.assertEqual(len(topologies), 2)
        self.assertEqual({row["concurrency"] for row in topologies}, {16})
        self.assertEqual(
            {row["request_topology"] for row in topologies},
            {"packed_homogeneous", "packed_ragged"},
        )

    def test_plan_rejects_decode_ladder_that_cannot_cover_concurrency(self):
        audit = audit_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(audit))
            spec = {
                "audit": str(path), "arch": "gfx942", "dtype": "w8a8",
                "packed_topology": "unified-token-batch", "max_concurrency": 128,
            }
            with self.assertRaisesRegex(campaign.CampaignError, "decode ladder ends at 32"):
                campaign.campaign_plan(spec)

    def test_inventory_has_every_shape_mode_and_stable_keys(self):
        profiles, rungs, decode_rungs = campaign.audit_inventory(
            audit_fixture(), "sm90a", "bf16", "packed-varlen", 128
        )
        self.assertEqual([x["rung"] for x in rungs], list(campaign.RUNGS))
        self.assertEqual([x["rung"] for x in decode_rungs], list(campaign.DECODE_RUNGS))
        self.assertEqual(len(profiles), 3612)
        self.assertEqual(sum(x["phase"] == "prefill" for x in profiles), 2753)
        self.assertEqual(sum(x["phase"] == "decode" for x in profiles), 859)
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

    def test_full_logit_plan_marks_every_rung_pending(self):
        cells = campaign.full_logit_plan()
        self.assertEqual(len(cells), 20)
        self.assertEqual(
            [(x["phase"], x["rung"]) for x in cells],
            [("prefill", rung) for rung in campaign.RUNGS]
            + [("decode", rung) for rung in campaign.DECODE_RUNGS],
        )
        self.assertEqual({x["status"] for x in cells}, {"pending"})
        self.assertEqual(len({x["profile_key"] for x in cells}), len(cells))

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

    def test_inventory_accepts_gfx942_split_qkv_glu_and_prefill_gemv_head(self):
        audit = audit_fixture()
        prefill = next(
            program for program in audit["programs"]
            if program["phase"] == "prefill" and program["rows"] == 1024
        )
        head = next(case for case in prefill["kernel_cases"] if case["i"][1] == 262144)
        head["op"] = "Gemv"
        decode = next(
            program for program in audit["programs"]
            if program["phase"] == "decode" and program["rows"] == 8
        )
        decode["kernel_cases"] = [
            case for case in decode["kernel_cases"]
            if "Qkv" not in case["op"] and "Glu" not in case["op"]
        ]
        decode["kernel_cases"].extend([
            {"op": "GemvFp8", "i": [8, 4096, 3840, 0, 0, 0, 0, 0], "pcs": list(range(40))},
            {"op": "GemvFp8", "i": [8, 2048, 3840, 0, 0, 0, 0, 0], "pcs": list(range(80))},
            {"op": "GemvFp8", "i": [8, 15360, 3840, 0, 0, 0, 0, 0], "pcs": list(range(96))},
        ])
        for case in decode["kernel_cases"]:
            if case["op"].startswith("FlashDecode"):
                if case["i"][4]:
                    case["i"][3], case["i"][5], case["i"][7] = 2048, 16, 2047
                else:
                    case["i"][3], case["i"][5] = 16512, 38
        profiles, _, _ = campaign.audit_inventory(
            audit, "gfx942", "w8a8", "unified", 128,
            (1024,), (8,), (128, 1024, 4096, 8192, 16384),
        )
        self.assertTrue(any(
            profile["phase"] == "prefill" and profile["dispatch_arm"] == "Gemv"
            and profile["n"] == 262144 for profile in profiles
        ))
        self.assertTrue(any(
            profile["phase"] == "decode" and profile["dispatch_arm"] == "GemvFp8"
            and profile["n"] == 15360 for profile in profiles
        ))

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

    def test_kernel_speedup_is_reduced_over_the_whole_rung_family(self):
        rows = []
        for kv_length, control, candidate in ((1024, 100.0, 90.0), (16384, 10.0, 10.0)):
            rows.append({
                "phase": "decode", "rung": 16, "request_topology": "decode_homogeneous",
                "family": "attention", "kv_length": kv_length, "occurrences": 1,
                "profile_key": f"kv{kv_length}", "control_us": control,
                "candidate_us": candidate, "weighted_control_us": control,
                "weighted_candidate_us": candidate,
            })
        gate, _ = campaign.gate_kernels(rows, {
            "minimum_weighted_speedup": 1.01, "kernel_regression_tolerance": 1.02,
        })
        self.assertTrue(gate["pass"])
        self.assertEqual(len(gate["rungs"]), 1)

    def test_serving_gate_separates_latency_and_max_throughput(self):
        cells = []
        for arm, ttft, throughput in (("candidate", 9.0, 101.0), ("control", 10.0, 100.0)):
            for concurrency in (1, 128):
                cells.append(serving_cell(arm, concurrency, ttft, throughput))
        self.assertTrue(campaign.gate_serving(cells, 128, {}, "control")["pass"])
        cells[0]["ttft_mean_ms"] = 11.0
        self.assertFalse(campaign.gate_serving(cells, 128, {}, "control")["pass"])

    def test_vllm_goal_requires_an_actual_latency_win(self):
        cells = []
        for arm, latency in (("candidate", 10.01), ("vllm", 10.0)):
            for concurrency in (1, 128):
                cells.append(serving_cell(arm, concurrency, latency, 100.0))
        self.assertFalse(campaign.gate_serving(cells, 128, {}, "vllm")["pass"])

    def test_throughput_win_cannot_hide_concurrent_latency_regression(self):
        cells = [serving_cell(
            arm,
            concurrency,
            20.0 if arm == "candidate" and concurrency == 128 else 10.0,
            110.0 if arm == "candidate" else 100.0,
        ) for arm in ("candidate", "control") for concurrency in (1, 128)]
        gate = campaign.gate_serving(cells, 128, {}, "control")
        self.assertFalse(gate["pass"])
        self.assertIn("C128 ttft_mean_ms", gate["failures"][0])

    def test_serving_comparison_requires_matching_workloads(self):
        records = [
            {"input": 16384, "concurrency": concurrency, "output": 128,
             "output_tok_s": 100.0, "request_s": 100.0 / 128,
             "metrics": {
                 "ttft_ms": {"mean": 10.0, "p50": 10.0, "p99": 10.0},
                 "tpot_ms": {"mean": 1.0, "p50": 1.0, "p99": 1.0},
                 "itl_ms": {"mean": 1.0, "p50": 1.0, "p99": 1.0},
                 "e2el_ms": {"mean": 30.0, "p50": 30.0, "p99": 30.0},
             }}
            for concurrency in (1, 128)
        ]
        control = campaign.reduce_serving(records, "control", 128, [16384])
        for axis, value in (("output", 256), ("requested_cached_prefix_tokens", 8192),
                            ("repeats", 2)):
            candidate = [dict(cell, arm="candidate") for cell in control]
            candidate[0][axis] = value
            with self.assertRaisesRegex(campaign.CampaignError, axis):
                campaign.gate_serving(control + candidate, 128, {}, "control")
        with self.assertRaisesRegex(campaign.CampaignError, "mixed output"):
            campaign.reduce_serving(records + [dict(records[0], output=256)],
                                    "control", 128, [16384])

    def test_serving_reduction_gates_every_metric_at_an_intermediate_rung(self):
        records = []
        for concurrency in (1, 16, 128):
            for repeat in range(3):
                records.append({
                    "input": 8192, "concurrency": concurrency, "output": 128,
                    "output_tok_s": 100 + repeat, "request_s": (100 + repeat) / 128,
                    "metrics": {
                        family: {"mean": 10 + repeat, "p50": 9 + repeat, "p99": 12 + repeat}
                        for family in ("ttft_ms", "tpot_ms", "itl_ms", "e2el_ms")
                    },
                })
        control = campaign.reduce_serving(records, "control", (1, 16, 128), [8192], (128, 512, 1024))
        candidate = [dict(cell, arm="candidate") for cell in control]
        for cell in candidate:
            cell["request_s"] *= 1.01
            cell["output_tok_s"] *= 1.01
            for metric in campaign.SERVING_LOWER_IS_BETTER:
                cell[metric] *= 0.99
        candidate[1]["itl_p99_ms"] = control[1]["itl_p99_ms"] * 1.01
        gate = campaign.gate_serving(
            control + candidate, (1, 16, 128), {"serving_regression_tolerance": 1.0}, "control"
        )
        self.assertFalse(gate["pass"])
        self.assertTrue(any("8192 C16 itl_p99_ms" in failure for failure in gate["failures"]))
        self.assertEqual((control[0]["prefill_bucket"], control[0]["prefill_chunks"]), (1024, 8))

    def test_serving_runs_are_paired_by_concurrency_rung(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            helper = root / "serve.py"
            order = root / "order"
            helper.write_text(
                "import json,sys\n"
                "arm,out,order,conc,*contexts=sys.argv[1:]\n"
                "open(order,'a').write(f'{conc}:{arm}\\n')\n"
                "m={x:{'mean':10,'p50':10,'p99':10} for x in ('ttft_ms','tpot_ms','itl_ms','e2el_ms')}\n"
                "req=[{'prompt_sha256':'a'*64,'text':'same','ttft_ms':10,'tpot_ms':1,'latency_ms':20} for _ in range(int(conc))]\n"
                "rows=[{'input':int(ctx),'concurrency':int(conc),'output':128,'repeat':0,'output_tok_s':100,'request_s':1,'metrics':m,'requests':req} for ctx in contexts]\n"
                "open(out,'w').write(''.join(json.dumps(x)+'\\n' for x in rows))\n"
            )
            template = [sys.executable, str(helper), "{arm}", "{output}", str(order),
                        "{concurrency}", "{contexts}"]
            cells = campaign.run_serving(
                {"max_concurrency": 128, "serving_commands": {
                    arm: template for arm in ("control", "candidate", "vllm")
                }},
                str(root), dict(os.environ), root, (128, 1024), (1, 2),
                (128, 512, 1024), (1, 2),
            )
            observed_order = order.read_text().splitlines()
        self.assertEqual(observed_order, [
            "1:control", "1:candidate", "1:control2", "1:candidate2", "1:vllm",
            "2:control", "2:candidate", "2:control2", "2:candidate2", "2:vllm",
        ])
        self.assertEqual(len(cells), 20)

    def test_t4_gate_requires_replicated_arms_and_negative_bootstrap_upper(self):
        cells = []
        for concurrency, count in ((1, 16), (16, 80)):
            for arm in ("control", "candidate", "control2", "candidate2"):
                candidate = arm.startswith("candidate")
                cell = serving_cell(arm, concurrency, 9.0 if candidate else 10.0, 101.0)
                cell.update({
                    "output": 128, "requested_cached_prefix_tokens": 0,
                    "repeats": count // concurrency if count >= concurrency else 1,
                    "prompt_set_sha256": "a" * 64,
                    "completion_set_sha256": "b" * 64,
                    "request_metrics": {
                        "ttft_ms": [9.0 if candidate else 10.0] * count,
                        "tpot_ms": [0.9 if candidate else 1.0] * count,
                        "e2el_ms": [20.0] * count,
                    },
                })
                cells.append(cell)
        gate = campaign.gate_t4_serving(cells, (1, 16), {"bootstrap_samples": 200})
        self.assertTrue(gate["pass"])
        cells[1]["request_metrics"]["ttft_ms"] = [10.0] * 16
        cells[3]["request_metrics"]["ttft_ms"] = [10.0] * 16
        gate = campaign.gate_t4_serving(cells, (1, 16), {"bootstrap_samples": 200})
        self.assertFalse(gate["pass"])
        self.assertTrue(any("ttft_ms CI upper" in failure for failure in gate["failures"]))

    def test_production_gate_requires_exact_prefix_suffix_copack(self):
        plan = {
            "packet_sha256": "a" * 64,
            "production_features": {
                "prefix_cache": True, "continuous_batching": True,
                "unified_token_batch": True, "suffix_copack": True,
            },
        }
        record = {
            "schema": "plowrt.production-gate.v1", "packet_sha256": "a" * 64,
            "features": plan["production_features"], "correct": True,
            "completed": 2, "cached_prefix_rows_per_request": 1024,
            "suffix_rows_per_request": 512, "copacked_rows": 1024,
            "decode_rows": 0, "prefill_requests": 2, "restore_calls": 2,
        }
        self.assertTrue(campaign.validate_production_gate_record(record, plan)["pass"])
        record["restore_calls"] = 1
        with self.assertRaisesRegex(campaign.CampaignError, "restore_calls=2"):
            campaign.validate_production_gate_record(record, plan)

    def test_prefix_copack_probe_requires_runtime_route_evidence(self):
        rows = prefix_gate.prompt_rows()
        report = {
            "schema": "plowrt.bench.v1", "vendor": "Some(Amd)", "num_gpus": 1,
            "concurrency": 2, "warmup_requests": 2, "requests": 2,
            "completed": 2, "failed": 0,
            "scheduler": {"rejected": 0, "admit_shed": 0},
            "engine": {"batch_capacity": 128},
            "token_audit": {
                "prompt_token_ids": list(rows[2:]),
                "output_token_ids": [[1] * 8, [2] * 8],
            },
        }
        log = (
            'AMD token batch rows=1024 decode=0 prefill=2 completed=2 fires=true\n'
            'PFX prefix restore calls=2\n'
        )
        prefix_gate.validate(report, log, rows)
        ansi_log = (
            '\x1b[2mAMD token batch\x1b[0m '
            '\x1b[3mrows\x1b[0m=\x1b[0m1024 '
            '\x1b[3mdecode\x1b[0m=\x1b[0m0 '
            '\x1b[3mprefill\x1b[0m=\x1b[0m2 '
            '\x1b[3mcompleted\x1b[0m=\x1b[0m2 '
            '\x1b[3mfires\x1b[0m=\x1b[0mtrue\n'
            '\x1b[32mINFO\x1b[0m PFX '
            '\x1b[3mphase\x1b[0m=\x1b[0m"prefix restore  (dtod, per rank)" '
            '\x1b[3mcalls\x1b[0m=\x1b[0m2\n'
        )
        prefix_gate.validate(report, ansi_log, rows)
        with self.assertRaisesRegex(ValueError, "VMM KV route"):
            prefix_gate.validate(report, log, rows, require_vmm=True)
        prefix_gate.validate(
            report, 'AMD engine ready vmm=true\n' + log, rows, require_vmm=True
        )
        with self.assertRaisesRegex(ValueError, "two-suffix co-pack"):
            prefix_gate.validate(report, log.replace("rows=1024", "rows=512"), rows)

    def test_prefix_copack_probe_passes_split_checkpoint_before_bench(self):
        command = prefix_gate.bench_command(
            "/tmp/plowrt", "/tmp/assets", "/tmp/rows.csv",
            "/tmp/bf16", "/tmp/fp8",
        )
        self.assertEqual(command[:5], [
            "/tmp/plowrt", "--rt-checkpoint", "/tmp/bf16", "--fp8-dir", "/tmp/fp8",
        ])
        self.assertEqual(command[5], "bench")
        self.assertEqual(command.count("--prompt-rows"), 1)

    def test_prefix_copack_probe_persists_failure_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate.jsonl"
            prefix_gate.write_diagnostics(output, '{"completed":2}\n', "scheduler trace\n")
            self.assertEqual(
                output.with_suffix(".jsonl.stdout").read_text(), '{"completed":2}\n'
            )
            self.assertEqual(
                output.with_suffix(".jsonl.stderr").read_text(), "scheduler trace\n"
            )

    def test_kernel_runner_brackets_candidate_with_control_anchors(self):
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
            self.assertEqual(order.read_text().splitlines(), [
                "control_before", "candidate", "control_after",
            ])
            self.assertEqual(rows[0]["speedup"], 1.0)
            self.assertEqual(rows[0]["control_noise_floor_us"], 0.0)
            self.assertFalse(rows[0]["above_control_noise"])

    def test_kernel_gate_rejects_a_win_inside_anchor_noise(self):
        row = {
            "phase": "prefill", "rung": 1024,
            "request_topology": "packed_homogeneous", "family": "gemm",
            "profile_key": "inside-noise", "occurrences": 10,
            "control_us": 101.0, "candidate_us": 100.0,
            "weighted_control_us": 1010.0,
            "weighted_candidate_us": 1000.0,
            "weighted_noise_floor_us": 20.0,
        }
        gate, _ = campaign.gate_kernels([row], {
            "minimum_weighted_speedup": 1.0,
            "kernel_regression_tolerance": 1.02,
        })
        self.assertFalse(gate["pass"])
        self.assertIn("noise floor", gate["failures"][0])

    def test_gfx942_kernel_evidence_requires_native_resource_axes(self):
        profile = {
            "arch": "gfx942", "dtype": "w8a8", "phase": "prefill",
            "family": "gemm", "rung": 1024, "request_topology": "single",
            "concurrency": 1, "packed_topology": "single", "m": 1024,
            "n": 15360, "k": 3840, "dispatch_arm": "GemmWideFp8",
        }
        profile["profile_key"] = campaign.canonical_key(profile)
        record = {
            "profile_key": profile["profile_key"], "correct": True,
            "samples_us": [100.0] * 5,
            "compiled_profile": {
                "object_sha256": "a" * 64, "kernel_symbol": "gemm_wide",
                "threads": 256, "wavefronts": 4, "vgprs": 180, "agprs": 0,
                "lds_bytes": 32768, "tile": [128, 128, 128], "stages": 3,
                "memory_path": "global-to-lds", "mfma": "intrawave-v3",
                "spills": 0, "segment_mode": "segmented",
            },
        }
        normalized = campaign.validate_kernel_record(record, profile, "candidate", 5)
        self.assertEqual(normalized["compiled_profile"]["vgprs"], 180)
        del record["compiled_profile"]["agprs"]
        with self.assertRaisesRegex(campaign.CampaignError, "incomplete compiled profile"):
            campaign.validate_kernel_record(record, profile, "candidate", 5)

    def test_attention_evidence_binds_live_kv_route_and_launch_envelope(self):
        profile = {
            "arch": "gfx942", "dtype": "w8a8", "kv_dtype": "bf16_kv",
            "phase": "prefill", "family": "attention", "rung": 1024,
            "concurrency": 1, "request_topology": "single",
            "packed_topology": "single", "history_layout": "homogeneous",
            "live_kv_bucket": 4096,
            "query_rows": 1024, "kv_length": 4096, "q_heads": 16,
            "kv_heads": 1, "head_dim": 512, "window": 0, "splits": 1,
        }
        profile["profile_key"] = campaign.canonical_key(profile)
        record = {
            "profile_key": profile["profile_key"], "correct": True,
            "samples_us": [100.0] * 5,
            "compiled_profile": {
                "object_sha256": "a" * 64, "program_digest": "b" * 64,
                "kernel_symbol": "flash_prefill_hd512", "threads": 256,
                "wavefronts": 4, "vgprs": 180, "agprs": 0,
                "lds_bytes": 32768, "tile": [64, 16, 512], "stages": 2,
                "memory_path": "global-to-lds", "mfma": "intrawave-v3",
                "bq": 64, "bkv": 16, "nsplit": 1, "occupancy": 0.25,
                "spills": 0, "segment_mode": "direct",
            },
        }
        normalized = campaign.validate_kernel_record(record, profile, "candidate", 5)
        self.assertEqual(normalized["compiled_profile"]["bkv"], 16)
        del record["compiled_profile"]["occupancy"]
        with self.assertRaisesRegex(campaign.CampaignError, "incomplete compiled profile"):
            campaign.validate_kernel_record(record, profile, "candidate", 5)

    def test_full_logit_runner_requires_isolated_exact_full_vocab_cells(self):
        packet = "a" * 64
        cells = campaign.full_logit_plan()
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            helper = root / "full_logits.py"
            order = root / "order"
            helper.write_text(
                "import json,sys\n"
                "phase,packet,output,order,*rungs=sys.argv[1:]\n"
                "open(order,'a').write(phase+'\\n')\n"
                "records=[{'profile_key':f'full-logit/{phase}/r{rung}',"
                "'phase':phase,'rung':int(rung),'packet_sha256':packet,"
                "'reference_sha256':'b'*64,'isolated':True,"
                "'snapshots':1 if phase=='prefill' else int(rung),"
                "'vocab':262144,'correct':True,'bitwise_equal':True,"
                "'all_finite':True} for rung in rungs]\n"
                "open(output,'w').write(''.join(json.dumps(x)+'\\n' for x in records))\n"
            )
            template = [
                sys.executable, str(helper), "{phase}", "{packet_sha256}",
                "{output}", str(order), "{rungs}",
            ]
            result = campaign.run_full_logits(
                {"full_logit_commands": {"prefill": template, "decode": template}},
                cells,
                packet,
                str(ROOT),
                dict(os.environ),
                root,
            )
            self.assertTrue(result["pass"])
            self.assertEqual(len(result["cells"]), 20)
            self.assertEqual(order.read_text().splitlines(), ["prefill", "decode"])
            self.assertTrue(all(x["status"] == "pass" for x in result["cells"]))

    def test_full_logit_record_rejects_nonisolated_or_partial_vocab(self):
        cell = campaign.full_logit_plan()[0]
        record = {
            **cell,
            "packet_sha256": "a" * 64,
            "reference_sha256": "b" * 64,
            "isolated": False,
            "snapshots": 1,
            "vocab": campaign.GEMMA4_VOCAB,
            "correct": True,
            "bitwise_equal": True,
            "all_finite": True,
        }
        with self.assertRaisesRegex(campaign.CampaignError, "not isolated"):
            campaign.validate_full_logit_record(record, cell, "a" * 64)
        record["isolated"] = True
        record["vocab"] -= 1
        with self.assertRaisesRegex(campaign.CampaignError, "full vocabulary"):
            campaign.validate_full_logit_record(record, cell, "a" * 64)

    def test_full_logit_record_accepts_only_control_floor_bounded_drift(self):
        cell = campaign.full_logit_plan()[0]
        record = {
            **cell,
            "packet_sha256": "a" * 64,
            "reference_sha256": "b" * 64,
            "isolated": True,
            "snapshots": 1,
            "vocab": campaign.GEMMA4_VOCAB,
            "correct": True,
            "bitwise_equal": False,
            "all_finite": True,
            "max_abs_error": 0.02,
            "control_max_abs_floor": 0.03,
            "rel_l2_error": 0.001,
            "control_rel_l2_floor": 0.002,
        }
        result = campaign.validate_full_logit_record(record, cell, "a" * 64)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["floor_bounded"])
        record["rel_l2_error"] = 0.003
        result = campaign.validate_full_logit_record(record, cell, "a" * 64)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["floor_bounded"])

    def test_tracked_summary_path_is_rejected(self):
        with self.assertRaisesRegex(campaign.CampaignError, "outside"):
            campaign.external_output(ROOT / "summary.json")


if __name__ == "__main__":
    unittest.main()
