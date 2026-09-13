import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gemma4_h100_kernel_tuner", ROOT / "scripts" / "gemma4_h100_kernel_tuner.py"
)
tuner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tuner)


def resource(mode):
    return {
        "object_sha256": "a" * 64,
        "kernel_symbol": f"kernel_{mode}",
        "threads": 256,
        "warps": 8,
        "registers": 128,
        "smem_bytes": 65536,
        "tile": [128, 128, 64],
        "stages": 3,
        "tma": True,
        "swizzle": "128b",
        "spills": 0,
        "stack_bytes": 0,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
        "segment_mode": mode,
        "sm_count": 132,
        "launch_blocks": 132,
        "blocks_per_sm": 1,
        "cluster": [1, 1, 1],
    }


class Gemma4H100KernelTunerTests(unittest.TestCase):
    def test_correctness_finishes_before_rotated_isolated_trials(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            helper = root / "bench.py"
            order = root / "order"
            helper.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "mode,variant,key,index,arm,compiled,order=sys.argv[1:]\n"
                "with open(order,'a') as f: f.write(f'{mode}:{variant}:{index}:{arm}\\n')\n"
                "base={'profile_key':key,'variant':variant,'compiled_profile':json.loads(compiled)}\n"
                "if mode=='verify':\n"
                " seed=int(index); base.update(seed=seed,correct=True,all_finite=True,"
                "input_sha256=format(seed,'064x'),reference_sha256=format(seed+10,'064x'),"
                "output_sha256=format(seed+20,'064x'))\n"
                "else:\n"
                " trial=int(index); value=8.0 if variant=='split' else 10.0; base.update("
                "trial=trial,isolated=True,correct=True,warmups=10,iterations=50,"
                "arm=arm,counters={'sm_active':0.8 if variant=='split' else 0.5},"
                "cache_state='hot',telemetry={'sm_clock_mhz':1800,'memory_clock_mhz':2600,'power_w':500},"
                "samples_us=[value]*50)\n"
                "print(json.dumps(base))\n"
            )
            helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
            command = [
                "{binary}", "verify", "{variant}", "{profile_key}", "{seed}",
                "verify", "{compiled_profile_json}", str(order),
            ]
            bench_command = [
                "{binary}", "bench", "{variant}", "{profile_key}", "{trial}",
                "{arm}", "{compiled_profile_json}", str(order),
            ]
            variants = []
            for name, mode, reference in (
                ("control", "persistent", True), ("split", "split", False)
            ):
                variants.append(
                    tuner.validate_variant(
                        {
                            "name": name,
                            "reference": reference,
                            **({} if reference else {
                                "hypothesis": "split work fills idle SMs",
                                "lever": "split-k",
                                "predicted_savings_us": 1.5,
                                "expected_counter_changes": {"sm_active": "increase"},
                            }),
                            "binary": str(helper),
                            "verify_command": command,
                            "benchmark_command": bench_command,
                            "compiled_profile": resource(mode),
                        },
                        root,
                    )
                )
            profile = {"profile_key": "prefill/gemm/m128n3840k15360", "occurrences": 1}
            entries = [{"profile": profile, "variants": variants}]
            spec = {"gates": {"minimum_speedup": 1.01}}
            raw = root / "raw"
            raw.mkdir()

            verification = tuner.verify_all(spec, entries, str(root), raw)
            verify_lines = order.read_text().splitlines()
            self.assertEqual(len(verify_lines), 10)
            self.assertEqual(len(verification), 10)
            self.assertTrue(all(row["correct"] for row in verification))
            self.assertTrue(all(line.startswith("verify:") for line in verify_lines))

            timings = tuner.benchmark_all(spec, entries, str(root), raw)
            lines = order.read_text().splitlines()
            self.assertTrue(all(line.startswith("verify:") for line in lines[:10]))
            self.assertEqual(
                lines[10:],
                [
                    "bench:control:0:control_before", "bench:split:0:candidate",
                    "bench:control:0:control_after",
                    "bench:control:1:control_before", "bench:split:1:candidate",
                    "bench:control:1:control_after",
                    "bench:control:2:control_before", "bench:split:2:candidate",
                    "bench:control:2:control_after",
                    "bench:control:3:control_before", "bench:split:3:candidate",
                    "bench:control:3:control_after",
                ],
            )
            winners = tuner.select_winners(spec, entries, timings)
            self.assertEqual(winners[0]["selected_variant"], "split")
            self.assertEqual(winners[0]["selected_execution_mode"], "split")

    def test_benchmark_protocol_is_exact(self):
        profile = {"profile_key": "cell"}
        variant = {
            "name": "candidate",
            "compiled_profile": resource("direct"),
        }
        record = {
            "profile_key": "cell",
            "variant": "candidate",
            "compiled_profile": resource("direct"),
            "trial": 0,
            "arm": "candidate",
            "isolated": True,
            "correct": True,
            "cache_state": "cold",
            "telemetry": {
                "sm_clock_mhz": 1800,
                "memory_clock_mhz": 2600,
                "power_w": 500,
            },
            "counters": {"sm_active": 0.75},
            "warmups": 10,
            "iterations": 50,
            "samples_us": [1.0] * 50,
        }
        self.assertEqual(
            len(
                tuner.validate_benchmark(
                    record, profile, variant, 0, "candidate", "cell"
                )["samples_us"]
            ),
            50,
        )
        record["samples_us"].pop()
        with self.assertRaisesRegex(tuner.TunerError, "exactly 50"):
            tuner.validate_benchmark(record, profile, variant, 0, "candidate", "cell")

    def test_output_must_stay_outside_repository(self):
        with self.assertRaisesRegex(tuner.TunerError, "outside"):
            tuner.external_output(ROOT / "kernel-search.json")
        self.assertEqual(tuner.external_output("/tmp/kernel-search.json"),
                         Path("/tmp/kernel-search.json"))

    def test_selector_is_exact(self):
        profiles = [
            {"profile_key": "a", "phase": "prefill", "family": "gemm", "rung": 128},
            {"profile_key": "b", "phase": "prefill", "family": "gemm", "rung": 256},
            {"profile_key": "c", "phase": "decode", "family": "gemm", "rung": 1},
        ]
        selected = tuner.select_profiles(
            profiles,
            [{"phase": "prefill", "family": "gemm", "rung": [128, 256]}],
        )
        self.assertEqual([profile["profile_key"] for profile in selected], ["a", "b"])

    def test_variant_resources_can_be_exact_profile_specific(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            binary = Path(directory) / "bench"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            base = {
                "reference": False,
                "binary": str(binary),
                "verify_command": ["{binary}"],
                "benchmark_command": ["{binary}"],
                "hypothesis": "reduce idle SMs",
                "lever": "execution mode",
                "predicted_savings_us": 1.0,
                "expected_counter_changes": {"sm_active": "increase"},
            }
            reference = {
                **base,
                "name": "control",
                "reference": True,
                "compiled_profile": resource("persistent"),
            }
            split128 = {
                **base,
                "name": "split",
                "profile_selector": {"rung": 128},
                "compiled_profile": resource("split"),
            }
            direct256 = {
                **base,
                "name": "direct",
                "profile_selector": {"rung": 256},
                "compiled_profile": resource("direct"),
            }
            spec = {"variant_groups": {"prefill_gemm": [reference, split128, direct256]}}
            profile = {"profile_key": "m128", "phase": "prefill", "family": "gemm", "rung": 128}
            variants = tuner.variants_for_profile(spec, profile, directory)
            self.assertEqual([variant["name"] for variant in variants], ["control", "split"])

    def test_candidate_requires_a_hypothesis_and_counter_direction(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            binary = Path(directory) / "bench"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            candidate = {
                "name": "candidate",
                "reference": False,
                "binary": str(binary),
                "verify_command": ["{binary}"],
                "benchmark_command": ["{binary}"],
                "compiled_profile": resource("direct"),
            }
            with self.assertRaisesRegex(tuner.TunerError, "hypothesis"):
                tuner.validate_variant(candidate, directory)

    def test_spilling_candidate_is_rejected_and_rung_rollup_is_weighted(self):
        profile = {
            "profile_key": "cell", "phase": "prefill", "family": "gemm",
            "rung": 4096, "request_topology": "single", "occurrences": 8,
        }
        control = {
            "name": "control", "reference": True,
            "compiled_profile": resource("persistent"), "binary_sha256": "b" * 64,
        }
        dirty = resource("direct")
        dirty["stack_bytes"] = 16
        candidate = {
            "name": "candidate", "reference": False,
            "compiled_profile": dirty, "binary_sha256": "c" * 64,
            "hypothesis": "reduce queue overhead", "lever": "direct launch",
            "predicted_savings_us": 1.5,
            "expected_counter_changes": {"launch_cycles": "decrease"},
        }
        def trial(value, index, arm, counter):
            return {
                "trial": index,
                "arm": arm,
                "samples_us": [value] * tuner.ITERATIONS,
                "cache_state": "hot",
                "telemetry": {
                    "sm_clock_mhz": 1800,
                    "memory_clock_mhz": 2600,
                    "power_w": 500,
                },
                "counters": {"launch_cycles": counter},
            }
        timings = {
            ("cell", "control"): [
                trial(10.0, index, arm, 100.0)
                for index in range(tuner.TRIALS)
                for arm in ("control_before", "control_after")
            ],
            ("cell", "candidate"): [
                trial(8.0, index, "candidate", 80.0)
                for index in range(tuner.TRIALS)
            ],
        }
        result = tuner.select_winners(
            {"gates": {"minimum_speedup": 1.01}},
            [{"profile": profile, "variants": [control, candidate]}],
            timings,
        )[0]
        self.assertEqual(result["promotion_decision"], "keep-reference")
        self.assertIn("local_memory", result["alternatives"][1]["rejection_reasons"])
        rollup = tuner.rung_rollup([result])[0]
        self.assertEqual(rollup["weighted_reference_us"], 80.0)
        self.assertEqual(rollup["weighted_savings_us"], 0.0)

    def test_counter_hypothesis_and_control_noise_are_promotion_gates(self):
        profile = {
            "profile_key": "cell", "phase": "prefill", "family": "gemm",
            "rung": 4096, "request_topology": "single", "occurrences": 4,
        }
        control = {
            "name": "control", "reference": True,
            "compiled_profile": resource("persistent"), "binary_sha256": "b" * 64,
        }
        candidate = {
            "name": "candidate", "reference": False,
            "compiled_profile": resource("direct"), "binary_sha256": "c" * 64,
            "hypothesis": "remove barrier stalls", "lever": "pipeline stages",
            "predicted_savings_us": 2.0,
            "expected_counter_changes": {"barrier_stalls": "decrease"},
        }
        def row(value, trial, arm, counter):
            return {
                "trial": trial, "arm": arm,
                "samples_us": [value] * tuner.ITERATIONS,
                "cache_state": "hot",
                "telemetry": {
                    "sm_clock_mhz": 1800,
                    "memory_clock_mhz": 2600,
                    "power_w": 500,
                },
                "counters": {"barrier_stalls": counter},
            }
        controls = [
            row(value, trial, arm, 100.0)
            for trial in range(tuner.TRIALS)
            for arm, value in (("control_before", 10.0), ("control_after", 10.4))
        ]
        candidates = [
            row(9.0, trial, "candidate", 105.0) for trial in range(tuner.TRIALS)
        ]
        result = tuner.select_winners(
            {"gates": {"minimum_speedup": 1.01}},
            [{"profile": profile, "variants": [control, candidate]}],
            {("cell", "control"): controls, ("cell", "candidate"): candidates},
        )[0]
        alternative = result["alternatives"][1]
        self.assertEqual(result["promotion_decision"], "keep-reference")
        self.assertIn("counter_hypothesis", alternative["rejection_reasons"])
        self.assertAlmostEqual(alternative["noise_floor_us"], 0.4)

        noise_limited = [
            row(10.0, trial, "candidate", 95.0) for trial in range(tuner.TRIALS)
        ]
        result = tuner.select_winners(
            {"gates": {"minimum_speedup": 1.0}},
            [{"profile": profile, "variants": [control, candidate]}],
            {("cell", "control"): controls, ("cell", "candidate"): noise_limited},
        )[0]
        alternative = result["alternatives"][1]
        self.assertIn("noise_floor", alternative["rejection_reasons"])
        self.assertTrue(alternative["counter_hypothesis_matched"])

    def test_resource_profile_checks_h100_occupancy(self):
        bad = resource("direct")
        bad["blocks_per_sm"] = 3
        with self.assertRaisesRegex(tuner.TunerError, "register file"):
            tuner.validate_resource(bad)


if __name__ == "__main__":
    unittest.main()
