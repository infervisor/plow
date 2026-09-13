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
        "segment_mode": mode,
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
                "mode,variant,key,index,compiled,order=sys.argv[1:]\n"
                "with open(order,'a') as f: f.write(f'{mode}:{variant}:{index}\\n')\n"
                "base={'profile_key':key,'variant':variant,'compiled_profile':json.loads(compiled)}\n"
                "if mode=='verify':\n"
                " seed=int(index); base.update(seed=seed,correct=True,all_finite=True,"
                "input_sha256=format(seed,'064x'),reference_sha256=format(seed+10,'064x'))\n"
                "else:\n"
                " trial=int(index); value=8.0 if variant=='split' else 10.0; base.update("
                "trial=trial,isolated=True,correct=True,warmups=10,iterations=50,"
                "samples_us=[value]*50)\n"
                "print(json.dumps(base))\n"
            )
            helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
            command = [
                "{binary}", "verify", "{variant}", "{profile_key}", "{seed}",
                "{compiled_profile_json}", str(order),
            ]
            bench_command = [
                "{binary}", "bench", "{variant}", "{profile_key}", "{trial}",
                "{compiled_profile_json}", str(order),
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
                            "binary": str(helper),
                            "verify_command": command,
                            "benchmark_command": bench_command,
                            "compiled_profile": resource(mode),
                        },
                        root,
                    )
                )
            profile = {"profile_key": "prefill/gemm/m128n3840k15360"}
            entries = [{"profile": profile, "variants": variants}]
            spec = {"gates": {"minimum_speedup": 1.01}}
            raw = root / "raw"
            raw.mkdir()

            tuner.verify_all(spec, entries, str(root), raw)
            verify_lines = order.read_text().splitlines()
            self.assertEqual(len(verify_lines), 10)
            self.assertTrue(all(line.startswith("verify:") for line in verify_lines))

            timings = tuner.benchmark_all(spec, entries, str(root), raw)
            lines = order.read_text().splitlines()
            self.assertTrue(all(line.startswith("verify:") for line in lines[:10]))
            self.assertEqual(
                lines[10:],
                [
                    "bench:control:0", "bench:split:0",
                    "bench:split:1", "bench:control:1",
                    "bench:control:2", "bench:split:2",
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
            "isolated": True,
            "correct": True,
            "warmups": 10,
            "iterations": 50,
            "samples_us": [1.0] * 50,
        }
        self.assertEqual(len(tuner.validate_benchmark(record, profile, variant, 0, "cell")), 50)
        record["samples_us"].pop()
        with self.assertRaisesRegex(tuner.TunerError, "exactly 50"):
            tuner.validate_benchmark(record, profile, variant, 0, "cell")

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


if __name__ == "__main__":
    unittest.main()
