import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gfx950_objects", ROOT / "scripts" / "gfx950_objects.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class ObjectRecipeTests(unittest.TestCase):
    def test_selection_does_not_depend_on_cmake_row_order(self):
        axes = {
            "pf": ["-DPLOW_BUCKET_DECODE=0"],
            "dec": ["-DPLOW_BUCKET_DECODE=1"],
            "kda": ["-DPLOW_K3=1"],
        }
        rows = [
            ("z_generic", "sym", "dec", "missing", 256, 2),
            ("a_reachable", "sym", "dec", "kda", 256, 2),
            ("prefill", "sym", "pf", "missing", 256, 2),
        ]
        req = ["PLOW_BUCKET_DECODE=0", "PLOW_K3=1"]
        want = ["prefill", "a_reachable"]
        self.assertEqual(MOD.stems_from_requires(axes, rows, req, []), want)
        self.assertEqual(MOD.stems_from_requires(axes, list(reversed(rows)), req, []), want)

    def test_resource_cliff_refuses_weak_row(self):
        contract = {
            "max_total_registers": 256,
            "min_occupancy_waves_per_simd": 2,
        }
        self.assertTrue(MOD.resource_row_accepts(256, 2, contract))
        self.assertFalse(MOD.resource_row_accepts(257, 2, contract))
        self.assertFalse(MOD.resource_row_accepts(256, 1, contract))

    def test_resource_certificate_refuses_phase_regressions(self):
        contract = {
            "max_total_registers": 256,
            "min_occupancy_waves_per_simd": 2,
            "wavefront_size": 64,
        }
        baseline = {
            "arch": "gfx950",
            "kernel": "plow_interp_gfx950",
            "vgpr_spill": 8,
            "sgpr_spill": 78,
            "private_segment_bytes": 1348,
        }
        candidate = {
            "arch": "gfx950",
            "kernel": "plow_interp_gfx950",
            "total_registers": 256,
            "occupancy_waves_per_simd": 2,
            "wavefront_size": 64,
            "vgpr_spill": 2,
            "sgpr_spill": 78,
            "private_segment_bytes": 752,
        }
        self.assertEqual(
            MOD.resource_certificate_violations(candidate, baseline, contract), []
        )
        candidate["private_segment_bytes"] = 1352
        candidate["sgpr_spill"] = 80
        self.assertEqual(
            MOD.resource_certificate_violations(candidate, baseline, contract),
            ["sgpr_spill 80 > 78", "private_segment_bytes 1352 > 1348"],
        )

    def test_resource_certificate_refuses_wrong_phase_or_missing_facts(self):
        contract = {"max_total_registers": 256}
        baseline = {
            "arch": "gfx950",
            "kernel": "plow_interp_gfx950",
            "vgpr_spill": 0,
            "sgpr_spill": 0,
            "private_segment_bytes": 0,
        }
        candidate = {
            "arch": "gfx950",
            "kernel": "plow_interp_dec_gfx950",
            "total_registers": 248,
            "occupancy_waves_per_simd": 2,
            "wavefront_size": 64,
            "vgpr_spill": 0,
            "sgpr_spill": 0,
        }
        self.assertEqual(
            MOD.resource_certificate_violations(candidate, baseline, contract),
            ["candidate certificate missing private_segment_bytes"],
        )
        candidate["private_segment_bytes"] = 0
        self.assertEqual(
            MOD.resource_certificate_violations(candidate, baseline, contract),
            ["kernel plow_interp_dec_gfx950 != baseline plow_interp_gfx950"],
        )

    def test_real_cmake_rows_carry_resource_contracts(self):
        _, rows = MOD.parse_cmake(MOD.CMAKE)
        self.assertGreater(len(rows), 10)
        self.assertTrue(all(max_regs > 0 and min_occ > 0 for *_, max_regs, min_occ in rows))


if __name__ == "__main__":
    unittest.main()
