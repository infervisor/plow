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

    def test_real_cmake_rows_carry_resource_contracts(self):
        _, rows = MOD.parse_cmake(MOD.CMAKE)
        self.assertGreater(len(rows), 10)
        self.assertTrue(all(max_regs > 0 and min_occ > 0 for *_, max_regs, min_occ in rows))


if __name__ == "__main__":
    unittest.main()
