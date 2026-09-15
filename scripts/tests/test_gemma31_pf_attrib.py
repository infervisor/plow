import importlib.util
import json
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "gemma31_pf_attrib.py"
SPEC = importlib.util.spec_from_file_location("gemma31_pf_attrib", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Gemma31PrefillAttributionTests(unittest.TestCase):
    def test_selects_requested_program(self):
        path = pathlib.Path(self._testMethodName + ".json")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(
            "tool preface\n"
            + json.dumps(
                {
                    "blob": "model.pkt",
                    "programs": [
                        {"t": 128, "insts": [{"op": 1}]},
                        {"t": 4096, "insts": [{"op": 2}]},
                    ],
                },
                indent=2,
            )
        )

        program = MODULE.load_disasm(path, 1)

        self.assertEqual(program["t"], 4096)
        self.assertEqual(program["insts"][0]["op"], 2)

    def test_rejects_out_of_range_program(self):
        path = pathlib.Path(self._testMethodName + ".json")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(json.dumps({"blob": "model.pkt", "programs": []}, indent=2))

        with self.assertRaises(SystemExit):
            MODULE.load_disasm(path, 0)


if __name__ == "__main__":
    unittest.main()
