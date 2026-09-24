import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TraceOpcodeTests(unittest.TestCase):
    def test_csv_enum_and_define_names(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            trace = path / "trace.bin"
            trace.write_bytes(struct.pack("<IIIHHQQQ", 0, 0, 0, 185, 0, 100, 110, 120))
            csv = path / "opcodes.csv"
            csv.write_text("GEMM_FP8_BLOCK128,185\n")
            header = path / "defines.h"
            header.write_text("#define PLOW_DOP_GEMM_FP8_BLOCK128 185\n")
            for names in (csv, header, root / "runtime/common/dev_isa.h"):
                with self.subTest(names=names):
                    result = subprocess.run([sys.executable, str(root / "scripts/glm53_trace_attrib.py"),
                        str(trace), "--opcodes", str(names), "--top-packets", "1"],
                        text=True, capture_output=True, check=True)
                    self.assertIn("GEMM_FP8_BLOCK128", result.stdout)
                    self.assertIn("not wall-time attribution", result.stdout)
                    self.assertIn("     0 GEMM_FP8_BLOCK128", result.stdout)


if __name__ == "__main__":
    unittest.main()
