import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SileroExportCliTests(unittest.TestCase):
    def test_explicit_checkpoint_is_documented(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("silero_export.py")), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--jit", result.stdout)

    def test_missing_explicit_checkpoint_does_not_fall_back_to_package(self):
        with tempfile.TemporaryDirectory() as out:
            missing = str(Path(out) / "missing.jit")
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("silero_export.py")),
                 out, "--jit", missing],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing.jit", result.stderr)
            self.assertNotIn("No module named 'silero_vad'", result.stderr)


if __name__ == "__main__":
    unittest.main()
