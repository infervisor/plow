import os
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform == "linux", "Linux process names")
class QuietMeasurementTests(unittest.TestCase):
    def test_finished_compiler_does_not_block_measurement(self):
        here = Path(__file__).resolve().parent
        child = os.fork()
        if child == 0:
            ctypes.CDLL(None).prctl(15, b"nvcc", 0, 0, 0)
            os._exit(0)
        try:
            deadline = time.monotonic() + 5
            while Path(f"/proc/{child}/stat").read_text().split()[2] != "Z":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
                result = subprocess.run(
                    [str(here / "quietx.sh"), str(Path(tmp) / "quiet.lock"), "printf", "measured"],
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual((result.returncode, result.stdout), (0, "measured"))
        finally:
            os.waitpid(child, 0)

    def test_compilers_must_finish_before_measurement(self):
        here = Path(__file__).resolve().parent
        compiler = (
            "import ctypes,pathlib,sys,time; "
            "assert ctypes.CDLL(None).prctl(15,b'nvcc',0,0,0)==0; "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(1)"
        )
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            lock = str(Path(tmp) / "quiet.lock")
            for cooperative in [False, True]:
                ready = Path(tmp) / f"ready-{cooperative}"
                command = [sys.executable, "-c", compiler, str(ready)]
                if cooperative:
                    command = [str(here / "quiets.sh"), lock, *command]
                child = subprocess.Popen(command)
                try:
                    deadline = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(ready.exists())
                    result = subprocess.run(
                        [str(here / "quietx.sh"), lock, "printf", "measured"],
                        capture_output=True, text=True, timeout=5,
                    )
                    self.assertEqual(result.returncode, 0 if cooperative else 75)
                    self.assertEqual(result.stdout, "measured" if cooperative else "")
                    if not cooperative:
                        self.assertIn("nvcc", result.stderr)
                    check = subprocess.run(
                        [str(here / "quietx.sh"), "--check-compilers"],
                        capture_output=True, text=True, timeout=5,
                    )
                    self.assertEqual(check.returncode, 0 if cooperative else 75)
                finally:
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
