from pathlib import Path
import subprocess
import tempfile
import unittest


class LaunchSnapshotTests(unittest.TestCase):
    def test_running_script_survives_edit_and_preserves_arguments(self):
        source = (Path(__file__).parent / "bench/llm_grid.sh").read_text()
        preamble = source.split('\nHERE=', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "grid.sh"
            script.write_text(preamble + '\nprintf "ready\\n"\nread -r proceed\n'
                              + '# padding\n' * 4096
                              + 'printf "%s %s\\n" "$1" "$2"\n')
            proc = subprocess.Popen(["bash", str(script), "plow", "result path"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(proc.stdout.readline(), "ready\n")
                script.write_text("echo 'unterminated\n")
                stdout, stderr = proc.communicate("continue\n", timeout=5)
                self.assertEqual(proc.returncode, 0, stderr)
                self.assertEqual(stdout, "plow result path\n")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()


if __name__ == "__main__":
    unittest.main()
