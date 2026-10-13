import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().with_name("vllm-rocm-client.sh")


class ClientMountTests(unittest.TestCase):
    def docker_argv(self, cwd, *args):
        """The `sudo -n docker run ...` argv the wrapper execs, captured by a stub `sudo`."""
        with tempfile.TemporaryDirectory() as bindir:
            log = Path(bindir) / "argv"
            stub = Path(bindir) / "sudo"
            stub.write_text(f"#!{sys.executable}\nimport sys\nopen({str(log)!r}, 'w').write('\\0'.join(sys.argv[1:]))\n")
            stub.chmod(0o755)
            env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", PLOW_CAMPAIGN_NO_NIX="1")
            env.pop("VLLM_CONTAINER", None)
            subprocess.run(["bash", str(WRAPPER), *args], cwd=cwd, env=env, check=True, timeout=30,
                           stdout=subprocess.DEVNULL)
            return log.read_text().split("\0")

    def mounts(self, argv):
        return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]

    def test_relative_script_runs_by_absolute_path(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "probe.py").write_text("")
            argv = self.docker_argv(root, "probe.py", "--x", "1")
            script = argv[argv.index("python3") + 2]
            self.assertEqual(script, str(Path(root).resolve() / "probe.py"))

    def test_file_outputs_mount_their_parent(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root).resolve()
            for name in ("results.csv", "r.json", "out.txt"):
                out = root / "res" / name
                argv = self.docker_argv(root, "-c", "pass", "--out", str(out))
                self.assertIn(f"{out.parent}:{out.parent}", self.mounts(argv))
                self.assertFalse(out.exists(), f"{out} must not be created as a directory")
                self.assertEqual(argv[-1], str(out))

    def test_extensionless_output_is_a_directory(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root).resolve() / "probe-out"
            argv = self.docker_argv(root, "-c", "pass", "--output", str(out))
            self.assertTrue(out.is_dir())
            self.assertIn(f"{out.parent}:{out.parent}", self.mounts(argv))


if __name__ == "__main__":
    unittest.main()
