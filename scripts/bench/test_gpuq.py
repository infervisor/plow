import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("gpuq", Path(__file__).with_name("gpuq.py"))
gpuq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpuq)


class QueueTests(unittest.TestCase):
    def test_submit_can_leave_the_queued_job_for_a_host_runner(self):
        for enqueue_only in (False, True):
            with self.subTest(enqueue_only=enqueue_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve() / "queue"
                root.mkdir()
                alias = Path(directory) / "queue-link"
                alias.symlink_to(root, target_is_directory=True)
                argv = ["gpuq", "--root", str(alias), "--vendor", "nvidia", "submit"]
                if enqueue_only:
                    argv.append("--enqueue-only")
                argv.extend(["m2s-test", "1", "printf", "%s", "hello world"])
                with patch.object(gpuq.sys, "argv", argv), \
                     patch.object(gpuq.platform, "system", return_value="Linux"), \
                     patch.dict(gpuq.os.environ, {"PLOW_CAMPAIGN_NO_NIX": "1", "GPU_LEASE_TIMEOUT": "777"}), \
                     patch.object(gpuq.subprocess, "Popen") as launch, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    gpuq.main()
                job = json.loads((root / (output.getvalue().strip() + ".json")).read_text())
                self.assertEqual(job["state"], "queued")
                self.assertEqual(job["label"], "m2s-test")
                self.assertEqual(job["ngpu"], 1)
                self.assertEqual(job["command"], ["printf", "%s", "hello world"])
                self.assertEqual(job["cwd"], gpuq.os.getcwd())
                self.assertEqual(job["lease_timeout"], 777)
                self.assertEqual(job["vendor"], "nvidia")
                if enqueue_only:
                    launch.assert_not_called()
                    self.assertFalse((root / "runner.log").exists())
                else:
                    launch.assert_called_once()
                    self.assertEqual(launch.call_args.args[0],
                                     [gpuq.sys.executable, str(Path(gpuq.__file__).resolve()),
                                      "--root", str(root), "run"])

    def test_waits_on_foreign_process_without_starting_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpuq.save(root / "1.json", dict(state="queued", ngpu=1))
            with patch.object(gpuq.os, "access", return_value=True), \
                 patch.object(gpuq, "gpu_count", return_value=1), \
                 patch.object(gpuq.time, "monotonic", side_effect=[0, 1, 1, 2000]), \
                 patch.object(gpuq.time, "sleep"), \
                 patch.object(gpuq.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "FOREIGN GPU PROCS:\ngpu0:123")), \
                 patch.object(gpuq.subprocess, "Popen") as launch:
                gpuq.work(root)
                launch.assert_not_called()
            self.assertEqual(json.loads((root / "1.json").read_text())["state"], "queued")

    def test_long_jobs_drain_fifo_and_record_failed_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpuq.save(root / "2.json", dict(state="queued", ngpu=1, label="2", command=["true"], cwd=directory))
            gpuq.save(root / "1.json", dict(state="queued", ngpu=1, label="1", command=["true"], cwd=directory, lease_timeout=600))
            with patch.object(gpuq.os, "access", return_value=True), \
                 patch.object(gpuq.time, "monotonic", side_effect=[0, 1, 1, 2001, 2002, 2002, 4002, 6003]), \
                 patch.object(gpuq.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "GPU: no foreign compute procs\n")), \
                 patch.dict(gpuq.os.environ, {"PLOW_MULTISTEP": "8", "PATH": "/nix/bin", "GPU_LEASE_TIMEOUT": "1800"}), \
                 patch.object(gpuq.subprocess, "Popen") as launch:
                launch.return_value.pid = 123
                launch.return_value.wait.side_effect = [76, 0]
                gpuq.work(root)
                self.assertEqual([call.args[0][3] for call in launch.call_args_list], ["1", "2"])
                self.assertTrue(all("PLOW_MULTISTEP" not in call.kwargs["env"]
                                    and call.kwargs["env"]["PATH"] == "/nix/bin"
                                    for call in launch.call_args_list))
                # Per-job submitter value wins; a job without one gets the queue default, not the runner's inherited env.
                self.assertEqual([call.kwargs["env"]["GPU_LEASE_TIMEOUT"] for call in launch.call_args_list], ["600", "14400"])
            self.assertEqual(json.loads((root / "1.json").read_text())["state"], "failed")
            self.assertEqual(json.loads((root / "2.json").read_text())["state"], "done")

    def test_vendor_job_leases_and_audits_its_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpuq.save(root / "1.json", dict(state="queued", ngpu=1, label="a", command=["true"], cwd=directory, vendor="amd"))
            with patch.object(gpuq.os, "access", return_value=True), \
                 patch.object(gpuq.time, "monotonic", side_effect=[0, 1, 1, 2, 2002]), \
                 patch.object(gpuq.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "GPU: no foreign compute procs\n")) as audit, \
                 patch.object(gpuq.subprocess, "Popen") as launch:
                launch.return_value.pid = 123
                launch.return_value.wait.return_value = 0
                gpuq.work(root)
                self.assertEqual(audit.call_args.args[0][1:], ["--vendor", "amd", "--audit"])
                self.assertEqual(launch.call_args.args[0][1:5], ["--vendor", "amd", "-n", "1"])

    def test_device_job_leases_named_cards_without_pool_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpuq.save(root / "1.json", dict(state="queued", ngpu=2, label="x", command=["true"], cwd=directory,
                                            devices="nvidia:0,amd:1"))
            with patch.object(gpuq.os, "access", return_value=True), \
                 patch.object(gpuq.time, "monotonic", side_effect=[0, 1, 1, 2, 2002]), \
                 patch.object(gpuq.subprocess, "run") as audit, \
                 patch.object(gpuq.subprocess, "Popen") as launch:
                launch.return_value.pid = 123
                launch.return_value.wait.return_value = 0
                gpuq.work(root)
                audit.assert_not_called()
                self.assertEqual(launch.call_args.args[0][1:4], ["--devices", "nvidia:0,amd:1", "x"])


if __name__ == "__main__":
    unittest.main()
