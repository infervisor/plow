import importlib.util
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
    def test_waits_on_foreign_process_without_starting_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpuq.save(root / "1.json", dict(state="queued"))
            with patch.object(gpuq.os, "access", return_value=True), \
                 patch.object(gpuq.time, "monotonic", side_effect=[0, 1, 1, 2000]), \
                 patch.object(gpuq.time, "sleep"), \
                 patch.object(gpuq.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "FOREIGN GPU PROCS:\ngpu0:123")), \
                 patch.object(gpuq.subprocess, "Popen") as launch:
                gpuq.work(root)
                launch.assert_not_called()
            self.assertEqual(json.loads((root / "1.json").read_text())["state"], "queued")

    def test_fifo_records_failed_lease_without_promoting_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in (2, 1):
                gpuq.save(root / f"{number}.json", dict(state="queued", ngpu=1, label=str(number), command=["true"], cwd=directory))
            with patch.object(gpuq.os, "access", return_value=True), \
                 patch.object(gpuq.time, "monotonic", side_effect=[0, 1, 1, 2, 2, 2000]), \
                 patch.object(gpuq.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "GPU: no foreign compute procs\n")), \
                 patch.object(gpuq.subprocess, "Popen") as launch:
                launch.return_value.pid = 123
                launch.return_value.wait.side_effect = [76, 0]
                gpuq.work(root)
                self.assertEqual([call.args[0][3] for call in launch.call_args_list], ["1", "2"])
            self.assertEqual(json.loads((root / "1.json").read_text())["state"], "failed")
            self.assertEqual(json.loads((root / "2.json").read_text())["state"], "done")


if __name__ == "__main__":
    unittest.main()
