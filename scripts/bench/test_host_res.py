import http.server
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_res  # noqa: E402

TOOL = Path(__file__).resolve().with_name("host_res.py")
# Parent + one child: the child holds ~64 MiB of touched anonymous memory and spins on one core.
TREE = (
    "import os, sys, time\n"
    "if os.fork() == 0:\n"
    "    b = bytearray(64 << 20)\n"
    "    for i in range(0, len(b), 4096): b[i] = 1\n"
    "    t = time.time()\n"
    "    while time.time() - t < 30: pass\n"
    "    os._exit(0)\n"
    "time.sleep(30)\n"
)


class Metrics(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = (b'plowrt_kv_store_host_bytes{model="a"} 1073741824\n'
                b'plowrt_kv_store_host_bytes{model="b"} 1073741824\n'
                b'plowrt_kv_store_device_bytes{model="a"} 5\n')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class SampleTests(unittest.TestCase):
    def test_tree_memory_cpu_libs_and_tier(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), Metrics)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        server = subprocess.Popen([sys.executable, "-c", TREE], start_new_session=True)
        with tempfile.TemporaryDirectory() as d:
            out, tsv = Path(d) / "host.json", Path(d) / "host.tsv"
            try:
                time.sleep(0.5)
                s = subprocess.Popen([sys.executable, str(TOOL), "sample", "--root-pid", str(server.pid),
                                      "--json", str(out), "--tsv", str(tsv), "--interval-ms", "200",
                                      "--metrics-url", f"http://127.0.0.1:{srv.server_port}/metrics", "--pss"])
                time.sleep(2.0)
                s.send_signal(signal.SIGTERM)
                self.assertEqual(s.wait(timeout=10), 0)
            finally:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
                srv.shutdown()
                srv.server_close()
            h = json.loads(out.read_text())
            self.assertEqual(h["procs"]["peak"], 2)
            self.assertGreaterEqual(h["threads"]["peak"], 2)
            self.assertGreater(h["rss_mib"]["peak"], 64)
            self.assertGreater(h["pss_mib"]["peak"], 60)
            self.assertGreater(h["anon_mib"]["peak"], 60)
            self.assertGreater(h["cores"]["avg"], 0.5)
            self.assertLess(h["cores"]["peak"], 1.6)
            self.assertAlmostEqual(h["tier_host_mib"]["peak"], 2048)
            self.assertGreater(h["mem_available_mib"]["min"], 0)
            self.assertEqual(len(h["processes"]), 2)
            self.assertTrue(any("libc" in lib for lib in h["libs"]), h["libs"])
            self.assertEqual(h["libs_bytes"], sum(h["libs"].values()))
            rows = host_res.read_tsv(tsv)
            self.assertGreaterEqual(len(rows), 5)
            self.assertEqual(rows[0]["cpu_s"], 0.0)

    def test_no_process_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "host.json"
            s = subprocess.Popen([sys.executable, str(TOOL), "sample", "--root-pid", "999999999",
                                  "--json", str(out), "--interval-ms", "50"])
            time.sleep(0.3)
            s.send_signal(signal.SIGTERM)
            s.wait(timeout=10)
            self.assertFalse(out.exists())


def row(t, rss, cpu, procs=3):
    return dict(t=t, procs=procs, threads=10, rss_kib=rss, pss_kib=rss, anon_kib=rss, lck_kib=0, pin_kib=0,
                cpu_s=cpu, dt_s=2.0, avail_kib=1 << 20, tier_host_bytes=None)


def write_tsv(path, rows):
    with open(path, "w") as f:
        f.write("\t".join(host_res.COLS) + "\n")
        for r in rows:
            f.write("\t".join("" if r[k] is None else str(r[k]) for k in host_res.COLS) + "\n")


class CellsTests(unittest.TestCase):
    def test_split_by_cell_markers(self):
        with tempfile.TemporaryDirectory() as d:
            res = Path(d)
            write_tsv(res / "host.tsv", [row(100, 1024, 1.0), row(102, 2048, 4.0), row(104, 4096, 2.0),
                                         row(106, 1024, 0.0), row(108, 8192, 8.0)])
            (res / "cells.log").write_text("CELL_BEGIN a 101\nCELL_END a 105\nCELL_BEGIN b 107\nCELL_END b 109\n")
            host_res.main(["cells", str(res)])
            a = json.loads((res / "a.host.json").read_text())
            self.assertEqual(a["samples"], 2)
            self.assertEqual(a["rss_mib"]["peak"], 4)
            self.assertEqual(a["rss_mib"]["steady"], 3)
            self.assertEqual(a["cores"]["avg"], 1.5)
            self.assertEqual(a["cores"]["peak"], 2.0)
            self.assertNotIn("tier_host_mib", a)
            b = json.loads((res / "b.host.json").read_text())
            self.assertEqual(b["rss_mib"]["peak"], 8)


class FootprintReportTests(unittest.TestCase):
    def test_footprint_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            plow, vllm, assets, sp = d / "plow", d / "vllm", d / "assets", d / "sp" / "site-packages"
            for p in (plow, vllm, assets, sp / "torch", sp / "vllm"):
                p.mkdir(parents=True)
            (plow / "plowrt").write_bytes(b"x" * 1000)
            (assets / "model.pkt").write_bytes(b"x" * 300)
            (assets / "checkpoint").symlink_to(d / "sp")  # a packet links its checkpoint: not counted
            (sp / "torch" / "lib.so").write_bytes(b"x" * 5000)
            (sp / "vllm" / "a.py").write_bytes(b"x" * 700)
            py = d / "py"
            py.write_text(f"#!/bin/sh\necho '[\"{sp}\"]'\n")
            py.chmod(0o755)
            for res, ready, rss in ((plow, "12.0", 2048), (vllm, "60.0", 8192)):
                (res / "ready_s.txt").write_text(ready + "\n")
                (res / "host.json").write_text(json.dumps({"libs_bytes": 1 << 30, "rss_mib": {"peak": rss}}))
                (res / "q1.g.r1.host.json").write_text(json.dumps(host_res.summarize([row(1, rss * 1024, 2.0)] * 2)))
                (res / "q1.g.r1.peak_gpu_memory_mib.txt").write_text("70000\n")
                (res / "q1.g.r1.client_time.txt").write_text("100.0 20.0 5.0 300000\n")
                (res / "q1.g.r1.json").write_text(json.dumps(
                    {"overall": {"total_tok_s": 4000.0, "ttft_p99_ms": 400.0, "tpot_p99_ms": 40.0}}))
            (d / "hf").mkdir()
            (d / "hf" / "model.safetensors").symlink_to(sp / "torch" / "lib.so")
            host_res.main(["footprint", str(plow), "--side", "plow", "--assets", str(assets), "--hf", str(d / "hf")])
            host_res.main(["footprint", str(vllm), "--side", "vllm", "--pyref", str(py)])
            fp = json.loads((plow / "footprint.json").read_text())
            self.assertEqual(fp["plowrt_bytes"], 1000)
            self.assertEqual(fp["packet_dir_bytes"], 300)
            self.assertEqual(fp["checkpoint_bytes"], 5000)
            self.assertEqual(fp["ready_s"], 12.0)
            self.assertEqual(fp["mapped_libs_bytes"], 1 << 30)
            fv = json.loads((vllm / "footprint.json").read_text())
            self.assertEqual(fv["site_packages_bytes"], 5700)
            self.assertEqual(list(fv["site_packages"][str(sp)]["top"]), ["torch", "vllm"])
            buf = StringIO()
            with redirect_stdout(buf):
                host_res.main(["report", str(plow), str(vllm)])
            table = buf.getvalue()
            self.assertIn("cell q1.g.r1", table)
            self.assertIn("| Host RSS peak (cell) | 2,048 MiB | 8,192 MiB | 0.25x |", table)
            self.assertIn("| Server ready time | 12.0 s | 60.0 s | 0.20x |", table)
            self.assertIn("| Client CPU cores avg | 0.25 cores | 0.25 cores | 1.00x |", table)
            self.assertIn("| GPU memory peak (cell) | 70,000 MiB | 70,000 MiB | 1.00x |", table)


if __name__ == "__main__":
    unittest.main()
