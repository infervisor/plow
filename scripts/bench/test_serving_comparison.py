import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serving_comparison as sc  # noqa: E402

PACKET = "ab" * 32
PROV = dict(model_version="org/model@abc (config+weights 123456789abc)",
            precision="weights fp8; KV cache bfloat16", gpu_name="NVIDIA H100 80GB HBM3", gpu_count=1,
            repeats=2, sampled="--temperature 1 --top-p 0.95")


def bench(duration, ttft, tpot, isl=1000, osl=128, n=384, conc=128):
    return dict(completed=n, failed=0, num_prompts=n, duration=duration, request_rate="inf",
                max_concurrency=conc, total_input_tokens=isl * n, total_output_tokens=osl * n,
                input_lens=[isl] * n, output_lens=[osl] * n, p99_ttft_ms=ttft, p99_tpot_ms=tpot)


def agentic(wall, ttft, tpot, sessions=32):
    reqs = [dict(prompt_tokens=8000, completion_tokens=128, error=None) for _ in range(sessions * 10)]
    return dict(config=dict(turns=10, system_tokens=1536, target_tokens=15600, max_tokens=128, api="chat",
                            temperature=0.0, top_p=None, sessions=sessions, sessions_per_worker=1,
                            session_header=True),
                overall=dict(errors=0, wall_s=wall, ttft_p99_ms=ttft, tpot_p99_ms=tpot), requests=reqs)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.base, self.plow = self.root / "vllm", self.root / "plow"
        for side, d, scale in (("baseline", self.base, 1.0), ("infervisor", self.plow, 1.25)):
            d.mkdir()
            prov = dict(PROV, side="vllm" if side == "baseline" else "plow")
            if side == "baseline":
                prov["stack"] = "vLLM 0.28.0 (--max-num-seqs 256)"
            else:
                prov.update(plowrt_git_sha="0123456789ab", packet_sha256=PACKET)
            self.write_prov(d, prov)
            for rep, jitter in ((1, 1.0), (2, 1.02)):
                cell = d / f"g128.r{rep}"
                cell.mkdir()
                (cell / "bench.json").write_text(json.dumps(bench(10 * scale * jitter, 2000 * scale, 20 * scale)))
                (d / f"g128.r{rep}.peak_gpu_memory_mib.txt").write_text(f"{int(65536 * scale)}\n")
                (d / f"a32.g.r{rep}.json").write_text(json.dumps(agentic(100 * scale * jitter, 3000 * scale, 30)))
                (d / f"a32.g.r{rep}.peak_gpu_memory_mib.txt").write_text("70000\n")
        self.gate = self.root / "gates.json"
        self.write_gate(True)

    @staticmethod
    def write_prov(d, prov):
        (d / "provenance.json").write_text(json.dumps(prov))

    def write_gate(self, passed, packet=PACKET):
        res = dict(kl_mean=0.10, vllm_kl_mean=0.13, top1_decisive=0.985, vllm_top1_decisive=0.982,
                   needle_acc=1.0, vllm_needle_acc=1.0, **{"pass": passed},
                   why=[] if passed else ["kl_mean 0.2 > vLLM x1.25 + 0.002"])
        self.gate.write_text(json.dumps(dict(packet_sha256=packet, gates=dict(llm_fp32_ref=res), **{"pass": passed})))

    def render(self, *extra):
        err = io.StringIO()
        out = self.root / "out"
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = sc.main(["render", "--baseline", str(self.base), "--infervisor", str(self.plow),
                          "--gate", str(self.gate), "--out", str(out), *extra])
        return rc, err.getvalue(), out


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Fixture(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def table(self, out, cell):
        report = json.loads((out / "comparison.json").read_text())
        c = next(c for c in report["cells"] if c["cell"] == cell)
        return report, c, {label: (b, i) for label, b, i in c["rows"]}

    def test_matched_pair_renders_strict_table(self):
        rc, err, out = self.f.render()
        self.assertEqual(rc, 0, err)
        report, c, rows = self.table(out, "g128")
        self.assertEqual([r[0] for r in c["rows"]], list(sc.LABELS))
        self.assertTrue(report["matched"] and c["matched"])
        for label in sc.MATCHED_ROWS:
            self.assertEqual(rows[label][1], sc.SAME)
        self.assertEqual(rows["Input / output length"][0], "1000 / 128 tokens")
        self.assertEqual(rows["Serving stack"],
                         ("vLLM 0.28.0 (--max-num-seqs 256)", f"Infervisor (plowrt 0123456789ab, packet {PACKET[:12]})"))
        self.assertEqual(rows["Output quality / correctness"][1], "Equivalent")
        self.assertTrue(all(b and i for b, i in rows.values()))
        md = (out / "comparison.md").read_text()
        self.assertIn("| Metric | Baseline | Infervisor |\n|---|---|---|\n| Model / version |", md)
        self.assertIn("## a32.g", md)
        csv_lines = (out / "comparison.csv").read_text().splitlines()
        self.assertEqual(len(csv_lines), 3)

    def test_ratios_means_and_per_gpu(self):
        _, _, out = self.f.render()
        _, c, rows = self.table(out, "g128")
        # Baseline: 1128 * 384 tokens over 10 s and 10.2 s; Infervisor 1.25x the duration.
        total = 1128 * 384
        base = (total / 10 + total / 10.2) / 2
        self.assertAlmostEqual(c["baseline"]["mean"]["total_tok_s"], base)
        self.assertEqual(rows["Total throughput"][0], f"{base:,.0f} tok/s")
        self.assertEqual(rows["Total throughput"][1], f"{base / 1.25:,.0f} tok/s (0.80x)")
        self.assertEqual(rows["Throughput / GPU"][1], f"{base / 1.25:,.0f} tok/s/GPU (0.80x)")
        self.assertEqual(rows["TTFT P99"], ("2,000.0 ms", "2,500.0 ms (1.25x)"))
        self.assertEqual(rows["TPOT P99"], ("20.00 ms", "25.00 ms (1.25x)"))
        self.assertEqual(rows["Peak GPU memory"], ("64.0 GiB", "80.0 GiB (1.25x)"))
        self.assertAlmostEqual(c["ratios"]["ttft_p99_ms"], 1.25)

    def test_two_gpus_halve_throughput_per_gpu(self):
        for d in (self.f.base, self.f.plow):
            prov = json.loads((d / "provenance.json").read_text())
            Fixture.write_prov(d, dict(prov, gpu_count=2))
        _, _, out = self.f.render()
        _, c, rows = self.table(out, "g128")
        self.assertAlmostEqual(c["baseline"]["mean"]["total_tok_s_gpu"] * 2, c["baseline"]["mean"]["total_tok_s"])
        self.assertEqual(rows["GPU type & count"], ("2 x NVIDIA H100 80GB HBM3", sc.SAME))

    def test_mismatch_shows_value_and_flags(self):
        prov = json.loads((self.f.plow / "provenance.json").read_text())
        Fixture.write_prov(self.f.plow, dict(prov, precision="weights fp8; KV cache fp8"))
        rc, err, out = self.f.render()
        self.assertEqual(rc, 1)
        report, c, rows = self.table(out, "g128")
        self.assertFalse(report["matched"])
        self.assertEqual(c["mismatched"], ["Precision / quantization"])
        self.assertEqual(rows["Precision / quantization"][1], "weights fp8; KV cache fp8")
        self.assertEqual(rows["Model / version"][1], sc.SAME)
        self.assertIn("NOT MATCHED", (out / "comparison.md").read_text())

    def test_per_request_length_mismatch_is_not_matched(self):
        for rep in (1, 2):
            p = self.f.plow / f"g128.r{rep}" / "bench.json"
            d = json.loads(p.read_text())
            d["input_lens"] = [1000] * 383 + [999]
            p.write_text(json.dumps(d))
        rc, _, out = self.f.render("--cells", "g128")
        self.assertEqual(rc, 1)
        _, c, rows = self.table(out, "g128")
        self.assertEqual(c["mismatched"], ["Input / output length"])
        self.assertNotEqual(rows["Input / output length"][1], sc.SAME)

    def test_failed_gate_is_not_equivalent(self):
        self.f.write_gate(False)
        rc, _, out = self.f.render()
        self.assertEqual(rc, 1)
        _, _, rows = self.table(out, "g128")
        self.assertEqual(rows["Output quality / correctness"][1], "Not equivalent (kl_mean 0.2 > vLLM x1.25 + 0.002)")

    def assertRefused(self, needle, *extra):
        rc, err, out = self.f.render(*extra)
        self.assertEqual(rc, 2)
        self.assertIn(needle, err)
        self.assertFalse(out.exists())

    def test_missing_gate_errors(self):
        self.f.gate.unlink()
        self.assertRefused("gate result missing")

    def test_gate_for_other_packet_errors(self):
        self.f.write_gate(True, packet="cd" * 32)
        self.assertRefused("gate scored packet cdcdcdcdcdcd")

    def test_missing_peak_memory_errors(self):
        (self.f.plow / "a32.g.r2.peak_gpu_memory_mib.txt").write_text("")
        self.assertRefused("a32.g.r2 peak GPU memory missing")

    def test_missing_provenance_field_errors(self):
        prov = json.loads((self.f.base / "provenance.json").read_text())
        Fixture.write_prov(self.f.base, dict(prov, stack=None))
        self.assertRefused("stack unrecorded")

    def test_unpaired_cell_and_single_repeat_error(self):
        (self.f.plow / "a32.g.r2.json").unlink()
        self.assertRefused("a32.g has 1 repeat(s)")
        cell = self.f.base / "s128.r1"
        cell.mkdir()
        self.assertRefused("cell s128 missing")

    def test_spread_over_ten_percent_is_flagged(self):
        p = self.f.plow / "g128.r2" / "bench.json"
        d = json.loads(p.read_text())
        d["p99_ttft_ms"] *= 1.5
        p.write_text(json.dumps(d))
        rc, _, out = self.f.render()
        self.assertEqual(rc, 0)
        _, c, rows = self.table(out, "g128")
        self.assertTrue(rows["TTFT P99"][1].endswith(" *"))
        self.assertIn("Infervisor TTFT P99 40.0%", c["spread_flagged"][0])
        self.assertIn("FLAGGED", (out / "comparison.md").read_text())


class RecordTest(unittest.TestCase):
    def test_precision_and_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf = Path(tmp) / "models--org--m" / "snapshots" / "0123456789abcdef"
            hf.mkdir(parents=True)
            (hf / "config.json").write_text(json.dumps({"text_config": {"dtype": "bfloat16"}, "quantization_config": {
                "quant_method": "compressed-tensors", "config_groups": {"g": {
                    "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
                    "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True}}}}}))
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PRECISION", None)
                os.environ.pop("MODEL_VERSION", None)
                os.environ.pop("KV_DTYPE", None)
                w = "compressed-tensors W8 float per-channel, A8 float dynamic per-token"
                self.assertEqual(sc.precision(hf, "vllm", "--max-num-seqs 256"), f"weights {w}; KV cache bfloat16")
                self.assertEqual(sc.precision(hf, "vllm", "--kv-cache-dtype fp8_per_token_head"),
                                 f"weights {w}; KV cache fp8_per_token_head")
                self.assertIsNone(sc.precision(hf, "plow", ""))
                os.environ["KV_DTYPE"] = "fp8_per_token_head"
                self.assertEqual(sc.precision(hf, "plow", ""), f"weights {w}; KV cache fp8_per_token_head")
                self.assertTrue(sc.model_version(hf).startswith("org/m@0123456789ab (config+weights "))


if __name__ == "__main__":
    unittest.main()
