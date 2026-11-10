import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import gen_bench
import gen_search

CEIL = {"bw": 3000e9, "peak": {"bf16": 800e12, "fp8": 1400e12, "tf32": 400e12, "fp32": 50e12},
        "source": "test"}


def spec(**over):
    s = {
        "name": "toy",
        "op": "Gemm 64x64",
        "dtype": "bf16",
        "classes": {"b1": {"B": 1}, "b8": {"B": 8, "step": {"us": 1000.0, "calls": 10}}},
        "cost": lambda c: {"bytes": 3000e3 * c["B"], "flops": 800e6, "peak": "bf16"},
        "inputs": lambda c, seed: {},
        "rotate": ["w"],
        "out": lambda c: ((c["B"], 4), "float32"),
        "reference": lambda c, inp: None,
        "tol": {"rel_l2": 1e-3},
        "generators": {
            "tilelang": {"grid": {"bm": [16, 32], "stages": [1, 2, 3]},
                         "valid": lambda cfg, c: c["B"] > 1 or cfg["stages"] < 3,
                         "build": lambda cfg, c: None, "bind": lambda k, c, i, o: None},
            "triton": {"sweep": [{"bk": 64}], "build": lambda cfg, c: None,
                       "bind": lambda k, c, i, o: None},
        },
        "current": {"label": "plow gemm", "measured_us": {"b1": 4.0, "b8": 20.0}, "source": "sweep"},
    }
    s.update(over)
    return s


T16, T32, TR = {"bm": 16, "stages": 1}, {"bm": 32, "stages": 1}, {"bk": 64}


def row(cls, arm, cfg, us, ok=True, l2=1e-5, **over):
    over.setdefault("fp", None) or over.update(fp=gen_search.class_fp(spec(), cls))
    return {"class": cls, "arm": arm, "cfg": cfg, "us": us, "ok": ok, "rel_l2": l2, **over}


class SpecTests(unittest.TestCase):
    def check(self, s):
        return gen_search.check_spec(s, "spec_toy.py")

    def test_valid_spec_and_shipped_specs_load(self):
        self.check(spec())
        for path in sorted(gen_search.HERE.glob("spec_*.py")):
            loaded = gen_search.load_spec(str(path))
            self.assertEqual(f"spec_{loaded['name']}.py", path.name)
            self.assertTrue(gen_search.candidates(loaded, list(loaded["classes"]),
                                                  list(loaded["generators"])))
        self.assertEqual(gen_search.load_spec("s3gen_cfm_attn")["name"], "s3gen_cfm_attn")

    def test_rejects_malformed_specs(self):
        missing = spec()
        del missing["reference"]
        gens = spec()["generators"]
        bad = [missing, spec(extra=1), spec(name="other"), spec(classes={}), spec(tol={}),
               spec(generators={}),
               spec(generators={"tilelang": dict(gens["tilelang"], sweep=[{}])}),
               spec(generators={"triton": {"sweep": [{}], "build": None, "bind": None}}),
               spec(current={"label": "x"}),
               spec(cost=lambda c: {"bytes": 1})]
        for s in bad:
            with self.assertRaises(SystemExit):
                self.check(s)

    def test_grid_expansion_filter_and_order(self):
        s = spec()
        self.assertEqual(len(gen_search.expand(s["generators"]["tilelang"])), 6)
        self.assertEqual(gen_search.expand(s["generators"]["triton"]), [{"bk": 64}])
        cands = gen_search.candidates(s, ["b1", "b8"], ["tilelang", "triton"])
        self.assertEqual(len(cands), 7)
        self.assertEqual(cands[0], ("tilelang", {"bm": 16, "stages": 1}, ["b1", "b8"]))
        self.assertEqual(cands[2], ("tilelang", {"bm": 16, "stages": 3}, ["b8"]))
        self.assertEqual(gen_search.candidates(s, ["b1"], ["tilelang"])[2][1], {"bm": 32, "stages": 1})
        self.assertEqual(gen_search.cfg_label({"bm": 16, "stages": 3}), "bm16 stages3")

    def test_class_and_generator_selection(self):
        s = spec()
        ns = lambda **kw: type("A", (), {"classes": None, "gens": None, **kw})
        self.assertEqual(gen_search.pick(s, ns()), (["b1", "b8"], ["tilelang", "triton"]))
        self.assertEqual(gen_search.pick(s, ns(classes="b8", gens="triton")), (["b8"], ["triton"]))
        with self.assertRaises(SystemExit):
            gen_search.pick(s, ns(classes="b9"))


class RooflineTests(unittest.TestCase):
    def test_floor_is_the_larger_of_bandwidth_and_compute(self):
        self.assertAlmostEqual(gen_bench.floor_us({"bytes": 3000e3, "flops": 0, "peak": "bf16"}, CEIL), 1.0)
        self.assertAlmostEqual(gen_bench.floor_us({"bytes": 0, "flops": 800e6, "peak": "bf16"}, CEIL), 1.0)
        self.assertAlmostEqual(
            gen_bench.floor_us({"bytes": 3000e3, "flops": 800e6, "peak": "bf16", "passes": 3}, CEIL), 3.0)
        self.assertAlmostEqual(gen_bench.floor_us({"bytes": 0, "flops": 1400e6, "peak": "fp8"}, CEIL), 1.0)
        with self.assertRaises(ValueError):
            gen_bench.floor_us({"bytes": 0, "flops": 1, "peak": "f64"}, CEIL)

    def test_default_ceilings_are_the_calibrated_h100_numbers(self):
        c = gen_bench.load_ceilings()
        self.assertEqual((c["bw"], c["peak"]["bf16"], c["peak"]["fp8"]), (3094e9, 771.3e12, 1407e12))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ceilings.json"
            p.write_text(json.dumps({"bw_gbps": 1000.0, "bf16_tflops": 100.0, "fp8_tflops": 200.0}))
            c = gen_bench.load_ceilings(str(p))
        self.assertEqual((c["bw"], c["peak"]["bf16"], c["peak"]["fp8"]), (1000e9, 100e12, 200e12))

    def test_cold_sets_exceed_l2(self):
        self.assertEqual(gen_bench.cold_sets(53e6), 4)
        self.assertEqual(gen_bench.cold_sets(1e9), 2)
        for nbytes in (53e6, 3e6, 1e9):
            self.assertGreaterEqual(gen_bench.cold_sets(nbytes) * nbytes, gen_bench.COLD_BYTES)
        # under the cap's reach the caller flags the class cache-warm
        self.assertEqual(gen_bench.cold_sets(1e3), gen_bench.COLD_SETS_MAX)

    def test_error_ratio_of_a_zero_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        z, one = torch.zeros(2, 4), torch.ones(2, 4)
        self.assertEqual(gen_bench.rel_err(z, z.double()), (0.0, 0.0))
        self.assertEqual(gen_bench.rel_err(one, z.double())[1], float("inf"))
        self.assertAlmostEqual(gen_bench.rel_err(one * 1.5, one.double())[1], 0.5)
        self.assertAlmostEqual(gen_bench.pcm_rms(one * 0.5, torch.cat([one[:1], one[:1] * 0.5])), 0.5)

    def test_pcm_bound(self):
        self.assertAlmostEqual(gen_bench.PCM_RMS_BOUND, 8.81e-6, delta=1e-8)

    def test_verdict_follows_the_role_object_policy(self):
        v = gen_bench.verdict
        self.assertEqual(v(10.0, None), "no valid candidate")
        self.assertEqual(v(None, 5.0), "no current time")
        self.assertEqual(v(10.0, 10.5), "leave: plow kernel ahead")
        self.assertIn("unmeasured", v(10.0, 5.0))
        self.assertEqual(v(20.0, 14.0, {"us": 1000.0, "calls": 10}), "role-object candidate (6.0% of step)")
        self.assertEqual(v(20.0, 16.0, {"us": 1000.0, "calls": 10}), "reference: hand-port (4.0% of step)")


class TableTests(unittest.TestCase):
    def test_summary_picks_the_fastest_passing_candidate(self):
        rows = [row("b8", "tilelang", T16, 16.0), row("b8", "tilelang", T32, 12.0, ok=False),
                row("b8", "triton", TR, 14.0), row("b8", "triton", TR, 14.5),
                row("b1", "tilelang", T16, 5.0)]
        b1, b8 = gen_search.summarize(spec(), rows, CEIL)
        self.assertEqual((b8["gen_us"], b8["gen_label"], b8["tried"], b8["passed"]),
                         (14.5, "triton bk64", 3, 2))
        only = gen_search.summarize(spec(), rows, CEIL, ["b8"], ["tilelang"])
        self.assertEqual((len(only), only[0]["gen_label"], only[0]["tried"]),
                         (1, "tilelang bm16 stages1", 2))
        self.assertAlmostEqual(b8["floor_us"], 8.0)
        self.assertEqual(b8["verdict"], "role-object candidate (5.5% of step)")
        self.assertEqual((b1["cur_us"], b1["verdict"]), (4.0, "leave: plow kernel ahead"))

    def test_stale_and_obsolete_rows_never_count(self):
        rows = [row("b8", "tilelang", T16, 16.0),
                row("b8", "tilelang", T32, 9.0, fp="0" * 12),         # another geometry / tol
                row("b8", "tilelang", {"bm": 64, "stages": 1}, 8.0),  # config left the grid
                row("b1", "tilelang", {"bm": 16, "stages": 3}, 1.0),  # no longer valid for b1
                row("b9", "tilelang", T16, 1.0, fp="x")]              # class left the spec
        b1, b8 = gen_search.summarize(spec(), rows, CEIL)
        self.assertEqual((b8["gen_us"], b8["tried"], b1["gen_us"], b1["tried"]), (16.0, 1, None, 0))
        self.assertNotEqual(gen_search.class_fp(spec(), "b8"), gen_search.class_fp(spec(), "b1"))
        self.assertNotEqual(gen_search.class_fp(spec(), "b8"),
                            gen_search.class_fp(spec(tol={"rel_l2": 1e-4}), "b8"))
        stepless = spec(classes={"b1": {"B": 1}, "b8": {"B": 8}})
        self.assertEqual(gen_search.class_fp(spec(), "b8"), gen_search.class_fp(stepless, "b8"))

    def test_measured_current_rows_and_missing_arms(self):
        s = spec(current={"label": "probe", "lib": {"src": "x.cu", "flags": []},
                          "bind": lambda *a: None})
        rows = [row("b8", "current", {}, 30.0), row("b8", "tilelang", T16, 24.0)]
        b1, b8 = gen_search.summarize(s, rows, CEIL)
        self.assertEqual((b8["cur_us"], b8["cur_note"]), (30.0, ""))
        self.assertEqual((b1["cur_us"], b1["gen_us"], b1["verdict"]), (None, None, "no valid candidate"))

    def test_table_format(self):
        text = gen_bench.table([
            {"block": "Gemm, b8", "floor_us": 8.0, "cur_us": 20.0, "cur_note": " (recorded)",
             "gen_us": 16.0, "gen_label": "tilelang bm16", "rel_l2": 1.5e-6, "verdict": "v"},
            {"block": "Gemm, b1", "floor_us": 1.0, "cur_us": None, "gen_us": None, "verdict": "w"}])
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("| block (op, class) | current plow us | best generated us |"))
        self.assertEqual(lines[0].count("|"), lines[1].count("|"))
        self.assertEqual(lines[2], "| Gemm, b8 | 20.0 (recorded) | 16.0 | tilelang bm16 | 8.0 | 40% | 50% | "
                                   "-20.0% | 1.5e-06 | v |")
        self.assertEqual(lines[3], "| Gemm, b1 | — | — | — | 1.0 | — | — | — | — | w |")

    def test_report_names_ceilings_and_recorded_current(self):
        rows = [row("b8", "tilelang", T16, 16.0, cold=False, device="H100")]
        text = gen_search.report(spec(), rows, CEIL, every=True)
        self.assertIn("floor = max(bytes / 3000 GB/s", text)
        self.assertIn("recorded, not same-lease: sweep", text)
        self.assertIn("b1 0/0, b8 1/1", text)
        self.assertIn("measured on: H100", text)
        self.assertIn("cache-warm timing (rotating inputs under 160 MB at the set cap): b8", text)
        self.assertIn("| b8 | tilelang | bm16 stages1 | 16.0 |", text)
        self.assertNotIn("cache-warm", gen_search.report(spec(), [row("b8", "tilelang", T16, 16.0)], CEIL))


class CliTests(unittest.TestCase):
    def test_submit_command_is_a_bounded_queue_job_with_the_compiler_env(self):
        env = {"PATH": "/x/bin", "NVCC_PREPEND_FLAGS": "-ccbin g++-14", "TL_WG": "1", "PYREF": "/py",
               "PLOW_NVCC": "nvcc", "HOME": "/h"}
        cmd = gen_search.submit_command(env, ["toy", "--out", "/s"])
        self.assertEqual(cmd[:4], ["timeout", "--kill-after=30s", "840", "env"])
        self.assertLessEqual(gen_search.JOB_TERM_S + gen_search.JOB_KILL_S, 900)
        # set what the submitter has, unset what it lacks: the runner's environment is another's
        # (env takes its -u options before any assignment)
        self.assertEqual(cmd[4:12], ["-u", "CUDA_HOME", "-u", "LD_LIBRARY_PATH", "PATH=/x/bin",
                                     "NVCC_PREPEND_FLAGS=-ccbin g++-14", "TL_WG=1", "GEN_SEARCH_TL=TL_WG"])
        real = gen_search.submit_command({"PATH": os.environ["PATH"], "TL_WG": "1"}, [])
        probe = subprocess.run([*real[3:real.index("python3")], "sh", "-c",
                                "echo ${CUDA_HOME-unset} $TL_WG $GEN_SEARCH_TL"],
                               env={"CUDA_HOME": "/stale", "PATH": os.environ["PATH"]},
                               capture_output=True, text=True)
        self.assertEqual(probe.stdout.split(), ["unset", "1", "TL_WG"])
        self.assertEqual(cmd[12], "/py")
        self.assertEqual(cmd[13:], [str(gen_search.HERE / "gen_search.py"), "run", "toy", "--out", "/s"])
        self.assertIn("GEN_SEARCH_TL=", gen_search.submit_command({}, []))

    def test_results_resume_key_and_repo_guard(self):
        with tempfile.TemporaryDirectory() as d:
            path = gen_search.results_path(spec(), d)
            self.assertEqual(path.name, "toy.results.jsonl")
            self.assertEqual(gen_search.read_rows(path), [])
            path.write_text(json.dumps(row("b8", "triton", {"b": 1, "a": 2}, 1.0)) + "\n")
            self.assertEqual(gen_search.row_key(gen_search.read_rows(path)[0]),
                             ("b8", "triton", '{"a": 2, "b": 1}'))
        for inside in (gen_search.HERE, gen_search.REPO):
            with self.assertRaises(SystemExit):
                gen_search.main(["report", "s3gen_cfm_attn", "--out", str(inside)])

    def test_plan_and_list_run_without_a_gpu(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen_search.main(["list"])
            gen_search.main(["plan", "s3gen_cfm_attn", "--classes", "b64_q136"])
        self.assertIn("s3gen_cfm_attn: AttentionF32", buf.getvalue())
        self.assertIn("tilelang: 24 configs, 24 runs", buf.getvalue())
        self.assertIn("b64_q136: 76.3 MB", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
