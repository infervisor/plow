import contextlib
import io
import itertools
import random
import tempfile
import unittest
from pathlib import Path

import bucket_demand as bd


def metrics(demand, launches, padded, seconds=(), engine=0, model="maya2", family="depth",
            caps=(1, 2, 4, 8, 16, 32, 64), overflow=0):
    l = f'model_name="{model}",engine="{engine}",family="{family}"'
    out = ["# TYPE plowrt_bucket_demand_total counter", "plowrt_requests_total 7"]
    out += [f'plowrt_bucket_capacity_info{{{l},capacity="{c}"}} 1' for c in caps]
    out += [f'plowrt_bucket_demand_total{{{l},rows="{r}"}} {n}' for r, n in demand.items()]
    out += [f'plowrt_bucket_launches_total{{{l},capacity="{c}"}} {n}' for c, n in launches.items()]
    out += [f'plowrt_bucket_padded_rows_total{{{l},capacity="{c}"}} {n}' for c, n in padded.items()]
    for c, (s, n) in dict(seconds).items():
        out += [f'plowrt_bucket_launch_seconds_sum{{{l},capacity="{c}"}} {s}',
                f'plowrt_bucket_launch_seconds_count{{{l},capacity="{c}"}} {n}']
    out += [f"plowrt_bucket_overflow_total{{{l}}} {overflow}", f"plowrt_bucket_demand_max{{{l}}} {max(demand)}",
            f'plowrt_bucket_missing_score{{{l},rows="20"}} 0.3553']
    return "\n".join(out) + "\n"


# 20 live rows on a 1..64 power-of-two ladder: every frame runs the 32-row program.
TWENTY = metrics({20: 900, 12: 100}, {32: 900, 16: 100}, {32: 10800, 16: 400},
                 seconds={16: (6.0, 100), 32: (108.0, 900)})


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = bd.main(list(argv))
    return code, out.getvalue()


class BucketDemandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def file(self, name, text):
        path = Path(self.tmp.name) / name
        path.write_text(text)
        return str(path)

    def test_twenty_rows_recommend_a_twenty_row_capacity(self):
        code, out = run(self.file("m.prom", TWENTY))
        self.assertEqual(code, 0, out)
        self.assertIn("model=maya2 family=depth", out)
        self.assertIn("declared: 1 2 4 8 16 32 64", out)
        self.assertIn("padded rows: 11200 of 30400 launched (36.8 %)", out)
        self.assertRegex(out, r"\n\s+20\s+900\s+90\.0 %\s+32\s+35\.5 %\n")
        # 900 * 20 + 100 * 16 = 19600 launched rows; 16 -> 60 ms and 32 -> 120 ms measured, so a
        # 20-row launch is priced 75 ms: 900 * 45 ms of 114 s saved.
        self.assertIn("add 1: [20]  launched rows -35.5 %, padding 2.0 %, launch time -35.5 % (40.5 of 114.0 s)", out)
        self.assertIn("add 2: [12, 20]  launched rows -36.8 %, padding 0.0 %", out)
        self.assertNotIn("add 3", out, "two undeclared row counts exist")

    def test_plan_is_exact_against_brute_force(self):
        rng = random.Random(7)
        for _ in range(200):
            caps = sorted(rng.sample(range(1, 25), rng.randint(1, 4)))
            demand = {r: rng.randint(1, 50) for r in rng.sample(range(1, caps[-1] + 1), min(caps[-1], rng.randint(1, 9)))}
            free = sorted(set(demand) - set(caps))
            for added, rows in bd.plan(caps, demand, 3):
                best = min(bd.launched(sorted(caps + list(pick)), demand)
                           for pick in itertools.combinations(free, len(added)))
                self.assertEqual(rows, best, (caps, demand, added))
                self.assertEqual(rows, bd.launched(sorted(caps + added), demand))
                self.assertFalse(set(added) & set(caps))

    def test_two_snapshots_give_the_window_and_engines_sum(self):
        before = metrics({20: 900, 12: 100}, {32: 900, 16: 100}, {32: 10800, 16: 400})
        after = (metrics({20: 900, 12: 100, 40: 50}, {32: 900, 16: 100, 64: 50}, {32: 10800, 16: 400, 64: 1200})
                 + metrics({40: 50}, {64: 50}, {64: 1200}, engine=1))
        code, out = run(self.file("a.prom", before), self.file("b.prom", after))
        self.assertEqual(code, 0, out)
        self.assertIn("frames: 100   launches: 100   padded rows: 2400 of 6400 launched (37.5 %)", out)
        self.assertIn("add 1: [40]  launched rows -37.5 %, padding 0.0 %", out)
        self.assertIn("launch time: n/a", out)
        code, split = run(self.file("b.prom", after), "--per-engine")
        self.assertIn("model=maya2#0 family=depth", split)
        self.assertIn("model=maya2#1 family=depth", split)
        with self.assertRaisesRegex(SystemExit, "went backwards"):
            run(self.file("b.prom", after), self.file("a.prom", before))

    def test_filters_overflow_and_binned_rows(self):
        text = TWENTY + metrics({320: 10, 4608: 5, 8192: 1}, {512: 10, 8192: 6}, {512: 1920, 8192: 17920},
                                model="g", family="prefill", caps=(512, 4160, 8192), overflow=3)
        code, out = run(self.file("m.prom", text), "--family", "prefill", "--add", "1")
        self.assertEqual(code, 0, out)
        self.assertNotIn("family=depth", out)
        self.assertIn("3 frames had more rows than the widest capacity 8192", out)
        self.assertIn("add 1: [4608]", out)
        code, out = run(self.file("m.prom", text), "--model", "none")
        self.assertEqual(code, 1)
        self.assertIn("no plowrt_bucket_* series", out)

    def test_cost_interpolates_only_between_measured_capacities(self):
        seconds = bd.cost_model({"launch_seconds_sum": {16: 6.0, 32: 108.0, 64: 0.0},
                                 "launch_seconds_count": {16: 100, 32: 900, 64: 0}})
        self.assertAlmostEqual(seconds(16), 0.06)
        self.assertAlmostEqual(seconds(24), 0.09)
        self.assertIsNone(seconds(8))
        self.assertIsNone(seconds(64))


if __name__ == "__main__":
    unittest.main()
