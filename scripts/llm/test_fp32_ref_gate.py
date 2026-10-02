import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaign"))
import fp32_ref_gate as g  # noqa: E402


def dist(*pairs):
    """Normalized top list from unnormalized (token, weight) pairs."""
    z = sum(w for _, w in pairs)
    return [[t, math.log(w / z)] for t, w in sorted(pairs, key=lambda x: -x[1])]


def stack(prefer):
    """A fake greedy stack: token = prefer(history), reported with a 2-token distribution."""
    def complete(ids, n):
        h, toks, tops = list(ids), [], []
        for _ in range(n):
            t = prefer(h)
            toks.append(t)
            tops.append(dist((t, 0.9), (t + 1000, 0.1)))
            h.append(t)
        return toks, tops, " ".join(map(str, toks))
    return complete


class KlTests(unittest.TestCase):
    def test_identical_is_zero(self):
        d = dist((1, 0.5), (2, 0.3), (3, 0.2))
        self.assertAlmostEqual(g.kl_top(d, d), 0.0, places=12)

    def test_exact_on_full_support(self):
        p, q = dist((1, 0.7), (2, 0.3)), dist((1, 0.5), (2, 0.5))
        want = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
        self.assertAlmostEqual(g.kl_top(p, q), want, places=12)

    def test_missing_token_takes_arm_floor(self):
        p = dist((1, 0.6), (2, 0.4))
        q = [[1, math.log(0.8)], [3, math.log(0.1)]]
        want = 0.6 * math.log(0.6 / 0.8) + 0.4 * math.log(0.4 / 0.1)
        self.assertAlmostEqual(g.kl_top(p, q), want, places=12)

    def test_percentile(self):
        self.assertEqual(g.percentile([3, 1, 2], 0.5), 2)
        self.assertAlmostEqual(g.percentile(list(range(101)), 0.99), 99.0)


class TeacherForceTests(unittest.TestCase):
    def test_exact_stack_one_request(self):
        cont = [5, 6, 7, 8]
        free, _, pos, n = g.teacher_force(stack(lambda h: h[-1] + 1), [4], cont)
        self.assertEqual((free, n, sorted(pos)), (cont, 1, [0, 1, 2, 3]))

    def test_flip_restarts_on_reference_history(self):
        cont = [5, 6, 7, 8]
        # Disagrees only after history [4, 5]: emits 99 where the reference has 6.
        complete = stack(lambda h: 99 if h == [4, 5] else h[-1] + 1)
        free, _, pos, n = g.teacher_force(complete, [4], cont)
        self.assertEqual(n, 2)
        self.assertEqual(free, [5, 99, 100, 101])
        self.assertEqual(sorted(pos), [0, 1, 2, 3])
        self.assertEqual(pos[1][0][0], 99)  # the flip position is scored on the exact history
        self.assertEqual(pos[2][0][0], 7)   # and the next one after restarting on the reference

    def test_short_reply_restarts(self):
        calls = []
        def complete(ids, n):
            calls.append(len(ids))
            t = ids[-1] + 1
            return [t], [dist((t, 1.0))], ""
        free, _, pos, n = g.teacher_force(complete, [0], [1, 2, 3])
        self.assertEqual((n, calls, sorted(pos)), (3, [1, 2, 3], [0, 1, 2]))


def make_ref(tmp):
    cases = [
        # decisive positions: margins ~2.2 nats; near tie at position 1 of "b"
        dict(id="a", kind="natural", prompt_ids=[1], cont=[5, 6],
             pos=[dict(top=dist((5, 0.9), (9, 0.1)), margin=2.2), dict(top=dist((6, 0.9), (9, 0.1)), margin=2.2)]),
        dict(id="b", kind="needle", prompt_ids=[2], cont=[7, 8], expect="7",
             pos=[dict(top=dist((7, 0.9), (9, 0.1)), margin=2.2), dict(top=dist((8, 0.51), (9, 0.49)), margin=0.04)]),
    ]
    path = Path(tmp) / "ref.json"
    path.write_text(json.dumps(dict(meta={}, cases=cases)))
    return g.load_ref(path)


def make_cap(ref, arm, flip_tie=False, flip_decisive=False):
    a = [[5, math.log(0.9)], [9, math.log(0.1)]]
    if flip_decisive:
        a = [[9, math.log(0.6)], [5, math.log(0.4)]]
    b1 = [[9, math.log(0.52)], [8, math.log(0.48)]] if flip_tie else [[8, math.log(0.51)], [9, math.log(0.49)]]
    return dict(arm=arm, ref_sha256=ref["sha256"], cases={
        "a": dict(free=[9 if flip_decisive else 5, 6], free_text="", pos={"0": a, "1": dist((6, 0.9), (9, 0.1))}),
        "b": dict(free=[7, 9 if flip_tie else 8], free_text="7", pos={"0": dist((7, 0.9), (9, 0.1)), "1": b1}),
    })


class ScoreTests(unittest.TestCase):
    def test_near_tie_flip_does_not_cost_top1(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            s = g.score(ref, make_cap(ref, "x", flip_tie=True))
            self.assertEqual((s["decisive"], s["top1_decisive"]), (3, 1.0))
            self.assertEqual(s["cont_full"], 1)
            self.assertEqual(s["needle_acc"], 1.0)
            self.assertEqual(s["missing_positions"], 0)
            self.assertAlmostEqual(s["by_kind"]["natural"]["kl_mean"], 0.0, places=12)

    def test_decisive_flip_costs_top1(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            s = g.score(ref, make_cap(ref, "x", flip_decisive=True))
            self.assertAlmostEqual(s["top1_decisive"], 2 / 3)
            self.assertEqual(s["by_kind"]["natural"]["cont_frac"], 0.0)

    def test_foreign_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            cap = make_cap(ref, "x")
            cap["ref_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                g.score(ref, cap)

    def test_missing_case_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            cap = make_cap(ref, "x")
            cap["cases"]["b"] = dict(error="boom")
            s = g.score(ref, cap)
            self.assertEqual(s["missing_positions"], 2)
            self.assertEqual(s["needle_acc"], 0.0)


class VerdictTests(unittest.TestCase):
    peer = dict(missing_positions=0, kl_mean=0.02, kl_p99=0.2, top1_decisive=0.99, cont_frac=0.8, needle_acc=1.0)

    def test_within_peer_passes(self):
        cand = dict(self.peer, kl_mean=0.024, kl_p99=0.25, top1_decisive=0.985, cont_frac=0.76)
        self.assertEqual(g.verdict(cand, self.peer), [])

    def test_each_metric_can_fail(self):
        for k, v in (("kl_mean", 0.03), ("kl_p99", 0.3), ("top1_decisive", 0.97), ("cont_frac", 0.7),
                     ("needle_acc", 0.9), ("missing_positions", 1)):
            why = g.verdict(dict(self.peer, **{k: v}), self.peer)
            self.assertEqual(len(why), 1, (k, why))

    def test_needle_floor_is_absolute(self):
        peer = dict(self.peer, needle_acc=0.5)
        self.assertTrue(any("needle_min" in w for w in g.verdict(dict(peer), peer)))

    def test_thresholds_override(self):
        cand = dict(self.peer, kl_mean=0.03)
        self.assertEqual(g.verdict(cand, self.peer, dict(kl_ratio_max=1.5)), [])


class CampaignScoreTests(unittest.TestCase):
    def test_gate_score_reads_captures(self):
        import campaign
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            d = Path(tmp) / "llm_fp32_ref"
            d.mkdir()
            (d / "plow.json").write_text(json.dumps(make_cap(ref, "plow", flip_tie=True)))
            (d / "vllm.json").write_text(json.dumps(make_cap(ref, "vllm")))
            cfg = dict(reference=str(Path(tmp) / "ref.json"), needle_min=0.5, cont_drop_max=0.3)
            res = campaign.gate_score("llm_fp32_ref", cfg, d)
            self.assertTrue(res["pass"], res["why"])
            self.assertEqual(res["vllm_top1_decisive"], 1.0)
            (d / "plow.json").write_text(json.dumps(make_cap(ref, "plow", flip_decisive=True)))
            res = campaign.gate_score("llm_fp32_ref", cfg, d)
            self.assertFalse(res["pass"])
            self.assertTrue((d / "fp32_ref.md").is_file())

    def test_gate_steps_capture_both(self):
        import campaign
        with tempfile.TemporaryDirectory() as tmp:
            up, down = campaign.gate_steps("llm_fp32_ref", dict(reference="/r.json", vllm_hf="/hf", vllm_args="--max-model-len 16384"),
                                           "python3", Path(tmp), Path(tmp))
            self.assertIn("--arm plow", up[0])
            self.assertIn("vllm.entrypoints.cli.main serve /hf", down[0])
            self.assertIn("--arm vllm", down[0])
            up, down = campaign.gate_steps("llm_fp32_ref", dict(reference="/r.json", vllm_capture="/v.json"),
                                           "python3", Path(tmp), Path(tmp))
            self.assertEqual(down, [])


if __name__ == "__main__":
    unittest.main()
