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
        free, _, pos, n, _e = g.teacher_force(stack(lambda h: h[-1] + 1), [4], cont)
        self.assertEqual((free, n, sorted(pos)), (cont, 1, [0, 1, 2, 3]))

    def test_flip_restarts_on_reference_history(self):
        cont = [5, 6, 7, 8]
        # Disagrees only after history [4, 5]: emits 99 where the reference has 6.
        complete = stack(lambda h: 99 if h == [4, 5] else h[-1] + 1)
        free, _, pos, n, _e = g.teacher_force(complete, [4], cont)
        self.assertEqual(n, 2)
        self.assertEqual(free, [5, 99, 100, 101])
        self.assertEqual(sorted(pos), [0, 1, 2, 3])
        self.assertEqual(pos[1][0][0], 99)  # the flip position is scored on the exact history
        self.assertEqual(pos[2][0][0], 7)   # and the next one after restarting on the reference

    def test_empty_reply_retried_and_counted(self):
        replies = iter([([], [], ""), ([1, 2], [dist((1, 1.0)), dist((2, 1.0))], "")])
        free, _, pos, n, empty = g.teacher_force(lambda ids, n: next(replies), [0], [1, 2])
        self.assertEqual((free, sorted(pos), n, empty), ([1, 2], [0, 1], 2, 1))

    def test_short_reply_restarts(self):
        calls = []
        def complete(ids, n):
            calls.append(len(ids))
            t = ids[-1] + 1
            return [t], [dist((t, 1.0))], ""
        free, _, pos, n, _e = g.teacher_force(complete, [0], [1, 2, 3])
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
    def test_near_tie_free_run_flip_is_not_decisive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            tie = g.score(ref, make_cap(ref, "p", flip_tie=True))
            self.assertLess(tie["cont_frac"], 1.0)
            self.assertEqual(tie["cont_decisive_frac"], 1.0)
            dec = g.score(ref, make_cap(ref, "p", flip_decisive=True))
            self.assertEqual(dec["cont_decisive_frac"], 0.5)

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

    def test_pooled_captures_score_their_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            ok, dec = make_cap(ref, "p"), make_cap(ref, "p", flip_decisive=True)
            s = g.score(ref, [ok, dec])
            self.assertEqual((s["cases"], s["positions"], s["decisive"]), (4, 8, 6))
            self.assertEqual(s["cont_decisive_frac"], 0.75)  # one flipped case of four, not of two
            self.assertAlmostEqual(s["top1_decisive"], 5 / 6)
            self.assertEqual(g.score(ref, [ok]), g.score(ref, ok))
            foreign = dict(dec, ref_sha256="0" * 64)
            with self.assertRaises(ValueError):
                g.score(ref, [ok, foreign])
            with self.assertRaises(ValueError):
                g.score(ref, [])

    def test_pooled_gate_replicates_the_peer(self):
        # Nine exact positions and one far one: the trimmed mean and p95 of [cap, cap] differ from cap's,
        # so a pooled candidate is only comparable with a peer pooled to the same size.
        ref = dict(sha256="x", cases=[dict(id=str(i), kind="natural", prompt_ids=[1], cont=[5],
                                           pos=[dict(top=dist((5, 0.9), (9, 0.1)), margin=2.2)]) for i in range(10)])
        cap = dict(arm="p", ref_sha256="x", cases={str(i): dict(free=[5], free_text="", pos={
            "0": dist((5, 0.9), (9, 0.1)) if i else dist((5, 0.5), (9, 0.5))}) for i in range(10)})
        th = dict(g.DEFAULTS)
        self.assertNotEqual(g.score_with(ref, [cap, cap], th)["kl_trim_mean"], g.score_with(ref, cap, th)["kl_trim_mean"])
        sc, sp = g.gate_scores(ref, [cap, cap], cap, th)
        self.assertEqual(sc, sp)
        self.assertEqual(g.verdict(sc, sp, th), [])

    def test_missing_case_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            cap = make_cap(ref, "x")
            cap["cases"]["b"] = dict(error="boom")
            s = g.score(ref, cap)
            self.assertEqual(s["missing_positions"], 2)
            self.assertEqual(s["needle_acc"], 0.0)


class VerdictTests(unittest.TestCase):
    peer = dict(missing_positions=0, kl_mean=0.02, kl_trim_mean=0.01, kl_p95=0.08, kl_p99=0.2, kl_over=0,
                kl_over_frac=0.0, top1_decisive=0.99, cont_frac=0.8, cont_decisive_frac=1.0, needle_acc=1.0)

    def test_within_peer_passes(self):
        cand = dict(self.peer, kl_mean=0.024, kl_trim_mean=0.012, kl_p95=0.1, kl_p99=0.25, kl_over=3,
                    kl_over_frac=0.004, top1_decisive=0.985, cont_frac=0.76)
        self.assertEqual(g.verdict(cand, self.peer), [])

    def test_each_metric_can_fail(self):
        for k, v in (("kl_mean", 0.03), ("kl_trim_mean", 0.02), ("kl_p95", 0.2), ("kl_over_frac", 0.01),
                     ("top1_decisive", 0.97), ("cont_frac", 0.7), ("cont_decisive_frac", 0.95), ("needle_acc", 0.9),
                     ("missing_positions", 1)):
            why = g.verdict(dict(self.peer, **{k: v}), self.peer)
            self.assertEqual(len(why), 1, (k, why))

    def test_p99_is_reported_not_gated(self):
        self.assertEqual(g.verdict(dict(self.peer, kl_p99=5.0), self.peer), [])

    def test_tail_statistics(self):
        ref = dict(sha256="x", cases=[dict(id=str(i), kind="natural", prompt_ids=[1], cont=[5],
                                           pos=[dict(top=dist((5, 0.9), (9, 0.1)), margin=2.2)]) for i in range(200)])
        # 198 exact positions, one mild and one gross disagreement
        cases = {str(i): dict(free=[5], free_text="", pos={"0": dist((5, 0.9), (9, 0.1))}) for i in range(200)}
        cases["0"]["pos"]["0"] = dist((5, 0.6), (9, 0.4))
        cases["1"]["pos"]["0"] = dist((9, 0.9), (5, 0.1))
        s = g.score(ref, dict(arm="a", ref_sha256="x", cases=cases))
        self.assertEqual(s["positions"], 200)
        self.assertEqual(s["kl_over"], 2)  # both > 0.1 nat
        self.assertAlmostEqual(s["kl_over_frac"], 0.01)
        self.assertLess(s["kl_trim_mean"], s["kl_mean"])  # the gross outlier is trimmed
        self.assertEqual(s["kl_p95"], 0.0)
        self.assertGreater(s["kl_p99"], 0.0)

    def test_scoring_stops_at_the_reference_stop_token(self):
        # cont = [5, 7(stop), 9]: the position after the stop token is neither scored nor part of the
        # continuation; a free-run divergence there is not a decisive divergence.
        pos = [dict(top=dist((5, 0.9), (9, 0.1)), margin=2.2), dict(top=dist((7, 0.9), (9, 0.1)), margin=2.2),
               dict(top=dist((9, 0.9), (5, 0.1)), margin=2.2)]
        ref = dict(sha256="x", meta=dict(eos=[7]), cases=[dict(id="a", kind="chat", prompt_ids=[1], cont=[5, 7, 9], pos=pos)])
        cap = dict(arm="a", ref_sha256="x", cases={"a": dict(
            free=[5, 7, 5], free_text="", pos={"0": pos[0]["top"], "1": pos[1]["top"], "2": dist((5, 0.9), (9, 0.1))})})
        s = g.score(ref, cap)
        self.assertEqual((s["positions"], s["kl_over"], s["cont_frac"], s["cont_decisive_frac"]), (2, 0, 1.0, 1.0))
        self.assertEqual(g.score(ref, cap, eos=[9])["positions"], 3)  # an override scores through its own stop
        self.assertEqual(g.score(dict(ref, meta={}), cap)["positions"], 3)  # no stop id: every position
        self.assertEqual(g.scored_len([5, 7, 9], {7}), 2)
        self.assertEqual(g.scored_len([5, 9], {7}), 2)

    def test_trimming_keeps_a_position(self):
        # one scored position (an immediate stop token): trimming never empties the sample, and
        # the capture scores against itself; a zero fraction trims nothing.
        pos = [dict(top=dist((7, 0.9), (9, 0.1)), margin=2.2)]
        ref = dict(sha256="x", meta=dict(eos=[7]), cases=[dict(id="a", kind="chat", prompt_ids=[1], cont=[7], pos=pos)])
        cap = dict(arm="a", ref_sha256="x", cases={"a": dict(free=[7], free_text="", pos={"0": pos[0]["top"]})})
        s = g.score(ref, cap)
        self.assertEqual((s["positions"], s["kl_trim_mean"], s["kl_mean"]), (1, 0.0, 0.0))
        self.assertEqual(g.verdict(s, s), [])
        ref["cases"][0].update(cont=[5, 7], pos=[dict(top=dist((5, 0.6), (9, 0.4)), margin=0.4), pos[0]])
        cap["cases"]["a"].update(free=[5, 7], pos={"0": dist((9, 0.9), (5, 0.1)), "1": pos[0]["top"]})
        self.assertGreater(g.score(ref, cap, kl_trim_frac=0.0)["kl_trim_mean"], 0.0)  # nothing trimmed
        self.assertEqual(g.score(ref, cap, kl_trim_frac=0.01)["kl_trim_mean"], 0.0)  # the one outlier

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
            self.assertIn("--out " + str(Path(tmp) / "llm_fp32_ref" / "plow.json"), up[0])

    def test_gate_steps_capture_each_concurrency(self):
        import campaign
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(reference="/r.json", vllm_hf="/hf", concurrency=[1, 8, 32])
            up, down = campaign.gate_steps("llm_fp32_ref", cfg, "python3", Path(tmp), Path(tmp))
            self.assertEqual(len(up), 3)
            self.assertIn("--arm plow-c8 --concurrency 8", up[1])
            self.assertIn("plow-c32.json", up[2])
            self.assertEqual(len(down), 1)
            self.assertIn("--arm vllm-c1 --concurrency 1", down[0])
            self.assertIn("--arm vllm-c32 --concurrency 32", down[0])

    def test_gate_score_per_concurrency(self):
        import campaign
        with tempfile.TemporaryDirectory() as tmp:
            ref = make_ref(tmp)
            d = Path(tmp) / "llm_fp32_ref"
            d.mkdir()
            (d / "plow-c1.json").write_text(json.dumps(make_cap(ref, "plow-c1")))
            (d / "plow-c8.json").write_text(json.dumps(make_cap(ref, "plow-c8", flip_decisive=True)))
            (Path(tmp) / "v1.json").write_text(json.dumps(make_cap(ref, "vllm-c1")))
            (Path(tmp) / "v8.json").write_text(json.dumps(make_cap(ref, "vllm-c8")))
            cfg = dict(reference=str(Path(tmp) / "ref.json"), needle_min=0.5, cont_drop_max=0.3, concurrency=[1, 8],
                       vllm_capture={"c1": str(Path(tmp) / "v1.json"), "c8": str(Path(tmp) / "v8.json")})
            res = campaign.gate_score("llm_fp32_ref", cfg, d)
            self.assertFalse(res["pass"])
            self.assertTrue(res["by_concurrency"]["c1"]["pass"])
            self.assertFalse(res["by_concurrency"]["c8"]["pass"])
            self.assertTrue(all(w.startswith("c8:") for w in res["why"]), res["why"])
            self.assertEqual(res["top1_decisive"], 1.0)  # headline = c1
            # a missing same-concurrency peer fails the gate (a c1 peer is not the c8 distribution)
            cfg["vllm_capture"] = {"c1": str(Path(tmp) / "v1.json")}
            res = campaign.gate_score("llm_fp32_ref", cfg, d)
            self.assertFalse(res["pass"])
            self.assertEqual(res["why"], ["peer at c8 missing in vllm_capture"])
            # a capture recorded at another concurrency than the one it is scored as fails
            cfg["vllm_capture"] = {"c1": str(Path(tmp) / "v1.json"), "c8": str(Path(tmp) / "v8.json")}
            (d / "plow-c8.json").write_text(json.dumps(dict(make_cap(ref, "plow-c8"), concurrency=1)))
            res = campaign.gate_score("llm_fp32_ref", cfg, d)
            self.assertFalse(res["pass"])
            self.assertEqual(res["why"], ["c8: plow-c8.json captured at c1, scored as c8"])
            (d / "plow-c8.json").write_text(json.dumps(dict(make_cap(ref, "plow-c8"), concurrency=8)))
            self.assertTrue(campaign.gate_score("llm_fp32_ref", cfg, d)["pass"])


class ScaledTensorTests(unittest.TestCase):
    """Synthetic 2-shard FP8 checkpoint; skipped without torch + safetensors."""

    def shards(self, tmp, drop=()):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch + safetensors required")
        p = "model.language_model.layers.0."
        w = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
        ex = torch.tensor([[[1.0, 2.0], [3.0, -1.0]]]).to(torch.float8_e4m3fn)
        a = {p + "mlp.down_proj.weight": w, p + "experts.down_proj": ex,
             p + "input_layernorm.weight": torch.ones(2, dtype=torch.bfloat16)}
        b = {p + "mlp.down_proj.weight_scale": torch.tensor([[2.0], [0.25]]),
             p + "experts.down_proj.weight_scale": torch.tensor([[[3.0], [0.5]]])}
        for k in drop:
            b.pop(p + k)
        files = [Path(tmp) / "model-00001-of-00002.safetensors", Path(tmp) / "model-00002-of-00002.safetensors"]
        save_file(a, str(files[0]))
        save_file(b, str(files[1]))
        return torch, p, files

    def test_scale_in_other_shard_dequantizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch, p, files = self.shards(tmp)
            got = {k: (w, s) for k, w, s in g.scaled_tensors(files, device="cpu")}
            self.assertEqual(sorted(got), sorted([p + "mlp.down_proj.weight", p + "experts.down_proj",
                                                  p + "input_layernorm.weight"]))
            w, s = got[p + "mlp.down_proj.weight"]
            self.assertTrue(torch.equal(w.float() * s, torch.tensor([[2.0, -4.0], [0.125, 1.0]])))
            w, s = got[p + "experts.down_proj"]
            self.assertTrue(torch.equal(w.float() * s, torch.tensor([[[3.0, 6.0], [1.5, -0.5]]])))
            self.assertIsNone(got[p + "input_layernorm.weight"][1])

    def test_missing_scale_refused(self):
        for k in ("mlp.down_proj.weight_scale", "experts.down_proj.weight_scale"):
            with tempfile.TemporaryDirectory() as tmp:
                _, _, files = self.shards(tmp, drop=(k,))
                with self.assertRaises(SystemExit) as e:
                    list(g.scaled_tensors(files, device="cpu"))
                self.assertIn(k, str(e.exception))


if __name__ == "__main__":
    unittest.main()
