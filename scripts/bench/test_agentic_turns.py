import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agentic_turns  # noqa: E402


class FakeServer:
    """OpenAI chat/completions SSE stub: records requests, replies with numbered words."""

    def __init__(self):
        from aiohttp import web
        self.requests = []
        self.web = web
        app = web.Application()
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/v1/chat/completions", self.complete)
        app.router.add_post("/v1/completions", self.complete)
        self.loop = asyncio.new_event_loop()
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self.hits = 0
        self.fail_from = None
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    async def models(self, request):
        return self.web.json_response({"data": [{"id": "stub"}]})

    async def metrics(self, request):
        self.hits += 100
        return self.web.Response(text=f"vllm:prefix_cache_hits_total {self.hits}\n"
                                      f"vllm:prefix_cache_queries_total {self.hits * 2}\n")

    async def complete(self, request):
        body = await request.json()
        self.requests.append((dict(request.headers), body))
        resp = self.web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                "X-Session-Cached-Tokens": "7"})
        await resp.prepare(request)
        n = body["max_tokens"]
        chat = "messages" in body
        if self.fail_from is not None and len(self.requests) > self.fail_from:
            await resp.write(b'data: {"error": {"message": "device fault"}}\n\n')
            return resp
        for i in range(n):
            piece = f"w{len(self.requests)}_{i} "
            choice = {"delta": {"content": piece}} if chat else {"text": piece}
            await resp.write(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n")
        usage = {"prompt_tokens": 1000, "completion_tokens": n}
        if chat:
            usage["prompt_tokens_details"] = {"cached_tokens": 400}
        await resp.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    def close(self):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)


class AgenticTurnsTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def run_bench(self, *extra):
        out = Path(self.tmp.name) / "res.json"
        rc = agentic_turns.main(["--url", f"http://127.0.0.1:{self.server.port}", "--sessions", "3",
                                 "--turns", "4", "--system-tokens", "200", "--target-tokens", "2000",
                                 "--max-tokens", "5", "--temperature", "0", "--seed", "9",
                                 "--out", str(out), *extra])
        return rc, json.loads(out.read_text())

    def test_chat_history_replays_real_replies_with_a_session_header(self):
        rc, res = self.run_bench()
        self.assertEqual(rc, 0)
        reqs = self.server.requests
        self.assertEqual(len(reqs), 12)
        by_session = {}
        for headers, body in reqs:
            by_session.setdefault(headers["X-Session-Id"], []).append(body)
            self.assertEqual(body["temperature"], 0)
            self.assertTrue(body["ignore_eos"] and body["stream"])
        self.assertEqual(len(by_session), 3)
        for bodies in by_session.values():
            self.assertEqual([len(b["messages"]) for b in bodies], [2, 4, 6, 8])
            systems = {b["messages"][0]["content"] for b in bodies}
            self.assertEqual(len(systems), 1)
            for prev, cur in zip(bodies, bodies[1:]):
                # The previous request's full history is a prefix, and the reply fed back is the
                # text the server streamed (words numbered by request).
                self.assertEqual(cur["messages"][:len(prev["messages"])], prev["messages"])
                self.assertRegex(cur["messages"][len(prev["messages"])]["content"], r"^w\d+_0 w\d+_1 ")
        o = res["overall"]
        self.assertEqual((o["requests"], o["errors"], o["output_tokens"]), (12, 0, 60))
        self.assertAlmostEqual(o["cached_fraction"], 0.4)
        self.assertEqual([t["requests"] for t in res["per_turn"]], [3, 3, 3, 3])
        self.assertAlmostEqual(res["server"]["prefix_token_hit"], 0.5)

    def test_content_is_deterministic_per_seed_and_grows_to_the_target(self):
        args = agentic_turns.argparse.Namespace(seed=3, system_tokens=200, target_tokens=4000, turns=10,
                                                max_tokens=100, sessions=2, sessions_per_worker=1)
        sizer = agentic_turns.Sizer(None)
        a = agentic_turns.build_sessions(args, sizer)
        b = agentic_turns.build_sessions(args, sizer)
        self.assertEqual(a, b)
        system, sessions, per_turn = a
        self.assertNotEqual(sessions[0], sessions[1])
        last = sizer.count(system) + sum(sizer.count(u) for u in sessions[0]) + 9 * args.max_tokens
        self.assertLess(abs(last - args.target_tokens) / args.target_tokens, 0.1)
        args.seed = 4
        self.assertNotEqual(agentic_turns.build_sessions(args, sizer)[0], system)

    def test_a_stream_error_is_an_error_and_ends_the_session(self):
        self.server.fail_from = 6
        rc, res = self.run_bench()
        self.assertEqual(rc, 1)
        o = res["overall"]
        # 6 good turns, then each of the 3 sessions fails once and stops.
        self.assertEqual((o["requests"], o["errors"]), (9, 3))
        self.assertTrue(all("device fault" in r["error"] for r in res["requests"] if r["error"]))

    def test_completions_api_and_header_off(self):
        rc, res = self.run_bench("--api", "completions", "--no-session-header")
        self.assertEqual(rc, 0)
        self.assertTrue(all("X-Session-Id" not in h for h, _ in self.server.requests))
        self.assertTrue(all(b["prompt"].endswith("Assistant:") for _, b in self.server.requests))
        # No usage details: the plowrt header is the cached count.
        self.assertAlmostEqual(res["overall"]["cached_fraction"], 7 / 1000)


def open_args(**kw):
    d = dict(seed=5, rate=1.0, duration=300.0, apps=4, system_median=1536, system_sigma=0.6, turns_mean=6,
             turns_max=20, first_median=1500, first_sigma=1.0, tool_median=700, tool_sigma=1.0, out_median=160,
             out_sigma=0.7, out_min=16, out_max=1024, think_median_s=5.0, think_sigma=0.8, think_max_s=60.0,
             max_model_len=16384, template_margin=256, warmup=60.0, cooldown=20.0, slo_ttft_ms=2000.0,
             slo_tpot_ms=100.0)
    d.update(kw)
    return agentic_turns.argparse.Namespace(**d)


class OpenLoopPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_per_seed(self):
        a, b = agentic_turns.build_plan(open_args()), agentic_turns.build_plan(open_args())
        self.assertEqual(a, b)
        self.assertNotEqual(a, agentic_turns.build_plan(open_args(seed=6)))
        apps, plan = a
        self.assertEqual(len(apps), 4)
        arrivals = [p["arrival_s"] for p in plan]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0 < t < 300 for t in arrivals))
        # Content: a pure function of the seed, sized to the plan.
        sizer = agentic_turns.Sizer(None)
        small = open_args(duration=20.0)
        apps, plan = agentic_turns.build_plan(small)
        c1 = agentic_turns.build_open_content(small, sizer, apps, plan)
        self.assertEqual(c1, agentic_turns.build_open_content(small, sizer, apps, plan))
        for sess, users in zip(plan, c1[1]):
            for turn, text in zip(sess["turns"], users):
                self.assertLess(abs(sizer.count(text) - turn["user_tokens"]), 0.1 * turn["user_tokens"] + 24)

    def test_distribution_parameters(self):
        args = open_args(rate=2.0, duration=2000.0)
        apps, plan = agentic_turns.build_plan(args)
        self.assertLess(abs(len(plan) - 4000) / 4000, 0.05)
        turns = [t for p in plan for t in p["turns"]]
        med = lambda xs: sorted(xs)[len(xs) // 2]
        self.assertLess(abs(med([t["max_tokens"] for t in turns]) - 160) / 160, 0.1)
        self.assertLess(abs(med([t["think_s"] for t in turns if t["think_s"]]) - 5.0) / 5.0, 0.1)
        self.assertTrue(all(16 <= t["max_tokens"] <= 1024 for t in turns))
        self.assertTrue(all(p["turns"][0]["think_s"] == 0 for p in plan))
        for p in plan:
            self.assertTrue(1 <= len(p["turns"]) <= 20)
            for t in p["turns"]:
                self.assertLessEqual(t["prompt_tokens"] + t["max_tokens"], 16384 - 256)
        # Without the context cap the turn count is geometric with the requested mean.
        _, wide = agentic_turns.build_plan(open_args(rate=2.0, duration=2000.0, max_model_len=10**9, turns_max=1000))
        mean = sum(len(p["turns"]) for p in wide) / len(wide)
        self.assertLess(abs(mean - 6) / 6, 0.05)
        prompts = [t["prompt_tokens"] for t in turns]
        self.assertLess(min(prompts), 2000)
        self.assertGreater(max(prompts), 15000)

    def test_rate_for_concurrency_is_littles_law(self):
        args = open_args()
        plan = agentic_turns.build_plan(open_args(rate=1.0, duration=2000.0))[1]
        turns = sum(len(p["turns"]) for p in plan) / len(plan)
        self.assertAlmostEqual(agentic_turns.rate_for_concurrency(args, 48, 8.0), 48 / (turns * 8.0))

    def test_window_metrics(self):
        args = open_args(duration=100.0, warmup=10.0, cooldown=10.0)
        row = lambda t0, t1, ttft, tpot, err=None: dict(t_start=t0, t_end=t1, ttft_ms=ttft, tpot_ms=tpot,
                                                       e2e_ms=(t1 - t0) * 1e3, prompt_tokens=900,
                                                       completion_tokens=100, cached_tokens=450, error=err)
        rows = [row(5, 15, 100, 20),        # before the window: only its 5 s overlap counts as in-flight
                row(20, 30, 1500, 50),      # good
                row(30, 40, 2500, 50),      # TTFT miss
                row(40, 60, 500, 150),      # TPOT miss
                row(50, 70, 1000, 99),      # good
                row(95, 99, 100, 10)]       # cooldown: unmeasured
        timeline = [(t, 2, 3) for t in range(0, 100, 10)]
        m = agentic_turns.window_metrics(rows, args, timeline)
        self.assertEqual(m["requests"], 4)
        self.assertAlmostEqual(m["goodput_req_s"], 2 / 80)
        self.assertAlmostEqual(m["slo_attainment"], 0.5)
        self.assertAlmostEqual(m["total_tok_s"], 4000 / 80)
        self.assertAlmostEqual(m["mean_inflight"], (5 + 10 + 10 + 20 + 20) / 80)
        self.assertAlmostEqual(m["cached_fraction"], 0.5)
        self.assertEqual((m["max_inflight"], m["mean_sessions"]), (2, 3))
        self.assertEqual(m["ttft_p50_ms"], 1500)
        rows.append(row(60, 61, None, None, err="RuntimeError: cut"))
        m = agentic_turns.window_metrics(rows, args, timeline)
        self.assertEqual((m["requests"], m["errors"]), (5, 1))
        self.assertAlmostEqual(m["slo_attainment"], 2 / 5)


class OpenLoopServeTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def test_open_loop_replays_history_with_planned_lengths(self):
        out = Path(self.tmp.name) / "q.json"
        argv = ["--url", f"http://127.0.0.1:{self.server.port}", "--open-loop", "--rate", "4", "--duration", "3",
                "--warmup", "0.5", "--cooldown", "0.5", "--think-median-s", "0.05", "--think-max-s", "0.2",
                "--system-median", "300", "--first-median", "200", "--tool-median", "100", "--out-median", "6",
                "--out-min", "2", "--out-max", "12", "--turns-mean", "3", "--temperature", "0", "--seed", "4",
                "--out", str(out)]
        self.assertEqual(agentic_turns.main(argv), 0)
        res = json.loads(out.read_text())
        args = agentic_turns.argparse.Namespace(**res["config"])
        apps, plan = agentic_turns.build_plan(args)
        rows = res["requests"]
        self.assertEqual(len(rows), len(self.server.requests))
        self.assertGreater(len(rows), 4)
        by_session = {}
        for headers, body in self.server.requests:
            by_session.setdefault(headers["X-Session-Id"], []).append(body)
            self.assertTrue(body["ignore_eos"] and body["temperature"] == 0)
        for sid, bodies in by_session.items():
            s = int(sid.rsplit("-", 1)[1])
            self.assertEqual([b["max_tokens"] for b in bodies], [t["max_tokens"] for t in plan[s]["turns"]][:len(bodies)])
            self.assertEqual(bodies[0]["messages"][0]["content"].split(")")[0], f"You are assistant app {plan[s]['app']} (seed 4")
            for prev, cur in zip(bodies, bodies[1:]):
                self.assertEqual(cur["messages"][:len(prev["messages"])], prev["messages"])
        self.assertTrue(all(r["t_start"] < 3 for r in rows))
        o = res["overall"]
        self.assertEqual((o["errors"], o["errors_total"]), (0, 0))
        self.assertEqual(o["requests"], sum(1 for r in rows if 0.5 <= r["t_start"] < 2.5))
        self.assertAlmostEqual(o["slo_attainment"], 1.0)
        self.assertGreater(o["mean_inflight"], 0)
        self.assertEqual(res["plan"]["sessions"], len(plan))
        self.assertTrue(res["timeline"])


if __name__ == "__main__":
    unittest.main()
