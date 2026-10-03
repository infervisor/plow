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


if __name__ == "__main__":
    unittest.main()
