from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from tts_bench import main, open_loop, open_loop_summary, poisson_plan, underruns

SR = 24000
B = 2 * SR  # bytes of one second of pcm16


class UnderrunTests(unittest.TestCase):
    def test_tag_policy_reaches_requests_without_changing_default(self):
        for mode in (None, "auto", "off"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args = ["tts_bench.py", "--url", "http://unused", "--model", "model", "--n", "1",
                        "--warmup", "0", "--out", directory]
                if mode:
                    args += ["--tags", mode]
                result = dict(audio_s=1, ttfa_s=.1, latency_s=.2, gaps=0, gap_s=0, pcm=b"", stalls=0)
                with mock.patch.object(sys, "argv", args), mock.patch("tts_bench.card_sample_rate", return_value=24000), \
                        mock.patch("tts_bench.one", return_value=result) as one:
                    main()
                self.assertEqual(one.call_args.args[-1], {} if mode is None else dict(tags=mode))

    def test_chunks_ahead_of_playback_have_no_gap(self):
        # 1 s chunks every 0.5 s: always ahead.
        self.assertEqual(underruns([(0.0, B), (0.5, B), (1.0, B)], SR), (0, 0.0, 0))

    def test_a_late_chunk_is_one_gap_and_playback_resumes_at_its_arrival(self):
        # 0.1 s of audio at t=0, next chunk at 0.3: 0.2 s gap; the third is in time again.
        gaps, gap_s, stalls = underruns([(0.0, B // 10), (0.3, B // 10), (0.35, B // 10)], SR)
        self.assertEqual((gaps, stalls), (1, 1))
        self.assertAlmostEqual(gap_s, 0.2)

    def test_banked_audio_covers_a_slow_chunk(self):
        gaps, _, stalls = underruns([(0.0, B), (0.01, B), (1.9, B // 10), (2.2, B // 10)], SR)
        self.assertEqual((gaps, stalls), (1, 1))

    def test_a_gap_of_at_most_100_ms_is_not_a_stall(self):
        gaps, _, stalls = underruns([(0.0, B // 10), (0.15, B // 10)], SR)
        self.assertEqual((gaps, stalls), (1, 0))

    def test_jitter_inside_the_tolerance_and_empty_streams(self):
        self.assertEqual(underruns([(0.0, B // 10), (0.1005, B // 10)], SR), (0, 0.0, 0))
        self.assertEqual(underruns([], SR), (0, 0.0, 0))
        self.assertEqual(underruns([(3.0, B)], SR), (0, 0.0, 0))

    def test_open_loop_plan_is_seeded_and_starts_do_not_wait_for_completions(self):
        plan = poisson_plan(200.0, 0.2, 7)
        self.assertEqual(plan, poisson_plan(200.0, 0.2, 7))
        self.assertTrue(all(0 < a < b < 0.2 for a, b in zip(plan, plan[1:])))
        # All three must be in flight at once for the barrier to release.
        together = threading.Barrier(3)
        res, late, peak = open_loop([0.0, 0.01, 0.02], lambda k: together.wait(5) * 0 + k)
        self.assertEqual((res, peak, len(late)), ([0, 1, 2], 3, 3))

    def test_open_loop_goodput_excludes_failed_late_and_stalled_requests(self):
        row = lambda ttfa, audio=1.0, stalls=0, lat=0.5: dict(ttfa_s=ttfa, latency_s=lat, audio_s=audio, stalls=stalls)
        rows = [row(0.1), row(0.9), row(0.1, stalls=1), row(None, audio=0.0), row(0.1, lat=9.0)]
        s = open_loop_summary(rows, [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.02, 0.0, 0.0], 5.0, 500.0)
        self.assertEqual((s["offered_rps"], s["goodput_rps"], s["completed_rps"], s["late_starts"]), (1.0, 0.4, 0.6, 1))
        self.assertAlmostEqual(s["slo_attainment"], 0.4)


if __name__ == "__main__":
    unittest.main()
