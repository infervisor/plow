from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from tts_bench import underruns

SR = 24000
B = 2 * SR  # bytes of one second of pcm16


class UnderrunTests(unittest.TestCase):
    def test_chunks_ahead_of_playback_have_no_gap(self):
        # 1 s chunks every 0.5 s: always ahead.
        self.assertEqual(underruns([(0.0, B), (0.5, B), (1.0, B)], SR), (0, 0.0))

    def test_a_late_chunk_is_one_gap_and_playback_resumes_at_its_arrival(self):
        # 0.1 s of audio at t=0, next chunk at 0.3: 0.2 s gap; the third is in time again.
        gaps, gap_s = underruns([(0.0, B // 10), (0.3, B // 10), (0.35, B // 10)], SR)
        self.assertEqual(gaps, 1)
        self.assertAlmostEqual(gap_s, 0.2)

    def test_banked_audio_covers_a_slow_chunk(self):
        gaps, _ = underruns([(0.0, B), (0.01, B), (1.9, B // 10), (2.2, B // 10)], SR)
        self.assertEqual(gaps, 1)

    def test_jitter_inside_the_tolerance_and_empty_streams(self):
        self.assertEqual(underruns([(0.0, B // 10), (0.1005, B // 10)], SR), (0, 0.0))
        self.assertEqual(underruns([], SR), (0, 0.0))
        self.assertEqual(underruns([(3.0, B)], SR), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
