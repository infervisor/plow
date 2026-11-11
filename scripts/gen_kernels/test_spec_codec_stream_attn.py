import unittest

from spec_codec_stream_attn import visible_range


class CausalWindow(unittest.TestCase):
    def test_fresh_stream_never_reads_carried_history(self):
        self.assertEqual(list(visible_range(0, 0, 249, 250, 251)), [249])
        self.assertEqual(list(visible_range(0, 1, 249, 250, 251)), [249, 250])

    def test_second_query_discards_oldest_window_key(self):
        first = list(visible_range(9999, 0, 249, 250, 251))
        second = list(visible_range(9999, 1, 249, 250, 251))
        self.assertEqual(first, list(range(250)))
        self.assertEqual(second, list(range(1, 251)))

    def test_partial_history_and_empty_row(self):
        self.assertEqual(list(visible_range(1, 0, 249, 250, 251)), [248, 249])
        self.assertEqual(list(visible_range(128, 1, 249, 250, 0)), [])


if __name__ == "__main__":
    unittest.main()
