import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'bench'))
from gpu_peak_mem import descendants, peak


class PeakMemoryTests(unittest.TestCase):
    def test_server_workers_exclude_unrelated_processes(self):
        pids = descendants('10', [('30', '20'), ('90', '1'), ('20', '10')])
        self.assertEqual(pids, {'10', '20', '30'})
        self.assertEqual(peak([
            't1, 20, 100\n', 't1, 30, 200\n', 't1, 90, 99999\n',
            't2, 20, 250\n', 't2, 30, 100\n', 't2, 10, N/A\n',
        ], pids), 350)

    def test_missing_is_unknown(self):
        self.assertIsNone(peak(['t, 90, 100', 't, 10, N/A'], {'10'}))


if __name__ == '__main__':
    unittest.main()
