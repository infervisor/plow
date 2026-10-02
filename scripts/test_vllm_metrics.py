import unittest
import math

from bench.vllm_metrics import KEYS, REQUEST_KEYS, cache_stats, cell_stats, metric_values


class PrefixAccountingTests(unittest.TestCase):
    def test_lookup_before_engine_iteration_is_counted(self):
        for hit_fraction in (0.2, 0.8):
            rows = []
            for t, steps in enumerate((0, 0, 1, 2, 2)):
                row = dict.fromkeys(KEYS, 0.0)
                row.update(t=float(t), iteration_tokens_total_count=steps,
                           iteration_tokens_total_sum=steps, generation_tokens_total=steps,
                           prefix_cache_queries_total=100 if t else 0,
                           prefix_cache_hits_total=100 * hit_fraction if t else 0)
                rows.append(row)
            result = cell_stats(rows, {'CELL_BEGIN': 0, 'CELL_END': 4})
            self.assertEqual(result['prefix_hit'], hit_fraction)
            self.assertEqual(result['steps'], 2)
            self.assertIsNone(result['prefix_request_hit'])

    def test_request_hits_are_distinct_from_reused_tokens(self):
        rows = []
        for t in range(3):
            row = dict.fromkeys(KEYS + REQUEST_KEYS, 0.0)
            row.update(t=t, iteration_tokens_total_count=t, iteration_tokens_total_sum=t,
                       prefix_cache_queries_total=t * 1000, prefix_cache_hits_total=t * 200,
                       plowrt_prefix_attach_hits_total=t * 8, plowrt_prefix_attach_misses_total=t * 2)
            rows.append(row)
        result = cell_stats(rows, {'CELL_BEGIN': 0, 'CELL_END': 2})
        self.assertEqual(result['prefix_token_hit'], 0.2)
        self.assertEqual(result['prefix_request_hit'], 0.8)

    def test_cache_rates_do_not_require_engine_iterations(self):
        rows = [dict(t=t, prefix_cache_hits_total=t * 200,
                     prefix_cache_queries_total=t * 1000,
                     plowrt_prefix_attach_hits_total=t * 8,
                     plowrt_prefix_attach_misses_total=t * 2) for t in range(3)]
        result = cache_stats(rows, {'CELL_BEGIN': 0, 'CELL_END': 2})
        self.assertEqual(result, dict(prefix_token_hit=0.2, prefix_request_hit=0.8))
        self.assertEqual(cache_stats([dict(t=0), dict(t=1)], {'CELL_BEGIN': 0}),
                         dict(prefix_token_hit=None, prefix_request_hit=None))
        self.assertIsNone(cache_stats([dict(t=0)], {'CELL_BEGIN': 0}))
        self.assertEqual(cache_stats(rows[::-1], {'CELL_BEGIN': 0}),
                         dict(prefix_token_hit=None, prefix_request_hit=None))

    def test_missing_request_metrics_are_unknown(self):
        values = metric_values('vllm:prefix_cache_hits_total{model_name="gemma"} 200\n')
        self.assertEqual(values[KEYS.index('prefix_cache_hits_total')], 200)
        self.assertTrue(all(math.isnan(v) for v in values[-2:]))
        values = metric_values('plowrt_prefix_attach_hits_total{model="gemma"} 8\n'
                               'plowrt_prefix_attach_misses_total{model="gemma"} 2\n')
        self.assertEqual(values[-2:], [8, 2])


if __name__ == '__main__':
    unittest.main()
