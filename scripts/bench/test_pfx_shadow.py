import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfx_shadow  # noqa: E402

HEADER = dict(ev="header", block_rows=4096, block_bytes=2 << 20, full_layers=5, kvh_full=2, hd_full=512,
              elem_full=1, full_scale=True, slide_layers=25, kvh_slide=8, hd_slide=256, elem_slide=1,
              window=1024, ring=2048, max_ctx=16384, batch=8, snap_row_bytes=104000, cache_cap=0,
              min_free=0, host_cap=0)


def h16(tokens):
    return [f"{hash(tuple(tokens[:16 * (k + 1)])) & (2**64 - 1):016x}" for k in range(len(tokens) // 16)]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def trace(convs, cache_bytes=1 << 40, think=10.0):
    """Requests of `convs` conversations (lists of (new tokens, output tokens) per turn), one
    request per conversation per turn, with the trace's ideal match computed exactly."""
    recs, seen, rid, t = [], [], 0, 0.0
    hist = {c: [] for c in range(len(convs))}
    for turn in range(max(len(c) for c in convs)):
        for c, turns in enumerate(convs):
            if turn >= len(turns):
                continue
            new, out = turns[turn]
            prompt = hist[c] + new
            ideal, src = 0, None
            for q, toks in seen:
                n = lcp(prompt, toks)
                if n > ideal:
                    ideal, src = n, q
            rid += 1
            t += 0.5
            full = prompt + out
            recs.append(dict(ev="req", rid=rid, slot=0, t_att=t, t_fin=t + 0.2, prompt=len(prompt),
                             out=len(out), ideal=ideal, ideal_rid=src, ideal_src="output", radix=0,
                             att=0, kind="miss", lost=len(prompt), deeper=None, cache_bytes=cache_bytes,
                             h16=h16(full)))
            seen.append((rid, full))
            hist[c] = full
        t += think
    return recs


def write(recs):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    f.write(json.dumps(HEADER) + "\n")
    for r in recs:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


SYSTEM = list(range(1000, 3048))


def conv(c, turns=3):
    first = SYSTEM + [10_000 * (c + 1) + i for i in range(3600)]
    return [(first if k == 0 else [50_000 * (c + 1) + 100 * k + i for i in range(100)],
             [70_000 * (c + 1) + 100 * k + i for i in range(40)]) for k in range(turns)]


class ShadowTest(unittest.TestCase):
    def test_unbounded_policies_attach_where_their_granularity_allows(self):
        recs = trace([conv(0), conv(1)])
        path = write(recs)
        out = pfx_shadow.main([path, "--skip-s", "0", "--budget-mib", "1e9", "--scale", ""])
        # Turn 2+ of each conversation: plowrt attaches the previous prompt end (no output rows),
        # the hybrid allocator also the previous reply's whole 16-row blocks.
        self.assertGreater(out["i"]["cached"], 0.6)
        self.assertGreater(out["iv"]["cached"], out["i"]["cached"])
        self.assertLess(out["iv"]["structural"], out["i"]["structural"])
        self.assertAlmostEqual(out["i"]["capacity"], 0.0)
        # The second conversation's first turn finds the system prompt's shared end.
        self.assertGreater(out["i"]["att"], 0)

    def test_tight_budget_costs_capacity_and_attribution_sums_to_lost(self):
        recs = trace([conv(c) for c in range(12)])
        path = write(recs)
        tight = pfx_shadow.main([path, "--skip-s", "0", "--budget-mib", "600", "--scale", "", "--only", "i,iv"])
        self.assertGreater(tight["i"]["capacity"], 0.05)
        a = tight["traced"]
        self.assertEqual(sum(a["cats"].values()), a["prompt"] - a["att"])

    def test_finer_full_blocks_shrink_the_snapshot_tail(self):
        c = pfx_shadow.Costs(HEADER)
        coarse = pfx_shadow.BoundaryCache(c, 1e18, 4096)
        fine = pfx_shadow.BoundaryCache(c, 1e18, 64)
        rows = 5664
        self.assertEqual(coarse.private_bytes(rows) - fine.private_bytes(rows),
                         c.full_row * ((rows % 4096) - (rows % 64)))
        seg = pfx_shadow.BoundaryCache(c, 1e18, 4096, chunk=64)
        self.assertEqual(seg.private_bytes(rows), c.scale_row * rows + c.full_row * (rows % 4096) +
                         c.slide_row * (rows % 64))


if __name__ == "__main__":
    unittest.main()
