#!/usr/bin/env python3
"""Prefix-cache shadow replay over a `PLOW_PFX_TRACE` trace.

    pfx_shadow.py <trace.jsonl> [--skip-s 90] [--until-s T] [--budget-mib M] [--json out.json]

Part A attributes every lost prompt token of the traced run (what plowrt did) by an explicit
precedence per request: `[att, deeper)` to the fate of the deepest boundary that would have
attached deeper (dropped by mechanism/trigger, refused, live), `[deeper, ideal)` to "no
boundary" (by whether the ideal match ended in an earlier prompt or only in an output, and
whether that request was still in flight), `[ideal, prompt)` to tokens never computed before.

Part B replays the same requests (attach at `t_att`, finish at `t_fin`, the 16-row prefix hash
chain `h16`) against model cache policies under one memory budget (default: the traced pool's
median `cache_bytes`) with the trace header's byte costs:

  i      plowrt today: whole full-attention blocks of `block_rows` shared through the radix,
         private boundary snapshots (sliding window + fp8 full-scale prefix + sub-block tail)
         at prompt ends (prompt-9, 32-aligned) and shared-prefix ends (recent-prompt
         divergence >= 256 rows past the attach), parent retired when its single continuation
         publishes, FP8 strict publish (orphaned blocks dropped), LRU of boundary + the blocks
         only it kept, turn-end touch/restore of the prompt end.
  ii-C   i with the sliding window held as shared refcounted C-row chunks (no parent retire:
         a parent costs only the chunks it does not share).
  iii-G  i with G-row full-attention blocks, FP8 scales co-owned with the blocks (no strict
         publish); iii-64-strict keeps the strict rule.
  v-C-G  ii-C and iii-G together.
  iv     vLLM-like hybrid allocator: 16-row blocks for both groups, a hit needs every full
         block of the prefix and the sliding blocks of its last window, LRU per block,
         generated blocks cached too.

Each policy also runs with an unbounded budget: `structural` loss = ideal - attached(inf)
(granularity, publication policy, uncached outputs), `capacity` loss = attached(inf) -
attached(budget). Bytes per retained conversation = integral of retained bytes over the
integral of conversations whose last prompt end is attachable (sampled every 10 s over
conversations active in the last 300 s); a conversation is a chain of requests whose ideal
match covers the previous request's whole prompt (`ideal_rid`). The replay holds arrival and
finish times fixed: a policy that hits more would also run faster, which it does not model.
"""
import argparse
import heapq
import json
import statistics
import sys
from collections import defaultdict

MiB = 1 << 20


def load(path):
    header, reqs, other = None, [], defaultdict(int)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ev = r.get("ev")
            if ev == "header":
                header = r
            elif ev == "req":
                reqs.append(r)
            else:
                other[ev] += 1
                if ev == "stats":
                    other["dropped_events"] = r.get("dropped_events", 0)
    if header is None:
        sys.exit("no header record: was the trace armed at pool bring-up?")
    reqs.sort(key=lambda r: r["t_att"])
    return header, reqs, other


class Costs:
    def __init__(self, h):
        self.full_row = h["full_layers"] * 2 * h["kvh_full"] * h["hd_full"] * h["elem_full"]
        self.scale_row = h["full_layers"] * 2 * h["kvh_full"] * 4 if h["full_scale"] else 0
        self.slide_row = h["snap_row_bytes"]
        self.window = h["window"] if h["slide_layers"] else 0
        self.block_rows = h["block_rows"]
        self.strict = bool(h["full_scale"])


# ----------------------------------------------------------------------------- part A


def attribute(reqs):
    """Lost tokens of the traced run by cause (precedence: deeper boundary fate, no boundary,
    never seen)."""
    cats = defaultdict(int)
    kinds = defaultdict(int)
    prompt = att = 0
    for r in reqs:
        p, a, ideal = r["prompt"], r["att"], min(r["ideal"], r["prompt"])
        prompt += p
        att += a
        kinds[r["kind"]] += 1
        cats["new"] += p - max(ideal, a)
        d = r.get("deeper")
        lo = a
        if d and d["rows"] > a:
            hi = min(d["rows"], max(ideal, a))
            cause = d["state"]
            if d["state"] == "dropped":
                cause += f":{d.get('cause')}/{d.get('trigger')}"
            elif d["state"] == "refused":
                cause += f":{d.get('cause')}"
            elif d.get("host"):
                cause += ":host"
            cats[cause] += hi - lo
            lo = hi
        if ideal > lo:
            src = r.get("ideal_src") or "none"
            tag = f"no_boundary:{src}" + (":inflight" if r.get("ideal_inflight") else "")
            cats[tag] += ideal - lo
    return dict(prompt=prompt, att=att, cached=att / prompt if prompt else 0.0, cats=dict(cats),
                kinds=dict(kinds), n=len(reqs))


# ----------------------------------------------------------------------------- part B


def keyat(h16, rows):
    """Identity of the prefix of `rows` rows (a multiple of 16), None past the chain."""
    i = rows // 16 - 1
    return h16[i] if 0 <= i < len(h16) else None


class Boundary:
    __slots__ = ("key", "rows", "kind", "bytes", "used", "parent", "kids", "held", "blocks",
                 "chunks", "conv", "alive")

    def __init__(self, key, rows, kind, conv):
        self.key, self.rows, self.kind, self.conv = key, rows, kind, conv
        self.bytes = 0
        self.used = 0.0
        self.parent = None
        self.kids = set()
        self.held = 0
        self.blocks = []
        self.chunks = []
        self.alive = True


class BoundaryCache:
    """Policies i / ii / iii / v: radix whole blocks + boundary snapshots."""

    def __init__(self, c, budget, block_rows, chunk=0, retire=True, strict=None, recent=32, share_min=256):
        self.c, self.budget, self.g, self.chunk, self.retire = c, budget, block_rows, chunk, retire
        self.strict_publish = c.strict if strict is None else strict
        self.bounds = {}            # key -> Boundary
        self.blocks = {}            # key(block end) -> refcount (boundaries + running holds)
        self.chunks = {}            # key(chunk end) -> refcount
        self.total = 0
        self.ever = set()           # boundary keys ever published
        self.lru = []               # (used, seq, key) lazy heap
        self.seq = 0
        self.recent = []            # recent prompts' h16 (shared-end detection)
        self.recent_n, self.share_min = recent, share_min
        self.retired = self.evicted = self.strict = 0

    # bytes
    def block_bytes(self):
        return self.g * self.c.full_row

    def private_bytes(self, rows):
        c = self.c
        b = c.scale_row * rows + c.full_row * (rows % self.g)
        if self.chunk:
            b += c.slide_row * (rows % self.chunk)
        else:
            b += c.slide_row * min(rows, c.window)
        return b

    def chunk_keys(self, h16, rows):
        if not self.chunk or not self.c.window:
            return []
        hi = rows - rows % self.chunk
        lo = max(0, rows - self.c.window)
        lo = lo - lo % self.chunk
        return [keyat(h16, e) for e in range(lo + self.chunk, hi + 1, self.chunk)]

    def block_keys(self, h16, rows):
        return [keyat(h16, e) for e in range(self.g, rows - rows % self.g + 1, self.g)]

    # refcounts
    def ref_block(self, k):
        if k not in self.blocks:
            self.blocks[k] = 0
            self.total += self.block_bytes()
        self.blocks[k] += 1

    def unref_block(self, k):
        self.blocks[k] -= 1
        if self.blocks[k] == 0:
            del self.blocks[k]
            self.total -= self.block_bytes()

    def ref_chunk(self, k):
        if k not in self.chunks:
            self.chunks[k] = 0
            self.total += self.chunk * self.c.slide_row
        self.chunks[k] += 1

    def unref_chunk(self, k):
        self.chunks[k] -= 1
        if self.chunks[k] == 0:
            del self.chunks[k]
            self.total -= self.chunk * self.c.slide_row

    def touch(self, b, t):
        b.used = t
        self.seq += 1
        heapq.heappush(self.lru, (t, self.seq, b.key))

    def lookup(self, h16, prompt, upto=None):
        """Deepest retained boundary strictly below the prompt end on its chain."""
        hi = min(prompt - 1, len(h16) * 16) if upto is None else upto
        r = hi - hi % 32
        while r > 0:
            b = self.bounds.get(keyat(h16, r))
            if b is not None:
                return b
            r -= 32
        return None

    def free(self, b, why):
        if not b.alive:
            return
        b.alive = False
        del self.bounds[b.key]
        self.total -= b.bytes
        for k in b.blocks:
            self.unref_block(k)
        for k in b.chunks:
            self.unref_chunk(k)
        p = b.parent
        if p is not None and p.alive:
            p.kids.discard(b)
            for kid in b.kids:
                kid.parent = p
                p.kids.add(kid)
        else:
            for kid in b.kids:
                kid.parent = None
        if why == "evict":
            self.evicted += 1

    def publish(self, h16, rows, kind, conv, t, held_blocks):
        key = keyat(h16, rows)
        if key is None or rows <= 0:
            return None
        b = self.bounds.get(key)
        if b is not None:
            if kind == "shared":
                b.kind = "shared"
            self.touch(b, t)
            return b
        bk = self.block_keys(h16, rows)
        if self.strict_publish:
            # FP8 strict publish: whole blocks below `rows` another publish made (not attached
            # by this request) are dropped with their boundaries, else the publish is refused.
            m = 0
            while m < len(bk) and bk[m] in self.blocks:
                m += 1
            if m > held_blocks:
                orphan = set(bk[held_blocks:m])
                victims = [x for x in self.bounds.values() if orphan.intersection(x.blocks)]
                if any(x.held for x in victims):
                    return None
                for x in victims:
                    self.strict += 1
                    self.free(x, "strict")
        b = Boundary(key, rows, kind, conv)
        b.bytes = self.private_bytes(rows)
        self.total += b.bytes
        b.blocks = bk
        for k in bk:
            self.ref_block(k)
        b.chunks = self.chunk_keys(h16, rows)
        for k in b.chunks:
            self.ref_chunk(k)
        parent = self.lookup(h16, rows, upto=rows - 1)
        b.parent = parent
        if parent is not None:
            parent.kids.add(b)
        self.bounds[key] = b
        self.ever.add(key)
        self.touch(b, t)
        self.trim()
        if (self.retire and b.alive and parent is not None and parent.alive
                and parent.kind == "prompt_end" and parent.kids == {b} and parent.held == 0):
            self.retired += 1
            self.free(parent, "retire")
        return b

    def trim(self):
        held = []
        while self.total > self.budget and self.lru:
            entry = heapq.heappop(self.lru)
            b = self.bounds.get(entry[2])
            if b is None or b.used != entry[0]:
                continue
            if b.held:
                held.append(entry)
                continue
            self.free(b, "evict")
        for entry in held:
            heapq.heappush(self.lru, entry)

    def shared_rows(self, h16, prompt, attached):
        ph = h16[: prompt // 16]
        best = 0
        for o in self.recent:
            n = 0
            for x, y in zip(o, ph):
                if x != y:
                    break
                n += 1
            if n < len(o) and n < len(ph):
                best = max(best, n * 16)
        self.recent.append(ph)
        if len(self.recent) > self.recent_n:
            self.recent.pop(0)
        rows = best - best % 32
        return rows if rows >= attached + self.share_min and rows < prompt else 0

    # requests
    def attach(self, r, t):
        h16, P = r["h16"], r["prompt"]
        b = self.lookup(h16, P)
        att = b.rows if b else 0
        if b:
            self.touch(b, t)
            a = b.parent
            while a is not None:
                a.used = t
                self.seq += 1
                heapq.heappush(self.lru, (t, self.seq, a.key))
                a = a.parent
        held = att // self.g
        st = dict(att=att, attached=b, hold=[], deeper_ever=False)
        s = self.shared_rows(h16, P, att)
        if s:
            if b:
                b.held += 1
            self.publish(h16, s, "shared", r["conv"], t, held)
            if b:
                b.held -= 1
        # The publisher's own attach hold ends inside its prompt-end publish, before the retire.
        end = (P - 9) // 32 * 32
        if end > att:
            self.publish(h16, end, "prompt_end", r["conv"], t, held)
        # The request maps its prompt's published whole blocks until it finishes.
        for k in self.block_keys(h16, min(P, len(h16) * 16)):
            if k in self.blocks:
                self.ref_block(k)
                st["hold"].append(k)
        st["end"] = end
        top = min(P - 1, len(h16) * 16)
        r_ = top - top % 32
        while r_ > att:
            if keyat(h16, r_) in self.ever:
                st["deeper_ever"] = True
                break
            r_ -= 32
        return st

    def finish(self, r, st, t):
        h16, end = r["h16"], st["end"]
        b = self.bounds.get(keyat(h16, end)) if end > 0 else None
        if b is not None:
            self.touch(b, t)
        elif end > st["att"]:
            # The turn-end pass restores an evicted prompt end while the rings still hold it.
            self.publish(h16, end, "prompt_end", r["conv"], t, end // self.g)
        for k in st["hold"]:
            self.unref_block(k)
        self.trim()

    def probe(self, h16, prompt):
        b = self.lookup(h16, prompt)
        return b.rows if b else 0


class HybridCache:
    """Policy iv: vLLM-like hybrid allocator, 16-row blocks per layer group, LRU per block."""

    def __init__(self, c, budget):
        self.c, self.budget = c, budget
        self.full = {}   # key -> [refs, used]
        self.slide = {}
        self.total = 0
        self.lru = []
        self.seq = 0
        self.ever = set()
        self.fb = 16 * (c.full_row + c.scale_row)
        self.sb = 16 * c.slide_row
        self.wb = (c.window + 15) // 16 if c.window else 0

    def _add(self, d, k, size, t, hold, rank, live=False):
        e = d.get(k)
        if e is None:
            # A running request's sliding window is its live state (plowrt's live ring), not
            # cache: charged once released.
            e = d[k] = [0, t, not live]
            if not live:
                self.total += size
        elif not live and not e[2]:
            e[2] = True
            self.total += size
        if hold:
            e[0] += 1
        e[1] = (t, rank)
        self.seq += 1
        heapq.heappush(self.lru, (e[1], self.seq, d is self.full, k))
        self.ever.add(k)

    def hit(self, h16, prompt):
        lim = min((prompt - 1) // 16, len(h16))
        n = 0
        while n < lim and h16[n] in self.full:
            n += 1
        if not self.wb:
            return n
        while n > 0:
            lo = max(0, n - self.wb)
            if all(h16[i] in self.slide for i in range(lo, n)):
                return n
            n -= 1
        return 0

    def attach(self, r, t):
        h16, P = r["h16"], r["prompt"]
        n = self.hit(h16, P)
        pb = min(P // 16, len(h16))
        held = []
        for i in range(pb):
            # Blocks freed later in a request are evicted first (vLLM frees in reverse order).
            self._add(self.full, h16[i], self.fb, t, True, -i)
            held.append((True, h16[i]))
            in_window = i >= pb - self.wb
            self._add(self.slide, h16[i], self.sb, t, in_window, -i, live=in_window)
            if in_window:
                held.append((False, h16[i]))
        self.trim()
        top = min((P - 1) // 16, len(h16))
        deeper = any(h16[i] in self.ever for i in range(n, top)) if top > n else False
        return dict(att=16 * n, held=held, deeper_ever=deeper)

    def finish(self, r, st, t):
        h16 = r["h16"]
        for full, k in st["held"]:
            d = self.full if full else self.slide
            e = d.get(k)
            if e is not None:
                e[0] -= 1
        n = len(h16)
        for i in range(n):
            self._add(self.full, h16[i], self.fb, t, False, -i)
            self._add(self.slide, h16[i], self.sb, t, False, -i)
        self.trim()

    def trim(self):
        while self.total > self.budget and self.lru:
            used, _, full, k = heapq.heappop(self.lru)
            d = self.full if full else self.slide
            e = d.get(k)
            if e is None or e[1] != used or e[0] > 0:
                continue
            del d[k]
            if e[2]:
                self.total -= self.fb if full else self.sb

    def probe(self, h16, prompt):
        return 16 * self.hit(h16, prompt)


def conversations(reqs):
    by_rid = {r["rid"]: r for r in reqs}
    nxt = 0
    for r in reqs:
        q = by_rid.get(r.get("ideal_rid"))
        if q is not None and "conv" in q and r["ideal"] >= q["prompt"]:
            r["conv"] = q["conv"]
        else:
            r["conv"] = nxt
            nxt += 1
    return nxt


def replay(reqs, make, t0, t1, sample_s=10.0, active_s=300.0):
    cache = make()
    evs = []
    for i, r in enumerate(reqs):
        evs.append((r["t_att"], 0, i))
        evs.append((r.get("t_fin") or r["t_att"], 1, i))
    evs.sort()
    state = {}
    res = dict(prompt=0, att=0, ideal=0, deeper_lost=0, n=0, misses=0, lost_evicted=0,
               lost_unpublished=0)
    last = {}            # conv -> (t_fin, request)
    next_sample = t0
    byte_int = conv_int = 0.0
    for t, kind, i in evs:
        r = reqs[i]
        while next_sample <= min(t, t1):
            alive = [q for (tf, q) in last.values() if next_sample - tf <= active_s]
            ret = sum(1 for q in alive if cache.probe(q["h16"], q["prompt"]) >= q["prompt"] - 64)
            if next_sample >= t0:
                byte_int += cache.total
                conv_int += ret
            next_sample += sample_s
        if kind == 0:
            st = cache.attach(r, t)
            state[i] = st
            if t0 <= t < t1:
                ideal = min(r["ideal"], r["prompt"])
                a = st["att"]
                res["n"] += 1
                res["prompt"] += r["prompt"]
                res["att"] += a
                res["ideal"] += ideal
                res["misses"] += a == 0
                gap = max(0, ideal - a)
                if st["deeper_ever"]:
                    res["lost_evicted"] += gap
                else:
                    res["lost_unpublished"] += gap
        else:
            cache.finish(r, state.pop(i), t)
            last[r["conv"]] = (t, r)
    res["cached"] = res["att"] / res["prompt"] if res["prompt"] else 0.0
    res["bytes_per_conv"] = byte_int / conv_int if conv_int else None
    res["mean_bytes"] = byte_int / max(1, (t1 - t0) / sample_s)
    for k in ("retired", "evicted", "strict"):
        if hasattr(cache, k):
            res[k] = getattr(cache, k)
    return res


def policies(c, budget):
    g0 = c.block_rows
    out = [("i", lambda: BoundaryCache(c, budget, g0))]
    for ch in (32, 64):
        out.append((f"ii-{ch}", lambda ch=ch: BoundaryCache(c, budget, g0, chunk=ch, retire=False)))
    # Finer full-attention blocks assume FP8 scales co-owned with their blocks (no strict
    # publish); `iii-64-strict` keeps today's rule, under which another conversation's
    # system-prompt blocks orphan every later publish over them.
    for g in (16, 64, 256, 1024):
        out.append((f"iii-{g}", lambda g=g: BoundaryCache(c, budget, g, strict=False)))
    out.append(("iii-64-strict", lambda: BoundaryCache(c, budget, 64)))
    out.append(("v-64-64", lambda: BoundaryCache(c, budget, 64, chunk=64, retire=False, strict=False)))
    out.append(("iv", lambda: HybridCache(c, budget)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--skip-s", type=float, default=90.0, help="warmup excluded from the counts")
    ap.add_argument("--until-s", type=float, default=None)
    ap.add_argument("--budget-mib", type=float, default=None, help="default: median traced cache_bytes")
    ap.add_argument("--scale", default="0.5,1,2", help="budget multiples to report for i and iv")
    ap.add_argument("--only", default=None, help="comma-separated policy names")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    header, reqs, other = load(a.trace)
    reqs = [r for r in reqs if r.get("h16") is not None and not r.get("unwound")]
    c = Costs(header)
    if not reqs:
        sys.exit("no request records")
    t_first = reqs[0]["t_att"]
    t0 = t_first + a.skip_s
    t1 = a.until_s + t_first if a.until_s else max(r["t_att"] for r in reqs) + 1e-3
    win = [r for r in reqs if t0 <= r["t_att"] < t1]
    nconv = conversations(reqs)
    cb = sorted(r["cache_bytes"] for r in win) or [0]
    budget = a.budget_mib * MiB if a.budget_mib else statistics.median(cb)
    out = dict(header=header, window=[t0 - t_first, t1 - t_first], requests=len(win), conversations=nconv,
               dropped_events=other.get("dropped_events", 0),
               budget_mib=budget / MiB, cache_mib_p10_p50_p90=[cb[len(cb) // 10] / MiB, statistics.median(cb) / MiB,
                                                                 cb[len(cb) * 9 // 10] / MiB])
    A = attribute(win)
    out["traced"] = A
    print(f"trace {a.trace}: {len(win)} requests in [{t0 - t_first:.0f}, {t1 - t_first:.0f}) s, "
          f"{nconv} conversations, dropped events {out['dropped_events']}")
    print(f"pool: block_rows {c.block_rows} full {c.full_row}+{c.scale_row} B/row, sliding "
          f"{c.slide_row} B/row x window {c.window}; budget {budget / MiB:.0f} MiB "
          f"(traced cache p10/p50/p90 {out['cache_mib_p10_p50_p90'][0]:.0f}/"
          f"{out['cache_mib_p10_p50_p90'][1]:.0f}/{out['cache_mib_p10_p50_p90'][2]:.0f} MiB)")
    print(f"\nA. traced run: cached {A['cached']:.4f} ({A['att']}/{A['prompt']}), kinds "
          + ", ".join(f"{k} {v / A['n']:.3f}" for k, v in sorted(A["kinds"].items())))
    lost = A["prompt"] - A["att"]
    for k, v in sorted(A["cats"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:42s} {v:>11d} tok  {v / A['prompt']:.4f} of prompt  {v / max(1, lost):.3f} of lost")
    names = set(a.only.split(",")) if a.only else None
    rows = []
    for name, make in policies(c, budget):
        if names and name not in names:
            continue
        r = replay(reqs, make, t0, t1)
        inf = replay(reqs, dict(policies(c, float("inf")))[name], t0, t1)
        r["cached_inf"] = inf["cached"]
        r["structural"] = (inf["ideal"] - inf["att"]) / inf["prompt"]
        r["capacity"] = (inf["att"] - r["att"]) / r["prompt"]
        r["new"] = (r["prompt"] - r["ideal"]) / r["prompt"]
        rows.append((name, r))
        out[name] = r
    if rows:
        print(f"\nB. shadow replay (budget {budget / MiB:.0f} MiB): cached = attached/prompt; "
              "loss columns are fractions of prompt tokens")
        print(f"   {'policy':8s} {'cached':>7s} {'unbounded':>9s} {'new':>6s} {'struct':>7s} {'capac':>7s} "
              f"{'miss':>6s} {'MiB/conv':>9s} {'meanMiB':>8s}")
        for name, r in rows:
            bpc = r["bytes_per_conv"]
            print(f"   {name:8s} {r['cached']:7.4f} {r['cached_inf']:9.4f} {r['new']:6.3f} {r['structural']:7.3f} "
                  f"{r['capacity']:7.3f} {r['misses'] / max(1, r['n']):6.3f} "
                  f"{(bpc / MiB if bpc else float('nan')):9.1f} {r['mean_bytes'] / MiB:8.0f}")
    scales = [float(x) for x in a.scale.split(",") if x]
    sens = {}
    for name in ("i", "iv", "ii-64", "iii-64"):
        if names and name not in names:
            continue
        sens[name] = {}
        for s in scales:
            if s == 1:
                continue
            make = dict(policies(c, budget * s))[name]
            sens[name][s] = replay(reqs, make, t0, t1)["cached"]
    if sens:
        print("\n   budget sensitivity (cached): " + "; ".join(
            f"{n} " + " ".join(f"x{s:g} {v:.4f}" for s, v in d.items()) for n, d in sens.items()))
        out["sensitivity"] = sens
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
