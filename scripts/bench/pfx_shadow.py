#!/usr/bin/env python3
"""Prefix-cache shadow replay over a `PLOW_PFX_TRACE` trace.

    pfx_shadow.py <trace.jsonl> [--skip-s 90] [--until-s T] [--budget-mib M] [--json out.json]

Part A attributes every lost prompt token of the traced run (what plowrt did) by an explicit
precedence per request: `[att, deeper)` to the fate of the deepest boundary that would have
attached deeper (dropped by mechanism/trigger, refused, live), `[deeper, ideal_ready)` to "no
boundary" (by whether the ideal match ended in an earlier prompt or only in an output),
`[ideal_ready, ideal)` to history seen only in requests still in flight, `[ideal, prompt)` to
tokens never seen before.

Part B replays the same requests against model cache policies under one device budget (default:
the traced pool's median `cache_bytes`) with the trace header's byte costs and publication rules.
A request attaches at `t_att`; its KV becomes reusable (boundaries published, blocks inserted)
at `t_ready` = its first traced publish/refusal (else `t_att` + uncached rows at 25K rows/s);
it finishes at `t_fin`. Each prefix is named by the 16-row hash chain `h16`.

  i       plowrt today: whole full-attention blocks of `block_rows` shared through the radix,
          private boundary snapshots (sliding window + fp8 full-scale prefix + sub-block tail)
          at prompt ends and shared-prefix ends, the boundary tree's retire rule (one retained
          continuation, no holder, no pending continuation, no unsuperseded prompt end, not a
          shared end), FP8 strict publish (orphaned blocks dropped, else refused), LRU of
          boundary + the blocks only it kept, turn-end touch/restore of the prompt end.
  ii-C    i with the sliding window held as shared refcounted C-row chunks; ii-C-keep also
          keeps parents (one costs only the chunks it does not share) instead of retiring them.
  iii-G   i with G-row full-attention blocks, FP8 scales co-owned with the blocks (no strict
          publish); iii-64-strict keeps today's strict rule and private scale prefix.
  v-C-G   ii-C and iii-G together.
  iv      vLLM-like hybrid allocator: 16-row blocks for both groups, a hit needs every full
          block of the prefix and the sliding blocks of its last window, LRU per block (a
          request's tail first), generated blocks cached too; a running request's own sliding
          window is live state (plowrt's live ring), charged once released.

Each policy also runs with an unbounded budget: `struct` = (ideal - attached at infinity) /
prompt: what the policy cannot reuse at any size (granularity, publication rules, uncached
outputs, the last rows a prompt end backs off); `budget` = (attached at infinity - attached) /
prompt: a signed counterfactual (hit-dependent publication makes the policies non-monotone).
Bytes per retained conversation = the integral of cached bytes over the integral of
conversations whose last prompt is attachable to within 64 rows of its end (sampled every 10 s
over conversations active in the last 300 s); a conversation is a chain of requests whose ideal
match covers the previous request's whole prompt (`ideal_rid`). Request times are held fixed: a
policy that hits more would also run faster, which the replay does not model.
"""
import argparse
import heapq
import json
import statistics
import sys
from collections import defaultdict

MiB = 1 << 20
PREFILL_ROWS_PER_S = 25000.0


def load(path):
    header, reqs, other = None, [], defaultdict(int)
    ends = defaultdict(list)  # rid -> [(t, rows, prompt-end publish)] of its publishes / refusals
    t_last = 0.0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ev = r.get("ev")
            t_last = max(t_last, r.get("t") or 0.0, r.get("t_att") or 0.0, r.get("t_fin") or 0.0)
            if ev == "header":
                header = r
            elif ev == "req":
                reqs.append(r)
            else:
                other[ev] += 1
                if ev == "stats":
                    other["dropped_events"] = max(other["dropped_events"], r.get("dropped_events", 0))
                if ev in ("pub", "refuse") and r.get("rid") is not None:
                    ends[r["rid"]].append((r["t"], r.get("rows") or 0, ev == "pub" and bool((r.get("kind") or 0) & 1)))
    if header is None:
        sys.exit("no header record: was the trace armed at pool bring-up?")
    backoff = header.get("prompt_end_backoff", 9) + header.get("boundary_quantum", 32)
    for r in reqs:
        other["dropped_events"] = max(other["dropped_events"], r.get("dropped_events", 0))
        est = r["t_att"] + (r["prompt"] - r["att"]) / PREFILL_ROWS_PER_S
        # Ready once its own prompt end publishes (or is refused there): a shared end or a
        # checkpoint publishes partway through the prefill.
        at_end = [t for t, rows, pe in ends.get(r["rid"], ()) if pe or rows >= r["prompt"] - backoff]
        t_ready = min(at_end) if at_end else est
        # Still in flight at shutdown: no finish; it holds its rows to the end of the trace.
        r["unfinished"] = r.get("t_fin") is None
        t_fin = max(t_last, t_ready) if r["unfinished"] else r["t_fin"]
        r["t_ready"] = min(t_ready, t_fin)
        r["t_fin"] = max(t_fin, r["t_ready"])
    reqs.sort(key=lambda r: r["t_att"])
    return header, reqs, other


class Costs:
    def __init__(self, h):
        self.full_row = h["full_layers"] * 2 * h["kvh_full"] * h["hd_full"] * h["elem_full"]
        self.scale_row = h["full_layers"] * 2 * h["kvh_full"] * 4 if h["full_scale"] else 0
        self.slide_row = h["snap_row_bytes"]
        self.window = h["window"] if h["slide_layers"] else 0
        self.block_rows = h["block_rows"]
        self.strict = bool(h.get("strict_publish", h["full_scale"]))
        self.quantum = h.get("boundary_quantum", 32)
        self.backoff = h.get("prompt_end_backoff", 9)
        self.share_min = h.get("share_min_rows", 256)
        self.recent = h.get("recent_prompts", 32)
        self.recent_rows = h.get("recent_rows", 16384)
        # Rows past a prompt end the sliding rings still hold its window for (0: no sliding layers).
        self.ring_slack = max(0, h.get("ring", 0) - h["window"]) if h["slide_layers"] else None


# ----------------------------------------------------------------------------- part A


def attribute(reqs):
    """Lost tokens of the traced run by cause (precedence: deeper boundary fate, no boundary,
    in-flight history, never seen)."""
    cats = defaultdict(int)
    kinds = defaultdict(int)
    prompt = att = 0
    for r in reqs:
        p, a = r["prompt"], r["att"]
        ideal = min(r["ideal"], p)
        ready = min(r.get("ideal_ready", ideal), ideal)
        prompt += p
        att += a
        kinds[r["kind"] + (":unwound" if r.get("unwound") else "")] += 1
        cats["new"] += p - max(ideal, a)
        lo = a
        d = r.get("deeper")
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
        if ready > lo:
            cats[f"no_boundary:{r.get('ideal_src') or 'none'}"] += ready - lo
            lo = ready
        if ideal > lo:
            cats["inflight_history"] += ideal - lo
    return dict(prompt=prompt, att=att, cached=att / prompt if prompt else 0.0, cats=dict(cats),
                kinds=dict(kinds), n=len(reqs))


# ----------------------------------------------------------------------------- part B


def keyat(h16, rows):
    """Identity of the prefix of `rows` rows (a multiple of 16), None past the chain."""
    i = rows // 16 - 1
    return h16[i] if 0 <= i < len(h16) else None


class Boundary:
    __slots__ = ("key", "rows", "kind", "bytes", "used", "parent", "kids", "users", "pending",
                 "ends", "blocks", "chunks", "alive")

    def __init__(self, key, rows, kind):
        self.key, self.rows, self.kind = key, rows, kind
        self.bytes = 0
        self.used = 0.0
        self.parent = None
        self.kids = set()
        self.users = self.pending = self.ends = 0
        self.blocks = []
        self.chunks = []
        self.alive = True


class BoundaryCache:
    """Policies i / ii / iii / v: radix whole blocks + boundary snapshots."""

    def __init__(self, c, budget, block_rows, chunk=0, retire=True, strict=None):
        self.c, self.budget, self.g, self.chunk, self.retire = c, budget, block_rows, chunk, retire
        self.strict = c.strict if strict is None else strict
        # Without strict publish the FP8 scales are co-owned by the blocks they scale.
        self.block_row = c.full_row + (0 if self.strict else c.scale_row)
        self.bounds = {}            # key -> Boundary
        self.blocks = {}            # key(block end) -> refcount (boundaries + running maps)
        self.chunks = {}            # key(chunk end) -> refcount
        self.total = 0
        self.ever = set()           # boundary keys ever published
        self.lru = []               # (used, seq, key) lazy heap
        self.seq = 0
        self.recent = []            # recent prompts' h16 heads (shared-end detection)
        self.retired = self.evicted = self.strict_drops = self.refused = 0

    # bytes
    def private_bytes(self, rows):
        c = self.c
        tail = rows % self.g
        b = c.full_row * tail + c.scale_row * (rows if self.strict else tail)
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

    def ref_block(self, k):
        if k not in self.blocks:
            self.blocks[k] = 0
            self.total += self.g * self.block_row
        self.blocks[k] += 1

    def unref_block(self, k):
        self.blocks[k] -= 1
        if self.blocks[k] == 0:
            del self.blocks[k]
            self.total -= self.g * self.block_row

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
        q = self.c.quantum
        hi = min(prompt - 1, len(h16) * 16) if upto is None else upto
        r = hi - hi % q
        while r > 0:
            b = self.bounds.get(keyat(h16, r))
            if b is not None:
                return b
            r -= q
        return None

    def try_retire(self, b):
        if (self.retire and b is not None and b.alive and len(b.kids) == 1 and b.users == 0
                and b.pending == 0 and b.ends == 0 and b.kind != "shared"):
            self.retired += 1
            self.free(b)

    def free(self, b, evicted=False):
        if not b.alive:
            return
        b.alive = False
        del self.bounds[b.key]
        self.total -= b.bytes
        for k in b.blocks:
            self.unref_block(k)
        for k in b.chunks:
            self.unref_chunk(k)
        g = b.parent if b.parent is not None and b.parent.alive else None
        for kid in b.kids:
            kid.parent = g
            if g is not None:
                g.kids.add(kid)
        if g is not None:
            g.kids.discard(b)
            self.try_retire(g)
        if evicted:
            self.evicted += 1

    def publish(self, h16, rows, kind, t, held_blocks):
        """Publish (or touch) the boundary at `rows`; returns it, or None when refused."""
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
        if self.strict:
            # FP8 strict publish: whole blocks below `rows` that another publish made (this
            # request did not attach them) go with their boundaries; still there (a running
            # request maps them) = refused.
            def present():
                m = 0
                while m < len(bk) and bk[m] in self.blocks:
                    m += 1
                return m
            m = present()
            if m > held_blocks:
                orphan = set(bk[held_blocks:m])
                for x in [x for x in self.bounds.values() if orphan.intersection(x.blocks)]:
                    if x.users == 0:
                        self.strict_drops += 1
                        self.free(x)
                if present() > held_blocks:
                    self.refused += 1
                    return None
        b = Boundary(key, rows, kind)
        b.bytes = self.private_bytes(rows)
        self.total += b.bytes
        b.blocks = bk
        for k in bk:
            self.ref_block(k)
        b.chunks = self.chunk_keys(h16, rows)
        for k in b.chunks:
            self.ref_chunk(k)
        b.parent = self.lookup(h16, rows, upto=rows - 1)
        if b.parent is not None:
            b.parent.kids.add(b)
        self.bounds[key] = b
        self.ever.add(key)
        self.touch(b, t)
        return b

    def note_end(self, b):
        """A request's prompt ended at `b`: the nearest ancestor an earlier prompt ended at is
        superseded once."""
        b.ends += 1
        a = b.parent
        while a is not None:
            if a.ends > 0:
                a.ends -= 1
                return a if a.ends == 0 else None
            a = a.parent
        return None

    def trim(self):
        held = []
        while self.total > self.budget and self.lru:
            entry = heapq.heappop(self.lru)
            b = self.bounds.get(entry[2])
            if b is None or b.used != entry[0]:
                continue
            if b.users:
                held.append(entry)
                continue
            self.free(b, evicted=True)
        for entry in held:
            heapq.heappush(self.lru, entry)

    def shared_rows(self, h16, prompt, attached):
        ph = h16[: min(prompt, self.c.recent_rows) // 16]
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
        if len(self.recent) > self.c.recent:
            self.recent.pop(0)
        q = self.c.quantum
        rows = best - best % q
        return rows if rows >= attached + self.c.share_min and rows < prompt else 0

    # requests
    def attach(self, r, t):
        h16, P = r["h16"], r["prompt"]
        b = self.lookup(h16, P)
        att = b.rows if b else 0
        st = dict(att=att, b=b, hold=[], deeper_ever=False, end=0)
        if b:
            b.users += 1
            b.pending += 1
            a = b
            while a is not None:
                self.touch(a, t)
                a = a.parent
        for k in self.block_keys(h16, att):
            self.ref_block(k)
            st["hold"].append(k)
        st["share"] = self.shared_rows(h16, P, att)
        q = self.c.quantum
        top = min(P - 1, len(h16) * 16)
        r_ = top - top % q
        while r_ > att:
            if keyat(h16, r_) in self.ever:
                st["deeper_ever"] = True
                break
            r_ -= q
        return st

    @staticmethod
    def ancestors(b):
        a, out = b.parent, []
        while a is not None:
            out.append(a)
            a = a.parent
        return out

    def ready(self, r, st, t):
        h16, P, att, b = r["h16"], r["prompt"], st["att"], st["b"]
        held = att // self.g
        if st["share"] and self.publish(h16, st["share"], "shared", t, held) is not None:
            held = max(held, st["share"] // self.g)
        end = (P - self.c.backoff) // self.c.quantum * self.c.quantum
        st["end"] = end
        c = None
        if end > att:
            c = self.publish(h16, end, "prompt_end", t, held)
            if c is not None:
                held = max(held, end // self.g)
        elif b is not None and end == att:
            c = b
        st["owned"] = held
        superseded = self.note_end(c) if c is not None and c.alive else None
        if b is not None:
            b.users -= 1
            # The prompt end continues the attached boundary: its pending continuation is done.
            # Without one it stays pending, as `Snap::pending` does (only LRU takes it then).
            if c is not None and (c is b or b in self.ancestors(c)):
                b.pending -= 1
        # The request maps its prompt's published whole blocks until it finishes.
        have = set(st["hold"])
        for k in self.block_keys(h16, min(P, len(h16) * 16)):
            if k in self.blocks and k not in have:
                self.ref_block(k)
                st["hold"].append(k)
        self.trim()
        if c is not None and c.alive:
            self.try_retire(c.parent)
            self.try_retire(superseded)
            self.try_retire(b)

    def finish(self, r, st, t):
        h16, end = r["h16"], st["end"]
        b = self.bounds.get(keyat(h16, end)) if end > 0 else None
        if b is not None:
            a = b
            while a is not None:
                self.touch(a, t)
                a = a.parent
        elif end > st["att"] and (self.c.ring_slack is None
                                   or r["prompt"] + (r.get("out") or 0) - end <= self.c.ring_slack):
            # The turn-end pass restores an evicted prompt end while the rings still hold it.
            self.publish(h16, end, "prompt_end", t, st.get("owned", st["att"] // self.g))
        for k in st["hold"]:
            self.unref_block(k)
        self.trim()

    def probe(self, h16, prompt):
        b = self.lookup(h16, prompt)
        return b.rows if b else 0

    def retained(self, h16, prompt):
        """The prompt's own end boundary is attachable."""
        end = (prompt - self.c.backoff) // self.c.quantum * self.c.quantum
        return end > 0 and keyat(h16, end) in self.bounds


class HybridCache:
    """Policy iv: vLLM-like hybrid allocator, 16-row blocks per layer group, LRU per block."""

    def __init__(self, c, budget):
        self.c, self.budget = c, budget
        self.full = {}   # key -> [refs, used, charged]
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
        e[1] = max(e[1], (t, rank)) if isinstance(e[1], tuple) else (t, rank)
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
        held = []
        # The hit's blocks are in use from now: full blocks of the prefix, the window's slides.
        for i in range(n):
            self._add(self.full, h16[i], self.fb, t, True, -i)
            held.append((True, h16[i]))
        for i in range(max(0, n - self.wb), n):
            self._add(self.slide, h16[i], self.sb, t, True, -i)
            held.append((False, h16[i]))
        top = min((P - 1) // 16, len(h16))
        deeper = any(h16[i] in self.ever for i in range(n, top)) if top > n else False
        return dict(att=16 * n, n=n, held=held, deeper_ever=deeper, t=t)

    def ready(self, r, st, t):
        h16, P = r["h16"], r["prompt"]
        pb = min(P // 16, len(h16))
        for i in range(st["n"], pb):
            self._add(self.full, h16[i], self.fb, t, True, -i)
            st["held"].append((True, h16[i]))
            in_window = i >= pb - self.wb
            self._add(self.slide, h16[i], self.sb, t, in_window, -i, live=in_window)
            if in_window:
                st["held"].append((False, h16[i]))
        st["t"] = t
        self.trim()

    def finish(self, r, st, t):
        h16 = r["h16"]
        for full, k in st["held"]:
            d = self.full if full else self.slide
            e = d.get(k)
            if e is not None:
                e[0] -= 1
                # A held block's heap entry may have been dropped by a trim: requeue it.
                self.seq += 1
                heapq.heappush(self.lru, (e[1], self.seq, full, k))
        n = len(h16)
        # Sliding blocks behind the final window were freed while the request ran (they keep
        # their earlier recency, and a live one is charged now); the window and every full
        # block are freed at the finish.
        for i in range(n):
            self._add(self.full, h16[i], self.fb, t, False, -i)
            self._add(self.slide, h16[i], self.sb, t if i >= n - self.wb else st["t"], False, -i)
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

    def retained(self, h16, prompt):
        """Every block a hit on the whole prompt can use is cached."""
        lim = min((prompt - 1) // 16, len(h16))
        return lim > 0 and self.hit(h16, prompt) == lim


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
        evs.append((r["t_ready"], 1, i))
        evs.append((r["t_fin"], 2, i))
    evs.sort()
    state = {}
    res = dict(prompt=0, att=0, ideal=0, n=0, misses=0, lost_evicted=0, lost_unpublished=0)
    last = {}            # conv -> (t_fin, request)
    next_sample = t0
    byte_int = conv_int = 0.0
    samples = 0
    for t, kind, i in evs:
        r = reqs[i]
        while next_sample <= min(t, t1):
            ret = 0
            for tf, q in last.values():
                if next_sample - tf <= active_s:
                    ret += cache.retained(q["h16"], q["prompt"])
            byte_int += cache.total
            conv_int += ret
            samples += 1
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
                res["lost_evicted" if st["deeper_ever"] else "lost_unpublished"] += gap
        elif kind == 1:
            cache.ready(r, state[i], t)
        else:
            cache.finish(r, state.pop(i), t)
            last[r["conv"]] = (t, r)
    res["cached"] = res["att"] / res["prompt"] if res["prompt"] else 0.0
    res["bytes_per_conv"] = byte_int / conv_int if conv_int else None
    res["mean_bytes"] = byte_int / max(1, samples)
    res["mean_retained_convs"] = conv_int / max(1, samples)
    for k in ("retired", "evicted", "strict_drops", "refused"):
        if hasattr(cache, k):
            res[k] = getattr(cache, k)
    return res


def policies(c, budget):
    g0 = c.block_rows
    out = [("i", lambda: BoundaryCache(c, budget, g0))]
    for ch in (32, 64):
        out.append((f"ii-{ch}", lambda ch=ch: BoundaryCache(c, budget, g0, chunk=ch)))
        out.append((f"ii-{ch}-keep", lambda ch=ch: BoundaryCache(c, budget, g0, chunk=ch, retire=False)))
    for g in (16, 64, 256, 1024):
        out.append((f"iii-{g}", lambda g=g: BoundaryCache(c, budget, g, strict=False)))
    out.append(("iii-64-strict", lambda: BoundaryCache(c, budget, 64)))
    out.append(("v-64-64", lambda: BoundaryCache(c, budget, 64, chunk=64, strict=False)))
    out.append(("iv", lambda: HybridCache(c, budget)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--skip-s", type=float, default=90.0, help="warmup excluded from the counts")
    ap.add_argument("--until-s", type=float, default=None)
    ap.add_argument("--budget-mib", type=float, default=None, help="default: median traced cache_bytes")
    ap.add_argument("--scale", default="0.5,2", help="budget multiples for the sensitivity line")
    ap.add_argument("--only", default=None, help="comma-separated policy names")
    ap.add_argument("--allow-host", action="store_true", help="replay a host-tier trace device-only")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    header, reqs, other = load(a.trace)
    if header.get("host_cap") and not a.allow_host:
        sys.exit("trace has a host tier (host_cap > 0), which the replay does not model; --allow-host to "
                 "replay it device-only")
    c = Costs(header)
    if not reqs:
        sys.exit("no request records")
    t_first = reqs[0]["t_att"]
    t0 = t_first + a.skip_s
    t1 = a.until_s + t_first if a.until_s is not None else max(r["t_att"] for r in reqs) + 1e-3
    win = [r for r in reqs if t0 <= r["t_att"] < t1]
    replayed = [r for r in reqs if r.get("h16") is not None]
    nconv = conversations(reqs)
    cb = sorted(r["cache_bytes"] for r in win) or [0]
    budget = a.budget_mib * MiB if a.budget_mib is not None else statistics.median(cb)
    out = dict(header=header, window=[t0 - t_first, t1 - t_first], requests=len(win), conversations=nconv,
               dropped_events=other.get("dropped_events", 0), budget_mib=budget / MiB,
               cache_mib_p10_p50_p90=[cb[len(cb) // 10] / MiB, statistics.median(cb) / MiB,
                                      cb[len(cb) * 9 // 10] / MiB])
    if not win:
        sys.exit(f"no requests in the window [{t0 - t_first:.0f}, {t1 - t_first:.0f}) s (--skip-s/--until-s)")
    A = attribute(win)
    out["traced"] = A
    print(f"trace {a.trace}: {len(win)} requests in [{t0 - t_first:.0f}, {t1 - t_first:.0f}) s, "
          f"{nconv} conversations, dropped events {out['dropped_events']}")
    if out["dropped_events"]:
        print("   WARNING: events were dropped; the boundary registry (part A) is incomplete")
    print(f"pool: block_rows {c.block_rows} full {c.full_row}+{c.scale_row} B/row, sliding "
          f"{c.slide_row} B/row x window {c.window}, strict {c.strict}; budget {budget / MiB:.0f} MiB "
          f"(traced cache p10/p50/p90 {out['cache_mib_p10_p50_p90'][0]:.0f}/"
          f"{out['cache_mib_p10_p50_p90'][1]:.0f}/{out['cache_mib_p10_p50_p90'][2]:.0f} MiB)")
    print(f"\nA. traced run: cached {A['cached']:.4f} ({A['att']}/{A['prompt']}), kinds "
          + ", ".join(f"{k} {v / max(1, A['n']):.3f}" for k, v in sorted(A["kinds"].items())))
    lost = A["prompt"] - A["att"]
    for k, v in sorted(A["cats"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:44s} {v:>11d} tok  {v / max(1, A['prompt']):.4f} of prompt  {v / max(1, lost):.3f} of lost")
    names = set(a.only.split(",")) if a.only else None
    rows = []
    for name, make in policies(c, budget):
        if names and name not in names:
            continue
        r = replay(replayed, make, t0, t1)
        inf = replay(replayed, dict(policies(c, float("inf")))[name], t0, t1)
        r["cached_inf"] = inf["cached"]
        if not r["prompt"]:
            print(f"   {name}: no replayed prompt tokens in the window (requests without h16?)")
            continue
        r["struct"] = (inf["ideal"] - inf["att"]) / inf["prompt"]
        r["budget_delta"] = (inf["att"] - r["att"]) / r["prompt"]
        r["new"] = (r["prompt"] - r["ideal"]) / r["prompt"]
        rows.append((name, r))
        out[name] = r
    if rows:
        print(f"\nB. shadow replay (budget {budget / MiB:.0f} MiB): cached = attached/prompt; new/struct/"
              "budget are fractions of prompt tokens")
        print(f"   {'policy':13s} {'cached':>7s} {'unbounded':>9s} {'new':>6s} {'struct':>7s} {'budget':>7s} "
              f"{'miss':>6s} {'MiB/conv':>9s} {'meanMiB':>8s} {'convs':>6s}")
        for name, r in rows:
            bpc = r["bytes_per_conv"]
            print(f"   {name:13s} {r['cached']:7.4f} {r['cached_inf']:9.4f} {r['new']:6.3f} {r['struct']:7.3f} "
                  f"{r['budget_delta']:7.3f} {r['misses'] / max(1, r['n']):6.3f} "
                  f"{(bpc / MiB if bpc else float('nan')):9.1f} {r['mean_bytes'] / MiB:8.0f} "
                  f"{r['mean_retained_convs']:6.1f}")
    scales = [float(x) for x in a.scale.split(",") if x]
    sens = {}
    for name in ("i", "ii-64", "iii-64", "iv"):
        if not scales or (names and name not in names):
            continue
        sens[name] = {s: replay(replayed, dict(policies(c, budget * s))[name], t0, t1)["cached"] for s in scales}
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
