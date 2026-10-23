#!/usr/bin/env python3
"""Reuse-distance replay of an agentic_turns plan under LRU-by-bytes prefix-cache retention
(docs/arch/26-kv-block-store.md §1.2).

Insertion models:
  snapshot     every turn inserts a fresh boundary snapshot and retires the previous one (plowrt today)
  incremental  the first turn inserts window + prefix rows, later turns only their new rows; a hit moves
               the whole session to MRU (vLLM's per-token cache, or a block store)
`--ended-first` evicts sessions whose last planned turn has run before any live session: the oracle bound
on what a session-end signal buys. The per-session byte constants are Gemma-4 26B-A4B FP8.

    prefix_retention_sim.py <load.json> [--think 10,20,45] [--rate 1.8,2.7] [--cap-gb 41,101,191] \
        [--model snapshot,incremental] [--ended-first] [--duration 1800]
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_turns as ag  # noqa: E402

WINDOW_BYTES = 1024 * 104_000      # 25 sliding layers, K+V, 8 heads, hd 256 fp8 + ring scales
PREFIX_ROW_BYTES = 10_320          # 5 full layers, K+V, 2 heads, hd 512 fp8 + scales
ROW_BYTES = 114_000                # one new row across both classes
SNAPSHOT_BYTES = 121e6             # measured prod mean boundary snapshot
SHARED_BLOCK_BYTES = 40 * 2**20    # one 4096-row block over 10 tracks x 2 heads


def simulate(plan, cap, model, ended_first, e2e=1.0):
    events = []
    for si, s in enumerate(plan):
        t = s["arrival_s"]
        for k, turn in enumerate(s["turns"]):
            if k:
                t += turn["think_s"]
            events.append((t, si, k, turn))
            t += e2e
    events.sort(key=lambda e: e[0])
    lru, dead, used, tot, hit = OrderedDict(), set(), 0, 0, 0
    for _, si, k, turn in events:
        p = turn["prompt_tokens"]
        tot += p
        present = si in lru
        prev = 0
        if present:
            if k:
                hit += p - turn["user_tokens"]
            prev = lru.pop(si)
            used -= prev
        if model == "snapshot":
            b = SNAPSHOT_BYTES + (p // 4096) * SHARED_BLOCK_BYTES
        elif present:
            b = prev + (turn["user_tokens"] + turn["max_tokens"]) * ROW_BYTES
        else:
            b = WINDOW_BYTES + p * PREFIX_ROW_BYTES
        lru[si] = b
        used += b
        if ended_first and k == len(plan[si]["turns"]) - 1:
            dead.add(si)
        while used > cap and len(lru) > 1:
            victim = next((d for d in lru if d in dead), None) if ended_first else None
            if victim is None:
                _, b = lru.popitem(last=False)
            else:
                b = lru.pop(victim)
                dead.discard(victim)
            used -= b
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("load_json")
    ap.add_argument("--think", default="10,20,45")
    ap.add_argument("--rate", default="1.8,2.7")
    ap.add_argument("--cap-gb", default="41,101,191")
    ap.add_argument("--model", default="snapshot,incremental")
    ap.add_argument("--ended-first", action="store_true")
    ap.add_argument("--duration", type=float, default=1800.0)
    o = ap.parse_args()
    cfg = json.load(open(o.load_json))["config"]
    for think in map(float, o.think.split(",")):
        for rate in map(float, o.rate.split(",")):
            a = argparse.Namespace(**cfg)
            a.rate, a.duration, a.think_median_s = rate, o.duration, think
            if think >= 45:
                a.think_sigma, a.think_max_s = 1.0, 600.0
            elif think >= 20:
                a.think_max_s = 240.0
            _, plan = ag.build_plan(a)
            for model in o.model.split(","):
                for cap in map(float, o.cap_gb.split(",")):
                    print(json.dumps(dict(think=think, rate=rate, req_s=round(rate * 4.0, 1), model=model,
                                          cap_gb=cap, ended_first=o.ended_first,
                                          hit=round(simulate(plan, cap * 1e9, model, o.ended_first), 3))))


if __name__ == "__main__":
    main()
