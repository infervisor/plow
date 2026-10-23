#!/usr/bin/env python3
"""Unbounded-cache prefix-hit ceiling of an agentic_turns open-loop plan (a pure function of the flags).

Turn k's prompt = system + users[:k+1] + replies[:k]; everything before user_k is an exact prefix of
turn k-1's prompt plus reply, and an app's system prompt repeats across sessions.

    prefix_oracle.py <load.json> <rate> <duration_s> [rows_lost_per_attach]

`load.json` is any run's output (its `config` supplies the plan flags); `rows_lost_per_attach` is the
floor-to-32 plus the 9-row prompt-end slack (41) for plowrt's boundary quantum, 0 for the raw ceiling.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_turns as ag  # noqa: E402


def oracle(args, quant):
    apps, plan = ag.build_plan(args)
    seen, tot, hit = set(), 0, 0
    for s in plan:
        for k, t in enumerate(s["turns"]):
            p = t["prompt_tokens"]
            tot += p
            if k == 0:
                if s["app"] in seen:
                    hit += max(0, apps[s["app"]] - quant)
                seen.add(s["app"])
            else:
                hit += max(0, p - t["user_tokens"] - quant)
    return dict(rate=args.rate, duration=args.duration, sessions=len(plan),
                turns=sum(len(s["turns"]) for s in plan), prompt_tokens=tot,
                oracle_hit=round(hit / tot, 4), rows_lost_per_attach=quant)


def main():
    cfg = json.load(open(sys.argv[1]))["config"]
    args = argparse.Namespace(**cfg)
    args.rate, args.duration = float(sys.argv[2]), float(sys.argv[3])
    quant = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(json.dumps(oracle(args, quant)))


if __name__ == "__main__":
    main()
