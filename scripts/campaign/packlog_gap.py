"""Split a PLOW_PF_PACKLOG=1 server log into prefill / decode / residual, per tick.

Usage: packlog_gap.py <server.log>

`t_ms` is elapsed-since-start (a TIMESTAMP, not a duration -- never sum it). Consecutive
differences give each tick's true wall duration; `prefill_ms + decode_ms` is what the tick
accounts for, and the rest is host time.

Measured on the Gemma-4-12B 8192/C32 cell (2026-09-23): the residual is NOT diffuse overhead.
Prefill-tick residual was median 0.06 ms / p90 3.36 ms with a single 15.7 s outlier (the
wave boundary, 64 requests through 32 slots); decode-only ticks totalled 0.17 s over 327
ticks. Report the median and the max, never just the mean -- the mean alone reads as 101 ms
of per-tick overhead that does not exist.

Caveat: a unified tick folds the riding decode rows into `prefill_ms`, so `prefill` here is
prefill + fused decode, not prefill alone.
"""
import re
import sys

TICK = re.compile(
    r"PACKLOG TICK t_ms=([\d.]+) prefill_ms=([\d.]+) decode_ms=([\d.]+) "
    r"did_prefill=(\w+) decode_rows=(\d+)")

ticks = []
with open(sys.argv[1], errors="replace") as fh:
    for line in fh:
        m = TICK.search(line)
        if m:
            ticks.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                          m.group(4) in ("true", "1")))

if len(ticks) < 2:
    raise SystemExit("  need at least two ticks")

ticks.sort(key=lambda t: t[0])
span = ticks[-1][0] - ticks[0][0]
acc_p = sum(t[1] for t in ticks)
acc_d = sum(t[2] for t in ticks)

print(f"  ticks                      {len(ticks)}")
print(f"  tick-loop wall span        {span / 1000:8.2f} s   (last t_ms - first t_ms)")
print(f"  accounted prefill          {acc_p / 1000:8.2f} s   {100 * acc_p / span:5.1f}%")
print(f"  accounted decode           {acc_d / 1000:8.2f} s   {100 * acc_d / span:5.1f}%")
resid = span - acc_p - acc_d
print(f"  UNACCOUNTED (host)         {resid / 1000:8.2f} s   {100 * resid / span:5.1f}%")

# per-tick residual, split by whether the tick carried prefill
pf, dec = [], []
for i in range(1, len(ticks)):
    dt = ticks[i][0] - ticks[i - 1][0]
    r = dt - ticks[i][1] - ticks[i][2]
    (pf if ticks[i][3] else dec).append(r)

for name, xs in (("prefill ticks", pf), ("decode-only ticks", dec)):
    if not xs:
        continue
    xs_sorted = sorted(xs)
    med = xs_sorted[len(xs) // 2]
    print(f"\n  {name}: n={len(xs)}  total residual {sum(xs) / 1000:.2f} s")
    print(f"    mean {sum(xs) / len(xs):7.2f} ms   median {med:7.2f} ms   "
          f"p90 {xs_sorted[int(0.9 * len(xs))]:7.2f} ms   max {xs_sorted[-1]:8.2f} ms")

print("\n  A large residual on DECODE-ONLY ticks is pure host overhead per emitted token batch")
print("  (sampling readback + gpu_finish_and_emit_token per slot). On prefill ticks it also")
print("  includes pack assembly. Both are addressable; neither is kernel time.")
