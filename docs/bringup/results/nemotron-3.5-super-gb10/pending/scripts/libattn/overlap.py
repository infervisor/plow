"""overlap.py <result.json> <compilers.log>: compilers seen inside timed windows (quiet.sh watch lines)."""
import json, sys
res = json.load(open(sys.argv[1]))
wins = [w for r in res.values() for w in r.get("timed_windows", [])]
seen = [l.split() for l in open(sys.argv[2]) if l.strip()]
bad = {(p, c) for t, p, c in seen if any(a - 2 <= int(t) <= b + 2 for a, b in wins)}
print(f"compilers_during_run={len(bad)} (windows {len(wins)}, compiler samples {len(seen)})")
