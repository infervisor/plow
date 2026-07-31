#!/usr/bin/env bash
# sc1_coverage.sh — PROVE that every global store in a PLOW_GATE_SC1 object carries a scope.
#
# PLOW_GATE_SC1 drops the release `buffer_wbl2` on the signal side. That is sound ONLY if every
# activation store is device/system-scoped, because one plain store leaves one dirty line in the
# producer's XCD L2 and the race the knob exists to remove is back. Coverage was reasoned about
# three times and was wrong three times (see runtime/amd/amd_common.h). This counts it instead.
#
#   ./scripts/sc1_coverage.sh <object.elf>            # fail if any plain global_store survives
#   ./scripts/sc1_coverage.sh <object.elf> --report   # list plain stores by source line
#
# --report needs the object built with -gline-tables-only.
#
# EXCLUSION, and it is the only one. `interp.hip`'s trace record (`prog.trace[base + ix] = r`) is
# written under PLOW_TRACE, read by the HOST after the kernel retires, and crosses no counter
# gate. Nothing else is excluded; a new op that forgets a store helper fails this script.
set -euo pipefail

OBJ="${1:?usage: sc1_coverage.sh <object.elf> [--report]}"
MODE="${2:-check}"
ARCH="${PLOW_HIP_ARCH:-gfx950}"
ROCM="${ROCM_PATH:-/opt/rocm}"
OBJDUMP="$(command -v llvm-objdump || echo "$ROCM/lib/llvm/bin/llvm-objdump")"

[ -r "$OBJ" ] || { echo "sc1_coverage: cannot read $OBJ" >&2; exit 2; }

ASM="$(mktemp)"; trap 'rm -f "$ASM"' EXIT
"$OBJDUMP" -d --line-numbers --mcpu="$ARCH" "$OBJ" > "$ASM"

python3 - "$ASM" "$MODE" <<'PY'
import re, sys, collections

asm, mode = sys.argv[1], sys.argv[2]
# The trace record is host-read and crosses no gate; see the header note.
EXCLUDE = (("amd/interp.hip", "prog.trace"),)

cur = None
plain, covered = collections.Counter(), 0
saw_line_info = False
seen_any = 0          # every line mentioning a global store AT ALL, however spelled
for line in open(asm):
    s = line.strip()
    m = re.match(r'^; (/\S+):(\d+)', s)
    if m:
        cur = (m.group(1).split('/runtime/')[-1], int(m.group(2)))
        saw_line_info = True
        continue
    if 'global_store' in s:
        seen_any += 1
    m = re.search(r'\bglobal_store_[a-z0-9_]+(.*)$', s)
    if not m:
        continue
    if 'sc0' in m.group(1) or 'sc1' in m.group(1):
        covered += 1
    else:
        plain[cur] += 1

# SELF-CHECK. The classifier above is a regex, and a regex that fails to match an opcode
# spelling counts it as neither scoped nor plain — i.e. it reports a CLEAN object while the
# store sits there unscoped. That is not hypothetical: `\bglobal_store_[a-z0-9]+\b` silently
# dropped all 320 `global_store_short_d16_hi` in the flash object, because `\b` cannot match
# between `short` and `_d16`. Assert the partition is total instead of trusting the pattern.
if covered + sum(plain.values()) != seen_any:
    print(f"sc1 coverage: BROKEN CLASSIFIER — {seen_any} lines mention global_store but only "
          f"{covered + sum(plain.values())} were classified.\nThe opcode regex missed a spelling; "
          f"fix it before trusting any result from this script.")
    sys.exit(3)

# WITHOUT line info the trace-record exclusion cannot resolve, so every object reports a
# small non-zero residual and a REAL miss hides inside it. That ambiguity is how a gap
# survives a green run, so refuse rather than guess.
if plain and not saw_line_info:
    print(f"sc1 coverage: {covered} scoped, {sum(plain.values())} plain — but the object carries "
          f"NO LINE INFO,\nso the trace-record exclusion cannot resolve and a real miss would hide "
          f"in the residual.\nRebuild with -gline-tables-only and re-run.")
    sys.exit(2)

# Drop the one documented exclusion by source file, resolved against the recorded line.
def excluded(k):
    return k is not None and any(k[0].endswith(f) for f, _ in EXCLUDE) and k[0].endswith('interp.hip')

kept = {k: v for k, v in plain.items() if not excluded(k)}
skipped = sum(v for k, v in plain.items() if excluded(k))

total_plain = sum(kept.values())
print(f"sc1 coverage: {covered} scoped, {total_plain} plain"
      f"{f' ({skipped} excluded: trace record)' if skipped else ''}")

if mode == '--report' or total_plain:
    if kept:
        print("\nPLAIN global stores — each one reinstates the race:")
        for k, v in sorted(kept.items(), key=lambda x: -x[1]):
            where = f"{k[0]}:{k[1]}" if k else "<no line info; rebuild with -gline-tables-only>"
            print(f"  {v:>4}  {where}")

if total_plain:
    print(f"\nFAIL: {total_plain} unscoped global store(s). PLOW_GATE_SC1 is NOT sound here.")
    sys.exit(1)
print("OK: every global store carries sc0/sc1.")
PY
