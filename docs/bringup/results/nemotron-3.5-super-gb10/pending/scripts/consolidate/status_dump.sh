#!/usr/bin/env bash
# status_dump.sh: queue states, key result tails, and the tcinv dev-tree diff (not a git tree) vs HEAD export.
P=/home/ifv/plow-scratch/pending
S=/home/ifv/plow-scratch/consolidate/stage
date +%T
python3 -I -c "
import json,glob,os,time
for p in sorted(glob.glob('/tmp/plow-gpuq/*.json'),key=os.path.getmtime):
  d=json.load(open(p)); s=d.get('started'); print(os.path.basename(p)[:28],d['label'],d['state'],d.get('rc'),int((time.time()-s)/60) if s and d['state']=='running' else '')"
cd /home/ifv/plow-tcinv || exit 1
: > "$P/qwen-fp8tc-row-invariant.patch"
for f in $(cd /home/ifv/plow-tcinv && find runtime crates scripts docs recipes -type f \( -name '*.cuh' -o -name '*.cu' -o -name '*.rs' -o -name '*.py' -o -name '*.md' -o -name '*.toml' -o -name '*.h' \) -newer "$S/Cargo.toml" 2>/dev/null); do
  diff -u --label "a/$f" --label "b/$f" "$S/$f" "/home/ifv/plow-tcinv/$f" >> "$P/qwen-fp8tc-row-invariant.patch" 2>/dev/null
done
grep '^+++ ' "$P/qwen-fp8tc-row-invariant.patch"
for f in /home/ifv/plow-scratch/libattn/summary* /home/ifv/plow-scratch/libattn/*.md /home/ifv/plow-scratch/mamba/g1/summary* /home/ifv/plow-scratch/tcinv/j2/summary*; do [ -f "$f" ] && { echo "== $f"; tail -40 "$f"; }; done
ls /home/ifv/plow-scratch/libattn /home/ifv/plow-scratch/mamba/g1 /home/ifv/plow-scratch/tcinv/j2 2>&1 | head -40
