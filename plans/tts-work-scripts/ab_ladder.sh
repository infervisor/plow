#!/usr/bin/env bash
# A/B in ONE lease: widest-only (probe2) vs narrow ladder (probe3), conc 1 and 8, greedy.
WT=/root/plow/.claude/worktrees/tts-veena-chatterbox
R=/root/tts-work/results/ladder-ab
mkdir -p $R
for c in 1 8; do
  for arm in probe2 probe3 probe2b; do
    bin=/root/tts-work/plowrt-${arm%b}
    PLOWRT_BIN=$bin $WT/scripts/tts/veena_serve_probe.sh /root/tts-work/assets/veena $R/$arm-c$c \
      --conc $c --n 16 2>&1 | grep -E '"engine"|throughput|med_' | tr -d '\n'; echo "  <- $arm c$c"
  done
done
/root/tts-work/venv-ref/bin/python - <<'EOF'
import json
R="/root/tts-work/results/ladder-ab"
for c in (1, 8):
    t = {a: json.load(open(f"{R}/{a}-c{c}/plow_c{c}.json"))["tokens"] for a in ("probe2", "probe3", "probe2b")}
    for x, y in (("probe2", "probe2b"), ("probe2", "probe3")):
        same = sum(t[x][k] == t[y][k] for k in t[x])
        print(f"c{c} {x} vs {y}: identical token streams {same}/{len(t[x])}")
EOF
