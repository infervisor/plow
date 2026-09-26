#!/usr/bin/env bash
# pf_ab.sh <asset...> — prefill TTFT p50 per prompt length, one load per asset, ABA order.
for a in "$@"; do
  /root/tts-work/plowrt-tts6 bench --assets /root/tts-work/assets/$a --prefill-sweep --prefill-lengths 60,128,256,512 \
    --prefill-reps 5 --prefill-warmups 2 2>/dev/null | /root/tts-work/venv-ref/bin/python -c "
import json,sys
d=json.load(sys.stdin)
print('$a', ' '.join(f\"{r['prompt_tokens']}:{r['ttft_ms']['p50']:.2f}ms\" for r in d['rows']))"
done
