# Kimi-K3 plowrt TP8 bench↔HTTP parity smoke (MI355X)

C1 production-checkpoint parity passed on 2026-09-03: gfx950, 8×MI355X, ROCm
7.14.0, BF16 KV, MXFP4 experts. It used the current-source throughput-control
asset documented in [the throughput-gates record](kimi-k3-plowrt-mi355x-throughput-gates.md),
including that record's packet, build manifest, and HSACO SHA-256 identities.

Corrected reproduction command (the redirect is inside the Nix environment, so
the Nix shell banner cannot contaminate the JSON file):

```bash
perf-data/tools/gpulease -n 8 k3-parity kp render -c \
  'nix develop /home/lava/plow --command bash -c '\''exec /home/lava/plow/target/release/plowrt --rt-checkpoint /home/lava/plow/build-amd/k3-gfx950-farm bench --assets /tmp/plow-k3-throughput-control-gq.37c4Il/assets --prompt-ids 19180,658,459,1332,85384 --concurrency 1 --requests 1 --warmup-requests 0 --output-len 4 --parity-report --engine-diagnostics >/tmp/kimi-k3-plowrt-tp8-parity-smoke.json'\'''
```

Matched bench and non-stream `/v1/completions` result:

- completed 1, failed 0
- HTTP prompt text: `Hello from Plow parity`
- prompt IDs: `[19180, 658, 459, 1332, 85384]`
- bench and HTTP output IDs: `[2598, 198, 5054, 220]`
- prefill: bucket 128, rows 5
- decode: B1, 3 steps (the first output token is produced by prefill)
- TP: 8 ranks agreed; token-audit cadence 1; every-dispatch counter audit and
  all-rank prefill completion evidence enabled
- TTFT 204.548364 ms; mean TPOT 76.790765 ms; output throughput 9.19618 token/s
- `check_bench_serve_parity.py` passed

This completes the exact C1 production-checkpoint bench↔HTTP parity gate. It
does not establish batched or ragged per-slot parity.

An earlier bench-only smoke on the same artifact produced prompt IDs
`[1008, 10484, 318, 15383, 387]` and output IDs
`[17374, 13, 646, 10484]`; the matched gate above supersedes it.
