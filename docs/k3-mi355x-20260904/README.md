# Kimi-K3 on 8x MI355X: 2026-09-04 planning snapshot

Planning documents and the gate/bundle scripts used for the 2026-09-04 campaign,
preserved here because `plans/` is gitignored and the working machine was retired.

- `k3-beat-vllm-0.28-v3.md`: the plan, with the live execution log (§7.0) and the
  TTFT lever research summary.
- `decode-gap-plan-20260904.md`: decode attribution vs vLLM 0.28 and ranked levers.
- `scaling-audit-20260904.md`: which promoted changes generalize beyond the C1
  8192→1024 cell (long context, batching, throughput) and the gates required.
- `seq-parallel-seams-feasibility-20260904.md`: the sequence-parallel seams lever
  (emit prototype on branch `codex/seq-parallel-seams`, runtime arms pending).
- `scripts/`: bundle build (`showdown_bundle.sh`), served showdown launcher
  (`run_showdown.sh`, needs `sg docker` when the login lacks the docker group),
  stack gates, the regstate exactness probe, and the C1 publication generator.
  Paths under `/tmp` and the worktree path are machine-specific.

Served state at the snapshot (`perf-data/kimi-k3-plowrt-mi355x-baseline.md`):
Plow 1113 ms TTFT / 25.25 ms TPOT / 38.0 tok/s vs vLLM 566 / 20.88 / 46.7, with a
further −22.9 ms TTFT engine-gated after publication. Remaining gap and the order
of attack are in the three plan documents.
