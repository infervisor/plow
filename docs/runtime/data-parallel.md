# Data parallel serving (`--dp`)

One `plowrt serve` can hold N full copies (DP ranks) of a TP1 model, one per device group, behind
one model name. Each rank has its own engine, dispatcher, KV/prefix cache and speech sidecars
(ASR encoder, TTS codec/vocoder) on its own GPU.

```sh
plowrt serve --assets models/gemma-4-e4b --dp 8          # 8 ranks on 8 GPUs
plowrt serve --assets a --assets b --dp all,b=2          # a on every group, b on 2
plowrt serve --assets a --dp 2 --pin a@0,a@5             # rank 0 on GPU 0, rank 1 on GPU 5
```

- `--dp N|all|slug=N` (`PLOW_DP`). `all` = one rank per device group. Refused for TP>1 bundles,
  without the CUDA manager, or with more ranks than groups. Ranks never share a group (placement
  anti-affinity); other models still take the emptiest group. Unset (or 1) is the single-copy
  serve, unchanged: no router runs.
- Rank `r` of model `m` is the instance `m#r`: it appears as `engine="r"` on every per-model
  metric (vLLM's label), in logs, and in `GET /v1/models/status` (`dp: [...]`). `/v1/models`
  lists the model once.

## Routing

Per request, after tokenizing (lock-free, relaxed atomics only):

1. **Session**: a request with `X-Session-Id` goes to the session's rank (its retained rows live
   there) unless that rank is down, or has more than `PLOW_ROUTE_SPILL` queued requests (default
   max(4, slots/4)) *and* carries 0.5 more load than the least-loaded rank. A new session goes to
   the rank with the least load + pinned sessions / slots. Sessions live in 64 shards and expire
   after 15 min idle.
2. **Prefix**: the router keeps the last 128-token block hash of each prompt it routed (a sampled
   chain, per model); a prompt extending one goes to that prompt's rank, before the rank has
   published the rows. A tail routed to two ranks (a shared system prompt) points nowhere. A
   prompt that follows nothing is checked against each rank's VMM prefix cache with `try_lock`
   (a busy cache is skipped, `plowrt_route_probe_contended_total`); those block hashes ride the job
   to the rank's admission and attach, so the cache's hash runs once. The longest match wins while
   its load is within 0.5 + `PLOW_ROUTE_PREFIX_SLACK` x (matched / prompt) of the least-loaded
   rank. `PLOW_ROUTE_PREFIX=0` disables.
3. **Least loaded**: load = (queued + routed-not-yet-queued + active) / slots + 4 x max(0, KV
   reserved - 0.8); ties rotate per thread.

A rank that closed or filled between the pick and the submit hands the same job (tokens, prefix
key) to another rank (`plowrt_dp_route_retries_total`). ASR (HTTP and realtime) and TTS route by
session and load; a realtime session re-routes its next turn when its rank goes away.

Metrics: `plowrt_dp_rank_{info,up,load}`, `plowrt_dp_route_decisions_total{reason}`,
`plowrt_dp_route_seconds{quantile}`, `plowrt_dp_sessions`. `/health` answers 200 `degraded: ...`
while some rank of each model is alive and 503 only when every rank of a model is dead.

## Control plane

`POST /v1/models/unload {"model": "gemma", "rank": 3}` (or `"device": 3`, or `"model":
"gemma#3"`) drains that rank for up to `PLOW_DRAIN_TIMEOUT_MS` (30 s unset) while new work
routes to the others, then frees its memory; `load` with the same selector brings it back. Without
a selector both act on every rank.

Each rank's dispatcher thread is pinned to its GPU's CPU socket (`PLOW_DP_NUMA_PIN=0` disables).
Startup loads the first group alone, then the rest four at a time; later engines of the same GPU
model reuse the first one's timed cuBLASLt picks, so ranks run identical kernels.

## Measured (8x L40S, gemma-4-e4b and qwen3-asr deploy bundles, 2026-10-08)

| test | result |
|---|---|
| gemma closed-loop agentic, 64N sessions x 6 turns (`X-Session-Id`) | out tok/s 1096 / 2215 / 4401 / 7825 at N=1/2/4/8 (2.02x, 4.02x, 7.14x); per-turn TTFT p50 within 3% of N=1; 0 errors |
| qwen3-asr, N open-loop clients at 60 req/s, 73-clip manifest | 397 / 783 / 1571 / 3147 audio-s/s (7.9x at N=8); WER 3.74% -> 3.82%; each GPU 38.6 GiB (encoders on their own GPU) |
| temperature 0, 16 prompts x 16 repeats over 8 ranks | identical to N=1 output (3 runs) |
| no `X-Session-Id`, VMM prefix cache on, N=8 | prefix routing: cache hit 75-77% (N=1 77.3%) vs 20.2% without; 1.3-1.9x tok/s; rank imbalance up to 1.5x (open risk) |
| unload rank 3 under load (9216 agentic requests) | 0 errors; 37.5 GiB freed; reload 6 s, serving again 31 ms after |
| realtime ASR, 64 sessions, rank unloaded | sessions re-routed (drain 59 ms), 0 DP-related failures |
| router cost (release microbench, dp=8) | route without probe 257 ns p50 / 265 ns p99; served route time p50 4-8 µs, p99 8-16 µs incl. hashing |
| dp=1 vs pre-DP binary, c1 TTFT | 109.4 vs 108.4 ms (run-to-run spread 1.4 ms); no router code runs at dp=1 |
