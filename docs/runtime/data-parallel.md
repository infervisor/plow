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
Startup loads ranks four at a time.
