# Serving metrics

`GET /metrics` is always enabled on the serving listener. It returns Prometheus
text exposition 0.0.4 with `text/plain; version=0.0.4; charset=utf-8`. No GPU
synchronization or engine mutex is needed to scrape it.

## Ownership and cardinality

Each registered API model slug owns one `Arc<obs::Metrics>`. The dispatcher
captures that handle when it starts. Each admitted request owns a small request
observer until its slot is released. Backend implementations do not own the
serving counters: the common mux records the same events for CPU, AMD, NVIDIA,
and tensor-parallel engines.

All new serving series have `model_name` and `engine` labels. `model_name` is the
registered API slug, **not** the checkpoint's network name. Registry registration
rejects duplicate slugs; two aliases of the same checkpoint therefore remain
independent. `engine="0"` denotes the one logical dispatcher for that slug. TP
ranks are parts of that engine, not replicas to count as separate requests.
Future independent replicas need distinct engine indices before they can share
a model label.

Other labels are closed sets: completion `finished_reason` is `stop|length`;
tick `phase` is `prefill|decode|mixed`; histogram `le` uses fixed boundaries.
Request IDs, prompt text, paths, arbitrary error messages, context lengths, and
run IDs never become labels. Quotes, backslashes, and newlines in slugs are
escaped. Context and batch distributions belong in histograms, not dynamic
labels.

Registration makes a zero-valued metric set visible even before a dispatcher is
installed. `plowrt_model_ready` becomes one when a dispatcher is installed and
the residency state admits work. This is admission availability, not a GPU
health probe. Residency unload/drain/reload preserves the registered slug's
counters. Running observers release their gauge on success, error, cancellation,
preemption, and worker unwind. A queued request is counted before channel
submission and rolled back if the channel refuses it.

Unregistering a slug removes its metric set once its final dispatcher/request
handle is dropped. Scrapes and dispatcher registration prune these retired sets;
historical aliases do not accumulate indefinitely. Legacy process counters retain
retired counter totals. Re-registering after retirement starts new model series
at zero; Prometheus handles this as a counter reset. If the same slug is
re-registered before retirement, it continues the logical slug's counters.

## Compatibility surface

The naming baseline is the current vLLM V1 [production metric catalog](https://docs.vllm.ai/en/stable/usage/metrics/)
and [Prometheus logger source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py),
reviewed on 2026-09-11. This is a documented serving subset, not a claim that
every version-specific vLLM feature exists in Plow. Prometheus histogram bucket
boundaries need not match vLLM's to use the same dashboard queries.

| Exported vLLM family | Type | Plow event / meaning |
|---|---|---|
| `num_requests_running`, `num_requests_waiting` | Gauge | Occupied request slots; pending ingress jobs |
| `prompt_tokens_total` | Counter | Computed prompt rows, excluding reused prefix tokens |
| `prompt_tokens_cached_total` | Counter | Prompt tokens attached from a prefix |
| `generation_tokens_total` | Counter | Generated tokens, including EOS/stop tokens |
| `request_success_total` | Counter | Normal completion, labeled by `finished_reason` |
| `time_to_first_token_seconds` | Histogram | Handler arrival to first generated token |
| `inter_token_latency_seconds` | Histogram | Host observation interval between generated tokens |
| `request_time_per_output_token_seconds` | Histogram | Per completed request `(last-first)/(outputs-1)`; no sample for one output |
| `e2e_request_latency_seconds` | Histogram | Arrival to normal completion |
| `request_queue_time_seconds` | Histogram | Mux submission to slot admission |
| `request_inference_time_seconds` | Histogram | Slot admission to normal completion |
| `request_prefill_time_seconds` | Histogram | Slot admission to first generated token |
| `request_decode_time_seconds` | Histogram | First to last generated token of a completed request |
| `request_prompt_tokens`, `request_prefill_kv_computed_tokens` | Histogram | Completed request's total and uncached prompt length |
| `request_generation_tokens` | Histogram | Completed request's generated length |
| `request_params_max_tokens` | Histogram | Requested output limit at mux submission, including refused submissions |

All names in the table carry the `vllm:` prefix. Token counters record prompt
accounting at the first output because that is when all prefix attachment and
prefill work is known. A request that fails before its first output can have
performed uncounted prefill work. Generated tokens remain counted if the client
disconnects later. Normal completion records success when the model finishes;
it does not certify that the client received the final HTTP/SSE bytes.

The HTTP arrival timestamp starts inside the handler after request validation
and model residency checks, before tokenization. It excludes network transit,
JSON extraction, and model loading. Direct mux users provide `Job::arrived`.
Inference/prefill/decode intervals are wall time, including scheduling waits;
they are not isolated GPU compute time. Multi-step decode returns multiple
tokens together: ITL observes their host emission spacing, while request TPOT
captures the full first-to-last interval. Server histograms and client benchmark
latencies consequently have different boundaries.

### Deliberately absent families

* `kv_cache_usage_perc`: no uniform physical used/capacity snapshot across flat
  MLA allocations, shared prefix pools, CPU, CUDA VMM, and TP. Slot occupancy is
  not a valid substitute for memory occupancy.
* `prefix_cache_queries_total`, `prefix_cache_hits_total`: completed request
  usage is not the number of tokens queried by the cache lookup algorithm.
  Cached prompt accounting is exported separately; CUDA's existing VMM attach
  counters also remain available.
* `iteration_tokens_total`: Plow's mux tick can contain several engine steps
  and prefill chunks. A generated-output-only count would misrepresent vLLM's
  engine-step input-plus-output measure.
* Scheduler KV preemption, block lifetime sampling, speculation, disaggregated
  KV connectors, multimodal caches, LoRA, NaN detection, CUDA-graph statistics,
  and MFU/per-GPU counters: no equivalent common serving event is implemented.
  Unsupported features are omitted rather than exported as reassuring zeros.

## Plow-specific surface

| Family | Type | Purpose |
|---|---|---|
| `plowrt_model_ready` | Gauge | Installed dispatcher with admission enabled |
| `plowrt_model_requests_total`, `plowrt_model_rejected_total` | Counter | Submitted jobs and ingress/admission refusals |
| `plowrt_requests_aborted_total` | Counter | Admitted work dropped without normal completion or residency preemption |
| `plowrt_requests_preempted_total` | Counter | Requests terminated by residency reclamation |
| `plowrt_tick_errors_total` | Counter | Device-fault or worker-failure ticks |
| `plowrt_tick_duration_seconds{phase=...}` | Histogram | Returned tick wall time, including worker submission |
| `plowrt_tick_batch_size` | Histogram | Occupied slots at tick submission |
| `plowrt_tick_generation_tokens` | Histogram | Generated outputs per returned tick |
| `plowrt_run_packets` | Histogram | Existing executor-reported packet accounting at normal request completion |

Existing scheduler metrics also have `plowrt_model_` counterparts: arrival rate,
queueing utilization, batch and hold sums/counts, admission shedding, selected
decode rung, admission rung, occupied slot extent, and rung switches. These
gauges no longer overwrite another model's values. Rungs describe the last
selection and are cleared when the dispatcher becomes idle. They do not claim
to identify the actual GPU kernel or its wave geometry.

Legacy unlabeled `plowrt_*` serving metrics remain for existing dashboards.
Counters aggregate model totals; queue and rate gauges sum; decode rung/extent
gauges take the maximum. The latter is only a compatibility display, not a
meaningful combined model shape. Use `plowrt_model_*` for scheduling analysis.
The distinct names avoid counting a labeled and an unlabeled copy in one
Prometheus `sum()`.

CUDA VMM `plowrt_prefix_*{model=...}` series retain their existing label spelling
and gain HELP/TYPE metadata and proper escaping. These belong to the backend
pool handle and may reset when that handle is replaced. They are not universal
AMD/CPU physical-cache metrics. Scraping them takes only the existing short
pool-statistics lock, never an execution lock.

## Hot-path contract and detailed diagnostics

Request admission clones one Arc. A token observation reads `Instant` once and
updates fixed atomic bins/counters; there is no allocation, label formatting,
registry lookup, mutex, syscall introduced explicitly, or GPU event. Histograms
update one noncumulative bin and a sum; only scrapes form cumulative buckets.
`_count` is derived from the same loaded bins as `+Inf`, so those two match even
under concurrent updates. Other relaxed values are approximate snapshots, not
an atomic transaction. Durations use integer microseconds internally and seconds
on the wire.

Prometheus is for bounded aggregates over every request/tick. It does not retain
an unbounded record of individual runs. Existing `/trace` and opt-in diagnostic
logging remain the detailed path; process-global TTFT/pack-log diagnostics are
not a substitute for these model-scoped series. Actual per-segment launch,
wave, tiling, context-rung, GPU-duration, and TP-collective observations need
explicit engine events before they can be added as metrics. Do not infer them
from packet declarations or enable device synchronization merely to scrape.

Example queries:

```promql
sum by (model_name) (rate(vllm:generation_tokens_total[5m]))
histogram_quantile(0.99, sum by (model_name, le) (
  rate(vllm:time_to_first_token_seconds_bucket[5m])))
sum by (model_name, phase) (rate(plowrt_tick_duration_seconds_sum[5m]))
  / sum by (model_name, phase) (rate(plowrt_tick_duration_seconds_count[5m]))
```

## Verification

Unit tests cover cumulative buckets, label escaping, completion idempotence,
abort/preemption distinction, and gauge release. A CPU reference endpoint test
runs distinct aliases concurrently through streaming and buffered HTTP paths,
checks exact token and histogram counts, rejects duplicate exposition series,
then drains/reloads/unregisters a model while checking counter isolation and
series retirement. The existing mux/co-serving/preemption suites cover the
request lifecycle around the instrumentation. GPU-enabled builds are checked
separately; this change does not establish measured GPU serving overhead.
