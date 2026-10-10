# Serving host path

This page covers the CPU budget of a served request outside the GPU work. It traces the HTTP
request through tokenization, the mux, the per-token emit, and the SSE write, and covers the ASR
audio frontend. Measured on the H100 box (AMD EPYC 7R13, 16 vCPU visible, kernel 7.0), E4B kit3, October 2026.

## Path

1. hyper/axum (`hyper_util` auto builder, `TCP_NODELAY`).
2. The handler parses the JSON, renders the template, and encodes. Encode is split across the
   `plow-encode` rayon pool for long prompts.
3. Mux ingress mpsc.
4. Inline dispatcher on the mux thread.
5. Engine tick.
6. `handle_produced_token`: `incremental_delta` detok, stop strings, `try_send`.
7. Per-request mpsc.
8. SSE unfold stream.
9. hyper `writev`.

On CUDA the decode is pipelined (`pipe_step`): the next step is enqueued before the host waits.
Steps 4–9 therefore overlap the GPU step, and host cuts do not move TPOT unless the host share
reaches the step time.

## Tools

| Tool | What it measures |
|---|---|
| `cargo test --release -p plowrt --test host_serve_bench -- --ignored --nocapture` | CPU-only load harness: production router, hyper and mux over a reference bundle, raw HTTP/1.1 client. Reports server CPU µs and allocations per request and per token, TTFT/ITL/E2E. `HOSTBENCH_TOKENIZER`, `HOSTBENCH_SCALE`, `HOSTBENCH_CELLS`. |
| `cargo test --release -p plowrt --features hf-tokenizer --lib host_path_microbench -- --ignored --nocapture` | Per-stage microbench on a real tokenizer: encode, detok, SSE frame, `try_send`, handler wake, engine handoff. |
| `PLOW_HOST_TIMING=1` | `HOSTT` lines: decode period, engine call, emit, tick_other, dispatcher, host_share; per request pre_submit, submit_to_emit, ttft. |
| `PLOW_TTFT_LOG=1` | TTFT phases. |
| `frontend_microbench` (asr, ignored) | Resampler and mel cost per audio second. |

## What changed

| Change | Effect |
|---|---|
| `text/detok.rs`: table detokenizer for the two decoder chains served models use (ByteLevel; Replace ▁ + ByteFallback + Fuse). Built once per tokenizer and self-checked against `Tokenizer::decode` on 1500 random windows; any mismatch falls back to the library. Other chains always use the library. | Detok 0.85 → 0.13 µs/token, output byte-identical. |
| `Tokenize::decode_append` + thread-local buffers in `incremental_delta` | Per-token delta without allocating |
| `serve/stream.rs` `FrameHead`: the constant `data: {"id",…,"choices":[` prefix is serialized once per stream; each token frame is the choice plus `]}\n\n` in one `Bytes`. Body is `Body::from_stream` with the same headers. A test compares the bytes against axum `Sse`. | SSE frame 0.38 → 0.12–0.16 µs, 1 allocation/frame |
| `SseState` owns id/model/logprob format | No per-poll `String` clones |
| ASR `Resampler::drain`: eight outputs' tap sums side by side, each in the original tap order | 2.0–2.6× faster, bit-identical |

No API, knob default, or output change. E4B token digests are identical before and after (`3d34c9473101de20`).

## Budget, CPU harness (byte tokenizer, reference engine)

`host_serve_bench`, base `fcb2250c` vs `afea708c`. Two runs each; numbers agree within 3%.

| Cell | srv CPU µs/tok base → new | allocs/tok base → new | tok/s base → new |
|---|---|---|---|
| stream c1, 128 tok | 16.3 → 16.0 | 18.5 → 5.4 | 99k → 100k |
| stream c64, 128 tok | 17.0 → 13.8 | 18.0 → 4.9 | 294k → 383k |

| Cell | srv CPU µs/req | allocs/req |
|---|---|---|
| non-stream c16, 1 tok | 55 → 55 | 94 → 92 |

Of the 4.9 allocations per token, 2 belong to the reference engine. The serving path has:

- 1 for the frame `Bytes`;
- about 2 for hyper/tokio chunk and queue bookkeeping.

With the Gemma tokenizer and a 1k-token prompt, a request costs:

- about 4150 allocations (HF BPE encode; `tokenizers` crate internals);
- 3.6 ms of CPU at c1, spread over the encode pool;
- 0.45 ms of latency at c1.

## Budget, GPU (E4B, ISL 1000 / OSL 128, H100)

Profile arm (`PLOW_HOST_TIMING=1`, a malloc counter and perf), base vs new.

| Per decode step | c1 base | c1 new | c64 base | c64 new |
|---|---|---|---|---|
| period µs | 5707 | 5706 | ~14550 | ~14500 |
| emit µs (per token) | 6.4–9.6 | 4.7–6.5 | 137.7 (2.15) | 29.0 (0.45) |
| tick_other µs | 6–9 | 4.5–7.8 | 14.1 | 8.1 |
| dispatcher µs | 11–15 | 8.4–13 | 19.7 | 11.7 |
| host_share | 0.42–0.61% | 0.32–0.49% | 2.33% | 0.67% |

| Per token, c64 (24576 tok) | base | new |
|---|---|---|
| tokio worker CPU µs | 22.4 | 15.5 |
| encode pool CPU µs (per-request work) | 49.2 | 47.6 |
| mallocs (libc, whole process) | 66.3 | 33.3 |
| syscalls tokio / mux / encode / total | 1.78 / 0.19 / 5.45 / 7.46 | 1.48 / 0.13 / 5.68 / 7.32 |

At c1 the counts are:

- mallocs per request: 8257 → 4109. The remaining ~4100 come from encode.
- syscalls per token, base: tokio 4.15, mux 2.19, encode 7.6 (about 970 per request in rayon
  futex wakes).

The mux thread's own CPU is about 245 µs per token at c64. It breaks down as:

- 71% `libcuda` spin-wait in the step sync;
- 10% vdso clock;
- 0.9% plowrt code.

That spin is the CUDA driver's sync, not host-path work.

End to end, two rounds per arm with the order alternated:

| | base | new |
|---|---|---|
| c1 TTFT p50 ms | 21.24 / 21.83 | 21.11 / 21.69 |
| c1 TPOT ms | 5.706 | 5.706 |
| c64 out tok/s | 4090 / 3971 | 4110 / 3989 |
| c64 TTFT p50 ms | 106.6 / 110.9 | 105.2 / 110.6 |
| digest | 3d34c9473101de20 | 3d34c9473101de20 |

The E2E change is within noise. Decode is GPU-bound, and the emit path was already hidden behind the
pipelined step. The cut is headroom: the CPU freed at c64 is about 1.3 ms per second of decode on the
mux thread, plus 7 µs per token on tokio. That matters for multi-model voice boxes, where several
mux threads and the speech frontends share cores.

Voice core (kit3: qwen3-asr, chatterbox-mtl and E4B in one process), base vs new, two rounds each. Digests are identical: asr `2f3e06bebe4d0c64`, tts `00624137f60ddfc8`.

| | base r1 / r2 | new r1 / r2 |
|---|---|---|
| ASR c1 latency p50 ms | 67 / 74 | 71 / 67 |
| ASR c16 RTFx | 506 / 474 | 501 / 496 |
| TTS c1 TTFA p50 ms | 135 / 135 | 135 / 134 |
| 32 calls, E2E first audio p50 / p95 ms | 1115 / 2324, 1191 / 2034 | 982 / 2255, 1164 / 2223 |

All differences are within run-to-run noise. Voice-agent runs vary by about 10% between rounds.
The resampler saving is about 0.6 ms of CPU per 1 s of 48 kHz audio. That is CPU headroom; it
does not change latency at this load.

## Against C-level references

| Stage | Rust now | C-level reference | Ratio | Note |
|---|---|---|---|---|
| SSE frame write (per token, server CPU) | ~13 µs (harness, c64) | 5.3 µs: `writev` of the same chunked frame on loopback from one C thread (`sse_floor.c`), 6.4 µs at c1 | ~40% of C speed | About 50% of the Rust cost is the same kernel `writev`; the rest is hyper/axum 12.6%, tokio futex/scheduler 13.6%, SSE 4.6%, alloc 2.2%. `TOKIO_WORKER_THREADS=4` gives 10.6 µs at equal throughput. |
| Detok | 0.13 µs/token | table lookup + memcpy | ~C | Was 0.85 µs (HF `decode` allocates per call) |
| SSE frame build | 0.12–0.16 µs | `snprintf` into a buffer | ~C | One `Bytes` |
| Emit per token (GPU c64) | 0.45 µs | — | — | detok + stop scan + frame + `try_send` |
| Request JSON parse | 0.02 ms | simdjson about 3× faster | — | Under 0.1% of TTFT |
| Template render | 0.015 ms | — | — | |
| Encode, 1k prompt | 0.39 ms (16-thread split), 1.39 ms single | HF `tokenizers` *is* the reference (Rust); a C++ SentencePiece BPE is about 1–1.5× | ~C | ~4100 allocations per request remain inside the crate |
| `try_send` + wake | 0.28 µs (16 streams), 2.8 µs (one stream, cross-thread wake) | eventfd/futex wake about 2–3 µs | ~C | |
| Engine-thread handoff | inline 0.06 µs (default) vs 25 µs channel | — | — | Inline dispatch is the default on GPU |
| Resampler (5 s audio, 48 kHz) | 3.3 ms | An auto-vectorized polyphase loop with reassociated sums is about 2× faster | ~50% | Bit-identity forbids reassociating each output's tap sum. |
| Silero VAD | 72 µs per 32 ms frame | — | ~20% of single-core FMA peak | Dot-product rewrites were bit-identical but slower (register spills); unchanged |
| Mel, 5 s | 1.5 ms | — | — | |

`LD_PRELOAD` jemalloc gave no measurable gain on either the harness or the GPU arm, so no allocator
crate was added.

## Remaining gaps

1. **SSE write path at ~40% of C.** Half of it is the kernel `writev`, which C pays too. The other
   half is hyper/axum framing and the tokio cross-thread wake per frame. Closing it means coalescing
   frames per connection per tick (one `writev` for several tokens). That changes the timing of
   delivery to clients, so it was not done here. Fewer tokio workers (4) cut CPU by 23% at the same
   throughput on the harness. On GPU E4B it was neutral: c64 4116 / 3969 tok/s and the same digest. The default is unchanged.
2. **Encode allocations** (about 4100 per 1k-token request) and the rayon wakes (about 700–970
   syscalls per request). These live inside the `tokenizers` crate. Splitting words before BPE and
   adapting the split were tried and reverted: cold-cache encode was equal or slower, and c16
   latency got worse.
3. **libcuda spin on the mux thread.** This is the step sync, not host work. A blocking-sync
   context would free the core but adds wake latency to every step.
4. **Resampler and VAD** are bound by the bit-identity rule. A reassociated version would roughly
   double the resampler speed but changes ASR features in the last bits.
