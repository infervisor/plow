# Kimi-K3 MI325X decode GEMV workgroup campaign

Date: 2026-08-10. Hardware: 8 leased MI325X GPUs (gfx942, 304 CUs each).
Toolchain: flake-pinned ROCm 7.14.0. Every emit and device run used
`nix develop`; timings held one repository `gpulease`.

## Experiment

The control uses 304 CUs for ordinary K3 decode GEMV-family packets. The
candidate is re-emitted with `PLOW_GLM_GEMV_WG=152`; it reuses the exact same
gfx942 code objects. Despite the legacy knob name, K3 consumes the shared GEMV
workgroup cap. The sharded lm_head deliberately remains at 304 CUs.

Fewer workgroups give each live wave more output rows on the collapsed K3
shapes, increasing independent weight loads in flight. This is an emitter-only
ownership change: instruction count, opcodes, operands, M/N/K, and reduction
order are unchanged.

The structural disassembly diff found only these block-count changes:

| packets | op | control | candidate |
|---:|---|---:|---:|
| 462 | Gemv | 299 | 150 |
| 117 | Gemv | 256 | 140 |
| 92 | GemvGlu | 256 | 128 |
| 48 | Gemv | 256 | 128 |
| 24 | Gemv | 293 | 150 |
| 2 | Gemv | 302 | 151 |

## Interleaved result

Four order-reversed pairs, 64 real-weight decode steps per run, TP8, context 5,
and direct per-token TP counter auditing:

| arm | median ms/token | min | max | sd | n |
|---|---:|---:|---:|---:|---:|
| control | 73.210 | 73.070 | 73.309 | 0.098 | 4 |
| wg152 | 58.508 | 58.496 | 58.532 | 0.015 | 4 |

Candidate vs control median: **-20.08%**. The candidate's worst run was faster
than the control's best run. Starting junction temperatures were 53--60 C; the
candidate remained faster when it started hotter.

The candidate's 64-token sequence was byte-for-byte identical to the control,
including the initial continuation ` Paris. The capital of Germany is Berlin.`.
Every step was token-identical across all eight ranks.

## Raw-trace attribution

The corrected last-program trace contains the same 2,459 packets in both arms.

| metric | control | wg152 |
|---|---:|---:|
| device trace span | 67.576 ms | 52.722 ms |
| aggregate body envelopes | 67.045 ms | 61.688 ms |
| Gemv body | 28.506 ms | 24.583 ms |
| Gemv weight rate | 492 GB/s | 570 GB/s |
| GemvGlu body | 3.316 ms | 2.488 ms |
| GemvGlu weight rate | 305 GB/s | 407 GB/s |

Body envelopes overlap, so their sum is not the token latency. The larger span
reduction shows that the workgroup change also improves inter-packet overlap;
the end-to-end delta must not be attributed only to ordinary Gemv.

## Reproduction

Emit both packets with the canonical K3 TP8 command, adding this environment
setting only to the candidate:

```bash
nix develop --command env PLOW_GLM_GEMV_WG=152 K3_FULL=1 \
  PLOW_FP8_KV=1 PLOW_MXFP4=1 ./target/release/plowc \
  --hf-dir /home/lava/models/k3_farm --emit devblob \
  --arch gfx942 --gpu MI325X --num-gpus 8 --parallel tp \
  --max-ctx 32768 --n-cu 304 --out /tmp/k3-wg152
```

Link both arms to the same `hsaco/`, then run:

```bash
nix develop --command env PLOW_TP_AUDIT_DIRECT=1 BATCHED=0 STEPS=64 \
  perf-data/harness/gpulease -n 8 k3-gemv-wg152-ab \
  scripts/k3_ab_bench.sh ctl /tmp/k3-wgctl wg152 /tmp/k3-wg152 4 0
```

This is a Plow A/B, not a same-session vLLM comparison.

## Composed safe-audit result

The adopted candidate was rebuilt with the opt-in compact TP audit. One small
device kernel checks all 464 logical counter lines after the interpreter drains;
the host then reads one status word per rank. The existing copy and direct audit
paths remain available.

A separate 64-token soak produced the same cross-rank and cross-arm token stream
at 55.639 ms/token (18.0 tok/s). Its step breakdown reports 52.679 ms in the
device program and 1.276 ms in the complete post-device audit, down from 4.147
ms for direct host reads. The combined result is not folded into the wg152 A/B
table because it changes a second axis.

## Follow-up bracket

The compact-audit packet was then bracketed at workgroup caps 200 and 128 in one
lease. The screen measured wg200 at 59.843 ms/token, wg152 at 55.686/55.661,
and wg128 at 54.486. wg200 was rejected.

Three order-reversed wg152/wg128 pairs confirmed a disjoint result:

| arm | median ms/token | min | max | sd | n |
|---|---:|---:|---:|---:|---:|
| wg152 | 55.617 | 55.605 | 55.625 | 0.010 | 3 |
| wg128 | 54.463 | 54.409 | 54.473 | 0.034 | 3 |

wg128 improves the median by 2.07%. Its corrected trace span is 51.468 ms vs
52.722 ms for wg152. The dominant `N=3584,K=7168` body improves 4.238→3.736
ms and `N=896,K=7168` improves 3.498→3.061 ms. The hot
`N=7168,K=768` row regresses 2.533→3.417 ms, so the predefined stop rule was
met: wg96 was not run. wg128 is the adopted packet.

The full 64-token stream remained byte-identical across arms and ranks. The
OpenAI four-prompt serve gate passed after installing wg128.
