# Kimi-K3 MI325X B1 KDA early-forget scheduling screen

Date: 2026-08-11. Result: rejected.

## Hypothesis

The current B1 trace shows `f_b` reaching `KdaConvStateStepG` after the fused QKVG projection in
all 69 KDA layers. Move the independent `f_a` and beta projections, plus the dependent `f_b`
projection, before QKVG so the global queue can overlap the two branches. This changes packet order
only; tensor handles, arithmetic, grids, dependencies, interpreter object, and prefill programs stay
unchanged.

This is not a single-block experiment. Its only mechanism is scheduling between four packets in the
global queue. A block or isolated-op harness cannot reproduce that queue interaction, so the screen
advanced directly from structural packet checks to served TP8.

## Structural gate

- Control packet SHA256: `66eb9409ea5f928bbd2a68359eb85a659e16b15ddc5e50e695d56fd53861d43c`.
  It exactly reproduces the installed B1/128K packet.
- Candidate packet SHA256: `45766de07f411473ab29960f081367ddff27475b721b0ba4aa67b7d6147b076c`.
- Both packets contain 2,274 decode instructions and 56,850 counters.
- The only decode disassembly change is `QKVG, f_a, beta, f_b` becoming
  `f_a, beta, f_b, QKVG` in each KDA layer.
- Prefill instruction order is unchanged.
- Both arms use the same GQ decode object, SHA256
  `1b73d5d8e434228738e988056f7f959eadc3ac7542df68c3c372fa3dea0e1d39`.

## Served result

Hardware and runtime: TP8 on 8x MI325X/gfx942, ROCm 7.14 Nix toolchain, native MXFP4 weights,
FP8 KV, V2 prefill, L2 placement, counter double buffering, device state clear, and compact exact TP
counter audit. Client: vLLM 0.27.0 `bench serve`, OpenAI chat, C1/N1, random input 32, output 512,
seed 0, temperature 0, ignore EOS, and one warmup.

| Arm | TPOT | ITL median | Output tok/s | GPU drain |
|---|---:|---:|---:|---:|
| control | 48.292 ms | 48.253 ms | 20.503 | about 45.4 ms |
| early forget | 48.012 ms | 47.976 ms | 20.620 | about 45.0 ms |

The candidate improves TPOT by 0.281 ms (0.58%). Both arms complete 1/1 requests, generate exactly
512 tokens, report no errors, and produce byte-identical generated text. The GPU lease is clean
after both runs.

Detailed JSON:

- `/tmp/k3-kda-early-control-result/seed0.json`
- `/tmp/k3-kda-early-candidate-result/seed0.json`

## Decision

Reject the flag and keep the original packet order. The gain is real but far below a useful B1
structural rung and does not materially close the 48 ms to 20 ms target. Continue using exact
single-block/full-grid sweeps for kernel-local axes, but use packet replay or served TP8 for queue,
counter, state, and cross-operation scheduling axes.
