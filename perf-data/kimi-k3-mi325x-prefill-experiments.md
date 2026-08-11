# Kimi-K3 MI325X prefill experiments

Date: 2026-08-11. Hardware: 8x MI325X (`gfx942`, 304 CU/card).
Toolchain: flake ROCm 7.14.0. Client: flake vLLM 0.27.0. Both axes were
default-off and are rejected.

| experiment | baseline | candidate | result |
|---|---:|---:|---:|
| KDA state residency, production drain | 4504.566 ms | 4510.505 ms | +0.132% |
| KDA state residency, client TTFT | 4568.470 ms | 4574.140 ms | +0.124% |
| selected-W2 touch, combined GLU+DOWN | 0.334506 ms | 0.373140 ms | +11.550% |

## KDA state residency

The roofline hypothesis was that K3 TP8 prefill rereads and rewrites the KDA
state inside the serial token loop. With `H=12`, `D=128`, 69 KDA layers, and
8149 real input rows, the state traffic estimate is
`12*128*128*8*69*8149 = 884.392 GB/rank`. At the measured 4.164 TB/s HBM
roof this is a 212.390 ms lower bound. The candidate assigned one state column
to each wave, loaded two f32 values per lane before the `T` loop, retained them
across the loop, and stored them once afterward.

Static gates:

- The default static/GQ objects were byte-identical to the pre-change objects:
  `191dbe432fbe2cc7b1ccb631c7204440f028ae5c1cf2d52d0d7f5a2d97809bff`
  and `91eeb45d459866838a128c88fd33751d0b0583c2136837738f4219fcbf73f05c`.
- Candidate static/GQ resources matched control: 256 VGPR, 0 AGPR, 64560 B
  LDS; spill counts remained 8 and 30. The full gfx942 assembly audit passed.
- The focused wrapper used 62 VGPR, occupancy 8, and no spill. Disassembly
  showed two state loads before the serial loop and two final stores.
- The f64 oracle passed BV8 serial prefill and BV16 batched decode. BV8 state
  RMS was `7.744e-08`; output RMS was `1.652e-03`. BV16 state RMS was
  `3.606e-08`; output RMS was `1.655e-03`.

Build and focused oracle:

```bash
nix develop --command env PLOW_ROWS_ONLY=interp_prefill_fp8kv_k3_moe_a4w4 \
  JOBS=1 scripts/build_gfx942.sh /tmp/k3-kda-state-default
nix develop --command env PLOW_ROWS_ONLY=interp_prefill_fp8kv_k3_moe_a4w4 \
  PLOW_KDA_PF_STATE_RESIDENT=1 JOBS=1 \
  scripts/build_gfx942.sh /tmp/k3-kda-state-resident
nix develop --command cmake -S runtime -B /tmp/k3-kda-oracle-cmake-clang \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DPLOW_HIP_ARCH=gfx942 \
  -DCMAKE_HIP_COMPILER=/nix/store/9i3g77yxafyrsiphzmpljmq6j5xj4imx-rocm-therock-gfx94X-dcgpu-7.14.0/lib/llvm/bin/clang++
nix develop --command cmake --build /tmp/k3-kda-oracle-cmake-clang \
  --target kda_state_resident_gfx942_test -- -j1
nix develop --command perf-data/harness/gpulease -n 1 \
  kda-state-resident-oracle \
  /tmp/k3-kda-oracle-cmake-clang/bench/kda_state_resident_gfx942_test \
  /tmp/k3-kda-oracle-cmake-clang/bench/kda_state_resident_gfx942.elf
```

The production A/B used the same B16 asset, compact TP audit, global-queue
objects, one vLLM warmup, one measured prompt, and one output token. The chat
template produced 8149 input tokens from the requested 8192-token prompt.

```bash
# Run in one terminal with ARM=base PORT=8027, then ARM=resident PORT=8028.
ARM=base PORT=8027
nix develop --command env \
  PLOW_HSACO="/tmp/k3-kda-state-serve-$ARM" \
  PLOW_TP_AUDIT_COMPACT=1 PLOW_TTFT_LOG=1 \
  perf-data/harness/gpulease -n 8 "k3-kda-prefill-$ARM" \
  ./target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x_b16 --port "$PORT"

nix develop --command vllm bench serve --backend openai-chat \
  --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
  --model k3_farm --tokenizer /home/lava/models/k3_tokz \
  --dataset-name random --random-input-len 8192 --random-output-len 1 \
  --random-range-ratio 0 --max-concurrency 1 --num-prompts 1 \
  --num-warmups 1 --ignore-eos --seed 0 --save-result \
  --result-dir "/tmp/k3-kda-prefill-$ARM-result" \
  --result-filename seed0.json
```

Artifacts are `/tmp/k3-kda-prefill-{base,resident}-server.log`,
`/tmp/k3-kda-prefill-{base,resident}-result/seed0.json`, and the object trees
`/tmp/k3-kda-state-serve-{base,resident}`. The measured production drain was
4504.566 vs 4510.505 ms; client TTFT was 4568.470 vs 4574.140 ms. Both arms
used the same safe 8-wave flash fallback because their flash object lacked the
K3 marker. Reject: the state-resident candidate did not improve production
prefill latency.

## Selected-expert W2 touch

The Laneformer-style hypothesis was that GLU workgroups could issue cacheable
lookahead loads for only routed experts' W2 payload and E8M0 scales, using the
same future DOWN `(slice,nblk)` ownership so useful data reached the local L2
before DOWN. The standalone K3 B4/TP8 case used 64 routed rows, 1024 padded
rows, top-16 routing, 896 experts, `H=3584`, and `I_tp8=384`. It touched one
byte per 64-byte cache line over 11,698,176 requested bytes (11.156 MiB).

Static and correctness gates:

- Baseline and candidate full GLU bridge bytes, E8M0 scales, and DOWN f32
  output were bit-identical. The touch checksum was nonzero.
- The independent f64 oracle passed: bridge RMS `0.1550` at the E2M1
  quantization floor, quantized-value mismatch `0%`, and DOWN RMS `2.548e-08`
  with worst error `5.953e-08`.
- Both arms used 248 VGPR, 40960 B LDS, and a 104 B private segment. The touch
  wrapper had no scratch or spill and emitted the intended byte load.

Build, oracle, and clean one-GPU timing:

```bash
nix develop --command cmake --build /tmp/k3-kda-oracle-cmake-clang \
  --target moe_prefill_a4w4_cdna3_test -- -j1
nix develop --command perf-data/harness/gpulease -n 1 \
  a4w4-w2-touch-oracle \
  /tmp/k3-kda-oracle-cmake-clang/bench/moe_prefill_a4w4_cdna3_test \
  /tmp/k3-kda-oracle-cmake-clang/bench/moe_prefill_a4w4_w2_touch_gfx942.elf
nix develop --command env MPA4C3_W2_TOUCH_BENCH=1 \
  MPA4C3_JSONL=/tmp/k3-w2-touch-clean.jsonl \
  PLOW_GPU=MI325X PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix \
  PLOW_BUILD_ID=w2-touch-43960b \
  PLOW_LEASE_LABEL=a4w4-w2-touch-clean \
  perf-data/harness/gpulease -n 1 a4w4-w2-touch-clean \
  /tmp/k3-kda-oracle-cmake-clang/bench/moe_prefill_a4w4_cdna3_test \
  /tmp/k3-kda-oracle-cmake-clang/bench/moe_prefill_a4w4_w2_touch_gfx942.elf
```

The clean run used 10 warmups and 12 interleaved paired samples with four
GLU+DOWN chains per sample. Median latency was 0.334506 vs 0.373140 ms; the
candidate won 0/12 pairs. Provenance and raw samples are in
`/tmp/k3-w2-touch-clean.jsonl`; the tested object is
`/tmp/k3-kda-oracle-cmake-clang/bench/moe_prefill_a4w4_w2_touch_gfx942.elf`
(`w2-touch-43960b`). Reject: the candidate added the 11.156 MiB touch without
a combined-latency win, so it is not promoted to production packets/runtime.
