# Kimi-K3 B1 exact-op interpreter subset (rejected)

Date: 2026-08-11. Hardware: 8x MI325X, TP8, gfx942. Toolchain: repository Nix
ROCm 7.14.0. Client: vLLM 0.27.0 `bench serve`, one warmup.

## Hypothesis

The adopted B1 packet uses 23 of the decode interpreter's dispatch opcodes. A default-off
candidate physically omitted every other arm, specialized the two live head-normalization
widths, advertised a capability marker, and made the loader reject any packet containing an
opcode outside the compiled set. Kernel bodies, packet bytes, counters, grids, weights, and
numeric order were unchanged.

## Static result

Both control and candidate retained 255 VGPR, 64,568 B LDS, and 3 VGPR spills in the GQ kernel.
The candidate passed its exact grouped-MXFP4 instruction-selection contract.

| GQ object | Control | Candidate | Delta |
|---|---:|---:|---:|
| ELF bytes | 576,648 | 210,928 | -63.42% |
| `plow_exec` instructions | 33,213 | 16,253 | -51.06% |
| `plow_exec` scratch instructions | 228 | 228 | 0 |

Control SHA256: `319688d9a07c04ccc1a61b746e8e8bbb901458371ca39eb233aa1e188f5ee074`.
Candidate SHA256: `bdd19079c2a50c99578e89791c3a8476ec10130e46badcf499687c5e3550791a`.

Build flags were identical except `PLOW_K3_B1_MINIMAL=1` on the candidate:

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=1 PLOW_K3_DECODE_GROUPED=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 PLOW_K3_B1_MINIMAL=1 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b1-minimal-candidate
```

## Served A/B

Asset: `/home/lava/models/k3_mi325x_b1_ctx131072_ns64`. Both arms used FP8 MLA V2,
L2 placement, exact compact TP counter audit, counter double buffering, device recurrent-state
clear, and the canonical B2/B4/B8 low-rung objects. The measured cell was C1, one request,
actual input 149, output 512, seed 0, temperature 0, ignore EOS.

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| TPOT | 53.387 ms | 53.198 ms | -0.189 ms (-0.35%) |
| output throughput | 18.501 tok/s | 18.566 tok/s | +0.35% |
| TTFT | 392.218 ms | 392.304 ms | +0.086 ms |
| E2EL | 27,672.964 ms | 27,576.426 ms | -0.35% |
| steady GPU drain | about 50.18 ms | about 50.01 ms | about -0.17 ms |

Both arms completed 1/1, generated exactly 512 tokens, returned empty errors, produced
byte-identical text, and passed every-rank compact counter auditing. Detailed JSON is retained at
`/tmp/k3-b1-minimal-result/{control,candidate}.json` on the measurement host.

The client command was:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:PORT \
  --endpoint /v1/chat/completions --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 128 --random-output-len 512 \
  --random-range-ratio 0 --request-rate inf --max-concurrency 1 \
  --num-prompts 1 --num-warmups 1 --ignore-eos --temperature 0 --seed 0 \
  --save-result --save-detailed
```

## Decision

Reject and remove the axis. Halving the interpreter body saves only 0.19 ms, far below the
roughly 33.4 ms required to reach 20 ms/token at short context. The result disproves instruction
footprint as a major B1 limiter; the next experiments must change live kernel work or overlap.
The 128K arm was not run because the short-context stop gate failed.
