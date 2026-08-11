# Kimi-K3 MI325X grouped-MoE tile lookup

Date: 2026-08-11. Hardware: 8x MI325X, TP8, gfx942. Toolchain: repository
Nix ROCm 7.14.0. Client: vLLM 0.27.0 `bench serve`, one warmup.

## Result

K3 has 896 experts. The grouped GLU and DOWN kernels previously found the
expert owning each M tile by scanning the expert tile-prefix table from expert
zero. The lookup is repeated across the GLU and DOWN N tiles. Replacing that
linear scan with an upper-bound binary search removes the repeated O(E) scalar
walk while preserving duplicate prefix entries for empty experts.

Matched C32/N32, random input 32, output 512:

| seed | linear tok/s | binary tok/s | change | linear mean TPOT | binary mean TPOT |
|---:|---:|---:|---:|---:|---:|
| 0 | 118.910 | 144.847 | +21.81% | 246.585 ms | 200.180 ms |
| 1 | 117.311 | 144.116 | +22.85% | 250.321 ms | 201.621 ms |

The final default build completed a longer C32/N32, input 32, output 2048
soak:

| metric | result |
|---|---:|
| completed / failed | 32 / 0 |
| generated tokens | 65,536 |
| output throughput | **161.121 tok/s** |
| median TTFT | 7,641.64 ms |
| mean / median TPOT | 193.445 / 193.558 ms |
| duration | 406.750 s |

This is aggregate throughput across 32 simultaneous streams. It is not B1
output speed. The previous same-shape ladder result was 131.162 tok/s; the new
long soak is 22.84% higher. Final DSTEP samples put B32 GPU drain near 186.4
ms/step and host work near 3.4 ms/step, so the remaining decode cost is still
device-side.

## Correctness

Every reported cell completed all requests and exact output-token totals, had
empty vLLM error strings, no in-band `[error:` text, and ran with compact TP
counter auditing. Post-run `gpulease --audit` found no foreign GPU process.

Random prompts are sensitive to an existing admission-order nondeterminism:
different-speed A/B arms can assign two prompt positions different valid
greedy continuations. To isolate the kernel, both objects were also run on 32
identical prompts from `/tmp/k3-identical32.jsonl` (SHA256
`d7f552d6178fecf198d4aa63023c74ff513bfb74abc855a8f348e84c6f97a7c2`).
The control and candidate `generated_texts` arrays are position-wise identical
32/32. Each arm independently has the same response multiplicities
`[30,1,1]`. This proves the tile selector itself preserves the live execution;
the baseline scheduling nondeterminism remains separate runtime work.

The binary search is default-on only for batched K3 decode objects. B1 and K3
prefill remain byte-compatible with their prior build flags. The opt-out is
`PLOW_K3_DECODE_TILE_BINSEARCH=0` in the script or
`-DPLOW_K3_DECODE_TILE_BINSEARCH=OFF` in CMake.

## Static gates

Both default and opt-out objects pass the gfx942 and grouped A4W4 ISA audits.
Resources are unchanged: 256 VGPR, 64,560/64,568 B LDS, and 32 reported spills
for static/GQ. `plow_exec` falls from 37,804 to 37,788 instructions and from
7,692 to 7,676 SALU instructions; scratch instruction count stays 673.

Final default hashes:

```text
16066e31d7223e0f76521f64ee3097e5aafe4183e812c3fedc4635d4727435f0  interp_decode_fp8kv_k3.elf
35560c8669aada7a8c7e67361ce115bee74d7a1c0ee84c3e40ce5d0f716aafdb  interp_decode_fp8kv_k3_gq.elf
```

Final opt-out hashes:

```text
dda7a551f1c652e3ac6be55ff7bd9be801268e33897f608c375fb1f3c1331ead  interp_decode_fp8kv_k3.elf
6a350176dd0d12b611f188e790e2cad3278336bccd2d74bfe4e28aad877445e4  interp_decode_fp8kv_k3_gq.elf
```

The final default and the initially measured candidate have identical
disassembly bodies; only HIP's generated compilation-unit symbol differs.
CMake generation was checked with the option both ON and OFF: the define is
present only on the four K3 decode rows and absent from prefill rows.
Fresh B1 default/opt-out static and GQ objects are byte-identical, and the
script rejects option values other than zero or one before compilation.

## Reproduction

Build the default and opt-out objects:

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b32-tile-default

nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 PLOW_K3_DECODE_TILE_BINSEARCH=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b32-tile-linear
```

The server uses the final objects overlaid on the complete V2 HSACO tree:

```bash
nix develop --command env \
  PLOW_MLA_PF_V2=1 PLOW_HSACO=/tmp/k3-b32-tile-default-serve \
  PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_AUDIT_COMPACT=1 PLOW_CTR_DBUF=1 \
  PLOW_DSTEP_LOG=1 PLOW_DSTEP_EVERY=64 \
  PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1,/home/lava/plow/build-amd/k3-b2-ladder-grouped:2,/home/lava/plow/build-amd/k3-b4-ladder-grouped:4,/home/lava/plow/build-amd/k3-b8-ladder-grouped:8 \
  perf-data/harness/gpulease -n 8 k3-tile-default \
  ./target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x_ladder_router_v2fp8_seg2 --port 8054
```

The long client command is:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8054 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 32 --random-output-len 2048 \
  --random-range-ratio 0 --request-rate inf --max-concurrency 32 \
  --num-prompts 32 --num-warmups 1 --ignore-eos --temperature 0 --seed 0 \
  --save-result --save-detailed \
  --result-dir /tmp/k3-tile-default-final-out2048 \
  --result-filename seed0.json
```

Raw JSON evidence:

```text
/tmp/k3-tile-bsearch-ab/{control,candidate}/out512-seed{0,1}.json
/tmp/k3-tile-bsearch-ab/{control,candidate}/identical-out128.json
/tmp/k3-tile-default-final-out2048/seed0.json
```
