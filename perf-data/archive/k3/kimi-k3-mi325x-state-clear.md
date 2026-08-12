# Kimi-K3 recurrent-state admission clear — MI325X TP8

K3 carries 69 recurrent states and 207 convolution windows per rank. Handing a
slot to a new request must clear 56.60 MiB/rank. The control issued 276 blocking
host-staged SDMA fills per rank, sequentially over TP8: 30.830 ms for the main
request (37.099 ms warmup).

`PLOW_STATE_CLEAR_DEVICE=1` builds a load-time descriptor table, enqueues one
`plow_state_clear` kernel on every rank, then drains all ranks. Descriptors split
large states at 256 KiB; there are no admission-time allocations or host data
copies. The serial path remains the unset control. An old decode object is
refused because it cannot resolve `plow_state_clear`.

## Result

Same B32 ladder packet, checkpoint, low-rung tiers, vLLM 0.27 client, input 32,
output 1, concurrency 1, one warmup:

| arm | recurrent clear, main | TTFT, main |
|---|---:|---:|
| serial SDMA control | 30.830 ms | 322.87 ms |
| device scatter-zero | 0.086 ms | 291.97 ms |

The clear improves 99.7% and removes 30.744 ms from admission. A 512-token B1
serve is byte-identical to the established low-rung control (18.635 vs 18.541
tok/s). A C32/N32/output64 run completed 32/32 with 2,048/2,048 output tokens,
empty errors, and every text equal to the corresponding prefix of the established
output128 control. Two subsequent C32/N32/output1 runs are byte-identical across
all 32 slots, proving reuse clears slots 1–31 after they carry live state.

The rejected predecessor batched 276 SDMA copies under one completion signal but
left main clear unchanged at 30.835 ms. The bottleneck was fine-grained host
staging bandwidth, not signal waits.

## Build and run

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /home/lava/plow/build-amd/k3-b32-state-clear-device

nix develop --command env \
  PLOW_HSACO=/tmp/k3-b32-state-clear-hsaco \
  PLOW_STATE_CLEAR_DEVICE=1 PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_TP_AUDIT_COMPACT=1 PLOW_CTR_DBUF=1 \
  PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1,/home/lava/plow/build-amd/k3-b2-ladder-grouped:2,/home/lava/plow/build-amd/k3-b4-ladder-grouped:4,/home/lava/plow/build-amd/k3-b8-ladder-grouped:8 \
  PLOW_TTFT_LOG=1 \
  perf-data/tools/gpulease -n 8 k3-state-clear-device \
  ./target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x_ladder_router --port 8018
```

Raw client JSON:

- `/tmp/k3-state-clear-control-result/seed0.json`
- `/tmp/k3-state-clear-device-result/seed0.json`
- `/tmp/k3-state-clear-device-out512/seed0.json`
- `/tmp/k3-state-clear-device-c32-out64/seed0.json`
- `/tmp/k3-state-clear-reuse-{1,2}/seed0.json`

Static gate: both B32 decode objects remain at 256 VGPR and 64,560/64,568 B
LDS with 32 reported spills. `plow_state_clear` is spill-free; both generic
gfx942 and grouped A4W4 audits pass.
