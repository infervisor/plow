# Kimi-K3 B1 KDA gated-norm workgroup fit on MI325X

Date: 2026-08-11

## Decision

Adopt exact workgroup sizing for `KdaGatedNorm`. Each wave owns one `(token, head)` row, so the
packet needs `ceil(T * local_heads / 8)` workgroups rather than all 304 CUs. For TP8 B1 this is two
workgroups for 12 local heads. The kernel body, object, tensor layout, dependencies, arithmetic, and
reduction order are unchanged.

This improves the current KDA double-buffer arm from 49.926 to 48.128 ms/token in a matched full
TP8 run. Served TPOT is 48.232 ms = 20.73 decode tok/s. The B1 50 tok/s goal remains unmet.

## Packet contract

Control asset:
`/home/lava/models/k3_mi325x_b1_ctx131072_v2fp8_kdadb_normctl_wg128`

- packet SHA256: `4c332e9a608607bda979639275eebf95199c4d22de52d34e9e53220d74c57260`
- this is byte-identical to the adopted KDA double-buffer packet

Candidate asset:
`/home/lava/models/k3_mi325x_b1_ctx131072_v2fp8_kdadb_normfit_wg128`

- packet SHA256: `66eb9409ea5f928bbd2a68359eb85a659e16b15ddc5e50e695d56fd53861d43c`
- all seven programs have identical instructions, operands, tensors, waits, counters, and grids
  except the 69 B1 `KdaGatedNorm` packets change from `b=304` to `b=2`
- both use `/home/lava/plow/build-amd/k3-mi325x-b1-v2fp8-kdadb`; no HSACO rebuild is involved

## Truncated-block screen

The five-layer TP8 block assets change exactly four gated-norm packets from 304 to two workgroups.
Three matched runs per arm give:

| arm | ms/token samples | mean |
|---|---|---:|
| control | 4.232, 4.229, 4.232 | 4.231 |
| exact workgroups | 4.102, 4.103, 4.104 | 4.103 |

The 0.128 ms gain over four KDA layers projects to about 2.21 ms over 69 layers. Every run has the
same greedy output SHA256:
`8bcb0fd5f263a9c364bc4c779293b02431487f5df106f57febf90bfdcfcfc970`.

Logs: `/tmp/k3blk-kdanorm-{ctl,fit}-{1,2,3}.log`.

## Full TP8 result

Matched `amd-bench`, ctx5, 128 generated tokens, compact exact TP audit:

| arm | ms/token | decode tok/s |
|---|---:|---:|
| control | 49.926 | 20.03 |
| exact workgroups | 48.128 | 20.78 |

Gain: 1.798 ms/token = 3.60%. Both outputs have SHA256
`3f19cca215335f39535b9665f154a7689923bf54ed05a6ccd223c8e03ac1d456`.

Logs: `/tmp/k3-kdanorm-wg128-{ctl,fit}.log`.

The served vLLM 0.27 C1/N1/input32/output128 cell passes the detailed JSON gate:

- actual input/output: 53/128
- completed/failed: 1/0
- errors: `[\"\"]`; no in-band error
- TTFT: 294.469 ms
- mean TPOT: 48.232 ms
- median ITL: 48.100 ms
- E2EL: 6,419.898 ms
- TTFT-inclusive output throughput: 19.937 tok/s
- steady DSTEP GPU drain: 45.13--45.16 ms; host work: 2.90--3.18 ms; compact audit: about 1.21 ms

Result: `/tmp/k3-kdanorm-serve-result/seed0.json`.

## Reproduction

Emit the matched assets from the production B1/128K recipe; omit only the final environment
variable for the control:

```bash
nix develop --command env \
  K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 PLOW_MLA_PF_V2=1 \
  PLOW_L2_PLACE=1 PLOW_K3_NS=64 PLOW_DECODE_BATCH=1 \
  PLOW_DECODE_BATCH_LADDER=1 PLOW_GLM_GEMV_WG=128 \
  PLOW_K3_KDA_CONV_STEP_DB=1 \
  ./target/release/plowc --hf-dir /home/lava/models/k3_farm \
    --emit devblob --arch gfx942 --gpu MI325X --num-gpus 8 --parallel tp \
    --max-ctx 131072 --n-cu 304 \
    --out /home/lava/models/k3_mi325x_b1_ctx131072_v2fp8_kdadb_normfit_wg128
```

Exact sizing is emitted directly by devgen and does not require a runtime or HSACO flag. Reproduce
the control with the parent commit's `plowc` using the same command and output path.
