# Kimi-K3 B1 double-buffered KDA Conv3/state fusion

Date: 2026-08-11. Hardware: 8x MI325X (`gfx942`, TP8). Toolchain: repository Nix
ROCm 7.14. Client: vLLM 0.27 `bench serve` against `plowrt serve`.

## Decision

Keep `PLOW_K3_KDA_CONV_STEP_DB=1` as an explicit B1 performance arm. It improves served TPOT
52.343 -> 50.140 ms (-4.21%, 18.44 -> 19.94 decode tok/s) and clears the predefined GSM8K gate at
196/200 with zero request errors. It stays default-off: the combined body changes BF16 reduction
association, first flips a near-tied greedy token at decode step 32, and scores one question below
the established 197/200 B1 baseline.

This is progress toward, not completion of, the 20 ms / 50 tok/s B1 target. The candidate remains
about 30.1 ms above the short-context goal.

## Contract

The shipping `KdaConv3 -> KdaStateStepG` chain cannot update its convolution window in place when
combined: the 16 value-column workgroups for a head all need the old q/k window, while one of them
would overwrite it before its siblings necessarily read it. The candidate therefore:

- allocates two q/k/v convolution-window banks per KDA layer;
- mirrors the legacy prefill bank once before decode;
- selects the source bank from `in.pos & 1` and writes the opposite bank;
- snapshots, restores, and clears both banks with the rest of recurrent state;
- skips a parked sequence before touching either recurrent state or output; and
- requires exact packet/object capability pairing through `plow_kda_conv_step_db_arm`.

The packet carries a 13-handle descriptor in `t7` and the new typed opcode
`KdaConvStateStepG = 120`. It is B1-only. Prefill retains the legacy Conv3/state packets.

## Artifacts and static gates

| artifact | control | candidate |
|---|---|---|
| packet | `7de1b6b7f86c950ece219a850dc6899f88eb90da55c22608b6c5f07610a15c42` | `4c332e9a608607bda979639275eebf95199c4d22de52d34e9e53220d74c57260` |
| decode static ELF | `8da1dbb606ade82e7bf4be7412a852f6492acebce27d6bf4f20326ee4202c8d1` | `2c1ee0f4c99d0fb8ff6c79a8a3edf703dd6a8ab22a155dcf84469185a13c293c` |
| decode GQ ELF | `834c29b4a98a8175e6af59ded0800d043190674fd8150bdfcd344bdc603bcf7f` | `1b73d5d8e434228738e988056f7f959eadc3ac7542df68c3c372fa3dea0e1d39` |
| decode packets / counters | 2,343 / 58,575 | 2,274 / 56,850 |
| critical-path levels | 1,715 | 1,715 |
| GQ VGPR / LDS / spill | 255 / 64,568 B / 3 | 255 / 64,568 B / 3 |
| `plow_exec` instructions / scratch | 33,197 / 228 | 32,849 / 233 |

The candidate has exactly 69 `KdaConvStateStepG`, zero legacy `KdaConv3`/`KdaStateStepG`, and
69 unchanged `KdaGatedNorm` packets. The T=128 prefill program retains 69 legacy Conv3 and state
steps and contains no candidate opcode. The full gfx942 ISA audit passes for both static and GQ
objects. CMake and `scripts/build_gfx942.sh` carry the opt-in only on K3 decode rows.

## TP8 performance

Matched real-weight `amd-bench`, 64 decode steps, prompt ids
`1008,10484,318,15383,387`, context 5:

| arm | ms/token | tok/s | steady GPU drain |
|---|---:|---:|---:|
| control | 53.718 | 18.6 | about 49.31 ms |
| candidate | **51.651** | **19.4** | about **47.06 ms** |

Matched served vLLM 0.27 C1/N1, requested random input 32 (actual chat input 53), output 128,
one warmup, seed 0, temperature 0:

| arm | TTFT | TPOT | median ITL | output tok/s including TTFT |
|---|---:|---:|---:|---:|
| control | 294.401 ms | 52.343 ms | 52.332 ms | 18.437 |
| candidate | 294.802 ms | **50.140 ms** | **50.004 ms** | **19.210** |

Both cells completed 1/1, produced exactly 128 tokens, and reported no request or in-band error.
Compact exact TP counter auditing and all-rank token agreement were active. The 2.2 ms served gain
is in GPU drain, not host launch time.

Detailed client JSON:

- `/tmp/k3-kdactl-serve-result/seed0.json`
- `/tmp/k3-kdadb-serve-result/seed0.json`

## Numerics and quality

The prefill logits are byte-identical. Decode argmax remains equal through step 31. At step 32 the
control chooses token 37079 and the candidate token 276; the control margin is 0.125 and the
candidate is an exact tie. Later logit comparisons are different token histories and are not a
valid per-step error measure. Evidence: `/tmp/k3-kdadb-logit-compare.txt`.

The adoption gate used the first 200 GSM8K questions, 8-shot chain of thought, greedy decoding,
and a 320-token cap:

- Paris coherence: PASS.
- Exact match: **196/200 = 0.9800**.
- Request errors: **0**.
- Median/mean latency: 5.55/5.99 seconds; total 1,198 seconds.
- Strict server fault/rank/counter scan: empty.
- Test/train SHA256:
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` /
  `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.

Server evidence: `/tmp/gsm8k_serve_8057.log` SHA256
`b3d47f7182d733f5271b3cac81a7c0051be71b71b39dcbf168f1d4487c6fe687`; strict error scan
`/tmp/k3-kdadb-gsm200-audit-errors-strict.txt` is empty.

## Reproduction

Build the candidate decode object; omit the KDA flag for the matched control:

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=1 PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 PLOW_K3_KDA_CONV_STEP_DB=1 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-kda-db-prod
```

Emit from the canonical K3 B1/128K recipe, changing only the packet flag:

```bash
nix develop --command env \
  K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 PLOW_MLA_PF_V2=1 \
  PLOW_L2_PLACE=1 PLOW_K3_NS=64 PLOW_DECODE_BATCH=1 \
  PLOW_DECODE_BATCH_LADDER=1 PLOW_GLM_GEMV_WG=128 \
  PLOW_K3_KDA_CONV_STEP_DB=1 \
  ./target/release/plowc --hf-dir /home/lava/models/k3_farm \
    --emit devblob --arch gfx942 --gpu MI325X --num-gpus 8 --parallel tp \
    --max-ctx 131072 --n-cu 304 \
    --out /home/lava/models/k3_mi325x_b1_ctx131072_v2fp8_kdadb4
```

Run the quality gate under one TP8 lease:

```bash
nix develop -c env \
  PLOW_MLA_PF_V2=1 PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_TP_AUDIT_COMPACT=1 PLOW_CTR_DBUF=1 PLOW_STATE_CLEAR_DEVICE=1 \
  N=200 SHOTS=8 MAXTOK=320 CONC=1 GSM8K_TEMPERATURE=0 \
  PLOWRT_BIN=$PWD/target/release/plowrt GPU_LEASE_DIR=/tmp/plow-gpulease-shared \
  perf-data/tools/gpulease -n 8 k3-kdadb-gsm200 \
  scripts/bench_gsm8k.sh \
    /home/lava/models/k3_mi325x_b1_ctx131072_v2fp8_kdadb4 8057 auto 1800
```
