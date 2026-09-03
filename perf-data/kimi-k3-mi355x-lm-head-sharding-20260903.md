# Kimi K3 TP8 lm_head sharding gate (MI355X, 2026-09-03)

Status: rejected by the multi-metric promotion gate; default remains off.

## Configuration

- Exact TP8 BF16 K3, 8192 random input tokens (`seed=0`) to 256 greedy output tokens.
- Three alternating control/candidate pairs, concurrency 1, no warmup requests.
- Control uses the replicated 163840-row head and `ArgmaxFin`.
- Candidate uses the 20480-row rank-local head and `XArgmaxFin`.
- Both arms use the same frozen runtime and full 60-object production matrix: K3, MLA,
  KDA chunk/intra, XR AttnRes, MoE stage1/stage2/combine, MXFP4, global queue, hierarchy,
  and L2 placement. Packed-prefill consumers and materialized MLA prefill are off.

The candidate reduces `lm_head.weight` per rank from 2,348,810,240 to 293,601,280 bytes
and logits from 327,680 to 40,960 bytes. Stream and instruction counts are unchanged.

## Correctness

All three pairs produced the same 256/256 token IDs and checksum
`fnv1a64:6bdfaa7b84ee4e7e`. Every run completed one request with zero failures. Diagnostics
were complete, supported, and non-overflowed; all eight ranks completed prefill, counter audit
ran on every dispatch, and token agreement was sampled every step.

The CPU equivalence comparator separately checks the rank-zero local logit slice and folds the
full-vocabulary greedy winner across rank-local slices. Its positive and negative self-tests
pass. This avoids treating different full-vocabulary and `vocab/tp` logit shapes as a failure.

## Paired performance

Candidate minus control:

| fold | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| 1 | +8.847286 ms (+0.588%) | -0.099634 ms (-0.225%) | -16.559390 ms (-0.129%) |
| 2 | +8.782353 ms (+0.584%) | -0.063587 ms (-0.143%) | -7.432245 ms (-0.058%) |
| 3 | +6.790981 ms (+0.452%) | -0.101628 ms (-0.229%) | -19.123967 ms (-0.149%) |
| paired mean | **+8.140207 ms (+0.541%)** | **-0.088283 ms (-0.199%)** | **-14.371867 ms (-0.112%)** |

Arm means were 1504.073252 to 1512.213459 ms TTFT, 44.359285 to 44.271002 ms TPOT,
and 12815.691070 to 12801.319203 ms E2E. The TTFT regression repeats in all three folds;
the TPOT and E2E wins are too small to compensate under an all-metrics promotion gate.

## Object and resource gate

The control and candidate directories contain the same 60 filenames. Every one of the 52
control interpreter objects carries packet stamp `0x48a4ccb34189de4a`; every candidate
interpreter carries `0x14deee86a2b35ad2`.

| selected object | wave | private | VGPR spills | SGPR spills | control vs candidate |
|---|---:|---:|---:|---:|---|
| prefill K3+MoE+A4W4 GQ | 64 | 1348 B | 8 | 74 | identical |
| decode K3 GQ | 64 | 624 B | 2 | 84 | identical |
| packed KDA GQ | 64 | 440 B | 0 | 25 | identical |

Sharding does not add an interpreter arm or change object resources; `XArgmaxFin` already
exists in the selected K3 interpreter. Its extra cross-rank counter handoff now dominates the
small amount of head traffic saved. The next prerequisite is to remove or materially reduce
that `XArgmaxFin` handoff cost, then repeat this exact paired gate. Do not enable
`PLOW_K3_SHARD_HEAD` before that result passes TTFT, TPOT, and E2E together.

Run artifacts and hashes are under `/tmp/k3-d7-4936bb9/results` on the measurement host.
