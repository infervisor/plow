# Xeon 6 L2-resident BF16: phase tracker

Status as of 2026-10-09, branch `feat/xeon6-l2-resident-bf16`, head 61163a0c. The campaign is paused and the lab
box (Xeon 6975P-C, 1 socket) is being shut down. This file is the resume point.

## Phases

| phase | scope | status | report | commits |
|---|---|---|---|---|
| P0 | topology, isolation (90 workers, reboot-free), bandwidth tiers, serving baseline | done | [p0_baseline.md](p0_baseline.md) | 0960cf0e |
| P1 | L2 residency: CAT + `pseudo_lock_sram` vs plain THP | done | [p1_residency.md](p1_residency.md) | 8efb0c86 |
| P2 | real BF16 Gemma-4 layer as a weight-stationary stage, FP32 gate | done (E2B, E4B L2/L3 split) | [p2_real_bf16_layer.md](p2_real_bf16_layer.md) | 4a83d3bf |
| P3 | sync / broadcast | done (≤ 4 µs target infeasible; ~20-25 µs barrier per layer at B=1) | [p3_sync.md](p3_sync.md) | 0ac91284 |
| P4 | long-context KV policy (2K-128K) | done (256 KiB tiles + T0 prefetch, no KV lock) | [p4_kv_policy.md](p4_kv_policy.md) | e9804c64, d9d51d87 |
| P5 | batch ≤ 16 with AMX, serving (strict vs vLLM 0.30), `PLOW_CPU_COMBINE=16`, stability | done | [p5_serving_report.md](p5_serving_report.md) | 17921a80, 53581658, 4f03ca12, 681ac441 |
| P6 | E2B / E4B / 12B / 26B-A4B / 31B socket slices through P1-P5; projected cluster | done | [p6_multimodel_cluster.md](p6_multimodel_cluster.md) | 0eb35106, 474673da, 3da0272f |
| P6.8 | MoE experts in L2 + L3 (3 sockets per layer) | measured, GO candidate | p6 § P6.8 | 61163a0c |
| P7 | multi-socket / multi-server experiment | **not started** (needs ≥ 2 sockets) | | |

## Per-model status (single-socket stages)

| model | socket slice | P2 gate | P5 batch | P1 residency | P4 long context | stability | cluster projection |
|---|---|---|---|---|---|---|---|
| E2B | whole layer (pipe plan, 33 sockets at 14 ways) | PASS except single-element `pact` / `pp` rows | done | done | 2K-128K | 36 min | 1 / 28 stages measured |
| E4B | TP2 | FAIL on single-element boundaries only, output passes; traced to one BF16 rounding | done | P2 split only | 128K (P2/P4) | none | 24 / 44 measured |
| 12B | TP4 | PASS | done | done (12 / 14 ways) | 16K warm + cold, 128K | 10 min | 48 / 49 measured |
| 26B-A4B | head socket + expert groups (13 L2-only, or 3 L2 + L3) | PASS | done | done | 16K, 128K (head) | 10 min (expert group) | 55 / 61 measured |
| 31B | TP8 (14 ways) | PASS | done | done (14 ways) | 16K warm + cold, 128K | 10 min | 60 / 61 measured |

## Decisions (do not re-litigate without new data)

* Run L2 residency **unlocked**. Weights stay resident at ≤ 1.6 MiB per core. The lock costs 15-53% at B=16, so it
  is kept as a diagnostic only.
* The P2 gate thresholds are fixed. Single-element boundary failures are recorded as FAIL with their cause.
* plowc decides the partition: `--split tp` (default) for big and MoE layers, `--split pipe` for E2B.
* The batched stage uses AMX for L2-resident slices and AVX for experts streamed from L3.
* Long-context cluster numbers use cold KV (P4 path A).
* MoE: L2-only experts are NO-GO (437 sockets, 78 tok/s per socket). Experts in L2 + L3 at 3 sockets per layer are
  the GO candidate (137 sockets, 250 tok/s per socket, projected).
* llama.cpp comparison dropped; baselines are vLLM 0.30 CPU only.

## Open items, in priority order

1. **P7 two-socket / two-server experiment.**
   * Measure the hop and the per-layer all-reduce at B=1 / 16, over UPI and across servers.
   * Chain two planned stages streaming tokens against a chained reference.
   * Cluster numbers assume hop 5 µs and all-reduce 10 µs; the sensitivity case is 10 / 25 µs.
2. **26B routing trace.** Per-group pair counts and their p99 across many tokens and all 30 layers. Assign experts
   to groups by load.
3. **26B full-layer head as TP2** (predicted only). On 1 socket it is a 590 µs bottleneck at B=16.
4. **plowc expert tier:** L2 + L3 bytes per core from the measured curve, AVX partition. Today it plans 13 L2-only
   groups.
5. **Sequence-parallel attention** for full-attention layers at long context. TP replicates the 1-4 global KV heads.
   Plus one KV tensor for `attention_k_eq_v` layers.
6. **INT8** (all INT8 numbers so far are assumed): the GEMV path and the gate on one E2B slice.
7. LM head as a vocabulary-parallel stage group (predicted only); E2B / E4B KV-shared layers (predicted only).
8. 31B single-socket serving baseline (vLLM + plowrt).
9. Balanced AMX expert partition (tiles across experts): max / mean 1.70 today.

## Resume

Code (in git):

* Bench and tools: `runtime/cpu/bench/l2r/`.
  * `l2r_layer.c`, `ref_layer.py`, `p2_gate.py`, `stage_sum.py`, `cluster.py`;
  * `p4_run.sh` / `p4_ctr.sh`, `p5_stab_sum.py`, `isolate.sh`.
* `runtime/cpu/bench/l2r/campaign/`: the as-run drivers copied from `/tmp/g4c/l2r`.
  * `m_*.sh` (P6), `gen_refs_*.sh`, `plan/*.sh`, `moe/*.sh`, P0-P5 drivers, `vllm-py`, `qos0`.
  * They hard-code `/tmp/g4c/...` paths and the 90-worker layout of this box. Read them as recipes, not portable
    tools.
* The driver is `runtime/cpu/driver/pseudo_lock_sram.c` (Intel only: L2 CAT MSRs, vendor check).
* The planner is `crates/plowc/src/stage_plan.rs`, invoked as `plowc --hf-dir <ckpt> stage-plan`.

Data (not in git):

* `~/g4c-l2r-archive/results-plan-20261009.tar.zst` (37 MB) holds `/tmp/g4c/l2r/results` and `/tmp/g4c/l2r/plan`.
  * It is on the home EBS volume, which survives a stop but not a terminate. Copy it off-box to keep it.
  * It excludes the root-owned P0 `perf` captures.
* **Lost at shutdown, all regenerable:**
  * `/tmp` is tmpfs.
  * The reference dumps (`/tmp/g4c/l2r/ref`, 41 GB) are regenerated by `ref_layer.py` (~12 s per 26B row).
  * The models are under `/tmp/models/google/gemma-4-*-it`, re-download from HF.
  * Built bench binaries; rebuild with the `cc` line in the bench README.

Host bring-up on a fresh box:

1. Run the `isolate.sh apply` workflow, then build and load the driver:
   ```
   sudo insmod pseudo_lock_sram.ko l2_ways=12 l3_ways=0
   sudo chgrp $(id -g) /dev/pseudo_lock
   sudo chmod 0660 /dev/pseudo_lock
   ```
2. Hold a zero-latency CPU QoS request (`qos0`).
3. THP madvise, `numa_balancing=0`.
4. The vLLM 0.30 CPU image (`vllm/vllm-openai-cpu:v0.30.0`) supplies torch / transformers for `ref_layer.py` via
   `vllm-py`.
5. On AWS this needs a bare-metal instance, both for the MSR writes and for core isolation.
