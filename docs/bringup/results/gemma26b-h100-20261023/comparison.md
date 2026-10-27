# Gemma-4-26B-A4B-it comparison, KV block store (1x H100, vLLM 0.29)

Qualified wins only: `campaign.py report` MATCHED + EQUIVALENT on the open-loop production mix
(`--prod-fidelity`, think lognormal 10 s), where total throughput is pinned by the arrival rate and
the wins are the latency rows. Every matched row is in [comparison.csv](comparison.csv).

- Infervisor: plowrt `f7eba863` (branch `kv-block-store`: the KV block store is the only boundary
  storage, `Prefix` blocks hold every full-layer row, nothing is multi-mapped;
  `docs/arch/26-kv-block-store.md` B.6a), FP8 packet `b86ec26e4262` from
  [`sm90a-h100-tp1-fp8.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-fp8.toml)
  (Kit 2 assets, max_ctx 16384). Serve defaults from the packet; server env
  `PLOW_VMM_LIVE_RINGS=1 PLOW_VMM_RING_PARK=8 PLOW_PF_ADMIT_OVERLAP=1 PLOW_DECODE_PIPELINE=1
  PLOW_PF_INTERLEAVE=0 PLOW_PF_ITL_FORCE=0 PLOW_VMM_STRICT_PUBLISH=1 PLOW_PF_ITL_BUDGET_MS=60
  PLOW_VMM_HOST_CACHE_MIB=0` (the host tier under the store costs TPOT at 10 s think, below) and the
  Kit 2 cuBLASLt 13.4 on the library path.
- Baseline: vLLM 0.29.0, same client, `fp8` KV, `--max-model-len 16384 --gpu-memory-utilization
  0.93`, prefix caching on. 2 repeats per cell, one server per arm, same seeds per repeat.
- Quality: FP32-reference gate PASS on the served packet (kl_mean 0.2034 vs vLLM 0.202,
  kl_trim_mean 0.1239 vs 0.1242, top1_decisive 0.9622, cont_frac 0.5348, needle 1.0).
- Raw evidence (gpu03 scratch, outside the repo): `/home/lava/scratch/kvstore-strict/{plow3,vllm2,gate,report3}`;
  replays `/home/lava/scratch/kvstore-gates/{r27-7-nohost,long7-nohost,long10-auto-end,r27-10-auto}`;
  engine-thread audit `packlog-{store,oldsnap}/packlog.txt`.

| Cell | Precision | Total throughput | TTFT P99 | TPOT P99 |
|---|---|---:|---:|---:|
| prod mix, 1.8 sessions/s | FP8 | 1.00x | 0.68x | 0.55x |
| prod mix, 2.7 sessions/s | FP8 | 1.01x * | 0.71x * | 0.64x * |
| agentic16k c32 (closed loop) | FP8 | 1.46x | 0.48x | 0.58x |
| agentic16k c64 (closed loop) | FP8 | 1.40x | 0.84x | 0.60x |

`*` = repeat spread > 10% (FLAGGED; direction only; the 2.7 cell's throughput spread is the
arrival process, identical on both arms). Supplementary, Baseline / Infervisor: TTFT P50 69.7 /
36.6 ms and TPOT P50 10.48 / 5.09 ms at 1.8; 119.3 / 53.3 ms and 16.19 / 8.37 ms at 2.7; cached
prompt tokens 80.7 % / 78.6 % (Infervisor); SLO met 100 % on both. Agentic16k (10 turns growing to
15,600 tokens, 128-token replies, closed loop; `llm_grid.sh --agentic`, raw evidence
`/home/lava/scratch/kvstore-agentic/{plow,vllm,report}`): server prefix hit 84.4 % / 84.5 % (Infervisor)
vs 83.2-83.5 % (vLLM); the c32 baseline TTFT P99 spread is flagged (34 %), direction only.

## Prefix cache retention on the production replay (not strict rows)

Ten-minute production-fidelity replays on gpu03, same stream per arm, sampled decoding
(`long_plow_arm.sh`; vLLM numbers from the same harness on 2026-10-22):

| Think median | Arm | Hit rate | TTFT p50/p99 ms | TPOT p50/p99 ms | Goodput |
|---|---|---:|---:|---:|---:|
| 10 s, rate 2.7 | store, no host tier (`f7eba863`) | 78.6 % | 53 / 358 | 10.1 / 37.7 | 100 % |
| 10 s, rate 2.7 | store + host tier (`120a2202`, auto, prefetch restores, admission hold) | 79.7 % | 65 / 707 | 16.5 / 89.2 | 99.5 % |
| 10 s, rate 2.7 | snapshot path (`a803cf92`, store off) | 74.6 % | 66 / 551 | 13.9 / 46.6 | 100 % |
| 10 s, rate 2.7 | vLLM 0.29 | 75.6 % | 119 / 592 | 16.9 / 61.6 | |
| 20 s, rate 1.8 | store, no host tier (`f7eba863`) | 71.7 % | 58 / 317 | 6.0 / 29.5 | 100 % |
| 20 s, rate 1.8 | store + host tier (`9d8fad4b`, auto) | 78.4 % | 53 / 446 | 8.1 / 41.1 | 100 % |
| 20 s, rate 1.8 | snapshot path (`a803cf92`, store off) | 67.1 % | 63 / 408 | 6.8 / 33.8 | 100 % |
| 20 s, rate 1.8 | vLLM 0.29 | 72.6 % | 100 / 489 | 12.2 / 42.1 | 99.6 % |
| 45 s, rate 1.8 | store, no host tier (`f7eba863`) | 58.3 % | 87 / 401 | 6.3 / 34.6 | 100 % |
| 45 s, rate 1.8 | store + host tier (`120a2202`, auto, session-end demotion) | 71.7 % | 70 / 376 | 6.5 / 31.8 | 100 % |
| 45 s, rate 1.8 | snapshot path (`a803cf92`, store off) | 48.0 % | | | |
| 45 s, rate 1.8 | vLLM 0.29 | 60.6 % | 147 / 569 | 13.5 / 53.0 | |

Without a host tier the store is within the ideal's reach at 10 s think (ideal 78.9 %), 7 points
under it at 20 s (ideal 78.5 %, vLLM 0.9 points ahead) and capacity-bound at 45 s think (the LRU drops boundaries at a
median age of 45 s, the think-time median). With the host tier it holds 71.7 % against an ideal
of 73.9 % (the rest is session-end demotion) at 45 s think and 78.4 % at 20 s, with lower TTFT
and TPOT than vLLM at both and than the store without the tier at 45 s; at 10 s think the tier
adds one point (79.7 % vs 78.6 %) and costs TTFT 65/707 ms and TPOT 16.5/89 ms (a per-tick cost
under heavy spilling, amplified by the speculation policy at the larger in-flight batch;
`docs/arch/26-kv-block-store.md` B.6a), so the strict rows above run without it. The recipe
keeps the tier at its default (auto); the production setting depends on the deployment's think
time, and production's is about 45 s. Raw evidence for the tier rows:
`/home/lava/scratch/kvstore-gates/{r27-15-auto,t20-12-auto,long15-auto-end}`.

## Production pass on the branch (2026-10-25; gates and strict reports per recipe, not win rows)

Every BF16/FP8 recipe worked on this branch was re-emitted from the final source, gated on the
branch binary with the store armed, and, where a vLLM 0.29 peer exists, run through
`campaign.py report`. Only the Gemma-4 26B FP8 rows above are qualified wins; the rest is
recorded here for the record, losses included.

| Recipe (packet) | Gate on the branch | Strict report vs vLLM 0.29 |
|---|---|---|
| gemma-4-26b-a4b FP8 (Kit 2 `b86ec26e4262`; re-emit `faaebced1d4a` keeps its `union`/`tuning`/weights) | c16 FP32 PASS on the re-emit: kl_trim 0.1332 vs 0.1242, top1_decisive 0.9582, cont 0.539 | host tier **auto** (the recipe default), prod mix: MATCHED + EQUIVALENT, 1.8/s tok/s 1.00x, TTFT P99 1.04x, TPOT P99 0.82x*; 2.7/s 1.00x / 1.52x* / 1.09x* (the tier pays at 10 s think; the win rows above run with it off) |
| gemma-4-26b-a4b BF16 (`475cb17f9986`, reproduced byte for byte) | c16 FP32 PASS: kl_mean 0.0899 vs 0.1005, kl_trim 0.0462 vs 0.0496, top1_decisive 0.9838 vs 0.9778 | not re-run |
| gemma-4-12b (`06da7686`, reproduced byte for byte from the re-keyed checkpoint) | c16 FP32 PASS (row below) | prod mix, fp8 KV both arms: MATCHED + EQUIVALENT and **loses**: 1.8/s tok/s 0.98x, TTFT P99 1.53x*, TPOT P99 1.65x; 2.7/s 0.95x / 2.62x* / 1.20x*, SLO met 54 % vs 96 % |
| qwen3.8-27b BF16 (`5f934c703d2c`) | prefix cache **on** c1/c8/c32 FP32 PASS: kl_trim 0.000436/0.000470/0.000456 vs 0.000338–0.000356, top1 1/0.9988/1 | agentic16k on vs off (same packet): c8 hit 82 %, out tok/s 180.6 vs 110.7, TTFT p50/p99 483/2333 vs 1721/3081 ms, TPOT 40.0/50.7 vs 58.5/81.8 ms; c32 hit 8–10 % (store capacity beside 48 live slots), within 4 % of off |
| qwen3.8-27b FP8 (`f71e777dea43`, decode ladder 1..48) | c1/c8/c32 FP32 PASS on default thresholds vs vLLM 0.29 hub-FP8 peers: kl_mean 0.0137/0.0148/0.0141 vs 0.0204/0.0213/0.0246, top1 0.9963/0.9951/0.9988 | ISL 1000/OSL 128: NOT MATCHED on model version by construction (plow re-quantizes BF16 per row, vLLM serves the hub's block-scaled FP8); c1 tok/s 0.51x, c8 0.22x, c32 0.23x; TPOT P99 1.9x / 5.2x / 4.8x. Serves where it refused before, not competitive |
| gemma-4-e4b (re-emit `ab2afc639357`, cuBLASLt decode from 1 row + generated prefill roles) | logit parity PASS after the kv-shared tail fix: top1 0.9948, kl_mean 7.5e-4, kl_max 0.0154 (tail route on, prefill-bucket path). Before the fix the same packet failed (top1 0.9754, kl_mean 0.0546, kl_max 7.22): only the first generated token differed, every later step bit-identical. Attribution (`e4b-arms`): prefix cache off, per-rung Lt algorithms off, segment graphs off and a repeat all fail with identical digits; `PLOW_PF_SHARED_TAIL=0` passes; the kit packet `3146fe3e` with the vendor attention off runs the tail on a decode rung and passes; the recipe with cuBLASLt decode from 48 rows (`e4b-3`) runs the tail on a decode rung and passes. Cause: the prefill-bucket tail path uploaded the chain's request table but not the attention's, so the sampled row attended over the first prompt token alone. | — |

Raw evidence (gpu03 NVMe scratch): `/opt/dlami/nvme/lava-tts/kvstore-scratch/{strict-g26-tier23,strict-g12-23,gemma-final-gates,qwenbf16-gate24,qwenbf16-prefix-ab,qwenfp8-ladder3b-run,qwenfp8-vllm-run,e4b-arms}`;
FP8 peer captures `/opt/dlami/nvme/lava-tts/qwen38/fp32gate/vllm029-fp8-c{1,8,32}.json`. `*` = repeat
spread over 10 %. The strict report keys the open-loop plan on a tokenizer digest (it keyed on the
path before, which rendered the 12B run NOT MATCHED), and `llm_grid.sh` records provenance for a
single-flag server argument again.

Packet-ABI slice 1 (2026-10-25, plowrt `821273d0`, runtime contract 2): the Qwen and Gemma packets
re-emitted with the `object_requirements` / `recurrent_state` sections keep their pairing hashes and
pass the same gates on the slice binary: qwen3.8-27b BF16 (`47a070e371094922`, prefix on)
c1/c8/c32 kl_trim 0.000454/0.000463/0.000456, top1_decisive 1/1/1; qwen3.8-27b FP8
(`f5e93bb3224745bb`) c1/c8/c32 kl_trim 0.01004/0.01059/0.01041 vs vLLM 0.01466/0.01529/0.01791,
top1_decisive 0.9963 at every concurrency; gemma-4-12b (`86bed84b8474e56a`) c16 digits identical to
the row above; gemma-4-26b-a4b FP8 (`16b834960cced7ae`) c16 kl_mean 0.1969, kl_trim 0.1174,
top1_decisive 0.9622. The E4B and 26B BF16 packets (no section) still load; the previous 12B packet
is refused with "re-emit it with the current compiler". Raw evidence: `kvstore-scratch/abi-gates`.

Packet-ABI slice 2 (2026-10-26, plowrt `ae17393f`, serve.json version 2): all six production packets
re-emitted with the slice compiler keep their device bytes, pairing hashes and every section except
`serve.json` (version 2 now declares the tool syntax and reasoning markers; the E4B and 26B BF16
packets also gain the slice-1 `object_requirements` section their predecessors predated). The serve
goldens (15 conversations per packet: prompt bytes, token ids, tool/opener/reasoning decisions, stop
set) are byte-identical on every re-emitted packet; only the serve.json digest and version facts
changed. Gates on the slice binary: qwen3.8-27b BF16 (`b9c47d9830b79ac6`, prefix on) c1/c8/c32
kl_trim 0.000454/0.000473/0.000461, top1_decisive 1/1/1; qwen3.8-27b FP8 (`5e8ebd0f9771b06e`)
c1/c8/c32 kl_trim 0.01004/0.01073/0.01166 vs vLLM 0.01466/0.01529/0.01791,
top1_decisive 0.9963/0.9951/0.9939 (second run; the first missed cont_decisive on one c8 case at the
default 0.0 margin, a gate-policy decision left open); gemma-4-12b (`cec0e7d5599192a6`) c16 kl_trim 0.04725, top1_decisive 0.9862
(vLLM 0.06527 / 0.9823); gemma-4-26b-a4b FP8 (`caf6c4a47ba62a50`) c16 kl_trim 0.117, top1_decisive
0.9632; gemma-4-26b-a4b BF16 (`f72d0330f9d8fd2c`) c16 kl_trim 0.05077, top1_decisive 0.9818;
gemma-4-e4b (`046539c8a8a75efa`) logit parity top1 0.9948, kl_max 0.0154 (slice-1 digits). Raw
evidence: `kvstore-scratch/abi2-gates`, packet comparisons `kvstore-scratch/abi2-evidence.md`.

Strict serving reports on the slice-2 binary (2026-10-26, bin31 = `ae17393f`; the later commits
touch tests and recipes only): the plowrt arm of each production-pass strict grid rerun on the
re-emitted version-2 packet, reported with `campaign.py report` against the production-pass vLLM
0.29 arm (same grid, same box). gemma-4-26b-a4b FP8 (`caf6c4a47ba62a50`, host tier auto, prod mix,
two runs): MATCHED + EQUIVALENT; 1.8/s tok/s 1.00x / 1.00x, TTFT P99 1.54x* / 0.96x*, TPOT P99
1.27x* / 1.32x*; 2.7/s tok/s 1.00x / 1.00x, TTFT P99 2.75x* / 1.05x*, TPOT P99 2.13x* / 1.0x* (the
production-pass run on bin23 read 1.04x / 0.82x at 1.8/s). Both bin31 runs carry the same ~1.3 s
stall 250 s into the second 1.8/s cell (TTFT max 1292 ms, TPOT max 130 ms, then 35 decodes released
at once) that the bin23 run lacks; the server log is silent through it (admission-rung churn only,
no store, graph or Lt event). Runtime commits between the binaries on that path: the Qwen FP8
decode ladder (`b403a5c1`, GEMV MMA slots in the arena) and the reclaim-queue class check
(`be6929de`). With the host tier off (`PLOW_VMM_HOST_CACHE_MIB=0`, same packet, same binary, same
grid) the stall is gone and the rows beat the baseline: MATCHED + EQUIVALENT, 1.8/s tok/s 1.00x,
TTFT P99 0.66x, TPOT P99 0.55x; 2.7/s 1.00x / 0.71x* / 0.64x* (`strict-abi2-g26fp8-notier`). The
host-tier path on bin31 is therefore the open item: whether the recipe keeps tier auto (chosen for
the hit rate on real think-time traffic) and what changed in that path since bin23 are decisions
and work left open here. The Kit 2 packet cannot serve as the bin31 control (refused: W8A8 arms
without `object_requirements`). gemma-4-12b (`cec0e7d5599192a6`, prod mix, fp8 KV both arms,
plow provenance recorded against the served FP8 checkpoint as in the production pass): MATCHED +
EQUIVALENT and loses as before: 1.8/s tok/s 0.98x, TTFT P99 1.12x*, TPOT P99 1.71x; 2.7/s 0.95x /
2.88x* / 1.21x*. qwen3.8-27b FP8 (`5e8ebd0f9771b06e`, ISL 1000 / OSL 128): greedy cells reproduce
the production-pass digits exactly (c1 0.51x tok/s, TTFT P99 2.54x, TPOT P99 1.90x; c8 0.22x /
5.31x / 5.14x), NOT MATCHED on model version by construction as recorded above; the sampled cells read the same
(s1 0.51x / 2.82x / 1.89x, s8 0.22x / 5.35x / 5.09x, s32 0.23x / 5.27x / 4.72x), i.e. no change on
the slice binary for the known Qwen FP8 serving gap.
Not re-run: qwen3.8-27b BF16 (no vLLM arm; the prefix A/B above is plow vs plow), 26B BF16 and
E4B (no strict grid in the production pass either). Raw evidence:
`kvstore-scratch/strict-abi2-{g26fp8,g26fp8-r2,g26fp8-notier,g12,qwenfp8,qwenfp8-full}`.

Host-tier tail resolved (2026-10-26, bin34 = `2dabbe2b`: the 10 s growth hold `0091a6a0` +
`2dabbe2b`, merged with the typed-ownership store `86f2cd0c`). The "stall" was not an engine
pause: the engine ticked every 10-50 ms through it. It was queueing in a ~95%-busy engine whose
decode ticks stretched 20 -> 75 ms in the seconds where the store's free worker issued bursts of
125-134 `cuMemFree` (a context-wide driver call): the store grew its device chunks into every ring
the live pool retired and the pool shrank it back at the next prompt burst, 15.5k frees + 20k
allocs per 300 s cell with the tier on, half that with it off (per-second `/metrics` samples +
`PACKLOG`; `kvstore-scratch/fix_g26_run.sh`, `free_burst_corr.py`). Every clean commit since
`b779aa91` shows it; only the dirty production-pass binary did not. Tier-on prod grid vs the
production-pass vLLM 0.29 arm, all MATCHED + EQUIVALENT, tok/s 1.00x: growth hold alone 1.8/s
TTFT/TPOT P99 1.11x*/1.10x*, 2.7/s 0.98x*/0.98x* (10.2k frees); hold + `PLOW_KV_POOL_MIB=2048`
(the pool's reserve, which is also the band the store leaves free above the pool's pressure
floor): 1.8/s 0.99x*/0.82x*, 2.7/s 0.92x/0.76x* (2958 frees, `fix-g26-band2g`); 4096: 0.90x/0.86x
and 1.43x*/0.67x* (452 frees, one 961 ms cell); reserve 2048 on the pre-hold binary `ae17393f`:
0.89x*/0.81x*, 0.91x/0.80x* (3111 frees), so the reserve is the lever and the hold only matters
without it. The 26B FP8 recipe sets `PLOW_KV_POOL_MIB = "2048"` in `[serve.env]`; tier auto
stays. c16 FP32 gates on bin34: 26B FP8 PASS (kl_trim 0.1234, top1 0.9612), 12B PASS (0.0568 /
0.9843), 26B BF16 PASS, Qwen BF16 c1/c8/c32 PASS (kl_trim 0.000454 / 0.000463 / 0.000419), E4B logit parity PASS (top1
0.9948, kl_max 0.0154), Qwen FP8 c1/c8/c32 PASS on the second run (0.01004 / 0.01073 / 0.01028;
the first run's c8 was the known cont_decisive re-roll, 0.9565 < 0.9783, as on `ae17393f`), serve
goldens check mode 2 passed. The six recipes pin plow_git `2dabbe2b`.
gemma-4-12b on the same binary and reserve (`fix-g12-res2g`, prod mix, provenance re-recorded):
480 frees and no change, MATCHED + EQUIVALENT, 1.8/s tok/s 0.97x, TTFT P99 1.16x*, TPOT P99
1.82x; 2.7/s 0.94x / 3.52x* / 1.28x* (goodput 28% in the second 2.7/s cell): the 12B loss is not
the store's churn (it runs without live rings) and stays the open production item.

## Merge-prep checks on the other recipes (2026-10-25, branch `a3bf4162` vs main `9f16f187`; not strict rows)

Every H100 recipe with a packet on this box ran its own `[gates]` on the branch binary. The LLM
packets also ran the recipe `[bench]` on both binaries (random prompts, 32 per cell, 128 output
tokens, input 128/1024/4096 at c1/c8/c32, same packet and cuBLASLt 13.4 on both arms, no
contention; `campaign.py compare`, branch over main):

| Recipe (packet) | Gate on the branch | Bench, branch / main |
|---|---|---|
| gemma-4-26b-a4b FP8 (Kit 2 `b86ec26e4262`, store + host tier auto) | strict rows above; smoke at 10 s think, rate 2.7: hit 79.8 %, TTFT 66/687 ms, TPOT 14.4/82.9 ms, no errors | TTFT 0.70–0.99x, TPOT 0.96–1.00x, tok/s 1.00–1.12x |
| gemma-4-12b (`06da7686`, store + host tier auto) | FP32 c16 PASS: kl_trim 0.049 vs vLLM 0.065, top1_decisive 0.987 vs 0.982, over-0.1 111 vs 123, cont_frac 0.604 vs 0.574 | TTFT 0.36–0.99x, TPOT 0.98–1.00x, tok/s 1.00–1.07x |
| gemma-4-e4b (`da55906d`, store + host tier auto) | logit parity PASS: top1 0.9948, kl_mean 7.6e-4 (limits 0.98, 2e-3) | not benched (chat-kind bench) |
| qwen3.8-27b BF16 (`3fb7cefa`, prefix cache off per recipe) | not re-gated | 1.00x on every cell |
| qwen3.8-27b FP8 TMA (`6c31b3e8`, prefix cache off per recipe) | c1 and c8 captures equal the 2026-10-20 qualifying run digit for digit; the recipe's own thresholds fail that packet on main too (arch/26 B3) | c1 1.00x; c8/c32 incomplete on both arms (single-slot decode) |
| veena, orpheus, chatterbox, chatterbox-mtl (speech kit `6680acc8` bundles) | tts_cer PASS (Veena 0.011, Orpheus 0), s3gen_rel_l2 PASS | — |
| qwen3-asr 1.7B / 0.6B (re-emitted with the branch compiler; the kit's packets carry speech contract 0) | asr_wer PASS 3.913 % / 4.435 % (the kit baseline figures) | — |
| nemotron-3.5-asr | not run: the bundle is an ASR packet, the recipe gate expects a model packet | — |

The non-prefix path costs the same to the digit (Qwen rows); the prefix-on recipes gain TTFT at
c8/c32 because the boundary publish no longer copies on the engine thread. Raw evidence (gpu03
scratch): `/opt/dlami/nvme/lava-tts/kvstore-scratch/{gates,bench,asr}`,
`/home/lava/scratch/kvstore-gates/{e4b,g12,qwen-prod,r27-21-auto}`.

## Strict tables

## q1800.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 2000), turns geometric mean 5 max 20, first message lognormal(3600, 0.35), tool output lognormal(100, 0.8), context cap 8192 / output lognormal(46, 0.6) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 1.8 sessions/s, think lognormal(10 s, 0.8) <= 120 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1252723/1260642 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.7 GiB (1.04x) |
| Total throughput | 42,394 tok/s | 42,485 tok/s (1.00x) |
| Throughput / GPU | 42,394 tok/s/GPU | 42,485 tok/s/GPU (1.00x) |
| TTFT P99 | 369.5 ms | 251.0 ms (0.68x) |
| TPOT P99 | 32.60 ms | 17.96 ms (0.55x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 7.8% / 8.2%; Throughput / GPU 7.8% / 8.2%; TTFT P99 2.4% / 6.7%; TPOT P99 6.0% / 4.9%.

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 7.328 req/s / 7.345 req/s; SLO met 100.0% / 100.0%; requests 7.328 req/s / 7.345 req/s; mean in-flight 5.5 / 2.7; mean live sessions 81.5 / 79.0; TTFT P50 69.7 ms / 36.6 ms; TPOT P50 10.48 ms / 5.09 ms; cached prompt tokens - / 80.7%.

## q2700.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 2000), turns geometric mean 5 max 20, first message lognormal(3600, 0.35), tool output lognormal(100, 0.8), context cap 8192 / output lognormal(46, 0.6) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 2.7 sessions/s, think lognormal(10 s, 0.8) <= 120 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1370623/1378542 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.8 GiB (1.04x) |
| Total throughput | 56,172 tok/s * | 56,494 tok/s (1.01x) * |
| Throughput / GPU | 56,172 tok/s/GPU * | 56,494 tok/s/GPU (1.01x) * |
| TTFT P99 | 485.2 ms | 344.3 ms (0.71x) * |
| TPOT P99 | 57.89 ms * | 37.02 ms (0.64x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 13.4% / 13.4%; Throughput / GPU 13.4% / 13.4%; TTFT P99 3.9% / 11.0%; TPOT P99 18.4% / 30.7%.

**FLAGGED: spread > 10%: Baseline Total throughput 13.4%; Infervisor Total throughput 13.4%; Baseline Throughput / GPU 13.4%; Infervisor Throughput / GPU 13.4%; Infervisor TTFT P99 11.0%; Baseline TPOT P99 18.4%; Infervisor TPOT P99 30.7%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 9.422 req/s / 9.477 req/s; SLO met 100.0% / 100.0%; requests 9.428 req/s / 9.477 req/s; mean in-flight 11.0 / 5.9; mean live sessions 104.1 / 99.3; TTFT P50 119.3 ms / 53.3 ms; TPOT P50 16.19 ms / 8.37 ms; cached prompt tokens - / 78.6%.

## a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.8 GiB (1.04x) |
| Total throughput | 59,696 tok/s | 87,206 tok/s (1.46x) |
| Throughput / GPU | 59,696 tok/s/GPU | 87,206 tok/s/GPU (1.46x) |
| TTFT P99 | 2,147.3 ms * | 1,033.8 ms (0.48x) |
| TPOT P99 | 50.54 ms | 29.29 ms (0.58x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 1.4% / 0.4%; Throughput / GPU 1.4% / 0.4%; TTFT P99 34.2% / 7.5%; TPOT P99 2.0% / 1.1%.

**FLAGGED: spread > 10%: Baseline TTFT P99 34.2%.**

## a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | hf-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.29.0 (--gpu-memory-utilization 0.93 --max-num-seqs 256 --tokenizer /opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it --kv-cache-dtype fp8 --max-model-len 16384 --tensor-parallel-size 1) | Infervisor (plowrt f7eba863, packet b86ec26e4262) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 74.6 GiB | 77.8 GiB (1.04x) |
| Total throughput | 77,115 tok/s | 108,105 tok/s (1.40x) |
| Throughput / GPU | 77,115 tok/s/GPU | 108,105 tok/s/GPU (1.40x) |
| TTFT P99 | 2,274.9 ms | 1,918.1 ms (0.84x) |
| TPOT P99 | 80.67 ms | 48.02 ms (0.60x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.9% / 2.2%; Throughput / GPU 0.9% / 2.2%; TTFT P99 1.2% / 8.0%; TPOT P99 1.2% / 3.1%.

## 2026-10-27: native vs cuBLASLt route sweep (Gemma-4 26B-A4B FP8 and BF16)

Protocol: `step_bench` on one H100 (plowrt bin34 6a4b75bc for current packets and emit-time
variants without code changes; plowrt 69215b1e = commit 8b811e45 for the Qwen arms), decode 48 steps
`--spread`, prefill `--packed-prefill --pf-reps 3` (best timed rep), two passes ABAB, current arm =
the abi2 packet of the recipe at 21ab0d7d, alternative = the same recipe with one route knob changed
(emit knob: re-emitted with the same plowc; runtime knob: same packet). ms = mean of the two passes
(passes agree within 1%). % floor = calibrated floor / measured: decode bytes (weights streamed once,
MoE experts touched at top-8 of 128, KV rows) / 3.35 TB/s; prefill max(FLOP / peak, bytes / 3.35
TB/s), peak 1979 TF/s FP8, 990 BF16 (MoE floors approximate). digest = TOK_STREAM fnv of the run.
Raw: /opt/dlami/nvme/lava-tts/route-sweep (A/, B/, C/, logs/, gates/).

FP8 result: rung 32 moves to the native FP8 GEMV (recipe `PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS` 32 ->
64, commit a707a0b0): B32 -10..-14% at ctx 1K/8K/16K, every other rung and prefill unchanged. FP32
gates on that packet (62f32036d4cc): c16 PASS (kl_mean 0.2118 vs vLLM 0.202, kl_trim 0.1406 vs
0.1242, top1_decisive 0.9602, needle 1), long c4 PASS (kl_mean 0.1511 vs 0.165, kl_trim 0.0991 vs
0.1011, top1_decisive 0.9555, needle 1). Kept: cuBLASLt from 64 (native 64 is 2.2-2.4x slower),
cuBLASLt prefill and MoE prefill, fused qkv. The FP8 packet has no MoE decode library route (the
experts are native on every rung). Whole-packet effect: decode steps with 17-32 rows in flight
(the c16-c32 serving range) are 1.4-1.9 ms faster; nothing else moves.

BF16 result: no change. cuBLASLt at rung 8 loses to the native rung (+0.6..+1%) and native rung 16
loses to cuBLASLt (+0.2..+5%), the
native MoE decode is 3.1-4.8 ms slower per step at every rung (PLOW_MOE_DEC_LT=0 is refused at
runtime on this packet, so it was measured as an emit variant), and native prefill, native MoE
prefill, the interpreter attention instead of the cuBLASLt attention route and the interpreter
terminal are all slower or equal.


**26B FP8 rung 32 native (PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS 32 -> 64) -- PROMOTED** (g26fp8-base vs g26fp8-lt64)

| phase | rows x ctx | g26fp8-base ms | g26fp8-lt64 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B16 x 1024 | 8.58 | 8.58 | +0.1% | 71 / 71 | equal |
| decode | B32 x 1024 | 13.40 | 11.55 | -13.8% | 63 / 73 | differs |
| decode | B32 x 8192 | 14.91 | 13.50 | -9.4% | 59 / 65 | differs |
| decode | B32 x 16000 | 17.01 | 15.34 | -9.9% | 54 / 60 | differs |
| decode | B64 x 1024 | 18.82 | 18.85 | +0.2% | 54 / 54 | equal |

**26B FP8 cuBLASLt from 8 rows** (g26fp8-base vs g26fp8-lt8)

| phase | rows x ctx | g26fp8-base ms | g26fp8-lt8 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B4 x 1024 | 5.68 | 5.69 | +0.2% | 50 / 50 | equal |
| decode | B8 x 1024 | 6.63 | 9.17 | +38.3% | 63 / 46 | differs |
| decode | B8 x 16000 | 7.64 | 10.21 | +33.6% | 57 / 43 | differs |
| decode | B16 x 1024 | 8.63 | 10.54 | +22.2% | 71 / 58 | differs |
| decode | B16 x 8192 | 9.22 | 11.42 | +23.8% | 68 / 55 | differs |
| decode | B16 x 16000 | 10.25 | 12.50 | +21.9% | 63 / 52 | differs |
| decode | B32 x 1024 | 13.38 | 13.39 | +0.1% | 63 / 63 | equal |

**26B FP8 cuBLASLt only at 128 (rung 64 native)** (g26fp8-base vs g26fp8-lt128)

| phase | rows x ctx | g26fp8-base ms | g26fp8-lt128 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B32 x 1024 | 13.37 | 11.54 | -13.7% | 63 / 73 | differs |
| decode | B64 x 1024 | 18.82 | 46.00 | +144.4% | 54 / 22 | differs |
| decode | B64 x 8192 | 22.07 | 49.06 | +122.3% | 49 / 22 | differs |
| decode | B64 x 16000 | 26.02 | 53.24 | +104.6% | 45 / 22 | differs |
| decode | B128 x 1024 | 29.72 | 29.73 | +0.0% | 42 / 42 | equal |
| decode | B128 x 16000 | 44.17 | 44.18 | +0.0% | 35 / 35 | equal |

**26B FP8 native prefill projections** (g26fp8-base vs g26fp8-pfnative)

| phase | rows x ctx | g26fp8-base ms | g26fp8-pfnative ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 4.32 | 4.32 | -0.1% | 38 / 38 | differs |
| prefill | 1 x 128 | 14.21 | 15.14 | +6.6% | 58 / 54 | differs |
| prefill | 1 x 256 | 15.68 | 16.76 | +6.9% | 52 / 49 | equal |
| prefill | 1 x 512 | 19.26 | 20.52 | +6.6% | 43 / 40 | equal |
| prefill | 1 x 1024 | 25.45 | 28.46 | +11.8% | 32 / 29 | equal |
| prefill | 1 x 1088 | 30.38 | 32.60 | +7.3% | 27 / 25 | equal |
| prefill | 1 x 2048 | 40.98 | 45.97 | +12.2% | 27 / 24 | equal |
| prefill | 1 x 2112 | 45.32 | 49.78 | +9.8% | 25 / 23 | equal |
| prefill | 1 x 4096 | 76.20 | 85.49 | +12.2% | 29 / 26 | equal |
| prefill | 2 x 4096 | 141.39 | 159.45 | +12.8% | 31 / 28 | differs |

**26B FP8 MoE prefill experts native: PLOW_MOE_PF_LT=0** (g26fp8-base vs g26fp8-moepf0)

| phase | rows x ctx | g26fp8-base ms | g26fp8-moepf0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| prefill | 1 x 128 | 14.42 | 19.31 | +33.9% | 57 / 43 | differs |
| prefill | 1 x 1024 | 25.51 | 40.23 | +57.7% | 32 / 20 | equal |
| prefill | 1 x 4096 | 76.27 | 112.30 | +47.2% | 29 / 20 | equal |

**26B FP8 PLOW_DECODE_LT_QKV=0** (g26fp8-base vs g26fp8-ltqkv0)

| phase | rows x ctx | g26fp8-base ms | g26fp8-ltqkv0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B32 x 1024 | 13.37 | 13.38 | +0.0% | 63 / 63 | equal |
| decode | B32 x 16000 | 16.99 | 17.00 | +0.0% | 54 / 54 | equal |
| decode | B64 x 1024 | 18.85 | 18.84 | -0.1% | 54 / 54 | equal |
| decode | B64 x 16000 | 26.17 | 26.12 | -0.2% | 45 / 45 | equal |
| decode | B128 x 1024 | 29.86 | 29.79 | -0.2% | 42 / 42 | equal |
| decode | B128 x 16000 | 44.24 | 44.27 | +0.1% | 35 / 35 | equal |

**26B BF16 cuBLASLt from 8 rows** (g26bf16-base vs g26bf16-lt8)

| phase | rows x ctx | g26bf16-base ms | g26bf16-lt8 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B4 x 1024 | 7.68 | 7.61 | -0.9% | 67 / 67 | equal |
| decode | B8 x 1024 | 8.86 | 8.92 | +0.7% | 88 / 87 | differs |
| decode | B8 x 8192 | 9.35 | 9.38 | +0.3% | 85 / 85 | differs |
| decode | B8 x 16000 | 10.51 | 10.62 | +1.1% | 78 / 77 | differs |
| decode | B16 x 1024 | 10.67 | 10.69 | +0.1% | 109 / 108 | equal |

**26B BF16 cuBLASLt from 32 rows (rung 16 native)** (g26bf16-base vs g26bf16-lt32)

| phase | rows x ctx | g26bf16-base ms | g26bf16-lt32 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B8 x 1024 | 8.86 | 8.91 | +0.6% | 88 / 87 | equal |
| decode | B16 x 1024 | 10.70 | 11.26 | +5.2% | 108 / 103 | differs |
| decode | B16 x 8192 | 11.93 | 12.12 | +1.6% | 100 / 99 | equal |
| decode | B16 x 16000 | 13.63 | 13.89 | +1.9% | 90 / 89 | differs |
| decode | B32 x 1024 | 13.73 | 13.71 | -0.2% | 115 / 115 | equal |

**26B BF16 MoE decode experts native (emit PLOW_EMIT_MOE_DEC_LT=0)** (g26bf16-base vs g26bf16-moedecnat)

| phase | rows x ctx | g26bf16-base ms | g26bf16-moedecnat ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B4 x 1024 | 7.58 | 10.98 | +44.8% | 68 / 47 | differs |
| decode | B4 x 16000 | 8.55 | 11.74 | +37.3% | 62 / 45 | equal |
| decode | B8 x 1024 | 8.85 | 11.96 | +35.2% | 88 / 65 | differs |
| decode | B8 x 16000 | 10.51 | 13.63 | +29.7% | 78 / 60 | differs |
| decode | B16 x 1024 | 10.64 | 13.83 | +30.0% | 109 / 84 | differs |
| decode | B16 x 16000 | 13.55 | 17.09 | +26.1% | 91 / 72 | differs |
| decode | B32 x 1024 | 13.63 | 17.26 | +26.7% | 116 / 91 | differs |
| decode | B32 x 16000 | 18.76 | 22.74 | +21.2% | 92 / 76 | differs |

**26B BF16 native prefill projections** (g26bf16-base vs g26bf16-pfnative)

| phase | rows x ctx | g26bf16-base ms | g26bf16-pfnative ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 5.56 | 5.58 | +0.2% | 48 / 48 | equal |
| prefill | 1 x 128 | 14.72 | 17.73 | +20.4% | 105 / 87 | equal |
| prefill | 1 x 256 | 16.04 | 20.25 | +26.2% | 96 / 76 | equal |
| prefill | 1 x 512 | 21.67 | 24.78 | +14.4% | 71 / 62 | equal |
| prefill | 1 x 1024 | 27.88 | 30.93 | +10.9% | 55 / 50 | equal |
| prefill | 1 x 1088 | 31.45 | 34.55 | +9.9% | 49 / 45 | equal |
| prefill | 1 x 2048 | 42.14 | 46.18 | +9.6% | 43 / 39 | equal |
| prefill | 1 x 2112 | 49.22 | 49.12 | -0.2% | 38 / 38 | equal |
| prefill | 1 x 4096 | 78.28 | 82.01 | +4.8% | 46 / 44 | equal |
| prefill | 2 x 4096 | 140.74 | 150.29 | +6.8% | 51 / 48 | equal |

**26B BF16 MoE prefill experts native: PLOW_MOE_PF_LT=0** (g26bf16-base vs g26bf16-moepf0)

| phase | rows x ctx | g26bf16-base ms | g26bf16-moepf0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| prefill | 1 x 128 | 14.49 | 16.43 | +13.3% | 106 / 94 | equal |
| prefill | 1 x 1024 | 28.20 | 36.89 | +30.8% | 55 / 42 | equal |
| prefill | 1 x 4096 | 77.71 | 109.09 | +40.4% | 47 / 33 | equal |

**26B BF16 prefill attention off the cuBLASLt route: PLOW_PF_ATTN_GEMM=0** (g26bf16-base vs g26bf16-pfattngemm0)

| phase | rows x ctx | g26bf16-base ms | g26bf16-pfattngemm0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| prefill | 1 x 128 | 14.49 | 15.02 | +3.6% | 106 / 102 | equal |
| prefill | 1 x 1024 | 28.20 | 28.23 | +0.1% | 55 / 55 | equal |
| prefill | 1 x 4096 | 77.71 | 80.12 | +3.1% | 47 / 45 | equal |

**26B BF16 PLOW_TERMINAL_LT=0** (g26bf16-base vs g26bf16-termlt0)

| phase | rows x ctx | g26bf16-base ms | g26bf16-termlt0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| prefill | 1 x 128 | 14.49 | 15.67 | +8.1% | 106 / 98 | equal |
| prefill | 1 x 1024 | 28.20 | 28.55 | +1.3% | 55 / 54 | equal |
| prefill | 1 x 4096 | 77.71 | 78.47 | +1.0% | 47 / 46 | equal |

**26B BF16 PLOW_DECODE_LT_QKV=0** (g26bf16-base vs g26bf16-ltqkv0)

| phase | rows x ctx | g26bf16-base ms | g26bf16-ltqkv0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B16 x 1024 | 10.70 | 10.65 | -0.5% | 108 / 109 | equal |
| decode | B16 x 16000 | 13.61 | 13.58 | -0.3% | 90 / 91 | equal |
| decode | B32 x 1024 | 13.68 | 13.72 | +0.2% | 115 / 115 | equal |
| decode | B32 x 16000 | 18.79 | 18.75 | -0.2% | 92 / 92 | equal |
