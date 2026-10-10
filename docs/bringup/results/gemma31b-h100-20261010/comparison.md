# Gemma-4-31B-it comparison (1x H100 SXM5, vLLM 0.28)

Qualified wins only: `campaign.py report` MATCHED + EQUIVALENT, Infervisor total throughput above vLLM.
Every rendered row, won or lost, is in [comparison.csv](comparison.csv) (`campaign_source`
`g31-20261010-<workload>-<precision>`).

- Infervisor FP8: plowrt `8818de12` (gemma31b), packet `a18768d2ad68` (max_ctx 131072) from
  [`sm90a-h100-tp1-fp8.toml`](../../../../recipes/infervisor/gemma-4-31b/sm90a-h100-tp1-fp8.toml)
  at 131072; the recipe ships 16384 (packet `3be48e1f5bf8`, the same build at max_ctx 16384, see
  "max_ctx"). Serve defaults from the packet; server env `PLOW_LIBCUDA` and the cuBLAS 13.4 library path.
- Infervisor BF16: plowrt `071d8f13`, packet `87176bdcb0f1` from
  [`sm90a-h100-tp1-bf16.toml`](../../../../recipes/infervisor/gemma-4-31b/sm90a-h100-tp1-bf16.toml).
- Baseline: vLLM 0.28.0, same client (`vllm bench serve` / `agentic_turns.py` from `/opt/pytorch`),
  prefix caching on, max-num-batched-tokens 8192, max-num-seqs 256, 2 repeats per cell, one server
  per arm. FP8: same RedHatAI checkpoint, `fp8_per_token_head` KV (TRITON_ATTN), `--max-model-len 65536`
  (the largest power of two it fits at 0.9: vLLM 0.28 sizes the sliding layers at full length for its
  startup check). BF16: google checkpoint, BF16 KV, `--max-model-len 16384` at
  `--gpu-memory-utilization 0.97` (at 0.9 it cannot hold one 16384-token request: 13.8 vs 10.3 GiB).
  Reproduce: `scripts/campaign/repro_gemma31b_h100.sh` (workloads lat st4k st15k agentic lc32k).

## FP8 (W8A8, FP8 per-token-head KV)

Quality: FP32-reference gate (46 cases to 15.9K tokens) PASS twice on packet `a18768d2ad68`:
kl_mean 0.0481 / 0.0643 vs vLLM 0.0633 (limit 0.0811), kl_p99 0.94 / 1.23 vs 1.42, top1_decisive 0.9890 / 0.9834 vs 0.9853, cont_frac 0.635 / 0.636 vs 0.646, needles 1.0. Needles 36/36 at 8192,
32768, 65536 and 130000 tokens.

`campaign.py report`, all 12 cells MATCHED + EQUIVALENT, 2 repeats (packet `a18768d2ad68`):

| Cell | Total tok/s vLLM | Infervisor | Ratio | TTFT p99 | TPOT p99 | Peak GiB vLLM / Inf |
|---|---:|---:|---:|---:|---:|---:|
| 1024/128 c1 | 531 | 445 | 0.84x | 1.37x* | 1.22x | 72.3 / 66.1 |
| 1024/128 c4 | 1,658 | 1,530 | 0.92x | 0.99x* | 1.09x | 72.3 / 66.1 |
| 4096/128 c32 | 5,386 | 8,461 | **1.57x** | 0.56x | 0.63x | 73.2 / 77.8 |
| 4096/128 c128 | 5,501 | 8,518 | **1.55x** | 0.63x | 0.42x | 73.2 / 77.9 |
| 15000/128 c32 | 3,674 | 8,292 | **2.26x** | 0.44x | 0.54x | 73.5 / 78.2 |
| 15000/128 c128 | 3,689 | 8,404 | **2.28x** | 0.44x | 0.55x | 73.5 / 78.5 |
| agentic c32 sessions | 4,663 | 10,959 | **2.35x** | 0.38x* | 0.52x | 73.3 / 78.9 |
| agentic c64 sessions | 4,607 | 9,464 | **2.05x** | 0.40x | 0.52x | 73.3 / 79.0 |
| agentic c128 sessions | 4,625 | 9,150 | **1.98x** | 0.44x | 0.53x | 73.3 / 79.1 |
| 32768/128 c1 | 1,973 | 4,338 | 2.20x | 0.38x | 0.75x | 72.9 / 78.4 |
| 32768/128 c4 | 2,362 | 5,789 | 2.45x | 0.37x | 0.42x | 72.9 / 78.8 |
| 32768/128 c16 | 2,381 | 5,924 | 2.49x | 0.39x | 0.40x | 72.9 / 79.2 |

Qualified wins (bold): 4K, 15K and agentic. The 32768 cells beat vLLM but sit past the shipped max_ctx
(see "max_ctx"); they are recorded, not claimed.

`*` = repeat spread > 10% (FLAGGED; direction only). Lost (in the CSV): 1024/128 c1 0.84x and c4
0.92x total throughput. Decode is the cause: B=1 step 19.7 ms vs vLLM TPOT 16.1 ms ("Rung gates").

## BF16 (BF16 weights and KV)

Not qualified. The FP32-reference gate fails on kl_mean twice (0.0215 on `fe8fcdd9bee9`, 0.0252 on
`87176bdcb0f1`, vs vLLM 0.0084, limit 0.0124); kl_p99 0.079 / 0.083 (limit 0.091), top1_decisive
0.9963 / 0.9972 (limit 0.9891) and cont_frac 0.843 / 0.885 (limit 0.805) pass. The excess is a handful
of positions inside degenerate repetition loops of the natural-text cases (" own own own ..." then the
FP32 model breaks out; nat-beagle-128 pos 29, nat-pride-15872 pos 15). transformers BF16 (sdpa) scored
on the same reference lands where plow does: kl_mean 0.0219 vs plow 0.0215, kl_p99 0.127 vs 0.079,
top1_decisive 0.9981 vs 0.9963, and flips nat-pride-15872 pos 15 to the same token. vLLM 0.28 BF16 is
unusually close to FP32 on this model; plow BF16 is at transformers BF16 parity, not at vLLM's.
Invariant across `PLOW_PREFIX_CACHE`, the light decode paths, `PLOW_PREFILL_LIGHT`, `PLOW_TB_SPLIT_ATTN`,
`PLOW_FUSE_ARGMAX`, decode/prefill cuBLASLt routes (all bit-identical at concurrency 1).

`campaign.py report` (NOT EQUIVALENT because of the gate; listed for direction, none is a qualified win):

| Cell | Status | Total tok/s vLLM | Infervisor | Ratio | TTFT p99 | TPOT p99 | Peak GiB vLLM / Inf |
|---|---|---:|---:|---:|---:|---:|---:|
| 1024/128 c1 | MATCHED, NOT EQUIVALENT | 364 | 353 | 0.97x | 1.03x* | 1.03x | 77.6 / 74.8 |
| 1024/128 c4 | MATCHED, NOT EQUIVALENT | 1,280 | 1,273 | 0.99x | 1.31x | 1.00x | 77.6 / 74.8 |
| 4096/128 c32 | MATCHED, NOT EQUIVALENT | 5,904 | 4,785 | 0.81x | 1.22x | 0.82x | 78.9 / 78.7 |
| 4096/128 c128 | MATCHED, NOT EQUIVALENT | 5,930 | 4,785 | 0.81x | 1.24x | 0.83x | 78.9 / 78.7 |
| 15000/128 c32 | MATCHED, NOT EQUIVALENT | 4,629 | 4,893 | 1.06x | 0.95x | 1.51x | 78.9 / 79.2 |
| 15000/128 c128 | MATCHED, NOT EQUIVALENT | 4,627 | 4,862 | 1.05x | 0.95x | 1.52x | 78.9 / 79.2 |

## Decisions and rung gates

Memory (one 80 GiB H100). A staged slot keeps a 2048-row sliding ring per slot (window 1024 + one
1024-row stage, rounded to a power of two): 50 sliding layers x 16 KV heads x 256 x K+V.
- FP8: weights 29.9 GiB, ring 800 MiB/slot; 32 slots = 25 GiB of rings, full-attention KV budget 16.2 GiB
  (40 KiB/token, ~420K rows). 64 slots would leave no full-attention KV.
- BF16: weights 57.2 GiB, ring 1.6 GiB/slot; 8 slots = 12.5 GiB, full-attention KV budget 5.0 GiB
  (80 KiB/token, ~65K rows). The cuBLASLt prefill attention route does not fit beside it. vLLM keeps only
  the window for sliding layers in steady state and holds ~13 4K requests against plow's 8 slots, which
  is why BF16 loses the 4K cells. FP8 KV for BF16 (16 slots) is not gated.
- The default queue bound (four batches) refused 351 of 384 c128 requests on the 8-slot BF16 packet;
  `PLOW_MAX_QUEUED_REQUESTS` (new, packet serve default 512) fixes it.

max_ctx (FP8). The long FP32 gate (20 cases to 32768 tokens, `reference --device offload`) fails twice
on cont_frac only (0.511 / 0.548 vs vLLM 0.608, limit 0.558) while plow is closer to FP32 than vLLM on
every distribution metric (kl_mean 0.069 / 0.075 vs 0.109, kl_p99 1.31 / 1.56 vs 2.38, top1_decisive
0.982 / 0.976 vs 0.971); the lost cases diverge at near ties (margins 0.02-1.3). The recipe therefore
ships 16384 (std gate on `3be48e1f5bf8`: PASS 1 of 2, kl_mean 0.0494 PASS, 0.0812 FAIL by 0.0002 against limit 0.0811, kl_p99 0.81 / 1.57 vs vLLM 1.42; four std runs across both packets span kl_mean 0.048-0.081); the 32768 cells above and the 130000-token
needles were measured on the 131072 packet and are not a qualified configuration.

Prefill projection routes (route matrix, `scripts/bench/gemma4_route_matrix.sh`, ws384 vs cuBLASLt,
6 rounds, cold weights, correctness checked): cuBLASLt wins every 31B shape and rung in BF16 (1.0-2.3x)
and in FP8 except down_proj (5376 x 21504) at 512 and >= 2048 rows, where the native FP8 body is 7-15%
faster (0.122 vs 0.136 ms at 512, 1.336 vs 1.573 ms at 8192): `CUBLASLT_PREFILL_GEMMA4_31B_SHAPES`.
step_bench packed prefill (FP8 / BF16): 1K 85 / 106 ms, 4K 330 / 425 ms, 8K 733 / 929 ms (two 4096
launches), 15K 1.67 / 2.03 s.

Decode routes (step_bench, ctx 1024 unless noted, 32 steps, sd < 0.05 ms):

| B | FP8 native | FP8 cuBLASLt from 16 | BF16 native | BF16 cuBLASLt from 4 |
|---:|---:|---:|---:|---:|
| 1 | 19.69 | 19.67 | 24.90 | 24.87 |
| 4 | 20.54 | 20.51 | 26.11 | 25.49 |
| 8 | 22.35 | 22.32 | 27.42 | 26.56 |
| 16 | 27.18 | 34.61 | | |
| 32 | 36.20 | 40.03 | | |
| 16, ctx 4096 | 30.24 | 37.58 | | |
| 32, ctx 4096 | 42.28 | 46.18 | | |
| 8, ctx 4096 | | | 28.55 | 27.69 |

FP8 ships native decode at every rung, BF16 cuBLASLt from 4 rows. `PLOW_GEMV_SPLIT=2`: identical
(19.72 / 36.19 ms at B1 / B32, same digests), not adopted.

FP8 decode roofline (B=1, `op_roof.py` on the step_bench instruction sweep): floor 9.74 ms, measured
19.63 ms = 50%. GemvGluFp8 6.43 ms (64% of its HBM floor), down 4.08 ms (51%), q/k/v 2.77 ms (58%),
o 1.81 ms (51%), FlashDecodeFp8 1.92 ms (60 layers at ~32 us, launch-bound at ctx 1024), lm_head
0.96 ms (88%). The interpreter FP8 GEMV at 5376/21504 shapes reaches ~1.7-2.2 TB/s; this is the open
gap behind the c1/c4 losses.

## Other checks

- Tool calling (OpenAI SDK loop, `toolcalls/tool_loop.py`, FP8 packet): every check passes (single,
  streamed, parallel calls, `tool_choice` none / refused forms). After a tool result the 31B answer
  starts with the text `thought\n`: the model opens `<|channel>thought ...<channel|>` itself (the
  template does not re-prime it after `<tool_response|>`), and plowrt has no Gemma-4 channel reasoning
  parser, so the channel name leaks into content. E4B does not emit the channel there.
- Chat-template token parity: `crates/plowrt/tests/fixtures/toolcall/gemma4-31b` (7 conversations,
  transformers `render_jinja_template` text + token ids) passes `serve::tools::parity_tests`. The 31B
  template is byte-identical to 12B/26B; RedHat's copy differs only by a trailing newline.
- Not run: the open-loop production mix; BF16 agentic (vLLM BF16 holds ~1 16K session; one repeat
  would take ~3 h).

## Evidence (scratch, outside the repo)

`/opt/dlami/nvme/lava-tts/gemma31b/`: `fin/` (FP8 packet, grids `res/`, reports `report/`, gates
`gate/`, FP32 references `fp32ref/`), `fin-g2/` (second FP8 gate), `fin16/` (16K packet and gate),
`fin-bf16/` (BF16 packet, grids, gate), `lcref/` (long reference), `route/` (route matrix),
`sb1/` + `grid/` (step_bench, op_roof sweeps), `natab*/` + `hfbf16/` (BF16 numerics study),
`tools-fp8/` (tool loop).
