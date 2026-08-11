# Kimi-K3 on 8x MI325X

This is the reproducible recipe for branch `kimi-k3-mi325x` at commit
`468674e985adf62c32b2178fbde29f0c5325c02e`. It serves the full 93-layer
Kimi-K3 checkpoint through `plowrt serve` on eight MI325X GPUs.

The measured result is **131.162 aggregate generated tok/s** at concurrency 32,
not 131 tok/s for one request. The client is vLLM 0.27 `bench serve`; the server
is Plow. This is not a vLLM-engine comparison.

## Branch audit

Relative to `origin/main`, the frozen performance commit changes 97 files
(9,408 insertions, 866 deletions) across the K3 graph/rewrite/Lean model,
devgen, AMD runtime, gfx942 kernels, build scripts, tuning data, hardware
oracles, serving scheduler, and performance records.

- Every opcode reachable from the final prefill and decode programs has an
  implementation in the selected gfx942 objects. There is no known live
  default/NOP arm.
- Loader interlocks cover K3, FP8 KV, grouped A4W4, L2 dispatch, GEMV capacity,
  GEMV walk, batched width, and recurrent sequence-row addressing.
- The final ladder fixes slot-stable recurrent state flow and exact TP counter
  auditing across rung changes. The post-soak reuse gate is byte-identical.
- The branch is production-capable for the measured TP8 path, but its accuracy
  and comparison evidence is incomplete: final-ladder GSM8K was not rerun and
  the upstream vLLM engine fails before model load on this host.
- CUDA can read ladder metadata but still executes its widest decode program;
  the dynamic-rung performance path is AMD-only in this branch.

## Frozen configuration

| axis | setting |
|---|---|
| hardware | 8x MI325X, gfx942, 304 CUs/GPU, TP8 |
| toolchain | Nix TheRock ROCm 7.14.0, HIP 7.14.60850, clang 23 |
| checkpoint | native compressed-tensors MXFP4, `/home/lava/models/Kimi-K3` |
| model | 93 layers: 69 KDA + 24 MLA; 896 experts, top-16 |
| KV | FP8 MLA KV, max context 32768 |
| prefill rungs | 128, 512, 1024, 2048, 4096, 8192 |
| decode rungs | 1, 2, 4, 8, 16, 32 |
| decode GEMV | 128-workgroup cap; MM16 object with row walk |
| MoE on gfx942 | packed MXFP4 weights, software decode to BF16, BF16 MFMA; fused A4 bridge |
| scheduling | L2/XCD placement, global queue, two-level gate hierarchy |
| TP safety | compact exact cross-rank counter audit every step |
| local counters | double-buffered; inactive bank is cleared behind the running token |
| prefix cache | off for the reported result |
| speculative decode | not implemented or enabled |
| benchmark | vLLM 0.27.0 client, random input 32, output 2048, C32/N32, one warmup |

The exact measured artifact hashes are:

```text
plowrt                                             1ebb5ee5d7ee8a11cfc352c8cf28d684d457ac0b5a6a7e90530822cd76f45ad2
model.pkt                                          f1f260d69105dffab3a7bd7f256d5fcbc215609f44c033c2cbb025949d14c709
interp_decode_fp8kv_k3.elf                         4c0d2ef95a2bef839965c977d53c873b74c1a1c50c92e0655e132e1bcfa16393
interp_decode_fp8kv_k3_gq.elf                      cc1c82c5d109150c6f29ae8c47cfc9b7fdcec6e4ead023a779dc60aaddeb999c
```

Hashes are reproduction gates only at the frozen commit, checkpoint, and Nix
lock. Do not copy these objects to a different source revision and call it the
same build.

## 1. Enter the pinned source and toolchain

All build, test, and benchmark commands must run through `nix develop`. Do not
use `/usr/bin/hipcc`, `/opt/rocm`, or a system compiler.

```bash
cd /home/lava/plow
nix develop --command git switch kimi-k3-mi325x
nix develop --command git checkout 468674e985adf62c32b2178fbde29f0c5325c02e
nix develop --command bash -lc '
  test "$PLOW_TOOLCHAIN_LABEL" = rocm-7.14.0-nix
  case "$(readlink -f "$PLOW_HIPCC")" in /nix/store/*) ;; *) exit 1;; esac
  "$PLOW_HIPCC" --version | grep "HIP version: 7.14."
'
```

The flake downloads AMD's stable relocatable SDK from:

```text
https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx94X-dcgpu-7.14.0.tar.gz
sha256-MuFtyn+EQKCKjWNqan2wA0xhUY8y6pFTR7mNn1UZmww=
```

## 2. Prepare the checkpoint

The checkpoint directory must contain all 96 safetensors shards. K3 also needs
a fast tokenizer and five derived tensors for each MLA layer. The runtime reads
one flat shard directory, so the preparation script creates a symlink farm in
which the derived sidecar sorts last.

```bash
nix develop .#quantize --command python3 scripts/kimi_k3_tokenizer.py \
  --model /home/lava/models/Kimi-K3 \
  --out /home/lava/models/k3_tokz --verify

nix develop .#quantize --command python3 scripts/kimi_k3_prep.py \
  --model /home/lava/models/Kimi-K3 \
  --out /home/lava/models/k3_derived \
  --derived --farm /home/lava/models/k3_farm

nix develop --command bash -lc '
  test -f /home/lava/models/k3_tokz/tokenizer.json
  test -f /home/lava/models/k3_farm/model-idx-derived-00001.safetensors
  test "$(find /home/lava/models/k3_farm -maxdepth 1 -name "*.safetensors" | wc -l)" -ge 97
  test -e /home/lava/models/k3_farm/tokenizer.json ||
    ln -s /home/lava/models/k3_tokz/tokenizer.json /home/lava/models/k3_farm/tokenizer.json
'
```

The sidecar is about 4.5 GB. Expert weights are not requantized or copied; the
farm points at the checkpoint-native packed MXFP4 and E8M0 tensors.

## 3. Build Plow

```bash
nix develop --command cargo build --release -p plowc
nix develop --command cargo build --release -p plowrt --features hsa
```

For the frozen commit, `target/release/plowrt` should match the hash above.

## 4. Build the complete gfx942 object inventory

Use the repository script, not hand-written `hipcc` commands. It builds both
static and global-queue interpreters, checks the 64 KiB LDS/256-VGPR cliffs,
checks required capability symbols, and runs the generic gfx942 and K3 grouped
A4W4 ISA audits.

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=32 \
  PLOW_GEMV_MM=16 \
  PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  JOBS=8 \
  scripts/build_gfx942.sh \
  /home/lava/plow/build-amd/k3-mi325x-b32-mm16-walk
```

Important build policy:

- `PLOW_K3_DECODE_MXFP4_PROJ=0` removes standalone projection bodies absent
  from the K3 batched-decode packet. It does not remove grouped MXFP4 experts.
- `PLOW_GEMV_WALK=1` is mandatory above batch 16. The ELF advertises
  `plow_gemv_mm_cap_16` and `plow_gemv_walk_1`; the loader refuses a mismatch.
- `PLOW_L2HIER=1`, `PLOW_GEMV_LG=1`, `PLOW_MOE_DEC_LG=1`,
  `PLOW_K3_A4W4_EPI=1`, and cached grouped-expert weights are script defaults.
- `PLOW_KDA_PF_STATE_RESIDENT=0`, `PLOW_OCC4=0`,
  `PLOW_DEC_SQUEEZE=0`, and grouped-weight NT loads remain off. Their tested
  alternatives did not win or were unsafe.
- Do not set `PLOW_ROWS_ONLY` for this canonical build. A partial directory is
  insufficient for serving.

Verify the two critical objects:

```bash
nix develop --command bash -lc '
  source scripts/nix_rocm_714.sh
  plow_init_rocm_714
  for f in interp_decode_fp8kv_k3.elf interp_decode_fp8kv_k3_gq.elf; do
    o=/home/lava/plow/build-amd/k3-mi325x-b32-mm16-walk/$f
    "$PLOW_K3_READELF" -sW "$o" |
      grep -E "plow_(k3_arms_1|fp8_kv_1|moe_pf_a4w4_arm|gemv_mm_cap_16|gemv_walk_1|l2_place_dispatch_1)"
  done
'
```

On gfx942, the active grouped kernels must contain
`v_mfma_f32_32x32x8_bf16` and software FP4 decode. Native CDNA4 scaled-MX
instructions are forbidden by the audit.

## 5. Emit the TP8 ladder asset

The packet and object settings are a pair. `PLOW_L2_PLACE=1` is the compiler
half of the runtime's XCD-local dispatch. `PLOW_GLM_GEMV_WG=128` is historical
naming; for K3 it caps ordinary GEMV-family packet ownership while leaving the
sharded vocabulary head at 304 CUs.

```bash
nix develop --command env \
  K3_FULL=1 \
  PLOW_FP8_KV=1 \
  PLOW_MXFP4=1 \
  PLOW_L2_PLACE=1 \
  PLOW_DECODE_BATCH=32 \
  PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 \
  PLOW_GEMV_MM=16 \
  PLOW_GEMV_WALK=1 \
  PLOW_GLM_GEMV_WG=128 \
  ./target/release/plowc \
  --hf-dir /home/lava/models/k3_farm \
  --emit devblob \
  --arch gfx942 --gpu MI325X \
  --num-gpus 8 --parallel tp \
  --max-ctx 32768 --n-cu 304 \
  --out /home/lava/models/k3_mi325x_ladder_router

nix develop --command ln -s \
  /home/lava/plow/build-amd/k3-mi325x-b32-mm16-walk \
  /home/lava/models/k3_mi325x_ladder_router/hsaco
```

Use a new output directory. The final assets must contain `model.pkt`,
`weights.json`, `checkpoint`, `tokenizer.json`, and `hsaco`. The PLOWDEV packet
embeds its program ladder; an empty `weights.json.buckets` is expected.

Structural checks:

```bash
nix develop --command sha256sum \
  target/release/plowrt \
  /home/lava/models/k3_mi325x_ladder_router/model.pkt \
  /home/lava/models/k3_mi325x_ladder_router/hsaco/interp_decode_fp8kv_k3.elf \
  /home/lava/models/k3_mi325x_ladder_router/hsaco/interp_decode_fp8kv_k3_gq.elf

nix develop --command ./target/release/plowrt disasm \
  /home/lava/models/k3_mi325x_ladder_router/model.pkt
```

Expect six prefill programs, decode rungs 1/2/4/8/16/32, `T=32` on the widest
rung, ordinary GEMV packets at `b=128`, RouterTopK at `b=32`, and the vocabulary
head at `b=304`. Every decode rung must use sequence-row KDA addressing.

## 6. Correctness gates

Before serving, run the TP logit-equivalence gate. Depth 2 is mandatory because
depth 1 has no latent MoE layer.

```bash
nix develop --command env \
  PLOW_K3_CKPT=/home/lava/models/k3_farm \
  PLOW_K3_HSACO=/home/lava/plow/build-amd/k3-mi325x-b32-mm16-walk \
  PLOW_K3_LAYERS=1,2 \
  scripts/k3_tp_equivalence.sh
```

The adopted single-rung B1 asset scored GSM8K 197/200. That accuracy result was
not repeated on the final ladder packet, so do not present it as a final-ladder
accuracy measurement. A production release should repeat GSM8K and the 16K/32K
context gates.

## 7. Serve

The reported run held one exclusive lease for the server lifetime:

```bash
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_TP_AUDIT_COMPACT=1 \
  PLOW_CTR_DBUF=1 \
  PLOW_DSTEP_LOG=1 \
  PLOW_DSTEP_EVERY=64 \
  perf-data/harness/gpulease -n 8 k3-ladder-slo-serve \
  ./target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x_ladder_router \
  --port 8018
```

`PLOW_DSTEP_*` is diagnostic and may be removed after reproduction. Keep the
placement, compact audit, and counter-double-buffer flags explicit in measured
runs. The runtime chooses the narrowest rung covering the highest occupied
slot; slots are never compacted or moved. The admission controller widens on
backlog/SLO pressure and narrows with hysteresis.

The measured run used the default chunked/ragged prefill, no prefix cache, no
prefill batching, no speculative decoding, shared checkpoint mappings, and the
global-queue object selected by packet capability.

## 8. Measure served throughput

Use the flake's client. It is vLLM 0.27.0 plus Nix `jq` and `curl`; it does not
provide a ROCm vLLM server.

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8018 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random \
  --random-input-len 32 --random-output-len 2048 \
  --random-range-ratio 0 \
  --request-rate inf --max-concurrency 32 \
  --num-prompts 32 --num-warmups 1 \
  --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir /tmp/k3-ladder-slo-c32-out2048 \
  --result-filename seed0.json
```

Hard-gate the saved JSON; HTTP 200 alone is insufficient:

```bash
nix develop .#vllm --command jq -e '
  .completed == 32 and .failed == 0 and
  .total_output_tokens == 65536 and
  all(.output_lens[]; . == 2048) and
  all(.errors[]; . == "") and
  all(.generated_texts[]; (ascii_downcase | contains("[error:")) | not)
' /tmp/k3-ladder-slo-c32-out2048/seed0.json
```

The frozen result is 131.162 output tok/s with p50 TPOT 238.49 ms. Repeat the
short C32/out128 run before and after the long soak and compare the complete
`generated_texts` arrays. The measured pre/post-soak array hash was
`a8f19bfa73d0dfd31cf161e1ac82c9d52146785b0fe99e610a052b0181a000a7`.

For context sweeps through 32K:

```bash
nix develop .#vllm --command env \
  PLOW_K3_BASE_URL=http://127.0.0.1:8018 \
  PLOW_K3_NWARM=1 \
  scripts/k3_context_sweep.sh
```

The script rejects 32768 because chat framing would exceed the asset limit and
checks every request's input/output lengths and error fields.

## Fusion and roofline status

The model is fused, but not fully fused.

- Shipping devgen folds AttnRes+norm, KDA Q/K/V/G decode projections,
  recurrent state+gate, KDA gated norm, selected shared-expert GLU at B1,
  grouped routed GLU+FP4 bridge, and TP collective variants where their
  numerical and slot contracts permit it.
- Batched K3 still has separate gate/up GEMVs, SiTU, RouterTopK, Align, grouped
  GLU, grouped DOWN, combine, and collectives. The final B32 schedule overlaps
  independent shared-expert projections with routing instead of pretending
  they are one kernel.
- The rewrite/Lean graph reports semantic fusions, but the shipping devgen path
  is hand-written. Lean checks ordering/framing/LDS bounds; it does not prove
  floating-point arithmetic, MXFP4 decode, cache/state transitions, or the
  hand-written kernel fusions. Hardware oracles remain mandatory.

Measured MI325X ceilings are 1,063 TF/s for the production BF16 MFMA wrapper
and 4,164 GB/s for a clean 16 GB HBM stream. The kernels do not generally touch
both roofs:

| path | achieved | utilization/observation |
|---|---:|---|
| best dense MXFP4 prefill GEMM | 420.2 TF/s | 39.5% of BF16 wrapper ceiling |
| grouped routed GLU | 219.1 TF/s | 42.5% of its roof |
| grouped routed DOWN | 57.4 TF/s | 11.3% of its roof |
| B1 ordinary decode GEMV family | about 570 GB/s after WG tuning | far below HBM stream roof; small/ragged rows and protocol dominate |

The main gap is not an absent native FP4 instruction. gfx942 has no native
MXFP4 MFMA, and the standalone A8W4 FP8-MFMA probe was 21.5% slower than the
current A4/BF16 path. Padding, selected-expert weight traffic, LDS staging,
register spills, and per-layer counter convergence dominate.

## Pending performance work

Rank these against this recipe as the control:

1. **K3 FP8 MLA flash prefill.** Add op110 to the specialized flash object and
   route only compatible segments. Projected flash share grows from about 5%
   at 8K to 17% at 32K.
2. **Cross-request packed prefill.** The mux currently runs one request chunk,
   then decode, synchronously. Pack pending KDA rows and block-diagonal MLA
   work to reduce cold C8/C32 TTFT and improve expert fill.
3. **Ragged/weight-stream-aware grouped MoE.** Remove expert padding without
   rereading weights. BM32, BK32, final-wave culling, implicit-pad removal,
   grouped weight NT, and selected-W2 cache touching were measured and rejected.
4. **Dedicated low-rung objects.** The ladder fixes correctness and admission,
   but all rungs share the MM16+walk megakernel. Select B1/B2/B4 objects with
   lower register and instruction footprints to recover single-stream latency.
5. **Pipeline recurrent-state admission.** Initializing one slot clears about
   56.6 MiB/rank across 276 KDA/conv tensors using many blocking fills. Batch or
   enqueue these clears without advancing already-live slots.
6. **Refresh MI325X tuning data.** The checked-in 686-row cell is stale against
   the final source/toolchain digest. Re-measure interpreter packets before
   allowing tuned selection; do not reuse MI300X records.

Rejected experiments stay off: KDA prefill state residency (+0.13%), selected
W2 touch (+11.55% combined latency), A8W4 FP8 MFMA (+21.5%), MM8+walk for B16
(-2.3% throughput), grouped weight NT, BM32/BK32, and implicit pad metadata.

## TP4 x PP2

TP4 alone cannot hold the full model: the measured TP8 load is about 191 GiB
per rank, so halving TP would exceed a 256 GiB MI325X. TP4 x PP2 could recover
weight capacity by placing half the layers on each four-GPU stage, but it is not
implemented: `--parallel pp` is parsed and then refused, stage-local layer/KV
ownership and boundary transfers are absent, and the serve mux has no PP
microbatch scheduler.

It is also not the next performance lever on this single-node full-XGMI system.
Each TP4 stage does roughly twice the per-layer shard work over half the layers;
pipeline microbatches spend the same idle capacity already captured more
cheaply by the decode ladder. TP4 also reduces each collective from seven peers
to three while doubling local shard work. Prior gfx942 TP4xPP2 analysis on a
similar MoE measured a 1.17x per-stage decode penalty and found batching the
better throughput lever. Re-evaluate PP only for multi-node/slow seams, a model
that no TP degree can fit, or KV capacity beyond the current 32K/B32 target.

## Known limitations

- The official K3 vLLM ROCm image reached TP8 initialization on this MI325X but
  all workers segfaulted in RCCL before weight load. No same-box vLLM-engine
  number exists, so this branch makes no "beats vLLM" claim.
- Prefix caching and speculative decoding are not part of the 131 tok/s result.
- Sustained C32 throughput is strong; cold bursts and B1 latency are not.
- `weights.json` does not preserve build flags or object hashes. Archive this
  recipe, the exact git SHA, packet, object inventory, and result JSON together.

Raw evidence is in:

- `perf-data/kimi-k3-mi325x-ladder-130tps.md`
- `perf-data/kimi-k3-mi325x-stage4.md`
- `perf-data/kimi-k3-mi325x-kernel-audit.md`
- `perf-data/kimi-k3-mi325x-prefill-experiments.md`
- `perf-data/kimi-k3-mi325x-b32-serve.md`
- `perf-data/kimi-k3-mi325x-vllm-eager.md`
