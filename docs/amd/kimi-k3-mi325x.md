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
- The branch is production-capable for the measured TP8 path. The V2 final
  ladder scored 197/200 GSM8K; comparison evidence remains incomplete because
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

Build the ladder-compatible B1 decode object separately. The grouped override
is required because ladder B1 uses grouped MXFP4 expert packets even though its
row count is one:

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=1 \
  PLOW_K3_DECODE_GROUPED=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  JOBS=2 \
  scripts/build_gfx942.sh \
  /home/lava/plow/build-amd/k3-b1-ladder-grouped
```

The partial directory is valid only as a rung override. Do not use it as the
primary HSACO inventory. Its decode objects must export
`plow_moe_pf_a4w4_arm`; the build and runtime both refuse an incompatible
object.

Build the B2/B4/B8 tiers:

```bash
for b in 2 4 8; do
  nix develop --command env \
    PLOW_DECODE_BATCH="$b" \
    PLOW_K3_DECODE_MXFP4_PROJ=0 \
    PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
    JOBS=2 \
    scripts/build_gfx942.sh \
    "/home/lava/plow/build-amd/k3-b${b}-ladder-grouped"
done
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
  PLOW_MLA_PF_V2=1 \
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

The V2 final-ladder asset scored GSM8K 197/200 (8-shot, greedy, N=200), exactly
matching the adopted single-rung B1 score. Its served 8K/16K/32K context gates
also passed. See `perf-data/archive/k3/kimi-k3-mi325x-fp8-mla-v2.md`.

## 7. Serve

The reported run held one exclusive lease for the server lifetime:

```bash
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_MLA_PF_V2=1 \
  PLOW_TP_AUDIT_COMPACT=1 \
  PLOW_CTR_DBUF=1 \
  PLOW_STATE_CLEAR_DEVICE=1 \
  PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1,/home/lava/plow/build-amd/k3-b2-ladder-grouped:2,/home/lava/plow/build-amd/k3-b4-ladder-grouped:4,/home/lava/plow/build-amd/k3-b8-ladder-grouped:8 \
  PLOW_DSTEP_LOG=1 \
  PLOW_DSTEP_EVERY=64 \
  perf-data/tools/gpulease -n 8 k3-ladder-slo-serve \
  ./target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x_ladder_router \
  --port 8018
```

`PLOW_DSTEP_*` is diagnostic and may be removed after reproduction. Keep the
placement, compact audit, and counter-double-buffer flags explicit in measured
runs. `PLOW_STATE_CLEAR_DEVICE=1` requires decode objects built from this tree;
it replaces 276 host-staged recurrent-state fills per rank with one local-HBM
kernel. The runtime chooses the narrowest rung covering the highest occupied
slot; slots are never compacted or moved. The admission controller widens on
backlog/SLO pressure and narrows with hysteresis.

The measured aggregate run used the default chunked/ragged prefill, no prefix cache, no
prefill batching, no speculative decoding, shared checkpoint mappings, and the
global-queue object selected by packet capability.

The low-rung override changes only packets whose occupied extent is at most
eight. It reduced served median TPOT by 11.9--37.4% at C1/C2/C4/C8 with
byte-identical output. B16 and B32 continue to use the primary MM16+walk
inventory.

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

## Performance status and pending work

Rank these against this recipe as the control:

The B1 long-context baseline is frozen in
`perf-data/archive/k3/kimi-k3-mi325x-b1-128k.md`: served TPOT grows from 56.33 ms at 8K
to 82.58 ms at 128K, while TTFT grows from 4.87 s to 156.63 s.

1. **K3 FP8 MLA flash prefill — adopted.** The op110 four-wave V2 arm reduces
   served TTFT by 5.01%, 9.19%, 15.38%, 23.35%, and 33.14% at
   8K/16K/32K/64K/128K. TPOT is effectively flat and GSM8K is 197/200.
   Machine-filling buckets trade L2-domain placement for wave segmentation;
   smaller buckets remain placed and single-launch.
2. **Cross-request packed prefill.** The mux currently runs one request chunk,
   then decode, synchronously. Pack pending KDA rows and block-diagonal MLA
   work to reduce cold C8/C32 TTFT and improve expert fill.
3. **Ragged/weight-stream-aware grouped MoE.** Remove expert padding without
   rereading weights. BM32, BK32, final-wave culling, implicit-pad removal,
   grouped weight NT, and selected-W2 cache touching were measured and rejected.
4. **Low-rung object selection.** B1/B2/B4/B8 are adopted; each exact-width
   object beats MM16+walk by 11.9--37.4% TPOT. Keep B16/B32 on the primary
   object; object capability must match each ladder packet.
5. **Pack cold admissions and prefill.** Device-local recurrent-state clear is
   adopted (one kernel/rank instead of 276 blocking fills). The remaining cold
   path serializes each request's prefill chunks; pack independent request rows
   without advancing parked KDA/conv state.
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

- `perf-data/archive/k3/kimi-k3-mi325x-ladder-130tps.md`
- `perf-data/archive/k3/kimi-k3-mi325x-stage4.md`
- `perf-data/archive/k3/kimi-k3-mi325x-kernel-audit.md`
- `perf-data/archive/k3/kimi-k3-mi325x-prefill-experiments.md`
- `perf-data/archive/k3/kimi-k3-mi325x-b32-serve.md`
- `perf-data/kimi-k3-vllm-mi355x-baseline.md` (different-hardware vLLM reference)

---

# Appendix A: AMD's Kimi-K3 Day-0 post — what it gives us, and what it does not

*Folded in from `docs/amd/kimi-k3-mi325x.md` Appendix A. Section numbers below are this part's own.*
Source: <https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus.html>
"Day 0 Kimi-K3 Inference Deployment with ATOM on AMD Instinct MI355X GPUs", fetched 2026-07-28.

## 1. THERE IS NO PERFORMANCE NUMBER TO BEAT. Read this before planning against it.

AMD states it explicitly:

> "This post does not make claims about throughput, time to first token (TTFT), time per output
> token (TPOT), or kernel efficiency. HBM optimization, TP8 collectives, MXFP4 Grouped MoE, KDA
> prefill and decode, MLA, and 1M-token context optimization will be addressed in dedicated
> performance-tuning posts."

and

> "The goal is not to pursue peak performance, but to answer three practical Day 0 questions: why
> the Kimi-K3 weights fit on these GPUs, how the weights are distributed under TP8, and how to
> bring up the model quickly with ATOM and run a minimal correctness check."

So unlike Kimi-K2.7 (where AMD published 5,369.6 tok/s/GPU @ conc 128 and 116.4 tok/s/user @ conc 4),
**there is no published K3 figure.** This changes the framing: K3 is not a catch-up target, it is an
open one. The only validation AMD claims is **GSM8K 5-shot, all 1,319 samples, MI355X TP8, 16K max
model length** — a correctness bar, and a reasonable one for us to aim at first.

## 2. It independently confirms four things our agents found from the checkpoint

Worth recording because they were derived here from config/tensors alone, and now have a second source:

| our finding | AMD's wording |
|---|---|
| 93 layers = 69 KDA + 24 MLA | "93 layers: 69 KDA layers and 24 Gated MLA layers" |
| tail is `KKK MM`, **not** a clean 3:1 motif | "interleaved KDA x 3 -> MLA x 1 pattern, **with one additional MLA layer at the end**" |
| 497,220 tensors | "safetensors headers of 497,220 tensors" |
| routed experts run at 3584, not hidden 7168 | "Stable LatentMoE first projects the 7168-dimensional hidden state down to 3584 dimensions before running the expert computation" |
| `attn_res_block_size = 12` | "Stores one block residual every 12 layers" |

The layer-pattern one matters most: a naive `i % 4 == 3` rule gets the last block wrong, and two
independent derivations now say so.

## 3. New information we did not have

**Official names.** It is **Gated** MLA (not plain MLA), **Stable LatentMoE**, and **AttnRes** —
worth using in code comments so future readers can find AMD's material.

**AMD's TP8 placement rules** (directly comparable to `crates/plowrt/src/asset/shard.rs`):
- attention heads sharded across ranks;
- Dense MLP + Shared Expert gate/up **column** parallel, down **row** parallel; routed expert
  w1/w2/w3 likewise;
- **"Every rank retains all 896 expert IDs; TP shards each expert's matrices rather than
  partitioning the expert IDs."** — i.e. AMD runs **TP, not EP**, for the experts. Note plow has an
  EP mode (`GLM_EP=1`, whole experts per rank) that is still unmeasured; this is a data point that
  the obvious production choice is TP-sharded experts, not expert-partitioned.
- **replicated**: MLA `q_a`/`kv_a`, KDA `f_a`, LatentMoE down/up, Norm, router, AttnRes score
  projections;
- token embedding and LM head sharded along **vocab** — which is exactly the `GLM_SHARD_HEAD=1` +
  `XArgmaxFin` work we just landed for GLM.
- text-only service does **not** load `vision_tower` / `mm_projector`.

**Weight distribution at TP8** (their table; full checkpoint 1.5609 TB, 2.78T params):

| category | full | TP8 per GPU |
|---|--:|--:|
| Routed Expert packed values + scales | 1446.456 GB | 180.807 GB |
| KDA Attention GEMM | 61.214 GB | 7.763 GB |
| Shared Expert | 24.310 GB | 3.039 GB |
| MLA Attention GEMM | 11.145 GB | 2.029 GB |
| Dense MLP | 1.453 GB | 0.182 GB |

Totals: **190.974 GiB weights/GPU**, +14.427 GiB for a 1M-token context = **205.401 GiB of 288 GiB
(71.3%)**, leaving ~82.6 GiB for everything they did not model.

**Runtime state at 1M tokens, TP8** — the number that makes the hybrid worth having:

| state | formula | layers | TP8/GPU |
|---|---|--:|--:|
| MLA latent KV | `1048576 x (512+64) x 2 B/layer` | 24 | 14.496 GB |
| KDA SSM state | `(96/TP) x 128 x 128 x 2 B/layer` | 69 | 0.054 GB |
| KDA conv state | `3 x (12288/TP) x (4-1) x 2 B/layer` | 69 | 0.002 GB |
| AttnRes 8K chunk | `8192 x ceil(93/12) x 7168 x 2 B` | 93 | 0.940 GB |

**69 layers of KDA cost 0.054 GB; 24 layers of MLA cost 14.496 GB.** That is the whole argument for
the architecture, and it matches our own `docs/kimi-k3-kda.md` conclusion (fixed-size state, 3.81x
better than all-MLA at 1M) from an independent direction.

**Their software stack is ATOM**, not vLLM or SGLang, with:
```
export AITER_USE_GROUPED_GEMM=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0
… --kv_cache_dtype fp8 -tp 8
```
`AITER_USE_GROUPED_GEMM=0` is worth noting: AMD turns AITER's grouped GEMM **off** for K3 day-0.

## 4. Two internal inconsistencies in their article — do not copy either blindly

1. The prose says "**FP32** KDA SSM states" but the formula uses **2 bytes** per element
   (`... x 128 x 128 x 2 bytes`), which is bf16. Our `docs/kimi-k3-kda.md` specifies **f32** state.
   The dtype changes the state size by 2x and is worth settling from the reference implementation
   (`fla.ops.kda`) rather than from either document.
2. The prose says "an **FP8** latent KV cache" but the MLA latent KV row also uses **2 bytes** and is
   labelled BF16 in the same table.

Neither affects our design; both are reasons to trust the checkpoint and the reference code over
prose, which is the same discipline that settled GLM's `qk_rope_head_dim` and rope convention.

## 5. What this changes for us

- **No K3 performance target exists yet.** Getting K3 running with real numbers would put us ahead of
  AMD's own published position, not behind it. Their follow-up posts are the eventual bar.
- **GSM8K 5-shot / 1319 samples / 16K length is the correctness bar to aim at**, and it is concrete.
- **Their TP8 placement rules are a free cross-check** for our shard classifier once K3 emits.
- **Experts are TP-sharded, not EP-partitioned**, in the one production config we can see.

---

# Appendix B: vLLM's Kimi-K3 day-0 post — targets, and five ideas that apply to GLM-5.2 today

*Folded in from `docs/amd/kimi-k3-vllm-day0.md`. Section numbers below are this part's own.*
> Deprecated performance reference (2026-09-02). The figures below are historical,
> cross-hardware research and must not be used as the Kimi-K3 vLLM baseline. The
> same-box 8×MI355X baseline is `perf-data/kimi-k3-vllm-mi355x-baseline.md`.

Source: <https://vllm.ai/blog/2026-07-27-k3> — "Kimi K3 Is Here: Efficient Day-0 Support on vLLM",
2026-07-27, vLLM Team and Inferact. Fetched 2026-07-28.

Companion to `docs/amd/kimi-k3-mi325x.md` Appendix A (AMD/ATOM). Where AMD published **no** performance
numbers, vLLM published plenty — but read the hardware line carefully.

## 0. SCOPE DECISION (user, 2026-07-28): DO NOT RUN vLLM FOR K3.

**K3 is a plow-only bring-up. We do not stand up vLLM for it.** Their published TP8 batch-1 figure —
**111 tok/s = 9.01 ms/token** — is the **TARGET REFERENCE** we aim at, taken as given from this post.

This is a deliberate departure from §0-BENCH's "every plow-vs-vLLM number comes from `vllm bench
serve` against a plowrt endpoint", and the honesty rule that replaces it is simple:

> **9.01 ms/token is an ASPIRATION TARGET on DIFFERENT HARDWARE (GB300 NVL72), not a head-to-head
> result.** Any K3 number we report says so in the same sentence. We never write "plow beats vLLM on
> K3" off this comparison — only "plow reaches X ms/token against vLLM's published 9.01 on Blackwell".

§0-BENCH still governs GLM-5.2 unchanged: that comparison is on one box, both engines, same client.

Why this is reasonable rather than a dodge: nobody — not vLLM, not AMD — has published a K3 number
on MI355X, and vLLM's own ROCm path is bring-up ("broader tuning on the roadmap"). A vLLM-on-MI355X
run we produced ourselves would be measuring an untuned path and would tell us less than the
arithmetic below already does.

## 1. The published numbers, and what they are NOT

| config | batch 1, per user |
|---|--:|
| TP8 | **111 tok/s** (9.01 ms/token) |
| TP16 | **118 tok/s** (8.47 ms/token) |
| TP8 + DSpark speculative | 331 tok/s |
| TP16 + DSpark speculative | **370 tok/s** (3.14x) |

**These are GB300 NVL72 — NVIDIA Blackwell, not MI355X.** vLLM's own FAQ: *"Does vLLM support Kimi
K3 on AMD GPUs? Yes. **ROCm support ships at launch, with broader tuning on the roadmap.**"* and the
acknowledgements thank *"AMD for ROCm bring-up"* while thanking *"NVIDIA for the fused KDA decode,
KDA prefill, and Attention Residual kernels"*.

So on MI355X:
- **vLLM has published no K3 number.** AMD has published no K3 number. **Nobody has.**
- The kernels that produce 111 tok/s are named as NVIDIA contributions. The ROCm path is bring-up.

That is the same shape as GLM-5.2, where vLLM's gfx950 run has no tuned AITER config and needs ~57
min of JIT — and it means **any K3 number we produce on MI355X should be compared against a
vLLM-on-MI355X number we measure ourselves**, not against 111 tok/s on Blackwell. Quoting their
Blackwell figure as "the bar" would be dishonest in both directions.

Recommended config is **8x MI355X or 8x B300, TP8**, with `--enable-prefix-caching` in the
quick-start command.

## 2. Architecture — confirms our reading, and sharpens two things

Everything our agents derived from the checkpoint holds. Two refinements:

- **Block AttnRes attends over "up to eight cached block representations plus the current
  within-block residual"** — so **<=9 sources**, exactly our spec's number, and vLLM implements it as
  an **online softmax across model DEPTH rather than sequence position**, fused into one kernel with
  the residual update at the input and optional RMSNorm on the output. That is a useful shape for
  our AttnRes work: it is FlashAttention's algorithm on a different axis.
- **"A single layer's KDA state is roughly equivalent to the MLA cache for a few thousand tokens."**
  Consistent with our 6.5625 MiB/layer/seq and with AMD's table (69 KDA layers = 0.054 GB of state
  vs 24 MLA layers = 14.496 GB at 1M).

## 3. FIVE THINGS WE CAN USE — three of them on GLM-5.2 *today*

### 3a. LatentMoE tail fusion — applies to GLM-5.2 now, and it is a COLLECTIVE restructuring
> "At the end of LatentMoE, the reduced activation from routed experts must be normalized with
> RMSNorm and up-projected before it is added to the shared-expert output. In the normal TP case,
> this requires **two all-reduces** ... vLLM instead performs **reduce-scatter on the shared experts
> and keeps all-reduce on the routed experts** because their activations need to be normalized. The
> replicated routed-expert activation then performs matrix multiplication with the up-projection in
> a **column-parallel** fashion and is added elementwise to the already-sharded shared-expert
> output. Finally, the results are **all-gathered** onto each rank using broadcast."
> — **~20% latency reduction in that step, ~7-8% end-to-end.**

**This is directly relevant to §6e-0 and to the 48% gate-stall work.** GLM-5.2 also pays two
collectives per layer (156 total, all on the critical path), and we measured the whole set at 3.84
ms. A restructuring that removes one of the two, and replaces a replicated up-projection with a
column-parallel one, is exactly the class of change we have not tried — we have only tried making
the *existing* collectives cheaper.

### 3b. skinnyGEMM — INDEPENDENT CONFIRMATION of plow's GEMV design
> "we replace generic BF16 GEMM ... with our own **skinnyGEMM**. Generic cuBLAS kernels do not
> achieve the best performance here because they are optimized for more general shapes. In the
> kernel, we **bypass shared-memory data staging, load activations and weights directly into
> registers, and use CUDA Core FMA instructions** ... This avoids the heavy TMA and Tensor Core setup
> phase." — 8-100% kernel speedup, **~10% end-to-end in small-batch**.

That is plow's GEMV thesis, arrived at independently. Our `Gemv` family already runs at **83-106% of
the 6200 GB/s ceiling** and `lm_head` at 94-106%, so this is a place where plow is *already* doing
the right thing — worth recording so nobody "improves" it toward a tensor-core path.

### 3c. Fused KDA decode — a DIFFERENT choice from ours, and worth understanding before we judge it
> "vLLM fuses the post-projection decode path — from the causal convolutions through gated RMSNorm —
> into **a single specialized CUDA kernel**. The kernel updates the convolution and recurrent states
> in place and writes the normalized output directly, avoiding intermediate tensors, repeated state
> traffic, and per-operation launch overhead."

**We deliberately did the opposite**: `docs/kimi-k3-kda.md` decomposes KDA into **14 packets**,
because a monolithic op is how `Mamba2Scan` died and because the register objection dissolves once
the state is a declared HBM tensor. Our decomposition measured **zero extra VGPRs, 256/256 blocks,
100% occupancy**, and the one-layer gate passes.

The two are not obviously in conflict: vLLM pays *per-operation launch overhead* because each op is
a CUDA kernel launch, whereas in plow a packet is not a launch — the whole decode is **one dispatch**
and packets are counter-gated work items. **The fusion argument that motivates their kernel is an
argument plow's architecture already answers.** Worth stating explicitly rather than assuming we are
behind. What we should still steal is the *in-place state update* and *no intermediate tensors*.

### 3d. FlashKDA is open source and is the reference to check numerics against
> "Moonshot AI first released **FlashKDA**, a high-performance CUTLASS implementation of KDA ...
> Shikhar Mishra then optimized the kernels for H100 and published **Flash-Flash-KDA**."

Our KDA gate currently checks against `fla.ops.kda`. FlashKDA is a second, independent
implementation and a better oracle for the prefill scan we have not built yet.

### 3e. Prefix caching is ON in their quick-start — which settles §21
`--enable-prefix-caching` appears in the recommended command and there is an FAQ entry for it. Our
harness never passes `--no-enable-prefix-caching`, so **every plow-vs-vLLM number so far had vLLM
caching and plow not**. Bounded at ~2-3% for `random` 1024-token prompts (only the chat template is
shared), but it should be disabled for a clean comparison, or matched.

## 4. Accuracy bar, if we want one

vLLM validated K3 through a served OpenAI endpoint: **GSM8K 0.976, GPQA-Diamond 0.939, OCRBench
0.889, MMMU Pro Vision 0.818**. AMD's ATOM post claims GSM8K 5-shot over all 1,319 samples at 16K
length. Their caveat is worth keeping: *"Kimi K3 thinks a lot before it answers. A low score is more
often a truncated answer than a wrong one."*

## 5. What this does NOT give us

- No MI355X numbers, from anyone.
- The KDA prefill scan, PD disaggregation, speculative decoding (DSpark), and hybrid prefix caching
  over recurrent state are all things vLLM has and plow does not. The **hybrid prefix caching** work
  is the deepest: KDA state is updated in place, so a snapshot must be copied at a chosen boundary,
  and they added interval-based and Marconi-style ("cache on the second hit") retention policies.
  plow has no prefix caching on AMD at all.
