# plow — full flag & knob reference

Exhaustive reference for every emit-time (`plowc` env), build-time (`nvcc`/CMake
`-D…`), and runtime (`plowrt` env) flag. The README keeps only the load-bearing
traps and the four configurations anyone actually builds; everything here is for
kernel/perf work. Measured results live in `perf-data/` (one JSON + MD per
campaign).

Perf features from the rtx-11/12/13 campaigns are gated so the **default build is
a fixed, validated configuration**; every flag is an A/B control with a
correctness gate (token-identity or a bit-exact vs-f32 oracle). **Unset = shipped
default**, and most flags are byte-identical when unset.

---

## Unified CLI args (new)

Both `plowc` and `plowrt` now support structured **clap CLI args with
environment-variable fallback**. Every knob formerly set via `PLOW_*` env can now
be passed as a `--long-flag`, and the old env var still works transparently
(clap's `env` attribute reads it). **CLI takes precedence over env.** Library
benches that do not initialize CLI configuration may change cold-path resource
options before constructing the next engine.

```bash
# These are equivalent:
PLOW_PF_INTERLEAVE=4096 plowrt serve …
plowrt serve --pf-interleave 4096 …

# CLI overrides env:
PLOW_PF_INTERLEAVE=2048 plowrt serve --pf-interleave 4096 …   # uses 4096
```

### Config structs

| binary | struct | file | scope |
|--------|--------|------|-------|
| `plowc` | `EmitConfig` | `crates/devgen/src/emit_config.rs` | Emit-time knobs (114 fields, 9 hidden diagnostics) |
| `plowrt` | `RuntimeConfig` | `crates/plowrt/src/config.rs` | Shared runtime knobs |
| `plowrt` | `NvidiaRuntimeConfig` | (nested in RuntimeConfig) | NVIDIA/sm_120 serving |
| `plowrt` | `AmdRuntimeConfig` | (nested in RuntimeConfig) | AMD/gfx950 serving |

### Hot-path access pattern

Both structs use a **process-global `OnceLock`** initialized once after CLI
parse, so deep call sites pay a single atomic load (identical cost to the old
`env_flag!` macro):

```rust
// devgen (emit time)
emit_config::active().k3_full

// plowrt (runtime)
RuntimeConfig::get().nv.pf_interleave_rows()
RuntimeConfig::get().pf_packlog
```

Use `get()`, never `global()`: `global()` panics when the config was never
installed, which is the normal state for every library embedder — GPU tests,
examples and benches build engines directly and never run `main()`'s CLI
parse. `get()` falls back to an env-only parse there.

### Discovering all flags

```bash
plowc --help          # shows all emit-time flags with env fallback
plowrt serve --help   # shows all runtime flags with env fallback
```

### Environment compatibility

Environment variables declared by the current config structs continue to work.
Three compiler/runtime ambiguities have explicit replacements:

| removed spelling | replacement |
|---|---|
| `PLOW_QWEN_DECODE_LT` | compiler option `PLOW_EMIT_DECODE_CUBLASLT` |
| `PLOW_SEG_PACKED_PREFILL` | compiler option `PLOW_EMIT_PACKED_PREFILL` |
| `PLOW_NV_PLACE_DISPATCH` | runtime option `PLOW_L2_PLACE_DISPATCH` |

PlowRT reads runtime configuration through `RuntimeConfig`; the retired
`env_flag!` and `env_usize!` macros are no longer part of the runtime API.

**Why there are so many.** The interpreter is one persistent megakernel that
inlines every op arm, so its **register and shared-memory footprint is the WORST
CASE over everything compiled in**, and smem is the *union* over all ops in the
object. A knob therefore usually does one of two things: compile an arm *out* to
buy back registers/occupancy for everything else, or A/B a body against the
shipped one.

---

## `plowc` emit-time knobs

- ⚠️ **`PLOW_UNISEG=1` is NVIDIA-only. Do not pass it when targeting gfx950** —
  every AMD recipe that sets it produces a *broken* AMD asset, and the breakage is
  silent. It collapses every op into one segment, which is right on sm_120 (that
  interpreter runs one cooperative launch and never reads a wave class) and
  destroys AMD's wave-class split. With one segment, segment 0 contains the flash
  packets, the class-4 test matches, and the **entire prefill program is dispatched
  on `interp_flash`** — whose body is `if (op == FLASH_PREFILL…)` with no switch, so
  every GEMM, norm and lm_head is silently dropped. Prefill "completes" in 8.7 ms
  instead of 72.1 and the logits are all zero. A correct Gemma-4 31B emit has
  **121 segments per prefill bucket** (`2·layers + 1`); check `build.json` if in
  doubt.
- `PLOW_UNISEG=1` — single-segment programs (required for the prefill buckets on
  the sm_120 interpreter).
- `PLOW_DECODE_BATCH=B` — emit a batched decode program for multi-user serving
  (WS-GEMV shares weight reads across streams). Kimi-K3 supports `B` in 1..32;
  `B>16` requires `PLOW_GEMV_WALK=1`. Other model families may impose a lower
  ceiling. `B=1` blobs are byte-identical to unset.
- `PLOW_MAX_CHUNK=N` — largest prefill chunk for this compile (power of two,
  ≤ 8192). **This caps the bucket ladder**, so it also sets the ceiling for the
  runtime `PLOW_PF_INTERLEAVE` — raising that knob above this value is a no-op.
  Default is *window-derived*: `next_pow2(window)` clamped to [128, 8192], so a
  Gemma-4 asset (window 1024) emits a **1024**-row max chunk, not 8192. The
  tradeoff is prefill launches against sliding KV: the ring is sized
  `next_pow2(window + chunk - 1)`, so at chunk 8192 Gemma-4 rings 16384 rows =
  5.0 GiB/seq, while chunk 1024 rings 2048 = 0.625 GiB/seq — **8×** the KV for
  fewer launches. Bigger chunks are modestly faster per prefill once px4 is on
  (65k prefill, B=2: 2048 → 12.39 s, 4096 → 11.57 s, 8192 → 11.31 s, i.e. **9%**),
  so this is a KV-capacity vs prefill-latency dial: worth raising at B≤2 on a
  large-VRAM part, not at B=8 on 32 GB. Check what your asset actually emitted
  before attributing a prefill cost to chunking.
- `PLOW_PF_LADDER=wave` — derive the prefill bucket rungs from the target's **SM
  count** instead of the default power-of-two ladder (PX-6,
  `perf-data/px6-sm-quantization.md`). Prefill GEMM cost is a *staircase* in
  `tm = ceil(t/128)`: flat between wave boundaries, so rows added inside a tread
  are free and one row past a tread top costs a whole extra wave of every op that
  stepped — measured, `N = 170·128` runs 1 wave in 0.18362 ms and `N = 171·128`
  runs 2 in 0.30368 ms, i.e. **0.6% more work for 65% more time**. The shipped
  `[128, 512, 1024, 2048, 4096, 8192]` rungs are powers of two, which is unrelated
  to where the treads are; on the Gemma-4-12B op mix at `n_cu=170` they give up
  **9.6%** of prefill GEMM time on average over L = 128…4096 (worst cells +41.9%
  at 640 rows, which must be served as 128+512). The tread-top rungs the model
  picks — 1408, 2176, 640, 1792, none a power of two — take the mean loss to
  **1.4%**. Same rung *count*, so blob size and compile time are unchanged; only
  the positions move. Unset ⇒ byte-identical.
  **The ladder is a function of `n_cu` and is NOT portable**: 170 = 2·5·17 and
  188 = 2²·47 put the treads in completely different places, which is the whole
  reason it is derived rather than hardcoded. Emitting for the wrong SM count is
  worse than the power-of-two default.
  **Scope:** this optimises *covering* loss — the padding waste when a prompt
  length falls between rungs. It is worth nothing at long context, where a prompt
  is served as many repetitions of the *max* rung and the interior rungs are never
  used (measured on a 127k prompt: 31.00 s → 30.94 s). Use it for short/medium
  prefill. NVIDIA-only (the rungs assume the sm_120 `PGM_BM/PGM_BN = 128` tile).
- `PLOW_FP8_HEAD=1` — emit an e4m3 tied embed/lm_head (rtx-19 E5). −3.4/−3.5%
  decode TPOT (the lm_head is the biggest fixed-cost decode op; the win is
  ctx-independent so it's largest at short ctx). Requires the fp8 twin to
  **include** the embed/lm_head tensor (the stock twins do not — regenerate).
- `PLOW_FUSE_ARGMAX=1` — fold greedy argmax into the lm_head GEMV epilogue
  (byte-identical, ~0 perf — the logit round-trip is ~0.1%; a correctness-neutral
  cleanup, kept as a flag).
- `PLOW_FINE_FORCE=1` — keep per-slice (**fine**) counter gates instead of the
  default whole-op (**coarse**) ones. The emitter declares a fine edge wherever a
  consumer slice reads only part of a producer (headnorm→flash, flash→merge, MoE
  down→GLU); by default `select_granularity` collapses every *homogeneous* region
  back to coarse, because `lean-plow/Plow/CounterGranularity.lean:collapse` proves
  fine buys nothing when per-slice work is uniform — and it isn't free (an extra
  counter per producer slice, an extra atomic per producer, a wider wait list).
  This lever keeps the fine edge iff it is genuinely *sparse* (some consumer slice
  waits on strictly fewer than all producer slices) so it isolates the recoverable
  straggler gates without paying the 256×256-atomic all-to-all cost. It exists to
  **measure** the real-hardware straggler delta the uniform cost model can't see:
  on dense Gemma it was a wash-to-loss (16.9 → 17.2 ms/token), which is why coarse
  is the default. Lean-safe (a fine list only lowers a threshold / narrows a wait
  set), and **unset = byte-identical** coarse. There is no all-to-all "everything
  fine" mode — see the design notes.
- `PLOW_L2_PLACE=1` — **L2-domain packet grouping** (compiler half of physical-SM
  locality). *Was `PLOW_NV_PLACE`, still accepted as a deprecated alias: an L2
  domain is a GPC on NVIDIA and an XCD on AMD, and `hwspec` describes both, so the
  NVIDIA-specific name was wrong about its own scope.* Groups the device blob's
  global-queue stream into P per-L2-domain windows, so a full op's slices spread
  evenly across all domains and slice `s` stays in one domain across ops (consumer
  reads producer from the same L2 slice). It does NOT touch `cus` (so it can't
  regress `Builder::split` disjointness) and prints a static allocation report
  (`l2 placement: … map … packets/domain […] skew …%`). **Unset = byte-identical**;
  no-op on unpartitioned GPUs (e.g. consumer Blackwell).

  **The workgroup→domain map is vendor-specific and is MEASURED, not assumed.**
  `interp`'s `cu` is `blockIdx.x`, a *logical* index. NVIDIA fills a GPC with
  consecutive blocks (`n / sms_per_partition`); AMD's dispatcher assigns
  workgroups to XCDs **round-robin** (`n % partition_count`) — measured at
  **100.0%** against `HW_REG_XCC_ID` on MI355X over six geometries
  (`runtime/tests/xcd_map_gfx950_test.hip`), where the block formula scores 12.5%.
  Using the wrong one still emits correct tokens; it just destroys the locality it
  claims to create, invisibly. `L2Map` in `packet::devbuild` carries this as data.

  The locality is realized by the runtime half (`-DPLOW_L2_PLACE_DISPATCH`, or the
  `PLOW_L2_PLACE=1` arm of `scripts/build_gfx950.sh`), where each workgroup takes
  its window from the domain it is **physically** running on and all domains drain
  concurrently in one launch. Costs nothing: 248 VGPR / occ 2 / 0 spill, identical
  to the plain GQ decode object (`scripts/l2_regcheck.sh`).

  Guards: placement is **skipped** (byte-identical, with a note) when the block map
  would run off the end (`n_cu > partition_count·sms` — occupancy>1 or a
  grid≠sm_count mismatch; round-robin needs no such guard and is measured to hold
  at occupancy 2), and on any program with **more than one wave class on a target
  that relaunches per segment** — there `seg` already carries the class the host
  dispatches on, and overwriting it sends the whole prefill to the 4-wave flash
  object and returns zero logits. So on gfx950 the **decode** program is placed and
  prefill is not. A placement blob carries a header flag (`PLOW_BLOB_F_L2DOM`, plus
  SMs/partition + domain count in `reserved`) that a runtime **without**
  `PLOW_L2_PLACE_DISPATCH` refuses at load.

  **Measured on Gemma-4-31B decode (MI355X, 3 interleaved folds, 64 steps): no
  effect** — 16.57 vs 16.54 ms/token, with fold deltas of +2.7%, −3.1%, −0.2%.
  Expected, and the arithmetic says so: decode streams **61.4 GB of weights per
  token** with zero reuse, against ~7 MB of activations, so even perfect L2 capture
  addresses ~0.01% of the traffic. The mechanism works (8 windows, 3.0% skew,
  token-identical); the *lever* is not on this workload. See the plan for where it
  is.

### Every `EmitConfig` knob (`plowc --help`, `crates/devgen/src/emit_config.rs`)

One row per field, grouped by what it controls. Every row is a `--flag` with an env
fallback (CLI wins). **Unset = the shipped default**; a `true` default with "`=0` is the
rollback" is a promoted MI355X campaign result and the `=0` arm is the pre-promotion
packet. Per-knob provenance (promoted / rollback / opt-in / rejected / diagnostic) and
the 2026-09-04 audit that removed the rejected experiment knobs are in
`docs/k3-mi355x-20260904/emit-knob-audit.md`.

#### Precision

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_FP8` | `--fp8` | false | Enable fp8 weight encoding. On dense families this is w8a16 (sm_120) or triggers a refusal pointing at --w8a8 (gfx950). On MLA+MoE families it enables block-fp8 expert arms. |
| `PLOW_W8A8` | `--w8a8` | false | fp8 weights + fp8 activations (the w8a8 profile). Mutually exclusive with --w8a16. |
| `PLOW_W8A16` | `--w8a16` | false | fp8 weights, bf16 activations (w8a16 profile). Mutually exclusive with --w8a8. |
| `PLOW_MXFP4` | `--mxfp4` | false | MXFP4 (A4W4) encoding — both operands are 4-bit with E8M0 microscales. |
| `PLOW_FP8_KV` | `--fp8-kv` | false | e4m3 KV cache (halves KV bytes). Lossy — greedy diverges after ~21 tokens. |
| `PLOW_FP8_KV_FULL` | `--fp8-kv-full` | false | Mixed fp8 KV: restrict e4m3 cache to full-attention (hd512) layers only. Requires --fp8-kv. |
| `PLOW_FP8_HEAD` | `--fp8-head` | false | Emit an e4m3 tied embed/lm_head (rtx-19). Requires the fp8 twin to include the embed/lm_head tensor. |

#### Scheduling / segmentation

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_UNISEG` | `--uniseg` | false | Single-segment programs. Required for sm_120 prefill interpreter. WARNING: do NOT set on gfx950 — silently breaks AMD assets. |
| `PLOW_SEG_DECODE_MLA` | `--emit-decode-mla-segments` | true | Isolate pure adjacent FlashMlaDecode+MlaMergeFold pairs in their own gfx950 segment. Default on; `=0` is the rollback to the interpreter-resident pair. |
| `PLOW_SEG_DECODE_GROUPED_MOE` | `--emit-decode-grouped-moe-segments` | unset | Isolate adjacent grouped MXFP4 GLU+DOWN decode pairs into ordered raw launches. Unset = decide from qualified per-geometry route measurements (`moe_decode_measurement.jsonl`, both routes, current digests); missing evidence keeps the interpreter route. `PLOW_MOE_DECODE_STANDALONE=1` remains the packet-level override. |
| `PLOW_DECODE_BATCH` | `--emit-decode-batch` | 1 | Single-program decode width used by emitters without a production ladder and when no ladder is selected. |
| `PLOW_DECODE_BATCH_LADDER` | `--emit-decode-batch-ladder` | up to `1,2,4,8,16` on qualified serving emitters | Decode widths emitted as separate programs in one blob. The runtime selects the smallest rung covering the highest live slot; multistep retains the same selection. Set `1` to emit only B1. The widest rung sizes per-slot state. Dense automatic ladders shrink to fit the target memory estimate at the full compiled context, with 2 GiB headroom; an explicit ladder is preserved. Live KV still grows on demand at runtime. |
| — | `--emit-decode-objects DIR` | unset | Bind packet-selected CUDA objects from `DIR` to every decode program, including a single-program B1 packet. Packet metadata selects them at load; there is no runtime route flag. |
| — | `--emit-decode-projection-tuning` | false | Apply measured per-projection bindings. Requires decode objects, SM90a, TP1, and a supported dense emitter. |
| `PLOW_MAX_CHUNK` | `--emit-max-chunk` | unset | Largest prefill chunk rows (power of two, ≤ 8192). Caps the bucket ladder and the runtime PLOW_PF_INTERLEAVE ceiling. |
| `PLOW_MAX_REQUEST_CHUNK` | `--emit-max-request-chunk` | unset | Maximum real rows one request contributes to a packed prefill launch. |
| `PLOW_GEMV_SPLIT` | `--gemv-split` | 1 | Emit S·n_cu decode slices for Gemv packets (finer work-stealing). |
| `PLOW_GEMV_DECODE_ROLE` | `--gemv-decode-role` | false | Select the isolated BF16 M1 GEMV role with 512 threads. Requires plain BF16 SM90a and one B1 decode rung. |
| `PLOW_DECODE_TILED` | `--decode-tiled` | false | AMD: emit prefill (tiled) opcodes into the decode bucket. |
| `PLOW_UNISEG_MAX_T` | `--uniseg-max-t` | unset | Force single-segment emit for buckets at or below this T. |

#### Fusion (cross-model)

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_FUSE_ARGMAX` | `--fuse-argmax` | false | Fold greedy argmax into the lm_head GEMV epilogue. |
| `PLOW_NO_FUSE_QKV` | `--no-fuse-qkv` | false | Revert fused QKV to split-3 path (A/B control). |
| `PLOW_FUSE_QKV_FP8` | `--fuse-qkv-fp8` | false | Fused Q\|K\|V, per-channel fp8. |
| `PLOW_NO_FUSE_NRN` | `--no-fuse-nrn` | false | Disable norm+residual+norm fusion. |
| `PLOW_FUSE_HNR` | `--fuse-hnr` | false | Fuse head-norm + reduce. |
| `PLOW_FUSE_MERGE` | `--fuse-merge` | false | Fuse merge fold. |
| `PLOW_HN_SPLIT` | `--hn-split` | false | Head-number split (3*nhn <= n_cu). |
| `PLOW_QNORM_FUSE` | `--qnorm-fuse` | false | Fuse the q/k RMSNorm into the QKV GEMV epilogue. |
| `PLOW_FUSE_QUANT` | `--fuse-quant` | true | Fuse activation quantisation into the producing epilogue. DEFAULT ON for AMD (opt out with `=0`); the `amd &&` guard stays at the call site. |
| `PLOW_FUSE_RESIDUAL_INPUT` | `--fuse-residual-input` | true | Fold graph-adjacent materialized Residual inputs into AttnRes. Bit-identical and model-independent. DEFAULT ON; set `PLOW_FUSE_RESIDUAL_INPUT=0` to roll back. |
| `PLOW_NO_GLU_FUSE` | `--no-glu-fuse` | false | Opt OUT of the fused GLU GEMM on non-AMD backends. DEFAULT ON (`=1` disables). |
| `PLOW_TMA_GEMM` | `--tma-gemm` | false | Emit TMA descriptors for GEMM operands (sm_90a+). |
| `PLOW_PF_GFUSE` | `--pf-gfuse` | false | Fuse the prefill norm pair on Gemma-4 even off the gemv family. |

#### Attention / prefill ladder geometry

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_FA_GF_FULL` | `--fa-gf-full` | unset | AMD flash-decode GQA fusion factor on full-attention layers. |
| `PLOW_ATTENTION_DECODE_BALANCE_GF` | `--attention-decode-balance-gf` | unset | Experimental full-attention decode GQA fusion factor with batch-aware balanced split counts. Values: 2, 4, 8, 16. |
| `PLOW_ATTENTION_PF_ROLE` | `--attention-pf-role` | false | Mark native HD256 prefill attention as a packet-selected object role. |
| `PLOW_ATTENTION_PF_ISOLATE` | `--attention-pf-isolate` | false | Isolate prefill attention while retaining the broad interpreter role. |
| `PLOW_NS_MUL` | `--ns-mul` | unset | Scale the CU-fill target for flash-decode nsplit. |
| `PLOW_NS_ABS` | `--ns-abs` | unset | Pin nsplit absolutely. |
| `PLOW_NS_FULL_ABS` | `--ns-full-abs` | unset | Pin nsplit for full-attention layers only. |
| `PLOW_MLA_NS` | `--mla-ns` | unset | Pin the MLA flash-decode `nsplit` (K3 and GLM). Unset = the measured/ctx-adaptive default; this is the sweep handle for a re-measurement. |
| `PLOW_PF_LADDER` | `--pf-ladder` | unset | Prefill bucket ladder derivation: "wave" for SM-count-derived rungs. NVIDIA-only. |
| `PLOW_PF_LADDER_APPEND` | `--pf-ladder-append` | unset | Extra prefill ladder rungs, comma-separated (T32: e.g. "640,1152,2176,4224" swallows the chat template's +14-row overhang in one chunk instead of a second full-model pass). Rungs above the chunk cap are filtered. |
| `PLOW_PF_GEMV_HEAD` | `--pf-gemv-head` | unset | Force prefill lm_head onto M=1 GEMV arm vs tiled. "1"/"0" to force. |

#### GEMM / GEMV geometry

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_GEMV_MM` | `--gemv-mm` | unset | AMD compile-time decode row-batch bucket. |
| `PLOW_GEMV_WALK` | `--gemv-walk` | false | Wide-arm walk loop for AMD GEMV. |
| `PLOW_GEMV_WG` | `--gemv-wg` | unset | Cap the dispatch width of the fused prefill GEMV. |
| `PLOW_GEMV_WG_TUNING` | `--gemv-wg-tuning` | unset | Shape-keyed workgroup caps for blocked decode GEMVs, `NxK=cap[,NxK=cap...]` (for example `896x7168=224,1536x7168=152`). An A/B override: there is no TuneDB record for GEMV width, so unset keeps the normal workgroup selection. |
| `PLOW_GEMM_WIDE_C8` | `--gemm-wide-c8` | true | Allow the gfx950 128x384x64 `GemmWide` body on a dense BF16 GEMM. The shape is derived, not configured: the tile is taken only at the ladder-cap chunk where the exact MxNxK has a qualified TuneDB measurement naming it the winner and its grid fills every CU. Default on; `=0` is the rollback to the 128x256x64 body everywhere. |

#### Collectives / TP seams

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_XR_CUS` | `--xr-cus` | unset | Cap XReduce participant CUs. |
| `PLOW_XR_DEC_CUS` | `--xr-dec-cus` | unset | Cap the DECODE one-shot `XReduce` at N workgroups (each thread then loops over `ceil(elems/(512·N))` elements); the prefill two-shot is untouched. At rows 20 the one-shot saturates at 240 workgroups, each polling the gate and taking a system-scope acquire. Bit-identical; opt-in pending the rung-20 A/B. |
| `PLOW_XR2_GATHER` | `--xr2-gather` | true | Use reduce-scatter/all-gather for complete folded-gather collectives. The second partial is added while the reduced slices are gathered. Default on; `=0` is the rollback to the one-shot collective. |
| `PLOW_SEQ_PAR_SEAMS` | `--seq-par-seams` | true | Sequence-parallel TP seams for prefill: run AttnRes / router / latent xe / top-k on the reduce-scatter-owned `t/tp` row band and all-gather the results (`XReduceScatter` + `XAllGather`) instead of replicating the row work on every rank. Default on; the manifest requires the paired seams arm. `=0` is the rollback to the replicated-row packet. |
| `PLOW_XR_COMBINE_FOLD` | `--xr-combine-fold` | true | Fold the decode latent `MoeCombine` into the tagged one-shot `XReduce` publish: the XReduce packet carries `t1 = part`, `i7 = top_k` and no combine packet is emitted. Needs a `PLOW_XR_COMBINE_FOLD=1` decode object. Default on; `=0` is the rollback. |
| `PLOW_ATTNRES_F32MIX` | `--attnres-f32mix` | true | Emit prefill AttnRes packets with the f32-mix contract (separate output-norm epsilon in `f[1]`) and isolate them for the gfx950 `attn_res_f32mix` object. Default on; tokens differ from the BF16-seam contract by design. `=0` is the rollback to the interpreter BF16-seam packet. |
| `PLOW_ATTNRES_DECODE_MWG` | `--attnres-decode-mwg` | unset | Decode AttnRes on N column-band workgroups with an in-packet tagged rendezvous (`d_attn_res_mwg`, C3 f32-mix contract). 0/unset = the single-workgroup arm. |

#### Grouped MoE prefill / decode

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_MOE_STAGE1_LEAN` | `--moe-stage1-lean` | true | Isolate compatible MXFP4 grouped-MoE gate/up prefill packets for the standalone stage-1 object. Default on; `=0` is the rollback to the interpreter route. |
| `PLOW_MOE_STAGE2_LEAN` | `--moe-stage2-lean` | true | Isolate compatible MXFP4 grouped-MoE Down+Combine prefill boundaries for the standalone deterministic stage-2 object. Default on; `=0` is the rollback to the interpreter route. |
| `PLOW_MOE_COMBINE_LEAN` | `--moe-combine-lean` | true | Isolate compatible fixed-order grouped-MoE prefill combines for the standalone combine object. Default on; `=0` is the rollback to the interpreter route. |
| `PLOW_MOE_ALIGN_PAR` | `--moe-align-par` | true | Split the grouped-MoE prefill align into expert-parallel count/prefix/scatter packets (T >= 1024). Default on; `=0` is the rollback to the single align packet. |
| `PLOW_MOE_PREFILL_EP` | `--moe-prefill-ep` | false | Whole-expert (expert-parallel) prefill route for graph-proven replicated MoE boundaries. Opt-in; the emitted EP asset is experiment input for `runtime/bench/amd/moe_ep_boundary`. |
| `PLOW_MOE_PF_DET` | `--moe-pf-det` | false | Deterministic fused DOWN->combine for the grouped MoE prefill: op 86 accumulates an integer-valued f64 per token so the k-way sum is exact and order-independent, op 87 reads one contiguous stream. Requires an object built `-DPLOW_MOE_PF_DET=1` (`plow_moe_pf_det_arm`). Opt-in: gate-passed on GLM-5.2/gfx942 (paired GSM8K 0.9613 vs 0.9613, TTFT -1.7..-2.9%) and the gfx942 recipe sets it at emit, but `moe_pf_fuse` serves every MLA+MoE model and a default-on would make Kimi/DeepSeek blobs require the arm on evidence measured only on GLM. Flip only alongside a Kimi/DeepSeek accuracy run. |

#### KDA / MLA chain

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_KDA_FB_FOLD` | `--kda-fb-fold` | false | Fold the K3 decode `f_b` forget-gate GEMV into `KdaStateStepG`'s prologue (L3): the step packet carries `t4 = f_a`, `j1 = W_fb`, flags bit 2, and the GEMV packet is not emitted. Needs a `PLOW_KDA_FB_FOLD=1` decode object. Opt-in candidate (default off). |
| `PLOW_KDA_DECODE_FUSED_ARM` | `--kda-decode-fused-arm` | false | Run the K3 decode KDA chain (Conv3 -> StateStepG -> GatedNorm) as ONE dataflow-gated `KdaStateStepG` packet on the step's slices (L7): flags bit 3, `t0 = y`, `t1..t3` raw q/k/v, `t7` = operand descriptor; the tiles of a head rendezvous through a loader-zeroed scratch. Composes with `PLOW_KDA_FB_FOLD`. Needs a `PLOW_KDA_DECODE_FUSED_ARM=1` decode object. Opt-in candidate (default off). |
| `PLOW_GEMV_PREFETCH` | `--gemv-prefetch` | false | Decode objects prefetch a claimed `Gemv` slice's weight rows to L2 (LDS-DMA, no VGPRs) before polling its gate (L8). Packet-inert; recorded in the manifest so the paired `plow_config.h` defaults the object's `PLOW_GEMV_PREFETCH`. Opt-in candidate (default off). |
| `PLOW_KDA_CHUNK` | `--kda-chunk` | unset | Emit the BT64 chunk-KDA prefill pipeline. Default on for gfx950; unsupported shapes keep the serial recurrence. `=0` forces the serial oracle (rollback). |
| `PLOW_KDA_CHUNK_QPRE` | `--kda-chunk-qpre` | true | Precompute the V-independent scaled/gated query in chunk W/U. Default on; `=0` rollback. |
| `PLOW_KDA_INTRA_WAVE_ITEMS` | `--kda-intra-wave-items` | true | Isolate exact BT64/D128 chunk-KDA intra packets for the wave-item gfx950 object. Default on; `=0` is the rollback to the interpreter path. |
| `PLOW_KDA_CARRY_REGSTATE` | `--kda-carry-regstate` | true | Mark exact qpre BT64/D128 carry segments for the register-resident gfx950 carry object. Default on; the marked packet requires its paired object at load. `=0` is the rollback to the interpreter carry. |
| `PLOW_KDA_KEY_FACTOR` | `--kda-key-factor` | true | Mark exact qpre BT64/D128 Wu->carry pairs for the spill-free key-factor gfx950 objects. Default on at emit; the runtime only takes the route when those objects are built (`PLOW_HSACO_KDA_KEY_FACTOR`, default OFF: the pair displaces the faster regstate carry). |
| `PLOW_KDA_WU_LEAN` | `--kda-wu-lean` | false | Mark exact qpre BT64/D128 chunk-KDA Wu segments for the lean four-wave gfx950 Wu object. Opt-in candidate (TP8 gate pending); the marked packet requires its paired object. |
| `PLOW_KDA_CARRY_KEYFEED` | `--kda-carry-keyfeed` | false | Feed the lean Wu's scaled-key hi/lo pair into the register-state carry (implies the lean Wu; needs `PLOW_KDA_CARRY_REGSTATE`). Opt-in candidate (TP8 gate pending). |
| `PLOW_KDA_DECODE_FUSED` | `--emit-kda-decode-fused` | false | Emit the standalone fused KDA decode boundary when its geometry is supported. Opt-in (benchmark-only so far); unsupported shapes keep the Conv3 -> StateStepG -> GatedNorm chain. |
| `PLOW_MLA_MATERIALIZED_PREFILL` | `--mla-materialized-prefill` | false | Materialize MLA Q/K/V and emit the standalone asymmetric gfx950 prefill boundary. Opt-in candidate: exact for the first chunk, continuation chunks still diverge. |
| `PLOW_K3_KDA_CONV_STEP_DB` | `--k3-kda-conv-step-db` | false | Emit `KdaConvStateStepG` (Conv3 + StateStepG with ping-pong convolution windows) for B1 decode. Opt-in; the decode object must be built `PLOW_K3_KDA_CONV_STEP_DB=1`. |

#### K3 family

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_K3_FUSE_A` | `--k3-fuse-a` | false | Fuse the MLA q/kv/k_rope/gate A-projection GEMVs into one `GemvQkvg` (decode only, LDS-bounded). Opt-in; not network-gated. |
| `PLOW_K3_FUSE_NGEMV` | `--k3-fuse-ngemv` | unset | Fold the decode B1 `RmsNorm -> GEMV` pairs into the GEMV's LDS staging. Default on (bit-exact); `0` is the unfused rollback, `lat`/`q` keep one site for bisection. |
| `PLOW_K3_FUSE_ARNORM` | `--k3-fuse-arnorm` | true | Fuse each AttnRes with its sole following RMSNorm (bit-exact). Default on; `=0` is the rollback, used to materialize the raw residual seam for a boundary capture. |
| `PLOW_K3_SHARD_HEAD` | `--k3-shard-head` | false | Vocab-column-parallel K3 `lm_head` with an `XArgmaxFin` handoff. Rejected for serving (TTFT +8 ms for TPOT -0.09 ms); kept for `scripts/k3_tp_equivalence.sh`. |
| `K3_PREFILL` | `--k3-prefill` | unset | K3 prefill bucket control: unset/`full` = the whole ladder, `0` = decode only, `512,1024` = those rungs. |

#### Gemma-MoE family

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_MOE_PREFILL` | `--moe-prefill` | unset | MoE prefill control. "0" to disable, unset = auto (on for MoE bf16). |
| `PLOW_GEMMA_MOE_ROUTER_FUSED` | `--gemma-moe-router-fused` | false | Disable split router, serialize score GEMV on one CTA. |
| `PLOW_GEMMA_MOE_ROUTER_BLOCKS` | `--gemma-moe-router-blocks` | unset | CTA count for the split router score GEMV. |
| `PLOW_GEMMA_MOE_ROUTER_EXACT` | `--gemma-moe-router-exact` | false | Exact MoeRouterGemmaScore op instead of ScoreFast. |
| `PLOW_GEMMA_MOE_TAIL_FUSE` | `--gemma-moe-tail-fuse` | false | Fuse MoE-combine residual/norm tail (B=1 only, reorders summation). |

#### GLM family

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_GLM_DSA` | `--glm-dsa` | unset | GLM sparse-attention arm control. "0" forces dense, unset = auto (on above ctx crossover). |
| `PLOW_GLM_GF` | `--glm-gf` | unset | Pin the MLA head-fusion factor. |
| `GLM_SHARD_HEAD` | `--glm-shard-head` | false | Vocab-column-parallel lm_head. |
| `GLM_MOE_CORESIDENT` | `--glm-moe-coresident` | unset | Co-resident shared expert mode (0/1/2). |
| `GLM_SHARED_CUS` | `--glm-shared-cus` | unset | CUs for shared expert. |
| `GLM_SPINE_CUS` | `--glm-spine-cus` | unset | Spine CU allocation (comma-separated or expression). |
| `GLM_LINEAR_FP8` | `--glm-linear-fp8` | false | fp8 shared-expert linear projections. |
| `GLM_SHARED_GLU_SPLIT` | `--glm-shared-glu-split` | false | Split GLU path for fp8 linear. |
| `PLOW_MLA_PREFILL` | `--mla-prefill` | unset | MLA prefill ladder (e.g. "full:512,2048,4096,8192"). |
| `GLM_EP` | `--glm-ep` | false | GLM expert-parallel mode. |
| `GLM_GROUP` | `--glm-group` | false | GLM grouped MoE dispatch. |
| `PLOW_GLM_FUSE_B1` | `--glm-fuse-b1` | false | GLM fuse block-1 residual+norm (opt-in, off by default). |
| `PLOW_GLM_FUSE_SEAM` | `--glm-fuse-seam` | false | GLM layer-seam fold: the FFN tail's residual and the next layer's input_layernorm as one AddNorm packet (opt-in, off by default; TP only). |
| `PLOW_GLM_FUSE_ROPE` | `--glm-fuse-rope` | false | GLM decode q-rope fold: apply the interleaved q RoPE inside the MLA flash decode's query staging and drop the `HeadNormRope` packet (opt-in, off by default). |
| `PLOW_GLM_FUSE_QNORM` | `--glm-fuse-qnorm` | false | GLM decode q-norm fold: compute `q_a_layernorm` inside fusion G's `GemvQkv` LDS staging and drop the one-workgroup `RmsNorm` packet (opt-in, off by default). |
| `GLM_ROUTER_OFF_SHARED` | `--glm-router-off-shared` | false | GLM router off-shared dispatch (co-resident mode 2 only). |
| `GLM_ROUTER_OLD` | `--glm-router-old` | false | GLM use legacy (unfused) single-CU router. |
| `PLOW_GLM_DSA_PF` | `--glm-dsa-pf` | false | Route GLM's DSA indexer through the prefill chain (requires `has_dsa`). |
| `PLOW_GLM_FP8_KV` | `--glm-fp8-kv` | on for GLM on gfx942 TP8 (production default), off elsewhere | Store the MLA latent cache as e4m3 + per-row f32 scale. NOT bit-identical. Rollback: `--glm-fp8-kv=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_GEMV_WG` | `--glm-gemv-wg` | unset | Cap the dispatch width of every blocked GEMV. Unset ⇒ byte-identical. |
| `PLOW_GLM_OFOLD` | `--glm-ofold` | false | Fold W_o into the MLA prefill flash epilogue. Reassociated, logit-gate class. |
| `PLOW_GLM_PF_NS` | `--glm-pf-ns` | unset | Causal KV-split factor for the V2 MLA prefill flash (2..=8; unset/1 = unsplit). |
| `PLOW_GLM_DSA_PF_SPAN` | `--glm-dsa-pf-span` | 1 | Sparse-prefill selection reuse span: layers after an indexer layer that gather against its union (0 = indexer layers only, 3 = every GLM-5.3 layer). |
| `PLOW_GLM_DSA_PF_DEXACT` | `--glm-dsa-pf-dexact` | unset | Reuse only at exactly this distance from an indexer layer (bisect aid; unset = 1..=span). |
| `PLOW_GLM_MOE_AITER` | `--glm-moe-aiter` | on for GLM on gfx942 TP8 (production default), off elsewhere | Native gfx942 A8 MoE prefill with BF16 routed accumulation (AITER adapter). Opt-in: A8 activation quant is ~3.5–3.9% rel-L2 vs the DET FP64 path; +6.2% C20 on GLM-5.3 TP8. Rollback: `--glm-moe-aiter=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_MOE_FLAT_DECODE` | `--glm-moe-flat-decode` | false | Flat A16 MoE for gfx942 TP8 decode rungs 2, 4 and 8. Serving qualification pending. |
| `PLOW_GLM_MLA_DEC_AITER` | `--glm-mla-dec-aiter` | false | Isolate GLM sparse FP8 decode attention (rungs 1..20) so the runtime dispatches the pinned gfx942 AITER QH8 MLA object (pack → attention → relayout into the existing merge). Falls back to the interpreter on steps where a row holds < 2048 keys. Requires `mla_sparse_adapter_gfx942.elf` with the decode marker. **Stays off: does not load on a HEAD-emitted packet** — the emit isolates `FlashMlaDecodeFp8` alone, while the runtime route needs the `FlashMlaDecode` + `MlaMergeFold` pair in ONE segment (`decode segment N contains only half of the FlashMlaDecode+MlaMergeFold pair`, 8x MI300X, 2026-09-11). Even once fixed its ceiling is small: 153 µs vs ≈176 µs per layer at rung 20 standalone, ≈1.8 ms of a ≈97 ms decode step (≈5 s on the 1511 s / 100-prompt run), below what a 20-prompt A/B resolves, and the route adds two segment boundaries per layer. |
| `PLOW_GLM_MOE_SHARED_FOLD` | `--glm-moe-shared-fold` | false | Fold GLM's shared expert into the native AITER MoE prefill call as expert 256 with a constant gate of 1.0 (router appends a 9th slot, align and the fused call run 257/9, the shared GEMM pair and the combine's `shared` operand go away). This is the shape AITER's gfx942 GLM-5 tuning table is indexed by; same pinned objects. Needs `PLOW_GLM_MOE_AITER`/`_RESIDENT`, TP, and a checkpoint that still holds the shared expert's block-FP8 bytes (the lite prep's are shadowed, and are found). Decode keeps its own shared GEMVs. Opt-in; default OFF. NOT bit-identical: the shared expert moves from bf16 to A8W8 block-FP8, so the FFN output's rel-L2 vs f64 goes from ~1.7% to ~4.8% (≈2.8×; the fold equals routed + shared to bf16 rounding, error flat across columns — a precision cost, not a defect). Measured on GLM-5.3 TP8 (ABA, one binary, per-packet objects, merged runtime): −12.9 ms per 8192-row prefill chunk (single-request TTFT at 59k 7.94 → 7.84 s; resolved against a 17.6 ms control gap), no resolvable C20 throughput change (−0.42% vs the control mean inside a +2.46% control drift), ≈ −11 s of the 1511 s 100-prompt reference; retrieval 18/18 on all five arms. About half of the earlier −25 ms/chunk went to the merged 64-row tile + XCD-swizzle fmoe, which makes the call the fold adds pairs to cheaper. A broader accuracy eval (GSM8K plus long retrieval at C20) is the gate for any default flip. Emits `…expert_{weight,scale}_table_sf`; needs packet-stamped objects, the `plow_moe_shared_fold_arm` prefill object and the `plow_moe_aiter_nexp_abi_1` adapter. |
| `PLOW_GLM_MOE_RESIDENT` | `--glm-moe-resident` | on for GLM on gfx942 TP8 (production default), off elsewhere | Pack GLM expert weights once for native gfx942 TP8 prefill and decode. +13.4% C20 / −10.9% TPOT on GLM-5.3; same A8 numerics contract as `PLOW_GLM_MOE_AITER`. Rollback: `--glm-moe-resident=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_INDEX_TP` | `--glm-index-tp` | on for GLM on gfx942 TP8 (production default), off elsewhere | Partition large GLM prefill index queries across eight gfx942 ranks instead of replicating them. Exact top-k; +8.2% serving; keep the ≥2048-row gate (small chunks regress). Also the precondition for packing a sparse bucket (`PLOW_PACKED_SPARSE_PF`): the native kernels take a per-request `PlowKvSpan` table (`dsa_tp_adapter_gfx942.elf`, `plow_dsa_tp_abi_2`), which the interpreter selectors cannot. Rollback: `--glm-index-tp=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_SELECT_LOCAL` | `--glm-select-local` | on for GLM on gfx942 TP8 (production default), off elsewhere | One independent single-workgroup radix selection per GLM decode row instead of a serialized shared-histogram chain. Exact top-k; +10.6% C20, −19% median ITL. Rollback: `--glm-select-local=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_DECODE_NORM_ROWS` | `--glm-decode-norm-rows` | on for GLM on gfx942 TP8 (production default), off elsewhere | Give each batched GLM RMSNorm / AddNorm row its own workgroup. Bit-identical; +2.7%, P99 TPOT −11%. Rollback: `--glm-decode-norm-rows=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_GEMM_LT` | `--glm-gemm-lt` | on for GLM on gfx942 TP8 (production default), off elsewhere | Qualified gfx942 hipBLASLt assembly for the large GLM prefill projections (3 shapes). Bit-exact vs capture; +2.1%. Rollback: `--glm-gemm-lt=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_GEMM_LT_DECODE` | `--glm-gemm-lt-decode` | on for GLM on gfx942 TP8 (production default), off elsewhere | Native gfx942 hipBLASLt BF16 projections at decode rungs 16 and 20 (633 GEMMs). ≤0.17% rel-L2; +5.7% then +2.4% across two screens. Rollback: `--glm-gemm-lt-decode=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_GEMM_LT_DECODE_EXT` | `--glm-gemm-lt-decode-ext` | on for GLM on gfx942 TP8 (production default), off elsewhere | Extends `PLOW_GLM_GEMM_LT_DECODE` to rung 8 and to the narrow decode projections (k_rope, q_rope, indexer k/weights, lm_head): +199 native GEMMs at rungs 16/20, all 11 shapes at rung 8, so those rungs run no interpreter GEMV. Same pinned object. Served A/B on 8x MI300X, C20 × 70k/700, same job and binary as its control: **50.68 → 51.65 out tok/s (+1.9%)**, median TPOT 220 → 202 ms, decode-only tick 96.8 → 90.3 ms (standalone predicted −7.5 ms at rung 20), 18-case retrieval 18/18. The rung-8 half is idle on that workload (a lone straggler already runs rung 20). Effect only where `PLOW_GLM_GEMM_LT_DECODE` is on. Rollback: `--glm-gemm-lt-decode-ext=false` (recorded as `cli`, and a `--replay-knobs` recipe keeps its own value). |
| `PLOW_GLM_GEMM_BLK` | `--glm-gemm-blk` | unset (off) | Native gfx942 W8A8 block-scale FP8 prefill projections (rows 2048-8192) on AITER's pre-shuffled assembly: `1` = q_a_proj, kv_a_latent, indexer wq_b and o_proj, or a comma list of `q_a,kv_a,wq_b,o_proj`. Reads the checkpoint's own FP8 bytes and `[128,128]` grids (overlay: `scripts/glm53_prep_blk.py`), declared in addition to the bf16 weights decode keeps (~2.4 GB/rank of HBM); q_absorb is a prep product and stays bf16. Objects: `scripts/build_gemm_blk.sh`. Excludes `GLM_LINEAR_FP8` and `PLOW_GLM_OFOLD`. Standalone on one MI300X at M=8192: 706-783 TF/s (84-94% of AITER's gfx942 CK table), W8A8 rel-L2 3.6-3.8e-2 vs FP64. Served (20-prompt C20 A/B, two controls, the same 161 steady chunks per arm): prefill chunk drain -24.4 ms (-3.0%) with all four, all of it from o_proj (q_a/kv_a/wq_b net 0 at the current activation-quant cost); prefill-tick wall -1.5%. Out tok/s 55.64 (all four) / 54.71 (without o_proj) vs controls 55.00 / 56.08: +0.2% / -1.5% vs their mean, inside their 2% spread, which is decode-side drift. Not decisive, so opt-in, not a default. Second A/B (v2 quant, socket-pinned binary, control spread 0.97%): prefill chunk drain -24.7 ms, prefill-tick wall -2.4%, out tok/s -0.3% vs the control mean, a resolved no-gain; the route arm's +5% ITL is decode-state drift, not the route (identical decode program, KV geometry and map counts). Retrieval 18/18; captured-row rel-L2 1.8-2.4e-2 vs the BF16 path. |
| `PLOW_GLM_FOLD_LT` | `--glm-fold-lt` | false | Native gfx942 FP32 MLA fold GEMMs during prefill. Measured +0.49% with P99 +3.6% — not a default candidate. |
| `PLOW_GLM_PF_WIDE` | `--glm-pf-wide` | true | Widen prefill norm/residual dispatch across CUs. DEFAULT ON (`=0` restores the single-workgroup emit for A/B). Bit-identical either way. |
| `PLOW_GLM_PLACE_PF` | `--glm-place-pf` | false | Per-XCD CU placement for the GLM prefill chain. |
| `PLOW_GLM_XR_BAND` | `--glm-xr-band` | unset | Band count for a prefill TP seam (2..=8; unset/1 = the unbanded emit). |
| `PLOW_GLM_XR_BAND_CUS` | `--glm-xr-band-cus` | unset | Restrict the banded seam to the first N of the seam's CU list. |
| `PLOW_GLM_XR_RES` | `--glm-xr-res` | false | Fold the post-collective Residual into the two-shot all-gather. Bit-identical. |
| `PLOW_GLM_DECODE_GLUE_CUS` | `--glm-decode-glue-cus` | false | Size the batched-decode glue packets to their work: the FP8 latent KV writer at one wave per row (was one workgroup for the whole rung), the router top-k at one workgroup per token, the MoE combine at one thread per element (were all 304 workgroups). Pure width changes, bit-identical; opt-in pending the C20 A/B. |
| `PLOW_GLM_DECODE_GEMM_GROUP` | `--glm-decode-gemm-group` | false | Batched decode (rungs with native `GemmLtPf`): emit the GEMMs that share an input next to each other — the shared gate/up straight after the router GEMM (top-k and Glu then share one interpreter segment), and on full-indexer layers the indexer k/weights projections beside q_a/kv_a/k_rope and its q projection beside q_absorb/q_rope. Same instructions and dependencies, reordered; pairing hash and objects unchanged; knob-off packet byte-identical. Rung-20 chain 1431 → 1293 packets. **Unproven:** 7-layer TP8, 14 runs, median +0.25 ms/step (per-process levels ~9.9/~10.9/~12.5 ms swamp it); one traced step −0.56 ms GPU span, class-weighted 78-layer projection −4.4 ms/tick. Code on branch `decode-latency`. Opt-in. |
| `GLM_FUSE_XRN` | `--glm-fuse-xrn` | false | Fuse the seam Residual+Norm into XReduceAddNorm (requires fuse_b1, tp>1). |
| `PLOW_GLM_WGFIT` | `--glm-wgfit` | true | Narrow GLM dispatch to the workgroups that own work. DEFAULT ON (`=0` for the A/B control arm); the emitted arithmetic is unchanged either way. |

##### The qualified GLM gfx942 TP8 recipe

Nine `glm_*` knobs are ON by default when the model is GLM (`glm_moe_dsa` / `glm5_next`), the
arch is `gfx942`, `--num-gpus 8` and the part has 304 CUs: `PLOW_GLM_FP8_KV`,
`PLOW_GLM_MOE_AITER`, `PLOW_GLM_MOE_RESIDENT`, `PLOW_GLM_INDEX_TP`, `PLOW_GLM_SELECT_LOCAL`,
`PLOW_GLM_DECODE_NORM_ROWS`, `PLOW_GLM_GEMM_LT`, `PLOW_GLM_GEMM_LT_DECODE`,
`PLOW_GLM_GEMM_LT_DECODE_EXT` (the ninth, added once served: +1.9%, retrieval 18/18). Off on every other
target, and off under `--mxfp4` (the native MoE arms are block-fp8 only).

That is the configuration serving GLM-5.3 on 8x MI300X — 47-50 out tok/s with the 18-case
retrieval screen at 18/18 — and each knob's own evidence is in its row above. It was previously
reachable only by naming all of them, so dropping one emitted a slower packet that still loaded and
still served. An emit with no `glm_*` flags at all now produces that packet byte-for-byte.

Precedence is strictly **explicit flag > env var > `--replay-knobs` > production default > plain
default**. A replayed recipe lands as an env assignment before clap parses (`apply_replay_knobs`),
so it outranks the production default and a recorded `false` survives — re-emitting a frozen
recipe reproduces it. `build.json` names the winner per knob (`cli` / `env` /
`production_default`), and production defaults are omitted from `emit_config.replay` so a replay
re-derives them from the target rather than pinning today's.

Rollback is per knob: `--glm-<knob>=false`. There is deliberately no single kill switch — the
knobs were qualified individually and are rolled back individually.

Knobs measured on the same target and deliberately NOT defaulted: `PLOW_GLM_MOE_FLAT_DECODE`,
`PLOW_GLM_MLA_DEC_AITER`, `PLOW_GLM_FOLD_LT`, `PLOW_GLM_PLACE_PF`, `PLOW_GLM_FUSE_ROPE`,
`PLOW_TOKEN_BATCH_TP` and `PLOW_PACKED_SPARSE_PF`. See each row for why.

#### Tuning

| env | flag | default | effect |
|---|---|---|---|
| `PLOW_TUNEDB` | `--tunedb` | unset | Tuning database root directory. |
| `PLOW_AUDIT_OCC_FLOOR` | `--audit-occ-floor` | 50 | Dispatch-audit occupancy floor, percent. A matmul filling less of its own CU set than this is named at emit. |
| `PLOW_AUDIT_GEMV_WASTE_MAX` | `--audit-gemv-waste-max` | 25 | Percent of GEMV row work a compiled `GV_MM_MAX` may spend on dead rows before it is a finding. |
| `PLOW_AUDIT_STRICT` | `--audit-strict` | false | Promote dispatch-audit findings from a `WARN` to a refusal (exit 1). For a build already tuned — a campaign re-emitting a measured configuration, or CI. |
| — | `--replay-knobs <build.json>` | unset | Replay a previous emit's knobs from its `build.json` `emit_config.replay`. Applied before parsing, so an explicit flag still wins. |

These three read the `dispatch_audit` section and the replay flag reads `emit_config`; both are
`build.json` sections documented in
[docs/arch/15-dispatch-audit-and-knob-manifest.md](arch/15-dispatch-audit-and-knob-manifest.md),
which also explains the compiled-ceiling bug class they exist to catch.

`PLOW_AUDIT_STRICT` is read a second time, with the same meaning, by the object-side contract in
`scripts/asm_audit.py --contract` (`scripts/build_gfx942.sh`), where it promotes the LDS-crossbar
and 2-byte-LDS advisories to build failures. `PLOW_AUDIT_JOBS` (default 8) sets its disassembly
concurrency. See [docs/arch/16-object-contract.md](arch/16-object-contract.md).

#### Diagnostics (hidden from `--help`; never serve)

| env | flag | default | effect |
|---|---|---|---|
| `K3_FULL` | `--k3-full` | true | Diagnostic: `K3_FULL=0` prints the legacy K3 capability report instead of emitting. |
| `PLOW_LAYERS` | `--layers` | all | Layers to emit (K3 and GLM): "all", a number N (first N layers), or "single:L". A truncation instrument for block sweeps and TP-equivalence checks; never a served packet. |
| `PLOW_K3_SEQ_ROWS` | `--k3-seq-rows` | false | Diagnostic: force the per-sequence GEMV row carrier at B=1 (bisects the batched-decode addressing against the known-good B=1 stream). |
| `PLOW_GLM_XR_BAND_SEAM` | `--glm-xr-band-seam` | unset | Diagnostic: restrict banding to one seam (`attn` \| `moe`) to bisect a divergence. |
| `PLOW_FLASH_MERGE_DSPLIT` | `--flash-merge-dsplit` | unset | Widen the flash-merge dispatch by this factor (diagnostic; measured no effect). |
| `PLOW_NO_XREDUCE` | `--no-xreduce` | false | Disable all XReduce collectives (diagnostic — numerically wrong). |
| `PLOW_TUNE_DUMP` | `--tune-dump` | false | Print a TUNEDUMP census line per resolved GEMV shape (tuning-harness diagnostic). |
| `PLOW_SKIP_COVERAGE` | `--skip-coverage` | false | Emit a model known to fail coverage checks (diagnostic only). |
| `PLOW_K3_ABLATE` | `--k3-ablate` | unset | K3 bisection instrument (diagnostic only). |

#### Emit-side knobs that are NOT `EmitConfig` fields

Raw `std::env::var` reads owned by `packet::devbuild` / `plowc` (see the module header of
`emit_config.rs` for why they stay raw):

- `PLOW_SEG_PER_OP=1` — one SEGMENT per op (host-side AQL chaining instead of batched);
  `PLOW_SEG_CLASS_SLICE=1` re-slices GEMM segments so both occ-2 blocks/SM get work.
- `PLOW_MOE_DECODE_STANDALONE=1` — packet-level override that forces the grouped-MoE
  standalone decode route regardless of the TuneDB rule.
- `PLOW_FUSE_XR_ATTNRES=1` / `PLOW_XR_WAVE_RS=1` / `PLOW_PHASE_OBJECTS=1` — rejected
  (+91.7 ms / +3.6 ms / +22.6 ms TTFT) segmentation experiments kept for their builder tests
  and the pending AQL-replay design; never set them for a served packet.
- `PLOW_GEMM_JSONL=<path>` — appends raw per-tile GEMM samples (tuning-harness diagnostic).
- `PLOW_MLA_PF_V2` — read in `crates/packet/src/devbuild.rs`, and it CHANGES THE PACKET. Being
  outside `EmitConfig` it is absent from `build.json`'s `emit_config` section, so it is the one
  knob a `--replay-knobs` rebuild has to be given by hand: a GLM-5.3 TP4 replay reproduces the
  blob byte-for-byte only with `PLOW_MLA_PF_V2=1` also set. Promoting it to an `EmitConfig`
  field is what would make it recordable.
- **Known-wrong escape hatches** (garbage tokens by design — never serve):
  `PLOW_CHAIN_BYPASS=<op[,op…]>` splices opcodes out of the chain; the hidden `EmitConfig`
  diagnostics above (`PLOW_SKIP_COVERAGE`, `PLOW_K3_ABLATE`, `PLOW_K3_SEQ_ROWS`,
  `PLOW_NO_XREDUCE`) are the same class.

---

## fp8 precision — full detail

fp8 is **not** a runtime toggle — it is baked when the model is compiled and the
interp cubins are built. The default build is the accuracy-safe **bf16** path;
fp8 is opt-in because it is *lossy*.

The precision flags are named by **axis**: `PLOW_W8A16` / `PLOW_W8A8` for
weights+activations, `PLOW_MXFP4` for the A4W4 (fp4+E8M0 microscale) path, and
`PLOW_KV_FP8` for the KV cache. `PLOW_FP8` and `PLOW_FP8_KV` remain as aliases.
Setting two weight flags is refused. The axes are independent and compose: bf16
weights with an fp8 KV is a legal combination and has its own object.

> **Not real flags** (comment-only axis labels, do not set them): `PLOW_W4A16`
> and `PLOW_MOE_ENC=`. The w4a16 / A4W4 encoding is realized by `PLOW_MXFP4=1`;
> the routed-expert encoding is *derived* from `PLOW_MXFP4`/`PLOW_FP8` (the
> `MoeEnc` enum in `crates/*/mla.rs`), not selected by a `PLOW_MOE_ENC` env var.

**The spelling differs by model family, because the kernels do.** An axis name
promises an encoding, so where a family cannot realize one it refuses rather than
resolving to the nearest thing it has:

| family | fp8 weights, bf16 acts | fp8 weights + fp8 acts | fp8 KV |
|---|---|---|---|
| dense (Gemma / Llama / Qwen) | `PLOW_W8A16=1` — **sm_120 only** | `PLOW_W8A8=1` — **the gfx950 recipe** | `PLOW_KV_FP8=1` |
| MLA+MoE (Kimi / GLM / DeepSeek) | `PLOW_W8A16=1` — works on gfx950 | **not implementable, refuses** | `PLOW_KV_FP8=1` |

The asymmetry is real, not an oversight. "fp8 weights, bf16 activations" is a
per-*channel* fp8 GEMM on the dense path, which gfx950 has no arm for, and a
*block*-fp8 GEMV on the MLA path, which it does. Same axis value, different
kernels, different coverage. The MLA family's expert ops (45/46/48/49) are w8a16
in **every** instantiation, so `PLOW_W8A8` there refuses and points at
`PLOW_W8A16` (fp8 weights) or `PLOW_MXFP4` (A4W4).

**On AMD (gfx950)** the fp8 profile is `PLOW_W8A8=1` (older spelling:
`PLOW_FP8=1 PLOW_W8A8=1`), plus `PLOW_KV_FP8=1` for an fp8 KV cache — plain
`PLOW_FP8=1` is refused at emit, deliberately: `PLOW_FP8=1` alone emits **w8a16**
(activation scale `t[3]` unbound), but the gfx950 `GEMM_FP8` arm is **w8a8** and
would fault on the null scale and misread A. plowc refuses, naming the fix, rather
than upgrading w8a16 → w8a8 on your behalf — every other emitter substitution is
computation-preserving, but this one quantizes the *activations* too and would
change a run's numerics under a flag set to mean something else. sm_120 has a real
w8a16 cubin and is unaffected.

**There is deliberately no separate activation flag.** w8a8 is one *profile*, not
a free cross-product with w8a16 — the kernels instantiate exactly those two. The
activation encoding is *derived* and reported in the manifest as
`precision.act_enc`, read off **`QuantFp8`'s presence** rather than inferred from
the weight flag (that inference is what once let a w8a16 packet reach a w8a8-only
object). It is read **per phase**: on the dense family the prefill GEMMs quantize
while `GemvFp8`/`GemvGluFp8` take bf16 activations, so a dense W8A8 packet reports
`act_enc: "mixed"`, not `"fp8"`. Reading the arm union alone reported `fp8` and
overstated the decode half — the same phase-blind judgement the axis exists to
replace. A consumer selecting an object on this axis must look at the phase it is
serving.

For a checkpoint that is already quantized — GLM-5.2-FP8, say — **no precision
flag is needed at all**: the encoding is read from `quantization_config`. Two
traps there: the key is **`dtype`, not `torch_dtype`** (HF renamed it), and
`dtype` reads `"bfloat16"` on an e4m3 checkpoint anyway (it describes the *compute*
dtype; storage dtype lives in `quantization_config`). The checkpoint wins over the
flags — it is a fact, they are a request — and a contradiction is refused.

The fp8 weight twins are keyed **verbatim, including the `fp8/` prefix** — the
emitter declares `fp8/<name>`, `quantize_fp8.py` writes `fp8/<name>`, and the
loader looks up `fp8/<name>`, with no stripping on either side.

**Keep `--arch` and `--gpu` in agreement** unless deliberately cross-compiling; a
mismatch now warns and is trusted toward the **GPU** (`--arch` records intent,
`--gpu` records what the packet was sized for).

Measured (gemma-4-31B, single-user): fp8 **decode beats vLLM-fp8** (−41% vs vLLM
bf16, parity-to-−3% vs vLLM fp8); fp8 **prefill** beats vLLM-bf16 at 32k and
closes the short-ctx gap to ~1.1–1.3× (still trails vLLM-fp8, which uses
cudagraphs + FA-class flash). `PLOW_FP8_KV` doubles the 31B concurrency ceiling.

---

## Interpreter compilation knobs (`nvcc -D…`)

Pass them through the build script, which forwards `PLOW_EXTRA_DEFINES` verbatim
to every object it compiles:

```bash
PLOW_EXTRA_DEFINES="-DPLOW_NV_W8A8=1 -DPGM_BN=64" scripts/build_sm120_cubin.sh <out.cubin>
PLOW_ROOT=$(pwd) …            # build a WORKTREE's sources; defaults to /root/plow
```

### The four configurations anyone actually builds

Pick a row, use the flags in it, and skip to the tables only if you are doing
kernel work. The **build** column is `nvcc -D…` / CMake; the **emit** column is
`plowc` env.

| you want | emit | build | notes |
|---|---|---|---|
| **Default** — validated, bf16 | `PLOW_UNISEG=1` | *(none)* | The shipped configuration. Everything below is a deviation from it. |
| **fp8 weights** — faster prefill GEMM | `PLOW_UNISEG=1 PLOW_W8A8=1` | `-DPLOW_NV_W8A8=1` | −48% GEMM, −30…34% prefill. Needs the fp8 weight twins. **Both flags or neither** — mismatch = `__trap()`. |
| **Long context, multi-user** — the fp8-KV path | `PLOW_UNISEG=1 PLOW_W8A8=1 PLOW_FP8_KV_FULL=1` | `-DPLOW_NV_W8A8=1 -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON` | Halves KV bytes (B=8 at 127k fits 32 GB). `FASTPF=ON` is what keeps prefill on the fast PIPE=1 arm — **−21% prefill at 67k** vs leaving it off. ⚠️ **Lossy, and it degrades with context**: at 7.8k every arm retrieves a needle; at 66.9k **only bf16 does**. Greedy also diverges ~21 tokens. Validate retrieval at *your* context length. |
| **Legacy all-layer fp8 KV** | `… PLOW_FP8_KV=1` (no `_FULL`) | `-DPLOW_FP8_KV=ON` (**FASTPF off**) | e4m3 on every layer. `FASTPF` must stay OFF — the hd256 fp8 prefill op traps under PIPE=1. Slower prefill; prefer the mixed row above. |

### Two build/emit traps

- **An emit flag and its `-D` must agree.** Emitting `PLOW_W8A8=1` packets against
  a cubin built *without* `-DPLOW_NV_W8A8=1` hits `default: __trap()` and every
  launch dies with `CUDA_ERROR_LAUNCH_FAILED`. The failure looks like a driver
  problem and is not.
- **Decode and prefill are separate objects.** `-DPLOW_NV_PREFILL=1` builds the
  prefill object rather than stacking those arms onto the decode megakernel's
  budget. A flag marked *prefill only* below is a no-op in the decode object, and
  vice versa.

### Defaults deliberately NOT flipped, and why

| flag | measured | why still off |
|---|---|---|
| `PLOW_FP8_KV_FASTPF` | **−21%** prefill at 67k, byte-identical token stream | Historically legality depended on the *packet* (valid only for MIXED fp8-KV packets, `PLOW_FP8_KV_FULL=1`; all-layer packets trapped under PIPE=1). **No longer true since PX-23** (`op_attention.cuh:2443`, `runtime/CMakeLists.txt:450-461`): both head dims now have a fast PIPE=1 arm (hd256 px23, hd512 px4/px8), so an ALL-LAYER e4m3 packet is served by the fast path end to end — live-verified `perf-data/sm120-iter1-fastpf-routing-2026-08-26.md` (2688-token prompt, all-layer fp8-KV + FASTPF=ON, no trap, correct output). Still off by default because a raw CMake build cannot see which packet it will load and the served nix package already defaults it ON (`flake.nix:292`) — **not** because all-layer packets are unsafe. |
| `PLOW_NV_PF_GEMV_HEAD` | −39% on prefill's `lm_head` | Traps on M≠1. Prefill emits `lm_head` at M=1 today, so *probably* safe to default, but not validated across every model family — and the failure is a hard launch failure. Worth ~0.5% at 127k. |
| `PLOW_NV_FA_FP8PV` | 1.40× on the hd512 flash op | Changes numerics, and only **+1.5%** end-to-end once `FASTPF` is on. |

Already defaulted **on** because they are bit-exact wins: `PGM_W8A8_LDS64` and
`PGM_SW8_V2` (px9, +2.2% weighted on the w8a8 GEMM).

### Object selection / model-family arms

These decide what is compiled in at all. The family arms buy **cubin size, smem
and stack frame** — not occupancy. Numbers below are `ptxas -v` on the megakernel
symbol, CUDA 13.0, and are **per-arch — they do not transfer**.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_PREFILL` | 0 | build the PREFILL object (`interp_sm120_pf`) instead of decode. |
| `PLOW_NV_GEMMA` | 0 | Gemma arms (hd512 full-attn, GF flash). Off ⇒ byte-identical to a non-Gemma build. |
| `PLOW_NV_HOPPER` | off | sm_90a wgmma GEMM + Hopper attention arms instead of the sm_120 ones. |
| `PLOW_NV_MLA` | **1** | MLA (DeepSeek/Kimi/GLM). `0` trims the decode cubin — sm_90a −43% / regs 208→188; sm_120a −33% / regs 241→224. Occupancy-neutral. Not compiled into prefill. |
| `PLOW_NV_MAMBA` | **1** | Mamba/Nemotron arms. Owns the *prefill* stack frame (sm_120a 1024→0 B). Costs 2 regs on the sm_120a prefill object to turn off. |
| `PLOW_NV_DSA` | **1** | DeepSeek sparse-attention arms. Owns the *decode* smem (2192→1168 B). Costs 9 regs on the sm_120a decode object to turn off. Not compiled into prefill. |
| `PLOW_NV_GF8_TWIN` | 0 | co-linkable GF=8 full-attn decode twin (234 vs 209 regs); host picks per model. |
| `PLOW_NV_SEG_GEMM` | 0 | lean GEMM-segment object targeting occupancy 2, separate from the register/smem-hungry flash object. |
| `PLOW_NV_SEG_GEMM_BN64` | off | that object at BN=64. PX-7: ~1.05× end-to-end, not the ~2× the occupancy argument implies. |
| `PLOW_MLA_PREFILL` | off | compile the MLA chunked-prefill ops (51/55); also read at emit for the tune census. |
| `PLOW_MOE_PREFILL` | off | grouped MoE prefill ops (83–87). Also an emit gate: on for MoE bf16 by default, `=0` opts the bf16 MoE-prefill path out. |
| `PLOW_MOE_PF_A4W4` | off | A4W4 grouped-expert GEMM body (ops 85/86, MXFP4 on both operands); set for K3 rows in CMake, required by the devgen manifest. |
| `PLOW_K3_DECODE_GROUPED` | off | Build-only K3 override for a B1 object that must serve grouped ladder packets. Adds the A4W4 expert body and capability marker without changing `PLOW_DECODE_BATCH`; required by the K3 MI325X rung-1 recipe. |
| `PLOW_DECODE_TIERS` (build_gfx942.sh) | unset = auto | Comma-separated narrow decode widths built as `lowrung<w>/` tier directories next to the wide object. UNSET builds every `w < PLOW_GEMV_MM` of 1/2/4/8 automatically; an explicit empty value builds none (required when a recipe drives its own `[[objects.lowrung]]`); plowrt discovers the tiers by layout. +24.9% tok/s at conc-1 on GLM-5.3 MI300X, neutral at C20. |
| `PLOW_MOE_ROUTER_SELECT` | =`PLOW_K3` | `1` = k parallel block-max router passes (K3's 896-expert / top-16); `0` = single all-pairs rank pass. |
| `PLOW_BUCKET_DECODE` / `PLOW_BUCKET_PREFILL` | decode=1 selects | which interp bucket the object serves (`PLOW_BUCKET_PREFILL` is derived as `!DECODE`); emitted into manifest `req` strings. |
| `PLOW_BUCKET_FLASH` | off | compile the standalone 4-wave flash-decode object. |
| `PLOW_GLOBAL_QUEUE` | 0 (build); on at runtime | build the `_gq` global-queue objects; also a runtime env selecting GQ vs static (see serving knobs). |
| `PLOW_GQ_BATCH` | 1 | packets claimed per fetch-add in the global queue. **Must stay 1.** |
| `PLOW_PACKET_HASH` | emitted | the manifest emits `#define PLOW_PACKET_HASH 0x…`; the loader refuses a packet whose hash does not match the object. |

### Precision

| flag | default | effect |
|---|---|---|
| `PLOW_NV_W8A8` | 0 | PX-2 native w8a8 fp8 mainloop (BK64 + `Swizzle<3,4,3>` + `mma.sync.m16n8k32`). −48% GEMM vs bf16, −30…34% end-to-end prefill. Needs the fp8 weight twins **and** the matching emit flag. |
| `PLOW_FP8_KV` | 0 | e4m3 KV cache, half the KV bytes; per-(token, kv_head) f32 dequant scale. **Lossy** — ~3–6% logit relL2, greedy diverges after ~21 tokens. Lifts the 31B batch cap (B=4 → 7–8). **Read `PLOW_FP8_KV_FASTPF` before using this** — on its own it silently costs the PIPE=1 prefill pipeline. |
| `PLOW_FP8_KV_FASTPF` | **OFF** (CMake default; **ON** in the served `plow-interp-sm120a` nix package, `flake.nix:292`) | ⚠️ **Without this, enabling fp8 KV makes prefill slower than not using it** (1670.3 ms vs bf16's 1315.6 ms at 7.8k). Turning on `PLOW_FP8_KV` alone forces prefill to `PLOW_NV_FA_PIPE=0` (fp8 dequants at the smem stage; cp.async cannot convert fp8 inline), losing the cp.async flash pipeline for prefill. `FASTPF=ON` keeps prefill on PIPE=1: **13843.3 → 10947.3 ms, −21% at 67k**. Since PX-23 (`op_attention.cuh:2443`) this covers **both** hd256 and hd512 — works for ALL-LAYER fp8-KV packets too, not just mixed (`PLOW_FP8_KV_FULL=1`); the old "hd512-only, traps otherwise" caveat is stale, see `runtime/CMakeLists.txt:450-461` and `perf-data/sm120-iter1-fastpf-routing-2026-08-26.md`. Decode is unaffected. |
| `PLOW_NV_FA_FP8MMA` | derived | feed the RAW e4m3 K tile to the mma — no dequant pass. Requires `PLOW_FP8_KV`. |
| `PLOW_NV_FP8_RB` | **1** | fp8 GEMV row-blocking. |
| `PLOW_FP8_FAST` | off | faster/looser fp8 conversion path. |

### Attention

`FA_PX4` and `FA_PIPE` carry the shipped long-context performance; the rest are
mostly A/B controls, several of which measured *negative* and are kept only so
nobody re-runs them.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_FA_PX4` | **1** | restructured hd512 full-layer flash (register softmax + 8-warp QK). −24% flash op, −16% end-to-end 128k prefill. |
| `PLOW_NV_FA_PIPE` | **1** | cp.async KV-stream pipeline. Bit-identical logits; −16%@4k → **−81%@128k** prefill. **Forced to 0 for the prefill objects when `PLOW_FP8_KV` is on unless `PLOW_FP8_KV_FASTPF=ON`** — the single easiest way to lose most of plow's long-context prefill performance without noticing. |
| `PLOW_NV_FA_FP8PV` | 0 | **px8/px12 — the largest single prefill lever in the campaign.** e4m3 P·V via 8-bit `ldmatrix.trans`. **1.18× on a 127k prefill end-to-end** (32.39 → 27.59 s), 1.40× on the flash op. **Unreachable without `PLOW_FP8_KV_FASTPF=ON`** (`op_attention.cuh` `#error`s without PIPE=1). Also needs `-DPLOW_FP8_KV=1`. sm_120a-only. ⚠️ **NOT parity-preserving: greedy diverges at completion token 28.** Run a retrieval test before shipping. |
| `PLOW_NV_FA_GF` | 4 | head-group fold factor for flash-decode. Correctness needs `gqa % GF == 0` (checked at dispatch). Register allocation is the worst case over instantiations, so do not add more. |
| `PLOW_NV_FA_GF_FULL` | 4 (via `build_sm120_cubin.sh`; `CMakeLists.txt` says 2) | **CONTESTED — do not change without an end-to-end measurement.** PX-11 measured `=8` at 1.52× on the flash-decode OP and recommended it; PX-15 then found `=4` wins **all 8 cells at ctx ≥ 8k** (−29% at 130k, B=1) via grid fill (`n_grp = 16/GF`, so `=8` leaves 2 groups for 170 SMs). **Trust the end-to-end result.** |
| `PLOW_FP8_LD16`, `PLOW_FP8_FAST` | unset | **UNVALIDATED END-TO-END.** PX-11: 1.61× on the flash-decode op with `GF_FULL=8`, bit-exact, register-neutral — but `GF_FULL=8` is itself contradicted end-to-end, and PX-15 could not measure these flags (**every fp8-KV block asset dies in prefill with `CUDA_ERROR_LAUNCH_FAILED`**, a live bug with a reproducer). Nothing in the tree sets either flag; nothing should until the crash is chased down. |
| `PLOW_NV_FA_TMA` | 0 | TMA (`cp.async.bulk`) KV staging — ~2× *slower* on sm_120. A/B control. |
| `PLOW_NV_FA_KUN` | **1** | K-stream pre-issue depth for flash-decode; 1 = original consume-immediately loop. |
| `PLOW_NV_FA_WPR` | 0 | warp-per-row score phase (vs one-row-at-a-time). |
| `PLOW_NV_FA_WPR_RB` | 1 | rows a warp carries concurrently in that phase. |
| `PLOW_NV_FA_QGLOB` | 0 | read Q from global instead of staging to smem (WPR path). ⚠️ **`FA_WPR=1` + `FA_QGLOB=1` silently CORRUPTS the fp8/SZ arms** (PX-11): those arms read a never-written `qsm`. Non-default; **not fixed**. |
| `PLOW_NV_FA_REDBOUND` | 0 | bound the softmax reductions to the tile's LIVE rows. |
| `PLOW_NV_FA_VDBUF` | 0 | V double-buffer. **MEASURED NEGATIVE**: wash at 32k/64k, **+2.2% slower at 128k** — the 128k full-attn flash is HBM-bound. |
| `PLOW_NV_FA_CORRSKIP` | 0 | fp8mma only — skip the softmax rescale when every lane's `corr` is exactly 1.0. Bitwise identical. |
| `PLOW_NV_KVBOUNDS` | 0 | per-batch KV bounds checking. |
| `PLOW_NV_FA256_BKV` / `PLOW_NV_FA512_BKV` | 32 / 16 | KV rows staged per flash-**prefill** tile, per head dim. The trade is smem footprint vs staging granularity. **64 / 32 are the sm_90a measured optima** and are what `build_sm90a_cubin.sh` is driven with; they are not validated as sm_120a defaults. |
| `PLOW_FLASH_HD128` | off | enable the fused-write path in inline flash for D=128 (Llama/Qwen), avoiding register spill on the 8-wave interpreter. |
| `PLOW_FA_GF_FULL` | 2 | **AMD** flash-decode GQA fusion factor on full-attention layers (paired env+define; the NVIDIA analogue is `PLOW_NV_FA_GF_FULL`, which it must agree with or kernel/packet disagree). |

### Counter-gate, collective, MLA & geometry (`-D`)

Mostly AMD (gfx950) codegen/geometry arms. The gate and collective families are
dominated by **measurement instruments** — several are numerically wrong or admit
a data race by design and must never be in a shipped build; those are listed at
the end, not tabled.

| flag | default | effect |
|---|---|---|
| `PLOW_GATE_HIER` | 0 | gfx950 two-level counter-gate rendezvous. The CMake option is default-off and applies only to decode global-queue objects; it requires `PLOW_HSACO_GQ=ON` and `PLOW_L2_PLACE_DISPATCH=ON`. Passing it through global `PLOW_HSACO_EXTRA_DEFINES` is rejected. The gfx942 shell build's existing default is unchanged. |
| `PLOW_GATE_SC1` | 0 | device-scope (not system-scope) activation stores so the release fence can be elided; the counter-gate carries the ordering. |
| `PLOW_MLA_FOLD_MAP` / `_UN` / `_VEC` / `_VT` | 0 | fold the MLA up-projection map / output un-projection / V-cache load / V^T transpose into the adjacent kernel to save a launch + round-trip. |
| `PLOW_MLA_FOLD_TB_FLASH` | 0 | build `interp_flash_*` with the token-blocked `MlaMergeFold` arm (`PLOW_MLA_FOLD_TB`, default 8 and already on for `interp_prefill_*`). GLM-5.3's sparse 8192 chunk dispatches its fold from the FLASH object, so without this the arm is unreachable on the shipped recipe. Opt-in until the retrieval screen runs on this object. |
| `PLOW_MLA_PF_MFMA` | 0 | MLA prefill uses MFMA matrix-core instructions for QK/PV instead of the vector-FMA fallback. |
| `PLOW_MLA_PF_WPM` | numeric | MLA-prefill waves-per-M-tile, clamped by `min(PLOW_WAVES, PLOW_MLA_PF_WPM)`. |
| `PLOW_XR_CUS` | 32 | **emit** — cap XReduce participant CUs (clamped 1..n_cu); a TP8 NUMA lever cutting L2 invalidates from idle WGs. |
| `PLOW_XR2_GATHER` | 1 | **emit** — use the two-shot reduce-scatter/all-gather path for complete folded-gather collectives when `row_w = n_gpu*gcols`; set `0` for the one-shot rollback. |
| `PLOW_NO_XREDUCE` | unset | **emit** — disable all XReduce all-reduce collectives (diagnostic; numerically wrong). |
| `PLOW_WG_WAVES` (alias `PLOW_WAVES`) | 8 | waves per AMD workgroup (8×64 = 512 threads); feeds reduction/tiling geometry. |
| `PLOW_ATTNRES_MAXB` | 16 | compile-time hard cap on K3 attention-residual batch (`nb`); over it traps. |
| `PLOW_ATTNRES_RG` | 9 | rows per sweep of the K3 hidden axis (tuned so one sweep covers `nb_max=8`). |

**Measurement/diagnostic instruments — never ship** (numerically wrong or admit a
data race; used only to price a protocol cost): `PLOW_GATE_HIER_CEIL`,
`PLOW_GATE_NOINV`, `PLOW_GATE_RELAXSIG`, `PLOW_GATE_SC1_KEEPREL`, `PLOW_XR_ACQ_N`,
`PLOW_XR_NOSIG`, `PLOW_XR_NOWAIT`, `PLOW_XR_NOWAIT_RS`, `PLOW_XR_SHUFFLE`,
`PLOW_ACT_NT`, `PLOW_ACT_SCOPE_AGENT`, `PLOW_ICACHE_INV_PROBE`,
`PLOW_FLASH_MERGE_DSPLIT` (measured dead), `PLOW_NSTAGE` (`experiments/` only).

### GEMM / GEMV

Hopper Gemma CMake builds select `PLOW_NV_SEG_OCC1=1` for ordinary W8A8 GEMM
segments and packed BF16-KV GEMM segments without W8A8. This removes the
128-register cap. The packed BF16 H100 qualification lives in the campaign's
`perf-data` readiness report, which is kept out of source control. Rebuild with
`PLOW_EXTRA_DEFINES="-DPLOW_NV_SEG_OCC1=0"` to restore that cap. This build
default does not by itself enable packed serving; the CUDA unified token-batch
executor is selected at runtime by `PLOW_TOKEN_BATCH` (default on) when the
packet carries packed-prefill metadata.

| flag | default | effect |
|---|---|---|
| `PGM_BM` | 128 | GEMM M-tile height (`BM/WARPS_M` must be ×16). Overridable but no shipped alternate object. |
| `PGM_BN` | 128 | GEMM N-tile. `64` shrinks the plain arena to 45 KiB so the occ-2 segment object fits (driven to 64 by `PLOW_NV_SEG_GEMM_BN64`). |
| `PGM_BK` | 32 | GEMM K-tile depth (bf16 elts/step). Fixed, not overridable. |
| `PGM_STAGES` | 3 | GEMM cp.async pipeline depth. **px9 measured 3→6 = 0%** — the mainloop is not latency-bound. |
| `GV_UNROLL` | 8 (NV) / 11 (AMD); 14 for K3 | dense bf16 GEMV inner-K unroll = 128-bit weight vectors prefetched before consumed (memory-level parallelism). Swept by `tunedb`, per-arch in CMake. Bit-exact. |
| `GV_UNROLL_GLU` | 4 (NV) / 6 (AMD) | same unroll for the bf16 GLU/SwiGLU-fused GEMV. Also swept. |
| `GV_UN16` / `GV_UN_GLU16` | 4 / 2 | inner-K unroll for the **MM=16** decode rung specifically (`GV_UNROLL` covers the base rungs). Needs `GV_MM_MAX=16` to be reachable at all. Worth ~2%; `=8` measured best on sm_120a. |
| `GV_UN32` / `GV_UN_GLU32` | 2 / 1 | same for the MM=32 rung. |
| `GV_UNROLL_FP8` / `GV_UNROLL_GLU_FP8` | = the bf16 twins | rung unroll on the fp8 GEMV arms. The optimum is **precision-dependent** — sweep at the precision you ship, do not inherit the bf16 winner. |
| `GV_MOE_RB` / `GV_MOE_RB_DN` / `GV_MOE_UN` | 2 / 2 / 2 | MoE GEMV output-channels-per-warp (main / `down` arm) and inner unroll. Shape-dependent; the source notes the previous optimum stopped being one once neighbouring arms moved. |
| `PGM_GLU_STAGES` | 2 | same for the fused GLU arm (kept shallower to fit the 100 KiB dynamic-smem cap). |
| `PGM_W8A8_LDS64` | **1** | **px9** — read the fp8 fragment as one `uint2`. **+6.5% on plain w8a8, 0% on GLU, +2.2% weighted.** Bit-exact. |
| `PGM_SW8_V2` | **1** | **px9** — `Swizzle<2,4,2>` matched to the ACTUAL 64-byte fp8 row. +0.5% on top of LDS64. |
| `PGM_SW8_OFF` | unset | A/B control: make `pgm_sw8` the identity. px9: removing the swizzle is +3.1% cycles/QMMA — a diagnostic, not a win. |
| `PLOW_NV_GEMV_RB` | 0 | MoE GEMV row-blocking. Off keeps every sm_120 object byte-identical; **the sm_90a build sets it to 1**. |
| `PLOW_NV_PF_GEMV_HEAD` | 0 | run prefill's `lm_head` on the M=1 GEMV arm. **1.991 → 1.213 ms, −39%.** Prefill only; traps on M≠1. |
| `GV_MM_MAX` | **8** | Widest `gemv_*_rows<MM>` rung for batched decode: batch `B` costs `ceil(B/GV_MM_MAX)` weight passes. **Match it to the batch you actually serve — mismatched, expensive both directions.** `=16` costs 1.1% at B=1 and 17% at B=8 to buy 34% at B=16. PX-10: an asset built `=16` and served at B=8 loses 19.4% at 131k. Pin `=16` only if you pin B≥16. |
| `PLOW_NV_GEMV_LS` | 0 | GEMV row-blocking. Wins in isolation (qkv 1.43×) but **loses in the megakernel**. Compiled out, intact. |
| `PLOW_NV_GEMV_NOSTAGE` | 0 | skip GEMV smem staging. |
| `PLOW_NV_GEMV_STAGE_MINROWS` | 16 | row threshold below which staging is skipped. |
| `PLOW_NV_RB_QKV`, `PLOW_NV_RB_LMHEAD`, `PLOW_NV_RB_GEMV` | 0 | per-op row-blocking A/B controls. |
| `PLOW_NV_SZ` | 0 | **experimental** lossless bf16 weight decompression (SplitZip GEMV twins). Bit-exact, measured non-viable; kept as an A/B reference. (`PLOW_NV_ZG` does not exist.) |

### MoE

| flag | default | effect |
|---|---|---|
| `PLOW_MOE_XN_BF16` | 0 | bf16 expert-N staging buffer. |
| `PLOW_MOE_XN_MAX` | 2816 | expert-N staging cap. |
| `PLOW_MOE_DOWN_SG` | 4 | subgroup count for the expert `down` arm. |
| `PLOW_MOE_DOWN_LANESPLIT`, `PLOW_MOE_DOWN_STAGE_FU` | 0 | `down` lane-split / staged fixups. |
| `PLOW_MOE_ROUTER_WIDE` | 0 | wide router arm. |
| `PLOW_MOE_COMBINE_ALLBLK` | 0 | all-block combine. |
| `PLOW_HSACO_EXTRA_DEFINES` | unset | `scripts/build_gfx942.sh`: raw `-D` appended to every row and recorded in `build_defines.json` (so `asm_audit.py --contract` sees the axis). For opt-in kernel arms whose header default is the shipped body. Refuses the tile / wave / decode-batch axes, which have their own variables and are cross-checked against the packet. |
| `PLOW_COMBINE_VEC` / `PLOW_COMBINE_VEC_U` | **1 (gfx942 prefill/flash objects)** / 2 | AMD `d_moe_combine_pf` 8-wide arm (16 B loads, `_U` iterations in flight) for the `k == 1` combine every native-MoE / `PLOW_MOE_PF_DET` blob emits. Bit-identical to the scalar loop (standalone harness, production `-D` sets, 8- and 4-wave geometries, ragged cases included); −25 ms per GLM-5.3 8192 chunk (31.7 → 6.7 ms), served +1.9%. **Default ON** in `build_gfx942.sh` for the prefill and flash objects; rollback `PLOW_COMBINE_VEC=0`. Decode/mixed/packed objects keep the scalar body. |
| `PLOW_RN_ROWS` | **2 (gfx942 prefill/flash objects)**, header default 1 | AMD `d_rmsnorm` multi-row arm: R rows' loads issued before any row is reduced (prefill norms hand each workgroup ~27 rows and paid one HBM round trip per row). Same per-thread element map and reduction tree; bit-identical by the standalone harness. **Default ON** (R=2) in `build_gfx942.sh` for the prefill and flash objects; rollback `PLOW_RN_ROWS=0`. |
| `PLOW_RESID_U` | **4 (gfx942 prefill/flash objects)**, header default 1 | AMD `d_residual` unroll: U iterations of loads in flight. Bit-identical by the standalone harness. **Default ON** (U=4) in `build_gfx942.sh` for the prefill and flash objects; rollback `PLOW_RESID_U=0`. |

### Scheduling, sync, occupancy

| flag | default | effect |
|---|---|---|
| `PLOW_NV_SCHED` | **1** | global-queue scheduler; `0` = static per-block streams (build-time A/B). Counter protocol byte-identical across both. |
| `PLOW_NV_SEGMENTS` | 0 | host relaunches once per segment (the AMD model) instead of one cooperative launch. |
| `PLOW_NV_PTXSYNC` | **1** | inline-PTX counter gate (`red` instead of a result-bus round trip). |
| `PLOW_NV_GATE_SLEEP` | 64 | backoff (ns) inside the counter-gate poll; `0` spins flat out. |
| `PLOW_NV_LEAN_DECODE` | 0 | drop arms owning the decode object's 208-reg / 1-blk-SM ceiling so ptxas + `PLOW_NV_FORCE_MINBLK` can reach 2–3 blk/SM. |
| `PLOW_NV_FORCE_MINBLK` | off | force a `__launch_bounds__` min-blocks-per-SM. |
| `PLOW_NV_THREADS` | 256 | NVIDIA block size (`op_attention.cuh`). Raising it is the precondition for BQ=64 flash tiling. Distinct from the AMD `PLOW_THREADS` (512 = 8 waves × 64); not a rename. |
| `PLOW_NV_EMBED_SMEM` | 0 | embed the object's smem requirement so `serve` reads it instead of guessing the GF=2 default. |
| `PLOW_L2_PLACE_DISPATCH` | off | L2 placement dispatch. Vendor-neutral — GPC on NVIDIA, XCD on AMD. |

### Measurement-only — never ship a build with these

All produce wrong logits by construction.

| flag | default | effect |
|---|---|---|
| `PLOW_NV_SKELETON` | 0 | run gates + signals with no op bodies: the interpreter's dispatch floor. Garbage logits. |
| `PLOW_NV_SKEL_PAD` | 160 | padding for that skeleton. |
| `PLOW_NV_ABLATE_LO`, `PLOW_NV_ABLATE_HI` | 0 | 128-bit opcode mask — skip those ops' BODIES, keep every gate/signal. Garbage logits. |
| `PLOW_NV_FA_FP8ABL` | 0 | flash fp8 ablation bitmask. **Never set on a shipped build.** |
| `PLOW_NV_TRACE` | 0 | per-op `gate`/`body`/`signal` cycle trace. **Read the SHAPE, not the absolute total.** |

Harness-only (not the served cubins): `PLOW_SM120_SMS` (188) and
`PLOW_SMP_THREADS` (256).

### sm_90a object selection (`PLOW_BUILD_*`) — **sm_90a only**

`scripts/build_sm90a_cubin.sh` compiles a **five-object** prefill stack rather
than one megakernel, and these envs — read by the build script, not passed as
raw `-D` — decide which objects it emits and which arms each carries. They have
no effect on any other `--arch`; the sm_120 script builds one decode + one
prefill object and ignores them.

The reason the stack exists is the register-allocation coupling that the section
header above describes: a heavyweight wgmma body loses probe-grade allocation
when compiled into the wide-armed interpreter TU, so on sm_90a the tuning axis
is **which object gets built**, not which tile a macro selects.

| build env | object it shapes | meaning |
|---|---|---|
| `PLOW_BUILD_SEG` | `_pfseg` + `_pfgemm` | build the segmented pair at all. Packets must be emitted **without** `PLOW_UNISEG`. |
| `PLOW_BUILD_FATLITE` | `_pfseg` | the fat object arm-stripped of flash → 128 regs, occupancy 2. |
| `PLOW_BUILD_GEMM_WS384` | `_pfgemm` | 384-thread producer/consumer GEMM; carries **both** precisions' n256 bodies, so one lean object serves bf16 and fp8. |
| `PLOW_BUILD_FA512` + `PLOW_BUILD_FA_WG` + `PLOW_BUILD_FA_HD256` | `_pffa` | the dedicated flash object: wgmma arms, hd512 and hd256. `--pf-seg-fa512 all` **requires** `FA_HD256=1` (the loader refuses the mismatch). |
| `PLOW_BUILD_TMA_GEMM` / `PLOW_BUILD_W8A8` | all | TMA GEMM bodies / fp8 w8a8 arms. Drop `W8A8` for bf16-only cubins. |

The canonical build measured in the GH200 campaign:

```bash
PLOW_EXTRA_DEFINES="-DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32" \
PLOW_BUILD_TMA_GEMM=1 PLOW_BUILD_W8A8=1 PLOW_BUILD_SEG=1 \
PLOW_BUILD_FATLITE=1 PLOW_BUILD_GEMM_WS384=1 \
PLOW_BUILD_FA512=1 PLOW_BUILD_FA_WG=1 PLOW_BUILD_FA_HD256=1 \
scripts/build_sm90a_cubin.sh <out-dir>/interp_sm90a.cubin
```

Its serve-side and emit-side counterparts are in
[Segmented prefill (sm_90a / GH200)](#segmented-prefill-sm_90a--gh200) below;
the emit knob and the serve knob **must pair** (`PLOW_SEG_PURE_GEMM` ↔
`--pf-seg-pure`, `PLOW_SEG_FA512` ↔ `--pf-seg-fa512`) — the classing decides
which object a packet lands on and a mismatched object `__trap()`s by design.

Ablation-only `PLOW_BUILD_*` switches, default off, leave them off unless
reproducing a specific finding: `FA_ROPE`, `FA_WGITEM`, `FP8KV`, `GEMM_OCC1`,
`GEMM_ONLY`, `GEMM_UNI256`, `GEMV_HEAD`, `SEG_NOGLU`, `SEG_WS`, `SEG_WS_ENTRY`,
`WS_BN256`.

Two sm_90a-only `-D` tuning knobs on the GEMM side:

| flag | default | effect |
|---|---|---|
| `PGM90_TILE_BAND` | 16 | band rasterization width for the sm_90a GEMM — how many M-tiles share a B-tile in L2 before the walk advances. |
| `PGM90_UNI256_NS` | 4 | TMA ring depth of the n256 body. bf16 256-byte k-stages at `NS=2` measured **−44 ms** (ring starvation); do not lower it casually. |

The runtime companion of an sm_90a decode build is `PLOW_NS_FULL_ABS`, and its
value is a **cliff, not a slope** — see the header of `build_sm90a_cubin.sh`,
which derives it as `n_cu / gcd(n_grp, n_cu)` and records what the neighbouring
values cost.

---

## CPU runtime knobs (`plowrt --cpu-*`)

Only present in a build carrying the `cpu` feature. Every one is a CLI flag with an
environment twin, and the CLI wins. Full prose, plus how to compile a CPU-loadable
bundle in the first place, in [CPU execution](runtime/cpu.md).

| flag | env | default | effect |
|---|---|---|---|
| `--cpu-threads N` | `PLOW_CPU_THREADS` | 0 | Persistent kernel workers. `0` = model-dependent width (physical cores for MoE, logical CPUs for dense decode). Independent of the packet's `--n-cu`: the pool maps any thread count onto the virtual executors, so one bundle serves any core count. |
| `--cpu-numa MODE` | `PLOW_CPU_NUMA` | `auto` | `auto` interleaves large tensors across allowed nodes (best effort); `off` keeps OS/`numactl` policy; `0,1` requires successful binding and rejects unavailable nodes. Topology honours cpusets and `taskset`. |
| `--cpu-isa TIER` | `PLOW_CPU_ISA` | `auto` | Tier ceiling `scalar` / `avx512` / `amx`. Never activates above what cpuid and OS register state permit, so this only ever narrows. |
| `--cpu-huge-pages=B` | `PLOW_CPU_HUGE_PAGES` | unset | Override THP *advice* (default: ordinary pages when interleaved, huge-page advice for single-node/OS placement). Advice only — not the system THP setting. |
| `--cpu-spin-us N` | `PLOW_CPU_SPIN_US` | 2000 | Spin budget (µs) before a blocked worker yields and parks. Decode packets are 100–500 µs apart; parking every gap measured **+17% TPOT** at 50 µs vs 1000. |
| `--cpu-prefill-chunk N` | `PLOW_CPU_PF_CHUNK` | 0 (off) | Largest prefill chunk (rows) a tick may run while other slots decode. Measured **negative at concurrency ≥ 4** — the threads are throughput-bound, not stall-bound — so it stays off. |
| `--cpu-mxfp4-dir DIR` | `PLOW_MXFP4_DIR` | unset | MXFP4 weight twin (`mxfp4/<name>` + `_scale` E8M0 rows; `perf-data/tools/quantize_mxfp4.py`). |
| `--fp8-dir DIR` | `PLOW_FP8_DIR` | unset | fp8 weight twin. Runtime-wide, but this is how a CPU bundle gets W8A16/W8A8 weights. |
| `--cpu-global-queue=B` | `PLOW_CPU_GQ` | off | Global op-major work queue (windowed per segment and L2 domain, with stealing) instead of static per-cu streams. **~2x slower** on the EPYC 9654; kept for A/B. |
| `--cpu-l2-place=B` | `PLOW_CPU_L2_PLACE` | off | Place executors by the packet's L2 locality domains instead of `cu % nodes`. **1.5x slower**, never faster (placement report in the `perf-data/cpu-numa-placement` campaign, kept out of source control). Inert without domains in the blob; a balance guard declines a losing plan even when on. |

The last two are off because they were *measured* worse, not because they are
unfinished — neither changes what is computed, so both are safe to flip for an A/B
on another host.

Compile-side, there is no CPU emit target: `plowc` emits for a device target and the
CPU interprets that packet. Use an **NVIDIA** `--gpu` (a gfx942/gfx950 packet is
rejected at load), and set `--n-cu` explicitly — it is the virtual executor count,
capped at 256, and need not match any thread count.

## Asset distribution (`plowrt pull` / `load` / `serve --model`)

Two knobs, both with CLI twins, both consumed only by the distribution
subcommands. `serve` reads the local store and never the network; see
[`DISTRIBUTION.md`](DISTRIBUTION.md).

| var | default | effect |
|---|---|---|
| `PLOW_HOME` / `--plow-home` | `$HOME/.plow` | Root of the local asset store: `blobs/sha256/` (content-addressed), `refs/` (pins), `bundles/` (materialized directories `--assets` receives), `checkpoints/` (the farms `prepare` builds). |
| `PLOW_REGISTRY` / `--registry` | `dist.infervisor.ai` | Where a bare model reference resolves. A `file://` URL or an absolute path selects a local mirror and needs no HTTP client, which is how an air-gapped host and every test fetch. |

`PLOW_CHECKPOINT` / `--rt-checkpoint` (below) doubles as `load`'s default
checkpoint location, because the weights a bundle needs are the same weights
`serve` binds.

## Serving / runtime knobs (`plowrt` env)

| var | default | effect |
|---|---|---|
| `PLOW_FUSION=1` / `--fusion` | off | AMD TP1 dense BF16 runtime prefill/decode fusion. Derives schedules from ordinary prefill/decode programs at model load and packs current rows per launch. Requires the normal gfx942 `interp_mixed_gq.elf`; unsupported programs or objects fall back with a warning. No fusion compiler flags or asset metadata. `--fusion=false` disables it. Multistep decode remains a separate setting. |
| `PLOW_PF_BATCH` | unset: AMD on, CUDA off | Cross-request prefill policy. **AMD: on by default** — mid-prefill requests whose next chunks are packable (non-final, no prefix-snapshot boundary, a capable packed sibling wide enough for the sum) share one rung each tick; `=0` is the rollback; an explicit `=1` additionally rotates isolated admission across slots instead of the default oldest-first (vLLM's running-queue order). CUDA packs waiting chunks into one launch only when set. The experimental combination with explicit `PLOW_VMM_PREFIX=1` requires compatible packed metadata and a prefix layout; it reserves prompt/output KV before launch and queues requests that cannot yet fit. A compatible BF16 prefill tail and RowGather object enable compact sampling after the final prompt row, preserving the ordinary GEMM head. Unsupported tails retain the legacy final-token decode route, whose output can differ from ordinary prefill. AMD TP1 and TP co-pack compatible initialized, non-final dense chunks into a larger compiled rung. Routed objects must carry the dense consumer marker; unsupported programs use isolated scheduling. Initial admission, final sampling and prefix snapshots remain isolated. See the Gemma MI300X qualification for measured results. |
| `PLOW_PF_INTERLEAVE=N` | unset: CUDA 2048, AMD widest rung | **The per-tick step token budget (vLLM's `max_num_batched_tokens`) — the default path, not a `PLOW_PF_BATCH` knob.** Once any slot is decoding, a tick admits at most `N` prefill rows, then runs decode. **AMD's default is the widest compiled prefill rung** (`0` = uncapped means the same): one widest chunk per tick, oldest request first; the tick loop admits further launches only while the budget holds a whole planned chunk, and never re-plans a request narrower to fit a remainder. The previous 2048 default capped every 8192-row chunk to 2048 rows as soon as anything decoded — on GLM-5.3 TP8 that also swapped the sparse (DSA) 8192 rung for dense 2048 attention over the full context (`docs/amd/tp-bringup-mi300x.md`, "Lifting the cap"). Set `N` to clamp; it can only clamp below the emitted ladder. |
| `PLOW_PF_CHUNK=C` | 0 (off) | Per-request prefill chunk-row cap for CUDA and AMD TP1/TP — the explicit override of the step budget's plan. AMD selects compiled packet rungs at or below `C`; compatible initialized middle chunks share a larger rung (packing is on by default on AMD), and with `C` below the budget one tick admits `budget/C` chunks, oldest request first. Gemma 4 31B BF16 on MI300X is qualified at 512 rows; performance depends on the workload. Off preserves the existing plan. **On AMD this is a PRECONDITION for co-packing, not a companion knob:** unset, every prompt up to the widest bucket plans as one chunk, the final chunk is always isolated, and two widest-rung chunks cannot share one rung — so `PLOW_PF_BATCH=1` packs nothing at any prompt length. A prompt needs `> 2C` rows before a middle chunk survives to meet a second cursor. |
| `PLOW_PF_CHUNK_COST=R` | 512 | cost of ONE prefill launch in padded-row equivalents (`rows + R × launches`). A launch re-streams every layer's weights: measured `ttft_ms = 0.112·rows + 60.1·chunks`, i.e. **60 ms ≈ 537 rows**. `0` = pure-minimum-padding. |
| `PLOW_PF_COVER=1` | off | restore the covering-bucket prefill policy (exact-parity A/B vs the cost-aware default). |
| `PLOW_PF_DEFER_DECODE=1` | off | **CUDA + AMD TP throughput mode — trades streaming latency for aggregate tok/s.** While pending prefill remains, skips decode so later decode ticks run at full batch. A completed prefill may emit its first token, but no request advances decode until admitted prefill drains. The 8×127k **+7.1% out tok/s** result was measured on CUDA; AMD is unmeasured. Wrong as an interactive default. |
| `PLOW_PF_PACKLOG=1` | off | per-launch pack diagnostics. |
| `PLOW_PF_NO_CHUNK=1` | off | restore whole-prompt-per-tick (disable chunked prefill). |
| `PLOW_PF_NO_INTERLEAVE=1` | off | restore a prefill-only tick (disable prefill/decode interleave). |
| `PLOW_VMM_PREFIX=0/1` | auto | VMM-backed KV prefix cache. Automatic on eligible Hopper hybrid BF16-KV packets (HD256 sliding, HD512 full, window1024), including Gemma 4 31B BF16 and FP8 weights. Full blocks are shared; partial full-KV and sliding snapshots allow reuse at 32-token boundaries. `0` disables; `1` explicitly requests other supported layouts. Automatic selection excludes TP, recurrent state, mixed/prepared decode and explicit packed-prefill/live-KV modes. It takes precedence over packed-prefill metadata alone. |
| `PLOW_VMM_LIVE=1` / `--vmm-live` | packet-selected for packed prefill with full-attention KV | Grow packet-described full-attention KV backing with the live frontier, without prefix reuse. Explicit enable remains available for legacy packets. Multistep maps the selected rung's full write frontier. Model admission counts startup backing and the retained block pool; virtual context capacity is not charged as resident memory. |
| `PLOW_VMM_LIVE_RINGS=1` / `--vmm-live-rings` | off | Retain sliding-ring backing on first use. Requires live KV. Sub-granularity logical slots share aligned physical mappings while preserving the packet's logical batch stride. |
| `PLOW_VMM_BLOCK_MIB=M` | 2 | VMM sharing block size. 2 MiB ≈ 4096 tokens at hd256 bf16. Raise (e.g. 64) for 128k-dedup work. |
| `PLOW_VMM_CACHE_MEMORY_UTILIZATION=F` | 0.05 | Soft cap on retained prefix blocks plus boundary snapshots (CUDA VMM pool, AMD slot snapshots) as a fraction 0..1 of the device's memory — the unit of vLLM's `--gpu-memory-utilization`, scoped to the prefix cache: 4 GiB on an 80 GiB H100, 9.6 GiB on a 192 GiB MI300X. Active pins and one recently reused snapshot can temporarily exceed it while radix leases are held; OOM reclamation can still evict that snapshot. `0` uses OOM-driven eviction only. Disable reuse with `PLOW_VMM_PREFIX=0` / `PLOW_PREFIX_CACHE=0`. |
| `PLOW_VMM_CACHE_MIB=M` | unset | Explicit cap in MiB (vLLM's `--kv-cache-memory-bytes` role); overrides `PLOW_VMM_CACHE_MEMORY_UTILIZATION`. `0` = OOM-driven eviction only. When the backend cannot report the device size the cap falls back to 4096 MiB. |
| `PLOW_VMM_KV=1` | off | **AMD** — VMM-backed KV on ROCr (`hsa_amd_vmem_*`); warns and falls back if the platform can't support it. |
| `PLOW_PREFIX_CACHE=0/1` | on | Prefix reuse on compatible AMD and NVIDIA assets: the VMM prefix pool on Hopper, slot-local prompt snapshots on AMD (single-GPU and TP). `0` disables both. Default-on since 3ca64e93; the isolated A/B against `0` on H100 and MI300X is still owed. |
| `PLOW_TOKEN_BATCH=0/1` | on | Select the unified token-batch executor (decode and prefill rows in one packed launch) when the backend, model and object support it; unsupported configurations use ordinary execution and `--fusion` takes precedence. AMD TP falls back today. Default-on since 33a5b7bf; A/B against `0` pending. |
| `PLOW_MLA_PF_AITER=0/1` | off | AMD: route isolated GLM sparse MLA prefill boundaries through the qualified gfx942 AITER assembly object (pinned SHA, 320-B ABI). −6.6% prefill time at 70k; 18/18 retrieval cases. |
| `PLOW_MOE_AITER_TILE64=0/1` | auto (on when the object is present) | AMD: native GLM MoE prefill rows ≥ 1024 use AITER's 64-row persistent fmoe tile (`psx_64x256.co`, pinned SHA), the kernel its GLM-5 gfx942 tuning table selects from 2048 tokens; below that the 32x256 object stays. Unset = on when the object dir has the pinned object and a tile64-marked adapter (`scripts/build_moe_aiter.sh OUT 32x256.co flat.co psx_64x256.co`), otherwise off with an info line; `=1` refuses to load without them. Qualified GLM-5.3 TP8 C20/70k: 49.79 → 50.57 tok/s (+1.6%), −0.8 ms per MoE layer at 8192 rows, 18/18 retrieval. |
| `PLOW_MOE_AITER_XCD=0/1` | auto (on when the adapter has the field) | AMD: launch the sorted AITER MoE blocks of the `ps_32x256` object in XCD-swizzled order — the 8 consecutive blocks of one expert sit 8 positions apart, so on gfx942 (workgroup w → XCD w mod 8) they run concurrently on one XCD and its L2 streams the expert's 4.7 MB of weights once instead of 8 times. Pure reorder of the same blocks (only the BF16 atomic accumulation order can differ). The 64-row `psx` object remaps workgroups to XCDs itself and is never swizzled, so with the 64-row tile on (default) this only touches rows < 1024. Single MI300X, random top-8: fmoe launch at 8192 rows 2628 → 1500 µs (−43 %), 2048 rows 851 → 547 µs. Qualified GLM-5.3 TP8 C20/70k with the 32-row object on every rung (`TILE64=0`, same binary): 49.89 → 50.54 tok/s (+1.3 %), TTFT med 100.8 → 97.3 s, 18/18 retrieval. Unset = on when the adapter carries `plow_moe_aiter_swizzle_abi_1` (`scripts/build_moe_aiter.sh`), otherwise off with an info line; `=1` refuses to load without it. |
| `PLOW_AMD_TAIL_SPARSE_CTX=<rows>` / `--amd-tail-sparse-ctx` | unset (off) | AMD: a request's DENSE final chunk whose prior context (prefix-cache resume + preceding chunks) is at least this many rows is planned into the sparse (DSA) prefill bucket instead of the smallest bucket that holds it; the chunk keeps its real row count. Measured GLM-5.3 TP8 at 65k prior: a 464-row tail in the dense 512 bucket = 1.9 s GPU, vs ≤0.3 s attention in the sparse bucket. No effect on packets without a sparse bucket. |
| `PLOW_NV_SCHED=1` | **on** | global-queue interpreter scheduler; the static per-block-stream path is the build-time A/B. |
| `PLOW_GLOBAL_QUEUE=0` | on | force the static per-block-stream scheduler (AMD runtime read; build-time A/B otherwise). |
| `PLOW_STATIC=both\|decode\|prefill` (`--amd-static`) | unset | force the static scheduler for both phases (`1`/`true` = `both`), decode only or prefill only; unset keeps the global queue where the blob carries its appendix. |
| `PLOW_SEG_WINDOW` | on | AMD segment enqueue/drain windowing (A/B; `=0` off). |
| `PLOW_MULTISTEP=K` / `--multistep K` | 8 (K∈[2,64]) | Bounded multi-step decode. CUDA can execute up to K steps per host synchronization; AMD dispatches and drains each step, captures tokens on device, then reads the capture once per quantum. CUDA device-loop execution needs dynamic KV-row addressing and the sampler object. It uses the packet's smallest eligible decode rung and supports live VMM. Each quantum is capped by remaining output and context budgets and the mux's batch-dependent limit: 4 steps for 1–2 live requests, 2 for 3–8, and 1 above 8. Disabling mux multistep also selects individual steps. Requests with stochastic sampling, repetition penalties, or logit bias use individual steps; unmodified greedy requests stream up to K tokens after each device quantum. `0`/`1` opts out. Decode objects/roles, context packets, recurrent decode, and cuBLASLt segments force single-step and log the decision. |
| `PLOW_LAUNCH_ROWS=N` | `LAUNCH_ROWS` | override the prefill pad/launch-rows tradeoff. |
| `PLOW_PREFETCH=N` | 256 | checkpoint prefetch depth in tensors. `PLOW_PREFETCH_THREADS=N` (16) sets prefetch threads/rank; `0` disables prefetch. |
| `PLOW_WEIGHT_SLAB` | on | single-allocation weight slab; `=0` turns it off (both backends). |
| `PLOW_UPLOAD_SLOTS=N` | 4 | AMD upload-ring pipeline depth; `1` = pre-pipeline one-slab shape. |
| `PLOW_SHARE_CKPT` | on | shared (vs per-rank) checkpoint mapping across TP ranks; `=0` restores per-rank. |
| `PLOW_VRAM_BUDGET_MIB=M` | unset | CUDA: cap each device group ModelManager VRAM budget (MiB). |
| `PLOW_WEIGHT_VMM` (`--weight-vmm`) | unset = CUDA on, AMD off | VMM (reserve+map) weight slab on either vendor; `=0` falls back to one flat allocation, `=1` opts AMD in. One knob for both backends. |
| `PLOW_SLAB_KEEP` | multi-model on | park evicted models' 256 MiB slab chunks in a per-device pool for the next load; `=0` releases them (`=1` forces on for single-model). |
| `PLOW_KV_POOL_MIB=N` | 512 | per-engine KV physical-block reuse pool cap (MiB); `0` disables pooling. |
| `PLOW_KV_MAP_AHEAD=0/1` (`--amd-kv-map-ahead`) | on | **AMD TP** — between a prefill chunk's enqueue and its drain, map the VMM KV block holding the row after the chunk's last written row on every rank (`AmdEngine::prefill_map_ahead`), so the decode that follows the chunk in the same tick finds its `frontier + 1` already mapped instead of mapping 99 blocks x 8 ranks synchronously (~40 ms at rung 20, measured). Maps exactly the set that decode would map — no memory beyond the load plan; a failed map-ahead only warns, the decode's `vmm_ensure` remains the backstop. `0` is the rollback. `PLOW_TICK_LOG=1` prints `dec_vmm`/`dec_maps` per tick and `map_ahead`/`map_ahead_maps` per `PFSEG`. |
| `PLOW_KV_MAP_NEXT_CHUNK=0/1` (`--amd-kv-map-next-chunk`) | on | **AMD TP** — extends `PLOW_KV_MAP_AHEAD`: while a prefill chunk drains, also map the KV rows of the same prompt's NEXT planned chunk (its `c0 + rows`), so that chunk's `prefill_prepare` finds them mapped (624 driver maps per steady 8192 chunk at TP8 on the frozen GLM packet). The rows are exactly what that prepare maps one chunk later; they lie past everything the running chunk writes or reads. A failed map only warns; `prefill_prepare`'s own `vmm_ensure` is the backstop. The first chunk of a request still maps its own rows (its slot is not known a tick early). `PLOW_TICK_LOG=1`: `PFSEG prepare_maps` / `prepare_vmm`. |
| `PLOW_AMD_DECODE_GEMM_OVERLAP=0/1` (`--amd-decode-gemm-overlap`) | off | **AMD decode** — dispatch a native decode GEMM without the AQL barrier bit when it directly follows native GEMMs it is independent of (byte ranges planned at load, `amd_gemm_lt::overlap_barriers`). Bit-identical by construction. **Measured loss, keep off:** 7-layer TP8, 14 runs, median +1.5 ms/step (+14 %); `PLOW_TRACE_RAW` shows the native GEMM runs do not shorten and, on the grouped packet, every native interval grows 40–50 µs. Code on branch `decode-latency`. |
| `PLOW_AMD_PUBLISH_DEFER=0/1` (`--amd-publish-defer`) | on | **AMD TP** shared-prefix cache — stash a completed chunk's prefix publish (snapshot allocation + scale/tail copy + radix insert, every 16384-row boundary and at the prompt's end) and run it at the next GPU drain window: the next chunk's, between its enqueue and drain, or the decode's `drain_and_audit`. The snapshot reads only rows below the frontier, which nothing writes before the flush; `begin_slot` / `stage_attach` / `commit_attach` / `publish` / `release` flush first, so no request sees the cache without it (`SharedPrefix::defer_publish`). `PLOW_TICK_LOG=1`: `TICK pub_deferred`, `PFSEG publish_deferred`. |
| `PLOW_TP_PREFILL_AUDIT_DIRECT=0/1` (`--amd-tp-prefill-audit-direct`) | on | **AMD TP** — read each prefill chunk's exact cross-GPU counter audit (same gates, same expectations) through the large-BAR mapping instead of one 12 KiB D2H copy per rank. Stays every chunk: the next dispatch's `zero_xctr` erases the evidence, and a timed-out collective corrupts KV every later row reads. The device compact audit (`PLOW_TP_AUDIT_COMPACT`, decode) is not usable for prefill: `plow_xctr_audit` has no `IndexTpPf` / `XReduceScatter` / `XAllGather` case. Needs host-writable peer memory (large BAR). |
| `PLOW_AMD_ENGINE_AFFINITY=auto\|off\|<cpulist>` (`--amd-engine-affinity`) | auto | **AMD** — where the engine thread (`plow-eng-<model>`, which runs every tick) is allowed to run, set once at its first tick. `auto` = every online CPU of the socket that holds rank 0's GPU, derived from the device's PCI NUMA node in sysfs (`exec::engine_affinity`); `off` = the scheduler decides; or an explicit CPU list such as `72-95,264-287`. That thread writes every AQL packet and kernarg, re-arms the counter banks over SDMA and polls completion; on the socket away from rank 0's GPU all of it, and the GPU's own dispatch retirement, run slower: GLM-5.3 TP8 decode tick 103.4 ms pinned to socket 1 vs 97.5 ms pinned to rank 0's node, and unpinned processes landed on either socket at random (96.2 / 103.7 ms). A device that reports no NUMA node leaves the thread unpinned (logged). Model load is not affected (the pin happens at the first tick). |
| `PLOW_HSA_DRAIN_BLOCKED=0/1` (`--amd-hsa-drain-blocked`) | off | **AMD, diagnostic** — drain with a blocked (interrupt-backed) wait on the queue's completion signal instead of busy-polling it. Measured: no effect on the socket penalty above (103.5 vs 103.4 ms on the far socket); kept for the record, not a tuning knob. |
| `PLOW_VMM_DEFERRED_RECLAIM=0/1` / `--vmm-deferred-reclaim` | 1 | **AMD** VMM KV pools (shared-prefix cache groups and `PLOW_VMM_KV`): recycling a slot keeps the previous occupant's row-0 block in place when it is private, unmaps a cache-shared row-0 block inline, and leaves every other block mapped-but-stale for the pool thread to unmap (`hsa_amd_vmem_unmap` is a serialized ~0.3 ms KFD call; a 66k-token occupant at TP8 cost ~1.9 s of engine-thread unmaps per recycle). Zero-ref handle releases move to that thread too. Budget unchanged (stale blocks reach the pool/driver as soon as the thread runs; pool and cache caps apply as before). `0` = synchronous unmap in `begin_slot`. CUDA pools are unaffected. |
| `PLOW_DRAIN_TIMEOUT_MS=N` | unset (unbounded) | S1 switch drain deadline; past it the victim's live generations are preempted (`Preempted` finish, queued jobs 429). `0` preempts immediately. |
| `PLOW_PRELOAD` | on | speculative next-model preload after an S1 switch; `=0` disables. |
| `PLOW_DEVICES=0,1` | all visible | CUDA device ordinals to serve on. Indices into the **visible** set, not the physical one. |
| `PLOW_PLACE=spread\|pack\|explicit` | `spread` | CUDA: how models are laid out over the visible devices. |
| `PLOW_PIN=slug@2,...` | unset | CUDA: pin a model to a device; required for every model under `--place explicit`. |
| `PLOW_CO_SCHED=free\|rr` | `free` | how co-resident models take a shared device group; `rr` adds FIFO turns on CUDA, AMD and CPU. AMD multi-model startup requires `rr`. |
| `PLOW_CO_SCHED_QUANTUM=N` | 4 | consecutive mux ticks one model keeps its group under `rr`; not a token or wall-time limit. |
| `PLOW_MODELS_ROOT=dir[:dir]` | startup `--assets` parents | directories `POST /v1/models/load` may take an assets dir from. Defaults closed. |
| `PLOW_TP_AGREE_EVERY=N` | 1 | TP cross-rank agreement interval. `PLOW_TP_NO_AUDIT=1` disables the redundant-rank audit (timing runs); `PLOW_TP_SERIAL_LOAD=1` restores one-at-a-time per-rank load. |
| `PLOW_LOAD_PROFILE=1` | off | split upload wall time into alloc / stage+DMA profiling. |
| `PLOW_STEP_TIME=1`, `PLOW_TTFT_LOG=1` | off | per-decode-step host-op timing / TTFT breakdown logging (diagnostics). |
| `PLOW_TICK_LOG=1` | off | AMD serve: one `TICK` line per mux tick (prefill launches/rows/ms, decode rows/ms, host remainder, idle before), one `PFCHUNK` line per prefill chunk (cursor / rebase / `prefill_chunk` / restore / snapshot / prefix publish ms) and one `PFSEG` line per TP `prefill_chunk` (prepare / rearm / xctr / enqueue / map_ahead / drain / audit ms, `prepare_maps` / `map_ahead_maps` VMM block mappings, cumulative per-rank drain); the `TICK` line also carries `dec_vmm` (ms the decode spent mapping KV blocks), `dec_maps`, `dec_segs` / `dec_inflight_enq` / `dec_inflight_sub` (rank 0's dispatches still in flight after the decode's enqueue and after its re-arm: near `dec_segs` = the GPU was behind the host, so both overlapped it) and `pub_deferred`; `PFSEG` also carries `prepare_vmm` / `prepare_patch` (the rest of `prepare` is the id/pos/kvlen uploads) and `publish_deferred`; `PFCHUNK` carries `publish_fill` (the snapshot copy inside a publish). Diagnostics; stderr. |
| `PLOW_NATIVE_LAUNCH_TIMING=1` | off | AMD prefill diagnostic: drain after each launch of the native sparse routes and print one line per segment — `route=sparse_mla[_single] rows=… pack_us=… attention_us=… [reduce_us=…]` for the AITER sparse MLA, `route=index_tp rows=… score_us=…` for the TP indexer (only its score pass can be drained: select/gather/complete open with an all-rank rendezvous the peers have not reached yet, so the rest of that segment is `critical_us - score_us` in the `PLOW_PREFILL_SEG_TIMING` line). Serialises those segments — use on a `plowrt bench` run, never in serving. |

## Visible devices: `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`

The three are not equivalent, and one of them used to do nothing at all.

| variable | who applies it | effect on plowrt |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | libcuda, at `cuInit` | honoured. Every ordinal plowrt uses is already an index into the masked set. |
| `ROCR_VISIBLE_DEVICES` | ROCr, at `hsa_init` | honoured. `hsa_iterate_agents` returns the masked set. |
| `HIP_VISIBLE_DEVICES` | the HIP runtime | **plowrt never loads HIP** — it dlopens ROCr directly. plowrt applies this one itself, but only when `ROCR_VISIBLE_DEVICES` is unset. |

Before this, `HIP_VISIBLE_DEVICES` had no effect whatsoever on plowrt
(measured: `HIP_VISIBLE_DEVICES=4,5,6,7` still enumerated all 8 agents), so an
operator who leased one GPU that way got a process quietly using every GPU on
the box.

**The two AMD variables do not compose the way they look.** HIP indexes into
the set ROCr already made visible. With `ROCR_VISIBLE_DEVICES=4` there is one
visible agent, at index 0, and `HIP_VISIBLE_DEVICES=4` then names nothing. So:

* `ROCR_VISIBLE_DEVICES` set → it decides the set; `HIP_VISIBLE_DEVICES` is
  ignored, with a warning saying so. This keeps lease tooling that exports both
  to the same absolute id working, and plowrt can only ever reach GPUs ROCr
  granted it.
* only `HIP_VISIBLE_DEVICES` set → plowrt applies it as an index mask over the
  enumerated agents, and logs that it did.
* `HIP_VISIBLE_DEVICES` naming devices by UUID (`GPU-...`) with no
  `ROCR_VISIBLE_DEVICES` → **refused**, because plowrt cannot resolve a UUID
  without HIP and silently ignoring a visible-device mask is how a lease gets
  violated.

Both vendor runtimes **truncate** a list at the first entry that does not parse
rather than skipping it, so `0,1,x,3` means devices 0 and 1 — device 3 is
silently gone. plowrt reports the truncation point.

CUDA `--devices` / `PLOW_DEVICES` indexes the visible set, so with
`CUDA_VISIBLE_DEVICES=4,5` the two GPUs are `--devices 0,1`. AMD currently uses
its first visible device (or the first TP-width visible devices) for every
startup model. Restrict that set with `ROCR_VISIBLE_DEVICES`; independent AMD
model placement and residency management are not implemented. Placement flags
on a non-CUDA backend are rejected.

Model load/unload/status routes are privileged and are served only on the
owner-only Unix socket (`--socket`), not the public TCP listener. For example:
`curl --unix-socket /tmp/plow.sock http://localhost/v1/models/status`.
The public `/v1/models` catalogue remains available on both listeners. CUDA
supports managed demand load, eviction and explicit load/unload. AMD and CPU
load their native models at startup; control-plane load/unload returns an error.

Round-robin turns are independent per device group. Each tick can contain a
bounded prefill chunk and multistep decode, so quantum 4 does not mean four
tokens or four milliseconds. Cold prefill yields between chunks under `rr`,
including when decode deferral or no-interleave is configured. Slow response
consumers receive an explicit error after the token buffer fills; they do not
block another model's submission thread.

`plowrt bench` always records AMD overlap capability under
`engine.amd_overlap`. Current HSA engines report shared prefill/decode scratch,
one global queue per rank, no per-XCD queues, and `overlap_safe=false`; this is
evidence only and does not enable overlap. With `--engine-diagnostics`, the
report also includes the per-rank queue identities and raw prefill/decode ranges
used to derive the fail-closed result.
| `PLOW_HSACO_LOWRUNG=dir:max[,dir:max…]` | unset | AMD decode-object tiers. The runtime selects the narrowest tier whose `max` covers the occupied decode rung, pairing-checks each tier at that width, and falls back to the primary HSACO inventory above it. A single legacy `dir` uses `PLOW_LOWRUNG_MAX` (default 2). |
| `PLOW_STATE_CLEAR_DEVICE=1` | off | AMD admission experiment: clear slot-major recurrent state with one device kernel per rank instead of host-staged SDMA fills. Requires rebuilt decode objects carrying `plow_state_clear`. |
| `PLOW_EMIT_DECODE_CUBLASLT=1` | off | Compiler option that marks eligible isolated BF16 decode projections with the packet `CUBLASLT` segment role. PlowRT validates and executes the declared roles; there is no runtime enable or opt-out flag. |
| `PLOW_EMIT_DECODE_NATIVE_TC=1` | off | Compiler option that marks eligible BF16 decode projections with the native sm_90a transposed tensor-core GEMV role (`gemv_sm90_transposed`). Serving qualification pending; 35.4 µs vs cuBLASLt 32.7 µs at B8 in isolation. |
| `PLOW_EMIT_PACKED_PREFILL` / `--emit-packed-prefill` | on for qualified Hopper TP1 BF16 dense packets and for GLM (`glm_moe_dsa`/`glm5_next`) on gfx942 | Emit the packed-prefill request ABI and dedicated packet-paired objects. Packed prefill uses one attention split across buckets so packing and chunk length do not change the softmax reduction. This trades short-prompt split parallelism for stable request numerics. The runtime activates it from metadata and uses ordinary prefill with automatic prefix reuse. Explicit `PLOW_PF_BATCH=1 PLOW_VMM_PREFIX=1` selects the experimental combined route with KV admission. `false` is the rollback. On AMD it emits a SECOND, family-segmented copy of every prefill bucket (a `packed_prefill_program_t` sibling the runtime resolves only while staging a packed binding); unset, the packet is byte-identical. Implemented for the Kimi-K3 and GLM emitters. **GLM gfx942 emits the siblings by default** (recorded as `production_default`; a `--replay-knobs` recipe keeps its own value): the ordinary programs are byte-identical with or without them, the siblings ride next to the native AITER MoE / hipBLASLt / resident-MoE segments (class A, row-agnostic over the dense live rows), and sparse (DSA) buckets get a sibling only under `PLOW_PACKED_SPARSE_PF=1` (their interpreter selectors are class C; the span-aware TP-indexer + AITER chain is not). Without siblings an MLA blob's norm ops share a segment with the Gemms and the packed route refuses every rung. |
| `PLOW_PACKED_PREFILL_ROUTE` | unset: follows the packet | Load the lean packed-family HSACO objects (`interp_packed_mla_norm`, `interp_packed_mla_flash`, their `_fp8kv` twins for FP8-KV packets, `_tb` twins for token-batch bodies, `interp_packed_kda`) and permit exact-family routing after metadata is staged. **Unset = on exactly when the blob carries packed-prefill siblings or token-batch bodies**, and then a family object those programs need that is missing from `PLOW_HSACO` is a load error naming the file — never a silent isolated fallback. `=0` is the rollback (every prefill isolated); `=1` forces the load on a packet without siblings. Missing/wrong markers and mixed segments refuse. Dense AMD packing uses the normal span-aware objects with `PLOW_PF_BATCH=1` and needs neither this route nor `PLOW_EMIT_PACKED_PREFILL`; an **MLA** blob needs BOTH, plus objects from a `PLOW_PACKED_PREFILL_CONSUMERS=1` build (gfx942) / `PLOW_HSACO_PACKED_PREFILL_CONSUMERS=ON` (cmake). The runtime says at load which of the three preconditions is missing — see `report_packed_prefill_route`. |
| `PLOW_TOKEN_BATCH_TP` / `--token-batch-tp` | off | Emit one token-batch BODY program per GLM prefill bucket wider than the decode band: the bucket's packed-segment topology plus the batched decode attention chain and a band-sampling tail over a slot-indexed band of `decode_rungs().last()` rows (`plans/unified-token-batch.md`, "AMD TP8 lowering decision"; tag `TOKEN_BATCH_PROG`). Sparse-prefill buckets are skipped unless `PLOW_PACKED_SPARSE_PF=1` (below). The body carries its own packed topology, so `PLOW_EMIT_PACKED_PREFILL` may stay off. Unset ⇒ byte-identical blob. Serving: `PLOW_TOKEN_BATCH=1` (default) + `PLOW_PACKED_PREFILL_ROUTE=1` + the `_tb` objects below; the load line `route="unified-token-batch/slot-band" armed=… bodies=…` names what is missing. |
| `PLOW_PACKED_SPARSE_PF` / `--packed-sparse-pf` | off | Let the packed siblings (`PLOW_EMIT_PACKED_PREFILL`) and token-batch bodies (`PLOW_TOKEN_BATCH_TP`) cover the SPARSE (DSA) prefill buckets too — on GLM-5.3 TP8 the 8192 rung, the one every 64K-context chunk runs. Requires `PLOW_GLM_INDEX_TP=1`, an unpooled indexer and FP8 KV: under the packed topology the native TP indexer (`IndexTpPf`) feeds the sparse FP8 flash directly (`fj[1] = iidx_pf + 1`; no per-8-query `IndexUnionPf`, whose tiles could straddle two requests), and at the runtime every row's position, causal bound and key base come from the launch's `PlowKvSpan` table (`dsa_tp_adapter_gfx942.elf` with `plow_dsa_tp_abi_2`, `scripts/build_dsa_tp.sh`) while the AITER sparse flash runs one pack→attention→reduce chain per span against that span's own slot cache. The interpreter selectors (`IndexScorePf`/`IndexSelectPf`/`IndexUnionPf`, pooled chains) stay class C and are refused in packed programs. A span packed onto a sparse rung must start ≥ 2047 keys deep (the fixed-width 2048-key CSR): the scheduler admits spans by that rule (`packed_span_admissible`; a request's first rows ride the widest dense body or run isolated) and the staging path refuses by name. Serving a packet with sparse siblings/bodies on an ABI-1 adapter or without `PLOW_MLA_PF_AITER=1` leaves those programs un-offered (`check_packed_prefill_program` names the missing piece); isolated prefill is unaffected. Unset ⇒ byte-identical blob. Default off until the 8-GPU A/B lands (`docs/amd/tp-bringup-mi300x.md` §22). |
| `PLOW_TOKEN_BATCH_TP_OBJECTS=1` | off (`scripts/build_gfx942.sh`) | Also build `interp_packed_mla_{norm,flash}_tb{,_fp8kv}`: the packed MLA family objects with the slot-band resolver compiled in (`PLOW_PACKED_PREFILL_BAND=1`, marker `plow_packed_prefill_band_1`), which a token-batch body's MLA family segments are routed to. Requires `PLOW_PACKED_PREFILL_CONSUMERS=1`. Shipped objects untouched. |

By default, a cold-prefill slot yields after emitting its first token so ready
decode rows can join the next tick. Admission and decode-rung policy use the
pending depth of that model's own ingress queue (`rx.len()`); requests queued
for another model do not widen this model's packet rung.

### Segmented prefill (sm_90a / GH200)

The five-object prefill stack from the GH200 campaign
(`perf-data/gemma12b-gh200-prefill-campaign.md`). The classing knobs are
**serve-side mirrors of emit-side knobs and must match the blob** — `--pf-seg-pure`
pairs with `plowc`'s `PLOW_SEG_PURE_GEMM`, `--pf-seg-fa512` with `PLOW_SEG_FA512`.
A mismatch is a wrong-object launch, i.e. a device trap, not a slowdown.

| flag (env) | default | effect |
|---|---|---|
| `--pf-seg-dir` (`PLOW_PF_SEG_DIR`) | unset | dir holding `interp_sm90a_pfseg/_pfgemm[/_pffa].cubin`. Unset = single-object prefill. Packets must be emitted **without** `PLOW_UNISEG`. |
| `--pf-seg-pure` (`PLOW_PF_SEG_PURE`) | unset | segment classing: `1` = every plain tiled GEMM is GEMM-class, `fp8` = only TMA-mapped fp8 GEMMs (the ws-entry object's sole arm). |
| `--pf-seg-fa512` (`PLOW_PF_SEG_FA512`) | unset | hd512 flash on the dedicated `_pffa` object: `1` = hd512 only, `all` = both head dims — `all` **requires** an object built `PLOW_BUILD_FA_HD256=1`; the loader refuses the mismatch rather than trapping. |
| `--pf-seg-graph` (`PLOW_PF_SEG_GRAPH`) | off | submit each chunk's whole segment chain as ONE CUDA graph (T35). |
| `--pf-seg-eqsmem` (`PLOW_PF_SEG_EQSMEM`) | off | launch every object with the same dynamic-smem request (avoids per-launch carveout reconfig). |
| `--pf-seg-v2` (`PLOW_PF_SEG_V2`) | unset | classing v2 (`1`) / q8 variant (`q8`). |
| `--pf-seg-time`, `--pf-seg-fatonly`, `--pf-seg-noncoop` | off | diagnostics: per-class event timing / every segment on the fat object / plain (non-cooperative) launches. |

The canonical serving configuration measured in the campaign:

```bash
plowrt serve --assets <dir> \
  --pf-seg-dir <cubins> --pf-seg-pure fp8 --pf-seg-fa512 all --pf-seg-graph
```

Loader/asset overrides: `PLOW_NV_CUBIN[_PF]`, `PLOW_NV_KERNEL[_PF]`, `PLOW_NV_SMEM`
/ `PLOW_NV_SMEM_PF` (override decode/prefill dynamic-smem arena bytes), `PLOW_HSACO`
(AMD `.hsaco` dir), `PLOW_CHECKPOINT`, `PLOW_LIBCUDA`.

### What plow does and does not fuse

Three things are easily conflated:

1. **Chunked prefill** — shipped, on by default, no flag (`PLOW_PF_INTERLEAVE` is the
   per-tick step budget; AMD's default is the widest compiled rung). A long prompt is admitted
   a chunk at a time so live decode streams are not stalled for a whole prompt.
2. **Cross-request prefill packing** — `PLOW_PF_BATCH`, on by default on AMD, `=1` on CUDA.
   Several *mid-prefill requests'* chunks share one launch when a capable packed rung holds
   their sum. On AMD the packet must carry packed siblings (GLM gfx942 emits them by default)
   and the family objects must be present (`PLOW_PACKED_PREFILL_ROUTE`, automatic); the final
   chunk of every prompt and sparse (DSA) buckets stay isolated, so on a ladder whose widest
   packable rung is 2048 the pack only ever holds chunks of at most 2048 rows in total.
3. **Mixed batching (prefill ⊕ decode in one launch)** — on AMD TP8 this is the opt-in
   token-batch body (`PLOW_TOKEN_BATCH_TP=1`, bodies emitted at the prefill rungs): decode rows ride
   as a band inside a prefill-rung body. By default a tick that does both runs the prefill chunk and
   then a separate decode pass. Today only each request's final chunk can ride, because middle
   chunks are planned whole at 8192 rows. The decode-row retirement bug is fixed (004d9bdf); the
   route's full-model requalification: passed on the full model after the fix (positions chain step by step, 2 of 12 prompts loop against 4 of 12 on the ordinary route, retrieval 18/18 with bodies armed); still opt-in. vLLM's chunked prefill carries the decode rows
   in the same forward pass on every step.

The gap in (3) depends on the model. On the 12B asset it was bounded by one weight read per tick
(~12% of a tick at 2k prompts, ~0.6% at 127k). On GLM-5.3 TP8 at C20 with ~70k prompts the separate
decode pass costs ~95 ms per prefill tick, so decode rows riding inside every chunk would be worth
~6% end to end (see the review log's roadmap and vLLM-parity entry).
**Measured, that estimate does not hold:** it assumed the decode pass disappears when its rows ride, but
inside the body the band's decode attention chain costs about as much per layer as the pass, so riding
middle chunks measures −14 ms per chunk at 78 layers (see the review log's decode-rows negative).
</content>
</invoke>
