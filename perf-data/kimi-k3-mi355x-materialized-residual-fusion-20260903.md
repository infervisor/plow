# Kimi-K3 TP8 materialized Residual → AttnRes fusion (MI355X)

Decision: **PROMOTED GLOBALLY**. The graph-derived fusion is bit-identical, removes 178
materialized `Residual` packets from both prefill and decode, and improves 8192-token-context
decode TPOT by a stable median **0.894 ms/token (1.97%)**. It is default-on for every model;
`PLOW_FUSE_RESIDUAL_INPUT=0` is the explicit rollback.

## Cell

- Hardware: 8×MI355X, TP8, gfx950, 256 CU, global-queue + L2-domain dispatch.
- Model: Kimi-K3 text tower, max context 32768; BF16 weights/activations/KV and MXFP4 experts.
- Packet: measured gfx950 tunedb fingerprint `gfx950-76ef5b9982d04cbd`, **7650/7650** dense
  tiles selected by measurement (`tile_source=measured`).
- Isolation: control/fused emits differed only by `PLOW_FUSE_RESIDUAL_INPUT=0/1`.
  P1 (`--moe-stage2-lean=false`), P3 (`--kda-chunk-qpre=false`), and P5
  (`--moe-stage1-lean=false`) were disabled in both arms.
- Benchmark: deterministic random input length 8192, C1, no warmup; prefill used output length 1
  with raw packet traces, decode used output length 256. Pair order C→F, F→C, C→F.
- Every run held the shared exclusive eight-GPU lease. Every before/after foreign-process audit
  was clean. `--amd-tp-agree-every 1`, `--parity-report`, and `--engine-diagnostics` were enabled.

## Graph and capability gate

- `Builder::finish` applies the rewrite after the complete graph exists and before flattening.
  Eligibility requires a sole adjacent `Residual` predecessor, its sole graph successor,
  the exact coarse dependency, unit scale, matching materialized output/input handles, and free
  AttnRes fused-input operands. The rewrite preserves the nested BF16 materialization rounding,
  predecessor dependencies, later fanout, and counter remapping.
- T8192 prefill: 3149 → 2971 instructions; 178 → 0 standalone `Residual`; 187 `AttnRes`, of
  which exactly 178 carry fused operands. Stream entries: 771197 → 725629, exactly
  `178 × 256` fewer.
- T1 decode: 2343 → 2165 instructions; the same 178 sites fuse and 187 `AttnRes` remain.
  Stream entries: 307810 → 307454, exactly `178 × 2` fewer.
- The nine unchanged `AttnRes` sites have no eligible materialized-residual predecessor: the
  layer-0 attention push and eight snapshot/reset sites.
- All seven final programs (prefill T128/512/1024/2048/4096/8192 and decode T1) passed the Lean
  GQ ordering certificate and `LdsFitSound`; `build.json` records `verified=true, oracle=true`.
- Operand-derived manifest capability, generated `PLOW_MATERIALIZED_RESIDUAL_INPUT`, object
  marker `plow_materialized_residual_input_1`, and runtime packet/object refusal all agree.
  The required V2 flash object also advertised `plow_mla_pf_v2_arm_1`.

## Results

### 8192 → 1 prefill

| Fold | Order | Control TTFT ms | Fused TTFT ms | Fused delta ms |
|---:|:---:|---:|---:|---:|
| 1 | C→F | 2699.454 | 2693.609 | -5.845 |
| 2 | F→C | 2698.573 | 2696.843 | -1.730 |
| 3 | C→F | 2698.529 | 2695.376 | -3.153 |

Median arm values: 2698.573 → 2695.376 ms, **-3.196 ms (-0.12%)**. All six runs emitted token
6896 and checksum `fnv1a64:7d749e3b002fafa7`.

Fold-1 trace attribution matches the endpoint delta. Chain span fell 2523.562 → 2517.414 ms
(-6.148 ms). The removed 178 `Residual` packets accounted for 18.277 ms; their materialization
work moved into `AttnRes`, whose aggregate body rose 73.679 → 88.865 ms. This is a packet-floor
win, not deletion of the BF16 arithmetic.

### 8192 → 256 decode

| Fold | Order | Control TPOT ms | Fused TPOT ms | Fused delta ms | Delta % |
|---:|:---:|---:|---:|---:|---:|
| 1 | C→F | 45.415454 | 44.523310 | -0.892144 | -1.964% |
| 2 | F→C | 45.447112 | 44.509728 | -0.937384 | -2.063% |
| 3 | C→F | 45.417670 | 44.526117 | -0.891553 | -1.963% |

Median arm values: 45.417670 → 44.523310 ms, **-0.894360 ms/token (-1.969%)**. Every run
completed 256 output tokens; all six full token-id arrays were identical and every checksum was
`fnv1a64:6bdfaa7b84ee4e7e`.

## Artifacts

Root: `/tmp/k3-whole-graph-fusion-measured.0hi01H`.

| Artifact | SHA-256 |
|:---|:---|
| control `model.pkt` | `ea41d91640d417227cfb88277473f57f5af0f0ae49c943380a4b4afbb255d59b` |
| fused `model.pkt` | `3968134fd268c9247fd9857b5fa235698430983eb1b154f6022cbba0eeead683` |
| control `build.json` | `292ed9a127fa2fff8b2e30ddc17c18e4c914c0b497edb21aa3e7704457b8e2e3` |
| fused `build.json` | `a6989f3312212dc4577fa621d4513ed77c36c0505c38dbb7874fe67b0e362286` |
| control prefill GQ object | `da26db08980cef35c5a3eb1b08bbd29e256338209524418ab3cfc62575bc8770` |
| fused prefill GQ object | `a3f5b76bdf2c8ee41c36a8c8000077ebacee2e8b6ecb16d87c6be0d59226f1d6` |
| control decode GQ object | `4480746a7667185497e9ee99d897449969e36667aeaf3d3af2481b9dea062843` |
| fused decode GQ object | `3933baddc5ea4dd62cf1a3702f1ff6bdc7c0e9d28afd33f7f01eff80e42cee0d` |

## Invalid attempts and safety fix

No number from the first asset pair was accepted: it reported `tile_measured=0` and used the
analytical fallback. Its first launch also found a flash object missing the required MLA-V2 arm.
The loader incorrectly swallowed that hard capability error as an optional flash fallback;
commit `8d16842` now propagates required V2 flash load failures and adds a regression. The final
campaign used newly emitted measured assets and rebuilt V2-capable objects only.
