# Kimi-K3 DSpark target-verifier freeze gate

Date: 2026-08-12. Hardware: 8x MI325X (`gfx942`, 304 CUs/rank). Toolchain:
repository Nix ROCm 7.14. All TP runs used compact exact counter audit and ended
with a clean GPU-process audit.

## Decision

Keep the target-verifier and recurrent-state journal as default-off bring-up
infrastructure. Do not connect it to `plowrt serve` or claim speculative
throughput. The 93-layer verifier is fast enough in principle, but it does not
match eight serial target steps at full depth.

Both attempted W4A16 alternatives are rejected and their kernel changes are
not retained. The frozen branch retains only deterministic fixture generation
and rank-local tensor inspection for further localization.

## Matched finite fixture

`scripts/k3_spec_fixture_seed.py` reads both packet tensor tables and writes a
matched verifier/serial fixture:

- one BF16 activation row, repeated over the verifier's eight rows;
- one committed plus eight candidate KDA state/conv banks for the verifier;
- identical finite committed recurrent state in both arms;
- identical 32-row FP8 MLA `ckv`, BF16 `krot`, and FP32 scale histories;
- zeroed cache capacity after the live history;
- verifier commit selector zero.

The verifier and serial runs use positions `[32..39]`, visible KV frontiers
`[33..40]`, the same eight forced token ids, the same checkpoint, and TP8. The
serial replay therefore starts at `kvlen=33`; starting it at 40 violates the
decode invariant `kvlen=pos+1` and is not a valid comparator.

## Result

| arm | 93-layer time | final comparison to serial |
|---|---:|---|
| A4 verifier | **76.994 ms** | comparator superseded by the corrected frontier run below |
| per-token W4A16 | 114.195 ms | rejected: slower and fails full-depth comparator |
| grouped W4A16 | 115.027 ms | rejected: slower and fails full-depth comparator |

The A4 result is finite, nonzero, rank-identical, and counter-clean. That is a
runtime correctness gate, not target-model equivalence. The full-depth numerical
failure blocks acceptance-rate and served-throughput measurements.

### Corrected frontier and first divergence

With the serial replay corrected to `kvlen=33`, layers 0 and 1 are byte exact.
Layer 3, the first MLA layer, is exact through Q/KV projections, RoPE, FP8 MLA,
the attention residual, router logits, route selection, and routed activation.

The verifier originally selected `d_mla_merge_fold<512,128>` because it carries
eight rows, while serial B1 selects `<512,32>`. The different reduction map moved
one BF16 element by `3.72529e-9`, which later changed an expert selection. The
dedicated `PLOW_K3_SPEC_VERIFY` object now uses the serial VT=32 map. On MI325X
the corrected verifier `o_attn` is byte exact to serial (SHA256
`be5d8df588d46fe1aab0fdad34fa1963d15a487f8a71001084c5fbee895a4471`).

The next first divergence is the MoE body. The verifier uses the T=8 prefill
chain (`MoeGroupGluPf`/`MoeGroupDownPf` plus separate shared gate/up/Situ), while
serial B1 uses the decode chain (`MoeGroupGluFp8Blk`/`MoeGroupDownFp8Blk` plus
`GemvGlu`). Inputs and router choices are byte exact, but the two accumulation
maps produce different routed and shared outputs. The corrected full output is
therefore still not target-equivalent: relative RMS `0.12295`, cosine `0.99244`,
max abs `1.78125`. The VT32 verifier also takes `94.434 ms` for the full block,
so arithmetic equivalence costs more than the old VT128 map. Speculative serving
remains frozen until a verifier-only per-row MoE body reproduces the B1 arithmetic
without losing the block's parallelism.

For capacity only, the measured five-layer draft is 5.208 ms. The A4 verifier
would require more than `(76.994 + 5.208) / 20 = 4.11` committed tokens per
cycle to reach 50 tok/s, but that bound is not actionable until verifier
equivalence passes.

## Fixture reproduction

```bash
nix develop --command scripts/k3_spec_fixture_seed.py \
  --verifier-blob /path/to/verifier/model.pkt \
  --serial-blob /path/to/serial/model.pkt \
  --verifier-out /tmp/k3-verifier-fixture \
  --serial-out /tmp/k3-serial-fixture \
  --history 32
```

Run the verifier with `amd-block --decode-pos 32 --decode-kvlen 40 --tp 8` and
the serial arm with `--decode-pos 32 --decode-kvlen 33 --tp 8`, forcing the same
eight ids. The serial arm replays eight one-row steps; the verifier executes one
eight-row step. Use `--inspect-rank N` only for explicitly sharded intermediates;
omit it for the final replicated-output gate so TP rank agreement remains
mandatory.

## Next valid gate

Add a verifier-only per-row MoE path that preserves the B1 reduction order, then
rerun layer-3 and full-depth equivalence. Promotion still requires deterministic
acceptance, state commit/rollback, GSM8K, and served B1 short-to-128K gates. No
speculative serving code should land before those pass.

## Current-emitter recheck (2026-08-12)

The verifier was regenerated from the current source with `PLOW_K3_SPEC_VERIFY=1`,
`PLOW_K3_NS=16`, `PLOW_DECODE_BATCH=1`, and `--max-ctx 64`. The packet is
`29cb1a21ed72bcf5baedb24dd4383dde9fbbc4c1d59a21b7f8f72ddee1ebf02d` and contains
2,666 decode instructions: 92 each of `MoeRouterTopkPf`,
`MoeGroupGluFp8Blk`, `MoeGroupDownFp8Blk`, `MoeCombinePf`, and `GemvGlu`.
The dedicated gfx942 object built with MM8/WALK has 256 VGPR, 64,568 B LDS, and
31 spills; the grouped ISA audit and baseline audit both pass.

The current packet was run against a matched finite T8 fixture on TP8. It is
finite, rank-identical, and counter-clean (136.65 ms). A serial B1 replay at
the corrected frontier (`pos=32`, `kvlen=33`) also completed (54.09 ms), but
the final `act.x` and logits differ materially for every verifier row; this is
not an equivalence or throughput gate. The next localization must dump the
first post-MLA MoE tensors; no speculative serving measurement is justified.
