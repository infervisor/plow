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
`[40..47]`, the same eight forced token ids, the same checkpoint, and TP8.

## Result

| arm | 93-layer time | final comparison to serial |
|---|---:|---|
| A4 verifier | **76.994 ms** | rel RMS **0.86953**, cosine **0.58463**, max abs 17.375 |
| per-token W4A16 | 114.195 ms | rejected: slower and fails full-depth comparator |
| grouped W4A16 | 115.027 ms | rejected: slower and fails full-depth comparator |

The A4 result is finite, nonzero, rank-identical, and counter-clean. That is a
runtime correctness gate, not target-model equivalence. The full-depth numerical
failure blocks acceptance-rate and served-throughput measurements.

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

Run both arms with `amd-block --decode-pos 32 --decode-kvlen 40 --tp 8`, forcing
the same eight ids. The serial arm replays eight one-row steps; the verifier
executes one eight-row step. Use `--inspect-rank N` only for explicitly sharded
intermediates; omit it for the final replicated-output gate so TP rank agreement
remains mandatory.

## Next valid gate

Localize the first full-depth divergence with rank-local intermediates and the
finite fixture. Promotion requires a verifier that matches the serial target,
then deterministic acceptance, state commit/rollback, GSM8K, and served B1
short-to-128K gates. No speculative serving code should land before those pass.
