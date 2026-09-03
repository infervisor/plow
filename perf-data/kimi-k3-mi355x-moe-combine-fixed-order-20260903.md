# Fixed-order MoE Combine — MI355X TP8

Date: 2026-09-03. Commit under test: `edffb73`.

## Contract and artifacts

The route is model-independent. It accepts only a pure `MoeCombinePf` segment
with nonzero H/T, topk 16, materialized f32 parts, and no banded, part16, or
deterministic-accumulator encoding. It retains the interpreter's exact f32
order: residual, shared, then slots 0 through 15.

Both BF16-KV assets came from one clean Nix snapshot. Both used the measured
TuneDB fingerprint `gfx950-76ef5b9982d04cbd`, resolved 7,650/7,650 cases to
measured choices, and had pairing `0x48a4ccb34189de4a`. Only
`PLOW_MOE_COMBINE_LEAN` differed. The candidate adds 92 pure singleton Combine
segments per prefill rung while retaining 17,619 instructions.

| Artifact | SHA-256 |
|---|---|
| control packet | `a4d8bee5d861daaa8b8db67e8bd8ff316480bf41c1d6d0c1ac4cc7dccf74db02` |
| candidate packet | `1fa49f8b7464bcc1ff6870c6c0390c44c1ce51aaeb0b7c597765ecf41b9228a6` |
| common config | `0430614fa8c10c6dc0c6e10b9260e919baf3609010e95b60c600f0f4b2fd0beb` |
| Combine object | `acba56a824dea17189b4065ac654623edeb9af834e7bc783726d7187864adc2d` |
| runtime | `a76210dcd4b17ce33c38e1332f5b66b5c1923b52593e00315a608da492970be7` |

The shipping object is wave64/WG256, VGPR37, SGPR49, occupancy 8, with zero
private memory, LDS, and VGPR/SGPR spills. Its five fixed-order, slots16,
materialized-f32, wave64, and no-spill capability markers are present.

## 8192→1 folds

Three uncontended order-alternated folds used one request, concurrency one, no
warmup, and all-rank agreement every token.

| Fold / order | Interpreter | Lean Combine | Gain |
|---|---:|---:|---:|
| 1, control → candidate | 1621.145048 ms | 1510.046001 ms | 111.099047 ms (6.853%) |
| 2, candidate → control | 1618.959442 ms | 1510.332192 ms | 108.627250 ms (6.710%) |
| 3, control → candidate | 1618.690766 ms | 1508.576991 ms | 110.113775 ms (6.803%) |
| Mean | 1619.598419 ms | 1509.651728 ms | **109.946691 ms (6.789%)** |

Paired-gain sample standard deviation is 1.244340 ms. All six arms completed
one request with zero failures, complete non-overflowed diagnostics, all-rank
prefill completion, token 6896, and checksum
`fnv1a64:7d749e3b002fafa7`.

Control traces charged 146.458/146.496/146.472 ms to the 92 interpreter Combine
packets. The candidates' 92 standalone drains totalled
38.879/38.864/38.854 ms. Total segment drains improved by
112.153/109.416/111.042 ms, within 0.93 ms of endpoint improvement. The
boundary costs about 0.422 ms/layer and does not erase the kernel gain.

## 8192→256 exact gate

Control/candidate TTFT was 1617.633152/1509.779947 ms; TPOT was
44.409768/44.382407 ms. The complete 256-token arrays are identical. Their
newline-ID SHA-256 is
`1398465e8212d27e43a6d52e95163ae34912b72255d6d14b82d7eacdcf4d718e`;
the runtime checksum is `fnv1a64:6bdfaa7b84ee4e7e`. Both diagnostics report
eight ranks, sampled agreement every token, per-dispatch counter audit, and
all-rank prefill completion.

## Promotion

The generic route is default-on. `PLOW_MOE_COMBINE_LEAN=0` or `false` is the
explicit packet rollback. Missing objects retain interpreter fallback; present
objects must pass the ABI/resource markers.

Raw evidence: `/tmp/k3-combine-edffb73/results`. Nix shell-hook output prefixed
the benchmark stdout, so original streams are retained as `.stdout`; canonical
`.json` files start at the first `{`, pass `jq`, and are hashed separately.
