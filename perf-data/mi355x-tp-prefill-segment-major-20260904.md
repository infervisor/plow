# MI355X TP prefill segment-major submission — 2026-09-04

## Status

Production default. `PLOW_TP_PREFILL_SEGMENT_MAJOR=1` (or
`--amd-tp-prefill-segment-major=true`) submits `(segment, rank)` in segment-major order and drains
all ranks once after the chunk. Every rank owns one ordered AQL queue and every dispatch packet
has the barrier bit, so local segment order remains identical. Cross-rank counters are zeroed once
before submission and the existing exact counter audit runs after the final drain.

`PLOW_TP_PREFILL_SEGMENT_MAJOR=0` restores the per-segment drain path. A configured tensor capture
retains that path because later queued segments may overwrite the requested intermediate boundary.

Prototype commit: `7731f5a`. Release HSA binary SHA256:
`a0525cf0bc6ec75ced224242428b6e0b516e9ce258632218bdf4ca269cf53b88`.

## Static gate

- `cargo fmt --all -- --check`.
- Nine `exec::amd_tp::tests` pass, including synthetic 4-segment/3-rank ordering and the existing
  decode segment-major/L2-domain tests.
- `config_env` proves the `0`/`1` environment contract.
- `cargo check -p plowrt --features hsa` and release HSA build pass.

## Full-network gate

Exact BF16-KV TP8 `8192 -> 256`, one request/concurrency, no warmup, current measured packet and
paired objects. Control explicitly sets the flag to 0; candidate sets it to 1. Counter audit is on,
all-rank agreement runs every step, and every arm completed. Fold order was control/candidate,
candidate/control, control/candidate.

| fold | control TTFT ms | candidate TTFT ms | TTFT delta ms | TPOT delta ms | E2E delta ms |
|---|---:|---:|---:|---:|---:|
| 1 | 1415.629 | 1404.273 | -11.356 | -0.0464 | -23.194 |
| 2 | 1417.009 | 1406.955 | -10.054 | -0.0482 | -22.348 |
| 3 | 1416.374 | 1404.911 | -11.463 | +0.0295 | -3.935 |
| mean | 1416.337 | 1405.380 | **-10.958** | **-0.0217** | **-16.492** |

All six token arrays are byte-identical, 256 IDs each, SHA256
`3b1345553d40748ce2baf58be3a0c20419d8662548dc3d4afa1d6ef04673a1ea`. The retained checksum is
`fnv1a64:6bdfaa7b84ee4e7e`. Fold summary SHA256:
`7607b7fe2bf9f7fa89405d35cda5dec64bebc6fb21131a694d2a4add4cf14902`. Exclusive TP8 lease
`tp-prefill-seg-major` returned 0 after 626 seconds with no overlapping lease.

## Repeated-prefill and rung safety gate

Each arm below ran two sequential requests on one loaded engine, eight generated tokens per
request. This exercises prefill counter re-arm/reuse rather than rebuilding the engine between
requests. Counter audit and all-rank agreement ran on every request/step. `plowrt bench` permits
full token serialization only at `requests=1`, so this gate compares its exact aggregate FNV
checksum; the preceding three 256-token folds provide byte-for-byte token arrays.

| rung | checksum, both arms | TTFT delta ms | TPOT delta ms | E2E delta ms |
|---:|---|---:|---:|---:|
| 128 | `fnv1a64:a9a5506762aa74e9` | -4.404 | +0.0302 | -4.192 |
| 1024 | `fnv1a64:f859e6e9c711d91f` | -5.848 | +0.0577 | -5.444 |
| 2048 | `fnv1a64:ff971827d59d000e` | -8.126 | -0.0328 | -8.355 |

All six processes completed both requests with zero failures. Safety summary SHA256:
`68bcf58d809ad25fa5ec8b7d7b309a8a0635b32e00579f2bc04bfcd9ccf2a927`. Exclusive TP8 lease
`tp-prefill-seg-major-safety-r2` returned 0 after 569 seconds. The initial attempt was rejected
before model load because `--parity-report` requires one request; it touched no GPU work.

## Decision gate

The 10.96 ms TTFT gain is material and repeatable. Audit/all-rank agreement, counter reuse, T128,
T1024, and T2048 pass. A matched raw trace also localizes the gain to the removed host/drain
residual:

| metric | control | segment-major | delta |
|---|---:|---:|---:|
| endpoint TTFT ms | 1420.763 | 1405.520 | **-15.243** |
| traced chain span ms | 1404.400 | 1390.551 | **-13.849** |
| traced gate ms | 5.779 | 5.746 | -0.033 |
| traced body ms | 1017.175 | 1024.334 | +7.159 |
| external residual ms | 381.446 | 360.471 | **-20.975** |
| convergence ms | 56.218 | 47.208 | **-9.010** |

Both arms emitted token `6896`; their token files have the same SHA256
`7202693c9fb5709db526cf3511e86ace6055786c338a6693440ddbc5e672c2e4`. The exclusive lease
`tp-prefill-seg-major-trace` returned 0. Control/candidate raw trace SHA256 values are
`b434e2c2881d384caf181982ad069d1342fa2e851ad8e5504f326671e507803f` and
`6d37203c82974e3270cfa4bbd5eabd7f9b635de6b5bb6b3fa35bb0e885a92469`; report SHA256 values are
`6667f590f356a1f85dbe71a47dfd5aa5f7ef76ceb58e9b316378a552c505f2c7` and
`f443eaefa7aee45955d6cd5c92d7e0aa586d0a23b6499ccb7ac00cb15141b461`.

The default is promoted. The trace shows the expected residual/convergence reduction despite a
7.16 ms adverse body-noise sample, while the three production-timing folds consistently recover
10.05–11.46 ms TTFT. Rollback is explicit with `PLOW_TP_PREFILL_SEGMENT_MAJOR=0`.

Any rung failure, counter timeout, output mismatch, or loss of the TTFT gain rejects the experiment.

Raw JSON SHA256 values, folds 1–3 control/candidate respectively:

`4c57d6270c686b1311ed5933e6581fecd5584a05b3f1b3411c94155a155bdd54`,
`831abaa5a0c47c7c8b5cef30a27f7f37962811a33779f62b75a5d232db424d31`,
`77a1a3bf7a14abb4be85a91409068b66e65511e7191b37888f515a1f409d8c71`,
`7acd213783c142993ac5245ede05e26fe67ae4d5c780766b4a8fc8b337fc1038`,
`6a1bfe19b94f62a73a2307c17b78c3cdb4dd0dbfd4780dc0a79bf34b6aeda4f4`,
`2ca4ba8be7594d58e4cbf3e909bb63fc1b5cc6b07fbcbfd6062ebc8aa98b60c3`.
