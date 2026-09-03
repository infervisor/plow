# Kimi-K3 MI355X independent BF16 KDA qpre gate

Date: 2026-09-03. Hardware: TP8 MI355X. Shape: exact random 8192-token
prompt, one greedy output token, C1, no warmup, BF16 KV cache.

## Arms

Both arms use the same checkpoint, isolated HSA runtime, measured TuneDB
profile, GQ scheduler, exact compact TP audit, and
`--amd-kda-family-route=true`. Both also use the promoted P1, P2, and P5
settings: `PLOW_MOE_STAGE2_LEAN=1`, `PLOW_XR2_GATHER=1`, and
`PLOW_MOE_STAGE1_LEAN=1`. `K3_FULL=1`, `PLOW_FP8_KV=0`, and
`PLOW_SEG_PACKED_PREFILL=1` are identical.

The only emit difference is `PLOW_KDA_CHUNK_QPRE=0` for control vs `1` for
candidate. Each arm uses its packet-matched KDA-family object. The manifest
diff is limited to the qpre backend requirement, pairing hash, and the
`KdaChunkCarry/d128_qpre` and `KdaChunkWu/d128_qpre` union variants. This
isolates qpre from both the ordinary mega-interpreter and other promoted
prefill routes.

Both manifests record `precision.kv_enc=bf16` and TuneDB
`tile_measured=tile_lookups=7650`, `tile_source=measured`. The emit ran in the
Nix ROCm environment with recipe fingerprint `gfx950-76ef5b9982d04cbd` and
toolchain label `rocm-7.14.0-nix`. Emits made outside that environment had a
different stale fingerprint and were discarded before timing.

## Three order-alternated folds

| fold / order | control | qpre | delta | exact parity |
|---|---:|---:|---:|---|
| 1, candidate -> control | 1896.014 ms | 1775.426 ms | -120.587 ms (-6.360%) | pass |
| 2, control -> candidate | 1896.157 ms | 1777.297 ms | -118.860 ms (-6.269%) | pass |
| 3, candidate -> control | 1895.774 ms | 1775.085 ms | -120.689 ms (-6.366%) | pass |
| mean | 1895.982 ms | 1775.936 ms | -120.046 ms (-6.332%) | pass |

Paired-delta sample SD is 1.028 ms. All six processes completed 1/1 requests
with zero failures. Every arm used the identical prompt token array (SHA-256
`dd51e931308683300f372862a682d56a332e0e3e8d71cd70ef06c34739362dcd`)
and produced token 6896. Per-run lease audits found no foreign GPU compute
processes before or after any accepted run.

## Trace attribution

The table is the mean critical-envelope body time from all three raw traces
per arm.

| category | control | qpre | delta |
|---|---:|---:|---:|
| `KdaChunkCarry` | 265.683 ms | 143.193 ms | -122.490 ms |
| `KdaChunkWu` | 35.315 ms | 39.567 ms | +4.252 ms |
| `KdaChunkIntra` | 121.219 ms | 120.975 ms | -0.244 ms |
| `KdaChunkPrepare` | 14.688 ms | 14.631 ms | -0.057 ms |
| full trace span | 1720.356 ms | 1600.262 ms | -120.094 ms |

The four KDA categories improve by 118.539 ms in aggregate. Full trace-span
and endpoint deltas agree within 0.048 ms, so the result is explained by the
intended KDA work rather than a host-side shift.

## Capability and resource gates

- Both family objects export `plow_kda_family_segments_1`,
  `plow_packed_prefill_kda_chunk_segments_1`, and their exact packet hash.
- Only the candidate exports `plow_kda_chunk_qpre_arm_1`; the control correctly
  lacks it.
- Both ordinary GQ flash objects export the required `plow_mla_pf_v2_arm_1`
  marker. The fail-closed runtime from commit `8d16842` loaded every accepted
  run without a missing-object, capability, or pairing error.
- Family resources are wave64, 248 VGPR, 106 SGPR, and zero VGPR spill. Private
  memory is 444 B/thread control vs 440 B/thread candidate. The candidate does
  not worsen the private-memory cost, and the family is a separate object
  rather than the mega interpreter.

| artifact | control | qpre |
|---|---|---|
| packet SHA-256 | `46e5296a1a9776f043c3e8701048ddb4145757582f7c875711d9870f8da02b63` | `3071ed2547ab2273a13e62a4f6c086e34b3d78b110a9e9bf5b8f621e844be45e` |
| build manifest SHA-256 | `ca22b81326c89e1b39f8752ebbb52350ab3ab3372f63164742a68ec79b0a7829` | `ac09cd09867301836891546db22d031fe9fa77c6664b66e64f4c9bc4bb2d5bf3` |
| pairing hash | `0x3d1373f347b64fde` | `0x00c8f138072a14b4` |
| KDA-family object SHA-256 | `17aeafdc2ea19b977e64777a58251e9982ce0597a1181c023f60c1e755375231` | `2096263ce0b24be49755606c6754d6c8b4d99752d24ac4002525b841107ca6be` |

The isolated HSA runtime SHA-256 is
`4f4db3e61aaef8476864405cb75b9879a8f9638f2ac2d39d604f6df9e6a3b619`.
Accepted JSON/log/trace/report files are under
`/tmp/k3-p3-bf16.3foM1i/{control,candidate}-r{1,2,3}.*`.

## Decision

Promote qpre emission and the KDA-family runtime route by default. The
independent BF16 result is exact, stable, material, and fully trace-attributed.
The separately isolated family object still has private memory, but qpre
slightly reduces it and improves full-network TTFT by 6.332%. Preserve
`PLOW_KDA_CHUNK_QPRE=0` and `--amd-kda-family-route=false` (or
`PLOW_KDA_FAMILY_ROUTE=false`) as explicit rollback controls. Exact packet
hashes and qpre/family capability markers remain mandatory.
