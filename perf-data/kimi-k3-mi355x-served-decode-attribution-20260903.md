# Kimi-K3 served decode attribution — MI355X TP8

Date: 2026-09-03. Hardware: 8× MI355X (`gfx950`), TP8. Workload: one
production `ModelMux` request, random seed 0, greedy BF16-KV 8192→256, C1,
no prefix cache, no MTP. The packet SHA256 is
`c75382f2d906ae7c5fcf2a761b18f6add9599ed2a61a53993db0c3cc055cd84d`.
The run held an exclusive eight-GPU `gpulease`, completed 256/256 output
tokens, and reported no reject, shed, device fault, counter-audit failure, or
rank disagreement.

## D0: the endpoint gap is the old counter audit

`PLOW_DSTEP_EVERY=255` covered every decode interval in the request.

| component | ms/token | share |
|---|---:|---:|
| device gate, normalized from the final packet trace | 2.369 | 4.3% |
| device body, normalized from the final packet trace | 40.248 | 72.2% |
| host launch/readback/stream, including safety audit | 13.082 | 23.5% |
| idle between mux ticks | 0.030 | 0.1% |
| inner mux remainder | 0.001 | <0.1% |
| **attributed total** | **55.731** | **100.0%** |
| endpoint mean ITL | 55.731001 | — |

The attributed total differs from endpoint mean ITL by 0.000053%. The raw
rank-0 trace covered 2,343 packets and 307,810 workgroup-packet records. Its
43.84064 ms critical envelope was 2.43741 ms gate and 41.40323 ms body. The
trace is one final token while DSTEP is the all-rank 255-token mean, so the
gate/body ratio is normalized to the independently measured 42.61780 ms mean
all-rank drain. No endpoint value is used to construct either ratio.

The trace reporter previously charged the complete body duration of
overlapping packets and produced an impossible 47.65 ms sum inside a 43.84 ms
span. It now advances a monotonic critical envelope and charges only the
uncovered part of each body. Its corrected gate + body equals the device span.

The host split was:

| host phase | µs/token |
|---|---:|
| seed ids | 69.30 |
| decode prepare | 251.45 |
| inactive local-counter rearm | 1,244.78 |
| zero cross-rank counters | 78.12 |
| enqueue TP8 launches | 3.51 |
| **post-drain TP counter safety audit** | **11,349.81** |
| read all sampled ids | 80.73 |
| rank compare | 0.03 |
| detokenize, stop, stream send | 4.10 |

`Admit::Defer` did not fire. The scheduler recorded 255 C1 decode ticks,
mean batch 1, zero rejects, and zero sheds. The measured 30.07 µs between mux
ticks also rules out the proposed 8 ms formation hold: that hold is cold-start
TTFT-only, not steady decode.

## D1: exact compact audit

The old healthy path allocated and pinned a 59,392-byte host buffer, issued
one synchronous D2H copy per rank (475,136 bytes total), and scanned 464
128-byte-strided gates for every token. The existing compact path launches
`plow_xctr_audit` concurrently on all ranks after the model drain, drains those
eight kernels, and reads one isolated large-BAR status word per rank. It checks
the same expected count for every gate, including zero for unused gates, and
falls back to the complete copy audit to diagnose a nonzero status.

| 8192→256 C1 | copy audit | compact audit | delta |
|---|---:|---:|---:|
| TP safety audit | 11.350 ms | 1.160 ms | −10.189 ms |
| mean TPOT / ITL | 55.731 ms | 45.298 ms | **−10.433 ms (−18.7%)** |
| output throughput | 14.94 tok/s | 17.69 tok/s | +18.4% |
| output checksum | `fnv1a64:6bdfaa7b84ee4e7e` | same | exact |

Compact audit is now the default. `PLOW_TP_AUDIT_COMPACT=0` retains the copy
path. Full all-rank token agreement remains every token by default, and the
counter audit still runs on every dispatch.

The compact-audit status pointer moved from `i7` to `j1` (`fj[2]`). This is
required because folded two-shot gather uses `i6/i7`; a loader patch must not
clobber its gather slot or column count. A regression covers all four audited
collective opcodes and preserves both XR2 fields. The exact gfx950 K3 decode
object compiles with this ABI.

## K1/K4 1024-token gate

Both arms used compact counter audit on every device step and generated exactly
1,024 tokens. K1 read and compared all eight sampled ids every step. K4 used
four device steps per scheduler tick and full sampled-id agreement every fourth
step; the per-step counter audit remained enabled.

| arm | TPOT | output throughput | engine selections | output checksum |
|---|---:|---:|---:|---|
| K1 | 45.290 ms | 20.791 tok/s | 1,023 × 1 step | `fnv1a64:45a6e9fa1ea42fc5` |
| K4 | 45.200 ms | 20.830 tok/s | 255 × 4 + 1 × 3 steps | `fnv1a64:45a6e9fa1ea42fc5` |

The complete 1,024-token arrays are byte-identical; newline-id SHA256 is
`73236d87cfb5c73938b037c8f5f8ffab30bb04f16e2cd279a5b927aff70d83ad`.
K4 is neutral at this model size; the promotion is the compact audit, not
deferred readback.

Against the matched vLLM 0.28 BF16-KV baseline, Plow's K1 TPOT gap improves
from 2.67× to 2.17×. Output throughput is 20.79 vs 46.73 tok/s (2.25× gap).
This closes the D0/D1 host unknown but does not satisfy the Tier-2 win.

## Commands and artifacts

The D0 command added `PLOW_TRACE_RAW=/tmp/k3-d0-decode.trace`,
`PLOW_DSTEP_LOG=1`, `PLOW_DSTEP_EVERY=255`, `--engine-diagnostics`, and
`--parity-report` to `plowrt bench --random-input-len 8192 --output-len 256
--requests 1 --warmup-requests 0 --concurrency 1`. Raw artifacts:

- `/tmp/k3-d0-decode.clean.json`
- `/tmp/k3-d0-decode.stderr`
- `/tmp/k3-d0-decode.trace`, SHA256
  `a6709e7604cd251da8dddfcbbf94b8edd087d55ef44e655e2a6127581f66adfe`
- `/tmp/k3-d0-decode-trace-report.txt`
- `/tmp/k3-d1-compact.clean.json`
- `/tmp/k3-d1-k1-1024.clean.json`
- `/tmp/k3-d1-k4-1024.clean.json`

The temporary object view excluded stale optional KDA-family and lean-MoE
objects so an older companion-free packet takes its defined interpreter
fallback. The decode object was
`interp_decode_k3_gq.elf`, SHA256
`3ffe0786b0a3367ebe2a380154692e6d2920ec3b59e214f5b799372831d8763f`.
