# gfx950 decode: ASAP global-queue window order (`PLOW_GQ_ORDER=asap`)

Date 2026-09-04. Kimi-K3, 8x MI355X, TP8, BF16 KV. Per-workgroup decode-trace analysis
(`scripts/k3_trace_wg.py`, `scripts/k3_trace_critpath.py`) showed the shared-expert `GEMV_GLU`
"straggler" (38.5 µs max-min workgroup end) is head-of-line blocking: GLU depends only on
AttnRes but is queued behind the gated routed-expert slices, and the layer-closing XReduce waits
on the following GEMV in 186/186 layers (24.2 µs/layer on the true critical path). The fix is an
emit-time stable sort of each (segment, XCD) global-queue window by a unit-cost ASAP rank; streams,
counters and windows are otherwise unchanged (packet crate, unit test `gq_asap_order`).

Both arms: the requalified standalone-route packet (7,650/7,650 measured tiles, 233 decode
segments), same `hsaco-stack` objects; ASAP packet SHA256 `c2040cf8…` vs control `8882113e…`.
Runtime `--amd-tp-no-audit`, compact audit, one warm-up, three requests, alternating a/s/s/a/a/s.

| fold | control TPOT p50 | ASAP TPOT p50 |
|---|---:|---:|
| 1 | 27.655 ms | 27.455 ms |
| 2 | 27.716 ms | 27.494 ms |
| 3 | 27.730 ms | 27.520 ms |
| **mean** | **27.700 ms** | **27.490 ms (−0.210 ms, −0.76%)** |

8192→1 TTFT pair: 1260.98 → 1254.92 ms (−6.1 ms). Every run reproduced the identical checksums
(`fnv1a64:b7682a38c151ac99` at 256 tokens, `337f0f290d5ae157` at one token).

## Reading

The measured gain is a tenth of the 2.2 ms projected from the critical-path model: the queue
already overlaps most of the blocked GLU with other ready work, so the projection counted
hidden time. Still exact and free. Promote as the emit default for AMD placed packets after the
served cell; `PLOW_GQ_ORDER=stream` restores the previous order.
