# MI355X TP8 prefill XReduce workgroup sweep — 2026-09-04

Eight MI355X GPUs, one exclusive `gpulease -n 8`, ROCm 7.14.0. The focused object
was built with the production `PLOW_XR_AGG=1` protocol; ELF SHA-256:
`b302232464af0310bc0518bca412f4461b07d7f897d85e97b84c3c26d8cf08e7`.

Each cell is the median of three independent benchmark processes, each timing 25
in-kernel collectives. The three hot classes match the T8192 graph: 92 half-width
plain collectives, 94 full-width plain collectives, and 92 full-width collectives
with a folded 896-column gather. The weighted projection is
`(92*half + 94*full + 92*gather) / 1000`; it is an isolated-kernel comparison, not
a production TTFT estimate because the production mega-interpreter has a different
resource envelope.

| workgroups | half plain, µs | full plain, µs | full + gather, µs | weighted, ms |
|---:|---:|---:|---:|---:|
| 64 | 115.738 | 230.611 | 354.372 | 64.928 |
| 80 | 93.635 | 185.286 | 284.591 | 52.214 |
| 96 | 78.231 | 155.092 | 237.425 | 43.619 |
| 128 | 60.821 | 119.434 | 178.249 | 33.221 |
| 160 | 51.530 | 99.535 | 144.846 | 27.423 |
| 192 | 44.942 | 85.887 | 122.089 | 23.440 |
| 224 | 40.898 | 78.133 | 108.319 | 21.072 |
| **256** | **38.487** | **72.976** | **96.434** | **19.272** |

All 72 measured cells reported full-vector TP8 parity `PASS`, `timeout=no`, and
`bad=0`. The closest alternative, 224 workgroups, regresses half/full/gather by
6.26%/7.07%/12.32% and the weighted projection by 9.34%. The AITER-like 80-WG cap
regresses the three classes by 143.29%/153.90%/195.11%.

Decision: retain 256 workgroups. Reject a generic lower `xr_cus` cap for these
gfx950 prefill shapes. AITER's 80-WG choice is coupled to its wave-per-peer,
16-byte/LDS schedule and is not transferable to Plow's current scalar body in
isolation.

Raw evidence remains in `/tmp/xr-nwg-sweep-20260904-003/{raw.txt,results.json}`.
The committed harness is `46cfe94`; its artifact-relative runner fix is `8e11c3b`.
