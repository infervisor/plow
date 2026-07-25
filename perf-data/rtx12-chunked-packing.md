| workload | kind | R-hist (R=1 / R≥2) | %R≥2 | meanR | maxR | tok/s | ITL p99 ms | TTFT p99 ms | pf wall% | reqs ok/fail |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off-u4k-vu1 | throughput | 217 / 0 | 0 | 1.00 | 1 | 35.9 | 21.9 | 812 | 22 | 26/0 |
| off-u4k-vu4 | throughput | 267 / 32 | 11 | 1.11 | 2 | 84.0 | 59.0 | 2390 | 36 | 60/0 |
| off-u4k-vu8 | throughput | 302 / 54 | 15 | 1.15 | 2 | 112.5 | 73.8 | 4645 | 47 | 81/0 |
| off-u4k-vu16 | throughput | 323 / 61 | 16 | 1.16 | 2 | 112.9 | 78.9 | 12795 | 57 | 81/0 |
| off-u4k-vu32 | throughput | 373 / 68 | 15 | 1.15 | 2 | 111.6 | 76.2 | 29586 | 54 | 81/0 |
| off-u512-vu1 | throughput | 74 / 0 | 0 | 1.00 | 1 | 43.6 | 22.0 | 165 | 18 | 31/0 |
| off-u512-vu4 | throughput | 172 / 28 | 14 | 1.14 | 2 | 140.5 | 32.9 | 454 | 13 | 102/0 |
| off-u512-vu8 | throughput | 221 / 67 | 23 | 1.27 | 4 | 245.6 | 37.6 | 658 | 20 | 175/0 |
| off-u512-vu16 | throughput | 241 / 72 | 23 | 1.26 | 4 | 249.8 | 36.5 | 5081 | 20 | 179/0 |
| off-u512-vu32 | throughput | 246 / 78 | 24 | 1.28 | 4 | 244.7 | 38.0 | 14679 | 22 | 174/0 |
| c512q2048-u4k-vu1 | throughput | 330 / 0 | 0 | 1.00 | 1 | 34.9 | 21.9 | 917 | 24 | 25/0 |
| c512q2048-u4k-vu4 | throughput | 446 / 68 | 13 | 1.17 | 3 | 71.5 | 66.2 | 3327 | 41 | 51/0 |
| c512q2048-u4k-vu8 | throughput | 311 / 144 | 32 | 1.67 | 5 | 101.1 | 81.8 | 5706 | 52 | 72/0 |
| c512q2048-u4k-vu16 | throughput | 372 / 153 | 29 | 1.61 | 5 | 101.4 | 85.5 | 14420 | 56 | 73/0 |
| c512q2048-u4k-vu32 | throughput | 502 / 176 | 26 | 1.50 | 5 | 100.8 | 89.7 | 32117 | 60 | 73/0 |
| c512q2048-u512-vu1 | throughput | 76 / 0 | 0 | 1.00 | 1 | 43.5 | 21.9 | 165 | 5 | 32/0 |
| c512q2048-u512-vu4 | throughput | 137 / 32 | 19 | 1.24 | 3 | 132.6 | 34.1 | 459 | 12 | 95/0 |
| c512q2048-u512-vu8 | throughput | 193 / 49 | 20 | 1.44 | 6 | 216.6 | 39.4 | 740 | 18 | 155/0 |
| c512q2048-u512-vu16 | throughput | 132 / 83 | 39 | 1.76 | 6 | 235.2 | 36.8 | 5170 | 20 | 169/0 |
| c512q2048-u512-vu32 | throughput | 207 / 66 | 24 | 1.48 | 6 | 230.5 | 39.5 | 15260 | 22 | 165/0 |
| c1024q2048-u4k-vu1 | throughput | 239 / 0 | 0 | 1.00 | 1 | 35.3 | 21.9 | 868 | 23 | 25/0 |
| c1024q2048-u4k-vu4 | throughput | 393 / 32 | 8 | 1.08 | 3 | 73.6 | 51.0 | 2539 | 42 | 53/0 |
| c1024q2048-u4k-vu8 | throughput | 347 / 90 | 21 | 1.22 | 3 | 101.2 | 83.6 | 4763 | 54 | 73/0 |
| c1024q2048-u4k-vu16 | throughput | 381 / 98 | 20 | 1.22 | 3 | 101.9 | 86.9 | 13025 | 57 | 73/0 |
| c1024q2048-u4k-vu32 | throughput | 513 / 103 | 17 | 1.18 | 3 | 101.4 | 92.9 | 31510 | 57 | 73/0 |
| c512q1024-u4k-vu1 | throughput | 329 / 0 | 0 | 1.00 | 1 | 34.9 | 21.9 | 915 | 24 | 25/0 |
| c512q1024-u4k-vu4 | throughput | 533 / 47 | 8 | 1.08 | 3 | 73.3 | 50.1 | 2987 | 41 | 52/0 |
| c512q1024-u4k-vu8 | throughput | 391 / 184 | 32 | 1.33 | 3 | 99.2 | 87.6 | 5858 | 54 | 70/0 |
| c512q1024-u4k-vu16 | throughput | 455 / 194 | 30 | 1.31 | 3 | 99.1 | 72.1 | 15109 | 57 | 70/0 |

## VERDICT — honest NEGATIVE at B=8 (correctness-clean, perf regression)

Gates PASS (Gate A per-request identity + canary C0==C256==C512 byte-identical;
default PLOW_PF_CHUNK=0 => usize::MAX => shipped behavior unchanged). But finer
chunking is a net perf LOSS:
- 4k VU8: OFF 112.5 tok/s @ ITL p99 73.8ms -> c512q2048 101.1 @ 81.8ms (-10% tok/s,
  +ITL) despite meanR 1.15->1.67, maxR 2->5 (co-packing DID rise as designed).
- short 512 VU8 regressed 245.6 -> 216.6 tok/s.
- Every chunk/quantum config (c512q2048, c1024q2048, c512q1024) loses on tok/s.
Mechanism: chunking a 4k prefill into 512-row pieces multiplies launches AND makes
each chunk's flash re-read the request's growing KV prefix; prefill wall% rose
47-57% -> 52-60%. That overhead exceeds the weight-read sharing gain at the R (<=5)
that the B=8 blob allows.
UNTESTED lever: higher batch (B=16/32) to raise the R cap so weight-sharing can
outweigh the finer-chunk overhead — blocked here by the foreign 33GB server
(B=16 ctx8k blob = 66.6GiB won't co-fit). Recommend: keep PLOW_PF_CHUNK OFF by
default (it is); revisit only with a bigger batch blob, or attack the 4k prefill
cost directly (the O(n^2) full-layer flash re-reads, not the packing).
