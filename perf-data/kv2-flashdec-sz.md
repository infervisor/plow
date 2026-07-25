# KV-2 — fused sz12 flash-decode A/B at real geometry: **KILLED (0.35–0.69×)**

RTX PRO 6000 Blackwell (188 SMs), 26B full-attn geometry (nh16 nkv2 gqa8 hd512),
1 block/SM `__launch_bounds__(256,1)`, real dumped KV rows (12B kv.47, 24584
valid rows tiled; 12074 escape rows in the 32k cache), best of 50.
Harness: `runtime/tests/flashdec_fp8_bw_sm120.cu` (sz arm + bit-exact gate);
codec arm: `fa_sz12_dec8` + `SZKV` template flag in `op_attention.cuh`
(default-off, shipped instantiations byte-identical).

## Gates

- ptxas: **PASS** — sz GF4 141 regs / GF8 174 regs, 0 spill (bf16: 128/171;
  envelope 222/0).
- Numerics: **PASS** — sz12 output BIT-EXACT vs bf16 arm, GF4 and GF8, with
  12k+ escape rows in the cache (codec v1.2 + inline decode correct on GPU).
- Perf: **FAIL/KILL** (bar: ≥1.10× @128k, ≥1.0× @32k; kill <1.05× @128k):

| ctx | bf16 GF4 ms | sz12 GF4 (×) | bf16 GF8 ms | sz12 GF8 (×) |
|---|---|---|---|---|
| 32k  | 0.114 | 0.427 (**0.267×**) | 0.135 | 0.311 (**0.434×**) |
| 64k  | 0.340 | 1.017 (0.334×) | 0.425 | 0.598 (0.711×) |
| 96k  | 0.508 | 1.429 (0.355×) | 0.544 | 0.843 (0.646×) |
| 128k | 0.681 | 1.968 (**0.346×**) | 0.841 | 1.225 (**0.686×**) |

Marginal KV cost 32k→128k (ns/tok/layer): bf16 5.8 (GF4) / 7.2 (GF8);
sz12 15.7 / 9.3; fp8 3.0 / 3.7.

## Mechanism (why this is a real kill, not a tuning miss)

sz12 GF4 issued-BW is FLAT at ~410–490 GB/s across 32k–128k — the signature
of an instruction-rate ceiling, not a latency/staging problem (the plan's
"wider unroll / cp.async-staged planes" hedge addresses latency exposure;
it cannot add ALU throughput). Decode is ~6–7 SASS ops/elem (nibble extract,
base add, two byte-ops, shift-or reassembly) on top of GF×1 fma/elem — at
GF4 that ~3.5×es the per-byte instruction count for only 1.28× fewer bytes.
The fp8 comparison bounds the family: fp8's ~3 ops/elem for 2.0× fewer
bytes only breaks even around 32k on this part — sz12 pays ~2× fp8's ops
for 0.64× fp8's byte savings. Even a prmt-optimized decode (~3 ops/elem
floor) projects to ~0.9× at GF8 — still a loss. There is no ctx at which
this crosses 1.0 on GB202.

Same kill class as C-1 (warp-GEMV sz12, 0.50–0.88×) and ZG-2 (ZipGEMM,
0.31–0.95×): plow's memory-bound kernels sit close enough to the achievable
byte rate that fixed-length recon math loses more issue slots than the byte
savings return. The fp8-KV win survives because its decode is ~3 ops for 2×
bytes — that ratio is the ceiling of what fuses profitably here.

## Consequences

- KV-3 (e2e wire into `d_headnorm_rope`/flash + PLOW_KV_SZ flag): **DROPPED**
  — nothing to wire; the resident-read win does not exist on this hardware.
- KV-5 (prefill fusion): dead by the same mechanism, stronger (tensor-core
  path is even more issue-bound; ZG-2 already measured it).
- **KV-4 (transfer + capacity) is unaffected and remains the live track**:
  encode-once wire blobs at 1.28×/1.23× (+ zstd ~5.4× on layer-0 V), and
  standalone decompress-on-arrival runs grid-OVERSUBSCRIBED — the regime
  where sz measured 1.28× (c1 GRID=1020) — not 1-block/SM. The codec, host
  oracle, and the (default-off) `fa_sz12_dec8` device decode all carry over.
- The `SZKV` arm stays in-tree for KV-4's standalone-kernel experiments;
  default build byte-identical (flag never set by any packet).

## P10 conclusion (stages KV-0..KV-2)

**No runtime perf win.** The headline hypothesis — lossless KV decompression
fused into flash decode buys long-ctx TPOT — is dead on GB202 with clean
data (0.35–0.69×, instruction-bound), and the mechanism strengthens on
H100/B200/MI300. Nothing here changes a shipped kernel or a default build.

What the experiments DID buy:
1. **Five decision-space closures, all measured, none judgment calls**:
   fused sz12 decode (this file); fp8 lossless (sz7 window, pair-code 0.99×,
   opaque-byte 1.2× order-0-bound); fp4 lossless (H=3.98/4 b); XOR-delta +
   block bit-packing (≤1.16×, entropy-increasing); joint/pair exponent
   coding (≤0.02 b/elem — PolarQuant's geometry does not survive into
   exponents). Future "why don't we just compress the KV cache" proposals
   land on this file.
2. **A validated KV-4 (transfer/capacity) basis**: real-KV lossless ratios
   1.280×/1.231× (hd512/hd256) with a frozen, oracle-proven, GPU-bit-exact
   codec; Deflate 1.25× (= B200 Decompression Engine for free); layer-0 V
   5.4× under zstd; fp8-KV wire ~1.2× zstd-class. Effective-interconnect
   and prefix-cache-capacity numbers for the disaggregated-serving design
   are now measured, not estimated.
3. **Reusable tooling**: PLOW_DUMP_KV (any future KV analysis), the audit
   stack (kv_audit/kv_xor_audit/kv_pair_audit/fp8_pair_check/
   kv_bytes_compress), the host oracle, and a default-off bit-exact device
   decode (`fa_sz12_dec8`/SZKV) ready for a standalone oversubscribed
   decompress-on-arrival kernel if KV-4 is pursued.

## Portability (MI300X / H100 / B200)

The kill is governed by SM issue slots per DRAM byte at full BW (~4 inst/clk
per SM/CU): GB202 ≈ 1.25 ops/byte budget, MI300X ≈ 0.5, H100 ≈ 0.3, B200 ≈
0.13 — sz12's inline decode needs ~3.8 ops/blob-byte, i.e. ~3× over budget
on GB202 (= the measured 3× GF4 slowdown) and 8–30× over on the datacenter
parts. Bandwidth-rich parts are issue-poor per byte: the fused kill gets
STRONGER on H100/B200/MI300, and occupancy can't help inside a 1-block/SM
megakernel. DMA engines (TMA, SDMA) move bytes, they don't transform them —
irrelevant to fused decode, useful for KV-4 overlap. B200's Decompression
Engine (Snappy/LZ4/Deflate, mem-to-mem, no SM cost) is on the wrong side for
resident reads (would need >8 TB/s inflating the ring every token) but is
exactly the KV-4 arrival-side consumer: our opaque-byte sweep measured
Deflate (gzip-6) at 1.25–1.26× on real bf16 KV, so B200 gets ~1.25×
effective interconnect for KV migration with ZERO kernel work — a custom
sz12 oversubscribed decoder only marginally beats that (1.28×). Caveats
that do NOT port: the kill assumes the megakernel's pinned occupancy — a
high-occupancy (vLLM-style) flash decode hides decode ALU far better (C-1
measured this family 1.28× FASTER at GRID=1020); and fp8-KV's own fused
dequant win (3 ops for 2×) likely thins out on H100/B200 for the same
ops-per-byte reason. KV-0's compressibility statistics are properties of
the model, not the GPU, and carry everywhere.
