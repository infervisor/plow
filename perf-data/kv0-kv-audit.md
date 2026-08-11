# KV-0 — KV-cache compressibility audit (P10 kv-zip, stage 0)

Real plow KV bytes, gemma-4-12B @ SM120, ctx 24584 (24576-token real-text
prompt + 8 gen), batch 1. Dumped with `PLOW_DUMP_KV` from `gemma4_sm120_chat`
(head-major ring bytes exactly as `d_headnorm_rope` wrote them, post
norm+RoPE, k_eq_v layers included). Two content domains: prose (repo
docs/plans) and code (runtime + crate sources), tokenized real text — NOT
repeated measure_tpot phrases, NOT random ids. 5.6 GB of KV per dump,
96 tensors (48 layers × K,V; 40 sliding kvh8/ring16384/hd256 + 8 full
kvh1/ring32768/hd512). fp8-e4m3 and nvfp4 twins derived offline with the
exact kernel schemes (per-row amax/448 RNE; block-16 amax/6).

Tooling: `perf-data/tools/make_kv_prompt.py`, `perf-data/tools/kv_audit.py`.
Raw numbers: `kv0-kv-audit-prose.{json,md}`, `kv0-kv-audit-code.{json,md}`.

## Headline (prose dump; code dump within 0.6% everywhere)

| class | H_exp (bits) | esc16 @per-tensor base | E slots (p<1e-6) | **sz12 net ratio** | fp8 esc8 | fp4 H_code | H_lo |
|---|---|---|---|---|---|---|---|
| full.K    | 2.70 | 0.025% | 6 | **1.293** | 2.61% | 3.976 | 7.97 |
| full.V    | 2.72 | 0.025% | 5 | **1.299** | 2.64% | 3.974 | 7.97 |
| sliding.K | 2.74 | 0.018% | 5 | **1.267** | 2.28% | 3.976 | 7.97 |
| sliding.V | 2.70 | 0.032% | 7 | **1.243** | 2.77% | 3.982 | 7.97 |

sz12 net ratio = 2·hd / (1.5·hd + 4·E), i.e. INCLUDES per-row fixed escape-slot
provisioning at the p(overflow)<1e-6 slot count. Worst observed per-row escape
count anywhere (10.5 GB, ~4.4 M rows): 8.

## Gate verdicts (bars set in the design notes §3 before measuring)

- **bf16 sz12 → KV-1 GO.** Net ratio 1.24–1.30 ≥ 1.20 on every class; escape
  0.017–0.032% ≤ 0.5%; row-overflow tail bounded at E ≤ 7 ≤ 8. The ring
  fixed-stride constraint is satisfiable with room to spare (12–4% of rows
  need zero slots; p(>4) ≈ 1e-5).
- **fp8 sz7 → KILL.** 3-bit exponent window escapes 2.3–2.8%/elem ≈ 13
  escapes/row mean at hd512 — no fixed slot count ≤ 64 bounds the tail, so no
  fixed-stride blob exists; and fp8 byte entropy 6.75–6.91 bits caps even an
  ideal entropy coder at 1.16–1.19×, under the VL kills. Per-row amax
  normalization did NOT concentrate the exponent field enough (H(exp4) ≈ 2.88
  vs weights' 2.6 — KV is wider, not narrower). Matches the weight-side fp8
  NO-GO (1.042, P9 C-3fp8).
- **fp4 → incompressible, datum recorded.** 4-bit code entropy 3.97–3.98/4
  bits → lossless ceiling 1.005–1.007×. nvfp4 KV needs no (and admits no)
  lossless stage. Roadmap: closed.
- **sz11 (3-bit window on bf16) → KILL.** esc8 ≈ 2.2%/elem ≈ 6–11 escapes/row
  mean; slot provisioning eats the 1.454 layout gain. sz12 (4-bit) is the code.
- **Domain stability → PASS.** prose vs code sz12 ratios differ ≤ 0.007×
  (bar 0.03); H_exp by ≤ 0.007 bits. Static codec constants are safe.
- **Positional stability → PASS (mild drift).** Full-layer H_exp rises from
  ~2.56 (first ring quarter) to ~2.77 (last quarter); escape behaviour
  unaffected. No per-position adaptation needed.

## Codec consequences for KV-1 (freeze)

1. **Base granularity: per-TENSOR.** The planned per-head/per-dim machinery is
   dead weight: H(exp|head) ≈ H(exp) (Δ < 0.05 bits), and per-dim buys only
   0.06–0.25 bits + esc 0.03% → 0.01% — neither moves the net ratio, both cost
   a decode-side table load. One u8 EXP_BASE per tensor (side scalar, like the
   fp8 scale plane). K/V asymmetry: none observed — same code both.
2. **Row blob (hd512 full layer):** 4-bit codes plane (256 B) + lo plane
   sign|mant7 (512 B) + E=6 escape slots (24 B: u8 dim-idx… n.b. hd512 needs
   u16 idx or 9-bit pack — freeze in KV-1) + escape count in the code plane
   (code 15). 792 B/row vs 1024 → 1.293×.
3. **Escape slots per class:** full 6, sliding 8 (round V's 7 up to keep one
   blob shape per hd). Overflow contingency (p<1e-6, max seen 8): per-row raw
   flag bit — row-uniform branch, never lane-divergent.
4. **Expected end-to-end ceiling** (@128k, 12B bf16, KV-read share ~28% of
   step): ratio ~1.27 avg → TPOT ceiling ≈ −6%; bit-exact. KV-4
   transfer/capacity: 1.24–1.30× fewer bytes, encode-once.

## Addendum: XOR-delta + per-block bit-packing (Gorilla/FastLanes family)

Follow-up question: treat the head-major slab as a u16 byte array, XOR each
element with a neighbor, bit-pack blocks of 128/256/512 at the block's max
significant width (+4-bit width header). Measured on the same dumps
(`kv_xor_audit.py`, 16 tensors/class, prose; `kv0-xor-audit-prose.json`):

| variant | B=128 | B=256 | B=512 |
|---|---|---|---|
| plain XOR (mem or seq axis) | **0.998×** | 0.999× | 1.000× |
| XOR + sign→LSB rotate, mem axis | 1.155× | 1.139× | 1.137× |
| XOR + rotate, seq axis (best) | **1.155×** | 1.144× | 1.137× |
| no delta, rotate only | 1.033× | 1.031× | 1.032× |

**KILL — 1.16× best vs sz12's 1.24–1.30×.** Three mechanisms, all measured:
1. Plain XOR packs to exactly 16 b/elem: sign disagreement is ~50%/pair, so
   every 128-block contains a bit-15 XOR and the block max-width is 16.
   Rotating sign to bit 0 fixes that but caps the family at ~1.16×.
2. Neighboring KV values are NOT numerically close (this is not a slowly
   varying time series): seq-axis XOR ≈ mem-axis XOR ≈ same gain, i.e. the
   delta exploits no sequential correlation — only the exponent-range
   concentration sz12 already captures per-plane, minus block-max slack
   (one wide element widens 127 neighbors).
3. XOR **increases** stream entropy: H(xor hi)=3.02–3.10 + H(xor lo)≈8.0 =
   11.1 b vs raw 2.70+7.97 = 10.67 b. Even an IDEAL entropy coder on XOR
   deltas (1.44×) is below the raw plane-split Shannon bound (1.50×).
   Mantissa bits are i.i.d.-like; differencing them adds noise.
Decode-side it is also strictly worse: data-dependent block widths break the
compile-time row stride (worst-case provisioning cuts the net further), and
the XOR chain adds a lane-scan dependency sz12's direct decode does not have.

## Addendum 2: PolarQuant / TurboQuant — what transfers to lossless

PolarQuant (arXiv:2502.00527 / 2502.02617) quantizes RoPE'd keys as 2D
(radius, angle) sub-vector pairs; TurboQuant (arXiv:2504.19874) random-rotates
then scalar-quantizes. Both are LOSSY: polar conversion and rotation are
continuous transforms that do not round-trip bit-exact in bf16, so neither is
directly usable here (lossless requires a bijection on bit patterns).

The one lossless shadow of PolarQuant's geometry — a RoPE pair (x,y)=r(cosφ,
sinφ) has max(|x|,|y|) ≥ r/√2, so pair EXPONENTS could be jointly redundant —
was measured on the dumps (`kv_pair_audit.py`, half-split (i, i+hd/2) pairing
per `d_headnorm_rope`; `kv0-pair-audit-prose.json`):

| class | 2·H_marg b/pair | H_joint (RoPE pair) | saving b/elem | saving (adj-dim control) |
|---|---|---|---|---|
| full.K    | 5.399 | 5.395 | 0.002 | 0.001 |
| full.V    | 5.432 | 5.429 | 0.001 | 0.002 |
| sliding.K | 5.502 | 5.465 | 0.019 | 0.008 |
| sliding.V | 5.448 | 5.443 | 0.003 | 0.003 |

**DEAD: exponent pairs are independent to within 0.02 bits/elem**, and the
adjacent-dim control matches the RoPE pairing — no polar structure survives in
the exponent field. (TurboQuant's own premise — rotation is NEEDED to make
coordinates independent — is thereby confirmed on raw KV, consistent with the
XOR-delta kill and the per-dim-base non-result.)

Side-observation, unrelated to polar structure: a 7-bit top-128 pair LUT code
(pure fixed-length entropy packing of the product distribution) would give
11.5 b/elem layout, escaping 0.38–0.59%/pair → net ≈1.33× on hd512 (vs sz12's
1.29×) after E≈12 slot provisioning. +2.5% ratio for a smem LUT in the decode
loop, 10× the escape traffic, and an encode-side LUT build. NOT adopted for
KV-1 (complexity/risk vs the KV-2 gate); recorded as a refinement option if
KV-2 passes with margin.

## Addendum 3: fp8/fp4 as opaque bytes + fp8 pair-code (both closed)

**fp8 pair-code** (last fixed-length fp8 shape, `fp8_pair_check.py`): 7-bit
top-128 code on adjacent exponent pairs + raw sign/mant = 1.067× layout
ceiling; measured pair escape 0.54–0.59% ⇒ ~1.4 escapes/row at hd512 ⇒ E≈12
slots ⇒ net ≈0.99× — a LOSS. With the sz7 window kill and H_byte=6.9 b, every
fixed-length fp8 shape is now measured dead. fp8 lossless: CLOSED.

**Opaque-byte sweep** (`kv_bytes_compress.py`; zstd -3/-19, gzip, xz on
kernel-exact twins, no float semantics — VL codecs, so transfer-track-only
by construction):

| stream | zstd-3 | zstd-19 | xz-6 | note |
|---|---|---|---|---|
| bf16 (layers ≥10) | 1.27–1.28 | 1.28 | 1.41 | = sz12's 1.28: the codec captures what LZ finds |
| fp8 | 1.18–1.20 | 1.20 | 1.20–1.24 | ≈ order-0 bound (H_byte 6.9); < bf16 sz12 |
| fp4 packed (full) | 1.00 | 1.00 | 1.01 | confirms entropy verdict |
| bf16 kv.0.v ONLY | **5.42** | — | — | layer-0 V ≈ token-embedding lookup; text repeats tokens ⇒ repeated 512 B rows |

The kv.0.v outlier is depth-local (layers 10/20/30/40/46: 1.27–1.28×) and
K never shows it (RoPE stamps position into every row). Fixed-stride sz12
cannot exploit cross-row matches by design; noted for KV-4: transfer of
layer-0 V slabs gets ~5× from plain zstd, and an fp8-KV wire could take
~1.2× from zstd where sz12 does not apply. No resident-KV implications.

## KV-1 result: KVZIP-SZ12 v1.2 frozen, oracle PASS

Layout (`kvzip_oracle.py` docstring = normative spec): 4-bit code plane +
lo plane + u32 header {per-ROW base, nesc} + 9×3 B escape slots, 32 B tail,
800 B/row hd512 (1.280×), 416 B/row hd256 (1.2308×), 16 B-aligned, no
model-side constants. Oracle over BOTH dumps (192 tensors, 21 GB): bit-exact
round-trip on every tensor, zero slot overflows (worst row: 8), negative
controls (lo/code/header/slot corruption) all detected.
`kv1-oracle-{prose,code}.json`.

Two design iterations forced by the oracle — worth recording:
- v1 per-TENSOR base (what the KV-0 audit granularity data suggested)
  FAILED: escapes cluster in low-magnitude rows (max 13/row). The audit
  measured per-elem escape RATES on 16-value windows; reserving code 15
  shrank the window to 15 and the per-ROW tail — the actual constraint —
  blew up. Per-row max-anchored base also fails (V channel outliers, max 21).
- v1.2 per-row OPTIMAL 15-window: worst row 8 escapes in 21 GB; 9 slots fit
  the same 32 B tail as 3-byte records.

## Decision

KV-0 PASSED for bf16; fp8/fp4 tracks closed with clean kill data. KV-1
(codec freeze + oracle) PASSED. KV-2 (fused flash-decode A/B) ran and
**KILLED the resident-read track** — see `kv2-flashdec-sz.md` for the
numbers, mechanism, portability analysis, and the P10 conclusion. The
compression ratios in this file remain live as the KV-4 transfer/capacity
basis only.
