# KDA carry key-factor precompute

Isolated gfx950 experiment for dense BT64 gated-delta carry. It is generic over token and
head counts; the oracle currently gates the production `D=V=128` recurrence.

The control recomputes `k * exp2(g_last - g)` in each of eight value tiles. The candidate uses
the existing Wu launch's spare grid to materialize its BF16 high/residual pair once per
`[token, head, key-channel]`, then consumes those exact operands without changing MFMA or chunk
recurrence order. It deliberately leaves Wu's four-way `beta * k * exp2(g)` recomputation in
place: separately materializing that operand costs more HBM traffic than it saves.

Run inside `nix develop`:

```sh
runtime/bench/amd/kda_carry_key_precompute/run.sh
```

`run.sh` fails if any gfx950 kernel spills, uses scratch, or falls below two waves/SIMD.

## MI355X production-path result

`T=8192, H=12, D=V=128`, 21 samples, one GPU:

- Exact: q, W, U, and output each have `0/12,582,912` bit mismatches; final f32 state has
  `0/196,608` bit mismatches.
- Control Wu + carry: `3.858436 ms` median.
- Wu-produced key factors + carry: `3.731595 ms` median, including both 50.33 MB factor
  stores; `-0.126841 ms` (`1.034x`).
- Candidate Wu: 134 VGPR, 65 SGPR, occupancy 3. Candidate carry: 138 VGPR, 81 SGPR,
  occupancy 3. Both have zero scratch/spills and use wave64 with 256 threads.

The two BF16 factor tensors are live only from Wu to carry. Integration must allocate them from
one reusable 50.33 MB scratch pair, not as 69 persistent per-layer tensors. At the current 69-layer
shape, the production-path result projects to about `8.75 ms` TTFT; it is useful but substantially below
the plan's original `100..130 ms` estimate.

## TP8 network qualification

Matched BF16-KV, 7650/7650 measured-TuneDB assets at 8192 tokens gave exact first-token
parity in three order-balanced folds. Control minus candidate TTFT was `4.607820`,
`5.141070`, and `3.847880 ms`: mean `4.532257 ms`, sample SD `0.649898 ms`. All folds
produced token `6896` and checksum `fnv1a64:7d749e3b002fafa7`. Raw trace attribution
measured Wu body `43.174 -> 41.376 ms` and carry body `156.093 -> 149.676 ms`; total
critical-chain span improved `1490.684 -> 1486.080 ms`.

The 8192-to-256 carried-state gate also passed: all 256 token IDs and checksum
`fnv1a64:6bdfaa7b84ee4e7e` match. Control versus candidate was
`1503.650667 -> 1499.005459 ms` TTFT, `44.365433 -> 44.335679 ms` TPOT, and
`12816.836247 -> 12804.603634 ms` end to end. A provenance audit later showed that this
runtime fell back to the ordinary interpreter because the standalone objects were absent.
With those objects present, an exact TP8 8192-to-1 trace regressed TTFT from `1500.858` to
`1605.464 ms`. The device span grew from `1486.153` to `1588.480 ms`: removing the 69 Wu
and 69 carry interpreter bodies saved `192.486 ms`, but their standalone launches added
`295.178 ms` of trace residual, a net `102.327 ms` device regression. The packet-side
key-factor math and segment topology remain enabled; standalone object production is
default-off. Set `PLOW_KDA_KEY_FACTOR=1` at object build time for explicit diagnostics.

Rejected variant: also materializing Wu's `beta*k*exp2(g)` gave an exact
`3.735592 -> 3.702952 ms` (`-0.032640 ms`) after paying for its third BF16 tensor and separate
precompute launch. The added HBM traffic consumed almost all of the arithmetic saving.
