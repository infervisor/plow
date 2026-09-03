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

## MI355X result

`T=8192, H=12, D=V=128`, 21 samples, one GPU:

- Exact: W `0/12,582,912`, U `0/12,582,912`, output `0/12,582,912`, and final-state
  `0/196,608` bit mismatches.
- Control Wu + carry: `3.746070 ms` median.
- Wu-produced key factors + carry: `3.600029 ms` median, including both 50.33 MB factor
  stores; `-0.146041 ms` (`1.041x`).
- Wu producer: 130 VGPR, 61 SGPR, occupancy 3, zero scratch/spills. Candidate carry:
  139 VGPR, 89 SGPR, occupancy 3, zero scratch/spills. Both are wave64, 256 threads.

The two BF16 factor tensors are live only from Wu to carry. Integration must allocate them from
one reusable 50.33 MB scratch pair, not as 69 persistent per-layer tensors. At the current 69-layer
shape, the isolated result projects to about `10.1 ms` TTFT; it is useful but substantially below
the plan's original `100..130 ms` estimate.

Rejected variant: also materializing Wu's `beta*k*exp2(g)` gave an exact
`3.735592 -> 3.702952 ms` (`-0.032640 ms`) after paying for its third BF16 tensor and separate
precompute launch. The added HBM traffic consumed almost all of the arithmetic saving.
