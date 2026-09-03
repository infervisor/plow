# Lean MoE stage-1 WG256 candidate

This isolated harness compares the shipping Plow A4W4 stage-1 object against one bounded
gfx950 candidate. It does not alter runtime routing.

| | Shipping | Candidate |
|---|---:|---:|
| tile | BM64/BN256/BK256 | BM64/BN128/BK256 |
| wave/workgroup | wave64 / WG512 | wave64 / WG256 |
| schedule | linear persistent slices | XCD8 un-round-robin + grouped-M WGM4 |
| activation | runtime `moe_glu` | SiTU pair, beta=4, linear beta=25 |
| stage-1 output | sorted-row MXFP4 + E8M0 | identical contract |
| dynamic LDS | 119,808 B | 52,224 B (post-mainloop bridge aliases staging) |

The candidate retains Plow's generic bf16-to-MXFP4 conversion. Its low-register mode commits
one conversion cell at a time instead of carrying a full next tile across the MFMA block. The
fully overlapped WG256 form compiles at 205 VGPR/occupancy 2; forcing occupancy 3 spills 50
VGPR values. The bounded form is therefore intentionally a scheduling tradeoff for measurement,
not a claim that less overlap is faster.

Build and run static gates inside the development shell:

```sh
nix develop -c runtime/bench/amd/lean_moe_stage1_ref/build_candidate.sh /tmp/plow-moe1-audit
```

The build checks the exact symbols, wave/workgroup metadata, zero private segment, zero SGPR/VGPR
spills, candidate register occupancy >=3, scaled A4W4 MFMA ISA, no scratch instructions, exact
N=384 sorted payload/scale coverage, and XCD8/WGM4 bijection over ragged tile grids. With 160 KiB
LDS, 52,224 B permits three WG256 blocks per CU, so the candidate's effective ceiling is three
waves/SIMD (register ceiling four).

After static review, the GPU comparison is explicitly enabled with `--run`:

```sh
nix develop -c python3 runtime/bench/amd/lean_moe_stage1_ref/compare.py \
  /tmp/plow-moe1-audit/shipping.elf /tmp/plow-moe1-audit/candidate.elf --run
```

The driver uses T=8192, H=3584, I=384, E=896, top-k=16, BM64-padded sorted rows, production
`meta=[rowoff,count,tile_prefix]`, production row-major expert MXFP4 payload/E8M0 scale rows,
and SiTU `(4,25)`.
It requires byte equality for the full MXFP4 payload and E8M0 scale buffers, including untouched
pad rows. Timing is 31 samples per object, cache-flushed before every launch, with object order
alternating each sample. Expect roughly 3 GiB of transient GPU allocation.
