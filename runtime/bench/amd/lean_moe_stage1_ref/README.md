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

The build also emits a Nix-ROCm-only HIP host driver at `$OUT/compare`. After static review,
the GPU comparison is explicitly enabled with `--run`:

```sh
nix develop -c perf-data/tools/gpulease -n 1 moe1-xcd8-wgm4-byte31 \
  /tmp/plow-moe1-audit/compare /tmp/plow-moe1-audit/shipping.elf \
  /tmp/plow-moe1-audit/candidate.elf --run
```

The driver uses T=8192, H=3584, I=384, E=896, top-k=16, BM64-padded sorted rows, production
`meta=[rowoff,count,tile_prefix]`, production row-major expert MXFP4 payload/E8M0 scale rows,
and SiTU `(4,25)`.
It requires byte equality for the full MXFP4 payload and E8M0 scale buffers, including untouched
pad rows. Timing is 31 samples per object, cache-flushed before every launch, with object order
alternating each sample. Index-sensitive device fill kernels initialize finite BF16 activations,
both FP4 expert branches, and E8M0 scales without staging the 1.2 GiB weights through host memory.
The 256 MiB cache buffer is read and rewritten by a compute kernel ordered immediately before the
timed event; no SDMA memset is accepted as a cache flush. Expect roughly 3 GiB of transient GPU
allocation.

For schedule exploration, `schedule_kernel.hip` exposes tile, workgroup, minimum-occupancy,
WGM/XCD, cache-hint, priority, and epilogue compile defines without instantiating unrelated
grouped-MoE entry points. The comparator's extended form accepts the candidate launch contract
and independent grids:

```sh
compare SHIPPING.elf CANDIDATE.elf --run CANDIDATE_SYMBOL THREADS LDS \
  SHIPPING_GRID CANDIDATE_GRID
```

It reports the arithmetic mean, median, minimum, and maximum of the 31 samples. See
`perf-data/gfx950-moe-stage1-schedule-screen-20260904.md` for the closed T8192 schedule matrix.

## Result: rejected

Measured on one leased MI355X GPU with the command above (`moe1-xcd8-wgm4-byte31-c2ed0c5`,
31 alternating samples, compute-cache-flushed before every launch):

| | Median | Min | Max |
|---|---:|---:|---:|
| shipping WG512/BN256 | 2.126738 ms | 2.113738 ms | 2.219937 ms |
| candidate WG256/BN128 | 3.841870 ms | 3.809469 ms | 3.930271 ms |

The exact oracle passed: 0 differing bytes out of 29,036,544 sorted MXFP4 payload bytes and
0 differing bytes out of 1,814,784 E8M0 scale bytes across 151,232 padded rows. The candidate
is nevertheless rejected: +1.715132 ms, or +80.65%, versus shipping. Its spill-free occupancy
does not recover the overlap lost when low-register staging synchronously converts and commits
one cell at a time. This object remains an isolated harness experiment; it is not a production
route or default.

## Reusable-A4 boundary

`build_reuse.sh` builds a gfx950 candidate that quantizes the sorted BF16 activation once, then
reuses its MXFP4 payload and E8M0 scales across 128-column N tiles. The four-wave WG256 GEMM keeps
only A in LDS and reads existing row-major expert tables into registers; it does not allocate a
second copy of expert weights. Eligibility is based on the MXFP4 contract, 128-aligned K, and at
least two shipping 256-column N tiles. The runtime owns one max-live scratch allocation shared by
sequential prefill segments, with rollback at `PLOW_MOE_STAGE1_A4_REUSE=0`.

```sh
nix develop -c runtime/bench/amd/lean_moe_stage1_ref/build_reuse.sh /tmp/plow-moe1-reuse
GPU_LEASE_DIR=/tmp/gpulease perf-data/tools/gpulease -n 1 moe1-a4-reuse \
  /tmp/plow-moe1-reuse/reuse_compare /tmp/plow-moe1-reuse/shipping.elf \
  /tmp/plow-moe1-reuse/reuse.elf --run
```

The comparator includes quant/sort and GEMM, alternates order over 31 compute-cache-flushed
samples, and requires byte-exact payload and scales. At T8192/H3584/I384/E896/top-k16 it measured
2.122398 ms shipping vs 1.728507 ms reusable A4 (means, -18.55%), with zero differences over
29,036,544 payload and 1,814,784 scale bytes. The candidate is wave64, four waves/WG, 182 VGPR,
44 SGPR, occupancy 2, private0, and VGPR/SGPR spill0.
