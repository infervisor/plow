# KDA Intra wave-item experiment

This gfx950 experiment preserves the production BT64/D128 arithmetic order while assigning
independent `(chunk, head)` items to separate wave64 waves. Each wave owns one 64x64 FP32 LDS
matrix and performs the existing ten MFMA block pairs followed by the existing strict serial
forward substitution. It does not parallelize or reassociate the recurrence.

Run under the repository Nix environment:

```sh
nix develop -c runtime/bench/amd/kda_intra_wave_items/run.sh
```

The gate compares every FP32 Aqk/Ainv slot and then propagates both arms through q-precompute,
W/U, output, and final-state production. All comparisons are bitwise. The script also rejects
scratch, register spills, nonzero SGPR spills, occupancy below two waves/SIMD, candidate VGPR
usage above 96, or candidate LDS above 128 KiB.

At `T=8192, H=12, D=V=128`, 256 workgroups cover 1,536 items with six active item-owning waves
per workgroup. The recorded MI355X gate measured 1.741776 ms for production and 0.570285 ms for
the wave-item body, a 3.054x speedup. Aqk/Ainv, q, W/U, output, and FP32 final state were all bitwise
identical. The candidate compiled with 79 VGPR, 68 SGPR, occupancy 2, 131,072 B LDS, and no
scratch or register spills.

This directory is an isolated kernel gate. Production routing remains default-off until the
singleton object passes the full TP8 carried-state and network gates.
