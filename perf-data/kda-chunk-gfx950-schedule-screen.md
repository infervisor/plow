# gfx950 KDA chunk schedule screen

Date: 2026-09-03

Shape: `T=8192 H=12 D=128 V=128`, bf16 KDA state, 64-token chunks. Runs were
serialized with `perf-data/tools/gpulease` and passed the existing numerical oracle unless marked
invalid. Timings are isolated HIP-event kernel times from `kda_step_cdna3_test.hip`.

## Baseline

The shipping geometry is eight wave64 waves per workgroup. Devgen emits carry only on
`min(n_cu, H * ceil(V / 16)) = 96` slices, so the apparent 160 empty workgroups in a 256-block
standalone launch are not present in the production stream.

| grid | prefix ms | intra ms | WU ms | carry ms | chunk total ms |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.4719 | 4.9767 | 1.3431 | 3.6873 | 10.5283 |
| 128 | 0.3598 | 3.7367 | 1.0131 | 3.6856 | 8.8421 |
| 192 | 0.2508 | 2.4948 | 0.6880 | 3.6819 | 7.1421 |
| 256 | 0.1966 | 1.8786 | 0.5256 | 3.6888 | 6.3175 |
| 384 | 0.2274 | 2.4981 | 0.6831 | 3.6962 | 7.1556 |
| 512 | 0.2030 | 1.8721 | 0.5230 | 3.6891 | 6.3497 |

Conclusion: 96 slices is already the correct carry count. The other chunk ops need 256 effective
slices; one common reduced grid is a regression.

Baseline resources:

| kernel | VGPR | SGPR | VGPR spill | SGPR spill | private bytes | wave |
|---|---:|---:|---:|---:|---:|---:|
| `k_intra_full` | 197 | 96 | 0 | 0 | 0 | 64 |
| `k_carry` | 248 | 106 | 0 | 0 | 160 | 64 |

## Rejected schedules

| candidate | result | reason |
|---|---|---|
| four waves, grid 256 | carry 4.6494 ms; intra 1.9100 ms; WU 0.9704 ms | Correct, but carry is 26.2% slower and WU is 85% slower than the eight-wave repeat. The state-update phase uses all eight baseline waves even though the first two carry phases use four. |
| six waves, grid 256 | output relL2 0.57684; state relL2 0.67683 | Invalid. KDA reductions support the power-of-two four/eight-wave geometries. |
| force-inline carry | carry 3.6850 ms at grid 96; VGPR 156; SGPR spill 46; private 0 | No speedup and violates the zero-spill gate. Outlining instead has zero register spills but a 160-byte call frame. |
| one interpreter opcode per object | carry VGPR 248 / SGPR spill 330 / private 204; other ops SGPR spill 268-360 | Per-op dead-code isolation alone does not clear the dispatcher ABI/live-range spill problem. |

No runtime kernel change is promoted from this screen. A carry improvement requires restructuring
its argument/live ranges or splitting phases; changing the grid or wave count cannot remove the
current floor.

## Reproduction

```sh
nix develop -c scripts/bench_kda_chunk_gfx950.sh
KDA3_BENCH=1 KDA_BENCH_T=8192 KDA_BENCH_GRID=256 \
  /tmp/plow-kda-chunk-bench /tmp/plow-kda-chunk-bench.co
KDA_CHUNK_WAVES=4 nix develop -c scripts/bench_kda_chunk_gfx950.sh
```
