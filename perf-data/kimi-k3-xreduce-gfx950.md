# Kimi-K3 TP8 prefill collective — gfx950

Date: 2026-09-03. Eight MI355X GPUs, exclusive `gpulease`, ROCm 7.14.0.

The T1024 packet contains 92 one-shot gather/reductions, 94 full-width two-shot
reductions, and 92 half-width two-shot reductions. Every packet has 256 workgroups.
The measured two-shot shapes are therefore exactly `512*7168` and `1024*7168` bf16
elements. The benchmark calls `d_xreduce_twoshot_mega` through the runtime's real HSA
peer-visible allocation; it is not a model of the collective.

| shape | baseline | `PLOW_XR_AGG=1` | delta | full-vector TP8 parity |
|---|---:|---:|---:|---:|
| T512 | 15.107 us | 9.032 us | -40.2% | exact |
| T1024 | 19.007 us | 12.931 us | -32.0% | exact |

Both full prefill objects are 256 VGPR, occupancy 2, with 32 bytes private memory.
On the complete 8192-token network, three timed repetitions after one warm-up:

| arm | median | repetitions |
|---|---:|---|
| baseline | 2919.417 ms | 2919.303, 2919.417, 2920.086 |
| aggregate | 2911.529 ms | 2910.488, 2911.529, 2911.762 |

Net: **-7.888 ms (-0.27%)** at 8K. The isolated full-vector oracle proves numerical
identity; the production object changes only the gate-ag signal aggregation and retains
the final `nranks*nblk` counter value.

This is not the final per-XCD hierarchy. Followers still use a SYSTEM-scope arrival on
one rank-local line. Per-XCD relaxed arrivals plus one elected maintenance leader per XCD
require the placed packet's static domain counts and separate cache lines. Cross-GPU
acquires must remain outside that local leader election.

Reproduction:

```bash
nix develop -c bash scripts/build_tp_allreduce.sh /tmp/xr2-base
nix develop -c env PLOW_XR_AGG=1 bash scripts/build_tp_allreduce.sh /tmp/xr2-agg
perf-data/tools/gpulease -n 8 xr2-base sg render -c \
  '/tmp/xr2-base/tp_allreduce_prefill_bench 0 1 2 3 4 5 6 7'
perf-data/tools/gpulease -n 8 xr2-agg sg render -c \
  '/tmp/xr2-agg/tp_allreduce_prefill_bench 0 1 2 3 4 5 6 7'
```
