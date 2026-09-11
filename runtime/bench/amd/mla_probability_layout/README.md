# MLA V2 probability layout

`PLOW_MLA_PF_PSWZ=1` permutes each 32-element BF16 probability row with
`col ^ ((row & 3) << 3)`. All probability producers and the PV fragment load
use the same permutation. The default is zero.

The gfx942 address model in `../lds_review/check.py` predicts two distinct
words per bank in the original PV `ds_read_b128` phases and one with the
permutation. It follows the lane groups documented by
[AMD's CK-Tile bank-conflict analysis](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html).
This is an address prediction, not a hardware counter measurement. It does
not establish the performance of probability stores or the full kernel.

The same review found a correctness defect in dense FP8 tail tiles. Masked
positions could read an uninitialized KV scale; zero probability multiplied
by a NaN scale then contaminated the output. The dense scale lookup now maps
positions beyond the live KV bound to a valid scale entry, as the gathered
path already does. KV scale allocations are not guaranteed to be zeroed.

## Numerical coverage

`kernels.hip` calls the actual V2 body with 512 latent and 64 rope dimensions,
eight heads, 256 threads and 304 workgroups. It covers dense/gathered and
BF16/FP8 KV, with BF16 or FP8 rope in the FP8 case. Separate library namespaces
prevent template-symbol interposition between the two layouts.

`check.py` checks six row/context/split/window configurations for each mode,
with both deferred and ordinary softmax builds: 72 cases total. Inactive FP8
bytes and scales are poisoned. Every output is compared exactly between
layouts, with finite-value checks, 512-byte output guards and three poisoned
output reuses. A normalized FP32 oracle checks sampled query rows at relative
L2 below 0.01. It is not a full CPU attention oracle.

Build each pair in `nix develop`, substituting `DEFER` with zero and one:

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -shared -fPIC \
  -DPLOW_MLA_PF_SV=1 -DFA_MLA_PF2_DEFER=DEFER -DPLOW_MLA_PF_PSWZ=0 \
  -Iruntime/amd -Iruntime/common \
  runtime/bench/amd/mla_probability_layout/kernels.hip -o /tmp/p-off.so
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -shared -fPIC \
  -DPLOW_MLA_PF_SV=1 -DFA_MLA_PF2_DEFER=DEFER -DPLOW_MLA_PF_PSWZ=1 \
  -Iruntime/amd -Iruntime/common \
  runtime/bench/amd/mla_probability_layout/kernels.hip -o /tmp/p-on.so
python runtime/bench/amd/mla_probability_layout/check.py \
  --off /tmp/p-off.so --on /tmp/p-on.so --out /tmp/p-numerics.json
```

Use a ROCm PyTorch environment for the checker. `--timing` checks twelve
larger cases and collects twelve alternating AB/BA graph replay samples.
Those timings exclude union construction, quantization, merge, interpreter
dispatch and tensor parallelism.

## Serving qualification

The campaign at `/tmp/tp-glm53-pswz` uses fresh TP8 servers in
safe/swizzle/safe order. Both arms include the dense tail fix, row-parallel
normalization and automatically selected decode tiers. Exactly one object,
`interp_flash_fp8kv_gq.elf`, differs between arms. Full interpreter resources
are unchanged: 512 VGPR, 256 AGPR, 58,376 bytes LDS and 1,552 private bytes;
notes report 114 SGPR spills and zero VGPR spills.

Each server runs 18 long retrieval checks, then 20 random requests at
70,000 input / 700 output tokens, range ratio 0.14, concurrency 20 and seed
zero. Speculation is absent. Prefix caching is enabled; TP unified token
batching is unavailable. Prefill chunks are 8192 rows with
`PLOW_PF_INTERLEAVE=0`.

After every arm terminates, `record.py ROOT --out RESULT.json` validates
frozen source/object/runtime/packet hashes, numerical library provenance,
request success, retrieval checks and identical per-request token counts.
It records the swizzle difference against the mean of the two controls and
the control drift separately. This campaign cannot establish a full100
speedup or H200 parity.

Completed results are in `mi300x-results.json`:

| Arm | Output tok/s | P99 TPOT ms | Retrieval | Requests |
|---|---:|---:|---:|---:|
| Original layout, tail fixed | 49.62 | 370.96 | 18/18 | 20/20 |
| XOR layout, tail fixed | 48.64 | 375.43 | 18/18 | 20/20 |
| Original layout repeated | 48.39 | 388.80 | 18/18 | 20/20 |

The candidate is 0.73% below the mean control throughput; the controls differ
by 2.48%. No serving improvement is established. Keep the XOR layout off.
All 72 numerical cases pass. The dense tail guard remains a correctness fix.
