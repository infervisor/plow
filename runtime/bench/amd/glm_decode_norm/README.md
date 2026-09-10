# GLM batched decode normalization

`--glm-decode-norm-rows=true` / `PLOW_GLM_DECODE_NORM_ROWS=1` gives each batched
GLM RMSNorm and AddNorm row its own workgroup, capped by the device CU count.
It defaults to false. Existing row-strided kernels perform the same arithmetic;
prefill, KV writers and single-row decode remain unchanged.

The resident batch-20 trace found 155 AddNorm packets each processing all 20
rows in one workgroup. Their summed completion tails are 8.258–8.333 ms per
rank, which motivates the experiment but does not predict a serving gain.
The full candidate changes 79 RMSNorm and 155 AddNorm workgroup counts at each
decode rung 2/4/8/16/20. Other instruction operands remain identical.

## Numerical check

`kernels.hip` calls the existing AMD bodies. `check.py` compares one workgroup
with one per row for rows 1/2/3/4/7/8/16/20 and features 2048/6144, including
optional gamma, in-place residual addition and 512-byte output guards.
All 80 MI300X cases are bit-identical between schedules; maximum relative L2
against the FP32 reference is 0.002142. This check contains no timing.

Inside `nix develop`:

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -shared -fPIC \
  -Iruntime/amd -Iruntime/common runtime/bench/amd/glm_decode_norm/kernels.hip \
  -o /tmp/glm-norm.so
```

Run `check.py --library /tmp/glm-norm.so --out /tmp/glm-norm.json` with the
ROCm PyTorch environment and an exclusive GPU lease.

The emitter regression covers arithmetic operands, row coverage, consumer
completion counts and byte-identical batch-1 output. All 38 GLM emitter tests,
23 configuration tests and 123 packet tests pass. The existing configuration
lint `no_raw_env_reads` fails on the pre-existing `PLOW_GLM_DSA_PF_SPAN` and
`PLOW_GLM_DSA_PF_DEXACT` reads; it is excluded from the 23-test run.
Both full-model packets pass Lean ordering and LDS checks for all ten programs,
and disabling the option reproduces the qualified resident packet byte-for-byte.

## Serving comparison

The matched 70k/700/.14/C20 seed-0 comparison completed on all eight MI300X GPUs.
Both arms passed 20/20 requests and 18/18 retrieval checks, with identical
per-request input/output lengths (1,414,538 input and 13,795 output tokens).

| Metric | Off | On | Change |
|---|---:|---:|---:|
| Output tokens/s | 47.586 | 48.889 | +2.74% |
| Mean TPOT, ms | 243.640 | 240.411 | -1.33% |
| P99 TPOT, ms | 393.541 | 350.170 | -11.02% |
| Median ITL, ms | 115.037 | 109.487 | -4.82% |

This is one ordered on/off pair, with the same frozen runtime and all 60 code
objects; only normalization workgroup counts differ. It does not establish
repeatability or the 100-request H200 target. The option remains default-off.
Source, artifact hashes, quality results and metrics are in
[mi300x-serving.json](mi300x-serving.json).
