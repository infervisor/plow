# Gemma FP8-weight / BF16-activation tensor-core probe

Experimental H100 projection kernel. E4M3 weights convert exactly to BF16 in
registers before native `mma.sync.m16n8k16`; activations remain BF16. Each channel's
FP32 scale is applied after the K reduction. Split-K uses a separate FP32 partial
buffer and deterministic reduction. No activation quantization or default change.

Build from the repository root with the installed CUDA toolkit:

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -arch=sm_90a -O3 -std=c++17 -I runtime/common -I runtime/nvidia \
  runtime/nvidia/experiments/gemma_fp8_w8a16_probe.cu -lcuda -o /tmp/gemma_fp8_w8a16_probe
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  /tmp/gemma_fp8_w8a16_probe
```

Optional arguments select one shape: `M N K [packet-blocks]`; K must be a positive multiple of
256. Exact integer cases always run first, including ragged M=17/N=71. The
standalone control calls production `d_gemv_fp8` with `GV_MM_MAX=16`, matching
the campaign's FP8 decode configuration. It measures 132/264/528 control CTAs
and split-K 1/2/4/8. The candidate's actual grid is
`ceil(N/64) × ceil(M/16) × splits`; the printed `blocks` field applies only to
the production control.

For a packet-derived sweep, run `w8a16_decode_ladder.py AUDIT_JSON PROBE OUTPUT_DIR`.
The output directory must be new. It tests every distinct plain `GemvFp8`
shape and the emitted instruction's CTA count, preserving program/PC references.
Fused GLU, BF16 head and attention require separate probes.

`--cubin MAIN_CUBIN` additionally loads the production interpreter. Build the
probe with matching `PLOW_NV_FP8_RB` and `GV_MM_MAX` settings. The interpreter's
embedded arena also selects the staged standalone path where K fits. Two extra
rows use `split=16` for queue plus dependency edges and `split=32` for queue
without edges; these are diagnostic mode identifiers, not split-K kernels.
Both launch132 persistent CTAs, while `blocks` records instruction slices.
The probe checks completed counters and numerical outputs. Resets occur before
the timing event. Standalone vs interpreter also changes grid, register pressure
and scheduling, so their difference is not pure counter overhead. Only the two
loaded modes isolate the dependency-edge change in this one-op program.

H100 80GB, CUDA 13.2, 2026-09-09: median of 15 CUDA-event measurements, 256MiB
cache flush before each, four warmups, no other GPU jobs or CPU builds during
timing. Times include the separate split reduction. Each row selects the best
measured control and candidate configuration, so this is a tuning screen.

| Batch | N | K | Native GEMV µs | Candidate µs | Split-K | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8192 | 5376 | 100.736 | 60.064 | 4 | 1.68× |
| 8 | 21504 | 5376 | 197.120 | 148.800 | 4 | 1.32× |
| 8 | 5376 | 21504 | 530.432 | 161.312 | 8 | 3.29× |
| 16 | 8192 | 5376 | 143.072 | 80.992 | 4 | 1.77× |
| 16 | 21504 | 5376 | 345.632 | 202.400 | 8 | 1.71× |
| 16 | 5376 | 21504 | 518.176 | 222.848 | 4 | 2.33× |
| 32 | 8192 | 5376 | 266.816 | 151.936 | 1 | 1.76× |
| 32 | 21504 | 5376 | 657.984 | 385.536 | 8 | 1.71× |
| 32 | 5376 | 21504 | 1007.168 | 436.384 | 8 | 2.31× |

All seven variants pass 1719 exact integer outputs and 257 sampled FP64
reference outputs per large shape. Full outputs are finite. For the selected
candidates, relative L2 difference from native GEMV is at most 6.42e-5 and
maximum absolute difference is 0.015625. Changed accumulation order means this
is not bit-exact GEMV. The candidate uses 34 registers and no shared memory or
spills; the standalone control uses 91 registers and no spills.

These are synthetic projection results, not model-quality, persistent
interpreter, cuBLASLt, vLLM, or serving results. Production instruction slice
ownership, scratch allocation, fused GLU behavior, sparse batches, and full-model
numerics still require integration and qualification. The standalone control
also has different register pressure from the full interpreter.

Compute Sanitizer memcheck passed all seven variants for both integer/tail cases
and M=16/N=8192/K=5376 with zero errors. Sanitizer timings are excluded above.

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  /tmp/gemma_fp8_w8a16_probe 16 8192 5376
```

Raw campaign artifacts under
`/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/`:
`gemma-fp8-w8a16-{build2.log,screen.log,screen.json,memcheck.log}`.
