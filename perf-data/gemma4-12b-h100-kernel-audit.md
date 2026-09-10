# Gemma 4 12B H100 kernel audit

Target: `google/gemma-4-12B-it`, checkpoint revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. These findings do not establish
that the entire prefill ladder is tuned or that Plow beats vLLM.

## Active implementation gaps

| Path | Implementation and limitation |
|---|---|
| BF16 prefill GEMM | SM90 WGMMA bodies use 128-byte operand swizzles. TMA requires valid tensor maps and the selected object/dispatch arm. Dedicated GEMM segments already exist; source availability does not prove every rung selects the best tile. |
| Packed HD256 attention | Q/K/V operands use 128-byte swizzled subtiles. The previous packed WGMMA route passed null descriptors, forcing asynchronous copies. The new opt-in `PLOW_NV_PACKED_FA_TMA=1` resolves descriptor pairs per request slot. Full tiles can use TMA; partial tiles retain asynchronous copies. Requires `PLOW_NV_PACKED_FA_WGMMA=1`. |
| HD512/BKV16 attention | Uses WGMMA and swizzled Q/K/V, but existing KV tensor maps have 32-row boxes. The BKV16 arm must retain asynchronous copies. Removing its eligibility guard would arm a barrier without issuing its expected transfers. |
| Attention probability tile | Uses a separate, unswizzled core-matrix layout; its dimensions differ from the Q/K/V 128-byte swizzle atom. No hardware-counter evidence yet certifies all its accesses. |
| W8A16 prefill | Dedicated WGMMA object exists, but the measured build has register spills and approximately 203 ms GEMM time per 1K chunk under segment instrumentation. It is not performance-qualified. |
| Decode | Small-M vector and tensor-core choices need separate tuning. Prefill TMA settings do not establish decode efficiency. |

The new descriptor route is disabled by default. It changes neither the packet
format nor the counter protocol. Descriptor tables are resolved inside the
existing SM90 work-item bodies; kernel math is reused.

## Actual vLLM launches

Captured vLLM 0.29 BF16 with Nsight Systems, CUDA graph node tracing, after
unprofiled warmup. Cases: 1K/16K input × C1/C16, two output tokens, no prefix cache.
These are diagnostic traces, not the output-128 serving comparison.

Observed kernel names include:

- `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT`
- Multiple additional `nvjet_sm90` tile/schedule variants and split-K reductions.
- `vllm_flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90` kernels.
- Triton fused GELU/multiply and residual/RMSNorm kernels.

At 16K/C1, the leading NVJET kernel family accounts for 407.49 ms (65.8% of
summed kernel duration); the leading attention family accounts for 101.71 ms
(16.4%). These totals include every launch in the capture, including decode.
They are not per-layer times or comparable directly to Plow's 1K segment totals.
Kernel names alone do not establish source-level TMA or bank-conflict behavior.

## Counter and correctness evidence

Nsight Compute was attempted with shared-memory load/store bank-conflict metrics.
The driver returned `ERR_NVGPUCTRPERM`. No bank-conflict counts were collected.
Nsight Systems does not substitute for those hardware counters. Plow's dependency
counter polling cost also remains unquantified; it must not be inferred from
shared-memory bank conflicts or the full duration of a FAT segment.

The interpreter currently polls dependency counters, synchronizes the CTA, and
publishes completion after the body. Its optional `PLOW_NV_TRACE` records
gate/body/signal cycles for block zero. That trace can expose a stalled block,
but is not an all-CTA critical-path measurement. An ordered-segment comparison
must preserve intra-segment dependencies and memory visibility; blindly dropping
counter operations is not a valid optimization experiment.

The packed attention test checks both absent and supplied descriptor tables,
HD256/HD512, noncontiguous reversed slots, ragged 65/33-row requests, 16K history,
sliding-window wrapping and zero padding against sampled FP64 attention.
All four cases pass: worst relative L2 0.002680, maximum absolute error 0.001174
(rounded upward). Compute Sanitizer memcheck reports zero errors.
The candidate server also passes concurrent-versus-isolated generation, ragged
requests, exact output limits, slot reuse, cancellation/recovery, and oversized
context rejection. This uses the candidate itself as the isolated control and
does not establish equivalence to another model implementation.

A warm-cache standalone screen (seven samples of ten launches, median) measured
HD256 at 84.85 us without maps and 81.54 us with maps. HD512 remains on the copy
fallback (2767/2781 us). This small, fixed ragged workload does not qualify a
serving improvement or the full ladder.

The candidate serving diagnostic (output2, one warmup and one measured repeat)
reports TTFT 96.89 ms at 1K/C1, 920.63 ms at 1K/C16, 1457.27 ms at 16K/C1,
and 25442.44 ms at 16K/C16. All reported cache counts are zero. The large vLLM gap
persists; keep this candidate opt-in. There is no interleaved serving A/B against
the previous packed object, so these results do not establish a regression or win.

## Implementation direction

Reuse Plow's validated packet, graph and segment machinery. Specialize the
compute objects by architecture, dtype and shape. Evaluate CUTLASS/CuTe-style
TMA producer/consumer pipelines, warpgroup register budgets, tile shapes and
epilogues against the exact emitted Gemma shapes. Existing Plow SM90 probes already
implement several of these techniques; avoid creating another unmeasured copy.
No new CUTLASS backend has been integrated by this change.

Required before promoting further changes: isolated body versus segment timings,
dependency gate/body/signal attribution, per-rung shape measurements, descriptor
contract support for HD512, and repeated serving/quality checks. AMD and CPU
remain separate compute backends sharing the execution contract.

References: [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
