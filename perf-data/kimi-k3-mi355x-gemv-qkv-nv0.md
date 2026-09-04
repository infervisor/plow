# MI355X two-output `GemvQkv(Nv=0)` kernel gate

Status: **isolated candidate qualified; full-network candidate rejected**.

## Cell

- GPU: AMD MI355X (`gfx950`), one-GPU exclusive `gpulease`, rc=0.
- Shape: `M=1, K=7168, N0=128, N1=12`, BF16.
- Control: two complete `d_gemv` launches, 128 + 12 workgroups.
- Candidate: one `d_gemv_qkv` launch with `Nv=0`, 140 workgroups.
- Both: 512 threads/workgroup, wave64.
- Timing: 100 warmups/cell, 5000 iterations/cell, 8 order-alternated folds.
- Oracle: the candidate must be bit-identical to both control outputs and must not write the
  absent third output.

This is the exact TP8 KDA decode projection shape extracted by the whole-graph rule. The workgroup
counts are the emitter's blocked-column fixed points, not a full-256 synthetic grid.

## Result

| arm | mean (us) | sample SD (us) | median (us) | range (us) |
|---|---:|---:|---:|---:|
| two `Gemv` sequence | 7.8879 | 0.0106 | 7.8848 | 7.8800–7.9135 |
| `GemvQkv(Nv=0)` | 3.5032 | 0.0020 | 3.5028 | 3.5017–3.5080 |
| candidate − control | **−4.3847 (−55.59%)** | 0.0087 | −4.3820 | −4.4055–−4.3772 |

Correctness: 0/128 and 0/12 differing BF16 halves. The poisoned third output had 0 writes.

## Resources

Compiled with ROCm 7.14, `-O3 --offload-arch=gfx950 -DPLOW_GEMV_MM=1` and compiler resource
analysis. Metadata was read from the extracted gfx950 code object.

| kernel | VGPR | SGPR | spills (V/S) | private | LDS | occupancy |
|---|---:|---:|---:|---:|---:|---:|
| standalone `d_gemv` | 114 | 44 | 0 / 0 | 0 B | 147,456 B | 2 waves/SIMD |
| standalone `d_gemv_qkv(Nv=0)` | 74 | 51 | 0 / 0 | 0 B | 147,456 B | 2 waves/SIMD |

Both objects select wave64. `d_gemv_qkv` already selects the `K=7168` unroll-7 rung, exactly two
14-chunk passes. No wave or unroll switch is needed for this candidate.

## vLLM/ROCm inspection

Pinned vLLM 0.28 routes each bias-free `M=1, K<=8192` BF16 linear independently to `LLMM1`
(`vllm/model_executor/layers/utils.py`). The two projections therefore remain separate kernels;
there is no same-input multi-output equivalent in that path. The `wvSplitKrc` path is for batch
sizes 10–128 and is not the B1 comparator.

## Qualification boundary

The isolated cell includes one fewer HIP launch. Plow's interpreter does not launch one HIP kernel
per packet, so the exact network saving is determined by one fewer packet handoff and by whether
the two control packets overlap in the global queue. This result qualifies an exact full-token
order-alternated A/B; it does not qualify enabling the rule by default.

## Full-network gate — rejected

The exact TP8 BF16-KV 8192→256 gate ran at `62e7130` under one exclusive eight-GPU lease.
Both arms used the promoted paired MLA segment route, manifest-derived decode inventory pruning,
and no KDA specialist objects. The compiler used the current gfx950 TuneDB digest
`gfx950-870078e93f2c92f0` under `rocm-7.14.0-nix`: a fresh auto-derived 96-shape campaign wrote
960 oracle-passing raw rows, published 480 dispatch-arm records, and closed 96 HIT / 0 MISS.
Both packets then reported 7650 / 7650 measured tile selections.

The only structural program difference was the candidate fusion:

| arm | decode instructions | `Gemv` | `GemvQkv` | pairing hash |
|---|---:|---:|---:|---|
| control | 2165 | 816 | 0 | `0x1df8ef184df9a71c` |
| candidate | 2096 | 678 | 69 | `0x73141741f858b5db` |

Fold 1 was candidate→control. All 256 token IDs were byte-identical across arms and all-rank TP
agreement passed. The predeclared 0.20 ms TPOT improvement gate failed, so later folds were not
run:

| metric | control | candidate | candidate − control |
|---|---:|---:|---:|
| TTFT | 1504.884 ms | 1506.744 ms | +1.860 ms |
| TPOT | 29.926394 ms | 30.023260 ms | **+0.096866 ms** |
| E2E | 9136.114 ms | 9162.676 ms | +26.561 ms |

Artifacts are under `/tmp/k3-gemv-nv0-network-gate`; packet/object hashes are recorded in
`sha256.txt`. The command shape was:

```text
K3_FULL=1 PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix plowc \
  --hf-dir /home/shaswot/models/Kimi-K3 --max-ctx 16384 --n-cu 256 --num-gpus 8 \
  tune gemm --gpu mi350 --obj <current-object-dir> --samples <samples.jsonl> \
  --shapes auto --lease --campaign k3-gfx950-870078e-requal

gpulease -n 8 k3-gemv-nv0-network <paired-8192-to-256-gate>
```

The isolated launch saving does not transfer to Plow's persistent global queue. The two control
packets can overlap and retain narrower work distribution; combining them removes one handoff but
does not reduce the network critical path. Keep `--experiment-parallel-linear2` default OFF.
