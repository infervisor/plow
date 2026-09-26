p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/runtime/nvidia/op_attention.cuh'
s = open(p).read()

a = """#ifndef PLOW_NV_FA_WPR_RB
#define PLOW_NV_FA_WPR_RB 2"""
b = """/* HALF-WARP-PER-ROW scores at D=128 (Llama/Qwen head size): 16 lanes x 8 elems cover a 256 B
 * row, so one warp load scores TWO coalesced rows. Opt-in; 0 keeps the per-thread body. */
#ifndef PLOW_NV_FA_WPR128
#define PLOW_NV_FA_WPR128 0
#endif
#ifndef PLOW_NV_FA_WPR_RB
#define PLOW_NV_FA_WPR_RB 2"""
assert s.count(a) == 1
s = s.replace(a, b)

a = """#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = Ssm[g * FA_DEC_TILE + tid];
            }
          } else
#endif
          {
            /* SCORES: each thread streams one whole K row"""
b = """#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = Ssm[g * FA_DEC_TILE + tid];
            }
          } else if constexpr (PLOW_NV_FA_WPR128 && !SZKV && !FP8KV && D == 128) {
            const unsigned rmax = rmax_t;
            for (unsigned i = tid + rmax; i < (unsigned)FA_DEC_TILE; i += PLOW_NV_THREADS) {
#pragma unroll
                for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + i] = FA_NEG_INF;
            }
            const unsigned half = lane >> 4, hl = lane & 15u;
            constexpr unsigned WRB = PLOW_NV_FA_WPR_RB;
            constexpr unsigned STRIDE = PLOW_NV_WARPS * 2u;
            /* Warp-uniform trip count: both halves run every iteration (the shuffles below are
             * full-warp); each half gates only its own dot and store. */
            for (unsigned base = warp * 2u; base < rmax; base += STRIDE * WRB) {
                bf16v8 k8[WRB];
                bool live[WRB];
#pragma unroll
                for (unsigned t = 0; t < WRB; t++) {
                    const unsigned r = base + half + t * STRIDE;
                    const unsigned kvr = kv0 + r;
                    live[t] = (r < rmax) && (kvr <= qpos) && (!window || (qpos - kvr) < window);
                    /* `kvr & kv_mask` is always a real ring row: load unconditionally. */
                    k8[t] = ld_glob8_cs(kbase + (size_t)(kvr & kv_mask) * D + hl * 8u);
                }
#pragma unroll
                for (unsigned t = 0; t < WRB; t++) {
                    const unsigned r = base + half + t * STRIDE;
                    float dt[GF];
#pragma unroll
                    for (int g = 0; g < GF; g++) {
#if PLOW_NV_FA_QGLOB
                        const bf16v8 q8 = ld_glob8(Q + ((size_t)b * n_head + h0 + (unsigned)g) * D + hl * 8u);
#else
                        const bf16v8 q8 = ld_smem8(qsm + g * D + hl * 8u);
#endif
                        dt[g] = live[t] ? dot8(k8[t], q8, 0.0f) : 0.0f;
                    }
#pragma unroll
                    for (int g = 0; g < GF; g++)
#pragma unroll
                        for (int off = 8; off > 0; off >>= 1)
                            dt[g] += __shfl_xor_sync(0xffffffffu, dt[g], off, 32);
                    if (hl == 0 && r < rmax) {
#pragma unroll
                        for (int g = 0; g < GF; g++)
                            Ssm[g * FA_DEC_TILE + r] = live[t] ? dt[g] * FA_SCALE(scale) : FA_NEG_INF;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = Ssm[g * FA_DEC_TILE + tid];
          } else
#endif
          {
            /* SCORES: each thread streams one whole K row"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
open(p, 'w').write(s)
print("ok")
