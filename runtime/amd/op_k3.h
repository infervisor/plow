/* op_k3.h — Kimi-K3 BLOCK-STRUCTURE ops. Not the mixer (that is op_kda.h), the block around it.
 *
 * A K3 layer is NOT `residual + attn` then `residual + mlp`. `attn_res_block_size: 12` routes
 * every layer through `_forward_attn_residual` (modeling_kimi_linear.py:973), and the plain
 * residual ADD is replaced by a softmax MIX over the running prefix sum and up to 8 snapshots of
 * it. AMD's day-0 post names the structure "AttnRes" and confirms the period: "stores one block
 * residual every 12 layers".
 *
 * Getting this wrong is the silent kind. `residual + attn` and AttnRes have the same shapes, the
 * same dtypes and similar magnitudes; a block wired the plain way produces fluent, wrong output.
 *
 * Three arms live here:
 *   op 104  PLOW_DOP_ATTN_RES      — the softmax mix (this file's reason to exist)
 *   op 105  PLOW_DOP_SITU_GLU      — K3's `situ` GLU, which transforms the UP branch as well as
 *                                    the gate, so it does not fit the `act(g) * u` shape every
 *                                    existing GLU site assumes.
 *   op 106  PLOW_DOP_MLA_OUT_GATE  — `mla_use_output_gate`, the sigmoid gate on the MLA attention
 *                                    output before o_proj. 24 of 93 layers.
 */
#ifndef PLOW_OP_K3_H
#define PLOW_OP_K3_H

#include "amd_common.h"

/* -------------------------------------------------------------------------------------------
 * op 104 — AttnRes, the residual-attention block.
 *
 * Reference, verbatim (modeling_kimi_linear.py:1075-1088):
 *
 *     v            = cat(block_residual, prefix_sum.unsqueeze(1))      # [T, nb+1, H]
 *     v_float      = v.float()
 *     variance     = v_float.pow(2).mean(-1, keepdim=True)
 *     k            = v_float * rsqrt(variance + eps)
 *     score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
 *     scores       = (k * score_weight).sum(-1)
 *     probs        = scores.softmax(-1).unsqueeze(1)
 *     hidden       = matmul(probs, v_float).squeeze(1)
 *
 * FOUR THINGS THAT LOOK LIKE DETAILS AND ARE NOT:
 *
 *  1. `score_weight` is the PRODUCT of the RMSNorm gain and the [1, H] projection row, and it is
 *     CONSTANT. It folds at prep time into one [H] f32 and arrives as `t3`. Neither factor is
 *     needed separately by this op, and computing them separately would cost a second pass.
 *  2. The MIX is over the RAW rows `v_float`, NOT the normalized rows `k`. The normalization
 *     exists only to make the scores scale-free. Mixing `k` instead is a plausible misreading
 *     that produces output with the right shape and the wrong magnitude per row.
 *  3. `variance` is `mean(x^2)`, so the divisor is H — this is RMSNorm's variance, not a
 *     mean-centred one, and `eps` is INSIDE the rsqrt.
 *  4. The prefix sum is the LAST row, so with `nb = 0` the softmax is over a single element,
 *     probs = 1, and the op is an exact copy. The reference skips the call entirely in that case
 *     (`if block_residual.shape[1] > 0`); this arm handles it so a caller cannot get a
 *     zero-initialised output by emitting it anyway.
 *
 * SLICE MAP, stated honestly. Work items = T tokens, one WORKGROUP each, because both reductions
 * (the per-row variance and the per-row score) span the whole 7168-wide row and a softmax couples
 * the rows. At T = 1 that is `blocks = 1` of 256 — the `Mamba2Scan` occupancy shape, and it is
 * recorded here rather than hidden. `perf-data/archive/k3/kimi-k3-kernel-gap.md` §10 item 7 requires it to be
 * ONE packet ("at ~5.9 us of gate per narrow packet, three packets x 186 is 3.3 ms/token of pure
 * protocol"), which rules out splitting the reduction across blocks with a second packet to finish
 * it. So the ONLY lever at T = 1 is the body of that one workgroup.
 *
 * WHY THAT BODY WAS WORTH 3.3x, and why §10 item 7's "bandwidth-trivial (~48 MB/token)" was the
 * wrong model of it. MEASURED on one gfx950, T = 1, HID = 7168, ONE workgroup
 * (`runtime/tests/k3_attn_res_bench_gfx950.hip`, which keeps the old body VERBATIM so the
 * comparison stays runnable rather than remembered). Microseconds per invocation, in the residency
 * K3 decode actually has — the rows were written by an earlier packet of the same token so they
 * are L2-hot, `score_w` is one of 186 distinct per-site weights (5.3 MB/token) so it is not:
 *
 *     nb       0      1      2      3      4      5      6      7      8
 *     before   7.11  10.52  14.37  18.12  20.89  24.32  27.23  30.50  35.81
 *     after    3.44   4.21   4.96   5.63   6.39   7.04   7.88   8.59   9.25
 *
 * Over K3's real schedule (`nb = ceil(layer/12)`, 93 layers x 2 sites, mean nb 4.1):
 * 4.09 ms/token -> 1.23 ms/token, 3.33x. Note the SHAPE as well as the size — the old body grew
 * ~3.5 us per extra row, the new one ~0.7 us, because what it was paying per row was not bytes.
 *
 * It was never near a bandwidth limit: at nb = 8 it touches ~300 KiB of DISTINCT data, which even
 * at a pessimistic single-CU 50 GB/s is 6 us against the 33 us it took. It was bound by MEMORY
 * INSTRUCTION COUNT, by SERIALIZED PER-ROW LATENCY, and by BARRIERS. Four things, all measured:
 *
 *  1. SCALAR bf16 LOADS. One `global_load_ushort` per element per lane — 2 B per memory
 *     instruction. On a single CU the ceiling is outstanding memory instructions, not bytes, so
 *     this was ~8x off what `ld_glob8`'s 16 B `global_load_dwordx4` sustains. Every other
 *     streaming op in this tree (op_elementwise.h, `d_situ_glu` below) already used the 8-wide
 *     form; this one did not. `score_w` is now two `f32x4` instead of eight `dword`s.
 *  2. ONE FULL MEMORY LATENCY PER ROW. Row-outer, the `wave_sum` that closes row r forces an
 *     `s_waitcnt vmcnt(0)` before row r+1's loads are issued, so the (nb+1) rows serialize their
 *     ~700-cycle L2 latencies. That is why the op scaled so badly in nb and so weakly in HID
 *     (16x the HID was only 4x the time). Rows are now the INNER loop over a sweep of PLOW_ATTNRES_RG
 *     of them, so all nine loads are in flight together — worth 12% at nb = 8 on top of everything
 *     else, and it is what makes the per-row slope 0.7 us instead of 3.5.
 *  3. 2 x (nb+1) SEPARATE `block_sum` CALLS — 36 `__syncthreads` at nb = 8, one PAIR per statistic
 *     per row, each draining all 8 waves. Every wave's (sum x^2, sum x*w) for every row now goes
 *     to LDS as it is produced and the cross-wave fold happens ONCE: 2 barriers, not 36.
 *  4. THE SOFTMAX RAN ON THREAD 0. It is nb+1 <= 17 values, which is one row per LANE of a
 *     half-wave, so the fold, the rsqrt, the max-subtract and the normalise are now four
 *     cross-lane reductions in wave 0 and cost no barrier at all.
 *
 * ONE THING THAT LOOKS LIKE A WIN AND MEASURED AS A LOSS, recorded because the arithmetic argues
 * for it and the hardware does not. CACHING THE (nb+1) ROWS IN LDS on the statistics pass so the
 * mix never re-reads them: at nb = 8 that is 129 KiB, which fits the interpreter's 144 KiB arena,
 * and it cuts global traffic from ~300 KiB to ~172 KiB. It is 4-9% SLOWER (nb=8: 10.23 -> 10.70 us;
 * nb=0: 2.46 -> 2.67), because the mix's second read is an L2 HIT — 129 KiB against a 4 MiB XCD L2
 * — so the copy trades cheap L2 hits for `ds_write_b128` + `ds_read_b128` pairs in the hot loop and
 * buys no HBM traffic at all. The op re-reads the rows on purpose.
 *
 * WHAT IS LEFT, AND WHY IT IS NOT TAKEN HERE. Splitting the 7168 axis across N workgroups is still
 * legal in ONE packet — the interpreter is persistent at one workgroup per CU, so all N are
 * co-resident and can barrier on a device-scope counter without a second dispatch. The bench
 * measures that barrier: 0.80-0.91 us at 2-16 blocks, 1.7 us at 32, 10.5 us at 256. Two are needed
 * per invocation, so at 8-16 blocks the arithmetic says ~1.7 us of barrier against ~8 us saved at
 * nb = 8. It is NOT done because of a hazard that lives outside this file: a workgroup spinning in
 * an intra-packet barrier has stopped draining its own stream, so it can no longer signal a
 * counter that a PEER workgroup is gated on further up — and plowc is free to order the per-CU
 * streams in a way that makes that a cycle. Nothing in this arm can establish that it cannot.
 * Doing it safely needs a scheduling invariant in the emitter, a global scratch tensor for the
 * partials, and a deadlock argument; all three are outside op_k3.h.
 */
enum { PLOW_ATTNRES_MAXB = 16 };

/* ROWS PER SWEEP of the hidden axis, and the one number in here that was tuned rather than
 * derived. 9 = K3's nb_max + 1 (`nb = ceil(93/12) = 8`), so every row of every real invocation is
 * covered by ONE sweep. Measured (rows L2-hot, score_w cold; ms/token over the 186 sites):
 *   RG   1      8      9      10     17
 *        1.310  1.411  1.227  1.240  1.290
 * 8 is the worst of the set precisely because it just misses: nb = 8 needs two sweeps. Above 9 the
 * extra live accumulators cost more than the sweep they save. Register cost is 2*RG live f32; at
 * 9 the interpreter's decode object is unchanged at 248 VGPRs / 288 B scratch. */
#ifndef PLOW_ATTNRES_RG
#define PLOW_ATTNRES_RG 9
#endif

/* Every wave's (sum x^2, sum x*w) for every row, then the nb+1 probs after it: 272 + 17 floats,
 * 1156 B, against the interpreter's 144 KiB arena. The whole op fits in `part[]`'s old footprint
 * plus a page — nothing here competes with the GEMM tile that sizes the union. */
enum { PLOW_ATTNRES_PART = PLOW_WAVES * 2 * (PLOW_ATTNRES_MAXB + 1) };

/* THE FUSED POST-NORM (`gamma != nullptr`), and why it is an operand rather than an opcode.
 *
 * EVERY AttnRes in a K3 program is followed IMMEDIATELY by an RMSNorm over its own output, and by
 * nothing else: the attention-side mix feeds the mixer's pre-norm, the MLP-side mix feeds the
 * post-attention norm, and a program dump shows all 186 of them read exactly once (`plowrt disasm
 * --program 1`). Both packets are ONE workgroup at T = 1 — the mix because its reductions span the
 * row, the norm because `rows = 1` — so they are two SERIAL narrow packets on a decode chain that
 * has nothing else ready behind them. That is the shape `d_norm_residual_norm`'s note priced from
 * the other direction: splitting one narrow op into three serial ones cost +1.28 ms/token over 120
 * sites, ~5.3 us per added packet, with the arithmetic proven bit-identical. K3 pays it 186 times.
 *
 * BIT-EXACT BY CONSTRUCTION, and that is the reason for the shape of the code below. The mix is
 * stored to `out` exactly as the unfused arm stores it — a bf16 ROUNDED value — and the norm's
 * reduction then runs over the value re-READ from `out`, so what the second pass reduces is
 * precisely what the separate RMSNORM packet would have re-read from HBM. The round trip is
 * reproduced, not skipped; only the packet gate between the two is gone.
 *
 * IN PLACE, so this needs no second output tensor and no second `t` slot: the normed row overwrites
 * the raw mix. Legal because the raw mix has exactly one consumer and it is this norm.
 *
 * THE RE-READ IS NOT A THIRD HBM TRIP. `out` was written by THIS workgroup microseconds earlier and
 * every thread reads back exactly the elements it wrote (same `i = threadIdx.x; i += PLOW_THREADS`
 * map), so it is an L1/L2 hit — the same argument the LDS-caching autopsy above makes about the
 * mix's second read of the rows. `gamma` is the one cold stream, and it is the one the separate
 * RMSNORM packet was paying for too.
 *
 * `gamma == nullptr` keeps the raw-mix arm BYTE-IDENTICAL, which is what makes the emitter knob an
 * A/B out of one binary. */
/* Materialize a graph-selected residual seam before its consumer runs. Keeping this helper
 * separate leaves the consumer ABI and live ranges unchanged. The operation is
 * consumer-independent; the packet graph decides where it is legal and dispatch invokes it before
 * the selected consumer. */
#if PLOW_MATERIALIZED_RESIDUAL_INPUT
__device__ __forceinline__ void d_materialize_residual(
    bf16* __restrict__ out, const bf16* __restrict__ a, const bf16* __restrict__ b,
    const bf16* __restrict__ pre, unsigned T, unsigned HID, unsigned slice, unsigned nblk) {
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    const auto* pg = as_glob(pre);
    auto* og = as_glob(out);
    for (unsigned t = slice; t < T; t += nblk) {
        const size_t base = (size_t)t * HID;
        for (unsigned d = threadIdx.x * 8u; d < HID; d += PLOW_THREADS * 8u) {
            if (d + 8u <= HID) {
                const bf16v8 va = ld_glob8(ag + base + d), vb = ld_glob8(bg + base + d);
                const bf16v8 vp = pre ? ld_glob8(pg + base + d) : bf16v8_zero();
                bf16v8 vo;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const bf16 inner = f2bf(bf2f(va[j]) + bf2f(vb[j]));
                    vo[j] = pre ? f2bf(bf2f(vp[j]) + bf2f(inner)) : inner;
                }
                st_glob8(og + base + d, vo);
            } else {
                for (unsigned j = d; j < HID; ++j) {
                    const bf16 inner = f2bf(bf2f(a[base + j]) + bf2f(b[base + j]));
                    st_act1(out + base + j,
                            pre ? f2bf(bf2f(pre[base + j]) + bf2f(inner)) : inner);
                }
            }
        }
    }
    __syncthreads();
}
#endif

__device__ void d_attn_res(bf16* __restrict__ out, const bf16* __restrict__ prefix,
                           const bf16* __restrict__ blkres, const float* __restrict__ score_w,
                           unsigned T, unsigned HID, unsigned NB, unsigned NBCAP, float eps,
                           unsigned slice, unsigned nblk, float* __restrict__ lds,
                           const bf16* __restrict__ push_src, unsigned push_row,
                           const bf16* __restrict__ gamma = nullptr) {
    /* THE RING IS `[T][NBCAP][HID]`, AND `NBCAP` IS NOT `NB`. `NB` is the number of rows LIVE at
     * this layer, which grows 0 -> 8 with depth; `NBCAP` is the allocated row count, constant for
     * the whole program. At T = 1 `t` is 0 and the stride never multiplies, so the two coincide
     * and NOTHING in a decode program can tell them apart — which is exactly why the operand had
     * to arrive before prefill did. Striding a T > 1 ring by the live count would give every layer
     * a differently-strided view of the same buffer and shift the rows under it: no fault, no NaN,
     * a fluent wrong model. `NBCAP < NB` poisons below.
     *
     * THE SNAPSHOT PUSH. At a snapshot layer (`l % attn_res_block_size == 0`) the reference pushes
     * the layer input onto the ring BETWEEN the two mixes, so this layer's MLP-side mix sees one
     * more row than its attention-side mix did. Carried here rather than as its own opcode because
     * the ring is addressed by ROW and no other op takes a destination row over a compiler-owned
     * buffer — and because §10 item 7's packet budget rules out a third packet.
     *
     * IT IS SOUND AT EVERY T, and the earlier `T != 1` refusal was over-conservative. The argument
     * it rested on — "at T > 1 there are several workgroups and no barrier between them" — is true
     * and does not matter, because no workgroup reads another's pushed row inside this packet:
     *
     *   * `blocks = min(T, n_cu)` and the loop is `for (t = slice; t < T; t += nblk)`, so the
     *     workgroups partition the TOKENS. Every one of them is disjoint.
     *   * the ring is PER TOKEN (`bofs` strides by `t`), the snapshot source is `push_src[t]`, and
     *     the push for token `t` lands in token `t`'s own slice. So a workgroup only ever writes
     *     rows of tokens it owns.
     *   * within a token, the mix reads rows `[0, NB)` and the push writes row `push_row == NB`,
     *     one PAST them. So even the owning workgroup's own mix does not read what it pushed.
     *
     * Both readers are therefore outside this packet: the MLP-side mix of the same layer (a
     * separate packet, at the post-push count) and every later layer. `Builder::emit` joins them
     * with `Dep::Coarse`, which waits on ALL of the producer's slices, so the interpreter's counter
     * gate already orders them. The `__syncthreads()` below is kept for the same reason it was
     * needed at T = 1 — it costs nothing and it makes the push a completed store before the mix's
     * loads issue, rather than relying on the disjointness argument alone. */
    /* An `nb` past the bound would index off the end of `part`/`sco`; a capacity below the live
     * count, or a push row outside the ring, would index off the end of `blkres`. POISON, do not
     * return: leaving `out` untouched is the silent-NOP failure this tree keeps finding, and the
     * dispatch `default:` is ALREADY a silent NOP, so a second one here would be indistinguishable
     * from a missing opcode. A qNaN row is loud at the first norm downstream. devgen asserts the
     * same three (`emit_attn_res`), so this arm is the backstop, not the check. */
    if (NB > PLOW_ATTNRES_MAXB || NBCAP < NB || (push_src != nullptr && push_row >= NBCAP)) {
        for (unsigned t = slice; t < T; t += nblk)
            for (unsigned d = threadIdx.x; d < HID; d += PLOW_THREADS)
                st_act1(&out[(size_t)t * HID + d], (bf16)0x7fc1u); /* qNaN */
        return;
    }
    if (push_src != nullptr) {
        for (unsigned t = slice; t < T; t += nblk) {
            bf16* ring = (bf16*)blkres + (size_t)t * NBCAP * HID + (size_t)push_row * HID;
            const bf16* src = push_src + (size_t)t * HID;
            for (unsigned d = threadIdx.x; d < HID; d += PLOW_THREADS) ring[d] = src[d];
        }
        __syncthreads();
    }

    float* part = lds;                    /* [PLOW_WAVES][2 * (NB+1)] */
    float* sco = lds + PLOW_ATTNRES_PART; /* [NB+1] — probs */

    /* The 8-wide path needs every ROW BASE 16-byte aligned, which holds iff HID is a multiple of 8
     * (tensor bases are 4 KiB from hsa_memory_allocate; K3's HID is 7168). A HID that is not sends
     * the WHOLE row down the scalar tail rather than issuing a misaligned `dwordx4`. */
    const unsigned NV = (HID & 7u) ? 0u : (HID >> 3);
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const unsigned pstr = 2u * (NB + 1u); /* per-wave stride into `part` */

    for (unsigned t = slice; t < T; t += nblk) {
        const size_t pofs = (size_t)t * HID;
        const size_t bofs = (size_t)t * NBCAP * HID; /* CAPACITY, not the live count — see above */

        /* One pass per row accumulating BOTH statistics: sum(x^2) for the variance and
         * sum(x * score_weight) for the un-normalized score. The score is then
         * `sum(k * w) = rsqrt(var + eps) * sum(x * w)` because the scale factor is constant
         * across the row — so `k` never has to be materialized. */
        for (unsigned g = 0; g <= NB; g += PLOW_ATTNRES_RG) {
            float ss[PLOW_ATTNRES_RG], sw[PLOW_ATTNRES_RG];
#pragma unroll
            for (int k = 0; k < PLOW_ATTNRES_RG; k++) ss[k] = sw[k] = 0.0f;
            for (unsigned i = threadIdx.x; i < NV; i += PLOW_THREADS) {
                const f32x4 w0 = *(const PLOW_GLOB f32x4*)(score_w + i * 8);
                const f32x4 w1 = *(const PLOW_GLOB f32x4*)(score_w + i * 8 + 4);
#pragma unroll
                for (int k = 0; k < PLOW_ATTNRES_RG; k++) {
                    const unsigned r = g + (unsigned)k;
                    if (r > NB) break;
                    const bf16* __restrict__ vr =
                        (r < NB) ? blkres + bofs + (size_t)r * HID : prefix + pofs;
                    const bf16v8 v = ld_glob8(vr + i * 8);
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float x = bf2f(v[j]);
                        ss[k] += x * x;
                        sw[k] += x * (j < 4 ? w0[j] : w1[j & 3]);
                    }
                }
            }
            for (unsigned d = NV * 8 + threadIdx.x; d < HID; d += PLOW_THREADS) {
                const float wd = score_w[d];
#pragma unroll
                for (int k = 0; k < PLOW_ATTNRES_RG; k++) {
                    const unsigned r = g + (unsigned)k;
                    if (r > NB) break;
                    const float x =
                        bf2f((r < NB) ? blkres[bofs + (size_t)r * HID + d] : prefix[pofs + d]);
                    ss[k] += x * x;
                    sw[k] += x * wd;
                }
            }
            /* Cross-LANE only. The cross-WAVE fold is deferred to one barrier below, which is the
             * whole point: `block_sum` here would be 2 barriers per row. */
#pragma unroll
            for (int k = 0; k < PLOW_ATTNRES_RG; k++) {
                const unsigned r = g + (unsigned)k;
                if (r > NB) break;
                const float a = wave_sum(ss[k]), b = wave_sum(sw[k]);
                if (lane == 0) {
                    part[wave * pstr + 2 * r] = a;
                    part[wave * pstr + 2 * r + 1] = b;
                }
            }
        }
        __syncthreads();

        /* FOLD, SCORE AND SOFTMAX, all in wave 0 and all in registers — ONE ROW PER LANE, so
         * nb+1 <= 17 fits inside the 32-lane half-wave that `half_wave_max`/`half_wave_sum`
         * reduce over.
         *
         * Max-subtracted, because the scores are a dot product of a unit-RMS row with an unbounded
         * learned vector and nothing bounds them a priori. Lanes past NB hold -INFINITY, which is
         * the identity of the max, and 0, which is the identity of the sum. */
        if (wave == 0) {
            float ss = 0.0f, sw = 0.0f;
            if (lane <= NB) {
#pragma unroll
                for (int w = 0; w < PLOW_WAVES; w++) {
                    ss += part[w * pstr + 2 * lane];
                    sw += part[w * pstr + 2 * lane + 1];
                }
            }
            const float s = (lane <= NB) ? sw * rsqrtf(ss / (float)HID + eps) : -INFINITY;
            const float m = half_wave_max(s);
            const float e = (lane <= NB) ? __expf(s - m) : 0.0f;
            const float z = half_wave_sum(e);
            if (lane <= NB) sco[lane] = e / z;
        }
        __syncthreads();

        /* The mix, over the RAW rows. Eight lanes' worth of f32 accumulator per thread so the
         * (nb+1) rows are walked once per 16-byte chunk instead of once per element.
         *
         * `ss` is the fused norm's sum-of-squares and it is accumulated over the bf16-ROUNDED
         * output, not over `a[j]` — that is what makes the fusion bit-exact against the separate
         * RMSNORM packet, which reduces the value it re-reads from HBM. */
        float ss = 0.0f;
        for (unsigned i = threadIdx.x; i < NV; i += PLOW_THREADS) {
            float a[8];
#pragma unroll
            for (int j = 0; j < 8; j++) a[j] = 0.0f;
            for (unsigned r = 0; r <= NB; r++) {
                const bf16* __restrict__ vr =
                    (r < NB) ? blkres + bofs + (size_t)r * HID : prefix + pofs;
                const bf16v8 v = ld_glob8(vr + i * 8);
                const float p = sco[r];
#pragma unroll
                for (int j = 0; j < 8; j++) a[j] += p * bf2f(v[j]);
            }
            bf16v8 o;
#pragma unroll
            for (int j = 0; j < 8; j++) o[j] = f2bf(a[j]);
            if (gamma) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = bf2f(o[j]);
                    ss += f * f;
                }
            }
            st_glob8(out + pofs + i * 8, o);
        }
        for (unsigned d = NV * 8 + threadIdx.x; d < HID; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned r = 0; r < NB; r++)
                acc += sco[r] * bf2f(blkres[bofs + (size_t)r * HID + d]);
            acc += sco[NB] * bf2f(prefix[pofs + d]);
            const bf16 ob = f2bf(acc);
            if (gamma) {
                const float f = bf2f(ob);
                ss += f * f;
            }
            st_act1(&out[pofs + d], ob);
        }
        /* THE FUSED NORM. `block_sum` reuses `part[0..PLOW_WAVES)`, which the statistics fold above
         * has already consumed, and leaves `sco` (which lives past PLOW_ATTNRES_PART) untouched —
         * so it costs no extra LDS and no extra barrier beyond the pair block_sum itself carries. */
        if (gamma) {
            const float inv = rsqrtf(block_sum(ss, part) / (float)HID + eps);
            const auto* gg = as_glob(gamma);
            auto* og = as_glob(out);
            for (unsigned i = threadIdx.x; i < NV; i += PLOW_THREADS) {
                const bf16v8 o = ld_glob8(og + pofs + i * 8);
                const bf16v8 g = ld_glob8(gg + i * 8);
                bf16v8 n;
#pragma unroll
                for (int j = 0; j < 8; j++) n[j] = f2bf(bf2f(o[j]) * inv * bf2f(g[j]));
                st_glob8(og + pofs + i * 8, n);
            }
            for (unsigned d = NV * 8 + threadIdx.x; d < HID; d += PLOW_THREADS)
                st_act1(&out[pofs + d], f2bf(bf2f(out[pofs + d]) * inv * bf2f(gamma[d])));
        }
        __syncthreads(); /* `sco`/`part` are rewritten by the next token and by the next op */
    }
}

/* -------------------------------------------------------------------------------------------
 * op 105 — `situ` GLU. K3's activation, on EVERY GLU in the model.
 *
 * Reference, verbatim (modeling_kimi_linear.py:64-85, registered as ACT2FN["situ"]):
 *
 *     situ_a = beta * tanh(gate / beta) * sigmoid(gate)
 *     if linear_beta is not None: up = linear_beta * tanh(up / linear_beta)
 *     return situ_a * up
 *
 * with `activation_situ_beta = 4.0` and `activation_situ_linear_beta = 25.0`.
 *
 * WHY THIS IS NOT A THIRD `act` CODE. Every GLU site in this tree computes `act(g) * u` and
 * selects the activation with a two-value ternary
 * (`(act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g)`, op_elementwise.h:69). situ
 * transforms the UP branch too, so the EXPRESSION SHAPE changes, not just the function: it is
 * `A(g) * B(u)`. A new `act` code alone would silently select the gate transform and leave `up`
 * un-clipped, which at |u| < 25 is a small error that grows with the tail — plausible output,
 * wrong model. A distinct opcode makes the omission impossible.
 *
 * It IS a soft-clipped SiLU: as beta -> inf, `beta*tanh(g/beta) -> g` and `situ_a -> g*sigmoid(g)`.
 * beta = 4 clamps the gate branch to +-4 and beta_l = 25 clamps the up branch to +-25.
 *
 * COST — AND THIS PARAGRAPH USED TO BE WRONG, WHICH IS WHY IT NOW CARRIES A MEASUREMENT.
 * It said the extra exponentials were "free here — this op is a pure streaming pass at ~1.5
 * B/FLOP and is memory-bound by an enormous margin". Three bf16 accesses per element is 6 B, so
 * the 8 TB/s roofline is 1.3e12 elem/s; the arithmetic as it stood was 109 VALU per element,
 * which this part issues at ~0.36e12 elem/s. It was VALU-bound by ~4x, not memory-bound, and the
 * word "free" was doing all the work. Measured with runtime/tests/act_bench_gfx950.hip at K3's
 * dense width (T=1024, 18432): 0.0404 ms and 2803 GB/s before the expensive-instruction sweep,
 * 0.0363 ms and 3122 GB/s after — a 10% gain from arithmetic alone, on an op that was supposed
 * to have no arithmetic problem. It is still not at the roofline. If this op ever matters more
 * than it does today (it is ~0.2% of a MoE layer), the remaining lever is the per-element
 * `g/beta` division that `k3_situ_gate` keeps ON PURPOSE for op85's sake — read its note in
 * amd_common.h before touching it, because the obvious fix costs more elsewhere than it gains
 * here.
 *
 * `linear_beta <= 0` disables the up transform, which is what `linear_beta is None` means. It is
 * a comparison rather than a flag bit so a zeroed immediate degrades to "no transform" rather
 * than to "clip everything to zero".
 */
/* `k3_situ_gate` / `k3_situ_up` live in amd_common.h, because the ROUTED EXPERT GLU in op_moe.h
 * needs the identical pair and op_moe.h must not depend on this header. Two copies of a
 * transcendental expression is exactly how a model ends up computing two different activations
 * in its dense and expert paths. */

__device__ void d_situ_glu(bf16* __restrict__ out, const bf16* __restrict__ gate,
                           const bf16* __restrict__ up, unsigned n, float beta, float linear_beta,
                           unsigned slice, unsigned nblk) {
    const unsigned stride = nblk * PLOW_THREADS * 8;
    const auto* gg = as_glob(gate);
    const auto* ug = as_glob(up);
    auto* og = as_glob(out);
    for (unsigned i = (slice * PLOW_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 vg = ld_glob8(gg + i), vu = ld_glob8(ug + i);
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++)
                vo[j] = f2bf(k3_situ_gate(bf2f(vg[j]), beta) * k3_situ_up(bf2f(vu[j]), linear_beta));
            st_glob8(og + i, vo);
        } else {
            for (unsigned j = i; j < n; j++)
                st_act1(&out[j], f2bf(k3_situ_gate(bf2f(gate[j]), beta) * k3_situ_up(bf2f(up[j]), linear_beta)));
        }
    }
}

/* -------------------------------------------------------------------------------------------
 * op 106 — the MLA OUTPUT GATE (`mla_use_output_gate: true`, 24 of Kimi-K3's 93 layers).
 *
 * Reference, verbatim (modeling_kimi_linear.py:470-473):
 *
 *     if self.use_output_gate:
 *         g = self.g_proj(hidden_states).sigmoid()
 *         attn_output = attn_output * g
 *     attn_output = self.o_proj(attn_output)
 *
 * `g_proj` is [n_head * v_head_dim, hidden] = [12288, 7168], so the gate is elementwise over the
 * flattened attention output — which in plow's ABSORBED MLA path is exactly what
 * PLOW_DOP_MLA_MERGE_FOLD writes: `O + (b*n_head + h)*V`, i.e. head-major [nh][v_head]. The
 * reference's `attn_output.reshape(batch_size, seq_length, -1)` is the same order, so no permute
 * is implied. If the two ever disagree this op is where it shows, which is why it is one op with
 * one layout note rather than a fold into the merge epilogue.
 *
 * WHY THIS IS AN OPCODE AND NOT A THIRD `act` CODE ON PLOW_DOP_GLU — the same argument op 105
 * had to make, with a different wrong answer at the end of it. `act=1` is SiLU, `x*sigmoid(x)`;
 * this is `sigmoid(x)`. They differ by a factor of the logit, which near 0 is a factor of ~0 and
 * in the tail is a factor of ~|x|. A `GLU(act=1)` here would produce finite, correctly-shaped,
 * wrong output on 24 layers of every token.
 *
 * WHY NOT FOLD IT INTO PLOW_DOP_MLA_MERGE_FOLD's EPILOGUE, which perf-data/archive/k3/kimi-k3-kernel-gap.md
 * §10 item 6 suggests and which is nearly free (that op runs at ~2.9% of the bandwidth ceiling).
 * Because MLA_MERGE_FOLD is GLM-5.2's op too and is on its critical path. A K3-only transform
 * inside it either costs GLM a branch and an operand slot, or costs a second template
 * instantiation in a decode object that has 8 VGPRs of headroom. A separate streaming pass over
 * 12288 bf16 elements is ~50 KB of traffic; it is not where a K3 token's time goes. Keeping it
 * separate ALSO keeps the fold and the gate independently diffable, which is the whole point of a
 * stage-by-stage gate — a merge bug and a gate bug cannot be confused.
 */
__device__ void d_mla_out_gate(bf16* __restrict__ out, const bf16* __restrict__ a,
                               const bf16* __restrict__ b, unsigned n, unsigned slice,
                               unsigned nblk) {
    const unsigned stride = nblk * PLOW_THREADS * 8;
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    auto* og = as_glob(out);
    for (unsigned i = (slice * PLOW_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 va = ld_glob8(ag + i), vb = ld_glob8(bg + i);
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++)
                vo[j] = f2bf(bf2f(va[j]) * k3_sigmoid(bf2f(vb[j])));
            st_glob8(og + i, vo);
        } else {
            for (unsigned j = i; j < n; j++)
                st_act1(&out[j], f2bf(bf2f(a[j]) * k3_sigmoid(bf2f(b[j]))));
        }
    }
}

#endif /* PLOW_OP_K3_H */
