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
 * recorded here rather than hidden. Two things make it tolerable and neither makes it good:
 * the op moves at most (nb+1) * 7168 * 2 B = 143 KiB per token, and
 * `perf-data/kimi-k3-kernel-gap.md` §10 item 7 requires it to be ONE packet ("at ~5.9 us of gate
 * per narrow packet, three packets x 186 is 3.3 ms/token of pure protocol"), which rules out the
 * obvious fix of splitting the reduction across blocks with a second packet to finish it.
 * The real fix is a batched form once T > 1 is the common case; at T = 1 it is 186 serialized
 * single-CU packets per token and that is a KNOWN, UNMEASURED cost.
 */
enum { PLOW_ATTNRES_MAXB = 16 };

__device__ void d_attn_res(bf16* __restrict__ out, const bf16* __restrict__ prefix,
                           const bf16* __restrict__ blkres, const float* __restrict__ score_w,
                           unsigned T, unsigned HID, unsigned NB, float eps, unsigned slice,
                           unsigned nblk, float* __restrict__ lds) {
    /* An `nb` past the bound would silently index off the end of `sco`. Refuse: leaving `out`
     * untouched is exactly the silent-NOP failure this tree keeps finding, so bound the emitter
     * instead — devgen asserts the same constant. */
    if (NB > PLOW_ATTNRES_MAXB) return;

    float* part = lds;                 /* [PLOW_WAVES] — block_sum scratch */
    float* sco = lds + PLOW_WAVES;     /* [PLOW_ATTNRES_MAXB + 1] — scores, then probs */

    for (unsigned t = slice; t < T; t += nblk) {
        const size_t pofs = (size_t)t * HID;
        const size_t bofs = (size_t)t * NB * HID;

        /* One pass per row accumulating BOTH statistics: sum(x^2) for the variance and
         * sum(x * score_weight) for the un-normalized score. The score is then
         * `sum(k * w) = rsqrt(var + eps) * sum(x * w)` because the scale factor is constant
         * across the row — so `k` never has to be materialized. */
        for (unsigned r = 0; r <= NB; r++) {
            const bf16* vr = (r < NB) ? blkres + bofs + (size_t)r * HID : prefix + pofs;
            float ss = 0.0f, sw = 0.0f;
            for (unsigned d = threadIdx.x; d < HID; d += PLOW_THREADS) {
                const float x = bf2f(vr[d]);
                ss += x * x;
                sw += x * score_w[d];
            }
            ss = block_sum(ss, part);
            sw = block_sum(sw, part);
            if (threadIdx.x == 0) sco[r] = sw * rsqrtf(ss / (float)HID + eps);
        }
        __syncthreads();

        /* Softmax over nb+1 <= 17 values, on one thread. Max-subtracted, because the scores are
         * a dot product of a unit-RMS row with an unbounded learned vector and nothing bounds
         * them a priori. */
        if (threadIdx.x == 0) {
            float m = -INFINITY;
            for (unsigned r = 0; r <= NB; r++) m = fmaxf(m, sco[r]);
            float s = 0.0f;
            for (unsigned r = 0; r <= NB; r++) {
                const float e = __expf(sco[r] - m);
                sco[r] = e;
                s += e;
            }
            const float inv = 1.0f / s;
            for (unsigned r = 0; r <= NB; r++) sco[r] *= inv;
        }
        __syncthreads();

        /* The mix, over the RAW rows. */
        for (unsigned d = threadIdx.x; d < HID; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned r = 0; r < NB; r++) acc += sco[r] * bf2f(blkres[bofs + (size_t)r * HID + d]);
            acc += sco[NB] * bf2f(prefix[pofs + d]);
            out[pofs + d] = f2bf(acc);
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
 * COST: tanh(x) = 2*sigmoid(2x) - 1, so the gate branch is 2 exponentials against SiLU's 1 and
 * the up branch adds a third. That is free here — this op is a pure streaming pass at ~1.5 B/FLOP
 * and is memory-bound by an enormous margin.
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
                out[j] = f2bf(k3_situ_gate(bf2f(gate[j]), beta) * k3_situ_up(bf2f(up[j]), linear_beta));
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
 * WHY NOT FOLD IT INTO PLOW_DOP_MLA_MERGE_FOLD's EPILOGUE, which perf-data/kimi-k3-kernel-gap.md
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
                out[j] = f2bf(bf2f(a[j]) * k3_sigmoid(bf2f(b[j])));
        }
    }
}

#endif /* PLOW_OP_K3_H */
