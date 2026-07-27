/* dev_isa.h — the instruction format the persistent on-device interpreter runs.
 *
 * SHARED, THREE WAYS. This header is the single definition of the device ISA:
 *   - the HIP interpreter (runtime/amd/interp.hip) executes it,
 *   - the C host runtime builds it,
 *   - crates/packet/src/dev.rs mirrors it for Rust, and
 *     crates/packet/tests/dev_abi.rs locks the layouts against the asserts below.
 * Change a field here and the Rust ABI test fails. That is the point.
 *
 * WHY IT IS FIXED-STRIDE. The wire packet stream (include/packet.h) is
 * variable-length and host-decoded. The GPU must not parse it: the host expands
 * the stream ONCE into a PlowDevInst[] in device memory, so the interpreter's
 * inner loop is an indexed load and a switch. Nothing is decoded on device.
 *
 * EXECUTION MODEL. One persistent kernel per device, grid = CU count, resident
 * for the life of the model. There is NO per-op launch. Each workgroup owns one
 * CU and walks its own stream of (inst, slice) entries:
 *
 *     for each entry in my stream:
 *         spin until every wait-counter reaches its threshold
 *         execute inst, computing only the `slice`-th share of its work
 *         __threadfence()                       (agent scope)
 *         atomically bump every successor counter
 *
 * An op spread over N workgroups appears once in `insts` and N times in the
 * streams, with slices 0..N-1; its consumers wait on a counter with threshold N.
 * "All producers done" therefore falls straight out of the counter protocol, and
 * no grid-wide barrier primitive is needed.
 *
 * CO-RESIDENCY IS THE SAFETY CONDITION. The spin is only sound because grid ==
 * CU count, so every workgroup is resident and a producer can never be starved
 * by a spinning consumer. Launching a larger grid DEADLOCKS. The fence and the
 * counter atomics must be AGENT-scope, not workgroup-scope: CDNA's L2 is
 * per-XCD, so a consumer on another XCD would otherwise read a stale line.
 */
#ifndef PLOW_DEV_ISA_H
#define PLOW_DEV_ISA_H

#include <stdint.h>

#ifdef __cplusplus
#define PLOW_SASSERT(c, m) static_assert(c, m)
#else
#define PLOW_SASSERT(c, m) _Static_assert(c, m)
#endif

/* Device opcodes. Deliberately a small closed set — the interpreter's switch is
 * the hot path, so this is an ISA, not an extension point.
 *
 * Operand slots: t[] = tensor handles, i[] = integers, f[] = floats.
 * A tensor handle of PLOW_TENSOR_NONE means "absent optional operand" and
 * reaches the op as a null pointer. */
enum {
    PLOW_DOP_NOP = 0,

    /* t0=out t1=x t2=gamma?            i0=rows i1=feat          f0=eps
     * gamma==NONE is the weightless RMSNorm (Gemma's v_norm). */
    PLOW_DOP_RMSNORM = 1,

    /* t0=rms(f32) t1=x                 i0=rows i1=feat          f0=eps
     * Row RMS scalars only, consumed by GEMM_NORM / GEMV so the normalized
     * activation never round-trips through HBM. */
    PLOW_DOP_ROWRMS = 2,

    /* t0=out t1=x t2=gamma? t3=cos? t4=sin? t5=pos(i32)
     * i0=ntok i1=nhead i2=hd i3=out_row0   f0=eps
     * cos==NONE skips RoPE (that is v_norm). out_row0 lets K/V land directly at
     * a row offset of the KV cache, so the cache write is not a separate copy. */
    PLOW_DOP_HEADNORM_ROPE = 3,

    /* t0=out t1=a t2=b                 i0=n                     f0=scale
     * out = (a + b) * scale. `scale` absorbs Gemma's per-layer layer_scalar on
     * the SECOND residual add; pass 1.0 for the first. */
    PLOW_DOP_RESIDUAL = 4,

    /* t0=out t1=gate t2=up             i0=n i1=act(0=gelu_tanh,1=silu)
     * Gemma is GeGLU (gelu_tanh), not SwiGLU. */
    PLOW_DOP_GLU = 5,

    /* t0=out t1=table t2=ids(i32)      i0=ntok i1=hidden        f0=scale
     * `scale` must be the BF16-ROUNDED sqrt(hidden): 73.5 (31B), 62.0 (12B). */
    PLOW_DOP_EMBED = 6,

    /* t0=out t1=x                      i0=n                     f0=cap */
    PLOW_DOP_SOFTCAP = 7,

    /* t0=C t1=A t2=B                   i0=M i1=N i2=K  i4=a_row0
     * C[M,N] = A[M,K] . B[N,K]^T. B is [out_features, in_features].
     *
     * i4 skips a_row0 rows of A. It exists so prefill's lm_head can be M=1 over the LAST
     * token instead of M=T over all of them: at T=4096 that is a 512 KB logit buffer and
     * ~0 ms instead of 2.1 GB and ~19 ms of GEMM we would immediately throw away. */
    PLOW_DOP_GEMM = 8,

    /* t0=C t1=A t2=B t3=rms(f32) t4=gamma   i0=M i1=N i2=K
     * As GEMM, with RMSNorm folded into the A-operand load. */
    PLOW_DOP_GEMM_NORM = 9,

    /* t0=C t1=A t2=B                   i0=M i1=N i2=K
     * Same math as GEMM, SMALL TILE (64x128 vs 256x256). A separate OPCODE, not a
     * runtime flag: the tile has to be a compile-time constant or the register
     * allocator budgets for the worst arm and every arm spills.
     *
     * This exists because prefill is not one big GEMM. At T=128 a 256x256 tile gives
     * q_proj 32 tiles and down_proj 21 -- on a 256-CU machine -- and pads M=128 up to
     * 256, wasting half the MFMA rows. The small tile gives 128 and 84 tiles with no
     * M-padding. plowc picks per shape: big tile when the shape fills the machine (it has
     * far better data reuse), small tile when it does not. */
    PLOW_DOP_GEMM_SMALL = 14, /* 64x128  */
    PLOW_DOP_GEMM_MED = 15,   /* 128x128 */

    /* t0=out t1=a t2=b t3=gamma?   i0=rows i1=feat   f0=eps f1=scale
     * out = (a + RMSNorm(b, gamma)) * scale — Gemma's sandwich tail in ONE packet.
     * Was an RMSNORM packet plus a RESIDUAL packet: two global gates, and the norm is a row
     * reduction, so in decode ONE workgroup did it while 255 waited on the counter. */
    PLOW_DOP_NORM_RESIDUAL = 16,

    /* t0=out t1=resid t2=a t3=b t4=gamma?   i0=rows i1=feat   f0=eps
     * Qwen/Llama PRE-NORM tail in ONE packet:
     *     resid = a + b ; out = RMSNorm(resid, gamma)
     * where a is the incoming residual and b the sublayer output (o_proj / down). Distinct from
     * NORM_RESIDUAL (Gemma's SANDWICH a + RMSNorm(b)): here the norm is on the SUM, and BOTH the
     * updated residual stream AND its norm are written. Was a RESIDUAL packet plus an RMSNORM
     * packet: two global gates, each a single-workgroup decode op with 255 CUs stalled behind it. */
    PLOW_DOP_ADD_NORM = 21,

    /* t0=out t1=resid t2=a t3=b t4=gamma_b? t5=gamma_n?   i0=rows i1=feat   f0=eps f1=scale
     * Gemma SANDWICH tail AND the norm that follows it, in ONE packet (Experiment N1):
     *     resid = (a + RMSNorm(b, gamma_b)) * scale ; out = RMSNorm(resid, gamma_n)
     * The narrow->narrow successor to ADD_NORM: fuses a NORM_RESIDUAL and the RMSNORM that
     * re-reads its output, deleting a global gate and a full HBM round trip (resid stays in
     * registers across the second reduction). Bit-exact to the pair: resid is rounded to bf16
     * before the second reduction, reproducing NORM_RESIDUAL's store + RMSNORM's reload. */
    PLOW_DOP_NORM_RESIDUAL_NORM = 23,

    /* t0=C t1=x t2=W t3=rms? t4=gamma?   i0=M i1=N i2=K i3=norm i4=a_row0   f0=eps
     * Decode path: M <= PLOW_GEMV_MAXM. Bandwidth-bound, no MFMA.
     *
     * i3 (norm): 0 = none; 1 = apply a PRECOMPUTED row RMS from t3; 2 = COMPUTE the row RMS
     * here, which deletes the whole RMSNORM packet and its gate. Mode 2 is nearly free: the
     * GEMV already stages x in LDS, so the RMS is one block reduction over data already
     * there. It is what removes decode's single-CU norm bottleneck. */
    PLOW_DOP_GEMV = 10,

    /* t0=Opart(f32) t1=mlpart(f32) t2=Q t3=K t4=V
     * i0=n_q i1=n_kv i2=n_head i3=n_kv_head i4=q_pos0 i5=window i6=hd i7=nsplit
     * f0=scale   (Gemma: 1.0 — there is NO 1/sqrt(d))
     * window==0 => full causal. hd must be 256 or 512.
     *
     * SPLIT-KV. Prefill emits UNNORMALIZED partials and must ALWAYS be followed by a
     * FLASH_MERGE, even at nsplit==1. Without the split there are only
     * ceil(n_q/(waves*BQ)) * n_head work units — 64 at Gemma-31B with n_q=512 — so most of
     * a 256-CU machine sits idle while the busy few walk the whole causal triangle. */
    PLOW_DOP_FLASH_PREFILL = 11,

    /* t0=Opart(f32) t1=mlpart(f32) t2=Q t3=K t4=V t5=kv_len(i32)
     * i0=n_batch i1=n_head i2=n_kv_head i3=kv_stride i4=window i5=nsplit i6=hd
     * f0=scale */
    PLOW_DOP_FLASH_DECODE = 12,

    /* t0=O t1=Opart t2=mlpart          i0=n_batch i1=n_head i2=nsplit i3=hd */
    PLOW_DOP_FLASH_MERGE = 13,

    /* t0=part(u64[blocks]) t1=x(bf16)  i0=n
     * t0=ids(i32)          t1=part     i0=blocks-of-the-producer
     *
     * Greedy sampling, ON DEVICE. The host used to pull the whole logit row back (512 KB per
     * token over PCIe), argmax 262144 entries on the CPU, and push the winning id back up --
     * ~0.2 ms of a 17 ms token, and the only part of a decode step still running on the host.
     *
     * ARGMAX_FIN writes straight into `in.ids`, which is exactly the tensor the NEXT step's
     * EMBED reads. So the token never leaves the GPU: the host reads 4 bytes to print it and
     * to check for EOS, and writes nothing.
     *
     * Two packets because a cross-block reduction needs a gate -- and gates are free here
     * (measured: deleting 120 of them changed nothing). Each block reduces its slice to one
     * packed u64; ARGMAX_FIN folds the blocks. The packing is what lets a plain unsigned max
     * do the work -- see amax_pack() in op_elementwise.h. */
    /* GEMV over gate|up in ONE pass, with act(gate)*up applied in the EPILOGUE -- the fusion
     * every BLAS ships. Output-stationary, so the GLU runs exactly once per element and nothing
     * is replicated. Deletes the GLU packet AND merges two GEMVs into one.
     * t[0]=fu t[1]=x t[2]=W_gate t[5]=W_up; i[0]=M i[1]=N i[2]=K i[5]=act. See op_gemm.h. */
    PLOW_DOP_GEMV_GLU = 19,

    /* GEMM over gate|up in ONE pass, act(gate)*up in the EPILOGUE -- the prefill twin of
     * PLOW_DOP_GEMV_GLU. Same tile, same registers, same MFMA count: the SN axis selects
     * gate vs up instead of a column block, so both land in the same lane.
     * t[0]=fu t[1]=x t[2]=W_gate t[5]=W_up; i[0]=M i[1]=N i[2]=K i[5]=act. */
    PLOW_DOP_GEMM_GLU = 20,

    PLOW_DOP_ARGMAX = 17,
    PLOW_DOP_ARGMAX_FIN = 18,

    /* t0=q_out t1=x t2=W_q t3=k_out t4=W_k t5=v_out t6=W_v   i0=M i1=Nq i2=K i3=Nk i4=Nv
     * FUSED Q|K|V GEMV: q/k/v share x and K, so their outputs concatenate into one N=Nq+Nk+Nv
     * sweep (col<Nq -> q_out, <Nq+Nk -> k_out, else v_out). Decode-only. Replaces three GEMVs on
     * disjoint CU sets with one uniform-fill op: two fewer gates/layer, no cross-op imbalance.
     * Single weight stream, so the plain GEMV register budget. See op_gemm.h. */
    PLOW_DOP_GEMV_QKV = 22,

    /* ===== CROSS-GPU (tensor-parallel) tile-graph ops. =========================
     * New opcodes assigned AFTER main's last (23), no collision. Names mirror the
     * generic infervisor RDMA-family variants (p2p/allreduce/allgather/reducescatter)
     * so the two ABIs converge (see plans/tp-design.md §1b, §8).
     *
     * These are the ONLY ops that touch peer VRAM. Their wait/succ counters live in
     * the SYSTEM-scope `xctr` region, not the agent-scope `counters` — the stream
     * entry carries PLOW_SE_XCTR to select it. Weights/KV/residual never cross the
     * fabric; only the H-wide reduction partials do (plans/tp-design.md §7). */

    /* ALL-REDUCE, the TP primitive (plans/tp-design.md §8a). One-shot: each rank's
     * producing GEMV (o_proj/down) has already published its partial H-vector into
     * its own peer_scratch slot and system-signalled every peer's xctr; this op is
     * the CONSUME half — it waits on N partials (SE_XCTR gate) then sums the N peer
     * slots into a local full H-vector, f32 accumulate rounded to bf16.
     *   t0=out (local, full H) i0=H i1=n_gpu i2=slot(byte offset into peer_scratch)
     *   wait = xctr: N-arrivals counter (coarse: 1 counter @ threshold n_gpu)
     *   succ = local counter feeding the following NormResidual */
    PLOW_DOP_XREDUCE = 24,

    /* REDUCE-SCATTER + ALL-GATHER — the symmetric decomposition of all-reduce, kept
     * defined for CP / larger worlds. For N<=8 one-shot XREDUCE is lower-latency, so
     * these are not emitted on the decode path yet (plans/tp-design.md §8a). */
    PLOW_DOP_XREDUCESCATTER = 25,
    PLOW_DOP_XALLGATHER = 26,

    /* CONTEXT-PARALLEL cross-GPU flash LSE-merge (plans/tp-design.md §8c, §9). Each
     * rank produced a local (O_partial,m,l) over its KV-position shard; this folds
     * the N peers' partials (numerically-stable log-sum-exp) into the replicated
     * attention output. ABI mirrors FLASH_MERGE with t1.. in peer_scratch + xctr
     * gates. STUB for now (dispatch present, body deferred to CP phase). */
    PLOW_DOP_XFLASHMERGE = 27,

    /* SHARDED lm_head argmax-merge (plans/tp-design.md §8d). lm_head is vocab-column
     * parallel: each rank argmaxes its V/N logits to a packed (key,idx) u64; this
     * reads the N peer packed maxima and folds them to the global token id, written
     * into every rank's in.ids so the next EMBED needs no broadcast.
     *   t0=ids(i32) t1=local_part(u64) i0=n_gpu i2=slot(byte offset)  wait=xctr */
    PLOW_DOP_XARGMAX_FIN = 28,

    /* TWO-SHOT all-reduce (reduce-scatter + all-gather) for the LARGE prefill
     * [T,hidden] message (plans/tp-prefill.md §4). Fused + self-contained like XREDUCE,
     * but bandwidth-optimal: each rank reduces only its OWNED 1/N slice from every peer's
     * partial (writing it in-place, peer-visible), then gathers every peer's reduced
     * slice into the local full vector. Fabric ~2(N-1)/N*msg/rank vs one-shot (N-1)*msg.
     * Two internal xctr rendezvous (gate_rs before the scatter, gate_ag between the two
     * phases). Bit-identical to one-shot (same f32-acc, r=0..N-1 order). DECODE keeps the
     * one-shot — its tiny [1,hidden] message is latency-bound, so 1 sync wins.
     *   t0=out i0=n(=t*hidden) i1=n_gpu i2=slot(byte offset) i3=gate_rs i4=gate_ag */
    PLOW_DOP_XREDUCE2 = 29,

    /* ===== FP8 ops. Renumbered to start at 30 (past tp's XREDUCE2=29) on the tp merge; the
     * fp8-consolidated branch had these at 24-33 which collides with the TP collectives. =========
     * FP8 DECODE GEMV (w8a16): fp8 e4m3 weight, bf16 activation. Twin of PLOW_DOP_GEMV, but the
     * weight row is uint8[K] (half the bytes -> ~2x the decode roofline) and carries a per-output-
     * channel f32 dequant scale applied ONCE in the epilogue.
     * t0=C(bf16) t1=x(bf16) t2=W(fp8) t5=w_scale(f32[N])   i0=M i1=N i2=K i4=a_row0. See op_gemm.h. */
    PLOW_DOP_GEMV_FP8 = 30,

    /* FP8 DECODE GEMV+GLU (w8a16): fp8 gate|up in ONE pass, act(gate)*up in the epilogue. Twin of
     * PLOW_DOP_GEMV_GLU. Two fp8 weight streams, each with its own per-output-channel f32 scale.
     * t0=fu t1=x t2=W_gate(fp8) t5=W_up(fp8) t3=gate_scale(f32[N]) t4=up_scale(f32[N])
     * i0=M i1=N i2=K i5=act. See op_gemm.h. */
    PLOW_DOP_GEMV_GLU_FP8 = 31,

    /* FP8 ACTIVATION QUANT (per-row/per-token). Reads a bf16 activation x[M,K], writes the OCP e4m3
     * quantized activation xq[M,K] (1 byte/elt) plus a per-row f32 dequant scale a_scale[M]. This is
     * the w8a8 prefill's activation half: a_scale[m] = rowmax|x[m,:]| / 448, xq[m,k] = round_e4m3(
     * x[m,k] / a_scale[m]). Emitted once per activation and reused by every fp8 GEMM that consumes
     * it (q/k/v share one, gate/up share one). Ideally fused into the producing norm's epilogue; a
     * separate op is the correctness-first fallback the design calls for.
     * t0=xq(fp8) t1=x(bf16) t2=a_scale(f32[M])   i0=M i1=K. See op_gemm.h d_quant_fp8. */
    PLOW_DOP_QUANT_FP8 = 32,

    /* FP8 PREFILL GEMM (w8a8): BOTH operands fp8 e4m3. The 2x-rate MFMA on gfx950 is the WIDE-K
     * mfma_scale_f32_32x32x64_f8f6f4 (measured 2x bf16), NOT mfma_f32_32x32x16_fp8_fp8 (measured 1x).
     * Twin of PLOW_DOP_GEMM.
     * A is the per-row-quantized activation (a_scale[M]) and B the per-channel-quantized weight
     * (w_scale[N]); the f32 accumulator is dequantized acc*a_scale[m]*w_scale[n] in the epilogue and
     * stored bf16. The accumulator layout is byte-identical to the bf16 mfma_f32_32x32x16, so the
     * epilogue + wave->column maps carry over; only the operand loads (fp8) and MFMA differ.
     * t0=C(bf16) t1=A(fp8) t2=B(fp8) t3=a_scale(f32[M]) t4=w_scale(f32[N])  i0=M i1=N i2=K i4=a_row0. */
    PLOW_DOP_GEMM_FP8 = 33,
    PLOW_DOP_GEMM_MED_FP8 = 34,   /* 128x128 fp8 tile */
    PLOW_DOP_GEMM_SMALL_FP8 = 35, /* 64x128 fp8 tile  */

    /* FP8 PREFILL GEMM+GLU (w8a8): fp8 gate|up in ONE pass, act(gate)*up in the epilogue. Twin of
     * PLOW_DOP_GEMM_GLU. Two fp8 weight streams + their per-channel scales, one shared a_scale.
     * t0=fu t1=A(fp8) t2=Wg(fp8) t5=Wu(fp8) t3=a_scale(f32[M]) t4=g_scale t6=u_scale  i0=M i1=N i2=K i5=act. */
    PLOW_DOP_GEMM_GLU_FP8 = 36,
    /* FP8 (e4m3) KV-CACHE ops (PLOW_FP8_KV). The decode KV stream is HBM-bound, so storing K/V as
     * e4m3 halves it (~2x the bandwidth-bound roofline) AND halves the KV footprint. All three are
     * mechanical fp8 twins: same math, HALF the KV bytes, plus a per-row f32 dequant scale.
     *
     * HEADNORM_ROPE_FP8: writes the K/V cache as e4m3 with a per-(token,kv_head) scale.
     *   t0=out(uint8 cache) t6=scale(f32[kv_head][ctx]); t1..t5,i,f,j as HEADNORM_ROPE.
     * FLASH_DECODE_FP8 / FLASH_PREFILL_FP8: read that fp8 cache. t3=K(fp8) t4=V(fp8), and
     *   t6=k_scale t7=v_scale; everything else as FLASH_DECODE / FLASH_PREFILL. */
    PLOW_DOP_HEADNORM_ROPE_FP8 = 37,
    PLOW_DOP_FLASH_DECODE_FP8 = 38,
    PLOW_DOP_FLASH_PREFILL_FP8 = 39,
    /* ===== MoE data-dependent counter-gate ops (plans/moe-plow-design.md §3, =====
     * plans/moe-ep-kernels.md §2-§3). Opcodes in the HIGH free range 40+ so they do NOT
     * collide with the tp collectives (24-29) or the fp8 merge (30+). These are the FIRST
     * ops whose BODY branches on a runtime buffer (the routing table): the counter DAG stays
     * static (deadlock-free, executed==total), each expert packet ALWAYS signals its counter
     * whether it computed or skipped, so the interp gate/signal loop is UNCHANGED. The
     * conditionality lives entirely inside plow_exec. */

    /* ROUTER (moe-ep-kernels §2b). One packet/token: GEMV x·Wr (n_exp experts) -> per-expert
     * score (sigmoid|softmax) -> k-pass masked argmax top-k with LOWEST-EXPERT-ID tie-break
     * (the bit-exactness linchpin) -> optional norm_topk (renormalise the k gates to sum 1) ->
     * xroute_scale -> writes routing_table[k] = (u32 expert_id, f32 gate), padding unused slots
     * with PLOW_EXPERT_UNUSED. Its completion counter is the data-dependent gate the K expert
     * slots wait on.
     *   t0=routing_table(out) t1=x t2=Wr   i0=H i1=n_exp i2=k i3=flags   f0=route_scale
     *   flags bit0 = scoring (1 sigmoid, 0 softmax), bit1 = norm_topk */
    PLOW_DOP_MOE_ROUTER = 40,

    /* EXPERT GATE/UP GEMV — the common expert segment's first half (moe-ep-kernels §3a), one
     * instance per top-k slot. Reads routing_table[slot].expert_id; if >= n_exp (sentinel) it
     * SKIPS (writes nothing, streams zero weight bytes) and the interp still signals its
     * counter; else wbase = expert_weight_table[expert_id] (a two-level indirection through a
     * table of device pointers) and it runs the fused act(gate·x)·(up·x) GEMV (identical
     * arithmetic to GEMV_GLU) into per-slot scratch fu[slot].
     *   t0=fu(out,[k,I_moe]) t1=x t2=routing_table t3=expert_weight_table
     *   i0=slot i1=I_moe(N) i2=H(K) i3=n_exp i5=act */
    PLOW_DOP_MOE_EXPERT_GLU = 41,

    /* EXPERT DOWN GEMV — the common expert segment's second half (moe-ep-kernels §3a). Same
     * sentinel skip; else wbase.down, runs W_down·fu[slot], multiplies by routing_table[slot].gate,
     * writes the gate-scaled partial part[slot]. On the skip path it zeroes part[slot] so the
     * combine sums a deterministic zero.
     *   t0=part(out,[k,H]) t1=fu t2=routing_table t3=expert_weight_table
     *   i0=slot i1=H(N) i2=I_moe(K) i3=n_exp */
    PLOW_DOP_MOE_EXPERT_DOWN = 42,

    /* COMBINE (moe-ep-kernels §3b) — the deterministic gather-combine. Waits on all k expert-down
     * slots + the shared expert, then out = residual + shared + sum_{j=0..k-1} part[j], f32
     * accumulate in FIXED slot order rounded to bf16 (independent of which expert finished first —
     * the MoE bit-exactness obligation). shared==NONE for a 0-shared-expert config.
     *   t0=out t1=residual t2=shared? t3=part_base([k,H])   i0=H i1=k */
    PLOW_DOP_MOE_COMBINE = 43,

    /* Block-fp8 (DeepSeek/GLM weight_block_size [128,128]) weight-stream variants (free band 44-49).
     * The weight is e4m3 and dequant uses a per-[128 out][128 K] f32 scale grid (ceil(N/128) x
     * ceil(K/128), row-major) folded into the K-reduction per 128-K block — NOT the per-channel
     * epilogue scale of the 30-39 fp8 ops. x stays bf16 (w8a16, the decode weight-stream path). */
    PLOW_DOP_GEMV_FP8_BLK = 44,          /* block-fp8 decode GEMV: t5=w_scale grid  (op_gemm.h)    */
    PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK = 45,/* block-fp8 expert gate/up: t3=wtab t4=stab (scale bases)*/
    PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK = 46,/* block-fp8 expert down:   t3=wtab t4=stab             */
    PLOW_DOP_DENSE_GLU_FP8_BLK = 47,     /* block-fp8 DENSE SwiGLU gate/up: t2=Wg t5=Wu t3=Sg t4=Su
                                          * i0=N(inter) i1=K(hidden) i5=act; dense DOWN reuses op 44 */
    /* GROUPED block-fp8 experts (op-count collapse for GLM M=1 decode). ONE packet loops ALL top-k
     * slots that ops 45/46 did one-per-packet — one counter edge + one interp dispatch instead of k.
     * Bit-identical output (same per-slot wave_dot_fp8_blk + slot layout). EP: a null weight base
     * (expert not owned by this rank) is skipped exactly like the eid>=n_exp sentinel. */
    PLOW_DOP_MOE_GROUP_GLU_FP8_BLK = 48, /* grouped expert gate/up: t0=fu[k,I] t1=x t2=tab t3=wtab
                                          * t4=stab; i0=k i1=I_moe i2=H i3=n_exp i5=act */
    PLOW_DOP_MOE_GROUP_DOWN_FP8_BLK = 49,/* grouped expert down:   t0=part[k,H] t1=fu t2=tab t3=wtab
                                          * t4=stab; i0=k i1=H i2=I_moe i3=n_exp */

    /* DeepSeek MLA (Multi-head Latent Attention) — the flash READ path only.  [DEEPSEEK-MLA]
     * Opcodes reserved at 50+ to avoid collision with the in-flight fp8-merge (30+) and MoE
     * (40+) work (sparse-attn-design.md §4 renumbered into that free band). The WRITE path
     * reuses existing ops: GEMV/GEMM for the c_kv/k_rope down-projections, HEADNORM_ROPE
     * (n_kv_head=1, hd=qk_rope_dim) for the shared rope key, RMSNORM on the latent.
     *
     * These are the two new device inner loops + the epilogue fold; validated STANDALONE via
     * test_kernels.hip (mla_flash_decode_512 / mla_o_uv_fold_512) against the Rust oracle
     * (runtime/tests/mla_ref.rs). Interpreter dispatch wiring (interp.hip switch + dev.rs +
     * devbuild) is deferred to the emitter-integration phase, after glm-prototype lands the
     * shared MoE core. FLASH_MLA_MERGE reuses the existing FLASH_MERGE at D=kv_lora_rank=512
     * (no new opcode needed — the merge is dimension-generic). */
    PLOW_DOP_FLASH_MLA_DECODE = 50,  /* q_abs.c_kv + q_rope.k_rope; PV on the latent (D=DK) */
    PLOW_DOP_FLASH_MLA_PREFILL = 51, /* MFMA twin (not yet built)                            */
    PLOW_DOP_O_UV_FOLD = 52,         /* per-head W_uv fold: latent accumulator -> v_head_dim */
    /* Sparse top-k DSA (sparse-attn-design.md §3) — reserved, not yet built. */
    PLOW_DOP_ATTN_SELECT = 53,          /* on-device top-k KV selection -> idx table */
    PLOW_DOP_FLASH_GATHER_DECODE = 54,  /* gathered flash (base = dense or MLA)       */
    PLOW_DOP_FLASH_GATHER_PREFILL = 55,

    /* ROUTER TOP-K tail (the router SPLIT): the score matmul logit=x·Wr is now the ordinary multi-CU
     * GEMV (op 6, all-CU), and this cheap 1-CU op does score-transform + group-limited top-k + norm +
     * scale over the <=256 precomputed logits. t0=table t1=logit(bf16[n_exp]) t3=bias(f32[n_exp] or 0);
     * i1=n_exp i2=k i3=flags; f0=route_scale. Byte-for-byte the selection logic of MOE_ROUTER (40),
     * minus the GEMV — moves the 1141us single-CU score dot onto the machine-filling GEMV. */
    PLOW_DOP_MOE_ROUTER_TOPK = 56,

    /* FUSED MLA merge + W_uv fold (d_mla_merge_fold) — replaces FLASH_MERGE<512> + O_UV_FOLD on the
     * MLA decode path (kills the separate merge pass + Olat round-trip + a gate). t0=O(v_head)
     * t1=Opart(f32) t2=mlpart(f32) t3=Wuv; i0=n_batch i1=n_head i2=V i4=nsplit. */
    PLOW_DOP_MLA_MERGE_FOLD = 57,

    /* DSA lightning-indexer SCORE (d_index_score / d_index_score_fast, op_attention.h): for every KV
     * position t, score[b][t] = sum_h w[b][h]*ReLU(q_idx[b][h].k_idx[b][t]) (scale folded; selection
     * scale-invariant). q_idx/k_idx arrive projected + k_norm'd + interleaved-RoPE'd; w = weights_proj·x.
     * t0=Score(f32) t1=Qidx(bf16) t2=Kidx(bf16) t3=W(bf16) t4=kv_len(i32); i0=n_batch i1=index_heads
     * i2=kv_stride i3=index_head_dim; f0=scale. Grid-strided over all CUs. [GLM52-DSA] */
    PLOW_DOP_INDEX_SCORE = 58,

    /* DSA lightning-indexer top-k SELECT (d_index_select_coop): ONE cooperative launch of exactly n_cu
     * co-resident WGs, 7-pass radix over the monotone packed key, emits exactly top_k highest-score
     * positions (lowest-index tie-break) into idx. t0=idx(i32) t1=Score(f32) t2=gHist(u32[7*256])
     * t3=gCtl(u32[3]); i0=len i1=top_k. Host zeroes gHist/gCtl once; kernel leaves them clean. [GLM52-DSA] */
    PLOW_DOP_INDEX_SELECT = 59,

    /* LayerNorm WITH bias + mean-subtract (d_layernorm_bias, op_norm.h): the DSA indexer k_norm
     * (nn.LayerNorm(128, eps=1e-6, bias) — the only non-RMS norm in GLM-5.2). y=(x-mean)*rsqrt(var+eps)*g+b.
     * t0=out t1=x t2=gamma t3=beta; i0=rows i1=feat i3=out_row0; f0=eps. [GLM52-DSA] */
    PLOW_DOP_LAYERNORM = 60,

    /* ===== Gemma-4 26B-A4B bf16 sparse-MoE DECODE ops (plans/rtx-08-gemma4-moe-26b.md) =====
     * bf16 twins of the block-fp8 MoE ops (40-49), specialised to the Gemma-4 MoE block:
     * SOFTMAX router with a weightless-RMS + per-channel scale + H^-0.5 pre-transform and a
     * PER-EXPERT gate scale, FUSED gate_up expert weights ([E,2I,H]), gelu_tanh activation.
     * expert_weight_table (ewt) = 2 u64 per expert: {gate_up base, down base}. */

    /* ROUTER: r=weightless_rms(resid); h2=r*scale[h]*root; logit=proj@h2; softmax; top-k
     * (lowest-id tie); norm_topk; gate*=per_expert_scale. Writes routing_table[k]={u32 id,f32 gate}.
     * ONE block. t0=table(out) t1=resid t2=proj t3=scale[H] t4=per_expert_scale[E]
     * i0=H i1=n_exp i2=k  f0=root(=H^-0.5) f1=eps. */
    PLOW_DOP_MOE_ROUTER_GEMMA = 61,

    /* EXPERT GATE/UP (fused): fu[slot][n]=gelu_tanh(gate_e.x)*(up_e.x). Flat one-warp-per-output.
     * t0=fu([k,I]) t1=x t2=table t3=ewt  i0=k i1=I_moe(N) i2=H(K) i3=n_exp. */
    PLOW_DOP_MOE_EXPERT_GLU_GEMMA = 62,

    /* EXPERT DOWN: part[slot][h]=gate_slot*(down_e[h].fu[slot]), f32. Flat one-warp-per-output.
     * t0=part([k,H],f32) t1=fu t2=table t3=ewt  i0=k i1=H(N) i2=I_moe(K) i3=n_exp. */
    PLOW_DOP_MOE_EXPERT_DOWN_GEMMA = 63,

    /* COMBINE: moe[h]=sum_slot part[slot][h] (f32, fixed slot order) -> bf16.
     * t0=moe(out) t1=part([k,H])  i0=H i1=k. */
    PLOW_DOP_MOE_COMBINE_GEMMA = 64,

    /* Per-output-channel e4m3 twins of the Gemma fused-expert bodies. These are decode-only:
     * ewt[e][0,1] points at fp8 gate_up/down rows and est[e][0,1] at their f32 row scales.
     * Gate/up scales are [2*I], down scales [H], independently for every expert. */
    PLOW_DOP_MOE_EXPERT_GLU_GEMMA_FP8 = 65,
    PLOW_DOP_MOE_EXPERT_DOWN_GEMMA_FP8 = 66,

    /* Gemma router split, opt-in at packet emission. SCORE uses eight warps/block for eight
     * coalesced expert dots; each block recomputes the (tiny) weightless-RMS scalar. The TOPK
     * tail waits for every score block and retains the old serial softmax/selection order.
     * SCORE: t0=score(f32[E]) t1=resid t2=proj t3=scale; i0=H i1=E; f0=root f1=eps.
     * TOPK:  t0=table t1=score(f32[E]) t2=per_expert_scale; i1=E i2=k. */
    PLOW_DOP_MOE_ROUTER_GEMMA_SCORE = 67,
    PLOW_DOP_MOE_ROUTER_GEMMA_TOPK = 68,
    /* Experimental association-changing fast twin of SCORE (67): ordinary per-lane f32 dot +
     * warp reduction. Same operands; default-off and retained separately from the exact scorer. */
    PLOW_DOP_MOE_ROUTER_GEMMA_SCORE_FAST = 69,
    /* Fused combine + RMSNorm + residual-add for Gemma-4 MoE. Replaces the 3-op tail
     * (MoeCombineGemma -> RmsNorm -> Residual). t0=out t1=part t2=resid t3=gamma i0=H i1=k f0=eps */
    PLOW_DOP_MOE_COMBINE_NORM_GEMMA = 70,
    /* Fused pre-FFN-norm-2 + expert GLU. Inline RMSNorm before the expert dots,
     * eliminating a separate RmsNorm packet. t0=fu t1=resid t2=table t3=ewt t4=gamma
     * i0=k i1=I i2=H i3=n_exp f0=eps */
    PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA = 71,
    /* Fused MoE layer tail: (MoeCombineNormGemma -> NormResidualNorm) in ONE packet.
     * b = h1 + RMSNorm(sum(part), g_pf2); x = (x + RMSNorm(b, g_po)) * ls; hn = RMSNorm(x, gn).
     * Bit-exact to the pair (b and x rounded to bf16 between reductions).
     * t0=hn t1=x(in/out) t2=part(f32[k,H]) t3=h1 t4=g_pf2 t5=g_po t6=gn
     * i0=H i1=k  f0=eps f1=layer_scalar */
    PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA = 72,

    /* ===== Gemma-4 26B-A4B bf16 grouped-MoE PREFILL ops (plans/p9-26b-prefill-moe.md) =====
     * Token-sorted grouped expert GEMM for T>1. Op ids 73+ (71/72 left free; 72 reserved for
     * another in-flight change). These build ONLY in the prefill (_pf) object. */

    /* T-TOKEN ROUTER: block-per-token loop of the exact decode router d_moe_router_gemma. Writes
     * routing_table[token*k + j] = {u32 eid, f32 gate}. Bit-identical per token to decode.
     * t0=table(out) t1=resid([T,H]) t2=proj t3=scale[H] t4=per_expert_scale[E]
     * i0=H i1=n_exp i2=k i3=T  f0=root(=H^-0.5) f1=eps. */
    PLOW_DOP_MOE_ROUTER_GEMMA_PF = 73,

    /* ALIGN/SORT: ONE block. Histogram the T*k routing slots by expert, padded prefix to BM=128
     * tile boundaries, scatter (token,slot,gate) into expert-contiguous gathered rows. Pad rows
     * get row_token = PLOW_EXPERT_UNUSED. meta layout (int32): [0,n_exp) rowoff (padded start),
     * [n_exp,2n_exp) cnt, [2n_exp,3n_exp+1) tile_prefix (tile_prefix[n_exp]=total_tiles).
     * row_token = source token id per gathered row (UNUSED for pad); row_partidx = token*k+slot
     * (the destination row of part[T*k,H]; UNUSED for pad); row_gate = the slot's gate.
     * t0=meta(i32) t1=table t2=row_token(u32) t3=row_partidx(u32) t4=row_gate(f32)
     * i0=T i1=n_exp i2=k. */
    PLOW_DOP_MOE_ALIGN_GEMMA_PF = 74,

    /* GROUPED GATE/UP GEMM + GeGLU (fused expert weights, gathered A). Flat tile list over
     * (workitem->(expert,m_tile)) x n_tiles; A rows gathered from xn2 via row_token; B = fused
     * gate_up ewt[e*2+0] ([2*I,H], gate rows [0,I) up rows [I,2I)); GeGLU epilogue; output to
     * fu_gathered[row*I_moe + n]. Reuses the d_gemm tile body (cp.async, m16n8k16, 128x128).
     * t0=fu_g t1=xn2([T,H]) t2=ewt t3=meta t4=row_token  i0=I_moe(N) i1=H(K) i2=n_exp i5=act. */
    PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF = 75,

    /* GROUPED DOWN GEMM + gate-scale + scatter. A = fu_gathered (contiguous per segment),
     * B = down ewt[e*2+1] ([H,I_moe]), N=H K=I_moe; epilogue multiplies row_gate and SCATTERS to
     * part[row_partidx*H + h] via row_partidx; pad rows (row_partidx==UNUSED) skipped.
     * t0=part([T,k,H] f32) t1=fu_g t2=ewt t3=meta t4=row_partidx t5=row_gate
     * i0=H(N) i1=I_moe(K) i2=n_exp. */
    PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF = 76,

    /* T-ROW COMBINE + SANDWICH: block-per-token loop of d_moe_combine_norm_gemma. For each token
     * out[t] = RMSNorm(Σ_slot part[t][slot], gamma) + h1[t] (h1 = post_ffn_norm_1(dense)).
     * t0=out t1=part([T,k,H]) t2=h1([T,H]) t3=gamma  i0=H i1=k i2=T  f0=eps. */
    PLOW_DOP_MOE_COMBINE_NORM_GEMMA_PF = 77,

    /* SplitZip (bf16 lossless) DECODE GEMV twins (p9-v2 C-1). Each compressed weight is one
     * self-describing blob tensor: lo|cd|eoff|epos|eval (see op_gemm.cuh SzBlob). The bf16
     * output is BIT-IDENTICAL to the plain GEMV twin — only the weight bytes crossing HBM shrink
     * 1.33x. Flag-gated by PLOW_NV_SZ in the interpreter.
     * Each blob is SELF-DESCRIBING (16-byte header {nesc, exp_base} + sized escape reservation),
     * so the op carries no data-dependent fields — the loader fills the header at bind time.
     *   GEMV_SZ      t0=C t1=x t2=blob            i0=M i1=N i2=K.
     *   GEMV_GLU_SZ  t0=C t1=x t2=gblob t3=ublob  i0=M i1=N i2=K i5=act. */
    PLOW_DOP_GEMV_SZ = 78,
    PLOW_DOP_GEMV_GLU_SZ = 79,

    /* E5 (rtx-19): lm_head GEMV with the greedy-argmax epilogue fused in (PLOW_FUSE_ARGMAX).
     * Twin of PLOW_DOP_GEMV (bf16, M=1), but each block folds its owned vocab slice into one
     * packed-u64 argmax partial part[block] in the epilogue — reproducing SOFTCAP->ARGMAX bit
     * for bit (f0=cap, 0=none) so the token is byte-identical — and still writes logits.
     *   GEMV_ARGMAX  t0=C(logits) t1=x t2=W t3=part(u64[nblk])  i0=1 i1=N i2=K i4=a_row0  f0=cap. */
    PLOW_DOP_GEMV_ARGMAX = 80,

    /* beat26b: fp8 (w8a8) GROUPED-MoE prefill twins of ops 75/76. BOTH operands e4m3
     * (mma.sync.m16n8k32); activation is per-row e4m3 + f32 scale from QUANT_FP8, weight from ewt
     * (e4m3) + per-output-channel scale from est. Gated behind PLOW_NV_W8A8 in the interpreter.
     *   GLU  t0=fu t1=xq8([T,H] e4m3) t2=ewt t3=meta t4=row_token t5=ascale(f32[T]) t6=est
     *        i0=I_moe i1=H i2=n_exp i5=act.
     *   DOWN t0=part t1=fu8([pad,I] e4m3) t2=ewt t3=meta t4=row_partidx t5=row_gate t6=est
     *        t7=fscale(f32[pad])  i0=H i1=I_moe i2=n_exp. */
    PLOW_DOP_MOE_GROUP_GLU_GEMMA_PF_W8A8 = 81,
    PLOW_DOP_MOE_GROUP_DOWN_GEMMA_PF_W8A8 = 82,

    /* Nemotron-3 Mamba-2 SSD mixer (plans/block-asset-harness.md §7, M4). NEW op family (90; 83-89 a
     * gap after the MoE-prefill band). Mirrors packet::dev::DevOp::Mamba2Scan. The mixer CORE only —
     * in_proj/out_proj are ordinary GEMV/GEMM. Causal depthwise conv1d + SiLU over (x,B,C), the
     * selective SSD scan with per-head scalar decay, D skip, gated RMSNorm; reads+writes the carried
     * conv_state + ssm_state (prefill = full scan, decode = single step, SAME op). Correctness-first,
     * UNVERIFIED on GPU (op_mamba.cuh).
     *   MAMBA2_SCAN t0=out([T,d_inner] bf16) t1=xBC([T,conv_dim] bf16) t2=dt([T,n_head] bf16 raw)
     *     t3=z([T,d_inner] bf16) t4=conv_w([conv_dim,d_conv] bf16)
     *     t5=params(f32: A_log[n_head]|D[n_head]|dt_bias[n_head]|conv_b[conv_dim]|norm_w[d_inner])
     *     t6=conv_state(f32[d_conv-1,conv_dim] in/out) t7=ssm_state(f32[n_head,head_dim,d_state] in/out)
     *     i0=T i1=d_inner i2=n_head i3=head_dim i4=d_state i5=n_groups i6=d_conv i7=conv_dim  f0=eps. */
    PLOW_DOP_MAMBA2_SCAN = 90,

    /* MXFP4 DECODE GEMV (w4a16) — OCP microscaling: e2m1 weights + one E8M0 scale per 32 K.
     * Decode is weight-bandwidth-bound, so this is where fp4 pays: 4.25 effective bits/element
     * (0.5 weight byte + 1 scale byte per 32) against fp8's 8, moving the roofline ~1.88x over
     * PLOW_DOP_GEMV_FP8 and ~3.76x over bf16 PLOW_DOP_GEMV.
     *
     * The E8M0 scale is folded into the fp4->bf16 convert, which is EXACT here (an MX scale is a
     * power of two by construction) — unlike PLOW_DOP_GEMV_FP8_BLK's arbitrary-f32 block scale,
     * which must stay a separate multiply. So there is no dequant in the epilogue and no scale
     * tensor read per output; see amd/amd_common.h fp4_to_bf16v8x4.
     *
     * Layout: W packed 2 fp4/byte, row stride K/2 bytes, low nibble = even k. S one E8M0 byte per
     * 32-K block, row stride K/32 bytes. A lane's 16-byte load is exactly one 32-element block.
     * t0=C(bf16) t1=x(bf16) t2=W(fp4) t3=S(e8m0)   i0=M i1=N i2=K. See op_gemm.h d_gemv_mxfp4. */
    PLOW_DOP_GEMV_MXFP4 = 91,

    /* MXFP4 DECODE fused gate|up GEMV+GLU (w4a16) — the mxfp4 twin of PLOW_DOP_GEMV_GLU_FP8. Gate
     * and up are two fp4 weight matrices, each with its own E8M0 scale row; the scale folds into the
     * fp4->bf16 convert (no epilogue dequant), and the SwiGLU fuses act(g)*u in one packet. Same
     * w4a16 bandwidth win as PLOW_DOP_GEMV_MXFP4, applied to the dense/shared SwiGLU decode.
     * t0=C(bf16) t1=x(bf16) t2=Wg(fp4) t5=Wu(fp4) t3=Sg(e8m0) t4=Su(e8m0)
     * i0=M i1=N i2=K i5=act. See op_gemm.h d_gemv_glu_mxfp4. */
    PLOW_DOP_GEMV_GLU_MXFP4 = 92,

    /* MXFP4 (w4a16) PREFILL GEMM — bf16 activations x packed-2/byte fp4 weights with E8M0 scale
     * rows (K/32 bytes/row). Reuses the bf16 wide-K MFMA; only the weight fetch dequants fp4->bf16
     * with the MX scale folded exactly, so this is the fp4 weight-bandwidth win at M>1 without an
     * activation-quant op. t0=C(bf16) t1=A(bf16) t2=W(fp4) t3=wscale(e8m0)  i0=M i1=N i2=K.
     * See op_gemm.h d_gemm_mxfp4. */
    PLOW_DOP_GEMM_MXFP4 = 93,

    PLOW_DOP__COUNT
};

/* Sentinel expert id a router writes for an unused top-k slot; the expert body skips compute
 * (still signals) when expert_id >= num_experts. Mirrors packet::dev::EXPERT_UNUSED. */
#define PLOW_EXPERT_UNUSED 0xFFFFFFFFu

/* A gate: wait until counters[id] >= threshold. `threshold` is the number of
 * producing workgroups, so it is determined by how the producer was sliced. */
typedef struct {
    uint32_t id;
    uint32_t threshold;
} PlowWait;

/* One float-or-integer operand word. The 64-byte instruction has three of these; which
 * member an op reads is fixed per opcode (see the fj[] slot map below). */
typedef union {
    float    f;
    uint32_t u;
} PlowFJ;

/* One instruction. Fixed 64-byte stride — no variable-length records on device.
 *
 * Wait/succ metadata is NOT here: it lives on EVERY PlowStreamEnt (coarse entries carry
 * the op's coarse lists, PLOW_SE_FINE entries their per-slice ones), so the interpreter
 * reads gates from the stream entry unconditionally.
 *
 * fj[] slot map (was f[4] + j[2]; f[2]/f[3] were dead across all opcodes, and no opcode
 * reads both f[1] and j[0] — norm/MoE vs attention families are disjoint):
 *   fj[0].f = old f[0]
 *   fj[1].f = old f[1]   |   fj[1].u = old j[0]   (per-opcode, mutually exclusive)
 *   fj[2].u = old j[1]
 * The compiler asserts the exclusivity at pack time (packet::dev::DevInst::pack).
 *
 * Field ORDER puts t[] at byte 16: the stride is 64, so t[8] is 16-byte aligned and
 * the interpreter can fetch all eight handles with one vector load (per-site u16
 * loads measured ~+0.4% decode TPOT on sm_120). */
typedef struct {
    uint16_t op;
    uint16_t blocks;   /* workgroups this op is sliced across (>= 1)   */
    PlowFJ   fj[3];    /* float/integer operands — slot map above      */
    uint16_t t[8];     /* tensor handles; PLOW_TENSOR_NONE = absent    */
    uint32_t i[8];     /* integer operands                             */
} PlowDevInst;

/* THE KV CACHE IS HEAD-MAJOR: [kv_head][ctx][head_dim], NOT [ctx][kv_head][head_dim].
 *
 * Token-major looks like the natural choice -- appending a token is one contiguous write across
 * every head -- and it is what this ran with for a long time. But it is the READER that decides.
 * flash_decode walks ONE head at a time, so under token-major its consecutive KV rows sit
 * n_kv_head * head_dim apart: at Gemma-31B that is 16 x 256 halves = 8 KB of stride around
 * 512 bytes of payload. A workgroup's 512 threads then span 512 KB of address space to read
 * 256 KB of data, and every row lands in a different DRAM page.
 *
 * Head-major makes one head's rows CONTIGUOUS, so those same 512 threads read 256 KB of
 * dead-sequential memory. And the writer pays nothing for it: headnorm_rope already runs one
 * wave per (token, head), so each wave still writes its own contiguous 512 bytes -- just at a
 * different address.
 *
 * The stride is the ALLOCATED context, not the current length, so it has to be carried:
 *   HEADNORM_ROPE   j0 = out_stride   (0 = plain [ntok][nhead][hd] -- that is the q norm)
 *   FLASH_PREFILL   j0 = kv_stride
 *   FLASH_DECODE    i3 = kv_stride    (it already had one) */

/* One entry in a CU's stream: run `inst`, taking share `slice` of its work.
 *
 * PER-SLICE GATES (PLOW_SE_FINE). By default an op's wait/succ lists live on the
 * INSTRUCTION, so all `blocks` workgroups running it block on the same counters — every
 * op is therefore a full N-way barrier, and a consumer waits for the SLOWEST producer
 * workgroup even when it only needs the output of a handful of them.
 *
 * That wait is not free. Measured on the real model (decode, ctx=3326): 256 CUs doing
 * IDENTICAL gemv work finish across a 9.6-16.6 us spread, and summed over the critical
 * path the gate spends 2.63 ms of a 16.9 ms token waiting for one straggler after half
 * the machine is already done. The straggler is DIFFUSE, not one structurally slow CU
 * (per-CU mean duration spreads only 5.1%, and the top-5 "finished last" CUs account for
 * 8% of instructions against 2% for random) — so the wait is recoverable.
 *
 * When PLOW_SE_FINE is set, this entry's OWN wait/succ lists replace the instruction's,
 * so slice `s` blocks only on the producer slices that actually feed it (q_proj column
 * block -> the 8 CUs owning head h, and so on). The counter protocol is otherwise
 * unchanged.
 *
 * WHY THIS CANNOT DEADLOCK, and the one rule that keeps it that way: `devbuild` appends
 * ops in topological order, so for any dependency A -> B EVERY slice of A precedes EVERY
 * slice of B in EVERY CU's stream. A fine list can only LOWER a threshold or NARROW a wait
 * set — it can never make a workgroup wait on something issued later in its own stream.
 * The moment a scheduler is added that INTERLEAVES tiles across ops, that argument dies
 * and this needs the relay machinery in plans/fine-counter-deadlock-fix.md. Do not reorder
 * streams without reading that file.
 *
 * Coarse ops leave `flags == 0` and cost nothing: the interpreter reads the instruction's
 * lists exactly as before, so the coarse path stays bit-identical and remains the A/B
 * reference. */
#define PLOW_SE_FINE 1u /* this entry carries its own wait/succ lists */

/* This entry's wait/succ counters live in the SYSTEM-scope, PEER-MAPPED `xctr` region
 * (a cross-GPU collective), not the agent-scope local `counters`. Orthogonal to
 * PLOW_SE_FINE. Set by the TP compiler on XREDUCE/XFLASHMERGE/XARGMAX_FIN packets and on
 * the producing GEMV whose successor is a cross-GPU "partial ready" bump. See
 * plans/tp-design.md §6a. Coarse xctr programs leave this clear and cost nothing. */
#define PLOW_SE_XCTR 2u /* wait/succ counters are cross-GPU (xctr, system scope)   */

typedef struct {
    uint32_t inst;
    uint32_t slice;
    uint32_t wait_ofs; /* index into the PlowWait table   (PLOW_SE_FINE only) */
    uint32_t succ_ofs; /* index into the counter-id table  (PLOW_SE_FINE only) */
    uint16_t wait_len;
    uint16_t succ_len;
    uint16_t flags;
    uint16_t seg; /* wave-class segment id (segmented dispatch); 0 when unsegmented. Was _pad. */
} PlowStreamEnt;

/* Per-(workgroup, packet) trace record.
 *
 * plow exists to schedule a NETWORK, not to run one op fast: packets from
 * different ops overlap on different CUs, gated by counters. A single-op
 * benchmark shows none of that, so the only way to see whether the schedule is
 * any good is to timestamp every packet boundary and look at the timeline.
 *
 * Clock: s_memrealtime, NOT clock64/s_memtime. s_memrealtime is a constant-rate
 * ~100 MHz counter that is coherent across every CU, so records from different
 * workgroups can be compared on one timeline. The shader clock is per-SIMD and
 * moves with DVFS -- and we MEASURED this GPU dropping 2.2 -> 1.58 GHz under load,
 * so a shader-clock timeline would be silently skewed exactly when it matters.
 *
 * The slot is stream_ofs[cu] + pc, which is unique per (workgroup, packet) -- so
 * no atomics, no ring buffer, no lost records, and the trace is deterministic. */
typedef struct {
    uint32_t cu;    /* workgroup id                                     */
    uint32_t pc;    /* index into this workgroup's stream               */
    uint32_t inst;  /* instruction index                                */
    uint16_t op;    /* PLOW_DOP_*                                       */
    uint16_t slice; /* which share of the op this workgroup took        */
    uint64_t t_arrive; /* reached the packet                            */
    uint64_t t_ready;  /* its wait-counters cleared (t_ready - t_arrive = STALL) */
    uint64_t t_end;    /* finished the op body                          */
} PlowTraceRec;

/* Everything the interpreter needs; passed once as the kernel's args. */
typedef struct {
    const PlowDevInst*   insts;
    const PlowStreamEnt* stream;     /* flattened, indexed by stream_ofs[cu] */
    const uint32_t*      stream_ofs; /* [n_cu] */
    const uint32_t*      stream_len; /* [n_cu] */
    const PlowWait*      waits;
    const uint32_t*      succs;
    uint32_t*            counters;   /* zeroed by the host before each run */
    void* const*         tensors;    /* device pointer table */
    PlowTraceRec*        trace;      /* NULL disables tracing entirely       */
    /* Segmented dispatch: the interpreter runs ONLY stream entries whose `seg == cur_seg`, so the
     * host can relaunch it once per wave-class segment (see plans/segmented-dispatch.md). An
     * unsegmented program has n_seg==1 and every entry seg==0, so cur_seg==0 runs everything. */
    uint32_t             cur_seg;
    uint32_t             _segpad;
    /* Global-queue interpreter (Experiment E1, built only under PLOW_GLOBAL_QUEUE). The static
     * kernel never reads these; the host leaves them NULL unless PLOW_GLOBAL_QUEUE is selected. */
    const PlowStreamEnt* gq_stream;  /* op-major (topological) permutation of `stream`          */
    const uint32_t*      gq_seg_ofs; /* [n_seg+1] segment window bounds into gq_stream           */
    uint32_t*            gq_cursor;   /* 1-word shared fetch-add cursor, zeroed per launch        */
    /* ===== CROSS-GPU (tensor-parallel) fields. Single-GPU runs leave these NULL/0 =====
     * Appended AFTER gq_cursor so every existing field (notably `trace`) keeps its offset
     * and the ABI-lock test only sees the size grow. See plans/tp-design.md §6a, §12.
     *
     * `xctr` points INTO this rank's own `peer_scratch[rank]` reduction region, at its
     * cross-GPU counter sub-region (SYSTEM-scope, peer-mapped, PLOW_CTR_STRIDE-strided).
     * The per-rank counter offset is `(char*)xctr - (char*)peer_scratch[rank]`, so a
     * producer signals peer r at `peer_scratch[r] + that_offset` — no 5th field needed.
     * The owner polls its LOCAL `xctr`. Peers write it with a system-scope release RMW. */
    uint32_t*            xctr;         /* cross-GPU, SYSTEM-scope, peer-mapped counters   */
    void* const*         peer_scratch; /* [n_gpu] each rank's peer-mapped reduction region*/
    uint32_t             rank;         /* this GPU's TP rank                              */
    uint32_t             n_gpu;        /* TP degree (1 = single-GPU, fields unused)       */
} PlowProgram;

/* Workgroup geometry of the persistent interpreter. The HOST needs this to size
 * the dispatch, the DEVICE needs it to stride its loops, and Rust needs it to build
 * the packet stream -- so it lives here, in the one header all three share, rather
 * than as a literal repeated at every launch site.
 *
 * 8 waves, not 4. The GEMM's ping-pong schedule is two groups of 4 waves with one wave
 * of each group per SIMD; at 4 waves there is no co-resident partner on a SIMD and the
 * memory/MFMA overlap has nothing to overlap with. Worth ~+7% through the interpreter
 * (573 vs 533 TF/s mean over the Gemma-31B projections) -- real, and free, but not the
 * 2.2x an earlier note claimed: THAT number was measured through a 4-byte-aligned LDS
 * arena and evaporated once ds_read_b128 was restored. */
#ifndef PLOW_WG_WAVES
#define PLOW_WG_WAVES 8
#endif
#define PLOW_WG_THREADS (PLOW_WG_WAVES * 64)

/* GQA FUSION FACTOR: query heads carried by ONE flash-decode work item.
 *
 * The KV cache is the only tensor in the network with un-exploited reuse. A work item used to be
 * (query_head, split), so each KV row crossed HBM once per query head sharing it — GQA times. On
 * Gemma-4 31B that is 2x on the sliding layers and EIGHT times on the full ones (32 heads over 4
 * KV heads), and the full layers carry 57% of all KV traffic because they hold the whole context.
 * Fusing GF query heads into one work item reads each row once and dots it against all of them:
 * 3.86 -> 1.11 GB per token.
 *
 * It lives HERE because two parties must agree on it. The kernel instantiates
 * d_flash_decode<D, GF>, and plowc must raise `nsplit` by the same GF — fusing shrinks n_work
 * from n_head*nsplit to n_kv_head*nsplit, and without more splits the machine would go idle.
 * Keyed on head_dim because that is what the interpreter has at dispatch time, and on Gemma the
 * two head dims are exactly the two GQA ratios. */
/* GF = 2 ON THE FULL LAYERS TOO, and this was NOT the expected answer.
 *
 * n_work = (n_head/GF) * nsplit, so GF trades KV traffic against parallelism:
 *
 *   GF=8  -> 4 head-groups -> only 64 of 256 CUs busy, each KV row read ONCE   (min traffic)
 *   GF=2  -> 16 groups     -> all 256 CUs busy,        each KV row read 4x     (max parallelism)
 *
 * The obvious model — parallelism wins at short context where the cache is small, bandwidth wins
 * at long context where it is huge — predicts GF should GROW with ctx, and that a decode bucket
 * is needed to switch it. MEASURED, on the real model:
 *
 *              ctx=3.3k    ctx=32k    ctx=128k
 *     GF=2      15.7 ms     17.3 ms     23.0 ms     <- best at EVERY length
 *     GF=4      15.9        17.3          -
 *     GF=8      16.0        18.1        25.6
 *
 * GF=2 wins everywhere, and even at 128k where GF=2 reads FOUR TIMES the KV. The reason is that
 * flash_decode runs at ~16% of the memory roofline: it is not bandwidth-bound, so the extra reads
 * are nearly free, while 192 idle CUs are not. Traffic you are not bound by is not a cost.
 *
 * So there is NO context-dependent kernel parameter to bucket on, and the decode program can stay
 * a single program. (Sliding layers have GQA=2, so GF=2 fuses them completely; full layers have
 * GQA=8, so GF=2 is a partial fusion that reads each row 4x rather than 8x.) */
#ifndef PLOW_FA_GF_FULL
#define PLOW_FA_GF_FULL 2
#endif
#define PLOW_FA_GF(HD) ((HD) == 512 ? PLOW_FA_GF_FULL : 2)

/* SLIDING-WINDOW KV RING.
 *
 * 50 of Gemma-4's 60 layers only ever look back `sliding_window` = 1024 tokens, but the cache was
 * allocated for the FULL context on every layer. At ctx=128k that is 110 GiB of KV of which
 * **99 GiB is never read**. It is what makes 128k cost 213 GiB instead of ~114, and it is the gate
 * on batch > 1 at long context.
 *
 * So a sliding layer's cache becomes a RING of PLOW_KV_RING rows, indexed `row & (RING-1)`.
 * A full-attention layer keeps a linear cache of `ctx` rows and passes mask = 0xFFFFFFFF, so the
 * AND is a no-op there — one `v_and_b32`, no branch, no runtime flag in the inner loop.
 *
 * WHY THE RING IS NOT 1024. A prefill CHUNK of C tokens has queries at [c0, c0+C), which between
 * them need KV rows [c0-1023, c0+C-1] — a span of W + C - 1. And the chunk writes all C of its
 * rows before flash reads any of them, so a row must not be clobbered before it is used:
 *
 *     RING >= window + max_chunk - 1
 *
 * With RING = 8192 and window = 1024 the largest legal chunk is 7169, so PLOW_MAX_CHUNK = 4096
 * (the next bucket down) is comfortably safe. The prefill chunker must not exceed it.
 *
 *     sliding KV at ctx=128k:  100 GiB  ->  6.7 GiB
 *
 * A ring is only possible AT ALL because prefill is chunked. A one-shot prefill writes every row
 * of K/V in one packet and all 256 CUs' flash items then read them concurrently — so between them
 * they need the whole context at once, and any ring would be clobbered before it was read. */
/* THESE TWO ARE ONE CHOICE, NOT TWO. The ring must still hold every row the window can reach
 * back to from the LAST row of a chunk, so the invariant is
 *
 *     PLOW_KV_RING >= window + PLOW_MAX_CHUNK - 1
 *
 * and MAX_CHUNK is whatever that leaves, rounded down to a power of two. It is therefore a
 * joint property of the MODEL (its sliding window) and the HW (VRAM for the ring), not a
 * constant -- a model with a 4096 window on the same GPU would have to halve the chunk, or
 * double the ring.
 *
 * RING = 8192 gave MAX_CHUNK = 4096 (7169 rounded down). Doubling the ring buys an 8192 chunk
 * and costs ~6.4 GiB of sliding-layer KV at 128k -- affordable only because sizing activations
 * by the CHUNK rather than the CONTEXT gave back 44 GiB. The static assert below is the whole
 * safety argument: break the invariant and a chunk's own rows wrap and overwrite their history,
 * which is a silent wrong answer, not a crash. */
#define PLOW_KV_RING 16384u
#define PLOW_MAX_CHUNK 8192u
/* These two were raw C11 `_Static_assert`s, bypassing this header's own PLOW_SASSERT macro
 * (line 41) which already resolves to C++ `static_assert` under __cplusplus. clang/hipcc
 * tolerated the raw form as an extension; nvcc's C++ front-end does not, which forced the
 * sm_120 PoC to `#define _Static_assert static_assert` before including this header. Using
 * the macro the header already defines removes the need for that shim entirely. */
PLOW_SASSERT(PLOW_KV_RING >= 1024u + PLOW_MAX_CHUNK - 1u,
             "KV ring too small for the chunk: a chunk's rows would wrap onto their own history"
             " (1024 = Gemma-4's sliding window)");
PLOW_SASSERT((PLOW_KV_RING & (PLOW_KV_RING - 1u)) == 0u, "ring must be a power of two: row & (RING-1)");
#define PLOW_KV_MASK_NONE 0xFFFFFFFFu /* full-attention layers: the AND is a no-op */


/* Counters are strided ONE CACHE LINE apart, not packed.
 *
 * A dense uint32 counter array puts 32 counters in every 128-byte line. While 256
 * workgroups fetch_add counter i, another 256 are polling counter i+1 -- the same line --
 * so it ping-pongs between 8 XCDs on every signal AND every poll. The gate latency that
 * shows up as "stall" in the trace is largely this.
 *
 * Cost: 1134 counters x 128 B = 145 KB. Irrelevant next to 57 GiB of weights. */
#define PLOW_CTR_STRIDE 32u /* uint32 slots per counter == 128 B */
#define PLOW_CTR(base, id) ((base) + (size_t)(id) * PLOW_CTR_STRIDE)

/* u16 wire sentinel: the tensor-handle slots are uint16_t, so usable handles are
 * 0..0xFFFE (the compiler asserts the count fits at pack time). */
#define PLOW_TENSOR_NONE 0xFFFFu

/* ABI lock. crates/packet/src/dev.rs mirrors these and its test asserts the
 * same numbers, so the Rust and device views cannot drift apart silently. */
PLOW_SASSERT(sizeof(PlowWait) == 8, "PlowWait size");
PLOW_SASSERT(sizeof(PlowDevInst) == 64, "PlowDevInst size");
/* The interpreter vector-loads t[8] as one 16-byte access; with the 64-byte stride
 * this only holds if t sits at a 16-byte boundary. */
PLOW_SASSERT(__builtin_offsetof(PlowDevInst, t) == 16, "PlowDevInst.t must be 16-byte aligned");
PLOW_SASSERT(sizeof(PlowStreamEnt) == 24, "PlowStreamEnt size");
PLOW_SASSERT(sizeof(PlowTraceRec) == 40, "PlowTraceRec size");
PLOW_SASSERT(sizeof(PlowProgram) == 128,
             "PlowProgram size (9 ptr + cur_seg + pad + 3 gq ptr + xctr + peer_scratch + rank + n_gpu)");

#endif /* PLOW_DEV_ISA_H */
