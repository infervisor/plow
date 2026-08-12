/* amd_common.h — CDNA (gfx942/gfx950) device primitives shared by every op.
 *
 * bf16 conversion, wave64 reductions, and the MFMA operand types. Nothing here
 * is model-specific and nothing here touches the HIP runtime.
 */
#ifndef PLOW_AMD_COMMON_H
#define PLOW_AMD_COMMON_H

#include <hip/hip_runtime.h>

/* Storage type. We keep bf16 as a bare u16 in memory and convert explicitly, so
 * no ROCm math header is needed and the rounding is ours (round-to-nearest-even,
 * matching torch). __bf16 is used only as an MFMA operand type. */
typedef unsigned short bf16;

typedef __bf16 bf16_t;
typedef bf16_t bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x16 __attribute__((ext_vector_type(16)));

/* CDNA3/CDNA4 instruction divergence. Included here, right after the MFMA operand
 * types it needs, so every primitive below can use the arch wrappers. */
#include "amd_arch.h"

/* ---------------------------------------------------------------------------
 * GLOBAL, AND SAY SO.
 *
 * The interpreter gets its operands out of a POINTER TABLE (`tensors[handle]`), so they
 * arrive as plain `void*` with no provenance whatsoever. The compiler cannot prove such a
 * pointer is in the global aperture -- it might be LDS, it might be scratch -- so every
 * access through it compiles to `flat_*` rather than `global_*`. Measured on the decode
 * interpreter: 141 flat_load_ushort, 31 flat_load_dwordx4, and ZERO global_load_dwordx4.
 * The entire weight stream -- 57 GiB per token -- was going through the generic path.
 *
 * flat is not merely a different encoding. It costs:
 *   - a full 64-bit address per lane in VGPRs (hence the v_lshl_add_u64 storm), instead of
 *     global_load's saddr form (a scalar base + a 32-bit vector offset),
 *   - conservative lgkmcnt waits, because a flat address MIGHT be LDS -- so the weight
 *     stream serialises against LDS traffic that has nothing to do with it,
 *   - and it defeats the 16-byte vectoriser, which is how 16-byte loads decayed into
 *     `flat_load_ushort`: TWO BYTES at a time, in the hottest loop in the model.
 *
 * Address space 1 IS the global aperture on AMDGPU. Tensor pointers come from
 * hsa_memory_allocate, so they are global by construction; the compiler just has no way to
 * know it. as_glob() is where we tell it. */
#define PLOW_GLOB __attribute__((address_space(1)))
template <typename T>
__device__ __forceinline__ PLOW_GLOB T* as_glob(T* p) {
    return (PLOW_GLOB T*)(PLOW_GLOB void*)p;
}
template <typename T>
__device__ __forceinline__ const PLOW_GLOB T* as_glob(const T* p) {
    return (const PLOW_GLOB T*)(const PLOW_GLOB void*)p;
}

/* ...AND ALIGNED, AND SAY THAT TOO.
 *
 * `bf16` is `unsigned short`, so a `const bf16*` has alignment 2 -- and
 * `__builtin_memcpy(dst, src, 16)` off an align-2 pointer is compiled EXACTLY as written:
 * the compiler may not assume more than it is told, so it emits eight 2-byte loads. That is
 * the other half of the flat_load_ushort story, and it is why address space alone does not
 * fix it. A 16-byte vector type carries both facts in its own type.
 *
 * The 16-byte alignment is real, not a hope: every tensor base comes from
 * hsa_memory_allocate (4 KiB), and every K in the model is a multiple of 8 -- which the
 * 8-wide loads here already require anyway, since the tail reads a full 16 bytes at the
 * last k. */
typedef bf16 bf16v8 __attribute__((ext_vector_type(8))); /* 16 B, align 16 */

__device__ __forceinline__ bf16v8 ld_glob8(const bf16* p) {
    return *(const PLOW_GLOB bf16v8*)(const PLOW_GLOB void*)p;
}
__device__ __forceinline__ bf16v8 ld_glob8(const PLOW_GLOB bf16* p) {
    return *(const PLOW_GLOB bf16v8*)(const PLOW_GLOB void*)p;
}
__device__ __forceinline__ bf16v8 ld_lds8(const bf16* p) { return *(const bf16v8*)p; }
/* NON-TEMPORAL. The weight stream has ZERO reuse -- 57 GiB read exactly once per token --
 * yet every line of it allocates in L2 and evicts the activations that ARE reused. `nt` tells
 * the cache not to keep it. */
__device__ __forceinline__ bf16v8 ld_glob8_nt(const PLOW_GLOB bf16* p) {
    return __builtin_nontemporal_load((const PLOW_GLOB bf16v8*)(const PLOW_GLOB void*)p);
}

/* PLOW_ACT_NT: probe knob. Makes the 16-byte ACTIVATION stores non-temporal, to test whether
 * the release RMW's buffer_wbl2 costs in proportion to DIRTY L2 LINES (then evicting activation
 * writes early would make the writeback cheap) or is a fixed L2-walk latency (then it cannot).
 * Numerically inert -- `nt` is an eviction-policy hint, not a coherence bit -- so output must
 * stay byte-identical either way; only the timing is the answer.
 *
 * ANSWERED, AND IT IS A DEAD END. Gemma-4 31B decode, 64 tokens/run, 7 runs, median:
 * 17.80 ms/token against 17.10 base -- 4% SLOWER, output byte-identical. So the writeback is
 * NOT volume-proportional, which the arithmetic already said (a GEMV dirties ~43 KB, writable
 * back in ~43 ns at HBM rate, against the ~0.8 us/packet the buffer_wbl2 actually costs), and
 * evicting activations early is a straight loss because they ARE re-read -- the residual stream
 * a norm writes is the very thing the next op loads. Do not re-try this. Kept as the knob that
 * records the measurement. */
#ifndef PLOW_ACT_NT
#define PLOW_ACT_NT 0
#endif

/* PLOW_GATE_SC1: DEVICE-SCOPE ACTIVATION STORES, so the release fence can go away.
 *
 * `buffer_wbl2` exists because the producer's activation writes sit DIRTY in its XCD's L2 and a
 * release fence is the only thing that pushes them device-wide. It is cache-wide, all 256
 * workgroups issue one, and each XCD's L2 therefore performs the same writeback 32 times, which
 * serialises: measured 13.20 us for one empty b=256 packet, against 5.55 with the writeback
 * deleted (`PLOW_GATE_RELAXSIG`, an admitted data race).
 *
 * CDNA4 puts the scope on the ACCESS, not only on the fence. `SC[1:0]=2` (`sc1` in asm) is device
 * scope: ISA Reference §9.1.10.2 Table 49/50 give it `Coherent Cache Bypass` in L2 on a multi-XCD
 * agent, so an `sc1` store publishes PAST the non-coherent XCD L2 into the device-wide Infinity
 * Cache and leaves no dirty line for a writeback to find. Then `s_waitcnt vmcnt(0)` before the
 * counter bump is the whole ordering the producer needs, and the release — and its cache-wide
 * writeback — is redundant.
 *
 * MEASURED (`runtime/bench/sc1_gate.hip`, 400-step chains, 256 WG, idle GPU): baseline 10.12 us
 * per packet vs 3.84 with device-scope data and workgroup fences, per-workgroup signal 3.16 ->
 * 0.08 us. The arm that matters is the CONTROL: identical workgroup fences with PLAIN data reads
 * 104,333,312 of 104,595,456 words stale (99.75%), reproducing the 100% stale rate `interp.hip`
 * records, while the `sc1` arm reads ZERO stale words. So the bits, not the harness.
 *
 * NOT the same knob as `PLOW_ACT_NT` above, which failed. `nt` is an EVICTION hint that kept the
 * line dirty (so the writeback still had work) and threw away reuse of a residual the next op
 * re-reads — 4% slower. `sc1` is a COHERENCE bit: the line is published, not evicted early, and
 * the fence it retires is worth ~7.6 us/packet against ~0.8 for the volume it was blamed on.
 *
 * COVERAGE IS THE WHOLE CORRECTNESS ARGUMENT. Dropping the release is sound only if EVERY
 * activation store is device-scope; one plain store leaves one dirty line and the race is back.
 * `st_glob8` is the 16-byte path, `st_act1` the ragged scalar tail (K3 needs it: `b_proj` is N=12,
 * so `12 % 8 = 4` and the tail runs). Both are below. A new activation store MUST use one of them.
 *
 * >>> AND AS WRITTEN THIS IS BROKEN UNDER TP. MEASURED, K3 93 layers TP8: <<<
 *
 *     RANKS DISAGREE: rank 0 sampled 2496, rank 1 sampled 10816
 *     (all: [2496, 10816, 109376, 127296, 24256, 1856, 1216, 37056])
 *     — a collective did not happen, or one rank bound the wrong shard
 *
 * `sc1` is DEVICE scope, and the TP partials are published by this very store path: op_collective.h
 * records that "each rank's producing GEMV (o_proj/down) wrote its partial H-vector straight into
 * its own peer_scratch slot, fused into the GEMV epilogue". A peer reads that over XGMI, which
 * needs SYSTEM scope. `sc1` bypasses L2 into the local Infinity Cache, so `xctr_signal`'s
 * system-scope release — which writes back L2 — has no dirty line left to push across XGMI. The
 * partial never leaves the GPU, XReduce sums stale slots, and every rank samples a different token.
 *
 * Note what this means about the evidence: the single-GPU microbench (`runtime/bench/sc1_gate.hip`)
 * measured ZERO stale words of 104,595,456 with a same-harness control at 99.75% stale, and it
 * still did not predict this — the property that breaks is cross-DEVICE visibility, which no
 * single-GPU harness can observe.
 *
 * COVERAGE IS BIGGER THAN THE HELPERS, and it caused three failures in a row because it was
 * reasoned about instead of counted. It is now COUNTED, and the count is a build gate:
 * `scripts/sc1_coverage.sh` disassembles the object and fails if any `global_store_*` does not
 * carry `sc0`/`sc1`. That is the whole correctness argument, mechanised — no store can be missed
 * silently, and a new op that forgets a helper breaks the build rather than the model.
 *
 *   (count it by hand with: llvm-objdump -d --mcpu=gfx950 <obj> | grep -oE "global_store_[a-z0-9]+[^/]*"
 *    then split on whether the line carries `sc`.)
 *
 * The audit that gate forced, on the K3 decode object: 243 plain stores against 54 covered, from
 * only 23 DISTINCT SOURCE LINES — 144 of them one GEMV epilogue line inlined repeatedly. The
 * biggest missed publisher was the one that matters most: the GEMV epilogue
 * (`C[(size_t)m * N + n] = f2bf(t)`), which is exactly what publishes the TP partials into
 * `act.og_tp`/`act.dg_tp`. Also converted: the f32 flash partials and (m,l) statistics, the MoE
 * GLU row, the argmax part/ids buffers, the DSA score and selection arrays, and every collective
 * output. The ONE store deliberately left plain is the trace record (`interp.hip`), which is
 * host-read after the kernel retires and crosses no gate; the gate script excludes it by name.
 *
 * >>> AND WITH COVERAGE COMPLETE, THE KNOB AS DESIGNED IS STILL WRONG. MEASURED. <<<
 *
 * K3 93 layers TP8, real weights, `--prompt 1008,10484,318,15383,387`, against a control that
 * produces the known-good continuation:
 *
 *     control                     [13, 646, 12259, 387, 14868, 220, 5807, 6017, ...]   coherent
 *     PLOW_GATE_SC1               [13, 646, 1272, 220, 17, 17, 646, 13, 18, 220, ...]  DEGENERATE
 *     + PLOW_ACT_SCOPE_AGENT      [418, 11, 276, 276, 276, 1356, 1356, 1356, ...]      DEGENERATE
 *     + PLOW_GATE_SC1_KEEPREL     [13, 646, 12259, 387, 14868, 220, 5807, 6017, ...]   IDENTICAL
 *
 * Both scopes fail and the one that keeps the RELEASE passes, so the defect is NOT the store
 * scope and NOT coverage — it is the ORDERING. `s_waitcnt vmcnt(0)` retires a store when it
 * leaves the CU, not when it is visible at the scope the consumer reads from, so a relaxed
 * counter bump can be observed before the data it is supposed to publish. The release RMW is
 * doing work that `vmcnt(0)` does not replace. Dropping it needs a publish primitive that
 * actually waits for visibility, and this file does not have one.
 *
 * What that leaves is the arm nobody had tried: scoped stores WITH the release kept
 * (`PLOW_GATE_SC1_KEEPREL`). It is token-identical, and the reason it can still be a win is
 * that an `sc0 sc1` store leaves no dirty line, so the `buffer_wbl2` the release still issues
 * has almost nothing to write back. That is a cheaper fence, not a deleted one. Its timing is
 * NOT recorded here: every A/B this campaign ran was contended by a concurrent agent holding
 * all 8 GPUs, and a bare `flock /tmp/plow_gpu.lock` neither waits for nor warns about that —
 * use `perf-data/tools/gpulease`, which does both. The one arm measured both ways
 * contradicted itself by 12.6 ms (46.418 at position 2 vs 33.802 at position 1).
 *
 * SCOPE, AND WHY ONE AND NOT TWO. Activation stores are really two classes —
 *   * LOCAL activations (residual stream, next op's input)  -> `sc1`      device scope
 *   * PEER-VISIBLE activations (the GEMV epilogue's write into peer_scratch — `act.og_tp`,
 *     `act.dg_tp`)                                          -> `sc0 sc1` system scope
 * and the second is what the TP failure above was: `sc1` alone bypasses L2 into the LOCAL
 * Infinity Cache, so the partial never crosses XGMI. A store site cannot tell which class its
 * destination is (it gets a bare `void*` out of the tensor table), so this code uses SYSTEM for
 * every activation store: correct for both, and the local ones pay for reach they do not need.
 * Splitting them is a measurable optimisation, not a correctness requirement, and it needs the
 * emitter to mark peer-visible outputs in the instruction. Correct first.
 */
#ifndef PLOW_GATE_SC1
#define PLOW_GATE_SC1 0
#endif

/* The scope every activation store carries under PLOW_GATE_SC1. See the note above for why this
 * is SYSTEM and not AGENT.
 *
 * PLOW_ACT_SCOPE_AGENT is a CEILING INSTRUMENT and MUST NOT SHIP. It narrows every activation
 * store to AGENT (`sc1`, device scope), which is what a LOCAL activation actually needs and is
 * cheaper than SYSTEM (`sc0 sc1`) because it publishes into the device-wide Infinity Cache
 * instead of write-through past it. It is WRONG UNDER TP by exactly the mechanism the note
 * above records: the GEMV epilogue's write into `peer_scratch` never crosses XGMI, so XReduce
 * sums stale slots. Its only job is to PRICE the two-class split (local `sc1` / peer `sc0 sc1`)
 * before anyone builds the emitter plumbing that split needs — if the agent-scope arm is not
 * meaningfully faster than the system-scope one, the split is not worth building. */
#ifndef PLOW_ACT_SCOPE_AGENT
#define PLOW_ACT_SCOPE_AGENT 0
#endif
#if PLOW_ACT_SCOPE_AGENT
#define PLOW_ACT_SCOPE __HIP_MEMORY_SCOPE_AGENT /* UNSAFE UNDER TP — measurement only */
#else
#define PLOW_ACT_SCOPE __HIP_MEMORY_SCOPE_SYSTEM
#endif

/* ONE SCALAR ACTIVATION ELEMENT, scoped under PLOW_GATE_SC1 and a plain store otherwise.
 *
 * `__hip_atomic_store` and NOT inline asm. A relaxed atomic store at system scope lowers to
 * exactly `global_store_<width> v, v, s[..] sc0 sc1` — the SADDR form (scalar base + 32-bit
 * vector offset), with compiler-tracked waitcnts and no `"memory"` clobber. The inline-asm form
 * this replaces forced a full 64-bit address into a VGPR pair at every site and fenced the
 * scheduler around it, which the GEMV epilogue pays 144 times.
 *
 * Verified lowering on gfx950 (ROCm 7.2.4), one probe kernel per width:
 *     bf16/u16 -> global_store_short     ... sc0 sc1
 *     f32/u32  -> global_store_dword     ... sc0 sc1
 *     u64      -> global_store_dwordx2   ... sc0 sc1
 *     u8       -> global_store_byte      ... sc0 sc1
 * (AGENT scope gives the same instructions with `sc1` alone, if the split above is ever built.) */
template <typename T>
__device__ __forceinline__ void st_act(T* p, T v) {
#if PLOW_GATE_SC1
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, PLOW_ACT_SCOPE);
#else
    *p = v;
#endif
}
template <typename T>
__device__ __forceinline__ void st_act(PLOW_GLOB T* p, T v) {
#if PLOW_GATE_SC1
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, PLOW_ACT_SCOPE);
#else
    *p = v;
#endif
}

/* SYSTEM scope (`sc0 sc1`, SC[1:0]=3), not device (`sc1`, SC=2), and the TP failure above is
 * why. A single scope for every activation store is the conservative choice: it is correct for
 * the peer-visible ones, and the local ones pay for reach they do not need. Splitting them —
 * local `sc1`, peer `sc0 sc1` — is the optimisation, and it needs the store site to know which
 * tensor class it is writing, which this code cannot yet tell. Correct first. */
__device__ __forceinline__ void st_glob8(bf16* p, bf16v8 v) {
#if PLOW_GATE_SC1
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(v) : "memory");
#elif PLOW_ACT_NT
    __builtin_nontemporal_store(v, (PLOW_GLOB bf16v8*)(PLOW_GLOB void*)p);
#else
    *(PLOW_GLOB bf16v8*)(PLOW_GLOB void*)p = v;
#endif
}
__device__ __forceinline__ void st_glob8(PLOW_GLOB bf16* p, bf16v8 v) {
#if PLOW_GATE_SC1
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(v) : "memory");
#elif PLOW_ACT_NT
    __builtin_nontemporal_store(v, (PLOW_GLOB bf16v8*)(PLOW_GLOB void*)p);
#else
    *(PLOW_GLOB bf16v8*)(PLOW_GLOB void*)p = v;
#endif
}

/* One activation half. This is the ragged tail every op writes when its width is not a multiple
 * of 8; under the default it is exactly `*p = v` and the emitted code is unchanged. */
__device__ __forceinline__ void st_act1(bf16* p, bf16 v) { st_act<bf16>(p, v); }
/* The address-space-1 twin, mirroring `st_glob8`'s pair. `d_headnorm_rope`
 * stores through an `as_glob` pointer, so any translation unit that instantiates
 * it needs this overload — which no unit did until a .hip test included
 * `op_norm.h` directly (the interpreter is built `--genco`, device-only, and the
 * generic overload satisfied it there). */
__device__ __forceinline__ void st_act1(PLOW_GLOB bf16* p, bf16 v) {
    *p = v;
}

/* The fp8 tail. `d_headnorm_rope_fp8` writes the KV cache one BYTE at a time when the head
 * width is ragged; that store is an activation publish exactly like the bf16 ones, and leaving
 * it plain is enough on its own to reinstate the race PLOW_GATE_SC1 exists to remove. */
__device__ __forceinline__ void st_act1_u8(PLOW_GLOB unsigned char* p, unsigned char v) {
    *p = v;
}
__device__ __forceinline__ void st_act1_u8(unsigned char* p, unsigned char v) {
    st_act<unsigned char>(p, v);
}
__device__ __forceinline__ bf16v8 bf16v8_zero(void) {
    bf16v8 z;
#pragma unroll
    for (int j = 0; j < 8; j++) z[j] = 0;
    return z;
}

/* ---------------------------------------------------------------------------
 * ASYNC HBM -> LDS. This is CDNA's cp.async, and it is what lets a gate stop the CU without
 * stopping the memory system.
 *
 * A packet's operands split in two: ACTIVATIONS, which a predecessor must produce, and
 * CONSTANTS -- weights, RoPE tables, all but the newest row of the KV cache -- which have no
 * producer at all and have been sitting unchanged in HBM since the model loaded. There is
 * nothing to wait for. So the interpreter issues the constant loads BEFORE the gate and lets
 * them fly while the CU spins: measured, every GEMV gate has ~11 us of idle behind it and the
 * memory system does nothing for all of it.
 *
 * global_load_lds writes straight into LDS with NO VGPRs, which is the only reason this is
 * affordable -- decode is at 223 of 256 registers, so a register-held prefetch is not on the
 * table. It is tracked on vmcnt, so the loads survive s_barrier and we wait for them once, on
 * the far side of the gate.
 *
 * CONTRACT, and it is sharp: the LDS destination comes from M0, which is UNIFORM. The
 * compiler emits `v_readfirstlane` on whatever LDS pointer you pass and the hardware then
 * lays the 64 lanes down CONTIGUOUSLY from there (lane l -> M0 + l*16). The global source may
 * be per-lane; the LDS destination may NOT. Hand it a non-contiguous LDS mapping and it will
 * quietly write somewhere else. */
/* The __HIP_DEVICE_COMPILE__ guard is not cosmetic: hipcc parses every __device__ body in the
 * HOST pass too, and this builtin does not exist there -- it fails with "invalid size value",
 * which reads like a bad argument and is nothing of the sort. */
__device__ __forceinline__ void cp_async16(const PLOW_GLOB bf16* src,
                                           bf16* dst_lane_contiguous) {
#ifdef __HIP_DEVICE_COMPILE__
#if !PLOW_CDNA4
    /* CDNA3's global_load_lds takes 1/2/4 bytes per lane; the 12/16-byte forms are a CDNA4
     * addition (clang rejects 16 with "invalid size value"). Four 4-byte issues do NOT
     * reconstruct this call: the LDS destination comes from M0 and the hardware lays lanes
     * down contiguously at the ISSUE width, so 4-byte issues give a lane*4 layout, not the
     * lane*16 one every fragment read below assumes.
     *
     * So the CDNA3 arm is a VGPR-staged copy instead. It costs registers where the CDNA4 path
     * costs none -- which is exactly the tradeoff the header comment above says decode cannot
     * afford at 8 waves. It is affordable in the 4-wave objects; the 8-wave decode object must
     * be re-measured before this path is trusted there. */
    const unsigned lane = __builtin_amdgcn_workitem_id_x() & 63u; /* wave64, PLOW_WAVE below */
    *(bf16v8*)(dst_lane_contiguous + lane * 8) = *(const PLOW_GLOB bf16v8*)src;
#elif defined(__clang_major__) && __clang_major__ >= 23
/* clang-23 (ROCm 7.14) retyped the builtin's global operand to a NON-const addrspace(1)
 * void*. The pre-23 spelling stops compiling against it — on EVERY arch, gfx950 included —
 * with "cannot initialize a parameter of type '__device__ void *'". Kept version-gated so a
 * ROCm 7.2.4 / clang-22 build still takes the exact call it takes today. */
    __builtin_amdgcn_global_load_lds(
        (PLOW_GLOB void*)(const PLOW_GLOB void*)src,
        (__attribute__((address_space(3))) void*)dst_lane_contiguous,
        16 /* bytes per lane */, 0 /* offset */, 0 /* aux */);
#else
    __builtin_amdgcn_global_load_lds(
        (const PLOW_GLOB unsigned*)(const PLOW_GLOB void*)src,
        (__attribute__((address_space(3))) unsigned*)(__attribute__((address_space(3))) void*)
            dst_lane_contiguous,
        16 /* bytes per lane */, 0 /* offset */, 0 /* aux */);
#endif
#else
    (void)src;
    (void)dst_lane_contiguous;
#endif
}
__device__ __forceinline__ void cp_async_wait(void) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

/* ---------------------------------------------------------------------------
 * BUFFER LOADS: a hardware bounds check that is FREE, and the only clean way to unroll a loop
 * whose trip count is ragged.
 *
 * A buffer load addresses memory through a 128-bit resource descriptor carrying `num_records`.
 * A load past the end returns ZERO and — this is the part that matters — **issues no memory
 * request at all**. Overshoot costs an instruction slot and NOTHING in bandwidth.
 *
 * That is what makes it different from predication, which this file already tried and reverted:
 * clamping an out-of-range offset to 0 still FETCHES (16.9 -> 20.6 ms/token, gemv 52 -> 70
 * us/inst). The hardware guard fetches nothing.
 *
 * The payoff is the GEMV's ragged tail. The k-loop advances by GV_UNROLL*512 halves, and whatever
 * K leaves over used to run in a SCALAR loop — one load, wait, consume, repeat. At Gemma's
 * K = 5376 that was 24% of every row with no memory-level parallelism, and it also BROKE the
 * compiler's software pipeline: it pipelines ~20 loads deep across the main loop and then hits a
 * dependent scalar loop. With a buffer load there is no tail at all — the row is one uniform
 * unrolled loop and the overshoot is free.
 *
 * Measured standalone on the real projections (scalar tail -> buffer_load, no tail):
 *     gate/up 21504x5376   3510 -> 4090 GB/s   (+16.5%)
 *     q_proj   8192x5376   3248 -> 3564        ( +9.7%)
 *     o_proj   5376x8192   5342 -> 5448        ( +2.0%)   <- K already divided the unroll
 *     down    5376x21504   6012 -> 6119        ( +1.8%)   <- ditto
 * The gain tracks the tail fraction exactly.
 *
 * THE DESCRIPTOR'S WORD 3 IS NOT OPTIONAL. On gfx9/CDNA it must be 0x00020000 (this is
 * CK's CK_BUFFER_RESOURCE_3RD_DWORD). Passing 0 compiles, runs, faults nothing — and returns
 * ZERO FOR EVERY LANE, in range or not. It fails as silently as anything in this codebase. */
#define PLOW_BUF_RSRC3 0x00020000u

/* `aux` = 3 is glc|slc: slc is the streaming / non-temporal hint. The weight stream has zero
 * reuse, so it must not allocate in L2 — the same reason GV_LDW uses ld_glob8_nt. */
__device__ __forceinline__ bf16v8 buf_ld8(__amdgpu_buffer_rsrc_t r, unsigned byte_off) {
    return __builtin_bit_cast(bf16v8,
                              __builtin_amdgcn_raw_buffer_load_b128(r, byte_off, 0, /*nt*/ 3));
}
__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc(const bf16* base, unsigned n_halves) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, /*stride*/ (short)0,
                                             /*num_records, BYTES*/ n_halves * 2u, PLOW_BUF_RSRC3);
}

/* ---------------------------------------------------------------------------
 * FP8 (OCP e4m3) WEIGHT LOADS — the decode w8a16 path.
 *
 * The fp8 weight row is uint8[K], so num_records is K BYTES (not K*2). The b128 buffer load then
 * pulls 16 fp8 per lane instead of 8 bf16 — HALF the bytes per output element, which is the whole
 * point: decode is weight-bandwidth-bound, so fp8 weights ~= 2x the roofline.
 *
 * gfx950 (CDNA4) has NATIVE OCP e4m3. cvt_pk_f32_fp8(word, sel) converts a packed pair of fp8 to
 * two f32 — sel=false picks bytes [0,1] of the u32, sel=true bytes [2,3] (verified: 0x38 -> 1.0,
 * 0x40 -> 2.0, 0x30 -> 0.5, 0xB8 -> -1.0, matching torch.float8_e4m3fn). The decoded fp8 is exact
 * in bf16 (fp8 e4m3 has 3 mantissa bits, bf16 has 7), so f2bf() then feeds the SAME fdot2 `dot8`
 * and the same wave reduction as the bf16 GEMV; the per-channel dequant scale is applied ONCE in
 * the epilogue. (cvt_scalef32_pk_bf16_fp8 was tried and its word_sel does NOT pick adjacent byte
 * pairs — it broadcasts one byte — so it is unusable here.) */
typedef unsigned fp8v16 __attribute__((ext_vector_type(4))); /* 16 fp8 == 4 u32, 16 B, align 16 */
typedef int fp8v32 __attribute__((ext_vector_type(8)));      /* 32 fp8 == 8 i32, 32 B — the K64 MX-fp8 MFMA operand fragment */

__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc_fp8(const unsigned char* base,
                                                               unsigned n_bytes) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, /*stride*/ (short)0,
                                             /*num_records, BYTES*/ n_bytes, PLOW_BUF_RSRC3);
}
/* PROBE: force a wave-uniform base pointer into SGPRs so make_buffer_rsrc emits a SCALAR resource
 * (no per-load readfirstlane waterfall). Valid ONLY when `base` is genuinely uniform across the wave
 * (it is in the GEMV: base = W + n*K, n = wave index, uniform per wave). */
__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc_fp8_u(const unsigned char* base,
                                                                 unsigned n_bytes) {
    unsigned long long u = (unsigned long long)(size_t)base;
    unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)u);
    unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(u >> 32));
    const unsigned char* ub = (const unsigned char*)(size_t)(((unsigned long long)hi << 32) | lo);
    return __builtin_amdgcn_make_buffer_rsrc((void*)ub, (short)0, n_bytes, PLOW_BUF_RSRC3);
}
/* The bf16 twin of buf_rsrc_fp8_u, and the reason it is worth having is visible in the shipped
 * decode object: EVERY buffer_load_dwordx4 in d_gemv_qkv / d_gemv_glu / d_gemv_fp8_blk carries
 *
 *     s_mov_b64 exec save / 4x v_readfirstlane_b32 / 2x v_cmp_eq_u64 / s_and_saveexec_b64
 *     buffer_load_dwordx4 / s_xor_b64 exec / s_cbranch_execnz <BACKWARDS> / s_mov_b64 exec
 *
 * That is not merely ~13 extra instructions -- the backward branch puts each load in its OWN
 * BASIC BLOCK, so the UN loads the source issues "before touching any of them" cannot be a
 * clause at all. Whatever memory-level parallelism the unroll was written to create, the
 * waterfall serialises.
 *
 * In a MEGAKERNEL the compiler can never prove the base uniform on its own: the op's operands
 * are read out of the packet into VGPRs, so `W + n*K` is VGPR-resident by construction, however
 * wave-invariant its VALUE is. readfirstlane is how you tell it. Valid here because `n` is the
 * wave's own row index -- identical in all 64 lanes. */
__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc_u(const bf16* base, unsigned n_halves) {
    unsigned long long u = (unsigned long long)(size_t)base;
    unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)u);
    unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(u >> 32));
    const bf16* ub = (const bf16*)(size_t)(((unsigned long long)hi << 32) | lo);
    return __builtin_amdgcn_make_buffer_rsrc((void*)ub, (short)0, n_halves * 2u, PLOW_BUF_RSRC3);
}
/* `aux` = 3 (glc|slc): streaming/non-temporal, exactly as the bf16 weight stream — zero reuse. */
__device__ __forceinline__ fp8v16 buf_ld_fp8(__amdgpu_buffer_rsrc_t r, unsigned byte_off) {
    return __builtin_bit_cast(fp8v16,
                              __builtin_amdgcn_raw_buffer_load_b128(r, byte_off, 0, /*nt*/ 3));
}

/* Wait until at most D vector-memory ops are still outstanding.
 *
 * This is the THROTTLE, and it is the whole difference between a prefetch that helps and one
 * that hurts. The window a mover runs in is genuinely idle -- the two norms in front of a
 * q_proj move 40 KB in 6 us, i.e. 7 GB/s of a 6200 GB/s machine -- but those norms are
 * LATENCY-bound: one CU, one row, one HBM round trip. Issuing the whole prefetch as a burst
 * puts 16 MB (256 CUs x 64 KB) into the memory controller ahead of their 10 KB and their round
 * trip goes from ~1.3 us to ~2 (measured: rmsnorm +18%, norm_residual +41%).
 *
 * Capping outstanding requests caps the queue the critical op sits behind. A CU sustains
 * roughly D KB / 1.3 us, so D is a direct rate knob: D=14 is ~10.7 GB/s per CU, enough to move
 * 64 KB across a 6 us window. */
template <int D>
__device__ __forceinline__ void cp_async_depth(void) {
    asm volatile("s_waitcnt vmcnt(%0)" ::"i"(D) : "memory");
}

/* An 8-wide bf16 dot product in FOUR instructions.
 *
 * gfx950 has v_dot2c_f32_bf16: two bf16 lanes multiplied and accumulated into an f32, as one
 * VALU op, with NO conversion. The obvious C -- `acc += bf2f(w[j]) * bf2f(x[j])` -- instead
 * costs a shift per operand plus an FMA, i.e. 24 VALU ops per 16 bytes rather than 4.
 *
 * This matters because the GEMV moves 57 GiB per token and every byte passes through here. */
typedef bf16_t bf16x2 __attribute__((ext_vector_type(2)));
union bf16v8_pairs {
    bf16v8 v;
    bf16x2 p[4];
};

__device__ __forceinline__ float dot8(bf16v8 a, bf16v8 b, float acc) {
    bf16v8_pairs x{a}, y{b};
#pragma unroll
    for (int j = 0; j < 4; j++) acc = plow_dot2_bf16(x.p[j], y.p[j], acc);
    return acc;
}

__device__ __forceinline__ float bf2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

/* BRANCHED, and the branchless form is REFUTED ON HARDWARE. `PLOW_F2BF_SELECT=1` restores it.
 *
 * The NaN guard is an `if` with an early return, which on gfx942 lowers to an exec-mask
 * save/restore around the rounding rather than to a select. Replacing it with a select looked
 * unambiguously right on every static measure -- over a 64-conversion tile (`probes/f2bf.hip`)
 * 1881 -> 838 instructions, 1098 -> 643 VALU, 151 -> 19 exec-mask ops, and on the real prefill
 * megakernel -5.0% instructions, -13.3% SALU, -30% exec-mask, object 243104 -> 234680 B.
 *
 * SERVED, IT IS SLOWER, and not marginally (4 interleaved arms, 2 rounds each, GLM-5.2 TP8):
 *
 *   ctx     branchless   branched      delta
 *   1024        325.6      323.1      +0.77%
 *   4096        751.4      718.9      +4.51%
 *   8192       1708.6     1622.2      +5.33%
 *  16384       3783.6     3535.8      +7.01%
 *   TPOT       26.485     26.271      +0.81%
 *
 * Ordering is ruled out as the cause: the branchless arm reproduces to within 0.1% whether it
 * runs first or third (751.6/751.1 at 4k), so sequence position has no effect on this box.
 *
 * TWO THINGS THIS COSTS, AND BOTH ARE WORTH KEEPING WRITTEN DOWN:
 *
 * 1. Static instruction count is not time. -5% instructions bought +5% wall clock. The select
 *    form computes BOTH the RNE and the quiet-NaN on every conversion where the branch computed
 *    one, and f2bf sits on every store path in this file's list -- so the deleted exec-mask
 *    overhead was cheaper than the arithmetic that replaced it. An exec-mask pair around a
 *    rarely-taken branch is not obviously a cost on this hardware.
 * 2. "Value-identical in isolation" does NOT imply "output-identical in situ". The branchless
 *    form is value-identical over ALL 2^32 float bit patterns (`probes/f2bf_gate.c`, 0
 *    mismatches, with a deliberately-wrong control the same gate catches 1.8 M times) -- and the
 *    served arms still disagree on GSM8K, reproducibly, 0.960 vs 0.970 in BOTH rounds. Changing
 *    a function this widely inlined perturbs surrounding codegen (scheduling, and at -O3 which
 *    fp contractions form), so downstream f32 arithmetic is not guaranteed to round the same
 *    way. An earlier version of this note claimed "logits byte-identical by construction"; that
 *    was wrong, and the accuracy column is what caught it.
 */
#ifndef PLOW_F2BF_SELECT /* =1 restores the REFUTED branchless form, for a reproducible A/B */
#define PLOW_F2BF_SELECT 0
#endif
__device__ __forceinline__ bf16 f2bf(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
#if PLOW_F2BF_SELECT
    const unsigned rne = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    const unsigned qnan = (u >> 16) | 0x0040u;
    return (bf16)(((u & 0x7fffffffu) > 0x7f800000u) ? qnan : rne);
#else
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u); /* qNaN */
    u += 0x7fffu + ((u >> 16) & 1u);                                          /* RNE */
    return (bf16)(u >> 16);
#endif
}

/* 16 fp8 (a b128 load) -> two bf16v8, in fdot2-ready lane order. Each u32 word holds 4 fp8.
 *
 * ONE INSTRUCTION per fp8 PAIR: gfx950's v_cvt_scalef32_pk_bf16_fp8 decodes a packed fp8 pair
 * straight to a bf16 pair (with an f32 scale; passed 1.0 — the per-channel w_scale stays a single
 * epilogue multiply). sel=false picks bytes [0,1] of the u32, sel=true bytes [2,3] — verified on
 * gfx950 (word 0xB8304038 -> 1.0, 2.0, 0.5, -1.0), the SAME byte order as cvt_pk_f32_fp8. This
 * replaces the old two-step decode (8x cvt_pk_f32_fp8 to f32 + 16 high-half truncations = ~24 VALU
 * per fp8v16) with 8 packed converts — ~3x fewer dequant VALU. NOTE: this is a cleaner/cheaper
 * kernel but PERF-NEUTRAL at bs=1 decode: measured the fp8 GEMV op-work is unchanged (gemv_fp8
 * 253 vs 251 us, gemv_glu_fp8 152 vs 152), because the dequant already hides fully under the fp8
 * weight-load latency — the decode GEMV op is memory-latency-bound (per-wave outstanding-load
 * limited), not dequant-bound. Kept because it is the correct native instruction and frees VALU
 * headroom. Bit-exact to the old truncation: fp8 e4m3 -> bf16 is lossless (<=3 mantissa bits) and
 * the instruction is the native round. Words 0,1 fill `lo` (elems 0..7), words 2,3 fill `hi` —
 * matching x[kx..kx+8] and x[kx+8..kx+16] in the fp8 GEMV.
 *
 * (The earlier note that this builtin "broadcasts one byte" was wrong — see the byte-order check
 * above; it is a drop-in for the two-step path.) */
#if PLOW_HAS_MX_CVT /* CDNA4 MX block-scale convert; CDNA3 has no fp4 type and no scalef32 */
__device__ __forceinline__ void fp8_to_bf16v8(fp8v16 w, bf16v8& lo, bf16v8& hi) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    union { bf16v8 v; unsigned u[4]; } ol, oh; /* pack pairs as u32 to avoid per-lane extraction */
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const bf16_2 a = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[i], 1.0f, false); /* bytes 0,1 */
        const bf16_2 c = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[i], 1.0f, true);  /* bytes 2,3 */
        auto& o = (i < 2) ? ol : oh;
        const int b = (i & 1) * 2; /* two u32 (= 4 bf16) per word */
        o.u[b + 0] = __builtin_bit_cast(unsigned, a); /* a0 in low16, a1 in high16 */
        o.u[b + 1] = __builtin_bit_cast(unsigned, c);
    }
    lo = ol.v;
    hi = oh.v;
}
#else
/* CDNA3 arm: the two-step path this instruction replaced. v_cvt_pk_f32_fp8 exists on gfx942, so
 * the decode is 8 packed converts to f32 plus a high-half pack -- more VALU than the CDNA4
 * single-step, and BIT-IDENTICAL: fp8 e4m3 carries <=3 mantissa bits, so f32 holds it exactly and
 * the bf16 narrowing is the same round. scale is 1.0 here for the same reason it is on CDNA4 --
 * the block scale is folded in the epilogue, see the note below. */
__device__ __forceinline__ void fp8_to_bf16v8(fp8v16 w, bf16v8& lo, bf16v8& hi) {
    union { bf16v8 v; unsigned u[4]; } ol, oh;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        auto& o = (i < 2) ? ol : oh;
        const int b = (i & 1) * 2;
        plow_fp8x4_ocp_to_bf16((unsigned)w[i], o.u[b + 0], o.u[b + 1]);
    }
    lo = ol.v;
    hi = oh.v;
}
#endif /* PLOW_HAS_MX_CVT */

/* WHY scale=1.0 above and NOT the block scale: v_cvt_scalef32_pk_bf16_fp8's scalef32 operand is an
 * E8M0 (power-of-2) MICROSCALING scale — the hardware uses ONLY the f32 exponent (2^floor(log2 s))
 * and DISCARDS the mantissa. Probed on gfx950 (2026-07-17): powers of two (0.5, 2, 2^-6) fold
 * exactly, but 0.01 folds as 2^-7 = 0.0078 (~22% low). That is correct for MX/mxfp8 (E8M0 block
 * scales, which is why AITER always feeds it e8m0_to_f32(byte)=2^(byte-127)), but DeepSeek/GLM
 * block-fp8 (weight_block_size [128,128]) carries ARBITRARY f32 weight_scale_inv. So the block scale
 * must stay a SEPARATE f32 multiply after the exact fp8->bf16 decode (gemv_rows_fp8_blk /
 * wave_dot_fp8_blk apply it per 128-K block); it cannot be folded into this cvt. */

/* ---------------------------------------------------------------------------
 * MXFP4 (OCP microscaling: e2m1 element + one shared E8M0 scale per 32) — weight decode.
 *
 * This is the case the fp8 block-scale comment above rules OUT for GLM/DeepSeek and IN for MX: an
 * MX scale IS a power of two by construction (E8M0 is a bare 8-bit exponent), so folding it into
 * the cvt's scalef32 operand is EXACT, not an approximation. The mantissa the hardware discards is
 * a mantissa the format does not have. So mxfp4 dequant costs the same VALU as fp8 dequant and the
 * scale is free — no separate multiply, no per-block FMA, nothing in the epilogue.
 *
 * THE ALIGNMENT THAT MAKES THIS CHEAP: a lane's b128 weight load is 16 bytes = 32 fp4 = EXACTLY one
 * MX block. So one load consumes one scale byte, the scale never varies within a lane's fragment,
 * and there is no cross-lane scale reshuffle (contrast gemv_rows_fp8_blk, which has to reason about
 * 16-element partials landing inside a 128-K block). Pick any other fragment width and this
 * property is lost. */
typedef unsigned fp4v32 __attribute__((ext_vector_type(4))); /* 32 fp4 == 4 u32, 16 B, align 16 */

/* E8M0 byte -> f32 2^(b-127). Bit-construct rather than exp2f: byte b placed in the f32 exponent
 * field with a zero mantissa IS 2^(b-127), so this is one shift, no transcendental. b=0xFF is the
 * MX NaN encoding; it is not special-cased here because the cvt consumes only the exponent and a
 * NaN scale would poison the block anyway (quantisers do not emit it for weights). */
__device__ __forceinline__ float e8m0_to_f32(unsigned char b) {
    return __builtin_bit_cast(float, (unsigned)b << 23);
}

/* 1 / e8m0_to_f32(b) = 2^(127-b), which the MXFP4 quantizers want far more often than the scale
 * itself. The reciprocal of a power of two is EXACT — but spelled `1.0f / e8m0_to_f32(b)` the
 * backend has to emit the full IEEE division expansion (v_div_scale, v_rcp, three FMA refinement
 * steps, v_div_fmas, v_div_fixup: thirteen VALU) for a value that is one exponent negation. The
 * A4W4 MoE prefill paid that TWICE per MX block, once in the staging quantizer and once in the
 * fused bridge.
 *
 * `v_rcp_f32` and not an exponent-field construction, because the ENDS matter and a bit twiddle
 * gets them wrong: e8m0_to_f32(0) is +0.0 (not 2^-127 — the byte lands in the exponent field, so
 * byte 0 has a zero exponent AND a zero mantissa), so the reciprocal there is +inf, and byte 255
 * is +inf so its reciprocal is +0.0. v_rcp_f32 reproduces both, and it is EXACT on the interior
 * because every input here is a power of two with a 1.0 mantissa. Bit-identical for all 256
 * bytes; runtime/tests/fp4_quant_identity_gfx950_test.hip checks every one of them. */
__device__ __forceinline__ float e8m0_inv_f32(unsigned char b) {
    /* b = 254 is the one byte whose reciprocal (2^-127) is SUBNORMAL, and the transcendental unit
     * flushes it while the division does not. Two instructions to keep the helper total over the
     * whole byte domain rather than correct-where-we-happen-to-call-it: e8m0_for_amax cannot
     * return 254 (it would need amax >= 6*2^127, past FLT_MAX), but nothing about this function
     * says so, and the next caller will not know. */
    const float r = __builtin_amdgcn_rcpf(e8m0_to_f32(b));
    return b == 254u ? __builtin_bit_cast(float, 0x00400000u) : r;
}

/* 32 fp4 + one E8M0 scale -> four bf16v8, in the SAME fdot2-ready lane order as fp8_to_bf16v8, so
 * the GEMV reduction, `dot8` and the wave reduction are shared verbatim with the fp8 path.
 * Each u32 holds 8 fp4 = 4 pairs; op_sel (the third operand, 0..3) picks the pair, so one word
 * becomes 8 bf16 in 4 packed converts — 16 converts per 32-element fragment. */
#if PLOW_HAS_MX_CVT /* CDNA4 MX block-scale convert; CDNA3 has no fp4 type and no scalef32 */
__device__ __forceinline__ void fp4_to_bf16v8x4(fp4v32 w, float scale, bf16v8& a, bf16v8& b,
                                                bf16v8& c, bf16v8& d) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    union { bf16v8 v; unsigned u[4]; } o[4];
    /* op_sel must be a literal (clang rejects a loop induction variable even under #pragma unroll,
     * the check runs before unrolling), so the pair select is spelled out. */
#define MXFP4_CVT(i)                                                                          \
    o[i].u[0] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 0));             \
    o[i].u[1] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 1));             \
    o[i].u[2] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 2));             \
    o[i].u[3] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 3));
    MXFP4_CVT(0)
    MXFP4_CVT(1)
    MXFP4_CVT(2)
    MXFP4_CVT(3)
#undef MXFP4_CVT
    a = o[0].v;
    b = o[1].v;
    c = o[2].v;
    d = o[3].v;
}
#else
/* CDNA3 arm: no fp4 datatype at all, so the nibble decode is the fp16-subnormal bit-construction
 * in amd_arch.h (plow_fp4x8_to_bf16x8) and the E8M0 scale is a plain multiply. Same op_sel ORDER
 * as the CDNA4 instruction -- word j of the output holds nibbles 2j and 2j+1 of the input, low
 * nibble first -- so the fdot2-ready lane order the GEMV reduction depends on is preserved, and
 * the VALUES are bit-identical to gfx950's native cvt. */
__device__ __forceinline__ void fp4_to_bf16v8x4(fp4v32 w, float scale, bf16v8& a, bf16v8& b,
                                                bf16v8& c, bf16v8& d) {
    union { bf16v8 v; unsigned u[4]; } o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) plow_fp4x8_to_bf16x8((unsigned)w[i], scale, o[i].u);
    a = o[0].v;
    b = o[1].v;
    c = o[2].v;
    d = o[3].v;
}
#endif /* PLOW_HAS_MX_CVT */

/* ---------------------------------------------------------------------------
 * 32 fp4 (EXACTLY one MX block) dotted against 32 bf16 activations.
 *
 * Every mxfp4 GEMV in this tree — the decode GEMV, its fused q|k|v and gate|up twins, and the
 * three MoE expert walks — spells the same four lines: one `fp4_to_bf16v8x4`, then four `dot8`.
 * This is that sequence, named, so the two arches can disagree about what happens in the middle.
 *
 * ON CDNA4 IT IS THE SAME FOUR LINES, byte for byte: one native cvt per pair with the scale
 * folded free, then v_dot2c_f32_bf16. Nothing about that path changes and nothing about its
 * numerics changes.
 *
 * ON CDNA3 THE bf16 IS PURE LOSS, and that is the whole reason this helper exists. gfx942 has no
 * v_dot2c_f32_bf16 (amd_arch.h: "needs target feature dot12-insts"), so `dot8` there UNPACKS the
 * bf16 back to f32 and issues two FMAs — meaning the shared path spent VALU packing a bf16 pair
 * that the very next instruction took apart again. MEASURED, and it is not a rounding error's
 * worth: at the large shapes the mxfp4 GEMV moved HALF the bytes of the fp8 GEMV and took the
 * SAME time (70.69 us vs 67.11 us at N=32768 K=6144), i.e. 1513 GB/s against fp8's 3002 — the op
 * was VALU-bound with the memory system half idle. So the CDNA3 arm goes fp4 -> fp16 -> f32 and
 * stops: no bf16 is ever materialised, and `v_fma_mix_f32` consumes the fp16 half directly as an
 * FMA source, which folds the widening into the multiply instead of paying a convert for it.
 *
 * TWO NUMERIC DIVERGENCES ON CDNA3, both deliberate, neither an approximation:
 *
 *   THE SCALE MOVES OUT OF THE ELEMENT AND ONTO THE SUM. All 32 elements of an MX block share one
 *   E8M0 scale, so `sum(v_i * s * x_i)` becomes `s * sum(v_i * x_i)` — 32 multiplies become one.
 *   This is EXACT, not merely close: `s` is a power of two (E8M0 is a bare exponent) and 2^14 is
 *   too, so scaling every partial sum by it is a lossless exponent shift. Only overflow or a
 *   subnormal partial could break that, and neither is reachable from e2m1's <= 3 significant
 *   bits (§ amd_arch.h).
 *
 *   THE ACCUMULATION GROUPING CHANGES. Four independent f32 accumulators, one per 8-element word,
 *   summed pairwise at the end, against CDNA4's single serial chain. It is the same license the
 *   fp8 matrix-core quartering already takes ("Only the f32 accumulation GROUPING differs from
 *   CDNA4", amd_arch.h) and it is taken for the same reason: a 32-deep dependent FMA chain at two
 *   waves per SIMD is latency the arch cannot hide. Results agree with CDNA4 to f32 rounding, NOT
 *   bit-for-bit. Any A/B that asserts elementwise equality must compare gfx942 against gfx942. */
/* SPLIT INTO PREP AND DOT, and it is not cosmetic: three of the five call sites dequant ONCE and
 * then dot against MM activation rows (the decode-batch loop). Fusing the two halves would move
 * the dequant inside that loop and repeat it per row. The carrier is register-neutral — 4 bf16v8
 * and 16 raw fp16 words are both 16 VGPRs — so the split costs nothing either arch pays for. */
struct fp4_frag32 {
#if PLOW_HAS_MX_CVT
    bf16v8 v[4]; /* scale already folded by the native cvt */
#else
    unsigned h[16]; /* raw fp16 pairs: the value * 2^-14 */
    float s;        /* scale * 2^14, applied once to the finished sum */
#endif
};

__device__ __forceinline__ fp4_frag32 fp4_prep32(fp4v32 w, float scale) {
    fp4_frag32 f;
#if PLOW_HAS_MX_CVT
    fp4_to_bf16v8x4(w, scale, f.v[0], f.v[1], f.v[2], f.v[3]);
#else
#pragma unroll
    for (int i = 0; i < 4; i++) plow_fp4x8_to_f16x4((unsigned)w[i], &f.h[i * 4]);
    f.s = scale * 16384.0f;
#endif
    return f;
}

__device__ __forceinline__ float fp4_dot32(const fp4_frag32& f, bf16v8 x0, bf16v8 x1, bf16v8 x2,
                                           bf16v8 x3, float acc) {
#if PLOW_HAS_MX_CVT
    return dot8(f.v[3], x3, dot8(f.v[2], x2, dot8(f.v[1], x1, dot8(f.v[0], x0, acc))));
#else
    const bf16v8 xs[4] = {x0, x1, x2, x3};
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        bf16v8_pairs xp{xs[i]};
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const plow_f16x2 wv = __builtin_bit_cast(plow_f16x2, f.h[i * 4 + j]);
            /* bf16 -> f32 is a 16-bit SHIFT, not a numeric conversion; see plow_dot2_bf16's note
             * on the cast that looks right and silently returns zero. */
            const unsigned xu = __builtin_bit_cast(unsigned, xp.p[j]);
            float xa, xb;
            unsigned t = xu << 16;
            __builtin_memcpy(&xa, &t, 4);
            t = xu & 0xffff0000u;
            __builtin_memcpy(&xb, &t, 4);
            s0 = __builtin_fmaf((float)wv[0], xa, s0); /* v_fma_mix_f32: f16 source, f32 acc */
            s1 = __builtin_fmaf((float)wv[1], xb, s1);
        }
        p[i] = s0 + s1;
    }
    return acc + ((p[0] + p[1]) + (p[2] + p[3])) * f.s;
#endif
}

/* One u32 (8 fp4) + one E8M0 scale -> bf16v8, the per-word slice of fp4_to_bf16v8x4. Used by the
 * w4a16 prefill GEMM's dequant-on-load B-fetch, where the 8-half load granularity wants exactly 8
 * bf16 at a time (an 8-element load never crosses a 32-element MX block, so one scale byte covers
 * it). Same 4 packed op_sel converts, same fdot2-ready order — the scale fold stays EXACT. */
#if PLOW_HAS_MX_CVT /* CDNA4 MX block-scale convert; CDNA3 has no fp4 type and no scalef32 */
__device__ __forceinline__ bf16v8 fp4_to_bf16v8(unsigned w, float scale) {
    union { bf16v8 v; unsigned u[4]; } o;
    o.u[0] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 0));
    o.u[1] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 1));
    o.u[2] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 2));
    o.u[3] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 3));
    return o.v;
}
#else
__device__ __forceinline__ bf16v8 fp4_to_bf16v8(unsigned w, float scale) {
    union { bf16v8 v; unsigned u[4]; } o;
    plow_fp4x8_to_bf16x8(w, scale, o.u);
    return o.v;
}
#endif /* PLOW_HAS_MX_CVT */

/* ---------------------------------------------------------------------------
 * A4W4 — BOTH OPERANDS MXFP4 THROUGH THE MATRIX CORE.  [MEASURED ON gfx950 2026-07-27]
 *
 * The mxfp4 path elsewhere in this file is w4a16: fp4 weights are dequantized to bf16 and fed
 * to a bf16 MFMA. That wins weight BANDWIDTH and nothing else — the matrix core still runs at
 * the bf16 rate and the activation still crosses at 16 bits. CDNA4 can take fp4 on BOTH
 * operands with per-32-element E8M0 scales applied by the hardware, which is what
 * v_mfma_scale_f32_32x32x64_f8f6f4 with cbsz=blgp=4 is for.
 *
 * EVERY CLAIM BELOW WAS MEASURED against a CPU reference on this GPU (/tmp/a4w4/probe.hip;
 * the same checks are in runtime/tests/a4w4_gfx950_test.hip). None of it is from documentation.
 *
 *   OPERAND LAYOUT (cbsz=blgp=4, K=64 per instruction):
 *       lane l supplies row = l % 32 and k = 32*(l/32) + [0..31]
 *       = 32 fp4 = 16 bytes, packed 2/byte, in the LOW 4 DWORDS of the v8i32 operand.
 *     The builtin's parameter type is v8i32 for every f8f6f4 format; for fp4 the upper 4
 *     dwords are not read. Passing 16 bytes and leaving the rest zero is correct, and it is
 *     why an fp4 fragment is a ds_read_b128 and not the fp8 path's ds_read_b256.
 *
 *   ACCUMULATOR LAYOUT is the SAME as the bf16 32x32 MFMA, so mfma_acc_m / mfma_acc_n and
 *     every epilogue in this codebase carry over unchanged (verified: exact match).
 *
 *   THE E8M0 SCALE BYTE IS BIASED BY 127, AND 0 IS NOT NEUTRAL. Byte b means 2^(b-127) —
 *     the same convention e8m0_to_f32() above already implements. NEUTRAL IS 127. Byte 0 is
 *     2^-127, which flushes the whole product to ZERO (measured: rel err 1.0, output exactly
 *     0.0). Per-block scales read from memory are applied EXACTLY (measured rel err 0.0).
 *
 *   THE TRAP THAT MAKES THAT DANGEROUS, and the reason d_gemm_fp8_t looks like it disagrees:
 *     when the scale arguments are COMPILE-TIME CONSTANTS the backend selects the UNSCALED
 *     v_mfma_f32_32x32x64_f8f6f4 and drops them, so d_gemm_fp8_t's literal 0s are harmless
 *     there — it is running the unscaled instruction (verified in its disassembly: 34x
 *     v_mfma_f32_32x32x64_f8f6f4, zero v_mfma_scale_*). Its comment calling byte 0 "the
 *     NEUTRAL MX scale" is wrong about the ISA and right about that kernel's behaviour, which
 *     is the worst possible combination to read while writing a new one. A RUNTIME scale
 *     operand selects the scaled instruction and byte 0 then silently zeroes your output.
 *     If you write an A4W4 kernel, assert the scaled mnemonic in scripts/asm_expect_gfx950.json
 *     — that is exactly the "correct but silently wrong instruction" case the audit exists for.
 *
 * Arg order (9): (a, b, acc, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b). */
typedef int mfma_f8f6f4_operand __attribute__((ext_vector_type(8)));

/* Pack a lane's 16 fp4 bytes into the low half of the builtin's v8i32 operand. */
__device__ __forceinline__ mfma_f8f6f4_operand fp4_frag(const void* p16) {
    mfma_f8f6f4_operand o;
    union { unsigned u[4]; } q;
    __builtin_memcpy(&q, p16, 16);
#pragma unroll
    for (int i = 0; i < 4; i++) o[i] = (int)q.u[i];
#pragma unroll
    for (int i = 4; i < 8; i++) o[i] = 0;
    return o;
}
/* E8M0 byte meaning 2^0. NOT zero — see the contract above. */
#define PLOW_E8M0_ONE 127

/* One A4W4 MFMA: 32x32 output, K=64, per-lane E8M0 scales for both operands. */
#if PLOW_HAS_MX_CVT /* CDNA4 MX block-scale convert; CDNA3 has no fp4 type and no scalef32 */
__device__ __forceinline__ f32x16 mfma_a4w4(mfma_f8f6f4_operand a, mfma_f8f6f4_operand b,
                                            f32x16 acc, int scale_a, int scale_b) {
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, acc, /*cbsz=fp4*/ 4,
                                                           /*blgp=fp4*/ 4, 0, scale_a, 0, scale_b);
}
#endif /* PLOW_HAS_MX_CVT */

/* f32 -> one OCP e2m1 nibble (RNE, saturating at 6.0). The quantizer half of A4W4: the fused
 * SwiGLU bridge writes the intermediate activation straight out in this format, so it never
 * crosses HBM at 16 bits and never needs a second pass to quantize. */
__device__ __forceinline__ unsigned quant_fp4(float v) {
    const unsigned sign = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    /* e2m1 codes: 0,0.5,1,1.5,2,3,4,6. Round-to-nearest on the 8-point ladder, TIES AWAY FROM
     * ZERO — not round-to-nearest-even, which is what this comment used to claim. The cut points
     * below are the exact midpoints and each uses `<`, so a value sitting on one rounds UP:
     * 0.25 -> 0.5 (RNE: 0), 1.25 -> 1.5 (RNE: 1.0), 2.5 -> 3.0 (RNE: 2.0). Only exact ties
     * differ, so real data almost never sees it — but a tie-heavy synthetic fixture will, and a
     * host-side reference that implements true RNE would disagree with the device on exactly
     * those elements. Stated rather than changed: the ladder is what the shipped objects were
     * measured with, and flipping it would move numerics to fix a comment.
     *
     * COUNTED, NOT BRANCHED. As an if/else chain this compiled to a SEVEN-DEEP nest of
     * `s_and_saveexec_b64` / `s_cbranch_execz` / `s_or_b64 exec` — and since a wavefront's 64
     * lanes land on different rungs, every rung executes anyway, so the chain cost all seven
     * comparisons PLUS ~21 exec-mask instructions per element. The A4W4 staging quantizer runs
     * this 32 times per MX block and it was the single largest VALU term in the MoE prefill.
     * The code is just "how many cut points is `a` not below".
     *
     * SEVEN FLOAT COMPARES AND NOT SOMETHING CLEVERER — MEASURED. Five of the seven rungs are an
     * arithmetic progression in the INTEGER encoding (1.25, 1.75, 2.5, 3.5, 5.0 are 0x3FA00000 +
     * k*0x400000), so their combined count can be had from one subtract, one arithmetic shift
     * and a clamp instead of five comparisons: ten VALU per element rather than sixteen. It is
     * SLOWER. `|x|` is a free source modifier on `v_cmp_*_f32` but not on the integer ops, so the
     * bit form has to materialise the magnitude, and it replaces seven independent compares —
     * which the scheduler interleaves with the `v_pk_mul_f32` scaling — with a serial
     * subtract -> shift -> med3 -> add chain. Measured at K3's MoE prefill shape: 7.617 ms the
     * way it is written below, 7.749 ms with the progression trick. Fewer instructions, less
     * throughput. Do not re-derive it.
     *
     * `!(a < T)` and not `a >= T`: both are false for a NaN, on opposite sides. The if/else chain
     * fell through EVERY rung for a NaN and returned 7, `!(a < T)` is true for NaN and reproduces
     * that, `a >= T` would return 0 instead. runtime/tests/fp4_quant_identity_gfx950_test.hip
     * walks all 2^32 inputs against the frozen chain — every NaN payload, both zeros, every
     * subnormal — which is what makes any of this admissible. */
    const unsigned n = (unsigned)!(a < 0.25f) + (unsigned)!(a < 0.75f) + (unsigned)!(a < 1.25f) +
                       (unsigned)!(a < 1.75f) + (unsigned)!(a < 2.5f) + (unsigned)!(a < 3.5f) +
                       (unsigned)!(a < 5.0f);
    return sign | n;
}
/* E8M0 exponent for a block whose max magnitude is `amax`, mapping it onto e2m1's 6.0 top code.
 * Power-of-two only by construction, which is what makes the MX scale exact. */
__device__ __forceinline__ unsigned char e8m0_for_amax(float amax) {
    if (!(amax > 0.0f)) return (unsigned char)PLOW_E8M0_ONE;
    /* WHAT THIS USED TO BE: `frexpf(amax / 6.0f, &e)` — amax/6 in [0.5,1) * 2^e, so 2^e covers
     * the block. Correct, and a full IEEE division (thirteen VALU) to extract ONE exponent.
     *
     * It is an integer question. Write a normal amax as 2^(E-127) * (1 + M/2^23) with E the
     * biased exponent field and M the mantissa field. Then amax/6 = (1 + M/2^23)/6 * 2^(E-127),
     * and (1 + M/2^23)/6 lands in [1/6, 1/3) — which straddles exactly ONE power of two, 1/4. So
     * frexp's exponent is E-128 if the quotient is below 1/4 and E-127 if not, and the quotient
     * is below 1/4 exactly when 1 + M/2^23 < 1.5, i.e. when M < 2^22. The +127 folds in and the
     * whole thing is `E - 1 - (M < 2^22)`.
     *
     * The boundary is EXACT and needs no rounding argument: 1.5/6 = 0.25 with both operands
     * powers-of-two-times-small-integers, so the division there is exact. The float one ulp below
     * 1.5 divides to strictly below 0.25 after rounding, which is the case the sweep test pins.
     *
     * SUBNORMAL amax (E = 0) mostly falls out: the formula gives -1 or -2 and clamps to 1, and the
     * old path clamped to 1 too (amax < 2^-126 => amax/6 < 2^-128 => e + 127 <= -1). TWO ENDS DO
     * NOT FALL OUT, and both are short-circuited above rather than left to the arithmetic:
     *
     *   +inf          `v_frexp_exp_i32_f32` answers 0 for it, so the old path returned 0+127=127.
     *   bits 1, 2, 3  the three smallest subnormals. amax/6 rounds to ZERO there (3*2^-149 / 6 is
     *                 exactly half an ulp and goes to even), and frexpf(0) also reports 0, so the
     *                 old path returned 127 rather than the clamped 1 the formula gives. Exactly
     *                 three inputs in 2^32, found by the sweep and not by reading the code.
     *
     * Bit-identical for every one of the 2^32 float inputs — see
     * runtime/tests/fp4_quant_identity_gfx950_test.hip, which is the only reason this is allowed
     * to be clever. */
    const unsigned bits = __builtin_bit_cast(unsigned, amax);
    const unsigned E = bits >> 23; /* amax > 0 here, so the sign bit is clear */
    if (E == 0xFFu || bits < 4u) return (unsigned char)PLOW_E8M0_ONE;
    int e = (int)E - 1 - (int)((bits & 0x7FFFFFu) < 0x400000u);
    if (e < 1) e = 1;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

/* ---------------------------------------------------------------------------
 * FP8 (OCP e4m3) KV-CACHE loads/stores — the decode flash-attention path.
 *
 * The KV cache is HBM-bound in decode (op_attention.h: "1.91 ms @ 2.0 TB/s"), so storing K/V as
 * e4m3 halves the KV stream — the same 2x roofline the fp8 GEMV gets. These mirror the fp8 weight
 * helpers above: the DECODE (fp8 -> bf16) side reuses fp8_to_bf16v8; this block adds the two things
 * the KV path needs that the weight path does not — an 8-wide (b64) decode for the V phase, which
 * reads only 8 head-dims per lane, and an ENCODE (f32 -> e4m3) for the HeadNormRope write. */

/* A whole K row is a b128 (16 fp8) load, exactly like the fp8 weight stream. */
__device__ __forceinline__ fp8v16 ld_glob_fp8v16(const unsigned char* p) {
    return *(const PLOW_GLOB fp8v16*)(const PLOW_GLOB void*)p;
}

/* 8 fp8 == 2 u32 == 8 B: the V phase reads only its 8 owned head-dims per lane, so it wants HALF
 * the b128 width. Decode each adjacent pair directly to packed bf16 with the same native
 * instruction as fp8_to_bf16v8; the old fp8->f32->high-half path spent roughly three times the
 * conversion VALU in the cache loop. */
typedef unsigned fp8v8 __attribute__((ext_vector_type(2))); /* 8 fp8, 8 B, align 8 */
__device__ __forceinline__ fp8v8 ld_glob_fp8v8(const unsigned char* p) {
    return *(const PLOW_GLOB fp8v8*)(const PLOW_GLOB void*)p;
}
#if PLOW_HAS_MX_CVT /* CDNA4 MX block-scale convert; CDNA3 has no fp4 type and no scalef32 */
__device__ __forceinline__ bf16v8 fp8v8_to_bf16v8(fp8v8 w) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    union {
        bf16v8 v;
        unsigned u[4];
    } d;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const bf16_2 a =
            __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[i], 1.0f, false); /* bytes 0,1 */
        const bf16_2 c =
            __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[i], 1.0f, true); /* bytes 2,3 */
        d.u[i * 2 + 0] = __builtin_bit_cast(unsigned, a);
        d.u[i * 2 + 1] = __builtin_bit_cast(unsigned, c);
    }
    return d.v;
}
#else
/* CDNA3 arm: same two-step decode as fp8_to_bf16v8 above, and bit-identical for the same
 * reason (fp8 e4m3 is exact in bf16). */
__device__ __forceinline__ bf16v8 fp8v8_to_bf16v8(fp8v8 w) {
    union {
        bf16v8 v;
        unsigned u[4];
    } d;
#pragma unroll
    for (int i = 0; i < 2; i++) plow_fp8x4_ocp_to_bf16((unsigned)w[i], d.u[i * 2], d.u[i * 2 + 1]);
    return d.v;
}
#endif /* PLOW_HAS_MX_CVT */

/* f32 -> one e4m3 byte. gfx950 native: cvt_pk_fp8_f32 packs (a,b) into two bytes of `old`; we use
 * one and keep byte 0. RNE + saturation are in hardware. */
__device__ __forceinline__ unsigned char quant_fp8(float v) {
#if !PLOW_CDNA4
    /* CDNA3's encoder emits e4m3FNUZ, a different format -- see amd_arch.h. */
    return plow_f32_to_fp8_ocp(v);
#else
    const unsigned packed = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
    return (unsigned char)(packed & 0xffu);
#endif
}

/* Reduce a MAX across the 64 lanes of a wave (the HeadNormRope per-row amax for the KV scale). */
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 64));
    return v;
}

/* e4m3 max finite magnitude (torch.float8_e4m3fn): 448. A row's scale maps its amax to 448 so the
 * largest element uses the full e4m3 range; the reciprocal is what the write multiplies by. */
#define PLOW_FP8_E4M3_MAX 448.0f

/* Wave is 64 lanes on CDNA. __shfl_sync/warpSize-32 assumptions from CUDA code
 * do not port — this is the reason the NVIDIA RoPE reference cannot be
 * transliterated (it pairs lane i with lane i+32). */
#include "../common/dev_isa.h"

#define PLOW_WAVE 64

/* Workgroup shape for the persistent interpreter: ONE workgroup per CU, 4 waves.
 *
 * A persistent interpreter owns the CU (co-residency is what makes the counter
 * spin safe), so it has NO occupancy to hide HBM latency with: when a wave stalls
 * on a global load, a 4-wave workgroup (1 wave/SIMD) has nothing to switch to.
 * That suggested 8 waves (2/SIMD) would be a large win, and in an isolated GEMM
 * it is a real one — but a small one, and it is NOT the main lever:
 *
 *   gate/up_proj (4096 x 21504 x 5376), full GPU, vs the 2308 TF/s MFMA peak:
 *     4 waves, duplicated fetch path (spilling)    230 TF/s
 *     8 waves, duplicated fetch path (spilling)    279
 *     8 waves, single fetch path                   449
 *     4 waves, single fetch path                   405-502   <- current
 *
 * The dominant factor was REGISTER PRESSURE, not wave count. A duplicated
 * "interior tile" / "edge tile" fetch path doubled the live set and spilled ~1 KB
 * per lane into scratch (i.e. HBM) inside the mainloop; collapsing it to one
 * predicated path was worth more than the extra waves.
 *
 * We stay at 4 waves because the interpreter inlines every op, so its register
 * allocation is the worst case over all of them — and flash at head_dim=512 holds
 * Q as MFMA fragments (D/2 halves = 128 VGPRs) plus the O accumulator, pushing the
 * kernel to 384 arch VGPRs. At 512 threads a wave may use at most 256 (two waves
 * must be resident per SIMD), so an 8-wave interpreter is REJECTED AT DISPATCH
 * with HSA_STATUS_ERROR_INVALID_ISA — which reads like a code-object problem and
 * is nothing of the sort. Empirically 256 launches, 384 does not.
 *
 * To unlock 8 waves for the interpreter, flash must stop holding all D/16 Q
 * fragments live: accumulate S over D-chunks so only FA_DC/16 are live at once.
 *
 * NOTE on __launch_bounds__: on AMD the second argument is MIN WAVES PER EU, not
 * min blocks per CU as in CUDA. Passing 1 lets the compiler take all 512
 * registers. Kernels use __launch_bounds__(PLOW_THREADS, PLOW_WAVES_PER_EU).
 *
 * Every op assumes PLOW_THREADS, never a hardcoded 256. */
#define PLOW_WAVES PLOW_WG_WAVES               /* from dev_isa.h: host + device agree */
#define PLOW_THREADS PLOW_WG_THREADS           /* 512: 8 waves, 2 per SIMD          */
#define PLOW_WAVES_PER_EU (PLOW_WAVES / 4)     /* 4 SIMDs per CU */

/* Elements one thread holds when a norm row fits in registers. 16 * 512 = 8192 covers every
 * RMSNorm in Gemma (hidden = 5376) and both of K3's fusable decode norms (3584, 1536); wider
 * rows fall back to the streaming path.
 *
 * Held as RN_VEC 16-BYTE VECTOR loads, not RN_REG scalar ones. A `const bf16*` is a generic,
 * align-2 pointer, so `x[base + i]` compiled to `flat_load_ushort` -- two bytes per
 * instruction, sixteen instructions per thread. That is ruinous precisely HERE: a decode norm
 * is a single row on a single CU with all 255 others stalled on its counter, so its cost is
 * pure issue-and-latency, and there is no other work to hide it behind. See as_glob() and
 * ld_glob8() below.
 *
 * HERE rather than in op_norm.h because `d_rmsnorm` (op_norm.h) and the fused-norm GEMV
 * (`gemv_norm_lds`, op_gemm.h, included FIRST) must walk the row with the identical
 * per-thread element map or the fused arm is not bit-exact. One constant, one definition. */
#define RN_REG 16
#define RN_VEC (RN_REG / 8) /* 16 halves = 2 x bf16v8 */

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, PLOW_WAVE);
    return v;
}

/* Reduce across the 32 lanes of a HALF-wave. The MFMA 32x32 accumulator layout
 * puts one output row entirely inside one half-wave (lanes 0-31 or 32-63), so a
 * row-wise softmax reduction must stop at 32 — going to 64 would fold two
 * different rows together. */
__device__ __forceinline__ float half_wave_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, PLOW_WAVE));
    return v;
}
__device__ __forceinline__ float half_wave_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, PLOW_WAVE);
    return v;
}

/* ONE spelling of the logistic for the whole K3 family. `situ`'s gate branch and the MLA OUTPUT
 * GATE (`mla_use_output_gate`, op 106, op_k3.h) both need it, and they must not drift apart: the
 * two are a factor of `beta*tanh(g/beta)` from each other, so an accidental substitution of one for
 * the other is finite, correctly-shaped and wrong.
 *
 * `v_rcp_f32` AND NOT `1.0f /`. The logistic's denominator is never zero and never overflows —
 * `1 + exp(-x)` is in [1, +inf) with a hard floor of exactly 1 — so none of the IEEE division
 * expansion's scaling machinery can ever fire, and it is thirteen VALU (v_div_scale x2, v_rcp,
 * three FMA refinements, v_div_fmas, v_div_fixup) to compute a reciprocal the transcendental
 * unit answers in one. The same argument `e8m0_inv_f32` above makes, for the same reason.
 *
 * The refinement steps the division does and the rcp does not are worth ~1 ULP of f32, against
 * an activation that is rounded to bf16 (8 mantissa bits, 2^-9 relative) before anything reads
 * it. runtime/tests/situ_identity_gfx950_test.hip walks all 2^32 inputs and reports both the f32
 * error and the count of inputs whose BF16 output moves; that count is what makes this
 * admissible, and it is the file to re-run if this line is ever touched again. */
__device__ __forceinline__ float k3_sigmoid(float x) {
    return __builtin_amdgcn_rcpf(1.0f + __expf(-x));
}

/* tanh from the hardware exponential:  tanh(y) = 1 - 2/(e^(2y) + 1).
 *
 * WHAT IT REPLACES. `tanhf` is OCML's correctly-rounded tanh, and on gfx950 it compiles to ~30
 * VALU plus a TWO-DEEP `s_and_saveexec_b64`/`s_cbranch_execz` nest — and a wavefront whose 64
 * lanes straddle the range split executes both sides regardless, which is the same trap
 * `quant_fp4`'s ladder was. This is five: v_mul, v_exp_f32, v_add, v_rcp_f32, v_fma.
 *
 * THE ENDS ARE THE FORMULA'S, NOT A CLAMP'S, and they come out right by themselves:
 *   y >= +44   e^(2y) overflows to +inf, rcp(+inf) = +0, so tanh -> exactly +1.
 *   y <= -44   e^(2y) flushes to +0, 0 + 1 == 1 exactly, rcp(1) = 1, so tanh -> exactly -1.
 *   y = +-inf  the same two limits, reached exactly.
 *   NaN        e^NaN = NaN and NaN propagates through add/rcp/fma, so NaN in, NaN out — which
 *              is what `tanhf` does and what the `situ` epilogue needs, since op_moe.h's
 *              `moe_act` uses a NaN as its deliberate poison for an unconverted caller.
 * There is no `1 + e` overflow to guard: e is non-negative, so the sum only ever grows toward
 * the +inf that already maps to the correct limit.
 *
 * WHERE IT IS WEAKEST, STATED PLAINLY: y -> 0. `1 - 2r` is a cancellation, so the ABSOLUTE error
 * stays at ~half an f32 ulp of 1.0 (~6e-8) while the value itself goes to zero — the RELATIVE
 * error therefore grows without bound as y -> 0. It is bounded absolute error on a quantity
 * whose true magnitude is |y|, so it cannot perturb a sum by more than 6e-8 per term, and the
 * exhaustive sweep in runtime/tests/situ_identity_gfx950_test.hip reports exactly which inputs
 * move a BF16 result and by how much. A degree-5 odd polynomial on |y| < 0.09 would remove it
 * for ~5 more VALU; the sweep says it is not needed, and that measurement is the reason it is
 * absent rather than an oversight.
 *
 * NOT `(1-E)/(1+E)` with E = e^(-2y): same cancellation, one more VALU, no accuracy gained. */
__device__ __forceinline__ float fast_tanhf(float y) {
    return __builtin_fmaf(-2.0f, __builtin_amdgcn_rcpf(__expf(2.0f * y) + 1.0f), 1.0f);
}

/* Kimi-K3's `situ` activation, as a PAIR, because it transforms BOTH GLU branches:
 *     out = beta*tanh(g/beta)*sigmoid(g)  *  linear_beta*tanh(u/linear_beta)
 * (modeling_kimi_linear.py:75-85, `activation_situ_beta` 4.0, `activation_situ_linear_beta` 25.0).
 *
 * It lives HERE rather than in op_k3.h because two call sites need it — the dense/shared GLU
 * (PLOW_DOP_SITU_GLU, op_k3.h) and the ROUTED EXPERT GLU (op_moe.h) — and op_moe.h must not
 * depend on op_k3.h. Two copies of a transcendental expression is exactly how a model ends up
 * computing subtly different activations in its dense and expert paths.
 *
 * It is a soft-clipped SiLU: as beta -> inf, `beta*tanh(g/beta) -> g` and the gate branch becomes
 * silu. `lb <= 0` means "no up transform" (`linear_beta is None`), chosen as a comparison rather
 * than a flag so a zeroed immediate degrades to the identity instead of clipping to zero.
 *
 * THIS IS THE HOTTEST ARITHMETIC IN KIMI-K3. It runs on every element of every routed expert
 * (top-16 of 896), both shared experts and the dense FFN, on 92 of 93 layers. Spelled the
 * obvious way it was THREE full IEEE divisions (`g/beta`, `u/lb`, and the logistic's `1.0f/`),
 * two `tanhf` and one exponential per output element: 109 VALU and a nine-instruction exec-mask
 * nest, measured on gfx950. Two of the three divisions and both `tanhf` are gone; what is left
 * is ~46 VALU and branchless, and the paragraph below is why the third division stayed.
 *
 * `g / beta` IS A PER-ELEMENT DIVISION AND IT STAYS ONE, WHICH IS NOT THE OBVIOUS ANSWER.
 * `beta` and `lb` are packet immediates — uniform over the whole kernel — so the textbook move is
 * `g * (2.0f/beta)`, hoisting one reciprocal out of the element loop and leaving a single v_mul
 * per element. That version was written, measured, and REJECTED. On K3's MoE prefill geometry
 * (T=1024, 896 experts, grid 512, gfx950):
 *
 *     op85 GLU+bridge          act=silu    act=situ
 *     base                      7.603 ms    7.843 ms
 *     + rcp logistic            7.603       7.792
 *     + fast_tanhf, g/beta      7.604       7.740      <- what is written below
 *     + hoisted 2.0f/beta       7.847       7.901      <- the "better" version
 *     + hoisted rcp(beta)       7.833       7.862      (VGPR 165 -> 167)
 *
 * The hoist costs 0.24 ms and it costs it on the SILU ARM TOO — an arm that never evaluates
 * situ at all. That is the tell: the penalty is not the arithmetic, it is that LICM lifts the
 * reciprocal to the kernel prologue, where its result is a VGPR that stays live across all 28
 * K-tiles of the MFMA main loop. `d_moe_group_pf_a4w4` runs at 165 VGPR and 2 waves/SIMD with no
 * slack (see MPF4_LDS_BYTES's note), so two more loop-lifetime VGPRs cost more than the
 * thirteen-VALU division saves in an epilogue the main loop already hides.
 *
 * THE ONE-SPELLING RULE IS WHY THIS IS NOT SPLIT IN TWO. The hoisted form IS faster for
 * `d_situ_glu` (op 105, the dense/shared streaming pass): 0.0248 ms vs 0.0365 at K3's 18432
 * width. Giving the streaming op its own hoisted variant would buy ~0.34 ms per forward pass at
 * T=1024 — against the ~9.8 ms the form below buys on op85 over the same 92 layers — and the
 * price would be that the dense FFN and the routed experts compute situ with two different
 * roundings, which is exactly the drift this header exists to prevent. 3% more, for the one
 * defect the file is written to make impossible. One spelling. If a future kernel really wants
 * the hoist, the honest way to get it is to make the PACKET carry 2/beta: the value then arrives
 * in an SGPR, costs no VGPR at all, and both call sites still share a single expression.
 *
 * The `lb > 0` test is on a UNIFORM value, so it is an `s_cmp`/`s_cbranch` over the whole
 * wavefront — a scalar branch with no exec-mask cost — and not the per-lane divergence a
 * per-element predicate would be. */
__device__ __forceinline__ float k3_situ_gate(float g, float beta) {
    return beta * fast_tanhf(g / beta) * k3_sigmoid(g);
}
__device__ __forceinline__ float k3_situ_up(float u, float lb) {
    return lb > 0.0f ? lb * fast_tanhf(u / lb) : u;
}

/* Block reduction over the PLOW_WAVES waves of the workgroup. */
__device__ __forceinline__ float block_sum(float v, float* part) {
    v = wave_sum(v);
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    if (lane == 0) part[wave] = v;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < PLOW_WAVES; i++) t += part[i];
    __syncthreads(); /* `part` is reused by the next op in the interpreter loop */
    return t;
}

/* ---------------------------------------------------------------------------
 * v_mfma_f32_32x32x16_bf16 register layout.
 *
 * CONFIRMED empirically on gfx950 against a CPU reference (0/1024 elements
 * wrong). Do not "correct" these from memory — a wrong lane map yields a GEMM
 * that is almost right, which is the worst possible failure mode.
 *
 *   A[32][16] : lane l holds  m = l%32,  k = 8*(l/32) + j        (j = 0..7)
 *   B[16][32] : lane l holds  n = l%32,  k = 8*(l/32) + j        (j = 0..7)
 *   D[32][32] : lane l holds  n = l%32,  m = 4*(l/32) + (i%4) + 8*(i/4)  (i = 0..15)
 *
 * Both operands want k contiguous, which is why every LDS tile below is stored
 * row-major with K innermost: one 16-byte LDS read fills a fragment.
 * ------------------------------------------------------------------------- */
#define MFMA_M 32
#define MFMA_N 32
#define MFMA_K 16

/* k-octet this lane supplies for an A/B fragment at k-step `kk`. */
__device__ __forceinline__ unsigned mfma_frag_k(unsigned lane, unsigned kk) {
    return kk + 8 * (lane / 32);
}
/* row (m for A, n for B) this lane supplies. */
__device__ __forceinline__ unsigned mfma_frag_row(unsigned lane) { return lane % 32; }
/* m of accumulator element `i` for this lane. */
__device__ __forceinline__ unsigned mfma_acc_m(unsigned lane, unsigned i) {
    return 4 * (lane / 32) + (i % 4) + 8 * (i / 4);
}
/* n of every accumulator element for this lane. */
__device__ __forceinline__ unsigned mfma_acc_n(unsigned lane) { return lane % 32; }

#endif /* PLOW_AMD_COMMON_H */
