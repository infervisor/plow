/* pdl.cuh — programmatic dependent launch (PDL) entry for sm_90a kernels.
 *
 * plowrt launches a kernel with programmatic stream serialization only when its object exports
 * `plow_pdl_entry__<kernel>` (PLOW_PDL_KERNEL). Such a launch may become resident while its
 * stream predecessor still runs, so the kernel MUST call plow_pdl_entry() before its first
 * global memory access (read or write; parameters and constant memory are fine). The wait blocks
 * until every prerequisite grid has completed and flushed, which keeps the launch ordering
 * semantically identical to a plain launch; the trigger then lets the next kernel's blocks
 * become resident while this one runs. scripts/bench/pdl_audit.py checks every marked kernel's
 * SASS for the wait before any memory instruction or branch. */
#pragma once

__device__ __forceinline__ void plow_pdl_entry() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

#define PLOW_PDL_CAT2(a, b) a##b
#define PLOW_PDL_CAT(a, b) PLOW_PDL_CAT2(a, b)
/* Marks `name` (after macro expansion, e.g. a PLOW_SYM name) as entering through plow_pdl_entry.
 * Hopper objects only: other targets keep plain launches. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define PLOW_PDL_KERNEL(name) extern "C" __device__ unsigned PLOW_PDL_CAT(plow_pdl_entry__, name) = 1;
#else
#define PLOW_PDL_KERNEL(name)
#endif
