/* hsa_backend.h — AMD device layer on ROCr/HSA. No HIP runtime.
 *
 * This is the "just above the kernel driver" layer: libhsa-runtime64 sits
 * directly on the amdkfd ioctl interface. We use it for agent discovery, memory
 * pools, code-object load, and — the part that matters for latency — writing
 * AQL kernel-dispatch packets straight into a hardware queue and ringing the
 * doorbell. A dispatch is a memcpy of the kernarg block, ~16 stores into the
 * queue's ring, one release store of the packet header, and one doorbell store.
 * There is no driver round-trip and no HIP launch path.
 *
 * Ownership: `plow_hsa_init` builds one context per process. Each GPU agent
 * gets its own AQL queue, so `plowrt`'s executors dispatch without contending.
 */
#ifndef PLOW_HSA_BACKEND_H
#define PLOW_HSA_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-process HSA context. */
typedef struct plow_hsa plow_hsa;

/* Everything the AQL packet processor needs to launch one kernel. Resolved once
 * at load time from the code object's metadata, then reused per dispatch. */
typedef struct {
    uint64_t kernel_object;      /* .kd address                                 */
    uint32_t kernarg_size;       /* explicit args + COv5 hidden block           */
    uint32_t group_segment_size; /* static LDS the kernel declares              */
    uint32_t private_segment_size;
    uint32_t kernarg_explicit;   /* byte offset where the hidden block starts   */
} plow_hsa_kernel;

/* Bring up ROCr, enumerate GPU agents, bind memory pools, create one AQL queue
 * per agent. Returns NULL on failure (see `plow_hsa_last_error`). */
plow_hsa* plow_hsa_init(void);
void      plow_hsa_shutdown(plow_hsa* h);
const char* plow_hsa_last_error(void);

/* Number of GPU agents discovered. */
int  plow_hsa_device_count(const plow_hsa* h);
/* gfx name (e.g. "gfx950"), CU count, LDS bytes/CU for agent `dev`. */
int  plow_hsa_device_info(const plow_hsa* h, int dev, char name[64],
                          uint32_t* cus, uint32_t* lds_bytes);

/* --- memory ---------------------------------------------------------------
 * `alloc` is coarse-grained device VRAM (the arena weights and KV live in).
 * `alloc_host` is fine-grained, page-locked system memory that every GPU agent
 * can read — the staging buffer safetensors shards are streamed through. */
void* plow_hsa_alloc(plow_hsa* h, int dev, size_t bytes);
void  plow_hsa_free(plow_hsa* h, void* p);
void* plow_hsa_alloc_host(plow_hsa* h, size_t bytes);

/* --- cross-GPU transport (tensor-parallel decode) -------------------------
 * `alloc_peer` is coarse-grained device VRAM on `owner_dev`, but mapped for
 * peer access by EVERY GPU agent — so a kernel on any device can load/store it
 * over XGMI, and a system-scope atomic on it synchronizes two GPUs' kernels
 * (the cross-GPU counter-gate primitive). Same free path as plow_hsa_free.
 *
 * `copy_p2p` is a blocking device-to-device SDMA copy over the XGMI fabric; it
 * is the bulk transport path (prefill activations) and the way we measure P2P
 * bandwidth/latency. Both dst and src must be peer-visible (alloc_peer). */
void* plow_hsa_alloc_peer(plow_hsa* h, int owner_dev, size_t bytes);
int   plow_hsa_copy_p2p(plow_hsa* h, int dst_dev, void* dst,
                        int src_dev, const void* src, size_t bytes);

/* Blocking H2D / D2H over the SDMA engines.
 *
 * CONTRACT: the host side MUST be memory from `plow_hsa_alloc_host`. The SDMA
 * engine can only touch host pages ROCr knows about — handing it a `malloc` or
 * stack pointer faults the GPU. This is the bulk path (weights, activations), so
 * it does no pinning of its own. */
int plow_hsa_copy_h2d(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes);
int plow_hsa_copy_d2h(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes);

/* Control-plane copies to/from ARBITRARY host memory (stack, malloc, mmap). These
 * pin the page range for the duration of the copy, so they are correct but not
 * free — use them for the program, the tensor table and counters, never in a
 * per-token path. */
int plow_hsa_upload(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes);
int plow_hsa_download(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes);

/* --- code objects ---------------------------------------------------------- */

/* Load a *raw* AMDGPU code object (an unbundled ELF — NOT the clang offload
 * bundle that `hipcc --genco` emits; see runtime/CMakeLists.txt, which runs
 * clang-offload-bundler to strip the wrapper). */
int plow_hsa_load_code_object(plow_hsa* h, int dev, const void* elf, size_t bytes);

/* Resolve a kernel by source name (we append the ".kd" the loader expects). */
int plow_hsa_get_kernel(plow_hsa* h, int dev, const char* name, plow_hsa_kernel* out);

/* --- dispatch -------------------------------------------------------------
 * `args` is the explicit kernarg block; the hidden COv5 block (block counts,
 * group sizes, remainders, grid dims) is filled in by this call from the grid
 * geometry. Getting that wrong is silent: blockDim reads back as 0 and every
 * workgroup recomputes tile 0.
 *
 * Asynchronous. `plow_hsa_wait` drains the agent's queue. */
int plow_hsa_launch(plow_hsa* h, int dev, const plow_hsa_kernel* k,
                    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                    uint16_t wg_x, uint16_t wg_y, uint16_t wg_z,
                    uint32_t dynamic_lds,
                    const void* args, size_t args_size);

int plow_hsa_wait(plow_hsa* h, int dev);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_HSA_BACKEND_H */
