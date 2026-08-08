/* cuda_backend.c — NVIDIA device layer on the CUDA **Driver** API.
 *
 * This implements the exact 18 `plow_hsa_*` entry points declared in
 * runtime/amd/hsa_backend.h, so `plowrt`'s host side is device-agnostic: the
 * AMD build links hsa_backend.c, the NVIDIA build links this file, and nothing
 * above the backend changes. The names keep the `plow_hsa_` prefix on purpose —
 * they are the *interface* name, not an HSA-specific one.
 *
 * Why the driver API and not the runtime API: we need (a) load-from-memory of a
 * cubin with no host-side fatbin registration (`cuModuleLoadData`), (b) a packed
 * kernarg blob rather than an argv-of-pointers (`CU_LAUNCH_PARAM_BUFFER_POINTER`),
 * and (c) cooperative launch. Only the driver API gives all three together, and
 * it keeps this file plain C — no nvcc needed to build it.
 *
 * Mapping (the design notes:100-116):
 *   hsa_init + agent enumerate      -> cuInit / cuDeviceGet / primary context
 *   code_object_reader + freeze     -> cuModuleLoadData
 *   executable_get_symbol(".kd")    -> cuModuleGetFunction
 *   memory_pool_allocate (VRAM)     -> cuMemAlloc
 *   fine-grained host + allow_access-> cuMemHostAlloc(PORTABLE|DEVICEMAP)
 *   AQL packet write + doorbell     -> cuLaunchKernelEx (cooperative when it fits)
 *   counting done signal / wait     -> per-device stream + cuStreamSynchronize
 *   COv5 hidden kernarg block       -> NOT NEEDED; the driver supplies gridDim/
 *                                      blockDim. `kernarg_explicit` is therefore
 *                                      always == kernarg_size here.
 *
 * DIVERGENCES from the AMD backend, all deliberate:
 *  - `grid_x/y/z` keep the HSA meaning: a count of WORK-ITEMS, not blocks. We
 *    divide by the workgroup size to get the CUDA grid. Getting this backwards
 *    is the single easiest way to silently launch 1/256th of the intended work.
 *  - Cooperative launch is selected per dispatch: if the requested block count
 *    fits co-residently (occupancy x SM count) we launch cooperatively, else we
 *    launch normally. Cooperative semantics are a superset — a kernel that does
 *    not use grid.sync() behaves identically either way — so this is always
 *    correct, and it means the persistent interpreter (which sizes its grid to
 *    exactly that capacity) always gets the co-residency guarantee, while
 *    ordinary tiled kernels with huge grids are not refused at launch.
 *  - `plow_hsa_free` accepts both device and pinned-host pointers (the AMD pool
 *    free does too); we ask the driver which kind it is.
 */
#include "../amd/hsa_backend.h"

#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define PLOW_CU_MAX_DEV 16
/* Modules loaded per device. AMD keeps one frozen executable; we keep a small
 * list so a test can load an op cubin alongside the interpreter cubin. */
#define PLOW_CU_MAX_MOD 16
/* Direct-mapped cache of the cooperative-fit decision, so the steady dispatch
 * path does not call the occupancy API. Keyed by (function, block, dyn smem). */
#define PLOW_CU_COOP_CACHE 32

static char g_err[256];

static void set_err(const char* what, CUresult r) {
    const char* n = NULL;
    const char* s = NULL;
    cuGetErrorName(r, &n);
    cuGetErrorString(r, &s);
    snprintf(g_err, sizeof(g_err), "%s: %s (%s)", what, n ? n : "?", s ? s : "?");
}

const char* plow_hsa_last_error(void) { return g_err; }

#define TRY(call, what) do { CUresult r_ = (call); \
    if (r_ != CUDA_SUCCESS) { set_err(what, r_); return -1; } } while (0)

typedef struct {
    CUfunction f;
    uint32_t   block;
    uint32_t   dyn;
    int        coop;
    int        valid;
} coop_ent_t;

typedef struct {
    CUdevice  dev;
    CUcontext ctx;
    CUstream  stream;
    int       sms;
    int       coop_supported;
    CUmodule  mod[PLOW_CU_MAX_MOD];
    int       n_mod;
    coop_ent_t coop[PLOW_CU_COOP_CACHE];
} plow_cu_dev_t;

struct plow_hsa {
    plow_cu_dev_t dev[PLOW_CU_MAX_DEV];
    int           n_dev;
};

/* Every driver call is context-relative. One primary context per device, made
 * current on the calling thread before use. */
static int use_dev(plow_hsa* h, int dev) {
    if (!h || dev < 0 || dev >= h->n_dev) {
        snprintf(g_err, sizeof(g_err), "bad device index %d", dev);
        return -1;
    }
    TRY(cuCtxSetCurrent(h->dev[dev].ctx), "cuCtxSetCurrent");
    return 0;
}

/* --- discovery ------------------------------------------------------------ */

plow_hsa* plow_hsa_init(void) {
    static struct plow_hsa h;
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { set_err("cuInit", r); return NULL; }

    int n = 0;
    r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS) { set_err("cuDeviceGetCount", r); return NULL; }
    if (n <= 0) { snprintf(g_err, sizeof(g_err), "no CUDA devices"); return NULL; }
    if (n > PLOW_CU_MAX_DEV) n = PLOW_CU_MAX_DEV;

    memset(&h, 0, sizeof(h));
    for (int i = 0; i < n; i++) {
        plow_cu_dev_t* d = &h.dev[i];
        r = cuDeviceGet(&d->dev, i);
        if (r != CUDA_SUCCESS) { set_err("cuDeviceGet", r); return NULL; }
        /* Primary context: shared with anything else in-process that uses the
         * runtime API, which is what lets a cubin built by nvcc and a buffer
         * allocated here refer to the same address space. */
        r = cuDevicePrimaryCtxRetain(&d->ctx, d->dev);
        if (r != CUDA_SUCCESS) { set_err("cuDevicePrimaryCtxRetain", r); return NULL; }
        r = cuCtxSetCurrent(d->ctx);
        if (r != CUDA_SUCCESS) { set_err("cuCtxSetCurrent(init)", r); return NULL; }
        /* NON_BLOCKING so we never implicitly serialize against the legacy
         * default stream some other library in the process may be using. */
        r = cuStreamCreate(&d->stream, CU_STREAM_NON_BLOCKING);
        if (r != CUDA_SUCCESS) { set_err("cuStreamCreate", r); return NULL; }
        cuDeviceGetAttribute(&d->sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d->dev);
        cuDeviceGetAttribute(&d->coop_supported,
                             CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, d->dev);
        h.n_dev = i + 1;
    }
    return &h;
}

void plow_hsa_shutdown(plow_hsa* h) {
    if (!h) return;
    for (int i = 0; i < h->n_dev; i++) {
        plow_cu_dev_t* d = &h->dev[i];
        cuCtxSetCurrent(d->ctx);
        for (int m = 0; m < d->n_mod; m++) cuModuleUnload(d->mod[m]);
        if (d->stream) cuStreamDestroy(d->stream);
        cuDevicePrimaryCtxRelease(d->dev);
    }
    h->n_dev = 0;
}

int plow_hsa_device_count(const plow_hsa* h) { return h ? h->n_dev : 0; }

int plow_hsa_device_info(const plow_hsa* h, int dev, char name[64],
                         uint32_t* cus, uint32_t* lds_bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    const plow_cu_dev_t* d = &h->dev[dev];
    if (name) {
        char raw[64];
        /* The HSA analogue reports the ISA name (gfx950). The closest true
         * analogue here is the compute capability, which is what -arch encodes,
         * so report "sm_XY (Product Name)". */
        int maj = 0, min = 0;
        cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d->dev);
        cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d->dev);
        raw[0] = 0;
        cuDeviceGetName(raw, (int)sizeof(raw), d->dev);
        snprintf(name, 64, "sm_%d%d (%s)", maj, min, raw);
    }
    if (cus) *cus = (uint32_t)d->sms;
    if (lds_bytes) {
        int v = 0;
        /* AMD reports LDS bytes per CU; the direct analogue is shared memory
         * per SM, not the 48 KB default per-block limit. */
        cuDeviceGetAttribute(&v,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, d->dev);
        *lds_bytes = (uint32_t)v;
    }
    return 0;
}

/* --- memory --------------------------------------------------------------- */

void* plow_hsa_alloc(plow_hsa* h, int dev, size_t bytes) {
    if (use_dev(h, dev) != 0) return NULL;
    CUdeviceptr p = 0;
    CUresult r = cuMemAlloc(&p, bytes);
    if (r != CUDA_SUCCESS) { set_err("cuMemAlloc", r); return NULL; }
    return (void*)(uintptr_t)p;
}

void plow_hsa_free(plow_hsa* h, void* p) {
    if (!p) return;
    /* Setting dev[0]'s context here is correct for BOTH free paths, even for a pointer
     * that was allocated on dev[1..]:
     *  - cuMemFree has been context-agnostic for device allocations since CUDA 4.0 (unified
     *    addressing): the driver resolves the owning context from the VA range, so freeing a
     *    dev[1] allocation while dev[0]'s context is current is well-defined, not a leak.
     *  - cuMemFreeHost / cuPointerGetAttribute operate on host allocations and the pointer's
     *    own attributes, neither of which is scoped to the current context.
     * A current context is nonetheless REQUIRED (the driver API faults without one), so we
     * pin dev[0] rather than depend on whatever the caller left current. */
    if (h && h->n_dev > 0) cuCtxSetCurrent(h->dev[0].ctx);
    /* One free path for both allocators, matching hsa_amd_memory_pool_free.
     * Ask the driver which kind of pointer this is rather than tracking it. */
    unsigned int type = 0;
    CUresult r = cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                       (CUdeviceptr)(uintptr_t)p);
    if (r == CUDA_SUCCESS && type == CU_MEMORYTYPE_HOST) cuMemFreeHost(p);
    else cuMemFree((CUdeviceptr)(uintptr_t)p);
}

void* plow_hsa_alloc_host(plow_hsa* h, size_t bytes) {
    if (use_dev(h, 0) != 0) return NULL;
    void* p = NULL;
    /* PORTABLE == the AMD `agents_allow_access(all agents)`: pinned once, visible
     * to every device's copy engine. DEVICEMAP additionally gives kernels a
     * device address for it (the host-visible counter / doorbell path). */
    CUresult r = cuMemHostAlloc(&p, bytes,
                                CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
    if (r != CUDA_SUCCESS) { set_err("cuMemHostAlloc", r); return NULL; }
    return p;
}

/* --- cross-GPU transport -------------------------------------------------- */

void* plow_hsa_alloc_peer(plow_hsa* h, int owner_dev, size_t bytes) {
    if (use_dev(h, owner_dev) != 0) return NULL;
    CUdeviceptr p = 0;
    CUresult r = cuMemAlloc(&p, bytes);
    if (r != CUDA_SUCCESS) { set_err("alloc_peer cuMemAlloc", r); return NULL; }
    /* Peer access is a property of the *context pair*, not the allocation, so
     * unlike hsa_amd_agents_allow_access this is idempotent and enabling it once
     * covers every future allocation. ALREADY_ENABLED is success. */
    for (int i = 0; i < h->n_dev; i++) {
        if (i == owner_dev) continue;
        if (cuCtxSetCurrent(h->dev[i].ctx) != CUDA_SUCCESS) continue;
        CUresult e = cuCtxEnablePeerAccess(h->dev[owner_dev].ctx, 0);
        if (e != CUDA_SUCCESS && e != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
            set_err("cuCtxEnablePeerAccess", e);
            cuCtxSetCurrent(h->dev[owner_dev].ctx);
            cuMemFree(p);
            return NULL;
        }
    }
    cuCtxSetCurrent(h->dev[owner_dev].ctx);
    return (void*)(uintptr_t)p;
}

int plow_hsa_copy_p2p(plow_hsa* h, int dst_dev, void* dst,
                      int src_dev, const void* src, size_t bytes) {
    if (!h || dst_dev < 0 || dst_dev >= h->n_dev ||
        src_dev < 0 || src_dev >= h->n_dev) return -1;
    if (use_dev(h, src_dev) != 0) return -1;
    TRY(cuMemcpyPeer((CUdeviceptr)(uintptr_t)dst, h->dev[dst_dev].ctx,
                     (CUdeviceptr)(uintptr_t)src, h->dev[src_dev].ctx, bytes),
        "cuMemcpyPeer");
    TRY(cuCtxSynchronize(), "cuCtxSynchronize(p2p)"); /* blocking, per contract */
    return 0;
}

/* --- host <-> device ------------------------------------------------------ */

/* cuMemcpyHtoD/DtoH are synchronous with respect to the host for pageable
 * memory and (because they are issued on the NULL stream) ordered against our
 * work for pinned memory, so bulk and control-plane copies collapse to the same
 * implementation. On AMD they differ because SDMA cannot touch unpinned pages;
 * the CUDA driver stages those itself. Keeping four names preserves the ABI and
 * documents intent at the call site.
 *
 * ORDERING CONTRACT (identical to the AMD backend, and easy to get wrong): these
 * copies are synchronous with respect to the HOST but are NOT ordered against an
 * in-flight `plow_hsa_launch`. Our launch stream is NON_BLOCKING, so it does not
 * implicitly synchronize with the legacy stream these copies use — exactly as
 * ROCr's SDMA copies carry their own signal and do not join the AQL queue. A
 * `plow_hsa_wait` MUST separate a launch from any copy that reads its output. */
int plow_hsa_copy_h2d(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (use_dev(h, dev) != 0) return -1;
    TRY(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, bytes), "cuMemcpyHtoD");
    return 0;
}

int plow_hsa_copy_d2h(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (use_dev(h, dev) != 0) return -1;
    TRY(cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, bytes), "cuMemcpyDtoH");
    return 0;
}

int plow_hsa_upload(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    return plow_hsa_copy_h2d(h, dev, dst, src, bytes);
}

int plow_hsa_download(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    return plow_hsa_copy_d2h(h, dev, dst, src, bytes);
}

/* --- code objects --------------------------------------------------------- */

int plow_hsa_load_code_object(plow_hsa* h, int dev, const void* elf, size_t bytes) {
    if (use_dev(h, dev) != 0) return -1;
    (void)bytes; /* cuModuleLoadData reads the cubin/fatbin header for its size */
    plow_cu_dev_t* d = &h->dev[dev];
    if (d->n_mod >= PLOW_CU_MAX_MOD) {
        snprintf(g_err, sizeof(g_err), "too many modules loaded on dev %d", dev);
        return -1;
    }
    TRY(cuModuleLoadData(&d->mod[d->n_mod], elf), "cuModuleLoadData");
    d->n_mod++;
    return 0;
}

int plow_hsa_get_kernel(plow_hsa* h, int dev, const char* name, plow_hsa_kernel* out) {
    if (use_dev(h, dev) != 0) return -1;
    if (!name || !out) return -1;
    plow_cu_dev_t* d = &h->dev[dev];

    CUfunction f = NULL;
    /* Search newest module first, mirroring "the last thing you loaded wins". */
    for (int m = d->n_mod - 1; m >= 0; m--)
        if (cuModuleGetFunction(&f, d->mod[m], name) == CUDA_SUCCESS) break;
    if (!f) {
        snprintf(g_err, sizeof(g_err), "kernel '%s' not found in %d module(s)",
                 name, d->n_mod);
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->kernel_object = (uint64_t)(uintptr_t)f;

    /* Total packed-parameter size, walked the same way the AMD path reads it out
     * of the kernel descriptor. cuFuncGetParamInfo returns INVALID_VALUE once we
     * step past the last parameter, which is the terminator. */
    size_t total = 0;
    for (size_t i = 0; i < 64; i++) {
        size_t off = 0, sz = 0;
        if (cuFuncGetParamInfo(f, i, &off, &sz) != CUDA_SUCCESS) break;
        if (off + sz > total) total = off + sz;
    }
    out->kernarg_size = (uint32_t)total;
    /* No COv5 hidden block on NVIDIA: the driver delivers gridDim/blockDim, so
     * the explicit args ARE the whole segment. */
    out->kernarg_explicit = (uint32_t)total;

    int v = 0;
    cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
    out->group_segment_size = (uint32_t)v;
    v = 0;
    cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    out->private_segment_size = (uint32_t)v;

    /* Opt this function into the full per-SM shared memory budget once, at load,
     * so a later launch may pass a `dynamic_lds` above the 48 KB default without
     * the caller having to know about the opt-in. Best-effort: functions with
     * large static shared memory legitimately reject part of the range. */
    int maxopt = 0;
    cuDeviceGetAttribute(&maxopt,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, d->dev);
    if (maxopt > (int)out->group_segment_size)
        cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           maxopt - (int)out->group_segment_size);
    return 0;
}

/* --- dispatch ------------------------------------------------------------- */

/* Can `blocks` blocks of `block` threads with `dyn` bytes of dynamic shared
 * memory all be resident at once? That is the co-residency condition the
 * counter-gated interpreter depends on (interp_sm120_poc.cu:176-179), and it is
 * exactly what cooperative launch enforces. Cached, because the answer is a
 * property of (function, geometry) and the dispatch path is latency-critical. */
static int coop_capacity(plow_cu_dev_t* d, CUfunction f, uint32_t block, uint32_t dyn) {
    uint32_t k = ((uint32_t)(uintptr_t)f >> 4) ^ block ^ (dyn * 2654435761u);
    coop_ent_t* e = &d->coop[k % PLOW_CU_COOP_CACHE];
    if (e->valid && e->f == f && e->block == block && e->dyn == dyn) return e->coop;
    int bps = 0;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, f, (int)block, dyn)
        != CUDA_SUCCESS) bps = 0;
    e->f = f; e->block = block; e->dyn = dyn; e->valid = 1;
    e->coop = bps * d->sms;
    return e->coop;
}

int plow_hsa_launch(plow_hsa* h, int dev, const plow_hsa_kernel* k,
                    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                    uint16_t wg_x, uint16_t wg_y, uint16_t wg_z,
                    uint32_t dynamic_lds,
                    const void* args, size_t args_size) {
    if (use_dev(h, dev) != 0) return -1;
    if (!k || !wg_x || !wg_y || !wg_z) {
        snprintf(g_err, sizeof(g_err), "launch: null kernel or zero workgroup dim");
        return -1;
    }
    if (args_size > k->kernarg_size) {
        snprintf(g_err, sizeof(g_err), "explicit args %zu B > kernarg segment %u B",
                 (size_t)args_size, k->kernarg_size);
        return -1;
    }
    plow_cu_dev_t* d = &h->dev[dev];
    CUfunction f = (CUfunction)(uintptr_t)k->kernel_object;

    /* HSA grid dims count WORK-ITEMS; CUDA grid dims count BLOCKS. */
    const unsigned bx = (grid_x + wg_x - 1u) / wg_x;
    const unsigned by = (grid_y + wg_y - 1u) / wg_y;
    const unsigned bz = (grid_z + wg_z - 1u) / wg_z;
    const uint32_t threads = (uint32_t)wg_x * wg_y * wg_z;
    const unsigned blocks = bx * by * bz;

    /* The kernarg blob, byte-for-byte as the AMD path lays it out. `extra` is
     * the driver API's packed-parameter form; it is the only one that accepts a
     * pre-marshalled buffer, and it is why this backend does not need to know
     * the argument *types* of the kernels it launches. */
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, (void*)(uintptr_t)args,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    (void*)&args_size,
        CU_LAUNCH_PARAM_END
    };

    CUlaunchConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDimX = bx; cfg.gridDimY = by; cfg.gridDimZ = bz;
    cfg.blockDimX = wg_x; cfg.blockDimY = wg_y; cfg.blockDimZ = wg_z;
    cfg.sharedMemBytes = dynamic_lds;
    cfg.hStream = d->stream;

    CUlaunchAttribute attr;
    memset(&attr, 0, sizeof(attr));
    if (d->coop_supported && blocks > 0 &&
        blocks <= (unsigned)coop_capacity(d, f, threads, dynamic_lds)) {
        attr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attr.value.cooperative = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
    }

    TRY(cuLaunchKernelEx(&cfg, f, NULL, extra), "cuLaunchKernelEx");
    return 0;
}

int plow_hsa_wait(plow_hsa* h, int dev) {
    if (use_dev(h, dev) != 0) return -1;
    TRY(cuStreamSynchronize(h->dev[dev].stream), "cuStreamSynchronize");
    return 0;
}
