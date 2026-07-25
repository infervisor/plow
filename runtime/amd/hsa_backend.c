/* hsa_backend.c — AMD device layer on ROCr/HSA. See hsa_backend.h. */
#include "hsa_backend.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PLOW_HSA_MAX_DEV 16
/* AQL ring depth. Also bounds in-flight dispatches, which is what makes the
 * kernarg ring below safe to reuse without tracking completion per slot. */
#define PLOW_HSA_QUEUE_SIZE 1024
/* Per-dispatch kernarg slot. Our widest kernel takes ~10 pointers + scalars,
 * and COv5 always appends a 256-byte hidden block. */
#define PLOW_HSA_KARG_SLOT 512
/* COv5 implicit kernarg block is a fixed 256 bytes at the tail of the segment. */
#define PLOW_HSA_HIDDEN_BYTES 256

static char g_err[256];
static void set_err(const char* what, hsa_status_t s) {
    const char* m = NULL;
    hsa_status_string(s, &m);
    snprintf(g_err, sizeof(g_err), "%s: %s", what, m ? m : "unknown");
}
const char* plow_hsa_last_error(void) { return g_err; }

#define TRY(call, what) do { hsa_status_t s_ = (call); \
    if (s_ != HSA_STATUS_SUCCESS) { set_err(what, s_); return -1; } } while (0)

typedef struct {
    hsa_agent_t          agent;
    hsa_amd_memory_pool_t vram;
    hsa_queue_t*         queue;
    hsa_signal_t         done;     /* counting: +1 per dispatch, -1 on completion */
    hsa_executable_t     exe;
    int                  has_exe;
    uint8_t*             karg_ring; /* PLOW_HSA_QUEUE_SIZE * PLOW_HSA_KARG_SLOT */
} plow_dev_t;

struct plow_hsa {
    hsa_agent_t           cpu;
    hsa_amd_memory_pool_t fine;    /* pinned, agent-visible system memory */
    hsa_amd_memory_pool_t kernarg;
    plow_dev_t                 dev[PLOW_HSA_MAX_DEV];
    int                   n_dev;
};

/* --- discovery ------------------------------------------------------------ */

static hsa_status_t on_agent(hsa_agent_t a, void* data) {
    struct plow_hsa* h = (struct plow_hsa*)data;
    hsa_device_type_t t;
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (t == HSA_DEVICE_TYPE_GPU) {
        if (h->n_dev < PLOW_HSA_MAX_DEV) h->dev[h->n_dev++].agent = a;
    } else if (t == HSA_DEVICE_TYPE_CPU && h->cpu.handle == 0) {
        h->cpu = a;
    }
    return HSA_STATUS_SUCCESS;
}

/* Pool predicates. `want` is the HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_* to match. */
typedef struct { hsa_amd_memory_pool_t pool; uint32_t want; int found; } pool_pick;

static hsa_status_t on_pool(hsa_amd_memory_pool_t p, void* data) {
    pool_pick* pk = (pool_pick*)data;
    if (pk->found) return HSA_STATUS_SUCCESS;
    hsa_amd_segment_t seg;
    if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg)
        != HSA_STATUS_SUCCESS || seg != HSA_AMD_SEGMENT_GLOBAL)
        return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags)
        != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (flags & pk->want) { pk->pool = p; pk->found = 1; }
    return HSA_STATUS_SUCCESS;
}

static int pick_pool(hsa_agent_t a, uint32_t want, hsa_amd_memory_pool_t* out) {
    pool_pick pk = {.want = want, .found = 0};
    if (hsa_amd_agent_iterate_memory_pools(a, on_pool, &pk) != HSA_STATUS_SUCCESS || !pk.found)
        return -1;
    *out = pk.pool;
    return 0;
}

plow_hsa* plow_hsa_init(void) {
    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { set_err("hsa_init", s); return NULL; }

    struct plow_hsa* h = calloc(1, sizeof(*h));
    if (!h) { snprintf(g_err, sizeof(g_err), "oom"); return NULL; }

    if (hsa_iterate_agents(on_agent, h) != HSA_STATUS_SUCCESS || h->n_dev == 0) {
        snprintf(g_err, sizeof(g_err), "no GPU agents (is the user in the 'render' group?)");
        free(h);
        return NULL;
    }
    if (pick_pool(h->cpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED, &h->fine) != 0 ||
        pick_pool(h->cpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, &h->kernarg) != 0) {
        snprintf(g_err, sizeof(g_err), "no fine-grained / kernarg system pool");
        free(h);
        return NULL;
    }

    for (int i = 0; i < h->n_dev; i++) {
        plow_dev_t* d = &h->dev[i];
        if (pick_pool(d->agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, &d->vram) != 0) {
            snprintf(g_err, sizeof(g_err), "device %d: no coarse-grained VRAM pool", i);
            free(h);
            return NULL;
        }
        /* UINT32_MAX = "configure for the largest possible private/group segment".
         * Naming a concrete bound here makes hsa_queue_create fail outright. */
        if (hsa_queue_create(d->agent, PLOW_HSA_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                             UINT32_MAX, UINT32_MAX, &d->queue)
            != HSA_STATUS_SUCCESS) {
            snprintf(g_err, sizeof(g_err), "device %d: hsa_queue_create", i);
            free(h);
            return NULL;
        }
        /* Counting signal: each dispatch adds 1, the packet processor subtracts
         * 1 on completion, so "queue drained" is simply value == 0. */
        if (hsa_signal_create(0, 0, NULL, &d->done) != HSA_STATUS_SUCCESS) {
            snprintf(g_err, sizeof(g_err), "device %d: hsa_signal_create", i);
            free(h);
            return NULL;
        }
        size_t ring = (size_t)PLOW_HSA_QUEUE_SIZE * PLOW_HSA_KARG_SLOT;
        if (hsa_amd_memory_pool_allocate(h->kernarg, ring, 0, (void**)&d->karg_ring)
            != HSA_STATUS_SUCCESS) {
            snprintf(g_err, sizeof(g_err), "device %d: kernarg ring alloc", i);
            free(h);
            return NULL;
        }
        hsa_amd_agents_allow_access(1, &d->agent, NULL, d->karg_ring);
    }
    return h;
}

void plow_hsa_shutdown(plow_hsa* h) {
    if (!h) return;
    for (int i = 0; i < h->n_dev; i++) {
        plow_dev_t* d = &h->dev[i];
        if (d->has_exe) hsa_executable_destroy(d->exe);
        if (d->karg_ring) hsa_amd_memory_pool_free(d->karg_ring);
        if (d->queue) hsa_queue_destroy(d->queue);
        hsa_signal_destroy(d->done);
    }
    free(h);
    hsa_shut_down();
}

int plow_hsa_device_count(const plow_hsa* h) { return h ? h->n_dev : 0; }

/* LDS is not an agent attribute — it is the size of the agent's GROUP region. */
static hsa_status_t on_region(hsa_region_t r, void* data) {
    hsa_region_segment_t seg;
    if (hsa_region_get_info(r, HSA_REGION_INFO_SEGMENT, &seg) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (seg != HSA_REGION_SEGMENT_GROUP) return HSA_STATUS_SUCCESS;
    size_t sz = 0;
    if (hsa_region_get_info(r, HSA_REGION_INFO_SIZE, &sz) == HSA_STATUS_SUCCESS)
        *(uint32_t*)data = (uint32_t)sz;
    return HSA_STATUS_SUCCESS;
}

int plow_hsa_device_info(const plow_hsa* h, int dev, char name[64],
                         uint32_t* cus, uint32_t* lds_bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    hsa_agent_t a = h->dev[dev].agent;
    TRY(hsa_agent_get_info(a, HSA_AGENT_INFO_NAME, name), "agent name");
    TRY(hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, cus),
        "agent CU count");
    *lds_bytes = 0;
    TRY(hsa_agent_iterate_regions(a, on_region, lds_bytes), "agent LDS region");
    return 0;
}

/* --- memory --------------------------------------------------------------- */

void* plow_hsa_alloc(plow_hsa* h, int dev, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return NULL;
    void* p = NULL;
    hsa_status_t s = hsa_amd_memory_pool_allocate(h->dev[dev].vram, bytes, 0, &p);
    if (s != HSA_STATUS_SUCCESS) { set_err("hsa_amd_memory_pool_allocate", s); return NULL; }
    return p;
}

void plow_hsa_free(plow_hsa* h, void* p) {
    (void)h;
    if (p) hsa_amd_memory_pool_free(p);
}

/* --- cross-GPU transport -------------------------------------------------- */

static int copy_blocking(plow_hsa* h, int dev, void* dst, hsa_agent_t dst_a,
                         const void* src, hsa_agent_t src_a, size_t bytes);

void* plow_hsa_alloc_peer(plow_hsa* h, int owner_dev, size_t bytes) {
    if (!h || owner_dev < 0 || owner_dev >= h->n_dev) return NULL;
    void* p = NULL;
    hsa_status_t s = hsa_amd_memory_pool_allocate(h->dev[owner_dev].vram, bytes, 0, &p);
    if (s != HSA_STATUS_SUCCESS) { set_err("alloc_peer pool allocate", s); return NULL; }

    /* Map the owner's VRAM into every GPU's address space so a kernel on any
     * device can load/store it over XGMI. This REPLACES the allowed-agent list,
     * so it must name all GPU agents at once (same footgun as alloc_host). The
     * owner is included: coarse-grained VRAM is not self-accessible until it is
     * on the list. This is what turns a system-scope atomic on `p` into a
     * cross-GPU handshake. */
    hsa_agent_t agents[PLOW_HSA_MAX_DEV];
    for (int i = 0; i < h->n_dev; i++) agents[i] = h->dev[i].agent;
    s = hsa_amd_agents_allow_access((uint32_t)h->n_dev, agents, NULL, p);
    if (s != HSA_STATUS_SUCCESS) {
        hsa_amd_memory_pool_free(p);
        set_err("agents_allow_access(peer)", s);
        return NULL;
    }
    return p;
}

int plow_hsa_copy_p2p(plow_hsa* h, int dst_dev, void* dst,
                      int src_dev, const void* src, size_t bytes) {
    if (!h || dst_dev < 0 || dst_dev >= h->n_dev ||
        src_dev < 0 || src_dev >= h->n_dev) return -1;
    /* SDMA D2D: the copy engine walks XGMI directly, no host bounce. The two
     * agents name the source and destination of the transfer. */
    return copy_blocking(h, dst_dev, dst, h->dev[dst_dev].agent,
                         src, h->dev[src_dev].agent, bytes);
}

void* plow_hsa_alloc_host(plow_hsa* h, size_t bytes) {
    if (!h) return NULL;
    void* p = NULL;
    hsa_status_t s = hsa_amd_memory_pool_allocate(h->fine, bytes, 0, &p);
    if (s != HSA_STATUS_SUCCESS) { set_err("host pool allocate", s); return NULL; }

    /* Every GPU must be able to read the staging buffer (all 8 replicas load
     * their weights from it). This call REPLACES the allowed-agent list, so it
     * has to name every agent at once — looping one-at-a-time silently leaves
     * only the last GPU with access, and the others' SDMA engines fault. */
    hsa_agent_t agents[PLOW_HSA_MAX_DEV];
    for (int i = 0; i < h->n_dev; i++) agents[i] = h->dev[i].agent;
    s = hsa_amd_agents_allow_access((uint32_t)h->n_dev, agents, NULL, p);
    if (s != HSA_STATUS_SUCCESS) {
        hsa_amd_memory_pool_free(p);
        set_err("agents_allow_access(host staging)", s);
        return NULL;
    }
    return p;
}

static int copy_blocking(plow_hsa* h, int dev, void* dst, hsa_agent_t dst_a,
                         const void* src, hsa_agent_t src_a, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    hsa_signal_t sig;
    TRY(hsa_signal_create(1, 0, NULL, &sig), "copy signal");
    hsa_status_t s = hsa_amd_memory_async_copy(dst, dst_a, src, src_a, bytes, 0, NULL, sig);
    if (s != HSA_STATUS_SUCCESS) {
        hsa_signal_destroy(sig);
        set_err("hsa_amd_memory_async_copy", s);
        return -1;
    }
    while (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0) {}
    hsa_signal_destroy(sig);
    return 0;
}

int plow_hsa_copy_h2d(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    return copy_blocking(h, dev, dst, h->dev[dev].agent, src, h->cpu, bytes);
}

int plow_hsa_copy_d2h(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    return copy_blocking(h, dev, dst, h->cpu, src, h->dev[dev].agent, bytes);
}

int plow_hsa_upload(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    /* Pin the caller's pages and get the address the agent should read them
     * through. Without this, an SDMA read of a stack/malloc pointer faults the
     * GPU — and the fault surfaces as an opaque "memory access fault", not as an
     * error from the copy. */
    void* pinned = NULL;
    hsa_agent_t agent = h->dev[dev].agent;
    TRY(hsa_amd_memory_lock((void*)src, bytes, &agent, 1, &pinned), "hsa_amd_memory_lock");
    const int rc = copy_blocking(h, dev, dst, agent, pinned, h->cpu, bytes);
    hsa_amd_memory_unlock((void*)src);
    return rc;
}

int plow_hsa_download(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    void* pinned = NULL;
    hsa_agent_t agent = h->dev[dev].agent;
    TRY(hsa_amd_memory_lock(dst, bytes, &agent, 1, &pinned), "hsa_amd_memory_lock");
    const int rc = copy_blocking(h, dev, pinned, h->cpu, src, agent, bytes);
    hsa_amd_memory_unlock(dst);
    return rc;
}

/* --- code objects --------------------------------------------------------- */

int plow_hsa_load_code_object(plow_hsa* h, int dev, const void* elf, size_t bytes) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    plow_dev_t* d = &h->dev[dev];
    hsa_code_object_reader_t rdr;
    TRY(hsa_code_object_reader_create_from_memory(elf, bytes, &rdr), "code_object_reader");
    TRY(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  NULL, &d->exe),
        "executable_create");
    TRY(hsa_executable_load_agent_code_object(d->exe, d->agent, rdr, NULL, NULL),
        "load_agent_code_object (raw ELF expected — did you unbundle?)");
    TRY(hsa_executable_freeze(d->exe, NULL), "executable_freeze");
    hsa_code_object_reader_destroy(rdr);
    d->has_exe = 1;
    return 0;
}

int plow_hsa_get_kernel(plow_hsa* h, int dev, const char* name, plow_hsa_kernel* out) {
    if (!h || dev < 0 || dev >= h->n_dev || !h->dev[dev].has_exe) return -1;
    plow_dev_t* d = &h->dev[dev];

    /* The loader exposes kernels under their descriptor symbol, "<name>.kd". */
    char sym_name[128];
    snprintf(sym_name, sizeof(sym_name), "%s.kd", name);

    hsa_executable_symbol_t sym;
    TRY(hsa_executable_get_symbol_by_name(d->exe, sym_name, &d->agent, &sym), name);
    TRY(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &out->kernel_object), "kernel_object");
    TRY(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                       &out->kernarg_size), "kernarg_size");
    TRY(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                       &out->group_segment_size), "group_segment_size");
    TRY(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                       &out->private_segment_size), "private_segment_size");

    if (out->kernarg_size > PLOW_HSA_KARG_SLOT) {
        snprintf(g_err, sizeof(g_err), "%s: kernarg %u B exceeds slot %d", name,
                 out->kernarg_size, PLOW_HSA_KARG_SLOT);
        return -1;
    }
    /* The size of the COv5 implicit block is NOT fixed: the compiler emits only
     * the implicit args a kernel actually references and truncates the tail. A
     * kernel that never reads blockDim (e.g. one that strides by a literal) gets
     * no implicit block at all. Field offsets within the block ARE fixed, so we
     * resolve the block's start from the caller's explicit size at launch and
     * write only the fields that fit. `kernarg_explicit` is unused here. */
    out->kernarg_explicit = 0;
    return 0;
}

/* --- dispatch ------------------------------------------------------------- */

int plow_hsa_launch(plow_hsa* h, int dev, const plow_hsa_kernel* k,
                    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                    uint16_t wg_x, uint16_t wg_y, uint16_t wg_z,
                    uint32_t dynamic_lds,
                    const void* args, size_t args_size) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    if (args_size > k->kernarg_size) {
        snprintf(g_err, sizeof(g_err), "explicit args %zu B > kernarg segment %u B",
                 args_size, k->kernarg_size);
        return -1;
    }
    plow_dev_t* d = &h->dev[dev];
    hsa_queue_t* q = d->queue;

    uint64_t idx = hsa_queue_add_write_index_screlease(q, 1);
    /* Ring is full only if we have QUEUE_SIZE packets outstanding; spin until the
     * packet processor retires one. This also bounds kernarg-ring reuse. */
    while (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {}

    const uint32_t slot = (uint32_t)(idx & (q->size - 1));
    uint8_t* karg = d->karg_ring + (size_t)slot * PLOW_HSA_KARG_SLOT;

    memcpy(karg, args, args_size);
    memset(karg + args_size, 0, k->kernarg_size - args_size);

    /* COv5 implicit block. The compiler reads blockDim/gridDim from HERE, not from
     * the AQL packet — leaving them zero makes blockDim 0, so every workgroup
     * recomputes tile 0 and the output is silently, plausibly wrong.
     *
     * The block starts right after the (8-aligned) explicit args, but its LENGTH
     * varies: the compiler emits only the implicit args the kernel references and
     * truncates the rest. So write each field only if the segment is long enough
     * to hold it. Field offsets within the block are fixed by the ABI. */
    const size_t hoff = (args_size + 7u) & ~(size_t)7u;
    if (k->kernarg_size > hoff) {
        uint8_t* hid = karg + hoff;
        const size_t avail = k->kernarg_size - hoff;
        const uint16_t dims3 = (uint16_t)(grid_z > 1 ? 3 : (grid_y > 1 ? 2 : 1));
#define PUT32(off, val) if (avail >= (off) + 4) *(uint32_t*)(hid + (off)) = (uint32_t)(val)
#define PUT16(off, val) if (avail >= (off) + 2) *(uint16_t*)(hid + (off)) = (uint16_t)(val)
        PUT32(0,  (grid_x + wg_x - 1) / wg_x); /* hidden_block_count_x */
        PUT32(4,  (grid_y + wg_y - 1) / wg_y);
        PUT32(8,  (grid_z + wg_z - 1) / wg_z);
        PUT16(12, wg_x);                       /* hidden_group_size_x   */
        PUT16(14, wg_y);
        PUT16(16, wg_z);
        PUT16(18, grid_x % wg_x);              /* hidden_remainder_x    */
        PUT16(20, grid_y % wg_y);
        PUT16(22, grid_z % wg_z);
        PUT16(64, dims3);                      /* hidden_grid_dims      */
#undef PUT32
#undef PUT16
    }

    hsa_kernel_dispatch_packet_t* p =
        (hsa_kernel_dispatch_packet_t*)q->base_address + slot;

    /* Everything except the 4-byte header|setup, which must be published last. */
    memset((uint8_t*)p + 4, 0, sizeof(*p) - 4);
    p->workgroup_size_x     = wg_x;
    p->workgroup_size_y     = wg_y;
    p->workgroup_size_z     = wg_z;
    p->grid_size_x          = grid_x;
    p->grid_size_y          = grid_y;
    p->grid_size_z          = grid_z;
    p->kernel_object        = k->kernel_object;
    p->kernarg_address      = karg;
    p->group_segment_size   = k->group_segment_size + dynamic_lds;
    p->private_segment_size = k->private_segment_size;
    p->completion_signal    = d->done;

    hsa_signal_add_screlease(d->done, 1);

    const uint16_t dims = (uint16_t)(grid_z > 1 ? 3 : (grid_y > 1 ? 2 : 1));
    uint16_t header = (uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                    | (1 << HSA_PACKET_HEADER_BARRIER)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    uint16_t setup = (uint16_t)(dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);

    /* One release store publishes the packet: the type field is what makes the
     * packet processor pick it up, so it has to land after the payload. */
    __atomic_store_n((uint32_t*)p, ((uint32_t)setup << 16) | header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
    return 0;
}

int plow_hsa_wait(plow_hsa* h, int dev) {
    if (!h || dev < 0 || dev >= h->n_dev) return -1;
    hsa_signal_t sig = h->dev[dev].done;
    while (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0) {}
    return 0;
}
