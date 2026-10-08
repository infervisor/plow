/*
 * pseudo_lock_sram.c - Intel RDT (CAT) cache pseudo-locking service for plowrt's CPU engine.
 *
 * The driver partitions the cache once at load (module params l2_ways / l3_ways) and then
 * locks memory the CALLER owns: plowrt allocates node-bound, huge-page-backed buffers and asks
 * /dev/pseudo_lock to load a range into
 *   level 2: the private L2 of the core `cpu` (that core's locked ways), or
 *   level 3: the L3 slices of `cpu`'s NUMA node (SNC domain), through its locked ways.
 * The driver pins the pages, checks node locality and the per-core / per-node way budget,
 * fills the lines from `cpu` in the lock CLOS with prefetchers off, and measures the result
 * from `cpu` against calibrated L2 / L3 / DRAM latencies. Locks die with the file descriptor.
 *
 * Why the earlier (v2, fixed-region) driver measured ~98% DRAM:
 *  - CLOS_L3_LOCK carried a full L2 mask, so the L3 preload on CPU 0 evicted CPU 0's locked L2.
 *  - The L3 region was one vmalloc on whatever node insmod ran on. Under SNC a line is cached
 *    only in its home die's slices, so the region exceeded that die's locked ways, and remote
 *    lines read from CPU 0 were slower than the fixed 150-cycle "L3" threshold.
 *  - L3 is non-inclusive (lines enter L3 as L2 victims); lines still in L2 when the lock CLOS
 *    was dropped were later evicted into the normal ways. CLDEMOTE pushes them while locked.
 * Masks come from CPUID leaf 0x10; locked L3 ways never overlap the ways shared with I/O.
 * Core C6 flushes L2, so a CPU-latency QoS request keeps cores out of deep idle while locks exist.
 *
 * The partition is LAZY: loading the module changes nothing. The CAT masks and the QoS request
 * are applied when the first region or worker cpu appears and fully restored when the last one
 * goes, so a loaded but idle driver leaves every other process the whole cache.
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/smp.h>
#include <linux/cpumask.h>
#include <linux/topology.h>
#include <linux/nodemask.h>
#include <linux/device.h>
#include <linux/sysfs.h>
#include <linux/pm_qos.h>
#include <linux/vmalloc.h>
#include <linux/list.h>
#include <asm/msr.h>
#include <asm/processor.h>
#include <asm/special_insns.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Lava Bokam <lavajnv@gmail.com>");
MODULE_DESCRIPTION("Intel RDT L2/L3 cache pseudo-locking of caller-owned memory (per core / per SNC node)");
MODULE_VERSION("3.0");

#define MSR_MISC_FEATURE_CONTROL    0x000001a4
#ifndef MSR_IA32_PQR_ASSOC
#define MSR_IA32_PQR_ASSOC          0x00000c8f
#endif
#define MSR_IA32_L3_QOS_MASK_BASE   0x00000c90
#define MSR_IA32_L2_QOS_MASK_BASE   0x00000d10
#define PREFETCH_DISABLE_ALL        0x0000000f

#define CLOS_DEFAULT                0
#define CLOS_L2_LOCK                1
#define CLOS_L3_LOCK                2
#define CLOS_WORKER                 3   /* readers of L3-locked data: fill the locked ways */

#define LINE                        64
#define CAL_LINES                   512
#define HIST_MAX                    4096
#define PL_VERSION                  3

/* ABI shared with plowrt (crates/plowrt/src/memory/sram.rs). */
struct pl_caps {
    uint32_t version;
    uint32_t nodes;
    uint32_t l2_cbm_full, l2_cbm_lock;
    uint32_t l3_cbm_full, l3_cbm_lock;
    uint64_t l2_lock_bytes_per_core;    /* capacity of the locked L2 ways of one core */
    uint64_t l3_lock_bytes_per_node;    /* capacity of the locked L3 ways in one SNC domain */
};

struct pl_lock_req {
    uint64_t addr;                      /* page aligned, caller-owned, populated or not */
    uint64_t len;                       /* page multiple */
    int32_t cpu;                        /* owner: the core (L2) or any cpu of the node (L3) */
    uint32_t level;                     /* 2 or 3 */
    uint32_t id;                        /* out */
    uint32_t pad;
};

struct pl_measure {
    uint32_t id;                        /* in */
    uint32_t pad;
    uint64_t lines, l1_l2, l3, dram;    /* out: lines by calibrated latency class */
    uint64_t p50, cal_l2, cal_l3, cal_dram; /* out: cycles */
};

#define PL_IOC_MAGIC    'P'
#define PL_IOC_CAPS     _IOR(PL_IOC_MAGIC, 10, struct pl_caps)
#define PL_IOC_LOCK     _IOWR(PL_IOC_MAGIC, 11, struct pl_lock_req)
#define PL_IOC_UNLOCK   _IOW(PL_IOC_MAGIC, 12, uint32_t)
#define PL_IOC_MEASURE  _IOWR(PL_IOC_MAGIC, 13, struct pl_measure)
#define PL_IOC_RELOAD   _IOW(PL_IOC_MAGIC, 14, uint32_t)
/* Run `cpu` in CLOS_WORKER (on = 1) or the default CLOS (on = 0) until this fd closes. */
struct pl_clos_req { int32_t cpu; uint32_t on; };
#define PL_IOC_WORKER   _IOW(PL_IOC_MAGIC, 15, struct pl_clos_req)

static int l2_ways = 14;
module_param(l2_ways, int, 0444);
MODULE_PARM_DESC(l2_ways, "L2 ways reserved for locking on every core (0..cbm-1, default 14 of 16)");

static int l3_ways = 14;
module_param(l3_ways, int, 0444);
MODULE_PARM_DESC(l3_ways, "L3 ways reserved for locking, outside the I/O-shared ways (default 14 of 16)");

static int l3_io_ways = 0;
module_param(l3_io_ways, int, 0444);
MODULE_PARM_DESC(l3_io_ways, "Allow locked L3 ways to overlap the I/O-shared ways (e.g. l3_ways=15)");

struct region {
    struct list_head node;              /* in the owning file's list */
    struct list_head all;               /* in g_regions (sysfs) */
    u32 id;
    int level, cpu, nid;
    unsigned long npages;
    struct page **pages;
    void *kva;
    size_t len;
    struct pl_measure m;
};

struct pl_file {
    struct list_head regions;
    cpumask_t workers;                  /* cpus this fd moved to CLOS_WORKER */
};

static DEFINE_MUTEX(g_lock);
static unsigned int g_users;            /* live regions + worker cpus, all files; g_lock */
static LIST_HEAD(g_regions);
static u32 g_next_id = 1;
static u32 l2_full, l3_full, l2_lock, l3_lock, l2_norm, l3_norm;
static size_t l2_cap, l3_cap;
static size_t *l2_used;                 /* per core (first SMT sibling) */
static size_t l3_used[MAX_NUMNODES];
static void *cal_buf[MAX_NUMNODES];
static struct pm_qos_request qos_req;

/* ------------------------------------------------------------------ CAT */

static void set_masks(void *unused)
{
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_DEFAULT, l3_norm, 0);
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_L2_LOCK, l3_norm, 0);
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_L3_LOCK, l3_lock ? l3_lock : l3_full, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_DEFAULT, l2_norm, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_L2_LOCK, l2_lock ? l2_lock : l2_full, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_L3_LOCK, l2_norm, 0);
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_WORKER, l3_lock ? l3_lock : l3_full, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_WORKER, l2_norm, 0);
}

static void set_clos_fn(void *info)
{
    u32 clos = (u32)(uintptr_t)info;
    u64 pqr = native_rdmsrq(MSR_IA32_PQR_ASSOC);
    native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)pqr, clos);
}

static void restore_masks(void *unused)
{
    int c;
    for (c = CLOS_DEFAULT; c <= CLOS_WORKER; c++) {
        native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + c, l3_full, 0);
        native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + c, l2_full, 0);
    }
}

static size_t cache_bytes(int idx)
{
    unsigned int a, b, c, d;
    cpuid_count(4, idx, &a, &b, &c, &d);
    return (size_t)(((b >> 22) & 0x3ff) + 1) * (((b >> 12) & 0x3ff) + 1) * ((b & 0xfff) + 1) * (c + 1);
}

static int cat_probe(void)
{
    unsigned int a, b, c, d, shared;

    if (boot_cpu_data.x86_vendor != X86_VENDOR_INTEL || !boot_cpu_has(X86_FEATURE_RDT_A))
        return -ENODEV;
    cpuid_count(0x10, 0, &a, &b, &c, &d);
    if (!(b & BIT(1)) || !(b & BIT(2))) {
        pr_err("pseudo_lock: CPU lacks L3 and/or L2 CAT (cpuid 0x10 ebx=%#x)\n", b);
        return -ENODEV;
    }
    cpuid_count(0x10, 1, &a, &b, &c, &d);
    l3_full = (u32)((1ULL << ((a & 0x1f) + 1)) - 1);
    shared = b;
    cpuid_count(0x10, 2, &a, &b, &c, &d);
    l2_full = (u32)((1ULL << ((a & 0x1f) + 1)) - 1);
    if (l3_ways < 0 || l3_ways >= hweight32(l3_full) || (!l3_io_ways && ((BIT(l3_ways) - 1) & shared))) {
        pr_err("pseudo_lock: l3_ways=%d must leave a normal way and avoid I/O-shared ways %#x (l3_io_ways=1 overrides)\n",
               l3_ways, shared);
        return -EINVAL;
    }
    if ((BIT(l3_ways) - 1) & shared)
        pr_warn("pseudo_lock: locked L3 ways %#lx overlap I/O-shared ways %#x: DMA fills can evict locked lines\n",
                BIT(l3_ways) - 1, shared);
    if (l2_ways < 0 || l2_ways >= hweight32(l2_full)) {
        pr_err("pseudo_lock: l2_ways=%d must leave a normal way (cbm %#x)\n", l2_ways, l2_full);
        return -EINVAL;
    }
    l3_lock = l3_ways ? (u32)(BIT(l3_ways) - 1) : 0;
    l2_lock = l2_ways ? (u32)(BIT(l2_ways) - 1) : 0;
    l3_norm = l3_full & ~l3_lock;
    l2_norm = l2_full & ~l2_lock;
    /* Fill from a way's capacity leaves set-conflict headroom: 7/8 of the locked ways. */
    l2_cap = cache_bytes(2) / hweight32(l2_full) * l2_ways * 7 / 8;
    l3_cap = cache_bytes(3) / hweight32(l3_full) * l3_ways / num_online_nodes() * 7 / 8;
    pr_info("pseudo_lock: v%d L3 cbm %#x lock %#x (%zu KiB/node), L2 cbm %#x lock %#x (%zu KiB/core), %d node(s)\n",
            PL_VERSION, l3_full, l3_lock, l3_cap >> 10, l2_full, l2_lock, l2_cap >> 10, num_online_nodes());
    return 0;
}

/* g_lock held. The first user partitions the cache, the last one gives it back. */
static void users_get(void)
{
    if (g_users++ == 0) {
        cpu_latency_qos_add_request(&qos_req, 0);
        on_each_cpu(set_masks, NULL, 1);
    }
}

static void users_put(void)
{
    if (--g_users == 0) {
        on_each_cpu(restore_masks, NULL, 1);
        cpu_latency_qos_remove_request(&qos_req);
    }
}

/* ------------------------------------------------------------------ lock / measure */

static __always_inline void cldemote(const void *p)
{
    asm volatile(".byte 0x0f, 0x1c, 0x07" : : "D"(p) : "memory");  /* cldemote (%rdi) */
}

static __always_inline void touch(const void *p)
{
    asm volatile("mov (%0), %%eax" : : "r"(p) : "eax", "memory");
}

static void lock_fn(void *info)
{
    struct region *r = info;
    char *mem = r->kva;
    u64 saved_pf, saved_pqr;
    size_t i;
    int pass;

    for (i = 0; i < r->len; i += LINE)
        clflushopt(mem + i);
    mb();
    local_irq_disable();
    saved_pf = native_rdmsrq(MSR_MISC_FEATURE_CONTROL);
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, PREFETCH_DISABLE_ALL);
    saved_pqr = native_rdmsrq(MSR_IA32_PQR_ASSOC);
    native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)saved_pqr, r->level == 2 ? CLOS_L2_LOCK : CLOS_L3_LOCK);
    for (i = 0; i < r->len; i += PAGE_SIZE)
        touch(mem + i);
    for (pass = 0; pass < 2; pass++)
        for (i = 0; i < r->len; i += LINE)
            touch(mem + i);
    if (r->level == 3) {
        /* Push what is still in L2 into L3 while the lock CLOS owns the fill. */
        for (pass = 0; pass < 2; pass++) {
            for (i = 0; i < r->len; i += LINE)
                cldemote(mem + i);
            mb();
        }
    }
    native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)saved_pqr, (u32)(saved_pqr >> 32));
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, saved_pf);
    local_irq_enable();
}

struct measure_args {
    struct region *r;
    u16 *cal;
    u32 *hist;
};

static u64 lat_median(const char *p, u16 *buf)
{
    u32 hist[64] = { 0 };
    u64 seen = 0;
    int i;

    for (i = 0; i < CAL_LINES; i++) {
        u64 t0 = rdtsc_ordered();
        touch(p + (size_t)i * LINE);
        buf[i] = (u16)min_t(u64, rdtsc_ordered() - t0, 63 * 16);
    }
    for (i = 0; i < CAL_LINES; i++)
        hist[buf[i] / 16]++;
    for (i = 0; i < 64; i++) {
        seen += hist[i];
        if (seen * 2 >= CAL_LINES)
            return i * 16 + 8;
    }
    return 63 * 16;
}

static void measure_fn(void *info)
{
    struct measure_args *a = info;
    struct region *r = a->r;
    struct pl_measure *m = &r->m;
    char *cal = cal_buf[r->nid];
    const char *mem = r->kva;
    size_t lines = r->len / LINE, i;
    u64 saved_pf, saved_pqr = 0, thr_l2, thr_l3, seen = 0;
    int k;

    memset(a->hist, 0, HIST_MAX * sizeof(u32));
    m->l1_l2 = m->l3 = m->dram = 0;
    m->lines = lines;
    local_irq_disable();
    saved_pf = native_rdmsrq(MSR_MISC_FEATURE_CONTROL);
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, PREFETCH_DISABLE_ALL);
    /* Calibrate on this CPU with a node-local buffer: DRAM (flushed), L3 (demoted), L2 (hot). */
    for (k = 0; k < CAL_LINES; k++)
        clflushopt(cal + (size_t)k * LINE);
    mb();
    m->cal_dram = lat_median(cal, a->cal);
    for (k = 0; k < CAL_LINES; k++)
        cldemote(cal + (size_t)k * LINE);
    mb();
    m->cal_l3 = lat_median(cal, a->cal);
    m->cal_l2 = lat_median(cal, a->cal);
    thr_l2 = (m->cal_l2 + m->cal_l3) / 2;
    thr_l3 = (m->cal_l3 + m->cal_dram) / 2;
    /*
     * The L3 is non-inclusive: a read hit moves the line to this core's L2, and its later
     * eviction refills L3 through the evicting core's CLOS. Read L3-locked lines in the lock
     * CLOS and demote each one straight back, or measuring would unlock what it measures.
     */
    if (r->level == 3) {
        saved_pqr = native_rdmsrq(MSR_IA32_PQR_ASSOC);
        native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)saved_pqr, CLOS_L3_LOCK);
    }
    for (i = 0; i < lines; i++) {
        u64 t0 = rdtsc_ordered(), d;
        touch(mem + i * LINE);
        d = rdtsc_ordered() - t0;
        if (r->level == 3)
            cldemote(mem + i * LINE);
        a->hist[min_t(u64, d, HIST_MAX - 1)]++;
        if (d <= thr_l2)
            m->l1_l2++;
        else if (d <= thr_l3)
            m->l3++;
        else
            m->dram++;
    }
    if (r->level == 3) {
        mb();
        native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)saved_pqr, (u32)(saved_pqr >> 32));
    }
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, saved_pf);
    local_irq_enable();
    for (k = 0; k < HIST_MAX; k++) {
        seen += a->hist[k];
        if (seen * 2 >= lines) {
            m->p50 = k;
            break;
        }
    }
}

static int measure(struct region *r)
{
    struct measure_args a = { .r = r };

    a.cal = kmalloc_array(CAL_LINES, sizeof(u16), GFP_KERNEL);
    a.hist = kvmalloc_array(HIST_MAX, sizeof(u32), GFP_KERNEL);
    if (a.cal && a.hist)
        smp_call_function_single(r->cpu, measure_fn, &a, 1);
    kfree(a.cal);
    kvfree(a.hist);
    return a.cal && a.hist ? 0 : -ENOMEM;
}

static size_t *budget(struct region *r)
{
    return r->level == 2 ? &l2_used[cpumask_first(topology_sibling_cpumask(r->cpu))] : &l3_used[r->nid];
}

static void free_region(struct region *r)
{
    if (r->kva)
        vunmap(r->kva);
    if (r->pages)
        unpin_user_pages(r->pages, r->npages);
    kvfree(r->pages);
    kfree(r);
}

/* g_lock held. */
static void unlock_region(struct region *r)
{
    *budget(r) -= r->len;
    list_del(&r->node);
    list_del(&r->all);
    free_region(r);
    users_put();
}

static long do_lock(struct pl_file *pf, struct pl_lock_req __user *ureq)
{
    struct pl_lock_req req;
    struct region *r;
    unsigned long i;
    long pinned;
    int ret;

    if (copy_from_user(&req, ureq, sizeof(req)))
        return -EFAULT;
    if ((req.level != 2 && req.level != 3) || !req.len || !PAGE_ALIGNED(req.addr) || !PAGE_ALIGNED(req.len) ||
        req.cpu < 0 || req.cpu >= nr_cpu_ids || !cpu_online(req.cpu))
        return -EINVAL;
    if ((req.level == 2 && !l2_ways) || (req.level == 3 && !l3_ways))
        return -EOPNOTSUPP;
    r = kzalloc(sizeof(*r), GFP_KERNEL);
    if (!r)
        return -ENOMEM;
    r->level = req.level;
    r->cpu = req.cpu;
    r->nid = cpu_to_node(req.cpu);
    r->len = req.len;
    r->npages = req.len >> PAGE_SHIFT;
    r->pages = kvcalloc(r->npages, sizeof(struct page *), GFP_KERNEL);
    if (!r->pages) {
        kfree(r);
        return -ENOMEM;
    }
    pinned = pin_user_pages_fast(req.addr, r->npages, FOLL_WRITE | FOLL_LONGTERM, r->pages);
    if (pinned != r->npages) {
        if (pinned > 0)
            unpin_user_pages(r->pages, pinned);
        kvfree(r->pages);
        kfree(r);
        return pinned < 0 ? pinned : -EFAULT;
    }
    /* Under SNC a line is cached only in its home node's slices: L3 locking needs local pages. */
    for (i = 0; req.level == 3 && i < r->npages; i++) {
        if (page_to_nid(r->pages[i]) != r->nid) {
            pr_warn_ratelimited("pseudo_lock: L3 range page %lu on node %d, cpu %d is on node %d\n",
                                i, page_to_nid(r->pages[i]), r->cpu, r->nid);
            free_region(r);
            return -EXDEV;
        }
    }
    r->kva = vmap(r->pages, r->npages, VM_MAP, PAGE_KERNEL);
    if (!r->kva) {
        free_region(r);
        return -ENOMEM;
    }
    mutex_lock(&g_lock);
    if (*budget(r) + r->len > (req.level == 2 ? l2_cap : l3_cap)) {
        mutex_unlock(&g_lock);
        free_region(r);
        return -ENOSPC;
    }
    *budget(r) += r->len;
    users_get();
    r->id = g_next_id++;
    list_add(&r->node, &pf->regions);
    list_add(&r->all, &g_regions);
    smp_call_function_single(r->cpu, lock_fn, r, 1);
    ret = measure(r);
    req.id = r->id;
    mutex_unlock(&g_lock);
    if (ret)
        return ret;
    return copy_to_user(ureq, &req, sizeof(req)) ? -EFAULT : 0;
}

static struct region *find(struct pl_file *pf, u32 id)
{
    struct region *r;
    list_for_each_entry(r, &pf->regions, node)
        if (r->id == id)
            return r;
    return NULL;
}

/* ------------------------------------------------------------------ fops */

static int pl_open(struct inode *inode, struct file *filp)
{
    struct pl_file *pf = kzalloc(sizeof(*pf), GFP_KERNEL);
    if (!pf)
        return -ENOMEM;
    INIT_LIST_HEAD(&pf->regions);
    filp->private_data = pf;
    return 0;
}

static int pl_release(struct inode *inode, struct file *filp)
{
    struct pl_file *pf = filp->private_data;
    struct region *r, *t;

    int cpu;

    mutex_lock(&g_lock);
    for_each_cpu(cpu, &pf->workers) {
        smp_call_function_single(cpu, set_clos_fn, (void *)(uintptr_t)CLOS_DEFAULT, 1);
        users_put();
    }
    list_for_each_entry_safe(r, t, &pf->regions, node)
        unlock_region(r);
    mutex_unlock(&g_lock);
    kfree(pf);
    return 0;
}

static long pl_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct pl_file *pf = filp->private_data;
    struct region *r;
    u32 id;
    long ret = 0;

    switch (cmd) {
    case PL_IOC_CAPS: {
        struct pl_caps caps = {
            .version = PL_VERSION, .nodes = num_online_nodes(),
            .l2_cbm_full = l2_full, .l2_cbm_lock = l2_lock, .l3_cbm_full = l3_full, .l3_cbm_lock = l3_lock,
            .l2_lock_bytes_per_core = l2_cap, .l3_lock_bytes_per_node = l3_cap,
        };
        return copy_to_user((void __user *)arg, &caps, sizeof(caps)) ? -EFAULT : 0;
    }
    case PL_IOC_LOCK:
        return do_lock(pf, (struct pl_lock_req __user *)arg);
    case PL_IOC_WORKER: {
        struct pl_clos_req q;
        if (copy_from_user(&q, (void __user *)arg, sizeof(q)))
            return -EFAULT;
        if (q.cpu < 0 || q.cpu >= nr_cpu_ids || !cpu_online(q.cpu))
            return -EINVAL;
        mutex_lock(&g_lock);
        if (q.on && !cpumask_test_cpu(q.cpu, &pf->workers)) {
            users_get();
            cpumask_set_cpu(q.cpu, &pf->workers);
        }
        smp_call_function_single(q.cpu, set_clos_fn, (void *)(uintptr_t)(q.on ? CLOS_WORKER : CLOS_DEFAULT), 1);
        if (!q.on && cpumask_test_cpu(q.cpu, &pf->workers)) {
            cpumask_clear_cpu(q.cpu, &pf->workers);
            users_put();
        }
        mutex_unlock(&g_lock);
        return 0;
    }
    case PL_IOC_UNLOCK:
    case PL_IOC_RELOAD:
        if (get_user(id, (u32 __user *)arg))
            return -EFAULT;
        mutex_lock(&g_lock);
        r = find(pf, id);
        if (!r)
            ret = -ENOENT;
        else if (cmd == PL_IOC_UNLOCK)
            unlock_region(r);
        else {
            smp_call_function_single(r->cpu, lock_fn, r, 1);
            ret = measure(r);
        }
        mutex_unlock(&g_lock);
        return ret;
    case PL_IOC_MEASURE: {
        struct pl_measure m;
        if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
            return -EFAULT;
        mutex_lock(&g_lock);
        r = find(pf, m.id);
        if (!r)
            ret = -ENOENT;
        else if (!(ret = measure(r))) {
            m = r->m;
            m.id = r->id;
        }
        mutex_unlock(&g_lock);
        if (ret)
            return ret;
        return copy_to_user((void __user *)arg, &m, sizeof(m)) ? -EFAULT : 0;
    }
    default:
        return -ENOTTY;
    }
}

static const struct file_operations pl_fops = {
    .owner          = THIS_MODULE,
    .open           = pl_open,
    .release        = pl_release,
    .unlocked_ioctl = pl_ioctl,
};

/* "<id> L<level> cpu <cpu> node <nid> <bytes> | last: l1/l2 l3 dram of lines, p50 (cal L2 L3 DRAM)" */
static ssize_t regions_show(struct device *d, struct device_attribute *a, char *buf)
{
    struct region *r;
    int n = 0;

    mutex_lock(&g_lock);
    list_for_each_entry(r, &g_regions, all) {
        if (n > PAGE_SIZE - 200)
            break;
        n += sysfs_emit_at(buf, n, "%u L%d cpu %d node %d %zu | %llu %llu %llu of %llu, p50 %llu (cal %llu %llu %llu)\n",
                           r->id, r->level, r->cpu, r->nid, r->len, r->m.l1_l2, r->m.l3, r->m.dram, r->m.lines,
                           r->m.p50, r->m.cal_l2, r->m.cal_l3, r->m.cal_dram);
    }
    mutex_unlock(&g_lock);
    return n;
}
static DEVICE_ATTR_RO(regions);

static ssize_t caps_show(struct device *d, struct device_attribute *a, char *buf)
{
    return sysfs_emit(buf, "v%d nodes %d l2 cbm %#x lock %#x %zu B/core, l3 cbm %#x lock %#x %zu B/node\n",
                      PL_VERSION, num_online_nodes(), l2_full, l2_lock, l2_cap, l3_full, l3_lock, l3_cap);
}
static DEVICE_ATTR_RO(caps);

static struct attribute *pl_attrs[] = { &dev_attr_regions.attr, &dev_attr_caps.attr, NULL };
ATTRIBUTE_GROUPS(pl);

static struct miscdevice pl_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "pseudo_lock",
    .fops = &pl_fops,
    .mode = 0660,
    .groups = pl_groups,
};

/* ------------------------------------------------------------------ init */

static void teardown(void)
{
    int n;

    on_each_cpu(restore_masks, NULL, 1);
    for (n = 0; n < MAX_NUMNODES; n++) {
        kfree(cal_buf[n]);
        cal_buf[n] = NULL;
    }
    kfree(l2_used);
    l2_used = NULL;
    if (cpu_latency_qos_request_active(&qos_req))
        cpu_latency_qos_remove_request(&qos_req);
}

static int __init pl_init(void)
{
    int node, ret;

    ret = cat_probe();
    if (ret)
        return ret;
    l2_used = kcalloc(nr_cpu_ids, sizeof(*l2_used), GFP_KERNEL);
    if (!l2_used)
        return -ENOMEM;
    for_each_online_node(node) {
        cal_buf[node] = kmalloc_node(CAL_LINES * LINE, GFP_KERNEL, node);
        if (!cal_buf[node]) {
            teardown();
            return -ENOMEM;
        }
    }
    on_each_cpu(restore_masks, NULL, 1);
    ret = misc_register(&pl_misc);
    if (ret)
        teardown();
    return ret;
}

static void __exit pl_exit(void)
{
    /* .owner pins the module while any fd is open, so no region outlives its file. */
    misc_deregister(&pl_misc);
    teardown();
    pr_info("pseudo_lock: unloaded, CAT masks restored\n");
}

module_init(pl_init);
module_exit(pl_exit);
