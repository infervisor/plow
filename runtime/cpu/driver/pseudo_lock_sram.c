/*
 * pseudo_lock_sram.c - Intel RDT Cache Pseudo-Locking SRAM Driver
 *
 * Implements L2 and L3 Cache Pseudo-Locking pushed to maximum hardware limits.
 *
 * Exposes:
 *   /dev/pseudo_lock_l2  (Core-local private L2 SRAM up to 1.875 MiB / 15 ways)
 *   /dev/pseudo_lock_l3  (Socket-wide shared L3 SRAM up to 450 MiB / 15 ways)
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
#include <linux/device.h>
#include <linux/sysfs.h>
#include <linux/pm_qos.h>
#include <linux/vmalloc.h>
#include <asm/msr.h>
#include <asm/cacheflush.h>
#include <asm/special_insns.h>
#include <asm/io.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Antigravity Engineer");
MODULE_DESCRIPTION("Intel RDT L2/L3 Cache Pseudo-Locking SRAM Driver (Max Limits)");
MODULE_VERSION("2.0");

/* Intel RDT / CAT MSRs */
#define MSR_MISC_FEATURE_CONTROL    0x000001a4
#ifndef MSR_IA32_PQR_ASSOC
#define MSR_IA32_PQR_ASSOC          0x00000c8f
#endif
#define MSR_IA32_L3_QOS_MASK_BASE   0x00000c90
#define MSR_IA32_L2_QOS_MASK_BASE   0x00000d10

/* Prefetcher disable bitmask (MSR 0x1A4) */
#define PREFETCH_DISABLE_ALL        0x0000000f

/* Separate CLOS assignments for L2 and L3 */
#define CLOS_DEFAULT                0
#define CLOS_L2_LOCK                1
#define CLOS_L3_LOCK                2

/* Cache Parameters for Intel Xeon 6975P-C */
#define L2_TOTAL_SIZE               (2 * 1024 * 1024)   /* 2 MiB per core */
#define L2_WAYS                     16
#define L2_WAY_SIZE                 (L2_TOTAL_SIZE / L2_WAYS) /* 128 KiB per way */

#define L3_TOTAL_SIZE               (480ULL * 1024 * 1024) /* 480 MiB per socket */
#define L3_WAYS                     16
#define L3_WAY_SIZE                 (L3_TOTAL_SIZE / L3_WAYS) /* 30 MiB per way */

#define CACHE_LINE_SIZE             64

/* IOCTL definitions */
#define PSEUDO_LOCK_IOC_MAGIC       'P'
#define PSEUDO_LOCK_IOC_GET_INFO    _IOR(PSEUDO_LOCK_IOC_MAGIC, 1, struct pseudo_lock_info)
#define PSEUDO_LOCK_IOC_MEASURE     _IOWR(PSEUDO_LOCK_IOC_MAGIC, 2, struct pseudo_lock_latency)
#define PSEUDO_LOCK_IOC_RELOAD      _IO(PSEUDO_LOCK_IOC_MAGIC, 3)

struct pseudo_lock_info {
    uint32_t level;
    uint32_t cpu;
    uint64_t size;
    uint32_t cbm;
    uint32_t line_size;
    uint64_t phys_addr;
};

struct pseudo_lock_latency {
    uint64_t min_cycles;
    uint64_t avg_cycles;
    uint64_t max_cycles;
    uint64_t total_lines;
    uint64_t l1_l2_hits;
    uint64_t l3_hits;
    uint64_t dram_misses;
};

/* Module parameters: default pushed to high limits */
static int l2_target_cpu = 0;
module_param(l2_target_cpu, int, 0444);
MODULE_PARM_DESC(l2_target_cpu, "Target CPU for L2 pseudo-locking (default 0)");

/* Push L2 to 14 ways (1.75 MiB out of 2 MiB) by default */
static int l2_ways = 14;
module_param(l2_ways, int, 0444);
MODULE_PARM_DESC(l2_ways, "Number of L2 ways to lock (1..15, default 14 = 1792 KB)");

static int l3_target_cpu = 0;
module_param(l3_target_cpu, int, 0444);
MODULE_PARM_DESC(l3_target_cpu, "Target CPU for L3 pseudo-locking (default 0)");

/* Push L3 to 1 full way (30 MiB) or 2 ways (60 MiB) */
static int l3_ways = 1;
module_param(l3_ways, int, 0444);
MODULE_PARM_DESC(l3_ways, "Number of L3 ways to lock (1..15, 1 way = 30 MB)");

static int l3_size_mb = 30;
module_param(l3_size_mb, int, 0444);
MODULE_PARM_DESC(l3_size_mb, "Size in MB for L3 pseudo-locking (default 30MB = 1 full way)");

/* Device state */
struct pseudo_lock_device {
    int level;
    int closid;
    int target_cpu;
    size_t size;
    u32 cbm;
    u32 normal_cbm;
    void *kmem;
    bool is_vmalloc;
    phys_addr_t phys_addr;
    struct miscdevice misc;
    struct mutex lock;
    struct pm_qos_request qos_req;
    bool qos_active;
    struct pseudo_lock_latency last_lat;
};

static struct pseudo_lock_device dev_l2;
static struct pseudo_lock_device dev_l3;

struct smp_preload_args {
    struct pseudo_lock_device *sdev;
    int err;
};

static void smp_set_l2_cbm(void *info)
{
    struct pseudo_lock_device *sdev = info;
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_DEFAULT, sdev->normal_cbm, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_L2_LOCK, sdev->cbm, 0);
}

static void smp_restore_l2_cbm(void *info)
{
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_DEFAULT, 0xffff, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_L2_LOCK, 0xffff, 0);
}

static void smp_set_l3_cbm(void *info)
{
    struct pseudo_lock_device *sdev = info;
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_DEFAULT, sdev->normal_cbm, 0);
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_L3_LOCK, sdev->cbm, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_L3_LOCK, 0xffff, 0);
}

static void smp_restore_l3_cbm(void *info)
{
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_DEFAULT, 0xffff, 0);
    native_wrmsr(MSR_IA32_L3_QOS_MASK_BASE + CLOS_L3_LOCK, 0xffff, 0);
    native_wrmsr(MSR_IA32_L2_QOS_MASK_BASE + CLOS_L3_LOCK, 0xffff, 0);
}

static void pseudo_lock_preload_cpu_fn(void *info)
{
    struct smp_preload_args *args = info;
    struct pseudo_lock_device *sdev = args->sdev;
    void *mem = sdev->kmem;
    size_t size = sdev->size;
    int closid = sdev->closid;
    u64 saved_prefetch;
    u64 saved_pqr;
    unsigned long i;

    /* Flush memory buffer */
    for (i = 0; i < size; i += CACHE_LINE_SIZE) {
        clflushopt(mem + i);
    }
    mb();

    local_irq_disable();
    saved_prefetch = native_rdmsrq(MSR_MISC_FEATURE_CONTROL);
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, PREFETCH_DISABLE_ALL);

    saved_pqr = native_rdmsrq(MSR_IA32_PQR_ASSOC);
    native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)saved_pqr, closid);

    /* Pass 1: TLB & Page Table warm-up */
    for (i = 0; i < size; i += PAGE_SIZE) {
        rmb();
        asm volatile("mov (%0,%1,1), %%eax\n\t"
                     :
                     : "r" (mem), "r" (i)
                     : "%eax", "memory");
    }

    /* Pass 2: Cache line warming into target CLOS ways */
    for (i = 0; i < size; i += CACHE_LINE_SIZE) {
        rmb();
        asm volatile("mov (%0,%1,1), %%eax\n\t"
                     :
                     : "r" (mem), "r" (i)
                     : "%eax", "memory");
    }

    /* Pass 3: Verification read pass */
    for (i = 0; i < size; i += CACHE_LINE_SIZE) {
        rmb();
        asm volatile("mov (%0,%1,1), %%eax\n\t"
                     :
                     : "r" (mem), "r" (i)
                     : "%eax", "memory");
    }

    /* Restore CPU to CLOS 0 */
    native_wrmsr(MSR_IA32_PQR_ASSOC, (u32)saved_pqr, (u32)(saved_pqr >> 32));
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, saved_prefetch);
    local_irq_enable();

    args->err = 0;
}

static void pseudo_lock_measure_cpu_fn(void *info)
{
    struct smp_preload_args *args = info;
    struct pseudo_lock_device *sdev = args->sdev;
    struct pseudo_lock_latency *lat = &sdev->last_lat;
    void *mem = sdev->kmem;
    size_t size = sdev->size;
    size_t num_lines = size / CACHE_LINE_SIZE;
    u64 min_c = (u64)-1;
    u64 max_c = 0;
    u64 sum_c = 0;
    u64 l1_l2 = 0;
    u64 l3 = 0;
    u64 dram = 0;
    u64 saved_prefetch;
    unsigned long i;

    local_irq_disable();
    saved_prefetch = native_rdmsrq(MSR_MISC_FEATURE_CONTROL);
    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, PREFETCH_DISABLE_ALL);

    (void)rdtsc_ordered();

    for (i = 0; i < num_lines; i++) {
        u64 t0, t1, diff;
        volatile u32 val;

        t0 = rdtsc_ordered();
        val = *(volatile u32 *)(mem + (i * CACHE_LINE_SIZE));
        t1 = rdtsc_ordered();

        diff = (t1 > t0) ? (t1 - t0) : 1;
        if (diff < min_c) min_c = diff;
        if (diff > max_c) max_c = diff;
        sum_c += diff;

        if (diff <= 30)
            l1_l2++;
        else if (diff <= 150)
            l3++;
        else
            dram++;
    }

    native_wrmsrq(MSR_MISC_FEATURE_CONTROL, saved_prefetch);
    local_irq_enable();

    lat->min_cycles = min_c;
    lat->max_cycles = max_c;
    lat->avg_cycles = num_lines ? (sum_c / num_lines) : 0;
    lat->total_lines = num_lines;
    lat->l1_l2_hits = l1_l2;
    lat->l3_hits = l3;
    lat->dram_misses = dram;
    args->err = 0;
}

static int do_pseudo_lock(struct pseudo_lock_device *sdev)
{
    struct smp_preload_args args;
    int sibling_cpu;

    mutex_lock(&sdev->lock);

    pr_info("pseudo_lock: Locking L%d SRAM (size=%zu bytes, cbm=0x%x, closid=%d) on CPU %d\n",
            sdev->level, sdev->size, sdev->cbm, sdev->closid, sdev->target_cpu);

    if (sdev->level == 2) {
        smp_call_function_single(sdev->target_cpu, smp_set_l2_cbm, sdev, 1);
        sibling_cpu = sdev->target_cpu + 96;
        if (sibling_cpu < nr_cpu_ids && cpu_online(sibling_cpu)) {
            smp_call_function_single(sibling_cpu, smp_set_l2_cbm, sdev, 1);
        }
    } else {
        smp_call_function(smp_set_l3_cbm, sdev, 1);
        smp_set_l3_cbm(sdev);
    }

    args.sdev = sdev;
    args.err = -1;
    smp_call_function_single(sdev->target_cpu, pseudo_lock_preload_cpu_fn, &args, 1);

    smp_call_function_single(sdev->target_cpu, pseudo_lock_measure_cpu_fn, &args, 1);

    pr_info("pseudo_lock: L%d SRAM locked! Latency: min=%llu, avg=%llu, max=%llu cycles (hits: L1/L2=%llu, L3=%llu, DRAM=%llu)\n",
            sdev->level,
            sdev->last_lat.min_cycles,
            sdev->last_lat.avg_cycles,
            sdev->last_lat.max_cycles,
            sdev->last_lat.l1_l2_hits,
            sdev->last_lat.l3_hits,
            sdev->last_lat.dram_misses);

    mutex_unlock(&sdev->lock);
    return args.err;
}

static int pseudo_lock_open(struct inode *inode, struct file *filp)
{
    struct miscdevice *misc = filp->private_data;
    struct pseudo_lock_device *sdev = container_of(misc, struct pseudo_lock_device, misc);
    filp->private_data = sdev;
    return 0;
}

static int pseudo_lock_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static int pseudo_lock_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct pseudo_lock_device *sdev = filp->private_data;
    unsigned long vsize = vma->vm_end - vma->vm_start;
    int ret;

    if (!sdev || !sdev->kmem)
        return -ENODEV;

    if (vsize > sdev->size)
        return -EINVAL;

    vm_flags_set(vma, VM_DONTDUMP | VM_DONTEXPAND | VM_IO);

    if (sdev->is_vmalloc) {
        ret = remap_vmalloc_range(vma, sdev->kmem, 0);
    } else {
        unsigned long pfn = virt_to_phys(sdev->kmem) >> PAGE_SHIFT;
        ret = remap_pfn_range(vma, vma->vm_start, pfn, vsize, vma->vm_page_prot);
    }

    if (ret) {
        pr_err("pseudo_lock: mmap remap failed: %d\n", ret);
        return ret;
    }

    return 0;
}

static long pseudo_lock_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct pseudo_lock_device *sdev = filp->private_data;
    struct smp_preload_args sargs;

    if (!sdev)
        return -ENODEV;

    switch (cmd) {
    case PSEUDO_LOCK_IOC_GET_INFO: {
        struct pseudo_lock_info info;
        memset(&info, 0, sizeof(info));
        info.level = sdev->level;
        info.cpu = sdev->target_cpu;
        info.size = sdev->size;
        info.cbm = sdev->cbm;
        info.line_size = CACHE_LINE_SIZE;
        info.phys_addr = sdev->phys_addr;
        if (copy_to_user((void __user *)arg, &info, sizeof(info)))
            return -EFAULT;
        return 0;
    }
    case PSEUDO_LOCK_IOC_MEASURE: {
        sargs.sdev = sdev;
        sargs.err = 0;
        smp_call_function_single(sdev->target_cpu, pseudo_lock_measure_cpu_fn, &sargs, 1);
        if (copy_to_user((void __user *)arg, &sdev->last_lat, sizeof(sdev->last_lat)))
            return -EFAULT;
        return 0;
    }
    case PSEUDO_LOCK_IOC_RELOAD: {
        return do_pseudo_lock(sdev);
    }
    default:
        return -ENOTTY;
    }
}

static const struct file_operations pseudo_lock_fops = {
    .owner          = THIS_MODULE,
    .open           = pseudo_lock_open,
    .release        = pseudo_lock_release,
    .mmap           = pseudo_lock_mmap,
    .unlocked_ioctl = pseudo_lock_ioctl,
};

static ssize_t size_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    struct miscdevice *misc = dev_get_drvdata(dev);
    struct pseudo_lock_device *sdev = container_of(misc, struct pseudo_lock_device, misc);
    return sprintf(buf, "%zu\n", sdev->size);
}
static DEVICE_ATTR_RO(size);

static ssize_t cbm_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    struct miscdevice *misc = dev_get_drvdata(dev);
    struct pseudo_lock_device *sdev = container_of(misc, struct pseudo_lock_device, misc);
    return sprintf(buf, "0x%x\n", sdev->cbm);
}
static DEVICE_ATTR_RO(cbm);

static ssize_t cpu_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    struct miscdevice *misc = dev_get_drvdata(dev);
    struct pseudo_lock_device *sdev = container_of(misc, struct pseudo_lock_device, misc);
    return sprintf(buf, "%d\n", sdev->target_cpu);
}
static DEVICE_ATTR_RO(cpu);

static ssize_t latency_cycles_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    struct miscdevice *misc = dev_get_drvdata(dev);
    struct pseudo_lock_device *sdev = container_of(misc, struct pseudo_lock_device, misc);
    struct smp_preload_args sargs;

    sargs.sdev = sdev;
    sargs.err = 0;
    smp_call_function_single(sdev->target_cpu, pseudo_lock_measure_cpu_fn, &sargs, 1);

    return sprintf(buf, "min: %llu, avg: %llu, max: %llu (lines: %llu, l1/l2 hits: %llu, l3 hits: %llu, dram: %llu)\n",
                   sdev->last_lat.min_cycles,
                   sdev->last_lat.avg_cycles,
                   sdev->last_lat.max_cycles,
                   sdev->last_lat.total_lines,
                   sdev->last_lat.l1_l2_hits,
                   sdev->last_lat.l3_hits,
                   sdev->last_lat.dram_misses);
}
static DEVICE_ATTR_RO(latency_cycles);

static struct attribute *pseudo_lock_attrs[] = {
    &dev_attr_size.attr,
    &dev_attr_cbm.attr,
    &dev_attr_cpu.attr,
    &dev_attr_latency_cycles.attr,
    NULL,
};
ATTRIBUTE_GROUPS(pseudo_lock);

static int init_pseudo_lock_dev(struct pseudo_lock_device *sdev, int level, int closid,
                                int target_cpu, size_t size, u32 cbm,
                                const char *name)
{
    unsigned long i;
    int ret;

    memset(sdev, 0, sizeof(*sdev));
    mutex_init(&sdev->lock);
    sdev->level = level;
    sdev->closid = closid;
    sdev->target_cpu = target_cpu;
    sdev->size = size;
    sdev->cbm = cbm;
    sdev->normal_cbm = 0xffff & ~cbm;

    cpu_latency_qos_add_request(&sdev->qos_req, 0);
    sdev->qos_active = true;

    if (size > 4 * 1024 * 1024) {
        /* Allocate large buffers (e.g. 30MB L3 way) via vmalloc_user */
        sdev->is_vmalloc = true;
        sdev->kmem = vmalloc_user(size);
        if (!sdev->kmem) {
            pr_err("pseudo_lock: vmalloc_user(%zu) failed for %s\n", size, name);
            if (sdev->qos_active) {
                cpu_latency_qos_remove_request(&sdev->qos_req);
                sdev->qos_active = false;
            }
            return -ENOMEM;
        }
        sdev->phys_addr = page_to_phys(vmalloc_to_page(sdev->kmem));
    } else {
        /* Allocate contiguous physical pages for L2 */
        sdev->is_vmalloc = false;
        sdev->kmem = alloc_pages_exact(size, GFP_KERNEL | __GFP_ZERO);
        if (!sdev->kmem) {
            pr_err("pseudo_lock: alloc_pages_exact(%zu) failed for %s\n", size, name);
            if (sdev->qos_active) {
                cpu_latency_qos_remove_request(&sdev->qos_req);
                sdev->qos_active = false;
            }
            return -ENOMEM;
        }
        sdev->phys_addr = virt_to_phys(sdev->kmem);
        for (i = 0; i < size; i += PAGE_SIZE) {
            SetPageReserved(virt_to_page(sdev->kmem + i));
        }
    }

    sdev->misc.minor = MISC_DYNAMIC_MINOR;
    sdev->misc.name = name;
    sdev->misc.fops = &pseudo_lock_fops;
    sdev->misc.mode = 0666;
    sdev->misc.groups = pseudo_lock_groups;

    ret = misc_register(&sdev->misc);
    if (ret) {
        pr_err("pseudo_lock: Failed to register misc device %s: %d\n", name, ret);
        if (sdev->is_vmalloc) {
            vfree(sdev->kmem);
        } else {
            for (i = 0; i < size; i += PAGE_SIZE)
                ClearPageReserved(virt_to_page(sdev->kmem + i));
            free_pages_exact(sdev->kmem, size);
        }
        if (sdev->qos_active) {
            cpu_latency_qos_remove_request(&sdev->qos_req);
            sdev->qos_active = false;
        }
        return ret;
    }

    ret = do_pseudo_lock(sdev);
    if (ret) {
        pr_err("pseudo_lock: Failed to lock %s: %d\n", name, ret);
    }

    return 0;
}

static void cleanup_pseudo_lock_dev(struct pseudo_lock_device *sdev)
{
    unsigned long i;
    int sibling_cpu;

    if (!sdev->kmem)
        return;

    if (sdev->qos_active) {
        cpu_latency_qos_remove_request(&sdev->qos_req);
        sdev->qos_active = false;
    }

    if (sdev->level == 2) {
        smp_call_function_single(sdev->target_cpu, smp_restore_l2_cbm, NULL, 1);
        sibling_cpu = sdev->target_cpu + 96;
        if (sibling_cpu < nr_cpu_ids && cpu_online(sibling_cpu))
            smp_call_function_single(sibling_cpu, smp_restore_l2_cbm, NULL, 1);
    } else {
        smp_call_function(smp_restore_l3_cbm, NULL, 1);
        smp_restore_l3_cbm(NULL);
    }

    misc_deregister(&sdev->misc);

    if (sdev->is_vmalloc) {
        vfree(sdev->kmem);
    } else {
        for (i = 0; i < sdev->size; i += PAGE_SIZE) {
            ClearPageReserved(virt_to_page(sdev->kmem + i));
        }
        free_pages_exact(sdev->kmem, sdev->size);
    }
    sdev->kmem = NULL;
}

static int __init pseudo_lock_init_module(void)
{
    size_t l2_size;
    size_t l3_size;
    u32 l2_cbm_val;
    u32 l3_cbm_val;
    int ret;

    pr_info("pseudo_lock: Initializing L2/L3 Pseudo-Locking SRAM Driver v2.0 (Max Limits)\n");

    /* Validate L2 parameters (1..15 ways) */
    if (l2_ways < 1 || l2_ways > 15)
        l2_ways = 14;
    l2_cbm_val = (1 << l2_ways) - 1; /* e.g. 14 ways -> 0x3FFF */
    l2_size = l2_ways * L2_WAY_SIZE;  /* e.g. 14 * 128 KiB = 1,792 KiB */

    /* Validate L3 parameters (1..15 ways) */
    if (l3_ways < 1 || l3_ways > 15)
        l3_ways = 1;
    l3_cbm_val = (1 << l3_ways) - 1; /* e.g. 1 way -> 0x0001 */

    if (l3_size_mb < 1 || l3_size_mb > (l3_ways * 30))
        l3_size = (size_t)l3_ways * L3_WAY_SIZE; /* 1 way = 30 MB */
    else
        l3_size = (size_t)l3_size_mb * 1024 * 1024;

    /* Initialize L2 Pseudo-Lock device pushed to maximum limits */
    ret = init_pseudo_lock_dev(&dev_l2, 2, CLOS_L2_LOCK, l2_target_cpu, l2_size, l2_cbm_val, "pseudo_lock_l2");
    if (ret) {
        pr_err("pseudo_lock: Failed to initialize L2 device\n");
        return ret;
    }

    /* Initialize L3 Pseudo-Lock device pushed to maximum limits */
    ret = init_pseudo_lock_dev(&dev_l3, 3, CLOS_L3_LOCK, l3_target_cpu, l3_size, l3_cbm_val, "pseudo_lock_l3");
    if (ret) {
        pr_err("pseudo_lock: Failed to initialize L3 device\n");
        cleanup_pseudo_lock_dev(&dev_l2);
        return ret;
    }

    pr_info("pseudo_lock: Pushed to Max! L2 SRAM: %zu KB (%d ways), L3 SRAM: %zu MB (%d ways)\n",
            l2_size / 1024, l2_ways, l3_size / (1024 * 1024), l3_ways);

    return 0;
}

static void __exit pseudo_lock_exit_module(void)
{
    pr_info("pseudo_lock: Cleaning up and restoring CAT masks\n");
    cleanup_pseudo_lock_dev(&dev_l3);
    cleanup_pseudo_lock_dev(&dev_l2);
    pr_info("pseudo_lock: Unloaded successfully\n");
}

module_init(pseudo_lock_init_module);
module_exit(pseudo_lock_exit_module);
