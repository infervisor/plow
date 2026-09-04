#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x, what) do {                                                        \
    hsa_status_t s_ = (x);                                                        \
    if (s_ != HSA_STATUS_SUCCESS) {                                               \
        fprintf(stderr, "FAIL %s: HSA status %d\n", what, (int)s_);             \
        exit(1);                                                                  \
    }                                                                             \
} while (0)

static hsa_agent_t gpu, cpu;
static int have_gpu, have_cpu;

static hsa_status_t find_agent(hsa_agent_t agent, void* unused) {
    hsa_device_type_t type;
    (void)unused;
    CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type), "agent device");
    if (type == HSA_DEVICE_TYPE_GPU && !have_gpu) { gpu = agent; have_gpu = 1; }
    if (type == HSA_DEVICE_TYPE_CPU && !have_cpu) { cpu = agent; have_cpu = 1; }
    return HSA_STATUS_SUCCESS;
}

struct pool_search { hsa_amd_memory_pool_t pool; uint32_t flag; int found; };
static hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void* opaque) {
    struct pool_search* search = opaque;
    hsa_amd_segment_t segment;
    uint32_t flags;
    if (search->found) return HSA_STATUS_SUCCESS;
    CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment),
          "pool segment");
    if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags),
          "pool flags");
    if (flags & search->flag) { search->pool = pool; search->found = 1; }
    return HSA_STATUS_SUCCESS;
}

static hsa_amd_memory_pool_t pool_with(uint32_t flag) {
    struct pool_search search = { .flag = flag };
    CHECK(hsa_amd_agent_iterate_memory_pools(cpu, find_pool, &search), "iterate pools");
    if (!search.found) { fprintf(stderr, "FAIL no CPU pool flag %#x\n", flag); exit(1); }
    return search.pool;
}

static void* shared_alloc(hsa_amd_memory_pool_t pool, size_t bytes) {
    void* ptr;
    CHECK(hsa_amd_memory_pool_allocate(pool, bytes, 0, &ptr), "pool allocate");
    CHECK(hsa_amd_agents_allow_access(1, &gpu, NULL, ptr), "pool allow access");
    memset(ptr, 0, bytes);
    return ptr;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s probe.co XCD\n", argv[0]);
        return 2;
    }
    unsigned target = (unsigned)strtoul(argv[2], NULL, 0);
    if (target >= 8) return 2;

    CHECK(hsa_init(), "init");
    CHECK(hsa_iterate_agents(find_agent, NULL), "iterate agents");
    if (!have_gpu || !have_cpu) { fprintf(stderr, "FAIL missing HSA agent\n"); return 1; }
    uint32_t cus = 0, xccs = 0;
    CHECK(hsa_agent_get_info(gpu, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cus), "CU count");
    CHECK(hsa_agent_get_info(gpu, HSA_AMD_AGENT_INFO_NUM_XCC, &xccs), "XCC count");
    if (cus != 256 || xccs != 8) {
        fprintf(stderr, "SKIP requires 256-CU/8-XCC agent, got %u/%u\n", cus, xccs);
        return 77;
    }

    hsa_queue_t* queue;
    CHECK(hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                           UINT32_MAX, UINT32_MAX, &queue), "queue create");
    uint32_t mask[8] = {};
    for (unsigned local = 0; local < 32; ++local) {
        unsigned bit = target + 8 * local;
        mask[bit / 32] |= 1u << (bit % 32);
    }
    CHECK(hsa_amd_queue_cu_set_mask(queue, 256, mask), "queue CU mask");
    uint32_t readback[8] = {};
    CHECK(hsa_amd_queue_cu_get_mask(queue, 256, readback), "queue CU mask readback");

    FILE* file = fopen(argv[1], "rb");
    if (!file) { perror("open code object"); return 1; }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void* image = malloc((size_t)size);
    if (fread(image, 1, (size_t)size, file) != (size_t)size) return 1;
    fclose(file);
    hsa_code_object_reader_t reader;
    hsa_executable_t executable;
    hsa_executable_symbol_t symbol;
    CHECK(hsa_code_object_reader_create_from_memory(image, (size_t)size, &reader), "reader");
    CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    NULL, &executable), "executable create");
    CHECK(hsa_executable_load_agent_code_object(executable, gpu, reader, NULL, NULL), "load");
    CHECK(hsa_executable_freeze(executable, NULL), "freeze");
    CHECK(hsa_executable_get_symbol_by_name(executable, "xcd_hsa_probe.kd", &gpu, &symbol),
          "symbol");
    uint64_t kernel_object;
    uint32_t kernarg_size, private_size, group_size;
    CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                         &kernel_object), "kernel object");
    CHECK(hsa_executable_symbol_get_info(symbol,
          HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size), "kernarg size");
    CHECK(hsa_executable_symbol_get_info(symbol,
          HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size), "private size");
    CHECK(hsa_executable_symbol_get_info(symbol,
          HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size), "group size");

    hsa_amd_memory_pool_t fine = pool_with(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
    hsa_amd_memory_pool_t kpool = pool_with(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);
    uint32_t* out = shared_alloc(fine, 32 * sizeof(*out));
    uint8_t* kernarg = shared_alloc(kpool, kernarg_size);
    memcpy(kernarg, &out, sizeof(out));
    hsa_signal_t done;
    CHECK(hsa_signal_create(1, 0, NULL, &done), "signal");

    uint64_t index = hsa_queue_add_write_index_screlease(queue, 1);
    hsa_kernel_dispatch_packet_t* packet =
        &((hsa_kernel_dispatch_packet_t*)queue->base_address)[index & (queue->size - 1)];
    memset(packet, 0, sizeof(*packet));
    packet->setup = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    packet->workgroup_size_x = 512;
    packet->workgroup_size_y = packet->workgroup_size_z = 1;
    packet->grid_size_x = 32 * 512;
    packet->grid_size_y = packet->grid_size_z = 1;
    packet->private_segment_size = private_size;
    packet->group_segment_size = group_size;
    packet->kernel_object = kernel_object;
    packet->kernarg_address = kernarg;
    packet->completion_signal = done;
    uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1u << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue->doorbell_signal, (hsa_signal_value_t)index);
    hsa_signal_wait_scacquire(done, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_ACTIVE);

    unsigned counts[16] = {};
    for (unsigned i = 0; i < 32; ++i) if (out[i] < 16) counts[out[i]]++;
    printf("target=%u mask_readback=%s\nrequested:", target,
           memcmp(mask, readback, sizeof(mask)) == 0 ? "exact" : "DIFFERS");
    for (unsigned i = 0; i < 8; ++i) printf(" %08x", mask[i]);
    printf("\nreadback :");
    for (unsigned i = 0; i < 8; ++i) printf(" %08x", readback[i]);
    printf("\ncounts   :");
    for (unsigned i = 0; i < 8; ++i) printf(" %u=%u", i, counts[i]);
    printf("\n");
    for (unsigned i = 0; i < 8; ++i) if (counts[i] != 4) return 1;
    return 0;
}
