/* block_fp8_gfx950_test.c — DeepSeek/GLM BLOCK-fp8 decode GEMV on device vs an f64 reference.
 *
 * GLM-5.2-FP8 (and DeepSeek-V3) quantise weights with weight_block_size [128,128]: the weight is
 * e4m3 and there is ONE f32 dequant scale per [128 out-channel][128 K] block, laid out as a
 * ceil(N/128) x ceil(K/128) row-major grid. This is DIFFERENT from plow's existing per-CHANNEL fp8
 * GEMV (one scale per output column, applied once in the epilogue): the block scale varies along K
 * and so must be folded into the reduction per 128-K block. This test drives the new gemv_fp8_blk
 * wrapper (d_gemv_fp8_blk -> gemv_rows_fp8_blk) and checks it against a decode-fp8-dequant f64
 * reference over the SAME block scheme, on real GLM decode shapes plus a ragged (non-128-multiple)
 * shape that exercises the ceil / overshoot-clamp path.
 *
 * w8a16: x (activation) is bf16, W (weight) is e4m3 block-scaled — the plow decode weight-stream
 * path. Truth is decode(W)*x*block_scale summed in f64 (measures the KERNEL, not the format).
 *
 * Build with scripts/build_block_fp8.sh; run under `sg render` on one GPU.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (bf16)(u >> 16);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* OCP e4m3 (torch.float8_e4m3fn) decode — same as gemm_gfx950_test.c. */
static double e4m3_decode(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    double v;
    if (e == 0) v = (m / 8.0) * 0.015625;
    else v = (1.0 + m / 8.0) * ldexp(1.0, e - 7);
    return s ? -v : v;
}

static int fails = 0;
static const char* a8w8_capture;

static void capture_a8w8(unsigned M, unsigned N, unsigned K, const void* a, const void* w,
                         const void* as, const void* ws, const void* out) {
    if (!a8w8_capture) return;
    char path[4096];
    int len = snprintf(path, sizeof(path), "%s/%u_%u_%u.bin", a8w8_capture, M, N, K);
    if (len < 0 || (size_t)len >= sizeof(path)) exit(1);
    FILE* f = fopen(path, "wbx");
    if (!f) { perror(path); exit(1); }
    const uint32_t header[] = {M, N, K};
    const size_t kb = (K + 127u) / 128u, nb = (N + 127u) / 128u;
    const void* parts[] = {header, a, w, as, ws, out};
    const size_t sizes[] = {sizeof(header), (size_t)M * K, (size_t)N * K,
                            kb * M * 4, nb * kb * 4, (size_t)M * N * 2};
    for (unsigned i = 0; i < sizeof(parts) / sizeof(parts[0]); i++)
        if (fwrite(parts[i], 1, sizes[i], f) != sizes[i]) exit(1);
    if (fclose(f)) exit(1);
}

static void check_hsa(int rc) {
    if (rc) {
        fprintf(stderr, "%s\n", plow_hsa_last_error());
        exit(1);
    }
}

static void load_pipeline_object(plow_hsa* h, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END)) exit(1);
    const long bytes = ftell(f);
    if (bytes <= 0 || fseek(f, 0, SEEK_SET)) exit(1);
    void* image = malloc(bytes);
    if (!image || fread(image, 1, bytes, f) != (size_t)bytes || fclose(f)) exit(1);
    check_hsa(plow_hsa_load_code_object(h, 0, image, bytes));
    free(image);
}

static void run_attention_pipeline(plow_hsa* h, unsigned m, unsigned ctx, unsigned nw, unsigned cap,
        const uint32_t* lengths, const uint32_t* kvptr,
        void** host, void** gpu, void* dcsr, void* part, void* lse, void* out, void* dmetadata,
        plow_hsa_kernel* stage, const void* stage_args, size_t stage_arg_size,
        float* actual, float* again, const char* output, const char* const* paths) {
    const unsigned live = ctx < 2048 ? ctx : 2048;
    const size_t bytes[] = {(size_t)m * 8 * 512 * 2, (size_t)m * 8 * 64 * 2,
        (size_t)m * ctx * 512 * 2, (size_t)m * ctx * 64 * 2, (size_t)m * 2048 * 4, m * 4,
        (m + 1) * 4, m * 2 * 4, cap * 4, (size_t)m * 8 * 512 * 2};
    void* input[10]; void* device[10];
    for (unsigned i = 0; i < 10; i++) {
        input[i] = plow_hsa_alloc_host(h, bytes[i]); device[i] = plow_hsa_alloc(h, 0, bytes[i]);
        if (!input[i] || !device[i]) exit(1);
        memset(input[i], 0xff, bytes[i]);
    }
    const bf16* q = host[0]; const bf16* kv = host[1]; const int32_t* indices = host[2];
    for (unsigned b = 0; b < m; b++) {
        ((int32_t*)input[5])[b] = lengths[b];
        for (unsigned head = 0; head < 8; head++) {
            const size_t source = ((size_t)b * 16 + head * 2) * 576;
            if (memcmp(q + source, q + source + 576, 576 * 2)) exit(1);
            memcpy((bf16*)input[0] + ((size_t)b * 8 + head) * 512, q + source, 512 * 2);
            memcpy((bf16*)input[1] + ((size_t)b * 8 + head) * 64, q + source + 512, 64 * 2);
        }
        for (unsigned j = kvptr[b]; j < kvptr[b + 1]; j++)
            ((int32_t*)input[4])[b * 2048 + j - kvptr[b]] = indices[j] - b * ctx;
    }
    for (size_t row = 0; row < (size_t)m * ctx; row++) {
        memcpy((bf16*)input[2] + row * 512, kv + row * 576, 512 * 2);
        memcpy((bf16*)input[3] + row * 64, kv + row * 576 + 512, 64 * 2);
    }
    for (unsigned i = 0; i < 6; i++) check_hsa(plow_hsa_copy_h2d(h, 0, device[i], input[i], bytes[i]));
    FILE* f = fopen(paths[0], "rb"); uint32_t hdr[6];
    if (!f || fread(hdr, sizeof(hdr), 1, f) != 1 || hdr[0] != 0x41505231 || hdr[1] != m
            || hdr[2] != nw || hdr[3] != cap || hdr[4] != m || hdr[5] > 256) exit(1);
    if (fseek(f, (long)(bytes[6] + bytes[7] + bytes[8]), SEEK_CUR)) exit(1);
    bf16* reference = plow_hsa_alloc_host(h, bytes[9] * 2);
    bf16* result = plow_hsa_alloc_host(h, bytes[9]);
    bf16* repeat = plow_hsa_alloc_host(h, bytes[9]);
    if (!reference || !result || !repeat || fread(reference, 1, bytes[9] * 2, f) != bytes[9] * 2
            || fgetc(f) != EOF || ferror(f) || fclose(f)) exit(1);
    plow_hsa_kernel pack, unpad, metadata, reduce;
    load_pipeline_object(h, paths[1]);
    check_hsa(plow_hsa_get_kernel(h, 0, "plow_mla_bf16_pack", &pack));
    check_hsa(plow_hsa_get_kernel(h, 0, "plow_mla_bf16_unpad", &unpad));
    load_pipeline_object(h, paths[2]);
    check_hsa(plow_hsa_get_kernel(h, 0,
        "_Z33kn_get_mla_metadata_v1_2_parallelI20MlaMetadataV12TraitsILi128ELb0ELi1ELb1ELb0EEEv28MlaMetadataV1KernelParameter", &metadata));
    load_pipeline_object(h, paths[3]);
    check_hsa(plow_hsa_get_kernel(h, 0,
        "_Z16kn_mla_reduce_v1I23MlaReduceKernelV1TraitsILi512ELi16ELi1EEfDF16bEv23MlaReduceKernelV1Params24MlaReduceKernelV1Configs", &reduce));
    if (metadata.kernarg_size != 392 || metadata.private_segment_size || metadata.group_segment_size
            || reduce.kernarg_size != 84 || reduce.private_segment_size || reduce.group_segment_size
            || pack.kernarg_size != 368 || pack.private_segment_size
            || unpad.kernarg_size != 280 || unpad.private_segment_size) {
        fprintf(stderr, "pipeline kernel ABI mismatch pack=%u unpad=%u\n", pack.kernarg_size, unpad.kernarg_size); exit(1);
    }
    const uint64_t qp = (uintptr_t)dcsr, kp = qp + (m + 1) * 4, last = qp + (2 * m + 2) * 4;
    uint64_t pa[14] = {(uintptr_t)gpu[0], (uintptr_t)gpu[1], (uintptr_t)gpu[2],
        (uintptr_t)device[0], (uintptr_t)device[1], (uintptr_t)device[2], (uintptr_t)device[3],
        (uintptr_t)device[4], (uintptr_t)device[5], qp, kp, last};
    const uint32_t pd[4] = {m, ctx, 2048, 0}; memcpy(pa + 12, pd, sizeof(pd));
    unsigned sc = 1; while (sc < live / 128) sc *= 2;
    unsigned ns = sc * m; if (ns > 256) ns = 256;
    uint64_t ma[17] = {(uintptr_t)dmetadata, (uintptr_t)gpu[3], (uintptr_t)gpu[4],
        (uintptr_t)device[6], (uintptr_t)device[7], (uintptr_t)device[8], qp, kp, last};
    const uint32_t md[13] = {m, 0, 16, 256, m + 1, 1, 16, 4, 1, 1, UINT32_MAX, 1, ns};
    memcpy((char*)ma + 72, md, sizeof(md)); ((unsigned char*)ma)[124] = 1;
    const uint32_t mt[2] = {16, 1}; memcpy((char*)ma + 128, mt, sizeof(mt));
    uint64_t ra[11] = {(uintptr_t)device[6], (uintptr_t)device[7], (uintptr_t)device[8],
        0, (uintptr_t)out, (uintptr_t)lse, (uintptr_t)part};
    const uint32_t rd[4] = {16 * 512, 512, 256, m};
    memcpy((char*)ra + 56, rd, sizeof(rd)); ((unsigned char*)ra)[73] = 1; ra[10] = 256;
    const uint64_t ua[3] = {(uintptr_t)device[9], (uintptr_t)out, m};
    const size_t part_bytes = (size_t)nw * 16 * 512 * 4, lse_bytes = (size_t)nw * 16 * 4;
    void* poison_targets[] = {gpu[0], gpu[1], gpu[2], gpu[3], gpu[4], dcsr, dmetadata, part, lse, out,
        device[6], device[7], device[8], device[9]};
    const size_t poison_sizes[] = {(size_t)m * 16 * 576 * 2, (size_t)kvptr[m] * 576 * 2,
        kvptr[m] * 4, 257 * 4, cap * 8 * 4, (3 * m + 2) * 4, 80,
        (size_t)cap * 16 * 512 * 4, (size_t)cap * 16 * 4, bytes[9] * 2,
        bytes[6], bytes[7], bytes[8], bytes[9]};
    size_t poison_bytes = 0;
    for (unsigned i = 0; i < 14; i++) if (poison_bytes < poison_sizes[i]) poison_bytes = poison_sizes[i];
    void* poison = plow_hsa_alloc_host(h, poison_bytes);
    if (!poison) exit(1);
    memset(poison, 0xff, poison_bytes);
    for (unsigned run = 0; run < 2; run++) {
        for (unsigned i = 0; i < 14; i++) check_hsa(plow_hsa_copy_h2d(h, 0, poison_targets[i], poison, poison_sizes[i]));
        check_hsa(plow_hsa_launch(h, 0, &pack, 256 * 256, 1, 1, 256, 1, 1, 0, pa, sizeof(pa)));
        check_hsa(plow_hsa_launch(h, 0, &metadata, 512, 1, 1, 512, 1, 1, 163840, ma, sizeof(ma)));
        check_hsa(plow_hsa_launch(h, 0, stage, 256 * 256, 1, 1, 256, 1, 1, 0, stage_args, stage_arg_size));
        check_hsa(plow_hsa_launch(h, 0, &reduce, 16 * 128, 1, m, 128, 1, 1, 2048, ra, 84));
        check_hsa(plow_hsa_launch(h, 0, &unpad, 16 * 256, 1, 1, 256, 1, 1, 0, ua, sizeof(ua)));
        check_hsa(plow_hsa_wait(h, 0));
        float* dst = run ? again : actual;
        check_hsa(plow_hsa_copy_d2h(h, 0, dst, part, part_bytes));
        check_hsa(plow_hsa_copy_d2h(h, 0, (char*)dst + part_bytes, lse, lse_bytes));
        check_hsa(plow_hsa_copy_d2h(h, 0, run ? repeat : result, device[9], bytes[9]));
    }
    size_t mismatch = 0; int finite = 1;
    for (size_t i = 0; i < bytes[9] / 2; i++) {
        const bf16 ref = reference[(i / 512) * 1024 + i % 512];
        mismatch += result[i] != ref;
        finite &= isfinite(bf2f(result[i])) && isfinite(bf2f(ref)) && isfinite(bf2f(repeat[i]));
    }
    const int stable = memcmp(result, repeat, bytes[9]) == 0;
    printf("attention-pipeline M=%u ctx=%u launches=5 host-boundaries=1 finite=%d mismatches=%zu repeat-bitwise=%d\n",
        m, ctx, finite, mismatch, stable);
    if (!finite || mismatch || !stable) fails++;
    char path[4096]; const int length = snprintf(path, sizeof(path), "%s.bf16", output);
    if (length < 0 || (size_t)length >= sizeof(path)) exit(1);
    f = fopen(path, "wbx");
    if (!f || fwrite(result, 1, bytes[9], f) != bytes[9] || fclose(f)) exit(1);
    for (unsigned i = 0; i < 10; i++) { plow_hsa_free(h, input[i]); plow_hsa_free(h, device[i]); }
    plow_hsa_free(h, reference); plow_hsa_free(h, result); plow_hsa_free(h, repeat); plow_hsa_free(h, poison);
}

static void replay_attention_ps(plow_hsa* h, const char* input, const char* output, const char* const* pipeline) {
    FILE* f = fopen(input, "rb");
    uint32_t header[7];
    if (!f || fread(header, sizeof(header), 1, f) != 1) { perror(input); exit(1); }
    const unsigned m = header[1], ctx = header[2], topk = header[3], nw = header[4];
    const unsigned nc = header[5], cap = header[6];
    if ((header[0] != 0x41505331 && header[0] != 0x41505332) || !m || m >= 32 || !ctx || ctx > 131072 || topk != 2048
            || nc != 256 || !nw || nw > cap || cap > nc + m) {
        fprintf(stderr, "invalid persistent attention fixture geometry\n"); exit(1);
    }
    uint32_t lengths[32], kvptr[33] = {0};
    if (header[0] == 0x41505332) {
        if (fread(lengths, sizeof(uint32_t), m, f) != m) exit(1);
    } else for (unsigned b = 0; b < m; b++) lengths[b] = ctx;
    for (unsigned b = 0; b < m; b++) {
        if (!lengths[b] || lengths[b] > ctx) exit(1);
        kvptr[b + 1] = kvptr[b] + (lengths[b] < topk ? lengths[b] : topk);
    }
    char name[64]; uint32_t cus = 0, lds = 0;
    check_hsa(plow_hsa_device_info(h, 0, name, &cus, &lds));
    if (cus != nc || strcmp(name, "gfx950")) { fprintf(stderr, "fixture device differs\n"); exit(1); }
    const size_t sizes[] = {(size_t)m * 16 * 576 * 2, (size_t)m * ctx * 576 * 2,
        (size_t)kvptr[m] * 4, (nc + 1) * 4, (size_t)cap * 8 * 4,
        (size_t)nw * 16 * 512 * 4, (size_t)nw * 16 * 4};
    void* host[7]; void* gpu[5];
    for (unsigned i = 0; i < 7; i++) {
        host[i] = plow_hsa_alloc_host(h, sizes[i]);
        if (!host[i] || fread(host[i], 1, sizes[i], f) != sizes[i]) {
            fprintf(stderr, "truncated persistent attention fixture\n"); exit(1);
        }
    }
    if (fgetc(f) != EOF || ferror(f) || fclose(f)) { fprintf(stderr, "unexpected fixture tail\n"); exit(1); }
    const int32_t* wi = host[4]; const int32_t* wp = host[3]; const int32_t* idx = host[2];
    if (wp[0] != 0 || wp[nc] != (int32_t)nw) exit(1);
    for (unsigned i = 0; i < nc; i++)
        if (wp[i] < 0 || wp[i] > wp[i + 1] || wp[i + 1] > (int32_t)nw) exit(1);
    unsigned seen[32] = {0}, works[32] = {0}, direct[32] = {0}, npartial = 0;
    for (unsigned i = 0; i < nw; i++) {
        const int32_t* row = wi + i * 8;
        if (row[0] < 0 || row[0] >= (int32_t)m || row[1] < -1
                || (row[1] >= 0 && row[1] != (int32_t)npartial)
                || row[2] != row[0] || row[3] != row[0] + 1 || row[4] < 0
                || row[4] != (int32_t)(kvptr[row[0]] + seen[row[0]]) || row[5] <= row[4]
                || row[5] > (int32_t)kvptr[row[0] + 1]
                || row[6] != (int32_t)kvptr[row[0] + 1] - row[5]) {
            fprintf(stderr, "invalid persistent work range\n"); exit(1);
        }
        seen[row[0]] += (unsigned)(row[5] - row[4]);
        works[row[0]]++;
        direct[row[0]] += row[1] == -1;
        npartial += row[1] >= 0;
    }
    for (unsigned b = 0; b < m; b++) {
        if (seen[b] != kvptr[b + 1] - kvptr[b] || direct[b] != (works[b] == 1)) exit(1);
        for (unsigned j = kvptr[b]; j < kvptr[b + 1]; j++)
            if (idx[j] < (int32_t)(b * ctx) || idx[j] >= (int32_t)(b * ctx + lengths[b])) exit(1);
    }
    for (unsigned i = 0; i < 2; i++)
        for (size_t j = 0; j < sizes[i] / 2; j++)
            if (!isfinite(bf2f(((bf16*)host[i])[j]))) exit(1);
    for (unsigned i = 0; i < 5; i++) {
        gpu[i] = plow_hsa_alloc(h, 0, sizes[i]);
        if (!gpu[i]) exit(1);
        check_hsa(plow_hsa_copy_h2d(h, 0, gpu[i], host[i], sizes[i]));
    }
    int32_t* csr = plow_hsa_alloc_host(h, (3 * m + 2) * 4);
    if (!csr) exit(1);
    for (unsigned i = 0; i <= m; i++) { csr[i] = i; csr[m + 1 + i] = kvptr[i]; }
    for (unsigned i = 0; i < m; i++) csr[2 * m + 2 + i] = 1;
    void* dcsr = plow_hsa_alloc(h, 0, (3 * m + 2) * 4);
    void* part = plow_hsa_alloc(h, 0, (size_t)cap * 16 * 512 * 4);
    void* lse = plow_hsa_alloc(h, 0, (size_t)cap * 16 * 4);
    void* out = plow_hsa_alloc(h, 0, (size_t)m * 16 * 512 * 2);
    uint64_t* metadata = plow_hsa_alloc_host(h, 80);
    void* dmetadata = plow_hsa_alloc(h, 0, 80);
    if (!dcsr || !part || !lse || !out || !metadata || !dmetadata) exit(1);
    memset(metadata, 0, 80); metadata[0] = (uintptr_t)gpu[3]; metadata[1] = (uintptr_t)gpu[4];
    check_hsa(plow_hsa_copy_h2d(h, 0, dcsr, csr, (3 * m + 2) * 4));
    check_hsa(plow_hsa_copy_h2d(h, 0, dmetadata, metadata, 80));
    uint64_t args[48] = {0};
    args[0] = (uintptr_t)part; args[2] = (uintptr_t)lse;
    args[4] = (uintptr_t)gpu[0]; args[6] = (uintptr_t)gpu[1];
    args[8] = (uintptr_t)dcsr + (m + 1) * 4; args[10] = (uintptr_t)gpu[2];
    args[12] = (uintptr_t)dcsr + (2 * m + 2) * 4;
    float scale = 0.0625f; memcpy(&args[14], &scale, 4);
    args[16] = 16; args[18] = 1; args[20] = 16 * 576 * 2; args[22] = 576 * 2;
    args[26] = (uintptr_t)dcsr; args[28] = (uintptr_t)dmetadata; args[30] = (uintptr_t)out;
    args[36] = 1; args[42] = 1;
    plow_hsa_kernel kernel;
    check_hsa(plow_hsa_get_kernel(h, 0, "_ZN5aiter41mla_a16w16_qh64_qseqlen1_gqaratio64_v3_psE", &kernel));
    if (kernel.kernarg_size != sizeof(args) || kernel.private_segment_size
            || kernel.group_segment_size != 163840) {
        fprintf(stderr, "persistent attention kernel ABI differs\n"); exit(1);
    }
    float* actual = plow_hsa_alloc_host(h, sizes[5] + sizes[6]);
    float* again = plow_hsa_alloc_host(h, sizes[5] + sizes[6]);
    if (!actual || !again) exit(1);
    if (pipeline) {
        run_attention_pipeline(h, m, ctx, nw, cap, lengths, kvptr, host, gpu, dcsr, part, lse, out, dmetadata,
            &kernel, args, sizeof(args), actual, again, output, pipeline);
    } else for (unsigned run = 0; run < 2; run++) {
        memset(again, 0xff, sizes[5] + sizes[6]);
        check_hsa(plow_hsa_copy_h2d(h, 0, part, again, sizes[5]));
        check_hsa(plow_hsa_copy_h2d(h, 0, lse, (char*)again + sizes[5], sizes[6]));
        check_hsa(plow_hsa_launch(h, 0, &kernel, nc * 256, 1, 1, 256, 1, 1, 0, args, sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
        float* dst = run ? again : actual;
        check_hsa(plow_hsa_copy_d2h(h, 0, dst, part, sizes[5]));
        check_hsa(plow_hsa_copy_d2h(h, 0, (char*)dst + sizes[5], lse, sizes[6]));
    }
    size_t mismatch = 0; int finite = 1;
    for (size_t i = 0; i < (sizes[5] + sizes[6]) / 4; i++) {
        const float* ref = i < sizes[5] / 4 ? (float*)host[5] + i : (float*)host[6] + i - sizes[5] / 4;
        const size_t slot = i < sizes[5] / 4 ? i / (16 * 512) : (i - sizes[5] / 4) / 16;
        if (slot < npartial) finite &= isfinite(actual[i]) && isfinite(*ref) && isfinite(again[i]);
        else {
            uint32_t bits; memcpy(&bits, ref, 4);
            if (bits != UINT32_MAX) exit(1);
        }
        mismatch += memcmp(actual + i, ref, 4) != 0;
    }
    const int stable = memcmp(actual, again, sizes[5] + sizes[6]) == 0;
    printf("attention-ps-replay M=%u ctx=%u works=%u partials=%u finite=%d mismatches=%zu repeat-bitwise=%d\n",
           m, ctx, nw, npartial, finite, mismatch, stable);
    if (!finite || mismatch || !stable) fails++;
    f = fopen(output, "wbx");
    const uint32_t shape[2] = {nw, 16};
    if (!f || fwrite(shape, sizeof(shape), 1, f) != 1
            || fwrite(actual, 1, sizes[5] + sizes[6], f) != sizes[5] + sizes[6] || fclose(f)) exit(1);
    for (unsigned i = 0; i < 7; i++) plow_hsa_free(h, host[i]);
    for (unsigned i = 0; i < 5; i++) plow_hsa_free(h, gpu[i]);
    void* allocations[] = {csr, dcsr, part, lse, out, metadata, dmetadata, actual, again};
    for (unsigned i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++) plow_hsa_free(h, allocations[i]);
}

static void replay_attention_reduce(plow_hsa* h, const char* partials, const char* metadata,
                                    const char* output) {
    FILE* f = fopen(metadata, "rb");
    uint32_t header[6];
    if (!f || fread(header, sizeof(header), 1, f) != 1) { perror(metadata); exit(1); }
    const unsigned m = header[1], nw = header[2], cap = header[3], nr = header[4];
    if (header[0] != 0x41505231 || !m || m >= 32 || !nw || nw > cap || cap > 256 + m
            || nr != m || header[5] > 256) {
        fprintf(stderr, "invalid persistent reducer geometry\n"); exit(1);
    }
    char name[64]; uint32_t cus, lds;
    check_hsa(plow_hsa_device_info(h, 0, name, &cus, &lds));
    if (strcmp(name, "gfx950") || cus != 256) exit(1);
    const size_t sizes[] = {(nr + 1) * 4, nr * 2 * 4, cap * 4, (size_t)m * 16 * 512 * 2,
        (size_t)nw * 16 * 512 * 4, (size_t)nw * 16 * 4};
    void* host[6]; void* gpu[6];
    for (unsigned i = 0; i < 6; i++) {
        host[i] = plow_hsa_alloc_host(h, sizes[i]);
        gpu[i] = plow_hsa_alloc(h, 0, sizes[i]);
        if (!host[i] || !gpu[i]) exit(1);
        if (i < 4 && fread(host[i], 1, sizes[i], f) != sizes[i]) exit(1);
    }
    if (fgetc(f) != EOF || ferror(f) || fclose(f)) exit(1);
    f = fopen(partials, "rb"); uint32_t shape[2];
    if (!f || fread(shape, sizeof(shape), 1, f) != 1 || shape[0] != nw || shape[1] != 16
            || fread(host[4], 1, sizes[4], f) != sizes[4]
            || fread(host[5], 1, sizes[5], f) != sizes[5]
            || fgetc(f) != EOF || ferror(f) || fclose(f)) {
        fprintf(stderr, "invalid reducer partial output file\n"); exit(1);
    }
    const int32_t* ptr = host[0]; const int32_t* final = host[1]; const int32_t* map = host[2];
    unsigned seen[288] = {0}, rows[32] = {0};
    if (ptr[0] != 0 || ptr[nr] != (int32_t)nw) exit(1);
    for (unsigned i = 0; i < nr; i++) {
        if (ptr[i] < 0 || ptr[i] > (int32_t)nw || ptr[i + 1] > (int32_t)nw
                || ptr[i + 1] < ptr[i] + 2
                || final[2 * i] < 0 || final[2 * i] >= (int32_t)m
                || final[2 * i + 1] != final[2 * i] + 1 || rows[final[2 * i]]++) exit(1);
    }
    for (unsigned i = 0; i < nw; i++)
        if (map[i] < 0 || map[i] >= (int32_t)nw || seen[map[i]]++) exit(1);
    for (unsigned i = 4; i < 6; i++)
        for (size_t j = 0; j < sizes[i] / 4; j++)
            if (!isfinite(((float*)host[i])[j])) exit(1);
    for (unsigned i = 0; i < 6; i++)
        if (i != 3) check_hsa(plow_hsa_copy_h2d(h, 0, gpu[i], host[i], sizes[i]));
    // Two kernel parameters: an 80-byte struct, then a four-byte LDS configuration.
    uint64_t args[11] = {0};
    args[0] = (uintptr_t)gpu[0]; args[1] = (uintptr_t)gpu[1]; args[2] = (uintptr_t)gpu[2];
    args[4] = (uintptr_t)gpu[3]; args[5] = (uintptr_t)gpu[5]; args[6] = (uintptr_t)gpu[4];
    uint32_t strides[4] = {16 * 512, 512, 256, nr};
    memcpy((char*)args + 56, strides, sizeof(strides));
    ((unsigned char*)args)[73] = 1;
    args[10] = 256;
    plow_hsa_kernel kernel;
    check_hsa(plow_hsa_get_kernel(h, 0,
        "_Z16kn_mla_reduce_v1I23MlaReduceKernelV1TraitsILi512ELi16ELi1EEfDF16bEv23MlaReduceKernelV1Params24MlaReduceKernelV1Configs", &kernel));
    if (kernel.kernarg_size != 84 || kernel.group_segment_size || kernel.private_segment_size) {
        fprintf(stderr, "persistent reducer ABI differs\n"); exit(1);
    }
    bf16* actual = plow_hsa_alloc_host(h, sizes[3]);
    bf16* again = plow_hsa_alloc_host(h, sizes[3]);
    if (!actual || !again) exit(1);
    for (unsigned run = 0; run < 2; run++) {
        memset(again, 0xff, sizes[3]);
        check_hsa(plow_hsa_copy_h2d(h, 0, gpu[3], again, sizes[3]));
        check_hsa(plow_hsa_launch(h, 0, &kernel, 16 * 128, 1, nr, 128, 1, 1, 2048, args, 84));
        check_hsa(plow_hsa_wait(h, 0));
        check_hsa(plow_hsa_copy_d2h(h, 0, run ? again : actual, gpu[3], sizes[3]));
    }
    size_t mismatch = 0; int finite = 1;
    for (size_t i = 0; i < sizes[3] / 2; i++) {
        finite &= isfinite(bf2f(actual[i])) && isfinite(bf2f(((bf16*)host[3])[i])) && isfinite(bf2f(again[i]));
        mismatch += actual[i] != ((bf16*)host[3])[i];
    }
    const int stable = memcmp(actual, again, sizes[3]) == 0;
    printf("attention-reduce-replay M=%u works=%u finite=%d mismatches=%zu repeat-bitwise=%d\n",
           m, nw, finite, mismatch, stable);
    if (!finite || mismatch || !stable) fails++;
    f = fopen(output, "wbx");
    if (!f || fwrite(actual, 1, sizes[3], f) != sizes[3] || fclose(f)) exit(1);
    for (unsigned i = 0; i < 6; i++) { plow_hsa_free(h, host[i]); plow_hsa_free(h, gpu[i]); }
    plow_hsa_free(h, actual); plow_hsa_free(h, again);
}

static void replay_attention_metadata(plow_hsa* h, const char* input, const char* reduction) {
    FILE* f = fopen(input, "rb"); uint32_t header[7];
    if (!f || fread(header, sizeof(header), 1, f) != 1) exit(1);
    const unsigned m = header[1], ctx = header[2], nw = header[4], cap = header[6];
    const unsigned live = ctx < 2048 ? ctx : 2048;
    if (header[0] != 0x41505331 || !m || m >= 32 || !ctx || ctx > 131072
            || header[3] != 2048 || header[5] != 256 || !nw || nw > cap || cap > 256 + m) exit(1);
    if (fseek(f, (long)((size_t)m * 16 * 576 * 2 + (size_t)m * ctx * 576 * 2 + m * live * 4), SEEK_CUR)) exit(1);
    const size_t sizes[] = {80, 257 * 4, cap * 8 * 4, (m + 1) * 4, m * 2 * 4, cap * 4};
    void* expected[6]; void* actual[6]; void* gpu[6];
    for (unsigned i = 0; i < 6; i++) {
        expected[i] = plow_hsa_alloc_host(h, sizes[i]); actual[i] = plow_hsa_alloc_host(h, sizes[i]);
        gpu[i] = plow_hsa_alloc(h, 0, sizes[i]);
        if (!expected[i] || !actual[i] || !gpu[i]) exit(1);
        if ((i == 1 || i == 2) && fread(expected[i], 1, sizes[i], f) != sizes[i]) exit(1);
    }
    if (fclose(f)) exit(1);
    f = fopen(reduction, "rb"); uint32_t reduce_header[6];
    if (!f || fread(reduce_header, sizeof(reduce_header), 1, f) != 1
            || reduce_header[0] != 0x41505231 || reduce_header[1] != m
            || reduce_header[2] != nw || reduce_header[3] != cap || reduce_header[4] != m) exit(1);
    for (unsigned i = 3; i < 6; i++) if (fread(expected[i], 1, sizes[i], f) != sizes[i]) exit(1);
    if (fclose(f)) exit(1);
    unsigned* csr = plow_hsa_alloc_host(h, (3 * m + 2) * 4);
    void* dcsr = plow_hsa_alloc(h, 0, (3 * m + 2) * 4);
    if (!csr || !dcsr) exit(1);
    for (unsigned b = 0; b <= m; b++) { csr[b] = b; csr[m + 1 + b] = b * live; }
    for (unsigned b = 0; b < m; b++) csr[2 * m + 2 + b] = 1;
    check_hsa(plow_hsa_copy_h2d(h, 0, dcsr, csr, (3 * m + 2) * 4));
    unsigned split_cap = 1;
    while (split_cap < live / 128) split_cap *= 2;
    unsigned splits = split_cap * m;
    if (splits > 256) splits = 256;
    uint64_t args[17] = {0};
    for (unsigned i = 0; i < 6; i++) args[i] = (uintptr_t)gpu[i];
    args[6] = (uintptr_t)dcsr; args[7] = (uintptr_t)dcsr + (m + 1) * 4;
    args[8] = (uintptr_t)dcsr + (2 * m + 2) * 4;
    const uint32_t dims[13] = {m, 0, 16, 256, m + 1, 1, 16, 4, 1, 1, UINT32_MAX, 1, splits};
    memcpy((char*)args + 72, dims, sizeof(dims));
    ((unsigned char*)args)[124] = 1;
    const uint32_t tail[2] = {16, 1};
    memcpy((char*)args + 128, tail, sizeof(tail));
    plow_hsa_kernel kernel;
    check_hsa(plow_hsa_get_kernel(h, 0,
        "_Z33kn_get_mla_metadata_v1_2_parallelI20MlaMetadataV12TraitsILi128ELb0ELi1ELb1ELb0EEEv28MlaMetadataV1KernelParameter", &kernel));
    if (kernel.kernarg_size != 392 || kernel.group_segment_size || kernel.private_segment_size) {
        fprintf(stderr, "metadata kernel ABI mismatch\n"); exit(1);
    }
    char name[64]; uint32_t cus, lds;
    check_hsa(plow_hsa_device_info(h, 0, name, &cus, &lds));
    if (strcmp(name, "gfx950") || cus != 256 || lds != 163840) {
        fprintf(stderr, "metadata device mismatch: %s CU=%u LDS=%u\n", name, cus, lds); exit(1);
    }
    for (unsigned run = 0; run < 2; run++) {
        for (unsigned i = 0; i < 6; i++) {
            memset(actual[i], 0xff, sizes[i]);
            check_hsa(plow_hsa_copy_h2d(h, 0, gpu[i], actual[i], sizes[i]));
        }
        check_hsa(plow_hsa_launch(h, 0, &kernel, 512, 1, 1, 512, 1, 1, lds, args, sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
        for (unsigned i = 0; i < 6; i++) check_hsa(plow_hsa_copy_d2h(h, 0, actual[i], gpu[i], sizes[i]));
        size_t mismatch = memcmp(actual[1], expected[1], sizes[1]) != 0;
        mismatch += ((uint64_t*)actual[0])[0] != (uintptr_t)gpu[1];
        mismatch += ((uint64_t*)actual[0])[1] != (uintptr_t)gpu[2];
        for (unsigned i = 0; i < nw; i++)
            mismatch += memcmp((int32_t*)actual[2] + i * 8, (int32_t*)expected[2] + i * 8, 7 * 4) != 0;
        mismatch += memcmp(actual[3], expected[3], sizes[3]) != 0;
        const int32_t* ptr = expected[3];
        if (ptr[m] < 0 || ptr[m] > (int32_t)cap) exit(1);
        for (unsigned b = 0; b < m; b++) {
            if (ptr[b] < 0 || ptr[b + 1] < ptr[b]) exit(1);
            if (ptr[b] != ptr[b + 1])
                mismatch += memcmp((int32_t*)actual[4] + b * 2, (int32_t*)expected[4] + b * 2, 8) != 0;
        }
        mismatch += memcmp(actual[5], expected[5], ptr[m] * 4) != 0;
        printf("attention-metadata-replay M=%u ctx=%u works=%u run=%u mismatches=%zu\n", m, ctx, nw, run, mismatch);
        if (mismatch) fails++;
    }
    for (unsigned i = 0; i < 6; i++) { plow_hsa_free(h, expected[i]); plow_hsa_free(h, actual[i]); plow_hsa_free(h, gpu[i]); }
    plow_hsa_free(h, csr); plow_hsa_free(h, dcsr);
}

static void test_attention_bf16_adapter(plow_hsa* h, unsigned m, unsigned ctx) {
    const unsigned live_cap = ctx < 2048 ? ctx : 2048;
    const size_t ins[] = {(size_t)m * 8 * 512 * 2, (size_t)m * 8 * 64 * 2,
        (size_t)m * ctx * 512 * 2, (size_t)m * ctx * 64 * 2, (size_t)m * 2048 * 4, m * 4};
    const size_t outs[] = {(size_t)m * 16 * 576 * 2, (size_t)m * live_cap * 576 * 2,
        (size_t)m * live_cap * 4, (m + 1) * 4, (m + 1) * 4, m * 4,
        (size_t)m * 8 * 512 * 2, (size_t)m * 16 * 512 * 2};
    void* in[6]; void* din[6]; void* out[8]; void* dout[8];
    for (unsigned i = 0; i < 6; i++) {
        in[i] = plow_hsa_alloc_host(h, ins[i]); din[i] = plow_hsa_alloc(h, 0, ins[i]);
        if (!in[i] || !din[i]) exit(1);
        if (i < 4) {
            for (size_t j = 0; j < ins[i] / 2; j++) ((bf16*)in[i])[j] = 0x3e00 + (j * 13 + i * 71) % 997;
            check_hsa(plow_hsa_copy_h2d(h, 0, din[i], in[i], ins[i]));
        }
    }
    for (unsigned i = 0; i < 8; i++) {
        out[i] = plow_hsa_alloc_host(h, outs[i] + 256); dout[i] = plow_hsa_alloc(h, 0, outs[i] + 256);
        if (!out[i] || !dout[i]) exit(1);
    }
    plow_hsa_kernel pack, unpad;
    check_hsa(plow_hsa_get_kernel(h, 0, "plow_mla_bf16_pack", &pack));
    check_hsa(plow_hsa_get_kernel(h, 0, "plow_mla_bf16_unpad", &unpad));
    uint64_t args[14] = {(uintptr_t)dout[0], (uintptr_t)dout[1], (uintptr_t)dout[2],
        (uintptr_t)din[0], (uintptr_t)din[1], (uintptr_t)din[2], (uintptr_t)din[3],
        (uintptr_t)din[4], (uintptr_t)din[5], (uintptr_t)dout[3], (uintptr_t)dout[4], (uintptr_t)dout[5]};
    uint32_t dims[4] = {m, ctx, 0, 0};
    uint64_t unpad_args[3] = {(uintptr_t)dout[6], (uintptr_t)dout[7], m};
    for (unsigned mode = 0; mode < 4; mode++) {
        unsigned prefix[65] = {0};
        int32_t* len = in[5]; int32_t* idx = in[4];
        memset(idx, 0xff, ins[4]);
        for (unsigned b = 0; b < m; b++) {
            len[b] = mode < 2 ? ctx : mode == 2 ? (b % 5 ? ((b + 1) * 37 < ctx ? (b + 1) * 37 : ctx) : 0)
                : b % 4 == 0 ? 0 : b % 4 == 1 ? 1 : b % 4 == 2 ? (ctx < 127 ? ctx : 127) : ctx;
            const unsigned live = len[b] < 2048 ? len[b] : 2048;
            prefix[b + 1] = prefix[b] + live;
            for (unsigned j = 0; j < live; j++)
                idx[b * 2048 + j] = mode ? (len[b] - 1 - (size_t)j * len[b] / live) : j;
        }
        for (unsigned i = 4; i < 6; i++) check_hsa(plow_hsa_copy_h2d(h, 0, din[i], in[i], ins[i]));
        for (unsigned i = 0; i < 8; i++) {
            memset(out[i], 0xff, outs[i] + 256);
            if (i == 7) for (size_t j = 0; j < outs[i] / 2; j++) ((bf16*)out[i])[j] = 0x3e00 + (j * 17) % 997;
            check_hsa(plow_hsa_copy_h2d(h, 0, dout[i], out[i], outs[i] + 256));
        }
        dims[2] = mode ? 2048 : 0;
        memcpy(args + 12, dims, sizeof(dims));
        check_hsa(plow_hsa_launch(h, 0, &pack, 256 * 256, 1, 1, 256, 1, 1, 0, args, sizeof(args)));
        check_hsa(plow_hsa_launch(h, 0, &unpad, 16 * 256, 1, 1, 256, 1, 1, 0, unpad_args, sizeof(unpad_args)));
        check_hsa(plow_hsa_wait(h, 0));
        for (unsigned i = 0; i < 7; i++) check_hsa(plow_hsa_copy_d2h(h, 0, out[i], dout[i], outs[i] + 256));
        size_t mismatch = 0;
        for (unsigned b = 0; b <= m; b++) {
            mismatch += ((unsigned*)out[3])[b] != b;
            mismatch += ((unsigned*)out[4])[b] != prefix[b];
            if (b == m) continue;
            mismatch += ((unsigned*)out[5])[b] != 1;
            for (unsigned head = 0; head < 16; head++) {
                const size_t dst = ((size_t)b * 16 + head) * 576, src = (size_t)b * 8 + head / 2;
                mismatch += memcmp((bf16*)out[0] + dst, (bf16*)in[0] + src * 512, 512 * 2) != 0;
                mismatch += memcmp((bf16*)out[0] + dst + 512, (bf16*)in[1] + src * 64, 64 * 2) != 0;
            }
            for (unsigned j = 0; j < prefix[b + 1] - prefix[b]; j++) {
                const unsigned slot = prefix[b] + j;
                const size_t src = (size_t)b * ctx + idx[b * 2048 + j];
                mismatch += ((unsigned*)out[2])[slot] != slot;
                mismatch += memcmp((bf16*)out[1] + (size_t)slot * 576, (bf16*)in[2] + src * 512, 512 * 2) != 0;
                mismatch += memcmp((bf16*)out[1] + (size_t)slot * 576 + 512, (bf16*)in[3] + src * 64, 64 * 2) != 0;
            }
        }
        for (unsigned i = 0; i < m * 8; i++)
            mismatch += memcmp((bf16*)out[6] + (size_t)i * 512, (bf16*)out[7] + (size_t)i * 1024, 512 * 2) != 0;
        for (unsigned i = 0; i < 7; i++) {
            const size_t used = i == 1 ? (size_t)prefix[m] * 576 * 2 : i == 2 ? prefix[m] * 4 : outs[i];
            for (size_t j = used; j < outs[i] + 256; j++) mismatch += ((unsigned char*)out[i])[j] != 0xff;
        }
        printf("attention-bf16-adapter M=%u ctx=%u mode=%u keys=%u mismatches=%zu\n", m, ctx, mode, prefix[m], mismatch);
        if (mismatch) fails++;
    }
    for (unsigned i = 0; i < 6; i++) { plow_hsa_free(h, in[i]); plow_hsa_free(h, din[i]); }
    for (unsigned i = 0; i < 8; i++) { plow_hsa_free(h, out[i]); plow_hsa_free(h, dout[i]); }
}

static void run_quant128(plow_hsa* h, plow_hsa_kernel* kernel, unsigned M, unsigned K) {
    const size_t count = (size_t)M * K, groups = count / 128;
    bf16* x = plow_hsa_alloc_host(h, count * 2);
    unsigned char* q = plow_hsa_alloc_host(h, count);
    float* scales = plow_hsa_alloc_host(h, groups * 4);
    void* dx = plow_hsa_alloc(h, 0, count * 2);
    void* dq = plow_hsa_alloc(h, 0, count);
    void* ds = plow_hsa_alloc(h, 0, groups * 4);
    if (!x || !q || !scales || !dx || !dq || !ds) exit(1);
    for (size_t i = 0; i < count; i++) {
        unsigned group = i / 128, lane = i % 128;
        float v = ((int)(rand() % 1025) - 512) / 128.0f;
        switch (group % 6) {
        case 0: v = 0; break;
        case 1: v *= 1e-10f; break;
        case 2: v *= 1e-20f; break;
        case 3: v = ldexpf(v, (int)(group % 241) - 120); break;
        case 4: v = ((int)lane - 64) / 16.0f; break;
        }
        x[i] = f2bf(v);
    }
    check_hsa(plow_hsa_copy_h2d(h, 0, dx, x, count * 2));
    struct __attribute__((packed)) {
        void* q; const void* x; void* scale; unsigned m, k;
    } args = {dq, dx, ds, M, K};
    check_hsa(plow_hsa_launch(h, 0, kernel, 256 * PLOW_WG_THREADS, 1, 1,
                             PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)));
    check_hsa(plow_hsa_wait(h, 0));
    check_hsa(plow_hsa_copy_d2h(h, 0, q, dq, count));
    check_hsa(plow_hsa_copy_d2h(h, 0, scales, ds, groups * 4));
    char path[4096];
    int len = snprintf(path, sizeof(path), "%s/%u_%u.quant", a8w8_capture, M, K);
    if (len < 0 || (size_t)len >= sizeof(path)) exit(1);
    FILE* f = fopen(path, "wbx");
    if (!f) { perror(path); exit(1); }
    const uint32_t header[] = {M, K};
    const void* parts[] = {header, x, q, scales};
    const size_t sizes[] = {sizeof(header), count * 2, count, groups * 4};
    for (unsigned i = 0; i < sizeof(parts) / sizeof(parts[0]); i++)
        if (fwrite(parts[i], 1, sizes[i], f) != sizes[i]) exit(1);
    if (fclose(f)) exit(1);
    printf("quant128 M=%u K=%u captured (reference comparison required)\n", M, K);
    void* allocations[] = {x, q, scales, dx, dq, ds};
    for (unsigned i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++)
        plow_hsa_free(h, allocations[i]);
}

static void run_a8w8_block_parts(plow_hsa* h, plow_hsa_kernel* kernel, unsigned M, unsigned N,
                                unsigned K, unsigned parts, unsigned glu) {
    if (parts != 1 && (parts != 4 || K % 512)) exit(1);
    if (glu && parts != 1) exit(1);
    const unsigned KB = (K + 127u) / 128u, NB = (N + 127u) / 128u;
    const size_t na = (size_t)M * K, nw = (size_t)N * K, nc = (size_t)M * N;
    unsigned char* a = plow_hsa_alloc_host(h, na);
    unsigned char* w = plow_hsa_alloc_host(h, nw * (1 + glu));
    float* as = plow_hsa_alloc_host(h, (size_t)KB * M * 4);
    float* ws = plow_hsa_alloc_host(h, (size_t)NB * KB * 4 * (1 + glu));
    bf16* out = plow_hsa_alloc_host(h, nc * parts * 2);
    float* af = malloc(na * sizeof(float));
    float* wf = malloc(nw * sizeof(float) * (1 + glu));
    if (!a || !w || !as || !ws || !out || !af || !wf) exit(1);
    for (size_t i = 0; i < na; i++) {
        a[i] = (unsigned char)((rand() % 2) * 128 + rand() % 64);
        if (M > 1 && i / K == M - 1) a[i] = 0;
        af[i] = (float)e4m3_decode(a[i]);
    }
    for (size_t i = 0; i < nw * (1 + glu); i++) {
        w[i] = (unsigned char)((rand() % 2) * 128 + rand() % 64);
        wf[i] = (float)e4m3_decode(w[i]);
    }
    for (size_t i = 0; i < (size_t)KB * M; i++) as[i] = 0.013f + 0.007f * (i % 19);
    for (size_t i = 0; i < (size_t)NB * KB * (1 + glu); i++) ws[i] = 0.009f + 0.003f * (i % 23);
    void* da = plow_hsa_alloc(h, 0, na);
    void* dw = plow_hsa_alloc(h, 0, nw * (1 + glu));
    void* das = plow_hsa_alloc(h, 0, (size_t)KB * M * 4);
    void* dws = plow_hsa_alloc(h, 0, (size_t)NB * KB * 4 * (1 + glu));
    void* dc = plow_hsa_alloc(h, 0, nc * parts * 2);
    if (!da || !dw || !das || !dws || !dc) exit(1);
    check_hsa(plow_hsa_copy_h2d(h, 0, da, a, na));
    check_hsa(plow_hsa_copy_h2d(h, 0, dw, w, nw * (1 + glu)));
    check_hsa(plow_hsa_copy_h2d(h, 0, das, as, (size_t)KB * M * 4));
    check_hsa(plow_hsa_copy_h2d(h, 0, dws, ws, (size_t)NB * KB * 4 * (1 + glu)));
    struct __attribute__((packed)) {
        void* c; const void* a; const void* w; const void* as; const void* ws;
        unsigned m, n, k;
    } args = {dc, da, dw, das, dws, M, N, K};
    check_hsa(plow_hsa_launch(h, 0, kernel, 256 * PLOW_WG_THREADS, 1, 1,
                             PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)));
    check_hsa(plow_hsa_wait(h, 0));
    check_hsa(plow_hsa_copy_d2h(h, 0, out, dc, nc * parts * 2));
    double worst = 0.0;
    int finite = 1;
    for (unsigned part = 0; part < parts; part++) for (unsigned m = 0; m < M; m++) {
        double error = 0.0, norm = 0.0;
        for (unsigned n = 0; n < N; n++) {
            double want = 0.0, up = 0.0;
            for (unsigned group = part * (KB / parts); group < (part + 1) * (KB / parts); group++) {
                double dot = 0.0, udot = 0.0;
                const unsigned end = (group + 1) * 128 < K ? (group + 1) * 128 : K;
                for (unsigned k = group * 128; k < end; k++) {
                    dot += (double)af[(size_t)m * K + k] * wf[(size_t)n * K + k];
                    if (glu) udot += (double)af[(size_t)m * K + k] * wf[nw + (size_t)n * K + k];
                }
                want += dot * as[(size_t)group * M + m] * ws[(size_t)(n / 128) * KB + group];
                if (glu) up += udot * as[(size_t)group * M + m] * ws[(size_t)(NB + n / 128) * KB + group];
            }
            if (glu) want = (want / (1.0 + exp(-want))) * up;
            const double got = bf2f(out[(size_t)part * nc + (size_t)m * N + n]);
            finite &= isfinite(got);
            error += (got - want) * (got - want);
            norm += want * want;
        }
        const double relative = sqrt(error / fmax(norm, 1e-30));
        finite &= isfinite(relative);
        worst = fmax(worst, relative);
    }
    const int ok = finite && worst < 0.004;
    printf("a8w8 block128 M=%u N=%u K=%u parts=%u glu=%u %s max-row-rel-L2=%.6f\n", M, N, K, parts, glu,
           ok ? "PASS" : "FAIL", worst);
    fails += !ok;
    if (parts == 1 && !glu) capture_a8w8(M, N, K, a, w, as, ws, out);
    void* allocations[] = {a, w, as, ws, out, da, dw, das, dws, dc};
    for (unsigned i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++)
        plow_hsa_free(h, allocations[i]);
    free(af);
    free(wf);
}

static void run_a8w8_block(plow_hsa* h, plow_hsa_kernel* kernel, unsigned M, unsigned N, unsigned K) {
    run_a8w8_block_parts(h, kernel, M, N, K, 1, 0);
}

static void replay_a8w8(plow_hsa* h, plow_hsa_kernel* kernel, const char* input, const char* output,
                        unsigned glu, unsigned weighted, unsigned n_first, unsigned n_second) {
    FILE* f = fopen(input, "rb");
    uint32_t dims[3];
    if (!f || fread(dims, sizeof(dims), 1, f) != 1) exit(1);
    const unsigned M = dims[0], N = dims[1], K = dims[2];
    const unsigned split = n_first || n_second;
    if (split && (glu || weighted || !n_first || !n_second || (uint64_t)n_first + n_second >= N)) exit(1);
    if (!M || !N || !K || M > (1u << 20) || N > (1u << 20) || K > (1u << 20)) exit(1);
    const size_t kb = (K + 127u) / 128u, nb = (N + 127u) / 128u;
    const size_t sizes[] = {(size_t)M * K, (size_t)N * K * (1 + glu), kb * M * 4,
                            nb * kb * 4 * (1 + glu), (size_t)M * N * 2, (size_t)M * 4};
    const unsigned count = weighted ? 6 : 5;
    size_t total = sizeof(dims);
    for (unsigned i = 0; i < count; i++) total += sizes[i];
    if (fseek(f, 0, SEEK_END) || ftell(f) != (long)total || fseek(f, sizeof(dims), SEEK_SET)) exit(1);
    void* host[6];
    void* device[6] = {0};
    for (unsigned i = 0; i < count; i++) {
        host[i] = plow_hsa_alloc_host(h, sizes[i]);
        device[i] = plow_hsa_alloc(h, 0, sizes[i]);
        if (!host[i] || !device[i] || fread(host[i], 1, sizes[i], f) != sizes[i]) exit(1);
        if (i != 4) check_hsa(plow_hsa_copy_h2d(h, 0, device[i], host[i], sizes[i]));
    }
    fclose(f);
    bf16* out = plow_hsa_alloc_host(h, sizes[4]);
    bf16* repeat = plow_hsa_alloc_host(h, sizes[4]);
    if (!out || !repeat) exit(1);
    struct __attribute__((packed)) {
        void* c; const void* a; const void* w; const void* as; const void* ws;
        unsigned m, n, k;
    } args = {device[4], device[0], device[1], device[2], device[3], M, N, K};
    struct __attribute__((packed)) {
        void* c; const void* a; const void* w; const void* as; const void* ws; const void* weights;
        unsigned m, n, k;
    } weighted_args = {device[4], device[0], device[1], device[2], device[3], device[5], M, N, K};
    struct __attribute__((packed)) {
        void* c; void* c1; void* c2; const void* a; const void* w; const void* as; const void* ws;
        unsigned m, n, k, n_first, n_second;
    } split_args = {device[4], (bf16*)device[4] + (size_t)M * n_first,
        (bf16*)device[4] + (size_t)M * (n_first + n_second), device[0], device[1], device[2], device[3],
        M, N, K, n_first, n_second};
    for (unsigned run = 0; run < 2; run++) {
        check_hsa(plow_hsa_launch(h, 0, kernel, 256 * PLOW_WG_THREADS, 1, 1,
                                 PLOW_WG_THREADS, 1, 1, 0,
                                 split ? (void*)&split_args : weighted ? (void*)&weighted_args : (void*)&args,
                                 split ? sizeof(split_args) : weighted ? sizeof(weighted_args) : sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
        check_hsa(plow_hsa_copy_d2h(h, 0, run ? repeat : out, device[4], sizes[4]));
        if (split) {
            bf16* target = run ? repeat : out;
            bf16* packed = malloc(sizes[4]);
            if (!packed) exit(1);
            memcpy(packed, target, sizes[4]);
            const unsigned widths[] = {n_first, n_second, N - n_first - n_second};
            unsigned col = 0;
            for (unsigned part = 0; part < 3; part++) {
                for (unsigned row = 0; row < M; row++)
                    memcpy(target + (size_t)row * N + col, packed + (size_t)M * col + (size_t)row * widths[part],
                           widths[part] * sizeof(bf16));
                col += widths[part];
            }
            free(packed);
        }
    }
    double worst = 0;
    size_t changed = 0;
    int finite = 1, stable = !memcmp(out, repeat, sizes[4]);
    for (unsigned m = 0; m < M; m++) {
        double error = 0, norm = 0;
        for (unsigned n = 0; n < N; n++) {
            const size_t i = (size_t)m * N + n;
            const double got = bf2f(out[i]), want = bf2f(((bf16*)host[4])[i]);
            finite &= isfinite(got) && isfinite(want);
            changed += out[i] != ((bf16*)host[4])[i];
            error += (got - want) * (got - want);
            norm += want * want;
        }
        const double rel = sqrt(error / fmax(norm, 1e-30));
        finite &= isfinite(rel);
        worst = fmax(worst, rel);
    }
    f = fopen(output, "wbx");
    if (!f || fwrite(out, 1, sizes[4], f) != sizes[4] || fclose(f)) exit(1);
    const int ok = finite && stable && worst < 0.004;
    printf("replay %s M=%u N=%u K=%u %s changed=%zu repeat-bitwise=%d max-row-rel-L2=%.10g\n",
           input, M, N, K, ok ? "PASS" : "FAIL", changed, stable, worst);
    fails += !ok;
    for (unsigned i = 0; i < count; i++) { plow_hsa_free(h, host[i]); plow_hsa_free(h, device[i]); }
    plow_hsa_free(h, out);
    plow_hsa_free(h, repeat);
}

static void replay_mla(plow_hsa* h, plow_hsa_kernel* kernel, const char* input, const char* output,
                       const char* qb_path) {
    FILE* f = fopen(input, "rb");
    uint32_t dims[4];
    if (!f || fread(dims, sizeof(dims), 1, f) != 1) exit(1);
    const unsigned M = dims[0], H = dims[1], N = dims[2], K = dims[3];
    if (!M || M > 128 || !H || H > 64 || !N || N > 512 || !K || K > 512 || K % 16) exit(1);
    const size_t sizes[] = {(size_t)M * H * K * 2, (size_t)H * N * K, 4, (size_t)M * H * N * 2};
    const size_t total = sizeof(dims) + sizes[0] + sizes[1] + sizes[2] + sizes[3];
    if (fseek(f, 0, SEEK_END) || ftell(f) != (long)total || fseek(f, sizeof(dims), SEEK_SET)) exit(1);
    void* host[4];
    void* device[4];
    for (unsigned i = 0; i < 4; i++) {
        host[i] = plow_hsa_alloc_host(h, sizes[i]);
        device[i] = plow_hsa_alloc(h, 0, sizes[i]);
        if (!host[i] || !device[i] || fread(host[i], 1, sizes[i], f) != sizes[i]) exit(1);
        if (i != 3) check_hsa(plow_hsa_copy_h2d(h, 0, device[i], host[i], sizes[i]));
    }
    fclose(f);
    bf16 *qx = NULL, *rope = NULL, *rope_device = NULL;
    const size_t rope_bytes = (size_t)M * H * 64 * 2;
    if (qb_path) {
        if (K != 192 || N != 512 || H != 8) exit(1);
        f = fopen(qb_path, "rb");
        uint32_t qdims[3];
        if (!f || fread(qdims, sizeof(qdims), 1, f) != 1 || qdims[0] != M || qdims[1] != H * 256 || qdims[2] != 2048) exit(1);
        const size_t qbytes = (size_t)M * H * 256 * 2;
        const size_t qtotal = 12 + (size_t)(M + H * 256) * 2048 + (size_t)(M + H * 2) * 16 * 4 + qbytes;
        qx = plow_hsa_alloc_host(h, qbytes);
        rope = plow_hsa_alloc_host(h, rope_bytes);
        rope_device = plow_hsa_alloc(h, 0, rope_bytes);
        if (!qx || !rope || !rope_device || fseek(f, 0, SEEK_END) || ftell(f) != (long)qtotal
                || fseek(f, -(long)qbytes, SEEK_END) || fread(qx, 1, qbytes, f) != qbytes) exit(1);
        fclose(f);
        for (unsigned row = 0; row < M * H; row++)
            if (memcmp(qx + (size_t)row * 256, (bf16*)host[0] + (size_t)row * K, K * 2)) exit(1);
        plow_hsa_free(h, device[0]);
        device[0] = plow_hsa_alloc(h, 0, qbytes);
        if (!device[0]) exit(1);
        check_hsa(plow_hsa_copy_h2d(h, 0, device[0], qx, qbytes));
    }
    if (!isfinite(*(float*)host[2]) || *(float*)host[2] <= 0) exit(1);
    bf16* out = plow_hsa_alloc_host(h, sizes[3]);
    bf16* repeat = plow_hsa_alloc_host(h, sizes[3]);
    if (!out || !repeat) exit(1);
    struct __attribute__((packed)) {
        void* c; const void* x; const void* w; const void* scale;
        unsigned m, heads, n, k;
    } args = {device[3], device[0], device[1], device[2], M, H, N, K};
    struct __attribute__((packed)) {
        void* c; void* rope; const void* x; const void* w; const void* scale;
        unsigned m, heads, n, k;
    } qargs = {device[3], rope_device, device[0], device[1], device[2], M, H, N, K};
    int rope_ok = 1;
    for (unsigned run = 0; run < 2; run++) {
        check_hsa(plow_hsa_launch(h, 0, kernel, 256 * PLOW_WG_THREADS, 1, 1,
            PLOW_WG_THREADS, 1, 1, 0, qb_path ? (void*)&qargs : (void*)&args, qb_path ? sizeof(qargs) : sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
        check_hsa(plow_hsa_copy_d2h(h, 0, run ? repeat : out, device[3], sizes[3]));
        if (qb_path) {
            check_hsa(plow_hsa_copy_d2h(h, 0, rope, rope_device, rope_bytes));
            for (unsigned row = 0; row < M * H; row++)
                rope_ok &= !memcmp(rope + (size_t)row * 64, qx + (size_t)row * 256 + 192, 64 * 2);
        }
    }
    double worst = 0;
    size_t changed = 0;
    int finite = 1, stable = !memcmp(out, repeat, sizes[3]);
    for (unsigned row = 0; row < M * H; row++) {
        double error = 0, norm = 0;
        for (unsigned n = 0; n < N; n++) {
            const size_t i = (size_t)row * N + n;
            const double got = bf2f(out[i]), want = bf2f(((bf16*)host[3])[i]);
            finite &= isfinite(got) && isfinite(want);
            changed += out[i] != ((bf16*)host[3])[i];
            error += (got - want) * (got - want);
            norm += want * want;
        }
        const double rel = sqrt(error / fmax(norm, 1e-30));
        finite &= isfinite(rel);
        worst = fmax(worst, rel);
    }
    f = fopen(output, "wbx");
    if (!f || fwrite(out, 1, sizes[3], f) != sizes[3] || fclose(f)) exit(1);
    const int ok = finite && stable && !changed && rope_ok;
    printf("mla-replay %s M=%u H=%u N=%u K=%u %s changed=%zu repeat-bitwise=%d max-row-rel-L2=%.10g rope-bitwise=%d\n",
        input, M, H, N, K, ok ? "PASS" : "FAIL", changed, stable, worst, rope_ok);
    if (qb_path) {
        char path[4096];
        if (snprintf(path, sizeof(path), "%s.rope", output) >= sizeof(path)) exit(1);
        f = fopen(path, "wbx");
        if (!f || fwrite(rope, 1, rope_bytes, f) != rope_bytes || fclose(f)) exit(1);
        plow_hsa_free(h, qx); plow_hsa_free(h, rope); plow_hsa_free(h, rope_device);
    }
    fails += !ok;
    for (unsigned i = 0; i < 4; i++) { plow_hsa_free(h, host[i]); plow_hsa_free(h, device[i]); }
    plow_hsa_free(h, out);
    plow_hsa_free(h, repeat);
}

static void bf16_sum_order_bounds(const bf16* parts, unsigned T, unsigned H, unsigned topk,
                                  float* lower, float* upper) {
    if (!topk || topk > 8) exit(1);
    const unsigned states = 1u << topk;
    for (unsigned t = 0; t < T; t++) {
        for (unsigned n = 0; n < H; n++) {
            float lo[256], hi[256], x[8];
            lo[0] = hi[0] = 0.0f;
            for (unsigned k = 0; k < topk; k++) {
                x[k] = bf2f(parts[((size_t)t * topk + k) * H + n]);
                if (!isfinite(x[k])) exit(1);
            }
            // Rounded addition is monotone: each subset needs only its extrema.
            for (unsigned mask = 1; mask < states; mask++) {
                float smallest = INFINITY, largest = -INFINITY;
                for (unsigned bits = mask; bits; bits &= bits - 1) {
                    const unsigned k = __builtin_ctz(bits), prior = mask ^ (1u << k);
                    const float a = bf2f(f2bf(lo[prior] + x[k])), b = bf2f(f2bf(hi[prior] + x[k]));
                    if (a < smallest) smallest = a;
                    if (b > largest) largest = b;
                }
                lo[mask] = smallest; hi[mask] = largest;
            }
            lower[(size_t)t * H + n] = lo[states - 1];
            upper[(size_t)t * H + n] = hi[states - 1];
        }
    }
}

static size_t check_bf16_bounds(const bf16* values, const float* lower, const float* upper, size_t n) {
    size_t bad = 0;
    for (size_t j = 0; j < n; j++) {
        const float x = bf2f(values[j]);
        bad += !isfinite(x) || x < lower[j] || x > upper[j];
    }
    return bad;
}

static void replay_qb_split(plow_hsa* h, const char* prefix, const char* output) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s.qb.bin", prefix) >= sizeof(path)) exit(1);
    FILE* f = fopen(path, "rb");
    uint32_t dims[3];
    if (!f || fread(dims, sizeof(dims), 1, f) != 1) exit(1);
    const unsigned M = dims[0], N = dims[1], K = dims[2];
    if (!(M == 1 || M == 8 || M == 16 || M == 32 || M == 64 || M == 128) || N != 2048 || K != 2048) exit(1);
    const unsigned splits = M <= 16 ? 8 : M <= 64 ? 4 : 1;
    const size_t elems = (size_t)M * N, bytes = elems * 2;
    const size_t sizes[] = {(size_t)M * K, (size_t)N * K, (size_t)M * (K / 128) * 4,
        (size_t)(N / 128) * (K / 128) * 4, bytes};
    const size_t total = 12 + sizes[0] + sizes[1] + sizes[2] + sizes[3] + sizes[4];
    if (fseek(f, 0, SEEK_END) || ftell(f) != (long)total || fseek(f, 12, SEEK_SET)) exit(1);
    void* host[5];
    void* device[5];
    for (unsigned i = 0; i < 5; i++) {
        host[i] = plow_hsa_alloc_host(h, sizes[i]);
        device[i] = plow_hsa_alloc(h, 0, sizes[i] * (i == 4 ? splits : 1));
        if (!host[i] || !device[i] || fread(host[i], 1, sizes[i], f) != sizes[i]) exit(1);
        if (i != 4) check_hsa(plow_hsa_copy_h2d(h, 0, device[i], host[i], sizes[i]));
    }
    fclose(f);
    bf16* expected = malloc(bytes * splits);
    bf16* interleaved = malloc(bytes * splits);
    bf16* parts = plow_hsa_alloc_host(h, bytes * splits);
    bf16* repeat = plow_hsa_alloc_host(h, bytes * splits);
    float* lower = malloc(elems * sizeof(float));
    float* upper = malloc(elems * sizeof(float));
    if (!expected || !interleaved || !parts || !repeat || !lower || !upper) exit(1);
    for (unsigned part = 0; part < splits; part++) {
        if (snprintf(path, sizeof(path), "%s.part%u.qb.bin", prefix, part) >= sizeof(path)) exit(1);
        f = fopen(path, "rb");
        uint32_t pd[3];
        if (!f || fread(pd, sizeof(pd), 1, f) != 1 || pd[0] != M || pd[1] != N || pd[2] != K / splits) exit(1);
        const size_t ps = 12 + (size_t)(M + N) * pd[2] + (size_t)(M + N / 128) * (pd[2] / 128) * 4 + bytes;
        if (fseek(f, 0, SEEK_END) || ftell(f) != (long)ps || fseek(f, -(long)bytes, SEEK_END)
                || fread(expected + part * elems, 1, bytes, f) != bytes) exit(1);
        fclose(f);
        for (unsigned m = 0; m < M; m++)
            memcpy(interleaved + ((size_t)m * splits + part) * N, expected + part * elems + (size_t)m * N, N * 2);
    }
    bf16_sum_order_bounds(interleaved, M, N, splits, lower, upper);
    size_t violations = check_bf16_bounds(host[4], lower, upper, elems);
    plow_hsa_kernel split_kernel, atomic_kernel;
    check_hsa(plow_hsa_get_kernel(h, 0, splits == 8 ? "gemm_qb_split8" : splits == 4 ? "gemm_qb_split4" : "gemm_a8w8_block128_m16", &split_kernel));
    check_hsa(plow_hsa_get_kernel(h, 0, splits == 8 ? "gemm_qb_atomic8" : splits == 4 ? "gemm_qb_atomic4" : "gemm_a8w8_block128_m16", &atomic_kernel));
    struct __attribute__((packed)) {
        void* c; const void* a; const void* w; const void* as; const void* ws;
        unsigned m, n, k;
    } args = {device[4], device[0], device[1], device[2], device[3], M, N, K};
    for (unsigned run = 0; run < 2; run++) {
        check_hsa(plow_hsa_launch(h, 0, &split_kernel, 256 * PLOW_WG_THREADS, 1, 1,
            PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
        check_hsa(plow_hsa_copy_d2h(h, 0, run ? repeat : parts, device[4], bytes * splits));
    }
    const int parts_ok = !memcmp(parts, expected, bytes * splits) && !memcmp(parts, repeat, bytes * splits);
    if (snprintf(path, sizeof(path), "%s.parts.bf16", output) >= sizeof(path)) exit(1);
    f = fopen(path, "wbx");
    if (!f || fwrite(parts, 1, bytes * splits, f) != bytes * splits || fclose(f)) exit(1);
    for (unsigned run = 0; run < 4; run++) {
        memset(repeat, 0, bytes);
        check_hsa(plow_hsa_copy_h2d(h, 0, device[4], repeat, bytes));
        check_hsa(plow_hsa_launch(h, 0, &atomic_kernel, 256 * PLOW_WG_THREADS, 1, 1,
            PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
        check_hsa(plow_hsa_copy_d2h(h, 0, repeat, device[4], bytes));
        violations += check_bf16_bounds(repeat, lower, upper, elems);
        if (snprintf(path, sizeof(path), "%s.atomic%u.bf16", output, run) >= sizeof(path)) exit(1);
        f = fopen(path, "wbx");
        if (!f || fwrite(repeat, 1, bytes, f) != bytes || fclose(f)) exit(1);
    }
    const int ok = parts_ok && !violations;
    printf("qb-split %s M=%u splits=%u %s parts-bitwise=%d reference/native-bound-violations=%zu\n",
        prefix, M, splits, ok ? "PASS" : "FAIL", parts_ok, violations);
    fails += !ok;
    for (unsigned i = 0; i < 5; i++) { plow_hsa_free(h, host[i]); plow_hsa_free(h, device[i]); }
    plow_hsa_free(h, parts); plow_hsa_free(h, repeat);
    free(expected); free(interleaved); free(lower); free(upper);
}

static void enumerate_bf16_orders(const bf16* parts, unsigned H, unsigned n, unsigned remaining,
                                  bf16 sum, float* lower, float* upper) {
    if (!remaining) {
        const float value = bf2f(sum);
        if (value < *lower) *lower = value;
        if (value > *upper) *upper = value;
        return;
    }
    for (unsigned bits = remaining; bits; bits &= bits - 1) {
        const unsigned k = __builtin_ctz(bits);
        enumerate_bf16_orders(parts, H, n, remaining ^ (1u << k),
            f2bf(bf2f(sum) + bf2f(parts[k * H + n])), lower, upper);
    }
}

static int test_bf16_bounds(void) {
    bf16 parts[8 * 32];
    float lower[32], upper[32];
    const float cancel[] = {256.0f, 1.0f, -256.0f, -1.0f, 0.00390625f, 1.0f, -1.0f, 0.0f};
    for (unsigned k = 0; k < 8; k++)
        for (unsigned n = 0; n < 32; n++)
            parts[k * 32 + n] = n == 0 ? f2bf(cancel[k]) :
                (bf16)((rand() & 0x8000) | (0x3800 + rand() % 0x1000));
    for (unsigned k = 1; k <= 8; k++) {
        bf16_sum_order_bounds(parts, 1, 32, k, lower, upper);
        for (unsigned n = 0; n < 32; n++) {
            float lo = INFINITY, hi = -INFINITY;
            enumerate_bf16_orders(parts, 32, n, (1u << k) - 1, 0, &lo, &hi);
            if (lo != lower[n] || hi != upper[n]) return 1;
        }
    }
    puts("BF16 subset bounds equal exhaustive permutations: topk1..8,32 columns PASS");
    return 0;
}

static void replay_grouped(plow_hsa* h, const char* input, const char* output, unsigned down,
                           const char* hidden_path, const char* reference_path, const char* repeat_path) {
    FILE* f = fopen(input, "rb");
    uint32_t dims[6];
    if (!f || fread(dims, sizeof(dims), 1, f) != 1) exit(1);
    const unsigned T = dims[0], I = dims[1], H = dims[2], E = dims[3], topk = dims[4], active = dims[5];
    if (!T || T > 65536 || !I || I > 32768 || !H || H > 32768 || I % 128 || H % 128
        || !E || E > 1024 || !topk || topk > 16 || !active || active > E) exit(1);
    const size_t slots = (size_t)T * topk, nw = (size_t)I * H, ns = (size_t)(I / 128) * (H / 128) * 4;
    const unsigned atomic = reference_path != NULL;
    if (atomic && (!down || !hidden_path || !repeat_path || topk > 8)) exit(1);
    const unsigned width = down ? H : I, input_width = down ? I : H;
    const size_t input_rows = down ? slots : T;
    const unsigned weight_parts = down ? 2 : 4;
    const size_t sizes[] = {input_rows * input_width, (input_width / 128) * input_rows * 4,
                            slots * 4, slots * 4, slots * width * 2};
    size_t bytes = sizeof(dims) + active * (4 + (down ? 1 : 2) * (nw + ns));
    for (unsigned i = 0; i < 5; i++) bytes += sizes[i];
    if (fseek(f, 0, SEEK_END) || ftell(f) != (long)bytes || fseek(f, sizeof(dims), SEEK_SET)) exit(1);
    void* host[5];
    for (unsigned i = 0; i < 5; i++) {
        host[i] = plow_hsa_alloc_host(h, sizes[i]);
        if (!host[i] || fread(host[i], 1, sizes[i], f) != sizes[i]) exit(1);
    }
    float *lower = NULL, *upper = NULL;
    if (atomic) {
        lower = malloc((size_t)T * H * 4); upper = malloc((size_t)T * H * 4);
        bf16* ref = malloc((size_t)T * H * 2);
        if (!lower || !upper || !ref) exit(1);
        bf16_sum_order_bounds(host[4], T, H, topk, lower, upper);
        const char* refs[] = {reference_path, repeat_path};
        for (unsigned r = 0; r < 2; r++) {
            FILE* rf = fopen(refs[r], "rb");
            if (!rf || fread(ref, 2, (size_t)T * H, rf) != (size_t)T * H || fgetc(rf) != EOF || fclose(rf)) exit(1);
            const size_t bad = check_bf16_bounds(ref, lower, upper, (size_t)T * H);
            printf("CK BF16 reduction repeat=%u %s outside-order-bounds=%zu\n", r, bad ? "FAIL" : "PASS", bad);
            fails += bad != 0;
        }
        free(ref);
    }
    bf16* hidden = NULL;
    if (hidden_path) {
        FILE* hf = fopen(hidden_path, "rb");
        hidden = malloc(slots * I * 2);
        if (!down || !hf || !hidden || fread(hidden, 2, slots * I, hf) != slots * I
            || fgetc(hf) != EOF || fclose(hf)) exit(1);
    }
    void* da = plow_hsa_alloc(h, 0, sizes[0]);
    void* das = plow_hsa_alloc(h, 0, sizes[1]);
    unsigned long long* wt = plow_hsa_alloc_host(h, E * 3u * 8);
    unsigned long long* st = plow_hsa_alloc_host(h, E * 3u * 8);
    void** weights = calloc(active * weight_parts, sizeof(void*));
    if (!da || !das || !wt || !st || !weights) exit(1);
    memset(wt, 0, E * 3u * 8); memset(st, 0, E * 3u * 8);
    check_hsa(plow_hsa_copy_h2d(h, 0, da, host[0], sizes[0]));
    check_hsa(plow_hsa_copy_h2d(h, 0, das, host[1], sizes[1]));
    for (unsigned n = 0; n < active; n++) {
        unsigned e;
        if (fread(&e, sizeof(e), 1, f) != 1 || e >= E || wt[e * 3u + (down ? 2 : 0)]) exit(1);
        for (unsigned part = 0; part < weight_parts; part++) {
            const unsigned scaled = part >= weight_parts / 2;
            const size_t size = scaled ? ns : nw;
            void* data = plow_hsa_alloc_host(h, size);
            void* device = plow_hsa_alloc(h, 0, size);
            if (!data || !device || fread(data, 1, size, f) != size) exit(1);
            check_hsa(plow_hsa_copy_h2d(h, 0, device, data, size));
            plow_hsa_free(h, data);
            weights[n * weight_parts + part] = device;
            (scaled ? st : wt)[e * 3u + (down ? 2 : part % 2)] = (uintptr_t)device;
        }
    }
    fclose(f);
    plow_hsa_kernel geometry, align, glu;
    check_hsa(plow_hsa_get_kernel(h, 0, "moe_group_fp8_geometry", &geometry));
    check_hsa(plow_hsa_get_kernel(h, 0, "moe_align_gemma_pf_k", &align));
    check_hsa(plow_hsa_get_kernel(h, 0, atomic ? "moe_group_down_atomic_a8w8_block128_m16" :
        down ? "moe_group_down_a8w8_block128_m16" :
        "moe_group_glu_a8w8_block128_m16", &glu));
    unsigned* geom = plow_hsa_alloc_host(h, 8);
    void* dg = plow_hsa_alloc(h, 0, 8);
    if (!geom || !dg) exit(1);
    check_hsa(plow_hsa_launch(h, 0, &geometry, 1, 1, 1, 1, 1, 1, 0, &dg, sizeof(dg)));
    check_hsa(plow_hsa_wait(h, 0));
    check_hsa(plow_hsa_copy_d2h(h, 0, geom, dg, 8));
    const unsigned tile_rows = geom[0];
    if (!tile_rows || tile_rows > 1024 || geom[1] != PLOW_WG_THREADS) exit(1);
    const size_t padded = slots + (size_t)E * (tile_rows - 1);
    const size_t outcount = atomic ? (size_t)T * H : slots * width;
    const size_t outbytes = ((atomic ? T : down ? slots : padded) * width + 16) * 2;
    bf16* sorted_hidden = hidden ? plow_hsa_alloc_host(h, padded * I * 2) : NULL;
    void* dhidden = hidden ? plow_hsa_alloc(h, 0, padded * I * 2) : NULL;
    void* actual_q = hidden ? plow_hsa_alloc_host(h, sizes[0]) : NULL;
    void* actual_scale = hidden ? plow_hsa_alloc_host(h, sizes[1]) : NULL;
    plow_hsa_kernel quant;
    if (hidden) {
        if (!sorted_hidden || !dhidden || !actual_q || !actual_scale) exit(1);
        check_hsa(plow_hsa_get_kernel(h, 0, "moe_quant_fp8_block128", &quant));
    }
    unsigned* table = plow_hsa_alloc_host(h, slots * 8);
    bf16* out = plow_hsa_alloc_host(h, outbytes);
    int* meta = plow_hsa_alloc_host(h, (3u * E + 1) * 4);
    unsigned* rt = plow_hsa_alloc_host(h, padded * 4);
    unsigned* rp = plow_hsa_alloc_host(h, padded * 4);
    float* rg = plow_hsa_alloc_host(h, padded * 4);
    bf16* scattered = malloc(sizes[4]);
    bf16* first = malloc(sizes[4]);
    unsigned char* seen = malloc(slots);
    if (!table || !out || !meta || !rt || !rp || !rg || !scattered || !first || !seen) exit(1);
    for (size_t s = 0; s < slots; s++) {
        unsigned e = ((unsigned*)host[2])[s];
        if (e >= E || !wt[e * 3u + (down ? 2 : 0)] || !isfinite(((float*)host[3])[s])) exit(1);
        table[s * 2] = e;
        memcpy(table + s * 2 + 1, (float*)host[3] + s, 4);
    }
    const size_t dsizes[] = {slots * 8, outbytes, (3u * E + 1) * 4, padded * 4,
                             padded * 4, padded * 4, E * 3u * 8, E * 3u * 8};
    void* dev[8];
    for (unsigned i = 0; i < 8; i++) { dev[i] = plow_hsa_alloc(h, 0, dsizes[i]); if (!dev[i]) exit(1); }
    check_hsa(plow_hsa_copy_h2d(h, 0, dev[0], table, dsizes[0]));
    check_hsa(plow_hsa_copy_h2d(h, 0, dev[6], wt, dsizes[6]));
    check_hsa(plow_hsa_copy_h2d(h, 0, dev[7], st, dsizes[7]));
    struct __attribute__((packed)) {
        void *meta, *table, *rt, *rp, *rg; unsigned t, e, k;
    } align_args = {dev[2], dev[0], dev[3], dev[4], dev[5], T, E, topk};
    struct __attribute__((packed)) {
        void *fu, *a, *asc, *wt, *st, *meta, *rt; unsigned i, h, e, t;
    } glu_args = {dev[1], da, das, dev[6], dev[7], dev[2], dev[3], I, H, E, T};
    struct __attribute__((packed)) {
        void *out, *a, *asc, *wt, *st, *meta, *rp, *rg; unsigned i, h, e, slots;
    } down_args = {dev[1], da, das, dev[6], dev[7], dev[2], dev[4], dev[5], I, H, E, (unsigned)slots};
    struct __attribute__((packed)) {
        void *out, *a, *asc, *wt, *st, *meta, *rp, *rg; unsigned i, h, e, slots, topk;
    } atomic_args = {dev[1], da, das, dev[6], dev[7], dev[2], dev[4], dev[5], I, H, E, (unsigned)slots, topk};
    struct __attribute__((packed)) {
        void *a, *hidden, *asc, *meta, *rp; unsigned i, e, slots;
    } quant_args = {da, dhidden, das, dev[2], dev[4], I, E, (unsigned)slots};
    if (atomic) {
        plow_hsa_kernel ordered;
        check_hsa(plow_hsa_get_kernel(h, 0, "bf16_atomic_ordered_add", &ordered));
        void* dp = plow_hsa_alloc(h, 0, sizes[4]);
        if (!dp) exit(1);
        check_hsa(plow_hsa_copy_h2d(h, 0, dp, host[4], sizes[4]));
        memset(out, 0xa5, outbytes); memset(out, 0, outcount * 2);
        memset(first, 0, outcount * 2);
        check_hsa(plow_hsa_copy_h2d(h, 0, dev[1], out, outbytes));
        for (unsigned slot = 0; slot < topk; slot++) {
            struct __attribute__((packed)) {
                void *out, *parts; unsigned t, h, k, slot;
            } args = {dev[1], dp, T, H, topk, slot};
            check_hsa(plow_hsa_launch(h, 0, &ordered, 7 * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                     0, &args, sizeof(args)));
            check_hsa(plow_hsa_wait(h, 0));
            for (size_t j = 0; j < outcount; j++)
                first[j] = f2bf(bf2f(first[j]) + bf2f(((bf16*)host[4])[(j / H * topk + slot) * H + j % H]));
        }
        check_hsa(plow_hsa_copy_d2h(h, 0, out, dev[1], outbytes));
        size_t bad = 0;
        for (size_t j = 0; j < outcount; j++) bad += first[j] != out[j];
        for (size_t j = outcount; j < outbytes / 2; j++) bad += out[j] != 0xa5a5;
        printf("ordered BF16 atomic %s changed=%zu\n", bad ? "FAIL" : "PASS", bad);
        fails += bad != 0;
        plow_hsa_free(h, dp);
    }
    const unsigned grids[] = {1, 7, 256};
    for (unsigned run = 0; run < 3; run++) {
        memset(out, 0xa5, outbytes); memset(seen, 0, slots); memset(scattered, 0, sizes[4]);
        if (atomic) memset(out, 0, outcount * 2);
        check_hsa(plow_hsa_copy_h2d(h, 0, dev[1], out, outbytes));
        check_hsa(plow_hsa_launch(h, 0, &align, PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                 0, &align_args, sizeof(align_args)));
        check_hsa(plow_hsa_wait(h, 0));
        if (hidden) {
            check_hsa(plow_hsa_copy_d2h(h, 0, meta, dev[2], dsizes[2]));
            check_hsa(plow_hsa_copy_d2h(h, 0, rp, dev[4], dsizes[4]));
            const size_t rows = (size_t)meta[3u * E] * tile_rows;
            if (rows > padded) exit(1);
            memset(sorted_hidden, 0xff, padded * I * 2);
            for (size_t r = 0; r < rows; r++) {
                if (rp[r] == UINT32_MAX) continue;
                if (rp[r] >= slots) exit(1);
                memcpy(sorted_hidden + r * I, hidden + (size_t)rp[r] * I, I * 2);
            }
            check_hsa(plow_hsa_copy_h2d(h, 0, dhidden, sorted_hidden, padded * I * 2));
            memset(actual_q, 0xa5, sizes[0]); memset(actual_scale, 0xff, sizes[1]);
            check_hsa(plow_hsa_copy_h2d(h, 0, da, actual_q, sizes[0]));
            check_hsa(plow_hsa_copy_h2d(h, 0, das, actual_scale, sizes[1]));
            check_hsa(plow_hsa_launch(h, 0, &quant, grids[run] * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                     0, &quant_args, sizeof(quant_args)));
            check_hsa(plow_hsa_wait(h, 0));
            check_hsa(plow_hsa_copy_d2h(h, 0, actual_q, da, sizes[0]));
            check_hsa(plow_hsa_copy_d2h(h, 0, actual_scale, das, sizes[1]));
            const int exact = !memcmp(actual_q, host[0], sizes[0]) && !memcmp(actual_scale, host[1], sizes[1]);
            printf("grouped-quant T=%u I=%u grid=%u %s bitwise=%d\n", T, I, grids[run], exact ? "PASS" : "FAIL", exact);
            fails += !exact;
        }
        check_hsa(plow_hsa_launch(h, 0, &glu, grids[run] * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                 0, atomic ? (void*)&atomic_args : down ? (void*)&down_args : (void*)&glu_args,
                                 atomic ? sizeof(atomic_args) : down ? sizeof(down_args) : sizeof(glu_args)));
        check_hsa(plow_hsa_wait(h, 0));
        check_hsa(plow_hsa_copy_d2h(h, 0, out, dev[1], outbytes));
        check_hsa(plow_hsa_copy_d2h(h, 0, meta, dev[2], dsizes[2]));
        check_hsa(plow_hsa_copy_d2h(h, 0, rt, dev[3], dsizes[3]));
        check_hsa(plow_hsa_copy_d2h(h, 0, rp, dev[4], dsizes[4]));
        check_hsa(plow_hsa_copy_d2h(h, 0, rg, dev[5], dsizes[5]));
        const size_t rows = (size_t)meta[3u * E] * tile_rows;
        if (rows > padded) exit(1);
        size_t changed = 0, pad_errors = 0;
        unsigned cursor = 0;
        int finite = 1;
        for (unsigned e = 0; e < E; e++) {
            const unsigned count = (unsigned)meta[E + e];
            const unsigned extent = (count + tile_rows - 1) / tile_rows * tile_rows;
            if (meta[e] != (int)cursor || cursor + extent > rows || meta[2u * E + e] != (int)(cursor / tile_rows)) exit(1);
            for (unsigned r = 0; r < extent; r++) {
                const size_t row = cursor + r;
                if (r >= count) {
                    if (rp[row] != UINT32_MAX || rt[row] != UINT32_MAX) exit(1);
                    if (!down) for (unsigned n = 0; n < width; n++) pad_errors += out[row * width + n] != 0;
                    continue;
                }
                const unsigned s = rp[row];
                if (s >= slots || seen[s] || rt[row] != s / topk || table[s * 2] != e
                    || memcmp(rg + row, (float*)host[3] + s, 4)) exit(1);
                seen[s] = 1;
                if (atomic) continue;
                for (unsigned n = 0; n < width; n++) {
                    const bf16 value = out[(down ? s : row) * width + n];
                    scattered[(size_t)s * width + n] = value;
                    finite &= isfinite(bf2f(value));
                    changed += value != ((bf16*)host[4])[(size_t)s * width + n];
                }
            }
            cursor += extent;
        }
        for (size_t s = 0; s < slots; s++) if (!seen[s]) exit(1);
        for (size_t j = (atomic ? T : down ? slots : rows) * width; j < outbytes / 2; j++) pad_errors += out[j] != 0xa5a5;
        if (atomic) {
            changed = check_bf16_bounds(out, lower, upper, outcount);
            memcpy(scattered, out, outcount * 2);
            char path[4096];
            if (snprintf(path, sizeof(path), "%s.grid%u", output, grids[run]) >= (int)sizeof(path)) exit(1);
            FILE* af = fopen(path, "wbx");
            if (!af || fwrite(out, 2, outcount, af) != outcount || fclose(af)) exit(1);
        }
        const int stable = !run || !memcmp(first, scattered, outcount * 2);
        if (!run) memcpy(first, scattered, outcount * 2);
        const int ok = finite && !changed && !pad_errors && (atomic || stable);
        printf("grouped-%s T=%u I=%u H=%u E=%u grid=%u %s changed=%zu pad-errors=%zu repeat-bitwise=%d\n",
               atomic ? "down-atomic-bounds" : down ? "down" : "glu", T, I, H, E, grids[run], ok ? "PASS" : "FAIL", changed, pad_errors, stable);
        fails += !ok;
    }
    f = fopen(output, "wbx");
    if (!f || fwrite(first, 2, outcount, f) != outcount || fclose(f)) exit(1);
    for (unsigned i = 0; i < 5; i++) plow_hsa_free(h, host[i]);
    for (unsigned i = 0; i < active * weight_parts; i++) plow_hsa_free(h, weights[i]);
    for (unsigned i = 0; i < 8; i++) plow_hsa_free(h, dev[i]);
    void* allocations[] = {da, das, wt, st, geom, dg, table, out, meta, rt, rp, rg};
    for (unsigned i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++) plow_hsa_free(h, allocations[i]);
    free(weights); free(scattered); free(first); free(seen);
    free(lower); free(upper);
    if (hidden) {
        free(hidden);
        plow_hsa_free(h, sorted_hidden); plow_hsa_free(h, dhidden);
        plow_hsa_free(h, actual_q); plow_hsa_free(h, actual_scale);
    }
}

static void run_sum4(plow_hsa* h, plow_hsa_kernel* fused, plow_hsa_kernel* add,
                     unsigned n, unsigned blocks) {
    const size_t bytes = ((size_t)n + 8) * 2;
    bf16* input[4];
    void* device[4];
    bf16* actual = plow_hsa_alloc_host(h, bytes);
    bf16* expected = plow_hsa_alloc_host(h, bytes);
    for (unsigned part = 0; part < 4; part++) {
        input[part] = plow_hsa_alloc_host(h, bytes);
        device[part] = plow_hsa_alloc(h, 0, bytes);
        if (!input[part] || !device[part] || !actual || !expected) exit(1);
        for (unsigned i = 0; i < n + 8; i++) {
            input[part][i] = (bf16)((rand() & 0x8000) | (rand() % 0x7800));
            if (i % 8 == 0) {
                const float rounds[] = {256.0f, 1.0f, -256.0f, 0.0f};
                input[part][i] = f2bf(rounds[part]);
            }
            if (i % 8 == 1) input[part][i] = (bf16)((i & 1) << 15);
        }
        check_hsa(plow_hsa_copy_h2d(h, 0, device[part], input[part], bytes));
    }
    for (unsigned part = 1; part < 4; part++) {
        struct __attribute__((packed)) {
            void* out; const void* a; const void* b; unsigned n; float scale;
        } args = {device[0], device[0], device[part], n, 1.0f};
        check_hsa(plow_hsa_launch(h, 0, add, blocks * PLOW_WG_THREADS, 1, 1,
                                 PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)));
        check_hsa(plow_hsa_wait(h, 0));
    }
    check_hsa(plow_hsa_copy_d2h(h, 0, expected, device[0], bytes));
    check_hsa(plow_hsa_copy_h2d(h, 0, device[0], input[0], bytes));
    struct __attribute__((packed)) {
        void* out; const void* b; const void* c; const void* d; unsigned n;
    } args = {device[0], device[1], device[2], device[3], n};
    check_hsa(plow_hsa_launch(h, 0, fused, blocks * PLOW_WG_THREADS, 1, 1,
                             PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args)));
    check_hsa(plow_hsa_wait(h, 0));
    check_hsa(plow_hsa_copy_d2h(h, 0, actual, device[0], bytes));
    unsigned changed = 0;
    for (unsigned i = 0; i < n; i++) changed += actual[i] != expected[i];
    const int guard = memcmp(actual + n, input[0] + n, 16) == 0;
    const int ok = changed == 0 && guard;
    printf("sum4 BF16 n=%u blocks=%u %s changed=%u guard=%d\n", n, blocks,
           ok ? "PASS" : "FAIL", changed, guard);
    fails += !ok;
    for (unsigned part = 0; part < 4; part++) {
        plow_hsa_free(h, input[part]);
        plow_hsa_free(h, device[part]);
    }
    plow_hsa_free(h, actual);
    plow_hsa_free(h, expected);
}

static void run_rows(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, const char* label, unsigned N,
                     unsigned K, unsigned M) {
    const unsigned NB = (N + 127u) / 128u, KB = (K + 127u) / 128u;
    const size_t nW = (size_t)N * K, nS = (size_t)NB * KB, nC = (size_t)M * N;

    unsigned char* hW = plow_hsa_alloc_host(h, nW);
    bf16* hx = plow_hsa_alloc_host(h, (size_t)M * K * 2);
    float* hS = plow_hsa_alloc_host(h, nS * 4);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);

    /* SMALL e4m3 magnitudes (exp field <= 7 -> |v| < 2) so the K-long dot stays well-conditioned;
     * the per-block scales carry the dynamic range, exactly as the per-channel fp8 GEMV test does. */
    for (size_t i = 0; i < nW; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2;
        hW[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (unsigned i = 0; i < M * K; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nS; i++) hS[i] = 0.005f + 0.02f * (rand() % 8) / 8.0f;

    void* dW = plow_hsa_alloc(h, 0, nW);
    void* dx = plow_hsa_alloc(h, 0, (size_t)M * K * 2);
    void* dS = plow_hsa_alloc(h, 0, nS * 4);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    plow_hsa_copy_h2d(h, 0, dW, hW, nW);
    plow_hsa_copy_h2d(h, 0, dx, hx, (size_t)M * K * 2);
    plow_hsa_copy_h2d(h, 0, dS, hS, nS * 4);

    struct __attribute__((packed)) {
        void* c; const void* x; const void* w; const void* ws; unsigned m, n, kk;
    } args = {dC, dx, dW, dS, M, N, K};

    plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);

    /* Perf: launch-overhead-dominated standalone (~30us floor per campaign notes), so this is a
     * per-shape PROFILE not a full-model per-op proxy. TB/s over the fp8 weight bytes streamed (N*K). */
    const int ITERS = 300;
    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double ms = (now() - t0) * 1e3 / ITERS;
    const double tbs = (double)nW / (ms * 1e-3) / 1e12;

    double worst = 0.0;
    for (unsigned m = 0; m < M; m++) for (unsigned n = 0; n < N; n++) {
        double want = 0.0;
        for (unsigned kk = 0; kk < K; kk++)
            want += (double)bf2f(hx[(size_t)m * K + kk]) * e4m3_decode(hW[(size_t)n * K + kk]) *
                    (double)hS[(size_t)(n >> 7) * KB + (kk >> 7)];
        const double got = bf2f(hC[(size_t)m * N + n]);
        if (!isfinite(got)) { worst = INFINITY; continue; }
        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    const int ok = worst < 3e-2;
    printf("  %-20s M=%u N=%5u K=%5u  %s rel %.4f | %.3f ms  %.2f TB/s\n", label, M, N, K,
           ok ? "PASS" : "FAIL", worst, ms, tbs);
    if (!ok) fails++;

    plow_hsa_free(h, dW); plow_hsa_free(h, dx); plow_hsa_free(h, dS); plow_hsa_free(h, dC);
}

static void run(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, const char* label, unsigned N,
                unsigned K) {
    run_rows(h, k, NCU, label, N, K, 1);
}

/* Block-fp8 DENSE PREFILL GEMM (op 107, d_gemm_fp8_blk) vs the SAME f64 reference the decode GEMV
 * above uses, plus a direct cross-kernel check against `gemv_fp8_blk` on identical weights.
 *
 * THE CROSS-KERNEL CHECK IS THE POINT, not a bonus. A block-fp8 GEMM with a wrong scale block reads
 * as plausible-but-wrong output and never crashes, and the two kernels index the grid completely
 * differently — the GEMV folds `wscale[(n>>7)*KB + (k>>7)]` into a per-lane chunk partial, the GEMM
 * promotes a whole MFMA accumulator by `wscale[(n0>>7)*KB + (kt>>1)]` at a k-tile boundary. An f64
 * reference proves each is right; agreeing with each other proves they read ONE convention, which
 * is what the emitter relies on when it hands both phases the same `.weight_scale_inv` handle.
 *
 * The reference SAMPLES (m,n) pairs rather than computing all M*N: at M=512, N=6144, K=4096 the full
 * product is 12.9 G f64 MACs on one core. Sampling is not a weaker test here — a tiling, swizzle or
 * scale-index bug is systematic across the output, not concentrated in a few elements. */
static void run_gemm_blk(plow_hsa* h, plow_hsa_kernel* kg, plow_hsa_kernel* kv, unsigned NCU,
                         const char* label, unsigned M, unsigned N, unsigned K) {
    const unsigned NB = (N + 127u) / 128u, KB = (K + 127u) / 128u;
    const size_t nW = (size_t)N * K, nS = (size_t)NB * KB, nA = (size_t)M * K, nC = (size_t)M * N;

    unsigned char* hW = plow_hsa_alloc_host(h, nW);
    bf16* hA = plow_hsa_alloc_host(h, nA * 2);
    float* hS = plow_hsa_alloc_host(h, nS * 4);
    bf16* hC = plow_hsa_alloc_host(h, nC * 2);
    bf16* hCv = plow_hsa_alloc_host(h, (size_t)N * 2);

    /* Same conditioning as the GEMV above: exp field <= 7 (|v| < 2) so a K-long dot stays
     * well-conditioned and a real layout bug shows as ~100% error rather than a few percent of
     * legitimate f32-vs-f64 cancellation. The per-block scales carry the dynamic range. */
    for (size_t i = 0; i < nW; i++) {
        const unsigned e = rand() % 8, m = rand() % 8, s = rand() % 2;
        hW[i] = (unsigned char)((s << 7) | (e << 3) | m);
    }
    for (size_t i = 0; i < nA; i++) hA[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);
    for (size_t i = 0; i < nS; i++) hS[i] = 0.005f + 0.02f * (rand() % 8) / 8.0f;

    void* dW = plow_hsa_alloc(h, 0, nW);
    void* dA = plow_hsa_alloc(h, 0, nA * 2);
    void* dS = plow_hsa_alloc(h, 0, nS * 4);
    void* dC = plow_hsa_alloc(h, 0, nC * 2);
    void* dCv = plow_hsa_alloc(h, 0, (size_t)N * 2);
    plow_hsa_copy_h2d(h, 0, dW, hW, nW);
    plow_hsa_copy_h2d(h, 0, dA, hA, nA * 2);
    plow_hsa_copy_h2d(h, 0, dS, hS, nS * 4);

    struct __attribute__((packed)) {
        void* c; const void* a; const void* w; const void* ws; unsigned m, n, kk;
    } args = {dC, dA, dW, dS, M, N, K};
    plow_hsa_launch(h, 0, kg, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args,
                    sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hC, dC, nC * 2);

    /* The decode GEMV on row 0 of the SAME A, W and scale grid. */
    struct __attribute__((packed)) {
        void* c; const void* x; const void* w; const void* ws; unsigned m, n, kk;
    } vargs = {dCv, dA, dW, dS, 1u, N, K};
    plow_hsa_launch(h, 0, kv, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &vargs,
                    sizeof(vargs));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hCv, dCv, (size_t)N * 2);

    double worst = 0.0;
    const int PROBES = 512;
    for (int p = 0; p < PROBES; p++) {
        const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
        double want = 0.0;
        for (unsigned kk = 0; kk < K; kk++)
            want += (double)bf2f(hA[(size_t)m * K + kk]) * e4m3_decode(hW[(size_t)n * K + kk]) *
                    (double)hS[(size_t)(n >> 7) * KB + (kk >> 7)];
        const double got = bf2f(hC[(size_t)m * N + n]);
        const double rel = fabs(got - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    /* Row 0 against the decode kernel, every column. Two f32 reduction orders, so this is a
     * numeric-agreement bound, not bit-identity. */
    double xworst = 0.0;
    for (unsigned n = 0; n < N; n++) {
        const double g = bf2f(hC[n]), v = bf2f(hCv[n]);
        const double rel = fabs(g - v) / (fabs(v) + 1e-2);
        if (rel > xworst) xworst = rel;
    }
    const int ok = worst < 3e-2 && xworst < 3e-2;
    printf("  %-24s M=%5u N=%5u K=%5u  %s  ref %.4f  vs-gemv %.4f\n", label, M, N, K,
           ok ? "PASS" : "FAIL", worst, xworst);
    if (!ok) fails++;

    plow_hsa_free(h, dW); plow_hsa_free(h, dA); plow_hsa_free(h, dS);
    plow_hsa_free(h, dC); plow_hsa_free(h, dCv);
}

/* Block-fp8 MoE expert gate/up + down for ONE expert (slot 0), through the real table indirection.
 * fu = act(gate·x)*(up·x) ; part = gate_weight · (down·fu). Validates the fp8 expert decode path. */
static void run_expert(plow_hsa* h, plow_hsa_kernel* kglu, plow_hsa_kernel* kdown, unsigned NCU,
                       unsigned I_moe, unsigned H) {
    const unsigned act = 1u;              /* silu (SwiGLU — GLM) */
    const float gate_w = 0.7f;            /* the router gate weight for this slot */
    const unsigned IB = (I_moe + 127u) / 128u, HB = (H + 127u) / 128u;

    /* Wg,Wu : [I_moe][H] fp8 + [IB][HB] scale.  Wd : [H][I_moe] fp8 + [HB][IB] scale. */
    unsigned char* hWg = plow_hsa_alloc_host(h, (size_t)I_moe * H);
    unsigned char* hWu = plow_hsa_alloc_host(h, (size_t)I_moe * H);
    unsigned char* hWd = plow_hsa_alloc_host(h, (size_t)H * I_moe);
    float* hSg = plow_hsa_alloc_host(h, (size_t)IB * HB * 4);
    float* hSu = plow_hsa_alloc_host(h, (size_t)IB * HB * 4);
    float* hSd = plow_hsa_alloc_host(h, (size_t)HB * IB * 4);
    bf16* hx = plow_hsa_alloc_host(h, (size_t)H * 2);
    bf16* hfu = plow_hsa_alloc_host(h, (size_t)I_moe * 2);
    float* hpart = plow_hsa_alloc_host(h, (size_t)H * 4);
    for (size_t i = 0; i < (size_t)I_moe * H; i++) {
        hWg[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
        hWu[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
    }
    for (size_t i = 0; i < (size_t)H * I_moe; i++)
        hWd[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
    for (size_t i = 0; i < (size_t)IB * HB; i++) { hSg[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f; hSu[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f; }
    for (size_t i = 0; i < (size_t)HB * IB; i++) hSd[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f;
    for (unsigned i = 0; i < H; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);

    void* dWg = plow_hsa_alloc(h, 0, (size_t)I_moe * H); plow_hsa_copy_h2d(h, 0, dWg, hWg, (size_t)I_moe * H);
    void* dWu = plow_hsa_alloc(h, 0, (size_t)I_moe * H); plow_hsa_copy_h2d(h, 0, dWu, hWu, (size_t)I_moe * H);
    void* dWd = plow_hsa_alloc(h, 0, (size_t)H * I_moe); plow_hsa_copy_h2d(h, 0, dWd, hWd, (size_t)H * I_moe);
    void* dSg = plow_hsa_alloc(h, 0, (size_t)IB * HB * 4); plow_hsa_copy_h2d(h, 0, dSg, hSg, (size_t)IB * HB * 4);
    void* dSu = plow_hsa_alloc(h, 0, (size_t)IB * HB * 4); plow_hsa_copy_h2d(h, 0, dSu, hSu, (size_t)IB * HB * 4);
    void* dSd = plow_hsa_alloc(h, 0, (size_t)HB * IB * 4); plow_hsa_copy_h2d(h, 0, dSd, hSd, (size_t)HB * IB * 4);
    void* dx = plow_hsa_alloc(h, 0, (size_t)H * 2); plow_hsa_copy_h2d(h, 0, dx, hx, (size_t)H * 2);
    void* dfu = plow_hsa_alloc(h, 0, (size_t)I_moe * 2);
    void* dpart = plow_hsa_alloc(h, 0, (size_t)H * 4);

    /* routing table slot 0 = {expert_id 0, gate gate_w} */
    /* Small pointer tables: use plow_hsa_upload (stages arbitrary host memory); copy_h2d requires
     * alloc_host-pinned memory and would SDMA-fault on these stack arrays. */
    unsigned char htab[8]; *(unsigned*)htab = 0u; *(float*)(htab + 4) = gate_w;
    void* dtab = plow_hsa_alloc(h, 0, 8); plow_hsa_upload(h, 0, dtab, htab, 8);
    uint64_t wtab[3] = {(uint64_t)(uintptr_t)dWg, (uint64_t)(uintptr_t)dWu, (uint64_t)(uintptr_t)dWd};
    uint64_t stab[3] = {(uint64_t)(uintptr_t)dSg, (uint64_t)(uintptr_t)dSu, (uint64_t)(uintptr_t)dSd};
    void* dwtab = plow_hsa_alloc(h, 0, sizeof(wtab)); plow_hsa_upload(h, 0, dwtab, wtab, sizeof(wtab));
    void* dstab = plow_hsa_alloc(h, 0, sizeof(stab)); plow_hsa_upload(h, 0, dstab, stab, sizeof(stab));

    struct __attribute__((packed)) {
        void* fu; const void* x; const void* table; const void* wtab; const void* stab;
        unsigned i_moe, h, n_exp, act;
    } aglu = {dfu, dx, dtab, dwtab, dstab, I_moe, H, 1u, act};
    plow_hsa_launch(h, 0, kglu, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aglu, sizeof(aglu));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hfu, dfu, (size_t)I_moe * 2);

    /* Perf profile: gate+up stream 2*I_moe*H fp8 bytes; down streams H*I_moe. */
    const int ITERS = 300;
    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, kglu, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aglu, sizeof(aglu));
    plow_hsa_wait(h, 0);
    double t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, kglu, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &aglu, sizeof(aglu));
    plow_hsa_wait(h, 0);
    const double glu_ms = (now() - t0) * 1e3 / ITERS;
    const double glu_tbs = (double)2 * I_moe * H / (glu_ms * 1e-3) / 1e12;

    struct __attribute__((packed)) {
        void* part; const void* fu; const void* table; const void* wtab; const void* stab;
        unsigned h, i_moe, n_exp;
    } adown = {dpart, dfu, dtab, dwtab, dstab, H, I_moe, 1u};
    plow_hsa_launch(h, 0, kdown, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &adown, sizeof(adown));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hpart, dpart, (size_t)H * 4);

    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, kdown, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &adown, sizeof(adown));
    plow_hsa_wait(h, 0);
    t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, kdown, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &adown, sizeof(adown));
    plow_hsa_wait(h, 0);
    const double down_ms = (now() - t0) * 1e3 / ITERS;
    const double down_tbs = (double)H * I_moe / (down_ms * 1e-3) / 1e12;

    /* Reference: fu[n]=silu(Σ_k x·decode(Wg)·Sg[n/128][k/128]) * (Σ_k x·decode(Wu)·Su);
     *            part[h]=gate_w·Σ_i fu·decode(Wd)·Sd[h/128][i/128]. */
    double worst_fu = 0.0, worst_p = 0.0;
    for (unsigned n = 0; n < I_moe; n++) {
        double g = 0.0, u = 0.0;
        for (unsigned kk = 0; kk < H; kk++) {
            const double xv = bf2f(hx[kk]);
            g += xv * e4m3_decode(hWg[(size_t)n * H + kk]) * (double)hSg[(size_t)(n >> 7) * HB + (kk >> 7)];
            u += xv * e4m3_decode(hWu[(size_t)n * H + kk]) * (double)hSu[(size_t)(n >> 7) * HB + (kk >> 7)];
        }
        const double silu = g / (1.0 + exp(-g));
        const double want = silu * u;
        const double rel = fabs(bf2f(hfu[n]) - want) / (fabs(want) + 1e-2);
        if (rel > worst_fu) worst_fu = rel;
    }
    for (unsigned hh = 0; hh < H; hh++) {
        double y = 0.0;
        for (unsigned ii = 0; ii < I_moe; ii++)
            y += (double)bf2f(hfu[ii]) * e4m3_decode(hWd[(size_t)hh * I_moe + ii]) *
                 (double)hSd[(size_t)(hh >> 7) * IB + (ii >> 7)];
        const double want = (double)gate_w * y;
        const double rel = fabs(hpart[hh] - want) / (fabs(want) + 1e-2);
        if (rel > worst_p) worst_p = rel;
    }
    const int ok = worst_fu < 3e-2 && worst_p < 3e-2;
    printf("  moe expert I_moe=%u H=%u  %s (glu rel %.4f, down rel %.4f)\n", I_moe, H,
           ok ? "PASS" : "FAIL", worst_fu, worst_p);
    printf("    gate+up (fused)  %.3f ms  %.2f TB/s   |   down  %.3f ms  %.2f TB/s\n",
           glu_ms, glu_tbs, down_ms, down_tbs);
    if (!ok) fails++;

    plow_hsa_free(h, dWg); plow_hsa_free(h, dWu); plow_hsa_free(h, dWd);
    plow_hsa_free(h, dSg); plow_hsa_free(h, dSu); plow_hsa_free(h, dSd);
    plow_hsa_free(h, dx); plow_hsa_free(h, dfu); plow_hsa_free(h, dpart);
    plow_hsa_free(h, dtab); plow_hsa_free(h, dwtab); plow_hsa_free(h, dstab);
}

/* Block-fp8 DENSE MLP gate/up (fused SwiGLU) on NAMED weights (no expert table) — GLM dense
 * layers 0-2. fu[n] = silu(gate_n·x) * (up_n·x). Validates the dense-GLU decode path (op 47). */
static void run_dense_glu(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, unsigned N, unsigned K) {
    const unsigned act = 1u; /* silu (SwiGLU) */
    const unsigned NB = (N + 127u) / 128u, KB = (K + 127u) / 128u;
    unsigned char* hWg = plow_hsa_alloc_host(h, (size_t)N * K);
    unsigned char* hWu = plow_hsa_alloc_host(h, (size_t)N * K);
    float* hSg = plow_hsa_alloc_host(h, (size_t)NB * KB * 4);
    float* hSu = plow_hsa_alloc_host(h, (size_t)NB * KB * 4);
    bf16* hx = plow_hsa_alloc_host(h, (size_t)K * 2);
    bf16* hfu = plow_hsa_alloc_host(h, (size_t)N * 2);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        hWg[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
        hWu[i] = (unsigned char)(((rand() % 2) << 7) | ((rand() % 8) << 3) | (rand() % 8));
    }
    for (size_t i = 0; i < (size_t)NB * KB; i++) {
        hSg[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f;
        hSu[i] = 0.01f + 0.02f * (rand() % 8) / 8.0f;
    }
    for (unsigned i = 0; i < K; i++) hx[i] = f2bf(((float)(rand() % 17) - 8.0f) / 16.0f);

    void* dWg = plow_hsa_alloc(h, 0, (size_t)N * K); plow_hsa_copy_h2d(h, 0, dWg, hWg, (size_t)N * K);
    void* dWu = plow_hsa_alloc(h, 0, (size_t)N * K); plow_hsa_copy_h2d(h, 0, dWu, hWu, (size_t)N * K);
    void* dSg = plow_hsa_alloc(h, 0, (size_t)NB * KB * 4); plow_hsa_copy_h2d(h, 0, dSg, hSg, (size_t)NB * KB * 4);
    void* dSu = plow_hsa_alloc(h, 0, (size_t)NB * KB * 4); plow_hsa_copy_h2d(h, 0, dSu, hSu, (size_t)NB * KB * 4);
    void* dx = plow_hsa_alloc(h, 0, (size_t)K * 2); plow_hsa_copy_h2d(h, 0, dx, hx, (size_t)K * 2);
    void* dfu = plow_hsa_alloc(h, 0, (size_t)N * 2);

    struct __attribute__((packed)) {
        void* fu; const void* x; const void* wg; const void* wu; const void* sg; const void* su;
        unsigned n, kk, act;
    } args = {dfu, dx, dWg, dWu, dSg, dSu, N, K, act};
    plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hfu, dfu, (size_t)N * 2);

    const int ITERS = 300;
    for (int w = 0; w < 20; w++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double t0 = now();
    for (int it = 0; it < ITERS; it++)
        plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    const double ms = (now() - t0) * 1e3 / ITERS;
    const double tbs = (double)2 * N * K / (ms * 1e-3) / 1e12;

    double worst = 0.0;
    for (unsigned n = 0; n < N; n++) {
        double g = 0.0, u = 0.0;
        for (unsigned kk = 0; kk < K; kk++) {
            const double xv = bf2f(hx[kk]);
            g += xv * e4m3_decode(hWg[(size_t)n * K + kk]) * (double)hSg[(size_t)(n >> 7) * KB + (kk >> 7)];
            u += xv * e4m3_decode(hWu[(size_t)n * K + kk]) * (double)hSu[(size_t)(n >> 7) * KB + (kk >> 7)];
        }
        const double want = (g / (1.0 + exp(-g))) * u;
        const double rel = fabs(bf2f(hfu[n]) - want) / (fabs(want) + 1e-2);
        if (rel > worst) worst = rel;
    }
    const int ok = worst < 3e-2;
    printf("  dense g/u %u->%u  %s rel %.4f | %.3f ms  %.2f TB/s\n", K, N,
           ok ? "PASS" : "FAIL", worst, ms, tbs);
    if (!ok) fails++;

    plow_hsa_free(h, dWg); plow_hsa_free(h, dWu); plow_hsa_free(h, dSg);
    plow_hsa_free(h, dSu); plow_hsa_free(h, dx); plow_hsa_free(h, dfu);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "test_kernels.elf";
    setbuf(stdout, NULL);
    srand(1234);
    if (argc == 2 && strcmp(argv[1], "bf16-bounds-selftest") == 0) return test_bf16_bounds();
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    FILE* f = fopen(elf, "rb");
    if (!f) { perror(elf); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    const int attention_pipeline = argc > 2 && strcmp(argv[2], "attention-pipeline-replay") == 0;
    if (argc > 2 && (strcmp(argv[2], "attention-ps-replay") == 0 || attention_pipeline)) {
        // The hash-pinned vendor ELF's descriptor omits the 384-byte size in its metadata.
        const uint32_t descriptor[16] = {0x28000, 0, 0, 0, 0x1100, 0, 0, 0,
            0, 0, 0, 0x3f, 0xc033f, 0x384, 8, 0};
        if (n != 68768 || memcmp((char*)co + 0x1700, descriptor, sizeof(descriptor))) {
            fprintf(stderr, "persistent attention descriptor differs from pinned ELF\n"); return 1;
        }
        uint32_t size = 384;
        memcpy((char*)co + 0x1708, &size, sizeof(size));
    }
    if (plow_hsa_load_code_object(h, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    if (argc > 2 && (strcmp(argv[2], "attention-ps-replay") == 0 || attention_pipeline)) {
        if (argc != (attention_pipeline ? 9 : 5)) {
            fprintf(stderr, "attention replay requires input/output, pipeline also requires reduce fixture/adapter/metadata/reducer\n"); return 1;
        }
        replay_attention_ps(h, argv[3], argv[4], attention_pipeline ? (const char* const*)(argv + 5) : NULL);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "attention-reduce-replay") == 0) {
        if (argc != 6) { fprintf(stderr, "attention-reduce-replay requires partials/metadata/output\n"); return 1; }
        replay_attention_reduce(h, argv[3], argv[4], argv[5]);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "attention-bf16-adapter") == 0) {
        const unsigned cases[][2] = {{1,16}, {8,512}, {16,2048}, {16,8192}, {31,512},
            {32,2048}, {64,8192}, {8,71680}};
        for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
            test_attention_bf16_adapter(h, cases[i][0], cases[i][1]);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "attention-metadata-replay") == 0) {
        if (argc != 5) { fprintf(stderr, "attention-metadata-replay requires stage1/reduce fixtures\n"); return 1; }
        replay_attention_metadata(h, argv[3], argv[4]);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && (strcmp(argv[2], "routed-glu-replay") == 0 || strcmp(argv[2], "routed-down-replay") == 0
                    || strcmp(argv[2], "routed-down-quant-replay") == 0
                    || strcmp(argv[2], "routed-down-atomic-replay") == 0)) {
        const unsigned atomic = strcmp(argv[2], "routed-down-atomic-replay") == 0;
        const unsigned quant = atomic || strcmp(argv[2], "routed-down-quant-replay") == 0;
        if (argc != (atomic ? 8 : quant ? 6 : 5)) { fprintf(stderr, "grouped replay requires input/output, optional hidden and two reference files\n"); return 1; }
        replay_grouped(h, argv[3], argv[4], strcmp(argv[2], "routed-glu-replay") != 0, quant ? argv[5] : NULL,
            atomic ? argv[6] : NULL, atomic ? argv[7] : NULL);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "qb-split-replay") == 0) {
        if (argc != 5) { fprintf(stderr, "qb-split-replay requires reference prefix and output prefix\n"); return 1; }
        replay_qb_split(h, argv[3], argv[4]);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && (strcmp(argv[2], "mla-replay") == 0 || strcmp(argv[2], "mla-qb-replay") == 0)) {
        const unsigned qb = strcmp(argv[2], "mla-qb-replay") == 0;
        if (argc != (qb ? 6 : 5)) { fprintf(stderr, "mla replay requires input/output and optional Q-B fixture\n"); return 1; }
        plow_hsa_kernel kernel;
        check_hsa(plow_hsa_get_kernel(h, 0, qb ? "mla_bmm_fp8_qrope" : "mla_bmm_fp8_m16", &kernel));
        replay_mla(h, &kernel, argv[3], argv[4], qb ? argv[5] : NULL);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && (strcmp(argv[2], "a8w8-replay") == 0 || strcmp(argv[2], "a8w8-m16-replay") == 0
                    || strcmp(argv[2], "a8w8-m16-qkv-replay") == 0
                    || strcmp(argv[2], "a8w8-m16-glu-replay") == 0
                    || strcmp(argv[2], "a8w8-m16-weighted-replay") == 0)) {
        const unsigned split = strcmp(argv[2], "a8w8-m16-qkv-replay") == 0;
        if (argc != (split ? 7 : 5)) { fprintf(stderr, "replay requires input/output and Q/K widths for split output\n"); return 1; }
        plow_hsa_kernel kernel;
        const unsigned glu = strcmp(argv[2], "a8w8-m16-glu-replay") == 0;
        const unsigned weighted = strcmp(argv[2], "a8w8-m16-weighted-replay") == 0;
        check_hsa(plow_hsa_get_kernel(h, 0, weighted ? "gemm_weighted_a8w8_block128_m16" :
            glu ? "gemm_glu_a8w8_block128_m16" :
            split ? "gemm_qkv_a8w8_block128_m16" : strcmp(argv[2], "a8w8-m16-replay") == 0 ? "gemm_a8w8_block128_m16" : "gemm_a8w8_block128", &kernel));
        replay_a8w8(h, &kernel, argv[3], argv[4], glu, weighted,
                     split ? (unsigned)strtoul(argv[5], NULL, 10) : 0,
                     split ? (unsigned)strtoul(argv[6], NULL, 10) : 0);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "quant128") == 0) {
        if (argc != 4) { fprintf(stderr, "quant128 requires capture directory\n"); return 1; }
        a8w8_capture = argv[3];
        plow_hsa_kernel kernel;
        check_hsa(plow_hsa_get_kernel(h, 0, "quant_fp8_block128", &kernel));
        const unsigned rungs[] = {1, 8, 16, 32, 64}, widths[] = {256, 2048, 6144};
        for (unsigned i = 0; i < sizeof(rungs) / sizeof(rungs[0]); i++)
            for (unsigned j = 0; j < sizeof(widths) / sizeof(widths[0]); j++)
                run_quant128(h, &kernel, rungs[i], widths[j]);
        plow_hsa_shutdown(h);
        free(co);
        return 0;
    }
    if (argc > 2 && strcmp(argv[2], "sum4") == 0) {
        plow_hsa_kernel fused, add;
        check_hsa(plow_hsa_get_kernel(h, 0, "sum4_bf16", &fused));
        check_hsa(plow_hsa_get_kernel(h, 0, "gemma_residual_add_bf16", &add));
        const unsigned sizes[] = {1, 7, 8, 9, 2049, 49152, 98304, 131073};
        for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
            run_sum4(h, &fused, &add, sizes[i], 1);
            run_sum4(h, &fused, &add, sizes[i], 32);
        }
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "a8w8-split4") == 0) {
        plow_hsa_kernel kernel;
        check_hsa(plow_hsa_get_kernel(h, 0, "gemm_a8w8_block128_split4", &kernel));
        const unsigned rungs[] = {1, 5, 8, 16, 32, 64, 65};
        for (unsigned i = 0; i < sizeof(rungs) / sizeof(rungs[0]); i++)
            run_a8w8_block_parts(h, &kernel, rungs[i], 130, 512, 4, 0);
        run_a8w8_block_parts(h, &kernel, 8, 6144, 2048, 4, 0);
        run_a8w8_block_parts(h, &kernel, 16, 6144, 2048, 4, 0);
        run_a8w8_block_parts(h, &kernel, 8, 256, 6144, 4, 0);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && strcmp(argv[2], "a8w8-m16-glu") == 0) {
        plow_hsa_kernel kernel;
        check_hsa(plow_hsa_get_kernel(h, 0, "gemm_glu_a8w8_block128_m16", &kernel));
        const unsigned rungs[] = {1, 8, 16, 32, 64};
        for (unsigned i = 0; i < sizeof(rungs) / sizeof(rungs[0]); i++) {
            run_a8w8_block_parts(h, &kernel, rungs[i], 256, 6144, 1, 1);
            run_a8w8_block_parts(h, &kernel, rungs[i], 512, 6144, 1, 1);
        }
        run_a8w8_block_parts(h, &kernel, 3, 130, 256, 1, 1);
        run_a8w8_block_parts(h, &kernel, 3, 130, 260, 1, 1);
        run_a8w8_block_parts(h, &kernel, 65, 129, 129, 1, 1);
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    if (argc > 2 && (strcmp(argv[2], "a8w8") == 0 || strcmp(argv[2], "a8w8-m16") == 0)) {
        a8w8_capture = argc > 3 ? argv[3] : NULL;
        plow_hsa_kernel kernel;
        check_hsa(plow_hsa_get_kernel(h, 0, strcmp(argv[2], "a8w8-m16") == 0
            ? "gemm_a8w8_block128_m16" : "gemm_a8w8_block128", &kernel));
        const unsigned rungs[] = {1, 8, 16, 32, 64};
        for (unsigned i = 0; i < sizeof(rungs) / sizeof(rungs[0]); i++) {
            run_a8w8_block(h, &kernel, rungs[i], 256, 6144);
            run_a8w8_block(h, &kernel, rungs[i], 6144, 256);
        }
        run_a8w8_block(h, &kernel, 8, 2048, 6144);
        run_a8w8_block(h, &kernel, 8, 6144, 2048);
        run_a8w8_block(h, &kernel, 3, 130, 260);
        run_a8w8_block(h, &kernel, 65, 129, 129);
        if (strcmp(argv[2], "a8w8-m16") == 0) {
            a8w8_capture = NULL;
            run_a8w8_block(h, &kernel, 3, 130, 256);
        }
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "gemv_fp8_blk", &k) != 0) {
        fprintf(stderr, "no kernel gemv_fp8_blk: %s\n", plow_hsa_last_error()); return 1;
    }
    if (argc > 2 && strcmp(argv[2], "gemv-batch") == 0) {
        const unsigned rungs[] = {1, 5, 8, 16, 32, 64};
        for (unsigned i = 0; i < sizeof(rungs) / sizeof(rungs[0]); i++) {
            run_rows(h, &k, 256, "shared gate/up", 256, 6144, rungs[i]);
            run_rows(h, &k, 256, "shared down", 6144, 256, rungs[i]);
        }
        plow_hsa_shutdown(h);
        free(co);
        return fails ? 1 : 0;
    }

    const unsigned NCU = 64;
    printf("block-fp8 decode GEMV (weight_block_size [128,128])   [N=out, K=in]:\n");
    /* Real GLM-5.2-FP8 decode GEMV shapes (hidden 6144). N=out-channels, K=in. All 128-multiples
     * except kv_a's N=576 (NB=ceil(576/128)=5). See the design notes. */
    /* --- attention projections --- */
    run(h, &k, NCU, "q_a  6144->2048",   2048,  6144);
    run(h, &k, NCU, "q_b  2048->16384", 16384,  2048);
    run(h, &k, NCU, "kv_a 6144->576",     576,  6144);
    run(h, &k, NCU, "kv_b 512->28672",  28672,   512);  /* narrow-K, CU-starved */
    run(h, &k, NCU, "o    16384->6144",  6144, 16384);
    /* --- dense-MLP (layers 0-2) --- */
    run(h, &k, NCU, "dense g/u 6144->12288", 12288, 6144);
    run(h, &k, NCU, "dense down 12288->6144", 6144, 12288);
    /* --- MoE expert / shared expert (per expert) --- */
    run(h, &k, NCU, "moe g/u 6144->2048",  2048, 6144);
    run(h, &k, NCU, "moe down 2048->6144",  6144, 2048); /* narrow-K, CU-starved */
    /* Ragged shape: not a 128-multiple in either dim — exercises ceil() blocks + overshoot clamp. */
    run(h, &k, NCU, "ragged (ceil+clamp)", 130, 260);

    /* Block-fp8 MoE expert path (GLM-5.2: I_moe=2048, H=6144). */
    plow_hsa_kernel kglu, kdown;
    if (plow_hsa_get_kernel(h, 0, "moe_expert_glu_fp8_blk_k", &kglu) == 0 &&
        plow_hsa_get_kernel(h, 0, "moe_expert_down_fp8_blk_k", &kdown) == 0)
        run_expert(h, &kglu, &kdown, NCU, 2048, 6144);
    else { printf("  no expert kernels\n"); fails++; }

    /* Block-fp8 DENSE MLP gate/up (op 47), GLM-5.2 dense layers 0-2: gate/up 6144->12288 fused. */
    plow_hsa_kernel kdglu;
    if (plow_hsa_get_kernel(h, 0, "dense_glu_fp8_blk_k", &kdglu) == 0)
        run_dense_glu(h, &kdglu, NCU, 12288, 6144);
    else { printf("  no dense-glu kernel\n"); fails++; }

    /* Block-fp8 DENSE PREFILL GEMM (op 107) — the T-row arm GLM_LINEAR_FP8 was blocked on.
     * These are exactly the four projections the knob re-declares, at both TP degrees the campaign
     * runs, plus a ragged N to exercise the ceil() column tail. K is always a 64-multiple because
     * the kernel is instantiated KEXACT and the emitter refuses anything else. */
    plow_hsa_kernel kgblk;
    if (plow_hsa_get_kernel(h, 0, "d_gemm_fp8_blk_k", &kgblk) == 0) {
        printf("block-fp8 DENSE PREFILL GEMM (op 107) vs f64 ref AND vs the decode GEMV:\n");
        /* TP4 (nh_l=16, v_head=256 -> o K=4096; imoe_l=512). M = the emitted bucket ladder. */
        run_gemm_blk(h, &kgblk, &k, NCU, "o_proj TP4",        512, 6144, 4096);
        run_gemm_blk(h, &kgblk, &k, NCU, "o_proj TP4 M=2048", 2048, 6144, 4096);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared gate TP4",    512,  512, 6144);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared down TP4",    512, 6144,  512);
        /* TP8 (nh_l=8 -> o K=2048; imoe_l=256). */
        run_gemm_blk(h, &kgblk, &k, NCU, "o_proj TP8",         512, 6144, 2048);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared gate TP8",    512,  256, 6144);
        run_gemm_blk(h, &kgblk, &k, NCU, "shared down TP8",    512, 6144,  256);
        /* Ragged M and N (K stays a 64-multiple): tile tails on both output axes at once. */
        run_gemm_blk(h, &kgblk, &k, NCU, "ragged M,N tails",   100,  130,  256);
    } else { printf("  no d_gemm_fp8_blk_k kernel\n"); fails++; }

    printf(fails ? "FAIL (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
