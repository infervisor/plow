/* Host driver for raggedkv_dev.hip. Same .hip/.cpp + hipModuleLoad pattern as
 * runtime/bench/amd/glm52_kbench_fold.{hip,cpp} -- compiling the op_*.h headers into a
 * host+device TU fails on the address-space qualifiers (`cannot pass pointer to address space
 * '1'`), because they are written for the `--genco` device-only compile the real build uses.
 *
 * QUESTION: does ONE flash-MLA-decode launch serve rows with DIFFERENT sequence lengths?
 * That is the prerequisite for packing decode rows into a prefill chunk's M dimension.
 *
 * METHOD, for each row r:
 *    A) one launch, n_batch=B, kv_len = the RAGGED vector
 *    B) one launch, n_batch=B, kv_len = uniform L_r on every row
 * Row r's output slice must be BIT-IDENTICAL. Layout is the same in both; only kv_len differs,
 * so a mismatch means the kernel does not honour kv_len per row, or rows contaminate each other.
 * The control arm at the end compares row 0 against a length it does NOT have: that must DIFFER,
 * or the comparison is vacuous.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));                \
            return 2;                                                                            \
        }                                                                                        \
    } while (0)

static const unsigned DK = 512, DR = 64, B = 4, NH = 8, NSPLIT = 4, KVS = 4096;
static const unsigned THREADS = 512, NBLK = 304;

int main(int argc, char** argv) {
    const char* mod = argc > 1 ? argv[1] : "raggedkv.co";
    hipModule_t m;
    hipFunction_t f;
    CK(hipModuleLoad(&m, mod));
    CK(hipModuleGetFunction(&f, m, "mla_dec"));

    const std::vector<int> ragged = {1024, 37, 4096, 511};  /* uneven on purpose: a tiny row, a
                                                               full ring, and two in between */
    const size_t nq = (size_t)B * NH, nkv = (size_t)B * KVS;
    const size_t nOp = (size_t)B * NH * NSPLIT * DK, nMl = (size_t)B * NH * NSPLIT * 2;

    std::vector<unsigned short> hQa(nq * DK), hQr(nq * DR), hC(nkv * DK), hKr(nkv * DR);
    unsigned s = 0x2545F491u;
    auto rnd = [&]() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (unsigned short)(0x3a00u | (s & 0x1ffu)); /* benign bf16 magnitudes */
    };
    for (auto& v : hQa) v = rnd();
    for (auto& v : hQr) v = rnd();
    for (auto& v : hC) v = rnd();
    for (auto& v : hKr) v = rnd();

    void *dQa, *dQr, *dC, *dKr, *dOp, *dMl, *dLen;
    CK(hipMalloc(&dQa, hQa.size() * 2)); CK(hipMalloc(&dQr, hQr.size() * 2));
    CK(hipMalloc(&dC, hC.size() * 2));   CK(hipMalloc(&dKr, hKr.size() * 2));
    CK(hipMalloc(&dOp, nOp * 4));        CK(hipMalloc(&dMl, nMl * 4));
    CK(hipMalloc(&dLen, B * sizeof(int)));
    CK(hipMemcpy(dQa, hQa.data(), hQa.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dQr, hQr.data(), hQr.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dC, hC.data(), hC.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dKr, hKr.data(), hKr.size() * 2, hipMemcpyHostToDevice));

    unsigned nb = B, nh = NH, kvs = KVS, win = 0, nsp = NSPLIT;
    float scale = 0.1352337788f;
    auto run = [&](const std::vector<int>& lens, std::vector<float>& op, std::vector<float>& ml) {
        if (hipMemcpy(dLen, lens.data(), B * sizeof(int), hipMemcpyHostToDevice) != hipSuccess)
            return 1;
        hipMemset(dOp, 0, nOp * 4);
        hipMemset(dMl, 0, nMl * 4);
        /* PACKED argument buffer, not a kernelParams pointer array. HIP_LAUNCH_PARAM_BUFFER_-
         * POINTER wants the arguments laid out exactly as the kernel's signature, with natural
         * alignment; passing an array of pointers-to-values instead makes the kernel read
         * pointer values as data and fault on a host address (seen: "Memory access fault ... on
         * address 0x7ffc...", a stack pointer). */
        struct __attribute__((packed)) Args {
            void *op, *ml, *qa, *qr, *c, *kr, *len;
            unsigned nb, nh, kvs, win;
            float scale;
            unsigned nsp;
        } a{dOp, dMl, dQa, dQr, dC, dKr, dLen, nb, nh, kvs, win, scale, nsp};
        void* args = &a;
        size_t sz = sizeof(a);
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END};
        if (hipModuleLaunchKernel(f, NBLK, 1, 1, THREADS, 1, 1, 0, nullptr, nullptr, cfg)
            != hipSuccess)
            return 1;
        if (hipDeviceSynchronize() != hipSuccess) return 1;
        op.resize(nOp); ml.resize(nMl);
        hipMemcpy(op.data(), dOp, nOp * 4, hipMemcpyDeviceToHost);
        hipMemcpy(ml.data(), dMl, nMl * 4, hipMemcpyDeviceToHost);
        return 0;
    };

    std::vector<float> opR, mlR;
    if (run(ragged, opR, mlR)) { printf("FAIL: ragged launch\n"); return 2; }
    const size_t opRow = nOp / B, mlRow = nMl / B;
    int bad = 0;
    printf("ragged kv_len = [%d, %d, %d, %d]   B=%u NH=%u nsplit=%u ring=%u\n", ragged[0],
           ragged[1], ragged[2], ragged[3], B, NH, NSPLIT, KVS);
    for (unsigned r = 0; r < B; r++) {
        std::vector<int> uni(B, ragged[r]);
        std::vector<float> opU, mlU;
        if (run(uni, opU, mlU)) { printf("FAIL: uniform launch\n"); return 2; }
        size_t d = 0;
        for (size_t i = 0; i < opRow; i++)
            if (((unsigned*)opR.data())[r * opRow + i] != ((unsigned*)opU.data())[r * opRow + i]) d++;
        for (size_t i = 0; i < mlRow; i++)
            if (((unsigned*)mlR.data())[r * mlRow + i] != ((unsigned*)mlU.data())[r * mlRow + i]) d++;
        printf("  row %u (kv_len %4d): %-14s %zu/%zu words differ\n", r, ragged[r],
               d ? "MISMATCH" : "BIT-IDENTICAL", d, opRow + mlRow);
        bad += (d != 0);
    }
    {
        std::vector<int> wrong(B, ragged[1]);
        std::vector<float> opW, mlW;
        if (run(wrong, opW, mlW)) return 2;
        size_t d = 0;
        for (size_t i = 0; i < opRow; i++)
            if (((unsigned*)opR.data())[i] != ((unsigned*)opW.data())[i]) d++;
        printf("  control row 0 vs kv_len=%d: %zu/%zu differ -> gate %s\n", ragged[1], d, opRow,
               d ? "CAN FAIL" : "IS VACUOUS");
        if (!d) bad++;
    }
    printf(bad ? "\nRAGGED KV: FAIL\n"
               : "\nRAGGED KV: PASS -- ONE launch serves per-row sequence lengths\n");
    return bad ? 1 : 0;
}
