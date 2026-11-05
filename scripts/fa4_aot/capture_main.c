// Capture harness: one exported FA4 entry, called exactly as runtime/nvidia/fa4_hd{512,256}.cpp call it.
// argv: total_q nreq max_pages num_pages head_stride splits
// Build with -DE=<entry prefix without plow_> -DKIND=0 whole / 1 split / 2 combine, -DH -DKVH -DD.
#include <stdlib.h>
#include <stdint.h>
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define T(x) CAT(CAT(E, _), x)
#define STR_(x) #x
#define STR(x) STR_(x)
#include STR(HDR)
#include <string.h>
// Fill the stack the host code will reuse with PLOW_CAP_FILL, so padding bytes (never written)
// differ between two captures and can be masked.
__attribute__((noinline)) static void scribble(int v) {
    volatile unsigned char buf[1 << 16];
    memset((void*)buf, v, sizeof buf);
}
int main(int argc, char** argv) {
    scribble(getenv("PLOW_CAP_FILL") ? atoi(getenv("PLOW_CAP_FILL")) : 0);
    int total_q = atoi(argv[1]), nreq = atoi(argv[2]), max_pages = atoi(argv[3]), num_pages = atoi(argv[4]);
    long long hs = atoll(argv[5]); int splits = atoi(argv[6]);
    T(Kernel_Module_t) m;
    T(Kernel_Module_Load)(&m);
    char *Q = (char*)0x7f0000000000ULL, *K = (char*)0x7e0000000000ULL, *V = (char*)0x7d0000000000ULL;
    char *O = (char*)0x7c0000000000ULL, *L = (char*)0x7b0000000000ULL, *CU = (char*)0x7a0000000000ULL;
    char *UK = (char*)0x790000000000ULL, *PT = (char*)0x780000000000ULL;
    long long os = (long long)total_q * H * D, ls = (long long)H * total_q;
#if KIND == 2
    T(Tensor_o_part_t) cp = {O, {splits, total_q, H, D}, {os}};
    T(Tensor_lse_part_t) cl = {L, {splits, total_q, H}, {ls, total_q}};
    T(Tensor_o_t) co = {Q, {total_q, H, D}};
    T(Tensor_cuq_t) cc = {CU, {nreq + 1}};
    return CAT(CAT(cute_dsl_, E), _wrapper)(&m, &cp, &cl, &co, &cc, (cudaStream_t)0x3000);
#else
    T(Tensor_q_t) tq = {Q, {total_q, H, D}};
#if HM
    T(Tensor_k_t) tk = {K, {num_pages, 64, KVH, D}, {hs}};
    T(Tensor_v_t) tv = {V, {num_pages, 64, KVH, D}, {hs}};
#else
    T(Tensor_k_t) tk = {K, {num_pages, 64, KVH, D}};
    T(Tensor_v_t) tv = {V, {num_pages, 64, KVH, D}};
#endif
    T(Tensor_cuq_t) tc = {CU, {nreq + 1}};
    T(Tensor_usedk_t) tu = {UK, {nreq}};
    T(Tensor_pt_t) tp = {PT, {nreq, max_pages}, {max_pages}};
#if KIND == 0
    T(Tensor_o_t) to = {O, {total_q, H, D}};
    return CAT(CAT(cute_dsl_, E), _wrapper)(&m, &tq, &tk, &tv, &to, &tc, &tu, &tp, 1.0f, (cudaStream_t)0x3000);
#else
    T(Tensor_o_t) to = {O, {splits, total_q, H, D}, {os}};
    T(Tensor_lse_t) tl = {L, {splits, H, total_q}, {ls, total_q}};
    return CAT(CAT(cute_dsl_, E), _wrapper)(&m, &tq, &tk, &tv, &to, &tl, &tc, &tu, &tp, 1.0f, (cudaStream_t)0x3000);
#endif
#endif
}
