// Capture harness: the exported GDN entry called exactly as plowrt's host stub
// (device/cuda/qwen_gdn.rs) calls it. argv: tokens nseq q k v o alpha beta state initial maps offsets
// (addresses as decimal integers; never dereferenced).
#include <stdlib.h>
#include <string.h>
#include "gdn_sm90.h"
__attribute__((noinline)) static void scribble(int v) {
    volatile unsigned char buf[1 << 16];
    memset((void*)buf, v, sizeof buf);
}
int main(int argc, char** argv) {
    if (argc != 13) return 2;
    scribble(getenv("PLOW_CAP_FILL") ? atoi(getenv("PLOW_CAP_FILL")) : 0);
    long long tokens = atoll(argv[1]), nseq = atoll(argv[2]);
    void* p[10];
    for (int i = 0; i < 10; i++) p[i] = (void*)(uintptr_t)strtoull(argv[3 + i], 0, 10);
    plow_gdn_sm90_Kernel_Module_t m;
    plow_gdn_sm90_Kernel_Module_Load(&m);
    plow_gdn_sm90_Tensor_q_t qd = {p[0], {tokens}};
    plow_gdn_sm90_Tensor_k_t kd = {p[1], {tokens}};
    plow_gdn_sm90_Tensor_v_t vd = {p[2], {tokens}};
    plow_gdn_sm90_Tensor_o_t od = {p[3], {tokens}};
    plow_gdn_sm90_Tensor_alpha_t ad = {p[4], {tokens * 48}};
    plow_gdn_sm90_Tensor_beta_t bd = {p[5], {tokens * 48}};
    plow_gdn_sm90_Tensor_state_t sd = {p[6], {nseq * 48 * 128 * 128}};
    plow_gdn_sm90_Tensor_initial_t id = {p[7], {nseq * 48 * 128 * 128}};
    plow_gdn_sm90_Tensor_maps_t md = {p[8]};
    plow_gdn_sm90_Tensor_offsets_t cd = {p[9], {nseq + 1}};
    return cute_dsl_plow_gdn_sm90_wrapper(&m, &qd, &kd, &vd, &od, &ad, &bd, &sd, &id, &md, &cd,
        0.08838834764831845f, (int)nseq, (cudaStream_t)0x3000);
}
