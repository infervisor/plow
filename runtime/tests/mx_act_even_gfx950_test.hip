/* mx_act_even_gfx950_test.hip — the PLOW_MX_ACT_EVEN activation quantizer (amd_common.h) as the
 * A4W4 MoE prefill runs it: `mpf4_quant_commit` (bf16 staging, GLU A-side) and `mpf4_quant_f32`
 * (f32 GLU bridge -> DOWN A-side), `mx_even_fq32` (the PLOW_MOE_DEC_A4 decode fake-quant), plus
 * gfx950's v_cvt_scalef32_pk_fp4_f32 at the same scale (the element convert AITER uses), on blocks
 * from mx_act_even_ref.py. The numpy reference there
 * decides pass/fail: codes and scale bytes must be bit-identical.
 *
 *   python3 runtime/tests/mx_act_even_ref.py gen /tmp/mxa
 *   hipcc --offload-arch=gfx950 -O2 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/mx_act_even_gfx950_test.hip -o /tmp/mxa/t
 *   /tmp/mxa/t /tmp/mxa            (GPU, via scripts/bench/gpuq.py)
 *   python3 runtime/tests/mx_act_even_ref.py check /tmp/mxa
 */
#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#define PLOW_MX_ACT_EVEN 1
#include <vector>
#include <cstring>
#include <string>
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1
#include <cstdio>
#include <cstdlib>

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

/* out per block (128 B): [0,16) staging codes, 16 staging scale, [17,33) bridge codes, 33 bridge
 * scale, [34,50) hw-cvt codes at the bridge scale, [64,128) fake-quant bf16. */
__global__ void quant(const float* x, unsigned char* out, unsigned n) {
    const unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    const float* v = x + (size_t)b * 32;
    unsigned char* o = out + (size_t)b * 128;
    mpf4_a32 a;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) {
            const unsigned u = __builtin_bit_cast(unsigned, v[i * 8 + j]);
            a.w[i][j] = (bf16)(u >> 16); /* bf16-exact inputs */
        }
    unsigned char sa, q16[16] __attribute__((aligned(16)));
    mpf4_quant_commit(a, q16, &sa);
    for (int i = 0; i < 16; i++) o[i] = q16[i];
    o[16] = sa;
    float f[32];
    for (int i = 0; i < 32; i++) f[i] = v[i];
    mpf4_b16 q;
    const unsigned char sb = mpf4_quant_f32(f, q);
    memcpy(o + 17, &q, 16);
    o[33] = sb;
    for (int k = 0; k < 4; k++) {
        unsigned w = 0;
        w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, f[k * 8 + 0], f[k * 8 + 1], e8m0_to_f32(sb), 0);
        w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, f[k * 8 + 2], f[k * 8 + 3], e8m0_to_f32(sb), 1);
        w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, f[k * 8 + 4], f[k * 8 + 5], e8m0_to_f32(sb), 2);
        w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, f[k * 8 + 6], f[k * 8 + 7], e8m0_to_f32(sb), 3);
        memcpy(o + 34 + 4 * k, &w, 4);
    }
    mx_even_fq32(a.w[0], a.w[1], a.w[2], a.w[3]);
    memcpy(o + 64, &a, 64);
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <dir with blocks.f32>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    FILE* fi = fopen((dir + "/blocks.f32").c_str(), "rb");
    if (!fi) { printf("no %s/blocks.f32\n", argv[1]); return 2; }
    std::vector<float> x;
    float t;
    while (fread(&t, 4, 1, fi) == 1) x.push_back(t);
    fclose(fi);
    const unsigned n = (unsigned)(x.size() / 32);
    float* dx; unsigned char* dout;
    HC(hipMalloc(&dx, x.size() * 4));
    HC(hipMalloc(&dout, (size_t)n * 128));
    HC(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    quant<<<(n + 63) / 64, 64>>>(dx, dout, n);
    HC(hipDeviceSynchronize());
    std::vector<unsigned char> out((size_t)n * 128);
    HC(hipMemcpy(out.data(), dout, out.size(), hipMemcpyDeviceToHost));
    FILE* fo = fopen((dir + "/device.u8").c_str(), "wb");
    fwrite(out.data(), 1, out.size(), fo);
    fclose(fo);
    printf("%u blocks -> %s/device.u8\n", n, argv[1]);
    return 0;
}
