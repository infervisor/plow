// Bit-exactness of the SplitZip round trip on a real Qwen3-4B tensor,
// with negative controls (corrupt code table / corrupt escape entry).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__constant__ unsigned char c_tab8[16];

__global__ __launch_bounds__(256) void k_store(const uint4* __restrict__ lo,
        const uint2* __restrict__ cd, size_t nvec, uint4* __restrict__ out) {
    __shared__ unsigned char stab[16];
    if (threadIdx.x < 16) stab[threadIdx.x] = c_tab8[threadIdx.x];
    __syncthreads();
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 l = lo[i]; uint2 c = cd[i];
        unsigned int lw[4] = {l.x, l.y, l.z, l.w}, cw[2] = {c.x, c.y};
        unsigned short o[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            unsigned int b  = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
            unsigned int cc = (cw[e >> 3] >> ((e & 7) * 4)) & 0xFu;
            unsigned int ex = stab[cc];
            o[e] = (unsigned short)(((b & 0x80u) << 8) | (ex << 7) | (b & 0x7Fu));
        }
        out[2 * i] = *(uint4*)&o[0];
        out[2 * i + 1] = *(uint4*)&o[8];
    }
}
__global__ void k_escape(const unsigned int* p, const unsigned short* v,
                         unsigned int n, unsigned short* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[p[i]] = v[i];
}

int main() {
    FILE* f = fopen("qwen_gate17.bin", "rb");
    if (!f) { printf("missing input\n"); return 1; }
    fseek(f, 0, SEEK_END); size_t nb = ftell(f); fseek(f, 0, SEEK_SET);
    size_t n = nb / 2; n = n / 16 * 16;              // whole 16-element groups
    std::vector<unsigned short> orig(n);
    if (fread(orig.data(), 2, n, f) != n) return 1;
    fclose(f);
    printf("tensor: %zu bf16 elements (%.2f MB)\n", n, n * 2 / 1048576.0);

    // ---- host compress, GLOBAL table = exponents 109..124 (measured) ----
    unsigned char tab[16]; for (int i = 0; i < 16; ++i) tab[i] = 109 + i;
    std::vector<unsigned char> lo(n), cd(n / 2, 0);
    std::vector<unsigned int> epos; std::vector<unsigned short> eval;
    for (size_t i = 0; i < n; ++i) {
        unsigned short u = orig[i];
        unsigned int ex = (u >> 7) & 0xFFu;
        lo[i] = (unsigned char)(((u >> 8) & 0x80u) | (u & 0x7Fu));
        int code = (ex >= 109 && ex <= 124) ? (int)ex - 109 : 0;
        if (ex < 109 || ex > 124) { epos.push_back((unsigned int)i); eval.push_back(u); }
        cd[i / 2] |= (unsigned char)(code << ((i & 1) * 4));
    }
    double comp = (double)n + n / 2.0 + 16 + epos.size() * 6.0;
    printf("escapes: %zu (%.5f%%)  compressed=%.2f MB  RATIO=%.4f\n",
           epos.size(), 100.0 * epos.size() / n, comp / 1048576.0, (double)(n * 2) / comp);

    unsigned char *d_lo, *d_cd; unsigned short *d_out, *d_ev; unsigned int* d_ep;
    CK(cudaMalloc(&d_lo, n)); CK(cudaMalloc(&d_cd, n / 2)); CK(cudaMalloc(&d_out, n * 2));
    CK(cudaMalloc(&d_ep, epos.size() * 4 + 4)); CK(cudaMalloc(&d_ev, eval.size() * 2 + 2));
    CK(cudaMemcpy(d_lo, lo.data(), n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cd, cd.data(), n / 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ep, epos.data(), epos.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ev, eval.data(), eval.size() * 2, cudaMemcpyHostToDevice));

    std::vector<unsigned short> got(n);
    auto roundtrip = [&](const char* label, int corrupt_tab, int corrupt_esc) {
        unsigned char t[16]; memcpy(t, tab, 16);
        if (corrupt_tab) t[5] = 108;                       // negative control 1
        CK(cudaMemcpyToSymbol(c_tab8, t, 16));
        if (corrupt_esc && eval.size()) {                  // negative control 2
            unsigned short bad = eval[0] ^ 0x0001;
            CK(cudaMemcpy(d_ev, &bad, 2, cudaMemcpyHostToDevice));
        } else if (eval.size()) {
            CK(cudaMemcpy(d_ev, eval.data(), 2, cudaMemcpyHostToDevice));
        }
        k_store<<<1020, 256>>>((uint4*)d_lo, (uint2*)d_cd, n / 16, (uint4*)d_out);
        if (epos.size())
            k_escape<<<(epos.size() + 255) / 256, 256>>>(d_ep, d_ev,
                       (unsigned int)epos.size(), d_out);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(got.data(), d_out, n * 2, cudaMemcpyDeviceToHost));
        int cmp = memcmp(got.data(), orig.data(), n * 2);
        size_t bad = 0; for (size_t i = 0; i < n; ++i) bad += (got[i] != orig[i]);
        printf("%-46s memcmp=%-6s mismatching_elements=%zu\n", label,
               cmp == 0 ? "EQUAL" : "DIFF", bad);
        return cmp == 0;
    };

    bool ok  = roundtrip("clean round trip (expect EQUAL)", 0, 0);
    bool n1  = roundtrip("NEG CONTROL: code table[5] 109->108", 1, 0);
    bool n2  = roundtrip("NEG CONTROL: escape value flipped 1 bit", 0, 1);
    roundtrip("restored (expect EQUAL again)", 0, 0);
    printf("\nVERDICT: %s\n", (ok && !n1 && !n2) ?
           "PASS - bit exact, and both negative controls detected"
         : "FAIL");
    return 0;
}
