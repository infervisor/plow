#include <hip/hip_runtime.h>
#include "dev_isa.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(expr) do { auto e = (expr); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s at %d\n", #expr, hipGetErrorString(e), __LINE__); std::exit(1); } } while (0)
constexpr unsigned GRID = 304, CTX = 81920, GUARD = 128;
template<class T> T* alloc(size_t n) { T* p; CK(hipMalloc(&p, n*sizeof(T))); return p; }
template<class T> T* upload(const std::vector<T>& x) {
    T* p = alloc<T>(x.size()); CK(hipMemcpy(p, x.data(), x.size()*sizeof(T), hipMemcpyHostToDevice)); return p;
}
uint32_t rng = 73;
uint32_t next() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
uint16_t bf16(float f) { uint32_t bits; std::memcpy(&bits, &f, 4); bits += 0x7fff + ((bits >> 16)&1); return bits >> 16; }
float from_bf16(uint16_t b) { uint32_t bits = uint32_t(b) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
float from_fp8(uint8_t b) {
    if ((b & 0x7f) == 0x7f) return NAN;
    const int exp = (b >> 3) & 15, mantissa = b & 7;
    const float value = exp ? std::ldexp(float(8 + mantissa), exp - 10)
                            : std::ldexp(float(mantissa), -9);
    return b & 128 ? -value : value;
}
std::vector<uint16_t> random_bf(size_t n) {
    std::vector<uint16_t> v(n); for (auto& x : v) x = bf16((int(next()%2001)-1000)/1000.f); return v;
}
struct Kernel { const char* label; hipModule_t mod; hipFunction_t fn; unsigned threads; };
Kernel load(const char* label, const char* path, const char* symbol, unsigned threads) {
    Kernel k{label, {}, {}, threads}; CK(hipModuleLoad(&k.mod, path)); CK(hipModuleGetFunction(&k.fn, k.mod, symbol)); return k;
}
void launch(Kernel& k, PlowProgram& p) {
    CK(hipMemsetAsync(p.gq_cursor, 0, 128));
    void* args[] = {&p};
    CK(hipModuleLaunchKernel(k.fn, GRID, 1, 1, k.threads, 1, 1, 0, nullptr, args, nullptr));
}
struct ReferenceRow { size_t row; std::vector<double> value; };
std::vector<ReferenceRow> reference(unsigned rows, unsigned length, float scale,
                                  const std::vector<uint16_t>& qa, const std::vector<uint16_t>& qr,
                                  const std::vector<float>& latent, const std::vector<float>& rope,
                                  const std::vector<float>& scales) {
    std::vector<unsigned> queries{0, rows/2, rows-1};
    queries.erase(std::unique(queries.begin(), queries.end()), queries.end());
    std::vector<ReferenceRow> result;
    for (unsigned q : queries) for (unsigned head : {0u, 3u, 7u}) {
        const size_t row = size_t(q)*8 + head;
        const unsigned visible = length - rows + q + 1;
        double query[576];
        for (unsigned c=0; c<512; ++c) query[c] = from_bf16(qa[row*512+c]);
        for (unsigned c=0; c<64; ++c) query[512+c] = from_bf16(qr[row*64+c]);
        std::vector<double> scores(visible);
        double maximum = -INFINITY;
        for (unsigned k=0; k<visible; ++k) {
            double ck=0, kr=0;
            for (unsigned c=0; c<512; ++c) ck += query[c]*double(latent[size_t(k)*512+c]);
            for (unsigned c=0; c<64; ++c) kr += query[512+c]*double(rope[size_t(k)*64+c]);
            scores[k] = (ck*double(scales[k]) + kr)*double(scale);
            maximum = std::max(maximum, scores[k]);
        }
        double denominator=0;
        std::vector<double> output(512, 0);
        for (unsigned k=0; k<visible; ++k) {
            const double probability = std::exp(scores[k] - maximum);
            denominator += probability;
            const double weighted = probability*double(scales[k]);
            for (unsigned c=0; c<512; ++c) output[c] += weighted*double(latent[size_t(k)*512+c]);
        }
        for (auto& value : output) value /= denominator;
        result.push_back({row, std::move(output)});
    }
    return result;
}
bool reference_selftest() {
    if (from_fp8(0) != 0 || from_fp8(0x38) != 1 || from_fp8(0xb8) != -1 ||
        from_fp8(1) != std::ldexp(1.f, -9) || from_fp8(0x7e) != 448 ||
        !std::signbit(from_fp8(0x80)) || !std::isnan(from_fp8(0x7f))) return false;
    std::vector<uint16_t> qa(2*8*512), qr(2*8*64);
    std::vector<float> latent(3*512), rope(3*64), scales{1,2,3};
    for (unsigned k=0; k<3; ++k)
        std::fill(latent.begin()+k*512, latent.begin()+(k+1)*512, float(k+1)/4);
    const auto result = reference(2, 3, 1.f, qa, qr, latent, rope, scales);
    if (result.size() != 6) return false;
    for (const auto& row : result) {
        const double expected = row.row < 8 ? 0.625 : 3.5/3;
        for (double value : row.value) if (std::abs(value-expected) > 1e-12) return false;
    }
    return true;
}
int main(int argc, char** argv) {
    if (!reference_selftest()) return 8;
    if (argc == 2 && !std::strcmp(argv[1], "--check-reference")) {
        std::puts("PASS: FP8 E4M3FN decoding, causal mask and scaled-value reference");
        return 0;
    }
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s broad8.elf small8.elf flash4.elf split4.elf\n", argv[0]);
        return 2;
    }
    CK(hipInit(0));
    auto broad = load("broad8", argv[1], "plow_interp_gfx942_gq", 512);
    auto lean = load("lean8", argv[2], "plow_interp_mla_small_gfx942_gq", 512);
    auto flash = load("flash4", argv[3], "plow_interp_flash_gfx942_gq", 256);
    auto split = load("split", argv[4], "plow_interp_mla_split_gfx942_gq", 256);
    Kernel kernels[] = {broad, lean, flash, split, split, split, split, split};
    const unsigned splits[] = {1,1,1,2,4,8,16,32};
    constexpr unsigned NK=8;
    std::vector<PlowStreamEnt> entries(GRID);
    for (unsigned i=0; i<GRID; ++i) entries[i].slice = i;
    PlowProgram p{};
    p.gq_stream = upload(entries);
    p.gq_seg_ofs = upload(std::vector<uint32_t>{0,GRID});
    p.gq_cursor = alloc<uint32_t>(32);
    p.n_seg = p.n_gpu = 1;
    auto* ins = alloc<PlowDevInst>(1); p.insts = ins;
    auto* len = alloc<int>(1);
    auto rope_bf = random_bf(CTX*64);
    auto* kr = upload(rope_bf);
    std::vector<float> rope(rope_bf.size());
    std::transform(rope_bf.begin(), rope_bf.end(), rope.begin(), from_bf16);
    std::vector<uint8_t> latent(CTX*512);
    for (auto& x : latent) x = 0x20 + (next()%33) + ((next()&1) << 7);
    auto* ck = upload(latent);
    std::vector<float> latent_f32(latent.size());
    std::transform(latent.begin(), latent.end(), latent_f32.begin(), from_fp8);
    std::vector<float> scales(CTX);
    for (auto& x : scales) x = 0.5f + (next()%101)/100.f;
    auto* scale = upload(scales);
    hipEvent_t start, stop; CK(hipEventCreate(&start)); CK(hipEventCreate(&stop));
    std::puts("rows,context,kernel,splits,median_us,max_abs_vs_broad,rms_vs_broad,max_abs_vs_cpu,cpu_rows");
    for (unsigned rows : {1u,2u,4u,8u,16u,20u,32u,64u,128u,256u,512u,1024u}) {
        const size_t no = size_t(rows)*8*512, nm = size_t(rows)*8*2;
        auto qa_host = random_bf(no), qr_host = random_bf(size_t(rows)*8*64);
        auto* qa = upload(qa_host);
        auto* qr = upload(qr_host);
        auto* out_base = alloc<float>(no*32 + 2*GUARD);
        auto* ml_base = alloc<float>(nm*32 + 2*GUARD);
        auto* out = out_base + GUARD;
        auto* ml = ml_base + GUARD;
        std::vector<void*> table{out,ml,qa,qr,ck,kr,len,scale};
        auto* tensors = upload(table); p.tensors = tensors;
        PlowDevInst d{}; d.op = PLOW_DOP_FLASH_MLA_PREFILL_FP8; d.blocks = GRID;
        for (unsigned i=0; i<8; ++i) d.t[i] = i;
        d.i[0]=1; d.i[1]=8; d.i[2]=CTX; d.i[4]=rows; d.i[5]=0xffffffffu; d.i[7]=4;
        d.fj[0].f = 1.f/std::sqrt(576.f);
        CK(hipMemcpy(ins, &d, sizeof(d), hipMemcpyHostToDevice));
        for (int length : {32,33,128,129,1024,4096,16384,70000,81920}) {
            if (length < int(rows)) continue;
            const auto expected = reference(rows, length, d.fj[0].f, qa_host, qr_host,
                                            latent_f32, rope, scales);
            CK(hipMemcpy(len, &length, 4, hipMemcpyHostToDevice));
            std::vector<std::vector<double>> times(NK);
            std::vector<std::vector<float>> normalized(NK);
            for (unsigned which=0; which<NK; ++which) {
                unsigned ns=splits[which];
                d.fj[2].u = which>=3 ? ns : 0;
                CK(hipMemcpy(ins,&d,sizeof(d),hipMemcpyHostToDevice));
                std::vector<float> poison(no*ns+2*GUARD, NAN), m_poison(nm*ns+2*GUARD,NAN);
                CK(hipMemcpy(out_base,poison.data(),poison.size()*4,hipMemcpyHostToDevice));
                CK(hipMemcpy(ml_base,m_poison.data(),m_poison.size()*4,hipMemcpyHostToDevice));
                launch(kernels[which],p); CK(hipDeviceSynchronize());
                CK(hipMemcpy(poison.data(),out_base,poison.size()*4,hipMemcpyDeviceToHost));
                CK(hipMemcpy(m_poison.data(),ml_base,m_poison.size()*4,hipMemcpyDeviceToHost));
                for (unsigned i=0; i<GUARD; ++i) {
                    if (!std::isnan(poison[i]) || !std::isnan(poison[no*ns+GUARD+i]) ||
                        !std::isnan(m_poison[i]) || !std::isnan(m_poison[nm*ns+GUARD+i])) return 3;
                }
                normalized[which].resize(no);
                for (size_t row=0; row<no/512; ++row) {
                    float maxm=-INFINITY;
                    for (unsigned sp=0; sp<ns; ++sp) maxm=std::max(maxm,m_poison[GUARD+(row*ns+sp)*2]);
                    double den=0; std::vector<double> weights(ns);
                    for (unsigned sp=0; sp<ns; ++sp) {
                        weights[sp]=std::exp2(double(m_poison[GUARD+(row*ns+sp)*2])-maxm);
                        den += weights[sp]*m_poison[GUARD+(row*ns+sp)*2+1];
                    }
                    if (!(den>0) || !std::isfinite(den)) return 4;
                    for (unsigned col=0; col<512; ++col) {
                        double val=0;
                        for (unsigned sp=0; sp<ns; ++sp) val += weights[sp]*poison[GUARD+(row*ns+sp)*512+col];
                        if (!std::isfinite(val)) return 4;
                        normalized[which][row*512+col] = val/den;
                    }
                }
            }
            for (unsigned rep=0; rep<9; ++rep) for (unsigned j=0; j<NK; ++j) {
                unsigned which=(rep+j)%NK;
                d.fj[2].u = which>=3 ? splits[which] : 0;
                CK(hipMemcpy(ins,&d,sizeof(d),hipMemcpyHostToDevice));
                // The reset is before the event so this measures actual interpreter execution.
                CK(hipMemsetAsync(p.gq_cursor,0,128));
                CK(hipEventRecord(start)); void* args[]={&p};
                CK(hipModuleLaunchKernel(kernels[which].fn,GRID,1,1,kernels[which].threads,1,1,0,nullptr,args,nullptr));
                CK(hipEventRecord(stop)); CK(hipEventSynchronize(stop));
                float ms; CK(hipEventElapsedTime(&ms,start,stop)); times[which].push_back(ms*1000.);
            }
            for (unsigned which=0; which<NK; ++which) {
                double mx=0, sum=0, cpu_max=0;
                for (size_t i=0; i<no; ++i) { double delta=normalized[which][i]-normalized[0][i]; mx=std::max(mx,std::abs(delta)); sum+=delta*delta; }
                for (const auto& row : expected) for (unsigned c=0; c<512; ++c) {
                    const double delta = double(normalized[which][row.row*512+c])-row.value[c];
                    if (!std::isfinite(delta)) return 7;
                    cpu_max = std::max(cpu_max, std::abs(delta));
                }
                std::sort(times[which].begin(),times[which].end());
                std::printf("%u,%d,%s,%u,%.6f,%.9g,%.9g,%.9g,%zu\n",rows,length,kernels[which].label,splits[which],times[which][4],mx,std::sqrt(sum/no),cpu_max,expected.size());
                if (which==1 && mx>1e-5) return 5;
                if (which>=2 && mx>0.02) return 6;
                if (!std::isfinite(cpu_max) || cpu_max>0.02) return 7;
            }
            std::fflush(stdout);
        }
        CK(hipFree(qa)); CK(hipFree(qr)); CK(hipFree(out_base)); CK(hipFree(ml_base)); CK(hipFree(tensors));
    }
    CK(hipDeviceSynchronize());
}
