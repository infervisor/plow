// Exact-object replay of the GLM-5.3 TP8 decode attention chain on gfx942:
//   FlashMlaDecodeFp8 (sparse gather / dense), MlaMergeFold, IndexScore, IndexSelect (local),
//   HeadNormRopeFp8 (KV writer), plus the AITER QH8 qseqlen1 object as the sparse alternative.
// Every interpreter op runs through the shipped decode object (plow_interp_dec_gfx942_gq),
// timings are medians of 9 launches, outputs are NaN-poisoned with guard bands and compared
// against an FP64 host reference on sampled rows.
#include "ladder_common.h"
#include <map>
#include <set>

static constexpr unsigned STRIDE = 81920, DK = 512, DR = 64, NH = 8, TOPK = 2048, VD = 256;

static std::vector<unsigned> parse_list(const char* s) {
    std::vector<unsigned> v; std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || !*p) { if (!cur.empty()) v.push_back(std::stoul(cur)); cur.clear(); if (!*p) break; }
        else cur += *p;
    }
    return v;
}
static unsigned gcd_u(unsigned a, unsigned b) { while (b) { unsigned t = a % b; a = b; b = t; } return a; }
static unsigned glm_nsplit(unsigned ctx, unsigned heads) {
    const unsigned n_grp = std::max(heads / 4u, 1u), fill = std::max((256 + n_grp - 1) / n_grp, 1u);
    const unsigned kv_tiles = std::max((ctx + 31) / 32, 1u);
    unsigned ns = std::max(ctx / 256u, 16u); ns = std::min(ns, 64u); ns = std::min(ns, fill); ns = std::min(ns, kv_tiles);
    return std::max(ns, 1u);
}
static unsigned fold_blocks(unsigned bh, unsigned v) {
    auto vt_of = [&](unsigned nblk) { return (bh && bh * 8 <= nblk) ? 32u : ((v >= 128 && v < 256) ? 128u : 256u); };
    const unsigned vt = vt_of(GRID);
    const unsigned need = std::min(std::max(bh * ((v + vt - 1) / vt), 1u), GRID);
    return vt_of(need) != vt ? GRID : need;
}

// FP64 reference attention for one (batch row, head) over an explicit key list.
static void ref_attention(const uint16_t* qa, const uint16_t* qr, const float* lat, const float* rope,
                          const float* sc, const std::vector<unsigned>& keys, float scale, std::vector<double>& out) {
    double q[576];
    for (unsigned c = 0; c < DK; ++c) q[c] = from_bf16(qa[c]);
    for (unsigned c = 0; c < DR; ++c) q[DK + c] = from_bf16(qr[c]);
    std::vector<double> s(keys.size());
    double mx = -INFINITY;
    for (size_t i = 0; i < keys.size(); ++i) {
        const unsigned k = keys[i];
        double dc = 0, dr = 0;
        for (unsigned c = 0; c < DK; ++c) dc += q[c] * lat[size_t(k) * DK + c];
        for (unsigned c = 0; c < DR; ++c) dr += q[DK + c] * rope[size_t(k) * DR + c];
        s[i] = (dc * sc[k] + dr) * scale;
        mx = std::max(mx, s[i]);
    }
    out.assign(DK, 0.0);
    double den = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const double p = std::exp(s[i] - mx);
        den += p;
        const double w = p * sc[keys[i]];
        const float* row = lat + size_t(keys[i]) * DK;
        for (unsigned c = 0; c < DK; ++c) out[c] += w * row[c];
    }
    for (auto& v : out) v /= den;
}

// ---- the AITER QH8 qseqlen1 object (BF16 packed K/V, CSR of TOPK keys per row) ----
__global__ void pack_selected(uint16_t* __restrict__ kv, const uint8_t* __restrict__ lat, const uint16_t* __restrict__ rope,
                              const float* __restrict__ sc, const int* __restrict__ idx, unsigned rows) {
    // two selected rows per wave: lanes 0-31 row A, 32-63 row B; 16 fp8 bytes per lane, 2 x 16 B bf16 stores
    const unsigned g = (blockIdx.x * blockDim.x + threadIdx.x) >> 6, lane = threadIdx.x & 63;
    const unsigned half = lane >> 5, l32 = lane & 31;
    const unsigned total = rows * TOPK;
    for (unsigned r2 = g; r2 * 2 < total; r2 += (gridDim.x * blockDim.x) >> 6) {
        const unsigned r = r2 * 2 + half;
        if (r >= total) continue;
        const unsigned b = r / TOPK;
        const int ip = idx[r];
        uint16_t* dst = kv + (size_t)r * 576;
        if (ip < 0) {
            for (unsigned c = l32; c < 576 / 8; c += 32) ((uint4*)dst)[c] = make_uint4(0, 0, 0, 0);
            continue;
        }
        const size_t src_row = (size_t)b * STRIDE + (unsigned)ip;
        const float s = sc[src_row];
        const uint4 raw = ((const uint4*)(lat + src_row * DK))[l32];
        const uint8_t* x8 = (const uint8_t*)&raw;
        uint32_t packed[8];
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint8_t x = x8[j];
            const int e = (x >> 3) & 15, m = x & 7;
            float v = e ? ldexpf(float(8 + m), e - 10) : ldexpf(float(m), -9);
            if (x & 128) v = -v;
            v *= s;
            uint32_t bits; __builtin_memcpy(&bits, &v, 4);
            bits += 0x7fff + ((bits >> 16) & 1);
            const uint32_t h = bits >> 16;
            if (j & 1) packed[j >> 1] |= h << 16; else packed[j >> 1] = h;
        }
        uint4* d4 = (uint4*)(dst + l32 * 16);
        d4[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        d4[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        if (l32 < 8) ((uint4*)(dst + DK))[l32] = ((const uint4*)(rope + src_row * DR))[l32];
    }
}
__global__ void reduce_splits(float* __restrict__ out, float* __restrict__ ml, const float* __restrict__ part,
                              const float* __restrict__ lse, unsigned rows, unsigned ns) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * NH * (DK / 4); i += blockDim.x * gridDim.x) {
        const unsigned hrow = i / (DK / 4), col = i % (DK / 4);
        const unsigned row = hrow / NH, head = hrow % NH;
        float m = -INFINITY;
        for (unsigned s = 0; s < ns; ++s) m = fmaxf(m, lse[(row * ns + s) * NH + head]);
        float4 acc = make_float4(0, 0, 0, 0); float den = 0;
        for (unsigned s = 0; s < ns; ++s) {
            const float w = expf(lse[(row * ns + s) * NH + head] - m);
            const float4 p = ((const float4*)part)[((row * ns + s) * NH + head) * (DK / 4) + col];
            acc.x += w * p.x; acc.y += w * p.y; acc.z += w * p.z; acc.w += w * p.w; den += w;
        }
        ((float4*)out)[hrow * (DK / 4) + col] = make_float4(acc.x / den, acc.y / den, acc.z / den, acc.w / den);
        if (col == 0) { ml[hrow * 2] = 0.f; ml[hrow * 2 + 1] = 1.f; }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <objdir> [--rows a,b] [--ns a,b] [--ctx a,b] [--ops flash,merge,dense,score,select,write,aiter] [--aiter path.co]\n", argv[0]); return 2; }
    std::string objdir = argv[1];
    std::vector<unsigned> rows_list{1, 2, 4, 8, 16, 20}, ns_list{1, 2, 4, 8, 16}, ctx_list{1024, 4096, 16384, 32768, 65536, 81920};
    std::set<std::string> ops{"flash", "merge", "dense", "score", "select", "write", "aiter"};
    std::string aiter_path = objdir + "/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co";
    for (int a = 2; a + 1 < argc; a += 2) {
        std::string k = argv[a];
        if (k == "--rows") rows_list = parse_list(argv[a + 1]);
        else if (k == "--ns") ns_list = parse_list(argv[a + 1]);
        else if (k == "--ctx") ctx_list = parse_list(argv[a + 1]);
        else if (k == "--aiter") aiter_path = argv[a + 1];
        else if (k == "--ops") { ops.clear(); std::string cur; for (const char* p = argv[a + 1];; ++p) { if (*p == ',' || !*p) { if (!cur.empty()) ops.insert(cur); cur.clear(); if (!*p) break; } else cur += *p; } }
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    CK(hipInit(0));
    Kernel dec = load_kernel("decode8", objdir + "/interp_decode_fp8kv_gq.elf", "plow_interp_dec_gfx942_gq", 512);
    hipModule_t amod{}; hipFunction_t afn{};
    if (ops.count("aiter")) {
        CK(hipModuleLoad(&amod, aiter_path.c_str()));
        CK(hipModuleGetFunction(&afn, amod, "_ZN5aiter36mla_a16w16_qh8_qseqlen1_gqaratio8_v3E"));
    }
    Replay rp;
    const float scale = 1.f / std::sqrt(576.f);
    std::puts("op,rows,ctx,arm,ns,gf,blocks,median_us,max_abs,rel_l2,note");
    int rc = 0;
    for (unsigned R : rows_list) {
        // ---- per-slot caches (R slots, STRIDE rows each) ----
        auto lat8 = random_fp8(size_t(R) * STRIDE * DK);
        auto rope_h = random_bf(size_t(R) * STRIDE * DR);
        auto sc_h = random_scales(size_t(2) * R * STRIDE);
        std::vector<float> lat_f(lat8.size()), rope_f(rope_h.size());
        std::transform(lat8.begin(), lat8.end(), lat_f.begin(), from_fp8);
        std::transform(rope_h.begin(), rope_h.end(), rope_f.begin(), from_bf16);
        auto qa_h = random_bf(size_t(R) * NH * DK), qr_h = random_bf(size_t(R) * NH * DR);
        auto* d_lat = upload(lat8); auto* d_rope = upload(rope_h); auto* d_sc = upload(sc_h);
        auto* d_qa = upload(qa_h); auto* d_qr = upload(qr_h);
        auto* d_len = dalloc<int>(R);
        auto* d_idx = dalloc<int>(size_t(R) * TOPK);
        const size_t no_max = size_t(R) * NH * 128 * DK, nm_max = size_t(R) * NH * 128 * 2;
        auto* d_out = dalloc<float>(no_max + 2 * GUARD);
        auto* d_ml = dalloc<float>(nm_max + 2 * GUARD);
        auto* d_oat = dalloc<uint16_t>(size_t(R) * NH * VD + 2 * GUARD);
        auto wuv_h = random_bf(size_t(NH) * DK * VD);
        auto* d_wuv = upload(wuv_h);
        std::vector<void*> table{d_out + GUARD, d_ml + GUARD, d_qa, d_qr, d_lat, d_rope, d_len, d_sc, d_idx, d_oat + GUARD, d_wuv};
        // 0 out, 1 ml, 2 qa, 3 qr, 4 ckv, 5 krope, 6 kvlen, 7 scale, 8 idx, 9 oat, 10 wuv
        auto* d_table = upload(table);
        rp.set_tensors(d_table);
        std::vector<unsigned> sample_b{0u, R / 2, R - 1};
        sample_b.erase(std::unique(sample_b.begin(), sample_b.end()), sample_b.end());
        const unsigned sample_h[3] = {0, 3, 7};

        auto run_flash_and_check = [&](const char* opname, unsigned ctx, const char* arm, unsigned ns, unsigned gf,
                                       bool gather, const std::vector<std::vector<unsigned>>& keys_of_b,
                                       std::vector<float>& out_host, std::vector<float>& ml_host) -> bool {
            PlowDevInst d{}; d.op = PLOW_DOP_FLASH_MLA_DECODE_FP8;
            const unsigned n_work = R * (NH / gf) * ns;
            d.blocks = std::min(n_work, GRID);
            for (unsigned i = 0; i < 8; ++i) d.t[i] = i;
            d.i[0] = R; d.i[1] = NH; d.i[2] = STRIDE; d.i[3] = 0; d.i[4] = ns; d.i[5] = 0xffffffffu; d.i[7] = gf;
            d.fj[0].f = scale;
            if (gather) { d.fj[1].u = 8 + 1; d.i[6] = TOPK; } else { d.fj[1].u = 0; d.i[6] = 0; }
            rp.set_inst(d);
            const size_t no = size_t(R) * NH * ns * DK, nm = size_t(R) * NH * ns * 2;
            poison(d_out, no + 2 * GUARD); poison(d_ml, nm + 2 * GUARD);
            rp.run(dec); CK(hipDeviceSynchronize());
            out_host.assign(no + 2 * GUARD, 0); ml_host.assign(nm + 2 * GUARD, 0);
            download(out_host, d_out); download(ml_host, d_ml);
            bool ok = guards_intact(out_host, no) && guards_intact(ml_host, nm);
            double mx = 0, e2 = 0, n2 = 0;
            std::vector<double> merged, ref;
            for (unsigned b : sample_b) for (unsigned h : sample_h) {
                const size_t bh = size_t(b) * NH + h;
                merge_partials(out_host.data() + GUARD + bh * ns * DK, ml_host.data() + GUARD + bh * ns * 2, ns, DK, merged);
                ref_attention(qa_h.data() + bh * DK, qr_h.data() + bh * DR, lat_f.data() + size_t(b) * STRIDE * DK,
                              rope_f.data() + size_t(b) * STRIDE * DR, sc_h.data() + size_t(b) * STRIDE, keys_of_b[b], scale, ref);
                for (unsigned c = 0; c < DK; ++c) {
                    const double dlt = merged[c] - ref[c];
                    if (!std::isfinite(dlt)) ok = false;
                    mx = std::max(mx, std::abs(dlt)); e2 += dlt * dlt; n2 += ref[c] * ref[c];
                }
            }
            const double us = rp.time_us(dec);
            std::printf("%s,%u,%u,%s,%u,%u,%u,%.3f,%.6g,%.6g,%s\n", opname, R, ctx, arm, ns, gf, d.blocks, us, mx,
                        std::sqrt(e2 / std::max(n2, 1e-300)), ok && mx <= 0.02 ? "PASS" : "FAIL");
            std::fflush(stdout);
            return ok && mx <= 0.02;
        };
        auto run_merge_fold = [&](unsigned ctx, const char* arm, unsigned ns, const std::vector<float>& out_host,
                                  const std::vector<float>& ml_host) {
            PlowDevInst d{}; d.op = PLOW_DOP_MLA_MERGE_FOLD;
            d.blocks = fold_blocks(R * NH, VD);
            d.t[0] = 9; d.t[1] = 0; d.t[2] = 1; d.t[3] = 10;
            d.i[0] = R; d.i[1] = NH; d.i[2] = VD; d.i[4] = ns;
            rp.set_inst(d);
            const size_t n = size_t(R) * NH * VD;
            std::vector<uint16_t> pz(n + 2 * GUARD, 0x7fc0);
            CK(hipMemcpy(d_oat, pz.data(), pz.size() * 2, hipMemcpyHostToDevice));
            rp.run(dec); CK(hipDeviceSynchronize());
            download(pz, d_oat);
            bool ok = true;
            for (unsigned i = 0; i < GUARD; ++i) if (pz[i] != 0x7fc0 || pz[GUARD + n + i] != 0x7fc0) ok = false;
            // reference: CPU merge of the SAME partials, FP64 fold with W_uv
            double mx = 0, e2 = 0, n2 = 0;
            std::vector<double> merged;
            for (unsigned b : sample_b) for (unsigned h : sample_h) {
                const size_t bh = size_t(b) * NH + h;
                merge_partials(out_host.data() + GUARD + bh * ns * DK, ml_host.data() + GUARD + bh * ns * 2, ns, DK, merged);
                for (unsigned v = 0; v < VD; ++v) {
                    double acc = 0;
                    for (unsigned l = 0; l < DK; ++l) acc += merged[l] * from_bf16(wuv_h[(size_t(h) * DK + l) * VD + v]);
                    const double got = from_bf16(pz[GUARD + bh * VD + v]);
                    const double dlt = got - acc;
                    if (!std::isfinite(dlt)) ok = false;
                    mx = std::max(mx, std::abs(dlt)); e2 += dlt * dlt; n2 += acc * acc;
                }
            }
            const double rel = std::sqrt(e2 / std::max(n2, 1e-300));
            const double us = rp.time_us(dec);
            std::printf("merge_fold,%u,%u,%s,%u,4,%u,%.3f,%.6g,%.6g,%s\n", R, ctx, arm, ns, d.blocks, us, mx, rel,
                        ok && rel <= 0.01 ? "PASS" : "FAIL");
            std::fflush(stdout);
            if (!(ok && rel <= 0.01)) rc = 5;
        };

        for (unsigned ctx : ctx_list) {
            std::vector<int> len_h(R, (int)ctx);
            CK(hipMemcpy(d_len, len_h.data(), R * 4, hipMemcpyHostToDevice));
            // ---- sparse selection: 2048 distinct causal positions per row ----
            std::vector<int> idx_h(size_t(R) * TOPK);
            std::vector<std::vector<unsigned>> keys(R);
            for (unsigned b = 0; b < R; ++b) {
                unsigned step = 37; while (gcd_u(step, ctx) != 1) ++step;
                const unsigned n_sel = std::min(TOPK, ctx);
                keys[b].resize(n_sel);
                for (unsigned k = 0; k < TOPK; ++k) {
                    const unsigned pos = (unsigned)(((size_t)k * step + (size_t)b * 313) % ctx);
                    idx_h[size_t(b) * TOPK + k] = k < n_sel ? (int)pos : -1;
                    if (k < n_sel) keys[b][k] = pos;
                }
            }
            CK(hipMemcpy(d_idx, idx_h.data(), idx_h.size() * 4, hipMemcpyHostToDevice));
            std::vector<float> out_host, ml_host;
            if (ops.count("flash") || ops.count("merge")) {
                for (unsigned ns : ns_list) {
                    if (ns > std::max(1u, std::min(TOPK, ctx) / 32u)) continue;
                    if (!run_flash_and_check("flash_sparse", ctx, "interp_gather_fp8", ns, 4, true, keys, out_host, ml_host)) rc = 4;
                    if (ops.count("merge")) run_merge_fold(ctx, "after_sparse", ns, out_host, ml_host);
                }
            }
            if (ops.count("dense")) {
                std::vector<std::vector<unsigned>> all(R);
                for (unsigned b = 0; b < R; ++b) { all[b].resize(ctx); for (unsigned k = 0; k < ctx; ++k) all[b][k] = k; }
                const unsigned pol = glm_nsplit(ctx, NH);
                std::set<unsigned> dns{pol, std::max(pol / 2, 1u), std::min(pol * 2, 128u), 4u, 8u};
                for (unsigned ns : dns) {
                    if (ns > std::max(1u, ctx / 32u)) continue;
                    for (unsigned gf : {4u, 2u}) {
                        if (gf == 2 && ns != pol) continue;
                        if (!run_flash_and_check("flash_dense", ctx, ns == pol ? "policy" : "sweep", ns, gf, false, all, out_host, ml_host)) rc = 4;
                        if (ops.count("merge") && ns == pol && gf == 4) run_merge_fold(ctx, "after_dense", ns, out_host, ml_host);
                    }
                }
            }
            if (ops.count("aiter") && ctx >= TOPK) {
                // pack selected rows (dequant to bf16) -> AITER -> reduce; CSR indexes the packed buffer
                auto* d_kvp = dalloc<uint16_t>(size_t(R) * TOPK * 576);
                std::vector<uint16_t> qpk(size_t(R) * NH * 576);
                for (size_t bh = 0; bh < size_t(R) * NH; ++bh) {
                    std::memcpy(&qpk[bh * 576], &qa_h[bh * DK], DK * 2);
                    std::memcpy(&qpk[bh * 576 + DK], &qr_h[bh * DR], DR * 2);
                }
                auto* d_q = upload(qpk);
                std::vector<uint32_t> flat(size_t(R) * TOPK), kp(R + 1), qp(R + 1), last(R, 1), sp(R + 1);
                for (size_t i = 0; i < flat.size(); ++i) flat[i] = (uint32_t)i;
                for (unsigned r = 0; r <= R; ++r) { kp[r] = r * TOPK; qp[r] = r; }
                auto* d_flat = upload(flat); auto* d_kp = upload(kp); auto* d_qp = upload(qp); auto* d_last = upload(last);
                auto* d_sp = dalloc<uint32_t>(R + 1);
                auto* d_part = dalloc<float>(size_t(R) * 4 * NH * DK + 2 * GUARD);
                auto* d_lse = dalloc<float>(size_t(R) * 4 * NH + 2 * GUARD);
                auto* d_red = dalloc<float>(size_t(R) * NH * DK);
                auto* d_rml = dalloc<float>(size_t(R) * NH * 2);
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                auto timed = [&](auto&& fn) {
                    std::vector<double> t;
                    for (int r = 0; r < 10; ++r) {
                        CK(hipEventRecord(e0)); fn(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                        float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (r) t.push_back(ms * 1000.);
                    }
                    std::sort(t.begin(), t.end()); return t[t.size() / 2];
                };
                auto pack = [&] { pack_selected<<<GRID, 512>>>(d_kvp, d_lat, d_rope, d_sc, d_idx, R); CK(hipGetLastError()); };
                pack(); CK(hipDeviceSynchronize());
                const double pack_us = timed(pack);
                for (unsigned ns : {1u, 2u, 4u}) {
                    for (unsigned r = 0; r <= R; ++r) sp[r] = r * ns;
                    CK(hipMemcpy(d_sp, sp.data(), sp.size() * 4, hipMemcpyHostToDevice));
                    uint64_t args[40] = {};
                    args[0] = (uint64_t)(d_part + GUARD); args[2] = (uint64_t)(d_lse + GUARD); args[4] = (uint64_t)d_q;
                    args[6] = (uint64_t)d_kvp; args[8] = (uint64_t)d_kp; args[10] = (uint64_t)d_flat; args[12] = (uint64_t)d_last;
                    std::memcpy(&args[14], &scale, 4); args[16] = NH; args[18] = ns; args[20] = NH * 576 * 2; args[22] = 576 * 2;
                    args[26] = (uint64_t)d_qp; args[28] = (uint64_t)d_sp;
                    size_t asz = sizeof(args);
                    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
                    auto launch = [&] { CK(hipModuleLaunchKernel(afn, 1, R, ns, 256, 1, 1, 0, nullptr, nullptr, cfg)); };
                    const size_t np = size_t(R) * ns * NH * DK, nl = size_t(R) * ns * NH;
                    poison(d_part, np + 2 * GUARD); poison(d_lse, nl + 2 * GUARD);
                    launch(); CK(hipDeviceSynchronize());
                    std::vector<float> ph(np + 2 * GUARD), lh(nl + 2 * GUARD);
                    download(ph, d_part); download(lh, d_lse);
                    bool ok = guards_intact(ph, np) && guards_intact(lh, nl);
                    double mx = 0, e2 = 0, n2 = 0;
                    std::vector<double> ref;
                    for (unsigned b : sample_b) for (unsigned h : sample_h) {
                        const size_t bh = size_t(b) * NH + h;
                        ref_attention(qa_h.data() + bh * DK, qr_h.data() + bh * DR, lat_f.data() + size_t(b) * STRIDE * DK,
                                      rope_f.data() + size_t(b) * STRIDE * DR, sc_h.data() + size_t(b) * STRIDE, keys[b], scale, ref);
                        double m = -INFINITY; std::vector<double> w(ns); double den = 0;
                        for (unsigned s = 0; s < ns; ++s) m = std::max(m, double(lh[GUARD + (size_t(b) * ns + s) * NH + h]));
                        for (unsigned s = 0; s < ns; ++s) { w[s] = std::exp(double(lh[GUARD + (size_t(b) * ns + s) * NH + h]) - m); den += w[s]; }
                        for (unsigned c = 0; c < DK; ++c) {
                            double v = 0;
                            for (unsigned s = 0; s < ns; ++s) v += w[s] * ph[GUARD + ((size_t(b) * ns + s) * NH + h) * DK + c];
                            const double dlt = v / den - ref[c];
                            if (!std::isfinite(dlt)) ok = false;
                            mx = std::max(mx, std::abs(dlt)); e2 += dlt * dlt; n2 += ref[c] * ref[c];
                        }
                    }
                    const double k_us = timed(launch);
                    const double red_us = timed([&] { reduce_splits<<<GRID, 256>>>(d_red, d_rml, d_part + GUARD, d_lse + GUARD, R, ns); CK(hipGetLastError()); });
                    std::printf("flash_sparse,%u,%u,aiter_qh8_kernel,%u,8,%u,%.3f,%.6g,%.6g,%s\n", R, ctx, ns, R * ns, k_us, mx,
                                std::sqrt(e2 / std::max(n2, 1e-300)), ok && mx <= 0.02 ? "PASS" : "FAIL");
                    std::printf("flash_sparse,%u,%u,aiter_qh8_pack+kernel+reduce,%u,8,%u,%.3f,0,0,pack=%.3f;reduce=%.3f\n", R, ctx, ns, R * ns,
                                pack_us + k_us + red_us, pack_us, red_us);
                    std::fflush(stdout);
                    if (!(ok && mx <= 0.02)) rc = 6;
                }
                for (void* p : {(void*)d_kvp, (void*)d_q, (void*)d_flat, (void*)d_kp, (void*)d_qp, (void*)d_last, (void*)d_sp, (void*)d_part, (void*)d_lse, (void*)d_red, (void*)d_rml}) CK(hipFree(p));
            }
            if (ops.count("score") || ops.count("select")) {
                // ---- lightning indexer: score (op 58) + local per-row select (op 59, i4=1) ----
                constexpr unsigned HI = 32, DI = 128;
                auto qi_h = random_bf(size_t(R) * HI * DI), w_h = random_bf(size_t(R) * HI);
                std::vector<uint16_t> ki_h(size_t(R) * STRIDE * DI);
                for (size_t i = 0; i < ki_h.size(); ++i) ki_h[i] = bf16_of((int(rnext() % 2001) - 1000) / 1000.f);
                auto* d_qi = upload(qi_h); auto* d_w = upload(w_h); auto* d_ki = upload(ki_h);
                const size_t nsc = size_t(R) * STRIDE;
                auto* d_score = dalloc<float>(nsc + 2 * GUARD);
                auto* d_sel = dalloc<int>(size_t(R) * TOPK + 2 * GUARD);
                std::vector<void*> t2{d_score + GUARD, d_qi, d_ki, d_w, d_len, d_sel + GUARD};
                auto* d_t2 = upload(t2);
                rp.set_tensors(d_t2);
                const float iscale = std::pow(float(DI), -0.5f) * std::pow(float(HI), -0.5f);
                std::vector<float> sc_host(nsc + 2 * GUARD);
                if (ops.count("score")) {
                    PlowDevInst d{}; d.op = PLOW_DOP_INDEX_SCORE; d.blocks = GRID;
                    d.t[0] = 0; d.t[1] = 1; d.t[2] = 2; d.t[3] = 3; d.t[4] = 4;
                    d.i[0] = R; d.i[1] = HI; d.i[2] = STRIDE; d.i[3] = DI; d.fj[0].f = iscale;
                    rp.set_inst(d);
                    poison(d_score, nsc + 2 * GUARD);
                    rp.run(dec); CK(hipDeviceSynchronize());
                    download(sc_host, d_score);
                    bool ok = guards_intact(sc_host, nsc);
                    double mx = 0, ref_max = 0;
                    for (unsigned b : sample_b) for (unsigned pos : {0u, ctx / 2, ctx - 1}) {
                        double s = 0;
                        for (unsigned h = 0; h < HI; ++h) {
                            double dot = 0;
                            for (unsigned c = 0; c < DI; ++c) dot += double(from_bf16(qi_h[(size_t(b) * HI + h) * DI + c])) * from_bf16(ki_h[(size_t(b) * STRIDE + pos) * DI + c]);
                            s += from_bf16(w_h[size_t(b) * HI + h]) * std::max(dot, 0.0);
                        }
                        s *= iscale;
                        const double got = sc_host[GUARD + size_t(b) * STRIDE + pos];
                        if (!std::isfinite(got)) ok = false;
                        mx = std::max(mx, std::abs(got - s)); ref_max = std::max(ref_max, std::abs(s));
                    }
                    // positions past kv_len must stay untouched (poison) when ctx < STRIDE
                    if (ctx < STRIDE) for (unsigned b : sample_b) if (!std::isnan(sc_host[GUARD + size_t(b) * STRIDE + ctx])) ok = false;
                    const double us = rp.time_us(dec);
                    const bool pass = ok && mx <= 1e-2 * std::max(ref_max, 1e-3);
                    std::printf("index_score,%u,%u,mfma,1,0,%u,%.3f,%.6g,%.6g,%s\n", R, ctx, d.blocks, us, mx, mx / std::max(ref_max, 1e-300), pass ? "PASS" : "FAIL");
                    std::fflush(stdout);
                    if (!pass) rc = 7;
                }
                if (ops.count("select")) {
                    // synthetic distinct-ish scores over [0, ctx) per row, NaN poison past kv_len
                    std::vector<float> synth(nsc + 2 * GUARD, NAN);
                    for (unsigned b = 0; b < R; ++b) for (unsigned s = 0; s < ctx; ++s) synth[GUARD + size_t(b) * STRIDE + s] = (int(rnext() % 2000001) - 1000000) / 1000000.f;
                    CK(hipMemcpy(d_score, synth.data(), synth.size() * 4, hipMemcpyHostToDevice));
                    // rows == 1 ships the 32-workgroup cooperative radix (grid barrier over gHist/gCtl);
                    // rows >= 2 ship the per-row local radix (i4 = 1, one workgroup per row).
                    const bool coop = R == 1;
                    auto* d_hist = dalloc<unsigned>(7 * 256 + 64);
                    CK(hipMemset(d_hist, 0, (7 * 256 + 64) * 4));
                    std::vector<void*> t4{d_score + GUARD, d_qi, d_ki, d_w, d_len, d_sel + GUARD, d_hist, d_hist + 7 * 256};
                    auto* d_t4 = upload(t4);
                    rp.set_tensors(d_t4);
                    PlowDevInst d{}; d.op = PLOW_DOP_INDEX_SELECT; d.blocks = coop ? 32 : R;
                    d.t[0] = 5; d.t[1] = 0; d.t[2] = coop ? 6 : PLOW_TENSOR_NONE; d.t[3] = coop ? 7 : PLOW_TENSOR_NONE; d.t[4] = 4;
                    for (unsigned i = 5; i < 8; ++i) d.t[i] = PLOW_TENSOR_NONE;
                    d.i[0] = STRIDE; d.i[1] = TOPK; d.i[3] = 0; d.i[4] = coop ? 0 : 1;
                    rp.set_inst(d);
                    std::vector<int> pz(size_t(R) * TOPK + 2 * GUARD, -777777);
                    CK(hipMemcpy(d_sel, pz.data(), pz.size() * 4, hipMemcpyHostToDevice));
                    rp.run(dec); CK(hipDeviceSynchronize());
                    download(pz, d_sel);
                    bool ok = true;
                    for (unsigned i = 0; i < GUARD; ++i) if (pz[i] != -777777 || pz[GUARD + size_t(R) * TOPK + i] != -777777) ok = false;
                    unsigned mismatched_rows = 0;
                    for (unsigned b = 0; b < R; ++b) {
                        const unsigned k = std::min(TOPK, ctx);
                        std::vector<std::pair<double, int>> keys_(ctx);
                        for (unsigned s = 0; s < ctx; ++s) keys_[s] = {-double(synth[GUARD + size_t(b) * STRIDE + s]), (int)s};
                        std::partial_sort(keys_.begin(), keys_.begin() + k, keys_.end());
                        std::vector<int> want(k), got(pz.begin() + GUARD + size_t(b) * TOPK, pz.begin() + GUARD + size_t(b) * TOPK + k);
                        for (unsigned i = 0; i < k; ++i) want[i] = keys_[i].second;
                        std::sort(want.begin(), want.end()); std::sort(got.begin(), got.end());
                        if (want != got) ++mismatched_rows;
                    }
                    const double us = rp.time_us(dec);
                    const bool pass = ok && mismatched_rows == 0;
                    std::printf("index_select,%u,%u,%s,1,0,%u,%.3f,%u,0,%s\n", R, ctx, coop ? "coop_radix32" : "local_radix", d.blocks, us, mismatched_rows, pass ? "PASS" : "FAIL");
                    std::fflush(stdout);
                    if (!pass) rc = 8;
                }
                rp.set_tensors(d_table);
                for (void* p : {(void*)d_qi, (void*)d_w, (void*)d_ki, (void*)d_score, (void*)d_sel, (void*)d_t2}) CK(hipFree(p));
            }
        }
        if (ops.count("write")) {
            // ---- FP8 latent writer (op 37): RMSNorm(x)*gamma -> e4m3 row + scale at pos[b] of slot b ----
            auto x_h = random_bf(size_t(R) * DK), g_h = random_bf(DK);
            std::vector<int> pos_h(R);
            for (unsigned b = 0; b < R; ++b) pos_h[b] = (int)((b * 7919u + 70000u) % STRIDE);
            auto* d_x = upload(x_h); auto* d_g = upload(g_h); auto* d_pos = upload(pos_h);
            std::vector<void*> t3{d_lat, d_x, d_g, nullptr, nullptr, d_pos, d_sc};
            auto* d_t3 = upload(t3);
            rp.set_tensors(d_t3);
            for (unsigned blocks : {1u, R, std::min(R * 8u, GRID)}) {
                PlowDevInst d{}; d.op = PLOW_DOP_HEADNORM_ROPE_FP8; d.blocks = blocks;
                d.t[0] = 0; d.t[1] = 1; d.t[2] = 2; d.t[3] = PLOW_TENSOR_NONE; d.t[4] = PLOW_TENSOR_NONE; d.t[5] = 5; d.t[6] = 6; d.t[7] = PLOW_TENSOR_NONE;
                d.i[0] = R; d.i[1] = 1; d.i[2] = DK; d.i[3] = 0; d.i[4] = 0; d.i[6] = R;
                d.fj[0].f = 1e-6f; d.fj[1].u = STRIDE; d.fj[2].u = 0xffffffffu;
                rp.set_inst(d);
                CK(hipMemcpy(d_lat, lat8.data(), lat8.size(), hipMemcpyHostToDevice));
                CK(hipMemcpy(d_sc, sc_h.data(), sc_h.size() * 4, hipMemcpyHostToDevice));
                rp.run(dec); CK(hipDeviceSynchronize());
                std::vector<uint8_t> lat_out(lat8.size()); std::vector<float> sc_out(sc_h.size());
                download(lat_out, d_lat); download(sc_out, d_sc);
                bool ok = true; double mx = 0, rmax = 0;
                size_t changed = 0;
                for (size_t i = 0; i < lat8.size(); ++i) changed += lat_out[i] != lat8[i];
                for (unsigned b = 0; b < R; ++b) {
                    double ss = 0; for (unsigned c = 0; c < DK; ++c) { const double v = from_bf16(x_h[size_t(b) * DK + c]); ss += v * v; }
                    const double inv = 1.0 / std::sqrt(ss / DK + 1e-6);
                    const size_t row = size_t(b) * STRIDE + pos_h[b];
                    const float s = sc_out[row];
                    for (unsigned c = 0; c < DK; ++c) {
                        const double ref = from_bf16(x_h[size_t(b) * DK + c]) * inv * from_bf16(g_h[c]);
                        const double got = double(from_fp8(lat_out[row * DK + c])) * s;
                        if (!std::isfinite(got)) ok = false;
                        mx = std::max(mx, std::abs(got - ref)); rmax = std::max(rmax, std::abs(ref));
                    }
                }
                if (changed > size_t(R) * DK) ok = false; // wrote outside the R target rows
                const double us = rp.time_us(dec);
                const bool pass = ok && mx <= rmax / 16.0 + 1e-3;
                std::printf("kv_write_fp8,%u,%u,%s,1,0,%u,%.3f,%.6g,%.6g,%s\n", R, STRIDE, blocks == 1 ? "emitted_1wg" : "wider", blocks, us, mx, mx / std::max(rmax, 1e-300), pass ? "PASS" : "FAIL");
                std::fflush(stdout);
                if (!pass) rc = 9;
            }
            rp.set_tensors(d_table);
            for (void* p : {(void*)d_x, (void*)d_g, (void*)d_pos, (void*)d_t3}) CK(hipFree(p));
        }
        for (void* p : {(void*)d_lat, (void*)d_rope, (void*)d_sc, (void*)d_qa, (void*)d_qr, (void*)d_len, (void*)d_idx, (void*)d_out, (void*)d_ml, (void*)d_oat, (void*)d_wuv, (void*)d_table}) CK(hipFree(p));
    }
    CK(hipDeviceSynchronize());
    return rc;
}
