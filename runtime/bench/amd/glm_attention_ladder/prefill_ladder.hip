// Exact-object replay of GLM-5.3 TP8 prefill attention on gfx942 (FP8 latent KV, 8 heads/rank).
//   dense : FlashMlaPrefillFp8 on the 8-wave prefill object (buckets 128/512 today), the 4-wave
//           flash object (V2, bucket 2048 today) and the 4-wave split-KV object (ns sweep);
//   sparse: IndexUnionPf + the flash object's GATHER arm (interpreter route) and the AITER QH8
//           qseqlen1 object behind a dequant/pack kernel (the native route), fixed 2048 keys/query.
// Outputs are NaN-poisoned with guard bands and checked against an FP64 host reference on
// sampled (query, head) rows; timings are medians of 9 launches.
#include "ladder_common.h"
#include <set>
#include <unistd.h>

static constexpr unsigned STRIDE = 81920, DK = 512, DR = 64, NH = 8, TOPK = 2048;

static std::vector<unsigned> parse_list(const char* s) {
    std::vector<unsigned> v; std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || !*p) { if (!cur.empty()) v.push_back(std::stoul(cur)); cur.clear(); if (!*p) break; }
        else cur += *p;
    }
    return v;
}
static unsigned gcd_u(unsigned a, unsigned b) { while (b) { unsigned t = a % b; a = b; b = t; } return a; }

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

__global__ void dequant_cache(uint16_t* __restrict__ kv, const uint8_t* __restrict__ lat, const uint16_t* __restrict__ rope,
                              const float* __restrict__ sc, unsigned n) {
    const unsigned g = (blockIdx.x * blockDim.x + threadIdx.x) >> 6, lane = threadIdx.x & 63;
    for (unsigned r = g; r < n; r += (gridDim.x * blockDim.x) >> 6) {
        const uint8_t* src = lat + (size_t)r * DK;
        const float s = sc[r];
        uint16_t* dst = kv + (size_t)r * 576;
        for (unsigned c = lane; c < DK; c += 64) {
            const uint8_t x = src[c];
            const int e = (x >> 3) & 15, m = x & 7;
            float v = e ? ldexpf(float(8 + m), e - 10) : ldexpf(float(m), -9);
            if (x & 128) v = -v;
            v *= s;
            uint32_t bits; __builtin_memcpy(&bits, &v, 4);
            bits += 0x7fff + ((bits >> 16) & 1);
            dst[c] = (uint16_t)(bits >> 16);
        }
        if (lane < DR) dst[DK + lane] = rope[(size_t)r * DR + lane];
    }
}
__global__ void reduce_splits(float* __restrict__ out, const float* __restrict__ part, const float* __restrict__ lse,
                              unsigned rows, unsigned ns) {
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
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <objdir> [--rows a,b] [--ctx a,b] [--ns a,b] [--mode dense,sparse] [--split path] [--small path] [--aiter path]\n", argv[0]); return 2; }
    std::string objdir = argv[1];
    std::vector<unsigned> rows_list{128, 512, 2048, 8192}, ctx_list{1024, 4096, 16384, 32768, 65536, 81920}, ns_list{2, 4, 8, 16, 32};
    std::set<std::string> modes{"dense", "sparse"};
    std::string split_path = objdir + "/interp_mla_split_fp8kv_gq.elf", small_path = objdir + "/interp_mla_small_fp8kv_gq.elf";
    std::string aiter_path = objdir + "/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co";
    for (int a = 2; a + 1 < argc; a += 2) {
        std::string k = argv[a];
        if (k == "--rows") rows_list = parse_list(argv[a + 1]);
        else if (k == "--ctx") ctx_list = parse_list(argv[a + 1]);
        else if (k == "--ns") ns_list = parse_list(argv[a + 1]);
        else if (k == "--split") split_path = argv[a + 1];
        else if (k == "--small") small_path = argv[a + 1];
        else if (k == "--aiter") aiter_path = argv[a + 1];
        else if (k == "--mode") { modes.clear(); std::string cur; for (const char* p = argv[a + 1];; ++p) { if (*p == ',' || !*p) { if (!cur.empty()) modes.insert(cur); cur.clear(); if (!*p) break; } else cur += *p; } }
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    CK(hipInit(0));
    Kernel broad = load_kernel("broad8", objdir + "/interp_prefill_fp8kv_mla_moe_gq.elf", "plow_interp_gfx942_gq", 512);
    Kernel flash = load_kernel("flash4", objdir + "/interp_flash_fp8kv_gq.elf", "plow_interp_flash_gfx942_gq", 256);
    Kernel split{}, small{};
    const bool have_split = access(split_path.c_str(), R_OK) == 0, have_small = access(small_path.c_str(), R_OK) == 0;
    if (have_split) split = load_kernel("split4", split_path, "plow_interp_mla_split_gfx942_gq", 256);
    if (have_small) small = load_kernel("small8", small_path, "plow_interp_mla_small_gfx942_gq", 512);
    hipModule_t amod{}; hipFunction_t afn{};
    const bool have_aiter = modes.count("sparse") && access(aiter_path.c_str(), R_OK) == 0;
    if (have_aiter) {
        CK(hipModuleLoad(&amod, aiter_path.c_str()));
        CK(hipModuleGetFunction(&afn, amod, "_ZN5aiter36mla_a16w16_qh8_qseqlen1_gqaratio8_v3E"));
    }
    Replay rp;
    const float scale = 1.f / std::sqrt(576.f);
    // ---- one KV slot (STRIDE rows), shared by every shape ----
    auto lat8 = random_fp8(size_t(STRIDE) * DK);
    auto rope_h = random_bf(size_t(STRIDE) * DR);
    auto sc_h = random_scales(size_t(2) * STRIDE);
    std::vector<float> lat_f(lat8.size()), rope_f(rope_h.size());
    std::transform(lat8.begin(), lat8.end(), lat_f.begin(), from_fp8);
    std::transform(rope_h.begin(), rope_h.end(), rope_f.begin(), from_bf16);
    auto* d_lat = upload(lat8); auto* d_rope = upload(rope_h); auto* d_sc = upload(sc_h);
    auto* d_len = dalloc<int>(1);
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timed = [&](auto&& fn) {
        std::vector<double> t;
        for (int r = 0; r < 10; ++r) {
            CK(hipEventRecord(e0)); fn(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (r) t.push_back(ms * 1000.);
        }
        std::sort(t.begin(), t.end()); return t[t.size() / 2];
    };
    std::puts("mode,rows,ctx,arm,ns,blocks,median_us,max_abs,rel_l2,note");
    int rc = 0;
    for (unsigned R : rows_list) {
        auto qa_h = random_bf(size_t(R) * NH * DK), qr_h = random_bf(size_t(R) * NH * DR);
        auto* d_qa = upload(qa_h); auto* d_qr = upload(qr_h);
        const unsigned ns_cap = R >= 8192 ? 4u : (R >= 2048 ? 8u : 32u);
        const size_t no_max = size_t(R) * NH * ns_cap * DK, nm_max = size_t(R) * NH * ns_cap * 2;
        auto* d_out = dalloc<float>(no_max + 2 * GUARD);
        auto* d_ml = dalloc<float>(nm_max + 2 * GUARD);
        auto* d_idx = dalloc<int>(size_t(R) * TOPK);
        const unsigned n_qt = (R + 7) / 8;
        std::vector<unsigned> sample_q{0u, R / 2, R - 1};
        sample_q.erase(std::unique(sample_q.begin(), sample_q.end()), sample_q.end());
        const unsigned sample_h[3] = {0, 3, 7};
        for (unsigned ctx : ctx_list) {
            if (ctx < R) continue;
            const int len = (int)ctx;
            CK(hipMemcpy(d_len, &len, 4, hipMemcpyHostToDevice));
            const unsigned q_pos0 = ctx - R;
            const unsigned cap = std::min(8u * TOPK, ctx);
            const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
            const size_t uni_bytes = hdr + size_t(n_qt) * cap * 12u;
            auto* d_uni = modes.count("sparse") ? dalloc<uint8_t>(uni_bytes) : nullptr;
            auto* d_umask = modes.count("sparse") ? dalloc<unsigned long long>(size_t(GRID) * STRIDE) : nullptr;
            std::vector<void*> table{d_out + GUARD, d_ml + GUARD, d_qa, d_qr, d_lat, d_rope, d_len, d_sc, d_idx, d_uni, d_umask};
            // 0 out 1 ml 2 qa 3 qr 4 ckv 5 krope 6 kvlen 7 scale 8 idx 9 uni 10 umask
            auto* d_table = upload(table);
            rp.set_tensors(d_table);

            auto check_dense_output = [&](unsigned ns, const std::vector<std::vector<unsigned>>* keys_of_q, double& mx, double& rel) -> bool {
                const size_t no = size_t(R) * NH * ns * DK, nm = size_t(R) * NH * ns * 2;
                std::vector<float> out_host(no + 2 * GUARD), ml_host(nm + 2 * GUARD);
                download(out_host, d_out); download(ml_host, d_ml);
                bool ok = guards_intact(out_host, no) && guards_intact(ml_host, nm);
                double e2 = 0, n2 = 0; mx = 0;
                std::vector<double> merged, ref;
                std::vector<unsigned> keys;
                for (unsigned q : sample_q) for (unsigned h : sample_h) {
                    const size_t qh = size_t(q) * NH + h;
                    merge_partials(out_host.data() + GUARD + qh * ns * DK, ml_host.data() + GUARD + qh * ns * 2, ns, DK, merged);
                    if (keys_of_q) keys = (*keys_of_q)[q];
                    else { keys.resize(q_pos0 + q + 1); for (unsigned k = 0; k < keys.size(); ++k) keys[k] = k; }
                    ref_attention(qa_h.data() + qh * DK, qr_h.data() + qh * DR, lat_f.data(), rope_f.data(), sc_h.data(), keys, scale, ref);
                    for (unsigned c = 0; c < DK; ++c) {
                        const double d = merged[c] - ref[c];
                        if (!std::isfinite(d)) ok = false;
                        mx = std::max(mx, std::abs(d)); e2 += d * d; n2 += ref[c] * ref[c];
                    }
                }
                rel = std::sqrt(e2 / std::max(n2, 1e-300));
                return ok && mx <= 0.02;
            };
            auto flash_inst = [&](unsigned ns_field, unsigned uni_plus1, unsigned cap_field) {
                PlowDevInst d{}; d.op = PLOW_DOP_FLASH_MLA_PREFILL_FP8; d.blocks = GRID;
                for (unsigned i = 0; i < 8; ++i) d.t[i] = i;
                d.i[0] = 1; d.i[1] = NH; d.i[2] = STRIDE; d.i[3] = 0; d.i[4] = R; d.i[5] = 0xffffffffu; d.i[6] = cap_field; d.i[7] = 4;
                d.fj[0].f = scale; d.fj[1].u = uni_plus1; d.fj[2].u = ns_field;
                return d;
            };
            if (modes.count("dense")) {
                struct Arm { const char* name; Kernel* k; unsigned ns; };
                std::vector<Arm> arms;
                if (R <= 2048) arms.push_back({"broad8_prefill_obj", &broad, 1});
                if (have_small && R <= 2048) arms.push_back({"small8_obj", &small, 1});
                arms.push_back({"flash4_v2", &flash, 1});
                if (have_split) for (unsigned ns : ns_list) if (ns <= ns_cap && ns <= 32) arms.push_back({"split4_v2", &split, ns});
                for (auto& arm : arms) {
                    // a split past the causal tile count is dead work; skip shapes the emitter would never pick
                    if (arm.ns > 1 && arm.ns > std::max(1u, ctx / 32u)) continue;
                    PlowDevInst d = flash_inst(arm.ns > 1 ? arm.ns : 0, 0, 0);
                    rp.set_inst(d);
                    const size_t no = size_t(R) * NH * arm.ns * DK, nm = size_t(R) * NH * arm.ns * 2;
                    poison(d_out, no + 2 * GUARD); poison(d_ml, nm + 2 * GUARD);
                    rp.run(*arm.k); CK(hipDeviceSynchronize());
                    double mx, rel;
                    const bool ok = check_dense_output(arm.ns, nullptr, mx, rel);
                    const double us = rp.time_us(*arm.k);
                    std::printf("dense,%u,%u,%s,%u,%u,%.3f,%.6g,%.6g,%s\n", R, ctx, arm.name, arm.ns, d.blocks, us, mx, rel, ok ? "PASS" : "FAIL");
                    std::fflush(stdout);
                    if (!ok) rc = 4;
                }
            }
            if (modes.count("sparse") && ctx > TOPK) {
                // ---- pack-structured top-2048 selection: 1800 shared per 8-query pack + 248 own ----
                std::vector<int> idx_h(size_t(R) * TOPK, -1);
                std::vector<std::vector<unsigned>> keys(R);
                std::vector<uint8_t> mark(ctx);
                for (unsigned pk = 0; pk < n_qt; ++pk) {
                    const unsigned q0 = pk * 8, q1 = std::min(q0 + 8, R);
                    const unsigned vis0 = q_pos0 + q0 + 1; // first query's causal count
                    std::fill(mark.begin(), mark.end(), 0);
                    std::vector<unsigned> core;
                    if (vis0 > TOPK) {
                        unsigned step = 37 + pk % 11; while (gcd_u(step, vis0) != 1) ++step;
                        for (unsigned k = 0; k < 1800; ++k) { const unsigned p = (unsigned)(((size_t)k * step + pk * 977u) % vis0); core.push_back(p); mark[p] = 1; }
                    }
                    for (unsigned q = q0; q < q1; ++q) {
                        const unsigned vis = q_pos0 + q + 1;
                        auto& ks = keys[q];
                        if (vis <= TOPK) { ks.resize(vis); for (unsigned k = 0; k < vis; ++k) ks[k] = k; }
                        else {
                            ks = core;
                            unsigned seed = q * 7919u + 13u;
                            while (ks.size() < TOPK) {
                                seed = seed * 1664525u + 1013904223u;
                                const unsigned p = seed % vis;
                                if (mark[p] & 2u) continue;   // this query's own pick
                                if (mark[p] & 1u) continue;   // in the shared core
                                mark[p] |= 2u; ks.push_back(p);
                            }
                            for (unsigned p : ks) if (!(mark[p] & 1u)) mark[p] &= ~2u; // release own marks for the next query
                        }
                        for (unsigned k = 0; k < ks.size(); ++k) idx_h[size_t(q) * TOPK + k] = (int)ks[k];
                    }
                }
                CK(hipMemcpy(d_idx, idx_h.data(), idx_h.size() * 4, hipMemcpyHostToDevice));
                // (a) union table, op 119 on the 8-wave prefill object
                {
                    PlowDevInst d{}; d.op = PLOW_DOP_INDEX_UNION_PF; d.blocks = GRID;
                    d.t[0] = 9; d.t[1] = 10; d.t[2] = 8; d.t[3] = 6; for (unsigned i = 4; i < 8; ++i) d.t[i] = PLOW_TENSOR_NONE;
                    d.i[0] = R; d.i[1] = TOPK; d.i[2] = STRIDE; d.i[3] = cap; d.i[4] = 8;
                    rp.set_inst(d);
                    CK(hipMemset(d_uni, 0xee, uni_bytes));
                    rp.run(broad); CK(hipDeviceSynchronize());
                    std::vector<unsigned> cnt(n_qt);
                    CK(hipMemcpy(cnt.data(), d_uni, n_qt * 4, hipMemcpyDeviceToHost));
                    double mean_union = 0; bool ok = true;
                    for (unsigned pk = 0; pk < n_qt; ++pk) {
                        std::set<unsigned> u;
                        for (unsigned q = pk * 8; q < std::min(pk * 8 + 8, R); ++q) u.insert(keys[q].begin(), keys[q].end());
                        if (cnt[pk] != u.size()) ok = false;
                        mean_union += cnt[pk];
                    }
                    const double us = rp.time_us(broad);
                    std::printf("sparse,%u,%u,index_union_pf,1,%u,%.3f,0,0,%s;mean_union=%.1f\n", R, ctx, d.blocks, us, ok ? "PASS" : "FAIL", mean_union / n_qt);
                    std::fflush(stdout);
                    if (!ok) rc = 5;
                }
                // (b) the flash object's GATHER arm (interpreter route)
                {
                    PlowDevInst d = flash_inst(0, 9 + 1, cap);
                    rp.set_inst(d);
                    const size_t no = size_t(R) * NH * DK, nm = size_t(R) * NH * 2;
                    poison(d_out, no + 2 * GUARD); poison(d_ml, nm + 2 * GUARD);
                    rp.run(flash); CK(hipDeviceSynchronize());
                    double mx, rel;
                    const bool ok = check_dense_output(1, &keys, mx, rel);
                    const double us = rp.time_us(flash);
                    std::printf("sparse,%u,%u,flash4_gather_union,1,%u,%.3f,%.6g,%.6g,%s\n", R, ctx, d.blocks, us, mx, rel, ok ? "PASS" : "FAIL");
                    std::fflush(stdout);
                    if (!ok) rc = 6;
                }
                // (c) AITER QH8 behind a full-cache dequant (the native route's packing), CSR = the selections
                if (have_aiter && R + TOPK <= ctx + 1) {
                    auto* d_kv = dalloc<uint16_t>(size_t(ctx) * 576);
                    auto pack = [&] { dequant_cache<<<GRID, 512>>>(d_kv, d_lat, d_rope, d_sc, ctx); CK(hipGetLastError()); };
                    pack(); CK(hipDeviceSynchronize());
                    const double pack_us = timed(pack);
                    std::vector<uint16_t> qpk(size_t(R) * NH * 576);
                    for (size_t qh = 0; qh < size_t(R) * NH; ++qh) {
                        std::memcpy(&qpk[qh * 576], &qa_h[qh * DK], DK * 2);
                        std::memcpy(&qpk[qh * 576 + DK], &qr_h[qh * DR], DR * 2);
                    }
                    auto* d_q = upload(qpk);
                    std::vector<uint32_t> flat(size_t(R) * TOPK), kp(R + 1), qp(R + 1), last(R, 1), sp(R + 1);
                    bool all_full = true;
                    for (unsigned q = 0; q < R; ++q) { if (keys[q].size() != TOPK) all_full = false; for (unsigned k = 0; k < TOPK; ++k) flat[size_t(q) * TOPK + k] = k < keys[q].size() ? keys[q][k] : keys[q][0]; }
                    for (unsigned r = 0; r <= R; ++r) { kp[r] = r * TOPK; qp[r] = r; }
                    auto* d_flat = upload(flat); auto* d_kp = upload(kp); auto* d_qp = upload(qp); auto* d_last = upload(last);
                    auto* d_sp = dalloc<uint32_t>(R + 1);
                    auto* d_part = dalloc<float>(size_t(R) * 2 * NH * DK + 2 * GUARD);
                    auto* d_lse = dalloc<float>(size_t(R) * 2 * NH + 2 * GUARD);
                    auto* d_red = dalloc<float>(size_t(R) * NH * DK);
                    for (unsigned ns : {2u, 1u}) {
                        for (unsigned r = 0; r <= R; ++r) sp[r] = r * ns;
                        CK(hipMemcpy(d_sp, sp.data(), sp.size() * 4, hipMemcpyHostToDevice));
                        uint64_t args[40] = {};
                        args[0] = (uint64_t)(d_part + GUARD); args[2] = (uint64_t)(d_lse + GUARD); args[4] = (uint64_t)d_q;
                        args[6] = (uint64_t)d_kv; args[8] = (uint64_t)d_kp; args[10] = (uint64_t)d_flat; args[12] = (uint64_t)d_last;
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
                        bool ok = guards_intact(ph, np) && guards_intact(lh, nl) && all_full;
                        double mx = 0, e2 = 0, n2 = 0;
                        std::vector<double> ref;
                        for (unsigned q : sample_q) for (unsigned h : sample_h) {
                            const size_t qh = size_t(q) * NH + h;
                            ref_attention(qa_h.data() + qh * DK, qr_h.data() + qh * DR, lat_f.data(), rope_f.data(), sc_h.data(), keys[q], scale, ref);
                            double m = -INFINITY; std::vector<double> w(ns); double den = 0;
                            for (unsigned s = 0; s < ns; ++s) m = std::max(m, double(lh[GUARD + (size_t(q) * ns + s) * NH + h]));
                            for (unsigned s = 0; s < ns; ++s) { w[s] = std::exp(double(lh[GUARD + (size_t(q) * ns + s) * NH + h]) - m); den += w[s]; }
                            for (unsigned c = 0; c < DK; ++c) {
                                double v = 0;
                                for (unsigned s = 0; s < ns; ++s) v += w[s] * ph[GUARD + ((size_t(q) * ns + s) * NH + h) * DK + c];
                                const double dlt = v / den - ref[c];
                                if (!std::isfinite(dlt)) ok = false;
                                mx = std::max(mx, std::abs(dlt)); e2 += dlt * dlt; n2 += ref[c] * ref[c];
                            }
                        }
                        const double k_us = timed(launch);
                        const double red_us = ns > 1 ? timed([&] { reduce_splits<<<GRID, 256>>>(d_red, d_part + GUARD, d_lse + GUARD, R, ns); CK(hipGetLastError()); }) : 0.0;
                        std::printf("sparse,%u,%u,aiter_qh8_kernel,%u,%u,%.3f,%.6g,%.6g,%s\n", R, ctx, ns, R * ns, k_us, mx, std::sqrt(e2 / std::max(n2, 1e-300)), ok && mx <= 0.02 ? "PASS" : "FAIL");
                        std::printf("sparse,%u,%u,aiter_qh8_pack+kernel+reduce,%u,%u,%.3f,0,0,pack=%.3f;reduce=%.3f\n", R, ctx, ns, R * ns, pack_us + k_us + red_us, pack_us, red_us);
                        std::fflush(stdout);
                        if (!(ok && mx <= 0.02)) rc = 7;
                    }
                    for (void* p : {(void*)d_kv, (void*)d_q, (void*)d_flat, (void*)d_kp, (void*)d_qp, (void*)d_last, (void*)d_sp, (void*)d_part, (void*)d_lse, (void*)d_red}) CK(hipFree(p));
                }
            }
            if (d_uni) CK(hipFree(d_uni));
            if (d_umask) CK(hipFree(d_umask));
            CK(hipFree(d_table));
        }
        for (void* p : {(void*)d_qa, (void*)d_qr, (void*)d_out, (void*)d_ml, (void*)d_idx}) CK(hipFree(p));
    }
    CK(hipDeviceSynchronize());
    return rc;
}
