#pragma once

#include "op_gemm.h"
#include "op_moe.h"

#if PLOW_CDNA4
__device__ void d_moe_group_glu_fp8_block128_m16(
    bf16* fu, const unsigned char* xq, const float* xscale,
    const unsigned long long* wtab, const unsigned long long* stab,
    const int* meta, const unsigned* row_token, unsigned I, unsigned H,
    unsigned n_exp, unsigned T, unsigned slice, unsigned nblk) {
    const int* tilep = meta + 2u * n_exp;
    const unsigned nt = (I + 16u * PLOW_WAVES - 1) / (16u * PLOW_WAVES);
    for (unsigned lin = slice; lin < (unsigned)tilep[n_exp] * nt; lin += nblk) {
        const unsigned mt = lin / nt, nc = lin % nt;
        const unsigned expert = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned offset = (mt - (unsigned)tilep[expert]) * MPF_BM;
        const unsigned row = (unsigned)meta[expert] + offset;
        const unsigned remaining = (unsigned)meta[n_exp + expert] - offset;
        unsigned count = remaining < MPF_BM ? remaining : MPF_BM;
        const auto* gate = (const unsigned char*)(size_t)wtab[expert * 3u];
        const auto* up = (const unsigned char*)(size_t)wtab[expert * 3u + 1u];
        if (gate && up) {
            const auto* gs = (const float*)(size_t)stab[expert * 3u];
            const auto* us = (const float*)(size_t)stab[expert * 3u + 1u];
            d_gemm_fp8_block128_m16<true, true>(fu + (size_t)row * I, xq, gate, xscale,
                gs, count, I, H, nc, nt, up, us, row_token + row, T);
        } else {
            count = 0;
        }
        const unsigned col = nc * 16u * PLOW_WAVES;
        const unsigned width = I - col < 16u * PLOW_WAVES ? I - col : 16u * PLOW_WAVES;
        for (unsigned j = threadIdx.x; j < (MPF_BM - count) * width; j += PLOW_THREADS)
            st_act1(fu + (size_t)(row + count + j / width) * I + col + j % width, f2bf(0.0f));
    }
}

__device__ void d_moe_quant_fp8_block128(
    unsigned char* xq, const bf16* fu, float* scale, const int* meta,
    const unsigned* row_partidx, unsigned I, unsigned n_exp, unsigned slots,
    unsigned slice, unsigned nblk) {
    const unsigned rows = (unsigned)meta[3u * n_exp] * MPF_BM;
    d_quant_fp8_block128<true>(xq, fu, scale, rows, I, slice, nblk, row_partidx, slots);
}

template <bool ATOMIC = false>
__device__ void d_moe_group_down_fp8_block128_m16(
    bf16* parts, const unsigned char* xq, const float* xscale,
    const unsigned long long* wtab, const unsigned long long* stab,
    const int* meta, const unsigned* row_partidx, const float* row_gate,
    unsigned I, unsigned H, unsigned n_exp, unsigned slots, unsigned slice, unsigned nblk,
    unsigned topk = 0) {
    const int* tilep = meta + 2u * n_exp;
    const unsigned nt = (H + 16u * PLOW_WAVES - 1) / (16u * PLOW_WAVES);
    for (unsigned lin = slice; lin < (unsigned)tilep[n_exp] * nt; lin += nblk) {
        const unsigned mt = lin / nt, nc = lin % nt;
        const unsigned expert = mpf_expert_of_tile(tilep, mt, n_exp);
        const unsigned offset = (mt - (unsigned)tilep[expert]) * MPF_BM;
        const unsigned row = (unsigned)meta[expert] + offset;
        const unsigned remaining = (unsigned)meta[n_exp + expert] - offset;
        const unsigned count = remaining < MPF_BM ? remaining : MPF_BM;
        const auto* down = (const unsigned char*)(size_t)wtab[expert * 3u + 2u];
        if (down) {
            const auto* scale = (const float*)(size_t)stab[expert * 3u + 2u];
            d_gemm_fp8_block128_m16<false, true, true, true, ATOMIC>(parts, xq, down, xscale,
                scale, count, H, I, nc, nt, nullptr, nullptr, row_partidx + row, slots,
                row_gate + row, topk);
        } else if constexpr (!ATOMIC) {
            const unsigned col = nc * 16u * PLOW_WAVES;
            const unsigned width = H - col < 16u * PLOW_WAVES ? H - col : 16u * PLOW_WAVES;
            for (unsigned j = threadIdx.x; j < count * width; j += PLOW_THREADS) {
                const unsigned dst = row_partidx[row + j / width];
                if (dst < slots) st_act1(parts + (size_t)dst * H + col + j % width, f2bf(0.0f));
            }
        }
    }
}
#endif
