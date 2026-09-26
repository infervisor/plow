"""Register DevOp::EmbedPosBf16 = 184 across the ISA and implement it (CPU golden + CUDA), and
give the CUDA interpreter the existing EmbedOverlayBf16 (179) arm."""
W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


DOC = """    /// Token embedding plus a learned POSITION embedding indexed from a per-row base:
    /// `out[r] = bf16(table[tokens[r]] + pos_table[pos[r] - base[r]])`. Chatterbox T3 decode
    /// (speech_emb + speech_pos_emb, the speech index counted from the row's speech start).
    /// `t0=out(bf16[rows,width]) t1=table(bf16[vocab,width]) t2=tokens(u32[rows])
    /// t3=pos_table(bf16[pos_rows,width]) t4=pos(u32[rows]) t5=base(u32[rows])` ·
    /// `i0=rows i1=width i2=vocab i3=pos_rows`.
    EmbedPosBf16 = 184,
}"""
patch('crates/packet/src/dev.rs', [
    ("    DcpKvScatter = 183,\n}", "    DcpKvScatter = 183,\n" + DOC),
    ("        DevOp::DcpKvScatter,\n    ];", "        DevOp::DcpKvScatter,\n        DevOp::EmbedPosBf16,\n    ];"),
    ('            DevOp::DcpKvScatter => "PLOW_DOP_DCP_KV_SCATTER",\n',
     '            DevOp::DcpKvScatter => "PLOW_DOP_DCP_KV_SCATTER",\n            DevOp::EmbedPosBf16 => "PLOW_DOP_EMBED_POS_BF16",\n'),
])
patch('runtime/common/dev_isa.h', [
    ("    PLOW_DOP_DCP_KV_SCATTER = 183,\n", "    PLOW_DOP_DCP_KV_SCATTER = 183,\n    PLOW_DOP_EMBED_POS_BF16 = 184,\n"),
])
patch('crates/packet/src/slots.rs', [
    ('    S { op: DevOp::EmbedOverlayBf16,',
     '    S { op: DevOp::EmbedPosBf16, t: &["out", "table", "tokens", "pos_table", "pos", "base"], i: &["rows", "width", "vocab", "pos_rows"], f: &[], j: &[] },\n    S { op: DevOp::EmbedOverlayBf16,'),
])
patch('crates/packet/src/opclass.rs', [
    ("        | EmbedOverlayBf16 | PackNcfwRowsF32 => &[\"elementwise\"],",
     "        | EmbedOverlayBf16 | EmbedPosBf16 | PackNcfwRowsF32 => &[\"elementwise\"],"),
])
patch('crates/packet/src/rowclass.rs', [
    ("| EmbedF16F32 | EmbedOverlayBf16\n", "| EmbedF16F32 | EmbedOverlayBf16 | EmbedPosBf16\n"),
])
patch('crates/plowrt/src/opaudit.rs', [
    ('        DevOp::EmbedOverlayBf16 => a_rows("i0=rows, token gather with explicit row overlay"),\n',
     '        DevOp::EmbedOverlayBf16 => a_rows("i0=rows, token gather with explicit row overlay"),\n'
     '        DevOp::EmbedPosBf16 => a_rows("i0=rows, token gather plus per-row learned position"),\n'),
])
patch('runtime/cpu/dev/golden/golden.h', [
    ("G_K(g_embed_overlay_bf16);\n", "G_K(g_embed_overlay_bf16);\nG_K(g_embed_pos_bf16);\n"),
])
patch('runtime/cpu/dev/golden/control.c', [
    ("    tab[PLOW_DOP_EMBED_OVERLAY_BF16] = g_embed_overlay_bf16;\n",
     "    tab[PLOW_DOP_EMBED_OVERLAY_BF16] = g_embed_overlay_bf16;\n    tab[PLOW_DOP_EMBED_POS_BF16] = g_embed_pos_bf16;\n"),
])
patch('runtime/cpu/dev/golden/f32_primitives.c', [
    ("G_K(g_lstm_cell_f32) {", """G_K(g_embed_pos_bf16) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* table = PLOW_CPU_TEN(in, T, 1);
    const uint32_t* tokens = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* pos_table = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* pos = PLOW_CPU_TEN(in, T, 4);
    const uint32_t* base = PLOW_CPU_TEN(in, T, 5);
    const uint32_t rows = in->i[0], width = in->i[1], vocab = in->i[2], pos_rows = in->i[3];
    const uint64_t count64 = (uint64_t)rows * width;
    if (!rows || !width || !vocab || !pos_rows || count64 > UINT32_MAX) return;
    uint32_t lo, hi;
    g_range((uint32_t)count64, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t row = index / width, column = index % width;
        const uint32_t p = pos[row] - base[row];
        if (tokens[row] >= vocab || pos[row] < base[row] || p >= pos_rows) return;
        out[index] = plow_f2bf(plow_bf2f(table[(size_t)tokens[row] * width + column]) +
                               plow_bf2f(pos_table[(size_t)p * width + column]));
    }
}

G_K(g_lstm_cell_f32) {"""),
])
patch('runtime/nvidia/op_elementwise.cuh', [
    ("/* Compact hidden-row gather — the unified token batch's terminal row selection.",
     """/* EmbedOverlayBf16 (op 179): a row is table[tokens[r]] unless overlay_index[r] names an
 * overlay row, which is BF16-rounded in (the CPU golden's plow_f2bf: round-to-nearest-even). */
static __device__ void d_embed_overlay(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                                       const unsigned* __restrict__ tokens, const float* __restrict__ overlay,
                                       const unsigned* __restrict__ overlay_index, unsigned rows, unsigned width,
                                       unsigned vocab, unsigned overlay_rows, unsigned slice, unsigned nblk) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned sel = overlay_index[r];
        if (sel == 0xFFFFFFFFu) {
            if (tokens[r] >= vocab) { __trap(); return; }
            const __nv_bfloat16* src = table + (size_t)tokens[r] * width;
            for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS) out[(size_t)r * width + i] = src[i];
        } else {
            if (sel >= overlay_rows) { __trap(); return; }
            const float* src = overlay + (size_t)sel * width;
            for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS)
                out[(size_t)r * width + i] = __float2bfloat16_rn(src[i]);
        }
    }
}

/* EmbedPosBf16 (op 184): out[r] = bf16(table[tokens[r]] + pos_table[pos[r] - base[r]]). */
static __device__ void d_embed_pos(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ table,
                                   const unsigned* __restrict__ tokens, const __nv_bfloat16* __restrict__ pos_table,
                                   const unsigned* __restrict__ pos, const unsigned* __restrict__ base,
                                   unsigned rows, unsigned width, unsigned vocab, unsigned pos_rows,
                                   unsigned slice, unsigned nblk) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const unsigned p = pos[r] - base[r];
        if (tokens[r] >= vocab || pos[r] < base[r] || p >= pos_rows) { __trap(); return; }
        const __nv_bfloat16* a = table + (size_t)tokens[r] * width;
        const __nv_bfloat16* b = pos_table + (size_t)p * width;
        for (unsigned i = threadIdx.x; i < width; i += PLOW_NV_THREADS)
            out[(size_t)r * width + i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

/* Compact hidden-row gather — the unified token batch's terminal row selection."""),
])
patch('runtime/nvidia/interp_sm120.cu', [
    ("""    /* Terminal row selection for the unified token batch. One arm serves BOTH NVIDIA images:""",
     """    case PLOW_DOP_EMBED_OVERLAY_BF16:
        d_embed_overlay((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const unsigned*)TEN(2),
                        (const float*)TEN(3), (const unsigned*)TEN(4), in->i[0], in->i[1], in->i[2],
                        in->i[3], slice, nblk);
        break;
    case PLOW_DOP_EMBED_POS_BF16:
        d_embed_pos((__nv_bfloat16*)TEN(0), (const __nv_bfloat16*)TEN(1), (const unsigned*)TEN(2),
                    (const __nv_bfloat16*)TEN(3), (const unsigned*)TEN(4), (const unsigned*)TEN(5),
                    in->i[0], in->i[1], in->i[2], in->i[3], slice, nblk);
        break;

    /* Terminal row selection for the unified token batch. One arm serves BOTH NVIDIA images:"""),
])
print("patched")
