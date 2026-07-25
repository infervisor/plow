// mla_ref.rs — DeepSeek MLA (Multi-head Latent Attention) CPU reference oracle. [DEEPSEEK-MLA]
//
// This is the authoritative golden for the on-device MLA decode kernels
// (runtime/amd/op_attention.h: d_flash_mla_decode + d_o_uv_fold). It implements the
// ABSORBED formulation — the exact math the kernel runs — in f32, reading the same
// bf16-rounded inputs the device consumes, so kernel and reference share float
// associativity down to bf16 tolerance (sparse-attn-design.md §2.6).
//
// MLA recap. The KV cache stores a low-rank latent per token, head-SHARED:
//     C_kv[t]   in R^DK   (kv_lora_rank, DeepSeek 512)
//     K_rope[t] in R^DR   (qk_rope_dim,  DeepSeek 64, shared across heads)
// The up-projections W_uk/W_uv are ABSORBED into the query and output, so the O(ctx)
// loop only ever touches the compact latent:
//     score[h][t] = q_abs[h] . C_kv[t]  +  q_rope[h] . K_rope[t]      (q_abs = W_uk^T q_nope)
//     p           = softmax_t(scale * score)
//     oacc[h]     = sum_t p[t] * C_kv[t]        in R^DK  (latent accumulator)
//     o[h]        = W_uv[h]^T . oacc[h]         in R^V   (the per-query fold)
//
// It emits a multi-case fixture (inputs as bf16 + golden o as f32) that
// mla_test.c uploads and checks the device output against.
//
// Build:  rustc -O runtime/tests/mla_ref.rs -o mla_ref
// Run:    ./mla_ref fixture.bin

/// Query amplitude for the golden fixture.
///
/// **This was 0.10 and the fixture could not fail.** With DK=512 and q seeded at the
/// same 0.10 as `C_kv`/`K_rope`, the score std is ~0.005, so the softmax is essentially
/// uniform and `oacc` is the mean of `C_kv` almost independently of the query. Measured:
/// at 0.10 all three negative controls scored INSIDE the 5e-3 device tolerance
/// (q_abs*1.01 -> 8.802e-4, q_rope=0 -> 3.231e-3, no-rescale -> 3.715e-3). At 20.0 they
/// score 1.301e-2 / 2.645e-1 / 3.901e-1. The score path and the online-softmax rescale
/// branch are only exercised at an amplitude that makes the softmax peaked.
const Q_AMP: f32 = 20.0;

use std::io::Write;

// bf16 as raw u16. Round-to-nearest-even, identical to f2bf() in amd_common.h so the
// reference and the device round the SAME bits.
fn f2bf(f: f32) -> u16 {
    let u = f.to_bits();
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (r >> 16) as u16
}
fn bf2f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
// Round a f32 through bf16 (what storing then reloading a bf16 does).
fn q(f: f32) -> f32 {
    bf2f(f2bf(f))
}

// splitmix64 — deterministic, reproducible seeding independent of platform RNG.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
// Uniform in [-amp, amp), seeded by an index tuple.
fn rnd(seed: u64, amp: f32) -> f32 {
    let u = mix(seed);
    let unit = ((u >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
    (unit * 2.0 - 1.0) * amp
}

struct Case {
    n_head: u32,
    ctx: u32,
    nsplit: u32,
    top_k: u32, // 0 = dense (attend to all ctx); >0 = gathered (attend to a selected set) [DSA §3.5]
}

const DK: usize = 512; // kv_lora_rank (latent width)
const DR: usize = 64; // qk_rope_dim (shared rope key)
const V: usize = 128; // v_head_dim

// ============================================================================================
// DSA LIGHTNING-INDEXER reference (GLM-5.2 GlmMoeDsa; arXiv 2512.02556 eq.1).            [GLM52-DSA G6]
//
// CPU golden for the on-device indexer path (op_attention.h d_index_score / d_index_score_fast +
// d_index_select_coop). Mirrors the HF `GlmMoeDsaIndexer.forward` score form VERBATIM (confirmed
// against transformers modeling_glm_moe_dsa.py):
//     score[t] = softmax_scale * Σ_h (w[h]/√HI) · ReLU( q_idx[h] · k_idx[t] )     softmax_scale = 1/√DI
// q_idx (`[HI][DI]`) / k_idx (`[ctx][DI]`) arrive already projected (wq_b/wk fp8 GEMV), k_norm'd
// (LayerNorm+bias) and interleaved-RoPE'd on the first qk_rope_head_dim dims (the emit's job); this
// reference takes them post-projection and only scores + selects. Selection is scale-invariant, so
// the folded scale is immaterial to the SELECTED SET (the thing the gather reads).
//
// The REAL-WEIGHT G6 gate lives in scripts/glm52_indexer_oracle.py: it runs the actual HF
// GlmMoeDsaIndexer on real layer-0 weights and confirms this select set == HF's topk EXACTLY
// (relmax 2e-7, 2048/2048). This Rust reference is the synthetic-input CPU golden for the kernels.

/// score[t] = Σ_h w[h]·ReLU(q_idx[h]·k_idx[t]) · scale  (mirrors d_index_score; f32, i-inner reduction).
fn index_score(qidx: &[f32], kidx: &[f32], w: &[f32], hi: usize, di: usize, ctx: usize,
               scale: f32) -> Vec<f32> {
    let mut score = vec![0f32; ctx];
    for t in 0..ctx {
        let mut s = 0f32;
        for h in 0..hi {
            let mut d = 0f32;
            for i in 0..di {
                d += qidx[h * di + i] * kidx[t * di + i];
            }
            s += w[h] * if d > 0.0 { d } else { 0.0 }; // ReLU
        }
        score[t] = s * scale;
    }
    score
}

/// Monotone 56-bit BYTE-ALIGNED packed key — byte-for-byte the sm_120 selector's dsa_pack_key_a
/// (op_dsa.cuh:104): score in the top 4 bytes, index tie-break in the low 3 (24 bits; len<2^24
/// always). top_k of these keys == top_k highest scores with LOWEST-INDEX tie-break. Aligned to the
/// DEVICE key width (was <<20) so the golden select set is identical to the kernel's, bit for bit.
fn dsa_pack_key(sc: f32, t: usize, len: usize) -> u64 {
    let mut sb = sc.to_bits();
    sb = if sb & 0x8000_0000 != 0 { !sb } else { sb | 0x8000_0000 };
    ((sb as u64) << 24) | (((len - 1 - t) as u64) & 0xFF_FFFF)
}

/// The device radix selector's RESULT set: the top_k positions by dsa_pack_key (as d_index_select_coop).
fn index_select(score: &[f32], top_k: usize) -> Vec<usize> {
    let len = score.len();
    let mut keyed: Vec<(u64, usize)> = (0..len).map(|t| (dsa_pack_key(score[t], t, len), t)).collect();
    keyed.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // descending key
    let mut sel: Vec<usize> = keyed[..top_k.min(len)].iter().map(|&(_, t)| t).collect();
    sel.sort_unstable();
    sel
}

/// Self-check the indexer reference: the packed-key selector must equal a brute-force top-k
/// argsort (score desc, index asc on ties) on synthetic bf16-rounded inputs, at several ctx/top_k.
fn indexer_selfcheck() {
    let (hi, di) = (32usize, 128usize); // GLM index_n_heads / index_head_dim
    let scale = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
    for (ctx, top_k) in [(4096usize, 2048usize), (32768, 2048), (2000, 512), (128 * 1024, 2048)] {
        let seed = ((ctx as u64) << 20) ^ top_k as u64;
        let mut qidx = vec![0f32; hi * di];
        for x in qidx.iter_mut().enumerate() { *x.1 = q(rnd(seed ^ 0x11 ^ x.0 as u64, 0.1)); }
        let mut w = vec![0f32; hi];
        for x in w.iter_mut().enumerate() { *x.1 = q(rnd(seed ^ 0x22 ^ x.0 as u64, 1.0)); }
        let mut kidx = vec![0f32; ctx * di];
        for x in kidx.iter_mut().enumerate() { *x.1 = q(rnd(seed ^ 0x33 ^ x.0 as u64, 0.1)); }
        let score = index_score(&qidx, &kidx, &w, hi, di, ctx, scale);
        let sel = index_select(&score, top_k);
        // brute-force golden: sort positions by (score desc, index asc), take top_k, sort ascending.
        let mut order: Vec<usize> = (0..ctx).collect();
        order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap().then(a.cmp(&b)));
        let mut want: Vec<usize> = order[..top_k].to_vec();
        want.sort_unstable();
        assert_eq!(sel, want, "index_select != brute-force top-k at ctx={ctx} top_k={top_k}");
        eprintln!("[indexer-selfcheck] ctx={ctx:>7} top_k={top_k}: select==brute-force top-k OK");
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "fixture.bin".into());
    if std::env::args().any(|a| a == "--indexer-selfcheck") {
        indexer_selfcheck();
        return;
    }

    // Cases: nsplit sweep (a split-KV decode must give the SAME answer for every nsplit),
    // GF=8 head fusion at 1/2/3 head-groups, and a non-multiple ctx so the KV tail-tile is
    // exercised. ctx up to 4096 so the O(ctx) latent loop is a real long-context decode.
    let cases = [
        Case { n_head: 8, ctx: 4096, nsplit: 1, top_k: 0 },
        Case { n_head: 8, ctx: 4096, nsplit: 8, top_k: 0 },
        Case { n_head: 16, ctx: 4096, nsplit: 4, top_k: 0 },
        Case { n_head: 8, ctx: 4093, nsplit: 4, top_k: 0 },
        Case { n_head: 24, ctx: 2000, nsplit: 8, top_k: 0 },
        // GLM-5.2 real head count (64 => the head-packed MFMA 2-M-tile path); dense + gather.
        Case { n_head: 64, ctx: 4096, nsplit: 1, top_k: 0 },
        Case { n_head: 64, ctx: 4096, nsplit: 8, top_k: 0 },
        Case { n_head: 64, ctx: 2000, nsplit: 8, top_k: 0 },
        // Sparse DSA compose: gathered MLA over a selected top_k subset of the 4k context.
        Case { n_head: 8, ctx: 4096, nsplit: 1, top_k: 2048 },
        Case { n_head: 16, ctx: 4096, nsplit: 4, top_k: 512 },
        Case { n_head: 8, ctx: 4096, nsplit: 8, top_k: 300 },
        Case { n_head: 64, ctx: 4096, nsplit: 4, top_k: 2048 },
    ];
    let scale: f32 = 0.08838835; // 1/sqrt(128) — the DeepSeek qk_head_dim scale

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&0x4d4c4131u32.to_le_bytes()); // "MLA1"
    out.extend_from_slice(&(cases.len() as u32).to_le_bytes());

    for (ci, c) in cases.iter().enumerate() {
        let nh = c.n_head as usize;
        let ctx = c.ctx as usize;
        let cs = (ci as u64) << 48; // per-case seed salt

        // --- seed inputs, rounded through bf16 (exactly what the fixture stores) ---
        // C_kv[t][l], K_rope[t][d]: the latent cache. Small so the softmax is well-conditioned.
        let mut ckv = vec![0f32; ctx * DK];
        let mut ckv_bf = vec![0u16; ctx * DK];
        for t in 0..ctx {
            for l in 0..DK {
                let v = rnd(cs ^ (0x01 << 40) ^ ((t as u64) << 12) ^ l as u64, Q_AMP);
                let b = f2bf(v);
                ckv_bf[t * DK + l] = b;
                ckv[t * DK + l] = bf2f(b);
            }
        }
        let mut krope = vec![0f32; ctx * DR];
        let mut krope_bf = vec![0u16; ctx * DR];
        for t in 0..ctx {
            for d in 0..DR {
                let v = rnd(cs ^ (0x02 << 40) ^ ((t as u64) << 12) ^ d as u64, Q_AMP);
                let b = f2bf(v);
                krope_bf[t * DR + d] = b;
                krope[t * DR + d] = bf2f(b);
            }
        }
        // Q_abs[h][l] (= W_uk^T q_nope, absorbed on the query side), Q_rope[h][d].
        let mut qabs = vec![0f32; nh * DK];
        let mut qabs_bf = vec![0u16; nh * DK];
        for h in 0..nh {
            for l in 0..DK {
                let v = rnd(cs ^ (0x03 << 40) ^ ((h as u64) << 20) ^ l as u64, Q_AMP);
                let b = f2bf(v);
                qabs_bf[h * DK + l] = b;
                qabs[h * DK + l] = bf2f(b);
            }
        }
        let mut qrope = vec![0f32; nh * DR];
        let mut qrope_bf = vec![0u16; nh * DR];
        for h in 0..nh {
            for d in 0..DR {
                let v = rnd(cs ^ (0x04 << 40) ^ ((h as u64) << 20) ^ d as u64, 0.10);
                let b = f2bf(v);
                qrope_bf[h * DR + d] = b;
                qrope[h * DR + d] = bf2f(b);
            }
        }
        // W_uv[h][l][v] — the value up-projection (l-major, v-minor).
        let mut wuv = vec![0f32; nh * DK * V];
        let mut wuv_bf = vec![0u16; nh * DK * V];
        for h in 0..nh {
            for l in 0..DK {
                for vv in 0..V {
                    let s = cs ^ (0x05 << 40) ^ ((h as u64) << 32) ^ ((l as u64) << 8) ^ vv as u64;
                    let val = rnd(s, 0.05);
                    let b = f2bf(val);
                    wuv_bf[(h * DK + l) * V + vv] = b;
                    wuv[(h * DK + l) * V + vv] = bf2f(b);
                }
            }
        }

        // --- the selected KV set the query attends to ---
        // Dense: all rows 0..=qpos (qpos = the newest token). Gather: a deterministic distinct
        // top_k subset (ascending) — the on-device index table the DSA selector would produce.
        let qpos = ctx - 1;
        let sel: Vec<usize> = if c.top_k == 0 {
            (0..=qpos).collect()
        } else {
            let mut ids: Vec<u32> = (0..ctx as u32).collect();
            ids.sort_by_key(|&i| mix(cs ^ (0x06u64 << 40) ^ i as u64));
            let mut s: Vec<usize> = ids[..c.top_k as usize].iter().map(|&i| i as usize).collect();
            s.sort();
            s
        };

        // --- golden: absorbed MLA over `sel`, f32, mirroring the kernel's reduction boundaries ---
        let mut golden = vec![0f32; nh * V];
        for h in 0..nh {
            // score[t] = scale * ( q_abs[h].C_kv[t] + q_rope[h].K_rope[t] ),  t in sel
            let mut mx = f32::NEG_INFINITY;
            let mut sc = vec![0f32; sel.len()];
            for (si, &t) in sel.iter().enumerate() {
                let mut d = 0f32;
                for l in 0..DK {
                    d += qabs[h * DK + l] * ckv[t * DK + l];
                }
                for r in 0..DR {
                    d += qrope[h * DR + r] * krope[t * DR + r];
                }
                sc[si] = d * scale;
                if sc[si] > mx {
                    mx = sc[si];
                }
            }
            let mut sum = 0f32;
            let mut p = vec![0f32; sel.len()];
            for si in 0..sel.len() {
                p[si] = (sc[si] - mx).exp();
                sum += p[si];
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            // oacc[l] = sum p[t]/sum * C_kv[t][l]; round to bf16 (the device merge writes bf16).
            let mut oacc = vec![0f32; DK];
            for l in 0..DK {
                let mut acc = 0f32;
                for (si, &t) in sel.iter().enumerate() {
                    acc += p[si] * ckv[t * DK + l];
                }
                oacc[l] = q(acc * inv);
            }
            // o[v] = sum_l oacc[l] * W_uv[h][l][v]  — f32 accumulate, l in natural order.
            for vv in 0..V {
                let mut acc = 0f32;
                for l in 0..DK {
                    acc += oacc[l] * wuv[(h * DK + l) * V + vv];
                }
                golden[h * V + vv] = acc;
            }
        }

        // --- serialize the case ---
        for x in [c.n_head, DK as u32, DR as u32, V as u32, c.ctx, c.nsplit, c.top_k] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&scale.to_le_bytes());
        if c.top_k > 0 {
            for &t in &sel {
                out.extend_from_slice(&(t as i32).to_le_bytes()); // the idx table
            }
        }
        let push_bf = |o: &mut Vec<u8>, a: &[u16]| {
            for &b in a {
                o.extend_from_slice(&b.to_le_bytes());
            }
        };
        push_bf(&mut out, &ckv_bf);
        push_bf(&mut out, &krope_bf);
        push_bf(&mut out, &qabs_bf);
        push_bf(&mut out, &qrope_bf);
        push_bf(&mut out, &wuv_bf);
        for &g in &golden {
            out.extend_from_slice(&g.to_le_bytes());
        }
        eprintln!(
            "case {ci}: n_head={} ctx={} nsplit={} top_k={}  (DK={DK} DR={DR} V={V})",
            c.n_head, c.ctx, c.nsplit, c.top_k
        );
    }

    let mut f = std::fs::File::create(&path).expect("create fixture");
    f.write_all(&out).expect("write fixture");
    eprintln!("wrote {} ({} bytes)", path, out.len());
}
