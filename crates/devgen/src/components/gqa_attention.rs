//! NoPE GQA attention on the sm_120 interpreter: q/k/v projections, K/V stored by
//! `QwenHeadNormRope` (no norm, no rotary) into the head-major cache `kv.<layer>.{k,v}`, the
//! fused one-split `FlashPrefill` (bf16 epilogue into `at`, no merge) or split `FlashDecode` +
//! `FlashMerge`, o projection.
use kernelcaps::arms::{Arms, MissingArm};
use packet::dev::{DevOp, TENSOR_NONE};
use packet::typed_abi::{
    FlashDecodeArgs, FlashMergeArgs, FlashPrefillArgs, QwenHeadNormRopeArgs, TypedOp,
};

use super::{Program, Proj};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct GqaAttention {
    g: Geometry,
}

impl GqaAttention {
    pub fn new(g: Geometry, arms: &Arms) -> Result<Self, MissingArm> {
        arms.require(DevOp::FlashPrefill, "hd", g.head_dim)?;
        arms.require(DevOp::FlashDecode, "hd", g.head_dim)?;
        Ok(Self { g })
    }

    /// The attention of `module` at `layer`: `mixed = o(attn(q(hn), k(hn), v(hn)))`. `kv_e4m3`:
    /// the cache holds unit-scale E4M3 (`live_kv::KV_E4M3`), one byte per element.
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &self,
        p: &mut Program,
        layer: usize,
        module: &str,
        hn: u32,
        mixed: u32,
        dep: u32,
        kv_e4m3: bool,
        proj: &mut Proj,
    ) -> u32 {
        let g = &self.g;
        let (hidden, heads, kvh, hd) = (g.hidden, g.heads, g.kv_heads, g.head_dim);
        let (qd, kd) = (heads * hd, kvh * hd);
        let q = p.act("q", qd);
        let kg = p.act("kg", kd);
        let vg = p.act("vg", kd);
        let at = p.act("at", qd);
        let kv_bytes =
            u64::from(p.slots) * u64::from(kd) * u64::from(p.ctx) * if kv_e4m3 { 1 } else { 2 };
        let kc = p.b.tensor(&format!("kv.{layer}.k"), kv_bytes);
        let vc = p.b.tensor(&format!("kv.{layer}.v"), kv_bytes);
        let dq = proj(p, q, hn, &format!("{module}.q_proj"), qd, hidden, dep);
        // Prefill chains q -> k -> v (k and v reuse q's quantized input). Decode GEMVs read hn
        // only: k and v wait on the norm and the cache writes follow all three, so no block
        // waits on a projection it does not read.
        let flash_deps = if p.prefill {
            let dk = proj(p, kg, hn, &format!("{module}.k_proj"), kd, hidden, dq);
            let dk = kv_write(p, kc, kg, kvh, hd, kv_e4m3, dk);
            let dv = proj(p, vg, hn, &format!("{module}.v_proj"), kd, hidden, dk);
            vec![kv_write(p, vc, vg, kvh, hd, kv_e4m3, dv)]
        } else {
            let dk = proj(p, kg, hn, &format!("{module}.k_proj"), kd, hidden, dep);
            let dv = proj(p, vg, hn, &format!("{module}.v_proj"), kd, hidden, dep);
            let wk = kv_write(p, kc, kg, kvh, hd, kv_e4m3, dk);
            let wv = kv_write(p, vc, vg, kvh, hd, kv_e4m3, dv);
            vec![dq, wk, wv]
        };
        let n_cu = p.b.n_cu();
        let rows = p.rows;
        // Prefill: one split with the fused bf16 epilogue (O_final = at, no merge).
        let ns = if p.prefill {
            1
        } else {
            let gf = crate::hybrid_decode_gf(heads / kvh, hd, kv_e4m3);
            let balanced = crate::balanced_attention_ns(rows, heads, kvh, n_cu, gf);
            crate::attention_decode_ns(rows, heads, kvh, n_cu, balanced)
        };
        let opart = p.b.tensor(
            "act.opart",
            u64::from(rows) * u64::from(qd) * u64::from(ns) * 4,
        );
        let mlpart = p.b.tensor(
            "act.mlpart",
            u64::from(rows) * u64::from(heads) * u64::from(ns) * 8,
        );
        let scale = 1.0 / (hd as f32).sqrt();
        let (ctx, kvlen) = (p.ctx, p.kvlen);
        let dm = if p.prefill {
            let cus = p.b.all();
            let da = FlashPrefillArgs {
                Opart: opart,
                mlpart,
                Q: q,
                K: kc,
                V: vc,
                O_final: at,
                n_q: rows,
                n_kv: rows,
                n_head: heads,
                n_kv_head: kvh,
                hd,
                nsplit: ns,
                scale,
                kv_stride: ctx,
                kv_mask: u32::MAX,
                ..Default::default()
            }
            .emit(&mut p.b, cus, &flash_deps);
            // The packed contract: request tiles change the attention's slice ownership.
            if p.packed {
                p.b.isolate(da);
            }
            da
        } else {
            let cus = p.b.all();
            let da = FlashDecodeArgs {
                Opart: opart,
                mlpart,
                Q: q,
                K: kc,
                V: vc,
                kv_len: kvlen,
                n_batch: rows,
                n_head: heads,
                n_kv_head: kvh,
                kv_stride: ctx,
                nsplit: ns,
                hd,
                kv_mask: u32::MAX,
                scale,
                ..Default::default()
            }
            .emit(&mut p.b, cus, &flash_deps);
            FlashMergeArgs {
                O: at,
                Opart: opart,
                mlpart,
                n_batch: rows,
                n_head: heads,
                nsplit: ns,
                hd,
                ..Default::default()
            }
            .emit(&mut p.b, (0..heads.min(n_cu)).collect(), &[da])
        };
        proj(p, mixed, at, &format!("{module}.o_proj"), hidden, qd, dm)
    }
}

/// Stores `src` (`heads` x `hd` per row) into the head-major cache `cache` at `in.pos` of each
/// row's slot (`QwenHeadNormRope` with no norm and no rotary).
pub(crate) fn kv_write(
    p: &mut Program,
    cache: u32,
    src: u32,
    heads: u32,
    hd: u32,
    kv_e4m3: bool,
    dep: u32,
) -> u32 {
    let e4m3 = if kv_e4m3 {
        plow_asset::live_kv::KV_E4M3
    } else {
        0
    };
    store(p, cache, src, (heads, hd), e4m3, TENSOR_NONE, dep)
}

/// [`kv_write`] into an E4M3 cache at the per-tensor scale `scale` (an f32 `[1]` tensor):
/// `sat_e4m3(x / scale)` (`live_kv::KV_SCALE`).
pub(crate) fn kv_write_scaled(
    p: &mut Program,
    cache: u32,
    src: u32,
    heads: u32,
    hd: u32,
    scale: u32,
    dep: u32,
) -> u32 {
    use plow_asset::live_kv::{KV_E4M3, KV_SCALE};
    store(p, cache, src, (heads, hd), KV_E4M3 | KV_SCALE, scale, dep)
}

fn store(p: &mut Program, cache: u32, src: u32, geom: (u32, u32), kv: u32, scale: u32, dep: u32) -> u32 {
    let args = QwenHeadNormRopeArgs {
        out: cache,
        x: src,
        cos: scale,
        pos: p.pos,
        active: p.active,
        rowslot: TENSOR_NONE,
        H: geom.0,
        D: geom.1,
        rows: p.rows,
        ctx: p.ctx,
        prefill: p.prefill as u32,
        layout: plow_asset::live_kv::LAYOUT_HEAD_MAJOR | kv,
        ..Default::default()
    };
    p.emit(args, &[dep])
}
