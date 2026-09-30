//! DeepSeek-V4.1 weights in the sm_90a engine's device formats, uploaded from the HF shards.
//!
//! Everything stays in its checkpoint encoding (fp8 e4m3 + ue8m0 [32,32] grids, fp4 experts + ue8m0
//! per-32 scales, bf16 norms, f32 hyper-connection parameters); the kernels dequantize on the fly.
//! The routed experts are repacked per layer as `W13[e][2*I][H/2]` (w1 rows then w3 rows) and
//! `W2[e][H][I/2]` so one grouped GEMM covers gate and up.

use std::sync::Arc;

use crate::asset::Checkpoint;
use crate::device::cuda::CudaBackend;
use crate::device::{Backend, DeviceMem};
use crate::error::{Result, RuntimeError};

use super::config::Cfg;

/// One device allocation holding many tensors, filled by offset.
pub struct Slab {
    pub mem: DeviceMem,
    off: u64,
}

impl Slab {
    pub fn new(dev: &CudaBackend, bytes: u64) -> Result<Self> {
        Ok(Slab { mem: dev.alloc(dev.device_ordinal, bytes)?, off: 0 })
    }
    /// Reserve `bytes` (256-aligned) and return the device pointer.
    pub fn take(&mut self, bytes: u64) -> Result<u64> {
        let at = self.off.next_multiple_of(256);
        if at + bytes > self.mem.len {
            return Err(RuntimeError::Device(format!(
                "dsv41 slab overflow: {} + {} > {}",
                at, bytes, self.mem.len
            )));
        }
        self.off = at + bytes;
        Ok(self.mem.base + at)
    }
}

fn get<'a>(ck: &'a Checkpoint, name: &str) -> Result<&'a [u8]> {
    ck.tensor(name)
        .ok_or_else(|| RuntimeError::Device(format!("dsv41: checkpoint has no {name}")))
}

/// Upload one checkpoint tensor into the slab.
fn up(dev: &CudaBackend, slab: &mut Slab, ck: &Checkpoint, name: &str) -> Result<u64> {
    let b = get(ck, name)?;
    let p = slab.take(b.len() as u64)?;
    dev.memcpy_htod(p, b)?;
    Ok(p)
}

/// Upload two tensors back to back (w1 then w3 rows) as one.
fn up2(dev: &CudaBackend, slab: &mut Slab, ck: &Checkpoint, a: &str, b: &str) -> Result<u64> {
    let (x, y) = (get(ck, a)?, get(ck, b)?);
    let p = slab.take((x.len() + y.len()) as u64)?;
    dev.memcpy_htod(p, x)?;
    dev.memcpy_htod(p + x.len() as u64, y)?;
    Ok(p)
}

#[derive(Default, Clone, Copy)]
pub struct Fp8 {
    pub w: u64,
    pub s: u64,
}

pub struct Layer {
    pub id: usize,
    pub ratio: usize,
    pub kv_source: bool,
    pub index_source: bool,
    pub cand_source: bool,
    pub uses_cand: bool,
    pub engram: bool,
    pub attn_norm: u64,
    pub ffn_norm: u64,
    pub hc_attn: [u64; 3], // fn f32 [24][4H], scale [3], base [24]
    pub hc_ffn: [u64; 3],
    pub wq_a: Fp8,
    pub q_norm: u64,
    pub wq_b: Fp8,
    pub wkv: Fp8,
    pub kv_norm: u64,
    pub wo_a: Fp8,
    pub wo_b: Fp8,
    pub sink: u64,
    // compressor (kv sources)
    pub c_wkv: u64,
    pub c_wgate: u64,
    pub c_norm: u64,
    // indexer (index sources)
    pub i_wq_b: Fp8,
    pub i_wproj: u64,
    pub i_wk: u64,
    pub i_knorm: u64,
    // ffn
    pub gate_w: u64,
    pub gate_b: u64,
    pub sh_w13: Fp8,
    pub sh_w2: Fp8,
    pub w13: u64,
    pub w13_s: u64,
    pub w2: u64,
    pub w2_s: u64,
    // engram
    pub e_wkv: Fp8,
    pub e_qw: u64,
    pub e_kw: u64,
    _slab: Slab,
}

/// Bytes a layer's slab needs: every tensor under `layers.{L}.` except the Engram tables, plus
/// alignment slack.
fn layer_bytes(ck: &Checkpoint, cfg: &Cfg, l: usize) -> u64 {
    let mut total = 0u64;
    let mut add = |n: &str| {
        if let Some(b) = ck.tensor(n) {
            total += (b.len() as u64).next_multiple_of(256) + 256;
        }
    };
    for n in cfg.layer_tensor_names(l) {
        add(&n);
    }
    total
}

impl Layer {
    pub fn load(dev: &Arc<CudaBackend>, ck: &Checkpoint, cfg: &Cfg, l: usize) -> Result<Layer> {
        let d: &CudaBackend = dev;
        let mut slab = Slab::new(d, layer_bytes(ck, cfg, l))?;
        let p = format!("layers.{l}.");
        let n = |s: &str| format!("{p}{s}");
        let fp8 = |slab: &mut Slab, base: &str| -> Result<Fp8> {
            Ok(Fp8 {
                w: up(d, slab, ck, &format!("{p}{base}.weight"))?,
                s: up(d, slab, ck, &format!("{p}{base}.scale"))?,
            })
        };
        let ratio = cfg.compress_ratios[l];
        let kv_source = cfg.kv_source.contains(&l);
        let index_source = cfg.index_source.contains(&l);
        let engram = ck.tensor(&n("engram.wkv.weight")).is_some();
        let hc = |slab: &mut Slab, k: &str| -> Result<[u64; 3]> {
            Ok([
                up(d, slab, ck, &n(&format!("hc_{k}_fn")))?,
                up(d, slab, ck, &n(&format!("hc_{k}_scale")))?,
                up(d, slab, ck, &n(&format!("hc_{k}_base")))?,
            ])
        };
        let attn_norm = up(d, &mut slab, ck, &n("attn_norm.weight"))?;
        let ffn_norm = up(d, &mut slab, ck, &n("ffn_norm.weight"))?;
        let hc_attn = hc(&mut slab, "attn")?;
        let hc_ffn = hc(&mut slab, "ffn")?;
        let wq_a = fp8(&mut slab, "attn.wq_a")?;
        let q_norm = up(d, &mut slab, ck, &n("attn.q_norm.weight"))?;
        let wq_b = fp8(&mut slab, "attn.wq_b")?;
        let wkv = fp8(&mut slab, "attn.wkv")?;
        let kv_norm = up(d, &mut slab, ck, &n("attn.kv_norm.weight"))?;
        let wo_a = fp8(&mut slab, "attn.wo_a")?;
        let wo_b = fp8(&mut slab, "attn.wo_b")?;
        let sink = up(d, &mut slab, ck, &n("attn.attn_sink"))?;
        let (mut c_wkv, mut c_wgate, mut c_norm) = (0, 0, 0);
        if kv_source {
            c_wkv = up(d, &mut slab, ck, &n("attn.compressor.wkv.weight"))?;
            c_norm = up(d, &mut slab, ck, &n("attn.compressor.norm.weight"))?;
            if ratio > 1 {
                c_wgate = up(d, &mut slab, ck, &n("attn.compressor.wgate.weight"))?;
            }
        }
        let (mut i_wq_b, mut i_wproj, mut i_wk, mut i_knorm) = (Fp8::default(), 0, 0, 0);
        if index_source {
            i_wq_b = fp8(&mut slab, "attn.indexer.wq_b")?;
            i_wproj = up(d, &mut slab, ck, &n("attn.indexer.weights_proj.weight"))?;
            if kv_source {
                i_wk = up(d, &mut slab, ck, &n("attn.indexer.wk.weight"))?;
                i_knorm = up(d, &mut slab, ck, &n("attn.indexer.k_norm.weight"))?;
            }
        }
        let gate_w = up(d, &mut slab, ck, &n("ffn.gate.weight"))?;
        let gate_b = up(d, &mut slab, ck, &n("ffn.gate.bias"))?;
        let sh_w13 = Fp8 {
            w: up2(d, &mut slab, ck, &n("ffn.shared_experts.w1.weight"), &n("ffn.shared_experts.w3.weight"))?,
            s: up2(d, &mut slab, ck, &n("ffn.shared_experts.w1.scale"), &n("ffn.shared_experts.w3.scale"))?,
        };
        let sh_w2 = fp8(&mut slab, "ffn.shared_experts.w2")?;
        // routed experts, repacked contiguously per expert
        let (h, mi, e) = (cfg.hidden as u64, cfg.moe_inter as u64, cfg.n_experts as u64);
        let w13 = slab.take(e * 2 * mi * h / 2)?;
        let w13_s = slab.take(e * 2 * mi * h / 32)?;
        let w2 = slab.take(e * h * mi / 2)?;
        let w2_s = slab.take(e * h * mi / 32)?;
        for x in 0..e {
            let ep = format!("{p}ffn.experts.{x}.");
            let put = |dst: u64, name: &str| -> Result<()> { d.memcpy_htod(dst, get(ck, &format!("{ep}{name}"))?) };
            put(w13 + x * 2 * mi * h / 2, "w1.weight")?;
            put(w13 + x * 2 * mi * h / 2 + mi * h / 2, "w3.weight")?;
            put(w13_s + x * 2 * mi * h / 32, "w1.scale")?;
            put(w13_s + x * 2 * mi * h / 32 + mi * h / 32, "w3.scale")?;
            put(w2 + x * h * mi / 2, "w2.weight")?;
            put(w2_s + x * h * mi / 32, "w2.scale")?;
        }
        let (mut e_wkv, mut e_qw, mut e_kw) = (Fp8::default(), 0, 0);
        if engram {
            e_wkv = fp8(&mut slab, "engram.wkv")?;
            e_qw = up(d, &mut slab, ck, &n("engram.q_weight"))?;
            e_kw = up(d, &mut slab, ck, &n("engram.k_weight"))?;
        }
        Ok(Layer {
            id: l,
            ratio,
            kv_source,
            index_source,
            cand_source: cfg.candidate_source == Some(l),
            uses_cand: cfg.candidate_source.is_some_and(|c| c < l),
            engram,
            attn_norm,
            ffn_norm,
            hc_attn,
            hc_ffn,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            wo_a,
            wo_b,
            sink,
            c_wkv,
            c_wgate,
            c_norm,
            i_wq_b,
            i_wproj,
            i_wk,
            i_knorm,
            gate_w,
            gate_b,
            sh_w13,
            sh_w2,
            w13,
            w13_s,
            w2,
            w2_s,
            e_wkv,
            e_qw,
            e_kw,
            _slab: slab,
        })
    }
}
