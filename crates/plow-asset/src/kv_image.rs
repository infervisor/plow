//! The vendor-neutral identity of what a packet's KV caches hold (docs/arch/30 §5.2).
//!
//! Two engines can exchange a KV image, whatever their vendor, exactly when their [`Header`]s are
//! equal. The header names each cache by model layer and states what its rows mean: class
//! (append or sliding window), heads, head dim, element type with its scale granularity, the
//! normalization and rotary embedding applied before the row was stored, and the checkpoint the
//! values came from. Native bindings stay out of it: tensor handles, layout, ring depth, stride,
//! slot count, programs and context capacity.
//!
//! Derived from the packet at load, so no re-emit is needed. Anything the derivation does not
//! recognise is refused rather than guessed. This includes a cache whose writer was fused away,
//! a rotary table baked into the blob, a TP packet, an F16 cache and any FP8 form other than
//! E4M3 with per-(head, row) F32 scales.

use crate::live_kv::Manifest;
use crate::program::Packet;
use packet::dev::{DevInst64, DevOp, ROPE_PAIR_HALF, TENSOR_NONE16};
use packet::rope::{
    GenTensor, GEN_AMD_ROPE_BF16_COS, GEN_AMD_ROPE_BF16_SIN, GEN_ROPE_COS, GEN_ROPE_SIN,
    ROPE_SCALE_NONE,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const VERSION: u32 = 1;

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub version: u32,
    /// [`checkpoint_digest`] of the weights the rows were computed from.
    pub weights: String,
    /// Owned caches in (layer, K before V) order.
    pub entries: Vec<Entry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// `L{layer}.k` or `L{layer}.v`.
    pub id: String,
    pub class: Class,
    pub heads: u32,
    pub hd: u32,
    pub dtype: Dtype,
    /// Applied to each head before the row is stored.
    pub norm: Norm,
    /// Applied after the norm; `None` stores unrotated rows.
    pub rope: Option<Rope>,
    /// Layers attending over this cache (more than one under KV sharing), ascending.
    pub readers: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Append,
    /// Attention sees the last `rows` positions.
    Window {
        rows: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Bf16,
    /// OCP E4M3 elements with one F32 dequantization scale per (head, row).
    E4m3RowHeadF32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Norm {
    None,
    /// RMS over the head dim; `gamma` false is the weightless form.
    Rms {
        eps: f32,
        gamma: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pairing {
    /// `(i, i + hd/2)`.
    Half,
    /// `(2i, 2i + 1)`.
    Interleaved,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub theta: f64,
    /// Rotated fraction of the head dim.
    pub frac: f64,
    pub pairing: Pairing,
    /// `packet::rope::ROPE_SCALE_*` and its parameters (zero when unused).
    pub scaling: u32,
    pub factor: f64,
    pub low: f64,
    pub high: f64,
    pub orig: f64,
    pub aux: u32,
}

impl Header {
    /// The exchange key: equal digests, equal headers.
    pub fn digest(&self) -> String {
        let json = serde_json::to_vec(self).expect("a header serializes");
        let mut h = Sha256::new();
        h.update(b"plow-kv-image-header-v1");
        h.update(&json);
        format!("{:x}", h.finalize())
    }

    /// The first place `other` differs from `self`, e.g. `L5.k.rope` or `weights`.
    pub fn first_difference(&self, other: &Self) -> Option<String> {
        if self.version != other.version {
            return Some("version".into());
        }
        if self.weights != other.weights {
            return Some("weights".into());
        }
        for (a, b) in self.entries.iter().zip(&other.entries) {
            if a.id != b.id {
                return Some(format!("entry {} vs {}", a.id, b.id));
            }
            let field = [
                ("class", a.class != b.class),
                ("heads", a.heads != b.heads),
                ("hd", a.hd != b.hd),
                ("dtype", a.dtype != b.dtype),
                ("norm", a.norm != b.norm),
                ("rope", a.rope != b.rope),
                ("readers", a.readers != b.readers),
            ]
            .into_iter()
            .find_map(|(name, differs)| differs.then_some(name));
            if let Some(field) = field {
                return Some(format!("{}.{field}", a.id));
            }
        }
        let (n, m) = (self.entries.len(), other.entries.len());
        (n != m).then(|| {
            let longer = if n > m { &self.entries } else { &other.entries };
            format!("entry {} present on one side only", longer[n.min(m)].id)
        })
    }
}

/// `packet`'s header, from its live-KV `manifest` and the [`checkpoint_digest`] of its weights.
pub fn derive(packet: &Packet<'_>, manifest: &Manifest, weights: String) -> Result<Header> {
    let refuse = |why: String| format!("KV image header: {why}");
    if packet.tp {
        return Err(refuse("a TP packet's caches hold a head slice".into()));
    }
    let name = |h: u16| packet.tensors.get(h as usize).map(|t| t.name);
    let mut entries = Vec::new();
    for c in &manifest.caches {
        let layer = cache_layer(packet, c.pair[0]).map_err(&refuse)?;
        if cache_layer_role(name(c.pair[1]), "v") != Some(layer) {
            return Err(refuse(format!("L{layer}: V cache is not kv.{layer}.v")));
        }
        let readers = readers(packet, c.pair).map_err(|e| refuse(format!("L{layer}: {e}")))?;
        for (handle, role) in c.pair.into_iter().zip(["k", "v"]) {
            let id = format!("L{layer}.{role}");
            let mut facts = None;
            for p in packet.programs {
                for d in p.insts.iter().filter(|d| writes_cache(d)) {
                    let f = if d.t[0] == handle {
                        writer(packet, d, c.hd, layer, role)
                    } else if fused_v(d) && d.t[6] == handle && role == "v" {
                        writer_fused_v(d, c.hd)
                    } else {
                        continue;
                    }
                    .map_err(|e| refuse(format!("{id}: {e}")))?;
                    match &facts {
                        None => facts = Some(f),
                        Some(seen) if *seen == f => {}
                        Some(_) => return Err(refuse(format!("{id}: its writers disagree"))),
                    }
                }
            }
            let (dtype, norm, rope) = facts.ok_or_else(|| {
                refuse(format!("{id}: no recognised writer (fused or unsupported)"))
            })?;
            if (dtype == Dtype::E4m3RowHeadF32) != c.scales.is_some() {
                return Err(refuse(format!("{id}: writer and reader disagree on FP8")));
            }
            entries.push(Entry {
                id,
                class: if c.window == 0 {
                    Class::Append
                } else {
                    Class::Window { rows: c.window }
                },
                heads: c.heads,
                hd: c.hd,
                dtype,
                norm,
                rope,
                readers: readers.clone(),
            });
        }
    }
    entries.sort_by_key(|e| {
        let (l, role) = e.id[1..].split_once('.').expect("L<layer>.<role>");
        (l.parse::<u32>().expect("parsed above"), role == "v")
    });
    Ok(Header {
        version: VERSION,
        weights,
        entries,
    })
}

/// Put `manifest`'s caches in the header's order (by layer), the order a KV image packs them in:
/// two packets declaring the same caches in different orders then exchange the same bytes.
pub fn canonical_order(packet: &Packet<'_>, manifest: &mut Manifest) -> Result<()> {
    let mut keyed = Vec::with_capacity(manifest.caches.len());
    for c in manifest.caches.drain(..) {
        keyed.push((cache_layer(packet, c.pair[0])?, c));
    }
    keyed.sort_by_key(|(l, _)| *l);
    manifest.caches = keyed.into_iter().map(|(_, c)| c).collect();
    Ok(())
}

fn cache_layer(packet: &Packet<'_>, k: u16) -> Result<u32> {
    let name = packet.tensors.get(k as usize).map(|t| t.name);
    cache_layer_role(name, "k").ok_or_else(|| format!("cache tensor {name:?} is not kv.<layer>.k"))
}

fn cache_layer_role(name: Option<&str>, role: &str) -> Option<u32> {
    name?
        .strip_prefix("kv.")?
        .strip_suffix(&format!(".{role}"))?
        .parse()
        .ok()
}

/// The layer of a `{prefix}layers.{l}.self_attn.{norm}.weight` tensor.
fn norm_layer(name: &str, norm: &str) -> Option<u32> {
    let rest = name.strip_suffix(&format!(".self_attn.{norm}.weight"))?;
    let (_, l) = rest.rsplit_once("layers.")?;
    l.parse().ok()
}

/// The layers attending over cache `pair`: each reader's query comes from the nearest earlier
/// `HeadNormRope` writing it, normed with its layer's `q_norm`. Every program reading the cache
/// must name the same layers.
fn readers(packet: &Packet<'_>, pair: [u16; 2]) -> Result<Vec<u32>> {
    let mut seen: Option<Vec<u32>> = None;
    for p in packet.programs {
        let mut layers = Vec::new();
        for (pc, d) in p.insts.iter().enumerate() {
            let op = DevOp::from_u16(d.op);
            if !matches!(
                op,
                Some(
                    DevOp::FlashDecode
                        | DevOp::FlashDecodeFp8
                        | DevOp::FlashPrefill
                        | DevOp::FlashPrefillFp8
                )
            ) || [d.t[3], d.t[4]] != pair
            {
                continue;
            }
            if op == Some(DevOp::FlashDecode) && d.t[7] != TENSOR_NONE16 {
                return Err("its decode writes rows inside the attention (folded writer)".into());
            }
            let q = p.insts[..pc]
                .iter()
                .rev()
                .find(|w| writes_cache(w) && w.t[0] == d.t[2])
                .and_then(|w| packet.tensors.get(w.t[2] as usize))
                .and_then(|g| norm_layer(g.name, "q_norm"))
                .ok_or("a reader's layer is not identifiable from its query norm")?;
            layers.push(q);
        }
        if layers.is_empty() {
            continue;
        }
        // A staged prefill repeats a layer's attention.
        layers.sort_unstable();
        layers.dedup();
        match &seen {
            None => seen = Some(layers),
            Some(s) if *s == layers => {}
            Some(_) => return Err("programs disagree on the layers reading it".into()),
        }
    }
    seen.ok_or_else(|| "no program reads it".into())
}

fn writes_cache(d: &DevInst64) -> bool {
    matches!(
        DevOp::from_u16(d.op),
        Some(DevOp::HeadNormRope | DevOp::HeadNormRopeFp8 | DevOp::QwenHeadNormRope)
    )
}

/// `HeadNormRope` `i7=1`: the instruction also stores V, from `t7` into `t6`, with the same norm
/// skip, no gamma and no rotation (devgen's K/V pair fusion).
fn fused_v(d: &DevInst64) -> bool {
    d.op == DevOp::HeadNormRope as u16 && d.i[7] == 1
}

fn writer_fused_v(d: &DevInst64, hd: u32) -> Result<(Dtype, Norm, Option<Rope>)> {
    if d.i[2] != hd {
        return Err(format!("fused V writer hd {} vs cache hd {hd}", d.i[2]));
    }
    let norm = if d.i[4] == 0 {
        Norm::Rms {
            eps: finite(f32::from_bits(d.fj[0]).into(), "norm eps")? as f32,
            gamma: false,
        }
    } else {
        Norm::None
    };
    Ok((Dtype::Bf16, norm, None))
}

/// A float the header can hold: finite, and `-0.0` as `0.0` (JSON writes them differently).
fn finite(x: f64, what: &str) -> Result<f64> {
    if x.is_finite() {
        Ok(if x == 0.0 { 0.0 } else { x })
    } else {
        Err(format!("{what} is not finite"))
    }
}

/// `HeadNormRope[Fp8]`: `t0=out t2=gamma? t3=cos? t4=sin?` · `i2=hd i4=skip_norm i5=pair_mode`
/// · `f0=eps`. A gamma must be this layer's `{role}_norm`.
fn writer(
    packet: &Packet<'_>,
    d: &DevInst64,
    hd: u32,
    layer: u32,
    role: &str,
) -> Result<(Dtype, Norm, Option<Rope>)> {
    let dtype = match DevOp::from_u16(d.op) {
        Some(DevOp::HeadNormRope) => Dtype::Bf16,
        Some(DevOp::HeadNormRopeFp8) => Dtype::E4m3RowHeadF32,
        _ => return Err("writer op is not supported".into()),
    };
    if d.i[2] != hd {
        return Err(format!("writer hd {} vs cache hd {hd}", d.i[2]));
    }
    let gamma = d.t[2] != TENSOR_NONE16;
    if gamma {
        let g = packet.tensors.get(d.t[2] as usize).map_or("", |t| t.name);
        if norm_layer(g, &format!("{role}_norm")) != Some(layer) {
            return Err(format!(
                "writer gamma {g:?} is not layer {layer}'s {role}_norm"
            ));
        }
    }
    let norm = match (d.i[4], gamma) {
        (0, _) => Norm::Rms {
            eps: finite(f32::from_bits(d.fj[0]).into(), "norm eps")? as f32,
            gamma,
        },
        (_, false) => Norm::None,
        (_, true) => return Err("a gamma with the norm skipped".into()),
    };
    let rope = if d.t[3] == TENSOR_NONE16 {
        None
    } else {
        let pairing = match (d.i[5], hd) {
            (0, 64) | (1, 64 | 128) => Pairing::Interleaved,
            (0 | 1, _) | (ROPE_PAIR_HALF, _) => Pairing::Half,
            (mode, _) => return Err(format!("rotary pairing mode {mode}")),
        };
        let cos = generated(packet, d.t[3], [GEN_ROPE_COS, GEN_AMD_ROPE_BF16_COS])?;
        let sin = generated(packet, d.t[4], [GEN_ROPE_SIN, GEN_AMD_ROPE_BF16_SIN])?;
        if cos.hd != hd || sin.hd != hd {
            return Err(format!("rotary table hd {} vs cache hd {hd}", cos.hd));
        }
        let rope = recipe(cos, pairing)?;
        if recipe(sin, pairing)? != rope {
            return Err("cos and sin tables disagree".into());
        }
        Some(rope)
    };
    Ok((dtype, norm, rope))
}

/// A rotary table's recipe with its unused scaling parameters zeroed.
fn recipe(g: &GenTensor, pairing: Pairing) -> Result<Rope> {
    let scaled = g.scale != ROPE_SCALE_NONE;
    let used = |x: f64, what: &str| if scaled { finite(x, what) } else { Ok(0.0) };
    Ok(Rope {
        theta: finite(g.theta, "rope theta")?,
        frac: finite(g.frac, "rope fraction")?,
        pairing,
        scaling: g.scale,
        factor: used(g.factor, "rope factor")?,
        low: used(g.low, "rope low")?,
        high: used(g.high, "rope high")?,
        orig: used(g.orig, "rope orig")?,
        aux: if scaled { g.aux } else { 0 },
    })
}

/// The generation recipe of tensor `handle`, of one of `kinds` (the f32 or the BF16 table: the
/// same rotation, so the header does not tell them apart).
fn generated<'a>(packet: &'a Packet<'_>, handle: u16, kinds: [u32; 2]) -> Result<&'a GenTensor> {
    packet
        .generated
        .iter()
        .find(|g| g.tensor == handle as u32 && kinds.contains(&g.kind))
        .ok_or_else(|| "rotary table is not a generated recipe (baked or missing)".into())
}

/// A checkpoint's identity: SHA-256 over its `config.json` and every `*.safetensors` file,
/// by name and full content. An AMD and an NVIDIA emit of the same checkpoint get the same
/// digest.
pub fn checkpoint_digest(dir: &Path) -> Result<String> {
    use std::io::Read;
    let err = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    let mut files = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| err(dir, e))? {
        files.push(e.map_err(|e| err(dir, e))?.path());
    }
    files.retain(|p| {
        p.file_name().is_some_and(|n| n == "config.json")
            || p.extension().is_some_and(|x| x == "safetensors")
    });
    files.sort();
    if !files
        .iter()
        .any(|p| p.extension().is_some_and(|x| x == "safetensors"))
    {
        return Err(format!("{}: no safetensors files", dir.display()));
    }
    let mut all = Sha256::new();
    all.update(b"plow-checkpoint-v1");
    let mut buf = vec![0u8; 8 << 20];
    for p in &files {
        let mut f = std::fs::File::open(p).map_err(|e| err(p, e))?;
        let mut h = Sha256::new();
        loop {
            let n = f.read(&mut buf).map_err(|e| err(p, e))?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        let name = p.file_name().expect("listed file").to_string_lossy();
        all.update((name.len() as u64).to_le_bytes());
        all.update(name.as_bytes());
        all.update(h.finalize());
    }
    Ok(format!("{:x}", all.finalize()))
}

#[cfg(test)]
#[path = "kv_image_tests.rs"]
mod tests;
