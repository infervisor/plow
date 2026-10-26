//! Weight-stationary pipeline stage plan: which weights each CPU socket holds L2-resident.
//!
//! A decoder is cut into units in execution order: the per-layer-input projection, then per layer attention
//! (q/k/v/o and their norms), FFN (gate/up/down), the per-layer-input block and, for MoE layers, the router and
//! the experts, then the tied LM head. A pipeline stage is one socket or a group of sockets that work on the same
//! microbatch at once:
//! - Layers that fit a socket are packed greedily, in order, into stages of `cores x l2_weight_bytes_per_core` BF16
//!   bytes. FFN units split by output rows (a slice of the intermediate dim: gate/up rows plus the matching down
//!   columns, so the next piece adds its partial down sum); attention and the per-layer-input block never split.
//! - `Split::Tp`: a layer that does not fit becomes a tensor-parallel group of S sockets (whole q heads with their KV
//!   heads, replicated when S exceeds the KV heads; a block of FFN rows; o / down partial sums all-reduced across the
//!   group, twice per layer). S is the smallest degree that divides the heads and FFN rows and fits. An MoE layer is a
//!   head group (attention, dense FFN, router) and an expert group (whole experts striped over every core of their
//!   socket). `Split::Pipe` instead cuts such a layer's FFN across consecutive stages.
//! - The LM head is one vocab-parallel group.
//!
//! Every stage gets a predicted time from the calibrated stage cost (L2 GEMV rate per core, cost per all-core
//! exchange, KV stream rate, per extra batch row the activation bytes every core gathers, cross-socket all-reduces),
//! which the single-socket stage measurements check.
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitKind {
    PleProj,
    Attn,
    Ffn,
    Ple,
    Router,
    Experts,
    Head,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Pipe,
    Tp,
}

#[derive(Clone, Debug, Serialize)]
pub struct Unit {
    pub kind: UnitKind,
    pub layer: Option<u32>,
    /// Output rows the unit may be split along (0 = indivisible; experts: the expert count).
    pub rows: u64,
    pub row_bytes: u64,
    /// Bytes not split with the rows (norm vectors; all of an indivisible unit).
    pub fixed_bytes: u64,
    /// All-core exchanges (barrier + broadcast) inside the unit.
    pub exchanges: u32,
    /// KV bytes attention reads per sequence per context token (0 outside attention).
    pub kv_row_bytes: u64,
    /// Attention window in tokens (None = full context).
    pub window: Option<u64>,
    /// KV-shared layer: the layer whose cache it reads (its stage needs a replica).
    pub kv_source: Option<u32>,
    /// Activation bytes every core gathers per batch row: fixed, per split row, and a part that divides by the TP
    /// degree (attention: q and its output).
    pub act_fixed: u64,
    pub act_per_row: u64,
    pub act_split: u64,
    /// Attention: q heads, KV heads, and the k / v / k_norm bytes inside `fixed_bytes`.
    pub heads: u64,
    pub kv_heads: u64,
    pub kv_weight_bytes: u64,
    /// Experts: experts chosen per token.
    pub top_k: u64,
}

impl Unit {
    pub fn bytes(&self) -> u64 {
        self.fixed_bytes + self.rows * self.row_bytes
    }
}

#[derive(Clone, Debug)]
pub struct Budget {
    pub cores: u32,
    pub l2_weight_bytes_per_core: u64,
    pub batch: u32,
    pub ctx: u64,
    /// Smallest split piece in rows per core (AMX tile = 16).
    pub min_rows_per_core: u64,
    pub split: Split,
}

#[derive(Clone, Debug, Serialize)]
pub struct Cost {
    pub gemv_gbps_per_core: f64,
    pub exchange_us: f64,
    pub kv_gbps: f64,
    /// Per-core all-gather rate for the activations of each batch row past the first.
    pub act_gbps: f64,
    pub hop_us: f64,
    /// One all-reduce of a hidden-size partial across a tensor-parallel group.
    pub allreduce_us: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Piece {
    pub unit: usize,
    pub kind: UnitKind,
    pub layer: Option<u32>,
    pub row0: u64,
    pub row1: u64,
    pub of_rows: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pred {
    pub gemv_us: f64,
    pub sync_us: f64,
    pub attn_us: f64,
    pub act_us: f64,
    pub allreduce_us: f64,
    pub total_us: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Stage {
    pub index: usize,
    pub label: String,
    pub pieces: Vec<Piece>,
    /// Sockets working on this stage at once (tensor-parallel, expert or vocab-parallel group).
    pub sockets: u32,
    /// Weight bytes per socket.
    pub bytes: u64,
    pub bytes_per_core: u64,
    pub fill: f64,
    pub exchanges: u32,
    pub allreduces: u32,
    pub kv_bytes_per_step: u64,
    pub kv_replicas_of: Vec<u32>,
    pub pred: Pred,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub stages: usize,
    pub sockets: u32,
    pub layers: u32,
    pub layers_per_stage: f64,
    pub weight_bytes: u64,
    pub bottleneck_us: f64,
    pub token_latency_us: f64,
    pub pipeline_tokens_per_s: f64,
    pub sequences_in_flight: u64,
}

const BF16: u64 = 2;

fn numel(shapes: &HashMap<String, Vec<i64>>, name: &str) -> Result<u64, String> {
    shapes
        .get(name)
        .map(|s| s.iter().product::<i64>() as u64)
        .ok_or_else(|| format!("checkpoint has no tensor {name}"))
}

fn dim(shapes: &HashMap<String, Vec<i64>>, name: &str, i: usize) -> Result<u64, String> {
    shapes
        .get(name)
        .and_then(|s| s.get(i))
        .map(|&d| d as u64)
        .ok_or_else(|| format!("checkpoint has no tensor {name} (dim {i})"))
}

fn unit(kind: UnitKind, layer: Option<u32>) -> Unit {
    Unit {
        kind,
        layer,
        rows: 0,
        row_bytes: 0,
        fixed_bytes: 0,
        exchanges: 0,
        kv_row_bytes: 0,
        window: None,
        kv_source: None,
        act_fixed: 0,
        act_per_row: 0,
        act_split: 0,
        heads: 0,
        kv_heads: 0,
        kv_weight_bytes: 0,
        top_k: 0,
    }
}

/// Units of a Gemma-4 text decoder (`config.json` with or without `text_config`), from checkpoint shapes.
pub fn gemma4_units(shapes: &HashMap<String, Vec<i64>>, config: &Value) -> Result<Vec<Unit>, String> {
    let tc = config.get("text_config").unwrap_or(config);
    let n_layers = tc["num_hidden_layers"].as_u64().ok_or("config: num_hidden_layers")? as u32;
    let shared = tc["num_kv_shared_layers"].as_u64().unwrap_or(0) as u32;
    let window = tc["sliding_window"].as_u64();
    let top_k = tc["top_k_experts"].as_u64().unwrap_or(0);
    let types: Vec<&str> = tc["layer_types"]
        .as_array()
        .ok_or("config: layer_types")?
        .iter()
        .map(|t| t.as_str().unwrap_or(""))
        .collect();
    let p = ["model.language_model.", "model."]
        .into_iter()
        .find(|p| shapes.contains_key(&format!("{p}embed_tokens.weight")))
        .ok_or("checkpoint has no embed_tokens.weight")?;
    let t = |s: &str| format!("{p}{s}");
    let hidden = dim(shapes, &t("embed_tokens.weight"), 1)?;
    let first_shared = n_layers - shared;
    let mut u = Vec::new();
    if shapes.contains_key(&t("per_layer_model_projection.weight")) {
        u.push(Unit {
            rows: dim(shapes, &t("per_layer_model_projection.weight"), 0)?,
            row_bytes: hidden * BF16,
            fixed_bytes: numel(shapes, &t("per_layer_projection_norm.weight")).unwrap_or(0) * BF16,
            exchanges: 1,
            // each core reads only its own PLE rows of this output; it travels with the hop, not an all-gather
            ..unit(UnitKind::PleProj, None)
        });
    }
    for l in 0..n_layers {
        let w = |s: &str| t(&format!("layers.{l}.{s}"));
        let own_kv = l < first_shared;
        // a KV-shared layer reads the cache of the last non-shared layer of its attention type
        let src = if own_kv {
            l
        } else {
            (0..first_shared)
                .rev()
                .find(|&s| types.get(s as usize) == types.get(l as usize))
                .ok_or_else(|| format!("layer {l}: no KV source layer"))?
        };
        let mut attn = 0;
        for s in ["self_attn.q_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"] {
            attn += numel(shapes, &w(s))?;
        }
        let mut kvw = 0;
        if own_kv {
            // attention_k_eq_v layers have no v_proj (V is the K projection)
            for s in ["self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"] {
                kvw += numel(shapes, &w(s)).unwrap_or(0);
            }
        }
        let head_dim = dim(shapes, &w("self_attn.q_norm.weight"), 0)?;
        let kv_rows = dim(shapes, &t(&format!("layers.{src}.self_attn.k_proj.weight")), 0)?;
        let q_rows = dim(shapes, &w("self_attn.q_proj.weight"), 0)?;
        u.push(Unit {
            fixed_bytes: (attn + kvw) * BF16,
            exchanges: 4,
            kv_row_bytes: 2 * kv_rows * BF16,
            window: if types.get(l as usize) == Some(&"sliding_attention") { window } else { None },
            kv_source: (!own_kv).then_some(src),
            // q (FP32), attention output (BF16), o (FP32)
            act_fixed: hidden * 4,
            act_split: q_rows * 4 + q_rows * BF16,
            heads: q_rows / head_dim,
            kv_heads: kv_rows / head_dim,
            kv_weight_bytes: kvw * BF16,
            ..unit(UnitKind::Attn, Some(l))
        });
        let inter = dim(shapes, &w("mlp.gate_proj.weight"), 0)?;
        u.push(Unit {
            rows: inter,
            row_bytes: 3 * hidden * BF16,
            fixed_bytes: (numel(shapes, &w("pre_feedforward_layernorm.weight"))? + numel(shapes, &w("post_feedforward_layernorm.weight"))?) * BF16,
            exchanges: 2,
            // gelu(gate) * up (BF16) per intermediate row, down output (FP32)
            act_fixed: hidden * 4,
            act_per_row: BF16,
            ..unit(UnitKind::Ffn, Some(l))
        });
        if shapes.contains_key(&w("per_layer_input_gate.weight")) {
            let mut b = 0;
            for s in ["per_layer_input_gate.weight", "per_layer_projection.weight", "post_per_layer_input_norm.weight"] {
                b += numel(shapes, &w(s))?;
            }
            u.push(Unit {
                fixed_bytes: b * BF16,
                exchanges: 2,
                act_fixed: dim(shapes, &w("per_layer_input_gate.weight"), 0)? * BF16 + hidden * 4,
                ..unit(UnitKind::Ple, Some(l))
            });
        }
        if shapes.contains_key(&w("experts.gate_up_proj")) {
            let mut b = 0;
            for s in ["router.proj.weight", "router.scale", "router.per_expert_scale", "pre_feedforward_layernorm_2.weight", "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight"] {
                b += numel(shapes, &w(s))?;
            }
            // router scores (FP32) on the head group; the experts get x and the routing with the dispatch hop
            u.push(Unit { fixed_bytes: b * BF16, exchanges: 1, act_fixed: dim(shapes, &w("router.proj.weight"), 0)? * 4, ..unit(UnitKind::Router, Some(l)) });
            let n = dim(shapes, &w("experts.gate_up_proj"), 0)?;
            let e_inter = dim(shapes, &w("experts.down_proj"), 2)?;
            u.push(Unit {
                rows: n,
                row_bytes: (numel(shapes, &w("experts.gate_up_proj"))? + numel(shapes, &w("experts.down_proj"))?) / n * BF16,
                exchanges: 2,
                // per (token, expert) pair every core gathers gelu(gate) * up (BF16); the weighted sum goes to the
                // sending core only
                act_per_row: e_inter * BF16,
                top_k,
                ..unit(UnitKind::Experts, Some(l))
            });
        }
    }
    u.push(Unit {
        rows: dim(shapes, &t("embed_tokens.weight"), 0)?,
        row_bytes: hidden * BF16,
        fixed_bytes: numel(shapes, &t("norm.weight"))? * BF16,
        exchanges: 1,
        // logits stay on their socket (top-k reduce); only the normed hidden state arrives
        act_fixed: hidden * 4,
        ..unit(UnitKind::Head, None)
    });
    Ok(u)
}

fn label(p: &Piece) -> String {
    let k = match p.kind {
        UnitKind::PleProj => "ple_proj",
        UnitKind::Attn => "attn",
        UnitKind::Ffn => "ffn",
        UnitKind::Ple => "ple",
        UnitKind::Router => "router",
        UnitKind::Experts => "experts",
        UnitKind::Head => "head",
    };
    let l = p.layer.map(|l| format!("L{l}.")).unwrap_or_default();
    if p.of_rows == 0 || (p.row0 == 0 && p.row1 == p.of_rows) {
        format!("{l}{k}")
    } else {
        format!("{l}{k}[{}:{}]", p.row0, p.row1)
    }
}

fn whole(units: &[Unit], i: usize) -> Piece {
    let u = &units[i];
    Piece { unit: i, kind: u.kind, layer: u.layer, row0: 0, row1: u.rows, of_rows: u.rows, bytes: u.bytes() }
}

/// Per-socket load of a stage, before the time model.
struct Load {
    bytes: u64,
    exchanges: u32,
    allreduces: u32,
    kv: u64,
    act: u64,
}

fn pred(l: &Load, b: &Budget, c: &Cost) -> (u64, Pred) {
    let bpc = l.bytes.div_ceil(b.cores as u64);
    let gemv_us = bpc as f64 / (c.gemv_gbps_per_core * 1e3);
    let sync_us = l.exchanges as f64 * c.exchange_us;
    let attn_us = l.kv as f64 / (c.kv_gbps * 1e3);
    let act_us = b.batch.saturating_sub(1) as f64 * l.act as f64 / (c.act_gbps * 1e3);
    let allreduce_us = l.allreduces as f64 * c.allreduce_us;
    (bpc, Pred { gemv_us, sync_us, attn_us, act_us, allreduce_us, total_us: gemv_us + sync_us + attn_us + act_us + allreduce_us })
}

fn kv_rows(u: &Unit, b: &Budget) -> u64 {
    u.window.map_or(b.ctx, |w| w.min(b.ctx))
}

fn make(index: usize, label: String, pieces: Vec<Piece>, sockets: u32, l: Load, replicas: Vec<u32>, b: &Budget, c: &Cost) -> Stage {
    let (bpc, p) = pred(&l, b, c);
    Stage {
        index,
        label,
        pieces,
        sockets,
        bytes: l.bytes,
        bytes_per_core: bpc,
        fill: bpc as f64 / b.l2_weight_bytes_per_core as f64,
        exchanges: l.exchanges,
        allreduces: l.allreduces,
        kv_bytes_per_step: l.kv,
        kv_replicas_of: replicas,
        pred: p,
    }
}

/// One socket holding whole units and row pieces.
fn finish(index: usize, pieces: Vec<Piece>, units: &[Unit], b: &Budget, c: &Cost) -> Stage {
    let mut l = Load { bytes: 0, exchanges: 0, allreduces: 0, kv: 0, act: 0 };
    let mut replicas = Vec::new();
    for p in &pieces {
        let u = &units[p.unit];
        l.bytes += p.bytes;
        l.exchanges += u.exchanges;
        l.act += u.act_fixed + u.act_split + u.act_per_row * (if u.rows == 0 { 0 } else { p.row1 - p.row0 });
        l.kv += u.kv_row_bytes * kv_rows(u, b) * b.batch as u64;
        if let Some(s) = u.kv_source {
            if !pieces.iter().any(|q| units[q.unit].kind == UnitKind::Attn && units[q.unit].layer == Some(s)) && !replicas.contains(&s) {
                replicas.push(s);
            }
        }
    }
    let first = pieces.first().map(label).unwrap_or_default();
    let last = pieces.last().map(label).unwrap_or_default();
    let name = if pieces.len() > 1 { format!("{first} .. {last}") } else { first };
    make(index, name, pieces, 1, l, replicas, b, c)
}

/// Per-socket load of dense units `idx` split S ways: whole q heads with their KV heads (one replicated head per
/// socket when S exceeds the KV heads), FFN rows / S; norms, router and the per-layer-input block on every socket.
fn tp_load(units: &[Unit], idx: &[usize], s: u64, b: &Budget) -> Option<Load> {
    let mut l = Load { bytes: 0, exchanges: 0, allreduces: if s > 1 { 2 } else { 0 }, kv: 0, act: 0 };
    for &i in idx {
        let u = &units[i];
        l.exchanges += u.exchanges;
        l.act += u.act_fixed;
        match u.kind {
            UnitKind::Attn => {
                if u.heads % s != 0 || (u.kv_heads > 0 && u.kv_heads % s != 0 && s % u.kv_heads != 0) {
                    return None;
                }
                let kvh = if u.kv_heads >= s { u.kv_heads / s } else { 1 };
                let kv_frac = |x: u64| if u.kv_heads == 0 { 0 } else { x * kvh / u.kv_heads };
                l.bytes += (u.fixed_bytes - u.kv_weight_bytes) / s + kv_frac(u.kv_weight_bytes);
                l.kv += kv_frac(u.kv_row_bytes) * kv_rows(u, b) * b.batch as u64;
                l.act += u.act_split / s;
            }
            UnitKind::Ffn => {
                if u.rows % (s * 16) != 0 {
                    return None;
                }
                l.bytes += u.fixed_bytes + u.rows / s * u.row_bytes;
                l.act += u.act_per_row * u.rows / s;
            }
            _ => l.bytes += u.bytes(),
        }
    }
    Some(l)
}

struct Packer<'a> {
    units: &'a [Unit],
    b: &'a Budget,
    c: &'a Cost,
    cap: u64,
    stages: Vec<Stage>,
    open: Vec<Piece>,
    used: u64,
}

impl Packer<'_> {
    fn flush(&mut self) {
        if !self.open.is_empty() {
            let s = finish(self.stages.len(), std::mem::take(&mut self.open), self.units, self.b, self.c);
            self.stages.push(s);
        }
        self.used = 0;
    }

    /// Greedy in-order packing of one unit: a row-splittable unit fills the open stage down to
    /// `min_rows_per_core x cores` rows; an indivisible unit that does not fit opens a new stage.
    fn pack(&mut self, i: usize) -> Result<(), String> {
        let (u, b, cap) = (&self.units[i], self.b, self.cap);
        let min_rows = b.min_rows_per_core * b.cores as u64;
        if u.rows == 0 {
            let need = u.bytes();
            if need > cap {
                return Err(format!("{:?} layer {:?}: {need} B does not fit one stage ({cap} B)", u.kind, u.layer));
            }
            if self.used + need > cap {
                self.flush();
            }
            self.open.push(whole(self.units, i));
            self.used += need;
            return Ok(());
        }
        let mut r0 = 0;
        while r0 < u.rows {
            let fixed = if r0 == 0 { u.fixed_bytes } else { 0 };
            let left = cap.saturating_sub(self.used + fixed);
            let fit = (left / u.row_bytes).min(u.rows - r0);
            let fit = if fit < u.rows - r0 { fit / b.min_rows_per_core * b.min_rows_per_core } else { fit };
            if fit < min_rows.min(u.rows - r0) {
                if self.open.is_empty() {
                    return Err(format!("{:?}: a stage holds fewer than {min_rows} rows", u.kind));
                }
                self.flush();
                continue;
            }
            let bytes = fixed + fit * u.row_bytes;
            self.open.push(Piece { unit: i, kind: u.kind, layer: u.layer, row0: r0, row1: r0 + fit, of_rows: u.rows, bytes });
            self.used += bytes;
            r0 += fit;
        }
        Ok(())
    }

    fn push(&mut self, label: String, pieces: Vec<Piece>, sockets: u32, l: Load) {
        let s = make(self.stages.len(), label, pieces, sockets, l, Vec::new(), self.b, self.c);
        self.stages.push(s);
    }

    /// A layer that does not fit one socket: the smallest valid tensor-parallel degree for its dense units, then
    /// (MoE) the expert group.
    fn tp_layer(&mut self, layer: u32, idx: &[usize]) -> Result<(), String> {
        self.flush();
        let dense: Vec<usize> = idx.iter().copied().filter(|&i| self.units[i].kind != UnitKind::Experts).collect();
        let heads = dense.iter().map(|&i| self.units[i].heads).max().unwrap_or(1).max(1);
        let (s, l) = (1..=heads)
            .find_map(|s| tp_load(self.units, &dense, s, self.b).filter(|l| l.bytes <= self.cap).map(|l| (s, l)))
            .ok_or_else(|| format!("layer {layer}: no tensor-parallel degree fits {} B per socket", self.cap))?;
        let pieces = dense.iter().map(|&i| whole(self.units, i)).collect();
        let name = if s > 1 { format!("L{layer} tp{s}") } else { format!("L{layer}") };
        self.push(name, pieces, s as u32, l);
        if let Some(&e) = idx.iter().find(|&&i| self.units[i].kind == UnitKind::Experts) {
            self.experts(layer, e)?;
        }
        Ok(())
    }

    /// Whole experts per socket, each striped over all cores; a step streams the experts its tokens picked
    /// (expected distinct experts of `batch` tokens choosing `top_k` uniformly).
    fn experts(&mut self, layer: u32, i: usize) -> Result<(), String> {
        let (u, b) = (&self.units[i], self.b);
        let per = self.cap / u.row_bytes;
        if per == 0 {
            return Err(format!("layer {layer}: one expert ({} B) does not fit a socket", u.row_bytes));
        }
        let sockets = u.rows.div_ceil(per);
        let held = u.rows as f64 / sockets as f64;
        let p = 1.0 - (1.0 - u.top_k as f64 / u.rows as f64).powi(b.batch as i32);
        let active = held * p;
        let picks = b.batch as f64 * u.top_k as f64 / sockets as f64; // (token, expert) pairs per socket
        let l = Load {
            bytes: (active * u.row_bytes as f64) as u64,
            exchanges: u.exchanges,
            allreduces: 0,
            kv: 0,
            act: u.act_fixed + (picks / b.batch.max(1) as f64 * u.act_per_row as f64) as u64,
        };
        let pieces = vec![whole(self.units, i)];
        self.push(format!("L{layer}.experts x{sockets} ({held:.1} per socket, {active:.1} active)"), pieces, sockets as u32, l);
        Ok(())
    }

    fn head(&mut self, i: usize) {
        self.flush();
        let u = &self.units[i];
        let sockets = u.bytes().div_ceil(self.cap);
        let l = Load {
            bytes: u.bytes().div_ceil(sockets),
            exchanges: u.exchanges,
            allreduces: if sockets > 1 { 1 } else { 0 }, // top-k merge across the vocab slices
            kv: 0,
            act: u.act_fixed,
        };
        let pieces = vec![whole(self.units, i)];
        self.push(if sockets > 1 { format!("head x{sockets}") } else { "head".into() }, pieces, sockets as u32, l);
    }
}

pub fn plan(units: &[Unit], b: &Budget, c: &Cost) -> Result<(Vec<Stage>, Summary), String> {
    let mut pk = Packer { units, b, c, cap: b.cores as u64 * b.l2_weight_bytes_per_core, stages: Vec::new(), open: Vec::new(), used: 0 };
    let mut i = 0;
    while i < units.len() {
        let u = &units[i];
        if u.kind == UnitKind::Head {
            pk.head(i);
            i += 1;
            continue;
        }
        let Some(layer) = u.layer else {
            pk.pack(i)?;
            i += 1;
            continue;
        };
        let j = (i..units.len()).find(|&j| units[j].layer != Some(layer)).unwrap_or(units.len());
        let idx: Vec<usize> = (i..j).collect();
        let moe = idx.iter().any(|&k| units[k].kind == UnitKind::Experts);
        let bytes: u64 = idx.iter().map(|&k| units[k].bytes()).sum();
        match b.split {
            Split::Tp if moe || bytes > pk.cap => pk.tp_layer(layer, &idx)?,
            Split::Pipe if moe => return Err("MoE layers need --split tp".into()),
            _ => {
                for k in idx {
                    pk.pack(k)?;
                }
            }
        }
        i = j;
    }
    pk.flush();
    let stages = pk.stages;
    let layers = units.iter().filter_map(|u| u.layer).max().map_or(0, |l| l + 1);
    let bottleneck = stages.iter().map(|s| s.pred.total_us).fold(0.0, f64::max);
    let lat = stages.iter().map(|s| s.pred.total_us).sum::<f64>() + c.hop_us * stages.len().saturating_sub(1) as f64;
    let summary = Summary {
        stages: stages.len(),
        sockets: stages.iter().map(|s| s.sockets).sum(),
        layers,
        layers_per_stage: layers as f64 / stages.len().max(1) as f64,
        weight_bytes: units.iter().map(Unit::bytes).sum(),
        bottleneck_us: bottleneck,
        token_latency_us: lat,
        // one microbatch of `batch` sequences per stage keeps every stage busy
        pipeline_tokens_per_s: if bottleneck > 0.0 { b.batch as f64 * 1e6 / (bottleneck + c.hop_us) } else { 0.0 },
        sequences_in_flight: b.batch as u64 * stages.len() as u64,
    };
    Ok((stages, summary))
}

/// L2 bytes per core the pseudo-lock driver can lock (`/sys/class/misc/pseudo_lock/caps`: `... l2 ... lock <cbm> <n> B/core`).
pub fn driver_l2_lock_bytes() -> Option<u64> {
    let caps = std::fs::read_to_string("/sys/class/misc/pseudo_lock/caps").ok()?;
    let l2 = caps.split(" l2 ").nth(1)?;
    let mut it = l2.split_whitespace();
    while let Some(w) = it.next() {
        if w == "lock" {
            it.next()?;
            return it.next()?.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(kind: UnitKind, layer: u32, rows: u64, row_bytes: u64, fixed: u64) -> Unit {
        Unit { rows, row_bytes, fixed_bytes: fixed, exchanges: 2, ..unit(kind, Some(layer)) }
    }

    fn attn(layer: u32, fixed: u64, heads: u64, kv_heads: u64, kv_weight: u64) -> Unit {
        Unit { heads, kv_heads, kv_weight_bytes: kv_weight, ..u(UnitKind::Attn, layer, 0, 0, fixed) }
    }

    fn cost() -> Cost {
        Cost { gemv_gbps_per_core: 1.0, exchange_us: 1.0, kv_gbps: 1.0, act_gbps: 1.0, hop_us: 0.0, allreduce_us: 1.0 }
    }

    fn budget(cores: u32, per_core: u64, min_rows: u64, split: Split) -> Budget {
        Budget { cores, l2_weight_bytes_per_core: per_core, batch: 1, ctx: 1, min_rows_per_core: min_rows, split }
    }

    #[test]
    fn splits_rows_and_keeps_indivisible_units_whole() {
        // 2 cores x 50 B: attn 60 B whole, ffn 10 rows x 10 B split 40 | 60, next attn 40 B fills stage 1, attn 30 B opens stage 2
        let units = vec![
            u(UnitKind::Attn, 0, 0, 0, 60),
            u(UnitKind::Ffn, 0, 10, 10, 0),
            u(UnitKind::Attn, 1, 0, 0, 40),
            u(UnitKind::Attn, 2, 0, 0, 30),
        ];
        let (s, sum) = plan(&units, &budget(2, 50, 1, Split::Pipe), &cost()).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!((s[0].pieces[1].row0, s[0].pieces[1].row1), (0, 4));
        assert_eq!((s[1].pieces[0].row0, s[1].pieces[0].row1), (4, 10));
        assert_eq!(s[1].pieces[1].kind, UnitKind::Attn);
        assert_eq!(s[2].pieces[0].layer, Some(2));
        assert!(s.iter().all(|s| s.bytes <= 100));
        assert_eq!(s.iter().map(|s| s.bytes).sum::<u64>(), units.iter().map(Unit::bytes).sum::<u64>());
        assert_eq!(sum.layers, 3);
    }

    #[test]
    fn indivisible_unit_larger_than_a_stage_is_an_error() {
        let units = vec![u(UnitKind::Attn, 0, 0, 0, 300)];
        assert!(plan(&units, &budget(2, 100, 1, Split::Pipe), &cost()).is_err());
    }

    #[test]
    fn split_pieces_respect_the_row_granularity() {
        let units = vec![u(UnitKind::Ffn, 0, 1000, 1, 0)];
        let (s, _) = plan(&units, &budget(3, 100, 16, Split::Pipe), &cost()).unwrap();
        for st in &s[..s.len() - 1] {
            assert_eq!((st.pieces[0].row1 - st.pieces[0].row0) % 16, 0);
        }
        assert_eq!(s.last().unwrap().pieces[0].row1, 1000);
    }

    #[test]
    fn oversized_layer_becomes_the_smallest_fitting_tp_group() {
        // 1 core x 100 B: attn 120 B (8 heads, 2 KV heads, 40 B of k/v), ffn 64 rows x 2 B. S=2: 40 + 20 + 64 = 124;
        // S=4: 20 + 20 (one replicated KV head = 20 B) + 32 = 72.
        let units = vec![attn(0, 120, 8, 2, 40), u(UnitKind::Ffn, 0, 64, 2, 0)];
        let (s, sum) = plan(&units, &budget(1, 100, 1, Split::Tp), &cost()).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sockets, 4);
        assert_eq!(s[0].bytes, 72);
        assert_eq!(s[0].allreduces, 2);
        assert_eq!(sum.sockets, 4);
    }

    #[test]
    fn experts_take_whole_sockets_and_head_is_vocab_parallel() {
        // 1 core x 100 B: attn 50 B + ffn 16 rows x 1 B fit; 8 experts x 30 B -> 3 per socket, 3 sockets; head 250 B -> 3
        let mut e = u(UnitKind::Experts, 0, 8, 30, 0);
        e.top_k = 2;
        let units = vec![attn(0, 50, 2, 1, 10), u(UnitKind::Ffn, 0, 16, 1, 0), e, u(UnitKind::Head, 0, 250, 1, 0)];
        let mut units = units;
        units[3].layer = None;
        let (s, sum) = plan(&units, &budget(1, 100, 1, Split::Tp), &cost()).unwrap();
        assert_eq!(s.iter().map(|s| s.sockets).collect::<Vec<_>>(), vec![1, 3, 3]);
        assert_eq!(sum.sockets, 7);
        assert!(plan(&units, &budget(1, 100, 1, Split::Pipe), &cost()).is_err());
    }
}
