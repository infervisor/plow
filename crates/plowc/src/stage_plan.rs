//! Weight-stationary pipeline stage plan: which weights each CPU socket holds L2-resident.
//!
//! A decoder is cut into units in execution order: the per-layer-input projection, then per layer attention
//! (q/k/v/o and their norms), FFN (gate/up/down) and the per-layer-input block, then the tied LM head. Units are
//! packed greedily into stages of `cores x l2_weight_bytes_per_core` BF16 bytes. FFN and head units split by output
//! rows (an FFN piece is a slice of the intermediate dim: gate/up rows plus the matching down columns, so the next
//! piece adds its partial down sum); attention and the per-layer-input block are never split. The plan is
//! therefore whole layers, several layers or part of a layer per socket, as the sizes dictate. Every stage gets a
//! predicted time from the calibrated stage cost (L2 GEMV rate per core, cost per all-core exchange, KV stream
//! rate), which the single-socket stage measurements check.
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
    Head,
}

#[derive(Clone, Debug, Serialize)]
pub struct Unit {
    pub kind: UnitKind,
    pub layer: Option<u32>,
    /// Output rows the unit may be split along (0 = indivisible).
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
}

#[derive(Clone, Debug, Serialize)]
pub struct Cost {
    pub gemv_gbps_per_core: f64,
    pub exchange_us: f64,
    pub kv_gbps: f64,
    pub hop_us: f64,
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
    pub total_us: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Stage {
    pub index: usize,
    pub label: String,
    pub pieces: Vec<Piece>,
    pub bytes: u64,
    pub bytes_per_core: u64,
    pub fill: f64,
    pub exchanges: u32,
    pub kv_bytes_per_step: u64,
    pub kv_replicas_of: Vec<u32>,
    pub pred: Pred,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub stages: usize,
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

/// Units of a Gemma-4 text decoder (`config.json` with or without `text_config`), from checkpoint shapes.
pub fn gemma4_units(shapes: &HashMap<String, Vec<i64>>, config: &Value) -> Result<Vec<Unit>, String> {
    let tc = config.get("text_config").unwrap_or(config);
    let n_layers = tc["num_hidden_layers"].as_u64().ok_or("config: num_hidden_layers")? as u32;
    let shared = tc["num_kv_shared_layers"].as_u64().unwrap_or(0) as u32;
    let window = tc["sliding_window"].as_u64();
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
            kind: UnitKind::PleProj,
            layer: None,
            rows: dim(shapes, &t("per_layer_model_projection.weight"), 0)?,
            row_bytes: hidden * BF16,
            fixed_bytes: numel(shapes, &t("per_layer_projection_norm.weight")).unwrap_or(0) * BF16,
            exchanges: 1,
            kv_row_bytes: 0,
            window: None,
            kv_source: None,
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
        if own_kv {
            for s in ["self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"] {
                attn += numel(shapes, &w(s))?;
            }
        }
        let kv_rows = dim(shapes, &t(&format!("layers.{src}.self_attn.k_proj.weight")), 0)?;
        u.push(Unit {
            kind: UnitKind::Attn,
            layer: Some(l),
            rows: 0,
            row_bytes: 0,
            fixed_bytes: attn * BF16,
            exchanges: 4,
            kv_row_bytes: 2 * kv_rows * BF16,
            window: if types.get(l as usize) == Some(&"sliding_attention") { window } else { None },
            kv_source: (!own_kv).then_some(src),
        });
        let inter = dim(shapes, &w("mlp.gate_proj.weight"), 0)?;
        u.push(Unit {
            kind: UnitKind::Ffn,
            layer: Some(l),
            rows: inter,
            row_bytes: 3 * hidden * BF16,
            fixed_bytes: (numel(shapes, &w("pre_feedforward_layernorm.weight"))? + numel(shapes, &w("post_feedforward_layernorm.weight"))?) * BF16,
            exchanges: 2,
            kv_row_bytes: 0,
            window: None,
            kv_source: None,
        });
        if shapes.contains_key(&w("per_layer_input_gate.weight")) {
            let mut b = 0;
            for s in ["per_layer_input_gate.weight", "per_layer_projection.weight", "post_per_layer_input_norm.weight"] {
                b += numel(shapes, &w(s))?;
            }
            u.push(Unit {
                kind: UnitKind::Ple,
                layer: Some(l),
                rows: 0,
                row_bytes: 0,
                fixed_bytes: b * BF16,
                exchanges: 2,
                kv_row_bytes: 0,
                window: None,
                kv_source: None,
            });
        }
    }
    u.push(Unit {
        kind: UnitKind::Head,
        layer: None,
        rows: dim(shapes, &t("embed_tokens.weight"), 0)?,
        row_bytes: hidden * BF16,
        fixed_bytes: numel(shapes, &t("norm.weight"))? * BF16,
        exchanges: 1,
        kv_row_bytes: 0,
        window: None,
        kv_source: None,
    });
    Ok(u)
}

fn label(p: &Piece) -> String {
    let k = match p.kind {
        UnitKind::PleProj => "ple_proj",
        UnitKind::Attn => "attn",
        UnitKind::Ffn => "ffn",
        UnitKind::Ple => "ple",
        UnitKind::Head => "head",
    };
    let l = p.layer.map(|l| format!("L{l}.")).unwrap_or_default();
    if p.of_rows == 0 || (p.row0 == 0 && p.row1 == p.of_rows) {
        format!("{l}{k}")
    } else {
        format!("{l}{k}[{}:{}]", p.row0, p.row1)
    }
}

fn finish(index: usize, pieces: Vec<Piece>, units: &[Unit], b: &Budget, c: &Cost) -> Stage {
    let bytes: u64 = pieces.iter().map(|p| p.bytes).sum();
    let bpc = bytes.div_ceil(b.cores as u64);
    let exchanges: u32 = pieces.iter().map(|p| units[p.unit].exchanges).sum();
    let mut kv = 0;
    let mut replicas = Vec::new();
    for p in &pieces {
        let u = &units[p.unit];
        let rows = u.window.map_or(b.ctx, |w| w.min(b.ctx));
        kv += u.kv_row_bytes * rows * b.batch as u64;
        if let Some(s) = u.kv_source {
            if !pieces.iter().any(|q| units[q.unit].kind == UnitKind::Attn && units[q.unit].layer == Some(s)) && !replicas.contains(&s) {
                replicas.push(s);
            }
        }
    }
    let gemv_us = bpc as f64 / (c.gemv_gbps_per_core * 1e3);
    let sync_us = exchanges as f64 * c.exchange_us;
    let attn_us = kv as f64 / (c.kv_gbps * 1e3);
    let first = pieces.first().map(label).unwrap_or_default();
    let last = pieces.last().map(label).unwrap_or_default();
    Stage {
        index,
        label: if pieces.len() > 1 { format!("{first} .. {last}") } else { first },
        pieces,
        bytes,
        bytes_per_core: bpc,
        fill: bpc as f64 / b.l2_weight_bytes_per_core as f64,
        exchanges,
        kv_bytes_per_step: kv,
        kv_replicas_of: replicas,
        pred: Pred { gemv_us, sync_us, attn_us, total_us: gemv_us + sync_us + attn_us },
    }
}

/// Greedy in-order packing. A row-splittable unit fills the open stage down to `min_rows_per_core x cores` rows;
/// an indivisible unit that does not fit opens a new stage, and one larger than a whole stage is an error.
pub fn plan(units: &[Unit], b: &Budget, c: &Cost) -> Result<(Vec<Stage>, Summary), String> {
    let cap = b.cores as u64 * b.l2_weight_bytes_per_core;
    let min_rows = b.min_rows_per_core * b.cores as u64;
    let mut stages = Vec::new();
    let mut open: Vec<Piece> = Vec::new();
    let mut used = 0u64;
    for (i, u) in units.iter().enumerate() {
        if u.rows == 0 {
            let need = u.bytes();
            if need > cap {
                return Err(format!("{:?} layer {:?}: {need} B does not fit one stage ({cap} B)", u.kind, u.layer));
            }
            if used + need > cap {
                stages.push(finish(stages.len(), std::mem::take(&mut open), units, b, c));
                used = 0;
            }
            open.push(Piece { unit: i, kind: u.kind, layer: u.layer, row0: 0, row1: 0, of_rows: 0, bytes: need });
            used += need;
            continue;
        }
        let mut r0 = 0;
        while r0 < u.rows {
            let fixed = if r0 == 0 { u.fixed_bytes } else { 0 };
            let left = cap.saturating_sub(used + fixed);
            let fit = (left / u.row_bytes).min(u.rows - r0);
            let fit = if fit < u.rows - r0 { fit / b.min_rows_per_core * b.min_rows_per_core } else { fit };
            if fit < min_rows.min(u.rows - r0) {
                if open.is_empty() {
                    return Err(format!("{:?}: a stage holds fewer than {min_rows} rows", u.kind));
                }
                stages.push(finish(stages.len(), std::mem::take(&mut open), units, b, c));
                used = 0;
                continue;
            }
            let bytes = fixed + fit * u.row_bytes;
            open.push(Piece { unit: i, kind: u.kind, layer: u.layer, row0: r0, row1: r0 + fit, of_rows: u.rows, bytes });
            used += bytes;
            r0 += fit;
        }
    }
    if !open.is_empty() {
        stages.push(finish(stages.len(), open, units, b, c));
    }
    let layers = units.iter().filter_map(|u| u.layer).max().map_or(0, |l| l + 1);
    let bottleneck = stages.iter().map(|s| s.pred.total_us).fold(0.0, f64::max);
    let lat = stages.iter().map(|s| s.pred.total_us).sum::<f64>() + c.hop_us * stages.len().saturating_sub(1) as f64;
    let summary = Summary {
        stages: stages.len(),
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

    fn unit(kind: UnitKind, layer: u32, rows: u64, row_bytes: u64, fixed: u64) -> Unit {
        Unit { kind, layer: Some(layer), rows, row_bytes, fixed_bytes: fixed, exchanges: 2, kv_row_bytes: 0, window: None, kv_source: None }
    }

    #[test]
    fn splits_rows_and_keeps_indivisible_units_whole() {
        // 2 cores x 50 B: attn 60 B whole, ffn 10 rows x 10 B split 40 | 60, next attn 40 B fills stage 1, attn 30 B opens stage 2
        let u = vec![
            unit(UnitKind::Attn, 0, 0, 0, 60),
            unit(UnitKind::Ffn, 0, 10, 10, 0),
            unit(UnitKind::Attn, 1, 0, 0, 40),
            unit(UnitKind::Attn, 2, 0, 0, 30),
        ];
        let b = Budget { cores: 2, l2_weight_bytes_per_core: 50, batch: 1, ctx: 1, min_rows_per_core: 1 };
        let c = Cost { gemv_gbps_per_core: 1.0, exchange_us: 1.0, kv_gbps: 1.0, hop_us: 0.0 };
        let (s, sum) = plan(&u, &b, &c).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!((s[0].pieces[1].row0, s[0].pieces[1].row1), (0, 4));
        assert_eq!((s[1].pieces[0].row0, s[1].pieces[0].row1), (4, 10));
        assert_eq!(s[1].pieces[1].kind, UnitKind::Attn);
        assert_eq!(s[2].pieces[0].layer, Some(2));
        assert!(s.iter().all(|s| s.bytes <= 100));
        assert_eq!(s.iter().map(|s| s.bytes).sum::<u64>(), u.iter().map(Unit::bytes).sum::<u64>());
        assert_eq!(sum.layers, 3);
    }

    #[test]
    fn indivisible_unit_larger_than_a_stage_is_an_error() {
        let u = vec![unit(UnitKind::Attn, 0, 0, 0, 300)];
        let b = Budget { cores: 2, l2_weight_bytes_per_core: 100, batch: 1, ctx: 1, min_rows_per_core: 1 };
        let c = Cost { gemv_gbps_per_core: 1.0, exchange_us: 1.0, kv_gbps: 1.0, hop_us: 0.0 };
        assert!(plan(&u, &b, &c).is_err());
    }

    #[test]
    fn split_pieces_respect_the_row_granularity() {
        let u = vec![unit(UnitKind::Ffn, 0, 1000, 1, 0)];
        let b = Budget { cores: 3, l2_weight_bytes_per_core: 100, batch: 1, ctx: 1, min_rows_per_core: 16 };
        let c = Cost { gemv_gbps_per_core: 1.0, exchange_us: 1.0, kv_gbps: 1.0, hop_us: 0.0 };
        let (s, _) = plan(&u, &b, &c).unwrap();
        for st in &s[..s.len() - 1] {
            assert_eq!((st.pieces[0].row1 - st.pieces[0].row0) % 16, 0);
        }
        assert_eq!(s.last().unwrap().pieces[0].row1, 1000);
    }
}
