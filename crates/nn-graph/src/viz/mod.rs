//! Network visualization: a serializable model bundle, a self-contained HTML viewer, a pipeline
//! SVG and an asset dumper.
//!
//! The viewer loads cytoscape/dagre from the jsdelivr CDN (not vendored), so it needs network
//! access. Large graphs are never laid out whole on load: the overview collapses repeated blocks
//! and the detail view shows one block at a time.

pub mod dump;
pub mod html;
pub mod svg;
pub mod types;

pub use dump::dump_assets;
pub use html::render_html;
pub use svg::bundle_to_svg;
pub use types::{
    ModelModality, OpCategory, VizBlock, VizGraph, VizModelBundle, VizNetwork, VizNode, VizStage,
    VizSummary, VizTensor,
};

use crate::graph::{Graph, NodeId, Origin, TensorId};
use crate::op::Op;
use std::collections::HashMap;

pub fn op_to_category(op: &Op) -> OpCategory {
    match op {
        Op::Linear { .. } | Op::MatMul | Op::Embedding => OpCategory::Linear,
        Op::Attention { .. }
        | Op::LinearAttention { .. }
        | Op::DsaAttention { .. }
        | Op::Rope { .. }
        | Op::Softmax { .. } => OpCategory::Attention,
        Op::Conv1d { .. }
        | Op::Conv1dDepthwise { .. }
        | Op::ConvTranspose1d { .. }
        | Op::Conv2d { .. }
        | Op::Conv3d { .. } => OpCategory::Convolution,
        Op::RmsNorm { .. }
        | Op::RmsNormZeroCentered { .. }
        | Op::RmsNormInputScaled { .. }
        | Op::LayerNorm { .. }
        | Op::GroupNorm { .. } => OpCategory::Normalization,
        Op::Act(_)
        | Op::SituGlu { .. }
        | Op::ResidualGate { .. }
        | Op::Elementwise(_)
        | Op::Scale(_) => OpCategory::Elementwise,
        Op::MoeRouter { .. } | Op::MoeExperts { .. } | Op::DsaIndexer { .. } => OpCategory::Routing,
        Op::Reshape { .. }
        | Op::Transpose { .. }
        | Op::Broadcast { .. }
        | Op::Concat { .. }
        | Op::Slice { .. }
        | Op::Reduce { .. }
        | Op::BlockResidual { .. } => OpCategory::Layout,
    }
}

/// Per tensor, how many checkpoint tensors it stands for. A weight under `….experts.0.…` with
/// no `….experts.1.…` sibling is the representative routed expert of a MoE layer: it counts
/// once per expert of the router in the same block.
fn expert_copies(g: &Graph) -> Vec<u32> {
    let mut copies = vec![1u32; g.tensors.len()];
    let names: std::collections::HashSet<&str> =
        g.tensors.iter().filter_map(|t| t.name.as_deref()).collect();
    let mut experts_in_block: HashMap<u32, u32> = HashMap::new();
    for n in &g.nodes {
        if let (Op::MoeRouter { num_experts, .. }, Some(b)) = (&n.op, n.block) {
            experts_in_block.entry(b).or_insert(*num_experts);
        }
    }
    for n in &g.nodes {
        let Some(&experts) = n.block.and_then(|b| experts_in_block.get(&b)) else {
            continue;
        };
        for t in &n.inputs {
            let tensor = &g.tensors[t.0 as usize];
            let Some(name) = tensor.name.as_deref() else {
                continue;
            };
            if matches!(tensor.origin, Origin::Weight)
                && name.contains(".experts.0.")
                && !names.contains(name.replacen(".experts.0.", ".experts.1.", 1).as_str())
            {
                copies[t.0 as usize] = experts;
            }
        }
    }
    copies
}

pub fn graph_to_viz_graph(g: &Graph) -> VizGraph {
    let mut total_params = 0u64;
    let mut total_weight_bytes = 0u64;
    let mut category_counts = HashMap::new();
    let mut seen_weights = std::collections::HashSet::new();

    let copies = expert_copies(g);
    let tensors: Vec<VizTensor> = g
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let shape = t
                .shape
                .as_ref()
                .map(|s| s.display_with(&g.syms))
                .unwrap_or_else(|| "?".into());
            let elements = t.shape.as_ref().and_then(|s| {
                let mut prod = 1i64;
                for d in s.dims() {
                    prod = prod.checked_mul(d.as_static()?)?;
                }
                u64::try_from(prod).ok()
            });
            let n = copies[i];
            let bytes = elements.map(|e| t.dtype.tile_bytes(e) * n as u64);
            let elements = elements.map(|e| e * n as u64);
            // Tied weights (e.g. embed_tokens reused as lm_head) are one checkpoint tensor.
            if matches!(t.origin, Origin::Weight)
                && t.name.as_deref().is_none_or(|n| seen_weights.insert(n))
            {
                total_params += elements.unwrap_or(0);
                total_weight_bytes += bytes.unwrap_or(0);
            }
            VizTensor {
                id: TensorId(i as u32),
                name: t.name.clone().unwrap_or_default(),
                dtype: t.dtype,
                shape,
                elements,
                bytes,
                origin: t.origin.clone(),
                copies: n,
            }
        })
        .collect();

    let nodes: Vec<VizNode> = g
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let category = op_to_category(&n.op);
            *category_counts.entry(category).or_insert(0) += 1;
            let (mut param_count, mut weight_bytes) = (0, 0);
            for t in n.inputs.iter().map(|t| &tensors[t.0 as usize]) {
                if matches!(t.origin, Origin::Weight) {
                    param_count += t.elements.unwrap_or(0);
                    weight_bytes += t.bytes.unwrap_or(0);
                }
            }
            VizNode {
                id: NodeId(i as u32),
                op: n.op.name(),
                category,
                detail: n.op.clone(),
                inputs: n.inputs.clone(),
                output: n.output,
                block: n.block.and_then(|b| g.block_label(b)).map(String::from),
                stage: None,
                param_count,
                weight_bytes,
            }
        })
        .collect();

    let (tensors, nodes, inputs, outputs) =
        collapse_unread_weights(tensors, nodes, &g.inputs, &g.outputs);

    let mut blocks: Vec<VizBlock> = g
        .blocks
        .iter()
        .map(|label| VizBlock {
            id: label.clone(),
            stage: None,
            node_ids: Vec::new(),
            param_count: 0,
            weight_bytes: 0,
        })
        .collect();
    let block_index: HashMap<&str, usize> = g
        .blocks
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();
    for n in &nodes {
        if let Some(&b) = n.block.as_deref().and_then(|l| block_index.get(l)) {
            blocks[b].node_ids.push(n.id);
            blocks[b].param_count += n.param_count;
            blocks[b].weight_bytes += n.weight_bytes;
        }
    }
    // Weights declared for the checkpoint but read by no op (e.g. every routed expert when the
    // graph runs one) belong to the block their name sits under.
    let mut used = vec![false; tensors.len()];
    for n in &nodes {
        for t in &n.inputs {
            used[t.0 as usize] = true;
        }
    }
    for t in tensors
        .iter()
        .filter(|t| matches!(t.origin, Origin::Weight))
    {
        if !used[t.id.0 as usize] {
            if let Some(b) = owning_block(&t.name, &block_index) {
                blocks[b].param_count += t.elements.unwrap_or(0);
                blocks[b].weight_bytes += t.bytes.unwrap_or(0);
            }
        }
    }

    let summary = VizSummary {
        total_params,
        total_weight_bytes,
        num_blocks: blocks.len() as u32,
        category_counts,
        total_nodes: nodes.len(),
        total_tensors: tensors.len(),
    };
    VizGraph {
        tensors,
        nodes,
        blocks,
        inputs,
        outputs,
        summary,
    }
}

/// Weights no op reads (every routed expert of a graph that runs one per layer) are counted,
/// never drawn. Fold each module pattern into one entry, `….experts.*.gate_proj.weight`, with
/// summed elements and bytes, so a 256-expert model does not ship 100k tensors to the viewer.
fn collapse_unread_weights(
    tensors: Vec<VizTensor>,
    mut nodes: Vec<VizNode>,
    inputs: &[TensorId],
    outputs: &[TensorId],
) -> (Vec<VizTensor>, Vec<VizNode>, Vec<TensorId>, Vec<TensorId>) {
    let mut read = vec![false; tensors.len()];
    for t in nodes
        .iter()
        .flat_map(|n| &n.inputs)
        .chain(inputs)
        .chain(outputs)
    {
        read[t.0 as usize] = true;
    }
    let mut groups: HashMap<String, usize> = HashMap::new();
    let mut remap = vec![0u32; tensors.len()];
    let mut out: Vec<VizTensor> = Vec::with_capacity(tensors.len());
    for (i, mut t) in tensors.into_iter().enumerate() {
        let pattern = (!read[i] && matches!(t.origin, Origin::Weight))
            .then(|| fold_indices(&t.name))
            .filter(|p| *p != t.name);
        let Some(pattern) = pattern else {
            remap[i] = out.len() as u32;
            out.push(t);
            continue;
        };
        if let Some(&j) = groups.get(&pattern) {
            let g = &mut out[j];
            g.copies += t.copies;
            g.elements = g.elements.zip(t.elements).map(|(a, b)| a + b);
            g.bytes = g.bytes.zip(t.bytes).map(|(a, b)| a + b);
            remap[i] = j as u32;
        } else {
            groups.insert(pattern.clone(), out.len());
            t.name = pattern;
            remap[i] = out.len() as u32;
            out.push(t);
        }
    }
    for (i, t) in out.iter_mut().enumerate() {
        t.id = TensorId(i as u32);
    }
    let map = |t: &TensorId| TensorId(remap[t.0 as usize]);
    for n in &mut nodes {
        n.inputs = n.inputs.iter().map(map).collect();
        n.output = map(&n.output);
    }
    (
        out,
        nodes,
        inputs.iter().map(map).collect(),
        outputs.iter().map(map).collect(),
    )
}

/// Every numeric segment after the first becomes `*`: `l.5.experts.7.w` -> `l.5.experts.*.w`.
fn fold_indices(name: &str) -> String {
    let mut seen = false;
    name.split('.')
        .map(|p| {
            if p.parse::<u32>().is_ok() && std::mem::replace(&mut seen, true) {
                "*"
            } else {
                p
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

struct StageDef {
    id: &'static str,
    name: &'static str,
    description: String,
}

fn stage(id: &'static str, name: &'static str, description: impl Into<String>) -> StageDef {
    StageDef {
        id,
        name,
        description: description.into(),
    }
}

/// Assign every op of `g` to a stage and derive the stages' params, block counts and boundary
/// shapes from the graph.
///
/// `classify(node, key)` picks a stage from the op's block label, or else its first weight's name
/// (`key`); `None` defers. Stages are monotonic along dataflow: an op is never in an earlier
/// stage than its producers. Ops left undecided take their earliest consumer's stage. A weight's
/// params count once per checkpoint name, in its first consumer's stage, so stage params sum to
/// the graph total.
fn assign_stages(
    g: &mut VizGraph,
    defs: Vec<StageDef>,
    classify: impl Fn(&VizNode, Option<&str>) -> Option<usize>,
) -> Vec<VizStage> {
    let producer: Vec<Option<usize>> = g
        .tensors
        .iter()
        .map(|t| match t.origin {
            Origin::Node(n) => Some(n.0 as usize),
            _ => None,
        })
        .collect();
    let is_weight = |t: TensorId| matches!(g.tensors[t.0 as usize].origin, Origin::Weight);

    let mut of: Vec<Option<usize>> = vec![None; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        let key = n.block.as_deref().or_else(|| {
            n.inputs
                .iter()
                .find(|&&t| is_weight(t))
                .map(|t| g.tensors[t.0 as usize].name.as_str())
        });
        let upstream = n
            .inputs
            .iter()
            .filter_map(|t| producer[t.0 as usize].and_then(|p| of[p]))
            .max();
        of[i] = match (classify(n, key).map(|s| s.min(defs.len() - 1)), upstream) {
            (Some(s), Some(u)) => Some(s.max(u)),
            (s, u) => s.or(u),
        };
    }
    for i in (0..g.nodes.len()).rev() {
        if of[i].is_none() {
            let out = g.nodes[i].output;
            of[i] = g
                .nodes
                .iter()
                .enumerate()
                .skip(i + 1)
                .filter(|(_, c)| c.inputs.contains(&out))
                .filter_map(|(j, _)| of[j])
                .min();
        }
    }
    let of: Vec<usize> = of.into_iter().map(|s| s.unwrap_or(0)).collect();

    let mut stages: Vec<VizStage> = defs
        .into_iter()
        .map(|d| VizStage {
            id: d.id.into(),
            name: d.name.into(),
            description: d.description,
            modules: Vec::new(),
            block_count: 0,
            param_count: 0,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        })
        .collect();

    let mut first_consumer: Vec<Option<usize>> = vec![None; g.tensors.len()];
    let mut consumer_stages: Vec<Vec<usize>> = vec![Vec::new(); g.tensors.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        for t in &n.inputs {
            first_consumer[t.0 as usize].get_or_insert(i);
            consumer_stages[t.0 as usize].push(of[i]);
        }
    }
    let block_index: HashMap<&str, usize> = g
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.as_str(), i))
        .collect();
    let block_stage: Vec<Option<usize>> = g
        .blocks
        .iter()
        .map(|b| b.node_ids.first().map(|n| of[n.0 as usize]))
        .collect();
    let mut counted = std::collections::HashSet::new();
    for (t, c) in g.tensors.iter().zip(&first_consumer) {
        if matches!(t.origin, Origin::Weight)
            && (t.name.is_empty() || counted.insert(t.name.as_str()))
        {
            let stage = match c {
                Some(c) => of[*c],
                None => owning_block(&t.name, &block_index)
                    .and_then(|b| block_stage[b])
                    .unwrap_or(0),
            };
            let st = &mut stages[stage];
            st.param_count += t.elements.unwrap_or(0);
            let m = module_of(&t.name);
            if !m.is_empty() && !st.modules.contains(&m) {
                st.modules.push(m);
            }
        }
    }

    let mut seen_in = vec![Vec::new(); stages.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        let s = of[i];
        for &t in &n.inputs {
            let from = producer[t.0 as usize].map(|p| of[p]);
            if !is_weight(t) && from != Some(s) && !seen_in[s].contains(&t) {
                seen_in[s].push(t);
                stages[s]
                    .input_shapes
                    .push(g.tensors[t.0 as usize].shape.clone());
            }
        }
        let out = n.output;
        if g.outputs.contains(&out) || consumer_stages[out.0 as usize].iter().any(|&c| c != s) {
            stages[s]
                .output_shapes
                .push(g.tensors[out.0 as usize].shape.clone());
        }
    }

    for b in &mut g.blocks {
        if let Some(&first) = b.node_ids.first() {
            let s = of[first.0 as usize];
            b.stage = Some(stages[s].id.clone());
            stages[s].block_count += 1;
        }
    }
    for (n, &s) in g.nodes.iter_mut().zip(&of) {
        n.stage = Some(stages[s].id.clone());
    }
    stages
}

/// The block whose label is the longest dotted prefix of `name`.
fn owning_block(name: &str, blocks: &HashMap<&str, usize>) -> Option<usize> {
    let mut end = name.len();
    while let Some(dot) = name[..end].rfind('.') {
        if let Some(&b) = blocks.get(&name[..dot]) {
            return Some(b);
        }
        end = dot;
    }
    None
}

/// `model.layers.3.mlp.up_proj.weight` -> `model.layers.*`; `model.norm.weight` -> `model.norm`.
fn module_of(weight: &str) -> String {
    let parts: Vec<&str> = weight.split('.').collect();
    match parts.iter().position(|p| p.parse::<u32>().is_ok()) {
        Some(i) => format!("{}.*", parts[..i].join(".")),
        None => parts[..parts.len().saturating_sub(1).max(1)].join("."),
    }
}

/// Input / Blocks / Output split for any graph with repeated blocks.
fn generic_stages(g: &mut VizGraph) -> Vec<VizStage> {
    let Some(first) = g.nodes.iter().position(|n| n.block.is_some()) else {
        let defs = vec![stage("network", "Network", "All ops")];
        return assign_stages(g, defs, |_, _| Some(0));
    };
    let last = g
        .nodes
        .iter()
        .rposition(|n| n.block.is_some())
        .unwrap_or(first);
    let defs = vec![
        stage(
            "input",
            "Input",
            "Ops before the first block: embeddings, position tables",
        ),
        stage(
            "blocks",
            "Blocks",
            format!("{} repeated blocks", g.blocks.len()),
        ),
        stage(
            "output",
            "Output",
            "Ops after the last block: final norm, output head",
        ),
    ];
    let mut stages = assign_stages(g, defs, |n, _| {
        let i = n.id.0 as usize;
        Some(if n.block.is_some() {
            1
        } else if i < first {
            0
        } else if i > last {
            2
        } else {
            return None;
        })
    });
    stages.retain(|st| {
        g.nodes
            .iter()
            .any(|n| n.stage.as_deref() == Some(st.id.as_str()))
    });
    stages
}

fn network(
    id: &str,
    title: &str,
    description: &str,
    graph: VizGraph,
    stages: Vec<VizStage>,
) -> VizNetwork {
    VizNetwork {
        id: id.into(),
        title: title.into(),
        description: description.into(),
        stages,
        graph,
    }
}

/// Single-network bundle with generic Input / Blocks / Output stages.
pub fn bundle_from_graph(
    g: &Graph,
    title: &str,
    source: &str,
    modality: ModelModality,
) -> VizModelBundle {
    let mut graph = graph_to_viz_graph(g);
    let stages = generic_stages(&mut graph);
    VizModelBundle {
        title: title.into(),
        source: source.into(),
        modality,
        default_network: "full".into(),
        networks: vec![network(
            "full",
            "Full model",
            "Every op of the model",
            graph,
            stages,
        )],
        notes: Vec::new(),
    }
}

/// Build the bundle for a model config. `bind` resolves symbolic dims (e.g. `B`, `S`, `L`) in
/// every network before conversion.
pub fn bundle_from_model_config(
    cfg: &crate::models::ModelConfig,
    title: &str,
    source: &str,
    bind: &crate::Bindings,
) -> Result<VizModelBundle, crate::models::BuildError> {
    let (modality, networks) = networks_for(cfg, bind)?;
    Ok(VizModelBundle {
        title: title.into(),
        source: source.into(),
        modality,
        default_network: networks[0].id.clone(),
        networks,
        notes: Vec::new(),
    })
}

type Networks = (ModelModality, Vec<VizNetwork>);

fn networks_for(
    cfg: &crate::models::ModelConfig,
    bind: &crate::Bindings,
) -> Result<Networks, crate::models::BuildError> {
    use crate::models::{build_graph, ModelConfig, ShapeBucket};
    let viz = |mut g: Graph| -> Result<VizGraph, crate::models::BuildError> {
        crate::infer_shapes(&mut g)?;
        g.bind(bind);
        Ok(graph_to_viz_graph(&g))
    };
    let full_net = |modality, desc: &str, graph, stages| {
        Ok((
            modality,
            vec![network("full", "Full model", desc, graph, stages)],
        ))
    };

    match cfg {
        ModelConfig::Qwen3Asr(c) => {
            let (a, t) = (&c.audio_config, &c.text_config);
            let defs = || {
                vec![
                    stage(
                        "audio_tower_conv",
                        "Audio tower: conv",
                        format!(
                            "2 strided Conv2d + GELU, {}x time subsampling",
                            a.downsample_rate
                        ),
                    ),
                    stage(
                        "audio_tower_layers",
                        "Audio tower: layers",
                        format!(
                            "{} transformer blocks (LayerNorm, attention, GELU MLP)",
                            a.encoder_layers
                        ),
                    ),
                    stage(
                        "audio_tower_proj",
                        "Audio tower: proj",
                        "Projects audio features to the text hidden size",
                    ),
                    stage(
                        "thinker",
                        "Thinker",
                        format!(
                            "Token embedding + {} Qwen3 blocks (RMSNorm, attention, SwiGLU MLP)",
                            t.num_hidden_layers
                        ),
                    ),
                    stage(
                        "lm_head",
                        "LM head",
                        "Final RMSNorm and vocabulary projection",
                    ),
                ]
            };
            // The composite prefixes every weight with `thinker.`; the standalone thinker does not.
            let classify = |_: &VizNode, key: Option<&str>| {
                let key = key?;
                let key = key.strip_prefix("thinker.").unwrap_or(key);
                Some(match key.strip_prefix("audio_tower.") {
                    Some(r) if r.starts_with("conv") => 0,
                    Some(r)
                        if r.starts_with("layers.") || r.starts_with("positional_embedding") =>
                    {
                        1
                    }
                    Some(_) => 2,
                    None if key.starts_with("model.embed_tokens")
                        || key.starts_with("model.layers.") =>
                    {
                        3
                    }
                    None => 4,
                })
            };
            let mut full = viz(crate::models::build_composite(c))?;
            let full_stages = assign_stages(&mut full, defs(), classify);
            let mut enc = viz(crate::models::build_audio_encoder(c))?;
            let enc_stages =
                assign_stages(&mut enc, defs().into_iter().take(3).collect(), classify);
            let mut dec = viz(crate::models::build_text_decoder(c))?;
            let dec_stages =
                assign_stages(&mut dec, defs().into_iter().skip(3).collect(), |n, k| {
                    classify(n, k).map(|s| s.saturating_sub(3))
                });

            Ok((
                ModelModality::Asr,
                vec![
                    network(
                        "full",
                        "Full model",
                        "Audio tower feeding the Qwen3 thinker",
                        full,
                        full_stages,
                    ),
                    network(
                        "audio_tower",
                        "Audio tower",
                        "Mel spectrogram to thinker-space audio embeddings",
                        enc,
                        enc_stages,
                    ),
                    network(
                        "thinker",
                        "Thinker",
                        "Qwen3 language model on its own",
                        dec,
                        dec_stages,
                    ),
                ],
            ))
        }
        ModelConfig::S3Gen(c) => {
            let mut full = viz(build_graph(cfg, &ShapeBucket::default())?)?;
            let defs = vec![
                stage(
                    "encoder",
                    "Encoder",
                    format!(
                        "Speech-token embedding + {} conformer blocks, 2x upsampling",
                        c.encoder_layers.0 + c.encoder_layers.1
                    ),
                ),
                stage(
                    "cfm",
                    "CFM decoder",
                    format!(
                        "Conditional flow matching estimator: {} transformer + {} mid blocks",
                        c.transformer_blocks, c.mid_blocks
                    ),
                ),
                stage(
                    "hift",
                    "HiFT vocoder",
                    format!(
                        "F0-driven source + {} transposed-conv upsampling stages to PCM",
                        c.upsample.len()
                    ),
                ),
            ];
            let stages = assign_stages(&mut full, defs, |_, key| {
                Some(match key?.split('.').next() {
                    Some("emb" | "enc") => 0,
                    Some("cfm") => 1,
                    _ => 2,
                })
            });
            full_net(
                ModelModality::Tts,
                "Speech tokens to waveform",
                full,
                stages,
            )
        }
        ModelConfig::Snac(c) => {
            let mut full = viz(build_graph(cfg, &ShapeBucket::default())?)?;
            let defs = vec![
                stage(
                    "quantizer",
                    "Quantizer",
                    format!(
                        "Sum of {} residual VQ codebooks (strides {:?})",
                        c.vq_strides.len(),
                        c.vq_strides
                    ),
                ),
                stage(
                    "decoder",
                    "Decoder",
                    format!(
                        "{} Snake + transposed-conv upsampling blocks",
                        c.blocks.len()
                    ),
                ),
                stage("output", "Output", "Projection to one audio channel + tanh"),
            ];
            let stages = assign_stages(&mut full, defs, |_, key| {
                Some(match key?.split('.').next() {
                    Some("codes" | "q") => 0,
                    Some("out") => 2,
                    _ => 1,
                })
            });
            full_net(ModelModality::AudioCodec, "Codes to waveform", full, stages)
        }
        ModelConfig::Flux(c) => {
            let mut g = viz(build_graph(cfg, &ShapeBucket::default())?)?;
            let defs = vec![
                stage(
                    "embedders",
                    "Embedders",
                    "x_embedder, context_embedder, time/guidance/pooled-text embedders",
                ),
                stage(
                    "double_stream",
                    "Double-stream blocks",
                    format!(
                        "{} MMDiT blocks: separate image/text weights, joint attention",
                        c.num_layers
                    ),
                ),
                stage(
                    "single_stream",
                    "Single-stream blocks",
                    format!(
                        "{} blocks over [text ; image] with fused attention + MLP",
                        c.num_single_layers
                    ),
                ),
                stage(
                    "output",
                    "Output",
                    "AdaLN norm_out and proj_out to packed latents",
                ),
            ];
            let stages = assign_stages(&mut g, defs, |_, key| {
                let key = key?;
                Some(if key.starts_with("single_transformer_blocks") {
                    2
                } else if key.starts_with("transformer_blocks") {
                    1
                } else if key.starts_with("norm_out") || key.starts_with("proj_out") {
                    3
                } else {
                    0
                })
            });
            full_net(
                ModelModality::Image,
                "Denoises packed image latents conditioned on text",
                g,
                stages,
            )
        }
        ModelConfig::QwenImageDit(c) => {
            let mut g = viz(build_graph(cfg, &ShapeBucket::default())?)?;
            let defs = vec![
                stage("embedders", "Embedders", "img_in, txt_norm, txt_in"),
                stage(
                    "transformer_blocks",
                    "Transformer blocks",
                    format!(
                        "{} MMDiT blocks: separate image/text weights, joint attention",
                        c.num_layers
                    ),
                ),
                stage(
                    "output",
                    "Output",
                    "AdaLN norm_out and proj_out to latent patches",
                ),
            ];
            let stages = assign_stages(&mut g, defs, |_, key| {
                let key = key?;
                Some(if key.starts_with("transformer_blocks") {
                    1
                } else if key.starts_with("norm_out") || key.starts_with("proj_out") {
                    2
                } else {
                    0
                })
            });
            full_net(
                ModelModality::Image,
                "Denoises image latent patches conditioned on text",
                g,
                stages,
            )
        }
        _ => {
            let modality = match cfg {
                ModelConfig::Gemma4Multimodal(_) | ModelConfig::QwenVl(_) => {
                    ModelModality::Multimodal
                }
                ModelConfig::Siglip(_) => ModelModality::Vision,
                ModelConfig::QwenImageVae(_) => ModelModality::Image,
                _ => ModelModality::Llm,
            };
            let mut g = viz(build_graph(cfg, &ShapeBucket::default())?)?;
            let stages = generic_stages(&mut g);
            full_net(modality, "Every op of the model", g, stages)
        }
    }
}

/// Build the bundle for a HuggingFace `config.json`. A multimodal checkpoint whose full graph
/// is not implemented (e.g. Gemma 4 with a vision tower) falls back to the language model that
/// the text-generation frontend serves, labelled as such.
pub fn bundle_from_config_json(
    json: &str,
    title: &str,
    source: &str,
    bind: &crate::Bindings,
) -> Result<VizModelBundle, crate::models::BuildError> {
    use crate::models::{build_text_generation_from_config_json_at, ModelConfig, ShapeBucket};
    let full_err = match ModelConfig::from_json(json) {
        Ok(cfg) => match bundle_from_model_config(&cfg, title, source, bind) {
            Ok(b) => return Ok(b),
            Err(e) => e,
        },
        Err(e) => e.into(),
    };
    let mut g = match build_text_generation_from_config_json_at(json, &ShapeBucket::default()) {
        Ok(g) => g,
        // Both failed: report the language-model reason too (e.g. Gemma 4 MoE).
        Err(text_err) if text_err.to_string() != full_err.to_string() => {
            return Err(crate::models::ConfigError::Unsupported(format!(
                "{}; language model: {}",
                strip_unsupported(&full_err.to_string()),
                strip_unsupported(&text_err.to_string())
            ))
            .into())
        }
        Err(_) => return Err(full_err),
    };
    g.bind(bind);
    let mut b = bundle_from_graph(&g, title, source, ModelModality::Multimodal);
    b.networks[0].title = "Language model".into();
    b.networks[0].description = "Text tower only".into();
    let v: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
    let towers: Vec<&str> = [("vision_config", "vision"), ("audio_config", "audio")]
        .into_iter()
        .filter(|(k, _)| v.get(k).is_some_and(|c| !c.is_null()))
        .map(|(_, t)| t)
        .collect();
    b.notes.push(match towers.as_slice() {
        [] => format!(
            "Language model only: {}",
            strip_unsupported(&full_err.to_string())
        ),
        t => format!(
            "Language model only; the {} tower{} not drawn yet.",
            t.join(" and "),
            if t.len() > 1 { "s are" } else { " is" }
        ),
    });
    Ok(b)
}

/// A pipeline text encoder as the graph the pipeline runs: the language model up to its final
/// hidden states (no LM head). Qwen2.5-VL's language model is a Qwen2 decoder; its vision tower
/// is not part of text encoding. `None` if the config has no encoder builder.
fn text_encoder_network(
    json: &str,
    bind: &crate::Bindings,
) -> Option<(VizNetwork, Option<String>)> {
    use crate::models::{build_encoder_graph, ModelConfig};
    let mut v: serde_json::Value = serde_json::from_str(json).ok()?;
    let mt = v
        .get("model_type")
        .and_then(|m| m.as_str())
        .unwrap_or_default()
        .to_string();
    let vl = matches!(mt.as_str(), "qwen2_5_vl" | "qwen2_vl");
    let mut text = v
        .get_mut("text_config")
        .map(serde_json::Value::take)
        .unwrap_or(v);
    let obj = text.as_object_mut()?;
    if vl {
        obj.remove("vision_config");
        obj.insert("model_type".into(), "qwen2".into());
    }
    let cfg = ModelConfig::from_json(&text.to_string()).ok()?;
    let mut g = build_encoder_graph(&cfg, &[]).ok()?;
    g.bind(bind);
    let mut graph = graph_to_viz_graph(&g);
    let stages = generic_stages(&mut graph);
    let note = vl
        .then(|| "language model only; the vision tower does not run for text prompts".to_string());
    Some((
        network("text_encoder", "text_encoder", "", graph, stages),
        note,
    ))
}

fn strip_unsupported(e: &str) -> &str {
    e.trim_start_matches("unsupported architecture: ")
}

/// Build the bundle for a diffusers pipeline repo. `model_index` is its `model_index.json`;
/// `read(path)` returns a repo file such as `transformer/config.json`. Each component becomes a
/// network named as in the repo; components without a graph builder are listed in `notes`.
pub fn bundle_from_diffusers(
    model_index: &str,
    read: impl Fn(&str) -> Result<String, String>,
    title: &str,
    source: &str,
    bind: &crate::Bindings,
) -> Result<VizModelBundle, String> {
    use crate::models::ModelConfig;
    let index: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(model_index).map_err(|e| format!("model_index.json: {e}"))?;
    let class = |v: &serde_json::Value| v.get(1).and_then(|c| c.as_str()).map(String::from);
    let mut components: Vec<(String, String)> = index
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .filter_map(|(k, v)| Some((k.clone(), class(v)?)))
        .collect();
    let rank = |name: &str| match name {
        "transformer" | "unet" => 0,
        "vae" => 2,
        _ => 1,
    };
    components.sort_by_key(|(name, _)| rank(name));

    let mut networks = Vec::new();
    let mut notes = Vec::new();
    let mut skipped = Vec::new();
    for (name, class) in components {
        let cfg = match read(&format!("{name}/config.json")) {
            Ok(json) => json,
            Err(_) => {
                if !matches!(
                    name.as_str(),
                    "scheduler" | "tokenizer" | "tokenizer_2" | "processor"
                ) {
                    skipped.push(format!("{name} ({class})"));
                }
                continue;
            }
        };
        let encoder = if name.starts_with("text_encoder") {
            text_encoder_network(&cfg, bind)
        } else {
            None
        };
        if let Some((net, note)) = encoder {
            let mut net = net;
            net.id = name.clone();
            net.title = name.clone();
            net.description = class.clone();
            networks.push(net);
            if let Some(note) = note {
                notes.push(format!("{name} ({class}): {note}"));
            }
            continue;
        }
        let built = ModelConfig::from_json(&cfg)
            .map_err(crate::models::BuildError::from)
            .and_then(|c| networks_for(&c, bind));
        match built {
            Ok((_, nets)) => {
                for mut n in nets.into_iter().filter(|n| n.id == "full") {
                    n.id = name.clone();
                    n.title = name.clone();
                    n.description = class.clone();
                    networks.push(n);
                }
            }
            Err(_) => skipped.push(format!("{name} ({class})")),
        }
    }
    if !skipped.is_empty() {
        notes.push(format!(
            "Not shown, no plow graph yet: {}",
            skipped.join(", ")
        ));
    }
    let Some(first) = networks.first() else {
        return Err(format!(
            "no component of {source} has a graph builder: {}",
            notes.join("; ")
        ));
    };
    Ok(VizModelBundle {
        title: title.into(),
        source: source.into(),
        modality: ModelModality::Image,
        default_network: first.id.clone(),
        networks,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelConfig, Qwen3AsrConfig};

    /// Invariants every bundle must hold, for every network.
    fn check(b: &VizModelBundle) {
        assert!(b.networks.iter().any(|n| n.id == b.default_network));
        for net in &b.networks {
            let g = &net.graph;
            let staged: u64 = net.stages.iter().map(|s| s.param_count).sum();
            assert_eq!(staged, g.summary.total_params, "{}: stage params", net.id);
            assert!(g.summary.total_params > 0, "{}: no params", net.id);
            let idx: HashMap<&str, usize> = net
                .stages
                .iter()
                .enumerate()
                .map(|(i, s)| (s.id.as_str(), i))
                .collect();
            let stage_of = |n: &VizNode| idx[n.stage.as_deref().expect("op has a stage")];
            for n in &g.nodes {
                for t in &n.inputs {
                    if let Origin::Node(p) = g.tensors[t.0 as usize].origin {
                        assert!(
                            stage_of(&g.nodes[p.0 as usize]) <= stage_of(n),
                            "{}: op {} ({}) precedes its producer's stage",
                            net.id,
                            n.id.0,
                            n.op
                        );
                    }
                }
            }
            for s in &net.stages {
                assert!(
                    !s.input_shapes.is_empty() || s.param_count > 0,
                    "{}: empty stage {s:?}",
                    net.id
                );
            }
            assert!(
                g.tensors.iter().all(|t| t.shape != "?"),
                "{}: tensor without an inferred shape",
                net.id
            );
            let blocks: u32 = net.stages.iter().map(|s| s.block_count).sum();
            assert_eq!(blocks, g.summary.num_blocks, "{}: stage blocks", net.id);
        }
        let html = render_html(b, false, None);
        assert!(html.contains("mode-standalone") && !html.contains("__BUNDLE__"));
        assert!(render_html(b, true, None).contains("mode-embed"));
        assert!(bundle_to_svg(b).starts_with("<svg"));
    }

    fn stage_ids(net: &VizNetwork) -> Vec<&str> {
        net.stages.iter().map(|s| s.id.as_str()).collect()
    }

    fn from_json(json: &str) -> VizModelBundle {
        bundle_from_config_json(json, "t", "t", &Default::default()).expect("bundle")
    }

    #[test]
    fn asr_networks_and_stages() {
        let mut c = Qwen3AsrConfig::default();
        c.audio_config.encoder_layers = 2;
        c.text_config.num_hidden_layers = 2;
        let b =
            bundle_from_model_config(&ModelConfig::Qwen3Asr(c), "asr", "asr", &Default::default())
                .unwrap();
        check(&b);
        assert_eq!(b.modality, ModelModality::Asr);
        let ids: Vec<&str> = b.networks.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["full", "audio_tower", "thinker"]);
        let full = &b.networks[0];
        assert_eq!(
            stage_ids(full),
            [
                "audio_tower_conv",
                "audio_tower_layers",
                "audio_tower_proj",
                "thinker",
                "lm_head"
            ]
        );
        assert_eq!(full.stages[1].block_count, 2);
        assert_eq!(full.stages[3].block_count, 2);
        assert!(full.stages[1]
            .modules
            .contains(&"thinker.audio_tower.layers.*".to_string()));
        assert!(full.stages[3]
            .modules
            .contains(&"thinker.model.layers.*".to_string()));
        assert!(full
            .stages
            .iter()
            .all(|s| s.param_count > 0 || s.id == "lm_head"));
    }

    #[test]
    fn graph_types_serialize_directly() {
        let mut c = Qwen3AsrConfig::default();
        c.audio_config.encoder_layers = 1;
        c.text_config.num_hidden_layers = 1;
        let bind = crate::Bindings::new().set("B", 2);
        let b = bundle_from_model_config(&ModelConfig::Qwen3Asr(c), "t", "t", &bind).unwrap();
        let v = serde_json::to_value(b.default_graph()).unwrap();
        let t0 = &v["tensors"][0];
        assert_eq!(t0["origin"], "input");
        assert_eq!(t0["dtype"], "bf16");
        assert!(t0["shape"].as_str().unwrap().starts_with("[2, "));
        let lin = v["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["op"] == "linear")
            .unwrap();
        assert!(lin["detail"]["Linear"]["out_features"].is_i64());
        assert_eq!(lin["category"], "linear");
        assert!(lin["inputs"][0].is_u64());
    }

    const GEMMA3: &str = r#"{"model_type": "gemma3", "vocab_size": 1000, "hidden_size": 256,
        "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "sliding_window": 512, "sliding_window_pattern": 2, "query_pre_attn_scalar": 256.0,
        "use_qk_norm": true, "torch_dtype": "bfloat16"}"#;

    const GEMMA4_TEXT: &str = r#"{"model_type": "gemma4_unified", "text_config": {
        "model_type": "gemma4_unified_text", "vocab_size": 1000, "hidden_size": 256,
        "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_global_key_value_heads": 1, "head_dim": 64,
        "global_head_dim": 128, "attention_k_eq_v": true, "rms_norm_eps": 1e-6,
        "sliding_window": 512,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
                            "sliding_attention": {"rope_theta": 10000.0}}}}"#;

    const GEMMA4_MULTIMODAL: &str = r#"{"model_type": "gemma4_unified",
        "vision_config": {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
          "num_attention_heads": 4, "num_channels": 3, "image_size": 56, "patch_size": 14,
          "layer_norm_eps": 1e-6, "hidden_act": "gelu_pytorch_tanh"},
        "mm_input_projection_config": {"input_size": 64, "output_size": 128},
        "text_config": {"model_type": "gemma4_unified_text", "vocab_size": 500, "hidden_size": 128,
          "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 32, "attention_k_eq_v": true, "use_qk_norm": true,
          "query_pre_attn_scalar": 32.0, "sliding_window": 512,
          "layer_types": ["sliding_attention", "full_attention"]}}"#;

    fn attention_head_dims(net: &VizNetwork) -> Vec<u32> {
        net.graph
            .nodes
            .iter()
            .filter_map(|n| match n.detail {
                Op::Attention { head_dim, .. } => Some(head_dim),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn gemma3_text() {
        let b = from_json(GEMMA3);
        check(&b);
        assert_eq!(b.modality, ModelModality::Llm);
        let net = &b.networks[0];
        assert_eq!(stage_ids(net), ["input", "blocks", "output"]);
        assert_eq!(net.stages[1].block_count, 4);
        assert_eq!(net.graph.summary.num_blocks, 4);
        assert!(b.notes.is_empty());
        let out = net.graph.outputs[0];
        assert_eq!(net.graph.tensors[out.0 as usize].shape, "[B, S, 1000]");
    }

    #[test]
    fn tied_embeddings_count_once() {
        // Qwen3 declares the tied lm_head as a second tensor named like embed_tokens.
        let json = r#"{"model_type": "qwen3", "vocab_size": 1000, "hidden_size": 64,
            "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "tie_word_embeddings": true}"#;
        let b = from_json(json);
        check(&b);
        let g = &b.networks[0].graph;
        let uses = g
            .tensors
            .iter()
            .filter(|t| t.name == "model.embed_tokens.weight")
            .count();
        assert_eq!(uses, 2);
        let all: u64 = g
            .tensors
            .iter()
            .filter(|t| matches!(t.origin, Origin::Weight))
            .map(|t| t.elements.unwrap())
            .sum();
        assert_eq!(g.summary.total_params, all - 1000 * 64);
        assert_eq!(b.networks[0].stages[0].param_count, 1000 * 64);
    }

    #[test]
    fn gemma4_text_keeps_per_layer_attention() {
        let b = from_json(GEMMA4_TEXT);
        check(&b);
        let net = &b.networks[0];
        assert_eq!(attention_head_dims(net), [64, 64, 128, 64]);
        assert_eq!(net.stages[1].block_count, 4);
        let bind = crate::Bindings::new().set("B", 1).set("S", 32);
        let b = bundle_from_config_json(GEMMA4_TEXT, "t", "t", &bind).unwrap();
        let out = b.networks[0].graph.outputs[0];
        assert_eq!(
            b.networks[0].graph.tensors[out.0 as usize].shape,
            "[1, 32, 1000]"
        );
    }

    #[test]
    fn gemma4_multimodal_falls_back_to_language_model() {
        let b = from_json(GEMMA4_MULTIMODAL);
        check(&b);
        assert_eq!(b.modality, ModelModality::Multimodal);
        assert_eq!(b.networks[0].title, "Language model");
        assert_eq!(b.notes.len(), 1);
        assert!(
            b.notes[0].contains("vision tower is not drawn"),
            "{}",
            b.notes[0]
        );
        assert_eq!(attention_head_dims(&b.networks[0]).len(), 2);
        assert!(!b.networks[0].graph.nodes.iter().any(|n| n.op == "conv2d"));
    }

    #[test]
    fn gemma_family_unsupported_is_an_error() {
        let e =
            bundle_from_config_json(r#"{"model_type": "gemma2"}"#, "t", "t", &Default::default())
                .unwrap_err();
        assert!(e.to_string().contains("only Gemma 3/4"), "{e}");
    }

    #[test]
    fn tts_snac() {
        let cfg = ModelConfig::Snac(crate::models::config::SnacConfig::default());
        let b = bundle_from_model_config(&cfg, "snac", "snac", &Default::default()).unwrap();
        check(&b);
        assert_eq!(b.modality, ModelModality::AudioCodec);
        assert_eq!(
            stage_ids(&b.networks[0]),
            ["quantizer", "decoder", "output"]
        );
        assert!(b.networks[0].stages.iter().all(|s| s.param_count > 0));
    }

    #[test]
    fn tts_s3gen() {
        const S3GEN: &str = r#"{"model_type": "chatterbox_s3gen", "vocab": 6561,
          "upsample": [[8, 16, 4], [5, 11, 3], [3, 7, 2]], "source_downs": [[30, 15, 7], [6, 3, 1], [1, 1, 0]],
          "source_resblocks": [[7, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]]],
          "resblocks": [[3, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]], [3, [1, 3, 5]], [7, [1, 3, 5]],
            [11, [1, 3, 5]], [3, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]]],
          "n_fft": 16, "harmonics": 9}"#;
        let b = from_json(S3GEN);
        check(&b);
        assert_eq!(b.modality, ModelModality::Tts);
        let net = &b.networks[0];
        assert_eq!(stage_ids(net), ["encoder", "cfm", "hift"]);
        assert!(net.stages.iter().all(|s| s.param_count > 0));
        assert!(net.stages[1].modules.iter().all(|m| m.starts_with("cfm.")));
        assert!(net.stages[2].modules.iter().all(|m| m.starts_with("hift.")));
    }

    const FLUX_SMALL: &str = r#"{"_class_name": "FluxTransformer2DModel", "patch_size": 1,
        "in_channels": 64, "out_channels": null, "num_layers": 2, "num_single_layers": 3,
        "attention_head_dim": 32, "num_attention_heads": 4, "joint_attention_dim": 128,
        "pooled_projection_dim": 64, "guidance_embeds": true, "axes_dims_rope": [8, 12, 12]}"#;

    #[test]
    fn flux_dit() {
        let bind = crate::Bindings::new().set("B", 1).set("L", 512);
        let b = bundle_from_config_json(FLUX_SMALL, "flux", "flux", &bind).unwrap();
        check(&b);
        assert_eq!(b.modality, ModelModality::Image);
        let net = &b.networks[0];
        assert_eq!(
            stage_ids(net),
            ["embedders", "double_stream", "single_stream", "output"]
        );
        assert_eq!(net.stages[1].block_count, 2);
        assert_eq!(net.stages[2].block_count, 3);
        assert_eq!(net.stages[1].modules, ["transformer_blocks.*"]);
        assert_eq!(net.stages[2].modules, ["single_transformer_blocks.*"]);
        let g = &net.graph;
        // 1024² image → 64² packed tokens; proj_out back to the 64 packed latent channels.
        assert_eq!(g.tensors[g.outputs[0].0 as usize].shape, "[1, 4096, 64]");
        let names: Vec<&str> = g
            .inputs
            .iter()
            .map(|t| g.tensors[t.0 as usize].name.as_str())
            .collect();
        assert!(names.contains(&"guidance_proj"));
        let attn = g.nodes.iter().filter(|n| n.op == "attention").count();
        assert_eq!(attn, 5);

        let schnell = FLUX_SMALL.replace("\"guidance_embeds\": true", "\"guidance_embeds\": false");
        let b = from_json(&schnell);
        let g = &b.networks[0].graph;
        assert!(!g
            .inputs
            .iter()
            .any(|t| g.tensors[t.0 as usize].name == "guidance_proj"));
    }

    #[test]
    fn flux_dev_param_count_matches_checkpoint() {
        // FLUX.1-dev transformer: 11.9B parameters (diffusers safetensors index). Norm weights
        // that the checkpoint does not store (affine-free LayerNorm) are the only extra params.
        let b = from_json(r#"{"_class_name": "FluxTransformer2DModel"}"#);
        let p = b.networks[0].graph.summary.total_params as f64;
        assert!((p / 11.9e9 - 1.0).abs() < 0.01, "{p}");
    }

    const QWEN_IMAGE_DIT: &str = r#"{"_class_name": "QwenImageTransformer2DModel", "num_layers": 2,
        "num_attention_heads": 4, "attention_head_dim": 32, "in_channels": 64, "out_channels": 16,
        "patch_size": 2, "joint_attention_dim": 128, "pooled_projection_dim": 64,
        "axes_dims_rope": [8, 12, 12], "guidance_embeds": false}"#;
    const QWEN_IMAGE_VAE: &str = r#"{"_class_name": "AutoencoderKLQwenImage", "base_dim": 32,
        "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1, "z_dim": 16,
        "temperal_downsample": [false, true, true]}"#;

    #[test]
    fn qwen_image_dit_and_vae() {
        let b = from_json(QWEN_IMAGE_DIT);
        check(&b);
        assert_eq!(b.modality, ModelModality::Image);
        assert_eq!(
            stage_ids(&b.networks[0]),
            ["embedders", "transformer_blocks", "output"]
        );
        assert_eq!(b.networks[0].stages[1].block_count, 2);
        let b = from_json(QWEN_IMAGE_VAE);
        check(&b);
        assert_eq!(b.modality, ModelModality::Image);
    }

    #[test]
    fn diffusers_pipeline_components() {
        let index = r#"{"_class_name": "QwenImagePipeline", "_diffusers_version": "0.35.0",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"]}"#;
        let read = |f: &str| match f {
            "transformer/config.json" => Ok(QWEN_IMAGE_DIT.to_string()),
            "vae/config.json" => Ok(QWEN_IMAGE_VAE.to_string()),
            "text_encoder/config.json" => Ok(r#"{"model_type": "clip_text_model"}"#.to_string()),
            "text_encoder_2/config.json" => Ok(r#"{"model_type": "qwen2_5_vl", "vocab_size": 1000,
                "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vision_config": {"depth": 2, "hidden_size": 32}}"#
                .to_string()),
            _ => Err("missing".to_string()),
        };
        let b = bundle_from_diffusers(
            index,
            read,
            "Qwen-Image",
            "Qwen/Qwen-Image",
            &Default::default(),
        )
        .unwrap();
        check(&b);
        let ids: Vec<&str> = b.networks.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(
            ids,
            ["transformer", "text_encoder_2", "vae"],
            "{:?}",
            b.notes
        );
        assert_eq!(b.default_network, "transformer");
        assert_eq!(b.networks[0].description, "QwenImageTransformer2DModel");
        let te = &b.networks[1].graph;
        assert_eq!(te.summary.num_blocks, 2);
        assert!(!te.tensors.iter().any(|t| t.name.starts_with("lm_head")));
        assert_eq!(te.tensors[te.outputs[0].0 as usize].shape, "[B, S, 64]");
        assert_eq!(b.notes.len(), 2, "{:?}", b.notes);
        assert_eq!(
            b.notes[1],
            "Not shown, no plow graph yet: text_encoder (CLIPTextModel)"
        );
        assert!(b.notes[0].contains("vision tower"), "{:?}", b.notes);
    }

    #[test]
    fn declared_experts_count_in_their_block() {
        // DeepSeek declares every routed expert for the checkpoint but runs one per layer.
        let json = r#"{"model_type": "deepseek_v3", "vocab_size": 1000, "hidden_size": 256,
            "intermediate_size": 512, "num_hidden_layers": 3, "num_attention_heads": 4,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "q_lora_rank": 96, "kv_lora_rank": 64,
            "qk_rope_head_dim": 16, "qk_nope_head_dim": 32, "v_head_dim": 32,
            "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "moe_intermediate_size": 128, "first_k_dense_replace": 1}"#;
        let b = from_json(json);
        check(&b);
        let net = &b.networks[0];
        let g = &net.graph;
        let folded: Vec<&VizTensor> = g
            .tensors
            .iter()
            .filter(|t| t.name.contains(".experts.*."))
            .collect();
        assert_eq!(
            folded.len(),
            2 * 3,
            "2 MoE layers x gate/up/down, one entry each"
        );
        assert!(folded
            .iter()
            .all(|t| t.copies == 8 && t.elements == Some(8 * 256 * 128)));
        let expert = 3 * 256 * 128;
        let moe_block = &g.blocks[1];
        assert!(
            moe_block.param_count >= 8 * expert as u64,
            "{}",
            moe_block.param_count
        );
        assert!(net.stages[1].param_count > net.stages[0].param_count);
    }

    #[test]
    fn representative_expert_counts_every_expert() {
        let mut nn = crate::Nn::new(crate::DType::BF16, crate::DType::BF16);
        let b = nn.sym("B");
        let s = nn.sym("S");
        let x = nn.input(
            "x",
            nn.shape([b, s, crate::Dim::stat(64)]),
            crate::DType::BF16,
        );
        nn.begin_block("layers.0");
        let _ = nn.moe_router("layers.0.gate", x, 64, 16, 2);
        let y = nn.linear("layers.0.experts.0.up_proj", x, 64, 32, false);
        nn.end_block();
        nn.mark_output(y);
        let mut g = nn.finish();
        crate::infer_shapes(&mut g).unwrap();
        let b = bundle_from_graph(&g, "t", "t", ModelModality::Llm);
        check(&b);
        let g = &b.networks[0].graph;
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.experts.0.up_proj.weight")
            .unwrap();
        assert_eq!(up.copies, 16);
        assert_eq!(up.elements, Some(16 * 64 * 32));
        assert_eq!(g.blocks[0].param_count, 16 * 64 * 32 + 64 * 16);
    }

    #[test]
    fn inline_json_cannot_close_the_script() {
        let mut b = from_json(GEMMA3);
        b.title = "</script><b>".into();
        let html = render_html(&b, false, None);
        assert!(!html.contains("</script><b>"));
        assert!(html.contains("&lt;/script&gt;"));
    }

    #[test]
    fn script_base_replaces_the_cdn() {
        let b = from_json(GEMMA3);
        let html = render_html(&b, true, Some("/viz/lib/"));
        assert!(html.contains(r#"<script src="/viz/lib/cytoscape.min.js">"#));
        assert!(html.contains(r#"<script src="/viz/lib/cytoscape-dagre.js">"#));
        assert!(!html.contains("jsdelivr"));
        assert!(render_html(&b, true, None).contains("cdn.jsdelivr.net/npm/cytoscape@3.30.4"));
    }

    #[test]
    fn dump_assets_writes_every_file() {
        let b = from_json(GEMMA3);
        let dir = std::env::temp_dir().join(format!("plow_viz_dump_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dump_assets(&b, &dir, None).unwrap();
        for f in [
            "index.html",
            "embed.html",
            "model.json",
            "pipeline.json",
            "pipeline.svg",
        ] {
            assert!(dir.join(f).is_file(), "{f}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
