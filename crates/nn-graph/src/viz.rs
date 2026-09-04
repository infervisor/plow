//! Graph visualization export — JSON + self-contained HTML viewer.
//!
//! Produces a JSON representation of the [`Graph`] DAG showing every tensor
//! (with name, dtype, inferred shape), every operation (with op-specific detail),
//! and the dataflow edges between them. The HTML viewer renders it in the
//! browser with Cytoscape.js (canvas) using a dagre layered layout.
//!
//! The viewer libraries (cytoscape, dagre, cytoscape-dagre) load from the
//! jsdelivr CDN — deliberately not vendored, to keep them out of the repo and
//! the binary; the page needs a network connection. The first version used
//! dagre-d3 (unmaintained since 2018) and laid out the ENTIRE graph on load; a
//! 93-layer model is ~3.7k nodes and the synchronous layout never returned.
//! The viewer therefore opens on the first block and loads "all blocks" only
//! on explicit request.
//!
//! # Usage
//!
//! ```ignore
//! let g = nn_graph::models::build_from_config_json(json)?;
//! let json = nn_graph::viz::graph_to_json(&g);
//! let html = nn_graph::viz::graph_to_html(&g, "kimi-k3");
//! std::fs::write("graph.html", html)?;
//! // open graph.html in browser
//! ```

use crate::graph::{Graph, Origin};
use crate::op::Op;

/// Serialize the graph to a JSON string suitable for visualization tools.
///
/// Schema:
/// ```json
/// {
///   "tensors": [ { "id", "name", "dtype", "shape", "origin" } ],
///   "nodes": [ { "id", "op", "detail", "inputs", "output", "block" } ],
///   "blocks": ["layers.0", ...],
///   "inputs": [0, ...],
///   "outputs": [42, ...]
/// }
/// ```
#[cfg(feature = "models")]
pub fn graph_to_json(g: &Graph) -> String {
    let v = graph_to_value(g);
    serde_json::to_string_pretty(&v).expect("graph serialization cannot fail")
}

/// Serialize the graph to a `serde_json::Value`.
#[cfg(feature = "models")]
pub fn graph_to_value(g: &Graph) -> serde_json::Value {
    use serde_json::{json, Value};

    let tensors: Vec<Value> = g
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let shape_str = t
                .shape
                .as_ref()
                .map(|s| s.display_with(&g.syms))
                .unwrap_or_else(|| "?".into());
            let origin = match &t.origin {
                Origin::Input => json!("input"),
                Origin::Weight => json!("weight"),
                Origin::Node(nid) => json!({ "node": nid.0 }),
            };
            json!({
                "id": i,
                "name": t.name.as_deref().unwrap_or(""),
                "dtype": format!("{}", t.dtype),
                "shape": shape_str,
                "origin": origin,
            })
        })
        .collect();

    let nodes: Vec<Value> = g
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let block = n
                .block
                .and_then(|idx| g.block_label(idx))
                .map(|s| json!(s))
                .unwrap_or(Value::Null);
            json!({
                "id": i,
                "op": n.op.name(),
                "detail": op_detail(&n.op),
                "inputs": n.inputs.iter().map(|t| t.0).collect::<Vec<_>>(),
                "output": n.output.0,
                "block": block,
            })
        })
        .collect();

    json!({
        "tensors": tensors,
        "nodes": nodes,
        "blocks": &g.blocks,
        "inputs": g.inputs.iter().map(|t| t.0).collect::<Vec<_>>(),
        "outputs": g.outputs.iter().map(|t| t.0).collect::<Vec<_>>(),
    })
}

/// Extract op-specific parameters as a JSON value for display.
#[cfg(feature = "models")]
fn op_detail(op: &Op) -> serde_json::Value {
    use serde_json::json;
    match op {
        Op::Linear { out_features, bias } => json!({
            "out_features": out_features,
            "bias": bias,
        }),
        Op::MatMul => json!({}),
        Op::RmsNorm { eps } => json!({ "eps": eps }),
        Op::RmsNormZeroCentered { eps } => json!({ "eps": eps }),
        Op::LayerNorm { eps } => json!({ "eps": eps }),
        Op::Rope {
            dim,
            theta,
            interleave,
            frequency_dim,
        } => json!({
            "dim": dim, "theta": theta, "interleave": interleave,
            "frequency_dim": frequency_dim,
        }),
        Op::Act(kind) => json!({ "activation": format!("{kind:?}") }),
        Op::Elementwise(kind) => json!({ "kind": format!("{kind:?}") }),
        Op::Scale(s) => json!({ "scale": s }),
        Op::Softmax { axis } => json!({ "axis": axis }),
        Op::Attention {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            sliding_window,
            logit_softcap,
        } => json!({
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "causal": causal,
            "sliding_window": sliding_window,
            "logit_softcap": logit_softcap,
        }),
        Op::Embedding => json!({}),
        Op::Conv2d { stride, padding } => json!({
            "stride": [stride.0, stride.1],
            "padding": [padding.0, padding.1],
        }),
        Op::Conv3d { stride, padding } => json!({
            "stride": [stride.0, stride.1, stride.2],
            "padding": [padding.0, padding.1, padding.2],
        }),
        Op::GroupNorm { groups, eps } => json!({ "groups": groups, "eps": eps }),
        Op::Reshape { shape } => json!({ "shape": format!("{shape}") }),
        Op::Transpose { perm } => json!({ "perm": perm }),
        Op::Broadcast { shape } => json!({ "shape": format!("{shape}") }),
        Op::Concat { axis } => json!({ "axis": axis }),
        Op::Slice { axis, start, len } => json!({
            "axis": axis, "start": format!("{start}"), "len": format!("{len}"),
        }),
        Op::Reduce {
            kind,
            axis,
            keepdim,
        } => json!({
            "kind": format!("{kind:?}"), "axis": axis, "keepdim": keepdim,
        }),
        Op::MoeRouter {
            num_experts,
            top_k,
            group,
        } => json!({
            "num_experts": num_experts,
            "top_k": top_k,
            "group": group.map(|g| serde_json::json!({
                "n_group": g.n_group,
                "topk_group": g.topk_group,
            })),
        }),
        Op::Conv1dDepthwise { kernel } => json!({ "kernel": kernel }),
        Op::LinearAttention {
            kind,
            num_heads,
            head_dim,
        } => json!({
            "kind": format!("{kind:?}"),
            "num_heads": num_heads,
            "head_dim": head_dim,
        }),
        Op::SituGlu { beta, linear_beta } => json!({
            "beta": beta, "linear_beta": linear_beta,
        }),
        Op::BlockResidual { max_snapshots } => json!({ "max_snapshots": max_snapshots }),
    }
}

/// Generate an HTML page that visualizes the graph in the browser.
///
/// Cytoscape.js (CDN-loaded, canvas-rendered) with a dagre layered layout. Nodes
/// are colored by op type; clicking a node shows its tensor details. Opens on
/// the first block — laying out every node of a large model at once is the
/// exact hang this viewer replaced, so "all blocks" is explicit opt-in.
/// `source` prefills the model form: the HF id / path the graph was built
/// from, so re-loading with a batch/seq binding is one click.
#[cfg(feature = "models")]
pub fn graph_to_html(g: &Graph, title: &str, source: &str) -> String {
    let json = graph_to_json(g);
    format!(
        r##"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title} — nn-graph viewer</title>
<style>
/* infervisor.ai design tokens (dark) */
:root {{
  --bg:#0a0b0d; --bg2:#0c0e12;
  --panel:#101318; --panel2:#0d0f13;
  --line:#1d2128; --line2:#272c34;
  --text:#e7e9ec; --muted:#8b929c; --dim:#565d67;
  --blue:#5a9fd4; --blue-dim:#3d6e9c; --blue-deep:#1F4E79; --stem:#8fcaf4;
  --green:#6cc0a0; --amber:#b8813a;
  --mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{ font-family: var(--mono); background: var(--bg); color: var(--text); display: flex; flex-direction: column; overflow: hidden; -webkit-font-smoothing: antialiased; }}
#toolbar {{ flex: 0 0 46px; background: var(--panel2); display: flex; align-items: center; padding: 0 16px; z-index: 100; border-bottom: 1px solid var(--line); }}
#toolbar h1 {{ font-size: 14px; font-weight: 600; color: var(--text); white-space: nowrap; }}
#toolbar h1 .hl {{ color: var(--blue); }}
#toolbar .controls {{ margin-left: auto; display: flex; gap: 6px; align-items: center; }}
button {{ background: transparent; border: 1px solid var(--line2); color: var(--text); padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: var(--mono); transition: .15s; }}
button:hover {{ border-color: var(--blue-dim); color: var(--blue); }}
select, input {{ background: var(--panel); border: 1px solid var(--line2); color: var(--text); padding: 4px 8px; border-radius: 4px; font-size: 12px; font-family: var(--mono); }}
select:focus, input:focus {{ outline: none; border-color: var(--blue-dim); }}
#toolbar select {{ max-width: 200px; }}
#status {{ color: var(--dim); font-size: 12px; white-space: nowrap; }}
#formbar {{ flex: 0 0 40px; background: var(--panel2); display: flex; align-items: center; gap: 8px; padding: 0 16px; border-bottom: 1px solid var(--line); font-size: 12px; color: var(--muted); }}
#formbar input#f-model {{ flex: 1; max-width: 460px; }}
#formbar input[type=number] {{ width: 76px; }}
#formbar label {{ display: flex; align-items: center; gap: 5px; color: var(--dim); }}
#main {{ flex: 1; display: flex; min-height: 0; position: relative; }}
#graph-container {{ flex: 1; min-width: 0; position: relative; background: var(--bg); }}
#sidebar {{ flex: 0 0 340px; background: var(--panel2); border-left: 1px solid var(--line); overflow-y: auto; padding: 14px; font-size: 13px; }}
#sidebar.collapsed {{ display: none; }}
@media (max-width: 900px) {{
  #sidebar {{ position: absolute; right: 0; top: 0; bottom: 0; width: 300px; z-index: 150; box-shadow: -4px 0 12px rgba(0,0,0,0.6); }}
}}
#sidebar h2 {{ color: var(--blue); font-size: 13px; margin-bottom: 8px; letter-spacing: 0.3px; }}
#sidebar .label {{ color: var(--dim); font-size: 10px; text-transform: uppercase; margin-top: 8px; letter-spacing: 0.5px; }}
#sidebar pre {{ white-space: pre-wrap; }}
.legend {{ position: fixed; bottom: 12px; left: 12px; background: var(--panel); border: 1px solid var(--line2); border-radius: 9px; padding: 8px 12px; font-size: 11px; z-index: 50; color: var(--muted); }}
.legend-item {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
.legend-swatch {{ width: 13px; height: 13px; border-radius: 3px; }}
#tooltip {{ position: fixed; display: none; background: var(--panel); border: 1px solid var(--line2); border-radius: 9px; padding: 9px 11px; font-size: 12px; z-index: 200; max-width: 420px; pointer-events: none; font-family: var(--mono); box-shadow: 0 6px 24px rgba(0,0,0,0.6); }}
#tooltip .label {{ color: var(--dim); font-size: 10px; text-transform: uppercase; margin-top: 7px; letter-spacing: 0.5px; }}
#tooltip .label:first-child {{ margin-top: 0; }}
#tooltip pre {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<div id="toolbar">
  <h1><span class="hl">nn-graph</span> · <span id="title">{title}</span></h1>
  <div class="controls">
    <span id="status"></span>
    <button id="prev-block" title="previous block">◀</button>
    <select id="block-filter"></select>
    <button id="next-block" title="next block">▶</button>
    <button onclick="cy && smartFit()">Fit</button>
    <button onclick="cy && cy.zoom(cy.zoom() * 1.3)">+</button>
    <button onclick="cy && cy.zoom(cy.zoom() / 1.3)">−</button>
    <button id="toggle-sidebar" title="toggle details panel">☰</button>
  </div>
</div>
<div id="formbar">
  <input id="f-model" placeholder="HF model id, checkpoint dir, or config.json path" value="{source}">
  <label>batch <input id="f-batch" type="number" min="1" placeholder="B"></label>
  <label>seq <input id="f-seq" type="number" min="1" placeholder="S"></label>
  <button id="f-load">Load</button>
</div>
<div id="main">
  <div id="graph-container"></div>
  <div id="sidebar">
    <h2>Node Details</h2>
    <p style="color:#666">Click a node to inspect</p>
    <div id="detail-content"></div>
  </div>
</div>
<div class="legend" id="legend"></div>
<div id="tooltip"></div>

<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<script>
let DATA = {json};

// infervisor.ai families: blues for compute, green for activations,
// amber for routing, slate for norms; structural ops stay neutral.
const OP_COLORS = {{
  'linear': ['#0e1820','#3d6e9c'], 'matmul': ['#0e1820','#3d6e9c'],
  'attention': ['#132c47','#8fcaf4'], 'linear_attention': ['#132c47','#8fcaf4'],
  'embedding': ['#0e1820','#5a9fd4'],
  'rmsnorm': ['#14171c','#8b929c'], 'layernorm': ['#14171c','#8b929c'], 'groupnorm': ['#14171c','#8b929c'],
  'act': ['#0f1d17','#6cc0a0'], 'situ_glu': ['#0f1d17','#6cc0a0'],
  'conv1d_depthwise': ['#10233a','#5a9fd4'], 'conv2d': ['#10233a','#5a9fd4'], 'conv3d': ['#10233a','#5a9fd4'],
  'moe_router': ['#231708','#b8813a'], 'block_residual': ['#211018','#b07a9a'],
  'softmax': ['#14171c','#565d67'], 'rope': ['#0e1820','#3d6e9c'],
}};
const DEFAULT_COLOR = ['#101318','#3a424c'];
const colorFor = op => OP_COLORS[op] || DEFAULT_COLOR;

// tensor id → consuming node ids, for tooltips. Rebuilt on model swap.
let consumers = [];
function reindex() {{
  consumers = DATA.tensors.map(() => []);
  DATA.nodes.forEach(n => n.inputs.forEach(tid => consumers[tid].push(n.id)));
}}
reindex();

// "layers.0.linear_attn.q_proj.weight" → "q_proj"
const short = name => name.replace(/\.weight$/, '').split('.').pop();

// The op INSTANCE name comes from its weight operand: linear + q_proj.weight
// is the q projection. Weightless ops (reshape, attention, …) have no
// instance name and show their type alone.
function opName(n) {{
  for (const tid of n.inputs) {{
    const t = DATA.tensors[tid];
    if (t.origin === 'weight') return short(t.name);
  }}
  return '';
}}

// Block filter. Default = first block, NOT 'all': the full graph of a large
// model is thousands of nodes and the layout is synchronous.
const blockFilter = document.getElementById('block-filter');
function rebuildBlockOptions() {{
  blockFilter.innerHTML = '';
  const globalCount = DATA.nodes.filter(n => n.block === null).length;
  const opt = document.createElement('option');
  opt.value = 'all';
  opt.textContent = `All blocks (${{DATA.nodes.length}} ops — slow)`;
  blockFilter.appendChild(opt);
  if (globalCount > 0) {{
    const g = document.createElement('option');
    g.value = 'globals';
    g.textContent = `(globals: ${{globalCount}} nodes)`;
    blockFilter.appendChild(g);
  }}
  DATA.blocks.forEach((b, i) => {{
    const o = document.createElement('option');
    o.value = i; o.textContent = b;
    blockFilter.appendChild(o);
  }});
  blockFilter.value = DATA.blocks.length > 0 ? '0' : 'all';
}}
rebuildBlockOptions();
blockFilter.addEventListener('change', () => render(blockFilter.value));

// Model/shape form. Works in serve mode: the plowc server rebuilds the graph
// (`/graph?model=…&batch=…&seq=…`) and binds B/S so every inferred shape
// comes back concrete. In file-dump mode the fetch fails and says so.
function setData(graph, title) {{
  DATA = graph;
  reindex();
  document.title = title + ' — nn-graph viewer';
  document.getElementById('title').textContent = title;
  rebuildBlockOptions();
  render(blockFilter.value);
}}
async function loadModel() {{
  const status = document.getElementById('status');
  const q = new URLSearchParams();
  const m = document.getElementById('f-model').value.trim();
  const b = document.getElementById('f-batch').value;
  const s = document.getElementById('f-seq').value;
  if (m) q.set('model', m);
  if (b) q.set('batch', b);
  if (s) q.set('seq', s);
  status.textContent = 'building…';
  try {{
    const r = await fetch('/graph?' + q.toString());
    const j = await r.json();
    if (j.error) {{
      status.textContent = 'build failed';
      document.getElementById('detail-content').innerHTML =
        `<div class="label">error</div><pre>${{j.error}}</pre>`;
      sidebar.classList.remove('collapsed');
      return;
    }}
    setData(j.graph, j.title);
  }} catch (e) {{
    status.textContent = 'no server — run plowc viz in serve mode';
  }}
}}
document.getElementById('f-load').onclick = loadModel;
document.getElementById('f-model').addEventListener('keydown', e => {{ if (e.key === 'Enter') loadModel(); }});
document.getElementById('f-batch').addEventListener('keydown', e => {{ if (e.key === 'Enter') loadModel(); }});
document.getElementById('f-seq').addEventListener('keydown', e => {{ if (e.key === 'Enter') loadModel(); }});

function visibleNodes(sel) {{
  if (sel === 'all') return DATA.nodes;
  if (sel === 'globals') return DATA.nodes.filter(n => n.block === null);
  const bl = DATA.blocks[parseInt(sel)];
  return DATA.nodes.filter(n => n.block === bl);
}}

let cy = null;

// Toolbar: block stepping, sidebar toggle, resize.
function stepBlock(dir) {{
  const i = blockFilter.selectedIndex + dir;
  if (i >= 0 && i < blockFilter.options.length) {{
    blockFilter.selectedIndex = i;
    render(blockFilter.value);
  }}
}}
document.getElementById('prev-block').onclick = () => stepBlock(-1);
document.getElementById('next-block').onclick = () => stepBlock(1);
const sidebar = document.getElementById('sidebar');
document.getElementById('toggle-sidebar').onclick = () => {{
  sidebar.classList.toggle('collapsed');
  if (cy) cy.resize();
}};
if (window.innerWidth < 900) sidebar.classList.add('collapsed');
window.addEventListener('resize', () => {{ if (cy) cy.resize(); }});

// Cytoscape canvas styles cannot read CSS variables — the literals below are
// the same infervisor.ai tokens as the :root block.
const STYLE = [
  {{ selector: 'node', style: {{
    'label': 'data(label)',
    'text-wrap': 'wrap',
    'text-valign': 'center',
    'color': '#e7e9ec',
    'font-size': '11px',
    'font-family': 'JetBrains Mono, Menlo, monospace',
    'width': 'label', 'height': 'label',
    'padding': '7px',
    'shape': 'round-rectangle',
  }} }},
  {{ selector: 'node.op', style: {{
    'background-color': 'data(bg)',
    'border-color': 'data(border)',
    'border-width': 1.5,
  }} }},
  {{ selector: 'node.weight', style: {{
    'background-color': '#16130b', 'border-color': '#6d5a2f',
    'border-width': 1, 'font-size': '9px', 'color': '#b8a26b',
    'padding': '4px',
  }} }},
  {{ selector: 'node.input', style: {{
    'background-color': '#0e1820', 'border-color': '#5a9fd4',
    'border-width': 1.5,
  }} }},
  {{ selector: 'node.act', style: {{
    'background-color': '#101318', 'background-opacity': 0.55,
    'border-color': '#3a424c', 'border-width': 1, 'border-style': 'dashed',
    'font-size': '9px', 'color': '#8b929c', 'padding': '4px',
  }} }},
  {{ selector: 'node:selected', style: {{
    'border-color': '#8fcaf4', 'border-width': 3,
  }} }},
  {{ selector: 'edge', style: {{
    'width': 1.2, 'line-color': '#3a424c',
    'target-arrow-color': '#3a424c', 'target-arrow-shape': 'triangle',
    'arrow-scale': 0.75,
    'curve-style': 'taxi',
    'taxi-direction': 'downward',
    'taxi-turn': '30%',
    'taxi-turn-min-distance': 8,
  }} }},
  {{ selector: 'edge.wedge', style: {{
    'curve-style': 'bezier',
    'line-style': 'dashed', 'width': 1,
    'line-color': '#6d5a2f', 'target-arrow-color': '#6d5a2f',
  }} }},
  {{ selector: '.faded', style: {{ 'opacity': 0.12 }} }},
  {{ selector: 'edge.hl', style: {{
    'line-color': '#5a9fd4', 'target-arrow-color': '#5a9fd4', 'width': 2,
  }} }},
];

// Full-fit shrinks a deep graph until labels are unreadable. If everything
// fits at a readable zoom, use it; otherwise fit the WIDTH (clamped) and
// anchor to the top, where reading a layered DAG starts.
function smartFit() {{
  const bb = cy.elements().boundingBox();
  const w = cy.width(), h = cy.height(), pad = 30;
  const zFull = Math.min((w - 2 * pad) / bb.w, (h - 2 * pad) / bb.h);
  if (zFull >= 0.3) {{
    cy.fit(undefined, pad);
    return;
  }}
  const z = Math.min(Math.max((w - 2 * pad) / bb.w, 0.25), 1.2);
  cy.zoom(z);
  cy.pan({{
    x: w / 2 - z * (bb.x1 + bb.x2) / 2,
    y: pad - z * bb.y1,
  }});
}}

function render(sel) {{
  const status = document.getElementById('status');
  status.textContent = 'laying out…';
  // Let the status line paint before the synchronous layout blocks the thread.
  setTimeout(() => build(sel), 20);
}}

function build(sel) {{
  const visible = visibleNodes(sel);
  const status = document.getElementById('status');

  // BIPARTITE dataflow: tensors are nodes too. weight + input → op → activation
  // → next op, so the picture shows the data, not just the schedule of ops.
  const elements = [];
  const added = new Set();
  function addTensor(tid) {{
    if (added.has(tid)) return;
    added.add(tid);
    const t = DATA.tensors[tid];
    const kind = t.origin === 'weight' ? 'weight' : t.origin === 'input' ? 'input' : 'act';
    const label = kind === 'act' ? t.shape : short(t.name || 'input') + '\n' + t.shape;
    elements.push({{ data: {{ id: 't' + tid, tid: tid, label: label }}, classes: kind }});
  }}

  visible.forEach(n => {{
    const name = opName(n);
    const [bg, border] = colorFor(n.op);
    elements.push({{ data: {{
      id: 'n' + n.id, nid: n.id, bg: bg, border: border,
      label: name ? n.op + '\n' + name : n.op,
    }}, classes: 'op' }});
    addTensor(n.output);
    elements.push({{ data: {{ source: 'n' + n.id, target: 't' + n.output }} }});
  }});
  visible.forEach(n => {{
    // A tensor produced outside the view still appears, as a boundary node.
    new Set(n.inputs).forEach(tid => {{
      addTensor(tid);
      const w = DATA.tensors[tid].origin === 'weight';
      elements.push({{
        data: {{ source: 't' + tid, target: 'n' + n.id }},
        classes: w ? 'wedge' : undefined,
      }});
    }});
  }});

  if (cy) cy.destroy();
  cy = cytoscape({{
    container: document.getElementById('graph-container'),
    elements: elements,
    style: STYLE,
    layout: {{ name: 'preset' }},
    wheelSensitivity: 0.2,
    minZoom: 0.02, maxZoom: 4,
  }});

  // Two-phase layout. Weights are NOT ranked by dagre: as pure rank-0 sources
  // they get flung to the top of the graph, far from the op that reads them —
  // that was the sprawl. Dagre lays out ops + dataflow only; each weight is
  // then pinned beside its consuming op.
  const flow = cy.elements().filter(el => el.isNode()
    ? !el.hasClass('weight')
    : !el.hasClass('wedge'));
  // nodeSep leaves room in each rank for the weight pinned beside its op —
  // dagre never sees the weight nodes, so the spacing has to reserve for them.
  const lay = flow.layout({{
    name: 'dagre', rankDir: 'TB', rankSep: 42, nodeSep: 105, edgeSep: 10,
  }});
  lay.one('layoutstop', () => {{
    cy.batch(() => {{
      const stacked = {{}};
      cy.nodes('.weight').forEach(w => {{
        const op = w.outgoers('node').first();
        if (op.length === 0) return;
        const i = stacked[op.id()] || 0;
        stacked[op.id()] = i + 1;
        w.position({{
          x: op.position('x') - op.width() / 2 - w.width() / 2 - 12,
          y: op.position('y') + i * (w.height() + 8),
        }});
      }});
    }});
    status.textContent = `${{visible.length}} ops · ${{added.size}} tensors`;
    smartFit();
  }});
  lay.run();

  // Hover: tooltip + light up the node's neighborhood. Click pins the same
  // detail into the sidebar. Highlighting is skipped on huge views — toggling
  // classes on 10k elements per mouse event is its own jank.
  const tip = document.getElementById('tooltip');
  const highlight = cy.elements().length < 3000;
  cy.on('mouseover', 'node', ev => {{
    tip.innerHTML = tipHtml(ev.target);
    tip.style.display = 'block';
    if (highlight) {{
      const hood = ev.target.closedNeighborhood();
      cy.batch(() => {{
        cy.elements().not(hood).addClass('faded');
        hood.edges().addClass('hl');
      }});
    }}
  }});
  cy.on('mouseout', 'node', () => {{
    tip.style.display = 'none';
    if (highlight) cy.batch(() => cy.elements().removeClass('faded hl'));
  }});
  cy.on('mousemove', 'node', ev => {{
    const e = ev.originalEvent;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 420) + 'px';
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - 220) + 'px';
  }});
  cy.on('tap', 'node', ev => {{
    document.getElementById('detail-content').innerHTML = tipHtml(ev.target);
    sidebar.classList.remove('collapsed');
  }});
}}

const nameOf = n => {{ const s = opName(n); return s ? n.op + ' · ' + s : n.op; }};

function tipHtml(el) {{
  const d = el.data();
  if (d.nid !== undefined) {{
    const n = DATA.nodes[d.nid];
    const outT = DATA.tensors[n.output];
    let h = `<div class="label">op</div>${{nameOf(n)}}`;
    h += `<div class="label">block</div>${{n.block || '(global)'}}`;
    const w = n.inputs.map(t => DATA.tensors[t]).filter(t => t.origin === 'weight');
    if (w.length) h += `<div class="label">weights</div>` +
      w.map(t => `${{t.name}} ${{t.shape}} ${{t.dtype}}`).join('<br>');
    h += `<div class="label">output</div>${{outT.shape}} ${{outT.dtype}}`;
    if (Object.keys(n.detail).length > 0)
      h += `<div class="label">params</div><pre>${{JSON.stringify(n.detail, null, 1)}}</pre>`;
    return h;
  }}
  const t = DATA.tensors[d.tid];
  const kind = t.origin === 'weight' ? 'weight' : t.origin === 'input' ? 'model input' : 'activation';
  let h = `<div class="label">${{kind}}</div>${{t.name || '(anonymous)'}}`;
  h += `<div class="label">shape · dtype</div>${{t.shape}} · ${{t.dtype}}`;
  if (t.origin && t.origin.node !== undefined)
    h += `<div class="label">produced by</div>${{nameOf(DATA.nodes[t.origin.node])}}`;
  const cs = consumers[d.tid];
  if (cs.length) h += `<div class="label">consumed by</div>` +
    cs.slice(0, 8).map(id => nameOf(DATA.nodes[id])).join('<br>') +
    (cs.length > 8 ? `<br>… +${{cs.length - 8}} more` : '');
  return h;
}}

// Legend: op palette plus the three tensor-node kinds.
let legendHtml = '';
for (const [op, [bg, border]] of Object.entries(OP_COLORS)) {{
  legendHtml += `<div class="legend-item"><div class="legend-swatch" style="background:${{bg}};border:1px solid ${{border}}"></div><span>${{op}}</span></div>`;
}}
for (const [kind, [bg, border]] of [['weight', ['#16130b','#6d5a2f']], ['model input', ['#0e1820','#5a9fd4']], ['activation', ['#101318','#3a424c']]]) {{
  legendHtml += `<div class="legend-item"><div class="legend-swatch" style="background:${{bg}};border:1px dashed ${{border}}"></div><span>${{kind}}</span></div>`;
}}
document.getElementById('legend').innerHTML = legendHtml;

render(blockFilter.value);
</script>
</body>
</html>"##,
        title = title,
        source = source,
        json = json,
    )
}
