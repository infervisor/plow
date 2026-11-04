//! HTML viewer generator for standalone and embeddable network visualization.

use super::types::VizModelBundle;

/// Render self-contained HTML for the visualizer.
/// When `embed` is true, renders an embeddable widget (no model load form, compact chrome, postMessage enabled).
pub fn render_html(bundle: &VizModelBundle, embed: bool) -> String {
    let json = serde_json::to_string(bundle).expect("bundle serialization cannot fail");
    let title = html_escape(&bundle.title);
    let source = html_escape(&bundle.source);
    let modality = bundle.modality.badge();

    let embed_class = if embed { "mode-embed" } else { "mode-standalone" };

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Plow Network Visualizer</title>
<style>
/* infervisor.ai design tokens */
:root {{
  --bg: #0a0b0d; --bg2: #0c0e12;
  --panel: #101318; --panel2: #0d0f13;
  --line: #1d2128; --line2: #272c34;
  --text: #e7e9ec; --muted: #8b929c; --dim: #565d67;
  --blue: #5a9fd4; --blue-dim: #3d6e9c; --blue-deep: #1F4E79; --stem: #8fcaf4;
  --green: #6cc0a0; --amber: #b8813a; --purple: #b07a9a;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; width: 100%; }}
body {{
  font-family: var(--mono);
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}}

/* Top Navigation / Header */
#header {{
  flex: 0 0 50px;
  background: var(--panel2);
  display: flex;
  align-items: center;
  padding: 0 16px;
  z-index: 100;
  border-bottom: 1px solid var(--line);
  gap: 12px;
}}
#header h1 {{
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.hl-logo {{ color: var(--blue); font-weight: 700; letter-spacing: 0.5px; }}
.modality-badge {{
  font-size: 10px;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 4px;
  background: rgba(90, 159, 212, 0.15);
  border: 1px solid var(--blue-dim);
  color: var(--blue);
  text-transform: uppercase;
}}
.metrics-strip {{
  display: flex;
  gap: 14px;
  margin-left: 10px;
  font-size: 11px;
  color: var(--muted);
}}
.metric-val {{ color: var(--text); font-weight: 600; }}
.header-controls {{ margin-left: auto; display: flex; gap: 8px; align-items: center; }}

/* Formbar (Standalone only) */
#formbar {{
  flex: 0 0 38px;
  background: var(--panel);
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 16px;
  border-bottom: 1px solid var(--line);
  font-size: 12px;
  color: var(--muted);
}}
.mode-embed #formbar {{ display: none; }}
#formbar input#f-model {{ flex: 1; max-width: 420px; }}
#formbar input[type=number] {{ width: 72px; }}
#formbar label {{ display: flex; align-items: center; gap: 5px; color: var(--dim); }}

/* Pipeline Stage Track */
#stage-bar {{
  flex: 0 0 42px;
  background: var(--bg2);
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 8px;
  overflow-x: auto;
  white-space: nowrap;
  scrollbar-width: thin;
}}
.stage-chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 4px;
  background: var(--panel);
  border: 1px solid var(--line2);
  font-size: 11px;
  color: var(--muted);
  cursor: pointer;
  transition: .15s;
}}
.stage-chip:hover, .stage-chip.active {{
  border-color: var(--blue-dim);
  color: var(--text);
  background: rgba(90, 159, 212, 0.1);
}}
.stage-chip .stage-num {{
  color: var(--blue);
  font-weight: 600;
  font-size: 10px;
}}
.stage-arrow {{ color: var(--dim); font-size: 10px; }}

/* Controls & Inputs */
button {{
  background: transparent;
  border: 1px solid var(--line2);
  color: var(--text);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--mono);
  transition: .15s;
}}
button:hover {{ border-color: var(--blue-dim); color: var(--blue); }}
button.active {{ background: rgba(90, 159, 212, 0.15); border-color: var(--blue); color: var(--blue); }}
select, input {{
  background: var(--panel);
  border: 1px solid var(--line2);
  color: var(--text);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-family: var(--mono);
}}
select:focus, input:focus {{ outline: none; border-color: var(--blue-dim); }}
#status {{ color: var(--dim); font-size: 11px; white-space: nowrap; }}
#search-input {{ width: 140px; }}

/* Main layout */
#main {{ flex: 1; display: flex; min-height: 0; position: relative; }}
#graph-container {{ flex: 1; min-width: 0; position: relative; background: var(--bg); }}
#sidebar {{
  flex: 0 0 340px;
  background: var(--panel2);
  border-left: 1px solid var(--line);
  overflow-y: auto;
  padding: 14px;
  font-size: 12px;
}}
#sidebar.collapsed {{ display: none; }}
#sidebar h2 {{ color: var(--blue); font-size: 13px; margin-bottom: 8px; letter-spacing: 0.3px; }}
#sidebar .label {{ color: var(--dim); font-size: 10px; text-transform: uppercase; margin-top: 8px; letter-spacing: 0.5px; }}
#sidebar pre {{ white-space: pre-wrap; font-family: var(--mono); font-size: 11px; }}
.trace-actions {{ display: flex; gap: 6px; margin-top: 10px; }}

/* Legend */
.legend {{
  position: absolute;
  bottom: 12px;
  left: 12px;
  background: var(--panel);
  border: 1px solid var(--line2);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 10px;
  z-index: 50;
  color: var(--muted);
  max-width: 380px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}}
.legend-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 12px; margin-top: 4px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0; }}

/* Tooltip */
#tooltip {{
  position: fixed;
  display: none;
  background: var(--panel);
  border: 1px solid var(--line2);
  border-radius: 6px;
  padding: 8px 11px;
  font-size: 11px;
  z-index: 200;
  max-width: 420px;
  pointer-events: none;
  font-family: var(--mono);
  box-shadow: 0 6px 24px rgba(0,0,0,0.6);
}}
#tooltip .label {{ color: var(--dim); font-size: 9px; text-transform: uppercase; margin-top: 6px; letter-spacing: 0.5px; }}
#tooltip .label:first-child {{ margin-top: 0; }}
#tooltip pre {{ white-space: pre-wrap; }}
</style>
</head>
<body class="{embed_class}">

<div id="header">
  <h1>
    <span class="hl-logo">PLOW</span>
    <span class="hl-logo" style="color:var(--text);font-weight:400">viz</span>
    <span class="modality-badge" id="m-badge">{modality}</span>
    <span id="title" style="color:var(--text);font-weight:500">{title}</span>
  </h1>

  <div class="metrics-strip" id="metrics-strip">
    <span>Params: <span class="metric-val" id="metric-params">-</span></span>
    <span>Weights: <span class="metric-val" id="metric-weights">-</span></span>
    <span>Layers: <span class="metric-val" id="metric-layers">-</span></span>
  </div>

  <div class="header-controls">
    <span id="status"></span>
    <select id="network-select" title="Switch sub-network" style="display:none"></select>
    <button id="prev-block" title="Previous block">◀</button>
    <select id="block-filter"></select>
    <button id="next-block" title="Next block">▶</button>
    <input id="search-input" placeholder="Search node/tensor…">
    <button onclick="smartFit()" title="Fit view">Fit</button>
    <button onclick="cy && cy.zoom(cy.zoom() * 1.3)">+</button>
    <button onclick="cy && cy.zoom(cy.zoom() / 1.3)">−</button>
    <button id="toggle-sidebar" title="Toggle inspector">☰</button>
  </div>
</div>

<div id="formbar">
  <input id="f-model" placeholder="HF model id, checkpoint dir, or config.json path" value="{source}">
  <label>batch <input id="f-batch" type="number" min="1" placeholder="B"></label>
  <label>seq <input id="f-seq" type="number" min="1" placeholder="S"></label>
  <button id="f-load">Reload</button>
</div>

<div id="stage-bar"></div>

<div id="main">
  <div id="graph-container"></div>
  <div id="sidebar">
    <h2>Inspector</h2>
    <p style="color:var(--dim)">Click any op or tensor node to inspect</p>
    <div id="detail-content"></div>
  </div>
</div>

<div class="legend" id="legend"></div>
<div id="tooltip"></div>

<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<script>
let BUNDLE = {json};
let CURRENT_NETWORK = BUNDLE.default_network;
let DATA = BUNDLE.networks[CURRENT_NETWORK] ? BUNDLE.networks[CURRENT_NETWORK].graph : BUNDLE.networks[Object.keys(BUNDLE.networks)[0]].graph;

const OP_CATEGORIES = {{
  'compute': ['#0e1820', '#3d6e9c'],
  'attention': ['#132c47', '#8fcaf4'],
  'convolution': ['#10233a', '#5a9fd4'],
  'normalization': ['#14171c', '#8b929c'],
  'activation': ['#0f1d17', '#6cc0a0'],
  'routing': ['#231708', '#b8813a'],
  'structural': ['#101318', '#3a424c'],
  'audio_codec': ['#1e1424', '#b07a9a'],
  'other': ['#101318', '#3a424c'],
}};

let consumers = [];
function reindex() {{
  consumers = DATA.tensors.map(() => []);
  DATA.nodes.forEach(n => n.inputs.forEach(tid => {{
    if (consumers[tid]) consumers[tid].push(n.id);
  }}));
}}
reindex();

function updateMetrics() {{
  const s = DATA.summary || BUNDLE.summary;
  document.getElementById('metric-params').textContent = formatParams(s.total_params);
  document.getElementById('metric-weights').textContent = formatBytes(s.total_weight_bytes);
  document.getElementById('metric-layers').textContent = s.num_layers;
}}
updateMetrics();

function formatParams(p) {{
  if (!p) return '0';
  if (p >= 1e9) return (p / 1e9).toFixed(2) + 'B';
  if (p >= 1e6) return (p / 1e6).toFixed(1) + 'M';
  if (p >= 1e3) return (p / 1e3).toFixed(1) + 'K';
  return p.toString();
}}
function formatBytes(b) {{
  if (!b) return '0 B';
  if (b >= 1073741824) return (b / 1073741824).toFixed(2) + ' GiB';
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MiB';
  if (b >= 1024) return (b / 1024).toFixed(1) + ' KiB';
  return b + ' B';
}}

// Setup sub-network selector if multiple networks exist
const netSelect = document.getElementById('network-select');
const netKeys = Object.keys(BUNDLE.networks);
if (netKeys.length > 1) {{
  netSelect.style.display = 'inline-block';
  netSelect.innerHTML = '';
  netKeys.forEach(k => {{
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = BUNDLE.networks[k].title || k;
    if (k === CURRENT_NETWORK) opt.selected = true;
    netSelect.appendChild(opt);
  }});
  netSelect.onchange = () => {{
    CURRENT_NETWORK = netSelect.value;
    DATA = BUNDLE.networks[CURRENT_NETWORK].graph;
    reindex();
    updateMetrics();
    rebuildBlockOptions();
    render(blockFilter.value);
  }};
}}

// Pipeline stages bar
function renderStages() {{
  const bar = document.getElementById('stage-bar');
  bar.innerHTML = '';
  if (!BUNDLE.stages || BUNDLE.stages.length === 0) {{
    bar.style.display = 'none';
    return;
  }}
  bar.style.display = 'flex';
  BUNDLE.stages.forEach((st, idx) => {{
    const chip = document.createElement('div');
    chip.className = 'stage-chip';
    chip.innerHTML = `<span class="stage-num">${{idx + 1}}</span><span>${{st.name}}</span>`;
    chip.title = st.description;
    chip.onclick = () => {{
      document.querySelectorAll('.stage-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      if (st.subnetwork_name && BUNDLE.networks[st.subnetwork_name]) {{
        CURRENT_NETWORK = st.subnetwork_name;
        netSelect.value = CURRENT_NETWORK;
        DATA = BUNDLE.networks[CURRENT_NETWORK].graph;
        reindex();
        updateMetrics();
        rebuildBlockOptions();
      }}
      // Select stage block if exists
      render('all');
    }};
    bar.appendChild(chip);
    if (idx + 1 < BUNDLE.stages.length) {{
      const arr = document.createElement('span');
      arr.className = 'stage-arrow';
      arr.textContent = '➔';
      bar.appendChild(arr);
    }}
  }});
}}
renderStages();

// Block filter
const blockFilter = document.getElementById('block-filter');
function rebuildBlockOptions() {{
  blockFilter.innerHTML = '';
  const optAll = document.createElement('option');
  optAll.value = 'all';
  optAll.textContent = `All Blocks (${{DATA.nodes.length}} ops)`;
  blockFilter.appendChild(optAll);

  const globalCount = DATA.nodes.filter(n => !n.block).length;
  if (globalCount > 0) {{
    const g = document.createElement('option');
    g.value = 'globals';
    g.textContent = `(Global / Unbound: ${{globalCount}})`;
    blockFilter.appendChild(g);
  }}

  if (DATA.blocks) {{
    DATA.blocks.forEach((b, idx) => {{
      const o = document.createElement('option');
      o.value = idx.toString();
      o.textContent = b.label || b.id || `Block ${{idx}}`;
      blockFilter.appendChild(o);
    }});
  }}
  blockFilter.value = DATA.blocks && DATA.blocks.length > 0 ? '0' : 'all';
}}
rebuildBlockOptions();
blockFilter.onchange = () => render(blockFilter.value);

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

// Search filter
const searchInput = document.getElementById('search-input');
searchInput.addEventListener('input', () => {{
  const query = searchInput.value.trim().toLowerCase();
  if (!cy) return;
  if (!query) {{
    cy.elements().removeClass('faded hl');
    return;
  }}
  cy.batch(() => {{
    cy.elements().addClass('faded');
    const matched = cy.elements().filter(el => {{
      const d = el.data();
      const label = (d.label || '').toLowerCase();
      const id = (d.id || '').toLowerCase();
      return label.includes(query) || id.includes(query);
    }});
    matched.removeClass('faded').addClass('hl');
  }});
}});

let cy = null;

const STYLE = [
  {{ selector: 'node', style: {{
    'label': 'data(label)',
    'text-wrap': 'wrap',
    'text-valign': 'center',
    'color': '#e7e9ec',
    'font-size': '10px',
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
    'background-color': '#101318', 'background-opacity': 0.65,
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
  {{ selector: '.hl', style: {{
    'border-color': '#8fcaf4', 'line-color': '#5a9fd4',
    'target-arrow-color': '#5a9fd4', 'width': 2.2, 'opacity': 1,
  }} }},
];

function visibleNodes(sel) {{
  if (sel === 'all') return DATA.nodes;
  if (sel === 'globals') return DATA.nodes.filter(n => !n.block);
  const blk = DATA.blocks[parseInt(sel)];
  if (!blk) return DATA.nodes;
  return DATA.nodes.filter(n => n.block === blk.id || n.block === blk.label);
}}

function smartFit() {{
  if (!cy) return;
  const bb = cy.elements().boundingBox();
  const w = cy.width(), h = cy.height(), pad = 30;
  const zFull = Math.min((w - 2 * pad) / bb.w, (h - 2 * pad) / bb.h);
  if (zFull >= 0.3) {{
    cy.fit(undefined, pad);
    return;
  }}
  const z = Math.min(Math.max((w - 2 * pad) / bb.w, 0.22), 1.2);
  cy.zoom(z);
  cy.pan({{
    x: w / 2 - z * (bb.x1 + bb.x2) / 2,
    y: pad - z * bb.y1,
  }});
}}

function render(sel) {{
  const status = document.getElementById('status');
  status.textContent = 'laying out…';
  setTimeout(() => buildGraph(sel), 20);
}}

function buildGraph(sel) {{
  const visible = visibleNodes(sel);
  const status = document.getElementById('status');
  const elements = [];
  const added = new Set();

  function addTensor(tid) {{
    if (added.has(tid)) return;
    added.add(tid);
    const t = DATA.tensors[tid];
    if (!t) return;
    const isWeight = t.origin && (t.origin === 'weight' || t.origin.type === 'weight');
    const isInput = t.origin && (t.origin === 'input' || t.origin.type === 'input');
    const kind = isWeight ? 'weight' : isInput ? 'input' : 'act';
    const label = kind === 'act' ? t.shape : (t.name ? t.name.split('.').pop() : 'input') + '\n' + t.shape;
    elements.push({{ data: {{ id: 't' + tid, tid: tid, label: label }}, classes: kind }});
  }}

  visible.forEach(n => {{
    const [bg, border] = OP_CATEGORIES[n.category] || OP_CATEGORIES['other'];
    elements.push({{ data: {{
      id: 'n' + n.id, nid: n.id, bg: bg, border: border,
      label: n.op + (n.block ? '\n(' + n.block + ')' : ''),
    }}, classes: 'op' }});
    addTensor(n.output);
    elements.push({{ data: {{ source: 'n' + n.id, target: 't' + n.output }} }});
  }});

  visible.forEach(n => {{
    new Set(n.inputs).forEach(tid => {{
      addTensor(tid);
      const t = DATA.tensors[tid];
      const isWeight = t && t.origin && (t.origin === 'weight' || t.origin.type === 'weight');
      elements.push({{
        data: {{ source: 't' + tid, target: 'n' + n.id }},
        classes: isWeight ? 'wedge' : undefined,
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

  const flow = cy.elements().filter(el => el.isNode() ? !el.hasClass('weight') : !el.hasClass('wedge'));
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
          x: op.position('x') - op.width() / 2 - w.width() / 2 - 14,
          y: op.position('y') + i * (w.height() + 8),
        }});
      }});
    }});
    status.textContent = `${{visible.length}} ops · ${{added.size}} tensors`;
    smartFit();
  }});
  lay.run();

  const tip = document.getElementById('tooltip');
  cy.on('mouseover', 'node', ev => {{
    tip.innerHTML = tipHtml(ev.target);
    tip.style.display = 'block';
  }});
  cy.on('mouseout', 'node', () => {{ tip.style.display = 'none'; }});
  cy.on('mousemove', 'node', ev => {{
    const e = ev.originalEvent;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 420) + 'px';
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - 220) + 'px';
  }});

  cy.on('tap', 'node', ev => {{
    inspectNode(ev.target);
  }});
}}

function inspectNode(target) {{
  const d = target.data();
  const content = document.getElementById('detail-content');
  content.innerHTML = tipHtml(target) + `
    <div class="trace-actions">
      <button onclick="traceAncestors('${{target.id()}}')">Trace Inputs</button>
      <button onclick="traceDescendants('${{target.id()}}')">Trace Outputs</button>
      <button onclick="cy.elements().removeClass('faded hl')">Clear</button>
    </div>
  `;
  sidebar.classList.remove('collapsed');

  // Broadcast postMessage in embed mode
  try {{
    window.parent.postMessage({{
      type: 'plow_viz_select',
      id: d.nid !== undefined ? d.nid : d.tid,
      kind: d.nid !== undefined ? 'node' : 'tensor',
    }}, '*');
  }} catch(e) {{}}
}}

window.traceAncestors = function(id) {{
  if (!cy) return;
  const node = cy.getElementById(id);
  const ancestors = node.predecessors().union(node);
  cy.batch(() => {{
    cy.elements().addClass('faded');
    ancestors.removeClass('faded').addClass('hl');
  }});
}};

window.traceDescendants = function(id) {{
  if (!cy) return;
  const node = cy.getElementById(id);
  const descendants = node.successors().union(node);
  cy.batch(() => {{
    cy.elements().addClass('faded');
    descendants.removeClass('faded').addClass('hl');
  }});
}};

function tipHtml(el) {{
  const d = el.data();
  if (d.nid !== undefined) {{
    const n = DATA.nodes[d.nid];
    const outT = DATA.tensors[n.output];
    let h = `<div class="label">Operation</div><b>${{n.op}}</b> (category: ${{n.category}})`;
    h += `<div class="label">Block / Stage</div>${{n.block || '(global)'}}`;
    const w = n.inputs.map(t => DATA.tensors[t]).filter(t => t && (t.origin === 'weight' || t.origin.type === 'weight'));
    if (w.length) {{
      h += `<div class="label">Weights</div>` +
        w.map(t => `${{t.name || 'weight'}} [${{t.shape}}] ${{t.dtype}} (${{formatBytes(t.bytes)}})`).join('<br>');
    }}
    if (outT) {{
      h += `<div class="label">Output Tensor</div>[${{outT.shape}}] ${{outT.dtype}}`;
    }}
    if (n.detail && Object.keys(n.detail).length > 0) {{
      h += `<div class="label">Parameters</div><pre>${{JSON.stringify(n.detail, null, 1)}}</pre>`;
    }}
    return h;
  }}
  const t = DATA.tensors[d.tid];
  if (!t) return 'Unknown tensor';
  const isWeight = t.origin && (t.origin === 'weight' || t.origin.type === 'weight');
  const isInput = t.origin && (t.origin === 'input' || t.origin.type === 'input');
  const kind = isWeight ? 'Weight' : isInput ? 'Graph Input' : 'Activation';
  let h = `<div class="label">${{kind}}</div><b>${{t.name || '(anonymous)'}}</b>`;
  h += `<div class="label">Shape · Dtype</div>[${{t.shape}}] · ${{t.dtype}} (${{formatBytes(t.bytes)}})`;
  if (t.origin && t.origin.node !== undefined) {{
    const prod = DATA.nodes[t.origin.node];
    if (prod) h += `<div class="label">Produced by</div>${{prod.op}} (#${{prod.id}})`;
  }}
  const cs = consumers[d.tid] || [];
  if (cs.length) {{
    h += `<div class="label">Consumed by</div>` +
      cs.slice(0, 6).map(id => DATA.nodes[id] ? `${{DATA.nodes[id].op}} (#${{id}})` : id).join('<br>') +
      (cs.length > 6 ? `<br>… +${{cs.length - 6}} more` : '');
  }}
  return h;
}}

// Build legend
let legHtml = '<div style="font-weight:600;margin-bottom:4px">Op Categories</div><div class="legend-grid">';
for (const [cat, [bg, border]] of Object.entries(OP_CATEGORIES)) {{
  legHtml += `<div class="legend-item"><div class="legend-swatch" style="background:${{bg}};border:1px solid ${{border}}"></div><span>${{cat.replace('_', ' ')}}</span></div>`;
}}
legHtml += '</div>';
document.getElementById('legend').innerHTML = legHtml;

// Reload model handler for standalone mode
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
    const r = await fetch('/bundle?' + q.toString());
    const bundle = await r.json();
    if (bundle.error) {{
      status.textContent = 'build failed';
      alert(bundle.error);
      return;
    }}
    BUNDLE = bundle;
    CURRENT_NETWORK = BUNDLE.default_network;
    DATA = BUNDLE.networks[CURRENT_NETWORK] ? BUNDLE.networks[CURRENT_NETWORK].graph : BUNDLE.networks[Object.keys(BUNDLE.networks)[0]].graph;
    reindex();
    updateMetrics();
    renderStages();
    rebuildBlockOptions();
    render(blockFilter.value);
  }} catch(e) {{
    status.textContent = 'offline mode';
  }}
}}
document.getElementById('f-load').onclick = loadModel;

render(blockFilter.value);
</script>
</body>
</html>"##,
        title = title,
        source = source,
        modality = modality,
        json = json,
        embed_class = embed_class,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
