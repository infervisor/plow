//! Static SVG of a network's stages, for docs and READMEs.

use super::types::VizModelBundle;
use std::fmt::Write;

const CARD_W: usize = 240;
const CARD_H: usize = 132;
const GAP: usize = 40;
const MARGIN: usize = 32;
const TOP: usize = 96;

/// Stage diagram of the bundle's default network.
pub fn bundle_to_svg(bundle: &VizModelBundle) -> String {
    let net = bundle.default_network();
    let summary = &net.graph.summary;
    let n = net.stages.len().max(1);
    let width = MARGIN * 2 + n * CARD_W + (n - 1) * GAP;
    let height = TOP + CARD_H + MARGIN;
    let total = summary.total_params.max(1) as f64;

    let mut svg = String::new();
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="'JetBrains Mono', ui-monospace, Menlo, monospace">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 1L10 5L0 9z" fill="#3d6e9c"/></marker></defs>
<rect width="100%" height="100%" fill="#0a0b0d"/>
<text x="{MARGIN}" y="44" fill="#e7e9ec" font-size="20" font-weight="600">{title} <tspan fill="#5a9fd4" font-size="12">{badge}</tspan></text>
<text x="{MARGIN}" y="70" fill="#8b929c" font-size="12">{net_title} · {params} params · {size} · {blocks} blocks</text>
"##,
        title = esc(&bundle.title),
        badge = bundle.modality.badge(),
        net_title = esc(&net.title),
        params = format_params(summary.total_params),
        size = format_bytes(summary.total_weight_bytes),
        blocks = summary.num_blocks,
    );

    for (i, st) in net.stages.iter().enumerate() {
        let x = MARGIN + i * (CARD_W + GAP);
        let share = st.param_count as f64 / total;
        let bar = ((CARD_W - 24) as f64 * share).round().max(1.0) as usize;
        let io = |v: &[String]| v.first().map_or("—".to_string(), |s| truncate(s, 24));
        let _ = write!(
            svg,
            r##"<g>
<rect x="{x}" y="{TOP}" width="{CARD_W}" height="{CARD_H}" rx="8" fill="#101318" stroke="#272c34"/>
<text x="{tx}" y="{y1}" fill="#5a9fd4" font-size="11">{num}</text>
<text x="{tx2}" y="{y1}" fill="#e7e9ec" font-size="13" font-weight="600">{name}</text>
<text x="{tx}" y="{y2}" fill="#8b929c" font-size="10">{desc}</text>
<text x="{tx}" y="{y3}" fill="#e7e9ec" font-size="11">{params} <tspan fill="#565d67">params · {pct:.1}%</tspan></text>
<rect x="{tx}" y="{y4}" width="{bw}" height="4" rx="2" fill="#1d2128"/>
<rect x="{tx}" y="{y4}" width="{bar}" height="4" rx="2" fill="#5a9fd4"/>
<text x="{tx}" y="{y5}" fill="#565d67" font-size="9">in  <tspan fill="#8b929c">{inp}</tspan></text>
<text x="{tx}" y="{y6}" fill="#565d67" font-size="9">out <tspan fill="#8b929c">{out}</tspan></text>
</g>
"##,
            tx = x + 12,
            tx2 = x + 30,
            y1 = TOP + 24,
            y2 = TOP + 44,
            y3 = TOP + 66,
            y4 = TOP + 76,
            y5 = TOP + 102,
            y6 = TOP + 118,
            bw = CARD_W - 24,
            num = i + 1,
            name = esc(&truncate(&st.name, 22)),
            desc = esc(&truncate(&st.description, 36)),
            params = format_params(st.param_count),
            pct = share * 100.0,
            inp = esc(&io(&st.input_shapes)),
            out = esc(&io(&st.output_shapes)),
        );
        if i + 1 < net.stages.len() {
            let y = TOP + CARD_H / 2;
            let _ = writeln!(
                svg,
                r##"<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#3d6e9c" stroke-width="1.5" marker-end="url(#arrow)"/>"##,
                x1 = x + CARD_W + 4,
                x2 = x + CARD_W + GAP - 4,
            );
        }
    }
    svg.push_str("</svg>\n");
    svg
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut t: String = s.chars().take(max - 1).collect();
    t.push('…');
    t
}

pub(crate) fn format_params(p: u64) -> String {
    match p {
        1_000_000_000.. => format!("{:.2}B", p as f64 / 1e9),
        1_000_000.. => format!("{:.1}M", p as f64 / 1e6),
        1_000.. => format!("{:.1}K", p as f64 / 1e3),
        _ => p.to_string(),
    }
}

pub(crate) fn format_bytes(b: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    match b {
        0x4000_0000.. => format!("{:.2} GiB", b as f64 / GIB),
        0x10_0000.. => format!("{:.1} MiB", b as f64 / MIB),
        0x400.. => format!("{:.1} KiB", b as f64 / 1024.0),
        _ => format!("{b} B"),
    }
}
