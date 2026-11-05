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
    let packet = net.kind == "packet";
    let mut handoffs = std::collections::BTreeSet::new();
    if packet {
        for node in &net.graph.nodes {
            let target = net
                .stages
                .iter()
                .position(|s| node.stage.as_deref() == Some(&s.id));
            for &input in &node.inputs {
                if let crate::Origin::Node(producer) = net.graph.tensors[input.0 as usize].origin {
                    let source = net.stages.iter().position(|s| {
                        net.graph.nodes[producer.0 as usize].stage.as_deref() == Some(&s.id)
                    });
                    if let (Some(from), Some(to)) = (source, target) {
                        if from != to {
                            handoffs.insert((from, to));
                        }
                    }
                }
            }
        }
    }
    let n = net.stages.len().max(1);
    let width = MARGIN * 2 + n * CARD_W + (n - 1) * GAP;
    let height = TOP + CARD_H + MARGIN + handoffs.len() * 12;
    let total = summary.total_params.max(1) as f64;

    let mut svg = String::new();
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="'JetBrains Mono', ui-monospace, Menlo, monospace">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 1L10 5L0 9z" fill="#3d6e9c"/></marker></defs>
<rect width="100%" height="100%" fill="#0a0b0d"/>
<text x="{MARGIN}" y="44" fill="#e7e9ec" font-size="20" font-weight="600">{title} <tspan fill="#5a9fd4" font-size="12">{badge}</tspan></text>
<text x="{MARGIN}" y="70" fill="#8b929c" font-size="12">{net_title} · {stats}</text>
"##,
        title = esc(&bundle.title),
        badge = bundle.modality.badge(),
        net_title = esc(&net.title),
        stats = if packet {
            format!(
                "{} declared buffers · parameter count not encoded",
                format_bytes(summary.total_weight_bytes)
            )
        } else {
            format!(
                "{} params · {} · {} blocks",
                format_params(summary.total_params),
                format_bytes(summary.total_weight_bytes),
                summary.num_blocks
            )
        },
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
<text x="{tx}" y="{y3}" fill="#e7e9ec" font-size="11">{metric}</text>
{bars}
<text x="{tx}" y="{y5}" fill="#565d67" font-size="9">in  <tspan fill="#8b929c">{inp}</tspan></text>
<text x="{tx}" y="{y6}" fill="#565d67" font-size="9">out <tspan fill="#8b929c">{out}</tspan></text>
</g>
"##,
            tx = x + 12,
            tx2 = x + 30,
            y1 = TOP + 24,
            y2 = TOP + 44,
            y3 = TOP + 66,
            y5 = TOP + 102,
            y6 = TOP + 118,
            num = i + 1,
            name = esc(&truncate(&st.name, 22)),
            desc = esc(&truncate(&st.description, 36)),
            metric = if packet {
                "Compiled execution".into()
            } else {
                format!(
                    "{} params · {:.1}%",
                    format_params(st.param_count),
                    share * 100.0
                )
            },
            bars = if packet {
                String::new()
            } else {
                format!(
                    r##"<rect x="{}" y="{}" width="{}" height="4" rx="2" fill="#1d2128"/><rect x="{}" y="{}" width="{bar}" height="4" rx="2" fill="#5a9fd4"/>"##,
                    x + 12,
                    TOP + 76,
                    CARD_W - 24,
                    x + 12,
                    TOP + 76
                )
            },
            inp = esc(&io(&st.input_shapes)),
            out = esc(&io(&st.output_shapes)),
        );
        if !packet && i + 1 < net.stages.len() {
            let y = TOP + CARD_H / 2;
            let _ = writeln!(
                svg,
                r##"<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#3d6e9c" stroke-width="1.5" marker-end="url(#arrow)"/>"##,
                x1 = x + CARD_W + 4,
                x2 = x + CARD_W + GAP - 4,
            );
        }
    }
    for (index, (from, to)) in handoffs.into_iter().enumerate() {
        let x1 = MARGIN + from * (CARD_W + GAP) + CARD_W / 2;
        let x2 = MARGIN + to * (CARD_W + GAP) + CARD_W / 2;
        let y = TOP + CARD_H;
        let route = y + 12 + index * 12;
        let _ = writeln!(
            svg,
            r##"<path data-from="{from}" data-to="{to}" d="M{x1} {y} V{route} H{x2} V{y}" fill="none" stroke="#3d6e9c" stroke-width="1.5" marker-end="url(#arrow)"/>"##
        );
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
