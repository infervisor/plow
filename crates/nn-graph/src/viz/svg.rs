//! Vector SVG diagram generator for model architectures and pipeline stages.

use super::types::{ModelModality, VizModelBundle};

/// Generate an SVG diagram representing the model pipeline stages and topology.
pub fn bundle_to_svg(bundle: &VizModelBundle) -> String {
    let stages = &bundle.stages;
    let stage_count = stages.len().max(1);
    let stage_width = 240;
    let stage_height = 110;
    let stage_gap = 45;
    let margin_x = 40;
    let margin_y = 90;

    let total_width = margin_x * 2 + stage_count * stage_width + (stage_count - 1) * stage_gap;
    let total_height = 270;

    let badge_color = match bundle.modality {
        ModelModality::Asr => "#5a9fd4",
        ModelModality::Tts => "#b07a9a",
        ModelModality::Llm => "#6cc0a0",
        ModelModality::Multimodal => "#b8813a",
        ModelModality::Vision => "#8fcaf4",
        ModelModality::AudioCodec => "#3d6e9c",
    };

    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width} {total_height}" width="{total_width}" height="{total_height}" style="background:#0a0b0d; font-family:'JetBrains Mono', monospace;">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#3d6e9c"/>
    </marker>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.6"/>
    </filter>
  </defs>

  <!-- Title & Modality Header -->
  <text x="{margin_x}" y="42" fill="#e7e9ec" font-size="20" font-weight="600">{title}</text>
  <rect x="{badge_x}" y="24" width="75" height="24" rx="4" fill="{badge_color}" fill-opacity="0.15" stroke="{badge_color}" stroke-width="1.2"/>
  <text x="{badge_text_x}" y="40" fill="{badge_color}" font-size="11" font-weight="600" text-anchor="middle">{badge}</text>

  <!-- Metrics Bar -->
  <text x="{margin_x}" y="68" fill="#8b929c" font-size="12">
    Params: <tspan fill="#e7e9ec" font-weight="600">{params_fmt}</tspan>  |  Weights: <tspan fill="#e7e9ec" font-weight="600">{mem_fmt}</tspan>  |  Layers: <tspan fill="#e7e9ec" font-weight="600">{layers}</tspan>
  </text>
"##,
        total_width = total_width,
        total_height = total_height,
        margin_x = margin_x,
        title = html_escape(&bundle.title),
        badge_x = margin_x + bundle.title.len() * 12 + 15,
        badge_text_x = margin_x + bundle.title.len() * 12 + 15 + 37,
        badge_color = badge_color,
        badge = bundle.modality.badge(),
        params_fmt = format_params(bundle.summary.total_params),
        mem_fmt = format_bytes(bundle.summary.total_weight_bytes),
        layers = bundle.summary.num_layers,
    );

    // Draw stages
    for (i, stage) in stages.iter().enumerate() {
        let x = margin_x + i * (stage_width + stage_gap);
        let y = margin_y;

        let in_shape = stage.input_shapes.first().cloned().unwrap_or_else(|| "-".into());
        let out_shape = stage.output_shapes.first().cloned().unwrap_or_else(|| "-".into());

        svg.push_str(&format!(
            r##"
  <!-- Stage {i}: {name} -->
  <g filter="url(#shadow)">
    <rect x="{x}" y="{y}" width="{stage_width}" height="{stage_height}" rx="8" fill="#101318" stroke="#1d2128" stroke-width="1.5"/>
    <rect x="{x}" y="{y}" width="{stage_width}" height="28" rx="8" fill="#141820" stroke="#1d2128" stroke-width="1.5"/>
    <rect x="{x}" y="{y_split}" width="{stage_width}" height="6" fill="#141820"/>
    <text x="{x_pad}" y="{y_head}" fill="#5a9fd4" font-size="12" font-weight="600">{name}</text>
    <text x="{x_pad}" y="{y_desc}" fill="#8b929c" font-size="10">{desc}</text>
    <text x="{x_pad}" y="{y_in}" fill="#565d67" font-size="9">In: <tspan fill="#8b929c">{in_shape}</tspan></text>
    <text x="{x_pad}" y="{y_out}" fill="#565d67" font-size="9">Out: <tspan fill="#8b929c">{out_shape}</tspan></text>
  </g>
"##,
            i = i + 1,
            x = x,
            y = y,
            y_split = y + 22,
            stage_width = stage_width,
            stage_height = stage_height,
            x_pad = x + 12,
            y_head = y + 19,
            y_desc = y + 46,
            y_in = y + 70,
            y_out = y + 88,
            name = html_escape(&stage.name),
            desc = html_escape(&truncate_str(&stage.description, 32)),
            in_shape = html_escape(&truncate_str(&in_shape, 26)),
            out_shape = html_escape(&truncate_str(&out_shape, 26)),
        ));

        // Connecting arrow to next stage
        if i + 1 < stage_count {
            let arrow_start_x = x + stage_width;
            let arrow_start_y = y + stage_height / 2;
            let arrow_end_x = arrow_start_x + stage_gap - 2;

            svg.push_str(&format!(
                r##"  <line x1="{arrow_start_x}" y1="{arrow_start_y}" x2="{arrow_end_x}" y2="{arrow_start_y}" stroke="#3d6e9c" stroke-width="2" marker-end="url(#arrow)"/>
"##,
                arrow_start_x = arrow_start_x,
                arrow_start_y = arrow_start_y,
                arrow_end_x = arrow_end_x,
            ));
        }
    }

    svg.push_str("</svg>\n");
    svg
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}…", &s[..max_len.saturating_sub(1)])
    } else {
        s.to_string()
    }
}

fn format_params(p: u64) -> String {
    if p >= 1_000_000_000 {
        format!("{:.2}B", p as f64 / 1e9)
    } else if p >= 1_000_000 {
        format!("{:.1}M", p as f64 / 1e6)
    } else if p >= 1_000 {
        format!("{:.1}K", p as f64 / 1e3)
    } else {
        p.to_string()
    }
}

fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.2} GiB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1} MiB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}
