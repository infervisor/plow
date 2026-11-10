//! Self-contained HTML viewer (`viewer.html`) with the bundle inlined.

use super::types::{OpCategory, VizModelBundle};

const VIEWER: &str = include_str!("viewer.html");

/// Graph libraries, pinned. `script_base` replaces the CDN with a host that serves these files
/// under the same names (a site self-hosting them from npm).
const SCRIPTS: [(&str, &str); 3] = [
    (
        "cytoscape.min.js",
        "https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js",
    ),
    (
        "dagre.min.js",
        "https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js",
    ),
    (
        "cytoscape-dagre.js",
        "https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js",
    ),
];

/// Render the viewer. `embed` drops standalone chrome (theme toggle) for an iframe on a model
/// page; the page sets the theme with `?theme=light|dark` or a `plow-viz:theme` message.
pub fn render_html(bundle: &VizModelBundle, embed: bool, script_base: Option<&str>) -> String {
    let scripts: String = SCRIPTS
        .iter()
        .map(|(file, cdn)| {
            let src = match script_base {
                Some(base) => format!("{}/{file}", base.trim_end_matches('/')),
                None => cdn.to_string(),
            };
            format!("<script src=\"{}\"></script>\n", html_escape(&src))
        })
        .collect();
    let categories: serde_json::Map<String, serde_json::Value> = OpCategory::ALL
        .iter()
        .map(|c| {
            let (dark, light) = c.colors();
            let key = serde_json::to_value(c).expect("category serializes");
            let key = key.as_str().expect("unit variant").to_string();
            (
                key,
                serde_json::json!({ "label": c.label(), "dark": dark, "light": light }),
            )
        })
        .collect();
    VIEWER
        .replace("__TITLE__", &html_escape(&bundle.title))
        .replace(
            "__MODE__",
            if embed {
                "mode-embed"
            } else {
                "mode-standalone"
            },
        )
        .replace("__SCRIPTS__", &scripts)
        .replace("__CATEGORIES__", &script_json(&categories))
        .replace("__BUNDLE__", &script_json(bundle))
}

/// JSON safe inside a `<script>` element.
fn script_json(v: &impl serde::Serialize) -> String {
    serde_json::to_string(v)
        .expect("viz bundle serializes")
        .replace("</", "<\\/")
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
