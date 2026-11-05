//! Asset dumping capabilities for offline visualization, documentation, and embedding.

use super::html::render_html;
use super::svg::bundle_to_svg;
use super::types::VizModelBundle;
use std::fs;
use std::path::Path;

/// Dump visualization assets into a target directory.
/// `script_base`: see [`render_html`].
pub fn dump_assets(
    bundle: &VizModelBundle,
    dir: &Path,
    script_base: Option<&str>,
) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("cannot create dir {}: {e}", dir.display()))?;

    // 1. Model data JSON
    let model_json = serde_json::to_string_pretty(bundle)
        .map_err(|e| format!("cannot serialize model bundle: {e}"))?;
    fs::write(dir.join("model.json"), &model_json)
        .map_err(|e| format!("cannot write model.json: {e}"))?;

    // 2. Pipeline summary JSON
    let pipeline_json = serde_json::to_string_pretty(&serde_json::json!({
        "schema": bundle.schema,
        "title": bundle.title,
        "source": bundle.source,
        "modality": bundle.modality,
        "default_network": bundle.default_network,
        "provenance": bundle.provenance,
        "notes": bundle.notes,
        "networks": bundle.networks.iter().map(|n| serde_json::json!({
            "id": n.id,
            "title": n.title,
            "stages": n.stages,
            "summary": n.graph.summary,
            "checkpoint": n.checkpoint,
        })).collect::<Vec<_>>(),
    }))
    .map_err(|e| format!("cannot serialize pipeline summary: {e}"))?;
    fs::write(dir.join("pipeline.json"), &pipeline_json)
        .map_err(|e| format!("cannot write pipeline.json: {e}"))?;

    // 3. Pipeline vector SVG
    let svg = bundle_to_svg(bundle);
    fs::write(dir.join("pipeline.svg"), &svg)
        .map_err(|e| format!("cannot write pipeline.svg: {e}"))?;

    // 4. Standalone HTML
    let index_html = render_html(bundle, false, script_base);
    fs::write(dir.join("index.html"), &index_html)
        .map_err(|e| format!("cannot write index.html: {e}"))?;

    // 5. Embeddable HTML
    let embed_html = render_html(bundle, true, script_base);
    fs::write(dir.join("embed.html"), &embed_html)
        .map_err(|e| format!("cannot write embed.html: {e}"))?;

    Ok(())
}
