//! Build the checkpoint farm a bundle needs, on the target.
//!
//! The distribution never carries weights. What it carries is what plow
//! produced; the checkpoint comes from the operator's HuggingFace snapshot or a
//! local directory. `prepare` joins the two into the flat, sorted directory the
//! loader expects.
//!
//! `Checkpoint::open` globs `*.safetensors` in **one non-recursive sorted
//! directory** and lets the last writer win, which is how a derived sidecar
//! overrides tensors from the base shards — `model-idx-derived-*` sorts after
//! `model-000NN-*` because `'i' > '0'`. That ordering is load-bearing and
//! invisible, so it is asserted here rather than assumed.

use std::path::{Path, PathBuf};

use plow_asset::dist::{Bundle, Provenance};

use crate::{Result, RuntimeError};

/// Files the loader reads from the checkpoint directory rather than the bundle.
/// `tokenizer_config.json` appears in both places for two different purposes:
/// the assets copy backs the chat template, the checkpoint copy backs
/// `chat_stop_ids`.
const SIDECAR_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
];

#[derive(Debug)]
pub struct Farm {
    pub dir: PathBuf,
    pub shards: usize,
    pub derived: Option<String>,
}

fn link(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() || dst.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(dst);
    }
    std::os::unix::fs::symlink(src, dst).map_err(|source| RuntimeError::Io {
        path: dst.to_path_buf(),
        source,
    })
}

/// Where a checkpoint's farm lives in the store.
pub fn farm_path(store_root: &Path, source: &str, revision: &str) -> PathBuf {
    let repo = source
        .strip_prefix("hf:")
        .unwrap_or(source)
        .replace('/', "--");
    store_root
        .join("checkpoints")
        .join(format!("{repo}@{revision}"))
}

/// Assemble the farm: the snapshot's shards, the bundle's derived sidecar if it
/// has one, and the tokenizer whichever side owns it.
///
/// `bundle_dir` is a materialized bundle; `snapshot` is the operator's
/// checkpoint. Neither is modified — the farm is symlinks.
pub fn build(
    store_root: &Path,
    bundle: &Bundle,
    bundle_dir: &Path,
    snapshot: &Path,
) -> Result<Farm> {
    if !snapshot.is_dir() {
        return Err(RuntimeError::Dist(format!(
            "checkpoint {} is not a directory. {} needs the {} snapshot at revision {}; \
             pass --checkpoint <dir> or fetch it with --fetch-weights.",
            snapshot.display(),
            bundle.name,
            bundle.checkpoint.source,
            bundle.checkpoint.revision
        )));
    }
    let dir = farm_path(
        store_root,
        &bundle.checkpoint.source,
        &bundle.checkpoint.revision,
    );
    std::fs::create_dir_all(&dir).map_err(|source| RuntimeError::Io {
        path: dir.clone(),
        source,
    })?;

    let mut shards = 0usize;
    let rd = std::fs::read_dir(snapshot).map_err(|source| RuntimeError::Io {
        path: snapshot.to_path_buf(),
        source,
    })?;
    let mut names: Vec<String> = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") {
            link(&e.path(), &dir.join(&name))?;
            names.push(name);
            shards += 1;
        }
    }
    if shards == 0 {
        return Err(RuntimeError::Dist(format!(
            "{}: no *.safetensors found — this does not look like a checkpoint",
            snapshot.display()
        )));
    }
    if shards < bundle.checkpoint.shards as usize {
        return Err(RuntimeError::Dist(format!(
            "{}: found {shards} shards, the bundle was built against {}. A partial snapshot \
             loads and then produces wrong output, so this is refused.",
            snapshot.display(),
            bundle.checkpoint.shards
        )));
    }

    for f in SIDECAR_FILES {
        let src = snapshot.join(f);
        if src.is_file() {
            link(&src, &dir.join(f))?;
        }
    }

    // The derived sidecar is a compiled asset and travels in the bundle; it must
    // sort AFTER every base shard, because that ordering is the override.
    let mut derived = None;
    for f in &bundle.files {
        if f.role == "derived_shard" {
            let src = bundle_dir.join(&f.name);
            if !src.is_file() {
                return Err(RuntimeError::Dist(format!(
                    "{}: the bundle declares a derived shard that was not materialized",
                    f.name
                )));
            }
            if names.iter().any(|n| n.as_str() >= f.name.as_str()) {
                return Err(RuntimeError::Dist(format!(
                    "{}: the derived sidecar must sort after every base shard, because the \
                     loader's last-writer-wins glob is what makes it an override. Rename it so \
                     it sorts last.",
                    f.name
                )));
            }
            link(&src, &dir.join(&f.name))?;
            derived = Some(f.name.clone());
        }
    }

    // The tokenizer: a real file either way, because the AMD and CPU engines
    // refuse a byte-fallback tokenizer and that refusal is what stands between a
    // mispackaged bundle and fluent wrong output.
    let tok = &bundle.tokenizer;
    match tok.source {
        Provenance::Derived => {
            let src = bundle_dir.join(&tok.file);
            if !src.is_file() {
                return Err(RuntimeError::Dist(format!(
                    "{}: the bundle declares a derived tokenizer that was not materialized",
                    tok.file
                )));
            }
            link(&src, &dir.join(&tok.file))?;
        }
        Provenance::Checkpoint => {
            let src = snapshot.join(&tok.file);
            if !src.is_file() {
                return Err(RuntimeError::Dist(format!(
                    "{}: {} declares its tokenizer comes from the checkpoint, but {} has none. \
                     A checkpoint that ships only `tiktoken.model` needs a generated \
                     tokenizer.json — that is what `scripts/kimi_k3_tokenizer.py` produces, and \
                     a bundle for such a model must declare `source = \"derived\"`.",
                    tok.file,
                    bundle.name,
                    snapshot.display()
                )));
            }
            link(&src, &dir.join(&tok.file))?;
            // The runtime reads the tokenizer from the ASSETS dir, so the
            // bundle needs it too.
            link(&src, &bundle_dir.join(&tok.file))?;
        }
    }
    for f in SIDECAR_FILES {
        let src = snapshot.join(f);
        if src.is_file() && !bundle_dir.join(f).exists() {
            link(&src, &bundle_dir.join(f))?;
        }
    }

    Ok(Farm {
        dir,
        shards,
        derived,
    })
}

#[cfg(test)]
#[path = "prepare_tests.rs"]
mod tests;
