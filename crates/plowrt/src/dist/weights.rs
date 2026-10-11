//! `plowrt load --fetch-weights`: the Hugging Face snapshot a bundle was built
//! against, downloaded into the store so the checkpoint farm can be built
//! from it. The distribution carries no weights; this is the consent step
//! that gets them.
//!
//! Only `hf:<org>/<repo>` sources can be fetched. The whole repository at the
//! bundle's revision is downloaded through `hf-hub` (its cache under
//! `HF_HOME`, its token from `HF_TOKEN`) and linked into
//! `snapshots/<org>--<repo>@<revision>/`, the layout a `huggingface-cli
//! download` leaves and an operator's `--checkpoint` directory has. A
//! snapshot that is already there with enough shards is reused without a
//! request.

use std::path::{Path, PathBuf};

use crate::{Result, RuntimeError};

/// The repo behind an `hf:` source; anything else cannot be fetched.
pub fn hf_repo(source: &str) -> Result<&str> {
    source
        .strip_prefix("hf:")
        .filter(|r| r.split('/').count() == 2 && !r.contains(".."))
        .ok_or_else(|| {
            RuntimeError::Dist(format!(
                "{source} is not an hf:<org>/<repo> source, so --fetch-weights cannot get it; \
                 pass --checkpoint <dir>"
            ))
        })
}

/// Where a fetched snapshot lives in the store. The revision comes from the
/// bundle manifest, so it is checked to be one path component: a `..` in it
/// would put the snapshot (and the `remove_dir_all` that replaces it) outside
/// the store.
pub fn snapshot_path(store_root: &Path, repo: &str, revision: &str) -> Result<PathBuf> {
    if revision.is_empty()
        || revision == "."
        || revision == ".."
        || revision.contains('/')
        || revision.contains('\\')
    {
        return Err(RuntimeError::Dist(format!(
            "{repo}: revision {revision:?} is not a plain commit or branch name"
        )));
    }
    Ok(store_root
        .join("snapshots")
        .join(format!("{}@{revision}", repo.replace('/', "--"))))
}

/// A repository path that may be written under the snapshot: relative,
/// nothing that escapes it.
pub fn safe_repo_path(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && name
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
}

/// A full 40-hex git commit id: the only revision whose content cannot move.
pub fn is_commit(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

fn count_shards(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                .count()
        })
        .unwrap_or(0)
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> RuntimeError + '_ {
    move |source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Fetch the snapshot, or reuse the one already in the store. Returns the
/// snapshot directory, ready for `prepare::build`.
#[cfg(feature = "dist")]
pub fn fetch(store_root: &Path, source: &str, revision: &str, shards: u32) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let repo = hf_repo(source)?;
    let dir = snapshot_path(store_root, repo, revision)?;
    // Only a commit hash names fixed content; a branch such as `main` may
    // have moved since it was fetched, so it is resolved again (hf-hub's own
    // cache keeps that cheap when nothing changed).
    if is_commit(revision) && dir.is_dir() && count_shards(&dir) >= shards as usize {
        return Ok(dir);
    }

    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| RuntimeError::Dist(format!("hugging face client: {e}")))?;
    let api_repo = api.repo(Repo::with_revision(
        repo.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let info = api_repo
        .info()
        .map_err(|e| RuntimeError::Dist(format!("{repo}@{revision}: {e}")))?;
    let mut names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    names.sort();
    if let Some(bad) = names.iter().find(|n| !safe_repo_path(n)) {
        return Err(RuntimeError::Dist(format!(
            "{repo}@{revision} lists a file path that cannot be written: {bad}"
        )));
    }
    let found = names
        .iter()
        .filter(|n| !n.contains('/') && n.ends_with(".safetensors"))
        .count();
    if found < shards as usize {
        return Err(RuntimeError::Dist(format!(
            "{repo}@{revision} lists {found} safetensors shards, the bundle was built against \
             {shards}: not the checkpoint this build is joined to"
        )));
    }

    // Download into the hub cache, then link into a staging directory that
    // becomes the snapshot only once every file is there.
    let staging = dir.with_extension(format!("partial-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(io(&staging))?;
    for name in &names {
        let cached = api_repo
            .get(name)
            .map_err(|e| RuntimeError::Dist(format!("{repo}@{revision}/{name}: {e}")))?;
        let cached = std::fs::canonicalize(&cached).map_err(io(&cached))?;
        let dst = staging.join(name);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(io(parent))?;
        }
        if std::fs::hard_link(&cached, &dst).is_err() {
            std::fs::copy(&cached, &dst).map_err(io(&dst))?;
        }
    }
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::rename(&staging, &dir).map_err(io(&dir))?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_hf_sources_can_be_fetched() {
        assert_eq!(hf_repo("hf:Qwen/Qwen3-0.6B").unwrap(), "Qwen/Qwen3-0.6B");
        assert!(hf_repo("Qwen/Qwen3-0.6B").is_err());
        assert!(hf_repo("hf:Qwen").is_err());
        assert!(hf_repo("hf:../x/y").is_err());
    }

    #[test]
    fn the_snapshot_sits_beside_the_farm_in_the_store() {
        let p = snapshot_path(Path::new("/s"), "Qwen/Qwen3-0.6B", "abc").unwrap();
        assert_eq!(p, PathBuf::from("/s/snapshots/Qwen--Qwen3-0.6B@abc"));
        for bad in ["", ".", "..", "../x", "a/b", "x/..", "a\\b"] {
            assert!(
                snapshot_path(Path::new("/s"), "Qwen/Qwen3-0.6B", bad).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn only_a_commit_hash_is_reused_without_asking() {
        assert!(is_commit(&"a".repeat(40)));
        assert!(!is_commit("main"));
        assert!(!is_commit(&"a".repeat(7)));
        assert!(!is_commit(&"g".repeat(40)));
    }

    #[test]
    fn every_repository_path_is_written_unless_it_escapes() {
        assert!(safe_repo_path("model-00001-of-00002.safetensors"));
        assert!(safe_repo_path("onnx/model.onnx"));
        assert!(safe_repo_path("README.md"));
        assert!(!safe_repo_path("../x"));
        assert!(!safe_repo_path("a/../b"));
        assert!(!safe_repo_path("/etc/passwd"));
        assert!(!safe_repo_path(""));
        assert!(!safe_repo_path("..\\..\\outside"), "a Windows separator");
    }

    /// Needs the network: `cargo test --features dist -- --ignored fetches_a_tiny`.
    /// A few-megabyte test repository, so the whole of it is cheap to pull.
    #[test]
    #[ignore]
    #[cfg(feature = "dist")]
    fn fetches_a_tiny_repository_whole_once_and_reuses_it() {
        let repo = "hf:hf-internal-testing/tiny-random-LlamaForCausalLM";
        let root = std::env::temp_dir().join(format!("plow-weights-{}", std::process::id()));
        let dir = fetch(&root, repo, "main", 1).unwrap();
        assert!(dir.join("model.safetensors").is_file());
        assert!(dir.join("config.json").is_file());
        assert!(dir.join("tokenizer.json").is_file());
        let n = std::fs::read_dir(&dir).unwrap().count();
        assert!(n >= 6, "the whole repository, not a selection: {n} entries");
        let again = fetch(&root, repo, "main", 1).unwrap();
        assert_eq!(again, dir);
        let _ = std::fs::remove_dir_all(root);
    }
}
