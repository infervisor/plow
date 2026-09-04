//! Resolve Hugging Face model metadata without downloading weight shards.
//!
//! The compiler needs the model config, the complete tensor-to-shard index,
//! and tokenizer/chat assets. Weight bytes are a runtime concern. Resolution
//! therefore uses a fixed allowlist and never follows shard names from the
//! safetensors index.

use std::path::{Path, PathBuf};

use crate::models::{build_text_generation_from_config_json_at, BuildError, ShapeBucket};
use crate::Graph;

/// Hugging Face files needed to compile packets and preserve serving metadata.
///
/// Weight files, custom Python modules, and unrelated repository files are
/// deliberately absent.
pub const METADATA_FILES: &[&str] = &[
    "config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "sentencepiece.bpe.model",
    "tekken.json",
    "vocab.txt",
];

#[derive(thiserror::Error, Debug)]
pub enum HubError {
    #[error(transparent)]
    Build(#[from] BuildError),
    #[error("hub error: {0}")]
    Hub(String),
}

/// A locally available, metadata-only snapshot of a Hugging Face model.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    model_id: String,
    revision: Option<String>,
    config_json: String,
    has_vision_config: bool,
    files: Vec<(String, PathBuf)>,
}

impl ModelMetadata {
    /// Model id supplied by the caller, or the local directory path.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Immutable Hub commit used for this snapshot, when resolved from HF.
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    /// Whether the source checkpoint declares a vision tower.
    pub fn has_vision_config(&self) -> bool {
        self.has_vision_config
    }

    /// Resolved path for one allowlisted artifact.
    pub fn path(&self, filename: &str) -> Option<&Path> {
        self.files
            .iter()
            .find(|(name, _)| name == filename)
            .map(|(_, path)| path.as_path())
    }

    /// Config contents used by the architecture frontend.
    pub fn config_json(&self) -> &str {
        &self.config_json
    }

    /// Names of all metadata artifacts that were resolved.
    pub fn filenames(&self) -> impl Iterator<Item = &str> {
        self.files.iter().map(|(name, _)| name.as_str())
    }

    /// Verify the compiler's required checkpoint tensors against the complete
    /// safetensors index.
    pub fn validate_checkpoint_manifest(&self, graph: &Graph) -> Result<(), HubError> {
        let index_path = self
            .path("model.safetensors.index.json")
            .expect("metadata resolution requires a safetensors index");
        let bytes = std::fs::read(index_path)
            .map_err(|e| HubError::Hub(format!("cannot read {}: {e}", index_path.display())))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| HubError::Hub(format!("invalid {}: {e}", index_path.display())))?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                HubError::Hub(format!(
                    "{} has no object-valued weight_map",
                    index_path.display()
                ))
            })?;
        if weight_map
            .values()
            .any(|shard| shard.as_str().is_none_or(str::is_empty))
        {
            return Err(HubError::Hub(format!(
                "{} has a non-string or empty shard name in weight_map",
                index_path.display()
            )));
        }
        let indexed = weight_map
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let expected = graph
            .checkpoint_manifest()
            .into_iter()
            .map(|weight| weight.name)
            .collect::<std::collections::BTreeSet<_>>();

        let missing = expected
            .iter()
            .copied()
            .filter(|expected_name| {
                !indexed
                    .iter()
                    .any(|indexed_name| checkpoint_name_matches(expected_name, indexed_name))
            })
            .collect::<Vec<_>>();
        let wrapped_language_model = expected
            .iter()
            .any(|name| name.starts_with("model.language_model."));
        let unexpected = indexed
            .iter()
            .copied()
            .filter(|name| checkpoint_tensor_is_text(name, wrapped_language_model))
            .filter(|indexed_name| {
                !expected
                    .iter()
                    .any(|expected_name| checkpoint_name_matches(expected_name, indexed_name))
            })
            .collect::<Vec<_>>();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(HubError::Hub(format!(
                "{} does not match the compiled text graph: {} missing [{}], {} unexpected [{}]",
                index_path.display(),
                missing.len(),
                summarize_names(&missing),
                unexpected.len(),
                summarize_names(&unexpected),
            )));
        }
        Ok(())
    }

    /// Copy the resolved snapshot metadata beside compiler output packets.
    pub fn copy_to(&self, out: &Path) -> Result<(), HubError> {
        std::fs::create_dir_all(out)
            .map_err(|e| HubError::Hub(format!("cannot create {}: {e}", out.display())))?;
        for &name in METADATA_FILES {
            let target = out.join(name);
            if self.path(name).is_none() {
                if target.is_file() {
                    std::fs::remove_file(&target).map_err(|e| {
                        HubError::Hub(format!("cannot remove stale {}: {e}", target.display()))
                    })?;
                }
            }
        }
        for (name, source) in &self.files {
            let target = out.join(name);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    HubError::Hub(format!("cannot create {}: {e}", parent.display()))
                })?;
            }
            std::fs::copy(source, &target).map_err(|e| {
                HubError::Hub(format!(
                    "cannot copy {} to {}: {e}",
                    source.display(),
                    target.display()
                ))
            })?;
        }
        let (index_tensors, index_shards) = self.index_counts()?;
        let manifest = serde_json::json!({
            "source": self.model_id(),
            "revision": self.revision(),
            "files": self.filenames().collect::<Vec<_>>(),
            "compile_scope": "text_generation",
            "source_modalities": if self.has_vision_config() {
                vec!["text", "vision"]
            } else {
                vec!["text"]
            },
            "compiled_modalities": ["text"],
            "safetensors_index": {
                "tensors": index_tensors,
                "shards": index_shards,
            },
            "weight_shards_downloaded": false,
        });
        std::fs::write(
            out.join("hf_metadata.json"),
            serde_json::to_vec_pretty(&manifest).expect("JSON value serializes"),
        )
        .map_err(|e| HubError::Hub(format!("cannot write HF metadata manifest: {e}")))?;
        Ok(())
    }

    fn index_counts(&self) -> Result<(usize, usize), HubError> {
        let index_path = self
            .path("model.safetensors.index.json")
            .expect("metadata resolution requires a safetensors index");
        let bytes = std::fs::read(index_path)
            .map_err(|e| HubError::Hub(format!("cannot read {}: {e}", index_path.display())))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| HubError::Hub(format!("invalid {}: {e}", index_path.display())))?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                HubError::Hub(format!(
                    "{} has no object-valued weight_map",
                    index_path.display()
                ))
            })?;
        if weight_map
            .values()
            .any(|shard| shard.as_str().is_none_or(str::is_empty))
        {
            return Err(HubError::Hub(format!(
                "{} has a non-string or empty shard name in weight_map",
                index_path.display()
            )));
        }
        let shards = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        Ok((weight_map.len(), shards))
    }
}

fn checkpoint_name_matches(expected: &str, indexed: &str) -> bool {
    indexed == expected
        || indexed
            .strip_prefix("model.")
            .is_some_and(|name| name == expected)
        || indexed
            .strip_prefix("model.language_model.")
            .is_some_and(|name| name == expected)
        || indexed
            .strip_prefix("language_model.model.")
            .is_some_and(|name| name == expected)
}

fn checkpoint_tensor_is_text(name: &str, wrapped_language_model: bool) -> bool {
    if wrapped_language_model {
        return name.starts_with("model.language_model.") || name.starts_with("lm_head.");
    }
    ![
        "model.visual.",
        "visual.",
        "model.embed_vision.",
        "model.vision_tower.",
        "vision_tower.",
        "vision_model.",
        "model.mm_projector.",
        "multi_modal_projector.",
        "model.mtp.",
        "mtp.",
        "model.multi_token_predictor.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn summarize_names(names: &[&str]) -> String {
    const LIMIT: usize = 8;
    let mut summary = names
        .iter()
        .take(LIMIT)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > LIMIT {
        summary.push_str(&format!(", … {} more", names.len() - LIMIT));
    }
    summary
}

/// Resolve a model id from a local directory, the HF cache, or the Hub.
///
/// `HF_HUB_OFFLINE=1` disables network access and requires `config.json` to be
/// present in the normal HF cache. A directory passed as `model_id` is always
/// resolved directly and is useful for hermetic/offline compilation.
pub fn resolve_model_metadata(model_id: &str) -> Result<ModelMetadata, HubError> {
    let local = Path::new(model_id);
    if local.is_dir() {
        return metadata_from_dir(model_id, local);
    }

    use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};

    let cache = Cache::from_env();
    let cached_main = cache.model(model_id.to_string());
    let offline = offline_enabled();
    if offline {
        return metadata_from_cached_main(model_id, &cached_main);
    }

    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| HubError::Hub(e.to_string()))?;
    let main_repo = api.model(model_id.to_string());
    let info = match main_repo.info() {
        Ok(info) => info,
        Err(_error) if cached_main.get("config.json").is_some() => {
            return metadata_from_cached_main(model_id, &cached_main);
        }
        Err(error) => return Err(HubError::Hub(error.to_string())),
    };
    let revision = info.sha;
    let available = info
        .siblings
        .into_iter()
        .map(|file| file.rfilename)
        .collect::<std::collections::HashSet<_>>();
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.clone(),
    ));
    let mut files = Vec::new();
    let mut names = available
        .iter()
        .filter(|name| is_metadata_filename(name))
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        if available.contains(&name) {
            let path = repo
                .get(&name)
                .map_err(|e| HubError::Hub(format!("cannot resolve {model_id}/{name}: {e}")))?;
            files.push((name, path));
        }
    }
    metadata_from_files(model_id, Some(revision), files)
}

fn metadata_from_dir(model_id: &str, dir: &Path) -> Result<ModelMetadata, HubError> {
    metadata_from_dir_at_revision(model_id, dir, None)
}

fn metadata_from_dir_at_revision(
    model_id: &str,
    dir: &Path,
    revision: Option<String>,
) -> Result<ModelMetadata, HubError> {
    let mut files = METADATA_FILES
        .iter()
        .filter_map(|name| {
            let path = dir.join(name);
            path.is_file().then(|| ((*name).to_string(), path))
        })
        .collect::<Vec<_>>();
    let templates = dir.join("chat_templates");
    if let Ok(entries) = std::fs::read_dir(templates) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let name = format!("chat_templates/{filename}");
            if path.is_file() && is_metadata_filename(&name) {
                files.push((name, path));
            }
        }
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    metadata_from_files(model_id, revision, files)
}

fn metadata_from_cached_main(
    model_id: &str,
    cache: &hf_hub::CacheRepo,
) -> Result<ModelMetadata, HubError> {
    let config = cache.get("config.json").ok_or_else(|| {
        HubError::Hub(format!(
            "{model_id}: config.json is not available in the Hugging Face cache \
             (HF_HUB_OFFLINE is set)"
        ))
    })?;
    let snapshot = config
        .parent()
        .ok_or_else(|| HubError::Hub(format!("cannot determine cached snapshot for {model_id}")))?;
    let revision = snapshot
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string);
    metadata_from_dir_at_revision(model_id, snapshot, revision)
}

fn metadata_from_files(
    model_id: &str,
    revision: Option<String>,
    files: Vec<(String, PathBuf)>,
) -> Result<ModelMetadata, HubError> {
    let Some(config_path) = files
        .iter()
        .find(|(name, _)| name == "config.json")
        .map(|(_, path)| path)
    else {
        return Err(HubError::Hub(format!(
            "{model_id}: config.json is not available{}",
            if offline_enabled() {
                " in the Hugging Face cache (HF_HUB_OFFLINE is set)"
            } else {
                ""
            }
        )));
    };
    let config_json = std::fs::read_to_string(config_path)
        .map_err(|e| HubError::Hub(format!("cannot read {}: {e}", config_path.display())))?;
    let has_vision_config = serde_json::from_str::<serde_json::Value>(&config_json)
        .ok()
        .and_then(|config| config.get("vision_config").cloned())
        .is_some_and(|vision| !vision.is_null());
    if !files
        .iter()
        .any(|(name, _)| name == "model.safetensors.index.json")
    {
        return Err(HubError::Hub(format!(
            "{model_id}: model.safetensors.index.json is required for metadata-only compilation; \
             plowc will not download a weight file to infer its tensor manifest"
        )));
    }
    Ok(ModelMetadata {
        model_id: model_id.to_string(),
        revision,
        config_json,
        has_vision_config,
        files,
    })
}

fn is_metadata_filename(name: &str) -> bool {
    METADATA_FILES.contains(&name)
        || name
            .strip_prefix("chat_templates/")
            .is_some_and(|filename| !filename.contains('/') && filename.ends_with(".jinja"))
}

fn offline_enabled() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

/// Resolve model metadata and build its text-generation graph.
pub fn build_from_pretrained(model_id: &str, bucket: &ShapeBucket) -> Result<Graph, HubError> {
    let metadata = resolve_model_metadata(model_id)?;
    build_from_metadata(&metadata, bucket)
}

/// Build a text-generation graph from already-resolved metadata.
pub fn build_from_metadata(
    metadata: &ModelMetadata,
    bucket: &ShapeBucket,
) -> Result<Graph, HubError> {
    Ok(build_text_generation_from_config_json_at(
        metadata.config_json(),
        bucket,
    )?)
}

/// Download only `config.json` and return its contents.
///
/// Visualization and architecture inspection do not require a safetensors
/// index or tokenizer assets. Compilation uses [`resolve_model_metadata`].
pub fn fetch_config(model_id: &str) -> Result<String, HubError> {
    let local = Path::new(model_id);
    if local.is_dir() {
        return read_config(model_id, local.join("config.json"));
    }

    use hf_hub::{api::sync::ApiBuilder, Cache};

    let cache = Cache::from_env().model(model_id.to_string());
    let cached = cache.get("config.json");
    let offline = std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    });
    if offline {
        let path = cached.ok_or_else(|| {
            HubError::Hub(format!(
                "{model_id}: config.json is not available in the Hugging Face cache \
                 (HF_HUB_OFFLINE is set)"
            ))
        })?;
        return read_config(model_id, path);
    }

    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| HubError::Hub(e.to_string()))?;
    let path = match api.model(model_id.to_string()).get("config.json") {
        Ok(path) => path,
        Err(_) if cached.is_some() => cached.expect("checked above"),
        Err(error) => return Err(HubError::Hub(error.to_string())),
    };
    read_config(model_id, path)
}

fn read_config(model_id: &str, path: PathBuf) -> Result<String, HubError> {
    std::fs::read_to_string(&path).map_err(|e| {
        HubError::Hub(format!(
            "cannot read {model_id} config {}: {e}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nn-graph-hub-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn local_resolution_copies_metadata_but_never_weights() {
        let dir = tempdir("allowlist");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","vocab_size":32,"hidden_size":16,
                "intermediate_size":32,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
        std::fs::write(dir.join("video_preprocessor_config.json"), "{}").unwrap();
        std::fs::create_dir(dir.join("chat_templates")).unwrap();
        std::fs::write(dir.join("chat_templates/tool_use.jinja"), "{{ tools }}").unwrap();
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), "weight bytes").unwrap();
        std::fs::write(dir.join("modeling_test.py"), "remote code").unwrap();

        let metadata = resolve_model_metadata(dir.to_str().unwrap()).unwrap();
        let names = metadata.filenames().collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "chat_template.jinja",
                "chat_templates/tool_use.jinja",
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "video_preprocessor_config.json",
            ]
        );

        let out = tempdir("copied");
        metadata.copy_to(&out).unwrap();
        assert!(out.join("config.json").is_file());
        assert!(out.join("model.safetensors.index.json").is_file());
        assert!(out.join("tokenizer.json").is_file());
        assert!(out.join("chat_template.jinja").is_file());
        assert!(out.join("video_preprocessor_config.json").is_file());
        assert!(out.join("chat_templates/tool_use.jinja").is_file());
        assert!(out.join("hf_metadata.json").is_file());
        assert!(!out.join("model-00001-of-00001.safetensors").exists());
        assert!(!out.join("modeling_test.py").exists());

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(out).ok();
    }

    #[test]
    fn manifest_marks_multimodal_checkpoints_as_text_only_compilation() {
        let dir = tempdir("vision-scope");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"fixture","text_config":{},"vision_config":{"hidden_size":16}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let metadata = resolve_model_metadata(dir.to_str().unwrap()).unwrap();
        assert!(metadata.has_vision_config());
        assert_eq!(metadata.revision(), None);
        let out = tempdir("vision-scope-out");
        metadata.copy_to(&out).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("hf_metadata.json")).unwrap()).unwrap();
        assert_eq!(manifest["compile_scope"], "text_generation");
        assert_eq!(
            manifest["source_modalities"],
            serde_json::json!(["text", "vision"])
        );
        assert_eq!(manifest["compiled_modalities"], serde_json::json!(["text"]));

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(out).ok();
    }

    #[test]
    fn cached_resolution_pins_one_snapshot() {
        use hf_hub::{Cache, Repo, RepoType};

        let root = tempdir("cache-root");
        let repo = Repo::with_revision("org/model".into(), RepoType::Model, "main".into());
        let snapshot = root
            .join(repo.folder_name())
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();
        let cache = Cache::new(root.clone()).repo(repo);
        cache.create_ref("abc123").unwrap();

        let metadata = metadata_from_cached_main("org/model", &cache).unwrap();
        assert_eq!(metadata.revision(), Some("abc123"));
        assert!(metadata
            .path("model.safetensors.index.json")
            .is_some_and(|path| path.starts_with(&snapshot)));

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn local_resolution_requires_config() {
        let dir = tempdir("missing-config");
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        let error = resolve_model_metadata(dir.to_str().unwrap()).unwrap_err();
        assert!(error.to_string().contains("config.json is not available"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn local_fetch_config_does_not_require_an_index() {
        let dir = tempdir("config-only");
        let config = r#"{"model_type":"llama"}"#;
        std::fs::write(dir.join("config.json"), config).unwrap();

        assert_eq!(fetch_config(dir.to_str().unwrap()).unwrap(), config);

        std::fs::remove_dir_all(dir).ok();
    }
}
