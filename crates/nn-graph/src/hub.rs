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
    config_json: String,
    files: Vec<(String, PathBuf)>,
}

impl ModelMetadata {
    /// Model id supplied by the caller, or the local directory path.
    pub fn model_id(&self) -> &str {
        &self.model_id
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
            let Some(source) = self.path(name) else {
                if target.is_file() {
                    std::fs::remove_file(&target).map_err(|e| {
                        HubError::Hub(format!("cannot remove stale {}: {e}", target.display()))
                    })?;
                }
                continue;
            };
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
            "files": self.filenames().collect::<Vec<_>>(),
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

    use hf_hub::{api::sync::ApiBuilder, Cache};

    let cache = Cache::from_env().model(model_id.to_string());
    let offline = std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    });
    if offline {
        return metadata_from_getter(model_id, |name| cache.get(name));
    }

    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| HubError::Hub(e.to_string()))?;
    let repo = api.model(model_id.to_string());
    let info = match repo.info() {
        Ok(info) => info,
        Err(_error) if cache.get("config.json").is_some() => {
            return metadata_from_getter(model_id, |name| cache.get(name));
        }
        Err(error) => return Err(HubError::Hub(error.to_string())),
    };
    let available = info
        .siblings
        .into_iter()
        .map(|file| file.rfilename)
        .collect::<std::collections::HashSet<_>>();
    let mut files = Vec::new();
    for &name in METADATA_FILES {
        if available.contains(name) {
            let path = repo
                .get(name)
                .map_err(|e| HubError::Hub(format!("cannot resolve {model_id}/{name}: {e}")))?;
            files.push((name.to_string(), path));
        }
    }
    metadata_from_files(model_id, files)
}

fn metadata_from_dir(model_id: &str, dir: &Path) -> Result<ModelMetadata, HubError> {
    metadata_from_getter(model_id, |name| {
        let path = dir.join(name);
        path.is_file().then_some(path)
    })
}

fn metadata_from_getter(
    model_id: &str,
    mut get: impl FnMut(&str) -> Option<PathBuf>,
) -> Result<ModelMetadata, HubError> {
    let files = METADATA_FILES
        .iter()
        .filter_map(|name| get(name).map(|path| ((*name).to_string(), path)))
        .collect::<Vec<_>>();
    metadata_from_files(model_id, files)
}

fn metadata_from_files(
    model_id: &str,
    files: Vec<(String, PathBuf)>,
) -> Result<ModelMetadata, HubError> {
    let Some(config_path) = files
        .iter()
        .find(|(name, _)| name == "config.json")
        .map(|(_, path)| path)
    else {
        return Err(HubError::Hub(format!(
            "{model_id}: config.json is not available{}",
            if std::env::var_os("HF_HUB_OFFLINE").is_some() {
                " in the Hugging Face cache (HF_HUB_OFFLINE is set)"
            } else {
                ""
            }
        )));
    };
    let config_json = std::fs::read_to_string(config_path)
        .map_err(|e| HubError::Hub(format!("cannot read {}: {e}", config_path.display())))?;
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
        config_json,
        files,
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
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), "weight bytes").unwrap();
        std::fs::write(dir.join("modeling_test.py"), "remote code").unwrap();

        let metadata = resolve_model_metadata(dir.to_str().unwrap()).unwrap();
        let names = metadata.filenames().collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "chat_template.jinja"
            ]
        );

        let out = tempdir("copied");
        metadata.copy_to(&out).unwrap();
        assert!(out.join("config.json").is_file());
        assert!(out.join("model.safetensors.index.json").is_file());
        assert!(out.join("tokenizer.json").is_file());
        assert!(out.join("chat_template.jinja").is_file());
        assert!(out.join("hf_metadata.json").is_file());
        assert!(!out.join("model-00001-of-00001.safetensors").exists());
        assert!(!out.join("modeling_test.py").exists());

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(out).ok();
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
