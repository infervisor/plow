//! Resolve Hugging Face model metadata without downloading weight shards.
//!
//! The compiler needs the model config, the complete tensor-to-shard index,
//! and tokenizer/chat assets. Weight bytes are a runtime concern. Resolution
//! therefore uses a fixed allowlist and never follows shard names from the
//! safetensors index.

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::models::{build_text_generation_from_config_json_at, BuildError, ShapeBucket};
use crate::{DType, Graph};

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

const SAFETENSORS_FILE: &str = "model.safetensors";
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

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
    has_audio_config: bool,
    files: Vec<(String, PathBuf)>,
    synthetic_index_json: Option<String>,
    unresolved_optional_files: Vec<String>,
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

    /// Whether the source checkpoint declares an audio tower.
    pub fn has_audio_config(&self) -> bool {
        self.has_audio_config
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
        self.files.iter().map(|(name, _)| name.as_str()).chain(
            self.synthetic_index_json
                .as_ref()
                .map(|_| SAFETENSORS_INDEX),
        )
    }

    /// Advertised optional metadata files that the Hub did not serve.
    pub fn unresolved_optional_files(&self) -> &[String] {
        &self.unresolved_optional_files
    }

    /// Logical tensor bytes reported by the complete safetensors index.
    pub fn indexed_total_size(&self) -> Result<Option<u64>, HubError> {
        let (index, _) = self.index_value()?;
        Ok(index
            .get("metadata")
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(serde_json::Value::as_u64))
    }

    /// Verify the compiler's required checkpoint tensors against the complete
    /// safetensors index.
    pub fn validate_checkpoint_manifest(&self, graph: &Graph) -> Result<(), HubError> {
        let (value, index_label) = self.index_value()?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                HubError::Hub(format!("{} has no object-valued weight_map", index_label))
            })?;
        if weight_map
            .values()
            .any(|shard| shard.as_str().is_none_or(str::is_empty))
        {
            return Err(HubError::Hub(format!(
                "{} has a non-string or empty shard name in weight_map",
                index_label
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
        let indexed_aliases = indexed
            .iter()
            .copied()
            .flat_map(checkpoint_name_aliases)
            .collect::<std::collections::BTreeSet<_>>();

        let missing = expected
            .iter()
            .copied()
            .filter(|expected_name| !indexed_aliases.contains(expected_name))
            .collect::<Vec<_>>();
        let wrapped_language_model = expected
            .iter()
            .any(|name| name.starts_with("model.language_model."));
        let omitted_mtp_layers = omitted_mtp_layer_range(&self.config_json);
        let unexpected = indexed
            .iter()
            .copied()
            .filter(|name| checkpoint_tensor_is_text(name, wrapped_language_model))
            .filter(|name| !checkpoint_tensor_is_omitted_mtp(name, omitted_mtp_layers))
            .filter(|indexed_name| {
                !checkpoint_name_aliases(indexed_name).any(|name| expected.contains(name))
            })
            .collect::<Vec<_>>();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(HubError::Hub(format!(
                "{} does not match the compiled text graph: {} missing [{}], {} unexpected [{}]",
                index_label,
                missing.len(),
                summarize_names(&missing),
                unexpected.len(),
                summarize_names(&unexpected),
            )));
        }
        if let Some(tensor_metadata) = value
            .get("plow_tensor_metadata")
            .and_then(serde_json::Value::as_object)
        {
            for weight in graph.checkpoint_manifest() {
                let indexed_name = indexed.iter().copied().find(|indexed_name| {
                    checkpoint_name_aliases(indexed_name).any(|alias| alias == weight.name)
                });
                let Some(indexed_name) = indexed_name else {
                    continue;
                };
                let actual = tensor_metadata
                    .get(indexed_name)
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| {
                        HubError::Hub(format!(
                            "{index_label} has no tensor metadata for {indexed_name}"
                        ))
                    })?;
                let dtype = actual
                    .get("dtype")
                    .and_then(serde_json::Value::as_str)
                    .and_then(DType::from_safetensors_name)
                    .ok_or_else(|| {
                        HubError::Hub(format!(
                            "{index_label} has an invalid dtype for {indexed_name}"
                        ))
                    })?;
                if dtype != weight.dtype {
                    return Err(HubError::Hub(format!(
                        "{index_label} tensor {indexed_name} has dtype {dtype}, expected {}",
                        weight.dtype
                    )));
                }
                let expected_shape = weight
                    .shape
                    .ok_or_else(|| {
                        HubError::Hub(format!("compiled weight {} has no shape", weight.name))
                    })?
                    .dims()
                    .iter()
                    .map(|dim| {
                        dim.as_static().ok_or_else(|| {
                            HubError::Hub(format!(
                                "compiled weight {} has a dynamic dimension",
                                weight.name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let actual_shape = actual
                    .get("shape")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|shape| {
                        shape
                            .iter()
                            .map(serde_json::Value::as_i64)
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| {
                        HubError::Hub(format!(
                            "{index_label} has an invalid shape for {indexed_name}"
                        ))
                    })?;
                if actual_shape != expected_shape {
                    return Err(HubError::Hub(format!(
                        "{index_label} tensor {indexed_name} has shape {actual_shape:?}, expected {expected_shape:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Copy the resolved snapshot metadata beside compiler output packets.
    pub fn copy_to(&self, out: &Path) -> Result<(), HubError> {
        std::fs::create_dir_all(out)
            .map_err(|e| HubError::Hub(format!("cannot create {}: {e}", out.display())))?;
        let output_root = std::fs::canonicalize(out)
            .map_err(|e| HubError::Hub(format!("cannot resolve {}: {e}", out.display())))?;
        if self.path("config.json").is_some_and(|config| {
            config
                .parent()
                .and_then(|parent| std::fs::canonicalize(parent).ok())
                .is_some_and(|source_root| source_root == output_root)
        }) {
            return Err(HubError::Hub(format!(
                "refusing to copy Hugging Face metadata onto its source directory {}",
                out.display()
            )));
        }
        let chat_templates = out.join("chat_templates");
        if chat_templates.is_dir() {
            std::fs::remove_dir_all(&chat_templates).map_err(|e| {
                HubError::Hub(format!(
                    "cannot remove stale {}: {e}",
                    chat_templates.display()
                ))
            })?;
        }
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
        if let Some(index) = &self.synthetic_index_json {
            std::fs::write(out.join(SAFETENSORS_INDEX), index).map_err(|e| {
                HubError::Hub(format!("cannot write synthesized safetensors index: {e}"))
            })?;
        }
        let (index_tensors, index_shards) = self.index_counts()?;
        let mut source_modalities = vec!["text"];
        if self.has_vision_config() {
            source_modalities.push("vision");
        }
        if self.has_audio_config() {
            source_modalities.push("audio");
        }
        let manifest = serde_json::json!({
            "source": self.model_id(),
            "revision": self.revision(),
            "files": self.filenames().collect::<Vec<_>>(),
            "compile_scope": "text_generation",
            "source_modalities": source_modalities,
            "compiled_modalities": ["text"],
            "unresolved_optional_files": self.unresolved_optional_files,
            "safetensors_index": {
                "tensors": index_tensors,
                "shards": index_shards,
                "synthetic": self.synthetic_index_json.is_some(),
                "source_file": if self.synthetic_index_json.is_some() {
                    Some(SAFETENSORS_FILE)
                } else {
                    None
                },
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
        let (value, index_label) = self.index_value()?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                HubError::Hub(format!("{} has no object-valued weight_map", index_label))
            })?;
        if weight_map
            .values()
            .any(|shard| shard.as_str().is_none_or(str::is_empty))
        {
            return Err(HubError::Hub(format!(
                "{} has a non-string or empty shard name in weight_map",
                index_label
            )));
        }
        let shards = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        Ok((weight_map.len(), shards))
    }

    fn index_value(&self) -> Result<(serde_json::Value, String), HubError> {
        let (bytes, label) = if let Some(index) = &self.synthetic_index_json {
            (
                index.as_bytes().to_vec(),
                "synthesized model.safetensors index".to_string(),
            )
        } else {
            let path = self.path(SAFETENSORS_INDEX).ok_or_else(|| {
                HubError::Hub("metadata resolution produced no safetensors index".into())
            })?;
            let bytes = std::fs::read(path)
                .map_err(|e| HubError::Hub(format!("cannot read {}: {e}", path.display())))?;
            (bytes, path.display().to_string())
        };
        let value = serde_json::from_slice(&bytes)
            .map_err(|e| HubError::Hub(format!("invalid {label}: {e}")))?;
        Ok((value, label))
    }
}

fn checkpoint_name_aliases(indexed: &str) -> impl Iterator<Item = &str> {
    const PREFIXES: [&str; 3] = ["model.", "model.language_model.", "language_model.model."];
    std::iter::once(indexed).chain(
        PREFIXES
            .into_iter()
            .filter_map(move |prefix| indexed.strip_prefix(prefix)),
    )
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
        "model.vision_embedder.",
        "model.embed_audio.",
        "model.mtp.",
        "mtp.",
        "model.multi_token_predictor.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn omitted_mtp_layer_range(config_json: &str) -> Option<(u32, u32)> {
    let config: serde_json::Value = serde_json::from_str(config_json).ok()?;
    let model_type = config.get("model_type")?.as_str()?;
    if !matches!(
        model_type,
        "deepseek" | "deepseek_v2" | "deepseek_v3" | "kimi" | "kimi_k2"
    ) {
        return None;
    }
    let first = u32::try_from(config.get("num_hidden_layers")?.as_u64()?).ok()?;
    let count = u32::try_from(config.get("num_nextn_predict_layers")?.as_u64()?).ok()?;
    (count > 0).then_some((first, first.saturating_add(count)))
}

fn checkpoint_tensor_is_omitted_mtp(name: &str, range: Option<(u32, u32)>) -> bool {
    let Some((first, end)) = range else {
        return false;
    };
    let Some(layer) = name
        .strip_prefix("model.layers.")
        .and_then(|name| name.split('.').next())
        .and_then(|layer| layer.parse::<u32>().ok())
    else {
        return false;
    };
    (first..end).contains(&layer)
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
    let auth_token = cache.token();
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
    let mut names = available
        .iter()
        .filter(|name| is_metadata_filename(name))
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    let (files, unresolved_optional_files) =
        collect_advertised_metadata(model_id, names, |name| {
            repo.get(name).map_err(|error| error.to_string())
        })?;
    let synthetic_index_json = if files.iter().any(|(name, _)| name == SAFETENSORS_INDEX) {
        None
    } else if available.contains(SAFETENSORS_FILE) {
        Some(fetch_remote_safetensors_index(
            &repo.url(SAFETENSORS_FILE),
            auth_token.as_deref(),
        )?)
    } else {
        None
    };
    if let Some(index) = &synthetic_index_json {
        let config = files
            .iter()
            .find(|(name, _)| name == "config.json")
            .map(|(_, path)| path)
            .ok_or_else(|| HubError::Hub(format!("{model_id}: config.json was not resolved")))?;
        let snapshot = config.parent().ok_or_else(|| {
            HubError::Hub(format!(
                "cannot determine cached snapshot from {}",
                config.display()
            ))
        })?;
        if let Err(error) = std::fs::write(snapshot.join(SAFETENSORS_INDEX), index) {
            eprintln!(
                "warning: could not persist synthesized {SAFETENSORS_INDEX} in {}: {error}",
                snapshot.display()
            );
        }
    }
    if let Err(error) = cached_main.create_ref(&revision) {
        eprintln!("warning: could not pin cached main revision {revision} for {model_id}: {error}");
    }
    metadata_from_files(
        model_id,
        Some(revision),
        files,
        synthetic_index_json,
        unresolved_optional_files,
    )
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
    let synthetic_index_json = if files.iter().any(|(name, _)| name == SAFETENSORS_INDEX) {
        None
    } else {
        let safetensors = dir.join(SAFETENSORS_FILE);
        safetensors
            .is_file()
            .then(|| synthesize_index_from_file(&safetensors))
            .transpose()?
    };
    metadata_from_files(model_id, revision, files, synthetic_index_json, Vec::new())
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
    synthetic_index_json: Option<String>,
    unresolved_optional_files: Vec<String>,
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
    let parsed_config = serde_json::from_str::<serde_json::Value>(&config_json).ok();
    let has_vision_config = parsed_config
        .as_ref()
        .and_then(|config| config.get("vision_config"))
        .is_some_and(|vision| !vision.is_null());
    let has_audio_config = parsed_config
        .as_ref()
        .and_then(|config| config.get("audio_config"))
        .is_some_and(|audio| !audio.is_null());
    if !files.iter().any(|(name, _)| name == SAFETENSORS_INDEX) && synthetic_index_json.is_none() {
        return Err(HubError::Hub(format!(
            "{model_id}: {SAFETENSORS_INDEX} or {SAFETENSORS_FILE} is required for metadata-only \
             compilation; plowc will not download weight payloads to infer a tensor manifest"
        )));
    }
    Ok(ModelMetadata {
        model_id: model_id.to_string(),
        revision,
        config_json,
        has_vision_config,
        has_audio_config,
        files,
        synthetic_index_json,
        unresolved_optional_files,
    })
}

fn collect_advertised_metadata(
    model_id: &str,
    names: Vec<String>,
    mut get: impl FnMut(&str) -> Result<PathBuf, String>,
) -> Result<(Vec<(String, PathBuf)>, Vec<String>), HubError> {
    let mut files = Vec::with_capacity(names.len());
    let mut unresolved_optional_files = Vec::new();
    for name in names {
        match get(&name) {
            Ok(path) => files.push((name, path)),
            Err(error) if is_required_metadata_filename(&name) => {
                return Err(HubError::Hub(format!(
                    "cannot resolve required {model_id}/{name}: {error}"
                )));
            }
            Err(_) => unresolved_optional_files.push(name),
        }
    }
    Ok((files, unresolved_optional_files))
}

fn is_required_metadata_filename(name: &str) -> bool {
    matches!(name, "config.json" | SAFETENSORS_INDEX)
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

fn synthesize_index_from_file(path: &Path) -> Result<String, HubError> {
    let file = std::fs::File::open(path)
        .map_err(|e| HubError::Hub(format!("cannot open {}: {e}", path.display())))?;
    let size = file
        .metadata()
        .map_err(|e| HubError::Hub(format!("cannot stat {}: {e}", path.display())))?
        .len();
    synthesize_index_from_reader(file, Some(size))
}

fn fetch_remote_safetensors_index(url: &str, token: Option<&str>) -> Result<String, HubError> {
    let agent = ureq::builder()
        .try_proxy_from_env(true)
        .redirect_auth_headers(ureq::RedirectAuthHeaders::SameHost)
        .build();
    let fetch = |start: u64, end: u64| -> Result<(Vec<u8>, u64), HubError> {
        let mut request = agent
            .get(url)
            .set("Range", &format!("bytes={start}-{end}"))
            .set("Accept-Encoding", "identity");
        if let Some(token) = token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = request
            .call()
            .map_err(|e| HubError::Hub(format!("cannot range-fetch {url}: {e}")))?;
        if response.status() != 206 {
            return Err(HubError::Hub(format!(
                "{url} ignored a bounded safetensors header request (expected HTTP 206, got {})",
                response.status()
            )));
        }
        let content_range = response.header("Content-Range").ok_or_else(|| {
            HubError::Hub(format!("{url} returned HTTP 206 without Content-Range"))
        })?;
        let (actual_start, actual_end, total) =
            parse_content_range(content_range).ok_or_else(|| {
                HubError::Hub(format!(
                    "{url} returned invalid Content-Range {content_range:?}"
                ))
            })?;
        if (actual_start, actual_end) != (start, end) {
            return Err(HubError::Hub(format!(
                "{url} returned Content-Range {content_range:?} for requested bytes={start}-{end}"
            )));
        }
        let expected = end - start + 1;
        let mut bytes = Vec::with_capacity(expected as usize);
        response
            .into_reader()
            .take(expected + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| HubError::Hub(format!("cannot read ranged header from {url}: {e}")))?;
        if bytes.len() as u64 != expected {
            return Err(HubError::Hub(format!(
                "{url} returned {} bytes for requested range of {expected} bytes",
                bytes.len()
            )));
        }
        Ok((bytes, total))
    };

    let (prefix, total) = fetch(0, 7)?;
    let header_len = u64::from_le_bytes(prefix.try_into().expect("range length checked"));
    validate_header_len(header_len)?;
    let header_end = 8u64
        .checked_add(header_len)
        .and_then(|end| end.checked_sub(1))
        .ok_or_else(|| HubError::Hub("safetensors header range overflow".into()))?;
    let (header, second_total) = fetch(8, header_end)?;
    if second_total != total {
        return Err(HubError::Hub(format!(
            "{url} changed size between safetensors header requests ({total} vs {second_total})"
        )));
    }
    synthesize_index_from_header(&header, header_len, Some(total))
}

fn synthesize_index_from_reader(
    mut reader: impl Read,
    total_size: Option<u64>,
) -> Result<String, HubError> {
    let mut prefix = [0u8; 8];
    reader
        .read_exact(&mut prefix)
        .map_err(|e| HubError::Hub(format!("cannot read safetensors header length: {e}")))?;
    let header_len = u64::from_le_bytes(prefix);
    validate_header_len(header_len)?;
    let mut header = vec![0u8; header_len as usize];
    reader
        .read_exact(&mut header)
        .map_err(|e| HubError::Hub(format!("cannot read safetensors header: {e}")))?;
    synthesize_index_from_header(&header, header_len, total_size)
}

fn validate_header_len(header_len: u64) -> Result<(), HubError> {
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(HubError::Hub(format!(
            "invalid safetensors header length {header_len}; expected 1..={MAX_SAFETENSORS_HEADER_BYTES}"
        )));
    }
    Ok(())
}

fn synthesize_index_from_header(
    header: &[u8],
    header_len: u64,
    total_size: Option<u64>,
) -> Result<String, HubError> {
    let value: serde_json::Value = serde_json::from_slice(header)
        .map_err(|e| HubError::Hub(format!("invalid safetensors header JSON: {e}")))?;
    let tensors = value
        .as_object()
        .ok_or_else(|| HubError::Hub("safetensors header must be a JSON object".into()))?;
    let payload_bytes = total_size
        .map(|total| {
            total.checked_sub(8 + header_len).ok_or_else(|| {
                HubError::Hub(format!(
                    "safetensors file size {total} is smaller than its {}-byte header",
                    8 + header_len
                ))
            })
        })
        .transpose()?;
    let mut weight_map = serde_json::Map::new();
    let mut tensor_metadata = serde_json::Map::new();
    let mut tensor_bytes = 0u64;
    let mut intervals = Vec::new();
    for (name, tensor) in tensors {
        if name == "__metadata__" {
            continue;
        }
        let tensor = tensor.as_object().ok_or_else(|| {
            HubError::Hub(format!("safetensors tensor {name:?} is not an object"))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| HubError::Hub(format!("safetensors tensor {name:?} has no dtype")))?;
        let element_bytes = match dtype {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1u64,
            "I16" | "U16" | "F16" | "BF16" => 2,
            "I32" | "U32" | "F32" => 4,
            "I64" | "U64" | "F64" => 8,
            _ => {
                return Err(HubError::Hub(format!(
                    "safetensors tensor {name:?} has unsupported dtype {dtype:?}"
                )))
            }
        };
        let shape = tensor
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| HubError::Hub(format!("safetensors tensor {name:?} has no shape")))?;
        if shape.iter().any(|dim| dim.as_u64().is_none()) {
            return Err(HubError::Hub(format!(
                "safetensors tensor {name:?} has an invalid shape"
            )));
        }
        let elements = shape.iter().try_fold(1u64, |elements, dim| {
            elements
                .checked_mul(dim.as_u64().expect("shape validated"))
                .ok_or_else(|| {
                    HubError::Hub(format!("safetensors tensor {name:?} shape overflows"))
                })
        })?;
        let expected_bytes = elements.checked_mul(element_bytes).ok_or_else(|| {
            HubError::Hub(format!("safetensors tensor {name:?} byte size overflows"))
        })?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                HubError::Hub(format!(
                    "safetensors tensor {name:?} has invalid data_offsets"
                ))
            })?;
        let start = offsets[0].as_u64().ok_or_else(|| {
            HubError::Hub(format!(
                "safetensors tensor {name:?} has invalid start offset"
            ))
        })?;
        let end = offsets[1].as_u64().ok_or_else(|| {
            HubError::Hub(format!(
                "safetensors tensor {name:?} has invalid end offset"
            ))
        })?;
        if end < start || payload_bytes.is_some_and(|payload| end > payload) {
            return Err(HubError::Hub(format!(
                "safetensors tensor {name:?} has out-of-range data_offsets [{start}, {end}]"
            )));
        }
        if end - start != expected_bytes {
            return Err(HubError::Hub(format!(
                "safetensors tensor {name:?} shape and dtype require {expected_bytes} bytes, \
                 but data_offsets contain {}",
                end - start
            )));
        }
        tensor_bytes = tensor_bytes.saturating_add(end - start);
        intervals.push((start, end, name.as_str()));
        weight_map.insert(name.clone(), SAFETENSORS_FILE.into());
        tensor_metadata.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape}),
        );
    }
    if weight_map.is_empty() {
        return Err(HubError::Hub(
            "safetensors header contains no tensor entries".into(),
        ));
    }
    intervals.sort_unstable_by_key(|&(start, _, _)| start);
    for pair in intervals.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(HubError::Hub(format!(
                "safetensors tensors {:?} and {:?} have overlapping data_offsets",
                pair[0].2, pair[1].2
            )));
        }
    }
    serde_json::to_string(&serde_json::json!({
        "metadata": {"total_size": tensor_bytes},
        "weight_map": weight_map,
        "plow_tensor_metadata": tensor_metadata,
    }))
    .map_err(|e| HubError::Hub(format!("cannot synthesize safetensors index: {e}")))
}

fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let range = value.strip_prefix("bytes ")?;
    let (bounds, total) = range.split_once('/')?;
    let (start, end) = bounds.split_once('-')?;
    let parsed = (start.parse().ok()?, end.parse().ok()?, total.parse().ok()?);
    (parsed.0 <= parsed.1 && parsed.1 < parsed.2).then_some(parsed)
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
    use std::io::Write as _;
    use std::sync::{Arc, Mutex};

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

    fn safetensors_fixture() -> Vec<u8> {
        let mut header = serde_json::to_vec(&serde_json::json!({
            "__metadata__": {"format": "pt"},
            "model.embed_tokens.weight": {
                "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]
            },
            "lm_head.weight": {
                "dtype": "BF16", "shape": [1, 2], "data_offsets": [8, 12]
            }
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + 12);
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&[0; 12]);
        file
    }

    fn range_server(
        file: Vec<u8>,
        status: u16,
        requests: usize,
    ) -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let thread_seen = Arc::clone(&seen);
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming().take(requests) {
                let mut stream = stream.unwrap();
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while request.len() < 16 * 1024 {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                    if request.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8(request).unwrap();
                let range = request
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("Range: ")
                            .or_else(|| line.strip_prefix("range: "))
                    })
                    .unwrap()
                    .to_string();
                thread_seen.lock().unwrap().push(range.clone());
                let (start, end) = range
                    .strip_prefix("bytes=")
                    .and_then(|range| range.split_once('-'))
                    .map(|(start, end)| {
                        (
                            start.parse::<usize>().unwrap(),
                            end.parse::<usize>().unwrap(),
                        )
                    })
                    .unwrap();
                let body = &file[start..=end];
                write!(
                    stream,
                    "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
                    body.len(),
                    file.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        (
            format!("http://{address}/org/model/resolve/deadbeef/{SAFETENSORS_FILE}"),
            seen,
            handle,
        )
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
        assert_eq!(metadata.indexed_total_size().unwrap(), None);
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
    fn copying_metadata_to_its_source_is_rejected_without_deleting_files() {
        let dir = tempdir("copy-self");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"fixture"}"#).unwrap();
        std::fs::write(
            dir.join(SAFETENSORS_INDEX),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();
        std::fs::create_dir(dir.join("chat_templates")).unwrap();
        let template = dir.join("chat_templates/tool_use.jinja");
        std::fs::write(&template, "{{ tools }}").unwrap();

        let metadata = resolve_model_metadata(dir.to_str().unwrap()).unwrap();
        let error = metadata.copy_to(&dir).unwrap_err().to_string();
        assert!(error.contains("onto its source directory"), "{error}");
        assert_eq!(std::fs::read_to_string(&template).unwrap(), "{{ tools }}");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn copying_metadata_removes_stale_nested_chat_templates() {
        let first = tempdir("templates-first");
        std::fs::write(first.join("config.json"), r#"{"model_type":"fixture"}"#).unwrap();
        std::fs::write(
            first.join(SAFETENSORS_INDEX),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();
        std::fs::create_dir(first.join("chat_templates")).unwrap();
        std::fs::write(first.join("chat_templates/tool.jinja"), "{{ tools }}").unwrap();

        let second = tempdir("templates-second");
        std::fs::write(second.join("config.json"), r#"{"model_type":"fixture"}"#).unwrap();
        std::fs::write(
            second.join(SAFETENSORS_INDEX),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();

        let out = tempdir("templates-out");
        resolve_model_metadata(first.to_str().unwrap())
            .unwrap()
            .copy_to(&out)
            .unwrap();
        assert!(out.join("chat_templates/tool.jinja").is_file());
        resolve_model_metadata(second.to_str().unwrap())
            .unwrap()
            .copy_to(&out)
            .unwrap();
        assert!(!out.join("chat_templates").exists());

        std::fs::remove_dir_all(first).ok();
        std::fs::remove_dir_all(second).ok();
        std::fs::remove_dir_all(out).ok();
    }

    #[test]
    fn manifest_marks_multimodal_checkpoints_as_text_only_compilation() {
        let dir = tempdir("vision-scope");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"fixture","text_config":{},"vision_config":{"hidden_size":16},
                "audio_config":{"hidden_size":16}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let metadata = resolve_model_metadata(dir.to_str().unwrap()).unwrap();
        assert!(metadata.has_vision_config());
        assert!(metadata.has_audio_config());
        assert_eq!(metadata.revision(), None);
        let out = tempdir("vision-scope-out");
        metadata.copy_to(&out).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("hf_metadata.json")).unwrap()).unwrap();
        assert_eq!(manifest["compile_scope"], "text_generation");
        assert_eq!(
            manifest["source_modalities"],
            serde_json::json!(["text", "vision", "audio"])
        );
        assert_eq!(manifest["compiled_modalities"], serde_json::json!(["text"]));
        assert!(!checkpoint_tensor_is_text(
            "model.vision_embedder.patch_dense.weight",
            false
        ));
        assert!(!checkpoint_tensor_is_text(
            "model.embed_audio.embedding_projection.weight",
            false
        ));

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(out).ok();
    }

    #[test]
    fn only_declared_deepseek_kimi_mtp_layer_range_is_excluded() {
        let deepseek = r#"{"model_type":"deepseek_v3","num_hidden_layers":61,
            "num_nextn_predict_layers":1}"#;
        let range = omitted_mtp_layer_range(deepseek);
        assert_eq!(range, Some((61, 62)));
        assert!(!checkpoint_tensor_is_omitted_mtp(
            "model.layers.60.mlp.gate.weight",
            range
        ));
        assert!(checkpoint_tensor_is_omitted_mtp(
            "model.layers.61.shared_head.head.weight",
            range
        ));
        assert!(!checkpoint_tensor_is_omitted_mtp(
            "model.layers.62.mlp.gate.weight",
            range
        ));

        let glm = r#"{"model_type":"glm_moe_dsa","num_hidden_layers":78,
            "num_nextn_predict_layers":1}"#;
        assert_eq!(omitted_mtp_layer_range(glm), None);
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
    fn cached_main_can_reuse_a_sha_revision_snapshot_and_synthetic_index() {
        use hf_hub::{Cache, Repo, RepoType};

        let root = tempdir("cache-sha-root");
        let main_repo = Repo::new("org/model".into(), RepoType::Model);
        let sha_repo = Repo::with_revision("org/model".into(), RepoType::Model, "deadbeef".into());
        let snapshot = root
            .join(main_repo.folder_name())
            .join("snapshots")
            .join("deadbeef");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(
            snapshot.join(SAFETENSORS_INDEX),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();

        let cache = Cache::new(root.clone());
        cache.repo(sha_repo).create_ref("deadbeef").unwrap();
        let cached_main = cache.repo(main_repo);
        cached_main.create_ref("deadbeef").unwrap();

        let metadata = metadata_from_cached_main("org/model", &cached_main).unwrap();
        assert_eq!(metadata.revision(), Some("deadbeef"));
        assert!(metadata.path(SAFETENSORS_INDEX).is_some());

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

    #[test]
    fn advertised_optional_download_failures_are_recorded_not_fatal() {
        let names = vec![
            "added_tokens.json".to_string(),
            "config.json".to_string(),
            SAFETENSORS_INDEX.to_string(),
            "tokenizer.json".to_string(),
        ];
        let (files, unresolved) = collect_advertised_metadata("org/model", names, |name| {
            if matches!(name, "added_tokens.json" | "tokenizer.json") {
                Err("gated".to_string())
            } else {
                Ok(PathBuf::from(name))
            }
        })
        .unwrap();

        assert_eq!(
            files
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["config.json", SAFETENSORS_INDEX]
        );
        assert_eq!(unresolved, vec!["added_tokens.json", "tokenizer.json"]);
    }

    #[test]
    fn advertised_required_download_failure_is_fatal() {
        for required in ["config.json", SAFETENSORS_INDEX] {
            let error =
                collect_advertised_metadata("org/model", vec![required.to_string()], |_| {
                    Err("denied".to_string())
                })
                .unwrap_err();
            assert!(error.to_string().contains("required"), "{error}");
            assert!(error.to_string().contains(required), "{error}");
        }
    }

    #[test]
    fn unresolved_optional_files_are_emitted_in_metadata_manifest() {
        let dir = tempdir("optional-source");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"fixture"}"#).unwrap();
        std::fs::write(
            dir.join(SAFETENSORS_INDEX),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();
        let files = vec![
            ("config.json".to_string(), dir.join("config.json")),
            (SAFETENSORS_INDEX.to_string(), dir.join(SAFETENSORS_INDEX)),
        ];
        let metadata = metadata_from_files(
            "org/model",
            Some("deadbeef".into()),
            files,
            None,
            vec!["added_tokens.json".into()],
        )
        .unwrap();
        let out = tempdir("optional-out");
        metadata.copy_to(&out).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("hf_metadata.json")).unwrap()).unwrap();
        assert_eq!(
            manifest["unresolved_optional_files"],
            serde_json::json!(["added_tokens.json"])
        );

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(out).ok();
    }

    #[test]
    fn local_monolithic_safetensors_synthesizes_index_without_copying_payload() {
        let dir = tempdir("monolithic");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"fixture"}"#).unwrap();
        std::fs::write(dir.join(SAFETENSORS_FILE), safetensors_fixture()).unwrap();

        let metadata = resolve_model_metadata(dir.to_str().unwrap()).unwrap();
        assert!(metadata.path(SAFETENSORS_INDEX).is_none());
        assert!(metadata.filenames().any(|name| name == SAFETENSORS_INDEX));

        let out = tempdir("monolithic-out");
        metadata.copy_to(&out).unwrap();
        assert!(!out.join(SAFETENSORS_FILE).exists());
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join(SAFETENSORS_INDEX)).unwrap()).unwrap();
        assert_eq!(index["weight_map"].as_object().unwrap().len(), 2);
        assert!(index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .all(|shard| shard == SAFETENSORS_FILE));
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("hf_metadata.json")).unwrap()).unwrap();
        assert_eq!(manifest["safetensors_index"]["synthetic"], true);
        assert_eq!(
            manifest["safetensors_index"]["source_file"],
            SAFETENSORS_FILE
        );
        assert_eq!(manifest["weight_shards_downloaded"], false);

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(out).ok();
    }

    #[test]
    fn remote_monolithic_safetensors_fetches_only_two_bounded_ranges() {
        let file = safetensors_fixture();
        let header_len = u64::from_le_bytes(file[..8].try_into().unwrap());
        let (url, seen, server) = range_server(file, 206, 2);

        let index = fetch_remote_safetensors_index(&url, None).unwrap();
        server.join().unwrap();
        let index: serde_json::Value = serde_json::from_str(&index).unwrap();
        assert_eq!(index["weight_map"].as_object().unwrap().len(), 2);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                "bytes=0-7".to_string(),
                format!("bytes=8-{}", 7 + header_len),
            ]
        );
        assert!(url.contains("/resolve/deadbeef/model.safetensors"));
    }

    #[test]
    fn remote_monolithic_safetensors_rejects_servers_that_ignore_range() {
        let (url, seen, server) = range_server(safetensors_fixture(), 200, 1);
        let error = fetch_remote_safetensors_index(&url, None).unwrap_err();
        server.join().unwrap();
        assert!(error.to_string().contains("expected HTTP 206"), "{error}");
        assert_eq!(*seen.lock().unwrap(), vec!["bytes=0-7".to_string()]);
    }

    #[test]
    fn monolithic_safetensors_rejects_oversized_header_before_allocating() {
        let bytes = (MAX_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes();
        let error = synthesize_index_from_reader(bytes.as_slice(), None).unwrap_err();
        assert!(error
            .to_string()
            .contains("invalid safetensors header length"));
    }

    #[test]
    fn monolithic_safetensors_validates_tensor_layout() {
        let check = |value: serde_json::Value, payload| {
            let header = serde_json::to_vec(&value).unwrap();
            synthesize_index_from_header(
                &header,
                header.len() as u64,
                Some(8 + header.len() as u64 + payload),
            )
        };

        let bad_dtype = check(
            serde_json::json!({"w":{"dtype":"NOPE","shape":[1],"data_offsets":[0,1]}}),
            1,
        )
        .unwrap_err();
        assert!(bad_dtype.to_string().contains("unsupported dtype"));

        let bad_size = check(
            serde_json::json!({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}),
            2,
        )
        .unwrap_err();
        assert!(bad_size.to_string().contains("require 4 bytes"));

        let overlap = check(
            serde_json::json!({
                "a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},
                "b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}
            }),
            3,
        )
        .unwrap_err();
        assert!(overlap.to_string().contains("overlapping"));
    }

    #[test]
    fn checkpoint_aliases_preserve_wrapped_name_matching() {
        assert_eq!(
            checkpoint_name_aliases("model.language_model.layers.0.weight")
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([
                "model.language_model.layers.0.weight",
                "language_model.layers.0.weight",
                "layers.0.weight",
            ])
        );
        assert_eq!(
            checkpoint_name_aliases("layers.0.weight").collect::<Vec<_>>(),
            vec!["layers.0.weight"]
        );
    }
}
