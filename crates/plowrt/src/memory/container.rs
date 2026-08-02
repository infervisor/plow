//! §C Layer 0 — checkpoint container abstraction.
//!
//! A [`Container`] enumerates a checkpoint's tensors as [`TensorDesc`] records
//! (name, dtype, shape, byte range, shard) without committing to *how* the
//! bytes get read. The streaming loader (Layer 2) consumes this: it splits each
//! descriptor into work units, reads them, and tiles on-device.
//!
//! The only impl today is [`Safetensors`]. It replaces the probe-by-`access()`
//! logic in `runtime/tests/gemma4_chat.c`, which had two bugs this module
//! exists to not repeat:
//!
//! * a `MAX_SHARD 8` ceiling with prefix-only probing, so
//!   `model-0000N-of-00002.partial.safetensors` (the Gemma-4 31B partial
//!   checkpoint) matched nothing at all, and
//! * a single-file `model.safetensors` fallback that was documented but never
//!   written, so the Gemma-4 12B checkpoint — one unsharded 23 GB file with no
//!   index — could not load.
//!
//! Discovery here is a single `readdir` + filename parse, so there is no shard
//! ceiling and `.partial.` naming is a first-class case. Ambiguity is an error:
//! two different `-of-NNNNN` totals in one directory, or a shard set with a
//! hole, hard-fails naming the files rather than silently loading a subset.
//! A silently-absent weight still produces fluent text; that is the failure
//! mode this whole path is built to prevent.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::{Result, RuntimeError};

/// Element type of a checkpoint tensor. Only the widths the loader must size
/// and validate are modelled; unknown dtype strings are an error, not a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F16,
    F32,
    F64,
    /// `F8_E4M3` — the quantized weight dtype used by the fp8 checkpoints.
    F8E4M3,
    F8E5M2,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Bool,
}

impl DType {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Bool => 1,
            DType::Bf16 | DType::F16 | DType::I16 | DType::U16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
        }
    }

    fn parse(s: &str) -> Option<DType> {
        Some(match s {
            "BF16" => DType::Bf16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            "F64" => DType::F64,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "I8" => DType::I8,
            "U8" => DType::U8,
            "I16" => DType::I16,
            "U16" => DType::U16,
            "I32" => DType::I32,
            "U32" => DType::U32,
            "I64" => DType::I64,
            "U64" => DType::U64,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }
}

/// One tensor in a checkpoint: where its bytes live and what they mean.
///
/// `byte_range` is an absolute file offset range into `shard_path(shard)` — the
/// container has already added the safetensors header length, so a reader can
/// `pread` it directly without re-deriving the data section base.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_range: Range<u64>,
    pub shard: u16,
}

impl TensorDesc {
    pub fn nbytes(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }

    /// Element count implied by `shape`.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A checkpoint container: an enumerable set of tensors over one or more files.
pub trait Container {
    /// Every tensor, in the container's own order.
    fn tensors(&self) -> &[TensorDesc];

    /// Path of the file holding shard `shard`.
    fn shard_path(&self, shard: u16) -> &Path;

    /// Look a tensor up by name, hard-failing *with the name* when absent.
    fn get(&self, name: &str) -> Result<&TensorDesc>;

    /// Raw on-disk bytes for a tensor (no layout transform).
    fn bytes(&self, desc: &TensorDesc) -> Result<&[u8]>;
}

/// A safetensors checkpoint: sharded, single-file, or `.partial.`-named.
pub struct Safetensors {
    shards: Vec<PathBuf>,
    mmaps: Vec<memmap2::Mmap>,
    tensors: Vec<TensorDesc>,
    by_name: HashMap<String, usize>,
}

/// Concise on purpose: dumping 677 `TensorDesc`s into a panic message helps
/// nobody.
impl std::fmt::Debug for Safetensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Safetensors {{ shards: {}, tensors: {}, bytes: {} }}",
            self.shards.len(),
            self.tensors.len(),
            self.total_bytes()
        )
    }
}

/// A filename that the safetensors shard naming scheme recognizes.
struct ShardName {
    index: u32,
    total: u32,
    partial: bool,
}

/// Parse `model-{NNNNN}-of-{MMMMM}.safetensors` and the `.partial.` variant.
///
/// Deliberately strict: the suffix must be exactly `.safetensors`, so sidecars
/// like `model-00001-of-00002.safetensors.header.json` (present in the 31B
/// partial checkpoint) do not match. Widths are not fixed at 5 — the digits are
/// parsed as integers, so there is no shard ceiling.
fn parse_shard_name(file: &str) -> Option<ShardName> {
    let rest = file.strip_prefix("model-")?;
    let (idx, rest) = rest.split_once("-of-")?;
    let (total, tail) = match rest.strip_suffix(".partial.safetensors") {
        Some(t) => (t, true),
        None => (rest.strip_suffix(".safetensors")?, false),
    };
    if idx.is_empty() || total.is_empty() {
        return None;
    }
    Some(ShardName {
        index: idx.parse().ok()?,
        total: total.parse().ok()?,
        partial: tail,
    })
}

impl Safetensors {
    /// Discover and open a checkpoint directory.
    ///
    /// Resolution order:
    /// 1. sharded `model-{i}-of-{n}.safetensors`,
    /// 2. sharded `model-{i}-of-{n}.partial.safetensors`,
    /// 3. single-file `model.safetensors`.
    ///
    /// A complete non-partial set wins over a partial set of the same total (a
    /// partial checkpoint is a subset by construction). Everything else that is
    /// ambiguous — two different totals, a missing shard — is an error.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let rd = std::fs::read_dir(dir).map_err(|source| RuntimeError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        // total -> (partial?) -> index -> filename
        let mut sets: HashMap<(u32, bool), HashMap<u32, PathBuf>> = HashMap::new();
        let mut single: Option<PathBuf> = None;
        for ent in rd {
            let ent = ent.map_err(|source| RuntimeError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let name = ent.file_name();
            let Some(name) = name.to_str() else { continue };
            if name == "model.safetensors" {
                single = Some(ent.path());
                continue;
            }
            if let Some(s) = parse_shard_name(name) {
                sets.entry((s.total, s.partial))
                    .or_default()
                    .insert(s.index, ent.path());
            }
        }

        let paths = if !sets.is_empty() {
            // Prefer a complete non-partial set; fall back to partial.
            let mut keys: Vec<_> = sets.keys().copied().collect();
            keys.sort_by_key(|&(total, partial)| (partial, total));
            let complete: Vec<_> = keys
                .iter()
                .copied()
                .filter(|k| sets[k].len() as u32 == k.0)
                .collect();
            let chosen = match complete.as_slice() {
                [] => {
                    // Nothing complete: name the holes in the best candidate.
                    let k = keys[0];
                    let set = &sets[&k];
                    let missing: Vec<u32> = (1..=k.0).filter(|i| !set.contains_key(i)).collect();
                    return Err(RuntimeError::Msg(format!(
                        "{}: incomplete safetensors shard set (-of-{:05}{}): have {} of {}, \
                         missing shard indices {missing:?}",
                        dir.display(),
                        k.0,
                        if k.1 { " .partial" } else { "" },
                        set.len(),
                        k.0,
                    )));
                }
                [only] => *only,
                // Two complete sets differing only in .partial: the full one wins.
                [a, b] if a.0 == b.0 && a.1 != b.1 => {
                    if a.1 {
                        *b
                    } else {
                        *a
                    }
                }
                many => {
                    return Err(RuntimeError::Msg(format!(
                        "{}: ambiguous checkpoint — {} complete shard sets present ({}); \
                         a stray shard-named file silently changes what loads",
                        dir.display(),
                        many.len(),
                        many.iter()
                            .map(|(t, p)| format!(
                                "-of-{t:05}{}",
                                if *p { " .partial" } else { "" }
                            ))
                            .collect::<Vec<_>>()
                            .join(", "),
                    )));
                }
            };
            let set = &sets[&chosen];
            (1..=chosen.0).map(|i| set[&i].clone()).collect::<Vec<_>>()
        } else if let Some(p) = single {
            vec![p]
        } else {
            return Err(RuntimeError::Msg(format!(
                "{}: no safetensors checkpoint found (looked for \
                 model-{{i}}-of-{{n}}[.partial].safetensors and model.safetensors)",
                dir.display()
            )));
        };

        Self::open_files(paths)
    }

    /// Open an explicit, already-ordered list of shard files.
    pub fn open_files(paths: Vec<PathBuf>) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut tensors = Vec::new();
        let mut by_name = HashMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let file = std::fs::File::open(path).map_err(|source| RuntimeError::Io {
                path: path.clone(),
                source,
            })?;
            // SAFETY: read-only checkpoint, held for the container's lifetime.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| RuntimeError::Io {
                path: path.clone(),
                source,
            })?;
            Self::parse_header(path, &mmap, shard as u16, &mut tensors, &mut by_name)?;
            mmaps.push(mmap);
        }
        Ok(Safetensors {
            shards: paths,
            mmaps,
            tensors,
            by_name,
        })
    }

    /// Parse one shard's JSON header into `TensorDesc`s.
    ///
    /// Validates, per tensor: known dtype, `shape.product() * dtype.size()` ==
    /// declared byte length, and byte range inside the file. Each failure names
    /// the tensor — a short or truncated weight must never reach the arena.
    fn parse_header(
        path: &Path,
        mmap: &[u8],
        shard: u16,
        tensors: &mut Vec<TensorDesc>,
        by_name: &mut HashMap<String, usize>,
    ) -> Result<()> {
        if mmap.len() < 8 {
            return Err(RuntimeError::Msg(format!(
                "{}: file is {} bytes, too short to hold a safetensors header length",
                path.display(),
                mmap.len()
            )));
        }
        let hlen = u64::from_le_bytes(mmap[..8].try_into().unwrap());
        let data0 = 8u64
            .checked_add(hlen)
            .filter(|&d| d <= mmap.len() as u64)
            .ok_or_else(|| {
                RuntimeError::Msg(format!(
                    "{}: header length {hlen} exceeds file size {}",
                    path.display(),
                    mmap.len()
                ))
            })?;
        let hdr: serde_json::Value =
            serde_json::from_slice(&mmap[8..data0 as usize]).map_err(|source| {
                RuntimeError::Json {
                    path: path.to_path_buf(),
                    source,
                }
            })?;
        let obj = hdr.as_object().ok_or_else(|| {
            RuntimeError::Msg(format!(
                "{}: safetensors header is not an object",
                path.display()
            ))
        })?;

        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let bad = |why: String| {
                RuntimeError::Msg(format!("{}: tensor '{name}': {why}", path.display()))
            };
            let dt_s = v["dtype"]
                .as_str()
                .ok_or_else(|| bad("header entry has no string 'dtype'".into()))?;
            let dtype =
                DType::parse(dt_s).ok_or_else(|| bad(format!("unsupported dtype '{dt_s}'")))?;
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| bad("header entry has no array 'shape'".into()))?
                .iter()
                .map(|x| x.as_u64().map(|u| u as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| bad("non-integer extent in 'shape'".into()))?;
            let off = v["data_offsets"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| bad("header entry has no 2-element 'data_offsets'".into()))?;
            let (a, b) = (
                off[0]
                    .as_u64()
                    .ok_or_else(|| bad("bad data_offsets[0]".into()))?,
                off[1]
                    .as_u64()
                    .ok_or_else(|| bad("bad data_offsets[1]".into()))?,
            );
            if b < a {
                return Err(bad(format!("inverted data_offsets [{a}, {b}]")));
            }
            let nbytes = b - a;
            let want = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
            if nbytes != want {
                return Err(bad(format!(
                    "short tensor: shape {shape:?} of {dt_s} needs {want} bytes, \
                     header declares {nbytes}"
                )));
            }
            let start = data0 + a;
            let end = data0 + b;
            if end > mmap.len() as u64 {
                return Err(bad(format!(
                    "truncated: bytes [{start}, {end}) run past end of file ({} bytes)",
                    mmap.len()
                )));
            }
            if let Some(&prev) = by_name.get(name.as_str()) {
                return Err(bad(format!(
                    "duplicate tensor, also in shard {}",
                    tensors[prev].shard
                )));
            }
            by_name.insert(name.clone(), tensors.len());
            tensors.push(TensorDesc {
                name: name.clone(),
                dtype,
                shape,
                byte_range: start..end,
                shard,
            });
        }
        Ok(())
    }

    /// Total bytes of tensor data across all shards.
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shards
    }
}

impl Container for Safetensors {
    fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    fn shard_path(&self, shard: u16) -> &Path {
        &self.shards[shard as usize]
    }

    fn get(&self, name: &str) -> Result<&TensorDesc> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| {
                RuntimeError::Msg(format!(
                    "checkpoint has no tensor '{name}' (searched {} tensors across {} shard(s) \
                     starting {})",
                    self.tensors.len(),
                    self.shards.len(),
                    self.shards[0].display(),
                ))
            })
    }

    fn bytes(&self, desc: &TensorDesc) -> Result<&[u8]> {
        let m = self.mmaps.get(desc.shard as usize).ok_or_else(|| {
            RuntimeError::Msg(format!(
                "tensor '{}': shard {} out of range ({} shards)",
                desc.name,
                desc.shard,
                self.mmaps.len()
            ))
        })?;
        m.get(desc.byte_range.start as usize..desc.byte_range.end as usize)
            .ok_or_else(|| {
                RuntimeError::Msg(format!(
                    "tensor '{}': byte range {:?} outside shard {} ({} bytes)",
                    desc.name,
                    desc.byte_range,
                    desc.shard,
                    m.len()
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_names_parse_including_partial() {
        let s = parse_shard_name("model-00001-of-00003.safetensors").unwrap();
        assert_eq!((s.index, s.total, s.partial), (1, 3, false));
        let s = parse_shard_name("model-00002-of-00002.partial.safetensors").unwrap();
        assert_eq!((s.index, s.total, s.partial), (2, 2, true));
        // No shard ceiling: the old C probe stopped at 8.
        let s = parse_shard_name("model-00042-of-00061.safetensors").unwrap();
        assert_eq!((s.index, s.total), (42, 61));
        // Sidecars must not match.
        assert!(parse_shard_name("model-00001-of-00002.safetensors.header.json").is_none());
        assert!(parse_shard_name("model.safetensors").is_none());
        assert!(parse_shard_name("consolidated-00001-of-00002.safetensors").is_none());
    }

    /// Build a minimal one-tensor safetensors file in `dir`.
    fn write_st(path: &Path, name: &str, dtype: &str, shape: &[usize], data: &[u8]) {
        let hdr = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[0,{}]}}}}"#,
            data.len()
        );
        let mut buf = (hdr.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(hdr.as_bytes());
        buf.extend_from_slice(data);
        std::fs::write(path, buf).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plow-container-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn single_file_fallback_loads() {
        let d = tmpdir("single");
        write_st(
            &d.join("model.safetensors"),
            "w",
            "BF16",
            &[2, 2],
            &[1u8; 8],
        );
        let c = Safetensors::open_dir(&d).unwrap();
        assert_eq!(c.tensors().len(), 1);
        assert_eq!(c.get("w").unwrap().dtype, DType::Bf16);
        assert_eq!(c.bytes(c.get("w").unwrap()).unwrap(), &[1u8; 8]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn partial_shard_set_loads() {
        let d = tmpdir("partial");
        write_st(
            &d.join("model-00001-of-00002.partial.safetensors"),
            "a",
            "F32",
            &[2],
            &[0u8; 8],
        );
        write_st(
            &d.join("model-00002-of-00002.partial.safetensors"),
            "b",
            "F32",
            &[2],
            &[0u8; 8],
        );
        // Sidecar headers like the real 31B dir must be ignored.
        std::fs::write(
            d.join("model-00001-of-00002.safetensors.header.json"),
            b"{}",
        )
        .unwrap();
        let c = Safetensors::open_dir(&d).unwrap();
        assert_eq!(c.tensors().len(), 2);
        assert_eq!(c.get("b").unwrap().shard, 1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn missing_tensor_names_itself() {
        let d = tmpdir("missing");
        write_st(
            &d.join("model.safetensors"),
            "w",
            "BF16",
            &[2, 2],
            &[1u8; 8],
        );
        let c = Safetensors::open_dir(&d).unwrap();
        let e = c
            .get("model.layers.0.mlp.up_proj.weight")
            .unwrap_err()
            .to_string();
        assert!(e.contains("model.layers.0.mlp.up_proj.weight"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn short_tensor_hard_fails_with_name() {
        let d = tmpdir("short");
        // shape [2,2] BF16 needs 8 bytes; declare 6.
        write_st(
            &d.join("model.safetensors"),
            "q_proj",
            "BF16",
            &[2, 2],
            &[0u8; 6],
        );
        let e = Safetensors::open_dir(&d).unwrap_err().to_string();
        assert!(e.contains("q_proj") && e.contains("short tensor"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn truncated_file_hard_fails_with_name() {
        let d = tmpdir("trunc");
        let p = d.join("model.safetensors");
        write_st(&p, "k_proj", "BF16", &[2, 2], &[0u8; 8]);
        // Lop off the last 4 data bytes.
        let mut b = std::fs::read(&p).unwrap();
        b.truncate(b.len() - 4);
        std::fs::write(&p, b).unwrap();
        let e = Safetensors::open_dir(&d).unwrap_err().to_string();
        assert!(e.contains("k_proj") && e.contains("truncated"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn incomplete_shard_set_names_the_hole() {
        let d = tmpdir("hole");
        write_st(
            &d.join("model-00001-of-00003.safetensors"),
            "a",
            "F32",
            &[1],
            &[0u8; 4],
        );
        write_st(
            &d.join("model-00003-of-00003.safetensors"),
            "c",
            "F32",
            &[1],
            &[0u8; 4],
        );
        let e = Safetensors::open_dir(&d).unwrap_err().to_string();
        assert!(e.contains("missing shard indices [2]"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The fp8 directory hazard: a stray shard-named file must not silently
    /// change what loads.
    #[test]
    fn ambiguous_shard_sets_hard_fail() {
        let d = tmpdir("ambig");
        write_st(
            &d.join("model-00001-of-00001.safetensors"),
            "a",
            "F32",
            &[1],
            &[0u8; 4],
        );
        write_st(
            &d.join("model-00001-of-00002.safetensors"),
            "b",
            "F32",
            &[1],
            &[0u8; 4],
        );
        write_st(
            &d.join("model-00002-of-00002.safetensors"),
            "c",
            "F32",
            &[1],
            &[0u8; 4],
        );
        let e = Safetensors::open_dir(&d).unwrap_err().to_string();
        assert!(e.contains("ambiguous checkpoint"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn empty_dir_hard_fails() {
        let d = tmpdir("empty");
        let e = Safetensors::open_dir(&d).unwrap_err().to_string();
        assert!(e.contains("no safetensors checkpoint found"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
