//! Build identity: what an [`Inventory`](crate::Inventory) was derived from.
//!
//! An inventory is **derived data**, not a registry. Nothing registers into it;
//! it is read out of one built interpreter object and can be recreated from that
//! object at any time. That is the whole reason this type exists: derived data
//! without the identity of its source is indistinguishable from a hand-written
//! claim, and a hand-written claim is exactly what silently disagrees with the
//! object you are compiling for.
//!
//! The same `interp_sm120.cu` becomes eight different objects under different
//! `-D` flags (`runtime/CMakeLists.txt:127-320`), and `interp_sm90a.cu` is that
//! same file included with `PLOW_NV_HOPPER=1`. "Which kernels exist" is a
//! question about one of those objects, never about the source tree, so a
//! [`BuildId`] names the object.

use std::collections::BTreeMap;

use hwspec::IsaLevel;

/// Identifies the one build an inventory describes.
///
/// Equality is what makes staleness detectable: if a `BuildId` re-derived from
/// the current sources differs from the one stored alongside an inventory, the
/// inventory describes an object nobody is building any more.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BuildId {
    /// Instruction set the object was compiled for.
    pub isa: IsaLevel,
    /// Preprocessor definitions, normalized and sorted. These are the whole
    /// difference between the eight NVIDIA interpreter objects, so they are part
    /// of the identity rather than metadata.
    pub defines: Vec<String>,
    /// Compiler identity, e.g. `cuda-13.0`. The sm90a interpreter's register
    /// count moved by 58 between the figure documented in the build script and
    /// what this toolchain produces, so this is load bearing.
    pub toolchain: String,
    /// Digest of the source the arms were read out of. Cheap content hash, not
    /// a cryptographic commitment — it exists to notice edits, not to resist
    /// forgery.
    pub source_digest: String,
}

impl BuildId {
    pub fn new(
        isa: IsaLevel,
        defines: impl IntoIterator<Item = String>,
        toolchain: impl Into<String>,
        source_digest: impl Into<String>,
    ) -> Self {
        let mut defines: Vec<String> = defines.into_iter().map(|d| normalize_define(&d)).collect();
        defines.sort();
        defines.dedup();
        BuildId {
            isa,
            defines,
            toolchain: toolchain.into(),
            source_digest: source_digest.into(),
        }
    }

    /// A short, filesystem-safe label for this build, used to name the stored
    /// inventory. Distinct builds must not collide, so the defines participate.
    pub fn label(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for part in self
            .defines
            .iter()
            .map(String::as_str)
            .chain([self.toolchain.as_str(), self.source_digest.as_str()])
        {
            for b in part.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            h ^= 0xff;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        format!("{}-{:016x}", self.isa.arch_flag(), h)
    }

    /// Whether `self` describes the same object as `other`, and if not, what
    /// differs. Returning the reasons rather than a bool is deliberate: "your
    /// inventory is stale" is not actionable, "your inventory was built with
    /// `PLOW_NV_PREFILL=1` and you are compiling without it" is.
    pub fn differences(&self, other: &BuildId) -> Vec<String> {
        let mut out = Vec::new();
        if self.isa != other.isa {
            out.push(format!(
                "isa: inventory {} vs build {}",
                self.isa.arch_flag(),
                other.isa.arch_flag()
            ));
        }
        if self.toolchain != other.toolchain {
            out.push(format!(
                "toolchain: inventory {} vs build {}",
                self.toolchain, other.toolchain
            ));
        }
        if self.source_digest != other.source_digest {
            out.push("source: the interpreter source has changed since this was probed".into());
        }
        let mine: BTreeMap<&str, ()> = self.defines.iter().map(|d| (d.as_str(), ())).collect();
        let theirs: BTreeMap<&str, ()> = other.defines.iter().map(|d| (d.as_str(), ())).collect();
        for d in mine.keys() {
            if !theirs.contains_key(d) {
                out.push(format!("define only in the inventory: {d}"));
            }
        }
        for d in theirs.keys() {
            if !mine.contains_key(d) {
                out.push(format!("define only in the build: {d}"));
            }
        }
        out
    }
}

/// `-DFOO=1` / `FOO=1` / `FOO` all name the same switch. Normalizing means a
/// build described two ways still hashes to one identity.
fn normalize_define(d: &str) -> String {
    let d = d.trim();
    let d = d.strip_prefix("-D").unwrap_or(d);
    match d.split_once('=') {
        // `FOO=1` and bare `FOO` are the same to the preprocessor.
        Some((k, "1")) => k.to_string(),
        _ => d.to_string(),
    }
}

/// Where an inventory's contents came from.
///
/// There is deliberately no `Declared` variant. A hand-written kernel table is
/// the failure this crate exists to prevent, so it is not representable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Read out of a built object by the probe.
    Probed(BuildId),
}

impl Provenance {
    pub fn build(&self) -> &BuildId {
        match self {
            Provenance::Probed(b) => b,
        }
    }
}

/// Content digest of preprocessed output, ignoring the compiler's line markers.
///
/// A preprocessed TU is prefixed with `# <line> "<path>"` markers carrying
/// absolute paths, which would make the digest machine-dependent. Stripping
/// them leaves the actual code — including everything `#include`d, so the
/// tile macros in `op_gemm.cuh` and the kernel bodies are all in scope. The
/// digest therefore changes when an included kernel changes, which hashing only
/// the top-level translation unit would miss.
pub fn preprocessed_digest(text: &str) -> String {
    let mut buf = String::with_capacity(text.len());
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("# ") || t.starts_with("#pragma") || line.trim().is_empty() {
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    digest(buf.as_bytes())
}

/// A cheap content digest. FNV-1a over the bytes — enough to notice that a file
/// changed, which is all staleness needs.
pub fn digest(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(defines: &[&str], toolchain: &str, src: &str) -> BuildId {
        BuildId::new(
            IsaLevel::Sm90a,
            defines.iter().map(|s| s.to_string()),
            toolchain,
            src,
        )
    }

    /// The eight NVIDIA objects differ only by `-D` flags, so the flags must be
    /// part of the identity or all eight collide.
    #[test]
    fn defines_participate_in_identity() {
        let decode = id(&["PLOW_NV_GEMMA=1"], "cuda-13.0", "abc");
        let prefill = id(
            &["PLOW_NV_GEMMA=1", "PLOW_NV_PREFILL=1"],
            "cuda-13.0",
            "abc",
        );
        assert_ne!(decode, prefill);
        assert_ne!(decode.label(), prefill.label());
    }

    #[test]
    fn define_spelling_is_normalized() {
        let a = id(&["-DPLOW_NV_PREFILL=1"], "cuda-13.0", "abc");
        let b = id(&["PLOW_NV_PREFILL"], "cuda-13.0", "abc");
        assert_eq!(a, b, "-DFOO=1 and FOO are the same switch");
        assert_eq!(a.label(), b.label());
    }

    #[test]
    fn define_order_does_not_change_identity() {
        let a = id(&["B=1", "A=1"], "cuda-13.0", "abc");
        let b = id(&["A=1", "B=1"], "cuda-13.0", "abc");
        assert_eq!(a, b);
    }

    /// A different compiler is a different build: the same source produced 208
    /// registers here and a documented 150 elsewhere.
    #[test]
    fn toolchain_participates_in_identity() {
        assert_ne!(
            id(&["X=1"], "cuda-13.0", "abc"),
            id(&["X=1"], "cuda-12.4", "abc")
        );
    }

    /// Differences must be actionable, not just "stale".
    #[test]
    fn differences_name_what_moved() {
        let inv = id(
            &["PLOW_NV_GEMMA=1", "PLOW_NV_PREFILL=1"],
            "cuda-13.0",
            "abc",
        );
        let build = id(&["PLOW_NV_GEMMA=1"], "cuda-12.4", "def");
        let d = inv.differences(&build);

        assert!(d.iter().any(|s| s.contains("PLOW_NV_PREFILL")), "{d:?}");
        assert!(d.iter().any(|s| s.contains("toolchain")), "{d:?}");
        assert!(d.iter().any(|s| s.contains("source")), "{d:?}");
        assert!(inv.differences(&inv).is_empty());
    }

    #[test]
    fn digest_notices_an_edit() {
        assert_eq!(digest(b"hello"), digest(b"hello"));
        assert_ne!(digest(b"hello"), digest(b"hellp"));
    }
}
