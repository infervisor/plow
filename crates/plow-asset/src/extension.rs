//! The extension load contract — docs/arch/19, phases 2 and 3.
//!
//! A packet is emitted once and frozen. An **extension** adds programs to it without rewriting
//! it: the runtime loads the parent, then merges every extension it was given, and serves the
//! union. An extension may only ADD — it can never edit or remove a parent program.
//!
//! This module is the contract and nothing else: six pure rules over facts a caller extracts
//! from a container, a `plow_config.h`, and a `build.json`. The loader (`plowrt`) and, from
//! phase 4, the emitter (`plowc extend`) both go through it, so the emitter cannot invent its
//! own rules — an extension that would be refused at load is refused at emit, by the same
//! code, with the same message.
//!
//! Every refusal names the extension and the rule, the way object-load refusals already do.
//!
//! # Backward compatibility
//!
//! [`merge`] with no extensions returns the parent's ladder unchanged and grows nothing. A
//! serving directory with no extensions therefore behaves bit-identically to today; nothing in
//! this module is on a code path a shipped packet runs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub const EXTENSION_FILE: &str = "extension.pkt";
pub const REQUIRES_FILE: &str = "requires.json";
pub const CONFIG_FILE: &str = "plow_config.h";

/// A 32-byte content hash. SHA-256 throughout: `decode_objects.rs` already pins objects with
/// it, so an extension pins its parent the same way rather than introducing a second notion of
/// "the same artifact".
pub type Digest = [u8; 32];

pub fn hex(d: &Digest) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// SHA-256 of a whole packet image. This is the "parent packet hash" of rule 1.
pub fn packet_hash(image: &[u8]) -> Digest {
    Sha256::digest(image).into()
}

// --- rule 2: the tensor-table digest -----------------------------------------

/// One tensor as the CONTAINER declares it.
///
/// The design names the tuple `(name, dtype, shape, tp-axis)`. The container carries none of
/// those three as separate fields: `bytes` IS dtype × shape, and a tp-sharded tensor already
/// declares its 1/tp share, so the tp axis is folded into the same number. `initialized` (the
/// tensor has init bytes or a generated-tensor recipe behind it) is the fourth axis a handle's
/// meaning depends on — an index that was a compiler-filled RoPE table and becomes a
/// runtime-filled buffer means something different to the same instruction.
///
/// The TP DEGREE is digested alongside the table (see [`tensor_table_digest`]) because it is
/// the one part of the tp axis the per-tensor byte count cannot show: two emits at tp=4 and
/// tp=8 of models whose hidden sizes differ by 2 declare the same per-rank bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorIdentity {
    pub name: String,
    pub bytes: u64,
    pub initialized: bool,
}

impl TensorIdentity {
    pub fn new(name: impl Into<String>, bytes: u64, initialized: bool) -> TensorIdentity {
        TensorIdentity {
            name: name.into(),
            bytes,
            initialized,
        }
    }
}

/// Digest the parent's tensor table, IN ORDER.
///
/// This is the invariant that makes an extension safe at all. The extension's instructions
/// carry tensor INDICES; an index only means something against the table it was emitted
/// against. Order is part of the digest because the index is a position, not a name.
pub fn tensor_table_digest(tensors: &[TensorIdentity], tp_degree: u32) -> Digest {
    let mut h = Sha256::new();
    h.update(b"plow.ext.tensors.v1\x1e");
    h.update((tensors.len() as u64).to_le_bytes());
    h.update(tp_degree.to_le_bytes());
    for (i, t) in tensors.iter().enumerate() {
        h.update((i as u64).to_le_bytes());
        h.update(t.name.as_bytes());
        h.update([0u8]);
        h.update(t.bytes.to_le_bytes());
        h.update([u8::from(t.initialized)]);
        h.update(b"\x1f");
    }
    h.finalize().into()
}

/// What an extension carries instead of a tensor table. Mirrors
/// [`packet::ext::BlobParentRef`] in owned form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentRef {
    pub parent_hash: Digest,
    pub tensor_count: u32,
    pub tensor_digest: Digest,
}

impl From<&packet::ext::BlobParentRef> for ParentRef {
    fn from(r: &packet::ext::BlobParentRef) -> ParentRef {
        ParentRef {
            parent_hash: r.parent_hash,
            tensor_count: r.n_tensor,
            tensor_digest: r.tensor_digest,
        }
    }
}

// --- rule 3: config compatibility --------------------------------------------

/// Macros in `plow_config.h` that describe SHARED device state, so an extension must agree
/// with its parent on every one of them.
///
/// The list is a CLOSED ALLOWLIST, not a prefix rule, and that is the whole point of rule 3:
/// the axes NOT on it — tile choice (`GM_*`), split count, object family, every
/// `PLOW_HAS_*`/`#ifndef` default an object compiles behind — are local to a program and MAY
/// differ. An extension exists to carry a program compiled differently.
///
/// Each entry is a fact about state the parent already allocated or a shape every program
/// addresses: head geometry (`PLOW_HAS_FLASH_HD*`, `PLOW_PACKET_GQA`), the decode band width
/// (`PLOW_PACKET_DECODE_BATCH`), the KV dtype (`PLOW_PACKET_HAS_FLASH_DECODE_FP8` and the MLA
/// twin), and the model-shape switches that change what a tensor holds.
pub const SHARED_CONFIG_MACROS: &[&str] = &[
    "PLOW_PACKET_GQA",
    "PLOW_PACKET_DECODE_BATCH",
    "PLOW_PACKET_LINEAR_BIAS",
    "PLOW_PACKET_ROPE_HALF_HD64",
    "PLOW_PACKET_ATTENTION_SINKS",
    "PLOW_PACKET_HAS_FLASH_DECODE_FP8",
    "PLOW_PACKET_HAS_FLASH_MLA_PREFILL_FP8",
    "PLOW_HAS_FLASH_HD64",
    "PLOW_HAS_FLASH_HD128",
    "PLOW_HAS_FLASH_HD256",
    "PLOW_HAS_FLASH_HD512",
];

/// Axis names the CONTAINER supplies rather than `plow_config.h`. Namespaced with a `blob.`
/// prefix so they can never collide with a macro name.
pub const BLOB_AXIS_N_CU: &str = "blob.n_cu";
pub const BLOB_AXIS_TP_DEGREE: &str = "blob.tp_degree";
pub const BLOB_AXIS_HIDDEN: &str = "blob.hidden";
pub const BLOB_AXIS_TP_SLOT_BYTES: &str = "blob.tp_slot_bytes";
pub const BLOB_AXIS_TARGET: &str = "blob.target";
pub const BLOB_AXIS_L2_DOMAINS: &str = "blob.l2_domains";
pub const BLOB_AXIS_L2_SMS: &str = "blob.l2_sms";
pub const BLOB_AXIS_KV_RING_ROWS: &str = "blob.kv_ring_rows";
pub const BLOB_AXIS_KV_WINDOW: &str = "blob.kv_window";
pub const BLOB_AXIS_MAX_SEGMENTS: &str = "blob.max_segments";
pub const BLOB_AXIS_MAX_COUNTERS: &str = "blob.max_counters";

/// The shared axes of one artifact, by name. Comparison is by name so a refusal can say WHICH
/// axis disagreed, which is the difference between a message an operator can act on and
/// "config mismatch".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigAxes(pub BTreeMap<String, i64>);

impl ConfigAxes {
    pub fn new() -> ConfigAxes {
        ConfigAxes(BTreeMap::new())
    }

    pub fn set(&mut self, k: impl Into<String>, v: i64) -> &mut ConfigAxes {
        self.0.insert(k.into(), v);
        self
    }

    pub fn get(&self, k: &str) -> Option<i64> {
        self.0.get(k).copied()
    }

    /// Read the shared axes out of a `plow_config.h`.
    ///
    /// The header is machine-written (`devgen::manifest::config_header`), so the grammar is
    /// exactly `#define NAME value` — the same one line `scripts/build_gfx942.sh`'s `cfg_get`
    /// reads, deliberately, so the script and the loader cannot disagree about what a macro
    /// says. Only [`SHARED_CONFIG_MACROS`] are taken; a macro with a non-integer value is
    /// skipped (the string-valued ones are recipe inputs for the build script, not shared
    /// state).
    pub fn from_config_header(src: &str) -> ConfigAxes {
        let want: BTreeSet<&str> = SHARED_CONFIG_MACROS.iter().copied().collect();
        let mut axes = ConfigAxes::new();
        for line in src.lines() {
            let Some(rest) = line.trim_start().strip_prefix("#define ") else {
                continue;
            };
            let mut it = rest.split_whitespace();
            let (Some(name), Some(value)) = (it.next(), it.next()) else {
                continue;
            };
            if !want.contains(name) || axes.0.contains_key(name) {
                continue;
            }
            if let Some(v) = parse_int(value) {
                axes.set(name, v);
            }
        }
        axes
    }

    /// The first axis the two disagree on, as `(axis, parent, extension)`. `None` on either
    /// side means the axis was absent there, which is itself a disagreement: a parent that
    /// states `PLOW_PACKET_GQA` and an extension that does not were emitted by different
    /// compilers and nothing here can say the shared state matches.
    pub fn disagreement(&self, other: &ConfigAxes) -> Option<(String, Option<i64>, Option<i64>)> {
        let mut names: BTreeSet<&String> = self.0.keys().collect();
        names.extend(other.0.keys());
        for n in names {
            let (a, b) = (self.0.get(n).copied(), other.0.get(n).copied());
            if a != b {
                return Some((n.clone(), a, b));
            }
        }
        None
    }
}

fn parse_int(tok: &str) -> Option<i64> {
    let t = tok
        .trim()
        .trim_end_matches("ull")
        .trim_end_matches("ULL")
        .trim_end_matches('u')
        .trim_end_matches('U');
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return i64::from_str_radix(h, 16).ok();
    }
    t.parse::<i64>().ok()
}

// --- rule 4: the KV ring invariant -------------------------------------------

/// Rows a sliding layer's ring needs for a `(window, chunk)` pair.
///
/// MIRROR of `devgen::kv_ring_rows` and of `PLOW_KV_RING` in `dev_isa.h`; `devgen`'s
/// `extension_kv_ring_rows_mirrors_devgen` test pins the two together. It is duplicated
/// rather than shared because `plow-asset` is below `devgen` in the crate graph and the
/// loader must apply the rule with no compiler in the build.
pub fn kv_ring_rows(window: u32, chunk: u32) -> u32 {
    (window + chunk - 1).next_power_of_two()
}

/// The parent's KV ring geometry, as its `build.json` states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvRing {
    /// Every attention layer is full-causal — `devgen::kv_ring` returned `(ctx, MASK_NONE)`
    /// and the ring invariant is vacuous. Full-causal MLA is this case, and it is the one that
    /// matters: the model this mechanism was designed for is unaffected by rule 4.
    FullCausal,
    /// A sliding model: `ring_rows` is what the parent ALLOCATED for window `window`. An
    /// extension cannot enlarge it — the ring is device state the parent sized.
    Windowed { window: u32, ring_rows: u32 },
    /// The parent's `build.json` states no KV geometry.
    ///
    /// This is the pre-phase-4 state of every shipped packet: `shapes` records `max_chunk` and
    /// the bucket list but not the window or the ring, and recovering them from the
    /// instructions means reading a mask operand that lives in `j[1]` on one op family and
    /// `i[7]` on another — the exact slot-blindness doc 16 records as a live defect class.
    ///
    /// Unknown is not "fine": a bucket WIDER than the parent's widest is refused by name,
    /// because a windowed parent that took it would wrap a chunk's rows onto their own history
    /// — a silent wrong answer. Everything else (a decode rung, a sibling, a body, a narrower
    /// bucket) never arms rule 4 and is unaffected.
    Unstated,
}

// --- rule 5: program roles and ladder well-formedness ------------------------

/// What a program IS, rather than where it sits in the table.
///
/// ONE type, defined where the emitter and every reader already live
/// (docs/arch/19, phase 1). `packet::devbuild::derive_roles` is the single derivation: a PARENT
/// falls back to the positional `decode_rung_lo` rule, an EXTENSION reads the role its
/// `DECODE_RUNG_PROG` bit states. The rules below only ever see `&[ProgramRole]` and do not
/// care which container produced them.
pub use packet::devbuild::ProgramRole;

// --- rule 6: budget ----------------------------------------------------------

/// What one program costs the arenas the parent reserved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    /// Bytes of instruction stream the program uploads (insts + stream + wait/succ arrays).
    pub inst_stream_bytes: u64,
    pub counters: u32,
    /// Ordered segments the program declares: `max(StreamEnt.seg) + 1`.
    pub segments: u32,
    /// Scratch-arena bytes the program needs (`BucketStat::arena_bytes`).
    pub workspace_bytes: u64,
}

impl Budget {
    fn max(self, o: Budget) -> Budget {
        Budget {
            inst_stream_bytes: self.inst_stream_bytes.max(o.inst_stream_bytes),
            counters: self.counters.max(o.counters),
            segments: self.segments.max(o.segments),
            workspace_bytes: self.workspace_bytes.max(o.workspace_bytes),
        }
    }
}

/// The arenas the parent reserved, and whether the runtime may grow each.
///
/// `growable` is a single flag rather than one per arena because the three that can grow do so
/// through the same mechanism — the VMM pools the runtime already allocates instruction
/// streams, counter banks and scratch from. `segments` is NOT growable: 2048 ordered segments
/// is a fixed ceiling in `DevProg::seg_classes_with`, so a program over it is refused whatever
/// the pools can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arenas {
    pub reserved: Budget,
    /// Hard ceiling on ordered segments — `2048`, mirroring `DevProg::seg_classes_with`.
    pub segment_ceiling: u32,
    /// The runtime can grow the instruction-stream, counter and workspace arenas through the
    /// VMM pools. `false` on a backend that maps them once at load.
    pub growable: bool,
}

/// How a merged program's demand landed against [`Arenas`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    /// One line per arena the runtime had to grow, for the log. Empty ⇒ everything fit.
    pub grew: Vec<String>,
}

// --- refusals ----------------------------------------------------------------

/// Which of the six rules refused. The discriminants are the numbers the design gives them, so
/// a message, the doc and the code all say "rule 4".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    ParentIdentity = 1,
    TensorTableDigest = 2,
    ConfigCompatibility = 3,
    KvRingInvariant = 4,
    LadderWellFormedness = 5,
    Budget = 6,
}

impl Rule {
    pub fn number(self) -> u32 {
        self as u32
    }

    pub fn name(self) -> &'static str {
        match self {
            Rule::ParentIdentity => "parent identity",
            Rule::TensorTableDigest => "tensor-table digest",
            Rule::ConfigCompatibility => "config compatibility",
            Rule::KvRingInvariant => "KV ring invariant",
            Rule::LadderWellFormedness => "ladder well-formedness",
            Rule::Budget => "budget",
        }
    }
}

/// A refusal names the EXTENSION and the RULE. Half-applying a merge is the one outcome that
/// must be impossible: a windowed model that took four of five programs would serve wrong
/// tokens rather than fail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub extension: String,
    pub rule: Rule,
    pub detail: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "extension `{}` refused by rule {} ({}): {}",
            self.extension,
            self.rule.number(),
            self.rule.name(),
            self.detail
        )
    }
}

impl Refusal {
    fn new(extension: &str, rule: Rule, detail: impl Into<String>) -> Refusal {
        Refusal {
            extension: extension.to_string(),
            rule,
            detail: detail.into(),
        }
    }
}

// --- phase 3: the objects an extension requires ------------------------------

/// `requires.json` — the object stems an extension's programs need, hash-pinned the way
/// `decode_objects.json` already pins them (doc 16).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requires {
    pub version: u32,
    /// The object arch the stems were built for, e.g. `"gfx942"`. Matched against the
    /// extension's `PLOW_PACKET_OBJECT_ARCH`.
    pub arch: String,
    /// The EXTENSION's own `PLOW_PACKET_HASH`, as `0x%016x`.
    ///
    /// Not the parent's. An object built for one bucket carries
    /// `plow_packet_hash_{lo,hi}` naming the artifact it was compiled against, and if an
    /// extension advertised the parent's hash then an object built for the parent would load
    /// beside a different bucket and trap — or worse, dispatch an arm that does not exist,
    /// which on AMD writes nothing.
    pub pairing_hash: String,
    pub objects: Vec<RequiredObject>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredObject {
    /// Object stem as `scripts/build_gfx942.sh` names its rows, e.g. `"interp_prefill"`.
    pub stem: String,
    pub sha256: String,
    /// The program width this object serves, when the stem is row-specific. Feeds
    /// `PLOW_ROWS_ONLY` in the build script's extension mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u32>,
}

pub fn parse_pairing_hash(s: &str) -> Option<u64> {
    let t = s.trim().trim_end_matches("ull").trim_end_matches("ULL");
    u64::from_str_radix(t.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()
}

impl Requires {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            return Err(format!("requires.json version {} — want 1", self.version));
        }
        if self.arch.is_empty() {
            return Err("requires.json names no object arch".into());
        }
        if parse_pairing_hash(&self.pairing_hash).is_none() {
            return Err(format!(
                "requires.json pairing_hash `{}` is not a hex packet hash",
                self.pairing_hash
            ));
        }
        if self.objects.is_empty() {
            return Err("requires.json lists no objects".into());
        }
        let mut seen = BTreeSet::new();
        for o in &self.objects {
            // A stem is a FILE NAME the loader opens under PLOW_HSACO. Anything with a path
            // separator would escape that directory.
            if o.stem.is_empty()
                || std::path::Path::new(&o.stem).components().count() != 1
                || o.stem.contains('/')
            {
                return Err(format!("requires.json object stem `{}` is not a bare name", o.stem));
            }
            if o.sha256.len() != 64 || !o.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("requires.json object `{}` has no sha256", o.stem));
            }
            if !seen.insert(o.stem.as_str()) {
                return Err(format!("requires.json lists `{}` twice", o.stem));
            }
        }
        Ok(())
    }

    /// The rows the extension's objects are built for — `PLOW_ROWS_ONLY` for
    /// `scripts/build_gfx942.sh --extension`.
    pub fn rows(&self) -> Vec<u32> {
        let mut r: Vec<u32> = self.objects.iter().filter_map(|o| o.rows).collect();
        r.sort_unstable();
        r.dedup();
        r
    }
}

/// Rule 3's object half: the packet-pairing stamp a code object carries must name the
/// EXTENSION, not the parent.
///
/// The parent's hash is passed in so the common mistake — rebuilding the objects against
/// `assets/plow_config.h` instead of `assets.ext/<name>/plow_config.h` — is named as itself
/// rather than reported as an opaque mismatch.
pub fn check_object_stamp(
    extension: &str,
    object: &str,
    stamped: Option<u64>,
    want: u64,
    parent_pairing_hash: Option<u64>,
) -> Result<(), Refusal> {
    let Some(stamped) = stamped else {
        return Err(Refusal::new(
            extension,
            Rule::ConfigCompatibility,
            format!(
                "object `{object}` carries no packet-pairing stamp \
                 (plow_packet_hash_{{lo,hi}}); an extension's objects must be stamped — build \
                 them with PLOW_HSACO_CONFIG=<the extension dir> (scripts/build_gfx942.sh)"
            ),
        ));
    };
    if stamped == want {
        return Ok(());
    }
    if Some(stamped) == parent_pairing_hash {
        return Err(Refusal::new(
            extension,
            Rule::ConfigCompatibility,
            format!(
                "object `{object}` stamps the PARENT packet 0x{stamped:016x}, not this \
                 extension's 0x{want:016x} — it was built against the parent's plow_config.h \
                 and has none of this extension's arms; rebuild it with \
                 PLOW_HSACO_CONFIG pointing at the extension"
            ),
        ));
    }
    Err(Refusal::new(
        extension,
        Rule::ConfigCompatibility,
        format!(
            "object `{object}` stamps packet 0x{stamped:016x}, this extension requires \
             0x{want:016x}"
        ),
    ))
}

// --- the facts each side supplies --------------------------------------------

/// Everything the contract needs to know about the loaded `model.pkt`.
#[derive(Clone, Debug)]
pub struct ParentFacts {
    pub hash: Digest,
    pub tensor_count: u32,
    pub tensor_digest: Digest,
    pub config: ConfigAxes,
    pub kv: KvRing,
    /// The parent's programs, in table order.
    pub programs: Vec<ProgramRole>,
    /// The widest decode rung the runtime will serve — `PLOW_DECODE_BATCH`. A merged rung
    /// above it has no band to sit in.
    pub decode_batch: u32,
    pub arenas: Arenas,
    /// The parent's `PLOW_PACKET_HASH`, for the phase-3 object-stamp message.
    pub pairing_hash: Option<u64>,
}

/// Everything the contract needs to know about one extension.
#[derive(Clone, Debug)]
pub struct ExtensionFacts {
    /// How the refusal names it — the extension directory, e.g. `assets.ext/bucket-4096`.
    pub name: String,
    pub parent: ParentRef,
    pub config: ConfigAxes,
    pub programs: Vec<ProgramRole>,
    /// The prefill CHUNK a new bucket runs at. Rule 4 sizes the ring from it.
    pub chunk: u32,
    pub budget: Budget,
}

/// Where a merged program came from. Phase 5 reports it; the ladder rules use it to name the
/// extension that introduced a duplicate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    Parent,
    Extension(String),
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Origin::Parent => f.write_str("model.pkt"),
            Origin::Extension(n) => f.write_str(n),
        }
    }
}

/// The union the runtime serves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Merged {
    /// Every program, parent first then extensions in the order they were given.
    pub programs: Vec<(Origin, ProgramRole)>,
    pub growth: Growth,
}

impl Merged {
    pub fn prefill_widths(&self) -> Vec<u32> {
        let mut w: Vec<u32> = self
            .programs
            .iter()
            .filter_map(|(_, r)| match r {
                ProgramRole::PrefillBucket { rows } => Some(*rows),
                _ => None,
            })
            .collect();
        w.sort_unstable();
        w
    }

    pub fn decode_rungs(&self) -> Vec<u32> {
        let mut w: Vec<u32> = self
            .programs
            .iter()
            .filter_map(|(_, r)| match r {
                ProgramRole::DecodeRung { rows } => Some(*rows),
                _ => None,
            })
            .collect();
        w.sort_unstable();
        w
    }
}

// --- the merge ---------------------------------------------------------------

/// Load the parent, then merge `exts`, refusing unless all six rules hold.
///
/// With no extensions this is the identity on the parent's ladder and grows nothing — the
/// backward-compatible case, which must behave exactly as it does today.
///
/// The rules run IN ORDER, per extension, and the first failure refuses the whole merge. Order
/// matters for the message: an extension for a different packet must be told that, not told
/// its ladder is malformed against a parent it was never meant for.
pub fn merge(parent: &ParentFacts, exts: &[ExtensionFacts]) -> Result<Merged, Refusal> {
    let mut programs: Vec<(Origin, ProgramRole)> = parent
        .programs
        .iter()
        .map(|r| (Origin::Parent, *r))
        .collect();

    // The parent's own ladder must be well formed before anything is merged onto it. Without
    // extensions this is the only check that runs, and it is the one that would have caught a
    // packet whose ladder was already wrong.
    check_ladder(&programs, parent.decode_batch).map_err(|d| Refusal {
        extension: "model.pkt".into(),
        rule: Rule::LadderWellFormedness,
        detail: d,
    })?;

    let parent_widest_prefill = parent
        .programs
        .iter()
        .filter_map(|r| match r {
            ProgramRole::PrefillBucket { rows } => Some(*rows),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let mut demand = Budget::default();
    for e in exts {
        // 1 — parent identity.
        if e.parent.parent_hash != parent.hash {
            return Err(Refusal::new(
                &e.name,
                Rule::ParentIdentity,
                format!(
                    "declares parent {} but the loaded model.pkt is {}",
                    hex(&e.parent.parent_hash),
                    hex(&parent.hash)
                ),
            ));
        }

        // 2 — tensor-table digest. Count first: it is the readable half of the same fact.
        if e.parent.tensor_count != parent.tensor_count {
            return Err(Refusal::new(
                &e.name,
                Rule::TensorTableDigest,
                format!(
                    "was emitted against a {}-tensor table, the parent declares {}",
                    e.parent.tensor_count, parent.tensor_count
                ),
            ));
        }
        if e.parent.tensor_digest != parent.tensor_digest {
            return Err(Refusal::new(
                &e.name,
                Rule::TensorTableDigest,
                format!(
                    "tensor-table digest {} does not match the parent's {} — every tensor \
                     index in this extension's instructions would mean something else",
                    hex(&e.parent.tensor_digest),
                    hex(&parent.tensor_digest)
                ),
            ));
        }

        // 3 — config compatibility, on the shared axes only.
        if let Some((axis, want, got)) = parent.config.disagreement(&e.config) {
            let show = |v: Option<i64>| match v {
                Some(v) => v.to_string(),
                None => "absent".to_string(),
            };
            return Err(Refusal::new(
                &e.name,
                Rule::ConfigCompatibility,
                format!(
                    "{axis} is {} here and {} in the parent — that axis shapes shared device \
                     state, so it cannot differ (tile choice, split count and object family \
                     may)",
                    show(got),
                    show(want)
                ),
            ));
        }

        // 4 — the KV ring invariant, for a bucket wider than the parent's widest.
        let widest_new = e
            .programs
            .iter()
            .filter_map(|r| match r {
                ProgramRole::PrefillBucket { rows } => Some(*rows),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if widest_new > parent_widest_prefill {
            match parent.kv {
                KvRing::FullCausal => {}
                KvRing::Windowed { window, ring_rows } => {
                    let need = kv_ring_rows(window, e.chunk);
                    if ring_rows < need {
                        return Err(Refusal::new(
                            &e.name,
                            Rule::KvRingInvariant,
                            format!(
                                "adds a {widest_new}-row prefill bucket (the parent's widest is \
                                 {parent_widest_prefill}) at chunk {}, which needs a KV ring of \
                                 {need} rows; the parent sized its sliding ring at {ring_rows} \
                                 for window {window}. The ring is device state the parent \
                                 allocated and an extension cannot enlarge it — a chunk's rows \
                                 would wrap onto their own history",
                                e.chunk
                            ),
                        ));
                    }
                }
                KvRing::Unstated => {
                    return Err(Refusal::new(
                        &e.name,
                        Rule::KvRingInvariant,
                        format!(
                            "adds a {widest_new}-row prefill bucket, wider than the parent's \
                             widest ({parent_widest_prefill}), but the parent's build.json \
                             states no KV window or ring size, so `ring >= window + chunk - 1` \
                             cannot be checked. A windowed parent would wrap a chunk's rows \
                             onto their own history — a silent wrong answer. Re-emit the parent \
                             with a plowc that records shapes.kv_window / shapes.kv_ring_rows, \
                             or add a bucket no wider than {parent_widest_prefill}"
                        ),
                    ));
                }
            }
        }

        programs.extend(e.programs.iter().map(|r| (Origin::Extension(e.name.clone()), *r)));

        // 5 — ladder well-formedness AFTER this extension is merged, so the refusal names the
        // extension that broke it rather than the last one in the directory.
        check_ladder(&programs, parent.decode_batch)
            .map_err(|d| Refusal::new(&e.name, Rule::LadderWellFormedness, d))?;

        demand = demand.max(e.budget);
    }

    // 6 — budget, over the union. Per-arena, so the log says which one grew.
    let growth = check_budget(exts, demand, &parent.arenas)?;

    Ok(Merged { programs, growth })
}

fn check_budget(
    exts: &[ExtensionFacts],
    demand: Budget,
    arenas: &Arenas,
) -> Result<Growth, Refusal> {
    // Four `(arena, demanded, reserved, who demanded it)` rows. `who` is the extension whose
    // budget set the maximum, so the refusal names the one to rebuild; with no extensions
    // there is no demand and no row can fire.
    let biggest = |key: fn(&Budget) -> u64| -> String {
        exts.iter()
            .max_by_key(|e| key(&e.budget))
            .map(|e| e.name.clone())
            .unwrap_or_else(|| "model.pkt".into())
    };
    let rows: [(&str, u64, u64, fn(&Budget) -> u64); 4] = [
        (
            "instruction-stream bytes",
            demand.inst_stream_bytes,
            arenas.reserved.inst_stream_bytes,
            |b| b.inst_stream_bytes,
        ),
        (
            "counters",
            demand.counters as u64,
            arenas.reserved.counters as u64,
            |b| b.counters as u64,
        ),
        (
            "segments",
            demand.segments as u64,
            arenas.reserved.segments as u64,
            |b| b.segments as u64,
        ),
        (
            "workspace bytes",
            demand.workspace_bytes,
            arenas.reserved.workspace_bytes,
            |b| b.workspace_bytes,
        ),
    ];

    if demand.segments > arenas.segment_ceiling {
        return Err(Refusal::new(
            &biggest(|b| b.segments as u64),
            Rule::Budget,
            format!(
                "declares {} ordered segments, over the fixed ceiling of {} — the segment \
                 class table is not growable",
                demand.segments, arenas.segment_ceiling
            ),
        ));
    }

    let mut grew = Vec::new();
    for (what, need, have, key) in rows {
        if need <= have {
            continue;
        }
        if !arenas.growable {
            return Err(Refusal::new(
                &biggest(key),
                Rule::Budget,
                format!(
                    "needs {need} {what}, the parent reserved {have}, and this backend maps \
                     its arenas once at load — it cannot grow them"
                ),
            ));
        }
        grew.push(format!(
            "grew {what} from {have} to {need} through the VMM pool for the merged extensions"
        ));
    }
    Ok(Growth { grew })
}

/// Rule 5, as a function of the merged table alone.
///
/// * prefill widths strictly ascending, no duplicates
/// * decode rungs strictly ascending, no duplicates, and `<= decode_batch`
/// * every `PackedSibling` / `TokenBatchBody` names a bucket present in the union
///
/// ASCENDING is by construction once the ladders derive by filtering and sorting (design phase
/// 1) — a merged table arrives in whatever order the extensions were found, so "ascending" as
/// a property of the TABLE is exactly what the merge gives up. What is left of the rule, and
/// what actually bites, is STRICTLY: no two programs of the same role at the same width, since
/// the sorted ladder could then not say which one it means.
fn check_ladder(programs: &[(Origin, ProgramRole)], decode_batch: u32) -> Result<(), String> {
    let mut prefill: Vec<(u32, &Origin)> = Vec::new();
    let mut decode: Vec<(u32, &Origin)> = Vec::new();
    for (o, r) in programs {
        match r {
            ProgramRole::PrefillBucket { rows } => prefill.push((*rows, o)),
            ProgramRole::DecodeRung { rows } => decode.push((*rows, o)),
            _ => {}
        }
    }
    let buckets: BTreeSet<u32> = prefill.iter().map(|(w, _)| *w).collect();

    for (what, ladder) in [("prefill bucket", &prefill), ("decode rung", &decode)] {
        let mut seen: BTreeMap<u32, &Origin> = BTreeMap::new();
        for (w, o) in ladder.iter() {
            if let Some(prev) = seen.insert(*w, o) {
                return Err(format!(
                    "duplicate {what} at {w} rows — {prev} already has one, and an extension \
                     may only ADD (replacing a program in place is a re-emit)"
                ));
            }
        }
        if what == "prefill bucket" && ladder.is_empty() {
            return Err("the union has no prefill bucket".into());
        }
    }
    if decode.is_empty() {
        return Err("the union has no decode rung".into());
    }
    for (w, o) in &decode {
        if *w == 0 || *w > decode_batch {
            return Err(format!(
                "decode rung {w} from {o} is outside the decode band 1..={decode_batch}"
            ));
        }
    }

    for (o, r) in programs {
        let (kind, of) = match r {
            ProgramRole::PackedSibling { of_rows } => ("packed sibling", *of_rows),
            ProgramRole::TokenBatchBody { rows, .. } => ("token-batch body", *rows),
            _ => continue,
        };
        if !buckets.contains(&of) {
            return Err(format!(
                "{kind} from {o} names a {of}-row prefill bucket, which the union does not \
                 have (buckets: {buckets:?})"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "extension_tests.rs"]
mod tests;
