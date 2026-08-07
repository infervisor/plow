//! Deriving an inventory from a built interpreter object.
//!
//! # Why this cannot be done by reading source
//!
//! Three facts about this tree make a source-level answer wrong:
//!
//! * `interp_sm90a.cu` is a 42-line wrapper that `#include`s `interp_sm120.cu`
//!   with `PLOW_NV_HOPPER=1`. Grepping it for opcodes finds **zero**.
//! * CMake builds **eight** interpreter objects from that one translation unit,
//!   differing only in `-D` flags (`runtime/CMakeLists.txt:127-320`). Their
//!   dispatch arms are not the same set.
//! * The arms are guarded by `#if PLOW_NV_PREFILL`, `#if PLOW_NV_W8A8`,
//!   `#if PLOW_BUCKET_PREFILL` and friends. Which ones survive is a question
//!   only the preprocessor can answer.
//!
//! So the probe runs the real preprocessor with the object's exact flags and
//! reads what is left. The `#if` arms are resolved, which is the point; the
//! opcode names survive, because `dev_isa.h` declares them as an **enum** and
//! the preprocessor does not touch enumerators. That is convenient: the arms
//! can be matched by name, so an unrelated `case 8:` in a vendor header cannot
//! be mistaken for a kernel.
//!
//! # What is probed
//!
//! * **dispatch arms** — the `case` labels reachable inside the interpreter's
//!   dispatch function, mapped back to [`DevOp`] by value;
//! * **tile shape** — the object's tile macros, expanded by the same
//!   preprocessor run, because on NVIDIA the tile is a compile-time constant of
//!   the object rather than a property of the opcode;
//! * **resource envelope** — registers, spill, and shared memory, from the
//!   compiler's own report.
//!
//! Everything the probe returns carries the [`BuildId`] it was derived from.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use packet::dev::DevOp;

use crate::build::{BuildId, Provenance};

#[derive(Debug)]
pub enum ProbeError {
    /// The compiler could not be run at all.
    CompilerMissing {
        program: String,
    },
    /// The preprocessor ran and failed.
    Preprocess {
        status: Option<i32>,
        stderr: String,
    },
    /// The dispatch function was not found in the preprocessed output.
    NoDispatchFn {
        name: String,
    },
    Io(std::io::Error),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbeError::CompilerMissing { program } => write!(
                f,
                "cannot run {program}: an inventory is derived from a built object, so probing \
                 needs the toolchain that builds it"
            ),
            ProbeError::Preprocess { status, stderr } => {
                write!(f, "preprocessing failed (status {status:?}):\n{stderr}")
            }
            ProbeError::NoDispatchFn { name } => write!(
                f,
                "no dispatch function {name:?} in the preprocessed output — the interpreter's \
                 entry point was renamed, or the wrong translation unit was probed"
            ),
            ProbeError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<std::io::Error> for ProbeError {
    fn from(e: std::io::Error) -> Self {
        ProbeError::Io(e)
    }
}

/// How to invoke the preprocessor for one object.
#[derive(Clone, Debug)]
pub struct ProbeTarget {
    /// Compiler driver, e.g. `/usr/local/cuda/bin/nvcc` or `hipcc`.
    pub compiler: String,
    /// Arch flag, e.g. `-arch=sm_90a` or `--offload-arch=gfx950`.
    pub arch_flag: String,
    /// Include directories.
    pub includes: Vec<String>,
    /// `-D` definitions, exactly as the build passes them.
    pub defines: Vec<String>,
    /// Translation unit to preprocess.
    pub source: String,
    /// Name of the dispatch function whose `case` labels are the arms.
    pub dispatch_fn: String,
}

impl ProbeTarget {
    /// Run the preprocessor and return its output.
    pub fn preprocess(&self) -> Result<String, ProbeError> {
        let mut cmd = Command::new(&self.compiler);
        // A clean environment: under `nix develop`, CPATH points nvcc's host
        // pass at headers that conflict with the CUDA ones. The build scripts
        // use `env -i` for the same reason.
        cmd.env_clear()
            .env("PATH", "/usr/local/cuda/bin:/opt/rocm/bin:/usr/bin:/bin")
            .arg("-E")
            .arg(&self.arch_flag);
        for i in &self.includes {
            cmd.arg("-I").arg(i);
        }
        for d in &self.defines {
            cmd.arg(if d.starts_with("-D") {
                d.clone()
            } else {
                format!("-D{d}")
            });
        }
        cmd.arg(&self.source);

        let out = match cmd.output() {
            Ok(o) => o,
            Err(_) => {
                return Err(ProbeError::CompilerMissing {
                    program: self.compiler.clone(),
                })
            }
        };
        if !out.status.success() {
            return Err(ProbeError::Preprocess {
                status: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(4000)
                    .collect(),
            });
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// The identity of the object this target describes, over `content`.
    ///
    /// `content` is expected to be the **preprocessed** translation unit -- see
    /// [`preprocessed_digest`]. Hashing the raw top-level source would miss
    /// changes in `#include`d kernel headers (`op_gemm.cuh` and friends), so a
    /// tile edit there would keep the same build identity and silently reuse
    /// stale measurements.
    pub fn build_id_from(&self, isa: hwspec::IsaLevel, toolchain: &str, content: &str) -> BuildId {
        BuildId::new(
            isa,
            self.defines.clone(),
            toolchain,
            crate::build::preprocessed_digest(content),
        )
    }
}

/// The opcodes with a live dispatch arm in the preprocessed text.
///
/// Scoped to the dispatch function's body rather than the whole translation
/// unit: a preprocessed CUDA TU is megabytes of headers containing plenty of
/// unrelated `case 8:` labels, and counting those would report arms that do not
/// exist.
pub fn dispatched_opcodes(
    preprocessed: &str,
    dispatch_fn: &str,
) -> Result<BTreeSet<u16>, ProbeError> {
    let body =
        dispatch_body(preprocessed, dispatch_fn).ok_or_else(|| ProbeError::NoDispatchFn {
            name: dispatch_fn.to_string(),
        })?;

    // Name -> value through the ABI itself, so a label the Rust side does not
    // know about is ignored rather than invented.
    let by_name: BTreeMap<&str, u16> = DevOp::ALL
        .iter()
        .map(|op| (op.c_name(), *op as u16))
        .collect();

    let mut found = BTreeSet::new();
    for name in case_opcode_names(body) {
        if let Some(v) = by_name.get(name.as_str()) {
            found.insert(*v);
        }
    }
    Ok(found)
}

/// Extract the text of `fn_name`'s body by brace matching from its opening `{`.
///
/// Preprocessed output has no comments, which removes the usual reason this is
/// fragile. String and char literals are still skipped, since a brace inside one
/// would otherwise unbalance the scan.
fn dispatch_body<'a>(text: &'a str, fn_name: &str) -> Option<&'a str> {
    let bytes = text.as_bytes();
    let mut from = 0usize;
    // The name appears in declarations and calls too; take the occurrence that
    // is followed by a parameter list and then a body.
    while let Some(rel) = text[from..].find(fn_name) {
        let at = from + rel;
        from = at + fn_name.len();
        let Some(paren) = text[at..].find('(') else {
            continue;
        };
        let Some(open_rel) = text[at + paren..].find('{') else {
            continue;
        };
        let open = at + paren + open_rel;
        // A definition has only the parameter list between the name and the
        // brace; a call followed by an unrelated block can have anything.
        if text[at + fn_name.len()..open].contains(';') {
            continue;
        }
        if let Some(end) = match_brace(bytes, open) {
            return Some(&text[open..=end]);
        }
    }
    None
}

/// Index of the `}` closing the `{` at `open`, skipping literals.
fn match_brace(b: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            q @ (b'"' | b'\'') => {
                i += 1;
                while i < b.len() && b[i] != q {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// One dispatch arm: the opcodes that reach it, and the function they reach.
///
/// Opcodes appear together when the source falls through — `case A: case B:`
/// with no statement between them — which is how the interpreter expresses
/// "these names are one kernel". Deriving that here means aliasing is detected
/// for every family by the same parser, rather than asserted per family by
/// whoever wrote the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchArm {
    /// Opcode values sharing this body, in source order.
    pub opcodes: Vec<u16>,
    /// The first function the arm calls, without template arguments. Identifies
    /// the body, and so the alias group.
    ///
    /// Empty means the arm has a `case` label and a `break` but calls nothing.
    /// Two very different things look like this and both must be excluded from
    /// selection: `PLOW_DOP_NOP`, which legitimately does nothing, and a
    /// reserved stub such as AMD's `XFLASH_MERGE`, whose body is a comment
    /// saying the implementation lands later. Dispatch-present is not
    /// kernel-present.
    pub callee: String,
    /// Template instantiations found in the arm, in source order.
    ///
    /// These are not decoration: the interpreter expresses shape specialization
    /// as template arguments, so `d_flash_prefill_mux<256, 64, 32>` *is* the
    /// attention tile (head_dim, bq, bkv) and `d_headnorm_rope<64, true>` is the
    /// head-dim specialization. Reading them off the object gives the tuner a
    /// shape predicate it would otherwise have to be told.
    pub specializations: Vec<Vec<String>>,
}

/// C keywords that can be followed by `(` but are not calls.
const NOT_CALLS: &[&str] = &[
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "do",
    "else",
    "case",
    "break",
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
];

/// Group the dispatch switch into arms.
///
/// Fallthrough is the signal. `case GEMM: case GEMM_MED: case GEMM_SMALL:
/// d_gemm(...)` yields one arm with three opcodes, which is exactly the NVIDIA
/// aliasing this crate exists to surface — and now it is read out of the object
/// rather than known in advance.
pub fn dispatch_arms(
    preprocessed: &str,
    dispatch_fn: &str,
) -> Result<Vec<DispatchArm>, ProbeError> {
    let body =
        dispatch_body(preprocessed, dispatch_fn).ok_or_else(|| ProbeError::NoDispatchFn {
            name: dispatch_fn.to_string(),
        })?;
    let by_name: BTreeMap<&str, u16> = DevOp::ALL
        .iter()
        .map(|op| (op.c_name(), *op as u16))
        .collect();

    let mut arms: Vec<DispatchArm> = Vec::new();
    let mut pending: Vec<u16> = Vec::new();
    let mut rest = body;

    while let Some(at) = rest.find("case") {
        let after = &rest[at + 4..];
        if !after.starts_with(|c: char| c.is_whitespace()) {
            rest = after;
            continue;
        }
        let t = after.trim_start();
        let name: String = t
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let tail = t[name.len()..].trim_start();
        if !tail.starts_with(':') {
            rest = after;
            continue;
        }
        if let Some(&v) = by_name.get(name.as_str()) {
            pending.push(v);
        }

        // The arm's text ends at whichever comes first: the next label, or the
        // `break` that terminates it. Stopping only at the next label would let
        // an empty arm swallow the switch's `default: __trap()` and report a
        // healthy no-op opcode as trapping -- which is what PLOW_DOP_NOP did.
        let seg_start = &tail[1..];
        let next_label = ["case", "default"]
            .iter()
            .filter_map(|k| seg_start.find(k))
            .min()
            .unwrap_or(seg_start.len());
        let brk = find_break(seg_start).unwrap_or(seg_start.len());
        let seg_end = next_label.min(brk);
        let segment = &seg_start[..seg_end];

        match calls_in(segment) {
            // A body: this arm terminates, taking any labels that fell through.
            Some((callee, specializations)) => arms.push(DispatchArm {
                opcodes: std::mem::take(&mut pending),
                callee,
                specializations,
            }),
            // No call, but a `break` before the next label: a real arm that does
            // nothing (PLOW_DOP_NOP). It terminates too.
            None if brk <= next_label => arms.push(DispatchArm {
                opcodes: std::mem::take(&mut pending),
                callee: String::new(),
                specializations: Vec::new(),
            }),
            // Neither: pure fallthrough, so the next label shares this body.
            None => {}
        }
        rest = &seg_start[seg_end..];
    }
    Ok(arms)
}

/// Offset of a `break` keyword, ignoring identifiers that merely contain it.
fn find_break(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = s[from..].find("break") {
        let at = from + rel;
        let before_ok = at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_');
        let after = at + 5;
        let after_ok = after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_');
        if before_ok && after_ok {
            return Some(at);
        }
        from = at + 5;
    }
    None
}

/// The first genuine call in a fragment, plus every template instantiation in
/// it.
///
/// Template calls must be recognized or the scan walks straight past the real
/// body: the attention arm is `d_flash_prefill_mux<256, 64, 32>(...)`, and a
/// scanner looking only for `ident(` skips it and reports the `else __trap()`
/// as the callee — which reads as "this opcode traps" for a perfectly healthy
/// kernel.
fn calls_in(seg: &str) -> Option<(String, Vec<Vec<String>>)> {
    let b = seg.as_bytes();
    let mut first: Option<String> = None;
    let mut specs: Vec<Vec<String>> = Vec::new();
    let mut i = 0usize;

    while i < b.len() {
        if !(b[i].is_ascii_alphabetic() || b[i] == b'_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        let ident = &seg[start..i];
        let mut j = i;
        while j < b.len() && (b[j] as char).is_whitespace() {
            j += 1;
        }

        // `ident<A, B>(` — a template instantiation.
        let mut targs: Option<Vec<String>> = None;
        if j < b.len() && b[j] == b'<' {
            let mut depth = 0i32;
            let open = j;
            while j < b.len() {
                match b[j] {
                    b'<' => depth += 1,
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    // A `;` or `{` before the brackets balance means this was a
                    // comparison, not a template.
                    b';' | b'{' => break,
                    _ => {}
                }
                j += 1;
            }
            if depth == 0 && j < b.len() {
                let inner = &seg[open + 1..j];
                j += 1;
                while j < b.len() && (b[j] as char).is_whitespace() {
                    j += 1;
                }
                if j < b.len() && b[j] == b'(' {
                    targs = Some(
                        inner
                            .split(',')
                            .map(|a| a.trim().to_string())
                            .filter(|a| !a.is_empty())
                            .collect(),
                    );
                }
            }
        }

        let is_call = targs.is_some() || (j < b.len() && b[j] == b'(');
        if is_call && !NOT_CALLS.contains(&ident) {
            if first.is_none() {
                first = Some(ident.to_string());
            }
            if let Some(t) = targs {
                if !specs.contains(&t) {
                    specs.push(t);
                }
            }
        }
    }
    first.map(|f| (f, specs))
}

/// Every `case PLOW_DOP_<NAME>:` in a block.
///
/// Matched by name rather than value: the opcodes are enumerators, so they
/// survive preprocessing intact, and a name match cannot collide with the
/// numeric `case` labels that appear throughout the CUDA headers.
fn case_opcode_names(body: &str) -> Vec<String> {
    const PREFIX: &str = "PLOW_DOP_";
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find("case") {
        let after = &rest[at + 4..];
        rest = after;
        if !after.starts_with(|c: char| c.is_whitespace()) {
            continue; // part of a longer identifier
        }
        let t = after.trim_start();
        if !t.starts_with(PREFIX) {
            continue;
        }
        let name: String = t
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        // A label, not a mention: the identifier must be followed by `:`.
        if !t[name.len()..].trim_start().starts_with(':') {
            continue;
        }
        out.push(name);
    }
    out
}

/// Expand integer macros through the same preprocessor run.
///
/// The trick is a marker line the preprocessor rewrites for us: emitting
/// `PLOWPROBE PGM_BM PGM_BN` yields `PLOWPROBE 128 128`. Only the preprocessor
/// is needed, so this works without a GPU and without linking.
pub fn probe_macros(
    target: &ProbeTarget,
    header: &str,
    names: &[&str],
) -> Result<Vec<Option<i64>>, ProbeError> {
    let dir = std::env::temp_dir().join(format!("plow_macro_probe_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let src = dir.join("macro_probe.cu");
    let marker = "PLOWPROBE_MARKER";
    // The extra `__HIP_DEVICE_COMPILE__` token identifies WHICH pass a marker line came from.
    // `hipcc -E` with an offload arch preprocesses the TU twice — device pass (arch macros
    // defined) and host pass (not) — and both emit the marker. Once a header's defaults
    // diverge on the arch (op_gemm.h keys its CDNA3 tile/stage defaults on PLOW_CDNA4, i.e.
    // __gfx950__), the passes DISAGREE, and taking the last line silently reported the HOST
    // pass's CDNA3 tiles for a gfx950 probe — every gfx950 inventory rung shrank to 192x256.
    let body = format!(
        "#include \"{header}\"\n{marker} __HIP_DEVICE_COMPILE__ {}\n",
        names.join(" ")
    );
    std::fs::write(&src, body)?;

    let mut t = target.clone();
    t.source = src.to_string_lossy().into_owned();
    let text = t.preprocess()?;
    let _ = std::fs::remove_dir_all(&dir);

    let lines: Vec<&str> = text
        .lines()
        .filter(|l| l.trim_start().starts_with(marker))
        .collect();
    // Prefer the device pass (token expands to 1). Single-pass preprocesses (nvcc host-only)
    // leave the token unexpanded or 0 on every line — fall back to the last, as before.
    let line = lines
        .iter()
        .find(|l| l.split_whitespace().nth(1) == Some("1"))
        .or(lines.last())
        .copied()
        .unwrap_or("");
    let toks: Vec<&str> = line.trim().split_whitespace().skip(2).collect();
    Ok(names
        .iter()
        .enumerate()
        .map(|(i, _)| toks.get(i).and_then(|t| t.trim().parse::<i64>().ok()))
        .collect())
}

/// A probed object: which opcodes it dispatches, and what it was built from.
#[derive(Clone, Debug)]
pub struct ProbedObject {
    pub provenance: Provenance,
    pub opcodes: BTreeSet<u16>,
    /// The arms, with their callee and shape specializations.
    pub arms: Vec<DispatchArm>,
}

impl ProbedObject {
    pub fn build(&self) -> &BuildId {
        self.provenance.build()
    }

    /// Whether this opcode reaches a real kernel body.
    ///
    /// Distinct from [`Self::dispatches`]: an opcode can have a live `case`
    /// label and still execute nothing. Selection must use this, or a tuner
    /// will happily measure a stub at ~0 ns and promote it.
    pub fn executes(&self, op: DevOp) -> bool {
        self.arms
            .iter()
            .any(|a| a.opcodes.contains(&(op as u16)) && !a.callee.is_empty())
    }

    /// Opcodes with a dispatch arm whose body is empty.
    pub fn stubs(&self) -> Vec<u16> {
        self.arms
            .iter()
            .filter(|a| a.callee.is_empty())
            .flat_map(|a| a.opcodes.iter().copied())
            .collect()
    }

    pub fn dispatches(&self, op: DevOp) -> bool {
        self.opcodes.contains(&(op as u16))
    }
}

/// Probe one object end to end.
pub fn probe(
    target: &ProbeTarget,
    isa: hwspec::IsaLevel,
    toolchain: &str,
) -> Result<ProbedObject, ProbeError> {
    if !Path::new(&target.source).exists() {
        return Err(ProbeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} does not exist", target.source),
        )));
    }
    let text = target.preprocess()?;
    // Identity is over the preprocessed output -- every header and macro
    // expansion the compiler sees, not just the top-level file.
    let build = target.build_id_from(isa, toolchain, &text);
    let opcodes = dispatched_opcodes(&text, &target.dispatch_fn)?;
    let arms = dispatch_arms(&text, &target.dispatch_fn)?;
    Ok(ProbedObject {
        provenance: Provenance::Probed(build),
        opcodes,
        arms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape the preprocessed interpreter actually has: symbolic
    /// enumerators (they are enum constants, not macros) and the aliased GEMM
    /// triple falling through to one body.
    const FAKE: &str = r#"
static void helper(void) { switch (x) { case PLOW_DOP_GEMM: break; } }
__device__ void plow_exec(const PlowDevInst* in) {
    switch (in->op) {
    case PLOW_DOP_NOP: break;
    case PLOW_DOP_GEMM:
    case PLOW_DOP_GEMM_MED:
    case PLOW_DOP_GEMM_SMALL:
        d_gemm();
        break;
    case PLOW_DOP_MAMBA2_SCAN: d_mamba(); break;
    default: __trap();
    }
}
"#;

    #[test]
    fn reads_fallthrough_labels_as_separate_arms() {
        let ops = dispatched_opcodes(FAKE, "plow_exec").unwrap();
        assert!(ops.contains(&(DevOp::Gemm as u16)));
        assert!(ops.contains(&(DevOp::GemmMed as u16)));
        assert!(ops.contains(&(DevOp::GemmSmall as u16)));
        assert!(ops.contains(&(DevOp::Nop as u16)));
        assert!(ops.contains(&(DevOp::Mamba2Scan as u16)));
    }

    /// A preprocessed CUDA TU is mostly vendor headers. Arms are scoped to the
    /// dispatch function so an unrelated switch cannot invent kernels.
    #[test]
    fn ignores_switches_outside_the_dispatch_function() {
        let text = "static void other(void){ switch(y){ case PLOW_DOP_LAYERNORM: break; } }\n"
            .to_string()
            + FAKE;
        let ops = dispatched_opcodes(&text, "plow_exec").unwrap();
        assert!(
            !ops.contains(&(DevOp::LayerNorm as u16)),
            "LAYERNORM is in an unrelated switch"
        );
    }

    /// The numeric `case` labels that litter vendor headers must never be read
    /// as opcodes. This is why matching is by name.
    #[test]
    fn numeric_case_labels_are_never_opcodes() {
        let text = "__device__ void plow_exec(int op){ switch(op){ case 8: case 15: break; } }";
        assert!(dispatched_opcodes(text, "plow_exec").unwrap().is_empty());
    }

    #[test]
    fn a_missing_dispatch_function_is_an_error() {
        assert!(matches!(
            dispatched_opcodes(FAKE, "not_here"),
            Err(ProbeError::NoDispatchFn { .. })
        ));
    }

    /// A name the Rust ABI does not know is ignored rather than invented.
    #[test]
    fn unknown_opcode_names_are_ignored() {
        let text =
            "__device__ void plow_exec(int op){ switch(op){ case PLOW_DOP_NOT_REAL_XYZ: break; } }";
        assert!(dispatched_opcodes(text, "plow_exec").unwrap().is_empty());
    }

    #[test]
    fn braces_inside_literals_do_not_unbalance_the_scan() {
        let text = r#"
__device__ void plow_exec(int op) {
    const char* s = "}{";
    switch (op) { case PLOW_DOP_GEMM: break; }
}
"#;
        assert!(dispatched_opcodes(text, "plow_exec")
            .unwrap()
            .contains(&(DevOp::Gemm as u16)));
    }

    /// A declaration is not a definition; the arms come from the latter.
    #[test]
    fn skips_forward_declarations() {
        let text = "__device__ void plow_exec(const PlowDevInst* in);\n".to_string() + FAKE;
        assert!(dispatched_opcodes(&text, "plow_exec")
            .unwrap()
            .contains(&(DevOp::Gemm as u16)));
    }

    /// A mention is not a label: the identifier must be followed by a colon.
    #[test]
    fn an_opcode_mention_without_a_colon_is_not_an_arm() {
        let text = "__device__ void plow_exec(int op){ int x = PLOW_DOP_GEMM; switch(op){ case PLOW_DOP_NOP: break; } }";
        let ops = dispatched_opcodes(text, "plow_exec").unwrap();
        assert!(
            !ops.contains(&(DevOp::Gemm as u16)),
            "a mention is not an arm"
        );
        assert!(ops.contains(&(DevOp::Nop as u16)));
    }

    #[test]
    fn a_missing_compiler_is_reported_as_such() {
        let t = ProbeTarget {
            compiler: "definitely-not-a-compiler-xyz".into(),
            arch_flag: "-arch=sm_90a".into(),
            includes: vec![],
            defines: vec![],
            source: "/dev/null".into(),
            dispatch_fn: "plow_exec".into(),
        };
        assert!(matches!(
            t.preprocess(),
            Err(ProbeError::CompilerMissing { .. })
        ));
    }
}
