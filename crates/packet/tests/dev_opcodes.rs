//! Cross-language lock for device *opcode values*.
//!
//! `dev_abi.rs` locks the struct layouts — how wide a `PlowDevInst` is, where
//! its fields sit. It never compares a single opcode number. So
//! `DevOp::GemmMed = 15` and `PLOW_DOP_GEMM_MED = 15` agreeing was, until this
//! file, pure human discipline. They do currently agree (84 opcodes, no
//! collisions), and this test is what keeps that true.
//!
//! A renumber on one side is the worst kind of bug: the interpreter dispatches
//! a real kernel, just not the one the compiler meant. There is no tag to catch
//! it, and on AMD the default arm silently no-ops rather than trapping, so a
//! drifted opcode can read as "the model got slightly worse" rather than as a
//! crash.
//!
//! Three directions of drift, three checks:
//!
//! 1. **Rust variant renumbered / C name missing** — the generated probe
//!    references every `DevOp::c_name()`, so a name absent from `dev_isa.h` is a
//!    C compile error, and a changed value is an assert.
//! 2. **C opcode with no Rust variant** — `dev_isa.h` is parsed for
//!    `PLOW_DOP_*` and every one must appear in `DevOp::ALL`.
//! 3. **Rust variant missing from `DevOp::ALL`** — `dev.rs` is parsed for its
//!    own enum body, so the hand-maintained `ALL` table cannot fall behind.

use std::collections::BTreeMap;
use std::process::Command;

use packet::dev::DevOp;

const ISA_H: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../runtime/common/dev_isa.h");
const DEV_RS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/dev.rs");

/// Compile a probe that prints every opcode the Rust side believes in, and read
/// the values back out of the C preprocessor.
fn c_opcode_values() -> Option<BTreeMap<String, u32>> {
    if !std::path::Path::new(ISA_H).exists() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("plow_dev_opcodes_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.c");
    let bin = dir.join("probe");

    let mut probe = format!("#include <stdio.h>\n#include \"{ISA_H}\"\nint main(void) {{\n");
    for op in DevOp::ALL {
        // Referencing the name is itself the existence check.
        probe.push_str(&format!(
            "    printf(\"{n} %u\\n\", (unsigned){n});\n",
            n = op.c_name()
        ));
    }
    probe.push_str("    printf(\"PLOW_DOP__COUNT %u\\n\", (unsigned)PLOW_DOP__COUNT);\n");
    probe.push_str("    return 0;\n}\n");
    std::fs::write(&src, probe).ok()?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let out = Command::new(&cc).arg(&src).arg("-o").arg(&bin).output().ok()?;
    if !out.status.success() {
        // A missing PLOW_DOP_* name lands here. Surface it: it is the whole point.
        panic!(
            "probe failed to compile — an opcode named by DevOp::c_name() is absent from \
             dev_isa.h:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let run = Command::new(&bin).output().ok()?;
    let text = String::from_utf8_lossy(&run.stdout).into_owned();
    let _ = std::fs::remove_dir_all(&dir);

    Some(
        text.lines()
            .filter_map(|l| {
                let (k, v) = l.split_once(' ')?;
                Some((k.to_string(), v.trim().parse().ok()?))
            })
            .collect(),
    )
}

/// Strip a trailing `/* … */` or `// …` and the separating comma, leaving the
/// bare integer. Most opcode lines carry an inline tile/shape comment, so
/// parsing the value without this silently drops them — it hid 15 of 84
/// opcodes from this test the first time around.
fn numeric_prefix(val: &str) -> Option<u32> {
    let val = val.trim();
    let digits: String = val.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    // Guard against `= 15abc`: whatever follows the digits must be a separator
    // or the start of a comment, never more value.
    let rest = val[digits.len()..].trim_start();
    let ok = rest.is_empty()
        || rest.starts_with(',')
        || rest.starts_with("/*")
        || rest.starts_with("//")
        || rest.starts_with('}');
    ok.then(|| digits.parse().ok()).flatten()
}

/// Every `PLOW_DOP_<NAME> = <n>` in the header, by source text.
fn c_declared_opcodes() -> Vec<(String, u32)> {
    let text = std::fs::read_to_string(ISA_H).expect("read dev_isa.h");
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("PLOW_DOP_") else { continue };
        let Some((name, val)) = rest.split_once('=') else { continue };
        if name.trim().starts_with('_') {
            continue; // PLOW_DOP__COUNT is a bound, not an opcode.
        }
        if let Some(v) = numeric_prefix(val) {
            out.push((format!("PLOW_DOP_{}", name.trim()), v));
        }
    }
    out
}

/// Every `Variant = <n>,` in the `DevOp` enum body, by source text.
fn rust_declared_variants() -> Vec<(String, u32)> {
    let text = std::fs::read_to_string(DEV_RS).expect("read dev.rs");
    let start = text.find("pub enum DevOp {").expect("DevOp enum");
    let body = &text[start..];
    let end = body.find("\n}").expect("DevOp enum end");
    let mut out = Vec::new();
    for line in body[..end].lines() {
        let line = line.trim();
        if line.starts_with("//") || line.starts_with("/*") {
            continue;
        }
        let Some((name, val)) = line.split_once('=') else { continue };
        let name = name.trim();
        if name.is_empty() || !name.chars().next().unwrap().is_ascii_uppercase() {
            continue;
        }
        if !name.chars().all(|c| c.is_ascii_alphanumeric()) {
            continue;
        }
        if let Some(v) = numeric_prefix(val) {
            out.push((name.to_string(), v));
        }
    }
    out
}

#[test]
fn all_table_covers_every_devop_variant() {
    let declared = rust_declared_variants();

    let listed: BTreeMap<&str, u32> = DevOp::ALL
        .iter()
        .map(|op| {
            let name = format!("{op:?}");
            (Box::leak(name.into_boxed_str()) as &str, *op as u32)
        })
        .collect();

    for (name, val) in &declared {
        match listed.get(name.as_str()) {
            None => panic!("DevOp::{name} = {val} is missing from DevOp::ALL"),
            Some(got) => assert_eq!(got, val, "DevOp::ALL has the wrong value for {name}"),
        }
    }
    assert_eq!(
        listed.len(),
        declared.len(),
        "DevOp::ALL lists {} opcodes but the enum declares {}",
        listed.len(),
        declared.len()
    );
}

#[test]
fn rust_and_c_agree_on_every_opcode_value() {
    let Some(c) = c_opcode_values() else {
        eprintln!("skipping: no C toolchain or dev_isa.h not found");
        return;
    };

    for op in DevOp::ALL {
        let name = op.c_name();
        let cval = *c
            .get(name)
            .unwrap_or_else(|| panic!("C probe did not report {name}"));
        assert_eq!(
            *op as u32, cval,
            "opcode value drift: Rust {op:?} = {}, C {name} = {cval}",
            *op as u32
        );
    }

    assert_eq!(
        DevOp::COUNT as u32,
        *c.get("PLOW_DOP__COUNT").expect("PLOW_DOP__COUNT"),
        "DevOp::COUNT must mirror PLOW_DOP__COUNT — it bounds the dispatch table"
    );
}

#[test]
fn every_c_opcode_has_a_rust_variant() {
    let declared = c_declared_opcodes();
    let known: BTreeMap<&str, u32> =
        DevOp::ALL.iter().map(|op| (op.c_name(), *op as u32)).collect();

    // Self-checking instead of a magic threshold: if the parser starts dropping
    // lines, the count diverges from `ALL` and this fires. A `> 50` guard here
    // passed at 69/84 while 15 opcodes went unchecked.
    assert_eq!(
        declared.len(),
        known.len(),
        "parsed {} opcodes out of dev_isa.h but DevOp::ALL has {} — if the header really \
         gained an opcode this is a genuine drift, otherwise the parser is dropping lines",
        declared.len(),
        known.len()
    );

    for (name, val) in &declared {
        match known.get(name.as_str()) {
            None => panic!(
                "{name} = {val} is declared in dev_isa.h but has no DevOp variant — the \
                 compiler can never emit it, and on AMD the default arm no-ops rather than traps"
            ),
            Some(got) => assert_eq!(got, val, "{name} disagrees between Rust and C"),
        }
    }
}

/// The opcode space is sparse by design (reserved bands for tp, fp8, MoE, MLA).
/// Duplicates are not: two names on one value means one of them never dispatches.
#[test]
fn opcode_values_are_unique() {
    let mut seen: BTreeMap<u32, &str> = BTreeMap::new();
    for op in DevOp::ALL {
        let name = op.c_name();
        if let Some(prev) = seen.insert(*op as u32, name) {
            panic!("opcode {} is claimed by both {prev} and {name}", *op as u32);
        }
    }
}
