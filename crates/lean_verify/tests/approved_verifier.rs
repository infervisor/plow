//! The verifier identity contract: receipts are accepted only from executables listed in
//! `lean-plow/approved-verifiers.json`, each paired with the Lean sources it was built from.

use plow_asset::certificates::approved_verifiers;

fn lean_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lean-plow")
}

/// A change to the Lean sources must add an approved entry for the verifier it builds.
#[test]
fn current_lean_sources_have_an_approved_verifier() {
    let sources = lean_verify::lean_sources_sha256(&lean_dir()).unwrap();
    let list = approved_verifiers().unwrap();
    assert!(
        list.verifiers.iter().any(|v| v.lean_sources_sha256 == sources),
        "lean-plow sources {sources} have no entry in lean-plow/approved-verifiers.json; \
         build plow_verify and add its sha256 with this lean_sources_sha256"
    );
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn built_verifier_is_approved_for_current_sources() {
    let sources = lean_verify::lean_sources_sha256(&lean_dir()).unwrap();
    let built = lean_verify::verifier_sha256().unwrap();
    let list = approved_verifiers().unwrap();
    assert!(
        list.verifiers.iter().any(|v| v.sha256 == built && v.lean_sources_sha256 == sources),
        "built verifier {built} is not approved for lean-plow sources {sources}"
    );
}
