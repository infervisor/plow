#[cfg(unix)]
#[test]
fn df_cache_binds_exact_request_and_verifier_and_retains_envelope() {
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!("plow-verifier-cache-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("verify");
    let write = |checkpoint: &str, ok: bool, note: &str| {
        let cert = serde_json::json!({"checkpoint": checkpoint, "ok": ok, "notes": note});
        std::fs::write(
            &bin,
            format!("#!/bin/sh\ncat >/dev/null\nprintf '%s' '{cert}'\n"),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
    };
    std::env::set_var("PLOW_VERIFY_BIN", &bin);
    write(
        "D",
        true,
        "scope=coarse-dependency; theorem=checkPaths_sound",
    );
    let payload = serde_json::json!({"task_graph": {"n": 2, "edges": [[0,1]]}});
    let original = lean_verify::call("D", payload.clone()).unwrap();
    let cached = lean_verify::call("D", payload.clone()).unwrap();
    assert_eq!(cached.checkpoint, "D");
    assert_eq!(original.notes, cached.notes);
    // Shared checker does not imply identical checkpoint-specific envelopes.
    assert!(lean_verify::call("F", payload.clone()).is_err());
    // An unequal request must execute: the stub returns D, not F.
    assert!(lean_verify::call("F", serde_json::json!({"different": true})).is_err());
    write("D", false, "new verifier rejects");
    let fresh = lean_verify::call("D", payload.clone()).unwrap();
    assert!(!fresh.ok);
    assert_eq!(fresh.notes.as_deref(), Some("new verifier rejects"));
    write("K", true, "wrong checkpoint");
    assert!(lean_verify::call("D", payload).is_err());
    std::env::remove_var("PLOW_VERIFY_BIN");
    std::fs::remove_dir_all(dir).unwrap();
}
