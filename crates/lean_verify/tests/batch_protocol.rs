#[cfg(unix)]
#[test]
fn batch_rejects_missing_reordered_and_inconsistent_envelopes() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("plow-verifier-batch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("verify");
    let requests = [("D", serde_json::json!({})), ("G", serde_json::json!({}))];
    std::env::set_var("PLOW_VERIFY_BIN", &bin);
    for response in [
        serde_json::json!({"ok": true, "certificates": []}),
        serde_json::json!({"ok": true, "certificates": [
            {"ok": true, "checkpoint": "G"}, {"ok": true, "checkpoint": "D"}]}),
        serde_json::json!({"ok": true, "certificates": [
            {"ok": true, "checkpoint": "D"}, {"ok": false, "checkpoint": "G"}]}),
    ] {
        std::fs::write(
            &bin,
            format!("#!/bin/sh\ncat >/dev/null\nprintf '%s' '{response}'\n"),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(lean_verify::call_batch(&requests).is_err());
    }
    std::env::remove_var("PLOW_VERIFY_BIN");
    std::fs::remove_dir_all(dir).unwrap();
}
