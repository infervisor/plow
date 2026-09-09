//! The `PLOW_*` env contract under the clap-backed RuntimeConfig: every bool
//! knob in this codebase is set `=1` / `=0` by scripts, benches and docs, so
//! the env fallback MUST read "1" as true and "0" as false — clap's default
//! bool parser accepts only "true"/"false", and a SetTrue flag would read
//! mere presence as true, silently inverting `PLOW_PRELOAD=0`.
//!
//! One test fn: `RuntimeConfig::get()` caches its env snapshot, and these
//! set_vars are process-global.

use plowrt::config::RuntimeConfig;

#[test]
fn env_zero_and_one_mean_false_and_true() {
    // Bool default-true, disabled by =0 (the PR #55 scripts' convention).
    std::env::set_var("PLOW_PRELOAD", "0");
    // Bool default-false, enabled by =1 (every campaign serve script).
    std::env::set_var("PLOW_PF_SEG_GRAPH", "1");
    // Value knobs ride along.
    std::env::set_var("PLOW_KV_POOL_MIB", "256");
    std::env::set_var("PLOW_PF_SEG_PURE", "fp8");
    std::env::set_var("PLOW_PF_CHUNK", "4096");
    std::env::set_var("PLOW_PF_INTERLEAVE", "1024");
    std::env::set_var("PLOW_PF_DEFER_DECODE", "1");
    std::env::set_var("PLOW_PF_BATCH", "1");
    std::env::set_var("PLOW_TP_PREFILL_SEGMENT_MAJOR", "0");
    std::env::set_var("PLOW_ANE_MLP", "1");
    std::env::set_var("PLOW_METAL_SERIAL", "0");
    std::env::set_var("PLOW_ANE_MLP_LAYERS", "2");
    std::env::set_var("PLOW_ANE_MLP_PLACEMENT", "/tmp/placement-probe");
    std::env::set_var("PLOW_ANE_MLP_CACHE", "/tmp/channel-cache");
    std::env::set_var("PLOW_ANE_MLP_FAIL", "after_join");
    std::env::set_var("PLOW_BACKEND", "cpu");

    let c = RuntimeConfig::get();
    assert!(!c.preload, "PLOW_PRELOAD=0 must disable preload");
    assert!(
        c.nv.pf_seg_graph,
        "PLOW_PF_SEG_GRAPH=1 must enable seg graphs"
    );
    assert_eq!(c.kv_pool_mib, 256);
    assert_eq!(c.nv.pf_seg_pure.as_deref(), Some("fp8"));
    assert_eq!(c.nv.pf_chunk, 4096);
    assert_eq!(c.nv.pf_chunk_rows(), 4096);
    assert_eq!(c.nv.pf_interleave, 1024);
    assert!(c.nv.pf_defer_decode);
    assert!(c.nv.pf_batch);
    assert!(!c.amd.tp_prefill_segment_major);
    assert!(c.apple.ane_mlp);
    assert!(!c.apple.serial);
    assert_eq!(c.apple.ane_mlp_layers, Some(2));
    assert_eq!(
        c.apple.ane_mlp_placement.as_deref(),
        Some(std::path::Path::new("/tmp/placement-probe"))
    );
    assert_eq!(
        c.apple.ane_mlp_cache.as_deref(),
        Some(std::path::Path::new("/tmp/channel-cache"))
    );
    assert_eq!(c.apple.ane_mlp_fail.as_deref(), Some("after_join"));
    assert_eq!(c.apple.backend.as_deref(), Some("cpu"));
}
