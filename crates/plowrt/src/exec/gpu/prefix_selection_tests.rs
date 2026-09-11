use super::*;
use crate::asset::devblob::{DevProg, DevSection, DevTensor};

#[test]
fn long_context_live_kv_uses_reserved_ring_windows() {
    assert!(!live_rings_for_capacity(
        false,
        true,
        Some(65_536),
        Some(32)
    ));
    assert!(live_rings_for_capacity(
        false,
        true,
        Some(131_072),
        Some(16)
    ));
    assert!(!live_rings_for_capacity(
        false,
        false,
        Some(131_072),
        Some(64)
    ));
    assert!(live_rings_for_capacity(true, true, Some(1), Some(1)));
}

#[test]
fn wide_live_kv_uses_reserved_ring_windows_without_changing_b32() {
    assert!(!live_rings_for_capacity(
        false,
        true,
        Some(20_480),
        Some(32)
    ));
    assert!(live_rings_for_capacity(false, true, Some(20_480), Some(64)));
}

#[test]
fn automatic_prefix_selection_requires_compatible_execution_and_valid_kv_layout() {
    let dir = std::env::temp_dir().join(format!("plow-prefix-selection-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "layer_types": ["sliding_attention", "full_attention"],
            "num_key_value_heads": 2, "num_global_key_value_heads": 1,
            "head_dim": 256, "global_head_dim": 512, "sliding_window": 1024,
        }))
        .unwrap(),
    )
    .unwrap();
    let tensor = |name: &str, bytes| DevTensor {
        name: name.into(),
        bytes,
        init: None,
    };
    let mut blob = DevBlob {
        n_cu: 132,
        flags: 0,
        target: 0,
        tensors: vec![
            tensor("in.pos", 4096 * 4),
            tensor("kv.0.k", 2 << 20),
            tensor("kv.0.v", 2 << 20),
            tensor("kv.1.k", 4 << 20),
            tensor("kv.1.v", 4 << 20),
        ],
        init: Vec::new(),
        kvrow: Vec::new(),
        sections: Vec::new(),
        gen: Vec::new(),
        tp: None,
        progs: vec![DevProg {
            t: 1,
            packed_prefill_only: false,
            token_batch_body: false,
            n_counter: 0,
            insts: Vec::new(),
            stream: Vec::new(),
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: vec![0, 0],
            l2_domains: 0,
        }],
    };
    let mut cfg = RuntimeConfig::get().clone();
    cfg.nv.vmm_prefix = None;
    cfg.nv.vmm_live = false;
    cfg.nv.vmm_live_rings = false;
    cfg.prefix_cache = true;
    cfg.pf_batch = false;
    let selected = |blob: &DevBlob, cfg: &RuntimeConfig, cc, gran| {
        GpuEngine::select_vmm_prefix_layout(blob, &dir, cfg, cc, gran).is_some()
    };
    assert!(selected(&blob, &cfg, (9, 0), 2 << 20));
    assert!(!selected(&blob, &cfg, (12, 0), 2 << 20));
    assert!(!selected(&blob, &cfg, (9, 0), 16 << 20));
    cfg.pf_batch = true;
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    cfg.pf_batch = false;
    cfg.nv.vmm_live = true;
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    assert!(cfg.nv_live_kv_enabled(true, true, false));
    cfg.nv.vmm_live = false;
    cfg.nv.vmm_prefix = Some(false);
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    cfg.nv.vmm_prefix = Some(true);
    assert!(selected(&blob, &cfg, (12, 0), 2 << 20));
    cfg.nv.vmm_prefix = None;
    for name in [
        plow_asset::mixed_step::SECTION,
        plow_asset::decode_objects::SECTION,
        plow_asset::decode_context::SECTION,
    ] {
        blob.sections.push(DevSection {
            kind: packet::devbuild::SECT_METADATA,
            name: name.into(),
            offset: 0,
            size: 0,
        });
        assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
        blob.sections.pop();
    }
    blob.tensors.push(tensor("state.qwen.0.gdn", 64));
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    blob.tensors.pop();
    blob.tensors[3].bytes /= 2;
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    blob.tensors[4].bytes /= 2;
    blob.tensors.push(tensor("kv.1.k_scale", 4096 * 4));
    blob.tensors.push(tensor("kv.1.v_scale", 4096 * 4));
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    cfg.nv.vmm_prefix = Some(true);
    assert!(selected(&blob, &cfg, (9, 0), 2 << 20));
    let mut geometry: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    geometry["num_key_value_heads"] = 0.into();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&geometry).unwrap(),
    )
    .unwrap();
    assert!(!selected(&blob, &cfg, (9, 0), 2 << 20));
    std::fs::remove_dir_all(dir).unwrap();
}
