//! Drift guard: parse JSON in the exact shape `plowc` emits, and round-trip our
//! own structs. If `plowc`'s reporting structs change a field name or the class
//! spelling, the literal-parse test below fails.

use plow_asset::*;

/// A `*.map.json` exactly as `plowc::MemoryReport` serializes it — snake_case
/// class strings, a KV paging block, two device segments.
const MAP_JSON: &str = r#"{
  "arena_bytes": 4096,
  "growable_base": 2048,
  "segments": [
    { "device": 0, "global_base": 0, "size": 2048, "growable_base": 1024 },
    { "device": 1, "global_base": 2048, "size": 2048, "growable_base": 3072 }
  ],
  "entries": [
    { "slot": 0, "name": "layers.0.self_attn.q_proj.weight", "class": "persistent",
      "offset": 0, "reserved": 512, "growable": false, "device": 0 },
    { "slot": 1, "name": "kv_cache", "class": "growable",
      "offset": 1024, "reserved": 1024, "growable": true, "device": 0 },
    { "slot": 2, "name": "act_scratch", "class": "scratch",
      "offset": 512, "reserved": 256, "growable": false, "device": 0 },
    { "slot": 3, "name": "tokens", "class": "request_io",
      "offset": 768, "reserved": 64, "growable": false, "device": 0 }
  ],
  "kv_paging": {
    "block_tokens": 16, "block_bytes": 4096,
    "kv_heads": 4, "head_dim": 128,
    "per_layer": [
      { "layer_idx": 0, "buffer_name": "kv_cache_L0", "initial_blocks": 4 }
    ]
  }
}"#;

/// `weights.json` as `plowc::Report` serializes it.
const MANIFEST_JSON: &str = r#"{
  "network": "transformer-block-gemma4-12b",
  "gpu": "H100 SXM5",
  "num_gpus": 1,
  "parallel": "tp",
  "weight_shared": true,
  "weight": { "bn": 128, "bk": 64 },
  "kv": { "block_seq": 16, "kv_heads": 4, "head_dim": 256 },
  "fusion": { "ops_before": 8, "ops_after": 5, "fused": 3 },
  "buckets": [
    { "phase": "decode", "batch": 1, "seq": 128,
      "packet_file": "decode_b1_s128.pkt", "packet_bytes": 2048,
      "instructions": 42, "tile_nodes": 30, "tasks": 40,
      "makespan": 100000, "ideal_makespan": 80000, "arena_bytes": 4096,
      "memory_file": "decode_b1_s128.map.json" }
  ]
}"#;

#[test]
fn parse_plowc_map_shape() {
    let m: MemoryMap = serde_json::from_str(MAP_JSON).expect("map.json parses");
    assert_eq!(m.arena_bytes, 4096);
    assert_eq!(m.segments.len(), 2);
    assert_eq!(m.entries.len(), 4);
    assert_eq!(m.get("kv_cache").unwrap().class, BufClass::Growable);
    assert_eq!(
        m.on_device("layers.0.self_attn.q_proj.weight", 0)
            .unwrap()
            .class,
        BufClass::Persistent
    );
    assert_eq!(m.local_offset(1, 2048), Some(0));
    let kv = m.kv_paging.as_ref().unwrap();
    assert_eq!(kv.block_tokens, 16);
    assert_eq!(kv.per_layer[0].initial_blocks, 4);
    m.validate().expect("structurally valid");
}

#[test]
fn parse_plowc_manifest_shape() {
    let man: Manifest = serde_json::from_str(MANIFEST_JSON).expect("weights.json parses");
    assert_eq!(man.parallel, "tp");
    assert_eq!(man.weight.unwrap().bn, 128);
    assert_eq!(man.buckets[0].packet_file, "decode_b1_s128.pkt");
    assert_eq!(man.buckets[0].phase, "decode");
}

#[test]
fn map_roundtrips() {
    let m: MemoryMap = serde_json::from_str(MAP_JSON).unwrap();
    let s = serde_json::to_string(&m).unwrap();
    let m2: MemoryMap = serde_json::from_str(&s).unwrap();
    assert_eq!(m2.entries.len(), m.entries.len());
    assert_eq!(m2.kv_paging.as_ref().unwrap().block_bytes, 4096);
}

#[test]
fn class_spellings_match_plowc() {
    // plowc writes these exact strings in MemoryReport::from_map.
    for (json, want) in [
        ("\"persistent\"", BufClass::Persistent),
        ("\"static\"", BufClass::Static),
        ("\"growable\"", BufClass::Growable),
        ("\"scratch\"", BufClass::Scratch),
        ("\"request_io\"", BufClass::RequestIo),
    ] {
        let c: BufClass = serde_json::from_str(json).unwrap();
        assert_eq!(c, want);
    }
}

/// Runtime consumes `weight_tiling` from `weights.json` and per-entry
/// `logical_shape` + `dtype` from `map.json` to arrange safetensor bytes
/// into the compiled arena. Guard the schema shape.
#[test]
fn parse_weight_tiling_and_logical_shape() {
    const WEIGHTS: &str = r#"{
      "network": "llama3-8b",
      "gpu": "H100 SXM5",
      "num_gpus": 1,
      "parallel": "tp",
      "weight_shared": true,
      "weight": { "bn": 256, "bk": 64 },
      "kv": null,
      "fusion": null,
      "buckets": [],
      "weight_tiling": {
        "bn": 256,
        "bk": 64,
        "element_dtype": "bf16",
        "elem_bytes": 2,
        "block_iteration": "n_major_k_inner",
        "within_block_layout": "n_outer_k_inner",
        "padding_policy": "zero_extend"
      }
    }"#;
    let man: Manifest = serde_json::from_str(WEIGHTS).expect("parses");
    let wt = man.weight_tiling.as_ref().expect("weight_tiling present");
    assert_eq!(wt.bn, 256);
    assert_eq!(wt.elem_bytes, 2);
    assert_eq!(wt.block_iteration, "n_major_k_inner");
    assert!(man.static_tensors.is_empty());
    assert!(!man.static_tensors_file_emitted);

    const MAP_WITH_SHAPES: &str = r#"{
      "arena_bytes": 4096, "growable_base": 4096,
      "segments": [ { "device": 0, "global_base": 0, "size": 4096, "growable_base": 4096 } ],
      "entries": [
        { "slot": 0, "name": "layers.0.q_proj.weight", "class": "persistent",
          "offset": 0, "reserved": 4096, "growable": false, "device": 0,
          "logical_shape": [4096, 4096], "dtype": "bf16" },
        { "slot": 1, "name": "input_norm.weight", "class": "persistent",
          "offset": 4096, "reserved": 8192, "growable": false, "device": 0 }
      ]
    }"#;
    let m: MemoryMap = serde_json::from_str(MAP_WITH_SHAPES).expect("parses");
    let q = m.get("layers.0.q_proj.weight").unwrap();
    assert_eq!(q.logical_shape.as_deref(), Some(&[4096i64, 4096i64][..]));
    assert_eq!(q.dtype.as_deref(), Some("bf16"));
    let norm = m.get("input_norm.weight").unwrap();
    assert!(norm.logical_shape.is_none()); // non-GEMM Persistent: no shape.
    assert!(norm.dtype.is_none());
}

/// The `block.json` descriptor exactly as the design notes
/// documents it (Kimi MLA+MoE block, layer 3). Guards the schema shape and the
/// mixed `["T", 7168]` symbolic/fixed dimension list.
const BLOCK_JSON: &str = r#"{
  "model": "kimi-k2.7", "arch": "mla_moe", "layer": 3,
  "kind": ["mla_attn", "moe_ffn"],
  "hidden": 7168, "dtype": "bf16",
  "dims": { "heads": 64, "kv_lora": 512, "q_lora": 1536,
            "n_exp": 384, "top_k": 8, "shared_exp": 1, "moe_inter": 2048 },
  "inputs":  [{"name":"act.x","shape":["T",7168],"dtype":"bf16"}],
  "outputs": [{"name":"act.x","shape":["T",7168],"dtype":"bf16"}],
  "carried_state": [
    {"role":"kv","tensors":["kv.0.k","kv.0.v"],"layout":"head_major"}
  ],
  "weights": {"mode":"symlink","ckpt":"kimi-k2.7","prefix":"model.layers.3."},
  "programs": {"prefill_buckets":[128,512,1024,2048,4096,8192],"decode_t":32}
}"#;

#[test]
fn parse_block_descriptor_shape() {
    let b: BlockDescriptor = serde_json::from_str(BLOCK_JSON).expect("parses");
    assert_eq!(b.model, "kimi-k2.7");
    assert_eq!(b.layer, 3);
    assert_eq!(b.kind, vec!["mla_attn", "moe_ffn"]);
    assert_eq!(b.dims.n_exp, Some(384));
    assert_eq!(b.dims.top_k, Some(8));
    // Mixed symbolic/fixed shape: ["T", 7168].
    assert_eq!(
        b.inputs[0].shape,
        vec![Dim::Symbolic("T".into()), Dim::Fixed(7168)]
    );
    assert_eq!(b.carried_state[0].role, "kv");
    assert_eq!(b.weights.prefix, "model.layers.3.");
    assert_eq!(b.programs.decode_t, 32);
}

#[test]
fn block_descriptor_roundtrips() {
    let b: BlockDescriptor = serde_json::from_str(BLOCK_JSON).unwrap();
    let s = serde_json::to_string(&b).unwrap();
    let b2: BlockDescriptor = serde_json::from_str(&s).unwrap();
    assert_eq!(b, b2);
}

/// A Gemma-4 dense-attn block (M1 `gemma4 --block`): `dims` carries `head_dim`
/// + `kv_heads`, and the omitted MLA/MoE keys stay absent (skip_serializing).
#[test]
fn dense_block_dims_roundtrip() {
    const DENSE_JSON: &str = r#"{
      "model": "gemma_dense", "arch": "gemma_dense", "layer": 5,
      "kind": ["dense_attn", "dense_ffn"],
      "hidden": 5376, "dtype": "bf16",
      "dims": { "heads": 32, "head_dim": 256, "kv_heads": 4 },
      "inputs":  [{"name":"act.x","shape":["T",5376],"dtype":"bf16"}],
      "outputs": [{"name":"act.x","shape":["T",5376],"dtype":"bf16"}],
      "carried_state": [
        {"role":"kv","tensors":["kv.5.k","kv.5.v"],"layout":"head_major"}
      ],
      "weights": {"mode":"symlink","ckpt":"gemma","prefix":"model.layers.5."},
      "programs": {"prefill_buckets":[128,512,1024,4096],"decode_t":8}
    }"#;
    let b: BlockDescriptor = serde_json::from_str(DENSE_JSON).expect("parses");
    assert_eq!(b.dims.head_dim, Some(256));
    assert_eq!(b.dims.kv_heads, Some(4));
    assert_eq!(b.dims.n_exp, None);
    assert_eq!(b.dsa_role, None, "dense block has no DSA role");
    let s = serde_json::to_string(&b).unwrap();
    assert!(
        !s.contains("n_exp"),
        "absent MLA/MoE keys must not serialize"
    );
    assert!(
        !s.contains("dsa_role"),
        "absent dsa_role must not serialize"
    );
    let b2: BlockDescriptor = serde_json::from_str(&s).unwrap();
    assert_eq!(b, b2);
}

/// A Nemotron-3 Mamba-2 mixer block (M4 `gemma4 --block` on the nemotron_h path):
/// `dims` carries the Mamba-2 keys (`d_conv`/`d_state`/`n_head`/`head_dim`/`d_inner`/
/// `n_groups`), the carried state is the conv + ssm state (roles `conv`/`ssm`, no KV),
/// and the omitted attn/MoE keys stay absent (skip_serializing).
#[test]
fn mamba2_block_descriptor_roundtrip() {
    const MAMBA_JSON: &str = r#"{
      "model": "Nemotron-H-30B", "arch": "nemotron_h", "layer": 0,
      "kind": ["mamba2"],
      "hidden": 4096, "dtype": "bf16",
      "dims": { "d_inner": 8192, "n_head": 128, "head_dim": 64,
                "d_state": 128, "d_conv": 4, "n_groups": 8 },
      "inputs":  [{"name":"act.x","shape":["T",4096],"dtype":"bf16"}],
      "outputs": [{"name":"act.x","shape":["T",4096],"dtype":"bf16"}],
      "carried_state": [
        {"role":"conv","tensors":["mamba.0.conv_state"],"layout":"conv"},
        {"role":"ssm","tensors":["mamba.0.ssm_state"],"layout":"ssm_head_major"}
      ],
      "weights": {"mode":"symlink","ckpt":"Nemotron-H-30B","prefix":"backbone.layers.0."},
      "programs": {"prefill_buckets":[],"decode_t":1}
    }"#;
    let b: BlockDescriptor = serde_json::from_str(MAMBA_JSON).expect("parses");
    assert_eq!(b.arch, "nemotron_h");
    assert_eq!(b.kind, vec!["mamba2"]);
    assert_eq!(b.dims.d_conv, Some(4));
    assert_eq!(b.dims.d_state, Some(128));
    assert_eq!(b.dims.n_head, Some(128));
    assert_eq!(b.dims.head_dim, Some(64));
    assert_eq!(b.dims.d_inner, Some(8192));
    assert_eq!(b.dims.n_groups, Some(8));
    assert_eq!(b.dims.kv_lora, None, "Mamba block has no KV/attn dims");
    // conv + ssm carried state, NO kv.
    assert_eq!(b.carried_state.len(), 2);
    assert_eq!(b.carried_state[0].role, "conv");
    assert_eq!(b.carried_state[1].role, "ssm");
    assert_eq!(b.carried_state[1].layout, "ssm_head_major");
    let s = serde_json::to_string(&b).unwrap();
    assert!(
        !s.contains("kv_lora"),
        "absent attn keys must not serialize"
    );
    assert!(
        !s.contains("\"heads\""),
        "absent attn keys must not serialize"
    );
    let b2: BlockDescriptor = serde_json::from_str(&s).unwrap();
    assert_eq!(b, b2);
}

/// A GLM-5.2 MLA+DSA block (M2 `gemma4 --block` on the glm_moe_dsa path): `dims`
/// carries MLA (`kv_lora`/`q_lora`) + MoE + DSA (`index_*`) keys, `dsa_role` marks
/// the IndexShare role, and a `reuse` layer carries `dsa_indices` in.
#[test]
fn glm_dsa_block_descriptor_roundtrip() {
    const GLM_JSON: &str = r#"{
      "model": "GLM-5.2-FP8", "arch": "glm_mla_dsa", "layer": 5,
      "kind": ["mla_dsa", "moe_ffn"],
      "hidden": 6144, "dtype": "fp8",
      "dims": { "heads": 64, "kv_lora": 512, "q_lora": 2048,
                "n_exp": 256, "top_k": 8, "shared_exp": 1, "moe_inter": 2048,
                "index_heads": 32, "index_dim": 128, "index_topk": 2048 },
      "dsa_role": "reuse",
      "inputs":  [{"name":"act.x","shape":["T",6144],"dtype":"bf16"}],
      "outputs": [{"name":"act.xnext","shape":["T",6144],"dtype":"bf16"}],
      "carried_state": [
        {"role":"kv","tensors":["kv.5.ckv","kv.5.krot"],"layout":"mla_latent"},
        {"role":"dsa_indices","tensors":["act.iidx"],"layout":"topk_positions"}
      ],
      "weights": {"mode":"symlink","ckpt":"GLM-5.2-FP8","prefix":"model.layers.5."},
      "programs": {"prefill_buckets":[],"decode_t":1}
    }"#;
    let b: BlockDescriptor = serde_json::from_str(GLM_JSON).expect("parses");
    assert_eq!(b.dsa_role.as_deref(), Some("reuse"));
    assert_eq!(b.dims.index_topk, Some(2048));
    assert_eq!(b.dims.kv_lora, Some(512));
    assert_eq!(b.carried_state[1].role, "dsa_indices");
    assert_eq!(b.carried_state[1].tensors, vec!["act.iidx"]);
    assert!(b.programs.prefill_buckets.is_empty());
    let s = serde_json::to_string(&b).unwrap();
    let b2: BlockDescriptor = serde_json::from_str(&s).unwrap();
    assert_eq!(b, b2);
}
