//! End-to-end golden: drive the full `devgen::run` pipeline (arch dispatch →
//! declare → emit_phase → `to_blob` → file write) on a synthetic checkpoint and
//! pin the output blob's byte hash.
//!
//! This is the parity net for the port out of `bin/gemma4.rs`: the emitter is
//! deterministic, so any refactor that changes a single emitted byte trips the
//! hash. The GLM op-sequence tests in the library lock the MLA/MoE path; this
//! locks the dense Qwen3 path and the `run()` wiring itself.

use std::path::Path;

/// FNV-1a over the blob — a stable fingerprint with no external crate.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A faithful-shape miniature Qwen3 config: dense, all-global attention, GQA,
/// tied embeddings. Small dims keep the blob tiny; the emit path is identical to
/// the production shapes.
fn write_qwen3_config(dir: &Path) {
    let cfg = r#"{
      "model_type": "qwen3",
      "hidden_size": 512,
      "intermediate_size": 1024,
      "num_hidden_layers": 2,
      "num_attention_heads": 8,
      "head_dim": 64,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 4096,
      "rope_theta": 1000000.0,
      "rope_scaling": null,
      "tie_word_embeddings": true
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

fn emit(dir: &Path, ctx: u32, n_cu: u32, tp: u32) -> Vec<u8> {
    emit_with(dir, ctx, n_cu, tp, true)
}

fn emit_with(dir: &Path, ctx: u32, n_cu: u32, tp: u32, rope_gen: bool) -> Vec<u8> {
    let out = dir.join("model.pkt");
    devgen::run(devgen::EmitArgs {
        dir: dir.to_path_buf(),
        ctx,
        out: out.to_str().unwrap().to_string(),
        n_cu,
        tp,
        block_spec: None,
        embed_cubin: None,
        embed_hsaco: None,
        rope_gen,
        l2_layout: None,
        gpu: String::new(),
    });
    std::fs::read(&out).unwrap()
}

/// The load-bearing property of the v7 format: what the runtime materialises from
/// a recipe is byte-for-byte what `--no-rope-gen` bakes into the init section.
///
/// If these ever diverge nothing fails — the model just serves subtly wrong
/// tokens — so this compares the actual bytes rather than trusting that both
/// sides call the same function.
#[test]
fn generated_tables_match_baked_tables() {
    use packet::devbuild::{
        BlobSectionEntry, BlobTensor, BLOB_MAGIC, BLOB_MAGIC_V7, INIT_NONE, NAME_LEN,
        SECT_GEN_TENSORS, SECT_MAGIC,
    };
    use packet::rope::GenTensor;

    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden_qwen3_gen");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    write_qwen3_config(&root);

    let baked = emit_with(&root, 512, 128, 1, false);
    let gen = emit_with(&root, 512, 128, 1, true);

    assert_eq!(&baked[..8], BLOB_MAGIC, "--no-rope-gen must stay v5");
    assert_eq!(&gen[..8], BLOB_MAGIC_V7, "rope-gen must bump the magic to v7");
    assert!(
        gen.len() < baked.len(),
        "v7 blob ({}) should be smaller than baked ({})",
        gen.len(),
        baked.len()
    );

    // Decl-table walk, shared by both blobs: (name, bytes, init_off).
    let decls = |b: &[u8]| -> Vec<(String, u64, u64)> {
        let n = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
        let sz = std::mem::size_of::<BlobTensor>();
        (0..n)
            .map(|k| {
                let d = 64 + k * sz;
                let e = b[d..d + NAME_LEN].iter().position(|&x| x == 0).unwrap();
                (
                    String::from_utf8(b[d..d + e].to_vec()).unwrap(),
                    u64::from_le_bytes(b[d + NAME_LEN..d + NAME_LEN + 8].try_into().unwrap()),
                    u64::from_le_bytes(
                        b[d + NAME_LEN + 8..d + NAME_LEN + 16].try_into().unwrap(),
                    ),
                )
            })
            .collect()
    };
    let baked_decls = decls(&baked);
    let gen_decls = decls(&gen);
    let init_at = 64 + baked_decls.len() * std::mem::size_of::<BlobTensor>();
    let init_bytes = u64::from_le_bytes(baked[32..40].try_into().unwrap()) as usize;
    let init = &baked[init_at..init_at + init_bytes];

    // The v7 recipe array, read through the section directory at reserved[0].
    let dir_off = u64::from_le_bytes(gen[40..48].try_into().unwrap()) as usize;
    assert_eq!(&gen[dir_off..dir_off + 4], SECT_MAGIC, "section directory magic");
    let n_sect = u32::from_le_bytes(gen[dir_off + 4..dir_off + 8].try_into().unwrap()) as usize;
    let ent_sz = std::mem::size_of::<BlobSectionEntry>();
    let recipes: Vec<GenTensor> = (0..n_sect)
        .map(|k| dir_off + 8 + k * ent_sz)
        .find(|&e| u32::from_le_bytes(gen[e..e + 4].try_into().unwrap()) == SECT_GEN_TENSORS)
        .map(|e| {
            let off = u64::from_le_bytes(gen[e + 8..e + 16].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(gen[e + 16..e + 24].try_into().unwrap()) as usize;
            let sz = std::mem::size_of::<GenTensor>();
            assert_eq!(size % sz, 0, "gen section is not a whole number of recipes");
            (0..size / sz)
                .map(|k| unsafe {
                    std::ptr::read_unaligned(gen[off + k * sz..].as_ptr() as *const GenTensor)
                })
                .collect()
        })
        .expect("v7 blob carries a SECT_GEN_TENSORS section");

    let mut compared = 0;
    for (name, bytes, off) in &baked_decls {
        if *off == INIT_NONE || !name.starts_with("in.") {
            continue; // act.* zero buffers still ride the init section
        }
        let want = &init[*off as usize..(*off + *bytes) as usize];
        let ti = gen_decls.iter().position(|(n, _, _)| n == name).unwrap();
        assert_eq!(
            gen_decls[ti].2, INIT_NONE,
            "{name} should carry no init bytes in a v7 blob"
        );
        let recipe = recipes
            .iter()
            .find(|g| g.tensor as usize == ti)
            .unwrap_or_else(|| panic!("{name} has no v7 recipe"));
        assert_eq!(
            recipe.generate().unwrap(),
            want,
            "{name}: generated table differs from the baked one"
        );
        compared += 1;
    }
    assert_eq!(compared, 4, "expected 4 RoPE tables, compared {compared}");
}

#[test]
fn qwen3_dense_blob_is_stable() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden_qwen3");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    write_qwen3_config(&root);

    let blob = emit(&root, 512, 128, 1);
    assert!(!blob.is_empty(), "run() produced an empty blob");

    // Determinism: a second emit into a fresh dir yields byte-identical output.
    let root2 = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden_qwen3_b");
    let _ = std::fs::remove_dir_all(&root2);
    std::fs::create_dir_all(&root2).unwrap();
    write_qwen3_config(&root2);
    let blob2 = emit(&root2, 512, 128, 1);
    assert_eq!(fnv1a(&blob), fnv1a(&blob2), "emit is not deterministic");

    // Golden fingerprint — pins the exact bytes of the dense Qwen3 program.
    // If an intentional emitter change moves this, update the constant in the
    // same commit and say why.
    //
    // Moved 694628 -> 432828 when the RoPE tables became v7 recipes instead of
    // expanded init bytes. The delta is exactly the four tables
    // (4 x ctx 512 x hd/2 32 x 4 B = 262144) less the 344 B the v7 container adds
    // (four 72-byte recipes, a 48-byte section entry, and the 8-byte directory
    // header). `generated_tables_match_baked_tables` above pins the bytes those
    // recipes expand to, which is the property this hash used to cover.
    const GOLDEN_FNV1A: u64 = 0x1323_437f_82f1_9604;
    const GOLDEN_LEN: usize = 432828;
    assert_eq!(blob.len(), GOLDEN_LEN, "dense Qwen3 blob length changed");
    assert_eq!(
        fnv1a(&blob),
        GOLDEN_FNV1A,
        "dense Qwen3 blob hash changed (len={})",
        blob.len()
    );
}

/// Miniature Gemma-4 text config: dense, sliding+full layer mix, GQA, tied embeds,
/// GeGLU, q/k/v norms. Exercises the Gemma-specific `emit_phase` branches. GQA=4
/// (heads 8 / kv 2) is even, so the flash-decode GF=2 divides it.
fn write_gemma_config(dir: &Path) {
    let cfg = r#"{
      "model_type": "gemma4_text",
      "hidden_size": 512,
      "intermediate_size": 1024,
      "num_hidden_layers": 2,
      "num_attention_heads": 8,
      "head_dim": 64,
      "global_head_dim": 64,
      "num_key_value_heads": 2,
      "num_global_key_value_heads": 2,
      "sliding_window": 512,
      "rms_norm_eps": 1e-6,
      "vocab_size": 4096,
      "final_logit_softcapping": 0.0,
      "tie_word_embeddings": true,
      "layer_types": ["sliding_attention", "full_attention"],
      "rope_parameters": {
        "sliding_attention": { "rope_theta": 10000.0, "partial_rotary_factor": 1.0 },
        "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 1.0 }
      }
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

/// Miniature Llama-3 config: dense, all-global attention, SwiGLU, no q/k norm,
/// untied lm_head. Exercises the Llama/Qwen `emit_phase` branches distinct from Gemma.
fn write_llama_config(dir: &Path) {
    let cfg = r#"{
      "model_type": "llama",
      "hidden_size": 512,
      "intermediate_size": 1024,
      "num_hidden_layers": 2,
      "num_attention_heads": 8,
      "head_dim": 64,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-5,
      "vocab_size": 4096,
      "rope_theta": 500000.0,
      "rope_scaling": null,
      "tie_word_embeddings": false
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

/// Byte-golden for the dense **Gemma** path — protects the Gemma `emit_phase`
/// branches across the trait refactor (plans/devgen-trait-refactor.md). Bootstrap:
/// the assert message prints the actual hash on first run; pin it here.
#[test]
fn gemma_dense_blob_is_stable() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden_gemma");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    write_gemma_config(&root);
    let blob = emit(&root, 512, 128, 1);
    const GEMMA_GOLDEN_FNV1A: u64 = 0xefe5_b0ec_5b7d_a84f;
    assert_eq!(
        fnv1a(&blob),
        GEMMA_GOLDEN_FNV1A,
        "gemma dense blob hash changed (len={}, actual={:#018x})",
        blob.len(),
        fnv1a(&blob)
    );
}

/// Byte-golden for the dense **Llama** path.
#[test]
fn llama_dense_blob_is_stable() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden_llama");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    write_llama_config(&root);
    let blob = emit(&root, 512, 128, 1);
    const LLAMA_GOLDEN_FNV1A: u64 = 0xca67_12ad_d306_d875;
    assert_eq!(
        fnv1a(&blob),
        LLAMA_GOLDEN_FNV1A,
        "llama dense blob hash changed (len={}, actual={:#018x})",
        blob.len(),
        fnv1a(&blob)
    );
}
