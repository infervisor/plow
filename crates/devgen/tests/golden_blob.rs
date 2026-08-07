//! End-to-end golden: drive the full `devgen::run` pipeline (arch dispatch →
//! declare → emit_phase → `to_blob` → file write) on a synthetic checkpoint and
//! pin the output blob's byte hash.
//!
//! This is the parity net for the port out of `bin/gemma4.rs`: the emitter is
//! deterministic, so any refactor that changes a single emitted byte trips the
//! hash. The GLM op-sequence tests in the library lock the MLA/MoE path; this
//! locks the dense Qwen3 path and the `run()` wiring itself.

use std::path::Path;

/// Serialises every test that EMITS.
///
/// `nvidia_conditioned_flags_never_change_the_amd_segment_count` has to set `PLOW_UNISEG` to test
/// it, and env is process-global while tests share a process. Without this, that test would
/// intermittently change the blob a concurrently-running golden HASH test emits — turning a real
/// regression net into a flaky one, which is worse than not having it. Poisoning is irrelevant
/// here (a panicking test has already failed), so the guard is unwrapped through.
static EMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn emit_guard() -> std::sync::MutexGuard<'static, ()> {
    EMIT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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
        // Empty arch ⇒ no build.json, so the golden blob's emission is
        // byte-identical to what it was before the manifest existed.
        arch: String::new(),
        emit_cfg: None,
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
    let _g = emit_guard();
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
    assert_eq!(
        &gen[..8],
        BLOB_MAGIC_V7,
        "rope-gen must bump the magic to v7"
    );
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
                    u64::from_le_bytes(b[d + NAME_LEN + 8..d + NAME_LEN + 16].try_into().unwrap()),
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
    assert_eq!(
        &gen[dir_off..dir_off + 4],
        SECT_MAGIC,
        "section directory magic"
    );
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
    let _g = emit_guard();
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

/// VERIFICATION IS READ-ONLY. Emitting with the Lean gate must produce THE SAME
/// BYTES as emitting without it.
///
/// This is the invariant that lets the gate be on by default: turning it on
/// cannot change what runs, so a blob built on a box with `plow_verify` and one
/// built on a box without it are the same artifact. Without that, every
/// measurement would silently depend on whether the person who built it had a
/// Lean toolchain, and no A/B across machines would be comparable.
///
/// The `&Model` signature already makes mutation impossible, but the point of
/// pinning it here is the whole PATH — hook construction, the pre-write call
/// site, and the manifest hand-off — not just the hook's type.
#[test]
fn the_verification_gate_does_not_change_a_single_emitted_byte() {
    let _g = emit_guard();
    let plain = tempdir("lean_gate_off");
    write_qwen3_config(&plain);
    let unverified = emit(&plain, 512, 128, 1);

    let gated = tempdir("lean_gate_on");
    write_qwen3_config(&gated);
    let out = gated.join("model.pkt");
    let saw_a_program = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = saw_a_program.clone();
    devgen::run_verified(
        devgen::EmitArgs {
            dir: gated.clone(),
            ctx: 512,
            out: out.to_str().unwrap().to_string(),
            n_cu: 128,
            tp: 1,
            block_spec: None,
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: String::new(),
            arch: String::new(),
            emit_cfg: None,
        },
        Some(Box::new(move |m: &packet::devbuild::Model| {
            // Guards against the test passing because the hook never ran.
            assert!(!m.progs.is_empty(), "the hook must see the real programs");
            seen.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(devgen::LeanReport {
                verified: true,
                oracle: true,
                reason: None,
            })
        })),
    );
    assert!(
        saw_a_program.load(std::sync::atomic::Ordering::SeqCst),
        "the verify hook was never invoked — this test would be vacuous"
    );

    let verified = std::fs::read(&out).unwrap();
    assert_eq!(unverified.len(), verified.len());
    assert_eq!(
        fnv1a(&unverified),
        fnv1a(&verified),
        "verification changed the emitted blob — it is supposed to be read-only"
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
    let _g = emit_guard();
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
    let _g = emit_guard();
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

// ===== WAVE-CLASS SEGMENTATION ==============================================================
//
// A prefill program is split into maximal same-wave-class runs, and the AMD host relaunches once
// per run: 4 waves for a `FlashPrefill`, 8 for everything else. The tags live in `StreamEnt::seg`.
//
// This test exists because losing them is INVISIBLE. A packet with no segment tags loads, runs, and
// differs from a correct one in nothing but that one field — but the host then dispatches the whole
// prefill program on the 4-wave flash object, whose body is `if (op == FLASH_PREFILL…)` with no
// switch, so every GEMM, norm and lm_head is silently dropped and `act.logits` comes back zero.
// It survived a full green suite and cost three agents a long time to find. One assertion on the
// segment count would have caught it at the first emit, which is the entire point of pinning it.
//
// The count is a FUNCTION OF THE LAYER COUNT, not a magic number: each layer contributes one
// class-8 run (its dense ops) and one class-4 run (its flash), and the output tail is one more
// class-8 run. So `2*L + 1` — 5 for this 2-layer fixture, 121 for Gemma-4 31B's 60 layers, which is
// exactly what the last known-good asset carries.
fn tempdir(name: &str) -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn segment_count(blob_dir: &Path, bucket: u32) -> usize {
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(blob_dir.join("build.json")).unwrap()).unwrap();
    man["programs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["kind"] == "prefill" && p["bucket"] == bucket)
        .count()
}

/// MI350X's L2 partitioning with AMD's MEASURED workgroup->XCD map. The map is not incidental:
/// the hardware dispatches workgroup *n* to XCD `n % 8` (100.0% on MI355X, see
/// `runtime/tests/xcd_map_gfx950_test.hip`), so the block formula would place a domain's packets
/// on workgroups spread across all eight XCDs.
fn amd_l2() -> packet::devbuild::L2Layout {
    packet::devbuild::L2Layout {
        sms: 32,
        domains: 8,
        map: packet::devbuild::L2Map::RoundRobin,
    }
}

/// The NVIDIA counterpart: consecutive blocks fill a GPC, so the domain is `n / sms`.
fn nv_l2() -> packet::devbuild::L2Layout {
    packet::devbuild::L2Layout {
        sms: 32,
        domains: 8,
        map: packet::devbuild::L2Map::Block,
    }
}

fn emit_arch(dir: &Path, arch: &str, gpu: &str, l2: Option<packet::devbuild::L2Layout>) {
    let out = dir.join("model.pkt");
    devgen::run(devgen::EmitArgs {
        dir: dir.to_path_buf(),
        ctx: 512,
        out: out.to_str().unwrap().to_string(),
        n_cu: 256,
        tp: 1,
        block_spec: None,
        embed_cubin: None,
        embed_hsaco: None,
        rope_gen: true,
        l2_layout: l2,
        gpu: gpu.to_string(),
        arch: arch.to_string(),
        emit_cfg: None,
    });
}

/// A dense prefill bucket must carry `2*layers + 1` wave-class segments.
///
/// PREFILL ONLY, and the qualifier is load-bearing. A DECODE program is legitimately one segment —
/// it contains no `FlashPrefill`, so every op is wave-class 8 and there is nothing to split. So is
/// a decode-only block asset (a GLM `--block` extraction, say): `n_seg == 1` there is the correct
/// answer, not the segmentation bug. Extending this expectation to those would fail a correct emit,
/// which is why the filter below keys on `kind == "prefill"` and this note exists.
#[test]
fn dense_prefill_is_wave_class_segmented() {
    let _g = emit_guard();
    let td = tempdir("segcount");
    write_qwen3_config(&td);
    emit_arch(&td, "gfx950", "MI350X", None);
    assert_eq!(
        segment_count(&td, 128),
        5,
        "2 layers => 2 dense runs + 2 flash runs + 1 output tail. Losing these tags sends the \
         whole prefill program to the 4-wave flash object, which drops every non-flash op."
    );
}

/// THE BACKSTOP FOR THE WHOLE CLASS: the segment count must survive every NVIDIA-conditioned flag.
///
/// Three flags have now been found that are correct on sm_120 and silently destructive on gfx950,
/// and two of the three destroyed segmentation specifically. Rather than add one test per flag as
/// each is discovered, this drives them TOGETHER against an AMD target and asserts the count is
/// unchanged. A fourth such flag lands here automatically the moment someone adds it to the list.
///
/// `PLOW_UNISEG` is the one that actually shipped broken: the documented Gemma recipe passed it, so
/// every asset built by following the documentation lost the wave-class split, dispatched the whole
/// prefill program on the 4-wave flash object, and returned zero logits from an 8.7 ms "prefill".
///
/// Env vars are process-global and tests share a process, so this sets and restores them around a
/// single-threaded body rather than relying on isolation.
#[test]
fn nvidia_conditioned_flags_never_change_the_amd_segment_count() {
    let _g = emit_guard();
    let flags = ["PLOW_UNISEG"];
    let saved: Vec<(&str, Option<String>)> =
        flags.iter().map(|k| (*k, std::env::var(k).ok())).collect();
    for k in &flags {
        std::env::set_var(k, "1");
    }
    let td = tempdir("seg_nvflags");
    write_qwen3_config(&td);
    // L2 placement is passed as an argument rather than an env var, so it joins here.
    emit_arch(&td, "gfx950", "MI350X", Some(amd_l2()));
    let got = segment_count(&td, 128);
    for (k, v) in saved {
        match v {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
    assert_eq!(
        got, 5,
        "an AMD packet must keep 2*layers+1 wave-class segments with every NVIDIA-conditioned flag \
         set. Losing them sends the whole prefill program to the 4-wave flash object, which drops \
         every non-flash op and returns zero logits — silently."
    );
}

/// `PLOW_L2_PLACE` REPURPOSES `seg` as an L2 domain. That is fine on sm_120, which never reads a
/// wave class, and destroys a SEGMENTED AMD dispatch. Placement must therefore leave an AMD
/// PREFILL program alone — this is the exact configuration that produced the zero-logits packets.
///
/// It is the prefill program, not the AMD target, that is protected: the gate moved from "is the
/// target AMD" to "does this program have more than one wave class" so that AMD *decode* — which
/// has no `FlashPrefill` op, so its `seg` is uniformly 0 and carries nothing — can be placed.
/// `a_single_wave_class_program_is_placed` in `packet::devbuild` covers the other side.
#[test]
fn l2_placement_never_clobbers_the_wave_class_on_amd() {
    let _g = emit_guard();
    let td = tempdir("segplace");
    write_qwen3_config(&td);
    // MI350X L2 partitioning + AMD's MEASURED round-robin workgroup->XCD map — the value
    // `run_devblob` passes for PLOW_L2_PLACE=1 on this target.
    emit_arch(&td, "gfx950", "MI350X", Some(amd_l2()));
    assert_eq!(
        segment_count(&td, 128),
        5,
        "L2 placement must not touch an AMD PREFILL program: `seg` is the wave class there, and \
         overwriting it collapses prefill to one segment"
    );
    // The same request on an NVIDIA target is honoured — placement is a real sm_120 feature, and
    // this test must not turn the fix into a blanket disable.
    let td2 = tempdir("segplace_nv");
    write_qwen3_config(&td2);
    // (32, 8) so `sms * partitions >= n_cu`; `Builder::finish` skips the BLOCK map below that
    // (it would orphan packets), which is a separate guard and not what this test is about.
    emit_arch(&td2, "sm_120a", "RTX5090", Some(nv_l2()));
    assert_eq!(
        segment_count(&td2, 128),
        1,
        "on sm_120 `seg` carries the L2 domain and the manifest reports one window — the fix must \
         not become a blanket disable of a real NVIDIA feature"
    );
}

// ===== THE AXIS NAMES, DRIVEN ACROSS BOTH EMITTER FAMILIES =====================================
//
// `PLOW_W8A16` / `PLOW_W8A8` / `PLOW_KV_FP8` were introduced so a caller could state a WEIGHT
// ENCODING without knowing which emitter would handle their model. That property is only real if
// both families are driven with them — the rename landed on one path first and the gap was found by
// a person running a command, not by a test, precisely because every test drove one family.
//
// So this drives BOTH from the same spelling and asserts what each produces. Where a family cannot
// implement an axis value it must REFUSE, not quietly hand back a different one: that is the whole
// reason the names exist.

/// A minimal MLA+MoE (Kimi-shaped) config — the other emitter family.
fn write_kimi_config(dir: &Path) {
    let cfg = r#"{
      "model_type": "kimi_k2", "vocab_size": 1000, "hidden_size": 256,
      "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 8,
      "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
      "q_lora_rank": 64, "kv_lora_rank": 32, "qk_rope_head_dim": 16,
      "qk_nope_head_dim": 48, "v_head_dim": 64,
      "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
      "moe_intermediate_size": 256, "first_k_dense_replace": 2,
      "routed_scaling_factor": 2.5, "torch_dtype": "bfloat16"
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

fn emit_block(dir: &Path, block: &str) {
    let out = dir.join("model.pkt");
    devgen::run(devgen::EmitArgs {
        dir: dir.to_path_buf(),
        ctx: 1024,
        out: out.to_str().unwrap().to_string(),
        n_cu: 256,
        tp: 1,
        block_spec: Some(block.to_string()),
        embed_cubin: None,
        embed_hsaco: None,
        rope_gen: true,
        l2_layout: None,
        gpu: "MI350X".into(),
        arch: "gfx950".into(),
        emit_cfg: None,
    });
}

fn emit_arch_ctx(dir: &Path) {
    emit_arch(dir, "gfx950", "MI350X", None);
}

fn precision(dir: &Path) -> serde_json::Value {
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("build.json")).unwrap()).unwrap();
    man["precision"].clone()
}

/// Set env vars for a scope, restoring them on Drop.
///
/// RAII rather than a wrapper closure, because these tests deliberately provoke PANICS — a
/// restore written after the call is skipped by the unwind, and the next test then runs with a
/// leaked flag. That is not hypothetical: the first draft of this file did exactly that and made
/// the second test fail with "W8A8 and W8A16 both set", which is a bug in the test, not the code.
struct EnvScope(Vec<(String, Option<String>)>);

impl EnvScope {
    fn set(kv: &[(&str, &str)]) -> Self {
        let saved = kv
            .iter()
            .map(|(k, _)| (k.to_string(), std::env::var(k).ok()))
            .collect();
        for (k, v) in kv {
            std::env::set_var(k, v);
        }
        EnvScope(saved)
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

/// `PLOW_W8A16` selects the same AXIS VALUE — fp8 weights, bf16 activations — in both families.
///
/// The targets differ, and the difference is real rather than a test artifact: "fp8 weights, bf16
/// activations" is a per-CHANNEL fp8 GEMM on the dense path, which gfx950 has no arm for (its
/// `d_gemm_fp8` is w8a8 unconditionally), and a BLOCK-fp8 GEMV on the MLA path, which it does. So
/// the dense case is emitted for sm_120a and the MLA case for gfx950. Same axis, same spelling,
/// each on a target that can realize it.
#[test]
fn w8a16_means_the_same_thing_to_both_emitter_families() {
    let _g = emit_guard();

    let dense = tempdir("axis_dense_w8a16");
    write_llama_config(&dense);
    {
        let _e = EnvScope::set(&[("PLOW_W8A16", "1"), ("PLOW_KV_FP8", "1")]);
        emit_arch(&dense, "sm_120a", "RTX5090", None);
    }
    let p = precision(&dense);
    assert_eq!(
        p["weight_enc"], "fp8",
        "dense: W8A16 selects the fp8 weight axis"
    );
    assert_eq!(p["act_enc"], "bf16", "dense: W8A16 leaves activations wide");
    assert_eq!(
        p["kv_enc"], "fp8",
        "dense: PLOW_KV_FP8 is a real alias for PLOW_FP8_KV"
    );

    let mla = tempdir("axis_mla_w8a16");
    write_kimi_config(&mla);
    {
        let _e = EnvScope::set(&[("PLOW_W8A16", "1")]);
        emit_block(&mla, "2");
    }
    let p = precision(&mla);
    assert_eq!(
        p["weight_enc"], "fp8",
        "MLA: the same spelling selects the same axis value"
    );
    assert_eq!(p["expert_enc"], "fp8blk");
    assert_eq!(
        p["act_enc"], "bf16",
        "MLA block-fp8 experts are w8a16 — x stays bf16"
    );
}

/// `PLOW_W8A8` is implementable on the dense path and NOT on the MLA path, whose expert arms are
/// w8a16 in every instantiation. The dense side must emit it; the MLA side must REFUSE rather than
/// hand back w8a16 under a flag that asked for something else — that substitution is the failure
/// the axis names were introduced to remove.
#[test]
fn w8a8_is_emitted_where_it_exists_and_refused_where_it_does_not() {
    let _g = emit_guard();

    let dense = tempdir("axis_dense_w8a8");
    write_llama_config(&dense);
    {
        let _e = EnvScope::set(&[("PLOW_W8A8", "1")]);
        emit_arch(&dense, "gfx950", "MI350X", None);
    }
    let p = precision(&dense);
    assert_eq!(p["weight_enc"], "fp8");
    assert_eq!(
        p["act_enc"], "fp8",
        "dense: W8A8 narrows the activation too — QuantFp8 is emitted"
    );

    let mla = tempdir("axis_mla_w8a8");
    write_kimi_config(&mla);
    let _e = EnvScope::set(&[("PLOW_W8A8", "1")]);
    let refused =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emit_block(&mla, "2"))).is_err();
    assert!(
        refused,
        "MLA must refuse PLOW_W8A8 rather than silently emit w8a16"
    );
}
