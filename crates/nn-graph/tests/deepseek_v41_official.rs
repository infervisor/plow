#![recursion_limit = "512"]
//! `DeepSeekV41Config`'s layer roles against the released checkpoint's tensors.
//!
//! `config.json` does not say which layers carry which optional tensor group;
//! that has to be DERIVED (`kv_source_layer_ids` -> a compressor, ratio > 1 ->
//! a gate, and so on). Every one of those derivations is a place to misread
//! V4.1 as V4, and a misread binds the wrong tensors instead of failing.
//!
//! So the fixture below is the ground truth, extracted once from
//! `deepseek-ai/DeepSeek-V4.1-Flash`'s `model.safetensors.index.json`
//! (96 085 tensors, 48 shards): for each optional tensor group, the layers
//! that actually carry it. The test asserts the config's predicates reproduce
//! those lists exactly. The index itself is 7 MB and not committed; these
//! lists are its load-bearing content.

use nn_graph::models::config::{ModelConfig, V41Attn};
use nn_graph::models::config::DeepSeekV41Config;

// ---- observed in model.safetensors.index.json ----

/// `layers.{L}.attn.compressor.wkv.weight` — 4 of 40.
const COMPRESSOR: &[u32] = &[2, 8, 14, 20];
/// `layers.{L}.attn.compressor.wgate.weight` — 3, NOT 4. Layer 20 runs at
/// ratio 1, which is a plain projection with no softmax gate.
const COMPRESSOR_GATE: &[u32] = &[2, 8, 14];
/// `layers.{L}.attn.compressor.norm.weight` — every compressor has one.
const COMPRESSOR_NORM: &[u32] = &[2, 8, 14, 20];
/// `layers.{L}.attn.indexer.wq_b.weight` and `.weights_proj.weight` — 8.
const INDEXER: &[u32] = &[2, 8, 14, 20, 24, 28, 32, 36];
/// `layers.{L}.attn.indexer.wk.weight` and `.k_norm.weight` — 4. Index keys
/// are derived from the compressor latent, so only a KV source owns them.
const INDEXER_KEYS: &[u32] = &[2, 8, 14, 20];
/// `layers.{L}.engram.embed.weight` — 2.
const ENGRAM: &[u32] = &[1, 14];
/// `layers.{L}.attn.wq_a.weight`, `.hc_attn_fn`, `.ffn.gate.bias_vl` — all 40.
const ALL_LAYERS: u32 = 40;
/// `layers.{L}.ffn.experts.{E}.w1.weight` — 384 per layer, on all 40.
const EXPERTS_PER_LAYER: u32 = 384;
/// `mtp.{M}.ffn.experts.{E}.w1.weight` — 128 per block, on all 3.
const MTP_EXPERTS_PER_BLOCK: u32 = 128;

/// The released `text_config`, with the outer `dtype` and `quantization_config`
/// folded in — what the text-generation frontend hands to the parser.
fn official() -> DeepSeekV41Config {
    let doc = serde_json::json!({
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn", "factor": 16,
            "beta_fast": 32, "beta_slow": 1,
            "original_max_position_embeddings": 65536
        },
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [
            0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0
        ],
        "compress_rope_theta": 160000,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168i64, 384016682i64],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4"
        }
    });
    match ModelConfig::from_json(&doc.to_string()).expect("official V4.1 config parses") {
        ModelConfig::DeepSeekV41(c) => c,
        other => panic!("official V4.1 config parsed as {other:?}"),
    }
}

fn layers_where(cfg: &DeepSeekV41Config, f: impl Fn(&DeepSeekV41Config, u32) -> bool) -> Vec<u32> {
    (0..cfg.num_hidden_layers).filter(|l| f(cfg, *l)).collect()
}

/// Only `kv_source_layer_ids` run a compressor — 4 layers, not the 38 a V4
/// reading of `compress_ratios` would predict.
#[test]
fn compressor_layers_match_the_checkpoint() {
    let cfg = official();
    assert_eq!(layers_where(&cfg, |c, l| c.is_kv_source(l)), COMPRESSOR);
    assert_eq!(COMPRESSOR_NORM, COMPRESSOR, "every compressor carries a norm");

    // The V4 reading, stated so the difference is visible rather than implied.
    let v4_reading = layers_where(&cfg, |c, l| c.attn_kind(l) != V41Attn::Window);
    assert_eq!(v4_reading.len(), 38);
    assert_ne!(v4_reading, COMPRESSOR);
}

/// 4 compressors, 3 gates. The ratio-1 compressor has no `wgate`, and this is
/// the assertion that would fail first if ratio 1 were treated as a pooling.
#[test]
fn only_pooling_compressors_carry_a_gate() {
    let cfg = official();
    assert_eq!(layers_where(&cfg, |c, l| c.compressor_has_gate(l)), COMPRESSOR_GATE);
    assert_eq!(COMPRESSOR_GATE.len(), COMPRESSOR.len() - 1);
    assert_eq!(cfg.attn_kind(20), V41Attn::Compressed { ratio: 1 });
}

/// 8 indexers, but only the 4 KV sources own index keys.
#[test]
fn indexer_layers_match_the_checkpoint() {
    let cfg = official();
    assert_eq!(layers_where(&cfg, |c, l| c.is_index_source(l)), INDEXER);
    assert_eq!(
        layers_where(&cfg, |c, l| c.is_index_source(l) && c.is_kv_source(l)),
        INDEXER_KEYS
    );
    // 32 of 40 layers carry no indexer at all: they reuse the published
    // selection. An emitter that runs one per layer does 5x the work.
    assert_eq!(cfg.num_hidden_layers as usize - INDEXER.len(), 32);
}

#[test]
fn engram_layers_match_the_checkpoint() {
    let cfg = official();
    assert_eq!(layers_where(&cfg, |c, l| c.engram_rows(l).is_some()), ENGRAM);
    assert_eq!(cfg.engram_rows(1), Some(384_006_168));
    assert_eq!(cfg.engram_rows(14), Some(384_016_682));
}

/// The groups every layer carries, against the ones only some do.
#[test]
fn per_layer_groups_are_complete() {
    let cfg = official();
    assert_eq!(cfg.num_hidden_layers, ALL_LAYERS);
    assert_eq!(cfg.n_routed_experts, EXPERTS_PER_LAYER);
    assert_eq!(cfg.dspark_n_routed_experts, MTP_EXPERTS_PER_BLOCK);
    assert_eq!(cfg.dspark_blocks(), 3);
    // 40 x 384 backbone + 3 x 128 DSpark, each with w1/w2/w3.
    let expert_tensors = (ALL_LAYERS * EXPERTS_PER_LAYER + 3 * MTP_EXPERTS_PER_BLOCK) * 3;
    assert_eq!(expert_tensors, 47_232);
}

/// The emit still refuses — this test pins the config, not a graph.
#[test]
fn graph_build_refuses_with_a_checklist() {
    let cfg = official();
    let err = nn_graph::models::build_graph(
        &ModelConfig::DeepSeekV41(cfg),
        &nn_graph::models::ShapeBucket::default(),
    )
    .expect_err("V4.1 has no graph builder yet");
    let text = err.to_string();
    assert!(text.contains("CSA2 cache SHARING"), "{text}");
    assert!(text.contains("graph builder"), "{text}");
}

/// Engram's hash tables, against the reference's own arithmetic.
///
/// The primes are DERIVED here (a trial-division walk replacing `sympy.isprime`) and the
/// derivation proves itself: each layer's bucket ranges are laid end to end, so they must sum
/// to that layer's `engram_num_embeddings`. `validate()` enforces it; this pins the individual
/// values too, so a walk that drifts in a way the sum happens to survive still fails.
///
/// The multipliers cannot be derived -- they come from numpy's PCG64 -- so they are constants,
/// and this checks the ones extracted from the released checkpoint.
#[test]
fn engram_hash_tables_match_the_reference() {
    let cfg = official();
    let t = cfg.engram_hash_tables().expect("released checkpoint's engram shape");

    assert_eq!(t.multipliers.len(), 2);
    assert_eq!(
        t.multipliers[0],
        vec![76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391]
    );
    assert_eq!(
        t.multipliers[1],
        vec![67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591]
    );
    // Odd by construction: the reference draws `v` and stores `2v + 1`.
    for row in &t.multipliers {
        for m in row {
            assert_eq!(m % 2, 1, "multiplier {m} must be odd");
        }
    }

    // (max_ngram_size - 1) * n_heads = 3 * 8 columns per layer.
    assert_eq!(t.primes[0].len(), 24);
    assert_eq!(t.primes[1].len(), 24);
    // The walk is GLOBAL: layer 14's ranges start above every one of layer 1's.
    assert_eq!(t.primes[0][0], 16_000_057);
    assert_eq!(t.primes[0][23], 16_000_463);
    assert_eq!(t.primes[1][0], 16_000_477);
    assert_eq!(t.primes[1][23], 16_000_889);
    let mut all: Vec<i64> = t.primes.iter().flatten().copied().collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "every bucket range must own a DISTINCT prime");

    // Offsets are the exclusive prefix sum, and the last range ends exactly at the table's end.
    for (primes, (offsets, rows)) in
        t.primes.iter().zip(t.offsets.iter().zip(cfg.engram_num_embeddings.iter()))
    {
        assert_eq!(offsets[0], 0);
        for i in 1..offsets.len() {
            assert_eq!(offsets[i], offsets[i - 1] + primes[i - 1]);
        }
        assert_eq!(offsets[offsets.len() - 1] + primes[primes.len() - 1], *rows);
    }
}
