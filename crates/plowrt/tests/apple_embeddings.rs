#![cfg(all(feature = "metal", target_os = "macos"))]

use plowrt::exec::apple::MetalEngine;
use plowrt::text::tokenizer::load_tokenizer;
use std::path::PathBuf;

#[test]
#[ignore = "requires PLOW_TEST_BLOB and PLOW_TEST_CHECKPOINT for a compiled Llama/Qwen model"]
fn supplied_embeddings_match_text_and_restore_normal_prefill() {
    let blob = PathBuf::from(std::env::var("PLOW_TEST_BLOB").unwrap());
    let checkpoint = PathBuf::from(std::env::var("PLOW_TEST_CHECKPOINT").unwrap());
    let tokenizer = load_tokenizer(&checkpoint);
    assert!(!tokenizer.is_byte_fallback());
    let prompt = tokenizer.encode("The capital of France is");
    let mut engine = MetalEngine::load(&blob, &checkpoint).unwrap();
    let hidden = engine.model.blob.progs[0]
        .insts
        .iter()
        .find(|d| d.op == packet::dev::DevOp::Embed as u16)
        .unwrap()
        .i[1] as usize;
    let table = engine
        .model
        .names
        .iter()
        .position(|n| n.ends_with("embed_tokens.weight"))
        .unwrap();
    let table = engine.tensor_bytes(table);
    let embeddings: Vec<u16> = prompt
        .iter()
        .flat_map(|&id| {
            table[id as usize * hidden * 2..(id as usize + 1) * hidden * 2]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
        })
        .collect();
    let logits = engine.model.wk.logits.unwrap();
    let first = engine.prefill(&prompt).unwrap();
    let expected_logits = engine.tensor_bytes(logits).to_vec();
    let mut expected = Vec::new();
    for step in 0..3 {
        expected.push(
            engine
                .decode_step(
                    (prompt.len() + step) as u32,
                    (prompt.len() + step + 1) as u32,
                )
                .unwrap(),
        );
    }
    assert_eq!(
        engine.prefill_embeddings(&prompt, &embeddings).unwrap(),
        first
    );
    assert_eq!(engine.tensor_bytes(logits), expected_logits);
    for (step, token) in expected.into_iter().enumerate() {
        assert_eq!(
            engine
                .decode_step(
                    (prompt.len() + step) as u32,
                    (prompt.len() + step + 1) as u32
                )
                .unwrap(),
            token
        );
    }
    let changed = vec![0u16; embeddings.len()];
    engine.prefill_embeddings(&prompt, &changed).unwrap();
    assert_ne!(
        engine.tensor_bytes(logits),
        expected_logits,
        "supplied rows must replace token embeddings"
    );
    assert_eq!(engine.prefill(&prompt).unwrap(), first);
    assert_eq!(
        engine.tensor_bytes(logits),
        expected_logits,
        "ordinary prefill must restore Embed"
    );
    assert!(engine
        .prefill_embeddings(&prompt, &embeddings[..embeddings.len() - 1])
        .is_err());
    let vocab = engine.tensor_bytes(table_handle(&engine)).len() / (hidden * 2);
    let max_ctx = engine
        .model
        .wk
        .pos
        .map(|h| engine.tensor_bytes(h).len() / 4)
        .unwrap();
    let dp = engine.model.decode_prog_for(1);
    for id in [vocab as u32, u32::MAX] {
        assert!(engine.prefill(&[id]).is_err());
        assert!(engine.set_token(id).is_err());
        assert!(engine.prepare_decode(0, 1, id).is_err());
        assert!(engine
            .decode_step_batched_at(&[0], &[1], &[id], dp)
            .is_err());
    }
    for (pos, kvlen) in [(max_ctx as u32, 1), (0, 0), (0, max_ctx as u32 + 1)] {
        assert!(engine.prepare_decode(pos, kvlen, prompt[0]).is_err());
    }
    assert_eq!(engine.prefill(&prompt).unwrap(), first);
}

fn table_handle(engine: &MetalEngine) -> usize {
    engine
        .model
        .names
        .iter()
        .position(|n| n.ends_with("embed_tokens.weight"))
        .unwrap()
}
