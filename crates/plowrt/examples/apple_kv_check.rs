//! FP8 KV serving check: chunked prefill, two slots, batched decode, and slot reuse.
//! `apple_kv_check <two-slot model.pkt> <checkpoint-dir>`

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use plowrt::exec::apple::MetalEngine;
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let blob = PathBuf::from(args.next().expect("model.pkt"));
    let ckpt = PathBuf::from(args.next().expect("checkpoint-dir"));
    let tok = load_tokenizer(&ckpt);
    let mut eng = MetalEngine::load(&blob, &ckpt).expect("Metal load");
    #[cfg(feature = "ane")]
    let channel_enabled = std::env::var("PLOW_ANE_MLP").as_deref() == Ok("1");
    #[cfg(feature = "ane")]
    if channel_enabled {
        assert!(eng.channel.is_some(), "channel offload must load");
    }
    assert_eq!(eng.model.batch, 2);
    let dp = eng.model.dec_ix;
    assert!(eng.insts_host(dp).iter().any(|d| d.op == 38));
    let prompts: Vec<Vec<u32>> = [
        "A computer executes instructions and stores intermediate results in memory. ".repeat(23),
        "The river flows through the valley, past farms and small villages. ".repeat(17),
    ]
    .iter()
    .map(|s| tok.encode_with_special_tokens(s, true))
    .collect();
    for p in &prompts {
        assert!(p.len() > 128 && p.len() + 8 < eng.max_ctx());
    }
    let mut expected = Vec::new();
    for p in &prompts {
        let first = eng.prefill_slot(0, p).expect("isolated prefill");
        let mut tokens = vec![first];
        for step in 0..7 {
            let pos = p.len() as u32 + step;
            let next = eng
                .decode_step_batched_at(&[pos, 0], &[pos + 1, 1], &[*tokens.last().unwrap(), 0], dp)
                .expect("isolated decode");
            tokens.push(next[0]);
        }
        expected.push(tokens);
    }
    for order in [[0, 1], [1, 0]] {
        let mut ids = [0; 2];
        let mut pos = [0; 2];
        for slot in 0..2 {
            let p = order[slot];
            ids[slot] = eng.prefill_slot(slot, &prompts[p]).expect("slot prefill");
            pos[slot] = prompts[p].len() as u32;
            assert_eq!(ids[slot], expected[p][0], "slot {slot} prefill");
        }
        for step in 1..8 {
            let next = eng
                .decode_step_batched_at(&pos, &[pos[0] + 1, pos[1] + 1], &ids, dp)
                .expect("batched decode");
            for slot in 0..2 {
                assert_eq!(
                    next[slot], expected[order[slot]][step],
                    "slot {slot} step {step}"
                );
                ids[slot] = next[slot];
                pos[slot] += 1;
            }
        }
    }
    #[cfg(feature = "ane")]
    if channel_enabled {
        let stats = &eng.channel.as_ref().unwrap().stats;
        assert!(stats.mlps > 0 && !stats.disabled && stats.fallbacks == 0);
        println!("channel MLPs executed: {}", stats.mlps);
    }
    println!(
        "FP8 KV: chunked prompts {:?}, two slots, 8 tokens and slot reuse passed",
        prompts.iter().map(Vec::len).collect::<Vec<_>>()
    );
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("build with --features metal on macOS");
    std::process::exit(1);
}
