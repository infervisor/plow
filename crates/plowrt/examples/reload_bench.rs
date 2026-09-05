//! Model reload benchmark — measures what `PLOW_SLAB_KEEP=1` buys.
//!
//! Loads the engine, drops it (slab physical chunks go to the backend pool),
//! and loads again: the second load re-maps pooled chunks instead of paying
//! the driver's serial ~13 GiB/s page commit. This is the S1 model-switch /
//! same-box restart shape.
//!
//!   PLOW_SLAB_KEEP=1 cargo run --release -p plowrt --features cuda \
//!       --example reload_bench -- <assets-dir> [rounds, default 3]
//!
//! (The flag is read from the process env so it can also be left off for a
//! control run.)

#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::Instant;

use plowrt::device::cuda::CudaBackend;
use plowrt::exec::gpu::GpuEngine;

/// Greedy-decode a short prompt and return the generated ids — reused
/// physical chunks are NOT zeroed, so round 2+ must reproduce round 1's
/// tokens exactly to prove every slab byte is written (or write-before-read)
/// regardless of what the previous life left behind.
fn generate(e: &mut GpuEngine, ids: &[u32], max_new: usize) -> Vec<u32> {
    e.begin_slot(0, ids.len() + max_new).expect("begin_slot");
    let mut toks = Vec::new();
    let mut t = if e.has_prefill() {
        e.prefill_slot(0, ids).expect("prefill_slot")
    } else {
        e.consume_prompt(0, ids, &mut toks).expect("consume_prompt")
    };
    let stop = Arc::clone(e.stop_ids());
    let mut out = Vec::new();
    for _ in 0..max_new {
        if stop.contains(&t) {
            break;
        }
        out.push(t);
        e.step_slots(&[(0, t)], &mut toks).expect("step");
        t = toks[0];
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("usage: reload_bench <assets-dir>"));
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let ckpt = dir.join("checkpoint");
    use plowrt::text::tokenizer::Tokenize as _;
    let tok = plowrt::text::tokenizer::load_tokenizer(&dir);
    let prompt = "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n\
                  <|turn>model\n";
    let ids = tok.encode(prompt);

    let be = Arc::new(CudaBackend::new(0).expect("cuda"));
    let mut first: Option<Vec<u32>> = None;
    for round in 1..=rounds {
        let t = Instant::now();
        let mut engine = GpuEngine::load(Arc::clone(&be), &dir, &ckpt).expect("load");
        let load_s = t.elapsed().as_secs_f64();
        let out = generate(&mut engine, &ids, 16);
        match &first {
            None => first = Some(out.clone()),
            Some(f) => assert_eq!(
                f, &out,
                "reused-chunk round diverged from fresh-commit round"
            ),
        }
        let t = Instant::now();
        drop(engine);
        let drop_s = t.elapsed().as_secs_f64();
        println!(
            "round {round}: load {load_s:.3} s, drop {drop_s:.3} s, reply {:?}",
            tok.decode(&out)
        );
    }
}
