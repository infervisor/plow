//! Stock packet defaults through the full mux: concurrent ragged prompts, output
//! limits, cancellation and greedy controls. Requires PLOW_GPU_TEST=1 and
//! PLOW_GPU_ASSETS pointing at a full packed-prefill bundle with a decode ladder.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::manager::ModelManager;
use plowrt::serve::mux::MuxConfig;
use plowrt::serve::stream::{self, StreamChunk};
use plowrt::serve::{AppState, GenParams};

fn submit(
    state: &AppState,
    slug: &str,
    prompt_ids: Vec<u32>,
    gen: GenParams,
) -> stream::ChunkReceiver {
    let (tx, rx) = stream::channel();
    state
        .mux(slug)
        .expect("mux")
        .submit(plowrt::serve::mux::Job {
            prompt_ids,
            gen,
            arrived: std::time::Instant::now(),
            respond: tx,
        })
        .map_err(|_| ())
        .expect("submit");
    rx
}

async fn reply(
    mut rx: stream::ChunkReceiver,
    prompt_len: usize,
    limit: usize,
    exact: bool,
) -> (Vec<u32>, String) {
    tokio::time::timeout(Duration::from_secs(300), async move {
        let mut ids = Vec::new();
        let mut text = String::new();
        while let Some(chunk) = rx.recv().await {
            match chunk {
                StreamChunk::Token { id, text: delta } => {
                    ids.push(id);
                    text.push_str(&delta);
                }
                StreamChunk::Done { usage, reason, .. } => {
                    assert_eq!(usage.prompt_tokens, prompt_len);
                    let hidden_stop = usize::from(reason.as_str() == "stop");
                    assert_eq!(usage.completion_tokens, ids.len() + hidden_stop);
                    assert!(ids.len() <= limit);
                    if exact {
                        assert_eq!(ids.len(), limit);
                        assert_eq!(reason.as_str(), "length");
                    }
                    return (ids, text);
                }
                StreamChunk::Err(e) => panic!("stream error: {e}"),
            }
        }
        panic!("stream closed without Done");
    })
    .await
    .expect("request timed out")
}

fn greedy(limit: usize) -> GenParams {
    let mut gen = GenParams {
        max_tokens: limit,
        ignore_eos: true,
        ..GenParams::default()
    };
    gen.params.temperature = 0.0;
    gen
}

#[tokio::test]
async fn production_defaults_serve_concurrent() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + assets)");
        return;
    }
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());
    let _ = tracing_subscriber::fmt()
        .with_env_filter("plowrt=info")
        .try_init();
    let raw = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).unwrap();
    let rungs = blob.decode_rungs();
    assert!(rungs.len() >= 3 && rungs[0] == 1);
    assert!(rungs.windows(2).all(|r| r[1] == 2 * r[0]));
    assert!(blob
        .reserved_metadata(&raw, plow_asset::packed_prefill::SECTION)
        .unwrap()
        .is_some());

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let registry = Registry::new();
    let slug = registry.load(assets.clone(), None).expect("load");
    let backend: Arc<dyn Backend> = Arc::clone(&be) as Arc<dyn Backend>;
    let execset = Arc::new(ExecutorSet::bringup(backend).expect("execset"));
    let state = Arc::new(AppState::with_trace(registry, execset, false));

    let mgr = Arc::new(
        ModelManager::new(
            Arc::clone(&be),
            &state,
            MuxConfig::default(),
            vec![(slug.clone(), assets.clone(), assets.join("checkpoint"))],
            None,
        )
        .expect("manager"),
    );
    let static_bytes = plowrt::serve::manager::BlobPlan::from_dir(&assets)
        .unwrap()
        .tensor_total();
    assert!(
        mgr.plan(&slug).unwrap().tensor_total() < static_bytes,
        "serving admission must budget resident live KV, not the full virtual context"
    );
    mgr.ensure_resident(&slug).await.expect("ensure_resident");

    let prompt = "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n\
                  <|turn>model\n<|channel>thought\n<channel|>";
    let ids = state
        .registry
        .get(&slug)
        .expect("bundle")
        .tokenizer()
        .encode(prompt);
    let mut gen = greedy(24);
    gen.ignore_eos = false;
    let (_, text) = reply(
        submit(&state, &slug, ids.clone(), gen),
        ids.len(),
        24,
        false,
    )
    .await;
    assert!(text.contains("Paris"), "coherence: {text:?}");

    let tokenizer = state.registry.get(&slug).unwrap().tokenizer().clone();
    let prompts = [
        ids.clone(),
        tokenizer.encode("The first ten prime numbers are"),
        tokenizer.encode(&format!(
            "{}The capital of France is",
            "A short sentence. ".repeat(260)
        )),
        tokenizer.encode("The largest planet in the solar system is"),
    ];
    let limits = [1, 3, 9, 17];
    let mut reference = Vec::new();
    for (prompt, limit) in prompts.iter().zip(limits) {
        reference.push(
            reply(
                submit(&state, &slug, prompt.clone(), greedy(limit)),
                prompt.len(),
                limit,
                true,
            )
            .await
            .0,
        );
    }
    for round in 0..2 {
        let pending: Vec<_> = (0..16)
            .map(|i| {
                let case = (i + round) % prompts.len();
                (
                    case,
                    submit(&state, &slug, prompts[case].clone(), greedy(limits[case])),
                )
            })
            .collect();
        for (case, rx) in pending {
            let (tokens, _) = reply(rx, prompts[case].len(), limits[case], true).await;
            assert_eq!(
                tokens, reference[case],
                "round {round}, case {case}: cross-request token parity"
            );
        }
    }

    // A greedy bias must survive the multistep eligibility gate at every token.
    let forced = tokenizer.encode("Paris")[0];
    let mut biased = greedy(9);
    biased.params.logit_bias.push((forced, 10000.0));
    let (tokens, _) = reply(
        submit(&state, &slug, ids.clone(), biased),
        ids.len(),
        9,
        true,
    )
    .await;
    assert_eq!(tokens, vec![forced; 9]);

    let mut cancelled = submit(&state, &slug, ids.clone(), greedy(64));
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(60), cancelled.recv())
            .await
            .unwrap(),
        Some(StreamChunk::Token { .. })
    ));
    drop(cancelled);
    let (tokens, _) = reply(
        submit(&state, &slug, prompts[3].clone(), greedy(limits[3])),
        prompts[3].len(),
        limits[3],
        true,
    )
    .await;
    assert_eq!(tokens, reference[3], "slot reuse after cancellation");
    eprintln!("production serving defaults: concurrent ragged requests, tails, bias, cancellation and slot reuse passed");
}
