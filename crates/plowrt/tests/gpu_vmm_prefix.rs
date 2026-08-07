//! VMM prefix-sharing GPU gates (plans/rtx-09-prefix-headmajor.md V1):
//!
//! 1. `shared_prefix_token_identity_and_dedup` — two sequences sharing a
//!    prefix under `PLOW_VMM_PREFIX=1` produce greedy tokens IDENTICAL to two
//!    independent sequences on the default cudaMalloc path (no cross-sequence
//!    bleed, byte-exact prefix reads), the sharer does NOT re-create the
//!    prefix's physical blocks (the HBM dedup), and decode TPOT stays within
//!    noise of the default path.
//! 2. `attach_latency_vs_copy_baseline` — pool-level 31B@128k-class attach
//!    timings (map + setaccess per sharing block) vs the D2D copy of the same
//!    bytes, at prefix 4k/32k/128k and block 2/16/64 MiB.
//! 3. `vmm_leak_cycle` — load-serve(share)-unload returns VRAM to baseline,
//!    twice (the gpu_lifecycle pattern with the VMM pool in the loop).
//!
//! Gated on `PLOW_GPU_TEST=1` + real assets (`PLOW_GPU_ASSETS`, default
//! /root/gpu-assets-b4/b4).
//!
//! Tests here mutate process env (`PLOW_VMM_PREFIX`, `PLOW_VMM_BLOCK_MIB`),
//! which the engine reads live. Every test takes `common::env_guard()` so they
//! cannot overlap; this file previously asked for `--test-threads=1` in a
//! comment, which no script or CI path passes.

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::gpu::GpuEngine;
use plowrt::memory::vmm::{VmmGeometry, VmmKv, VmmOps};
use plowrt::text::tokenizer::{load_tokenizer, Tokenize};

const MIB: u64 = 1 << 20;

fn gated() -> Option<PathBuf> {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + model assets)");
        return None;
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "plowrt=info".into()))
        .try_init();
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());
    Some(assets)
}

fn used(be: &CudaBackend) -> u64 {
    let (free, total) = be.mem_info().expect("cuMemGetInfo");
    total - free
}

/// A deterministic long prompt: repeated text until >= `min_tokens` ids.
fn long_ids(tok: &Arc<dyn Tokenize>, min_tokens: usize, seed_text: &str) -> Vec<u32> {
    let mut text = String::from("<bos>");
    while tok.encode(&text).len() < min_tokens {
        text.push_str(seed_text);
    }
    let mut ids = tok.encode(&text);
    ids.truncate(min_tokens);
    ids
}

/// Prefill `ids` into slot `b` and greedily decode `max_new` tokens
/// (no stop-set early exit — fixed length keeps the comparison exact).
/// Returns (tokens, decode seconds).
fn serve(e: &mut GpuEngine, b: usize, ids: &[u32], max_new: usize) -> (Vec<u32>, f64) {
    e.begin_slot(b, ids.len() + max_new).expect("begin_slot");
    let mut t = e.prefill_slot(b, ids).expect("prefill");
    let mut out = vec![t];
    let mut toks = Vec::new();
    let t0 = Instant::now();
    for _ in 1..max_new {
        e.step_slots(&[(b, t)], &mut toks).expect("step");
        t = toks[0];
        out.push(t);
    }
    (out, t0.elapsed().as_secs_f64())
}

/// `serve` split at the prefill/decode boundary, with a pool-quiesce pause
/// after decode so the async pre-mapper's lookahead block is settled before
/// the caller reads `vmm_stats` (block counts stay deterministic).
fn serve_measured(e: &mut GpuEngine, b: usize, ids: &[u32], max_new: usize) -> (Vec<u32>, f64) {
    let r = serve(e, b, ids, max_new);
    std::thread::sleep(std::time::Duration::from_millis(300));
    r
}

/// Block acquisitions: fresh driver creates PLUS reuse-pool draws. The dedup
/// ledger below counts how many blocks a sequence NEEDED; whether one came
/// from `cuMemCreate` or the engine's block pool (`VmmKv::enable_block_pool`,
/// on by default) is an optimization the ledger must be blind to.
fn created(s: &plowrt::memory::vmm::VmmStats) -> u64 {
    s.blocks_created + s.blocks_reused
}

#[test]
fn shared_prefix_token_identity_and_dedup() {
    let Some(assets) = gated() else { return };
    let _env = common::env_guard();
    let ckpt = assets.join("checkpoint");
    let tok = load_tokenizer(&assets);
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));

    // 2 MiB sharing blocks (2048 tokens) so a ctx-8k asset can share a
    // multi-block prefix; the dedup mechanism is block-size-independent.
    std::env::set_var("PLOW_VMM_BLOCK_MIB", "2");

    // prefix: 2 sharing blocks + change; two different suffixes; one long-ctx
    // sequence (same seed text => it attaches the same published boundary).
    let prefix = long_ids(&tok, 4200, "The quick brown fox jumps over the lazy dog. ");
    let long = long_ids(&tok, 7400, "The quick brown fox jumps over the lazy dog. ");
    let sufa = tok.encode("Now summarize the story in one word.");
    let sufb = tok.encode("Now count the animals mentioned above.");
    let (mut a, mut b) = (prefix.clone(), prefix);
    a.extend(&sufa);
    b.extend(&sufb);
    let max_new = 32;

    // ---- baseline: independent sequences, default cudaMalloc path ----
    std::env::remove_var("PLOW_VMM_PREFIX");
    let (toks_a0, toks_b0, toks_l0, tpot0, tpot0_long) = {
        let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("load (default)");
        let (ta, _) = serve(&mut e, 0, &a, max_new);
        let (tb, dt) = serve(&mut e, 1, &b, max_new);
        let (tl, dtl) = serve(&mut e, 2, &long, max_new);
        (
            ta,
            tb,
            tl,
            dt / (max_new - 1) as f64,
            dtl / (max_new - 1) as f64,
        )
    };

    // ---- VMM: A publishes, B and the long sequence attach ----
    std::env::set_var("PLOW_VMM_PREFIX", "1");
    let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("load (vmm)");
    let s0 = e.vmm_stats().expect("VMM did not come up on these assets");

    let (toks_a1, _) = serve_measured(&mut e, 0, &a, max_new);
    let s_after_a = e.vmm_stats().unwrap();
    let used_after_a = used(&be);

    let (toks_b1, dt_b) = serve_measured(&mut e, 1, &b, max_new);
    let tpot1 = dt_b / (max_new - 1) as f64;
    let s_after_b = e.vmm_stats().unwrap();
    let used_after_b = used(&be);

    let (toks_l1, dt_l) = serve_measured(&mut e, 2, &long, max_new);
    let tpot1_long = dt_l / (max_new - 1) as f64;
    std::env::remove_var("PLOW_VMM_PREFIX");
    std::env::remove_var("PLOW_VMM_BLOCK_MIB");

    // THE correctness bar: byte-exact prefix reads => identical greedy paths.
    assert_eq!(toks_a1, toks_a0, "VMM path changed sequence A's tokens");
    assert_eq!(
        toks_b1, toks_b0,
        "shared-prefix B diverged from independent B"
    );
    assert_eq!(
        toks_l1, toks_l0,
        "long-ctx sharer diverged from independent"
    );

    // The dedup ledger. Per track (full_layer × {K,V}) both A and B walk the
    // same block schedule (row-0 remap at begin, prefill blocks, one
    // pre-mapper lookahead); B additionally displaces its row-0 blocks at
    // attach (+1/track) but does NOT create the shared span. So exactly:
    //   created_B + shared == created_A + tracks
    let shared = s_after_b.blocks_shared_mapped - s_after_a.blocks_shared_mapped;
    let created_a = created(&s_after_a) - created(&s0);
    let created_b = created(&s_after_b) - created(&s_after_a);
    let tracks = 8 * 2; // 12B: 8 full layers × {K,V} × 1 kv head
    assert!(shared > 0, "B did not attach any shared block");
    assert!(
        created_b < created_a,
        "B created as many blocks as the non-sharing A ({created_b} vs {created_a}) — no dedup"
    );
    assert_eq!(
        created_b + shared,
        created_a + tracks,
        "dedup ledger mismatch (created_a={created_a} created_b={created_b} shared={shared})"
    );
    let dedup_mib = shared * (2 * MIB) / MIB;
    eprintln!(
        "dedup: B shared {shared} blocks ({dedup_mib} MiB HBM not re-created; A created \
         {created_a}, B {created_b}); VRAM after A {} MiB, after B {} MiB (+{} MiB)",
        used_after_a / MIB,
        used_after_b / MIB,
        (used_after_b - used_after_a) / MIB
    );

    // TPOT neutrality (gate d): within noise of the cudaMalloc path.
    for (what, t0v, t1v) in [
        ("~4.2k ctx", tpot0, tpot1),
        ("~7.4k ctx", tpot0_long, tpot1_long),
    ] {
        let delta = (t1v - t0v) / t0v * 100.0;
        eprintln!(
            "TPOT @{what}: default {:.3} ms, vmm {:.3} ms ({delta:+.1}%)",
            t0v * 1e3,
            t1v * 1e3
        );
        assert!(
            delta < 15.0,
            "VMM decode @{what} regressed {delta:+.1}% vs cudaMalloc"
        );
    }
}

/// Pool-level remap cycle: map, release the sequence (unmap + handle
/// release), map again at the same VA — the begin_slot path.
#[test]
fn remap_after_release_cycle() {
    let Some(_assets) = gated() else { return };
    let _env = common::env_guard();
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let ops: Arc<dyn VmmOps> = Arc::clone(&be) as Arc<dyn VmmOps>;
    let geo = VmmGeometry {
        full_layers: vec![0, 1],
        kvh_full: 1,
        hd_full: 512,
        slide_layers: vec![],
        kvh_slide: 0,
        hd_slide: 0,
        window: 0,
        elem: 2,
        // Sliding-layer element width. These cases have no sliding
        // layers, so it is inert here; 2 matches `elem` as everywhere
        // outside the mixed fp8-KV mode.
        elem_slide: 2,
        max_ctx: 8192,
        batch: 4,
    };
    let kv = VmmKv::new(ops, geo, 2 * MIB, 0).expect("pool");
    for cycle in 0..3 {
        for b in 0..4 {
            kv.ensure_rows(b, 1)
                .unwrap_or_else(|e| panic!("cycle {cycle} seq {b} row1: {e}"));
        }
        kv.ensure_rows(0, 4096)
            .unwrap_or_else(|e| panic!("cycle {cycle} grow: {e}"));
        for b in 0..4 {
            kv.begin_seq(b);
        }
    }
    assert_eq!(kv.stats().blocks_live, 0);
}

#[test]
fn attach_latency_vs_copy_baseline() {
    let Some(_assets) = gated() else { return };
    let _env = common::env_guard();
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let ops: Arc<dyn VmmOps> = Arc::clone(&be) as Arc<dyn VmmOps>;

    // 31B-class full-layer geometry: 10 layers × 4 kv heads × {K,V},
    // 1 KiB rows, max_ctx 128k => 10 GiB per fully-built sequence.
    let geo = |max_ctx: u32| VmmGeometry {
        full_layers: (0..10).collect(),
        kvh_full: 4,
        hd_full: 512,
        slide_layers: vec![],
        kvh_slide: 0,
        hd_slide: 0,
        window: 0,
        elem: 2,
        // Sliding-layer element width. These cases have no sliding
        // layers, so it is inert here; 2 matches `elem` as everywhere
        // outside the mixed fp8-KV mode.
        elem_slide: 2,
        max_ctx,
        batch: 2,
    };
    let track_bytes_per_row: u64 = 10 * 2 * 4 * 1024; // 80 KiB/token (plan §3)

    // D2D copy baseline at each prefix size.
    let copy_ms = |rows: u64| -> f64 {
        let bytes = rows * track_bytes_per_row;
        let src = Backend::alloc(be.as_ref(), 0, bytes).expect("src");
        let dst = Backend::alloc(be.as_ref(), 0, bytes).expect("dst");
        be.memcpy_dtod(dst.base, src.base, bytes).expect("warm");
        be.synchronize().unwrap();
        let t0 = Instant::now();
        be.memcpy_dtod(dst.base, src.base, bytes).expect("copy");
        be.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3
    };

    eprintln!(
        "prefix rows | block MiB | owner-build ms | ATTACH ms | detach(begin_seq) ms | D2D copy ms"
    );
    for &rows in &[4096u32, 32768, 131072] {
        let cp = copy_ms(rows as u64);
        for &blk_mib in &[2u64, 16, 64] {
            let block_rows = (blk_mib * MIB / 1024) as u32;
            if rows < 2 * block_rows {
                continue; // attach needs >= 1 whole block below the limit
            }
            let kv = VmmKv::new(Arc::clone(&ops), geo(131072), blk_mib * MIB, 0).expect("pool");
            // The "prompt" only feeds the hash chain — content is irrelevant.
            // The borrower's prompt is one token longer so the whole `rows`
            // span sits below its attach limit (the tail token is always
            // recomputed, never shared).
            let prompt: Vec<u32> = (0..rows).collect();
            let borrower: Vec<u32> = (0..=rows).collect();
            assert!(kv.try_attach(0, &prompt).unwrap().is_none());
            let t0 = Instant::now();
            kv.ensure_rows(0, rows).unwrap();
            let owner_ms = t0.elapsed().as_secs_f64() * 1e3;
            // The TOKENS, not a row count: `publish` hashes the block chain out
            // of them, which is what `try_attach` below matches the borrower's
            // prefix against. Passing a length here compiled while the parameter
            // was a `usize` and stopped when it became `&[u32]` (8a38331).
            kv.publish(0, &prompt, 4, |_| Ok(())).unwrap();

            let t0 = Instant::now();
            let at = kv.try_attach(1, &borrower).unwrap().expect("attach");
            let attach_ms = t0.elapsed().as_secs_f64() * 1e3;
            let shared_rows = at.rows;

            let t0 = Instant::now();
            kv.begin_seq(1);
            let detach_ms = t0.elapsed().as_secs_f64() * 1e3;

            eprintln!(
                "{rows:>10} | {blk_mib:>9} | {owner_ms:>13.2} | {attach_ms:>9.2} \
                 | {detach_ms:>19.2} | {cp:>10.2}   (shared {} rows, {} GiB deduped)",
                shared_rows,
                shared_rows as u64 * track_bytes_per_row / (1 << 30)
            );
        }
    }
}

#[test]
fn vmm_leak_cycle() {
    let Some(assets) = gated() else { return };
    let _env = common::env_guard();
    let ckpt = assets.join("checkpoint");
    let tok = load_tokenizer(&assets);
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let baseline = used(&be);
    const TOLERANCE: u64 = 64 * MIB;

    std::env::set_var("PLOW_VMM_PREFIX", "1");
    std::env::set_var("PLOW_VMM_BLOCK_MIB", "2");
    let prefix = long_ids(
        &tok,
        4200,
        "In the beginning the packet machine decoded tokens. ",
    );
    for cycle in 0..2usize {
        let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("load");
        // Publish + attach so the pool's cache/snapshot paths are exercised.
        let (t0, _) = serve(&mut e, 0, &prefix, 8);
        let (t1, _) = serve(&mut e, 1, &prefix, 8);
        assert_eq!(t0, t1, "same prompt must decode identically across slots");
        let s = e.vmm_stats().unwrap();
        assert!(
            s.blocks_shared_mapped > 0,
            "cycle {cycle}: no sharing happened"
        );
        drop(e);
        let after = used(&be);
        eprintln!(
            "cycle {cycle}: after unload {} MiB (baseline {} MiB, shared {} blocks)",
            after / MIB,
            baseline / MIB,
            s.blocks_shared_mapped
        );
        assert!(
            after <= baseline + TOLERANCE,
            "cycle {cycle}: VRAM did not return to baseline: {} vs {} MiB",
            after / MIB,
            baseline / MIB
        );
    }
    std::env::remove_var("PLOW_VMM_PREFIX");
    std::env::remove_var("PLOW_VMM_BLOCK_MIB");
}
