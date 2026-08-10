use super::{default_chunk, kv_ring, kv_ring_rows, MAX_CHUNK_MAX};

/// An all-global model (`window == 0`) must keep the full chunk. `kv_ring`
/// returns `(ctx, MASK_NONE)` for full layers, so a smaller chunk buys no
/// KV there and only costs prefill launches — the Gemma-shaped default
/// must not leak onto Llama-shaped networks.
#[test]
fn all_global_models_keep_the_full_chunk() {
    assert_eq!(default_chunk(0), MAX_CHUNK_MAX);
    // and the chunk genuinely does not size a full layer's cache
    let (rows_big, mask) = kv_ring(true, 8192, 0, MAX_CHUNK_MAX);
    let (rows_small, _) = kv_ring(true, 8192, 0, 1024);
    assert_eq!(rows_big, rows_small, "chunk must not change a full layer");
    assert_eq!(mask, super::KV_MASK_NONE);
}

/// A windowed model derives the chunk from its own window, so the ring
/// lands at 2 x next_pow2(window) — the floor the invariant allows.
#[test]
fn windowed_models_derive_chunk_from_window() {
    assert_eq!(default_chunk(1024), 1024); // Gemma-4
    assert_eq!(default_chunk(4096), 4096);
    assert_eq!(default_chunk(768), 1024); // rounded up to a power of two
                                          // never below the bucket floor, never above the ladder top
    assert_eq!(default_chunk(1), super::MAX_CHUNK_MIN);
    assert_eq!(default_chunk(1 << 20), MAX_CHUNK_MAX);
}

/// The wrap invariant must hold for every window the default picks —
/// violating it aliases a chunk's rows onto its own history, which is a
/// silent wrong answer rather than a crash.
#[test]
fn derived_chunk_satisfies_the_wrap_invariant() {
    for w in [128u32, 512, 768, 1024, 2048, 4096, 8192, 16384] {
        let c = default_chunk(w);
        let ring = kv_ring_rows(w, c);
        assert!(
            ring >= w + c - 1,
            "window {w} chunk {c} ring {ring} violates ring >= window + chunk - 1"
        );
    }
}
