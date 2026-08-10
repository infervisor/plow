//! `Mode` exists to split the old `decode: bool` into two independent axes. The ONLY thing
//! that keeps that refactor honest is that the two pre-existing corners still decode to the
//! same pair of booleans they were hardcoded to before — `Prefill` was `decode=false`
//! everywhere, `Decode` was `decode=true` everywhere. If either row below changes, every
//! emitted program changes with it, silently. (Verified once against real packets: the
//! Qwen3-4B blob is byte-identical pre/post refactor at ctx 4256/16544 and n_cu 170/256.)
use super::Mode;

#[test]
fn legacy_corners_are_unchanged() {
    assert!(!Mode::Prefill.decode_shape() && !Mode::Prefill.gemv());
    assert!(Mode::Decode.decode_shape() && Mode::Decode.gemv());
}

#[test]
fn decode_tiled_is_decode_shape_on_prefill_kernels() {
    // The whole point: decode's shape (one row, KV append, ring mask) with prefill's
    // kernels (tiled GEMM, FlashPrefill). Neither legacy corner can express this.
    assert!(Mode::DecodeTiled.decode_shape());
    assert!(!Mode::DecodeTiled.gemv());
}
