//! §I Batch formation + bucket/pkt-stream selection.
//!
//! The compiler emits a discrete ladder of `(phase, batch, seq)` buckets. Given
//! the live batch size and max sequence length, pick the compiled bucket that
//! covers them by rounding **up** the ladder — that tuple selects the exact
//! `.pkt` stream to run. Picking a different stream as load changes is the "pick
//! a different pkt stream based on arrival rate" behaviour.

use crate::asset::{BucketKey, ModelBundle, Phase};

/// Round `(phase, batch, seq)` up to the nearest compiled bucket in `bundle`.
/// Returns the covering bucket key, or `None` if the request exceeds every
/// compiled bucket (caller sheds or chunks).
pub fn select_bucket(
    bundle: &ModelBundle,
    phase: Phase,
    batch: i64,
    seq: i64,
) -> Option<BucketKey> {
    bundle
        .bucket_keys()
        .filter(|k| k.phase == phase && k.batch >= batch && k.seq >= seq)
        .min_by_key(|k| (k.batch, k.seq))
}

/// Adaptive batch-formation window: short when arrivals are frequent (the next
/// bucket fills on its own), capped when they are rare (don't starve a lone
/// request). `lambda` is arrivals/sec; returns the hold time in milliseconds.
pub fn formation_window_ms(lambda: f64, max_hold_ms: f64) -> f64 {
    if lambda <= f64::EPSILON {
        return 0.0;
    }
    // ~ time to accrue one more arrival, capped.
    (1000.0 / lambda).min(max_hold_ms)
}

/// The hold for an arrival into an empty slot table. `pending` counts requests queued behind it
/// or still inside a handler (tokenizing). With `idle_dispatch`, a lone arrival dispatches at once;
/// a burst's first arrival still sees its peers in `pending` and holds as before.
pub fn cold_start_hold_ms(lambda: f64, max_hold_ms: f64, idle_dispatch: bool, pending: usize) -> f64 {
    if idle_dispatch && pending == 0 {
        return 0.0;
    }
    formation_window_ms(lambda, max_hold_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_dispatch_skips_the_hold_only_for_a_lone_arrival() {
        for lambda in [0.0, 1.3, 50.0, 5000.0] {
            for pending in [0, 1, 15] {
                assert_eq!(
                    cold_start_hold_ms(lambda, 8.0, false, pending),
                    formation_window_ms(lambda, 8.0)
                );
            }
            assert_eq!(cold_start_hold_ms(lambda, 8.0, true, 0), 0.0);
            for pending in [1, 15] {
                assert_eq!(
                    cold_start_hold_ms(lambda, 8.0, true, pending),
                    formation_window_ms(lambda, 8.0)
                );
            }
        }
        assert_eq!(cold_start_hold_ms(1.3, 8.0, false, 0), 8.0);
    }
}
