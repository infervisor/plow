//! CPU-only kernels shared by residency admission and its external contract checks.
//!
//! `u64::MAX` is a fail-closed accounting sentinel, never an admissible requirement.
//! Registration may oversubscribe; this module governs resident loads and returns.

pub fn sum_bytes(parts: impl IntoIterator<Item = u64>) -> u64 {
    parts.into_iter().fold(0, u64::saturating_add)
}

pub fn admits(required: u64, available: u64) -> bool {
    required != u64::MAX && required <= available
}

/// Preserve resident KV claims and the separately accounted planned headroom.
pub fn available(free: u64, credit: u64, claimed: u64, headroom: u64) -> u64 {
    free.checked_add(credit)
        .map(|n| n.saturating_sub(claimed).min(headroom))
        .unwrap_or(0)
}

/// A dispatcher acknowledgement authorizes one removal of the owned engine.
/// The caller must establish that acknowledgement implies device quiescence;
/// a timeout or a dropped acknowledgement channel is not completion.
#[derive(Debug, Default)]
pub struct ReleaseGate {
    completed: bool,
    returned: bool,
}

impl ReleaseGate {
    pub fn complete(&mut self, acknowledged: bool) {
        self.completed |= acknowledged;
    }

    pub fn claim_return(&mut self) -> bool {
        if !self.completed || self.returned {
            return false;
        }
        self.returned = true;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_and_claims_fail_closed() {
        assert!(!admits(sum_bytes([u64::MAX, 1]), u64::MAX));
        assert_eq!(available(u64::MAX, 1, 0, u64::MAX), 0);
        assert_eq!(available(100, 20, 30, 100), 90);
        assert!(!admits(91, available(100, 20, 30, 100)));
    }

    #[test]
    fn return_requires_acknowledgement_and_is_once_only() {
        let mut gate = ReleaseGate::default();
        assert!(!gate.claim_return());
        gate.complete(false);
        assert!(!gate.claim_return());
        gate.complete(true);
        assert!(gate.claim_return());
        assert!(!gate.claim_return());
        gate.complete(true);
        assert!(!gate.claim_return());
    }
}
