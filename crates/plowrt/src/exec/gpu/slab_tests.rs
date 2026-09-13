use super::*;
// `gpu.rs` has its own `slab_carve` (section carving); the per-tensor rule under test is the
// shared one in `memory`.
use crate::memory::{slab_carve, slab_pad};

#[test]
fn pad_rounds_up_and_leaves_exact_multiples_alone() {
    assert_eq!(slab_pad(0), 0);
    assert_eq!(slab_pad(1), SLAB_ALIGN);
    assert_eq!(slab_pad(SLAB_ALIGN - 1), SLAB_ALIGN);
    assert_eq!(slab_pad(SLAB_ALIGN), SLAB_ALIGN);
    assert_eq!(slab_pad(SLAB_ALIGN + 1), 2 * SLAB_ALIGN);
}

/// The property the carve actually depends on: summing `slab_pad` to size
/// the allocation and advancing a cursor by `slab_pad` over the same list
/// must agree, and no tensor may extend past the total. An overshoot here
/// would alias two tensors onto the same bytes — silently wrong weights
/// rather than a crash, which is why it is asserted rather than trusted.
#[test]
fn carve_cursor_lands_exactly_on_the_sized_total() {
    // Deliberately mixed: sub-stride, exact-stride, stride+1, zero, large.
    let sizes = [
        1u64,
        SLAB_ALIGN - 1,
        SLAB_ALIGN,
        SLAB_ALIGN + 1,
        0,
        1 << 20,
        (1 << 20) + 7,
    ];
    let total: u64 = sizes.iter().copied().map(slab_pad).sum();

    let mut off = 0u64;
    for s in sizes {
        assert!(off + s <= total, "tensor at {off} (+{s}) runs past {total}");
        off += slab_pad(s);
    }
    assert_eq!(off, total, "cursor must consume exactly the sized span");
}

/// The carve uses `slab_carve` (`bytes.max(1)`), the same rule as the AMD loader:
/// a zero-byte tensor still gets an address of its own, so a blob of nothing but
/// zero-byte tensors sizes to one stride each instead of aliasing them all.
#[test]
fn zero_byte_tensors_still_advance_the_cursor() {
    let sizes = [0u64, 0, 0];
    let total: u64 = sizes.iter().copied().map(slab_carve).sum();
    assert_eq!(total, 3 * SLAB_ALIGN);

    let mut off = 0u64;
    let mut seen = Vec::new();
    for s in sizes {
        seen.push(off);
        off += slab_carve(s);
    }
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 3, "every tensor must get a distinct address");
}
