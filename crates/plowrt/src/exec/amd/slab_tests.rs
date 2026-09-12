use super::*;
use crate::memory::slab_pad;

#[test]
fn pad_rounds_up_and_leaves_exact_multiples_alone() {
    assert_eq!(slab_pad(0), 0);
    assert_eq!(slab_pad(1), SLAB_ALIGN);
    assert_eq!(slab_pad(SLAB_ALIGN - 1), SLAB_ALIGN);
    assert_eq!(slab_pad(SLAB_ALIGN), SLAB_ALIGN);
    assert_eq!(slab_pad(SLAB_ALIGN + 1), 2 * SLAB_ALIGN);
}

/// The property the carve depends on: sizing the allocation by summing
/// `slab_pad` and advancing a cursor by `slab_pad` over the same list must
/// agree, and no tensor may extend past the total. An overshoot would alias
/// two tensors onto the same bytes — silently wrong weights rather than a
/// crash, which is why the loader asserts it rather than trusting it.
#[test]
fn carve_cursor_lands_exactly_on_the_sized_total() {
    // Sub-stride, exact-stride, stride+1, zero, and a real expert
    // projection (1.4 MiB) — the size ROCr rounds worst.
    let sizes = [
        1u64,
        SLAB_ALIGN - 1,
        SLAB_ALIGN,
        SLAB_ALIGN + 1,
        0,
        1_468_006,
        1 << 20,
    ];
    let total: u64 = sizes.iter().copied().map(slab_pad).sum();

    let mut off = 0u64;
    for s in sizes {
        assert!(off + s <= total, "tensor at {off} (+{s}) runs past {total}");
        off += slab_pad(s);
    }
    assert_eq!(off, total, "cursor must consume exactly the sized span");
}

/// The loader carves `slab_carve` = `bytes.max(1)`, never `bytes`: a zero-byte
/// tensor still needs an address of its own, and a zero-length carve would hand
/// the next tensor the same one. This pins that the `.max(1)` is load-bearing.
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
