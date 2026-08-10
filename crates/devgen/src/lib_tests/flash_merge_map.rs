use super::*;

/// The kernel's work walk, re-implemented here. If `d_flash_merge`'s decomposition and this
/// map ever disagree the failure is a SILENT wrong token, so pin the contract in a test.
fn kernel_items(n_bh: u32, nblk_m: u32, j: u32) -> Vec<u32> {
    let dsplit = nblk_m.div_ceil(n_bh.max(1)).max(1);
    (j..n_bh * dsplit).step_by(nblk_m as usize).collect()
}

/// Every merge item is run by EXACTLY ONE workgroup, at any width — including widths that are
/// not a multiple of `n_bh` (there `n_work > nblk_m` and the walk wraps).
#[test]
fn every_d_chunk_is_covered_exactly_once() {
    for nblk_m in [1u32, 8, 32, 64, 200, 256] {
        let n_bh = 32;
        let dsplit = nblk_m.div_ceil(n_bh).max(1);
        let mut seen: Vec<u32> = (0..nblk_m)
            .flat_map(|j| kernel_items(n_bh, nblk_m, j))
            .collect();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..n_bh * dsplit).collect::<Vec<_>>(),
            "nblk_m={nblk_m}"
        );
    }
}

/// `d_flash_prefill`'s OWN q-tile walk, re-implemented from `runtime/amd/op_attention.h:221`
/// and `:228-230`: `q_tiles = ceil(n_q / (PLOW_WAVES*FA_BQ))`, item
/// `w = (qt*n_head + h)*nsplit + sp`, run by workgroup `w % nblk_f`. The tile height is
/// `PLOW_WAVES*FA_BQ` and `FlashPrefill` runs 4-wave, so it is 128 — `FA_BQ`'s own comment at
/// `op_attention.h:49` spells that out.
///
/// The `128` is written out here rather than taken from [`FLASH_Q_TILE_ROWS`] ON PURPOSE:
/// this is the KERNEL's number, transcribed from the kernel, and a test that read the
/// emitter's constant for both sides would be a tautology that passes at any value.
fn kernel_flash_producers(row: u32, h: u32, nsplit: u32, n_head: u32, nblk_f: u32) -> Vec<u32> {
    const KERNEL_WAVES: u32 = 4; // scripts/build_gfx950.sh: the flash object is -DPLOW_WG_WAVES=4
    const KERNEL_FA_BQ: u32 = 32; // runtime/amd/op_attention.h:49
    let qt = row / (KERNEL_WAVES * KERNEL_FA_BQ);
    (0..nsplit)
        .map(|sp| ((qt * n_head + h) * nsplit + sp) % nblk_f)
        .collect()
}

/// EVERY flash slice that wrote a merge item's partials must be in that item's wait set.
///
/// This is the flash -> merge `Dep::Fine` edge, and prefill programs KEEP their fine edges
/// (`crates/packet/src/devbuild.rs`, the `PLOW_CHAIN_BYPASS` note: "the prefill programs carry
/// Fine edges and are left untouched"). A missing edge is not a hang and not a fault: the
/// merge reads `(o, m, l)` partials the flash has not written yet and folds garbage into
/// `n.at`. Fluent, wrong, and invisible.
///
/// The head counts below are the point. At 8/16/32/64 the producer indices alias mod `nblk_f`
/// and a WRONG `rows_per_item` still yields a complete map, which is why this survived; at 40
/// (Qwen3-14B, Qwen2.5-14B, Llama-2-13B) and 28 (Qwen2-57B) it does not.
#[test]
fn merge_waits_on_the_slice_that_actually_wrote_the_row() {
    let n_cu = 256u32;
    for heads in [8u32, 16, 20, 24, 28, 32, 40, 48, 64] {
        for t in [128u32, 192, 256, 384, 512, 768, 1024, 2048] {
            let ns = n_cu
                .div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
                .max(1);
            if ns <= 1 {
                continue; // fused: flash normalizes in its epilogue, no FlashMerge op
            }
            let (n_bh, nblk_f) = (t * heads, n_cu);
            let nblk_m = (n_bh * flash_merge_dsplit()).min(n_cu).max(1);
            let map = flash_merge_map(n_bh, ns, FLASH_Q_TILE_ROWS, heads, nblk_f, nblk_m);
            let dsplit = nblk_m.div_ceil(n_bh.max(1)).max(1);
            for j in 0..nblk_m {
                for w in kernel_items(n_bh, nblk_m, j) {
                    let hb = w / dsplit;
                    let (row, h) = (hb / heads, hb % heads);
                    for p in kernel_flash_producers(row, h, ns, heads, nblk_f) {
                        assert!(
                            map[j as usize].contains(&p),
                            "heads={heads} t={t} ns={ns}: merge wg {j} folds row {row} \
                                 head {h} but does not wait on flash slice {p} that wrote it"
                        );
                    }
                }
            }
        }
    }
}

/// The fix must be a NO-OP for every head count shipped today, so it needs no re-measurement:
/// at 8/16/32/64 the emitted wait sets are byte-identical either way.
#[test]
fn shipped_head_counts_are_unaffected_by_the_tile_correction() {
    let n_cu = 256u32;
    for heads in [8u32, 16, 32, 64] {
        for t in [128u32, 256, 512, 1024, 2048] {
            let ns = n_cu
                .div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
                .max(1);
            if ns <= 1 {
                continue;
            }
            let (n_bh, nblk_f) = (t * heads, n_cu);
            let nblk_m = (n_bh * flash_merge_dsplit()).min(n_cu).max(1);
            assert_eq!(
                flash_merge_map(n_bh, ns, Q_TILE_ROWS, heads, nblk_f, nblk_m),
                flash_merge_map(n_bh, ns, FLASH_Q_TILE_ROWS, heads, nblk_f, nblk_m),
                "heads={heads} t={t}: the correction changed a shipped program"
            );
        }
    }
}

/// dsplit=1 must leave the map byte-identical: the widening is opt-in and the default path
/// has to stay the shipped one.
#[test]
fn dsplit_one_is_the_old_map() {
    let (n_bh, ns, n_head, nblk_f) = (32, 8, 32, 256);
    let map = flash_merge_map(n_bh, ns, 1, n_head, nblk_f, n_bh);
    for (j, s) in map.iter().enumerate() {
        let (b, h) = (j as u32 / n_head, j as u32 % n_head);
        let mut want: Vec<u32> = (0..ns)
            .map(|sp| ((b * n_head + h) * ns + sp) % nblk_f)
            .collect();
        want.sort_unstable();
        want.dedup();
        assert_eq!(s, &want, "wg {j}");
    }
}

/// Widened: a merge workgroup must depend on exactly the flash slices of the `(b,h)` whose
/// D-chunks it runs — no more (that would re-widen the gate), no fewer (that is a race).
#[test]
fn dsplit_eight_gates_on_its_own_bh_only() {
    let (n_bh, ns, n_head, nblk_f, nblk_m) = (32u32, 8u32, 32u32, 256u32, 256u32);
    let map = flash_merge_map(n_bh, ns, 1, n_head, nblk_f, nblk_m);
    assert_eq!(map.len(), nblk_m as usize);
    let dsplit = nblk_m / n_bh;
    for (j, s) in map.iter().enumerate() {
        let items = kernel_items(n_bh, nblk_m, j as u32);
        assert_eq!(items.len(), 1, "at 256 wgs each runs exactly one D-chunk");
        let hb = items[0] / dsplit;
        let (b, h) = (hb / n_head, hb % n_head);
        let mut want: Vec<u32> = (0..ns)
            .map(|sp| ((b * n_head + h) * ns + sp) % nblk_f)
            .collect();
        want.sort_unstable();
        want.dedup();
        assert_eq!(s, &want, "wg {j}");
        // 8 of 256 producers, per the doc comment above — not a dense 256-wide gate.
        assert_eq!(s.len(), ns as usize);
    }
}
