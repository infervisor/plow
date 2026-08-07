//! Hardware gates for `impl VmmOps for HsaBackend` — the ROCr
//! (`hsa_amd_vmem_*`) counterpart of the CUDA VMM surface that
//! `crates/plowrt/src/memory/vmm.rs` drives.
//!
//! These are the AMD versions of the CUDA "probes" the VMM plan cites:
//!
//! * `[1] granularity`   — what the physical block size must be a multiple of.
//! * `[2] multi-map`     — ONE physical handle mapped at TWO virtual addresses,
//!   both readable/writable, byte-identical. This is the whole basis of prefix
//!   sharing: without it, dedup is impossible.
//! * `[5] map under load`— mapping into a live reservation while the device is
//!   busy must not fault or implicitly synchronise.
//! * `[6] reserve cost`  — a multi-GiB reservation is cheap (no physical
//!   backing), which is what lets the pool reserve `[batch][kvh][max_ctx][hd]`
//!   in full and back only the decode frontier.
//!
//! Plus the number the block-size choice depends on: **`hsa_amd_vmem_set_access`
//! µs per granule**. The CUDA path measured `cuMemSetAccess` at ~69 µs/granule,
//! which is why its sharing blocks are 64 MiB-class. If ROCr is materially
//! cheaper or dearer the AMD block size should differ, so it is measured here
//! rather than assumed.
//!
//! Needs a real gfx9xx GPU + ROCr, gated like the other device tests:
//!
//!   PLOW_GPU_TEST=1 cargo test -p plowrt --features hsa --test hsa_vmm \
//!       -- --nocapture --test-threads=1

#![cfg(feature = "hsa")]

use std::time::Instant;

use plowrt::device::hsa::HsaBackend;
use plowrt::memory::vmm::VmmOps;

const MIB: u64 = 1 << 20;

fn gpu_enabled() -> bool {
    std::env::var("PLOW_GPU_TEST").as_deref() == Ok("1")
}

/// One process-global backend — `hsa_init` is not re-entrant across backends
/// (see the note in `hsa_primitives.rs`). Held in an `Arc` so the slab-pool
/// test can hand the SAME instance to `VmmSlab::new` (a second backend is not
/// an option).
static BE: std::sync::OnceLock<std::sync::Arc<HsaBackend>> = std::sync::OnceLock::new();

fn backend_init() -> &'static std::sync::Arc<HsaBackend> {
    BE.get_or_init(|| {
        std::sync::Arc::new(HsaBackend::new(0).unwrap_or_else(|e| {
            panic!("PLOW_GPU_TEST=1 but no HSA device: {e}");
        }))
    })
}

fn backend_arc() -> std::sync::Arc<HsaBackend> {
    std::sync::Arc::clone(backend_init())
}

fn backend() -> &'static HsaBackend {
    &**backend_init()
}

/// Read `dst.len()` bytes back from a raw device VA. `DeviceMem` has no public
/// non-owning constructor, so go through the backend's own raw-pointer D2H.
fn d2h(be: &HsaBackend, dptr: u64, dst: &mut [u8]) {
    let s = be.stream_create().expect("stream_create");
    // SAFETY: `dptr` is a live, RW-mapped range of at least `dst.len()` bytes;
    // the copy is awaited inside `memcpy_dtoh_async` before it returns.
    unsafe { be.memcpy_dtoh_async(dst, dptr, &s) }.expect("D2H");
}

/// Probe [1]. The granule is what every `create`/`map` size must divide by, and
/// what `VmmKv::new` validates the head window against.
#[test]
fn granularity_is_a_usable_power_of_two() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1");
        return;
    }
    let be = backend();
    assert!(
        be.has_vmm(),
        "libhsa-runtime64 has no hsa_amd_vmem_* — needs ROCm >= 5.7"
    );
    let g = VmmOps::granularity(be).expect("granularity");
    println!("hsa_amd_vmem granularity = {g} B ({} KiB)", g >> 10);
    assert!(g.is_power_of_two(), "granule {g} is not a power of two");
    assert!((4096..=(256 * MIB)).contains(&g), "implausible granule {g}");
}

/// Probe [6] + the basic lifecycle: reserve a large VA range, back one granule
/// of it, grant access, write through it from the host, read it back, then tear
/// the whole thing down. Reservation must be cheap — it is sized at
/// `[batch][kvh][max_ctx][hd]`, tens of GiB, and must not touch HBM.
#[test]
fn reserve_map_set_access_roundtrip() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1");
        return;
    }
    let be = backend();
    let gran = VmmOps::granularity(be).expect("granularity");
    let span = 8 * (1u64 << 30); // 8 GiB of VA, 0 B of HBM

    let t = Instant::now();
    let va = VmmOps::reserve(be, span).expect("reserve 8 GiB VA");
    let reserve_us = t.elapsed().as_secs_f64() * 1e6;
    println!("reserve {} GiB VA: {reserve_us:.1} us", span >> 30);
    assert_eq!(va % gran, 0, "reservation {va:#x} not granule-aligned");
    assert!(
        reserve_us < 50_000.0,
        "reserving unbacked VA cost {reserve_us:.0} us — that is not a pure VA op"
    );

    let h = VmmOps::create(be, gran).expect("handle_create");
    VmmOps::map(be, va, gran, h).expect("map");
    VmmOps::set_access(be, va, gran).expect("set_access");

    // The mapping is real memory: round-trip a pattern through it.
    let pattern: Vec<u8> = (0..4096u32).map(|i| (i * 31 + 7) as u8).collect();
    be.memcpy_htod(va, &pattern).expect("H2D into mapped VA");
    let mut back = vec![0u8; pattern.len()];
    d2h(be, va, &mut back);
    assert_eq!(back, pattern, "mapped VMM range did not round-trip");

    VmmOps::unmap(be, va, gran);
    VmmOps::release(be, h);
    VmmOps::address_free(be, va, span);
}

/// Probe [2] — THE gate for prefix sharing. One physical handle, two virtual
/// windows: a write through window A must be visible through window B, because
/// that is what "held once in HBM, mapped into every sharing sequence" means.
/// If this fails, `VmmKv`'s dedup is a lie and the AMD path must not enable it.
#[test]
fn one_handle_multi_mapped_aliases() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1");
        return;
    }
    let be = backend();
    let gran = VmmOps::granularity(be).expect("granularity");
    let span = 4 * gran;

    let va = VmmOps::reserve(be, span).expect("reserve");
    let h = VmmOps::create(be, gran).expect("create");
    let (a, b) = (va, va + 2 * gran);
    VmmOps::map(be, a, gran, h).expect("map A");
    VmmOps::set_access(be, a, gran).expect("set_access A");
    VmmOps::map(be, b, gran, h).expect("map B (multi-map of one handle)");
    VmmOps::set_access(be, b, gran).expect("set_access B");

    let pattern: Vec<u8> = (0..8192u32).map(|i| (i ^ 0xA5) as u8).collect();
    be.memcpy_htod(a, &pattern).expect("H2D through window A");
    let mut back = vec![0u8; pattern.len()];
    d2h(be, b, &mut back);
    assert_eq!(
        back, pattern,
        "multi-mapped handle did not alias — no HBM dedup is possible on this runtime"
    );

    VmmOps::unmap(be, b, gran);
    VmmOps::unmap(be, a, gran);
    VmmOps::release(be, h);
    VmmOps::address_free(be, va, span);
}

/// THE COST MODEL. `map` + `set_access` per granule, and per candidate sharing
/// block, so the AMD block size is chosen on a measurement instead of on the
/// CUDA number (~69 µs/granule for `cuMemSetAccess`).
///
/// Reported per-block and normalised per-granule: a block is `n` granules, and
/// what decides the block size is whether `set_access` cost scales with the
/// number of granules (then bigger blocks are free) or with the number of
/// CALLS (then bigger blocks are strictly better anyway, but by more).
#[test]
fn map_and_set_access_cost_per_block() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1");
        return;
    }
    let be = backend();
    let gran = VmmOps::granularity(be).expect("granularity");
    println!("\ngranule = {} KiB", gran >> 10);
    println!(
        "{:>10} {:>10} {:>12} {:>12} {:>12} {:>14}",
        "block", "granules", "create us", "map us", "setacc us", "setacc/gran"
    );

    for mult in [1u64, 2, 8, 32] {
        let block = gran * mult;
        // 8 independent blocks per point — one measurement is noise.
        let reps = 8u32;
        let span = block * reps as u64;
        let va = VmmOps::reserve(be, span).expect("reserve");

        let mut handles = Vec::with_capacity(reps as usize);
        let t = Instant::now();
        for _ in 0..reps {
            handles.push(VmmOps::create(be, block).expect("create"));
        }
        let create_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let t = Instant::now();
        for (i, &h) in handles.iter().enumerate() {
            VmmOps::map(be, va + i as u64 * block, block, h).expect("map");
        }
        let map_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let t = Instant::now();
        for i in 0..reps as u64 {
            VmmOps::set_access(be, va + i * block, block).expect("set_access");
        }
        let set_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        println!(
            "{:>9} M {:>10} {:>12.1} {:>12.1} {:>12.1} {:>14.1}",
            block / MIB,
            mult,
            create_us,
            map_us,
            set_us,
            set_us / mult as f64
        );

        for i in 0..reps as u64 {
            VmmOps::unmap(be, va + i * block, block);
        }
        for h in handles {
            VmmOps::release(be, h);
        }
        VmmOps::address_free(be, va, span);
    }
}

/// The physical-chunk pool on ROCr (`pool_put`/`pool_bytes`/`pool_trim`/
/// `pool_take` — the AMD side of `PLOW_SLAB_KEEP`), then the full reload
/// shape: a kept `VmmSlab`'s chunks must feed the next slab's mapper instead
/// of re-paying `hsa_amd_vmem_handle_create`.
///
/// Mutates `PLOW_SLAB_KEEP` — run with `--test-threads=1` as this file's
/// header already requires.
#[test]
fn slab_chunk_pool_roundtrip_and_reuse() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs an HSA GPU)");
        return;
    }
    let be = backend();
    let gran = VmmOps::granularity(be).expect("granularity");
    let chunk = 4 * gran;

    // Raw ledger: put two chunks, trim to one, take the survivor.
    let h1 = VmmOps::create(be, chunk).expect("create h1");
    let h2 = VmmOps::create(be, chunk).expect("create h2");
    be.pool_put(vec![(h1, chunk), (h2, chunk)]);
    assert_eq!(be.pool_bytes(), 2 * chunk);
    assert_eq!(
        be.pool_trim(chunk),
        chunk,
        "trim releases down to the bound"
    );
    assert_eq!(be.pool_bytes(), chunk);
    let took = be.pool_take();
    assert_eq!(took.len(), 1);
    assert_eq!(be.pool_bytes(), 0, "take drains the pool");
    for (h, _) in took {
        VmmOps::release(be, h);
    }

    // Reload shape: slab 1 drops with PLOW_SLAB_KEEP=1 (chunks pool), slab 2
    // of the same chunking re-maps them (pool drains at its bringup), and its
    // drop with the flag off releases everything.
    let bytes = 4 * chunk;
    std::env::set_var("PLOW_SLAB_KEEP", "1");
    let slab = plowrt::memory::vmm::VmmSlab::new(
        backend_arc() as std::sync::Arc<dyn VmmOps>,
        bytes,
        chunk,
    )
    .expect("slab 1");
    slab.wait_mapped(bytes).expect("slab 1 mapped");
    drop(slab);
    assert_eq!(be.pool_bytes(), bytes, "kept drop pooled every chunk");

    let slab2 = plowrt::memory::vmm::VmmSlab::new(
        backend_arc() as std::sync::Arc<dyn VmmOps>,
        bytes,
        chunk,
    )
    .expect("slab 2");
    slab2.wait_mapped(bytes).expect("slab 2 mapped");
    assert_eq!(
        be.pool_bytes(),
        0,
        "slab 2's mapper drew every pooled chunk"
    );
    std::env::remove_var("PLOW_SLAB_KEEP");
    drop(slab2);
    assert_eq!(
        be.pool_bytes(),
        0,
        "flag off: drop releases, nothing pooled"
    );
}
