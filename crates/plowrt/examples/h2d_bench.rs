//! H2D upload-path microbenchmark (GH200 staging investigation).
//!
//! The load-profile shows the warm weight upload is bounded by the
//! single-threaded mmap→pinned staging memcpy (13.2 GiB/s), not the DMA
//! (306 GiB/s). On coherent platforms (Grace-Hopper ATS) the driver may be
//! able to DMA straight from pageable file-backed memory — this measures it.
//!
//!   cargo run --release -p plowrt --features cuda --example h2d_bench -- \
//!       <safetensors-file> [span GiB, default 8]
//!
//! Paths measured over the same warm span of the real checkpoint file:
//!   A  staged   — 2×64 MiB pinned double-buffer (what UploadPipe does today)
//!   B  direct   — cuMemcpyHtoDAsync per 64 MiB chunk straight from the mmap
//!   C  oneshot  — one cuMemcpyHtoDAsync of the whole span from the mmap
//!   D  par-copy — N-thread mmap→heap memcpy (upper bound for parallel staging)

#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::Instant;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;

const CHUNK: usize = 64 << 20;

/// Committed (va, bytes, handle) triples of the current E round, drained
/// serially after each timing window.
static HANDLES: std::sync::Mutex<Vec<(u64, u64, u64)>> = std::sync::Mutex::new(Vec::new());

fn gibs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64 / secs
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: h2d_bench <safetensors> [span GiB]");
    let span_gib: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let span = span_gib << 30;

    let file = std::fs::File::open(&path).expect("open");
    // SAFETY: read-only mapping of the checkpoint, held for the whole run.
    let map = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
    assert!(map.len() >= span, "file smaller than span");
    let src = &map[..span];

    // Warm the span (fault every page into cache + this page table).
    let t = Instant::now();
    let mut sum = 0u8;
    for i in (0..span).step_by(4096) {
        sum ^= src[i];
    }
    println!(
        "warm touch: {:.2} s (checksum {sum})",
        t.elapsed().as_secs_f64()
    );

    let be = Arc::new(CudaBackend::new(0).expect("cuda"));
    let dev = be.alloc(0, span as u64).expect("device alloc");
    let stream = be.stream_create().expect("stream");

    // ---- A: staged double-buffer (UploadPipe's shape) ----
    {
        let mut bufs = [
            be.host_alloc_pinned(CHUNK).expect("pinned"),
            be.host_alloc_pinned(CHUNK).expect("pinned"),
        ];
        let ends = [
            be.event_create(false).expect("ev"),
            be.event_create(false).expect("ev"),
        ];
        let mut primed = [false, false];
        let t = Instant::now();
        for (n, chunk) in src.chunks(CHUNK).enumerate() {
            let slot = n & 1;
            if primed[slot] {
                be.event_synchronize(&ends[slot]).unwrap();
            }
            bufs[slot].as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: pinned buffer outlives the synchronize below.
            unsafe {
                be.memcpy_htod_async(
                    dev.base + (n * CHUNK) as u64,
                    &bufs[slot].as_slice()[..chunk.len()],
                    &stream,
                )
                .unwrap();
            }
            be.event_record(&ends[slot], &stream).unwrap();
            primed[slot] = true;
        }
        be.stream_synchronize(&stream).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "A staged   : {:>7.1} ms  {:>6.1} GiB/s",
            s * 1e3,
            gibs(span, s)
        );
    }

    // ---- B: direct per-chunk from the mmap ----
    {
        let t = Instant::now();
        for (n, chunk) in src.chunks(CHUNK).enumerate() {
            // SAFETY: the mmap outlives the synchronize below.
            unsafe {
                be.memcpy_htod_async(dev.base + (n * CHUNK) as u64, chunk, &stream)
                    .unwrap();
            }
        }
        be.stream_synchronize(&stream).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "B direct   : {:>7.1} ms  {:>6.1} GiB/s",
            s * 1e3,
            gibs(span, s)
        );
    }

    // ---- C: one-shot direct ----
    {
        let t = Instant::now();
        // SAFETY: as B.
        unsafe {
            be.memcpy_htod_async(dev.base, src, &stream).unwrap();
        }
        be.stream_synchronize(&stream).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "C oneshot  : {:>7.1} ms  {:>6.1} GiB/s",
            s * 1e3,
            gibs(span, s)
        );
    }

    // ---- E: VMM physical-commit rate — thread scaling on disjoint ranges ----
    // With the copy at 330 GiB/s the mapper's serial ~13 GiB/s commit becomes
    // the load's critical path; this asks whether create+map+set_access
    // parallelizes across threads (disjoint 256 MiB chunks of one reservation).
    {
        use plowrt::memory::vmm::VmmOps;
        const CCHUNK: u64 = 256 << 20;
        for threads in [1usize, 2, 4, 8] {
            let va = be.reserve(span as u64).expect("reserve");
            let n_chunks = (span as u64).div_ceil(CCHUNK);
            let next = std::sync::atomic::AtomicU64::new(0);
            let t = Instant::now();
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n_chunks {
                            return;
                        }
                        let off = i * CCHUNK;
                        let n = CCHUNK.min(span as u64 - off);
                        let h = be.create(n).expect("create");
                        be.map(va + off, n, h).expect("map");
                        be.set_access(va + off, n).expect("set_access");
                        // Leak-free teardown happens below, serially.
                        HANDLES.lock().unwrap().push((va + off, n, h));
                    });
                }
            });
            let s = t.elapsed().as_secs_f64();
            println!(
                "E commit  {threads:>2}: {:>7.1} ms  {:>6.1} GiB/s",
                s * 1e3,
                gibs(span, s)
            );
            for (cva, n, h) in HANDLES.lock().unwrap().drain(..) {
                be.unmap(cva, n);
                be.release(h);
            }
            be.address_free(va, span as u64);
        }
    }

    // ---- F: commit phase split — where do the bytes/sec go? ----
    {
        use plowrt::memory::vmm::VmmOps;
        const CCHUNK: u64 = 256 << 20;
        let va = be.reserve(span as u64).expect("reserve");
        let n_chunks = (span as u64).div_ceil(CCHUNK);
        let (mut t_create, mut t_map, mut t_access) = (0f64, 0f64, 0f64);
        let mut handles = Vec::new();
        for i in 0..n_chunks {
            let off = i * CCHUNK;
            let n = CCHUNK.min(span as u64 - off);
            let t = Instant::now();
            let h = be.create(n).expect("create");
            t_create += t.elapsed().as_secs_f64();
            let t = Instant::now();
            be.map(va + off, n, h).expect("map");
            t_map += t.elapsed().as_secs_f64();
            let t = Instant::now();
            be.set_access(va + off, n).expect("set_access");
            t_access += t.elapsed().as_secs_f64();
            handles.push((va + off, n, h));
        }
        println!(
            "F split    : create {:.1} ms | map {:.1} ms | set_access {:.1} ms",
            t_create * 1e3,
            t_map * 1e3,
            t_access * 1e3
        );
        // Same, but ONE bulk set_access over the whole span after all maps.
        for (cva, n, h) in handles.drain(..) {
            be.unmap(cva, n);
            be.release(h);
        }
        let (mut t_create, mut t_map) = (0f64, 0f64);
        for i in 0..n_chunks {
            let off = i * CCHUNK;
            let n = CCHUNK.min(span as u64 - off);
            let t = Instant::now();
            let h = be.create(n).expect("create");
            t_create += t.elapsed().as_secs_f64();
            let t = Instant::now();
            be.map(va + off, n, h).expect("map");
            t_map += t.elapsed().as_secs_f64();
            handles.push((va + off, n, h));
        }
        let t = Instant::now();
        be.set_access(va, span as u64).expect("bulk set_access");
        println!(
            "F bulk     : create {:.1} ms | map {:.1} ms | bulk set_access {:.1} ms",
            t_create * 1e3,
            t_map * 1e3,
            t.elapsed().as_secs_f64() * 1e3
        );
        for (cva, n, h) in handles.drain(..) {
            be.unmap(cva, n);
            be.release(h);
        }
        be.address_free(va, span as u64);
    }

    // ---- D: parallel host memcpy (staging upper bound, no GPU) ----
    for threads in [1usize, 4, 8, 16] {
        let per = span / threads;
        let t = Instant::now();
        std::thread::scope(|sc| {
            for i in 0..threads {
                let part = &src[i * per..(i + 1) * per];
                sc.spawn(move || {
                    let mut dst = vec![0u8; CHUNK.min(part.len())];
                    for chunk in part.chunks(dst.len()) {
                        dst[..chunk.len()].copy_from_slice(chunk);
                    }
                    std::hint::black_box(&dst);
                });
            }
        });
        let s = t.elapsed().as_secs_f64();
        println!(
            "D par-copy {threads:>2}: {:>7.1} ms  {:>6.1} GiB/s",
            s * 1e3,
            gibs(span, s)
        );
    }
}
