//! Verify the `Container` layer against a real checkpoint and measure the
//! honest single-threaded mmap load baseline.
//!
//! ```text
//! checkpoint_scan <dir> [--read]
//! ```
//!
//! Without `--read` it only parses headers (enumerate + validate). With
//! `--read` it touches every tensor byte through the mmap and reports GB/s —
//! that is the *baseline* the parallel loader has to beat. Page-cache state is
//! reported, not assumed: a warm-cache number measures RAM, not the loader.

use std::path::PathBuf;
use std::time::Instant;

use plowrt::memory::container::{Container, Safetensors};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: checkpoint_scan <dir> [--read]"));
    let read = args.any(|a| a == "--read");

    let t0 = Instant::now();
    let c = match Safetensors::open_dir(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL {}: {e}", dir.display());
            std::process::exit(1);
        }
    };
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let total = c.total_bytes();
    let mut sizes: Vec<u64> = c.tensors().iter().map(|t| t.nbytes()).collect();
    sizes.sort_unstable();
    let median = sizes[sizes.len() / 2];
    let largest = c.tensors().iter().max_by_key(|t| t.nbytes()).unwrap();
    let mut dtypes: std::collections::BTreeMap<String, usize> = Default::default();
    for t in c.tensors() {
        *dtypes.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }

    println!("dir           {}", dir.display());
    println!("shards        {}", c.shard_paths().len());
    for (i, p) in c.shard_paths().iter().enumerate() {
        println!("  [{i}] {}", p.file_name().unwrap().to_string_lossy());
    }
    println!("tensors       {}", c.tensors().len());
    println!("dtypes        {dtypes:?}");
    println!(
        "total bytes   {total} ({:.2} GiB)",
        total as f64 / (1 << 30) as f64
    );
    println!(
        "largest       {:.1} MiB  {}  {:?}",
        largest.nbytes() as f64 / (1 << 20) as f64,
        largest.name,
        largest.shape
    );
    println!("median        {:.3} MiB", median as f64 / (1 << 20) as f64);
    let ge64: Vec<u64> = sizes.iter().copied().filter(|&s| s >= 64 << 20).collect();
    println!(
        "tensors>=64MiB {} holding {:.2} GiB ({:.1}% of bytes)",
        ge64.len(),
        ge64.iter().sum::<u64>() as f64 / (1 << 30) as f64,
        100.0 * ge64.iter().sum::<u64>() as f64 / total as f64
    );
    println!("header parse  {open_ms:.1} ms");

    if read {
        // Touch every byte via the mmap, exactly as `SafetensorsWeights` would.
        // `black_box`-free: a wrapping checksum the optimizer cannot elide.
        let t = Instant::now();
        let mut acc: u64 = 0;
        for d in c.tensors() {
            let b = c.bytes(d).unwrap_or_else(|e| panic!("{e}"));
            for chunk in b.chunks(4096) {
                acc = acc.wrapping_add(chunk[0] as u64);
            }
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "FAULT-IN      {total} bytes in {secs:.3} s = {:.2} GB/s  [checksum {acc}]",
            total as f64 / 1e9 / secs,
        );

        // What a loader actually does: copy each tensor into a reusable
        // staging buffer (stand-in for the pinned ring). Now warm, so this is
        // the memcpy/RAM ceiling, not the disk path.
        let cap = c.tensors().iter().map(|t| t.nbytes()).max().unwrap() as usize;
        let mut staging = vec![0u8; cap];
        let t = Instant::now();
        for d in c.tensors() {
            let b = c.bytes(d).unwrap();
            staging[..b.len()].copy_from_slice(b);
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "COPY (warm)   {total} bytes in {secs:.3} s = {:.2} GB/s  [last byte {}]",
            total as f64 / 1e9 / secs,
            staging[0]
        );
    }
}
