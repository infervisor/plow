//! Safetensors checkpoint loading: mmap every `*.safetensors` shard once,
//! parse header metadata up front, and serve tensor bytes as zero-copy slices.

use std::ops::Range;
use std::path::Path;

use rustc_hash::FxHashMap;

use crate::{Result, RuntimeError};

/// A safetensors checkpoint directory: every `*.safetensors` shard mmap'd,
/// with one metadata parse per shard (name → shard/offset resolved up front,
/// tensor bytes served as zero-copy mmap slices).
pub(crate) struct Checkpoint {
    shards: Vec<(memmap2::Mmap, usize)>, // (map, data-section offset)
    /// name → where the bytes are, and what shape they are.
    index: FxHashMap<String, Entry>,
}

/// One tensor's location and shape.
///
/// The shape is kept because a ROW-parallel shard cannot be expressed without
/// it. A column shard is a contiguous byte range and needs only sizes; a row
/// shard of an `[out, in]` matrix is a *strided column range* — `in/N`
/// contiguous elements out of every one of `out` rows — and the stride IS `in`.
/// Nothing else in the byte count recovers it: `[4096, 5376]` and `[5376, 4096]`
/// have identical lengths and gather to different tensors.
struct Entry {
    shard: usize,
    range: Range<usize>,
    shape: Vec<usize>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        let t0 = std::time::Instant::now();
        tracing::info!(dir = %dir.display(), "opening safetensors checkpoint...");
        let mut shards = Vec::new();
        let mut index = FxHashMap::default();
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|source| RuntimeError::Io { path: dir.to_path_buf(), source })?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file() && p.extension().map(|e| e == "safetensors").unwrap_or(false)
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(RuntimeError::Device(format!(
                "no *.safetensors in {}",
                dir.display()
            )));
        }
        tracing::info!(shards = paths.len(), "found safetensors shards");
        for path in &paths {
            let file = std::fs::File::open(path)
                .map_err(|source| RuntimeError::Io { path: path.clone(), source })?;
            // SAFETY: read-only checkpoint mapping held for the engine lifetime.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|source| RuntimeError::Io { path: path.clone(), source })?;
            let (header_len, meta) = safetensors::SafeTensors::read_metadata(&map)
                .map_err(|e| {
                    RuntimeError::Device(format!("safetensors {}: {e}", path.display()))
                })?;
            let data_off = 8 + header_len;
            let shard = shards.len();
            let tensors = meta.tensors();
            tracing::debug!(
                shard = shard,
                file = %path.display(),
                tensors = tensors.len(),
                mib = map.len() / (1 << 20),
                "mapped checkpoint shard"
            );
            index.reserve(tensors.len());
            for (name, info) in tensors {
                index.insert(
                    name.clone(),
                    Entry {
                        shard,
                        range: info.data_offsets.0..info.data_offsets.1,
                        shape: info.shape.clone(),
                    },
                );
            }
            shards.push((map, data_off));
        }
        tracing::info!(
            shards = shards.len(),
            tensors = index.len(),
            ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3).as_str(),
            "checkpoint ready (all shards mmap'd)"
        );
        Ok(Checkpoint { shards, index })
    }

    /// Tensor bytes alone. The CUDA engine binds full tensors and wants only
    /// these; the AMD engine binds shards and needs [`Checkpoint::tensor_ex`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensor_ex(name).map(|(bytes, _)| bytes)
    }

    /// Tensor bytes **and** shape — what a row-parallel shard needs.
    pub fn tensor_ex(&self, name: &str) -> Option<(&[u8], &[usize])> {
        let e = self.index.get(name)?;
        let (map, off) = &self.shards[e.shard];
        let bytes = map.get(off + e.range.start..off + e.range.end)?;
        Some((bytes, &e.shape))
    }

    /// Locate a sub-range of a tensor as a [`Span`] the prefetcher can carry to
    /// another thread.
    ///
    /// `off`/`len` are a sub-range **of the tensor**, because a column-parallel
    /// rank binds a contiguous 1/tp slice and prefetching the whole tensor would
    /// read four times what that rank will touch.
    pub fn span(&self, name: &str, off: usize, len: usize) -> Option<Span> {
        let e = self.index.get(name)?;
        let (map, base) = &self.shards[e.shard];
        let lo = base + e.range.start + off;
        let hi = lo.saturating_add(len).min(base + e.range.end);
        if hi <= lo || hi > map.len() {
            return None;
        }
        Some(Span { shard: e.shard, off: lo, len: hi - lo })
    }

    /// Fault `span` in — page cache AND this process's page tables — and block
    /// until it is resident. Returns `false` if the kernel refused the hint.
    ///
    /// # Why `MADV_POPULATE_READ` and not `MADV_WILLNEED`
    ///
    /// Both were measured on this box, cold, on untouched 9.5 GiB shards of
    /// GLM-5.2 (`/dev/nvme0n1`, ext4 on LVM; `mincore` confirms 0 % resident
    /// before each run):
    ///
    /// | | 1 thr | 4 | 16 | 32 |
    /// |---|---|---|---|---|
    /// | mmap fault (what the loader did) | 2.52 | 2.65 | 3.13 | 2.10 GiB/s |
    /// | `MADV_WILLNEED` then fault | 2.65 | — | — | — |
    /// | `MADV_POPULATE_READ` | 2.27 | 2.31 | **5.30** | **5.34** GiB/s |
    /// | `pread` (reference ceiling) | 1.51 | 3.78 | 6.06 | 6.13 GiB/s |
    ///
    /// Two things fall out and both shaped this code:
    ///
    /// 1. **`MADV_WILLNEED` is worth nothing here** — 2.65 vs 2.49 GiB/s, inside
    ///    the noise. It submits readahead in 2 MiB chunks from the calling
    ///    thread and the toucher catches up immediately, so the queue never gets
    ///    deep. It was implemented, measured, and removed.
    /// 2. **The drive needs ~16 concurrent readers to reach its ~6 GiB/s**, and
    ///    NOTHING single-threaded gets past 2.9. That is why this is called from
    ///    a thread pool rather than inline: the depth has to come from
    ///    concurrency, and no amount of hinting from one thread supplies it.
    ///
    /// Warm it still pays — 33.8 GiB/s vs 15.9 for a userspace fault loop — for
    /// a different reason. Warm the pages are in cache but this process's PTEs
    /// are empty, and populating 44 M of them one minor fault at a time is what
    /// cost a warm rank 44 s. One syscall per 1.4 MiB does the same work in a
    /// kernel loop with no trap per page.
    ///
    /// Not `MADV_HUGEPAGE`: this box runs 5.15 with `CONFIG_READ_ONLY_THP_FOR_FS`
    /// unset, and ext4 has no large-folio page cache before 6.x, so a huge-page
    /// hint on a file-backed mapping does nothing at all here. It would also aim
    /// at the wrong term — the cold cost is device I/O, which a huge page does
    /// not reduce by one byte.
    pub fn populate(&self, span: Span) -> bool {
        let (map, _) = &self.shards[span.shard];
        map.advise_range(memmap2::Advice::PopulateRead, span.off, span.len).is_ok()
    }
}

/// A byte range of one mmap'd shard, resolved away from the name so it can be
/// handed to a prefetch thread without borrowing the [`Checkpoint`] index.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    shard: usize,
    off: usize,
    len: usize,
}

/// A pool of threads that fault checkpoint ranges in ahead of the loader.
///
/// The loader pushes the span it will read `depth` tensors from now and carries
/// on copying; by the time it gets there the pages are resident and its own
/// access is a hit. See [`Checkpoint::populate`] for the measurements that say
/// this has to be a POOL and not an inline hint.
///
/// Bounded queue, deliberately: an unbounded one would let the prefetcher run
/// arbitrarily far ahead of a slow upload and evict the very pages it read. The
/// bound is the whole back-pressure mechanism.
pub(crate) struct Prefetcher {
    tx: Option<std::sync::mpsc::SyncSender<Span>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Prefetcher {
    /// `threads = 0` disables prefetching and returns `None`, which every call
    /// site already handles — the loader then faults pages in itself, correctly
    /// and slowly, exactly as it did before.
    pub fn start(
        ckpt: std::sync::Arc<Checkpoint>,
        threads: usize,
        queue: usize,
    ) -> Option<Prefetcher> {
        if threads == 0 {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::sync_channel::<Span>(queue.max(threads));
        // `Receiver` is single-consumer; a mutex around it is what makes the
        // fan-out possible. Contention is a non-issue — one lock per 1.4 MiB of
        // I/O, against a hold time of a `VecDeque` pop.
        let rx = std::sync::Arc::new(parking_lot::Mutex::new(rx));
        let workers = (0..threads)
            .map(|_| {
                let (rx, ckpt) = (std::sync::Arc::clone(&rx), std::sync::Arc::clone(&ckpt));
                std::thread::spawn(move || {
                    loop {
                        let job = { rx.lock().recv() };
                        let Ok(span) = job else { return };
                        // A refused hint (no `MADV_POPULATE_READ` before 5.14)
                        // is not an error: the loader's own fault still gets the
                        // right bytes. Keep draining so the sender never blocks.
                        ckpt.populate(span);
                    }
                })
            })
            .collect();
        Some(Prefetcher { tx: Some(tx), workers })
    }

    /// Queue a span, blocking while the pool is `queue` jobs behind.
    pub fn push(&self, span: Span) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(span);
        }
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Dropping the sender is what ends the workers' `recv`. Joining them
        // before the `Arc<Checkpoint>` can fall is not optional: a worker is
        // inside `madvise` on a mapping the checkpoint owns.
        self.tx = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// The checkpoint's stop-token set: `generation_config.json` `eos_token_id`
/// (int or list), falling back to `config.json`, falling back to empty (the
/// caller then stops on max_tokens only).
///
/// Metadata, not tensors — so it reads the directory directly and does not need
/// the shards mmap'd. Shared by both engines: a backend that skipped this emits
/// its eos id as ordinary text and runs every request to `max_tokens`.
pub(crate) fn read_eos_ids(dir: &Path) -> Vec<u32> {
    for file in ["generation_config.json", "config.json"] {
        let Ok(bytes) = std::fs::read(dir.join(file)) else { continue };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    return vec![id as u32];
                }
            }
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> =
                    a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
                if !ids.is_empty() {
                    return ids;
                }
            }
            _ => {}
        }
    }
    Vec::new()
}

/// Stop ids a served CHAT turn needs beyond `eos_token_id`.
///
/// `generation_config.json` names the id that ends a *sequence*. Some chat formats close a
/// structured turn before they get there, and a served completion that runs to the sequence eos
/// leaks the framing into the user's text.
///
/// Kimi-K3 is the live case and the only one so far. Its turn is XTML-ish
/// (`encoding_k3.py::build_chat_segments`): the assistant opens `<|open|>response<|sep|>` and the
/// answer ends at the MATCHING `<|close|>` — after which the template goes on to emit
/// `response<|sep|><|close|>message<|sep|>` and only then `<|end_of_msg|>` (163586), which is what
/// `eos_token_id` names. Stopping only at eos returned
///
///     The capital of France is Paris.<|close|>response<|sep|><|close|>message<|sep|>
///
/// — a correct answer with four markers of channel bookkeeping stapled to it.
///
/// Keyed on the CHECKPOINT's own tokens rather than on a model name: the extra stop is added only
/// when this checkpoint both declares `<|end_of_msg|>` as its eos AND ships a `<|close|>` token,
/// which is exactly the K3 turn structure and cannot fire on a checkpoint shaped differently.
/// `serve::chat::k3_chat_prompt` renders the matching prompt; the two must stay in step, so if
/// that template ever opens the `think` channel instead, this rule needs revisiting — `<|close|>`
/// would then end the THOUGHT, not the answer.
pub(crate) fn chat_stop_ids(dir: &Path, eos: &[u32]) -> Vec<u32> {
    let Ok(bytes) = std::fs::read(dir.join("tokenizer_config.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    let Some(added) = v.get("added_tokens_decoder").and_then(|a| a.as_object()) else {
        return Vec::new();
    };
    let id_of = |want: &str| -> Option<u32> {
        added.iter().find_map(|(k, e)| {
            (e.get("content").and_then(|c| c.as_str()) == Some(want))
                .then(|| k.parse::<u32>().ok())
                .flatten()
        })
    };
    match (id_of("<|end_of_msg|>"), id_of("<|close|>")) {
        (Some(eom), Some(close)) if eos.contains(&eom) => vec![close],
        _ => Vec::new(),
    }
}
