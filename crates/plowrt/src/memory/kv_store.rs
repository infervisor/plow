//! Content-addressed KV block store (`docs/arch/26-kv-block-store.md`, Part B).
//!
//! A boundary snapshot is no longer one contiguous buffer: each **row class** (the sliding
//! windows, the full-layer partial-block tail, the whole-block scale prefix, a recurrent state)
//! is cut into fixed-size **blocks** of `rows_per_block` rows, keyed by the chained hash of the
//! tokens that produced them. Two boundaries of one conversation share every block their rows
//! have in common, so a turn publishes only its new rows; a shared system prompt's blocks are
//! shared across conversations the same way.
//!
//! The store owns the bytes and their residency (a device slab per class, an optional pinned
//! host slab per class); the boundary tree in [`super::vmm`] owns the policy (which boundary
//! to keep) and holds one **dependent** count per block per boundary. **Transfer holds** keep
//! a block's bytes in place while a copy reads them. A block is `Filling` until its scatter copy
//! completed, `Ready` once it may be read, `Failed` if the copy did not complete; only `Ready`
//! blocks are served. Blocks with no dependents go to a reclaim list in release order and are
//! reused only when a slab needs the space, so a demoted boundary can still hit.
//!
//! Model kernels never see any of this: a block is gathered into the slot's flat rings and
//! tracks before the slot runs, and scattered out of them at a boundary.

use super::vmm::VmmOps;
use crate::{Result, RuntimeError};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// The device/host operations the store needs; every [`VmmOps`] backend provides them.
pub trait StoreOps: Send + Sync {
    fn alloc(&self, bytes: u64) -> Result<u64>;
    fn free(&self, va: u64);
    fn host_alloc(&self, bytes: u64) -> Result<u64>;
    fn host_free(&self, ptr: u64, bytes: u64);
    /// Queue a D2H copy; the fence retires when it completed.
    fn copy_dtoh_async(&self, host: u64, va: u64, bytes: u64) -> Result<u64>;
    fn fence_done(&self, fence: u64) -> Result<bool>;
    /// H2D, complete on return.
    fn copy_htod(&self, va: u64, host: u64, bytes: u64) -> Result<()>;
}

impl<T: VmmOps + ?Sized> StoreOps for T {
    fn alloc(&self, bytes: u64) -> Result<u64> {
        VmmOps::alloc(self, bytes)
    }
    fn free(&self, va: u64) {
        VmmOps::free(self, va)
    }
    fn host_alloc(&self, bytes: u64) -> Result<u64> {
        VmmOps::host_tier_alloc(self, bytes)
    }
    fn host_free(&self, ptr: u64, bytes: u64) {
        VmmOps::host_tier_free(self, ptr, bytes)
    }
    fn copy_dtoh_async(&self, host: u64, va: u64, bytes: u64) -> Result<u64> {
        VmmOps::copy_dtoh_async(self, host, va, bytes)
    }
    fn fence_done(&self, fence: u64) -> Result<bool> {
        VmmOps::fence_done(self, fence)
    }
    fn copy_htod(&self, va: u64, host: u64, bytes: u64) -> Result<()> {
        VmmOps::copy_htod(self, va, host, bytes)
    }
}

/// One row class: what a row is made of and how many rows one block holds.
#[derive(Clone, Debug)]
pub struct ClassSpec {
    pub name: &'static str,
    /// Bytes of one row across every tensor of the class.
    pub row_bytes: u64,
    pub rows_per_block: u32,
}

impl ClassSpec {
    pub fn block_bytes(&self) -> u64 {
        self.row_bytes * u64::from(self.rows_per_block)
    }
}

/// A block's identity: the packet namespace, the token chain hash at the block's end, and a
/// second independent hash of the same tokens so a 64-bit chain collision between two
/// publishers cannot alias their bytes.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct BlockKey {
    pub namespace: u64,
    pub chain: u64,
    pub check: u64,
}

pub type BlockId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Filling,
    Ready,
    Failed,
}

/// Where a block's bytes are. `Spilling` keeps both slots until the fence retires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Where {
    Device(u32),
    Spilling { dev: u32, host: u32, fence: u64 },
    Host(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Device(u64),
    Host,
}

struct Block {
    class: usize,
    key: BlockKey,
    state: State,
    place: Where,
    dependents: u32,
    holds: u32,
    /// Monotonic use tick for the reclaim order.
    tick: u64,
}

/// A slab of equal-size slots on one tier. Slabs of one tier share the tier's byte budget:
/// a class grows a chunk only while the budget allows, and chunks never move between classes.
struct Slab {
    slot_bytes: u64,
    chunk_slots: u32,
    chunks: Vec<u64>,
    free: Vec<u32>,
    slots: u32,
}

impl Slab {
    fn new(slot_bytes: u64, chunk_bytes: u64) -> Self {
        let chunk_slots =
            u32::try_from((chunk_bytes / slot_bytes.max(1)).max(1)).unwrap_or(u32::MAX);
        Slab {
            slot_bytes,
            chunk_slots,
            chunks: Vec::new(),
            free: Vec::new(),
            slots: 0,
        }
    }

    fn chunk_bytes(&self) -> u64 {
        u64::from(self.chunk_slots) * self.slot_bytes
    }

    fn va(&self, slot: u32) -> u64 {
        self.chunks[(slot / self.chunk_slots) as usize]
            + u64::from(slot % self.chunk_slots) * self.slot_bytes
    }

    fn used_bytes(&self) -> u64 {
        u64::from(self.slots - self.free.len() as u32) * self.slot_bytes
    }

    /// A free slot, growing by one chunk when none is free and `budget` (bytes left on the
    /// tier) covers a chunk.
    fn take(&mut self, budget: u64, grow: impl FnOnce(u64) -> Result<u64>) -> Result<Option<u32>> {
        if let Some(s) = self.free.pop() {
            return Ok(Some(s));
        }
        let n = self.chunk_slots;
        if u64::from(n) * self.slot_bytes > budget {
            return Ok(None);
        }
        let base = grow(u64::from(n) * self.slot_bytes)?;
        self.chunks.push(base);
        let first = self.slots;
        self.slots += n;
        // Hand out the chunk's slots in order; the first goes to the caller.
        for s in (first + 1..first + n).rev() {
            self.free.push(s);
        }
        Ok(Some(first))
    }

    fn reserved_bytes(&self) -> u64 {
        self.chunks.len() as u64 * self.chunk_bytes()
    }
}

struct Class {
    spec: ClassSpec,
    device: Slab,
    host: Slab,
}

#[derive(Clone, Debug, Default)]
pub struct ClassStats {
    pub device_blocks: u64,
    pub host_blocks: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StoreStats {
    pub classes: Vec<ClassStats>,
    /// Publishes that found the block already present.
    pub dedup_hits: u64,
    pub fills: u64,
    pub fill_failures: u64,
    pub reclaims: u64,
    pub refusals: u64,
    pub spills: u64,
    pub restores: u64,
    pub evictions: u64,
}

struct Inner {
    classes: Vec<Class>,
    device_cap: u64,
    host_cap: u64,
    blocks: Vec<Option<Block>>,
    free_ids: Vec<BlockId>,
    index: FxHashMap<(usize, BlockKey), BlockId>,
    /// Zero-dependent blocks in release order (oldest first); an entry may be stale (the block
    /// was re-acquired or dropped), checked when popped.
    reclaim: VecDeque<BlockId>,
    /// Blocks with a D2H copy in flight, oldest first ([`KvStore::reap`]).
    spilling: VecDeque<BlockId>,
    tick: u64,
    stats: StoreStats,
}

pub struct KvStore {
    ops: Arc<dyn StoreOps>,
    inner: parking_lot::Mutex<Inner>,
}

/// What [`KvStore::acquire`] handed out.
#[derive(Clone, Copy, Debug)]
pub struct Acquire {
    pub id: BlockId,
    /// The device bytes to fill (`fill`) or read.
    pub va: u64,
    /// The caller must scatter the rows and then [`KvStore::seal`] or [`KvStore::fail`].
    pub fill: bool,
}

impl KvStore {
    /// `device_cap` / `host_cap`: bytes the store may reserve on each tier over every class
    /// (`host_cap == 0` = no host tier); slabs grow in `chunk_bytes` steps.
    pub fn new(
        ops: Arc<dyn StoreOps>,
        classes: Vec<ClassSpec>,
        chunk_bytes: u64,
        device_cap: u64,
        host_cap: u64,
    ) -> Self {
        let classes = classes
            .into_iter()
            .map(|spec| {
                let block = spec.block_bytes();
                Class {
                    device: Slab::new(block, chunk_bytes),
                    host: Slab::new(block, chunk_bytes),
                    spec,
                }
            })
            .collect::<Vec<_>>();
        let stats = StoreStats {
            classes: classes.iter().map(|_| ClassStats::default()).collect(),
            ..Default::default()
        };
        KvStore {
            ops,
            inner: parking_lot::Mutex::new(Inner {
                classes,
                device_cap,
                host_cap,
                blocks: Vec::new(),
                free_ids: Vec::new(),
                index: FxHashMap::default(),
                reclaim: VecDeque::new(),
                spilling: VecDeque::new(),
                tick: 0,
                stats,
            }),
        }
    }

    pub fn class_spec(&self, class: usize) -> ClassSpec {
        self.inner.lock().classes[class].spec.clone()
    }

    /// Take a dependency on the block `(class, key)`, creating it when absent. A present
    /// `Ready`/`Filling` block is returned as is (`fill == false`); an absent or `Failed` one
    /// comes back `Filling` with `fill == true`. A host-resident block is restored first.
    /// `Err` = no device slot could be made (the caller evicts a boundary and retries, or
    /// refuses the publish).
    pub fn acquire(&self, class: usize, key: BlockKey) -> Result<Acquire> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        if let Some(&id) = inner.index.get(&(class, key)) {
            let va = self.ensure_device_locked(&mut inner, id)?;
            let b = inner.blocks[id as usize].as_mut().expect("indexed block");
            b.dependents += 1;
            b.tick = tick;
            if b.state == State::Failed {
                b.state = State::Filling;
                inner.stats.fills += 1;
                return Ok(Acquire { id, va, fill: true });
            }
            inner.stats.dedup_hits += 1;
            return Ok(Acquire { id, va, fill: false });
        }
        let slot = self.device_slot(&mut inner, class)?;
        let va = inner.classes[class].device.va(slot);
        let id = match inner.free_ids.pop() {
            Some(id) => id,
            None => {
                inner.blocks.push(None);
                (inner.blocks.len() - 1) as BlockId
            }
        };
        inner.blocks[id as usize] = Some(Block {
            class,
            key,
            state: State::Filling,
            place: Where::Device(slot),
            dependents: 1,
            holds: 0,
            tick,
        });
        inner.index.insert((class, key), id);
        inner.stats.fills += 1;
        inner.stats.classes[class].device_blocks += 1;
        inner.stats.classes[class].device_bytes += inner.classes[class].spec.block_bytes();
        Ok(Acquire { id, va, fill: true })
    }

    /// The fill completed: the block may be read.
    pub fn seal(&self, id: BlockId) {
        let mut inner = self.inner.lock();
        if let Some(b) = inner.blocks[id as usize].as_mut() {
            if b.state == State::Filling {
                b.state = State::Ready;
            }
        }
    }

    /// The fill did not complete: never serve these bytes; the next acquire refills.
    pub fn fail(&self, id: BlockId) {
        let mut inner = self.inner.lock();
        if let Some(b) = inner.blocks[id as usize].as_mut() {
            b.state = State::Failed;
        }
        inner.stats.fill_failures += 1;
    }

    pub fn state(&self, id: BlockId) -> Option<State> {
        self.inner.lock().blocks[id as usize].as_ref().map(|b| b.state)
    }

    /// Drop one dependency each; a block with none left goes to the reclaim list (its bytes
    /// stay until a slab needs the slot).
    pub fn release(&self, ids: &[BlockId]) {
        let mut inner = self.inner.lock();
        for &id in ids {
            let Some(b) = inner.blocks[id as usize].as_mut() else {
                continue;
            };
            b.dependents = b.dependents.saturating_sub(1);
            if b.dependents == 0 {
                inner.reclaim.push_back(id);
            }
        }
    }

    /// A transfer is reading the block: keep its bytes in place until [`Self::unhold`].
    pub fn hold(&self, ids: &[BlockId]) {
        let mut inner = self.inner.lock();
        for &id in ids {
            if let Some(b) = inner.blocks[id as usize].as_mut() {
                b.holds += 1;
            }
        }
    }

    pub fn unhold(&self, ids: &[BlockId]) {
        let mut inner = self.inner.lock();
        for &id in ids {
            if let Some(b) = inner.blocks[id as usize].as_mut() {
                b.holds = b.holds.saturating_sub(1);
            }
        }
    }

    pub fn location(&self, id: BlockId) -> Option<Location> {
        let inner = self.inner.lock();
        let b = inner.blocks[id as usize].as_ref()?;
        Some(match b.place {
            Where::Device(s) | Where::Spilling { dev: s, .. } => {
                Location::Device(inner.classes[b.class].device.va(s))
            }
            Where::Host(_) => Location::Host,
        })
    }

    /// The block's device bytes, restoring it from the host tier when needed, and a use tick.
    /// `Err` = no device slot could be made for the restore.
    pub fn ensure_device(&self, id: BlockId) -> Result<u64> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let va = self.ensure_device_locked(&mut inner, id)?;
        if let Some(b) = inner.blocks[id as usize].as_mut() {
            b.tick = tick;
        }
        Ok(va)
    }

    fn ensure_device_locked(&self, inner: &mut Inner, id: BlockId) -> Result<u64> {
        let (class, place, state) = {
            let b = inner.blocks[id as usize]
                .as_ref()
                .ok_or_else(|| RuntimeError::Rejected("kv store: no such block".into()))?;
            (b.class, b.place, b.state)
        };
        match place {
            Where::Device(s) | Where::Spilling { dev: s, .. } => Ok(inner.classes[class].device.va(s)),
            Where::Host(h) => {
                let slot = self.device_slot(inner, class)?;
                let bytes = inner.classes[class].spec.block_bytes();
                let va = inner.classes[class].device.va(slot);
                let host = inner.classes[class].host.va(h);
                if state == State::Ready {
                    if let Err(e) = self.ops.copy_htod(va, host, bytes) {
                        inner.classes[class].device.free.push(slot);
                        return Err(e);
                    }
                }
                inner.classes[class].host.free.push(h);
                let b = inner.blocks[id as usize].as_mut().expect("block");
                b.place = Where::Device(slot);
                inner.stats.restores += 1;
                inner.stats.classes[class].host_blocks -= 1;
                inner.stats.classes[class].host_bytes -= bytes;
                inner.stats.classes[class].device_blocks += 1;
                inner.stats.classes[class].device_bytes += bytes;
                Ok(va)
            }
        }
    }

    /// A free device slot for `class`: from the free list, by growing the slab, or by
    /// reclaiming the oldest zero-dependent, unheld, settled block of the class.
    fn device_slot(&self, inner: &mut Inner, class: usize) -> Result<u32> {
        let ops = &self.ops;
        let reserved: u64 = inner.classes.iter().map(|c| c.device.reserved_bytes()).sum();
        let budget = inner.device_cap.saturating_sub(reserved);
        if let Some(s) = inner.classes[class]
            .device
            .take(budget, |bytes| ops.alloc(bytes))?
        {
            return Ok(s);
        }
        // Reclaim in release order; stale entries (re-acquired, dropped, wrong tier) are
        // skipped and re-queued where they still matter.
        let mut requeue = Vec::new();
        let mut found = None;
        while let Some(id) = inner.reclaim.pop_front() {
            let Some(b) = inner.blocks[id as usize].as_ref() else {
                continue;
            };
            if b.dependents > 0 {
                continue;
            }
            if b.holds > 0 || b.state == State::Filling || b.class != class {
                requeue.push(id);
                continue;
            }
            match b.place {
                Where::Device(s) => {
                    found = Some((id, s));
                    break;
                }
                _ => requeue.push(id),
            }
        }
        for id in requeue.into_iter().rev() {
            inner.reclaim.push_front(id);
        }
        let Some((id, slot)) = found else {
            inner.stats.refusals += 1;
            return Err(RuntimeError::Oom("kv store: class slab full".into()));
        };
        self.drop_block(inner, id);
        inner.stats.reclaims += 1;
        Ok(slot)
    }

    /// Forget a block entirely (its slots go back to their slabs).
    fn drop_block(&self, inner: &mut Inner, id: BlockId) {
        let Some(b) = inner.blocks[id as usize].take() else {
            return;
        };
        inner.index.remove(&(b.class, b.key));
        inner.free_ids.push(id);
        let bytes = inner.classes[b.class].spec.block_bytes();
        let c = &mut inner.classes[b.class];
        let st = &mut inner.stats.classes[b.class];
        match b.place {
            Where::Device(s) => {
                c.device.free.push(s);
                st.device_blocks -= 1;
                st.device_bytes -= bytes;
            }
            Where::Spilling { dev, host, .. } => {
                c.device.free.push(dev);
                c.host.free.push(host);
                st.device_blocks -= 1;
                st.device_bytes -= bytes;
            }
            Where::Host(h) => {
                c.host.free.push(h);
                st.host_blocks -= 1;
                st.host_bytes -= bytes;
            }
        }
    }

    /// Free these blocks now where nothing depends on or holds them; others stay.
    pub fn evict(&self, ids: &[BlockId]) -> u32 {
        let mut inner = self.inner.lock();
        let mut n = 0;
        for &id in ids {
            let Some(b) = inner.blocks[id as usize].as_ref() else {
                continue;
            };
            if b.dependents == 0 && b.holds == 0 && b.state != State::Filling {
                if !matches!(b.place, Where::Spilling { .. }) {
                    self.drop_block(&mut inner, id);
                    inner.stats.evictions += 1;
                    n += 1;
                }
            }
        }
        n
    }

    /// Move the device-resident, unheld, `Ready` blocks among `ids` to the host tier (D2H
    /// queued; the device slot frees when [`Self::reap`] sees the fence). Dependents may
    /// remain: an attach restores a host block ([`Self::ensure_device`]). Returns `true` when
    /// every block of `ids` is off the device afterwards (queued, already on the host, or not
    /// servable); `false` = the host tier is full, and whatever was queued stays queued.
    pub fn spill(&self, ids: &[BlockId]) -> bool {
        let mut inner = self.inner.lock();
        if inner.host_cap == 0 {
            return false;
        }
        let mut all = true;
        for &id in ids {
            let Some(b) = inner.blocks[id as usize].as_ref() else {
                continue;
            };
            let class = b.class;
            let Where::Device(dev) = b.place else {
                continue;
            };
            if b.state != State::Ready {
                continue;
            }
            if b.holds > 0 {
                all = false;
                continue;
            }
            let Some(h) = self.host_slot(&mut inner, class) else {
                all = false;
                continue;
            };
            let bytes = inner.classes[class].spec.block_bytes();
            let va = inner.classes[class].device.va(dev);
            let host = inner.classes[class].host.va(h);
            match self.ops.copy_dtoh_async(host, va, bytes) {
                Ok(fence) => {
                    let b = inner.blocks[id as usize].as_mut().expect("block");
                    b.place = Where::Spilling { dev, host: h, fence };
                    inner.spilling.push_back(id);
                    inner.stats.spills += 1;
                }
                Err(_) => {
                    inner.classes[class].host.free.push(h);
                    all = false;
                }
            }
        }
        all
    }

    fn host_slot(&self, inner: &mut Inner, class: usize) -> Option<u32> {
        let ops = &self.ops;
        let reserved: u64 = inner.classes.iter().map(|c| c.host.reserved_bytes()).sum();
        let budget = inner.host_cap.saturating_sub(reserved);
        let taken = inner.classes[class]
            .host
            .take(budget, |bytes| ops.host_alloc(bytes))
            .ok()?;
        if let Some(s) = taken {
            return Some(s);
        }
        // The oldest zero-dependent host block of this class gives up its slot.
        let victim = inner
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|b| (i as BlockId, b)))
            .filter(|(_, b)| b.class == class && b.dependents == 0 && b.holds == 0)
            .filter_map(|(id, b)| match b.place {
                Where::Host(h) => Some((b.tick, id, h)),
                _ => None,
            })
            .min()?;
        self.drop_block(inner, victim.1);
        inner.stats.evictions += 1;
        Some(victim.2)
    }

    /// Retire completed spills in order: the device slot frees, the block is host-resident. A
    /// fence that cannot be queried is treated as retired only for a block that was dropped
    /// meanwhile; otherwise the block stays `Spilling` (both copies kept, never reused).
    pub fn reap(&self) {
        let mut inner = self.inner.lock();
        while let Some(&id) = inner.spilling.front() {
            let (class, dev, host, fence) = match inner.blocks[id as usize].as_ref() {
                Some(Block {
                    class,
                    place: Where::Spilling { dev, host, fence },
                    ..
                }) => (*class, *dev, *host, *fence),
                // Dropped or restored meanwhile: nothing to reap.
                _ => {
                    inner.spilling.pop_front();
                    continue;
                }
            };
            match self.ops.fence_done(fence) {
                Ok(true) => {}
                Ok(false) => return,
                Err(_) => {
                    inner.spilling.pop_front();
                    inner.stats.fill_failures += 1;
                    continue;
                }
            }
            inner.spilling.pop_front();
            let bytes = inner.classes[class].spec.block_bytes();
            inner.classes[class].device.free.push(dev);
            let b = inner.blocks[id as usize].as_mut().expect("block");
            b.place = Where::Host(host);
            let st = &mut inner.stats.classes[class];
            st.device_blocks -= 1;
            st.device_bytes -= bytes;
            st.host_blocks += 1;
            st.host_bytes += bytes;
        }
    }

    pub fn stats(&self) -> StoreStats {
        self.inner.lock().stats.clone()
    }

    pub fn device_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        inner.classes.iter().map(|c| c.device.used_bytes()).sum()
    }
}

impl Drop for KvStore {
    fn drop(&mut self) {
        let inner = self.inner.get_mut();
        for c in &inner.classes {
            for &va in &c.device.chunks {
                self.ops.free(va);
            }
            let chunk = c.host.chunk_bytes();
            for &ptr in &c.host.chunks {
                self.ops.host_free(ptr, chunk);
            }
        }
    }
}

/// Chained 64-bit hashes of `tokens` per `rows` rows: entry `i` covers `tokens[..(i+1)*rows]`,
/// hashed with `H` from `seed`. The key's `chain` and `check` use two unrelated hash
/// functions (`FxHasher` and SipHash), not two seeds of one.
pub fn chain_hashes<H: std::hash::Hasher + Default>(tokens: &[u32], rows: u32, seed: u64) -> Vec<u64> {
    use std::hash::Hasher;
    let rows = rows.max(1) as usize;
    let mut out = Vec::with_capacity(tokens.len() / rows);
    let mut prev = seed;
    for chunk in tokens.chunks_exact(rows) {
        let mut h = H::default();
        h.write_u64(prev);
        for &t in chunk {
            h.write_u32(t);
        }
        prev = h.finish();
        out.push(prev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct MockOps {
        next: AtomicU64,
        dev_alloc: AtomicU64,
        host_alloc: AtomicU64,
        htod: AtomicU64,
    }

    impl MockOps {
        fn new() -> Arc<Self> {
            Arc::new(MockOps {
                next: AtomicU64::new(1 << 20),
                dev_alloc: AtomicU64::new(0),
                host_alloc: AtomicU64::new(0),
                htod: AtomicU64::new(0),
            })
        }
    }

    impl StoreOps for MockOps {
        fn alloc(&self, bytes: u64) -> Result<u64> {
            self.dev_alloc.fetch_add(bytes, Ordering::SeqCst);
            Ok(self.next.fetch_add(bytes + 4096, Ordering::SeqCst))
        }
        fn free(&self, _va: u64) {}
        fn host_alloc(&self, bytes: u64) -> Result<u64> {
            self.host_alloc.fetch_add(bytes, Ordering::SeqCst);
            Ok(self.next.fetch_add(bytes + 4096, Ordering::SeqCst))
        }
        fn host_free(&self, _ptr: u64, _bytes: u64) {}
        fn copy_dtoh_async(&self, _host: u64, _va: u64, _bytes: u64) -> Result<u64> {
            Ok(7)
        }
        fn fence_done(&self, _fence: u64) -> Result<bool> {
            Ok(true)
        }
        fn copy_htod(&self, _va: u64, _host: u64, _bytes: u64) -> Result<()> {
            self.htod.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn store(device_cap: u64, host_cap: u64) -> (Arc<MockOps>, KvStore) {
        let ops = MockOps::new();
        let s = KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            32,
            device_cap,
            host_cap,
        );
        (ops, s)
    }

    fn key(i: u64) -> BlockKey {
        BlockKey {
            namespace: 1,
            chain: i,
            check: i * 31,
        }
    }

    #[test]
    fn same_key_is_one_block_with_two_dependents() {
        let (_, s) = store(1024, 0);
        let a = s.acquire(0, key(1)).unwrap();
        assert!(a.fill);
        s.seal(a.id);
        let b = s.acquire(0, key(1)).unwrap();
        assert!(!b.fill);
        assert_eq!(a.id, b.id);
        assert_eq!(a.va, b.va);
        assert_eq!(s.stats().dedup_hits, 1);
        assert_eq!(s.stats().classes[0].device_blocks, 1);
    }

    #[test]
    fn check_hash_distinguishes_a_chain_collision() {
        let (_, s) = store(1024, 0);
        let a = s.acquire(0, key(1)).unwrap();
        let b = s
            .acquire(
                0,
                BlockKey {
                    namespace: 1,
                    chain: 1,
                    check: 99,
                },
            )
            .unwrap();
        assert!(b.fill && a.id != b.id);
    }

    #[test]
    fn reclaim_reuses_released_blocks_oldest_first_when_full() {
        let (_, s) = store(2 * 32, 0); // two 32-byte blocks
        let a = s.acquire(0, key(1)).unwrap();
        let b = s.acquire(0, key(2)).unwrap();
        s.seal(a.id);
        s.seal(b.id);
        assert!(s.acquire(0, key(3)).is_err(), "full, nothing released");
        assert_eq!(s.stats().refusals, 1);
        s.release(&[b.id]);
        s.release(&[a.id]);
        // Both released: b first, so b's slot goes first; a is still present (a demoted
        // boundary can still hit).
        let c = s.acquire(0, key(3)).unwrap();
        assert!(c.fill);
        assert_eq!(c.va, b.va);
        let again = s.acquire(0, key(1)).unwrap();
        assert!(!again.fill, "a survived on the reclaim list");
        assert_eq!(s.stats().reclaims, 1);
    }

    #[test]
    fn held_or_filling_blocks_are_not_reclaimed() {
        let (_, s) = store(32, 0);
        let a = s.acquire(0, key(1)).unwrap();
        s.release(&[a.id]);
        s.hold(&[a.id]);
        assert!(s.acquire(0, key(2)).is_err(), "held");
        s.unhold(&[a.id]);
        assert!(s.acquire(0, key(2)).is_err(), "still filling");
        s.seal(a.id);
        assert!(s.acquire(0, key(2)).is_ok());
    }

    #[test]
    fn failed_block_is_refilled_not_served() {
        let (_, s) = store(1024, 0);
        let a = s.acquire(0, key(1)).unwrap();
        s.fail(a.id);
        assert_eq!(s.state(a.id), Some(State::Failed));
        let b = s.acquire(0, key(1)).unwrap();
        assert!(b.fill && b.id == a.id);
        assert_eq!(s.state(a.id), Some(State::Filling));
    }

    #[test]
    fn spill_then_restore_round_trips_through_the_host_tier() {
        let (ops, s) = store(32, 64);
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert_eq!(s.stats().classes[0].host_blocks, 1);
        // The device slot is free again for another block.
        let b = s.acquire(0, key(2)).unwrap();
        assert!(b.fill);
        s.seal(b.id);
        s.release(&[b.id]);
        // Restoring a needs b's slot: b is reclaimed (zero dependents), a comes back.
        let again = s.acquire(0, key(1)).unwrap();
        assert!(!again.fill, "restored, not refilled");
        assert_eq!(ops.htod.load(Ordering::SeqCst), 1);
        assert_eq!(s.stats().restores, 1);
        assert_eq!(s.location(a.id), Some(Location::Device(again.va)));
    }

    #[test]
    fn a_live_block_spills_and_restores_on_the_next_use() {
        let (ops, s) = store(32, 64);
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        assert!(s.spill(&[a.id]), "dependents do not pin a block to the device");
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        let va = s.ensure_device(a.id).unwrap();
        assert_eq!(s.location(a.id), Some(Location::Device(va)));
        assert_eq!(ops.htod.load(Ordering::SeqCst), 1);
        s.hold(&[a.id]);
        assert!(!s.spill(&[a.id]), "a held block stays");
    }

    #[test]
    fn host_tier_off_refuses_every_spill() {
        let (_, s) = store(32, 0);
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        assert!(!s.spill(&[a.id]));
        assert_eq!(s.location(a.id), Some(Location::Device(a.va)));
    }

    #[test]
    fn evict_frees_only_unreferenced_settled_blocks() {
        let (_, s) = store(1024, 0);
        let a = s.acquire(0, key(1)).unwrap();
        let b = s.acquire(0, key(2)).unwrap();
        s.seal(a.id);
        s.seal(b.id);
        s.release(&[b.id]);
        assert_eq!(s.evict(&[a.id, b.id]), 1);
        assert_eq!(s.stats().classes[0].device_blocks, 1);
        assert!(s.acquire(0, key(1)).map(|x| !x.fill).unwrap());
    }

    #[test]
    fn chain_hashes_are_prefix_stable() {
        let t: Vec<u32> = (0..64).collect();
        let a = chain_hashes::<rustc_hash::FxHasher>(&t, 32, 0);
        let b = chain_hashes::<rustc_hash::FxHasher>(&t[..32], 32, 0);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], b[0]);
        assert_ne!(a[0], a[1]);
        assert_ne!(chain_hashes::<rustc_hash::FxHasher>(&t, 32, 1)[0], a[0]);
        let c = chain_hashes::<std::collections::hash_map::DefaultHasher>(&t, 32, 0);
        assert_ne!(c[0], a[0]);
    }
}
