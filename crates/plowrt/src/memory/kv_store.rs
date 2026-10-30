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
use std::collections::{BTreeSet, VecDeque};
use std::marker::PhantomData;
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
    /// Queue an H2D copy; the fence retires when it completed.
    fn copy_htod_async(&self, va: u64, host: u64, bytes: u64) -> Result<u64>;
    /// D2D copies `(dst, src, bytes)`, all complete on return.
    fn copy_dtod_batch(&self, copies: &[(u64, u64, u64)]) -> Result<()>;
    /// Free device bytes now; `None` where the backend cannot say.
    fn free_bytes(&self) -> Option<u64>;
    /// Every queued copy has completed (`VmmOps::quiesce`).
    fn quiesce(&self) -> Result<()> {
        Ok(())
    }
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
    fn copy_htod_async(&self, va: u64, host: u64, bytes: u64) -> Result<u64> {
        VmmOps::copy_htod_async(self, va, host, bytes)
    }
    fn copy_dtod_batch(&self, copies: &[(u64, u64, u64)]) -> Result<()> {
        VmmOps::copy_dtod_batch_sync(self, copies)
    }
    fn free_bytes(&self) -> Option<u64> {
        VmmOps::free_bytes(self)
    }
    fn quiesce(&self) -> Result<()> {
        VmmOps::quiesce(self)
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

/// A block's identity: its index in the store's block table and the entry's generation when
/// the block was created. The generation moves on when the block is dropped, so an id kept
/// past that (a reclaim-queue entry, a hold, a boundary) names nothing: the next block in the
/// entry has a new id. Distinct from slot, chunk and class indices and from the pool's `u64`
/// boundary handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId {
    index: u32,
    gen: u32,
}

impl BlockId {
    #[inline]
    fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Filling,
    Ready,
    Failed,
}

/// Where a block's bytes are. `Spilling` and `Restoring` keep both slots until the fence
/// retires; a restoring block's device slot is complete only behind its fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Where {
    Device(Slot<Dev>),
    Spilling {
        dev: Slot<Dev>,
        host: Slot<Host>,
        fence: u64,
    },
    Host(Slot<Host>),
    Restoring {
        dev: Slot<Dev>,
        host: Slot<Host>,
        fence: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Device(u64),
    Host,
}

/// Where a gather reads a block ([`KvStore::gather_src`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Src {
    pub va: u64,
    /// The host bytes in place (the tier is device-mapped).
    pub from_host: bool,
    /// The device bytes are complete only behind this copy fence: the reading stream waits on
    /// it ([`StoreOps::copy_htod_async`]).
    pub fence: Option<u64>,
}

/// A compaction in flight ([`KvStore::compaction_plan`]): `moves` are `(block, from, to)`
/// with a transfer hold on each block and `to` taken from its slab.
struct CompactionPlan {
    class: usize,
    moves: Vec<(BlockId, Slot<Dev>, Slot<Dev>)>,
    copies: Vec<(u64, u64, u64)>,
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

/// One block-table entry: the generation its current or next block carries.
struct Entry {
    gen: u32,
    block: Option<Block>,
}

/// The tier a slab, its slots and its chunks are on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dev;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Host;

/// A slot of one tier's slab: a device slot cannot be handed to the host slab or the other
/// way round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot<T>(u32, PhantomData<T>);

/// A chunk's index in its slab (stays through release: slot arithmetic is unchanged).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ChunkIx(u32);

/// A chunk of one tier's bytes the store owns. Move-only: the one call that frees it takes it
/// by value, so a chunk is freed once and a released chunk cannot be read again.
struct Chunk<T> {
    va: u64,
    bytes: u64,
    _tier: PhantomData<T>,
}

impl<T> Chunk<T> {
    fn new(va: u64, bytes: u64) -> Self {
        Chunk {
            va,
            bytes,
            _tier: PhantomData,
        }
    }
}

/// Device chunk frees and their wall time, process-wide (`PLOW_PF_PACKLOG` attribution).
pub static DEVICE_FREES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DEVICE_FREE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Chunk<Dev> {
    fn free(self, ops: &dyn StoreOps) {
        let t = std::time::Instant::now();
        ops.free(self.va);
        DEVICE_FREES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DEVICE_FREE_US.fetch_add(
            t.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

impl Chunk<Host> {
    fn free(self, ops: &dyn StoreOps) {
        ops.host_free(self.va, self.bytes);
    }
}

/// A slab of equal-size slots on one tier. Slabs of one tier share the tier's byte budget:
/// a class grows a chunk only while the budget allows, and chunks never move between classes.
struct Slab<T> {
    slot_bytes: u64,
    chunk_slots: u32,
    /// `None` = released (the index stays, so slot arithmetic is unchanged).
    chunks: Vec<Option<Chunk<T>>>,
    /// Free slots per chunk.
    free: Vec<Vec<Slot<T>>>,
    /// Chunks with free slots as `(free slots, chunk)`: the fullest first, the emptiest last.
    by_free: BTreeSet<(u32, ChunkIx)>,
    /// The block in each slot.
    owner: Vec<Option<BlockId>>,
    /// Released chunk indices, reused by the next growth.
    dead: Vec<ChunkIx>,
    live_chunks: u32,
    free_slots: u64,
}

impl<T: Copy> Slab<T> {
    fn new(slot_bytes: u64, chunk_bytes: u64) -> Self {
        let chunk_slots =
            u32::try_from((chunk_bytes / slot_bytes.max(1)).max(1)).unwrap_or(u32::MAX);
        Slab {
            slot_bytes,
            chunk_slots,
            chunks: Vec::new(),
            free: Vec::new(),
            by_free: BTreeSet::new(),
            owner: Vec::new(),
            dead: Vec::new(),
            live_chunks: 0,
            free_slots: 0,
        }
    }

    fn chunk_bytes(&self) -> u64 {
        u64::from(self.chunk_slots) * self.slot_bytes
    }

    fn va(&self, slot: Slot<T>) -> u64 {
        self.chunks[self.chunk_of(slot).0 as usize]
            .as_ref()
            .expect("a handed-out slot's chunk is live")
            .va
            + u64::from(slot.0 % self.chunk_slots) * self.slot_bytes
    }

    fn reserved_bytes(&self) -> u64 {
        u64::from(self.live_chunks) * self.chunk_bytes()
    }

    fn used_bytes(&self) -> u64 {
        self.reserved_bytes() - self.free_slots * self.slot_bytes
    }

    /// A free slot from the fullest chunk that has one, so that emptier chunks keep draining
    /// towards release; a new chunk when none is free and `budget` (bytes left on the tier)
    /// covers it.
    fn take(
        &mut self,
        budget: u64,
        grow: impl FnOnce(u64) -> Result<Chunk<T>>,
    ) -> Result<Option<Slot<T>>> {
        if let Some(&(n, c)) = self.by_free.iter().next() {
            self.by_free.remove(&(n, c));
            let s = self.free[c.0 as usize].pop().expect("counted free slot");
            if n > 1 {
                self.by_free.insert((n - 1, c));
            }
            self.free_slots -= 1;
            return Ok(Some(s));
        }
        let n = self.chunk_slots;
        if u64::from(n) * self.slot_bytes > budget {
            return Ok(None);
        }
        let chunk = grow(u64::from(n) * self.slot_bytes)?;
        let c = match self.dead.pop() {
            Some(c) => {
                self.chunks[c.0 as usize] = Some(chunk);
                c
            }
            None => {
                self.chunks.push(Some(chunk));
                self.free.push(Vec::new());
                ChunkIx(self.chunks.len() as u32 - 1)
            }
        };
        let first = c.0 * n;
        // The chunk's slots go out in order; the first to the caller.
        self.free[c.0 as usize] = (first + 1..first + n)
            .rev()
            .map(|s| Slot(s, PhantomData))
            .collect();
        if self.owner.len() < (first + n) as usize {
            self.owner.resize((first + n) as usize, None);
        }
        if n > 1 {
            self.by_free.insert((n - 1, c));
        }
        self.free_slots += u64::from(n - 1);
        self.live_chunks += 1;
        Ok(Some(Slot(first, PhantomData)))
    }

    /// A free slot in any chunk but `not`, the fullest first; no growth.
    fn take_outside(&mut self, not: ChunkIx) -> Option<Slot<T>> {
        let &(n, c) = self.by_free.iter().find(|&&(_, c)| c != not)?;
        self.by_free.remove(&(n, c));
        let s = self.free[c.0 as usize].pop().expect("counted free slot");
        if n > 1 {
            self.by_free.insert((n - 1, c));
        }
        self.free_slots -= 1;
        Some(s)
    }

    fn holes_outside(&self, not: ChunkIx) -> u64 {
        self.free_slots - self.free[not.0 as usize].len() as u64
    }

    fn chunk_of(&self, slot: Slot<T>) -> ChunkIx {
        ChunkIx(slot.0 / self.chunk_slots)
    }

    fn assign(&mut self, slot: Slot<T>, id: BlockId) {
        self.owner[slot.0 as usize] = Some(id);
    }

    /// Take `slot` out of use for good: never handed out again, so its chunk is never released.
    fn leak(&mut self, slot: Slot<T>) {
        self.owner[slot.0 as usize] = None;
    }

    /// Return `slot` to its chunk.
    fn put(&mut self, slot: Slot<T>) {
        self.owner[slot.0 as usize] = None;
        let c = self.chunk_of(slot);
        let n = self.free[c.0 as usize].len() as u32;
        if n > 0 {
            self.by_free.remove(&(n, c));
        }
        self.free[c.0 as usize].push(slot);
        self.by_free.insert((n + 1, c));
        self.free_slots += 1;
    }

    /// Release every chunk whose slots are all free (the chunk index stays, dead, so slot
    /// arithmetic is unchanged) to `free`. Returns the bytes released.
    fn release_empty_chunks(&mut self, mut free: impl FnMut(Chunk<T>)) -> u64 {
        let n = self.chunk_slots;
        let mut released = 0;
        while let Some(&(count, c)) = self.by_free.iter().next_back() {
            if count < n {
                break;
            }
            self.by_free.remove(&(count, c));
            self.free[c.0 as usize].clear();
            self.free_slots -= u64::from(n);
            free(
                self.chunks[c.0 as usize]
                    .take()
                    .expect("counted live chunk"),
            );
            self.dead.push(c);
            self.live_chunks -= 1;
            released += self.chunk_bytes();
        }
        released
    }

    /// Every live chunk, the ones with the most free slots first, then the full ones.
    fn release_candidates(&self) -> Vec<ChunkIx> {
        let mut out: Vec<ChunkIx> = self.by_free.iter().rev().map(|&(_, c)| c).collect();
        out.extend(
            (0..self.chunks.len())
                .filter(|&c| self.chunks[c].is_some() && self.free[c].is_empty())
                .map(|c| ChunkIx(c as u32)),
        );
        out
    }

    fn occupants(&self, chunk: ChunkIx) -> Vec<BlockId> {
        let first = (chunk.0 * self.chunk_slots) as usize;
        self.owner[first..first + self.chunk_slots as usize]
            .iter()
            .flatten()
            .copied()
            .collect()
    }
}

struct Class {
    spec: ClassSpec,
    device: Slab<Dev>,
    host: Slab<Host>,
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
    /// Publishes that found their block on the host and refilled it on the device instead.
    pub refills: u64,
    pub refusals: u64,
    pub spills: u64,
    pub restores: u64,
    /// Blocks a gather read from the host tier in place.
    pub host_reads: u64,
    pub evictions: u64,
    /// Live blocks moved out of a chunk so that it could be released (`shrink`).
    pub compactions: u64,
    pub compact_bytes: u64,
    /// Device bytes reserved in chunks (used + holes).
    pub device_reserved_bytes: u64,
    /// Chunk allocations per tier and the wall time they took (the pool lock is held).
    pub device_grows: u64,
    pub device_grow_us: u64,
    pub host_grows: u64,
    pub host_grow_us: u64,
    /// Wall time in `spill` and `shrink` (both run under the pool lock), and shrink calls.
    pub spill_us: u64,
    pub shrink_us: u64,
    pub shrinks: u64,
    /// The longest single host chunk allocation and shrink.
    pub host_grow_max_us: u64,
    pub shrink_max_us: u64,
}

struct Inner {
    classes: Vec<Class>,
    device_cap: u64,
    /// Device bytes the store leaves free: growth asks the backend (every `DEVICE_FLOOR_EVERY`
    /// chunks) and stops while a chunk would take free memory under it. 0 = unchecked. The pool
    /// shrinks the store when the slots need memory; without this check the store grew right
    /// back to its cap and the two thrashed (58k chunk allocations, 66 GB compacted in 10 min).
    device_floor: u64,
    /// Until when `device_slot` takes the class reclaim list before growing: set by a `shrink`
    /// that released bytes. The floor check alone left a 51 MiB band between the store's growth and the
    /// pool's trim, so every ring the pool retired was re-taken within seconds and handed back
    /// at the next prompt burst (16k chunk frees per 300 s; a `cuMemFree` may synchronise the
    /// context, and seconds with a free burst had decode ticks stretched 20 -> 75 ms).
    grow_hold_until: Option<std::time::Instant>,
    host_cap: u64,
    /// Host RAM the tier leaves to the rest of the machine; growth stops (the cap freezes at
    /// what is reserved) once `MemAvailable` would fall under it. 0 = unchecked.
    host_reserve: u64,
    blocks: Vec<Entry>,
    /// Table entries with no block, reused by the next acquire.
    free_entries: Vec<u32>,
    index: FxHashMap<(usize, BlockKey), BlockId>,
    /// Zero-dependent device blocks per class in release order (oldest first); an entry may be
    /// stale (the block was re-acquired, dropped or spilled), checked when popped.
    reclaim: Vec<VecDeque<BlockId>>,
    /// The same for host-resident blocks (spilled with no dependent, or released on the host).
    host_reclaim: Vec<VecDeque<BlockId>>,
    /// Blocks with a D2H copy in flight, oldest first, with their bytes ([`KvStore::reap`]).
    spilling: VecDeque<(BlockId, u64)>,
    /// Bytes of those copies: device memory the pool may count as coming free.
    spilling_bytes: u64,
    /// Blocks with an H2D copy in flight, oldest first ([`KvStore::restore_async`]).
    restoring: VecDeque<BlockId>,
    /// Host slots of blocks being refilled on the device, kept until the fill seals: a failed
    /// refill goes back to its host bytes, which other boundaries still depend on.
    refilling: FxHashMap<BlockId, Slot<Host>>,
    /// Device slots leaked under a copy whose completion is unproven ([`KvStore::quarantine_ids`]):
    /// nonzero = the device chunks are never freed.
    quarantined: u64,
    /// Pinned chunks per class allocated ahead by [`KvStore::host_grow_step`] (a pinned
    /// allocation costs ~4 ms per 8 MiB; under the pool lock it was the host tier's TTFT tail).
    host_spare: Vec<Vec<Chunk<Host>>>,
    tick: u64,
    stats: StoreStats,
}

impl Inner {
    /// The block `id` names; `None` once it was dropped, whatever its entry holds now.
    #[inline]
    fn block(&self, id: BlockId) -> Option<&Block> {
        let e = &self.blocks[id.index()];
        if e.gen != id.gen {
            return None;
        }
        e.block.as_ref()
    }

    #[inline]
    fn block_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        let e = &mut self.blocks[id.index()];
        if e.gen != id.gen {
            return None;
        }
        e.block.as_mut()
    }
}

pub struct KvStore {
    ops: Arc<dyn StoreOps>,
    inner: parking_lot::Mutex<Inner>,
    /// Chunks `shrink` released, freed by [`Self::free_step`] on a worker once
    /// [`Self::defer_frees`] was called: a device free synchronises the context, which behind
    /// the spill backlog held the engine thread 150 ms per shrink. Until freed they count as
    /// device memory coming free ([`Self::pending_spill_bytes`]).
    frees: parking_lot::Mutex<Vec<Chunk<Dev>>>,
    deferred: std::sync::atomic::AtomicBool,
    pending_free_bytes: std::sync::atomic::AtomicU64,
}

/// What [`KvStore::acquire`] handed out.
pub struct Acquired<'s> {
    pub id: BlockId,
    /// The device bytes to fill or read.
    pub va: u64,
    /// `Some` = the block is `Filling` and this caller scatters its rows.
    pub fill: Option<FillToken<'s>>,
}

/// A block this caller must fill: scatter the rows, then [`Self::seal`]. A token dropped
/// unsealed (an early return, a failed copy) fails the block, exactly as [`Self::fail`]; a
/// token dropped by an unwind leaves it `Filling` (its slot stays pinned: a copy submitted
/// before the panic may still write it).
#[must_use = "an unsealed fill fails its block"]
pub struct FillToken<'s> {
    store: &'s KvStore,
    id: BlockId,
    done: bool,
}

impl FillToken<'_> {
    pub fn id(&self) -> BlockId {
        self.id
    }

    /// The fill completed: the block may be read.
    pub fn seal(mut self) {
        self.done = true;
        self.store.seal_id(self.id);
    }

    /// The fill's copy is queued on the stream every reader of the block's bytes before the seal
    /// is ordered on: the caller seals ([`KvStore::seal_ids`]) or fails ([`KvStore::fail_ids`])
    /// the block once that copy is known to have completed or failed. Until then it stays
    /// `Filling`: never spilled, reclaimed, evicted or compacted.
    pub fn defer(mut self) -> BlockId {
        self.done = true;
        self.id
    }

    /// The fill did not complete: never serve these bytes; the next acquire refills.
    pub fn fail(mut self) {
        self.done = true;
        self.store.fail_id(self.id);
    }

    /// The fill's copy may still be running: [`KvStore::quarantine_ids`].
    pub fn quarantine(mut self) {
        self.done = true;
        self.store.quarantine_ids(&[self.id]);
    }
}

impl Drop for FillToken<'_> {
    fn drop(&mut self) {
        if !self.done && !std::thread::panicking() {
            self.store.fail_id(self.id);
        }
    }
}

/// Transfer holds on a set of blocks ([`KvStore::hold`]): their bytes stay in place until the
/// guard drops. Dropped by an unwind the holds stay (a copy submitted before the panic may
/// still read the blocks).
#[must_use = "dropping the guard releases the holds"]
pub struct Hold {
    store: Arc<KvStore>,
    ids: Vec<BlockId>,
}

impl Hold {
    pub fn ids(&self) -> &[BlockId] {
        &self.ids
    }

    /// Keep the holds: the blocks are left pinned (teardown, where their copies' completion
    /// is unproven).
    pub fn abandon(mut self) {
        self.ids.clear();
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.store.unhold(&self.ids);
        }
    }
}

/// Host chunks allocated between two `MemAvailable` reads of the host-reserve check.
const HOST_FLOOR_EVERY: u64 = 16;
/// Device chunks allocated between two free-memory queries of the device-floor check.
const DEVICE_FLOOR_EVERY: u64 = 8;
/// Most bytes one `shrink` call releases (lock hold bound; the caller loops).
const SHRINK_STEP_BYTES: u64 = 128 << 20;
/// How long after a pressure shrink the store reclaims its own LRU before growing again
/// (`Inner::grow_hold_until`); the pool's prompt bursts last seconds, its think-time period
/// is tens of seconds.
const GROW_HOLD: std::time::Duration = std::time::Duration::from_secs(10);
/// Pinned bytes per class the host grower keeps allocated ahead (at least one chunk): a burst
/// of evictions spills several boundaries before the grower refills, a spill with no spare
/// ready is refused and the boundary dropped instead.
const HOST_SPARE_BYTES: u64 = 1 << 30;

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
        let n_classes = classes.len();
        let stats = StoreStats {
            classes: classes.iter().map(|_| ClassStats::default()).collect(),
            ..Default::default()
        };
        KvStore {
            ops,
            inner: parking_lot::Mutex::new(Inner {
                classes,
                device_cap,
                device_floor: 0,
                grow_hold_until: None,
                host_cap,
                host_reserve: 0,
                blocks: Vec::new(),
                free_entries: Vec::new(),
                index: FxHashMap::default(),
                reclaim: (0..n_classes).map(|_| VecDeque::new()).collect(),
                host_reclaim: (0..n_classes).map(|_| VecDeque::new()).collect(),
                spilling: VecDeque::new(),
                spilling_bytes: 0,
                restoring: VecDeque::new(),
                refilling: FxHashMap::default(),
                quarantined: 0,
                host_spare: (0..n_classes).map(|_| Vec::new()).collect(),
                tick: 0,
                stats,
            }),
            frees: parking_lot::Mutex::new(Vec::new()),
            deferred: std::sync::atomic::AtomicBool::new(false),
            pending_free_bytes: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// From now on `shrink` queues released chunks for [`Self::free_step`] instead of freeing
    /// them on the caller's thread.
    pub fn defer_frees(&self) {
        self.deferred
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// One step of the free worker: free every queued chunk. `false` = nothing queued.
    pub fn free_step(&self) -> bool {
        let queued = std::mem::take(&mut *self.frees.lock());
        if queued.is_empty() {
            return false;
        }
        // The bytes count as pending until each free completed, not until it was taken.
        for chunk in queued {
            let bytes = chunk.bytes;
            chunk.free(&*self.ops);
            self.pending_free_bytes
                .fetch_sub(bytes, std::sync::atomic::Ordering::AcqRel);
        }
        true
    }

    fn free_chunks(&self, frees: &mut Vec<Chunk<Dev>>) {
        if frees.is_empty() {
            return;
        }
        if self.deferred.load(std::sync::atomic::Ordering::Acquire) {
            let bytes: u64 = frees.iter().map(|c| c.bytes).sum();
            self.pending_free_bytes
                .fetch_add(bytes, std::sync::atomic::Ordering::AcqRel);
            self.frees.lock().append(frees);
        } else {
            for chunk in frees.drain(..) {
                chunk.free(&*self.ops);
            }
        }
    }

    pub fn class_spec(&self, class: usize) -> ClassSpec {
        self.inner.lock().classes[class].spec.clone()
    }

    /// Take a dependency on the block `(class, key)`, creating it when absent. A present
    /// `Ready`/`Filling` block is returned as is (no fill token); an absent or `Failed` one
    /// comes back `Filling` with a token. A host-resident block is restored first.
    /// `Err` = no device slot could be made (the caller evicts a boundary and retries, or
    /// refuses the publish).
    pub fn acquire(&self, class: usize, key: BlockKey) -> Result<Acquired<'_>> {
        // The token is built with the lock released: its drop takes the lock.
        let (id, va, fill) = self.acquire_locked(class, key)?;
        Ok(Acquired {
            id,
            va,
            // `then`, not `then_some`: an eagerly built token would fail a dedup hit on drop.
            fill: fill.then(|| FillToken {
                store: self,
                id,
                done: false,
            }),
        })
    }

    fn acquire_locked(&self, class: usize, key: BlockKey) -> Result<(BlockId, u64, bool)> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        if let Some(&id) = inner.index.get(&(class, key)) {
            // A host-resident block is refilled from the rows being published (one more segment
            // of the publish's launch), never copied back: the host slot goes once the fill
            // seals.
            // A held host block is being read in place (a gather in flight): it stays where it
            // is and the publish takes it as present.
            let (refill, held_host) = match inner.block(id).map(|b| (b.place, b.holds)) {
                Some((Where::Host(h), 0)) => (Some(h), None),
                Some((Where::Host(h), _)) => (None, Some(h)),
                _ => (None, None),
            };
            let va = match (refill, held_host) {
                (None, Some(h)) => inner.classes[class].host.va(h),
                (Some(h), _) => {
                    let slot = self.device_slot(&mut inner, class)?;
                    let bytes = inner.classes[class].spec.block_bytes();
                    inner.classes[class].device.assign(slot, id);
                    inner.refilling.insert(id, h);
                    let st = &mut inner.stats.classes[class];
                    st.host_blocks -= 1;
                    st.host_bytes -= bytes;
                    st.device_blocks += 1;
                    st.device_bytes += bytes;
                    let b = inner.block_mut(id).expect("indexed block");
                    b.place = Where::Device(slot);
                    b.state = State::Filling;
                    inner.classes[class].device.va(slot)
                }
                (None, None) => self.ensure_device_locked(&mut inner, id)?,
            };
            let b = inner.block_mut(id).expect("indexed block");
            b.dependents += 1;
            b.tick = tick;
            if b.state == State::Filling && refill.is_some() {
                inner.stats.fills += 1;
                inner.stats.refills += 1;
                return Ok((id, va, true));
            }
            if b.state == State::Failed {
                b.state = State::Filling;
                inner.stats.fills += 1;
                return Ok((id, va, true));
            }
            inner.stats.dedup_hits += 1;
            return Ok((id, va, false));
        }
        let slot = self.device_slot(&mut inner, class)?;
        let va = inner.classes[class].device.va(slot);
        let index = match inner.free_entries.pop() {
            Some(index) => index,
            None => {
                inner.blocks.push(Entry {
                    gen: 0,
                    block: None,
                });
                (inner.blocks.len() - 1) as u32
            }
        };
        let e = &mut inner.blocks[index as usize];
        let id = BlockId { index, gen: e.gen };
        e.block = Some(Block {
            class,
            key,
            state: State::Filling,
            place: Where::Device(slot),
            dependents: 1,
            holds: 0,
            tick,
        });
        inner.index.insert((class, key), id);
        inner.classes[class].device.assign(slot, id);
        inner.stats.fills += 1;
        inner.stats.classes[class].device_blocks += 1;
        inner.stats.classes[class].device_bytes += inner.classes[class].spec.block_bytes();
        Ok((id, va, true))
    }

    fn seal_id(&self, id: BlockId) {
        let mut inner = self.inner.lock();
        if let Some(b) = inner.block_mut(id) {
            if b.state == State::Filling {
                b.state = State::Ready;
            }
        }
        if let Some(h) = inner.refilling.remove(&id) {
            let class = inner.block(id).map(|b| b.class);
            if let Some(class) = class {
                inner.classes[class].host.put(h);
            }
        }
    }

    /// A failed refill is back on the host, `Ready`: its host bytes never stopped being valid
    /// and its other dependents keep reading them. Any other failed block is `Failed`.
    fn fail_id(&self, id: BlockId) {
        let mut inner = self.inner.lock();
        inner.stats.fill_failures += 1;
        let refill = inner.refilling.remove(&id);
        let Some(b) = inner.block_mut(id) else {
            return;
        };
        let (class, place, idle) = (b.class, b.place, b.dependents == 0);
        match (refill, place) {
            (Some(h), Where::Device(slot)) => {
                b.place = Where::Host(h);
                b.state = State::Ready;
                if idle {
                    inner.host_reclaim[class].push_back(id);
                }
                let bytes = inner.classes[class].spec.block_bytes();
                inner.classes[class].device.put(slot);
                let st = &mut inner.stats.classes[class];
                st.device_blocks -= 1;
                st.device_bytes -= bytes;
                st.host_blocks += 1;
                st.host_bytes += bytes;
            }
            _ => b.state = State::Failed,
        }
    }

    pub fn state(&self, id: BlockId) -> Option<State> {
        self.inner.lock().block(id).map(|b| b.state)
    }

    /// Drop one dependency each; a block with none left goes to the reclaim list (its bytes
    /// stay until a slab needs the slot).
    pub fn release(&self, ids: &[BlockId]) {
        let mut inner = self.inner.lock();
        for &id in ids {
            let Some(b) = inner.block_mut(id) else {
                continue;
            };
            b.dependents = b.dependents.saturating_sub(1);
            if b.dependents == 0 {
                let class = b.class;
                match b.place {
                    Where::Host(_) => inner.host_reclaim[class].push_back(id),
                    _ => inner.reclaim[class].push_back(id),
                }
            }
        }
    }

    /// A transfer is reading the blocks: keep their bytes in place while the guard lives.
    pub fn hold(self: &Arc<Self>, ids: Vec<BlockId>) -> Hold {
        {
            let mut inner = self.inner.lock();
            for &id in &ids {
                if let Some(b) = inner.block_mut(id) {
                    b.holds += 1;
                }
            }
        }
        Hold {
            store: Arc::clone(self),
            ids,
        }
    }

    /// Seal fills deferred by [`FillToken::defer`].
    pub fn seal_ids(&self, ids: &[BlockId]) {
        for &id in ids {
            self.seal_id(id);
        }
    }

    /// Fills whose copy may still be writing their device slots (its completion could not be
    /// proven): never served, and the slot is never reused or freed. A refill goes back to its
    /// host bytes; any other block is forgotten, so the next acquire fills a fresh slot.
    pub fn quarantine_ids(&self, ids: &[BlockId]) {
        let mut inner = self.inner.lock();
        for &id in ids {
            let Some(b) = inner.block(id) else {
                continue;
            };
            let (class, idle) = (b.class, b.dependents == 0);
            let Where::Device(slot) = b.place else {
                // Fills target device slots only.
                inner.block_mut(id).expect("block").state = State::Failed;
                continue;
            };
            inner.stats.fill_failures += 1;
            inner.quarantined += 1;
            inner.classes[class].device.leak(slot);
            let bytes = inner.classes[class].spec.block_bytes();
            let refill = inner.refilling.remove(&id);
            let st = &mut inner.stats.classes[class];
            st.device_blocks -= 1;
            st.device_bytes -= bytes;
            if let Some(h) = refill {
                st.host_blocks += 1;
                st.host_bytes += bytes;
                let b = inner.block_mut(id).expect("block");
                b.place = Where::Host(h);
                b.state = State::Ready;
                if idle {
                    inner.host_reclaim[class].push_back(id);
                }
                continue;
            }
            let e = &mut inner.blocks[id.index()];
            let b = e.block.take().expect("block");
            if let Some(gen) = e.gen.checked_add(1) {
                e.gen = gen;
                inner.free_entries.push(id.index);
            }
            inner.index.remove(&(b.class, b.key));
        }
    }

    /// Fail fills deferred by [`FillToken::defer`].
    pub fn fail_ids(&self, ids: &[BlockId]) {
        for &id in ids {
            self.fail_id(id);
        }
    }

    /// [`Self::hold`] plus [`Self::gather_src`] of every block, under one lock acquisition: a
    /// gather no longer takes the store lock once per block (each acquisition could wait
    /// behind a spill or reap). `Err` (a block is gone) takes no hold.
    pub fn hold_sources(self: &Arc<Self>, ids: Vec<BlockId>, out: &mut Vec<Src>) -> Result<Hold> {
        out.clear();
        let mut inner = self.inner.lock();
        if ids.iter().any(|&id| inner.block(id).is_none()) {
            return Err(RuntimeError::Rejected("kv store: no such block".into()));
        }
        // A deferred fill whose copy failed (`fail_ids`) is never served.
        if ids
            .iter()
            .any(|&id| inner.block(id).is_some_and(|b| b.state == State::Failed))
        {
            return Err(RuntimeError::Rejected("kv store: block failed".into()));
        }
        inner.tick += 1;
        let tick = inner.tick;
        for &id in &ids {
            let b = inner.block_mut(id).expect("checked above");
            b.holds += 1;
            b.tick = tick;
            let (class, place) = (b.class, b.place);
            out.push(match place {
                Where::Device(s) | Where::Spilling { dev: s, .. } => Src {
                    va: inner.classes[class].device.va(s),
                    from_host: false,
                    fence: None,
                },
                Where::Restoring { dev, fence, .. } => Src {
                    va: inner.classes[class].device.va(dev),
                    from_host: false,
                    fence: Some(fence),
                },
                Where::Host(h) => {
                    inner.stats.host_reads += 1;
                    Src {
                        va: inner.classes[class].host.va(h),
                        from_host: true,
                        fence: None,
                    }
                }
            });
        }
        drop(inner);
        Ok(Hold {
            store: Arc::clone(self),
            ids,
        })
    }

    fn unhold(&self, ids: &[BlockId]) {
        let mut inner = self.inner.lock();
        for &id in ids {
            if let Some(b) = inner.block_mut(id) {
                b.holds = b.holds.saturating_sub(1);
            }
        }
    }

    pub fn location(&self, id: BlockId) -> Option<Location> {
        let inner = self.inner.lock();
        let b = inner.block(id)?;
        Some(match b.place {
            Where::Device(s) | Where::Spilling { dev: s, .. } | Where::Restoring { dev: s, .. } => {
                Location::Device(inner.classes[b.class].device.va(s))
            }
            Where::Host(_) => Location::Host,
        })
    }

    /// Where a gather reads the block: its device bytes (behind the restore fence while one is
    /// in flight), or its host bytes in place when it is host-resident. Nothing moves; a later
    /// publish that meets a host block refills it on the device. The caller holds the block
    /// meanwhile.
    pub fn gather_src(&self, id: BlockId) -> Result<Src> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let b = inner
            .block_mut(id)
            .ok_or_else(|| RuntimeError::Rejected("kv store: no such block".into()))?;
        b.tick = tick;
        let (class, place) = (b.class, b.place);
        Ok(match place {
            Where::Device(s) | Where::Spilling { dev: s, .. } => Src {
                va: inner.classes[class].device.va(s),
                from_host: false,
                fence: None,
            },
            Where::Restoring { dev, fence, .. } => Src {
                va: inner.classes[class].device.va(dev),
                from_host: false,
                fence: Some(fence),
            },
            Where::Host(h) => {
                inner.stats.host_reads += 1;
                Src {
                    va: inner.classes[class].host.va(h),
                    from_host: true,
                    fence: None,
                }
            }
        })
    }

    /// Start bringing a host-resident block back to a device slot (H2D queued on the copy
    /// stream; [`Self::reap`] commits it when the fence retires). A gather meanwhile reads the
    /// device slot behind the fence ([`Self::gather_src`]). `Err` = no device slot could be
    /// made: the block stays on the host and gathers read it in place. `Ok(true)` = a copy
    /// was queued.
    pub fn restore_async(&self, id: BlockId) -> Result<bool> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let (class, place, state) = {
            let b = inner
                .block_mut(id)
                .ok_or_else(|| RuntimeError::Rejected("kv store: no such block".into()))?;
            b.tick = tick;
            (b.class, b.place, b.state)
        };
        let Where::Host(h) = place else {
            return Ok(false);
        };
        if state != State::Ready {
            return self.ensure_device_locked(&mut inner, id).map(|_| false);
        }
        let slot = self.device_slot(&mut inner, class)?;
        let bytes = inner.classes[class].spec.block_bytes();
        let va = inner.classes[class].device.va(slot);
        let host = inner.classes[class].host.va(h);
        let fence = match self.ops.copy_htod_async(va, host, bytes) {
            Ok(f) => f,
            Err(e) => {
                inner.classes[class].device.put(slot);
                return Err(e);
            }
        };
        inner.classes[class].device.assign(slot, id);
        let b = inner.block_mut(id).expect("block");
        b.place = Where::Restoring {
            dev: slot,
            host: h,
            fence,
        };
        inner.restoring.push_back(id);
        Ok(true)
    }

    /// Whether any of `ids` still has its restore in flight (landed ones are retired first).
    pub fn restoring_any(&self, ids: &[BlockId]) -> bool {
        let mut inner = self.inner.lock();
        self.reap_restores(&mut inner);
        ids.iter().any(|&id| {
            matches!(
                inner.block(id),
                Some(Block {
                    place: Where::Restoring { .. },
                    ..
                })
            )
        })
    }

    /// Whether any of `ids` is not readable on the device without a wait: host-resident, or
    /// with its restore in flight (landed ones are retired first).
    pub fn off_device_any(&self, ids: &[BlockId]) -> bool {
        let mut inner = self.inner.lock();
        self.reap_restores(&mut inner);
        ids.iter().any(|&id| {
            matches!(
                inner.block(id),
                Some(Block {
                    place: Where::Restoring { .. } | Where::Host(_),
                    ..
                })
            )
        })
    }

    pub fn ensure_device(&self, id: BlockId) -> Result<u64> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let va = self.ensure_device_locked(&mut inner, id)?;
        if let Some(b) = inner.block_mut(id) {
            b.tick = tick;
        }
        Ok(va)
    }

    fn ensure_device_locked(&self, inner: &mut Inner, id: BlockId) -> Result<u64> {
        let (class, place, state) = {
            let b = inner
                .block(id)
                .ok_or_else(|| RuntimeError::Rejected("kv store: no such block".into()))?;
            (b.class, b.place, b.state)
        };
        match place {
            Where::Device(s) | Where::Spilling { dev: s, .. } | Where::Restoring { dev: s, .. } => {
                Ok(inner.classes[class].device.va(s))
            }
            Where::Host(h) => {
                let slot = self.device_slot(inner, class)?;
                let bytes = inner.classes[class].spec.block_bytes();
                let va = inner.classes[class].device.va(slot);
                let host = inner.classes[class].host.va(h);
                if state == State::Ready {
                    if let Err(e) = self.ops.copy_htod(va, host, bytes) {
                        inner.classes[class].device.put(slot);
                        return Err(e);
                    }
                }
                inner.classes[class].device.assign(slot, id);
                inner.classes[class].host.put(h);
                let b = inner.block_mut(id).expect("block");
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
    /// reclaiming the oldest zero-dependent, unheld, settled block of the class. Under a
    /// growth hold (`Inner::grow_hold_until`) the reclaim goes before the growth.
    fn device_slot(&self, inner: &mut Inner, class: usize) -> Result<Slot<Dev>> {
        let ops = &self.ops;
        if let Some(s) = inner.classes[class]
            .device
            .take(0, |_| Err(RuntimeError::Oom("kv store: no growth".into())))?
        {
            return Ok(s);
        }
        let hold = inner
            .grow_hold_until
            .is_some_and(|t| std::time::Instant::now() < t);
        if hold {
            if let Some(s) = self.reclaim_slot(inner, class) {
                return Ok(s);
            }
        }
        let reserved: u64 = inner
            .classes
            .iter()
            .map(|c| c.device.reserved_bytes())
            .sum();
        let mut budget = inner.device_cap.saturating_sub(reserved);
        if budget > 0
            && inner.device_floor > 0
            && inner.stats.device_grows % DEVICE_FLOOR_EVERY == 0
        {
            let chunk = inner.classes[class].device.chunk_bytes();
            if let Some(free) = ops.free_bytes() {
                if free
                    < chunk
                        .saturating_mul(DEVICE_FLOOR_EVERY)
                        .saturating_add(inner.device_floor)
                {
                    budget = 0;
                }
            }
        }
        let mut grow_us = None;
        let taken = inner.classes[class].device.take(budget, |bytes| {
            let t = std::time::Instant::now();
            let r = ops.alloc(bytes);
            grow_us = Some(t.elapsed().as_micros() as u64);
            r.map(|va| Chunk::new(va, bytes))
        });
        if let Some(us) = grow_us {
            inner.stats.device_grows += 1;
            inner.stats.device_grow_us += us;
        }
        if let Some(s) = taken? {
            return Ok(s);
        }
        if let Some(s) = (!hold).then(|| self.reclaim_slot(inner, class)).flatten() {
            return Ok(s);
        }
        inner.stats.refusals += 1;
        Err(RuntimeError::Oom("kv store: class slab full".into()))
    }

    /// The slot of the oldest reclaimable block of `class`, dropped for it; `None` = nothing
    /// to reclaim.
    fn reclaim_slot(&self, inner: &mut Inner, class: usize) -> Option<Slot<Dev>> {
        // Reclaim in release order; stale entries (re-acquired, dropped, wrong tier) are
        // skipped and re-queued where they still matter.
        let mut requeue = Vec::new();
        let mut found = None;
        while let Some(id) = inner.reclaim[class].pop_front() {
            let Some(b) = inner.block(id) else {
                continue;
            };
            if b.dependents > 0 {
                continue;
            }
            if b.holds > 0 || b.state == State::Filling {
                requeue.push(id);
                continue;
            }
            match b.place {
                Where::Device(_) => {
                    found = Some(id);
                    break;
                }
                // In flight to the host: `reap` queues it on the host list when it lands.
                Where::Spilling { .. } | Where::Restoring { .. } => requeue.push(id),
                Where::Host(_) => {}
            }
        }
        for id in requeue.into_iter().rev() {
            inner.reclaim[class].push_front(id);
        }
        let id = found?;
        // `drop_block` returns the victim's slot to its chunk; the take may prefer another
        // hole of the class (the fullest chunk's).
        self.drop_block(inner, id);
        inner.stats.reclaims += 1;
        let taken = inner.classes[class]
            .device
            .take(0, |_| Err(RuntimeError::Oom("kv store: no growth".into())))
            .ok()
            .flatten();
        Some(taken.expect("the reclaimed slot is free"))
    }

    /// Forget a block entirely (its slots go back to their slabs); every copy of its id is
    /// stale from here on.
    fn drop_block(&self, inner: &mut Inner, id: BlockId) {
        let e = &mut inner.blocks[id.index()];
        if e.gen != id.gen {
            return;
        }
        let Some(b) = e.block.take() else {
            return;
        };
        inner.index.remove(&(b.class, b.key));
        // An exhausted generation retires the entry: a wrap would make old ids valid again.
        if let Some(gen) = e.gen.checked_add(1) {
            e.gen = gen;
            inner.free_entries.push(id.index);
        }
        if let Some(h) = inner.refilling.remove(&id) {
            inner.classes[b.class].host.put(h);
        }
        let bytes = inner.classes[b.class].spec.block_bytes();
        let c = &mut inner.classes[b.class];
        let st = &mut inner.stats.classes[b.class];
        match b.place {
            Where::Device(s) => {
                c.device.put(s);
                st.device_blocks -= 1;
                st.device_bytes -= bytes;
            }
            // Its copy may still read one slot and write the other: neither is reused (the
            // only such drop is teardown or a failed fence).
            Where::Spilling { .. } => {
                st.device_blocks -= 1;
                st.device_bytes -= bytes;
            }
            Where::Restoring { .. } => {
                st.host_blocks -= 1;
                st.host_bytes -= bytes;
            }
            Where::Host(h) => {
                c.host.put(h);
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
            let Some(b) = inner.block(id) else {
                continue;
            };
            if b.dependents == 0 && b.holds == 0 && b.state != State::Filling {
                if !matches!(b.place, Where::Spilling { .. } | Where::Restoring { .. }) {
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
    ///
    /// The lock is taken per block: a boundary's spill submits a copy per block (tens of
    /// microseconds each), and a gather waiting for the lock behind a whole boundary waited
    /// milliseconds.
    pub fn spill(&self, ids: &[BlockId]) -> bool {
        if self.inner.lock().host_cap == 0 {
            return false;
        }
        let mut all = true;
        for &id in ids {
            let mut inner = self.inner.lock();
            let t0 = std::time::Instant::now();
            let Some(b) = inner.block(id) else {
                continue;
            };
            let class = b.class;
            let dev = match b.place {
                Where::Device(dev) => dev,
                Where::Spilling { .. } | Where::Host(_) => continue,
                // Coming back to the device: it stays.
                Where::Restoring { .. } => {
                    all = false;
                    continue;
                }
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
                    let b = inner.block_mut(id).expect("block");
                    b.place = Where::Spilling {
                        dev,
                        host: h,
                        fence,
                    };
                    inner.spilling.push_back((id, bytes));
                    inner.spilling_bytes += bytes;
                    inner.stats.spills += 1;
                }
                Err(_) => {
                    inner.classes[class].host.put(h);
                    all = false;
                }
            }
            inner.stats.spill_us += t0.elapsed().as_micros() as u64;
        }
        all
    }

    fn host_slot(&self, inner: &mut Inner, class: usize) -> Option<Slot<Host>> {
        // Spares are already paid for against the cap: only live chunks count here.
        let reserved: u64 = inner.classes.iter().map(|c| c.host.reserved_bytes()).sum();
        let budget = inner.host_cap.saturating_sub(reserved);
        let Inner {
            classes,
            host_spare,
            ..
        } = inner;
        let spare = &mut host_spare[class];
        // Growth takes a chunk the grower allocated ahead; none ready = no growth now, the
        // slot comes from a reclaim below or the block stays on the device.
        let taken = classes[class].host.take(budget, |_| {
            spare
                .pop()
                .ok_or_else(|| RuntimeError::Oom("kv store: no spare host chunk".into()))
        });
        if let Ok(Some(s)) = taken {
            return Some(s);
        }
        // The oldest zero-dependent host block of this class gives up its slot; stale entries
        // (re-acquired, dropped, back on the device) are skipped.
        let mut requeue = Vec::new();
        let mut victim = None;
        while let Some(id) = inner.host_reclaim[class].pop_front() {
            let Some(b) = inner.block(id) else {
                continue;
            };
            if b.dependents > 0 || !matches!(b.place, Where::Host(_)) {
                continue;
            }
            if b.holds > 0 {
                requeue.push(id);
                continue;
            }
            victim = Some(id);
            break;
        }
        for id in requeue.into_iter().rev() {
            inner.host_reclaim[class].push_front(id);
        }
        self.drop_block(inner, victim?);
        inner.stats.evictions += 1;
        inner.classes[class]
            .host
            .take(0, |_| Err(RuntimeError::Oom("kv store: no growth".into())))
            .ok()
            .flatten()
    }

    /// Retire completed spills in order: the device slot frees, the block is host-resident. A
    /// fence that cannot be queried is treated as retired only for a block that was dropped
    /// meanwhile; otherwise the block stays `Spilling` (both copies kept, never reused).
    /// Completed restores retire the same way: the host slot frees, the block is on the device.
    pub fn reap(&self) {
        let mut inner = self.inner.lock();
        self.reap_restores(&mut inner);
        let mut left = inner.spilling.len();
        while left > 0 {
            left -= 1;
            let Some((id, queued)) = inner.spilling.pop_front() else {
                return;
            };
            let (class, dev, host, fence, holds) = match inner.block(id) {
                Some(Block {
                    class,
                    place: Where::Spilling { dev, host, fence },
                    holds,
                    ..
                }) => (*class, *dev, *host, *fence, *holds),
                // Dropped meanwhile (the id is stale): nothing to reap.
                _ => {
                    inner.spilling_bytes -= queued;
                    continue;
                }
            };
            // A transfer still reads the device copy: keep it until the hold drops. Checked
            // before the fence is queried, which consumes it.
            if holds > 0 {
                inner.spilling.push_back((id, queued));
                continue;
            }
            match self.ops.fence_done(fence) {
                Ok(true) => {}
                // Copies retire in order: the rest are still in flight.
                Ok(false) => {
                    inner.spilling.push_front((id, queued));
                    return;
                }
                // Completion unproven: both slots stay as they are, never reused.
                Err(_) => {
                    inner.spilling_bytes -= queued;
                    inner.stats.fill_failures += 1;
                    continue;
                }
            }
            inner.spilling_bytes -= queued;
            let bytes = inner.classes[class].spec.block_bytes();
            inner.classes[class].device.put(dev);
            let b = inner.block_mut(id).expect("block");
            b.place = Where::Host(host);
            if b.dependents == 0 {
                inner.host_reclaim[class].push_back(id);
            }
            let st = &mut inner.stats.classes[class];
            st.device_blocks -= 1;
            st.device_bytes -= bytes;
            st.host_blocks += 1;
            st.host_bytes += bytes;
        }
    }

    /// Retire landed spills, waiting up to `limit` for one while any is queued: `true` = spilled
    /// bytes retired (their device slots are free). A publish refused for want of a device slot calls
    /// this before asking the pool for another victim: each victim's spill lands
    /// asynchronously, and without the wait the retry loop spilled every device boundary in
    /// turn and then took host-resident ones.
    pub fn reap_wait(&self, limit: std::time::Duration) -> bool {
        let t0 = std::time::Instant::now();
        loop {
            let before = self.inner.lock().spilling_bytes;
            if before == 0 {
                return false;
            }
            self.reap();
            if self.inner.lock().spilling_bytes < before {
                return true;
            }
            if t0.elapsed() >= limit {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }

    fn reap_restores(&self, inner: &mut Inner) {
        let mut left = inner.restoring.len();
        while left > 0 {
            left -= 1;
            let Some(id) = inner.restoring.pop_front() else {
                return;
            };
            let (class, host, fence, holds) = match inner.block(id) {
                Some(Block {
                    class,
                    place: Where::Restoring { host, fence, .. },
                    holds,
                    ..
                }) => (*class, *host, *fence, *holds),
                _ => continue,
            };
            // A gather may still read the host copy in place (it started before the restore)
            // or be about to order its stream on this fence: both slots stay and the fence is
            // not queried (a query consumes it) until the hold drops.
            if holds > 0 {
                inner.restoring.push_back(id);
                continue;
            }
            match self.ops.fence_done(fence) {
                Ok(true) => {}
                Ok(false) => {
                    inner.restoring.push_front(id);
                    return;
                }
                Err(_) => {
                    inner.stats.fill_failures += 1;
                    continue;
                }
            }
            let bytes = inner.classes[class].spec.block_bytes();
            inner.classes[class].host.put(host);
            let b = inner.block_mut(id).expect("block");
            let Where::Restoring { dev, .. } = b.place else {
                unreachable!("matched above")
            };
            b.place = Where::Device(dev);
            if b.dependents == 0 {
                inner.reclaim[class].push_back(id);
            }
            inner.stats.restores += 1;
            let st = &mut inner.stats.classes[class];
            st.host_blocks -= 1;
            st.host_bytes -= bytes;
            st.device_blocks += 1;
            st.device_bytes += bytes;
        }
    }

    /// Give device memory back under pressure: release whole chunks whose remaining blocks are
    /// all reclaimable (zero-dependent, unheld, settled, on the device), the emptiest chunks
    /// first, until `bytes` are released; chunks that reclaims emptied go back too. Returns the
    /// bytes released to the device. The pool evicts a boundary when this returns 0: its blocks
    /// then become reclaimable and the chunks they sit in releasable.
    pub fn shrink(&self, bytes: u64) -> u64 {
        let t0 = std::time::Instant::now();
        // One call releases at most `SHRINK_STEP_BYTES` so the lock is dropped between steps
        // (the pool's pressure loop calls again): a whole floor deficit in one call freed a
        // hundred chunks while the engine thread's gather waited 150 ms for the lock.
        let want = bytes.min(SHRINK_STEP_BYTES);
        let mut frees = Vec::new();
        let mut inner = self.inner.lock();
        let mut released = self.release_reclaimable_chunks(&mut inner, want, &mut frees);
        let plan = if released < want {
            self.compaction_plan(&mut inner)
        } else {
            None
        };
        // Driver work with the lock released: the chunk frees and the compaction drain, which
        // a gather or a publish on the engine thread would otherwise wait behind.
        drop(inner);
        self.free_chunks(&mut frees);
        let copied = match &plan {
            Some(p) if !p.copies.is_empty() => self.ops.copy_dtod_batch(&p.copies),
            _ => Ok(()),
        };
        let mut inner = self.inner.lock();
        if let Some(plan) = plan {
            released += self.compaction_commit(&mut inner, plan, copied, &mut frees);
        }
        if released > 0 {
            inner.grow_hold_until = Some(std::time::Instant::now() + GROW_HOLD);
        }
        let us = t0.elapsed().as_micros() as u64;
        inner.stats.shrink_us += us;
        inner.stats.shrink_max_us = inner.stats.shrink_max_us.max(us);
        inner.stats.shrinks += 1;
        drop(inner);
        self.free_chunks(&mut frees);
        released
    }

    /// Release whole chunks whose remaining blocks are all reclaimable (zero-dependent, unheld,
    /// settled, on the device), the emptiest first, until `bytes`; the freed chunk addresses
    /// go to `frees`. Every live chunk is checked before 0 is returned: the pool evicts a
    /// boundary on 0.
    fn release_reclaimable_chunks(
        &self,
        inner: &mut Inner,
        bytes: u64,
        frees: &mut Vec<Chunk<Dev>>,
    ) -> u64 {
        let mut released = 0u64;
        for ci in 0..inner.classes.len() {
            for chunk in inner.classes[ci].device.release_candidates() {
                if released >= bytes {
                    break;
                }
                let ids = inner.classes[ci].device.occupants(chunk);
                let reclaimable = ids.iter().all(|&id| match inner.block(id) {
                    Some(b) => {
                        b.dependents == 0
                            && b.holds == 0
                            && b.state != State::Filling
                            && matches!(b.place, Where::Device(_))
                    }
                    None => true,
                });
                if !reclaimable {
                    continue;
                }
                for id in ids {
                    self.drop_block(inner, id);
                    inner.stats.evictions += 1;
                }
                released += inner.classes[ci]
                    .device
                    .release_empty_chunks(|c| frees.push(c));
            }
        }
        released
    }

    /// Compaction, planned: the emptiest chunk that can be emptied by dropping its reclaimable
    /// blocks and moving its live ones into holes elsewhere. Holes are made first from the
    /// oldest blocks nothing references (the reclaim list), in other chunks. The moving blocks
    /// take a transfer hold and their destinations are taken now; the copies run with the
    /// lock released and [`Self::compaction_commit`] lands them. A chunk with a held, filling,
    /// failed-but-referenced or in-transfer block, or more live blocks than holes can be made
    /// for, is left alone; the pool then evicts a boundary.
    fn compaction_plan(&self, inner: &mut Inner) -> Option<CompactionPlan> {
        for ci in 0..inner.classes.len() {
            'chunks: for chunk in inner.classes[ci].device.release_candidates() {
                let ids = inner.classes[ci].device.occupants(chunk);
                let mut live = 0u64;
                for &id in &ids {
                    match inner.block(id) {
                        Some(b) if b.holds > 0 || b.state == State::Filling => continue 'chunks,
                        Some(b) if !matches!(b.place, Where::Device(_)) => continue 'chunks,
                        // A refill may write it while the copy reads it.
                        Some(b) if b.state == State::Failed && b.dependents > 0 => continue 'chunks,
                        Some(b) if b.dependents > 0 => live += 1,
                        _ => {}
                    }
                }
                let mut deficit =
                    live.saturating_sub(inner.classes[ci].device.holes_outside(chunk));
                if deficit > 0 {
                    let mut requeue = Vec::new();
                    while deficit > 0 {
                        let Some(id) = inner.reclaim[ci].pop_front() else {
                            break;
                        };
                        let Some(b) = inner.block(id) else {
                            continue;
                        };
                        if b.dependents > 0 {
                            continue;
                        }
                        let Where::Device(slot) = b.place else {
                            requeue.push(id);
                            continue;
                        };
                        if b.holds > 0
                            || b.state == State::Filling
                            || inner.classes[ci].device.chunk_of(slot) == chunk
                        {
                            requeue.push(id);
                            continue;
                        }
                        self.drop_block(inner, id);
                        inner.stats.evictions += 1;
                        deficit -= 1;
                    }
                    for id in requeue.into_iter().rev() {
                        inner.reclaim[ci].push_front(id);
                    }
                    if deficit > 0 {
                        continue;
                    }
                }
                let bytes_per = inner.classes[ci].spec.block_bytes();
                let mut moves: Vec<(BlockId, Slot<Dev>, Slot<Dev>)> = Vec::new();
                let mut copies: Vec<(u64, u64, u64)> = Vec::new();
                for id in ids {
                    let Some(b) = inner.block(id) else {
                        continue;
                    };
                    if b.dependents == 0 {
                        self.drop_block(inner, id);
                        inner.stats.evictions += 1;
                        continue;
                    }
                    let Where::Device(from) = b.place else {
                        continue;
                    };
                    let Some(to) = inner.classes[ci].device.take_outside(chunk) else {
                        break;
                    };
                    let d = &inner.classes[ci].device;
                    copies.push((d.va(to), d.va(from), bytes_per));
                    moves.push((id, from, to));
                    inner.block_mut(id).expect("block").holds += 1;
                }
                return Some(CompactionPlan {
                    class: ci,
                    moves,
                    copies,
                });
            }
        }
        None
    }

    /// Land a plan's moves after their copies drained. A block another transfer held meanwhile
    /// (a gather reads its old slot on the engine stream) stays where it is and its destination
    /// goes back; a failed batch leaves every block as it was. Returns the bytes the emptied
    /// chunks release, their addresses in `frees`.
    fn compaction_commit(
        &self,
        inner: &mut Inner,
        plan: CompactionPlan,
        copied: Result<()>,
        frees: &mut Vec<Chunk<Dev>>,
    ) -> u64 {
        let ci = plan.class;
        let bytes_per = inner.classes[ci].spec.block_bytes();
        let mut moved = 0u64;
        for &(id, from, to) in &plan.moves {
            let lands = copied.is_ok()
                && matches!(
                    inner.block(id),
                    Some(b) if b.holds == 1 && b.place == Where::Device(from)
                );
            if lands {
                inner.classes[ci].device.assign(to, id);
                inner.classes[ci].device.put(from);
                inner.block_mut(id).expect("block").place = Where::Device(to);
                moved += 1;
            } else {
                inner.classes[ci].device.put(to);
            }
            if let Some(b) = inner.block_mut(id) {
                b.holds -= 1;
            }
        }
        if let Err(e) = &copied {
            tracing::warn!(error = %e, "kv store: compaction copy failed");
        }
        inner.stats.compactions += moved;
        inner.stats.compact_bytes += moved * bytes_per;
        let mut released = 0u64;
        for c in inner.classes.iter_mut() {
            released += c.device.release_empty_chunks(|c| frees.push(c));
        }
        released
    }

    /// The device bytes the store may reserve from now on (chunks already reserved stay until
    /// [`Self::shrink`] releases them).
    pub fn set_device_cap(&self, bytes: u64) {
        self.inner.lock().device_cap = bytes;
    }

    /// Host bytes the tier may still reserve: the cap less live chunks and spares.
    fn host_budget(&self, inner: &Inner) -> u64 {
        let reserved: u64 = inner.classes.iter().map(|c| c.host.reserved_bytes()).sum();
        let spare: u64 = inner
            .classes
            .iter()
            .zip(&inner.host_spare)
            .map(|(c, s)| s.len() as u64 * c.host.chunk_bytes())
            .sum();
        inner.host_cap.saturating_sub(reserved + spare)
    }

    /// One step of the host grower thread: allocate a pinned chunk ahead for the class with
    /// the fewest spare bytes under [`HOST_SPARE_BYTES`], outside the lock. `false` = nothing
    /// to do (every class has its spares, or the cap or the host reserve is reached: then the
    /// tier stops growing for good).
    pub fn host_grow_step(&self) -> bool {
        let (class, chunk) = {
            let mut inner = self.inner.lock();
            if inner.host_cap == 0 {
                return false;
            }
            let budget = self.host_budget(&inner);
            let pick = (0..inner.classes.len())
                .map(|c| {
                    let chunk = inner.classes[c].host.chunk_bytes();
                    (inner.host_spare[c].len() as u64 * chunk, c, chunk)
                })
                .filter(|&(bytes, _, chunk)| bytes < HOST_SPARE_BYTES.max(chunk) && chunk <= budget)
                .min();
            let Some((_, class, chunk)) = pick else {
                return false;
            };
            if inner.host_reserve > 0 && inner.stats.host_grows % HOST_FLOOR_EVERY == 0 {
                if let Some((avail, _)) = crate::memory::vmm::host_mem_available() {
                    if avail
                        < chunk
                            .saturating_mul(HOST_FLOOR_EVERY)
                            .saturating_add(inner.host_reserve)
                    {
                        let reserved = inner.host_cap - budget;
                        tracing::warn!(
                            available_mib = avail >> 20,
                            reserve_mib = inner.host_reserve >> 20,
                            reserved_mib = reserved >> 20,
                            "kv store: host tier frozen at the host reserve"
                        );
                        inner.host_cap = reserved;
                        return false;
                    }
                }
            }
            // Counted before the allocation so a failed one is not retried every step.
            inner.stats.host_grows += 1;
            (class, chunk)
        };
        let t = std::time::Instant::now();
        let ptr = self.ops.host_alloc(chunk);
        let us = t.elapsed().as_micros() as u64;
        let mut inner = self.inner.lock();
        inner.stats.host_grow_us += us;
        inner.stats.host_grow_max_us = inner.stats.host_grow_max_us.max(us);
        match ptr {
            Ok(va) => {
                inner.host_spare[class].push(Chunk::new(va, chunk));
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "kv store: host tier allocation failed; tier frozen");
                inner.host_cap = inner.host_cap - self.host_budget(&inner);
                false
            }
        }
    }

    /// The host tier's cap now (0 = none; frozen caps are lower than the one armed).
    pub fn host_cap(&self) -> u64 {
        self.inner.lock().host_cap
    }

    /// Bytes of D2H copies in flight: device memory the pool may count as coming free.
    /// Device bytes coming free: spills in flight and released chunks the worker has not freed.
    pub fn pending_spill_bytes(&self) -> u64 {
        self.inner.lock().spilling_bytes
            + self
                .pending_free_bytes
                .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Device bytes the store leaves free (see `Inner::device_floor`).
    pub fn set_device_floor(&self, bytes: u64) {
        self.inner.lock().device_floor = bytes;
    }

    /// Host RAM the tier must leave available (see `Inner::host_reserve`).
    pub fn set_host_reserve(&self, bytes: u64) {
        self.inner.lock().host_reserve = bytes;
    }

    pub fn stats(&self) -> StoreStats {
        let inner = self.inner.lock();
        let mut s = inner.stats.clone();
        s.device_reserved_bytes = inner
            .classes
            .iter()
            .map(|c| c.device.reserved_bytes())
            .sum();
        s
    }

    pub fn device_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        inner.classes.iter().map(|c| c.device.used_bytes()).sum()
    }

    /// Device bytes the store's slabs occupy, used or not: what it took from free memory.
    pub fn device_reserved_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        inner
            .classes
            .iter()
            .map(|c| c.device.reserved_bytes())
            .sum()
    }

    /// Device bytes acquires can take without a boundary being evicted: free slots of the
    /// reserved chunks, blocks on the reclaim lists (an upper bound: stale entries count) and
    /// the growth room under the cap and the floor.
    pub fn device_headroom_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        let mut bytes = 0u64;
        for (c, class) in inner.classes.iter().enumerate() {
            let slot = class.spec.block_bytes();
            bytes += (class.device.free_slots + inner.reclaim[c].len() as u64) * slot;
        }
        let reserved: u64 = inner
            .classes
            .iter()
            .map(|c| c.device.reserved_bytes())
            .sum();
        // The growth hold is not subtracted: the spill-ahead thread would otherwise drain the
        // device tier for ten seconds after every shrink (2x the spills, 10x the restores).
        let growth = inner.device_cap.saturating_sub(reserved);
        let chunk = inner.classes.first().map_or(0, |c| c.device.chunk_bytes());
        let room = match self.ops.free_bytes() {
            Some(free) => growth.min(
                free.saturating_sub(chunk.saturating_mul(DEVICE_FLOOR_EVERY) + inner.device_floor),
            ),
            None => growth,
        };
        bytes + room
    }
}

impl Drop for KvStore {
    fn drop(&mut self) {
        let KvStore {
            ops, inner, frees, ..
        } = self;
        let ops: &dyn StoreOps = &**ops;
        // A spill or restore queued after the engine's last sync (the spill-ahead and restore
        // threads run until the pool drops) may still be copying: free nothing under it.
        if let Err(e) = ops.quiesce() {
            tracing::warn!(error = %e, "kv store: copies unproven at drop, chunks leaked");
            return;
        }
        for chunk in frees.get_mut().drain(..) {
            chunk.free(ops);
        }
        let inner = inner.get_mut();
        let quarantined = inner.quarantined > 0;
        if quarantined {
            tracing::warn!(
                slots = inner.quarantined,
                "kv store: a fill copy is unproven, device chunks leaked"
            );
        }
        for c in &mut inner.classes {
            for chunk in c.device.chunks.drain(..).flatten() {
                if !quarantined {
                    chunk.free(ops);
                }
            }
            for chunk in c.host.chunks.drain(..).flatten() {
                chunk.free(ops);
            }
        }
        for chunk in inner.host_spare.drain(..).flatten() {
            chunk.free(ops);
        }
    }
}

/// Chained 64-bit hashes of `tokens` per `rows` rows: entry `i` covers `tokens[..(i+1)*rows]`,
/// hashed with `H` from `seed`. The key's `chain` and `check` use two unrelated hash
/// functions (`FxHasher` and SipHash), not two seeds of one.
pub fn chain_hashes<H: std::hash::Hasher + Default>(
    tokens: &[u32],
    rows: u32,
    seed: u64,
) -> Vec<u64> {
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
        freed: AtomicU64,
        host_alloc: AtomicU64,
        htod: AtomicU64,
        htod_async: AtomicU64,
        dtod: AtomicU64,
        /// Every fence reads as unretired while set.
        pending: std::sync::atomic::AtomicBool,
        next_fence: AtomicU64,
        consumed: parking_lot::Mutex<std::collections::HashSet<u64>>,
        /// Runs inside `copy_dtod_batch` (the store lock must be free there).
        on_dtod: parking_lot::Mutex<Option<Box<dyn Fn() + Send>>>,
        /// Block contents per address, moved by the copies (the model test).
        mem: parking_lot::Mutex<std::collections::HashMap<u64, u64>>,
    }

    impl MockOps {
        fn new() -> Arc<Self> {
            Arc::new(MockOps {
                next: AtomicU64::new(1 << 20),
                dev_alloc: AtomicU64::new(0),
                freed: AtomicU64::new(0),
                host_alloc: AtomicU64::new(0),
                htod: AtomicU64::new(0),
                htod_async: AtomicU64::new(0),
                dtod: AtomicU64::new(0),
                pending: std::sync::atomic::AtomicBool::new(false),
                next_fence: AtomicU64::new(100),
                consumed: parking_lot::Mutex::new(Default::default()),
                on_dtod: parking_lot::Mutex::new(None),
                mem: parking_lot::Mutex::new(Default::default()),
            })
        }

        fn mv(&self, dst: u64, src: u64) {
            let mut m = self.mem.lock();
            if let Some(v) = m.get(&src).copied() {
                m.insert(dst, v);
            }
        }
    }

    impl StoreOps for MockOps {
        fn alloc(&self, bytes: u64) -> Result<u64> {
            self.dev_alloc.fetch_add(bytes, Ordering::SeqCst);
            Ok(self.next.fetch_add(bytes + 4096, Ordering::SeqCst))
        }
        fn free(&self, _va: u64) {
            self.freed.fetch_add(1, Ordering::SeqCst);
        }
        fn host_alloc(&self, bytes: u64) -> Result<u64> {
            self.host_alloc.fetch_add(bytes, Ordering::SeqCst);
            Ok(self.next.fetch_add(bytes + 4096, Ordering::SeqCst))
        }
        fn host_free(&self, _ptr: u64, _bytes: u64) {}
        fn copy_dtoh_async(&self, host: u64, va: u64, _bytes: u64) -> Result<u64> {
            self.mv(host, va);
            Ok(self.next_fence.fetch_add(1, Ordering::SeqCst))
        }
        /// Like the CUDA tier: a retired fence is consumed, a second query of it is an error.
        fn fence_done(&self, fence: u64) -> Result<bool> {
            if self.pending.load(Ordering::SeqCst) {
                return Ok(false);
            }
            if self.consumed.lock().insert(fence) {
                Ok(true)
            } else {
                Err(RuntimeError::Device("mock: unknown fence".into()))
            }
        }
        fn copy_htod(&self, va: u64, host: u64, _bytes: u64) -> Result<()> {
            self.mv(va, host);
            self.htod.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn copy_htod_async(&self, va: u64, host: u64, _bytes: u64) -> Result<u64> {
            self.mv(va, host);
            self.htod_async.fetch_add(1, Ordering::SeqCst);
            Ok(self.next_fence.fetch_add(1, Ordering::SeqCst))
        }
        fn copy_dtod_batch(&self, copies: &[(u64, u64, u64)]) -> Result<()> {
            for &(dst, src, _) in copies {
                self.mv(dst, src);
            }
            self.dtod.fetch_add(copies.len() as u64, Ordering::SeqCst);
            if let Some(f) = self.on_dtod.lock().as_ref() {
                f();
            }
            Ok(())
        }
        fn free_bytes(&self) -> Option<u64> {
            None
        }
    }

    fn store(device_cap: u64, host_cap: u64) -> (Arc<MockOps>, Arc<KvStore>) {
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
        while s.host_grow_step() {}
        (ops, Arc::new(s))
    }

    fn seal(a: &mut Acquired<'_>) {
        a.fill.take().expect("a fill").seal();
    }

    fn acquire_sealed(s: &KvStore, class: usize, key: BlockKey) -> Acquired<'_> {
        let mut a = s.acquire(class, key).unwrap();
        seal(&mut a);
        a
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
        let mut a = s.acquire(0, key(1)).unwrap();
        assert!(a.fill.is_some());
        seal(&mut a);
        let b = s.acquire(0, key(1)).unwrap();
        assert!(b.fill.is_none());
        assert_eq!(a.id, b.id);
        assert_eq!(a.va, b.va);
        assert_eq!(s.stats().dedup_hits, 1);
        assert_eq!(s.stats().classes[0].device_blocks, 1);
        // A dedup hit neither fails nor re-fills the block.
        drop(b);
        assert_eq!(s.state(a.id), Some(State::Ready));
        assert_eq!(s.stats().fill_failures, 0);
        let c = s.acquire(0, key(2)).unwrap();
        let d = s.acquire(0, key(2)).unwrap();
        assert!(c.fill.is_some() && d.fill.is_none());
        drop(d);
        assert_eq!(
            s.state(c.id),
            Some(State::Filling),
            "still the first's fill"
        );
        assert_eq!(s.stats().fill_failures, 0);
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
        assert!(b.fill.is_some() && a.id != b.id);
    }

    #[test]
    fn reclaim_reuses_released_blocks_oldest_first_when_full() {
        let (_, s) = store(2 * 32, 0); // two 32-byte blocks
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut b = s.acquire(0, key(2)).unwrap();
        seal(&mut a);
        seal(&mut b);
        assert!(s.acquire(0, key(3)).is_err(), "full, nothing released");
        assert_eq!(s.stats().refusals, 1);
        s.release(&[b.id]);
        s.release(&[a.id]);
        // Both released: b first, so b's slot goes first; a is still present (a demoted
        // boundary can still hit).
        let c = s.acquire(0, key(3)).unwrap();
        assert!(c.fill.is_some());
        assert_eq!(c.va, b.va);
        let again = s.acquire(0, key(1)).unwrap();
        assert!(again.fill.is_none(), "a survived on the reclaim list");
        assert_eq!(s.stats().reclaims, 1);
    }

    #[test]
    fn reclaim_skips_an_id_another_class_reused() {
        let spec = |name| ClassSpec {
            name,
            row_bytes: 8,
            rows_per_block: 4,
        };
        let s = KvStore::new(
            MockOps::new(),
            vec![spec("window"), spec("state")],
            32,
            3 * 32,
            0,
        );
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut c = s.acquire(0, key(2)).unwrap();
        seal(&mut a);
        seal(&mut c);
        s.release(&[a.id]);
        // a is dropped; its id is free and its queue entry in class 0 stale.
        assert_eq!(s.evict(&[a.id]), 1);
        let mut b = s.acquire(1, key(3)).unwrap();
        assert_eq!(
            b.id.index, a.id.index,
            "the freed entry is reused by the other class"
        );
        assert_ne!(b.id, a.id, "under a new generation");
        seal(&mut b);
        s.release(&[b.id]);
        let mut d = s.acquire(0, key(4)).unwrap();
        assert_eq!(d.va, a.va);
        seal(&mut d);
        // Class 0 is full and its queue holds only the stale id: a refusal, never b's slot.
        assert!(s.acquire(0, key(5)).is_err());
        assert_eq!(s.stats().refusals, 1);
        assert!(s.acquire(1, key(3)).unwrap().fill.is_none(), "b survived");
        s.release(&[c.id]);
        let e = s.acquire(0, key(5)).unwrap();
        assert_eq!(e.va, c.va);
        assert_eq!(s.stats().reclaims, 1);
    }

    #[test]
    fn host_reclaim_skips_an_id_another_class_reused() {
        let spec = |name| ClassSpec {
            name,
            row_bytes: 8,
            rows_per_block: 4,
        };
        let s = KvStore::new(
            MockOps::new(),
            vec![spec("window"), spec("state")],
            32,
            3 * 32,
            2 * 32,
        );
        while s.host_grow_step() {}
        let park = |a: &mut Acquired<'_>| {
            let id = a.id;
            seal(a);
            s.release(&[id]);
            assert!(s.spill(&[id]));
            s.reap();
            assert_eq!(s.location(id), Some(Location::Host));
        };
        let mut a = s.acquire(0, key(1)).unwrap();
        park(&mut a);
        // a is dropped off the host: its id is free, its host-queue entry in class 0 stale.
        assert_eq!(s.evict(&[a.id]), 1);
        let mut b = s.acquire(1, key(2)).unwrap();
        assert_eq!(
            b.id.index, a.id.index,
            "the freed entry is reused by the other class"
        );
        assert_ne!(b.id, a.id, "under a new generation");
        park(&mut b);
        let mut c = s.acquire(0, key(3)).unwrap();
        park(&mut c);
        // Class 0's host slab is full and both host chunks are taken: the stale entry must not
        // give up b's slot; c, the class's own oldest, does.
        let mut d = s.acquire(0, key(4)).unwrap();
        seal(&mut d);
        s.release(&[d.id]);
        assert!(s.spill(&[d.id]));
        s.reap();
        assert_eq!(s.location(d.id), Some(Location::Host));
        assert_eq!(s.location(c.id), None);
        assert_eq!(s.location(b.id), Some(Location::Host), "b survived");
        assert_eq!(s.stats().classes[1].host_blocks, 1);
    }

    /// Release A then B; A is evicted and its table entry reused by C in the same class; C is
    /// released. A's stale queue entry must not reclaim C ahead of B (release order).
    #[test]
    fn a_stale_reclaim_entry_does_not_take_its_entrys_new_block_before_an_older_release() {
        let (_, s) = store(2 * 32, 0);
        let a = acquire_sealed(&s, 0, key(1));
        let b = acquire_sealed(&s, 0, key(2));
        s.release(&[a.id]);
        s.release(&[b.id]);
        assert_eq!(s.evict(&[a.id]), 1);
        let c = acquire_sealed(&s, 0, key(3));
        assert_eq!(
            (c.id.index, c.va),
            (a.id.index, a.va),
            "a's entry and slot, reused in the same class"
        );
        assert_ne!(c.id, a.id);
        s.release(&[c.id]);
        // The queue is [a (stale), b, c]: b, the oldest release, gives up its slot.
        let d = acquire_sealed(&s, 0, key(4));
        assert_eq!(d.va, b.va, "release order: b before c");
        assert!(
            s.acquire(0, key(3)).unwrap().fill.is_none(),
            "c survived its predecessor's queue entry"
        );
        assert_eq!(s.stats().reclaims, 1);
        assert_eq!(s.state(a.id), None, "the stale id names nothing");
    }

    /// An entry whose generation is exhausted is never reused: an id from before the wrap
    /// cannot name a later block.
    #[test]
    fn an_exhausted_generation_retires_its_entry() {
        let (_, s) = store(2 * 32, 0);
        let a = acquire_sealed(&s, 0, key(1));
        s.inner.lock().blocks[a.id.index()].gen = u32::MAX;
        let a = BlockId {
            index: a.id.index,
            gen: u32::MAX,
        };
        s.release(&[a]);
        assert_eq!(s.evict(&[a]), 1);
        let b = acquire_sealed(&s, 0, key(2));
        assert_ne!(b.id.index, a.index, "the retired entry is not reused");
        assert_eq!(s.state(a), None);
    }

    #[test]
    fn held_or_filling_blocks_are_not_reclaimed() {
        let (_, s) = store(32, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        s.release(&[a.id]);
        let hold = s.hold(vec![a.id]);
        assert!(s.acquire(0, key(2)).is_err(), "held");
        drop(hold);
        assert!(s.acquire(0, key(2)).is_err(), "still filling");
        seal(&mut a);
        assert!(s.acquire(0, key(2)).is_ok());
    }

    #[test]
    fn failed_block_is_refilled_not_served() {
        let (_, s) = store(1024, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        a.fill.take().expect("a fill").fail();
        assert_eq!(s.state(a.id), Some(State::Failed));
        let b = s.acquire(0, key(1)).unwrap();
        assert!(b.fill.is_some() && b.id == a.id);
        assert_eq!(s.state(a.id), Some(State::Filling));
    }

    /// A seal event whose completion could not be proven: the copy may still write the slot,
    /// so the block is never served or refilled in place and the slot is never reused or freed.
    #[test]
    fn an_unproven_fill_leaks_its_slot() {
        let (ops, s) = store(2 * 32, 0); // chunk = 32 B = one block
        let mut a = s.acquire(0, key(1)).unwrap();
        let va = a.va;
        let id = a.fill.take().expect("a fill").defer();
        drop(a);
        s.quarantine_ids(&[id]);
        assert_eq!(s.state(id), None, "never served");
        s.release(&[id]);
        let mut b = s.acquire(0, key(1)).unwrap();
        assert!(b.id != id && b.va != va, "refilled in a fresh slot");
        seal(&mut b);
        s.release(&[b.id]);
        assert_eq!(
            s.shrink(u64::MAX),
            32,
            "only the fresh slot's chunk goes back"
        );
        let c = acquire_sealed(&s, 0, key(2));
        assert!(
            s.acquire(0, key(3)).is_err(),
            "the leaked slot is never handed out"
        );
        assert_eq!(s.stats().fill_failures, 1);
        drop((b, c));
        let freed = ops.freed.load(Ordering::SeqCst);
        drop(s);
        assert_eq!(
            ops.freed.load(Ordering::SeqCst),
            freed,
            "device chunks leaked at drop"
        );
    }

    #[test]
    fn an_unproven_refill_goes_back_to_its_host_bytes() {
        let (_ops, s) = store(2 * 32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        assert!(s.spill(&[a.id]));
        s.reap();
        let host_va = s.gather_src(a.id).unwrap().va;
        let again = s.acquire(0, key(1)).unwrap();
        let dev_va = again.va;
        again.fill.expect("refill").quarantine();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert_eq!(s.state(a.id), Some(State::Ready));
        assert_eq!(s.gather_src(a.id).unwrap().va, host_va);
        assert_eq!(s.stats().classes[0].device_blocks, 0);
        let mut b = s.acquire(0, key(2)).unwrap();
        assert_ne!(b.va, dev_va, "the refill's slot is never reused");
        seal(&mut b);
    }

    /// A boundary still depends on a host block a publish refills: the refill's failure must
    /// leave those bytes servable (they were read as garbage from the fresh device slot).
    #[test]
    fn a_failed_refill_goes_back_to_its_host_bytes() {
        let (_ops, s) = store(32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        let host_va = s.gather_src(a.id).unwrap().va;
        let again = s.acquire(0, key(1)).unwrap();
        assert!(again.fill.is_some(), "refilled on the device");
        drop(again.fill);
        s.release(&[a.id]);
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert_eq!(s.state(a.id), Some(State::Ready));
        let src = s.gather_src(a.id).unwrap();
        assert!(src.from_host);
        assert_eq!(src.va, host_va);
        assert_eq!(
            (
                s.stats().classes[0].host_blocks,
                s.stats().classes[0].device_blocks
            ),
            (1, 0)
        );
        // A sealed refill lets the host slot go.
        let mut third = s.acquire(0, key(1)).unwrap();
        seal(&mut third);
        assert!(matches!(s.location(a.id), Some(Location::Device(_))));
        assert_eq!(s.stats().classes[0].host_blocks, 0);
        assert!(s.inner.lock().refilling.is_empty());
    }

    #[test]
    fn spill_then_restore_round_trips_through_the_host_tier() {
        let (ops, s) = store(32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert_eq!(s.stats().classes[0].host_blocks, 1);
        // The device slot is free again for another block.
        let mut b = s.acquire(0, key(2)).unwrap();
        assert!(b.fill.is_some());
        seal(&mut b);
        s.release(&[b.id]);
        // A publish that meets the host block refills it from its own rows (b's slot is
        // reclaimed); nothing is copied back and the host slot goes.
        let mut again = s.acquire(0, key(1)).unwrap();
        assert!(again.fill.is_some(), "refilled on the device, not restored");
        assert_eq!(ops.htod.load(Ordering::SeqCst), 0);
        assert_eq!(
            (s.stats().refills, s.stats().classes[0].host_blocks),
            (1, 0)
        );
        assert_eq!(s.location(a.id), Some(Location::Device(again.va)));
        // A gather (`ensure_device`) is the restore.
        seal(&mut again);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        let va = s.ensure_device(a.id).unwrap();
        assert_eq!(ops.htod.load(Ordering::SeqCst), 1);
        assert_eq!(s.stats().restores, 1);
        assert_eq!(s.location(a.id), Some(Location::Device(va)));
    }

    #[test]
    fn a_live_block_spills_and_restores_on_the_next_use() {
        let (ops, s) = store(32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        assert!(
            s.spill(&[a.id]),
            "dependents do not pin a block to the device"
        );
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        let va = s.ensure_device(a.id).unwrap();
        assert_eq!(s.location(a.id), Some(Location::Device(va)));
        assert_eq!(ops.htod.load(Ordering::SeqCst), 1);
        let _hold = s.hold(vec![a.id]);
        assert!(!s.spill(&[a.id]), "a held block stays");
    }

    #[test]
    fn reap_wait_frees_the_slot_of_a_landed_spill_and_gives_up_on_a_stuck_one() {
        let (ops, s) = store(32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        assert!(!s.reap_wait(std::time::Duration::ZERO), "nothing queued");
        assert!(s.spill(&[a.id]));
        ops.pending.store(true, Ordering::SeqCst);
        assert!(
            s.acquire(0, key(2)).is_err(),
            "the spilling block keeps its slot"
        );
        assert!(
            !s.reap_wait(std::time::Duration::from_millis(2)),
            "still in flight"
        );
        ops.pending.store(false, Ordering::SeqCst);
        assert!(s.reap_wait(std::time::Duration::from_millis(2)));
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.acquire(0, key(2)).is_ok());
    }

    #[test]
    fn host_tier_off_refuses_every_spill() {
        let (_, s) = store(32, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        assert!(!s.spill(&[a.id]));
        assert_eq!(s.location(a.id), Some(Location::Device(a.va)));
    }

    #[test]
    fn no_spare_host_chunk_reclaims_a_host_block_under_the_cap() {
        // Cap for two chunks, one grown ahead: the second spill finds headroom but no spare.
        let ops = MockOps::new();
        let s = KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            32,
            2 * 32,
            2 * 32,
        );
        assert!(s.host_grow_step());
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut b = s.acquire(0, key(2)).unwrap();
        seal(&mut a);
        seal(&mut b);
        s.release(&[a.id, b.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.spill(&[b.id]), "a's host slot is reclaimed, not refused");
        s.reap();
        assert_eq!(
            s.location(a.id),
            None,
            "the oldest zero-dependent host block gave way"
        );
        assert_eq!(s.location(b.id), Some(Location::Host));
        assert_eq!(s.stats().classes[0].host_blocks, 1);
        assert_eq!(ops.host_alloc.load(Ordering::SeqCst), 32, "no second chunk");
    }

    #[test]
    fn device_headroom_counts_growth_free_slots_and_reclaimable_blocks() {
        let (_, s) = store(4 * 32, 0); // chunk = 32 B = one block
        assert_eq!(s.device_headroom_bytes(), 4 * 32, "all growth");
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        assert_eq!(s.device_headroom_bytes(), 3 * 32, "one block live");
        s.release(&[a.id]);
        assert_eq!(
            s.device_headroom_bytes(),
            4 * 32,
            "released: reclaimable again"
        );
        s.set_device_cap(32);
        assert_eq!(
            s.device_headroom_bytes(),
            32,
            "no growth under the cap, one reclaimable"
        );
    }

    #[test]
    fn a_reclaimed_slot_is_handed_out_once() {
        let (_, s) = store(2 * 32, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut b = s.acquire(0, key(2)).unwrap();
        seal(&mut a);
        seal(&mut b);
        s.release(&[a.id, b.id]);
        let c = s.acquire(0, key(3)).unwrap();
        let d = s.acquire(0, key(4)).unwrap();
        assert_ne!(c.va, d.va, "two live blocks never alias");
        assert!(s.acquire(0, key(5)).is_err());
    }

    #[test]
    fn a_held_spilling_block_keeps_its_device_copy_until_unheld() {
        let (_, s) = store(32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        let hold = s.hold(vec![a.id]);
        s.reap();
        assert_eq!(
            s.location(a.id),
            Some(Location::Device(a.va)),
            "held: not retired"
        );
        assert!(s.acquire(0, key(2)).is_err(), "its device slot is not free");
        drop(hold);
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.acquire(0, key(2)).is_ok());
    }

    #[test]
    fn shrink_drops_reclaimable_blocks_and_releases_empty_chunks() {
        let (_, s) = store(4 * 32, 0); // chunk = 32 B = one block
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        s.release(&ids[..2]);
        assert_eq!(s.device_bytes(), 4 * 32);
        let released = s.shrink(64);
        assert_eq!(released, 64, "two empty chunks went back");
        assert_eq!(s.device_bytes(), 2 * 32);
        assert!(
            s.acquire(0, key(1)).unwrap().fill.is_some(),
            "dropped, refilled"
        );
    }

    #[test]
    fn a_pressure_shrink_holds_growth_behind_the_reclaim_list() {
        let (ops, s) = store(8 * 32, 0); // chunk = 32 B = one block, cap eight
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        s.release(&ids[..2]);
        assert_eq!(s.shrink(32), 32, "one empty chunk went back");
        let allocs = ops.dev_alloc.load(Ordering::SeqCst);
        // Under the hold the other released block's slot goes first: no growth.
        let e = acquire_sealed(&s, 0, key(5));
        assert_eq!(ops.dev_alloc.load(Ordering::SeqCst), allocs, "reclaimed");
        assert_eq!(s.stats().reclaims, 1);
        // Nothing left to reclaim: growth is the last resort even under the hold.
        acquire_sealed(&s, 0, key(6));
        assert_eq!(ops.dev_alloc.load(Ordering::SeqCst), allocs + 32, "grown");
        // The headroom report keeps the growth room through the hold (spill-ahead stays calm).
        s.release(&[e.id]);
        assert_eq!(s.device_headroom_bytes(), 32 + 8 * 32 - 4 * 32);
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        s.inner.lock().grow_hold_until = Some(past);
        // Past the deadline a reclaimable block stays and the slab grows under the cap.
        acquire_sealed(&s, 0, key(7));
        assert_eq!(ops.dev_alloc.load(Ordering::SeqCst), allocs + 64, "grown");
        assert!(
            s.acquire(0, key(5)).unwrap().fill.is_none(),
            "block 5 survived"
        );
    }

    #[test]
    fn a_shrink_that_released_nothing_starts_no_hold() {
        let (ops, s) = store(8 * 32, 0);
        let a = acquire_sealed(&s, 0, key(1));
        acquire_sealed(&s, 0, key(2));
        assert_eq!(s.shrink(32), 0, "every block is referenced");
        assert!(s.inner.lock().grow_hold_until.is_none());
        s.release(&[a.id]);
        let allocs = ops.dev_alloc.load(Ordering::SeqCst);
        acquire_sealed(&s, 0, key(3));
        assert_eq!(ops.dev_alloc.load(Ordering::SeqCst), allocs + 32, "grown");
        assert!(
            s.acquire(0, key(1)).unwrap().fill.is_none(),
            "block 1 survived"
        );
    }

    #[test]
    fn a_releasing_shrink_refreshes_the_hold_and_an_empty_one_keeps_it() {
        let (_, s) = store(8 * 32, 0);
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        s.release(&ids[..2]);
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        assert_eq!(s.shrink(32), 32);
        assert!(s.inner.lock().grow_hold_until.is_some_and(|t| t > past));
        s.inner.lock().grow_hold_until = Some(past);
        assert_eq!(s.shrink(32), 32, "the other released block's chunk");
        assert!(
            s.inner.lock().grow_hold_until.is_some_and(|t| t > past),
            "refreshed"
        );
        s.inner.lock().grow_hold_until = Some(past);
        assert_eq!(s.shrink(32), 0, "nothing left to release");
        assert_eq!(s.inner.lock().grow_hold_until, Some(past), "kept");
    }

    #[test]
    fn a_free_slot_goes_before_the_reclaim_list_under_the_hold() {
        let ops = MockOps::new();
        let s = KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            64, // two blocks per chunk
            4 * 64,
            0,
        );
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        s.release(&ids[..2]);
        assert_eq!(s.shrink(64), 64, "chunk 0 went back: the hold is on");
        // Nothing reclaimable (1 and 2 are gone): the slab grows, one slot of it free.
        let allocs = ops.dev_alloc.load(Ordering::SeqCst);
        let e = acquire_sealed(&s, 0, key(5));
        assert_eq!(ops.dev_alloc.load(Ordering::SeqCst), allocs + 64);
        s.release(&ids[2..3]);
        // The free slot wins over the reclaimable block 3.
        acquire_sealed(&s, 0, key(6));
        assert_eq!(
            ops.dev_alloc.load(Ordering::SeqCst),
            allocs + 64,
            "no growth"
        );
        assert_eq!(s.stats().reclaims, 0);
        assert!(
            s.acquire(0, key(3)).unwrap().fill.is_none(),
            "block 3 survived"
        );
    }

    #[test]
    fn allocation_fills_the_fullest_chunk_and_shrink_releases_whole_chunks() {
        let ops = MockOps::new();
        let s = KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            64, // two blocks per chunk
            2 * 64,
            0,
        );
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        assert_eq!(
            s.device_bytes(),
            2 * 64,
            "chunk 0 = blocks 1,2; chunk 1 = blocks 3,4"
        );
        // A dead block in each chunk: compaction packs the two live ones into one chunk.
        s.release(&ids[1..3]);
        assert_eq!(s.shrink(64), 64);
        assert_eq!(s.device_bytes(), 64, "blocks 1 and 4 remain");
        assert_eq!(ops.dtod.load(Ordering::SeqCst), 1);
        // Under the cap a new block grows a chunk again (one slot of it free).
        let e = s.acquire(0, key(5)).unwrap();
        assert!(e.fill.is_some());
        assert_eq!(s.device_bytes(), 64 + 32);
        assert_eq!(
            s.shrink(64),
            0,
            "every block is referenced: nothing to release"
        );
        // Block 4 dies: its slot takes block 5, the sparse chunk goes back.
        s.release(&ids[3..]);
        assert_eq!(s.shrink(64), 64);
        assert_eq!(s.device_bytes(), 64, "blocks 1 and 5, packed");
        assert!(
            s.acquire(0, key(4)).unwrap().fill.is_some(),
            "block 4 was dropped"
        );
    }

    #[test]
    fn shrink_checks_every_chunk_and_growth_reuses_released_indices() {
        let ops = MockOps::new();
        let s = KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            64, // two blocks per chunk
            40 * 64,
            0,
        );
        // 40 chunks, each holding one referenced block and one reclaimable one, except the
        // last, whose two blocks are both reclaimable.
        let ids: Vec<_> = (1..=80).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        for c in 0..40 {
            s.release(&[ids[2 * c + 1]]);
        }
        s.release(&[ids[78]]);
        assert_eq!(
            s.shrink(64),
            64,
            "the last chunk is the only releasable one"
        );
        assert_eq!(s.device_bytes(), 39 * 64);
        // Growth reuses the released chunk index: no new metadata.
        s.inner.lock().grow_hold_until = None;
        let before = s.inner.lock().classes[0].device.chunks.len();
        let n = s.acquire(0, key(100)).unwrap();
        assert!(n.fill.is_some());
        assert_eq!(s.inner.lock().classes[0].device.chunks.len(), before);
        assert_eq!(
            s.device_bytes(),
            40 * 64 - 32,
            "one slot of the regrown chunk is free"
        );
    }

    #[test]
    fn host_victims_are_per_class() {
        let ops = MockOps::new();
        let classes = ["window", "tail"]
            .iter()
            .map(|name| ClassSpec {
                name,
                row_bytes: 8,
                rows_per_block: 4,
            })
            .collect();
        // Device: one block per class; host: one slot per class (chunk = 32 B = one block).
        let s = KvStore::new(ops.clone(), classes, 32, 2 * 32, 2 * 32);
        while s.host_grow_step() {}
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut t = s.acquire(1, key(2)).unwrap();
        seal(&mut a);
        seal(&mut t);
        s.release(&[a.id, t.id]);
        assert!(s.spill(&[a.id, t.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        // A second window block spills: the host is full for its class, the window block on
        // the host is the victim, the tail block stays.
        let mut a2 = s.acquire(0, key(3)).unwrap();
        seal(&mut a2);
        s.release(&[a2.id]);
        assert!(s.spill(&[a2.id]));
        s.reap();
        assert_eq!(s.location(a2.id), Some(Location::Host));
        assert_eq!(s.location(a.id), None, "window victim dropped");
        assert_eq!(s.location(t.id), Some(Location::Host), "tail untouched");
    }

    #[test]
    fn a_gather_reads_a_host_block_in_place() {
        let (ops, s) = store(32, 32); // one device slot, one host slot
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        let hold = s.hold(vec![a.id]);
        let Src {
            va,
            from_host,
            fence,
        } = s.gather_src(a.id).unwrap();
        assert!(from_host && fence.is_none());
        assert_eq!(ops.htod.load(Ordering::SeqCst), 0, "nothing copied back");
        assert_eq!(
            s.location(a.id),
            Some(Location::Host),
            "the block stays on the host"
        );
        assert_eq!(s.stats().host_reads, 1);
        // While the gather reads it, a publish meeting the block takes it as present and
        // a spill of another block cannot take its host slot.
        let again = s.acquire(0, key(1)).unwrap();
        assert!(again.fill.is_none(), "held on the host: not refilled");
        let mut b = s.acquire(0, key(2)).unwrap();
        seal(&mut b);
        s.release(&[b.id]);
        assert!(!s.spill(&[b.id]), "the only host slot is held");
        drop(hold);
        assert_ne!(va, 0);
    }

    /// A restore queues the copy and leaves both slots in place; a gather meanwhile reads the
    /// device slot behind the fence; `reap` lands it once the fence retired and the hold
    /// dropped. Nothing spills a restoring block.
    #[test]
    fn restore_async_queues_the_copy_and_a_gather_waits_on_its_fence() {
        let (ops, s) = store(32, 64); // one device slot
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        ops.pending.store(true, Ordering::SeqCst);
        assert!(s.restore_async(a.id).unwrap(), "queued");
        assert_eq!(
            (
                ops.htod_async.load(Ordering::SeqCst),
                ops.htod.load(Ordering::SeqCst)
            ),
            (1, 0)
        );
        assert_eq!(s.location(a.id), Some(Location::Device(a.va)));
        assert_eq!(
            s.stats().classes[0].host_blocks,
            1,
            "the host slot stays until the fence"
        );
        assert!(
            !s.spill(&[a.id]),
            "a restoring block is coming to the device"
        );
        let hold = s.hold(vec![a.id]);
        let src = s.gather_src(a.id).unwrap();
        assert_eq!((src.va, src.from_host), (a.va, false));
        let fence = src.fence.expect("restore in flight");
        assert_eq!(s.stats().host_reads, 0);
        s.reap();
        assert_eq!(s.stats().restores, 0, "fence not retired");
        ops.pending.store(false, Ordering::SeqCst);
        s.reap();
        s.reap();
        assert_eq!(
            s.stats().restores,
            0,
            "held: the host copy may still be read in place"
        );
        assert!(
            !ops.consumed.lock().contains(&fence),
            "the fence stays for the gather's stream wait while the block is held"
        );
        assert_eq!(s.gather_src(a.id).unwrap().fence, Some(fence));
        assert!(
            s.restoring_any(&[a.id]),
            "held: still in flight for admission"
        );
        drop(hold);
        assert!(!s.restoring_any(&[a.id]), "retired by the query itself");
        s.reap();
        let st = s.stats();
        assert_eq!(
            (
                st.restores,
                st.classes[0].host_blocks,
                st.classes[0].device_blocks
            ),
            (1, 0, 1)
        );
        assert_eq!(s.gather_src(a.id).unwrap().fence, None);
        assert_eq!(
            s.restore_async(a.id).unwrap(),
            false,
            "on the device: nothing to do"
        );
        assert_eq!(ops.htod_async.load(Ordering::SeqCst), 1);
        // The restored block is reclaimable again (zero dependents): its slot goes next.
        let b = s.acquire(0, key(2)).unwrap();
        assert_eq!(
            (b.fill.is_some(), b.va, s.stats().reclaims),
            (true, a.va, 1)
        );
    }

    #[test]
    fn restore_async_without_a_device_slot_leaves_the_block_on_the_host() {
        let (ops, s) = store(32, 64);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        let mut b = s.acquire(0, key(2)).unwrap(); // the only device slot, referenced
        seal(&mut b);
        assert!(s.restore_async(a.id).is_err());
        assert_eq!(ops.htod_async.load(Ordering::SeqCst), 0);
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.gather_src(a.id).unwrap().from_host, "read in place");
    }

    /// With deferred frees a shrink releases the chunk to the pool's accounting at once but
    /// the device free runs on the worker; until then the bytes count as coming free.
    #[test]
    fn deferred_frees_run_on_the_worker_and_count_as_coming_free() {
        let (ops, s) = store(64, 0); // two chunks
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut b = s.acquire(0, key(2)).unwrap();
        seal(&mut a);
        seal(&mut b);
        s.release(&[a.id, b.id]);
        s.defer_frees();
        assert_eq!(s.shrink(32), 32);
        assert_eq!(
            ops.freed.load(Ordering::SeqCst),
            0,
            "not freed on the caller's thread"
        );
        assert_eq!(s.pending_spill_bytes(), 32);
        assert!(s.free_step());
        assert_eq!(
            (ops.freed.load(Ordering::SeqCst), s.pending_spill_bytes()),
            (1, 0)
        );
        assert!(!s.free_step(), "nothing queued");
        assert_eq!(s.shrink(32), 32, "the second chunk goes the same way");
        drop(a);
        drop(b);
        drop(s);
        assert_eq!(
            ops.freed.load(Ordering::SeqCst),
            2,
            "a queued free is run at teardown"
        );
    }

    /// The compaction copies run with the store lock released; a block a gather holds
    /// meanwhile keeps its slot (the gather reads it on the engine stream) and the chunk is
    /// not released this round.
    #[test]
    fn compaction_copies_with_the_lock_released_and_skips_a_block_held_meanwhile() {
        let ops = MockOps::new();
        let s = Arc::new(KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            64, // two blocks per chunk
            2 * 64,
            0,
        ));
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        s.release(&ids[1..3]);
        // Both chunks are full (released blocks keep their slots): chunk 0 is planned first
        // and block 1 is the one that moves.
        let moving = ids[0];
        let held = Arc::clone(&s);
        let stash: Arc<parking_lot::Mutex<Option<Hold>>> = Arc::new(parking_lot::Mutex::new(None));
        let stashed = Arc::clone(&stash);
        *ops.on_dtod.lock() = Some(Box::new(move || {
            *stashed.lock() = Some(held.hold(vec![moving]));
        }));
        let before = s.location(moving);
        assert_eq!(s.shrink(64), 0, "the move was abandoned: nothing released");
        assert_eq!(
            ops.dtod.load(Ordering::SeqCst),
            1,
            "the copy ran (lock free)"
        );
        assert_eq!(s.location(moving), before, "block 1 stays in its slot");
        assert_eq!(s.stats().compactions, 0);
        *ops.on_dtod.lock() = None;
        let hold = stash.lock().take();
        drop(hold);
        assert_eq!(
            s.shrink(64),
            64,
            "the next step compacts the two live blocks"
        );
        let st = s.stats();
        assert_eq!((st.compactions, st.device_reserved_bytes), (1, 64));
        assert!(
            s.acquire(0, key(1)).unwrap().fill.is_none(),
            "block 1 is still present"
        );
        assert!(
            s.acquire(0, key(4)).unwrap().fill.is_none(),
            "block 4 is still present"
        );
    }

    #[test]
    fn shrink_compacts_a_sparse_chunk_into_holes_elsewhere() {
        let ops = MockOps::new();
        let s = Arc::new(KvStore::new(
            ops.clone(),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            64, // two blocks per chunk
            2 * 64,
            0,
        ));
        // Chunk 0 = blocks 1, 2; chunk 1 = blocks 3, 4. Blocks 2 and 3 die: a hole in each
        // chunk, none releasable as is.
        let ids: Vec<_> = (1..=4).map(|i| acquire_sealed(&s, 0, key(i)).id).collect();
        s.release(&ids[1..3]);
        assert_eq!(
            s.shrink(64),
            64,
            "block 4 moved into chunk 0's hole, chunk 1 released"
        );
        assert_eq!(ops.dtod.load(Ordering::SeqCst), 1);
        let st = s.stats();
        assert_eq!(
            (st.compactions, st.compact_bytes, st.device_reserved_bytes),
            (1, 32, 64)
        );
        assert_eq!(s.device_bytes(), 64, "blocks 1 and 4 remain, packed");
        assert!(matches!(s.location(ids[3]), Some(Location::Device(_))));
        assert!(
            s.acquire(0, key(4)).unwrap().fill.is_none(),
            "block 4 is still present"
        );
        // A held block pins its chunk: nothing to release.
        let hold = s.hold(vec![ids[0]]);
        s.release(&[ids[3]]);
        assert_eq!(s.shrink(64), 0);
        drop(hold);
    }

    /// A fill token dropped unsealed (an early return, a failed copy) fails its block: the slot
    /// is not pinned and the next publish of the key refills it. A sealed token's drop changes
    /// nothing.
    #[test]
    fn a_fill_dropped_unsealed_fails_its_block_and_the_next_acquire_refills() {
        let (_, s) = store(32, 0);
        let a = s.acquire(0, key(1)).unwrap();
        let id = a.id;
        assert_eq!(s.state(id), Some(State::Filling));
        drop(a);
        assert_eq!(s.state(id), Some(State::Failed));
        assert_eq!(s.stats().fill_failures, 1);
        let mut again = s.acquire(0, key(1)).unwrap();
        assert!(
            again.fill.is_some() && again.id == id,
            "refilled, not served"
        );
        seal(&mut again);
        drop(again);
        assert_eq!(s.state(id), Some(State::Ready));
        assert_eq!(s.stats().fill_failures, 1);
    }

    /// An unwind leaves a fill `Filling` and its slot pinned: a copy submitted before the
    /// panic may still write it.
    #[test]
    fn a_fill_dropped_by_an_unwind_stays_filling() {
        let (_, s) = store(32, 0);
        let id = std::cell::Cell::new(None);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let a = s.acquire(0, key(1)).unwrap();
            id.set(Some(a.id));
            // No dependent either: only the `Filling` state pins the slot below.
            s.release(&[a.id]);
            panic!("mid-publish");
        }));
        assert!(r.is_err());
        let id = id.get().expect("acquired");
        assert_eq!(s.state(id), Some(State::Filling));
        assert_eq!(s.stats().fill_failures, 0);
        assert!(s.acquire(0, key(2)).is_err(), "the slot stays pinned");
    }

    /// An unwind leaves a gather's holds in place: a copy submitted before the panic may
    /// still read the blocks. `abandon` keeps them on purpose (teardown).
    #[test]
    fn holds_dropped_by_an_unwind_or_abandoned_stay() {
        let (_, s) = store(32, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _hold = s.hold(vec![a.id]);
            panic!("mid-gather");
        }));
        assert!(r.is_err());
        assert!(s.acquire(0, key(2)).is_err(), "still held");
        s.unhold(&[a.id]);
        s.hold(vec![a.id]).abandon();
        assert!(s.acquire(0, key(2)).is_err(), "still held");
        s.unhold(&[a.id]);
        assert!(s.acquire(0, key(2)).is_ok());
    }

    /// A hold guard dropped on an error path (`?`, early return) unpins its blocks.
    #[test]
    fn a_hold_guard_dropped_on_an_error_path_unpins_the_block() {
        let (_, s) = store(32, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        seal(&mut a);
        s.release(&[a.id]);
        let gather = |fail: bool| -> Result<()> {
            let _hold = s.hold(vec![a.id]);
            assert!(s.acquire(0, key(2)).is_err(), "held");
            if fail {
                return Err(RuntimeError::Rejected("drain failed".into()));
            }
            Ok(())
        };
        assert!(gather(true).is_err());
        assert!(s.acquire(0, key(2)).is_ok(), "the hold went with the guard");
    }

    #[test]
    fn evict_frees_only_unreferenced_settled_blocks() {
        let (_, s) = store(1024, 0);
        let mut a = s.acquire(0, key(1)).unwrap();
        let mut b = s.acquire(0, key(2)).unwrap();
        seal(&mut a);
        seal(&mut b);
        s.release(&[b.id]);
        assert_eq!(s.evict(&[a.id, b.id]), 1);
        assert_eq!(s.stats().classes[0].device_blocks, 1);
        assert!(s.acquire(0, key(1)).map(|x| x.fill.is_none()).unwrap());
    }

    /// Random publishes, releases, gathers, spills, restores, evictions and shrinks over two
    /// classes with the host tier on; every block's bytes are tracked per address through the
    /// copies, and a gather must always read the bytes its key was filled with.
    #[test]
    fn model_every_gather_reads_its_own_key() {
        let spec = |name| ClassSpec {
            name,
            row_bytes: 8,
            rows_per_block: 4,
        };
        for seed in 1..=24u64 {
            let ops = MockOps::new();
            let s = Arc::new(KvStore::new(
                ops.clone(),
                vec![spec("window"), spec("tail")],
                64, // two blocks per chunk
                6 * 64,
                4 * 64,
            ));
            while s.host_grow_step() {}
            let mut rng = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            let mut next = move |n: u64| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng % n
            };
            let tag = |class: usize, k: u64| (class as u64 + 1) * 1000 + k;
            // Dependencies held: (class, key, id).
            let mut deps: Vec<(usize, u64, BlockId)> = Vec::new();
            // Every id ever seen with its class and key (gathers try present ones).
            let mut seen: Vec<(usize, u64, BlockId)> = Vec::new();
            let mut holds: Vec<Hold> = Vec::new();
            for step in 0..600 {
                match next(10) {
                    0..=3 => {
                        let class = next(2) as usize;
                        let k = next(10);
                        let Ok(a) = s.acquire(class, key(k)) else {
                            // Full: give the class a boundary back.
                            if let Some(i) = (0..deps.len()).find(|&i| deps[i].0 == class) {
                                let (_, _, id) = deps.swap_remove(i);
                                s.release(&[id]);
                            }
                            continue;
                        };
                        if let Some(fill) = a.fill {
                            ops.mem.lock().insert(a.va, tag(class, k));
                            if next(8) == 0 {
                                fill.fail();
                            } else {
                                fill.seal();
                            }
                        } else if s.state(a.id) == Some(State::Ready) {
                            let _h = s.hold(vec![a.id]);
                            let src = s.gather_src(a.id).unwrap();
                            assert_eq!(
                                ops.mem.lock().get(&src.va).copied(),
                                Some(tag(class, k)),
                                "seed {seed} step {step}: dedup hit of class {class} key {k} reads other bytes"
                            );
                        }
                        deps.push((class, k, a.id));
                        seen.push((class, k, a.id));
                    }
                    4 => {
                        if !deps.is_empty() {
                            let (_, _, id) = deps.swap_remove(next(deps.len() as u64) as usize);
                            s.release(&[id]);
                        }
                    }
                    5 => {
                        if !seen.is_empty() {
                            let (class, k, id) = seen[next(seen.len() as u64) as usize];
                            if s.state(id) == Some(State::Ready) {
                                let h = s.hold(vec![id]);
                                let src = s.gather_src(id).unwrap();
                                assert_eq!(
                                    ops.mem.lock().get(&src.va).copied(),
                                    Some(tag(class, k)),
                                    "seed {seed} step {step}: gather of class {class} key {k} reads other bytes"
                                );
                                if next(3) == 0 {
                                    holds.push(h);
                                }
                            }
                        }
                    }
                    6 => {
                        let ids: Vec<BlockId> = seen
                            .iter()
                            .filter(|_| next(2) == 0)
                            .map(|&(_, _, id)| id)
                            .collect();
                        s.spill(&ids);
                    }
                    7 => {
                        ops.pending.store(next(3) == 0, Ordering::SeqCst);
                        s.reap();
                        ops.pending.store(false, Ordering::SeqCst);
                        if !seen.is_empty() {
                            let (_, _, id) = seen[next(seen.len() as u64) as usize];
                            let _ = s.restore_async(id);
                        }
                    }
                    8 => {
                        holds.clear();
                        let ids: Vec<BlockId> = seen
                            .iter()
                            .filter(|_| next(2) == 0)
                            .map(|&(_, _, id)| id)
                            .collect();
                        s.evict(&ids);
                    }
                    _ => {
                        s.shrink(64 * (1 + next(3)));
                    }
                }
                // Live blocks never alias on either tier.
                let mut places: Vec<(bool, u64)> = Vec::new();
                for &(_, _, id) in &seen {
                    let Some(loc) = s.location(id) else { continue };
                    let va = match loc {
                        Location::Device(va) => va,
                        Location::Host => {
                            let _h = s.hold(vec![id]);
                            s.gather_src(id).unwrap().va
                        }
                    };
                    places.push((matches!(loc, Location::Host), va));
                }
                places.sort_unstable();
                places.dedup();
                let n = seen
                    .iter()
                    .filter(|&&(_, _, id)| s.location(id).is_some())
                    .map(|&(c, k, _)| (c, k))
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                assert_eq!(
                    places.len(),
                    n,
                    "seed {seed} step {step}: two live blocks share an address"
                );
            }
        }
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
