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

/// Where a block's bytes are. `Spilling` and `Restoring` keep both slots until the fence
/// retires; a restoring block's device slot is complete only behind its fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Where {
    Device(u32),
    Spilling { dev: u32, host: u32, fence: u64 },
    Host(u32),
    Restoring { dev: u32, host: u32, fence: u64 },
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
    moves: Vec<(BlockId, u32, u32)>,
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

/// A slab of equal-size slots on one tier. Slabs of one tier share the tier's byte budget:
/// a class grows a chunk only while the budget allows, and chunks never move between classes.
struct Slab {
    slot_bytes: u64,
    chunk_slots: u32,
    chunks: Vec<u64>,
    /// Free slots per chunk.
    free: Vec<Vec<u32>>,
    /// Chunks with free slots as `(free slots, chunk)`: the fullest first, the emptiest last.
    by_free: BTreeSet<(u32, u32)>,
    /// The block in each slot.
    owner: Vec<Option<BlockId>>,
    /// Released chunk indices, reused by the next growth.
    dead: Vec<u32>,
    live_chunks: u32,
    free_slots: u64,
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

    fn va(&self, slot: u32) -> u64 {
        self.chunks[(slot / self.chunk_slots) as usize]
            + u64::from(slot % self.chunk_slots) * self.slot_bytes
    }

    fn reserved_bytes(&self) -> u64 {
        u64::from(self.live_chunks) * self.chunk_bytes()
    }

    fn has_free(&self) -> bool {
        self.free_slots > 0
    }

    fn used_bytes(&self) -> u64 {
        self.reserved_bytes() - self.free_slots * self.slot_bytes
    }

    /// A free slot from the fullest chunk that has one, so that emptier chunks keep draining
    /// towards release; a new chunk when none is free and `budget` (bytes left on the tier)
    /// covers it.
    fn take(&mut self, budget: u64, grow: impl FnOnce(u64) -> Result<u64>) -> Result<Option<u32>> {
        if let Some(&(n, c)) = self.by_free.iter().next() {
            self.by_free.remove(&(n, c));
            let s = self.free[c as usize].pop().expect("counted free slot");
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
        let base = grow(u64::from(n) * self.slot_bytes)?;
        let c = match self.dead.pop() {
            Some(c) => {
                self.chunks[c as usize] = base;
                c
            }
            None => {
                self.chunks.push(base);
                self.free.push(Vec::new());
                self.chunks.len() as u32 - 1
            }
        };
        let first = c * n;
        // The chunk's slots go out in order; the first to the caller.
        self.free[c as usize] = (first + 1..first + n).rev().collect();
        if self.owner.len() < (first + n) as usize {
            self.owner.resize((first + n) as usize, None);
        }
        if n > 1 {
            self.by_free.insert((n - 1, c));
        }
        self.free_slots += u64::from(n - 1);
        self.live_chunks += 1;
        Ok(Some(first))
    }

    /// A free slot in any chunk but `not`, the fullest first; no growth.
    fn take_outside(&mut self, not: u32) -> Option<u32> {
        let &(n, c) = self.by_free.iter().find(|&&(_, c)| c != not)?;
        self.by_free.remove(&(n, c));
        let s = self.free[c as usize].pop().expect("counted free slot");
        if n > 1 {
            self.by_free.insert((n - 1, c));
        }
        self.free_slots -= 1;
        Some(s)
    }

    fn holes_outside(&self, not: u32) -> u64 {
        self.free_slots - self.free[not as usize].len() as u64
    }

    fn chunk_of(&self, slot: u32) -> u32 {
        slot / self.chunk_slots
    }

    fn assign(&mut self, slot: u32, id: BlockId) {
        self.owner[slot as usize] = Some(id);
    }

    /// Return `slot` to its chunk.
    fn put(&mut self, slot: u32) {
        self.owner[slot as usize] = None;
        let c = slot / self.chunk_slots;
        let n = self.free[c as usize].len() as u32;
        if n > 0 {
            self.by_free.remove(&(n, c));
        }
        self.free[c as usize].push(slot);
        self.by_free.insert((n + 1, c));
        self.free_slots += 1;
    }

    /// Release every chunk whose slots are all free (the chunk index stays, dead, so slot
    /// arithmetic is unchanged). Returns the bytes released.
    fn release_empty_chunks(&mut self, mut free: impl FnMut(u64)) -> u64 {
        let n = self.chunk_slots;
        let mut released = 0;
        while let Some(&(count, c)) = self.by_free.iter().next_back() {
            if count < n {
                break;
            }
            self.by_free.remove(&(count, c));
            self.free[c as usize].clear();
            self.free_slots -= u64::from(n);
            free(self.chunks[c as usize]);
            self.chunks[c as usize] = 0;
            self.dead.push(c);
            self.live_chunks -= 1;
            released += self.chunk_bytes();
        }
        released
    }

    /// Every live chunk, the ones with the most free slots first, then the full ones.
    fn release_candidates(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self.by_free.iter().rev().map(|&(_, c)| c).collect();
        out.extend(
            (0..self.chunks.len() as u32)
                .filter(|&c| self.chunks[c as usize] != 0 && self.free[c as usize].is_empty()),
        );
        out
    }

    fn occupants(&self, chunk: u32) -> Vec<BlockId> {
        let first = (chunk * self.chunk_slots) as usize;
        self.owner[first..first + self.chunk_slots as usize]
            .iter()
            .flatten()
            .copied()
            .collect()
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
    host_cap: u64,
    /// Host RAM the tier leaves to the rest of the machine; growth stops (the cap freezes at
    /// what is reserved) once `MemAvailable` would fall under it. 0 = unchecked.
    host_reserve: u64,
    blocks: Vec<Option<Block>>,
    free_ids: Vec<BlockId>,
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
    /// Pinned chunks per class allocated ahead by [`KvStore::host_grow_step`] (a pinned
    /// allocation costs ~4 ms per 8 MiB; under the pool lock it was the host tier's TTFT tail).
    host_spare: Vec<Vec<u64>>,
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

/// Host chunks allocated between two `MemAvailable` reads of the host-reserve check.
const HOST_FLOOR_EVERY: u64 = 16;
/// Device chunks allocated between two free-memory queries of the device-floor check.
const DEVICE_FLOOR_EVERY: u64 = 8;
/// Most bytes one `shrink` call releases (lock hold bound; the caller loops).
const SHRINK_STEP_BYTES: u64 = 128 << 20;
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
                host_cap,
                host_reserve: 0,
                blocks: Vec::new(),
                free_ids: Vec::new(),
                index: FxHashMap::default(),
                reclaim: (0..n_classes).map(|_| VecDeque::new()).collect(),
                host_reclaim: (0..n_classes).map(|_| VecDeque::new()).collect(),
                spilling: VecDeque::new(),
                spilling_bytes: 0,
                restoring: VecDeque::new(),
                host_spare: (0..n_classes).map(|_| Vec::new()).collect(),
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
            // A host-resident block is refilled from the rows being published (one more segment
            // of the publish's launch), never copied back: the host slot goes.
            // A held host block is being read in place (a gather in flight): it stays where it
            // is and the publish takes it as present.
            let (refill, held_host) =
                match inner.blocks[id as usize].as_ref().map(|b| (b.place, b.holds)) {
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
                    inner.classes[class].host.put(h);
                    let st = &mut inner.stats.classes[class];
                    st.host_blocks -= 1;
                    st.host_bytes -= bytes;
                    st.device_blocks += 1;
                    st.device_bytes += bytes;
                    let b = inner.blocks[id as usize].as_mut().expect("indexed block");
                    b.place = Where::Device(slot);
                    b.state = State::Filling;
                    inner.classes[class].device.va(slot)
                }
                (None, None) => self.ensure_device_locked(&mut inner, id)?,
            };
            let b = inner.blocks[id as usize].as_mut().expect("indexed block");
            b.dependents += 1;
            b.tick = tick;
            if b.state == State::Filling && refill.is_some() {
                inner.stats.fills += 1;
                inner.stats.refills += 1;
                return Ok(Acquire { id, va, fill: true });
            }
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
        inner.classes[class].device.assign(slot, id);
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
                let class = b.class;
                match b.place {
                    Where::Host(_) => inner.host_reclaim[class].push_back(id),
                    _ => inner.reclaim[class].push_back(id),
                }
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
        let b = inner.blocks[id as usize]
            .as_mut()
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
    /// made: the block stays on the host and gathers read it in place.
    pub fn restore_async(&self, id: BlockId) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let (class, place, state) = {
            let b = inner.blocks[id as usize]
                .as_mut()
                .ok_or_else(|| RuntimeError::Rejected("kv store: no such block".into()))?;
            b.tick = tick;
            (b.class, b.place, b.state)
        };
        let Where::Host(h) = place else {
            return Ok(());
        };
        if state != State::Ready {
            return self.ensure_device_locked(&mut inner, id).map(|_| ());
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
        let b = inner.blocks[id as usize].as_mut().expect("block");
        b.place = Where::Restoring {
            dev: slot,
            host: h,
            fence,
        };
        inner.restoring.push_back(id);
        Ok(())
    }

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
        let mut budget = inner.device_cap.saturating_sub(reserved);
        if budget > 0
            && !inner.classes[class].device.has_free()
            && inner.device_floor > 0
            && inner.stats.device_grows % DEVICE_FLOOR_EVERY == 0
        {
            let chunk = inner.classes[class].device.chunk_bytes();
            if let Some(free) = ops.free_bytes() {
                if free < chunk.saturating_mul(DEVICE_FLOOR_EVERY).saturating_add(inner.device_floor) {
                    budget = 0;
                }
            }
        }
        let mut grow_us = None;
        let taken = inner.classes[class].device.take(budget, |bytes| {
            let t = std::time::Instant::now();
            let r = ops.alloc(bytes);
            grow_us = Some(t.elapsed().as_micros() as u64);
            r
        });
        if let Some(us) = grow_us {
            inner.stats.device_grows += 1;
            inner.stats.device_grow_us += us;
        }
        if let Some(s) = taken? {
            return Ok(s);
        }
        // Reclaim in release order; stale entries (re-acquired, dropped, wrong tier) are
        // skipped and re-queued where they still matter.
        let mut requeue = Vec::new();
        let mut found = None;
        while let Some(id) = inner.reclaim[class].pop_front() {
            let Some(b) = inner.blocks[id as usize].as_ref() else {
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
        let Some(id) = found else {
            inner.stats.refusals += 1;
            return Err(RuntimeError::Oom("kv store: class slab full".into()));
        };
        // `drop_block` returns the victim's slot to its chunk; the take may prefer another
        // hole of the class (the fullest chunk's).
        self.drop_block(inner, id);
        inner.stats.reclaims += 1;
        let taken = inner.classes[class]
            .device
            .take(0, |_| Err(RuntimeError::Oom("kv store: no growth".into())))?;
        Ok(taken.expect("the reclaimed slot is free"))
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
            let Some(b) = inner.blocks[id as usize].as_ref() else {
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
    pub fn spill(&self, ids: &[BlockId]) -> bool {
        let mut inner = self.inner.lock();
        if inner.host_cap == 0 {
            return false;
        }
        let t0 = std::time::Instant::now();
        let mut all = true;
        for &id in ids {
            let Some(b) = inner.blocks[id as usize].as_ref() else {
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
                    let b = inner.blocks[id as usize].as_mut().expect("block");
                    b.place = Where::Spilling { dev, host: h, fence };
                    inner.spilling.push_back((id, bytes));
                    inner.spilling_bytes += bytes;
                    inner.stats.spills += 1;
                }
                Err(_) => {
                    inner.classes[class].host.put(h);
                    all = false;
                }
            }
        }
        inner.stats.spill_us += t0.elapsed().as_micros() as u64;
        all
    }

    fn host_slot(&self, inner: &mut Inner, class: usize) -> Option<u32> {
        // Spares are already paid for against the cap: only live chunks count here.
        let reserved: u64 = inner.classes.iter().map(|c| c.host.reserved_bytes()).sum();
        let budget = inner.host_cap.saturating_sub(reserved);
        let Inner { classes, host_spare, .. } = inner;
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
        // (re-acquired, dropped, back on the device) are skipped, other classes' re-queued.
        let mut requeue = Vec::new();
        let mut victim = None;
        while let Some(id) = inner.host_reclaim[class].pop_front() {
            let Some(b) = inner.blocks[id as usize].as_ref() else {
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
            let (class, dev, host, fence, holds) = match inner.blocks[id as usize].as_ref() {
                Some(Block {
                    class,
                    place: Where::Spilling { dev, host, fence },
                    holds,
                    ..
                }) => (*class, *dev, *host, *fence, *holds),
                // Dropped or restored meanwhile: nothing to reap.
                _ => {
                    inner.spilling_bytes -= queued;
                    continue;
                }
            };
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
            // A transfer still reads the device copy: keep it until the hold drops.
            if holds > 0 {
                inner.spilling.push_back((id, queued));
                continue;
            }
            inner.spilling_bytes -= queued;
            let bytes = inner.classes[class].spec.block_bytes();
            inner.classes[class].device.put(dev);
            let b = inner.blocks[id as usize].as_mut().expect("block");
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

    fn reap_restores(&self, inner: &mut Inner) {
        let mut left = inner.restoring.len();
        while left > 0 {
            left -= 1;
            let Some(id) = inner.restoring.pop_front() else {
                return;
            };
            let (class, host, fence, holds) = match inner.blocks[id as usize].as_ref() {
                Some(Block {
                    class,
                    place: Where::Restoring { host, fence, .. },
                    holds,
                    ..
                }) => (*class, *host, *fence, *holds),
                _ => continue,
            };
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
            // A gather may still read the host copy in place (it started before the restore)
            // or wait on this fence: both slots stay until the hold drops.
            if holds > 0 {
                inner.restoring.push_back(id);
                continue;
            }
            let bytes = inner.classes[class].spec.block_bytes();
            inner.classes[class].host.put(host);
            let b = inner.blocks[id as usize].as_mut().expect("block");
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
        for va in frees.drain(..) {
            self.ops.free(va);
        }
        let copied = match &plan {
            Some(p) if !p.copies.is_empty() => self.ops.copy_dtod_batch(&p.copies),
            _ => Ok(()),
        };
        let mut inner = self.inner.lock();
        if let Some(plan) = plan {
            released += self.compaction_commit(&mut inner, plan, copied, &mut frees);
        }
        let us = t0.elapsed().as_micros() as u64;
        inner.stats.shrink_us += us;
        inner.stats.shrink_max_us = inner.stats.shrink_max_us.max(us);
        inner.stats.shrinks += 1;
        drop(inner);
        for va in frees {
            self.ops.free(va);
        }
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
        frees: &mut Vec<u64>,
    ) -> u64 {
        let mut released = 0u64;
        for ci in 0..inner.classes.len() {
            for chunk in inner.classes[ci].device.release_candidates() {
                if released >= bytes {
                    break;
                }
                let ids = inner.classes[ci].device.occupants(chunk);
                let reclaimable = ids.iter().all(|&id| match inner.blocks[id as usize].as_ref() {
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
                    .release_empty_chunks(|va| frees.push(va));
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
                    match inner.blocks[id as usize].as_ref() {
                        Some(b) if b.holds > 0 || b.state == State::Filling => continue 'chunks,
                        Some(b) if !matches!(b.place, Where::Device(_)) => continue 'chunks,
                        // A refill may write it while the copy reads it.
                        Some(b) if b.state == State::Failed && b.dependents > 0 => continue 'chunks,
                        Some(b) if b.dependents > 0 => live += 1,
                        _ => {}
                    }
                }
                let mut deficit = live.saturating_sub(inner.classes[ci].device.holes_outside(chunk));
                if deficit > 0 {
                    let mut requeue = Vec::new();
                    while deficit > 0 {
                        let Some(id) = inner.reclaim[ci].pop_front() else {
                            break;
                        };
                        let Some(b) = inner.blocks[id as usize].as_ref() else {
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
                let mut moves: Vec<(BlockId, u32, u32)> = Vec::new();
                let mut copies: Vec<(u64, u64, u64)> = Vec::new();
                for id in ids {
                    let Some(b) = inner.blocks[id as usize].as_ref() else {
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
                    inner.blocks[id as usize].as_mut().expect("block").holds += 1;
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
        frees: &mut Vec<u64>,
    ) -> u64 {
        let ci = plan.class;
        let bytes_per = inner.classes[ci].spec.block_bytes();
        let mut moved = 0u64;
        for &(id, from, to) in &plan.moves {
            let lands = copied.is_ok()
                && matches!(
                    inner.blocks[id as usize].as_ref(),
                    Some(b) if b.holds == 1 && b.place == Where::Device(from)
                );
            if lands {
                inner.classes[ci].device.assign(to, id);
                inner.classes[ci].device.put(from);
                inner.blocks[id as usize].as_mut().expect("block").place = Where::Device(to);
                moved += 1;
            } else {
                inner.classes[ci].device.put(to);
            }
            if let Some(b) = inner.blocks[id as usize].as_mut() {
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
            released += c.device.release_empty_chunks(|va| frees.push(va));
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
            Ok(ptr) => {
                inner.host_spare[class].push(ptr);
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
    pub fn pending_spill_bytes(&self) -> u64 {
        self.inner.lock().spilling_bytes
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
        s.device_reserved_bytes = inner.classes.iter().map(|c| c.device.reserved_bytes()).sum();
        s
    }

    pub fn device_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        inner.classes.iter().map(|c| c.device.used_bytes()).sum()
    }

    /// Device bytes the store's slabs occupy, used or not: what it took from free memory.
    pub fn device_reserved_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        inner.classes.iter().map(|c| c.device.reserved_bytes()).sum()
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
        let reserved: u64 = inner.classes.iter().map(|c| c.device.reserved_bytes()).sum();
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
        let inner = self.inner.get_mut();
        for c in &inner.classes {
            for &va in c.device.chunks.iter().filter(|&&va| va != 0) {
                self.ops.free(va);
            }
            let chunk = c.host.chunk_bytes();
            for &ptr in &c.host.chunks {
                self.ops.host_free(ptr, chunk);
            }
        }
        for (c, spare) in inner.classes.iter().zip(&inner.host_spare) {
            for &ptr in spare {
                self.ops.host_free(ptr, c.host.chunk_bytes());
            }
        }
    }
}

/// Chained 64-bit hashes of `tokens` per `rows` rows: entry `i` covers `tokens[..(i+1)*rows]`,
/// hashed with `H` from `seed`. The key's `chain` and `check` use two unrelated hash
/// functions (`FxHasher` and SipHash), not two seeds of one.
pub fn chain_hashes<H: std::hash::Hasher + Default>(tokens: &[u32], rows: u32, seed: u64) -> Vec<u64> {
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
        htod_async: AtomicU64,
        dtod: AtomicU64,
        /// Every fence reads as unretired while set.
        pending: std::sync::atomic::AtomicBool,
        /// Runs inside `copy_dtod_batch` (the store lock must be free there).
        on_dtod: parking_lot::Mutex<Option<Box<dyn Fn() + Send>>>,
    }

    impl MockOps {
        fn new() -> Arc<Self> {
            Arc::new(MockOps {
                next: AtomicU64::new(1 << 20),
                dev_alloc: AtomicU64::new(0),
                host_alloc: AtomicU64::new(0),
                htod: AtomicU64::new(0),
                htod_async: AtomicU64::new(0),
                dtod: AtomicU64::new(0),
                pending: std::sync::atomic::AtomicBool::new(false),
                on_dtod: parking_lot::Mutex::new(None),
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
            Ok(!self.pending.load(Ordering::SeqCst))
        }
        fn copy_htod(&self, _va: u64, _host: u64, _bytes: u64) -> Result<()> {
            self.htod.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn copy_htod_async(&self, _va: u64, _host: u64, _bytes: u64) -> Result<u64> {
            self.htod_async.fetch_add(1, Ordering::SeqCst);
            Ok(9)
        }
        fn copy_dtod_batch(&self, copies: &[(u64, u64, u64)]) -> Result<()> {
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
        while s.host_grow_step() {}
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
        // A publish that meets the host block refills it from its own rows (b's slot is
        // reclaimed); nothing is copied back and the host slot goes.
        let again = s.acquire(0, key(1)).unwrap();
        assert!(again.fill, "refilled on the device, not restored");
        assert_eq!(ops.htod.load(Ordering::SeqCst), 0);
        assert_eq!((s.stats().refills, s.stats().classes[0].host_blocks), (1, 0));
        assert_eq!(s.location(a.id), Some(Location::Device(again.va)));
        // A gather (`ensure_device`) is the restore.
        s.seal(again.id);
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
        let a = s.acquire(0, key(1)).unwrap();
        let b = s.acquire(0, key(2)).unwrap();
        s.seal(a.id);
        s.seal(b.id);
        s.release(&[a.id, b.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.spill(&[b.id]), "a's host slot is reclaimed, not refused");
        s.reap();
        assert_eq!(s.location(a.id), None, "the oldest zero-dependent host block gave way");
        assert_eq!(s.location(b.id), Some(Location::Host));
        assert_eq!(s.stats().classes[0].host_blocks, 1);
        assert_eq!(ops.host_alloc.load(Ordering::SeqCst), 32, "no second chunk");
    }

    #[test]
    fn device_headroom_counts_growth_free_slots_and_reclaimable_blocks() {
        let (_, s) = store(4 * 32, 0); // chunk = 32 B = one block
        assert_eq!(s.device_headroom_bytes(), 4 * 32, "all growth");
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        assert_eq!(s.device_headroom_bytes(), 3 * 32, "one block live");
        s.release(&[a.id]);
        assert_eq!(s.device_headroom_bytes(), 4 * 32, "released: reclaimable again");
        s.set_device_cap(32);
        assert_eq!(s.device_headroom_bytes(), 32, "no growth under the cap, one reclaimable");
    }

    #[test]
    fn a_reclaimed_slot_is_handed_out_once() {
        let (_, s) = store(2 * 32, 0);
        let a = s.acquire(0, key(1)).unwrap();
        let b = s.acquire(0, key(2)).unwrap();
        s.seal(a.id);
        s.seal(b.id);
        s.release(&[a.id, b.id]);
        let c = s.acquire(0, key(3)).unwrap();
        let d = s.acquire(0, key(4)).unwrap();
        assert_ne!(c.va, d.va, "two live blocks never alias");
        assert!(s.acquire(0, key(5)).is_err());
    }

    #[test]
    fn a_held_spilling_block_keeps_its_device_copy_until_unheld() {
        let (_, s) = store(32, 64);
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.hold(&[a.id]);
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Device(a.va)), "held: not retired");
        assert!(s.acquire(0, key(2)).is_err(), "its device slot is not free");
        s.unhold(&[a.id]);
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.acquire(0, key(2)).is_ok());
    }

    #[test]
    fn shrink_drops_reclaimable_blocks_and_releases_empty_chunks() {
        let (_, s) = store(4 * 32, 0); // chunk = 32 B = one block
        let ids: Vec<_> = (1..=4).map(|i| s.acquire(0, key(i)).unwrap().id).collect();
        for &id in &ids {
            s.seal(id);
        }
        s.release(&ids[..2]);
        assert_eq!(s.device_bytes(), 4 * 32);
        let released = s.shrink(64);
        assert_eq!(released, 64, "two empty chunks went back");
        assert_eq!(s.device_bytes(), 2 * 32);
        assert!(s.acquire(0, key(1)).unwrap().fill, "dropped, refilled");
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
        let ids: Vec<_> = (1..=4).map(|i| s.acquire(0, key(i)).unwrap().id).collect();
        for &id in &ids {
            s.seal(id);
        }
        assert_eq!(s.device_bytes(), 2 * 64, "chunk 0 = blocks 1,2; chunk 1 = blocks 3,4");
        // A dead block in each chunk: compaction packs the two live ones into one chunk.
        s.release(&ids[1..3]);
        assert_eq!(s.shrink(64), 64);
        assert_eq!(s.device_bytes(), 64, "blocks 1 and 4 remain");
        assert_eq!(ops.dtod.load(Ordering::SeqCst), 1);
        // Under the cap a new block grows a chunk again (one slot of it free).
        let e = s.acquire(0, key(5)).unwrap();
        assert!(e.fill);
        assert_eq!(s.device_bytes(), 64 + 32);
        assert_eq!(s.shrink(64), 0, "every block is referenced: nothing to release");
        // Block 4 dies: its slot takes block 5, the sparse chunk goes back.
        s.release(&ids[3..]);
        assert_eq!(s.shrink(64), 64);
        assert_eq!(s.device_bytes(), 64, "blocks 1 and 5, packed");
        assert!(s.acquire(0, key(4)).unwrap().fill, "block 4 was dropped");
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
        let ids: Vec<_> = (1..=80).map(|i| s.acquire(0, key(i)).unwrap().id).collect();
        for &id in &ids {
            s.seal(id);
        }
        for c in 0..40 {
            s.release(&[ids[2 * c + 1]]);
        }
        s.release(&[ids[78]]);
        assert_eq!(s.shrink(64), 64, "the last chunk is the only releasable one");
        assert_eq!(s.device_bytes(), 39 * 64);
        // Growth reuses the released chunk index: no new metadata.
        let before = s.inner.lock().classes[0].device.chunks.len();
        let n = s.acquire(0, key(100)).unwrap();
        assert!(n.fill);
        assert_eq!(s.inner.lock().classes[0].device.chunks.len(), before);
        assert_eq!(s.device_bytes(), 40 * 64 - 32, "one slot of the regrown chunk is free");
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
        let a = s.acquire(0, key(1)).unwrap();
        let t = s.acquire(1, key(2)).unwrap();
        s.seal(a.id);
        s.seal(t.id);
        s.release(&[a.id, t.id]);
        assert!(s.spill(&[a.id, t.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        // A second window block spills: the host is full for its class, the window block on
        // the host is the victim, the tail block stays.
        let a2 = s.acquire(0, key(3)).unwrap();
        s.seal(a2.id);
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
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        s.hold(&[a.id]);
        let Src { va, from_host, fence } = s.gather_src(a.id).unwrap();
        assert!(from_host && fence.is_none());
        assert_eq!(ops.htod.load(Ordering::SeqCst), 0, "nothing copied back");
        assert_eq!(s.location(a.id), Some(Location::Host), "the block stays on the host");
        assert_eq!(s.stats().host_reads, 1);
        // While the gather reads it, a publish meeting the block takes it as present and
        // a spill of another block cannot take its host slot.
        let again = s.acquire(0, key(1)).unwrap();
        assert!(!again.fill, "held on the host: not refilled");
        let b = s.acquire(0, key(2)).unwrap();
        s.seal(b.id);
        s.release(&[b.id]);
        assert!(!s.spill(&[b.id]), "the only host slot is held");
        s.unhold(&[a.id]);
        assert_ne!(va, 0);
    }

    /// A restore queues the copy and leaves both slots in place; a gather meanwhile reads the
    /// device slot behind the fence; `reap` lands it once the fence retired and the hold
    /// dropped. Nothing spills a restoring block.
    #[test]
    fn restore_async_queues_the_copy_and_a_gather_waits_on_its_fence() {
        let (ops, s) = store(32, 64); // one device slot
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        assert_eq!(s.location(a.id), Some(Location::Host));
        ops.pending.store(true, Ordering::SeqCst);
        s.restore_async(a.id).unwrap();
        assert_eq!(
            (ops.htod_async.load(Ordering::SeqCst), ops.htod.load(Ordering::SeqCst)),
            (1, 0)
        );
        assert_eq!(s.location(a.id), Some(Location::Device(a.va)));
        assert_eq!(s.stats().classes[0].host_blocks, 1, "the host slot stays until the fence");
        assert!(!s.spill(&[a.id]), "a restoring block is coming to the device");
        s.hold(&[a.id]);
        let src = s.gather_src(a.id).unwrap();
        assert_eq!(
            src,
            Src {
                va: a.va,
                from_host: false,
                fence: Some(9)
            }
        );
        assert_eq!(s.stats().host_reads, 0);
        s.reap();
        assert_eq!(s.stats().restores, 0, "fence not retired");
        ops.pending.store(false, Ordering::SeqCst);
        s.reap();
        assert_eq!(s.stats().restores, 0, "held: the host copy may still be read in place");
        s.unhold(&[a.id]);
        s.reap();
        let st = s.stats();
        assert_eq!(
            (st.restores, st.classes[0].host_blocks, st.classes[0].device_blocks),
            (1, 0, 1)
        );
        assert_eq!(s.gather_src(a.id).unwrap().fence, None);
        assert!(s.restore_async(a.id).is_ok(), "on the device: nothing to do");
        assert_eq!(ops.htod_async.load(Ordering::SeqCst), 1);
        // The restored block is reclaimable again (zero dependents): its slot goes next.
        let b = s.acquire(0, key(2)).unwrap();
        assert_eq!((b.fill, b.va, s.stats().reclaims), (true, a.va, 1));
    }

    #[test]
    fn restore_async_without_a_device_slot_leaves_the_block_on_the_host() {
        let (ops, s) = store(32, 64);
        let a = s.acquire(0, key(1)).unwrap();
        s.seal(a.id);
        s.release(&[a.id]);
        assert!(s.spill(&[a.id]));
        s.reap();
        let b = s.acquire(0, key(2)).unwrap(); // the only device slot, referenced
        s.seal(b.id);
        assert!(s.restore_async(a.id).is_err());
        assert_eq!(ops.htod_async.load(Ordering::SeqCst), 0);
        assert_eq!(s.location(a.id), Some(Location::Host));
        assert!(s.gather_src(a.id).unwrap().from_host, "read in place");
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
        let ids: Vec<_> = (1..=4).map(|i| s.acquire(0, key(i)).unwrap().id).collect();
        for &id in &ids {
            s.seal(id);
        }
        s.release(&ids[1..3]);
        // Both chunks are full (released blocks keep their slots): chunk 0 is planned first
        // and block 1 is the one that moves.
        let moving = ids[0];
        let held = Arc::clone(&s);
        *ops.on_dtod.lock() = Some(Box::new(move || held.hold(&[moving])));
        let before = s.location(moving);
        assert_eq!(s.shrink(64), 0, "the move was abandoned: nothing released");
        assert_eq!(ops.dtod.load(Ordering::SeqCst), 1, "the copy ran (lock free)");
        assert_eq!(s.location(moving), before, "block 1 stays in its slot");
        assert_eq!(s.stats().compactions, 0);
        *ops.on_dtod.lock() = None;
        s.unhold(&[moving]);
        assert_eq!(s.shrink(64), 64, "the next step compacts the two live blocks");
        let st = s.stats();
        assert_eq!((st.compactions, st.device_reserved_bytes), (1, 64));
        assert!(!s.acquire(0, key(1)).unwrap().fill, "block 1 is still present");
        assert!(!s.acquire(0, key(4)).unwrap().fill, "block 4 is still present");
    }

    #[test]
    fn shrink_compacts_a_sparse_chunk_into_holes_elsewhere() {
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
        // Chunk 0 = blocks 1, 2; chunk 1 = blocks 3, 4. Blocks 2 and 3 die: a hole in each
        // chunk, none releasable as is.
        let ids: Vec<_> = (1..=4).map(|i| s.acquire(0, key(i)).unwrap().id).collect();
        for &id in &ids {
            s.seal(id);
        }
        s.release(&ids[1..3]);
        assert_eq!(s.shrink(64), 64, "block 4 moved into chunk 0's hole, chunk 1 released");
        assert_eq!(ops.dtod.load(Ordering::SeqCst), 1);
        let st = s.stats();
        assert_eq!((st.compactions, st.compact_bytes, st.device_reserved_bytes), (1, 32, 64));
        assert_eq!(s.device_bytes(), 64, "blocks 1 and 4 remain, packed");
        assert!(matches!(s.location(ids[3]), Some(Location::Device(_))));
        assert!(!s.acquire(0, key(4)).unwrap().fill, "block 4 is still present");
        // A held block pins its chunk: nothing to release.
        s.hold(&[ids[0]]);
        s.release(&[ids[3]]);
        assert_eq!(s.shrink(64), 0);
        s.unhold(&[ids[0]]);
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
