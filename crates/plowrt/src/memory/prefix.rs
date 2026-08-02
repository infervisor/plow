//! §L Prefix cache (RadixAttention / automatic prefix caching) over the
//! **head-major** per-head KV pool.
//!
//! ## Why this is not a block list
//!
//! plow stores KV head-major: the bytes for `(kv, head, seq)` are one contiguous
//! head-slot of `max_seq_len × head_dim × elem` bytes ([`GrowablePool`]), and a
//! token's row sits at `tok × head_dim × elem` *inside* that slot. A shared
//! prefix is therefore **not one block** — it is a **strided set of runs, one per
//! `(kv, head)`**, all at the same offset within their respective head-slots.
//! For Qwen3-4B (36 layers × kv_factor 2 × 8 kv-heads) a shared prefix resolves
//! to 576 runs. That is the page-table-over-pool view the old docstring called
//! for, and it is what [`PrefixCache::lookup`] emits.
//!
//! ## Block granularity
//!
//! The sharing unit is `block_tokens × head_dim × elem` — a whole number of
//! token rows *within one head-slot*. This is the granularity the assignment
//! requires and it is forced by the layout: token rows are the inner-most axis
//! of a head-slot, so any multiple of one row is contiguous, and no block can
//! straddle a `(kv, head)` boundary as long as `block_tokens ≤ max_seq_len`
//! (checked in [`PrefixCache::new`]). Choosing the unit *smaller* than a head
//! would not shrink the run list — the run count is `kv_factor × kv_heads`
//! regardless — so `block_tokens` trades match granularity against tree depth
//! only, and `KvPaging::block_tokens` is used as-is.
//!
//! ## Coalescing is what keeps the run list small
//!
//! A naive resolution emits `kv_factor × kv_heads` runs *per block*. But
//! consecutive blocks owned by the same sequence at consecutive block indices are
//! **physically adjacent inside the head-slot**, so they merge. A 2048-token
//! prefix that was produced by a single sequence therefore resolves to exactly
//! `kv_factor × kv_heads` runs, not `blocks × kv_factor × kv_heads`. See
//! [`PrefixCache::lookup`] and the `coalesces_same_owner` test.
//!
//! ## ABI limitation (measured, not assumed)
//!
//! Resolving the runs is only half the problem: the device must be able to *read*
//! them. It currently cannot read two of them. `FlashDecode` addresses KV as
//! `kbase = K + ((b·n_kv_head + hkv)·kv_stride)·D` then `krow = kbase + (kv &
//! kv_mask)·D` (`runtime/nvidia/op_attention.cuh:174`), and `kv_mask` is a
//! power-of-two **ring modulo** (`KV_MASK_NONE = 0xFFFFFFFF` on a full layer,
//! `kvr-1` on a sliding one — `gemma4.rs::kv_ring`). That is a sliding-window
//! wrap, **not** a page table: exactly ONE contiguous run per `(b, kv_head)` is
//! addressable, so "shared prefix run + private suffix run" is **not expressible**.
//! `op_mla.cuh` already carries the fix on its own path — a `GATHER` arm that
//! reads `row = ibase[kv]` from an index array — and that arm is what
//! `FlashDecode` needs for zero-copy sharing.
//!
//! Until then the runs this module emits are consumed by **copying**: the runs
//! are blitted D2D into the new sequence's head-slots, which skips the prefill
//! but not the KV storage. `runtime/tests/qwen3_prefix.cu` measures exactly that.

use rustc_hash::FxHashMap;

use crate::memory::pool::GrowablePool;

/// Hash of a token block (the compiler/tokenizer-agnostic prefix key).
pub type BlockHash = u64;

/// Index into [`PrefixCache::nodes`].
type NodeId = u32;

/// One resolved physical KV run: `bytes` contiguous bytes at `addr`, holding the
/// rows for one `(kv, head)` over some token range. Emitted in a fixed order —
/// `kv` outer, `head` inner — so a consumer can zip it against its own tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub kv: u32,
    pub head: u32,
    pub addr: u64,
    pub bytes: u64,
}

/// A radix-tree node owning one shared, refcounted block of tokens.
struct Node {
    parent: Option<NodeId>,
    children: FxHashMap<BlockHash, NodeId>,
    /// The block's exact token ids. Matching verifies these — the hash only
    /// routes the walk. A 64-bit non-crypto hash collision must cost a miss,
    /// never serve another request's KV.
    tokens: Box<[u32]>,
    /// Sequence slot whose head-slots physically hold this block's KV.
    owner_seq: u32,
    /// Logical block index within `owner_seq` — the block's offset in the slot.
    block_idx: u32,
    refs: u32,
    /// Monotonic tick of last use, for LRU eviction among zero-ref nodes.
    last_used: u64,
    /// Unlinked by `evict_lru`. Node ids are indices held by sibling `children`
    /// maps, so an evicted node cannot be removed from `nodes` — it is tombstoned
    /// instead. Without this flag a tombstone still looks like a zero-ref leaf and
    /// `evict_lru` hands the SAME (owner_seq, block_idx) out again, telling the
    /// caller to free storage it already freed. Caught by
    /// `refcount_blocks_eviction_and_release_unblocks_it`.
    evicted: bool,
}

/// The result of a prefix lookup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Match {
    /// How many leading blocks matched. `tokens = blocks × block_tokens`.
    pub blocks: usize,
    /// Physical runs backing those blocks, coalesced.
    pub runs: Vec<Run>,
    /// `(owner_seq, block_idx)` of every matched node, in prefix order — the
    /// stable identity a payload side-table can key off (the VMM prefix pool
    /// resolves granule handles through this instead of `runs`).
    pub placed: Vec<(u32, u32)>,
}

impl Match {
    /// Total bytes reused — i.e. KV that does **not** have to be recomputed.
    pub fn bytes(&self) -> u64 {
        self.runs.iter().map(|r| r.bytes).sum()
    }
}

/// Refcounted radix prefix cache resolving to head-major KV runs.
pub struct PrefixCache {
    pool: GrowablePool,
    block_tokens: u64,
    /// `head_dim × elem_bytes` — the stride of one token row inside a head-slot.
    token_bytes: u64,
    nodes: Vec<Node>,
    roots: FxHashMap<BlockHash, NodeId>,
    tick: u64,
    /// Hash collisions caught by token verification (each one was a would-be
    /// wrong-KV serve).
    collisions: u64,
}

impl PrefixCache {
    /// `token_bytes = head_dim × elem_bytes`. Returns `None` if a block would not
    /// fit inside a head-slot, since a block that straddles a `(kv, head)`
    /// boundary would make the run list explode — the one geometry this design
    /// cannot represent, so it is rejected rather than silently mis-resolved.
    pub fn new(pool: GrowablePool, block_tokens: u64, token_bytes: u64) -> Option<Self> {
        if block_tokens == 0 || token_bytes == 0 {
            return None;
        }
        if block_tokens.checked_mul(token_bytes)? > pool.head_slot_bytes {
            return None;
        }
        Some(PrefixCache {
            pool,
            block_tokens,
            token_bytes,
            nodes: Vec::new(),
            roots: FxHashMap::default(),
            tick: 0,
            collisions: 0,
        })
    }

    /// Hash collisions caught by token verification since construction.
    pub fn collisions(&self) -> u64 {
        self.collisions
    }

    /// Token ids of block `i` within `tokens`, or `None` when the caller's
    /// token stream is shorter than its hash list claims (treated as mismatch).
    fn block_slice<'a>(&self, tokens: &'a [u32], i: usize) -> Option<&'a [u32]> {
        let bt = self.block_tokens as usize;
        tokens.get(i * bt..(i + 1) * bt)
    }

    /// Tokens per shared block — the granularity a caller must hash at.
    pub fn block_tokens(&self) -> u64 {
        self.block_tokens
    }

    /// Bytes one block occupies in one `(kv, head)` run.
    pub fn block_run_bytes(&self) -> u64 {
        self.block_tokens * self.token_bytes
    }

    /// Match the longest cached prefix of `hashes`, bump refcounts + recency on
    /// every node along the matched path, and resolve it to physical runs.
    ///
    /// `tokens` is the prompt the hashes were computed from (block `i` =
    /// `tokens[i·block_tokens .. (i+1)·block_tokens]`). Every hash match is
    /// verified against the node's stored tokens — a collision terminates the
    /// match (a miss), it never serves the colliding node's KV.
    ///
    /// Runs are **coalesced**: consecutive matched blocks with the same
    /// `owner_seq` and consecutive `block_idx` are physically adjacent inside
    /// each head-slot and merge into one run per `(kv, head)`.
    pub fn lookup(&mut self, hashes: &[BlockHash], tokens: &[u32]) -> Match {
        self.tick += 1;
        let tick = self.tick;

        // Walk the path first; refcounts are bumped only for what actually matched.
        let mut path: Vec<NodeId> = Vec::new();
        let mut cur: Option<NodeId> = None;
        for (i, h) in hashes.iter().enumerate() {
            let next = match cur {
                None => self.roots.get(h).copied(),
                Some(p) => self.nodes[p as usize].children.get(h).copied(),
            };
            match next {
                Some(n) => {
                    if self.block_slice(tokens, i) != Some(&self.nodes[n as usize].tokens) {
                        self.collisions += 1;
                        break;
                    }
                    path.push(n);
                    cur = Some(n);
                }
                None => break,
            }
        }
        for &n in &path {
            let e = &mut self.nodes[n as usize];
            e.refs += 1;
            e.last_used = tick;
        }

        // (owner_seq, block_idx) per matched block, in prefix order.
        let placed: Vec<(u32, u32)> = path
            .iter()
            .map(|&n| {
                let e = &self.nodes[n as usize];
                (e.owner_seq, e.block_idx)
            })
            .collect();

        Match {
            blocks: path.len(),
            runs: self.runs_for(&placed),
            placed,
        }
    }

    /// Resolve placed blocks to coalesced per-`(kv, head)` runs. Split out so the
    /// coalescing is testable without touching the tree.
    fn runs_for(&self, placed: &[(u32, u32)]) -> Vec<Run> {
        // Group the block list into maximal (owner_seq, contiguous block_idx) spans
        // ONCE, then expand each span across every (kv, head). Doing it in this
        // order is what makes the run count kv_factor×kv_heads×spans rather than
        // ×blocks.
        let mut spans: Vec<(u32, u32, u32)> = Vec::new(); // (owner, first_blk, n_blk)
        for &(owner, blk) in placed {
            match spans.last_mut() {
                Some((o, first, n)) if *o == owner && *first + *n == blk => *n += 1,
                _ => spans.push((owner, blk, 1)),
            }
        }

        let mut runs = Vec::with_capacity(
            spans.len() * (self.pool.kv_factor as usize) * (self.pool.kv_heads as usize),
        );
        for kv in 0..self.pool.kv_factor {
            for head in 0..self.pool.kv_heads {
                for &(owner, first, n) in &spans {
                    // Not in_range ⇒ the slot was never valid; skip rather than
                    // emit an address outside the pool.
                    if !self.pool.in_range(kv, head, owner) {
                        continue;
                    }
                    let slot = self.pool.head_slot_offset(kv, head, owner);
                    runs.push(Run {
                        kv,
                        head,
                        addr: slot + first as u64 * self.block_tokens * self.token_bytes,
                        bytes: n as u64 * self.block_tokens * self.token_bytes,
                    });
                }
            }
        }
        runs
    }

    /// Publish blocks computed by `owner_seq` under `hashes`, starting at
    /// `from_block` (the first block the lookup did **not** match). `tokens`
    /// is the prompt the hashes were computed from; each published node stores
    /// its block's token ids for match-time verification. Each new node
    /// is inserted with refcount 1, held by the inserting sequence.
    ///
    /// Blocks already present are left alone — that is the copy-on-write point:
    /// a divergent sequence simply gets a new sibling node pointing at its own
    /// head-slot, and the shared ancestor is untouched. A hash-equal node with
    /// DIFFERENT tokens (collision) stops the publish: `children` holds one
    /// node per hash, and replacing the resident would orphan its holders.
    ///
    /// Returns the verified path length now present — how many leading blocks
    /// of `hashes` a caller may key payloads off. Early stop (collision,
    /// stale `from_block`, short token stream) returns less than
    /// `hashes.len()`; the caller must not reference blocks past it.
    pub fn insert(
        &mut self,
        hashes: &[BlockHash],
        tokens: &[u32],
        owner_seq: u32,
        from_block: usize,
    ) -> usize {
        self.tick += 1;
        let tick = self.tick;
        // Re-walk the shared prefix to find the attachment point.
        let mut cur: Option<NodeId> = None;
        for (i, h) in hashes.iter().take(from_block).enumerate() {
            let next = match cur {
                None => self.roots.get(h).copied(),
                Some(p) => self.nodes[p as usize].children.get(h).copied(),
            };
            match next {
                Some(n) if self.block_slice(tokens, i) == Some(&self.nodes[n as usize].tokens) => {
                    cur = Some(n)
                }
                // The caller's `from_block` disagrees with the tree (or a
                // collision sits on the path); attaching anyway would key this
                // block under the wrong prefix and serve wrong KV, so stop.
                _ => return i,
            }
        }
        for (i, &h) in hashes.iter().enumerate().skip(from_block) {
            let existing = match cur {
                None => self.roots.get(&h).copied(),
                Some(p) => self.nodes[p as usize].children.get(&h).copied(),
            };
            let Some(blk) = self.block_slice(tokens, i) else {
                return i;
            };
            if let Some(n) = existing {
                if blk != &*self.nodes[n as usize].tokens {
                    self.collisions += 1;
                    return i;
                }
                cur = Some(n);
                continue;
            }
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Node {
                parent: cur,
                children: FxHashMap::default(),
                tokens: blk.into(),
                owner_seq,
                block_idx: i as u32,
                refs: 1,
                last_used: tick,
                evicted: false,
            });
            match cur {
                None => {
                    self.roots.insert(h, id);
                }
                Some(p) => {
                    self.nodes[p as usize].children.insert(h, id);
                }
            }
            cur = Some(id);
        }
        hashes.len()
    }

    /// Drop one reference along the matched path of `hashes` (its first `blocks`
    /// entries). Nodes at zero refs become eligible for eviction.
    pub fn release(&mut self, hashes: &[BlockHash], blocks: usize) {
        let mut cur: Option<NodeId> = None;
        for h in hashes.iter().take(blocks) {
            let next = match cur {
                None => self.roots.get(h).copied(),
                Some(p) => self.nodes[p as usize].children.get(h).copied(),
            };
            match next {
                Some(n) => {
                    let e = &mut self.nodes[n as usize];
                    e.refs = e.refs.saturating_sub(1);
                    cur = Some(n);
                }
                None => return,
            }
        }
    }

    /// Evict the least-recently-used zero-ref **leaf**, returning the
    /// `(owner_seq, block_idx)` whose storage the caller may now reclaim.
    ///
    /// Only leaves are evictable: dropping an interior node would orphan its
    /// children, and a child resolved through a freed parent is precisely the
    /// "returns wrong KV, produces fluent wrong text" failure this cache must
    /// never have.
    pub fn evict_lru(&mut self) -> Option<(u32, u32)> {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.evicted && n.refs == 0 && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i as NodeId)?;

        // Unlink from the parent (or the root map) so no path can reach it again.
        let (parent, owner, blk) = {
            let n = &self.nodes[victim as usize];
            (n.parent, n.owner_seq, n.block_idx)
        };
        match parent {
            Some(p) => self.nodes[p as usize].children.retain(|_, v| *v != victim),
            None => self.roots.retain(|_, v| *v != victim),
        }
        // The node itself is left in `nodes` as a tombstone: node ids are indices
        // held by `children` maps elsewhere, so compacting the vector would
        // invalidate them. It is unreachable and holds no storage claim.
        self.nodes[victim as usize].refs = 0;
        self.nodes[victim as usize].children.clear();
        self.nodes[victim as usize].evicted = true;
        Some((owner, blk))
    }

    /// Number of nodes ever allocated (including evicted tombstones).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-4B's per-layer geometry: kv_factor 2 (separate K/V), 8 kv-heads,
    /// head_dim 128 bf16 ⇒ 256 B/token row, ctx 4256 ⇒ 1_089_536 B head-slot.
    fn qwen3_pool() -> GrowablePool {
        GrowablePool {
            base: 0,
            kv_factor: 2,
            kv_heads: 8,
            max_seqs: 4,
            head_slot_bytes: 4256 * 128 * 2,
        }
    }

    fn cache() -> PrefixCache {
        PrefixCache::new(qwen3_pool(), 256, 128 * 2).unwrap()
    }

    /// A token stream whose block `i` content is derived from `hashes[i]` —
    /// distinct hashes ⇒ distinct block tokens, mirroring real chained hashing.
    fn toks(hashes: &[BlockHash]) -> Vec<u32> {
        hashes
            .iter()
            .flat_map(|&h| std::iter::repeat(h as u32).take(256))
            .collect()
    }

    #[test]
    fn rejects_block_larger_than_a_head_slot() {
        // A block that does not fit inside a head-slot would straddle a
        // (kv, head) boundary. That geometry must be refused, not resolved.
        assert!(PrefixCache::new(qwen3_pool(), 8192, 256).is_none());
        assert!(PrefixCache::new(qwen3_pool(), 4256, 256).is_some());
    }

    #[test]
    fn miss_on_empty_cache() {
        let mut c = cache();
        let m = c.lookup(&[1, 2, 3], &toks(&[1, 2, 3]));
        assert_eq!(m.blocks, 0);
        assert!(m.runs.is_empty());
        assert_eq!(m.bytes(), 0);
    }

    #[test]
    fn coalesces_same_owner() {
        // 8 consecutive blocks from ONE sequence must resolve to exactly
        // kv_factor × kv_heads = 16 runs, not 8 × 16 = 128.
        let mut c = cache();
        let h: Vec<BlockHash> = (0..8).collect();
        c.insert(&h, &toks(&h), 0, 0);
        let m = c.lookup(&h, &toks(&h));
        assert_eq!(m.blocks, 8);
        assert_eq!(m.runs.len(), 16, "coalescing failed: {:?}", m.runs.len());
        // Each run spans all 8 blocks: 8 × 256 tokens × 256 B.
        for r in &m.runs {
            assert_eq!(r.bytes, 8 * 256 * 256);
        }
        assert_eq!(m.bytes(), 16 * 8 * 256 * 256);
    }

    #[test]
    fn runs_land_inside_their_head_slots() {
        let mut c = cache();
        let pool = qwen3_pool();
        let h: Vec<BlockHash> = (0..4).collect();
        c.insert(&h, &toks(&h), 2, 0);
        let m = c.lookup(&h, &toks(&h));
        for r in &m.runs {
            let slot = pool.head_slot_offset(r.kv, r.head, 2);
            assert!(r.addr >= slot, "run before its slot");
            assert!(
                r.addr + r.bytes <= slot + pool.head_slot_bytes,
                "run {:?} overruns head-slot [{}, {})",
                r,
                slot,
                slot + pool.head_slot_bytes
            );
        }
        // Distinct (kv, head) must be byte-disjoint — the R-K2 property.
        let mut iv: Vec<(u64, u64)> = m.runs.iter().map(|r| (r.addr, r.addr + r.bytes)).collect();
        iv.sort();
        for w in iv.windows(2) {
            assert!(w[0].1 <= w[1].0, "runs overlap: {:?} {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn partial_match_and_cow_divergence() {
        let mut c = cache();
        // Sequence A: blocks [10, 11, 12, 13].
        let ha = [10, 11, 12, 13];
        c.insert(&ha, &toks(&ha), 0, 0);
        // Sequence B shares [10, 11] then diverges.
        let hb = [10, 11, 99, 98];
        let m = c.lookup(&hb, &toks(&hb));
        assert_eq!(m.blocks, 2, "should match only the shared 2 blocks");
        // The divergent tail is published against B's OWN slot (copy-on-write);
        // A's nodes are untouched.
        c.insert(&hb, &toks(&hb), 1, m.blocks);
        let again = c.lookup(&hb, &toks(&hb));
        assert_eq!(again.blocks, 4);
        // Two spans now: blocks 0-1 owned by seq 0, blocks 2-3 owned by seq 1.
        assert_eq!(again.runs.len(), 32, "expected 2 spans × 16");
        // And A still resolves entirely to its own slot.
        let ma = c.lookup(&ha, &toks(&ha));
        assert_eq!(ma.blocks, 4);
        assert_eq!(ma.runs.len(), 16, "A must still be one coalesced span");
    }

    #[test]
    fn refcount_blocks_eviction_and_release_unblocks_it() {
        let mut c = cache();
        c.insert(&[1, 2], &toks(&[1, 2]), 0, 0); // refs 1 on both
        assert_eq!(c.evict_lru(), None, "live nodes must not be evictable");
        c.release(&[1, 2], 2);
        // Only the LEAF is evictable; the interior node would orphan it.
        let v = c
            .evict_lru()
            .expect("leaf should be evictable at zero refs");
        assert_eq!(v, (0, 1), "the leaf is block_idx 1 of seq 0");
        // Now the former interior node is a leaf and can go too.
        assert_eq!(c.evict_lru(), Some((0, 0)));
        // ...and NOTHING may be handed out twice. This assertion failed before
        // `Node::evicted` existed: the tombstone still looked like a zero-ref leaf,
        // so evict_lru returned (0, 0) a second time and the caller would have
        // double-freed that head-slot range.
        assert_eq!(
            c.evict_lru(),
            None,
            "an evicted node must not be evicted again"
        );
    }

    #[test]
    fn evicted_node_is_unreachable() {
        let mut c = cache();
        c.insert(&[7, 8], &toks(&[7, 8]), 0, 0);
        c.release(&[7, 8], 2);
        c.evict_lru(); // drops the leaf (block 1)
        let m = c.lookup(&[7, 8], &toks(&[7, 8]));
        assert_eq!(m.blocks, 1, "evicted leaf must no longer match");
    }

    /// NEGATIVE CONTROL for the resolver. These assertions are the ones the whole
    /// module rests on, so each is shown to FAIL under a deliberate corruption of
    /// the kind a page-table-over-pool actually introduces. A prefix cache that
    /// returns wrong KV produces fluent wrong text, so "the test passes" is only
    /// evidence if the test can distinguish right from wrong runs.
    #[test]
    fn negative_control_corrupted_placement_changes_the_runs() {
        let c = cache();
        let good: Vec<(u32, u32)> = (0..4).map(|b| (0u32, b)).collect();

        // (a) OFF-BY-ONE BLOCK OFFSET — one block placed at the wrong index.
        let mut bad_off = good.clone();
        bad_off[2].1 += 1;
        let rg = c.runs_for(&good);
        let rb = c.runs_for(&bad_off);
        assert_ne!(rg, rb, "a shifted block_idx must change the resolved runs");
        // It also breaks coalescing, which is the observable symptom.
        assert_eq!(rg.len(), 16);
        assert!(
            rb.len() > 16,
            "a gap must split the span, got {} runs",
            rb.len()
        );

        // (b) WRONG OWNER — one block attributed to another sequence's slot.
        // This is the failure that reads another request's KV.
        let mut bad_owner = good.clone();
        bad_owner[1].0 = 1;
        let ro = c.runs_for(&bad_owner);
        assert_ne!(
            rg, ro,
            "a wrong owner_seq must change the resolved addresses"
        );
        let pool = qwen3_pool();
        assert!(
            ro.iter()
                .any(|r| r.addr >= pool.head_slot_offset(r.kv, r.head, 1)
                    && r.addr < pool.head_slot_offset(r.kv, r.head, 1) + pool.head_slot_bytes),
            "the corrupted run should point into seq 1's slot"
        );

        // (c) The control must not fire on an IDENTICAL placement — otherwise
        // assert_ne! above would pass for the trivial reason that runs_for is
        // nondeterministic.
        assert_eq!(rg, c.runs_for(&good), "runs_for must be deterministic");
    }

    /// A forged 64-bit collision (same hash, different tokens) must MISS —
    /// serving the resident node would hand another request's KV across the
    /// tenant boundary. Hashes are caller-supplied here, so the forgery is
    /// exact: identical hash lists over different token streams.
    #[test]
    fn hash_collision_is_a_miss_not_a_wrong_kv_serve() {
        let mut c = cache();
        let h = [42u64, 43];
        let ta = toks(&h); // resident tokens
        let mut tb = ta.clone(); // colliding tokens: differ in block 0
        tb[0] ^= 1;

        c.insert(&h, &ta, 0, 0);

        // Lookup under the colliding stream: zero blocks, collision counted.
        let m = c.lookup(&h, &tb);
        assert_eq!(m.blocks, 0, "collision must not match");
        assert_eq!(c.collisions(), 1);

        // Insert under the colliding stream must refuse to adopt the resident
        // node (one child per hash — replacing it would orphan its holders).
        c.insert(&h, &tb, 1, 0);
        assert_eq!(c.collisions(), 2);
        let mb = c.lookup(&h, &tb);
        assert_eq!(mb.blocks, 0, "colliding publish must not be adopted");

        // The rightful owner still hits in full.
        let ma = c.lookup(&h, &ta);
        assert_eq!(ma.blocks, 2);
        assert_eq!(ma.placed, vec![(0, 0), (0, 1)]);
    }

    /// Divergence INSIDE a block (same hash prefix, tokens differ mid-block)
    /// is exactly the shape a truncated/aliased tokenizer bug produces —
    /// verification must catch it even when the caller's stream is short.
    #[test]
    fn short_token_stream_never_matches() {
        let mut c = cache();
        let h = [7u64];
        let t = toks(&h);
        c.insert(&h, &t, 0, 0);
        // Stream shorter than one block: block_slice = None ⇒ miss, not panic.
        let m = c.lookup(&h, &t[..100]);
        assert_eq!(m.blocks, 0);
    }
}
