//! §C KV cache — paged block pool + per-request page tables (vLLM PagedAttention
//! model). Host-side bookkeeping; the device only ever sees resolved block
//! addresses via the indirection table.

use plow_asset::KvPaging;

/// A physical KV block id (index into the growable region).
pub type BlockId = u32;

/// Fixed-size paged block allocator over the growable KV region.
pub struct BlockAllocator {
    block_bytes: u64,
    /// Base physical address of block 0 (KV region start).
    base: u64,
    /// Free list of block ids (LIFO — hot blocks stay warm in cache).
    free: Vec<BlockId>,
    total: u32,
}

impl BlockAllocator {
    /// Carve `[base, base + reserved)` into `reserved / block_bytes` blocks.
    pub fn new(base: u64, reserved: u64, paging: &KvPaging) -> Self {
        let block_bytes = paging.block_bytes.max(1);
        let total = (reserved / block_bytes) as u32;
        BlockAllocator {
            block_bytes,
            base,
            free: (0..total).rev().collect(),
            total,
        }
    }

    /// Pop a free block, or `None` under pressure (caller preempts/evicts/defers).
    #[inline]
    pub fn alloc(&mut self) -> Option<BlockId> {
        self.free.pop()
    }

    /// Return a block to the pool (sequence finished / preempted).
    #[inline]
    pub fn free(&mut self, id: BlockId) {
        self.free.push(id);
    }

    /// Physical address of a block, for writing into an indirection slot.
    #[inline]
    pub fn addr(&self, id: BlockId) -> u64 {
        self.base + id as u64 * self.block_bytes
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }
}

/// One sequence's page table: logical token window → physical blocks. Appended
/// to (via `UPDATE_INDIRECTION` OOB) as the sequence grows — never reallocated.
#[derive(Default)]
pub struct PageTable {
    blocks: Vec<BlockId>,
}

impl PageTable {
    /// Blocks needed to hold `tokens` given `block_tokens` per block.
    pub fn blocks_for(tokens: i64, block_tokens: i64) -> usize {
        if block_tokens <= 0 {
            return 0;
        }
        ((tokens + block_tokens - 1) / block_tokens) as usize
    }

    /// Ensure capacity for `tokens`, pulling blocks from `alloc`. Returns the
    /// number of blocks newly appended, or `Err` on pressure.
    pub fn ensure(
        &mut self,
        tokens: i64,
        block_tokens: i64,
        alloc: &mut BlockAllocator,
    ) -> Result<usize, ()> {
        let need = Self::blocks_for(tokens, block_tokens);
        let have = self.blocks.len();
        if need <= have {
            return Ok(0);
        }
        let mut added = 0;
        for _ in have..need {
            match alloc.alloc() {
                Some(b) => {
                    self.blocks.push(b);
                    added += 1;
                }
                None => {
                    // roll back this growth so the pool stays consistent
                    for b in self.blocks.drain(have..) {
                        alloc.free(b);
                    }
                    return Err(());
                }
            }
        }
        Ok(added)
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Release every block back to the pool.
    pub fn release(&mut self, alloc: &mut BlockAllocator) {
        for b in self.blocks.drain(..) {
            alloc.free(b);
        }
    }
}
