//! §D Indirection table — per-executor logical-slot → physical-address map,
//! refreshed per iteration (the `SETUP_INDIRECTION` analogue).
//!
//! Packets reference logical slots (e.g. "load A from slot 3"); the interpreter
//! resolves slot → address at dispatch. Keeping the cached packet stream
//! slot-relative is what lets one compiled schedule serve every iteration —
//! only this table changes (per-request I/O pointers, KV page-table bases).

/// Representative slot layout (design §7.2). Small and fixed so it lives in
/// shared memory on-device.
pub mod slots {
    /// 0..7 — per-request input/output pointers.
    pub const REQUEST_IO: std::ops::Range<usize> = 0..8;
    /// First slot reserved for KV page-table addresses.
    pub const KV_PAGES_START: usize = 8;
    /// Per-request, per-layer KV page-table addresses.
    ///
    /// The range is sized for the model at mux construction; a fixed eight
    /// slots is insufficient even for a single sequence on a 32-layer model.
    pub fn kv_pages(n_layers: usize, max_batch: usize) -> std::ops::Range<usize> {
        KV_PAGES_START..KV_PAGES_START.saturating_add(n_layers.saturating_mul(max_batch))
    }
    /// Weight slots follow the model-sized KV region, avoiding overlap.
    pub fn weights(n_layers: usize, max_batch: usize) -> std::ops::Range<usize> {
        let start = kv_pages(n_layers, max_batch).end;
        start..start.saturating_add(16)
    }
    /// Scratch slots follow weights, so every region is disjoint.
    pub fn scratch(n_layers: usize, max_batch: usize) -> std::ops::Range<usize> {
        let start = weights(n_layers, max_batch).end;
        start..start.saturating_add(32)
    }
    /// Complete table size for this model, preserving the legacy 64-slot floor.
    pub fn table_size(n_layers: usize, max_batch: usize) -> usize {
        64usize.max(scratch(n_layers, max_batch).end)
    }
}

/// One executor's indirection table. Fixed capacity, no reallocation.
///
/// ## Device affinity (TP-3)
///
/// For multi-device (TP/PP), the device id is encoded in the high 8 bits of
/// each 64-bit slot entry: `bits[63:56] = device_id`, `bits[47:0] = address`.
/// The GPU address space is 48-bit (256 TiB), leaving bits 48–63 free for
/// metadata. The interpreter masks off the device bits when dereferencing:
/// `phys_addr = slot & 0x0000_FFFF_FFFF_FFFF`. The device bits tell the DMA
/// engine / transport layer which physical device owns the address.
pub struct IndirectionTable {
    slots: Box<[u64]>,
}

/// Mask to extract the 48-bit physical address from a tagged slot entry.
pub const ADDR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
/// Shift to extract the device id from the high 8 bits.
pub const DEVICE_SHIFT: u32 = 56;

/// Encode an address with a device qualifier. Single-device (device=0) is
/// backward-compatible: the tag is zero, leaving the address unchanged.
#[inline]
pub fn tag_addr(addr: u64, device: u8) -> u64 {
    (addr & ADDR_MASK) | ((device as u64) << DEVICE_SHIFT)
}

/// Extract the raw 48-bit physical address from a tagged entry.
#[inline]
pub fn untag_addr(tagged: u64) -> u64 {
    tagged & ADDR_MASK
}

/// Extract the device id from a tagged entry.
#[inline]
pub fn device_of(tagged: u64) -> u8 {
    (tagged >> DEVICE_SHIFT) as u8
}

impl IndirectionTable {
    pub fn new(n: usize) -> Self {
        IndirectionTable {
            slots: vec![0u64; n].into_boxed_slice(),
        }
    }

    /// Set a slot to a raw address (device 0 implied). Backward-compatible.
    #[inline]
    pub fn set(&mut self, slot: usize, addr: u64) {
        self.slots[slot] = addr;
    }

    /// Set a slot with explicit device affinity. The address is tagged with
    /// the device id in the high bits so the interpreter/transport can route.
    #[inline]
    pub fn set_with_device(&mut self, slot: usize, addr: u64, device: u8) {
        self.slots[slot] = tag_addr(addr, device);
    }

    /// Get the raw 64-bit entry (including device tag).
    #[inline]
    pub fn get(&self, slot: usize) -> u64 {
        self.slots[slot]
    }

    /// Get the physical address (device tag stripped).
    #[inline]
    pub fn get_addr(&self, slot: usize) -> u64 {
        untag_addr(self.slots[slot])
    }

    /// Get the device id for a slot entry.
    #[inline]
    pub fn get_device(&self, slot: usize) -> u8 {
        device_of(self.slots[slot])
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Raw view — what a `SETUP_INDIRECTION` packet would DMA to the device.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.slots)
    }
}
