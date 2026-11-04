//! A [`GpuEngine`](super::GpuEngine)'s place in a process rank group
//! ([`crate::exec::proc_group`]): its tensor-parallel coordinates, the device addresses of its
//! peers' collective regions in THIS process, and the per-launch-unit arm/audit protocol.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::device::cuda::ipc::IpcRegion;
use crate::device::cuda::{CudaBackend, CudaStream};
use crate::device::{Backend, DeviceMem};
use crate::exec::proc_group::ProcGroup;
use crate::exec::tp::XCTR_STRIDE;
use crate::{Result, RuntimeError};

/// How long a rank waits at a launch-unit barrier for its peers.
const ARM_TIMEOUT: Duration = Duration::from_secs(600);

/// Ranks per subgroup: a 128 B gate line holds one arrival word per source rank, and its last
/// word is the unstaged block count. Staged relay/count words live only in the local mirror.
const MAX_TP: u32 = 31;

pub(super) const STAGED_MARKER: &str = "plow_xr_staged_1";
pub(super) const PARTIAL_F32_MARKER: &str = "plow_xr_partial_f32_1";

fn partial_f32_bounds(d: &packet::dev::DevInst64, slot_bytes: u64, output_bytes: u64) -> bool {
    use packet::dev::{DevOp, TENSOR_NONE16, XREDUCE_PARTIAL_F32_MARK};
    d.op == DevOp::XReduce as u16
        && d.fj[0] == XREDUCE_PARTIAL_F32_MARK
        && d.fj[1] == 0
        && d.t[1..].iter().all(|&t| t == TENSOR_NONE16)
        && d.i[0] > 0
        && d.i[1] == 2
        && [0, slot_bytes].contains(&u64::from(d.i[2]))
        && slot_bytes > 0
        && slot_bytes % 128 == 0
        && slot_bytes <= u64::from(u32::MAX)
        && u64::from(d.i[0]) * 4 <= slot_bytes
        && u64::from(d.i[0]) * 2 <= output_bytes
        && d.i[4..].iter().all(|&v| v == 0)
}

pub(super) fn validate_partial_f32(blob: &crate::asset::devblob::DevBlob) -> Result<bool> {
    use packet::dev::XREDUCE_PARTIAL_F32_MARK;
    let mut marked = blob
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .filter(|d| {
            d.fj[0] == XREDUCE_PARTIAL_F32_MARK
                && matches!(
                    packet::dev::DevOp::from_u16(d.op),
                    Some(packet::dev::DevOp::XReduce | packet::dev::DevOp::XReduceTwoShot)
                )
        })
        .peekable();
    if marked.peek().is_none() {
        return Ok(false);
    }
    let invalid = || RuntimeError::Rejected("invalid NVIDIA FP32 peer partial layout".into());
    let tp = blob.tp.filter(|tp| tp.n_gpu == 2).ok_or_else(invalid)?;
    for name in ["act.og_tp", "act.dg_tp"] {
        if !blob
            .tensors
            .iter()
            .any(|t| t.name == name && t.bytes == tp.slot_bytes && t.init.is_none())
            || blob
                .tensors
                .iter()
                .any(|t| t.name.starts_with(&format!("{name}@")))
        {
            return Err(invalid());
        }
    }
    for d in marked {
        let output = blob.tensors.get(d.t[0] as usize).ok_or_else(invalid)?;
        if !partial_f32_bounds(d, tp.slot_bytes, output.bytes) {
            return Err(invalid());
        }
    }
    Ok(true)
}

pub(super) fn staged_supported(markers: &[Option<u32>]) -> bool {
    !markers.is_empty() && markers.iter().all(|m| *m == Some(1))
}

pub struct RankBind {
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    /// Tensor-parallel width and this rank's index inside its subgroup.
    pub tp: u32,
    pub tp_rank: u32,
    host_base: *mut u8,
    /// NVIDIA `PlowProgram::peer_scratch`: `[tp]` regions followed by this rank's
    /// host-route VRAM mirror, or NULL for P2P and unstaged host objects.
    peer_table: DeviceMem,
    local_mirror: Option<DeviceMem>,
    p2p_regions: Vec<IpcRegion>,
    unsafe_to_free: AtomicBool,
    /// Device address of this rank's own region.
    pub own_region: u64,
    /// Units armed so far: part of every plan fingerprint, so a rank that skipped or repeated
    /// a launch unit fails the next arm instead of pairing its collectives with the wrong unit.
    seq: u64,
}

// SAFETY: `host_base` is the group mapping, which `group` keeps alive; it is only read for
// unregistering.
unsafe impl Send for RankBind {}
unsafe impl Sync for RankBind {}

impl RankBind {
    /// Register the group mapping with this rank's device and upload the subgroup's region
    /// table. The group must carry a collective layout.
    pub fn new(be: Arc<CudaBackend>, group: Arc<ProcGroup>) -> Result<Self> {
        if group.layout().is_none() {
            return Err(RuntimeError::Rejected(
                "rank group carries no collective regions".into(),
            ));
        }
        let (tp, rank) = (group.tp(), group.rank());
        if tp > MAX_TP {
            return Err(RuntimeError::Rejected(format!(
                "tensor-parallel degree {tp} exceeds the {MAX_TP} arrival words of a gate line"
            )));
        }
        let (host_base, len) = group.mapping();
        // SAFETY: the mapping lives as long as `group`, which this value holds.
        let dev_base = unsafe { be.host_register_shared(host_base, len)? };
        let first = rank / tp * tp;
        let mut regions: Vec<u64> = (first..first + tp)
            .map(|r| dev_base + group.region_offset(r))
            .collect();
        regions.push(0);
        let bytes: Vec<u8> = regions.iter().flat_map(|a| a.to_le_bytes()).collect();
        let peer_table = be.alloc(0, bytes.len() as u64).and_then(|t| {
            be.upload(&t, 0, &bytes)?;
            Ok(t)
        });
        let peer_table = match peer_table {
            Ok(t) => t,
            Err(e) => {
                // SAFETY: registered above and never handed out.
                unsafe { be.host_unregister(host_base) };
                return Err(e);
            }
        };
        Ok(Self {
            own_region: regions[(rank - first) as usize],
            be,
            group,
            tp,
            tp_rank: rank - first,
            host_base,
            peer_table,
            local_mirror: None,
            p2p_regions: Vec::new(),
            unsafe_to_free: AtomicBool::new(false),
            seq: 0,
        })
    }

    pub(crate) fn new_p2p(
        be: Arc<CudaBackend>,
        group: Arc<ProcGroup>,
        mut regions: Vec<IpcRegion>,
    ) -> Result<Self> {
        let tp = group.tp();
        let tp_rank = group.rank() % tp;
        if group.layout().is_none() || tp > MAX_TP || regions.len() != tp as usize {
            return Err(RuntimeError::Rejected("invalid P2P rank geometry".into()));
        }
        let own_region = regions[tp_rank as usize].base;
        let mut addresses: Vec<_> = regions.iter().map(|r| r.base).collect();
        addresses.push(0);
        let bytes: Vec<_> = addresses.iter().flat_map(|a| a.to_le_bytes()).collect();
        let peer_table = be.alloc(0, bytes.len() as u64)?;
        be.upload(&peer_table, 0, &bytes)?;
        // Drop imported mappings before the local allocation.
        let own = regions.remove(tp_rank as usize);
        regions.push(own);
        Ok(Self {
            be,
            group,
            tp,
            tp_rank,
            host_base: std::ptr::null_mut(),
            peer_table,
            local_mirror: None,
            p2p_regions: regions,
            own_region,
            seq: 0,
            unsafe_to_free: AtomicBool::new(false),
        })
    }

    pub(crate) fn is_p2p(&self) -> bool {
        !self.p2p_regions.is_empty()
    }

    pub(super) fn failed_drain(&self) {
        self.unsafe_to_free.store(true, Ordering::Relaxed);
        self.group.abort();
    }

    pub(crate) fn transport_tier(&self) -> &'static str {
        if self.is_p2p() {
            "p2p"
        } else if self.local_mirror.is_some() {
            "staged"
        } else {
            "unstaged"
        }
    }

    pub(super) fn enable_staging(&mut self) -> Result<()> {
        if self.is_p2p() {
            return Ok(());
        }
        let mirror = self.be.alloc(0, self.layout().bytes())?;
        self.be.upload(
            &self.peer_table,
            u64::from(self.tp) * 8,
            &mirror.base.to_le_bytes(),
        )?;
        self.local_mirror = Some(mirror);
        Ok(())
    }

    pub(super) fn partial_region(&self) -> u64 {
        self.local_mirror
            .as_ref()
            .map_or(self.own_region, |m| m.base)
    }

    pub fn peer_table(&self) -> u64 {
        self.peer_table.base
    }

    pub fn group(&self) -> &Arc<ProcGroup> {
        &self.group
    }

    fn layout(&self) -> crate::exec::tp::PeerLayout {
        self.group.layout().expect("checked at bind")
    }

    /// `PlowProgram::xctr`.
    pub fn xctr(&self) -> u64 {
        self.own_region + self.layout().xctr_off()
    }

    /// The status line id the collectives write a deadline bail to (`fj[2] = id + 1`).
    pub fn status_id(&self) -> u32 {
        self.layout().n_xctr
    }

    /// Queue the reset after the P2P subgroup has retired the previous launch.
    pub(super) fn reset(&self, stream: &CudaStream) -> Result<()> {
        let layout = self.layout();
        let (region, bytes) = if self.is_p2p() {
            (self.own_region, layout.xstate_bytes())
        } else if let Some(mirror) = &self.local_mirror {
            (mirror.base, layout.xctr_bytes() + XCTR_STRIDE as u64)
        } else {
            return Ok(());
        };
        self.be
            .memset_d8_async(region + layout.xctr_off(), 0, bytes as usize, stream)
    }

    /// The engine has drained the queued reset before entering this group barrier.
    pub fn arm(&mut self, plan: &[u64]) -> Result<()> {
        self.seq += 1;
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ self.seq;
        for w in plan {
            h = (h ^ w).wrapping_mul(0x0100_0000_01b3);
        }
        self.group.arm(h, ARM_TIMEOUT)
    }

    /// After a launch unit retired: fail it if a collective gave up (deadline or a peer that
    /// never arrived).
    pub fn audit(&self) -> Result<()> {
        let status = if self.is_p2p() {
            let mut bytes = [0u8; 4];
            self.be
                .memcpy_dtoh(&mut bytes, self.own_region + self.layout().xstatus_off())?;
            u32::from_le_bytes(bytes)
        } else {
            self.group.own_xline(self.status_id())
        };
        if status != 0 {
            self.group.abort();
            return Err(RuntimeError::CollectiveBail(format!(
                "tensor-parallel collective bailed on rank {} (status {status:#x})",
                self.group.rank()
            )));
        }
        Ok(())
    }

    /// Byte offset of a gate line within the region (for tests and diagnostics).
    pub fn xline_offset(&self, id: u32) -> u64 {
        self.layout().xctr_off() + id as u64 * XCTR_STRIDE as u64
    }
}

impl Drop for RankBind {
    fn drop(&mut self) {
        if self.unsafe_to_free.load(Ordering::Relaxed) || self.be.ipc_must_retain() {
            let table = std::mem::replace(&mut self.peer_table, DeviceMem::view(0, 0));
            std::mem::forget(table);
            std::mem::forget(self.local_mirror.take());
            for region in &mut self.p2p_regions {
                region.leak_on_drop();
            }
            // Preserve a registered host mapping too; a kernel may still hold its address.
            std::mem::forget(self.group.clone());
            return;
        }
        // SAFETY: registered in `new`; the engine that launched against it is torn down first.
        if !self.host_base.is_null() {
            unsafe { self.be.host_unregister(self.host_base) };
        }
    }
}

/// The four `PlowProgram` fields a rank fills; [`TpKernarg::SINGLE`] off a group.
#[derive(Clone, Copy, Debug)]
pub struct TpKernarg {
    pub xctr: u64,
    pub peer_scratch: u64,
    pub rank: u32,
    pub n_gpu: u32,
}

impl TpKernarg {
    pub const SINGLE: Self = Self {
        xctr: 0,
        peer_scratch: 0,
        rank: 0,
        n_gpu: 1,
    };

    pub fn of(bind: Option<&RankBind>) -> Self {
        bind.filter(|b| b.tp > 1).map_or(Self::SINGLE, |b| Self {
            xctr: b.xctr(),
            peer_scratch: b.peer_table(),
            rank: b.tp_rank,
            n_gpu: b.tp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_f32_rejects_overlapping_slots_and_unsupported_operands() {
        use packet::dev::{DevInst64, DevOp, TENSOR_NONE16, XREDUCE_PARTIAL_F32_MARK};
        let d = DevInst64 {
            op: DevOp::XReduce as u16,
            blocks: 1,
            fj: [XREDUCE_PARTIAL_F32_MARK, 0, 0],
            t: [
                0,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
            ],
            i: [32, 2, 0, 1, 0, 0, 0, 0],
        };
        assert!(partial_f32_bounds(&d, 128, 64));
        let mut second = d;
        second.i[2] = 128;
        assert!(partial_f32_bounds(&second, 128, 64));
        for case in 0..8 {
            let mut bad = d;
            match case {
                0 => bad.i[0] = 33,
                1 => bad.i[1] = 4,
                2 => bad.i[2] = 64,
                3 => bad.i[4] = 2,
                4 => bad.fj[1] = 1,
                5 => bad.t[1] = 1,
                6 => bad.op = DevOp::XReduceTwoShot as u16,
                _ => bad.i[0] = 0,
            }
            assert!(!partial_f32_bounds(&bad, 128, 64));
        }
        assert!(!partial_f32_bounds(&d, 128, 63));
        assert!(!partial_f32_bounds(&d, 127, 64));
    }

    #[test]
    fn uninspectable_or_required_objects_without_marker_block_staging() {
        let junk = b"not an ELF image";
        for marker in [STAGED_MARKER, PARTIAL_F32_MARKER] {
            assert!(super::super::blocks_xr_capability(junk, 120, false, marker));
            assert!(super::super::blocks_xr_capability(junk, 120, true, marker));
        }
    }

    #[test]
    fn staged_requires_every_interpreter() {
        assert!(!staged_supported(&[]));
        assert!(staged_supported(&[Some(1)]));
        let all = [Some(1); 6]; // decode, wide decode, prefill, flash, GEMM, optional segment
        assert!(staged_supported(&all));
        for i in 0..all.len() {
            for stale in [None, Some(0), Some(2)] {
                let mut mixed = all;
                mixed[i] = stale;
                assert!(!staged_supported(&mixed));
            }
        }
    }
}
