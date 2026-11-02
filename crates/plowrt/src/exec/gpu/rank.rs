//! A [`GpuEngine`](super::GpuEngine)'s place in a process rank group
//! ([`crate::exec::proc_group`]): its tensor-parallel coordinates, the device addresses of its
//! peers' collective regions in THIS process, and the per-launch-unit arm/audit protocol.

use std::sync::Arc;
use std::time::Duration;

use crate::device::cuda::CudaBackend;
use crate::device::{Backend, DeviceMem};
use crate::exec::proc_group::ProcGroup;
use crate::exec::tp::XCTR_STRIDE;
use crate::{Result, RuntimeError};

/// How long a rank waits at a launch-unit barrier for its peers.
const ARM_TIMEOUT: Duration = Duration::from_secs(600);

/// Ranks per subgroup: a 128 B gate line holds one arrival word per source rank, and its last
/// word is the two-shot collective's block count (`kXrAggWord` in the NVIDIA collectives).
const MAX_TP: u32 = 31;

pub struct RankBind {
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    /// Tensor-parallel width and this rank's index inside its subgroup.
    pub tp: u32,
    pub tp_rank: u32,
    host_base: *mut u8,
    /// `[tp]` device table of the subgroup's regions — `PlowProgram::peer_scratch`.
    peer_table: DeviceMem,
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
        let regions: Vec<u64> = (first..first + tp)
            .map(|r| dev_base + group.region_offset(r))
            .collect();
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
            seq: 0,
        })
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

    /// Start a launch unit described by `plan` (every rank must pass the same words). The
    /// device must be idle: callers synchronize the engine stream first.
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
        let status = self.group.own_xline(self.status_id());
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
        // SAFETY: registered in `new`; the engine that launched against it is torn down first.
        unsafe { self.be.host_unregister(self.host_base) };
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
