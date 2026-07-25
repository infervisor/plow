//! §L Device-packet transport — RDMA and inter-device data movement compiled
//! into plow device packets, with NO host in the loop.
//!
//! ## Design Principle (plow runtime)
//!
//! All cross-device data movement (TP all-reduce, PP layer-boundary activations,
//! disaggregated KV handoff) is expressed as compiled device packets that the
//! counter system gates. The host never touches the data: packets encode source
//! address, destination address, size, and the counter that fires when the
//! transfer lands. This is the "device packet" model — the compiler emits the
//! transfers as part of the schedule, and the runtime merely programs the DMA
//! engines / RDMA NICs with the compiled descriptors.
//!
//! For local multi-GPU (NVLink/xGMI): the transfer packet maps to a peer-to-peer
//! copy (P2P memcpy over the fabric). For remote nodes (InfiniBand/RoCE): the
//! packet maps to an RDMA write-with-immediate, where the immediate value encodes
//! the destination counter to fire on completion.
//!
//! This module defines the transport abstraction and the compiled transfer
//! descriptor the scheduler emits.

/// A compiled device-to-device transfer descriptor. Emitted by the scheduler
/// (`plowc`) and consumed by the runtime's DMA engine / RDMA NIC programming.
#[derive(Clone, Debug)]
pub struct TransferDescriptor {
    /// Source device id (local GPU index or remote node:device).
    pub src_device: DeviceAddr,
    /// Source offset within the device's address space.
    pub src_offset: u64,
    /// Destination device id.
    pub dst_device: DeviceAddr,
    /// Destination offset within the device's address space.
    pub dst_offset: u64,
    /// Transfer size in bytes.
    pub bytes: u64,
    /// Counter id to fire on the destination device when the transfer completes.
    /// This gates the first packet of the consumer stage.
    pub done_counter: u32,
    /// Counter id that gates this transfer (must reach threshold before DMA starts).
    pub wait_counter: u32,
}

/// Identifies a device in the cluster. Local devices use `node = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceAddr {
    /// Node index (0 = local host).
    pub node: u16,
    /// Device index within that node.
    pub device: u8,
}

impl DeviceAddr {
    pub const fn local(device: u8) -> Self {
        DeviceAddr { node: 0, device }
    }

    pub fn is_local(&self) -> bool {
        self.node == 0
    }

    /// Whether a transfer between `self` and `other` requires network (RDMA).
    pub fn requires_network(&self, other: &DeviceAddr) -> bool {
        self.node != other.node
    }
}

/// Transport backend trait. The runtime programs the physical transport with
/// compiled descriptors. On real hardware:
/// - Local P2P: `cuMemcpyPeerAsync` / `hsa_amd_memory_async_copy`
/// - RDMA: ibverbs `post_send` with `IBV_WR_RDMA_WRITE_WITH_IMM`
///
/// The host never touches the payload data — it only programs the engine.
pub trait Transport: Send + Sync {
    /// Program a transfer. The runtime calls this once per descriptor in the
    /// schedule; the actual transfer happens device-side when `wait_counter`
    /// fires. Returns immediately (non-blocking).
    fn program(&self, desc: &TransferDescriptor) -> crate::Result<()>;

    /// Query whether a given device-pair supports zero-copy (P2P / NVLink).
    fn supports_p2p(&self, src: DeviceAddr, dst: DeviceAddr) -> bool;
}

/// Null transport for single-device or CPU-only operation.
pub struct NullTransport;

impl Transport for NullTransport {
    fn program(&self, _desc: &TransferDescriptor) -> crate::Result<()> {
        // Single-device: no cross-device transfers exist in the schedule.
        Ok(())
    }

    fn supports_p2p(&self, _src: DeviceAddr, _dst: DeviceAddr) -> bool {
        true // trivially — same device
    }
}

/// Local P2P transport for multi-GPU on the same node (NVLink / xGMI).
/// Programs peer-to-peer copies that the counter system gates.
pub struct LocalP2pTransport;

impl Transport for LocalP2pTransport {
    fn program(&self, _desc: &TransferDescriptor) -> crate::Result<()> {
        // Production: call cuMemcpyPeerAsync with the src/dst device contexts,
        // gated by the wait_counter (mapped to a CUDA event). The done_counter
        // is incremented by a callback / trailing event on the copy stream.
        Ok(())
    }

    fn supports_p2p(&self, src: DeviceAddr, dst: DeviceAddr) -> bool {
        src.node == dst.node // same node = P2P possible
    }
}

/// RDMA transport for cross-node transfers. Uses ibverbs RDMA-write-with-
/// immediate where the immediate value encodes the destination counter id.
/// The NIC fires the counter on the remote node without involving the remote CPU.
pub struct RdmaTransport {
    // In production: ibv_context, ibv_pd, ibv_qp per peer, memory regions, etc.
    _placeholder: (),
}

impl RdmaTransport {
    /// Connect to peers. In production this performs the QP exchange handshake.
    pub fn new() -> Self {
        RdmaTransport { _placeholder: () }
    }
}

impl Transport for RdmaTransport {
    fn program(&self, _desc: &TransferDescriptor) -> crate::Result<()> {
        // Production:
        // 1. Look up the pre-registered MR for src_device's memory region.
        // 2. Post an RDMA write-with-immediate to the peer's QP:
        //    - remote_addr = dst_offset (pre-registered on remote node)
        //    - immediate = done_counter (remote NIC fires this on completion)
        // 3. The transfer is fully device-driven: no host polling needed.
        Ok(())
    }

    fn supports_p2p(&self, src: DeviceAddr, dst: DeviceAddr) -> bool {
        // RDMA is always remote — P2P is for local-only
        src.node == dst.node
    }
}
