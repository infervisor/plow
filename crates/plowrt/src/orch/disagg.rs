//! §L Prefill/decode disaggregation.
//!
//! Route prefill and decode to different executor pools (or instances); the
//! prefill pool produces KV blocks that transfer to the decode pool over
//! NVLink/RDMA (reusing the `Rdma` opcode + `KV_STORE` path). Prefill-heavy and
//! decode-heavy work stop interfering, improving TTFT and TPOT tails.

/// Which pool a stage runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Prefill,
    Decode,
}

/// A KV hand-off from the prefill pool to the decode pool: the block range to
/// transfer (device-to-device, never through host RAM).
#[derive(Clone, Copy, Debug)]
pub struct KvHandoff {
    pub first_block: u32,
    pub block_count: u32,
    pub src_device: u8,
    pub dst_device: u8,
}
