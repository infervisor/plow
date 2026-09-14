//! Planning the byte copies that move a CPU-prefilled head's KV rows into a
//! device sequence slot.
//!
//! Pure functions over the device ISA and the packet's tensor table, like
//! [`super::kvrow`], so the addressing can be tested without a device. Getting
//! it wrong is not a fault: it is a slot whose rows `[0, n)` hold the right
//! bytes at the wrong offsets, which reads as fluent wrong output.
//!
//! ## Only seq-major caches, and the packet says which those are
//!
//! A row range is one contiguous run inside a slot only when the cache's
//! sequence axis is its outer one. The two families in the ISA differ here:
//!
//! * **MLA** declares `kv.{l}.ckv` as `dbatch × ctx × dk` and `kv.{l}.krot` as
//!   `dbatch × ctx × dr` (`devgen::mla`) — `[slot][seq][width]`. Rows `[0, n)`
//!   are one run of `n × row_bytes`.
//! * **Dense GQA** declares `kv.{l}.k` as `dbatch × kv_heads × ring × head_dim`
//!   (`devgen::gptoss`) — `[slot][head][seq][dim]`. The same rows are
//!   `kv_heads` runs strided by the ring, and a single-run copy would write
//!   head 0's rows over the front of head 0 and nothing else correctly.
//!
//! The discriminator is already in the tree and already load-bearing:
//! `rebase_chunk_rows` separates the two by `HeadNormRope`'s `fj[1]`, the KV
//! ring stride, which dense GQA sets and MLA's k_rope leaves at zero. So the
//! head-major case is REFUSED BY NAME here rather than addressed by a guess.
//! Supporting it is a scatter plus the head count, which the packet also
//! carries; it is not in this path yet.

use packet::dev::{DevInst64, DevOp};

use super::kvrow::KvSlotTensor;
use crate::{Result, RuntimeError};

/// One planned byte copy: `bytes` from `src_off` in the head's slot to
/// `dst_off` in the device slot, both offsets relative to the tensor's base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopySpan {
    /// Index into the packet's tensor table — the same handle on both sides,
    /// which the KV contract digest is what guarantees.
    pub handle: usize,
    pub src_off: u64,
    pub dst_off: u64,
    pub bytes: u64,
}

/// Refuse a prefill program whose KV caches are not addressable as row ranges.
///
/// Called at load, so a model that cannot hand off says so once rather than
/// producing a wrong answer per request.
pub(crate) fn check_seq_major(insts: &[DevInst64], names: &[String]) -> Result<()> {
    for d in insts {
        let head_major = (d.op == DevOp::HeadNormRope as u16
            || d.op == DevOp::HeadNormRopeFp8 as u16)
            && d.fj[1] != 0;
        if !head_major {
            continue;
        }
        let dst = names.get(d.t[0] as usize).map(String::as_str).unwrap_or("?");
        return Err(RuntimeError::Device(format!(
            "CPU head prefill needs seq-major KV caches; `{dst}` is written with a KV ring \
             stride (head-major `[slot][head][seq][dim]`), so its rows are a strided scatter \
             rather than one run. Head-major handoff is not implemented."
        )));
    }
    Ok(())
}

/// How many prompt rows one entry of each cache covers, for the caches where
/// that is not 1.
///
/// The DSA pooled indexer key cache holds one softmax-pooled vector per
/// `index_kpool` consecutive tokens (`devgen::mla`), so its slot is
/// `ctx / pool` entries, not `ctx`. `DsaPoolCompress` is the instruction that
/// writes it and carries the pool size in `i[1]`.
///
/// **This has to be read off the packet; arithmetic does not find it.** A
/// pooled cache's per-slot extent is usually still divisible by the context —
/// `(ctx/pool) × width` divides `ctx` whenever `pool` divides `width` — so a
/// divisibility check accepts it and then copies `rows × (per_slot/ctx)` bytes,
/// which is the wrong length by exactly the pooling factor. Silent, and wrong
/// in the direction that leaves the indexer scoring against stale keys.
pub(crate) fn pooled_caches(insts: &[DevInst64], names: &[String]) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    for d in insts {
        if d.op != DevOp::DsaPoolCompress as u16 {
            continue;
        }
        let Some(name) = names.get(d.t[0] as usize) else {
            continue;
        };
        let pool = d.i[1].max(1);
        if pool > 1 && !out.iter().any(|(n, _)| n == name) {
            out.push((name.clone(), pool));
        }
    }
    out
}

/// Plan the copies moving `rows` prompt rows from `src_slot` of the head engine
/// into `dst_slot` of the device engine.
///
/// `rows_per_slot` is the compiled context. `pools` is [`pooled_caches`]; a
/// cache named there advances one entry per `pool` prompt rows, so a head must
/// end on a pool boundary — a head stopping mid-group would hand over an entry
/// pooled from only part of its tokens, and the device would not recompute it.
pub(crate) fn plan(
    tensors: &[KvSlotTensor],
    rows_per_slot: u32,
    pools: &[(String, u32)],
    src_slot: u32,
    dst_slot: u32,
    rows: u32,
) -> Result<Vec<CopySpan>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    if rows_per_slot == 0 || rows > rows_per_slot {
        return Err(RuntimeError::Rejected(format!(
            "head of {rows} rows past the compiled context {rows_per_slot}"
        )));
    }
    let mut plan = Vec::with_capacity(tensors.len());
    for t in tensors {
        let pool = pools
            .iter()
            .find(|(n, _)| *n == t.name)
            .map(|&(_, p)| p)
            .unwrap_or(1);
        if !rows.is_multiple_of(pool) {
            return Err(RuntimeError::Rejected(format!(
                "head of {rows} rows does not end on a pool boundary of `{}` (pool {pool}); \
                 the boundary entry would be pooled from part of its tokens",
                t.name
            )));
        }
        if !rows_per_slot.is_multiple_of(pool) {
            return Err(RuntimeError::Device(format!(
                "pooled cache `{}` has pool {pool}, which does not divide the compiled context \
                 {rows_per_slot}",
                t.name
            )));
        }
        let entries_per_slot = u64::from(rows_per_slot / pool);
        if !t.per_slot_bytes.is_multiple_of(entries_per_slot) {
            return Err(RuntimeError::Device(format!(
                "KV cache `{}` holds {} bytes per slot, not a whole number of entries at \
                 context {rows_per_slot} pool {pool}; its rows are not addressable as a byte \
                 range",
                t.name, t.per_slot_bytes
            )));
        }
        let entry_bytes = t.per_slot_bytes / entries_per_slot;
        if entry_bytes == 0 {
            continue;
        }
        plan.push(CopySpan {
            handle: t.handle,
            src_off: u64::from(src_slot) * t.per_slot_bytes,
            dst_off: u64::from(dst_slot) * t.per_slot_bytes,
            bytes: u64::from(rows / pool) * entry_bytes,
        });
    }
    Ok(plan)
}

/// Total bytes a plan moves — the transfer term the head budget prices.
pub(crate) fn plan_bytes(plan: &[CopySpan]) -> u64 {
    plan.iter().map(|c| c.bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(handle: usize, name: &str, per_slot_bytes: u64) -> KvSlotTensor {
        KvSlotTensor {
            handle,
            name: name.to_string(),
            per_slot_bytes,
        }
    }

    /// GLM MLA at ctx 8: `ckv` is 512 wide bf16, `krot` 64 wide bf16.
    fn mla() -> Vec<KvSlotTensor> {
        vec![t(3, "kv.0.ckv", 8 * 512 * 2), t(4, "kv.0.krot", 8 * 64 * 2)]
    }

    #[test]
    fn a_head_is_one_run_per_cache_at_the_front_of_each_slot() {
        let p = plan(&mla(), 8, &[], 0, 0, 3).unwrap();
        assert_eq!(
            p,
            vec![
                CopySpan { handle: 3, src_off: 0, dst_off: 0, bytes: 3 * 512 * 2 },
                CopySpan { handle: 4, src_off: 0, dst_off: 0, bytes: 3 * 64 * 2 },
            ]
        );
    }

    #[test]
    fn the_head_slot_and_the_device_slot_are_independent() {
        // The head pool's slot count has nothing to do with the GPU batch, so
        // the two indices must be applied to their own sides and not swapped.
        let p = plan(&mla(), 8, &[], 1, 5, 2).unwrap();
        assert_eq!(p[0].src_off, 8 * 512 * 2);
        assert_eq!(p[0].dst_off, 5 * 8 * 512 * 2);
        assert_eq!(p[0].bytes, 2 * 512 * 2);
    }

    #[test]
    fn a_zero_row_head_moves_nothing() {
        assert!(plan(&mla(), 8, &[], 0, 0, 0).unwrap().is_empty());
    }

    #[test]
    fn a_head_cannot_run_past_the_compiled_context() {
        let err = plan(&mla(), 8, &[], 0, 0, 9).unwrap_err().to_string();
        assert!(err.contains("past the compiled context 8"), "{err}");
        // The whole context is legal: that is the whole-request case.
        assert!(plan(&mla(), 8, &[], 0, 0, 8).is_ok());
    }

    /// The defect a divisibility check does NOT catch, pinned: `kv.0.kidx` at
    /// pool 4 holds 2 entries per slot of 64 bytes each, and 128 bytes IS
    /// divisible by the context 8. Treating it as 8 rows of 16 copies a
    /// quarter of the bytes a 4-row head owes.
    #[test]
    fn a_pooled_cache_is_copied_by_entries_not_rows() {
        let pooled = vec![t(7, "kv.0.kidx", 2 * 64)];
        let pools = [("kv.0.kidx".to_string(), 4u32)];
        let p = plan(&pooled, 8, &pools, 0, 0, 4).unwrap();
        assert_eq!(p[0].bytes, 64, "one pooled entry, not four rows of 16");
        // Unpooled, the same tensor would have been read as 8 rows of 16.
        assert_eq!(plan(&pooled, 8, &[], 0, 0, 4).unwrap()[0].bytes, 4 * 16);
    }

    #[test]
    fn a_head_must_end_on_a_pool_boundary() {
        let pooled = vec![t(7, "kv.0.kidx", 2 * 64)];
        let pools = [("kv.0.kidx".to_string(), 4u32)];
        let err = plan(&pooled, 8, &pools, 0, 0, 6).unwrap_err().to_string();
        assert!(err.contains("pool boundary"), "{err}");
        assert!(err.contains("kv.0.kidx"), "{err}");
    }

    #[test]
    fn a_pool_size_is_read_off_the_compress_instruction() {
        let names = vec!["x".to_string(), "kv.0.kidx".to_string()];
        let mut d = inst(DevOp::DsaPoolCompress, 1, 0);
        d.i[1] = 4;
        assert_eq!(
            pooled_caches(&[d], &names),
            vec![("kv.0.kidx".to_string(), 4)]
        );
        // pool 1 is the no-op path GLM ships by default: not a pooled cache.
        let mut one = inst(DevOp::DsaPoolCompress, 1, 0);
        one.i[1] = 1;
        assert!(pooled_caches(&[one], &names).is_empty());
    }

    #[test]
    fn plan_bytes_is_what_the_budget_prices() {
        let p = plan(&mla(), 8, &[], 0, 0, 3).unwrap();
        assert_eq!(plan_bytes(&p), 3 * (512 + 64) * 2);
    }

    fn inst(op: DevOp, dst: u16, ring_stride: u32) -> DevInst64 {
        let mut d = DevInst64 {
            op: op as u16,
            ..Default::default()
        };
        d.t[0] = dst;
        d.fj[1] = ring_stride;
        d
    }

    #[test]
    fn a_ring_strided_cache_write_refuses_the_head_path() {
        let names = vec!["x".to_string(), "kv.0.k".to_string()];
        let insts = [inst(DevOp::HeadNormRope, 1, 4096)];
        let err = check_seq_major(&insts, &names).unwrap_err().to_string();
        assert!(err.contains("kv.0.k"), "{err}");
        assert!(err.contains("head-major"), "{err}");
    }

    #[test]
    fn mla_k_rope_leaves_the_ring_stride_at_zero_and_is_accepted() {
        let names = vec!["x".to_string(), "kv.0.krot".to_string()];
        // Same opcode as the dense-GQA write; `fj[1]` is the whole difference.
        let insts = [inst(DevOp::HeadNormRope, 1, 0)];
        assert!(check_seq_major(&insts, &names).is_ok());
    }

    #[test]
    fn the_fp8_twin_is_checked_too() {
        // A bf16-only test silently matches nothing on an fp8-KV packet.
        let names = vec!["x".to_string(), "kv.0.k".to_string()];
        let insts = [inst(DevOp::HeadNormRopeFp8, 1, 4096)];
        assert!(check_seq_major(&insts, &names).is_err());
    }
}
