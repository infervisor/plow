//! `PLOW_MLA_PF_DENSE`: the NoPE absorbed-MLA flash prefill segment on its own gfx950 object
//! (`runtime/amd/mla_pf_dense_gfx950.hip`) instead of the four-wave flash interpreter.
//!
//! Same operand contract as the interpreter's `d_flash_mla_prefill_v2<512, 0>`: unnormalized
//! f32 partials `[row][head][ns][512]` plus base-2 `(m, l)` pairs, so the merge/fold consumer is
//! unchanged, or under the W_ofold epilogue normalized bf16 rows. A work item is 8 query tokens x 8 heads for one causal KV split.

use packet::dev::{DevInst64, DevOp, PrefillSpan, TENSOR_NONE16};

use crate::asset::devblob::DevProg;
use crate::device::DeviceMem;
use crate::{Result, RuntimeError};

pub(super) const OBJECT: &str = "mla_pf_dense_gfx950.elf";
pub(super) const SYMBOL: &str = "plow_mla_pf_dense_gfx950";
pub(super) const MARKER: &str = "plow_mla_pf_dense_abi_1";
/// The DSA sparse (union-table gather) entry point in the same object.
pub(super) const GATHER_SYMBOL: &str = "plow_mla_pf_gather_gfx950";
pub(super) const GATHER_MARKER: &str = "plow_mla_pf_gather_abi_1";
/// gfx950 LDS per CU; the object is static-LDS only.
pub(super) const LDS_MAX: u32 = 160 * 1024;
pub(super) const THREADS: u32 = 256;
const TOKENS_PER_ITEM: u32 = 8;
const HEADS: u32 = 8;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct Args {
    pub opart: u64,
    pub mlpart: u64,
    pub q: u64,
    pub ckv: u64,
    pub kv_len: u64,
    pub spans: u64,
    pub n_spans: u32,
    pub n_tok: u32,
    pub n_head: u32,
    pub kv_stride: u32,
    pub kv_mask: u32,
    pub ns: u32,
    pub scale: f32,
    /// Bit 0: the W_ofold epilogue (normalized bf16 rows, ns == 1, mlpart unwritten).
    pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<Args>() == 80);

/// `plow_mla_pf_gather_gfx950` kernargs: single sequence, ns = 1, the per-8-query union table.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct GatherArgs {
    pub opart: u64,
    pub mlpart: u64,
    pub q: u64,
    pub ckv: u64,
    pub uni: u64,
    pub n_tok: u32,
    pub cap: u32,
    pub kv_mask: u32,
    pub scale: f32,
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<GatherArgs>() == 64);

#[derive(Clone, Copy, Debug)]
pub(super) struct Route {
    pub args: Args,
    /// Instruction index of the flash, so a ragged chunk reads its patched `i[4]`.
    pub inst: usize,
    /// Packet tensor ids of `opart, mlpart, q, ckv, kv_len` for KV-slot rebasing.
    pub tens: [u16; 5],
    /// DSA sparse prefill: (union table address, its packet tensor id, cap).
    pub gather: Option<(u64, u16, u32)>,
}

impl Route {
    pub fn gather_args(&self) -> Option<GatherArgs> {
        let (uni, _, cap) = self.gather?;
        Some(GatherArgs {
            opart: self.args.opart,
            mlpart: self.args.mlpart,
            q: self.args.q,
            ckv: self.args.ckv,
            uni,
            n_tok: self.args.n_tok,
            cap,
            kv_mask: self.args.kv_mask,
            scale: self.args.scale,
            _pad: [0; 2],
        })
    }

    /// Single-sequence grid: one workgroup per (8-token pack, split).
    pub fn grid(&self) -> u32 {
        self.args.n_tok.div_ceil(TOKENS_PER_ITEM) * self.args.ns
    }

    pub fn rebase(&mut self, n_tok: u32) {
        self.args.n_tok = n_tok;
    }
}

/// Packed grid: every span's packs, times the split count.
pub(super) fn span_grid(spans: &[PrefillSpan], ns: u32) -> u32 {
    spans
        .iter()
        .map(|s| s.n_rows.div_ceil(TOKENS_PER_ITEM))
        .sum::<u32>()
        * ns
}

/// A dense bf16 NoPE absorbed-MLA prefill the object covers: one batch, eight heads, no window,
/// no gather/sparse table, no MHA form. The W_ofold epilogue (i6 bit 8) needs the un-split walk.
pub(super) fn eligible(d: &DevInst64) -> bool {
    d.op == DevOp::FlashMlaPrefill as u16
        && d.i[3] & 0x8000_0000 != 0
        && d.i[3] & 0x7fff_ffff == 0
        && d.i[0] == 1
        && d.i[1] == HEADS
        && d.i[4] != 0
        && d.i[2] != 0
        && d.t[7] == TENSOR_NONE16
        && ((d.i[6] >> 8) == 0 || ((d.i[6] >> 8) == 1 && d.i[6] & 0xff <= 1))
        && (d.i[6] >> 8 == 1 || d.t[1] != TENSOR_NONE16)
        && [d.t[0], d.t[2], d.t[4], d.t[6]]
            .iter()
            .all(|&t| t != TENSOR_NONE16)
        && f32::from_bits(d.fj[0]).is_finite()
}

/// The DSA sparse prefill flash (t7 = the per-8-query union table, i6 = its cap): the gather
/// entry point. Single sequence only; the packed topology emits no union table.
pub(super) fn eligible_gather(d: &DevInst64) -> bool {
    d.op == DevOp::FlashMlaPrefill as u16
        && d.i[3] & 0x8000_0000 != 0
        && d.i[3] & 0x7fff_ffff == 0
        && d.i[0] == 1
        && d.i[1] == HEADS
        && d.i[4] != 0
        && d.t[7] != TENSOR_NONE16
        && d.i[6] != 0
        && d.i[6] & 0x8000_0000 == 0
        && [d.t[0], d.t[1], d.t[2], d.t[4], d.t[6]]
            .iter()
            .all(|&t| t != TENSOR_NONE16)
        && f32::from_bits(d.fj[0]).is_finite()
}

/// Segments that are exactly one eligible flash with no counter obligations. Token-batch bodies
/// and row-split siblings keep the interpreter (their rows are not a plain chunk or span set).
pub(super) fn routes(
    prog: &DevProg,
    devp: &[DeviceMem],
    segments: usize,
) -> Result<Vec<Option<Route>>> {
    let mut out = vec![None; segments];
    // A GLM-5 row-split sibling's 64-head flash is not eligible below; the GLM-5.3 A2A
    // sibling's virtual eight-head gather is.
    if prog.role.is_token_batch_body() {
        return Ok(out);
    }
    let mut seg_inst: Vec<Option<Option<usize>>> = vec![None; segments];
    for e in &prog.stream {
        let seg = e.seg as usize;
        let Some(slot) = seg_inst.get_mut(seg) else {
            continue;
        };
        let ok = e.wait_len == 0
            && e.succ_len == 0
            && prog
                .insts
                .get(e.inst as usize)
                .is_some_and(|d| eligible(d) || eligible_gather(d));
        *slot = match *slot {
            None if ok => Some(Some(e.inst as usize)),
            Some(Some(i)) if ok && i == e.inst as usize => Some(Some(i)),
            _ => Some(None),
        };
    }
    let addr = |h: u16, what: &str| -> Result<u64> {
        devp.get(h as usize).map(|m| m.base).ok_or_else(|| {
            RuntimeError::Device(format!(
                "dense MLA prefill operand `{what}` handle {h} is invalid"
            ))
        })
    };
    for (seg, inst) in seg_inst.into_iter().enumerate() {
        let Some(Some(ix)) = inst else {
            continue;
        };
        let d = &prog.insts[ix];
        let gather = eligible_gather(d);
        let ns = if gather { 1 } else { (d.i[6] & 0xff).max(1) };
        out[seg] = Some(Route {
            args: Args {
                opart: addr(d.t[0], "Opart")?,
                mlpart: if d.t[1] == TENSOR_NONE16 {
                    0
                } else {
                    addr(d.t[1], "mlpart")?
                },
                q: addr(d.t[2], "Qabs")?,
                ckv: addr(d.t[4], "Ckv")?,
                kv_len: addr(d.t[6], "kv_len")?,
                spans: 0,
                n_spans: 0,
                n_tok: d.i[4],
                n_head: d.i[1],
                kv_stride: d.i[2],
                kv_mask: d.i[5],
                ns,
                scale: f32::from_bits(d.fj[0]),
                flags: if gather { 0 } else { (d.i[6] >> 8) & 1 },
            },
            inst: ix,
            tens: [d.t[0], d.t[1], d.t[2], d.t[4], d.t[6]],
            gather: if gather {
                Some((addr(d.t[7], "union")?, d.t[7], d.i[6]))
            } else {
                None
            },
        });
    }
    Ok(out)
}
