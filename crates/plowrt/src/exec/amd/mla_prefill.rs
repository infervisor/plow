use super::DevProg;
use crate::asset::devblob::DevTensor;
use crate::{Result, RuntimeError};
use packet::dev::{DevInst64, DevOp};

pub(super) struct SplitSite {
    flash: usize,
    merge: usize,
    partial_rows: u32,
    query_capacity: u32,
}

pub(super) fn split_sites(
    prog: &DevProg,
    tensors: &[DevTensor],
) -> Result<(Vec<SplitSite>, Vec<bool>)> {
    let nseg = super::derive_segments(prog)?.len();
    let mut segments = vec![false; nseg];
    let mut sites = Vec::new();
    for (ix, flash) in prog.insts.iter().enumerate() {
        if flash.op != DevOp::FlashMlaPrefillFp8 as u16 || flash.fj[2] == 0 {
            continue;
        }
        let error = || {
            RuntimeError::Device(format!(
            "small MLA split instruction {ix} requires a pure dense QH8 FP8 segment, bounded partials and its matching merge"
        ))
        };
        let ns = flash.fj[2];
        let Some(merge) = prog.insts.get(ix + 1) else {
            return Err(error());
        };
        if prog.packed_prefill_only
            || prog.t == 0
            || prog.t >= 2048
            || !ns.is_power_of_two()
            || !(2..=32).contains(&ns)
            || flash.i[0] != 1
            || flash.i[1] != 8
            || flash.i[3] != 0
            || flash.i[4] != prog.t
            || flash.i[5] != u32::MAX
            || flash.i[6] != 0
            || flash.i[2] < prog.t
            || flash.fj[1] != 0
            || flash.t[0] == flash.t[1]
            || merge.op != DevOp::MlaMergeFold as u16
            || merge.t[1] != flash.t[0]
            || merge.t[2] != flash.t[1]
            || merge.i[0] != prog.t
            || merge.i[1] != 8
            || merge.i[4] != ns
            || merge.i[5] != 0
        {
            return Err(error());
        }
        for (tensor, width) in [(flash.t[0], 512u64), (flash.t[1], 2u64)] {
            let bytes = u64::from(prog.t) * 8 * u64::from(ns) * width * 4;
            if tensors
                .get(tensor as usize)
                .is_none_or(|tensor| tensor.bytes < bytes)
            {
                return Err(error());
            }
        }
        for (tensor, bytes) in [
            (flash.t[2], u64::from(prog.t) * 8 * 512 * 2),
            (flash.t[3], u64::from(prog.t) * 8 * 64 * 2),
            (flash.t[4], u64::from(flash.i[2]) * 512),
            (flash.t[5], u64::from(flash.i[2]) * 64 * 2),
            (flash.t[6], 4),
            (flash.t[7], u64::from(flash.i[2]) * 4),
        ] {
            if tensor == flash.t[0]
                || tensor == flash.t[1]
                || tensors
                    .get(tensor as usize)
                    .is_none_or(|tensor| tensor.bytes < bytes)
            {
                return Err(error());
            }
        }
        let mut selected = None;
        for entry in prog.stream.iter().filter(|e| e.inst as usize == ix) {
            if selected.is_some_and(|seg| seg != entry.seg) {
                return Err(error());
            }
            selected = Some(entry.seg);
        }
        let Some(seg) = selected else {
            return Err(error());
        };
        if prog
            .stream
            .iter()
            .any(|e| e.seg == seg && e.inst as usize != ix)
        {
            return Err(error());
        }
        segments[seg as usize] = true;
        sites.push(SplitSite {
            flash: ix,
            merge: ix + 1,
            partial_rows: prog.t * ns,
            query_capacity: prog.t,
        });
    }
    Ok((sites, segments))
}

pub(super) fn live_splits(partial_rows: u32, rows: u32, kv_len: u32) -> u32 {
    let tiles = kv_len.div_ceil(32).max(1);
    let occupancy = 304 / (rows.div_ceil(64) * 8);
    let capacity = (partial_rows / rows)
        .min(occupancy)
        .min(32)
        .min(tiles)
        .max(1);
    1 << capacity.ilog2()
}

pub(super) fn rebase(sites: &[SplitSite], insts: &mut [DevInst64], kv_len: u32) -> Result<()> {
    for site in sites {
        let rows = insts[site.flash].i[4];
        if rows == 0 || rows > site.query_capacity || rows > kv_len {
            return Err(RuntimeError::Device(
                "small MLA append exceeds query or KV capacity".into(),
            ));
        }
        // Flash and merge use the same compact [rows, heads, splits] layout.
        let ns = live_splits(site.partial_rows, rows, kv_len);
        insts[site.flash].fj[2] = ns;
        insts[site.merge].i[4] = ns;
    }
    Ok(())
}
