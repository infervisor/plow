use crate::{asset::devblob::DevProg, Result, RuntimeError};
use packet::dev::{DevInst64, DevOp};

fn compatible(inst: &DevInst64) -> bool {
    [DevOp::Gemm, DevOp::GemmSmall, DevOp::GemmMed]
        .iter()
        .any(|op| inst.op == *op as u16)
        && (2..=128).contains(&inst.i[0])
        && inst.i[1] > 0
        && inst.i[2] > 0
        && inst.i[2] % 8 == 0
        && inst.i[6] != 0
        && inst.i[7] != 0
        && inst.blocks > 0
}

pub(super) fn small_gemm_segments(g: &DevProg, classes: &[u8], abi: u32) -> Result<Vec<bool>> {
    let reject = || RuntimeError::Rejected("invalid small-GEMM segment queue".into());
    if !matches!(abi, 1 | 2 | 3)
        || g.l2_domains != 0
        || g.gq_seg_ofs.len() != classes.len() + 1
        || g.gq_seg_ofs.first() != Some(&0)
        || g.gq_seg_ofs.last().copied() != Some(g.gq_stream.len() as u32)
    {
        return Err(reject());
    }
    let mut selected = vec![false; classes.len()];
    for (seg, bounds) in g.gq_seg_ofs.windows(2).enumerate() {
        let entries = g
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(reject)?;
        if entries
            .iter()
            .any(|e| e.seg as usize != seg || e.inst as usize >= g.insts.len())
        {
            return Err(reject());
        }
        if classes[seg] != 8 || entries.is_empty() {
            continue;
        }
        // Fine successors describe tile ownership; only complete coarse packets may change tiles.
        if entries
            .iter()
            .any(|e| e.flags != 0 || !compatible(&g.insts[e.inst as usize]))
        {
            continue;
        }
        let mut slices = std::collections::BTreeMap::<u32, Vec<u32>>::new();
        for entry in entries {
            slices.entry(entry.inst).or_default().push(entry.slice);
        }
        if abi == 2 || abi == 3 {
            let inst = &g.insts[entries[0].inst as usize];
            let shape = (inst.i[1], inst.i[2]);
            let shape_ok = match abi {
                2 => matches!(shape, (3840, 4096 | 8192 | 15360)),
                3 => matches!(shape, (5376, 8192 | 16384 | 21504)),
                _ => unreachable!(),
            };
            if slices.len() != 1 || inst.i[0] != 128 || inst.i[4] != 0 || !shape_ok {
                continue;
            }
        }
        selected[seg] = slices.iter_mut().all(|(&inst, owned)| {
            owned.sort_unstable();
            owned.len() == g.insts[inst as usize].blocks as usize
                && owned
                    .iter()
                    .enumerate()
                    .all(|(i, &slice)| slice as usize == i)
        });
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    fn fixture() -> DevProg {
        let mut inst = DevInst64::default();
        inst.op = DevOp::Gemm as u16;
        inst.blocks = 2;
        inst.i = [128, 512, 3840, 0, 0, 0, 8, 9];
        let stream: Vec<_> = (0..2)
            .map(|slice| StreamEnt {
                slice,
                ..Default::default()
            })
            .collect();
        DevProg {
            t: 128,
            packed_prefill_only: false,
            n_counter: 1,
            insts: vec![inst],
            stream: stream.clone(),
            stream_ofs: vec![0],
            stream_len: vec![2],
            waits: vec![],
            succs: vec![],
            gq_stream: stream,
            gq_seg_ofs: vec![0, 2],
            l2_domains: 0,
        }
    }

    #[test]
    fn selects_complete_mapped_bf16_packet() {
        assert_eq!(small_gemm_segments(&fixture(), &[8], 1).unwrap(), [true]);
    }

    #[test]
    fn leaves_large_fp8_unmapped_and_fine_packets_on_default() {
        for variant in 0..5 {
            let mut g = fixture();
            match variant {
                0 => g.insts[0].i[0] = 512,
                1 => g.insts[0].op = DevOp::GemmFp8 as u16,
                2 => g.insts[0].i[6] = 0,
                3 => g.gq_stream[0].flags = 1,
                _ => g.insts[0].i[0] = 1,
            }
            assert_eq!(small_gemm_segments(&g, &[8], 1).unwrap(), [false]);
        }
    }

    #[test]
    fn rejects_duplicate_or_partial_slice_ownership() {
        let mut g = fixture();
        g.gq_stream[1].slice = 0;
        assert_eq!(small_gemm_segments(&g, &[8], 1).unwrap(), [false]);
        g.insts[0].blocks = 3;
        assert_eq!(small_gemm_segments(&g, &[8], 1).unwrap(), [false]);
    }

    #[test]
    fn mixed_segment_keeps_one_compatible_object_for_all_instructions() {
        let mut g = fixture();
        g.insts.push(g.insts[0]);
        let mut next = g.gq_stream.clone();
        for entry in &mut next {
            entry.inst = 1;
        }
        g.gq_stream.extend(next);
        g.gq_seg_ofs[1] = 4;
        assert_eq!(small_gemm_segments(&g, &[8], 1).unwrap(), [true]);
        g.insts[1].op = DevOp::GemmFp8 as u16;
        assert_eq!(small_gemm_segments(&g, &[8], 1).unwrap(), [false]);
    }

    #[test]
    fn m64n64_requires_isolated_gemma12_o_down_shape() {
        let mut g = fixture();
        assert_eq!(small_gemm_segments(&g, &[8], 2).unwrap(), [false]);
        g.insts[0].i[1] = 3840;
        g.insts[0].i[2] = 15360;
        assert_eq!(small_gemm_segments(&g, &[8], 2).unwrap(), [true]);
        for (n, k) in [
            (5376, 8192),
            (5376, 16384),
            (5376, 21504),
            (5120, 6144),
            (5120, 17408),
        ] {
            g.insts[0].i[1] = n;
            g.insts[0].i[2] = k;
            assert_eq!(small_gemm_segments(&g, &[8], 2).unwrap(), [false]);
        }
        g.insts[0].i[1] = 3840;
        g.insts[0].i[2] = 15360;
        g.insts.push(g.insts[0]);
        let mut entries = g.gq_stream.clone();
        for entry in &mut entries {
            entry.inst = 1;
        }
        g.gq_stream.extend(entries);
        g.gq_seg_ofs[1] = 4;
        assert_eq!(small_gemm_segments(&g, &[8], 2).unwrap(), [false]);
        assert_eq!(small_gemm_segments(&g, &[8], 1).unwrap(), [true]);
        assert!(small_gemm_segments(&g, &[8], 4).is_err());
    }

    #[test]
    fn m64n128_requires_isolated_gemma31_o_down() {
        for (n, k, selected) in [
            (5376, 8192, true),
            (5376, 16384, true),
            (5376, 21504, true),
            (5376, 5376, false),
            (3840, 15360, false),
            (5120, 17408, false),
        ] {
            let mut g = fixture();
            g.insts[0].i[1] = n;
            g.insts[0].i[2] = k;
            assert_eq!(small_gemm_segments(&g, &[8], 3).unwrap(), [selected]);
            if selected {
                g.insts[0].i[0] = 65;
                assert_eq!(small_gemm_segments(&g, &[8], 3).unwrap(), [false]);
                g.insts[0].i[0] = 128;
                g.insts[0].i[4] = 64;
                assert_eq!(small_gemm_segments(&g, &[8], 3).unwrap(), [false]);
                g.insts[0].i[4] = 0;
                g.gq_stream[1].slice = 0;
                assert_eq!(small_gemm_segments(&g, &[8], 3).unwrap(), [false]);
            }
        }
    }

    #[test]
    fn rejects_malformed_queue_windows() {
        let mut g = fixture();
        g.gq_seg_ofs[1] = 3;
        assert!(small_gemm_segments(&g, &[8], 1).is_err());
        let mut g = fixture();
        g.gq_stream[1].seg = 1;
        assert!(small_gemm_segments(&g, &[8], 1).is_err());
    }
}
