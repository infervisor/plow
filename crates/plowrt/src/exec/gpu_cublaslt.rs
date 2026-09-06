use crate::asset::devblob::{DevProg, DevTensor};
use crate::{Result, RuntimeError};
use packet::dev::DevOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecodeSegment {
    pub(super) instruction: usize,
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) output_bytes: u64,
    pub(super) input_bytes: u64,
    pub(super) weight_bytes: u64,
}

pub(super) fn decode_segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
) -> Result<Vec<Option<DecodeSegment>>> {
    let fail = || {
        RuntimeError::Rejected(
            "packet-declared cuBLASLt decode requires isolated BF16 GEMV segments".into(),
        )
    };
    if !(1..=32).contains(&program.t) || program.l2_domains != 0 || program.gq_stream.is_empty() {
        return Err(fail());
    }
    let count = program.gq_seg_ofs.len().checked_sub(1).ok_or_else(fail)?;
    if count == 0
        || roles.len() != count
        || !roles.contains(&plow_asset::segment_roles::CUBLASLT)
        || program.gq_seg_ofs.first() != Some(&0)
        || program.gq_seg_ofs.last().copied() != Some(program.gq_stream.len() as u32)
    {
        return Err(fail());
    }
    let mut routes = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        let entries = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(fail)?;
        if entries.is_empty() || entries.iter().any(|entry| entry.seg as usize != segment) {
            return Err(fail());
        }
        if roles[segment] != plow_asset::segment_roles::CUBLASLT {
            continue;
        }
        let instruction = entries[0].inst as usize;
        let op = program.insts.get(instruction).ok_or_else(fail)?;
        if op.op != DevOp::Gemv as u16
            || op.i[0] != program.t
            || op.i[1] == 0
            || op.i[2] == 0
            || op.i[3..].iter().any(|&value| value != 0)
            || entries
                .iter()
                .any(|entry| entry.inst as usize != instruction)
            || program.stream.iter().any(|entry| {
                (entry.inst as usize == instruction) != (entry.seg as usize == segment)
            })
        {
            return Err(fail());
        }
        let [m, n, k] = [op.i[0], op.i[1], op.i[2]];
        let bytes = |a: u32, b: u32| {
            u64::from(a)
                .checked_mul(u64::from(b))
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(fail)
        };
        let output_bytes = bytes(m, n)?;
        let input_bytes = bytes(m, k)?;
        let weight_bytes = bytes(n, k)?;
        for (handle, required) in [
            (op.t[0], output_bytes),
            (op.t[1], input_bytes),
            (op.t[2], weight_bytes),
        ] {
            if tensors
                .get(handle as usize)
                .is_none_or(|tensor| tensor.bytes < required)
            {
                return Err(fail());
            }
        }
        if op.t[0] == op.t[1] || op.t[0] == op.t[2] {
            return Err(fail());
        }
        routes[segment] = Some(DecodeSegment {
            instruction,
            m,
            n,
            k,
            output_bytes,
            input_bytes,
            weight_bytes,
        });
    }
    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};

    fn fixture(batch: u32) -> (DevProg, Vec<DevTensor>) {
        let ordinary = DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        };
        let mut gemv = ordinary;
        gemv.op = DevOp::Gemv as u16;
        gemv.t[..3].copy_from_slice(&[0, 1, 2]);
        gemv.i[..3].copy_from_slice(&[batch, 48, 5120]);
        let stream: Vec<_> = (0..3)
            .map(|index| StreamEnt {
                inst: index,
                seg: index as u16,
                ..Default::default()
            })
            .collect();
        let tensors = [
            ("act.out", batch as u64 * 48 * 2),
            ("act.in", batch as u64 * 5120 * 2),
            ("model.layers.0.projection.weight", 48 * 5120 * 2),
        ]
        .into_iter()
        .map(|(name, bytes)| DevTensor {
            name: name.into(),
            bytes,
            init: None,
        })
        .collect();
        (
            DevProg {
                t: batch,
                packed_prefill_only: false,
                n_counter: 0,
                insts: vec![ordinary, gemv, ordinary],
                stream: stream.clone(),
                stream_ofs: vec![0],
                stream_len: vec![3],
                waits: vec![],
                succs: vec![],
                gq_stream: stream,
                gq_seg_ofs: vec![0, 1, 2, 3],
                l2_domains: 0,
            },
            tensors,
        )
    }

    fn roles() -> [u8; 3] {
        [
            plow_asset::segment_roles::INTERPRETER,
            plow_asset::segment_roles::CUBLASLT,
            plow_asset::segment_roles::INTERPRETER,
        ]
    }

    #[test]
    fn accepts_packet_selected_bf16_projections() {
        for batch in [1, 4] {
            let (program, tensors) = fixture(batch);
            assert_eq!(
                decode_segments(&program, &tensors, &roles()).unwrap(),
                vec![
                    None,
                    Some(DecodeSegment {
                        instruction: 1,
                        m: batch,
                        n: 48,
                        k: 5120,
                        output_bytes: u64::from(batch) * 48 * 2,
                        input_bytes: u64::from(batch) * 5120 * 2,
                        weight_bytes: 48 * 5120 * 2,
                    }),
                    None,
                ]
            );
        }
    }

    #[test]
    fn rejects_invalid_precision_geometry_and_extents() {
        let (program, tensors) = fixture(4);
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].op = DevOp::GemvFp8 as u16;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].i[4] = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].i[0] = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].t[0] = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (_, mut small) = fixture(program.t);
        small[0].bytes -= 1;
        assert!(decode_segments(&program, &small, &roles()).is_err());
        let (mut overflow, _) = fixture(program.t);
        overflow.insts[1].i[1] = u32::MAX;
        overflow.insts[1].i[2] = u32::MAX;
        assert!(decode_segments(&overflow, &tensors, &roles()).is_err());
    }

    #[test]
    fn rejects_invalid_roles_and_queue_windows() {
        let (program, tensors) = fixture(1);
        assert!(decode_segments(&program, &tensors, &[0, 0, 0]).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.gq_stream[0].seg = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.stream[0].seg = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.gq_seg_ofs[1] = 4;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.l2_domains = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
    }
}
