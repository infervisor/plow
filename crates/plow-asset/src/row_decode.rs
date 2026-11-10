use crate::decode_objects::DecodeObjects;
use crate::program::Packet;
use packet::dev::{DevOp, TENSOR_NONE16};
use std::collections::BTreeMap;

pub const SECTION: &str = "row_decode_objects.json";
pub const CAPABILITY: &str = "plow_bf16_token_batch_1";

pub fn programs(packet: &Packet<'_>) -> Vec<(usize, u32)> {
    packet
        .programs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.role.is_row_decode_body())
        .map(|(index, p)| (index, p.rows))
        .collect()
}

pub fn validate(
    packet: &Packet<'_>,
    objects: &DecodeObjects,
    kernarg_bytes: usize,
) -> Result<(), String> {
    objects.validate(&programs(packet), packet.n_cu, kernarg_bytes)?;
    if packet.tp {
        return Err("row decode requires single-GPU programs".into());
    }
    for (index, _) in programs(packet) {
        let p = &packet.programs[index];
        let dependencies = crate::splitk::dependencies(p)?;
        let mut writers = BTreeMap::new();
        let ordered = |consumer: usize, producer: usize| {
            let mut pending = vec![consumer];
            let mut seen = std::collections::BTreeSet::new();
            while let Some(pc) = pending.pop() {
                if pc == producer {
                    return true;
                }
                if seen.insert(pc) {
                    pending.extend(dependencies[pc].iter().copied());
                }
            }
            false
        };
        let mut attention = 0;
        for (pc, d) in p.insts.iter().enumerate() {
            match DevOp::from_u16(d.op).ok_or("row decode unknown opcode")? {
                DevOp::HeadNormRope if d.fj[1] != 0 => {
                    writers.insert(d.t[0], pc);
                    if d.i[7] == 1 {
                        writers.insert(d.t[6], pc);
                    }
                }
                DevOp::FlashDecode => {
                    if d.i[0] != p.rows
                        || d.i[1] >> 16 != 0
                        || d.fj[2] != 0
                        || d.t[6] != TENSOR_NONE16
                        || d.t[7] != TENSOR_NONE16
                    {
                        return Err("row decode requires plain BF16 attention".into());
                    }
                    for cache in &d.t[3..5] {
                        let writer = writers.get(cache).ok_or("row decode missing KV writer")?;
                        if !ordered(pc, *writer) {
                            return Err(
                                "row decode KV stores are not ordered before attention".into()
                            );
                        }
                    }
                    attention += 1;
                }
                DevOp::FlashDecodeFp8
                | DevOp::FlashPrefill
                | DevOp::FlashPrefillFp8
                | DevOp::QwenHeadNormRope
                | DevOp::EmbedPosBf16 => {
                    return Err("unsupported row decode cache or position operation".into());
                }
                DevOp::Nop
                | DevOp::Embed
                | DevOp::EmbedFrameBf16
                | DevOp::EmbedOverlayBf16
                | DevOp::RmsNorm
                | DevOp::HeadNormRope
                | DevOp::Gemv
                | DevOp::GemvQkv
                | DevOp::GemvGlu
                | DevOp::Glu
                | DevOp::Residual
                | DevOp::AddNorm
                | DevOp::NormResidual
                | DevOp::NormResidualNorm
                | DevOp::FlashMerge
                | DevOp::Argmax
                | DevOp::ArgmaxFin
                | DevOp::SoftCap => {}
                _ => return Err("operation outside the BF16 row-decode contract".into()),
            }
        }
        if attention == 0 {
            return Err("row decode has no attention".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_objects::{DecodeObject, DecodeProgramObject};
    use packet::devbuild::{row_decode_program_t, Builder, Model};

    fn fixture() -> (Model, DecodeObjects) {
        let mut b = Builder::new(132);
        b.force_uniseg();
        let k = b.tensor("kv.0.k", 4096);
        let v = b.tensor("kv.0.v", 4096);
        let kv = b.emit(DevOp::HeadNormRope, (0..8).collect(), &[], |d| {
            d.t[0] = k;
            d.t[6] = v;
            d.i[7] = 1;
            d.j[0] = 16;
        });
        b.emit(DevOp::FlashDecode, (0..16).collect(), &[kv], |d| {
            d.i[0] = 16;
            d.t[3] = k;
            d.t[4] = v;
        });
        let tensors = b.tensors();
        let program = b.finish();
        let model = Model {
            n_cu: 132,
            target: 0,
            tensors,
            progs: vec![program],
            kv_row_insts: vec![],
            prog_t: vec![row_decode_program_t(16)],
            gen: vec![],
        };
        let metadata = DecodeObjects {
            version: 1,
            kernarg_bytes: 192,
            objects: BTreeMap::from([(
                0,
                DecodeObject {
                    file: "row.cubin".into(),
                    sha256: "a".repeat(64),
                    profile: "sm90a".into(),
                    entry: "_Z12interp_sm90a11PlowProgram".into(),
                    threads: 256,
                    arena_bytes: 73728,
                    grid: 132,
                },
            )]),
            programs: vec![DecodeProgramObject {
                index: 0,
                rows: 16,
                object: 0,
            }],
        };
        (model, metadata)
    }

    #[test]
    fn requires_all_kv_producer_blocks_before_every_query() {
        let (model, metadata) = fixture();
        assert!(crate::program::with_model(&model, |p| validate(p, &metadata, 192)).is_ok());
        for mutation in 0..4 {
            let (mut bad, _) = fixture();
            let p = &mut bad.progs[0];
            match mutation {
                0 => p.waits[0].threshold -= 1,
                1 => {
                    for e in p
                        .stream
                        .iter_mut()
                        .chain(&mut p.gq_stream)
                        .filter(|e| e.inst == 1)
                    {
                        e.wait_len = 0;
                    }
                }
                2 => p.insts[0].i[7] = 0,
                3 => p.insts[1].i[0] = 8,
                _ => unreachable!(),
            }
            assert!(
                crate::program::with_model(&bad, |p| validate(p, &metadata, 192)).is_err(),
                "mutation {mutation}"
            );
        }
    }
}
