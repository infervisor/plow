use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

use crate::packet_pipeline::{HandoffAccessMode, PipelineDType};
use crate::program::Packet;

const TYPES: [PipelineDType; 7] = [
    PipelineDType::U8,
    PipelineDType::U32,
    PipelineDType::U64,
    PipelineDType::I16,
    PipelineDType::F16,
    PipelineDType::Bf16,
    PipelineDType::F32,
];
const ALL: u8 = (1 << TYPES.len()) - 1;

fn bit(dtype: PipelineDType) -> u8 {
    1 << TYPES.iter().position(|&t| t == dtype).unwrap()
}

/// IR types/accesses only; bounds, row mappings, ordering and lifetimes are separate obligations.
pub struct NativePacketTypes {
    parent: Vec<usize>,
    allowed: Vec<u8>,
    untyped: Vec<bool>,
    accesses: Vec<(bool, bool)>,
}

impl NativePacketTypes {
    pub fn infer(packet: &Packet<'_>, program: usize) -> Result<Self, String> {
        let program = packet
            .programs
            .get(program)
            .ok_or("native types: missing program")?;
        let mut result = Self {
            parent: (0..packet.tensors.len()).collect(),
            allowed: vec![ALL; packet.tensors.len()],
            untyped: vec![false; packet.tensors.len()],
            accesses: vec![(false, false); packet.tensors.len()],
        };
        for instruction in program.insts {
            result.instruction(instruction)?;
        }
        Ok(result)
    }

    fn root(&self, mut id: usize) -> usize {
        while self.parent[id] != id {
            id = self.parent[id];
        }
        id
    }

    pub fn dtype(&self, tensor: usize) -> Result<PipelineDType, String> {
        if tensor >= self.parent.len() {
            return Err("native types: tensor out of range".into());
        }
        let root = self.root(tensor);
        if self.untyped[root] || self.allowed[root].count_ones() != 1 {
            return Err("native types: ambiguous, conflicting or unaudited tensor use".into());
        }
        Ok(TYPES[self.allowed[root].trailing_zeros() as usize])
    }

    pub fn access(&self, tensor: usize) -> Option<HandoffAccessMode> {
        match *self.accesses.get(tensor)? {
            (true, true) => Some(HandoffAccessMode::ReadWrite),
            (true, false) => Some(HandoffAccessMode::Read),
            (false, true) => Some(HandoffAccessMode::Write),
            _ => None,
        }
    }

    fn instruction(&mut self, d: &DevInst64) -> Result<(), String> {
        use PipelineDType::{Bf16, F16, F32, U32, U64};
        let op = DevOp::from_u16(d.op).ok_or("native types: unknown opcode")?;
        let writes = crate::logical_effects::outputs(d)?;
        let mut types = [None; 8];
        let mut copy = false;
        let mut read_output = false;
        let mut set = |slots: &[usize], dtype| {
            for &slot in slots {
                types[slot] = Some(dtype);
            }
        };
        match op {
            DevOp::Nop => return Ok(()),
            DevOp::CopyColsF32 if d.i[7] & 1 == 0 => {
                copy = true;
                set(&[2], U32);
            }
            DevOp::GatherRowsF32 => {
                set(&[0], F32);
                set(&[1], if d.i[7] & 1 != 0 { F16 } else { F32 });
                set(&[2], U32);
                read_output = d.i[7] & 2 != 0;
            }
            DevOp::StreamTailF32 => {
                set(&[0, 1, 2], F32);
                set(&[3, 4, 5], U32);
            }
            DevOp::Conv1dF32 | DevOp::ConvTranspose1dF32 => {
                set(&[0, 1, 3, 4, 5], F32);
                set(&[2], if d.fj[2] & (1 << 12) != 0 { F16 } else { F32 });
                set(&[6], U32);
                if op == DevOp::Conv1dF32 && d.fj[2] & (1 << 20) == 0 {
                    set(&[7], F32);
                }
            }
            DevOp::LayerNormF32 => set(&[0, 1, 2, 3], F32),
            DevOp::UnaryF32
            | DevOp::SiluF32
            | DevOp::ReluF32
            | DevOp::RowStatsF32
            | DevOp::RmsNormF32 => set(&[0, 1, 2], F32),
            DevOp::BinaryF32 | DevOp::ScaledAddF32 | DevOp::BroadcastAddF32 => set(&[0, 1, 2], F32),
            DevOp::ZeroF32 => set(&[0], F32),
            DevOp::LengthAffineU32 => set(&[0, 1, 2], U32),
            DevOp::ArgmaxF32 => {
                set(&[0], U32);
                set(&[1], F32);
            }
            DevOp::RandF32 => {
                set(&[0], F32);
                set(&[1], U64);
            }
            DevOp::RopeAxialF32 => {
                set(&[0], F32);
                set(&[1], U32);
                read_output = true;
            }
            DevOp::DenseGemmF32 => {
                set(&[0, 1, 3, 5, 6, 7], F32);
                set(&[2], if d.i[7] & 4 != 0 { Bf16 } else { F32 });
                if d.i[7] & 128 != 0 {
                    types[5] = None;
                }
            }
            DevOp::AttentionF32 => {
                if crate::speech_flag_layout::is_ring_attention(d.i[6]) {
                    set(&[0, 1, 2], F32);
                    set(&[3, 4, 5], U32);
                } else if crate::speech_flag_layout::half_attention(d.op, d.fj[2]) {
                    set(&[0, 1, 2, 3], F16);
                    set(&[4], U32);
                } else {
                    set(&[0, 1, 2, 3, 6], F32);
                    set(&[4, 7], U32);
                    if d.t[6] != TENSOR_NONE16 && d.i[6] & 32 != 0 {
                        set(&[5], U32);
                    } else if d.i[6] & 8 == 0 {
                        set(&[5], F32);
                    }
                    if d.i[6] & 16 != 0 {
                        types[6] = None;
                    }
                }
            }
            _ => return Err(format!("native types: unaudited opcode or variant {op:?}")),
        }
        for (slot, &handle) in d.t.iter().enumerate() {
            if handle == TENSOR_NONE16 {
                continue;
            }
            let id = handle as usize;
            if id >= self.parent.len() {
                return Err("native types: operand out of range".into());
            }
            let access = &mut self.accesses[id];
            let write = writes.contains(&slot);
            access.0 |= !write || (slot == 0 && read_output);
            access.1 |= write;
            let root = self.root(id);
            if copy && slot < 2 {
                self.allowed[root] &= bit(F32) | bit(U32);
            } else if let Some(dtype) = types[slot] {
                self.allowed[root] &= bit(dtype);
            } else {
                self.untyped[root] = true;
            }
        }
        if copy {
            if d.t[..2].contains(&TENSOR_NONE16) {
                return Err("native types: raw copy lacks operands".into());
            }
            let a = self.root(d.t[0] as usize);
            let b = self.root(d.t[1] as usize);
            if a != b {
                let allowed = self.allowed[b];
                let untyped = self.untyped[b];
                self.parent[b] = a;
                self.allowed[a] &= allowed;
                self.untyped[a] |= untyped;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Builder, Model};

    fn model(conflict: bool, ambiguous: bool) -> Model {
        let mut b = Builder::new(1);
        b.force_uniseg();
        let codes = b.tensor("codes", 16);
        let level = b.tensor("level", 16);
        let table = b.tensor("table", 64);
        let output = b.tensor("out", 64);
        b.emit(DevOp::CopyColsF32, vec![0], &[], |d| {
            d.t[0] = level;
            d.t[1] = codes;
            d.i[..7].copy_from_slice(&[1, 4, 1, 1, 0, 1, 0]);
            d.j = [4, 4];
        });
        if !ambiguous {
            b.emit(DevOp::GatherRowsF32, vec![0], &[], |d| {
                d.t[0] = output;
                d.t[1] = table;
                d.t[2] = level;
                d.i[..3].copy_from_slice(&[4, 4, 4]);
            });
        }
        if conflict {
            b.emit(DevOp::UnaryF32, vec![0], &[], |d| {
                d.t[0] = level;
                d.t[1] = codes;
                d.i[..2].copy_from_slice(&[4, 1]);
            });
        }
        Model {
            n_cu: 1,
            target: 0,
            tensors: b.tensors(),
            progs: vec![b.finish()],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        }
    }

    #[test]
    fn native_packet_types_propagate_integer_indices_through_raw_copies() {
        crate::program::with_model(&model(false, false), |packet| {
            let inferred = NativePacketTypes::infer(packet, 0).unwrap();
            assert_eq!(inferred.dtype(0).unwrap(), PipelineDType::U32);
            assert_eq!(inferred.dtype(1).unwrap(), PipelineDType::U32);
            assert_eq!(inferred.dtype(3).unwrap(), PipelineDType::F32);
            assert_eq!(inferred.access(0), Some(HandoffAccessMode::Read));
            assert_eq!(inferred.access(1), Some(HandoffAccessMode::ReadWrite));
            assert_eq!(inferred.access(3), Some(HandoffAccessMode::Write));
        });
    }

    #[test]
    fn native_packet_types_read_half_short_attention_as_f16() {
        for half in [false, true] {
            let mut b = Builder::new(1);
            b.force_uniseg();
            let out = b.tensor("out", 256);
            let qkv = b.tensor("qkv", 768);
            b.emit(DevOp::AttentionF32, vec![0], &[], |d| {
                d.t[..4].copy_from_slice(&[out, qkv, qkv, qkv]);
                d.i[..6].copy_from_slice(&[1, 2, 2, 2, 64, 384]);
                d.f[0] = 0.125;
                d.j = [128, 256 | u32::from(half) << 31];
            });
            let m = Model {
                n_cu: 1,
                target: 0,
                tensors: b.tensors(),
                progs: vec![b.finish()],
                prog_t: vec![1],
                kv_row_insts: vec![],
                gen: vec![],
            };
            let want = if half {
                PipelineDType::F16
            } else {
                PipelineDType::F32
            };
            crate::program::with_model(&m, |packet| {
                let inferred = NativePacketTypes::infer(packet, 0).unwrap();
                assert_eq!(inferred.dtype(0).unwrap(), want);
                assert_eq!(inferred.dtype(1).unwrap(), want);
            });
        }
    }

    #[test]
    fn native_packet_types_refuse_ambiguous_conflicting_and_unknown_uses() {
        for (conflict, ambiguous) in [(true, false), (false, true)] {
            crate::program::with_model(&model(conflict, ambiguous), |packet| {
                assert!(NativePacketTypes::infer(packet, 0)
                    .unwrap()
                    .dtype(0)
                    .is_err());
            });
        }
        let mut unknown = model(false, false);
        unknown.progs[0].insts[0].op = 65534;
        crate::program::with_model(&unknown, |packet| {
            assert!(NativePacketTypes::infer(packet, 0).is_err());
        });
    }
}
