//! SPEECH FLAG LAYOUT: which meaning a packet's speech-op flag bits carry.
//!
//! The speech ops' flag words gained fields after the first released layout (layout 0), and one
//! bit changed meaning between branches before it settled. A packet whose instructions set any
//! bit assigned since layout 0 ([`protected`]) declares the layout they were emitted for in
//! `abi.json` (`speech_flag_layout`); the speech interpreter object exports the layout it was
//! compiled for ([`OBJECT_SYMBOL`]). A loader runs the instructions only when the two agree.
//! A packet that sets none of those bits means the same under every layout: it stays undeclared
//! (and byte-identical to an emit from before this module) and loads on any object.
//!
//! Reassigning a speech-op flag bit bumps [`LAYOUT`] and extends the masks.
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

/// The layout this tree emits and compiles: conv flags bit 2 SnakeBeta output, bit 3 ragged
/// wgmma tiles, 21..29 pinned arms, 30 / 31 split-image input / output; AttentionF32 `i6` bit 6
/// query view, bit 7 split-image output; LayerNormF32 `i2` bit 2 split-image output; GemvF32Out
/// `i7` bit 12 row view. Layout 2 adds LayerNormF32 bit 3 (f16 output) and GemvF32Out
/// bits 13 / 14 (f16 input / output). Layout 3 adds slot-indexed ring attention and tail modes.
pub const LAYOUT: u32 = 3;
pub const DEFAULT_LAYOUT: u32 = 2;
/// `extern "C" __device__ unsigned` exported by every speech interpreter object.
pub const OBJECT_SYMBOL: &str = "plow_speech_flag_layout_speech";

/// Conv1dF32 / ConvTranspose1dF32 flags (`j1`, wire `fj[2]`) assigned since layout 0.
pub const CONV_FLAGS: u32 = 0xffe0_000c;
/// AttentionF32 `i6` bits assigned since layout 0.
pub const ATTENTION_I6: u32 = 0xc0;
/// LayerNormF32 `i2` bits assigned since layout 0.
pub const LAYER_NORM_I2: u32 = 0xc;
/// GemvF32Out `i7` bits assigned since layout 0.
pub const GEMV_F32_OUT_I7: u32 = 0x7000;

pub fn is_ring_attention(flags: u32) -> bool {
    flags & 0xff08 == 0x108
}

pub fn ring_kv(op: u16, i: &[u32; 8]) -> bool {
    match DevOp::from_u16(op) {
        Some(DevOp::AttentionF32) => is_ring_attention(i[6]),
        Some(DevOp::StreamTailF32) => matches!(i[7], 2 | 3),
        _ => false,
    }
}

/// Whether an instruction (`j1` = `DevInst::j[1]` = `DevInst64::fj[2]`) sets a flag bit that
/// layout 0 left clear.
pub fn protected(op: u16, j1: u32, i: &[u32; 8]) -> bool {
    match DevOp::from_u16(op) {
        Some(DevOp::Conv1dF32 | DevOp::ConvTranspose1dF32) => j1 & CONV_FLAGS != 0,
        Some(DevOp::AttentionF32) => i[6] & ATTENTION_I6 != 0 || ring_kv(op, i),
        Some(DevOp::StreamTailF32) => ring_kv(op, i),
        Some(DevOp::LayerNormF32) => i[2] & LAYER_NORM_I2 != 0,
        Some(DevOp::GemvF32Out) => i[7] & GEMV_F32_OUT_I7 != 0,
        _ => false,
    }
}

/// The opcode of the first instruction of `insts` (`(op, j1, i)`) that sets a protected bit.
pub fn first_protected(insts: impl IntoIterator<Item = (u16, u32, [u32; 8])>) -> Option<u16> {
    insts
        .into_iter()
        .find(|(op, j1, i)| protected(*op, *j1, i))
        .map(|(op, ..)| op)
}

pub fn required_layout(insts: impl IntoIterator<Item = (u16, u32, [u32; 8])>) -> Option<u32> {
    insts
        .into_iter()
        .filter_map(|(op, j1, i)| {
            protected(op, j1, &i).then(|| {
                if ring_kv(op, &i) {
                    3
                } else if (op == DevOp::LayerNormF32 as u16 && i[2] & 8 != 0)
                    || (op == DevOp::GemvF32Out as u16 && i[7] & 0x6000 != 0)
                {
                    2
                } else {
                    1
                }
            })
        })
        .max()
}

/// Whether a metadata section holds auxiliary programs (`aux_program` wire form).
pub fn is_aux_programs(section: &[u8]) -> bool {
    section.starts_with(crate::aux_program::MAGIC)
}

/// The instructions of an auxiliary-program section, for the scan only: tensor handles are not
/// checked against the packet (each consumer parses its own section with its tensor count).
pub fn aux_insts(section: &[u8], n_cu: u32) -> Result<Vec<DevInst64>, String> {
    let parsed = crate::aux_program::parse(section, n_cu, TENSOR_NONE16 as usize)?;
    Ok(parsed.programs.into_iter().flat_map(|p| p.insts).collect())
}

/// Refuses auxiliary programs bound for an object that is not a speech object (the decode
/// objects read no flag bit assigned since layout 0) when one of them sets a protected bit.
pub fn refuse_on_decode_objects<'a>(
    insts: impl IntoIterator<Item = &'a DevInst64>,
) -> Result<(), String> {
    match first_protected(insts.into_iter().map(|d| (d.op, d.fj[2], d.i))) {
        None => Ok(()),
        Some(op) => Err(format!(
            "opcode {op} sets a speech flag bit the decode objects do not read"
        )),
    }
}

/// The baseline object layout; ring programs use [`required_layout`].
pub fn declare(used: Option<u16>) -> Option<u32> {
    used.map(|_| DEFAULT_LAYOUT)
}

/// Whether the instructions an object runs, whose first protected use is `used`, from a packet
/// declaring `declared`, may run on that object, which exports `object` (`None` = no symbol).
/// Instructions with no protected bit read the same under every layout and run on any object.
pub fn check(declared: Option<u32>, used: Option<u16>, object: Option<u32>) -> Result<(), String> {
    if let Some(layout) = declared.filter(|&l| l == 0 || l > LAYOUT) {
        return Err(format!(
            "the packet declares speech flag layout {layout}, this runtime reads {LAYOUT}"
        ));
    }
    let Some(op) = used else {
        return Ok(());
    };
    let Some(layout) = declared else {
        return Err(format!(
            "opcode {op} sets a speech flag bit assigned after the packet's undeclared flag \
             layout: re-emit the packet"
        ));
    };
    match object {
        Some(have) if have >= layout && have <= LAYOUT => Ok(()),
        Some(have) => Err(format!(
            "opcode {op} needs speech flag layout {layout}, the object is compiled for {have}: \
             rebuild the object for this packet"
        )),
        None => Err(format!(
            "opcode {op} needs speech flag layout {layout}, the object exports none: it is not a \
             speech object of this layout"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: DevOp, j1: u32, slot: usize, v: u32) -> (u16, u32, [u32; 8]) {
        let mut i = [0; 8];
        i[slot] = v;
        (op as u16, j1, i)
    }

    #[test]
    fn ring_modes_need_layout_three_without_reassigning_old_fields() {
        assert_eq!(
            required_layout([inst(DevOp::AttentionF32, 0, 6, 9 | (250 << 16))]),
            None
        );
        assert_eq!(
            required_layout([inst(DevOp::AttentionF32, 0, 6, 128)]),
            Some(1)
        );
        assert_eq!(
            required_layout([inst(DevOp::AttentionF32, 0, 6, 0x109 | (250 << 16))]),
            Some(3)
        );
        for mode in [0, 1] {
            assert_eq!(
                required_layout([inst(DevOp::StreamTailF32, 0, 7, mode)]),
                None
            );
        }
        for mode in [2, 3] {
            let i = inst(DevOp::StreamTailF32, 0, 7, mode);
            assert_eq!(required_layout([i]), Some(3));
            assert!(check(Some(3), first_protected([i]), Some(2)).is_err());
            assert!(check(Some(3), first_protected([i]), Some(3)).is_ok());
        }
        assert_eq!(declare(Some(DevOp::AttentionF32 as u16)), Some(2));
    }

    /// Every bit the layout-0 (RC9, 45e1da259) emitters could set stays unprotected. The values
    /// are those emitters' fields, not the masks' complements: conv pad 0..1, activations 4..11,
    /// routes 12..20; attention options 0..5, key-length groups 8..15, window 16..31; LayerNorm
    /// rounding / statistics 0..1 and activation from bit 4; GemvF32Out modes through bit 11.
    #[test]
    fn layout_zero_bits_are_not_protected() {
        for op in [DevOp::Conv1dF32, DevOp::ConvTranspose1dF32] {
            assert_eq!(first_protected([inst(op, 0x001f_fff3, 0, 0)]), None);
        }
        assert_eq!(
            first_protected([inst(DevOp::AttentionF32, 0, 6, 0xffff_ff3f)]),
            None
        );
        assert_eq!(
            first_protected([inst(DevOp::LayerNormF32, 0, 2, 0xffff_fff3)]),
            None
        );
        assert_eq!(
            first_protected([inst(DevOp::GemvF32Out, 0, 7, 0x0fff)]),
            None
        );
        // The same bit patterns on other ops are not flags.
        assert_eq!(
            first_protected([inst(DevOp::Gemv, u32::MAX, 7, u32::MAX)]),
            None
        );
        assert_eq!(
            first_protected([inst(DevOp::UnaryF32, 1 << 3, 6, 0xc0)]),
            None
        );
    }

    #[test]
    fn each_reassigned_bit_is_protected() {
        for bit in (2..=3).chain(21..=31) {
            for op in [DevOp::Conv1dF32, DevOp::ConvTranspose1dF32] {
                assert_eq!(
                    first_protected([inst(op, 1 << bit, 0, 0)]),
                    Some(op as u16),
                    "{bit}"
                );
            }
        }
        for bit in [6, 7] {
            let op = DevOp::AttentionF32;
            assert_eq!(first_protected([inst(op, 0, 6, 1 << bit)]), Some(op as u16));
        }
        let ln = DevOp::LayerNormF32;
        for bit in [2, 3] {
            assert_eq!(first_protected([inst(ln, 0, 2, 1 << bit)]), Some(ln as u16));
        }
        let head = DevOp::GemvF32Out;
        for bit in [12, 13, 14] {
            assert_eq!(
                first_protected([inst(head, 0, 7, 1 << bit)]),
                Some(head as u16)
            );
        }
        // Found behind unprotected instructions.
        let conv = DevOp::Conv1dF32;
        assert_eq!(
            first_protected([inst(conv, 0x6e000, 0, 0), inst(head, 0, 7, 4096 | 2048)]),
            Some(head as u16)
        );
    }

    #[test]
    fn auxiliary_program_sections_are_scanned() {
        use packet::devbuild::{Builder, Model, TensorDecl};
        let section = |view: bool| {
            let mut b = Builder::new(4);
            b.force_uniseg();
            b.emit(DevOp::Nop, b.all(), &[], |_| {});
            b.emit(DevOp::GemvF32Out, b.all(), &[], |i| {
                i.t = [0; 8];
                i.i[7] = 2048 | if view { 4096 } else { 0 };
            });
            let model = Model {
                n_cu: 4,
                target: 0,
                tensors: vec![TensorDecl {
                    name: "x".into(),
                    bytes: 8,
                    init: None,
                }],
                progs: vec![b.finish()],
                kv_row_insts: vec![],
                prog_t: vec![1],
                gen: vec![],
            };
            crate::program::with_model(&model, |p| {
                crate::aux_program::encode(4, 1, &[p.programs[0]]).unwrap()
            })
        };
        let scan = |bytes: &[u8]| {
            assert!(is_aux_programs(bytes));
            first_protected(
                aux_insts(bytes, 4)
                    .unwrap()
                    .iter()
                    .map(|d| (d.op, d.fj[2], d.i)),
            )
        };
        assert_eq!(scan(&section(false)), None);
        assert_eq!(scan(&section(true)), Some(DevOp::GemvF32Out as u16));
        assert!(!is_aux_programs(b"{\"version\":1}"));
        assert!(aux_insts(&section(true), 5).is_err());
    }

    #[test]
    fn decode_objects_refuse_protected_auxiliary_instructions() {
        let wire = |i7: u32| {
            let mut d = DevInst64 {
                op: DevOp::GemvF32Out as u16,
                ..Default::default()
            };
            d.i[7] = i7;
            d
        };
        assert!(refuse_on_decode_objects(&[wire(0), wire(2048)]).is_ok());
        let e = refuse_on_decode_objects(&[wire(2048), wire(2048 | 4096)]).unwrap_err();
        assert!(e.contains("do not read"), "{e}");
    }

    #[test]
    fn undeclared_packets_load_only_without_protected_bits() {
        let conv = Some(DevOp::Conv1dF32 as u16);
        for object in [None, Some(0), Some(LAYOUT), Some(LAYOUT + 1)] {
            assert!(check(None, None, object).is_ok());
            let e = check(None, conv, object).unwrap_err();
            assert!(e.contains("re-emit"), "{e}");
        }
        assert_eq!(declare(None), None);
        assert_eq!(declare(conv), Some(DEFAULT_LAYOUT));
    }

    #[test]
    fn declared_packets_need_the_same_object_layout() {
        let conv = Some(DevOp::Conv1dF32 as u16);
        assert!(check(Some(LAYOUT), conv, Some(LAYOUT)).is_ok());
        assert!(check(Some(LAYOUT), conv, None)
            .unwrap_err()
            .contains("exports none"));
        for have in [0, 1, LAYOUT + 1] {
            let e = check(Some(LAYOUT), conv, Some(have)).unwrap_err();
            assert!(e.contains("compiled for"), "{e}");
        }
        // An object that runs none of the protected instructions (a Maya 2 packet's backbone
        // decode object, its depth programs being the speech object's) is not constrained.
        for object in [None, Some(0), Some(LAYOUT), Some(LAYOUT + 1)] {
            assert!(check(Some(LAYOUT), None, object).is_ok());
        }
        // A layout this runtime does not read is refused on every object.
        for used in [None, conv] {
            let e = check(Some(LAYOUT + 1), used, Some(LAYOUT + 1)).unwrap_err();
            assert!(e.contains("this runtime reads"), "{e}");
        }
    }

    #[test]
    fn layout_one_packets_load_on_layout_two_objects() {
        let op = Some(DevOp::GemvF32Out as u16);
        assert!(check(Some(1), op, Some(1)).is_ok());
        assert!(check(Some(1), op, Some(2)).is_ok());
        assert!(check(Some(2), op, Some(1)).is_err());
        assert_eq!(
            required_layout([inst(DevOp::GemvF32Out, 0, 7, 4096)]),
            Some(1)
        );
        assert_eq!(
            required_layout([inst(DevOp::GemvF32Out, 0, 7, 8192)]),
            Some(2)
        );
        assert_eq!(
            required_layout([inst(DevOp::LayerNormF32, 0, 2, 8)]),
            Some(2)
        );
    }
}
