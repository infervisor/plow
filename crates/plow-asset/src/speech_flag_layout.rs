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
/// `i7` bit 12 row view.
pub const LAYOUT: u32 = 1;
/// `extern "C" __device__ unsigned` exported by every speech interpreter object.
pub const OBJECT_SYMBOL: &str = "plow_speech_flag_layout_speech";

/// Conv1dF32 / ConvTranspose1dF32 flags (`j1`, wire `fj[2]`) assigned since layout 0.
pub const CONV_FLAGS: u32 = 0xffe0_000c;
/// AttentionF32 `i6` bits assigned since layout 0.
pub const ATTENTION_I6: u32 = 0xc0;
/// LayerNormF32 `i2` bits assigned since layout 0.
pub const LAYER_NORM_I2: u32 = 0x4;
/// GemvF32Out `i7` bits assigned since layout 0.
pub const GEMV_F32_OUT_I7: u32 = 0x1000;

/// Whether an instruction (`j1` = `DevInst::j[1]` = `DevInst64::fj[2]`) sets a flag bit that
/// layout 0 left clear.
pub fn protected(op: u16, j1: u32, i: &[u32; 8]) -> bool {
    match DevOp::from_u16(op) {
        Some(DevOp::Conv1dF32 | DevOp::ConvTranspose1dF32) => j1 & CONV_FLAGS != 0,
        Some(DevOp::AttentionF32) => i[6] & ATTENTION_I6 != 0,
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

/// What an emit declares: [`LAYOUT`] when an instruction sets a protected bit, else nothing.
pub fn declare(used: Option<u16>) -> Option<u32> {
    used.map(|_| LAYOUT)
}

/// Whether instructions whose first protected use is `used`, from a packet declaring `declared`,
/// may run on a speech object exporting `object` (`None` = no symbol).
pub fn check(declared: Option<u32>, used: Option<u16>, object: Option<u32>) -> Result<(), String> {
    let Some(layout) = declared else {
        return match used {
            None => Ok(()),
            Some(op) => Err(format!(
                "opcode {op} sets a speech flag bit assigned after the packet's undeclared flag \
                 layout: re-emit the packet"
            )),
        };
    };
    if layout != LAYOUT {
        return Err(format!(
            "the packet declares speech flag layout {layout}, this runtime reads {LAYOUT}"
        ));
    }
    match object {
        Some(have) if have == layout => Ok(()),
        Some(have) => Err(format!(
            "the packet declares speech flag layout {layout}, the speech object is compiled for \
             {have}: rebuild the object for this packet"
        )),
        None => Err(format!(
            "the packet declares speech flag layout {layout}, the speech object exports none: \
             rebuild the object for this packet"
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
        assert_eq!(first_protected([inst(ln, 0, 2, 4)]), Some(ln as u16));
        let head = DevOp::GemvF32Out;
        assert_eq!(first_protected([inst(head, 0, 7, 4096)]), Some(head as u16));
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
    fn undeclared_packets_load_only_without_protected_bits() {
        let conv = Some(DevOp::Conv1dF32 as u16);
        for object in [None, Some(0), Some(LAYOUT), Some(LAYOUT + 1)] {
            assert!(check(None, None, object).is_ok());
            let e = check(None, conv, object).unwrap_err();
            assert!(e.contains("re-emit"), "{e}");
        }
        assert_eq!(declare(None), None);
        assert_eq!(declare(conv), Some(LAYOUT));
    }

    #[test]
    fn declared_packets_need_the_same_object_layout() {
        let conv = Some(DevOp::Conv1dF32 as u16);
        for used in [None, conv] {
            assert!(check(Some(LAYOUT), used, Some(LAYOUT)).is_ok());
            assert!(check(Some(LAYOUT), used, None)
                .unwrap_err()
                .contains("exports none"));
            for have in [0, LAYOUT + 1, LAYOUT | 2] {
                let e = check(Some(LAYOUT), used, Some(have)).unwrap_err();
                assert!(e.contains("compiled for"), "{e}");
            }
            // A layout this runtime does not read is refused even on a matching object.
            let e = check(Some(LAYOUT + 1), used, Some(LAYOUT + 1)).unwrap_err();
            assert!(e.contains("this runtime reads"), "{e}");
        }
    }
}
