//! Turn a wire [`DevInst64`] into named operands.
//!
//! This is the format-agnostic half of the disassembler: it produces a
//! structure, and the caller renders it as text or JSON. It lives in `packet`
//! because that is where [`DevOp`] and the slot table are — and `packet` has no
//! dependencies, which is why serialization is deliberately *not* here.
//!
//! # The raw view is not optional
//!
//! [`Inst::raw`] is always populated. The named view is an interpretation of the
//! bytes via [`crate::slots`], and an interpretation can be wrong — the table's
//! own first test found [`DevOp::FlashPrefill`]'s doc spec stale by six
//! operands. A renderer that drops `raw` leaves a reader with no way to see
//! that; one that keeps it turns a wrong name into an obvious discrepancy
//! rather than a silent one.

use crate::dev::{DevInst64, DevOp, TENSOR_NONE16};
use crate::slots::{slots_for, Provenance};

/// A resolved tensor operand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorOperand<'a> {
    /// Wire position, `0..8`.
    pub slot: usize,
    /// Name from the slot table, or `None` when the op has no spec.
    pub name: Option<&'static str>,
    /// `None` when the slot holds [`TENSOR_NONE16`].
    pub handle: Option<u16>,
    /// Resolved against the blob's tensor table, when one was supplied.
    pub tensor: Option<&'a str>,
    /// The op tolerates an absent operand here.
    pub optional: bool,
}

/// A resolved integer operand (`i0..i7`, or the integer half of the overlay).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntOperand {
    pub slot: usize,
    pub name: Option<&'static str>,
    pub value: u32,
}

/// A resolved `f32` operand.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatOperand {
    pub slot: usize,
    pub name: Option<&'static str>,
    pub value: f32,
}

/// The wire bytes, verbatim. See the module note.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raw {
    pub op: u16,
    pub blocks: u16,
    pub t: [u16; 8],
    pub i: [u32; 8],
    pub fj: [u32; 3],
}

/// One disassembled instruction.
#[derive(Clone, Debug, PartialEq)]
pub struct Inst<'a> {
    /// Index within the program's instruction array.
    pub idx: usize,
    /// `None` for a discriminant no [`DevOp`] claims — a corrupt or
    /// newer-than-this-build blob.
    pub op: Option<DevOp>,
    pub op_name: Option<&'static str>,
    pub blocks: u16,
    /// Only slots the op actually uses; absent optional operands are included
    /// (with `handle: None`) because "this op could take a `gamma` and did not"
    /// is information.
    pub tensors: Vec<TensorOperand<'a>>,
    pub ints: Vec<IntOperand>,
    pub floats: Vec<FloatOperand>,
    pub provenance: Provenance,
    pub raw: Raw,
}

/// Disassemble one instruction.
///
/// `names` resolves tensor handles, indexed by handle — exactly the shape of the
/// blob's tensor table, so a caller passes what it already has instead of
/// building a map. Pass `&[]` to leave handles unresolved.
///
/// The slice's own lifetime is deliberately unconstrained: only the `&'a str`
/// elements outlive the call, so a caller may build the index in a local `Vec`
/// whose contents borrow the blob. `idx` is carried through only so renderers
/// can label the line.
pub fn disasm<'a>(idx: usize, w: &DevInst64, names: &[&'a str]) -> Inst<'a> {
    let op = DevOp::from_u16(w.op);
    let slots = op.map(slots_for);

    // An op with a spec reports exactly its spec'd slots. An op WITHOUT one
    // reports every slot, because there is nothing better to go on.
    //
    // The tempting middle ground — "also show unnamed slots that look
    // populated" — does not work: an emitter fills only the slots it uses and
    // `DevInst::default()` leaves the rest at `0`, which is a perfectly valid
    // tensor handle. So an unused slot is indistinguishable from a reference to
    // tensor 0, and printing the difference would be inventing it. `raw` carries
    // all eight either way.
    let known = slots
        .as_ref()
        .is_some_and(|s| s.provenance != Provenance::Undocumented);

    let mut tensors = Vec::new();
    for k in 0..8 {
        let spec = slots.as_ref().and_then(|s| s.t[k]);
        if known && spec.is_none() {
            continue;
        }
        let present = w.t[k] != TENSOR_NONE16;
        tensors.push(TensorOperand {
            slot: k,
            name: spec.map(|s| s.name),
            handle: present.then_some(w.t[k]),
            tensor: present
                .then(|| names.get(w.t[k] as usize).copied())
                .flatten(),
            optional: spec.map(|s| s.optional).unwrap_or(false),
        });
    }

    let mut ints = Vec::new();
    for k in 0..8 {
        let name = slots.as_ref().and_then(|s| s.i[k]);
        if known && name.is_none() {
            continue;
        }
        ints.push(IntOperand {
            slot: k,
            name,
            value: w.i[k],
        });
    }

    let mut floats = Vec::new();
    let f0 = slots.as_ref().and_then(|s| s.f0);
    if f0.is_some() || !known {
        floats.push(FloatOperand {
            slot: 0,
            name: f0,
            value: f32::from_bits(w.fj[0]),
        });
    }

    // The overlay. `fj[1]` is `f1` or `j0` and never both (asserted in
    // `DevInst::pack`); which of the two is a property of the OP, not of the
    // bytes — nothing in the word distinguishes them. For an op with no spec
    // there is no answer, so it is reported as the integer and `raw` keeps the
    // bit pattern for anyone who needs the other reading.
    let fj1_float = slots.as_ref().is_some_and(|s| s.fj1_is_float());
    let j0 = slots.as_ref().and_then(|s| s.j0);
    if fj1_float {
        let name = slots.as_ref().and_then(|s| s.f1);
        floats.push(FloatOperand {
            slot: 1,
            name,
            value: f32::from_bits(w.fj[1]),
        });
    } else if j0.is_some() || !known {
        ints.push(IntOperand {
            slot: 8,
            name: j0,
            value: w.fj[1],
        });
    }

    let j1 = slots.as_ref().and_then(|s| s.j1);
    if j1.is_some() || !known {
        ints.push(IntOperand {
            slot: 9,
            name: j1,
            value: w.fj[2],
        });
    }

    Inst {
        idx,
        op,
        op_name: op.map(op_name),
        blocks: w.blocks,
        tensors,
        ints,
        floats,
        provenance: slots
            .map(|s| s.provenance)
            .unwrap_or(Provenance::Undocumented),
        raw: Raw {
            op: w.op,
            blocks: w.blocks,
            t: w.t,
            i: w.i,
            fj: w.fj,
        },
    }
}

/// The Rust spelling of an opcode, e.g. `"GemmNorm"`.
///
/// From the `Debug` derive rather than a second hand-maintained table — a name
/// table that can disagree with the enum is the class of bug `slots` exists to
/// prevent, so it is not reintroduced here. Leaked once per opcode, bounded by
/// [`DevOp::ALL`]; callers are cold-path tooling.
pub fn op_name(op: DevOp) -> &'static str {
    use std::sync::OnceLock;
    static NAMES: OnceLock<Vec<(DevOp, &'static str)>> = OnceLock::new();
    NAMES
        .get_or_init(|| {
            DevOp::ALL
                .iter()
                .map(|o| (*o, &*Box::leak(format!("{o:?}").into_boxed_str())))
                .collect()
        })
        .iter()
        .find(|(o, _)| *o == op)
        .map(|(_, n)| *n)
        .unwrap_or("?")
}

/// Name for a WIRE opcode, falling back to `op<n>` for a discriminant no
/// variant claims — a corrupt blob, or one built by a newer compiler.
///
/// The fallback is why this returns `String` rather than `&'static str`: an
/// unknown opcode has no static name, and inventing one ("Nop", "?") would hide
/// exactly the case worth seeing.
pub fn op_label(op: u16) -> String {
    DevOp::from_u16(op)
        .map(|o| op_name(o).to_string())
        .unwrap_or_else(|| format!("op{op}"))
}

/// Label for an int slot: `i0..i7`, then the two overlay words.
pub fn int_slot_label(slot: usize) -> &'static str {
    const L: [&str; 10] = ["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "j0", "j1"];
    L.get(slot).copied().unwrap_or("i?")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::{DevInst, TENSOR_NONE};

    fn inst(op: DevOp) -> DevInst {
        DevInst {
            op: op as u16,
            ..Default::default()
        }
    }

    #[test]
    fn names_operands_from_the_slot_table() {
        let mut d = inst(DevOp::Gemm);
        // Unused slots carry the builder's TENSOR_NONE, not 0: t7 is Gemm's optional bias.
        d.t = [3, 4, 5, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE];
        d.i = [1, 6144, 2048, 0, 0, 0, 0, 0];
        let names = ["w0", "w1", "w2", "act.hn", "act.x", "blk.q.w"];
        let out = disasm(7, &d.pack(), &names);

        assert_eq!(out.op, Some(DevOp::Gemm));
        assert_eq!(out.op_name, Some("Gemm"));
        assert_eq!(out.idx, 7);

        let t: Vec<_> = out.tensors.iter().map(|o| (o.name, o.tensor)).collect();
        assert_eq!(
            t,
            [
                (Some("C"), Some("act.hn")),
                (Some("A"), Some("act.x")),
                (Some("B"), Some("blk.q.w")),
                (Some("bias"), None),
            ]
        );
        let i: Vec<_> = out.ints.iter().map(|o| (o.name, o.value)).collect();
        assert_eq!(i, [(Some("M"), 1), (Some("N"), 6144), (Some("K"), 2048)]);
    }

    /// `t[k] == TENSOR_NONE16` in a documented optional slot is "absent", not
    /// the handle 65535.
    #[test]
    fn absent_optional_operand_is_not_handle_65535() {
        let mut d = inst(DevOp::RmsNorm);
        d.t = [1, 2, crate::dev::TENSOR_NONE, 0, 0, 0, 0, 0];
        let out = disasm(0, &d.pack(), &["a", "out", "x"]);

        let gamma = out
            .tensors
            .iter()
            .find(|o| o.name == Some("gamma"))
            .unwrap();
        assert_eq!(gamma.handle, None);
        assert!(gamma.optional);
        assert_eq!(gamma.tensor, None);
        assert_ne!(out.raw.t[2], 0, "raw still carries the sentinel");
    }

    /// The overlay, both ways round. This is the case the module header calls
    /// the likeliest source of a plausible-but-wrong dump.
    #[test]
    fn fj1_reads_as_float_or_int_per_op() {
        // NormResidual: `f0=eps f1=scale`.
        let mut d = inst(DevOp::NormResidual);
        d.f = [1e-5, 2.0];
        let out = disasm(0, &d.pack(), &[]);
        let f: Vec<_> = out.floats.iter().map(|o| (o.name, o.value)).collect();
        assert_eq!(f, [(Some("eps"), 1e-5), (Some("scale"), 2.0)]);
        assert!(
            out.ints.iter().all(|o| o.slot < 8),
            "no j operand on a float op"
        );

        // FlashPrefill: fj[1] is j0 = kv_stride. Same bit pattern, different read.
        let mut d = inst(DevOp::FlashPrefill);
        d.f = [1.0, 0.0];
        d.j = [4096, 7];
        let out = disasm(0, &d.pack(), &[]);
        let j: Vec<_> = out
            .ints
            .iter()
            .filter(|o| o.slot >= 8)
            .map(|o| (o.name, o.value))
            .collect();
        assert_eq!(j, [(Some("kv_stride"), 4096), (Some("kv_mask"), 7)]);
        assert_eq!(out.floats.len(), 1, "only f0 is a float here");
    }

    /// An op with no spec still disassembles — raw only, and flagged as such.
    #[test]
    fn undocumented_op_degrades_to_raw() {
        let mut d = inst(DevOp::Mamba2Scan);
        d.t = [9, 0, 0, 0, 0, 0, 0, 0];
        d.i = [42, 0, 0, 0, 0, 0, 0, 0];
        let out = disasm(
            0,
            &d.pack(),
            &["a", "b", "c", "d", "e", "f", "g", "h", "i", "st.x"],
        );

        assert_eq!(out.provenance, Provenance::Undocumented);
        assert!(out.tensors.iter().all(|o| o.name.is_none()));
        // Populated-but-unnamed slots are still surfaced, resolved where possible.
        assert_eq!(out.tensors[0].tensor, Some("st.x"));
        assert_eq!(out.ints[0].value, 42);
    }

    #[test]
    fn unknown_opcode_is_reported_not_guessed() {
        let w = DevInst64 {
            op: 60000,
            blocks: 1,
            fj: [0; 3],
            t: [TENSOR_NONE16; 8],
            i: [0; 8],
        };
        let out = disasm(0, &w, &[]);
        assert_eq!(out.op, None);
        assert_eq!(out.op_name, None);
        assert_eq!(out.raw.op, 60000);
    }

    /// Nothing in the named view may lose a bit of the wire instruction.
    #[test]
    fn raw_round_trips_every_opcode() {
        for (n, op) in DevOp::ALL.iter().enumerate() {
            let mut d = inst(*op);
            d.blocks = (n as u16) + 1;
            d.t = [0, 1, 2, 3, 4, 5, 6, 7];
            d.i = [n as u32; 8];
            d.f[0] = 0.5;
            let w = d.pack();
            let out = disasm(n, &w, &[]);
            assert_eq!(out.raw.op, w.op, "{op:?}");
            assert_eq!(out.raw.blocks, w.blocks, "{op:?}");
            assert_eq!(out.raw.t, w.t, "{op:?}");
            assert_eq!(out.raw.i, w.i, "{op:?}");
            assert_eq!(out.raw.fj, w.fj, "{op:?}");
        }
    }

    #[test]
    fn op_name_covers_every_opcode() {
        for op in DevOp::ALL {
            assert_ne!(op_name(*op), "?", "{op:?} has no name");
        }
    }
}
