//! Interpreter arms: the compiled instantiations an op dispatches to, by axis.
//!
//! The emit-side twin of the object contract's class 5 (docs/arch/16-object-contract.md): an
//! `hd` (or state width, or conv kernel) with no arm traps on NVIDIA and silently writes nothing
//! on AMD. An emitter component states the geometry it is about to emit and [`require`]s each
//! axis against the target's table, so a model the interpreter cannot run is refused at emit time
//! with the op and the arm it lacks, rather than by an assert written next to the emitter.
//!
//! A row lists what the arm *implements*; several rows on one `(op, axis)` must all hold.
use core::fmt;

use packet::dev::DevOp;

/// The values an arm covers on one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Have {
    OneOf(&'static [u32]),
    MultipleOf(u32),
    AtMost(u32),
    Within(u32, u32),
    Named(&'static [&'static str]),
}

impl Have {
    fn covers(&self, v: u32) -> bool {
        match *self {
            Have::OneOf(set) => set.contains(&v),
            Have::MultipleOf(m) => v % m == 0,
            Have::AtMost(m) => v <= m,
            Have::Within(lo, hi) => (lo..=hi).contains(&v),
            Have::Named(_) => false,
        }
    }
}

impl fmt::Display for Have {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Have::OneOf(set) => write!(f, "one of {set:?}"),
            Have::MultipleOf(m) => write!(f, "a multiple of {m}"),
            Have::AtMost(m) => write!(f, "at most {m}"),
            Have::Within(lo, hi) => write!(f, "{lo}..={hi}"),
            Have::Named(names) => write!(f, "one of {names:?}"),
        }
    }
}

/// One declared arm row.
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub op: DevOp,
    pub axis: &'static str,
    pub have: Have,
}

/// An emit the target cannot run: `op` has no arm for `value` on `axis`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingArm {
    pub target: &'static str,
    pub op: DevOp,
    pub axis: &'static str,
    pub value: String,
    pub have: String,
}

impl fmt::Display for MissingArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:?} has no arm for {} = {} (implemented: {})",
            self.target, self.op, self.axis, self.value, self.have
        )
    }
}

impl std::error::Error for MissingArm {}

/// A target's arm table.
#[derive(Clone, Copy, Debug)]
pub struct Arms {
    pub target: &'static str,
    pub rows: &'static [Arm],
}

impl Arms {
    fn rows<'a>(&'a self, op: DevOp, axis: &'a str) -> impl Iterator<Item = &'a Arm> + 'a {
        let rows: Vec<&Arm> = self
            .rows
            .iter()
            .filter(|a| a.op == op && a.axis == axis)
            .collect();
        assert!(
            !rows.is_empty(),
            "{}: no arm row declares {op:?} / {axis}",
            self.target
        );
        rows.into_iter()
    }

    fn missing(&self, op: DevOp, axis: &'static str, value: String, have: &Have) -> MissingArm {
        MissingArm {
            target: self.target,
            op,
            axis,
            value,
            have: have.to_string(),
        }
    }

    /// `value` on `axis` is covered by every row `op` declares for it.
    pub fn require(&self, op: DevOp, axis: &'static str, value: u32) -> Result<(), MissingArm> {
        match self.rows(op, axis).find(|a| !a.have.covers(value)) {
            Some(a) => Err(self.missing(op, axis, value.to_string(), &a.have)),
            None => Ok(()),
        }
    }

    /// The named variant `value` (an activation, a layout) on `axis`.
    pub fn require_named(
        &self,
        op: DevOp,
        axis: &'static str,
        value: &str,
    ) -> Result<(), MissingArm> {
        let ok = |a: &&Arm| matches!(a.have, Have::Named(names) if names.contains(&value));
        match self.rows(op, axis).find(|a| !ok(a)) {
            Some(a) => Err(self.missing(op, axis, value.to_string(), &a.have)),
            None => Ok(()),
        }
    }

    /// A feature the arm has (`true`) or lacks (`false`), as the 0/1 axis `axis`.
    pub fn require_flag(
        &self,
        op: DevOp,
        axis: &'static str,
        present: bool,
    ) -> Result<(), MissingArm> {
        self.require(op, axis, u32::from(present))
    }
}

/// The sm_120 interpreter (sm_120a, sm_121a): the Mamba-2, NVFP4 latent MoE, per-tensor FP8 and
/// hd128 attention arms the Nemotron-H components emit (runtime/nvidia/op_mamba.cuh,
/// op_moe_nvfp4.cuh, op_flash_pf128.cuh, op_attention.cuh).
#[rustfmt::skip]
pub const SM120: Arms = Arms {
    target: "sm_120 interpreter",
    rows: &[
        // op_mamba.cuh MAMBA_P / MAMBA_N / MAMBA_L; the conv is width 4 with its bias.
        Arm { op: DevOp::Mamba2Step, axis: "head_dim", have: Have::OneOf(&[64]) },
        Arm { op: DevOp::Mamba2Step, axis: "state", have: Have::OneOf(&[128]) },
        Arm { op: DevOp::Mamba2ChunkState, axis: "chunk", have: Have::OneOf(&[128]) },
        Arm { op: DevOp::Mamba2Conv, axis: "kernel", have: Have::OneOf(&[4]) },
        Arm { op: DevOp::Mamba2Conv, axis: "bias", have: Have::OneOf(&[1]) },
        // `j1` sources are `pitch | column << 16`.
        Arm { op: DevOp::Mamba2Conv, axis: "source pitch", have: Have::AtMost(0xFFFF) },
        Arm { op: DevOp::Mamba2GatedNorm, axis: "group", have: Have::MultipleOf(256) },
        Arm { op: DevOp::Mamba2GatedNorm, axis: "group", have: Have::AtMost(2048) },
        // The slot state is unsharded and the experts unpartitioned: no TP arm.
        Arm { op: DevOp::Mamba2Step, axis: "tensor parallel", have: Have::OneOf(&[1]) },
        Arm { op: DevOp::MoeUpNvfp4, axis: "tensor parallel", have: Have::OneOf(&[1]) },
        // op_moe_nvfp4.cuh: relu^2 experts without bias, decode rows <= NF4_DEC_MAXROWS.
        Arm { op: DevOp::MoeUpNvfp4, axis: "activation", have: Have::Named(&["relu2"]) },
        Arm { op: DevOp::MoeUpNvfp4, axis: "bias", have: Have::OneOf(&[0]) },
        Arm { op: DevOp::MoeUpNvfp4, axis: "decode rows", have: Have::AtMost(64) },
        // Router tiles: E over 128-expert blocks, k <= 32, one group.
        Arm { op: DevOp::MoeRouterSigmoidTopk, axis: "experts", have: Have::MultipleOf(128) },
        Arm { op: DevOp::MoeRouterSigmoidTopk, axis: "experts", have: Have::AtMost(1024) },
        Arm { op: DevOp::MoeRouterSigmoidTopk, axis: "top-k", have: Have::Within(1, 32) },
        Arm { op: DevOp::MoeRouterSigmoidTopk, axis: "groups", have: Have::OneOf(&[1]) },
        // Native per-row W8A8 prefill GEMM: 128-wide N and K tiles.
        Arm { op: DevOp::GemmFp8, axis: "N", have: Have::MultipleOf(128) },
        Arm { op: DevOp::GemmFp8, axis: "K", have: Have::MultipleOf(128) },
        // The fused one-split prefill epilogue (op_flash_pf128.cuh) and the GQA-16 decode item.
        Arm { op: DevOp::FlashPrefill, axis: "hd", have: Have::OneOf(&[128]) },
        Arm { op: DevOp::FlashDecode, axis: "hd", have: Have::OneOf(&[128]) },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_names_the_op_axis_and_arm() {
        assert!(SM120.require(DevOp::Mamba2Step, "head_dim", 64).is_ok());
        let e = SM120
            .require(DevOp::Mamba2Step, "head_dim", 80)
            .unwrap_err();
        assert_eq!(
            e.to_string(),
            "sm_120 interpreter Mamba2Step has no arm for head_dim = 80 (implemented: one of [64])"
        );
        // Every row of an axis must hold.
        assert!(SM120.require(DevOp::Mamba2GatedNorm, "group", 2048).is_ok());
        assert!(SM120
            .require(DevOp::Mamba2GatedNorm, "group", 2304)
            .is_err());
        assert!(SM120.require(DevOp::Mamba2GatedNorm, "group", 384).is_err());
        assert!(SM120
            .require_named(DevOp::MoeUpNvfp4, "activation", "relu2")
            .is_ok());
        assert!(SM120
            .require_named(DevOp::MoeUpNvfp4, "activation", "silu")
            .is_err());
        assert!(SM120
            .require_flag(DevOp::Mamba2Conv, "bias", false)
            .is_err());
    }

    #[test]
    #[should_panic(expected = "no arm row declares")]
    fn an_undeclared_axis_is_a_table_bug() {
        let _ = SM120.require(DevOp::Mamba2Step, "kernel", 4);
    }
}
