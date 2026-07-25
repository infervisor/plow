//! The vocabulary: what an operation means, and what a kernel promises.

use hwspec::{HardwareFingerprint, IsaLevel, MmaDtype};
use packet::dev::DevOp;

/// What an operation computes, independent of any kernel that implements it.
///
/// Deliberately coarser than `DevOp` and broader than `OpKind`. `OpKind`
/// (`rewrite::tilegraph`) has four physical variants — Gemm, Flash, Row, Layout
/// — which is enough to *schedule* an MoE-DSA block but not enough to say what
/// is being scheduled. Tuning needs the distinction: a grouped expert GEMM and
/// a dense projection are both `OpKind::Gemm`, and a record measured for one is
/// worthless for the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SemanticOp {
    /// Dense `out[M,N] = act[M,K] · wᵀ`.
    DenseMatmul,
    /// Per-expert matmul over a routed token permutation.
    GroupedMatmul,
    /// Softmax attention: prefill, decode, paged, sliding.
    Attention,
    /// Latent/absorbed attention (MLA).
    LatentAttention,
    /// Sparse index scoring and deterministic top-k selection (DSA).
    SparseSelect,
    /// Recurrence / selective scan (Mamba-style state update).
    Recurrence,
    /// Norms and reductions over a row.
    Reduction,
    /// Elementwise and activation.
    Elementwise,
    /// Transpose, gather/scatter, KV append/page — data movement.
    LayoutMove,
    /// Cross-device reduction/gather.
    Collective,
    /// Embedding, logits, sampling.
    Token,
}

/// Which regime the op runs in. The same math at `M=1` and `M=4096` is not the
/// same tuning problem, and a kernel is usually built for one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Decode,
    Prefill,
    /// Chunked prefill interleaved with live decode.
    Mixed,
}

/// A persistent interpreter object.
///
/// Not an abstraction over the megakernel — a name for one *build* of it. The
/// interpreter inlines every dispatch arm, so its register allocation is the
/// worst case over all inlined code; a profile is the unit within which that
/// worst case is shared. Splitting profiles is how an expensive arm is kept
/// from lowering another op's occupancy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProfileId {
    DecodeDense,
    DecodeMoe,
    DecodeLatent,
    PrefillDense,
    PrefillMoe,
    RecurrentMamba,
    PortableReference,
}

impl ProfileId {
    pub fn phase(self) -> Phase {
        match self {
            ProfileId::DecodeDense | ProfileId::DecodeMoe | ProfileId::DecodeLatent => Phase::Decode,
            ProfileId::PrefillDense | ProfileId::PrefillMoe => Phase::Prefill,
            ProfileId::RecurrentMamba | ProfileId::PortableReference => Phase::Mixed,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ProfileId::DecodeDense => "decode_dense",
            ProfileId::DecodeMoe => "decode_moe",
            ProfileId::DecodeLatent => "decode_latent",
            ProfileId::PrefillDense => "prefill_dense",
            ProfileId::PrefillMoe => "prefill_moe",
            ProfileId::RecurrentMamba => "recurrent_mamba",
            ProfileId::PortableReference => "portable_reference",
        }
    }
}

/// How weights and activations are quantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantScheme {
    /// No quantization; compute dtype is the storage dtype.
    None,
    /// fp8 weights, higher-precision activations.
    W8A16,
    /// fp8 weights and activations.
    W8A8,
    /// Block-scaled fp8 over a `[128, 128]` grid with **arbitrary f32** scales,
    /// as DeepSeek and GLM emit (`weight_scale_inv`).
    ///
    /// Not MX, and not interchangeable with it. The scale is a full f32, so
    /// hardware that only accepts a power-of-two scale operand cannot fold it:
    /// `runtime/amd/amd_common.h:302` records 0.01 folding as 2^-7, 22% low.
    BlockFp8,
    /// 4-bit weights, 8-bit activations.
    W4A8,
    /// Plain fp4 operands with no block scale.
    Fp4,
    /// OCP MX fp8: e4m3 elements in **32-element blocks**, one shared **E8M0**
    /// (power-of-two) exponent per block.
    ///
    /// Distinct from [`Self::BlockFp8`] in both block size and scale type, so a
    /// measurement taken for one says nothing about the other.
    ///
    /// Two realizations exist and they are different kernels:
    /// where [`IsaCaps::block_scale_mma`](hwspec::IsaCaps) holds, the scale is
    /// applied inside the matrix instruction; elsewhere the kernel must dequant
    /// the UE8M0 scale in software and feed plain e4m3. A kernel says which it
    /// is via [`KernelSpec::needs_block_scale_mma`].
    Mxfp8,
    /// OCP MX fp4: e2m1 elements in 32-element blocks with a shared E8M0
    /// exponent. Same two-realization split as [`Self::Mxfp8`].
    Mxfp4,
}

impl QuantScheme {
    /// Whether this scheme carries a per-block shared exponent.
    pub fn is_block_scaled(self) -> bool {
        matches!(self, QuantScheme::BlockFp8 | QuantScheme::Mxfp8 | QuantScheme::Mxfp4)
    }

    /// Whether this is an OCP MX format (32-element blocks, E8M0 scale).
    pub fn is_mx(self) -> bool {
        matches!(self, QuantScheme::Mxfp8 | QuantScheme::Mxfp4)
    }

    /// Elements sharing one scale, where the scheme defines it.
    pub fn block_elems(self) -> Option<u32> {
        match self {
            QuantScheme::Mxfp8 | QuantScheme::Mxfp4 => Some(32),
            // A [128,128] tile, quoted along K as the GEMV path folds it.
            QuantScheme::BlockFp8 => Some(128),
            _ => None,
        }
    }
}

/// Whether results must be bit-reproducible.
///
/// Split-K and atomic reductions reorder floating-point addition, so a kernel
/// that is faster and a kernel that is deterministic can be different kernels.
/// Recording it prevents a tuning campaign from silently trading reproducibility
/// for latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Determinism {
    /// Run-to-run bit-identical on fixed hardware.
    BitExact,
    /// Reduction order may vary.
    Relaxed,
}

/// Coarse size regime. Buckets exist because measuring every shape is not
/// possible and interpolating between measured shapes is not honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShapeClass {
    /// `M == 1`: GEMV, weight-bandwidth bound.
    Gemv,
    /// Small batched decode, `M` up to ~16.
    SmallM,
    /// Prefill below the point where the chip saturates.
    MediumM,
    /// Prefill large enough to saturate.
    LargeM,
}

impl ShapeClass {
    pub fn of(m: i64) -> Self {
        match m {
            i64::MIN..=1 => ShapeClass::Gemv,
            2..=16 => ShapeClass::SmallM,
            17..=512 => ShapeClass::MediumM,
            _ => ShapeClass::LargeM,
        }
    }
}

/// The problem size an op instance is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeSignature {
    pub m: i64,
    pub n: i64,
    pub k: i64,
}

impl ShapeSignature {
    pub fn class(&self) -> ShapeClass {
        ShapeClass::of(self.m)
    }
}

/// One deduplicated op instance the tuner may be asked to serve.
///
/// This is the *query*. A [`KernelSpec`] is an answer, and it is only a legal
/// answer if it accepts this signature.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpSignature {
    pub semantic: SemanticOp,
    pub phase: Phase,
    pub shape: ShapeSignature,
    pub quant: QuantScheme,
    pub determinism: Determinism,
}

impl OpSignature {
    /// A plain bf16 dense matmul — the common case, and the one the first
    /// tuning campaign covers.
    pub fn gemm(phase: Phase, m: i64, n: i64, k: i64) -> Self {
        OpSignature {
            semantic: SemanticOp::DenseMatmul,
            phase,
            shape: ShapeSignature { m, n, k },
            quant: QuantScheme::None,
            determinism: Determinism::Relaxed,
        }
    }
}

/// A kernel's identity.
///
/// Wraps the live device opcode rather than introducing a parallel numbering.
/// Allocating tuning-side ids independently is how a plan ends up reserving a
/// band that another change has already occupied; here it is not expressible,
/// and `packet`'s opcode tests keep the underlying value honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelId(pub DevOp);

// Ordered and hashed by the opcode value rather than by deriving on `DevOp`.
// `DevOp` is the interpreter's hot-path ISA type and carries a deliberately
// minimal derive set; widening it to satisfy a downstream map is the wrong
// direction for that dependency.
impl PartialOrd for KernelId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KernelId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.raw().cmp(&other.raw())
    }
}

impl std::hash::Hash for KernelId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw().hash(state);
    }
}

impl KernelId {
    pub fn raw(self) -> u16 {
        self.0 as u16
    }

    /// The `dev_isa.h` spelling, for reports that a C-side reader must match.
    pub fn c_name(self) -> &'static str {
        self.0.c_name()
    }
}

/// One executable kernel variant.
///
/// "Executable" is the whole point: a `KernelSpec` asserts that *this build* of
/// the runtime contains this kernel, at this ISA level, in this interpreter
/// profile, within this resource envelope. Producing one without probing the
/// artifact re-creates the problem it exists to solve.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSpec {
    pub id: KernelId,
    pub semantic: SemanticOp,
    pub isa: IsaLevel,
    pub profile: ProfileId,
    pub phase: Phase,
    pub quant: QuantScheme,
    pub determinism: Determinism,
    /// Tile, where the kernel has one. `None` for ops that are not tiled.
    pub tile: Option<TileConfig>,
    /// Shape classes this kernel is built to serve. Empty means "any".
    pub shape_classes: Vec<ShapeClass>,
    /// Matrix-engine dtype the body issues, if it uses one. Checked against the
    /// ISA's capabilities, so a kernel cannot claim an instruction the target
    /// does not have.
    ///
    /// For an MX kernel this is the **operand** dtype the matrix instruction
    /// actually sees -- `Fp8` for MXFP8, `Fp4` for MXFP4 -- because MX does not
    /// change the operand rate, only where the shared exponent is applied.
    pub mma_dtype: Option<MmaDtype>,
    /// Whether the body applies the block scale *inside* the matrix instruction.
    ///
    /// `true` restricts the kernel to hardware with
    /// [`IsaCaps::block_scale_mma`](hwspec::IsaCaps). A software-dequant MX
    /// kernel sets `false` and runs anywhere the operand dtype is accelerated,
    /// which is how sm_120a and gfx950 serve MX at all.
    pub needs_block_scale_mma: bool,
    /// Measured cost of having this arm in its interpreter object.
    pub resource: Option<ResourceEnvelopeRef>,
    /// Identifies the *body*, not the opcode. Two specs sharing this value are
    /// the same code reached two ways — see `Inventory::alias_groups`.
    pub implementation_hash: String,
    /// Whether this build actually dispatches the opcode. A declared opcode
    /// with no `case` arm is recorded with `false` rather than omitted, so the
    /// gap is reportable instead of invisible.
    pub dispatched: bool,
}

/// Registers/shared-memory cost, by reference to the profile that carries it.
/// The envelope belongs to the interpreter object, not to one arm, because the
/// megakernel's allocation is the worst case over everything inlined into it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceEnvelopeRef {
    pub profile: ProfileId,
    pub isa: IsaLevel,
}

/// A GEMM tile. Wider than `costmodel::TileShape`, which carries only
/// `bm/bn/bk/split_k` and so cannot distinguish two kernels that differ in
/// staging depth or epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileConfig {
    pub bm: i64,
    pub bn: i64,
    pub bk: i64,
    pub split_k: i64,
    /// Pipeline stages. On NVIDIA this is a compile-time macro per interpreter
    /// object, which is why it belongs to the kernel and not to the packet.
    pub stages: u32,
}

impl TileConfig {
    pub fn new(bm: i64, bn: i64, bk: i64) -> Self {
        TileConfig { bm, bn, bk, split_k: 1, stages: 1 }
    }
}

impl KernelSpec {
    /// A tiled dense-GEMM kernel. The constructor used to seed the registry
    /// from the tile tables that exist today.
    pub fn gemm_tile(
        op: DevOp,
        isa: IsaLevel,
        bm: i64,
        bn: i64,
        bk: i64,
        implementation_hash: &str,
    ) -> Self {
        KernelSpec {
            id: KernelId(op),
            semantic: SemanticOp::DenseMatmul,
            isa,
            profile: ProfileId::PrefillDense,
            phase: Phase::Prefill,
            quant: QuantScheme::None,
            determinism: Determinism::Relaxed,
            tile: Some(TileConfig::new(bm, bn, bk)),
            // Unrestricted: a tiled GEMM body executes any M, and a tile that is
            // too large for the shape is merely wasteful, not illegal. `phase`
            // is what separates decode from prefill. Restricting shape class
            // here instead would make the selector refuse M=1 outright, which is
            // a hard error where the old picker simply returned a poor tile.
            // Kernels that genuinely cannot serve a class set it explicitly.
            shape_classes: Vec::new(),
            mma_dtype: Some(MmaDtype::Bf16),
            needs_block_scale_mma: false,
            resource: None,
            implementation_hash: implementation_hash.to_string(),
            dispatched: true,
        }
    }

    /// Whether this kernel can run on `hw` at all.
    ///
    /// ISA equality, not ordering: capability is not monotonic in release order
    /// (Hopper has `wgmma`, the newer consumer Blackwell does not), so "at least
    /// as new" is not a safe test and is not offered.
    pub fn runs_on(&self, hw: &HardwareFingerprint) -> bool {
        if self.isa != hw.isa {
            return false;
        }
        // A kernel may not claim a matrix dtype the target cannot accelerate.
        if let Some(dt) = self.mma_dtype {
            if !hw.caps().accelerates(dt) {
                return false;
            }
        }
        // Nor an in-MMA block scale on hardware whose assembler rejects the
        // instruction. This is the sm_120 case, settled by mxfp8_probe.cu.
        if self.needs_block_scale_mma && !hw.caps().block_scale_mma {
            return false;
        }
        true
    }

    /// Whether this kernel serves `op`.
    pub fn accepts(&self, op: &OpSignature) -> bool {
        if !self.dispatched {
            return false; // Declared, but this build cannot execute it.
        }
        if self.semantic != op.semantic || self.phase != op.phase || self.quant != op.quant {
            return false;
        }
        // A relaxed-reduction kernel cannot answer a bit-exact request; the
        // reverse is fine.
        if op.determinism == Determinism::BitExact && self.determinism == Determinism::Relaxed {
            return false;
        }
        if !self.shape_classes.is_empty() && !self.shape_classes.contains(&op.shape.class()) {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hwspec::registry as hwreg;

    fn fp(name: &str) -> HardwareFingerprint {
        HardwareFingerprint::from_spec(hwreg::lookup(name).unwrap()).unwrap()
    }

    #[test]
    fn shape_class_boundaries() {
        assert_eq!(ShapeClass::of(1), ShapeClass::Gemv);
        assert_eq!(ShapeClass::of(2), ShapeClass::SmallM);
        assert_eq!(ShapeClass::of(16), ShapeClass::SmallM);
        assert_eq!(ShapeClass::of(17), ShapeClass::MediumM);
        assert_eq!(ShapeClass::of(4096), ShapeClass::LargeM);
    }

    /// fp4 is accelerated on both Blackwell levels but on neither Hopper nor
    /// CDNA. A kernel claiming it must be rejected on the latter.
    #[test]
    fn a_kernel_cannot_claim_an_instruction_the_target_lacks() {
        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "x");
        k.mma_dtype = Some(MmaDtype::Fp4);
        assert!(!k.runs_on(&fp("H100 NVL")), "Hopper does not accelerate fp4");

        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm120a, 128, 128, 32, "x");
        k.mma_dtype = Some(MmaDtype::Fp4);
        assert!(k.runs_on(&fp("RTX 5090")));
    }

    /// Determinism is one-directional: relaxed cannot serve a bit-exact query.
    #[test]
    fn relaxed_reduction_cannot_answer_a_bit_exact_request() {
        let k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "x");
        assert_eq!(k.determinism, Determinism::Relaxed);

        let mut want = OpSignature::gemm(Phase::Prefill, 4096, 4096, 4096);
        want.determinism = Determinism::BitExact;
        assert!(!k.accepts(&want));

        want.determinism = Determinism::Relaxed;
        assert!(k.accepts(&want));
    }

    /// Phase separates decode from prefill, not shape class. A prefill-object
    /// kernel refuses a decode-phase query outright.
    #[test]
    fn phase_separates_decode_from_prefill() {
        let k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "x");
        assert!(!k.accepts(&OpSignature::gemm(Phase::Decode, 1, 4096, 4096)));
        assert!(k.accepts(&OpSignature::gemm(Phase::Prefill, 4096, 4096, 4096)));
    }

    /// A general tiled GEMM must still accept `M=1` within its own phase. It is
    /// a bad tile for that shape, not an illegal one, and refusing it turns a
    /// suboptimal choice into a compile failure.
    #[test]
    fn a_general_tile_kernel_accepts_small_m_in_its_phase() {
        let k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Gfx950, 256, 256, 64, "x");
        assert!(k.shape_classes.is_empty(), "unrestricted by default");
        assert!(k.accepts(&OpSignature::gemm(Phase::Prefill, 1, 128, 128)));
    }

    /// A kernel that genuinely cannot serve a shape class says so explicitly,
    /// and is then filtered.
    #[test]
    fn an_explicitly_restricted_kernel_is_filtered_by_shape_class() {
        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "x");
        k.shape_classes = vec![ShapeClass::LargeM];
        assert!(!k.accepts(&OpSignature::gemm(Phase::Prefill, 1, 4096, 4096)));
        assert!(!k.accepts(&OpSignature::gemm(Phase::Prefill, 64, 4096, 4096)));
        assert!(k.accepts(&OpSignature::gemm(Phase::Prefill, 4096, 4096, 4096)));
    }

    #[test]
    fn quantization_mismatch_is_disqualifying() {
        let k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "x");
        let mut want = OpSignature::gemm(Phase::Prefill, 4096, 4096, 4096);
        want.quant = QuantScheme::W8A8;
        assert!(!k.accepts(&want), "a bf16 body cannot serve a w8a8 request");
    }

    #[test]
    fn profile_phase_is_consistent() {
        assert_eq!(ProfileId::DecodeDense.phase(), Phase::Decode);
        assert_eq!(ProfileId::PrefillMoe.phase(), Phase::Prefill);
    }

    /// MX is not the same format as the `[128,128]` grid, and the type system
    /// must not let a measurement for one answer for the other.
    #[test]
    fn mx_and_deepseek_block_fp8_are_different_schemes() {
        assert_ne!(QuantScheme::Mxfp8, QuantScheme::BlockFp8);
        assert!(QuantScheme::Mxfp8.is_mx());
        assert!(!QuantScheme::BlockFp8.is_mx());
        // Both are block scaled, but over different block sizes.
        assert!(QuantScheme::BlockFp8.is_block_scaled());
        assert_eq!(QuantScheme::Mxfp8.block_elems(), Some(32));
        assert_eq!(QuantScheme::Mxfp4.block_elems(), Some(32));
        assert_eq!(QuantScheme::BlockFp8.block_elems(), Some(128));
        assert_eq!(QuantScheme::Fp4.block_elems(), None);

        // And a kernel built for one must refuse a request for the other.
        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm100a, 128, 128, 32, "x");
        k.quant = QuantScheme::Mxfp8;
        let mut want = OpSignature::gemm(Phase::Prefill, 4096, 4096, 4096);
        want.quant = QuantScheme::BlockFp8;
        assert!(!k.accepts(&want));
    }

    /// The finding from `runtime/nvidia/experiments/mxfp8_probe.cu`: ptxas
    /// rejects block-scale MMA on sm_120 outright. A native MX kernel must
    /// therefore be unselectable there, while the software-dequant variant of
    /// the same format remains available.
    #[test]
    fn native_mx_is_datacenter_blackwell_only() {
        let native = |isa| {
            let mut k = KernelSpec::gemm_tile(DevOp::Gemm, isa, 128, 128, 32, "mx-native");
            k.quant = QuantScheme::Mxfp8;
            k.mma_dtype = Some(MmaDtype::Fp8);
            k.needs_block_scale_mma = true;
            k
        };
        assert!(native(IsaLevel::Sm100a).runs_on(&fp("B200")), "tcgen05 has it");
        assert!(
            !native(IsaLevel::Sm120a).runs_on(&fp("RTX 5090")),
            "ptxas rejects 'mma with block scale' on sm_120"
        );
        assert!(!native(IsaLevel::Sm90a).runs_on(&fp("H100 NVL")), "Hopper has no block scale");
    }

    /// The fallback the probe measured: dequant UE8M0 in software, feed plain
    /// e4m3. That kernel needs only fp8 operands, so it runs on consumer
    /// Blackwell and on Hopper.
    #[test]
    fn software_dequant_mx_runs_where_native_cannot() {
        let sw = |isa| {
            let mut k = KernelSpec::gemm_tile(DevOp::Gemm, isa, 128, 128, 32, "mx-swdequant");
            k.quant = QuantScheme::Mxfp8;
            k.mma_dtype = Some(MmaDtype::Fp8);
            k.needs_block_scale_mma = false;
            k
        };
        assert!(sw(IsaLevel::Sm120a).runs_on(&fp("RTX 5090")));
        assert!(sw(IsaLevel::Sm90a).runs_on(&fp("H100 NVL")));
        assert!(sw(IsaLevel::Gfx950).runs_on(&fp("MI350X")));
    }

    /// MXFP4 needs fp4 operands, which Hopper does not have at all -- so the
    /// software fallback that rescues MXFP8 on Hopper does not rescue MXFP4.
    #[test]
    fn mxfp4_needs_fp4_operands_which_hopper_lacks() {
        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm90a, 128, 128, 32, "mxfp4-sw");
        k.quant = QuantScheme::Mxfp4;
        k.mma_dtype = Some(MmaDtype::Fp4);
        assert!(!k.runs_on(&fp("H100 NVL")), "Hopper has no fp4 matrix path");

        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm120a, 128, 128, 32, "mxfp4-sw");
        k.quant = QuantScheme::Mxfp4;
        k.mma_dtype = Some(MmaDtype::Fp4);
        assert!(k.runs_on(&fp("RTX 5090")), "consumer Blackwell has fp4 mma.sync");
    }

    /// gfx950 is the strongest MXFP4 target in the registry, and the reason is
    /// two independent capabilities meeting: CDNA4 added fp4 matrix cores
    /// (`mi350.rs:58`) *and* its scale conversion takes E8M0 only
    /// (`amd_common.h:302`) -- which is precisely the MX scale format. CDNA3 has
    /// neither, so the two AMD generations are not interchangeable here.
    #[test]
    fn gfx950_serves_mxfp4_and_gfx942_does_not() {
        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Gfx950, 128, 128, 64, "mxfp4");
        k.quant = QuantScheme::Mxfp4;
        k.mma_dtype = Some(MmaDtype::Fp4);
        assert!(k.runs_on(&fp("MI350X")));
        assert!(IsaLevel::Gfx950.caps().mx_scale_cvt, "E8M0 scale operand");

        let mut k = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Gfx942, 128, 128, 64, "mxfp4");
        k.quant = QuantScheme::Mxfp4;
        k.mma_dtype = Some(MmaDtype::Fp4);
        assert!(!k.runs_on(&fp("MI300X")), "CDNA3 has no fp4 matrix path");
    }

    /// Native and software MX are different kernels for the same format, so a
    /// registry holding both must not report them as aliases.
    #[test]
    fn native_and_software_mx_are_distinct_implementations() {
        let mut a = KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Sm100a, 128, 128, 32, "mx-native");
        a.quant = QuantScheme::Mxfp8;
        a.needs_block_scale_mma = true;
        let mut b = KernelSpec::gemm_tile(DevOp::GemmMed, IsaLevel::Sm100a, 128, 128, 32, "mx-sw");
        b.quant = QuantScheme::Mxfp8;
        b.needs_block_scale_mma = false;
        assert_ne!(a.implementation_hash, b.implementation_hash);
    }

}