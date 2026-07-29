//! Kimi-K3 BLOCK structure — AttnRes and the `situ` GLU.
//!
//! `crates/devgen/src/kda.rs` emits K3's MIXER. This emits the BLOCK AROUND IT, and that is where
//! K3 stops looking like every other model in this tree:
//!
//! ```text
//!   a K3 layer is NOT  `residual + attn` then `residual + mlp`.
//! ```
//!
//! `attn_res_block_size: 12` routes every layer through `_forward_attn_residual`
//! (`modeling_kimi_linear.py:973`). The plain residual ADD is replaced by **AttnRes** — a softmax
//! mix over the running prefix sum and up to 8 snapshots of it — applied TWICE per layer, and the
//! block's OUTPUT is the prefix sum. AMD's day-0 post names the structure and confirms the period:
//! *"stores one block residual every 12 layers"*.
//!
//! # The control flow, and the two shapes it produces
//!
//! ```text
//!   prefix_in                                        (the layer input)
//!   if nb > 0:  h = AttnRes(prefix_in, blkres, attn_score_w)   else  h = prefix_in
//!   if l % B == 0:  blkres.push(prefix_in) ; prefix = None     (a SNAPSHOT layer)
//!   attn = KDA_or_MLA(input_layernorm(h))
//!   prefix = (prefix_in + attn) if prefix is not None else attn
//!   h2  = AttnRes(prefix, blkres, mlp_score_w)
//!   ffn = MLP_or_MoE(post_attention_layernorm(h2))
//!   out = prefix + ffn
//! ```
//!
//! Layer 0 is a snapshot layer with an EMPTY `blkres`, so it skips the first AttnRes AND resets
//! `prefix` — its output is `attn + ffn`, and the layer input reaches it only through the mix.
//! Layer 1 is neither, so both AttnRes calls are live and `prefix` accumulates.
//!
//! **The two calls use DIFFERENT weights** (`self_attention_res_*` vs `mlp_res_*`). Layer 0 never
//! reads the first pair at all, so a loader that derives coverage from the tensor set will report
//! two dead tensors on that layer and be right.
//!
//! # Where a wrong wiring hides, measured
//!
//! At a SNAPSHOT layer the block output is `attn + ffn` while the plain wiring gives
//! `hidden + attn + ffn`: a **1.0** relative difference, which any output check catches.
//! At a NON-snapshot layer `prefix = prefix_in + attn`, so the block output is
//! `prefix_in + attn + ffn` — **exactly what the plain wiring produces**. Measured on real layer-1
//! weights: **3.0e-3** at the block output, against **8.1e-1** and **7.6e-1** at the two AttnRes
//! outputs themselves.
//!
//! > A block-output-only gate does not see AttnRes at 68 of K3's 93 layers. The gate has to diff
//! > the sub-layer INPUTS. `runtime/tests/k3_moe_block_gfx950_test.c` does, and says so.
//!
//! # `situ`, and why it is an opcode and not an `act` code
//!
//! Every GLU site in this tree computes `act(g) * u` and selects the activation with a two-value
//! ternary. `situ` transforms the UP branch too — `A(g) * B(u)` — so the EXPRESSION SHAPE changes,
//! not just the function. A third act code alone would apply the gate transform and leave `up`
//! un-clipped: a small error at `|u| < 25` that grows with the tail, i.e. plausible output and the
//! wrong model. [`DevOp::SituGlu`] makes the omission impossible.
//!
//! The routed experts cannot use it — their GLU is inside the expert kernel — so `op_moe.h` grew
//! `PLOW_MOE_ACT_SITU = 2` and a PAIR-form epilogue (`moe_glu`), with the betas riding in `f0`/`f1`
//! (free on every GLU-family op). `moe_act` itself returns **NaN** for that code, on purpose: any
//! epilogue not converted to the pair form poisons its output instead of silently computing
//! `gelu_tanh(g) * u`.

use packet::dev::DevOp;
use packet::devbuild::Builder;
use packet::rope::{GenTensor, RopeScale};

/// K3's block-structure constants (`text_config`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct K3BlockCfg {
    /// `hidden_size` — 7168.
    pub hidden: u32,
    /// `rms_norm_eps` — 1e-5, used by both norms and by AttnRes's per-row normalization.
    pub eps: f32,
    /// `attn_res_block_size` — 12. A snapshot is taken when `layer_idx % this == 0`.
    pub attn_res_block_size: u32,
    /// `activation_situ_beta` — 4.0. Clips the GATE branch to +-beta.
    pub situ_beta: f32,
    /// `activation_situ_linear_beta` — 25.0. Clips the UP branch. `<= 0` disables it, which is
    /// what `linear_beta is None` means in the reference.
    pub situ_linear_beta: f32,
}

/// The AttnRes kernel's compile-time bound on `nb`, mirroring `PLOW_ATTNRES_MAXB` in
/// `runtime/amd/op_k3.h`.
///
/// K3 needs at most `ceil(93/12) = 8`. The arm RETURNS on a larger `nb` rather than indexing off
/// the end of its LDS carve, so the emitter has to refuse first — a silent no-write is the failure
/// this tree keeps finding.
pub const K3_ATTNRES_MAXB: u32 = 16;

impl K3BlockCfg {
    /// How many block-residual rows layer `l` (0-based) sees on ENTRY.
    ///
    /// A snapshot is pushed at the TOP of every layer with `l % B == 0`, before the attention, so
    /// layer `l` enters with one row per earlier snapshot: `ceil(l / B)`. Layer 0 enters with
    /// **zero**, which is why its attn-side AttnRes is skipped entirely.
    pub fn blocks_at(&self, l: u32) -> u32 {
        l.div_ceil(self.attn_res_block_size)
    }
    /// Does layer `l` push a snapshot and therefore RESET the prefix sum?
    ///
    /// Driven by the modulus because the reference is (`layer_idx % attn_res_block_size == 0`) —
    /// unlike the KDA/MLA layer split, which is a config LIST and where a modulus is wrong at
    /// 0-based layer 92.
    pub fn snapshots(&self, l: u32) -> bool {
        l % self.attn_res_block_size == 0
    }
    /// The largest `nb` any layer of a `n_layers`-deep model sees. Must be `<= K3_ATTNRES_MAXB`.
    pub fn max_blocks(&self, n_layers: u32) -> u32 {
        self.blocks_at(n_layers - 1) + u32::from(self.snapshots(n_layers - 1))
    }
}

/// Emit AttnRes: `out = softmax_r(rms(v_r) . score_w) @ v`, over `blkres` rows then `prefix`.
///
/// `score_w` is `norm.weight * proj.weight`, **folded at prep time** into one `[hidden]` f32 —
/// both factors are parameters, so the product is constant and neither is needed separately on
/// device.
///
/// `nb == 0` is an exact copy (softmax over one element). The reference SKIPS the call in that
/// case; the caller should too, and [`emit_k3_block_out`] does. The arm handles it so that
/// emitting it anyway cannot produce a zero-filled buffer.
///
/// **One workgroup per token.** Both reductions span the full 7168-wide row and the softmax couples
/// the rows, so `blocks = min(T, n_cu)` — at `T = 1` that is 1 of 256. It is recorded rather than
/// hidden: `perf-data/kimi-k3-kernel-gap.md` §10 item 7 requires this to stay ONE packet ("three
/// packets x 186 is 3.3 ms/token of pure protocol"), which rules out splitting the reduction across
/// blocks and finishing it in a second packet. The batched form is the fix and it is not written.
#[allow(clippy::too_many_arguments)]
pub fn emit_attn_res(
    b: &mut Builder,
    c: &K3BlockCfg,
    out: u32,
    prefix: u32,
    blkres: u32,
    score_w: u32,
    t: u32,
    nb: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    assert!(
        nb <= K3_ATTNRES_MAXB,
        "AttnRes: nb = {nb} exceeds PLOW_ATTNRES_MAXB = {K3_ATTNRES_MAXB}; the arm RETURNS rather \
         than overrunning its LDS carve, so this would leave `out` untouched — a silent NOP, not a \
         wrong number"
    );
    let blocks: Vec<u32> = (0..t.min(n_cu).max(1)).collect();
    b.emit(DevOp::AttnRes, blocks, deps, |d| {
        d.t[0] = out;
        d.t[1] = prefix;
        d.t[2] = blkres;
        d.t[3] = score_w;
        d.i[0] = t;
        d.i[1] = c.hidden;
        d.i[2] = nb;
        d.f[0] = c.eps;
    })
}

/// Emit the `situ` GLU over `n` elements: `A(gate) * B(up)`.
///
/// The two betas ride in `f0`/`f1`, which are free on every GLU-family op — so this consumes no
/// `i` slot and no existing emitter changes.
#[allow(clippy::too_many_arguments)]
pub fn emit_situ_glu(
    b: &mut Builder,
    c: &K3BlockCfg,
    out: u32,
    gate: u32,
    up: u32,
    n: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::SituGlu, all, deps, |d| {
        d.t[0] = out;
        d.t[1] = gate;
        d.t[2] = up;
        d.i[0] = n;
        d.f[0] = c.situ_beta;
        d.f[1] = c.situ_linear_beta;
    })
}

/// Emit the MLA **output gate**: `out = attn * sigmoid(g)`, applied BEFORE `o_proj`.
///
/// `mla_use_output_gate: true` on all 24 of K3's MLA layers. `g` is the raw `g_proj` output of the
/// MLA sub-layer INPUT (the post-`input_layernorm` hidden), NOT of the attention output — the
/// reference reads `hidden_states`, which at that point is the norm output
/// (`modeling_kimi_linear.py:471`). Feeding it the attention output has the right shape and the
/// wrong model.
///
/// `n` is `n_head_local * v_head_dim` and both operands are **head-major**, which is exactly what
/// [`DevOp::MlaMergeFold`] writes and exactly what the reference's `.reshape(batch, seq, -1)`
/// produces, so no permute is implied.
///
/// It is `sigmoid(g)`, not `silu(g)` — which is why it is [`DevOp::MlaOutGate`] rather than a third
/// `act` code on [`DevOp::Glu`], whose `act=1` IS silu. The two differ by a factor of the logit:
/// finite, correctly-shaped, and the wrong model on 24 layers of every token.
pub fn emit_mla_out_gate(
    b: &mut Builder,
    out: u32,
    attn: u32,
    g: u32,
    n: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::MlaOutGate, all, deps, |d| {
        d.t[0] = out;
        d.t[1] = attn;
        d.t[2] = g;
        d.i[0] = n;
    })
}

/// The `(cos, sin)` recipe pair for K3's **NoPE** MLA: an all-identity table, `cos = 1, sin = 0`.
///
/// `mla_use_nope: true` and `text_config` carries no `rope_theta`
/// (`configuration_kimi_k3.py`; `KimiMLAAttention.__init__` ends `self.rotary_emb = None` and
/// `forward` concatenates `q_rot`/`k_rot` back UNROTATED). The 64 "rope" dims are extra CONTENT
/// channels of the 192-wide key.
///
/// # THIS IS NOT A REMOVAL, and that is the whole point
///
/// `perf-data/kimi-k3-kernel-gap.md` §8c calls the fix "skip both `HeadNormRope` emits — a removal,
/// effort XS". It is not. The k-side [`DevOp::HeadNormRope`] is the **only writer of the
/// `kv.{l}.krot` cache row**, AND it is the instruction that
/// `plowrt::exec::amd::kv_write_row_field` and `runtime/tests/glm52_decode.c:419` both SCAN FOR in
/// order to patch that row's position each step. Delete it and (a) the rope half of every cached
/// key is never written while `FlashMlaDecode` keeps reading it at `i[5]`, and (b) the layer
/// silently drops out of the KV-row-writer list — the scan just finds fewer, with **no count
/// check**. In `glm52_decode.c` the two arrays are index-paired and `nlk` is incremented only on
/// the krot match, so losing the krot writer silently loses the ckv patch too.
///
/// So: **keep the WRITE, remove the ROTATION.** With `cos = 1, sin = 0` the rotation is
/// `v*1 - partner*0 = v` — both constants exact in f32, `gamma` already `TENSOR_NONE` and
/// `skip_norm` already 1 on this emit, so `HeadNormRope` becomes a bit-exact bf16 copy that still
/// writes the cache row and still answers the scan.
///
/// # No new generator kind was needed
///
/// [`packet::rope::rope_tables`] already emits `(1.0, 0.0)` for every angle past
/// `rope_angles = frac * hd / 2`, because GLM's partial rotary needs exactly that for its NoPE
/// tail. `frac = 0.0` puts EVERY angle past it. `theta` is then never read, so it is pinned to
/// `1.0` rather than left to a caller's `unwrap_or` default — a defaulted theta reaching a table
/// that ignores it is the shape of bug this tree keeps finding, and the assertion in
/// [`k3_nope_table_is_exactly_identity`](self) is what keeps it that way.
pub fn k3_nope_rope_pair(ctx: u32, qk_rope: u32) -> [GenTensor; 2] {
    GenTensor::rope_pair(ctx, qk_rope, 1.0, 0.0, RopeScale::None)
}

/// The block's final combine: `out = prefix + ffn`.
///
/// Its own function because the operand that is NOT here is the point. A plain block would add the
/// LAYER INPUT; K3 adds the PREFIX SUM, and at a snapshot layer those differ by the whole embedding
/// state.
pub fn emit_k3_block_out(
    b: &mut Builder,
    out: u32,
    prefix: u32,
    ffn: u32,
    n: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::Residual, all, deps, |d| {
        d.t[0] = out;
        d.t[1] = prefix;
        d.t[2] = ffn;
        d.i[0] = n;
        d.f[0] = 1.0;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k3() -> K3BlockCfg {
        K3BlockCfg {
            hidden: 7168,
            eps: 1e-5,
            attn_res_block_size: 12,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
        }
    }

    /// The block-residual count per layer, and the two layers the gates were run on.
    #[test]
    fn snapshot_schedule_matches_the_reference_control_flow() {
        let c = k3();
        // Layer 0: snapshot, and NOTHING to mix with yet — so the attn-side AttnRes is skipped and
        // the prefix sum is reset. Its block output is `attn + ffn`.
        assert_eq!(c.blocks_at(0), 0);
        assert!(c.snapshots(0));
        // Layer 1: no snapshot, one row available — BOTH AttnRes calls live, prefix accumulates.
        assert_eq!(c.blocks_at(1), 1);
        assert!(!c.snapshots(1));
        // Layer 12 snapshots again; layer 13 sees two rows.
        assert!(c.snapshots(12));
        assert_eq!(c.blocks_at(12), 1);
        assert_eq!(c.blocks_at(13), 2);
        // 93 layers -> snapshots at 0,12,...,84 = 8 of them, so nb never exceeds 8.
        let snaps: Vec<u32> = (0..93).filter(|&l| c.snapshots(l)).collect();
        assert_eq!(snaps, vec![0, 12, 24, 36, 48, 60, 72, 84]);
        assert_eq!(c.max_blocks(93), 8);
        assert!(c.max_blocks(93) <= K3_ATTNRES_MAXB);
    }

    /// `emit_attn_res` must refuse an `nb` the kernel would silently drop.
    #[test]
    #[should_panic(expected = "PLOW_ATTNRES_MAXB")]
    fn attn_res_refuses_an_nb_the_arm_would_silently_ignore() {
        let c = k3();
        let mut b = Builder::new(256);
        let o = b.tensor("act.o", 7168 * 2);
        let p = b.tensor("act.p", 7168 * 2);
        let r = b.tensor("act.r", 7168 * 2 * 17);
        let w = b.tensor("act.w", 7168 * 4);
        emit_attn_res(&mut b, &c, o, p, r, w, 1, K3_ATTNRES_MAXB + 1, 256, &[]);
    }

    /// Both new opcodes are reachable from an emitter, and AttnRes's honest occupancy is asserted
    /// rather than left to be discovered.
    #[test]
    fn the_two_new_opcodes_are_emitted_and_their_slice_maps_are_pinned() {
        let c = k3();
        let mut b = Builder::new(256);
        let out = b.tensor("act.out", 7168 * 2);
        let prefix = b.tensor("act.prefix", 7168 * 2);
        let blkres = b.tensor("act.blkres", 7168 * 2);
        let sw = b.tensor("act.score_w", 7168 * 4);
        let g = b.tensor("act.g", 33792 * 2);
        let u = b.tensor("act.u", 33792 * 2);
        let a = b.tensor("act.a", 33792 * 2);
        let seed = b.emit(DevOp::Nop, (0..256).collect::<Vec<u32>>(), &[], |_| {});
        let c_ar = emit_attn_res(&mut b, &c, out, prefix, blkres, sw, 1, 1, 256, &[seed]);
        let c_su = emit_situ_glu(&mut b, &c, a, g, u, 33792, 256, &[c_ar]);
        emit_k3_block_out(&mut b, out, prefix, a, 7168, 256, &[c_su]);
        let p = b.finish();
        let ops: Vec<u16> = p.insts.iter().map(|i| i.op).collect();
        for op in [DevOp::AttnRes, DevOp::SituGlu] {
            assert!(
                ops.contains(&(op as u16)),
                "{op:?} is not emitted — an opcode nothing reaches is how Mamba2Scan became dead \
                 code"
            );
        }
        let ar = p.insts.iter().find(|i| i.op == DevOp::AttnRes as u16).unwrap();
        assert_eq!(ar.blocks, 1, "one workgroup per token: 1 of 256 at T=1, a KNOWN perf gap");
        assert_eq!(ar.i[2], 1, "nb");
        let su = p.insts.iter().find(|i| i.op == DevOp::SituGlu as u16).unwrap();
        assert_eq!(su.blocks, 256, "elementwise: the whole chip");
        assert_eq!(su.i[0], 33792);
        // The betas must actually reach the packet. An immediate that is emitted but not read is
        // the contract's SS3 bug shape; one that is READ but never emitted is this one.
        assert_eq!(su.f[0], 4.0);
        assert_eq!(su.f[1], 25.0);
    }

    /// K3's NoPE table must be EXACTLY identity at every position and every angle.
    ///
    /// Not "close to" identity: `HeadNormRope` computes `v*cos ± partner*sin`, and the claim this
    /// gate rests on is that the op becomes a BIT-EXACT copy. `1.0` and `0.0` are exact in f32 and
    /// `x*1 - y*0 == x` for every finite `x`, so the table has to be those two constants and
    /// nothing else. A table that were merely tiny-angled would round-trip through bf16 and pass a
    /// 1.5e-2 residual check while quietly rotating.
    #[test]
    fn k3_nope_table_is_exactly_identity() {
        let ctx = 64u32;
        let hd = 64u32; // qk_rope_head_dim
        let [gc, gs] = k3_nope_rope_pair(ctx, hd);
        let cos = gc.generate().expect("cos recipe must materialize");
        let sin = gs.generate().expect("sin recipe must materialize");
        let n = (ctx * hd / 2) as usize;
        assert_eq!(cos.len(), n * 4);
        assert_eq!(sin.len(), n * 4);
        for i in 0..n {
            let c = f32::from_le_bytes(cos[i * 4..i * 4 + 4].try_into().unwrap());
            let s = f32::from_le_bytes(sin[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(c.to_bits(), 1.0f32.to_bits(), "cos[{i}] must be exactly 1.0, got {c}");
            assert_eq!(s.to_bits(), 0.0f32.to_bits(), "sin[{i}] must be exactly 0.0, got {s}");
        }
        // And it must NOT coincide with a real table — otherwise the test proves nothing about the
        // emit having actually selected the NoPE recipe.
        let [rc, _] = GenTensor::rope_pair(ctx, hd, 8_000_000.0, 1.0, RopeScale::None);
        assert_ne!(rc.generate().unwrap(), cos, "the NoPE recipe produced GLM's rotating table");
    }

    /// `MlaOutGate` reaches a packet, with the right width and the right operand ORDER.
    ///
    /// The operand order is the load-bearing part: `t1` is the attention output and `t2` is the
    /// g_proj logits, and the kernel applies the sigmoid to `t2` only. Swapping them yields
    /// `sigmoid(attn) * g` — finite, same shape, wrong model — which no shape or symbol check sees.
    #[test]
    fn mla_out_gate_is_emitted_with_the_gated_operand_in_t2() {
        let mut b = Builder::new(256);
        let out = b.tensor("act.oatg", 12288 * 2);
        let attn = b.tensor("act.oat", 12288 * 2);
        let g = b.tensor("act.gproj", 12288 * 2);
        emit_mla_out_gate(&mut b, out, attn, g, 96 * 128, 256, &[]);
        let p = b.finish();
        let i = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MlaOutGate as u16)
            .expect("MlaOutGate is not emitted — an opcode nothing reaches is how Mamba2Scan died");
        assert_eq!(i.blocks, 256, "elementwise: the whole chip");
        assert_eq!(i.i[0], 12288, "n_head * v_head_dim = 96 * 128");
        assert_eq!(i.t[0], out);
        assert_eq!(i.t[1], attn, "t1 is the UNGATED attention output");
        assert_eq!(i.t[2], g, "t2 is the g_proj logits — the operand the sigmoid is applied to");
    }

    /// At T > 1 AttnRes gets more workgroups, capped by the CU count.
    #[test]
    fn attn_res_blocks_track_the_token_count() {
        let c = k3();
        for (t, want) in [(1u32, 1u32), (4, 4), (512, 256)] {
            let mut b = Builder::new(256);
            let o = b.tensor("act.o", 1);
            let p = b.tensor("act.p", 1);
            let r = b.tensor("act.r", 1);
            let w = b.tensor("act.w", 1);
            emit_attn_res(&mut b, &c, o, p, r, w, t, 1, 256, &[]);
            assert_eq!(b.finish().insts[0].blocks, want as u16, "T={t}");
        }
    }
}
