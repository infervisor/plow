//! Nemotron-H (`nemotron_h`, or the `llm_config` text tower of `nemotron_h_omni`) on the sm_120
//! interpreter (sm_120a, sm_121a): Mamba-2 / NoPE GQA / latent NVFP4 MoE hybrid.
//!
//! Block: `x + mixer(RMSNorm(x))`. Mamba mixer: FP8 per-tensor in_proj -> causal conv + SiLU
//! (`Mamba2Conv`) -> SSD (`Mamba2Step` decode; `Mamba2ChunkState/Pass/Out` prefill, chunk 128,
//! continuing from the slot state) -> grouped gated RMSNorm (`Mamba2GatedNorm`) -> FP8 out_proj.
//! Attention: FP8 q/k/v/o, NoPE, BF16 KV written by `QwenHeadNormRope` (no norm, no rotary),
//! `FlashDecode` + `FlashMerge` / fused one-split `FlashPrefill` at hd128. The checkpoint's FP8 linears keep
//! their e4m3 bytes; the per-tensor `weight_scale` is tiled to the per-channel vector the FP8
//! GEMV/GEMM read (`derived_weights.json` `repeat`). Activations: decode W8A16 (bf16 into the
//! FP8 GEMV); prefill the peer's static per-tensor FP8 (`QuantFp8` with the checkpoint
//! `input_scale` + a cuBLASLt `GemmFp8` with scalar scales, `FP8_TENSOR_SCALE`) at the buckets
//! that route holds, per-row dynamic FP8 on the native `GemmFp8` elsewhere.
use super::*;
use crate::components::gqa_attention::{self, GqaAttention};
use crate::components::mamba2_mixer::{self, Mamba2Mixer, CHUNK};
use crate::components::nvfp4_moe::{self, Nvfp4Moe};
use crate::components::{self, fp8_proj, Program};
use crate::derived::checkpoint_sources;
use nn_graph::models::config::{NemotronHConfig, NemotronHLayer, NemotronHLinearQuant};
use nn_graph::models::ModelConfig;
use packet::typed_abi::{GemvFp8Args, QwenRmsNormArgs, ResidualArgs, TypedOp};
use std::path::Path;

#[path = "nemotron_h_mtp.rs"]
mod mtp;

/// The prefill buckets: the W8A8 prefill GEMM's validated row set, 8192 the always rung.
const PREFILL_BUCKETS: [u32; 4] = [128, 1024, 4096, 8192];
/// Bytes per element of the slot SSM state `state.nemo.{l}.ssm` (fp16, the peer's
/// `--mamba-ssm-cache-dtype float16`; op_mamba.cuh `MAMBA_STATE_BYTES`). The single source of the
/// state width: buffer sizes, `recurrent_state` slot bytes and the ops' `state_bytes` immediate.
pub const SSM_STATE_BYTES: u64 = 2;

struct Cfg {
    g: NemotronHConfig,
    /// `<weight prefix>backbone.`
    backbone: String,
    head: String,
    block: Option<std::ops::Range<usize>>,
    /// The target ISA: its cuBLASLt FP8 scale modes pick the prefill projection route.
    isa: Option<hwspec::IsaLevel>,
    mixer: Mamba2Mixer,
    attn: GqaAttention,
    moe: Nvfp4Moe,
}

impl Cfg {
    fn parse(root: &Value) -> Self {
        let g = match ModelConfig::from_json(&root.to_string())
            .unwrap_or_else(|e| panic!("nemotron_h config: {e}"))
        {
            ModelConfig::NemotronH(g) => g,
            other => panic!("nemotron_h emitter given a non-Nemotron-H config: {other:?}"),
        };
        let u = |v: i64| u32::try_from(v).expect("nemotron_h dimension");
        assert!(!g.tie_word_embeddings, "nemotron_h: tied embeddings");
        let arms = kernelcaps::arms::SM120;
        let hidden = u(g.hidden_size);
        let refuse = |e: kernelcaps::arms::MissingArm| -> ! { panic!("nemotron_h: {e}") };
        let mixer = Mamba2Mixer::new(
            mamba2_mixer::Geometry {
                hidden,
                heads: g.mamba_num_heads,
                head_dim: u(g.mamba_head_dim),
                state: u(g.ssm_state_size),
                groups: g.n_groups,
                conv_kernel: g.conv_kernel,
                conv_bias: g.use_conv_bias,
                chunk: g.chunk_size,
                dt_limit: g.time_step_limit,
                eps: g.layer_norm_epsilon,
                state_bytes: SSM_STATE_BYTES as u32,
            },
            "nemo",
            true,
            &arms,
        )
        .unwrap_or_else(|e| refuse(e));
        assert_eq!(
            i64::from(mixer.geometry().in_proj_width()),
            g.in_proj_width(),
            "nemotron_h: in_proj width"
        );
        let attn = GqaAttention::new(
            gqa_attention::Geometry {
                hidden,
                heads: g.num_attention_heads,
                kv_heads: g.num_key_value_heads,
                head_dim: u(g.head_dim),
            },
            &arms,
        )
        .unwrap_or_else(|e| refuse(e));
        let moe = Nvfp4Moe::new(
            nvfp4_moe::Geometry {
                hidden,
                experts: g.n_routed_experts,
                top_k: g.num_experts_per_tok,
                inter: u(g.moe_intermediate_size),
                shared_inter: u(g.moe_shared_expert_intermediate_size),
                latent: g.moe_latent_size.map(u),
                routed_scale: g.routed_scaling_factor,
                norm_topk: g.norm_topk_prob,
            },
            nvfp4_moe::ExpertMlp {
                act: &g.mlp_hidden_act,
                bias: g.mlp_bias,
                n_group: g.n_group,
                topk_group: g.topk_group,
            },
            &arms,
        )
        .unwrap_or_else(|e| refuse(e));
        let backbone = format!("{}backbone.", g.weight_prefix);
        let head = format!("{}lm_head.weight", g.weight_prefix);
        Self {
            g,
            backbone,
            head,
            block: None,
            isa: None,
            mixer,
            attn,
            moe,
        }
    }
    fn hidden(&self) -> u32 {
        self.g.hidden_size as u32
    }
    fn in_block(&self, l: usize) -> bool {
        self.block.as_ref().is_none_or(|r| r.contains(&l))
    }
    /// `module` is a checkpoint FP8 per-tensor linear.
    fn check_fp8(&self, module: &str) {
        assert_eq!(
            self.g.linear_quant(module),
            NemotronHLinearQuant::Fp8,
            "{module}: expected an FP8 per-tensor linear"
        );
    }
}

struct Em<'a> {
    p: Program,
    c: &'a Cfg,
    pending_residual: Option<(u32, u32)>,
}

/// The FP8 per-tensor projection route every backbone mixer takes.
fn fp8_route(c: &Cfg) -> impl FnMut(&mut Program, u32, u32, &str, u32, u32, u32) -> u32 + '_ {
    move |p, out, src, module, n, k, dep| {
        c.check_fp8(module);
        fp8_proj::emit(p, out, src, module, n, k, dep)
    }
}

impl Em<'_> {
    fn weight(&mut self, name: &str, bytes: u64) -> u32 {
        self.p.b.tensor(name, bytes)
    }
    /// A bf16 linear (`[n, k]`): GEMV in decode, the bf16 GEMM in prefill.
    fn bf16_proj(&mut self, out: u32, src: u32, name: &str, n: u32, k: u32, dep: u32) -> u32 {
        components::bf16_proj(&mut self.p, out, src, name, n, k, dep)
    }
    fn norm(&mut self, out: u32, src: u32, gamma_name: &str, dep: u32) -> u32 {
        self.p.quant = None;
        let hidden = self.c.hidden();
        let gamma = self.weight(gamma_name, u64::from(hidden) * 2);
        let (dep, radd) = match self.pending_residual.take() {
            Some((x, mixed)) if x == src => (dep, mixed),
            Some(pending) => {
                self.pending_residual = Some(pending);
                (self.flush_residual(dep), TENSOR_NONE)
            }
            None => (dep, TENSOR_NONE),
        };
        self.rms_norm(out, src, gamma, radd, dep)
    }
    /// `out = RMSNorm(src [+= radd]) * gamma` over this program's rows (no gamma offset).
    fn rms_norm(&mut self, out: u32, src: u32, gamma: u32, radd: u32, dep: u32) -> u32 {
        let args = QwenRmsNormArgs {
            out,
            x: src,
            gamma,
            active: self.p.active,
            radd,
            width: self.c.hidden(),
            B: self.p.rows,
            eps: self.c.g.layer_norm_epsilon,
            gamma_offset: 0.0,
        };
        let cus = (0..self.p.rows.min(self.p.b.n_cu())).collect();
        args.emit(&mut self.p.b, cus, &[dep])
    }
    /// x += mixed, deferred onto the next RMSNorm of x (QwenRmsNorm `radd`).
    fn residual(&mut self, x: u32, mixed: u32, dep: u32) -> u32 {
        assert!(self.pending_residual.is_none(), "two residual adds without a norm");
        self.pending_residual = Some((x, mixed));
        dep
    }
    fn flush_residual(&mut self, dep: u32) -> u32 {
        let Some((x, mixed)) = self.pending_residual.take() else {
            return dep;
        };
        let n = self.p.rows * self.c.hidden();
        self.p.emit(
            ResidualArgs {
                out: x,
                a: x,
                b: mixed,
                n,
                scale: 1.0,
                ..Default::default()
            },
            &[dep],
        )
    }
    fn mamba(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32) -> u32 {
        let c = self.c;
        let module = format!("{}layers.{layer}.mixer", c.backbone);
        c.mixer
            .emit(&mut self.p, layer, &module, hn, mixed, dep, &mut fp8_route(c))
    }
    fn attention(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32) -> u32 {
        let c = self.c;
        let module = format!("{}layers.{layer}.mixer", c.backbone);
        c.attn.emit(
            &mut self.p,
            layer,
            &module,
            hn,
            mixed,
            dep,
            kv_e4m3(),
            &mut fp8_route(c),
        )
    }
    fn kv_write(&mut self, cache: u32, src: u32, heads: u32, dep: u32) -> u32 {
        let hd = self.c.g.head_dim as u32;
        gqa_attention::kv_write(&mut self.p, cache, src, heads, hd, kv_e4m3(), dep)
    }
    /// The backbone MoE of `layer`: its expert and shared-expert linears are NVFP4 (group 16),
    /// the latent projections bf16.
    fn moe(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32) -> u32 {
        let p = format!("{}layers.{layer}.mixer", self.c.backbone);
        let mut want = vec![
            (format!("{p}.experts.0.up_proj"), true),
            (format!("{p}.shared_experts.up_proj"), true),
        ];
        if self.c.moe.geometry().latent.is_some() {
            want.push((format!("{p}.fc1_latent_proj"), false));
        }
        for (m, nvfp4) in want {
            let q = self.c.g.linear_quant(&m);
            assert_eq!(
                matches!(q, NemotronHLinearQuant::Nvfp4 { group: 16 }),
                nvfp4,
                "{m}: {q:?}"
            );
        }
        self.moe_at(&p, hn, mixed, dep)
    }
    /// The MoE of module `p` (a backbone layer's mixer, or the MTP head's).
    fn moe_at(&mut self, p: &str, hn: u32, mixed: u32, dep: u32) -> u32 {
        self.c.moe.emit(&mut self.p, p, hn, mixed, dep)
    }
}

/// One program: a prefill bucket (`prefill`) or a decode rung, adopting `tensors`.
#[allow(clippy::too_many_arguments)]
fn phase(
    c: &Cfg,
    ctx: u32,
    n_cu: u32,
    target: u32,
    rows: u32,
    slots: u32,
    prefill: bool,
    tensors: Option<Vec<packet::devbuild::TensorDecl>>,
) -> Model {
    let mut b = Builder::new(n_cu);
    if let Some(tensors) = tensors {
        b.adopt_tensors(tensors);
    }
    b.set_tensor_dedup(true);
    b.force_uniseg();
    // sm_12x only (run() asserts the target): the W4A4 MoE walks launch on their own entry.
    b.set_w4a4_moe_segments(prefill);
    let ids = b.tensor("in.ids", u64::from(ctx) * 4);
    // The loader derives max_ctx from this input arena extent.
    let pos = b.tensor("in.pos", u64::from(ctx) * 4);
    let kvlen = b.tensor("in.kvlen", u64::from(slots) * 4);
    // The slots' mask, then the step seed word (recurrent_state `step_seed`).
    let active = b.tensor("in.active", (u64::from(slots) + 1) * 4);
    let active = if prefill {
        b.tensor_init(
            "in.prefill_active",
            (0..rows).flat_map(|_| 1i32.to_le_bytes()).collect(),
        )
    } else {
        active
    };
    let mtp_kv = (mtp::enabled() && c.block.is_none()).then(|| mtp::declare_kv(&mut b, c, slots, ctx));
    let mut e = Em {
        p: Program {
            b,
            prefill,
            rows,
            slots,
            ctx,
            active,
            pos,
            kvlen,
            quant: None,
            isa: c.isa,
            packed: prefill && packed_prefill(),
        },
        c,
        pending_residual: None,
    };
    let hidden = c.hidden();
    let vocab = c.g.vocab_size as u32;
    let x = e.p.act("x", hidden);
    let hn = e.p.act("hn", hidden);
    let mixed = e.p.act("mixed", hidden);
    let logits = e
        .p
        .b
        .tensor("act.logits", u64::from(slots) * u64::from(vocab) * 2);
    let amax_blocks = AMAX_BLOCKS.min(n_cu);
    let amax = e
        .p
        .b
        .tensor("act.amax", u64::from(slots) * u64::from(amax_blocks) * 8);
    let mut dep = if c.block.is_some() {
        e.p.b.emit(DevOp::Nop, vec![0], &[], |_| {})
    } else {
        let emb = e.weight(
            &format!("{}embeddings.weight", c.backbone),
            u64::from(vocab) * u64::from(hidden) * 2,
        );
        e.p.b.emit(DevOp::Embed, (0..rows.min(n_cu)).collect(), &[], |d| {
            d.t[..3].copy_from_slice(&[x, emb, ids]);
            d.i[..2].copy_from_slice(&[rows, hidden]);
            d.f[0] = 1.0;
        })
    };
    for (layer, kind) in c.g.layers.iter().enumerate() {
        if !c.in_block(layer) {
            continue;
        }
        dep = e.norm(hn, x, &format!("{}layers.{layer}.norm.weight", c.backbone), dep);
        dep = match kind {
            NemotronHLayer::Mamba => e.mamba(layer, hn, mixed, dep),
            NemotronHLayer::Attention => e.attention(layer, hn, mixed, dep),
            NemotronHLayer::Moe => e.moe(layer, hn, mixed, dep),
        };
        dep = e.residual(x, mixed, dep);
    }
    if c.block.is_some() {
        e.flush_residual(dep);
    } else {
        dep = e.norm(hn, x, &format!("{}norm_f.weight", c.backbone), dep);
        let mtp_reads_ids = mtp_kv.map(|kv| mtp::target_tail(&mut e, hn, kv, ids, slots, dep));
        // lm_head: FP8 per-tensor. Prefill scores the bucket's last row (plowrt moves it to
        // the last real row, `lmhead_sites`).
        let head = c.head.trim_end_matches(".weight").to_string();
        let w = e.weight(&c.head, u64::from(vocab) * u64::from(hidden));
        let scale_name = crate::derived::repeat(&format!("{head}.weight_scale"), vocab);
        let scale = e.p.b.tensor(&scale_name, u64::from(vocab) * 4);
        assert_eq!(
            c.g.linear_quant(&head),
            NemotronHLinearQuant::Fp8,
            "nemotron_h lm_head: expected FP8 per-tensor"
        );
        dep = e.p.emit(
            GemvFp8Args {
                C: logits,
                x: hn,
                W: w,
                w_scale: scale,
                M: if prefill { 1 } else { rows },
                N: vocab,
                K: hidden,
                a_row0: if prefill { rows - 1 } else { 0 },
                ..Default::default()
            },
            &[dep],
        );
        let row_slots = if prefill { 1 } else { rows };
        dep = e
            .p
            .b
            .emit(DevOp::Argmax, (0..amax_blocks).collect(), &[dep], |d| {
                d.t[..2].copy_from_slice(&[amax, logits]);
                d.i[0] = vocab;
                d.i[1] = row_slots;
            });
        let fin_deps: Vec<u32> = std::iter::once(dep).chain(mtp_reads_ids).collect();
        e.p.b.emit(DevOp::ArgmaxFin, vec![0], &fin_deps, |d| {
            d.t[..2].copy_from_slice(&[ids, amax]);
            d.i[0] = amax_blocks;
            d.i[1] = row_slots;
        });
    }
    let tensors = e.p.b.tensors();
    let gen = e.p.b.gen_tensors();
    Model {
        n_cu,
        target,
        tensors,
        progs: vec![e.p.b.finish()],
        kv_row_insts: vec![],
        prog_t: vec![rows],
        gen,
    }
}

/// Prefill buckets (ascending) then one decode program per rung (ascending), every program
/// adopting the tensor table of the one before it (KV, recurrent state and inputs sized at the
/// widest rung).
fn ladder(c: &Cfg, ctx: u32, n_cu: u32, target: u32, buckets: &[u32], rungs: &[u32]) -> Model {
    assert!(!rungs.is_empty() && rungs.windows(2).all(|w| w[0] < w[1]));
    let slots = *rungs.last().unwrap();
    assert!(
        (1..=packet::devbuild::DECODE_RUNG_MAX).contains(&slots),
        "nemotron_h decode rungs must lie in 1..={}",
        packet::devbuild::DECODE_RUNG_MAX
    );
    let mut tensors = None;
    let mut progs = Vec::new();
    let mut last = None;
    let phases = buckets
        .iter()
        .map(|&r| (r, true))
        .chain(rungs.iter().map(|&r| (r, false)));
    for (rows, prefill) in phases {
        if prefill {
            assert!(rows % CHUNK == 0 && rows <= ctx, "prefill bucket {rows}");
        }
        let mut m = phase(c, ctx, n_cu, target, rows, slots, prefill, tensors);
        tensors = Some(std::mem::take(&mut m.tensors));
        progs.push(m.progs.remove(0));
        last = Some(m);
    }
    let mut m = last.unwrap();
    m.tensors = tensors.unwrap();
    m.progs = progs;
    m.prog_t = buckets.iter().chain(rungs).copied().collect();
    m
}

/// The RECURRENT STATE section: every `state.nemo.*` buffer, slot-major over the decode slots.
fn recurrent_state_section(m: &Model, slots: u32) -> packet::devbuild::SectionData {
    crate::recurrent::section(
        m,
        slots,
        &crate::recurrent::Family {
            prefix: "state.nemo.",
            prefill_op: DevOp::Mamba2ChunkPass,
            stateless_prefill_updates: true,
            initstate: None,
            step_seed: true,
            replay: true,
        },
    )
}

/// `PLOW_FP8_KV`: E4M3 KV at vLLM's default unit `k_scale` / `v_scale` (the backbone has no
/// checkpoint scales; one would fail `validate_coverage`), written by `QwenHeadNormRope`
/// (`live_kv::KV_E4M3`) and read by the e4m3 flash arms (`PLOW_NV_KV_E4M3` objects).
fn kv_e4m3() -> bool {
    emit_config::active().fp8_kv
}

/// `PLOW_EMIT_PACKED_PREFILL`: several requests' prompts share one prefill launch.
fn packed_prefill() -> bool {
    emit_config::active().emit_packed_prefill == Some(true)
}

/// The packed-request contract (`plow_asset::packed_prefill`) over the final prefill programs:
/// the KV writers take the per-row slot table (`QwenHeadNormRope` t7), the attention and the
/// Mamba-2 prefill chain the request table (`FlashPrefill` t6; `Mamba2Conv` / `ChunkState` /
/// `ChunkPass` / `ChunkOut` t7). Pad rows are masked (slot -1).
fn packed_section(
    m: &mut Model,
    slots: u32,
    max_rows: u32,
    with_live: bool,
) -> Vec<packet::devbuild::SectionData> {
    // A block without attention declares no KV cache, so no request contract: it runs serial.
    let attends = m.progs.last().is_some_and(|g| {
        g.insts
            .iter()
            .any(|d| d.op == DevOp::FlashDecode as u16 || d.op == DevOp::FlashDecodeFp8 as u16)
    });
    if !attends {
        return Vec::new();
    }
    let mut declare = |name: &str, bytes: u64| {
        assert!(m.tensors.len() < packet::dev::TENSOR_NONE16 as usize);
        m.tensors.push(packet::devbuild::TensorDecl {
            name: name.into(),
            bytes,
            init: None,
        });
        (m.tensors.len() - 1) as u16
    };
    let slot = declare("pf.request.slot", u64::from(max_rows) * 4);
    let request = declare("pf.request.table", (1 + 4 * u64::from(slots)) * 4);
    let (manifest, live) = plow_asset::program::with_model(m, |p| -> Result<_, String> {
        let live = plow_asset::live_kv::emit(p)?;
        let request = plow_asset::packed_prefill::Manifest {
            version: live.version,
            max_request_rows: Some(max_rows),
            stage_rows: None,
            stages: Vec::new(),
            slot,
            request,
            maps: Vec::new(),
            programs: p.programs[..p.prefill_count]
                .iter()
                .map(plow_asset::live_kv::program_digest)
                .collect(),
        };
        request.validate(p, &live)?;
        Ok((request, live))
    })
    .unwrap_or_else(|error| panic!("nemotron_h packed request contract: {error}"));
    let mut out = vec![packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: plow_asset::packed_prefill::SECTION.into(),
        data: serde_json::to_vec(&manifest).expect("packed prefill manifest"),
    }];
    if with_live {
        out.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: plow_asset::live_kv::SECTION.into(),
            data: serde_json::to_vec(&live).expect("serialize LIVE KV manifest"),
        });
    }
    out
}

/// Emit options this route does not lower, refused by name.
fn refuse_routes(ec: &emit_config::EmitConfig) {
    let routes = [
        (ec.prefill_cublaslt, "PLOW_EMIT_PREFILL_CUBLASLT"),
        (ec.decode_cublaslt, "PLOW_EMIT_DECODE_CUBLASLT"),
        (ec.gen_kernels.is_some(), "PLOW_EMIT_GEN_KERNELS"),
        (ec.attn_library, "PLOW_EMIT_ATTN_LIBRARY"),
        (ec.emit_spec && !mtp::enabled(), "PLOW_EMIT_SPEC without PLOW_SPEC_KIND=nemotron_h_mtp"),
        (ec.tma_gemm, "PLOW_TMA_GEMM"),
        (ec.decode_objects.is_some(), "PLOW_EMIT_DECODE_OBJECTS"),
    ];
    let set: Vec<&str> = routes.iter().filter(|r| r.0).map(|r| r.1).collect();
    assert!(set.is_empty(), "nemotron_h does not lower {set:?}");
}

pub(super) fn run(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    block: Option<&str>,
    rope_gen: bool,
    arch: &str,
    gpu: &str,
    verify: Option<&VerifyHook>,
) {
    assert!(
        hwspec::IsaLevel::arch_is_sm12x(arch),
        "nemotron_h device emit targets the sm_120 interpreter (sm_120a, sm_121a); got {arch:?}"
    );
    let arms = kernelcaps::arms::SM120;
    for op in [DevOp::Mamba2Step, DevOp::MoeUpNvfp4] {
        arms.require(op, "tensor parallel", tp)
            .unwrap_or_else(|e| panic!("nemotron_h: {e}"));
    }
    let ec = emit_config::active();
    refuse_routes(ec);
    let rungs = ec.decode_rungs();
    let slots = *rungs.last().expect("decode rungs");
    assert!(ctx >= slots && ctx >= 8192, "nemotron_h needs max_ctx >= 8192 (the always rung)");
    let root: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("nemotron_h config JSON");
    let mut c = Cfg::parse(&root);
    c.isa = hwspec::IsaLevel::from_arch_flag(arch);
    if let Some(spec) = block {
        c.block = Some(crate::block::parse_block(spec, c.g.layers.len()));
    }
    let target = packet::devbuild::gpu_fingerprint(gpu);
    let _target = EmitAmdGuard::set(false);
    let mut buckets = crate::with_always_rung(PREFILL_BUCKETS.to_vec(), ctx);
    // PLOW_PF_LADDER_APPEND rungs up to max_ctx; one above the always rung widens the long-prompt
    // chunk (scratch is sized by the widest).
    if let Some(append) = ec.pf_ladder_append.as_deref() {
        buckets.extend(
            append
                .split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .filter(|&x| x > 0 && x <= ctx && x % CHUNK == 0),
        );
        buckets.sort_unstable();
        buckets.dedup();
    }
    let mut m = ladder(&c, ctx, n_cu, target, &buckets, &rungs);
    // Span verify and the draft programs extend the tensor table: built before anything digests it.
    let replay_section = (ec.span_verify && c.block.is_none()).then(|| {
        mtp::span_verify(&mut m, slots, crate::spec::verify_depth());
        crate::spec::span_verify::replay_section(&m)
    });
    let spec_sections = if mtp::enabled() && c.block.is_none() {
        let shard = mtp::check_checkpoint(dir, &root, &c);
        let sections = mtp::sections(&mut m, &c, ctx, shard);
        mtp::validate_coverage_mtp(dir, &c, &m);
        sections
    } else {
        Vec::new()
    };
    // Static activation scales: superseded by the per-row dynamic quant (prefill) and the bf16
    // GEMV input (decode); covered whenever their linear's weight is bound.
    let paired = [(".input_scale", ".weight")];
    validate_coverage(
        dir,
        &c.backbone,
        &m.tensors
            .iter()
            .flat_map(|t| checkpoint_sources(&t.name))
            .collect::<Vec<_>>(),
        c.block.clone(),
        &[],
        &paired,
        &[],
    )
    .unwrap_or_else(|err| panic!("{err}"));
    if !rope_gen {
        m.bake_gen();
    }
    let mut sections = Vec::new();
    sections.extend(crate::derived::section(&m));
    sections.extend(spec_sections);
    sections.extend(replay_section);
    // The per-tensor FP8 prefill projections (`fp8_proj`) as isolated cuBLASLt segments.
    let lt = crate::dense_cublaslt::apply_prefill(&mut m, &mut sections, arch)
        .expect("nemotron_h prefill projection segments");
    if lt > 0 {
        eprintln!("  cuBLASLt prefill: {lt} FP8 per-tensor projection segments");
    }
    // After every prefill rewrite: the manifest digests the final programs. A full packet pushes
    // its LIVE KV section below; a block packet carries it here.
    if packed_prefill() {
        sections.extend(packed_section(&mut m, slots, *buckets.last().unwrap(), c.block.is_some()));
    }
    let lean = apply_verify_gate(&m, verify);
    // A packed packet's manifest asks for the packed segment pair (`objects.packed_prefill`).
    let man = if packed_prefill() {
        let mut man = manifest::build_for_packet(&m, arch, &lean, &sections);
        // No depth programs either way: the serial packet's default radix sampler keeps the
        // decode and unpacked prefill objects compiled exactly as the serial packet's.
        man.as_object_mut().map(|o| o.remove("sample_radix"));
        man
    } else {
        manifest::build_without_depth(&m, arch, &lean)
    };
    crate::report_dispatch_audit(&man);
    let out = Path::new(out);
    // Per-slot KV commit: 256K x 16 slots of bf16 KV over 8 attention layers is 64 GiB flat.
    if c.block.is_none() {
        let live = plow_asset::program::with_model(&m, plow_asset::live_kv::emit)
            .unwrap_or_else(|error| panic!("compiled LIVE KV geometry: {error}"));
        sections.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: plow_asset::live_kv::SECTION.into(),
            data: serde_json::to_vec(&live).expect("serialize LIVE KV manifest"),
        });
    }
    crate::object_requirements::append(&m, &mut sections)
        .unwrap_or_else(|error| panic!("object requirements: {error}"));
    sections.push(recurrent_state_section(&m, slots));
    if c.block.is_none() {
        sections.push(crate::serve_section::section_for(dir));
        crate::abi_section::append(&m, &mut sections, arch, gpu);
    }
    let bytes = if let Some(range) = c.block.clone() {
        use plow_asset::*;
        let kind = |l: usize| match c.g.layers[l] {
            NemotronHLayer::Mamba => "mamba2",
            NemotronHLayer::Attention => "full_attn",
            NemotronHLayer::Moe => "moe",
        };
        let first_attn = range
            .clone()
            .find(|&l| c.g.layers[l] == NemotronHLayer::Attention);
        let dims = match first_attn {
            Some(_) => BlockDims {
                heads: Some(i64::from(c.g.num_attention_heads)),
                head_dim: Some(c.g.head_dim),
                kv_heads: Some(i64::from(c.g.num_key_value_heads)),
                ..Default::default()
            },
            None => BlockDims {
                heads: Some(i64::from(c.g.mamba_num_heads)),
                head_dim: Some(c.g.mamba_head_dim),
                kv_heads: Some(i64::from(c.g.n_groups)),
                ..Default::default()
            },
        };
        let desc = BlockDescriptor {
            model: dir.to_string_lossy().into_owned(),
            arch: "nemotron_h".into(),
            layer: range.start as u32,
            kind: range.clone().map(|l| kind(l).to_string()).collect(),
            hidden: c.g.hidden_size,
            dtype: "fp8".into(),
            dims,
            dsa_role: None,
            inputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(c.g.hidden_size)],
                dtype: "bf16".into(),
            }],
            outputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(c.g.hidden_size)],
                dtype: "bf16".into(),
            }],
            carried_state: range
                .clone()
                .filter(|&l| c.g.layers[l] != NemotronHLayer::Moe)
                .map(|l| {
                    let attn = c.g.layers[l] == NemotronHLayer::Attention;
                    CarriedState {
                        role: if attn { "kv" } else { "recurrent" }.into(),
                        tensors: if attn {
                            vec![format!("kv.{l}.k"), format!("kv.{l}.v")]
                        } else {
                            vec![format!("state.nemo.{l}.conv"), format!("state.nemo.{l}.ssm")]
                        },
                        layout: if attn { "head_major" } else { "slot_major" }.into(),
                    }
                })
                .collect(),
            weights: BlockWeights {
                mode: "symlink".into(),
                ckpt: dir.to_string_lossy().into_owned(),
                prefix: format!("{}layers.{}.", c.backbone, range.start),
            },
            programs: BlockPrograms {
                prefill_buckets: m.prog_t[..m.prog_t.len() - rungs.len()]
                    .iter()
                    .map(|&x| i64::from(x))
                    .collect(),
                decode_t: i64::from(slots),
            },
        };
        let section =
            crate::block::write_block_descriptor(out.to_str().expect("output path"), &desc);
        sections.push(section);
        m.to_blob_v6(&sections)
    } else {
        m.to_blob_v6(&sections)
    };
    std::fs::write(out, bytes).expect("write nemotron_h blob");
    if !arch.is_empty() {
        crate::write_lean_receipts(out, &lean);
    }
    let mut man = man;
    man["runtime_requires"] = crate::dense_cublaslt::runtime_requires(&m, &sections);
    manifest::write_config_header(&out.with_file_name("plow_config.h"), &man)
        .expect("nemotron_h config header");
    std::fs::write(
        out.with_file_name("build.json"),
        serde_json::to_vec_pretty(&man).unwrap(),
    )
    .expect("nemotron_h build manifest");
    let count = |k: NemotronHLayer| {
        (0..c.g.layers.len())
            .filter(|&l| c.in_block(l) && c.g.layers[l] == k)
            .count()
    };
    eprintln!(
        "nemotron_h: {} mamba / {} attention / {} moe layers, buckets {:?}, rungs {:?} -> {}",
        count(NemotronHLayer::Mamba),
        count(NemotronHLayer::Attention),
        count(NemotronHLayer::Moe),
        buckets,
        rungs,
        out.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An MoE-only block declares no KV cache: the packed emit carries no request contract (the
    /// block runs serial) instead of failing on the live-KV manifest.
    #[test]
    fn packed_section_skips_block_without_attention() {
        let mut b = packet::devbuild::Builder::new(48);
        let x = b.tensor("act.x", 4096);
        let n = b.tensor("act.norm", 4096);
        let z = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        b.emit(DevOp::QwenRmsNorm, vec![0], &[z], |d| {
            d.t[0] = n;
            d.t[1] = x;
            d.i[0] = 1;
            d.i[1] = 2048;
        });
        let gen = b.gen_tensors();
        let p = b.finish();
        let mut m = Model {
            n_cu: 48,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen,
        };
        let tensors = m.tensors.len();
        assert!(packed_section(&mut m, 8, 128, true).is_empty());
        assert_eq!(m.tensors.len(), tensors);
    }
}
