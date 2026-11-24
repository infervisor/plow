//! Qwen3.5 MTP draft programs (`PLOW_EMIT_SPEC`, `PLOW_SPEC_KIND=qwen3_5_mtp`).
//!
//! The checkpoint's own `mtp.*` layer (`mtp_num_hidden_layers = 1`, shared embeddings and head)
//! drafts as a chain: step `j` embeds `tokens[j]`, joins it with the carried hidden (the target's
//! final-norm row at step 0, then the previous step's `mtp.norm` output) through
//! `fc(cat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)))`, runs one full-attention decoder
//! layer whose K/V it stores in its own cache (`kv.mtp.{k,v}`) at position `committed - 1 + j`,
//! and writes the greedy draft to `tokens[j + 1]`. `fc` runs as two GEMVs over the column blocks
//! of `mtp.fc.weight` (`derived_weights` `cols`). Every prefill program fills the drafter cache
//! for the prompt rows (the MTP input of row `i` is `(h_i, embed(x_{i+1}))`), so the first drafts
//! attend the whole prompt. The target decode rungs verify spans with recurrent rollback
//! (`recurrent_state.verify`).
use super::*;
use crate::spec::chain_drafter::{self, Inputs, Step};

pub(super) const FAMILY: &str = "qwen3_5_mtp";

pub(super) fn enabled() -> bool {
    let ec = emit_config::active();
    ec.emit_spec && ec.spec_kind.as_deref() == Some(FAMILY)
}

/// The drafter's cache: `[slots][kv heads][ctx][hd]` bf16 each, the target's full-layer geometry.
pub(super) fn declare_kv(b: &mut Builder, c: &Config, slots: u32, ctx: u32) -> (u32, u32) {
    chain_drafter::declare_kv(b, u64::from(c.kv_heads * c.hd), slots, ctx, 2)
}

/// `(name, shape)` of every `mtp.*` tensor the drafter binds.
fn expected(c: &Config) -> Vec<(String, Vec<u64>)> {
    let (h, i) = (u64::from(c.hidden), u64::from(c.inter));
    let (qd, kd, hd) = (
        u64::from(c.heads * c.hd),
        u64::from(c.kv_heads * c.hd),
        u64::from(c.hd),
    );
    let p = "mtp.layers.0.";
    [
        ("mtp.fc.weight".to_string(), vec![h, 2 * h]),
        ("mtp.norm.weight".into(), vec![h]),
        ("mtp.pre_fc_norm_embedding.weight".into(), vec![h]),
        ("mtp.pre_fc_norm_hidden.weight".into(), vec![h]),
        (format!("{p}input_layernorm.weight"), vec![h]),
        (format!("{p}post_attention_layernorm.weight"), vec![h]),
        (format!("{p}mlp.gate_proj.weight"), vec![i, h]),
        (format!("{p}mlp.up_proj.weight"), vec![i, h]),
        (format!("{p}mlp.down_proj.weight"), vec![h, i]),
        (format!("{p}self_attn.q_proj.weight"), vec![2 * qd, h]),
        (format!("{p}self_attn.k_proj.weight"), vec![kd, h]),
        (format!("{p}self_attn.v_proj.weight"), vec![kd, h]),
        (format!("{p}self_attn.o_proj.weight"), vec![h, qd]),
        (format!("{p}self_attn.q_norm.weight"), vec![hd]),
        (format!("{p}self_attn.k_norm.weight"), vec![hd]),
    ]
    .into_iter()
    .collect()
}

/// The config and checkpoint the drafter assumes, refused by name otherwise. Returns the shard
/// that holds `mtp.fc.weight`.
pub(super) fn check_checkpoint(dir: &Path, root: &Value, c: &Config) -> String {
    let v = root.get("text_config").unwrap_or(root);
    assert_eq!(
        v["mtp_num_hidden_layers"].as_u64(),
        Some(1),
        "{FAMILY}: exactly one MTP layer is supported"
    );
    assert_ne!(
        v["mtp_use_dedicated_embeddings"], true,
        "{FAMILY}: dedicated MTP embeddings are unsupported (the drafter shares the target's)"
    );
    let ec = emit_config::active();
    assert!(
        ec.spec_draft_dir.is_none()
            && ec.spec_shard_sha256.is_none()
            && ec.spec_draft_fp8.is_none(),
        "{FAMILY}: the drafter is the checkpoint's own mtp.* layer; unset PLOW_SPEC_DRAFT_DIR, \
         PLOW_SPEC_SHARD_SHA256 and PLOW_SPEC_DRAFT_FP8"
    );
    let mut found = std::collections::BTreeMap::new();
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    for shard in &shards {
        let (header, _) = crate::mtp::safetensors_header(shard);
        for (name, t) in header.into_iter().filter(|(n, _)| n.starts_with("mtp.")) {
            let shape: Vec<u64> = t["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            let file = shard.file_name().unwrap().to_string_lossy().into_owned();
            found.insert(name, (shape, t["dtype"].clone(), file));
        }
    }
    let want = expected(c);
    for (name, shape) in &want {
        let (got, dtype, _) = found
            .get(name)
            .unwrap_or_else(|| panic!("{FAMILY}: checkpoint lacks {name}"));
        assert!(
            got == shape && dtype == "BF16",
            "{FAMILY}: {name} is {got:?} {dtype}, expected {shape:?} BF16"
        );
    }
    assert_eq!(
        found.len(),
        want.len(),
        "{FAMILY}: the checkpoint's mtp.* tensors differ from the one-layer dense drafter"
    );
    found["mtp.fc.weight"].2.clone()
}

/// The handles a draft program shares with the target programs, beyond the chain's own.
pub(super) struct Shared {
    pub chain: chain_drafter::Shared,
    pub emb: u32,
    pub head: u32,
    pub cos: u32,
    pub sin: u32,
}

impl Shared {
    pub(super) fn find(m: &Model, c: &Config) -> Self {
        let handle = |name: &str| chain_drafter::handle(m, FAMILY, name);
        let head = if c.tied {
            format!("{}embed_tokens.weight", c.prefix)
        } else {
            "lm_head.weight".into()
        };
        Shared {
            chain: chain_drafter::Shared::find(m, FAMILY),
            emb: handle(&format!("{}embed_tokens.weight", c.prefix)),
            head: handle(&head),
            cos: handle("in.cos_full"),
            sin: handle("in.sin_full"),
        }
    }
}

/// One `(rows, k)` draft program over the packet's tensor table `tensors`; returns the program
/// and the table it extends.
pub(super) fn program(
    c: &Config,
    s: &Shared,
    tensors: Vec<packet::devbuild::TensorDecl>,
    n_cu: u32,
    ctx: u32,
    rows: u32,
    k: u32,
    kmax: u32,
) -> (
    packet::devbuild::Program,
    Vec<packet::devbuild::TensorDecl>,
    Inputs,
) {
    let (mut b, inp) = chain_drafter::builder(tensors, n_cu, rows, kmax);
    let (h, i) = (c.hidden, c.inter);
    let (qd, kd) = (c.heads * c.hd, c.kv_heads * c.hd);
    let r = u64::from(rows);
    let act = |b: &mut Builder, name: &str, width: u32| {
        b.tensor(&format!("act.mtp.{name}"), r * u64::from(width) * 2)
    };
    let w = |b: &mut Builder, name: &str, elems: u64| b.tensor(name, elems * 2);
    let p = "mtp.layers.0.";
    let fc_e = b.tensor(
        &derive_cols("mtp.fc.weight", 0, u64::from(h)),
        u64::from(h) * u64::from(h) * 2,
    );
    let fc_h = b.tensor(
        &derive_cols("mtp.fc.weight", u64::from(h), 2 * u64::from(h)),
        u64::from(h) * u64::from(h) * 2,
    );
    let g_e = w(&mut b, "mtp.pre_fc_norm_embedding.weight", h.into());
    let g_h = w(&mut b, "mtp.pre_fc_norm_hidden.weight", h.into());
    let g_in = w(&mut b, &format!("{p}input_layernorm.weight"), h.into());
    let g_post = w(
        &mut b,
        &format!("{p}post_attention_layernorm.weight"),
        h.into(),
    );
    let g_out = w(&mut b, "mtp.norm.weight", h.into());
    let wq = w(
        &mut b,
        &format!("{p}self_attn.q_proj.weight"),
        2 * u64::from(qd) * u64::from(h),
    );
    let wk = w(
        &mut b,
        &format!("{p}self_attn.k_proj.weight"),
        u64::from(kd) * u64::from(h),
    );
    let wv = w(
        &mut b,
        &format!("{p}self_attn.v_proj.weight"),
        u64::from(kd) * u64::from(h),
    );
    let wo = w(
        &mut b,
        &format!("{p}self_attn.o_proj.weight"),
        u64::from(h) * u64::from(qd),
    );
    let qn = w(&mut b, &format!("{p}self_attn.q_norm.weight"), c.hd.into());
    let kn = w(&mut b, &format!("{p}self_attn.k_norm.weight"), c.hd.into());
    let wg = w(
        &mut b,
        &format!("{p}mlp.gate_proj.weight"),
        u64::from(i) * u64::from(h),
    );
    let wu = w(
        &mut b,
        &format!("{p}mlp.up_proj.weight"),
        u64::from(i) * u64::from(h),
    );
    let wd = w(
        &mut b,
        &format!("{p}mlp.down_proj.weight"),
        u64::from(h) * u64::from(i),
    );
    let e = act(&mut b, "e", h);
    let hid = act(&mut b, "h", h);
    let en = act(&mut b, "en", h);
    let hn = act(&mut b, "hn", h);
    let x1 = act(&mut b, "x1", h);
    let x2 = act(&mut b, "x2", h);
    let x = act(&mut b, "x", h);
    let xn = act(&mut b, "xn", h);
    let xf = act(&mut b, "xf", h);
    let qgate = act(&mut b, "qgate", 2 * qd);
    let qg = act(&mut b, "qg", qd);
    let gate = act(&mut b, "gate", qd);
    let q = act(&mut b, "q", qd);
    let kg = act(&mut b, "kg", kd);
    let vg = act(&mut b, "vg", kd);
    let at = act(&mut b, "at", qd);
    let og = act(&mut b, "og", h);
    let gt = act(&mut b, "gt", i);
    let ut = act(&mut b, "ut", i);
    let fu = act(&mut b, "fu", i);
    let dg = act(&mut b, "dg", h);
    let logits = act(&mut b, "logits", c.vocab);
    let amax = b.tensor("act.mtp.amax", r * u64::from(AMAX_BLOCKS.min(n_cu)) * 8);
    let fallback = n_cu.div_ceil(c.heads / 2).max(1);
    let ns = emit_config::active()
        .ns_full_abs
        .unwrap_or_else(|| crate::attention_decode_ns(rows, c.heads, c.kv_heads, n_cu, fallback));
    let opart = b.tensor("act.mtp.opart", r * u64::from(qd) * u64::from(ns) * 4);
    let mlpart = b.tensor("act.mtp.mlpart", r * u64::from(c.heads) * u64::from(ns) * 8);
    let step = Step::new(rows, h, n_cu);
    let rblocks = step.rblocks.clone();
    let gemv = |b: &mut Builder, dep: u32, out: u32, src: u32, wt: u32, n: u32, kin: u32| {
        b.emit(DevOp::Gemv, b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[out, src, wt]);
            d.i[..3].copy_from_slice(&[rows, n, kin]);
        })
    };
    let norm = |b: &mut Builder, dep: u32, out: u32, src: u32, gamma: u32| {
        b.emit(DevOp::QwenRmsNorm, rblocks.clone(), &[dep], |d| {
            d.t[..5].copy_from_slice(&[out, src, gamma, TENSOR_NONE, TENSOR_NONE]);
            d.i[..2].copy_from_slice(&[h, rows]);
            d.f[0] = c.eps;
            d.f[1] = 1.0;
        })
    };
    let residual = |b: &mut Builder, deps: &[u32], out: u32, a: u32, bb: u32| {
        b.emit(DevOp::Residual, b.all(), deps, |d| {
            d.t[..3].copy_from_slice(&[out, a, bb]);
            d.i[0] = rows * h;
            d.f[0] = 1.0;
        })
    };
    #[allow(clippy::too_many_arguments)]
    let headnorm = |b: &mut Builder,
                    dep: u32,
                    out: u32,
                    src: u32,
                    gamma: u32,
                    heads: u32,
                    cache: bool,
                    rope: bool,
                    position: u32| {
        b.emit(DevOp::QwenHeadNormRope, b.all(), &[dep], |d| {
            d.t[..7].copy_from_slice(&[
                out,
                src,
                gamma,
                if rope { s.cos } else { TENSOR_NONE },
                if rope { s.sin } else { TENSOR_NONE },
                position,
                TENSOR_NONE,
            ]);
            d.i[..6].copy_from_slice(&[
                heads,
                c.hd,
                if rope { c.rotary } else { 0 },
                rows,
                if cache { ctx } else { 0 },
                rope as u32,
            ]);
            if cache {
                // The per-row slot table: row r stores into slot `slot[r]` at its own position.
                d.t[7] = inp.slot;
            }
            d.i[7] = plow_asset::live_kv::LAYOUT_HEAD_MAJOR;
            d.f[0] = c.eps;
            d.f[1] = if rope { 1.0 } else { 0.0 };
        })
    };
    let mut prev: Option<u32> = None;
    let mut hsrc = hid;
    let mut hdep = None;
    for j in 0..k as usize {
        let (position, kv_len) = inp.steps[j];
        let c_e = step.embed(&mut b, prev, e, s.emb, inp.tokens[j]);
        let c_h = match hdep {
            Some(dep) => dep,
            None => step.gather_hidden(&mut b, hid, &s.chain, inp.hidden_row),
        };
        let c_en = norm(&mut b, c_e, en, e, g_e);
        let c_hn = norm(&mut b, c_h, hn, hsrc, g_h);
        let c_x1 = gemv(&mut b, c_en, x1, en, fc_e, h, h);
        let c_x2 = gemv(&mut b, c_hn, x2, hn, fc_h, h, h);
        let c_x = residual(&mut b, &[c_x1, c_x2], x, x1, x2);
        let c_xn = norm(&mut b, c_x, xn, x, g_in);
        let c_q = gemv(&mut b, c_xn, qgate, xn, wq, 2 * qd, h);
        let c_split = b.emit(DevOp::QwenQGateSplit, b.all(), &[c_q], |d| {
            d.t[..4].copy_from_slice(&[qg, gate, qgate, TENSOR_NONE]);
            d.i[..3].copy_from_slice(&[c.heads, c.hd, rows]);
        });
        let c_qn = headnorm(&mut b, c_split, q, qg, qn, c.heads, false, true, position);
        let c_k = gemv(&mut b, c_qn, kg, xn, wk, kd, h);
        let c_kn = headnorm(
            &mut b, c_k, s.chain.kv.0, kg, kn, c.kv_heads, true, true, position,
        );
        let c_v = gemv(&mut b, c_kn, vg, xn, wv, kd, h);
        let c_vn = headnorm(
            &mut b,
            c_v,
            s.chain.kv.1,
            vg,
            TENSOR_NONE,
            c.kv_heads,
            true,
            false,
            position,
        );
        let c_mg = step.attention(
            &mut b,
            c_vn,
            (c.heads, c.kv_heads, c.hd),
            (opart, mlpart, ns),
            q,
            &s.chain,
            kv_len,
            inp.slot,
            ctx,
            at,
            TENSOR_NONE,
        );
        let c_sg = b.emit(DevOp::QwenSigmoidGate, b.all(), &[c_mg], |d| {
            d.t[..4].copy_from_slice(&[at, at, gate, TENSOR_NONE]);
            d.i[..2].copy_from_slice(&[qd, rows]);
        });
        let c_o = gemv(&mut b, c_sg, og, at, wo, h, qd);
        let c_r1 = residual(&mut b, &[c_o], x, x, og);
        let c_pn = norm(&mut b, c_r1, xn, x, g_post);
        let c_g = gemv(&mut b, c_pn, gt, xn, wg, i, h);
        let c_u = gemv(&mut b, c_g, ut, xn, wu, i, h);
        let c_glu = b.emit(DevOp::Glu, b.all(), &[c_u], |d| {
            d.t[..3].copy_from_slice(&[fu, gt, ut]);
            d.i[..2].copy_from_slice(&[rows * i, 1]);
        });
        let c_d = gemv(&mut b, c_glu, dg, fu, wd, h, i);
        let c_r2 = residual(&mut b, &[c_d], x, x, dg);
        let c_f = norm(&mut b, c_r2, xf, x, g_out);
        let c_lm = gemv(&mut b, c_f, logits, xf, s.head, c.vocab, h);
        prev = Some(step.greedy(&mut b, c_lm, amax, logits, c.vocab, inp.tokens[j + 1]));
        // The next step's hidden is this step's `mtp.norm` output.
        hsrc = xf;
        hdep = Some(c_f);
    }
    let tensors = b.tensors();
    (b.finish(), tensors, inp)
}

/// Every `(rows, k)` draft program appended to `m`'s tensor table, plus `spec.json`.
pub(super) fn sections(
    m: &mut Model,
    c: &Config,
    ctx: u32,
    shard: String,
) -> Vec<packet::devbuild::SectionData> {
    let shared = Shared::find(m, c);
    let n_cu = m.n_cu;
    chain_drafter::sections(m, FAMILY, &shared.chain, shard, |tensors, rows, k, kmax| {
        program(c, &shared, tensors, n_cu, ctx, rows, k, kmax)
    })
}

impl Emitter<'_> {
    /// The drafter's prompt rows (prefill only): row `i` of the bucket stores the MTP layer's K/V
    /// of `(h_i, embed(ids[i + 1]))` at its own position. The bucket's last row has no next id
    /// in the launch: the prompt's last row is rewritten by the first draft step, an inner chunk
    /// end stays approximate (draft quality only). Returns the instruction that reads `ids`,
    /// which the bucket's ArgmaxFin must follow, and the last one.
    pub(super) fn mtp_prefill_tail(
        &mut self,
        hn: u32,
        kv: (u32, u32),
        ids: u32,
        dep: u32,
    ) -> (u32, u32) {
        let c = self.c;
        let rows = self.batch;
        let h = c.hidden;
        let kd = c.kv_heads * c.hd;
        let shift = self.b.tensor_init(
            &format!("in.mtp.shift.{rows}"),
            (0..rows)
                .flat_map(|r| ((r + 1).min(rows - 1) as i32).to_le_bytes())
                .collect(),
        );
        let e = self.act("mtp.pe", h);
        let es = self.act("mtp.pes", h);
        let en = self.act("mtp.pen", h);
        let hh = self.act("mtp.phn", h);
        let x1 = self.act("mtp.px1", h);
        let x2 = self.act("mtp.px2", h);
        let xn = self.act("mtp.pxn", h);
        let kg = self.act("mtp.pkg", kd);
        let vg = self.act("mtp.pvg", kd);
        let emb = self.weight(
            &format!("{}embed_tokens.weight", c.prefix),
            c.vocab as u64 * h as u64,
        );
        let blocks: Vec<u32> = (0..rows.min(self.b.n_cu())).collect();
        let c_e = self.b.emit(DevOp::Embed, blocks.clone(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[e, emb, ids]);
            d.i[..2].copy_from_slice(&[rows, h]);
            d.f[0] = 1.0;
        });
        let c_s = self.b.emit(DevOp::RowGather, blocks, &[c_e], |d| {
            d.t[..3].copy_from_slice(&[es, e, shift]);
            d.i[..3].copy_from_slice(&[rows, h, rows]);
        });
        let norm = |e: &mut Self, dep: u32, out: u32, src: u32, gamma: &str| {
            let gamma = e.weight(gamma, h as u64);
            e.b.emit(
                DevOp::QwenRmsNorm,
                (0..rows.min(e.b.n_cu())).collect(),
                &[dep],
                |d| {
                    d.t[..5].copy_from_slice(&[out, src, gamma, e.active, TENSOR_NONE]);
                    d.i[..2].copy_from_slice(&[h, rows]);
                    d.f[0] = c.eps;
                    d.f[1] = 1.0;
                },
            )
        };
        let c_en = norm(self, c_s, en, es, "mtp.pre_fc_norm_embedding.weight");
        let c_hn = norm(self, c_en, hh, hn, "mtp.pre_fc_norm_hidden.weight");
        let fc_e = derive_cols("mtp.fc.weight", 0, u64::from(h));
        let fc_h = derive_cols("mtp.fc.weight", u64::from(h), 2 * u64::from(h));
        let c_x1 = self.bf16_proj(x1, en, &fc_e, h, h, c_hn);
        let c_x2 = self.bf16_proj(x2, hh, &fc_h, h, h, c_x1);
        let c_x = self.b.emit(DevOp::Residual, self.b.all(), &[c_x2], |d| {
            d.t[..3].copy_from_slice(&[x1, x1, x2]);
            d.i[0] = rows * h;
            d.f[0] = 1.0;
        });
        let c_xn = norm(self, c_x, xn, x1, "mtp.layers.0.input_layernorm.weight");
        let c_k = self.bf16_proj(kg, xn, "mtp.layers.0.self_attn.k_proj.weight", kd, h, c_xn);
        let kn = self.weight("mtp.layers.0.self_attn.k_norm.weight", c.hd as u64);
        let c_kn = self.headnorm(kv.0, kg, kn, c.kv_heads, true, true, c_k);
        let c_v = self.bf16_proj(vg, xn, "mtp.layers.0.self_attn.v_proj.weight", kd, h, c_kn);
        let c_vn = self.headnorm(kv.1, vg, TENSOR_NONE, c.kv_heads, true, false, c_v);
        (c_e, c_vn)
    }

    /// A BF16 projection outside the FP8 route (the drafter's weights stay BF16).
    fn bf16_proj(&mut self, out: u32, src: u32, name: &str, n: u32, k: u32, dep: u32) -> u32 {
        let w = self.weight(name, n as u64 * k as u64);
        let op = if self.prefill {
            pick_tile(
                self.batch,
                n,
                k,
                self.b.n_cu(),
                kernelcaps::QuantScheme::None,
            )
        } else {
            DevOp::Gemv
        };
        self.b.emit(op, self.b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[out, src, w]);
            d.t[5] = TENSOR_NONE;
            d.i[..3].copy_from_slice(&[self.batch, n, k]);
        })
    }
}
