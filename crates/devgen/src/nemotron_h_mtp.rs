//! Nemotron-H MTP draft programs and span verify (`PLOW_EMIT_SPEC`, `PLOW_EMIT_SPAN_VERIFY`,
//! `PLOW_SPEC_KIND=nemotron_h_mtp`).
//!
//! The checkpoint's own `mtp.*` head (`num_nextn_predict_layers = 1`, sublayers
//! `['attention', 'moe']`, embeddings and lm_head shared with the target) drafts as a chain: step
//! `j` embeds `tokens[j]` and joins it with the carried hidden (the target's `norm_f` row at step 0,
//! then the previous step's `final_layernorm` output) through
//! `eh_proj(cat(enorm(e), hnorm(h)))`; the attention sublayer (BF16 q/k/v/o, NoPE GQA) stores its
//! K/V in its own cache (`kv.mtp.{k,v}`) at position `committed - 1 + j`; the latent NVFP4 MoE
//! sublayer is the target's decode MoE emitter over `mtp.layers.1.mixer`; the greedy draft of the
//! target's FP8 lm_head lands in `tokens[j + 1]`. `eh_proj` runs as two GEMVs over its column
//! blocks (`derived_weights` `cols`). Every prefill program fills the drafter cache for the prompt
//! rows (row `i` stores the K/V of `(h_i, embed(x_{i+1}))`).
//!
//! Span verify (plowrt exec/gpu/spec.rs) is a pass over the finished decode rungs: every KV writer
//! and reader binds `in.rowslot`, every Mamba-2 conv / step site its span operands and a
//! per-launch-row snapshot of its buffer (op_mamba_span.cuh), and the recurrent section declares
//! the rollback the runtime commits with.
use super::*;
use crate::spec::chain_drafter::{self, Inputs, Step};
use packet::typed_abi::QwenHeadNormRopeArgs;

pub(super) const FAMILY: &str = "nemotron_h_mtp";

pub(super) fn enabled() -> bool {
    let ec = emit_config::active();
    ec.emit_spec && ec.spec_kind.as_deref() == Some(FAMILY)
}

fn prefix(c: &Cfg) -> String {
    format!("{}mtp.layers.", c.g.weight_prefix)
}

/// The drafter's cache: `[slots][kv heads][ctx][hd]`, the target's attention geometry; bf16, or
/// E4M3 under `PLOW_FP8_KV` ([`kv_scales`]).
pub(super) fn declare_kv(b: &mut Builder, c: &Cfg, slots: u32, ctx: u32) -> (u32, u32) {
    let kd = u64::from(c.g.num_key_value_heads) * c.g.head_dim as u64;
    chain_drafter::declare_kv(b, kd, slots, ctx, if kv_e4m3() { 1 } else { 2 })
}

/// Under `PLOW_FP8_KV` the drafter cache is E4M3 at the attention sublayer's own checkpoint
/// `k_scale` / `v_scale`, which vLLM loads for the MTP layer (`maybe_remap_kv_scale_name`) and
/// applies with `--kv-cache-dtype fp8`: stored `sat_e4m3(x / scale)`, read as FlashInfer XQA's
/// bmm1 / bmm2 scales. Returns the k and v writer scales (f32 `[1]`) and the reader's `{k, v}`.
fn kv_scales(e: &mut Em) -> Option<(u32, u32, u32)> {
    if !kv_e4m3() {
        return None;
    }
    let p = prefix(e.c);
    let (k, v) = (
        format!("{p}0.mixer.k_proj.k_scale"),
        format!("{p}0.mixer.v_proj.v_scale"),
    );
    let pair = crate::derived::concat(&[&k, &v], false);
    Some((e.weight(&k, 4), e.weight(&v, 4), e.weight(&pair, 8)))
}

/// `(name, shape, dtype)` of every `mtp.*` tensor the drafter binds, experts collapsed to `N`.
fn expected(c: &Cfg) -> Vec<(String, Vec<u64>, &'static str)> {
    let g = &c.g;
    let h = g.hidden_size as u64;
    let (qd, kd) = (
        u64::from(g.num_attention_heads) * g.head_dim as u64,
        u64::from(g.num_key_value_heads) * g.head_dim as u64,
    );
    let e = u64::from(g.n_routed_experts);
    // The routed experts' input width: the latent space, or the hidden rows without one.
    let lat = g.moe_latent_size.map_or(h, |l| l as u64);
    let inter = g.moe_intermediate_size as u64;
    let sinter = g.moe_shared_expert_intermediate_size as u64;
    let p = prefix(c);
    let mut v: Vec<(String, Vec<u64>, &'static str)> = vec![
        (format!("{p}0.eh_proj.weight"), vec![h, 2 * h], "BF16"),
        (format!("{p}0.enorm.weight"), vec![h], "BF16"),
        (format!("{p}0.hnorm.weight"), vec![h], "BF16"),
        (format!("{p}0.norm.weight"), vec![h], "BF16"),
        (format!("{p}0.mixer.q_proj.weight"), vec![qd, h], "BF16"),
        (format!("{p}0.mixer.k_proj.weight"), vec![kd, h], "BF16"),
        (format!("{p}0.mixer.v_proj.weight"), vec![kd, h], "BF16"),
        (format!("{p}0.mixer.o_proj.weight"), vec![h, qd], "BF16"),
        (format!("{p}0.mixer.k_proj.k_scale"), vec![1], "F32"),
        (format!("{p}0.mixer.v_proj.v_scale"), vec![1], "F32"),
        (format!("{p}1.norm.weight"), vec![h], "BF16"),
        (format!("{p}1.final_layernorm.weight"), vec![h], "BF16"),
        (format!("{p}1.mixer.gate.weight"), vec![e, h], "BF16"),
        (format!("{p}1.mixer.gate.e_score_correction_bias"), vec![e], "F32"),
    ];
    if g.moe_latent_size.is_some() {
        v.push((format!("{p}1.mixer.fc1_latent_proj.weight"), vec![lat, h], "BF16"));
        v.push((format!("{p}1.mixer.fc2_latent_proj.weight"), vec![h, lat], "BF16"));
    }
    let nvfp4 = |v: &mut Vec<_>, m: String, n: u64, k: u64| {
        v.push((format!("{m}.weight"), vec![n, k / 2], "U8"));
        v.push((format!("{m}.weight_scale"), vec![n, k / 16], "F8_E4M3"));
        v.push((format!("{m}.weight_scale_2"), vec![1], "F32"));
        v.push((format!("{m}.input_scale"), vec![1], "F32"));
    };
    nvfp4(&mut v, format!("{p}1.mixer.experts.N.up_proj"), inter, lat);
    nvfp4(&mut v, format!("{p}1.mixer.experts.N.down_proj"), lat, inter);
    nvfp4(&mut v, format!("{p}1.mixer.shared_experts.up_proj"), sinter, h);
    nvfp4(&mut v, format!("{p}1.mixer.shared_experts.down_proj"), h, sinter);
    v
}

/// The config and checkpoint the drafter assumes, refused by name otherwise. Returns the shard that
/// holds `eh_proj`.
pub(super) fn check_checkpoint(dir: &Path, root: &Value, c: &Cfg) -> String {
    let v = root.get("llm_config").unwrap_or(root);
    assert_eq!(
        c.g.num_nextn_predict_layers, 1,
        "{FAMILY}: exactly one MTP step is supported"
    );
    assert_eq!(
        v["mtp_layers_block_type"],
        serde_json::json!(["attention", "moe"]),
        "{FAMILY}: the drafter lowers the ['attention', 'moe'] MTP pattern"
    );
    assert!(
        !c.g.attention_bias && c.g.sliding_window.is_none(),
        "{FAMILY}: biased or windowed MTP attention"
    );
    let ec = emit_config::active();
    assert!(
        ec.span_verify,
        "{FAMILY}: the drafter's spans verify on the decode rungs; set PLOW_EMIT_SPAN_VERIFY"
    );
    assert!(
        ec.spec_draft_dir.is_none() && ec.spec_shard_sha256.is_none() && ec.spec_draft_fp8.is_none(),
        "{FAMILY}: the drafter is the checkpoint's own mtp.* head; unset PLOW_SPEC_DRAFT_DIR, \
         PLOW_SPEC_SHARD_SHA256 and PLOW_SPEC_DRAFT_FP8"
    );
    let p = prefix(c);
    let index: Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json"))
            .unwrap_or_else(|e| panic!("{FAMILY}: model.safetensors.index.json: {e}")),
    )
    .unwrap_or_else(|e| panic!("{FAMILY}: index: {e}"));
    let map = index["weight_map"].as_object().expect("weight_map");
    let mut shards: Vec<&str> = map
        .iter()
        .filter(|(k, _)| k.starts_with(&p))
        .filter_map(|(_, s)| s.as_str())
        .collect();
    shards.sort_unstable();
    shards.dedup();
    let collapse = |name: &str| -> String {
        let Some(rest) = name.strip_prefix(&format!("{p}1.mixer.experts.")) else {
            return name.to_string();
        };
        let (x, tail) = rest.split_once('.').expect("expert tensor");
        assert!(x.parse::<u32>().is_ok(), "{name}");
        format!("{p}1.mixer.experts.N.{tail}")
    };
    let mut found: std::collections::BTreeMap<String, (Vec<u64>, Value, usize)> =
        Default::default();
    let mut eh_shard = None;
    for shard in &shards {
        let (header, _) = crate::mtp::safetensors_header(&dir.join(shard));
        for (name, t) in header.into_iter().filter(|(n, _)| n.starts_with(&p)) {
            let shape: Vec<u64> = t["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            if name == format!("{p}0.eh_proj.weight") {
                eh_shard = Some(shard.to_string());
            }
            let entry = found
                .entry(collapse(&name))
                .or_insert((shape.clone(), t["dtype"].clone(), 0));
            assert!(
                entry.0 == shape && entry.1 == t["dtype"],
                "{FAMILY}: {name} differs from its sibling experts"
            );
            entry.2 += 1;
        }
    }
    let want = expected(c);
    for (name, shape, dtype) in &want {
        let (got, dt, count) = found
            .get(name)
            .unwrap_or_else(|| panic!("{FAMILY}: checkpoint lacks {name}"));
        assert!(
            got == shape && dt == dtype,
            "{FAMILY}: {name} is {got:?} {dt}, expected {shape:?} {dtype}"
        );
        let copies = if name.contains(".experts.N.") {
            c.g.n_routed_experts as usize
        } else {
            1
        };
        assert_eq!(*count, copies, "{FAMILY}: {name}: {count} tensors");
    }
    assert_eq!(
        found.len(),
        want.len(),
        "{FAMILY}: the checkpoint's mtp.* tensors differ from the attention + latent MoE head"
    );
    eh_shard.expect("eh_proj shard")
}

impl Em<'_> {
    /// RMSNorm `out = norm(src [+= radd]) * gamma` over this program's rows (no gamma offset).
    fn mtp_norm(&mut self, out: u32, src: u32, gamma: &str, radd: u32, dep: u32) -> u32 {
        let gamma = self.weight(gamma, u64::from(self.c.hidden()) * 2);
        self.rms_norm(out, src, gamma, radd, dep)
    }

    /// `x = eh_proj(cat(enorm(e), hnorm(h)))`: the two column blocks of eh_proj and their sum.
    fn mtp_fuse(&mut self, x: u32, e: u32, h: u32, dep_e: u32, dep_h: u32) -> u32 {
        let p = prefix(self.c);
        let hidden = self.c.hidden();
        let hh = u64::from(hidden);
        let (en, hn, x2) = (
            self.p.act("mtp.en", hidden),
            self.p.act("mtp.hn", hidden),
            self.p.act("mtp.x2", hidden),
        );
        let c_en = self.mtp_norm(en, e, &format!("{p}0.enorm.weight"), TENSOR_NONE, dep_e);
        let c_hn = self.mtp_norm(hn, h, &format!("{p}0.hnorm.weight"), TENSOR_NONE, dep_h);
        let eh = format!("{p}0.eh_proj.weight");
        let c_x1 = self.bf16_proj(x, en, &crate::derived::cols(&eh, 0, hh), hidden, hidden, c_en);
        let c_x2 =
            self.bf16_proj(x2, hn, &crate::derived::cols(&eh, hh, 2 * hh), hidden, hidden, c_hn);
        let n = self.p.rows * hidden;
        self.p.emit(
            ResidualArgs {
                out: x,
                a: x,
                b: x2,
                n,
                scale: 1.0,
                ..Default::default()
            },
            &[c_x1, c_x2],
        )
    }

    /// A draft program's K/V store of `src` into `cache` at `position[r]` of slot `slot[r]`;
    /// `scale`: the E4M3 cache's per-tensor scale ([`kv_scales`]), else `TENSOR_NONE` (bf16).
    fn mtp_kv_store(
        &mut self,
        cache: u32,
        src: u32,
        position: u32,
        slot: u32,
        scale: u32,
        dep: u32,
    ) -> u32 {
        use plow_asset::live_kv::{KV_E4M3, KV_SCALE, LAYOUT_HEAD_MAJOR};
        let args = QwenHeadNormRopeArgs {
            out: cache,
            x: src,
            cos: scale,
            pos: position,
            rowslot: slot,
            H: self.c.g.num_key_value_heads,
            D: self.c.g.head_dim as u32,
            rows: self.p.rows,
            ctx: self.p.ctx,
            layout: LAYOUT_HEAD_MAJOR
                | if scale == TENSOR_NONE {
                    0
                } else {
                    KV_E4M3 | KV_SCALE
                },
            ..Default::default()
        };
        self.p.emit(args, &[dep])
    }
    /// A target program's drafter-cache store: the target's KV writer, at the drafter's scale.
    fn mtp_kv_write(&mut self, cache: u32, src: u32, scale: Option<u32>, dep: u32) -> u32 {
        let (heads, hd) = (self.c.g.num_key_value_heads, self.c.g.head_dim as u32);
        match scale {
            Some(s) => gqa_attention::kv_write_scaled(&mut self.p, cache, src, heads, hd, s, dep),
            None => self.kv_write(cache, src, heads, dep),
        }
    }
}

/// The drafter cache rows a target program owns: row `r` stores the attention sublayer's K/V of
/// `(h_r, embed(ids[r + 1]))` at its own position, through the target's own KV writer (prefill:
/// the bucket's slot; decode: the row's slot, `in.rowslot` under span verify). In a verify span
/// row `r + 1` holds the token after row `r`, so every committed row's entry is the exact one;
/// the last row of a span (and of a plain step or a prompt bucket) has no next id in the launch
/// and is rewritten by the next draft's step 0 (an inner prompt-chunk end, and the prompt's last
/// row, which the first verify commits before any draft, stay approximate: draft quality only).
/// `slots` sizes the decode shift table (one table for every rung, so the rungs stay one shape).
/// Returns the instruction that reads `ids`, which the program's ArgmaxFin must follow.
pub(super) fn target_tail(
    e: &mut Em,
    hn: u32,
    kv: (u32, u32),
    ids: u32,
    slots: u32,
    dep: u32,
) -> u32 {
    let c = e.c;
    let p = prefix(c);
    let (rows, h) = (e.p.rows, c.hidden());
    let kd = c.g.num_key_value_heads * c.g.head_dim as u32;
    let (name, live) = if e.p.prefill {
        (format!("in.mtp.shift.{rows}"), rows)
    } else {
        ("in.mtp.dshift".to_string(), slots)
    };
    let shift = match e.p.b.tensors().iter().position(|t| t.name == name) {
        Some(t) => t as u32,
        None => e.p.b.tensor_init(
            &name,
            (0..live)
                .flat_map(|r| ((r + 1).min(live - 1) as i32).to_le_bytes())
                .collect(),
        ),
    };
    let (ps, x, xn, kg, vg) = (
        e.p.act("mtp.pes", h),
        e.p.act("mtp.x", h),
        e.p.act("mtp.xn", h),
        e.p.act("mtp.kg", kd),
        e.p.act("mtp.vg", kd),
    );
    // The decode gather reads up to row `slots - 1` of the embedded rows.
    let pe = e
        .p
        .b
        .tensor("act.mtp.pe", u64::from(live.max(rows)) * u64::from(h) * 2);
    let emb = e.weight(
        &format!("{}embeddings.weight", c.backbone),
        c.g.vocab_size as u64 * u64::from(h) * 2,
    );
    let blocks: Vec<u32> = (0..rows.min(e.p.b.n_cu())).collect();
    let c_e = e.p.b.emit(DevOp::Embed, blocks.clone(), &[dep], |d| {
        d.t[..3].copy_from_slice(&[pe, emb, ids]);
        d.i[..2].copy_from_slice(&[rows, h]);
        d.f[0] = 1.0;
    });
    let c_s = e.p.b.emit(DevOp::RowGather, blocks, &[c_e], |d| {
        d.t[..3].copy_from_slice(&[ps, pe, shift]);
        d.i[..3].copy_from_slice(&[rows, h, live]);
    });
    let c_x = e.mtp_fuse(x, ps, hn, c_s, dep);
    let c_xn = e.mtp_norm(xn, x, &format!("{p}0.norm.weight"), TENSOR_NONE, c_x);
    let sc = kv_scales(e);
    let c_k = e.bf16_proj(kg, xn, &format!("{p}0.mixer.k_proj.weight"), kd, h, c_xn);
    let c_kw = e.mtp_kv_write(kv.0, kg, sc.map(|s| s.0), c_k);
    let c_v = e.bf16_proj(vg, xn, &format!("{p}0.mixer.v_proj.weight"), kd, h, c_kw);
    e.mtp_kv_write(kv.1, vg, sc.map(|s| s.1), c_v);
    c_e
}

/// One `(rows, k)` draft program over the packet's tensor table `tensors`; returns the program
/// and the table it extends.
#[allow(clippy::too_many_arguments)]
fn program(
    c: &Cfg,
    s: &chain_drafter::Shared,
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
    let (b, inp) = chain_drafter::builder(tensors, n_cu, rows, kmax);
    let mut e = Em {
        p: Program {
            b,
            prefill: false,
            rows,
            slots: rows,
            ctx,
            active: TENSOR_NONE,
            pos: TENSOR_NONE,
            kvlen: TENSOR_NONE,
            quant: None,
            isa: c.isa,
            packed: false,
        },
        c,
        pending_residual: None,
    };
    let g = &c.g;
    let p = prefix(c);
    let (h, heads, kvh, hd) = (
        c.hidden(),
        g.num_attention_heads,
        g.num_key_value_heads,
        g.head_dim as u32,
    );
    let (qd, kd) = (heads * hd, kvh * hd);
    let vocab = g.vocab_size as u32;
    let r = u64::from(rows);
    let emb = e.weight(
        &format!("{}embeddings.weight", c.backbone),
        u64::from(vocab) * u64::from(h) * 2,
    );
    let head = e.weight(&c.head, u64::from(vocab) * u64::from(h));
    let head_scale_name = crate::derived::repeat(
        &format!("{}.weight_scale", c.head.trim_end_matches(".weight")),
        vocab,
    );
    let head_scale = e.p.b.tensor(&head_scale_name, u64::from(vocab) * 4);
    let emb_t = e.p.act("mtp.e", h);
    let hid = e.p.act("mtp.h", h);
    let x = e.p.act("mtp.x", h);
    let xn = e.p.act("mtp.xn", h);
    let xf = e.p.act("mtp.xf", h);
    let q = e.p.act("mtp.q", qd);
    let kg = e.p.act("mtp.kg", kd);
    let vg = e.p.act("mtp.vg", kd);
    let at = e.p.act("mtp.at", qd);
    let ao = e.p.act("mtp.ao", h);
    let mo = e.p.act("mtp.mo", h);
    let logits = e.p.act("mtp.logits", vocab);
    let amax = e
        .p
        .b
        .tensor("act.mtp.amax", r * u64::from(AMAX_BLOCKS.min(n_cu)) * 8);
    // E4M3: the GQA-16 item (one per KV head and split), split like the target's decode.
    let fallback = if kv_e4m3() {
        crate::balanced_attention_ns(rows, heads, kvh, n_cu, crate::hybrid_decode_gf(heads / kvh, hd, true))
    } else {
        n_cu.div_ceil(heads / 2).max(1)
    };
    let ns = crate::attention_decode_ns(rows, heads, kvh, n_cu, fallback);
    let sc = kv_scales(&mut e);
    let (ks, vs, kvs) = sc.unwrap_or((TENSOR_NONE, TENSOR_NONE, TENSOR_NONE));
    let opart = e
        .p
        .b
        .tensor("act.mtp.opart", r * u64::from(qd) * u64::from(ns) * 4);
    let mlpart = e
        .p
        .b
        .tensor("act.mtp.mlpart", r * u64::from(heads) * u64::from(ns) * 8);
    let step = Step::new(rows, h, n_cu);
    let mut prev: Option<u32> = None;
    let mut hsrc = hid;
    let mut hdep = None;
    for j in 0..k as usize {
        let (position, kv_len) = inp.steps[j];
        let c_e = step.embed(&mut e.p.b, prev, emb_t, emb, inp.tokens[j]);
        let c_h = match hdep {
            Some(dep) => dep,
            None => step.gather_hidden(&mut e.p.b, hid, s, inp.hidden_row),
        };
        let c_x = e.mtp_fuse(x, emb_t, hsrc, c_e, c_h);
        let c_xn = e.mtp_norm(xn, x, &format!("{p}0.norm.weight"), TENSOR_NONE, c_x);
        let c_q = e.bf16_proj(q, xn, &format!("{p}0.mixer.q_proj.weight"), qd, h, c_xn);
        let c_k = e.bf16_proj(kg, xn, &format!("{p}0.mixer.k_proj.weight"), kd, h, c_q);
        let c_kw = e.mtp_kv_store(s.kv.0, kg, position, inp.slot, ks, c_k);
        let c_v = e.bf16_proj(vg, xn, &format!("{p}0.mixer.v_proj.weight"), kd, h, c_kw);
        let c_vw = e.mtp_kv_store(s.kv.1, vg, position, inp.slot, vs, c_v);
        let c_mg = step.attention(
            &mut e.p.b,
            c_vw,
            (heads, kvh, hd),
            (opart, mlpart, ns),
            q,
            s,
            kv_len,
            inp.slot,
            ctx,
            at,
            kvs,
        );
        let c_o = e.bf16_proj(ao, at, &format!("{p}0.mixer.o_proj.weight"), h, qd, c_mg);
        // MoE sublayer: residual = x + attention; norm1 feeds the MoE.
        let c_n1 = e.mtp_norm(xn, x, &format!("{p}1.norm.weight"), ao, c_o);
        let c_m = e.moe_at(&format!("{p}1.mixer"), xn, mo, c_n1);
        let c_f = e.mtp_norm(xf, x, &format!("{p}1.final_layernorm.weight"), mo, c_m);
        let c_lm = e.p.emit(
            GemvFp8Args {
                C: logits,
                x: xf,
                W: head,
                w_scale: head_scale,
                M: rows,
                N: vocab,
                K: h,
                ..Default::default()
            },
            &[c_f],
        );
        prev = Some(step.greedy(&mut e.p.b, c_lm, amax, logits, vocab, inp.tokens[j + 1]));
        // The next step's hidden is this step's final_layernorm output.
        hsrc = xf;
        hdep = Some(c_f);
    }
    let tensors = e.p.b.tensors();
    (e.p.b.finish(), tensors, inp)
}

/// Every `(rows, k)` draft program appended to `m`'s tensor table, plus `spec.json`.
pub(super) fn sections(
    m: &mut Model,
    c: &Cfg,
    ctx: u32,
    shard: String,
) -> Vec<packet::devbuild::SectionData> {
    let shared = chain_drafter::Shared::find(m, FAMILY);
    let n_cu = m.n_cu;
    chain_drafter::sections(m, FAMILY, &shared, shard, |tensors, rows, k, kmax| {
        program(c, &shared, tensors, n_cu, ctx, rows, k, kmax)
    })
}

/// The checkpoint tensors the drafter binds, for the `mtp.*` coverage gate.
pub(super) fn validate_coverage_mtp(dir: &Path, c: &Cfg, m: &Model) {
    let declared: Vec<String> = m
        .tensors
        .iter()
        .flat_map(|t| checkpoint_sources(&t.name))
        .collect();
    // input_scale: the decode MoE is W4A16; k_scale / v_scale: the peer's FP8 KV scales (bound
    // under PLOW_FP8_KV, else the drafter cache is bf16). Each is covered only while its linear's
    // weight is bound.
    let paired = [
        (".input_scale", ".weight"),
        (".k_scale", ".weight"),
        (".v_scale", ".weight"),
    ];
    validate_coverage(dir, &prefix(c), &declared, None, &[], &paired, &[])
        .unwrap_or_else(|err| panic!("{FAMILY}: {err}"));
}

/// Span verify on the decode rungs: bind `in.rowslot` on every KV writer and reader and the span
/// operands (the replay table and a step record per buffer, `recurrent_state.verify.replay`) on
/// every Mamba-2 conv and step site.
pub(super) fn span_verify(m: &mut Model, slots: u32, depth: u32) {
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    for d in m.progs[dec_lo..].iter().flat_map(|p| &p.insts) {
        match DevOp::from_u16(d.op) {
            Some(DevOp::Mamba2Conv) => assert_eq!(d.i[4], 0, "a prefill conv on a decode rung"),
            Some(DevOp::Mamba2Step) => {
                let state = &m.tensors[d.t[6] as usize];
                let elems = u64::from(d.i[0]) * u64::from(d.i[1]) * u64::from(d.i[2]);
                let esize = state.bytes / u64::from(slots) / elems;
                let declared = d.i[5] & 0xFF;
                assert!(
                    (declared == 0 && esize == 4) || u64::from(declared) == esize,
                    "Mamba2Step state element bytes {declared} vs a {esize}-byte buffer"
                );
            }
            _ => {}
        }
    }
    let snapshotted = crate::spec::span_verify::bind_decode_rungs(m, slots, depth);
    assert!(snapshotted > 0, "{FAMILY}: no Mamba-2 decode site to verify");
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::spec_draft;

    fn fixture() -> Value {
        let p = "language_model.backbone.layers";
        let mut quantized = serde_json::Map::new();
        let fp8 = serde_json::json!({"quant_algo": "FP8"});
        let nvfp4 = serde_json::json!({"quant_algo": "NVFP4", "group_size": 16});
        for m in ["0.mixer.in_proj", "0.mixer.out_proj"]
            .into_iter()
            .map(String::from)
            .chain(["q", "k", "v", "o"].map(|x| format!("2.mixer.{x}_proj")))
        {
            quantized.insert(format!("{p}.{m}"), fp8.clone());
        }
        for l in [1, 3] {
            for m in ["shared_experts.up_proj", "shared_experts.down_proj"] {
                quantized.insert(format!("{p}.{l}.mixer.{m}"), nvfp4.clone());
            }
            for e in 0..128 {
                for m in ["up_proj", "down_proj"] {
                    quantized.insert(format!("{p}.{l}.mixer.experts.{e}.{m}"), nvfp4.clone());
                }
            }
        }
        quantized.insert("language_model.lm_head".into(), fp8);
        let mut llm = serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 256, "hidden_size": 256, "layer_norm_epsilon": 1e-5,
            "layers_block_type": ["mamba", "moe", "attention", "moe"],
            "mtp_layers_block_type": ["attention", "moe"],
            "tie_word_embeddings": false, "num_nextn_predict_layers": 1,
            "mamba_num_heads": 128, "mamba_head_dim": 64, "ssm_state_size": 128,
            "n_groups": 8, "conv_kernel": 4, "chunk_size": 128,
            "mamba_hidden_act": "silu", "use_conv_bias": true, "use_bias": false,
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 128,
            "attention_bias": false, "sliding_window": null
        });
        let moe = serde_json::json!({
            "n_routed_experts": 128, "num_experts_per_tok": 4,
            "moe_intermediate_size": 128, "moe_latent_size": 64,
            "moe_shared_expert_intermediate_size": 256, "n_shared_experts": 1,
            "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "routed_scaling_factor": 5.0, "mlp_hidden_act": "relu2", "mlp_bias": false,
            "dtype": "bfloat16"
        });
        llm.as_object_mut()
            .unwrap()
            .extend(moe.as_object().unwrap().clone());
        serde_json::json!({
            "architectures": ["NemotronH_Omni_Reasoning_V3"],
            "model_type": "nemotron_h_omni",
            "dtype": "bfloat16",
            "vision_config": {"model_type": "radio"},
            "quantization_config": {
                "quant_method": "modelopt_mixed",
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
                "quantized_layers": quantized
            },
            "llm_config": llm
        })
    }

    struct SpecCfg(emit_config::EmitConfig);
    impl SpecCfg {
        fn set(edit: impl FnOnce(&mut emit_config::EmitConfig)) -> Self {
            let saved = emit_config::active().clone();
            let mut cfg = saved.clone();
            edit(&mut cfg);
            emit_config::install(cfg);
            SpecCfg(saved)
        }
    }
    impl Drop for SpecCfg {
        fn drop(&mut self) {
            emit_config::install(self.0.clone());
        }
    }

    fn handle(m: &Model, name: &str) -> u32 {
        m.tensors
            .iter()
            .position(|t| t.name == name)
            .unwrap_or_else(|| panic!("no {name}")) as u32
    }

    // A geometry the interpreter has no arm for is refused by naming the op and the arm.
    #[test]
    #[should_panic(expected = "Mamba2Step has no arm for head_dim = 80 (implemented: one of [64])")]
    fn components_refuse_a_missing_arm() {
        let mut root = fixture();
        root["llm_config"]["mamba_head_dim"] = 80.into();
        Cfg::parse(&root);
    }

    // Without a latent space the routed experts read the hidden rows directly: no fc1/fc2.
    #[test]
    fn moe_without_latent_routes_hidden_rows() {
        let _env = crate::test_env::env_guard();
        let mut root = fixture();
        root["llm_config"]
            .as_object_mut()
            .unwrap()
            .remove("moe_latent_size");
        let c = Cfg::parse(&root);
        let m = ladder(&c, 8192, 48, 0, &[128], &[1, 4]);
        assert!(!m.tensors.iter().any(|t| t.name.contains("latent_proj")));
        for p in &m.progs {
            let up = p
                .insts
                .iter()
                .find(|d| d.op == DevOp::MoeUpNvfp4 as u16 || d.op == DevOp::MoeUpNvfp4Pf as u16)
                .unwrap();
            // K = hidden (256 in the fixture), not a latent width.
            assert_eq!(up.i[3], 256);
        }
    }

    // Plain packets are untouched; a span-verify packet binds the row map on every KV writer and
    // reader of a decode rung, the span operands and a per-row snapshot on every Mamba-2
    // site, and declares the rollback; the prefill buckets fill the drafter cache.
    #[test]
    fn span_verify_binds_mamba_spans_and_declares_rollback() {
        let _env = crate::test_env::env_guard();
        let c = Cfg::parse(&fixture());
        let plain = ladder(&c, 8192, 48, 0, &[128], &[1, 4]);
        assert!(!plain.tensors.iter().any(|t| t.name == "in.rowslot"
            || t.name.ends_with(".rec")
            || t.name.starts_with("kv.mtp")));
        let section = recurrent_state_section(&plain, 4);
        let manifest: plow_asset::recurrent_state::Manifest =
            serde_json::from_slice(&section.data).unwrap();
        assert!(manifest.verify.is_none());

        let _cfg = SpecCfg::set(|cfg| {
            cfg.span_verify = true;
            cfg.emit_spec = true;
            cfg.spec_kind = Some(FAMILY.into());
            cfg.spec_kmax = Some(2);
            cfg.spec_rows = Some("1,2,4".into());
        });
        let mut m = ladder(&c, 8192, 48, 0, &[128], &[1, 4]);
        span_verify(&mut m, 4, 2);
        let rowslot = handle(&m, "in.rowslot");
        assert_eq!(m.tensors[rowslot as usize].bytes, 4 * 4);
        let table = handle(&m, "in.spec.replay");
        assert_eq!(m.tensors[table as usize].bytes, 6 * 4);
        let conv_rec = handle(&m, "state.nemo.0.conv.rec");
        let ssm_rec = handle(&m, "state.nemo.0.ssm.rec");
        assert!(!m.tensors.iter().any(|t| t.name.ends_with(".snap")));
        let bytes = |h: u32| m.tensors[h as usize].bytes;
        let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
        let (kk, kv) = (handle(&m, "kv.mtp.k"), handle(&m, "kv.mtp.v"));
        for (pi, p) in m.progs.iter().enumerate() {
            let decode = pi >= dec_lo;
            for d in &p.insts {
                match DevOp::from_u16(d.op) {
                    Some(DevOp::QwenHeadNormRope) if d.i[4] != 0 && decode => {
                        assert_eq!(d.t[7], rowslot, "cache writer without the row map");
                    }
                    Some(DevOp::FlashDecode) => assert_eq!(d.t[6], rowslot),
                    Some(DevOp::Mamba2Conv) if decode && d.t[4] == handle(&m, "state.nemo.0.conv") => {
                        assert_eq!((d.t[6], d.t[7], d.i[5]), (rowslot, conv_rec, 2 | table << 16));
                        let row = (u64::from(d.i[0]) * 2).next_multiple_of(16);
                        assert_eq!(bytes(conv_rec), 2 * 4 * row);
                    }
                    Some(DevOp::Mamba2Conv) if decode => {
                        assert_eq!((d.t[6], d.i[5]), (rowslot, 2 | table << 16));
                    }
                    Some(DevOp::Mamba2Conv) => {
                        assert_eq!((d.t[6], d.t[7]), (TENSOR_NONE, TENSOR_NONE))
                    }
                    Some(DevOp::Mamba2Step) => {
                        assert_eq!(
                            ((d.i[5] >> 8) & 0xFF, d.i[5] >> 16, d.i[6] >> 16),
                            (2, table, rowslot)
                        );
                        if d.t[6] == handle(&m, "state.nemo.0.ssm") {
                            assert_eq!(d.i[7] >> 16, ssm_rec);
                            let (h, p, n, g) = (d.i[0], d.i[1], d.i[2], d.i[3]);
                            let row = (u64::from(h * p + g * n) * 2 + u64::from(h) * 4)
                                .next_multiple_of(16);
                            assert_eq!(bytes(ssm_rec), 2 * 4 * row);
                        }
                    }
                    _ => {}
                }
            }
            // Every prefill bucket and decode rung stores the drafter K/V of its rows, from the
            // shifted embedding of the next row's id.
            let writes = |t: u32| {
                p.insts
                    .iter()
                    .any(|d| d.op == DevOp::QwenHeadNormRope as u16 && d.t[0] == t)
            };
            assert!(writes(kk) && writes(kv));
            let gather = p
                .insts
                .iter()
                .find(|d| d.op == DevOp::RowGather as u16)
                .unwrap();
            let (shift, live) = if decode {
                ("in.mtp.dshift", 4)
            } else {
                ("in.mtp.shift.128", 128)
            };
            assert_eq!((gather.t[2], gather.i[2]), (handle(&m, shift), live));
        }
        let section = recurrent_state_section(&m, 4);
        let manifest: plow_asset::recurrent_state::Manifest =
            serde_json::from_slice(&section.data).unwrap();
        let verify = manifest.verify.as_ref().unwrap();
        let replay = verify.replay.as_ref().unwrap();
        assert_eq!(
            (verify.depth, manifest.buffers.len(), verify.snapshots.len(), replay.records.len()),
            (2, 2, 0, 2)
        );
        assert_eq!(replay.table as u32, table);
        for r in &replay.records {
            assert_eq!(
                m.tensors[r.tensor as usize].name,
                format!("{}.rec", m.tensors[r.buffer as usize].name)
            );
        }
        let replay_prog = crate::spec::span_verify::replay_section(&m);
        let parsed = plow_asset::aux_program::parse(&replay_prog.data, 48, m.tensors.len()).unwrap();
        let sites: Vec<_> = parsed.programs[0]
            .insts
            .iter()
            .map(|d| (d.op, if d.op == DevOp::Mamba2Conv as u16 { d.i[2] } else { d.i[4] }))
            .collect();
        assert_eq!(
            sites,
            [(DevOp::Mamba2Conv as u16, 0), (DevOp::Mamba2Step as u16, 0)]
        );
        let live = plow_asset::program::with_model(&m, plow_asset::live_kv::emit)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(live.row_slot, Some(rowslot as u16));
        let man = manifest::build(&m, "sm_121a", &LeanReport::skipped("fixture"));
        assert_eq!(man["tuning"]["row_slot_twin"], 1);
    }

    // The chain drafter: per step, the target's final-norm row (step 0) or the previous step's
    // final_layernorm output joins the embedded token through eh_proj's two column blocks; the
    // attention sublayer stores into the drafter cache at its own slot and position; the MoE
    // sublayer binds the head's own experts; the FP8 lm_head writes the next token.
    #[test]
    fn draft_programs_chain_steps_through_the_drafter_cache() {
        let _env = crate::test_env::env_guard();
        let c = Cfg::parse(&fixture());
        let _cfg = SpecCfg::set(|cfg| {
            cfg.span_verify = true;
            cfg.emit_spec = true;
            cfg.spec_kind = Some(FAMILY.into());
            cfg.spec_kmax = Some(3);
            cfg.spec_rows = Some("1,2,4,8".into());
        });
        let mut m = ladder(&c, 8192, 48, 0, &[128], &[1, 4]);
        span_verify(&mut m, 4, 3);
        let sections = sections(&mut m, &c, 8192, "model-00088-of-00088.safetensors".into());
        let manifest: spec_draft::Manifest = serde_json::from_slice(&sections[0].data).unwrap();
        assert_eq!((manifest.family.as_str(), manifest.kmax), (FAMILY, 3));
        // Rows 8 exceeds the 4 decode slots.
        assert_eq!(manifest.programs.len(), 3 * 3);
        let parsed =
            plow_asset::aux_program::parse(&sections[1].data, 48, m.tensors.len()).unwrap();
        let (kk, kv) = (handle(&m, "kv.mtp.k"), handle(&m, "kv.mtp.v"));
        let slot = u32::from(manifest.tensors.slot);
        let hidden = handle(&m, "act.hn");
        let gate = handle(&m, "language_model.mtp.layers.1.mixer.gate.weight");
        let eh: Vec<u32> = m
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.name.contains("eh_proj.weight[:,"))
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(eh.len(), 2);
        for (e, p) in manifest.programs.iter().zip(&parsed.programs) {
            let count = |op: DevOp| p.insts.iter().filter(|d| d.op == op as u16).count() as u32;
            assert_eq!(count(DevOp::ArgmaxFin), e.k);
            assert_eq!(count(DevOp::FlashDecode), e.k);
            assert_eq!(count(DevOp::MoeRouterLogitsF32), e.k);
            assert_eq!(count(DevOp::GemvFp8), e.k);
            let gathers: Vec<_> = p
                .insts
                .iter()
                .filter(|d| d.op == DevOp::RowGather as u16)
                .collect();
            assert_eq!(gathers.len(), 1);
            assert_eq!(
                (u32::from(gathers[0].t[1]), gathers[0].i[0], gathers[0].i[2]),
                (hidden, e.rows, 4)
            );
            for d in &p.insts {
                if d.op == DevOp::QwenHeadNormRope as u16 {
                    assert!([kk, kv].contains(&u32::from(d.t[0])));
                    assert_eq!((u32::from(d.t[7]), d.i[3], d.i[4]), (slot, e.rows, 8192));
                }
                if d.op == DevOp::FlashDecode as u16 {
                    assert_eq!(
                        (u32::from(d.t[3]), u32::from(d.t[4]), u32::from(d.t[6])),
                        (kk, kv, slot)
                    );
                }
                if d.op == DevOp::MoeRouterLogitsF32 as u16 {
                    assert_eq!(u32::from(d.t[2]), gate);
                }
                if d.op == DevOp::Gemv as u16 && eh.contains(&u32::from(d.t[2])) {
                    assert_eq!((d.i[1], d.i[2]), (256, 256));
                }
            }
        }
        let derived = crate::derived::section(&m).unwrap();
        let derived: plow_asset::derived_weights::DerivedWeights =
            serde_json::from_slice(&derived.data).unwrap();
        let cols: Vec<_> = derived
            .tensors
            .iter()
            .filter(|t| t.sources == ["language_model.mtp.layers.0.eh_proj.weight"])
            .filter_map(|t| t.cols)
            .collect();
        assert_eq!(cols, [[0, 256], [256, 512]]);
    }

    // Packed prefill on: every single-layer block emits; only an attention block carries the
    // packed request and LIVE KV sections (Mamba-only and MoE-only blocks declare no cache).
    #[test]
    fn single_layer_blocks_emit_with_packed_prefill() {
        let _env = crate::test_env::env_guard();
        let _cfg = SpecCfg::set(|c| c.emit_packed_prefill = Some(true));
        for (range, attends) in [(0..1, false), (1..2, false), (2..3, true)] {
            let mut c = Cfg::parse(&fixture());
            c.block = Some(range.clone());
            let mut m = ladder(&c, 8192, 48, 0, &[128], &[1, 4, 8]);
            let sections = packed_section(&mut m, 8, 128, true);
            assert_eq!(sections.len(), if attends { 2 } else { 0 }, "block {range:?}");
        }
    }

    #[test]
    fn blocks_without_attention_size_the_decode_objects_like_the_full_packet() {
        let _env = crate::test_env::env_guard();
        let lean = LeanReport::skipped("fixture");
        let full = Cfg::parse(&fixture());
        let man = manifest::build(&ladder(&full, 8192, 48, 0, &[128], &[1, 4, 8]), "sm_121a", &lean);
        for range in [0..1, 1..2] {
            let mut c = Cfg::parse(&fixture());
            c.block = Some(range.clone());
            let m = ladder(&c, 8192, 48, 0, &[128], &[1, 4, 8]);
            let insts = m.progs.iter().flat_map(|p| &p.insts);
            assert!(!insts.into_iter().any(|d| d.op == DevOp::FlashDecode as u16));
            let blk = manifest::build(&m, "sm_121a", &lean);
            assert_eq!(blk["shapes"]["decode_batch"], 8, "block {range:?}");
            assert_eq!(blk["tuning"]["gv_mm_max"], man["tuning"]["gv_mm_max"], "block {range:?}");
        }
    }
}
