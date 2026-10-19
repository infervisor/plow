//! Fish S2's per-frame depth stage as auxiliary decode programs (`fish_depth.json` + programs).
//!
//! One unrolled program per decode rung of `rows` rows, launched after the backbone's decode
//! program (or after prefill, with `in.fish.hrow` naming the prompt's terminal row): sample the
//! backbone class from its restricted head (`SampleFishBf16`, RAS on), then run the depth
//! transformer over positions `0..P` on a per-row KV that every frame overwrites from position 0
//! — position 0 the gathered final-norm hidden, position `j` the embedding of codebook `j - 1`'s
//! code — sampling codebook `j` from position `j`'s head. Class and codes land in
//! `in.fish.dcodes[row][0..P]`, which the next decode step's `EmbedFrameBf16` reads (prefill reads
//! the host's `in.frame_codes` prompt rows instead); class 4096 (stop) selects the depth table's
//! zero row, so a stop frame runs with defined inputs. A row with `top_k` 0 samples nothing and
//! keeps its codes; the rest of its work lands in per-frame scratch, so a prompt's first frame
//! runs at any width covering its slot (its logits copied to that row, `hrow` naming the prompt's
//! terminal row).

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Program, SectionData, SECT_METADATA};
use plow_asset::fish_depth as manifest;

use crate::config::{Cfg, FishDepth};
use crate::Tn;

const BF16: u64 = 2;
const I32: u64 = 4;
const F32: u64 = 4;
/// Backbone sampler RAS window (Fish `inference.py`: the last 10 row-0 tokens).
const RAS_WINDOW: u32 = 10;

struct LayerW {
    in_ln: u32,
    q: u32,
    k: u32,
    v: u32,
    o: u32,
    post_ln: u32,
    gate: u32,
    up: u32,
    down: u32,
}

pub(crate) struct Tensors {
    d: FishDepth,
    classes: u32,
    params: u32,
    top_k: u32,
    seed: u32,
    step: u32,
    hist: u32,
    hrow: u32,
    pos: Vec<u32>,
    len: Vec<u32>,
    codes: Vec<u32>,
    x: u32,
    xn: u32,
    qg: u32,
    q: u32,
    kg: u32,
    vg: u32,
    kc: Vec<u32>,
    vc: Vec<u32>,
    opart: u32,
    mlpart: u32,
    at: u32,
    og: u32,
    fu: u32,
    dg: u32,
    logits: u32,
    embed: u32,
    head: u32,
    norm: u32,
    layers: Vec<LayerW>,
}

pub(crate) fn declare(b: &mut Builder, c: &Cfg, dbatch: u32) -> Tensors {
    let f = c.frame.as_ref().expect("fish depth needs the frame embedding");
    let d = f.depth.clone();
    assert!(
        d.positions == f.codebooks && d.head_dim == c.hd_full && f64::from(d.theta) == c.theta_full,
        "fish depth: positions == codebooks, and the backbone's full RoPE tables (head_dim, theta)"
    );
    let (r, h) = (u64::from(dbatch), u64::from(d.hidden));
    let (qd, kd) = (u64::from(d.heads * d.head_dim), u64::from(d.kv_heads * d.head_dim));
    let classes = c.out_vocab();
    let w = |b: &mut Builder, s: &str, n: u64| b.tensor(&format!("fish.depth.{s}"), n * BF16);
    let a = |b: &mut Builder, s: &str, n: u64| b.tensor(&format!("act.fish.{s}"), n);
    let konst = |b: &mut Builder, name: String, v: u32| {
        b.tensor_init(&name, (0..dbatch).flat_map(|_| v.to_le_bytes()).collect())
    };
    let kv = u64::from(d.kv_heads * d.positions * d.head_dim) * r * BF16;
    Tensors {
        classes,
        params: b.tensor("in.fish.params", r * 4 * F32),
        top_k: b.tensor("in.fish.top_k", r * I32),
        seed: b.tensor("in.fish.seed", r * 8),
        step: b.tensor("in.fish.step", r * I32),
        hist: b.tensor("in.fish.hist", r * u64::from(RAS_WINDOW) * I32),
        hrow: b.tensor("in.fish.hrow", r * I32),
        pos: (0..d.positions)
            .map(|j| konst(b, format!("c.fish.pos.{j}"), j))
            .collect(),
        len: (0..d.positions)
            .map(|j| konst(b, format!("c.fish.len.{j}"), j + 1))
            .collect(),
        codes: (0..d.positions)
            .map(|j| a(b, &format!("code.{j}"), r * I32))
            .collect(),
        x: a(b, "x", r * h * BF16),
        xn: a(b, "xn", r * h * BF16),
        qg: a(b, "qg", r * qd * BF16),
        q: a(b, "q", r * qd * BF16),
        kg: a(b, "kg", r * kd * BF16),
        vg: a(b, "vg", r * kd * BF16),
        kc: (0..d.layers).map(|l| a(b, &format!("k.{l}"), kv)).collect(),
        vc: (0..d.layers).map(|l| a(b, &format!("v.{l}"), kv)).collect(),
        opart: a(b, "opart", r * qd * F32),
        mlpart: a(b, "mlpart", r * u64::from(d.heads) * 2 * F32),
        at: a(b, "at", r * qd * BF16),
        og: a(b, "og", r * h * BF16),
        fu: a(b, "fu", r * u64::from(d.inter) * BF16),
        dg: a(b, "dg", r * h * BF16),
        logits: a(b, "logits", r * u64::from(d_classes(f)) * BF16),
        // One row past the codebook: the stop class embeds as zeros (scripts/tts/fish_prep.py).
        embed: w(b, "embed.weight", u64::from(d_classes(f) + 1) * h),
        head: w(b, "head.weight", u64::from(d_classes(f)) * h),
        norm: w(b, "norm.weight", h),
        layers: (0..d.layers)
            .map(|l| {
                let p = format!("layers.{l}.");
                let i = u64::from(d.inter);
                LayerW {
                    in_ln: w(b, &format!("{p}input_layernorm.weight"), h),
                    q: w(b, &format!("{p}self_attn.q_proj.weight"), qd * h),
                    k: w(b, &format!("{p}self_attn.k_proj.weight"), kd * h),
                    v: w(b, &format!("{p}self_attn.v_proj.weight"), kd * h),
                    o: w(b, &format!("{p}self_attn.o_proj.weight"), h * qd),
                    post_ln: w(b, &format!("{p}post_attention_layernorm.weight"), h),
                    gate: w(b, &format!("{p}mlp.gate_proj.weight"), i * h),
                    up: w(b, &format!("{p}mlp.up_proj.weight"), i * h),
                    down: w(b, &format!("{p}mlp.down_proj.weight"), h * i),
                }
            })
            .collect(),
        d,
    }
}

/// Classes of the depth head: one codebook's codes.
fn d_classes(f: &crate::config::FrameEmbed) -> u32 {
    f.cb_size
}

fn program(c: &Cfg, n: &Tn, t: &Tensors, tensors: &[packet::devbuild::TensorDecl], n_cu: u32, rows: u32, hidden_rows: u32) -> Program {
    let f = c.frame.as_ref().unwrap();
    let d = &t.d;
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.to_vec());
    b.force_uniseg();
    let all: Vec<u32> = (0..n_cu).collect();
    let span = |units: u32| -> Vec<u32> { (0..units.clamp(1, n_cu)).collect() };
    let rset = span(rows);
    let (qd, kd) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
    let sample = |b: &mut Builder, dep: &[u32], logits: u32, vocab: u32, j: u32, ras: bool| {
        b.emit(DevOp::SampleFishBf16, rset.clone(), dep, |i| {
            i.t[0] = n.frame_dcodes;
            i.t[1] = logits;
            i.t[2] = t.params;
            i.t[3] = t.top_k;
            i.t[4] = t.seed;
            i.t[5] = t.step;
            i.t[6] = if ras { t.hist } else { TENSOR_NONE };
            i.t[7] = t.codes[j as usize];
            i.i = [rows, vocab, f.codebooks, j, j, if ras { RAS_WINDOW } else { 0 }, u32::from(ras), 0];
            i.j[0] = f.cb_size;
        })
    };
    let gemv = |b: &mut Builder, dep: &[u32], out: u32, x: u32, w: u32, nout: u32, kin: u32| {
        b.emit(DevOp::Gemv, all.clone(), dep, |i| {
            i.t[0] = out;
            i.t[1] = x;
            i.t[2] = w;
            i.i[0] = rows;
            i.i[1] = nout;
            i.i[2] = kin;
        })
    };
    let c0 = sample(&mut b, &[], n.logits, t.classes, 0, true);
    let g = b.emit(DevOp::RowGather, rset.clone(), &[], |i| {
        i.t[0] = t.x;
        i.t[1] = n.hn;
        i.t[2] = t.hrow;
        i.i[0] = rows;
        i.i[1] = c.hidden;
        i.i[2] = hidden_rows;
    });
    let mut prev = vec![g];
    let mut code_dep = c0;
    for j in 0..d.positions {
        if j > 0 {
            prev = vec![b.emit(DevOp::Embed, rset.clone(), &[code_dep], |i| {
                i.t[0] = t.x;
                i.t[1] = t.embed;
                i.t[2] = t.codes[j as usize - 1];
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = 1.0;
            })];
        }
        let mut dep = b.emit(DevOp::RmsNorm, rset.clone(), &prev, |i| {
            i.t[0] = t.xn;
            i.t[1] = t.x;
            i.t[2] = t.layers[0].in_ln;
            i.i[0] = rows;
            i.i[1] = d.hidden;
            i.f[0] = d.eps;
        });
        for (l, w) in t.layers.iter().enumerate() {
            let c_qkv = b.emit(DevOp::GemvQkv, all.clone(), &[dep], |i| {
                i.t[..7].copy_from_slice(&[t.qg, t.xn, w.q, t.kg, w.k, t.vg, w.v]);
                i.i[..5].copy_from_slice(&[rows, qd, d.hidden, kd, kd]);
            });
            let hnr = |b: &mut Builder, out: u32, x: u32, heads: u32, rope: bool, kv: bool| {
                b.emit(DevOp::HeadNormRope, span((rows * heads).div_ceil(8)), &[c_qkv], |i| {
                    i.t[0] = out;
                    i.t[1] = x;
                    if rope {
                        i.t[3] = n.cos_f;
                        i.t[4] = n.sin_f;
                    }
                    i.t[5] = t.pos[j as usize];
                    i.i[..7].copy_from_slice(&[rows, heads, d.head_dim, 0, 1, 0, if kv { rows } else { 0 }]);
                    i.f[0] = d.eps;
                    if kv {
                        // Per-row ring of `positions` rows written at pos[row] = j (n_batch_kv armed
                        // at every width: the legacy arm addresses by out_row0 + row).
                        i.j[0] = d.positions;
                        i.j[1] = u32::MAX;
                    }
                })
            };
            let c_q = hnr(&mut b, t.q, t.qg, d.heads, true, false);
            let c_k = hnr(&mut b, t.kc[l], t.kg, d.kv_heads, true, true);
            let c_v = hnr(&mut b, t.vc[l], t.vg, d.kv_heads, false, true);
            let c_fa = b.emit(DevOp::FlashDecode, all.clone(), &[c_q, c_k, c_v], |i| {
                i.t[..6].copy_from_slice(&[t.opart, t.mlpart, t.q, t.kc[l], t.vc[l], t.len[j as usize]]);
                i.i = [rows, d.heads, d.kv_heads, d.positions, 0, 1, d.head_dim, u32::MAX];
                i.f[0] = 1.0 / (d.head_dim as f32).sqrt();
                if rows > 1 {
                    i.j[0] = rows * d.kv_heads * d.positions;
                }
            });
            let c_mg = b.emit(DevOp::FlashMerge, span(rows * d.heads), &[c_fa], |i| {
                i.t[..3].copy_from_slice(&[t.at, t.opart, t.mlpart]);
                i.i[..4].copy_from_slice(&[rows, d.heads, 1, d.head_dim]);
            });
            let c_o = gemv(&mut b, &[c_mg], t.og, t.at, w.o, d.hidden, qd);
            let c_r1 = b.emit(DevOp::AddNorm, rset.clone(), &[c_o], |i| {
                i.t[..5].copy_from_slice(&[t.xn, t.x, t.x, t.og, w.post_ln]);
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = d.eps;
            });
            let c_gu = b.emit(DevOp::GemvGlu, all.clone(), &[c_r1], |i| {
                i.t[0] = t.fu;
                i.t[1] = t.xn;
                i.t[2] = w.gate;
                i.t[5] = w.up;
                i.i[0] = rows;
                i.i[1] = d.inter;
                i.i[2] = d.hidden;
                i.i[5] = 1;
            });
            let c_d = gemv(&mut b, &[c_gu], t.dg, t.fu, w.down, d.hidden, d.inter);
            let next = t.layers.get(l + 1).map_or(t.norm, |n| n.in_ln);
            dep = b.emit(DevOp::AddNorm, rset.clone(), &[c_d], |i| {
                i.t[..5].copy_from_slice(&[t.xn, t.x, t.x, t.dg, next]);
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = d.eps;
            });
        }
        if j > 0 {
            let c_lm = gemv(&mut b, &[dep], t.logits, t.xn, t.head, d_classes(f), d.hidden);
            code_dep = sample(&mut b, &[c_lm], t.logits, d_classes(f), j, false);
        } else {
            // Position 0's logits are discarded; its KV write is the point.
            code_dep = b.emit(DevOp::Nop, vec![0], &[dep, c0], |_| {});
        }
    }
    b.finish()
}

/// The depth programs for every decode rung, and their manifest.
pub(crate) fn sections(
    c: &Cfg,
    n: &Tn,
    t: &Tensors,
    model: &packet::devbuild::Model,
    rungs: &[u32],
) -> Vec<SectionData> {
    let hidden_rows =
        (model.tensors[n.hn as usize].bytes / (u64::from(c.hidden) * BF16)) as u32;
    let programs: Vec<Program> = rungs
        .iter()
        .map(|&rows| program(c, n, t, &model.tensors, model.n_cu, rows, hidden_rows))
        .collect();
    let packed: Vec<Vec<_>> = programs
        .iter()
        .map(|p| p.insts.iter().map(|d| d.pack()).collect())
        .collect();
    let views: Vec<plow_asset::program::Program<'_>> = programs
        .iter()
        .zip(&packed)
        .zip(rungs)
        .map(|((p, insts), &rows)| plow_asset::program::Program {
            rows,
            role: packet::devbuild::ProgramRole::DecodeRung { rows },
            n_counter: p.n_counter,
            insts,
            stream: &p.stream,
            stream_ofs: &p.stream_ofs,
            stream_len: &p.stream_len,
            waits: &p.waits,
            succs: &p.succs,
            gq_stream: &p.gq_stream,
            gq_seg_ofs: &p.gq_seg_ofs,
            l2_domains: p.l2_domains,
        })
        .collect();
    let bytes = plow_asset::aux_program::encode(model.n_cu, model.tensors.len(), &views)
        .unwrap_or_else(|e| panic!("fish depth programs: {e}"));
    let f = c.frame.as_ref().unwrap();
    let h = |x: u32| u16::try_from(x).unwrap_or_else(|_| panic!("fish depth: handle {x}"));
    let m = manifest::Manifest {
        version: manifest::VERSION,
        rows: rungs.to_vec(),
        codebooks: f.codebooks,
        semantic_base: f.lo,
        semantic_tokens: f.rows,
        stop_class: t.classes - 1,
        ras_window: RAS_WINDOW,
        tensors: manifest::Tensors {
            params: h(t.params),
            top_k: h(t.top_k),
            seed: h(t.seed),
            step: h(t.step),
            history: h(t.hist),
            hidden_row: h(t.hrow),
            frame_codes: h(n.frame_dcodes),
            prompt_codes: h(n.frame_codes),
        },
    };
    m.validate(programs.len(), model.tensors.len())
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("  fish depth: {} programs (rows {rungs:?})", programs.len());
    vec![
        SectionData {
            kind: SECT_METADATA,
            name: manifest::SECTION.into(),
            data: serde_json::to_vec(&m).expect("fish_depth.json"),
        },
        SectionData {
            kind: SECT_METADATA,
            name: manifest::PROGRAMS_SECTION.into(),
            data: bytes,
        },
    ]
}
