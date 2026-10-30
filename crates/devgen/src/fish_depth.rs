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
//!
//! Breeze TTS 2 (`FishDepth::breeze`) runs the same frame schedule with its own pieces: the backbone
//! class comes from an fp32 head (`GemvF32Out` over `lm_head`) sampled with the HF repetition
//! penalty over a per-slot class bitmap (`SampleSmallF32`), each position's input (the gathered
//! hidden, then frame-table row `(j - 1) * cb_size + code`) is projected from the backbone width,
//! RoPE is the depth decoder's own llama3 table, and position `j` samples from fp32 head `j - 1`.
//! `in.breeze.force` (u32::MAX = sample) teacher-forces classes for the gates.

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Program, SectionData, SECT_METADATA};
use packet::rope::GenTensor;
use plow_asset::fish_depth as manifest;

use crate::config::{Cfg, FishDepth};
use crate::Tn;

const BF16: u64 = 2;
const I32: u64 = 4;
const F32: u64 = 4;
/// Backbone sampler RAS window (Fish `inference.py`: the last 10 row-0 tokens).
const RAS_WINDOW: u32 = 10;
/// Breeze's backbone repetition bitmap: one bit per class (codes and the stop class).
fn bitmap_words(classes: u32) -> u32 {
    classes.div_ceil(32)
}

struct LayerW {
    /// W8A16 (Fish): the projections' e4m3 twins and per-output-channel f32 scales replace the
    /// bf16 matrices below, which are then not declared.
    fp8: Option<LayerW8>,
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

struct LayerW8 {
    q: (u32, u32),
    k: (u32, u32),
    v: (u32, u32),
    o: (u32, u32),
    gate: (u32, u32),
    up: (u32, u32),
    down: (u32, u32),
}

pub(crate) struct Tensors {
    d: FishDepth,
    classes: u32,
    /// Breeze: the backbone-width input row, its projector, the fp32 heads and logits, the next
    /// position's embedding row, forced classes and the depth RoPE tables.
    xin: u32,
    proj: u32,
    heads: Vec<u32>,
    blogits: u32,
    dlogits: Vec<u32>,
    eidx: u32,
    force: u32,
    cos: u32,
    sin: u32,
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
    let f = c
        .frame
        .as_ref()
        .expect("fish depth needs the frame embedding");
    let d = f.depth.clone().expect("fish depth config");
    assert!(
        d.positions == f.codebooks
            && (d.breeze || (d.head_dim == c.hd_full && f64::from(d.theta) == c.theta_full)),
        "fish depth: positions == codebooks, and the backbone's full RoPE tables (head_dim, theta)"
    );
    let (r, h) = (u64::from(dbatch), u64::from(d.hidden));
    let (qd, kd) = (
        u64::from(d.heads * d.head_dim),
        u64::from(d.kv_heads * d.head_dim),
    );
    let classes = c.out_vocab();
    let pre = if d.breeze {
        "breeze.depth"
    } else {
        "fish.depth"
    };
    let w = |b: &mut Builder, s: &str, n: u64| b.tensor(&format!("{pre}.{s}"), n * BF16);
    let a = |b: &mut Builder, s: &str, n: u64| b.tensor(&format!("act.fish.{s}"), n);
    let konst = |b: &mut Builder, name: String, v: u32| {
        b.tensor_init(&name, (0..dbatch).flat_map(|_| v.to_le_bytes()).collect())
    };
    let kv = u64::from(d.kv_heads * d.positions * d.head_dim) * r * BF16;
    let dc = u64::from(d_classes(f));
    let (top_k, step) = if d.breeze {
        let ctl = b.tensor("in.breeze.ctl", r * 2 * I32);
        (ctl, ctl)
    } else {
        (
            b.tensor("in.fish.top_k", r * I32),
            b.tensor("in.fish.step", r * I32),
        )
    };
    let (cos, sin) = if d.breeze {
        let [cs, sn] = GenTensor::rope_pair(
            d.positions,
            d.head_dim,
            f64::from(d.theta),
            1.0,
            d.rope_scale,
        );
        (
            b.tensor_gen("in.breeze.cos", cs.byte_len(), cs),
            b.tensor_gen("in.breeze.sin", sn.byte_len(), sn),
        )
    } else {
        (TENSOR_NONE, TENSOR_NONE)
    };
    let opt = |b: &mut Builder, s: &str, n: u64| if d.breeze { a(b, s, n) } else { TENSOR_NONE };
    // W8A16 (`--w8a16`): the depth layers' projections stream e4m3, as the backbone's do.
    let fp8 = !d.breeze && crate::emit_config::active().w8a16;
    Tensors {
        classes,
        xin: opt(b, "xin", r * u64::from(c.hidden) * BF16),
        proj: if d.breeze {
            w(b, "proj.weight", h * u64::from(c.hidden))
        } else {
            TENSOR_NONE
        },
        heads: if d.breeze {
            (0..d.positions - 1)
                .map(|j| w(b, &format!("head.{j}.weight"), dc * h))
                .collect()
        } else {
            Vec::new()
        },
        blogits: opt(b, "blogits", r * u64::from(classes) * F32),
        dlogits: if d.breeze {
            (1..d.positions)
                .map(|j| a(b, &format!("dlogits.{j}"), r * dc * F32))
                .collect()
        } else {
            Vec::new()
        },
        eidx: opt(b, "eidx", r * I32),
        force: if d.breeze {
            b.tensor_init(
                "in.breeze.force",
                vec![0xFF; (r * u64::from(f.codebooks) * I32) as usize],
            )
        } else {
            TENSOR_NONE
        },
        cos,
        sin,
        params: b.tensor("in.fish.params", r * 4 * F32),
        top_k,
        seed: b.tensor("in.fish.seed", r * 8),
        step,
        hist: if d.breeze {
            b.tensor(
                "in.breeze.bitmap",
                r * u64::from(bitmap_words(classes)) * I32,
            )
        } else {
            b.tensor("in.fish.hist", r * u64::from(RAS_WINDOW) * I32)
        },
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
        logits: if d.breeze {
            TENSOR_NONE
        } else {
            a(b, "logits", r * dc * BF16)
        },
        // One row past the codebook: the stop class embeds as zeros (scripts/tts/fish_prep.py).
        embed: if d.breeze {
            TENSOR_NONE
        } else {
            w(b, "embed.weight", (dc + 1) * h)
        },
        head: if d.breeze {
            TENSOR_NONE
        } else {
            w(b, "head.weight", dc * h)
        },
        norm: w(b, "norm.weight", h),
        layers: (0..d.layers)
            .map(|l| {
                let p = format!("layers.{l}.");
                let i = u64::from(d.inter);
                let mat =
                    |b: &mut Builder, s: &str, n: u64| if fp8 { TENSOR_NONE } else { w(b, s, n) };
                LayerW {
                    fp8: fp8.then(|| {
                        let mut t8 = |s: &str, out: u64, k: u64| {
                            let name = format!("fp8/{pre}.{p}{s}.weight");
                            (
                                b.tensor(&name, out * k),
                                b.tensor(&format!("{name}_scale"), out * F32),
                            )
                        };
                        LayerW8 {
                            q: t8("self_attn.q_proj", qd, h),
                            k: t8("self_attn.k_proj", kd, h),
                            v: t8("self_attn.v_proj", kd, h),
                            o: t8("self_attn.o_proj", h, qd),
                            gate: t8("mlp.gate_proj", i, h),
                            up: t8("mlp.up_proj", i, h),
                            down: t8("mlp.down_proj", h, i),
                        }
                    }),
                    in_ln: w(b, &format!("{p}input_layernorm.weight"), h),
                    q: mat(b, &format!("{p}self_attn.q_proj.weight"), qd * h),
                    k: mat(b, &format!("{p}self_attn.k_proj.weight"), kd * h),
                    v: mat(b, &format!("{p}self_attn.v_proj.weight"), kd * h),
                    o: mat(b, &format!("{p}self_attn.o_proj.weight"), h * qd),
                    post_ln: w(b, &format!("{p}post_attention_layernorm.weight"), h),
                    gate: mat(b, &format!("{p}mlp.gate_proj.weight"), i * h),
                    up: mat(b, &format!("{p}mlp.up_proj.weight"), i * h),
                    down: mat(b, &format!("{p}mlp.down_proj.weight"), h * i),
                }
            })
            .collect(),
        d,
    }
}

/// Classes of the depth head: one codebook's codes (Breeze: the 2048 valid of 2051 rows; the
/// reference masks the rest).
fn d_classes(f: &crate::config::FrameEmbed) -> u32 {
    if f.depth.as_ref().is_some_and(|d| d.breeze) {
        f.rows
    } else {
        f.cb_size
    }
}

fn program(
    c: &Cfg,
    n: &Tn,
    t: &Tensors,
    tensors: &[packet::devbuild::TensorDecl],
    n_cu: u32,
    rows: u32,
    hidden_rows: u32,
) -> Program {
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
            i.i = [
                rows,
                vocab,
                f.codebooks,
                j,
                j,
                if ras { RAS_WINDOW } else { 0 },
                u32::from(ras),
                0,
            ];
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
    let gemv8 = |b: &mut Builder,
                 dep: &[u32],
                 out: u32,
                 x: u32,
                 (w8, sc): (u32, u32),
                 nout: u32,
                 kin: u32| {
        b.emit(DevOp::GemvFp8, all.clone(), dep, |i| {
            i.t[0] = out;
            i.t[1] = x;
            i.t[2] = w8;
            i.t[5] = sc;
            i.i[0] = rows;
            i.i[1] = nout;
            i.i[2] = kin;
        })
    };
    let sample_small =
        |b: &mut Builder, dep: &[u32], logits: u32, vocab: u32, j: u32, penalty: bool| {
            b.emit(DevOp::SampleSmallF32, rset.clone(), dep, |i| {
                i.t = [
                    n.frame_dcodes,
                    logits,
                    t.params,
                    t.top_k,
                    t.seed,
                    if penalty { t.hist } else { TENSOR_NONE },
                    t.force,
                    t.eidx,
                ];
                i.i = [
                    rows,
                    vocab,
                    f.codebooks,
                    j,
                    j,
                    bitmap_words(t.classes),
                    0,
                    0,
                ];
                i.j[0] = j * f.cb_size;
            })
        };
    let gemv_f32 = |b: &mut Builder, dep: &[u32], out: u32, x: u32, w: u32, nout: u32, kin: u32| {
        b.emit(DevOp::GemvF32Out, all.clone(), dep, |i| {
            i.t[0] = out;
            i.t[1] = x;
            i.t[2] = w;
            i.i[0] = rows;
            i.i[1] = nout;
            i.i[2] = kin;
        })
    };
    let g = b.emit(DevOp::RowGather, rset.clone(), &[], |i| {
        i.t[0] = if d.breeze { t.xin } else { t.x };
        i.t[1] = n.hn;
        i.t[2] = t.hrow;
        i.i[0] = rows;
        i.i[1] = c.hidden;
        i.i[2] = hidden_rows;
    });
    let c0 = if d.breeze {
        let bl = gemv_f32(&mut b, &[g], t.blogits, t.xin, n.head, t.classes, c.hidden);
        sample_small(&mut b, &[bl], t.blogits, t.classes, 0, true)
    } else {
        sample(&mut b, &[], n.logits, t.classes, 0, true)
    };
    let mut prev = vec![g];
    let mut code_dep = c0;
    let (cos, sin) = if d.breeze {
        (t.cos, t.sin)
    } else {
        (n.cos_f, n.sin_f)
    };
    for j in 0..d.positions {
        if j > 0 {
            prev = vec![b.emit(DevOp::Embed, rset.clone(), &[code_dep], |i| {
                if d.breeze {
                    i.t[..3].copy_from_slice(&[t.xin, n.frame_cb, t.eidx]);
                    i.i[1] = c.hidden;
                } else {
                    i.t[..3].copy_from_slice(&[t.x, t.embed, t.codes[j as usize - 1]]);
                    i.i[1] = d.hidden;
                }
                i.i[0] = rows;
                i.f[0] = 1.0;
            })];
        }
        if d.breeze {
            prev = vec![gemv(&mut b, &prev, t.x, t.xin, t.proj, d.hidden, c.hidden)];
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
            let c_qkv = match &w.fp8 {
                // Three GemvFp8, not GemvQkvFp8: the decode object carries the arms its packet's
                // programs use, and a W8A16 backbone issues q / k / v as GemvFp8.
                Some(w8) => {
                    let q = gemv8(&mut b, &[dep], t.qg, t.xn, w8.q, qd, d.hidden);
                    let k = gemv8(&mut b, &[dep], t.kg, t.xn, w8.k, kd, d.hidden);
                    let v = gemv8(&mut b, &[dep], t.vg, t.xn, w8.v, kd, d.hidden);
                    b.emit(DevOp::Nop, vec![0], &[q, k, v], |_| {})
                }
                None => b.emit(DevOp::GemvQkv, all.clone(), &[dep], |i| {
                    i.t[..7].copy_from_slice(&[t.qg, t.xn, w.q, t.kg, w.k, t.vg, w.v]);
                    i.i[..5].copy_from_slice(&[rows, qd, d.hidden, kd, kd]);
                }),
            };
            let hnr = |b: &mut Builder, out: u32, x: u32, heads: u32, rope: bool, kv: bool| {
                b.emit(
                    DevOp::HeadNormRope,
                    span((rows * heads).div_ceil(8)),
                    &[c_qkv],
                    |i| {
                        i.t[0] = out;
                        i.t[1] = x;
                        if rope {
                            i.t[3] = cos;
                            i.t[4] = sin;
                        }
                        i.t[5] = t.pos[j as usize];
                        i.i[..7].copy_from_slice(&[
                            rows,
                            heads,
                            d.head_dim,
                            0,
                            1,
                            0,
                            if kv { rows } else { 0 },
                        ]);
                        i.f[0] = d.eps;
                        if kv {
                            // Per-row ring of `positions` rows written at pos[row] = j (n_batch_kv armed
                            // at every width: the legacy arm addresses by out_row0 + row).
                            i.j[0] = d.positions;
                            i.j[1] = u32::MAX;
                        }
                    },
                )
            };
            let c_q = hnr(&mut b, t.q, t.qg, d.heads, true, false);
            let c_k = hnr(&mut b, t.kc[l], t.kg, d.kv_heads, true, true);
            let c_v = hnr(&mut b, t.vc[l], t.vg, d.kv_heads, false, true);
            let c_fa = b.emit(DevOp::FlashDecode, all.clone(), &[c_q, c_k, c_v], |i| {
                i.t[..6].copy_from_slice(&[
                    t.opart,
                    t.mlpart,
                    t.q,
                    t.kc[l],
                    t.vc[l],
                    t.len[j as usize],
                ]);
                i.i = [
                    rows,
                    d.heads,
                    d.kv_heads,
                    d.positions,
                    0,
                    1,
                    d.head_dim,
                    u32::MAX,
                ];
                i.f[0] = 1.0 / (d.head_dim as f32).sqrt();
                if rows > 1 {
                    i.j[0] = rows * d.kv_heads * d.positions;
                }
            });
            let c_mg = b.emit(DevOp::FlashMerge, span(rows * d.heads), &[c_fa], |i| {
                i.t[..3].copy_from_slice(&[t.at, t.opart, t.mlpart]);
                i.i[..4].copy_from_slice(&[rows, d.heads, 1, d.head_dim]);
            });
            let c_o = match &w.fp8 {
                Some(w8) => gemv8(&mut b, &[c_mg], t.og, t.at, w8.o, d.hidden, qd),
                None => gemv(&mut b, &[c_mg], t.og, t.at, w.o, d.hidden, qd),
            };
            let c_r1 = b.emit(DevOp::AddNorm, rset.clone(), &[c_o], |i| {
                i.t[..5].copy_from_slice(&[t.xn, t.x, t.x, t.og, w.post_ln]);
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = d.eps;
            });
            let c_gu = match &w.fp8 {
                Some(w8) => b.emit(DevOp::GemvGluFp8, all.clone(), &[c_r1], |i| {
                    i.t[0] = t.fu;
                    i.t[1] = t.xn;
                    i.t[2] = w8.gate.0;
                    i.t[5] = w8.up.0;
                    i.t[3] = w8.gate.1;
                    i.t[4] = w8.up.1;
                    i.i[0] = rows;
                    i.i[1] = d.inter;
                    i.i[2] = d.hidden;
                    i.i[5] = 1;
                }),
                None => b.emit(DevOp::GemvGlu, all.clone(), &[c_r1], |i| {
                    i.t[0] = t.fu;
                    i.t[1] = t.xn;
                    i.t[2] = w.gate;
                    i.t[5] = w.up;
                    i.i[0] = rows;
                    i.i[1] = d.inter;
                    i.i[2] = d.hidden;
                    i.i[5] = 1;
                }),
            };
            let c_d = match &w.fp8 {
                Some(w8) => gemv8(&mut b, &[c_gu], t.dg, t.fu, w8.down, d.hidden, d.inter),
                None => gemv(&mut b, &[c_gu], t.dg, t.fu, w.down, d.hidden, d.inter),
            };
            let next = t.layers.get(l + 1).map_or(t.norm, |n| n.in_ln);
            dep = b.emit(DevOp::AddNorm, rset.clone(), &[c_d], |i| {
                i.t[..5].copy_from_slice(&[t.xn, t.x, t.x, t.dg, next]);
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = d.eps;
            });
        }
        if j > 0 && d.breeze {
            let lg = t.dlogits[j as usize - 1];
            let c_lm = gemv_f32(
                &mut b,
                &[dep],
                lg,
                t.xn,
                t.heads[j as usize - 1],
                d_classes(f),
                d.hidden,
            );
            code_dep = sample_small(&mut b, &[c_lm], lg, d_classes(f), j, false);
        } else if j > 0 {
            let c_lm = gemv(
                &mut b,
                &[dep],
                t.logits,
                t.xn,
                t.head,
                d_classes(f),
                d.hidden,
            );
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
    let hidden_rows = (model.tensors[n.hn as usize].bytes / (u64::from(c.hidden) * BF16)) as u32;
    let programs: Vec<Program> = rungs
        .iter()
        .map(|&rows| program(c, n, t, &model.tensors, model.n_cu, rows, hidden_rows))
        .collect();
    let bytes = encode(model.n_cu, model.tensors.len(), &programs, rungs);
    let f = c.frame.as_ref().unwrap();
    let h = |x: u32| u16::try_from(x).unwrap_or_else(|_| panic!("fish depth: handle {x}"));
    let m = manifest::Manifest {
        version: manifest::VERSION,
        rows: rungs.to_vec(),
        codebooks: f.codebooks,
        semantic_base: f.lo,
        semantic_tokens: f.rows,
        stop_class: t.classes - 1,
        ras_window: if t.d.breeze {
            bitmap_words(t.classes)
        } else {
            RAS_WINDOW
        },
        sampler: if t.d.breeze {
            manifest::Sampler::Breeze
        } else {
            manifest::Sampler::Fish
        },
        pair_cfg: t.d.breeze,
        depth_sampling: None,
        slot_memory: Vec::new(),
        tensors: manifest::Tensors {
            params: h(t.params),
            top_k: h(t.top_k),
            seed: h(t.seed),
            step: h(t.step),
            history: h(t.hist),
            hidden_row: h(t.hrow),
            frame_codes: h(n.frame_dcodes),
            prompt_codes: h(n.frame_codes),
            force: (t.force != TENSOR_NONE).then(|| h(t.force)),
        },
    };
    eprintln!("  fish depth: {} programs (rows {rungs:?})", programs.len());
    manifest_sections(&m, programs.len(), model.tensors.len(), bytes)
}

/// The two packet sections of a depth stage: its manifest and its programs.
pub(crate) fn manifest_sections(
    m: &manifest::Manifest,
    programs: usize,
    tensors: usize,
    bytes: Vec<u8>,
) -> Vec<SectionData> {
    m.validate(programs, tensors)
        .unwrap_or_else(|e| panic!("{e}"));
    vec![
        SectionData {
            kind: SECT_METADATA,
            name: manifest::SECTION.into(),
            data: serde_json::to_vec(m).expect("fish_depth.json"),
        },
        SectionData {
            kind: SECT_METADATA,
            name: manifest::PROGRAMS_SECTION.into(),
            data: bytes,
        },
    ]
}

/// Programs of `rows[k]` rows as an [`plow_asset::aux_program`] section (decode-rung roles).
pub(crate) fn encode(n_cu: u32, tensors: usize, programs: &[Program], rungs: &[u32]) -> Vec<u8> {
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
    plow_asset::aux_program::encode(n_cu, tensors, &views)
        .unwrap_or_else(|e| panic!("depth programs: {e}"))
}
