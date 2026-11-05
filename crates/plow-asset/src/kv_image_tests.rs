use super::*;
use crate::live_kv::{Cache, LAYOUT_HEAD_MAJOR, LAYOUT_ROW_INTERLEAVED};
use crate::program::{Program, Tensor};
use packet::devbuild::ProgramRole;
use packet::rope::ROPE_SCALE_NONE;

/// One attention layer of a synthetic packet. `src` = the layer whose cache it reads.
#[derive(Clone, Copy)]
struct Layer {
    src: usize,
    heads: u32,
    hd: u32,
    window: u32,
    theta: f64,
    frac: f64,
    fp8: bool,
    /// Qwen-style V (no norm) instead of Gemma's weightless RMS.
    v_skip_norm: bool,
}

struct Model {
    names: Vec<String>,
    insts: Vec<DevInst64>,
    generated: Vec<GenTensor>,
    caches: Vec<Cache>,
    tp: bool,
}

fn inst(op: DevOp) -> DevInst64 {
    DevInst64 {
        op: op as u16,
        blocks: 1,
        fj: [0; 3],
        t: [TENSOR_NONE16; 8],
        i: [0; 8],
    }
}

/// Layer `l`'s query: an RMS-normed, rotated head written to `act.q` (handle 2).
fn query(m: &mut Model, l: usize) {
    m.names
        .push(format!("model.layers.{l}.self_attn.q_norm.weight"));
    let mut d = inst(DevOp::HeadNormRope);
    d.t[0] = 2;
    d.t[2] = (m.names.len() - 1) as u16;
    m.insts.push(d);
}

fn build(layers: &[Layer], stride: u32, layout: u32) -> Model {
    let mut m = Model {
        names: vec!["in.pos".into(), "in.kvlen".into(), "act.q".into()],
        insts: Vec::new(),
        generated: Vec::new(),
        caches: Vec::new(),
        tp: false,
    };
    let add = |names: &mut Vec<String>, n: String| {
        names.push(n);
        (names.len() - 1) as u16
    };
    let mut handles = Vec::new();
    for (l, ly) in layers.iter().enumerate() {
        if ly.src != l {
            handles.push(handles[ly.src]);
            let (k, v, scales): (u16, u16, Option<[u16; 2]>) = handles[ly.src];
            query(&mut m, l);
            let mut d = inst(if ly.fp8 {
                DevOp::FlashDecodeFp8
            } else {
                DevOp::FlashDecode
            });
            d.t[2] = 2;
            d.t[3] = k;
            d.t[4] = v;
            if let Some(s) = scales {
                d.t[6..8].copy_from_slice(&s);
            }
            m.insts.push(d);
            continue;
        }
        let k = add(&mut m.names, format!("kv.{l}.k"));
        let v = add(&mut m.names, format!("kv.{l}.v"));
        let scales = ly.fp8.then(|| {
            [
                add(&mut m.names, format!("kv.{l}.k_scale")),
                add(&mut m.names, format!("kv.{l}.v_scale")),
            ]
        });
        let gamma = add(
            &mut m.names,
            format!("model.layers.{l}.self_attn.k_norm.weight"),
        );
        let cos = add(&mut m.names, format!("rope.{l}.cos"));
        let sin = add(&mut m.names, format!("rope.{l}.sin"));
        for (t, kind) in [(cos, GEN_ROPE_COS), (sin, GEN_ROPE_SIN)] {
            m.generated.push(GenTensor {
                tensor: t as u32,
                kind,
                ctx: stride,
                hd: ly.hd,
                aux: 0,
                scale: ROPE_SCALE_NONE,
                theta: ly.theta,
                frac: ly.frac,
                factor: 0.0,
                low: 0.0,
                high: 0.0,
                orig: 0.0,
            });
        }
        handles.push((k, v, scales));
        let op = if ly.fp8 {
            DevOp::HeadNormRopeFp8
        } else {
            DevOp::HeadNormRope
        };
        for (out, scale, is_k) in [
            (k, scales.map(|s| s[0]), true),
            (v, scales.map(|s| s[1]), false),
        ] {
            let mut d = inst(op);
            d.t[0] = out;
            d.t[1] = 2;
            d.t[5] = 0;
            if let Some(s) = scale {
                d.t[6] = s;
            }
            if is_k {
                d.t[2] = gamma;
                d.t[3] = cos;
                d.t[4] = sin;
            }
            d.i[1] = ly.heads;
            d.i[2] = ly.hd;
            d.i[4] = u32::from(!is_k && ly.v_skip_norm);
            d.fj[0] = 1e-6f32.to_bits();
            m.insts.push(d);
        }
        query(&mut m, l);
        let mut d = inst(if ly.fp8 {
            DevOp::FlashDecodeFp8
        } else {
            DevOp::FlashDecode
        });
        d.t[2] = 2;
        d.t[3] = k;
        d.t[4] = v;
        if let Some(s) = scales {
            d.t[6..8].copy_from_slice(&s);
        }
        m.insts.push(d);
        m.caches.push(Cache {
            pair: [k, v],
            scales,
            heads: ly.heads,
            hd: ly.hd,
            stride,
            window: ly.window,
            mask: if ly.window == 0 { u32::MAX } else { stride - 1 },
            layout,
        });
    }
    m
}

fn header(m: &Model) -> Result<Header> {
    let tensors: Vec<Tensor<'_>> = m
        .names
        .iter()
        .map(|n| Tensor {
            name: n,
            bytes: 64,
            initialized: false,
        })
        .collect();
    let program = Program {
        rows: 4,
        role: ProgramRole::DecodeRung { rows: 4 },
        n_counter: 0,
        insts: &m.insts,
        stream: &[],
        stream_ofs: &[],
        stream_len: &[],
        waits: &[],
        succs: &[],
        gq_stream: &[],
        gq_seg_ofs: &[],
        l2_domains: 1,
    };
    let programs = [program];
    let packet = Packet {
        n_cu: 1,
        target: 0,
        tp: m.tp,
        prefill_count: 0,
        tensors: &tensors,
        programs: &programs,
        generated: &m.generated,
        kv_row_insts: &[],
    };
    let manifest = Manifest {
        version: 1,
        n_cu: 1,
        batch: 4,
        max_ctx: 4096,
        position: 0,
        kv_length: 1,
        row_slot: None,
        caches: m.caches.clone(),
        maps: Vec::new(),
        programs: vec!["native".into()],
        splitk: Vec::new(),
    };
    derive(&packet, &manifest, "w".repeat(64))
}

/// Qwen3-0.6B: 28 full layers, 8 KV heads of 128, qk-norm, theta 1e6.
fn qwen3_0_6b() -> Vec<Layer> {
    (0..28)
        .map(|l| Layer {
            src: l,
            heads: 8,
            hd: 128,
            window: 0,
            theta: 1e6,
            frac: 1.0,
            fp8: false,
            v_skip_norm: true,
        })
        .collect()
}

/// Gemma-4 E4B: 42 layers, every sixth full (hd 512, partial rotary 0.25, theta 1e6), the rest
/// sliding over 512 rows (hd 256, theta 1e4), 2 KV heads; the last 18 layers read the last
/// owner of their type.
fn gemma4_e4b() -> Vec<Layer> {
    let full = |l: usize| l % 6 == 5;
    (0..42)
        .map(|l| {
            let src = if l < 24 {
                l
            } else if full(l) {
                23
            } else {
                22
            };
            Layer {
                src,
                heads: 2,
                hd: if full(l) { 512 } else { 256 },
                window: if full(l) { 0 } else { 512 },
                theta: if full(l) { 1e6 } else { 1e4 },
                frac: if full(l) { 0.25 } else { 1.0 },
                fp8: false,
                v_skip_norm: false,
            }
        })
        .collect()
}

#[test]
fn qwen3_0_6b_golden_header() {
    let h = header(&build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR)).unwrap();
    assert_eq!(h.entries.len(), 56);
    assert_eq!(h.entries[0].id, "L0.k");
    assert_eq!(h.entries[1].id, "L0.v");
    assert_eq!(h.entries[1].norm, Norm::None);
    assert_eq!(h.entries[1].rope, None);
    assert_eq!(h.entries[0].rope.unwrap().pairing, Pairing::Half);
    assert_eq!(
        h.digest(),
        "b802815fe8e710891e176b9c827d29fd94c64ed65d13bcc9badf22f8f6666d4b",
        "a change here changes which images Qwen3-0.6B engines exchange"
    );
}

#[test]
fn gemma4_e4b_golden_header() {
    let h = header(&build(&gemma4_e4b(), 1024, LAYOUT_HEAD_MAJOR)).unwrap();
    assert_eq!(h.entries.len(), 48);
    let e = |id: &str| h.entries.iter().find(|e| e.id == id).unwrap();
    let sliding_tail: Vec<u32> = (24..42).filter(|l| l % 6 != 5).collect();
    assert_eq!(e("L22.k").readers, [&[22][..], &sliding_tail].concat());
    assert_eq!(e("L23.k").readers, [23, 29, 35, 41]);
    assert_eq!(e("L0.k").readers, [0]);
    assert_eq!(e("L0.k").class, Class::Window { rows: 512 });
    assert_eq!(e("L5.k").class, Class::Append);
    assert_eq!(e("L5.k").rope.unwrap().frac, 0.25);
    assert_eq!(
        e("L5.v").norm,
        Norm::Rms {
            eps: 1e-6,
            gamma: false
        }
    );
    assert_eq!(
        h.digest(),
        "ce5e7b9df3b8eb8e243c8138706adb91791193e42d7a9f74251d3e0c18a8efd9",
        "a change here changes which images Gemma-4 E4B engines exchange"
    );
}

#[test]
fn native_bindings_stay_out_of_the_header() {
    let base = header(&build(&gemma4_e4b(), 1024, LAYOUT_HEAD_MAJOR)).unwrap();
    assert_eq!(
        header(&build(&gemma4_e4b(), 2048, LAYOUT_HEAD_MAJOR)).unwrap(),
        base
    );
    let mut m = build(&qwen3_0_6b(), 4096, LAYOUT_ROW_INTERLEAVED);
    let row = header(&m).unwrap();
    // Extra tensors shift every handle; the AMD BF16 rotary tables are the same rotation.
    m.names.insert(0, "pad".into());
    for d in &mut m.insts {
        for t in d.t.iter_mut().filter(|t| **t != TENSOR_NONE16) {
            *t += 1;
        }
    }
    for c in &mut m.caches {
        c.pair = c.pair.map(|h| h + 1);
    }
    for g in &mut m.generated {
        g.tensor += 1;
        g.kind = if g.kind == GEN_ROPE_COS {
            GEN_AMD_ROPE_BF16_COS
        } else {
            GEN_AMD_ROPE_BF16_SIN
        };
    }
    assert_eq!(header(&m).unwrap(), row);
    assert_eq!(
        header(&build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR)).unwrap(),
        row
    );
}

#[test]
fn every_semantic_field_changes_the_key() {
    let base = header(&build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR)).unwrap();
    let layer = |f: fn(&mut Layer)| {
        let mut ls = qwen3_0_6b();
        f(&mut ls[3]);
        header(&build(&ls, 4096, LAYOUT_HEAD_MAJOR)).unwrap()
    };
    let inst_of = |f: fn(&mut Model)| {
        let mut m = build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR);
        f(&mut m);
        header(&m).unwrap()
    };
    let cases: Vec<(Header, &str)> = vec![
        (layer(|l| l.heads = 4), "L3.k.heads"),
        (layer(|l| l.hd = 64), "L3.k.hd"),
        (layer(|l| l.window = 256), "L3.k.class"),
        (layer(|l| l.theta = 1e4), "L3.k.rope"),
        (layer(|l| l.frac = 0.5), "L3.k.rope"),
        (layer(|l| l.fp8 = true), "L3.k.dtype"),
        (layer(|l| l.v_skip_norm = false), "L3.v.norm"),
        (inst_of(|m| m.insts[0].t[2] = TENSOR_NONE16), "L0.k.norm"),
        (
            inst_of(|m| m.insts[0].fj[0] = 1e-5f32.to_bits()),
            "L0.k.norm",
        ),
        (inst_of(|m| m.insts[0].i[5] = 1), "L0.k.rope"),
        (
            inst_of(|m| {
                let q = m
                    .names
                    .iter()
                    .position(|n| n.contains("layers.5.self_attn.q_"))
                    .unwrap();
                m.names[q] = "model.layers.9.self_attn.q_norm.weight".into();
            }),
            "L5.k.readers",
        ),
        (
            inst_of(|m| {
                m.generated[0].scale = 4;
                m.generated[1].scale = 4;
            }),
            "L0.k.rope",
        ),
        (
            Header {
                weights: "x".repeat(64),
                ..base.clone()
            },
            "weights",
        ),
    ];
    for (h, field) in cases {
        assert_ne!(h.digest(), base.digest(), "{field}");
        assert_eq!(base.first_difference(&h).as_deref(), Some(field));
    }
    let mut short = base.clone();
    short.entries.pop();
    assert_eq!(
        base.first_difference(&short).as_deref(),
        Some("entry L27.v present on one side only")
    );
    assert_eq!(base.first_difference(&base), None);
}

#[test]
fn unrecognised_caches_are_refused() {
    let refused = |f: fn(&mut Model)| {
        let mut m = build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR);
        f(&mut m);
        header(&m).unwrap_err()
    };
    // A fused writer: no standalone store of the cache.
    assert!(refused(|m| m.insts.retain(|d| d.t[0] != 3)).contains("no recognised writer"));
    // A baked rotary table.
    assert!(refused(|m| m.generated.clear()).contains("baked"));
    assert!(refused(|m| m.insts[0].i[5] = 3).contains("pairing"));
    assert!(refused(|m| {
        let mut d = m.insts[0];
        d.fj[0] = 7;
        m.insts.push(d);
    })
    .contains("disagree"));
    assert!(refused(|m| m.caches[0].scales = Some([1, 1])).contains("FP8"));
    assert!(refused(|m| m.names[3] = "kv.k".into()).contains("kv.<layer>.k"));
    assert!(refused(|m| m.tp = true).contains("TP"));
    // The decode attention writes the new rows itself (folded writer descriptor).
    assert!(refused(|m| m.insts[3].t[7] = 1).contains("folded"));
    assert!(refused(|m| m.generated[1].scale = 4).contains("disagree"));
    assert!(refused(|m| m.insts[0].op = DevOp::QwenHeadNormRope as u16).contains("not supported"));
    assert!(refused(|m| {
        m.generated[0].theta = f64::NAN;
        m.generated[1].theta = f64::NAN;
    })
    .contains("finite"));
    assert!(
        refused(|m| m.names[5] = "model.layers.1.self_attn.k_norm.weight".into())
            .contains("k_norm")
    );
    assert!(refused(|m| m.insts[1].t[2] = 5).contains("v_norm"));
    assert!(refused(|m| {
        let q = m.names.iter().position(|n| n.contains("q_norm")).unwrap();
        m.names[q] = "act.qn".into();
    })
    .contains("query norm"));
}

#[test]
fn checkpoint_digest_covers_names_and_content() {
    let dir = std::env::temp_dir().join(format!("plow-kvimg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    std::fs::write(dir.join("model.safetensors"), b"abc").unwrap();
    std::fs::write(dir.join("README.md"), b"ignored").unwrap();
    let a = checkpoint_digest(&dir).unwrap();
    std::fs::write(dir.join("README.md"), b"still ignored").unwrap();
    assert_eq!(checkpoint_digest(&dir).unwrap(), a);
    std::fs::write(dir.join("model.safetensors"), b"abd").unwrap();
    let b = checkpoint_digest(&dir).unwrap();
    assert_ne!(a, b);
    std::fs::rename(dir.join("model.safetensors"), dir.join("m.safetensors")).unwrap();
    assert_ne!(checkpoint_digest(&dir).unwrap(), b);
    std::fs::remove_file(dir.join("m.safetensors")).unwrap();
    assert!(checkpoint_digest(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn caches_pack_in_layer_order() {
    let mut m = build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR);
    let tensors: Vec<Tensor<'_>> = m
        .names
        .iter()
        .map(|n| Tensor {
            name: n,
            bytes: 64,
            initialized: false,
        })
        .collect();
    m.caches.reverse();
    let packet = Packet {
        n_cu: 1,
        target: 0,
        tp: false,
        prefill_count: 0,
        tensors: &tensors,
        programs: &[],
        generated: &[],
        kv_row_insts: &[],
    };
    let mut manifest = Manifest {
        version: 1,
        n_cu: 1,
        batch: 4,
        max_ctx: 4096,
        position: 0,
        kv_length: 1,
        row_slot: None,
        caches: m.caches.clone(),
        maps: Vec::new(),
        programs: Vec::new(),
        splitk: Vec::new(),
    };
    canonical_order(&packet, &mut manifest).unwrap();
    let first = &tensors[manifest.caches[0].pair[0] as usize];
    assert_eq!(first.name, "kv.0.k");
}

#[test]
fn a_fused_kv_writer_states_the_same_v() {
    let base = header(&build(&gemma4_e4b(), 1024, LAYOUT_HEAD_MAJOR)).unwrap();
    let mut m = build(&gemma4_e4b(), 1024, LAYOUT_HEAD_MAJOR);
    // Layer 0: K writer at 0, V writer at 1. Fold V into K's instruction (devgen's pair fusion).
    let v = m.insts[1];
    m.insts[0].i[7] = 1;
    m.insts[0].t[6] = v.t[0];
    m.insts[0].t[7] = v.t[1];
    m.insts.remove(1);
    assert_eq!(header(&m).unwrap(), base);
    // The fused V then normalizes as K does: a V stored unnormed elsewhere disagrees.
    let mut prefill_v = v;
    prefill_v.i[4] = 1;
    m.insts.push(prefill_v);
    assert!(header(&m).unwrap_err().contains("disagree"));
}

#[test]
fn unused_rope_parameters_and_signed_zero_do_not_split_keys() {
    let base = header(&build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR)).unwrap();
    let mut m = build(&qwen3_0_6b(), 4096, LAYOUT_HEAD_MAJOR);
    m.generated[0].factor = 3.0;
    m.generated[1].low = -0.0;
    assert_eq!(header(&m).unwrap(), base);
}
