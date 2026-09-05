use super::*;
use std::path::Path;

fn prefill_segment_roles(m: &Model) -> Option<packet::devbuild::SectionData> {
    let cfg = emit_config::active();
    let fp8 = cfg.fp8_pf_gemm_role || cfg.fp8_pf_isolate;
    let attention = cfg.attention_pf_role || cfg.attention_pf_isolate;
    if !fp8 && !attention {
        return None;
    }
    let mut programs = Vec::new();
    let mut used = [false; 3];
    for (index, p) in m.progs.iter().enumerate().take(m.progs.len() - 1) {
        let mut roles = vec![0_u8; p.gq_seg_ofs.len() - 1];
        let mut found_fp8 = false;
        for (ix, d) in p.insts.iter().enumerate() {
            let role = if fp8 && d.op == DevOp::GemmFp8 as u16 {
                assert!(
                    d.i[6] != 0 && d.i[7] != 0,
                    "FP8 role needs mapped prefill GEMMs"
                );
                found_fp8 = true;
                u8::from(cfg.fp8_pf_gemm_role)
            } else if attention && d.op == DevOp::FlashPrefill as u16 {
                assert_eq!(d.i[6], 256, "attention role requires HD256");
                assert!(d.t[5..8].iter().all(|&t| t == TENSOR_NONE));
                2 * u8::from(cfg.attention_pf_role)
            } else if attention && d.op == DevOp::FlashMerge as u16 {
                assert_eq!(d.i[3], 256, "attention role requires HD256");
                2 * u8::from(cfg.attention_pf_role)
            } else {
                continue;
            };
            let seg = p.stream.iter().find(|e| e.inst as usize == ix).unwrap().seg as usize;
            assert!(p
                .stream
                .iter()
                .all(|e| (e.inst as usize == ix) == (e.seg as usize == seg)));
            roles[seg] = role;
            used[role as usize] = true;
        }
        assert!(
            !fp8 || found_fp8,
            "FP8 role metadata requires FP8 prefill programs"
        );
        programs.push(serde_json::json!({"index":index,"roles":roles}));
    }
    assert!(
        !programs.is_empty(),
        "prefill roles require prefill programs"
    );
    let mut objects = serde_json::Map::new();
    if used[1] {
        objects.insert(
            "1".into(),
            serde_json::json!({"abi":"fp8_gemm_tma128_v1","file":"interp_sm90a_pfgemm_fp8.cubin"}),
        );
    }
    if used[2] {
        objects.insert("2".into(), serde_json::json!({"abi":"attention_sm90_hd256_v1","file":"interp_sm90a_pfattn_hd256.cubin"}));
    }
    Some(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: "segment_roles.json".into(),
        data: serde_json::to_vec(
            &serde_json::json!({"version":1,"objects":objects,"programs":programs}),
        )
        .unwrap(),
    })
}

struct Config {
    hidden: u32,
    inter: u32,
    vocab: u32,
    heads: u32,
    kv_heads: u32,
    hd: u32,
    hk: u32,
    hv: u32,
    dk: u32,
    dv: u32,
    conv: u32,
    rotary: u32,
    theta: f64,
    eps: f32,
    layers: Vec<bool>,
    prefix: String,
    tied: bool,
    block: Option<usize>,
}

impl Config {
    fn parse(root: &Value) -> Self {
        let v = root.get("text_config").unwrap_or(root);
        let u = |name: &str| -> u32 {
            let n = v[name]
                .as_u64()
                .unwrap_or_else(|| panic!("qwen3_5 missing {name}"));
            u32::try_from(n).expect("qwen3_5 dimension exceeds u32")
        };
        assert!(
            v.get("quantization_config").is_none() && root.get("quantization_config").is_none(),
            "qwen3_5 native checkpoint FP8 lowering is not implemented yet"
        );
        assert_eq!(
            v["attention_bias"], false,
            "qwen3_5 biased attention unsupported"
        );
        assert_eq!(v["attn_output_gate"], true);
        assert_eq!(v["hidden_act"], "silu");
        assert_eq!(v["mamba_ssm_dtype"], "float32");
        assert_eq!(v["output_gate_type"], "swish");
        assert_eq!(v["rope_parameters"]["rope_type"], "default");
        let layers: Vec<_> = v["layer_types"]
            .as_array()
            .expect("layer_types")
            .iter()
            .map(|x| match x.as_str() {
                Some("full_attention") => true,
                Some("linear_attention") => false,
                other => panic!("qwen3_5 unsupported layer type {other:?}"),
            })
            .collect();
        assert_eq!(layers.len(), u("num_hidden_layers") as usize);
        let hd = u("head_dim");
        let rotary = hd as f64
            * v["rope_parameters"]["partial_rotary_factor"]
                .as_f64()
                .expect("partial_rotary_factor");
        assert!(
            rotary > 0.0 && rotary <= hd as f64 && rotary.fract() == 0.0 && rotary as u32 % 2 == 0
        );
        let c = Self {
            hidden: u("hidden_size"),
            inter: u("intermediate_size"),
            vocab: u("vocab_size"),
            heads: u("num_attention_heads"),
            kv_heads: u("num_key_value_heads"),
            hd,
            hk: u("linear_num_key_heads"),
            hv: u("linear_num_value_heads"),
            dk: u("linear_key_head_dim"),
            dv: u("linear_value_head_dim"),
            conv: u("linear_conv_kernel_dim"),
            rotary: rotary as u32,
            theta: v["rope_parameters"]["rope_theta"]
                .as_f64()
                .expect("rope_theta"),
            eps: v["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32,
            layers,
            prefix: v
                .get("_plow_weight_prefix")
                .and_then(Value::as_str)
                .map(|p| format!("{}.", p.trim_end_matches('.')))
                .unwrap_or_else(|| {
                    if root.get("text_config").is_some() {
                        "model.language_model."
                    } else {
                        "model."
                    }
                    .into()
                }),
            tied: v["tie_word_embeddings"].as_bool().unwrap_or(false),
            block: None,
        };
        assert!(c.hk > 0 && c.hv % c.hk == 0 && c.kv_heads > 0 && c.heads % c.kv_heads == 0);
        assert_eq!(
            (c.dk, c.dv),
            (128, 128),
            "qwen3_5 CUDA recurrence currently supports D128"
        );
        assert_eq!(c.hd, 256, "qwen3_5 initial full attention supports HD256");
        assert_eq!(
            (c.heads / c.kv_heads) % 2,
            0,
            "qwen3_5 requires CUDA FA_GF=2"
        );
        assert!(c.conv >= 2 && c.hidden > 0 && c.inter > 0 && c.vocab > 0);
        c
    }
}

struct Emitter<'a> {
    b: Builder,
    c: &'a Config,
    fp8: bool,
    w8a8: bool,
    prefill: bool,
    batch: u32,
    decode_slots: u32,
    decode_lt: bool,
    fuse_ab: bool,
    ab_blocks: u32,
    projection_dag: bool,
    share_quant: bool,
    quant_input: Option<(u32, u32)>,
    ctx: u32,
    active: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
}

impl Emitter<'_> {
    fn act(&mut self, name: &str, width: u32) -> u32 {
        self.b
            .tensor(&format!("act.{name}"), self.batch as u64 * width as u64 * 2)
    }
    fn weight(&mut self, name: &str, elems: u64) -> u32 {
        self.b.tensor(name, elems * 2)
    }
    fn proj(&mut self, out: u32, src: u32, name: &str, n: u32, k: u32, dep: u32) -> u32 {
        let quantized = self.fp8 && name.contains(".layers.");
        let (op, w, scale) = if quantized {
            (
                DevOp::GemvFp8,
                self.b.tensor(&format!("fp8/{name}"), n as u64 * k as u64),
                self.b.tensor(&format!("fp8/{name}_scale"), n as u64 * 4),
            )
        } else {
            (
                DevOp::Gemv,
                self.weight(name, n as u64 * k as u64),
                TENSOR_NONE,
            )
        };
        if quantized && self.w8a8 {
            let rows = self.batch;
            let xq = self
                .b
                .tensor(&format!("act.fp8.x{k}"), u64::from(rows) * u64::from(k));
            let ascale = self.b.tensor("act.fp8.scale", u64::from(rows) * 4);
            // Projections are chained, so each quant waits for the previous scratch consumer.
            let dq = if self.share_quant && self.quant_input == Some((src, k)) {
                dep
            } else {
                self.quant_input = Some((src, k));
                let blocks = if self.prefill { self.b.all() } else { vec![0] };
                self.b.emit(DevOp::QuantFp8, blocks, &[dep], |d| {
                    d.t[..3].copy_from_slice(&[xq, src, ascale]);
                    d.i[..2].copy_from_slice(&[rows, k]);
                })
            };
            let input_map = if self.prefill {
                let g = GenTensor::tmap_e4m3(xq, rows, k, 128);
                self.b
                    .tensor_gen(&format!("tmap.fp8.pf.x{xq}.m{rows}"), g.byte_len(), g)
            } else {
                0
            };
            let weight_map = if self.prefill {
                let g = GenTensor::tmap_e4m3(w, n, k, 128);
                self.b
                    .tensor_gen(&format!("tmap.fp8.pf.w{w}"), g.byte_len(), g)
            } else if emit_config::active().qwen_fp8_m1_tma {
                let g = GenTensor::tmap_e4m3(w, n, k, 64);
                self.b.tensor_gen(&format!("tmap.fp8.{w}"), g.byte_len(), g)
            } else {
                0
            };
            let done = self.b.emit(DevOp::GemmFp8, self.b.all(), &[dq], |d| {
                d.t[..5].copy_from_slice(&[out, xq, w, ascale, scale]);
                d.i[..3].copy_from_slice(&[rows, n, k]);
                d.i[6] = input_map;
                d.i[7] = weight_map;
            });
            if self.prefill
                && (emit_config::active().fp8_pf_isolate || emit_config::active().fp8_pf_gemm_role)
            {
                self.b.isolate(done);
            }
            return done;
        }
        let op = if self.prefill {
            assert!(
                !quantized,
                "Qwen native prefill initially requires BF16 projections"
            );
            pick_tile(
                self.batch,
                n,
                k,
                self.b.n_cu(),
                kernelcaps::QuantScheme::None,
            )
        } else {
            op
        };
        let maps = (self.prefill && emit_config::active().tma_gemm).then(|| {
            let mut map = |target, rows| {
                let g = GenTensor::tmap_bf16(target, rows, k, 128);
                self.b
                    .tensor_gen(&format!("tmap.{target}.{rows}x{k}"), g.byte_len(), g)
            };
            (map(src, self.batch), map(w, n))
        });
        let done = self.b.emit(op, self.b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[out, src, w]);
            d.t[5] = scale;
            d.i[..3].copy_from_slice(&[self.batch, n, k]);
            if let Some((a, w)) = maps {
                d.i[6] = a;
                d.i[7] = w;
            }
        });
        if self.decode_lt && name.contains(".layers.") {
            self.b.isolate(done);
        }
        done
    }
    fn norm(&mut self, out: u32, src: u32, name: &str, dep: u32) -> u32 {
        // The next norm overwrites the shared hn arena, ending its quantized input lifetime.
        self.quant_input = None;
        let gamma = self.weight(name, self.c.hidden as u64);
        self.b.emit(
            DevOp::QwenRmsNorm,
            (0..self.batch.min(self.b.n_cu())).collect(),
            &[dep],
            |d| {
                d.t[..4].copy_from_slice(&[out, src, gamma, self.active]);
                d.i[..2].copy_from_slice(&[self.c.hidden, self.batch]);
                d.f[0] = self.c.eps;
                d.f[1] = 1.0;
            },
        )
    }
    fn residual(&mut self, x: u32, mixed: u32, dep: u32) -> u32 {
        self.b.emit(
            DevOp::Residual,
            if self.prefill { self.b.all() } else { vec![0] },
            &[dep],
            |d| {
                d.t[..3].copy_from_slice(&[x, x, mixed]);
                d.i[0] = self.batch * self.c.hidden;
                d.f[0] = 1.0;
            },
        )
    }
    fn linear(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32) -> u32 {
        let c = self.c;
        let p = format!("{}layers.{layer}.linear_attn", c.prefix);
        let channels = 2 * c.hk * c.dk + c.hv * c.dv;
        let value = c.hv * c.dv;
        let raw = self.act("gdn.raw", channels);
        let conv = self.act("gdn.conv", channels);
        let z = self.act("gdn.z", value);
        let a = self.act("gdn.a", c.hv);
        let beta = self.act("gdn.b", c.hv);
        let core = self.act("gdn.core", value);
        let gated = self.act("gdn.gated", value);
        let history = self.b.tensor(
            &format!("state.qwen.{layer}.conv"),
            u64::from(self.decode_slots) * channels as u64 * (c.conv - 1) as u64 * 2,
        );
        let state = self.b.tensor(
            &format!("state.qwen.{layer}.gdn"),
            u64::from(self.decode_slots) * c.hv as u64 * c.dv as u64 * c.dk as u64 * 4,
        );
        let cw = self.weight(
            &format!("{p}.conv1d.weight"),
            channels as u64 * c.conv as u64,
        );
        let alog = self.weight(&format!("{p}.A_log"), c.hv as u64);
        let dt = self.weight(&format!("{p}.dt_bias"), c.hv as u64);
        let gamma = self.weight(&format!("{p}.norm.weight"), c.dv as u64);
        let dq = self.proj(
            raw,
            hn,
            &format!("{p}.in_proj_qkv.weight"),
            channels,
            c.hidden,
            dep,
        );
        let conv_op = if self.prefill {
            DevOp::QwenGdnConvPrefill
        } else {
            DevOp::QwenGdnConv
        };
        let dc = self.b.emit(conv_op, self.b.all(), &[dq], |d| {
            d.t[..5].copy_from_slice(&[conv, raw, cw, history, self.active]);
            d.i[..3].copy_from_slice(&[channels, c.conv, self.batch]);
        });
        let dz = self.proj(
            z,
            hn,
            &format!("{p}.in_proj_z.weight"),
            value,
            c.hidden,
            if self.projection_dag { dq } else { dc },
        );
        let ab_dep = if self.projection_dag { dep } else { dz };
        let db = if self.fuse_ab {
            let wa = self.weight(
                &format!("{p}.in_proj_a.weight"),
                c.hv as u64 * c.hidden as u64,
            );
            let wb = self.weight(
                &format!("{p}.in_proj_b.weight"),
                c.hv as u64 * c.hidden as u64,
            );
            self.b.emit(
                DevOp::GemvQkv,
                (0..self.ab_blocks).collect(),
                &[ab_dep],
                |d| {
                    d.t[..7].copy_from_slice(&[a, hn, wa, beta, wb, TENSOR_NONE, TENSOR_NONE]);
                    d.i[..5].copy_from_slice(&[self.batch, c.hv, c.hidden, c.hv, 0]);
                },
            )
        } else {
            let da = self.proj(
                a,
                hn,
                &format!("{p}.in_proj_a.weight"),
                c.hv,
                c.hidden,
                ab_dep,
            );
            self.proj(
                beta,
                hn,
                &format!("{p}.in_proj_b.weight"),
                c.hv,
                c.hidden,
                da,
            )
        };
        let ds = if self.prefill {
            let q = self.act("gdn.q", c.hk * c.dk);
            let k = self.act("gdn.k", c.hk * c.dk);
            let v = self.act("gdn.v", value);
            let alpha = self
                .b
                .tensor("act.gdn.alpha", self.batch as u64 * c.hv as u64 * 4);
            let prepared_beta = self
                .b
                .tensor("act.gdn.beta", self.batch as u64 * c.hv as u64 * 4);
            let outstate = self.b.tensor(
                "act.gdn.outstate",
                c.hv as u64 * c.dv as u64 * c.dk as u64 * 4,
            );
            let dq = self
                .b
                .emit(DevOp::QwenGdnQkvPrep, self.b.all(), &[db], |d| {
                    d.t[..4].copy_from_slice(&[q, k, v, conv]);
                    d.i[..5].copy_from_slice(&[c.hk, c.hv, c.dk, c.dv, self.batch]);
                    d.f[0] = 1e-6;
                });
            let dg = self
                .b
                .emit(DevOp::QwenGdnGatePrep, self.b.all(), &[dq], |d| {
                    d.t[..6].copy_from_slice(&[alpha, prepared_beta, a, beta, alog, dt]);
                    d.i[..2].copy_from_slice(&[c.hv, self.batch]);
                });
            self.b.emit(DevOp::QwenGdnPrefill, vec![0], &[dg], |d| {
                d.t.copy_from_slice(&[core, q, k, v, alpha, prepared_beta, state, outstate]);
                d.i[..5].copy_from_slice(&[self.batch, c.hk, c.hv, c.dk, c.dv]);
                d.f[0] = 1.0 / (c.dk as f32).sqrt();
            })
        } else {
            let deps = if self.projection_dag {
                vec![dc, db]
            } else {
                vec![db]
            };
            self.b.emit(DevOp::QwenGdnStep, self.b.all(), &deps, |d| {
                d.t.copy_from_slice(&[core, conv, a, beta, alog, dt, state, self.active]);
                d.i[..5].copy_from_slice(&[c.hk, c.hv, c.dk, c.dv, self.batch]);
                d.f[0] = 1.0 / (c.dk as f32).sqrt();
                d.f[1] = 1e-6;
            })
        };
        let deps = if self.projection_dag {
            vec![ds, dz]
        } else {
            vec![ds]
        };
        let dn = self.b.emit(DevOp::QwenGatedNorm, self.b.all(), &deps, |d| {
            d.t[..5].copy_from_slice(&[gated, core, z, gamma, self.active]);
            d.i[..3].copy_from_slice(&[c.hv, c.dv, self.batch]);
            d.f[0] = c.eps;
        });
        self.proj(
            mixed,
            gated,
            &format!("{p}.out_proj.weight"),
            c.hidden,
            value,
            dn,
        )
    }
    fn headnorm(
        &mut self,
        out: u32,
        src: u32,
        gamma: u32,
        heads: u32,
        cache: bool,
        normalize: bool,
        dep: u32,
    ) -> u32 {
        self.b
            .emit(DevOp::QwenHeadNormRope, self.b.all(), &[dep], |d| {
                d.t[..7].copy_from_slice(&[
                    out,
                    src,
                    gamma,
                    if normalize { self.cos } else { TENSOR_NONE },
                    if normalize { self.sin } else { TENSOR_NONE },
                    self.pos,
                    self.active,
                ]);
                d.i[..6].copy_from_slice(&[
                    heads,
                    self.c.hd,
                    if normalize { self.c.rotary } else { 0 },
                    self.batch,
                    if cache { self.ctx } else { 0 },
                    normalize as u32,
                ]);
                d.i[6] = self.prefill as u32;
                d.f[0] = self.c.eps;
                d.f[1] = if normalize { 1.0 } else { 0.0 };
            })
    }
    fn full(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32) -> u32 {
        let c = self.c;
        let p = format!("{}layers.{layer}.self_attn", c.prefix);
        let qd = c.heads * c.hd;
        let kd = c.kv_heads * c.hd;
        let packed = self.act("qgate", 2 * qd);
        let qg = self.act("qg", qd);
        let gate = self.act("qgate.gate", qd);
        let kg = self.act("kg", kd);
        let vg = self.act("vg", kd);
        let q = self.act("q", qd);
        let at = self.act("at", qd);
        let kc = self.b.tensor(
            &format!("kv.{layer}.k"),
            u64::from(self.decode_slots) * kd as u64 * self.ctx as u64 * 2,
        );
        let vc = self.b.tensor(
            &format!("kv.{layer}.v"),
            u64::from(self.decode_slots) * kd as u64 * self.ctx as u64 * 2,
        );
        let qn = self.weight(&format!("{p}.q_norm.weight"), c.hd as u64);
        let kn = self.weight(&format!("{p}.k_norm.weight"), c.hd as u64);
        let dp = self.proj(
            packed,
            hn,
            &format!("{p}.q_proj.weight"),
            2 * qd,
            c.hidden,
            dep,
        );
        let dq = self
            .b
            .emit(DevOp::QwenQGateSplit, self.b.all(), &[dp], |d| {
                d.t[..4].copy_from_slice(&[qg, gate, packed, self.active]);
                d.i[..3].copy_from_slice(&[c.heads, c.hd, self.batch]);
            });
        let dq = self.headnorm(q, qg, qn, c.heads, false, true, dq);
        let dk = self.proj(
            kg,
            hn,
            &format!("{p}.k_proj.weight"),
            kd,
            c.hidden,
            if self.projection_dag { dep } else { dq },
        );
        let dk = self.headnorm(kc, kg, kn, c.kv_heads, true, true, dk);
        let dv = self.proj(
            vg,
            hn,
            &format!("{p}.v_proj.weight"),
            kd,
            c.hidden,
            if self.projection_dag { dep } else { dk },
        );
        let dv = self.headnorm(vc, vg, TENSOR_NONE, c.kv_heads, true, false, dv);
        // HD256 uses the slide GF selector in the Hopper interpreter even for full attention.
        let ns = if self.prefill {
            self.b
                .n_cu()
                .div_ceil(self.batch.div_ceil(64) * c.heads)
                .max(2)
        } else {
            self.b.n_cu().div_ceil(c.heads / 2).max(1)
        };
        let opart = self
            .b
            .tensor("act.opart", self.batch as u64 * qd as u64 * ns as u64 * 4);
        let mlpart = self.b.tensor(
            "act.mlpart",
            self.batch as u64 * c.heads as u64 * ns as u64 * 8,
        );
        let attention_deps = if self.projection_dag {
            vec![dq, dk, dv]
        } else {
            vec![dv]
        };
        let da = if self.prefill {
            self.b.emit(DevOp::FlashPrefill, self.b.all(), &[dv], |d| {
                d.t[..5].copy_from_slice(&[opart, mlpart, q, kc, vc]);
                d.i.copy_from_slice(&[self.batch, self.batch, c.heads, c.kv_heads, 0, 0, c.hd, ns]);
                d.j[0] = self.ctx;
                d.j[1] = u32::MAX;
                d.f[0] = 1.0 / (c.hd as f32).sqrt();
            })
        } else {
            self.b
                .emit(DevOp::FlashDecode, self.b.all(), &attention_deps, |d| {
                    d.t[..6].copy_from_slice(&[opart, mlpart, q, kc, vc, self.kvlen]);
                    d.i.copy_from_slice(&[
                        self.batch,
                        c.heads,
                        c.kv_heads,
                        self.ctx,
                        0,
                        ns,
                        c.hd,
                        u32::MAX,
                    ]);
                    d.f[0] = 1.0 / (c.hd as f32).sqrt();
                })
        };
        let dm = self.b.emit(
            DevOp::FlashMerge,
            (0..c.heads.min(self.b.n_cu())).collect(),
            &[da],
            |d| {
                d.t[..3].copy_from_slice(&[at, opart, mlpart]);
                d.i[..4].copy_from_slice(&[self.batch, c.heads, ns, c.hd]);
            },
        );
        if self.prefill
            && (emit_config::active().attention_pf_isolate
                || emit_config::active().attention_pf_role)
        {
            self.b.isolate(da);
            self.b.isolate(dm);
        }
        let dg = self
            .b
            .emit(DevOp::QwenSigmoidGate, self.b.all(), &[dm], |d| {
                d.t[..4].copy_from_slice(&[at, at, gate, self.active]);
                d.i[..2].copy_from_slice(&[qd, self.batch]);
            });
        self.proj(mixed, at, &format!("{p}.o_proj.weight"), c.hidden, qd, dg)
    }
}

fn model(c: &Config, ctx: u32, n_cu: u32, target: u32, fp8: bool, batch: u32) -> Model {
    assert!(matches!(batch, 1 | 4), "Qwen decode supports batch 1 or 4");
    phase(c, ctx, n_cu, target, fp8, batch, batch, false, None)
}

fn model_prefill(
    c: &Config,
    ctx: u32,
    n_cu: u32,
    target: u32,
    buckets: &[u32],
    decode_slots: u32,
    fp8: bool,
) -> Model {
    assert!(matches!(decode_slots, 1 | 4));
    assert!(!buckets.is_empty());
    assert!(
        !fp8 || (emit_config::active().w8a8
            && emit_config::active().qwen_w8a8_prefill
            && decode_slots == 1
            && buckets
                .iter()
                .all(|rows| matches!(rows, 128 | 1024 | 4096 | 8192))),
        "Qwen FP8 prefill requires explicit W8A8 prefill, B1 and buckets128/1024/4096/8192"
    );
    let mut tensors = None;
    let mut progs = Vec::new();
    let mut gen = Vec::new();
    for &rows in buckets {
        assert!(rows >= 128 && rows % 128 == 0 && rows <= ctx);
        let mut pf = phase(c, ctx, n_cu, target, fp8, rows, decode_slots, true, tensors);
        tensors = Some(pf.tensors);
        gen.extend(pf.gen);
        progs.push(pf.progs.remove(0));
    }
    let mut dec = phase(
        c,
        ctx,
        n_cu,
        target,
        fp8,
        decode_slots,
        decode_slots,
        false,
        tensors,
    );
    // Adopting tensor declarations does not carry their generation recipes.
    for recipe in gen {
        if !dec.gen.iter().any(|g| g.tensor == recipe.tensor) {
            dec.gen.push(recipe);
        }
    }
    progs.append(&mut dec.progs);
    dec.progs = progs;
    dec.prog_t = buckets
        .iter()
        .copied()
        .chain(std::iter::once(decode_slots))
        .collect();
    dec
}

fn phase(
    c: &Config,
    ctx: u32,
    n_cu: u32,
    target: u32,
    fp8: bool,
    batch: u32,
    decode_slots: u32,
    prefill: bool,
    tensors: Option<Vec<packet::devbuild::TensorDecl>>,
) -> Model {
    let mut b = Builder::new(n_cu);
    if let Some(tensors) = tensors {
        b.adopt_tensors(tensors);
    }
    b.set_tensor_dedup(true);
    b.force_uniseg();
    let ids = b.tensor("in.ids", ctx as u64 * 4);
    // The loader derives max_ctx from this legacy input arena extent.
    let pos = b.tensor("in.pos", ctx as u64 * 4);
    let slots = if prefill { 1 } else { batch };
    let kvlen = b.tensor("in.kvlen", u64::from(decode_slots) * 4);
    let active = b.tensor("in.active", u64::from(decode_slots) * 4);
    let active = if prefill {
        b.tensor_init(
            "in.prefill_active",
            (0..batch).flat_map(|_| 1i32.to_le_bytes()).collect(),
        )
    } else {
        active
    };
    let [gc, gs] = GenTensor::rope_pair(ctx, c.rotary, c.theta, 1.0, RopeScale::None);
    let cos = b.tensor_gen("in.cos_full", gc.byte_len(), gc);
    let sin = b.tensor_gen("in.sin_full", gs.byte_len(), gs);
    let decode_lt = !prefill && emit_config::active().qwen_decode_lt;
    let fuse_ab = !prefill && emit_config::active().qwen_fuse_ab;
    let fuse_mlp = !prefill && emit_config::active().qwen_fuse_mlp;
    assert!(
        !fuse_mlp || (!fp8 && !decode_lt),
        "Qwen MLP projection fusion requires native BF16 decode"
    );
    let projection_dag = !prefill && emit_config::active().qwen_projection_dag;
    assert!(
        !projection_dag || (!fp8 && !decode_lt),
        "Qwen projection DAG requires native BF16 decode"
    );
    assert!(
        !fuse_ab || (!fp8 && !decode_lt),
        "Qwen a/b fusion requires native BF16 decode"
    );
    let ab_blocks = if fuse_ab {
        emit_config::active().qwen_ab_blocks.unwrap_or(n_cu)
    } else {
        n_cu
    };
    assert!(
        ab_blocks > 0 && ab_blocks <= n_cu,
        "Qwen a/b blocks must fit the compiled grid"
    );
    let w8a8 = emit_config::active().w8a8;
    assert!(
        !emit_config::active().qwen_fp8_m1_tma || w8a8,
        "Qwen FP8 M1 TMA maps require W8A8"
    );
    let share_quant = emit_config::active().qwen_share_quant;
    assert!(!share_quant || w8a8, "Qwen quant sharing requires W8A8");
    assert!(
        !w8a8
            || (fp8
                && decode_slots == 1
                && ((!prefill && batch == 1)
                    || (prefill
                        && matches!(batch, 128 | 1024 | 4096 | 8192)
                        && emit_config::active().qwen_w8a8_prefill))),
        "Qwen W8A8 requires B1 decode or explicit supported prefill buckets"
    );
    assert!(
        !decode_lt || !fp8,
        "Qwen decode Lt requires BF16 projections"
    );
    let mut e = Emitter {
        b,
        c,
        fp8,
        w8a8,
        prefill,
        batch,
        decode_slots,
        decode_lt,
        fuse_ab,
        ab_blocks,
        projection_dag,
        share_quant,
        quant_input: None,
        ctx,
        active,
        pos,
        kvlen,
        cos,
        sin,
    };
    let x = e.act("x", c.hidden);
    let hn = e.act("hn", c.hidden);
    let mixed = e.act("mixed", c.hidden);
    let gate = e.act("gt", c.inter);
    let up = e.act("ut", c.inter);
    let fu = e.act("fu", c.inter);
    let logits =
        e.b.tensor("act.logits", u64::from(decode_slots) * c.vocab as u64 * 2);
    let amax =
        e.b.tensor("act.amax", u64::from(decode_slots) * AMAX_BLOCKS as u64 * 8);
    let mut dep = if c.block.is_some() {
        e.b.emit(DevOp::Nop, vec![0], &[], |_| {})
    } else {
        let emb = e.weight(
            &format!("{}embed_tokens.weight", c.prefix),
            c.vocab as u64 * c.hidden as u64,
        );
        e.b.emit(DevOp::Embed, (0..batch.min(n_cu)).collect(), &[], |d| {
            d.t[..3].copy_from_slice(&[x, emb, ids]);
            d.i[..2].copy_from_slice(&[batch, c.hidden]);
            d.f[0] = 1.0;
        })
    };
    for (layer, &full) in c.layers.iter().enumerate() {
        if c.block.is_some_and(|selected| selected != layer) {
            continue;
        }
        let p = format!("{}layers.{layer}", c.prefix);
        dep = e.norm(hn, x, &format!("{p}.input_layernorm.weight"), dep);
        dep = if full {
            e.full(layer, hn, mixed, dep)
        } else {
            e.linear(layer, hn, mixed, dep)
        };
        dep = e.residual(x, mixed, dep);
        dep = e.norm(hn, x, &format!("{p}.post_attention_layernorm.weight"), dep);
        dep = if fuse_mlp {
            let wg = e.weight(
                &format!("{p}.mlp.gate_proj.weight"),
                c.inter as u64 * c.hidden as u64,
            );
            let wu = e.weight(
                &format!("{p}.mlp.up_proj.weight"),
                c.inter as u64 * c.hidden as u64,
            );
            e.b.emit(DevOp::GemvQkv, e.b.all(), &[dep], |d| {
                d.t[..7].copy_from_slice(&[gate, hn, wg, up, wu, TENSOR_NONE, TENSOR_NONE]);
                d.i[..5].copy_from_slice(&[batch, c.inter, c.hidden, c.inter, 0]);
            })
        } else {
            let gate_dep = e.proj(
                gate,
                hn,
                &format!("{p}.mlp.gate_proj.weight"),
                c.inter,
                c.hidden,
                dep,
            );
            e.proj(
                up,
                hn,
                &format!("{p}.mlp.up_proj.weight"),
                c.inter,
                c.hidden,
                gate_dep,
            )
        };
        dep = e.b.emit(DevOp::Glu, e.b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[fu, gate, up]);
            d.i[..2].copy_from_slice(&[batch * c.inter, 1]);
        });
        dep = e.proj(
            mixed,
            fu,
            &format!("{p}.mlp.down_proj.weight"),
            c.hidden,
            c.inter,
            dep,
        );
        dep = e.residual(x, mixed, dep);
    }
    if c.block.is_none() {
        dep = e.norm(hn, x, &format!("{}norm.weight", c.prefix), dep);
        let head = if c.tied {
            format!("{}embed_tokens.weight", c.prefix)
        } else {
            "lm_head.weight".into()
        };
        if prefill {
            let w = e.weight(&head, c.vocab as u64 * c.hidden as u64);
            dep = e.b.emit(DevOp::Gemv, e.b.all(), &[dep], |d| {
                d.t[..3].copy_from_slice(&[logits, hn, w]);
                d.i[..3].copy_from_slice(&[1, c.vocab, c.hidden]);
                d.i[4] = batch - 1;
            });
        } else {
            dep = e.proj(logits, hn, &head, c.vocab, c.hidden, dep);
        }
        dep =
            e.b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[dep], |d| {
                d.t[..2].copy_from_slice(&[amax, logits]);
                d.i[0] = c.vocab;
                d.i[1] = slots;
            });
        e.b.emit(DevOp::ArgmaxFin, vec![0], &[dep], |d| {
            d.t[..2].copy_from_slice(&[ids, amax]);
            d.i[0] = AMAX_BLOCKS;
            d.i[1] = slots;
        });
    }
    let tensors = e.b.tensors();
    let gen = e.b.gen_tensors();
    Model {
        n_cu,
        target,
        tensors,
        progs: vec![e.b.finish()],
        kv_row_insts: vec![],
        prog_t: vec![batch],
        gen,
    }
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
    assert_eq!(
        arch, "sm_90a",
        "qwen3_5 native CUDA path currently requires sm_90a"
    );
    assert_eq!(tp, 1, "qwen3_5 tensor parallelism is not implemented");
    let fp8 = emit_config::active().fp8;
    let batch = emit_config::active().decode_batch;
    assert!(matches!(batch, 1 | 4), "Qwen decode supports batch 1 or 4");
    assert!(ctx >= batch && n_cu >= AMAX_BLOCKS);
    let root: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("Qwen config JSON");
    let mut c = Config::parse(&root);
    if let Some(spec) = block {
        let selected = crate::block::parse_block(spec, c.layers.len());
        assert_eq!(
            selected.len(),
            1,
            "Qwen --block currently selects one layer"
        );
        assert_eq!(batch, 1, "Qwen --block currently requires B1");
        c.block = Some(selected.start);
    }
    let target = packet::devbuild::gpu_fingerprint(gpu);
    let _target = EmitAmdGuard::set(false);
    let mut m = match emit_config::active().qwen_prefill.as_deref() {
        None | Some("0") => model(&c, ctx, n_cu, target, fp8, batch),
        Some(value) => {
            let mut buckets: Vec<u32> = if value == "1" {
                vec![128]
            } else {
                value
                    .split(',')
                    .map(|v| {
                        v.parse()
                            .expect("PLOW_QWEN_PREFILL expects comma-separated rows")
                    })
                    .collect()
            };
            buckets.sort_unstable();
            buckets.dedup();
            model_prefill(&c, ctx, n_cu, target, &buckets, batch, fp8)
        }
    };
    validate_coverage(
        dir,
        &c.prefix,
        &m.tensors
            .iter()
            .filter_map(|t| {
                // Sidecars are derived from these exact checkpoint matrices; scales add no source.
                match t.name.strip_prefix("fp8/") {
                    Some(name) if name.ends_with("_scale") => None,
                    Some(name) => Some(name.to_string()),
                    None => Some(t.name.clone()),
                }
            })
            .collect::<Vec<_>>(),
        c.block.map(|layer| layer..layer + 1),
        &[],
        &[],
        &[],
    )
    .unwrap_or_else(|err| panic!("{err}"));
    if !rope_gen {
        m.bake_gen();
    }
    let lean = apply_verify_gate(&m, verify);
    let man = manifest::build(&m, arch, &lean);
    let out = Path::new(out);
    let mut sections = Vec::new();
    if let Some(section) = prefill_segment_roles(&m) {
        sections.push(section);
    }
    let bytes = if let Some(layer) = c.block {
        use plow_asset::*;
        let full = c.layers[layer];
        let desc = BlockDescriptor {
            model: dir.to_string_lossy().into_owned(),
            arch: "qwen3_5".into(),
            layer: layer as u32,
            kind: vec![
                if full { "full_attn" } else { "gdn" }.into(),
                "dense_ffn".into(),
            ],
            hidden: c.hidden as i64,
            dtype: if fp8 { "fp8" } else { "bf16" }.into(),
            dims: BlockDims {
                heads: Some(if full { c.heads } else { c.hv } as i64),
                head_dim: Some(if full { c.hd } else { c.dv } as i64),
                kv_heads: Some(if full { c.kv_heads } else { c.hk } as i64),
                ..Default::default()
            },
            dsa_role: None,
            inputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(c.hidden as i64)],
                dtype: "bf16".into(),
            }],
            outputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(c.hidden as i64)],
                dtype: "bf16".into(),
            }],
            carried_state: vec![CarriedState {
                role: if full { "kv" } else { "recurrent" }.into(),
                tensors: if full {
                    vec![format!("kv.{layer}.k"), format!("kv.{layer}.v")]
                } else {
                    vec![
                        format!("state.qwen.{layer}.conv"),
                        format!("state.qwen.{layer}.gdn"),
                    ]
                },
                layout: if full { "head_major" } else { "slot_major" }.into(),
            }],
            weights: BlockWeights {
                mode: "symlink".into(),
                ckpt: dir.to_string_lossy().into_owned(),
                prefix: format!("{}layers.{layer}.", c.prefix),
            },
            programs: BlockPrograms {
                prefill_buckets: m.prog_t[..m.prog_t.len() - 1]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                decode_t: batch as i64,
            },
        };
        let section =
            crate::block::write_block_descriptor(out.to_str().expect("output path"), &desc);
        sections.push(section);
        m.to_blob_v6(&sections)
    } else if !sections.is_empty() {
        m.to_blob_v6(&sections)
    } else {
        m.to_blob()
    };
    std::fs::write(out, bytes).expect("write Qwen blob");
    manifest::write_config_header(&out.with_file_name("plow_config.h"), &man)
        .expect("Qwen config header");
    std::fs::write(
        out.with_file_name("build.json"),
        serde_json::to_vec_pretty(&man).unwrap(),
    )
    .expect("Qwen build manifest");
    eprintln!(
        "qwen3_5: {} layers, {} full, {} batch{batch} decode -> {}",
        c.layers.len(),
        c.layers.iter().filter(|x| **x).count(),
        if emit_config::active().w8a8 {
            "W8A8"
        } else if fp8 {
            "W8A16"
        } else {
            "BF16"
        },
        out.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_emission_checks_only_selected_checkpoint_layers() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_QWEN_PREFILL", "128"),
            ("PLOW_DECODE_BATCH", "1"),
        ]);
        let c = Config::parse(&fixture());
        let m = model(&c, 128, 132, 0, false, 1);
        let mut header = serde_json::Map::new();
        for t in &m.tensors {
            if t.name.starts_with(&c.prefix) {
                header.insert(
                    t.name.clone(),
                    serde_json::json!({"dtype":"BF16","shape":[1],"data_offsets":[0,2]}),
                );
            }
        }
        let dir = std::env::temp_dir().join(format!("qwen-block-coverage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), fixture().to_string()).unwrap();
        let write_checkpoint = |header: &serde_json::Map<String, Value>| {
            let json = serde_json::to_vec(header).unwrap();
            let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
            bytes.extend(json);
            bytes.extend([0, 0]);
            std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        };
        let emit = |layer: &str| {
            run(
                &dir,
                128,
                dir.join("model.pkt").to_str().unwrap(),
                132,
                1,
                Some(layer),
                true,
                "sm_90a",
                "H100 NVL",
                None,
            )
        };
        write_checkpoint(&header);
        emit("0");
        emit("3");
        let required = format!("{}layers.0.input_layernorm.weight", c.prefix);
        let weight = header.remove(&required).unwrap();
        write_checkpoint(&header);
        assert!(std::panic::catch_unwind(|| emit("0")).is_err());
        header.insert(required, weight);
        header.insert(
            format!("{}layers.0.unimplemented.weight", c.prefix),
            serde_json::json!({"dtype":"BF16","shape":[1],"data_offsets":[0,2]}),
        );
        write_checkpoint(&header);
        assert!(std::panic::catch_unwind(|| emit("0")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fused_mlp_keeps_bf16_gate_up_and_separate_silu() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_QWEN_FUSE_MLP", "0"),
            ("PLOW_QWEN_FUSE_AB", "0"),
            ("PLOW_QWEN_PROJECTION_DAG", "0"),
            ("PLOW_QWEN_AB_BLOCKS", "12"),
        ]);
        let c = Config::parse(&fixture());
        let plain_pf = model_prefill(&c, 8192, 132, 0, &[128], 1, false);
        std::env::set_var("PLOW_QWEN_FUSE_MLP", "1");
        {
            let mut config = emit_config::active().clone();
            config.qwen_fuse_mlp = true;
            emit_config::install(config);
        }
        let fused_pf = model_prefill(&c, 8192, 132, 0, &[128], 1, false);
        assert_eq!(plain_pf.progs[0].insts, fused_pf.progs[0].insts);
        let mut c = c;
        c.layers = (0..64).map(|i| i % 4 == 3).collect();
        for batch in [1, 4] {
            for other in ["0", "1"] {
                std::env::set_var("PLOW_QWEN_FUSE_AB", other);
                {
                    let mut config = emit_config::active().clone();
                    config.qwen_fuse_ab = other == "1";
                    emit_config::install(config);
                }
                std::env::set_var("PLOW_QWEN_PROJECTION_DAG", other);
                {
                    let mut config = emit_config::active().clone();
                    config.qwen_projection_dag = other == "1";
                    emit_config::install(config);
                }
                std::env::set_var("PLOW_QWEN_FUSE_MLP", "0");
                {
                    let mut config = emit_config::active().clone();
                    config.qwen_fuse_mlp = false;
                    emit_config::install(config);
                }
                let plain = model(&c, 8192, 132, 0, false, batch);
                std::env::set_var("PLOW_QWEN_FUSE_MLP", "1");
                {
                    let mut config = emit_config::active().clone();
                    config.qwen_fuse_mlp = true;
                    emit_config::install(config);
                }
                let fused = model(&c, 8192, 132, 0, false, batch);
                let g = &fused.progs[0];
                assert_eq!(plain.progs[0].insts.len(), g.insts.len() + 64);
                let mut mlp = 0;
                let mut ab = 0;
                for (i, d) in g.insts.iter().enumerate() {
                    if d.op != DevOp::GemvQkv as u16 {
                        continue;
                    }
                    if d.i[1] == 48 {
                        ab += 1;
                        continue;
                    }
                    mlp += 1;
                    assert_eq!(d.blocks, 132);
                    assert_eq!(&d.i[..5], &[batch, 17408, 5120, 17408, 0]);
                    assert_eq!(&d.t[5..7], &[TENSOR_NONE, TENSOR_NONE]);
                    for (out, weight, act, suffix) in [
                        (d.t[0], d.t[2], "act.gt", "gate_proj.weight"),
                        (d.t[3], d.t[4], "act.ut", "up_proj.weight"),
                    ] {
                        assert_eq!(fused.tensors[out as usize].name, act);
                        assert_eq!(
                            fused.tensors[out as usize].bytes,
                            u64::from(batch) * 17408 * 2
                        );
                        assert!(fused.tensors[weight as usize].name.ends_with(suffix));
                        assert_eq!(fused.tensors[weight as usize].bytes, 17408 * 5120 * 2);
                    }
                    let glu = &g.insts[i + 1];
                    assert_eq!(glu.op, DevOp::Glu as u16);
                    assert_eq!(&glu.t[1..3], &[d.t[0], d.t[3]]);
                    assert_eq!(&glu.i[..2], &[batch * 17408, 1]);
                }
                assert_eq!(mlp, 64);
                assert_eq!(ab, if other == "1" { 48 } else { 0 });
                check_operand_dependencies(g);
            }
        }
    }
    fn check_operand_dependencies(g: &packet::devbuild::Program) {
        use std::collections::{BTreeSet, HashMap};
        let mut producers: HashMap<u32, BTreeSet<usize>> = HashMap::new();
        for (i, d) in g.insts.iter().enumerate() {
            for id in &g.succs[d.succ_ofs as usize..(d.succ_ofs + u32::from(d.succ_len)) as usize] {
                producers.entry(*id).or_default().insert(i);
            }
        }
        let mut ancestors: Vec<BTreeSet<usize>> = Vec::new();
        let mut writers = HashMap::new();
        let mut readers: HashMap<u32, BTreeSet<usize>> = HashMap::new();
        for (i, d) in g.insts.iter().enumerate() {
            let mut prior = BTreeSet::new();
            for wait in &g.waits[d.wait_ofs as usize..(d.wait_ofs + u32::from(d.wait_len)) as usize]
            {
                for &p in &producers[&wait.id] {
                    assert!(p < i);
                    prior.insert(p);
                    prior.extend(ancestors[p].iter().copied());
                }
            }
            let mut output_slots = vec![0];
            if d.op == DevOp::GemvQkv as u16 {
                output_slots.extend([3, 5]);
            }
            if d.op == DevOp::FlashDecode as u16 || d.op == DevOp::QwenQGateSplit as u16 {
                output_slots.push(1);
            }
            if d.op == DevOp::QuantFp8 as u16 {
                output_slots.push(2);
            }
            if d.op == DevOp::QwenGdnQkvPrep as u16 {
                output_slots.extend([1, 2]);
            }
            if d.op == DevOp::QwenGdnGatePrep as u16 {
                output_slots.push(1);
            }
            if d.op == DevOp::QwenGdnPrefill as u16 {
                output_slots.push(7);
            }
            let state_slot = if d.op == DevOp::QwenGdnConv as u16
                || d.op == DevOp::QwenGdnConvPrefill as u16
            {
                Some(3)
            } else if d.op == DevOp::QwenGdnStep as u16 || d.op == DevOp::QwenGdnPrefill as u16 {
                Some(6)
            } else {
                None
            };
            let mut reads: BTreeSet<_> =
                d.t.iter()
                    .enumerate()
                    .filter(|(slot, t)| !output_slots.contains(slot) && **t != TENSOR_NONE)
                    .map(|(_, t)| *t)
                    .collect();
            if let Some(slot) = state_slot {
                reads.insert(d.t[slot]);
                output_slots.push(slot);
            }
            let writes: BTreeSet<_> = output_slots
                .iter()
                .map(|&slot| d.t[slot])
                .filter(|&t| t != TENSOR_NONE)
                .collect();
            for &t in &reads {
                if let Some(p) = writers.get(&t) {
                    assert!(prior.contains(p), "missing RAW tensor{t}: {p}->{i}");
                }
            }
            for &t in &writes {
                if let Some(p) = writers.get(&t) {
                    assert!(prior.contains(p), "missing WAW tensor{t}: {p}->{i}");
                }
                for p in readers.get(&t).into_iter().flatten() {
                    assert!(prior.contains(p), "missing WAR tensor{t}: {p}->{i}");
                }
                readers.remove(&t);
                writers.insert(t, i);
            }
            for t in reads {
                readers.entry(t).or_default().insert(i);
            }
            ancestors.push(prior);
        }
    }
    #[test]
    fn projection_dag_covers_operand_joins_and_reused_arenas() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_QWEN_PROJECTION_DAG", "1"),
            ("PLOW_QWEN_FUSE_AB", "0"),
        ]);
        let c = Config::parse(&fixture());
        for batch in [1, 4] {
            for fuse in ["0", "1"] {
                std::env::set_var("PLOW_QWEN_FUSE_AB", fuse);
                {
                    let mut config = emit_config::active().clone();
                    config.qwen_fuse_ab = fuse == "1";
                    emit_config::install(config);
                }
                let m = model(&c, 8192, 132, 0, false, batch);
                check_operand_dependencies(&m.progs[0]);
            }
        }
    }
    #[test]
    fn shared_quant_preserves_consumers_and_invalidates_at_every_norm() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_QWEN_SHARE_QUANT", "0")]);
        struct Restore(emit_config::EmitConfig);
        impl Drop for Restore {
            fn drop(&mut self) {
                emit_config::install(self.0.clone());
            }
        }
        let restore = Restore(emit_config::active().clone());
        let mut cfg = restore.0.clone();
        cfg.fp8 = true;
        cfg.w8a8 = true;
        emit_config::install(cfg);
        let mut c = Config::parse(&fixture());
        c.layers = (0..64).map(|i| i % 4 == 3).collect();
        let plain = model(&c, 8192, 132, 0, true, 1);
        std::env::set_var("PLOW_QWEN_SHARE_QUANT", "1");
        {
            let mut config = emit_config::active().clone();
            config.qwen_share_quant = true;
            emit_config::install(config);
        }
        let shared = model(&c, 8192, 132, 0, true, 1);
        let count = |m: &Model| {
            m.progs[0]
                .insts
                .iter()
                .filter(|d| d.op == DevOp::QuantFp8 as u16)
                .count()
        };
        assert_eq!(count(&plain), 496);
        assert_eq!(count(&shared), 256);
        let plain_body: Vec<_> = plain.progs[0]
            .insts
            .iter()
            .filter(|d| d.op != DevOp::QuantFp8 as u16)
            .collect();
        let shared_body: Vec<_> = shared.progs[0]
            .insts
            .iter()
            .filter(|d| d.op != DevOp::QuantFp8 as u16)
            .collect();
        assert_eq!(plain_body.len(), shared_body.len());
        for (a, b) in plain_body.iter().zip(&shared_body) {
            assert_eq!(
                (a.op, a.blocks, a.t, a.i, a.f, a.j),
                (b.op, b.blocks, b.t, b.i, b.f, b.j)
            );
        }
        let mut quant = None;
        for d in &shared.progs[0].insts {
            if d.op == DevOp::QwenRmsNorm as u16 {
                quant = None;
            }
            if d.op == DevOp::QuantFp8 as u16 {
                quant = Some((d.t[0], d.t[2], d.i[1]));
            }
            if d.op == DevOp::GemmFp8 as u16 {
                assert_eq!(quant, Some((d.t[1], d.t[3], d.i[2])));
            }
        }
        check_operand_dependencies(&shared.progs[0]);
    }
    #[test]
    fn fused_ab_preserves_two_bf16_outputs_and_batch_geometry() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_QWEN_FUSE_AB", "0"),
            ("PLOW_QWEN_AB_BLOCKS", "12"),
        ]);
        let c = Config::parse(&fixture());
        for batch in [1, 4] {
            std::env::set_var("PLOW_QWEN_FUSE_AB", "0");
            {
                let mut config = emit_config::active().clone();
                config.qwen_fuse_ab = false;
                emit_config::install(config);
            }
            let plain = model_prefill(&c, 8192, 132, 0, &[128], batch, false);
            std::env::set_var("PLOW_QWEN_FUSE_AB", "1");
            {
                let mut config = emit_config::active().clone();
                config.qwen_fuse_ab = true;
                emit_config::install(config);
            }
            let fused = model_prefill(&c, 8192, 132, 0, &[128], batch, false);
            assert_eq!(plain.progs[0].insts, fused.progs[0].insts);
            assert_eq!(plain.progs[1].insts.len(), fused.progs[1].insts.len() + 3);
            let ops: Vec<_> = fused.progs[1]
                .insts
                .iter()
                .filter(|d| d.op == DevOp::GemvQkv as u16)
                .collect();
            assert_eq!(ops.len(), 3);
            for d in ops {
                assert_eq!(d.blocks, 12);
                assert_eq!(&d.i[..5], &[batch, 48, 5120, 48, 0]);
                assert_eq!(&d.t[5..7], &[TENSOR_NONE, TENSOR_NONE]);
                for (out, weight, suffix) in [(d.t[0], d.t[2], "a"), (d.t[3], d.t[4], "b")] {
                    assert_eq!(
                        fused.tensors[out as usize].name,
                        format!("act.gdn.{suffix}")
                    );
                    assert_eq!(
                        fused.tensors[out as usize].bytes,
                        plain.tensors[out as usize].bytes
                    );
                    assert!(fused.tensors[out as usize].bytes >= u64::from(batch) * 48 * 2);
                    assert!(fused.tensors[weight as usize]
                        .name
                        .ends_with(&format!("in_proj_{suffix}.weight")));
                    assert_eq!(fused.tensors[weight as usize].bytes, 48 * 5120 * 2);
                }
            }
        }
    }
    fn fixture() -> Value {
        serde_json::json!({"model_type":"qwen3_5", "text_config": {
            "hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,
            "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,
            "linear_num_key_heads":16,"linear_num_value_heads":48,
            "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
            "num_hidden_layers":4,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
            "attention_bias":false,"attn_output_gate":true,"hidden_act":"silu","mamba_ssm_dtype":"float32",
            "output_gate_type":"swish","tie_word_embeddings":false,"rms_norm_eps":1e-6,
            "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"rope_type":"default"}
        }})
    }
    #[test]
    fn hybrid_decode_has_exact_state_and_projection_contract() {
        let _env = crate::test_env::env_guard();
        let c = Config::parse(&fixture());
        let m = model(&c, 8192, 132, 0, false, 1);
        assert_eq!(m.prog_t, vec![1]);
        assert!(m.kv_row_insts.is_empty());
        let tensor = |name: &str| m.tensors.iter().find(|t| t.name == name).unwrap();
        assert_eq!(tensor("state.qwen.0.conv").bytes, 10240 * 3 * 2);
        assert_eq!(tensor("state.qwen.0.gdn").bytes, 48 * 128 * 128 * 4);
        assert_eq!(
            tensor("model.language_model.layers.0.linear_attn.A_log").bytes,
            48 * 2
        );
        assert_eq!(
            tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight").bytes,
            48 * 5120 * 2
        );
        assert_eq!(
            tensor("model.language_model.layers.3.self_attn.q_proj.weight").bytes,
            24 * 256 * 2 * 5120 * 2
        );
        assert_eq!(tensor("kv.3.k").bytes, 4 * 256 * 8192 * 2);
        assert_eq!(tensor("act.logits").bytes, 248320 * 2);
        assert_eq!(tensor("in.pos").bytes, 8192 * 4);
        assert_eq!(tensor("in.ids").bytes, 8192 * 4);
        assert_eq!(tensor("in.active").bytes, 4);
        assert_eq!(tensor("in.kvlen").bytes, 4);
        assert_eq!(tensor("lm_head.weight").bytes, 248320 * 5120 * 2);
        assert!(!m.tensors.iter().any(|t| t.name == "kv.0.k"));
        let insts = &m.progs[0].insts;
        assert_eq!(
            insts
                .iter()
                .filter(|d| d.op == DevOp::QwenGdnStep as u16)
                .count(),
            3
        );
        let step = insts
            .iter()
            .find(|d| d.op == DevOp::QwenGdnStep as u16)
            .unwrap();
        assert_eq!(&step.i[..6], &[16, 48, 128, 128, 1, 0]);
        assert_eq!(m.tensors[step.t[7] as usize].name, "in.active");
        let norm = insts
            .iter()
            .find(|d| d.op == DevOp::QwenHeadNormRope as u16)
            .unwrap();
        assert_eq!(&norm.i[..6], &[24, 256, 64, 1, 0, 1]);
        let fa = insts
            .iter()
            .find(|d| d.op == DevOp::FlashDecode as u16)
            .unwrap();
        assert_eq!(&fa.i[..7], &[1, 24, 4, 8192, 0, 11, 256]);
        assert_eq!(fa.i[7], u32::MAX);
        assert_eq!(m.gen.len(), 2);
        assert!(m.gen.iter().all(|g| g.hd == 64 && g.frac == 1.0));
        let man = manifest::build(&m, "sm_90a", &LeanReport::skipped("fixture"));
        let req = man["backends"]["nvcc"]["requires"].as_array().unwrap();
        assert!(req.contains(&serde_json::json!("PLOW_NV_QWEN_GDN=1")));
        assert!(req.contains(&serde_json::json!("PLOW_NV_FA_GF=2")));
    }
    #[test]
    fn batch_four_has_independent_slots_and_batched_projections() {
        let _env = crate::test_env::env_guard();
        let c = Config::parse(&fixture());
        for fp8 in [false, true] {
            let m = model(&c, 8192, 132, 0, fp8, 4);
            assert_eq!(m.prog_t, vec![4]);
            let bytes = |name: &str| m.tensors.iter().find(|t| t.name == name).unwrap().bytes;
            for layer in 0..3 {
                assert_eq!(
                    bytes(&format!("state.qwen.{layer}.conv")),
                    4 * 10240 * 3 * 2
                );
                assert_eq!(
                    bytes(&format!("state.qwen.{layer}.gdn")),
                    4 * 48 * 128 * 128 * 4
                );
            }
            for name in ["kv.3.k", "kv.3.v"] {
                assert_eq!(bytes(name), 4 * 4 * 256 * 8192 * 2);
            }
            assert_eq!(bytes("act.logits"), 4 * 248320 * 2);
            assert_eq!(bytes("act.amax"), 4 * AMAX_BLOCKS as u64 * 8);
            assert_eq!(bytes("in.active"), 16);
            assert_eq!(bytes("in.kvlen"), 16);
            assert_eq!(bytes("in.ids"), 8192 * 4);
            assert_eq!(bytes("in.pos"), 8192 * 4);
            let insts = &m.progs[0].insts;
            let projections: Vec<_> = insts
                .iter()
                .filter(|d| d.op == DevOp::Gemv as u16 || d.op == DevOp::GemvFp8 as u16)
                .collect();
            assert_eq!(projections.len(), 32);
            assert!(projections.iter().all(|d| d.i[0] == 4));
            assert_eq!(
                projections
                    .iter()
                    .filter(|d| d.i[1] == 48 && d.i[2] == 5120)
                    .count(),
                6
            );
            for d in insts {
                if d.op == DevOp::QwenGdnStep as u16 {
                    assert_eq!(d.i[4], 4);
                }
                if d.op == DevOp::QwenGdnConv as u16 {
                    assert_eq!(d.i[2], 4);
                }
                if d.op == DevOp::QwenHeadNormRope as u16 {
                    assert_eq!(d.i[3], 4);
                    assert_eq!(d.i[6], 0);
                }
                if d.op == DevOp::FlashDecode as u16 {
                    assert_eq!(d.i[0], 4);
                    assert_eq!(d.i[7], u32::MAX);
                }
                if d.op == DevOp::Argmax as u16 || d.op == DevOp::ArgmaxFin as u16 {
                    assert_eq!(d.i[1], 4);
                }
            }
        }
    }
    #[test]
    fn prefill_tma_is_opt_in_and_preserves_recipes_across_buckets() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        struct Restore(emit_config::EmitConfig);
        impl Drop for Restore {
            fn drop(&mut self) {
                emit_config::install(self.0.clone());
            }
        }
        let restore = Restore(emit_config::active().clone());
        let mut cfg = restore.0.clone();
        cfg.tma_gemm = false;
        emit_config::install(cfg.clone());
        let c = Config::parse(&fixture());
        let plain = model_prefill(&c, 8192, 132, 0, &[128], 1, false);
        assert!(!plain.tensors.iter().any(|t| t.name.starts_with("tmap.")));
        assert!(plain.progs[0]
            .insts
            .iter()
            .filter(|d| d.op == DevOp::Gemm as u16)
            .all(|d| d.i[6] == 0 && d.i[7] == 0));
        cfg.tma_gemm = true;
        emit_config::install(cfg);
        let mapped = model_prefill(&c, 8192, 132, 0, &[128, 256], 1, false);
        for (pf, rows) in mapped.progs[..2].iter().zip([128, 256]) {
            let projections: Vec<_> = pf
                .insts
                .iter()
                .filter(|d| d.op == DevOp::Gemm as u16)
                .collect();
            assert_eq!(projections.len(), 31);
            for d in projections {
                for (handle, target, extent) in [(d.i[6], d.t[1], rows), (d.i[7], d.t[2], d.i[1])] {
                    let recipes: Vec<_> =
                        mapped.gen.iter().filter(|g| g.tensor == handle).collect();
                    assert_eq!(recipes.len(), 1);
                    let g = recipes[0];
                    assert_eq!(
                        (g.kind, g.aux, g.ctx, g.hd, g.scale),
                        (
                            packet::rope::GEN_TMAP_BF16,
                            target as u32,
                            extent,
                            d.i[2],
                            128
                        )
                    );
                    assert_eq!(mapped.tensors[handle as usize].bytes, 128);
                }
            }
            let head = pf
                .insts
                .iter()
                .find(|d| d.op == DevOp::Gemv as u16)
                .unwrap();
            assert_eq!(
                (head.i[0], head.i[4], head.i[6], head.i[7]),
                (1, rows - 1, 0, 0)
            );
        }
        assert!(mapped.progs[2]
            .insts
            .iter()
            .filter(|d| d.op == DevOp::Gemv as u16)
            .all(|d| d.i[6] == 0 && d.i[7] == 0));
    }

    #[test]
    fn prefill_batch_four_keeps_capacity_separate_from_query_rows() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let c = Config::parse(&fixture());
        let m = model_prefill(&c, 8192, 132, 0, &[128], 4, false);
        assert_eq!(m.prog_t, vec![128, 4]);
        let bytes = |name: &str| m.tensors.iter().find(|t| t.name == name).unwrap().bytes;
        assert_eq!(bytes("act.gdn.outstate"), 48 * 128 * 128 * 4);
        assert_eq!(bytes("in.active"), 16);
        assert_eq!(bytes("in.kvlen"), 16);
        assert_eq!(bytes("in.prefill_active"), 128 * 4);
        assert_eq!(bytes("act.logits"), 4 * 248320 * 2);
        assert_eq!(bytes("act.amax"), 4 * AMAX_BLOCKS as u64 * 8);
        for layer in 0..3 {
            assert_eq!(
                bytes(&format!("state.qwen.{layer}.conv")),
                4 * 10240 * 3 * 2
            );
            assert_eq!(
                bytes(&format!("state.qwen.{layer}.gdn")),
                4 * 48 * 128 * 128 * 4
            );
        }
        assert_eq!(bytes("kv.3.k"), 4 * 4 * 256 * 8192 * 2);
        assert_eq!(bytes("kv.3.v"), 4 * 4 * 256 * 8192 * 2);
        for d in &m.progs[0].insts {
            if d.op == DevOp::Gemm as u16 || d.op == DevOp::QwenGdnPrefill as u16 {
                assert_eq!(d.i[0], 128);
            }
            if d.op == DevOp::Gemv as u16 {
                assert_eq!((d.i[0], d.i[4]), (1, 127));
            }
            if d.op == DevOp::Argmax as u16 || d.op == DevOp::ArgmaxFin as u16 {
                assert_eq!(d.i[1], 1);
            }
        }
        for d in &m.progs[1].insts {
            if d.op == DevOp::Gemv as u16 {
                assert_eq!(d.i[0], 4);
            }
            if d.op == DevOp::QwenGdnStep as u16 {
                assert_eq!(d.i[4], 4);
            }
            if d.op == DevOp::QwenGdnConv as u16 {
                assert_eq!(d.i[2], 4);
            }
        }
    }

    #[test]
    fn decode_lt_isolates_only_bf16_body_projections() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_QWEN_DECODE_LT", "0")]);
        let c = Config::parse(&fixture());
        for batch in [1, 4] {
            std::env::set_var("PLOW_QWEN_DECODE_LT", "0");
            {
                let mut config = emit_config::active().clone();
                config.qwen_decode_lt = false;
                emit_config::install(config);
            }
            let plain = model_prefill(&c, 8192, 132, 0, &[128], batch, false);
            std::env::set_var("PLOW_QWEN_DECODE_LT", "1");
            {
                let mut config = emit_config::active().clone();
                config.qwen_decode_lt = true;
                emit_config::install(config);
            }
            let isolated = model_prefill(&c, 8192, 132, 0, &[128], batch, false);
            assert_eq!(plain.progs[0].gq_seg_ofs, isolated.progs[0].gq_seg_ofs);
            assert_eq!(plain.progs[0].waits, isolated.progs[0].waits);
            let a = &plain.progs[1];
            let b = &isolated.progs[1];
            assert_eq!(a.waits, b.waits);
            assert_eq!(a.succs, b.succs);
            assert_eq!(a.n_counter, b.n_counter);
            let mut count = 0;
            for (ix, d) in b.insts.iter().enumerate() {
                if d.op != DevOp::Gemv as u16 {
                    continue;
                }
                let seg = b
                    .gq_stream
                    .iter()
                    .find(|e| e.inst == ix as u32)
                    .unwrap()
                    .seg;
                let entries: Vec<_> = b.gq_stream.iter().filter(|e| e.seg == seg).collect();
                let weight = &isolated.tensors[d.t[2] as usize].name;
                if weight.contains(".layers.") {
                    count += 1;
                    assert!(entries.iter().all(|e| e.inst == ix as u32));
                    assert_eq!(d.i[0], batch);
                } else {
                    assert!(entries.iter().any(|e| e.inst != ix as u32));
                }
            }
            assert_eq!(count, 31);
        }
    }

    #[test]
    fn prefill_keeps_state_and_external_segments_exact() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let c = Config::parse(&fixture());
        let m = model_prefill(&c, 8192, 132, 0, &[128], 1, false);
        assert_eq!(m.prog_t, vec![128, 1]);
        let tensor = |name: &str| m.tensors.iter().find(|t| t.name == name).unwrap();
        assert_eq!(tensor("state.qwen.0.gdn").bytes, 48 * 128 * 128 * 4);
        assert_eq!(tensor("state.qwen.0.conv").bytes, 10240 * 3 * 2);
        assert_eq!(tensor("kv.3.k").bytes, 4 * 256 * 8192 * 2);
        assert_eq!(tensor("in.active").bytes, 4);
        assert_eq!(tensor("in.prefill_active").bytes, 128 * 4);
        assert_eq!(tensor("act.logits").bytes, 248320 * 2);
        let pf = &m.progs[0];
        let external: Vec<_> = pf
            .insts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.op == DevOp::QwenGdnPrefill as u16)
            .collect();
        assert_eq!(external.len(), 3);
        for (ix, inst) in external {
            assert_eq!(&inst.i[..5], &[128, 16, 48, 128, 128]);
            let entries: Vec<_> = pf
                .gq_stream
                .iter()
                .filter(|e| e.inst == ix as u32)
                .collect();
            assert_eq!(entries.len(), 1);
            let seg = entries[0].seg;
            assert!(pf
                .gq_stream
                .iter()
                .filter(|e| e.seg == seg)
                .all(|e| e.inst == ix as u32));
            assert!(pf.insts[ix + 1].op == DevOp::QwenGatedNorm as u16);
        }
        assert_eq!(
            pf.insts
                .iter()
                .filter(|d| d.op == DevOp::QwenGdnConvPrefill as u16)
                .count(),
            3
        );
        assert!(!pf.insts.iter().any(|d| d.op == DevOp::QwenGdnStep as u16));
        let head = pf
            .insts
            .iter()
            .find(|d| d.op == DevOp::Gemv as u16)
            .unwrap();
        assert_eq!((head.i[0], head.i[4]), (1, 127));
        let fa = pf
            .insts
            .iter()
            .find(|d| d.op == DevOp::FlashPrefill as u16)
            .unwrap();
        assert_eq!(&fa.i[..6], &[128, 128, 24, 4, 0, 0]);
        assert_eq!(fa.j[1], u32::MAX);
        assert!(m.progs[1]
            .insts
            .iter()
            .filter(|d| d.op == DevOp::QwenHeadNormRope as u16)
            .all(|d| d.i[6] == 0));
    }

    #[test]
    fn fp8_quantizes_all_projections_without_duplicate_bf16_weights() {
        let _env = crate::test_env::env_guard();
        let mut v = fixture();
        v["text_config"]["num_hidden_layers"] = serde_json::json!(64);
        v["text_config"]["layer_types"] = serde_json::json!((0..64)
            .map(|l| if l % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>());
        let c = Config::parse(&v);
        let m = model(&c, 8192, 132, 0, true, 1);
        let insts = &m.progs[0].insts;
        assert_eq!(
            insts
                .iter()
                .filter(|d| d.op == DevOp::GemvFp8 as u16)
                .count(),
            496
        );
        assert_eq!(
            insts.iter().filter(|d| d.op == DevOp::Gemv as u16).count(),
            1
        );
        for d in insts.iter().filter(|d| d.op == DevOp::GemvFp8 as u16) {
            let w = &m.tensors[d.t[2] as usize];
            let scale = &m.tensors[d.t[5] as usize];
            assert!(w.name.starts_with("fp8/"));
            assert_eq!(scale.name, format!("{}_scale", w.name));
            assert_eq!(w.bytes, d.i[1] as u64 * d.i[2] as u64);
            assert_eq!(scale.bytes, d.i[1] as u64 * 4);
            assert!(!m
                .tensors
                .iter()
                .any(|t| Some(t.name.as_str()) == w.name.strip_prefix("fp8/")));
        }
        for suffix in ["A_log", "dt_bias", "conv1d.weight", "norm.weight"] {
            let name = format!("model.language_model.layers.0.linear_attn.{suffix}");
            assert!(m.tensors.iter().any(|t| t.name == name));
            assert!(!m.tensors.iter().any(|t| t.name == format!("fp8/{name}")));
        }
        let head = insts.iter().find(|d| d.op == DevOp::Gemv as u16).unwrap();
        assert_eq!(m.tensors[head.t[2] as usize].name, "lm_head.weight");
        let man = manifest::build(&m, "sm_90a", &LeanReport::skipped("fixture"));
        assert_eq!(man["features"]["fp8_weights"], true);
        assert_eq!(man["features"]["w8a8"], false);
    }

    #[test]
    #[should_panic(expected = "native checkpoint FP8 lowering")]
    fn rejects_unimplemented_fp8_checkpoint() {
        let mut v = fixture();
        v["quantization_config"] = serde_json::json!({"quant_method":"fp8"});
        Config::parse(&v);
    }
    #[test]
    fn attention_role_preserves_packets_and_combines_with_fp8_role() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        let c = Config::parse(&fixture());
        for fp8 in [false, true] {
            let mut cfg = original.clone();
            cfg.fp8 = fp8;
            cfg.w8a8 = fp8;
            cfg.qwen_w8a8_prefill = fp8;
            cfg.fp8_pf_gemm_role = fp8;
            cfg.fp8_pf_isolate = false;
            cfg.attention_pf_role = false;
            cfg.attention_pf_isolate = false;
            emit_config::install(cfg.clone());
            let plain = model_prefill(&c, 65536, 132, 0, &[128, 1024, 4096, 8192], 1, fp8);
            cfg.attention_pf_isolate = true;
            emit_config::install(cfg.clone());
            let control = model_prefill(&c, 65536, 132, 0, &[128, 1024, 4096, 8192], 1, fp8);
            let meta = prefill_segment_roles(&control).unwrap();
            let meta: Value = serde_json::from_slice(&meta.data).unwrap();
            assert!(meta["objects"].get("2").is_none());
            cfg.attention_pf_role = true;
            cfg.attention_pf_isolate = false;
            emit_config::install(cfg);
            let candidate = model_prefill(&c, 65536, 132, 0, &[128, 1024, 4096, 8192], 1, fp8);
            let meta = prefill_segment_roles(&candidate).unwrap();
            let meta: Value = serde_json::from_slice(&meta.data).unwrap();
            assert_eq!(meta["objects"]["2"]["abi"], "attention_sm90_hd256_v1");
            assert_eq!(meta["objects"].get("1").is_some(), fp8);
            for (index, ((a, b), c)) in plain
                .progs
                .iter()
                .zip(&control.progs)
                .zip(&candidate.progs)
                .enumerate()
            {
                assert_eq!(a.insts, b.insts);
                assert_eq!(a.waits, b.waits);
                assert_eq!(a.succs, b.succs);
                assert_eq!(a.n_counter, b.n_counter);
                assert_eq!(b.insts, c.insts);
                assert_eq!(b.stream, c.stream);
                assert_eq!(b.gq_seg_ofs, c.gq_seg_ofs);
                check_operand_dependencies(c);
                if index == 4 {
                    assert_eq!(a.stream, c.stream);
                    assert_eq!(a.gq_seg_ofs, c.gq_seg_ofs);
                    continue;
                }
                let roles = meta["programs"][index]["roles"].as_array().unwrap();
                assert_eq!(roles.len(), c.gq_seg_ofs.len() - 1);
                assert_eq!(roles.iter().filter(|r| **r == 2).count(), 2);
                for (ix, d) in c.insts.iter().enumerate().filter(|(_, d)| {
                    d.op == DevOp::FlashPrefill as u16 || d.op == DevOp::FlashMerge as u16
                }) {
                    let entries: Vec<_> =
                        c.stream.iter().filter(|e| e.inst as usize == ix).collect();
                    assert_eq!(entries.len(), d.blocks as usize);
                    let seg = entries[0].seg;
                    assert_eq!(roles[seg as usize], 2);
                    assert!(c
                        .stream
                        .iter()
                        .all(|e| (e.inst as usize == ix) == (e.seg == seg)));
                    let mut slices: Vec<_> = entries.iter().map(|e| e.slice).collect();
                    slices.sort_unstable();
                    assert_eq!(slices, (0..u32::from(d.blocks)).collect::<Vec<_>>());
                }
            }
        }
        emit_config::install(original);
    }

    #[test]
    fn w8a8_prefill_isolation_preserves_arithmetic_and_dependencies() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_QWEN_W8A8_PREFILL", "1"),
            ("PLOW_QWEN_SHARE_QUANT", "1"),
        ]);
        let original = emit_config::active().clone();
        let mut cfg = original.clone();
        cfg.fp8 = true;
        cfg.w8a8 = true;
        cfg.fp8_pf_isolate = false;
        cfg.fp8_pf_gemm_role = false;
        emit_config::install(cfg.clone());
        let mut v = fixture();
        v["text_config"]["num_hidden_layers"] = serde_json::json!(64);
        v["text_config"]["layer_types"] = serde_json::json!((0..64)
            .map(|l| if l % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>());
        let c = Config::parse(&v);
        let plain = model_prefill(&c, 32768, 132, 0, &[128, 1024, 4096, 8192], 1, true);
        cfg.fp8_pf_isolate = true;
        emit_config::install(cfg.clone());
        let isolated = model_prefill(&c, 32768, 132, 0, &[128, 1024, 4096, 8192], 1, true);
        let control = prefill_segment_roles(&isolated).unwrap();
        let control: Value = serde_json::from_slice(&control.data).unwrap();
        assert!(control["objects"].as_object().unwrap().is_empty());
        assert!(control["programs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["roles"].as_array().unwrap().iter().all(|r| r == 0)));
        cfg.fp8_pf_gemm_role = true;
        emit_config::install(cfg.clone());
        let candidate = prefill_segment_roles(&isolated).unwrap();
        let candidate: Value = serde_json::from_slice(&candidate.data).unwrap();
        assert_eq!(candidate["objects"]["1"]["abi"], "fp8_gemm_tma128_v1");
        assert!(candidate["programs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["roles"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| *r == 1)
                .count()
                == 496));
        cfg.fp8_pf_gemm_role = false;
        cfg.fp8_pf_isolate = false;
        emit_config::install(cfg);
        assert!(prefill_segment_roles(&plain).is_none());
        emit_config::install(original);
        for (a, b) in plain.progs.iter().zip(&isolated.progs) {
            assert_eq!(a.insts, b.insts);
            assert_eq!(a.waits, b.waits);
            assert_eq!(a.succs, b.succs);
            assert_eq!(a.n_counter, b.n_counter);
            check_operand_dependencies(b);
        }
        assert_eq!(
            plain.progs.last().unwrap().gq_seg_ofs,
            isolated.progs.last().unwrap().gq_seg_ofs
        );
        for (a, b) in plain.progs[..4].iter().zip(&isolated.progs[..4]) {
            let mut count = 0;
            for (ix, d) in b.insts.iter().enumerate() {
                if d.op != DevOp::GemmFp8 as u16 {
                    continue;
                }
                let seg = b.stream.iter().find(|e| e.inst as usize == ix).unwrap().seg;
                assert!(b
                    .stream
                    .iter()
                    .all(|e| (e.inst as usize == ix) == (e.seg == seg)));
                count += 1;
            }
            eprintln!(
                "FP8 PF rows={} GEMMs={} launches {} -> {}",
                b.insts
                    .iter()
                    .find(|d| d.op == DevOp::GemmFp8 as u16)
                    .unwrap()
                    .i[0],
                count,
                a.gq_seg_ofs.len() - 1,
                b.gq_seg_ofs.len() - 1
            );
        }
    }

    #[test]
    fn w8a8_prefill_bucket_maps_keep_global_rows_and_maximum_capacity() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_QWEN_W8A8_PREFILL", "1"),
            ("PLOW_QWEN_SHARE_QUANT", "1"),
            ("PLOW_QWEN_FP8_M1_TMA", "1"),
        ]);
        let original = emit_config::active().clone();
        let mut cfg = original.clone();
        cfg.fp8 = true;
        cfg.w8a8 = true;
        emit_config::install(cfg);
        let c = Config::parse(&fixture());
        let m = model_prefill(&c, 32768, 132, 0, &[128, 1024, 4096, 8192], 1, true);
        emit_config::install(original);
        assert_eq!(m.prog_t, [128, 1024, 4096, 8192, 1]);
        let mut first_maps = std::collections::BTreeSet::new();
        for (g, rows) in m.progs[..4].iter().zip([128, 1024, 4096, 8192]) {
            check_operand_dependencies(g);
            assert_eq!(
                g.insts
                    .iter()
                    .filter(|d| d.op == DevOp::QuantFp8 as u16)
                    .count(),
                16
            );
            for d in g.insts.iter().filter(|d| d.op == DevOp::GemmFp8 as u16) {
                assert_eq!(d.i[0], rows);
                assert_eq!(m.tensors[d.t[1] as usize].bytes, 8192 * u64::from(d.i[2]));
                assert_eq!(m.tensors[d.t[3] as usize].bytes, 8192 * 4);
                let a = m.gen.iter().find(|r| r.tensor == d.i[6]).unwrap();
                assert_eq!((a.ctx, a.hd, a.scale), (rows, d.i[2], 128));
                let w = m.gen.iter().find(|r| r.tensor == d.i[7]).unwrap();
                assert_eq!((w.ctx, w.hd, w.scale), (d.i[1], d.i[2], 128));
            }
            let first = g
                .insts
                .iter()
                .find(|d| d.op == DevOp::GemmFp8 as u16)
                .unwrap();
            assert!(first_maps.insert(first.i[6]));
        }
        assert_eq!(
            m.tensors
                .iter()
                .find(|t| t.name == "act.logits")
                .unwrap()
                .bytes,
            248320 * 2
        );
        assert_eq!(
            m.tensors
                .iter()
                .find(|t| t.name == "in.active")
                .unwrap()
                .bytes,
            4
        );
    }

    #[test]
    fn single_blocks_keep_real_layer_weights_and_carried_state() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        for fp8 in [false, true] {
            let mut cfg = original.clone();
            cfg.fp8 = fp8;
            cfg.w8a8 = fp8;
            cfg.qwen_w8a8_prefill = true;
            emit_config::install(cfg);
            for layer in [0, 3] {
                let mut c = Config::parse(&fixture());
                c.block = Some(layer);
                let m = model_prefill(&c, 8192, 132, 0, &[128], 1, fp8);
                assert_eq!(m.prog_t, [128, 1]);
                for g in &m.progs {
                    assert!(g.insts.iter().all(|d| !matches!(
                        DevOp::from_u16(d.op),
                        Some(DevOp::Embed | DevOp::Argmax | DevOp::ArgmaxFin)
                    )));
                    assert_eq!(
                        g.insts
                            .iter()
                            .filter(|d| matches!(
                                DevOp::from_u16(d.op),
                                Some(DevOp::GemmFp8 | DevOp::Gemv | DevOp::Gemm)
                            ))
                            .count(),
                        if layer == 0 { 8 } else { 7 }
                    );
                    check_operand_dependencies(g);
                }
                assert!(!m.tensors.iter().any(|t| t.name.contains("embed_tokens")
                    || t.name.contains("lm_head")
                    || t.name == "model.language_model.norm.weight"));
                for t in m.tensors.iter().filter(|t| t.name.contains(".layers.")) {
                    assert!(t.name.contains(&format!(".layers.{layer}.")));
                }
                assert_eq!(
                    m.tensors
                        .iter()
                        .filter(|t| t.name.starts_with("state.qwen."))
                        .count(),
                    if layer == 0 { 2 } else { 0 }
                );
                assert_eq!(
                    m.tensors
                        .iter()
                        .filter(|t| t.name.starts_with("kv."))
                        .count(),
                    if layer == 3 { 2 } else { 0 }
                );
                assert!(m
                    .tensors
                    .iter()
                    .find(|t| t.name == "in.active")
                    .unwrap()
                    .init
                    .is_none());
            }
        }
        emit_config::install(original);
    }

    #[test]
    fn w8a8_prefill_preserves_precision_maps_and_quant_dependencies() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        struct Restore(emit_config::EmitConfig);
        impl Drop for Restore {
            fn drop(&mut self) {
                emit_config::install(self.0.clone());
                for name in [
                    "PLOW_QWEN_W8A8_PREFILL",
                    "PLOW_QWEN_SHARE_QUANT",
                    "PLOW_QWEN_FP8_M1_TMA",
                ] {
                    unsafe { std::env::remove_var(name) };
                }
            }
        }
        let restore = Restore(emit_config::active().clone());
        let mut cfg = restore.0.clone();
        cfg.fp8 = false;
        cfg.w8a8 = false;
        emit_config::install(cfg.clone());
        let c = Config::parse(&fixture());
        let bf16 = model_prefill(&c, 8192, 132, 0, &[128], 1, false);
        cfg.fp8 = true;
        cfg.w8a8 = true;
        emit_config::install(cfg);
        unsafe {
            std::env::set_var("PLOW_QWEN_W8A8_PREFILL", "1");
            {
                let mut config = emit_config::active().clone();
                config.qwen_w8a8_prefill = true;
                emit_config::install(config);
            }
            std::env::set_var("PLOW_QWEN_FP8_M1_TMA", "1");
            {
                let mut config = emit_config::active().clone();
                config.qwen_fp8_m1_tma = true;
                emit_config::install(config);
            }
        }
        for shared in ["0", "1"] {
            unsafe { std::env::set_var("PLOW_QWEN_SHARE_QUANT", shared) };
            {
                let mut config = emit_config::active().clone();
                config.qwen_share_quant = shared == "1";
                emit_config::install(config);
            }
            let m = model_prefill(&c, 8192, 132, 0, &[128], 1, true);
            assert_eq!(m.prog_t, [128, 1]);
            for (g, rows) in m.progs.iter().zip([128, 1]) {
                let matmuls: Vec<_> = g
                    .insts
                    .iter()
                    .filter(|d| d.op == DevOp::GemmFp8 as u16)
                    .collect();
                assert_eq!(matmuls.len(), 31);
                for d in matmuls {
                    assert_eq!(d.i[0], rows);
                    assert_eq!(m.tensors[d.t[1] as usize].bytes, 128 * u64::from(d.i[2]));
                    assert_eq!(m.tensors[d.t[3] as usize].bytes, 128 * 4);
                    assert_eq!(
                        m.tensors[d.t[0] as usize].bytes,
                        128 * u64::from(d.i[1]) * 2
                    );
                    for (map, target, extent) in [(d.i[6], d.t[1], rows), (d.i[7], d.t[2], d.i[1])]
                    {
                        if map == 0 {
                            assert_eq!(rows, 1);
                            continue;
                        }
                        let recipe = m.gen.iter().find(|g| g.tensor == map).unwrap();
                        assert_eq!(
                            (recipe.kind, recipe.aux, recipe.ctx, recipe.hd, recipe.scale),
                            (
                                packet::rope::GEN_TMAP_E4M3,
                                target as u32,
                                extent,
                                d.i[2],
                                if rows == 128 { 128 } else { 64 }
                            )
                        );
                    }
                }
                assert_eq!(
                    g.insts
                        .iter()
                        .filter(|d| d.op == DevOp::QuantFp8 as u16)
                        .count(),
                    if shared == "1" { 16 } else { 31 }
                );
                check_operand_dependencies(g);
            }
            for t in bf16.tensors.iter().filter(|t| {
                t.name.starts_with("state.")
                    || t.name.starts_with("kv.")
                    || t.name == "act.logits"
                    || t.name == "in.active"
            }) {
                assert_eq!(
                    m.tensors.iter().find(|q| q.name == t.name).unwrap().bytes,
                    t.bytes
                );
            }
            for suffix in ["A_log", "dt_bias", "conv1d.weight", "norm.weight"] {
                let name = format!("model.language_model.layers.0.linear_attn.{suffix}");
                assert!(m.tensors.iter().any(|t| t.name == name));
                assert!(!m.tensors.iter().any(|t| t.name == format!("fp8/{name}")));
            }
            assert_eq!(
                m.progs[0]
                    .insts
                    .iter()
                    .filter(|d| d.op == DevOp::QwenGdnPrefill as u16)
                    .count(),
                3
            );
            let head = m.progs[0]
                .insts
                .iter()
                .find(|d| d.op == DevOp::Gemv as u16)
                .unwrap();
            assert_eq!(m.tensors[head.t[2] as usize].name, "lm_head.weight");
            assert_eq!(head.i[0], 1);
        }
        assert!(
            std::panic::catch_unwind(|| model_prefill(&c, 8192, 132, 0, &[256], 1, true)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| model_prefill(&c, 8192, 132, 0, &[128], 4, true)).is_err()
        );
        unsafe { std::env::remove_var("PLOW_QWEN_W8A8_PREFILL") };
        {
            let mut config = emit_config::active().clone();
            config.qwen_w8a8_prefill = false;
            emit_config::install(config);
        }
        assert!(
            std::panic::catch_unwind(|| model_prefill(&c, 8192, 132, 0, &[128], 1, true)).is_err()
        );
    }

    #[test]
    fn w8a8_m1_tma_maps_describe_exact_weight_layout() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        let mut cfg = original.clone();
        cfg.fp8 = true;
        cfg.w8a8 = true;
        emit_config::install(cfg);
        let c = Config::parse(&fixture());
        let plain = model(&c, 8192, 132, 0, true, 1);
        unsafe { std::env::set_var("PLOW_QWEN_FP8_M1_TMA", "1") };
        {
            let mut config = emit_config::active().clone();
            config.qwen_fp8_m1_tma = true;
            emit_config::install(config);
        }
        let mapped = model(&c, 8192, 132, 0, true, 1);
        unsafe { std::env::remove_var("PLOW_QWEN_FP8_M1_TMA") };
        {
            let mut config = emit_config::active().clone();
            config.qwen_fp8_m1_tma = false;
            emit_config::install(config);
        }
        emit_config::install(original);
        let mut count = 0;
        for (a, b) in plain.progs[0].insts.iter().zip(&mapped.progs[0].insts) {
            assert_eq!(a.op, b.op);
            if b.op == DevOp::GemmFp8 as u16 {
                count += 1;
                assert_eq!(a.i[7], 0);
                let g = mapped.gen.iter().find(|g| g.tensor == b.i[7]).unwrap();
                assert_eq!(
                    (g.kind, g.aux, g.ctx, g.hd, g.scale),
                    (
                        packet::rope::GEN_TMAP_E4M3,
                        b.t[2] as u32,
                        b.i[1],
                        b.i[2],
                        64
                    )
                );
                assert_eq!(mapped.tensors[b.i[7] as usize].bytes, 128);
            }
        }
        assert_eq!(count, 31);
        check_operand_dependencies(&mapped.progs[0]);
    }

    #[test]
    fn w8a8_decode_quantizes_body_inputs_and_keeps_head_wide() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        struct Restore(emit_config::EmitConfig);
        impl Drop for Restore {
            fn drop(&mut self) {
                emit_config::install(self.0.clone());
            }
        }
        let restore = Restore(emit_config::active().clone());
        let mut cfg = restore.0.clone();
        cfg.fp8 = true;
        cfg.w8a8 = true;
        emit_config::install(cfg);
        let c = Config::parse(&fixture());
        let m = model(&c, 8192, 132, 0, true, 1);
        let g = &m.progs[0];
        assert_eq!(g.gq_seg_ofs.len(), 2);
        let mut count = 0;
        for (ix, d) in g.insts.iter().enumerate() {
            if d.op != DevOp::GemmFp8 as u16 {
                continue;
            }
            count += 1;
            let q = &g.insts[ix - 1];
            assert_eq!(q.op, DevOp::QuantFp8 as u16);
            assert_eq!((q.t[0], q.t[2]), (d.t[1], d.t[3]));
            assert_eq!(&q.i[..2], &[1, d.i[2]]);
            assert_eq!(m.tensors[d.t[1] as usize].bytes, u64::from(d.i[2]));
            assert_eq!(m.tensors[d.t[3] as usize].bytes, 4);
            assert_eq!(m.tensors[d.t[4] as usize].bytes, u64::from(d.i[1]) * 4);
            assert!(m.tensors[d.t[2] as usize].name.starts_with("fp8/"));
        }
        assert_eq!(count, 31);
        let head = g.insts.iter().find(|d| d.op == DevOp::Gemv as u16).unwrap();
        assert_eq!(m.tensors[head.t[2] as usize].name, "lm_head.weight");
        let man = manifest::build(&m, "sm_90a", &LeanReport::skipped("fixture"));
        assert_eq!(man["features"]["w8a8"], true);
        let req = man["backends"]["nvcc"]["requires"].as_array().unwrap();
        for flag in [
            "PLOW_NV_W8A8=1",
            "PLOW_NV_FP8_M1=1",
            "PLOW_NV_QUANT_FP8_VLLM=1",
        ] {
            assert!(req.contains(&serde_json::json!(flag)));
        }
        assert!(std::panic::catch_unwind(|| model(&c, 8192, 132, 0, true, 4)).is_err());
        assert!(std::panic::catch_unwind(|| model(&c, 8192, 132, 0, false, 1)).is_err());
    }
    #[test]
    #[should_panic(expected = "unsupported layer type")]
    fn rejects_unknown_hybrid_layer() {
        let mut v = fixture();
        v["text_config"]["layer_types"][0] = serde_json::json!("mamba");
        Config::parse(&v);
    }
}
