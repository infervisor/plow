use super::*;
use std::path::Path;

fn segment_roles(m: &Model) -> Option<packet::devbuild::SectionData> {
    let cfg = emit_config::active();
    let fp8 = cfg.fp8_pf_gemm_role || cfg.fp8_pf_isolate;
    let attention = cfg.attention_pf_role || cfg.attention_pf_isolate;
    let decode_cublaslt = cfg.decode_cublaslt;
    if !fp8 && !attention && !decode_cublaslt {
        return None;
    }
    let mut programs = Vec::new();
    let mut used = [false; 3];
    for (index, p) in m
        .progs
        .iter()
        .enumerate()
        .take(m.progs.len() - 1)
        .filter(|_| fp8 || attention)
    {
        let mut roles = vec![0_u8; p.gq_seg_ofs.len() - 1];
        let mut found_fp8 = false;
        for (ix, d) in p.insts.iter().enumerate() {
            let role = if fp8 && d.op == DevOp::GemmFp8 as u16 {
                assert!(
                    d.i[6] != 0 && d.i[7] != 0,
                    "FP8 role needs mapped prefill GEMMs"
                );
                found_fp8 = true;
                plow_asset::segment_roles::FP8_PREFILL_GEMM * u8::from(cfg.fp8_pf_gemm_role)
            } else if attention && d.op == DevOp::FlashPrefill as u16 {
                assert_eq!(d.i[6], 256, "attention role requires HD256");
                assert!(d.t[5..8].iter().all(|&t| t == TENSOR_NONE));
                plow_asset::segment_roles::PREFILL_ATTENTION * u8::from(cfg.attention_pf_role)
            } else if attention && d.op == DevOp::FlashMerge as u16 {
                assert_eq!(d.i[3], 256, "attention role requires HD256");
                plow_asset::segment_roles::PREFILL_ATTENTION * u8::from(cfg.attention_pf_role)
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
    let min_rows = cfg.decode_cublaslt_min_rows.unwrap_or(1);
    let decode_lo = if m.prog_t.len() == m.progs.len() {
        packet::devbuild::decode_rung_lo(&m.prog_t)
    } else {
        m.progs.len() - 1
    };
    for index in (decode_lo..m.progs.len())
        .filter(|&index| decode_cublaslt && m.prog_t.get(index).is_none_or(|&t| t >= min_rows))
    {
        let p = &m.progs[index];
        let mut roles = vec![plow_asset::segment_roles::INTERPRETER; p.gq_seg_ofs.len() - 1];
        for (ix, inst) in p.insts.iter().enumerate() {
            if inst.op != DevOp::Gemv as u16
                || !m.tensors[inst.t[2] as usize].name.contains(".layers.")
            {
                continue;
            }
            let seg = p
                .stream
                .iter()
                .find(|entry| entry.inst as usize == ix)
                .expect("cuBLASLt projection stream entry")
                .seg as usize;
            assert!(p
                .stream
                .iter()
                .all(|entry| (entry.inst as usize == ix) == (entry.seg as usize == seg)));
            roles[seg] = plow_asset::segment_roles::CUBLASLT;
        }
        assert!(
            roles.contains(&plow_asset::segment_roles::CUBLASLT),
            "cuBLASLt metadata requires eligible decode projections"
        );
        programs.push(serde_json::json!({"index":index,"roles":roles}));
    }
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
    decode_cublaslt: bool,
    fuse_ab: bool,
    ab_blocks: u32,
    projection_dag: bool,
    share_quant: bool,
    quant_input: Option<(u32, u32)>,
    /// Decode projection fusions (one walk for z/a/b, for k/v, for gate/up/SiLU) on every rung.
    fuse: bool,
    /// The residual + norm prologue (`Fold`) on this rung.
    fold: bool,
    /// `(stream, addend)` of a residual add waiting for the stream's next norm.
    pending_residual: Option<(u32, u32)>,
    ctx: u32,
    active: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
}

/// PLOW_QWEN_FOLD_NORM: the residual add and the RMSNorm feeding a decode projection run as that
/// projection's prologue (interp GEMV / GEMV_GLU arms, i3 == 4, op_qwen_gdn.cuh
/// d_qwen_fold_norm). Every block computes hn = norm(bf16(a + b)) for all rows itself, so the two
/// grid-wide dependency rounds of the separate Residual and QwenRmsNorm instructions (~10 us
/// each at this grid, 128 norms per step at B1) disappear. `a` and `out` differ: another block
/// may still be reading `a` (the residual stream ping-pongs between two buffers). Rungs above
/// PLOW_QWEN_FOLD_MAX (4) keep the separate instructions: every block redoing all rows costs
/// more than the barriers there (H100 27B, ms: B4 27.9 vs 29.8 unfolded, B8 34.5 vs 33.2, B32
/// 74.6 vs 60.8). plowrt's ladder dispatch (exec/gpu/decode_rung.rs) compares folded and
/// unfolded rungs as separate classes.
struct Fold {
    a: u32,
    b: u32,
    out: u32,
    gamma: u32,
}

impl Fold {
    fn fields(&self, d: &mut DevInst, a_slot: usize, eps: f32) {
        d.i[3] = 4;
        d.t[3] = self.out;
        d.i[a_slot] = self.a;
        d.i[6] = if self.b == TENSOR_NONE { 0 } else { self.b };
        d.i[7] = self.gamma;
        d.f[0] = eps;
        d.f[1] = 1.0;
    }
}

impl Emitter<'_> {
    fn proj_fold(
        &mut self,
        out: u32,
        src: u32,
        name: &str,
        n: u32,
        k: u32,
        f: &Fold,
        dep: u32,
    ) -> u32 {
        let w = self.weight(name, n as u64 * k as u64);
        let (batch, eps) = (self.batch, self.c.eps);
        self.b.emit(DevOp::Gemv, self.b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[out, src, w]);
            d.i[..3].copy_from_slice(&[batch, n, k]);
            f.fields(d, 5, eps);
        })
    }
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
                let blocks = if self.prefill || rows > 1 {
                    self.b.all()
                } else {
                    vec![0]
                };
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
            } else if rows == 1 && emit_config::active().qwen_fp8_m1_tma {
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
        if self.decode_cublaslt && name.contains(".layers.") {
            self.b.isolate(done);
        }
        done
    }
    fn norm(&mut self, out: u32, src: u32, name: &str, dep: u32) -> u32 {
        // The next norm overwrites the shared hn arena, ending its quantized input lifetime.
        self.quant_input = None;
        let gamma = self.weight(name, self.c.hidden as u64);
        // A pending residual add of this stream rides the norm (t4); one of another tensor
        // runs first as its own instruction.
        let (dep, radd) = match self.pending_residual.take() {
            Some((x, mixed)) if x == src => (dep, mixed),
            Some(pending) => {
                self.pending_residual = Some(pending);
                (self.flush_residual(dep), TENSOR_NONE)
            }
            None => (dep, TENSOR_NONE),
        };
        self.b.emit(
            DevOp::QwenRmsNorm,
            (0..self.batch.min(self.b.n_cu())).collect(),
            &[dep],
            |d| {
                d.t[..5].copy_from_slice(&[out, src, gamma, self.active, radd]);
                d.i[..2].copy_from_slice(&[self.c.hidden, self.batch]);
                d.f[0] = self.c.eps;
                d.f[1] = 1.0;
            },
        )
    }
    /// x += mixed, deferred onto the next RMSNorm of x (op_qwen_gdn.cuh d_qwen_rmsnorm `radd`):
    /// the same rounding as the separate instruction with one grid-wide dependency round per
    /// pair instead of two (128 pairs per program).
    fn residual(&mut self, x: u32, mixed: u32, dep: u32) -> u32 {
        debug_assert!(
            self.pending_residual.is_none(),
            "two residual adds without a norm"
        );
        self.pending_residual = Some((x, mixed));
        dep
    }
    fn flush_residual(&mut self, dep: u32) -> u32 {
        let Some((x, mixed)) = self.pending_residual.take() else {
            return dep;
        };
        // d_residual strides by nblk; one block for the B x 5120 decode rows cost 6 us per
        // instruction (128 per step) against the all-blocks spread.
        self.b.emit(DevOp::Residual, self.b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[x, x, mixed]);
            d.i[0] = self.batch * self.c.hidden;
            d.f[0] = 1.0;
        })
    }
    fn linear(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32, fold: Option<&Fold>) -> u32 {
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
        let qkv_name = format!("{p}.in_proj_qkv.weight");
        let dq = match fold {
            Some(f) => self.proj_fold(raw, hn, &qkv_name, channels, c.hidden, f, dep),
            None => self.proj(raw, hn, &qkv_name, channels, c.hidden, dep),
        };
        let conv_op = if self.prefill {
            DevOp::QwenGdnConvPrefill
        } else {
            DevOp::QwenGdnConv
        };
        let dc = self.b.emit(conv_op, self.b.all(), &[dq], |d| {
            d.t[..5].copy_from_slice(&[conv, raw, cw, history, self.active]);
            d.i[..3].copy_from_slice(&[channels, c.conv, self.batch]);
        });
        let (db, dz) = if self.fuse && self.fuse_ab {
            // z, a and b share the normalized input: one three-output walk (N = value + 2 hv).
            let wz = self.weight(
                &format!("{p}.in_proj_z.weight"),
                value as u64 * c.hidden as u64,
            );
            let wa = self.weight(
                &format!("{p}.in_proj_a.weight"),
                c.hv as u64 * c.hidden as u64,
            );
            let wb = self.weight(
                &format!("{p}.in_proj_b.weight"),
                c.hv as u64 * c.hidden as u64,
            );
            let zab_dep = if self.projection_dag { dq } else { dc };
            let dzab = self.b.emit(DevOp::GemvQkv, self.b.all(), &[zab_dep], |d| {
                d.t[..7].copy_from_slice(&[z, hn, wz, a, wa, beta, wb]);
                d.i[..5].copy_from_slice(&[self.batch, value, c.hidden, c.hv, c.hv]);
            });
            (dzab, dzab)
        } else {
            let dz = self.proj(
                z,
                hn,
                &format!("{p}.in_proj_z.weight"),
                value,
                c.hidden,
                if self.projection_dag { dq } else { dc },
            );
            // With the norm folded into the qkv projection, the a/b walk's input is that op's.
            let ab_dep = if self.projection_dag {
                if self.fold {
                    dq
                } else {
                    dep
                }
            } else {
                dz
            };
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
            (db, dz)
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
            // One final-state slot per decode slot: a packed prefill scans N requests at once
            // over N contiguous slots; plowrt gathers their slot states into `initstate` and
            // scatters the finals back from `outstate`.
            let state_slots =
                u64::from(self.decode_slots) * c.hv as u64 * c.dv as u64 * c.dk as u64 * 4;
            let outstate = self.b.tensor("act.gdn.outstate", state_slots);
            let _initstate = self.b.tensor("act.gdn.initstate", state_slots);
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
            let step = self.b.emit(DevOp::QwenGdnStep, self.b.all(), &deps, |d| {
                d.t.copy_from_slice(&[core, conv, a, beta, alog, dt, state, self.active]);
                d.i[..5].copy_from_slice(&[c.hk, c.hv, c.dk, c.dv, self.batch]);
                d.f[0] = 1.0 / (c.dk as f32).sqrt();
                d.f[1] = 1e-6;
            });
            // PLOW_QWEN_GDN_ISOLATE: the step gets its own segment on rungs from this many rows,
            // so the runtime can launch it at its own occupancy (gen_decode.rs) instead of inside
            // the interpreter window (B32: 144 vs 76 us per layer).
            if emit_config::active()
                .qwen_gdn_isolate
                .is_some_and(|rows| self.batch >= rows)
            {
                self.b.isolate(step);
            }
            step
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
                d.i[7] = if cache && emit_config::active().qwen_kv_interleave {
                    plow_asset::live_kv::LAYOUT_ROW_INTERLEAVED
                } else {
                    plow_asset::live_kv::LAYOUT_HEAD_MAJOR
                };
                d.f[0] = self.c.eps;
                d.f[1] = if normalize { 1.0 } else { 0.0 };
            })
    }
    fn full(&mut self, layer: usize, hn: u32, mixed: u32, dep: u32, fold: Option<&Fold>) -> u32 {
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
        let q_name = format!("{p}.q_proj.weight");
        let dp = match fold {
            Some(f) => self.proj_fold(packed, hn, &q_name, 2 * qd, c.hidden, f, dep),
            None => self.proj(packed, hn, &q_name, 2 * qd, c.hidden, dep),
        };
        let dq = self
            .b
            .emit(DevOp::QwenQGateSplit, self.b.all(), &[dp], |d| {
                d.t[..4].copy_from_slice(&[qg, gate, packed, self.active]);
                d.i[..3].copy_from_slice(&[c.heads, c.hd, self.batch]);
            });
        let dq = self.headnorm(q, qg, qn, c.heads, false, true, dq);
        let (dk, dv) = if self.fuse {
            // k and v share the normalized input: one two-output walk.
            let wk = self.weight(&format!("{p}.k_proj.weight"), kd as u64 * c.hidden as u64);
            let wv = self.weight(&format!("{p}.v_proj.weight"), kd as u64 * c.hidden as u64);
            let kv_dep = if self.projection_dag { dp } else { dq };
            let dkv = self.b.emit(DevOp::GemvQkv, self.b.all(), &[kv_dep], |d| {
                d.t[..7].copy_from_slice(&[kg, hn, wk, vg, wv, TENSOR_NONE, TENSOR_NONE]);
                d.i[..5].copy_from_slice(&[self.batch, kd, c.hidden, kd, 0]);
            });
            let dk = self.headnorm(kc, kg, kn, c.kv_heads, true, true, dkv);
            let dv = self.headnorm(vc, vg, TENSOR_NONE, c.kv_heads, true, false, dk);
            (dk, dv)
        } else {
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
            (
                dk,
                self.headnorm(vc, vg, TENSOR_NONE, c.kv_heads, true, false, dv),
            )
        };
        // PLOW_EMIT_GEN_KERNELS: one-split flash prefill with the fused bf16 epilogue (t5 = at,
        // no FlashMerge), the contract of a generated flash-prefill role (gen_flash_prefill.cu);
        // the interpreter's hd256 body honours the same operands when no role binds the rung.
        let gen_attention = self.prefill && emit_config::active().gen_kernels.is_some();
        // A decode entry in the list: the FlashDecode + FlashMerge pair gets its own segments,
        // so the runtime can replace the pair with one launch of the generated decode object
        // (gen_kernels::bind_decode records the role) or run the untouched interpreter pair.
        let gen_decode = !self.prefill
            && emit_config::active()
                .gen_kernels
                .as_deref()
                .is_some_and(|list| {
                    crate::gen_kernels::parse(list)
                        .is_ok_and(|entries| entries.iter().any(|e| e.decode))
                });
        // HD256 uses the slide GF selector in the Hopper interpreter even for full attention.
        let ns = if gen_attention {
            1
        } else if self.prefill {
            self.b
                .n_cu()
                .div_ceil(self.batch.div_ceil(64) * c.heads)
                .max(2)
        } else {
            let fallback = self.b.n_cu().div_ceil(c.heads / 2).max(1);
            let ns = crate::attention_decode_ns(
                self.batch,
                c.heads,
                c.kv_heads,
                self.b.n_cu(),
                fallback,
            );
            emit_config::active().ns_full_abs.unwrap_or(ns)
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
        // The hd256 role stages K/V through TMA when t[7] carries a GEN_TMAP_KV_PAIR (the
        // runtime encodes one descriptor pair per slot over [kv_heads][ctx][hd]); without it the
        // role falls back to cp.async staging at ~0.75x (hand harness, 24/4, a 1024-row chunk
        // over 14K context: 689 us with maps, 929 without). bf16 KV only: t[7] is the FP8-KV
        // V-scale slot.
        let fa_tm = (gen_attention && c.hd == 256).then(|| {
            let mut g = GenTensor::tmap_kv_pair_box(kc, vc, self.ctx, c.hd, c.kv_heads, 32);
            if emit_config::active().qwen_kv_interleave {
                g.low = plow_asset::live_kv::LAYOUT_ROW_INTERLEAVED as f64;
            }
            self.b.tensor_gen(&format!("tmap.kv.{kc}"), g.byte_len(), g)
        });
        let da = if self.prefill {
            self.b.emit(DevOp::FlashPrefill, self.b.all(), &[dv], |d| {
                d.t[..5].copy_from_slice(&[opart, mlpart, q, kc, vc]);
                if gen_attention {
                    d.t[5] = at;
                }
                if let Some(h) = fa_tm {
                    d.t[7] = h;
                }
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
        let dm = if gen_attention {
            self.b.isolate(da);
            da
        } else {
            self.b.emit(
                DevOp::FlashMerge,
                (0..c.heads.min(self.b.n_cu())).collect(),
                &[da],
                |d| {
                    d.t[..3].copy_from_slice(&[at, opart, mlpart]);
                    d.i[..4].copy_from_slice(&[self.batch, c.heads, ns, c.hd]);
                },
            )
        };
        if gen_decode
            || (self.prefill
                && !gen_attention
                && (emit_config::active().attention_pf_isolate
                    || emit_config::active().attention_pf_role))
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

#[cfg(test)]
fn model(c: &Config, ctx: u32, n_cu: u32, target: u32, fp8: bool, batch: u32) -> Model {
    ladder(c, ctx, n_cu, target, fp8, &[], &[batch])
}

#[cfg(test)]
fn model_prefill(
    c: &Config,
    ctx: u32,
    n_cu: u32,
    target: u32,
    buckets: &[u32],
    decode_slots: u32,
    fp8: bool,
) -> Model {
    assert!(!buckets.is_empty());
    ladder(c, ctx, n_cu, target, fp8, buckets, &[decode_slots])
}

/// Prefill buckets (ascending) then one decode program per rung (ascending); every program
/// adopts the tensor table of the one before it, so KV, recurrent state, `in.kvlen` and
/// `in.active` are sized at the widest rung (`decode_slots`) and a sequence keeps its slot
/// while the rung under it changes. `prog_t = buckets ++ rungs` (packet decode-ladder rule).
fn ladder(
    c: &Config,
    ctx: u32,
    n_cu: u32,
    target: u32,
    fp8: bool,
    buckets: &[u32],
    rungs: &[u32],
) -> Model {
    assert!(!rungs.is_empty() && rungs.windows(2).all(|w| w[0] < w[1]));
    let decode_slots = *rungs.last().unwrap();
    assert!(
        (1..=packet::devbuild::DECODE_RUNG_MAX).contains(&decode_slots),
        "Qwen decode rungs must lie in 1..={}",
        packet::devbuild::DECODE_RUNG_MAX
    );
    assert!(
        !fp8 || buckets.is_empty()
            || (emit_config::active().w8a8
                && emit_config::active().qwen_w8a8_prefill
                && buckets
                    .iter()
                    .all(|rows| matches!(rows, 128 | 1024 | 4096 | 8192))),
        "Qwen FP8 prefill requires explicit W8A8 prefill and buckets 128/1024/4096/8192"
    );
    let mut tensors = None;
    let mut progs = Vec::new();
    let mut gen: Vec<GenTensor> = Vec::new();
    let mut last = None;
    let phases = buckets
        .iter()
        .map(|&rows| (rows, true))
        .chain(rungs.iter().map(|&rows| (rows, false)));
    for (rows, prefill) in phases {
        if prefill {
            assert!(rows >= 128 && rows % 128 == 0 && rows <= ctx);
        }
        let mut m = phase(
            c,
            ctx,
            n_cu,
            target,
            fp8,
            rows,
            decode_slots,
            prefill,
            tensors,
        );
        tensors = Some(std::mem::take(&mut m.tensors));
        progs.push(m.progs.remove(0));
        // Adopting tensor declarations does not carry their generation recipes.
        for recipe in std::mem::take(&mut m.gen) {
            if !gen.iter().any(|g| g.tensor == recipe.tensor) {
                gen.push(recipe);
            }
        }
        last = Some(m);
    }
    let mut m = last.unwrap();
    m.tensors = tensors.unwrap();
    m.progs = progs;
    m.gen = gen;
    m.prog_t = buckets.iter().chain(rungs).copied().collect();
    m
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
    // Rungs below PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS stay native (fusions on, one segment).
    let decode_cublaslt = !prefill
        && emit_config::active().decode_cublaslt
        && batch >= emit_config::active().decode_cublaslt_min_rows.unwrap_or(1);
    let fuse_ab = !prefill && !decode_cublaslt && emit_config::active().qwen_fuse_ab;
    let fuse_mlp = !prefill && !decode_cublaslt && emit_config::active().qwen_fuse_mlp;
    assert!(
        !fuse_mlp || (!fp8 && !decode_cublaslt),
        "Qwen MLP projection fusion requires native BF16 decode"
    );
    let projection_dag = !prefill && !decode_cublaslt && emit_config::active().qwen_projection_dag;
    assert!(
        !projection_dag || (!fp8 && !decode_cublaslt),
        "Qwen projection DAG requires native BF16 decode"
    );
    assert!(
        !fuse_ab || (!fp8 && !decode_cublaslt),
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
                && ((!prefill && batch <= W8A8_DECODE_ROWS)
                    || (prefill
                        && matches!(batch, 128 | 1024 | 4096 | 8192)
                        && emit_config::active().qwen_w8a8_prefill))),
        "Qwen W8A8 requires decode rungs within one GEMM M tile or explicit supported prefill buckets"
    );
    assert!(
        !decode_cublaslt || !fp8,
        "Qwen decode Lt requires BF16 projections"
    );
    let fuse_decode = !prefill
        && emit_config::active().qwen_fold_norm
        && !fp8
        && !decode_cublaslt
        && c.block.is_none();
    let fold_norm = fuse_decode && batch <= emit_config::active().qwen_fold_max.unwrap_or(4);
    let mut e = Emitter {
        b,
        c,
        fp8,
        w8a8,
        prefill,
        batch,
        decode_slots,
        decode_cublaslt,
        fuse_ab,
        ab_blocks,
        projection_dag,
        share_quant,
        quant_input: None,
        fuse: fuse_decode,
        fold: fold_norm,
        pending_residual: None,
        ctx,
        active,
        pos,
        kvlen,
        cos,
        sin,
    };
    let x = e.act("x", c.hidden);
    // The folded residual stream alternates between x and x2 (see `Fold`); `pending` is the
    // block output not yet added into it.
    let x2 = if fold_norm {
        e.act("x2", c.hidden)
    } else {
        TENSOR_NONE
    };
    let (mut cur, mut nxt) = (x, x2);
    let mut pending = TENSOR_NONE;
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
        let fold_in = if fold_norm {
            let gamma = e.weight(&format!("{p}.input_layernorm.weight"), c.hidden as u64);
            let out = if pending == TENSOR_NONE {
                TENSOR_NONE
            } else {
                nxt
            };
            Some(Fold {
                a: cur,
                b: pending,
                out,
                gamma,
            })
        } else {
            dep = e.norm(hn, x, &format!("{p}.input_layernorm.weight"), dep);
            None
        };
        dep = if full {
            e.full(layer, hn, mixed, dep, fold_in.as_ref())
        } else {
            e.linear(layer, hn, mixed, dep, fold_in.as_ref())
        };
        if fold_norm {
            if pending != TENSOR_NONE {
                std::mem::swap(&mut cur, &mut nxt);
            }
            pending = mixed;
        } else {
            dep = e.residual(x, mixed, dep);
            dep = e.norm(hn, x, &format!("{p}.post_attention_layernorm.weight"), dep);
        }
        dep = if fuse_decode {
            let wg = e.weight(
                &format!("{p}.mlp.gate_proj.weight"),
                c.inter as u64 * c.hidden as u64,
            );
            let wu = e.weight(
                &format!("{p}.mlp.up_proj.weight"),
                c.inter as u64 * c.hidden as u64,
            );
            let f = fold_norm.then(|| {
                let gamma = e.weight(
                    &format!("{p}.post_attention_layernorm.weight"),
                    c.hidden as u64,
                );
                let f = Fold {
                    a: cur,
                    b: pending,
                    out: nxt,
                    gamma,
                };
                std::mem::swap(&mut cur, &mut nxt);
                f
            });
            // gate, up and SiLU(gate) * up as one walk (GEMV_GLU, act 1 = SiLU).
            e.b.emit(DevOp::GemvGlu, e.b.all(), &[dep], |d| {
                d.t[..6].copy_from_slice(&[fu, hn, wg, TENSOR_NONE, TENSOR_NONE, wu]);
                d.i[..3].copy_from_slice(&[batch, c.inter, c.hidden]);
                d.i[5] = 1;
                if let Some(f) = &f {
                    f.fields(d, 4, c.eps);
                }
            })
        } else if fuse_mlp {
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
        if !fuse_decode {
            dep = e.b.emit(DevOp::Glu, e.b.all(), &[dep], |d| {
                d.t[..3].copy_from_slice(&[fu, gate, up]);
                d.i[..2].copy_from_slice(&[batch * c.inter, 1]);
            });
        }
        dep = e.proj(
            mixed,
            fu,
            &format!("{p}.mlp.down_proj.weight"),
            c.hidden,
            c.inter,
            dep,
        );
        if fold_norm {
            pending = mixed;
        } else {
            dep = e.residual(x, mixed, dep);
        }
    }
    if c.block.is_some() {
        dep = e.flush_residual(dep);
    }
    if c.block.is_none() {
        let head = if c.tied {
            format!("{}embed_tokens.weight", c.prefix)
        } else {
            "lm_head.weight".into()
        };
        if fold_norm {
            let gamma = e.weight(&format!("{}norm.weight", c.prefix), c.hidden as u64);
            let f = Fold {
                a: cur,
                b: pending,
                out: TENSOR_NONE,
                gamma,
            };
            dep = e.proj_fold(logits, hn, &head, c.vocab, c.hidden, &f, dep);
        } else {
            dep = e.norm(hn, x, &format!("{}norm.weight", c.prefix), dep);
            if prefill {
                let w = e.weight(&head, c.vocab as u64 * c.hidden as u64);
                // A packed packet's compact terminal re-runs the head over the gathered last
                // rows (M = requests), which only the tiled arm takes; the GEMV arm is M = 1.
                let packed = emit_config::active().emit_packed_prefill == Some(true);
                let op = if packed {
                    pick_tile(
                        1,
                        c.vocab,
                        c.hidden,
                        e.b.n_cu(),
                        kernelcaps::QuantScheme::None,
                    )
                } else {
                    DevOp::Gemv
                };
                let maps = (packed && emit_config::active().tma_gemm).then(|| {
                    let mut map = |target, rows| {
                        let g = GenTensor::tmap_bf16(target, rows, c.hidden, 128);
                        e.b.tensor_gen(
                            &format!("tmap.{target}.{rows}x{}", c.hidden),
                            g.byte_len(),
                            g,
                        )
                    };
                    (map(hn, batch), map(w, c.vocab))
                });
                dep = e.b.emit(op, e.b.all(), &[dep], |d| {
                    d.t[..3].copy_from_slice(&[logits, hn, w]);
                    d.i[..3].copy_from_slice(&[1, c.vocab, c.hidden]);
                    d.i[4] = batch - 1;
                    if let Some((a, w)) = maps {
                        d.i[6] = a;
                        d.i[7] = w;
                    }
                });
            } else {
                dep = e.proj(logits, hn, &head, c.vocab, c.hidden, dep);
            }
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

/// Set once `run` below emits for an AMD target: the AMD tile inventory, an AMD manifest, and
/// an AMD interpreter object in place of the paired CUDA one. This is the DECODE gate.
/// Decode rungs the W8A8 decode arm serves: rung 1 on the M1 kernels, wider rungs as one M tile
/// of the Hopper W8A8 GEMM (`plow_w8a8_vllm_decode_rows` on the object).
pub(crate) const W8A8_DECODE_ROWS: u32 = 128;
const AMD_QWEN_EMIT: bool = false;

/// Set once a HIP chunked Gated DeltaNet prefill kernel and its `plowrt` adapter exist.
/// PREFILL only — a decode-only blob emits `QwenGdnStep` and never reaches this.
const AMD_GDN_PREFILL: bool = false;

/// The Qwen3.5-specific opcodes. Derived from `DevOp::ALL` by `PLOW_DOP_QWEN_*` spelling rather
/// than listed, so a new Qwen op joins the coverage question the moment it is defined.
fn qwen_opcodes() -> Vec<DevOp> {
    DevOp::ALL
        .iter()
        .copied()
        .filter(|o| o.c_name().starts_with("PLOW_DOP_QWEN_"))
        .collect()
}

/// The Qwen opcodes `runtime/amd/interp.hip` has no arm for.
///
/// Reads the same `GFX950_DISPATCHED` list `check_gfx950_opcode_coverage` reads, which a drift
/// test keeps equal to the `case` labels actually present in `interp.hip`. So this shrinks by
/// itself as arms land: implement the arm, add its spelling to the list, and the refusal below
/// stops naming it.
pub(crate) fn qwen_amd_missing_arms() -> Vec<&'static str> {
    qwen_opcodes()
        .into_iter()
        .map(|o| o.c_name())
        .filter(|c| !GFX950_DISPATCHED.contains(c))
        .collect()
}

/// Refuse a Qwen3.5 emit for a target that cannot run the packet it would produce.
///
/// WHY A REFUSAL AND NOT A BEST EFFORT. AMD's dispatch `default:` is `/* PLOW_DOP_NOP */` — it
/// writes NOTHING and does not trap — so an AMD packet carrying an op with no arm would RUN to
/// completion over untouched output buffers and serve fluent nonsense. That is what
/// `check_gfx950_opcode_coverage` catches generically; this is the same question asked early,
/// before the emitter has done any work, with each gap named so its size is legible instead of
/// being an arch string mismatch.
///
/// THE THREE GAPS ARE NOT THE SAME SHAPE, and the order matters to whoever picks this up:
///
///   1. The interpreter arms. CLOSED — `runtime/amd/op_qwen_gdn.h` implements all ten, and
///      `qwen_amd_missing_arms` reads the live `GFX950_DISPATCHED` so this reopens by itself if
///      one is ever dropped.
///   2. The EMITTER. `run` below is NVIDIA-shaped throughout: it pins the NVIDIA tile inventory
///      (`EmitAmdGuard::set(false)`), builds the manifest for `sm_90a`, and asserts an external
///      paired CUDA interpreter. Porting that is what a decode-only AMD blob actually needs, and
///      it is the SHORTEST path to a working AMD Qwen3.5 — see (3).
///   3. `PLOW_DOP_QWEN_GDN_PREFILL`. Only a PREFILL program emits it (the `self.prefill` branch in
///      `Emitter::gdn`); a decode-only model — which is what `model()` builds whenever
///      `PLOW_QWEN_PREFILL` is unset, and how Kimi-K3 spent its whole AMD bring-up — carries
///      `QwenGdnStep` instead and needs nothing from this gap. So (3) does NOT block (2). It is
///      also not an interpreter arm on EITHER backend: sm_90a dispatches it host-side into a
///      generated CuTe kernel (`runtime/nvidia/gdn_prefill.cpp`, `plow_gdn_run`) from
///      `crates/plowrt/src/exec/gpu.rs`, so AMD needs a chunked HIP body AND an adapter.
fn refuse_unimplemented_target(arch: &str, gpu: &str) {
    if !target_is_amd(arch, gpu) {
        return;
    }
    let mut gaps: Vec<String> = Vec::new();
    let missing = qwen_amd_missing_arms();
    if !missing.is_empty() {
        gaps.push(format!(
            "`qwen35_gdn_amd`: runtime/amd/interp.hip has no arm for {missing:?}. Implement them \
             and add their `PLOW_DOP_*` spellings to `GFX950_DISPATCHED`"
        ));
    }
    if !AMD_QWEN_EMIT {
        gaps.push(
            "`qwen35_amd_emit`: this emitter is NVIDIA-shaped — it pins the NVIDIA tile inventory \
             (`EmitAmdGuard::set(false)`), builds the manifest for sm_90a, and asserts an external \
             paired CUDA interpreter. The ten Gated DeltaNet arms exist and are oracle-checked \
             (docs/amd/qwen35-gdn-mi300x.md); nothing routes a packet to them yet. A DECODE-ONLY \
             blob needs only this gap closed"
                .to_string(),
        );
    }
    if !AMD_GDN_PREFILL {
        gaps.push(
            "`qwen35_gdn_prefill_amd`: PREFILL additionally needs \
             `PLOW_DOP_QWEN_GDN_PREFILL`, a HOST-side chunked Gated DeltaNet dispatch rather than \
             an interpreter arm — sm_90a calls the generated CuTe kernel in \
             runtime/nvidia/gdn_prefill.cpp via `plow_gdn_run`, and there is no HIP equivalent or \
             AMD adapter. Decode does not emit this op and is not blocked on it"
                .to_string(),
        );
    }
    assert!(
        gaps.is_empty(),
        "qwen3_5 cannot be emitted for {arch}. Missing capability:\n  - {}",
        gaps.join("\n  - ")
    );
}

/// The RECURRENT STATE section (`plow_asset::recurrent_state`): every `state.qwen.*` buffer,
/// the active mask and the packed-prefill initstate scratch, slot-major over the decode slots.
fn recurrent_state_section(m: &Model, slots: u32) -> packet::devbuild::SectionData {
    use plow_asset::recurrent_state::{Buffer, Manifest, TensorFacts, SECTION, VERSION};
    let buffer = |index: usize| {
        let t = &m.tensors[index];
        assert!(
            t.bytes % u64::from(slots) == 0,
            "{} is not slot-major",
            t.name
        );
        Buffer {
            tensor: u16::try_from(index).expect("tensor handle"),
            slot_bytes: u32::try_from(t.bytes / u64::from(slots)).expect("slot bytes"),
        }
    };
    let position = |name: &str| m.tensors.iter().position(|t| t.name == name);
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    let manifest = Manifest {
        version: VERSION,
        slots,
        buffers: (0..m.tensors.len())
            .filter(|&i| m.tensors[i].name.starts_with("state.qwen."))
            .map(buffer)
            .collect(),
        active: u16::try_from(position("in.active").expect("in.active")).expect("tensor handle"),
        prefill_initstate: position("act.gdn.initstate").map(buffer),
        prefill_updates_state: m.progs[..dec_lo]
            .iter()
            .all(|p| p.insts.iter().any(|i| i.op == DevOp::QwenGdnPrefill as u16)),
    };
    manifest
        .validate(slots, |h| {
            m.tensors.get(h as usize).map(|t| TensorFacts {
                bytes: t.bytes,
                initialized: t.init.is_some(),
            })
        })
        .unwrap_or_else(|error| panic!("recurrent state: {error}"));
    packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&manifest).expect("recurrent state"),
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
    refuse_unimplemented_target(arch, gpu);
    assert_eq!(
        arch, "sm_90a",
        "qwen3_5 native CUDA path currently requires sm_90a"
    );
    assert_eq!(tp, 1, "qwen3_5 tensor parallelism is not implemented");
    let fp8 = emit_config::active().fp8;
    // `[decode_batch]`, or the PLOW_DECODE_BATCH_LADDER rungs (one decode program each).
    let rungs = emit_config::active().decode_rungs();
    let batch = *rungs.last().expect("decode rungs");
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
        None | Some("0") => ladder(&c, ctx, n_cu, target, fp8, &[], &rungs),
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
            let buckets = crate::with_always_rung(buckets, ctx);
            ladder(&c, ctx, n_cu, target, fp8, &buckets, &rungs)
        }
    };
    let shipped = crate::checkpoint::shipped_names(dir);
    validate_coverage(
        dir,
        &c.prefix,
        &m.tensors
            .iter()
            .filter_map(|t| {
                // Sidecars are derived from these exact checkpoint matrices; scales add no source.
                // A re-keyed checkpoint ships the twin itself in place of its bf16 source.
                match t.name.strip_prefix("fp8/") {
                    Some(name) if name.ends_with("_scale") => None,
                    Some(_) if shipped.contains(&t.name) => Some(t.name.clone()),
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
    let mut sections = Vec::new();
    if let Some(section) = segment_roles(&m) {
        sections.push(section);
    }
    // PLOW_EMIT_PREFILL_CUBLASLT: the dense projections of every prefill bucket as isolated
    // cuBLASLt segments (the Gemma route; shapes in segment_roles::CUBLASLT_PREFILL_QWEN3_8_27B_SHAPES).
    // The ordinary Qwen prefill otherwise runs every GEMM on the generic interpreter object.
    if emit_config::active().prefill_cublaslt {
        let selected = crate::dense_cublaslt::apply_prefill(&mut m, &mut sections, arch)
            .expect("Qwen prefill projection segments");
        assert!(
            selected > 0,
            "no eligible Qwen prefill projections for cuBLASLt"
        );
        eprintln!("  cuBLASLt prefill: {selected} projection segments");
    }
    // PLOW_EMIT_GEN_KERNELS: the full-attention prefill flash as a generated role object
    // (attn_pf_hd256_causal) in every bucket from the entry's smallest rung.
    if let Some(list) = emit_config::active().gen_kernels.as_deref() {
        crate::gen_kernels::apply_output_objects(
            &mut m,
            &mut sections,
            arch,
            Path::new(out),
            gpu,
            false,
            list,
            false,
        )
        .unwrap_or_else(|error| panic!("generated kernels: {error}"));
    }
    let lean = apply_verify_gate(&m, verify);
    let man = manifest::build(&m, arch, &lean);
    crate::report_dispatch_audit(&man);
    let out = Path::new(out);
    crate::decode_objects::append(
        &m,
        &mut sections,
        crate::emit_config::active().decode_objects.as_deref(),
        arch,
        out,
    )
    .unwrap_or_else(|error| panic!("decode objects: {error}"));
    // PLOW_EMIT_PACKED_PREFILL=1: the packed-request contract (plow_asset::packed_prefill) for the
    // hybrid chain. Emitted after every program rewrite above: the manifest digests the final
    // prefill programs. The KV writer takes the per-row slot table (QwenHeadNormRope t7), the conv
    // prefill the request table (QwenGdnConvPrefill t5); the host GDN scan runs the same table.
    if emit_config::active().emit_packed_prefill == Some(true) {
        let max_rows = m.prog_t[..m.prog_t.len() - 1]
            .iter()
            .copied()
            .max()
            .expect("prefill buckets");
        // The prefill attention's GEN_TMAP_KV_PAIR maps get a per-slot pointer table each: a
        // packed launch binds the table at t[7] (packed_prefill::bind_request), the role indexes
        // it by the request's slot.
        let originals: Vec<u16> = m
            .gen
            .iter()
            .filter(|g| g.kind == packet::rope::GEN_TMAP_KV_PAIR)
            .map(|g| g.tensor as u16)
            .collect();
        let mut declare = |name: String, bytes: u64| {
            assert!(m.tensors.len() < packet::dev::TENSOR_NONE16 as usize);
            let h = m.tensors.len() as u16;
            m.tensors.push(packet::devbuild::TensorDecl {
                name,
                bytes,
                init: None,
            });
            h
        };
        let slot = declare("pf.request.slot".into(), u64::from(max_rows) * 4);
        let request = declare("pf.request.table".into(), (1 + 4 * u64::from(batch)) * 4);
        let maps: Vec<_> = originals
            .into_iter()
            .map(|original| plow_asset::packed_prefill::Map {
                original,
                slots: declare(format!("pf.request.maps.{original}"), 8 * u64::from(batch)),
            })
            .collect();
        let manifest = plow_asset::program::with_model(&m, |p| -> Result<_, String> {
            let live = plow_asset::live_kv::emit(p)?;
            let request = plow_asset::packed_prefill::Manifest {
                version: live.version,
                // Masked padding: pad rows carry slot -1 (the rope, conv and scan skip them)
                // instead of charging one request's physical context.
                max_request_rows: Some(emit_config::active().max_request_chunk.unwrap_or(max_rows)),
                stage_rows: None,
                stages: Vec::new(),
                slot,
                request,
                maps: maps.clone(),
                programs: p.programs[..p.prefill_count]
                    .iter()
                    .map(plow_asset::live_kv::program_digest)
                    .collect(),
            };
            request.validate(p, &live)?;
            Ok(request)
        });
        let manifest = manifest.unwrap_or_else(|error| panic!("packed request contract: {error}"));
        sections.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: plow_asset::packed_prefill::SECTION.into(),
            data: serde_json::to_vec(&manifest).unwrap(),
        });
        // plowrt binds a packed packet through its compiled LIVE KV geometry.
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
    sections.push(recurrent_state_section(&m, batch));
    if c.block.is_none() {
        sections.push(crate::serve_section::section_for(dir));
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
    let mut man = man;
    man["runtime_requires"] = crate::dense_cublaslt::runtime_requires(&m, &sections);
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

    /// The AMD gate is a REFUSAL, not a filter: every Qwen opcode is still missing an arm, so an
    /// AMD target must not reach the emitter at all. Reads the live arm list, so the day an arm
    /// lands this test's expectation moves with it rather than pinning a stale gap.
    #[test]
    fn amd_targets_are_refused_by_name() {
        let missing = qwen_amd_missing_arms();
        let all: Vec<&str> = qwen_opcodes().into_iter().map(|o| o.c_name()).collect();
        assert!(!all.is_empty(), "no PLOW_DOP_QWEN_* opcodes found at all");
        assert!(
            missing.iter().all(|c| all.contains(c)),
            "missing-arm list must be a subset of the Qwen opcodes"
        );
        for (arch, gpu) in [("gfx942", ""), ("gfx950", ""), ("", "MI300X")] {
            let err = std::panic::catch_unwind(|| refuse_unimplemented_target(arch, gpu))
                .expect_err("an AMD target must be refused while arms are missing");
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()).unwrap());
            assert!(
                msg.contains("Missing capability:"),
                "the refusal must name the capability: {msg}"
            );
            for c in &missing {
                assert!(msg.contains(c), "the refusal must name {c}: {msg}");
            }
            assert!(
                AMD_QWEN_EMIT || msg.contains("qwen35_amd_emit"),
                "the emitter gap must be named while it stands: {msg}"
            );
            assert!(
                AMD_GDN_PREFILL || msg.contains("qwen35_gdn_prefill_amd"),
                "the host-side prefill gap must be named while it stands: {msg}"
            );
        }
        // sm_90a is the path that works; it must pass this gate untouched.
        refuse_unimplemented_target("sm_90a", "H100");
    }

    /// The refusal claims a DECODE-only blob needs nothing from `qwen35_gdn_prefill_amd`, and that
    /// claim is what tells the next person to port the emitter rather than the CuTe kernel. Pin it
    /// against the emitter, not against a reading of it.
    #[test]
    fn a_decode_only_model_emits_no_prefill_op() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let c = Config::parse(&fixture());
        let m = model(&c, 32768, 132, 0, false, 1);
        let decode_ops: std::collections::BTreeSet<u16> = m
            .progs
            .iter()
            .flat_map(|p| p.insts.iter())
            .map(|d| d.op)
            .collect();
        assert!(
            decode_ops.contains(&(DevOp::QwenGdnStep as u16)),
            "a decode model must carry the step op"
        );
        assert!(
            !decode_ops.contains(&(DevOp::QwenGdnPrefill as u16)),
            "a decode-only model must not carry the host-dispatched prefill op"
        );
        // ... and every op it does carry must already have an AMD arm, or the claim is false.
        let unarmed: Vec<&str> = decode_ops
            .iter()
            .filter_map(|&op| DevOp::ALL.iter().copied().find(|o| *o as u16 == op))
            .map(|o| o.c_name())
            .filter(|c| !GFX950_DISPATCHED.contains(c))
            .collect();
        assert!(
            unarmed.is_empty(),
            "a decode-only Qwen packet still carries {unarmed:?} with no AMD arm — the \
             `qwen35_amd_emit` refusal claims this set is empty"
        );
    }

    #[test]
    fn decode_balanced_splits_reject_gf6_and_preserve_prefill() {
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
        cfg.attention_decode_balance_gf = None;
        cfg.ns_full_abs = None;
        emit_config::install(cfg.clone());
        let mut c = Config::parse(&fixture());
        c.heads = 24;
        c.kv_heads = 4;
        c.hd = 256;
        let plain = model_prefill(&c, 32768, 132, 0, &[128], 1, false);
        cfg.attention_decode_balance_gf = Some(6);
        emit_config::install(cfg.clone());
        assert!(std::panic::catch_unwind(|| model(&c, 32768, 132, 0, false, 1)).is_err());
        for batch in [1, 4] {
            for (gf, expected) in [(None, 11), (Some(2), 11)] {
                cfg.attention_decode_balance_gf = gf;
                emit_config::install(cfg.clone());
                let m = model(&c, 32768, 132, 0, false, batch);
                for d in m.progs[0]
                    .insts
                    .iter()
                    .filter(|d| d.op == DevOp::FlashDecode as u16)
                {
                    assert_eq!(d.i[5], expected);
                    assert_eq!(
                        m.tensors[d.t[0] as usize].bytes,
                        u64::from(batch * 24 * 256 * expected * 4)
                    );
                    assert_eq!(
                        m.tensors[d.t[1] as usize].bytes,
                        u64::from(batch * 24 * expected * 8)
                    );
                    let merge = m.progs[0]
                        .insts
                        .iter()
                        .find(|x| x.op == DevOp::FlashMerge as u16 && x.t[1] == d.t[0])
                        .unwrap();
                    assert_eq!(&merge.i[..4], &[batch, 24, expected, 256]);
                }
            }
        }
        cfg.attention_decode_balance_gf = Some(2);
        cfg.ns_full_abs = None;
        emit_config::install(cfg);
        let balanced = model_prefill(&c, 32768, 132, 0, &[128], 1, false);
        assert_eq!(plain.progs[0].insts, balanced.progs[0].insts);
        assert_eq!(plain.progs[0].waits, balanced.progs[0].waits);
        assert_eq!(plain.progs[0].succs, balanced.progs[0].succs);
    }

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
            // The folded residual + norm prologue writes the activation (t1) and the residual
            // stream out (t3); its inputs ride i-slot handles (see `Fold`).
            if (d.op == DevOp::Gemv as u16 || d.op == DevOp::GemvGlu as u16) && d.i[3] == 4 {
                output_slots.extend([1, 3]);
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
    // The runtime's live-KV (VMM) allocator derives its geometry from this manifest, so the
    // packet's full-attention caches and their writers must pass the live_kv contract.
    #[test]
    fn live_kv_manifest_covers_the_full_attention_caches() {
        let _env = crate::test_env::env_guard();
        let c = Config::parse(&fixture());
        let m = model_prefill(&c, 8192, 132, 0, &[128], 4, false);
        let live = plow_asset::program::with_model(&m, plow_asset::live_kv::emit)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((live.batch, live.max_ctx), (4, 8192));
        assert_eq!(live.caches.len(), 1);
        let cache = &live.caches[0];
        assert_eq!(
            (
                cache.heads,
                cache.hd,
                cache.stride,
                cache.window,
                cache.mask
            ),
            (4, 256, 8192, 0, u32::MAX)
        );
        assert!(cache.scales.is_none());
        assert_eq!(live.programs.len(), m.progs.len());
    }
    // PLOW_QWEN_KV_INTERLEAVE: the cache writers, the KV maps and the live_kv manifest declare the
    // row-interleaved layout, and only the `_ilv` catalog twin binds the prefill flash.
    #[test]
    fn kv_interleave_declares_the_layout_and_binds_the_twin() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        let c = Config::parse(&fixture());
        let mut cfg = original.clone();
        cfg.gen_kernels = Some("attn_pf_hd256_causal_ilv".into());
        cfg.qwen_kv_interleave = true;
        emit_config::install(cfg);
        let m = model_prefill(&c, 16384, 132, 0, &[128, 1024], 2, false);
        let twin = |name: &str| {
            crate::gen_kernels::CATALOG
                .iter()
                .find(|e| e.name == name)
                .unwrap()
        };
        for program in &m.progs[..2] {
            for op in program
                .insts
                .iter()
                .filter(|op| op.op == DevOp::FlashPrefill as u16)
            {
                assert!(twin("attn_pf_hd256_causal_ilv").matches(op), "{op:?}");
                assert!(!twin("attn_pf_hd256_causal").matches(op), "{op:?}");
                let map = m.gen.iter().find(|g| g.tensor == op.t[7]).unwrap();
                assert_eq!(map.low, plow_asset::live_kv::LAYOUT_ROW_INTERLEAVED as f64);
            }
        }
        emit_config::install(original);
        for program in &m.progs {
            for op in program
                .insts
                .iter()
                .filter(|op| op.op == DevOp::QwenHeadNormRope as u16)
            {
                let cache = op.i[4] != 0;
                assert_eq!(op.i[7], cache as u32, "{op:?}");
            }
        }
        let live = plow_asset::program::with_model(&m, plow_asset::live_kv::emit)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(live
            .caches
            .iter()
            .all(|c| c.layout == plow_asset::live_kv::LAYOUT_ROW_INTERLEAVED));
    }
    // PLOW_EMIT_GEN_KERNELS: the prefill flash takes the generated role's contract (one split,
    // fused bf16 output, no merge) and attn_pf_hd256_causal binds every bucket; decode unchanged.
    #[test]
    fn gen_kernels_fuse_the_prefill_attention_and_bind_the_causal_role() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        let c = Config::parse(&fixture());
        let plain = model_prefill(&c, 16384, 132, 0, &[128, 1024, 4096], 2, false);
        let mut cfg = original.clone();
        cfg.gen_kernels = Some("attn_pf_hd256_causal".into());
        emit_config::install(cfg);
        let mut m = model_prefill(&c, 16384, 132, 0, &[128, 1024, 4096], 2, false);
        emit_config::install(original);
        let entry = crate::gen_kernels::CATALOG
            .iter()
            .find(|e| e.name == "attn_pf_hd256_causal")
            .unwrap();
        let full_layers = c.layers.iter().filter(|&&full| full).count();
        for program in &m.progs[..3] {
            let flash: Vec<_> = program
                .insts
                .iter()
                .filter(|op| op.op == DevOp::FlashPrefill as u16)
                .collect();
            assert_eq!(flash.len(), full_layers);
            for op in flash {
                assert!(entry.matches(op), "{op:?}");
                assert_ne!(op.t[5], TENSOR_NONE);
                assert!(op.t[..5].iter().all(|&t| t != op.t[5]));
                assert_eq!(op.i[7], 1);
                // The role's TMA K/V staging: t[7] names the layer's GEN_TMAP_KV_PAIR.
                let map = m
                    .gen
                    .iter()
                    .find(|g| g.tensor == op.t[7])
                    .unwrap_or_else(|| panic!("no KV map for {op:?}"));
                assert_eq!(map.kind, packet::rope::GEN_TMAP_KV_PAIR);
                assert_eq!(
                    (map.aux, map.scale, map.hd, map.ctx),
                    (op.t[3], op.t[4], 256, 16384)
                );
            }
            assert!(!program
                .insts
                .iter()
                .any(|op| op.op == DevOp::FlashMerge as u16));
            check_operand_dependencies(program);
        }
        // Decode unchanged, modulo the tensor handles the prefill map declarations shift.
        assert_eq!(m.progs[3].insts.len(), plain.progs[3].insts.len());
        let mut renumber = std::collections::BTreeMap::new();
        for (gen, base) in m.progs[3].insts.iter().zip(&plain.progs[3].insts) {
            let mut same = *gen;
            same.t = base.t;
            assert_eq!(same, *base);
            for (&g, &b) in gen.t.iter().zip(&base.t) {
                assert_eq!(*renumber.entry(b).or_insert(g), g, "{gen:?} vs {base:?}");
            }
        }
        let directory = std::env::temp_dir().join(format!("plow-qwen-gen-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let globals = [
            ("plow_gen_flash_prefill_abi", 1),
            ("plow_gen_block", 384),
            ("plow_gen_arena_bytes", 147_456),
            ("plow_attention_head_dim", 256),
            ("plow_attention_query_tile", 128),
            ("plow_attention_kv_tile", 64),
            ("plow_attention_warps", 12),
            ("plow_pf_request_abi", 2),
        ];
        let image =
            plow_asset::cubin::synthetic_elf(crate::gen_kernels::ENTRY_SYMBOL, &globals, 90);
        std::fs::write(directory.join(&entry.file), image).unwrap();
        let mut sections = Vec::new();
        crate::gen_kernels::apply_output_objects(
            &mut m,
            &mut sections,
            "sm_90a",
            &directory.join("model.pkt"),
            "h100",
            false,
            "attn_pf_hd256_causal",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let roles = plow_asset::segment_roles::SegmentRoles::from_bytes(&sections[0].data).unwrap();
        let role = entry.role;
        assert!(roles.objects[&role]
            .abi
            .starts_with("gen_flash_prefill_v1:attn_pf_hd256_causal:"));
        for program in &roles.programs[..3] {
            assert_eq!(
                program.roles.iter().filter(|&&r| r == role).count(),
                full_layers
            );
        }
        std::fs::remove_dir_all(&directory).ok();
    }
    // A decode entry in PLOW_EMIT_GEN_KERNELS: the decode programs keep their instructions, but
    // every FlashDecode + FlashMerge pair sits in two single-instruction segments, and the
    // packet records the generated decode role on both of them for every decode program.
    #[test]
    fn gen_kernels_isolate_the_decode_attention_pair_and_bind_the_decode_role() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        // Two full-attention layers: the pairs reuse the partial tensors, so one layer cannot
        // tell a per-layer merge search from a whole-program one.
        let mut v = fixture();
        v["text_config"]["num_hidden_layers"] = serde_json::json!(8);
        v["text_config"]["layer_types"] = serde_json::json!((0..8)
            .map(|l| if l % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>());
        let c = Config::parse(&v);
        let plain = model(&c, 32768, 132, 0, false, 4);
        let mut cfg = original.clone();
        cfg.gen_kernels = Some("attn_dec_hd256_gqa6".into());
        emit_config::install(cfg);
        let mut m = model(&c, 32768, 132, 0, false, 4);
        emit_config::install(original);
        let entry = crate::gen_kernels::CATALOG
            .iter()
            .find(|e| e.name == "attn_dec_hd256_gqa6")
            .unwrap();
        let lo = packet::devbuild::decode_rung_lo(&m.prog_t);
        let full_layers = c.layers.iter().filter(|&&full| full).count();
        assert!(lo < m.progs.len());
        let mut expected_pairs = Vec::new();
        for (index, program) in m.progs.iter().enumerate() {
            assert_eq!(program.insts, plain.progs[index].insts);
            if index < lo {
                assert_eq!(program.gq_seg_ofs, plain.progs[index].gq_seg_ofs);
                continue;
            }
            let sites: Vec<_> = program
                .insts
                .iter()
                .enumerate()
                .filter(|(_, op)| entry.matches(op))
                .collect();
            assert_eq!(sites.len(), full_layers);
            let mut pair_segments = Vec::new();
            for (ix, op) in sites {
                let merge = program
                    .insts
                    .iter()
                    .enumerate()
                    .skip(ix + 1)
                    .find(|(_, m)| m.op == DevOp::FlashMerge as u16 && m.t[1] == op.t[0])
                    .map(|(m, _)| m)
                    .unwrap();
                let seg = crate::gen_kernels::lone_segment(program, ix).unwrap();
                let merge_seg = crate::gen_kernels::lone_segment(program, merge).unwrap();
                assert!(seg < merge_seg);
                pair_segments.push((seg, merge_seg));
            }
            assert!(program.gq_seg_ofs.len() > 2 * full_layers);
            expected_pairs.push((index, pair_segments));
        }
        let directory =
            std::env::temp_dir().join(format!("plow-qwen-gen-dec-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let globals = [
            ("plow_gen_flash_decode_abi", 1),
            ("plow_gen_block", 128),
            ("plow_gen_arena_bytes", 75_776),
            ("plow_attention_head_dim", 256),
            ("plow_attention_query_tile", 6),
            ("plow_attention_kv_tile", 32),
            ("plow_attention_warps", 4),
            ("plow_gen_decode_gqa", 6),
            ("plow_gen_decode_nsplit_max", 8),
        ];
        let image =
            plow_asset::cubin::synthetic_elf(crate::gen_kernels::DECODE_ENTRY_SYMBOL, &globals, 90);
        std::fs::write(directory.join(&entry.file), image).unwrap();
        let mut sections = Vec::new();
        crate::gen_kernels::apply_output_objects(
            &mut m,
            &mut sections,
            "sm_90a",
            &directory.join("model.pkt"),
            "h100",
            false,
            "attn_dec_hd256_gqa6",
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let roles = plow_asset::segment_roles::SegmentRoles::from_bytes(&sections[0].data).unwrap();
        assert!(roles.objects[&entry.role]
            .abi
            .starts_with("gen_flash_decode_v1:attn_dec_hd256_gqa6:"));
        assert!(roles.programs.iter().all(|p| p.index >= lo));
        for (index, pairs) in expected_pairs {
            let p = roles.programs.iter().find(|p| p.index == index).unwrap();
            assert_eq!(p.roles.len(), m.progs[index].gq_seg_ofs.len() - 1);
            // Every layer's pair carries the role (a whole-program merge search found layer 0's
            // merge for every layer: one merge role, the runtime rejecting the rest).
            let bound: Vec<usize> = p
                .roles
                .iter()
                .enumerate()
                .filter(|(_, &r)| r == entry.role)
                .map(|(seg, _)| seg)
                .collect();
            let mut expected: Vec<usize> = pairs.iter().flat_map(|&(d, m)| [d, m]).collect();
            expected.sort_unstable();
            assert_eq!(bound, expected);
            assert_eq!(bound.len(), 2 * full_layers);
        }
        std::fs::remove_dir_all(&directory).ok();
    }
    // PLOW_QWEN_FOLD_NORM: every decode rung fuses the MLP into one GEMV_GLU and k/v and z/a/b into
    // one GemvQkv each; rungs up to PLOW_QWEN_FOLD_MAX (4) also carry no Residual / QwenRmsNorm
    // instruction (the folds ride the projections) on a ping-pong residual stream; wider rungs
    // keep the separate instructions and prefill is untouched.
    #[test]
    fn fold_norm_folds_the_rungs_up_to_the_cap_and_fuses_every_decode_rung() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let original = emit_config::active().clone();
        let c = Config::parse(&fixture());
        let mut cfg = original.clone();
        cfg.qwen_fold_norm = true;
        cfg.qwen_fuse_ab = true;
        emit_config::install(cfg);
        let m = ladder(&c, 8192, 132, 0, false, &[128], &[1, 16, 32]);
        emit_config::install(original);
        let layers = c.layers.len();
        let full_layers = c.layers.iter().filter(|&&full| full).count();
        let count = |program: &packet::devbuild::Program, op: DevOp| {
            program.insts.iter().filter(|d| d.op == op as u16).count()
        };
        for (index, program) in m.progs.iter().enumerate() {
            check_operand_dependencies(program);
            let folded: Vec<_> = program
                .insts
                .iter()
                .filter(|d| {
                    (d.op == DevOp::Gemv as u16 || d.op == DevOp::GemvGlu as u16) && d.i[3] == 4
                })
                .collect();
            if index == 1 {
                assert_eq!(count(program, DevOp::QwenRmsNorm), 0);
                assert_eq!(count(program, DevOp::Residual), 0);
                assert_eq!(count(program, DevOp::Glu), 0);
                assert_eq!(count(program, DevOp::GemvGlu), layers);
                // One input norm per layer, the lm_head's, and the MLP norms.
                assert_eq!(folded.len(), 2 * layers + 1);
                // z/a/b and k/v walks.
                assert_eq!(count(program, DevOp::GemvQkv), layers);
                let mut with_residual = 0;
                for d in &folded {
                    let a_slot = if d.op == DevOp::Gemv as u16 { 5 } else { 4 };
                    assert_ne!(d.i[a_slot], 0);
                    assert_ne!(d.i[7], 0);
                    assert_eq!(d.f[1], 1.0);
                    if d.i[6] != 0 {
                        with_residual += 1;
                        if d.t[3] != TENSOR_NONE {
                            assert_ne!(d.t[3], d.i[a_slot], "residual stream must ping-pong");
                        }
                    } else {
                        assert_eq!(d.t[3], TENSOR_NONE);
                    }
                }
                assert_eq!(with_residual, 2 * layers);
                assert_eq!(count(program, DevOp::QwenHeadNormRope), 3 * full_layers);
            } else {
                assert!(folded.is_empty());
                assert_eq!(count(program, DevOp::QwenRmsNorm), 2 * layers + 1);
                assert_eq!(count(program, DevOp::Residual), 0);
                let riding = program
                    .insts
                    .iter()
                    .filter(|d| d.op == DevOp::QwenRmsNorm as u16 && d.t[4] != TENSOR_NONE)
                    .count();
                assert_eq!(riding, 2 * layers, "every residual add rides a norm");
                let decode = index > 0;
                assert_eq!(count(program, DevOp::Glu), if decode { 0 } else { layers });
                assert_eq!(
                    count(program, DevOp::GemvGlu),
                    if decode { layers } else { 0 }
                );
                assert_eq!(
                    count(program, DevOp::GemvQkv),
                    if decode { layers } else { 0 }
                );
            }
        }
        // plowrt's ladder dispatch compares rungs within a class: the unfolded rungs normalize to
        // one instruction list, the folded rung is its own class.
        let shape = |program: &packet::devbuild::Program| {
            program
                .insts
                .iter()
                .map(|d| {
                    let proj = d.op == DevOp::Gemv as u16 || d.op == DevOp::GemvGlu as u16;
                    (d.op, proj && d.i[3] == 4)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(shape(&m.progs[2]), shape(&m.progs[3]));
        assert_ne!(shape(&m.progs[1]), shape(&m.progs[2]));
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
        assert_eq!(bytes("act.gdn.outstate"), 4 * 48 * 128 * 128 * 4);
        assert_eq!(bytes("act.gdn.initstate"), 4 * 48 * 128 * 128 * 4);
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
    fn decode_cublaslt_isolates_only_bf16_body_projections() {
        let _env = crate::test_env::env_guard();
        let _target = EmitAmdGuard::set(false);
        let _scope = crate::test_env::EnvScope::set(&[("PLOW_EMIT_DECODE_CUBLASLT", "0")]);
        let c = Config::parse(&fixture());
        for batch in [1, 4] {
            std::env::set_var("PLOW_EMIT_DECODE_CUBLASLT", "0");
            {
                let mut config = emit_config::active().clone();
                config.decode_cublaslt = false;
                emit_config::install(config);
            }
            let plain = model_prefill(&c, 8192, 132, 0, &[128], batch, false);
            std::env::set_var("PLOW_EMIT_DECODE_CUBLASLT", "1");
            {
                let mut config = emit_config::active().clone();
                config.decode_cublaslt = true;
                emit_config::install(config);
            }
            let isolated = model_prefill(&c, 8192, 132, 0, &[128], batch, false);
            let metadata = segment_roles(&isolated).unwrap();
            let metadata: plow_asset::segment_roles::SegmentRoles =
                plow_asset::segment_roles::SegmentRoles::from_bytes(&metadata.data).unwrap();
            let decode_roles = &metadata.programs.last().unwrap().roles;
            assert!(metadata.objects.is_empty());
            assert!(decode_roles.contains(&plow_asset::segment_roles::CUBLASLT));
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
            let meta = segment_roles(&control).unwrap();
            let meta: Value = serde_json::from_slice(&meta.data).unwrap();
            assert!(meta["objects"].get("2").is_none());
            cfg.attention_pf_role = true;
            cfg.attention_pf_isolate = false;
            emit_config::install(cfg);
            let candidate = model_prefill(&c, 65536, 132, 0, &[128, 1024, 4096, 8192], 1, fp8);
            let meta = segment_roles(&candidate).unwrap();
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
        let control = segment_roles(&isolated).unwrap();
        let control: Value = serde_json::from_slice(&control.data).unwrap();
        assert!(control["objects"].as_object().unwrap().is_empty());
        assert!(control["programs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["roles"].as_array().unwrap().iter().all(|r| r == 0)));
        cfg.fp8_pf_gemm_role = true;
        emit_config::install(cfg.clone());
        let candidate = segment_roles(&isolated).unwrap();
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
        assert!(segment_roles(&plain).is_none());
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
            let man = manifest::build(&m, "sm_90a", &LeanReport::skipped("fixture"));
            let req = man["backends"]["nvcc"]["requires"].as_array().unwrap();
            for flag in ["PLOW_NV_FP8_PF_SCALE_WFIRST=1", "PLOW_NV_TMA_GEMM=1"] {
                assert!(req.contains(&serde_json::json!(flag)), "missing {flag}");
            }
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
        // A wider decode rung beside the W8A8 prefill buckets is a ladder, not a refusal.
        let laddered = model_prefill(&c, 8192, 132, 0, &[128], 4, true);
        assert_eq!(laddered.prog_t, vec![128, 4]);
        assert!(laddered.progs[1]
            .insts
            .iter()
            .any(|d| d.op == DevOp::GemmFp8 as u16 && d.i[0] == 4));
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
        let man = manifest::build(&mapped, "sm_90a", &LeanReport::skipped("fixture"));
        assert_eq!(man["features"]["qwen_fp8_m1_tma"], true);
        let req = man["backends"]["nvcc"]["requires"].as_array().unwrap();
        for flag in [
            "PLOW_NV_W8A8=1",
            "PLOW_NV_FP8_M1=1",
            "PLOW_NV_QUANT_FP8_VLLM=1",
            "PLOW_NV_TMA_GEMM=1",
            "PLOW_NV_FP8_M1_TMA=1",
            "PLOW_NV_FP8_M1_XCACHE=1",
            "PLOW_NV_FP8_M1_FAST_ACCUM=1",
            "PLOW_NV_FP8_M1_BK1024=1",
        ] {
            assert!(
                req.contains(&serde_json::json!(flag)),
                "missing required flag {flag}"
            );
        }
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
        // Wider rungs quantize on every block and run the GEMM at M = rows without the M1 map.
        let wide = model(&c, 8192, 132, 0, true, 4);
        let g = &wide.progs[0];
        let mut count = 0;
        for (ix, d) in g.insts.iter().enumerate() {
            if d.op != DevOp::GemmFp8 as u16 {
                continue;
            }
            count += 1;
            let q = &g.insts[ix - 1];
            assert_eq!(q.op, DevOp::QuantFp8 as u16);
            assert_eq!(&q.i[..2], &[4, d.i[2]]);
            assert_eq!(q.blocks, 132);
            assert_eq!(d.i[0], 4);
            assert_eq!((d.i[6], d.i[7]), (0, 0));
            assert_eq!(wide.tensors[d.t[1] as usize].bytes, 4 * u64::from(d.i[2]));
            assert_eq!(wide.tensors[d.t[3] as usize].bytes, 16);
        }
        assert_eq!(count, 31);
        assert!(std::panic::catch_unwind(|| model(&c, 8192, 132, 0, true, 129)).is_err());
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
