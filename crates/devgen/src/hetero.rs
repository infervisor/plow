//! Heterogeneous prefill on a unified-memory SoC (plans/apple-heterogeneous-emit.md).
//!
//! The prefill bucket's rows are split into contiguous blocks owned by the GPU, the Neural
//! Engine and the CPU. The GPU packet carries only its own rows of every projection / norm /
//! residual (`M = rows_gpu`); attention, RoPE, embedding and the lm_head stay on the GPU over
//! all rows. Two host joins per layer (before RoPE, before o_proj) plus one before the final
//! norm are the only synchronization: each is a segment boundary, and the runtime runs the
//! other units' rows of the same segment concurrently with the GPU's command buffer.
//!
//! This sidecar (`hetero.json` beside the `.pkt`) tells the runtime what the other lanes do:
//! per program and segment, the ANE program (one coarse CoreML graph per layer: post-attention
//! of layer `l` fused with pre-attention of layer `l+1`) and the GPU instructions whose row
//! range the CPU re-bases onto its block. Weights are named so the runtime builds the ANE
//! programs from the packet's own tensors (quantized twins dequantized to fp16).

pub use plow_asset::hetero::{ActTensors, AneLane, HeteroPlan, LayerWeights, ProgPlan, SegPlan};

/// `PLOW_ROW_SPLIT=ane=<pct>[,cpu=<pct>]` — row percentages for the non-GPU units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RowSplit {
    pub ane_pct: u32,
    pub cpu_pct: u32,
}

impl RowSplit {
    pub fn parse(spec: &str) -> Result<RowSplit, String> {
        let mut s = RowSplit::default();
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| format!("PLOW_ROW_SPLIT: expected unit=pct, got {part:?}"))?;
            let v: u32 = v
                .trim()
                .parse()
                .map_err(|e| format!("PLOW_ROW_SPLIT: {k}: {e}"))?;
            match k.trim() {
                "ane" | "npu" => s.ane_pct = v,
                "cpu" => s.cpu_pct = v,
                "gpu" => {}
                other => return Err(format!("PLOW_ROW_SPLIT: unknown unit {other:?}")),
            }
        }
        if s.ane_pct + s.cpu_pct >= 100 {
            return Err("PLOW_ROW_SPLIT: the GPU must keep some rows".into());
        }
        Ok(s)
    }

    pub fn from_env() -> Option<RowSplit> {
        let spec = crate::emit_config::active().row_split.clone()?;
        match RowSplit::parse(&spec) {
            Ok(s) if s.ane_pct + s.cpu_pct > 0 => Some(s),
            Ok(_) => None,
            Err(e) => panic!("{e}"),
        }
    }

    /// Row blocks `(gpu, ane, cpu)` for a bucket of `t` rows. Non-GPU blocks are multiples of
    /// 8 (the CPU/ANE lanes have no tile constraint; the GPU GEMM handles any M) and the GPU
    /// keeps at least 8 rows.
    pub fn rows(&self, t: u32) -> (u32, u32, u32) {
        let r8 = |pct: u32| (t * pct / 100) / 8 * 8;
        let (mut a, mut c) = (r8(self.ane_pct), r8(self.cpu_pct));
        while a + c + 8 > t {
            if a >= c && a > 0 {
                a -= 8;
            } else if c > 0 {
                c -= 8;
            } else {
                break;
            }
        }
        (t - a - c, a, c)
    }
}

/// Emission-time collector for one program; `emit_phase` drives it and `flush` closes a segment.
#[derive(Default)]
pub struct ProgCollector {
    pub seg: u32,
    pub ane: Option<AneLane>,
    pub cpu: Vec<u32>,
    pub segments: Vec<SegPlan>,
}

impl ProgCollector {
    pub fn split_op(&mut self, counter: u32) {
        self.cpu.push(counter);
    }
    pub fn flush(&mut self) {
        let ane = self.ane.take();
        let cpu = std::mem::take(&mut self.cpu);
        if ane.is_some() || !cpu.is_empty() {
            self.segments.push(SegPlan {
                seg: self.seg,
                ane,
                cpu_insts: cpu,
            });
        }
        self.seg += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::RowSplit;

    #[test]
    fn rows_are_multiples_of_eight_and_gpu_keeps_some() {
        let s = RowSplit::parse("ane=50,cpu=10").unwrap();
        assert_eq!(s.rows(128), (56, 64, 8));
        assert_eq!(s.rows(512), (208, 256, 48));
        let s = RowSplit::parse("ane=90").unwrap();
        assert_eq!(s.rows(16), (8, 8, 0));
        assert!(RowSplit::parse("ane=60,cpu=40").is_err());
        assert!(RowSplit::parse("gpu=100").unwrap().rows(128) == (128, 0, 0));
    }

    #[test]
    fn apple_mxfp4_emits_ane_weight_bindings() {
        use crate::{EmitArgs, WholeGraphFusionDecisions};
        use plow_asset::hetero::{HeteroPlan, WeightEncoding, SCHEMA};
        let _env = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_MXFP4", "1"),
            ("PLOW_ROW_SPLIT", "ane=50,cpu=10"),
            ("PLOW_MAX_CHUNK", "128"),
        ]);
        let dir = std::env::temp_dir().join(format!("plow-apple-mx4-ane-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
            "model_type":"qwen3","hidden_size":512,"intermediate_size":1024,
            "num_hidden_layers":2,"num_attention_heads":8,"head_dim":64,
            "num_key_value_heads":2,"rms_norm_eps":1e-6,"vocab_size":4096,
            "rope_theta":1000000.0,"tie_word_embeddings":true
        }"#,
        )
        .unwrap();
        crate::run(EmitArgs {
            dir: dir.clone(),
            ctx: 256,
            out: dir.join("model.pkt").to_str().unwrap().into(),
            n_cu: 16,
            tp: 1,
            block_spec: None,
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: "m4pro".into(),
            arch: "metal3".into(),
            emit_cfg: None,
            whole_graph_fusions: WholeGraphFusionDecisions::default(),
        });
        let plan: HeteroPlan =
            serde_json::from_slice(&std::fs::read(dir.join("hetero.json")).unwrap()).unwrap();
        assert_eq!(plan.schema, SCHEMA);
        assert_eq!(plan.weight_encoding, WeightEncoding::Mxfp4);
        assert!(!plan.fp8);
        assert_eq!(plan.layers.len(), 2);
        for layer in &plan.layers {
            for (weight, scale) in [
                (&layer.wq, &layer.sq),
                (&layer.wk, &layer.sk),
                (&layer.wv, &layer.sv),
                (&layer.wo, &layer.so),
                (&layer.wg, &layer.sg),
                (&layer.wu, &layer.su),
                (&layer.wd, &layer.sd),
            ] {
                assert!(weight.starts_with("mxfp4/"), "{weight}");
                assert_eq!(scale.as_deref(), Some(format!("{weight}_scale").as_str()));
            }
        }
        assert!(!plan.programs.is_empty());
        for program in plan.programs {
            assert_eq!(
                (program.rows_gpu, program.rows_ane, program.rows_cpu),
                (56, 64, 8)
            );
            assert!(program.segments.iter().any(|s| s.ane.is_some()));
            assert!(program.segments.iter().any(|s| !s.cpu_insts.is_empty()));
        }
    }
}
