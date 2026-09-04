use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use nn_graph::{Graph, LinearAttnKind, Op, TensorId};
use packet::dev::{DevInst, DevOp};
use packet::devbuild::Model;
use rewrite::FusedGraph;

const KDA_GATE: &str = "FusedKdaGatedNorm";

#[derive(Clone, Debug)]
pub struct FusionCoverage {
    pub graph_ops: usize,
    pub extracted: usize,
    pub by_op: BTreeMap<String, usize>,
    pub same_input_narrow_pairs: usize,
    parallel_linear2: Vec<devgen::ParallelLinear2Decision>,
    kda_gate: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverageReport {
    pub gpu_equivalent_covered: usize,
    pub not_opcode_equivalent: usize,
}

pub enum Analysis {
    Ineligible,
    Covered(FusionCoverage),
    AdvisoryFailure(String),
    RequiredFailure(String),
}

impl FusionCoverage {
    pub fn analyze(dir: &Path) -> Analysis {
        let json = match std::fs::read_to_string(dir.join("config.json")) {
            Ok(json) => json,
            Err(e) => return Analysis::AdvisoryFailure(format!("no readable config.json: {e}")),
        };
        let mut graph = match nn_graph::models::build_text_generation_from_config_json_at(
            &json,
            &nn_graph::models::ShapeBucket::default(),
        ) {
            Ok(graph) => graph,
            Err(e) => return Analysis::AdvisoryFailure(format!("graph build failed: {e}")),
        };
        graph.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 8192));
        if !requires_kda_coverage(&graph) {
            return Analysis::Ineligible;
        }
        match rewrite::rewrite_graph(&graph) {
            Ok((fused, stats)) => {
                Analysis::Covered(Self::from_graphs(&graph, &fused, stats.ops_before))
            }
            Err(e) => Analysis::RequiredFailure(format!("whole-graph extraction failed: {e}")),
        }
    }

    fn from_graphs(graph: &Graph, fused: &FusedGraph, graph_ops: usize) -> Self {
        let mut by_op = BTreeMap::new();
        for node in &fused.nodes {
            if node.op.starts_with("Fused") || node.op == "SwiGLU" {
                *by_op.entry(node.op.clone()).or_default() += 1;
            }
        }
        let extracted = by_op.values().sum();
        let kda_gate = by_op.get(KDA_GATE).copied().unwrap_or(0);
        let parallel_linear2 = same_input_narrow_linear2(graph);
        Self {
            graph_ops,
            extracted,
            by_op,
            same_input_narrow_pairs: parallel_linear2.iter().map(|d| d.instances).sum(),
            parallel_linear2,
            kda_gate,
        }
    }

    pub fn decisions(&self, tp: u32, qualified: bool) -> devgen::WholeGraphFusionDecisions {
        let mut parallel_linear2 = self.parallel_linear2.clone();
        for decision in &mut parallel_linear2 {
            decision.qualified = qualified;
        }
        devgen::WholeGraphFusionDecisions {
            tp,
            parallel_linear2,
        }
    }

    pub fn parallel_linear2(&self) -> &[devgen::ParallelLinear2Decision] {
        &self.parallel_linear2
    }

    pub fn validate(&self, model: &Model) -> Result<CoverageReport, String> {
        let decode = model
            .progs
            .last()
            .ok_or_else(|| "emitted model has no decode program".to_string())?;
        // The egg graph is built from the full checkpoint config, while an
        // intentional PLOW_K3_LAYERS truncation emits only a prefix.  Count the
        // emitted KDA layers from their unique checkpoint tensor instead of
        // imposing the full graph's obligation on a diagnostic artifact.
        let emitted_kda = model
            .tensors
            .iter()
            .filter(|tensor| tensor.name.ends_with(".self_attn.dt_bias"))
            .count();
        if emitted_kda > self.kda_gate {
            return Err(format!(
                "emitted {emitted_kda} KDA layers but whole-graph extraction found only {}",
                self.kda_gate
            ));
        }
        self.validate_decode_insts_for(&decode.insts, emitted_kda)
    }

    #[cfg(test)]
    fn validate_decode_insts(&self, insts: &[DevInst]) -> Result<CoverageReport, String> {
        self.validate_decode_insts_for(insts, self.kda_gate)
    }

    fn validate_decode_insts_for(
        &self,
        insts: &[DevInst],
        expected_kda: usize,
    ) -> Result<CoverageReport, String> {
        let count = |op: DevOp| insts.iter().filter(|i| i.op == op as u16).count();
        let qkvg = count(DevOp::GemvQkvg);
        // KdaDecodeFused subsumes the conv/state/gated-norm half, but the gate
        // projection still rides GemvQkvg. Either spelling covers the egg
        // target exactly; neither opcode covers it alone.
        let gated_norm = count(DevOp::KdaGatedNorm) + count(DevOp::KdaDecodeFused);
        if qkvg < expected_kda || gated_norm < expected_kda {
            return Err(format!(
                "{KDA_GATE}: expected {expected_kda} of {} extracted, but decode carries GemvQkvg={qkvg} and \
                 KdaGatedNorm-or-KdaDecodeFused={gated_norm}; exact semantic coverage requires both",
                self.kda_gate
            ));
        }
        Ok(CoverageReport {
            gpu_equivalent_covered: expected_kda,
            not_opcode_equivalent: self.extracted - self.kda_gate,
        })
    }
}

fn requires_kda_coverage(graph: &Graph) -> bool {
    graph.nodes.iter().any(|node| {
        matches!(
            &node.op,
            Op::LinearAttention {
                kind: LinearAttnKind::KimiDelta,
                ..
            }
        )
    })
}

fn same_input_narrow_linear2(graph: &Graph) -> Vec<devgen::ParallelLinear2Decision> {
    let mut groups: HashMap<TensorId, Vec<(u32, u32)>> = HashMap::new();
    for node in &graph.nodes {
        let Op::Linear { out_features, .. } = &node.op else {
            continue;
        };
        if *out_features > 128 {
            continue;
        }
        if let Some(&input) = node.inputs.first() {
            let Some(k) = graph
                .tensor(input)
                .shape
                .as_ref()
                .and_then(|s| s.last())
                .and_then(|d| d.as_static())
                .and_then(|k| u32::try_from(k).ok())
            else {
                continue;
            };
            groups
                .entry(input)
                .or_default()
                .push((*out_features as u32, k));
        }
    }
    let mut shapes: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    for group in groups.values() {
        for (i, &(n0, k0)) in group.iter().enumerate() {
            for &(n1, k1) in &group[i + 1..] {
                if k0 == k1 {
                    *shapes.entry((n0, n1, k0)).or_default() += 1;
                }
            }
        }
    }
    shapes
        .into_iter()
        .map(|((n0, n1, k), instances)| devgen::ParallelLinear2Decision {
            n0,
            n1,
            k,
            instances,
            qualified: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rewrite::{Arg, FNode};

    fn coverage(kda: usize, other: usize) -> FusionCoverage {
        let mut fused = FusedGraph::default();
        for _ in 0..kda {
            fused.nodes.push(FNode {
                op: KDA_GATE.into(),
                args: vec![],
            });
        }
        for _ in 0..other {
            fused.nodes.push(FNode {
                op: "FusedMlaOutGate".into(),
                args: vec![Arg::Int(0)],
            });
        }
        FusionCoverage::from_graphs(
            &Graph {
                syms: Default::default(),
                tensors: vec![],
                nodes: vec![],
                blocks: vec![],
                inputs: vec![],
                outputs: vec![],
                fp8_scale_bindings: vec![],
            },
            &fused,
            0,
        )
    }

    fn inst(op: DevOp) -> DevInst {
        let mut i = DevInst::default();
        i.op = op as u16;
        i
    }

    #[test]
    fn exact_kda_bundle_reaches_gpu_semantics() {
        let c = coverage(2, 3);
        let report = c
            .validate_decode_insts(&[
                inst(DevOp::GemvQkvg),
                inst(DevOp::KdaGatedNorm),
                inst(DevOp::GemvQkvg),
                inst(DevOp::KdaGatedNorm),
            ])
            .unwrap();
        assert_eq!(report.gpu_equivalent_covered, 2);
        assert_eq!(report.not_opcode_equivalent, 3);
    }

    #[test]
    fn exact_kda_bundle_fails_closed_when_either_half_is_missing() {
        let c = coverage(2, 0);
        let err = c
            .validate_decode_insts(&[
                inst(DevOp::GemvQkvg),
                inst(DevOp::GemvQkvg),
                inst(DevOp::KdaGatedNorm),
            ])
            .unwrap_err();
        assert!(err.contains("exact semantic coverage requires both"));
    }

    #[test]
    fn truncated_kda_bundle_checks_only_emitted_prefix() {
        let c = coverage(69, 0);
        let report = c
            .validate_decode_insts_for(&[inst(DevOp::GemvQkvg), inst(DevOp::KdaGatedNorm)], 1)
            .unwrap();
        assert_eq!(report.gpu_equivalent_covered, 1);
    }

    #[test]
    fn finds_structural_same_input_narrow_linear_pair() {
        let json = r#"{
          "model_type": "kimi_k3",
          "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 1000, "hidden_size": 256, "intermediate_size": 512,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "q_lora_rank": 64, "kv_lora_rank": 32,
            "qk_rope_head_dim": 16, "qk_nope_head_dim": 32, "v_head_dim": 32,
            "mla_use_output_gate": true,
            "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 1,
            "moe_intermediate_size": 128, "routed_expert_hidden_size": 192,
            "first_k_dense_replace": 1, "attn_res_block_size": 2,
            "linear_attn_config": {
              "num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4,
              "use_full_rank_gate": true,
              "full_attn_layers": [], "kda_layers": [1]
            }
          }
        }"#;
        let graph = nn_graph::models::build_from_config_json(json).unwrap();
        let decisions = same_input_narrow_linear2(&graph);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].instances, 1);
        assert_eq!((decisions[0].n0, decisions[0].n1), (128, 96));
    }

    #[test]
    fn non_kda_full_graph_has_no_opcode_obligation() {
        let json = r#"{
          "model_type": "llama", "vocab_size": 100, "hidden_size": 64,
          "intermediate_size": 128, "num_hidden_layers": 1,
          "num_attention_heads": 2, "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5, "rope_theta": 10000.0
        }"#;
        let graph = nn_graph::models::build_from_config_json(json).unwrap();
        assert!(!requires_kda_coverage(&graph));
        let (fused, stats) = rewrite::rewrite_graph(&graph).unwrap();
        let coverage = FusionCoverage::from_graphs(&graph, &fused, stats.ops_before);
        let report = coverage.validate_decode_insts(&[]).unwrap();
        assert_eq!(report.gpu_equivalent_covered, 0);
        assert_eq!(report.not_opcode_equivalent, coverage.extracted);
    }

    #[test]
    fn parallel_linear2_qualification_is_explicit_and_eligibility_preserving() {
        let candidate = devgen::ParallelLinear2Decision {
            n0: 128,
            n1: 96,
            k: 7168,
            instances: 69,
            qualified: false,
        };
        let coverage = FusionCoverage {
            graph_ops: 0,
            extracted: 0,
            by_op: BTreeMap::new(),
            same_input_narrow_pairs: 69,
            parallel_linear2: vec![candidate.clone()],
            kda_gate: 0,
        };

        let control = coverage.decisions(8, false);
        assert_eq!(control.parallel_linear2, vec![candidate]);

        let experiment = coverage.decisions(8, true);
        assert_eq!(experiment.parallel_linear2.len(), 1);
        assert!(experiment.parallel_linear2[0].qualified);
        assert_eq!(experiment.parallel_linear2[0].instances, 69);
    }
}
