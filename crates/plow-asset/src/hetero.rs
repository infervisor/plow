//! `hetero.json`: the heterogeneous-prefill sidecar the compiler writes beside a `.pkt` for a
//! unified-memory SoC (plans/apple-heterogeneous-emit.md). The GPU packet carries only its own
//! row block of every row-split op; this file tells the runtime which rows the Neural Engine
//! and CPU lanes own, which instructions the CPU re-bases onto its block, and which per-layer
//! ANE program runs in each host segment. Weights are named so the runtime builds the ANE
//! programs from the packet's own tensors.

use serde::{Deserialize, Serialize};

pub const FILE: &str = "hetero.json";
pub const SCHEMA: &str = "plow-hetero-v1";

/// Per-layer weight tensor names (fp8 twin + scale when the packet is w8a16, else bf16).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LayerWeights {
    pub g_in: String,
    pub g_pa: String,
    pub wq: String,
    pub wk: String,
    pub wv: String,
    pub wo: String,
    pub wg: String,
    pub wu: String,
    pub wd: String,
    #[serde(default)]
    pub sq: Option<String>,
    #[serde(default)]
    pub sk: Option<String>,
    #[serde(default)]
    pub sv: Option<String>,
    #[serde(default)]
    pub so: Option<String>,
    #[serde(default)]
    pub sg: Option<String>,
    #[serde(default)]
    pub su: Option<String>,
    #[serde(default)]
    pub sd: Option<String>,
}

/// One ANE program: `pre` = norm + QKV of `layer`; `mid` = o_proj..MLP of `layer` then norm +
/// QKV of `layer + 1`; `post` = o_proj..MLP of the last layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AneLane {
    pub kind: String,
    pub layer: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SegPlan {
    pub seg: u32,
    #[serde(default)]
    pub ane: Option<AneLane>,
    /// Instruction indices of this segment's row-split GPU ops, in dependency order.
    #[serde(default)]
    pub cpu_insts: Vec<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProgPlan {
    pub prog: u32,
    pub t: u32,
    pub rows_gpu: u32,
    pub rows_ane: u32,
    pub rows_cpu: u32,
    pub segments: Vec<SegPlan>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ActTensors {
    pub x: String,
    pub hn: String,
    pub qg: String,
    pub kg: String,
    pub vg: String,
    pub at: String,
    pub og: String,
    pub fu: String,
    pub dg: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HeteroPlan {
    pub schema: String,
    pub arch: String,
    pub hidden: u32,
    pub inter: u32,
    pub qd: u32,
    pub kd: u32,
    pub eps: f32,
    /// 0 = GeGLU (Gemma), 1 = SwiGLU (Llama/Qwen).
    pub mlp_act: u32,
    pub fp8: bool,
    pub ane_pct: u32,
    pub cpu_pct: u32,
    pub tensors: ActTensors,
    pub layers: Vec<LayerWeights>,
    pub programs: Vec<ProgPlan>,
}
