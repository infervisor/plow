//! `hetero.json`: the heterogeneous-prefill sidecar the compiler writes beside a `.pkt` for a
//! unified-memory SoC (plans/apple-heterogeneous-emit.md). The GPU packet carries only its own
//! row block of every row-split op; this file tells the runtime which rows the Neural Engine
//! and CPU lanes own, which instructions the CPU re-bases onto its block, and which per-layer
//! ANE program runs in each host segment. Weights are named so the runtime builds the ANE
//! programs from the packet's own tensors.

use serde::{Deserialize, Serialize};

pub const FILE: &str = "hetero.json";
pub const SCHEMA: &str = "plow-hetero-v2";

#[derive(Debug)]
pub enum Plan {
    Row(HeteroPlan),
    Channel(crate::hetero_channel::ChannelPlan),
}

pub fn parse(bytes: &[u8]) -> Result<Plan, String> {
    #[derive(Deserialize)]
    struct Header {
        schema: String,
    }
    let header: Header = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    match header.schema.as_str() {
        "plow-hetero-v1" | SCHEMA => serde_json::from_slice(bytes)
            .map(Plan::Row)
            .map_err(|e| e.to_string()),
        crate::hetero_channel::SCHEMA => serde_json::from_slice(bytes)
            .map(Plan::Channel)
            .map_err(|e| e.to_string()),
        other => Err(format!("unsupported heterogeneous schema {other:?}")),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightEncoding {
    #[default]
    Bf16,
    Fp8,
    Mxfp4,
}

/// Per-layer weight tensor names and scales for the plan's weight encoding.
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
    /// Legacy v1 encoding flag; v2 carries `weight_encoding`.
    #[serde(default)]
    pub fp8: bool,
    #[serde(default)]
    pub weight_encoding: WeightEncoding,
    pub ane_pct: u32,
    pub cpu_pct: u32,
    pub tensors: ActTensors,
    pub layers: Vec<LayerWeights>,
    pub programs: Vec<ProgPlan>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_encoding_and_mxfp4_roundtrip() {
        let mut old = serde_json::to_value(HeteroPlan::default()).unwrap();
        old.as_object_mut().unwrap().remove("weight_encoding");
        old["schema"] = "plow-hetero-v1".into();
        old["fp8"] = true.into();
        let legacy: HeteroPlan = serde_json::from_value(old).unwrap();
        assert!(legacy.fp8);
        assert_eq!(legacy.weight_encoding, WeightEncoding::Bf16);
        let plan = HeteroPlan {
            schema: SCHEMA.into(),
            weight_encoding: WeightEncoding::Mxfp4,
            ..HeteroPlan::default()
        };
        let bytes = serde_json::to_vec(&plan).unwrap();
        let loaded: HeteroPlan = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(loaded.weight_encoding, WeightEncoding::Mxfp4);
        assert!(!loaded.fp8);
    }
}
