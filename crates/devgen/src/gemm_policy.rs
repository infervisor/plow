use std::collections::HashMap;

use kernelcaps::{
    HardwareFingerprint, Inventory, KernelId, OpSignature, Phase, ProfileId, QuantScheme,
};
use packet::dev::DevOp;
use serde_json::{json, Value};

/// Supplied-current-cost ordering only; legality and measurement provenance remain external.
pub fn request(
    inventory: &Inventory,
    shape: [u32; 3],
    hardware: &HardwareFingerprint,
    n_cu: u32,
    quant: QuantScheme,
    costs: &HashMap<u16, f64>,
    selected: DevOp,
) -> Option<Value> {
    if shape.contains(&0) || n_cu == 0 {
        return None;
    }
    let [m, n, k] = shape;
    let mut op = OpSignature::gemm(Phase::Prefill, m.into(), n.into(), k.into());
    op.quant = quant;
    let legal = inventory.candidates(&op, hardware, ProfileId::PrefillDense);
    if legal.len() < 2 || !legal.iter().any(|c| c.id == KernelId(selected)) {
        return None;
    }
    let candidates: Vec<_> = legal
        .iter()
        .map(|candidate| {
            let cost = *costs.get(&candidate.id.raw())?;
            if !cost.is_finite() || cost <= 0.0 {
                return None;
            }
            Some((
                format!("{}:{}", candidate.id.raw(), candidate.implementation_hash),
                cost,
            ))
        })
        .collect::<Option<_>>()?;
    let geometry = json!({"m":m,"n":n,"k":k,"n_cu":n_cu,"quant":format!("{quant:?}"),
        "hardware":hardware.tuning_path(),"profile":ProfileId::PrefillDense.label(),
        "inventory_build":inventory.build().label(),
        "case":tunedb::gemm_op_case(m.into(),n.into(),k.into(),quant)});
    let domain = plow_asset::decode_objects::image_sha256(&serde_json::to_vec(&geometry).ok()?);
    let selected = legal.iter().find(|c| c.id == KernelId(selected))?;
    let key = format!("{}:{}", selected.id.raw(), selected.implementation_hash);
    Some(
        json!({"policy_kind":"dense_gemm_selection_v1","selected_opcode":selected.id.raw(),
        "required":[domain],"geometry":geometry,
        "candidates":candidates.iter().map(|(key,cost)|json!({
            "domain":domain,"key":key,"cost":cost,"qualified":true
        })).collect::<Vec<_>>(),"choices":[{"domain":domain,"key":key}],
        "cost_scope":"current kernel-store medians for exact M/N/K and legal registry population; not T4 or an FP implementation proof"}),
    )
}
