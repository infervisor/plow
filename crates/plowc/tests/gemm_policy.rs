#![cfg(feature = "lean-verify")]

use std::collections::HashMap;

use kernelcaps::{BuildId, HardwareFingerprint, Inventory, IsaLevel, KernelSpec, QuantScheme};
use packet::dev::DevOp;

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn actual_gemm_policy_population_checks_exact_shape_and_rejects_wrong_choice() {
    let inventory = Inventory::probed(
        BuildId::new(IsaLevel::Gfx950, [], "test", "source"),
        [
            KernelSpec::gemm_tile(DevOp::Gemm, IsaLevel::Gfx950, 256, 256, 64, "large"),
            KernelSpec::gemm_tile(DevOp::GemmMed, IsaLevel::Gfx950, 128, 128, 64, "medium"),
            KernelSpec::gemm_tile(DevOp::GemmSmall, IsaLevel::Gfx950, 64, 128, 64, "small"),
        ],
    );
    let hardware =
        HardwareFingerprint::from_spec(hwspec::registry::lookup("MI350X").unwrap()).unwrap();
    let mut costs = HashMap::from([
        (DevOp::Gemm as u16, 300.0),
        (DevOp::GemmMed as u16, 100.0),
        (DevOp::GemmSmall as u16, 200.0),
    ]);
    let make = |costs: &HashMap<u16, f64>, shape, selected| {
        devgen::gemm_policy::request(
            &inventory,
            shape,
            &hardware,
            256,
            QuantScheme::None,
            costs,
            selected,
        )
    };
    let valid = make(&costs, [128, 4096, 6144], DevOp::GemmMed).unwrap();
    assert_eq!(valid["candidates"].as_array().unwrap().len(), 3);
    assert!(lean_verify::call("R", valid.clone()).unwrap().ok);
    let mut b = packet::devbuild::Builder::new(256);
    let out = b.tensor("output", 128 * 4096 * 2);
    let input = b.tensor("input", 128 * 6144 * 2);
    let weight = b.tensor("weight", 4096 * 6144 * 2);
    b.emit(DevOp::GemmMed, (0..256).collect(), &[], |d| {
        d.t[..3].copy_from_slice(&[out, input, weight]);
        d.i[..3].copy_from_slice(&[128, 4096, 6144]);
    });
    let p = b.finish();
    let model = packet::devbuild::Model {
        n_cu: 256,
        target: 0,
        tensors: p.tensors.clone(),
        progs: vec![p, packet::devbuild::Builder::new(256).finish()],
        prog_t: vec![128, 1],
        gen: vec![],
        kv_row_insts: vec![],
    };
    let bound = plow_asset::program::with_model(&model, |packet| {
        plow_asset::gemm_policy::bind(packet, 0, &valid).unwrap()
    });
    assert!(lean_verify::call("R", bound).unwrap().ok);
    let wrong = make(&costs, [128, 4096, 6144], DevOp::Gemm).unwrap();
    assert!(!lean_verify::call("R", wrong).unwrap().ok);
    let tail = make(&costs, [128, 4097, 6145], DevOp::GemmMed).unwrap();
    assert_ne!(valid["required"], tail["required"]);
    assert!(make(&costs, [0, 4096, 6144], DevOp::GemmMed).is_none());
    assert!(make(&costs, [128, 4096, 6144], DevOp::Residual).is_none());
    costs.remove(&(DevOp::Gemm as u16));
    assert!(make(&costs, [128, 4096, 6144], DevOp::GemmMed).is_none());
    for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        costs.insert(DevOp::Gemm as u16, invalid);
        assert!(make(&costs, [128, 4096, 6144], DevOp::GemmMed).is_none());
    }
}
