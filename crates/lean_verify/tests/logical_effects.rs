use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};

fn fixture(kind: &str, ordered: bool) -> serde_json::Value {
    let mut b = Builder::new(1);
    let x = b.tensor("x", 16);
    let y = b.tensor("y", 16);
    let z = b.tensor("z", 16);
    let first = b.emit(DevOp::Residual, vec![0], &[], |d| {
        d.t[..3].copy_from_slice(&[y, x, x]);
        d.i[0] = 8;
    });
    let deps = if ordered { vec![first] } else { vec![] };
    let (output, input) = match kind {
        "waw" => (y, x),
        "read_read" => (z, x),
        "transitive" => (z, y),
        _ => (x, y),
    };
    let second = b.emit(DevOp::Residual, vec![0], &deps, |d| {
        d.t[..3].copy_from_slice(&[output, input, input]);
        d.i[0] = 8;
    });
    if kind == "transitive" {
        b.emit(DevOp::Residual, vec![0], &[second], |d| {
            d.t[..3].copy_from_slice(&[x, z, z]);
            d.i[0] = 8;
        });
    }
    let p = b.finish();
    let model = Model {
        n_cu: 1,
        target: 0,
        tensors: p.tensors.clone(),
        progs: vec![p],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen: vec![],
    };
    plow_asset::program::with_model(&model, |packet| {
        plow_asset::logical_effects::obligation(packet, 0).unwrap()
    })
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn packet_derived_effects_reject_missing_raw_war_waw() {
    for kind in ["raw_war", "waw", "transitive", "read_read"] {
        let valid = fixture(kind, true);
        let cert = lean_verify::call("D", valid).unwrap();
        assert!(cert.ok, "{kind}: {:?}", cert.reason);
        let broken = fixture(kind, false);
        let cert = lean_verify::call("D", broken).unwrap();
        assert_eq!(cert.ok, kind == "read_read", "{kind}: {:?}", cert.reason);
    }
}
