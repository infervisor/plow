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

fn speech_obligation(build: impl FnOnce(&mut Builder)) -> serde_json::Value {
    let mut b = Builder::new(1);
    build(&mut b);
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

fn frame_embedding(mode: u32, missing: Option<usize>) -> serde_json::Value {
    speech_obligation(|b| {
        let out = b.tensor("act.frame", 16);
        let inputs = [
            ("in.table", 256),
            ("in.tokens", 4),
            ("in.codebooks", 512),
            ("in.codes", 8),
            ("in.norm", 32),
        ];
        let count = if mode & 2 == 0 { 4 } else { 5 };
        let mut handles = Vec::new();
        let mut deps = Vec::new();
        for (slot, &(name, bytes)) in inputs[..count].iter().enumerate() {
            let h = b.tensor(name, bytes);
            handles.push(h);
            let producer = b.emit(DevOp::ZeroF32, vec![0], &[], |d| {
                d.t[0] = h;
                d.i[0] = (bytes / 4) as u32;
            });
            if missing != Some(slot) {
                deps.push(producer);
            }
        }
        let embed = b.emit(DevOp::EmbedFrameBf16, vec![0], &deps, |d| {
            d.t[0] = out;
            d.t[1..=count].copy_from_slice(&handles);
            d.i[..8].copy_from_slice(&[1, 8, 16, 2, 16, 0, 16, mode]);
            d.f[0] = 1.0;
            d.f[1] = 1e-6;
        });
        let next = b.tensor("act.next", 16);
        let deps = if missing == Some(count) {
            vec![]
        } else {
            vec![embed]
        };
        b.emit(DevOp::Residual, vec![0], &deps, |d| {
            d.t[..3].copy_from_slice(&[next, out, out]);
            d.i[0] = 8;
        });
    })
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn frame_embedding_effects_reject_missing_input_and_output_dependencies() {
    for mode in 0..4 {
        let cert = lean_verify::call("D", frame_embedding(mode, None)).unwrap();
        assert!(cert.ok, "mode {mode}: {:?}", cert.reason);
        let inputs = if mode & 2 == 0 { 4 } else { 5 };
        for missing in 0..=inputs {
            let cert = lean_verify::call("D", frame_embedding(mode, Some(missing))).unwrap();
            assert!(
                !cert.ok,
                "mode {mode}, missing dependency {missing}: {:?}",
                cert.reason
            );
        }
    }
}

/// The codec's shared noise buffer: the next RandF32 must wait for the previous noise reader
/// (write-after-read), not just for the previous RandF32.
fn rand_noise(war_ordered: bool) -> serde_json::Value {
    speech_obligation(|b| {
        let seed = b.tensor("in.seed", 8);
        let noise = b.tensor("act.noise", 64);
        let x = b.tensor("act.x", 64);
        let y = b.tensor("act.y", 64);
        let rand = |b: &mut Builder, deps: &[u32]| {
            b.emit(DevOp::RandF32, vec![0], deps, |d| {
                d.t[..2].copy_from_slice(&[noise, seed]);
                d.i[..3].copy_from_slice(&[1, 4, 4]);
            })
        };
        let first = rand(b, &[]);
        let reader = b.emit(DevOp::BinaryF32, vec![0], &[first], |d| {
            d.t[..3].copy_from_slice(&[y, x, noise]);
            d.i[..4].copy_from_slice(&[1, 4, 4, 2]);
        });
        rand(b, &[if war_ordered { reader } else { first }]);
    })
}

/// Two CopyColsF32 demux writes into columns of one stride-2 tensor, unordered.
fn banded_writes(second_col: u32) -> serde_json::Value {
    speech_obligation(|b| {
        let codes = b.tensor("in.codes", 64);
        let level = b.tensor("act.level", 64);
        let out = b.tensor("act.out", 64);
        let copy = |b: &mut Builder, col: u32| {
            b.emit(DevOp::CopyColsF32, vec![0], &[], |d| {
                d.t[..2].copy_from_slice(&[level, codes]);
                d.i[..7].copy_from_slice(&[1, 8, 1, 2, col, 2, col]);
            })
        };
        let a = copy(b, 0);
        let c = copy(b, second_col);
        b.emit(DevOp::UnaryF32, vec![0], &[a, c], |d| {
            d.t[..2].copy_from_slice(&[out, level]);
            d.i[..2].copy_from_slice(&[8, 2]);
        });
    })
}

/// Two projections sharing one split-K workspace (tickets + partials).
fn shared_splitk_scratch(op: DevOp, ordered: bool) -> serde_json::Value {
    speech_obligation(|b| {
        let a = b.tensor("act.a", 64);
        let w = b.tensor("w.w", 64);
        let c0 = b.tensor("act.c0", 64);
        let c1 = b.tensor("act.c1", 64);
        let scratch = b.tensor("act.splitk", 4096 + 64);
        let gemm = |b: &mut Builder, out: u32, deps: &[u32]| {
            b.emit(op, vec![0], deps, |d| {
                d.t[..3].copy_from_slice(&[out, a, w]);
                match op {
                    DevOp::DenseGemmF32 => {
                        d.t[4] = scratch;
                        d.i[..3].copy_from_slice(&[4, 4, 4]);
                    }
                    DevOp::Conv1dF32 => {
                        d.t[7] = scratch;
                        d.i[..8].copy_from_slice(&[1, 4, 4, 4, 1, 1, 1, 1]);
                        d.j[1] = 1 << 20;
                    }
                    _ => unreachable!(),
                }
            })
        };
        let first = gemm(b, c0, &[]);
        let deps = if ordered { vec![first] } else { vec![] };
        gemm(b, c1, &deps);
    })
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn speech_effects_order_noise_war_bands_and_splitk_scratch() {
    let check = |name: &str, request: serde_json::Value, expect: bool| {
        let cert = lean_verify::call("D", request).unwrap();
        assert_eq!(cert.ok, expect, "{name}: {:?}", cert.reason);
    };
    check("noise WAR ordered", rand_noise(true), true);
    check("noise WAR missing", rand_noise(false), false);
    check("disjoint column bands", banded_writes(1), true);
    check("same column band", banded_writes(0), false);
    for op in [DevOp::DenseGemmF32, DevOp::Conv1dF32] {
        check(
            "split-K scratch ordered",
            shared_splitk_scratch(op, true),
            true,
        );
        check(
            "split-K scratch shared unordered",
            shared_splitk_scratch(op, false),
            false,
        );
    }
    // Forged parent-pointer certificates: a self parent (not decreasing) or a non-edge.
    for forge in [0u64, 1] {
        let mut request = rand_noise(true);
        let trees = request["address_trees"].as_array_mut().unwrap();
        let mut forged = false;
        for tree in trees.iter_mut() {
            for (node, parent) in tree["parent"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                if parent.is_u64() && parent.as_u64() != Some(0) {
                    *parent = serde_json::json!(if forge == 0 { node as u64 } else { 0 });
                    forged = true;
                }
            }
        }
        assert!(forged);
        check("forged tree", request, false);
    }
}
