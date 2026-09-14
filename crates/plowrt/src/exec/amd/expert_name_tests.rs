use super::*;
use crate::asset::checkpoint::Checkpoint;

/// A one-shard safetensors directory holding `(name, dtype, shape)` with
/// zeroed data — the resolver reads names, dtypes and shapes, never payload.
struct Fake(std::path::PathBuf);

impl Fake {
    fn new(tag: &str, tensors: &[(&str, &str, &[usize])]) -> Fake {
        let dir = std::env::temp_dir().join(format!("plowrt-expert-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let width = |dt: &str| match dt {
            "F32" => 4,
            "U8" | "F8_E4M3" => 1,
            other => panic!("test helper has no width for {other}"),
        };
        let (mut hdr, mut data, mut off) = (String::from("{"), Vec::new(), 0usize);
        for (i, (n, dt, sh)) in tensors.iter().enumerate() {
            let len = sh.iter().product::<usize>() * width(dt);
            if i > 0 {
                hdr.push(',');
            }
            hdr.push_str(&format!(
                "{n:?}:{{\"dtype\":\"{dt}\",\"shape\":{sh:?},\"data_offsets\":[{off},{}]}}",
                off + len
            ));
            off += len;
            data.resize(off, 0u8);
        }
        hdr.push('}');
        let mut blob = (hdr.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(hdr.as_bytes());
        blob.extend_from_slice(&data);
        std::fs::write(dir.join("model.safetensors"), blob).unwrap();
        Fake(dir)
    }

    fn open(&self) -> Checkpoint {
        Checkpoint::open(&self.0).unwrap()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn refs<'a>(t: &'a [(String, &'static str, Vec<usize>)]) -> Vec<(&'a str, &'a str, &'a [usize])> {
    t.iter()
        .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
        .collect()
}

/// GLM-5.2's layout, and the ONE case that must stay bit-for-bit what it
/// was: the first candidate probed is the pair of names this resolver
/// replaced hardcoded, so a block-fp8 checkpoint never reaches a second
/// lookup and every name built downstream is the name it was built before.
#[test]
fn block_fp8_resolves_to_exactly_the_names_it_always_did() {
    // zai-org/GLM-5.2-FP8, scaled down: [N, K] fp8 + [N/128, K/128] f32.
    let mut t: Vec<(String, &str, Vec<usize>)> = Vec::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        t.push((
            format!("model.layers.3.mlp.experts.0.{p}.weight"),
            "F8_E4M3",
            vec![256, 384],
        ));
        t.push((
            format!("model.layers.3.mlp.experts.0.{p}.weight_scale_inv"),
            "F32",
            vec![2, 3],
        ));
    }
    let f = Fake::new("blockfp8", &refs(&t));
    let c = f.open();
    let en = resolve_expert_names(&c, "model.layers.3.mlp.").unwrap();
    assert_eq!(en.ns, "model.layers.3.mlp.experts.");
    assert_eq!(en.proj, ["gate_proj", "up_proj", "down_proj"]);
    assert_eq!(en.payload, ".weight");
    assert_eq!(en.scale, ".weight_scale_inv");
    assert!(!en.microscaled());
    assert_eq!(
        en.weight_of(0, 0, false),
        "model.layers.3.mlp.experts.0.gate_proj.weight"
    );
    assert_eq!(
        en.scale_of(0, 2, false),
        "model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"
    );
    check_expert_geometry(&c, &en).unwrap();
}

/// amd/Kimi-K2.7-Code-MXFP4: the STANDARD projection names with an E8M0
/// scale. This is why the layout cannot be one boolean — "mxfp4" and
/// "Mixtral-spelled" are independent facts, and this checkpoint has the
/// first without the second.
#[test]
fn mxfp4_under_the_standard_projection_names_resolves_on_the_scale_alone() {
    let mut t: Vec<(String, &str, Vec<usize>)> = Vec::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        // [N, K/2] packed + [N, K/32] E8M0, K = 96.
        t.push((
            format!("m.layers.3.mlp.experts.0.{p}.weight"),
            "U8",
            vec![64, 48],
        ));
        t.push((
            format!("m.layers.3.mlp.experts.0.{p}.weight_scale"),
            "U8",
            vec![64, 3],
        ));
    }
    let f = Fake::new("mxstd", &refs(&t));
    let c = f.open();
    let en = resolve_expert_names(&c, "m.layers.3.mlp.").unwrap();
    assert_eq!(en.proj, ["gate_proj", "up_proj", "down_proj"]);
    assert_eq!(en.payload, ".weight");
    assert_eq!(en.scale, ".weight_scale");
    assert!(en.microscaled());
    check_expert_geometry(&c, &en).unwrap();
}

/// Kimi-K3: `block_sparse_moe.experts.{e}.w1|w2|w3` + `weight_packed` /
/// `weight_scale`, reached from a table declared under the compiler's own
/// `moe.` namespace. Three things are discovered at once — the namespace,
/// the projection names, and the payload suffix — and the slot order is
/// gate, up, down, so `w3` must come back SECOND and `w2` LAST.
#[test]
fn mixtral_mxfp4_resolves_namespace_projections_and_payload_together() {
    const P: &str = "language_model.model.layers.1.block_sparse_moe.experts.0.";
    let t: Vec<(String, &str, Vec<usize>)> = vec![
        (format!("{P}w1.weight_packed"), "U8", vec![64, 48]),
        (format!("{P}w1.weight_scale"), "U8", vec![64, 3]),
        (format!("{P}w3.weight_packed"), "U8", vec![64, 48]),
        (format!("{P}w3.weight_scale"), "U8", vec![64, 3]),
        (format!("{P}w2.weight_packed"), "U8", vec![96, 32]),
        (format!("{P}w2.weight_scale"), "U8", vec![96, 2]),
    ];
    let f = Fake::new("k3", &refs(&t));
    let c = f.open();
    // The K3 emitter declares `moe.{lp}expert_weight_table` (devgen/k3.rs),
    // so this is the prefix the loader is actually handed.
    let en = resolve_expert_names(&c, "moe.language_model.model.layers.1.").unwrap();
    assert_eq!(en.ns, P.trim_end_matches("0."));
    assert_eq!(en.proj, ["w1", "w3", "w2"], "slot order is gate, up, DOWN");
    assert_eq!(en.payload, ".weight_packed");
    assert_eq!(en.scale, ".weight_scale");
    assert_eq!(en.weight_of(0, 2, false), format!("{P}w2.weight_packed"));
    check_expert_geometry(&c, &en).unwrap();
}

/// A scale that is the wrong size for its weight must be refused BY NAME.
/// Nothing downstream would notice: the bytes are all u8, the shape is
/// plausible, the packed buffer comes out the declared size, and every group
/// after the first is scaled by the wrong exponent.
#[test]
fn a_scale_that_covers_the_wrong_k_is_refused_and_named() {
    const P: &str = "m.layers.0.mlp.experts.0.";
    let t: Vec<(String, &str, Vec<usize>)> = vec![
        (format!("{P}gate_proj.weight"), "U8", vec![64, 48]),
        // 4 groups of 32 = 128 elements, but the payload packs 96.
        (format!("{P}gate_proj.weight_scale"), "U8", vec![64, 4]),
        (format!("{P}up_proj.weight"), "U8", vec![64, 48]),
        (format!("{P}up_proj.weight_scale"), "U8", vec![64, 3]),
        (format!("{P}down_proj.weight"), "U8", vec![48, 32]),
        (format!("{P}down_proj.weight_scale"), "U8", vec![48, 2]),
    ];
    let f = Fake::new("badk", &refs(&t));
    let c = f.open();
    let en = resolve_expert_names(&c, "m.layers.0.mlp.").unwrap();
    let e = check_expert_geometry(&c, &en).unwrap_err().to_string();
    assert!(e.contains("gate_proj.weight_scale"), "{e}");
    assert!(e.contains("K disagrees"), "{e}");
}

/// The same for a block-fp8 grid — the arm GLM-5.2 ships on, so it is pinned
/// that the check accepts the real geometry and rejects a near miss.
#[test]
fn a_block_fp8_grid_of_the_wrong_shape_is_refused() {
    const P: &str = "m.layers.0.mlp.experts.0.";
    for (tag, grid, ok) in [("good", vec![2usize, 3], true), ("bad", vec![3, 2], false)] {
        let t: Vec<(String, &str, Vec<usize>)> = ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .flat_map(|p| {
                [
                    (format!("{P}{p}.weight"), "F8_E4M3", vec![256, 384]),
                    (format!("{P}{p}.weight_scale_inv"), "F32", grid.clone()),
                ]
            })
            .collect();
        let f = Fake::new(&format!("grid{tag}"), &refs(&t));
        let c = f.open();
        let en = resolve_expert_names(&c, "m.layers.0.mlp.").unwrap();
        assert_eq!(check_expert_geometry(&c, &en).is_ok(), ok, "{tag}");
    }
}

/// A payload with NO scale under either spelling is a broken checkpoint, not
/// a spelling this loader has yet to learn — say so, and name both.
#[test]
fn a_payload_without_any_scale_names_both_spellings() {
    let t: Vec<(String, &str, Vec<usize>)> = vec![(
        "m.layers.0.mlp.experts.0.gate_proj.weight".into(),
        "U8",
        vec![64, 48],
    )];
    let f = Fake::new("noscale", &refs(&t));
    let e = resolve_expert_names(&f.open(), "m.layers.0.mlp.")
        .unwrap_err()
        .to_string();
    assert!(e.contains("MISSING EXPERT SCALE"), "{e}");
    assert!(
        e.contains("weight_scale_inv") && e.contains("weight_scale"),
        "{e}"
    );
}

/// No routed experts under ANY spelling fails loudly with what was probed.
/// A zero-filled expert buffer is read by the kernel as real weights, so the
/// alternative to this error is a model that loads and is nonsense.
#[test]
fn no_experts_at_all_reports_every_name_it_probed() {
    let t: Vec<(String, &str, Vec<usize>)> =
        vec![("m.layers.0.mlp.gate.weight".into(), "U8", vec![8, 8])];
    let f = Fake::new("noexp", &refs(&t));
    let e = resolve_expert_names(&f.open(), "m.layers.0.mlp.")
        .unwrap_err()
        .to_string();
    assert!(e.contains("MISSING EXPERT WEIGHT"), "{e}");
    assert!(e.contains("experts.0.gate_proj.weight"), "{e}");
    assert!(
        e.contains("block_sparse_moe.experts.0.w1.weight_packed"),
        "{e}"
    );
}
