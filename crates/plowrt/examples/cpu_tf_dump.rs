//! Teacher-forced prefill logprobs: for each `k < --k`, prefill a reference case's prompt plus the
//! first `k` reference continuation tokens from scratch and print the last row's top-20 logprobs
//! as one JSON line `{"k":k,"top":[[token,logprob],..]}` (score against the reference offline).
//!
//! `cargo run --release --no-default-features --features cpu --example cpu_tf_dump -- \
//!     <model.pkt> <ckpt> <ref.json> <case id> [--k 16]`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let reference: PathBuf = args.next().expect("usage").into();
    let id = args.next().expect("usage");
    let mut kmax = 16usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--k" => kmax = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let case = j["cases"].as_array().unwrap().iter().find(|c| c["id"] == id.as_str()).expect("case");
    let ids = |key: &str| -> Vec<u32> {
        case[key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    };
    let (prompt, cont) = (ids("prompt_ids"), ids("cont"));
    let mut eng = CpuEngine::load(&blob, &ckpt, &CpuEngineOpts::default()).expect("load");
    for k in 0..kmax.min(cont.len()) {
        let q = [&prompt[..], &cont[..k]].concat();
        eng.prefill_slot(0, &q).expect("prefill");
        let mut lg = Vec::new();
        assert!(eng.logits_row(0, &mut lg));
        let m = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = m + lg.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
        let mut top: Vec<(usize, f64)> = lg.iter().enumerate().map(|(t, &x)| (t, x as f64 - lse)).collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<String> = top[..20].iter().map(|(t, l)| format!("[{t},{l:.6}]")).collect();
        println!("{{\"k\":{k},\"top\":[{}]}}", top.join(","));
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}
