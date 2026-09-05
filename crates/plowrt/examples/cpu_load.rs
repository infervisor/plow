//! Load a devblob + checkpoint into the CPU engine's host tensor table and
//! report what was bound — the loader's end-to-end check on a real model.
//!
//! `cargo run --release --features cpu --example cpu_load -- <model.pkt> <checkpoint-dir>`

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::{engine::CpuModel, ffi};
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage: cpu_load <model.pkt> <checkpoint-dir>").into();
    let ckpt: PathBuf = args.next().expect("usage: cpu_load <model.pkt> <checkpoint-dir>").into();

    let isa = ffi::init(ffi::Isa::Amx).expect("cpu kernel init");
    println!("kernel tier: {isa:?}");
    let m = CpuModel::load(&blob, &ckpt).expect("load");
    println!(
        "tensors={} weights={:.2} GiB load={:.0} ms n_cu={} programs={} dec_ix={} kvrow_sites={}",
        m.names.len(),
        m.weight_bytes as f64 / (1u64 << 30) as f64,
        m.load_ms,
        m.blob.n_cu,
        m.blob.progs.len(),
        m.dec_ix,
        m.kvrow.len()
    );
    for (i, p) in m.blob.progs.iter().enumerate() {
        println!(
            "  prog[{i}] T={:<5} insts={:<4} stream={:<7} gq={:<7} counters={}",
            p.t,
            p.insts.len(),
            p.stream.len(),
            p.gq_stream.len(),
            p.n_counter
        );
    }
    println!(
        "wellknown: ids={:?} pos={:?} kvlen={:?} logits={:?}",
        m.wk.ids, m.wk.pos, m.wk.kvlen, m.wk.logits
    );
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}
