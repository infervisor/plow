//! Ask whether a bundle can serve CPU prefill heads, and say why not.
//!
//! `cargo run --release --features cpu --example head_probe -- <twin.pkt> <ckpt> [cores]`
//!
//! The packet-level refusals run before the checkpoint is touched, so a model
//! whose caches are not addressable as row ranges answers in milliseconds
//! rather than after a 22 GiB load.

#[cfg(feature = "cpu")]
fn main() {
    use plowrt::exec::cpu::engine::CpuEngineOpts;
    use plowrt::exec::cpu::ffi::Isa;
    use plowrt::serve::head::{host_supports_heads, HeadPool};
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let twin: PathBuf = args.next().expect("usage: head_probe <twin.pkt> <ckpt>").into();
    let ckpt: PathBuf = args.next().expect("usage: head_probe <twin.pkt> <ckpt>").into();
    let n_cores: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let cores: Vec<u32> = (0..n_cores as u32).collect();

    match host_supports_heads(Isa::Avx512, cores.len()) {
        Ok(()) => println!("host gate: OK ({} cores reserved)", cores.len()),
        Err(why) => {
            println!("host gate: REFUSED — {why}");
            return;
        }
    }

    let opts = CpuEngineOpts {
        threads: n_cores,
        ..Default::default()
    };
    match HeadPool::load(&twin, &ckpt, &opts, &cores) {
        Ok(pool) => println!(
            "twin ACCEPTED: slots={} kv_contract={}",
            pool.slots(),
            pool.kv_contract()
        ),
        Err(e) => println!("twin REFUSED — {e}"),
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}
