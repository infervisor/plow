//! Run one program sequence of any packet on the CUDA packet runtime:
//! `packet_run PACKET PIPELINE ROLE [--stages N] [--iters K] [--in NAME=FILE]... [--out NAME=FILE]...`
//! NAME is a pipeline tensor role or a packet tensor name. Inputs are raw little-endian bytes,
//! zero-padded to the tensor; outputs are the whole tensor. Runs the first N programs of the
//! sequence `ROLE.0, ROLE.1, ...` K times and prints the median GPU time per program.
//! `--before ROLE2` runs the sequence `ROLE2.*` once first, with the `--before-in NAME=FILE`
//! inputs (e.g. a packet's cache-filling programs).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("packet_run requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::packet_runtime::{load_packet_runtime, PacketAsset, PacketTensor};
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: packet_run PACKET PIPELINE ROLE [--stages N] [--iters K] [--in NAME=FILE]... [--out NAME=FILE]...".into());
    }
    let path = std::path::Path::new(&args[1]);
    let mut loaded = load_packet_runtime(path, "cuda")?;
    let asset = PacketAsset::load(path)?;
    let pipeline = asset.bind(&args[2], loaded.runtime.as_ref())?;
    let rt = &mut loaded.runtime;
    let mut programs = pipeline.program_sequence(&args[3])?;
    let (mut iters, mut inputs, mut outputs) = (1usize, Vec::new(), Vec::new());
    let (mut before, mut before_inputs) = (None, Vec::new());
    let mut it = args[4..].iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--stages" => programs.truncate(value.parse()?),
            "--iters" => iters = value.parse()?,
            "--before" => before = Some(pipeline.program_sequence(value)?),
            "--in" | "--out" | "--before-in" => {
                let (name, file) = value.split_once('=').ok_or(format!("{flag} {value}: expected NAME=FILE"))?;
                let tensor: PacketTensor = match pipeline.tensor(name) {
                    Ok(t) => t,
                    Err(_) => rt.tensor(name).ok_or(format!("no tensor {name}"))?,
                };
                match flag.as_str() {
                    "--in" => inputs.push((tensor, file.to_string())),
                    "--before-in" => before_inputs.push((tensor, file.to_string())),
                    _ => outputs.push((tensor, file.to_string())),
                }
            }
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    let write = |rt: &mut Box<dyn plowrt::exec::packet_runtime::PacketRuntime>, inputs: &[(PacketTensor, String)]| -> Result<(), Box<dyn std::error::Error>> {
        for (tensor, file) in inputs {
            let mut bytes = std::fs::read(file)?;
            if bytes.len() > tensor.bytes {
                return Err(format!("{file}: {} bytes exceed the tensor ({})", bytes.len(), tensor.bytes).into());
            }
            bytes.resize(tensor.bytes, 0);
            rt.write_tensor(*tensor, &bytes)?;
        }
        Ok(())
    };
    if let Some(before) = &before {
        write(rt, &before_inputs)?;
        rt.run_sequence(before)?;
    }
    write(rt, &inputs)?;
    let mut us = vec![Vec::with_capacity(iters); programs.len()];
    for _ in 0..iters {
        for (k, &program) in programs.iter().enumerate() {
            rt.run(program)?;
            us[k].push(rt.last_run_us());
        }
    }
    let mut total = 0.0;
    for (k, t) in us.iter_mut().enumerate() {
        t.sort_by(f64::total_cmp);
        total += t[t.len() / 2];
        println!("stage {k} program {} median_us {:.1}", programs[k], t[t.len() / 2]);
    }
    println!("total_us {total:.1}");
    for (tensor, file) in &outputs {
        let mut bytes = vec![0u8; tensor.bytes];
        rt.read_tensor(*tensor, &mut bytes)?;
        std::fs::write(file, bytes)?;
    }
    Ok(())
}
