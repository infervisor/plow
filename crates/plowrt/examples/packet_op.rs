//! Run ONE packet opcode on a chosen CUDA object -- the interpreter or a segment role object -- and
//! dump its outputs: the harness for checking an opcode's sm_90a arm against a reference inside a
//! real packet (stream entries, counters, segment launch).
//!
//! `packet_op --object FILE --symbol SYM [--block B] [--arena-symbol NAME] --op NAME [--n-cu N]
//!   [--i a,b,..] [--f x,y] [--t NAME:BYTES[:INFILE]]... [--x NAME:BYTES[:INFILE]]...
//!   [--ptrs NAME=T,T,..]... [--out NAME=FILE]... [--iters K]`
//!
//! `--op` is the `DevOp` variant name (e.g. `GemmFp8Mx`). `--t` declares the tensors in operand
//! order (`t0`, `t1`, ...); `-` as NAME leaves a slot absent. Inputs are raw little-endian bytes,
//! zero-padded; an output is the whole tensor. `--x` declares a tensor that is no operand (e.g. an
//! expert's weights), and `--ptrs` fills a tensor with the u64 device addresses of the named ones
//! (`T+BYTES` for an offset into one, `0` for a null entry) -- a pointer table such as `expert_weight_table`. Prints the median GPU
//! time over `--iters` runs.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("packet_op requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use packet::dev::{DevOp, TENSOR_NONE};
    use packet::devbuild::{Builder, Model};
    use plowrt::exec::gpu::packet_exec::CudaPacketRuntime;
    use plowrt::exec::packet_runtime::PacketRuntime;
    use std::path::PathBuf;
    use std::sync::Arc;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut object, mut symbol, mut block, mut arena) = (None, None, 256u32, "plow_arena_bytes".to_string());
    let (mut op, mut n_cu, mut ints, mut floats, mut iters) = (None, 132u32, Vec::new(), Vec::new(), 1usize);
    let (mut tensors, mut outputs): (Vec<(String, u64, Option<String>)>, Vec<(String, String)>) = (Vec::new(), Vec::new());
    let (mut extra, mut ptrs): (Vec<(String, u64, Option<String>)>, Vec<(String, Vec<String>)>) = (Vec::new(), Vec::new());
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let v = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--object" => object = Some(PathBuf::from(v)),
            "--symbol" => symbol = Some(v.clone()),
            "--block" => block = v.parse()?,
            "--arena-symbol" => arena = v.clone(),
            "--op" => op = Some(DevOp::ALL.iter().copied().find(|o| format!("{o:?}") == *v).ok_or(format!("no DevOp {v}"))?),
            "--n-cu" => n_cu = v.parse()?,
            "--i" => ints = v.split(',').map(str::parse::<u32>).collect::<Result<_, _>>()?,
            "--f" => floats = v.split(',').map(str::parse::<f32>).collect::<Result<_, _>>()?,
            "--iters" => iters = v.parse()?,
            "--t" | "--x" => {
                let mut p = v.splitn(3, ':');
                let name = p.next().unwrap().to_string();
                let bytes = p.next().ok_or(format!("{flag} {v}: NAME:BYTES[:FILE]"))?.parse()?;
                let list = if flag == "--t" { &mut tensors } else { &mut extra };
                list.push((name, bytes, p.next().map(str::to_string)));
            }
            "--ptrs" => {
                let (n, l) = v.split_once('=').ok_or(format!("--ptrs {v}: NAME=T,T,.."))?;
                ptrs.push((n.to_string(), l.split(',').map(str::to_string).collect()));
            }
            "--out" => {
                let (n, f) = v.split_once('=').ok_or(format!("--out {v}: NAME=FILE"))?;
                outputs.push((n.to_string(), f.to_string()));
            }
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    let (object, symbol, op) = (object.ok_or("--object")?, symbol.ok_or("--symbol")?, op.ok_or("--op")?);
    if tensors.len() > 8 || ints.len() > 8 || floats.len() > 2 {
        return Err("at most 8 tensors, 8 ints, 2 floats".into());
    }

    let mut b = Builder::new(n_cu);
    b.force_uniseg();
    let handles: Vec<u32> = tensors.iter().map(|(n, bytes, _)| if n == "-" { TENSOR_NONE } else { b.tensor(n, *bytes) }).collect();
    for (n, bytes, _) in &extra {
        b.tensor(n, *bytes);
    }
    b.emit(op, b.all(), &[], |d| {
        d.t = [TENSOR_NONE; 8];
        d.t[..handles.len()].copy_from_slice(&handles);
        d.i[..ints.len()].copy_from_slice(&ints);
        d.f[..floats.len()].copy_from_slice(&floats);
    });
    let prog = b.finish();
    let model = Model { n_cu, target: 0, tensors: prog.tensors.clone(), progs: vec![prog], kv_row_insts: vec![], prog_t: vec![1], gen: vec![] };
    let pkt = std::env::temp_dir().join(format!("packet_op_{}.pkt", std::process::id()));
    std::fs::write(&pkt, model.to_blob_v6(&[]))?;

    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut rt = CudaPacketRuntime::load_object(be, &pkt, &object, &symbol, block, &arena)?;
    std::fs::remove_file(&pkt).ok();
    for (name, _, file) in tensors.iter().chain(&extra) {
        let (Some(file), false) = (file, name == "-") else { continue };
        let t = rt.tensor(name).ok_or(format!("no tensor {name}"))?;
        let mut bytes = std::fs::read(file)?;
        if bytes.len() > t.bytes {
            return Err(format!("{file}: {} bytes exceed tensor {name} ({})", bytes.len(), t.bytes).into());
        }
        bytes.resize(t.bytes, 0);
        rt.write_tensor(t, &bytes)?;
    }
    for (name, list) in &ptrs {
        let t = rt.tensor(name).ok_or(format!("no tensor {name}"))?;
        let mut bytes = Vec::with_capacity(list.len() * 8);
        for e in list {
            let (tn, off) = e.split_once('+').map_or(Ok((e.as_str(), 0u64)), |(n, o)| o.parse().map(|o| (n, o)))?;
            let a = if tn == "0" { 0 } else { rt.device_ptr(rt.tensor(tn).ok_or(format!("no tensor {tn}"))?).ok_or(format!("{tn}: no address"))? + off };
            bytes.extend_from_slice(&a.to_le_bytes());
        }
        if bytes.len() > t.bytes {
            return Err(format!("--ptrs {name}: {} entries exceed the tensor", list.len()).into());
        }
        bytes.resize(t.bytes, 0);
        rt.write_tensor(t, &bytes)?;
    }
    let mut us = Vec::with_capacity(iters);
    for _ in 0..iters {
        rt.run(0)?;
        us.push(rt.last_run_us());
    }
    us.sort_by(f64::total_cmp);
    println!("median_us {:.1}", us[us.len() / 2]);
    for (name, file) in &outputs {
        let t = rt.tensor(name).ok_or(format!("no tensor {name}"))?;
        let mut bytes = vec![0u8; t.bytes];
        rt.read_tensor(t, &mut bytes)?;
        std::fs::write(file, bytes)?;
    }
    Ok(())
}
