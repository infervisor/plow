//! Print the ops behind each `hetero.json` segment of a blob (index → opcode), to check the
//! sidecar's instruction indices line up with the packet.
#[cfg(feature = "cpu")]
fn main() {
    use packet::dev::DevOp;
    use plowrt::exec::cpu::engine::CpuModel;
    use std::path::PathBuf;
    let blob: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: hetero_dump <model.pkt> <ckpt>")
        .into();
    let ckpt: PathBuf = std::env::args().nth(2).expect("usage").into();
    plowrt::exec::cpu::ffi::init(plowrt::exec::cpu::ffi::Isa::Amx).expect("kernels");
    let model = CpuModel::load(&blob, &ckpt).expect("load");
    let plan: plow_asset::hetero::HeteroPlan =
        serde_json::from_slice(&std::fs::read(blob.with_file_name("hetero.json")).unwrap())
            .unwrap();
    let pp = &plan.programs[0];
    let p = &model.blob.progs[pp.prog as usize];
    println!(
        "prog {} T={} rows gpu/ane/cpu {}/{}/{}; {} insts, max seg {}",
        pp.prog,
        pp.t,
        pp.rows_gpu,
        pp.rows_ane,
        pp.rows_cpu,
        p.insts.len(),
        p.stream.iter().map(|e| e.seg).max().unwrap_or(0)
    );
    for sp in pp.segments.iter().take(3) {
        let ops: Vec<String> = sp
            .cpu_insts
            .iter()
            .map(|&i| {
                let d = &p.insts[i as usize];
                format!(
                    "{i}:{}(M={},N={},K={},t0={},t1={})",
                    DevOp::from_u16(d.op).map(|o| o.c_name()).unwrap_or("?"),
                    d.i[0],
                    d.i[1],
                    d.i[2],
                    model.names[d.t[0] as usize],
                    model.names[d.t[1] as usize]
                )
            })
            .collect();
        println!("seg {} ane {:?}: {}", sp.seg, sp.ane, ops.join(" | "));
    }
    // Which segment does each instruction belong to, for the first 40 instructions?
    let mut seg_of = vec![u16::MAX; p.insts.len()];
    for e in &p.stream {
        seg_of[e.inst as usize] = e.seg;
    }
    let head: Vec<String> = (0..40.min(p.insts.len()))
        .map(|i| {
            format!(
                "{i}:{}@{}",
                DevOp::from_u16(p.insts[i].op)
                    .map(|o| o.c_name().trim_start_matches("PLOW_DOP_"))
                    .unwrap_or("?"),
                seg_of[i]
            )
        })
        .collect();
    println!("{}", head.join(" "));
}
#[cfg(not(feature = "cpu"))]
fn main() {}
