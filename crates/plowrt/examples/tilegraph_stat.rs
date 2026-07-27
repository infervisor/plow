//! PX-19 — measure the CO-LOCATION headroom of the emitted device task graph.
//!
//! The devblob already *is* an SM-level task graph: one [`StreamEnt`] per
//! `(op, slice)` work item, per-task dependency edges, `n_cu` streams. This tool
//! answers the only question that decides whether making that graph
//! co-location-aware can pay:
//!
//! > For a producer→consumer edge, how much of the producer's output does ONE
//! > consumer tile read, and how many producer tiles must therefore land on the
//! > consumer's SM for the value to be handed over in SRAM instead of HBM/L2?
//!
//! Two numbers fall out per edge and both are hard limits:
//!
//! * `handoff_KiB` — the producer bytes one consumer tile needs resident. If
//!   this exceeds the dynamic-smem cap (101,376 B on sm_120a) an SRAM handoff is
//!   impossible at ANY schedule.
//! * `max_busy_cu` — how many CUs can still be busy once every producer tile
//!   feeding one consumer tile is pinned to that consumer's CU. `n_cu` = free;
//!   anything below it is the machine you give up to buy the handoff.
//!
//! Usage: `tilegraph_stat <asset-dir-or-model.pkt> [T]`  (default T = largest
//! prefill bucket).
//!
//! No GPU is touched. This reads the blob only.

use std::collections::HashMap;

use packet::dev::TENSOR_NONE16;
use plowrt::asset::devblob::DevBlob;

/// sm_120a `sharedMemPerBlockOptin` — the hard ceiling on a handoff buffer.
const SMEM_CAP: u64 = 101_376;
/// The GEMM output tile the sm_120 bodies walk (`PGM_BM` / `PGM_BN`).
const BM: u64 = 128;
const BN: u64 = 128;

fn op_name(op: u16) -> &'static str {
    use packet::dev::DevOp::*;
    match op {
        x if x == Nop as u16 => "Nop",
        x if x == RmsNorm as u16 => "RmsNorm",
        x if x == HeadNormRope as u16 => "HeadNormRope",
        x if x == Residual as u16 => "Residual",
        x if x == Glu as u16 => "Glu",
        x if x == Embed as u16 => "Embed",
        x if x == SoftCap as u16 => "SoftCap",
        x if x == Gemm as u16 => "Gemm",
        x if x == Gemv as u16 => "Gemv",
        x if x == FlashPrefill as u16 => "FlashPrefill",
        x if x == FlashDecode as u16 => "FlashDecode",
        x if x == FlashMerge as u16 => "FlashMerge",
        x if x == GemmSmall as u16 => "GemmSmall",
        x if x == GemmMed as u16 => "GemmMed",
        x if x == NormResidual as u16 => "NormResidual",
        x if x == AddNorm as u16 => "AddNorm",
        x if x == Argmax as u16 => "Argmax",
        x if x == ArgmaxFin as u16 => "ArgmaxFin",
        x if x == GemvGlu as u16 => "GemvGlu",
        x if x == GemmGlu as u16 => "GemmGlu",
        x if x == GemvQkv as u16 => "GemvQkv",
        x if x == GemvFp8 as u16 => "GemvFp8",
        x if x == GemvGluFp8 as u16 => "GemvGluFp8",
        x if x == QuantFp8 as u16 => "QuantFp8",
        x if x == GemmFp8 as u16 => "GemmFp8",
        x if x == GemmMedFp8 as u16 => "GemmMedFp8",
        x if x == GemmSmallFp8 as u16 => "GemmSmallFp8",
        x if x == GemmGluFp8 as u16 => "GemmGluFp8",
        x if x == HeadNormRopeFp8 as u16 => "HeadNormRopeFp8",
        x if x == FlashPrefillFp8 as u16 => "FlashPrefillFp8",
        x if x == NormResidualNorm as u16 => "NormResidualNorm",
        _ => "other",
    }
}

/// Is this op one of the tiled prefill matmul bodies (`for tile = slice; tile <
/// ntiles; tile += nblk` over `BM x BN` output tiles)?
fn is_tiled_gemm(op: u16) -> bool {
    use packet::dev::DevOp::*;
    matches!(
        op,
        x if x == Gemm as u16
            || x == GemmMed as u16
            || x == GemmSmall as u16
            || x == GemmGlu as u16
            || x == GemmFp8 as u16
            || x == GemmMedFp8 as u16
            || x == GemmSmallFp8 as u16
            || x == GemmGluFp8 as u16
    )
}

/// A GLU body writes `N` fused columns while reading `2N` weight columns, so its
/// *output* feature count is `i[1]`, same as a plain GEMM. Kept explicit because
/// the K-match test below compares a consumer's K against this.
fn out_features(op: u16, i: &[u32; 8]) -> u64 {
    let _ = op;
    i[1] as u64
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: tilegraph_stat <asset-dir|model.pkt> [T]");
    let want_t: Option<u32> = args.next().and_then(|s| s.parse().ok());

    let p = std::path::Path::new(&path);
    let file = if p.is_dir() { p.join("model.pkt") } else { p.to_path_buf() };
    let buf = std::fs::read(&file).expect("read blob");
    let blob = DevBlob::parse(&buf).expect("parse devblob");

    let names: Vec<&str> = blob.tensors.iter().map(|t| t.name.as_str()).collect();
    let tname = |h: u16| -> &str {
        if h == TENSOR_NONE16 { "-" } else { names.get(h as usize).copied().unwrap_or("?") }
    };

    println!("blob          {}", file.display());
    println!("n_cu          {}", blob.n_cu);
    println!("programs      {:?}", blob.progs.iter().map(|p| p.t).collect::<Vec<_>>());

    let prog = match want_t {
        Some(t) => blob.progs.iter().find(|p| p.t == t).expect("no program with that T"),
        None => blob.progs.iter().max_by_key(|p| p.t).expect("no programs"),
    };
    let n_cu = blob.n_cu as u64;

    println!("\n=== program T={}  insts={}  tasks={}  counters={}",
        prog.t, prog.insts.len(), prog.stream.len(), prog.n_counter);

    // ---- 1. the task graph, per op ------------------------------------------
    println!("\n-- ops (a 'task' is one StreamEnt = one (op, slice) work item) --");
    println!("{:>3}  {:<18} {:>6} {:>7} {:>7} {:>7} {:>8} {:>8}  {}",
        "id", "op", "blocks", "M", "N", "K", "tiles", "tiles/wg", "out");
    let mut ntiles_of: Vec<Option<u64>> = vec![None; prog.insts.len()];
    for (id, inst) in prog.insts.iter().enumerate() {
        let (m, n, k) = (inst.i[0] as u64, inst.i[1] as u64, inst.i[2] as u64);
        let tiles = if is_tiled_gemm(inst.op) {
            let t = m.div_ceil(BM) * n.div_ceil(BN);
            ntiles_of[id] = Some(t);
            format!("{t}")
        } else {
            "-".into()
        };
        let per = match ntiles_of[id] {
            Some(t) => format!("{:.2}", t as f64 / inst.blocks.max(1) as f64),
            None => "-".into(),
        };
        println!("{id:>3}  {:<18} {:>6} {:>7} {:>7} {:>7} {:>8} {:>8}  {}",
            op_name(inst.op), inst.blocks, m, n, k, tiles, per, tname(inst.t[0]));
    }

    // ---- 2. tasks per CU, and the counter/gate bill --------------------------
    let mut per_cu = vec![0u64; blob.n_cu as usize];
    for (cu, (&ofs, &len)) in prog.stream_ofs.iter().zip(prog.stream_len.iter()).enumerate() {
        let _ = ofs;
        per_cu[cu] = len as u64;
    }
    let (lo, hi) = (per_cu.iter().copied().min().unwrap_or(0), per_cu.iter().copied().max().unwrap_or(0));
    let fine = prog.stream.iter().filter(|e| e.flags & packet::dev::SE_FINE != 0).count();
    println!("\n-- schedule --");
    println!("tasks/CU        min {lo}  max {hi}");
    println!("SE_FINE tasks   {fine} of {}", prog.stream.len());
    println!("wait edges      {}", prog.waits.len());
    println!("succ edges      {}", prog.succs.len());

    // ---- 3. the co-location question ----------------------------------------
    // Producer of a tensor = the last inst that writes it (t[0]).
    let mut writer: HashMap<u16, usize> = HashMap::new();
    println!("\n-- producer -> consumer edges, and what an SRAM handoff would cost --");
    println!("{:<32} {:>9} {:>12} {:>12} {:>11}  {}",
        "edge (tensor)", "fanin", "handoff KiB", "vs 99 KiB cap", "max busy CU", "kind");
    let mut worst_busy = n_cu;
    let mut any_fits = 0usize;
    let mut edges = 0usize;
    for (id, inst) in prog.insts.iter().enumerate() {
        for slot in 1..8 {
            let h = inst.t[slot];
            if h == TENSOR_NONE16 { continue; }
            let Some(&pid) = writer.get(&h) else { continue };
            let p = &prog.insts[pid];
            edges += 1;

            let pm = p.i[0] as u64;
            let pn = out_features(p.op, &p.i);
            let ck = inst.i[2] as u64;

            // How much of the producer's output does ONE consumer work item read,
            // and how many producer tiles is that?
            let (fanin, bytes, busy, kind);
            if is_tiled_gemm(inst.op) && slot == 1 {
                // The consumer is a tiled matmul and this is its A operand: ONE
                // output tile contracts over the WHOLE K axis, so it needs the
                // producer's entire `BM x K` row block resident. `elem` is 1 for
                // the fp8 (w8a8) bodies, 2 for bf16.
                let elem = if op_name(inst.op).ends_with("Fp8") { 1 } else { 2 };
                fanin = if is_tiled_gemm(p.op) { pn.div_ceil(BN) } else { p.blocks as u64 };
                bytes = BM * ck * elem;
                busy = (inst.i[0] as u64).div_ceil(BM);
                kind = "consumer contracts the WHOLE K axis";
            } else if is_tiled_gemm(p.op) && pm == inst.i[0] as u64 {
                // Row-aligned consumer (norm / quant / headnorm / residual): it
                // reads its own rows only, but still all of the feature axis.
                fanin = pn.div_ceil(BN);
                bytes = BM * pn * 2;
                busy = pm.div_ceil(BM);
                kind = "row-aligned, full feature axis";
            } else {
                fanin = 0;
                bytes = 0;
                busy = n_cu;
                kind = "non-tiled producer (not analysed)";
            }
            if fanin == 0 { continue; }
            let kib = bytes as f64 / 1024.0;
            let over = bytes as f64 / SMEM_CAP as f64;
            if bytes <= SMEM_CAP { any_fits += 1; }
            worst_busy = worst_busy.min(busy);
            println!("{:<32} {fanin:>9} {kib:>12.1} {over:>11.1}x {busy:>11}  {kind}",
                format!("{}->{} ({})", pid, id, tname(h)));
        }
        if inst.t[0] != TENSOR_NONE16 {
            writer.insert(inst.t[0], id);
        }
    }
    println!("\nedges examined         {edges}");
    println!("handoffs under the cap {any_fits}");
    println!("tightest max_busy_CU   {worst_busy} of {n_cu} ({:.1}% of the machine)",
        100.0 * worst_busy as f64 / n_cu as f64);

    // ---- 4. the escape hatch, and why it is worse ---------------------------
    // A handoff that does not fit at BM=128 could be made to fit by SHRINKING the
    // row block. That is not free: a matmul re-reads its whole weight once per
    // M-tile, so weight traffic scales as ceil(M/BM). This prints the largest BM
    // that fits and the weight-traffic multiplier it costs.
    println!("\n-- shrink BM until the handoff fits? weight traffic is ceil(M/BM) x |W| --");
    println!("{:<24} {:>7} {:>7} {:>8} {:>8} {:>12} {:>12}",
        "matmul", "K", "elem", "BM_cap", "BM_free", "W traffic x", "M-tiles");
    for (id, inst) in prog.insts.iter().enumerate() {
        if !is_tiled_gemm(inst.op) { continue; }
        let m = inst.i[0] as u64;
        let k = inst.i[2] as u64;
        let elem = if op_name(inst.op).ends_with("Fp8") { 1 } else { 2 };
        // The arena is a UNION over op bodies, and this body already claims its
        // cp.async staging buffers out of it (PGM_ARENA_BF16 = 30720 bf16 for the
        // plain body, doubled B-buffers for GLU). What is left for a handoff:
        let staging: u64 = if op_name(inst.op).contains("Glu") { 2 * 30720 } else { 30720 };
        let free = SMEM_CAP.saturating_sub(staging.min(SMEM_CAP));
        let bm_cap = (SMEM_CAP / (k * elem)).max(1);
        let bm_free = (free / (k * elem)).max(1);
        let mult = m.div_ceil(bm_free) as f64 / m.div_ceil(BM) as f64;
        println!("{:<24} {k:>7} {elem:>7} {bm_cap:>8} {bm_free:>8} {mult:>11.1}x {:>12}",
            format!("{id} {}", op_name(inst.op)), m.div_ceil(bm_free));
    }
}
