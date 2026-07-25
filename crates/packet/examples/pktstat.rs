//! Decode a scheduled `.pkt` (packet::Program) and print instruction/counter stats.
//! Throwaway analysis helper for the egg-vs-devblob comparison.
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: pktstat <file.pkt> [lo hi]");
    let lo: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let bytes = std::fs::read(&path).expect("read pkt");
    let prog = packet::Program::decode(&bytes).expect("decode");
    let hi: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(prog.insts.len());

    let n = prog.insts.len();
    let total_wait: usize = prog.insts.iter().map(|i| i.wait.len()).sum();
    let total_succ: usize = prog.insts.iter().map(|i| i.succ.len()).sum();
    println!("file            {path}");
    println!("bytes           {}", bytes.len());
    println!("insts           {n}");
    println!("counters        {}", prog.counters.len());
    println!("wait-edges      {total_wait}");
    println!("succ-edges      {total_succ}");

    // Counter scope histogram (0 intra-SM, 1 intra-GPU, 2 cross-unit).
    let mut scope: BTreeMap<u8, usize> = BTreeMap::new();
    for c in &prog.counters {
        *scope.entry(c.scope).or_default() += 1;
    }
    println!("counter-scopes  {scope:?}  (0=intraSM 1=intraGPU 2=crossUnit)");

    // Op-kind histogram (by Body variant) + resource-lane histogram.
    let kind = |b: &packet::Body| -> &'static str {
        use packet::Body::*;
        match b {
            Dma { .. } => "Dma",
            Rdma { .. } => "Rdma",
            Gemm { .. } => "Gemm",
            _ => "Other",
        }
    };
    let mut ops: BTreeMap<&str, usize> = BTreeMap::new();
    let mut res: BTreeMap<String, usize> = BTreeMap::new();
    for i in &prog.insts {
        *ops.entry(kind(&i.body)).or_default() += 1;
        *res.entry(format!("{:?}", i.resource)).or_default() += 1;
    }
    println!("op-kinds        {ops:?}");
    println!("resource-lanes  {res:?}");

    // Slice (e.g. the single block's task range from blocks.json).
    if lo != 0 || hi != n {
        let slice = &prog.insts[lo.min(n)..hi.min(n)];
        let sw: usize = slice.iter().map(|i| i.wait.len()).sum();
        let ss: usize = slice.iter().map(|i| i.succ.len()).sum();
        let mut sops: BTreeMap<&str, usize> = BTreeMap::new();
        for i in slice {
            *sops.entry(kind(&i.body)).or_default() += 1;
        }
        println!("--- slice [{lo}..{hi}] ---");
        println!("slice insts      {}", slice.len());
        println!("slice wait-edges {sw}");
        println!("slice succ-edges {ss}");
        println!("slice opcodes    {sops:?}");
    }
}
