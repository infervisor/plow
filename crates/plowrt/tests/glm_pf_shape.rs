//! Host-side shape of a GLM prefill packet: buckets, chunk cover, segments.
//!
//! No GPU. Reads the packet named by `PLOW_PF_PKT` and prints what the prefill
//! driver will do with a `PLOW_PF_TOKENS`-token prompt, so the launch/barrier
//! count in the §TTFT breakdown can be predicted before a lease is spent.
#![cfg(feature = "hsa")]

#[test]
fn glm_prefill_shape() {
    let Ok(path) = std::env::var("PLOW_PF_PKT") else {
        eprintln!("PLOW_PF_PKT unset — skipping");
        return;
    };
    let n_prompt: u32 = std::env::var("PLOW_PF_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1030);

    let raw = std::fs::read(&path).expect("read packet");
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).expect("parse");
    eprintln!(
        "packet {path}: {} programs, tp={:?}",
        blob.progs.len(),
        blob.tp.as_ref().map(|t| t.n_gpu)
    );
    let mut buckets = Vec::new();
    for (i, p) in blob.progs.iter().enumerate() {
        let seg = plowrt::exec::amd::derive_segments(p).expect("segments");
        let n4 = seg.iter().filter(|&&c| c == 4).count();
        eprintln!(
            "  prog {i}: T={:<6} insts={:<6} stream={:<7} segments={:<5} (class-4: {n4})",
            p.t,
            p.insts.len(),
            p.stream.len(),
            seg.len(),
        );
        buckets.push((p.t, seg.len()));
    }
    // The decode program is the LAST one; prefill buckets precede it.
    let bkt: Vec<u32> = buckets.iter().map(|&(t, _)| t).take(buckets.len().saturating_sub(1)).collect();
    let chunks = plowrt::exec::amd::plan_chunks(&bkt, n_prompt).expect("plan");
    let cover: u32 = chunks.iter().sum();
    let segs: usize = chunks
        .iter()
        .map(|c| buckets.iter().find(|&&(t, _)| t == *c).map(|&(_, s)| s).unwrap_or(0))
        .sum();
    eprintln!(
        "\nprompt {n_prompt} -> chunks {chunks:?} = {cover} padded rows ({}% pad); \
         {segs} segment launches per rank, {} at TP4 with a drain after each",
        100 * (cover - n_prompt) / cover.max(1),
        segs * 4,
    );
}
