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
    let bkt: Vec<u32> = buckets
        .iter()
        .map(|&(t, _)| t)
        .take(buckets.len().saturating_sub(1))
        .collect();
    // THE COVER SWEEP, both policies and both launch prices, on ONE line per
    // length. This is the table that shows WHERE the ragged-tail cliff bites and
    // how wide it is: `pad` is the shipped padding-vs-launch DP, `rag` is the
    // fewest-launch cover `PLOW_RAGGED_CHUNK` uses, and the `LR=` columns are the
    // DP re-run at a corrected launch price (LAUNCH_ROWS understates a GLM
    // launch by ~3.4x: 416 rows charged against ~1400 rows measured).
    //
    // Read it for the finding it encodes: repricing moves almost NOTHING, because
    // the DP is not making a mistake. At 4097 it is choosing between 4224 padded
    // rows in two launches and 8192 padded rows in one, and under the PADDED
    // regime two launches really is cheaper. Only the row shrink, which makes the
    // padding free, changes the answer.
    eprintln!(
        "\nlen      pad-cover                          n  rag-cover            n   LR=1400  LR=4000"
    );
    for &n in &[
        128u32, 1024, 1025, 1152, 2048, 4096, 4097, 4224, 8192, 8193, 12345, 16384, 16385,
    ] {
        let pad = plowrt::exec::amd::plan_chunks_cfg(&bkt, n, 416, false).unwrap_or_default();
        let rag = plowrt::exec::amd::plan_chunks_cfg(&bkt, n, 416, true).unwrap_or_default();
        let lr14 = plowrt::exec::amd::plan_chunks_cfg(&bkt, n, 1400, false).unwrap_or_default();
        let lr40 = plowrt::exec::amd::plan_chunks_cfg(&bkt, n, 4000, false).unwrap_or_default();
        eprintln!(
            "{n:<8} {:<34} {}  {:<20} {}   {:<8} {}",
            format!("{pad:?}"),
            pad.len(),
            format!("{rag:?}"),
            rag.len(),
            lr14.len(),
            lr40.len(),
        );
    }

    let chunks = plowrt::exec::amd::plan_chunks(&bkt, n_prompt).expect("plan");
    let cover: u32 = chunks.iter().sum();
    let segs: usize = chunks
        .iter()
        .map(|c| {
            buckets
                .iter()
                .find(|&&(t, _)| t == *c)
                .map(|&(_, s)| s)
                .unwrap_or(0)
        })
        .sum();
    eprintln!(
        "\nprompt {n_prompt} -> chunks {chunks:?} = {cover} padded rows ({}% pad); \
         {segs} segment launches per rank, {} at TP4 with a drain after each",
        100 * (cover - n_prompt) / cover.max(1),
        segs * 4,
    );
}
