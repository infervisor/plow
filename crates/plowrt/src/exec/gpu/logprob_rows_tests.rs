use super::*;

/// `plow_logprob_stats_rows` returns, per request, exactly what one `plow_logprob_stats`
/// launch returns for that row.
#[test]
#[ignore = "requires H100 assets with a sampler object (LOGPROB_ROWS_TEST_ASSETS)"]
fn batched_logprob_stats_match_the_per_row_kernel() {
    let assets = PathBuf::from(std::env::var("LOGPROB_ROWS_TEST_ASSETS").unwrap());
    let mut e = GpuEngine::load(
        Arc::new(CudaBackend::new(0).unwrap()),
        &assets,
        &assets.join("checkpoint"),
    )
    .unwrap();
    let slots = e.batch.min(8);
    let mut feeds = Vec::new();
    for b in 0..slots {
        let prompt: Vec<u32> = (0..48u32).map(|i| 100 + (i * 7 + 131 * b as u32) % 1000).collect();
        e.begin_slot(b, 64).unwrap();
        feeds.push((b, e.prefill_slot(b, &prompt).unwrap()));
    }
    let mut toks = Vec::new();
    e.step_slots(&feeds, &mut toks).unwrap();
    let reqs: Vec<(u32, u32, u32)> =
        (0..slots).map(|b| (b as u32, toks[b], [0u32, 1, 5, 20][b % 4])).collect();
    let mut rows = Vec::new();
    assert!(e.logprob_stats_rows(&reqs, &mut rows).unwrap());
    assert_eq!(rows.len(), slots);
    for (&(row, tok, k), got) in reqs.iter().zip(&rows) {
        let mut one = [0f32; LOGPROB_STATS_MAX];
        assert!(e.logprob_stats(row as usize, tok, k, &mut one).unwrap());
        let n = 3 + 2 * k as usize;
        assert_eq!(
            got[..n].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            one[..n].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "row {row}"
        );
    }
}
