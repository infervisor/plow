use super::*;

/// One prompt seated on a sequence of slots (`SLOT_REUSE_TEST_ORDER`, default `0,1,0,1,5,0`):
/// prefill, `steps` greedy decode steps, retire. Every seating must give the same prefill and
/// decode tokens. `REUSE` lines carry each seating's tokens, decode rung rows and logits hashes.
#[test]
#[ignore = "GPU; set SLOT_REUSE_TEST_ASSETS (a batch > 1 packet)"]
fn reseated_slot_decodes_like_a_fresh_one() {
    let assets = PathBuf::from(std::env::var("SLOT_REUSE_TEST_ASSETS").unwrap());
    let order: Vec<usize> = std::env::var("SLOT_REUSE_TEST_ORDER")
        .unwrap_or_else(|_| "0,1,0,1,5,0".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let steps = 4;
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
    assert!(order.iter().all(|&s| s < e.batch));
    let prompt: Vec<u32> = (0..1200u32).map(|i| 100 + (i * 7919) % 20_000).collect();
    let fnv = |v: &[f32]| {
        v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
    };
    let mut logits = Vec::new();
    let mut runs = Vec::new();
    for &slot in &order {
        e.begin_slot(slot, prompt.len() + steps + 8).unwrap();
        let mut token = e.prefill_slot(slot, &prompt).unwrap();
        e.logits_row(0, &mut logits).unwrap();
        let mut hashes = vec![fnv(&logits)];
        let mut tokens = vec![token];
        let rows = e
            .selected_decode(e.select_decode(std::iter::once(slot)).unwrap())
            .map_or(e.batch, |r| r.rows);
        for _ in 0..steps {
            let mut next = Vec::new();
            e.step_slots(&[(slot, token)], &mut next).unwrap();
            e.logits_row(slot, &mut logits).unwrap();
            hashes.push(fnv(&logits));
            token = next[0];
            tokens.push(token);
        }
        e.retire_slot(slot, false);
        eprintln!(
            "REUSE slot={slot} rows={rows} tokens={tokens:?} fnv={}",
            hashes
                .iter()
                .map(|h| format!("{:08x}", h >> 32))
                .collect::<Vec<_>>()
                .join(",")
        );
        runs.push(tokens);
    }
    let bad: Vec<usize> = order
        .iter()
        .zip(&runs)
        .filter(|(_, t)| **t != runs[0])
        .map(|(&s, _)| s)
        .collect();
    assert!(
        bad.is_empty(),
        "seatings on slots {bad:?} differ from the first"
    );
}
