use super::*;

/// One prompt seated on a sequence of slots (`SLOT_REUSE_TEST_ORDER`, default `0,1,0,1,5,0`):
/// prefill, `steps` greedy decode steps, retire. Decode is not batch-invariant across rung widths,
/// and a seating that attaches the previous occupant's prefix recomputes only the tail rows, so
/// seatings are compared within one (rung rows, attached) class: there they must be bit-identical
/// (no state left behind by an earlier occupant). `REUSE` lines carry tokens and logits hashes.
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
    let len: u32 = std::env::var("SLOT_REUSE_TEST_LEN").map_or(1200, |v| v.parse().unwrap());
    let attach = std::env::var("SLOT_REUSE_TEST_ATTACH").map_or(true, |v| v != "0");
    let prompt: Vec<u32> = (0..len).map(|i| 100 + (i * 7919) % 20_000).collect();
    let fnv = |v: &[f32]| {
        v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
    };
    let mut logits = Vec::new();
    let mut runs = Vec::new();
    for &slot in &order {
        e.begin_slot(slot, prompt.len() + steps + 8).unwrap();
        e.set_prefix_attach(slot, attach);
        let mut token = e.prefill_slot(slot, &prompt).unwrap();
        let attached = e.attached_rows(slot);
        e.dump_slot_kv(slot, "prefill", e.pos[slot]);
        e.logits_row(0, &mut logits).unwrap();
        let mut hashes = vec![fnv(&logits)];
        let mut tokens = vec![token];
        let selection = e.select_decode(std::iter::once(slot)).unwrap();
        let rows = e.selected_decode(selection).map_or(e.batch, |r| r.rows);
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
            "REUSE slot={slot} rows={rows} attached={attached} tokens={tokens:?} fnv={}",
            hashes
                .iter()
                .map(|h| format!("{:08x}", h >> 32))
                .collect::<Vec<_>>()
                .join(",")
        );
        runs.push(((rows, attached), slot, tokens, hashes));
    }
    let bad: Vec<usize> = runs
        .iter()
        .filter(|r| {
            let first = runs.iter().find(|f| f.0 == r.0).expect("own class");
            (&first.2, &first.3) != (&r.2, &r.3)
        })
        .map(|r| r.1)
        .collect();
    assert!(
        bad.is_empty(),
        "seatings on slots {bad:?} differ within their class"
    );
}
