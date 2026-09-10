use super::*;

#[test]
#[ignore = "requires packed-prefix assets whose full-context slots exceed available HBM"]
fn packed_admission_retries_after_retirement_and_preserves_waiter_priority() {
    let assets = PathBuf::from(std::env::var("PACKED_ADMISSION_TEST_ASSETS").unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
    assert!(e.pf_batch_enabled() && e.vmm_prefix_enabled() && e.batch >= 3);
    let total = e.max_ctx;
    let mut ready = Vec::new();
    let mut waiting = Vec::new();
    for slot in 0..e.batch {
        if e.admit_packed_slot(slot, &[1, 2], total).unwrap().is_some() {
            ready.push(slot);
        } else {
            waiting.push(slot);
        }
    }
    assert!(!ready.is_empty() && !waiting.is_empty());
    let epoch = e.kv_admission_epoch;
    let created = e.vmm.as_ref().unwrap().kv.stats().blocks_created;
    for _ in 0..16 {
        for &slot in &waiting {
            assert!(e.admit_packed_slot(slot, &[1, 2], total).unwrap().is_none());
        }
    }
    assert_eq!(e.kv_admission_epoch, epoch);
    assert_eq!(e.vmm.as_ref().unwrap().kv.stats().blocks_created, created);
    let freed = ready[0];
    e.retire_slot(freed, false);
    assert!(e
        .admit_packed_slot(freed, &[1, 2], total)
        .unwrap()
        .is_none());
    assert!(e
        .admit_packed_slot(waiting[0], &[1, 2], total)
        .unwrap()
        .is_some());

    // Cancel admitted and waiting requests, then execute a one-token prompt
    // in the highest slot to exercise every inactive-row mapping backstop.
    for slot in 0..e.batch {
        e.retire_slot(slot, false);
    }
    let slot = e.batch - 1;
    assert_eq!(e.admit_packed_slot(slot, &[1], 2).unwrap(), Some(0));
    let mut tokens = Vec::new();
    e.step_slots(&[(slot, 1)], &mut tokens).unwrap();
    assert_eq!(tokens.len(), 1);
    assert!((tokens[0] as usize) < e.vocab);
    eprintln!("PASS packed admission: {} admitted, {} waiting, retirement, priority, cancellation, widest-rung recovery", ready.len(), waiting.len());
}

#[test]
fn packed_runtime_tables_are_excluded_from_both_weight_consumers() {
    let m = plow_asset::packed_prefill::Manifest {
        version: 1,
        max_request_rows: None,
        slot: 4,
        request: 5,
        maps: vec![plow_asset::packed_prefill::Map {
            original: 6,
            slots: 7,
        }],
        programs: vec![],
    };
    for (index, name) in [
        (4, "pf.request.slot"),
        (5, "pf.request.table"),
        (7, "pf.request.maps.6"),
    ] {
        assert!(!is_checkpoint_tensor(index, name, Some(&m)));
        assert!(
            is_checkpoint_tensor(index, name, None),
            "undeclared runtime-looking name must not bypass weight lookup"
        );
        assert!(
            is_checkpoint_tensor(index + 10, name, Some(&m)),
            "wrong handle"
        );
    }
    for (index, name) in [
        (4, "model.layers.0.weight"),
        (5, "pf.request.other"),
        (7, "pf.request.maps.9"),
        (6, "pf.request.maps.6"),
    ] {
        assert!(
            is_checkpoint_tensor(index, name, Some(&m)),
            "unbound or mismatched declaration"
        );
    }
    assert!(is_checkpoint_tensor(0, "model.layers.0.weight", Some(&m)));
    assert!(!is_checkpoint_tensor(1, "act.x", Some(&m)));
}

#[test]
#[ignore = "GPU Gemma direct-KV block qualification; root-owned launch only"]
fn packed_segmented_block_matches_serialized() -> Result<()> {
    assert_eq!(std::env::var("TEST_PACKED_PREFILL_GPU").as_deref(), Ok("1"));
    let config = crate::config::RuntimeConfig::get();
    assert!(!config.pf_batch);
    assert_ne!(config.nv_vmm_prefix(), Some(true));
    let assets = std::path::PathBuf::from(std::env::var("TEST_PACKED_PREFILL_ASSETS").unwrap());
    let bytes = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = DevBlob::parse(&bytes).unwrap();
    let live = crate::memory::vmm::LiveKvLayout::manifest(&blob, &bytes)
        .unwrap()
        .unwrap();
    let live_requested = config.nv_vmm_live() || live.caches.iter().any(|cache| cache.window == 0);
    let block: serde_json::Value =
        serde_json::from_slice(&std::fs::read(assets.join("block.json")).unwrap()).unwrap();
    let hidden = block["hidden"].as_u64().unwrap() as usize;
    assert_eq!(block["arch"], "gemma_dense");
    assert_eq!(block["inputs"][0]["name"], "act.x");
    assert_eq!(block["outputs"][0]["name"], "act.x");
    let be = Arc::new(CudaBackend::new(0)?);
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint"))?;
    assert!(e.packed_prefill.is_some() && e.pf_batch.is_some() && e.batch >= 16);
    assert_eq!(
        e.vmm.as_ref().is_some_and(|v| !v.kv.prefix_reuse()),
        live_requested
    );
    assert!(e.prefill.iter().all(|b| b.seg_class.len() > 1));
    let tensor_rows = e.tensor_bytes("act.x").unwrap() as usize / 2 / hidden;
    let input = |slot: usize, row: usize, col: usize| {
        (((slot * 101 + row * 17 + col * 13) % 251) as f32 - 125.0) / 128.0
    };
    let stage = |e: &mut GpuEngine, spans: &[(usize, usize, usize)]| {
        let mut x = vec![0.0; tensor_rows * hidden];
        let mut offset = 0;
        for &(slot, start, len) in spans {
            for row in start..start + len {
                for col in 0..hidden {
                    x[offset * hidden + col] = input(slot, row, col);
                }
                offset += 1;
            }
        }
        e.upload_activation("act.x", &x).unwrap();
    };
    let output = |e: &GpuEngine, rows: usize| {
        let mut raw = vec![0; rows * hidden * 2];
        e.read_tensor("act.x", &mut raw).unwrap();
        for x in raw.chunks_exact(2) {
            assert!(f32::from_bits(u32::from(u16::from_le_bytes([x[0], x[1]])) << 16).is_finite());
        }
        raw
    };
    let kv = |e: &GpuEngine, spans: &[(usize, usize)]| {
        let mut all = Vec::new();
        for c in &live.caches {
            let elem = if c.scales.is_some() { 1 } else { 2 };
            let tensors = c
                .pair
                .into_iter()
                .map(|h| (h, c.hd as usize * elem))
                .chain(c.scales.into_iter().flatten().map(|h| (h, 4)));
            for (handle, row_bytes) in tensors {
                for &(slot, len) in spans {
                    for head in 0..c.heads as usize {
                        assert!(len <= c.stride as usize);
                        let offset = ((slot * c.heads as usize + head)
                            * c.stride as usize
                            * row_bytes) as u64;
                        let mut raw = vec![0; len * row_bytes];
                        e.be.download(&e.devp[handle as usize], offset, &mut raw)
                            .unwrap();
                        all.extend(raw);
                    }
                }
            }
        }
        all
    };
    let lens = [31usize, 63];
    let prompts: Vec<Vec<u32>> = lens.iter().map(|&len| vec![100; len]).collect();
    for (slots, idle_slot) in [([0usize, 3], 15usize), ([3, 15], 0)] {
        let mut reference: Option<Vec<u8>> = None;
        for arm in 0..3 {
            e.begin_slot(idle_slot, 256).unwrap();
            stage(&mut e, &[(idle_slot, 0, 17)]);
            e.prefill_slot(idle_slot, &vec![100; 17]).unwrap();
            let idle = kv(&e, &[(idle_slot, 17)]);
            for &slot in &slots {
                e.begin_slot(slot, 256).unwrap();
            }
            let mut got = Vec::new();
            if arm == 0 {
                for i in 0..2 {
                    stage(&mut e, &[(slots[i], 0, lens[i])]);
                    e.prefill_slot(slots[i], &prompts[i]).unwrap();
                    got.extend(output(&e, lens[i]));
                }
            } else {
                stage(&mut e, &[(slots[0], 0, lens[0]), (slots[1], 0, lens[1])]);
                let requests: Vec<_> = (0..2)
                    .map(|i| PfBatchReq {
                        slot: slots[i],
                        prompt: &prompts[i],
                        c0: 0,
                        len: lens[i],
                    })
                    .collect();
                e.prefill_batched(&requests).unwrap();
                got = output(&e, lens.iter().sum());
                let before = kv(&e, &[(slots[0], lens[0]), (slots[1], lens[1])]);
                let duplicate_prompt = vec![100; lens[0] + 1];
                let duplicate = [
                    PfBatchReq {
                        slot: slots[0],
                        prompt: &duplicate_prompt,
                        c0: lens[0],
                        len: 1,
                    },
                    PfBatchReq {
                        slot: slots[0],
                        prompt: &duplicate_prompt,
                        c0: lens[0],
                        len: 1,
                    },
                ];
                assert!(e.prefill_batched(&duplicate).is_err());
                assert_eq!(before, kv(&e, &[(slots[0], lens[0]), (slots[1], lens[1])]));
            }
            assert_eq!(
                idle,
                kv(&e, &[(idle_slot, 17)]),
                "request isolation: slot{idle_slot} idle KV"
            );
            got.extend(kv(&e, &[(slots[0], lens[0]), (slots[1], lens[1])]));
            for i in 0..2 {
                let continuation = vec![100; lens[i] + 1];
                stage(&mut e, &[(slots[i], lens[i], 1)]);
                e.prefill_slot(slots[i], &continuation).unwrap();
                got.extend(output(&e, 1));
            }
            got.extend(kv(&e, &[(slots[0], lens[0] + 1), (slots[1], lens[1] + 1)]));
            stage(&mut e, &[(idle_slot, 17, 1)]);
            e.prefill_slot(idle_slot, &vec![100; 18]).unwrap();
            got.extend(output(&e, 1));
            got.extend(kv(&e, &[(idle_slot, 18)]));
            assert_eq!(e.pos[slots[0]], 32);
            assert_eq!(e.pos[slots[1]], 64);
            if let Some(expected) = &reference {
                assert_eq!(got.len(), expected.len());
                assert!(
                    got.iter().zip(expected).all(|(a, b)| a == b),
                    "slots {slots:?} arm {arm}: first differing byte {:?}",
                    got.iter().zip(expected).position(|(a, b)| a != b)
                );
            } else {
                reference = Some(got);
            }
        }
    }
    eprintln!("packed segmented block PASS: physical slots0/3 and3/15,94 real+34 pad rows,2 repeats,full activations/KV and continuation exact");
    Ok(())
}

#[test]
fn no_patch_sites_produce_an_empty_upload_range() {
    for n_inst in [0, 1, 23] {
        let insts = vec![DevInst64::default(); n_inst];
        let range = prefill_patch_range(std::iter::empty());
        assert_eq!(range, 0..0);
        assert!(pod_bytes(&insts[range]).is_empty());
    }
}

#[test]
fn patch_upload_covers_only_the_first_through_last_site() {
    let insts: Vec<_> = (0..23)
        .map(|i| DevInst64 {
            i: [i; 8],
            ..Default::default()
        })
        .collect();
    for (sites, expected) in [
        (vec![0], 0..1),
        (vec![22], 22..23),
        (vec![8, 2, 20, 8, 4], 2..21),
    ] {
        let range = prefill_patch_range(sites.into_iter());
        assert_eq!(range, expected);
        let bytes = pod_bytes(&insts[range.clone()]);
        assert_eq!(bytes.len(), range.len() * 64);
        assert_eq!(bytes, &pod_bytes(&insts)[range.start * 64..range.end * 64]);
    }
}
