use super::*;

fn state_row(slot: u32) -> Vec<u8> {
    let mut data = vec![0; 2296];
    // The qualified control-sample ABI1 fixes State.last=224, stopped=236.
    data[224..228].copy_from_slice(&(100 + slot).to_le_bytes());
    data[236..240].copy_from_slice(&u32::from(slot % 3 == 0).to_le_bytes());
    data
}

fn admit(executor: &mut CudaHandoff, slot: u32) -> RowKey {
    let admission = executor.prepare_admission("physical", slot).unwrap();
    executor
        .admit(
            admission,
            &[HandoffBind {
                name: "states",
                bytes: &state_row(slot),
            }],
        )
        .unwrap()
}

fn fixture(root: &Path) -> (HandoffPlan, HandoffProofCatalog) {
    let capacity = 8;
    let mut builder = Builder::new(1);
    builder.force_uniseg();
    let mut views = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    let mut offset = 0u64;
    for (name, dtype, width, initialize) in [
        ("states", PipelineDType::U8, 2296, true),
        ("slots", PipelineDType::U32, 1, false),
        ("pcm", PipelineDType::F32, 7, false),
        ("out", PipelineDType::U8, 24, false),
    ] {
        let stride = u64::from(width) * dtype.bytes();
        let bytes = u64::from(capacity) * stride;
        if initialize {
            let data = (0..capacity).flat_map(state_row).collect();
            builder.tensor_init(name, data);
        } else {
            builder.tensor(name, bytes);
        }
        offset = offset.next_multiple_of(256);
        views.insert(
            name.into(),
            HandoffView {
                component: "aux".into(),
                tensor: name.into(),
                offset,
                bytes,
                dtype,
                rows: capacity,
                row_width: width,
                row_stride: stride,
                row_map: if initialize {
                    HandoffRowMap::Gather {
                        map: "slots".into(),
                        domain: "physical".into(),
                        unique: false,
                    }
                } else {
                    HandoffRowMap::Contiguous {
                        domain: "packed".into(),
                    }
                },
                lifetime: if initialize {
                    HandoffLifetime::Persistent
                } else {
                    HandoffLifetime::Transient { bank_stride: 0 }
                },
                packet_initialize: initialize,
                immutable: name == "slots",
            },
        );
        tensors.insert(
            name.into(),
            PipelineTensor {
                name: name.into(),
                dtype,
                shape: vec![u64::from(capacity), u64::from(width)],
            },
        );
        offset += bytes;
    }
    builder.emit(DevOp::Nop, vec![0], &[], |_| {});
    let model = Model {
        n_cu: 1,
        target: 0,
        tensors: builder.tensors(),
        progs: vec![builder.finish()],
        prog_t: vec![1],
        kv_row_insts: vec![],
        gen: vec![],
    };
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: "aux".into(),
            driver: "forward.v1".into(),
            programs: BTreeMap::from([("forward".into(), 0)]),
            tensors,
            parameters: BTreeMap::new(),
            strings: BTreeMap::new(),
        }],
    };
    std::fs::write(
        root.join("aux.pkt"),
        model.to_blob_v6(&[SectionData {
            kind: SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&metadata).unwrap(),
        }]),
    )
    .unwrap();
    let live = HandoffRows::Live {
        extent: "live".into(),
    };
    let accesses: Vec<_> = [
        ("states", HandoffAccessMode::Read),
        ("slots", HandoffAccessMode::Read),
        ("pcm", HandoffAccessMode::Read),
        ("out", HandoffAccessMode::Write),
    ]
    .into_iter()
    .map(|(view, mode)| HandoffAccess {
        view: view.into(),
        mode,
        rows: live.clone(),
        mask: None,
    })
    .collect();
    let mut parameters: Vec<_> = accesses
        .iter()
        .map(|a| HandoffParameter::Tensor {
            view: a.view.clone(),
            dtype: views[&a.view].dtype,
            access: a.mode,
            alignment: if a.view == "states" { 8 } else { 4 },
        })
        .collect();
    parameters.extend(
        [
            HandoffExpr::LiveRows {
                extent: "live".into(),
            },
            HandoffExpr::Constant {
                value: u64::from(capacity),
            },
            HandoffExpr::Constant { value: 7 },
            HandoffExpr::Constant { value: 24 },
        ]
        .into_iter()
        .map(|value| HandoffParameter::Scalar {
            dtype: HandoffScalarType::U32,
            value,
        }),
    );
    let entry = HandoffExecutable::Kernel {
        object_role: "control".into(),
        symbol: "handoff_pack_pcm16".into(),
        abi_version: 1,
        capability: "plow_control_sample_abi".into(),
        capability_value: 1,
        parameters,
        grid: [
            HandoffExpr::LiveRows {
                extent: "live".into(),
            },
            HandoffExpr::Constant { value: 1 },
            HandoffExpr::Constant { value: 1 },
        ],
        block: [256, 1, 1],
        shared_bytes: 0,
    };
    let program = reference("aux");
    let readback = HandoffReadback {
        producer: program.clone(),
        view: "out".into(),
        extent: "live".into(),
        row_stride: 24,
        fields: [
            ("stop", 0, PipelineDType::U32, 1),
            ("cb0", 4, PipelineDType::U32, 1),
            ("pcm", 8, PipelineDType::I16, 7),
        ]
        .into_iter()
        .map(|(name, offset, dtype, elements)| HandoffReadbackField {
            name: name.into(),
            offset,
            dtype,
            elements,
        })
        .collect(),
        sample_lengths: None,
    };
    let arena = HandoffArena {
        bytes: offset,
        alignment: 256,
        banks: 1,
    };
    let extents = BTreeMap::from([("live".into(), capacity)]);
    let native = HandoffProofCatalog {
        arena: Some(arena.clone()),
        extents: extents.clone(),
        views: views.clone(),
        programs: BTreeMap::from([(
            program.clone(),
            HandoffProgramProof {
                entry: entry.clone(),
                accesses: accesses.clone(),
                copies: vec![],
                private_transients_banked: true,
                persistent_stop_safe: true,
            },
        )]),
        readbacks: BTreeMap::from([(program.clone(), readback.clone())]),
    };
    let plan = HandoffPlan {
        arena,
        extents,
        views,
        components: BTreeMap::from([(
            "aux".into(),
            HandoffComponent {
                packet: "aux.pkt".into(),
                driver: "forward.v1".into(),
                objects: BTreeMap::from([("control".into(), "control.cubin".into())]),
                programs: BTreeMap::from([("copy".into(), HandoffProgram { entry, accesses })]),
            },
        )]),
        schedule: vec![program.clone()],
        edges: vec![HandoffEdge {
            producer: HandoffEndpoint {
                program,
                view: "out".into(),
            },
            consumer: HandoffConsumer::Host,
            class: HandoffClass::Host,
            dtype: PipelineDType::U8,
            rows: live.clone(),
            order: HandoffOrder::SameStream,
            generation_delta: 0,
        }],
        bind_writes: ["slots", "pcm"]
            .into_iter()
            .map(|name| HandoffBindWrite {
                name: name.into(),
                view: name.into(),
                rows: if name == "slots" {
                    HandoffRows::Fixed { rows: capacity }
                } else {
                    live.clone()
                },
                order: HandoffOrder::SameStream,
                phase: HandoffBindPhase::Tick,
            })
            .chain(std::iter::once(HandoffBindWrite {
                name: "states".into(),
                view: "states".into(),
                rows: HandoffRows::Fixed { rows: capacity },
                order: HandoffOrder::SameStream,
                phase: HandoffBindPhase::Admission,
            }))
            .collect(),
        readback,
        execution: HandoffExecution::Synchronous,
    };
    (
        serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap(),
        native,
    )
}

#[test]
fn handoff_auxiliary_pcm_fixture_has_native_typed_proof() {
    let keep = std::env::var_os("HANDOFF_TEST_ASSETS");
    let root = keep
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("plow-handoff-aux-{}", std::process::id()))
        });
    std::fs::create_dir_all(&root).unwrap();
    let (plan, native) = fixture(&root);
    plan.prove(&native).unwrap();
    if keep.is_none() {
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "CUDA; HANDOFF_TEST_ASSETS must contain qualified control.cubin and speech interpreter; use gpuq"]
fn handoff_auxiliary_pcm_executes_typed_arguments_and_live_readback() {
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, native) = fixture(&root);
    let proven = plan.prove(&native).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let mut executor = CudaHandoff::load(backend.clone(), &root, &proven, 2).unwrap();
    let slots = [7u32, 2, 5, 0, 6, 1, 4, 3];
    let pcm = [
        f32::NEG_INFINITY,
        -1.0,
        -0.5,
        f32::NAN,
        0.5,
        1.0,
        f32::INFINITY,
    ];
    let mut held = None;
    let mut keys: Vec<Option<RowKey>> = (0..8).map(|_| None).collect();
    for live in [0, 1, 3, 8, 1] {
        for &slot in &slots[..live as usize] {
            if keys[slot as usize].is_none() {
                keys[slot as usize] = Some(admit(&mut executor, slot));
            }
        }
        let selected: Vec<_> = slots[..live as usize]
            .iter()
            .map(|&s| keys[s as usize].as_ref().unwrap())
            .collect();
        let maps: Vec<_> = slots[..live as usize]
            .iter()
            .copied()
            .chain(std::iter::repeat_n(u32::MAX, 8 - live as usize))
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let audio: Vec<_> = (0..live)
            .flat_map(|_| pcm.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let result = executor
            .run_mapped(
                &BTreeMap::from([("live".into(), live)]),
                &BTreeMap::new(),
                &[
                    HandoffBind {
                        name: "slots",
                        bytes: &maps,
                    },
                    HandoffBind {
                        name: "pcm",
                        bytes: &audio,
                    },
                ],
                &selected,
                &[HandoffSelection {
                    map: "slots",
                    rows: &selected,
                }],
            )
            .unwrap();
        let HandoffRun::Complete(output) = result else {
            panic!("auxiliary PCM completion")
        };
        let mut expected = Vec::new();
        for &slot in &slots[..live as usize] {
            let stopped = slot % 3 == 0;
            expected.extend(u32::from(stopped).to_le_bytes());
            expected.extend((100 + slot).to_le_bytes());
            for x in pcm {
                let sample = if stopped || x.is_nan() {
                    0
                } else {
                    (x.clamp(-1.0, 1.0) * 32767.0).round() as i16
                };
                expected.extend(sample.to_le_bytes());
            }
            expected.extend([0, 0]);
        }
        assert_eq!(output.host_slice().unwrap(), expected);
        if live == 1 && held.is_none() {
            held = Some((output.clone(), expected));
        }
        if let Some((old, bytes)) = &held {
            assert_eq!(old.host_slice().unwrap(), bytes);
        }
    }
    let old = keys[7].take().unwrap();
    executor.release(&old).unwrap();
    let next = admit(&mut executor, 7);
    let padded: Vec<_> = std::iter::once(7u32)
        .chain(std::iter::repeat_n(u32::MAX, 7))
        .flat_map(u32::to_le_bytes)
        .collect();
    let audio: Vec<_> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
    let binds = [
        HandoffBind {
            name: "slots",
            bytes: &padded,
        },
        HandoffBind {
            name: "pcm",
            bytes: &audio,
        },
    ];
    let one = BTreeMap::from([("live".into(), 1)]);
    assert!(executor
        .run_mapped(
            &one,
            &BTreeMap::new(),
            &binds,
            &[&old],
            &[HandoffSelection {
                map: "slots",
                rows: &[&old]
            }]
        )
        .is_err());
    assert!(matches!(
        executor
            .run_mapped(
                &one,
                &BTreeMap::new(),
                &binds,
                &[&next],
                &[HandoffSelection {
                    map: "slots",
                    rows: &[&next]
                }]
            )
            .unwrap(),
        HandoffRun::Complete(_)
    ));
    let mut wrong_plan = plan.clone();
    let mut wrong_native = native.clone();
    for entry in [
        &mut wrong_plan
            .components
            .get_mut("aux")
            .unwrap()
            .programs
            .get_mut("copy")
            .unwrap()
            .entry,
        &mut wrong_native
            .programs
            .get_mut(&reference("aux"))
            .unwrap()
            .entry,
    ] {
        let HandoffExecutable::Kernel {
            capability_value, ..
        } = entry
        else {
            unreachable!()
        };
        *capability_value = 2;
    }
    let modules = backend.live_modules();
    assert!(CudaHandoff::load(
        backend.clone(),
        &root,
        &wrong_plan.prove(&wrong_native).unwrap(),
        2
    )
    .is_err());
    assert_eq!(backend.live_modules(), modules);
    use crate::tts::handoff::{SemanticFrame, SpeechHandoff, SpeechHandoffRun};
    let mut executor = CudaHandoff::load(backend, &root, &proven, 2).unwrap();
    let keys: Vec<_> = [7u32, 0, 2]
        .into_iter()
        .map(|slot| admit(&mut executor, slot))
        .collect();
    let selected: Vec<_> = keys.iter().collect();
    let selections = [HandoffSelection {
        map: "slots",
        rows: &selected,
    }];
    let mut caller = SpeechHandoff::new(executor, "stop", "cb0", "pcm", 7).unwrap();
    let semantic = SemanticFrame {
        base: 1000,
        stop_class: 200,
        stop_token: 99,
    };
    let semantics = [semantic; 3];
    let live = BTreeMap::from([("live".into(), 3)]);
    let maps: Vec<_> = [7u32, 0, 2]
        .into_iter()
        .chain(std::iter::repeat_n(u32::MAX, 5))
        .flat_map(u32::to_le_bytes)
        .collect();
    let audio: Vec<_> = (0..3)
        .flat_map(|_| pcm.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    let binds = [
        HandoffBind {
            name: "slots",
            bytes: &maps,
        },
        HandoffBind {
            name: "pcm",
            bytes: &audio,
        },
    ];
    assert!(caller
        .run(
            &semantics[..2],
            &live,
            &BTreeMap::new(),
            &binds,
            &selected,
            &selections
        )
        .is_err());
    let SpeechHandoffRun::Complete(first) = caller
        .run(
            &semantics,
            &live,
            &BTreeMap::new(),
            &binds,
            &selected,
            &selections,
        )
        .unwrap()
    else {
        panic!("speech completion")
    };
    assert_eq!(
        first.iter().map(|f| f.0).collect::<Vec<_>>(),
        [1107, 99, 1102]
    );
    assert!(first[1].1.is_none());
    let SpeechHandoffRun::Complete(second) = caller
        .run(
            &semantics,
            &live,
            &BTreeMap::new(),
            &binds,
            &selected,
            &selections,
        )
        .unwrap()
    else {
        panic!("second speech completion")
    };
    assert!(matches!(
        caller
            .run(
                &semantics,
                &live,
                &BTreeMap::new(),
                &binds,
                &selected,
                &selections
            )
            .unwrap(),
        SpeechHandoffRun::DeliveryBusy
    ));
    drop(first);
    let SpeechHandoffRun::Complete(third) = caller
        .run(
            &semantics,
            &live,
            &BTreeMap::new(),
            &binds,
            &selected,
            &selections,
        )
        .unwrap()
    else {
        panic!("reused speech completion")
    };
    drop(caller);
    for frames in [second, third] {
        let frame = frames[0].1.as_ref().unwrap();
        assert!(matches!(frame, crate::serve::stream::SpeechFrame::Pcm(_)));
        assert_eq!(
            &**frame.pcm().unwrap(),
            &[-32767, -32767, -16384, 0, 16384, 32767, 32767]
        );
    }
}
