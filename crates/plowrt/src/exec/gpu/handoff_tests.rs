use super::super::handoff_provider::{HandoffProvider, ProviderTensor};
use super::*;
use packet::dev::DevOp;
use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
use plow_asset::packet_pipeline::*;

#[path = "handoff_aux_tests.rs"]
mod auxiliary;

struct ExistingPacketProvider {
    inner: super::super::packet_exec::PacketHandoffProvider,
    facts: HandoffProofCatalog,
    bad_address: bool,
    fail_ready: bool,
    retirements: Vec<bool>,
    enqueues: Arc<std::sync::atomic::AtomicUsize>,
}

// The wrapper only injects mismatches and counts submissions; the native owner does the work.
unsafe impl HandoffProvider for ExistingPacketProvider {
    fn backend(&self) -> &Arc<CudaBackend> {
        self.inner.backend()
    }
    fn facts(&self) -> &HandoffProofCatalog {
        &self.facts
    }
    fn tensor(&mut self, name: &str) -> Result<ProviderTensor> {
        let mut tensor = self.inner.tensor(name)?;
        tensor.address += if self.bad_address { 4 } else { 0 };
        Ok(tensor)
    }
    fn resolve_packet(&mut self, reference: &HandoffProgramRef) -> Result<usize> {
        self.inner.resolve_packet(reference)
    }
    fn resolve_kernel(&mut self, reference: &HandoffProgramRef) -> Result<KernelFn> {
        self.inner.resolve_kernel(reference)
    }
    fn ready(&mut self) -> Result<()> {
        if self.fail_ready {
            return Err(reject("fixture readiness refusal"));
        }
        self.inner.runtime_mut()?;
        self.inner.ready()?;
        assert!(self.inner.runtime_mut().is_err());
        Ok(())
    }
    fn retired(&mut self, succeeded: bool) {
        self.retirements.push(succeeded);
        self.inner.retired(succeeded);
    }
    fn quiescent(&self) -> bool {
        self.inner.quiescent()
    }
    unsafe fn enqueue_packet(&mut self, handle: usize, stream: &CudaStream) -> Result<()> {
        self.enqueues
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: the native owner and executor retain all resources through retirement.
        unsafe { self.inner.enqueue_packet(handle, stream) }
    }
}

fn existing_providers(
    root: &Path,
    arena: &Arc<SharedHandoffArena>,
    facts: &HandoffProofCatalog,
    plan: &HandoffPlan,
    enqueues: Arc<std::sync::atomic::AtomicUsize>,
    bad_address: bool,
    bad_type: bool,
) -> OwnedHandoffProviders {
    let mut providers = OwnedHandoffProviders::new();
    for (id, component) in &plan.components {
        let backend = Arc::new(CudaBackend::new(0).unwrap());
        assert!(backend.shares_context(arena.backend()));
        let views = facts
            .views
            .values()
            .filter(|v| &v.component == id)
            .map(|v| (v.tensor.clone(), (v.offset, v.bytes, v.packet_initialize)))
            .collect();
        let runtime = CudaPacketRuntime::load_with_views(
            backend,
            &root.join(&component.packet),
            Some(SharedPacketTensors {
                backend: arena.backend().clone(),
                memory: arena.memory().clone(),
                views,
            }),
        )
        .unwrap();
        let native = HandoffProofCatalog {
            arena: facts.arena.clone(),
            extents: facts.extents.clone(),
            views: facts
                .views
                .iter()
                .filter(|(_, v)| &v.component == id)
                .map(|(n, v)| (n.clone(), v.clone()))
                .collect(),
            programs: facts
                .programs
                .iter()
                .filter(|(r, _)| &r.component == id)
                .map(|(r, p)| (r.clone(), p.clone()))
                .collect(),
            readbacks: facts
                .readbacks
                .iter()
                .filter(|(r, _)| &r.component == id)
                .map(|(r, p)| (r.clone(), p.clone()))
                .collect(),
        };
        // SAFETY: the fixture lowering constructed these native facts before JSON parsing.
        let inner = unsafe {
            super::super::packet_exec::PacketHandoffProvider::from_native(
                runtime,
                id,
                "interpreter",
                native.clone(),
            )
        }
        .unwrap();
        let mut reported = native;
        if bad_type {
            if let Some(view) = reported.views.get_mut("input") {
                view.dtype = PipelineDType::U32;
            }
        }
        providers.insert(
            id.clone(),
            Box::new(ExistingPacketProvider {
                inner,
                facts: reported,
                bad_address,
                fail_ready: false,
                retirements: Vec::new(),
                enqueues: enqueues.clone(),
            }) as Box<dyn HandoffProvider>,
        );
    }
    arena.backend().synchronize().unwrap();
    providers
}

#[test]
fn handoff_failed_retirement_retains_all_resources_and_preserves_driver_faults() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let fault = |code, fatal| RuntimeError::DeviceFault {
        info: crate::error::DeviceErrorInfo {
            operation: "fixture".into(),
            code,
            name: "fixture".into(),
            fatal,
        },
    };
    for original_fatal in [false, true] {
        for drain_mode in [0, 1, 2] {
            let drops = Arc::new(AtomicUsize::new(0));
            let original = fault(1, original_fatal);
            let drain = match drain_mode {
                0 => Ok(()),
                1 => Err(fault(2, true)),
                _ => Err(reject("unproven drain")),
            };
            let error = retire_failed(Resource(drops.clone()), original, drain);
            assert_eq!(drops.load(Ordering::Relaxed), usize::from(drain_mode == 0));
            assert_eq!(
                error.device_code(),
                Some(if drain_mode == 1 && !original_fatal {
                    2
                } else {
                    1
                })
            );
        }
    }
}

fn program() -> HandoffProgramRef {
    HandoffProgramRef {
        component: "consumer".into(),
        program: "copy".into(),
    }
}

fn reference(component: &str) -> HandoffProgramRef {
    HandoffProgramRef {
        component: component.into(),
        program: "copy".into(),
    }
}

fn access(view: &str, mode: HandoffAccessMode, rows: u32) -> HandoffAccess {
    HandoffAccess {
        view: view.into(),
        mode,
        rows: HandoffRows::Fixed { rows },
        mask: None,
    }
}

fn fixture(root: &Path, rows: u32, admission: bool) -> (HandoffPlan, HandoffProofCatalog) {
    fixture_with_gather(root, rows, admission, None)
}

fn fixture_with_gather(
    root: &Path,
    rows: u32,
    admission: bool,
    gather: Option<(&str, u32)>,
) -> (HandoffPlan, HandoffProofCatalog) {
    let bytes = u64::from(rows) * 12;
    let stride = bytes.next_multiple_of(256);
    let consumer_rows = gather
        .filter(|(component, _)| *component == "consumer")
        .map_or(rows, |(_, n)| n);
    for component in ["producer", "consumer"] {
        let mut builder = Builder::new(1);
        builder.force_uniseg();
        let values = if gather.is_some_and(|(component, _)| component == "producer")
            && component == "producer"
        {
            (0..rows * 3)
                .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
                .collect()
        } else {
            vec![0x5a; bytes as usize]
        };
        let input = builder.tensor_init("input", values);
        let output = builder.tensor("output", bytes);
        if let Some((_, count)) = gather.filter(|(target, _)| *target == component) {
            let map = builder.tensor("map", u64::from(count) * 4);
            builder.emit(DevOp::GatherRowsF32, vec![0], &[], |d| {
                d.t[0] = output;
                d.t[1] = input;
                d.t[2] = map;
                d.i[..8].copy_from_slice(&[count, 3, rows, rows, 1, 0, 0, 0]);
            });
        } else {
            builder.emit(DevOp::CopyColsF32, vec![0], &[], |d| {
                d.t[0] = output;
                d.t[1] = input;
                d.i[..7].copy_from_slice(&[1, rows, 3, 3, 0, 3, 0]);
            });
        }
        let tensors = builder.tensors();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors,
            progs: vec![builder.finish()],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let pipeline = PacketPipeline {
            name: component.into(),
            driver: "forward.v1".into(),
            programs: BTreeMap::from([("forward".into(), 0)]),
            tensors: BTreeMap::from([
                (
                    "input".into(),
                    PipelineTensor {
                        name: "input".into(),
                        dtype: PipelineDType::F32,
                        shape: vec![rows as u64, 3],
                    },
                ),
                (
                    "output".into(),
                    PipelineTensor {
                        name: "output".into(),
                        dtype: PipelineDType::F32,
                        shape: vec![rows as u64, 3],
                    },
                ),
            ]),
            parameters: BTreeMap::new(),
            strings: BTreeMap::new(),
        };
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![pipeline],
        };
        std::fs::write(
            root.join(format!("{component}.pkt")),
            model.to_blob_v6(&[SectionData {
                kind: SECT_METADATA,
                name: SECTION.into(),
                data: serde_json::to_vec(&metadata).unwrap(),
            }]),
        )
        .unwrap();
    }
    let view = |component: &str, tensor: &str, offset, dtype, width, initialize| HandoffView {
        component: component.into(),
        tensor: tensor.into(),
        offset,
        bytes,
        dtype,
        rows,
        row_width: width,
        row_stride: 12,
        row_map: HandoffRowMap::Contiguous {
            domain: "packed".into(),
        },
        lifetime: HandoffLifetime::Transient { bank_stride: 0 },
        packet_initialize: initialize,
        immutable: false,
    };
    let mut plan = HandoffPlan {
        arena: HandoffArena {
            bytes: stride * 3,
            alignment: 256,
            banks: 1,
        },
        extents: BTreeMap::from([("live".into(), consumer_rows)]),
        components: [
            (
                "producer",
                vec![
                    access("input", HandoffAccessMode::Read, rows),
                    access("produced", HandoffAccessMode::Write, rows),
                ],
            ),
            (
                "consumer",
                vec![
                    access("aliased", HandoffAccessMode::Read, consumer_rows),
                    access("out", HandoffAccessMode::Write, consumer_rows),
                ],
            ),
        ]
        .into_iter()
        .map(|(name, accesses)| {
            (
                name.into(),
                HandoffComponent {
                    packet: format!("{name}.pkt"),
                    driver: "forward.v1".into(),
                    objects: BTreeMap::new(),
                    programs: BTreeMap::from([(
                        "copy".into(),
                        HandoffProgram {
                            entry: HandoffExecutable::Packet {
                                section: "packet".into(),
                                program: 0,
                            },
                            accesses,
                        },
                    )]),
                },
            )
        })
        .collect(),
        views: BTreeMap::from([
            (
                "input".into(),
                view("producer", "input", stride, PipelineDType::F32, 3, false),
            ),
            (
                "produced".into(),
                view("producer", "output", 0, PipelineDType::F32, 3, true),
            ),
            (
                "aliased".into(),
                view("consumer", "input", 0, PipelineDType::F32, 3, false),
            ),
            (
                "out".into(),
                view(
                    "consumer",
                    "output",
                    stride * 2,
                    PipelineDType::U8,
                    12,
                    true,
                ),
            ),
        ]),
        schedule: vec![reference("producer"), reference("consumer")],
        edges: vec![
            HandoffEdge {
                producer: HandoffEndpoint {
                    program: reference("producer"),
                    view: "produced".into(),
                },
                consumer: HandoffConsumer::Program {
                    endpoint: HandoffEndpoint {
                        program: reference("consumer"),
                        view: "aliased".into(),
                    },
                },
                class: HandoffClass::Alias,
                dtype: PipelineDType::F32,
                rows: HandoffRows::Fixed {
                    rows: consumer_rows,
                },
                order: HandoffOrder::Event {
                    name: "copied".into(),
                },
                generation_delta: 0,
            },
            HandoffEdge {
                producer: HandoffEndpoint {
                    program: reference("consumer"),
                    view: "out".into(),
                },
                consumer: HandoffConsumer::Host,
                class: HandoffClass::Host,
                dtype: PipelineDType::U8,
                rows: HandoffRows::Live {
                    extent: "live".into(),
                },
                order: HandoffOrder::SameStream,
                generation_delta: 0,
            },
        ],
        bind_writes: vec![HandoffBindWrite {
            name: "input".into(),
            view: "input".into(),
            rows: HandoffRows::Fixed { rows },
            order: HandoffOrder::SameStream,
            phase: HandoffBindPhase::Tick,
        }],
        readback: HandoffReadback {
            producer: program(),
            view: "out".into(),
            extent: "live".into(),
            row_stride: 12,
            fields: vec![HandoffReadbackField {
                name: "values".into(),
                offset: 0,
                dtype: PipelineDType::F32,
                elements: 3,
            }],
            sample_lengths: None,
        },
        execution: HandoffExecution::Synchronous,
    };
    if admission {
        let input = plan.views.get_mut("input").unwrap();
        input.packet_initialize = true;
        input.lifetime = HandoffLifetime::Persistent;
        plan.bind_writes[0].phase = HandoffBindPhase::Admission;
    }
    if let Some((component, count)) = gather {
        let input = plan
            .views
            .get_mut(if component == "producer" {
                "input"
            } else {
                "aliased"
            })
            .unwrap();
        input.row_map = HandoffRowMap::Gather {
            map: "map".into(),
            domain: "physical".into(),
            unique: false,
        };
        if component == "producer" {
            input.packet_initialize = true;
        } else {
            plan.views.get_mut("produced").unwrap().row_map = HandoffRowMap::Contiguous {
                domain: "physical".into(),
            };
        }
        plan.arena.bytes = stride * 4;
        plan.views.insert(
            "map".into(),
            HandoffView {
                component: component.into(),
                tensor: "map".into(),
                offset: stride * 3,
                bytes: u64::from(count) * 4,
                dtype: PipelineDType::U32,
                rows: count,
                row_width: 1,
                row_stride: 4,
                row_map: HandoffRowMap::Contiguous {
                    domain: "packed".into(),
                },
                lifetime: HandoffLifetime::Transient { bank_stride: 0 },
                packet_initialize: false,
                immutable: true,
            },
        );
        if component == "producer" {
            plan.bind_writes[0].name = "map".into();
            plan.bind_writes[0].view = "map".into();
        } else {
            plan.bind_writes.push(HandoffBindWrite {
                name: "map".into(),
                view: "map".into(),
                rows: HandoffRows::Fixed { rows: count },
                order: HandoffOrder::SameStream,
                phase: HandoffBindPhase::Tick,
            });
        }
        plan.components
            .get_mut(component)
            .unwrap()
            .programs
            .get_mut("copy")
            .unwrap()
            .accesses
            .push(access("map", HandoffAccessMode::Read, count));
    }
    // The copy fixture emitter supplies facts before any metadata is deserialized.
    let catalog = HandoffProofCatalog {
        arena: Some(plan.arena.clone()),
        extents: plan.extents.clone(),
        views: plan.views.clone(),
        programs: [
            (
                "producer",
                if gather.is_some_and(|(component, _)| component == "producer") {
                    vec![
                        access("input", HandoffAccessMode::Read, rows),
                        access("produced", HandoffAccessMode::Write, rows),
                        access("map", HandoffAccessMode::Read, rows),
                    ]
                } else {
                    vec![
                        access("input", HandoffAccessMode::Read, rows),
                        access("produced", HandoffAccessMode::Write, rows),
                    ]
                },
            ),
            ("consumer", {
                let mut accesses = vec![
                    access("aliased", HandoffAccessMode::Read, consumer_rows),
                    access("out", HandoffAccessMode::Write, consumer_rows),
                ];
                if gather.is_some_and(|(component, _)| component == "consumer") {
                    accesses.push(access("map", HandoffAccessMode::Read, consumer_rows));
                }
                accesses
            }),
        ]
        .into_iter()
        .map(|(component, accesses)| {
            (
                reference(component),
                HandoffProgramProof {
                    entry: HandoffExecutable::Packet {
                        section: "packet".into(),
                        program: 0,
                    },
                    accesses,
                    copies: Vec::new(),
                    private_transients_banked: false,
                    persistent_stop_safe: false,
                },
            )
        })
        .collect(),
        readbacks: BTreeMap::from([(program(), plan.readback.clone())]),
    };
    let declared = serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap();
    (declared, catalog)
}

#[test]
fn handoff_fixture_has_a_provable_alias_and_packed_live_readback() {
    let keep = std::env::var_os("HANDOFF_TEST_ASSETS");
    let root = keep
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("plow-handoff-fixture-{}", std::process::id()))
        });
    std::fs::create_dir_all(&root).unwrap();
    let (mut plan, catalog) = fixture(&root, 64, false);
    plan.validate().unwrap();
    plan.prove(&catalog).unwrap();
    let mut duplicate = plan.bind_writes[0].clone();
    duplicate.name = "second_input".into();
    plan.bind_writes.push(duplicate);
    assert!(CudaHandoff::supported(&plan, 2).is_err());
    let (gather, native) = fixture_with_gather(&root, 8, false, Some(("producer", 8)));
    gather.prove(&native).unwrap();
    CudaHandoff::supported(&gather, 2).unwrap();
    let (gather, native) = fixture_with_gather(&root, 8, false, Some(("consumer", 2)));
    gather.prove(&native).unwrap();
    CudaHandoff::supported(&gather, 2).unwrap();
    if keep.is_none() {
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "CUDA; set HANDOFF_TEST_ASSETS to scratch with the speech object; use gpuq"]
fn handoff_gather_uploads_match_ordered_live_generations() {
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, native) = fixture_with_gather(&root, 8, false, Some(("producer", 8)));
    let proven = plan.prove(&native).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let mut executor = CudaHandoff::load(backend, &root, &proven, 2).unwrap();
    let mut keys: Vec<_> = (0..8)
        .map(|slot| {
            let admission = executor.prepare_admission("physical", slot).unwrap();
            executor.admit(admission, &[]).unwrap()
        })
        .collect();
    for iteration in 0..12 {
        let selected: Vec<_> = (0..8).map(|i| &keys[(7 - i + iteration) % 8]).collect();
        let map: Vec<_> = selected
            .iter()
            .flat_map(|key| key.slot().to_le_bytes())
            .collect();
        let live_keys: Vec<_> = keys.iter().collect();
        let live = BTreeMap::from([("live".into(), 3)]);
        let binds = [HandoffBind {
            name: "map",
            bytes: &map,
        }];
        assert!(executor
            .run(&live, &BTreeMap::new(), &binds, &live_keys)
            .is_err());
        let mut wrong = selected.clone();
        wrong.swap(0, 1);
        assert!(executor
            .run_mapped(
                &live,
                &BTreeMap::new(),
                &binds,
                &live_keys,
                &[HandoffSelection {
                    map: "map",
                    rows: &wrong
                }]
            )
            .is_err());
        let output = executor
            .run_mapped(
                &live,
                &BTreeMap::new(),
                &binds,
                &live_keys,
                &[HandoffSelection {
                    map: "map",
                    rows: &selected,
                }],
            )
            .unwrap();
        let HandoffRun::Complete(output) = output else {
            panic!("gather completion")
        };
        let expected: Vec<_> = selected[..3]
            .iter()
            .flat_map(|key| {
                (0..3)
                    .flat_map(move |column| ((key.slot() * 3 + column) as f32 * 0.25).to_le_bytes())
            })
            .collect();
        assert_eq!(output.host_slice().unwrap(), expected);
        drop(selected);
        drop(live_keys);
        let old = keys[3].clone();
        executor.release(&old).unwrap();
        let admission = executor.prepare_admission("physical", 3).unwrap();
        keys[3] = executor.admit(admission, &[]).unwrap();
        let stale: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| if i == 3 { &old } else { k })
            .collect();
        assert!(executor
            .run_mapped(&live, &BTreeMap::new(), &binds, &stale, &[])
            .is_err());
        assert_eq!(output.host_slice().unwrap(), expected);
    }
}

#[test]
#[ignore = "CUDA; set HANDOFF_TEST_ASSETS to scratch with the speech object; use gpuq"]
fn handoff_fixed_physical_output_aliases_a_sparse_gather_with_generation_checks() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, native) = fixture_with_gather(&root, 8, false, Some(("consumer", 2)));
    let proven = plan.prove(&native).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let arena = SharedHandoffArena::allocate(backend, &proven).unwrap();
    let enqueues = Arc::new(AtomicUsize::new(0));
    let providers = existing_providers(
        &root,
        &arena,
        &native,
        &plan,
        enqueues.clone(),
        false,
        false,
    );
    let mut executor = CudaHandoff::bind_owned(&proven, arena, providers, 2).unwrap();
    let blocked = executor.prepare_admission("physical", 0).unwrap();
    executor.resources.as_mut().unwrap().programs.providers[0]
        .ready()
        .unwrap();
    assert!(executor.admit(blocked, &[]).is_err());
    executor.resources.as_mut().unwrap().programs.providers[0].retired(true);
    for name in ["producer", "consumer"] {
        let owner = executor
            .provider_mut::<ExistingPacketProvider>(name)
            .unwrap();
        owner.retirements.clear();
        owner.fail_ready = true;
    }
    let mut keys: Vec<_> = [7, 1]
        .into_iter()
        .map(|slot| {
            let admission = executor.prepare_admission("physical", slot).unwrap();
            executor.admit(admission, &[]).unwrap()
        })
        .collect();
    assert_eq!(enqueues.load(Ordering::Relaxed), 0);
    for name in ["producer", "consumer"] {
        let owner = executor
            .provider_mut::<ExistingPacketProvider>(name)
            .unwrap();
        assert!(owner.retirements.is_empty());
        owner.fail_ready = false;
    }
    for iteration in 0..12 {
        let selected = [&keys[0], &keys[1]];
        let values: Vec<_> = (0..24)
            .map(|n| (iteration * 24 + n) as f32 * 0.25)
            .collect();
        let input: Vec<_> = values.iter().flat_map(|n| n.to_le_bytes()).collect();
        let map: Vec<_> = selected
            .iter()
            .flat_map(|key| key.slot().to_le_bytes())
            .collect();
        let binds = [
            HandoffBind {
                name: "input",
                bytes: &input,
            },
            HandoffBind {
                name: "map",
                bytes: &map,
            },
        ];
        let live = BTreeMap::from([("live".into(), 2)]);
        let wrong = [&keys[1], &keys[0]];
        assert!(executor
            .run_mapped(
                &live,
                &BTreeMap::new(),
                &binds,
                &selected,
                &[HandoffSelection {
                    map: "map",
                    rows: &wrong
                }]
            )
            .is_err());
        let HandoffRun::Complete(output) = executor
            .run_mapped(
                &live,
                &BTreeMap::new(),
                &binds,
                &selected,
                &[HandoffSelection {
                    map: "map",
                    rows: &selected,
                }],
            )
            .unwrap()
        else {
            panic!("sparse alias completion")
        };
        let expected: Vec<_> = selected
            .iter()
            .flat_map(|key| {
                values[key.slot() as usize * 3..key.slot() as usize * 3 + 3]
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
            })
            .collect();
        assert_eq!(output.host_slice().unwrap(), expected);
        let old = keys[0].clone();
        executor.release(&old).unwrap();
        let admission = executor.prepare_admission("physical", old.slot()).unwrap();
        keys[0] = executor.admit(admission, &[]).unwrap();
        let live_keys = [&keys[0], &keys[1]];
        let stale = [&old, &keys[1]];
        assert!(executor
            .run_mapped(
                &live,
                &BTreeMap::new(),
                &binds,
                &live_keys,
                &[HandoffSelection {
                    map: "map",
                    rows: &stale
                }]
            )
            .is_err());
        assert_eq!(output.host_slice().unwrap(), expected);
    }
    assert_eq!(enqueues.load(Ordering::Relaxed), 24);
}

#[test]
#[ignore = "CUDA; set HANDOFF_TEST_ASSETS to a scratch directory containing the speech object; use gpuq"]
fn handoff_empty_and_backpressured_ticks_propagate_a_quarantined_backend() {
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, catalog) = fixture(&root, 1, false);
    let proven = plan.prove(&catalog).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let mut executor = CudaHandoff::load(backend.clone(), &root, &proven, 1).unwrap();
    let live = BTreeMap::from([("live".into(), 0)]);
    let input = [0u8; 12];
    let binds = [HandoffBind {
        name: "input",
        bytes: &input,
    }];
    let HandoffRun::Complete(output) = executor.run(&live, &BTreeMap::new(), &binds, &[]).unwrap()
    else {
        panic!("empty completion expected")
    };
    assert!(output.host_slice().unwrap().is_empty());
    let active = BTreeMap::from([("live".into(), 1)]);
    let HandoffRun::Complete(held) = executor
        .run(&active, &BTreeMap::new(), &binds, &[])
        .unwrap()
    else {
        panic!("live completion expected")
    };
    assert!(matches!(
        executor.run(&active, &BTreeMap::new(), &binds, &[]),
        Ok(HandoffRun::DeliveryBusy)
    ));
    let fault = backend.quarantine_unretired(&RuntimeError::Rejected(
        "injected unretired handoff work".into(),
    ));
    assert!(fault.device_fault().is_some());
    for rows in [&live, &active] {
        match executor.run(rows, &BTreeMap::new(), &binds, &[]) {
            Err(error) => {
                let actual = error.device_fault().expect("quarantine fault");
                let expected = fault.device_fault().unwrap();
                assert_eq!(actual.operation, expected.operation);
                assert_eq!(actual.code, expected.code);
                assert!(actual.fatal);
            }
            Ok(_) => panic!("empty or backpressured tick hid a quarantined backend"),
        }
    }
    assert_eq!(held.host_slice().unwrap(), &input[..]);
}

#[test]
#[ignore = "CUDA; set HANDOFF_TEST_ASSETS to a scratch directory containing the speech object; use gpuq"]
fn handoff_shared_views_preserve_producer_and_read_only_the_live_prefix() {
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, catalog) = fixture(&root, 128, false);
    let proven = plan.prove(&catalog).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let mut executor = CudaHandoff::load(backend, &root, &proven, 2).unwrap();
    let mut timings = Vec::new();
    for iteration in 0..24u32 {
        let values: Vec<f32> = (0..128 * 3)
            .map(|i| (i + iteration * 11) as f32 * 0.25)
            .collect();
        let input: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        for live in [1, 32, 64, 96, 128] {
            let started = std::time::Instant::now();
            let actual = executor
                .run(
                    &BTreeMap::from([("live".into(), live)]),
                    &BTreeMap::new(),
                    &[HandoffBind {
                        name: "input",
                        bytes: &input,
                    }],
                    &[],
                )
                .unwrap();
            let HandoffRun::Complete(actual) = actual else {
                panic!("readback pool unexpectedly busy")
            };
            assert_eq!(actual.host_slice().unwrap(), &input[..live as usize * 12]);
            if iteration >= 4 {
                timings.push((live, started.elapsed().as_secs_f64() * 1e6));
            }
        }
        assert!(executor
            .run(
                &BTreeMap::from([("live".into(), 129)]),
                &BTreeMap::new(),
                &[HandoffBind {
                    name: "input",
                    bytes: &input
                }],
                &[],
            )
            .is_err());
        assert!(executor
            .run(
                &BTreeMap::from([("live".into(), 1)]),
                &BTreeMap::new(),
                &[HandoffBind {
                    name: "input",
                    bytes: &input[..4]
                }],
                &[],
            )
            .is_err());
    }
    for live in [1, 32, 64, 96, 128] {
        let mut samples: Vec<f64> = timings
            .iter()
            .filter(|(n, _)| *n == live)
            .map(|(_, us)| *us)
            .collect();
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "handoff_single_block rows={live} launches=2 readback_bytes={} roundtrip_us={:.3}",
            live * 12,
            samples[samples.len() / 2]
        );
    }
    let staged = executor
        .resources
        .as_ref()
        .unwrap()
        .input
        .as_slice()
        .to_vec();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = executor.resources.as_mut().unwrap();
        r.inflight = true;
        // SAFETY: the executor owns and retains this pinned source through its Drop drain.
        unsafe {
            r.backend
                .memcpy_htod_async(r.arena.base, &r.input.as_slice()[..12], &r.stream)
                .unwrap();
        }
        panic!("submission unwind fixture");
    }));
    assert!(panic.is_err());
    assert!(executor
        .run(
            &BTreeMap::from([("live".into(), 1)]),
            &BTreeMap::new(),
            &[HandoffBind {
                name: "input",
                bytes: &vec![0; 128 * 12]
            }],
            &[],
        )
        .is_err());
    assert_eq!(
        executor.resources.as_ref().unwrap().input.as_slice(),
        staged
    );
}

#[test]
#[ignore = "CUDA; set HANDOFF_TEST_ASSETS to scratch with the speech object; use gpuq"]
fn handoff_admission_retires_resets_and_refuses_recycled_keys_without_overwriting_pcm() {
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, catalog) = fixture(&root, 1, true);
    let proven = plan.prove(&catalog).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let mut executor = CudaHandoff::load(backend, &root, &proven, 2).unwrap();
    let live = BTreeMap::from([("live".into(), 1)]);
    assert!(executor.run(&live, &BTreeMap::new(), &[], &[]).is_err());
    let input_a: Vec<_> = [1f32, 2., 3.]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let input_b: Vec<_> = [4f32, 5., 6.]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let prepared = executor.prepare_admission("packed", 0).unwrap();
    let key_a = executor
        .admit(
            prepared,
            &[HandoffBind {
                name: "input",
                bytes: &input_a,
            }],
        )
        .unwrap();
    let HandoffRun::Complete(first) = executor
        .run(&live, &BTreeMap::new(), &[], &[&key_a])
        .unwrap()
    else {
        panic!("first completed slab")
    };
    assert_eq!(first.host_slice().unwrap(), input_a);
    assert!(executor.prepare_admission("packed", 0).is_err());
    executor.release(&key_a).unwrap();
    assert!(executor
        .run(&live, &BTreeMap::new(), &[], &[&key_a])
        .is_err());
    let prepared = executor.prepare_admission("packed", 0).unwrap();
    let key_b = executor
        .admit(
            prepared,
            &[HandoffBind {
                name: "input",
                bytes: &input_b,
            }],
        )
        .unwrap();
    assert_ne!(key_a.epoch(), key_b.epoch());
    assert_eq!(key_b.slot(), 0);
    assert!(executor
        .run(&live, &BTreeMap::new(), &[], &[&key_a])
        .is_err());
    let HandoffRun::Complete(second) = executor
        .run(&live, &BTreeMap::new(), &[], &[&key_b])
        .unwrap()
    else {
        panic!("second completed slab")
    };
    assert_eq!(second.host_slice().unwrap(), input_b);
    assert_eq!(first.host_slice().unwrap(), input_a);
    assert!(matches!(
        executor
            .run(&live, &BTreeMap::new(), &[], &[&key_b])
            .unwrap(),
        HandoffRun::DeliveryBusy
    ));
    drop(first);
    assert!(matches!(
        executor
            .run(&live, &BTreeMap::new(), &[], &[&key_b])
            .unwrap(),
        HandoffRun::Complete(_)
    ));
    assert_eq!(second.host_slice().unwrap(), input_b);
}

#[test]
#[ignore = "CUDA; set HANDOFF_TEST_ASSETS to scratch with the speech object; use gpuq"]
fn handoff_owned_existing_providers_share_context_and_reject_stale_facts() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let root = std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap());
    let (plan, facts) = fixture(&root, 8, false);
    let proven = plan.prove(&facts).unwrap();
    let backend = Arc::new(CudaBackend::new(0).unwrap());
    let arena = SharedHandoffArena::allocate(backend.clone(), &proven).unwrap();
    let enqueues = Arc::new(AtomicUsize::new(0));
    let mut providers =
        existing_providers(&root, &arena, &facts, &plan, enqueues.clone(), false, false);
    for (component, provider) in &mut providers {
        let live_modules = provider.backend().live_modules();
        for (name, view) in plan
            .views
            .iter()
            .filter(|(_, view)| &view.component == component)
        {
            let legacy = provider.tensor(name).unwrap();
            let bank0 = provider.tensor_bank(name, 0).unwrap();
            assert_eq!((bank0.address, bank0.bytes), (legacy.address, legacy.bytes));
            assert_eq!(bank0.bytes, view.bytes);
            for bank in [1, u32::MAX] {
                assert!(provider.tensor_bank(name, bank).is_err());
            }
        }
        for reference in plan.schedule.iter().filter(|p| &p.component == component) {
            assert_eq!(
                provider.resolve_packet_bank(reference, 0).unwrap(),
                provider.resolve_packet(reference).unwrap()
            );
            assert_eq!(
                provider
                    .resolve_kernel_bank(reference, 0)
                    .err()
                    .unwrap()
                    .to_string(),
                provider
                    .resolve_kernel(reference)
                    .err()
                    .unwrap()
                    .to_string()
            );
            for bank in [1, u32::MAX] {
                assert!(provider.resolve_packet_bank(reference, bank).is_err());
                assert!(provider.resolve_kernel_bank(reference, bank).is_err());
            }
        }
        assert!(provider.quiescent());
        assert_eq!(provider.backend().live_modules(), live_modules);
    }
    assert_eq!(enqueues.load(Ordering::Relaxed), 0);
    let modules: Vec<_> = providers
        .values()
        .map(|p| (p.backend().clone(), p.backend().live_modules()))
        .collect();
    let mut executor = CudaHandoff::bind_owned(&proven, arena.clone(), providers, 2).unwrap();
    for name in ["producer", "consumer"] {
        assert!(executor
            .provider_mut::<ExistingPacketProvider>(name)
            .unwrap()
            .retirements
            .is_empty());
    }
    assert!(executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .is_ok());
    assert!(executor
        .provider_mut::<super::super::packet_exec::PacketHandoffProvider>("producer")
        .is_err());
    assert!(executor
        .provider_mut::<ExistingPacketProvider>("unknown")
        .is_err());
    executor.resources.as_mut().unwrap().programs.providers[0]
        .ready()
        .unwrap();
    assert!(executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .is_err());
    executor.resources.as_mut().unwrap().programs.providers[0].retired(true);
    executor.resources.as_mut().unwrap().inflight = true;
    assert!(executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .is_err());
    executor.resources.as_mut().unwrap().inflight = false;
    let resources = executor.resources.take();
    assert!(executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .is_err());
    executor.resources = resources;
    for (backend, count) in &modules {
        assert_eq!(backend.live_modules(), *count);
    }
    let input: Vec<_> = (0..24)
        .flat_map(|v| (v as f32 * 0.125).to_le_bytes())
        .collect();
    let live = BTreeMap::from([("live".into(), 8)]);
    let binds = [HandoffBind {
        name: "input",
        bytes: &input,
    }];
    executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .unwrap()
        .fail_ready = true;
    assert!(executor.run(&live, &BTreeMap::new(), &binds, &[]).is_err());
    assert_eq!(enqueues.load(Ordering::Relaxed), 0);
    assert!(executor
        .provider_mut::<ExistingPacketProvider>("consumer")
        .unwrap()
        .retirements
        .iter()
        .all(|&ok| ok));
    executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .unwrap()
        .fail_ready = false;
    for live in [1, 3, 8] {
        let HandoffRun::Complete(output) = executor
            .run(
                &BTreeMap::from([("live".into(), live)]),
                &BTreeMap::new(),
                &[HandoffBind {
                    name: "input",
                    bytes: &input,
                }],
                &[],
            )
            .unwrap()
        else {
            panic!("owned provider completion")
        };
        assert_eq!(output.host_slice().unwrap(), &input[..live as usize * 12]);
    }
    assert_eq!(enqueues.load(Ordering::Relaxed), 6);
    let mut slabs = Vec::new();
    for _ in 0..2 {
        let HandoffRun::Complete(slab) =
            executor.run(&live, &BTreeMap::new(), &binds, &[]).unwrap()
        else {
            panic!("owned completed slab")
        };
        slabs.push(slab);
    }
    let before = executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .unwrap()
        .retirements
        .len();
    assert!(matches!(
        executor.run(&live, &BTreeMap::new(), &binds, &[]).unwrap(),
        HandoffRun::DeliveryBusy
    ));
    let owner = executor
        .provider_mut::<ExistingPacketProvider>("producer")
        .unwrap();
    assert_eq!(owner.retirements.len(), before);
    assert!(owner.retirements.iter().all(|&ok| ok));
    assert_eq!(enqueues.load(Ordering::Relaxed), 10);
    drop(slabs);
    drop(executor);
    let providers = existing_providers(&root, &arena, &facts, &plan, enqueues.clone(), false, true);
    assert!(CudaHandoff::bind_owned(&proven, arena.clone(), providers, 2).is_err());
    assert_eq!(enqueues.load(Ordering::Relaxed), 10);
    let providers = existing_providers(&root, &arena, &facts, &plan, enqueues.clone(), true, false);
    assert!(CudaHandoff::bind_owned(&proven, arena, providers, 2).is_err());
    assert_eq!(enqueues.load(Ordering::Relaxed), 10);
}
