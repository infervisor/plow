use super::*;

fn reference(component: &str, program: &str) -> HandoffProgramRef {
    HandoffProgramRef {
        component: component.into(),
        program: program.into(),
    }
}

fn rows() -> HandoffRows {
    HandoffRows::Live {
        extent: "live".into(),
    }
}

fn access(view: &str, mode: HandoffAccessMode) -> HandoffAccess {
    HandoffAccess {
        view: view.into(),
        mode,
        rows: rows(),
        mask: None,
    }
}

fn kernel(accesses: Vec<HandoffAccess>) -> HandoffProgram {
    HandoffProgram {
        entry: HandoffExecutable::Kernel {
            object_role: "ops".into(),
            symbol: "typed_fixture".into(),
            abi_version: 1,
            capability: "fixture_abi".into(),
            capability_value: 1,
            parameters: accesses
                .iter()
                .map(|a| HandoffParameter::Tensor {
                    view: a.view.clone(),
                    dtype: if a.view == "out" {
                        PipelineDType::U8
                    } else {
                        PipelineDType::U32
                    },
                    access: a.mode,
                    alignment: if a.view == "out" { 1 } else { 4 },
                })
                .chain([HandoffParameter::Scalar {
                    dtype: HandoffScalarType::U32,
                    value: HandoffExpr::LiveRows {
                        extent: "live".into(),
                    },
                }])
                .collect(),
            grid: [
                HandoffExpr::CeilDiv {
                    value: Box::new(HandoffExpr::LiveRows {
                        extent: "live".into(),
                    }),
                    divisor: 128,
                },
                HandoffExpr::Constant { value: 1 },
                HandoffExpr::Constant { value: 1 },
            ],
            block: [128, 1, 1],
            shared_bytes: 0,
        },
        accesses,
    }
}

fn fixture() -> HandoffPlan {
    let producer = reference("up", "produce");
    let consumer = reference("down", "consume");
    let pack = reference("down", "pack");
    let view =
        |component: &str, tensor: &str, offset, dtype, width, stride, initialize| HandoffView {
            component: component.into(),
            tensor: tensor.into(),
            offset,
            bytes: 4 * stride,
            dtype,
            rows: 4,
            row_width: width,
            row_stride: stride,
            row_map: HandoffRowMap::Contiguous {
                domain: "packed".into(),
            },
            lifetime: HandoffLifetime::Transient { bank_stride: 0 },
            packet_initialize: initialize,
            immutable: false,
        };
    let readback = HandoffReadback {
        producer: pack.clone(),
        view: "out".into(),
        extent: "live".into(),
        row_stride: 12,
        fields: vec![
            HandoffReadbackField {
                name: "stop".into(),
                offset: 0,
                dtype: PipelineDType::U32,
                elements: 1,
            },
            HandoffReadbackField {
                name: "cb0".into(),
                offset: 4,
                dtype: PipelineDType::U32,
                elements: 1,
            },
            HandoffReadbackField {
                name: "pcm".into(),
                offset: 8,
                dtype: PipelineDType::I16,
                elements: 2,
            },
        ],
        sample_lengths: None,
    };
    HandoffPlan {
        components: BTreeMap::from([
            (
                "up".into(),
                HandoffComponent {
                    packet: "up.pkt".into(),
                    driver: "test.v1".into(),
                    objects: BTreeMap::from([("ops".into(), "ops.cubin".into())]),
                    programs: BTreeMap::from([(
                        "produce".into(),
                        HandoffProgram {
                            entry: HandoffExecutable::Packet {
                                section: "packet".into(),
                                program: 0,
                            },
                            accesses: vec![access("a", HandoffAccessMode::Write)],
                        },
                    )]),
                },
            ),
            (
                "down".into(),
                HandoffComponent {
                    packet: "down.pkt".into(),
                    driver: "test.v1".into(),
                    objects: BTreeMap::from([("ops".into(), "ops.cubin".into())]),
                    programs: BTreeMap::from([
                        (
                            "consume".into(),
                            kernel(vec![access("b", HandoffAccessMode::ReadWrite)]),
                        ),
                        (
                            "pack".into(),
                            kernel(vec![
                                access("b", HandoffAccessMode::Read),
                                access("out", HandoffAccessMode::Write),
                            ]),
                        ),
                    ]),
                },
            ),
        ]),
        arena: HandoffArena {
            bytes: 128,
            alignment: 16,
            banks: 1,
        },
        extents: BTreeMap::from([("live".into(), 4)]),
        views: BTreeMap::from([
            (
                "a".into(),
                view("up", "codes", 0, PipelineDType::U32, 2, 8, true),
            ),
            (
                "b".into(),
                view("down", "codes", 0, PipelineDType::U32, 2, 8, false),
            ),
            (
                "out".into(),
                view("down", "block", 64, PipelineDType::U8, 12, 12, false),
            ),
        ]),
        schedule: vec![producer.clone(), consumer.clone(), pack.clone()],
        edges: vec![
            HandoffEdge {
                producer: HandoffEndpoint {
                    program: producer,
                    view: "a".into(),
                },
                consumer: HandoffConsumer::Program {
                    endpoint: HandoffEndpoint {
                        program: consumer.clone(),
                        view: "b".into(),
                    },
                },
                class: HandoffClass::Alias,
                dtype: PipelineDType::U32,
                rows: rows(),
                order: HandoffOrder::SameStream,
                generation_delta: 0,
            },
            HandoffEdge {
                producer: HandoffEndpoint {
                    program: consumer,
                    view: "b".into(),
                },
                consumer: HandoffConsumer::Program {
                    endpoint: HandoffEndpoint {
                        program: pack.clone(),
                        view: "b".into(),
                    },
                },
                class: HandoffClass::Alias,
                dtype: PipelineDType::U32,
                rows: rows(),
                order: HandoffOrder::Event {
                    name: "ready".into(),
                },
                generation_delta: 0,
            },
            HandoffEdge {
                producer: HandoffEndpoint {
                    program: pack,
                    view: "out".into(),
                },
                consumer: HandoffConsumer::Host,
                class: HandoffClass::Host,
                dtype: PipelineDType::U8,
                rows: rows(),
                order: HandoffOrder::SameStream,
                generation_delta: 0,
            },
        ],
        bind_writes: vec![],
        readback,
        execution: HandoffExecution::Synchronous,
    }
}

fn physical_gather_fixture() -> HandoffPlan {
    let mut p = fixture();
    p.extents.insert("live".into(), 2);
    for name in ["a", "b"] {
        let view = p.views.get_mut(name).unwrap();
        view.rows = 8;
        view.bytes = 64;
    }
    p.components
        .get_mut("up")
        .unwrap()
        .programs
        .get_mut("produce")
        .unwrap()
        .accesses[0]
        .rows = HandoffRows::Fixed { rows: 8 };
    p.views.insert(
        "map".into(),
        HandoffView {
            component: "down".into(),
            tensor: "selection".into(),
            offset: 112,
            bytes: 8,
            dtype: PipelineDType::U32,
            rows: 2,
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
    p.views.get_mut("b").unwrap().row_map = HandoffRowMap::Gather {
        map: "map".into(),
        domain: "packed".into(),
        unique: true,
    };
    for program in p.components.get_mut("down").unwrap().programs.values_mut() {
        let mut accesses = program.accesses.clone();
        accesses.push(access("map", HandoffAccessMode::Read));
        *program = kernel(accesses);
    }
    p.bind_writes.push(HandoffBindWrite {
        name: "selection".into(),
        view: "map".into(),
        rows: rows(),
        phase: HandoffBindPhase::Tick,
        order: HandoffOrder::SameStream,
    });
    p
}

#[test]
fn fixed_physical_producer_covers_a_live_gathered_alias_only_in_the_forward_direction() {
    let p = physical_gather_fixture();
    p.prove(&catalog(&p)).unwrap();
    for count in [0, 1, 2, 7] {
        let mut partial = p.clone();
        partial
            .components
            .get_mut("up")
            .unwrap()
            .programs
            .get_mut("produce")
            .unwrap()
            .accesses[0]
            .rows = HandoffRows::Fixed { rows: count };
        assert!(partial.prove(&catalog(&partial)).is_err(), "{count}");
    }
    for mutation in 0..7 {
        let mut bad = p.clone();
        match mutation {
            0 => {
                bad.components
                    .get_mut("up")
                    .unwrap()
                    .programs
                    .get_mut("produce")
                    .unwrap()
                    .accesses[0]
                    .rows = rows()
            }
            1 => {
                bad.views.get_mut("b").unwrap().row_map = HandoffRowMap::Gather {
                    map: "map".into(),
                    domain: "other".into(),
                    unique: true,
                }
            }
            2 => bad.views.get_mut("b").unwrap().offset = 8,
            3 => bad.views.get_mut("b").unwrap().row_stride = 4,
            4 => bad.views.get_mut("b").unwrap().lifetime = HandoffLifetime::Persistent,
            5 => {
                let gathered = bad.views["b"].row_map.clone();
                bad.views.get_mut("b").unwrap().row_map = bad.views["a"].row_map.clone();
                bad.views.get_mut("a").unwrap().row_map = gathered;
            }
            _ => bad.edges[0].generation_delta = 1,
        }
        assert!(bad.prove(&catalog(&bad)).is_err(), "{mutation}");
    }
}

#[test]
fn a_masked_physical_write_does_not_prove_a_gathered_alias() {
    let mut p = physical_gather_fixture();
    p.arena.bytes = 256;
    let mut valid = p.views["map"].clone();
    valid.tensor = "validity".into();
    valid.offset = 128;
    valid.rows = 8;
    valid.bytes = 32;
    valid.packet_initialize = true;
    p.views.insert("valid".into(), valid);
    let produce = p
        .components
        .get_mut("up")
        .unwrap()
        .programs
        .get_mut("produce")
        .unwrap();
    produce.accesses[0].mask = Some("valid".into());
    let mut valid_read = access("valid", HandoffAccessMode::Read);
    valid_read.rows = HandoffRows::Fixed { rows: 8 };
    produce.accesses.push(valid_read);
    for program in p.components.get_mut("down").unwrap().programs.values_mut() {
        let mut accesses = program.accesses.clone();
        for access in &mut accesses {
            if access.view == "b" {
                access.mask = Some("valid".into());
            }
        }
        accesses.push(access("valid", HandoffAccessMode::Read));
        *program = kernel(accesses);
    }
    assert!(p
        .validate()
        .unwrap_err()
        .contains("alias address functions differ"));
}

fn catalog(plan: &HandoffPlan) -> HandoffProofCatalog {
    HandoffProofCatalog {
        arena: Some(plan.arena.clone()),
        extents: plan.extents.clone(),
        views: plan.views.clone(),
        programs: plan
            .schedule
            .iter()
            .map(|p| {
                let program = plan.program(p).unwrap();
                (
                    p.clone(),
                    HandoffProgramProof {
                        entry: program.entry.clone(),
                        accesses: program.accesses.clone(),
                        copies: Vec::new(),
                        private_transients_banked: false,
                        persistent_stop_safe: false,
                    },
                )
            })
            .collect(),
        readbacks: BTreeMap::from([(plan.readback.producer.clone(), plan.readback.clone())]),
    }
}

#[test]
fn explicit_alias_order_and_i16_readback_require_typed_program_proofs() {
    let plan = fixture();
    plan.validate().unwrap();
    assert!(plan.prove(&HandoffProofCatalog::default()).is_err());
    assert!(plan.prove(&catalog(&plan)).is_ok());
    let bytes = serde_json::to_vec(&plan).unwrap();
    let parsed: HandoffPlan = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed, plan);
    assert_eq!(PipelineDType::I16.bytes(), 2);
    let mut stale = catalog(&plan);
    stale
        .programs
        .get_mut(&reference("down", "consume"))
        .unwrap()
        .accesses[0]
        .mode = HandoffAccessMode::Read;
    assert!(plan.prove(&stale).is_err());
    let mut stale = catalog(&plan);
    stale
        .readbacks
        .get_mut(&plan.readback.producer)
        .unwrap()
        .fields[2]
        .dtype = PipelineDType::F16;
    assert!(plan.prove(&stale).is_err());
}

#[test]
fn refuses_geometry_ordering_abi_and_readback_mutations() {
    for mutation in 0..36 {
        let mut p = fixture();
        match mutation {
            0 => p.arena.bytes = 80,
            1 => p.arena.alignment = 3,
            2 => p.arena.banks = 0,
            3 => p.views.get_mut("a").unwrap().dtype = PipelineDType::F32,
            4 => p.views.get_mut("b").unwrap().offset = 4,
            5 => p.views.get_mut("b").unwrap().row_stride = 4,
            6 => p.views.get_mut("b").unwrap().row_width = 1,
            7 => p.views.get_mut("b").unwrap().bytes = 8,
            8 => p.components.get_mut("up").unwrap().packet = "../up.pkt".into(),
            9 => p.schedule[0].program = "missing".into(),
            10 => p.schedule.reverse(),
            11 => p.schedule.push(p.schedule[0].clone()),
            12 => {
                p.edges[0].rows = HandoffRows::Live {
                    extent: "missing".into(),
                }
            }
            13 => {
                p.extents.insert("live".into(), 0);
            }
            14 => p.views.get_mut("a").unwrap().row_width = u32::MAX,
            15 => p.views.get_mut("a").unwrap().offset = u64::MAX,
            16 => p.readback.fields[2].offset = 4,
            17 => {
                p.extents.insert("live".into(), 9);
            }
            18 => p.readback.row_stride = 16,
            19 => p.readback.fields[2].elements = 3,
            20 => p.readback.fields[2].name = "stop".into(),
            21 => p.edges.push(p.edges[2].clone()),
            22 => p.edges[0].consumer = HandoffConsumer::Host,
            23 => p.edges[0].generation_delta = 1,
            24 => {
                p.components
                    .get_mut("down")
                    .unwrap()
                    .programs
                    .get_mut("consume")
                    .unwrap()
                    .accesses[0]
                    .mask = Some("out".into())
            }
            25 => p.views.get_mut("b").unwrap().packet_initialize = true,
            26..=31 | 35 => {
                let HandoffExecutable::Kernel {
                    parameters,
                    abi_version,
                    capability,
                    grid,
                    ..
                } = &mut p
                    .components
                    .get_mut("down")
                    .unwrap()
                    .programs
                    .get_mut("consume")
                    .unwrap()
                    .entry
                else {
                    unreachable!()
                };
                match mutation {
                    26 => {
                        parameters[0] = HandoffParameter::Tensor {
                            view: "b".into(),
                            dtype: PipelineDType::F16,
                            access: HandoffAccessMode::ReadWrite,
                            alignment: 4,
                        }
                    }
                    27 => *abi_version = 0,
                    28 => capability.clear(),
                    29 => {
                        grid[0] = HandoffExpr::CeilDiv {
                            value: Box::new(HandoffExpr::Constant { value: 1 }),
                            divisor: 0,
                        }
                    }
                    30 => grid[0] = HandoffExpr::Constant { value: u64::MAX },
                    31 => parameters.push(HandoffParameter::Scalar {
                        dtype: HandoffScalarType::U32,
                        value: HandoffExpr::Constant { value: 1u64 << 32 },
                    }),
                    35 => parameters.push(HandoffParameter::Scalar {
                        dtype: HandoffScalarType::U32,
                        value: HandoffExpr::Field {
                            name: "n".into(),
                            min: 10,
                            max: 1,
                        },
                    }),
                    _ => unreachable!(),
                }
            }
            32 => {
                p.views.get_mut("b").unwrap().row_map = HandoffRowMap::Gather {
                    map: "b".into(),
                    domain: "physical".into(),
                    unique: true,
                }
            }
            33 => {
                p.execution = HandoffExecution::Overlap {
                    feedback: "b".into(),
                }
            }
            34 => p.bind_writes.push(HandoffBindWrite {
                name: "reset".into(),
                view: "missing".into(),
                rows: rows(),
                order: HandoffOrder::SameStream,
                phase: HandoffBindPhase::Admission,
            }),
            _ => unreachable!(),
        }
        assert!(p.validate().is_err(), "mutation {mutation} was accepted");
    }
}

#[test]
fn metadata_keeps_legacy_bytes_and_rejects_unknown_versions_and_variants() {
    let legacy = br#"{"version":1,"pipelines":[{"name":"legacy","driver":"test.v1","programs":{"run":0},"tensors":{},"parameters":{}}]}"#;
    let p: PacketPipelines = serde_json::from_slice(legacy).unwrap();
    assert_eq!(serde_json::to_vec(&p).unwrap(), legacy);
    PipelineMetadata::parse(legacy)
        .unwrap()
        .validate(1, |_| None)
        .unwrap();
    let v2 = PacketPipelinesV2 {
        version: HANDOFF_VERSION,
        pipelines: p.pipelines,
        handoff_plans: BTreeMap::from([("tick".into(), fixture())]),
    };
    let json = serde_json::to_vec(&v2).unwrap();
    PipelineMetadata::parse(&json)
        .unwrap()
        .validate(1, |_| None)
        .unwrap();
    assert!(serde_json::from_slice::<PacketPipelines>(&json).is_err());
    for mutation in 0..5 {
        let mut value = serde_json::to_value(&v2).unwrap();
        match mutation {
            0 => value["version"] = 3.into(),
            1 => value["version"] = 1.into(),
            2 => value["handoff_plans"]["tick"]["edges"][0]["class"]["kind"] = "callback".into(),
            3 => value["handoff_plans"]["tick"]["execution"]["kind"] = "guess".into(),
            4 => value["handoff_plans"]["tick"]["unrecognized"] = true.into(),
            _ => unreachable!(),
        }
        assert!(PipelineMetadata::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}

#[test]
fn bounded_launch_fields_fail_closed_and_ceil_div_does_not_overflow() {
    let expr = HandoffExpr::CeilDiv {
        value: Box::new(HandoffExpr::LiveRows {
            extent: "live".into(),
        }),
        divisor: 256,
    };
    assert_eq!(
        expr.evaluate(&BTreeMap::from([("live".into(), 0)]), &BTreeMap::new())
            .unwrap(),
        0
    );
    assert_eq!(
        expr.evaluate(
            &BTreeMap::from([("live".into(), u32::MAX)]),
            &BTreeMap::new()
        )
        .unwrap(),
        16_777_216
    );
    let field = HandoffExpr::Field {
        name: "count".into(),
        min: 2,
        max: 7,
    };
    for value in [0, 1, 8, u64::MAX] {
        assert!(field
            .evaluate(&BTreeMap::new(), &BTreeMap::from([("count".into(), value)]))
            .is_err());
    }
    assert!(field.evaluate(&BTreeMap::new(), &BTreeMap::new()).is_err());
    assert_eq!(
        field
            .evaluate(&BTreeMap::new(), &BTreeMap::from([("count".into(), 5)]))
            .unwrap(),
        5
    );
}

#[test]
fn typed_pack_is_required_for_distinct_source_and_destination_rows() {
    let mut p = fixture();
    p.views.get_mut("b").unwrap().offset = 32;
    assert!(p.validate().is_err());
    let copy = reference("down", "codes_pack");
    p.schedule.insert(1, copy.clone());
    p.components.get_mut("down").unwrap().programs.insert(
        "codes_pack".into(),
        kernel(vec![
            access("a", HandoffAccessMode::Read),
            access("b", HandoffAccessMode::Write),
        ]),
    );
    p.edges[0].class = HandoffClass::DeviceCopy {
        program: copy.clone(),
    };
    p.validate().unwrap();
    let mut proof = catalog(&p);
    assert!(p.prove(&proof).is_err());
    proof
        .programs
        .get_mut(&copy)
        .unwrap()
        .copies
        .push(HandoffCopyProof {
            source: "a".into(),
            destination: "b".into(),
            rows: rows(),
            mask: None,
        });
    p.prove(&proof).unwrap();
}

#[test]
fn missing_reads_and_intervening_alias_writes_are_rejected() {
    let p = fixture();
    let proof = catalog(&p);
    let mut no_edge = p.clone();
    no_edge.edges.remove(0);
    assert!(no_edge.prove(&proof).is_err());
    let mut clobbered = p;
    let clobber = reference("up", "clobber");
    clobbered.schedule.insert(1, clobber.clone());
    clobbered.components.get_mut("up").unwrap().programs.insert(
        "clobber".into(),
        HandoffProgram {
            entry: HandoffExecutable::Packet {
                section: "packet".into(),
                program: 1,
            },
            accesses: vec![access("a", HandoffAccessMode::Write)],
        },
    );
    assert!(clobbered.prove(&catalog(&clobbered)).is_err());
}

#[test]
fn host_payload_must_survive_until_final_readback_enqueue() {
    let mut p = fixture();
    let overwrite = reference("down", "overwrite");
    p.schedule.push(overwrite);
    p.components.get_mut("down").unwrap().programs.insert(
        "overwrite".into(),
        kernel(vec![access("out", HandoffAccessMode::Write)]),
    );
    assert!(p.validate().is_err());
    assert!(p.prove(&catalog(&p)).is_err());
}

#[test]
fn actual_layout_and_pointer_alignment_are_proof_inputs() {
    let p = fixture();
    let mut proof = catalog(&p);
    proof.views.get_mut("a").unwrap().dtype = PipelineDType::F32;
    assert!(p.prove(&proof).is_err());
    let mut p = p;
    let HandoffExecutable::Kernel { parameters, .. } = &mut p
        .components
        .get_mut("down")
        .unwrap()
        .programs
        .get_mut("consume")
        .unwrap()
        .entry
    else {
        unreachable!()
    };
    let HandoffParameter::Tensor { alignment, .. } = &mut parameters[0] else {
        unreachable!()
    };
    *alignment = 16;
    assert!(p.validate().is_err());
}

#[test]
fn generation_owned_maps_and_masks_need_initialized_typed_reads() {
    for gather in [false, true] {
        let mut p = fixture();
        p.arena.bytes = 144;
        let mut map = p.views["a"].clone();
        map.offset = 128;
        map.bytes = 16;
        map.row_width = 1;
        map.row_stride = 4;
        map.packet_initialize = false;
        map.immutable = true;
        p.views.insert("map".into(), map);
        if gather {
            for name in ["a", "b"] {
                p.views.get_mut(name).unwrap().row_map = HandoffRowMap::Gather {
                    map: "map".into(),
                    domain: "physical".into(),
                    unique: true,
                };
            }
        } else {
            p.components
                .get_mut("down")
                .unwrap()
                .programs
                .get_mut("consume")
                .unwrap()
                .accesses[0]
                .mask = Some("map".into());
            p.components
                .get_mut("down")
                .unwrap()
                .programs
                .get_mut("pack")
                .unwrap()
                .accesses[0]
                .mask = Some("map".into());
        }
        assert!(p.prove(&catalog(&p)).is_err());
        // A typed read alone does not initialize the map or mask.
        for component in p.components.values_mut() {
            for program in component.programs.values_mut() {
                program
                    .accesses
                    .push(access("map", HandoffAccessMode::Read));
            }
        }
        assert!(p.prove(&catalog(&p)).is_err());
        p.views.get_mut("map").unwrap().packet_initialize = true;
        p.prove(&catalog(&p)).unwrap();
        // Mutability through a second view cannot bypass immutable-map ownership.
        let mut alias = p.views["map"].clone();
        alias.immutable = false;
        alias.packet_initialize = false;
        p.views.insert("mutable_alias".into(), alias);
        p.components
            .get_mut("up")
            .unwrap()
            .programs
            .get_mut("produce")
            .unwrap()
            .accesses
            .push(access("mutable_alias", HandoffAccessMode::Write));
        assert!(p.validate().is_err());
    }
}

#[test]
fn next_generation_feedback_needs_a_seed_before_the_first_tick() {
    let mut p = fixture();
    p.arena.bytes = 256;
    p.arena.banks = 2;
    p.schedule.rotate_left(1);
    for name in ["a", "b"] {
        p.views.get_mut(name).unwrap().lifetime = HandoffLifetime::Persistent;
    }
    p.views.get_mut("out").unwrap().lifetime = HandoffLifetime::Transient { bank_stride: 128 };
    p.execution = HandoffExecution::Overlap {
        feedback: "a".into(),
    };
    p.edges[0].generation_delta = 1;
    let prove = |p: &HandoffPlan| {
        let mut proof = catalog(p);
        for program in proof.programs.values_mut() {
            program.private_transients_banked = true;
            program.persistent_stop_safe = true;
        }
        p.prove(&proof).map(|_| ())
    };
    prove(&p).unwrap();
    for name in ["a", "b"] {
        let mut clobbered = p.clone();
        clobbered.bind_writes.push(HandoffBindWrite {
            name: "overwrite".into(),
            view: name.into(),
            rows: rows(),
            phase: HandoffBindPhase::Tick,
            order: HandoffOrder::SameStream,
        });
        assert!(clobbered
            .validate()
            .unwrap_err()
            .contains("tick bind destroys"));
    }
    let mut subview = p.clone();
    let mut control = p.views["a"].clone();
    control.tensor = "control".into();
    control.offset = 16;
    control.bytes = 16;
    control.row_width = 1;
    control.row_stride = 4;
    control.packet_initialize = false;
    subview.views.insert("control".into(), control);
    subview.bind_writes.push(HandoffBindWrite {
        name: "control".into(),
        view: "control".into(),
        rows: rows(),
        phase: HandoffBindPhase::Tick,
        order: HandoffOrder::SameStream,
    });
    assert!(subview
        .validate()
        .unwrap_err()
        .contains("tick bind destroys"));
    subview.views.get_mut("control").unwrap().offset = 32;
    prove(&subview).unwrap();
    p.views.get_mut("a").unwrap().packet_initialize = false;
    assert!(p
        .validate()
        .unwrap_err()
        .contains("first-tick initialization"));
    assert!(prove(&p).is_err());
    p.views.get_mut("b").unwrap().packet_initialize = true;
    prove(&p).unwrap();
    p.views.get_mut("b").unwrap().packet_initialize = false;
    for name in ["a", "b"] {
        p.bind_writes = vec![HandoffBindWrite {
            name: "seed".into(),
            view: name.into(),
            rows: rows(),
            phase: HandoffBindPhase::Admission,
            order: HandoffOrder::SameStream,
        }];
        prove(&p).unwrap();
        p.bind_writes[0].rows = HandoffRows::Fixed { rows: 3 };
        assert!(prove(&p).is_err());
        p.bind_writes[0].rows = rows();
        p.bind_writes[0].phase = HandoffBindPhase::Tick;
        assert!(prove(&p).is_err());
    }
}

#[test]
fn next_generation_transient_alias_and_persistent_overlap_are_rejected() {
    let mut p = fixture();
    p.views.get_mut("a").unwrap().lifetime = HandoffLifetime::Persistent;
    assert!(p.validate().is_err());
    let mut p = fixture();
    p.arena.bytes = 272;
    p.arena.banks = 2;
    for view in p.views.values_mut() {
        view.lifetime = HandoffLifetime::Transient { bank_stride: 128 };
    }
    let mut feedback = p.views["a"].clone();
    feedback.offset = 256;
    feedback.bytes = 16;
    feedback.row_width = 1;
    feedback.row_stride = 4;
    feedback.lifetime = HandoffLifetime::Persistent;
    p.views.insert("feedback".into(), feedback);
    p.execution = HandoffExecution::Overlap {
        feedback: "feedback".into(),
    };
    p.edges[0].generation_delta = 1;
    assert!(p.validate().is_err());
}

#[test]
fn distinct_operands_cannot_hide_same_program_write_aliases() {
    let mut p = fixture();
    p.components
        .get_mut("down")
        .unwrap()
        .programs
        .get_mut("consume")
        .unwrap()
        .accesses
        .push(access("a", HandoffAccessMode::Read));
    assert!(p.validate().is_err());
    let mut p = fixture();
    p.components
        .get_mut("down")
        .unwrap()
        .programs
        .get_mut("consume")
        .unwrap()
        .accesses
        .push(access("a", HandoffAccessMode::Write));
    assert!(p.validate().is_err());
}
