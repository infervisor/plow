//! The emit-side knob registry: every `EmitConfig` knob, every raw `PLOW_*` read on the emit side
//! (devgen, packet, plowc, kernelcaps), every object define, the cross-knob constraints over them,
//! and the declared targets with their qualified recipes.
//!
//! Checkpoint K runs at emit through [`gate`]: the recorded knob sources and the emit target go to
//! `plow_verify`, which resolves them against this registry, checks every constraint, and checks
//! the registry itself for consistency. `plowc` installs the verifier; a rejection aborts emission
//! before a byte is written, like checkpoint D. `build.json` records the verdict under `knobs`,
//! with the constraint-relevant values plowrt re-checks at load.
//!
//! The tables are one line per knob and are held to their sources by the tests below: clap for
//! `EMIT`, the source tree for `RAW_ENV` and `OBJECT_DEFINES`.

use plow_asset::knob::{
    Allow, Check, Cmp, Constraint, Default, DefaultCase, Domain, Formula as F, KnobSpec, Layer,
    OpSel, ScopeField, Source, Status, Target, TargetAtom as T, TargetSpec, Val, U32,
};
use serde_json::{json, Value};

const TRUE: Val = Val::Bool(true);
const FALSE: Val = Val::Bool(false);
const OFF: Default = Default::Static(FALSE);
const ON: Default = Default::Static(TRUE);
const UNSET: Default = Default::Static(Val::Unset);

const OPT_IN: Status = Status::OptIn;
const DIAG: Status = Status::Diagnostic;
const REMOVED: Status = Status::Removed;
const PROMOTED: Status = Status::Qualified {
    evidence: &["docs/flags-reference.md: a promoted default; `=0` is the rollback"],
};
const GLM_RECIPE: Status = Status::Qualified {
    evidence: &[
        "crates/devgen/src/lib.rs apply_production_defaults: 47-50 out tok/s on 8x MI300X, retrieval 18/18",
        "docs/flags-reference.md: The qualified GLM gfx942 TP8 recipe",
    ],
};
const GLM_RECIPE_PF_EXT: Status = Status::Qualified {
    evidence: &[
        "crates/devgen/src/emit_config.rs GLM_GEMM_LT_PF_EXT_QUALIFIED: router excluded, review log #65",
    ],
};
const DECODE_LADDER: Status = Status::Qualified {
    evidence: &[
        "crates/devgen/src/lib.rs apply_production_defaults: gfx942 MM=16 object costs c=8 -27.5%; the ladder stops at 8",
    ],
};
const PACKED_SIBLINGS: Status = Status::Qualified {
    evidence: &["crates/devgen/src/lib.rs apply_production_defaults: ordinary programs byte-identical with the packed siblings (glm_tests)"],
};
const GEMMA_NATIVE_PURE_GEMM: Status = Status::Qualified {
    evidence: &[
        "plans/gemma4-4k-8k-native-block.md: exact SM90a BF16 and W8A8 4K/8K pure-GEMM packet A/B",
    ],
};
const UNISEG: Status = Status::Qualified {
    evidence: &["crates/plowc/src/main.rs effective_uniseg: the sm_120 interpreter implements the single-segment path only"],
};
const TOKEN_BATCH_TP_PARKED: Status = Status::Parked {
    reason: "the seams-aware body corrupted long-context prefill (retrieval 9/18 base, 0/21 tail)",
    evidence: &[
        "review log #63",
        "docs/flags-reference.md: PLOW_TOKEN_BATCH_TP",
    ],
};
const MOE_SHARED_SEED_CANDIDATE: Status = Status::Candidate {
    evidence: &["review log #86: glue2-t3-g4-rung, P8192-S 648.5 -> 641.4 ms (floor 2.3), P4096-S -3.3, P8192-0 -8.6; values inside the cross-process floor (glue2-t3-g4-value); served A/B pending"],
};
const SEG_EXPERIMENT_PARKED: Status = Status::Parked {
    reason: "rejected segmentation experiment (+91.7 / +3.6 / +22.6 ms TTFT)",
    evidence: &["docs/flags-reference.md: Emit-side knobs that are NOT EmitConfig fields"],
};

/// `apply_production_defaults`'s gate for the qualified GLM recipe.
const GLM_TARGET: F = F::And(&[
    F::Target(T::Cap("glm")),
    F::Target(T::Arch("gfx942")),
    F::Target(T::Tp(8)),
    F::Target(T::NCu(304)),
    F::Not(&F::Atom("emit.mxfp4", Cmp::Eq, TRUE)),
]);

const GLM_RECIPE_ON: Default = Default::Production {
    cases: &[DefaultCase {
        when: GLM_TARGET,
        value: TRUE,
    }],
    otherwise: Val::Unset,
};

/// `EmitConfig::glm_seq_par`: the default stands aside for the two-shot seams it replaces, and
/// `glm_recipe_unset` records the resolved value either way.
const GLM_SEQ_PAR_DEFAULT: Default = Default::Production {
    cases: &[
        DefaultCase {
            when: F::And(&[
                GLM_TARGET,
                F::Not(&F::Atom("emit.glm_xr_res", Cmp::Eq, TRUE)),
                F::Not(&F::Atom("emit.glm_xr_band", Cmp::Gt, Val::Nat(1))),
            ]),
            value: TRUE,
        },
        DefaultCase {
            when: GLM_TARGET,
            value: FALSE,
        },
    ],
    otherwise: Val::Unset,
};

const GLM_SEQ_PAR_PROJ_DEFAULT: Default = Default::Production {
    cases: &[
        DefaultCase {
            when: F::And(&[GLM_TARGET, F::Atom("emit.glm_seq_par", Cmp::Eq, TRUE)]),
            value: TRUE,
        },
        DefaultCase {
            when: GLM_TARGET,
            value: FALSE,
        },
    ],
    otherwise: Val::Unset,
};

const GLM_GEMM_LT_PF_EXT_DEFAULT: Default = Default::Production {
    cases: &[DefaultCase {
        when: GLM_TARGET,
        value: Val::Str(crate::emit_config::GLM_GEMM_LT_PF_EXT_QUALIFIED),
    }],
    otherwise: Val::Unset,
};

const DECODE_LADDER_DEFAULT: Default = Default::Production {
    cases: &[
        DefaultCase {
            when: F::And(&[
                F::Target(T::Arch("sm_90a")),
                F::Target(T::Cap("decode_ladder")),
                F::Target(T::Tp(1)),
                F::Atom("emit.decode_batch", Cmp::Eq, Val::Nat(1)),
            ]),
            value: Val::Str("1,2,4,8,16"),
        },
        DefaultCase {
            when: F::And(&[
                F::Target(T::Arch("gfx942")),
                F::Target(T::Cap("decode_ladder")),
                F::Target(T::Tp(1)),
                F::Atom("emit.decode_batch", Cmp::Eq, Val::Nat(1)),
            ]),
            value: Val::Str("1,2,4,8"),
        },
    ],
    otherwise: Val::Unset,
};

const PURE_GEMM_DEFAULT: Default = Default::Production {
    cases: &[DefaultCase {
        when: F::And(&[
            F::Target(T::Cap("gemma")),
            F::Target(T::Arch("sm_90a")),
            F::Target(T::Tp(1)),
            F::Or(&[
                F::Atom("emit.w8a8", Cmp::Eq, TRUE),
                F::And(&[
                    F::Atom("emit.fp8", Cmp::Eq, FALSE),
                    F::Atom("emit.w8a8", Cmp::Eq, FALSE),
                    F::Atom("emit.w8a16", Cmp::Eq, FALSE),
                    F::Atom("emit.mxfp4", Cmp::Eq, FALSE),
                ]),
            ]),
        ]),
        value: Val::Str("1"),
    }],
    otherwise: Val::Unset,
};

const PACKED_PREFILL_DEFAULT: Default = Default::Production {
    cases: &[DefaultCase {
        when: F::And(&[
            F::Target(T::Cap("packed_prefill_siblings")),
            F::Target(T::Arch("gfx942")),
        ]),
        value: TRUE,
    }],
    otherwise: Val::Unset,
};

/// `plowc`'s `effective_uniseg`; `segmented` is the `--segmented` flag, noted as a target cap.
const UNISEG_DEFAULT: Default = Default::Production {
    cases: &[
        DefaultCase {
            when: F::Target(T::Cap("segmented")),
            value: FALSE,
        },
        DefaultCase {
            when: F::Or(&[
                F::Target(T::Arch("sm_120a")),
                F::Target(T::Arch("sm_120")),
                F::Target(T::Arch("metal3")),
            ]),
            value: TRUE,
        },
    ],
    otherwise: FALSE,
};

const C_W8A8: &[Constraint] = &[Constraint {
    id: "w8a8_excludes_w8a16",
    formula: F::Not(&F::And(&[
        F::Atom("emit.w8a8", Cmp::Eq, TRUE),
        F::Atom("emit.w8a16", Cmp::Eq, TRUE),
    ])),
    site: "crates/devgen/src/lib.rs: PLOW_W8A8=1 and PLOW_W8A16=1 name two activation profiles",
    check: Check::Load,
}];

const C_MXFP4: &[Constraint] = &[Constraint {
    id: "mxfp4_excludes_fp8_activations",
    formula: F::Not(&F::And(&[
        F::Atom("emit.mxfp4", Cmp::Eq, TRUE),
        F::Or(&[
            F::Atom("emit.w8a8", Cmp::Eq, TRUE),
            F::Atom("emit.w8a16", Cmp::Eq, TRUE),
        ]),
    ])),
    site: "crates/devgen/src/emit_config.rs: PLOW_MXFP4=1 is A4W4; it is incompatible with PLOW_W8A8/PLOW_W8A16",
    check: Check::Load,
}];

const C_CHANNEL_MLP: &[Constraint] = &[
    Constraint {
        id: "channel_mlp_target",
        formula: F::Implies(
            &F::Atom("emit.ane_mlp_channels", Cmp::Ne, Val::Unset),
            &F::And(&[
                F::Target(T::Arch("metal3")),
                F::Target(T::Tp(1)),
                F::Or(&[F::Target(T::Model("llama")), F::Target(T::Model("qwen3"))]),
            ]),
        ),
        site: "crates/devgen/src/lib.rs: channel MLP requires Metal dense Llama/Qwen3 TP1",
        check: Check::Load,
    },
    Constraint {
        id: "channel_mlp_exclusive",
        formula: F::Implies(
            &F::Atom("emit.ane_mlp_channels", Cmp::Ne, Val::Unset),
            &F::And(&[
                F::Atom("emit.row_split", Cmp::Eq, Val::Unset),
                F::Not(&F::Atom("emit.w8a8", Cmp::Eq, TRUE)),
                F::Not(&F::Atom("emit.emit_packed_prefill", Cmp::Eq, TRUE)),
                F::Atom("env.PLOW_BLOCK", Cmp::Eq, Val::Unset),
            ]),
        ),
        site: "crates/devgen/src/lib.rs: channel MLP cannot combine row split, packed prefill, W8A8 or block mode",
        check: Check::Load,
    },
];

const C_OFOLD: &[Constraint] = &[Constraint {
    id: "ofold_excludes_dsa_pf_and_fp8_kv",
    formula: F::Implies(
        &F::Atom("emit.glm_ofold", Cmp::Eq, TRUE),
        &F::And(&[
            F::Not(&F::Atom("emit.glm_dsa_pf", Cmp::Eq, TRUE)),
            F::Not(&F::Atom("emit.glm_fp8_kv", Cmp::Eq, TRUE)),
        ]),
    ),
    site: "crates/devgen/src/mla.rs: PLOW_GLM_OFOLD=1 cannot combine with PLOW_GLM_DSA_PF or PLOW_GLM_FP8_KV",
    check: Check::Load,
}];

const C_PACKED_SPARSE_PF: &[Constraint] = &[Constraint {
    id: "packed_sparse_pf_contract",
    formula: F::Implies(
        &F::Atom("emit.packed_sparse_pf", Cmp::Eq, TRUE),
        &F::And(&[
            F::Atom("emit.glm_index_tp", Cmp::Eq, TRUE),
            F::Atom("emit.glm_fp8_kv", Cmp::Eq, TRUE),
            F::Not(&F::Target(T::Cap("indexer_pooled"))),
        ]),
    ),
    site: "crates/devgen/src/mla.rs: packed sparse prefill requires PLOW_PACKED_SPARSE_PF=1 with PLOW_GLM_INDEX_TP=1, an unpooled indexer and FP8 KV",
    check: Check::Load,
}];

const C_SEQ_PAR: &[Constraint] = &[Constraint {
    id: "seq_par_excludes_two_shot_seams",
    formula: F::Implies(
        &F::Atom("emit.glm_seq_par", Cmp::Eq, TRUE),
        &F::And(&[
            F::Not(&F::Atom("emit.glm_xr_band", Cmp::Gt, Val::Nat(1))),
            F::Not(&F::Atom("emit.glm_xr_res", Cmp::Eq, TRUE)),
        ]),
    ),
    site: "crates/devgen/src/mla.rs: PLOW_GLM_SEQ_PAR cannot combine with PLOW_GLM_XR_RES or PLOW_GLM_XR_BAND",
    check: Check::Load,
}];

const C_SEQ_PAR_PROJ: &[Constraint] = &[Constraint {
    id: "seq_par_proj_requires_seq_par",
    formula: F::Implies(
        &F::Atom("emit.glm_seq_par_proj", Cmp::Eq, TRUE),
        &F::Atom("emit.glm_seq_par", Cmp::Eq, TRUE),
    ),
    site: "crates/devgen/src/mla.rs: PLOW_GLM_SEQ_PAR_PROJ extends PLOW_GLM_SEQ_PAR; set both",
    check: Check::Load,
}];

const C_TOKEN_BATCH_TP: &[Constraint] = &[Constraint {
    id: "token_batch_tp_excludes_seq_par",
    formula: F::Not(&F::And(&[
        F::Atom("emit.token_batch_tp", Cmp::Eq, TRUE),
        F::Atom("emit.glm_seq_par", Cmp::Eq, TRUE),
    ])),
    site: "review log #63: token-batch bodies with seams corrupted long-context prefill",
    check: Check::Load,
}];

const C_K3_WALK: &[Constraint] = &[Constraint {
    id: "k3_wide_decode_requires_walk",
    formula: F::Implies(
        &F::And(&[
            F::Target(T::Model("kimi_k3")),
            F::Atom("emit.decode_batch", Cmp::Gt, Val::Nat(16)),
            F::Atom("emit.decode_ladder", Cmp::Eq, Val::Unset),
        ]),
        &F::Atom("emit.gemv_walk", Cmp::Eq, TRUE),
    ),
    site: "crates/devgen/src/mla/kimi_k3.rs: K3 PLOW_DECODE_BATCH requires PLOW_GEMV_WALK=1 above 16 rows",
    check: Check::Load,
}];

/// The GLM-5.3-FP8 TP8 serving recipe (`build.json` `emit_config.replay` plus the replay harness's
/// environment).
const GLM53_RECIPE: &[(&str, Val)] = &[
    ("emit.fp8", TRUE),
    ("emit.decode_ladder", Val::Str("1,2,4,8,16,20")),
    ("emit.emit_packed_prefill", FALSE),
    ("emit.uniseg", FALSE),
    ("emit.mla_prefill", Val::Str("full:128,512,2048,8192")),
    ("emit.moe_pf_det", TRUE),
    ("emit.glm_dsa", Val::Str("1")),
    ("emit.glm_shard_head", TRUE),
    ("emit.glm_moe_coresident", Val::Nat(2)),
    ("emit.glm_shared_cus", Val::Nat(48)),
    ("emit.glm_fuse_b1", TRUE),
    ("emit.glm_fuse_seam", TRUE),
    ("emit.glm_fuse_rope", FALSE),
    ("emit.glm_dsa_pf", TRUE),
    ("emit.glm_fp8_kv", TRUE),
    ("emit.glm_moe_aiter", TRUE),
    ("emit.glm_moe_flat_decode", FALSE),
    ("emit.glm_moe_resident", TRUE),
    ("emit.glm_index_tp", TRUE),
    ("emit.glm_select_local", TRUE),
    ("emit.glm_decode_norm_rows", TRUE),
    ("emit.glm_gemm_lt", TRUE),
    ("emit.glm_gemm_lt_decode", TRUE),
    ("emit.glm_dsa_pf_span", Val::Nat(3)),
    ("emit.glm_place_pf", FALSE),
    ("env.PLOW_UNISEG", Val::Str("0")),
    ("env.PLOW_MLA_PF_V2", Val::Str("1")),
    ("env.PLOW_MLA_PF_AITER", Val::Str("1")),
];

const DENSE_CAPS: &[&str] = &["dense_packet_contracts", "decode_objects", "decode_ladder"];
const GEMMA4_W8A8_RECIPE: &[(&str, Val)] = &[("emit.w8a8", TRUE)];

/// G4 (review log #86): the AITER MoE call accumulates onto the shared partial, so the combine pass
/// leaves the 2048..8192 prefill buckets, and the adapter object is the only object that changes.
/// Measured on the GLM-5.3 recipe: `MoeCombinePf` removed, `MoeAiterFp8Pf` re-pointed at the shared
/// partial (operands, mode immediate, output size) and the shared down `GemmLtPf` (op 158) writing
/// that partial, 75 layers in each of the two programs.
const MOE_SHARED_SEED_SCOPE: &[Allow] = &[
    Allow {
        kinds: &["prefill"],
        rows: (2048, 8192),
        ops: OpSel::In(&["moe", "op:158"]),
        fields: &[
            ScopeField::Op,
            ScopeField::Operands,
            ScopeField::Shape,
            ScopeField::Segments,
            ScopeField::TensorBytes,
        ],
        ..Allow::ANY
    },
    Allow {
        kinds: &["global"],
        fields: &[ScopeField::ObjectFacts],
        facts: &["moe_aiter"],
        ..Allow::ANY
    },
];

/// The small-rung workgroup cap narrows the compute of the small GLM prefill buckets and nothing
/// else: not their collectives (6deb4025 did, and this is the scope that says it may not).
const SMALL_CUS_SCOPE: &[Allow] = &[Allow {
    kinds: &["prefill"],
    rows: (0, 2047),
    topology: Some("ordinary"),
    model: Some("glm_moe_dsa"),
    ops: OpSel::NotIn(&["collective"]),
    fields: &[ScopeField::Cus, ScopeField::Segments],
    ..Allow::ANY
}];

/// The row-split attention arm: attention and the all-to-all in the 2048..8192 prefill buckets,
/// and the packet-wide object facts its arm adds.
const ROWSPLIT_ATTN_SCOPE: &[Allow] = &[
    Allow {
        kinds: &["prefill"],
        rows: (2048, 8192),
        ops: OpSel::In(&["attention", "op:160"]),
        fields: &[
            ScopeField::Op,
            ScopeField::Operands,
            ScopeField::Cus,
            ScopeField::Segments,
            ScopeField::TensorBytes,
            ScopeField::ObjectFacts,
        ],
        ..Allow::ANY
    },
    Allow {
        kinds: &["global"],
        fields: &[ScopeField::ObjectFacts, ScopeField::TensorBytes],
        ..Allow::ANY
    },
];

/// Declared targets. Formulas speak in capabilities (`emit_capabilities`), so a family joins by
/// adding its target here, not by changing checkpoint K.
pub const TARGETS: &[TargetSpec] = &[
    TargetSpec {
        name: "glm53_fp8_gfx942_tp8",
        arch: "gfx942",
        tp: 8,
        n_cu: 304,
        model: "glm_moe_dsa",
        caps: &["packed_prefill_siblings", "glm"],
        recipe: GLM53_RECIPE,
    },
    TargetSpec {
        name: "gemma4_sm90a_tp1",
        arch: "sm_90a",
        tp: 1,
        n_cu: 132,
        model: "gemma4",
        caps: &[
            "gemma",
            "dense_packet_contracts",
            "decode_objects",
            "cublaslt_decode",
            "decode_ladder",
        ],
        recipe: &[],
    },
    TargetSpec {
        name: "gemma4_w8a8_sm90a_tp1",
        arch: "sm_90a",
        tp: 1,
        n_cu: 132,
        model: "gemma4",
        caps: &[
            "gemma",
            "dense_packet_contracts",
            "decode_objects",
            "cublaslt_decode",
            "decode_ladder",
        ],
        recipe: GEMMA4_W8A8_RECIPE,
    },
    TargetSpec {
        name: "gemma4_gfx942_tp1",
        arch: "gfx942",
        tp: 1,
        n_cu: 304,
        model: "gemma4",
        caps: &[
            "gemma",
            "dense_packet_contracts",
            "decode_objects",
            "cublaslt_decode",
            "decode_ladder",
        ],
        recipe: &[],
    },
    TargetSpec {
        name: "qwen3_metal3_tp1",
        arch: "metal3",
        tp: 1,
        n_cu: 10,
        model: "qwen3",
        caps: DENSE_CAPS,
        recipe: &[],
    },
    TargetSpec {
        name: "kimi_k3_gfx950_tp8",
        arch: "gfx950",
        tp: 8,
        n_cu: 256,
        model: "kimi_k3",
        caps: &[],
        recipe: &[],
    },
];

#[rustfmt::skip]
pub const EMIT: &[KnobSpec] = &[
    KnobSpec::new("emit.fp8", Some("PLOW_FP8"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.w8a8", Some("PLOW_W8A8"), Layer::Emit, Domain::Bool, OFF, OPT_IN).with(C_W8A8),
    KnobSpec::new("emit.w8a16", Some("PLOW_W8A16"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.mxfp4", Some("PLOW_MXFP4"), Layer::Emit, Domain::Bool, OFF, OPT_IN).with(C_MXFP4),
    KnobSpec::new("emit.fp8_kv", Some("PLOW_FP8_KV"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fp8_kv_full", Some("PLOW_FP8_KV_FULL"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fp8_head", Some("PLOW_FP8_HEAD"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.mx4_head", Some("PLOW_MX4_HEAD"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.mx4_prefill", Some("PLOW_MX4_PREFILL"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.uniseg", Some("PLOW_UNISEG"), Layer::Emit, Domain::Bool, UNISEG_DEFAULT, UNISEG),
    KnobSpec::new("emit.seg_pure_gemm", Some("PLOW_SEG_PURE_GEMM"), Layer::Emit, Domain::Str, PURE_GEMM_DEFAULT, GEMMA_NATIVE_PURE_GEMM),
    KnobSpec::new("emit.emit_packed_prefill", Some("PLOW_EMIT_PACKED_PREFILL"), Layer::Emit, Domain::Bool, PACKED_PREFILL_DEFAULT, PACKED_SIBLINGS),
    KnobSpec::new("emit.decode_mla_segments", Some("PLOW_SEG_DECODE_MLA"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.decode_grouped_moe_segments", Some("PLOW_SEG_DECODE_GROUPED_MOE"), Layer::Emit, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("emit.decode_batch", Some("PLOW_DECODE_BATCH"), Layer::Emit, U32, Default::Static(Val::Nat(1)), OPT_IN),
    KnobSpec::new("emit.decode_ladder", Some("PLOW_DECODE_BATCH_LADDER"), Layer::Emit, Domain::Str, DECODE_LADDER_DEFAULT, DECODE_LADDER),
    KnobSpec::new("emit.decode_objects", None, Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.decode_projection_tuning", None, Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.max_chunk", Some("PLOW_MAX_CHUNK"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.max_request_chunk", Some("PLOW_MAX_REQUEST_CHUNK"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.gemv_split", Some("PLOW_GEMV_SPLIT"), Layer::Emit, U32, Default::Static(Val::Nat(1)), OPT_IN),
    KnobSpec::new("emit.decode_tiled", Some("PLOW_DECODE_TILED"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.l2_place_prefill", Some("PLOW_L2_PLACE_PREFILL"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.fuse_argmax", Some("PLOW_FUSE_ARGMAX"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.no_fuse_qkv", Some("PLOW_NO_FUSE_QKV"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fuse_qkv_fp8", Some("PLOW_FUSE_QKV_FP8"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.no_fuse_nrn", Some("PLOW_NO_FUSE_NRN"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fuse_hnr", Some("PLOW_FUSE_HNR"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fuse_merge", Some("PLOW_FUSE_MERGE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.hn_split", Some("PLOW_HN_SPLIT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fa_gf_full", Some("PLOW_FA_GF_FULL"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.attention_decode_balance_gf", Some("PLOW_ATTENTION_DECODE_BALANCE_GF"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.flash_merge_dsplit", Some("PLOW_FLASH_MERGE_DSPLIT"), Layer::Emit, U32, UNSET, DIAG),
    KnobSpec::new("emit.ns_mul", Some("PLOW_NS_MUL"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.ns_abs", Some("PLOW_NS_ABS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.ns_full_abs", Some("PLOW_NS_FULL_ABS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.pf_ladder", Some("PLOW_PF_LADDER"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.pf_ladder_append", Some("PLOW_PF_LADDER_APPEND"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.pf_gemv_head", Some("PLOW_PF_GEMV_HEAD"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.xr_cus", Some("PLOW_XR_CUS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_pf_small_cus", Some("PLOW_GLM_PF_SMALL_CUS"), Layer::Emit, Domain::Str, UNSET, OPT_IN).scoped(SMALL_CUS_SCOPE),
    KnobSpec::new("emit.xr_dec_cus", Some("PLOW_XR_DEC_CUS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.xr2_gather", Some("PLOW_XR2_GATHER"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.no_xreduce", Some("PLOW_NO_XREDUCE"), Layer::Emit, Domain::Bool, OFF, DIAG),
    KnobSpec::new("emit.moe_prefill", Some("PLOW_MOE_PREFILL"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.gemma_moe_router_fused", Some("PLOW_GEMMA_MOE_ROUTER_FUSED"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemma_moe_router_blocks", Some("PLOW_GEMMA_MOE_ROUTER_BLOCKS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.gemma_moe_router_exact", Some("PLOW_GEMMA_MOE_ROUTER_EXACT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemma_moe_tail_fuse", Some("PLOW_GEMMA_MOE_TAIL_FUSE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.k3_full", Some("K3_FULL"), Layer::Emit, Domain::Bool, ON, DIAG),
    KnobSpec::new("emit.k3_fuse_a", Some("PLOW_K3_FUSE_A"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.mla_ns", Some("PLOW_MLA_NS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.legacy_k3_ns", Some("PLOW_K3_NS"), Layer::Emit, U32, UNSET, DIAG),
    KnobSpec::new("emit.legacy_glm_ns", Some("PLOW_GLM_NS"), Layer::Emit, U32, UNSET, DIAG),
    KnobSpec::new("emit.layers", Some("PLOW_LAYERS"), Layer::Emit, Domain::Str, UNSET, DIAG),
    KnobSpec::new("emit.legacy_k3_layers", Some("PLOW_K3_LAYERS"), Layer::Emit, Domain::Str, UNSET, DIAG),
    KnobSpec::new("emit.legacy_glm_layers", Some("PLOW_GLM_LAYERS"), Layer::Emit, Domain::Str, UNSET, DIAG),
    KnobSpec::new("emit.k3_prefill", Some("K3_PREFILL"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_dsa", Some("PLOW_GLM_DSA"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_gf", Some("PLOW_GLM_GF"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_shard_head", Some("GLM_SHARD_HEAD"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_moe_coresident", Some("GLM_MOE_CORESIDENT"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_shared_cus", Some("GLM_SHARED_CUS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_spine_cus", Some("GLM_SPINE_CUS"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_linear_fp8", Some("GLM_LINEAR_FP8"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_shared_glu_split", Some("GLM_SHARED_GLU_SPLIT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.mla_prefill", Some("PLOW_MLA_PREFILL"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_ep", Some("GLM_EP"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_group", Some("GLM_GROUP"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_fuse_b1", Some("PLOW_GLM_FUSE_B1"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_fuse_seam", Some("PLOW_GLM_FUSE_SEAM"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_fuse_rope", Some("PLOW_GLM_FUSE_ROPE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_fuse_qnorm", Some("PLOW_GLM_FUSE_QNORM"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_router_off_shared", Some("GLM_ROUTER_OFF_SHARED"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_router_old", Some("GLM_ROUTER_OLD"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.k3_fuse_ngemv", Some("PLOW_K3_FUSE_NGEMV"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.k3_kda_conv_step_db", Some("PLOW_K3_KDA_CONV_STEP_DB"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.kda_decode_fused", Some("PLOW_KDA_DECODE_FUSED"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.mla_materialized_prefill", Some("PLOW_MLA_MATERIALIZED_PREFILL"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.kda_chunk", Some("PLOW_KDA_CHUNK"), Layer::Emit, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("emit.kda_chunk_qpre", Some("PLOW_KDA_CHUNK_QPRE"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.kda_intra_wave_items", Some("PLOW_KDA_INTRA_WAVE_ITEMS"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.kda_carry_regstate", Some("PLOW_KDA_CARRY_REGSTATE"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.kda_key_factor", Some("PLOW_KDA_KEY_FACTOR"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.kda_wu_lean", Some("PLOW_KDA_WU_LEAN"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.kda_carry_keyfeed", Some("PLOW_KDA_CARRY_KEYFEED"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.k3_shard_head", Some("PLOW_K3_SHARD_HEAD"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.k3_seq_rows", Some("PLOW_K3_SEQ_ROWS"), Layer::Emit, Domain::Bool, OFF, DIAG),
    KnobSpec::new("emit.gemv_mm", Some("PLOW_GEMV_MM"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.gemv_walk", Some("PLOW_GEMV_WALK"), Layer::Emit, Domain::Bool, OFF, OPT_IN).with(C_K3_WALK),
    KnobSpec::new("emit.dec_stage_halves", Some("PLOW_DEC_STAGE_HALVES"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.fuse_residual_input", Some("PLOW_FUSE_RESIDUAL_INPUT"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.k3_fuse_arnorm", Some("PLOW_K3_FUSE_ARNORM"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.qnorm_fuse", Some("PLOW_QNORM_FUSE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glu_quant_fuse", Some("PLOW_GLU_QUANT_FUSE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fuse_quant", Some("PLOW_FUSE_QUANT"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.gemv_wg", Some("PLOW_GEMV_WG"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.gemv_wg_tuning", Some("PLOW_GEMV_WG_TUNING"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_dsa_pf", Some("PLOW_GLM_DSA_PF"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_fp8_kv", Some("PLOW_GLM_FP8_KV"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_moe_aiter", Some("PLOW_GLM_MOE_AITER"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_moe_flat_decode", Some("PLOW_GLM_MOE_FLAT_DECODE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_mla_dec_aiter", Some("PLOW_GLM_MLA_DEC_AITER"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_moe_resident", Some("PLOW_GLM_MOE_RESIDENT"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_moe_shared_fold", Some("PLOW_GLM_MOE_SHARED_FOLD"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_moe_shared_seed", Some("PLOW_GLM_MOE_SHARED_SEED"), Layer::Emit, Domain::Bool, OFF, MOE_SHARED_SEED_CANDIDATE).scoped(MOE_SHARED_SEED_SCOPE),
    KnobSpec::new("emit.token_batch_tp", Some("PLOW_TOKEN_BATCH_TP"), Layer::Emit, Domain::Bool, OFF, TOKEN_BATCH_TP_PARKED).with(C_TOKEN_BATCH_TP),
    KnobSpec::new("emit.packed_sparse_pf", Some("PLOW_PACKED_SPARSE_PF"), Layer::Emit, Domain::Bool, OFF, OPT_IN).with(C_PACKED_SPARSE_PF),
    KnobSpec::new("emit.glm_index_tp", Some("PLOW_GLM_INDEX_TP"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_select_local", Some("PLOW_GLM_SELECT_LOCAL"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_select_split", Some("PLOW_GLM_SELECT_SPLIT"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_decode_norm_rows", Some("PLOW_GLM_DECODE_NORM_ROWS"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_gemm_lt", Some("PLOW_GLM_GEMM_LT"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_gemm_lt_decode", Some("PLOW_GLM_GEMM_LT_DECODE"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_gemm_lt_decode_ext", Some("PLOW_GLM_GEMM_LT_DECODE_EXT"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_fold_lt", Some("PLOW_GLM_FOLD_LT"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_gemm_blk", Some("PLOW_GLM_GEMM_BLK"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_gemm_lt_pf_ext", Some("PLOW_GLM_GEMM_LT_PF_EXT"), Layer::Emit, Domain::Str, GLM_GEMM_LT_PF_EXT_DEFAULT, GLM_RECIPE_PF_EXT),
    KnobSpec::new("emit.glm_gemv_wg", Some("PLOW_GLM_GEMV_WG"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_ofold", Some("PLOW_GLM_OFOLD"), Layer::Emit, Domain::Bool, OFF, OPT_IN).with(C_OFOLD),
    KnobSpec::new("emit.glm_pf_ns", Some("PLOW_GLM_PF_NS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_dsa_pf_span", Some("PLOW_GLM_DSA_PF_SPAN"), Layer::Emit, U32, Default::Static(Val::Nat(1)), OPT_IN),
    KnobSpec::new("emit.glm_dsa_pf_dexact", Some("PLOW_GLM_DSA_PF_DEXACT"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.pf_floor", Some("PLOW_PF_FLOOR"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.dense_pf_ns", Some("PLOW_DENSE_PF_NS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_pf_wide", Some("PLOW_GLM_PF_WIDE"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.glm_place_pf", Some("PLOW_GLM_PLACE_PF"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_xr_band", Some("PLOW_GLM_XR_BAND"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_xr_band_cus", Some("PLOW_GLM_XR_BAND_CUS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.attnres_decode_mwg", Some("PLOW_ATTNRES_DECODE_MWG"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.glm_xr_band_seam", Some("PLOW_GLM_XR_BAND_SEAM"), Layer::Emit, Domain::Str, UNSET, DIAG),
    KnobSpec::new("emit.glm_xr_res", Some("PLOW_GLM_XR_RES"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_seq_par", Some("PLOW_GLM_SEQ_PAR"), Layer::Emit, Domain::Bool, GLM_SEQ_PAR_DEFAULT, GLM_RECIPE).with(C_SEQ_PAR),
    KnobSpec::new("emit.glm_seq_par_proj", Some("PLOW_GLM_SEQ_PAR_PROJ"), Layer::Emit, Domain::Bool, GLM_SEQ_PAR_PROJ_DEFAULT, GLM_RECIPE).with(C_SEQ_PAR_PROJ),
    KnobSpec::new("emit.glm_rowsplit_attn", Some("PLOW_GLM_ROWSPLIT_ATTN"), Layer::Emit, Domain::Bool, UNSET, OPT_IN).scoped(ROWSPLIT_ATTN_SCOPE),
    KnobSpec::new("emit.glm_decode_glue_cus", Some("PLOW_GLM_DECODE_GLUE_CUS"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.glm_decode_gemm_group", Some("PLOW_GLM_DECODE_GEMM_GROUP"), Layer::Emit, Domain::Bool, GLM_RECIPE_ON, GLM_RECIPE),
    KnobSpec::new("emit.glm_fuse_xrn", Some("GLM_FUSE_XRN"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.xr_combine_fold", Some("PLOW_XR_COMBINE_FOLD"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.kda_fb_fold", Some("PLOW_KDA_FB_FOLD"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.kda_decode_fused_arm", Some("PLOW_KDA_DECODE_FUSED_ARM"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemv_prefetch", Some("PLOW_GEMV_PREFETCH"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.moe_stage2_lean", Some("PLOW_MOE_STAGE2_LEAN"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.moe_stage1_lean", Some("PLOW_MOE_STAGE1_LEAN"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.moe_combine_lean", Some("PLOW_MOE_COMBINE_LEAN"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.attnres_f32mix", Some("PLOW_ATTNRES_F32MIX"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.moe_align_par", Some("PLOW_MOE_ALIGN_PAR"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.seq_par_seams", Some("PLOW_SEQ_PAR_SEAMS"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.moe_prefill_ep", Some("PLOW_MOE_PREFILL_EP"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.moe_pf_det", Some("PLOW_MOE_PF_DET"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.moe_stage1_body", Some("PLOW_MOE_STAGE1_BODY"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.moe_stage2_body", Some("PLOW_MOE_STAGE2_BODY"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.no_glu_fuse", Some("PLOW_NO_GLU_FUSE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.tma_gemm", Some("PLOW_TMA_GEMM"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemma4_sm90_gemm_glu_role", Some("PLOW_GEMMA4_SM90_GEMM_GLU_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemma4_sm90_w8a8_gemm_glu_role", Some("PLOW_GEMMA4_SM90_W8A8_GEMM_GLU_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemma4_sm90_hd256_gqa2_role", Some("PLOW_GEMMA4_SM90_HD256_GQA2_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemma4_sm90_hd512_px4_bq64_role", Some("PLOW_GEMMA4_SM90_HD512_PX4_BQ64_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fp8_pf_gemm_role", Some("PLOW_FP8_PF_GEMM_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.fp8_pf_isolate", Some("PLOW_QWEN_FP8_PF_ISOLATE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.attention_pf_role", Some("PLOW_ATTENTION_PF_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.attention_pf_isolate", Some("PLOW_ATTENTION_PF_ISOLATE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.gemv_decode_role", Some("PLOW_GEMV_DECODE_ROLE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_fp8_m1_tma", Some("PLOW_QWEN_FP8_M1_TMA"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_w8a8_prefill", Some("PLOW_QWEN_W8A8_PREFILL"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.decode_cublaslt", Some("PLOW_EMIT_DECODE_CUBLASLT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.prefill_cublaslt", Some("PLOW_EMIT_PREFILL_CUBLASLT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.decode_native_tc", Some("PLOW_EMIT_DECODE_NATIVE_TC"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_fuse_ab", Some("PLOW_QWEN_FUSE_AB"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_fuse_mlp", Some("PLOW_QWEN_FUSE_MLP"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_projection_dag", Some("PLOW_QWEN_PROJECTION_DAG"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_share_quant", Some("PLOW_QWEN_SHARE_QUANT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.qwen_ab_blocks", Some("PLOW_QWEN_AB_BLOCKS"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.qwen_prefill", Some("PLOW_QWEN_PREFILL"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.pf_gfuse", Some("PLOW_PF_GFUSE"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.uniseg_max_t", Some("PLOW_UNISEG_MAX_T"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.row_split", Some("PLOW_ROW_SPLIT"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.ane_mlp_channels", Some("PLOW_ANE_MLP_CHANNELS"), Layer::Emit, U32, UNSET, OPT_IN).with(C_CHANNEL_MLP),
    KnobSpec::new("emit.glm_wgfit", Some("PLOW_GLM_WGFIT"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.tunedb", Some("PLOW_TUNEDB"), Layer::Emit, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("emit.gemm_wide_c8", Some("PLOW_GEMM_WIDE_C8"), Layer::Emit, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("emit.audit_occ_floor", Some("PLOW_AUDIT_OCC_FLOOR"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.audit_gemv_waste_max", Some("PLOW_AUDIT_GEMV_WASTE_MAX"), Layer::Emit, U32, UNSET, OPT_IN),
    KnobSpec::new("emit.audit_strict", Some("PLOW_AUDIT_STRICT"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.tune_dump", Some("PLOW_TUNE_DUMP"), Layer::Emit, Domain::Bool, OFF, DIAG),
    KnobSpec::new("emit.skip_coverage", Some("PLOW_SKIP_COVERAGE"), Layer::Emit, Domain::Bool, OFF, DIAG),
    KnobSpec::new("emit.k3_ablate", Some("PLOW_K3_ABLATE"), Layer::Emit, Domain::Str, UNSET, DIAG),
];

#[rustfmt::skip]
pub const RAW_ENV: &[KnobSpec] = &[
    KnobSpec::new("env.GLM_FULL", Some("GLM_FULL"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.GLM_LAYER", Some("GLM_LAYER"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.GLM_NLAYERS", Some("GLM_NLAYERS"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_AMD_DECODE_GEMM_OVERLAP", Some("PLOW_AMD_DECODE_GEMM_OVERLAP"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_ATTNRES_MAXB", Some("PLOW_ATTNRES_MAXB"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_AUDIT_JOBS", Some("PLOW_AUDIT_JOBS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_BLOB_F_L2DOM", Some("PLOW_BLOB_F_L2DOM"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_BLOCK", Some("PLOW_BLOCK"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_FA512", Some("PLOW_BUILD_FA512"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_FATLITE", Some("PLOW_BUILD_FATLITE"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_FA_HD256", Some("PLOW_BUILD_FA_HD256"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_FA_HD256_ONLY", Some("PLOW_BUILD_FA_HD256_ONLY"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_FA_WG", Some("PLOW_BUILD_FA_WG"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_GEMM_WS384", Some("PLOW_BUILD_GEMM_WS384"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_TMA_GEMM", Some("PLOW_BUILD_TMA_GEMM"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_BUILD_W8A8", Some("PLOW_BUILD_W8A8"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_CHAIN_BYPASS", Some("PLOW_CHAIN_BYPASS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_DECODE_TIERS", Some("PLOW_DECODE_TIERS"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_FINE_FORCE", Some("PLOW_FINE_FORCE"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_FLASH_HD128", Some("PLOW_FLASH_HD128"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_FP8_FAST", Some("PLOW_FP8_FAST"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_FP8_LD16", Some("PLOW_FP8_LD16"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_FUSE_XR_ATTNRES", Some("PLOW_FUSE_XR_ATTNRES"), Layer::RawEnv, Domain::Str, UNSET, SEG_EXPERIMENT_PARKED),
    KnobSpec::new("env.PLOW_GATE_RELAXSIG", Some("PLOW_GATE_RELAXSIG"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_GEMM_JSONL", Some("PLOW_GEMM_JSONL"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_GLM_FOLD_LT_DECODE", Some("PLOW_GLM_FOLD_LT_DECODE"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_GLM_GF", Some("PLOW_GLM_GF"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_GQ_ORDER", Some("PLOW_GQ_ORDER"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_KV_FP8", Some("PLOW_KV_FP8"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_L2_PLACE", Some("PLOW_L2_PLACE"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_L2_PLACE_PREFILL", Some("PLOW_L2_PLACE_PREFILL"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_MLA_FOLD_TB_FLASH", Some("PLOW_MLA_FOLD_TB_FLASH"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_MLA_PF_AITER", Some("PLOW_MLA_PF_AITER"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_MLA_PF_V2", Some("PLOW_MLA_PF_V2"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_MOE_DECODE_STANDALONE", Some("PLOW_MOE_DECODE_STANDALONE"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_MOE_ENC", Some("PLOW_MOE_ENC"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_NV_ABLATE_HI", Some("PLOW_NV_ABLATE_HI"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_NV_ABLATE_LO", Some("PLOW_NV_ABLATE_LO"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_NV_FA512_BKV", Some("PLOW_NV_FA512_BKV"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_NV_PLACE", Some("PLOW_NV_PLACE"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_NV_ZG", Some("PLOW_NV_ZG"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_PACKED_PREFILL_CONSUMERS", Some("PLOW_PACKED_PREFILL_CONSUMERS"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY", Some("PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_PACKET_HAS_TOKEN_BATCH_BODIES", Some("PLOW_PACKET_HAS_TOKEN_BATCH_BODIES"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_PHASE_OBJECTS", Some("PLOW_PHASE_OBJECTS"), Layer::RawEnv, Domain::Str, UNSET, SEG_EXPERIMENT_PARKED),
    KnobSpec::new("env.PLOW_PLACE_REPORT", Some("PLOW_PLACE_REPORT"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_QWEN_DECODE_LT", Some("PLOW_QWEN_DECODE_LT"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_ROOT", Some("PLOW_ROOT"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SEG_CLASS_SLICE", Some("PLOW_SEG_CLASS_SLICE"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SEG_DUMP", Some("PLOW_SEG_DUMP"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SEG_FA256_GQA2", Some("PLOW_SEG_FA256_GQA2"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SEG_FA512", Some("PLOW_SEG_FA512"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SEG_PACKED_PREFILL", Some("PLOW_SEG_PACKED_PREFILL"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_SEG_PER_OP", Some("PLOW_SEG_PER_OP"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SEG_PURE_GEMM", Some("PLOW_SEG_PURE_GEMM"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SEG_SLICE_ALL", Some("PLOW_SEG_SLICE_ALL"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SEG_V2", Some("PLOW_SEG_V2"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_SOURCE_ROOT", Some("PLOW_SOURCE_ROOT"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TOKEN_BATCH_TP_OBJECTS", Some("PLOW_TOKEN_BATCH_TP_OBJECTS"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_TOOLCHAIN_LABEL", Some("PLOW_TOOLCHAIN_LABEL"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TR_QUIET", Some("PLOW_TR_QUIET"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TUNE_DUMP", Some("PLOW_TUNE_DUMP"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_UNISEG", Some("PLOW_UNISEG"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_W4A16", Some("PLOW_W4A16"), Layer::RawEnv, Domain::Str, UNSET, REMOVED),
    KnobSpec::new("env.PLOW_XR_NOSIG", Some("PLOW_XR_NOSIG"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_XR_SCHED_AG_U", Some("PLOW_XR_SCHED_AG_U"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_XR_SCHED_NWG", Some("PLOW_XR_SCHED_NWG"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_XR_SCHED_NWG_RS", Some("PLOW_XR_SCHED_NWG_RS"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_XR_SCHED_NWG_SAG", Some("PLOW_XR_SCHED_NWG_SAG"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_XR_SCHED_NWG_SRS", Some("PLOW_XR_SCHED_NWG_SRS"), Layer::RawEnv, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("env.PLOW_XR_WAVE_RS", Some("PLOW_XR_WAVE_RS"), Layer::RawEnv, Domain::Str, UNSET, SEG_EXPERIMENT_PARKED),
];

#[rustfmt::skip]
pub const OBJECT_DEFINES: &[KnobSpec] = &[
    KnobSpec::new("def.PLOW_ACT_NT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ACT_SCOPE_AGENT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ARCH_SUFFIX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ARM64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ATTNRES_DECODE_MWG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ATTNRES_F32MIX_BODY_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ATTNRES_RG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ATTNRES_TOKENS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_FP8_ABI", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_GEMMA4_ALL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_GEMMA_QGATE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_GEMM_ODOWN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_GEMV_CTA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_GEMV_M16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_GEMV_M16_BK128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_M64N128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_M64N64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_QWEN_GEMV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BENCH_WS384", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_DECODE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_DECODE_MLA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_FLASH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_MLA_PREFILL_SMALL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_MLA_PREFILL_SPLIT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_MLA_V2_RAW", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_PACKED_KDA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_PACKED_MLA_NORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUCKET_XREDUCE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_BUILD_SEG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CDNA4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_COMBINE_VEC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_COMBINE_VEC_U", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CONFIG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CUBIN_ARCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CUBIN_CONFIG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CUBIN_DIR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CUBIN_GEMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_CUBIN_PACKED_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DECODE_INVENTORY_PRUNE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DEC_ARENA_HALVES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DOP_KDA_CONV3", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DOP_KDA_STATE_STEP_G", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DSA_DECODE_BATCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DSA_IDX64_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DSA_IDX_ROW", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DSA_PF_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_EXPERIMENT_PX4_BQ64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLM_OFOLD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DSA_SELECT_SPLIT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_DSA_TP_TILE_N", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_EXPERIMENT_HD512_PC_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_EXPERIMENT_LAUNCH_SMEM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_EXPERIMENT_PX4_TMA_DESC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_EXTRA_DEFINES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_F2BF_SELECT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA512_PC_CONSUMER_REGS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA512_PC_PRODUCER_REGS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_HD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_KVH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_NH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_RING", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_SCALE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_SHORT_BURST", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_BENCH_WINDOW", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_GF_FULL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_LIBRARY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_SWEEP_BKV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_SWEEP_BQ", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_SWEEP_HD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FA_SWEEP_THREADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FLASH_DECODE_MIXED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FLASH_DECODE_REFERENCE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FLASH_HD128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FP8_FAST", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FP8_KV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FP8_KV_FASTPF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FP8_LD16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_FP8_W8A8_PERSISTENT_PROBE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GATE_HIER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GATE_HIER_CEIL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GATE_NOINV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GATE_RELAXSIG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GATE_SC1", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GATE_SC1_KEEPREL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMM_ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_LG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_LG_RG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_LG_UNR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_MAXM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_MM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_PERK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_PF_MAX_BYTES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_PREFETCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_TRANSPOSE_SWIZZLE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GEMV_WALK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLM_FUSE_QNORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLM_GF8_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLOBAL_QUEUE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLU_K4096_UN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLU_NS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GLU_UN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GM_DIRECT_STAGE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GM_FP8_PACK2", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GQ_BATCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GV_ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GV_NOLDS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GV_UN_BIG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GV_UN_FP8_KDIV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_GV_UN_LEGACY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_ADD_NORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_ARGMAX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_ARGMAX_FIN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_ATTN_RES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_ATTN_SELECT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_DENSE_GLU_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_EMBED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_DECODE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_DECODE_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_GATHER_DECODE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_GATHER_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_HD64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_MERGE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_MLA_DECODE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_MLA_DECODE_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_MLA_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_MLA_PREFILL_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_FLASH_PREFILL_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_C5", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_C5_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_C5_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_GLU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_GLU_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_GLU_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_MED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_MED_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_MED_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_SMALL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_SMALL_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_SMALL_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_WIDE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_WIDE_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMM_WIDE_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_GLU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_GLU_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_QKV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_QKVG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GEMV_QKV_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_GLU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_HEADNORM_HD128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_HEADNORM_HD256", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_HEADNORM_HD512", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_HEADNORM_HD64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_HEADNORM_ROPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_INDEX_SCORE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_INDEX_SCORE_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_INDEX_SELECT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_INDEX_SELECT_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_INDEX_UNION_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CHUNK_CARRY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CHUNK_INTRA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CHUNK_PREPARE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CHUNK_WU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CONV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CONV3", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_CONV_STATE_STEP_G", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_GATE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_GATED_NORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_STATE_STEP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_KDA_STATE_STEP_G", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_LAYERNORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MLA_MERGE_FOLD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MLA_OUT_GATE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_ALIGN_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_COMBINE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_COMBINE_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_DOWN_MX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_DOWN_MX_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_EXPERT_DOWN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_EXPERT_DOWN_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_EXPERT_GLU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_EXPERT_GLU_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_EXPERT_GLU_GEMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_GEMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_GLU_MX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_GLU_MX_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_GROUP_DOWN_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_GROUP_GLU_FP8_BLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_ROUTER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_ROUTER_GEMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_ROUTER_TOPK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MOE_ROUTER_TOPK_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MX_CVT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_MX_MMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_NORM_RESIDUAL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_NORM_RESIDUAL_NORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_O_UV_FOLD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_QUANT_FP8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_RESIDUAL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_RMSNORM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_ROWRMS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_SITU_GLU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_SOFTCAP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_XALLGATHER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_XALLTOALL_HEADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HAS_XREDUCESCATTER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HC_SINKHORN_RCP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HSACO_CONFIG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HSACO_EXTRA_DEFINES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HSACO_GQ", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HSACO_KDA_KEY_FACTOR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_HSACO_PACKED_PREFILL_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ICACHE_INV_PROBE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_INST_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_K3", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_K3_DECODE_GROUPED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_CHUNK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_CHUNK_QPRE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_CONV_STEP_DB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_DECODE_FUSED_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_FB_CB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_FB_FOLD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_PF_STATE_RESIDENT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_KDA_SOFTPLUS_FLA_COMPAT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_L2_PLACE_DISPATCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_LDS_MAX_BYTES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MATERIALIZED_RESIDUAL_INPUT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MIXED_STEP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_DEC_MS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_DVT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_MAP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_TB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_TB_LDS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_UN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_VEC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_FOLD_VT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF2_ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF2_DBUF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF2_NOPE_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_KSPLIT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_MFMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_NOPE_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_PSWZ", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_QK1", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_SMX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_SV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_TR16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_V2_ARM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PF_WPM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MLA_SPARSE_SINGLE_PASS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE1_BODY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE1_EXPERTS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE1_INTER_DIM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE1_WAVES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE1_XCD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE2_BODY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE2_EP_FULL_I", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE2_EP_INTER_DIM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE2_NT_STORE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE2_XCD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_A4W4_STAGE2_BENCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_ALIGN_PAR_PREFIX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_BAD_SCALE_EPILOGUE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_BENCH_DECODE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_COMBINE_ALLBLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DEC_ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DEC_LG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DEC_LG_RG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DEC_LG_UNR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DEC_X2", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DEC_X2_UN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DOWN_LANESPLIT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DOWN_SG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_DOWN_STAGE_FU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_GEMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_GEMMA_PF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_GROUP_FLAT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_GROUP_FORCEINLINE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_MFMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4_BRIDGE_ALIAS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4_C3_BK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4_DIRECT_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4_LOWREG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4_PRIO", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_A4W4_WEIGHT_NT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_ATOMIC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_DET", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_EPI", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_EPI_SIB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_GH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_PIPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_SCHED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_SITU_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PF_XCD_WGM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_ROUTER_SELECT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_ROUTER_SELECT_LOCAL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_ROUTER_WIDE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_SLOT_MAJOR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_TILE_BINSEARCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_TREE_SLOTS_PER_LEAF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_TREE_THREADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_XN_BF16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MOE_XN_MAX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MXFP4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_MXFP4_DEC_NT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NORM_RANGE_CHECK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NORM_SS_MAX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NO_MLA_DEC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NSTAGE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_ABLATE_HI", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_ABLATE_LO", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_ARENA_MIN_BYTES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_BF16_T", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_DSA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_EMBED_SMEM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA128_BKV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA256_BKV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_BKV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_FIXED_HEADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_KV16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_KV64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_N_SPLIT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_PC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_PX4_BQ64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_QK_HALVES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA512_WG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FATLITE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_CORRSKIP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_FP8ABL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_FP8MMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_FP8PV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_GF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_GF16_BENCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_GF_FULL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_GF_HD256", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_GQA2_PAIR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_KUN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_ONLY_HD256", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_ONLY_HD256_EXACT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_ONLY_HD256_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_PIPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_PX4", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_QGLOB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_QK_UNROLL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_QREG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_REDBOUND", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_ROPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new(
        "def.PLOW_NV_FA_SCORE_SWIZZLE",
        None,
        Layer::ObjectDefine,
        Domain::Str,
        UNSET,
        OPT_IN,
    ),
    KnobSpec::new("def.PLOW_NV_FA_SPLIT_OUTER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_TC_GQA8_HD512", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_TMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_TMA_DESC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_TMA_ROW_WARP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_VDBUF", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_WGITEM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_WGITEM_ONE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_WPR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FA_WPR_RB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FORCE_MINBLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_DECODE_MMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_DECODE_WGMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_DECODE_WGMMA_ACTIVE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_BK1024", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_BK256", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_BK512", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_BLOCKED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_FAST_ACCUM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_PIPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_PROMOTE_K512", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_ROLE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_TMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_M1_XCACHE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_PF_SCALE_WFIRST", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FP8_RB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_FUTURE_OP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GATE_SLEEP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GDN_STEP_VROWS8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMMA3", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMMA_GLU_BF16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMMA_HNR_BF16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMMA_NRN_BF16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMM_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMM_SPLITK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV512_ROLE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_KPANEL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_KPANEL_F32", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_LS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_M16_ARENA_BYTES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_M16_BK128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_M16_MMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_M16_PIPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_NOSTAGE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_RB", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_STAGE_MINROWS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_STAGING_BYTES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GEMV_XREG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_GF8_TWIN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_HOPPER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_KVBOUNDS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_LANE_MASK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_LEAN_DECODE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MAMBA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MASKED_PADDING", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MINBLK", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MLA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MOE_COMMON", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MXFP4_MOE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_MXFP4_PROJ", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_NRN_WPR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_OP_MOE_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_OP_MOE_SM90_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PACKED_FA_TMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PACKED_FA_WGMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PACKED_REQUEST", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PF_GEMV_HEAD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PLACE_DISPATCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_PTXSYNC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_QUANT_FP8_VLLM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_QUANT_WPR", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_QWEN_GDN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_RB_GEMV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_RB_LMHEAD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_RB_QKV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SCHED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEGMENTS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_GEMM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_GEMM_BN64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_M128N128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_M64N128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_M64N64", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_NOGLU", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_OCC1", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_SMALL_BF16", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_WS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_WS384", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SEG_WS_ENTRY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SKELETON", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SKEL_PAD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_SZ", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_T17_MIN_ROWS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_THREADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_TMA_GEMM", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_TRACE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_W8A16_ASYNC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_W8A16_PREFETCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_W8A16_WGMMA", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_NV_W8A8", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_OP_ATTENTION_ARCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_OP_ATTENTION_SM90_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_OP_DSA_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_OP_GEMM_ARCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_OP_GEMM_SM90_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_OP_MLA_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_ANY_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_BAND", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_DENSE_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_KDA_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_MLA_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKET_HASH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_PACKET_LINEAR_BIAS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_QUANT_SCALE_EXP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_QWEN_GDN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_QWEN_GDN_VROWS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_RESID_U", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_RN_ROWS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_ROWSPLIT_A2A", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SEG_FA512", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SEQ_PAR_SEAMS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SM120_CUBIN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SM120_CUBIN_FP8KV", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SM120_CUBIN_SEG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SM120_SMS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SM90A_CUBIN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SM90_WGMMA_CUH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_SMP_THREADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_STAGE1_MIN_OCC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_STAGE1_WG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TB_DEVICE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TEST_ARCH_SUFFIX", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TEST_FA_HD256_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TEST_FA_ROWS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TEST_FP8_PREFILL", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_THREADS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TK_SAFE_ONLY", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TOKEN_BATCH", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TOKEN_BATCH_DESC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TRACE_PHASE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_TRY_FENCE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_WAVE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_WAVES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_WAVE_RED_DPP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_WG_WAVES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_WPE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_X86", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XA2A_NWG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XCTR_DEADLINE_TICKS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR2_SKIP_AG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR2_SKIP_RS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_ACQ_N", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_AGG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_AGG_ON", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_ATTNRES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_COMBINE_FOLD", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_MLP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_MLP_ON", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_NOSIG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_NOWAIT", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_NOWAIT_RS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_RS_U", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_AG_U", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_AG_WAVE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_AITER", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_CAP", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_NWG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_NWG_RS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_NWG_SAG", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_NWG_SRS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_ON", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_RS_U", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SCHED_SEAM_CAPS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_SHUFFLE", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_TAGGED", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_TRACE_PHASES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_WAVE_RS", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("def.PLOW_XR_WAVE_RS_ON", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),
];

/// The specs checkpoint K resolves at emit. Object defines carry no default and no emit-time
/// constraint, so they stay out of the payload; the digest covers them.
pub fn emit_specs() -> Vec<&'static KnobSpec> {
    EMIT.iter().chain(RAW_ENV).collect()
}

pub fn registry_digest() -> &'static str {
    static DIGEST: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DIGEST.get_or_init(|| {
        let specs: Vec<&KnobSpec> = EMIT.iter().chain(RAW_ENV).chain(OBJECT_DEFINES).collect();
        let constraints = plow_asset::knob::constraints_of(specs.iter().copied());
        plow_asset::knob::registry_digest(&specs, &constraints, TARGETS)
    })
}

type Verifier = Box<dyn Fn(&Value) -> Result<String, String> + Send>;

enum Policy {
    Unset,
    Disabled(String),
    Verify(Verifier),
}

#[derive(Clone)]
struct Verdict {
    verified: bool,
    notes: Option<String>,
    reason: Option<String>,
}

impl Verdict {
    fn skipped(reason: impl Into<String>) -> Verdict {
        Verdict {
            verified: false,
            notes: None,
            reason: Some(reason.into()),
        }
    }
}

struct Gate {
    policy: Policy,
    caps: Vec<String>,
    target: Option<Target>,
    block: Option<String>,
    verdict: Option<Verdict>,
}

#[cfg(not(test))]
static GATE: std::sync::Mutex<Gate> = std::sync::Mutex::new(Gate {
    policy: Policy::Unset,
    caps: Vec::new(),
    target: None,
    block: None,
    verdict: None,
});
#[cfg(test)]
thread_local! {
    static TEST_GATE: std::cell::RefCell<Gate> = const { std::cell::RefCell::new(Gate {
        policy: Policy::Unset,
        caps: Vec::new(),
        target: None,
        block: None,
        verdict: None,
    }) };
}

fn with_gate<R>(f: impl FnOnce(&mut Gate) -> R) -> R {
    #[cfg(not(test))]
    return f(&mut GATE.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(test)]
    return TEST_GATE.with_borrow_mut(f);
}

/// Checkpoint K for every later emit in this process. `verify` returns the certificate notes, or
/// the rejection.
pub fn install_verifier(verify: impl Fn(&Value) -> Result<String, String> + Send + 'static) {
    with_gate(|g| g.policy = Policy::Verify(Box::new(verify)));
}

/// Skip checkpoint K, recorded in `build.json` as `knobs.K = "skipped"` with `reason`.
pub fn disable(reason: impl Into<String>) {
    with_gate(|g| g.policy = Policy::Disabled(reason.into()));
}

/// A target fact only the caller knows (`plowc --segmented`).
pub fn note_cap(cap: &str) {
    with_gate(|g| {
        if !g.caps.iter().any(|c| c == cap) {
            g.caps.push(cap.into());
        }
    });
}

pub(crate) fn note_target(
    arch: &str,
    tp: u32,
    n_cu: u32,
    model: &str,
    caps: Vec<&str>,
    block: Option<&str>,
) {
    with_gate(|g| {
        let mut all: Vec<String> = caps.into_iter().map(String::from).collect();
        all.extend(g.caps.iter().cloned());
        g.target = Some(Target {
            name: String::new(),
            arch: arch.into(),
            tp,
            n_cu,
            model: model.into(),
            caps: all,
        });
        g.block = block.map(String::from);
        g.verdict = None;
    });
}

/// A recorded value typed by its knob's domain. Text the domain cannot carry stays a string, which
/// checkpoint K reports as out of domain.
fn typed<'a>(spec: &KnobSpec, value: &'a str, explicit: bool) -> Val<'a> {
    if value.is_empty() && (!explicit || spec.domain != Domain::Str) {
        return Val::Unset;
    }
    spec.domain.parse(value).unwrap_or(Val::Str(value))
}

fn raw_env_value(spec: &KnobSpec, block: Option<&str>) -> Option<String> {
    let env = spec
        .env
        .and_then(|e| std::env::var(e).ok())
        .filter(|v| !v.is_empty());
    if spec.id == "env.PLOW_BLOCK" {
        block.map(String::from).or(env)
    } else {
        env
    }
}

/// The checkpoint K payload for the recorded knobs and the noted target, or `None` before a
/// target is noted.
pub fn payload() -> Option<Value> {
    let (target, block) = with_gate(|g| (g.target.clone(), g.block.clone()));
    let target = target?;
    let knobs = crate::emit_config::knobs_or_env();
    let raw: Vec<(&KnobSpec, Option<String>)> = RAW_ENV
        .iter()
        .map(|k| (k, raw_env_value(k, block.as_deref())))
        .collect();
    let mut sources = Vec::with_capacity(knobs.len() + raw.len());
    let mut recorded = Vec::with_capacity(knobs.len());
    for k in &knobs {
        let Some(spec) = EMIT.iter().find(|s| s.name() == k.id) else {
            continue;
        };
        let cli = (k.source == "cli").then(|| typed(spec, &k.value, true));
        let env = if k.source == "env" {
            Some(typed(spec, &k.value, true))
        } else {
            k.env_value.as_deref().map(|v| typed(spec, v, true))
        };
        sources.push((spec.id, Source { cli, env }));
        recorded.push(
            json!({"id": spec.id, "value": typed(spec, &k.value, k.source != "default").to_json()}),
        );
    }
    for (spec, v) in &raw {
        let env = v.as_deref().map(Val::Str);
        sources.push((spec.id, Source { cli: None, env }));
    }
    let specs = emit_specs();
    let constraints = plow_asset::knob::constraints_of(specs.iter().copied());
    let mut p = plow_asset::knob::registry_json(&specs, &constraints, TARGETS);
    p["target"] = target.to_json();
    p["sources"] = plow_asset::knob::sources_json(sources);
    p["recorded"] = Value::Array(recorded);
    Some(p)
}

/// Checkpoint K, called by `apply_verify_gate` before the blob is written. A rejection panics:
/// nothing reaches disk.
pub(crate) fn gate() {
    let run = with_gate(|g| match &g.policy {
        Policy::Unset => Err("no knob verifier installed (library emit path)".to_string()),
        Policy::Disabled(reason) => Err(reason.clone()),
        Policy::Verify(_) => Ok(()),
    });
    let verdict = match run.map(|()| payload()) {
        Err(reason) => Verdict::skipped(reason),
        Ok(None) => Verdict::skipped("no emit target was noted before the gate"),
        Ok(Some(p)) => match with_gate(|g| match &g.policy {
            Policy::Verify(verify) => verify(&p),
            _ => unreachable!("policy checked above"),
        }) {
            Ok(notes) => Verdict {
                verified: true,
                notes: Some(notes),
                reason: None,
            },
            Err(e) => panic!("checkpoint K rejected the knob configuration: {e}"),
        },
    };
    with_gate(|g| g.verdict = Some(verdict));
}

/// `build.json` `knobs`: the K verdict, the target, and the values plowrt's load evaluator reads
/// (`plow_asset::knob_gen::KNOBS`, runtime knobs excepted).
pub(crate) fn manifest_section(arch: &str, n_cu: u32, backends: &Value) -> Value {
    let (verdict, target, block) =
        with_gate(|g| (g.verdict.clone(), g.target.clone(), g.block.clone()));
    let verdict =
        verdict.unwrap_or_else(|| Verdict::skipped("checkpoint K did not run for this blob"));
    let target = target.unwrap_or_else(|| Target {
        arch: arch.into(),
        n_cu,
        ..Target::default()
    });
    let knobs = crate::emit_config::knobs_or_env();
    let requires: Vec<&str> = backends
        .as_object()
        .into_iter()
        .flat_map(|b| b.values())
        .filter_map(|v| v.get("requires")?.as_array())
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut values = serde_json::Map::new();
    for k in plow_asset::knob_gen::KNOBS {
        let v = match k.layer {
            Layer::Emit => knobs
                .iter()
                .find(|r| r.id == k.name())
                .map(|r| typed(k, &r.value, r.source != "default").to_json()),
            Layer::RawEnv => {
                Some(raw_env_value(k, block.as_deref()).map_or(Value::Null, Value::from))
            }
            Layer::ObjectDefine => Some(
                requires
                    .iter()
                    .find_map(|r| r.strip_prefix(k.name())?.strip_prefix('='))
                    .map_or(Value::Null, Value::from),
            ),
            Layer::Runtime => None,
        };
        if let Some(v) = v {
            values.insert(k.id.into(), v);
        }
    }
    json!({
        "K": if verdict.verified { "verified" } else { "skipped" },
        "reason": verdict.reason,
        "notes": verdict.notes,
        "registry_digest": registry_digest(),
        "target": target.to_json(),
        "values": values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::knob::test_util::{check_table, env_reads, files, plow_tokens, ArgFacts};
    use plow_asset::knob::{lookup, resolve, verdict, well_formed, USIZE};
    use std::path::{Path, PathBuf};

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    pub(crate) fn clap_facts(cmd: &clap::Command) -> Vec<ArgFacts> {
        use std::any::TypeId;
        cmd.get_arguments()
            .map(|a| {
                let ty = a.get_value_parser().type_id();
                let domain = if ty == TypeId::of::<bool>() {
                    Domain::Bool
                } else if ty == TypeId::of::<u32>() {
                    U32
                } else if ty == TypeId::of::<usize>() || ty == TypeId::of::<u64>() {
                    USIZE
                } else {
                    Domain::Str
                };
                ArgFacts {
                    id: a.get_id().as_str().into(),
                    env: a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    domain,
                    default: a
                        .get_default_values()
                        .iter()
                        .map(|v| v.to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                        .join(","),
                    hide: a.is_hide_set(),
                }
            })
            .collect()
    }

    #[test]
    fn every_emit_arg_has_exactly_one_spec() {
        use clap::Args;
        let cmd = crate::emit_config::EmitConfig::augment_args(clap::Command::new("plowc"));
        check_table(&clap_facts(&cmd), "emit.", EMIT);
    }

    #[test]
    fn registry_is_well_formed() {
        let specs: Vec<&KnobSpec> = EMIT.iter().chain(RAW_ENV).chain(OBJECT_DEFINES).collect();
        let mut ids: Vec<&str> = specs.iter().map(|k| k.id).collect();
        ids.sort_unstable();
        let dups: Vec<_> = ids.windows(2).filter(|w| w[0] == w[1]).collect();
        assert!(dups.is_empty(), "duplicate knob ids: {dups:?}");
        assert!(well_formed(&specs, &|_: &str| Source::default()));
        for k in &specs {
            match k.status {
                Status::Qualified { evidence }
                | Status::Candidate { evidence }
                | Status::Parked { evidence, .. } => {
                    assert!(!evidence.is_empty(), "{} cites no evidence", k.id)
                }
                _ => {}
            }
        }
        let constraints = plow_asset::knob::constraints_of(specs.iter().copied());
        for c in &constraints {
            let mut vars = Vec::new();
            c.formula.vars(&mut vars);
            for v in vars {
                assert!(
                    ids.binary_search(&v).is_ok(),
                    "constraint {} reads unregistered {v}",
                    c.id
                );
            }
        }
    }

    #[test]
    fn every_emit_side_env_read_is_registered() {
        let mut sources = Vec::new();
        for d in [
            "crates/devgen/src",
            "crates/packet/src",
            "crates/plowc/src",
            "crates/kernelcaps/src",
        ] {
            files(&root().join(d), Some("rs"), &mut sources);
        }
        assert!(
            sources.len() > 50,
            "sources not found under {}",
            root().display()
        );
        let emit_envs: Vec<&str> = EMIT.iter().filter_map(|k| k.env).collect();
        let raw: Vec<&str> = RAW_ENV.iter().filter_map(|k| k.env).collect();
        let mut unregistered = Vec::new();
        let mut read = Vec::new();
        for (path, text) in &sources {
            let own = path.ends_with("emit_config.rs");
            for name in env_reads(text) {
                read.push(name.to_string());
                if !(raw.contains(&name) || own && emit_envs.contains(&name)) {
                    unregistered.push(format!("{}: {name}", path.display()));
                }
            }
        }
        assert!(
            unregistered.is_empty(),
            "raw env reads with no RAW_ENV spec (declare the knob in EmitConfig, or add an \
             `env.` line with a status): {unregistered:#?}"
        );
        let docs =
            std::fs::read_to_string(root().join("docs/flags-reference.md")).unwrap_or_default();
        let mut scripts = Vec::new();
        files(&root().join("scripts"), None, &mut scripts);
        for k in RAW_ENV {
            let name = k.name();
            let is_read = read.iter().any(|r| r == name);
            if k.status == Status::Removed {
                assert!(!is_read, "{} is Removed but still read", k.id);
            } else {
                assert!(
                    is_read || docs.contains(name) || scripts.iter().any(|(_, t)| t.contains(name)),
                    "{} is read, documented and scripted nowhere: mark it Removed or delete it",
                    k.id
                );
            }
        }
    }

    #[test]
    fn every_object_define_is_registered() {
        let mut runtime = Vec::new();
        files(&root().join("runtime"), None, &mut runtime);
        let mut crates = Vec::new();
        files(&root().join("crates"), Some("rs"), &mut crates);
        crates.retain(|(p, _)| !p.ends_with("knob_spec.rs") && !p.ends_with("knob_gen.rs"));
        assert!(
            !runtime.is_empty(),
            "runtime/ not found under {}",
            root().display()
        );
        let names: Vec<&str> = OBJECT_DEFINES.iter().map(|k| k.name()).collect();
        let mut unregistered = std::collections::BTreeSet::new();
        for (path, text) in &runtime {
            for line in text.lines() {
                let Some(d) = line.trim_start().strip_prefix('#') else {
                    continue;
                };
                let d = d.trim_start();
                if !(d.starts_with("if") || d.starts_with("elif")) {
                    continue;
                }
                for t in plow_tokens(line) {
                    if !t.ends_with("_H") && !names.contains(&t) {
                        unregistered.insert(format!("{}: {t}", path.display()));
                    }
                }
            }
        }
        for (path, text) in &crates {
            for (i, _) in text.match_indices("\"-DPLOW_") {
                let t = plow_tokens(&text[i + 3..])
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                if !names.contains(&t) {
                    unregistered.insert(format!("{}: {t}", path.display()));
                }
            }
        }
        assert!(
            unregistered.is_empty(),
            "object defines with no OBJECT_DEFINES spec: {unregistered:#?}"
        );
        for k in OBJECT_DEFINES {
            let name = k.name();
            assert!(
                runtime.iter().chain(&crates).any(|(_, t)| t.contains(name)),
                "{} appears in no runtime or crate source: delete the line",
                k.id
            );
        }
    }

    fn recipe_sources<'a>(recipe: &'a [(&'a str, Val<'a>)]) -> impl Fn(&str) -> Source<'a> + 'a {
        move |id| Source {
            cli: None,
            env: recipe.iter().find(|(k, _)| *k == id).map(|(_, v)| *v),
        }
    }

    #[test]
    fn declared_recipes_satisfy_every_constraint() {
        let specs = emit_specs();
        let constraints = plow_asset::knob::constraints_of(specs.iter().copied());
        for t in TARGETS {
            assert_eq!(
                verdict(&specs, &constraints, &recipe_sources(t.recipe), &t.target()),
                "ok",
                "{}",
                t.name
            );
        }
    }

    #[test]
    fn negative_fixtures_are_rejected() {
        let specs = emit_specs();
        let constraints = plow_asset::knob::constraints_of(specs.iter().copied());
        let glm = TARGETS[0];
        let with = |extra: &'static [(&'static str, Val<'static>)]| {
            let mut r: Vec<(&str, Val)> = extra.to_vec();
            r.extend(glm.recipe.iter().copied());
            r
        };
        let cases: &[(&[(&str, Val)], &str)] = &[
            (
                &[
                    ("emit.glm_seq_par", TRUE),
                    ("emit.glm_xr_band", Val::Nat(2)),
                ],
                "seq_par_excludes_two_shot_seams",
            ),
            (
                &[("emit.glm_ofold", TRUE)],
                "ofold_excludes_dsa_pf_and_fp8_kv",
            ),
            (
                &[("emit.token_batch_tp", TRUE)],
                "token_batch_tp_excludes_seq_par",
            ),
            (
                &[("emit.glm_seq_par_proj", TRUE), ("emit.glm_seq_par", FALSE)],
                "seq_par_proj_requires_seq_par",
            ),
        ];
        for (extra, expect) in cases {
            let r = with(extra);
            assert_eq!(
                verdict(&specs, &constraints, &recipe_sources(&r), &glm.target()),
                *expect
            );
        }
        let mut pooled = glm.target();
        pooled.caps.push("indexer_pooled".into());
        let r = with(&[("emit.packed_sparse_pf", TRUE)]);
        assert_eq!(
            verdict(&specs, &constraints, &recipe_sources(&r), &pooled),
            "packed_sparse_pf_contract"
        );
    }

    /// The Rust mirror of K's record agreement on the production recipe: the registry's
    /// production defaults must reproduce what `apply_production_defaults` noted.
    #[test]
    fn registry_defaults_reproduce_the_emitter_record() {
        let _guard = crate::test_env::env_guard();
        // A fresh thread: earlier tests on this one leave production-default overrides in the
        // thread-local knob record.
        std::thread::spawn(record_agreement).join().unwrap();
    }

    fn record_agreement() {
        use clap::{Args, FromArgMatches};
        let glm = TARGETS[0];
        // The recipe as flags, not env: this test must not move the environment other tests read.
        let cmd = crate::emit_config::EmitConfig::augment_args(clap::Command::new("plowc"));
        let mut argv = vec!["plowc".to_string()];
        for (id, v) in glm.recipe {
            let Some(arg) = id
                .strip_prefix("emit.")
                .and_then(|name| cmd.get_arguments().find(|a| a.get_id() == name))
            else {
                continue;
            };
            let text = match v {
                Val::Bool(b) => b.to_string(),
                Val::Nat(n) => n.to_string(),
                Val::Str(s) => s.to_string(),
                Val::Unset => continue,
            };
            argv.push(format!("--{}={text}", arg.get_long().expect("long flag")));
        }
        let m = cmd.try_get_matches_from(argv).expect("the recipe parses");
        crate::emit_config::record_knobs(Some(&m));
        let mut cfg = crate::emit_config::EmitConfig::from_arg_matches(&m).expect("emit config");
        let caps = crate::emit_capabilities(glm.model);
        crate::apply_production_defaults(&mut cfg, caps, glm.arch, glm.tp, glm.n_cu);
        note_target(glm.arch, glm.tp, glm.n_cu, glm.model, caps.caps(), None);
        let p = payload().expect("target noted");
        let src_of = |id: &str| -> Source {
            let Some(s) = p["sources"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["id"] == id)
            else {
                return Source::default();
            };
            fn val(v: &Value) -> Val<'_> {
                match v {
                    Value::Bool(b) => Val::Bool(*b),
                    Value::Number(n) => Val::Nat(n.as_u64().unwrap()),
                    Value::String(s) => Val::Str(s.as_str()),
                    _ => Val::Unset,
                }
            }
            Source {
                cli: s.get("cli").map(val),
                env: s.get("env").map(val),
            }
        };
        let specs = emit_specs();
        let resolved = resolve(&specs, &src_of, &glm.target());
        let mut mismatches = Vec::new();
        for r in p["recorded"].as_array().unwrap() {
            let id = r["id"].as_str().unwrap();
            let got = lookup(&resolved, id).to_json();
            if got != r["value"] {
                mismatches.push(format!("{id}: registry {got} vs emitter {}", r["value"]));
            }
        }
        assert!(mismatches.is_empty(), "{mismatches:#?}");
        assert_eq!(lookup(&resolved, "emit.glm_seq_par"), TRUE);
        assert_eq!(
            lookup(&resolved, "emit.glm_gemm_lt_pf_ext").to_json(),
            json!("o_proj,band,shared")
        );
    }
}
