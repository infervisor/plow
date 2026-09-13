//! The runtime-side knob registry: every `RuntimeConfig` knob and every raw `PLOW_*` read in
//! plowrt, with the constraints that read runtime knobs, and the load check.
//!
//! [`check_assets`] runs the generated evaluator of every registered constraint
//! ([`plow_asset::knob_gen`]) over the packet's `build.json` `knobs` block and this process's
//! runtime config. It is fatal, and it reads only the head of `build.json`.

use crate::config::RuntimeConfig;
use crate::{Result, RuntimeError};
use plow_asset::knob::{
    Check, Cmp, Constraint, Default, Domain, Formula as F, KnobSpec, Layer, Status, Target,
    TargetAtom as T, Val, U32, USIZE,
};
use plow_asset::knob_gen;
use serde_json::Value;
use std::path::Path;

const TRUE: Val = Val::Bool(true);
const OFF: Default = Default::Static(Val::Bool(false));
const ON: Default = Default::Static(TRUE);
const UNSET: Default = Default::Static(Val::Unset);

const OPT_IN: Status = Status::OptIn;
const DIAG: Status = Status::Diagnostic;
const PREFIX_CACHE_CANDIDATE: Status = Status::Candidate {
    evidence: &[
        "review log #83: attach-verify4, 0/0 mismatches over 63 attaches; opt-in, unset by default",
    ],
};
const ROW_SPLIT_CANDIDATE: Status = Status::Candidate {
    evidence: &["review log #84: p0split-full-depth, 8192-row first chunk 725.9 -> 616.8 ms, retrieval 39/39; flip waits on the served A/B"],
};
const PROMOTED: Status = Status::Qualified {
    evidence: &["docs/flags-reference.md: a promoted default; `=false` is the rollback"],
};

const C_PF_SEG_GEMM_SMALL: &[Constraint] = &[Constraint {
    id: "pf_seg_gemm_small_requires_seg_dir",
    formula: F::Implies(
        &F::Atom("rt.pf_seg_gemm_small", Cmp::Ne, Val::Unset),
        &F::Or(&[
            F::Atom("rt.pf_seg_dir", Cmp::Ne, Val::Unset),
            F::Target(T::Cap("bundled_segment_pair")),
        ]),
    ),
    site: "crates/plowrt/src/exec/gpu.rs: PLOW_PF_SEG_GEMM_SMALL requires PLOW_PF_SEG_DIR",
    check: Check::Site,
}];

const C_CHANNEL_MLP_RUNTIME: &[Constraint] = &[Constraint {
    id: "channel_mlp_runtime_exclusive",
    formula: F::Implies(
        &F::Atom("rt.ane_mlp", Cmp::Eq, TRUE),
        &F::And(&[
            F::Not(&F::Atom("rt.serial", Cmp::Eq, TRUE)),
            F::Atom("emit.row_split", Cmp::Eq, Val::Unset),
        ]),
    ),
    site: "crates/plowrt/src/exec/apple/mod.rs: channel MLP cannot combine serial, row, per-op ANE or CPU offload",
    check: Check::Load,
}];

const C_MLA_PF_V2: &[Constraint] = &[
    Constraint {
        id: "ofold_packet_requires_mla_pf_v2",
        formula: F::Implies(
            &F::Atom("def.PLOW_GLM_OFOLD", Cmp::Eq, Val::Str("1")),
            &F::Atom("rt.mla_pf_v2", Cmp::Eq, TRUE),
        ),
        site: "crates/plowrt/src/exec/amd.rs: this packet requires PLOW_GLM_OFOLD=1 ... serve with PLOW_MLA_PF_V2=1",
        check: Check::Load,
    },
    Constraint {
        id: "dsa_pf_packet_requires_mla_pf_v2",
        formula: F::Implies(
            &F::Atom("def.PLOW_DSA_PF_ARM", Cmp::Eq, Val::Str("1")),
            &F::Atom("rt.mla_pf_v2", Cmp::Eq, TRUE),
        ),
        site: "crates/plowrt/src/exec/amd/object.rs: this packet requires PLOW_DSA_PF_ARM=1 ... serve with PLOW_MLA_PF_V2=1",
        check: Check::Load,
    },
];

fn opt_str(s: &Option<String>) -> Val<'_> {
    s.as_deref().map_or(Val::Unset, Val::Str)
}

/// The runtime knobs a registered constraint reads. `generated_evaluator_is_current` holds this
/// list to the runtime ids in `knob_gen::KNOBS`.
type Reader = fn(&RuntimeConfig) -> Val<'_>;
const READERS: &[(&str, Reader)] = &[
    ("rt.pf_seg_dir", |c| opt_str(&c.nv.pf_seg_dir)),
    ("rt.pf_seg_gemm_small", |c| opt_str(&c.nv.pf_seg_gemm_small)),
    ("rt.mla_pf_v2", |c| Val::Bool(c.amd.mla_pf_v2)),
    ("rt.serial", |c| Val::Bool(c.apple.serial)),
    ("rt.ane_mlp", |c| Val::Bool(c.apple.ane_mlp)),
];

/// Refuse a packet checkpoint K failed, or whose recorded knobs violate a registered constraint
/// under this runtime config. A packet with no `knobs` block predates checkpoint K and one with
/// `K = "skipped"` is a bring-up packet: both load, with a warning.
pub fn check_assets(blob: &Path) -> Result<()> {
    let path = blob.with_file_name("build.json");
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(RuntimeError::Io { path, source }),
    };
    let rejected = |e: String| RuntimeError::Rejected(format!("{}: {e}", path.display()));
    let knobs = read_knobs(std::io::BufReader::new(file)).map_err(rejected)?;
    let Some(knobs) = knobs else {
        tracing::warn!(
            manifest = %path.display(),
            "no checkpoint K certificate (build.json has no `knobs` block): this packet predates \
             knob verification and loads unverified"
        );
        return Ok(());
    };
    check_knobs(&knobs, RuntimeConfig::get()).map_err(rejected)
}

/// The `knobs` block of a `build.json`, reading no further than its end.
fn read_knobs(reader: impl std::io::Read) -> std::result::Result<Option<Value>, String> {
    use serde::de::{Error as _, IgnoredAny, MapAccess, Visitor};

    struct Head<'s>(&'s mut Option<Value>);

    impl<'de> Visitor<'de> for Head<'_> {
        type Value = ();

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a build.json object")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> std::result::Result<(), A::Error> {
            while let Some(key) = m.next_key::<String>()? {
                if key == "knobs" {
                    *self.0 = Some(m.next_value()?);
                    // Stop: the rest of the manifest is megabytes the check does not read.
                    return Err(A::Error::custom("knobs block read"));
                }
                m.next_value::<IgnoredAny>()?;
            }
            Ok(())
        }
    }

    let mut out = None;
    let mut de = serde_json::Deserializer::from_reader(reader);
    let r = serde::Deserializer::deserialize_map(&mut de, Head(&mut out));
    match (r, out) {
        (_, Some(v)) => Ok(Some(v)),
        (Ok(()), None) => Ok(None),
        (Err(e), None) => Err(format!("not a valid build.json: {e}")),
    }
}

fn json_val(v: &Value) -> Val<'_> {
    match v {
        Value::Bool(b) => Val::Bool(*b),
        Value::Number(n) => n.as_u64().map_or(Val::Unset, Val::Nat),
        Value::String(s) => Val::Str(s),
        _ => Val::Unset,
    }
}

pub fn check_knobs(knobs: &Value, rt: &RuntimeConfig) -> std::result::Result<(), String> {
    let reason = knobs
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("none recorded");
    match knobs.get("K").and_then(Value::as_str) {
        Some("verified") => {}
        Some("skipped") => tracing::warn!(
            reason,
            "checkpoint K was skipped for this packet (build.json knobs.K = skipped): a bring-up \
             packet, not a serving set"
        ),
        k => {
            return Err(format!(
                "knobs.K = {k:?} (reason: {reason}): checkpoint K did not verify this packet"
            ))
        }
    }
    let target = knobs
        .get("target")
        .and_then(Target::from_json)
        .unwrap_or_default();
    let values = knobs.get("values").and_then(Value::as_object);
    let vals: Vec<Val> = knob_gen::KNOBS
        .iter()
        .map(|k| match k.layer {
            Layer::Runtime => READERS
                .iter()
                .find(|(id, _)| *id == k.id)
                .map_or(Val::Unset, |(_, read)| read(rt)),
            _ => values
                .and_then(|m| m.get(k.id))
                .map_or(Val::Unset, json_val),
        })
        .collect();
    let Some(i) = knob_gen::violation(&vals, &target) else {
        return Ok(());
    };
    let c = &knob_gen::CONSTRAINTS[i];
    let mut vars = Vec::new();
    c.formula.vars(&mut vars);
    let shown: Vec<String> = vars
        .iter()
        .filter_map(|v| {
            let j = knob_gen::KNOBS.iter().position(|k| k.id == *v)?;
            Some(format!("{v}={:?}", vals[j]))
        })
        .collect();
    Err(format!(
        "knob constraint `{}` is violated ({}) by {}",
        c.id,
        c.site,
        shown.join(" ")
    ))
}

#[rustfmt::skip]
pub const RUNTIME: &[KnobSpec] = &[
    KnobSpec::new("rt.rt_checkpoint", Some("PLOW_CHECKPOINT"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.plow_home", Some("PLOW_HOME"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.registry", Some("PLOW_REGISTRY"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("dist.infervisor.ai")), OPT_IN),
    KnobSpec::new("rt.prefetch", Some("PLOW_PREFETCH"), Layer::Runtime, USIZE, Default::Static(Val::Nat(256)), OPT_IN),
    KnobSpec::new("rt.prefetch_threads", Some("PLOW_PREFETCH_THREADS"), Layer::Runtime, USIZE, Default::Static(Val::Nat(16)), OPT_IN),
    KnobSpec::new("rt.weight_slab", Some("PLOW_WEIGHT_SLAB"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.fusion", Some("PLOW_FUSION"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.token_batch", Some("PLOW_TOKEN_BATCH"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.prefix_cache", Some("PLOW_PREFIX_CACHE"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.vmm_cache_memory_utilization", Some("PLOW_VMM_CACHE_MEMORY_UTILIZATION"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("0.05")), OPT_IN),
    KnobSpec::new("rt.vmm_cache_min_free_mib", Some("PLOW_VMM_CACHE_MIN_FREE_MIB"), Layer::Runtime, U32, UNSET, PREFIX_CACHE_CANDIDATE),
    KnobSpec::new("rt.amd_prefix_fine_rows", Some("PLOW_AMD_PREFIX_FINE_ROWS"), Layer::Runtime, U32, UNSET, PREFIX_CACHE_CANDIDATE),
    KnobSpec::new("rt.mla_pf_row_split", Some("PLOW_MLA_PF_ROW_SPLIT"), Layer::Runtime, Domain::Bool, OFF, ROW_SPLIT_CANDIDATE),
    KnobSpec::new("rt.vmm_cache_mib", Some("PLOW_VMM_CACHE_MIB"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.vmm_block_mib", Some("PLOW_VMM_BLOCK_MIB"), Layer::Runtime, U32, Default::Static(Val::Nat(2)), OPT_IN),
    KnobSpec::new("rt.weight_vmm", Some("PLOW_WEIGHT_VMM"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_batch", Some("PLOW_PF_BATCH"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_interleave", Some("PLOW_PF_INTERLEAVE"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_chunk", Some("PLOW_PF_CHUNK"), Layer::Runtime, U32, Default::Static(Val::Nat(0)), OPT_IN),
    KnobSpec::new("rt.pf_no_chunk", Some("PLOW_PF_NO_CHUNK"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_no_interleave", Some("PLOW_PF_NO_INTERLEAVE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_defer_decode", Some("PLOW_PF_DEFER_DECODE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.tbt_slo_ms", Some("PLOW_TBT_SLO_MS"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ttft_slo_ms", Some("PLOW_TTFT_SLO_MS"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.slab_keep", Some("PLOW_SLAB_KEEP"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.dstep_every", Some("PLOW_DSTEP_EVERY"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.drain_timeout_ms", Some("PLOW_DRAIN_TIMEOUT_MS"), Layer::Runtime, USIZE, UNSET, OPT_IN),
    KnobSpec::new("rt.devices", Some("PLOW_DEVICES"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.place", Some("PLOW_PLACE"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("spread")), OPT_IN),
    KnobSpec::new("rt.pin", Some("PLOW_PIN"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.co_sched", Some("PLOW_CO_SCHED"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("free")), OPT_IN),
    KnobSpec::new("rt.co_sched_quantum", Some("PLOW_CO_SCHED_QUANTUM"), Layer::Runtime, U32, Default::Static(Val::Nat(4)), OPT_IN),
    KnobSpec::new("rt.models_root", Some("PLOW_MODELS_ROOT"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.preload", Some("PLOW_PRELOAD"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.kv_pool_mib", Some("PLOW_KV_POOL_MIB"), Layer::Runtime, USIZE, Default::Static(Val::Nat(512)), OPT_IN),
    KnobSpec::new("rt.vmm_deferred_reclaim", Some("PLOW_VMM_DEFERRED_RECLAIM"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.ttft_log", Some("PLOW_TTFT_LOG"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pfx_log", Some("PLOW_PFX_LOG"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.tick_log", Some("PLOW_TICK_LOG"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.dstep_log", Some("PLOW_DSTEP_LOG"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_packlog", Some("PLOW_PF_PACKLOG"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.load_profile", Some("PLOW_LOAD_PROFILE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.multistep", Some("PLOW_MULTISTEP"), Layer::Runtime, U32, Default::Static(Val::Nat(8)), OPT_IN),
    KnobSpec::new("rt.vmm_prefix", Some("PLOW_VMM_PREFIX"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.vmm_live", Some("PLOW_VMM_LIVE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.vmm_live_rings", Some("PLOW_VMM_LIVE_RINGS"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.upload_direct", Some("PLOW_UPLOAD_DIRECT"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.cubin", Some("PLOW_NV_CUBIN"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.cubin_pf", Some("PLOW_NV_CUBIN_PF"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.kernel", Some("PLOW_NV_KERNEL"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.kernel_pf", Some("PLOW_NV_KERNEL_PF"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.smem", Some("PLOW_NV_SMEM"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.smem_pf", Some("PLOW_NV_SMEM_PF"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.dev_sample", Some("PLOW_DEV_SAMPLE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.cubin_sample", Some("PLOW_NV_CUBIN_SAMPLE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.kernel_sample", Some("PLOW_NV_KERNEL_SAMPLE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.libcuda", Some("PLOW_LIBCUDA"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.vram_budget_mib", Some("PLOW_VRAM_BUDGET_MIB"), Layer::Runtime, USIZE, UNSET, OPT_IN),
    KnobSpec::new("rt.step_time", Some("PLOW_STEP_TIME"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.l2_place_dispatch", Some("PLOW_L2_PLACE_DISPATCH"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_cover", Some("PLOW_PF_COVER"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_chunk_cost", Some("PLOW_PF_CHUNK_COST"), Layer::Runtime, USIZE, Default::Static(Val::Nat(512)), OPT_IN),
    KnobSpec::new("rt.pf_seg_dir", Some("PLOW_PF_SEG_DIR"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_seg_gemm_small", Some("PLOW_PF_SEG_GEMM_SMALL"), Layer::Runtime, Domain::Str, UNSET, OPT_IN).with(C_PF_SEG_GEMM_SMALL),
    KnobSpec::new("rt.pf_seg_pure", Some("PLOW_PF_SEG_PURE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_seg_fa512", Some("PLOW_PF_SEG_FA512"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_seg_fa256_gqa2", Some("PLOW_PF_SEG_FA256_GQA2"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_seg_graph", Some("PLOW_PF_SEG_GRAPH"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_seg_v2", Some("PLOW_PF_SEG_V2"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.pf_seg_time", Some("PLOW_PF_SEG_TIME"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_seg_fatonly", Some("PLOW_PF_SEG_FATONLY"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_seg_noncoop", Some("PLOW_PF_SEG_NONCOOP"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_trace_log", Some("PLOW_PF_TRACE_LOG"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.pf_seg_eqsmem", Some("PLOW_PF_SEG_EQSMEM"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.ctr_dbuf", Some("PLOW_CTR_DBUF"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.state_clear_device", Some("PLOW_STATE_CLEAR_DEVICE"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.packed_prefill_route", Some("PLOW_PACKED_PREFILL_ROUTE"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.kda_family_route", Some("PLOW_KDA_FAMILY_ROUTE"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.global_queue", Some("PLOW_GLOBAL_QUEUE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.static_sched", Some("PLOW_STATIC"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.seg_window", Some("PLOW_SEG_WINDOW"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.tp_prefill_segment_major", Some("PLOW_TP_PREFILL_SEGMENT_MAJOR"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.prefill_seg_timing", Some("PLOW_PREFILL_SEG_TIMING"), Layer::Runtime, Domain::Bool, OFF, DIAG),
    KnobSpec::new("rt.native_launch_timing", Some("PLOW_NATIVE_LAUNCH_TIMING"), Layer::Runtime, Domain::Bool, OFF, DIAG),
    KnobSpec::new("rt.tb_dump", Some("PLOW_TB_DUMP"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.trace_allranks", Some("PLOW_TRACE_ALLRANKS"), Layer::Runtime, Domain::Bool, OFF, DIAG),
    KnobSpec::new("rt.attnres_f32mix_grid", Some("PLOW_ATTNRES_F32MIX_GRID"), Layer::Runtime, U32, UNSET, DIAG),
    KnobSpec::new("rt.moe_prefill_ep_max_extra_bytes", Some("PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES"), Layer::Runtime, USIZE, UNSET, DIAG),
    KnobSpec::new("rt.phase_objects", Some("PLOW_PHASE_OBJECTS"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.vmm_kv", Some("PLOW_VMM_KV"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.kv_map_ahead", Some("PLOW_KV_MAP_AHEAD"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.kv_map_next_chunk", Some("PLOW_KV_MAP_NEXT_CHUNK"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.publish_defer", Some("PLOW_AMD_PUBLISH_DEFER"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.hsa_drain_blocked", Some("PLOW_HSA_DRAIN_BLOCKED"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.engine_affinity", Some("PLOW_AMD_ENGINE_AFFINITY"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("auto")), OPT_IN),
    KnobSpec::new("rt.shared_prefix", Some("PLOW_AMD_SHARED_PREFIX"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.upload_slots", Some("PLOW_UPLOAD_SLOTS"), Layer::Runtime, U32, Default::Static(Val::Nat(1)), OPT_IN),
    KnobSpec::new("rt.oversub", Some("PLOW_OVERSUB"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.share_ckpt", Some("PLOW_SHARE_CKPT"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.tp_serial_load", Some("PLOW_TP_SERIAL_LOAD"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.tp_agree_every", Some("PLOW_TP_AGREE_EVERY"), Layer::Runtime, U32, Default::Static(Val::Nat(1)), OPT_IN),
    KnobSpec::new("rt.tp_no_audit", Some("PLOW_TP_NO_AUDIT"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.tp_audit_direct", Some("PLOW_TP_AUDIT_DIRECT"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.tp_audit_compact", Some("PLOW_TP_AUDIT_COMPACT"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.tp_prefill_audit_direct", Some("PLOW_TP_PREFILL_AUDIT_DIRECT"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.tp_prefill_audit_pinned", Some("PLOW_TP_PREFILL_AUDIT_PINNED"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.launch_rows", Some("PLOW_LAUNCH_ROWS"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.token_batch_rows", Some("PLOW_TOKEN_BATCH_ROWS"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.token_batch_solo", Some("PLOW_TOKEN_BATCH_SOLO"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.decode_min_rung", Some("PLOW_AMD_DECODE_MIN_RUNG"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.tail_sparse_ctx", Some("PLOW_AMD_TAIL_SPARSE_CTX"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.token_batch_wide_tiles", Some("PLOW_TOKEN_BATCH_WIDE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.ragged_chunk", Some("PLOW_RAGGED_CHUNK"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.ragged_seams", Some("PLOW_AMD_RAGGED_SEAMS"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.mla_ns_live", Some("PLOW_MLA_NS_LIVE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.rt_hsaco", Some("PLOW_HSACO"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.fp8_dir", Some("PLOW_FP8_DIR"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.trace_raw", Some("PLOW_TRACE_RAW"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ctr_snap", Some("PLOW_CTR_SNAP"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.tens_snap", Some("PLOW_TENS_SNAP"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.snap_tensors", Some("PLOW_SNAP_TENSORS"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.snap_slot", Some("PLOW_SNAP_SLOT"), Layer::Runtime, USIZE, Default::Static(Val::Nat(5)), DIAG),
    KnobSpec::new("rt.pf_capture", Some("PLOW_PF_CAPTURE"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.hsaco_lowrung", Some("PLOW_HSACO_LOWRUNG"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.lowrung_max", Some("PLOW_LOWRUNG_MAX"), Layer::Runtime, U32, Default::Static(Val::Nat(2)), OPT_IN),
    KnobSpec::new("rt.lm_row0", Some("PLOW_LM_ROW0"), Layer::Runtime, Domain::Bool, OFF, DIAG),
    KnobSpec::new("rt.mla_pf_v2", Some("PLOW_MLA_PF_V2"), Layer::Runtime, Domain::Bool, ON, PROMOTED).with(C_MLA_PF_V2),
    KnobSpec::new("rt.mla_pf_aiter", Some("PLOW_MLA_PF_AITER"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.moe_aiter_tile64", Some("PLOW_MOE_AITER_TILE64"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.moe_aiter_xcd", Some("PLOW_MOE_AITER_XCD"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.moe_stage1_a4_reuse", Some("PLOW_MOE_STAGE1_A4_REUSE"), Layer::Runtime, Domain::Bool, ON, PROMOTED),
    KnobSpec::new("rt.dump_act", Some("PLOW_DUMP_ACT"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.dump_kv", Some("PLOW_DUMP_KV"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.numa_host_pools", Some("PLOW_AMD_NUMA_HOST_POOLS"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.kernarg_vram", Some("PLOW_AMD_KERNARG_VRAM"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.threads", Some("PLOW_CPU_THREADS"), Layer::Runtime, U32, Default::Static(Val::Nat(0)), OPT_IN),
    KnobSpec::new("rt.numa", Some("PLOW_CPU_NUMA"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("auto")), OPT_IN),
    KnobSpec::new("rt.huge_pages", Some("PLOW_CPU_HUGE_PAGES"), Layer::Runtime, Domain::Bool, UNSET, OPT_IN),
    KnobSpec::new("rt.isa", Some("PLOW_CPU_ISA"), Layer::Runtime, Domain::Str, Default::Static(Val::Str("auto")), OPT_IN),
    KnobSpec::new("rt.spin_us", Some("PLOW_CPU_SPIN_US"), Layer::Runtime, U32, Default::Static(Val::Nat(2000)), OPT_IN),
    KnobSpec::new("rt.prefill_chunk", Some("PLOW_CPU_PF_CHUNK"), Layer::Runtime, U32, Default::Static(Val::Nat(0)), OPT_IN),
    KnobSpec::new("rt.mxfp4_dir", Some("PLOW_MXFP4_DIR"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.gq_opt_in", Some("PLOW_CPU_GQ"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.l2_place", Some("PLOW_CPU_L2_PLACE"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.backend", Some("PLOW_BACKEND"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.serial", Some("PLOW_METAL_SERIAL"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.spin_max", Some("PLOW_METAL_SPIN_MAX"), Layer::Runtime, U32, UNSET, OPT_IN),
    KnobSpec::new("rt.cpu_share", Some("PLOW_CPU_SHARE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ane", Some("PLOW_ANE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ane_mlp", Some("PLOW_ANE_MLP"), Layer::Runtime, Domain::Bool, OFF, OPT_IN).with(C_CHANNEL_MLP_RUNTIME),
    KnobSpec::new("rt.ane_mlp_layers", Some("PLOW_ANE_MLP_LAYERS"), Layer::Runtime, USIZE, UNSET, OPT_IN),
    KnobSpec::new("rt.ane_mlp_placement", Some("PLOW_ANE_MLP_PLACEMENT"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ane_mlp_cache", Some("PLOW_ANE_MLP_CACHE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ane_mlp_fail", Some("PLOW_ANE_MLP_FAIL"), Layer::Runtime, Domain::Str, UNSET, DIAG),
    KnobSpec::new("rt.ane_w8", Some("PLOW_ANE_W8"), Layer::Runtime, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("rt.ane_units", Some("PLOW_ANE_UNITS"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.ane_resid", Some("PLOW_ANE_RESID"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.out_range", Some("PLOW_OUT_RANGE"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.cores", Some("PLOW_HET_CORES"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
    KnobSpec::new("rt.reserve_cores", Some("PLOW_HET_RESERVE_CORES"), Layer::Runtime, U32, Default::Static(Val::Nat(0)), OPT_IN),
    KnobSpec::new("rt.twin", Some("PLOW_HET_TWIN"), Layer::Runtime, Domain::Str, UNSET, OPT_IN),
];

#[rustfmt::skip]
pub const RAW_ENV: &[KnobSpec] = &[
    KnobSpec::new("env.PLOW_DEV_SAMPLE", Some("PLOW_DEV_SAMPLE"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_DSA_VERIFY_CKPT", Some("PLOW_DSA_VERIFY_CKPT"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_DSA_VERIFY_OUT", Some("PLOW_DSA_VERIFY_OUT"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_GPU_ASSETS", Some("PLOW_GPU_ASSETS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_LT_PACKETS", Some("PLOW_LT_PACKETS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_NV_CUBIN_SAMPLE", Some("PLOW_NV_CUBIN_SAMPLE"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_PREFIX_EMIT_CHILD", Some("PLOW_PREFIX_EMIT_CHILD"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SHARED_PREFIX_BASELINE", Some("PLOW_SHARED_PREFIX_BASELINE"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SHARED_PREFIX_EMIT_CHILD", Some("PLOW_SHARED_PREFIX_EMIT_CHILD"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SHARED_PREFIX_PACKET", Some("PLOW_SHARED_PREFIX_PACKET"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SLAB_KEEP", Some("PLOW_SLAB_KEEP"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SLO_REPLAY_LOGS", Some("PLOW_SLO_REPLAY_LOGS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SLO_SIM_LOGS", Some("PLOW_SLO_SIM_LOGS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_SLO_SIM_TARGETS", Some("PLOW_SLO_SIM_TARGETS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_ABI144_DECODE_ELF", Some("PLOW_TEST_ABI144_DECODE_ELF"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_AITER_DIR", Some("PLOW_TEST_AITER_DIR"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_BLK_CAPTURE", Some("PLOW_TEST_BLK_CAPTURE"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_DSA_DIR", Some("PLOW_TEST_DSA_DIR"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_FOLD_DIR", Some("PLOW_TEST_FOLD_DIR"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_FOLD_TILE64", Some("PLOW_TEST_FOLD_TILE64"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_GLM52_PREFILL_ELF", Some("PLOW_TEST_GLM52_PREFILL_ELF"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_K3_SNAPSHOTS", Some("PLOW_TEST_K3_SNAPSHOTS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_TEST_MOE_ROWS", Some("PLOW_TEST_MOE_ROWS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_VMM_DEFERRED_RECLAIM", Some("PLOW_VMM_DEFERRED_RECLAIM"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_VMM_LIVE", Some("PLOW_VMM_LIVE"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_VMM_LIVE_RINGS", Some("PLOW_VMM_LIVE_RINGS"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
    KnobSpec::new("env.PLOW_VMM_PREFIX", Some("PLOW_VMM_PREFIX"), Layer::RawEnv, Domain::Str, UNSET, DIAG),
];

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::knob::test_util::{check_table, env_reads, files, plow_tokens, ArgFacts};
    use plow_asset::knob::{Formula, KnobSpec, TargetSpec};
    use std::path::PathBuf;

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The whole registry, in the order `knob_gen::KNOBS` keeps.
    fn registry() -> Vec<&'static KnobSpec> {
        devgen::knob_spec::EMIT
            .iter()
            .chain(devgen::knob_spec::RAW_ENV)
            .chain(devgen::knob_spec::OBJECT_DEFINES)
            .chain(RUNTIME)
            .chain(RAW_ENV)
            .collect()
    }

    #[test]
    fn every_runtime_arg_has_exactly_one_spec() {
        use clap::Args;
        use std::any::TypeId;
        let cmd = RuntimeConfig::augment_args(clap::Command::new("plowrt"));
        let facts: Vec<ArgFacts> = cmd
            .get_arguments()
            .map(|a| {
                let ty = a.get_value_parser().type_id();
                ArgFacts {
                    id: a.get_id().as_str().into(),
                    env: a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    domain: if ty == TypeId::of::<bool>() {
                        Domain::Bool
                    } else if ty == TypeId::of::<u32>() {
                        U32
                    } else if ty == TypeId::of::<usize>() || ty == TypeId::of::<u64>() {
                        USIZE
                    } else {
                        Domain::Str
                    },
                    default: a
                        .get_default_values()
                        .iter()
                        .map(|v| v.to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                        .join(","),
                    hide: a.is_hide_set(),
                }
            })
            .collect();
        check_table(&facts, "rt.", RUNTIME);
    }

    #[test]
    fn registry_ids_are_unique_across_crates() {
        let mut ids: Vec<&str> = registry().iter().map(|k| k.id).collect();
        ids.sort_unstable();
        let dups: Vec<_> = ids.windows(2).filter(|w| w[0] == w[1]).collect();
        assert!(dups.is_empty(), "duplicate knob ids: {dups:?}");
        for c in registry().iter().flat_map(|k| k.constraints) {
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
    fn every_plowrt_env_read_is_registered() {
        let mut sources = Vec::new();
        files(&root().join("crates/plowrt/src"), Some("rs"), &mut sources);
        assert!(sources.len() > 50, "plowrt sources not found");
        let runtime_envs: Vec<&str> = RUNTIME.iter().filter_map(|k| k.env).collect();
        let raw: Vec<&str> = RAW_ENV.iter().filter_map(|k| k.env).collect();
        let mut read = Vec::new();
        let mut unregistered = Vec::new();
        for (path, text) in &sources {
            let own = path.ends_with("config.rs");
            for name in env_reads(text) {
                read.push(name.to_string());
                if !(raw.contains(&name) || own && runtime_envs.contains(&name)) {
                    unregistered.push(format!("{}: {name}", path.display()));
                }
            }
        }
        assert!(
            unregistered.is_empty(),
            "raw env reads with no RAW_ENV spec: {unregistered:#?}"
        );
        for k in RAW_ENV {
            assert!(
                read.iter().any(|r| r == k.name()),
                "{} is read nowhere",
                k.id
            );
        }
    }

    #[test]
    fn flags_reference_names_only_registered_knobs() {
        let doc = std::fs::read_to_string(root().join("docs/flags-reference.md")).expect("docs");
        let names: std::collections::BTreeSet<&str> = registry()
            .iter()
            .flat_map(|k| [Some(k.name()), k.env])
            .flatten()
            .collect();
        let unknown: std::collections::BTreeSet<&str> = doc
            .lines()
            .flat_map(plow_tokens)
            .filter(|t| !names.contains(t))
            .collect();
        assert!(
            unknown.is_empty(),
            "docs/flags-reference.md names knobs the registry does not declare (register them, \
             or mark a retired spelling Removed): {unknown:?}"
        );
    }

    enum Site {
        Encoded(&'static str),
        NotAConstraint(&'static str),
    }

    /// Every `cannot combine` / `requires PLOW_` site in devgen and plowrt, and what encodes it.
    /// The asserts stay as defence; this keeps a new one from landing unencoded.
    const ASSERT_SITES: &[(&str, &str, Site)] = &[
        (
            "emit_config.rs",
            "Runtime offload additionally requires PLOW_ANE_MLP=1",
            Site::NotAConstraint("doc: rt.ane_mlp gates the offload"),
        ),
        (
            "lib.rs",
            "channel MLP cannot combine row split",
            Site::Encoded("channel_mlp_exclusive"),
        ),
        (
            "lib.rs",
            "T8 w8a8 (PLOW_W8A8=1, requires PLOW_FP8=1)",
            Site::NotAConstraint("comment of the next assert"),
        ),
        (
            "lib.rs",
            "PLOW_W8A8=1 requires PLOW_FP8=1",
            Site::NotAConstraint("vacuous: its `fp8` is any_fp8_weights(), which includes w8a8"),
        ),
        (
            "apple/mod.rs",
            "cannot combine serial, row, per-op ANE or CPU offload",
            Site::Encoded("channel_mlp_runtime_exclusive"),
        ),
        (
            "kimi_k3.rs",
            "requires PLOW_GEMV_WALK=1 above 16 rows",
            Site::Encoded("k3_wide_decode_requires_walk"),
        ),
        (
            "shared_prefix.rs",
            "requires PLOW_SHARED_PREFIX_PACKET",
            Site::NotAConstraint("test fixture path"),
        ),
        (
            "mla.rs",
            "PLOW_GLM_OFOLD=1 cannot combine with PLOW_GLM_DSA_PF",
            Site::Encoded("ofold_excludes_dsa_pf_and_fp8_kv"),
        ),
        (
            "mla.rs",
            "requires PLOW_PACKED_SPARSE_PF=1 with PLOW_GLM_INDEX_TP=1",
            Site::Encoded("packed_sparse_pf_contract"),
        ),
        (
            "mla.rs",
            "PLOW_GLM_SEQ_PAR cannot combine with PLOW_GLM_XR_BAND",
            Site::Encoded("seq_par_excludes_two_shot_seams"),
        ),
        (
            "mla.rs",
            "it cannot combine with PLOW_GLM_XR_RES",
            Site::Encoded("seq_par_excludes_two_shot_seams"),
        ),
        (
            "gpu.rs",
            "PLOW_PF_SEG_GEMM_SMALL requires PLOW_PF_SEG_DIR",
            Site::Encoded("pf_seg_gemm_small_requires_seg_dir"),
        ),
        (
            "amd.rs",
            "this packet requires PLOW_GLM_OFOLD=1",
            Site::Encoded("ofold_packet_requires_mla_pf_v2"),
        ),
        (
            "object.rs",
            "this packet requires PLOW_DSA_PF_ARM=1",
            Site::Encoded("dsa_pf_packet_requires_mla_pf_v2"),
        ),
    ];

    #[test]
    fn cross_knob_asserts_are_encoded() {
        let mut sources = Vec::new();
        files(&root().join("crates/devgen/src"), Some("rs"), &mut sources);
        files(&root().join("crates/plowrt/src"), Some("rs"), &mut sources);
        let constraint_ids: Vec<&str> = registry()
            .iter()
            .flat_map(|k| k.constraints)
            .map(|c| c.id)
            .collect();
        let mut hits = 0;
        let mut unmapped = Vec::new();
        for (path, text) in sources.iter().filter(|(p, _)| !p.ends_with("knob_spec.rs")) {
            for line in text.lines() {
                if !(line.contains("cannot combine") || line.contains("requires PLOW_")) {
                    continue;
                }
                hits += 1;
                let site = ASSERT_SITES
                    .iter()
                    .find(|(file, needle, _)| path.ends_with(file) && line.contains(needle));
                match site {
                    Some((_, _, Site::Encoded(id))) => {
                        assert!(
                            constraint_ids.contains(id),
                            "{id} is not a registered constraint"
                        )
                    }
                    Some((_, _, Site::NotAConstraint(why))) => assert!(!why.is_empty()),
                    None => unmapped.push(format!("{}: {}", path.display(), line.trim())),
                }
            }
        }
        assert!(hits >= ASSERT_SITES.len(), "found {hits} sites");
        assert!(
            unmapped.is_empty(),
            "cross-knob asserts with no registered constraint — encode them in a knob table \
             and list them in ASSERT_SITES: {unmapped:#?}"
        );
    }

    // ── The generated evaluator ─────────────────────────────────────────────────────────────

    fn val_src(v: Val) -> String {
        match v {
            Val::Unset => "Val::Unset".into(),
            Val::Bool(b) => format!("Val::Bool({b})"),
            Val::Nat(n) => format!("Val::Nat({n})"),
            Val::Str(s) => format!("Val::Str({s:?})"),
        }
    }

    fn strs_src(vs: &[&str]) -> String {
        format!(
            "&[{}]",
            vs.iter()
                .map(|s| format!("{s:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn formula_src(f: &Formula) -> String {
        let list = |fs: &[Formula]| fs.iter().map(formula_src).collect::<Vec<_>>().join(", ");
        match f {
            Formula::True => "Formula::True".into(),
            Formula::Atom(k, c, v) => format!("Formula::Atom({k:?}, Cmp::{c:?}, {})", val_src(*v)),
            Formula::Target(a) => format!("Formula::Target(TargetAtom::{a:?})"),
            Formula::Not(f) => format!("Formula::Not(&{})", formula_src(f)),
            Formula::And(fs) => format!("Formula::And(&[{}])", list(fs)),
            Formula::Or(fs) => format!("Formula::Or(&[{}])", list(fs)),
            Formula::Implies(a, b) => {
                format!("Formula::Implies(&{}, &{})", formula_src(a), formula_src(b))
            }
        }
    }

    fn spec_src(k: &KnobSpec) -> String {
        let domain = match k.domain {
            Domain::Bool => "Domain::Bool".into(),
            Domain::Nat { min, max } => format!("Domain::Nat {{ min: {min}, max: {max} }}"),
            Domain::Enum(vs) => format!("Domain::Enum({})", strs_src(vs)),
            Domain::List(vs) => format!("Domain::List({})", strs_src(vs)),
            Domain::Str => "Domain::Str".into(),
        };
        let default = match k.default {
            Default::Static(v) => format!("Default::Static({})", val_src(v)),
            Default::Production { cases, otherwise } => format!(
                "Default::Production {{ cases: &[{}], otherwise: {} }}",
                cases
                    .iter()
                    .map(|c| format!(
                        "DefaultCase {{ when: {}, value: {} }}",
                        formula_src(&c.when),
                        val_src(c.value)
                    ))
                    .collect::<Vec<_>>()
                    .join(", "),
                val_src(otherwise)
            ),
        };
        let status = match k.status {
            Status::Qualified { evidence } => {
                format!("Status::Qualified {{ evidence: {} }}", strs_src(evidence))
            }
            Status::OptIn => "Status::OptIn".into(),
            Status::Candidate { evidence } => {
                format!("Status::Candidate {{ evidence: {} }}", strs_src(evidence))
            }
            Status::Parked { reason, evidence } => format!(
                "Status::Parked {{ reason: {reason:?}, evidence: {} }}",
                strs_src(evidence)
            ),
            Status::Diagnostic => "Status::Diagnostic".into(),
            Status::Removed => "Status::Removed".into(),
        };
        format!(
            "    KnobSpec::new({:?}, {:?}, Layer::{:?}, {domain}, {default}, {status}),",
            k.id, k.env, k.layer
        )
    }

    fn expr_src(f: &Formula, index: &dyn Fn(&str) -> usize) -> String {
        let join = |fs: &[Formula], op: &str, empty: &str| {
            if fs.is_empty() {
                empty.to_string()
            } else {
                format!(
                    "({})",
                    fs.iter()
                        .map(|f| expr_src(f, index))
                        .collect::<Vec<_>>()
                        .join(op)
                )
            }
        };
        match f {
            Formula::True => "true".into(),
            Formula::Atom(k, c, v) => format!("cmp(Cmp::{c:?}, v[{}], {})", index(k), val_src(*v)),
            Formula::Target(a) => format!("TargetAtom::{a:?}.eval(t)"),
            Formula::Not(f) => format!("!{}", expr_src(f, index)),
            Formula::And(fs) => join(fs, " && ", "true"),
            Formula::Or(fs) => join(fs, " || ", "false"),
            Formula::Implies(a, b) => {
                format!("(!{} || {})", expr_src(a, index), expr_src(b, index))
            }
        }
    }

    /// The knobs the evaluator carries: every constraint var, every production-defaulted knob and
    /// the knobs its cases read — in registry order.
    fn generated_knobs() -> Vec<&'static KnobSpec> {
        let reg = registry();
        let mut wanted: Vec<&str> = Vec::new();
        for k in &reg {
            for c in k.constraints {
                c.formula.vars(&mut wanted);
            }
            if matches!(k.default, Default::Production { .. }) {
                wanted.push(k.id);
                wanted.extend(k.default.vars());
            }
        }
        reg.into_iter().filter(|k| wanted.contains(&k.id)).collect()
    }

    fn generated_source() -> String {
        let knobs = generated_knobs();
        let constraints: Vec<&Constraint> = registry().iter().flat_map(|k| k.constraints).collect();
        let index = |id: &str| {
            knobs
                .iter()
                .position(|k| k.id == id)
                .expect("generated knob")
        };
        let mut s = String::from(
            "//! @generated by `KNOB_GEN_WRITE=1 cargo test -p plowrt --lib knob_spec::tests::generated_evaluator_is_current`.\n\
             //! Do not edit.\n\
             use crate::knob::*;\n\n",
        );
        s += "#[rustfmt::skip]\npub const KNOBS: &[KnobSpec] = &[\n";
        for k in &knobs {
            s += &spec_src(k);
            s += "\n";
        }
        s += "];\n\n#[rustfmt::skip]\npub const CONSTRAINTS: &[Constraint] = &[\n";
        for c in &constraints {
            s += &format!(
                "    Constraint {{ id: {:?}, formula: {}, site: {:?}, check: Check::{:?} }},\n",
                c.id,
                formula_src(&c.formula),
                c.site,
                c.check
            );
        }
        s += "];\n\n#[rustfmt::skip]\npub const TARGETS: &[TargetSpec] = &[\n";
        for t in devgen::knob_spec::TARGETS {
            s += &target_src(t);
        }
        s += "];\n\n#[rustfmt::skip]\npub fn holds(i: usize, v: &[Val<'_>], t: &Target) -> bool {\n    match i {\n";
        for (i, c) in constraints.iter().enumerate() {
            let e = expr_src(&c.formula, &index);
            // A match arm takes the bare expression; rustc warns on the outer parentheses.
            let bare = e
                .strip_prefix('(')
                .and_then(|x| x.strip_suffix(')'))
                .filter(|x| {
                    x.chars()
                        .try_fold(0i32, |d, ch| {
                            let d = d + (ch == '(') as i32 - (ch == ')') as i32;
                            (d >= 0).then_some(d)
                        })
                        .is_some()
                })
                .unwrap_or(&e);
            s += &format!("        {i} => {bare},\n");
        }
        s += "        _ => true,\n    }\n}\n\n";
        s += "/// The first `Check::Load` constraint `v` (indexed like [`KNOBS`]) violates on `t`.\n\
              pub fn violation(v: &[Val<'_>], t: &Target) -> Option<usize> {\n    \
              (0..CONSTRAINTS.len()).find(|&i| CONSTRAINTS[i].check == Check::Load && !holds(i, v, t))\n}\n";
        s
    }

    fn target_src(t: &TargetSpec) -> String {
        format!(
            "    TargetSpec {{ name: {:?}, arch: {:?}, tp: {}, n_cu: {}, model: {:?}, caps: {}, recipe: &[{}] }},\n",
            t.name,
            t.arch,
            t.tp,
            t.n_cu,
            t.model,
            strs_src(t.caps),
            t.recipe
                .iter()
                .map(|(id, v)| format!("({id:?}, {})", val_src(*v)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    #[test]
    fn generated_evaluator_is_current() {
        let path = root().join("crates/plow-asset/src/knob_gen.rs");
        let want = generated_source();
        if std::env::var_os("KNOB_GEN_WRITE").is_some() {
            std::fs::write(&path, &want).expect("write knob_gen.rs");
        }
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            have == want,
            "plow-asset/src/knob_gen.rs is stale: rerun with KNOB_GEN_WRITE=1"
        );
        let runtime_ids: Vec<&str> = knob_gen::KNOBS
            .iter()
            .filter(|k| k.layer == Layer::Runtime)
            .map(|k| k.id)
            .collect();
        let mut readers: Vec<&str> = READERS.iter().map(|(id, _)| *id).collect();
        let mut sorted = runtime_ids.clone();
        sorted.sort_unstable();
        readers.sort_unstable();
        assert_eq!(
            readers, sorted,
            "READERS must cover exactly the runtime knobs a constraint reads"
        );
    }

    /// A deterministic xorshift stream; the Lean differential lives in lean_verify.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn sample(rng: &mut Rng, k: &KnobSpec) -> Val<'static> {
        const STRS: &[&str] = &["1", "0", "", "o_proj,band,shared", "1,2,4,8"];
        match (rng.next() % 4, k.domain) {
            (0, _) => Val::Unset,
            (_, Domain::Bool) => Val::Bool(rng.next() % 2 == 0),
            (_, Domain::Nat { .. }) => Val::Nat(rng.next() % 20),
            _ => Val::Str(STRS[(rng.next() % STRS.len() as u64) as usize]),
        }
    }

    #[test]
    fn generated_evaluator_agrees_with_the_interpreter() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let targets: Vec<Target> = knob_gen::TARGETS.iter().map(TargetSpec::target).collect();
        for _ in 0..10_000 {
            let vals: Vec<Val> = knob_gen::KNOBS
                .iter()
                .map(|k| sample(&mut rng, k))
                .collect();
            let t = &targets[(rng.next() % targets.len() as u64) as usize];
            let get = |id: &str| {
                knob_gen::KNOBS
                    .iter()
                    .position(|k| k.id == id)
                    .map_or(Val::Unset, |i| vals[i])
            };
            for (i, c) in knob_gen::CONSTRAINTS.iter().enumerate() {
                assert_eq!(
                    knob_gen::holds(i, &vals, t),
                    c.formula.eval(&get, t),
                    "{} on {vals:?}",
                    c.id
                );
            }
        }
    }

    // ── The load check ──────────────────────────────────────────────────────────────────────

    fn runtime_config() -> RuntimeConfig {
        use clap::{Args, FromArgMatches};
        let cmd = RuntimeConfig::augment_args(clap::Command::new("plowrt"));
        let m = cmd
            .try_get_matches_from(["plowrt"])
            .expect("defaults parse");
        RuntimeConfig::from_arg_matches(&m).expect("runtime config")
    }

    /// The GLM-5.3 TP8 packet's `knobs` block: its recipe resolved on its target, plus the object
    /// defines its `backends.gfx942.requires` carries.
    fn production_knobs() -> Value {
        let glm = knob_gen::TARGETS[0];
        let specs: Vec<&KnobSpec> = knob_gen::KNOBS.iter().collect();
        let src = |id: &str| plow_asset::knob::Source {
            cli: None,
            env: glm.recipe.iter().find(|(k, _)| *k == id).map(|(_, v)| *v),
        };
        let resolved = plow_asset::knob::resolve(&specs, &src, &glm.target());
        let mut values = serde_json::Map::new();
        for (id, v) in resolved {
            values.insert(id.into(), v.to_json());
        }
        values.insert("def.PLOW_DSA_PF_ARM".into(), "1".into());
        serde_json::json!({"K": "verified", "target": glm.target().to_json(), "values": values})
    }

    /// `bench100.sh`'s serve environment on the production packet.
    #[test]
    fn load_check_accepts_the_production_packet_and_serve_env() {
        let mut rt = runtime_config();
        rt.amd.mla_pf_v2 = true;
        rt.tick_log = true;
        check_knobs(&production_knobs(), &rt).unwrap();
    }

    #[test]
    fn load_check_rejects_violations() {
        let mut rt = runtime_config();
        rt.amd.mla_pf_v2 = false;
        let err = check_knobs(&production_knobs(), &rt).unwrap_err();
        assert!(err.contains("dsa_pf_packet_requires_mla_pf_v2"), "{err}");

        let rt = runtime_config();
        let mut k = production_knobs();
        k["values"]["emit.glm_xr_band"] = 2.into();
        let err = check_knobs(&k, &rt).unwrap_err();
        assert!(err.contains("seq_par_excludes_two_shot_seams"), "{err}");
    }

    #[test]
    fn load_check_reads_only_the_head() {
        let dir = std::env::temp_dir().join(format!("plowrt-knobs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let blob = dir.join("model.pkt");
        let head = serde_json::json!({"schema": 1, "arch": "gfx942", "knobs": production_knobs()});
        let mut text = serde_json::to_string(&head).unwrap();
        text.pop();
        // Not JSON past the block: a reader that went on would fail.
        text.push_str(", \"programs\": [ this is not json");
        std::fs::write(dir.join("build.json"), &text).unwrap();
        let rt = runtime_config_with_v2();
        let mut best = std::time::Duration::MAX;
        for _ in 0..20 {
            let t0 = std::time::Instant::now();
            let knobs = read_knobs(std::io::BufReader::new(
                std::fs::File::open(dir.join("build.json")).unwrap(),
            ))
            .unwrap()
            .unwrap();
            check_knobs(&knobs, &rt).unwrap();
            best = best.min(t0.elapsed());
        }
        eprintln!("knob load check: {best:?}");
        let _ = blob;
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Absent and skipped warn and load; failed and a violated constraint refuse.
    #[test]
    fn load_check_by_certificate_state() {
        let dir = std::env::temp_dir().join(format!("plowrt-knobs-k-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let blob = dir.join("model.pkt");
        let write = |v: Value| std::fs::write(dir.join("build.json"), v.to_string()).unwrap();
        let rt = runtime_config_with_v2();

        write(serde_json::json!({"schema": 1, "arch": "gfx942", "programs": []}));
        check_assets(&blob).expect("absent: a packet from before checkpoint K loads");

        let mut skipped = production_knobs();
        skipped["K"] = "skipped".into();
        skipped["reason"] = "disabled on the command line (--no-knob-verify)".into();
        check_knobs(&skipped, &rt).expect("skipped: a bring-up packet loads");

        let mut failed = production_knobs();
        failed["K"] = "rejected".into();
        let err = check_knobs(&failed, &rt).unwrap_err();
        assert!(err.contains("did not verify"), "{err}");
        write(serde_json::json!({"schema": 1, "knobs": failed}));
        let err = check_assets(&blob).unwrap_err().to_string();
        assert!(err.contains("did not verify"), "{err}");

        let mut violating = skipped;
        violating["values"]["emit.glm_xr_band"] = 2.into();
        let err = check_knobs(&violating, &rt).unwrap_err();
        assert!(err.contains("seq_par_excludes_two_shot_seams"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn runtime_config_with_v2() -> RuntimeConfig {
        let mut rt = runtime_config();
        rt.amd.mla_pf_v2 = true;
        rt
    }
}
