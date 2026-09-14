//! Unified token-batch route — the AMD engine's load-time capability gate.
//!
//! `plans/unified-token-batch.md` §1: "Every family or backend the route cannot execute is
//! refused at load or admission **with the capability named**. On AMD the interpreter's dispatch
//! `default:` writes nothing and does not trap, so a missing arm is a silent-wrong-answer
//! hazard." This module is that refusal for the dense-GQA (Phase 2) pair.
//!
//! It also draws a line the branch has been burned by three times: **armed is not fires.**
//! An object carrying the arms is armed. A step that actually ran with a descriptor covering
//! more than one request has fired. Only the second licenses a measurement, so the two are
//! separate fields of one log line rather than one word that flatters both.

use std::path::{Path, PathBuf};

/// The `extern "C" __device__` markers `runtime/amd/interp.hip` emits under
/// `PLOW_TOKEN_BATCH=1`. Each says a different thing on purpose, so a partially built object is
/// refused by NAME rather than by a single yes/no:
///
/// * `plow_token_batch_1` — the axis was compiled at all.
/// * `plow_token_batch_dense_gqa_1` — the dense-GQA operator set is present.
/// * `plow_token_batch_combined_m_1` — projections run at combined M, with no phase band.
/// * `plow_token_batch_span_attn_1` — FlashPrefill/FlashDecode read their bounds from spans.
/// * `plow_token_batch_split_merge_1` — split FlashMerge reads the prefix-free span table.
/// * `plow_token_batch_fp8_gemm_1` — required separately for FP8-weight programs.
///
/// `scripts/build_gfx942.sh`'s object contract fails a build that drops any of them, so a
/// missing marker here means a stale object, not a fresh one built wrong.
pub(super) const TOKEN_BATCH_MARKERS: [&str; 5] = [
    "plow_token_batch_1",
    "plow_token_batch_dense_gqa_1",
    "plow_token_batch_combined_m_1",
    "plow_token_batch_span_attn_1",
    "plow_token_batch_split_merge_1",
];

/// The object file and kernel the route needs. The kernel symbol is deliberately NOT
/// `plow_interp_mixed_*`: the two objects have the same shape and different dispatch, so a
/// shared name would let a stale `interp_mixed_gq.elf` answer this lookup and serve the
/// phase-band route against a descriptor that has no decode prefix.
pub(super) const TOKEN_BATCH_OBJECT: &str = "interp_tokbatch_gq.elf";

pub(super) fn token_batch_kernel_symbol(arch: &str) -> String {
    format!("plow_interp_tokbatch_{arch}_gq")
}

/// Why the route cannot run, named as a capability rather than described as a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TokenBatchRefusal {
    /// No `interp_tokbatch_gq.elf` beside the other objects.
    ObjectMissing(PathBuf),
    /// The file is there but is not a loadable ELF, or its markers are absent/wrong-valued.
    MarkersMissing {
        object: PathBuf,
        missing: Vec<&'static str>,
    },
    /// The descriptor the host would hand the device still carries a decode PREFIX: rows
    /// `[0, decode_rows)` are not covered by a span. §4.4 requires spans to cover exactly
    /// `[0, M)`, and `runtime/amd/token_batch.h` traps on anything else rather than guess.
    SpansNotPrefixFree { decode_rows: u32 },
    /// FlashPrefill's split count and output binding disagree. An unsplit flash writes its final
    /// output directly; a split flash leaves partials for the following FlashMerge.
    AttentionEpilogue { nsplit: u32, fused: bool },
    /// The route was not asked for. Distinct from every other variant: nothing is wrong, and
    /// reporting it as a capability failure would make an opt-in read like a defect.
    NotRequested,
    /// The object is armed but this blob has no prefill bucket the route can execute — every
    /// one splits its attention over KV.
    NoLegalBucket,
}

impl std::fmt::Display for TokenBatchRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ObjectMissing(p) => write!(
                f,
                "capability `token_batch_dense_gqa`: no {} — build it with \
                 scripts/build_gfx942.sh (row `interp_tokbatch`)",
                p.display()
            ),
            Self::MarkersMissing { object, missing } => write!(
                f,
                "capability `token_batch_dense_gqa`: {} lacks {missing:?}; the object predates \
                 the route or was built without PLOW_TOKEN_BATCH=1. AMD's dispatch `default:` \
                 writes nothing, so this pairing is refused rather than served",
                object.display()
            ),
            Self::SpansNotPrefixFree { decode_rows } => write!(
                f,
                "capability `token_batch_prefix_free_spans`: the plan puts {decode_rows} decode \
                 row(s) in a band ahead of the span table. §4.4 requires spans to cover exactly \
                 [0, M) with decode spans of length one included"
            ),
            Self::AttentionEpilogue { nsplit, fused } => write!(
                f,
                "capability `token_batch_attention_epilogue`: FlashPrefill nsplit={nsplit}, \
                 fused_epilogue={fused}; nsplit=1 requires the fused output and nsplit>1 \
                 requires contiguous partials for FlashMerge"
            ),
            Self::NotRequested => {
                write!(f, "disabled by --token-batch=false or PLOW_TOKEN_BATCH=0")
            }
            Self::NoLegalBucket => write!(
                f,
                "capability `token_batch_attention_epilogue`: no prefill bucket in this blob has \
                 either nsplit=1 with a fused flash epilogue or nsplit>1 with a contiguous \
                 FlashMerge partial output"
            ),
        }
    }
}

/// What the object can do, decided before anything reaches a device.
#[derive(Debug, Clone)]
pub(super) struct TokenBatchCapability {
    pub object: PathBuf,
    pub kernel: String,
    /// The object carries every arm. This is ARMED, not "in use" and not "correct".
    pub armed: bool,
    pub refusal: Option<TokenBatchRefusal>,
}

/// Read the object's `.symtab` and decide. `symbol_value` is the engine's existing ELF reader,
/// injected so this is testable without an ELF fixture: the loader reads the file before the
/// object is on a device, which is the whole point of using symbols for this.
pub(super) fn probe_token_batch(
    hsaco_dir: &Path,
    arch: &str,
    read: impl Fn(&Path) -> std::io::Result<Vec<u8>>,
    symbol_value: impl Fn(&[u8], &str) -> Option<u32>,
) -> TokenBatchCapability {
    let object = hsaco_dir.join(TOKEN_BATCH_OBJECT);
    let kernel = token_batch_kernel_symbol(arch);
    let Ok(image) = read(&object) else {
        return TokenBatchCapability {
            armed: false,
            refusal: Some(TokenBatchRefusal::ObjectMissing(object.clone())),
            object,
            kernel,
        };
    };
    let missing: Vec<&'static str> = TOKEN_BATCH_MARKERS
        .iter()
        .copied()
        .filter(|m| symbol_value(&image, m) != Some(1))
        .collect();
    if missing.is_empty() {
        TokenBatchCapability {
            object,
            kernel,
            armed: true,
            refusal: None,
        }
    } else {
        TokenBatchCapability {
            armed: false,
            refusal: Some(TokenBatchRefusal::MarkersMissing {
                object: object.clone(),
                missing,
            }),
            object,
            kernel,
        }
    }
}

/// Whether a step the host is about to build can actually run on the route.
///
/// Separate from `probe_token_batch` because the two answer different questions and the branch
/// has repeatedly conflated them: this one is about THIS PLAN, and a `Ok(())` here is what
/// licenses saying the route fired.
pub(super) fn admit_token_batch(
    decode_rows: u32,
    nsplit: u32,
    fused_epilogue: bool,
) -> Result<(), TokenBatchRefusal> {
    if decode_rows != 0 {
        return Err(TokenBatchRefusal::SpansNotPrefixFree { decode_rows });
    }
    if nsplit == 0 || ((nsplit == 1) != fused_epilogue) {
        return Err(TokenBatchRefusal::AttentionEpilogue {
            nsplit,
            fused: fused_epilogue,
        });
    }
    Ok(())
}

pub(super) fn log_route(cap: &TokenBatchCapability, ready: bool, reason: Option<&str>) {
    let refusal = cap.refusal.as_ref().map(|r| r.to_string());
    tracing::info!(
        route = "unified-token-batch/dense-gqa",
        object = %cap.object.display(),
        kernel = %cap.kernel,
        armed = cap.armed,
        ready,
        fires = false,
        reason = reason.or(refusal.as_deref()).unwrap_or("-"),
        "amd: token-batch route status"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "CPU packet inspection; set TEST_AMD_TOKEN_BATCH_ASSETS to a fresh Gemma4 packet"]
    fn fresh_gemma_packet_preserves_rungs_and_passes_unified_contracts() {
        use crate::asset::devblob::DevBlob;
        use packet::dev::{DevOp, TENSOR_NONE16};
        use plow_asset::mixed_step::TensorContract;

        let directory = PathBuf::from(std::env::var_os("TEST_AMD_TOKEN_BATCH_ASSETS").unwrap());
        let raw = std::fs::read(directory.join("model.pkt")).unwrap();
        let blob = DevBlob::parse_l2(&raw, true).unwrap();
        assert_eq!(
            blob.decode_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
            [1, 2, 4, 8, 16, 32, 64, 128]
        );
        let batch = blob.decode_progs().last().unwrap().t as usize;
        let synthesized = crate::exec::mixed_program::synthesize(&blob, batch, false).unwrap();
        assert_eq!(
            synthesized
                .programs
                .iter()
                .map(|p| p.program.rows)
                .collect::<Vec<_>>(),
            [128, 512, 1024]
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("build.json")).unwrap()).unwrap();
        let inventory: std::collections::BTreeSet<String> = manifest["objects"]["ordinary"]
            ["token_batch"]["arms"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .map(|arm| arm.split('/').next().unwrap().to_string())
            .collect();
        for inst in synthesized.programs.iter().flat_map(|p| &p.program.insts) {
            let op = DevOp::from_u16(inst.op).unwrap();
            assert!(
                inventory.contains(&format!("{op:?}")),
                "generated token-batch inventory omitted synthesized {op:?}"
            );
        }
        let mut tensors: Vec<_> = blob
            .tensors
            .iter()
            .map(|t| TensorContract {
                name: &t.name,
                bytes: t.bytes,
                initialized: t.init.is_some(),
            })
            .collect();
        for tensor in &synthesized.tensors {
            tensors.resize(
                tensors.len().max(tensor.handle as usize + 1),
                TensorContract {
                    name: "",
                    bytes: 0,
                    initialized: false,
                },
            );
            tensors[tensor.handle as usize] = TensorContract {
                name: &tensor.name,
                bytes: tensor.bytes,
                initialized: false,
            };
        }
        for spec in &synthesized.programs {
            let program = &spec.program;
            let fp8_ops = |insts: &[packet::dev::DevInst64]| {
                insts
                    .iter()
                    .filter_map(|i| {
                        matches!(
                            DevOp::from_u16(i.op),
                            Some(
                                DevOp::QuantFp8
                                    | DevOp::GemmFp8
                                    | DevOp::GemmMedFp8
                                    | DevOp::GemmSmallFp8
                                    | DevOp::GemmWideFp8
                                    | DevOp::GemmC5Fp8
                                    | DevOp::GemmGluFp8
                            )
                        )
                        .then_some(i.op)
                    })
                    .collect::<Vec<_>>()
            };
            let source = blob
                .progs
                .iter()
                .find(|p| {
                    p.t == program.rows
                        && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16)
                })
                .unwrap();
            assert_eq!(
                fp8_ops(&program.insts),
                fp8_ops(&source.insts),
                "bucket {} changed its FP8 GEMM rung",
                program.rows
            );
            assert!(program
                .insts
                .iter()
                .any(|i| i.op == DevOp::FlashPrefill as u16));
            for inst in program
                .insts
                .iter()
                .filter(|i| i.op == DevOp::FlashPrefill as u16)
            {
                admit_token_batch(0, inst.i[7], inst.t[5] != TENSOR_NONE16)
                    .unwrap_or_else(|e| panic!("bucket {}: {e}", program.rows));
            }
            plow_asset::mixed_step::dense_amd_capacity_consumer_contract(
                program,
                spec.decode_rows,
                &tensors,
            )
            .unwrap();
            plow_asset::token_batch::Capabilities::amd_dense_gqa(
                "gfx942",
                program.rows,
                spec.decode_rows,
            )
            .refuse_program(program.insts.iter().map(|i| i.op))
            .unwrap();
        }
    }

    #[test]
    #[ignore = "CPU ELF inspection; set TEST_AMD_TOKEN_BATCH_OBJECTS to the compiled object directory"]
    fn compiled_objects_advertise_token_batch_and_reject_mixed_control() {
        use super::super::{elf_symbol_names, elf_symbol_u32};

        let directory = PathBuf::from(std::env::var_os("TEST_AMD_TOKEN_BATCH_OBJECTS").unwrap());
        let cap = probe_token_batch(&directory, "gfx942", |p| std::fs::read(p), elf_symbol_u32);
        assert!(cap.armed, "{:?}", cap.refusal);
        let mut packet_hash = None;
        for (name, suffix) in [("interp_tokbatch.elf", ""), (TOKEN_BATCH_OBJECT, "_gq")] {
            let image = std::fs::read(directory.join(name)).unwrap();
            let symbol = format!("plow_interp_tokbatch_gfx942{suffix}");
            assert!(elf_symbol_names(&image).contains(&symbol.as_str()));
            for marker in TOKEN_BATCH_MARKERS.into_iter().chain([
                "plow_mixed_dynamic_rows_1",
                "plow_mixed_step_bf16_1",
                "plow_mixed_gemm_glu_1",
                "plow_mixed_prefill_split_1",
            ]) {
                assert_eq!(elf_symbol_u32(&image, marker), Some(1), "{name}: {marker}");
            }
            assert_eq!(elf_symbol_u32(&image, "plow_mixed_block"), Some(256));
            let hash = (
                elf_symbol_u32(&image, "plow_packet_hash_hi").unwrap(),
                elf_symbol_u32(&image, "plow_packet_hash_lo").unwrap(),
            );
            assert_eq!(*packet_hash.get_or_insert(hash), hash, "scheduler twins");
        }
        let mixed = std::fs::read(directory.join("interp_mixed_gq.elf")).unwrap();
        let cap = probe_token_batch(&directory, "gfx942", |_| Ok(mixed.clone()), elf_symbol_u32);
        assert!(!cap.armed);
        assert!(matches!(
            cap.refusal,
            Some(TokenBatchRefusal::MarkersMissing { .. })
        ));
    }

    fn syms(present: &[&str]) -> impl Fn(&[u8], &str) -> Option<u32> {
        let owned: Vec<String> = present.iter().map(|s| s.to_string()).collect();
        move |_img: &[u8], want: &str| owned.iter().any(|s| s == want).then_some(1)
    }

    #[test]
    fn a_missing_object_is_refused_by_capability_name() {
        let cap = probe_token_batch(
            Path::new("/nonexistent"),
            "gfx942",
            |_| Err(std::io::Error::other("nope")),
            syms(&[]),
        );
        assert!(!cap.armed);
        let text = cap.refusal.unwrap().to_string();
        assert!(text.contains("token_batch_dense_gqa"), "{text}");
        assert!(text.contains("interp_tokbatch"), "{text}");
    }

    #[test]
    fn a_partially_built_object_names_the_arms_it_lacks() {
        let cap = probe_token_batch(
            Path::new("/objects"),
            "gfx942",
            |_| Ok(vec![0u8; 4]),
            syms(&["plow_token_batch_1", "plow_token_batch_dense_gqa_1"]),
        );
        assert!(!cap.armed);
        let text = cap.refusal.unwrap().to_string();
        assert!(text.contains("plow_token_batch_combined_m_1"), "{text}");
        assert!(text.contains("plow_token_batch_span_attn_1"), "{text}");
        assert!(text.contains("plow_token_batch_split_merge_1"), "{text}");
    }

    #[test]
    fn a_complete_object_is_armed() {
        let cap = probe_token_batch(
            Path::new("/objects"),
            "gfx942",
            |_| Ok(vec![0u8; 4]),
            syms(&TOKEN_BATCH_MARKERS),
        );
        assert!(cap.armed);
        assert!(cap.refusal.is_none());
        assert_eq!(cap.kernel, "plow_interp_tokbatch_gfx942_gq");
    }

    /// The kernel symbol must differ from the mixed object's: a shared name would let a stale
    /// `interp_mixed_gq.elf` answer this lookup and serve the phase-band dispatch against a
    /// descriptor with no decode prefix.
    #[test]
    fn the_kernel_symbol_is_not_the_mixed_objects() {
        assert_ne!(
            token_batch_kernel_symbol("gfx942"),
            "plow_interp_mixed_gfx942_gq"
        );
    }

    #[test]
    fn a_decode_prefix_is_refused_not_silently_packed() {
        let err = admit_token_batch(3, 1, true).unwrap_err();
        assert_eq!(
            err,
            TokenBatchRefusal::SpansNotPrefixFree { decode_rows: 3 }
        );
        assert!(err.to_string().contains("token_batch_prefix_free_spans"));
    }

    #[test]
    fn split_attention_requires_merge_partials() {
        assert!(admit_token_batch(0, 4, false).is_ok());
        let err = admit_token_batch(0, 4, true).unwrap_err();
        assert!(err.to_string().contains("token_batch_attention_epilogue"));
    }

    #[test]
    fn unsplit_attention_requires_fused_epilogue() {
        let err = admit_token_batch(0, 1, false).unwrap_err();
        assert!(err.to_string().contains("token_batch_attention_epilogue"));
    }

    #[test]
    fn a_prefix_free_unsplit_fused_plan_is_admitted() {
        assert!(admit_token_batch(0, 1, true).is_ok());
    }
}
