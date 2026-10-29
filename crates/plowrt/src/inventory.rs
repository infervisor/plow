//! Static capability inventory of a packet: which plowrt routes the packet DECLARES, each with
//! the reason it is on, off or not applicable. CPU only, like `disasm`. The doctor runs it before
//! a lease and refuses an `off` the recipe does not waive, so a campaign measures a packet with
//! every route it can carry instead of rediscovering the runtime's capabilities one by one.
use crate::asset::devblob::DevBlob;
use crate::exec::terminal_layout::layout;
use packet::dev::{DevInst64, DevOp};
use packet::devbuild::ProgramRole;
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    On,
    Off,
    NotApplicable,
}

#[derive(Clone, Debug, Serialize)]
pub struct Item {
    pub name: &'static str,
    pub state: State,
    pub detail: String,
}

fn item(name: &'static str, state: State, detail: impl Into<String>) -> Item {
    Item {
        name,
        state,
        detail: detail.into(),
    }
}

fn op_name(op: u16) -> String {
    DevOp::from_u16(op).map_or_else(|| format!("op{op}"), |o| format!("{o:?}"))
}

fn tail_ops(insts: &[DevInst64]) -> String {
    let n = insts.len();
    insts[n.saturating_sub(3)..]
        .iter()
        .map(|d| op_name(d.op))
        .collect::<Vec<_>>()
        .join(", ")
}

fn files_with(assets: &Path, prefix: &str, suffix: &str) -> Vec<String> {
    let Ok(dir) = std::fs::read_dir(assets) else {
        return Vec::new();
    };
    let mut out: Vec<String> = dir
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(suffix))
        .collect();
    out.sort();
    out
}

/// The inventory of `blob` (parsed from `raw`), with the object-side facts `assets` carries
/// (generated roles, the sampler object, the emit replay in `build.json`) when it is given.
pub fn inventory(blob: &DevBlob, raw: &[u8], assets: Option<&Path>) -> crate::Result<Vec<Item>> {
    let mut out = Vec::new();
    let mut buckets: Vec<u32> = Vec::new();
    let mut rungs: Vec<u32> = Vec::new();
    for p in &blob.progs {
        match p.role {
            ProgramRole::PrefillBucket { rows } => buckets.push(rows),
            ProgramRole::DecodeRung { rows } => rungs.push(rows),
            _ => {}
        }
    }
    buckets.sort_unstable();
    rungs.sort_unstable();
    let list = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    out.push(item(
        "prefill_buckets",
        if buckets.is_empty() {
            State::Off
        } else {
            State::On
        },
        format!("rows {}", list(&buckets)),
    ));
    out.push(match rungs.len() {
        0 => item("decode_ladder", State::NotApplicable, "no decode rung"),
        1 => item(
            "decode_ladder",
            State::Off,
            format!(
                "one decode rung ({} rows): every batch pays the widest body",
                rungs[0]
            ),
        ),
        _ => item(
            "decode_ladder",
            State::On,
            format!("rungs {}", list(&rungs)),
        ),
    });

    let recurrent: Vec<&str> = blob
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("state."))
        .map(|t| t.name.as_str())
        .collect();
    if recurrent.is_empty() {
        out.push(item(
            "recurrent_state",
            State::NotApplicable,
            "no state.* tensors",
        ));
        out.push(item(
            "prefix_cache",
            State::On,
            "KV prefix reuse available (PLOW_PREFIX_CACHE / PLOW_VMM_PREFIX at serve)",
        ));
    } else {
        out.push(item(
            "recurrent_state",
            State::On,
            format!("{} per-layer state tensors", recurrent.len()),
        ));
        // The block store's State class captures every declared buffer at a boundary
        // (`exec/gpu/kv_blocks.rs`); a `state.` tensor the section does not declare is refused at
        // load (`plow_asset::recurrent_state`).
        let declared = blob
            .reserved_metadata(raw, plow_asset::recurrent_state::SECTION)?
            .map(|bytes| {
                serde_json::from_slice::<plow_asset::recurrent_state::Manifest>(bytes)
                    .map(|m| m.buffers.len())
            })
            .transpose()
            .map_err(|e| crate::RuntimeError::Rejected(format!("recurrent_state: {e}")))?;
        out.push(match declared {
            Some(n) if n > 0 => item(
                "prefix_cache",
                State::On,
                format!(
                    "recurrent state captured at boundaries ({n} declared buffers); CUDA does not \
                     auto-select it: serve with PLOW_VMM_PREFIX=1"
                ),
            ),
            _ => item(
                "prefix_cache",
                State::Off,
                "state.* tensors without declared recurrent_state buffers: the packet is refused",
            ),
        });
    }

    let live = blob.reserved_metadata(raw, plow_asset::live_kv::SECTION)?;
    out.push(match live {
        Some(bytes) => {
            let v: serde_json::Value = serde_json::from_slice(bytes)
                .map_err(|e| crate::RuntimeError::Rejected(format!("live_kv section: {e}")))?;
            item(
                "live_kv",
                State::On,
                format!(
                    "LIVE KV contract v{} (per-slot commit under PLOW_VMM_LIVE)",
                    v.get("version").and_then(|x| x.as_u64()).unwrap_or(0)
                ),
            )
        }
        None => item("live_kv", State::Off, "no live_kv section: flat KV only"),
    });

    let packed = blob
        .reserved_metadata(raw, plow_asset::packed_prefill::SECTION)?
        .map(|bytes| {
            serde_json::from_slice::<plow_asset::packed_prefill::Manifest>(bytes)
                .map_err(|e| crate::RuntimeError::Rejected(format!("packed_prefill section: {e}")))
        })
        .transpose()?;
    let mut compact_terminal = false;
    match &packed {
        None => {
            out.push(item(
                "packed_prefill",
                State::Off,
                "no packed_prefill section: one prompt per prefill launch",
            ));
            out.push(item(
                "masked_padding",
                State::NotApplicable,
                "no packed prefill",
            ));
            out.push(item(
                "compact_terminal",
                State::NotApplicable,
                "no packed prefill",
            ));
        }
        Some(m) => {
            out.push(item(
                "packed_prefill",
                State::On,
                format!(
                    "contract v{}: requests share one prefill launch{}{}",
                    m.version,
                    if m.maps.is_empty() {
                        String::new()
                    } else {
                        format!(", {} slot maps", m.maps.len())
                    },
                    match (m.stages.len(), m.stage_rows) {
                        (0, _) => String::new(),
                        (n, Some(r)) => format!(", {n} stages of {r} rows"),
                        (n, None) => format!(", {n} stages"),
                    }
                ),
            ));
            out.push(match m.max_request_rows {
                Some(rows) => item(
                    "masked_padding",
                    State::On,
                    format!("pad rows carry slot -1; requests chunk at {rows} rows"),
                ),
                None => item(
                    "masked_padding",
                    State::Off,
                    "unmasked: a launch's pad rows charge one request's physical context",
                ),
            });
            let ids = blob.tensors.iter().position(|t| t.name == "in.ids");
            let logits = blob.tensors.iter().position(|t| t.name == "act.logits");
            let prefill: Vec<_> = blob
                .progs
                .iter()
                .filter(|p| matches!(p.role, ProgramRole::PrefillBucket { .. }))
                .collect();
            let terminal = match (ids, logits) {
                (Some(ids), Some(logits)) => prefill.iter().all(|p| {
                    let ProgramRole::PrefillBucket { rows } = p.role else {
                        return false;
                    };
                    layout(&p.insts, rows, logits, ids).is_some()
                }),
                _ => false,
            };
            compact_terminal = terminal;
            out.push(if terminal {
                let post_norm = prefill.first().is_some_and(|p| {
                    p.insts
                        .len()
                        .checked_sub(4)
                        .is_some_and(|i| p.insts[i].op != DevOp::RmsNorm as u16)
                });
                item(
                    "compact_terminal",
                    State::On,
                    format!(
                        "{} tail: the first token comes from the prefill launch (object must carry the row-gather arm)",
                        if post_norm { "post-norm" } else { "norm+head" }
                    ),
                )
            } else {
                item(
                    "compact_terminal",
                    State::Off,
                    format!(
                        "prefill tail [{}] is not a terminal layout: the mux packs n-1 rows and takes the first token from a decode step",
                        prefill.first().map_or(String::new(), |p| tail_ops(&p.insts))
                    ),
                )
            });
        }
    }

    out.push(
        match blob.reserved_metadata(raw, plow_asset::segment_roles::SECTION)? {
            Some(bytes) => {
                let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
                    crate::RuntimeError::Rejected(format!("segment_roles section: {e}"))
                })?;
                let objects = v
                    .get("objects")
                    .and_then(|o| o.as_object())
                    .map_or(0, |o| o.len());
                let launches: usize =
                    v.get("programs")
                        .and_then(|p| p.as_array())
                        .map_or(0, |ps| {
                            ps.iter()
                                .map(|p| {
                                    p.get("roles")
                                        .and_then(|r| r.as_array())
                                        .map_or(0, |r| r.len())
                                })
                                .sum()
                        });
                item(
                    "segment_roles",
                    State::On,
                    format!("{objects} segment objects, {launches} role launches declared"),
                )
            }
            None => item(
                "segment_roles",
                State::Off,
                "no segment_roles section: no cuBLASLt or generated role launches",
            ),
        },
    );

    // CUDA arms the unified token batch from the packed-prefill contract and the compact terminal
    // (`exec/gpu/token_batch.rs`); a recurrent chain needs its packed scan's init-state scratch.
    let token_batch = blob.reserved_metadata(raw, plow_asset::token_batch::SECTION)?;
    let packs_state =
        recurrent.is_empty() || blob.tensors.iter().any(|t| t.name == "act.gdn.initstate");
    out.push(match (token_batch, packed.is_some() && compact_terminal, packs_state) {
        (Some(_), _, _) => item("token_batch", State::On, "token-batch bodies declared"),
        (None, true, true) => item(
            "token_batch",
            State::On,
            "decode rows ride the packed prefill launch (packed prefill + compact terminal)",
        ),
        (None, true, false) => item(
            "token_batch",
            State::Off,
            "recurrent state without a packed init-state scratch: decode rows never ride a prefill launch",
        ),
        (None, false, _) => item(
            "token_batch",
            State::Off,
            "needs packed prefill and a compact terminal: decode rows never ride a prefill launch",
        ),
    });

    if let Some(assets) = assets {
        let gen = files_with(assets, "gen_", ".cubin");
        out.push(if gen.is_empty() {
            item(
                "generated_kernels",
                State::Off,
                "no gen_*.cubin role object",
            )
        } else {
            item("generated_kernels", State::On, gen.join(", "))
        });
        let sampler = files_with(assets, "sample_", ".cubin");
        out.push(if sampler.is_empty() {
            item(
                "device_sampler",
                State::Off,
                "no sample_*.cubin: host sampling",
            )
        } else {
            item("device_sampler", State::On, sampler.join(", "))
        });
        let build = std::fs::read_to_string(assets.join("build.json")).ok();
        out.push(match &build {
            Some(text) if text.contains("\"PLOW_EMIT_PREFILL_CUBLASLT\": \"1\"") => item(
                "cublaslt_prefill",
                State::On,
                "PLOW_EMIT_PREFILL_CUBLASLT=1 in the emit replay",
            ),
            Some(_) => item(
                "cublaslt_prefill",
                State::Off,
                "emit replay has no PLOW_EMIT_PREFILL_CUBLASLT=1: projections on the interpreter GEMM",
            ),
            None => item("cublaslt_prefill", State::NotApplicable, "no build.json next to the packet"),
        });
    }
    Ok(out)
}

pub fn text(items: &[Item]) -> String {
    let width = items.iter().map(|i| i.name.len()).max().unwrap_or(0);
    let mut s = String::new();
    for i in items {
        let state = match i.state {
            State::On => "on ",
            State::Off => "OFF",
            State::NotApplicable => "n/a",
        };
        s.push_str(&format!(
            "{:width$}  {state}  {}\n",
            i.name,
            i.detail,
            width = width
        ));
    }
    s
}
