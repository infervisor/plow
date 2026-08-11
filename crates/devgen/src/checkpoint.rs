//! Checkpoint (safetensors) reads: shard discovery, per-layer `layer_scalar`
//! immediates, and tensor-name coverage. Split out of `lib.rs` (module breakdown);
//! pure I/O — no `Cfg`/`Builder` deps. `layer_scalars`/`validate_coverage` are the
//! cross-module entry points; the rest are module-private.
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Read the `layer_scalar` values out of the checkpoint.
///
/// The RESIDUAL op takes its scale as an IMMEDIATE in the packet, not as a tensor — so the
/// compiler, not the runtime, has to know it. That means plowc reads the safetensors
/// headers. This is the right place for it: `layer_scalar` is a compile-time constant of
/// the network, exactly like the tile size.
/// Discover the checkpoint's shard files, newest-bug-first.
///
/// FIXED: this used to read `model.safetensors.index.json` unconditionally and
/// panic on the Gemma-4 12B checkpoint, which is a single unsharded
/// `model.safetensors` with **no index file at all**. It also trusted the index
/// on the 31B *partial* checkpoint, whose `model.safetensors.index.json` names
/// files (`model-0000N-of-00002.safetensors`) that do not exist on disk — only
/// the `.partial.safetensors` ones do.
///
/// So don't trust the index: enumerate what is actually there. Same resolution
/// order as `plowrt::memory::container::Safetensors::open_dir` — a complete
/// non-partial shard set, else a partial set, else single-file
/// `model.safetensors`.
fn shard_files(dir: &Path) -> Vec<PathBuf> {
    let mut sets: HashMap<(u32, bool), Vec<(u32, PathBuf)>> = HashMap::new();
    let mut single = None;
    for ent in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let ent = ent.unwrap();
        let fname = ent.file_name();
        let Some(f) = fname.to_str() else { continue };
        if f == "model.safetensors" {
            single = Some(ent.path());
            continue;
        }
        // Suffix must be exactly ".safetensors" so sidecars like
        // "model-00001-of-00002.safetensors.header.json" don't match.
        let Some(rest) = f.strip_prefix("model-") else {
            continue;
        };
        let Some((i, rest)) = rest.split_once("-of-") else {
            continue;
        };
        let (t, partial) = match rest.strip_suffix(".partial.safetensors") {
            Some(t) => (t, true),
            None => match rest.strip_suffix(".safetensors") {
                Some(t) => (t, false),
                None => continue,
            },
        };
        let (Ok(i), Ok(t)) = (i.parse::<u32>(), t.parse::<u32>()) else {
            continue;
        };
        sets.entry((t, partial)).or_default().push((i, ent.path()));
    }
    let mut complete: Vec<_> = sets
        .iter()
        .filter(|((t, _), v)| v.len() as u32 == *t)
        .collect();
    // Prefer the non-partial set when both are complete at the same total.
    complete.sort_by_key(|((t, p), _)| (*p, *t));
    if complete.len() > 1 {
        let keys: Vec<_> = complete.iter().map(|(k, _)| **k).collect();
        assert!(
            keys.len() == 2 && keys[0].0 == keys[1].0 && keys[0].1 != keys[1].1,
            "{}: ambiguous checkpoint — {} complete shard sets {keys:?}; a stray shard-named \
             file silently changes what loads",
            dir.display(),
            keys.len()
        );
    }
    if let Some((_, v)) = complete.first() {
        let mut v = (*v).clone();
        v.sort_by_key(|(i, _)| *i);
        return v.into_iter().map(|(_, p)| p).collect();
    }
    if let Some((k, v)) = sets.iter().next() {
        panic!(
            "{}: incomplete shard set (-of-{:05}{}): {} of {} present",
            dir.display(),
            k.0,
            if k.1 { " .partial" } else { "" },
            v.len(),
            k.0
        );
    }
    // THE fallback that was missing: single-file, no index (Gemma-4 12B).
    vec![single.unwrap_or_else(|| {
        panic!(
            "{}: no safetensors checkpoint (looked for \
             model-{{i}}-of-{{n}}[.partial].safetensors and model.safetensors)",
            dir.display()
        )
    })]
}

pub(crate) fn layer_scalars(dir: &Path, layers: u32, prefix: &str) -> Vec<f32> {
    // Config-only / net-json emit: with no safetensors there is nothing to read
    // the learned scalar from, but `layer_scalar` is only a RESIDUAL immediate —
    // it changes numerics, never the packet structure or the counter graph. Fall
    // back to the identity scale (1.0) so a structural devblob emit succeeds, and
    // say so on stderr, mirroring the coverage gate's own no-safetensors skip.
    // `shard_files` PANICS on an empty dir, so this guard must precede it.
    let has_shards = std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(Result::ok)
            .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
    });
    if !has_shards {
        eprintln!(
            "devgen: {} has no .safetensors — layer_scalar defaults to 1.0 \
             (structural emit; weights are unbound and numerics are not representative)",
            dir.display()
        );
        return vec![1.0f32; layers as usize];
    }
    // name -> shard file, built from the files that actually exist rather than
    // from an index that may be absent (12B) or stale (31B partial).
    let mut hdr_cache: HashMap<PathBuf, (Value, u64)> = HashMap::new();
    let mut map: HashMap<String, PathBuf> = HashMap::new();
    for p in shard_files(dir) {
        // Read ONLY the header. The old code did `fs::read(&p)` — the whole
        // shard — to fetch a handful of `[1]` scalars; that is 23 GB of I/O on
        // the 12B checkpoint.
        use std::io::Read;
        let mut f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut len = [0u8; 8];
        f.read_exact(&mut len)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let n = u64::from_le_bytes(len);
        let mut hbuf = vec![0u8; n as usize];
        f.read_exact(&mut hbuf)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let h: Value = serde_json::from_slice(&hbuf)
            .unwrap_or_else(|e| panic!("{}: bad safetensors header: {e}", p.display()));
        for k in h.as_object().expect("header object").keys() {
            if k != "__metadata__" {
                map.insert(k.clone(), p.clone());
            }
        }
        hdr_cache.insert(p, (h, 8 + n));
    }
    let mut out = Vec::with_capacity(layers as usize);
    for l in 0..layers {
        let name = format!("{prefix}layers.{l}.layer_scalar");
        let path = map
            .get(&name)
            .unwrap_or_else(|| panic!("checkpoint has no {name}"))
            .clone();
        let (hdr, data0) = &hdr_cache[&path];
        let (data0, path) = (*data0, path.clone());
        let ent = &hdr[&name];
        assert_eq!(ent["dtype"].as_str().unwrap(), "BF16", "{name} dtype");
        let off = ent["data_offsets"][0].as_u64().unwrap();
        let mut f = std::fs::File::open(path).unwrap();
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::Start(data0 + off)).unwrap();
        let mut b = [0u8; 2];
        f.read_exact(&mut b).unwrap();
        let bits = (u16::from_le_bytes(b) as u32) << 16;
        out.push(f32::from_bits(bits));
    }
    out
}

/// Every tensor name the checkpoint actually ships, unfiltered.
fn ckpt_names_all(dir: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    for p in shard_files(dir) {
        use std::io::Read;
        let mut f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut len = [0u8; 8];
        f.read_exact(&mut len)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut hbuf = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut hbuf)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let h: Value = serde_json::from_slice(&hbuf)
            .unwrap_or_else(|e| panic!("{}: bad safetensors header: {e}", p.display()));
        for k in h.as_object().expect("header object").keys() {
            if k != "__metadata__" {
                out.insert(k.clone());
            }
        }
    }
    out
}

pub(crate) fn validate_bf16_sidecar(
    dir: &Path,
    filename: &str,
    expected: &[(String, Vec<u64>)],
) -> Result<(), String> {
    use std::io::Read;

    let path = dir.join(filename);
    let mut file = std::fs::File::open(&path).map_err(|e| {
        format!(
            "{}: required derived-weight sidecar is unavailable: {e}",
            path.display()
        )
    })?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(|e| {
        format!(
            "{}: cannot read safetensors header length: {e}",
            path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|e| format!("{}: cannot read safetensors header: {e}", path.display()))?;
    let header: Value = serde_json::from_slice(&header)
        .map_err(|e| format!("{}: bad safetensors header: {e}", path.display()))?;
    let header = header
        .as_object()
        .ok_or_else(|| format!("{}: safetensors header is not an object", path.display()))?;
    let data_bytes = file
        .metadata()
        .map_err(|e| format!("{}: cannot stat sidecar: {e}", path.display()))?
        .len()
        .checked_sub(8 + header_len)
        .ok_or_else(|| format!("{}: truncated safetensors header", path.display()))?;

    for (name, shape) in expected {
        let entry = header
            .get(name)
            .ok_or_else(|| format!("{}: missing required tensor {name}", path.display()))?;
        if entry["dtype"].as_str() != Some("BF16") {
            return Err(format!(
                "{}: {name} must be BF16, got {}",
                path.display(),
                entry["dtype"]
            ));
        }
        let got_shape: Vec<u64> = entry["shape"]
            .as_array()
            .ok_or_else(|| format!("{}: {name} has no shape", path.display()))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| format!("{}: {name} has a non-integer shape", path.display()))
            })
            .collect::<Result<_, _>>()?;
        if &got_shape != shape {
            return Err(format!(
                "{}: {name} shape {got_shape:?}, expected {shape:?}",
                path.display()
            ));
        }
        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or_else(|| format!("{}: {name} has no data_offsets", path.display()))?;
        if offsets.len() != 2 {
            return Err(format!(
                "{}: {name} has invalid data_offsets",
                path.display()
            ));
        }
        let lo = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("{}: {name} has invalid start offset", path.display()))?;
        let hi = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("{}: {name} has invalid end offset", path.display()))?;
        let want_bytes = shape.iter().try_fold(2u64, |n, dim| n.checked_mul(*dim));
        if want_bytes != hi.checked_sub(lo) || hi > data_bytes {
            return Err(format!(
                "{}: {name} byte range [{lo},{hi}) is inconsistent with BF16 {shape:?}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// The namespaces this checkpoint puts its transformer layers under, DERIVED from the tensor
/// names rather than assumed, with a count each.
///
/// `<something>.layers.<N>.<...>` is the one naming rule every architecture in this tree shares,
/// so the text before the first `.layers.` *is* the weight prefix. Measured, on the checkpoints on
/// this box:
///
/// | checkpoint | derived |
/// |---|---|
/// | GLM-5.2, Llama, Qwen3 | `model.` |
/// | Gemma-4 multimodal | `model.language_model.` (+ a vision-tower namespace) |
/// | Kimi-K3 | `language_model.model.` — and **nothing** under `model.` |
///
/// Returned as a map because a multimodal checkpoint legitimately has more than one (the vision
/// tower has `.layers.` too). The caller does not need to pick the right one — it only needs to
/// know whether the prefix it was *given* is among them.
fn layer_namespaces(names: &HashSet<String>) -> std::collections::BTreeMap<String, usize> {
    let mut out = std::collections::BTreeMap::new();
    for n in names {
        if n.starts_with("layers.") {
            *out.entry("layers.".to_string()).or_insert(0) += 1;
        } else if let Some((head, _)) = n.split_once(".layers.") {
            *out.entry(format!("{head}.layers.")).or_insert(0) += 1;
        }
    }
    out
}

/// Bidirectional checkpoint coverage gate.
///
/// `--hf-dir` has had this since day one (`hf_config::validate_against_checkpoint`,
/// and `tests/hf_dir_compile.rs` calls the two failure modes out by name), but THIS
/// binary — the one every asset-build script actually runs — had no check at all. It
/// declares weights by name and never reads the checkpoint back, and the runtime's
/// only net (`plowrt::memory::container`) errors on a MISSING name, never on an
/// unused one. That is a pull model: extra checkpoint tensors are simply never
/// looked up.
///
/// Measured consequence, gemma-4-E4B-it: after one null-default the emitter produced
/// a clean, loadable, warning-free packet that had silently dropped **5.4 GiB** of
/// per-layer-embedding weights (reporting `weights 8.6 GiB` for a 14.0 GiB model).
/// It would have loaded and generated fluent, wrong text. Both directions matter:
/// the forward check catches a typo'd/renamed weight, the reverse check catches an
/// architecture the emitter does not implement.
///
/// Only `prefix*` names participate. Activations (`act.`), inputs (`in.`), KV rings
/// (`kv.`), compiler-materialised tables (rope) and the fp8 twins (`fp8/`, which live
/// in a sibling directory, not `dir`) are all out of scope by construction.
/// Layer index in a `...layers.<N>....` tensor name, if it has one.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name
        .strip_prefix("layers.")
        .or_else(|| name.split_once(".layers.").map(|(_, rest)| rest))?;
    rest.split('.').next()?.parse().ok()
}

pub(crate) fn validate_coverage(
    dir: &Path,
    prefix: &str,
    declared: &[String],
    // `--block l..r`: the plan deliberately covers ONLY these layers, so every
    // other layer's weights — and the global ones (embeddings, final norm,
    // lm_head) — are legitimately uncovered. Without this the reverse check
    // reads a block asset as "an architecture the emitter does not implement"
    // and refuses to emit, which is what kept the single-block harness
    // (`examples/block_run.rs`) unusable on any real checkpoint. The FORWARD
    // check is untouched: a weight the block plan binds must still exist.
    block: Option<std::ops::Range<usize>>,
    // Substrings marking a checkpoint tensor as legitimately consumed WITHOUT being declared.
    //
    // The reverse check's whole value is that an unclaimed weight is a missing op, so every entry
    // here is a hole punched in it and must name a mechanism that actually reads the bytes —
    // never "we do not use this". Kimi-K3 needs three, and each is a different mechanism:
    //
    //   `.experts.`        bound by NAME PATTERN, not as packet tensors (`bind_packed_experts`,
    //                      plowrt exec/amd.rs:1621). 494,592 tensors; declaring them would put
    //                      the whole 1.5 TB expert set in the blob's tensor table.
    //   `_res_norm.weight` / `_res_proj.weight`
    //                      FOLDED at load: plowrt's `fold_res_score` (exec/amd.rs:1912) derives
    //                      the declared `*_res_score.weight` [H] f32 from the norm/proj pair,
    //                      because `score_weight = norm.weight * proj.weight.squeeze(0)` and the
    //                      model never uses either factor alone.
    //   `q_b_proj` / `kv_b_proj`
    //                      ABSORBED host-side into `derived.{q_absorb,q_rope,v_absorb}` by
    //                      scripts/kimi_k3_prep.py. The raw factors are genuinely dead once
    //                      absorbed — verified numerically equivalent to ~1e-6 relative.
    //
    // A waiver is a SUBSTRING and therefore blunt: `_res_proj.weight` waives the model-level
    // `output_attn_res_proj.weight` too. That is intended here (same fold, same mechanism) but it
    // is exactly how `output_attn_res` stayed invisible, so widen one only with the mechanism in
    // hand.
    indirect: &[&str],
    // CONDITIONAL waivers: `(suffix, produced_suffix)`. A checkpoint tensor ending `suffix` is
    // covered ONLY IF the plan declares the same name with `suffix` replaced by
    // `produced_suffix` — i.e. only if the thing that consumes it is actually emitted.
    //
    // This exists because a flat substring waiver is what HID the original bug. K3's
    // `output_attn_res_{norm,proj}` end in `_res_norm.weight`/`_res_proj.weight`, so any blanket
    // waiver for the per-layer pairs silently covers the model-level pair too — and the model-level
    // AttnRes being absent was precisely the defect. Keyed on the produced name, a dropped op
    // stops declaring `…_res_score.weight`, its two factors go unclaimed, and the gate fires.
    paired: &[(&str, &str)],
    // The mirror of `indirect`: substrings marking a DECLARED name the checkpoint legitimately
    // does not contain, because something produces it before the bind rather than shipping it.
    // Same rule — each entry names a producer, never "it is fine that this is absent".
    synthesized: &[&str],
) -> Result<(), String> {
    // A directory with NO safetensors at all is not a coverage failure -- there is nothing to
    // cross-check against. Emitting a blob from a bare config.json is legitimate (the structural
    // golden tests do exactly that), and `shard_files` PANICS rather than returning empty, so
    // reaching it here aborts a valid emit. Say so on stderr rather than skipping silently: a
    // genuinely missing checkpoint is still visible, it just no longer kills the process from
    // inside a gate whose whole purpose is comparing two sets, one of which does not exist.
    let has_shards = std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(Result::ok)
            .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
    });
    if !has_shards {
        eprintln!(
            "devgen: {} has no .safetensors — skipping the checkpoint coverage gate \
             (nothing to compare the plan against)",
            dir.display()
        );
        return Ok(());
    }
    // THE PREFIX ITSELF IS CHECKED FIRST, because a wrong one turns this entire gate into a
    // no-op rather than a failure: `ckpt` and `want` are both filtered by it, so a prefix that
    // matches nothing compares the empty set against the empty set and returns Ok. That is the
    // §4 shape one level up — the check exists, is correct, and nothing routes to it.
    //
    // Kimi-K3 is the live case: all 497 052 language-tower tensors are `language_model.model.…`
    // and ZERO start with `model.`, so a plan declared under `model.` would pass this gate
    // silently, allocate every weight, upload none, and decode from zeroed memory.
    let all = ckpt_names_all(dir);
    let ckpt: HashSet<String> = all
        .iter()
        .filter(|k| k.starts_with(prefix))
        .cloned()
        .collect();
    if ckpt.is_empty() {
        let ns = layer_namespaces(&all);
        return Err(format!(
            "weight prefix `{prefix}` matches NOTHING in {} ({} tensors).\n\
             The coverage gate below compares declared-vs-shipped WITHIN the prefix, so an \
             unmatched prefix makes it compare two empty sets and pass — every weight would then \
             be allocated, never uploaded, and read as zeros at run time.\n\
             This checkpoint's own layer namespaces, derived from its tensor names:\n    {}\n\
             Fix the prefix the plan declares (devgen `Cfg::prefix` / `GlmCfg::prefix`, \
             plowc `HfSynthesis::prefix`), not this gate.",
            dir.display(),
            all.len(),
            if ns.is_empty() {
                "(none — no tensor name contains `.layers.`)".to_string()
            } else {
                ns.iter()
                    .map(|(p, n)| format!("{p}*  ({n} tensors)"))
                    .collect::<Vec<_>>()
                    .join("\n    ")
            },
        ));
    }
    // A weight is covered if the plan binds EITHER the bf16 tensor or its fp8 twin.
    // Under PLOW_FP8 the projections are declared as `fp8/<name>` (the twins live in a
    // sibling dir, so they are not in `ckpt`) and the bf16 original is deliberately
    // superseded — counting it "uncovered" would fail every fp8 build.
    //
    // THIS STRIP IS A COVERAGE MAPPING, NOT A KEY LOOKUP. It answers "is the bf16 weight <name>
    // covered by something the plan binds?", so it maps the twin's name back to the bf16 name it
    // supersedes. It is NOT the fp8 checkpoint key rule: that key is `fp8/<name>` VERBATIM, prefix
    // included, and nothing strips it on the way in (see the `fp8/` key contract in lib.rs). The
    // distinction matters because a loader that copied this line is precisely how the two-spelling
    // bug happened — a freshly quantized checkpoint could not load at all.
    let want: HashSet<&str> = declared
        .iter()
        .map(|s| s.strip_prefix("fp8/").unwrap_or(s.as_str()))
        .filter(|n| n.starts_with(prefix))
        .collect();

    // Forward: only bf16 names are resolvable against `dir`. fp8 twins are checked by
    // the loader against the twin directory, not here.
    let mut missing: Vec<&str> = declared
        .iter()
        .map(|s| s.as_str())
        .filter(|n| n.starts_with(prefix) && !ckpt.contains(*n))
        // Produced before the bind — see `synthesized`'s contract above.
        .filter(|n| !synthesized.iter().any(|w| n.contains(w)))
        .collect();
    // `layer_scalar` is read by `layer_scalars()` as a compile-time immediate and
    // folded into the residual epilogue, so it is legitimately never declared.
    let mut uncovered: Vec<&str> = ckpt
        .iter()
        .map(|s| s.as_str())
        .filter(|n| !want.contains(*n) && !n.ends_with(".layer_scalar"))
        // Consumed indirectly — see `indirect`'s contract above.
        .filter(|n| !indirect.iter().any(|w| n.contains(w)))
        // Consumed only if its consumer was emitted — see `paired`'s contract above.
        .filter(|n| {
            !paired.iter().any(|(suf, produced)| {
                n.strip_suffix(suf)
                    .is_some_and(|stem| want.contains(format!("{stem}{produced}").as_str()))
            })
        })
        // In block mode only the block's own layers are in scope.
        .filter(|n| match &block {
            None => true,
            Some(r) => layer_of(n).is_some_and(|l| r.contains(&l)),
        })
        .collect();
    if missing.is_empty() && uncovered.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    uncovered.sort_unstable();

    let sample = |v: &[&str]| v.iter().take(8).copied().collect::<Vec<_>>().join("\n    ");
    let mut e = String::from("checkpoint coverage failed:\n");
    if !missing.is_empty() {
        e.push_str(&format!(
            "  plan weights NOT in the checkpoint ({}):\n    {}\n",
            missing.len(),
            sample(&missing)
        ));
    }
    if !uncovered.is_empty() {
        e.push_str(&format!(
            "  checkpoint tensors NOT covered by the plan ({}):\n    {}\n",
            uncovered.len(),
            sample(&uncovered)
        ));
        e.push_str(
            "  -> this checkpoint uses weights this emitter does not implement.\n     \
             Compiling anyway would DROP them and emit a silently-wrong model.\n",
        );
    }
    Err(e)
}

#[cfg(test)]
mod paired_tests {
    use super::*;

    const K3_PAIRED: &[(&str, &str)] = &[
        ("_res_norm.weight", "_res_score.weight"),
        ("_res_proj.weight", "_res_score.weight"),
    ];

    /// The `paired` rule reproduced on the two names that mattered.
    ///
    /// This is the regression test for the Kimi-K3 bring-up bug: the model-level
    /// `_apply_output_attn_res` was never emitted, so `output_attn_res_{norm,proj}` sat in the
    /// checkpoint claimed by nothing and the model decoded one constant token forever. A FLAT
    /// substring waiver for `_res_proj.weight` — the obvious way to write this — covers the
    /// model-level pair as collateral and would have kept the bug invisible. Keyed on the
    /// PRODUCED name it does not.
    fn covered(ckpt: &str, declared: &[&str]) -> bool {
        !K3_PAIRED.iter().any(|(suf, produced)| {
            ckpt.strip_suffix(suf)
                .is_some_and(|stem| declared.contains(&format!("{stem}{produced}").as_str()))
        })
    }

    #[test]
    fn the_output_mix_weights_are_covered_only_when_its_op_is_emitted() {
        let norm = "language_model.model.output_attn_res_norm.weight";
        let proj = "language_model.model.output_attn_res_proj.weight";
        let emitted = ["language_model.model.output_attn_res_score.weight"];

        // Op emitted -> its two factors are legitimately undeclared.
        assert!(
            !covered(norm, &emitted),
            "norm should be waived when the mix is emitted"
        );
        assert!(
            !covered(proj, &emitted),
            "proj should be waived when the mix is emitted"
        );

        // Op DROPPED -> both go unclaimed and the gate must fail the emit. Note the per-layer
        // score weights are still declared, which is exactly the state the real bug was in.
        let per_layer = ["language_model.model.layers.0.self_attention_res_score.weight"];
        assert!(
            covered(norm, &per_layer),
            "a dropped output mix must un-cover its norm"
        );
        assert!(
            covered(proj, &per_layer),
            "a dropped output mix must un-cover its proj"
        );
    }

    /// The per-layer pairs keep working, and one layer's score does not cover another's.
    #[test]
    fn a_paired_waiver_does_not_leak_across_stems() {
        let l0 = "language_model.model.layers.0.mlp_res_norm.weight";
        assert!(!covered(
            l0,
            &["language_model.model.layers.0.mlp_res_score.weight"]
        ));
        // A DIFFERENT layer's score must not cover layer 0's weight.
        assert!(covered(
            l0,
            &["language_model.model.layers.1.mlp_res_score.weight"]
        ));
        // Nor must a different stem on the same layer.
        assert!(covered(
            l0,
            &["language_model.model.layers.0.self_attention_res_score.weight"]
        ));
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    fn names(v: &[&str]) -> HashSet<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// The weight prefix is DERIVED from the checkpoint's own names, so a checkpoint that nests
    /// its tower under a wrapper is describable without a per-model patch. Spellings are verbatim
    /// from the three checkpoints on this box (Kimi-K3's from its
    /// `model.safetensors.index.json`: 497 052 `language_model.…` entries, ZERO `model.…`).
    #[test]
    fn the_layer_namespace_is_derived_from_the_tensor_names() {
        let glm = names(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.77.mlp.gate.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]);
        assert_eq!(
            layer_namespaces(&glm).keys().collect::<Vec<_>>(),
            vec!["model.layers."]
        );

        let k3 = names(&[
            "language_model.model.embed_tokens.weight",
            "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            "language_model.lm_head.weight",
            "vision_tower.encoder.blocks.0.wqkv.weight",
            "mm_projector.proj.0.weight",
        ]);
        let k3ns = layer_namespaces(&k3);
        assert_eq!(
            k3ns.keys().collect::<Vec<_>>(),
            vec!["language_model.model.layers."]
        );
        // The point of the derivation: `model.` is NOT among them, and a plan declared under it
        // would have matched nothing — which is what made the coverage gate below a no-op.
        assert!(!k3ns.contains_key("model.layers."));

        // A multimodal checkpoint legitimately has MORE THAN ONE, which is why the caller checks
        // membership rather than taking a unique answer.
        let gemma = names(&[
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        ]);
        assert_eq!(layer_namespaces(&gemma).len(), 2);
    }

    /// Counts come out with the namespaces, because the refusal quotes them: "`model.` matches
    /// nothing; this checkpoint has `language_model.model.layers.*` (497 051 tensors)" is
    /// actionable and "prefix mismatch" is not.
    #[test]
    fn the_namespaces_carry_their_tensor_counts() {
        let n = names(&[
            "a.layers.0.w",
            "a.layers.1.w",
            "b.layers.0.w",
            "no_layers_here.weight",
        ]);
        let ns = layer_namespaces(&n);
        assert_eq!(ns.get("a.layers."), Some(&2));
        assert_eq!(ns.get("b.layers."), Some(&1));
        assert_eq!(
            ns.len(),
            2,
            "a name without `.layers.` contributes no namespace"
        );
    }
}
