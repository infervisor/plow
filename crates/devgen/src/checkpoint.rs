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

/// Every `prefix*` tensor name the checkpoint actually ships.
fn ckpt_names(dir: &Path, prefix: &str) -> HashSet<String> {
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
            if k != "__metadata__" && k.starts_with(prefix) {
                out.insert(k.clone());
            }
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
    let rest = name.split_once(".layers.")?.1;
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
) -> Result<(), String> {
    // A directory with NO safetensors at all is not a coverage failure -- there is nothing to
    // cross-check against. Emitting a blob from a bare config.json is legitimate (the structural
    // golden tests do exactly that), and `shard_files` PANICS rather than returning empty, so
    // reaching it here aborts a valid emit. Say so on stderr rather than skipping silently: a
    // genuinely missing checkpoint is still visible, it just no longer kills the process from
    // inside a gate whose whole purpose is comparing two sets, one of which does not exist.
    let has_shards = std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(Result::ok).any(|e| {
            e.path().extension().and_then(|x| x.to_str()) == Some("safetensors")
        })
    });
    if !has_shards {
        eprintln!(
            "devgen: {} has no .safetensors — skipping the checkpoint coverage gate \
             (nothing to compare the plan against)",
            dir.display()
        );
        return Ok(());
    }
    let ckpt = ckpt_names(dir, prefix);
    // A weight is covered if the plan binds EITHER the bf16 tensor or its fp8 twin.
    // Under PLOW_FP8 the projections are declared as `fp8/<name>` (the twins live in a
    // sibling dir, so they are not in `ckpt`) and the bf16 original is deliberately
    // superseded — counting it "uncovered" would fail every fp8 build.
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
        .collect();
    // `layer_scalar` is read by `layer_scalars()` as a compile-time immediate and
    // folded into the residual epilogue, so it is legitimately never declared.
    let mut uncovered: Vec<&str> = ckpt
        .iter()
        .map(|s| s.as_str())
        .filter(|n| !want.contains(*n) && !n.ends_with(".layer_scalar"))
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

    let sample = |v: &[&str]| {
        v.iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>()
            .join("\n    ")
    };
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
