//! `plowrt drafter-extract`: lift a packet's embedded chain drafter (`spec.json` +
//! `spec.draft.programs`) into an external drafter directory (`plow_asset::spec_drafter`) that
//! attaches to the same packet by its digest.
//!
//! A tensor the draft programs use becomes an import when any target program also uses it (or it
//! is a KV cache); the rest are the drafter's own: checkpoint weights (their shards are linked
//! into the directory), init bytes (copied), or scratch. Handles are renumbered into the
//! drafter's table, input token 0 first, so a remapped `0` never reads as an absent addend.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16, TENSOR_NONE_I};
use plow_asset::aux_program::{self, IntHandle};
use plow_asset::spec_drafter::{
    self as sd, Access, Kind, Requires, Source, Tensor, TokenizerFacts, TokenizerReq,
};
use sha2::Digest;

use crate::asset::devblob::DevBlob;
use crate::{Result, RuntimeError};

fn reject(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("drafter-extract: {e}"))
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> RuntimeError + '_ {
    move |source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Every tensor handle `inst` names, in `t` and in integer fields.
fn handles(inst: &DevInst64) -> impl Iterator<Item = u32> + '_ {
    let op = DevOp::from_u16(inst.op);
    let ints = op
        .map(|op| aux_program::integer_tensor_handles(op, inst))
        .unwrap_or_default()
        .into_iter()
        .filter_map(|h| match h {
            IntHandle::Required(s) => Some(inst.i[s]),
            IntHandle::Optional(s) => (inst.i[s] != TENSOR_NONE_I).then_some(inst.i[s]),
            IntHandle::ZeroNone(s) => (inst.i[s] != 0).then_some(inst.i[s]),
            IntHandle::Fj2 => Some(inst.fj[2]),
            IntHandle::Fj2ZeroNone => (inst.fj[2] != 0).then_some(inst.fj[2]),
        });
    inst.t
        .iter()
        .filter(|&&t| t != TENSOR_NONE16)
        .map(|&t| u32::from(t))
        .chain(ints)
}

fn remap(inst: &mut DevInst64, map: &BTreeMap<u32, u32>) -> Result<()> {
    let to = |h: u32| {
        map.get(&h)
            .copied()
            .ok_or_else(|| reject(format!("handle {h} unmapped")))
    };
    for t in inst.t.iter_mut().filter(|t| **t != TENSOR_NONE16) {
        *t = to(u32::from(*t))? as u16;
    }
    let op = DevOp::from_u16(inst.op).ok_or_else(|| reject("unknown opcode"))?;
    for h in aux_program::integer_tensor_handles(op, inst) {
        match h {
            IntHandle::Required(s) => inst.i[s] = to(inst.i[s])?,
            IntHandle::Optional(s) if inst.i[s] != TENSOR_NONE_I => inst.i[s] = to(inst.i[s])?,
            IntHandle::ZeroNone(s) if inst.i[s] != 0 => {
                inst.i[s] = to(inst.i[s])?;
                if inst.i[s] == 0 {
                    return Err(reject("an addend remapped to handle 0"));
                }
            }
            IntHandle::Fj2 => inst.fj[2] = to(inst.fj[2])?,
            IntHandle::Fj2ZeroNone if inst.fj[2] != 0 => {
                inst.fj[2] = to(inst.fj[2])?;
                if inst.fj[2] == 0 {
                    return Err(reject("a row-slot map remapped to handle 0"));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Shard file (in `checkpoint`) holding each tensor name.
fn shard_index(checkpoint: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(checkpoint).map_err(io(checkpoint))? {
        let path = e.map_err(io(checkpoint))?.path();
        if path.extension().is_none_or(|x| x != "safetensors") {
            continue;
        }
        let mut f = std::fs::File::open(&path).map_err(io(&path))?;
        let mut len = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut len).map_err(io(&path))?;
        let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
        std::io::Read::read_exact(&mut f, &mut header).map_err(io(&path))?;
        let v: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| reject(format!("{}: {e}", path.display())))?;
        for name in v.as_object().into_iter().flatten().map(|(k, _)| k) {
            if name != "__metadata__" {
                out.insert(name.clone(), path.clone());
            }
        }
    }
    Ok(out)
}

/// A target activation (`act.*`, no init, no generated table, not a weight): every launch
/// rewrites it before reading, so a draft may use it as stream-ordered scratch. Caches,
/// recurrent state, pointer tables and inputs never qualify.
pub fn activation(blob: &DevBlob, h: usize) -> bool {
    let d = &blob.tensors[h];
    d.name.starts_with("act.")
        && d.init.is_none()
        && !blob.gen.iter().any(|g| g.tensor as usize == h)
        && !packet::names::is_checkpoint_weight(&d.name)
}

/// A drafter's pinned weight shards, memory-mapped; every tensor name bound once.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", feature = "hub"))]
pub struct Shards {
    maps: Vec<memmap2::Mmap>,
    index: BTreeMap<String, (usize, std::ops::Range<usize>)>,
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", feature = "hub"))]
impl Shards {
    pub fn open(dir: &Path, files: &[String]) -> std::result::Result<Self, String> {
        let (mut maps, mut index) = (Vec::new(), BTreeMap::new());
        for (k, name) in files.iter().enumerate() {
            let path = dir.join(name);
            let file =
                std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            // SAFETY: read-only mapping of a pinned shard, held as long as the drafter.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let (hlen, meta) = safetensors::SafeTensors::read_metadata(&map)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            for (tensor, info) in meta.tensors() {
                let (a, b) = info.data_offsets;
                let range = 8 + hlen + a..8 + hlen + b;
                if index.insert(tensor.clone(), (k, range)).is_some() {
                    return Err(format!("tensor {tensor} is in more than one shard"));
                }
            }
            maps.push(map);
        }
        Ok(Self { maps, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        let (k, r) = self.index.get(name)?;
        self.maps[*k].get(r.clone())
    }
}

pub struct Extracted {
    pub imports: usize,
    pub weights: usize,
    pub scratch: usize,
    pub shards: Vec<String>,
}

/// Write the drafter directory `out` for the packet in `assets` (weights from `checkpoint`).
/// `pin`: the target packet the drafter attaches to, when it is not `assets` itself (the same
/// model emitted without the embedded drafter): every import must exist there with its bytes.
pub fn extract(
    assets: &Path,
    checkpoint: &Path,
    out: &Path,
    pin: Option<&Path>,
) -> Result<Extracted> {
    let pkt = DevBlob::find_in_dir(assets)?
        .ok_or_else(|| reject(format!("no packet in {}", assets.display())))?;
    let raw = std::fs::read(&pkt).map_err(io(&pkt))?;
    let blob = DevBlob::parse(&raw)?;
    let meta = blob
        .reserved_metadata(&raw, plow_asset::spec_draft::SECTION)?
        .ok_or_else(|| reject("packet carries no embedded drafter (spec.json)"))?;
    let mut draft: plow_asset::spec_draft::Manifest =
        serde_json::from_slice(meta).map_err(reject)?;
    if draft.block.is_some() {
        return Err(reject(
            "a block drafter's context is written by the target programs; it cannot be \
             detached from this packet",
        ));
    }
    let bytes = blob
        .reserved_metadata(&raw, plow_asset::spec_draft::PROGRAMS_SECTION)?
        .ok_or_else(|| reject("draft programs section missing"))?;
    let n = blob.tensors.len();
    let mut section = aux_program::parse(bytes, blob.n_cu, n).map_err(reject)?;
    for inst in section.programs.iter().flat_map(|p| &p.insts) {
        let op = DevOp::from_u16(inst.op).ok_or_else(|| reject("unknown opcode"))?;
        if sd::vetted_writes(op, inst).is_none() {
            return Err(reject(format!(
                "opcode {op:?} is not allowed in an external drafter"
            )));
        }
    }
    std::fs::create_dir_all(out).map_err(io(out))?;
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    if canon(out) == canon(checkpoint) || canon(out) == canon(assets) {
        return Err(reject(
            "--out must be a directory of its own, not the packet or checkpoint",
        ));
    }

    // Tensors any target program uses: the program table and every other aux-program section.
    let mut target: BTreeSet<u32> = blob
        .progs
        .iter()
        .flat_map(|p| p.insts.iter().flat_map(handles))
        .collect();
    for s in &blob.sections {
        if s.name == plow_asset::spec_draft::PROGRAMS_SECTION {
            continue;
        }
        let data = &raw[s.offset..s.offset + s.size];
        if data.starts_with(aux_program::MAGIC) {
            let other = aux_program::parse(data, blob.n_cu, n).map_err(reject)?;
            target.extend(
                other
                    .programs
                    .iter()
                    .flat_map(|p| p.insts.iter().flat_map(handles)),
            );
        }
    }

    let t = &draft.tensors;
    let first = *t
        .tokens
        .first()
        .ok_or_else(|| reject("spec.json has no token tensors"))?;
    let mut used: Vec<u32> = vec![u32::from(first)];
    let named = t.tokens[1..]
        .iter()
        .chain([&t.position, &t.kv_len, &t.slot])
        .chain(t.hidden_row.iter())
        .chain(t.hidden.iter())
        .map(|&h| u32::from(h));
    let mut rest: BTreeSet<u32> = named.collect();
    rest.extend(
        section
            .programs
            .iter()
            .flat_map(|p| p.insts.iter().flat_map(handles)),
    );
    rest.remove(&used[0]);
    used.extend(rest);
    let map: BTreeMap<u32, u32> = used
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new as u32))
        .collect();

    let shards = shard_index(checkpoint)?;
    let gen: BTreeMap<u32, &packet::rope::GenTensor> =
        blob.gen.iter().map(|g| (g.tensor, g)).collect();
    let (mut tensors, mut init, mut linked) = (Vec::new(), Vec::new(), BTreeSet::new());
    let (mut imports, mut weights, mut scratch) = (0, 0, 0);
    for &h in &used {
        let d = &blob.tensors[h as usize];
        // Only what the target binds is an import: a draft weight the target never loads is
        // the drafter's own, even when it sits in one of the target's shards.
        let source = if target.contains(&h) || d.name.starts_with("kv.") {
            imports += 1;
            // Target activations are rewritten by every launch; the draft may use them as
            // scratch (stream-ordered). Caches, weights and tables stay read-only.
            let access = if activation(&blob, h as usize) {
                Access::ReadWrite
            } else {
                Access::Read
            };
            Source::Import { access }
        } else if let Some(r) = &d.init {
            let offset = init.len() as u64;
            init.extend_from_slice(&blob.init[r.clone()]);
            Source::Init {
                offset,
                len: d.bytes,
            }
        } else if let Some(g) = gen.get(&h) {
            let data = g
                .generate()
                .ok_or_else(|| reject(format!("{}: unknown gen recipe", d.name)))?;
            let offset = init.len() as u64;
            init.extend_from_slice(&data);
            Source::Init {
                offset,
                len: data.len() as u64,
            }
        } else if packet::names::is_checkpoint_weight(&d.name) {
            let shard = shards.get(&d.name).ok_or_else(|| {
                reject(format!(
                    "weight {} is in no shard of {}",
                    d.name,
                    checkpoint.display()
                ))
            })?;
            linked.insert(shard.clone());
            weights += 1;
            Source::Weight
        } else {
            scratch += 1;
            Source::Scratch
        };
        tensors.push(Tensor {
            name: d.name.clone(),
            bytes: d.bytes,
            source,
        });
    }

    for p in &mut section.programs {
        for inst in &mut p.insts {
            remap(inst, &map)?;
        }
    }
    let to16 = |h: u16| map[&u32::from(h)] as u16;
    let dt = &mut draft.tensors;
    dt.tokens = dt.tokens.iter().map(|&h| to16(h)).collect();
    dt.position = to16(dt.position);
    dt.kv_len = to16(dt.kv_len);
    dt.slot = to16(dt.slot);
    dt.hidden_row = dt.hidden_row.map(to16);
    dt.hidden = dt.hidden.map(to16);
    let programs = section.encode(used.len()).map_err(reject)?;

    let mut pins = Vec::new();
    let mut shard_names = Vec::new();
    for src in &linked {
        let name = src.file_name().expect("shard file").to_owned();
        let dst = out.join(&name);
        let real = std::fs::canonicalize(src).map_err(io(src))?;
        if canon(&dst) == real {
            return Err(reject(format!("{} is its own source shard", dst.display())));
        }
        let _ = std::fs::remove_file(&dst);
        if std::fs::hard_link(&real, &dst).is_err() {
            std::fs::copy(&real, &dst).map_err(io(&dst))?;
        }
        pins.push(plow_asset::serve_manifest::pin_shard(&dst).map_err(reject)?);
        shard_names.push(name.to_string_lossy().into_owned());
    }

    let tokenizer_facts = |candidates: [PathBuf; 2]| -> Result<TokenizerFacts> {
        let p = candidates
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| reject("no tokenizer.json beside the packet or in its checkpoint"))?;
        TokenizerFacts::from_json(&std::fs::read(&p).map_err(io(&p))?).map_err(reject)
    };
    // The drafter's own vocabulary; a pinned target must carry it unchanged.
    let facts = tokenizer_facts([
        assets.join("tokenizer.json"),
        checkpoint.join("tokenizer.json"),
    ])?;
    if let Some(dir) = pin {
        let t = tokenizer_facts([
            dir.join("tokenizer.json"),
            dir.join("checkpoint").join("tokenizer.json"),
        ])?;
        if t.vocab_sha256 != facts.vocab_sha256
            || facts
                .added_tokens
                .iter()
                .any(|(id, c)| t.added_tokens.get(id) != Some(c))
        {
            return Err(reject(format!(
                "the target in {} has a different tokenizer than the drafter's source",
                dir.display()
            )));
        }
    }
    let target_sha = match pin {
        None => format!("{:x}", sha2::Sha256::digest(&raw)),
        Some(dir) => {
            let tp = DevBlob::find_in_dir(dir)?
                .ok_or_else(|| reject(format!("no packet in {}", dir.display())))?;
            let traw = std::fs::read(&tp).map_err(io(&tp))?;
            let tblob = DevBlob::parse(&traw)?;
            let have: BTreeMap<&str, u64> = tblob
                .tensors
                .iter()
                .map(|t| (t.name.as_str(), t.bytes))
                .collect();
            for t in tensors
                .iter()
                .filter(|t| matches!(t.source, Source::Import { .. }))
            {
                match have.get(t.name.as_str()) {
                    Some(&b) if b == t.bytes => {}
                    Some(&b) => {
                        return Err(reject(format!(
                            "import {}: {} bytes here, {b} in the pinned target",
                            t.name, t.bytes
                        )))
                    }
                    None => {
                        return Err(reject(format!(
                            "import {} is not in the pinned target",
                            t.name
                        )))
                    }
                }
            }
            format!("{:x}", sha2::Sha256::digest(&traw))
        }
    };
    let manifest = sd::Manifest {
        contract: sd::CONTRACT.into(),
        kind: Kind::ChainSharedKv,
        family: draft.family.clone(),
        n_cu: blob.n_cu,
        requires: Requires {
            target_packet_sha256: Some(target_sha),
            tokenizer: Some(TokenizerReq {
                vocab_sha256: facts.vocab_sha256,
                added_tokens: facts.added_tokens.into_iter().collect(),
            }),
            object_symbols: [("plow_row_slot_1".to_string(), 1)].into(),
            ..Default::default()
        },
        tensors,
        draft,
        weights: pins,
    };
    manifest.validate().map_err(reject)?;
    let write = |name: &str, data: &[u8]| {
        let path = out.join(name);
        std::fs::write(&path, data).map_err(io(&path))
    };
    write(sd::PROGRAMS_FILE, &programs)?;
    if !init.is_empty() {
        write(sd::INIT_FILE, &init)?;
    }
    write(
        sd::FILE,
        &serde_json::to_vec_pretty(&manifest).map_err(reject)?,
    )?;
    Ok(Extracted {
        imports,
        weights,
        scratch,
        shards: shard_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: DevOp) -> DevInst64 {
        DevInst64 {
            op: op as u16,
            blocks: 1,
            fj: [0; 3],
            t: [TENSOR_NONE16; 8],
            i: [0; 8],
        }
    }

    #[test]
    fn remap_covers_tensor_and_integer_handles() {
        // Folded-prologue GEMV: i[5], i[7] handles, i[6] an optional addend (0 = none).
        let mut g = inst(DevOp::Gemv);
        g.t[0] = 40;
        g.t[1] = 41;
        g.i[3] = 4;
        g.i[5] = 42;
        g.i[7] = 43;
        let map: BTreeMap<u32, u32> = [(40, 1), (41, 2), (42, 3), (43, 4), (44, 5)].into();
        assert_eq!(handles(&g).collect::<Vec<_>>(), [40, 41, 42, 43]);
        let mut r = g;
        remap(&mut r, &map).unwrap();
        assert_eq!((r.t[0], r.t[1], r.i[5], r.i[6], r.i[7]), (1, 2, 3, 0, 4));
        g.i[6] = 44;
        let mut r = g;
        remap(&mut r, &map).unwrap();
        assert_eq!(r.i[6], 5);
        // An addend that would land on handle 0 reads as "none": refused.
        let mut r = g;
        assert!(remap(
            &mut r,
            &[(40, 1), (41, 2), (42, 3), (43, 4), (44, 0)].into()
        )
        .is_err());
        // An unmapped handle is an error, never a silent pass-through.
        let mut r = g;
        assert!(remap(&mut r, &[(40, 1)].into()).is_err());
    }
}
