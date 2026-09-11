//! Reader for the compiler→runtime device container (`PLOWDEV\x05`).
//!
//! `plowc` (e.g. the `gemma4` bin) writes this blob via
//! `packet::devbuild::Model::to_blob`; the C harnesses parse it through the
//! shared structs in `runtime/common/dev_blob.h`. This is the Rust runtime's
//! reader — every record goes through the `#[repr(C)]` mirrors in
//! `packet::devbuild` / `packet::dev`, whose sizes and offsets are locked
//! against that header by `crates/packet/tests/dev_abi.rs`. No offsets are
//! hand-rolled here (the format broke twice in one afternoon when they were).

use std::path::{Path, PathBuf};

use packet::dev::{DevInst64, DevOp, StreamEnt, Wait, SE_DOMAIN_MASK};
use packet::devbuild::{
    is_blob_magic, BlobHeader, BlobProgHeader, BlobSectionEntry, BlobTensor, BLOB_MAGIC_V7,
    BLOB_MAGIC_V7_L2SEG, INIT_NONE, NAME_LEN, SECT_GEN_TENSORS, SECT_MAGIC, SECT_NAME_LEN,
};
use packet::rope::GenTensor;

use crate::{Result, RuntimeError};

/// One tensor the programs address by handle: its name, byte size, and (for
/// compiler-computed data such as the RoPE tables) its slice of the init
/// section.
pub struct DevTensor {
    pub name: String,
    pub bytes: u64,
    /// Byte range into [`DevBlob::init`], or `None` (runtime-filled).
    pub init: Option<std::ops::Range<usize>>,
}

/// One compiled program (a prefill bucket, or the T=1 decode program last).
pub struct DevProg {
    /// The T this program was compiled for (decode = 1).
    pub t: u32,
    /// This topology is selected only for a genuinely packed prefill dispatch.
    pub packed_prefill_only: bool,
    /// A token-batch BODY (`packet::devbuild::TOKEN_BATCH_PROG`): prefill width `t` with a
    /// slot-indexed decode band; selected only by the token-batch route, never as a rung.
    pub token_batch_body: bool,
    /// An EXPLICIT decode rung (`packet::devbuild::DECODE_RUNG_PROG`). Always `false` in a
    /// parent packet, where the ladder is positional; an extension states it, because a lone
    /// program has no position to read the role out of.
    pub decode_rung: bool,
    pub n_counter: u32,
    pub insts: Vec<DevInst64>,
    pub stream: Vec<StreamEnt>,
    pub stream_ofs: Vec<u32>,
    pub stream_len: Vec<u32>,
    pub waits: Vec<Wait>,
    pub succs: Vec<u32>,
    /// Op-major (topological) permutation of `stream` from the blob's `GQ01`
    /// appendix — the global-queue interpreter's work list. Empty when the
    /// blob predates the appendix.
    pub gq_stream: Vec<StreamEnt>,
    /// Queue-window bounds into `gq_stream`. Current L2-placed blobs use
    /// `[ordered_segment][physical_domain]` windows.
    pub gq_seg_ofs: Vec<u32>,
    /// L2-domain placement (`PLOW_L2_PLACE`): domains per ordered segment, or `0`.
    ///
    /// The header's domain count is shared, but placement is per program. Current placed
    /// programs have one queue window per ordered segment and domain; unplaced programs
    /// have one per ordered segment. Legacy placed programs have one per domain.
    pub l2_domains: u32,
}

/// How a blob is sharded across GPUs, RECOVERED from the program rather than
/// read from a header field — because there is no header field.
///
/// `plowc --num-gpus N` bakes the sharding into every collective it emits
/// (`crates/devgen` `emit_xreduce`): `i[0]` = elements reduced, `i[1]` = the TP
/// degree, `i[2]` = the partial slot's byte offset. So a `--tp 4` blob is
/// self-describing and a `--tp 1` blob carries no collective at all, which is
/// exactly the distinction a loader must make BEFORE it binds 60 GiB of
/// weights.
///
/// Without this the failure is late and unreadable: a tp=4 blob declares every
/// projection at 1/4 size, so a single-GPU loader that binds full tensors dies
/// at the first `q_proj` with `SIZE MISMATCH ... blob says 5.5 MB, checkpoint
/// has 22 MB` — a message that describes the symptom and names neither TP nor
/// the flag that would fix it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevTp {
    /// TP degree the blob was compiled for (`XReduce.i[1]`). Always `> 1`:
    /// a tp=1 blob emits no collective, so it has no [`DevTp`] at all.
    pub n_gpu: u32,
    /// Model hidden size (`collective.i[0] / program.t`). Sizes the all-reduce
    /// message and hence the peer region.
    pub hidden: u32,
    /// Byte offset of partial slot B within the peer region — `max(i[2])` over
    /// the program's collectives, since slot A carries `i[2] == 0`.
    ///
    /// `devgen` computes it as `rows_max·hidden·2` where `rows_max` is the
    /// LARGEST prefill chunk, and bakes that same value into every bucket AND
    /// the decode program.
    pub slot_bytes: u64,
}

/// A section embedded in a v6 blob (cubin, hsaco, weight map, etc.).
pub struct DevSection {
    pub kind: u32,
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// A parsed device blob.
pub struct DevBlob {
    pub n_cu: u32,
    pub flags: u32,
    /// Target-GPU fingerprint the blob was compiled for (`gpu_fingerprint`; 0 =
    /// unknown). A backend that resolves its device to the same canonical spec
    /// name can warn on mismatch — the header stamp closes Gap 4 (only `n_cu`
    /// was checked before). Model arch tag + HF id live in the SECT_METADATA
    /// `block.json` descriptor.
    pub target: u32,
    pub tensors: Vec<DevTensor>,
    pub init: Vec<u8>,
    /// Instruction indices in the decode program whose `i[3]` is the KV-cache
    /// write row — the entire dynamic surface of a decode step.
    pub kvrow: Vec<u32>,
    pub progs: Vec<DevProg>,
    /// v6 section directory entries (empty on v5 blobs).
    pub sections: Vec<DevSection>,
    /// Tensors the runtime materialises at bind time rather than uploading from
    /// [`Self::init`] — the RoPE tables on a v7 blob. Empty on v5/v6, where the
    /// same bytes arrive via [`DevTensor::init`].
    pub gen: Vec<GenTensor>,
    /// TP sharding recovered from the decode program's collectives, or `None`
    /// for a single-GPU blob. See [`DevTp`].
    pub tp: Option<DevTp>,
    /// The parent this container references INSTEAD of declaring a tensor table — `Some` iff
    /// this is an `extension.pkt` (docs/arch/19, phase 2). A parent packet is always `None`,
    /// and the two are told apart by container magic before anything else is read.
    pub parent: Option<plow_asset::extension::ParentRef>,
}

/// Which container [`DevBlob::parse_inner`] was asked for. The two are distinct files with
/// distinct magics, and reading one as the other is the silent failure the extension magic
/// exists to prevent — so the caller says which it wants and gets a named refusal otherwise.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    Model,
    Extension,
}

/// Copy `n` `T` records out of `buf` at `*off` (unaligned-safe — the blob's
/// sections are packed back to back with no padding between them).
fn take<T: Copy>(buf: &[u8], off: &mut usize, n: usize, what: &str) -> Result<Vec<T>> {
    let sz = std::mem::size_of::<T>();
    let need = n
        .checked_mul(sz)
        .ok_or_else(|| RuntimeError::Device(format!("devblob: {what} count overflows")))?;
    let end = off
        .checked_add(need)
        .filter(|&e| e <= buf.len())
        .ok_or_else(|| {
            RuntimeError::Device(format!(
                "devblob: truncated at {what} (need {need} B at offset {off}, have {})",
                buf.len()
            ))
        })?;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: bounds checked above; T is a #[repr(C)] POD mirror.
        v.push(unsafe { std::ptr::read_unaligned(buf[*off + i * sz..].as_ptr() as *const T) });
    }
    *off = end;
    Ok(v)
}

/// Recover the TP sharding by scanning every program's collectives.
///
/// EVERY program is scanned, not just decode: a sharded prefill bucket is just
/// as unloadable on one GPU as a sharded decode program, and a scan that missed
/// it would send the caller straight back to the `SIZE MISMATCH` this exists to
/// replace.
///
/// The three fields come from different places on purpose:
///
/// * `n_gpu` — `i[1]`, identical on every collective.
/// * `hidden` — `i[0] / program.t` for every collective. Both one-shot and
///   two-shot packets carry `t·width`; the maximum width is the model hidden
///   size when narrower latent collectives are also present.
/// * `slot_bytes` — `max(i[2])`, because slot A carries 0 and slot B carries the
///   offset. `devgen` derives it from the LARGEST prefill chunk and bakes the
///   same value into every program, so the max over all of them is that one
///   value and not a per-program quantity.
fn recover_tp(progs: &[DevProg]) -> Option<DevTp> {
    let (mut n_gpu, mut hidden, mut slot_bytes) = (0u32, 0u32, 0u64);
    for p in progs {
        for d in &p.insts {
            if d.op != DevOp::XReduce as u16 && d.op != DevOp::XReduceTwoShot as u16 {
                continue;
            }
            n_gpu = n_gpu.max(d.i[1]);
            hidden = hidden.max(d.i[0] / p.t.max(1));
            slot_bytes = slot_bytes.max(d.i[2] as u64);
        }
    }
    // A tp==1 blob emits no collective at all, so "no collective" and "not
    // sharded" are the same fact and `None` says it once.
    (n_gpu > 1).then_some(DevTp {
        n_gpu,
        hidden,
        slot_bytes,
    })
}

impl DevBlob {
    /// Parse a blob image. Fails loudly on a bad magic or a truncated section,
    /// never mid-serve.
    /// Parse a blob, refusing L2-domain placement unless the caller can honour it.
    ///
    /// `l2_dispatch_ok` says the CALLER will verify the code object actually carries the
    /// dispatch axis (AMD does, via the `plow_l2_place_dispatch_1` marker at object-load time).
    /// Backends that cannot check keep the old behaviour through [`DevBlob::parse`], which is
    /// this with `false` -- placement is then refused unless runtime configuration opts in.
    pub fn parse_l2(buf: &[u8], l2_dispatch_ok: bool) -> Result<DevBlob> {
        Self::parse_inner(buf, l2_dispatch_ok, Container::Model)
    }

    pub fn parse(buf: &[u8]) -> Result<DevBlob> {
        Self::parse_inner(buf, false, Container::Model)
    }

    /// Parse an `extension.pkt` (docs/arch/19, phase 2): the same container with the tensor
    /// table replaced by a reference to the parent's.
    ///
    /// This only decodes the container. Whether the extension may be MERGED is the six-rule
    /// contract in `plow_asset::extension`, applied by [`crate::asset::extension`].
    pub fn parse_extension(buf: &[u8], l2_dispatch_ok: bool) -> Result<DevBlob> {
        Self::parse_inner(buf, l2_dispatch_ok, Container::Extension)
    }

    fn parse_inner(buf: &[u8], l2_dispatch_ok: bool, want: Container) -> Result<DevBlob> {
        let mut off = 0usize;
        let hdr: BlobHeader = take::<BlobHeader>(buf, &mut off, 1, "header")?[0];
        let is_ext = packet::ext::is_ext_magic(&hdr.magic);
        match (want, is_ext, is_blob_magic(&hdr.magic)) {
            (Container::Model, false, true) | (Container::Extension, true, _) => {}
            (Container::Model, true, _) => {
                return Err(RuntimeError::Device(
                    "devblob: this is an extension.pkt, not a model packet — an extension \
                     carries programs and a reference to its parent's tensor table, and binding \
                     it as a model would leave every tensor handle unresolved. Load the parent \
                     and merge this as an extension."
                        .into(),
                ))
            }
            (Container::Extension, false, true) => {
                return Err(RuntimeError::Device(
                    "devblob: this is a model.pkt, not an extension — it declares its own \
                     tensor table and cannot be merged onto a parent."
                        .into(),
                ))
            }
            _ => {
                return Err(RuntimeError::Device(
                    "devblob: bad magic — recompile with plowc (format changed)".into(),
                ))
            }
        }
        let is_v7 = &hdr.magic == BLOB_MAGIC_V7 || &hdr.magic == BLOB_MAGIC_V7_L2SEG;

        let decls = take::<BlobTensor>(buf, &mut off, hdr.n_tensor as usize, "tensor decls")?;
        let init = take::<u8>(buf, &mut off, hdr.init_bytes as usize, "init section")?;
        let tensors = decls
            .iter()
            .map(|d| {
                let len = d.name.iter().position(|&b| b == 0).unwrap_or(NAME_LEN);
                let name = String::from_utf8_lossy(&d.name[..len]).into_owned();
                let init_range = if d.init_off == INIT_NONE {
                    None
                } else {
                    let s = d.init_off as usize;
                    let e = s
                        .checked_add(d.bytes as usize)
                        .filter(|&e| e <= init.len())
                        .ok_or_else(|| {
                            RuntimeError::Device(format!(
                                "devblob: tensor {name} init range out of bounds"
                            ))
                        })?;
                    Some(s..e)
                };
                Ok(DevTensor {
                    name,
                    bytes: d.bytes,
                    init: init_range,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let kvrow = take::<u32>(buf, &mut off, hdr.n_kvrow as usize, "kvrow table")?;

        let mut progs = Vec::with_capacity(hdr.n_prog as usize);
        for p in 0..hdr.n_prog {
            let ph: BlobProgHeader = take::<BlobProgHeader>(buf, &mut off, 1, "prog header")?[0];
            let what = |s: &str| format!("prog {p} {s}");
            progs.push(DevProg {
                t: packet::devbuild::program_rows(ph.t),
                packed_prefill_only: packet::devbuild::is_packed_prefill_program(ph.t),
                token_batch_body: packet::devbuild::is_token_batch_program(ph.t),
                decode_rung: packet::devbuild::is_decode_rung_program(ph.t),
                n_counter: ph.n_counter,
                insts: take(buf, &mut off, ph.n_inst as usize, &what("insts"))?,
                stream: take(buf, &mut off, ph.n_stream as usize, &what("stream"))?,
                stream_ofs: take(buf, &mut off, hdr.n_cu as usize, &what("stream_ofs"))?,
                stream_len: take(buf, &mut off, hdr.n_cu as usize, &what("stream_len"))?,
                waits: take(buf, &mut off, ph.n_wait as usize, &what("waits"))?,
                succs: take(buf, &mut off, ph.n_succ as usize, &what("succs"))?,
                gq_stream: Vec::new(),
                gq_seg_ofs: Vec::new(),
                l2_domains: 0,
            });
        }

        // Optional GQ01 appendix: per program { n_seg, gq_stream[n_stream],
        // gq_seg_ofs[n_seg+1] }. Loaders that stop after the programs never
        // see it, so its absence is not an error.
        if buf.len() >= off + 4 && &buf[off..off + 4] == b"GQ01" {
            off += 4;
            for p in 0..hdr.n_prog as usize {
                let n_seg = take::<u32>(buf, &mut off, 1, "gq n_seg")?[0] as usize;
                let n_stream = progs[p].stream.len();
                progs[p].gq_stream = take(buf, &mut off, n_stream, "gq_stream")?;
                progs[p].gq_seg_ofs = take(buf, &mut off, n_seg + 1, "gq_seg_ofs")?;
                let l2_dom = hdr.reserved[2] as u32;
                let combined = hdr.flags & packet::devbuild::PLOW_BLOB_F_L2SEG != 0;
                let ordered_segments = progs[p]
                    .gq_stream
                    .iter()
                    .map(|entry| entry.seg as usize + 1)
                    .max()
                    .unwrap_or(0);
                if hdr.flags & packet::devbuild::PLOW_BLOB_F_L2DOM != 0
                    && l2_dom != 0
                    && ((!combined && n_seg == l2_dom as usize)
                        || (combined
                            && ordered_segments.checked_mul(l2_dom as usize) == Some(n_seg)))
                {
                    progs[p].l2_domains = l2_dom;
                }
            }
        }

        // v6 section directory: if reserved[0] (sect_dir_offset) is non-zero,
        // parse the directory. Section DATA stays in the original buffer — we
        // only store the metadata here; callers use `section_data()` to slice it.
        let sections = if hdr.reserved[0] != 0 {
            let dir_off = hdr.reserved[0] as usize;
            if dir_off + 8 > buf.len() {
                return Err(RuntimeError::Device(
                    "devblob: sect_dir_offset past end of buffer".into(),
                ));
            }
            if &buf[dir_off..dir_off + 4] != SECT_MAGIC {
                return Err(RuntimeError::Device(
                    "devblob: bad section directory magic".into(),
                ));
            }
            let n = u32::from_le_bytes(buf[dir_off + 4..dir_off + 8].try_into().unwrap()) as usize;
            let ent_start = dir_off + 8;
            let ent_size = std::mem::size_of::<BlobSectionEntry>();
            if ent_start + n * ent_size > buf.len() {
                return Err(RuntimeError::Device(
                    "devblob: section directory truncated".into(),
                ));
            }
            let mut sects = Vec::with_capacity(n);
            for i in 0..n {
                let base = ent_start + i * ent_size;
                // SAFETY: `base + ent_size <= buf.len()` is enforced by the
                // truncation check above, and `read_unaligned` is what makes a
                // file-offset-derived pointer legal — a `BlobSectionEntry` in
                // the directory is only byte-aligned. The type is `#[repr(C)]`
                // POD, so every bit pattern is a valid value.
                let ent = unsafe {
                    std::ptr::read_unaligned(buf[base..].as_ptr() as *const BlobSectionEntry)
                };
                let name_len = ent
                    .name
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(SECT_NAME_LEN);
                let name = String::from_utf8_lossy(&ent.name[..name_len]).into_owned();
                sects.push(DevSection {
                    kind: ent.kind,
                    name,
                    offset: ent.offset as usize,
                    size: ent.size as usize,
                });
            }
            sects
        } else {
            Vec::new()
        };

        // v7 carries RoPE tables as recipes instead of init bytes. Parse them
        // eagerly (a handful of 72-byte records) so the upload path does not need
        // the backing buffer, and reject anything we cannot materialise: a
        // recipe we skipped would leave the table zeroed, which reads as fluent
        // but wrong output rather than a failure.
        let gen = if is_v7 {
            let raw = sections
                .iter()
                .find(|s| s.kind == SECT_GEN_TENSORS)
                .and_then(|s| buf.get(s.offset..s.offset + s.size))
                .ok_or_else(|| {
                    RuntimeError::Device("devblob: v7 blob has no SECT_GEN_TENSORS section".into())
                })?;
            let sz = std::mem::size_of::<GenTensor>();
            if raw.len() % sz != 0 {
                return Err(RuntimeError::Device(format!(
                    "devblob: gen-tensor section is {} B, not a multiple of {sz}",
                    raw.len()
                )));
            }
            let mut off = 0usize;
            let g = take::<GenTensor>(raw, &mut off, raw.len() / sz, "gen tensors")?;
            for r in &g {
                if r.tensor as usize >= tensors.len() {
                    return Err(RuntimeError::Device(format!(
                        "devblob: gen recipe targets tensor {} of {}",
                        r.tensor,
                        tensors.len()
                    )));
                }
                if r.generate().is_none() {
                    return Err(RuntimeError::Device(format!(
                        "devblob: gen recipe for `{}` is unreadable — kind {}, scale {} \
                         — this blob needs a newer plowrt. `generate()` refuses on an unknown \
                         KIND or an unknown SCALE; both are printed because naming only the \
                         kind sent a reader looking at the wrong field when it was the scale \
                         (a DeepSeek-family YaRN blob read by a pre-ROPE_SCALE_YARN_DS runtime).",
                        tensors[r.tensor as usize].name, r.kind, r.scale
                    )));
                }
            }
            g
        } else {
            Vec::new()
        };

        // An extension references its parent's tensor table instead of declaring one. Both
        // halves are checked here rather than at merge time: a container that declares tensors
        // OR has no parent reference is not an extension at all, whatever its magic says, and
        // the merge contract would have nothing to check it against.
        let parent = if is_ext {
            if hdr.n_tensor != 0 || hdr.init_bytes != 0 {
                return Err(RuntimeError::Device(format!(
                    "devblob: extension declares {} tensors and {} B of init data — an \
                     extension carries programs only",
                    hdr.n_tensor, hdr.init_bytes
                )));
            }
            let mut refs = sections
                .iter()
                .filter(|s| s.kind == packet::ext::SECT_PARENT_REF);
            let s = refs.next().ok_or_else(|| {
                RuntimeError::Device(
                    "devblob: extension has no parent-ref section — nothing says which packet \
                     it extends"
                        .into(),
                )
            })?;
            if refs.next().is_some() {
                return Err(RuntimeError::Device(
                    "devblob: extension carries two parent-ref sections".into(),
                ));
            }
            let raw = buf.get(s.offset..s.offset + s.size).ok_or_else(|| {
                RuntimeError::Device("devblob: parent-ref section outside the container".into())
            })?;
            let wire = packet::ext::BlobParentRef::from_bytes(raw)
                .map_err(|e| RuntimeError::Device(format!("devblob: {e}")))?;
            Some(plow_asset::extension::ParentRef::from(&wire))
        } else {
            if sections
                .iter()
                .any(|s| s.kind == packet::ext::SECT_PARENT_REF)
            {
                return Err(RuntimeError::Device(
                    "devblob: model packet carries a parent-ref section — a packet cannot \
                     extend another packet"
                        .into(),
                ));
            }
            None
        };

        // PLOW_L2_PLACE guard: a placed blob requires physical-domain queue dispatch.
        //
        if hdr.flags & packet::devbuild::PLOW_BLOB_F_L2DOM != 0
            && !l2_dispatch_ok
            && !crate::config::RuntimeConfig::get().nv.l2_place_dispatch
        {
            return Err(RuntimeError::Device(
                "devblob: blob uses L2-domain packet placement (PLOW_L2_PLACE), so a standard \
                 interpreter would mis-dispatch its per-domain queues. Build the objects with \
                 -DPLOW_L2_PLACE_DISPATCH and set \
                 PLOW_L2_PLACE_DISPATCH=1, or recompile the model without PLOW_L2_PLACE."
                    .to_string(),
            ));
        }

        if hdr.target != 0 {
            tracing::info!(
                target_fp = format_args!("{:#010x}", hdr.target),
                n_cu = hdr.n_cu,
                "devblob: assets compiled for a specific GPU target — a backend that resolves \
                 its device can cross-check this fingerprint (Gap 4)"
            );
        }
        let tp = recover_tp(&progs);
        if let Some(t) = tp {
            tracing::info!(
                n_gpu = t.n_gpu,
                hidden = t.hidden,
                slot_bytes = t.slot_bytes,
                "devblob: SHARDED blob — every projection is 1/n_gpu wide"
            );
        }

        Ok(DevBlob {
            n_cu: hdr.n_cu,
            flags: hdr.flags,
            target: hdr.target,
            tensors,
            init,
            kvrow,
            progs,
            sections,
            gen,
            tp,
            parent,
        })
    }

    /// The tensor table as the extension contract digests it — see
    /// `plow_asset::extension::tensor_table_digest`.
    pub fn tensor_identities(&self) -> Vec<plow_asset::extension::TensorIdentity> {
        self.tensors
            .iter()
            .map(|t| plow_asset::extension::TensorIdentity {
                name: t.name.clone(),
                bytes: t.bytes,
                initialized: t.init.is_some(),
            })
            .collect()
    }

    /// TP degree the container was compiled for. `1` when there is no collective — a tp=1 blob
    /// emits none, so "no collective" and "not sharded" are the same fact.
    pub fn tp_degree(&self) -> u32 {
        self.tp.map_or(1, |t| t.n_gpu)
    }

    /// This container's tensor-table digest (rule 2). On a parent this is what an extension
    /// must match; on an extension it is meaningless (the table is empty) and the extension's
    /// CLAIM lives in [`DevBlob::parent`].
    pub fn tensor_table_digest(&self) -> plow_asset::extension::Digest {
        plow_asset::extension::tensor_table_digest(&self.tensor_identities(), self.tp_degree())
    }

    /// The programs' roles, derived the way the positional code does today.
    ///
    /// **Phase 1 adapter.** Once `ProgramRole` lands on `plow_asset::program::Program`, this
    /// reads `p.role` instead of re-deriving it, and the derivation and its `decode_rung_lo`
    /// call go away. Callers see the same `Vec<ProgramRole>` either way.
    pub fn program_roles(&self) -> Vec<plow_asset::extension::ProgramRole> {
        let bits: Vec<plow_asset::extension::RoleBits> = self
            .progs
            .iter()
            .map(|p| (p.t, p.packed_prefill_only, p.token_batch_body, p.decode_rung))
            .collect();
        // An extension has no positional boundary to derive from — see `roles_from_flags`.
        if self.parent.is_some() {
            plow_asset::extension::roles_from_flags(&bits)
        } else {
            plow_asset::extension::roles_from_positional(&bits, self.decode_rung_lo())
        }
    }

    /// What one program costs the arenas (rule 6). Instruction-stream bytes are the records
    /// the loader uploads per program — the same four arrays `AmdEngine::load` sends — plus
    /// the GQ appendix when the program carries one.
    pub fn program_budget(&self, p: &DevProg) -> plow_asset::extension::Budget {
        let bytes = |n: usize, sz: usize| (n * sz) as u64;
        plow_asset::extension::Budget {
            inst_stream_bytes: bytes(p.insts.len(), std::mem::size_of::<DevInst64>())
                + bytes(p.stream.len(), std::mem::size_of::<StreamEnt>())
                + bytes(p.stream_ofs.len() + p.stream_len.len() + p.succs.len(), 4)
                + bytes(p.waits.len(), std::mem::size_of::<Wait>())
                + bytes(p.gq_stream.len(), std::mem::size_of::<StreamEnt>())
                + bytes(p.gq_seg_ofs.len(), 4),
            counters: p.n_counter,
            segments: p.stream.iter().map(|e| e.seg as u32 + 1).max().unwrap_or(1),
            workspace_bytes: 0,
        }
    }

    /// The largest demand over every program in this container.
    pub fn budget(&self) -> plow_asset::extension::Budget {
        self.progs
            .iter()
            .map(|p| self.program_budget(p))
            .fold(plow_asset::extension::Budget::default(), |a, b| {
                plow_asset::extension::Budget {
                    inst_stream_bytes: a.inst_stream_bytes.max(b.inst_stream_bytes),
                    counters: a.counters.max(b.counters),
                    segments: a.segments.max(b.segments),
                    workspace_bytes: a.workspace_bytes.max(b.workspace_bytes),
                }
            })
    }

    /// Get a section by kind and architecture-specific name.
    pub fn section_data_named<'a>(&self, buf: &'a [u8], kind: u32, name: &str) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|s| s.kind == kind && s.name == name)
            .and_then(|s| buf.get(s.offset..s.offset + s.size))
    }

    pub fn reserved_metadata<'a>(&self, buf: &'a [u8], name: &str) -> Result<Option<&'a [u8]>> {
        let mut matches = self.sections.iter().filter(|section| section.name == name);
        let Some(section) = matches.next() else {
            return Ok(None);
        };
        if matches.next().is_some() || section.kind != packet::devbuild::SECT_METADATA {
            return Err(RuntimeError::Rejected(format!(
                "{name} requires exactly one metadata section"
            )));
        }
        let end = section
            .offset
            .checked_add(section.size)
            .ok_or_else(|| RuntimeError::Rejected(format!("{name} section range overflow")))?;
        buf.get(section.offset..end)
            .map(Some)
            .ok_or_else(|| RuntimeError::Rejected(format!("{name} section range outside packet")))
    }

    pub fn section_data<'a>(&self, buf: &'a [u8], kind: u32) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|s| s.kind == kind)
            .and_then(|s| buf.get(s.offset..s.offset + s.size))
    }

    pub fn with_packet_view<T>(&self, f: impl FnOnce(&plow_asset::program::Packet<'_>) -> T) -> T {
        use plow_asset::program::{Packet, Program, Tensor};
        let tensors: Vec<_> = self
            .tensors
            .iter()
            .map(|t| Tensor {
                name: &t.name,
                bytes: t.bytes,
                initialized: t.init.is_some(),
            })
            .collect();
        let programs: Vec<_> = self
            .progs
            .iter()
            .map(|p| Program {
                rows: p.t,
                packed_prefill_only: p.packed_prefill_only,
                token_batch_body: p.token_batch_body,
                n_counter: p.n_counter,
                insts: &p.insts,
                stream: &p.stream,
                stream_ofs: &p.stream_ofs,
                stream_len: &p.stream_len,
                waits: &p.waits,
                succs: &p.succs,
                gq_stream: &p.gq_stream,
                gq_seg_ofs: &p.gq_seg_ofs,
                l2_domains: p.l2_domains,
            })
            .collect();
        f(&Packet {
            n_cu: self.n_cu,
            tp: self.tp.is_some(),
            prefill_count: self.decode_rung_lo(),
            tensors: &tensors,
            programs: &programs,
            generated: &self.gen,
            kv_row_insts: &self.kvrow,
        })
    }

    /// Index of the first decode rung. The compiler emits prefill buckets first,
    /// then a trailing ascending decode ladder whose widths are at most 128.
    pub fn decode_rung_lo(&self) -> usize {
        let widths: Vec<u32> = self.progs.iter().map(|p| p.t).collect();
        packet::devbuild::decode_rung_lo(&widths)
    }

    /// Prefill bucket programs, excluding every decode rung.
    pub fn prefill_progs(&self) -> &[DevProg] {
        &self.progs[..self.decode_rung_lo()]
    }

    /// Decode rung programs in ascending width order.
    pub fn decode_progs(&self) -> &[DevProg] {
        &self.progs[self.decode_rung_lo()..]
    }

    /// Widths advertised by the decode ladder.
    pub fn decode_rungs(&self) -> Vec<u32> {
        self.decode_progs().iter().map(|p| p.t).collect()
    }

    /// The widest decode program. This remains the last program for both the
    /// legacy one-rung blob and a decode ladder.
    pub fn decode_prog(&self) -> Result<&DevProg> {
        let g = self
            .decode_progs()
            .last()
            .ok_or_else(|| RuntimeError::Device("devblob: no programs".into()))?;
        if g.t == 0 || g.t > packet::devbuild::DECODE_RUNG_MAX {
            return Err(RuntimeError::Device(format!(
                "devblob: last program has T={} — not the decode program (batch 1..={})",
                g.t,
                packet::devbuild::DECODE_RUNG_MAX
            )));
        }
        Ok(g)
    }

    /// Find the (single) device blob in an assets dir: any file whose first 8
    /// bytes are the `PLOWDEV` magic. Two candidates is an error — the layout
    /// is ambiguous and picking one silently would serve the wrong model.
    pub fn find_in_dir(dir: &Path) -> Result<Option<PathBuf>> {
        let mut found: Option<PathBuf> = None;
        let entries = std::fs::read_dir(dir).map_err(|source| RuntimeError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        for e in entries.flatten() {
            let path = e.path();
            if !path.is_file() {
                continue;
            }
            let mut magic = [0u8; 8];
            let ok = std::fs::File::open(&path)
                .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
                .is_ok();
            if ok && is_blob_magic(&magic) {
                if let Some(prev) = &found {
                    return Err(RuntimeError::Device(format!(
                        "devblob: both {} and {} carry the PLOWDEV magic — ambiguous",
                        prev.display(),
                        path.display()
                    )));
                }
                found = Some(path);
            }
        }
        Ok(found)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SegmentClassPolicy {
    pub pure_mode: u8,
    pub fa512_mode: u8,
    pub fa256_gqa2: bool,
}

impl DevProg {
    pub(crate) fn inferred_segment_policy(&self) -> SegmentClassPolicy {
        let mut n_seg = 1usize;
        for entry in &self.stream {
            n_seg = n_seg.max(entry.seg as usize + 1);
        }
        let mut mapped_gemm = vec![false; n_seg];
        let mut mapless_w8a16 = vec![false; n_seg];
        let mut hd512 = vec![false; n_seg];
        let mut hd256_gqa2 = vec![false; n_seg];
        let mut other_for_gemm = vec![false; n_seg];
        let mut other_for_hd512 = vec![false; n_seg];
        let mut other_for_gqa2 = vec![false; n_seg];
        for entry in &self.stream {
            let Some(inst) = self.insts.get(entry.inst as usize) else {
                continue;
            };
            let seg = entry.seg as usize;
            let plain_gemm = matches!(
                DevOp::from_u16(inst.op),
                Some(
                    DevOp::Gemm
                        | DevOp::GemmMed
                        | DevOp::GemmSmall
                        | DevOp::GemmFp8
                        | DevOp::GemmMedFp8
                        | DevOp::GemmSmallFp8
                )
            );
            let mapped = plain_gemm && inst.i[6] != 0 && inst.i[7] != 0;
            let w8a16 = inst.is_mapless_w8a16_gemm();
            mapped_gemm[seg] |= mapped;
            mapless_w8a16[seg] |= w8a16;
            other_for_gemm[seg] |= !mapped && !w8a16;

            let is_hd512 = inst.op == DevOp::FlashPrefill as u16 && inst.i[6] == 512;
            hd512[seg] |= is_hd512;
            other_for_hd512[seg] |= !is_hd512;

            let is_gqa2 = inst.is_hd256_gqa2_sliding_prefill();
            hd256_gqa2[seg] |= is_gqa2;
            other_for_gqa2[seg] |= !is_gqa2;
        }
        let pure = mapped_gemm
            .iter()
            .zip(&mapless_w8a16)
            .zip(&other_for_gemm)
            .any(|((&mapped, &w8a16), &other)| (mapped || w8a16) && !other);
        let w8a16 = mapless_w8a16
            .iter()
            .zip(&other_for_gemm)
            .any(|(&present, &other)| present && !other);
        SegmentClassPolicy {
            pure_mode: if w8a16 { 3 } else if pure { 1 } else { 0 },
            fa512_mode: if hd512
                .iter()
                .zip(&other_for_hd512)
                .any(|(&present, &other)| present && !other)
            {
                1
            } else {
                0
            },
            fa256_gqa2: hd256_gqa2
                .iter()
                .zip(&other_for_gqa2)
                .any(|(&present, &other)| present && !other),
        }
    }

    /// The coarse single-segment gate the sm_120 interpreter implements: every
    /// stream entry must be unsegmented (`seg == 0`) with no per-slice or
    /// cross-GPU counters. Mirrors the harness's fatal check.
    /// Per-segment wave class of a wave-class segmented program: 8 = GEMM-class,
    /// 4 = flash-class (contains a FlashPrefill op). Mirror of the AMD engine's
    /// `derive_segments`, hoisted here so the CUDA engine (which builds without
    /// the `hsa` module) can classify segments for the SegPf launcher.
    pub fn seg_classes(&self) -> Result<Vec<u8>> {
        let rt = crate::config::RuntimeConfig::get();
        let pure_mode = match rt.nv.pf_seg_pure.as_deref() {
            Some("1") => 1u8,
            Some("fp8") => 2u8,
            Some("w8a16") => 3u8,
            _ => 0u8,
        };
        let fa512_mode = match rt.nv.pf_seg_fa512.as_deref() {
            Some("1") => 1u8,
            Some("all") => 2u8,
            _ => 0u8,
        };
        self.seg_classes_with(SegmentClassPolicy {
            pure_mode,
            fa512_mode,
            fa256_gqa2: rt.nv.pf_seg_fa256_gqa2,
        })
    }

    pub(crate) fn seg_classes_with(&self, policy: SegmentClassPolicy) -> Result<Vec<u8>> {
        let mut n_seg: u32 = 1;
        for e in &self.stream {
            n_seg = n_seg.max(e.seg as u32 + 1);
        }
        // T37: 2048 covers a 60-layer model's ~10 wave-class runs per layer with headroom
        // (512 was sized for 48 layers and tripped on Gemma-4-31B's 603).
        if n_seg > 2048 {
            return Err(RuntimeError::Device(format!(
                "program declares {n_seg} segments (max 2048) — corrupt stream?"
            )));
        }
        // Pure mode mirrors the emit-side PLOW_SEG_PURE_GEMM classing — a segment
        // is GEMM-class (8) only if EVERY op in it is a GEMM-family op; anything else (norms,
        // rope, quant, glu, flash) makes it flash-class (4). Required when the lean object is
        // built PLOW_NV_GEMM_ONLY: its dispatch traps on any non-GEMM opcode, so a light-op
        // segment classified 8 would land there. The CUDA loader recovers this mode from the
        // packet's isolated segments; PLOW_PF_SEG_PURE remains an explicit override.
        // Must match the emit-side classing in devbuild.rs: "1" = every plain tiled GEMM,
        // "fp8" = only TMA-mapped fp8 GEMMs (the ws-entry object's sole arm).
        // "w8a16" pairs mapless W8A16 plus mapped BF16 with a capability-checked object.
        let pure_mode = policy.pure_mode;
        use packet::dev::DevOp;
        // PLOW_PF_SEG_FA512=1 (T12): hd512 FlashPrefill segments class 2 — launched on the
        // dedicated *_pffa object. Mirror of the emit-side PLOW_SEG_FA512.
        let fa512_mode = policy.fa512_mode;
        let fa256_gqa2 = policy.fa256_gqa2;
        let rt = crate::config::RuntimeConfig::get();
        let v2_env = rt.nv.pf_seg_v2.as_deref();
        let seg_v2 = v2_env == Some("1");
        let seg_q8 = seg_v2 || v2_env == Some("q8");
        const FP8_OPS: [DevOp; 3] = [DevOp::GemmFp8, DevOp::GemmMedFp8, DevOp::GemmSmallFp8];
        const BF16_OPS: [DevOp; 3] = [DevOp::Gemm, DevOp::GemmSmall, DevOp::GemmMed];
        let mut class = vec![8u8; n_seg as usize];
        let mut exact_fa256 = vec![false; n_seg as usize];
        let mut other_in_exact_segment = vec![false; n_seg as usize];
        for e in &self.stream {
            let inst = self.insts.get(e.inst as usize).ok_or_else(|| {
                RuntimeError::Device(format!(
                    "stream entry references instruction {} of {}",
                    e.inst,
                    self.insts.len()
                ))
            })?;
            let op = inst.op;
            if fa256_gqa2 && inst.is_hd256_gqa2_sliding_prefill() {
                exact_fa256[e.seg as usize] = true;
            } else {
                other_in_exact_segment[e.seg as usize] = true;
            }
            let flash_op = op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16;
            if flash_op
                && ((fa512_mode == 2 && (inst.i[6] == 256 || inst.i[6] == 512))
                    || (fa512_mode == 1 && inst.i[6] == 512))
            {
                class[e.seg as usize] = 2;
                continue;
            }
            // PLOW_PF_SEG_V2=1 (T16): mirror of the emit-side PLOW_SEG_V2 classing.
            if seg_v2 {
                if fa512_mode == 2
                    && (op == DevOp::HeadNormRope as u16
                        || op == DevOp::HeadNormRopeFp8 as u16
                        || op == DevOp::FlashMerge as u16)
                {
                    class[e.seg as usize] = 2;
                    continue;
                }
            }
            if seg_q8 && pure_mode == 2 && op == DevOp::QuantFp8 as u16 {
                // class stays 8 (the default) — fall through without forcing 4.
                continue;
            }
            let flashy = match pure_mode {
                3 => {
                    !(inst.is_mapless_w8a16_gemm()
                        || (BF16_OPS.iter().any(|g| *g as u16 == op)
                            && inst.i[6] != 0
                            && inst.i[7] != 0))
                }
                1 => {
                    // T37 mirror: maps required in mode 1 too (see devbuild.rs).
                    !((FP8_OPS.iter().any(|g| *g as u16 == op)
                        || BF16_OPS.iter().any(|g| *g as u16 == op))
                        && inst.i[6] != 0
                        && inst.i[7] != 0)
                }
                2 => {
                    // T24 mirror: mapped bf16 GEMMs class 8 too (see devbuild.rs).
                    !((FP8_OPS.iter().any(|g| *g as u16 == op)
                        || BF16_OPS.iter().any(|g| *g as u16 == op))
                        && inst.i[6] != 0
                        && inst.i[7] != 0)
                }
                _ => op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16,
            };
            if flashy {
                class[e.seg as usize] = 4;
            }
        }
        for seg in 0..class.len() {
            if exact_fa256[seg] {
                if other_in_exact_segment[seg] {
                    return Err(RuntimeError::Rejected(format!(
                        "segment {seg} mixes exact HD256/GQA2 attention with another operator; \
                         recompile with PLOW_SEG_FA256_GQA2=1"
                    )));
                }
                class[seg] = 3;
            }
        }
        Ok(class)
    }

    pub fn check_coarse_single_segment(&self) -> Result<()> {
        // Legacy L2 blobs encoded the domain in `seg`; current blobs keep ordered segments in
        // `seg` and encode the physical domain in flags. The CUDA backend may accept the former
        // compatibility layout, but it must reject any current blob with more than one ordered
        // segment instead of mistaking low segment ids for legacy domains.
        let legacy_l2 = self.l2_domains != 0
            && self.gq_seg_ofs.len() == self.l2_domains as usize + 1
            && self.stream.iter().all(|e| e.flags & SE_DOMAIN_MASK == 0);
        let seg_lim = if legacy_l2 { self.l2_domains as u16 } else { 1 };
        for (j, e) in self.stream.iter().enumerate() {
            if e.seg >= seg_lim || (e.flags & (packet::dev::SE_FINE | packet::dev::SE_XCTR)) != 0 {
                return Err(RuntimeError::Device(format!(
                    "devblob: prog T={} stream entry {j} is segmented/fine-gated; the sm_120 \
                     interpreter implements the coarse single-segment path only",
                    self.t
                )));
            }
        }
        Ok(())
    }

    /// The global-queue topology check ported from the harness: for every
    /// wait, the latest producer of that counter must precede the waiter in
    /// instruction order — otherwise the single-cursor GQ schedule deadlocks.
    ///
    /// Gates live on the stream entries (the 64-byte wire instruction carries
    /// none), so this walks the stream; entries of the same coarse inst repeat
    /// the same lists, which changes nothing about the max/compare below.
    pub fn check_gq_topological(&self) -> Result<()> {
        let ni = self.insts.len();
        let mut prod_max = vec![-1i64; self.n_counter as usize];
        for e in &self.stream {
            let i = e.inst as i64;
            for s in 0..e.succ_len as usize {
                let c = self.succs[e.succ_ofs as usize + s] as usize;
                if i > prod_max[c] {
                    prod_max[c] = i;
                }
            }
        }
        for e in &self.stream {
            let i = e.inst as i64;
            for w in 0..e.wait_len as usize {
                let c = self.waits[e.wait_ofs as usize + w].id as usize;
                if prod_max[c] >= i {
                    return Err(RuntimeError::Device(format!(
                        "devblob: prog T={} inst {i} waits on counter {c} whose latest \
                         producer is inst {} — not topological, GQ would deadlock (of {ni})",
                        self.t, prod_max[c]
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevInst;
    use packet::devbuild::{Model, Program, TensorDecl};

    #[test]
    fn packet_segments_recover_safe_cuda_object_routing() {
        let mut insts = vec![DevInst64 {
            op: DevOp::Embed as u16,
            blocks: 1,
            fj: [0; 3],
            t: [0; 8],
            i: [0; 8],
        }; 4];
        insts[1].op = DevOp::Gemm as u16;
        insts[1].i[6] = 1;
        insts[1].i[7] = 2;
        insts[2].op = DevOp::FlashPrefill as u16;
        insts[2].i[6] = 512;
        insts[3].op = DevOp::FlashPrefill as u16;
        insts[3].i = [128, 128, 16, 8, 0, 1024, 256, 1];
        insts[3].t[5] = 1;
        let stream = (0..4)
            .map(|index| StreamEnt {
                inst: index,
                seg: index as u16,
                ..StreamEnt::default()
            })
            .collect();
        let program = DevProg {
            t: 128,
            packed_prefill_only: false,
            token_batch_body: false,
            decode_rung: false,
            n_counter: 0,
            insts,
            stream,
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains: 0,
        };
        let policy = program.inferred_segment_policy();
        assert_eq!(
            policy,
            SegmentClassPolicy {
                pure_mode: 1,
                fa512_mode: 1,
                fa256_gqa2: true,
            }
        );
        assert_eq!(program.seg_classes_with(policy).unwrap(), [4, 8, 2, 3]);
    }

    #[test]
    fn mixed_light_and_gemm_segment_does_not_claim_a_gemm_only_object() {
        let insts = vec![
            DevInst64 {
                op: DevOp::Gemm as u16,
                blocks: 1,
                fj: [0; 3],
                t: [0; 8],
                i: [0, 0, 0, 0, 0, 0, 1, 2],
            },
            DevInst64 {
                op: DevOp::RmsNorm as u16,
                blocks: 1,
                fj: [0; 3],
                t: [0; 8],
                i: [0; 8],
            },
        ];
        let stream = (0..2)
            .map(|inst| StreamEnt {
                inst,
                ..StreamEnt::default()
            })
            .collect();
        let program = DevProg {
            t: 128,
            packed_prefill_only: false,
            token_batch_body: false,
            decode_rung: false,
            n_counter: 0,
            insts,
            stream,
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains: 0,
        };
        assert_eq!(
            program.inferred_segment_policy(),
            SegmentClassPolicy::default()
        );
    }

    /// `hidden` is a ROW width, and a one-shot collective in a PREFILL program says
    /// `t * hidden`. This used to read `i[0]` outright, which was right only while the
    /// one-shot belonged to decode alone. Kimi-K3's shared-expert reduce carries a folded
    /// all-gather the two-shot cannot express, so it is one-shot at every T — and the
    /// 8192-row bucket then reported hidden = 58,720,256. The host divides `slot_bytes` by
    /// `hidden * 2` to recover `max_tokens`, so a wrong width is a wrong peer layout with no
    /// message: every rank's partial lands where no peer reads it.
    #[test]
    fn tp_hidden_is_recovered_per_row_from_any_phase() {
        let xr = |width: u32, slot: u32| DevInst64 {
            op: DevOp::XReduce as u16,
            blocks: 1,
            fj: [0; 3],
            t: [0; 8],
            i: [width, 8, slot, 0, 0, 0, 0, 0],
        };
        let prog = |t: u32, insts: Vec<DevInst64>| DevProg {
            t,
            packed_prefill_only: false,
            token_batch_body: false,
            decode_rung: false,
            n_counter: 0,
            insts,
            stream: Vec::new(),
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains: 0,
        };
        // Decode reduces one row at hidden AND one at K3's narrower latent; the 8192-row
        // prefill bucket reduces the same hidden width, 8192 rows at a time.
        let hidden = 7168u32;
        let progs = vec![
            prog(1, vec![xr(hidden, 0), xr(3584, 117_440_512)]),
            prog(8192, vec![xr(8192 * hidden, 0)]),
        ];
        let tp = recover_tp(&progs).expect("n_gpu > 1 must be recovered");
        assert_eq!(tp.n_gpu, 8);
        assert_eq!(tp.hidden, hidden, "the widest ROW, not the widest message");
        assert_eq!(tp.slot_bytes, 117_440_512);
        // And a blob with no collective at all is not a sharded blob.
        assert!(recover_tp(&[prog(1, vec![])]).is_none());
    }

    /// A tiny two-program model exercised through the REAL writer
    /// (`Model::to_blob`) — reader and writer cannot drift apart unnoticed.
    fn tiny_model() -> Model {
        let inst = |op: u16, wait_len: u16, succ_len: u16, wait_ofs: u32, succ_ofs: u32| DevInst {
            op,
            blocks: 1,
            wait_len,
            succ_len,
            wait_ofs,
            succ_ofs,
            t: [0; 8],
            i: [7; 8],
            f: [0.5; 2],
            j: [0; 2],
        };
        let se = |inst: u32, slice: u32| StreamEnt {
            inst,
            slice,
            wait_ofs: 0,
            succ_ofs: 0,
            wait_len: 0,
            succ_len: 0,
            flags: 0,
            seg: 0,
        };
        // Two CUs. inst 0 signals counter 0; inst 1 waits on it.
        let prog = || Program {
            n_cu: 2,
            n_counter: 1,
            hier_base: 0,
            insts: vec![inst(6, 0, 1, 0, 0), inst(18, 1, 0, 0, 0)],
            stream: vec![se(0, 0), se(1, 0), se(0, 1), se(1, 1)],
            stream_ofs: vec![0, 2],
            stream_len: vec![2, 2],
            waits: vec![Wait {
                id: 0,
                threshold: 1,
            }],
            succs: vec![0],
            tensors: Vec::new(),
            gq_stream: vec![se(0, 0), se(0, 1), se(1, 0), se(1, 1)],
            gq_seg_ofs: vec![0, 4],
            // Unplaced: `seg` is a wave-class, not an L2 domain (PLOW_L2_PLACE).
            l2_sms: 0,
            l2_domains: 0,
        };
        Model {
            n_cu: 2,
            // Unspecified target GPU ⇒ the runtime skips the mismatch warning.
            target: 0,
            tensors: vec![
                TensorDecl {
                    name: "in.ids".into(),
                    bytes: 4,
                    init: None,
                },
                TensorDecl {
                    name: "rope.cos".into(),
                    bytes: 8,
                    init: Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
                },
            ],
            progs: vec![prog(), prog()],
            kv_row_insts: vec![1],
            prog_t: vec![128, 1],
            gen: Vec::new(),
        }
    }

    /// Give program `p` the two collectives `devgen` emits per layer at tp=N:
    /// slot A (`i[2] == 0`) and slot B (`i[2] == slot_b`).
    fn with_xreduce(m: &mut Model, p: usize, one_shot: bool, n_gpu: u32, elems: u32, slot_b: u32) {
        let op = if one_shot {
            DevOp::XReduce
        } else {
            DevOp::XReduceTwoShot
        } as u16;
        for slot in [0, slot_b] {
            let mut d = DevInst {
                op,
                blocks: 1,
                ..Default::default()
            };
            d.i[0] = elems;
            d.i[1] = n_gpu;
            d.i[2] = slot;
            m.progs[p].insts.push(d);
        }
    }

    /// A tp=1 blob emits no collective, so it must report `None` — not
    /// `Some(n_gpu: 1)`. The single-GPU path keys off exactly this, and a blob
    /// that claimed to be a 1-way shard would be refused by the load-time check.
    #[test]
    fn an_unsharded_blob_reports_no_tp() {
        let b = DevBlob::parse(&tiny_model().to_blob()).unwrap();
        assert_eq!(b.tp, None);
    }

    /// At decode `t == 1`, `i[0]` is the hidden size directly. `slot_bytes` is
    /// `max(i[2])` because slot A carries 0.
    #[test]
    fn a_sharded_decode_blob_describes_itself() {
        let mut m = tiny_model();
        let dp = m.progs.len() - 1;
        with_xreduce(&mut m, dp, true, 4, 5376, 5376 * 2);
        let tp = DevBlob::parse(&m.to_blob()).unwrap().tp.expect("sharded");
        assert_eq!(tp.n_gpu, 4);
        assert_eq!(tp.hidden, 5376, "decode's i[0] at t==1 is hidden");
        assert_eq!(tp.slot_bytes, 5376 * 2, "max(i[2]), since slot A is 0");
    }

    /// Both collective forms carry `i[0] = t·hidden`, so dividing by the
    /// program row count recovers hidden from either form.
    #[test]
    fn two_shot_supplies_hidden() {
        let h = 5376u32;
        let slot_b = 128 * h * 2; // tiny_model's largest program is 128 rows
        let mut m = tiny_model();
        let dp = m.progs.len() - 1;
        with_xreduce(&mut m, 0, false, 4, 128 * h, slot_b); // prefill bucket
        with_xreduce(&mut m, dp, true, 4, h, slot_b); // decode

        let tp = DevBlob::parse(&m.to_blob()).unwrap().tp.expect("sharded");
        assert_eq!(tp.n_gpu, 4);
        assert_eq!(tp.hidden, h, "NOT 1024*h — two-shot's i[0] is t*hidden");
        assert_eq!(
            tp.slot_bytes, slot_b as u64,
            "devgen bakes the SAME slot_b into every program"
        );

        // A prefill-only asset is just as unloadable on one GPU, so the scan
        // must see it even with no decode collective to fall back on.
        let mut pf = tiny_model();
        with_xreduce(&mut pf, 0, false, 2, 128 * h, slot_b);
        let only = DevBlob::parse(&pf.to_blob()).unwrap().tp.expect("sharded");
        assert_eq!((only.n_gpu, only.slot_bytes), (2, slot_b as u64));
        assert_eq!(only.hidden, h, "two-shot supplies t*hidden");
    }

    /// A v7 blob must be DISCOVERED, parsed, and its recipes materialised.
    ///
    /// The discovery half is the regression: `parse` and `find_in_dir` each had
    /// their own magic list, v7 was added to `parse` only, and the result was a
    /// blob that loaded perfectly but that `plowrt serve` never found. Both now
    /// go through `is_blob_magic`; this pins that they agree.
    #[test]
    fn v7_blob_is_found_parsed_and_materialised() {
        use packet::rope::{rope_tables, RopeScale};

        let ctx = 8u32;
        let hd = 16u32;
        let [cos, sin] = GenTensor::rope_pair(ctx, hd, 10000.0, 1.0, RopeScale::None);
        let mut m = tiny_model();
        // Two tensors the runtime must build rather than upload.
        m.tensors.push(TensorDecl {
            name: "in.cos_full".into(),
            bytes: cos.byte_len(),
            init: None,
        });
        m.tensors.push(TensorDecl {
            name: "in.sin_full".into(),
            bytes: sin.byte_len(),
            init: None,
        });
        m.gen = vec![
            GenTensor { tensor: 2, ..cos },
            GenTensor { tensor: 3, ..sin },
        ];
        let image = m.to_blob();
        assert_eq!(
            &image[..8],
            BLOB_MAGIC_V7,
            "a model with recipes must be v7"
        );

        // Discovery: written into an assets dir, `find_in_dir` must see it.
        let dir = std::env::temp_dir().join(format!("plow_v7_find_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.pkt"), &image).unwrap();
        let found = DevBlob::find_in_dir(&dir).unwrap();
        assert_eq!(
            found.as_deref(),
            Some(dir.join("model.pkt").as_path()),
            "find_in_dir must discover a v7 blob"
        );

        let b = DevBlob::parse(&image).unwrap();
        assert_eq!(b.gen.len(), 2, "both recipes survive the round trip");
        assert!(b.tensors[2].init.is_none() && b.tensors[3].init.is_none());
        // And they expand to exactly what the compiler would have baked.
        let (want_cos, want_sin) = rope_tables(ctx, hd, 10000.0, 1.0, RopeScale::None);
        let by = |i: u32| {
            b.gen
                .iter()
                .find(|g| g.tensor == i)
                .unwrap()
                .generate()
                .unwrap()
        };
        assert_eq!(by(2), want_cos);
        assert_eq!(by(3), want_sin);
        assert_eq!(
            by(2).len() as u64,
            b.tensors[2].bytes,
            "decl size matches recipe"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A recipe pointing past the tensor table must be rejected, not indexed —
    /// otherwise a corrupt blob panics the server mid-load.
    #[test]
    fn v7_rejects_out_of_range_recipe() {
        use packet::rope::RopeScale;
        let [cos, _] = GenTensor::rope_pair(4, 8, 10000.0, 1.0, RopeScale::None);
        let mut m = tiny_model();
        m.tensors.push(TensorDecl {
            name: "in.cos_full".into(),
            bytes: cos.byte_len(),
            init: None,
        });
        m.gen = vec![GenTensor { tensor: 2, ..cos }];
        let mut image = m.to_blob();

        // Patch the serialised recipe's `tensor` field to a handle that does not
        // exist — what a truncated or mismatched blob looks like on the wire.
        let dir_off = u64::from_le_bytes(image[40..48].try_into().unwrap()) as usize;
        let ent = dir_off + 8; // first section entry: SECT_GEN_TENSORS
        let off = u64::from_le_bytes(image[ent + 8..ent + 16].try_into().unwrap()) as usize;
        assert_eq!(
            &image[off..off + 4],
            &2u32.to_le_bytes(),
            "recipe 0 targets tensor 2"
        );
        image[off..off + 4].copy_from_slice(&99u32.to_le_bytes());

        let err = match DevBlob::parse(&image) {
            Ok(_) => panic!("parse accepted an out-of-range gen recipe"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("gen recipe targets tensor 99"), "got: {err}");
    }

    #[test]
    fn mixed_placement_preserves_unplaced_segment_windows() {
        let mut model = tiny_model();
        let prefill = &mut model.progs[0];
        for entry in prefill.stream.iter_mut().chain(&mut prefill.gq_stream) {
            entry.seg = entry.inst as u16;
        }
        prefill.gq_seg_ofs = vec![0, 2, 4];

        let decode = &mut model.progs[1];
        decode.l2_domains = 2;
        decode.l2_sms = 1;
        for entry in &mut decode.stream {
            entry.flags = (entry.slice as u16) << packet::dev::SE_DOMAIN_SHIFT;
        }
        decode.gq_stream = decode.stream.clone();
        decode.gq_seg_ofs = vec![0, 2, 4];

        let parsed = DevBlob::parse_l2(&model.to_blob(), true).unwrap();
        assert_eq!(parsed.progs[0].l2_domains, 0);
        assert_eq!(parsed.progs[1].l2_domains, 2);
    }

    #[test]
    fn roundtrip_through_the_real_writer() {
        let blob = tiny_model().to_blob();
        let mut b = DevBlob::parse(&blob).unwrap();
        assert_eq!(b.n_cu, 2);
        assert_eq!(b.tensors.len(), 2);
        assert_eq!(b.tensors[0].name, "in.ids");
        assert!(b.tensors[0].init.is_none());
        assert_eq!(b.tensors[1].name, "rope.cos");
        assert_eq!(
            &b.init[b.tensors[1].init.clone().unwrap()],
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(b.kvrow, vec![1]);
        assert_eq!(b.progs.len(), 2);
        assert_eq!(b.progs[0].t, 128);

        let g = b.decode_prog().unwrap();
        assert_eq!(g.t, 1);
        assert_eq!(g.insts.len(), 2);
        assert_eq!(g.insts[1].op, 18);
        assert_eq!(g.stream.len(), 4);
        assert_eq!(g.stream_ofs, vec![0, 2]);
        assert_eq!(g.waits[0].threshold, 1);
        // GQ appendix decoded: op-major permutation, single segment window.
        assert_eq!(g.gq_stream.len(), 4);
        assert_eq!(g.gq_stream[1].inst, 0);
        assert_eq!(g.gq_seg_ofs, vec![0, 4]);

        g.check_coarse_single_segment().unwrap();
        g.check_gq_topological().unwrap();

        // Current placed blobs keep ordered segments independent of their domain windows.
        // A coarse-only backend must not confuse segment 1 with legacy domain 1.
        let g = b.progs.last_mut().unwrap();
        g.l2_domains = 8;
        g.gq_seg_ofs = vec![0; 9];
        g.stream[0].seg = 1;
        g.stream[0].flags = 1u16 << packet::dev::SE_DOMAIN_SHIFT;
        assert!(g.check_coarse_single_segment().is_err());
    }

    #[test]
    fn program_roles_cover_a_decode_ladder() {
        let mut m = tiny_model();
        m.progs.pop();
        for _ in 0..5 {
            m.progs.push(tiny_model().progs.pop().unwrap());
        }
        m.prog_t = vec![128, 1, 2, 4, 8, 16];

        let b = DevBlob::parse(&m.to_blob()).unwrap();
        assert_eq!(b.decode_rung_lo(), 1);
        assert_eq!(
            b.prefill_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
            vec![128]
        );
        assert_eq!(b.decode_rungs(), vec![1, 2, 4, 8, 16]);
        assert_eq!(b.decode_prog().unwrap().t, 16);
    }

    #[test]
    fn packed_prefill_program_tag_is_normalized_but_retained_as_a_role() {
        let mut m = tiny_model();
        m.progs.insert(1, tiny_model().progs.pop().unwrap());
        m.prog_t = vec![128, packet::devbuild::packed_prefill_program_t(128), 1];

        let b = DevBlob::parse(&m.to_blob()).unwrap();
        assert_eq!(b.decode_rung_lo(), 2);
        assert_eq!(b.progs[0].t, 128);
        assert!(!b.progs[0].packed_prefill_only);
        assert_eq!(b.progs[1].t, 128);
        assert!(b.progs[1].packed_prefill_only);
        assert_eq!(b.decode_rungs(), vec![1]);
    }

    #[test]
    fn bad_magic_and_truncation_fail_loudly() {
        let blob = tiny_model().to_blob();
        let mut bad = blob.clone();
        bad[0] = b'X';
        assert!(DevBlob::parse(&bad).is_err());
        assert!(DevBlob::parse(&blob[..blob.len() / 3]).is_err());
    }

    #[test]
    fn reserved_metadata_is_name_strict_and_bounds_checked() {
        let raw = tiny_model().to_blob();
        let mut blob = DevBlob::parse(&raw).unwrap();
        let name = "reserved.json";
        assert!(blob.reserved_metadata(&raw, name).unwrap().is_none());

        let good = DevSection {
            kind: packet::devbuild::SECT_METADATA,
            name: name.into(),
            offset: 1,
            size: 3,
        };
        blob.sections.push(good);
        assert_eq!(
            blob.reserved_metadata(&raw, name).unwrap(),
            Some(&raw[1..4])
        );

        blob.sections.last_mut().unwrap().kind = packet::devbuild::SECT_METADATA + 1;
        assert!(blob.reserved_metadata(&raw, name).is_err());
        blob.sections.last_mut().unwrap().kind = packet::devbuild::SECT_METADATA;
        blob.sections.push(DevSection {
            kind: packet::devbuild::SECT_METADATA + 1,
            name: name.into(),
            offset: 0,
            size: 0,
        });
        assert!(blob.reserved_metadata(&raw, name).is_err());
        blob.sections.pop();
        blob.sections.last_mut().unwrap().offset = usize::MAX;
        blob.sections.last_mut().unwrap().size = 1;
        assert!(blob.reserved_metadata(&raw, name).is_err());
    }

    #[test]
    fn gq_cycle_is_rejected() {
        let mut m = tiny_model();
        // Make inst 0 wait on the counter inst 1 signals: producer follows
        // the waiter, which the single-cursor GQ schedule cannot survive.
        // Gates live on the stream entries in the 64-byte wire format.
        for p in &mut m.progs {
            for e in p.stream.iter_mut().chain(p.gq_stream.iter_mut()) {
                (e.wait_len, e.succ_len) = if e.inst == 0 { (1, 0) } else { (0, 1) };
            }
        }
        let b = DevBlob::parse(&m.to_blob()).unwrap();
        assert!(b.decode_prog().unwrap().check_gq_topological().is_err());
    }
}
