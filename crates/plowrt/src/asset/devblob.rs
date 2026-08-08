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

use packet::dev::{DevInst64, DevOp, StreamEnt, Wait};
use packet::devbuild::{
    is_blob_magic, BlobHeader, BlobProgHeader, BlobSectionEntry, BlobTensor, BLOB_MAGIC_V7,
    INIT_NONE, NAME_LEN, SECT_GEN_TENSORS, SECT_MAGIC, SECT_NAME_LEN,
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
    /// `[n_seg+1]` segment window bounds into `gq_stream`.
    pub gq_seg_ofs: Vec<u32>,
    /// L2-domain placement (`PLOW_L2_PLACE`): the number of L2 domains `gq_seg_ofs` is windowed
    /// by, `0` when this program is not placed and the windows are wave-class segments.
    ///
    /// RECOVERED, not read: the blob header carries one `F_L2DOM` flag and one domain count for
    /// the whole blob (`reserved[2]`), but placement is decided per PROGRAM — `Builder::finish`
    /// declines it for a multi-wave-class program, so a blob can hold a placed decode program
    /// beside an unplaced, segmented prefill one. A program is placed iff the blob says placement
    /// happened and this program's window count equals the domain count.
    ///
    /// That test is exact rather than a heuristic, and the reason is a parity argument worth
    /// stating: a wave-class window count is `2*layers + 1`, which is always ODD, while a domain
    /// count is a hardware L2 partition count, which is a power of two and so always EVEN. The
    /// two ranges cannot collide.
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
    /// Model hidden size (`XReduce.i[0]`, decode's `t·hidden` at `t == 1`).
    /// Sizes the all-reduce message and hence the peer region.
    pub hidden: u32,
    /// Byte offset of partial slot B within the peer region — `max(i[2])` over
    /// the program's collectives, since slot A carries `i[2] == 0`.
    ///
    /// `devgen` computes it as `rows_max·hidden·2` where `rows_max` is the
    /// LARGEST prefill chunk, and bakes that same value into every bucket AND
    /// the decode program. So decode's copy is authoritative for the whole
    /// blob, which is why scanning one program is enough.
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
/// * `hidden` — `i[0]` of a ONE-SHOT [`DevOp::XReduce`] only. Decode compiles at
///   `t == 1`, so there `i[0] == hidden`; prefill's two-shot carries
///   `t·hidden`, and reading hidden from it would be wrong by the chunk size.
/// * `slot_bytes` — `max(i[2])`, because slot A carries 0 and slot B carries the
///   offset. `devgen` derives it from the LARGEST prefill chunk and bakes the
///   same value into every program, so the max over all of them is that one
///   value and not a per-program quantity.
fn recover_tp(progs: &[DevProg]) -> Option<DevTp> {
    let (mut n_gpu, mut hidden, mut slot_bytes) = (0u32, 0u32, 0u64);
    for p in progs {
        for d in &p.insts {
            let one_shot = d.op == DevOp::XReduce as u16;
            if !one_shot && d.op != DevOp::XReduceTwoShot as u16 {
                continue;
            }
            n_gpu = n_gpu.max(d.i[1]);
            if one_shot {
                // PER ROW, not per message. `i[0]` is `t * width`, and this used to read it
                // as the width outright — correct only while the one-shot was DECODE's alone
                // (`emit_xreduce` picked the two-shot for every t > 1) and t was therefore
                // always 1. Kimi-K3's shared-expert reduce carries a folded all-gather, which
                // the two-shot's decomposition cannot express, so it stays one-shot on BOTH
                // phases — and the widest prefill bucket then reported `hidden` as
                // `8192 * 7168`. Dividing by the program's own `t` is right for every case
                // and needs no phase test: decode divides by 1.
                //
                // The max over the result is what picks `hidden` out of a model with narrower
                // collectives — K3 also reduces the expert combine at its 3584-wide LATENT.
                hidden = hidden.max(d.i[0] / p.t.max(1));
            }
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
    /// this with `false` -- placement is then refused unless the operator opts in by env.
    pub fn parse_l2(buf: &[u8], l2_dispatch_ok: bool) -> Result<DevBlob> {
        Self::parse_inner(buf, l2_dispatch_ok)
    }

    pub fn parse(buf: &[u8]) -> Result<DevBlob> {
        Self::parse_inner(buf, false)
    }

    fn parse_inner(buf: &[u8], l2_dispatch_ok: bool) -> Result<DevBlob> {
        let mut off = 0usize;
        let hdr: BlobHeader = take::<BlobHeader>(buf, &mut off, 1, "header")?[0];
        if !is_blob_magic(&hdr.magic) {
            return Err(RuntimeError::Device(
                "devblob: bad magic — recompile with plowc (format changed)".into(),
            ));
        }
        let is_v7 = &hdr.magic == BLOB_MAGIC_V7;

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
                t: ph.t,
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
                // Which of these programs is L2-PLACED. See `DevProg::l2_domains` for why the
                // window count identifies it exactly.
                let l2_dom = hdr.reserved[2] as u32;
                if hdr.flags & packet::devbuild::PLOW_BLOB_F_L2DOM != 0
                    && l2_dom != 0
                    && n_seg == l2_dom as usize
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
                        "devblob: gen recipe for `{}` has unknown kind {} — this blob \
                         needs a newer plowrt",
                        tensors[r.tensor as usize].name, r.kind
                    )));
                }
            }
            g
        } else {
            Vec::new()
        };

        // PLOW_L2_PLACE guard: a blob whose gq `seg` is an L2 domain (F_L2DOM)
        // will be MIS-dispatched by a wave-class / static interp (it reads `seg`
        // as a wave-class segment). Refuse it unless this runtime opts into
        // physical-SM domain dispatch. (reserved[1]/[2] carry SMs/partition and
        // the domain count for that dispatch.) See the design notes
        //
        // `PLOW_NV_PLACE_DISPATCH` stays accepted alongside the new spelling: the flag was
        // renamed because an L2 domain is a GPC on NVIDIA and an XCD on AMD, and a run that
        // opted in under the old name must not start failing to load.
        // `--l2-place-dispatch` counts too: declaring the flag and then reading
        // only the environment makes it parse and do nothing.
        let dispatch_on = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
        if hdr.flags & packet::devbuild::PLOW_BLOB_F_L2DOM != 0
            && !l2_dispatch_ok
            && !crate::config::RuntimeConfig::get().nv.l2_place_dispatch
            && !dispatch_on("PLOW_L2_PLACE_DISPATCH")
            && !dispatch_on("PLOW_NV_PLACE_DISPATCH")
        {
            return Err(RuntimeError::Device(
                "devblob: blob uses L2-domain packet placement (PLOW_L2_PLACE) — its \
                 global-queue `seg` is an L2 domain, not a wave-class, so a standard interp \
                 would mis-dispatch it. Build the cubins with -DPLOW_L2_PLACE_DISPATCH and set \
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
        })
    }

    /// Get the raw bytes of a section by kind, sliced from the original buffer.
    /// Returns `None` if the section is not present.
    /// Get a section by kind and architecture-specific name.
    pub fn section_data_named<'a>(&self, buf: &'a [u8], kind: u32, name: &str) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|s| s.kind == kind && s.name == name)
            .and_then(|s| buf.get(s.offset..s.offset + s.size))
    }

    pub fn section_data<'a>(&self, buf: &'a [u8], kind: u32) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|s| s.kind == kind)
            .and_then(|s| buf.get(s.offset..s.offset + s.size))
    }

    /// The decode program: last, `t == B` (the compiler's contract; `plowc
    /// gemma4` emits the decode program last with `t` = `PLOW_DECODE_BATCH`,
    /// 1 by default, capped at 32). Prefill buckets all have `t >= 128`, so a
    /// small `t` unambiguously identifies the decode program.
    pub fn decode_prog(&self) -> Result<&DevProg> {
        let g = self
            .progs
            .last()
            .ok_or_else(|| RuntimeError::Device("devblob: no programs".into()))?;
        if g.t == 0 || g.t > 32 {
            return Err(RuntimeError::Device(format!(
                "devblob: last program has T={} — not the decode program (batch 1..=32)",
                g.t
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

impl DevProg {
    /// The coarse single-segment gate the sm_120 interpreter implements: every
    /// stream entry must be unsegmented (`seg == 0`) with no per-slice or
    /// cross-GPU counters. Mirrors the harness's fatal check.
    /// Per-segment wave class of a wave-class segmented program: 8 = GEMM-class,
    /// 4 = flash-class (contains a FlashPrefill op). Mirror of the AMD engine's
    /// `derive_segments`, hoisted here so the CUDA engine (which builds without
    /// the `hsa` module) can classify segments for the SegPf launcher.
    pub fn seg_classes(&self) -> Result<Vec<u8>> {
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
        // PLOW_PF_SEG_PURE=1: mirror of the emit-side PLOW_SEG_PURE_GEMM classing — a segment
        // is GEMM-class (8) only if EVERY op in it is a GEMM-family op; anything else (norms,
        // rope, quant, glu, flash) makes it flash-class (4). Required when the lean object is
        // built PLOW_NV_GEMM_ONLY: its dispatch traps on any non-GEMM opcode, so a light-op
        // segment classified 8 would land there. The two envs must be set together — this one
        // at serve time, the emit one at plowc time.
        // Must match the emit-side classing in devbuild.rs: "1" = every plain tiled GEMM,
        // "fp8" = only TMA-mapped fp8 GEMMs (the ws-entry object's sole arm).
        let rt = crate::config::RuntimeConfig::get();
        let pure_mode = match rt.nv.pf_seg_pure.as_deref() {
            Some("1") => 1u8,
            Some("fp8") => 2u8,
            _ => 0u8,
        };
        use packet::dev::DevOp;
        // PLOW_PF_SEG_FA512=1 (T12): hd512 FlashPrefill segments class 2 — launched on the
        // dedicated *_pffa object. Mirror of the emit-side PLOW_SEG_FA512.
        let fa512_mode = match rt.nv.pf_seg_fa512.as_deref() {
            Some("1") => 1u8,
            Some("all") => 2u8,
            _ => 0u8,
        };
        let v2_env = rt.nv.pf_seg_v2.as_deref();
        let seg_v2 = v2_env == Some("1");
        let seg_q8 = seg_v2 || v2_env == Some("q8");
        const FP8_OPS: [DevOp; 3] = [DevOp::GemmFp8, DevOp::GemmMedFp8, DevOp::GemmSmallFp8];
        const BF16_OPS: [DevOp; 3] = [DevOp::Gemm, DevOp::GemmSmall, DevOp::GemmMed];
        let mut class = vec![8u8; n_seg as usize];
        for e in &self.stream {
            let inst = self.insts.get(e.inst as usize).ok_or_else(|| {
                RuntimeError::Device(format!(
                    "stream entry references instruction {} of {}",
                    e.inst,
                    self.insts.len()
                ))
            })?;
            let op = inst.op;
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
        Ok(class)
    }

    pub fn check_coarse_single_segment(&self) -> Result<()> {
        // An L2-PLACED program (l2_domains != 0) legitimately carries seg = L2 domain in
        // [0, l2_domains); the placed interpreter partitions by per-domain gq windows and
        // never reads seg as a wave-class. The parse-time PLOW_BLOB_F_L2DOM gate has already
        // verified the runtime attested a placed cubin, so only out-of-range domains and the
        // still-unimplemented fine/xctr flags are errors here.
        let seg_lim = if self.l2_domains != 0 {
            self.l2_domains as u16
        } else {
            1
        };
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

    /// Decode's one-shot is where `hidden` comes from: at `t == 1`, `i[0]` IS
    /// the hidden size. `slot_bytes` is `max(i[2])` because slot A carries 0.
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

    /// PREFILL's two-shot carries `i[0] = t·hidden`, so reading `hidden` from it
    /// would be wrong by the chunk size — 1024x here. Only the one-shot may
    /// supply it, and `slot_bytes` still comes from the max across BOTH.
    #[test]
    fn prefills_two_shot_never_supplies_hidden() {
        let h = 5376u32;
        let slot_b = 1024 * h * 2; // devgen: rows_max * hidden * 2
        let mut m = tiny_model();
        let dp = m.progs.len() - 1;
        with_xreduce(&mut m, 0, false, 4, 1024 * h, slot_b); // prefill bucket
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
        with_xreduce(&mut pf, 0, false, 2, 1024 * h, slot_b);
        let only = DevBlob::parse(&pf.to_blob()).unwrap().tp.expect("sharded");
        assert_eq!((only.n_gpu, only.slot_bytes), (2, slot_b as u64));
        assert_eq!(
            only.hidden, 0,
            "unrecoverable from two-shot alone, not guessed"
        );
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
    fn roundtrip_through_the_real_writer() {
        let blob = tiny_model().to_blob();
        let b = DevBlob::parse(&blob).unwrap();
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
