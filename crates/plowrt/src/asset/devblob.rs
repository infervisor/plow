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

use packet::dev::{DevInst64, StreamEnt, Wait};
use packet::rope::GenTensor;
use packet::devbuild::{
    is_blob_magic, BlobHeader, BlobProgHeader, BlobSectionEntry, BlobTensor, BLOB_MAGIC_V7,
    INIT_NONE, NAME_LEN, SECT_GEN_TENSORS, SECT_MAGIC, SECT_NAME_LEN,
};

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
        v.push(unsafe {
            std::ptr::read_unaligned(buf[*off + i * sz..].as_ptr() as *const T)
        });
    }
    *off = end;
    Ok(v)
}

impl DevBlob {
    /// Parse a blob image. Fails loudly on a bad magic or a truncated section,
    /// never mid-serve.
    pub fn parse(buf: &[u8]) -> Result<DevBlob> {
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
            let ph: BlobProgHeader =
                take::<BlobProgHeader>(buf, &mut off, 1, "prog header")?[0];
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
            let n = u32::from_le_bytes(
                buf[dir_off + 4..dir_off + 8].try_into().unwrap(),
            ) as usize;
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
                let ent = unsafe {
                    std::ptr::read_unaligned(buf[base..].as_ptr() as *const BlobSectionEntry)
                };
                let name_len = ent.name.iter().position(|&b| b == 0).unwrap_or(SECT_NAME_LEN);
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
                    RuntimeError::Device(
                        "devblob: v7 blob has no SECT_GEN_TENSORS section".into(),
                    )
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

        // PLOW_NV_PLACE guard: a blob whose gq `seg` is an L2 domain (F_L2DOM)
        // will be MIS-dispatched by a wave-class / static interp (it reads `seg`
        // as a wave-class segment). Refuse it unless this runtime opts into
        // physical-SM domain dispatch. (reserved[1]/[2] carry SMs/partition and
        // the domain count for that dispatch.) See devblob-locality-placement.md.
        if hdr.flags & packet::devbuild::PLOW_BLOB_F_L2DOM != 0
            && std::env::var("PLOW_NV_PLACE_DISPATCH").ok().as_deref() != Some("1")
        {
            return Err(RuntimeError::Device(
                "devblob: blob uses L2-domain packet placement (PLOW_NV_PLACE) — its \
                 global-queue `seg` is an L2 domain, not a wave-class, so a standard interp \
                 would mis-dispatch it. Build the cubins with -DPLOW_NV_PLACE_DISPATCH and set \
                 PLOW_NV_PLACE_DISPATCH=1, or recompile the model without PLOW_NV_PLACE."
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
        })
    }

    /// Get the raw bytes of a section by kind, sliced from the original buffer.
    /// Returns `None` if the section is not present.
    /// Get a section by kind and architecture-specific name.
    pub fn section_data_named<'a>(
        &self,
        buf: &'a [u8],
        kind: u32,
        name: &str,
    ) -> Option<&'a [u8]> {
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
    pub fn check_coarse_single_segment(&self) -> Result<()> {
        for (j, e) in self.stream.iter().enumerate() {
            if e.seg != 0 || (e.flags & (packet::dev::SE_FINE | packet::dev::SE_XCTR)) != 0 {
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

    /// A tiny two-program model exercised through the REAL writer
    /// (`Model::to_blob`) — reader and writer cannot drift apart unnoticed.
    fn tiny_model() -> Model {
        let inst = |op: u16, wait_len: u16, succ_len: u16, wait_ofs: u32, succ_ofs: u32| {
            DevInst {
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
            }
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
            insts: vec![inst(6, 0, 1, 0, 0), inst(18, 1, 0, 0, 0)],
            stream: vec![se(0, 0), se(1, 0), se(0, 1), se(1, 1)],
            stream_ofs: vec![0, 2],
            stream_len: vec![2, 2],
            waits: vec![Wait { id: 0, threshold: 1 }],
            succs: vec![0],
            tensors: Vec::new(),
            gq_stream: vec![se(0, 0), se(0, 1), se(1, 0), se(1, 1)],
            gq_seg_ofs: vec![0, 4],
            // Unplaced: `seg` is a wave-class, not an L2 domain (PLOW_NV_PLACE).
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
        assert_eq!(&image[..8], BLOB_MAGIC_V7, "a model with recipes must be v7");

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
        let by = |i: u32| b.gen.iter().find(|g| g.tensor == i).unwrap().generate().unwrap();
        assert_eq!(by(2), want_cos);
        assert_eq!(by(3), want_sin);
        assert_eq!(by(2).len() as u64, b.tensors[2].bytes, "decl size matches recipe");
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
        assert_eq!(&image[off..off + 4], &2u32.to_le_bytes(), "recipe 0 targets tensor 2");
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
        assert_eq!(&b.init[b.tensors[1].init.clone().unwrap()], &[1, 2, 3, 4, 5, 6, 7, 8]);
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
