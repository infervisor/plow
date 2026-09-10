//! NVIDIA CUDA execution for Hopper and Blackwell.
//!
//! Decode object selection, prefix caching and token-batch execution live in
//! dedicated modules. This engine owns device allocations, stream completion
//! and the serving slots shared by those paths.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use packet::dev::{DevInst64, DevOp, DevProgram, CTR_STRIDE, TENSOR_NONE16};

use crate::config::RuntimeConfig;

/// PX-1 packing measurement. Reads `RuntimeConfig::get().pf_packlog`.
fn pf_packlog_on() -> bool {
    RuntimeConfig::get().pf_packlog
}

/// Covering bucket-pick policy. Reads `RuntimeConfig::get().nv.pf_cover`.
fn pf_cover_on() -> bool {
    RuntimeConfig::get().nv.pf_cover
}

/// Fixed cost of ONE prefill launch, in padded-row equivalents.
/// Reads `RuntimeConfig::get().nv.pf_chunk_cost`.
fn pf_chunk_cost_rows() -> usize {
    RuntimeConfig::get().nv.pf_chunk_cost
}

use crate::asset::devblob::DevBlob;
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::memory::slab_pad;
#[cfg(test)]
use crate::memory::SLAB_ALIGN;
use crate::{Result, RuntimeError};
use plow_asset::cubin::{self, Role};

fn check_norm_weight_offset(be: &CudaBackend, module: &Module, blob: &DevBlob) -> Result<()> {
    let expected = u32::from(
        blob.progs
            .iter()
            .flat_map(|p| &p.insts)
            .any(|d| d.op == DevOp::RmsNorm as u16 && d.i[7] == 1),
    );
    let actual = be
        .module_global_u32(module, "plow_norm_weight_offset")?
        .unwrap_or(0);
    if actual != expected {
        return Err(RuntimeError::Rejected(format!(
            "norm weight offset mismatch: packet requires {expected}, object provides {actual}"
        )));
    }
    Ok(())
}

mod prefix;
use prefix::VmmServe;

mod decode_context;
mod decode_object;
use decode_object::{BoundDecodeObject, DecodeModule};
mod decode_rung;
mod mixed_step;
mod native_decode;
mod packed_terminal;
mod token_batch;
use cublaslt::CublasLtDecodeRoute;
use decode_rung::{
    decode_rung_index, decode_selection, effective_decode_widths, validate_cublaslt_ladder,
    validate_decode_ladder, DecodeRung, DecodeSelection,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InterpreterProfile {
    tag: &'static str,
    decode_file: &'static str,
    prefill_file: &'static str,
    decode_symbol: &'static str,
    prefill_symbol: &'static str,
    embedded_decode: &'static str,
}

fn interpreter_profile(cc: (u32, u32)) -> Option<InterpreterProfile> {
    match cc {
        (9, 0) => Some(InterpreterProfile {
            tag: "sm90a",
            decode_file: "interp_sm90a.cubin",
            prefill_file: "interp_sm90a_pf.cubin",
            decode_symbol: "_Z12interp_sm90a11PlowProgram",
            prefill_symbol: "_Z15interp_sm90a_pf11PlowProgram",
            embedded_decode: "interp_sm90a",
        }),
        (12, 0) => Some(InterpreterProfile {
            tag: "sm120",
            decode_file: "interp_sm120.cubin",
            prefill_file: "interp_sm120_pf.cubin",
            decode_symbol: "_Z12interp_sm12011PlowProgram",
            prefill_symbol: "_Z15interp_sm120_pf11PlowProgram",
            embedded_decode: "interp_sm120",
        }),
        _ => None,
    }
}

impl InterpreterProfile {
    /// The file name this arch's build scripts write for `role`. Tried FIRST
    /// when scanning the assets dir — and never trusted, because the image
    /// itself says what it is (see [`resolve_interp_image`]).
    fn file(&self, role: Role) -> &'static str {
        match role {
            Role::Decode => self.decode_file,
            Role::Prefill => self.prefill_file,
        }
    }

    /// The entry symbol this arch's objects are EXPECTED to carry. Only a
    /// fallback for an image with no readable symbol table; the discovered
    /// symbol wins.
    fn symbol(&self, role: Role) -> &'static str {
        match role {
            Role::Decode => self.decode_symbol,
            Role::Prefill => self.prefill_symbol,
        }
    }
}

/// `PLOW_NV_CUBIN{,_PF}` — force one image, bypassing discovery.
fn image_override_name(role: Role) -> &'static str {
    match role {
        Role::Decode => "PLOW_NV_CUBIN",
        Role::Prefill => "PLOW_NV_CUBIN_PF",
    }
}

/// A resolved interpreter object: its bytes, the entry symbol IT declares, and
/// a human-readable provenance for the log line.
struct InterpImage {
    image: Vec<u8>,
    entry: String,
    source: String,
}

/// Nothing in an assets dir that an interpreter object could plausibly be is
/// anywhere near this large; the cap keeps a stray checkpoint shard sharing the
/// directory from being slurped into memory during discovery.
const CUBIN_SCAN_MAX: u64 = 64 << 20;

/// Read a candidate only if it is small enough and starts with ELF64 magic —
/// the two cheap tests that keep discovery off `model.pkt` and the weights.
fn read_cubin_candidate(path: &Path) -> Option<Vec<u8>> {
    let md = std::fs::metadata(path).ok()?;
    if !md.is_file() || md.len() > CUBIN_SCAN_MAX {
        return None;
    }
    let image = std::fs::read(path).ok()?;
    cubin::is_elf64_le(&image).then_some(image)
}

fn interp_candidate(
    image: &[u8],
    profile: &InterpreterProfile,
    want_sm: u32,
    role: Role,
) -> std::result::Result<String, String> {
    let Some(info) = cubin::inspect(image) else {
        return Err("not an ELF cubin".into());
    };
    if cubin::global_u32(image, plow_asset::mixed_step::OBJECT_CAPABILITY).is_some() {
        return Err("mixed-step auxiliary object is not an ordinary interpreter".into());
    }
    if cubin::global_u32(image, plow_asset::packed_prefill::CAPABILITY).is_some() {
        return Err("packed-prefill auxiliary object is not an ordinary interpreter".into());
    }
    if info.sm != want_sm {
        return Err(format!("built for sm_{}, device is sm_{want_sm}", info.sm));
    }
    match info.interp_entry(role) {
        Some(sym) => Ok(sym.to_string()),
        None if info.entries.is_empty() => Ok(profile.symbol(role).to_string()),
        None => Err(format!(
            "no {} entry — has {}",
            role.as_str(),
            cubin::describe(image)
        )),
    }
}

fn embedded_interp_image(
    blob: &DevBlob,
    raw: &[u8],
    profile: &InterpreterProfile,
    want_sm: u32,
    role: Role,
    rejected: &mut Vec<String>,
) -> Option<InterpImage> {
    for s in blob
        .sections
        .iter()
        .filter(|s| s.kind == packet::devbuild::SECT_CUBIN)
    {
        let Some(data) = raw.get(s.offset..s.offset + s.size) else {
            continue;
        };
        match interp_candidate(data, profile, want_sm, role) {
            Ok(entry) => {
                return Some(InterpImage {
                    image: data.to_vec(),
                    entry,
                    source: format!("embedded section '{}'", s.name),
                })
            }
            Err(why) => rejected.push(format!("embedded '{}': {why}", s.name)),
        }
    }
    None
}

fn filesystem_interp_image(
    assets_dir: &Path,
    profile: &InterpreterProfile,
    want_sm: u32,
    role: Role,
    rejected: &mut Vec<String>,
) -> Option<InterpImage> {
    let mut paths = vec![assets_dir.join(profile.file(role))];
    if let Ok(rd) = std::fs::read_dir(assets_dir) {
        let mut rest: Vec<std::path::PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p != &paths[0])
            .collect();
        rest.sort();
        paths.extend(rest);
    }
    for path in paths {
        let Some(image) = read_cubin_candidate(&path) else {
            continue;
        };
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        match interp_candidate(&image, profile, want_sm, role) {
            Ok(entry) => {
                return Some(InterpImage {
                    image,
                    entry,
                    source: name,
                })
            }
            Err(why) => rejected.push(format!("{name}: {why}")),
        }
    }
    None
}

/// Find the interpreter object for `role` BY CONTENT.
///
/// A cubin names its own SM (`e_flags`) and its own entry points (`.symtab`),
/// so the loader can select the right image out of a bundle instead of trusting
/// file names — which are the one part of the artifact nothing validates. The
/// decode/prefill pair is trivially swappable (`build_sm90a_cubin.sh` derives
/// the prefill path from the decode one, so an argument without the `.cubin`
/// suffix writes decode to `<x>` and prefill to `<x>_pf.cubin`), and the
/// symptom of getting it wrong used to be
/// `cuModuleGetFunction(_Z12interp_sm90a11PlowProgram): CUDA_ERROR_NOT_FOUND`
/// with no indication of what the loaded image actually was.
///
/// Priority: `PLOW_NV_CUBIN{,_PF}` (forced — an override that does not match is
/// an error, never a silent fallback) → embedded `SECT_CUBIN` sections → the
/// assets dir, profile file name first. Candidates whose SM disagrees with the
/// live device are rejected here, where the message can say so, rather than in
/// the driver as `CUDA_ERROR_INVALID_IMAGE` or worse.
///
/// `Ok(None)` = no candidate anywhere carries this role. That is fatal for
/// decode and a documented decode-only fallback for prefill, so the decision
/// belongs to the caller.
fn resolve_interp_image(
    assets_dir: &Path,
    blob: &DevBlob,
    raw: &[u8],
    profile: &InterpreterProfile,
    want_sm: u32,
    role: Role,
) -> Result<Option<InterpImage>> {
    // Accept an image iff it is a cubin for THIS device carrying THIS role.
    // `None` from `inspect` means unparseable, not "no entry" — a hand-built
    // object with a stripped symbol table still loads under the profile's
    // expected symbol, which is what the pre-discovery loader always did.
    let judge = |image: &[u8]| interp_candidate(image, profile, want_sm, role);

    // 1. Operator override: forced, and loud when wrong. The CLI flag is read
    // as well as the env var — declaring `--nv-cubin` and then consulting only
    // the environment makes the flag parse and do nothing.
    let var = image_override_name(role);
    let cfg_image = {
        let nv = &crate::config::RuntimeConfig::get().nv;
        match role {
            Role::Decode => nv.cubin.clone(),
            Role::Prefill => nv.cubin_pf.clone(),
        }
    };
    if let Some(p) = cfg_image {
        let path = std::path::PathBuf::from(&p);
        let image = std::fs::read(&path).map_err(|source| RuntimeError::Io {
            path: path.clone(),
            source,
        })?;
        let entry =
            judge(&image).map_err(|why| RuntimeError::Device(format!("{var}={p}: {why}")))?;
        return Ok(Some(InterpImage {
            image,
            entry,
            source: format!("{var}={p}"),
        }));
    }

    let mut rejected: Vec<String> = Vec::new();

    // 2. Embedded sections. The section NAME is not load-bearing: `plowc
    //    --embed-cubin` labels every image `interp_sm120` regardless of the arch
    //    it just compiled, so only the content can decide.
    if let Some(image) = embedded_interp_image(blob, raw, profile, want_sm, role, &mut rejected) {
        return Ok(Some(image));
    }

    // 3. The assets dir — the profile's expected name first, then everything
    //    else that looks like a cubin, so a misnamed bundle still serves.
    if let Some(image) = filesystem_interp_image(assets_dir, profile, want_sm, role, &mut rejected)
    {
        return Ok(Some(image));
    }

    if !rejected.is_empty() {
        tracing::debug!(
            role = role.as_str(),
            rejected = rejected.join("; "),
            "no interpreter object matched"
        );
    }
    Ok(None)
}

/// Every cubin-shaped file in the assets dir and what it really contains — the
/// body of the "no object found" errors, so an operator sees the mismatch
/// (wrong arch, swapped pair, stale bundle) instead of just a missing name.
fn describe_candidates(assets_dir: &Path) -> String {
    let Ok(rd) = std::fs::read_dir(assets_dir) else {
        return "  (assets dir unreadable)".into();
    };
    let mut paths: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    let lines: Vec<String> = paths
        .iter()
        .filter_map(|p| {
            let image = read_cubin_candidate(p)?;
            Some(format!(
                "  {} — {}",
                p.file_name().unwrap_or_default().to_string_lossy(),
                cubin::describe(&image)
            ))
        })
        .collect();
    if lines.is_empty() {
        "  (no cubin in the assets dir)".into()
    } else {
        lines.join("\n")
    }
}

/// Model-load timeline. Reads `RuntimeConfig::get().load_profile`.
fn load_profile() -> bool {
    RuntimeConfig::get().load_profile
}

/// The shared storage-source enum (`Vmm` is the CUDA default —
/// `PLOW_WEIGHT_VMM=0` drops to `Flat`, `PLOW_WEIGHT_SLAB=0` to `PerTensor`;
/// see the `_weight_slab` field doc for the measurements) and its commit
/// chunk.
use crate::memory::vmm::{WeightSlab, WEIGHT_SLAB_CHUNK};

/// Block until a VMM-slab byte range ending at device address `dend` is
/// committed (no-op for every other storage kind — VMM has no demand paging,
/// so a DMA or memset below the mapper watermark must wait, not fault).
/// Accumulates blocked time into `acc` (ms).
fn slab_commit_wait(ws: &WeightSlab, dend: u64, acc: &mut f64) -> Result<()> {
    if let WeightSlab::Vmm(s) = ws {
        if dend > s.base() && dend - s.base() <= s.len() {
            let t = std::time::Instant::now();
            s.wait_mapped(dend - s.base())?;
            *acc += t.elapsed().as_secs_f64() * 1e3;
        }
    }
    Ok(())
}

/// One block per executor, 8 worker warps — `dev_isa.h` workgroup geometry.
const BLOCK: u32 = 256;
/// Weight-upload staging chunk (pinned), as in the harness.
const STAGE: usize = 64 << 20;
/// Prefill dynamic-smem default: the union arena is the hd=256 flash tile,
/// 21312 floats = 83.25 KiB (`interp_sm120.cu` `PLOW_NV_PRE_A`, grown by the
/// T4 mma P·V Ps/m/l/corr staging). `PLOW_NV_SMEM_PF` overrides. Opt-in past
/// 48 KiB, so the launch always sets the attribute. Under-provisioning fails
/// the launch loudly (never silent), so a stale value here cannot corrupt.
const SMEM_PF: u32 = 21312 * 4;



use crate::asset::checkpoint::Checkpoint;

/// One uploaded prefill bucket (`t == chunk size`), run under the `_pf`
/// object. Its instruction stream is host-patched per chunk (KV write row,
/// flash `seq_kv`/`q_pos0`, lm_head `a_row0`) then the covering range is
/// re-uploaded — the port of the harness's chunked-prefill inner loop.
/// Segmented-prefill object pair (T9c, first wired on Hopper): flash-class (wave-class-4)
/// segments launch on the fat `_pfseg` object (every arm, occ-1); GEMM-class segments on
/// the lean `_pfgemm` object (flash arms compiled out, 128 regs, occ-2). Segments of one
/// chunk launch SEQUENTIALLY on the engine stream — dependencies only point backward, so
/// stream order replaces cross-segment gating; counters are re-armed once per chunk.
struct SegGemm {
    abi: u32,
    function: KernelFn,
    smem: u32,
    grid: u32,
    block: u32,
    _module: Module,
}

use plow_asset::segment_roles::{ProgramRoles, SegmentObject, SegmentRoles};

fn validate_segment_windows(g: &crate::asset::devblob::DevProg) -> Result<()> {
    if g.l2_domains != 0
        || g.gq_seg_ofs.len() < 2
        || g.gq_seg_ofs.first() != Some(&0)
        || g.gq_seg_ofs.last().copied() != Some(g.gq_stream.len() as u32)
    {
        return Err(RuntimeError::Rejected(
            "invalid packet segment windows".into(),
        ));
    }
    for (seg, w) in g.gq_seg_ofs.windows(2).enumerate() {
        let entries = g
            .gq_stream
            .get(w[0] as usize..w[1] as usize)
            .ok_or_else(|| RuntimeError::Rejected("invalid packet queue bounds".into()))?;
        if entries.is_empty()
            || entries
                .iter()
                .any(|e| e.seg as usize != seg || e.inst as usize >= g.insts.len())
        {
            return Err(RuntimeError::Rejected(
                "invalid packet queue membership".into(),
            ));
        }
    }
    let key = |e: &packet::dev::StreamEnt| {
        (
            e.inst, e.slice, e.wait_ofs, e.succ_ofs, e.wait_len, e.succ_len, e.flags, e.seg,
        )
    };
    let mut a: Vec<_> = g.stream.iter().map(key).collect();
    let mut b: Vec<_> = g.gq_stream.iter().map(key).collect();
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return Err(RuntimeError::Rejected(
            "packet queues do not cover the same work".into(),
        ));
    }
    for (ix, d) in g.insts.iter().enumerate() {
        let mut slices: Vec<_> = g
            .gq_stream
            .iter()
            .filter(|e| e.inst as usize == ix)
            .map(|e| e.slice)
            .collect();
        slices.sort_unstable();
        if slices != (0..u32::from(d.blocks)).collect::<Vec<_>>() {
            return Err(RuntimeError::Rejected(
                "packet queue has missing or duplicate instruction slices".into(),
            ));
        }
    }
    Ok(())
}

fn segment_role_metadata(blob: &DevBlob, raw: &[u8]) -> Result<Option<SegmentRoles>> {
    let Some(bytes) = blob.reserved_metadata(raw, plow_asset::segment_roles::SECTION)? else {
        return Ok(None);
    };
    SegmentRoles::parse(bytes, blob).map(Some)
}

fn prefill_needs_segment_pair(blob: &DevBlob, roles: Option<&SegmentRoles>) -> bool {
    blob.prefill_progs().iter().enumerate().any(|(index, g)| {
        g.check_coarse_single_segment().is_err()
            && roles.and_then(|roles| roles.program(index)).is_none()
    })
}

trait SegmentRoleValidation: Sized {
    fn parse(bytes: &[u8], blob: &DevBlob) -> Result<Self>;
    fn validate(
        &self,
        programs: &[crate::asset::devblob::DevProg],
        prefill: &[usize],
        tensors: &[crate::asset::devblob::DevTensor],
    ) -> Result<()>;
    fn program(&self, index: usize) -> Option<&ProgramRoles>;
}
impl SegmentRoleValidation for SegmentRoles {
    fn parse(bytes: &[u8], blob: &DevBlob) -> Result<Self> {
        let value = Self::from_bytes(bytes)
            .map_err(|e| RuntimeError::Rejected(format!("segment_roles.json: {e}")))?;
        let indices: Vec<_> = blob
            .progs
            .iter()
            .enumerate()
            .filter(|(_, g)| blob.prefill_progs().iter().any(|p| std::ptr::eq(p, *g)))
            .map(|(i, _)| i)
            .collect();
        value.validate(&blob.progs, &indices, &blob.tensors)?;
        for program in &value.programs {
            let fp8_pcs: Vec<_> = program
                .roles
                .iter()
                .enumerate()
                .filter(|(_, role)| **role == plow_asset::segment_roles::FP8_M1)
                .map(|(seg, _)| {
                    let g = &blob.progs[program.index];
                    g.gq_stream[g.gq_seg_ofs[seg] as usize].inst as usize
                })
                .collect();
            if !fp8_pcs.is_empty() {
                blob.with_packet_view(|p| {
                    plow_asset::fp8_m1_role::validate_all(p, program.index, &fp8_pcs)
                })
                .map_err(RuntimeError::Rejected)?;
            }
            for (seg, &role) in program.roles.iter().enumerate() {
                if role == plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32 {
                    let g = &blob.progs[program.index];
                    let pc = g.gq_stream[g.gq_seg_ofs[seg] as usize].inst as usize;
                    if u32::from(g.insts[pc].blocks) != blob.n_cu {
                        return Err(RuntimeError::Rejected(
                            "HD512 WG32 role requires one slice per packet block".into(),
                        ));
                    }
                }
            }
        }
        Ok(value)
    }
    fn validate(
        &self,
        programs: &[crate::asset::devblob::DevProg],
        prefill: &[usize],
        tensors: &[crate::asset::devblob::DevTensor],
    ) -> Result<()> {
        self.validate_schema().map_err(RuntimeError::Rejected)?;
        let mut seen = std::collections::BTreeSet::new();
        let mut used = std::collections::BTreeSet::new();
        for p in &self.programs {
            let g = programs.get(p.index).ok_or_else(|| {
                RuntimeError::Rejected("packet role program index out of bounds".into())
            })?;
            let decode = !prefill.contains(&p.index);
            let object_decode = p.roles.iter().any(|&role| {
                matches!(
                    role,
                    plow_asset::segment_roles::GEMV_CTA512
                        | plow_asset::segment_roles::FP8_M1
                        | plow_asset::segment_roles::MXFP4_MOE
                )
            });
            let library_decode = p
                .roles
                .iter()
                .copied()
                .any(plow_asset::segment_roles::is_projection);
            if !seen.insert(p.index)
                || (p.roles.contains(&plow_asset::segment_roles::FP8_M1)
                    && p.roles.contains(&plow_asset::segment_roles::GEMV_CTA512))
                || (object_decode && library_decode)
                || (decode
                    && ((!library_decode
                        && (p.index + 1 != programs.len()
                            || programs.len() != prefill.len() + 1))
                        || (!object_decode && !library_decode)
                        || (object_decode && g.t != 1)
                        || p.roles.iter().any(|&role| {
                            !matches!(
                                role,
                                plow_asset::segment_roles::INTERPRETER
                                    | plow_asset::segment_roles::GEMV_CTA512
                                    | plow_asset::segment_roles::FP8_M1
                                    | plow_asset::segment_roles::MXFP4_MOE
                                    | plow_asset::segment_roles::CUBLASLT
                                    | plow_asset::segment_roles::NATIVE_DECODE_TC
                            )
                        })))
                || (!decode
                    && p.roles.iter().any(|&role| {
                        matches!(
                            role,
                            plow_asset::segment_roles::GEMV_CTA512
                                | plow_asset::segment_roles::FP8_M1
                                | plow_asset::segment_roles::CUBLASLT
                                | plow_asset::segment_roles::NATIVE_DECODE_TC
                        )
                    }))
            {
                return Err(RuntimeError::Rejected(
                    "invalid packet role program or decode rung".into(),
                ));
            }
            if decode
                && (g
                    .stream
                    .iter()
                    .any(|e| e.flags & (packet::dev::SE_FINE | packet::dev::SE_XCTR) != 0)
                    || !g.gq_stream.windows(2).all(|w| w[0].inst <= w[1].inst))
            {
                return Err(RuntimeError::Rejected(
                    "decode segment roles require coarse local counters".into(),
                ));
            }
            packet_role_segments(g, &p.roles, tensors)?;
            used.extend(
                p.roles
                    .iter()
                    .copied()
                    .filter(|&role| plow_asset::segment_roles::requires_object(role)),
            );
        }
        if used != self.objects.keys().copied().collect() {
            return Err(RuntimeError::Rejected(
                "packet role object declaration does not match use".into(),
            ));
        }
        Ok(())
    }
    fn program(&self, index: usize) -> Option<&ProgramRoles> {
        self.programs.iter().find(|p| p.index == index)
    }
}

struct PacketRole {
    function: KernelFn,
    grid: u32,
    smem: u32,
    block: u32,
    _module: Arc<DecodeModule>,
}

fn check_fp8_gemm_role(capability: Option<u32>, block: Option<u32>) -> Result<()> {
    if capability != Some(1) || block != Some(BLOCK) {
        return Err(RuntimeError::Rejected("incompatible FP8 GEMM role".into()));
    }
    Ok(())
}

fn check_attention_role(arch: &str, capability: Option<u32>, block: Option<u32>) -> Result<()> {
    if arch != "sm90a" || capability != Some(1) || block != Some(BLOCK) {
        return Err(RuntimeError::Rejected(
            "incompatible HD256 attention role".into(),
        ));
    }
    Ok(())
}

fn check_attention_hd512_role(
    arch: &str,
    object: &plow_asset::segment_roles::SegmentObject,
    capability: Option<u32>,
    block: Option<u32>,
    geometry: [Option<u32>; 4],
) -> Result<()> {
    let expected = object.attention.as_ref();
    if arch != "sm90a"
        || capability != Some(1)
        || block != Some(BLOCK)
        || expected.is_none_or(|a| {
            a.profile != arch
                || a.dtype != "bf16"
                || geometry
                    != [
                        Some(a.head_dim),
                        Some(a.query_tile),
                        Some(a.kv_tile),
                        Some(a.warps),
                    ]
        })
    {
        return Err(RuntimeError::Rejected(
            "incompatible HD512 WG32 attention role".into(),
        ));
    }
    Ok(())
}

fn validate_gemv_decode_role_inst(
    d: &DevInst64,
    rows: u32,
    slices: usize,
    tensors: &[crate::asset::devblob::DevTensor],
) -> Result<()> {
    let reject = || RuntimeError::Rejected("invalid BF16 M1 GEMV512 role geometry".into());
    let k = u64::from(d.i[2]);
    if rows != 1
        || d.i[0] != 1
        || k == 0
        || k > 32768
        || k % 8 != 0
        || usize::from(d.blocks) != slices
        || d.i[1] == 0
    {
        return Err(reject());
    }
    let extent = |slot: usize, elements: u64| -> Result<()> {
        if elements == 0 {
            return Ok(());
        }
        let bytes = elements.checked_mul(2).ok_or_else(reject)?;
        if tensors
            .get(d.t[slot] as usize)
            .is_none_or(|t| t.bytes < bytes)
        {
            return Err(reject());
        }
        Ok(())
    };
    let n = u64::from(d.i[1]);
    extent(0, n)?;
    extent(2, n * k)?;
    match DevOp::from_u16(d.op) {
        Some(DevOp::Gemv) if d.i[3] == 0 => {
            extent(1, (u64::from(d.i[4]) + 1) * k)?;
        }
        Some(DevOp::GemvQkv) => {
            extent(1, k)?;
            extent(3, u64::from(d.i[3]))?;
            extent(4, u64::from(d.i[3]) * k)?;
            extent(5, u64::from(d.i[4]))?;
            extent(6, u64::from(d.i[4]) * k)?;
            n.checked_add(u64::from(d.i[3]))
                .and_then(|n| n.checked_add(u64::from(d.i[4])))
                .filter(|&n| n <= u64::from(u32::MAX))
                .ok_or_else(reject)?;
        }
        Some(DevOp::GemvGlu) if d.i[5] <= 2 => {
            extent(1, k)?;
            extent(5, n * k)?;
        }
        _ => return Err(reject()),
    }
    Ok(())
}

fn validate_mxfp4_moe_role_inst(
    d: &DevInst64,
    rows: u32,
    slices: usize,
    tensors: &[crate::asset::devblob::DevTensor],
) -> Result<()> {
    let reject = || RuntimeError::Rejected("invalid MXFP4 MoE role geometry".into());
    if rows != 1
        || d.i[6] != rows
        || usize::from(d.blocks) != slices
        || d.i[0] == 0
        || d.i[0] > 16
        || d.i[3] == 0
    {
        return Err(reject());
    }
    let extent = |slot: usize, bytes: u64, optional: bool| -> Result<()> {
        if optional && d.t[slot] == TENSOR_NONE16 {
            return Ok(());
        }
        if d.t[slot] == TENSOR_NONE16
            || tensors
                .get(d.t[slot] as usize)
                .is_none_or(|tensor| tensor.bytes < bytes)
        {
            return Err(reject());
        }
        Ok(())
    };
    let checked = |values: &[u64]| {
        values
            .iter()
            .try_fold(1u64, |acc, &value| acc.checked_mul(value))
            .ok_or_else(reject)
    };
    let ksel = u64::from(d.i[0]);
    let experts = u64::from(d.i[3]);
    extent(2, ksel.checked_mul(8).ok_or_else(reject)?, false)?;
    match DevOp::from_u16(d.op) {
        Some(DevOp::MoeGluMx)
            if d.i[1] > 0
                && d.i[2] > 0
                && d.i[2] % 32 == 0
                && d.i[4] <= 1
                && d.i[5] <= 3
                && f32::from_bits(d.fj[0]).is_finite()
                && f32::from_bits(d.fj[1]).is_finite() =>
        {
            let inter = u64::from(d.i[1]);
            let hidden = u64::from(d.i[2]);
            extent(0, checked(&[ksel, inter, 2])?, false)?;
            extent(1, checked(&[hidden, 2])?, false)?;
            extent(
                3,
                checked(&[experts, 2 * inter, hidden.div_ceil(2)])?,
                false,
            )?;
            extent(
                4,
                checked(&[experts, 2 * inter, hidden.div_ceil(32)])?,
                false,
            )?;
            extent(5, checked(&[experts, 2 * inter, 2])?, true)?;
        }
        Some(DevOp::MoeDownMx) if d.i[1] > 0 && d.i[2] > 0 && d.i[2] % 32 == 0 => {
            let hidden = u64::from(d.i[1]);
            let inter = u64::from(d.i[2]);
            extent(0, checked(&[ksel, hidden, 4])?, false)?;
            extent(1, checked(&[ksel, inter, 2])?, false)?;
            extent(3, checked(&[experts, hidden, inter.div_ceil(2)])?, false)?;
            extent(4, checked(&[experts, hidden, inter.div_ceil(32)])?, false)?;
            extent(5, checked(&[experts, hidden, 2])?, true)?;
        }
        _ => return Err(reject()),
    }
    Ok(())
}

fn validate_attention_role_inst(
    d: &DevInst64,
    rows: u32,
    tensors: &[crate::asset::devblob::DevTensor],
    hd_required: u32,
    require_tma_map: bool,
    allow_fused: bool,
) -> Result<()> {
    let reject =
        || RuntimeError::Rejected("unsupported attention role operands or geometry".into());
    let prefill = d.op == DevOp::FlashPrefill as u16;
    let (heads, splits, hd) = if prefill {
        (d.i[2], d.i[7], d.i[6])
    } else if d.op == DevOp::FlashMerge as u16 {
        (d.i[1], d.i[2], d.i[3])
    } else {
        return Err(reject());
    };
    if rows == 0 || d.i[0] != rows || heads == 0 || splits == 0 || hd != hd_required {
        return Err(reject());
    }
    let work = rows.checked_mul(heads).ok_or_else(reject)?;
    let partials = u64::from(work)
        .checked_mul(u64::from(splits))
        .ok_or_else(reject)?;
    let output_bytes = u64::from(work) * u64::from(hd) * 2;
    let partial_bytes = partials.checked_mul(u64::from(hd) * 4).ok_or_else(reject)?;
    let ml_bytes = partials.checked_mul(8).ok_or_else(reject)?;
    let extent = |slot: usize, bytes: u64| -> Result<()> {
        if d.t[slot] == TENSOR_NONE16
            || tensors
                .get(d.t[slot] as usize)
                .is_none_or(|t| t.bytes < bytes)
        {
            return Err(reject());
        }
        Ok(())
    };
    for output in 0..if prefill { 2 } else { 1 } {
        if (0..if prefill { 5 } else { 3 }).any(|i| i != output && d.t[i] == d.t[output]) {
            return Err(reject());
        }
    }
    if prefill {
        let fused = d.t[5] != TENSOR_NONE16;
        if d.i[1] == 0
            || d.i[3] == 0
            || heads % d.i[3] != 0
            || (fused && (!allow_fused || splits != 1))
            || d.t[6] != TENSOR_NONE16
            || (require_tma_map
                && (d.t[7] == TENSOR_NONE16
                    || tensors
                        .get(d.t[7] as usize)
                        .is_none_or(|tensor| tensor.bytes != 256)))
            || (!require_tma_map && d.t[7] != TENSOR_NONE16)
            || !f32::from_bits(d.fj[0]).is_finite()
            || rows
                .div_ceil(64)
                .checked_mul(heads)
                .and_then(|v| v.checked_mul(splits))
                .is_none()
        {
            return Err(reject());
        }
        let stride = if d.fj[1] == 0 { d.i[1] } else { d.fj[1] };
        let kv_bytes = u64::from(stride)
            .checked_mul(u64::from(d.i[3]))
            .and_then(|bytes| bytes.checked_mul(u64::from(hd) * 2))
            .ok_or_else(reject)?;
        extent(0, partial_bytes)?;
        extent(1, ml_bytes)?;
        extent(2, output_bytes)?;
        extent(3, kv_bytes)?;
        extent(4, kv_bytes)?;
        if fused {
            if d.t[..5].contains(&d.t[5]) {
                return Err(reject());
            }
            extent(5, output_bytes)?;
        }
    } else {
        extent(0, output_bytes)?;
        extent(1, partial_bytes)?;
        extent(2, ml_bytes)?;
    }
    Ok(())
}

fn segment_window(arg: &mut DevProgram, base: &DevProgram, seg: usize, role: bool) {
    arg.cur_seg = if role { seg as u32 } else { 0 };
    arg.gq_seg_ofs = base.gq_seg_ofs + if role { 0 } else { (seg * 4) as u64 };
    arg.gq_cursor = base.gq_cursor
        + if role {
            0
        } else {
            (seg * CTR_STRIDE as usize * 4) as u64
        };
}

fn packet_role_segments(
    g: &crate::asset::devblob::DevProg,
    roles: &[u8],
    tensors: &[crate::asset::devblob::DevTensor],
) -> Result<Vec<u8>> {
    validate_segment_windows(g)?;
    if roles.len() + 1 != g.gq_seg_ofs.len()
        || roles
            .iter()
            .any(|&role| role > plow_asset::segment_roles::MAX_ROLE)
        || (g.packed_prefill_only
            && roles.iter().any(|&role| {
                matches!(
                    role,
                    plow_asset::segment_roles::PREFILL_ATTENTION
                        | plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32
                )
            }))
    {
        return Err(RuntimeError::Rejected(
            "invalid packet segment role count or id".into(),
        ));
    }
    if roles
        .iter()
        .copied()
        .any(plow_asset::segment_roles::is_projection)
    {
        cublaslt::decode_segments(g, tensors, roles)?;
    }
    let mut selected = Vec::new();
    for (seg, bounds) in g.gq_seg_ofs.windows(2).enumerate() {
        let entries = g
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(|| RuntimeError::Rejected("invalid packet role queue window".into()))?;
        if entries.is_empty()
            || entries
                .iter()
                .any(|e| e.seg as usize != seg || e.inst as usize >= g.insts.len())
        {
            return Err(RuntimeError::Rejected(
                "invalid packet role stream entry".into(),
            ));
        }
        let role = roles[seg];
        if role != plow_asset::segment_roles::INTERPRETER {
            let ix = entries[0].inst;
            let d = &g.insts[ix as usize];
            if entries.iter().any(|e| e.inst != ix)
                || g.gq_stream
                    .iter()
                    .any(|e| (e.inst == ix) != (e.seg as usize == seg))
                || g.stream
                    .iter()
                    .any(|e| (e.inst == ix) != (e.seg as usize == seg))
            {
                return Err(RuntimeError::Rejected(
                    "packet role requires one complete isolated instruction".into(),
                ));
            }
            if role == plow_asset::segment_roles::FP8_PREFILL_GEMM {
                if d.op != DevOp::GemmFp8 as u16 || d.i[0] != g.t || d.i[6] == 0 || d.i[7] == 0 {
                    return Err(RuntimeError::Rejected(
                        "FP8 GEMM role requires mapped GEMMs".into(),
                    ));
                }
            } else if role == plow_asset::segment_roles::PREFILL_ATTENTION {
                validate_attention_role_inst(d, g.t, tensors, 256, false, false)?;
            } else if role == plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32 {
                if d.op != DevOp::FlashPrefill as u16 {
                    return Err(RuntimeError::Rejected(
                        "HD512 WG32 role requires FlashPrefill".into(),
                    ));
                }
                validate_attention_role_inst(d, g.t, tensors, 512, true, true)?;
            } else if role == plow_asset::segment_roles::GEMV_CTA512 {
                validate_gemv_decode_role_inst(d, g.t, g.stream_ofs.len(), tensors)?;
            } else if role == plow_asset::segment_roles::MXFP4_MOE {
                validate_mxfp4_moe_role_inst(d, g.t, g.stream_ofs.len(), tensors)?;
            } else if role == plow_asset::segment_roles::FP8_M1
                && (g.t != 1 || d.op != DevOp::GemmFp8 as u16)
            {
                return Err(RuntimeError::Rejected(
                    "invalid FP8 M1 role instruction".into(),
                ));
            }
            let mut queue: Vec<_> = entries
                .iter()
                .map(|e| {
                    (
                        e.inst, e.slice, e.wait_ofs, e.succ_ofs, e.wait_len, e.succ_len, e.flags,
                        e.seg,
                    )
                })
                .collect();
            let mut stream: Vec<_> = g
                .stream
                .iter()
                .filter(|e| e.seg as usize == seg)
                .map(|e| {
                    (
                        e.inst, e.slice, e.wait_ofs, e.succ_ofs, e.wait_len, e.succ_len, e.flags,
                        e.seg,
                    )
                })
                .collect();
            queue.sort_unstable();
            stream.sort_unstable();
            let mut slices: Vec<_> = entries.iter().map(|e| e.slice).collect();
            slices.sort_unstable();
            if slices != (0..u32::from(d.blocks)).collect::<Vec<_>>() || queue != stream {
                return Err(RuntimeError::Rejected(
                    "packet role queue omits or duplicates packet work".into(),
                ));
            }
        }
        selected.push(role);
    }
    Ok(selected)
}

fn packet_role_index(roles: &[u8], segment: usize) -> Option<usize> {
    roles
        .get(segment)
        .copied()
        .filter(|&role| role != plow_asset::segment_roles::INTERPRETER)
        .map(|role| role as usize - 1)
}

struct SegPf {
    f_flash: KernelFn,
    smem_flash: u32,
    grid_flash: u32,
    f_gemm: KernelFn,
    smem_gemm: u32,
    grid_gemm: u32,
    /// T31: the GEMM object's launch block size (`plow_block_pfgemm` global; 256 = legacy).
    block_gemm: u32,
    small_gemm: Option<SegGemm>,
    /// T12: dedicated hd512 flash object (`interp_<tag>_pffa.cubin` in the pair dir,
    /// optional). Class-2 segments (PLOW_PF_SEG_FA512) launch here.
    fa512: Option<(KernelFn, u32, u32)>,
    _m_flash: Module,
    _m_gemm: Module,
    _m_fa512: Option<Module>,
}

impl SegPf {
    fn gemm(&self, small: bool) -> (KernelFn, u32, u32, u32) {
        if small {
            let alt = self
                .small_gemm
                .as_ref()
                .expect("validated small GEMM segment");
            (alt.function, alt.grid, alt.block, alt.smem)
        } else {
            (self.f_gemm, self.grid_gemm, self.block_gemm, self.smem_gemm)
        }
    }
}

mod seg_gemm;
use seg_gemm::small_gemm_segments;

struct KvTensorMap {
    tensor: usize,
    pair: [usize; 2],
    rows: u32,
    hd: u32,
    heads: u32,
    stride: u64,
    batch: usize,
}

fn kv_tensor_maps(
    tensors: &[crate::asset::devblob::DevTensor],
    recipes: &[packet::rope::GenTensor],
    batch: usize,
) -> Result<Vec<KvTensorMap>> {
    let mut maps: Vec<KvTensorMap> = Vec::new();
    for g in recipes
        .iter()
        .filter(|g| g.kind == packet::rope::GEN_TMAP_KV_PAIR)
    {
        let reject = || {
            RuntimeError::Rejected(format!(
                "GEN_TMAP_KV_PAIR tensor {} has invalid targets, extent or geometry",
                g.tensor
            ))
        };
        let tensor = g.tensor as usize;
        let pair = [g.aux as usize, g.scale as usize];
        if batch == 0
            || g.ctx == 0
            || g.hd == 0
            || g.hd % 64 != 0
            || !g.frac.is_finite()
            || g.frac < 1.0
            || g.frac > u32::MAX as f64
            || g.frac.fract() != 0.0
            || pair[0] == pair[1]
            || pair.contains(&tensor)
            || tensors.get(tensor).is_none_or(|t| t.bytes != 256)
            || recipes.iter().filter(|r| r.tensor == g.tensor).count() != 1
            || recipes.iter().any(|r| pair.contains(&(r.tensor as usize)))
        {
            return Err(reject());
        }
        let heads = g.frac as u32;
        let stride = u64::from(g.ctx)
            .checked_mul(u64::from(g.hd))
            .and_then(|bytes| bytes.checked_mul(u64::from(heads)))
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or_else(reject)?;
        let bytes = stride.checked_mul(batch as u64).ok_or_else(reject)?;
        if pair.iter().any(|&id| {
            tensors
                .get(id)
                .is_none_or(|t| t.bytes % batch as u64 != 0 || t.bytes != bytes)
        }) || maps.iter().any(|map| {
            map.pair.iter().any(|id| pair.contains(id))
                && (map.pair != pair || map.rows != g.ctx || map.hd != g.hd || map.heads != heads)
        }) {
            return Err(reject());
        }
        maps.push(KvTensorMap {
            tensor,
            pair,
            rows: g.ctx,
            hd: g.hd,
            heads,
            stride,
            batch,
        });
    }
    Ok(maps)
}

impl KvTensorMap {
    fn slot_bindings(
        &self,
        ptrs: &[u64],
        slot: usize,
        descriptor: u64,
    ) -> Result<[(usize, u64); 3]> {
        let reject = || RuntimeError::Rejected("invalid per-slot KV tensormap address".into());
        if slot >= self.batch || descriptor == 0 || descriptor % 128 != 0 {
            return Err(reject());
        }
        let offset = self.stride.checked_mul(slot as u64).ok_or_else(reject)?;
        let mut bindings = [
            (self.tensor, descriptor),
            (self.pair[0], 0),
            (self.pair[1], 0),
        ];
        for (id, address) in &mut bindings[1..] {
            let base = *ptrs.get(*id).ok_or_else(reject)?;
            if base == 0 || base % 16 != 0 {
                return Err(reject());
            }
            *address = base.checked_add(offset).ok_or_else(reject)?;
            address.checked_add(self.stride).ok_or_else(reject)?;
        }
        Ok(bindings)
    }

    fn encode_slot(
        &self,
        be: &CudaBackend,
        ptrs: &[u64],
        slot: usize,
        descriptor: &DeviceMem,
    ) -> Result<[(usize, u64); 3]> {
        let bindings = self.slot_bindings(ptrs, slot, descriptor.base)?;
        for (i, &(_, base)) in bindings[1..].iter().enumerate() {
            let bytes = be.encode_tmap_kv3(base, self.rows, self.hd, self.heads, 32)?;
            be.upload(descriptor, (i * 128) as u64, &bytes)?;
        }
        Ok(bindings)
    }
}

fn uses_segmented_prefill(
    segmented_objects: bool,
    qwen_chain: bool,
    segments: usize,
    roles: &[u8],
) -> bool {
    segmented_objects
        && !qwen_chain
        && (segments > 1
            || roles
                .iter()
                .any(|&role| role != plow_asset::segment_roles::INTERPRETER))
}

struct PrefillBucket {
    /// Chunk size this bucket was compiled for.
    t: u32,
    /// Per-segment wave class (8 = GEMM-class, 4 = flash-class) when the program is
    /// wave-class segmented AND the SegPf pair is loaded; empty = single launch.
    seg_class: Vec<u8>,
    small_gemm_segments: Vec<bool>,
    qwen_segments: Vec<Option<DevInst64>>,
    packet_segment_roles: Vec<u8>,
    /// `PlowProgram` kernarg (shares `tensors` + `gq_cursor` with the decode path).
    kernarg: DevProgram,
    /// Device instruction stream (patched per chunk over `inst_range`).
    d_inst: DeviceMem,
    /// Host copy of the instructions for the per-chunk patch.
    h_inst: Vec<DevInst64>,
    /// Contiguous instruction window covering every patch site.
    inst_range: std::ops::Range<usize>,
    /// KV-writing `HeadNormRope` sites (`j[0] != 0`): patch `i[3] = c0`.
    rope_sites: Vec<usize>,
    /// `FlashPrefill` sites: patch `i[1] = c0+real`, `i[4] = c0`.
    flash_sites: Vec<usize>,
    /// lm_head GEMM sites (`M == 1`): patch `i[4] = real-1`.
    lmhead_sites: Vec<usize>,
    /// `FlashMerge` sites — neutered (`i[0] = 0`) in PX-1 batched mode, where
    /// the flash op runs the fused (`nsplit=1`, `t5=at`) epilogue per request.
    merge_sites: Vec<usize>,
    /// This bucket runs the fp8-KV opcodes (`HeadNormRopeFp8`/`FlashPrefillFp8`).
    /// PX-1 batched prefill is NOT available for them: the fp8 arm spends t6/t7
    /// on the k/v dequant scales, not on the request table, so the batched patch
    /// would overwrite a scale handle with a slot map.
    fp8_kv: bool,
    /// Whether the one-time PX-1 batched-mode patch has been applied (t6 = the
    /// slot-map / request-table handles, t5 = fused output, merge neutered).
    batch_patched: bool,
    /// This bucket's combined counter+cursor slab (its GQ cursor sits at
    /// the tail; `ctr_bytes` is the full reset size).
    d_ctr: DeviceMem,
    ctr_bytes: usize,
    /// The bucket's other tables, kept alive for the engine's lifetime.
    _tables: Vec<DeviceMem>,
}

fn prefill_patch_range(sites: impl Iterator<Item = usize>) -> std::ops::Range<usize> {
    sites.fold(0..0, |range, ix| {
        if range.is_empty() {
            ix..ix + 1
        } else {
            range.start.min(ix)..range.end.max(ix + 1)
        }
    })
}

fn qwen_prefill_segments(
    program: &crate::asset::devblob::DevProg,
    tensors: &[crate::asset::devblob::DevTensor],
) -> Result<Vec<Option<DevInst64>>> {
    if !program
        .insts
        .iter()
        .any(|i| i.op == DevOp::QwenGdnPrefill as u16)
    {
        return Ok(Vec::new());
    }
    if program.l2_domains != 0 || !(1..=8192).contains(&program.t) {
        return Err(RuntimeError::Rejected(
            "unsupported Qwen prefill topology".into(),
        ));
    }
    let segments = program
        .stream
        .iter()
        .map(|e| e.seg as usize + 1)
        .max()
        .unwrap_or(0);
    if segments == 0 || segments > 2048 {
        return Err(RuntimeError::Rejected(
            "invalid Qwen prefill segment count".into(),
        ));
    }
    if program.gq_seg_ofs.len() != segments + 1
        || program.gq_seg_ofs.first() != Some(&0)
        || program.gq_seg_ofs.last().copied() != Some(program.gq_stream.len() as u32)
    {
        return Err(RuntimeError::Rejected(
            "invalid native GDN queue windows".into(),
        ));
    }
    for (seg, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        let entries = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(|| RuntimeError::Rejected("invalid native GDN queue range".into()))?;
        if entries.iter().any(|e| e.seg as usize != seg) {
            return Err(RuntimeError::Rejected(
                "native GDN queue crosses segment boundaries".into(),
            ));
        }
    }
    let mut external = vec![None; segments];
    for (ix, inst) in program.insts.iter().enumerate() {
        if inst.op != DevOp::QwenGdnPrefill as u16 {
            continue;
        }
        if inst.i[..5] != [program.t, 16, 48, 128, 128]
            || !f32::from_bits(inst.fj[0]).is_finite()
            || (f32::from_bits(inst.fj[0]) - 1.0 / 128_f32.sqrt()).abs() > f32::EPSILON
        {
            return Err(RuntimeError::Rejected(
                "unsupported native GDN geometry".into(),
            ));
        }
        let rows = u64::from(program.t);
        let required = [
            rows * 12288,
            rows * 4096,
            rows * 4096,
            rows * 12288,
            rows * 192,
            rows * 192,
            48 * 128 * 128 * 4,
            48 * 128 * 128 * 4,
        ];
        for (&handle, bytes) in inst.t.iter().zip(required) {
            if tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes) {
                return Err(RuntimeError::Rejected(
                    "native GDN tensor is missing or undersized".into(),
                ));
            }
        }
        let seg = program
            .stream
            .iter()
            .find(|e| e.inst as usize == ix)
            .ok_or_else(|| {
                RuntimeError::Rejected("native GDN instruction has no stream entry".into())
            })?
            .seg as usize;
        if program
            .stream
            .iter()
            .any(|e| (e.inst as usize == ix) != (e.seg as usize == seg))
            || external[seg].is_some()
        {
            return Err(RuntimeError::Rejected(
                "native GDN must occupy one isolated segment".into(),
            ));
        }
        if !program.gq_stream.iter().any(|e| e.inst as usize == ix)
            || program
                .gq_stream
                .iter()
                .any(|e| (e.inst as usize == ix) != (e.seg as usize == seg))
        {
            return Err(RuntimeError::Rejected(
                "native GDN queue segment is not isolated".into(),
            ));
        }
        external[seg] = Some(*inst);
    }
    Ok(external)
}

fn check_qwen_w8a8_capability(prefill: bool, rows: u32, capability: Option<u32>) -> Result<()> {
    let supported_rows = if prefill {
        matches!(rows, 128 | 1024 | 4096 | 8192)
    } else {
        rows == 1
    };
    if !supported_rows || capability != Some(1) {
        return Err(RuntimeError::Rejected(if prefill {
            "Qwen W8A8 prefill requires a supported bucket and paired native FP8 prefill interpreter".into()
        } else {
            "Qwen W8A8 requires batch-1 decode and the paired native FP8 M1 interpreter".into()
        }));
    }
    Ok(())
}

fn qwen_state_slot(state: &DeviceMem, slot: usize, batch: usize) -> Result<DeviceMem> {
    const STRIDE: u64 = 48 * 128 * 128 * 4;
    if batch == 0 || slot >= batch || state.len != batch as u64 * STRIDE {
        return Err(RuntimeError::Rejected(
            "invalid Qwen prefill state slot extent".into(),
        ));
    }
    Ok(DeviceMem::view(state.base + slot as u64 * STRIDE, STRIDE))
}

struct RecurrentState {
    active: usize,
    tensors: Vec<(usize, u64)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackedAdmission {
    Pending,
    Waiting(u64),
    Ready,
}

fn recurrent_state_layout(
    tensors: &[crate::asset::devblob::DevTensor],
    batch: usize,
) -> Result<Option<RecurrentState>> {
    let active_required = tensors.iter().any(|tensor| tensor.name == "in.active");
    recurrent_state_layout_with_active(tensors, batch, active_required)
}

fn recurrent_state_layout_with_active(
    tensors: &[crate::asset::devblob::DevTensor],
    batch: usize,
    active_required: bool,
) -> Result<Option<RecurrentState>> {
    let mut states = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        if !tensor.name.starts_with("state.") {
            continue;
        }
        if !tensor.name.starts_with("state.qwen.")
            || !(tensor.name.ends_with(".conv") || tensor.name.ends_with(".gdn"))
            || batch == 0
            || tensor.bytes == 0
            || tensor.bytes % batch as u64 != 0
            || tensor.init.is_some()
        {
            return Err(RuntimeError::Device(format!(
                "invalid recurrent state tensor {} for batch {batch}",
                tensor.name
            )));
        }
        states.push((index, tensor.bytes / batch as u64));
    }
    if states.is_empty() && !active_required {
        return Ok(None);
    }
    let active = tensors
        .iter()
        .position(|t| t.name == "in.active")
        .filter(|&i| tensors[i].bytes == batch as u64 * 4 && tensors[i].init.is_none())
        .ok_or_else(|| {
            RuntimeError::Device("recurrent state requires in.active i32[batch]".into())
        })?;
    Ok(Some(RecurrentState {
        active,
        tensors: states,
    }))
}

/// The per-model GPU engine: packet decode programs sharing one KV cache and physical slots.
pub struct GpuEngine {
    be: Arc<CudaBackend>,
    f: KernelFn,
    grid: u32,
    smem: u32,
    /// The engine's single ordered device queue: every decode/prefill copy,
    /// memset, and launch is enqueued here. Decode retires with ONE
    /// `cuStreamSynchronize` per step; decode-loop prompt consumption
    /// ([`Self::consume_prompt`]) retires L launches with one sync.
    /// Steady-state serving performs no `cuCtxSynchronize` (plan gate). One
    /// stream by design: decode and prefill share mutable run state (`in.*`,
    /// activations, the GQ cursor), so overlapping streams would race until
    /// every in-flight command owns separate run-state storage.
    stream: CudaStream,
    /// The interpreter module: its own lifetime anchor (unloaded in `Drop`)
    /// AND the handle `trace_reset`/`trace_summary` read the device trace
    /// globals through. NOT `_module` — the leading-underscore convention in
    /// this file means "held for liveness only, never read", which is true of
    /// [`Sampler::_module`] and [`MultiStep::_module`] and false of this one.
    module: Arc<DecodeModule>,

    /// The prefill object's kernel + smem, and the uploaded bucket programs.
    /// `None`/empty when no `_pf` cubin is present — the mux then falls back to
    /// decode-only prompt consumption.
    f_pf: Option<KernelFn>,
    /// Segmented-prefill object pair (PLOW_PF_SEG_DIR); None = single-object prefill.
    seg_pf: Option<SegPf>,
    qwen_prefill: Option<crate::device::cuda::qwen_gdn::NativeGdn>,
    packet_roles: [Option<PacketRole>; plow_asset::segment_roles::MAX_ROLE as usize],
    decode_packet_roles: Vec<u8>,
    cublaslt_decode: Vec<Option<CublasLtDecodeRoute>>,
    cublaslt_decode_graph: Option<crate::device::cuda::GraphExec>,
    cublaslt_decode_capture: bool,
    /// T35 (PLOW_PF_SEG_GRAPH=1): cached instantiated segment-chain graphs, keyed by
    /// (bucket, slot-tensor-base) — one cuGraphLaunch replaces ~480 kernel submits.
    seg_graphs: std::collections::HashMap<(usize, u64), crate::device::cuda::GraphExec>,
    smem_pf: u32,
    prefill: Vec<PrefillBucket>,
    /// Read in `Drop` (unloaded separately from [`Self::module`]), so not
    /// underscore-prefixed either.
    module_pf: Option<Module>,

    /// Device stochastic sampler (`PLOW_DEV_SAMPLE=1` + a sampler cubin;
    /// plan stage 4). `None` = the host path (full-logit D2H + CPU sampler),
    /// byte-identical to before. When present, `step_slots_sampled` launches
    /// `plow_sample` after the decode kernel to write each stochastic row's
    /// sampled token into `in.ids[b]` — no per-row vocab D2H.
    sampler: Option<Sampler>,
    /// Bounded device multi-step (`PLOW_MULTISTEP=K`; plan stage 5). `None` =
    /// the per-token launch model. When present, [`Self::multi_step`] runs a
    /// K-token greedy quantum with one host sync.
    multistep: Option<MultiStep>,

    /// Per-tensor device buffers, indexed by blob tensor handle. Ordinarily
    /// **views** into `_weight_slab`, not owners — see it for why.
    devp: Vec<DeviceMem>,
    /// Owner of the one span every non-VMM-prefix tensor is carved out of.
    ///
    /// The blob declares every tensor and its size before a byte moves, so the
    /// whole layout is known up front and there is no reason to ask the driver
    /// for it a tensor at a time.
    ///
    /// The slab always bought MEMORY: per-allocation rounding waste on a 12B
    /// model drops from 322 MiB to 21 MiB, which the co-residency planner
    /// spends directly (the AMD loader carries the same carve — see
    /// `exec::amd`'s `_weight_slab`, where ROCr's next-power-of-two rounding
    /// under 2 MiB makes the waste far worse).
    ///
    /// What a flat `cuMemAlloc` slab could never buy was TIME: the driver
    /// charges by COMMITTED BYTES, not call count — 737 per-tensor allocs cost
    /// 1.97 s on a 12B load and one 25 GiB request costs 1.92 s, one ~13 GiB/s
    /// commit rate across sizes (`perf-data/coldstart-plow-vs-vllm-gh200.md`
    /// §4b). The [`WeightSlab::Vmm`] default kills the term the only way it
    /// can be killed: the commit is not batched but taken OFF the critical
    /// path — VA reserved in µs, pages committed by a background mapper
    /// overlapped with the upload. Measured on GH200 (12B warm): the 1.74 s
    /// upfront stall becomes 0.1 ms of watermark waits; total load 3.69 s →
    /// 1.99 s.
    ///
    /// Views never free, so this owner must outlive them; both live here, and a
    /// `View`'s Drop is a no-op, so field drop order cannot matter. The Vmm
    /// arm's Drop unmaps and releases its physical chunks.
    _weight_slab: WeightSlab,
    d_inst: DeviceMem,
    /// Owner of the decode counter+cursor allocation. `d_ctr` and the decode
    /// GQ cursor are aliased **views** into it (never freed by their own
    /// Drop) — the owner must sit in the engine so their addresses stay valid
    /// for its whole lifetime. The cursor's address is baked into `kernarg`
    /// and re-armed as part of the `d_ctr` combined memset; the view is kept
    /// only to anchor its storage (prefill buckets carry their own cursor).
    _ctr_block: DeviceMem,
    d_ctr: DeviceMem,
    _d_gq_cursor: DeviceMem,
    /// The device tensor-pointer table (`kernarg.tensors`) — the batch-major
    /// bases decode expects. Immutable after load.
    d_tens: DeviceMem,
    /// Per-slot prefill tables (index b-1 = slot b; slot 0 is `d_tens`): the
    /// same table with KV bases and rank-3 descriptors bound to that slot,
    /// since the prefill programs address the cache slot-relative. A per-slot
    /// launch selects its table through the kernarg (`tens_slot_base`) —
    /// nothing is rewritten or restored. Empty at B == 1.
    d_tens_slots: Vec<DeviceMem>,
    /// Rank-3 KV descriptors bind slot-specific addresses and outlive every slot table use.
    _kv_tmap_slots: Vec<DeviceMem>,
    /// The other decode tables live for the engine's lifetime and are never
    /// re-uploaded; their device pointers are baked into `kernarg`.
    _tables: Vec<DeviceMem>,

    /// The `PlowProgram` kernarg (built once; pointers never move).
    kernarg: DevProgram,
    /// Host copy of the decode instructions for the per-step kv-row patch.
    h_inst: Vec<DevInst64>,
    decode_rungs: Vec<DecodeRung>,
    decode_contexts: Option<decode_context::MaterializedContexts>,
    kvrow: Vec<u32>,
    /// Contiguous instruction range covering every kv-row patch site.
    kvrow_lo: usize,
    kvrow_hi: usize,
    ctr_bytes: usize,
    /// Bytes of GQ cursor lines at the counter slab's tail — one PLOW_CTR line
    /// per gq segment (P lines for an L2-placed blob, 1 otherwise).
    cursor_bytes: usize,

    t_ids: usize,
    t_pos: usize,
    t_kvlen: usize,
    t_logits: usize,
    recurrent: Option<RecurrentState>,
    /// Blob tensor names, indexed by handle (same order as `devp`). Lets the
    /// block harness resolve an activation handle by name — see
    /// [`Self::handle_of`] / [`Self::download_activation`] /
    /// [`Self::upload_activation`]. Not on any hot path.
    tensor_names: Vec<String>,
    vocab: usize,
    max_ctx: usize,

    /// Decode batch B — the last program's `t` (`PLOW_DECODE_BATCH` at
    /// compile time). The engine drives B independent sequence slots.
    batch: usize,
    /// Per-slot sequence position (== tokens consumed so far). For a slot
    /// mid-prefill (`prefill_chunk`) this is the prefill frontier — kept
    /// current per chunk so the batched-decode garbage KV write for unfed
    /// rows always lands in the row the slot's next chunk overwrites.
    pos: Vec<u32>,
    /// Per-slot rows served from the prefix cache by the current sequence's
    /// attach (0 = cold). Feeds per-request `usage.cached_tokens`.
    vmm_attached: Vec<u32>,
    vmm_active: Vec<bool>,
    packed_admission: Vec<PackedAdmission>,
    kv_admission_epoch: u64,
    /// Per-slot token ids whose KV rows the slot currently holds (prompt,
    /// then every decode-fed token) — `seq_tokens[b].len() == pos[b]` when
    /// consistent. Lets `begin_slot` publish the finished sequence's
    /// GENERATED blocks into the prefix cache, so a follow-up turn embedding
    /// this turn's output attaches instead of re-prefilling it.
    seq_tokens: Vec<Vec<u32>>,
    /// Stop-token set (the checkpoint's `eos_token_id`). `Arc` so the mux can
    /// take a per-tick handle without cloning the Vec while the engine stays
    /// mutably borrowed.
    stop_ids: std::sync::Arc<Vec<u32>>,
    /// Reusable download buffer for the bf16 logits row.
    logits_raw: Vec<u8>,
    /// Pinned per-step staging (`step_slots` is the per-token hot path — no
    /// per-step allocation, and pinned pages make the stream copies async).
    stage: StepStage,
    /// Ordering-only event recorded after each `consume_prompt` H2D so the
    /// next iteration may overwrite pinned staging without waiting on the
    /// interpreter (the kernel stays queued on `stream`).
    h2d_ev: CudaEvent,
    /// Pre-allocated prefill ids/pos buffers (sized to max prefill bucket t).
    pf_ids: Vec<i32>,
    pf_pos: Vec<i32>,
    /// Reusable f32 logits buffer for host sampling (gpu_finish_token).
    logits_f32: Vec<f32>,
    /// Env-gated (`PLOW_STEP_TIME=1`) host-op timing for the decode step.
    timing: Option<StepTiming>,
    /// VMM prefix sharing (`PLOW_VMM_PREFIX=1`); `None` = the cudaMalloc
    /// default path, byte-identical behavior to before this feature.
    vmm: Option<VmmServe>,
    /// Cross-request batched prefill selected by packet metadata or legacy
    /// runtime opt-in; `None` uses per-slot serialized prefill.
    pf_batch: Option<PfBatch>,
    /// Next physical slot considered first by cross-request prefill admission.
    prefill_turn: usize,
    packed_prefill: Option<plow_asset::packed_prefill::Manifest>,
    packed_terminal: Option<packed_terminal::PackedTerminal>,
    mixed_step: Option<mixed_step::MixedCudaStep>,
    token_batch: Option<token_batch::CudaTokenBatch>,
    slot_generations: Vec<u32>,
}

/// Full model-load wall timeline (`PLOW_LOAD_PROFILE=1`).
///
/// Exclusive phases are sequential Instant walls that should sum ≈ total
/// `GpuEngine::load` elapsed. Overlay metrics (DMA events, prefetch wall/worker
/// sum, upload sub-Instants) overlap those phases and are printed as annotations.
struct LoadTiming {
    t0: std::time::Instant,
    // ---- exclusive wall phases ----
    blob_ms: f64,
    module_ms: f64,
    vmm_ms: f64,
    /// Weight-slab bringup: VMM reserve+mapper spawn (µs), or the flat
    /// `cuMemAlloc` fallback (the §4b commit stall, when it is paid at all).
    slab_ms: f64,
    /// Upload time actually blocked on the VMM mapper watermark (overlay
    /// inside upload_all; ≈0 when the mapper outruns the upload).
    slab_wait_ms: f64,
    /// Final wait for the slab tail (KV is never written at load) after
    /// pipe.finish.
    slab_join_ms: f64,
    ckpt_open_ms: f64,
    pipe_setup_ms: f64,
    pipe_stream_ms: f64,
    pipe_pinned_ms: f64,
    pipe_events_ms: f64,
    prefetch_spawn_ms: f64,
    upload_all_ms: f64,
    pipe_finish_ms: f64,
    prefetch_join_ms: f64,
    moe_ms: f64,
    decode_tables_ms: f64,
    prefill_ms: f64,
    final_init_ms: f64,
    // ---- checkpoint open sub ----
    ckpt_scan_ms: f64,
    ckpt_mmap_ms: f64,
    ckpt_meta_ms: f64,
    ckpt_index_ms: f64,
    // ---- upload overlays (inside upload_all / finish) ----
    host_memcpy_ms: f64,
    host_memcpy_bytes: u64,
    htod_enqueue_ms: f64,
    dma_gpu_ms: f64,
    dma_bytes: u64,
    event_sync_ms: f64,
    stream_sync_ms: f64,
    memset_ms: f64,
    n_tensors_uploaded: usize,
    n_chunks: usize,
    // ---- alloc aggregates ----
    alloc_ms: f64,
    alloc_count: usize,
    alloc_bytes: u64,
    /// Alloc Instant sum inside `upload_all` only (for upload_other residual).
    alloc_upload_ms: f64,
    // ---- prefetch overlays ----
    prefetch_wall_ms: f64,
    prefetch_worker_ms: f64,
    prefetch_bytes: u64,
    prefetch_workers: usize,
    // ---- flame spans (name, t0_ms, t1_ms relative to load t0) ----
    spans: Vec<(String, f64, f64)>,
}

impl LoadTiming {
    fn new(t0: std::time::Instant) -> Self {
        Self {
            t0,
            blob_ms: 0.0,
            module_ms: 0.0,
            vmm_ms: 0.0,
            slab_ms: 0.0,
            slab_wait_ms: 0.0,
            slab_join_ms: 0.0,
            ckpt_open_ms: 0.0,
            pipe_setup_ms: 0.0,
            pipe_stream_ms: 0.0,
            pipe_pinned_ms: 0.0,
            pipe_events_ms: 0.0,
            prefetch_spawn_ms: 0.0,
            upload_all_ms: 0.0,
            pipe_finish_ms: 0.0,
            prefetch_join_ms: 0.0,
            moe_ms: 0.0,
            decode_tables_ms: 0.0,
            prefill_ms: 0.0,
            final_init_ms: 0.0,
            ckpt_scan_ms: 0.0,
            ckpt_mmap_ms: 0.0,
            ckpt_meta_ms: 0.0,
            ckpt_index_ms: 0.0,
            host_memcpy_ms: 0.0,
            host_memcpy_bytes: 0,
            htod_enqueue_ms: 0.0,
            dma_gpu_ms: 0.0,
            dma_bytes: 0,
            event_sync_ms: 0.0,
            stream_sync_ms: 0.0,
            memset_ms: 0.0,
            n_tensors_uploaded: 0,
            n_chunks: 0,
            alloc_ms: 0.0,
            alloc_count: 0,
            alloc_bytes: 0,
            alloc_upload_ms: 0.0,
            prefetch_wall_ms: 0.0,
            prefetch_worker_ms: 0.0,
            prefetch_bytes: 0,
            prefetch_workers: 0,
            spans: Vec::new(),
        }
    }

    fn ms_since_t0(&self) -> f64 {
        self.t0.elapsed().as_secs_f64() * 1e3
    }

    fn cum_ms(&self) -> f64 {
        self.ms_since_t0()
    }

    fn epoch_ms(t: std::time::SystemTime) -> u128 {
        t.duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    fn gib(bytes: u64) -> f64 {
        bytes as f64 / (1u64 << 30) as f64
    }

    fn gib_s(bytes: u64, ms: f64) -> f64 {
        if ms <= 0.0 {
            0.0
        } else {
            Self::gib(bytes) / (ms / 1e3)
        }
    }

    /// Run `f`; return `(result, elapsed_ms)`. Caller stores into an exclusive slot
    /// and we record the flame span + stage log.
    fn phase<R>(&mut self, name: &str, f: impl FnOnce() -> R) -> (R, f64) {
        let sys0 = std::time::SystemTime::now();
        let t0_ms = self.ms_since_t0();
        let t = std::time::Instant::now();
        let out = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let t1_ms = self.ms_since_t0();
        self.spans.push((name.to_string(), t0_ms, t1_ms));
        self.log_stage(name, sys0, std::time::SystemTime::now(), ms);
        (out, ms)
    }

    fn note_alloc(&mut self, bytes: u64, ms: f64, in_upload: bool) {
        self.alloc_ms += ms;
        self.alloc_count += 1;
        self.alloc_bytes += bytes;
        if in_upload {
            self.alloc_upload_ms += ms;
        }
        tracing::debug!(
            bytes,
            elapsed_ms = format!("{ms:.3}").as_str(),
            api = "cuMemAlloc_v2",
            "load alloc"
        );
    }

    fn log_stage(
        &self,
        name: &str,
        start: std::time::SystemTime,
        end: std::time::SystemTime,
        elapsed_ms: f64,
    ) {
        tracing::info!(
            stage = name,
            start_ms = Self::epoch_ms(start),
            end_ms = Self::epoch_ms(end),
            elapsed_ms = format!("{elapsed_ms:.3}").as_str(),
            cumulative_ms = format!("{:.3}", self.cum_ms()).as_str(),
            "load stage"
        );
    }

    fn log_stage_bytes(
        &self,
        name: &str,
        start: std::time::SystemTime,
        end: std::time::SystemTime,
        elapsed_ms: f64,
        bytes: u64,
    ) {
        tracing::info!(
            stage = name,
            start_ms = Self::epoch_ms(start),
            end_ms = Self::epoch_ms(end),
            elapsed_ms = format!("{elapsed_ms:.3}").as_str(),
            bytes,
            gib = format!("{:.2}", Self::gib(bytes)).as_str(),
            gib_s = format!("{:.2}", Self::gib_s(bytes, elapsed_ms)).as_str(),
            cumulative_ms = format!("{:.3}", self.cum_ms()).as_str(),
            "load stage"
        );
    }

    fn absorb_pipe_setup(&mut self, p: &PipeTiming) {
        self.pipe_stream_ms = p.stream_ns as f64 / 1e6;
        self.pipe_pinned_ms = p.pinned_ns as f64 / 1e6;
        self.pipe_events_ms = p.events_ns as f64 / 1e6;
    }

    fn absorb_pipe_upload(&mut self, p: &PipeTiming) {
        self.host_memcpy_ms = p.host_memcpy_ns as f64 / 1e6;
        self.host_memcpy_bytes = p.bytes;
        self.htod_enqueue_ms = p.htod_enq_ns as f64 / 1e6;
        self.dma_gpu_ms = p.dma_ms;
        self.dma_bytes = p.bytes;
        self.event_sync_ms = p.event_sync_ns as f64 / 1e6;
        self.stream_sync_ms = p.stream_sync_ns as f64 / 1e6;
        if p.bytes > 0 && STAGE > 0 {
            self.n_chunks = (p.bytes as usize).div_ceil(STAGE);
        }
    }

    fn exclusive_sum_ms(&self) -> f64 {
        self.blob_ms
            + self.module_ms
            + self.vmm_ms
            + self.slab_ms
            + self.slab_join_ms
            + self.ckpt_open_ms
            + self.pipe_setup_ms
            + self.prefetch_spawn_ms
            + self.upload_all_ms
            + self.pipe_finish_ms
            + self.prefetch_join_ms
            + self.moe_ms
            + self.decode_tables_ms
            + self.prefill_ms
            + self.final_init_ms
    }

    fn upload_other_ms(&self) -> f64 {
        (self.upload_all_ms
            - self.host_memcpy_ms
            - self.htod_enqueue_ms
            - self.event_sync_ms
            - self.memset_ms
            - self.alloc_upload_ms
            - self.slab_wait_ms)
            .max(0.0)
    }

    fn print_summary(&self, total_load_ms: f64) {
        let excl = self.exclusive_sum_ms();
        let other = (total_load_ms - excl).max(0.0);
        let host_gib = Self::gib(self.host_memcpy_bytes);
        let dma_gib = Self::gib(self.dma_bytes);
        let pref_gib = Self::gib(self.prefetch_bytes);
        let avg_chunk = if self.n_chunks > 0 {
            self.host_memcpy_bytes as f64 / self.n_chunks as f64
        } else {
            0.0
        };
        tracing::info!(
            "\n\
================= MODEL LOAD BREAKDOWN =================\n\
Blob find/read/parse         : {:>8.1} ms\n\
Interpreter module           : {:>8.1} ms\n\
VMM bringup                  : {:>8.1} ms\n\
Weight slab bringup          : {:>8.1} ms\n\
Checkpoint open              : {:>8.1} ms  (scan {:.1} + mmap {:.1} + meta {:.1} + index {:.1})\n\
Upload pipe setup            : {:>8.1} ms  (stream {:.1} + pinned {:.1} + events {:.1})\n\
Prefetch spawn               : {:>8.1} ms\n\
upload_all (wall)            : {:>8.1} ms\n\
    tensors uploaded         : {:>8}\n\
    chunks / avg chunk       : {:>8} / {:.1} MiB\n\
    RAM→Pinned memcpy        : {:>8.1} ms  ({:.2} GiB, {:.2} GiB/s)\n\
    H2D enqueue (CPU)        : {:>8.1} ms\n\
    event_synchronize        : {:>8.1} ms\n\
    memset                   : {:>8.1} ms\n\
    alloc (cuMemAlloc)       : {:>8.1} ms  ({} allocs, {:.2} GiB)\n\
    slab commit wait         : {:>8.1} ms\n\
    other (iter/lookup)      : {:>8.1} ms\n\
    Pinned→GPU DMA (events)  : {:>8.1} ms  ({:.2} GiB, {:.2} GiB/s; overlay)\n\
pipe.finish                  : {:>8.1} ms  (stream_sync {:.1})\n\
Slab tail commit join        : {:>8.1} ms\n\
Prefetch join                : {:>8.1} ms\n\
    Disk→RAM wall (overlay)  : {:>8.1} ms  ({:.2} GiB, {:.2} GiB/s)\n\
    worker time (overlay)    : {:>8.1} ms  ({} workers; parallel sum)\n\
MoE tables                   : {:>8.1} ms\n\
Decode tables                : {:>8.1} ms\n\
Prefill load                 : {:>8.1} ms\n\
Final init                   : {:>8.1} ms\n\
--------------------------------------------------------\n\
Exclusive sum                : {:>8.1} ms\n\
Total GpuEngine::load()      : {:>8.1} ms\n\
Other (residual)             : {:>8.1} ms\n\
========================================================",
            self.blob_ms,
            self.module_ms,
            self.vmm_ms,
            self.slab_ms,
            self.ckpt_open_ms,
            self.ckpt_scan_ms,
            self.ckpt_mmap_ms,
            self.ckpt_meta_ms,
            self.ckpt_index_ms,
            self.pipe_setup_ms,
            self.pipe_stream_ms,
            self.pipe_pinned_ms,
            self.pipe_events_ms,
            self.prefetch_spawn_ms,
            self.upload_all_ms,
            self.n_tensors_uploaded,
            self.n_chunks,
            avg_chunk / (1 << 20) as f64,
            self.host_memcpy_ms,
            host_gib,
            Self::gib_s(self.host_memcpy_bytes, self.host_memcpy_ms),
            self.htod_enqueue_ms,
            self.event_sync_ms,
            self.memset_ms,
            self.alloc_ms,
            self.alloc_count,
            Self::gib(self.alloc_bytes),
            self.slab_wait_ms,
            self.upload_other_ms(),
            self.dma_gpu_ms,
            dma_gib,
            Self::gib_s(self.dma_bytes, self.dma_gpu_ms),
            self.pipe_finish_ms,
            self.stream_sync_ms,
            self.slab_join_ms,
            self.prefetch_join_ms,
            self.prefetch_wall_ms,
            pref_gib,
            Self::gib_s(self.prefetch_bytes, self.prefetch_wall_ms),
            self.prefetch_worker_ms,
            self.prefetch_workers,
            self.moe_ms,
            self.decode_tables_ms,
            self.prefill_ms,
            self.final_init_ms,
            excl,
            total_load_ms,
            other,
        );
    }

    fn print_flame(&self, total_load_ms: f64) {
        let mut lines = String::from("\n0 ms\n│\n");
        for (name, t0, t1) in &self.spans {
            lines.push_str(&format!("├── {name}  [{t0:.1} → {t1:.1} ms]\n"));
        }
        lines.push_str(&format!("└── load_finished\n{total_load_ms:.0} ms"));
        tracing::info!("{lines}");
    }
}

/// Per-pipe Instant / CUDA-event accumulators for the model-load breakdown.
struct PipeTiming {
    stream_ns: u128,
    pinned_ns: u128,
    events_ns: u128,
    host_memcpy_ns: u128,
    htod_enq_ns: u128,
    event_sync_ns: u128,
    stream_sync_ns: u128,
    dma_ms: f64,
    bytes: u64,
}

/// Double-buffered checkpoint-upload pipeline (plan stage 9). Two pinned
/// staging buffers ping-pong on a dedicated stream so the host copy of chunk
/// N+1 (safetensors mmap → pinned) overlaps the async H2D DMA of chunk N —
/// the whole checkpoint (tens of GiB) is bandwidth-bound, and the serial
/// `copy; blocking-upload` loop left the copy and the DMA strictly sequential.
struct UploadPipe<'a> {
    be: &'a CudaBackend,
    stream: CudaStream,
    bufs: [PinnedHost; 2],
    /// End-of-H2D markers (also gate buffer reuse). Timing-enabled when profiling.
    ends: [CudaEvent; 2],
    /// Start-of-H2D markers for `cuEventElapsedTime` (only when profiling).
    starts: Option<[CudaEvent; 2]>,
    primed: [bool; 2],
    n: usize,
    /// Direct mode: the device reads pageable memory coherently
    /// ([`crate::device::Backend::coherent_host_dma`]) so long-lived sources
    /// (checkpoint mmap, blob init) skip the staging memcpy entirely via
    /// [`Self::push_direct`] — 332 GiB/s vs 13 GiB/s staged on GH200. Staged
    /// [`Self::push`] stays available in this mode for short-lived sources
    /// (generated tensors).
    direct: bool,
    /// A direct-mode DMA window is open (first push_direct recorded
    /// `dwin.0`; finish() records `dwin.1` and folds the elapsed into the
    /// dma_ms overlay).
    direct_started: bool,
    /// Direct-window timing events (profiling only) — the per-slot
    /// `starts`/`ends` pairs are re-recorded by every staged push, so the
    /// direct window needs its own markers.
    dwin: Option<(CudaEvent, CudaEvent)>,
    timing: Option<PipeTiming>,
}

impl<'a> UploadPipe<'a> {
    fn new(be: &'a CudaBackend, profile: bool) -> Result<Self> {
        let mut timing = profile.then(|| PipeTiming {
            stream_ns: 0,
            pinned_ns: 0,
            events_ns: 0,
            host_memcpy_ns: 0,
            htod_enq_ns: 0,
            event_sync_ns: 0,
            stream_sync_ns: 0,
            dma_ms: 0.0,
            bytes: 0,
        });
        let t_stream = profile.then(std::time::Instant::now);
        let stream = be.stream_create()?;
        if let (Some(t), Some(tm)) = (t_stream, timing.as_mut()) {
            tm.stream_ns = t.elapsed().as_nanos();
        }
        let t_pin = profile.then(std::time::Instant::now);
        let bufs = [be.host_alloc_pinned(STAGE)?, be.host_alloc_pinned(STAGE)?];
        if let (Some(t), Some(tm)) = (t_pin, timing.as_mut()) {
            tm.pinned_ns = t.elapsed().as_nanos();
        }
        let t_ev = profile.then(std::time::Instant::now);
        // Profiling needs timing-capable events for DMA elapsed; otherwise
        // sync-only (cheaper) — used purely to gate buffer reuse.
        let ends = [be.event_create(profile)?, be.event_create(profile)?];
        let starts = if profile {
            Some([be.event_create(true)?, be.event_create(true)?])
        } else {
            None
        };
        if let (Some(t), Some(tm)) = (t_ev, timing.as_mut()) {
            tm.events_ns = t.elapsed().as_nanos();
        }
        let direct = be.coherent_host_dma() && crate::asset::checkpoint::upload_direct_enabled();
        tracing::info!(direct, "upload pipe mode");
        let dwin = if profile {
            Some((be.event_create(true)?, be.event_create(true)?))
        } else {
            None
        };
        Ok(UploadPipe {
            stream,
            bufs,
            ends,
            starts,
            primed: [false, false],
            n: 0,
            direct,
            direct_started: false,
            dwin,
            timing,
            be,
        })
    }

    fn is_direct(&self) -> bool {
        self.direct
    }

    /// Direct-mode enqueue: async H2D straight from `src`, no staging memcpy.
    /// Only meaningful when [`Self::is_direct`] — the caller falls back to
    /// [`Self::push`] otherwise.
    ///
    /// # Safety
    /// `src` must stay valid (unmoved, undropped) until [`Self::finish`]
    /// returns — the loader only passes slices of the checkpoint mmap and the
    /// blob, both of which outlive the pipe.
    unsafe fn push_direct(&mut self, dst: u64, src: &[u8]) -> Result<()> {
        if !self.direct_started {
            self.direct_started = true;
            if let Some((start, _)) = &self.dwin {
                self.be.event_record(start, &self.stream)?;
            }
        }
        let t_enq = self.timing.is_some().then(std::time::Instant::now);
        // SAFETY: caller upholds the source-lifetime contract above; dst is
        // inside a live (committed) allocation.
        unsafe {
            self.be.memcpy_htod_async(dst, src, &self.stream)?;
        }
        if let Some(tm) = self.timing.as_mut() {
            if let Some(t) = t_enq {
                tm.htod_enq_ns += t.elapsed().as_nanos();
            }
            tm.bytes += src.len() as u64;
        }
        Ok(())
    }

    /// Stage `chunk` (≤ STAGE) into the next free pinned buffer and enqueue its
    /// async H2D to `dst` on the pipe's stream. Blocks only when that buffer's
    /// previous DMA has not yet retired (two-deep pipeline).
    fn push(&mut self, dst: u64, chunk: &[u8]) -> Result<()> {
        let slot = self.n & 1;
        if self.primed[slot] {
            let t_wait = self.timing.is_some().then(std::time::Instant::now);
            self.be.event_synchronize(&self.ends[slot])?;
            if let Some(tm) = self.timing.as_mut() {
                if let Some(t) = t_wait {
                    tm.event_sync_ns += t.elapsed().as_nanos();
                }
                // Previous use of this slot has retired — fold its GPU DMA time.
                if let Some(starts) = &self.starts {
                    tm.dma_ms += self.be.event_elapsed_ms(&starts[slot], &self.ends[slot])? as f64;
                }
            }
        }
        let t_copy = self.timing.is_some().then(std::time::Instant::now);
        self.bufs[slot].as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
        if let Some(tm) = self.timing.as_mut() {
            if let Some(t) = t_copy {
                tm.host_memcpy_ns += t.elapsed().as_nanos();
            }
            tm.bytes += chunk.len() as u64;
        }
        if let Some(starts) = &self.starts {
            self.be.event_record(&starts[slot], &self.stream)?;
        }
        // SAFETY: the pinned buffer stays alive (owned by self) until finish()
        // synchronizes the stream; dst is inside a live allocation (caller).
        let t_enq = self.timing.is_some().then(std::time::Instant::now);
        unsafe {
            self.be.memcpy_htod_async(
                dst,
                &self.bufs[slot].as_slice()[..chunk.len()],
                &self.stream,
            )?;
        }
        if let (Some(t), Some(tm)) = (t_enq, self.timing.as_mut()) {
            tm.htod_enq_ns += t.elapsed().as_nanos();
        }
        self.be.event_record(&self.ends[slot], &self.stream)?;
        self.primed[slot] = true;
        self.n += 1;
        Ok(())
    }

    /// Retire every enqueued upload (call before the pinned buffers drop —
    /// and, in direct mode, before the borrowed sources go away).
    fn finish(&mut self) -> Result<()> {
        // Close the direct-mode DMA window at the stream tail.
        if self.direct_started {
            if let Some((_, end)) = &self.dwin {
                self.be.event_record(end, &self.stream)?;
            }
        }
        // Flush DMA elapsed for the last use of each primed slot.
        if let Some(tm) = self.timing.as_mut() {
            if let Some(starts) = &self.starts {
                for slot in 0..2 {
                    if !self.primed[slot] {
                        continue;
                    }
                    self.be.event_synchronize(&self.ends[slot])?;
                    tm.dma_ms += self.be.event_elapsed_ms(&starts[slot], &self.ends[slot])? as f64;
                }
            }
        }
        let t_sync = self.timing.is_some().then(std::time::Instant::now);
        self.be.stream_synchronize(&self.stream)?;
        if let (Some(t), Some(tm)) = (t_sync, self.timing.as_mut()) {
            tm.stream_sync_ns = t.elapsed().as_nanos();
        }
        if self.direct_started {
            if let (Some((start, end)), Some(tm)) = (&self.dwin, self.timing.as_mut()) {
                tm.dma_ms += self.be.event_elapsed_ms(start, end)? as f64;
            }
        }
        Ok(())
    }

    fn take_timing(&mut self) -> Option<PipeTiming> {
        self.timing.take()
    }
}

/// Pinned per-step staging slab: `[ids B][pos B][kvlen B]` i32, uploaded with
/// three async H2D copies on the engine stream, plus an optional active-slot mask
/// for recurrent models. `ids` doubles as the token
/// readback destination — the same `in.ids` tensor round-trips (`ARGMAX_FIN`
/// writes the next token there), so no separate download buffer exists.
struct StepStage {
    slab: PinnedHost,
    batch: usize,
}

impl StepStage {
    fn new(be: &CudaBackend, batch: usize, recurrent: bool) -> Result<StepStage> {
        Ok(StepStage {
            slab: be.host_alloc_pinned((3 + usize::from(recurrent)) * batch * 4)?,
            batch,
        })
    }

    /// The three staging sections for the host fill.
    fn parts_mut(&mut self) -> (&mut [i32], &mut [i32], &mut [i32]) {
        let all: &mut [i32] = bytemuck::cast_slice_mut(self.slab.as_mut_slice());
        let (ids, rest) = all.split_at_mut(self.batch);
        let (pos, rest) = rest.split_at_mut(self.batch);
        let kvlen = &mut rest[..self.batch];
        (ids, pos, kvlen)
    }

    /// Byte view of section `k` (0 = ids, 1 = pos, 2 = kvlen) for the copies.
    fn section(&self, k: usize) -> &[u8] {
        &self.slab.as_slice()[k * self.batch * 4..(k + 1) * self.batch * 4]
    }

    /// Mutable byte view of section `k` — the async D2H destination.
    fn section_mut(&mut self, k: usize) -> &mut [u8] {
        let b = self.batch;
        &mut self.slab.as_mut_slice()[k * b * 4..(k + 1) * b * 4]
    }

    /// Token `b` read back into the ids section by the post-launch D2H.
    fn token(&self, b: usize) -> u32 {
        let ids: &[i32] = bytemuck::cast_slice(self.section(0));
        ids[b] as u32
    }
}

/// Accumulated `step_slots` timing, logged every 128 steps. Device phases
/// (upload+re-arm, interpreter, token D2H) come from CUDA events on the
/// engine stream — the plan's instrumentation rule; CPU timestamps around
/// async submission measure only enqueue cost. Host phases: `gap` is the time
/// between consecutive calls (the mux/serve layer per token), `submit` the
/// enqueue cost, `sync` the wait for the stream.
struct StepTiming {
    steps: u64,
    gap_ns: u64,
    submit_ns: u64,
    sync_ns: u64,
    upload_ms: f64,
    kernel_ms: f64,
    download_ms: f64,
    /// Stream-tail marks: start, uploads+re-arm done, interpreter done,
    /// token D2H done.
    ev: [CudaEvent; 4],
    last_end: Option<std::time::Instant>,
}

impl StepTiming {
    fn new(be: &CudaBackend) -> Result<StepTiming> {
        Ok(StepTiming {
            steps: 0,
            gap_ns: 0,
            submit_ns: 0,
            sync_ns: 0,
            upload_ms: 0.0,
            kernel_ms: 0.0,
            download_ms: 0.0,
            ev: [
                be.event_create(true)?,
                be.event_create(true)?,
                be.event_create(true)?,
                be.event_create(true)?,
            ],
            last_end: None,
        })
    }

    fn log_every(&mut self, n: u64) {
        if self.steps % n != 0 {
            return;
        }
        let us = |v: u64| v as f64 / 1000.0 / self.steps as f64;
        let ms = |v: f64| v / self.steps as f64;
        tracing::info!(
            steps = self.steps,
            gap_us = us(self.gap_ns),
            submit_us = us(self.submit_ns),
            sync_wait_ms = us(self.sync_ns) / 1000.0,
            dev_upload_us = ms(self.upload_ms) * 1000.0,
            dev_interp_ms = ms(self.kernel_ms),
            dev_download_us = ms(self.download_ms) * 1000.0,
            "step_slots means (host ns clocks + device CUDA events)"
        );
    }
}

/// PX-1 cross-request batched prefill state (`PLOW_PF_BATCH=1`). The two
/// device buffers are appended to the tensor table PAST the blob's handles:
/// the kernel sees them only where the host patches `t[6]` on prefill sites.
struct PfBatch {
    /// Per-row seq-slot map `i32[max bucket t]` — consumed by the KV-writing
    /// `HeadNormRope` sites (each packed row writes its own slot's ring).
    d_slot: DeviceMem,
    /// Request table `i32[1 + 4B]`: `[R, {q0, qlen, slot, kvlen} x R]` —
    /// consumed by the `FlashPrefill` sites (per-request-serial attention).
    d_req: DeviceMem,
    /// Tensor-table handles of the two buffers (blob handle count, +1).
    h_slot: u32,
    h_req: u32,
    /// `(t5, hd)` per flash site, harvested from a fused (`ns==1`) bucket —
    /// the per-layer `n.at` handles that turn a non-fused bucket's flash into
    /// the fused epilogue (its merge op is neutered).
    at_sites: Vec<(u32, u32)>,
    /// Host staging reused across launches (hot path — no per-launch alloc).
    slot_buf: Vec<i32>,
    req_buf: Vec<i32>,
}

/// One request's chunk inside a PX-1 batched prefill launch: rows
/// `prompt[c0..c0+len]` land at the pack's running row cursor, KV in seq-slot
/// `slot`. The mux packs chunks of N waiting requests under a token budget.
pub struct PfBatchReq<'a> {
    pub slot: usize,
    pub prompt: &'a [u32],
    pub c0: usize,
    pub len: usize,
}

struct PackedTokenReq<'a> {
    slot: usize,
    tokens: &'a [u32],
    c0: usize,
    prompt_len: usize,
}

/// Per-row device sampling request (plan stage 4). `temp <= 0` is greedy
/// (the device sampler writes the argmax, identical to `ARGMAX_FIN`), so a
/// spec array can carry greedy and stochastic rows together. `rng01` is the
/// request's per-step uniform draw (`seeded_unit`), so a fixed seed is
/// reproducible.
#[derive(Clone, Copy)]
pub struct DevSample {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub rng01: f32,
}

impl DevSample {
    /// Greedy row (device argmax == ARGMAX_FIN).
    pub fn greedy() -> Self {
        DevSample {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            rng01: 0.0,
        }
    }
}

/// Device sampler state: the `plow_sample` kernel + its per-slot parameter and
/// scratch buffers, all sized to the engine batch × vocab at load.
struct Sampler {
    f: KernelFn,
    _module: Module,
    /// Pinned staging for the 5 per-slot param arrays `[temp|topk|topp|minp|rng]`
    /// (each `batch` wide) — one async H2D per step.
    params: PinnedHost,
    /// Device copies of the 5 param arrays (contiguous, same layout as `params`).
    d_params: DeviceMem,
    /// `[batch][vocab]` f32 softmax-weight scratch the kernel reuses each pass.
    d_escratch: DeviceMem,
    #[allow(dead_code)]
    batch: usize,
}

/// Bounded device multi-step state (plan stage 5: `PLOW_MULTISTEP=K`). The
/// `plow_advance` kernel advances each active row's device-owned pos/kvlen and
/// appends its token to a `[batch][K]` ring BETWEEN decode launches, so a
/// K-token quantum runs with ONE host sync + ONE D2H instead of K — no
/// per-token host round trip. Requires a dynamic-kvrow decode cubin (the KV
/// write row comes from device pos) and greedy rows (device argmax).
struct MultiStep {
    f_advance: KernelFn,
    _module: Module,
    /// Token quantum K (steps per host round trip).
    quantum: usize,
    /// `[batch][K]` i32 token ring (device) + its pinned D2H staging.
    d_ring: DeviceMem,
    ring_host: PinnedHost,
    /// `[batch]` i32 active-row flags (device) + pinned staging.
    d_fed: DeviceMem,
    fed_host: PinnedHost,
}

/// Outcome of one [`GpuEngine::prefill_chunk`] call.
pub enum PrefillStep {
    /// The chunk advanced the prefill frontier to `.0` (< prompt len).
    Progress(usize),
    /// The prompt is fully consumed; `.0` is the first generated token
    /// (the exact postcondition of [`GpuEngine::prefill_slot`]).
    Done(u32),
}

fn is_checkpoint_tensor(
    index: usize,
    name: &str,
    packed: Option<&plow_asset::packed_prefill::Manifest>,
) -> bool {
    let declared_runtime = packed.is_some_and(|m| {
        (index == usize::from(m.slot) && name == "pf.request.slot")
            || (index == usize::from(m.request) && name == "pf.request.table")
            || m.maps.iter().any(|map| {
                index == usize::from(map.slots)
                    && name == format!("pf.request.maps.{}", map.original)
            })
    });
    !declared_runtime && packet::names::is_checkpoint_weight(name)
}

impl GpuEngine {
    /// Bring the model up on the device: blob + cubin + weights + decode
    /// tables. Slow (a 12B checkpoint is ~22 GiB of H2D) — called once at
    /// server startup, never on the request path.
    pub fn load(be: Arc<CudaBackend>, assets_dir: &Path, checkpoint_dir: &Path) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let load_prof = load_profile();
        let mut load_tim = load_prof.then(|| LoadTiming::new(t0));
        if load_prof {
            tracing::info!(
                start_ms = LoadTiming::epoch_ms(std::time::SystemTime::now()),
                "load stage enter GpuEngine::load"
            );
        }
        tracing::info!(
            assets = %assets_dir.display(),
            checkpoint = %checkpoint_dir.display(),
            device = %be.device_name(),
            "loading model onto GPU..."
        );

        // ---- blob ----
        let (pkt, raw, blob) = {
            let run = || -> Result<_> {
                let pkt = DevBlob::find_in_dir(assets_dir)?.ok_or_else(|| {
                    RuntimeError::Device(format!("no PLOWDEV blob in {}", assets_dir.display()))
                })?;
                let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
                    path: pkt.clone(),
                    source,
                })?;
                let blob = DevBlob::parse(&raw)?;
                if blob.progs.iter().flat_map(|p| &p.insts).any(|d| {
                    d.op == DevOp::MoeAiterFp8Pf as u16
                        || d.op == DevOp::IndexTpPf as u16
                        || d.op == DevOp::GemmLtPf as u16
                        || (d.op == DevOp::MlaMergeFold as u16 && d.i[5] != 0)
                }) {
                    return Err(RuntimeError::Device(
                        "native AITER MoE, TP indexer and hipBLASLt projections are supported only on gfx942".into(),
                    ));
                }
                if blob.progs.iter().flat_map(|p| &p.insts).any(|d| {
                    (d.op == DevOp::IndexSelect as u16 && (d.i[3] != 0 || d.i[4] != 0))
                        || (matches!(
                            DevOp::from_u16(d.op),
                            Some(DevOp::FlashMlaDecodeFp8 | DevOp::FlashMlaPrefillFp8)
                        ) && d.fj[1] != 0)
                }) {
                    return Err(RuntimeError::Device(
                        "batched DSA selection and sparse FP8 MLA are currently supported only by the AMD interpreter"
                            .into(),
                    ));
                }
                Ok((pkt, raw, blob))
            };
            if let Some(tm) = load_tim.as_mut() {
                let (r, ms) = tm.phase("blob_find_read_parse", run);
                tm.blob_ms = ms;
                r?
            } else {
                run()?
            }
        };
        tracing::info!(
            blob = %pkt.display(),
            tensors = blob.tensors.len(),
            n_cu = blob.n_cu,
            programs = blob.progs.len(),
            "parsed PLOWDEV blob"
        );
        let live_kv_manifest = crate::memory::vmm::LiveKvLayout::manifest(&blob, &raw)?;
        let packed_prefill_metadata = blob
            .reserved_metadata(&raw, plow_asset::packed_prefill::SECTION)?
            .map(|bytes| {
                serde_json::from_slice::<plow_asset::packed_prefill::Manifest>(bytes)
                    .map_err(|e| RuntimeError::Rejected(format!("packed prefill metadata: {e}")))
            })
            .transpose()?;
        let mixed_packet = if blob
            .sections
            .iter()
            .any(|section| section.name == plow_asset::mixed_step::SECTION)
        {
            crate::exec::mixed_packet::load_from_devblob(
                &blob,
                &raw,
                plow_asset::mixed_step::PayloadKind::Cubin,
                |object, name| Ok(cubin::global_u32(object.bytes, name)),
            )?
        } else {
            None
        };
        if let Some(pack) = &packed_prefill_metadata {
            let live = live_kv_manifest.as_ref().ok_or_else(|| {
                RuntimeError::Rejected("packed prefill requires compiled LIVE contract".into())
            })?;
            blob.with_packet_view(|p| pack.validate(p, live))
                .map_err(RuntimeError::Rejected)?;
        }
        let config = RuntimeConfig::get();
        let prefix_layout = crate::memory::vmm::VmmOps::granularity(be.as_ref())
            .ok()
            .and_then(|granularity| {
                Self::select_vmm_prefix_layout(
                    &blob,
                    checkpoint_dir,
                    config,
                    be.compute_capability(),
                    granularity,
                )
            });
        let prefix_requested = config.nv_vmm_prefix() == Some(true) || prefix_layout.is_some();
        let unified_packed = config.token_batch
            && !config.fusion
            && prefix_layout.is_some()
            && packed_prefill_metadata.is_some();
        let packed_prefix = prefix_requested && (config.pf_batch || unified_packed);
        if packed_prefix && (prefix_layout.is_none() || packed_prefill_metadata.is_none()) {
            return Err(RuntimeError::Rejected(
                "packed prefix reuse requires compiled packed-prefill metadata and a valid VMM layout"
                    .into(),
            ));
        }
        tracing::info!(
            requested = ?config.nv_vmm_prefix(),
            selected = prefix_layout.is_some(),
            "VMM prefix cache selection"
        );
        let packed_prefill = if prefix_requested && !packed_prefix {
            if packed_prefill_metadata.is_some() {
                tracing::info!("packed prefill disabled because prefix reuse is active");
            }
            None
        } else {
            packed_prefill_metadata.clone()
        };
        let decode_objects = decode_object::parse(&blob, &raw)?;
        let prepared_contexts = decode_context::prepare(&blob, &raw, assets_dir)?;
        if blob
            .with_packet_view(plow_asset::splitk::validate)
            .map_err(RuntimeError::Rejected)?
            .is_some()
            && (decode_objects.is_none() || live_kv_manifest.is_none())
        {
            return Err(RuntimeError::Rejected(
                "splitK packets require compiled decode object and LIVE KV manifests".into(),
            ));
        }
        let segment_roles = segment_role_metadata(&blob, &raw)?;
        let nv_config = &RuntimeConfig::get().nv;
        let decode_roles = segment_roles
            .as_ref()
            .and_then(|r| r.program(blob.progs.len() - 1))
            .map(|p| p.roles.clone())
            .unwrap_or_default();
        let native_enabled = decode_roles.contains(&plow_asset::segment_roles::NATIVE_DECODE_TC);
        let cublaslt_enabled = decode_roles
            .iter()
            .copied()
            .any(plow_asset::segment_roles::is_projection);
        let decode_packet_roles = decode_roles
            .iter()
            .any(|&role| {
                matches!(
                    role,
                    plow_asset::segment_roles::GEMV_CTA512
                        | plow_asset::segment_roles::FP8_M1
                        | plow_asset::segment_roles::MXFP4_MOE
                )
            })
            .then(|| decode_roles.clone())
            .unwrap_or_default();
        let kv_maps = kv_tensor_maps(&blob.tensors, &blob.gen, blob.decode_prog()?.t as usize)?;
        let recurrent = recurrent_state_layout(&blob.tensors, blob.decode_prog()?.t as usize)?;
        let configured_multistep = RuntimeConfig::get().nv_multistep();
        let multistep_disabled_by_decode = decode_objects.is_some()
            || prepared_contexts.is_some()
            || !decode_packet_roles.is_empty()
            || cublaslt_enabled
            || recurrent.is_some();
        let effective_multistep = if multistep_disabled_by_decode {
            if configured_multistep > 1 {
                tracing::info!(
                    configured = configured_multistep,
                    "multistep disabled by incompatible decode mode"
                );
            }
            0
        } else {
            configured_multistep
        };
        if decode_packet_roles.contains(&plow_asset::segment_roles::FP8_M1) {
            plow_asset::fp8_m1_role::options(effective_multistep, cublaslt_enabled, false)
                .map_err(RuntimeError::Rejected)?;
        }
        if decode_objects.is_some() || prepared_contexts.is_some() {
            decode_object::check_options(
                segment_roles.as_ref().is_some_and(|roles| {
                    roles
                        .programs
                        .iter()
                        .any(|program| program.index >= blob.prefill_progs().len())
                }),
                cublaslt_enabled,
                effective_multistep as usize,
                nv_config.cubin.is_some() || nv_config.kernel.is_some() || nv_config.smem.is_some(),
            )?;
        }
        let single_bound_decode = decode_objects.is_some() && blob.decode_progs().len() == 1;
        let mut select_decode_rungs = if cublaslt_enabled {
            if decode_objects.is_some() || prepared_contexts.is_some() {
                return Err(RuntimeError::Rejected(
                    "cuBLASLt decode cannot use decode objects or context variants".into(),
                ));
            }
            validate_cublaslt_ladder(&blob, segment_roles.as_ref().expect("library roles"))?
        } else {
            single_bound_decode || validate_decode_ladder(&blob)?
        };
        if recurrent.is_some() {
            if prefix_requested {
                return Err(RuntimeError::Rejected(
                    "recurrent state does not support prefix caching".into(),
                ));
            }
            if !recurrent.as_ref().unwrap().tensors.is_empty()
                && !blob.prefill_progs().is_empty()
                && (blob
                    .prefill_progs()
                    .iter()
                    .any(|p| !p.insts.iter().any(|i| i.op == DevOp::QwenGdnPrefill as u16)))
            {
                return Err(RuntimeError::Rejected(
                    "recurrent state currently requires decode-only prompt consumption".into(),
                ));
            }
        }
        let unsupported_hd128_fp8_kv = blob.progs.iter().flat_map(|p| &p.insts).any(|inst| {
            (inst.op == DevOp::HeadNormRopeFp8 as u16 && inst.i[2] == 128)
                || (matches!(
                    DevOp::from_u16(inst.op),
                    Some(DevOp::FlashPrefillFp8 | DevOp::FlashDecodeFp8)
                ) && inst.i[6] == 128)
        });
        if unsupported_hd128_fp8_kv {
            return Err(RuntimeError::Device(
                "hd128 Qwen packets with fp8 KV are not supported by the NVIDIA interpreter; \
                 use bf16 KV (weight-FP8 checkpoints remain supported)"
                    .into(),
            ));
        }
        let cc = be.compute_capability();
        let profile = interpreter_profile(cc).ok_or_else(|| {
            RuntimeError::Device(format!(
                "{} compute capability {}.{} has no persistent interpreter",
                be.device_name(),
                cc.0,
                cc.1
            ))
        })?;

        // ---- module ----
        // Selected BY CONTENT: each candidate's own ELF says which SM it targets
        // and which entry points it has, so a bundle whose decode/prefill files
        // are misnamed still serves, and a wrong-arch image is refused here with
        // a message instead of by the driver with an opaque code.
        let want_sm = cc.0 * 10 + cc.1;
        let (module, f, _kname, smem, grid, _dec_source, _image_len) = {
            let run = || -> Result<_> {
                let selected = if let Some(metadata) = &decode_objects {
                    let spec = &metadata.objects
                        [&metadata.programs.last().expect("validated coverage").object];
                    Some(decode_object::image(spec, assets_dir, &profile, want_sm)?)
                } else {
                    resolve_interp_image(assets_dir, &blob, &raw, &profile, want_sm, Role::Decode)?
                };
                let dec = selected.ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "no sm_{want_sm} decode interpreter object for {} in {} — expected a cubin \
                     carrying `{}` (embedded in the blob, or a file; {} is only the conventional \
                     name, the symbol table decides). Candidates found:\n{}",
                        be.device_name(),
                        assets_dir.display(),
                        profile.decode_symbol,
                        profile.decode_file,
                        describe_candidates(assets_dir),
                    ))
                })?;
                let image = dec.image;
                let image_len = image.len();
                let dec_source = dec.source.clone();
                let module = DecodeModule::load(&be, &image)?;
                check_norm_weight_offset(&be, &module, &blob)?;
                let kname = crate::config::RuntimeConfig::get()
                    .nv
                    .kernel
                    .clone()
                    .unwrap_or(dec.entry);
                let f = be.get_function(&module, &kname)?;
                Self::check_packet_pairing(&be, &module, assets_dir)?;
                let smem: u32 = match crate::config::RuntimeConfig::get().nv.smem {
                    Some(v) => v,
                    None => be
                        .module_global_u32(&module, "plow_arena_bytes")?
                        .unwrap_or(12352),
                };
                if smem > 48 * 1024 {
                    be.set_max_dynamic_smem(f, smem)?;
                }
                let occ = be.occupancy_blocks_per_sm(f, BLOCK, smem as usize)?;
                let grid = decode_object::initial_grid(
                    decode_objects.as_ref(),
                    blob.n_cu,
                    be.sm_count(),
                    occ,
                )?;
                Ok((module, f, kname, smem, grid, dec_source, image_len, occ))
            };
            if let Some(tm) = load_tim.as_mut() {
                let (r, ms) = tm.phase("interp_module", run);
                tm.module_ms = ms;
                let (module, f, kname, smem, grid, dec_source, image_len, occ) = r?;
                tracing::info!(
                    profile = profile.tag,
                    source = %dec_source,
                    kernel = %kname,
                    grid,
                    smem,
                    occ_per_sm = occ,
                    cubin_bytes = image_len,
                    "interpreter module loaded"
                );
                (module, f, kname, smem, grid, dec_source, image_len)
            } else {
                let (module, f, kname, smem, grid, dec_source, image_len, occ) = run()?;
                tracing::info!(
                    profile = profile.tag,
                    source = %dec_source,
                    kernel = %kname,
                    grid,
                    smem,
                    occ_per_sm = occ,
                    cubin_bytes = image_len,
                    "interpreter module loaded"
                );
                (module, f, kname, smem, grid, dec_source, image_len)
            }
        };
        let bound_objects = decode_objects
            .as_ref()
            .map(|metadata| {
                decode_object::bind(
                    metadata, &blob, assets_dir, &be, &module, f, &profile, want_sm,
                )
            })
            .transpose()?;
        let gemv_mm_cap = be.module_global_u32(&module, "plow_gemv_mm_cap")?;
        if recurrent.is_some()
            && blob
                .decode_prog()?
                .insts
                .iter()
                .any(|d| d.op == DevOp::GemmFp8 as u16)
        {
            check_qwen_w8a8_capability(
                false,
                blob.decode_prog()?.t,
                be.module_global_u32(&module, "plow_fp8_m1_arm")?,
            )?;
        }
        if gemv_mm_cap == Some(0) {
            return Err(RuntimeError::Device(
                "decode interpreter advertises GV_MM_MAX=0 — its GEMV row walk cannot make progress"
                    .into(),
            ));
        }
        if select_decode_rungs
            && blob.decode_progs()[0].t == 1
            && be.module_global_u32(&module, "plow_dyn_kvrow")? != Some(1)
        {
            select_decode_rungs = false;
        }
        if prepared_contexts.is_some()
            && be.module_global_u32(&module, "plow_dyn_kvrow")? != Some(1)
        {
            return Err(RuntimeError::Rejected(
                "context base requires dynamic KV ABI1".into(),
            ));
        }
        if (decode_objects.is_some() || prepared_contexts.is_some()) && !select_decode_rungs {
            return Err(RuntimeError::Rejected(
                "decode objects require qualified narrow dispatch".into(),
            ));
        }
        let decode_rungs = blob.decode_rungs();
        if decode_rungs.len() > 1 && !select_decode_rungs {
            tracing::warn!(
                ?decode_rungs,
                "decode ladder retains widest execution: narrower addressing is not qualified"
            );
        }
        if let Some(cap) = gemv_mm_cap {
            let widest = decode_rungs.last().copied().unwrap_or(1);
            tracing::info!(
                ?decode_rungs,
                gemv_mm_cap = cap,
                gemv_weight_passes = widest.div_ceil(cap),
                "NVIDIA decode capacity metadata"
            );
        } else if decode_rungs.len() > 1 {
            tracing::warn!(
                ?decode_rungs,
                "decode ladder uses a legacy NVIDIA object without GV_MM_MAX metadata"
            );
        }

        // ---- VMM allocation and prefix sharing ----
        let vmm = {
            let run = || {
                let config = RuntimeConfig::get();
                let live = config.nv_live_kv_enabled(
                    packed_prefill.is_some(),
                    live_kv_manifest.as_ref().is_some_and(|manifest| {
                        manifest.caches.iter().any(|cache| cache.window == 0)
                    }),
                    prefix_requested,
                );
                if live && !config.nv_vmm_live() {
                    tracing::info!("live KV allocation enabled by packet metadata");
                }
                let rings = config.nv_vmm_live_rings();
                if rings && !live {
                    return Err(RuntimeError::Rejected(
                        "live rings require PLOW_VMM_LIVE=1".into(),
                    ));
                }
                if live {
                    if prefix_requested {
                        return Err(RuntimeError::Rejected(
                            "live KV allocation requires prefix caching off".into(),
                        ));
                    }
                    Self::vmm_live_bringup(&be, &blob, rings, live_kv_manifest.as_ref()).map(Some)
                } else {
                    Ok(Self::vmm_bringup(&be, &blob, prefix_layout))
                }
            };
            if let Some(tm) = load_tim.as_mut() {
                let (v, ms) = tm.phase("vmm_bringup", run);
                tm.vmm_ms = ms;
                v
            } else {
                run()
            }
        }?;

        // ---- weight slab ----
        // Where a tensor's storage comes from, decided once so the sizing pass
        // and the upload loop cannot disagree about it.
        let vmm_va_of = |id: usize| -> Option<u64> {
            let v = vmm.as_ref()?;
            if let Some(&(_, layer, tensor)) = v.tensor_tracks.iter().find(|&&(i, _, _)| i == id) {
                v.kv.tensor_va(layer, tensor)
            } else {
                v.rings.as_ref().and_then(|rings| rings.tensor_va(id))
            }
        };
        let slab_bytes: u64 = blob
            .tensors
            .iter()
            .enumerate()
            .filter(|(id, _)| vmm_va_of(*id).is_none())
            .map(|(_, td)| slab_pad(td.bytes))
            .sum();
        // Brought up BEFORE the checkpoint opens: the VMM reserve returns in
        // µs and its mapper then commits pages concurrently with the open,
        // the prefetch spawn, and the upload itself.
        let mk_slab = || -> WeightSlab {
            if slab_bytes == 0 || !crate::asset::checkpoint::weight_slab_enabled() {
                return WeightSlab::PerTensor;
            }
            if crate::asset::checkpoint::weight_vmm_enabled() {
                match crate::memory::vmm::VmmSlab::new(
                    Arc::clone(&be) as Arc<dyn crate::memory::vmm::VmmOps>,
                    slab_bytes,
                    WEIGHT_SLAB_CHUNK,
                ) {
                    Ok(s) => return WeightSlab::Vmm(s),
                    Err(e) => tracing::warn!(
                        bytes = slab_bytes,
                        error = %e,
                        "vmm weight slab refused — falling back to flat cuMemAlloc"
                    ),
                }
            }
            match be.alloc(0, slab_bytes) {
                Ok(m) => WeightSlab::Flat(m),
                Err(e) => {
                    tracing::warn!(
                        bytes = slab_bytes,
                        error = %e,
                        "single weight allocation refused — falling back to per-tensor alloc"
                    );
                    WeightSlab::PerTensor
                }
            }
        };
        let weight_slab = if let Some(tm) = load_tim.as_mut() {
            let (s, ms) = tm.phase("weight_slab", mk_slab);
            tm.slab_ms = ms;
            if matches!(s, WeightSlab::Flat(_)) {
                tm.note_alloc(slab_bytes, ms, false);
            }
            s
        } else {
            mk_slab()
        };

        // ---- weights ----
        let t_weights = std::time::Instant::now();
        let mut ckpt_sub = load_prof.then(crate::asset::checkpoint::CheckpointOpenTiming::default);
        let ckpt = {
            if let Some(tm) = load_tim.as_mut() {
                let (c, ms) = tm.phase("Checkpoint::open", || {
                    Checkpoint::open_with_timing(checkpoint_dir, ckpt_sub.as_mut())
                        .map(std::sync::Arc::new)
                });
                tm.ckpt_open_ms = ms;
                if let Some(sub) = ckpt_sub.as_ref() {
                    tm.ckpt_scan_ms = sub.scan_ms;
                    tm.ckpt_mmap_ms = sub.mmap_ms;
                    tm.ckpt_meta_ms = sub.meta_ms;
                    tm.ckpt_index_ms = sub.index_ms;
                }
                c?
            } else {
                Checkpoint::open(checkpoint_dir).map(std::sync::Arc::new)?
            }
        };
        if decode_packet_roles.contains(&plow_asset::segment_roles::FP8_M1) {
            validate_fp8_role_checkpoint(
                segment_roles.as_ref().expect("role metadata"),
                &blob,
                &ckpt,
            )?;
        }
        tracing::info!(
            checkpoint = %checkpoint_dir.display(),
            "checkpoint opened, starting weight upload to GPU..."
        );

        let mut pipe = {
            let run = || UploadPipe::new(&be, load_prof);
            if let Some(tm) = load_tim.as_mut() {
                let (p, ms) = tm.phase("UploadPipe::new", run);
                tm.pipe_setup_ms = ms;
                let p = p?;
                if let Some(pt) = p.timing.as_ref() {
                    tm.absorb_pipe_setup(pt);
                }
                p
            } else {
                run()?
            }
        };

        let depth = crate::asset::checkpoint::prefetch_depth();
        let pref_threads = crate::asset::checkpoint::prefetch_threads();
        let prefetch_stats =
            load_prof.then(|| std::sync::Arc::new(crate::asset::checkpoint::PrefetchStats::new()));
        let t_pref_wall = std::time::Instant::now();
        let sys_pref = std::time::SystemTime::now();
        let prefetch = {
            if let Some(tm) = load_tim.as_mut() {
                let stats = prefetch_stats.clone();
                let (p, ms) = tm.phase("Prefetcher::start", || {
                    crate::asset::checkpoint::Prefetcher::start(
                        std::sync::Arc::clone(&ckpt),
                        pref_threads,
                        depth,
                        stats,
                    )
                });
                tm.prefetch_spawn_ms = ms;
                p
            } else {
                crate::asset::checkpoint::Prefetcher::start(
                    std::sync::Arc::clone(&ckpt),
                    pref_threads,
                    depth,
                    None,
                )
            }
        };
        // Runs `depth` WEIGHT tensors ahead of the copy over the same list in the
        // same order, so the cursor only moves forward and each tensor is queued
        // exactly once. Non-weights are skipped so the depth counts reads, not
        // table entries — most of a blob's tensors are scratch that touches no
        // checkpoint at all. TP1 here, so the touched bytes are the whole tensor.
        let prefetch_ahead = |cur: &mut usize, budget: usize| {
            let Some(pool) = prefetch.as_ref() else {
                return;
            };
            let mut n = 0;
            while *cur < blob.tensors.len() && n < budget {
                let td = &blob.tensors[*cur];
                *cur += 1;
                if !is_checkpoint_tensor(*cur - 1, &td.name, packed_prefill_metadata.as_ref()) {
                    continue;
                }
                if let Some(s) = ckpt.span(&td.name, 0, td.bytes as usize) {
                    pool.push(s);
                }
                n += 1;
            }
        };
        let mut pf = 0usize;
        prefetch_ahead(&mut pf, depth);

        let gen_by_tensor: std::collections::HashMap<u32, &packet::rope::GenTensor> =
            blob.gen.iter().map(|g| (g.tensor, g)).collect();

        // upload_all wall covers the tensor loop (exclusive phase).
        let t_upload = std::time::Instant::now();
        let sys_upload = std::time::SystemTime::now();
        let upload_t0_ms = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let mut slab_off: u64 = 0;
        let mut slab_wait_ms = 0f64;

        let mut devp: Vec<DeviceMem> = Vec::with_capacity(blob.tensors.len());
        let (mut t_ids, mut t_pos, mut t_kvlen, mut t_logits) = (None, None, None, None);
        let (mut wb, mut kvb, mut nw) = (0u64, 0u64, 0usize);
        let mut upload_all = || -> Result<()> {
            for (i, td) in blob.tensors.iter().enumerate() {
                let vmm_va = vmm_va_of(i);
                let packet_cache = vmm.as_ref().is_some_and(|v| v.cache_tensors.contains(&i));
                let t_alloc =
                    (load_prof && vmm_va.is_none() && matches!(weight_slab, WeightSlab::PerTensor))
                        .then(std::time::Instant::now);
                let mem = match (vmm_va, &weight_slab) {
                    (Some(va), _) => DeviceMem::view(va, td.bytes),
                    (None, WeightSlab::Vmm(slab)) => {
                        let m = DeviceMem::view(slab.base() + slab_off, td.bytes);
                        slab_off += slab_pad(td.bytes);
                        m
                    }
                    (None, WeightSlab::Flat(slab)) => {
                        let m = DeviceMem::view(slab.base + slab_off, td.bytes);
                        slab_off += slab_pad(td.bytes);
                        m
                    }
                    (None, WeightSlab::PerTensor) => be.alloc(0, td.bytes)?,
                };
                if let (Some(t), Some(tm)) = (t_alloc, load_tim.as_mut()) {
                    tm.note_alloc(td.bytes, t.elapsed().as_secs_f64() * 1e3, true);
                }
                match td.name.as_str() {
                    "in.ids" => t_ids = Some(i),
                    "in.pos" => t_pos = Some(i),
                    "in.kvlen" => t_kvlen = Some(i),
                    "act.logits" => t_logits = Some(i),
                    _ => {}
                }
                if td.name.starts_with("kv.") || packet_cache {
                    kvb += td.bytes;
                }
                if packet::names::is_host_filled_table(&td.name) {
                    return Err(RuntimeError::Device(format!(
                        "packet declares the host-filled expert pointer table `{}`, which \
                         this engine cannot fill (no `bind_packed_experts` on the CUDA \
                         path; only Gemma's fused `moe.ewt.`/`moe.est.` tables are wired). \
                         Leaving it zeroed would route every token to expert 0.",
                        td.name
                    )));
                }
                if !packet_cache
                    && is_checkpoint_tensor(i, &td.name, packed_prefill_metadata.as_ref())
                {
                    let src = ckpt.tensor(&td.name).ok_or_else(|| {
                        RuntimeError::Device(format!("MISSING WEIGHT: {}", td.name))
                    })?;
                    if src.len() as u64 != td.bytes {
                        return Err(RuntimeError::Device(format!(
                            "SIZE MISMATCH {} (want {} got {})",
                            td.name,
                            td.bytes,
                            src.len()
                        )));
                    }
                    tracing::debug!(
                        tensor = %td.name,
                        bytes = td.bytes,
                        mib = td.bytes / (1 << 20),
                        "uploading weight tensor"
                    );
                    prefetch_ahead(&mut pf, 1);
                    for (o, chunk) in src.chunks(STAGE).enumerate() {
                        let dst = mem.base + (o * STAGE) as u64;
                        slab_commit_wait(
                            &weight_slab,
                            dst + chunk.len() as u64,
                            &mut slab_wait_ms,
                        )?;
                        if pipe.is_direct() {
                            // SAFETY: `chunk` borrows the checkpoint mmap
                            // (`ckpt`, whose Arc outlives pipe.finish()).
                            unsafe { pipe.push_direct(dst, chunk)? }
                        } else {
                            pipe.push(dst, chunk)?;
                        }
                    }
                    wb += td.bytes;
                    nw += 1;
                } else if let Some(r) = &td.init {
                    for (o, chunk) in blob.init[r.clone()].chunks(STAGE).enumerate() {
                        let dst = mem.base + (o * STAGE) as u64;
                        slab_commit_wait(
                            &weight_slab,
                            dst + chunk.len() as u64,
                            &mut slab_wait_ms,
                        )?;
                        if pipe.is_direct() {
                            // SAFETY: `chunk` borrows `blob.init`, which lives
                            // past pipe.finish().
                            unsafe { pipe.push_direct(dst, chunk)? }
                        } else {
                            pipe.push(dst, chunk)?;
                        }
                    }
                } else if let Some(g) = gen_by_tensor.get(&(i as u32)) {
                    let data = g.generate().ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "devblob: gen recipe for `{}` has unknown kind {}",
                            td.name, g.kind
                        ))
                    })?;
                    if data.len() as u64 != td.bytes {
                        return Err(RuntimeError::Device(format!(
                            "devblob: gen recipe for `{}` produced {} B, decl says {}",
                            td.name,
                            data.len(),
                            td.bytes
                        )));
                    }
                    for (o, chunk) in data.chunks(STAGE).enumerate() {
                        let dst = mem.base + (o * STAGE) as u64;
                        slab_commit_wait(
                            &weight_slab,
                            dst + chunk.len() as u64,
                            &mut slab_wait_ms,
                        )?;
                        pipe.push(dst, chunk)?;
                    }
                    tracing::debug!(
                        tensor = %td.name, bytes = td.bytes, kind = g.kind,
                        "materialised generated tensor"
                    );
                } else if vmm_va.is_none() && !packet_cache && !td.name.starts_with("kv.") {
                    slab_commit_wait(&weight_slab, mem.base + td.bytes, &mut slab_wait_ms)?;
                    let t_ms = load_prof.then(std::time::Instant::now);
                    be.memset_d8(mem.base, 0, td.bytes as usize)?;
                    if let (Some(t), Some(tm)) = (t_ms, load_tim.as_mut()) {
                        tm.memset_ms += t.elapsed().as_secs_f64() * 1e3;
                    }
                }
                devp.push(mem);
            }
            Ok(())
        };
        let enqueued = upload_all();
        drop(upload_all);
        if enqueued.is_ok() && !matches!(weight_slab, WeightSlab::PerTensor) {
            debug_assert_eq!(
                slab_off, slab_bytes,
                "weight slab carve did not consume exactly the sized span"
            );
        }
        if let Some(tm) = load_tim.as_mut() {
            tm.slab_wait_ms = slab_wait_ms;
            let ms = t_upload.elapsed().as_secs_f64() * 1e3;
            tm.upload_all_ms = ms;
            tm.n_tensors_uploaded = nw;
            tm.spans
                .push(("upload_all".into(), upload_t0_ms, tm.ms_since_t0()));
            tm.log_stage("upload_all", sys_upload, std::time::SystemTime::now(), ms);
        }

        let sys_fin = std::time::SystemTime::now();
        let fin_t0 = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let t_fin = std::time::Instant::now();
        // finish() runs even when the enqueue loop failed: it synchronizes the
        // pipe stream, and the error return below tears down the pinned
        // buffers and (VMM slab) unmaps device ranges that any still-in-flight
        // H2D would otherwise touch.
        let finished = pipe.finish();
        let uploaded = enqueued.and(finished);
        if let Some(tm) = load_tim.as_mut() {
            let finish_ms = t_fin.elapsed().as_secs_f64() * 1e3;
            tm.pipe_finish_ms = finish_ms;
            if let Some(pt) = pipe.take_timing() {
                tm.absorb_pipe_upload(&pt);
            }
            tm.spans
                .push(("pipe.finish".into(), fin_t0, tm.ms_since_t0()));
            tm.log_stage(
                "pipe.finish",
                sys_fin,
                std::time::SystemTime::now(),
                finish_ms,
            );
        }
        drop(pipe);
        uploaded?;
        // The slab tail (KV is never written at load) must be committed before
        // the engine serves — VMM has no demand paging, and the first prefill
        // writes KV rows. In practice the mapper (≈13 GiB/s) finished long
        // before the upload (≈6 GiB/s) did, so this join is ≈0.
        if let WeightSlab::Vmm(slab) = &weight_slab {
            let t_join = std::time::Instant::now();
            slab.wait_mapped(slab_bytes)?;
            if let Some(tm) = load_tim.as_mut() {
                tm.slab_join_ms = t_join.elapsed().as_secs_f64() * 1e3;
            }
        }

        let had_prefetch = prefetch.is_some();
        let join_t0 = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let t_join = std::time::Instant::now();
        drop(prefetch);
        if let Some(tm) = load_tim.as_mut() {
            let join_ms = t_join.elapsed().as_secs_f64() * 1e3;
            tm.prefetch_join_ms = join_ms;
            tm.spans
                .push(("prefetch_join".into(), join_t0, tm.ms_since_t0()));
            tm.log_stage(
                "prefetch_join",
                sys_pref,
                std::time::SystemTime::now(),
                join_ms,
            );
            if had_prefetch {
                use std::sync::atomic::Ordering;
                let wall_ms = t_pref_wall.elapsed().as_secs_f64() * 1e3;
                let (ns, bytes) = prefetch_stats
                    .as_ref()
                    .map(|s| {
                        (
                            s.ns.load(Ordering::Relaxed),
                            s.bytes.load(Ordering::Relaxed),
                        )
                    })
                    .unwrap_or((0, 0));
                tm.prefetch_wall_ms = wall_ms;
                tm.prefetch_worker_ms = ns as f64 / 1e6;
                tm.prefetch_bytes = bytes;
                tm.prefetch_workers = pref_threads;
                tm.log_stage_bytes(
                    "Prefetcher populate (Disk→RAM wall)",
                    sys_pref,
                    std::time::SystemTime::now(),
                    wall_ms,
                    bytes,
                );
                tracing::info!(
                    stage = "Prefetcher populate (parallel worker time)",
                    elapsed_ms = format!("{:.3}", tm.prefetch_worker_ms).as_str(),
                    workers = pref_threads,
                    cumulative_ms = format!("{:.3}", tm.cum_ms()).as_str(),
                    "load stage"
                );
            }
        }
        let weight_elapsed = t_weights.elapsed();
        let weight_gib = wb as f64 / (1u64 << 30) as f64;
        let throughput = if weight_elapsed.as_secs_f64() > 0.0 {
            weight_gib / weight_elapsed.as_secs_f64()
        } else {
            0.0
        };
        tracing::info!(
            tensors = nw,
            weight_gib = format!("{weight_gib:.2}").as_str(),
            kv_gib = format!("{:.2}", kvb as f64 / (1u64 << 30) as f64).as_str(),
            elapsed_s = format!("{:.1}", weight_elapsed.as_secs_f64()).as_str(),
            throughput_gib_s = format!("{throughput:.2}").as_str(),
            "checkpoint weights uploaded to GPU"
        );
        let (t_ids, t_pos, t_kvlen, t_logits) = match (t_ids, t_pos, t_kvlen, t_logits) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => {
                return Err(RuntimeError::Device(
                    "blob is missing in.ids/in.pos/in.kvlen/act.logits".into(),
                ))
            }
        };

        // ---- Gemma-4 26B-A4B fused-MoE expert pointer tables ----
        // Port of the chat harness's name-driven fill (gemma4_sm120_chat.cu §MoE):
        // for each `moe.ewt.{l}` slot, resolve the two FUSED expert tensors by name
        // suffix, derive E and the per-expert byte strides from tensor sizes alone,
        // and upload the [E][2] {gate_up, down} base table the SM indexes
        // (orch::moe::build_fused_expert_table). fp8 packets additionally fill
        // `moe.est.{l}` with the per-row scale bases. Guarded on the `moe.ewt.`
        // prefix, so dense (12B/31B/Qwen) blobs are untouched. Suffix scans keep
        // the LAST match, mirroring the harness (an fp8 twin shadows a bf16 name).
        let t_moe = std::time::Instant::now();
        let moe_t0 = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let sys_moe = std::time::SystemTime::now();
        for i in 0..blob.tensors.len() {
            let Some(layer) = blob.tensors[i].name.strip_prefix("moe.ewt.") else {
                continue;
            };
            let layer = layer.to_string();
            let find = |suf: &str| blob.tensors.iter().rposition(|t| t.name.ends_with(suf));
            let suf_gu = format!("layers.{layer}.experts.gate_up_proj");
            let suf_dn = format!("layers.{layer}.experts.down_proj");
            let (Some(gu), Some(dn)) = (find(&suf_gu), find(&suf_dn)) else {
                return Err(RuntimeError::Device(format!(
                    "MoE: layer {layer} missing fused expert tensor(s)"
                )));
            };
            let e = blob.tensors[i].bytes / 16; // ewt = E*2 u64
            let table = crate::orch::moe::build_fused_expert_table(
                devp[gu].base,
                devp[dn].base,
                e as u32,
                blob.tensors[gu].bytes / e,
                blob.tensors[dn].bytes / e,
            );
            be.upload(&devp[i], 0, bytemuck::cast_slice(&table))?;
            if blob.tensors[gu].name.starts_with("fp8/") {
                let est_name = format!("moe.est.{layer}");
                let suf_gs = format!("layers.{layer}.experts.gate_up_proj_scale");
                let suf_ds = format!("layers.{layer}.experts.down_proj_scale");
                let est = blob.tensors.iter().position(|t| t.name == est_name);
                let (Some(est), Some(gs), Some(ds)) = (est, find(&suf_gs), find(&suf_ds)) else {
                    return Err(RuntimeError::Device(format!(
                        "MoE fp8: layer {layer} missing expert scale tensor/table"
                    )));
                };
                let stable = crate::orch::moe::build_fused_expert_table(
                    devp[gs].base,
                    devp[ds].base,
                    e as u32,
                    blob.tensors[gs].bytes / e,
                    blob.tensors[ds].bytes / e,
                );
                be.upload(&devp[est], 0, bytemuck::cast_slice(&stable))?;
            }
        }
        if let Some(tm) = load_tim.as_mut() {
            let ms = t_moe.elapsed().as_secs_f64() * 1e3;
            tm.moe_ms = ms;
            tm.spans
                .push(("moe_tables".into(), moe_t0, tm.ms_since_t0()));
            tm.log_stage("moe_tables", sys_moe, std::time::SystemTime::now(), ms);
        }

        // ---- Cross-request prefill buffers ----
        // A validated packet manifest owns packet-declared tables. Legacy
        // packets retain the explicit runtime opt-in and appended tables.
        let t_decode = std::time::Instant::now();
        let decode_t0 = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let sys_decode = std::time::SystemTime::now();
        let pf_batch_env = crate::config::RuntimeConfig::get().pf_batch;
        let pf_max_t_blob = blob
            .prefill_progs()
            .iter()
            .map(|g| g.t as usize)
            .max()
            .unwrap_or(0);
        let dbatch_blob = blob.decode_prog().map(|g| g.t as usize).unwrap_or(1);
        let pf_batch_requested = packed_prefill.is_some() || pf_batch_env;
        let pf_bufs = if let Some(p) = &packed_prefill {
            Some((
                DeviceMem::view(devp[p.slot as usize].base, devp[p.slot as usize].len),
                DeviceMem::view(devp[p.request as usize].base, devp[p.request as usize].len),
            ))
        } else if pf_batch_requested && pf_max_t_blob > 0 {
            Some((
                be.alloc(0, (pf_max_t_blob * 4) as u64)?,
                be.alloc(0, ((1 + 4 * dbatch_blob) * 4) as u64)?,
            ))
        } else {
            None
        };

        // ---- GEN_TMAP_BF16 re-encode (sm_90a TMA prefill GEMM) ----
        // The upload loop staged these tensors' 128-byte ZERO placeholders (a tensormap
        // is a function of the TARGET's device address, unknowable before the carve).
        // devp is complete here, so encode the real descriptor over the target and
        // overwrite the placeholder in place — before d_tens is built, so the handle
        // the packet carries in i[6]/i[7] resolves to finished bytes. A TMA-bearing
        // packet on a driver that cannot encode fails HERE, loudly, not on first prefill.
        for g in blob.gen.iter().filter(|g| {
            g.kind == packet::rope::GEN_TMAP_BF16 || g.kind == packet::rope::GEN_TMAP_E4M3
        }) {
            let (map, tgt) = (g.tensor as usize, g.aux as usize);
            if tgt >= devp.len() || map >= devp.len() {
                return Err(RuntimeError::Device(format!(
                    "GEN_TMAP tensor {} targets out-of-range handle {}",
                    g.tensor, g.aux
                )));
            }
            // TMA reads the descriptor by device address; SLAB_ALIGN carving makes this
            // unbreakable today — assert so a future carve change fails loud.
            assert!(
                devp[map].base % 128 == 0,
                "tensormap tensor not 128 B aligned"
            );
            let bytes = if g.kind == packet::rope::GEN_TMAP_E4M3 {
                be.encode_tmap_e4m3(devp[tgt].base, g.ctx, g.hd, g.scale)?
            } else {
                be.encode_tmap_bf16(devp[tgt].base, g.ctx, g.hd, g.scale)?
            };
            be.upload(&devp[map], 0, &bytes)?;
            tracing::debug!(
                map = %blob.tensors[map].name, target = %blob.tensors[tgt].name,
                rows = g.ctx, k = g.hd, box_rows = g.scale, "encoded tensormap"
            );
        }

        let mut ptrs: Vec<u64> = devp.iter().map(|m| m.base).collect();
        for map in &kv_maps {
            map.encode_slot(&be, &ptrs, 0, &devp[map.tensor])?;
        }
        let pf_handles = pf_bufs.as_ref().map(|(s, r)| {
            if let Some(p) = &packed_prefill {
                return (u32::from(p.slot), u32::from(p.request));
            }
            let h_slot = ptrs.len() as u32;
            // These runtime-appended handles are patched into u16 wire slots
            // (`DevInst64::t`), which the compiler's pack-time assert cannot see.
            assert!(
                h_slot + 1 < TENSOR_NONE16 as u32,
                "tensor table overflows u16 handles"
            );
            ptrs.push(s.base);
            ptrs.push(r.base);
            (h_slot, h_slot + 1)
        });
        let d_tens = be.alloc(0, (ptrs.len() * 8) as u64)?;
        be.upload(&d_tens, 0, bytemuck::cast_slice(&ptrs))?;

        // ---- decode program tables ----
        let g = blob.decode_prog()?;
        // Decode batch: the compiler emits the decode program with t == B
        // (PLOW_DECODE_BATCH). Cross-check against the [B]-sized in.kvlen.
        let batch = g.t as usize;
        if batch > 16
            && decode_roles.contains(&plow_asset::segment_roles::NATIVE_DECODE_TC)
            && be.module_global_u32(&module, "plow_decode_block_norm_abi")? != Some(1)
        {
            return Err(RuntimeError::Rejected(
                "native B32 decode requires rung-invariant block normalization".into(),
            ));
        }
        let kvlen_bytes = blob.tensors[t_kvlen].bytes;
        if (mixed_packet.is_none() && kvlen_bytes != (batch * 4) as u64)
            || kvlen_bytes < (batch * 4) as u64
        {
            return Err(RuntimeError::Device(format!(
                "in.kvlen is {} B but the decode program's batch is {batch} \
                 (want {} B) — blob/tensor mismatch",
                kvlen_bytes,
                batch * 4
            )));
        }
        if decode_packet_roles.contains(&plow_asset::segment_roles::FP8_M1) {
            plow_asset::fp8_m1_role::options(
                effective_multistep,
                false,
                be.module_global_u32(&module, "plow_packet_hash_lo")?
                    .is_some()
                    || be
                        .module_global_u32(&module, "plow_packet_hash_hi")?
                        .is_some(),
            )
            .map_err(RuntimeError::Rejected)?;
        }
        if !decode_packet_roles.is_empty()
            && (cublaslt_enabled
                || be.module_global_u32(&module, "plow_segment_gq_abi")? != Some(1)
                || be.module_global_u32(&module, "plow_dyn_kvrow")? != Some(1))
        {
            return Err(RuntimeError::Rejected(
                "native GEMV decode roles require a dynamic-row GQ interpreter".into(),
            ));
        }
        let cublaslt_segments = if cublaslt_enabled {
            if be.module_global_u32(&module, "plow_cublaslt_decode_abi")? != Some(1) {
                return Err(RuntimeError::Rejected(
                    "cuBLASLt decode requires a paired GQ interpreter".into(),
                ));
            }
            cublaslt::decode_segments(g, &blob.tensors, &decode_roles)?
        } else {
            if decode_packet_roles.is_empty() {
                g.check_coarse_single_segment()?;
            }
            Vec::new()
        };
        // Single segment normally; an L2-PLACED program (l2_domains != 0) carries one
        // window per domain and the placed interpreter picks its window by physical SM.
        let want_seg = if !decode_packet_roles.is_empty() {
            decode_packet_roles.len() + 1
        } else if !cublaslt_segments.is_empty() {
            cublaslt_segments.len() + 1
        } else if g.l2_domains != 0 {
            g.l2_domains as usize + 1
        } else {
            2
        };
        if g.gq_stream.is_empty() || g.gq_seg_ofs.len() != want_seg {
            return Err(RuntimeError::Device(format!(
                "decode program has no single-segment GQ appendix (n_seg bounds: {:?}) — \
                 recompile the packet with a GQ-capable plowc",
                g.gq_seg_ofs
            )));
        }
        g.check_gq_topological()?;

        let upload_pod = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        // Dynamic B=1 KV row (plan stage 2): when the cubin advertises
        // `plow_dyn_kvrow`, arm the pos[t]-derived KV row on every B=1
        // KV-write site ONCE (i[6] = 1) — the decode instruction stream is
        // then immutable and `step_slots` never rewrites/re-uploads it.
        // Byte-identical rows: the dynamic formula reads in.pos[0], which the
        // step uploads with exactly the value the host used to patch i[3].
        // B>1 programs already carry i[6] = B from the compiler; old cubins
        // (no metadata symbol) keep the legacy per-token patch.
        let mut insts = g.insts.clone();
        let mut kvrow = blob.kvrow.clone();
        if batch == 1
            && !kvrow.is_empty()
            && be.module_global_u32(&module, "plow_dyn_kvrow")?.is_some()
        {
            for &ix in &kvrow {
                insts[ix as usize].i[6] = 1;
            }
            kvrow.clear();
            tracing::info!("decode: dynamic B=1 KV row — immutable instruction stream");
        }
        // Rung graphs run serially on the engine stream and share one Lt workspace.
        let cublaslt = if native_enabled {
            let object = &segment_roles.as_ref().expect("native roles").objects
                [&plow_asset::segment_roles::NATIVE_DECODE_TC];
            Some(cublaslt::ProjectionBackend::Native(
                native_decode::Native::load(
                    &be,
                    assets_dir,
                    object,
                    profile.tag,
                    &cublaslt_segments,
                )?,
            ))
        } else if cublaslt_enabled {
            Some(cublaslt::ProjectionBackend::Lt(
                crate::device::cuda::lt::Lt::load(&be)?,
            ))
        } else {
            None
        };
        let ordered_waits = if cublaslt_enabled {
            Some(cublaslt::ordered_waits(g, &cublaslt_segments)?)
        } else {
            None
        };
        let cublaslt_decode = if let Some(lt) = &cublaslt {
            cublaslt::prepare_routes(lt, cublaslt_segments, &mut insts, &devp, None)?
        } else {
            Vec::new()
        };
        let d_inst = upload_pod(pod_bytes(&insts))?;
        let d_stream = upload_pod(pod_bytes(&g.stream))?;
        let d_sofs = upload_pod(pod_bytes(&g.stream_ofs))?;
        let d_slen = upload_pod(pod_bytes(&g.stream_len))?;
        let d_waits = upload_pod(pod_bytes(ordered_waits.as_deref().unwrap_or(&g.waits)))?;
        let d_succs = upload_pod(pod_bytes(&g.succs))?;
        let d_gq_stream = upload_pod(pod_bytes(&g.gq_stream))?;
        let d_gq_seg = upload_pod(pod_bytes(&g.gq_seg_ofs))?;
        // The decode counter block and the GQ cursor share a lifecycle (both
        // re-zeroed before every launch), so they share one allocation: the
        // cursor sits at the counter block's tail and ONE memset re-arms both.
        let ctr_bytes = g.n_counter as usize * CTR_STRIDE as usize * 4;
        // One cursor line per GQ segment (see the prefill-bucket twin of this note):
        // an L2-placed blob's interpreter fetch-adds PLOW_CTR(gq_cursor, domain).
        let cursor_bytes = g.gq_seg_ofs.len().saturating_sub(1).max(1) * CTR_STRIDE as usize * 4;
        let ctr_block = be.alloc(0, (ctr_bytes.max(4) + cursor_bytes) as u64)?;
        // Two aliased views of the one owned block; `ctr_block` is stored in
        // the engine so the storage outlives both.
        let d_ctr = DeviceMem::view(ctr_block.base, ctr_bytes.max(4) as u64);
        let d_gq_cursor = DeviceMem::view(
            ctr_block.base + ctr_bytes.max(4) as u64,
            cursor_bytes as u64,
        );

        let kernarg = DevProgram {
            insts: d_inst.base,
            stream: d_stream.base,
            stream_ofs: d_sofs.base,
            stream_len: d_slen.base,
            waits: d_waits.base,
            succs: d_succs.base,
            counters: d_ctr.base,
            tensors: d_tens.base,
            trace: 0,
            cur_seg: 0,
            l2_domains: 0,
            // Hierarchy off: this engine never sets `l2_domains`, and the
            // two-level maintenance scratch is meaningless without it.
            hier_base: 0,
            n_seg: decode_packet_roles.len().max(1) as u32,
            gq_stream: d_gq_stream.base,
            gq_seg_ofs: d_gq_seg.base,
            gq_cursor: d_gq_cursor.base,
            xctr: 0,
            peer_scratch: 0,
            rank: 0,
            n_gpu: 1,
            seg_ofs: 0,
            prefill_spans: 0,
            prefill_parked: 0,
            n_prefill_spans: 0,
            n_prefill_rows: 0,
            token_batch: 0,
        };

        let stream = be.stream_create()?;
        let decode_rungs = if select_decode_rungs {
            blob.decode_progs()[..blob.decode_progs().len() - 1]
                .iter()
                .enumerate()
                .map(|(index, g)| {
                    let mut rung = if let Some(lt) = &cublaslt {
                        let roles = &segment_roles
                            .as_ref()
                            .unwrap()
                            .program(blob.progs.len() - blob.decode_progs().len() + index)
                            .unwrap()
                            .roles;
                        let segments = cublaslt::decode_segments(g, &blob.tensors, roles)?;
                        let waits = cublaslt::ordered_waits(g, &segments)?;
                        let mut insts = g.insts.clone();
                        let routes = cublaslt::prepare_routes(
                            lt, segments, &mut insts, &devp, Some(&cublaslt_decode),
                        )?;
                        let mut rung = DecodeRung::upload_with_insts(&be, g, kernarg, &insts, &waits)?;
                        rung.library = Some(cublaslt::CublasLtDecodeGraph::capture(
                            &be, &stream, rung.kernarg, f, grid, smem, routes,
                        )?);
                        rung
                    } else {
                        DecodeRung::upload(&be, g, kernarg)?
                    };
                    if let (Some(metadata), Some(objects)) = (&decode_objects, &bound_objects) {
                        rung.object = Some(Arc::clone(&objects[&metadata.programs[index].object]));
                        tracing::info!(
                            rows = g.t,
                            object = metadata.programs[index].object,
                            "decode program object selected at load"
                        );
                    }
                    Ok(rung)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let decode_contexts = prepared_contexts
            .map(|contexts| contexts.materialize(&be, kernarg, assets_dir))
            .transpose()?;
        if !decode_rungs.is_empty() {
            tracing::info!(
                rungs = decode_rungs.len() + 1,
                "packet decode rung selection enabled"
            );
        }

        // kv-row patch range: the contiguous [lo..hi] instruction window the
        // per-step upload rewrites (harness `run_step`).
        let (mut lo, mut hi) = (g.insts.len().saturating_sub(1), 0usize);
        for &ix in &blob.kvrow {
            lo = lo.min(ix as usize);
            hi = hi.max(ix as usize);
        }

        // act.logits is [B][vocab] bf16; in.pos is [ctx] i32 (shared by the
        // prefill chunk positions and the B decode positions).
        let vocab = (blob.tensors[t_logits].bytes / 2) as usize / batch;
        let max_ctx = (blob.tensors[t_pos].bytes / 4) as usize;

        // Per-slot stride of every batch-major KV or recurrent-state tensor (slot b of
        // tensor i lives at base + b*stride). B==1: strides never used.
        let kv_slots: Vec<(usize, u64)> = blob
            .tensors
            .iter()
            .enumerate()
            .filter(|(id, td)| {
                td.name.starts_with("kv.")
                    || td.name.starts_with("state.qwen.")
                    || vmm.as_ref().is_some_and(|v| v.cache_tensors.contains(id))
            })
            .map(|(i, td)| (i, td.bytes / batch as u64))
            .collect();

        // Stage 3: one immutable device tensor table PER SLOT, built at load
        // (index b-1 = slot b; slot 0 is `d_tens` itself). A per-slot prefill
        // launch selects its table through the kernarg instead of rewriting
        // and restoring the shared table around every chunk chain — the
        // tables never change after this point, and decode can never observe
        // a shifted table. A few KiB per slot.
        let mut d_tens_slots: Vec<DeviceMem> = Vec::new();
        let mut kv_tmap_slots: Vec<DeviceMem> = Vec::new();
        let mut packed_map_ptrs: Vec<Vec<u64>> = packed_prefill
            .as_ref()
            .map(|p| {
                p.maps
                    .iter()
                    .map(|m| vec![ptrs[m.original as usize]])
                    .collect()
            })
            .unwrap_or_default();
        if batch > 1 && (!kv_slots.is_empty() || !kv_maps.is_empty()) {
            let mut shifted = ptrs.clone();
            for b in 1..batch {
                for &(i, stride) in &kv_slots {
                    shifted[i] = ptrs[i] + b as u64 * stride;
                }
                for map in &kv_maps {
                    let descriptor = be.alloc(0, 256)?;
                    for (id, base) in map.encode_slot(&be, &ptrs, b, &descriptor)? {
                        shifted[id] = base;
                    }
                    kv_tmap_slots.push(descriptor);
                }
                if let Some(pack) = &packed_prefill {
                    for (m, table) in pack.maps.iter().zip(&mut packed_map_ptrs) {
                        table.push(shifted[m.original as usize]);
                    }
                }
                let mem = be.alloc(0, (shifted.len() * 8) as u64)?;
                be.upload(&mem, 0, bytemuck::cast_slice(&shifted))?;
                d_tens_slots.push(mem);
            }
        }

        if let Some(pack) = &packed_prefill {
            for (m, table) in pack.maps.iter().zip(&packed_map_ptrs) {
                if table.len() != batch {
                    return Err(RuntimeError::Rejected(
                        "packed descriptor slot coverage".into(),
                    ));
                }
                be.upload(&devp[m.slots as usize], 0, bytemuck::cast_slice(table))?;
            }
            be.synchronize()?;
        }

        // Every slot's row 0 must be mapped before any batched decode: unfed
        // rows write garbage KV at their own pos (mux contract), and pos
        // starts at 0.
        if let Some(v) = &vmm {
            for b in 0..batch {
                v.kv.ensure_rows(b, 1)?;
            }
        }

        // Stop set from the checkpoint's generation_config (fallback config).
        // BOTH sources, matching the AMD and CPU engines. This read only
        // `read_eos_ids`, so a family whose turn actually ends at a token that
        // lives in `tokenizer_config.json` rather than in the eos list — K3
        // closes on `<|close|>` — ran past its own turn boundary on NVIDIA
        // while stopping correctly on the other two backends.
        let mut stop_ids = crate::asset::checkpoint::read_eos_ids(checkpoint_dir);
        stop_ids.extend(crate::asset::checkpoint::chat_stop_ids(checkpoint_dir, &stop_ids));
        stop_ids.sort_unstable();
        stop_ids.dedup();
        if let Some(tm) = load_tim.as_mut() {
            let ms = t_decode.elapsed().as_secs_f64() * 1e3;
            tm.decode_tables_ms = ms;
            tm.spans
                .push(("decode_tables".into(), decode_t0, tm.ms_since_t0()));
            tm.log_stage(
                "decode_tables",
                sys_decode,
                std::time::SystemTime::now(),
                ms,
            );
        }

        // ---- prefill object + bucket programs (optional) ----
        let t_pf = std::time::Instant::now();
        let pf_t0 = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let sys_pf = std::time::SystemTime::now();
        // Load the `_pf` cubin and upload every non-decode (T!=1) program so a
        // prompt is consumed in chunks by the tiled-GEMM/flash-prefill buckets
        // instead of O(n) decode launches. Absent cubin, a segmented bucket, or
        // a missing GQ appendix disables prefill (mux falls back to decode-only
        // consumption) rather than failing the whole engine.
        // Same content-addressed resolution as decode: the `_pf` object is
        // identified by the prefill entry in its symbol table, so the pair being
        // swapped on disk no longer costs the prefill path (it used to load the
        // DECODE image here and fail on the missing `_pf` symbol).
        let pf = resolve_interp_image(assets_dir, &blob, &raw, &profile, want_sm, Role::Prefill)?;
        let (f_pf, smem_pf, module_pf, prefill, seg_pf) = if let Some(pf) = pf {
            let pf_src = pf.source.clone();
            match Self::load_prefill(
                &be,
                pf,
                &blob,
                assets_dir,
                d_tens.base,
                grid,
                profile.tag,
                segment_roles.as_ref(),
                packed_prefill.as_ref(),
            ) {
                Ok((f_pf, smem_pf, module_pf, buckets, seg_pf)) => {
                    tracing::info!(
                        pf_cubin = %pf_src,
                        buckets = buckets.len(),
                        smem_pf,
                        segmented = seg_pf.is_some(),
                        "prefill object loaded"
                    );
                    (Some(f_pf), smem_pf, Some(module_pf), buckets, seg_pf)
                }
                // HARD ERROR, not a fallback. The cubin is PRESENT and failed to
                // load — a broken deployment, not a configuration. Falling back
                // to decode-only means consuming the prompt ONE TOKEN AT A TIME:
                // measured >=707 s on a 127k prompt (PX-13) against ~32 s with
                // the buckets, and the only symptom is that it is slow. The
                // usual cause is an arena over the 101376 B opt-in cap
                // (`PGM_STAGES=5`, `PGM_BN=256`), which is a build mistake the
                // operator must see. A genuinely absent cubin still falls back,
                // below — that path is documented and intentional.
                Err(e) => {
                    return Err(RuntimeError::Device(format!(
                        "prefill object {} failed to load: {e}\n\
                         Refusing to start: falling back to decode-only prompt consumption \
                         would prefill one token at a time (measured >=707 s vs ~32 s on a \
                         127k prompt) with no symptom but slowness.\n\
                         Common cause: the object's shared-memory arena exceeds this device's \
                         opt-in cap — check `-Xptxas -v` against the 101376 B limit (e.g. \
                         PGM_STAGES=5 or PGM_BN=256 overflow it). Remove the prefill cubin \
                         to serve decode-only deliberately.",
                        pf_src
                    )));
                }
            }
        } else {
            tracing::info!(
                expected = profile.prefill_file,
                "no prefill object for sm_{want_sm} — decode-only prompt consumption"
            );
            (None, SMEM_PF, None, Vec::new(), None)
        };

        let mut packet_roles: [Option<PacketRole>; plow_asset::segment_roles::MAX_ROLE as usize] =
            std::array::from_fn(|_| None);
        for (&id, object) in segment_roles.iter().flat_map(|r| &r.objects) {
            if id == plow_asset::segment_roles::NATIVE_DECODE_TC {
                continue;
            }
            if id == plow_asset::segment_roles::FP8_M1 {
                packet_roles[plow_asset::segment_roles::FP8_M1 as usize - 1] = Some(
                    load_fp8_m1_role(&be, assets_dir, object, profile.tag, grid)?,
                );
                continue;
            }
            if !matches!(
                id,
                plow_asset::segment_roles::GEMV_CTA512 | plow_asset::segment_roles::MXFP4_MOE
            ) && prefill.is_empty()
            {
                return Err(RuntimeError::Rejected(
                    "prefill packet roles require a prefill object".into(),
                ));
            }
            if matches!(
                id,
                plow_asset::segment_roles::PREFILL_ATTENTION
                    | plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32
                    | plow_asset::segment_roles::GEMV_CTA512
            ) && profile.tag != "sm90a"
            {
                return Err(RuntimeError::Rejected("packet role requires SM90".into()));
            }
            let (marker, block, entry, arena) = match id {
                plow_asset::segment_roles::FP8_PREFILL_GEMM => (
                    "plow_fp8_gemm_tma128_abi",
                    "plow_block_pfgemm",
                    "_Z19interp_sm90a_pfgemm11PlowProgram",
                    "plow_arena_bytes_pfgemm",
                ),
                plow_asset::segment_roles::PREFILL_ATTENTION => (
                    "plow_attention_sm90_hd256_abi",
                    "plow_block_pffa",
                    "_Z17interp_sm90a_pffa11PlowProgram",
                    "plow_arena_bytes_pffa",
                ),
                plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32 => (
                    "plow_attention_sm90_hd512_wg32_abi",
                    "plow_block_pfattn_hd512",
                    "plow_sm90a_pfattn_hd512",
                    "plow_arena_bytes_pfattn_hd512",
                ),
                plow_asset::segment_roles::GEMV_CTA512 => (
                    "plow_gemv_sm90_cta512_abi",
                    "plow_block_gemv512",
                    "_Z20interp_sm90a_gemv51211PlowProgram",
                    "plow_arena_bytes_gemv512",
                ),
                plow_asset::segment_roles::MXFP4_MOE => (
                    "plow_mxfp4_moe_sm90_abi",
                    "plow_block_mxfp4_moe",
                    "plow_sm90a_mxfp4_moe",
                    "plow_arena_bytes_mxfp4_moe",
                ),
                _ => {
                    return Err(RuntimeError::Rejected(
                        "unsupported packet object role".into(),
                    ));
                }
            };
            let path = assets_dir.join(&object.file);
            let image = std::fs::read(&path)
                .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
            if id == plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32
                && object.sha256.as_deref()
                    != Some(plow_asset::decode_objects::image_sha256(&image).as_str())
            {
                return Err(RuntimeError::Rejected(
                    "HD512 attention role object hash mismatch".into(),
                ));
            }
            if id == plow_asset::segment_roles::MXFP4_MOE
                && object.sha256.as_deref()
                    != Some(plow_asset::decode_objects::image_sha256(&image).as_str())
            {
                return Err(RuntimeError::Rejected(
                    "MXFP4 MoE role object hash mismatch".into(),
                ));
            }
            if let Some(pack) = &packed_prefill {
                pack.validate_object(|name| cubin::global_u32(&image, name))
                    .map_err(RuntimeError::Rejected)?;
            }
            let module = DecodeModule::load(&be, &image)?;
            let capability = be.module_global_u32(&module, marker)?;
            let block = be.module_global_u32(&module, block)?;
            match id {
                plow_asset::segment_roles::FP8_PREFILL_GEMM => {
                    check_fp8_gemm_role(capability, block)?
                }
                plow_asset::segment_roles::PREFILL_ATTENTION => {
                    check_attention_role(profile.tag, capability, block)?
                }
                plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32 => {
                    check_attention_hd512_role(
                        profile.tag,
                        object,
                        capability,
                        block,
                        [
                            be.module_global_u32(&module, "plow_attention_head_dim")?,
                            be.module_global_u32(&module, "plow_attention_query_tile")?,
                            be.module_global_u32(&module, "plow_attention_kv_tile")?,
                            be.module_global_u32(&module, "plow_attention_warps")?,
                        ],
                    )?
                }
                plow_asset::segment_roles::GEMV_CTA512 => {
                    if capability != Some(1) || block != Some(512) {
                        return Err(RuntimeError::Rejected("incompatible GEMV512 role".into()));
                    }
                }
                plow_asset::segment_roles::MXFP4_MOE => {
                    if capability != Some(1) || block != Some(BLOCK) {
                        return Err(RuntimeError::Rejected("incompatible MXFP4 MoE role".into()));
                    }
                }
                _ => unreachable!("object role validated above"),
            }
            let block = block.expect("validated role block");
            let function = be.get_function(&module, entry)?;
            let smem = be
                .module_global_u32(&module, arena)?
                .filter(|&bytes| bytes > 0)
                .ok_or_else(|| RuntimeError::Rejected("packet role lacks arena metadata".into()))?;
            be.set_max_dynamic_smem(function, smem)?;
            if id == plow_asset::segment_roles::GEMV_CTA512 && smem != 65536 {
                return Err(RuntimeError::Rejected(
                    "GEMV512 role requires the full 64 KiB arena".into(),
                ));
            }
            let capacity =
                be.occupancy_blocks_per_sm(function, block, smem as usize)? * be.sm_count();
            let role_grid = if id == plow_asset::segment_roles::MXFP4_MOE {
                let multiplier = be
                    .module_global_u32(&module, "plow_mxfp4_moe_ctas_per_sm")?
                    .filter(|&value| value == 4)
                    .ok_or_else(|| {
                        RuntimeError::Rejected("MXFP4 MoE role lacks CTA capability".into())
                    })?;
                grid.checked_mul(multiplier)
                    .ok_or_else(|| RuntimeError::Rejected("MXFP4 MoE role grid overflows".into()))?
            } else {
                grid
            };
            if (id == plow_asset::segment_roles::GEMV_CTA512 && capacity < grid)
                || (id == plow_asset::segment_roles::MXFP4_MOE && capacity < role_grid)
                || (!matches!(
                    id,
                    plow_asset::segment_roles::GEMV_CTA512 | plow_asset::segment_roles::MXFP4_MOE
                ) && capacity != grid)
            {
                return Err(RuntimeError::Rejected(
                    "packet role occupancy must equal packet grid".into(),
                ));
            }
            packet_roles[id as usize - 1] = Some(PacketRole {
                function,
                grid: role_grid,
                smem,
                block,
                _module: module,
            });
        }

        let qwen_prefill = if prefill
            .iter()
            .any(|p| p.qwen_segments.iter().any(Option::is_some))
        {
            Some(crate::device::cuda::qwen_gdn::NativeGdn::load(
                be.clone(),
                &assets_dir.join("libplow_gdn_prefill.so"),
            )?)
        } else {
            None
        };

        // ---- Cross-request prefill mode (finalized once the prefill object is up) ----
        // A validated manifest selects its packet-defined chain without a
        // runtime knob. The legacy prototype retains its old opt-in/fallback.
        let pf_batch: Option<PfBatch> = match (pf_bufs, pf_handles) {
            (Some((d_slot, d_req)), Some((h_slot, h_req)))
                if f_pf.is_some() && !prefill.is_empty() =>
            {
                if packed_prefill.is_some() {
                    if (vmm.as_ref().is_some_and(|v| v.kv.prefix_reuse()) && !packed_prefix)
                        || recurrent.is_some()
                        || prefill.iter().any(|b| {
                            b.seg_class.len() < 2 || b.qwen_segments.iter().any(Option::is_some)
                        })
                    {
                        return Err(RuntimeError::Rejected("packed requests require complete direct-KV segmented chains and a compatible prefix layout".into()));
                    }
                    Some(PfBatch {
                        d_slot,
                        d_req,
                        h_slot,
                        h_req,
                        at_sites: Vec::new(),
                        slot_buf: vec![0; pf_max_t_blob],
                        req_buf: Vec::with_capacity(1 + 4 * dbatch_blob),
                    })
                } else if vmm.is_some() {
                    tracing::warn!("PLOW_PF_BATCH=1 ignored: incompatible with VMM KV allocation");
                    None
                } else {
                    let fused = prefill.iter().find(|b| {
                        // `!b.fp8_kv`: the fp8 flash arm reads t6/t7 as the k/v
                        // scales, so the batched patch (t6 = request table) would
                        // hand it garbage. The kernel has no fp8 mux arm either.
                        !b.fp8_kv
                            && !b.flash_sites.is_empty()
                            && b.flash_sites.iter().all(|&ix| {
                                b.h_inst[ix].i[7] == 1 && b.h_inst[ix].t[5] != TENSOR_NONE16
                            })
                    });
                    if fused.is_none() && prefill.iter().any(|b| b.fp8_kv) {
                        tracing::warn!(
                            "PLOW_PF_BATCH=1 ignored: fp8-KV packets have no batched prefill arm"
                        );
                    }
                    match fused {
                        Some(b) => {
                            let at_sites: Vec<(u32, u32)> = b
                                .flash_sites
                                .iter()
                                .map(|&ix| (b.h_inst[ix].t[5] as u32, b.h_inst[ix].i[6]))
                                .collect();
                            tracing::info!(
                                fused_bucket_t = b.t,
                                flash_sites = at_sites.len(),
                                "PX-1 cross-request batched prefill enabled (PLOW_PF_BATCH=1)"
                            );
                            Some(PfBatch {
                                d_slot,
                                d_req,
                                h_slot,
                                h_req,
                                at_sites,
                                slot_buf: vec![0; pf_max_t_blob],
                                req_buf: Vec::with_capacity(1 + 4 * dbatch_blob),
                            })
                        }
                        None => {
                            tracing::warn!(
                                "PLOW_PF_BATCH=1 ignored: no fused (nsplit==1) prefill bucket"
                            );
                            None
                        }
                    }
                }
            }
            (Some(_), _) if packed_prefill.is_some() => {
                return Err(RuntimeError::Rejected(
                    "packed prefill manifest requires a loaded prefill object and buckets".into(),
                ));
            }
            (Some(_), _) => {
                tracing::warn!("PLOW_PF_BATCH=1 ignored: prefill object not loaded");
                None
            }
            _ => None,
        };
        if let Some(tm) = load_tim.as_mut() {
            let ms = t_pf.elapsed().as_secs_f64() * 1e3;
            tm.prefill_ms = ms;
            tm.spans
                .push(("prefill_load".into(), pf_t0, tm.ms_since_t0()));
            tm.log_stage("prefill_load", sys_pf, std::time::SystemTime::now(), ms);
        }

        tracing::info!(
            pkt = %pkt.display(),
            grid,
            weights_gib = wb as f64 / (1u64 << 30) as f64,
            kv_gib = kvb as f64 / (1u64 << 30) as f64,
            n_weights = nw,
            vocab,
            max_ctx,
            batch,
            prefill_buckets = prefill.len(),
            stop_ids = ?stop_ids,
            vmm_prefix = vmm.as_ref().is_some_and(|v| v.kv.prefix_reuse()),
            vmm_live = vmm.as_ref().is_some_and(|v| !v.kv.prefix_reuse()),
            elapsed_s = t0.elapsed().as_secs_f32(),
            // Was a hardcoded "sm_120" from the sm120-only era. On a Hopper card it
            // printed sm_120 while running the sm90a object, which reads as a
            // wrong-profile bug during exactly the kind of bring-up that needs this
            // line to be trustworthy. Report the profile actually selected.
            interp = profile.tag,
            "GPU engine loaded (decode program)"
        );

        let t_final = std::time::Instant::now();
        let final_t0 = load_tim.as_ref().map(|t| t.ms_since_t0()).unwrap_or(0.0);
        let sys_final = std::time::SystemTime::now();

        let pf_max_t = prefill.last().map_or(0, |b| b.t as usize);

        // The engine's ordered device queue + pinned per-step staging + the
        // flag-gated (`--step-time` / PLOW_STEP_TIME=1) CUDA-event timing
        // (plan stage 1: async submission path).
        let stage = StepStage::new(&be, batch, recurrent.is_some())?;
        let h2d_ev = be.event_create(false)?;
        let timing = match crate::config::RuntimeConfig::get().nv.step_time {
            true => Some(StepTiming::new(&be)?),
            false => None,
        };
        let mixed_step = mixed_packet
            .map(|packet| {
                if recurrent.is_some() {
                    return Err(RuntimeError::Rejected(
                        "mixed dense step does not support recurrent state".into(),
                    ));
                }
                mixed_step::MixedCudaStep::load(&be, packet, &blob, &devp, d_tens.base, batch)
            })
            .transpose()?;

        // ---- device stochastic sampler (PLOW_DEV_SAMPLE=1; plan stage 4) ----
        // Loads a `plow_sample` cubin and allocates its per-slot param + [B][V]
        // scratch buffers. Absent flag/cubin → host sampling (unchanged).
        let sampler = Self::sampler_bringup(&be, assets_dir, batch, vocab).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "device sampler disabled — host sampling");
            None
        });
        // Stage 5: bounded device multi-step. Requires the KV write row to come
        // from device `pos` (no per-step host patch — the thing multi-step
        // removes). True for B>1 (the kernel uses n_batch_kv from pos[b], and
        // step_slots never patches i[3] there) and for B==1 only when the
        // dynamic-kvrow arm fired (the local `kvrow` was cleared above). A B==1
        // legacy cubin still host-patches i[3] each step, so multi-step is off.
        let dyn_kvrow = batch > 1 || kvrow.is_empty();
        let multistep =
            Self::multistep_bringup(&be, assets_dir, batch, dyn_kvrow, effective_multistep)
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "multi-step disabled");
                    None
                });

        if let Some(tm) = load_tim.as_mut() {
            let ms = t_final.elapsed().as_secs_f64() * 1e3;
            tm.final_init_ms = ms;
            tm.spans
                .push(("final_init".into(), final_t0, tm.ms_since_t0()));
            tm.log_stage("final_init", sys_final, std::time::SystemTime::now(), ms);
            let total = t0.elapsed().as_secs_f64() * 1e3;
            tm.print_summary(total);
            tm.print_flame(total);
        }

        let mut engine = GpuEngine {
            be,
            f,
            grid,
            smem,
            stream,
            module,
            f_pf,
            seg_pf,
            qwen_prefill,
            packet_roles,
            cublaslt_decode,
            cublaslt_decode_graph: None,
            cublaslt_decode_capture: !decode_packet_roles.is_empty() || cublaslt_enabled,
            decode_packet_roles,
            seg_graphs: std::collections::HashMap::new(),
            smem_pf,
            prefill,
            module_pf,
            sampler,
            multistep,
            h_inst: insts,
            decode_rungs,
            decode_contexts,
            kvrow,
            kvrow_lo: lo,
            kvrow_hi: hi,
            ctr_bytes,
            cursor_bytes,
            devp,
            _weight_slab: weight_slab,
            d_inst,
            _ctr_block: ctr_block,
            d_ctr,
            _d_gq_cursor: d_gq_cursor,
            d_tens,
            d_tens_slots,
            _kv_tmap_slots: kv_tmap_slots,
            _tables: vec![
                d_stream,
                d_sofs,
                d_slen,
                d_waits,
                d_succs,
                d_gq_stream,
                d_gq_seg,
            ],
            kernarg,
            t_ids,
            t_pos,
            t_kvlen,
            t_logits,
            recurrent,
            tensor_names: blob.tensors.iter().map(|t| t.name.clone()).collect(),
            vocab,
            max_ctx,
            batch,
            pos: vec![0; batch],
            vmm_attached: vec![0; batch],
            vmm_active: vec![false; batch],
            packed_admission: vec![PackedAdmission::Pending; batch],
            kv_admission_epoch: 0,
            seq_tokens: vec![Vec::new(); batch],
            stop_ids: std::sync::Arc::new(stop_ids),
            logits_raw: Vec::new(),
            stage,
            h2d_ev,
            pf_ids: vec![0i32; pf_max_t],
            pf_pos: vec![0i32; pf_max_t],
            logits_f32: Vec::new(),
            timing,
            vmm,
            pf_batch,
            prefill_turn: 0,
            packed_prefill,
            packed_terminal: None,
            mixed_step,
            token_batch: None,
            slot_generations: vec![0; batch],
        };
        engine.packed_terminal = packed_terminal::PackedTerminal::load(&engine)?;
        engine.token_batch = token_batch::CudaTokenBatch::load(&engine);
        if config.token_batch {
            tracing::info!(
                route = "unified-token-batch",
                backend = "cuda",
                ready = engine.token_batch_enabled(),
                fires = false,
                reason = if engine.token_batch_enabled() {
                    "packed-prefill and compact-output capabilities loaded"
                } else {
                    "unsupported device, model, object or execution mode; using ordinary execution"
                },
                "token-batch route status"
            );
        }
        if engine.cublaslt_decode_capture {
            engine.capture_decode_graph()?;
        }
        Ok(engine)
    }

    /// Refuse a specialised interpreter object paired with a different packet.
    ///
    /// The object stamps `plow_packet_hash_{lo,hi}` (interp_sm120.cu, next to
    /// `plow_arena_bytes`) ONLY when it was built from one packet's `build.json`
    /// — a general object, with every arm compiled, carries no stamp and pairs
    /// with anything, so every asset shipped today is unaffected.
    ///
    /// The expected value is read from `<assets>/build.json`, which `plowc`
    /// writes beside `model.pkt` from the very programs it serialized. No hashing
    /// happens here on purpose: recomputing it would be a second implementation of
    /// the rule and could disagree with the first, which is exactly the class of
    /// bug this check exists to end.
    ///
    /// FAILURE IS FATAL, not a warning. A stamped object is missing arms; running
    /// it against the wrong packet does not degrade, it traps mid-serve.
    fn check_packet_pairing(
        be: &Arc<CudaBackend>,
        module: &Module,
        assets_dir: &Path,
    ) -> Result<()> {
        Self::check_packet_pairing_suffix(be, module, assets_dir, "")
    }

    fn check_packet_pairing_suffix(
        be: &Arc<CudaBackend>,
        module: &Module,
        assets_dir: &Path,
        suffix: &str,
    ) -> Result<()> {
        let lo_name = format!("plow_packet_hash_lo{suffix}");
        let hi_name = format!("plow_packet_hash_hi{suffix}");
        let (lo, hi) = (
            be.module_global_u32(module, &lo_name)?,
            be.module_global_u32(module, &hi_name)?,
        );
        let (Some(lo), Some(hi)) = (lo, hi) else {
            // Unstamped ⇒ a general object. Nothing to check.
            return Ok(());
        };
        let stamped = ((hi as u64) << 32) | lo as u64;

        let mpath = assets_dir.join("build.json");
        let manifest = std::fs::read(&mpath).map_err(|source| RuntimeError::Io {
            path: mpath.clone(),
            source,
        })?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest).map_err(|e| {
            RuntimeError::Device(format!("{}: not valid JSON: {e}", mpath.display()))
        })?;
        let want = manifest
            .get("pairing")
            .and_then(|p| p.get("hash"))
            .and_then(|h| h.as_str())
            .and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok())
            .ok_or_else(|| {
                RuntimeError::Device(format!(
                    "the interpreter object is SPECIALISED (packet hash \
                     0x{stamped:016x}) but {} has no `pairing.hash` — it was written \
                     by an older plowc. Rebuild the packet, or serve a general \
                     interpreter object (one built without -DPLOW_CONFIG).",
                    mpath.display()
                ))
            })?;
        if want != stamped {
            // Name what differs, not just that something does: the arm set and the
            // rule-derived constants are the two things the hash covers.
            let arms = manifest
                .get("union")
                .and_then(|u| u.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            return Err(RuntimeError::Device(format!(
                "packet/interpreter MISMATCH: the loaded cubin was specialised for \
                 packet 0x{stamped:016x}, but the packet in {} is 0x{want:016x} \
                 ({arms} arms, tuning {}). A specialised object carries ONLY that \
                 packet's op arms, so running this pair would trap mid-serve on the \
                 first bucket needing a dropped arm. Rebuild the cubin from this \
                 packet's build.json (plowc --emit devblob+cubin), or serve a \
                 general object built without -DPLOW_CONFIG.",
                assets_dir.display(),
                manifest
                    .get("tuning")
                    .map(|t| t.to_string())
                    .unwrap_or_default(),
            ))
            .into());
        }
        tracing::info!(
            packet_hash = format!("0x{stamped:016x}"),
            "specialised interpreter paired"
        );
        Ok(())
    }






    /// Bring up the device sampler when `--dev-sample 1` / `PLOW_DEV_SAMPLE=1`
    /// and a `plow_sample` cubin is found (`--nv-cubin-sample` /
    /// `PLOW_NV_CUBIN_SAMPLE`, else `<assets>/sample_sm120.cubin`).
    /// Any missing piece returns `Ok(None)` (host sampling). The `[B][V]` f32
    /// scratch is the only sizeable allocation (~1 MiB per slot per 256k vocab).
    fn sampler_bringup(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        batch: usize,
        vocab: usize,
    ) -> Result<Option<Sampler>> {
        // DEFAULT ON (`PLOW_DEV_SAMPLE=0` opts out). Host sampling costs a
        // per-token round trip that dominates the step at batch: measured on
        // Gemma-4-12B/RTX 5090 at B=16, 83.6 ms of host gap against a 41.0 ms
        // kernel. Device sampling + `PLOW_MULTISTEP` together took that blob
        // 102.74 -> 185.60 tok/s (1.74x). Every failure path below already
        // degrades to host sampling, so defaulting on cannot break a bundle
        // that lacks the cubin. See perf-data/gemma4-12b-sm120-serving.md.
        let rt = crate::config::RuntimeConfig::get();
        let explicit = rt.nv_dev_sample();
        if explicit.as_deref() == Some("0") {
            return Ok(None);
        }
        let cubin = rt
            .nv_cubin_sample()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| assets_dir.join("sample_sm120.cubin"));
        if !cubin.is_file() {
            // Only a warning when the operator ASKED for it; on the default
            // path a bundle without the cubin is an ordinary host-sampling
            // deployment, not a misconfiguration.
            if explicit.is_some() {
                tracing::warn!(cubin = %cubin.display(), "PLOW_DEV_SAMPLE set but no sampler cubin");
            } else {
                tracing::debug!(cubin = %cubin.display(), "no sampler cubin — host sampling");
            }
            return Ok(None);
        }
        let image = std::fs::read(&cubin).map_err(|source| RuntimeError::Io {
            path: cubin.clone(),
            source,
        })?;
        let module = be.module_load(&image)?;
        let kname = rt
            .nv
            .kernel_sample
            .clone()
            .unwrap_or_else(|| "plow_sample".into());
        let f = be.get_function(&module, &kname)?;
        let params = be.host_alloc_pinned(5 * batch * 4)?;
        let d_params = be.alloc(0, (5 * batch * 4) as u64)?;
        let d_escratch = be.alloc(0, (batch * vocab * 4) as u64)?;
        tracing::info!(cubin = %cubin.display(), "device sampler enabled (PLOW_DEV_SAMPLE=1)");
        Ok(Some(Sampler {
            f,
            _module: module,
            params,
            d_params,
            d_escratch,
            batch,
        }))
    }

    /// Bring up bounded device multi-step (`--multistep K` /
    /// `PLOW_MULTISTEP=K`, K in [2,64];
    /// plan stage 5). Loads the `plow_advance` kernel from the sampler cubin
    /// and allocates the `[batch][K]` token ring + `[batch]` active-flag
    /// buffers. `Ok(None)` when the flag is unset, K is out of range, the
    /// cubin is absent, or the decode cubin is not dynamic-kvrow (multi-step
    /// needs device-owned pos).
    fn multistep_bringup(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        batch: usize,
        dyn_kvrow: bool,
        k: u32,
    ) -> Result<Option<MultiStep>> {
        // DEFAULT ON at K=8 (`PLOW_MULTISTEP=0` or `=1` opts out). K=8 captures
        // nearly all of the win — measured 179.18 tok/s vs 185.60 at K=32, i.e.
        // the last 3.6% costs 4x the quantum. The quantum is also how far ahead
        // of the client the device runs, so a small K keeps streaming delivery
        // fine-grained and bounds work generated past a stop token; that is why
        // the default is not the throughput-optimal 32.
        //
        let rt = crate::config::RuntimeConfig::get();
        let k = k as usize;
        match k {
            0 | 1 => return Ok(None),
            2..=64 => {}
            _ => {
                tracing::warn!("PLOW_MULTISTEP out of range [2,64] — multi-step off");
                return Ok(None);
            }
        }
        if !dyn_kvrow {
            // Expected on a B=1 legacy cubin that host-patches the KV row, so
            // this is only noteworthy when the operator asked for multi-step.
            tracing::warn!(
                "PLOW_MULTISTEP ignored: decode cubin is not dynamic-kvrow (needs device pos)"
            );
            return Ok(None);
        }
        let cubin = rt
            .nv_cubin_sample()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| assets_dir.join("sample_sm120.cubin"));
        if !cubin.is_file() {
            tracing::debug!(cubin = %cubin.display(), "no sampler cubin — multi-step off");
            return Ok(None);
        }
        let image = std::fs::read(&cubin).map_err(|source| RuntimeError::Io {
            path: cubin.clone(),
            source,
        })?;
        let module = be.module_load(&image)?;
        let f_advance = be.get_function(&module, "plow_advance")?;
        let d_ring = be.alloc(0, (batch * k * 4) as u64)?;
        let ring_host = be.host_alloc_pinned(batch * k * 4)?;
        let d_fed = be.alloc(0, (batch * 4) as u64)?;
        let fed_host = be.host_alloc_pinned(batch * 4)?;
        tracing::info!(
            quantum = k,
            "bounded device multi-step enabled (PLOW_MULTISTEP)"
        );
        Ok(Some(MultiStep {
            f_advance,
            _module: module,
            quantum: k,
            d_ring,
            ring_host,
            d_fed,
            fed_host,
        }))
    }








    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Decode batch B — how many independent sequence slots this engine
    /// drives (the compiled `PLOW_DECODE_BATCH`).
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Decode widths this loaded engine can execute, in ascending order.
    ///
    /// `decode_rungs` contains only the narrow programs; the main program is
    /// the widest rung. Lt and multi-step execution use the main program, so
    /// advertise only that width when either path is active.
    pub fn effective_decode_rungs(&self) -> Box<[u32]> {
        effective_decode_widths(
            self.decode_rungs.iter().map(|r| r.rows),
            self.batch,
            self.decode_rungs.is_empty(),
        )
    }






    /// Shared handle to the stop-token set (checkpoint `eos_token_id`).
    pub fn stop_ids(&self) -> &std::sync::Arc<Vec<u32>> {
        &self.stop_ids
    }

    /// Take the reusable f32 logits buffer (leaves an empty Vec in its place).
    /// Caller must return it via [`Self::return_logits_buf`] after use.
    pub fn take_logits_buf(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.logits_f32)
    }

    /// Return the reusable f32 logits buffer previously taken.
    pub fn return_logits_buf(&mut self, buf: Vec<f32>) {
        self.logits_f32 = buf;
    }

    /// Start a new sequence in slot `b`: `total` tokens (prompt + generation
    /// budget) must fit the compiled context. The KV cache needs no explicit
    /// reset — `in.kvlen` bounds what the attention reads, so rewinding
    /// `pos[b]` to 0 makes the slot's old cache rows unreachable.
    pub fn begin_slot(&mut self, b: usize, total: usize) -> Result<()> {
        if b >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {b} out of range (engine batch {})",
                self.batch
            )));
        }
        if total > self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "prompt + max_tokens = {total} exceeds the compiled context {}",
                self.max_ctx
            )));
        }
        self.reset_packed_admission(b);
        self.slot_generations[b] = self.slot_generations[b].wrapping_add(1);
        if let Some(state) = &self.recurrent {
            for &(index, stride) in &state.tensors {
                self.be.memset_d8_async(
                    self.devp[index].base + b as u64 * stride,
                    0,
                    stride as usize,
                    &self.stream,
                )?;
            }
        }
        if let Some(rings) = self.vmm.as_mut().and_then(|v| v.rings.as_mut()) {
            rings.ensure_slot(b)?;
        }
        // VMM: publish the finished sequence's generated whole blocks first —
        // a follow-up turn embedding this turn's output then attaches instead
        // of re-prefilling it — then drop the previous sequence's mappings/
        // cache references. Prefix admission maps after lookup; every execution
        // path maps the rows it writes, including inactive decode rows.
        self.vmm_publish(b, self.pos[b]);
        self.pos[b] = 0;
        self.vmm_attached[b] = 0;
        if let Some(v) = &self.vmm {
            self.vmm_active[b] = true;
            self.seq_tokens[b].clear();
            self.seq_tokens[b].reserve(total);
            v.kv.begin_seq(b);
            if !v.kv.prefix_reuse() {
                v.kv.ensure_rows(b, 1)?;
            }
        }
        Ok(())
    }

    pub fn retire_slot(&mut self, b: usize, cache_output: bool) {
        self.slot_generations[b] = self.slot_generations[b].wrapping_add(1);
        self.reset_packed_admission(b);
        if !self.vmm_active[b] {
            return;
        }
        if cache_output {
            self.vmm_publish(b, self.pos[b]);
        }
        self.pos[b] = 0;
        self.vmm_attached[b] = 0;
        // Decode's backstop maps row zero before any inactive-row write.
        self.vmm.as_ref().unwrap().kv.begin_seq(b);
        self.seq_tokens[b].clear();
        self.vmm_active[b] = false;
    }

    fn reset_packed_admission(&mut self, b: usize) {
        if self.packed_admission[b] == PackedAdmission::Ready {
            self.kv_admission_epoch = self.kv_admission_epoch.wrapping_add(1);
        }
        self.packed_admission[b] = PackedAdmission::Pending;
    }

    pub(crate) fn packed_slot_ready(&self, b: usize) -> bool {
        self.packed_admission[b] == PackedAdmission::Ready
    }

    pub(crate) fn admit_packed_slot(
        &mut self,
        b: usize,
        prompt: &[u32],
        total: usize,
    ) -> Result<Option<usize>> {
        match self.packed_admission.get(b) {
            Some(PackedAdmission::Ready) => return Ok(Some(self.pos[b] as usize)),
            Some(PackedAdmission::Waiting(epoch)) if *epoch == self.kv_admission_epoch => {
                return Ok(None);
            }
            Some(PackedAdmission::Pending)
                if self.packed_admission.iter().any(|state| {
                    matches!(state, PackedAdmission::Waiting(epoch) if *epoch != self.kv_admission_epoch)
                }) =>
            {
                // Retry older waiters before a new arrival takes released pages.
                return Ok(None);
            }
            None => return Err(RuntimeError::Rejected(format!("slot {b} out of range"))),
            _ => {}
        }
        let started = (|| {
            self.begin_slot(b, total)?;
            let frontier = self.attach_prompt(b, prompt)?;
            if let Some(v) = self.vmm.as_ref().filter(|v| v.kv.prefix_reuse()) {
                // Wider decode rungs write idle rows too. Reserve those before
                // one request can consume their remaining physical pages.
                for slot in 0..self.batch {
                    v.kv.ensure_rows(slot, 1)?;
                }
                let margin = (v.kv.block_rows() as usize).max(self.pf_max_rows());
                let rows = total.saturating_add(margin).min(self.max_ctx);
                v.kv.ensure_rows(b, rows as u32)?;
            }
            Ok(frontier)
        })();
        match started {
            Ok(frontier) => {
                self.packed_admission[b] = PackedAdmission::Ready;
                Ok(Some(frontier))
            }
            Err(error) => {
                self.retire_slot(b, false);
                let oom = matches!(error, RuntimeError::Oom(_)) || error.device_code() == Some(2);
                if self.vmm_prefix_enabled()
                    && oom
                    && !error.is_fatal()
                    && self.packed_admission.contains(&PackedAdmission::Ready)
                {
                    self.vmm.as_ref().unwrap().kv.ensure_rows(b, 1)?;
                    self.packed_admission[b] = PackedAdmission::Waiting(self.kv_admission_epoch);
                    tracing::info!(slot = b, total, "gpu: packed KV admission waiting");
                    Ok(None)
                } else {
                    Err(error)
                }
            }
        }
    }


    fn upload_active_slots(&mut self, slots: impl Iterator<Item = usize>) -> Result<()> {
        let Some(state) = &self.recurrent else {
            return Ok(());
        };
        let mask: &mut [i32] = bytemuck::cast_slice_mut(self.stage.section_mut(3));
        mask.fill(0);
        for slot in slots {
            mask[slot] = 1;
        }
        // The pinned mask survives until the step sync or prompt H2D event.
        unsafe {
            self.be.memcpy_htod_async(
                self.devp[state.active].base,
                self.stage.section(3),
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// One batched decode step: feed `(slot, token)` for every live slot,
    /// advance each fed slot's KV cache by one row, and write the
    /// device-argmax next token per feed (same order) into `toks` — cleared
    /// first; caller-provided so the per-token hot path allocates nothing.
    /// Port of the harness `run_step` (kv-row patch + ids/pos/kvlen + counter
    /// zero + cooperative launch + `in.ids` readback), generalized to B rows:
    /// every row `b` carries its own `pos[b]`/`kvlen[b]`; rows not fed this
    /// step get `kvlen = 1` and their own `pos[b]` (the row a garbage KV
    /// write lands in is exactly the row the slot's next real step
    /// overwrites, so idle rows never corrupt a live sequence).
    ///
    /// Submission is asynchronous on the engine stream: patch, uploads,
    /// counter re-arm, launch, and the token D2H are enqueued back-to-back
    /// and retired by ONE `cuStreamSynchronize` — no `cuCtxSynchronize`.
    pub fn step_slots(&mut self, feeds: &[(usize, u32)], toks: &mut Vec<u32>) -> Result<()> {
        self.step_slots_sampled(feeds, None, toks)
    }

    /// [`Self::step_slots`] with optional per-slot device sampling (plan
    /// stage 4). `specs`, if given, is indexed by slot (len == batch); a row
    /// with `temp > 0` is sampled on-device (the `plow_sample` kernel launches
    /// after the decode kernel and overwrites `in.ids[b]` with the sampled
    /// token, so `toks[b]` is the final token — no vocab-row D2H). `temp <= 0`
    /// rows keep the greedy `ARGMAX_FIN` token. `None`, an all-greedy `specs`,
    /// or no loaded sampler → the greedy path, byte-identical to before.
    pub fn step_slots_sampled(
        &mut self,
        feeds: &[(usize, u32)],
        specs: Option<&[DevSample]>,
        toks: &mut Vec<u32>,
    ) -> Result<()> {
        toks.clear();
        if feeds.is_empty() {
            return Ok(());
        }
        let bsz = self.batch;
        for &(b, _) in feeds {
            if b >= bsz {
                return Err(RuntimeError::Rejected(format!(
                    "slot {b} out of range (engine batch {bsz})"
                )));
            }
            if self.pos[b] as usize >= self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "context exhausted at {} (compiled max {})",
                    self.pos[b], self.max_ctx
                )));
            }
        }

        let rung = self.select_decode(feeds.iter().map(|&(slot, _)| slot))?;
        let launch_rows = self.selected_decode(rung).map_or(bsz, |r| r.rows);

        // VMM backstop: every row this launch writes (fed rows at pos, idle
        // rows' garbage write at their own pos) must be mapped. The
        // pre-mapper keeps this a lock-free frontier check in steady state.
        if let Some(v) = &mut self.vmm {
            if let Some(rings) = &mut v.rings {
                rings.ensure_prefix(launch_rows)?;
            }
            for b in 0..launch_rows {
                let need = self.pos[b] + 1;
                if v.kv.mapped_rows(b) < need {
                    v.kv.ensure_rows(b, need)?;
                }
            }
        }

        let timed = self.timing.is_some();
        let now = || std::time::Instant::now();
        let t_enter = timed.then(now);
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[0], &self.stream)?;
        }

        // B == 1 program: the KV write row is the host-patched `i[3]` (the
        // legacy single-ring formula). B > 1 programs ignore `i[3]` — the
        // kernel derives each row from `pos[b]` (`n_batch_kv > 1`).
        if bsz == 1 && !self.kvrow.is_empty() {
            let pos = self.pos[feeds[0].0];
            for &ix in &self.kvrow {
                self.h_inst[ix as usize].i[3] = pos;
            }
            let lo = self.kvrow_lo;
            let n = self.kvrow_hi - lo + 1;
            let sz = std::mem::size_of::<DevInst64>();
            // SAFETY: DevInst64 is a #[repr(C)] POD mirror; range within
            // h_inst, which lives on `self` past the stream synchronize.
            unsafe {
                let bytes =
                    std::slice::from_raw_parts(self.h_inst[lo..].as_ptr() as *const u8, n * sz);
                self.be.memcpy_htod_async(
                    self.d_inst.base + (lo * sz) as u64,
                    bytes,
                    &self.stream,
                )?;
            }
        }

        // Pinned staging (sized once at load) — no per-step allocation, and
        // the copies below are truly asynchronous.
        {
            let (ids, pos, kvlen) = self.stage.parts_mut();
            for b in 0..bsz {
                ids[b] = 0;
                kvlen[b] = 1;
                pos[b] = self.pos[b] as i32;
            }
            for &(b, token) in feeds {
                ids[b] = token as i32;
                kvlen[b] = self.pos[b] as i32 + 1;
            }
        }
        // SAFETY: the pinned slab lives on `self` past the stream synchronize;
        // each section matches its [B]-sized i32 tensor.
        unsafe {
            self.be.memcpy_htod_async(
                self.devp[self.t_ids].base,
                self.stage.section(0),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_pos].base,
                self.stage.section(1),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_kvlen].base,
                self.stage.section(2),
                &self.stream,
            )?;
        }
        self.upload_active_slots(feeds.iter().map(|&(slot, _)| slot))?;
        // Counters and the GQ cursor have the same lifecycle (one launch
        // consumes them) and share one allocation — one fill re-arms both.
        self.reset_selected_decode_counters(rung)?;
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[1], &self.stream)?;
        }

        self.launch_selected_decode(rung)?;
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[2], &self.stream)?;
        }

        // Device sampling (plan stage 4): if any fed row wants stochastic
        // sampling and the sampler is loaded, launch `plow_sample` on the
        // stream after the decode kernel. It reads act.logits (written by the
        // decode program; PLOW_FUSE_ARGMAX must be off, the default) and
        // OVERWRITES in.ids[b] for temp>0 rows; greedy rows keep the
        // ARGMAX_FIN token (the sampler's temp<=0 branch reproduces it).
        let want_sample =
            specs.is_some_and(|s| self.sampler.is_some() && s.iter().any(|x| x.temp > 0.0));
        if want_sample {
            let specs = specs.expect("checked");
            debug_assert_eq!(specs.len(), bsz, "specs must be batch-wide");
            // Fill the pinned param slab [temp|topk|topp|minp|rng], each B.
            let smp = self.sampler.as_mut().expect("checked");
            {
                let raw = smp.params.as_mut_slice();
                let (s_temp, r) = raw.split_at_mut(bsz * 4);
                let (s_topk, r) = r.split_at_mut(bsz * 4);
                let (s_topp, r) = r.split_at_mut(bsz * 4);
                let (s_minp, s_rng) = r.split_at_mut(bsz * 4);
                let temp: &mut [f32] = bytemuck::cast_slice_mut(s_temp);
                let topk: &mut [i32] = bytemuck::cast_slice_mut(s_topk);
                let topp: &mut [f32] = bytemuck::cast_slice_mut(s_topp);
                let minp: &mut [f32] = bytemuck::cast_slice_mut(s_minp);
                let rng: &mut [f32] = bytemuck::cast_slice_mut(s_rng);
                for b in 0..bsz {
                    temp[b] = specs[b].temp;
                    topk[b] = specs[b].top_k;
                    topp[b] = specs[b].top_p;
                    minp[b] = specs[b].min_p;
                    rng[b] = specs[b].rng01;
                }
            }
            let (sf, sdp, ses) = (smp.f, smp.d_params.base, smp.d_escratch.base);
            // SAFETY: pinned slab lives on self past the synchronize.
            unsafe {
                self.be
                    .memcpy_htod_async(sdp, smp.params.as_slice(), &self.stream)?;
            }
            let dp = |k: u64| sdp + k * (bsz * 4) as u64;
            let mut a_logits = self.devp[self.t_logits].base;
            let mut a_ids = self.devp[self.t_ids].base;
            let (mut a_temp, mut a_topk, mut a_topp) = (dp(0), dp(1), dp(2));
            let (mut a_minp, mut a_rng, mut a_es) = (dp(3), dp(4), ses);
            let (mut a_v, mut a_b) = (self.vocab as u32, bsz as u32);
            let mut a = [
                &mut a_logits as *mut u64 as *mut std::ffi::c_void,
                &mut a_ids as *mut u64 as *mut std::ffi::c_void,
                &mut a_temp as *mut u64 as *mut std::ffi::c_void,
                &mut a_topk as *mut u64 as *mut std::ffi::c_void,
                &mut a_topp as *mut u64 as *mut std::ffi::c_void,
                &mut a_minp as *mut u64 as *mut std::ffi::c_void,
                &mut a_rng as *mut u64 as *mut std::ffi::c_void,
                &mut a_es as *mut u64 as *mut std::ffi::c_void,
                &mut a_v as *mut u32 as *mut std::ffi::c_void,
                &mut a_b as *mut u32 as *mut std::ffi::c_void,
            ];
            self.be
                .launch_kernel(sf, bsz as u32, 256, 0, &mut a, Some(&self.stream))?;
        }

        // Token readback: `in.ids` (rewritten by `ARGMAX_FIN`, or by the
        // device sampler above) round-trips through the pinned ids section.
        // SAFETY: as the uploads — the slab outlives the synchronize.
        unsafe {
            let base = self.devp[self.t_ids].base;
            self.be
                .memcpy_dtoh_async(self.stage.section_mut(0), base, &self.stream)?;
        }
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[3], &self.stream)?;
        }
        let t_submit = timed.then(now);

        // The step's single sync point (the whole queue retires in order).
        // A failure here is where an async kernel trap surfaces — capture the
        // launch shape at the site before propagating.
        if let Err(e) = self.be.stream_synchronize(&self.stream) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                fed = feeds.len(),
                batch = bsz,
                launch_rows,
                grid = self.grid,
                block = BLOCK,
                smem = self.smem,
                "decode step: stream sync failed"
            );
            return Err(e);
        }
        let t_sync = timed.then(now);

        toks.reserve(feeds.len());
        for &(b, tok) in feeds {
            toks.push(self.stage.token(b));
            self.pos[b] += 1;
            if self.vmm_prefix_enabled() {
                self.seq_tokens[b].push(tok);
            }
        }
        // VMM: hint the pre-mapper so the next block is mapped before the
        // frontier reaches it (map-during-decode is safe — probe [5]).
        if let Some(v) = &self.vmm {
            for &(b, _) in feeds {
                v.kv.advise(b, self.pos[b]);
            }
        }

        if let Some(t) = self.timing.as_mut() {
            let (e, s0, s1) = (
                t_enter.expect("timed"),
                t_submit.expect("timed"),
                t_sync.expect("timed"),
            );
            t.upload_ms += self.be.event_elapsed_ms(&t.ev[0], &t.ev[1])? as f64;
            t.kernel_ms += self.be.event_elapsed_ms(&t.ev[1], &t.ev[2])? as f64;
            t.download_ms += self.be.event_elapsed_ms(&t.ev[2], &t.ev[3])? as f64;
            if let Some(prev) = t.last_end {
                t.gap_ns += (e - prev).as_nanos() as u64;
            }
            t.submit_ns += (s0 - e).as_nanos() as u64;
            t.sync_ns += (s1 - s0).as_nanos() as u64;
            t.last_end = Some(std::time::Instant::now());
            t.steps += 1;
            t.log_every(128);
        }
        Ok(())
    }

    /// Decode-loop prompt consumption: one T=1 launch per token, D2H + stream
    /// sync only after the last. Token-identical to calling [`Self::step_slots`]
    /// once per id. This is the TTFT fallback when no compatible prefill
    /// object is available. Pinned staging is reused; an H2D-complete event
    /// (not a kernel wait) gates overwrite so kernels stay overlapped.
    pub fn consume_prompt(
        &mut self,
        slot: usize,
        tokens: &[u32],
        toks: &mut Vec<u32>,
    ) -> Result<u32> {
        toks.clear();
        if tokens.is_empty() {
            return Err(RuntimeError::Rejected("empty prompt".into()));
        }
        let bsz = self.batch;
        if slot >= bsz {
            return Err(RuntimeError::Rejected(format!(
                "slot {slot} out of range (engine batch {bsz})"
            )));
        }
        if self.pos[slot] as usize + tokens.len() > self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "context exhausted at {} (compiled max {})",
                self.pos[slot], self.max_ctx
            )));
        }
        // Map the whole prompt span once. Per-token `step_slots` only needs
        // `pos+1`; a cold walk of L tokens would otherwise ensure L times.
        if self.vmm.is_some() {
            let launch_rows = self
                .decode_rung(slot)
                .map_or(bsz, |ix| self.decode_rungs[ix].rows);
            let v = self.vmm.as_mut().expect("checked");
            if let Some(rings) = &mut v.rings {
                rings.ensure_prefix(launch_rows)?;
            }
            let need = self.pos[slot] + tokens.len() as u32;
            if v.kv.mapped_rows(slot) < need {
                v.kv.ensure_rows(slot, need)?;
            }
            for b in 0..launch_rows {
                if b == slot {
                    continue;
                }
                let n = self.pos[b] + 1;
                if v.kv.mapped_rows(b) < n {
                    v.kv.ensure_rows(b, n)?;
                }
            }
        }

        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            if i > 0 {
                self.be.event_synchronize(&self.h2d_ev)?;
            }
            self.enqueue_prompt_token(slot, token)?;
            if i == last {
                self.retire_prompt_token(slot, token, toks)?;
            } else {
                self.pos[slot] += 1;
                if self.vmm_prefix_enabled() {
                    self.seq_tokens[slot].push(token);
                }
                if let Some(v) = &self.vmm {
                    v.kv.advise(slot, self.pos[slot]);
                }
            }
        }
        Ok(toks[0])
    }

    /// Patch + H2D + memset + cooperative launch for one prompt token.
    /// Records `h2d_ev` after the copies so the next fill may reuse `stage`
    /// without waiting on the interpreter.
    fn enqueue_prompt_token(&mut self, slot: usize, token: u32) -> Result<()> {
        let bsz = self.batch;
        let rung = self.select_decode(std::iter::once(slot))?;
        if bsz == 1 && !self.kvrow.is_empty() {
            let pos = self.pos[slot];
            for &ix in &self.kvrow {
                self.h_inst[ix as usize].i[3] = pos;
            }
            let lo = self.kvrow_lo;
            let n = self.kvrow_hi - lo + 1;
            let sz = std::mem::size_of::<DevInst64>();
            // SAFETY: DevInst64 is a #[repr(C)] POD mirror; range within
            // h_inst, which lives on `self` past the stream synchronize.
            unsafe {
                let bytes =
                    std::slice::from_raw_parts(self.h_inst[lo..].as_ptr() as *const u8, n * sz);
                self.be.memcpy_htod_async(
                    self.d_inst.base + (lo * sz) as u64,
                    bytes,
                    &self.stream,
                )?;
            }
        }
        {
            let (ids, pos, kvlen) = self.stage.parts_mut();
            for b in 0..bsz {
                ids[b] = 0;
                kvlen[b] = 1;
                pos[b] = self.pos[b] as i32;
            }
            ids[slot] = token as i32;
            kvlen[slot] = self.pos[slot] as i32 + 1;
        }
        // SAFETY: the pinned slab lives on `self` past the stream synchronize;
        // each section matches its [B]-sized i32 tensor.
        unsafe {
            self.be.memcpy_htod_async(
                self.devp[self.t_ids].base,
                self.stage.section(0),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_pos].base,
                self.stage.section(1),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_kvlen].base,
                self.stage.section(2),
                &self.stream,
            )?;
        }
        self.upload_active_slots(std::iter::once(slot))?;
        self.be.event_record(&self.h2d_ev, &self.stream)?;
        self.reset_selected_decode_counters(rung)?;
        self.launch_selected_decode(rung)?;
        Ok(())
    }

    fn decode_rung(&self, highest_slot: usize) -> Option<usize> {
        decode_rung_index(self.decode_rungs.iter().map(|r| r.rows), highest_slot)
    }

    fn select_decode(&self, slots: impl Iterator<Item = usize> + Clone) -> Result<DecodeSelection> {
        let highest = slots
            .clone()
            .max()
            .ok_or_else(|| RuntimeError::Rejected("empty decode selection".into()))?;
        let band = self
            .decode_contexts
            .as_ref()
            .map(|contexts| contexts.select(&self.pos, slots))
            .transpose()?
            .flatten();
        Ok(decode_selection(self.decode_rung(highest), band))
    }

    fn selected_decode(&self, selection: DecodeSelection) -> Option<&DecodeRung> {
        match selection {
            DecodeSelection::Base(index) => index.map(|i| &self.decode_rungs[i]),
            DecodeSelection::Context(index) => Some(
                self.decode_contexts
                    .as_ref()
                    .expect("selected loaded context")
                    .rung(index),
            ),
        }
    }

    fn reset_selected_decode_counters(&self, selection: DecodeSelection) -> Result<()> {
        if let Some(r) = self.selected_decode(selection) {
            self.be
                .memset_d8_async(r.counters.base, 0, r.counter_bytes, &self.stream)
        } else {
            self.reset_decode_counters()
        }
    }

    fn launch_selected_decode(&mut self, selection: DecodeSelection) -> Result<()> {
        if let Some(r) = self.selected_decode(selection) {
            if let Some(library) = &r.library {
                return library.launch(&self.stream);
            }
            let mut arg = r.kernarg;
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            let object = r.object.as_deref();
            self.be.launch_cooperative(
                object.map_or(self.f, |o| o.function),
                object.map_or(self.grid, |o| o.grid),
                object.map_or(BLOCK, |o| o.block),
                object.map_or(self.smem, |o| o.smem),
                &mut params,
                Some(&self.stream),
            )
        } else {
            self.launch_decode()
        }
    }

    fn reset_decode_counters(&self) -> Result<()> {
        let bytes = self.ctr_bytes.max(4) + self.cursor_bytes;
        self.be
            .memset_d8_async(self.d_ctr.base, 0, bytes, &self.stream)
    }

    fn retire_prompt_token(&mut self, slot: usize, token: u32, toks: &mut Vec<u32>) -> Result<()> {
        // SAFETY: as the uploads — the slab outlives the synchronize.
        unsafe {
            let base = self.devp[self.t_ids].base;
            self.be
                .memcpy_dtoh_async(self.stage.section_mut(0), base, &self.stream)?;
        }
        if let Err(e) = self.be.stream_synchronize(&self.stream) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                slot,
                grid = self.grid,
                block = BLOCK,
                smem = self.smem,
                "consume_prompt: stream sync failed"
            );
            return Err(e);
        }
        toks.clear();
        toks.push(self.stage.token(slot));
        self.pos[slot] += 1;
        if self.vmm_prefix_enabled() {
            self.seq_tokens[slot].push(token);
        }
        if let Some(v) = &self.vmm {
            v.kv.advise(slot, self.pos[slot]);
        }
        Ok(())
    }

    /// Bounded device multi-step decode (plan stage 5; `PLOW_MULTISTEP=K`).
    /// GREEDY only. Runs a K-token quantum for every fed row with ONE host
    /// sync: the host uploads ids/pos/kvlen ONCE, then enqueues
    /// `[memset → decode → advance] × K` on the stream — the `plow_advance`
    /// kernel advances each row's device-owned pos/kvlen and appends its token
    /// to the ring between launches, so no per-token host round trip happens.
    /// `out` is filled row-major (`feeds.len() × K`, fed row r → `out[r*K..]`)
    /// and the executed `K` is returned, capped by remaining context.
    /// Token-identical to K [`Self::step_slots`] calls.
    pub fn multi_step(&mut self, feeds: &[(usize, u32)], out: &mut Vec<u32>) -> Result<usize> {
        self.multi_step_at_most(feeds, usize::MAX, out)
    }

    /// Cap a quantum by the caller's remaining output budget and live context.
    pub fn multi_step_at_most(
        &mut self,
        feeds: &[(usize, u32)],
        requested: usize,
        out: &mut Vec<u32>,
    ) -> Result<usize> {
        out.clear();
        let Some(ms) = self.multistep.as_ref() else {
            return Err(RuntimeError::Rejected("multi-step not enabled".into()));
        };
        let f_adv = ms.f_advance;
        let k = crate::sched::multistep::decode_quantum(
            feeds.iter().map(|&(slot, _)| slot),
            &self.pos,
            self.max_ctx,
            requested,
            ms.quantum,
        )
        .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
        if k == 0 {
            return Ok(k);
        }
        let bsz = self.batch;
        let rung = self.select_decode(feeds.iter().map(|&(slot, _)| slot))?;
        let launch_rows = self.selected_decode(rung).map_or(bsz, |r| r.rows);
        // VMM: map every row this quantum will write. Fed rows need the full
        // quantum; idle rows in the selected rung need their one garbage row.
        if let Some(v) = &mut self.vmm {
            if let Some(rings) = &mut v.rings {
                rings.ensure_prefix(launch_rows)?;
            }
            for b in 0..launch_rows {
                let active = feeds.iter().any(|&(slot, _)| slot == b);
                let need = self.pos[b] + if active { k as u32 } else { 1 };
                if v.kv.mapped_rows(b) < need {
                    v.kv.ensure_rows(b, need)?;
                }
            }
        }

        // Active-row flags (advance only touches fed rows; idle pos must not drift).
        {
            let fed: &mut [i32] = bytemuck::cast_slice_mut(self.mst_fed_host_mut());
            fed[..bsz].fill(0);
            for &(b, _) in feeds {
                fed[b] = 1;
            }
        }
        // Step-0 run state (subsequent steps advance device-side).
        {
            let (ids, pos, kvlen) = self.stage.parts_mut();
            for b in 0..bsz {
                ids[b] = 0;
                kvlen[b] = 1;
                pos[b] = self.pos[b] as i32;
            }
            for &(b, token) in feeds {
                ids[b] = token as i32;
                kvlen[b] = self.pos[b] as i32 + 1;
            }
        }
        // Uploads (once). fed + ids/pos/kvlen.
        let (ring_base, fed_base) = {
            let ms = self.multistep.as_ref().expect("checked");
            (ms.d_ring.base, ms.d_fed.base)
        };
        // SAFETY: pinned slabs live on self past the synchronize; sections match
        // their [B]-sized i32 tensors.
        unsafe {
            let ms = self.multistep.as_ref().expect("checked");
            self.be
                .memcpy_htod_async(fed_base, ms.fed_host.as_slice(), &self.stream)?;
            self.be.memcpy_htod_async(
                self.devp[self.t_ids].base,
                self.stage.section(0),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_pos].base,
                self.stage.section(1),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_kvlen].base,
                self.stage.section(2),
                &self.stream,
            )?;
        }

        // Enqueue [memset → decode → advance] × K on the stream — no sync.
        let advance_grid = (bsz as u32).div_ceil(256);
        for step in 0..k {
            self.reset_selected_decode_counters(rung)?;
            self.launch_selected_decode(rung)?;
            let mut a_ids = self.devp[self.t_ids].base;
            let mut a_pos = self.devp[self.t_pos].base;
            let mut a_kvl = self.devp[self.t_kvlen].base;
            let mut a_ring = ring_base;
            let mut a_fed = fed_base;
            let (mut a_step, mut a_k, mut a_b) = (step as u32, k as u32, bsz as u32);
            let mut a = [
                &mut a_ids as *mut u64 as *mut std::ffi::c_void,
                &mut a_pos as *mut u64 as *mut std::ffi::c_void,
                &mut a_kvl as *mut u64 as *mut std::ffi::c_void,
                &mut a_ring as *mut u64 as *mut std::ffi::c_void,
                &mut a_fed as *mut u64 as *mut std::ffi::c_void,
                &mut a_step as *mut u32 as *mut std::ffi::c_void,
                &mut a_k as *mut u32 as *mut std::ffi::c_void,
                &mut a_b as *mut u32 as *mut std::ffi::c_void,
            ];
            self.be
                .launch_kernel(f_adv, advance_grid, 256, 0, &mut a, Some(&self.stream))?;
        }

        // One D2H of the whole ring, then the single sync.
        // SAFETY: ring_host lives on self past the synchronize.
        unsafe {
            let ms = self.multistep.as_mut().expect("checked");
            let ring_slice = ms.ring_host.as_mut_slice();
            self.be
                .memcpy_dtoh_async(ring_slice, ring_base, &self.stream)?;
        }
        if let Err(e) = self.be.stream_synchronize(&self.stream) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                fed = feeds.len(),
                quantum = k,
                grid = self.grid,
                block = BLOCK,
                smem = self.smem,
                "multi-step: stream sync failed"
            );
            return Err(e);
        }

        // Extract each fed row's K tokens (row-major) and advance host pos.
        {
            let ms = self.multistep.as_ref().expect("checked");
            let ring: &[i32] = bytemuck::cast_slice(ms.ring_host.as_slice());
            out.reserve(feeds.len() * k);
            for &(b, _) in feeds {
                for step in 0..k {
                    out.push(ring[b * k + step] as u32);
                }
            }
        }
        for (ri, &(b, tok)) in feeds.iter().enumerate() {
            self.pos[b] += k as u32;
            // Rows written this quantum: the fed token, then the device's own
            // feed chain (the first k-1 produced tokens).
            if self.vmm_prefix_enabled() {
                self.seq_tokens[b].push(tok);
                self.seq_tokens[b].extend_from_slice(&out[ri * k..ri * k + (k - 1)]);
            }
            if let Some(v) = &self.vmm {
                v.kv.advise(b, self.pos[b]);
            }
        }
        Ok(k)
    }

    /// Mutable pinned active-flag staging for [`Self::multi_step`].
    fn mst_fed_host_mut(&mut self) -> &mut [u8] {
        self.multistep
            .as_mut()
            .expect("multi-step enabled")
            .fed_host
            .as_mut_slice()
    }

    /// Whether bounded device multi-step is enabled, and its quantum K.
    pub fn multistep_quantum(&self) -> Option<usize> {
        self.multistep.as_ref().map(|m| m.quantum)
    }

    /// Whether the prefill object + bucket programs are loaded.
    pub fn has_prefill(&self) -> bool {
        self.f_pf.is_some() && !self.prefill.is_empty()
    }

    /// Whether the device stochastic sampler is loaded (`PLOW_DEV_SAMPLE=1` +
    /// a sampler cubin). When true the mux routes eligible `temperature>0`
    /// rows through [`Self::step_slots_sampled`] instead of the vocab-row D2H.
    pub fn dev_sample_enabled(&self) -> bool {
        self.sampler.is_some()
    }

    /// Whether packet-selected or legacy opt-in cross-request prefill is active.
    /// The mux routes all prefill through [`Self::prefill_batched`] and takes
    /// each request's first token from a decode step of its last prompt token.
    pub fn pf_batch_enabled(&self) -> bool {
        self.pf_batch.is_some()
    }

    pub fn prefill_turn(&self) -> usize {
        self.prefill_turn % self.batch.max(1)
    }

    pub fn advance_prefill_turn(&mut self, slot: usize) {
        self.prefill_turn = (slot + 1) % self.batch.max(1);
    }

    /// Largest prefill bucket's row count — the mux's per-launch token budget.
    pub fn pf_max_rows(&self) -> usize {
        self.prefill.last().map_or(0, |b| b.t as usize)
    }

    pub fn pf_request_max_rows(&self) -> usize {
        self.packed_prefill
            .as_ref()
            .and_then(|p| p.max_request_rows)
            .map_or_else(|| self.pf_max_rows(), |rows| rows as usize)
    }

    /// Pack budget for `avail` waiting prefill rows (PX-1 batched path), in
    /// rows. Delegates to [`Self::pick_prefill_bucket`] so the batched and
    /// serialized paths share ONE policy — the cost-aware pick that charges
    /// each launch its real weight-restream cost. It had its own copy of the
    /// minimize-padded-rows rule, which cascaded the same way: a 8190-row pack
    /// budgeted 4096 where `[8192]` is one launch with 2 rows of padding.
    ///
    /// Still keeps a cold 4.5k-row pack out of the ~45%-padded 8192 bucket:
    /// `[4096, tail]` stays cheaper than a single `[8192]` under the cost model.
    pub fn pf_pack_budget(&self, avail: usize) -> usize {
        if self.prefill.is_empty() {
            return 0;
        }
        self.prefill[self.pick_prefill_bucket(avail, usize::MAX)].t as usize
    }

    /// Bring up the `_pf` prefill object and upload every prefill bucket program.
    /// Port of the harness `prep_prog` for the prefill objects:
    /// upload tables, verify the single coarse segment, and precompute the
    /// per-chunk patch sites. Returns `Err` (prefill then disabled) if the cubin
    /// grid disagrees with `n_cu`, a bucket is segmented, or the GQ appendix is
    /// missing.
    fn load_prefill(
        be: &Arc<CudaBackend>,
        pf: InterpImage,
        blob: &DevBlob,
        assets_dir: &Path,
        d_tens: u64,
        grid: u32,
        interp_tag: &str,
        segment_roles: Option<&SegmentRoles>,
        packed: Option<&plow_asset::packed_prefill::Manifest>,
    ) -> Result<(KernelFn, u32, Module, Vec<PrefillBucket>, Option<SegPf>)> {
        let packed_requests = packed.is_some();
        let module = be.module_load(&pf.image)?;
        check_norm_weight_offset(be, &module, blob)?;
        Self::check_packet_pairing_suffix(be, &module, assets_dir, "_pf")?;
        let kname = crate::config::RuntimeConfig::get()
            .nv
            .kernel_pf
            .clone()
            .unwrap_or(pf.entry);
        let f_pf = be.get_function(&module, &kname)?;

        // Same contract as decode: `--nv-smem-pf` / PLOW_NV_SMEM_PF override >
        // the cubin's own `plow_arena_bytes_pf` metadata > legacy default.
        let smem_pf: u32 = match crate::config::RuntimeConfig::get().nv.smem_pf {
            Some(v) => v,
            None => be
                .module_global_u32(&module, "plow_arena_bytes_pf")?
                .unwrap_or(SMEM_PF),
        };
        be.set_max_dynamic_smem(f_pf, smem_pf)?;

        // Fine-gated packets select the segmented pair from their own asset directory.
        // `--pf-seg-dir` remains an explicit object-directory override.
        let small_gemm_path = crate::config::RuntimeConfig::get()
            .nv
            .pf_seg_gemm_small
            .as_deref();
        let configured_seg_dir = crate::config::RuntimeConfig::get()
            .nv
            .pf_seg_dir
            .as_deref()
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from);
        let segment_dir = configured_seg_dir.or_else(|| {
            prefill_needs_segment_pair(blob, segment_roles).then(|| assets_dir.to_path_buf())
        });
        if small_gemm_path.is_some() && segment_dir.is_none() {
            return Err(RuntimeError::Rejected(
                "PLOW_PF_SEG_GEMM_SMALL requires PLOW_PF_SEG_DIR".into(),
            ));
        }
        let seg_pf = match segment_dir {
            Some(dir) => {
                if !packed_requests
                    && blob.progs.iter().flat_map(|p| &p.insts).any(|inst| {
                        inst.op == DevOp::HeadNormRopeFp8 as u16
                            || inst.op == DevOp::FlashPrefillFp8 as u16
                    })
                {
                    return Err(RuntimeError::Device(
                        "FP8-KV segmented prefill requires a compiled packed request contract"
                            .into(),
                    ));
                }
                let load = |file: &str,
                            sym: &str,
                            suffix: &str,
                            arena: &str|
                 -> Result<(Module, KernelFn, u32, u32)> {
                    let img = std::fs::read(dir.join(file)).map_err(|e| {
                        RuntimeError::Device(format!("segmented prefill object: read {file}: {e}"))
                    })?;
                    if let Some(pack) = packed {
                        pack.validate_object(|name| cubin::global_u32(&img, name))
                            .map_err(|e| RuntimeError::Rejected(format!("{file}: {e}")))?;
                    }
                    let m = be.module_load(&img)?;
                    check_norm_weight_offset(be, &m, blob)?;
                    Self::check_packet_pairing_suffix(be, &m, assets_dir, suffix)?;
                    let f = be.get_function(&m, sym)?;
                    let sm = be.module_global_u32(&m, arena)?.unwrap_or(smem_pf);
                    be.set_max_dynamic_smem(f, sm)?;
                    let occ = be.occupancy_blocks_per_sm(f, BLOCK, sm as usize)?;
                    Ok((m, f, sm, occ * be.sm_count()))
                };
                let suffix = if packed_requests { "pfpacked" } else { "pf" };
                let kv_suffix = if packed.is_some_and(|p| p.version == 2) {
                    "_fp8kv"
                } else {
                    ""
                };
                let seg_file = format!("interp_{interp_tag}_{suffix}seg{kv_suffix}.cubin");
                let gemm_file = format!("interp_{interp_tag}_{suffix}gemm{kv_suffix}.cubin");
                let seg_sym_name = format!("interp_{interp_tag}_{suffix}seg");
                let gemm_sym_name = format!("interp_{interp_tag}_{suffix}gemm");
                let seg_sym = format!("_Z{}{}11PlowProgram", seg_sym_name.len(), seg_sym_name);
                let gemm_sym = format!("_Z{}{}11PlowProgram", gemm_sym_name.len(), gemm_sym_name);
                let seg_global_suffix = format!("_{suffix}seg");
                let gemm_global_suffix = format!("_{suffix}gemm");
                let seg_arena = format!("plow_arena_bytes{seg_global_suffix}");
                let gemm_arena = format!("plow_arena_bytes{gemm_global_suffix}");
                let (m1, f1, s1, g1) = load(&seg_file, &seg_sym, &seg_global_suffix, &seg_arena)?;
                let (m2, f2, s2, _g2unused) =
                    load(&gemm_file, &gemm_sym, &gemm_global_suffix, &gemm_arena)?;
                if crate::config::RuntimeConfig::get().nv.pf_seg_pure.as_deref() == Some("w8a16")
                    && be.module_global_u32(
                        &m2,
                        &format!("plow_w8a16_gemm_abi{gemm_global_suffix}"),
                    )? != Some(1)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "w8a16 segments require a compatible W8A16 GEMM object: {gemm_file}"
                    )));
                }
                // T31: the GEMM object may declare its own launch block size (384-thread ws).
                let gemm_block = format!("plow_block{gemm_global_suffix}");
                let blk2 = be.module_global_u32(&m2, &gemm_block)?.unwrap_or(BLOCK);
                let g2 = be.occupancy_blocks_per_sm(f2, blk2, s2 as usize)? * be.sm_count();
                let small_gemm = if let Some(path) = small_gemm_path {
                    if blk2 != 384 {
                        return Err(RuntimeError::Rejected(
                            "small GEMM selection requires a WS384 default GEMM object".into(),
                        ));
                    }
                    let img = std::fs::read(path).map_err(|e| {
                        RuntimeError::Device(format!("PLOW_PF_SEG_GEMM_SMALL: read {path}: {e}"))
                    })?;
                    if let Some(pack) = packed {
                        pack.validate_object(|name| cubin::global_u32(&img, name))
                            .map_err(RuntimeError::Rejected)?;
                    }
                    let module = be.module_load(&img)?;
                    let abi = be
                        .module_global_u32(&module, "plow_gemm_shape_abi_pfgemm")?
                        .unwrap_or(0);
                    if !matches!(abi, 1 | 2 | 3)
                        || be.module_global_u32(&module, "plow_block_pfgemm")? != Some(BLOCK)
                    {
                        return Err(RuntimeError::Rejected(
                            "small GEMM object requires native BF16 TMA ABI 1, 2 or 3 and 256 threads"
                                .into(),
                        ));
                    }
                    let function = be.get_function(&module, &gemm_sym)?;
                    let smem = be
                        .module_global_u32(&module, "plow_arena_bytes_pfgemm")?
                        .filter(|v| *v > 0)
                        .ok_or_else(|| {
                            RuntimeError::Rejected(
                                "small GEMM object is missing its shared-memory requirement".into(),
                            )
                        })?;
                    be.set_max_dynamic_smem(function, smem)?;
                    let grid =
                        be.occupancy_blocks_per_sm(function, BLOCK, smem as usize)? * be.sm_count();
                    if grid == 0 {
                        return Err(RuntimeError::Rejected(
                            "small GEMM object has no resident blocks".into(),
                        ));
                    }
                    tracing::info!(path, grid, smem, "experimental small GEMM object loaded");
                    Some(SegGemm {
                        abi,
                        function,
                        smem,
                        grid,
                        block: BLOCK,
                        _module: module,
                    })
                } else {
                    None
                };
                // Optional third object (T12): dedicated hd512 flash. Only loaded when the
                // file exists — the classing env (PLOW_PF_SEG_FA512) decides whether class-2
                // segments are emitted at all.
                let fa_file = format!("interp_{interp_tag}_{suffix}fa{kv_suffix}.cubin");
                let fa = if dir.join(&fa_file).exists() {
                    let fa_sym_name = format!("interp_{interp_tag}_{suffix}fa");
                    let fa_sym = format!("_Z{}{}11PlowProgram", fa_sym_name.len(), fa_sym_name);
                    let fa_global_suffix = format!("_{suffix}fa");
                    let fa_arena = format!("plow_arena_bytes{fa_global_suffix}");
                    let (m3, f3, s3, g3) =
                        load(&fa_file, &fa_sym, &fa_global_suffix, &fa_arena)?;
                    // PLOW_PF_SEG_FA512=all classes hd256 FlashPrefill onto this object too,
                    // but its hd256 arm exists only when built PLOW_BUILD_FA_HD256=1 —
                    // without it the dispatch hits a bare __trap(): LAUNCH_FAILED, poisoned
                    // context, dead engine, every request 503. Refuse the mismatch here, the
                    // way a missing object is already refused. Absent symbol = older cubin,
                    // unconstrained (same convention as plow_arena_bytes).
                    if crate::config::RuntimeConfig::get()
                        .nv
                        .pf_seg_fa512
                        .as_deref()
                        == Some("all")
                        && be.module_global_u32(&m3, &format!("plow_fa_hd256{fa_global_suffix}"))? == Some(0)
                    {
                        return Err(RuntimeError::Device(format!(
                            "PLOW_PF_SEG_FA512=all classes hd256 flash onto {fa_file}, but that \
                             object was built without its hd256 arm (PLOW_BUILD_FA_HD256=1) — \
                             it would trap on the first hd256 segment. Rebuild the object or set \
                             PLOW_PF_SEG_FA512=1."
                        )));
                    }
                    Some((m3, f3, s3, g3))
                } else {
                    None
                };
                tracing::info!(
                    grid_flash = g1,
                    grid_gemm = g2,
                    smem_flash = s1,
                    smem_gemm = s2,
                    block_gemm = blk2,
                    fa512 = fa.is_some(),
                    "segmented prefill pair loaded"
                );
                let (m_fa, fa512) = match fa {
                    Some((m3, f3, s3, g3)) => (Some(m3), Some((f3, s3, g3))),
                    None => (None, None),
                };
                let mut sp = SegPf {
                    f_flash: f1,
                    smem_flash: s1,
                    grid_flash: g1,
                    f_gemm: f2,
                    smem_gemm: s2,
                    grid_gemm: g2,
                    block_gemm: blk2,
                    small_gemm,
                    fa512,
                    _m_flash: m1,
                    _m_gemm: m2,
                    _m_fa512: m_fa,
                };
                // PLOW_PF_SEG_EQSMEM=1 (T18): launch every object with the SAME dynamic-smem
                // request (the max of the three). Alternating smem sizes between back-to-back
                // launches forces an SM shared-memory carveout reconfig each time — measured
                // ~150-300us per transition at ~480 segments/chunk. A uniform request makes
                // every launch reuse the standing carveout. Occupancy re-derives from the max
                // (the fat object drops to occ-1 — measured neutral on its latency-bound rows).
                if crate::config::RuntimeConfig::get().nv.pf_seg_eqsmem {
                    let mut mx = sp.smem_flash.max(sp.smem_gemm);
                    if let Some(small) = &sp.small_gemm {
                        mx = mx.max(small.smem);
                    }
                    if let Some((_, s3, _)) = sp.fa512 {
                        mx = mx.max(s3);
                    }
                    // Occupancy is per (function, BLOCK SIZE): the ws384 GEMM object
                    // launches at `block_gemm`, so re-querying it at BLOCK over-reports
                    // its grid (384-thread blocks cap at 5/SM vs 8/SM at 256) and the
                    // cooperative launch then fails — or, non-cooperative, spins on
                    // counters owned by blocks that were never resident.
                    let requery = |f: KernelFn, block: u32| -> Result<u32> {
                        be.set_max_dynamic_smem(f, mx)?;
                        Ok(be.occupancy_blocks_per_sm(f, block, mx as usize)? * be.sm_count())
                    };
                    sp.grid_flash = requery(sp.f_flash, BLOCK)?;
                    sp.smem_flash = mx;
                    sp.grid_gemm = requery(sp.f_gemm, sp.block_gemm)?;
                    sp.smem_gemm = mx;
                    if let Some(small) = &mut sp.small_gemm {
                        small.grid = requery(small.function, small.block)?;
                        small.smem = mx;
                    }
                    if let Some((f3, _, _)) = sp.fa512 {
                        let g3 = requery(f3, BLOCK)?;
                        sp.fa512 = Some((f3, mx, g3));
                    }
                    tracing::info!(
                        smem = mx,
                        grid_flash = sp.grid_flash,
                        grid_gemm = sp.grid_gemm,
                        "seg pair smem equalized"
                    );
                }
                Some(sp)
            }
            _ => None,
        };
        let seg_mode = seg_pf.is_some();

        let occ = be.occupancy_blocks_per_sm(f_pf, BLOCK, smem_pf as usize)?;
        let grid_pf = occ * be.sm_count();

        let upload_pod = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        let mut buckets = Vec::new();
        for g in blob.prefill_progs() {
            // Wave-class segmented programs are legal exactly when the SegPf pair is
            // loaded: segments launch per class in order. Otherwise the coarse
            // single-segment contract holds as before.
            let program_index = blob
                .progs
                .iter()
                .position(|p| std::ptr::eq(p, g))
                .expect("prefill belongs to blob");
            let declared_roles = segment_roles.and_then(|r| r.program(program_index));
            let mut qwen_segments = qwen_prefill_segments(g, &blob.tensors)?;
            let packet_segment_roles = if let Some(p) = declared_roles {
                if qwen_segments.is_empty() && !seg_mode {
                    qwen_segments = vec![None; p.roles.len()];
                }
                let selected = packet_role_segments(g, &p.roles, &blob.tensors)?;
                tracing::info!(
                    program_index,
                    launches = p.roles.len(),
                    gemm_launches = selected.iter().filter(|&&x| x == 1).count(),
                    attention_launches = selected
                        .iter()
                        .filter(|&&role| {
                            matches!(
                                role,
                                plow_asset::segment_roles::PREFILL_ATTENTION
                                    | plow_asset::segment_roles::PREFILL_ATTENTION_HD512_WG32
                            )
                        })
                        .count(),
                    "packet segment roles loaded"
                );
                selected
            } else {
                Vec::new()
            };
            let qwen_program = g.insts.iter().any(|d| d.op == DevOp::QwenRmsNorm as u16);
            if qwen_program && g.insts.iter().any(|d| d.op == DevOp::GemmFp8 as u16) {
                check_qwen_w8a8_capability(
                    true,
                    g.t,
                    be.module_global_u32(&module, "plow_qwen_w8a8_prefill_arm")?,
                )?;
            }
            if (qwen_program || qwen_segments.iter().any(Option::is_some))
                && be.module_global_u32(&module, "plow_qwen_prefill_arm")? != Some(1)
            {
                return Err(RuntimeError::Rejected(
                    "Qwen prefill requires a paired prefill interpreter".into(),
                ));
            }
            let seg_class = if !qwen_segments.is_empty() {
                vec![4; qwen_segments.len()]
            } else if seg_mode {
                g.seg_classes()?
            } else {
                g.check_coarse_single_segment()?;
                Vec::new()
            };
            let small_gemm_segments = if seg_pf.as_ref().is_some_and(|s| s.small_gemm.is_some()) {
                let abi = seg_pf
                    .as_ref()
                    .and_then(|s| s.small_gemm.as_ref())
                    .expect("checked")
                    .abi;
                let selected = small_gemm_segments(g, &seg_class, abi)?;
                tracing::info!(
                    bucket = g.t,
                    selected = selected.iter().filter(|&&s| s).count(),
                    total = selected.len(),
                    "small GEMM segment selection"
                );
                selected
            } else {
                Vec::new()
            };
            let want_seg = if g.l2_domains != 0 {
                g.l2_domains as usize + 1
            } else if seg_mode || !qwen_segments.is_empty() {
                seg_class.len().max(1) + 1
            } else {
                2
            };
            if g.gq_stream.is_empty() || g.gq_seg_ofs.len() != want_seg {
                return Err(RuntimeError::Device(format!(
                    "prefill bucket T={} has no single-segment GQ appendix (n_seg bounds: {:?}) — \
                     recompile with `PLOW_UNISEG=1 plowc` (GQ-capable, single segment)",
                    g.t, g.gq_seg_ofs
                )));
            }
            g.check_gq_topological()?;

            let mut h_inst = g.insts.clone();
            for inst in &mut h_inst {
                if inst.op == DevOp::QwenGdnPrefill as u16 {
                    inst.op = DevOp::Nop as u16;
                }
            }
            let d_inst = upload_pod(pod_bytes(&h_inst))?;
            let d_stream = upload_pod(pod_bytes(&g.stream))?;
            let d_sofs = upload_pod(pod_bytes(&g.stream_ofs))?;
            let d_slen = upload_pod(pod_bytes(&g.stream_len))?;
            let d_waits = upload_pod(pod_bytes(&g.waits))?;
            let d_succs = upload_pod(pod_bytes(&g.succs))?;
            let d_gq_stream = upload_pod(pod_bytes(&g.gq_stream))?;
            let d_gq_seg = upload_pod(pod_bytes(&g.gq_seg_ofs))?;
            // Combined counter/cursor slab (plan: counter improvements #1):
            // the bucket's own GQ cursor sits at its counter block's tail, so
            // ONE fill re-arms both and no cursor is shared with the decode
            // program or other buckets. `ctr_bytes` is the full reset size.
            // ONE CURSOR LINE PER GQ SEGMENT: an L2-placed blob (PLOW_NV_PLACE)
            // carries P per-domain windows in gq_seg_ofs, and the placed
            // interpreter fetch-adds PLOW_CTR(gq_cursor, domain) — sizing for a
            // single line would put domains 1..P-1 past the allocation.
            let ctr_only = g.n_counter as usize * CTR_STRIDE as usize * 4;
            let cursor_off = ctr_only.max(4);
            let cursor_lines = g.gq_seg_ofs.len().saturating_sub(1).max(1);
            let ctr_bytes = cursor_off + cursor_lines * CTR_STRIDE as usize * 4;
            let d_ctr = be.alloc(0, ctr_bytes as u64)?;

            let kernarg = DevProgram {
                insts: d_inst.base,
                stream: d_stream.base,
                stream_ofs: d_sofs.base,
                stream_len: d_slen.base,
                waits: d_waits.base,
                succs: d_succs.base,
                counters: d_ctr.base,
                tensors: d_tens,
                trace: 0,
                cur_seg: 0,
                l2_domains: 0,
                hier_base: 0,
                n_seg: seg_class.len().max(1) as u32,
                gq_stream: d_gq_stream.base,
                gq_seg_ofs: d_gq_seg.base,
                gq_cursor: d_ctr.base + cursor_off as u64,
                xctr: 0,
                peer_scratch: 0,
                rank: 0,
                n_gpu: 1,
                seg_ofs: 0,
                prefill_spans: 0,
                prefill_parked: 0,
                n_prefill_spans: 0,
                n_prefill_rows: 0,
                token_batch: 0,
            };

            // Precompute the per-chunk patch sites (harness inner loop): KV-write
            // HeadNormRope (j[0]!=0), FlashPrefill, the M==1 lm_head GEMM, and
            // (PX-1 batched mode) the FlashMerge sites to neuter.
            let (mut rope, mut flash, mut lmhead, mut merge) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            // fp8-KV packets emit the Fp8 TWINS of these opcodes. They carry the
            // SAME operands at the same indices (rope: i[3]=out_row0,
            // fj[1]=out_stride; flash: i[1]=seq_kv, i[4]=q_pos0 — see the
            // HEADNORM_ROPE_FP8 / FLASH_PREFILL_FP8 arms in interp_sm120.cu), so
            // they need the IDENTICAL per-chunk patch. Matching only the bf16
            // opcodes left every fp8 chunk after the first writing its KV at
            // row 0 and reading with q_pos0=0 — a silent wrong answer for any
            // prompt long enough to need a second prefill launch (measured: a
            // single-launch prompt is correct, 8192+2048 is not), and a prefill
            // that also LOOKS 1.75x faster because the flash never grows past
            // the first chunk's keys (PX-17 measured exactly that on main).
            let mut fp8_kv = false;
            for (ix, inst) in g.insts.iter().enumerate() {
                if (inst.op == DevOp::HeadNormRope as u16
                    || inst.op == DevOp::HeadNormRopeFp8 as u16)
                    && inst.fj[1] != 0
                {
                    fp8_kv |= inst.op == DevOp::HeadNormRopeFp8 as u16;
                    rope.push(ix);
                } else if inst.op == DevOp::FlashPrefill as u16
                    || inst.op == DevOp::FlashPrefillFp8 as u16
                {
                    fp8_kv |= inst.op == DevOp::FlashPrefillFp8 as u16;
                    flash.push(ix);
                } else if inst.op == DevOp::FlashMerge as u16 {
                    merge.push(ix);
                } else if (inst.op == DevOp::Gemm as u16
                    || inst.op == DevOp::GemmSmall as u16
                    || inst.op == DevOp::GemmMed as u16)
                    && inst.i[0] == 1
                {
                    lmhead.push(ix);
                }
            }

            buckets.push(PrefillBucket {
                t: g.t,
                seg_class: seg_class.clone(),
                small_gemm_segments,
                qwen_segments,
                packet_segment_roles,
                kernarg,
                d_inst,
                h_inst,
                inst_range: prefill_patch_range(
                    rope.iter()
                        .chain(&flash)
                        .chain(&lmhead)
                        .chain(&merge)
                        .copied(),
                ),
                rope_sites: rope,
                flash_sites: flash,
                lmhead_sites: lmhead,
                merge_sites: merge,
                fp8_kv,
                batch_patched: false,
                d_ctr,
                ctr_bytes,
                _tables: vec![
                    d_stream,
                    d_sofs,
                    d_slen,
                    d_waits,
                    d_succs,
                    d_gq_stream,
                    d_gq_seg,
                ],
            });
        }
        if buckets.is_empty() {
            return Err(RuntimeError::Device(
                "blob has no prefill buckets (only the T=1 decode program)".into(),
            ));
        }
        // Only complete segmented chains avoid launching f_pf at the decode grid.
        if grid_pf != grid
            && !buckets.iter().all(|b| {
                uses_segmented_prefill(
                    seg_mode,
                    !b.qwen_segments.is_empty(),
                    b.seg_class.len(),
                    &b.packet_segment_roles,
                )
            })
        {
            return Err(RuntimeError::Device(format!(
                "prefill grid {grid_pf} ({occ}/SM × {} SMs) != decode grid {grid} (n_cu {}) — \
                 ordinary prefill launches must share the decode grid",
                be.sm_count(),
                blob.n_cu
            )));
        }
        buckets.sort_by_key(|b| b.t);
        Ok((f_pf, smem_pf, module, buckets, seg_pf))
    }

    /// Consume the whole prompt for slot `b` through the prefill bucket chain
    /// (chunked per the largest bucket), leaving slot `b`'s KV cache built to
    /// `prompt.len()` rows and `in.ids[0]` holding the first generated token —
    /// the exact postcondition of the decode-only consumption loop, so
    /// `step_slots()` continues from here unchanged. Port of
    /// `gemma4_sm120_chat.cu`'s `PLOW_PREFILL=1` path.
    ///
    /// The prefill programs address the KV cache slot-relative (the legacy
    /// single-ring headnorm/flash formulas), so for `b > 0` the `kv.*` entries
    /// of the tensor table are repointed at slot `b`'s ring for the duration
    /// of the chunk chain, then restored to the batch-major bases the decode
    /// program expects. Serialized per slot — the engine mutex already makes
    /// prefill and decode mutually exclusive.
    ///
    /// Requires `begin_slot()` to have validated the sequence budget first.
    pub fn prefill_slot(&mut self, b: usize, prompt: &[u32]) -> Result<u32> {
        loop {
            match self.prefill_chunk(b, prompt, usize::MAX)? {
                PrefillStep::Done(tok) => return Ok(tok),
                PrefillStep::Progress(_) => {}
            }
        }
    }

    /// Advance slot `b`'s prefill by ONE chunk (bucket capped at `cap` rows —
    /// the serve-layer interleave bound; `usize::MAX` = uncapped). The frontier
    /// lives in `pos[b]` (`begin_slot` resets it to 0), which also keeps the
    /// batched-decode unfed-row garbage KV write landing in exactly the row
    /// the next chunk overwrites — so decode ticks may interleave between
    /// chunks without corrupting the partially-built cache. On the final chunk
    /// the first generated token is read back (`PrefillStep::Done`), the exact
    /// postcondition of the whole-prompt [`Self::prefill_slot`].
    pub fn prefill_chunk(&mut self, b: usize, prompt: &[u32], cap: usize) -> Result<PrefillStep> {
        let cap = cap.min(self.pf_request_max_rows());
        let Some(f_pf) = self.f_pf else {
            return Err(RuntimeError::Rejected("prefill object not loaded".into()));
        };
        if self.prefill.is_empty() {
            return Err(RuntimeError::Rejected("no prefill buckets".into()));
        }
        // Batched mode owns the bucket instruction streams (t5/t6/merge are
        // patched to the batched contract) — the serialized chunk path must
        // never launch against them.
        if self.pf_batch.is_some() && self.packed_prefill.is_none() {
            return Err(RuntimeError::Rejected(
                "PLOW_PF_BATCH=1: serialized prefill_chunk disabled — use prefill_batched".into(),
            ));
        }
        if b >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {b} out of range (engine batch {})",
                self.batch
            )));
        }
        let n = prompt.len();
        if n == 0 {
            return Err(RuntimeError::Rejected("empty prompt".into()));
        }
        if n > self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "prompt {n} exceeds the compiled context {}",
                self.max_ctx
            )));
        }
        // VMM: first chunk of a fresh sequence consults the prefix cache —
        // a hit shares the whole-block prefix and moves the frontier there.
        if self.pos[b] == 0 && self.vmm_prefix_enabled() {
            self.vmm_attach(b, prompt)?;
        }
        let c0 = self.pos[b] as usize;
        debug_assert!(c0 < n, "prefill_chunk past the prompt end");

        if self.qwen_prefill.is_some()
            && !self
                .prefill
                .iter()
                .any(|p| p.t as usize <= (n - c0).min(cap))
        {
            let end = c0.saturating_add(cap.max(1)).min(n);
            let mut tokens = Vec::with_capacity(1);
            let token = self.consume_prompt(b, &prompt[c0..end], &mut tokens)?;
            return Ok(if end == n {
                PrefillStep::Done(token)
            } else {
                PrefillStep::Progress(end)
            });
        }

        // The chunk launches against slot b's prebuilt tensor table (the
        // kernarg selects it) — the shared table is never touched, so decode
        // and other slots' sequences cannot observe a shifted table.
        let real = self.run_one_prefill_chunk(f_pf, b, prompt, c0, cap)?;
        self.pos[b] = (c0 + real) as u32;

        if c0 + real < n {
            return Ok(PrefillStep::Progress(c0 + real));
        }
        // VMM: prefill finished — publish the prompt's whole sharing blocks
        // and the boundary's sliding-window snapshot into the prefix cache.
        // The consumed prompt also seeds the slot's row-token record so the
        // sequence's GENERATED blocks can publish at the next begin_slot.
        if self.vmm_prefix_enabled() {
            self.seq_tokens[b].clear();
            self.seq_tokens[b].extend_from_slice(prompt);
        }
        if let Some(v) = self.vmm.as_ref().filter(|v| v.kv.prefix_reuse()) {
            let p_a = (n.saturating_sub(1) as u32 / 32) * 32;
            if p_a > 0 {
                let snap_bytes = self.vmm_snap_bytes(p_a);
                if let Err(e) = v.kv.publish_at(b, prompt, p_a, snap_bytes, |dst| {
                    self.vmm_snap_copy(b, p_a, dst, true)
                }) {
                    tracing::warn!(error = %e, slot = b, "vmm: publish failed (serving continues)");
                }
            }
        }
        // in.ids[0] now holds the first generated token (argmax over the last
        // real prompt row) — identical to decode-only step n_prompt-1.
        let mut out = [0u8; 4];
        self.be.download(&self.devp[self.t_ids], 0, &mut out)?;
        Ok(PrefillStep::Done(i32::from_le_bytes(out) as u32))
    }

    /// Pick the prefill bucket for `rem` remaining rows under `cap`.
    ///
    /// Default policy minimizes `padded_rows + CHUNK_COST × launches` (see
    /// [`pf_chunk_cost_rows`]). A launch is NOT free — it re-streams every
    /// layer's weights — so minimizing padded rows alone is the wrong
    /// objective: it cascades a just-under-rung tail down the ladder, e.g.
    /// 8190 rows ran `[4096, 2048, 1024, 512, 128, 128, 128, 128]` (8 launches)
    /// where the 2-row-padded `[8192]` cover is one. Measured on sm_120 /
    /// gemma-4-12B, that single case cost **+25% TTFT**, and an 8390-row prompt
    /// finished 317 ms SOONER than an 8190-row one. Charging each launch its
    /// real cost removes the whole staircase (ms/tok across the 8.2k–10.4k
    /// boundary band went 0.133–0.175 → 0.129–0.139).
    ///
    /// A 4137-row prompt runs `[4096, 128]` = 4224 padded rows; a 4500-row pack
    /// still takes `[4096, …]` rather than the ~45%-padded single `[8192]`.
    /// `PLOW_PF_COVER=1` restores the covering pick (A/B control / exact parity
    /// with harness trajectories, which chunk the covering way);
    /// `PLOW_PF_CHUNK_COST=0` recovers the pure-minimum-padding objective.
    fn pick_prefill_bucket(&self, rem: usize, cap: usize) -> usize {
        let allowed = |t: usize| t <= cap;
        // Fall back to the smallest bucket when the cap is under every rung.
        let smallest = 0usize; // buckets are sorted by t
        if pf_cover_on() {
            // Old policy: smallest allowed bucket >= rem, else largest allowed.
            let mut pick = smallest;
            for (i, bkt) in self.prefill.iter().enumerate() {
                let t = bkt.t as usize;
                if i > 0 && !allowed(t) {
                    break;
                }
                pick = i;
                if t >= rem {
                    break;
                }
            }
            return pick;
        }
        // Cost-aware pick: minimize `padded_rows + CHUNK_COST × launches`.
        //
        // Minimizing padded rows ALONE cascades a just-under-rung tail into a
        // pile of tiny launches — 8190 rows ran [4096,2048,1024,512,128×4], 8
        // launches, measured 28% SLOWER than an 8390-row prompt's [8192,128,128]
        // (1434 ms vs 1117 ms: 200 MORE tokens finished 317 ms sooner). Charging
        // each launch `pf_chunk_cost_rows()` makes the 2-row-padded 8192 cover
        // win, which is what the hardware actually prefers.
        let n_allowed = self
            .prefill
            .iter()
            .take_while(|b| allowed(b.t as usize))
            .count();
        if n_allowed == 0 {
            return smallest;
        }
        // While the largest allowed rung still FILLS, it is optimal outright:
        // minimal padding and minimal launches at the same time.
        let top = n_allowed - 1;
        if rem >= self.prefill[top].t as usize {
            return top;
        }
        // Tail. Every rung is a multiple of the smallest, so quantizing the
        // state on it bounds the table at `top_rung / smallest_rung` entries
        // (64 for the shipped 128…8192 ladder).
        let unit = (self.prefill[smallest].t as usize).max(1);
        let chunk_cost = pf_chunk_cost_rows();
        let goal = rem.div_ceil(unit);
        // best[s] = (cost of consuming s units, first rung of that plan)
        let mut best = vec![(usize::MAX, smallest); goal + 1];
        best[0] = (0, smallest);
        for s in 1..=goal {
            let left = s * unit;
            for (i, bkt) in self.prefill.iter().take(n_allowed).enumerate() {
                let t = bkt.t as usize;
                // `t >= unit` ⇒ `next < s`, so the table fills in one pass.
                let next = left.saturating_sub(t).div_ceil(unit);
                let (prev, _) = best[next];
                if prev == usize::MAX {
                    continue;
                }
                let cost = prev + t + chunk_cost;
                if cost < best[s].0 {
                    best[s] = (cost, i);
                }
            }
        }
        best[goal].1
    }

    /// Slot `b`'s immutable prefill tensor table (built once at load).
    fn tens_slot_base(&self, b: usize) -> u64 {
        if b == 0 {
            self.d_tens.base
        } else {
            self.d_tens_slots[b - 1].base
        }
    }

    /// One prefill chunk at frontier `c0` (the harness's inner-loop body) —
    /// bucket pick, instruction patch, ids/pos/kvlen upload, one cooperative
    /// launch. KV lands wherever the tensor table currently points
    /// (`bind_kv_slot`). Returns the number of real rows consumed.
    fn run_one_prefill_chunk(
        &mut self,
        f_pf: KernelFn,
        b: usize,
        prompt: &[u32],
        c0: usize,
        cap: usize,
    ) -> Result<usize> {
        let n = prompt.len();
        let rem = n - c0;
        let sz = std::mem::size_of::<DevInst64>();

        let bi = if self.qwen_prefill.is_some() {
            self.prefill
                .iter()
                .rposition(|p| p.t as usize <= rem.min(cap))
                .ok_or_else(|| {
                    RuntimeError::Rejected("Qwen prefill requires an exact chunk".into())
                })?
        } else {
            self.pick_prefill_bucket(rem, cap)
        };
        let tc = self.prefill[bi].t as usize;
        let real = rem.min(tc);

        // VMM: the bucket writes all tc rows (pad rows write garbage past
        // `real`) — map the chunk's full row span before launching.
        if let Some(v) = &mut self.vmm {
            if !v.kv.prefix_reuse() && c0 + tc > self.max_ctx {
                return Err(RuntimeError::Rejected(
                    "live KV prefill padding exceeds the reserved context".into(),
                ));
            }
            if let Some(rings) = &mut v.rings {
                rings.ensure_slot(b)?;
            }
            v.kv.ensure_rows(b, ((c0 + tc) as u32).min(self.max_ctx as u32))?;
        }

        if self.packed_prefill.is_some() && self.prefill[bi].batch_patched {
            let pack = self.packed_prefill.as_ref().unwrap();
            let bucket = &mut self.prefill[bi];
            if let Some(terminal) = &self.packed_terminal {
                terminal.patch_discarded_tail(bucket, false);
            }
            for &pc in &bucket.rope_sites {
                pack.bind_request(&mut bucket.h_inst[pc], false);
            }
            for &pc in &bucket.flash_sites {
                pack.bind_request(&mut bucket.h_inst[pc], false);
            }
            for &pc in &bucket.merge_sites {
                pack.bind_request(&mut bucket.h_inst[pc], false);
            }
            bucket.batch_patched = false;
        }
        // Patch this bucket's instruction stream for the chunk, then enqueue
        // the covering window as an async H2D on the engine stream.
        {
            let b = &mut self.prefill[bi];
            for &ix in &b.rope_sites {
                b.h_inst[ix].i[3] = c0 as u32;
            }
            for &ix in &b.flash_sites {
                b.h_inst[ix].i[1] = (c0 + real) as u32;
                b.h_inst[ix].i[4] = c0 as u32;
            }
            for &ix in &b.lmhead_sites {
                b.h_inst[ix].i[4] = (real - 1) as u32;
            }
            if !b.inst_range.is_empty() {
                let bytes = pod_bytes(&b.h_inst[b.inst_range.clone()]);
                // SAFETY: h_inst lives on self past the stream_synchronize below.
                unsafe {
                    self.be.memcpy_htod_async(
                        b.d_inst.base + (b.inst_range.start * sz) as u64,
                        bytes,
                        &self.stream,
                    )?;
                }
            }
        }

        // ids (real tokens + zero pad) and absolute positions for the chunk.
        // Reuse pre-allocated buffers (sized to max prefill bucket t).
        self.pf_ids.resize(tc, 0);
        self.pf_pos.resize(tc, 0);
        for i in 0..tc {
            self.pf_ids[i] = if i < real { prompt[c0 + i] as i32 } else { 0 };
            self.pf_pos[i] = (c0 + i) as i32;
        }
        // SAFETY: pf_ids, pf_pos live on self past the stream_synchronize;
        // devp[t_ids/t_pos/t_kvlen] are live device ranges sized to hold the
        // uploaded data (tc ≤ max prefill bucket t ≤ tensor capacity).
        let kvlen_bytes = ((c0 + real) as i32).to_le_bytes();
        unsafe {
            self.be.memcpy_htod_async(
                self.devp[self.t_ids].base,
                bytemuck::cast_slice(&self.pf_ids[..tc]),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_pos].base,
                bytemuck::cast_slice(&self.pf_pos[..tc]),
                &self.stream,
            )?;
            self.be
                .memcpy_htod_async(self.devp[self.t_kvlen].base, &kvlen_bytes, &self.stream)?;
        }

        let (ctr_base, ctr_bytes, mut arg) = {
            let bkt = &self.prefill[bi];
            (bkt.d_ctr.base, bkt.ctr_bytes, bkt.kernarg)
        };
        // Slot selection is a kernarg field: the launch reads slot b's
        // prebuilt tensor table (kv.* shifted to its ring); nothing uploads.
        arg.tensors = self.tens_slot_base(b);
        // One async fill re-arms the bucket's counters AND its tail GQ cursor.
        self.be
            .memset_d8_async(ctr_base, 0, ctr_bytes, &self.stream)?;

        // All uploads/memsets are enqueued on the engine stream — the launch
        // follows them in stream order (no context sync needed).
        self.launch_prefill_chain(bi, arg, f_pf, b, c0, n, tc)?;
        Ok(real)
    }

    fn launch_prefill_chain(
        &mut self,
        bi: usize,
        mut arg: DevProgram,
        f_pf: KernelFn,
        b: usize,
        c0: usize,
        n: usize,
        tc: usize,
    ) -> Result<()> {
        // SEGMENTED MODE (SegPf): one launch per wave-class segment, in blob order,
        // alternating the fat (flash) and lean occ-2 (GEMM) objects. Sequential stream
        // launches make cross-segment gates trivially satisfied (dependencies only point
        // backward); counters were re-armed ONCE above and persist across the segments.
        // cuLaunchCooperativeKernel snapshots the param buffer at enqueue, so mutating
        // `arg.cur_seg` between launches is sound.
        let seg_class = self.prefill[bi].seg_class.clone();
        // len 1 = a force_uniseg small bucket: ONE launch on the full fat _pf object beats
        // ~480 segment launches (T18) — take the single-launch path below.
        if !self.prefill[bi].qwen_segments.is_empty() {
            for (seg, external) in self.prefill[bi].qwen_segments.iter().enumerate() {
                if let Some(inst) = external {
                    let native = self
                        .qwen_prefill
                        .as_mut()
                        .expect("validated Qwen prefill adapter");
                    let t = &inst.t;
                    let state = &self.devp[t[6] as usize];
                    let slot_state = qwen_state_slot(state, b, self.batch)?;
                    let tensors = [
                        &self.devp[t[1] as usize],
                        &self.devp[t[2] as usize],
                        &self.devp[t[3] as usize],
                        &self.devp[t[0] as usize],
                        &self.devp[t[4] as usize],
                        &self.devp[t[5] as usize],
                        &self.devp[t[7] as usize],
                        &slot_state,
                    ];
                    // Packet loading validates tensor extents and fixed kernel geometry.
                    unsafe {
                        native.launch(tensors, inst.i[0] as usize, &self.stream)?;
                    }
                }
                let role =
                    packet_role_index(&self.prefill[bi].packet_segment_roles, seg).map(|index| {
                        self.packet_roles[index]
                            .as_ref()
                            .expect("validated packet role object")
                    });
                segment_window(&mut arg, &self.prefill[bi].kernarg, seg, role.is_some());
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(
                    role.map_or(f_pf, |r| r.function),
                    self.grid,
                    BLOCK,
                    role.map_or(self.smem_pf, |r| r.smem),
                    &mut params,
                    Some(&self.stream),
                )?;
            }
        } else if uses_segmented_prefill(
            self.seg_pf.is_some(),
            false,
            seg_class.len(),
            &self.prefill[bi].packet_segment_roles,
        ) {
            let sp = self.seg_pf.as_ref().expect("checked");
            // T35 (PLOW_PF_SEG_GRAPH=1): submit the whole per-chunk segment chain as ONE
            // CUDA graph. Per-node kernargs (cur_seg baked per node) are copied at build;
            // everything else in `arg` is constant per (bucket, slot tensor table), which
            // keys the cache. Falls through to the loop under the timing/fatonly diags.
            let rt = crate::config::RuntimeConfig::get();
            let seg_time_probe = rt.nv.pf_seg_time;
            let fat_only_probe = rt.nv.pf_seg_fatonly;
            if !seg_time_probe && !fat_only_probe && rt.nv.pf_seg_graph {
                let key = (bi, arg.tensors as u64);
                if !self.seg_graphs.contains_key(&key) {
                    let mut blobs: Vec<DevProgram> = (0..seg_class.len())
                        .map(|i| {
                            let mut a2 = arg;
                            a2.cur_seg = i as u32;
                            a2
                        })
                        .collect();
                    let nodes: Vec<(KernelFn, u32, u32, u32)> = seg_class
                        .iter()
                        .enumerate()
                        .map(|(seg, &cls)| {
                            if let Some(index) =
                                packet_role_index(&self.prefill[bi].packet_segment_roles, seg)
                            {
                                let role = self.packet_roles[index]
                                    .as_ref()
                                    .expect("validated packet role object");
                                return Ok((role.function, self.grid, role.block, role.smem));
                            }
                            Ok(match cls {
                                4 => (sp.f_flash, sp.grid_flash, BLOCK, sp.smem_flash),
                                2 => {
                                    let (f3, s3, g3) = sp.fa512.ok_or_else(|| {
                                        RuntimeError::Device(
                                            "class-2 segment without the pffa object".into(),
                                        )
                                    })?;
                                    (f3, g3, BLOCK, s3)
                                }
                                _ => sp.gemm(
                                    self.prefill[bi].small_gemm_segments.get(seg) == Some(&true),
                                ),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut ptrs: Vec<*mut std::ffi::c_void> = blobs
                        .iter_mut()
                        .map(|bb| bb as *mut DevProgram as *mut std::ffi::c_void)
                        .collect();
                    let g = self.be.graph_build_chain(&nodes, &mut ptrs)?;
                    self.seg_graphs.insert(key, g);
                    tracing::info!(nodes = seg_class.len(), bucket = bi, "seg graph built");
                }
                self.be
                    .graph_launch(self.seg_graphs.get(&key).expect("just built"), &self.stream)?;
                if let Err(e) = self.be.stream_synchronize(&self.stream) {
                    tracing::warn!(
                        error = %e,
                        error_code = ?e.device_code(),
                        fatal = e.is_fatal(),
                        slot = b,
                        bucket = bi,
                        bucket_t = tc,
                        chunk_start = c0,
                        prompt_len = n,
                        "prefill chunk (seg graph): stream sync failed"
                    );
                    return Err(e);
                }
                if rt.nv.pf_trace_log {
                    if let Ok(Some(sdump)) = self.trace_summary_pf() {
                        tracing::info!("prefill {sdump}");
                    }
                }
                return Ok(());
            }
            // PLOW_PF_SEG_TIME=1: per-CLASS wall attribution via one event pair per segment
            // (diagnostic; events cost ~us each — never on by default).
            let seg_time = seg_time_probe;
            // Both diagnostics are resolved ONCE per chunk, not per segment: a
            // `std::env::var` inside the ~480-iteration launch loop takes the
            // process-global env lock and allocates a String every time.
            // PLOW_PF_SEG_FATONLY=1: every segment on the fat object, isolating
            // the launch-serialization cost from the occ-2 effect.
            let fat_only = fat_only_probe;
            // PLOW_PF_SEG_NONCOOP=1 (T14b): plain cuLaunchKernel per segment. The
            // cooperative wrapper's only guarantee — co-residency — already holds: the
            // grid equals the module's queried resident capacity and the stream is
            // otherwise idle, so all blocks schedule together. Diagnostic-grade knob to
            // price the cooperative-launch overhead (241 fat launches/chunk).
            let noncoop = rt.nv.pf_seg_noncoop;
            let mut evs: Vec<(u8, CudaEvent, CudaEvent)> = Vec::new();
            for (seg, &cls) in seg_class.iter().enumerate() {
                arg.cur_seg = seg as u32;
                let role =
                    packet_role_index(&self.prefill[bi].packet_segment_roles, seg).map(|index| {
                        self.packet_roles[index]
                            .as_ref()
                            .expect("validated packet role object")
                    });
                let (f, gr, sm, blk) = if let Some(role) = role {
                    (role.function, self.grid, role.smem, role.block)
                } else if fat_only || cls == 4 {
                    (sp.f_flash, sp.grid_flash, sp.smem_flash, BLOCK)
                } else if cls == 2 {
                    // T12: hd512 flash segments on the dedicated *_pffa object.
                    let (f3, s3, g3) = sp.fa512.ok_or_else(|| {
                        RuntimeError::Device(
                            "class-2 (hd512 flash) segment but no interp_sm90a_pffa.cubin in \
                             PLOW_PF_SEG_DIR — unset PLOW_PF_SEG_FA512 or add the object"
                                .into(),
                        )
                    })?;
                    (f3, g3, s3, BLOCK)
                } else {
                    let (f, gr, blk, sm) =
                        sp.gemm(self.prefill[bi].small_gemm_segments.get(seg) == Some(&true));
                    (f, gr, sm, blk)
                };
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                let go = |params: &mut [*mut std::ffi::c_void]| -> Result<()> {
                    if noncoop {
                        self.be
                            .launch_kernel(f, gr, blk, sm, params, Some(&self.stream))
                    } else {
                        self.be
                            .launch_cooperative(f, gr, blk, sm, params, Some(&self.stream))
                    }
                };
                if seg_time {
                    tracing::debug!(
                        segment = seg,
                        class = cls,
                        grid = gr,
                        block = blk,
                        smem = sm,
                        packet_role = role.is_some(),
                        "prefill segment launch"
                    );
                    let e0 = self.be.event_create(true)?;
                    let e1 = self.be.event_create(true)?;
                    self.be.event_record(&e0, &self.stream)?;
                    go(&mut params)?;
                    self.be.event_record(&e1, &self.stream)?;
                    evs.push((cls, e0, e1));
                } else {
                    go(&mut params)?;
                }
            }
            if seg_time {
                self.be.stream_synchronize(&self.stream)?;
                let mut by_class = [0f64; 3]; // [gemm(8), flash-fat(4), fa512(2)]
                let mut n_by = [0u32; 3];
                for (cls, e0, e1) in &evs {
                    let ix = match cls {
                        8 => 0,
                        4 => 1,
                        _ => 2,
                    };
                    by_class[ix] += self.be.event_elapsed_ms(e0, e1)? as f64;
                    n_by[ix] += 1;
                }
                tracing::info!(
                    gemm_ms = format!("{:.1}", by_class[0]).as_str(),
                    fat_ms = format!("{:.1}", by_class[1]).as_str(),
                    fa_ms = format!("{:.1}", by_class[2]).as_str(),
                    n_gemm = n_by[0],
                    n_fat = n_by[1],
                    n_fa = n_by[2],
                    "seg-class wall time (chunk)"
                );
                // Top-10 slowest segments, to attribute inside a class.
                let mut per: Vec<(usize, u8, f32)> = Vec::with_capacity(evs.len());
                for (i, (cls, e0, e1)) in evs.iter().enumerate() {
                    per.push((i, *cls, self.be.event_elapsed_ms(e0, e1)?));
                }
                per.sort_by(|a, b| b.2.total_cmp(&a.2));
                let top: Vec<String> = per
                    .iter()
                    .take(10)
                    .map(|(i, c, ms)| format!("seg{i}(c{c})={ms:.2}ms"))
                    .collect();
                tracing::info!(top = top.join(" ").as_str(), "slowest segments");
            }
        } else {
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            self.be.launch_cooperative(
                f_pf,
                self.grid,
                BLOCK,
                self.smem_pf,
                &mut params,
                Some(&self.stream),
            )?;
        }
        if let Err(e) = self.be.stream_synchronize(&self.stream) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                slot = b,
                bucket = bi,
                bucket_t = tc,
                chunk_start = c0,
                prompt_len = n,
                grid = self.grid,
                block = BLOCK,
                smem = self.smem_pf,
                "prefill chunk: stream sync failed"
            );
            return Err(e);
        }

        // Per-op attribution hook: with a `-DPLOW_NV_TRACE=1` prefill cubin and
        // `--pf-trace-log` / PLOW_PF_TRACE_LOG=1, dump block 0's gate/body/signal
        // per opcode after each chunk.
        if crate::config::RuntimeConfig::get().nv.pf_trace_log {
            if let Ok(Some(s)) = self.trace_summary_pf() {
                tracing::info!("prefill {s}");
            }
        }

        Ok(())
    }

    /// One-time PX-1 batched-mode patch of bucket `bi`'s instruction stream:
    /// KV-write `HeadNormRope` t6 = the per-row slot map; `FlashPrefill` t6 =
    /// the request table and (non-fused buckets) t5 = the fused `n.at` output;
    /// `FlashMerge` neutered (`i[0]=0` — the flash epilogue already normalized
    /// and wrote the bf16); lm_head `a_row0 = 0` (row 0 always holds a real
    /// row; the argmax result is unused — the first token comes from a batched
    /// decode step of the last prompt token). One covering-window upload.
    fn ensure_batch_patch(&mut self, bi: usize) -> Result<()> {
        if self.prefill[bi].batch_patched {
            return Ok(());
        }
        if let Some(pack) = &self.packed_prefill {
            let b = &mut self.prefill[bi];
            if let Some(terminal) = &self.packed_terminal {
                terminal.patch_discarded_tail(b, true);
            }
            for &pc in &b.rope_sites {
                pack.bind_request(&mut b.h_inst[pc], true);
            }
            for &pc in &b.flash_sites {
                pack.bind_request(&mut b.h_inst[pc], true);
            }
            for &pc in &b.merge_sites {
                pack.bind_request(&mut b.h_inst[pc], true);
            }
            for &pc in &b.lmhead_sites {
                b.h_inst[pc].i[4] = 0;
            }
            if !b.inst_range.is_empty() {
                let offset = b.inst_range.start * std::mem::size_of::<DevInst64>();
                self.be.memcpy_htod(
                    b.d_inst.base + offset as u64,
                    pod_bytes(&b.h_inst[b.inst_range.clone()]),
                )?;
                self.be.synchronize()?;
            }
            b.batch_patched = true;
            return Ok(());
        }
        let (h_slot, h_req, at_sites) = {
            let pb = self.pf_batch.as_ref().expect("pf_batch checked by caller");
            (pb.h_slot, pb.h_req, pb.at_sites.clone())
        };
        let sz = std::mem::size_of::<DevInst64>();
        let b = &mut self.prefill[bi];
        if b.flash_sites.len() != at_sites.len() {
            return Err(RuntimeError::Device(format!(
                "pf-batch: bucket T={} has {} flash sites, fused bucket has {}",
                b.t,
                b.flash_sites.len(),
                at_sites.len()
            )));
        }
        for (k, &ix) in b.flash_sites.iter().enumerate() {
            let (t5, hd) = at_sites[k];
            if b.h_inst[ix].i[6] != hd {
                return Err(RuntimeError::Device(format!(
                    "pf-batch: bucket T={} flash site {k} hd {} != fused bucket's {hd}",
                    b.t, b.h_inst[ix].i[6]
                )));
            }
            if b.h_inst[ix].t[5] == TENSOR_NONE16 {
                b.h_inst[ix].t[5] = t5 as u16;
            }
            b.h_inst[ix].t[6] = h_req as u16;
        }
        for &ix in &b.rope_sites {
            b.h_inst[ix].t[6] = h_slot as u16;
        }
        for &ix in &b.merge_sites {
            b.h_inst[ix].i[0] = 0;
        }
        for &ix in &b.lmhead_sites {
            b.h_inst[ix].i[4] = 0;
        }
        if !b.inst_range.is_empty() {
            let bytes = pod_bytes(&b.h_inst[b.inst_range.clone()]);
            self.be
                .memcpy_htod(b.d_inst.base + (b.inst_range.start * sz) as u64, bytes)?;
        }
        b.batch_patched = true;
        Ok(())
    }

    /// PX-1: run ONE batched prefill launch over the packed chunks of `reqs`
    /// (in order — request r's rows occupy the tile rows after request r-1's).
    /// The GEMMs see one M = Σ len matrix (weights read once, shared across
    /// requests); attention runs per-request-serial against each request's own
    /// seq-slot KV (block-diagonal by construction — see d_flash_prefill_mux);
    /// each row's KV write lands in its own slot's batch-major ring at its own
    /// absolute position. Advances `pos[slot]` per request. The mux packs to
    /// `n-1` prompt rows and takes the first generated token from a batched
    /// decode step of the last prompt token (`step_slots`), so no lm_head
    /// readback happens here.
    pub fn prefill_batched(&mut self, reqs: &[PfBatchReq]) -> Result<()> {
        let chunks = reqs
            .iter()
            .map(|r| {
                let end = r.c0.checked_add(r.len).ok_or_else(|| {
                    RuntimeError::Rejected("packed chunk end overflow".into())
                })?;
                let tokens = r.prompt.get(r.c0..end).ok_or_else(|| {
                    RuntimeError::Rejected("packed chunk outside prompt".into())
                })?;
                Ok(PackedTokenReq {
                    slot: r.slot,
                    tokens,
                    c0: r.c0,
                    prompt_len: r.prompt.len(),
                })
            })
            .collect::<Result<smallvec::SmallVec<[_; 16]>>>()?;
        self.packed_token_body(&chunks)?;
        for r in reqs {
            self.pos[r.slot] = (r.c0 + r.len) as u32;
            if self.vmm_prefix_enabled() && r.c0 + r.len + 1 >= r.prompt.len() {
                let end = r.c0 + r.len;
                self.seq_tokens[r.slot].clear();
                self.seq_tokens[r.slot].extend_from_slice(&r.prompt[..end]);
                self.vmm_publish(r.slot, r.prompt.len().saturating_sub(1) as u32);
            }
        }
        Ok(())
    }

    fn packed_token_body(&mut self, reqs: &[PackedTokenReq<'_>]) -> Result<()> {
        let Some(f_pf) = self.f_pf else {
            return Err(RuntimeError::Rejected("prefill object not loaded".into()));
        };
        if self.pf_batch.is_none() {
            return Err(RuntimeError::Rejected("PLOW_PF_BATCH not enabled".into()));
        }
        if reqs.is_empty() {
            return Ok(());
        }
        let request_plan = if self.packed_prefill.is_some() {
            let total = reqs
                .iter()
                .try_fold(0usize, |n, r| n.checked_add(r.tokens.len()))
                .ok_or_else(|| RuntimeError::Rejected("packed row overflow".into()))?;
            let bucket = self
                .prefill
                .iter()
                .find(|b| b.t as usize >= total)
                .ok_or_else(|| RuntimeError::Rejected("packed bucket overflow".into()))?;
            let requests: Vec<_> = reqs
                .iter()
                .map(|r| plow_asset::packed_prefill::Request {
                    slot: r.slot,
                    start: r.c0,
                    len: r.tokens.len(),
                    prompt: r.prompt_len,
                })
                .collect();
            Some(
                plow_asset::packed_prefill::plan_with_limit(
                    &requests,
                    &self.pos,
                    bucket.t as usize,
                    self.max_ctx,
                    self.packed_prefill.as_ref().and_then(|p| p.max_request_rows),
                )
                .map_err(RuntimeError::Rejected)?,
            )
        } else {
            None
        };
        let mut total = 0usize;
        for r in reqs {
            if r.slot >= self.batch {
                return Err(RuntimeError::Rejected(format!(
                    "slot {} out of range (engine batch {})",
                    r.slot, self.batch
                )));
            }
            let end = r.c0.checked_add(r.tokens.len()).ok_or_else(|| {
                RuntimeError::Rejected("packed chunk end overflow".into())
            })?;
            if r.tokens.is_empty() || end > r.prompt_len {
                return Err(RuntimeError::Rejected(format!(
                    "pf-batch: bad chunk c0={} len={} prompt={}",
                    r.c0,
                    r.tokens.len(),
                    r.prompt_len
                )));
            }
            if r.c0 != self.pos[r.slot] as usize {
                return Err(RuntimeError::Rejected(format!(
                    "pf-batch: slot {} frontier {} != chunk c0 {}",
                    r.slot, self.pos[r.slot], r.c0
                )));
            }
            if end > self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "pf-batch: chunk end {} exceeds compiled context {}",
                    end,
                    self.max_ctx
                )));
            }
            total = total.checked_add(r.tokens.len()).ok_or_else(|| {
                RuntimeError::Rejected("packed row overflow".into())
            })?;
        }
        // Covering bucket for the pack: smallest T >= Σ len (minimal padding).
        let bi = self
            .prefill
            .iter()
            .position(|b| b.t as usize >= total)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "pf-batch: pack of {total} rows exceeds the largest bucket"
                ))
            })?;
        // The batched path single-launches the plain `_pf` object, whose gq window
        // is segment 0 only. A wave-class-segmented bucket (SegPf loaded) would run
        // its first segment and silently skip every op after it — stale KV, garbage
        // logits, launch reports success. Refuse instead: the two features are
        // mutually exclusive until the batched path learns the segment chain.
        if self.packed_prefill.is_none() && !self.prefill[bi].seg_class.is_empty() {
            return Err(RuntimeError::Rejected(
                "PLOW_PF_BATCH with a wave-class-segmented prefill program \
                 (PLOW_PF_SEG_DIR): the batched path launches segment 0 only — \
                 unset one of the two"
                    .into(),
            ));
        }
        if let (Some(plan), Some(v)) = (&request_plan, &mut self.vmm) {
            for &(slot, end) in &plan.mapped_ends {
                if let Some(rings) = &mut v.rings {
                    rings.ensure_slot(slot)?;
                }
                v.kv.ensure_rows(slot, end)?;
            }
        }
        self.ensure_batch_patch(bi)?;
        let tc = self.prefill[bi].t as usize;

        // Stage ids/pos/slot rows + the request table, then upload. `pf_batch`
        // is taken out for the duration so `self` stays borrowable.
        let mut pb = self.pf_batch.take().expect("checked Some");
        // Declared outside the staging closure so it outlives the closure and
        // stays valid until the stream_synchronize below — the closure enqueues
        // an async H2D of these bytes (memcpy_htod_async src-lifetime contract).
        let kvlen_bytes = reqs
            .iter()
            .map(|r| (r.c0 + r.tokens.len()) as i32)
            .max()
            .unwrap_or(0)
            .to_le_bytes();
        let staged = (|| -> Result<()> {
            self.pf_ids.resize(tc, 0);
            self.pf_pos.resize(tc, 0);
            pb.slot_buf.resize(tc, 0);
            pb.req_buf.clear();
            pb.req_buf.push(reqs.len() as i32);
            let mut cur = 0usize;
            for r in reqs {
                for (k, &token) in r.tokens.iter().enumerate() {
                    self.pf_ids[cur + k] = token as i32;
                    self.pf_pos[cur + k] = (r.c0 + k) as i32;
                    pb.slot_buf[cur + k] = r.slot as i32;
                }
                pb.req_buf.extend_from_slice(&[
                    cur as i32,
                    r.tokens.len() as i32,
                    r.slot as i32,
                    (r.c0 + r.tokens.len()) as i32,
                ]);
                cur += r.tokens.len();
            }
            // Trailing pad rows continue the LAST request's positions (legacy
            // pad semantics: garbage KV past a frontier lands in rows that
            // slot's own next writes overwrite before they become readable).
            let last = reqs.last().expect("non-empty");
            let (mut p, s) = (last.c0 + last.tokens.len(), last.slot as i32);
            for k in cur..tc {
                self.pf_ids[k] = 0;
                self.pf_pos[k] = p.min(self.max_ctx - 1) as i32;
                pb.slot_buf[k] = s;
                p += 1;
            }
            if let Some(plan) = &request_plan {
                self.pf_pos.copy_from_slice(&plan.positions);
                pb.slot_buf.copy_from_slice(&plan.slots);
                pb.req_buf.clone_from(&plan.table);
            }
            // SAFETY: pf_ids, pf_pos live on self past the stream_synchronize;
            // pb.slot_buf, pb.req_buf are put back into self.pf_batch (below)
            // before the sync; kvlen_bytes is declared above the closure so it
            // outlives the sync. All device destinations are live ranges.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[self.t_ids].base,
                    bytemuck::cast_slice(&self.pf_ids[..tc]),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    self.devp[self.t_pos].base,
                    bytemuck::cast_slice(&self.pf_pos[..tc]),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    pb.d_slot.base,
                    bytemuck::cast_slice(&pb.slot_buf[..tc]),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    pb.d_req.base,
                    bytemuck::cast_slice(&pb.req_buf),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    self.devp[self.t_kvlen].base,
                    &kvlen_bytes,
                    &self.stream,
                )?;
            }
            Ok(())
        })();
        self.pf_batch = Some(pb);
        if let Err(error) = staged {
            let _ = self.be.stream_synchronize(&self.stream);
            return Err(error);
        }

        let (ctr_base, ctr_bytes, mut arg) = {
            let b = &self.prefill[bi];
            (b.d_ctr.base, b.ctr_bytes, b.kernarg)
        };
        // One async fill re-arms the bucket's counters AND its tail GQ cursor.
        let launched = (|| -> Result<()> {
            self.be
                .memset_d8_async(ctr_base, 0, ctr_bytes, &self.stream)?;
            if self.packed_prefill.is_some() {
                self.launch_prefill_chain(bi, arg, f_pf, reqs[0].slot, reqs[0].c0, total, tc)
            } else {
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(
                    f_pf,
                    self.grid,
                    BLOCK,
                    self.smem_pf,
                    &mut params,
                    Some(&self.stream),
                )
            }
        })();
        if let Err(error) = launched {
            let _ = self.be.stream_synchronize(&self.stream);
            return Err(error);
        }
        if let Err(e) = self.be.stream_synchronize(&self.stream) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                requests = reqs.len(),
                bucket = bi,
                rows = total,
                grid = self.grid,
                block = BLOCK,
                smem = self.smem_pf,
                "batched prefill: stream sync failed"
            );
            return Err(e);
        }

        tracing::debug!(
            requests = reqs.len(),
            rows = total,
            bucket_t = tc,
            slots = ?reqs.iter().map(|r| r.slot).collect::<Vec<_>>(),
            "gpu: batched prefill launch"
        );
        if pf_packlog_on() {
            // One line per launch → parsed into R-per-launch histograms by the
            // RTX-12 packing bench. chunks = per-request packed rows (this tick).
            use std::fmt::Write as _;
            let mut chunks = String::new();
            for (i, r) in reqs.iter().enumerate() {
                let _ = write!(chunks, "{}{}", if i == 0 { "" } else { "," }, r.tokens.len());
            }
            eprintln!(
                "PACKLOG R={} rows={} bucket={} chunks=[{}]",
                reqs.len(),
                total,
                tc,
                chunks
            );
        }
        Ok(())
    }

    /// Download row `b` of the `[B][vocab]` logits (bf16 → f32) into `out`.
    /// Used when the request wants stochastic sampling; the greedy path
    /// consumes the device argmax without ever moving the row. Prefill writes
    /// its logits to row 0 regardless of the slot (lm_head M == 1).
    pub fn logits_row(&mut self, b: usize, out: &mut Vec<f32>) -> Result<()> {
        if b >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {b} out of range (engine batch {})",
                self.batch
            )));
        }
        self.logits_raw.resize(self.vocab * 2, 0);
        self.be.download(
            &self.devp[self.t_logits],
            (b * self.vocab * 2) as u64,
            &mut self.logits_raw,
        )?;
        out.clear();
        out.resize(self.vocab, 0.0);
        bf16_to_f32_slice(&self.logits_raw, out);
        Ok(())
    }

    /// Handle (blob tensor index, = `devp` index) of the tensor named `name`,
    /// or `None`. Block-harness accessor; not a hot path.
    pub fn handle_of(&self, name: &str) -> Option<usize> {
        self.tensor_names.iter().position(|n| n == name)
    }

    /// Read a named tensor as raw bytes for diagnostics.
    pub fn read_tensor(&self, name: &str, dst: &mut [u8]) -> Result<()> {
        let i = self.handle_of(name).ok_or_else(|| {
            RuntimeError::Rejected(format!("no tensor named {name:?} in the blob"))
        })?;
        self.be.download(&self.devp[i], 0, dst)
    }

    /// Read a byte range from a named tensor for block-harness diagnostics.
    pub fn read_tensor_range(&self, name: &str, offset: u64, dst: &mut [u8]) -> Result<()> {
        let i = self.handle_of(name).ok_or_else(|| {
            RuntimeError::Rejected(format!("no tensor named {name:?} in the blob"))
        })?;
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| RuntimeError::Rejected("tensor read range overflow".into()))?;
        if end > self.devp[i].len {
            return Err(RuntimeError::Rejected(format!(
                "read_tensor_range {name}: range {offset}..{end} exceeds {} bytes",
                self.devp[i].len
            )));
        }
        self.be.download(&self.devp[i], offset, dst)
    }

    /// Byte size of a named tensor's device allocation.
    pub fn tensor_bytes(&self, name: &str) -> Option<u64> {
        self.handle_of(name).map(|i| self.devp[i].len)
    }

    /// Download an activation tensor by name as f32 (bf16 → f32, the same
    /// widening [`Self::logits_row`] does). Length = the tensor's byte size / 2
    /// (bf16). Used by the block harness to read `act.x` out. Not a hot path.
    pub fn download_activation(&self, name: &str) -> Result<Vec<f32>> {
        let i = self.handle_of(name).ok_or_else(|| {
            RuntimeError::Rejected(format!("no tensor named {name:?} in the blob"))
        })?;
        let bytes = self.devp[i].len as usize;
        let mut raw = vec![0u8; bytes];
        self.be.download(&self.devp[i], 0, &mut raw)?;
        let mut out = vec![0.0f32; bytes / 2];
        bf16_to_f32_slice(&raw, &mut out);
        Ok(out)
    }

    /// Upload an f32 slice into a bf16 activation tensor by name (f32 → bf16 by
    /// truncating the low 16 mantissa bits, `(bits >> 16) as u16`). Used by the
    /// block harness to feed `act.x` in. Not a hot path.
    pub fn upload_activation(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let i = self.handle_of(name).ok_or_else(|| {
            RuntimeError::Rejected(format!("no tensor named {name:?} in the blob"))
        })?;
        let cap = self.devp[i].len as usize / 2;
        if data.len() > cap {
            return Err(RuntimeError::Rejected(format!(
                "upload_activation {name}: {} f32 > tensor capacity {cap} bf16",
                data.len()
            )));
        }
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            let h = (v.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        self.be.upload(&self.devp[i], 0, &bytes)?;
        // Pageable H2D can return before DMA completes; kernels use a nonblocking stream.
        self.be.synchronize()?;
        Ok(())
    }

    /// Reset the PLOW_NV_TRACE packet counter so [`Self::trace_summary`]
    /// reports only launches after this call (drop prefill/warmup). No-op on
    /// a normal cubin.
    pub fn trace_reset(&self) -> Result<()> {
        self.be.module_global_zero(&self.module, "g_tr_n", 4)?;
        Ok(())
    }

    /// [`Self::trace_summary`] against the PREFILL module (`-DPLOW_NV_TRACE=1`
    /// on the `_pf` build): block-0 per-packet gate/body/signal by opcode.
    /// `Ok(None)` when there is no prefill module or it carries no trace.
    pub fn trace_summary_pf(&self) -> Result<Option<String>> {
        // Segmented mode: the bundle's _pf module never launches — the trace lives in the
        // SegPf pair's FAT module (the lean ws role loop is not instrumented).
        if let Some(sp) = &self.seg_pf {
            let mut s = self.trace_summary_of(&sp._m_flash)?.unwrap_or_default();
            // The FA object runs the same instrumented loop — append its block-0 profile.
            if let Some(m3) = &sp._m_fa512 {
                if let Some(fa) = self.trace_summary_of(m3)? {
                    s.push_str("\n  [FA object] ");
                    s.push_str(&fa);
                }
            }
            return Ok(Some(s));
        }
        let Some(m) = &self.module_pf else {
            return Ok(None);
        };
        self.trace_summary_of(m)
    }

    /// Stage-7 profiling (plan §Instrumentation): if the decode cubin was
    /// built `-DPLOW_NV_TRACE=1`, read back block 0's per-packet gate/body/
    /// signal cycle attribution (device globals `g_tr_*`) and aggregate it by
    /// opcode. `Ok(None)` on a normal cubin (no trace globals). The trace
    /// accumulates across launches, so this reports the whole run up to the
    /// 4096-entry cap — read the SHAPE (gate vs body vs signal per op), not
    /// the absolute total (clock64 in the recording thread over-reports). See
    /// the trace header in interp_sm120.cu.
    pub fn trace_summary(&self) -> Result<Option<String>> {
        self.trace_summary_of(&self.module)
    }

    fn trace_summary_of(&self, module: &Module) -> Result<Option<String>> {
        let mut raw = Vec::new();
        if !self.be.module_global_bytes(module, "g_tr_n", 4, &mut raw)? {
            return Ok(None);
        }
        let n = u32::from_le_bytes(raw[..4].try_into().expect("4B")) as usize;
        if n == 0 {
            return Ok(Some("trace: no packets recorded".into()));
        }
        let cap = n.min(4096);
        let read_u32 = |name: &str, out: &mut Vec<u32>| -> Result<()> {
            let mut b = Vec::new();
            self.be.module_global_bytes(module, name, cap * 4, &mut b)?;
            *out = b
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("4B")))
                .collect();
            Ok(())
        };
        let read_u64 = |name: &str, out: &mut Vec<u64>| -> Result<()> {
            let mut b = Vec::new();
            self.be.module_global_bytes(module, name, cap * 8, &mut b)?;
            *out = b
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().expect("8B")))
                .collect();
            Ok(())
        };
        let (mut op, mut wait) = (Vec::new(), Vec::new());
        let (mut gate, mut body, mut sig) = (Vec::new(), Vec::new(), Vec::new());
        read_u32("g_tr_op", &mut op)?;
        read_u32("g_tr_wait", &mut wait)?;
        read_u64("g_tr_gate", &mut gate)?;
        read_u64("g_tr_body", &mut body)?;
        read_u64("g_tr_sig", &mut sig)?;

        // Per-opcode accumulation: (count, Σgate, Σbody, Σsig, Σwait_edges).
        let mut acc: rustc_hash::FxHashMap<u32, (u64, u64, u64, u64, u64)> =
            rustc_hash::FxHashMap::default();
        let (mut tg, mut tb, mut ts) = (0u64, 0u64, 0u64);
        for i in 0..cap {
            let e = acc.entry(op[i]).or_default();
            e.0 += 1;
            e.1 += gate[i];
            e.2 += body[i];
            e.3 += sig[i];
            e.4 += wait[i] as u64;
            tg += gate[i];
            tb += body[i];
            ts += sig[i];
        }
        let total = (tg + tb + ts).max(1) as f64;
        let mut rows: Vec<_> = acc.into_iter().collect();
        rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.1 + v.2 + v.3));
        let mut s = format!(
            "PLOW_NV_TRACE block-0 profile: {cap} packets | gate {:.1}% body {:.1}% \
             signal {:.1}% of {} kcyc\n  {:<16} {:>5} {:>10} {:>10} {:>10} {:>8}\n",
            100.0 * tg as f64 / total,
            100.0 * tb as f64 / total,
            100.0 * ts as f64 / total,
            (tg + tb + ts) / 1000,
            "op",
            "n",
            "gate_kcyc",
            "body_kcyc",
            "sig_kcyc",
            "waits",
        );
        for (op, (cnt, g, b, sg, w)) in rows {
            s += &format!(
                "  {:<16} {:>5} {:>10} {:>10} {:>10} {:>8}\n",
                devop_name(op),
                cnt,
                g / 1000,
                b / 1000,
                sg / 1000,
                w,
            );
        }
        Ok(Some(s))
    }
}

/// A slice of `#[repr(C)]` POD mirrors as the raw bytes an upload wants.
///
/// One definition, because this is the crate's only routine `unsafe` cast and
/// two copies of it were two places to get the length expression wrong.
/// `size_of_val` (not `len() * size_of::<T>()`) so padding is never dropped.
///
/// # Safety contract (not `unsafe`, but a contract nonetheless)
/// `T` must be a `#[repr(C)]` POD with no padding the caller cares about and no
/// pointers the device would dereference as host addresses. Every caller passes
/// a devgen mirror type, which is exactly that.
fn pod_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T: Copy` `#[repr(C)]` mirrors read as raw bytes for an upload;
    // the range is exactly the slice's own allocation (`size_of_val`), and the
    // borrow keeps `v` alive for the returned lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Convert a raw byte buffer of bf16 values into an f32 slice in-place.
/// `src` must be exactly `dst.len() * 2` bytes (one `u16` per `f32` output).
///
/// The inner loop is a trivial indexed pattern that auto-vectorises on x86-64
/// (LLVM emits `vpmovzxwd` + `vpslld` + `vmovups` for AVX2). **Measured
/// 2026-07-28**: 12.30 µs at `vocab = 262144` — 128 GB/s of read+write traffic,
/// i.e. memory-bound, and byte-for-byte the same time as the iterator/
/// `bytemuck::cast_slice` form (12.27 µs). There is nothing left to vectorise
/// here; do not "optimise" it again.
#[inline]
fn bf16_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len() * 2);
    // SAFETY: &[u8] of even length → &[u16] with half the count.
    let halves: &[u16] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, dst.len()) };
    for i in 0..dst.len() {
        dst[i] = f32::from_bits((halves[i] as u32) << 16);
    }
}

/// Device opcode → short name for the trace profile (the `DevOp` values the
/// kernel records in `g_tr_op`). Unlisted opcodes print their number.
fn devop_name(op: u32) -> String {
    match op {
        1 => "RmsNorm",
        2 => "RowRms",
        3 => "HeadNormRope",
        4 => "Residual",
        5 => "Glu",
        6 => "Embed",
        7 => "SoftCap",
        8 => "Gemm",
        9 => "GemmNorm",
        10 => "Gemv",
        11 => "FlashPrefill",
        12 => "FlashDecode",
        13 => "FlashMerge",
        14 => "GemmSmall",
        15 => "GemmMed",
        16 => "NormResidual",
        17 => "Argmax",
        18 => "ArgmaxFin",
        19 => "GemvGlu",
        20 => "GemmGlu",
        21 => "AddNorm",
        22 => "GemvQkv",
        23 => "NormResidualNorm",
        30 => "GemvFp8",
        31 => "GemvGluFp8",
        32 => "QuantFp8",
        37 => "HeadNormRopeFp8",
        38 => "FlashDecodeFp8",
        44 => "GemvFp8Blk",
        80 => "GemvArgmax",
        107 => "GemmFp8Blk",
        _ => return op.to_string(),
    }
    .to_string()
}

impl Drop for GpuEngine {
    /// Model unload: quiesce the device and unload both cubin modules; every
    /// owned `DeviceMem` field (weights, KV rings, tables, counter block)
    /// then frees itself after this body, returning the engine's whole VRAM
    /// footprint to the driver. `d_ctr`/`d_gq_cursor` are views and free
    /// nothing — `_ctr_block` owns that storage. Errors are logged: Drop has
    /// no error channel, and a failed unload only pins module storage.
    fn drop(&mut self) {
        // On a poisoned context these calls are EXPECTED to fail (bind
        // short-circuits) — that is old news, logged once at the poisoning
        // site, so the per-call reports drop to debug.
        let poisoned = self.be.is_poisoned();
        let report = |e: &crate::RuntimeError, what: &str| {
            if poisoned {
                tracing::debug!(error = %e, "{what} (context poisoned)");
            } else {
                tracing::warn!(error = %e, "{what}");
            }
        };
        // Every public step/prefill path synchronizes before returning, but a
        // failed launch may leave work queued — never free under a running
        // cooperative kernel.
        if let Err(e) = self.be.synchronize() {
            report(&e, "synchronize at engine unload");
        }
        drop(self.decode_contexts.take());
        self.decode_rungs.clear();
        drop(self.qwen_prefill.take());
        if let Some(m) = self.module_pf.take() {
            if let Err(e) = self.be.module_unload(&m) {
                report(&e, "unload prefill module");
            }
        }
        if let Some(s) = self.sampler.take() {
            if let Err(e) = self.be.module_unload(&s._module) {
                report(&e, "unload sampler module");
            }
        }
        if let Some(m) = self.multistep.take() {
            if let Err(e) = self.be.module_unload(&m._module) {
                report(&e, "unload multistep module");
            }
        }
        // T35 segment-chain graphs and the SegPf pair's own modules. Engines
        // share one backend whose primary-context retain outlives them, so the
        // backend-drop drain never runs during a serving session — without
        // these, every S1 model swap pins three cubins and leaks one
        // instantiated graph per (bucket, slot) until process exit.
        if let Some(g) = self.cublaslt_decode_graph.take() {
            self.be.graph_destroy(g);
        }
        self.cublaslt_decode.clear();
        for (_, g) in self.seg_graphs.drain() {
            self.be.graph_destroy(g);
        }
        for role in &mut self.packet_roles {
            drop(role.take());
        }
        if let Some(sp) = self.seg_pf.take() {
            if let Some(small) = sp.small_gemm {
                if let Err(e) = self.be.module_unload(&small._module) {
                    report(&e, "unload small GEMM object");
                }
            }
            if let Err(e) = self.be.module_unload(&sp._m_flash) {
                report(&e, "unload seg flash module");
            }
            if let Err(e) = self.be.module_unload(&sp._m_gemm) {
                report(&e, "unload seg gemm module");
            }
            if let Some(m) = sp._m_fa512 {
                if let Err(e) = self.be.module_unload(&m) {
                    report(&e, "unload seg fa512 module");
                }
            }
        }
    }
}

#[cfg(test)]
mod kv_tmap_tests;

#[cfg(test)]
mod attention_role_tests;

#[cfg(test)]
mod prefill_patch_tests;

#[cfg(test)]
mod slab_tests;

#[cfg(test)]
mod profile_tests;

#[cfg(test)]
mod prefix_selection_tests;

#[cfg(test)]
mod recurrent_tests;

#[cfg(test)]
mod qwen_tests;

#[cfg(test)]
#[test]
fn qwen_prefill_state_views_select_exact_batch_slots() {
    const STRIDE: u64 = 48 * 128 * 128 * 4;
    let state = DeviceMem::view(4096, 4 * STRIDE);
    for slot in 0..4 {
        let view = qwen_state_slot(&state, slot, 4).unwrap();
        assert_eq!(view.base, 4096 + slot as u64 * STRIDE);
        assert_eq!(view.len, STRIDE);
    }
    assert!(qwen_state_slot(&state, 4, 4).is_err());
    assert!(qwen_state_slot(&state, 0, 1).is_err());
    assert!(qwen_state_slot(&state, 0, 0).is_err());
    assert_eq!(
        qwen_state_slot(&DeviceMem::view(4096, STRIDE), 0, 1)
            .unwrap()
            .base,
        4096
    );
}

#[cfg(test)]
mod decode_rung_tests;

#[cfg(test)]
mod gemv_role_tests;

#[cfg(test)]
mod mxfp4_moe_role_tests;

mod fp8_m1_role;
use fp8_m1_role::{load_fp8_m1_role, validate_fp8_role_checkpoint};

mod cublaslt;
