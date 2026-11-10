//! Fish S2 depth-stage programs (`plow_asset::fish_depth`): after a backbone step the runtime
//! stages each row's sampling inputs, launches the packet's depth program of a width covering the
//! rows' slots on the decode interpreter, and reads each row's frame codes back.

use plow_asset::fish_depth::{Manifest, Sampler};

use crate::asset::devblob::DevBlob;
use crate::device::cuda::{KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::exec::packet_runtime::{PacketBuffer, PacketBufferPool};
use crate::{Result, RuntimeError};
use std::sync::Arc;

use super::decode_object::BoundDecodeObject;
use super::{DevInst64, DevOp, DevProgram, GpuEngine, BLOCK, TENSOR_NONE16};

const STAGE_CONTROL_BYTES: usize = 8 + 16 + 8 + 4 + 4;

fn prompt_token_matches_decode(m: &Manifest, token: u32) -> bool {
    m.packed.as_ref().is_some_and(|p| {
        token < p.input_vocab
            && (p.decode_token.is_none()
                || token
                    .checked_sub(m.semantic_base)
                    .is_some_and(|class| class < m.semantic_tokens))
    })
}

struct Loaded {
    rows: u32,
    arg: DevProgram,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

pub(super) struct FishDepth<H = PinnedHost> {
    manifest: Manifest,
    programs: Vec<Loaded>,
    /// Staging for the widest program: seed u64 (first, so aligned), params `[rows][4]` f32, then
    /// top_k, step, hidden_row u32 (Maya 2: `[rows][2]` control, hidden_row, cb0), then the
    /// read-back codes `[rows][codebooks]`.
    stage: Arc<H>,
    rows: usize,
    maya2: Option<Maya2>,
    compact: Option<CompactDepth<H>>,
    source: Option<Arc<dyn Send + Sync>>,
    pool: Option<PacketBufferPool>,
    retain_output: bool,
    captured: Option<PacketBuffer>,
}

struct CompactDepth<H = PinnedHost> {
    remap: Loaded,
    index: DeviceMem,
    host_index: Arc<H>,
    _remap_tensors: DeviceMem,
    depth_tensors: DeviceMem,
    logits: Option<DeviceMem>,
}

fn inverse_rows(index: &mut [u32], owners: &[u32]) -> Result<()> {
    index.fill(u32::MAX);
    if owners.is_empty() || owners.len() > index.len() {
        return Err(reject("invalid compact output count"));
    }
    for (row, &slot) in owners.iter().enumerate() {
        let dst = index
            .get_mut(slot as usize)
            .ok_or_else(|| reject("compact slot out of range"))?;
        if *dst != u32::MAX {
            return Err(reject("duplicate compact slot"));
        }
        *dst = row as u32;
    }
    for row in index {
        if *row == u32::MAX {
            *row = 0;
        }
    }
    Ok(())
}

impl CompactDepth {
    fn load(e: &GpuEngine, m: &Manifest) -> Result<Option<Self>> {
        let (Some(contract), Some(terminal)) = (&m.packed, &e.packed_terminal) else {
            return Ok(None);
        };
        let (hidden, pitch) = terminal.hidden_rows();
        let output = &e.devp[contract.hidden as usize];
        let logits = &e.devp[e.t_logits];
        if pitch == 0
            || pitch % 2 != 0
            || output.len < e.batch as u64 * pitch
            || (output.base < logits.base + logits.len && logits.base < output.base + output.len)
            || e.handle_of("in.frame_codes") != Some(m.tensors.prompt_codes as usize)
            || e.devp[m.tensors.prompt_codes as usize].len
                < e.pf_max_rows() as u64 * u64::from(m.codebooks) * 4
            || e.prefill.iter().any(|b| {
                !b.h_inst
                    .iter()
                    .any(|i| i.op == DevOp::EmbedFrameBf16 as u16 && i.i[2] == contract.input_vocab)
            })
        {
            return Err(reject("packed depth hidden or embedding extent mismatch"));
        }
        let blocks = u16::try_from(e.batch).map_err(|_| reject("packed depth batch overflow"))?;
        let index = e.be.alloc(0, e.batch as u64 * 4)?;
        let logits = contract
            .backbone_logits
            .then(|| e.be.alloc(0, e.batch as u64 * e.vocab as u64 * 2))
            .transpose()?;
        let mut pointers: Vec<u64> = e.devp.iter().map(|p| p.base).collect();
        let base = u16::try_from(pointers.len())
            .ok()
            .filter(|&h| h < TENSOR_NONE16 - 2)
            .ok_or_else(|| reject("packed depth tensor handles exhausted"))?;
        pointers.extend([hidden, index.base, logits.as_ref().map_or(0, |p| p.base)]);
        let remap_tensors = e.be.alloc(0, pointers.len() as u64 * 8)?;
        e.be.upload(&remap_tensors, 0, super::pod_bytes(&pointers))?;
        if let Some(logits) = &logits {
            pointers[e.t_logits] = logits.base;
        }
        let depth_tensors = e.be.alloc(0, pointers.len() as u64 * 8)?;
        e.be.upload(&depth_tensors, 0, super::pod_bytes(&pointers))?;
        let gather = |dst, src, width| {
            let mut i = DevInst64 {
                op: DevOp::RowGather as u16,
                blocks,
                t: [TENSOR_NONE16; 8],
                ..Default::default()
            };
            i.t[..3].copy_from_slice(&[dst, src, base + 1]);
            i.i[..3].copy_from_slice(&[e.batch as u32, width, e.batch as u32]);
            i
        };
        let mut insts = vec![gather(contract.hidden, base, (pitch / 2) as u32)];
        if logits.is_some() {
            insts.push(gather(base + 2, e.t_logits as u16, e.vocab as u32));
        }
        let program = super::packed_terminal::chain(insts, e.grid_pf, e.batch as u32);
        plow_asset::aux_program::Section {
            n_cu: e.grid_pf,
            programs: vec![program.clone()],
        }
        .validate(pointers.len())
        .map_err(reject)?;
        let (arg, _, counter_bytes, tables) =
            super::mixed_step::upload_program(&e.be, &program, remap_tensors.base, 0, 0)?;
        Ok(Some(Self {
            remap: Loaded {
                rows: e.batch as u32,
                arg,
                counter_bytes,
                _tables: tables,
            },
            index,
            host_index: Arc::new(e.be.host_alloc_pinned(e.batch * 4)?),
            _remap_tensors: remap_tensors,
            depth_tensors,
            logits,
        }))
    }

    fn launch(&mut self, e: &GpuEngine, owners: &[u32]) -> Result<()> {
        inverse_rows(
            bytemuck::cast_slice_mut(depth_stage(&mut self.host_index)?),
            owners,
        )?;
        // The predictor drains the stream before this pinned stage can be reused.
        unsafe {
            e.be.memcpy_htod_async(self.index.base, self.host_index.as_slice(), &e.stream)?;
        }
        e.be.memset_d8_async(
            self.remap.arg.counters,
            0,
            self.remap.counter_bytes,
            &e.stream,
        )?;
        let mut arg = self.remap.arg;
        let mut args = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        e.be.launch_cooperative(
            e.f_pf.unwrap(),
            e.grid_pf,
            BLOCK,
            e.smem_pf,
            &mut args,
            Some(&e.stream),
        )
    }
}

#[derive(Clone, Copy)]
struct DepthOutputs {
    tensors: u64,
    logits: u64,
}

/// Maya 2's host cb0 controller: per-slot state, and the widest program's read-back cb0 logits.
struct Maya2 {
    slots: Vec<Option<crate::tts::maya2::Slot>>,
    memory: Vec<SpeakerMemory>,
    logits: PinnedHost,
    scratch: Vec<f32>,
    /// The depth levels' (temperature, top-k).
    depth: (f32, u32),
    /// The depth programs run FP32 speech ops: they launch on the bundle's speech object, over the
    /// engine's own tensor table.
    speech: SpeechFn,
    /// The null speaker's depth memory, `[layers][hidden]` (named, plain and `icl` voices).
    null_memory: Vec<f32>,
}

#[derive(Clone)]
enum SpeakerMemory {
    Unbound,
    Null,
    Voice(std::sync::Arc<[f32]>),
}

impl SpeakerMemory {
    fn matches(&self, memory: Option<&std::sync::Arc<[f32]>>) -> bool {
        match (self, memory) {
            (Self::Null, None) => true,
            (Self::Voice(old), Some(new)) => std::sync::Arc::ptr_eq(old, new),
            _ => false,
        }
    }
}

struct SpeechFn {
    _module: Module,
    f: KernelFn,
    smem: u32,
}

impl SpeechFn {
    /// The speech object beside the packet, checked to carry every speech op of `programs`.
    fn load(
        e: &GpuEngine,
        assets: &std::path::Path,
        programs: &[plow_asset::aux_program::Program],
        (blob, raw): (&DevBlob, &[u8]),
    ) -> Result<Self> {
        let (file, symbol) = super::packet_exec::speech_object(e.be.compute_capability());
        let path = assets.join(file);
        let image = std::fs::read(&path).map_err(|source| RuntimeError::Io {
            path: path.clone(),
            source,
        })?;
        let module = e.be.module_load(&image)?;
        let f = e.be.get_function(&module, symbol)?;
        let need = programs
            .iter()
            .flat_map(|p| p.insts.iter())
            .map(|i| u32::from(i.op))
            .filter(|op| (163..=204).contains(op))
            .fold(0u64, |m, op| m | 1 << (op - 163));
        if let (Some(lo), Some(hi)) = (
            e.be.module_global_u32(&module, "plow_speech_ops_lo_speech")?,
            e.be.module_global_u32(&module, "plow_speech_ops_hi_speech")?,
        ) {
            let missing =
                super::packet_exec::missing_speech_ops(need, u64::from(hi) << 32 | u64::from(lo));
            if !missing.is_empty() {
                return Err(reject(format!(
                    "{} lacks speech ops {missing:?}",
                    path.display()
                )));
            }
        }
        super::packet_exec::check_speech_flag_layout(&e.be, &module, &path, blob, raw)?;
        let caps = super::packet_exec::missing_speech_caps(
            super::packet_exec::speech_caps_required(programs.iter().flat_map(|p| &p.insts)),
            e.be.module_global_u32(&module, "plow_speech_caps_speech")?,
        );
        if caps != 0 {
            return Err(reject(format!(
                "{} lacks speech capabilities {caps:#x}",
                path.display()
            )));
        }
        let smem =
            e.be.module_global_u32(&module, "plow_arena_bytes_speech")?
                .unwrap_or(49152);
        e.be.set_max_dynamic_smem(f, smem)?;
        if e.be.occupancy_blocks_per_sm(f, BLOCK, smem as usize)? * e.be.sm_count() < e.grid {
            return Err(reject("speech object cannot hold the cooperative grid"));
        }
        Ok(SpeechFn {
            _module: module,
            f,
            smem,
        })
    }
}

/// One row of a depth launch: the slot whose frame it samples and its sampling state.
#[derive(Clone, Copy, Debug)]
pub struct FishRow {
    pub slot: usize,
    /// Row of the backbone's final-norm activation that produced this frame's logits.
    pub hidden_row: u32,
    pub seed: u64,
    /// Frame index within the request (RAS is off at 0).
    pub step: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub ras_temperature: f32,
    pub ras_top_p: f32,
    /// HF repetition penalty on the backbone class (Breeze sampler; 1.0 = none).
    pub repetition_penalty: f32,
    /// A CFG pair (`pair_cfg` packets): (scale s, the unconditional member's hidden row). Slot
    /// `slot + 1` runs the unconditional depth pass and receives the same codes.
    pub cfg: Option<(f32, u32)>,
}

/// What a Fish depth packet emits per frame.
#[derive(Clone, Copy, Debug)]
pub struct FishInfo {
    pub codebooks: usize,
    pub semantic_base: u32,
    pub semantic_tokens: u32,
    pub stop_class: u32,
    pub pair_cfg: bool,
}

/// The packet's null depth memory (`maya2.depth.cross.{l}`, sized like a slot row of `slot_memory`).
fn null_memory(e: &GpuEngine, m: &Manifest) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    for (l, &h) in m.slot_memory.iter().enumerate() {
        let w = e
            .handle_of(&format!("maya2.depth.cross.{l}"))
            .ok_or_else(|| reject(format!("maya2.depth.cross.{l} missing")))?;
        let mem = &e.devp[w];
        if e.devp[usize::from(h)].len < mem.len * e.batch as u64 {
            return Err(reject("slot memory smaller than its slots"));
        }
        let mut b = vec![0u8; mem.len as usize];
        e.be.download(mem, 0, &mut b)?;
        out.extend(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
    }
    if out.is_empty() {
        return Err(reject("maya2 depth stage without slot memory"));
    }
    Ok(out)
}

/// cb0 head rows: the codebook and the stop class.
const STOP_ROWS: usize = crate::tts::maya2::STOP + 1;

fn reject(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("fish depth: {e}"))
}

fn seed_frame_codes(
    staging: &mut [u8],
    seeds: &[(usize, &[u32])],
    batch: usize,
    codebooks: usize,
) -> Result<usize> {
    let bytes = seeds
        .len()
        .checked_mul(codebooks)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= staging.len())
        .ok_or_else(|| reject("seed staging capacity exceeded"))?;
    if codebooks == 0
        || codebooks > 64
        || seeds
            .iter()
            .any(|&(slot, codes)| slot >= batch || codes.len() > codebooks)
        || seeds.windows(2).any(|s| s[0].0 >= s[1].0)
    {
        return Err(reject("seed slots must be ordered, distinct and in range"));
    }
    staging[..bytes].fill(0);
    for (row, &(_, codes)) in seeds.iter().enumerate() {
        let start = row * codebooks * 4;
        staging[start..start + codes.len() * 4].copy_from_slice(bytemuck::cast_slice(codes));
    }
    Ok(bytes)
}

fn depth_stage(stage: &mut Arc<PinnedHost>) -> Result<&mut [u8]> {
    Arc::get_mut(stage)
        .map(PinnedHost::as_mut_slice)
        .ok_or_else(|| reject("depth staging transfer did not retire"))
}

impl<H> FishDepth<H> {
    fn quarantine(&self) {
        std::mem::forget(self.stage.clone());
        if let Some(compact) = &self.compact {
            std::mem::forget(compact.host_index.clone());
        }
        if let Some(source) = &self.source {
            std::mem::forget(source.clone());
        }
    }
}

impl FishDepth {
    /// The packet's depth stage, `None` when it carries none; read from the engine's own blob.
    pub(super) fn load_if_present(
        e: &mut GpuEngine,
        blob: &DevBlob,
        raw: &[u8],
        assets: &std::path::Path,
    ) -> Result<Option<Self>> {
        let Some(meta) = blob.reserved_metadata(raw, plow_asset::fish_depth::SECTION)? else {
            return Ok(None);
        };
        let manifest: Manifest = serde_json::from_slice(meta).map_err(reject)?;
        let bytes = blob
            .reserved_metadata(raw, plow_asset::fish_depth::PROGRAMS_SECTION)?
            .ok_or_else(|| reject("depth programs section missing"))?;
        Self::load(e, manifest, bytes, assets, (blob, raw)).map(Some)
    }

    fn load(
        e: &mut GpuEngine,
        manifest: Manifest,
        bytes: &[u8],
        assets: &std::path::Path,
        packet: (&DevBlob, &[u8]),
    ) -> Result<Self> {
        let section =
            plow_asset::aux_program::parse(bytes, e.grid, e.devp.len()).map_err(reject)?;
        manifest
            .validate(section.programs.len(), e.devp.len())
            .map_err(reject)?;
        let rows = manifest.rows.iter().copied().max().unwrap_or(0) as usize;
        let t = &manifest.tensors;
        let k = manifest.codebooks as usize;
        let need = |h: u16, bytes: usize| e.devp[usize::from(h)].len >= bytes as u64;
        let maya2 = manifest.sampler == Sampler::Maya2;
        if maya2
            && (rows > e.batch
                || !need(t.params, rows * 16)
                || !need(t.seed, rows * 8)
                || !need(t.top_k, rows * 8)
                || !need(t.history, rows * 4)
                || !need(t.hidden_row, rows * 4)
                || !need(t.frame_codes, e.batch * k * 4)
                || e.vocab() <= crate::tts::maya2::STOP
                || manifest.depth_sampling.is_none())
        {
            return Err(reject(
                "maya2 tensor sizes disagree with the program widths",
            ));
        }
        if !maya2
            && (rows > e.batch
                || !need(t.params, rows * 16)
                || !need(t.seed, rows * 8)
                || !need(t.history, e.batch * manifest.ras_window as usize * 4)
                || !need(t.frame_codes, e.batch * k * 4)
                || !need(
                    t.top_k,
                    rows * if manifest.sampler == Sampler::Breeze {
                        8
                    } else {
                        4
                    },
                )
                || [t.step, t.hidden_row].iter().any(|&h| !need(h, rows * 4)))
        {
            return Err(reject("tensor sizes disagree with the program widths"));
        }
        let programs = section
            .programs
            .iter()
            .zip(&manifest.rows)
            .map(|(p, &rows)| {
                let (arg, _, counter_bytes, tables) =
                    super::mixed_step::upload_program(&e.be, p, e.d_tens.base, 0, 0)?;
                Ok(Loaded {
                    rows,
                    arg,
                    counter_bytes,
                    _tables: tables,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stage =
            e.be.host_alloc_pinned(rows * STAGE_CONTROL_BYTES + e.batch * k * 4)?;
        let maya2 = match (maya2, manifest.depth_sampling) {
            (true, Some([t, k])) => Some(Maya2 {
                slots: vec![None; e.batch],
                memory: vec![SpeakerMemory::Unbound; e.batch],
                logits: e.be.host_alloc_pinned(rows * e.vocab() * 2)?,
                scratch: Vec::new(),
                depth: (f32::from_bits(t), k),
                speech: SpeechFn::load(e, assets, &section.programs, packet)?,
                null_memory: null_memory(e, &manifest)?,
            }),
            _ => None,
        };
        let compact = CompactDepth::load(e, &manifest)?;
        let h = manifest.tensors.frame_codes as usize;
        let mem = &e.devp[h];
        let contains = |base: u64, bytes: u64| {
            mem.base >= base
                && mem
                    .base
                    .checked_add(mem.len)
                    .is_some_and(|end| base.checked_add(bytes).is_some_and(|limit| end <= limit))
        };
        let source: Option<Arc<dyn Send + Sync>> = if mem.owns_storage() {
            let view = mem.subview(0, mem.len)?;
            Some(Arc::new(std::mem::replace(&mut e.devp[h], view)))
        } else if match e._weight_slab.as_ref() {
            super::WeightSlab::Flat(slab) => contains(slab.base, slab.len),
            super::WeightSlab::Vmm(slab) => contains(slab.base(), slab.len()),
            super::WeightSlab::PerTensor => false,
        } {
            Some(e._weight_slab.clone())
        } else {
            None
        };
        let pool = if source.is_some() {
            match PacketBufferPool::cuda(e.be.clone(), e.batch * k * 4, e.max_ctx) {
                Ok(pool) => Some(pool),
                Err(error) => {
                    tracing::warn!(%error, "depth output pool unavailable; using host frames");
                    None
                }
            }
        } else {
            None
        };
        tracing::info!(
            programs = programs.len(),
            rows,
            codebooks = k,
            "fish depth programs loaded"
        );
        Ok(Self {
            manifest,
            programs,
            stage: Arc::new(stage),
            rows,
            maya2,
            compact,
            source,
            pool,
            retain_output: false,
            captured: None,
        })
    }
}

impl GpuEngine {
    pub(super) fn quarantine_depth_source(&self) {
        if let Some(d) = &self.fish {
            d.quarantine();
        }
        std::mem::forget(self.be.clone());
    }

    /// The loaded depth stage's frame layout (a depth-stage packet).
    pub fn fish_info(&self) -> Result<FishInfo> {
        let m = &self
            .fish
            .as_ref()
            .ok_or_else(|| reject("not enabled"))?
            .manifest;
        Ok(FishInfo {
            codebooks: m.codebooks as usize,
            semantic_base: m.semantic_base,
            semantic_tokens: m.semantic_tokens,
            stop_class: m.stop_class,
            pair_cfg: m.pair_cfg,
        })
    }

    /// Start a request on `slot`: an empty RAS history.
    pub fn fish_reset_slot(&mut self, slot: usize) -> Result<()> {
        let d = self.fish.as_ref().ok_or_else(|| reject("not enabled"))?;
        if slot >= self.batch {
            return Err(reject("slot out of range"));
        }
        if d.maya2.is_some() {
            return Ok(());
        }
        let w = d.manifest.ras_window as usize * 4;
        let h = &self.devp[usize::from(d.manifest.tensors.history)];
        self.be
            .memset_d8_async(h.base + (slot * w) as u64, 0, w, &self.stream)
    }

    /// Start a Maya 2 piece on `slot` under `plan` (after [`Self::fish_reset_slot`]), with its
    /// speaker's depth memory (`[layers][hidden]`; `None`: the null speaker's).
    pub fn maya2_bind(
        &mut self,
        slot: usize,
        plan: crate::tts::maya2::Plan,
        memory: Option<&std::sync::Arc<[f32]>>,
    ) -> Result<()> {
        let d = self.fish.as_mut().ok_or_else(|| reject("not enabled"))?;
        let m = d
            .maya2
            .as_mut()
            .ok_or_else(|| reject("not a maya2 depth stage"))?;
        let s = m
            .slots
            .get_mut(slot)
            .ok_or_else(|| reject("slot out of range"))?;
        *s = Some(crate::tts::maya2::Slot::new(plan));
        let mem = memory.map_or(m.null_memory.as_slice(), |m| m.as_ref());
        let layers = d.manifest.slot_memory.len();
        if mem.len() != m.null_memory.len() {
            return Err(reject("speaker memory width disagrees with the packet"));
        }
        if m.memory[slot].matches(memory) {
            return Ok(());
        }
        // These tensors are read-only depth inputs. Retain the owner so pointer identity cannot
        // outlive its allocation, and publish only after every layer's upload succeeds.
        m.memory[slot] = SpeakerMemory::Unbound;
        let width = mem.len() / layers;
        for (l, &h) in d.manifest.slot_memory.iter().enumerate() {
            let row: &[u8] = bytemuck::cast_slice(&mem[l * width..(l + 1) * width]);
            self.be
                .upload(&self.devp[usize::from(h)], (slot * row.len()) as u64, row)?;
        }
        m.memory[slot] = memory.map_or(SpeakerMemory::Null, |m| SpeakerMemory::Voice(m.clone()));
        Ok(())
    }

    pub(super) fn forget_depth_memory(&mut self) {
        if let Some(m) = self.fish.as_mut().and_then(|d| d.maya2.as_mut()) {
            m.memory.fill(SpeakerMemory::Unbound);
        }
    }

    /// A prompt's first frame samples from prefill's logits (row 0): move them to `slot`'s row.
    pub fn fish_logits_to_slot(&mut self, slot: usize) -> Result<()> {
        if slot >= self.batch {
            return Err(reject("slot out of range"));
        }
        if slot == 0 {
            return Ok(());
        }
        let row = self.vocab() as u64 * 2;
        let l = &self.devp[self.t_logits];
        self.be
            .memcpy_dtod_async(l.base + slot as u64 * row, l.base, row, &self.stream)
    }

    /// Sample one frame for each of `rows` (distinct slots) and append its codes
    /// (`codebooks` per row, column 0 the backbone class) to `out`.
    pub fn fish_depth_step(&mut self, rows: &[FishRow], out: &mut Vec<u32>) -> Result<()> {
        self.fish_depth_step_inner(rows, None, false, out)
            .map(|_| ())
    }

    pub fn fish_depth_step_compact(
        &mut self,
        rows: &[FishRow],
        owners: &[u32],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        self.fish_depth_step_inner(rows, Some(owners), false, out)
            .map(|_| ())
    }

    pub fn fish_depth_step_retained(
        &mut self,
        rows: &[FishRow],
        owners: Option<&[u32]>,
        out: &mut Vec<u32>,
    ) -> Result<Option<PacketBuffer>> {
        self.fish_depth_step_inner(rows, owners, true, out)
    }

    fn fish_depth_step_inner(
        &mut self,
        rows: &[FishRow],
        owners: Option<&[u32]>,
        retain: bool,
        out: &mut Vec<u32>,
    ) -> Result<Option<PacketBuffer>> {
        let mut d = self.fish.take().ok_or_else(|| reject("not enabled"))?;
        d.captured = None;
        d.retain_output = retain;
        let mut r = (|| {
            let outputs = if let Some(owners) = owners {
                if rows.iter().any(|r| {
                    r.cfg.is_some()
                        || r.hidden_row as usize != r.slot
                        || !owners.contains(&(r.slot as u32))
                }) {
                    return Err(reject("invalid compact predictor row"));
                }
                let compact = d
                    .compact
                    .as_mut()
                    .ok_or_else(|| reject("no packed depth contract"))?;
                compact.launch(self, owners)?;
                Some(DepthOutputs {
                    tensors: compact.depth_tensors.base,
                    logits: compact
                        .logits
                        .as_ref()
                        .map_or(self.devp[self.t_logits].base, |p| p.base),
                })
            } else {
                None
            };
            if d.maya2.is_some() {
                self.maya2_launch(&mut d, rows, outputs, out)
            } else {
                self.fish_launch(&mut d, rows, outputs, out)
            }
        })();
        if r.is_err() {
            if let Err(error) = self.be.stream_synchronize(&self.stream) {
                d.quarantine();
                std::mem::forget(self.be.clone());
                #[cfg(target_os = "linux")]
                let error = self.be.quarantine_staging(&error);
                r = Err(error);
            }
        }
        let captured = d.captured.take();
        self.fish = Some(d);
        self.note_step_result(r.is_err());
        r.map(|()| captured)
    }

    fn depth_read_codes(&self, d: &mut FishDepth, codes_at: usize, bytes: usize) -> Result<()> {
        let pending = if d.retain_output {
            d.source
                .as_ref()
                .and_then(|owner| d.pool.as_mut()?.acquire(bytes, owner.clone()))
        } else {
            None
        };
        let address = self.devp[d.manifest.tensors.frame_codes as usize].base;
        let queued = (|| {
            if let Some(copy) = &pending {
                self.be
                    .memcpy_dtod_async(copy.address(), address, bytes as u64, &self.stream)?;
            }
            // SAFETY: the persistent host buffer and device source outlive the drain below.
            unsafe {
                self.be.memcpy_dtoh_async(
                    &mut depth_stage(&mut d.stage)?[codes_at..codes_at + bytes],
                    address,
                    &self.stream,
                )
            }
        })();
        self.be.stream_synchronize(&self.stream)?;
        // SAFETY: even an enqueue error is drained before either endpoint is released.
        let captured = pending.map(|copy| unsafe { copy.retire() });
        queued?;
        d.captured = captured;
        Ok(())
    }

    /// Depth programs of 8 rows up to the `_gw` sibling's wide-arm rung launch on it (full claim),
    /// as the backbone's batched rungs do; at and past that rung the depth's projections are below
    /// the wide arm's size floor and the main object's walk is faster. Qwen3-TTS depth ms, B=16 /
    /// 32: 5.87 / 7.10 -> 5.29 / 6.58; B=48 / 64 on the `_gw` object: 10.35 / 10.62 -> 10.76 / 11.17.
    fn depth_object(&self, rows: usize) -> Option<&BoundDecodeObject> {
        self.depth_gw
            .as_ref()
            .filter(|(min_rows, _)| (8..*min_rows).contains(&rows))
            .map(|(_, o)| &**o)
    }

    fn fish_launch(
        &mut self,
        d: &mut FishDepth,
        rows: &[FishRow],
        outputs: Option<DepthOutputs>,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let batch = self.batch;
        let width = rows
            .iter()
            .map(|r| r.slot + 1 + usize::from(r.cfg.is_some()))
            .max()
            .unwrap_or(0);
        if rows.iter().any(|r| r.cfg.is_some()) && !d.manifest.pair_cfg {
            return Err(reject("packet's sampler takes no CFG pairs"));
        }
        let p = d
            .programs
            .iter()
            .filter(|p| p.rows as usize >= width)
            .min_by_key(|p| p.rows)
            .ok_or_else(|| {
                reject(format!(
                    "no program covers slot {}",
                    width.saturating_sub(1)
                ))
            })?;
        let partner = |r: &FishRow| r.cfg.map(|_| r.slot + 1);
        if rows.iter().any(|r| {
            r.slot + usize::from(r.cfg.is_some()) >= batch
                || r.top_k == 0
                || r.top_k & (plow_asset::fish_depth::GUIDE | plow_asset::fish_depth::FOLLOW) != 0
                || rows.iter().any(|o| partner(o) == Some(r.slot))
        }) {
            return Err(reject(
                "row slot out of range, top_k 0 or out of range, or a slot on a CFG partner",
            ));
        }
        let n = p.rows as usize;
        let k = d.manifest.codebooks as usize;
        {
            let s = depth_stage(&mut d.stage)?;
            let (seeds, rest) = s.split_at_mut(d.rows * 8);
            let (params, rest) = rest.split_at_mut(d.rows * 16);
            let words = &mut rest[..d.rows * 12];
            let params: &mut [f32] = bytemuck::cast_slice_mut(params);
            let words: &mut [u32] = bytemuck::cast_slice_mut(words);
            let seeds: &mut [u64] = bytemuck::cast_slice_mut(seeds);
            for r in 0..n {
                params[r * 4..r * 4 + 4].fill(1.0);
                words[r] = 0;
                words[d.rows + r] = 0;
                words[2 * d.rows + r] = r as u32;
                seeds[r] = 0;
            }
            let breeze = d.manifest.sampler == Sampler::Breeze;
            for row in rows {
                let r = row.slot;
                params[r * 4..r * 4 + 4].copy_from_slice(&if breeze {
                    [row.temperature, row.top_p, row.repetition_penalty, 1.0]
                } else {
                    [
                        row.temperature,
                        row.top_p,
                        row.ras_temperature,
                        row.ras_top_p,
                    ]
                });
                words[r] = row.top_k;
                words[d.rows + r] = row.step;
                words[2 * d.rows + r] = row.hidden_row;
                seeds[r] = row.seed;
                if let Some((s, uncond_row)) = row.cfg {
                    use plow_asset::fish_depth::{FOLLOW, GUIDE};
                    params[r * 4 + 3] = s;
                    words[r] |= GUIDE;
                    params.copy_within(r * 4..r * 4 + 4, (r + 1) * 4);
                    words[r + 1] = row.top_k | FOLLOW;
                    words[d.rows + r + 1] = row.step;
                    words[2 * d.rows + r + 1] = uncond_row;
                    seeds[r + 1] = row.seed;
                }
            }
            if breeze {
                // One [rows][2] (top_k, step) tensor: interleave into the top_k/step words.
                let (k, s) = words[..2 * d.rows].split_at_mut(d.rows);
                let ctl: Vec<u32> = (0..d.rows).flat_map(|r| [k[r], s[r]]).collect();
                words[..2 * d.rows].copy_from_slice(&ctl);
            }
        }
        let t = d.manifest.tensors.clone();
        let bytes = d.stage.as_slice();
        let (sw, pw, ww) = (d.rows * 8, d.rows * 16, d.rows * 12);
        let words = sw + pw;
        let ctl = if d.manifest.sampler == Sampler::Breeze {
            [(t.top_k, words, n * 8), (t.top_k, words, 0)]
        } else {
            [(t.top_k, words, n * 4), (t.step, words + d.rows * 4, n * 4)]
        };
        for (h, at, len) in [
            (t.seed, 0, n * 8),
            (t.params, sw, n * 16),
            ctl[0],
            ctl[1],
            (t.hidden_row, words + d.rows * 8, n * 4),
        ] {
            if len == 0 {
                continue;
            }
            // SAFETY: the pinned stage is rewritten only after this call's synchronize.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[usize::from(h)].base,
                    &bytes[at..at + len],
                    &self.stream,
                )?;
            }
        }
        self.be
            .memset_d8_async(p.arg.counters, 0, p.counter_bytes, &self.stream)?;
        let mut arg = p.arg;
        if let Some(outputs) = outputs {
            arg.tensors = outputs.tensors;
        }
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        let object = self.depth_object(n);
        self.be.launch_cooperative(
            object.map_or(self.f, |o| o.function),
            object.map_or(self.grid, |o| o.grid),
            object.map_or(BLOCK, |o| o.block),
            object.map_or(self.smem, |o| o.smem),
            &mut params,
            Some(&self.stream),
        )?;
        let codes_at = sw + pw + ww;
        self.depth_read_codes(d, codes_at, n * k * 4)?;
        let codes: &[u32] =
            bytemuck::cast_slice(&d.stage.as_slice()[codes_at..codes_at + n * k * 4]);
        for row in rows {
            out.extend_from_slice(&codes[row.slot * k..row.slot * k + k]);
        }
        Ok(())
    }

    /// Maya 2: cb0 for each row from the host controller over the backbone's logits, then the
    /// depth program samples codebooks `1..` from the row's hidden and cb0. A stopping row yields
    /// `[stop_class, 0, ..]`.
    fn maya2_launch(
        &mut self,
        d: &mut FishDepth,
        rows: &[FishRow],
        outputs: Option<DepthOutputs>,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        use crate::tts::maya2::Tick;
        let width = rows.iter().map(|r| r.slot + 1).max().unwrap_or(0);
        let p = d
            .programs
            .iter()
            .filter(|p| p.rows as usize >= width)
            .min_by_key(|p| p.rows)
            .ok_or_else(|| {
                reject(format!(
                    "no program covers slot {}",
                    width.saturating_sub(1)
                ))
            })?;
        let n = p.rows as usize;
        let k = d.manifest.codebooks as usize;
        let vocab = self.vocab();
        let m = d.maya2.as_mut().unwrap();
        if rows
            .iter()
            .any(|r| m.slots.get(r.slot).is_none_or(Option::is_none))
        {
            return Err(reject("maya2 row on an unbound slot"));
        }
        let l = &self.devp[self.t_logits];
        // SAFETY: the pinned buffer is read only after this call's synchronize.
        unsafe {
            self.be.memcpy_dtoh_async(
                &mut m.logits.as_mut_slice()[..n * vocab * 2],
                outputs.map_or(l.base, |o| o.logits),
                &self.stream,
            )?;
        }
        self.be.stream_synchronize(&self.stream)?;
        let logits: &[u16] = bytemuck::cast_slice(&m.logits.as_slice()[..n * vocab * 2]);
        let mut stops = Vec::with_capacity(rows.len());
        let (seeds, params, ctl, hrow, cb0): (u64, usize, usize, usize, usize) =
            (0, d.rows * 8, d.rows * 24, d.rows * 32, d.rows * 36);
        let _ = seeds;
        {
            let s = depth_stage(&mut d.stage)?;
            for r in 0..n {
                let last = m.slots[r].as_ref().map_or(0, |s| s.last);
                s[ctl + r * 8..ctl + r * 8 + 8].fill(0);
                s[hrow + r * 4..hrow + r * 4 + 4].copy_from_slice(&(r as u32).to_le_bytes());
                s[cb0 + r * 4..cb0 + r * 4 + 4].copy_from_slice(&last.to_le_bytes());
                s[r * 8..r * 8 + 8].fill(0);
                for (j, v) in [m.depth.0, 1.0, 1.0, 1.0].iter().enumerate() {
                    let at = params + r * 16 + j * 4;
                    s[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }

            thread_local! {
                static MAYA2_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }

            let tick_code: Vec<(Tick, u32)> = if rows.len() > 1 {
                let mut active_indices = vec![usize::MAX; m.slots.len()];
                for (idx, row) in rows.iter().enumerate() {
                    active_indices[row.slot] = idx;
                }
                #[cfg(feature = "hf-tokenizer")]
                {
                    use rayon::prelude::*;
                    let by_slot: Vec<Option<(Tick, u32)>> = m
                        .slots
                        .par_iter_mut()
                        .enumerate()
                        .map(|(r, slot_opt)| {
                            let row_idx = active_indices[r];
                            if row_idx == usize::MAX {
                                return None;
                            }
                            let slot = slot_opt.as_mut().unwrap();
                            let tick = MAYA2_SCRATCH.with_borrow_mut(|scratch| {
                                slot.tick(&logits[r * vocab..r * vocab + STOP_ROWS], scratch)
                            });
                            let code = match tick {
                                Tick::Code(c) => c,
                                Tick::Stop => slot.last,
                            };
                            Some((tick, code))
                        })
                        .collect();
                    rows.iter().map(|row| by_slot[row.slot].unwrap()).collect()
                }
                #[cfg(not(feature = "hf-tokenizer"))]
                {
                    rows.iter()
                        .map(|row| {
                            let r = row.slot;
                            let slot = m.slots[r].as_mut().unwrap();
                            let tick = slot
                                .tick(&logits[r * vocab..r * vocab + STOP_ROWS], &mut m.scratch);
                            let code = match tick {
                                Tick::Code(c) => c,
                                Tick::Stop => slot.last,
                            };
                            (tick, code)
                        })
                        .collect()
                }
            } else {
                rows.iter()
                    .map(|row| {
                        let r = row.slot;
                        let slot = m.slots[r].as_mut().unwrap();
                        let tick =
                            slot.tick(&logits[r * vocab..r * vocab + STOP_ROWS], &mut m.scratch);
                        let code = match tick {
                            Tick::Code(c) => c,
                            Tick::Stop => slot.last,
                        };
                        (tick, code)
                    })
                    .collect()
            };

            for (row, (tick, code)) in rows.iter().zip(tick_code) {
                let r = row.slot;
                stops.push(tick == Tick::Stop);
                s[ctl + r * 8..ctl + r * 8 + 4].copy_from_slice(&m.depth.1.to_le_bytes());
                s[ctl + r * 8 + 4..ctl + r * 8 + 8].copy_from_slice(&row.step.to_le_bytes());
                s[hrow + r * 4..hrow + r * 4 + 4].copy_from_slice(&row.hidden_row.to_le_bytes());
                s[cb0 + r * 4..cb0 + r * 4 + 4].copy_from_slice(&code.to_le_bytes());
                s[r * 8..r * 8 + 8].copy_from_slice(&row.seed.to_le_bytes());
            }
        }
        let t = d.manifest.tensors.clone();
        let bytes = d.stage.as_slice();
        for (h, at, len) in [
            (t.seed, 0, n * 8),
            (t.params, params, n * 16),
            (t.top_k, ctl, n * 8),
            (t.hidden_row, hrow, n * 4),
            (t.history, cb0, n * 4),
        ] {
            // SAFETY: the pinned stage is rewritten only after this call's synchronize.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[usize::from(h)].base,
                    &bytes[at..at + len],
                    &self.stream,
                )?;
            }
        }
        self.be
            .memset_d8_async(p.arg.counters, 0, p.counter_bytes, &self.stream)?;
        let mut arg = p.arg;
        if let Some(outputs) = outputs {
            arg.tensors = outputs.tensors;
        }
        let mut args = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        self.be.launch_cooperative(
            m.speech.f,
            self.grid,
            BLOCK,
            m.speech.smem,
            &mut args,
            Some(&self.stream),
        )?;
        let codes_at = d.rows * STAGE_CONTROL_BYTES;
        self.depth_read_codes(d, codes_at, n * k * 4)?;
        let m = d.maya2.as_mut().unwrap();
        let codes: &[u32] =
            bytemuck::cast_slice(&d.stage.as_slice()[codes_at..codes_at + n * k * 4]);
        for (row, &stop) in rows.iter().zip(&stops) {
            let slot = m.slots[row.slot].as_mut().unwrap();
            if stop {
                out.push(d.manifest.stop_class);
                out.extend(std::iter::repeat_n(0, k - 1));
            } else {
                let c = &codes[row.slot * k..row.slot * k + k];
                slot.emitted(c[0]);
                out.extend_from_slice(c);
            }
        }
        Ok(())
    }

    /// Whether a depth stage is loaded.
    pub fn has_depth_stage(&self) -> bool {
        self.fish.is_some()
    }

    /// Whether depth prompts may take packed prefill: the last prompt row is withheld and fed to the
    /// decode step, whose depth launch samples the first frame beside every other row's. Only when
    /// decode embeds that row as prefill does. The mux checks per-request overlays and binds a
    /// host controller before the first depth launch. CFG pairs keep their serial prefill tail.
    pub fn depth_packs(&self) -> bool {
        self.fish.is_some()
    }

    /// Depth packets without a compact output contract keep their withheld prompt tail.
    pub fn pf_compact(&self) -> bool {
        self.has_packed_terminal() && (!self.has_depth_stage() || self.packed_depth_enabled())
    }

    pub fn packed_depth_enabled(&self) -> bool {
        self.fish.as_ref().is_some_and(|d| d.compact.is_some())
    }

    pub fn depth_prompt_tail_can_decode(&self, token: u32) -> bool {
        self.fish
            .as_ref()
            .is_some_and(|d| prompt_token_matches_decode(&d.manifest, token))
    }

    pub(super) fn depth_embedding_token(&self) -> Option<u32> {
        self.fish.as_ref()?.manifest.packed.as_ref()?.decode_token
    }

    pub(super) fn token_input_vocab(&self) -> usize {
        self.fish
            .as_ref()
            .and_then(|d| d.manifest.packed.as_ref())
            .map_or(self.vocab, |p| p.input_vocab as usize)
    }

    pub fn stage_depth_decode_codes(&self, plan: &plow_asset::token_batch::Plan) -> Result<()> {
        let d = self.fish.as_ref().ok_or_else(|| reject("not enabled"))?;
        let pitch = u64::from(d.manifest.codebooks) * 4;
        let t = &d.manifest.tensors;
        for (span, phase) in plan.spans.iter().zip(&plan.phases) {
            match phase {
                plow_asset::token_batch::Phase::Prefill => {}
                plow_asset::token_batch::Phase::Decode if span.n_rows == 1 => {
                    self.be.memcpy_dtod_async(
                        self.devp[t.prompt_codes as usize].base + u64::from(span.row0) * pitch,
                        self.devp[t.frame_codes as usize].base + u64::from(span.slot) * pitch,
                        pitch,
                        &self.stream,
                    )?;
                }
                _ => return Err(reject("speech token batch requires single-row decode")),
            }
        }
        Ok(())
    }

    /// Seed `slot`'s decode frame input (`in.fish.dcodes[slot]`) with a withheld prompt row's
    /// codes (`codebooks` entries; missing = 0) and start its sampling history.
    pub fn fish_seed_slot(&mut self, slot: usize, codes: &[u32]) -> Result<()> {
        self.fish_seed_slots(&[(slot, codes)])
    }

    pub fn fish_seed_slots(&mut self, seeds: &[(usize, &[u32])]) -> Result<()> {
        if seeds.is_empty() {
            return Ok(());
        }
        let mut result = self.fish_seed_slots_inner(seeds);
        if result.is_err() {
            if let Err(error) = self.be.stream_synchronize(&self.stream) {
                self.quarantine_depth_source();
                #[cfg(target_os = "linux")]
                let error = self.be.quarantine_staging(&error);
                result = Err(error);
            }
        }
        self.note_step_result(result.is_err());
        result
    }

    fn fish_seed_slots_inner(&mut self, seeds: &[(usize, &[u32])]) -> Result<()> {
        let d = self.fish.as_mut().ok_or_else(|| reject("not enabled"))?;
        let k = d.manifest.codebooks as usize;
        let at = d.rows * STAGE_CONTROL_BYTES;
        let stage = depth_stage(&mut d.stage)?
            .get_mut(at..)
            .ok_or_else(|| reject("seed staging missing"))?;
        let bytes = seed_frame_codes(stage, seeds, self.batch, k)?;
        let h = usize::from(d.manifest.tensors.frame_codes);
        if self.devp[h].len < (self.batch * k * 4) as u64 {
            return Err(reject("seed tensor smaller than its slots"));
        }
        let stage = &d.stage.as_slice()[at..at + bytes];
        let mut row = 0;
        for run in seeds.chunk_by(|a, b| b.0 == a.0 + 1) {
            let slot = run[0].0;
            let end = row + run.len();
            // SAFETY: validated slot ranges; pinned staging survives errors and is unchanged
            // until every upload completes at the shared stream synchronization.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[h].base + (slot * k * 4) as u64,
                    &stage[row * k * 4..end * k * 4],
                    &self.stream,
                )?;
            }
            row = end;
        }
        self.be.stream_synchronize(&self.stream)?;
        for &(slot, _) in seeds {
            self.fish_reset_slot(slot)?;
        }
        Ok(())
    }

    /// The row of `act.hn` holding the last serial prefill's terminal hidden.
    pub fn prefill_terminal_row(&self) -> u32 {
        self.pf_terminal_row
    }

    /// Write prompt frame codes (`[rows][codebooks]`, row = prompt position) before a prefill.
    pub fn fish_prompt_codes(&mut self, codes: &[u32]) -> Result<()> {
        let d = self.fish.as_ref().ok_or_else(|| reject("not enabled"))?;
        let h = &self.devp[usize::from(d.manifest.tensors.prompt_codes)];
        if codes.len() as u64 * 4 > h.len {
            return Err(reject("prompt codes exceed the context"));
        }
        self.be.upload(h, 0, bytemuck::cast_slice(codes))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        inverse_rows, prompt_token_matches_decode, seed_frame_codes, Manifest, SpeakerMemory,
    };
    use std::sync::Arc;

    #[test]
    fn failed_depth_transfer_retains_all_host_sources_after_drop() {
        use super::{CompactDepth, DeviceMem, FishDepth, Loaded};
        let stage = Arc::new(vec![0u8; 64]);
        let index = Arc::new(vec![0u8; 16]);
        let source = Arc::new(vec![0u32; 4]);
        let owners = [Arc::downgrade(&stage), Arc::downgrade(&index)];
        let frame_source = Arc::downgrade(&source);
        let d = FishDepth {
            manifest: serde_json::from_str(
                r#"{"version":1,"rows":[1],"codebooks":2,"semantic_base":100,
                "semantic_tokens":10,"stop_class":10,"ras_window":0,
                "tensors":{"params":0,"top_k":1,"seed":2,"step":3,"history":4,
                           "hidden_row":5,"frame_codes":6,"prompt_codes":7}}"#,
            )
            .unwrap(),
            programs: Vec::new(),
            stage,
            rows: 1,
            maya2: None,
            compact: Some(CompactDepth {
                remap: Loaded {
                    rows: 1,
                    // SAFETY: DevProgram contains only integers; this fixture never launches it.
                    arg: unsafe { std::mem::zeroed() },
                    counter_bytes: 0,
                    _tables: Vec::new(),
                },
                index: DeviceMem::view(0, 0),
                host_index: index,
                _remap_tensors: DeviceMem::view(0, 0),
                depth_tensors: DeviceMem::view(0, 0),
                logits: None,
            }),
            source: Some(source),
            pool: None,
            retain_output: false,
            captured: None,
        };
        d.quarantine();
        drop(d);
        for owner in owners {
            assert!(owner.upgrade().is_some(), "unretired host source was freed");
        }
        assert!(frame_source.upgrade().is_some());
    }

    #[test]
    fn native_prompt_tail_requires_the_declared_embedding_semantics() {
        let mut m: Manifest = serde_json::from_str(
            r#"{"version":1,"rows":[1],"codebooks":2,"semantic_base":100,
            "semantic_tokens":10,"stop_class":10,"ras_window":0,
            "packed":{"version":1,"input_vocab":120,"backbone_logits":true,
                      "hidden":8,"decode_token":100},
            "tensors":{"params":0,"top_k":1,"seed":2,"step":3,"history":4,
                       "hidden_row":5,"frame_codes":6,"prompt_codes":7}}"#,
        )
        .unwrap();
        for token in [0, 99, 110, 119, 120, u32::MAX] {
            assert!(!prompt_token_matches_decode(&m, token));
        }
        for token in [100, 103, 109] {
            assert!(prompt_token_matches_decode(&m, token));
        }
        m.packed.as_mut().unwrap().decode_token = None;
        for token in [0, 99, 100, 109, 110, 119] {
            assert!(prompt_token_matches_decode(&m, token));
        }
        assert!(!prompt_token_matches_decode(&m, 120));
        m.packed = None;
        assert!(!prompt_token_matches_decode(&m, 100));
    }

    #[test]
    fn compact_depth_remap_preserves_permuted_slots_and_source_rows() {
        let mut index = [0; 5];
        inverse_rows(&mut index, &[3, 0, 2]).unwrap();
        assert_eq!(index, [1, 0, 2, 0, 0]);
        let compact = [[31, 32], [11, 12], [21, 22]];
        let slots = index.map(|r| compact[r as usize]);
        assert_eq!(slots[0], [11, 12]);
        assert_eq!(slots[2], [21, 22]);
        assert_eq!(slots[3], [31, 32]);
        assert_eq!(compact[0], [31, 32]);
        assert!(inverse_rows(&mut index, &[0, 0]).is_err());
        assert!(inverse_rows(&mut index, &[5]).is_err());
        assert!(inverse_rows(&mut index, &[]).is_err());
    }

    #[test]
    fn speaker_memory_retains_identity_and_distinguishes_null_and_uninitialized() {
        let memory: Arc<[f32]> = Arc::from([1.0, 2.0]);
        let same = Arc::clone(&memory);
        let other: Arc<[f32]> = Arc::from([1.0, 2.0]);
        let mut bound = SpeakerMemory::Voice(memory);
        assert!(bound.matches(Some(&same)));
        assert!(!bound.matches(Some(&other)));
        assert!(!bound.matches(None));
        bound = SpeakerMemory::Unbound;
        assert!(!bound.matches(Some(&same)));
        assert!(!bound.matches(None));
        bound = SpeakerMemory::Null;
        assert!(bound.matches(None));
        assert!(!bound.matches(Some(&same)));
    }

    #[test]
    fn frame_seed_staging_preserves_sparse_slot_order_and_zero_pads_short_rows() {
        let mut stage = [0xff; 48];
        let seeds: [(usize, &[u32]); 3] = [(0, &[11, 12]), (3, &[]), (7, &[71])];
        let bytes = seed_frame_codes(&mut stage, &seeds, 8, 2).unwrap();
        assert_eq!(bytes, 24);
        let words: Vec<_> = stage[..bytes]
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words, [11, 12, 0, 0, 71, 0]);
        assert_eq!(stage[bytes..], [0xff; 24]);
        assert_eq!(seed_frame_codes(&mut stage, &[], 8, 2).unwrap(), 0);
    }

    #[test]
    fn frame_seed_staging_rejects_invalid_batches_before_writing() {
        let cases: &[&[(usize, &[u32])]] = &[
            &[(0, &[1, 2, 3])],
            &[(8, &[1])],
            &[(1, &[1]), (1, &[2])],
            &[(2, &[1]), (0, &[2])],
        ];
        for seeds in cases {
            let mut stage = [0xaa; 32];
            assert!(seed_frame_codes(&mut stage, seeds, 8, 2).is_err());
            assert_eq!(stage, [0xaa; 32]);
        }
        let mut stage = [0xaa; 4];
        assert!(seed_frame_codes(&mut stage, &[(0, &[1])], 8, 2).is_err());
        assert_eq!(stage, [0xaa; 4]);
        assert!(seed_frame_codes(&mut stage, &[(0, &[])], 8, 0).is_err());
        assert!(seed_frame_codes(&mut stage, &[], 8, 65).is_err());
    }
}

#[cfg(test)]
#[path = "fish_depth_tests.rs"]
mod packed_tests;
