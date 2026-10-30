//! Fish S2 depth-stage programs (`plow_asset::fish_depth`): after a backbone step the runtime
//! stages each row's sampling inputs, launches the packet's depth program of a width covering the
//! rows' slots on the decode interpreter, and reads each row's frame codes back.

use plow_asset::fish_depth::{Manifest, Sampler};

use crate::asset::devblob::DevBlob;
use crate::device::cuda::{KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::{Result, RuntimeError};

use super::{DevProgram, GpuEngine, BLOCK};

struct Loaded {
    rows: u32,
    arg: DevProgram,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

pub(super) struct FishDepth {
    manifest: Manifest,
    programs: Vec<Loaded>,
    /// Staging for the widest program: seed u64 (first, so aligned), params `[rows][4]` f32, then
    /// top_k, step, hidden_row u32 (Maya 2: `[rows][2]` control, hidden_row, cb0), then the
    /// read-back codes `[rows][codebooks]`.
    stage: PinnedHost,
    rows: usize,
    maya2: Option<Maya2>,
}

/// Maya 2's host cb0 controller: per-slot state, and the widest program's read-back cb0 logits.
struct Maya2 {
    slots: Vec<Option<crate::tts::maya2::Slot>>,
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

impl FishDepth {
    /// The packet's depth stage, `None` when it carries none; read from the engine's own blob.
    pub(super) fn load_if_present(
        e: &GpuEngine,
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
        Self::load(e, manifest, bytes, assets).map(Some)
    }

    fn load(
        e: &GpuEngine,
        manifest: Manifest,
        bytes: &[u8],
        assets: &std::path::Path,
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
            e.be.host_alloc_pinned(rows * (16 + 16 + 8) + e.batch * k * 4)?;
        let maya2 = match (maya2, manifest.depth_sampling) {
            (true, Some([t, k])) => Some(Maya2 {
                slots: vec![None; e.batch],
                logits: e.be.host_alloc_pinned(rows * e.vocab() * 2)?,
                scratch: Vec::new(),
                depth: (f32::from_bits(t), k),
                speech: SpeechFn::load(e, assets, &section.programs)?,
                null_memory: null_memory(e, &manifest)?,
            }),
            _ => None,
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
            stage,
            rows,
            maya2,
        })
    }
}

impl GpuEngine {
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
        memory: Option<&[f32]>,
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
        let mem = memory.unwrap_or(&m.null_memory);
        let layers = d.manifest.slot_memory.len();
        if mem.len() != m.null_memory.len() {
            return Err(reject("speaker memory width disagrees with the packet"));
        }
        let width = mem.len() / layers;
        for (l, &h) in d.manifest.slot_memory.iter().enumerate() {
            let row: &[u8] = bytemuck::cast_slice(&mem[l * width..(l + 1) * width]);
            self.be
                .upload(&self.devp[usize::from(h)], (slot * row.len()) as u64, row)?;
        }
        Ok(())
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
        let mut d = self.fish.take().ok_or_else(|| reject("not enabled"))?;
        let r = if d.maya2.is_some() {
            self.maya2_launch(&mut d, rows, out)
        } else {
            self.fish_launch(&mut d, rows, out)
        };
        self.fish = Some(d);
        r
    }

    fn fish_launch(
        &mut self,
        d: &mut FishDepth,
        rows: &[FishRow],
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
            let s = d.stage.as_mut_slice();
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
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        self.be.launch_cooperative(
            self.f,
            self.grid,
            BLOCK,
            self.smem,
            &mut params,
            Some(&self.stream),
        )?;
        let codes_at = sw + pw + ww;
        // SAFETY: as the uploads.
        unsafe {
            self.be.memcpy_dtoh_async(
                &mut d.stage.as_mut_slice()[codes_at..codes_at + n * k * 4],
                self.devp[usize::from(t.frame_codes)].base,
                &self.stream,
            )?;
        }
        self.be.stream_synchronize(&self.stream)?;
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
                l.base,
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
            let s = d.stage.as_mut_slice();
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
        let mut args = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        self.be.launch_cooperative(
            m.speech.f,
            self.grid,
            BLOCK,
            m.speech.smem,
            &mut args,
            Some(&self.stream),
        )?;
        let codes_at = d.rows * 40;
        // SAFETY: as the uploads.
        unsafe {
            self.be.memcpy_dtoh_async(
                &mut d.stage.as_mut_slice()[codes_at..codes_at + n * k * 4],
                self.devp[usize::from(t.frame_codes)].base,
                &self.stream,
            )?;
        }
        self.be.stream_synchronize(&self.stream)?;
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

    /// Whether a depth stage is loaded: its prompts take serial prefill.
    pub fn has_depth_stage(&self) -> bool {
        self.fish.is_some()
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
