//! Fish S2 depth-stage programs (`plow_asset::fish_depth`): after a backbone step the runtime
//! stages each row's sampling inputs, launches the packet's depth program of a width covering the
//! rows' slots on the decode interpreter, and reads each row's frame codes back.

use std::path::Path;

use plow_asset::fish_depth::{Manifest, Sampler};

use crate::asset::devblob::DevBlob;
use crate::device::cuda::PinnedHost;
use crate::device::{Backend, DeviceMem};
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
    /// top_k, step, hidden_row u32, then the read-back codes `[rows][codebooks]`.
    stage: PinnedHost,
    rows: usize,
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

fn reject(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("fish depth: {e}"))
}

impl FishDepth {
    /// The packet's depth stage, `None` when it carries none.
    pub(super) fn load_if_present(e: &GpuEngine, assets: &Path) -> Result<Option<Self>> {
        let Some(pkt) = DevBlob::find_in_dir(assets)? else {
            return Ok(None);
        };
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
            path: pkt.clone(),
            source,
        })?;
        let blob = DevBlob::parse(&raw)?;
        if blob
            .reserved_metadata(&raw, plow_asset::fish_depth::SECTION)?
            .is_none()
        {
            return Ok(None);
        }
        Self::load(e, assets).map(Some)
    }

    fn load(e: &GpuEngine, assets: &Path) -> Result<Self> {
        let pkt = DevBlob::find_in_dir(assets)?
            .ok_or_else(|| reject(format!("no packet in {}", assets.display())))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
            path: pkt.clone(),
            source,
        })?;
        let blob = DevBlob::parse(&raw)?;
        let meta = blob
            .reserved_metadata(&raw, plow_asset::fish_depth::SECTION)?
            .ok_or_else(|| reject("packet carries no depth stage"))?;
        let manifest: Manifest = serde_json::from_slice(meta).map_err(reject)?;
        let bytes = blob
            .reserved_metadata(&raw, plow_asset::fish_depth::PROGRAMS_SECTION)?
            .ok_or_else(|| reject("depth programs section missing"))?;
        if blob.n_cu != e.grid
            || blob.tensors.len() != e.devp.len()
            || blob
                .tensors
                .iter()
                .zip(e.tensor_names.iter().zip(&e.devp))
                .any(|(t, (name, mem))| t.name != *name || mem.len < t.bytes)
        {
            return Err(reject("packet does not match the loaded engine"));
        }
        let section =
            plow_asset::aux_program::parse(bytes, e.grid, e.devp.len()).map_err(reject)?;
        manifest
            .validate(section.programs.len(), e.devp.len())
            .map_err(reject)?;
        let rows = manifest.rows.iter().copied().max().unwrap_or(0) as usize;
        let t = &manifest.tensors;
        let k = manifest.codebooks as usize;
        let need = |h: u16, bytes: usize| e.devp[usize::from(h)].len >= bytes as u64;
        if rows > e.batch
            || !need(t.params, rows * 16)
            || !need(t.seed, rows * 8)
            || !need(t.history, e.batch * manifest.ras_window as usize * 4)
            || !need(t.frame_codes, e.batch * k * 4)
            || [t.top_k, t.step, t.hidden_row]
                .iter()
                .any(|&h| !need(h, rows * 4))
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
            e.be.host_alloc_pinned(rows * (16 + 12 + 8) + e.batch * k * 4)?;
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
        })
    }
}

impl GpuEngine {
    /// Load the packet's depth stage (a Fish S2 packet).
    pub fn fish_depth_enable(&mut self, assets: &Path) -> Result<FishInfo> {
        let d = FishDepth::load(self, assets)?;
        let m = &d.manifest;
        let info = FishInfo {
            codebooks: m.codebooks as usize,
            semantic_base: m.semantic_base,
            semantic_tokens: m.semantic_tokens,
            stop_class: m.stop_class,
            pair_cfg: m.pair_cfg,
        };
        self.fish = Some(d);
        Ok(info)
    }

    /// Start a request on `slot`: an empty RAS history.
    pub fn fish_reset_slot(&mut self, slot: usize) -> Result<()> {
        let d = self.fish.as_ref().ok_or_else(|| reject("not enabled"))?;
        let w = d.manifest.ras_window as usize * 4;
        let h = &self.devp[usize::from(d.manifest.tensors.history)];
        self.be
            .memset_d8_async(h.base + (slot * w) as u64, 0, w, &self.stream)
    }

    /// A prompt's first frame samples from prefill's logits (row 0): move them to `slot`'s row.
    pub fn fish_logits_to_slot(&mut self, slot: usize) -> Result<()> {
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
        let r = self.fish_launch(&mut d, rows, out);
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
