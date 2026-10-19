//! Packet draft programs for speculative decode (`plow_asset::spec_draft`).
//!
//! The packet owns the drafter: the runtime stages each sequence's input token, the verify row
//! holding the hidden that produced it, its committed length and slot, launches the packet's
//! `(rows, k)` program on the decode interpreter and reads back `k` tokens per sequence.

use std::path::Path;

use plow_asset::spec_draft::Manifest;

use crate::asset::devblob::DevBlob;
use crate::device::cuda::PinnedHost;
use crate::device::DeviceMem;
use crate::{Result, RuntimeError};

use super::{CudaBackend, DevProgram, GpuEngine, BLOCK};

struct Loaded {
    arg: DevProgram,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

pub(super) struct SpecDraft {
    manifest: Manifest,
    programs: Vec<Loaded>,
    /// `[kmax + 1 tokens, hidden_row, position, kv_len, slot][rows]` i32 staging.
    stage: PinnedHost,
    rows: usize,
    selected: Option<(u32, usize, usize)>,
}

fn reject(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("spec draft: {e}"))
}

impl SpecDraft {
    fn load(e: &GpuEngine, assets: &Path) -> Result<Self> {
        let pkt = DevBlob::find_in_dir(assets)?
            .ok_or_else(|| reject(format!("no packet in {}", assets.display())))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
            path: pkt.clone(),
            source,
        })?;
        let blob = DevBlob::parse(&raw)?;
        let meta = blob
            .reserved_metadata(&raw, plow_asset::spec_draft::SECTION)?
            .ok_or_else(|| reject("packet was not emitted with PLOW_EMIT_SPEC"))?;
        let manifest: Manifest = serde_json::from_slice(meta).map_err(reject)?;
        let bytes = blob
            .reserved_metadata(&raw, plow_asset::spec_draft::PROGRAMS_SECTION)?
            .ok_or_else(|| reject("draft programs section missing"))?;
        if blob.tensors.len() != e.devp.len() || blob.n_cu != e.grid {
            return Err(reject("packet does not match the loaded engine"));
        }
        let section =
            plow_asset::aux_program::parse(bytes, e.grid, e.devp.len()).map_err(reject)?;
        manifest
            .validate(section.programs.len(), e.devp.len())
            .map_err(reject)?;
        let rows = manifest.programs.iter().map(|p| p.rows).max().unwrap_or(0) as usize;
        let t = &manifest.tensors;
        for &h in t
            .tokens
            .iter()
            .chain([t.hidden_row, t.position, t.kv_len, t.slot].iter())
        {
            if e.devp[usize::from(h)].len < rows as u64 * 4 {
                return Err(reject(format!("tensor {h} holds fewer than {rows} rows")));
            }
        }
        let programs = section
            .programs
            .iter()
            .map(|p| {
                let (arg, _, counter_bytes, tables) =
                    super::mixed_step::upload_program(&e.be, p, e.d_tens.base, 0, 0)?;
                Ok(Loaded {
                    arg,
                    counter_bytes,
                    _tables: tables,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stage =
            e.be.host_alloc_pinned((manifest.kmax as usize + 5) * rows * 4)?;
        tracing::info!(
            programs = programs.len(),
            kmax = manifest.kmax,
            rows,
            family = %manifest.family,
            "spec draft programs loaded"
        );
        Ok(Self {
            manifest,
            programs,
            stage,
            rows,
            selected: None,
        })
    }

    fn launch(
        &self,
        be: &CudaBackend,
        f: super::KernelFn,
        grid: u32,
        smem: u32,
        stream: &crate::device::cuda::CudaStream,
        index: usize,
    ) -> Result<()> {
        let p = &self.programs[index];
        be.memset_d8_async(p.arg.counters, 0, p.counter_bytes, stream)?;
        let mut arg = p.arg;
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        be.launch_cooperative(f, grid, BLOCK, smem, &mut params, Some(stream))
    }
}

impl GpuEngine {
    /// Load the packet's draft programs. Returns the deepest draft `k` they offer.
    pub fn spec_draft_enable(&mut self, assets: &Path) -> Result<u32> {
        let d = SpecDraft::load(self, assets)?;
        let kmax = d.manifest.kmax;
        self.spec_draft = Some(d);
        Ok(kmax)
    }

    /// Stage `(slot, token, hidden_row)` per sequence and select the `(rows, k)` program.
    /// `hidden_row` is the row of the packet's hidden activation that produced `token` in the
    /// last decode or verify launch; the draft reads the slot's committed KV only.
    pub fn spec_draft_stage(&mut self, rows: &[(usize, u32, u32)], k: u32) -> Result<()> {
        let vocab = self.vocab() as u32;
        let batch = self.batch;
        let pos: Vec<u32> = rows
            .iter()
            .map(|&(s, _, _)| self.pos.get(s).copied().unwrap_or(0) as u32)
            .collect();
        let d = self
            .spec_draft
            .as_mut()
            .ok_or_else(|| reject("not enabled"))?;
        let p = *d
            .manifest
            .select(rows.len(), k)
            .ok_or_else(|| reject(format!("no program for {} rows at k {k}", rows.len())))?;
        for (&(slot, token, hrow), &pos) in rows.iter().zip(&pos) {
            if slot >= batch || token >= vocab || hrow >= d.manifest.tensors.hidden_rows || pos == 0
            {
                return Err(reject(format!(
                    "row slot {slot} token {token} hidden row {hrow} pos {pos}"
                )));
            }
        }
        let n = d.rows;
        let kmax = d.manifest.kmax as usize;
        let words: &mut [i32] = bytemuck::cast_slice_mut(d.stage.as_mut_slice());
        for r in 0..p.rows as usize {
            let (slot, token, hrow) = rows[r.min(rows.len() - 1)];
            let pos = pos[r.min(rows.len() - 1)] as i32;
            words[r] = token as i32;
            words[(kmax + 1) * n + r] = hrow as i32;
            words[(kmax + 2) * n + r] = pos - 1;
            words[(kmax + 3) * n + r] = pos;
            words[(kmax + 4) * n + r] = slot as i32;
        }
        let t = d.manifest.tensors.clone();
        let bytes = d.stage.as_slice();
        for (i, h) in [
            (0, t.tokens[0]),
            (kmax + 1, t.hidden_row),
            (kmax + 2, t.position),
            (kmax + 3, t.kv_len),
            (kmax + 4, t.slot),
        ] {
            // SAFETY: the pinned stage outlives the stream synchronize in `spec_draft_run`.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[usize::from(h)].base,
                    &bytes[i * n * 4..(i * n + p.rows as usize) * 4],
                    &self.stream,
                )?;
            }
        }
        d.selected = Some((k, p.index as usize, rows.len()));
        Ok(())
    }

    /// Launch the staged draft and read `k` tokens per staged row into `out` (row-major).
    pub fn spec_draft_run(&mut self, out: &mut Vec<u32>) -> Result<()> {
        self.spec_draft_launch()?;
        let d = self.spec_draft.as_mut().expect("staged");
        let (k, _, live) = d.selected.expect("staged");
        let n = d.rows;
        for j in 1..=k as usize {
            let h = d.manifest.tensors.tokens[j];
            // SAFETY: as the uploads.
            unsafe {
                self.be.memcpy_dtoh_async(
                    &mut d.stage.as_mut_slice()[j * n * 4..(j * n + live) * 4],
                    self.devp[usize::from(h)].base,
                    &self.stream,
                )?;
            }
        }
        self.be.stream_synchronize(&self.stream)?;
        let words: &[u32] = bytemuck::cast_slice(d.stage.as_slice());
        out.clear();
        for r in 0..live {
            out.extend((1..=k as usize).map(|j| words[j * n + r]));
        }
        Ok(())
    }

    /// Launch the staged draft program without reading it back (bench timing).
    pub fn spec_draft_launch(&mut self) -> Result<()> {
        let d = self
            .spec_draft
            .as_ref()
            .ok_or_else(|| reject("not enabled"))?;
        let (_, index, _) = d.selected.ok_or_else(|| reject("nothing staged"))?;
        d.launch(&self.be, self.f, self.grid, self.smem, &self.stream, index)
    }

    pub fn spec_draft_sync(&self) -> Result<()> {
        self.be.stream_synchronize(&self.stream)
    }
}
