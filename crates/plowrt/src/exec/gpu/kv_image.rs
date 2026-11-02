//! KV image export and import for one sequence slot (docs/arch/30 §5, `exec::kv_image`).

use super::GpuEngine;
use crate::exec::device_api::EngineDevice;
use crate::exec::kv_image::{compat_key, plan};
use crate::{Result, RuntimeError};

impl GpuEngine {
    /// The key an importing engine's image must match; `None` when the packet declares no caches
    /// an image can carry (a TP packet, or one without a live-KV manifest).
    pub fn kv_image_key(&self) -> Option<String> {
        self.kv_image.as_ref().map(compat_key)
    }

    fn kv_image_manifest(&self) -> Result<&plow_asset::live_kv::Manifest> {
        self.kv_image.as_ref().ok_or_else(|| {
            RuntimeError::Rejected("this packet declares no KV image (live-KV manifest)".into())
        })
    }

    /// Slot `b`'s rows `[.., n)` into `out`. `n` is the slot's frontier and nothing was written
    /// past it: a sliding ring then still holds every row the image carries.
    pub fn export_kv_image(&mut self, b: usize, n: u32, out: &mut Vec<u8>) -> Result<()> {
        if b >= self.batch || self.pos[b] != n || self.kv_hw[b] > n {
            return Err(RuntimeError::Rejected(format!(
                "KV image export: slot {b} stands at {} (written to {}), {n} asked",
                self.pos.get(b).copied().unwrap_or(0),
                self.kv_hw.get(b).copied().unwrap_or(0)
            )));
        }
        let (spans, total) = plan(self.kv_image_manifest()?, b as u32, n)?;
        // The rows were written by launches on the engine stream.
        self.be.synchronize()?;
        out.clear();
        out.resize(total as usize, 0);
        for s in &spans {
            let dst = &mut out[s.image_off as usize..(s.image_off + s.bytes) as usize];
            EngineDevice::download(&*self.be, &self.devp[s.handle], s.native_off, dst)?;
        }
        Ok(())
    }

    /// Seat slot `b`, fresh from [`Self::begin_slot`], with `prompt`'s KV from an image another
    /// engine exported. The slot then stands where its own prefill of `prompt` would leave it:
    /// the first generated token is the caller's to feed at position `prompt.len()`.
    pub fn import_kv_image(&mut self, b: usize, prompt: &[u32], image: &[u8]) -> Result<()> {
        let n = u32::try_from(prompt.len())
            .map_err(|_| RuntimeError::Rejected("KV image: prompt too long".into()))?;
        if b >= self.batch || self.pos[b] != 0 {
            return Err(RuntimeError::Rejected(format!(
                "KV image import: slot {b} is not freshly begun"
            )));
        }
        if prompt.len() >= self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "KV image import: {} rows leave no room under the context {}",
                prompt.len(),
                self.max_ctx
            )));
        }
        // State the image does not carry: recurrent layers and a block drafter's context caches.
        if self.kv_blocks().is_some_and(|bs| bs.captures_state())
            || self.recurrent.is_some()
            || self.spec_draft.is_some()
        {
            return Err(RuntimeError::Rejected(
                "KV image import: packets with recurrent state or a drafter are not supported"
                    .into(),
            ));
        }
        let (spans, total) = plan(self.kv_image_manifest()?, b as u32, n)?;
        if total != image.len() as u64 {
            return Err(RuntimeError::Rejected(format!(
                "KV image import: image is {} bytes, this packet's is {total}",
                image.len()
            )));
        }
        if let Some(v) = &self.vmm {
            v.ensure_rows(b, n)?;
        }
        for s in &spans {
            let src = &image[s.image_off as usize..(s.image_off + s.bytes) as usize];
            EngineDevice::upload(&*self.be, &self.devp[s.handle], s.native_off, src)?;
        }
        self.be.synchronize()?;
        self.pos[b] = n;
        self.kv_hw[b] = n;
        // An imported slot publishes nothing to this engine's prefix cache: its rings hold only
        // the image's window, not the rows an earlier boundary would copy, and the exporting
        // engine already holds the prompt's blocks. `seq_tokens` shorter than the frontier is
        // what every publish path refuses on.
        self.seq_tokens[b].clear();
        Ok(())
    }
}
