//! KV image export and import for one sequence slot (docs/arch/30 §5, `exec::kv_image`).

use super::GpuEngine;
use crate::exec::device_api::EngineDevice;
use crate::exec::kv_image::plan;
use crate::{Result, RuntimeError};

impl GpuEngine {
    #[cfg(target_os = "linux")]
    fn transfer_kv_image(
        &mut self,
        image: &crate::exec::kv_image::KvBytes,
        spans: &[crate::exec::kv_image::ImageSpan],
        export: bool,
    ) -> Result<bool> {
        match image.transfer(&*self.be, spans, &self.devp, export) {
            Ok(staged) => Ok(staged),
            Err(failure) if failure.unretired => {
                self.quarantine_depth_source();
                let mut empty = crate::memory::vmm::WeightSlab::PerTensor;
                std::mem::forget(std::mem::replace(
                    &mut self._weight_slab,
                    std::sync::Arc::new(crate::memory::vmm::WeightSlab::PerTensor),
                ));
                image.quarantine(&mut self.devp, &mut empty, &mut self.vmm, self.be.clone());
                self.kv_image_key = None;
                Err(self.be.quarantine_staging(&failure.error))
            }
            Err(failure) => Err(failure.error),
        }
    }

    /// The key an imported image must carry: the digest of the caches' semantic header
    /// (`plow_asset::kv_image`). `None` when the packet's caches have no header (a TP packet, no
    /// live-KV manifest, or a cache the header does not recognise).
    pub fn kv_image_key(&self) -> Option<String> {
        self.kv_image_key.clone()
    }

    fn kv_image_manifest(&self) -> Result<&plow_asset::live_kv::Manifest> {
        self.kv_image.as_ref().ok_or_else(|| {
            RuntimeError::Rejected("this packet declares no KV image (live-KV manifest)".into())
        })
    }

    /// Slot `b`'s rows `[.., n)` into `out`. `n` is the slot's frontier and the slot has not
    /// decoded (a P/D prefill job's slot, right after its first token): a sliding ring then still
    /// holds every row the image carries.
    pub fn export_kv_image(
        &mut self,
        b: usize,
        n: u32,
        out: &mut crate::serve::pd::KvBytes,
    ) -> Result<()> {
        self.export_kv_image_with(b, n, out, None)
    }

    pub fn export_kv_image_stream(
        &mut self,
        b: usize,
        n: u32,
        out: &mut crate::serve::pd::KvBytes,
        publish: &mut dyn FnMut(crate::serve::pd::KvBytes),
    ) -> Result<()> {
        self.export_kv_image_with(b, n, out, Some(publish))
    }

    fn export_kv_image_with(
        &mut self,
        b: usize,
        n: u32,
        out: &mut crate::serve::pd::KvBytes,
        publish: Option<&mut dyn FnMut(crate::serve::pd::KvBytes)>,
    ) -> Result<()> {
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
        out.prepare(total as usize, hwspec::Vendor::Nvidia);
        out.export_with(publish, |out| {
            #[cfg(target_os = "linux")]
            if self.transfer_kv_image(out, &spans, true)? {
                return Ok(());
            }
            let crate::serve::pd::KvBytes::Pageable(out) = out else {
                return Err(RuntimeError::Device(
                    "KV image: unregistered export slab".into(),
                ));
            };
            for s in &spans {
                let dst = &mut out[s.image_off as usize..(s.image_off + s.bytes) as usize];
                EngineDevice::download(&*self.be, &self.devp[s.handle], s.native_off, dst)?;
            }
            Ok(())
        })
    }

    pub fn prepare_kv_import(&mut self, b: usize, prompt_rows: usize, total: usize) -> Result<()> {
        let started = std::time::Instant::now();
        // Inactive decode rows may still write padding into this slot's old mappings.
        self.be.synchronize()?;
        self.begin_slot(b, total)?;
        let n = self.check_kv_import_slot(b, prompt_rows)?;
        if let Some(v) = &self.vmm {
            v.ensure_rows(b, n)?;
        }
        self.be.prepare_staging()?;
        tracing::debug!(
            slot = b,
            rows = n,
            prepare_ms = started.elapsed().as_secs_f64() * 1e3,
            "KV image import prepared"
        );
        Ok(())
    }

    fn check_kv_import_slot(&self, b: usize, prompt_rows: usize) -> Result<u32> {
        let n = u32::try_from(prompt_rows)
            .map_err(|_| RuntimeError::Rejected("KV image: prompt too long".into()))?;
        if b >= self.batch || self.pos[b] != 0 {
            return Err(RuntimeError::Rejected(format!(
                "KV image import: slot {b} is not freshly begun"
            )));
        }
        if prompt_rows >= self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "KV image import: {} rows leave no room under the context {}",
                prompt_rows, self.max_ctx
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
        self.kv_image_manifest()?;
        Ok(n)
    }

    /// Seat slot `b`, fresh from [`Self::begin_slot`], with `prompt`'s KV from an image another
    /// engine exported. The slot then stands where its own prefill of `prompt` would leave it:
    /// the first generated token is the caller's to feed at position `prompt.len()`.
    pub fn import_kv_image(
        &mut self,
        b: usize,
        prompt: &[u32],
        image: &crate::serve::pd::KvBytes,
    ) -> Result<()> {
        let n = self.check_kv_import_slot(b, prompt.len())?;
        let (spans, total) = plan(self.kv_image_manifest()?, b as u32, n)?;
        if total != image.len() as u64 {
            return Err(RuntimeError::Rejected(format!(
                "KV image import: image is {} bytes, this packet's is {total}",
                image.len()
            )));
        }
        if let Some(v) = &self.vmm {
            let timing = tracing::enabled!(tracing::Level::DEBUG).then(std::time::Instant::now);
            v.ensure_rows(b, n)?;
            if let Some(t) = timing {
                tracing::debug!(
                    rows = n,
                    mapping_ms = t.elapsed().as_secs_f64() * 1e3,
                    "KV image import mapping"
                );
            }
        }
        // The staging stream must not race earlier writes to this slot.
        self.be.synchronize()?;
        #[cfg(target_os = "linux")]
        let staged = self.transfer_kv_image(image, &spans, false)?;
        #[cfg(not(target_os = "linux"))]
        let staged = false;
        if !staged {
            for s in &spans {
                let src = &image[s.image_off as usize..(s.image_off + s.bytes) as usize];
                EngineDevice::upload(&*self.be, &self.devp[s.handle], s.native_off, src)?;
            }
            self.be.synchronize()?;
        }
        self.pos[b] = n;
        self.kv_hw[b] = n;
        // The history stays complete (decode consumers check it against the frontier), but the
        // slot publishes nothing to this engine's prefix cache: its rings hold only the image's
        // window, and the exporting engine already holds the prompt's blocks.
        self.seq_tokens[b].clear();
        self.seq_tokens[b].extend_from_slice(prompt);
        self.publish_off[b] = true;
        Ok(())
    }
}
