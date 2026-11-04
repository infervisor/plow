//! Prefill/decode disaggregation inside one serve (docs/arch/30 §7).
//!
//! A `--pd` model's DP ranks split into a prefill pool and a decode pool. An eligible request
//! prefills on a prefill rank as a one-token job; when that slot finishes, its dispatcher exports
//! the slot's KV image before the slot is retired. The request then continues on a decode rank,
//! seated from the image with the first token already sampled: that rank streams it, then
//! decodes as if it had prefilled the prompt itself.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use std::time::Instant;

use crate::obs::Metrics;
use crate::serve::dp::{DpSet, MAX_DP};
use crate::serve::mux::{IngressGuard, Job, JobOpts, ModelMux, SubmitError};
use crate::serve::stream::{self, FinishReason, StreamChunk};
use crate::serve::AppState;
use crate::RuntimeError;

pub use crate::exec::kv_image::KvBytes;

/// One slot's canonical KV image.
pub struct KvImage {
    /// The exporting engine's key: its caches' semantic header.
    pub key: String,
    pub rows: u32,
    pub bytes: KvBytes,
    pub handoff_started: Instant,
}

impl KvImage {
    pub(crate) fn handoff_ms(&self) -> f64 {
        let pd_handoff_ms = self.handoff_started.elapsed().as_secs_f64() * 1e3;
        tracing::debug!(
            pd_handoff_ms,
            rows = self.rows,
            bytes = self.bytes.len(),
            "pd: handoff complete"
        );
        pd_handoff_ms
    }
}

/// Where a prefill job's image goes when its slot finishes.
pub struct KvExport {
    tx: Mutex<Option<oneshot::Sender<crate::Result<KvImage>>>>,
    outcome: Arc<Outcome>,
}

impl KvExport {
    pub fn new(outcome: Arc<Outcome>) -> (Arc<Self>, oneshot::Receiver<crate::Result<KvImage>>) {
        let (tx, rx) = oneshot::channel();
        (
            Arc::new(Self {
                tx: Mutex::new(Some(tx)),
                outcome,
            }),
            rx,
        )
    }

    /// Whether a receiver still waits (an export nobody takes is skipped).
    pub fn wanted(&self) -> bool {
        self.tx.lock().as_ref().is_some_and(|tx| !tx.is_closed())
    }

    pub fn deliver(&self, image: crate::Result<KvImage>) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(image);
        }
    }

    pub(super) fn produce(
        &self,
        key: Option<String>,
        rows: u32,
        handoff_started: Instant,
        copy: impl FnOnce(&mut KvBytes, &mut dyn FnMut(KvBytes)) -> crate::Result<()>,
    ) -> crate::Result<()> {
        let Some(key) = key else {
            let error = RuntimeError::Rejected("P/D: this packet has no KV image".into());
            self.deliver(Err(crate::serve::mux::fanout_err(
                &error,
                &error.to_string(),
            )));
            return Err(error);
        };
        let mut bytes = KvBytes::default();
        let result = copy(&mut bytes, &mut |ready| {
            self.deliver(Ok(KvImage {
                key: key.clone(),
                rows,
                bytes: ready,
                handoff_started,
            }))
        });
        if let Err(error) = &result {
            self.deliver(Err(crate::serve::mux::fanout_err(
                error,
                &error.to_string(),
            )));
        } else {
            self.deliver(Ok(KvImage {
                key,
                rows,
                bytes,
                handoff_started,
            }));
        }
        result
    }
}

pub struct ReadyImport {
    pub image: KvImage,
    pub token: u32,
    pub logprobs: Option<Box<crate::text::logprobs::TokenLogprobs>>,
    pub first: Instant,
    /// Prompt rows the prefill rank served from its prefix cache (`usage.cached_tokens`).
    pub cached_tokens: usize,
}

enum ImportState {
    Pending(oneshot::Receiver<ReadyImport>),
    Ready(ReadyImport),
    Abandoned,
}

/// A decode reservation can prepare its mappings while its prefill rank is still working.
pub struct KvImport {
    state: ImportState,
    pub arrived: Instant,
    pub outcome: Arc<Outcome>,
    pub(crate) prepared: bool,
    pub(crate) metrics: Option<Arc<Metrics>>,
}

impl KvImport {
    fn new(state: ImportState, arrived: Instant, outcome: Arc<Outcome>) -> Self {
        Self {
            state,
            arrived,
            outcome,
            prepared: false,
            metrics: None,
        }
    }

    pub(crate) fn pending(
        arrived: Instant,
        outcome: Arc<Outcome>,
    ) -> (Self, oneshot::Sender<ReadyImport>) {
        let (tx, rx) = oneshot::channel();
        (Self::new(ImportState::Pending(rx), arrived, outcome), tx)
    }

    #[cfg(test)]
    pub(crate) fn ready(payload: ReadyImport, arrived: Instant, outcome: Arc<Outcome>) -> Self {
        Self::new(ImportState::Ready(payload), arrived, outcome)
    }

    pub(crate) fn poll(&mut self) {
        if let ImportState::Pending(rx) = &mut self.state {
            match rx.try_recv() {
                Ok(payload) => self.state = ImportState::Ready(payload),
                Err(oneshot::error::TryRecvError::Empty) => {}
                Err(oneshot::error::TryRecvError::Closed) => self.state = ImportState::Abandoned,
            }
        }
    }

    pub(crate) fn waiting(&self) -> bool {
        matches!(self.state, ImportState::Pending(_))
    }

    pub(crate) fn prepare_with(
        &mut self,
        prepare: impl FnOnce() -> crate::Result<()>,
    ) -> crate::Result<bool> {
        if !self.waiting() || self.prepared {
            return Ok(false);
        }
        prepare()?;
        self.prepared = true;
        Ok(true)
    }

    pub(crate) fn abandoned(&self) -> bool {
        matches!(self.state, ImportState::Abandoned)
    }

    pub(crate) fn preempt(&mut self) -> bool {
        if let Some(metrics) = self.metrics.take() {
            if self.outcome.claim_terminal() {
                if self.outcome.settle() {
                    Metrics::inc(&metrics.serving.preempted);
                }
                return true;
            }
        }
        false
    }

    pub(crate) fn payload(&self) -> Option<&ReadyImport> {
        match &self.state {
            ImportState::Ready(payload) => Some(payload),
            _ => None,
        }
    }

    pub(crate) fn take_ready(mut self) -> ReadyImport {
        match std::mem::replace(&mut self.state, ImportState::Abandoned) {
            ImportState::Ready(payload) => payload,
            _ => unreachable!("import has no payload"),
        }
    }

    pub(crate) fn start_metrics(
        &mut self,
        prompt: usize,
        max_tokens: usize,
    ) -> Option<crate::obs::serving::RequestMetrics> {
        let first = self.payload()?.first;
        let metrics = self.metrics.take()?;
        self.outcome.settle();
        Some(crate::obs::serving::RequestMetrics::leg(
            metrics,
            crate::obs::serving::Leg::Decode { first },
            self.arrived,
            self.arrived,
            prompt,
            max_tokens,
        ))
    }
}

/// A P/D request's terminal outcome until its decode leg owns it. Settled by whoever counts it:
/// the decode leg's telemetry, a rank's refusal of either leg, a failed handoff, a preempted
/// prefill. The prefill leg's telemetry counts none; dropped unsettled (a cancellation, a failed
/// prefill, a job never seated) it counts one abort on the prefill rank.
pub struct Outcome {
    metrics: Arc<Metrics>,
    arrived: Instant,
    open: AtomicBool,
    terminal: AtomicBool,
}

impl Outcome {
    pub fn new(metrics: Arc<Metrics>, arrived: Instant) -> Arc<Self> {
        Arc::new(Self {
            metrics,
            arrived,
            open: AtomicBool::new(true),
            terminal: AtomicBool::new(false),
        })
    }

    pub fn settle(&self) -> bool {
        self.open.swap(false, Ordering::Relaxed)
    }

    fn claim_terminal(&self) -> bool {
        !self.terminal.swap(true, Ordering::Relaxed)
    }

    fn count_abort(&self) {
        let m = &self.metrics.serving;
        Metrics::inc(&m.aborted);
        Metrics::inc(&m.aborted_before_first);
        m.aborted_elapsed.duration(self.arrived.elapsed());
    }

    pub(crate) fn report_abort(&self) -> bool {
        if !self.claim_terminal() {
            return false;
        }
        if self.settle() {
            self.count_abort();
        }
        true
    }
}

impl Drop for Outcome {
    fn drop(&mut self) {
        if *self.open.get_mut() {
            self.count_abort();
        }
    }
}

/// A rank counted `job`'s end (a refusal or a queue expiry): a P/D leg's request is settled.
pub(crate) fn settle_refused(job: &Job) {
    if let Some(x) = &job.opts.export {
        x.outcome.settle();
    }
    if let Some(i) = &job.opts.import {
        i.outcome.settle();
    }
}

pub(crate) fn refuse(job: &Job, counter: &std::sync::atomic::AtomicU64) -> bool {
    let outcome = job
        .opts
        .import
        .as_ref()
        .map(|i| &i.outcome)
        .or_else(|| job.opts.export.as_ref().map(|e| &e.outcome));
    if let Some(outcome) = outcome {
        let counted = outcome.settle();
        if counted {
            Metrics::inc(counter);
        }
        if job.opts.import.is_some() {
            outcome.claim_terminal()
        } else {
            counted
        }
    } else {
        Metrics::inc(counter);
        true
    }
}

/// The KV image header of the bundle in `dir`, its checkpoint hashed once per checkpoint
/// directory (`digests`).
pub fn kv_header(
    dir: &std::path::Path,
    digests: &mut std::collections::HashMap<std::path::PathBuf, String>,
) -> Result<plow_asset::kv_image::Header, String> {
    let (blob, manifest) = kv_packet(dir)?;
    let ckpt = crate::asset::serve::checkpoint_dir(dir);
    let ckpt = ckpt.canonicalize().unwrap_or(ckpt);
    let weights = match digests.get(&ckpt) {
        Some(d) => d.clone(),
        None => {
            let d = plow_asset::kv_image::checkpoint_digest(&ckpt)?;
            digests.insert(ckpt, d.clone());
            d
        }
    };
    blob.with_packet_view(|p| plow_asset::kv_image::derive(p, &manifest, weights))
}

fn kv_packet(
    dir: &std::path::Path,
) -> Result<
    (
        crate::asset::devblob::DevBlob,
        plow_asset::live_kv::Manifest,
    ),
    String,
> {
    use crate::asset::devblob::DevBlob;
    let pkt = DevBlob::find_in_dir(dir)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no packet in {}", dir.display()))?;
    let raw = std::fs::read(&pkt).map_err(|e| format!("{}: {e}", pkt.display()))?;
    let blob = DevBlob::parse_l2(&raw, true).map_err(|e| e.to_string())?;
    let manifest =
        match crate::memory::vmm::LiveKvLayout::manifest(&blob, &raw).map_err(|e| e.to_string())? {
            Some(m) => m,
            None => blob.with_packet_view(plow_asset::live_kv::emit)?,
        };
    Ok((blob, manifest))
}

pub fn kv_image_capacity(dir: &std::path::Path, rows: Option<u32>) -> Result<usize, String> {
    let (_, manifest) = kv_packet(dir)?;
    let (_, bytes) = crate::exec::kv_image::plan(&manifest, 0, rows.unwrap_or(manifest.max_ctx))
        .map_err(|e| e.to_string())?;
    usize::try_from(bytes).map_err(|e| e.to_string())
}

/// `--pd` startup check: every rank's KV image header (`headers[r]`, rank `r`'s bundle) equals
/// the first prefill rank's, or the model is refused naming the first difference.
pub fn check_headers(
    model: &str,
    spec: PdSpec,
    headers: &[Result<plow_asset::kv_image::Header, String>],
) -> Result<(), String> {
    let in_pd = |r: usize| (spec.prefill | spec.decode) >> r & 1 != 0;
    let first = spec.prefill.trailing_zeros() as usize;
    let base = headers[first]
        .as_ref()
        .map_err(|e| format!("--pd {model}: rank {first}: {e}"))?;
    for (r, h) in headers.iter().enumerate().filter(|(r, _)| in_pd(*r)) {
        let h = h
            .as_ref()
            .map_err(|e| format!("--pd {model}: rank {r}: {e}"))?;
        if let Some(diff) = base.first_difference(h) {
            return Err(format!(
                "--pd {model}: rank {r}'s KV cache differs from rank {first}'s at {diff}"
            ));
        }
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
pub fn check_engines(state: &AppState, model: &str, spec: PdSpec) -> Result<(), String> {
    let set = state
        .dp_set(model)
        .ok_or_else(|| format!("--pd {model}: no DP set"))?;
    let keys: Vec<_> = set
        .ranks
        .iter()
        .map(|r| {
            state
                .gpu_engine(&r.key)
                .and_then(|e| e.lock().kv_image_key())
        })
        .collect();
    check_engine_keys(model, spec, &keys)
}

fn check_engine_keys(model: &str, spec: PdSpec, keys: &[Option<String>]) -> Result<(), String> {
    let first = spec.prefill.trailing_zeros() as usize;
    let key =
        |rank: usize| {
            keys.get(rank).and_then(Option::as_ref).ok_or_else(|| {
        format!("--pd {model}: rank {rank}'s engine has no KV image (unsupported cache storage)")
    })
        };
    let base = key(first)?;
    for rank in 0..MAX_DP {
        if (spec.prefill | spec.decode) >> rank & 1 != 0 && key(rank)? != base {
            return Err(format!(
                "--pd {model}: rank {rank}'s engine KV image key differs from rank {first}'s"
            ));
        }
    }
    Ok(())
}

/// `--pd model=P/D`: the DP ranks of each pool, as bit masks over the model's ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PdSpec {
    pub prefill: u32,
    pub decode: u32,
}

/// Parse `--pd gemma=0/1` or `--pd gemma=0+1/2+3` (rank indices of the model's `--dp` set).
pub fn parse_pd(entries: &[String]) -> Result<Vec<(String, PdSpec)>, String> {
    let ranks = |list: &str, entry: &str| -> Result<u32, String> {
        let mut mask = 0u32;
        for r in list.split('+') {
            let r: usize = r
                .parse()
                .map_err(|_| format!("--pd {entry:?}: {r:?} is not a rank index"))?;
            if r >= MAX_DP || mask >> r & 1 != 0 {
                return Err(format!("--pd {entry:?}: rank {r} out of range or repeated"));
            }
            mask |= 1 << r;
        }
        Ok(mask)
    };
    let mut out: Vec<(String, PdSpec)> = Vec::new();
    for entry in entries {
        let (model, pools) = entry
            .rsplit_once('=')
            .ok_or_else(|| format!("--pd {entry:?} is not MODEL=PREFILL/DECODE"))?;
        let (p, d) = pools
            .split_once('/')
            .ok_or_else(|| format!("--pd {entry:?} is not MODEL=PREFILL/DECODE"))?;
        let spec = PdSpec {
            prefill: ranks(p, entry)?,
            decode: ranks(d, entry)?,
        };
        if model.is_empty() || spec.prefill & spec.decode != 0 {
            return Err(format!(
                "--pd {entry:?}: a rank is in both pools, or no model"
            ));
        }
        if out.iter().any(|(m, _)| m == model) {
            return Err(format!("--pd names {model:?} twice"));
        }
        out.push((model.to_string(), spec));
    }
    Ok(out)
}

/// Whether `job` can run disaggregated: plain text, no host-side per-token state the decode rank
/// would lose (prompt logprobs, a retained session, media rows).
pub(crate) fn eligible(job: &Job) -> bool {
    let o = &job.opts;
    o.speech.is_none()
        && o.mm.is_none()
        && o.session.is_none()
        && o.turn.is_none()
        && o.export.is_none()
        && o.import.is_none()
        && !o.prompt_logprobs
}

/// Submit `job`, routed to `set`'s prefill rank `mux`, through the P/D flow when the set has
/// one and the job is eligible; otherwise as a whole request. A refusal hands the original job
/// back, as [`ModelMux::submit_arrived`] does.
pub(crate) async fn submit(
    state: &AppState,
    set: &DpSet,
    mux: &ModelMux,
    mut job: Job,
    arrived: Instant,
    ingress: Option<IngressGuard<'_>>,
) -> Result<(), SubmitError> {
    let (Some(pd), Some(cfg)) = (set.pd, state.dp().map(|d| d.cfg)) else {
        return mux.submit_arrived(job, arrived, ingress);
    };
    let decoders: Vec<(usize, ModelMux)> = set
        .ranks
        .iter()
        .filter(|r| pd.decode >> r.rank & 1 != 0)
        .filter_map(|r| Some((r.rank, state.mux(&r.key)?)))
        .collect();
    if decoders.is_empty() {
        return mux.submit_arrived(job, arrived, ingress);
    }
    // The prefill rank samples the first token and nothing else: no stop handling, no thinking
    // control (the decode rank commits the token through both), its id streamed raw.
    let outcome = Outcome::new(Arc::clone(mux.metrics()), arrived);
    let (sink, image) = KvExport::new(Arc::clone(&outcome));
    let (ptx, prx) = stream::channel();
    let mut gen = job.gen.clone();
    gen.max_tokens = 1;
    gen.min_tokens = 0;
    gen.ignore_eos = true;
    gen.stop.clear();
    gen.think = None;
    let prefill = Job {
        prompt_ids: job.prompt_ids.clone(),
        gen,
        arrived: job.arrived,
        respond: ptx,
        opts: JobOpts {
            class: job.opts.class,
            tenant: job.opts.tenant,
            raw_tokens: true,
            prefix: job.opts.prefix.take(),
            export: Some(sink),
            ..Default::default()
        },
    };
    let source_prompt = (tracing::enabled!(tracing::Level::DEBUG)
        && job.gen.params.logprobs.is_some())
    .then(|| job.prompt_ids.clone());
    if let Err(e) = mux.submit_wait_arrived(prefill, arrived, ingress).await {
        // The refusal is the caller's to retry or report, as for a whole request.
        outcome.settle();
        let (SubmitError::Full(p) | SubmitError::Closed(p)) = &e;
        job.opts.prefix = p.opts.prefix.clone();
        return Err(match e {
            SubmitError::Full(_) => SubmitError::Full(job),
            SubmitError::Closed(_) => SubmitError::Closed(job),
        });
    }
    let set = state.dp_set(&set.model).map(Arc::clone);
    tokio::spawn(async move {
        let respond = job.respond.clone();
        let fail = |outcome: &Outcome, msg: String| {
            if outcome.claim_terminal() {
                if outcome.settle() {
                    Metrics::inc(&outcome.metrics.rejected);
                }
                let _ = respond.try_send(StreamChunk::Err(RuntimeError::Unavailable(msg)));
            }
        };
        let Some(set) = set else {
            return fail(&outcome, "P/D: the model's DP set is gone".into());
        };
        let Some(routed) = set.route_decode(&cfg, job.gen.params.samples()) else {
            return fail(&outcome, "P/D: no decode rank is up".into());
        };
        let Some((_, mux)) = decoders.iter().find(|(r, _)| *r == routed.rank) else {
            return fail(&outcome, "P/D: the decode rank has no dispatcher".into());
        };
        drive_handoff(mux, job, arrived, prx, image, outcome, source_prompt).await;
        drop(routed);
    });
    Ok(())
}

pub(super) async fn drive_handoff(
    mux: &ModelMux,
    mut job: Job,
    arrived: Instant,
    prx: stream::ChunkReceiver,
    image: oneshot::Receiver<crate::Result<KvImage>>,
    outcome: Arc<Outcome>,
    source_prompt: Option<Vec<u32>>,
) {
    let respond = job.respond.clone();
    let (import, mut deliver) = KvImport::pending(arrived, Arc::clone(&outcome));
    job.opts.import = Some(Box::new(import));
    let admission = async {
        match submit_decode(mux, job, Instant::now()).await {
            Some(Ok(())) => Ok(()),
            None => Err(()),
            Some(Err(SubmitError::Full(job) | SubmitError::Closed(job))) => {
                settle_refused(&job);
                let _ = respond.try_send(StreamChunk::Err(RuntimeError::Unavailable(
                    "P/D: the decode rank refused the request".into(),
                )));
                Err(())
            }
        }
    };
    let handoff = async {
        let payload = collect_prefill(
            prx,
            image,
            &mut deliver,
            &respond,
            &outcome,
            source_prompt.as_deref(),
        )
        .await?;
        deliver.send(payload).map_err(|_| ())?;
        mux.wake_import();
        Ok::<_, ()>(())
    };
    let _ = tokio::try_join!(admission, handoff);
}

async fn collect_prefill(
    mut prx: stream::ChunkReceiver,
    image: oneshot::Receiver<crate::Result<KvImage>>,
    deliver: &mut oneshot::Sender<ReadyImport>,
    respond: &stream::ChunkSender,
    outcome: &Outcome,
    source_prompt: Option<&[u32]>,
) -> Result<ReadyImport, ()> {
    let fail = |outcome: &Outcome, msg: String| {
        if outcome.claim_terminal() {
            if outcome.settle() {
                Metrics::inc(&outcome.metrics.rejected);
            }
            let _ = respond.try_send(StreamChunk::Err(RuntimeError::Unavailable(msg)));
        }
    };
    // A client that leaves stops the handoff: dropping the prefill stream ends that job, and
    // its dispatcher skips an export nobody waits for. `outcome` counts the abort.
    let (mut first, mut cached_tokens) = (None, 0);
    loop {
        let chunk = tokio::select! {
            c = prx.recv() => c,
            _ = respond.closed() => return Err(()),
            _ = deliver.closed() => return Err(()),
        };
        match chunk {
            Some(StreamChunk::Token { id, logprobs, .. })
                if first.is_none() && id != stream::TEXT_ONLY =>
            {
                let first_at = Instant::now();
                if let (Some(lp), Some(prompt)) = (&logprobs, source_prompt) {
                    use sha2::{Digest, Sha256};
                    let mut hash = Sha256::new();
                    for token in prompt {
                        hash.update(token.to_le_bytes());
                    }
                    let digest = format!("{:x}", hash.finalize());
                    let rows = prompt.len();
                    let top_bits: Vec<_> = lp
                        .top
                        .iter()
                        .map(|&(token, value)| (token, value.to_bits()))
                        .collect();
                    tracing::debug!(prompt_sha256 = %digest, rows, token = id,
                            logprob_bits = lp.logprob.to_bits(), ?top_bits,
                            "pd: source first logprobs");
                }
                first = Some((id, first_at, logprobs))
            }
            Some(StreamChunk::Done {
                reason: FinishReason::Preempted,
                executed,
                usage,
            }) => {
                if outcome.claim_terminal() {
                    if outcome.settle() {
                        Metrics::inc(&outcome.metrics.serving.preempted);
                    }
                    let _ = respond.try_send(StreamChunk::Done {
                        executed,
                        reason: FinishReason::Preempted,
                        usage,
                    });
                }
                return Err(());
            }
            Some(StreamChunk::Done { usage, .. }) => cached_tokens = usage.cached_tokens,
            Some(StreamChunk::Err(err)) => {
                if outcome.report_abort() {
                    let _ = respond.try_send(StreamChunk::Err(err));
                }
                return Err(());
            }
            Some(_) => {}
            None => break,
        }
    }
    let image = tokio::select! {
        i = image => i,
        _ = respond.closed() => return Err(()),
            _ = deliver.closed() => return Err(()),
    };
    let image = match (&first, image) {
        (Some(_), Ok(Ok(image))) => image,
        (None, _) => {
            let msg = "P/D: the prefill rank produced no token".into();
            fail(outcome, msg);
            return Err(());
        }
        (_, Ok(Err(err))) => {
            fail(outcome, format!("P/D: KV export failed: {err}"));
            return Err(());
        }
        (_, Err(_)) => {
            let msg = "P/D: the prefill rank dropped the KV image".into();
            fail(outcome, msg);
            return Err(());
        }
    };
    if respond.is_closed() {
        return Err(());
    }
    let (token, first, logprobs) = first.expect("matched Some");
    Ok(ReadyImport {
        image,
        token,
        first,
        logprobs,
        cached_tokens,
    })
}

pub(crate) async fn submit_decode(
    mux: &ModelMux,
    job: Job,
    arrived: Instant,
) -> Option<Result<(), SubmitError>> {
    let respond = job.respond.clone();
    tokio::select! {
        biased;
        _ = respond.closed() => None,
        result = mux.submit_wait_arrived(job, arrived, None) => Some(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(now: Instant) -> ReadyImport {
        ReadyImport {
            image: KvImage {
                key: "key".into(),
                rows: 3,
                bytes: vec![42].into(),
                handoff_started: now,
            },
            token: 120,
            first: now,
            logprobs: Some(Box::new(crate::text::logprobs::TokenLogprobs {
                logprob: -0.125,
                top: vec![(120, -0.125), (121, -4.)],
            })),
            cached_tokens: 2,
        }
    }

    #[test]
    fn pending_import_prepares_once_and_starts_only_one_decode_telemetry() {
        let now = Instant::now();
        let source = Arc::new(Metrics::default());
        let decode = Arc::new(Metrics::default());
        let (mut import, deliver) = KvImport::pending(now, Outcome::new(source.clone(), now));
        import.metrics = Some(decode.clone());
        import.poll();
        assert!(import.waiting() && !import.abandoned());
        assert!(import.start_metrics(3, 1).is_none());
        assert!(import.prepare_with(|| Ok(())).unwrap());
        assert!(!import
            .prepare_with(|| panic!("must not begin the slot twice"))
            .unwrap());
        assert_eq!(decode.serving.running.load(Ordering::Relaxed), 0);
        assert!(deliver.send(payload(now)).is_ok());
        import.poll();
        assert!(!import.waiting() && !import.abandoned() && import.prepared);
        assert_eq!(import.arrived, now);
        let mut telemetry = import.start_metrics(3, 1).unwrap();
        assert!(import.start_metrics(3, 1).is_none());
        assert_eq!(decode.serving.running.load(Ordering::Relaxed), 1);
        let ready = import.take_ready();
        assert_eq!(
            (ready.first, ready.token, ready.cached_tokens),
            (now, 120, 2)
        );
        assert_eq!(ready.logprobs, payload(now).logprobs);
        telemetry.token(ready.cached_tokens);
        telemetry.finish(FinishReason::Length, 1);
        drop(telemetry);
        let snapshot = crate::obs::serving::snapshot(&[("m".into(), decode, true)]);
        let m = &snapshot.models[0];
        assert_eq!(
            (
                m.completed,
                m.generated_tokens,
                m.running,
                m.aborted,
                m.e2e.count
            ),
            (1, 1, 0, 0, 1)
        );
        assert_eq!(source.serving.aborted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn failed_preparation_retries_and_an_already_ready_image_does_not_prepare() {
        let now = Instant::now();
        let (mut import, deliver) = KvImport::pending(now, Outcome::new(Arc::default(), now));
        assert!(import
            .prepare_with(|| Err(RuntimeError::Oom("pressure".into())))
            .is_err());
        assert!(!import.prepared);
        assert!(import.prepare_with(|| Ok(())).unwrap());
        assert!(!import
            .prepare_with(|| panic!("retry after success"))
            .unwrap());
        drop(deliver);
        import.poll();
        assert!(import.abandoned());
        assert!(!import
            .prepare_with(|| panic!("prepare cancelled request"))
            .unwrap());

        let mut ready = KvImport::ready(payload(now), now, Outcome::new(Arc::default(), now));
        assert!(!ready
            .prepare_with(|| panic!("ready import uses ordinary seating"))
            .unwrap());
    }

    #[test]
    fn abandoned_pending_import_counts_one_abort_unless_a_rank_already_settled_it() {
        for settled in [false, true] {
            for prepared in [false, true] {
                let now = Instant::now();
                let source = Arc::new(Metrics::default());
                let decode = Arc::new(Metrics::default());
                let outcome = Outcome::new(source.clone(), now);
                if settled {
                    outcome.settle();
                }
                let (mut import, deliver) = KvImport::pending(now, outcome);
                import.metrics = Some(decode.clone());
                if prepared {
                    import.prepare_with(|| Ok(())).unwrap();
                }
                drop(deliver);
                import.poll();
                assert!(import.abandoned() && !import.waiting());
                assert!(import.start_metrics(3, 1).is_none());
                drop(import);
                assert_eq!(
                    source.serving.aborted.load(Ordering::Relaxed),
                    u64::from(!settled)
                );
                assert_eq!(decode.serving.running.load(Ordering::Relaxed), 0);
                assert_eq!(decode.serving.aborted.load(Ordering::Relaxed), 0);
            }
        }
    }

    #[tokio::test]
    async fn preempted_pending_decode_counts_once_and_closes_the_producer_channel() {
        let now = Instant::now();
        let source = Arc::new(Metrics::default());
        let decode = Arc::new(Metrics::default());
        let (mut import, mut deliver) = KvImport::pending(now, Outcome::new(source.clone(), now));
        import.metrics = Some(decode.clone());
        import.preempt();
        import.preempt();
        drop(import);
        tokio::time::timeout(std::time::Duration::from_secs(1), deliver.closed())
            .await
            .unwrap();
        assert_eq!(decode.serving.preempted.load(Ordering::Relaxed), 1);
        assert_eq!(decode.serving.aborted.load(Ordering::Relaxed), 0);
        assert_eq!(source.serving.aborted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn refusal_accounting_distinguishes_internal_prefill_and_external_decode_terminals() {
        for source_first in [false, true] {
            let now = Instant::now();
            let source = Arc::new(Metrics::default());
            let decode = Arc::new(Metrics::default());
            let outcome = Outcome::new(source.clone(), now);
            let (export, _image) = KvExport::new(outcome.clone());
            let (import, _deliver) = KvImport::pending(now, outcome.clone());
            let job = |opts| Job {
                prompt_ids: vec![1, 2, 3],
                gen: crate::serve::GenParams::default(),
                arrived: now,
                respond: stream::channel().0,
                opts,
            };
            let prefill = job(JobOpts {
                export: Some(export),
                ..Default::default()
            });
            let decoder = job(JobOpts {
                import: Some(Box::new(import)),
                ..Default::default()
            });
            if source_first {
                assert!(refuse(&prefill, &source.rejected));
                assert!(refuse(&decoder, &decode.rejected));
                assert!(
                    !outcome.report_abort(),
                    "decoder reports the already-counted failure"
                );
            } else {
                assert!(refuse(&decoder, &decode.rejected));
                assert!(!refuse(&prefill, &source.rejected));
                assert!(
                    !outcome.report_abort(),
                    "decoder already reported externally"
                );
            }
            drop((prefill, decoder, outcome));
            assert_eq!(
                source.rejected.load(Ordering::Relaxed),
                u64::from(source_first)
            );
            assert_eq!(
                decode.rejected.load(Ordering::Relaxed),
                u64::from(!source_first)
            );
            assert_eq!(source.serving.aborted.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn concurrent_source_abort_and_pending_decode_preempt_have_one_counter_and_terminal() {
        for _ in 0..64 {
            let now = Instant::now();
            let source = Arc::new(Metrics::default());
            let decode = Arc::new(Metrics::default());
            let outcome = Outcome::new(source.clone(), now);
            let (mut import, _deliver) = KvImport::pending(now, outcome.clone());
            import.metrics = Some(decode.clone());
            let terminals = std::sync::atomic::AtomicU64::new(0);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    if import.preempt() {
                        terminals.fetch_add(1, Ordering::Relaxed);
                    }
                });
                if outcome.report_abort() {
                    terminals.fetch_add(1, Ordering::Relaxed);
                }
            });
            drop((import, outcome));
            assert_eq!(terminals.load(Ordering::Relaxed), 1);
            assert_eq!(
                source.serving.aborted.load(Ordering::Relaxed)
                    + decode.serving.preempted.load(Ordering::Relaxed),
                1
            );
            assert_eq!(decode.serving.aborted.load(Ordering::Relaxed), 0);
        }
    }

    fn source_token(now: Instant) -> StreamChunk {
        StreamChunk::Token {
            id: 120,
            text: String::new(),
            logprobs: payload(now).logprobs,
            frame: None,
        }
    }

    fn source_done(reason: FinishReason) -> StreamChunk {
        StreamChunk::Done {
            executed: 1,
            reason,
            usage: stream::TokenUsage {
                prompt_tokens: 3,
                cached_tokens: 2,
                completion_tokens: 1,
                think: None,
            },
        }
    }

    #[tokio::test]
    async fn source_collection_does_not_wait_for_decoder_seating_or_image_completion() {
        let now = Instant::now();
        let metrics = Arc::new(Metrics::default());
        let outcome = Outcome::new(metrics.clone(), now);
        let (pending, mut deliver) = KvImport::pending(now, outcome.clone());
        let (ptx, prx) = stream::channel();
        let (respond, _client) = stream::channel();
        let (sink, image_rx) = KvExport::new(outcome.clone());
        ptx.try_send(source_token(now)).unwrap();
        ptx.try_send(source_done(FinishReason::Length)).unwrap();
        drop(ptx);
        let mut collect = Box::pin(collect_prefill(
            prx,
            image_rx,
            &mut deliver,
            &respond,
            &outcome,
            None,
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(5), collect.as_mut())
                .await
                .is_err()
        );
        let image_released = Instant::now();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let producer = tokio::task::spawn_blocking(move || {
            sink.produce(Some("key".into()), 3, now, |out, publish| {
                *out = vec![42].into();
                publish(vec![42].into());
                release_rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                Ok(())
            })
        });
        let ready = tokio::time::timeout(std::time::Duration::from_secs(1), collect).await;
        assert!(!producer.is_finished());
        release_tx.send(()).unwrap();
        producer.await.unwrap().unwrap();
        let ready = ready.unwrap().unwrap();
        assert!(ready.first >= now && ready.first <= image_released);
        assert_eq!((ready.token, ready.cached_tokens), (120, 2));
        assert_eq!(ready.logprobs, payload(now).logprobs);
        assert_eq!(&*ready.image.bytes, [42]);
        assert!(pending.waiting());
        outcome.settle();
        drop((pending, outcome));
        assert_eq!(metrics.serving.aborted.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn simultaneous_internal_and_decode_refusals_keep_one_external_terminal() {
        for source_first in [false, true] {
            for _ in 0..32 {
                let now = Instant::now();
                let source = Arc::new(Metrics::default());
                let decode = Arc::new(Metrics::default());
                let outcome = Outcome::new(source.clone(), now);
                let (pending, mut deliver) = KvImport::pending(now, outcome.clone());
                let (respond, mut client) = stream::channel();
                let (ptx, prx) = stream::channel();
                let (export, _exported) = KvExport::new(outcome.clone());
                let job = |opts, respond| Job {
                    prompt_ids: vec![1, 2, 3],
                    gen: crate::serve::GenParams::default(),
                    arrived: now,
                    respond,
                    opts,
                };
                let prefill = job(
                    JobOpts {
                        export: Some(export),
                        ..Default::default()
                    },
                    ptx,
                );
                let decoder = job(
                    JobOpts {
                        import: Some(Box::new(pending)),
                        ..Default::default()
                    },
                    respond.clone(),
                );
                let reject_source = || {
                    if refuse(&prefill, &source.rejected) {
                        prefill
                            .respond
                            .try_send(StreamChunk::Err(RuntimeError::Unavailable(
                                "prefill refused".into(),
                            )))
                            .unwrap();
                    }
                };
                if source_first {
                    reject_source();
                }
                assert!(refuse(&decoder, &decode.rejected));
                decoder
                    .respond
                    .try_send(StreamChunk::Err(RuntimeError::Unavailable(
                        "decode refused".into(),
                    )))
                    .unwrap();
                if !source_first {
                    reject_source();
                }
                drop(decoder);
                let (_image_tx, image_rx) = oneshot::channel();
                assert!(tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    collect_prefill(prx, image_rx, &mut deliver, &respond, &outcome, None)
                )
                .await
                .unwrap()
                .is_err());
                assert!(matches!(client.try_recv().unwrap(), StreamChunk::Err(_)));
                assert!(client.try_recv().is_err());
                drop((prefill, outcome));
                assert_eq!(
                    source.rejected.load(Ordering::Relaxed)
                        + decode.rejected.load(Ordering::Relaxed),
                    1
                );
                assert_eq!(
                    source.serving.aborted.load(Ordering::Relaxed)
                        + decode.serving.aborted.load(Ordering::Relaxed),
                    0
                );
            }
        }
    }

    #[tokio::test]
    async fn source_failures_and_closed_peers_terminate_once_without_waiting() {
        for case in [
            "preempt",
            "source-error",
            "image-error",
            "missing-token",
            "missing-image",
            "prefill-refused",
            "client-gone",
            "decode-gone",
        ] {
            let now = Instant::now();
            let metrics = Arc::new(Metrics::default());
            let outcome = Outcome::new(metrics.clone(), now);
            let (pending, mut deliver) = KvImport::pending(now, outcome.clone());
            let mut pending = Some(pending);
            let (ptx, prx) = stream::channel();
            let (respond, client) = stream::channel();
            let mut client = Some(client);
            let (image_tx, image_rx) = oneshot::channel();
            if case == "client-gone" {
                drop(client.take());
            }
            if case == "decode-gone" {
                drop(pending.take());
            }
            let closed = case == "client-gone" || case == "decode-gone";
            if case == "prefill-refused" {
                outcome.settle();
                Metrics::inc(&metrics.rejected);
            }
            if !closed {
                if case != "missing-token" && case != "prefill-refused" {
                    ptx.try_send(source_token(now)).unwrap();
                }
                ptx.try_send(if case == "source-error" || case == "prefill-refused" {
                    StreamChunk::Err(RuntimeError::Unavailable("source failed".into()))
                } else {
                    source_done(if case == "preempt" {
                        FinishReason::Preempted
                    } else {
                        FinishReason::Length
                    })
                })
                .unwrap();
                drop(ptx);
            }
            if case == "missing-image" {
                drop(image_tx);
            } else if !closed {
                assert!(image_tx
                    .send(if case == "image-error" {
                        Err(RuntimeError::Rejected("export failed".into()))
                    } else {
                        Ok(payload(now).image)
                    })
                    .is_ok());
            }
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                collect_prefill(prx, image_rx, &mut deliver, &respond, &outcome, None),
            )
            .await
            .unwrap();
            assert!(result.is_err(), "{case}");
            if let Some(client) = client.as_mut() {
                if closed {
                    assert!(client.try_recv().is_err());
                } else {
                    let terminal = client.try_recv().unwrap();
                    assert!(
                        matches!(terminal, StreamChunk::Err(_))
                            || matches!(
                                terminal,
                                StreamChunk::Done {
                                    reason: FinishReason::Preempted,
                                    ..
                                }
                            )
                    );
                    assert!(client.try_recv().is_err());
                }
            }
            drop((pending, outcome));
            let rejected = matches!(
                case,
                "image-error" | "missing-token" | "missing-image" | "prefill-refused"
            );
            assert_eq!(
                metrics.rejected.load(Ordering::Relaxed),
                u64::from(rejected),
                "{case}"
            );
            assert_eq!(
                metrics.serving.preempted.load(Ordering::Relaxed),
                u64::from(case == "preempt"),
                "{case}"
            );
            assert_eq!(
                metrics.serving.aborted.load(Ordering::Relaxed),
                u64::from(!rejected && case != "preempt"),
                "{case}"
            );
        }
    }

    #[test]
    fn engine_keys_refuse_missing_storage_and_mismatch_only_selected_ranks() {
        let spec = PdSpec {
            prefill: 1,
            decode: 2,
        };
        assert!(check_engine_keys("m", spec, &[Some("a".into()), Some("a".into()), None]).is_ok());
        for keys in [
            vec![Some("a".into()), None],
            vec![None, Some("a".into())],
            vec![Some("a".into()), Some("b".into())],
            vec![Some("a".into())],
        ] {
            assert!(check_engine_keys("m", spec, &keys).is_err());
        }
    }

    #[test]
    fn generated_logprobs_are_eligible_but_prompt_logprobs_are_not() {
        let mut job = Job {
            prompt_ids: vec![1, 2, 3],
            gen: crate::serve::GenParams::default(),
            arrived: Instant::now(),
            respond: stream::channel().0,
            opts: JobOpts::default(),
        };
        job.gen.params.logprobs = Some(crate::text::logprobs::LogprobRequest {
            top: 20,
            raw_logits: false,
        });
        assert!(eligible(&job));
        job.opts.prompt_logprobs = true;
        assert!(!eligible(&job));
    }

    #[test]
    fn pd_specs_parse_and_refuse_overlap() {
        assert_eq!(
            parse_pd(&["m=0/1".into(), "org/x=0+1/2+3".into()]).unwrap(),
            vec![
                (
                    "m".to_string(),
                    PdSpec {
                        prefill: 1,
                        decode: 2
                    }
                ),
                (
                    "org/x".to_string(),
                    PdSpec {
                        prefill: 3,
                        decode: 12
                    }
                )
            ]
        );
        for bad in ["m", "m=0", "m=0/0", "m=x/1", "=0/1", "m=0+0/1"] {
            assert!(parse_pd(&[bad.into()]).is_err(), "{bad}");
        }
        assert!(parse_pd(&["m=0/1".into(), "m=1/0".into()]).is_err());
    }

    #[test]
    fn an_unsettled_outcome_is_one_abort() {
        let m = Arc::new(Metrics::default());
        let open = || Outcome::new(Arc::clone(&m), Instant::now());
        let (sink, _rx) = KvExport::new(open());
        drop(sink);
        let shared = open();
        let (sink, _rx) = KvExport::new(Arc::clone(&shared));
        sink.outcome.settle();
        drop((sink, shared));
        assert_eq!(
            m.serving.aborted.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn an_export_reaches_its_receiver_once() {
        let (sink, mut rx) = KvExport::new(Outcome::new(Arc::default(), Instant::now()));
        assert!(sink.wanted());
        sink.deliver(Ok(KvImage {
            key: "k".into(),
            rows: 3,
            bytes: vec![1, 2].into(),
            handoff_started: Instant::now(),
        }));
        assert!(!sink.wanted());
        sink.deliver(Err(crate::RuntimeError::Msg("late".into())));
        assert_eq!(rx.try_recv().unwrap().unwrap().rows, 3);
    }
}
