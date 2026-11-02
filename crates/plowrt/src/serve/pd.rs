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

/// One slot's KV, as `GpuEngine::export_kv_image` wrote it.
pub struct KvImage {
    /// The exporting engine's key (`GpuEngine::kv_image_key`): its caches' semantic header.
    pub key: String,
    pub rows: u32,
    pub bytes: Vec<u8>,
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
}

/// A decode job's starting point: the prompt's KV and the first token, sampled by the prefill
/// rank.
pub struct KvImport {
    pub image: KvImage,
    pub token: u32,
    /// The request's arrival, and when the prefill rank streamed the first token: the decode
    /// rank's e2e and SLO count from them.
    pub arrived: Instant,
    pub first: Instant,
    pub outcome: Arc<Outcome>,
    /// Prompt rows the prefill rank served from its prefix cache (`usage.cached_tokens`).
    pub cached_tokens: usize,
}

/// A P/D request's terminal outcome until its decode leg owns it. Settled by whoever counts it:
/// the decode leg's telemetry, a rank's refusal of either leg, a failed handoff, a preempted
/// prefill. The prefill leg's telemetry counts none; dropped unsettled (a cancellation, a failed
/// prefill, a job never seated) it counts one abort on the prefill rank.
pub struct Outcome {
    metrics: Arc<Metrics>,
    arrived: Instant,
    open: AtomicBool,
}

impl Outcome {
    pub fn new(metrics: Arc<Metrics>, arrived: Instant) -> Arc<Self> {
        Arc::new(Self {
            metrics,
            arrived,
            open: AtomicBool::new(true),
        })
    }

    pub fn settle(&self) {
        self.open.store(false, Ordering::Relaxed);
    }
}

impl Drop for Outcome {
    fn drop(&mut self) {
        if *self.open.get_mut() {
            let m = &self.metrics.serving;
            Metrics::inc(&m.aborted);
            Metrics::inc(&m.aborted_before_first);
            m.aborted_elapsed.duration(self.arrived.elapsed());
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

/// The KV image header of the bundle in `dir`, its checkpoint hashed once per checkpoint
/// directory (`digests`).
pub fn kv_header(
    dir: &std::path::Path,
    digests: &mut std::collections::HashMap<std::path::PathBuf, String>,
) -> Result<plow_asset::kv_image::Header, String> {
    use crate::asset::devblob::DevBlob;
    let pkt = DevBlob::find_in_dir(dir)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no packet in {}", dir.display()))?;
    let raw = std::fs::read(&pkt).map_err(|e| format!("{}: {e}", pkt.display()))?;
    let blob = DevBlob::parse_l2(&raw, true).map_err(|e| e.to_string())?;
    let manifest = match crate::memory::vmm::LiveKvLayout::manifest(&blob, &raw) {
        Ok(Some(m)) => m,
        _ => blob.with_packet_view(plow_asset::live_kv::emit)?,
    };
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
/// would lose (logprobs, a retained session, media rows).
pub(crate) fn eligible(job: &Job) -> bool {
    let o = &job.opts;
    o.speech.is_none()
        && o.mm.is_none()
        && o.session.is_none()
        && o.turn.is_none()
        && o.export.is_none()
        && o.import.is_none()
        && job.gen.params.logprobs.is_none()
}

/// Submit `job`, routed to `set`'s prefill rank `mux`, through the P/D flow when the set has
/// one and the job is eligible; otherwise as a whole request. A refusal hands the original job
/// back, as [`ModelMux::submit_arrived`] does.
pub(crate) fn submit(
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
    let (ptx, mut prx) = stream::channel();
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
    if let Err(e) = mux.submit_arrived(prefill, arrived, ingress) {
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
        let fail = |job: &Job, outcome: &Outcome, msg: String| {
            outcome.settle();
            Metrics::inc(&outcome.metrics.rejected);
            let _ = job
                .respond
                .try_send(StreamChunk::Err(RuntimeError::Unavailable(msg)));
        };
        // A client that leaves stops the handoff: dropping the prefill stream ends that job, and
        // its dispatcher skips an export nobody waits for. `outcome` counts the abort.
        let (mut first, mut cached_tokens) = (None, 0);
        loop {
            let chunk = tokio::select! {
                c = prx.recv() => c,
                _ = job.respond.closed() => return,
            };
            match chunk {
                Some(StreamChunk::Token { id, .. })
                    if first.is_none() && id != stream::TEXT_ONLY =>
                {
                    first = Some((id, Instant::now()))
                }
                Some(StreamChunk::Done {
                    reason: FinishReason::Preempted,
                    executed,
                    usage,
                }) => {
                    outcome.settle();
                    Metrics::inc(&outcome.metrics.serving.preempted);
                    let _ = job.respond.try_send(StreamChunk::Done {
                        executed,
                        reason: FinishReason::Preempted,
                        usage,
                    });
                    return;
                }
                Some(StreamChunk::Done { usage, .. }) => cached_tokens = usage.cached_tokens,
                Some(StreamChunk::Err(err)) => {
                    let _ = job.respond.try_send(StreamChunk::Err(err));
                    return;
                }
                Some(_) => {}
                None => break,
            }
        }
        let image = tokio::select! {
            i = image => i,
            _ = job.respond.closed() => return,
        };
        let image = match (first, image) {
            (Some(_), Ok(Ok(image))) => image,
            (None, _) => {
                let msg = "P/D: the prefill rank produced no token".into();
                return fail(&job, &outcome, msg);
            }
            (_, Ok(Err(err))) => {
                return fail(&job, &outcome, format!("P/D: KV export failed: {err}"))
            }
            (_, Err(_)) => {
                let msg = "P/D: the prefill rank dropped the KV image".into();
                return fail(&job, &outcome, msg);
            }
        };
        if job.respond.is_closed() {
            return;
        }
        let Some(set) = set else {
            return fail(&job, &outcome, "P/D: the model's DP set is gone".into());
        };
        let sampled = job.gen.params.samples();
        let Some(routed) = set.route_decode(&cfg, sampled) else {
            return fail(&job, &outcome, "P/D: no decode rank is up".into());
        };
        let Some((_, mux)) = decoders.iter().find(|(r, _)| *r == routed.rank) else {
            let msg = "P/D: the decode rank has no dispatcher".into();
            return fail(&job, &outcome, msg);
        };
        let (token, first) = first.expect("matched Some");
        job.opts.import = Some(Box::new(KvImport {
            image,
            token,
            arrived,
            first,
            cached_tokens,
            outcome,
        }));
        // The decode queue's clock starts now; `job.arrived` keeps the request's own.
        if let Err(SubmitError::Full(job) | SubmitError::Closed(job)) =
            mux.submit_arrived(job, Instant::now(), None)
        {
            // The decode rank counted its refusal.
            settle_refused(&job);
            let _ = job
                .respond
                .try_send(StreamChunk::Err(RuntimeError::Unavailable(
                    "P/D: the decode rank refused the request".into(),
                )));
        }
        drop(routed);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            bytes: vec![1, 2],
        }));
        assert!(!sink.wanted());
        sink.deliver(Err(crate::RuntimeError::Msg("late".into())));
        assert_eq!(rx.try_recv().unwrap().unwrap().rows, 3);
    }
}
