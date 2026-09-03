//! In-process benchmark client for the production model mux.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use futures::stream::{FuturesUnordered, StreamExt};
use serde::Serialize;

use super::mux::{Job, ModelMux, SubmitError};
use super::stream::{self, FinishReason, StreamChunk};
use super::{AppState, GenParams};
use crate::text::sample::SamplingParams;
use crate::{Result, RuntimeError};

#[derive(Clone, Debug)]
pub enum Input {
    TokenIds(Vec<u32>),
    TokenRows(Vec<Vec<u32>>),
    Random { len: usize, seed: u64 },
}

#[derive(Clone, Debug)]
pub struct Config {
    pub model: String,
    pub input: Input,
    pub concurrency: usize,
    pub warmup_requests: usize,
    pub requests: usize,
    pub output_tokens: usize,
    pub runtime: serde_json::Value,
    pub parity_report: bool,
    pub token_audit: bool,
}

#[derive(Clone, Debug)]
pub struct PrefillSweepConfig {
    pub model: String,
    pub inputs: Vec<Input>,
    pub warmup_requests: usize,
    pub repetitions: usize,
    pub runtime: serde_json::Value,
}

const MAX_DIAGNOSTIC_SELECTIONS: usize = 16 * 1024;

pub fn read_prompt_rows(path: &Path) -> Result<Vec<Vec<u32>>> {
    let raw = fs::read_to_string(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_prompt_rows(&raw).map_err(|error| {
        RuntimeError::Msg(format!(
            "invalid prompt rows file {}: {error}",
            path.display()
        ))
    })
}

fn parse_prompt_rows(raw: &str) -> Result<Vec<Vec<u32>>> {
    let mut rows = Vec::new();
    for (line_index, line) in raw.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            return Err(RuntimeError::Msg(format!(
                "line {line_number} is an empty prompt row"
            )));
        }
        let mut row = Vec::new();
        for (column_index, field) in line.split(',').enumerate() {
            let field = field.trim();
            if field.is_empty() {
                return Err(RuntimeError::Msg(format!(
                    "line {line_number}, field {} is empty",
                    column_index + 1
                )));
            }
            row.push(field.parse::<u32>().map_err(|error| {
                RuntimeError::Msg(format!(
                    "line {line_number}, field {} is not a u32 token id: {error}",
                    column_index + 1
                ))
            })?);
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(RuntimeError::Msg(
            "file must contain at least one prompt row".into(),
        ));
    }
    Ok(rows)
}

pub fn validate_request_layout(
    input: &Input,
    warmup_requests: usize,
    requests: usize,
) -> Result<()> {
    if let Input::TokenRows(rows) = input {
        let expected = warmup_requests
            .checked_add(requests)
            .ok_or_else(|| RuntimeError::Msg("bench request count overflow".into()))?;
        if rows.len() != expected {
            return Err(RuntimeError::Msg(format!(
                "bench prompt rows must contain exactly warmup_requests + requests rows: got {}, expected {} + {} = {expected}",
                rows.len(),
                warmup_requests,
                requests
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineDiagnostics {
    pub supported: bool,
    pub complete: bool,
    pub overflowed: bool,
    pub scope: &'static str,
    pub prefill_selections: Vec<PrefillSelection>,
    pub decode_selections: Vec<DecodeSelection>,
    pub rank_agreement: Option<RankAgreement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PrefillSelection {
    pub slot: usize,
    pub row_start: u32,
    pub rows: u32,
    pub bucket: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DecodeSelection {
    pub occupied_rows: usize,
    pub bucket: u32,
    pub steps: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RankAgreement {
    pub ranks: usize,
    pub sampled_token_every: u32,
    pub counter_audit_every_dispatch: bool,
    pub prefill_completion_all_ranks: bool,
}

impl EngineDiagnostics {
    pub(crate) fn unsupported() -> Self {
        Self {
            supported: false,
            complete: false,
            overflowed: false,
            scope: "warmup_and_measured",
            prefill_selections: Vec::new(),
            decode_selections: Vec::new(),
            rank_agreement: None,
        }
    }

    pub(crate) fn push_prefill(&mut self, selection: PrefillSelection) {
        if self.prefill_selections.len() == MAX_DIAGNOSTIC_SELECTIONS {
            self.overflowed = true;
            self.complete = false;
        } else if !self.overflowed {
            self.prefill_selections.push(selection);
        }
    }

    pub(crate) fn push_decode(&mut self, selection: DecodeSelection) {
        if self.decode_selections.len() == MAX_DIAGNOSTIC_SELECTIONS {
            self.overflowed = true;
            self.complete = false;
        } else if !self.overflowed {
            self.decode_selections.push(selection);
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema: &'static str,
    pub model: String,
    pub asset_dir: String,
    pub target: String,
    pub parallel: String,
    pub num_gpus: usize,
    pub backend: String,
    pub vendor: String,
    pub input: InputReport,
    pub concurrency: usize,
    pub warmup_requests: usize,
    pub requests: usize,
    pub completed: usize,
    pub failed: usize,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub duration_ms: f64,
    pub request_throughput: f64,
    pub output_token_throughput: f64,
    pub total_token_throughput: f64,
    pub ttft_ms: Distribution,
    pub tpot_ms: Option<Distribution>,
    pub itl_ms: Option<Distribution>,
    pub e2e_ms: Distribution,
    pub output_checksum: String,
    pub artifacts: ArtifactReport,
    pub runtime: serde_json::Value,
    pub engine: Option<EngineReport>,
    pub scheduler: SchedulerReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<EngineDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_audit: Option<TokenAuditReport>,
}

#[derive(Debug, Serialize)]
pub struct ParityReport {
    pub prompt_token_ids: Vec<Vec<u32>>,
    pub output_token_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Serialize)]
pub struct TokenAuditReport {
    pub prompt_token_ids: Vec<Vec<u32>>,
    pub output_token_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Serialize)]
pub struct PrefillSweepReport {
    pub schema: &'static str,
    pub model: String,
    pub asset_dir: String,
    pub target: String,
    pub parallel: String,
    pub num_gpus: usize,
    pub backend: String,
    pub vendor: String,
    pub warmup_requests_per_length: usize,
    pub repetitions_per_length: usize,
    pub rows: Vec<PrefillSweepRow>,
    pub artifacts: ArtifactReport,
    pub runtime: serde_json::Value,
    pub engine: Option<EngineReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<EngineDiagnostics>,
}

#[derive(Debug, Serialize)]
pub struct PrefillSweepRow {
    pub input: InputReport,
    pub prompt_tokens: usize,
    pub warmup_requests: usize,
    pub repetitions: usize,
    pub measured_request_offset: usize,
    pub completed: usize,
    pub failed: usize,
    pub duration_ms: f64,
    pub ttft_ms: Distribution,
    pub prompt_checksum: String,
    pub output_checksum: String,
}

#[derive(Debug, Serialize)]
pub struct ArtifactReport {
    pub packet: FileIdentity,
    pub weights_manifest: FileIdentity,
    pub build_manifest: Option<FileIdentity>,
    pub object_inventory: Vec<FileIdentity>,
    pub checkpoint: Option<CheckpointLayout>,
}

#[derive(Debug, Serialize)]
pub struct FileIdentity {
    pub path: String,
    pub bytes: u64,
    pub checksum: String,
}

#[derive(Debug, Serialize)]
pub struct CheckpointLayout {
    pub path: String,
    pub safetensor_shards: usize,
    pub bytes: u64,
    pub layout_checksum: String,
}

#[derive(Debug, Serialize)]
pub struct InputReport {
    pub mode: &'static str,
    pub tokens_per_request: Option<usize>,
    pub row_count: usize,
    pub min_tokens_per_request: usize,
    pub max_tokens_per_request: usize,
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct EngineReport {
    pub batch_capacity: usize,
    pub decode_rungs: Box<[u32]>,
}

#[derive(Debug, Serialize)]
pub struct SchedulerReport {
    pub decode_rung_actual: u64,
    pub decode_rung_admission: u64,
    pub decode_occupied_extent: u64,
    pub decode_rung_switches: u64,
    pub batch_count: u64,
    pub mean_batch_size: f64,
    pub rejected: u64,
    pub admit_shed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Distribution {
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

struct RequestResult {
    request: usize,
    prompt_tokens: usize,
    cached_tokens: usize,
    ids: Vec<u32>,
    ttft: Duration,
    tpot: Option<Duration>,
    itl: Vec<Duration>,
    e2e: Duration,
}

/// Write the last completed raw AMD packet trace after the caller has
/// quiesced the model mux.
#[cfg(feature = "hsa")]
pub fn write_amd_packet_trace(state: &AppState, model: &str, path: &Path) -> Result<()> {
    let engine = state
        .gpu_engine(model)
        .ok_or_else(|| RuntimeError::Msg(format!("no GPU engine for '{model}'")))?;
    let result = engine.lock().write_amd_packet_trace(path);
    result
}

#[cfg(any(feature = "cuda", feature = "hsa"))]
pub fn begin_engine_diagnostics(state: &AppState, model: &str) -> Result<()> {
    let engine = state
        .gpu_engine(model)
        .ok_or_else(|| RuntimeError::Msg(format!("no GPU engine for '{model}'")))?;
    engine.lock().begin_diagnostics();
    Ok(())
}

#[cfg(not(any(feature = "cuda", feature = "hsa")))]
pub fn begin_engine_diagnostics(_state: &AppState, _model: &str) -> Result<()> {
    Err(RuntimeError::Msg(
        "engine diagnostics require a GPU backend".into(),
    ))
}

#[cfg(any(feature = "cuda", feature = "hsa"))]
pub fn finish_engine_diagnostics(state: &AppState, model: &str) -> Result<EngineDiagnostics> {
    let engine = state
        .gpu_engine(model)
        .ok_or_else(|| RuntimeError::Msg(format!("no GPU engine for '{model}'")))?;
    let diagnostics = engine.lock().finish_diagnostics();
    validate_engine_diagnostics(diagnostics)
}

fn validate_engine_diagnostics(diagnostics: EngineDiagnostics) -> Result<EngineDiagnostics> {
    if !diagnostics.supported {
        return Err(RuntimeError::Msg(
            "engine diagnostics are not supported by this backend".into(),
        ));
    }
    if diagnostics.overflowed {
        return Err(RuntimeError::Msg(
            "engine diagnostic selection capture overflowed; refusing a partial report".into(),
        ));
    }
    if !diagnostics.complete {
        return Err(RuntimeError::Msg(
            "engine diagnostics did not complete; refusing a partial report".into(),
        ));
    }
    Ok(diagnostics)
}

#[cfg(not(any(feature = "cuda", feature = "hsa")))]
pub fn finish_engine_diagnostics(_state: &AppState, _model: &str) -> Result<EngineDiagnostics> {
    Err(RuntimeError::Msg(
        "engine diagnostics require a GPU backend".into(),
    ))
}

pub async fn run_prefill_sweep(
    state: &AppState,
    cfg: PrefillSweepConfig,
) -> Result<PrefillSweepReport> {
    if cfg.inputs.is_empty() || cfg.repetitions == 0 {
        return Err(RuntimeError::Msg(
            "bench prefill sweep requires inputs and positive repetitions".into(),
        ));
    }
    if cfg
        .inputs
        .iter()
        .any(|input| matches!(input, Input::TokenRows(_)))
    {
        return Err(RuntimeError::Msg(
            "bench prompt rows are not supported by prefill sweep".into(),
        ));
    }
    if state.execset.backend().vendor().is_none() {
        return Err(RuntimeError::Msg(
            "bench requires a matching GPU backend; refusing CPU fallback performance".into(),
        ));
    }
    let bundle = state.registry.get(&cfg.model)?;
    let vocab = bundle.tokenizer().vocab_size();
    if vocab == 0 {
        return Err(RuntimeError::Msg(
            "bench tokenizer reports an empty vocabulary".into(),
        ));
    }
    for input in &cfg.inputs {
        validate_input(input, vocab)?;
    }
    let runtime = crate::config::RuntimeConfig::get();
    let prefix_cache = match state.execset.backend().vendor() {
        Some(hwspec::Vendor::Nvidia) => runtime.nv.vmm_prefix,
        Some(hwspec::Vendor::Amd) => runtime.nv.prefix_cache,
        _ => false,
    };
    if prefix_cache {
        return Err(RuntimeError::Msg(
            "bench prefill sweep requires cold prompts; disable --prefix-cache/--vmm-prefix".into(),
        ));
    }
    let mux = state
        .mux(&cfg.model)
        .ok_or_else(|| RuntimeError::Msg(format!("no model mux for '{}'", cfg.model)))?;
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    let engine = {
        let engine = state
            .gpu_engine(&cfg.model)
            .ok_or_else(|| RuntimeError::Msg(format!("no GPU engine for '{}'", cfg.model)))?;
        let engine = engine.lock();
        Some(EngineReport {
            batch_capacity: engine.batch(),
            decode_rungs: engine.decode_rungs(),
        })
    };
    #[cfg(not(any(feature = "cuda", feature = "hsa")))]
    let engine = None;

    let mut rows = Vec::with_capacity(cfg.inputs.len());
    let mut request_offset = 0usize;
    for input in &cfg.inputs {
        if cfg.warmup_requests > 0 {
            crate::obs::Metrics::add(&state.metrics.requests, cfg.warmup_requests as u64);
            let warmup = drive(
                &mux,
                input,
                vocab,
                1,
                cfg.warmup_requests,
                1,
                request_offset,
            )
            .await?;
            validate(&warmup, 1)?;
            reject_cached_prefill(&warmup)?;
            request_offset += cfg.warmup_requests;
        }

        crate::obs::Metrics::add(&state.metrics.requests, cfg.repetitions as u64);
        let measured_offset = request_offset;
        let started = Instant::now();
        let results = drive(&mux, input, vocab, 1, cfg.repetitions, 1, measured_offset).await?;
        let elapsed = started.elapsed();
        validate(&results, 1)?;
        reject_cached_prefill(&results)?;
        request_offset += cfg.repetitions;
        let ttft = distribution(results.iter().map(|r| ms(r.ttft)).collect())
            .expect("positive repetitions");
        rows.push(PrefillSweepRow {
            input: input_report(input),
            prompt_tokens: single_input_len(input),
            warmup_requests: cfg.warmup_requests,
            repetitions: cfg.repetitions,
            measured_request_offset: measured_offset,
            completed: results.len(),
            failed: 0,
            duration_ms: ms(elapsed),
            ttft_ms: ttft,
            prompt_checksum: prompt_checksum(
                input,
                vocab,
                measured_offset..measured_offset + cfg.repetitions,
            ),
            output_checksum: checksum(&results),
        });
    }

    let artifacts = artifact_report(
        &bundle.dir,
        state.execset.backend().vendor() == Some(hwspec::Vendor::Amd),
    )?;
    Ok(PrefillSweepReport {
        schema: "plowrt.bench.prefill-sweep.v1",
        model: cfg.model,
        asset_dir: bundle.dir.display().to_string(),
        target: bundle.manifest.gpu.clone(),
        parallel: bundle.manifest.parallel.clone(),
        num_gpus: bundle.manifest.num_gpus,
        backend: format!("{:?}", state.execset.backend().class()),
        vendor: format!("{:?}", state.execset.backend().vendor()),
        warmup_requests_per_length: cfg.warmup_requests,
        repetitions_per_length: cfg.repetitions,
        rows,
        artifacts,
        runtime: cfg.runtime,
        engine,
        diagnostics: None,
    })
}

pub async fn run(state: &AppState, cfg: Config) -> Result<Report> {
    if cfg.concurrency == 0 || cfg.requests == 0 || cfg.output_tokens == 0 {
        return Err(RuntimeError::Msg(
            "bench concurrency, requests, and output tokens must be positive".into(),
        ));
    }
    validate_request_layout(&cfg.input, cfg.warmup_requests, cfg.requests)?;
    if cfg.parity_report
        && (!matches!(&cfg.input, Input::TokenIds(_))
            || cfg.concurrency != 1
            || cfg.requests != 1
            || cfg.warmup_requests != 0)
    {
        return Err(RuntimeError::Msg(
            "bench parity report requires token-id input, concurrency=1, requests=1, and no warmup"
                .into(),
        ));
    }
    validate_token_audit_config(&cfg)?;
    if state.execset.backend().vendor().is_none() {
        return Err(RuntimeError::Msg(
            "bench requires a matching GPU backend; refusing CPU fallback performance".into(),
        ));
    }
    let bundle = state.registry.get(&cfg.model)?;
    let vocab = bundle.tokenizer().vocab_size();
    if vocab == 0 {
        return Err(RuntimeError::Msg(
            "bench tokenizer reports an empty vocabulary".into(),
        ));
    }
    validate_input(&cfg.input, vocab)?;
    let mux = state
        .mux(&cfg.model)
        .ok_or_else(|| RuntimeError::Msg(format!("no model mux for '{}'", cfg.model)))?;
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    let engine = {
        let engine = state
            .gpu_engine(&cfg.model)
            .ok_or_else(|| RuntimeError::Msg(format!("no GPU engine for '{}'", cfg.model)))?;
        let engine = engine.lock();
        Some(EngineReport {
            batch_capacity: engine.batch(),
            decode_rungs: engine.decode_rungs(),
        })
    };
    #[cfg(not(any(feature = "cuda", feature = "hsa")))]
    let engine = None;

    if cfg.warmup_requests > 0 {
        crate::obs::Metrics::add(&state.metrics.requests, cfg.warmup_requests as u64);
        let warmup = drive(
            &mux,
            &cfg.input,
            vocab,
            cfg.concurrency,
            cfg.warmup_requests,
            cfg.output_tokens,
            0,
        )
        .await?;
        validate(&warmup, cfg.output_tokens)?;
    }

    let metrics = &state.metrics;
    let batch_count_before = metrics.batch_count.load(Ordering::Relaxed);
    let batch_sum_before = metrics.batch_size_sum.load(Ordering::Relaxed);
    let rejected_before = metrics.rejected.load(Ordering::Relaxed);
    let admit_shed_before = metrics.admit_shed.load(Ordering::Relaxed);
    let rung_switches_before = metrics.decode_rung_switches.load(Ordering::Relaxed);
    crate::obs::Metrics::add(&state.metrics.requests, cfg.requests as u64);
    let started = Instant::now();
    let results = drive(
        &mux,
        &cfg.input,
        vocab,
        cfg.concurrency,
        cfg.requests,
        cfg.output_tokens,
        cfg.warmup_requests,
    )
    .await?;
    let elapsed = started.elapsed();
    validate(&results, cfg.output_tokens)?;

    let prompt_tokens = results.iter().map(|r| r.prompt_tokens).sum::<usize>();
    let output_tokens = results.iter().map(|r| r.ids.len()).sum::<usize>();
    let ttft = results.iter().map(|r| ms(r.ttft)).collect::<Vec<_>>();
    let tpot = results
        .iter()
        .filter_map(|r| r.tpot.map(ms))
        .collect::<Vec<_>>();
    let itl = results
        .iter()
        .flat_map(|r| r.itl.iter().copied().map(ms))
        .collect::<Vec<_>>();
    let e2e = results.iter().map(|r| ms(r.e2e)).collect::<Vec<_>>();
    let secs = elapsed.as_secs_f64();
    let batch_count = metrics
        .batch_count
        .load(Ordering::Relaxed)
        .saturating_sub(batch_count_before);
    let batch_sum = metrics
        .batch_size_sum
        .load(Ordering::Relaxed)
        .saturating_sub(batch_sum_before);
    let input = input_report(&cfg.input);
    let artifacts = artifact_report(
        &bundle.dir,
        state.execset.backend().vendor() == Some(hwspec::Vendor::Amd),
    )?;
    let parity = cfg.parity_report.then(|| ParityReport {
        prompt_token_ids: results
            .iter()
            .map(|result| prompt(&cfg.input, vocab, result.request))
            .collect(),
        output_token_ids: results.iter().map(|result| result.ids.clone()).collect(),
    });
    let token_audit = cfg.token_audit.then(|| TokenAuditReport {
        prompt_token_ids: results
            .iter()
            .map(|result| prompt(&cfg.input, vocab, result.request))
            .collect(),
        output_token_ids: results.iter().map(|result| result.ids.clone()).collect(),
    });

    Ok(Report {
        schema: "plowrt.bench.v1",
        model: cfg.model,
        asset_dir: bundle.dir.display().to_string(),
        target: bundle.manifest.gpu.clone(),
        parallel: bundle.manifest.parallel.clone(),
        num_gpus: bundle.manifest.num_gpus,
        backend: format!("{:?}", state.execset.backend().class()),
        vendor: format!("{:?}", state.execset.backend().vendor()),
        input,
        concurrency: cfg.concurrency,
        warmup_requests: cfg.warmup_requests,
        requests: cfg.requests,
        completed: results.len(),
        failed: 0,
        prompt_tokens,
        output_tokens,
        duration_ms: ms(elapsed),
        request_throughput: results.len() as f64 / secs,
        output_token_throughput: output_tokens as f64 / secs,
        total_token_throughput: (prompt_tokens + output_tokens) as f64 / secs,
        ttft_ms: distribution(ttft).expect("one result"),
        tpot_ms: distribution(tpot),
        itl_ms: distribution(itl),
        e2e_ms: distribution(e2e).expect("one result"),
        output_checksum: checksum(&results),
        artifacts,
        runtime: cfg.runtime,
        engine,
        scheduler: SchedulerReport {
            decode_rung_actual: metrics.decode_rung_actual.load(Ordering::Relaxed),
            decode_rung_admission: metrics.decode_rung_admission.load(Ordering::Relaxed),
            decode_occupied_extent: metrics.decode_occupied_extent.load(Ordering::Relaxed),
            decode_rung_switches: metrics
                .decode_rung_switches
                .load(Ordering::Relaxed)
                .saturating_sub(rung_switches_before),
            batch_count,
            mean_batch_size: if batch_count == 0 {
                0.0
            } else {
                batch_sum as f64 / batch_count as f64
            },
            rejected: metrics
                .rejected
                .load(Ordering::Relaxed)
                .saturating_sub(rejected_before),
            admit_shed: metrics
                .admit_shed
                .load(Ordering::Relaxed)
                .saturating_sub(admit_shed_before),
        },
        diagnostics: None,
        parity,
        token_audit,
    })
}

const MAX_TOKEN_AUDIT_REQUESTS: usize = 64;
const MAX_TOKEN_AUDIT_IDS: usize = 65_536;

fn validate_token_audit_config(cfg: &Config) -> Result<()> {
    if !cfg.token_audit {
        return Ok(());
    }
    validate_request_layout(&cfg.input, cfg.warmup_requests, cfg.requests)?;
    if cfg.parity_report || !matches!(&cfg.input, Input::TokenIds(_) | Input::TokenRows(_)) {
        return Err(RuntimeError::Msg(
            "bench token audit requires exact token-id input and is distinct from parity-report"
                .into(),
        ));
    }
    let prompt_ids = match &cfg.input {
        Input::TokenIds(ids) => ids.len().checked_mul(cfg.requests),
        Input::TokenRows(rows) => rows[cfg.warmup_requests..]
            .iter()
            .try_fold(0usize, |total, row| total.checked_add(row.len())),
        Input::Random { .. } => unreachable!("exact input checked above"),
    };
    let total_ids = cfg
        .output_tokens
        .checked_mul(cfg.requests)
        .and_then(|outputs| prompt_ids.and_then(|prompts| prompts.checked_add(outputs)))
        .ok_or_else(|| RuntimeError::Msg("bench token audit token count overflow".into()))?;
    if cfg.requests > MAX_TOKEN_AUDIT_REQUESTS || total_ids > MAX_TOKEN_AUDIT_IDS {
        return Err(RuntimeError::Msg(format!(
            "bench token audit is bounded to {MAX_TOKEN_AUDIT_REQUESTS} requests and \
             {MAX_TOKEN_AUDIT_IDS} total prompt/output token IDs"
        )));
    }
    Ok(())
}

async fn drive(
    mux: &ModelMux,
    input: &Input,
    vocab: usize,
    concurrency: usize,
    requests: usize,
    output_tokens: usize,
    request_offset: usize,
) -> Result<Vec<RequestResult>> {
    let mut next = 0usize;
    let mut inflight = FuturesUnordered::new();
    let mut out = Vec::with_capacity(requests);
    while next < requests && inflight.len() < concurrency {
        inflight.push(submit(
            mux,
            prompt(input, vocab, request_offset + next),
            output_tokens,
            request_offset + next,
        )?);
        next += 1;
    }
    while let Some(joined) = inflight.next().await {
        let result = joined.map_err(|e| RuntimeError::Msg(format!("bench collector: {e}")))??;
        out.push(result);
        if next < requests {
            inflight.push(submit(
                mux,
                prompt(input, vocab, request_offset + next),
                output_tokens,
                request_offset + next,
            )?);
            next += 1;
        }
    }
    out.sort_unstable_by_key(|r| r.request);
    Ok(out)
}

fn submit(
    mux: &ModelMux,
    prompt_ids: Vec<u32>,
    output_tokens: usize,
    request: usize,
) -> Result<tokio::task::JoinHandle<Result<RequestResult>>> {
    let submitted = Instant::now();
    let (tx, rx) = stream::channel();
    let prompt_tokens = prompt_ids.len();
    let gen = GenParams {
        max_tokens: output_tokens,
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        ignore_eos: true,
    };
    let job = Job {
        prompt_ids,
        gen,
        arrived: submitted,
        respond: tx,
    };
    mux.submit(job).map_err(|e| match e {
        SubmitError::Full(_) => RuntimeError::Rejected("bench request queue full".into()),
        SubmitError::Closed(_) => RuntimeError::Rejected("bench model dispatcher closed".into()),
    })?;
    Ok(tokio::spawn(collect(rx, submitted, prompt_tokens, request)))
}

async fn collect(
    mut rx: stream::ChunkReceiver,
    submitted: Instant,
    prompt_tokens: usize,
    request: usize,
) -> Result<RequestResult> {
    let mut ids = Vec::new();
    let mut first = None;
    let mut previous = None;
    let mut itl = Vec::new();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, .. } => {
                let now = Instant::now();
                if first.is_none() {
                    first = Some(now);
                }
                if let Some(prev) = previous {
                    itl.push(now.duration_since(prev));
                }
                previous = Some(now);
                ids.push(id);
            }
            StreamChunk::Done { reason, usage, .. } => {
                if !matches!(reason, FinishReason::Length) {
                    return Err(RuntimeError::Msg(format!(
                        "bench request ended with {}",
                        reason.as_str()
                    )));
                }
                if usage.prompt_tokens != prompt_tokens || usage.completion_tokens != ids.len() {
                    return Err(RuntimeError::Msg(format!(
                        "bench usage mismatch: expected prompt={prompt_tokens}, observed prompt={} completion={} streamed={}",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        ids.len()
                    )));
                }
                let done = Instant::now();
                let first = first
                    .ok_or_else(|| RuntimeError::Msg("bench completed without a token".into()))?;
                return Ok(RequestResult {
                    request,
                    prompt_tokens,
                    cached_tokens: usage.cached_tokens,
                    tpot: (ids.len() > 1)
                        .then(|| done.duration_since(first) / (ids.len() - 1) as u32),
                    ids,
                    ttft: first.duration_since(submitted),
                    itl,
                    e2e: done.duration_since(submitted),
                });
            }
            StreamChunk::Err(e) => return Err(e),
        }
    }
    Err(RuntimeError::Msg(
        "bench response channel closed without completion".into(),
    ))
}

fn validate_input(input: &Input, vocab: usize) -> Result<()> {
    match input {
        Input::Random { len: 0, .. } => Err(RuntimeError::Msg(
            "bench random input length must be positive".into(),
        )),
        Input::TokenIds(ids) if ids.is_empty() => {
            Err(RuntimeError::Msg("bench token-id input is empty".into()))
        }
        Input::TokenIds(ids) => {
            if let Some(&id) = ids.iter().find(|&&id| id as usize >= vocab) {
                return Err(RuntimeError::Msg(format!(
                    "bench token id {id} is outside vocabulary size {vocab}"
                )));
            }
            Ok(())
        }
        Input::TokenRows(rows) => {
            for (row_index, row) in rows.iter().enumerate() {
                if row.is_empty() {
                    return Err(RuntimeError::Msg(format!(
                        "bench prompt row {} is empty",
                        row_index + 1
                    )));
                }
                if let Some(&id) = row.iter().find(|&&id| id as usize >= vocab) {
                    return Err(RuntimeError::Msg(format!(
                        "bench token id {id} in prompt row {} is outside vocabulary size {vocab}",
                        row_index + 1
                    )));
                }
            }
            Ok(())
        }
        Input::Random { .. } => Ok(()),
    }
}

fn reject_cached_prefill(results: &[RequestResult]) -> Result<()> {
    if let Some(result) = results.iter().find(|result| result.cached_tokens != 0) {
        return Err(RuntimeError::Msg(format!(
            "bench prefill sweep request {} used {} cached prompt tokens; refusing cache-hit TTFT",
            result.request, result.cached_tokens
        )));
    }
    Ok(())
}

fn input_report(input: &Input) -> InputReport {
    match input {
        Input::TokenIds(ids) => InputReport {
            mode: "token_ids",
            tokens_per_request: Some(ids.len()),
            row_count: 1,
            min_tokens_per_request: ids.len(),
            max_tokens_per_request: ids.len(),
            seed: None,
        },
        Input::TokenRows(rows) => InputReport {
            mode: "token_rows",
            tokens_per_request: None,
            row_count: rows.len(),
            min_tokens_per_request: rows.iter().map(Vec::len).min().unwrap_or(0),
            max_tokens_per_request: rows.iter().map(Vec::len).max().unwrap_or(0),
            seed: None,
        },
        Input::Random { len, seed } => InputReport {
            mode: "random",
            tokens_per_request: Some(*len),
            row_count: 1,
            min_tokens_per_request: *len,
            max_tokens_per_request: *len,
            seed: Some(*seed),
        },
    }
}

fn single_input_len(input: &Input) -> usize {
    match input {
        Input::TokenIds(ids) => ids.len(),
        Input::TokenRows(_) => unreachable!("prompt rows are rejected by prefill sweep"),
        Input::Random { len, .. } => *len,
    }
}

fn prompt(input: &Input, vocab: usize, request: usize) -> Vec<u32> {
    match input {
        Input::TokenIds(ids) => ids.clone(),
        Input::TokenRows(rows) => rows[request].clone(),
        Input::Random { len, seed } => {
            let mut x = seed
                .wrapping_add(request as u64)
                .wrapping_add(0x9e37_79b9_7f4a_7c15);
            (0..*len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % vocab as u64) as u32
                })
                .collect()
        }
    }
}

fn validate(results: &[RequestResult], output_tokens: usize) -> Result<()> {
    if let Some((i, got)) = results
        .iter()
        .enumerate()
        .find_map(|(i, r)| (r.ids.len() != output_tokens).then_some((i, r.ids.len())))
    {
        return Err(RuntimeError::Msg(format!(
            "bench request {i} produced {got}/{output_tokens} tokens"
        )));
    }
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn distribution(mut xs: Vec<f64>) -> Option<Distribution> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    Some(Distribution {
        mean,
        p50: percentile(&xs, 0.50),
        p90: percentile(&xs, 0.90),
        p99: percentile(&xs, 0.99),
        min: xs[0],
        max: xs[xs.len() - 1],
    })
}

fn percentile(xs: &[f64], q: f64) -> f64 {
    let rank = (q * xs.len() as f64).ceil().clamp(1.0, xs.len() as f64) as usize;
    xs[rank - 1]
}

fn artifact_report(assets: &Path, amd: bool) -> Result<ArtifactReport> {
    let packet_path = crate::asset::devblob::DevBlob::find_in_dir(assets)?
        .ok_or_else(|| RuntimeError::Device(format!("no PLOWDEV blob in {}", assets.display())))?;
    let packet = file_identity(&packet_path)?;
    let weights_manifest = file_identity(&assets.join("weights.json"))?;
    let build_path = assets.join("build.json");
    let build_manifest = build_path
        .exists()
        .then(|| file_identity(&build_path))
        .transpose()?;
    let runtime = crate::config::RuntimeConfig::get();
    let mut object_paths = Vec::new();
    let mut object_dirs = vec![assets.join("cubin")];
    if amd {
        object_dirs.push(
            runtime
                .amd
                .hsaco
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| assets.join("hsaco")),
        );
        if let Some(spec) = runtime.amd.hsaco_lowrung.as_deref() {
            for entry in spec.split(',').filter(|entry| !entry.is_empty()) {
                let dir = entry
                    .rsplit_once(':')
                    .filter(|(_, max)| max.parse::<u32>().is_ok())
                    .map_or(entry, |(dir, _)| dir);
                object_dirs.push(PathBuf::from(dir));
            }
        }
    }
    object_dirs.sort();
    object_dirs.dedup();
    for dir in object_dirs {
        if !dir.exists() {
            continue;
        }
        let entries = fs::read_dir(&dir).map_err(|source| RuntimeError::Io {
            path: dir.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| RuntimeError::Io {
                path: dir.clone(),
                source,
            })?;
            let path = entry.path();
            if path.is_file()
                && matches!(
                    path.extension().and_then(|x| x.to_str()),
                    Some("elf" | "hsaco" | "co" | "cubin")
                )
            {
                object_paths.push(path);
            }
        }
    }
    if !amd {
        for path in [
            runtime.nv.cubin.as_deref(),
            runtime.nv.cubin_pf.as_deref(),
            runtime.nv.cubin_sample.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            let path = PathBuf::from(path);
            if path.is_file() {
                object_paths.push(path);
            }
        }
    }
    object_paths.sort();
    object_paths.dedup();
    let object_inventory = object_paths
        .iter()
        .map(|path| file_identity(path))
        .collect::<Result<Vec<_>>>()?;
    let checkpoint_path = runtime
        .checkpoint
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| assets.join("checkpoint"));
    let checkpoint = checkpoint_path
        .exists()
        .then(|| checkpoint_layout(&checkpoint_path))
        .transpose()?;
    Ok(ArtifactReport {
        packet,
        weights_manifest,
        build_manifest,
        object_inventory,
        checkpoint,
    })
}

fn file_identity(path: &Path) -> Result<FileIdentity> {
    let mut file = fs::File::open(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bytes = file
        .metadata()
        .map_err(|source| RuntimeError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let checksum = format!(
        "fnv1a64:{:016x}",
        hash_reader(&mut file).map_err(|source| {
            RuntimeError::Io {
                path: path.to_path_buf(),
                source,
            }
        })?
    );
    Ok(FileIdentity {
        path: path.display().to_string(),
        bytes,
        checksum,
    })
}

fn checkpoint_layout(path: &Path) -> Result<CheckpointLayout> {
    let resolved = fs::canonicalize(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let entries = fs::read_dir(&resolved).map_err(|source| RuntimeError::Io {
        path: resolved.clone(),
        source,
    })?;
    let mut shards = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|source| RuntimeError::Io {
                path: resolved.clone(),
                source,
            })?
            .path();
        if path.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            shards.push(path);
        }
    }
    shards.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let mut hash = FNV_OFFSET;
    let mut bytes = 0u64;
    for shard in &shards {
        let len = fs::metadata(shard)
            .map_err(|source| RuntimeError::Io {
                path: shard.clone(),
                source,
            })?
            .len();
        bytes = bytes.saturating_add(len);
        hash = hash_bytes(
            hash,
            shard.file_name().unwrap_or_default().as_encoded_bytes(),
        );
        hash = hash_bytes(hash, &len.to_le_bytes());
    }
    Ok(CheckpointLayout {
        path: resolved.display().to_string(),
        safetensor_shards: shards.len(),
        bytes,
        layout_checksum: format!("fnv1a64-layout:{hash:016x}"),
    })
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

fn hash_reader(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut hash = FNV_OFFSET;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(hash);
        }
        hash = hash_bytes(hash, &buf[..n]);
    }
}

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn checksum(results: &[RequestResult]) -> String {
    let mut h = FNV_OFFSET;
    for r in results {
        for &id in &r.ids {
            for b in id.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
    }
    format!("fnv1a64:{h:016x}")
}

fn prompt_checksum(input: &Input, vocab: usize, requests: std::ops::Range<usize>) -> String {
    let mut hash = FNV_OFFSET;
    for request in requests {
        for id in prompt(input, vocab, request) {
            hash = hash_bytes(hash, &id.to_le_bytes());
        }
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_result(request: usize, cached_tokens: usize) -> RequestResult {
        RequestResult {
            request,
            prompt_tokens: 4,
            cached_tokens,
            ids: vec![7],
            ttft: Duration::from_millis(2),
            tpot: None,
            itl: Vec::new(),
            e2e: Duration::from_millis(2),
        }
    }

    fn token_audit_config(input: Input, requests: usize, output_tokens: usize) -> Config {
        Config {
            model: "model".into(),
            input,
            concurrency: 1,
            warmup_requests: 0,
            requests,
            output_tokens,
            runtime: serde_json::Value::Null,
            parity_report: false,
            token_audit: true,
        }
    }

    #[test]
    fn token_audit_config_is_exact_and_bounded() {
        assert!(validate_token_audit_config(&token_audit_config(
            Input::TokenIds(vec![1, 2, 3]),
            4,
            8,
        ))
        .is_ok());
        assert!(validate_token_audit_config(&token_audit_config(
            Input::Random { len: 3, seed: 0 },
            4,
            8,
        ))
        .is_err());
        assert!(
            validate_token_audit_config(&token_audit_config(Input::TokenIds(vec![1]), 65, 8,))
                .is_err()
        );
        let mut ragged = token_audit_config(
            Input::TokenRows(vec![vec![9], vec![1, 2], vec![3, 4, 5]]),
            2,
            8,
        );
        ragged.warmup_requests = 1;
        assert!(validate_token_audit_config(&ragged).is_ok());
        if let Input::TokenRows(rows) = &mut ragged.input {
            rows.pop();
        }
        assert!(validate_token_audit_config(&ragged).is_err());
    }

    #[test]
    fn prompt_rows_parse_strict_exact_rows() {
        assert_eq!(
            parse_prompt_rows("1, 2,3\n4\n5,6\n").unwrap(),
            [vec![1, 2, 3], vec![4], vec![5, 6]]
        );
        for invalid in ["", "1\n\n2", "1,,2", "1,x"] {
            assert!(parse_prompt_rows(invalid).is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn ragged_rows_map_directly_by_global_request_index() {
        let input = Input::TokenRows(vec![vec![7], vec![8, 9], vec![10, 11, 12]]);
        validate_request_layout(&input, 1, 2).unwrap();
        assert_eq!(prompt(&input, 100, 0), [7]);
        assert_eq!(prompt(&input, 100, 1), [8, 9]);
        assert_eq!(prompt(&input, 100, 2), [10, 11, 12]);
        assert!(validate_request_layout(&input, 0, 2).is_err());
    }

    #[test]
    fn ragged_input_report_is_truthful() {
        let report = input_report(&Input::TokenRows(vec![vec![1], vec![2, 3, 4], vec![5, 6]]));
        assert_eq!(report.mode, "token_rows");
        assert_eq!(report.tokens_per_request, None);
        assert_eq!(report.row_count, 3);
        assert_eq!(report.min_tokens_per_request, 1);
        assert_eq!(report.max_tokens_per_request, 3);
        assert_eq!(report.seed, None);
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["tokens_per_request"], serde_json::Value::Null);
        assert_eq!(json["row_count"], 3);
        assert_eq!(json["min_tokens_per_request"], 1);
        assert_eq!(json["max_tokens_per_request"], 3);
    }

    #[test]
    fn deterministic_random_is_bounded_and_request_specific() {
        let input = Input::Random { len: 64, seed: 7 };
        let a = prompt(&input, 100, 3);
        let b = prompt(&input, 100, 3);
        let c = prompt(&input, 100, 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|&x| x < 100));
    }

    #[test]
    fn distribution_uses_nearest_rank_percentiles() {
        let d = distribution(vec![4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(d.mean, 2.5);
        assert_eq!(d.p50, 2.0);
        assert_eq!(d.p90, 4.0);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.max, 4.0);
    }

    #[test]
    fn artifact_checksum_is_deterministic_and_content_sensitive() {
        let mut a = std::io::Cursor::new(b"packet".as_slice());
        let mut b = std::io::Cursor::new(b"packet".as_slice());
        let mut c = std::io::Cursor::new(b"object".as_slice());
        let packet = hash_reader(&mut a).unwrap();
        assert_eq!(packet, hash_reader(&mut b).unwrap());
        assert_ne!(packet, hash_reader(&mut c).unwrap());
    }

    #[test]
    fn prefill_sweep_rejects_any_cache_hit() {
        assert!(reject_cached_prefill(&[request_result(0, 0)]).is_ok());
        let err = reject_cached_prefill(&[request_result(4, 3)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("request 4 used 3 cached prompt tokens"));
    }

    #[test]
    fn measured_prompt_checksum_records_request_variation() {
        let input = Input::Random { len: 8, seed: 11 };
        let a = prompt_checksum(&input, 100, 0..3);
        assert_eq!(a, prompt_checksum(&input, 100, 0..3));
        assert_ne!(a, prompt_checksum(&input, 100, 1..4));
    }

    #[test]
    fn diagnostic_capture_preserves_selection_order_and_boundaries() {
        let mut diagnostics = EngineDiagnostics {
            supported: true,
            complete: true,
            overflowed: false,
            scope: "warmup_and_measured",
            prefill_selections: Vec::new(),
            decode_selections: Vec::new(),
            rank_agreement: None,
        };
        let a = PrefillSelection {
            slot: 2,
            row_start: 0,
            rows: 512,
            bucket: 512,
        };
        let b = PrefillSelection {
            slot: 2,
            row_start: 512,
            rows: 17,
            bucket: 128,
        };
        diagnostics.push_prefill(a.clone());
        diagnostics.push_prefill(b.clone());
        diagnostics.push_decode(DecodeSelection {
            occupied_rows: 3,
            bucket: 4,
            steps: 8,
        });
        assert_eq!(diagnostics.prefill_selections, [a, b]);
        assert_eq!(
            diagnostics.decode_selections,
            [DecodeSelection {
                occupied_rows: 3,
                bucket: 4,
                steps: 8,
            }]
        );
        assert!(diagnostics.complete);
    }

    #[test]
    fn diagnostic_capture_overflow_is_fail_closed() {
        let mut diagnostics = EngineDiagnostics::unsupported();
        diagnostics.supported = true;
        diagnostics.complete = true;
        for _ in 0..=MAX_DIAGNOSTIC_SELECTIONS {
            diagnostics.push_decode(DecodeSelection {
                occupied_rows: 1,
                bucket: 1,
                steps: 1,
            });
        }
        assert!(diagnostics.overflowed);
        assert!(!diagnostics.complete);
        assert_eq!(
            diagnostics.decode_selections.len(),
            MAX_DIAGNOSTIC_SELECTIONS
        );
        assert!(validate_engine_diagnostics(diagnostics).is_err());
    }

    #[test]
    fn diagnostic_validation_rejects_unsupported_and_incomplete_reports() {
        let unsupported = EngineDiagnostics::unsupported();
        assert!(validate_engine_diagnostics(unsupported).is_err());

        let mut incomplete = EngineDiagnostics::unsupported();
        incomplete.supported = true;
        assert!(validate_engine_diagnostics(incomplete).is_err());
    }
}
