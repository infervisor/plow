use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        DefaultBodyLimit, Multipart, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};

use super::{
    frontend::{decode_wav, AudioError, MAX_SAMPLES, SAMPLE_RATE},
    FinalizationPolicy, Transcriber, Transcript, TranscriptionInput,
};

const BATCH_FORMATION_WINDOW: Duration = Duration::from_millis(5);

pub struct AsrServer {
    pub model: String,
    mux: AsrMux,
    uploads: Arc<Semaphore>,
    sessions: Arc<Semaphore>,
    next_session: AtomicU64,
    finalization: FinalizationPolicy,
}

#[derive(Clone)]
struct AsrMux {
    tx: mpsc::Sender<AsrJob>,
}

struct AsrJob {
    samples: Vec<f32>,
    language: Option<String>,
    context: String,
    cancel: Arc<AtomicBool>,
    respond: oneshot::Sender<crate::Result<Transcript>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmitError {
    Full,
    Closed,
}

impl AsrMux {
    fn spawn(mut engine: Box<dyn Transcriber>) -> (Self, usize, FinalizationPolicy) {
        let batch_capacity = engine.batch_capacity().max(1);
        let finalization = engine.finalization_policy();
        let ingress_capacity = batch_capacity.saturating_mul(4).max(4);
        let (tx, mut rx) = mpsc::channel::<AsrJob>(ingress_capacity);
        std::thread::Builder::new()
            .name("plow-asr-engine".into())
            .spawn(move || {
                let mut cohort = VecDeque::with_capacity(batch_capacity);
                while let Some(first) = rx.blocking_recv() {
                    cohort.push_back(first);
                    if batch_capacity > 1 {
                        let deadline = Instant::now() + BATCH_FORMATION_WINDOW;
                        while cohort.len() < batch_capacity {
                            match rx.try_recv() {
                                Ok(job) => cohort.push_back(job),
                                Err(mpsc::error::TryRecvError::Disconnected) => break,
                                Err(mpsc::error::TryRecvError::Empty) => {
                                    let remaining =
                                        deadline.saturating_duration_since(Instant::now());
                                    if remaining.is_zero() {
                                        break;
                                    }
                                    std::thread::sleep(remaining.min(Duration::from_micros(100)));
                                }
                            }
                        }
                        cohort
                            .make_contiguous()
                            .sort_by_key(|job| std::cmp::Reverse(job.samples.len()));
                    }
                    let requests: Vec<_> = cohort
                        .iter()
                        .map(|job| TranscriptionInput {
                            samples: &job.samples,
                            language: job.language.as_deref(),
                            context: &job.context,
                            cancel: &job.cancel,
                        })
                        .collect();
                    match engine.transcribe_batch(&requests) {
                        Ok(results) if results.len() == cohort.len() => {
                            for (job, result) in cohort.drain(..).zip(results) {
                                let _ = job.respond.send(result);
                            }
                        }
                        Ok(results) => {
                            let error = crate::RuntimeError::Msg(format!(
                                "ASR engine returned {} results for {} requests",
                                results.len(),
                                cohort.len()
                            ));
                            fanout_batch_error(&mut cohort, error);
                        }
                        Err(error) => fanout_batch_error(&mut cohort, error),
                    }
                }
            })
            .expect("failed to spawn ASR engine thread");
        (Self { tx }, ingress_capacity, finalization)
    }

    fn submit(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
        context: String,
        cancel: Arc<AtomicBool>,
    ) -> Result<oneshot::Receiver<crate::Result<Transcript>>, SubmitError> {
        let (respond, receive) = oneshot::channel();
        let job = AsrJob {
            samples,
            language,
            context,
            cancel,
            respond,
        };
        self.tx.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => SubmitError::Full,
            mpsc::error::TrySendError::Closed(_) => SubmitError::Closed,
        })?;
        Ok(receive)
    }
}

fn fanout_batch_error(cohort: &mut VecDeque<AsrJob>, error: crate::RuntimeError) {
    let message = error.to_string();
    for job in cohort.drain(..) {
        let error = match &error {
            crate::RuntimeError::Rejected(reason) => crate::RuntimeError::Rejected(reason.clone()),
            crate::RuntimeError::ContextLength(reason) => {
                crate::RuntimeError::ContextLength(reason.clone())
            }
            crate::RuntimeError::DeviceFault { info } => {
                crate::RuntimeError::DeviceFault { info: info.clone() }
            }
            _ => crate::RuntimeError::Msg(message.clone()),
        };
        let _ = job.respond.send(Err(error));
    }
}

struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl AsrServer {
    pub fn new(model: String, engine: impl Transcriber + 'static) -> Arc<Self> {
        let (mux, ingress_capacity, finalization) = AsrMux::spawn(Box::new(engine));
        Arc::new(Self {
            model,
            mux,
            uploads: Arc::new(Semaphore::new(ingress_capacity)),
            sessions: Arc::new(Semaphore::new(ingress_capacity)),
            next_session: AtomicU64::new(1),
            finalization,
        })
    }
    pub fn router(self: Arc<Self>, websocket: bool) -> Router {
        let mut router = Router::new()
            .route("/v1/audio/transcriptions", post(transcription))
            .route("/health", get(|| async { StatusCode::OK }));
        if websocket {
            router = router.route("/v1/audio/transcriptions/stream", get(upgrade));
        }
        router
            .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
            .with_state(self)
    }
}

fn failure(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"transcription_error"}})),
    )
        .into_response()
}

fn runtime_failure(error: crate::RuntimeError) -> Response {
    let status = match &error {
        crate::RuntimeError::Rejected(_) | crate::RuntimeError::ContextLength(_) => {
            StatusCode::BAD_REQUEST
        }
        crate::RuntimeError::DeviceFault { ref info } if info.fatal => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    failure(status, error)
}

async fn transcription(State(state): State<Arc<AsrServer>>, mut multipart: Multipart) -> Response {
    let upload = match state.uploads.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return failure(StatusCode::TOO_MANY_REQUESTS, "too many ASR uploads"),
    };
    let mut file = None;
    let mut fields = std::collections::HashMap::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let field = match tokio::time::timeout_at(deadline, multipart.next_field()).await {
            Ok(Ok(Some(f))) => f,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => return failure(e.status(), e),
            Err(_) => return failure(StatusCode::REQUEST_TIMEOUT, "upload timed out"),
        };
        let name = field.name().unwrap_or("").to_owned();
        if name == "file" {
            if file.is_some() {
                return failure(StatusCode::BAD_REQUEST, "duplicate file");
            }
            file = match tokio::time::timeout_at(deadline, field.bytes()).await {
                Ok(Ok(b)) => Some(b),
                Ok(Err(e)) => return failure(e.status(), e),
                Err(_) => return failure(StatusCode::REQUEST_TIMEOUT, "upload timed out"),
            };
        } else {
            if !matches!(
                name.as_str(),
                "model" | "language" | "prompt" | "response_format" | "temperature"
            ) || fields.contains_key(&name)
            {
                return failure(StatusCode::BAD_REQUEST, "unknown or duplicate field");
            }
            let value = match tokio::time::timeout_at(deadline, field.text()).await {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => return failure(e.status(), e),
                Err(_) => return failure(StatusCode::REQUEST_TIMEOUT, "upload timed out"),
            };
            fields.insert(name, value);
        }
    }
    let Some(model) = fields.get("model") else {
        return failure(StatusCode::BAD_REQUEST, "model is required");
    };
    if model != &state.model {
        return failure(StatusCode::NOT_FOUND, "unknown ASR model");
    }
    let format = fields
        .remove("response_format")
        .unwrap_or_else(|| "json".into());
    if !matches!(format.as_str(), "json" | "text") {
        return failure(
            StatusCode::BAD_REQUEST,
            "response_format must be json or text",
        );
    }
    if fields
        .get("temperature")
        .is_some_and(|v| v.parse::<f32>().ok() != Some(0.0))
    {
        return failure(StatusCode::BAD_REQUEST, "ASR supports greedy decoding only");
    }
    let Some(file) = file else {
        return failure(StatusCode::BAD_REQUEST, "file is required");
    };
    let language = fields.remove("language");
    let context = fields.remove("prompt").unwrap_or_default();
    let samples = match tokio::task::spawn_blocking(move || decode_wav(&file)).await {
        Ok(Ok(samples)) => samples,
        Ok(Err(error)) => {
            let status = match &error {
                AudioError::Invalid(_) => StatusCode::BAD_REQUEST,
                AudioError::Unsupported(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
                AudioError::TooLong => StatusCode::PAYLOAD_TOO_LARGE,
            };
            return failure(status, error);
        }
        Err(error) => return failure(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    drop(upload);
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let work = match state
        .mux
        .submit(samples, language, context, cancel.0.clone())
    {
        Ok(work) => work,
        Err(SubmitError::Full) => return failure(StatusCode::TOO_MANY_REQUESTS, "ASR queue full"),
        Err(SubmitError::Closed) => {
            return failure(StatusCode::SERVICE_UNAVAILABLE, "ASR engine unavailable")
        }
    };
    match work.await {
        Ok(Ok(result)) if format == "text" => result.text.into_response(),
        Ok(Ok(result)) => Json(json!({"text":result.text})).into_response(),
        Ok(Err(error)) => runtime_failure(error),
        Err(error) => failure(StatusCode::SERVICE_UNAVAILABLE, error),
    }
}

async fn upgrade(State(state): State<Arc<AsrServer>>, ws: WebSocketUpgrade) -> Response {
    let permit = match state.sessions.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return failure(StatusCode::TOO_MANY_REQUESTS, "too many ASR sessions"),
    };
    ws.max_message_size(65536)
        .max_frame_size(65536)
        .on_upgrade(move |socket| async move {
            stream(state, socket, permit).await;
        })
        .into_response()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Start {
    #[serde(rename = "type")]
    kind: String,
    version: u32,
    model: String,
    sample_rate: u32,
    format: String,
    language: Option<String>,
    #[serde(default)]
    prompt: String,
}

async fn send(socket: &mut WebSocket, value: serde_json::Value) -> bool {
    matches!(
        tokio::time::timeout(
            Duration::from_secs(30),
            socket.send(Message::Text(value.to_string()))
        )
        .await,
        Ok(Ok(()))
    )
}

fn append_final_padding(samples: &mut Vec<f32>, count: usize, amplitude: f32) {
    if amplitude == 0.0 {
        samples.resize(samples.len() + count, 0.0);
        return;
    }
    let mut state = 0x9e3779b97f4a7c15u64;
    samples.reserve(count);
    for _ in 0..count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let centered = ((state >> 48) as i32 - 32768) as f32 / 32768.0;
        samples.push(centered * amplitude);
    }
}

async fn stream(state: Arc<AsrServer>, mut socket: WebSocket, _permit: OwnedSemaphorePermit) {
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let Some(Ok(Message::Text(text))) =
        tokio::time::timeout(Duration::from_secs(30), socket.recv())
            .await
            .ok()
            .flatten()
    else {
        return;
    };
    let start = match serde_json::from_str::<Start>(&text) {
        Ok(s)
            if s.kind == "start"
                && s.version == 1
                && s.model == state.model
                && s.sample_rate == SAMPLE_RATE
                && s.format == "pcm_s16le" =>
        {
            s
        }
        _ => {
            send(
                &mut socket,
                json!({"type":"error","message":"invalid start","terminal":true}),
            )
            .await;
            return;
        }
    };
    let language = start.language;
    let session = state.next_session.fetch_add(1, Ordering::Relaxed);
    let max_audio_samples = MAX_SAMPLES.saturating_sub(state.finalization.final_padding_samples);
    let initial_credit = 16000usize.min(max_audio_samples);
    if !send(&mut socket,json!({"type":"ready","version":1,"session_id":session.to_string(),
        "sample_rate":SAMPLE_RATE,"format":"pcm_s16le","max_chunk_bytes":32000,"credit_samples":initial_credit,
        "max_audio_samples":max_audio_samples,"partial_mode":"final_only"})).await{return;}
    let mut samples = Vec::new();
    let mut sequence = 0u64;
    let mut credit = initial_credit;
    loop {
        let message = match tokio::time::timeout(Duration::from_secs(30), socket.recv()).await {
            Ok(Some(Ok(m))) => m,
            _ => return,
        };
        let finish = match message {
            Message::Binary(bytes) => {
                if bytes.len() < 10
                    || bytes.len() > 32008
                    || (bytes.len() - 8) % 2 != 0
                    || u64::from_le_bytes(bytes[..8].try_into().unwrap()) != sequence
                    || (bytes.len() - 8) / 2 > credit
                    || samples.len() + (bytes.len() - 8) / 2 > max_audio_samples
                {
                    send(&mut socket,json!({"type":"error","message":"invalid PCM sequence or credit/length exceeded","terminal":true})).await;
                    return;
                }
                sequence += 1;
                credit -= (bytes.len() - 8) / 2;
                samples.extend(
                    bytes[8..]
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
                );
                false
            }
            Message::Text(text) => match serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v["type"].as_str().map(str::to_owned))
                .as_deref()
            {
                Some("finish") => true,
                Some("cancel") => return,
                _ => {
                    send(&mut socket,json!({"type":"error","message":"expected finish or cancel","terminal":true})).await;
                    return;
                }
            },
            Message::Close(_) => return,
            Message::Ping(_) | Message::Pong(_) => continue,
        };
        if finish {
            if samples.len() < SAMPLE_RATE as usize / 2 {
                send(&mut socket,json!({"type":"error","message":"audio must contain at least 0.5 seconds","terminal":true})).await;
                return;
            }
            append_final_padding(
                &mut samples,
                state.finalization.final_padding_samples,
                state.finalization.final_padding_amplitude,
            );
            let mut work = match state
                .mux
                .submit(samples, language, start.prompt, cancel.0.clone())
            {
                Ok(work) => work,
                Err(SubmitError::Full) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":"ASR queue full","terminal":true}),
                    )
                    .await;
                    return;
                }
                Err(SubmitError::Closed) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":"ASR engine unavailable","terminal":true}),
                    )
                    .await;
                    return;
                }
            };
            let result = loop {
                tokio::select! {
                    result=&mut work=>break result,
                    incoming=socket.recv()=>match incoming {
                        Some(Ok(Message::Ping(_)|Message::Pong(_)))=>{},
                        other=>{
                            let cancelled=matches!(&other,Some(Ok(Message::Text(text))) if serde_json::from_str::<serde_json::Value>(text)
                                .ok().is_some_and(|v|v["type"]=="cancel"));
                            cancel.0.store(true,Ordering::Relaxed);
                            let _=work.await;
                            if !cancelled {send(&mut socket,json!({"type":"error","message":"input sent after finish or invalid event","terminal":true})).await;}
                            return;
                        }
                    }
                }
            };
            match result {
                Ok(Ok(result)) => {
                    send(
                        &mut socket,
                        json!({"type":"final","revision":1,
                        "text":result.text,"language":result.language,
                        "stable_prefix_bytes":result.text.len()}),
                    )
                    .await;
                }
                Ok(Err(error)) => {
                    send(
                        &mut socket,
                        json!({"type":"error","message":error.to_string(),"terminal":true}),
                    )
                    .await;
                }
                Err(_) => {
                    send(&mut socket,json!({"type":"error","message":"ASR engine response channel closed","terminal":true})).await;
                }
            }
            return;
        }
        let grant = (16000 - samples.len() % 16000)
            .min(max_audio_samples - samples.len())
            .saturating_sub(credit);
        credit += grant;
        if grant > 0 && !send(&mut socket, json!({"type":"credit","credit_samples":grant})).await {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use futures::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn final_padding_noise_is_bounded_and_deterministic() {
        let mut left = vec![0.5];
        let mut right = left.clone();
        append_final_padding(&mut left, 1024, 100.0 / 32768.0);
        append_final_padding(&mut right, 1024, 100.0 / 32768.0);
        assert_eq!(left, right);
        assert!(left[1..].iter().any(|sample| *sample != 0.0));
        assert!(left[1..]
            .iter()
            .all(|sample| sample.abs() <= 100.0 / 32768.0));
    }

    #[tokio::test]
    async fn mux_forms_one_batch_and_preserves_member_results() {
        struct Batched {
            sizes: Arc<std::sync::Mutex<Vec<usize>>>,
            sample_lengths: Arc<std::sync::Mutex<Vec<usize>>>,
        }
        impl Transcriber for Batched {
            fn language(&self, language: Option<&str>) -> crate::Result<Option<String>> {
                Ok(language.map(str::to_owned))
            }
            fn batch_capacity(&self) -> usize {
                4
            }
            fn transcribe(
                &mut self,
                _: &[f32],
                _: Option<&str>,
                _: &str,
                _: &AtomicBool,
            ) -> crate::Result<Transcript> {
                unreachable!("batch override must be used")
            }
            fn transcribe_batch(
                &mut self,
                requests: &[TranscriptionInput<'_>],
            ) -> crate::Result<Vec<crate::Result<Transcript>>> {
                self.sizes.lock().unwrap().push(requests.len());
                self.sample_lengths
                    .lock()
                    .unwrap()
                    .extend(requests.iter().map(|request| request.samples.len()));
                Ok(requests
                    .iter()
                    .map(|request| {
                        if request.cancel.load(Ordering::Relaxed) {
                            Err(crate::RuntimeError::Rejected("cancelled".into()))
                        } else {
                            Ok(Transcript {
                                text: request.context.to_owned(),
                                language: request.language.map(str::to_owned),
                            })
                        }
                    })
                    .collect())
            }
        }
        let sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sample_lengths = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mux, _, _) = AsrMux::spawn(Box::new(Batched {
            sizes: sizes.clone(),
            sample_lengths: sample_lengths.clone(),
        }));
        let cancels: Vec<_> = (0..4).map(|i| Arc::new(AtomicBool::new(i == 2))).collect();
        let mut replies = Vec::new();
        for (i, cancel) in cancels.into_iter().enumerate() {
            replies.push(
                mux.submit(
                    vec![i as f32; i + 1],
                    Some(format!("language-{i}")),
                    format!("request-{i}"),
                    cancel,
                )
                .unwrap(),
            );
        }
        for (i, reply) in replies.into_iter().enumerate() {
            let result = reply.await.unwrap();
            if i == 2 {
                assert!(result.is_err());
            } else {
                let result = result.unwrap();
                assert_eq!(result.text, format!("request-{i}"));
                assert_eq!(
                    result.language.as_deref(),
                    Some(format!("language-{i}").as_str())
                );
            }
        }
        assert_eq!(*sizes.lock().unwrap(), [4]);
        assert_eq!(*sample_lengths.lock().unwrap(), [4, 3, 2, 1]);
    }

    #[tokio::test]
    async fn mux_rejects_only_after_its_bounded_queue_is_full() {
        struct Held {
            started: Arc<tokio::sync::Notify>,
            release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        }
        impl Transcriber for Held {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(
                &mut self,
                _: &[f32],
                _: Option<&str>,
                _: &str,
                _: &AtomicBool,
            ) -> crate::Result<Transcript> {
                self.started.notify_one();
                let (lock, condition) = &*self.release;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = condition.wait(released).unwrap();
                }
                Ok(Transcript {
                    text: "done".into(),
                    language: None,
                })
            }
        }
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let (mux, ingress_capacity, _) = AsrMux::spawn(Box::new(Held {
            started: started.clone(),
            release: release.clone(),
        }));
        let submit = || {
            mux.submit(
                vec![0.0],
                None,
                String::new(),
                Arc::new(AtomicBool::new(false)),
            )
        };
        let active = submit().unwrap();
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        let queued: Vec<_> = (0..ingress_capacity).map(|_| submit().unwrap()).collect();
        assert_eq!(submit().unwrap_err(), SubmitError::Full);
        *release.0.lock().unwrap() = true;
        release.1.notify_one();
        assert!(active.await.unwrap().is_ok());
        for reply in queued {
            assert!(reply.await.unwrap().is_ok());
        }
    }

    struct Fake;
    impl Transcriber for Fake {
        fn language(&self, language: Option<&str>) -> crate::Result<Option<String>> {
            if language.is_some_and(|l| l != "English") {
                return Err(crate::RuntimeError::Rejected("language".into()));
            }
            Ok(language.map(str::to_owned))
        }
        fn transcribe(
            &mut self,
            samples: &[f32],
            language: Option<&str>,
            _: &str,
            cancel: &AtomicBool,
        ) -> crate::Result<Transcript> {
            assert!(samples.len() >= 8000);
            std::thread::sleep(Duration::from_millis(30));
            if cancel.load(Ordering::Relaxed) {
                return Err(crate::RuntimeError::Rejected("cancelled".into()));
            }
            Ok(Transcript {
                text: "hello".into(),
                language: self.language(language)?,
            })
        }
    }

    fn request(model: &str, format: &str) -> Request<Body> {
        request_wav(model, format, 16000, 8000)
    }

    fn request_wav(model: &str, format: &str, sample_rate: u32, samples: usize) -> Request<Body> {
        let mut wav = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut wav,
            hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..samples {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        let mut body=format!("--audio\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--audio\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{format}\r\n--audio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n").into_bytes();
        body.extend(wav.into_inner());
        body.extend(b"\r\n--audio--\r\n");
        Request::post("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=audio")
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn offline_api_formats_and_admission() {
        let server = AsrServer::new("test".into(), Fake);
        let app = server.clone().router(false);
        for (format, expected) in [("json", r#"{"text":"hello"}"#), ("text", "hello")] {
            let response = app.clone().oneshot(request("test", format)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.into_body().collect().await.unwrap().to_bytes(),
                expected
            );
        }
        assert_eq!(
            app.clone()
                .oneshot(request("missing", "json"))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            app.clone()
                .oneshot(request("test", "srt"))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        for (rate, samples, status) in [
            (8000, 8000, StatusCode::UNSUPPORTED_MEDIA_TYPE),
            (16000, 7999, StatusCode::BAD_REQUEST),
            (16000, 480001, StatusCode::PAYLOAD_TOO_LARGE),
        ] {
            assert_eq!(
                app.clone()
                    .oneshot(request_wav("test", "json", rate, samples))
                    .await
                    .unwrap()
                    .status(),
                status
            );
        }
        let upload_capacity = server.uploads.available_permits();
        let permit = server
            .uploads
            .clone()
            .acquire_many_owned(upload_capacity as u32)
            .await
            .unwrap();
        assert_eq!(
            app.clone()
                .oneshot(request("test", "json"))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(permit);
        assert_eq!(
            app.oneshot(
                Request::get("/v1/audio/transcriptions/stream")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn websocket_finishes_and_rejects_bad_sequence() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new("test".into(), Fake).router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let url = format!("ws://{address}/v1/audio/transcriptions/stream");
        for invalid in [false, true] {
            let (mut socket, _) = connect_async(&url).await.unwrap();
            socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
            let ready = socket.next().await.unwrap().unwrap().into_text().unwrap();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&ready).unwrap()["type"],
                "ready"
            );
            let mut audio = vec![0u8; 32008];
            if invalid {
                audio[0] = 1;
            }
            socket.send(ClientMessage::Binary(audio)).await.unwrap();
            if !invalid {
                socket
                    .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
                    .await
                    .unwrap();
            }
            loop {
                let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                let event: serde_json::Value =
                    serde_json::from_str(&message.into_text().unwrap()).unwrap();
                if invalid {
                    assert_eq!(event["type"], "error");
                    break;
                }
                if event["type"] == "final" {
                    assert_eq!(event["text"], "hello");
                    break;
                }
            }
            let _ = socket.close(None).await;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        task.abort();
    }

    #[tokio::test]
    async fn websocket_streams_audio_into_one_padded_final() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        struct Counting {
            calls: Arc<std::sync::atomic::AtomicUsize>,
            sample_lengths: Arc<std::sync::Mutex<Vec<usize>>>,
        }
        impl Transcriber for Counting {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(
                &mut self,
                samples: &[f32],
                _: Option<&str>,
                _: &str,
                _: &AtomicBool,
            ) -> crate::Result<Transcript> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.sample_lengths.lock().unwrap().push(samples.len());
                Ok(Transcript {
                    text: "hello".into(),
                    language: None,
                })
            }
            fn finalization_policy(&self) -> FinalizationPolicy {
                FinalizationPolicy {
                    final_padding_samples: 8000,
                    ..FinalizationPolicy::default()
                }
            }
        }
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sample_lengths = Arc::new(std::sync::Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new(
            "test".into(),
            Counting {
                calls: calls.clone(),
                sample_lengths: sample_lengths.clone(),
            },
        )
        .router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
        socket.next().await.unwrap().unwrap();
        for sequence in 0u64..2 {
            let mut audio = Vec::with_capacity(32008);
            audio.extend(sequence.to_le_bytes());
            audio.resize(32008, 0);
            socket.send(ClientMessage::Binary(audio)).await.unwrap();
            if sequence == 0 {
                loop {
                    let event: serde_json::Value = serde_json::from_str(
                        &socket.next().await.unwrap().unwrap().into_text().unwrap(),
                    )
                    .unwrap();
                    if event["type"] == "credit" {
                        break;
                    }
                }
            }
        }
        socket
            .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        loop {
            let event: serde_json::Value =
                serde_json::from_str(&socket.next().await.unwrap().unwrap().into_text().unwrap())
                    .unwrap();
            if event["type"] == "final" {
                break;
            }
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(*sample_lengths.lock().unwrap(), [40000]);
        task.abort();
    }

    #[tokio::test]
    async fn websocket_cancel_retains_session_until_worker_returns() {
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        struct Held {
            started: Arc<tokio::sync::Notify>,
            cancelled: Arc<tokio::sync::Notify>,
            release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        }
        impl Transcriber for Held {
            fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
                Ok(None)
            }
            fn transcribe(
                &mut self,
                _: &[f32],
                _: Option<&str>,
                _: &str,
                cancel: &AtomicBool,
            ) -> crate::Result<Transcript> {
                self.started.notify_one();
                while !cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.cancelled.notify_one();
                let (lock, condition) = &*self.release;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = condition.wait(released).unwrap();
                }
                Err(crate::RuntimeError::Rejected("cancelled".into()))
            }
        }
        let started = Arc::new(tokio::sync::Notify::new());
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let server = AsrServer::new(
            "test".into(),
            Held {
                started: started.clone(),
                cancelled: cancelled.clone(),
                release: release.clone(),
            },
        );
        let session_capacity = server.sessions.available_permits();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = server.clone().router(true);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket.send(ClientMessage::Text(json!({"type":"start","version":1,"model":"test","sample_rate":16000,"format":"pcm_s16le"}).to_string())).await.unwrap();
        socket.next().await.unwrap().unwrap();
        socket
            .send(ClientMessage::Binary(vec![0; 32008]))
            .await
            .unwrap();
        socket
            .send(ClientMessage::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        socket
            .send(ClientMessage::Text(r#"{"type":"cancel"}"#.into()))
            .await
            .unwrap();
        let observed = tokio::time::timeout(Duration::from_secs(5), cancelled.notified()).await;
        let held = server.sessions.available_permits() + 1 == session_capacity;
        *release.0.lock().unwrap() = true;
        release.1.notify_one();
        observed.unwrap();
        assert!(held, "cancellation cannot retire an active engine job");
        let permits = tokio::time::timeout(
            Duration::from_secs(5),
            server
                .sessions
                .clone()
                .acquire_many_owned(session_capacity as u32),
        )
        .await
        .unwrap()
        .unwrap();
        drop(permits);
        task.abort();
    }
}
