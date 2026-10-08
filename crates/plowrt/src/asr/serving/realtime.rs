//! OpenAI Realtime-compatible transcription sessions: `GET /v1/realtime?intent=transcription`.
//! Base64 audio arrives in `input_audio_buffer.append` (pcm16 at 24 kHz, G.711 μ-law / A-law at
//! 8 kHz) and is resampled to 16 kHz. With `server_vad` the continuous-mode endpointer cuts turns;
//! with `turn_detection: null` audio accumulates until `input_audio_buffer.commit`. Each turn is
//! an item transcribed on the model's route, answered as
//! `conversation.item.input_audio_transcription.delta` events and one `.completed`, in turn order.

use base64::Engine as _;

use super::*;

/// One client event (base64 inflates audio by 4/3: about 10 s of pcm16).
const MAX_MESSAGE_BYTES: usize = 1 << 20;
/// Turns waiting behind the in-flight ones; past it the client is sending faster than real time.
const MAX_WAITING_TURNS: usize = 16;
/// OpenAI clients may pause between turns without streaming silence.
const REALTIME_IDLE: Duration = Duration::from_secs(120);
/// `input_audio_buffer.commit` needs this much audio (100 ms), as OpenAI's does.
const MIN_COMMIT: usize = SAMPLE_RATE as usize / 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Pcm16,
    Ulaw,
    Alaw,
}

impl Format {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "pcm16" | "audio/pcm" => Some(Self::Pcm16),
            "g711_ulaw" | "audio/pcmu" => Some(Self::Ulaw),
            "g711_alaw" | "audio/pcma" => Some(Self::Alaw),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Pcm16 => "pcm16",
            Self::Ulaw => "g711_ulaw",
            Self::Alaw => "g711_alaw",
        }
    }

    fn rate(self) -> u32 {
        match self {
            Self::Pcm16 => 24_000,
            Self::Ulaw | Self::Alaw => 8_000,
        }
    }

    fn decode(self, bytes: &[u8]) -> Option<Vec<f32>> {
        match self {
            Self::Pcm16 => (bytes.len() % 2 == 0)
                .then(|| bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect()),
            Self::Ulaw => Some(bytes.iter().map(|&b| ulaw(b) as f32 / 32768.0).collect()),
            Self::Alaw => Some(bytes.iter().map(|&b| alaw(b) as f32 / 32768.0).collect()),
        }
    }
}

/// ITU-T G.711 μ-law expansion.
fn ulaw(byte: u8) -> i16 {
    let u = !byte;
    let magnitude = ((((u & 0x0f) as i32) << 3) + 0x84) << ((u >> 4) & 7);
    (if u & 0x80 != 0 { 0x84 - magnitude } else { magnitude - 0x84 }) as i16
}

/// ITU-T G.711 A-law expansion.
fn alaw(byte: u8) -> i16 {
    let a = byte ^ 0x55;
    let (exponent, mantissa) = ((a >> 4) & 7, (a & 0x0f) as i32);
    let magnitude = if exponent == 0 { (mantissa << 4) + 8 } else { ((mantissa << 4) + 0x108) << (exponent - 1) };
    (if a & 0x80 != 0 { magnitude } else { -magnitude }) as i16
}

/// `server_vad` settings. Only `silence_duration_ms` drives the endpointer; `threshold` and
/// `prefix_padding_ms` are accepted and echoed (its margin and 200 ms context are fixed).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Vad {
    threshold: f64,
    prefix_padding_ms: u32,
    silence_duration_ms: u32,
}

#[derive(Clone, Debug)]
struct Config {
    model: Option<String>,
    language: Option<String>,
    prompt: String,
    format: Format,
    vad: Option<Vad>,
}

impl Config {
    fn session(&self, id: &str) -> serde_json::Value {
        json!({
            "id": id,
            "object": "realtime.transcription_session",
            "input_audio_format": self.format.name(),
            "input_audio_transcription": {"model": self.model, "language": self.language, "prompt": self.prompt},
            "turn_detection": self.vad.map(|v| json!({"type": "server_vad", "threshold": v.threshold,
                "prefix_padding_ms": v.prefix_padding_ms, "silence_duration_ms": v.silence_duration_ms})),
            "input_audio_noise_reduction": null,
            "include": null,
        })
    }

    /// Merge a `transcription_session.update` session (or the GA `session.update` transcription
    /// session); `Err((param, message))` leaves the config unchanged.
    fn update(&self, session: &serde_json::Value) -> Result<Self, (&'static str, String)> {
        let mut next = self.clone();
        let ga = session.get("audio").is_some() || session.get("type").is_some();
        if ga && session["type"].as_str().is_some_and(|t| t != "transcription") {
            return Err(("session.type", "only transcription sessions are served".into()));
        }
        let input = if ga { &session["audio"]["input"] } else { session };
        let format = if ga { &input["format"]["type"] } else { &input["input_audio_format"] };
        if let Some(name) = format.as_str() {
            next.format = Format::parse(name).ok_or(("session.input_audio_format", format!("unsupported format {name:?}")))?;
            if ga && next.format == Format::Pcm16 && input["format"]["rate"].as_u64().is_some_and(|r| r != 24_000) {
                return Err(("session.audio.input.format.rate", "audio/pcm is 24000 Hz".into()));
            }
        }
        let transcription = if ga { &input["transcription"] } else { &input["input_audio_transcription"] };
        if let Some(t) = transcription.as_object() {
            if let Some(model) = t.get("model").and_then(|v| v.as_str()) {
                next.model = Some(model.to_owned());
            }
            if let Some(language) = t.get("language") {
                next.language = language.as_str().filter(|l| !l.is_empty()).map(str::to_owned);
            }
            if let Some(prompt) = t.get("prompt") {
                next.prompt = prompt.as_str().unwrap_or_default().to_owned();
            }
        }
        if let Some(turn) = input.get("turn_detection") {
            next.vad = match turn {
                serde_json::Value::Null => None,
                turn if turn["type"].as_str().unwrap_or("server_vad") == "server_vad" => {
                    let current = self.vad.unwrap_or(DEFAULT_VAD);
                    let vad = Vad {
                        threshold: turn["threshold"].as_f64().unwrap_or(current.threshold),
                        prefix_padding_ms: turn["prefix_padding_ms"].as_u64().map_or(current.prefix_padding_ms, |v| v as u32),
                        silence_duration_ms: turn["silence_duration_ms"].as_u64().map_or(current.silence_duration_ms, |v| v as u32),
                    };
                    if !(200..=2000).contains(&vad.silence_duration_ms) {
                        return Err(("session.turn_detection.silence_duration_ms", "must be 200..=2000".into()));
                    }
                    Some(vad)
                }
                _ => return Err(("session.turn_detection.type", "only server_vad (or null) is supported".into())),
            };
        }
        Ok(next)
    }
}

const DEFAULT_VAD: Vad = Vad { threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500 };

/// Server events: every one carries `type` and a unique `event_id`.
struct Events(u64);

impl Events {
    fn event(&mut self, kind: &str, mut body: serde_json::Value) -> serde_json::Value {
        self.0 += 1;
        body["type"] = kind.into();
        body["event_id"] = format!("event_{:x}", self.0).into();
        body
    }

    fn error(&mut self, code: &str, message: impl ToString, param: Option<&str>, event_id: Option<&str>) -> serde_json::Value {
        let kind = if matches!(code, "rate_limit_exceeded" | "server_error" | "unavailable") { "server_error" } else { "invalid_request_error" };
        self.event("error", json!({"error": {"type": kind, "code": code, "message": message.to_string(),
            "param": param, "event_id": event_id}}))
    }
}

pub(super) async fn upgrade(
    State(state): State<Arc<AsrServer>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(query): axum::extract::Query<HashMap<String, String>>,
    ws: WebSocketUpgrade,
) -> Response {
    let mut ids = match RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => return failure(StatusCode::BAD_REQUEST, e),
    };
    if query.get("intent").is_some_and(|i| i != "transcription") {
        return failure(StatusCode::BAD_REQUEST, "only intent=transcription is served");
    }
    if state.draining() {
        return failure(StatusCode::SERVICE_UNAVAILABLE, SHUTTING_DOWN);
    }
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    let permit = match state.sessions.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return busy("too many ASR sessions"),
    };
    ids.session.get_or_insert_with(|| crate::serve::session::minted::session().into());
    let model = query.get("model").cloned();
    let echo = ids.clone();
    let mut response = ws
        .protocols(["realtime"])
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            let _permit = permit;
            session(state, socket, ids, model).await;
        })
        .into_response();
    echo.stamp(&mut response);
    response
}

/// An `error` event, then close 1001 (going away).
async fn going_away(socket: &mut WebSocket, out: &mut Events, code: &str, message: &str) {
    send(socket, out.error(code, message, None, None)).await;
    super::close(socket, 1001, message).await;
}

async fn session(state: Arc<AsrServer>, mut socket: WebSocket, ids: RequestIds, model: Option<String>) {
    let session_id = format!("sess_{}", ids.session.as_deref().unwrap_or_default());
    let mut out = Events(0);
    let mut route: Option<(String, Route, FinalizationPolicy)> = None;
    let mut _metrics: Option<AsrSessionMetrics> = None;
    // `?model=` routes now, as a session update's model does: an unknown one fails the connect.
    if let Some(model) = &model {
        match state.route(model, ids.session.as_deref()).await {
            Ok((r, finalization)) => {
                _metrics = Some(AsrSessionMetrics::new(state.metrics(model)));
                route = Some((model.clone(), r, finalization));
            }
            Err(_) => {
                let message = format!("unknown ASR model {model:?}");
                send(&mut socket, out.error("model_not_found", &message, Some("model"), None)).await;
                super::close(&mut socket, 1008, "model_not_found").await;
                return;
            }
        }
    }
    let mut config = Config { model, language: None, prompt: String::new(), format: Format::Pcm16, vad: Some(DEFAULT_VAD) };
    if !send(&mut socket, out.event("transcription_session.created", json!({"session": config.session(&session_id)}))).await {
        return;
    }
    let mut shutdown = state.shutdown.subscribe();
    let mut resampler = Resampler::new(config.format.rate()).expect("a listed format rate");
    let new_endpointer = |vad: Option<Vad>, finalization: Option<&FinalizationPolicy>| {
        let cap_ms = (MAX_SAMPLES.saturating_sub(finalization.map_or(0, |f| f.final_padding_samples)) / 16) as u32;
        vad.map(|v| Endpointer::new(EndpointConfig { min_silence_ms: v.silence_duration_ms, max_segment_ms: cap_ms.min(25_000) }))
    };
    let mut endpointer = new_endpointer(config.vad, route.as_ref().map(|r| &r.2));
    // 16 kHz samples fed since the session (or the last clear) and before the current endpointer.
    let (mut epoch, mut fed) = (0u64, 0u64);
    let mut manual: Vec<f32> = Vec::new();
    let mut item_seq = 0u64;
    let mut next_item = || {
        item_seq += 1;
        format!("item_{}_{item_seq:04}", &session_id[5..session_id.len().min(17)])
    };
    let mut speaking: Option<(u64, String)> = None;
    let mut previous: Option<String> = None;
    let mut waiting: VecDeque<(String, Segment, Turn)> = VecDeque::new();
    let mut flights: VecDeque<Flight> = VecDeque::new();
    let mut flight_items: VecDeque<String> = VecDeque::new();
    let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + PING_INTERVAL, PING_INTERVAL);
    let mut last_pong = tokio::time::Instant::now();
    let mut idle_at = tokio::time::Instant::now() + REALTIME_IDLE;
    loop {
        // Launch waiting turns with the settings current at their commit.
        while flights.len() < MAX_SEGMENTS_IN_FLIGHT {
            // A rank unloaded under the session: route its turns (and the session) again.
            if let Some((_, _, turn)) = waiting.front_mut().filter(|(_, _, t)| t.route.stale(&state)) {
                if let Ok((r, finalization)) = state.route(&turn.model, ids.session.as_deref()).await {
                    if let Some(current) = route.as_mut().filter(|c| c.0 == turn.model) {
                        current.1 = r.clone();
                        current.2 = finalization;
                    }
                    turn.route = r;
                    turn.finalization = finalization;
                }
            }
            let Some((_, segment, turn)) = waiting.front() else { break };
            match launch(&state, &ids, turn, segment) {
                Ok(flight) => {
                    let (item, _, _) = waiting.pop_front().expect("guarded");
                    flights.push_back(flight);
                    flight_items.push_back(item);
                }
                Err(SubmitError::Full) if !flights.is_empty() => break,
                Err(SubmitError::Full) => {
                    let (item, _, _) = waiting.pop_front().expect("guarded");
                    let failed = out.event("conversation.item.input_audio_transcription.failed", json!({"item_id": item,
                        "content_index": 0, "error": {"type": "server_error", "code": "rate_limit_exceeded",
                        "message": "ASR queue full", "param": null}}));
                    if !send(&mut socket, failed).await {
                        return;
                    }
                }
                Err(SubmitError::Closed) => {
                    going_away(&mut socket, &mut out, "unavailable", "ASR engine unavailable").await;
                    return;
                }
            }
        }
        if waiting.len() > MAX_WAITING_TURNS {
            going_away(&mut socket, &mut out, "rate_limit_exceeded", "audio arrives faster than it is transcribed").await;
            return;
        }
        if !flights.is_empty() {
            idle_at = tokio::time::Instant::now() + REALTIME_IDLE;
        }
        let message = tokio::select! {
            m = tokio::time::timeout_at(idle_at, socket.recv()) => match m {
                Ok(Some(Ok(m))) => m,
                Err(_) => {
                    going_away(&mut socket, &mut out, "timeout", "no client event for 120 s").await;
                    return;
                }
                _ => return,
            },
            event = front_event(&mut flights), if !flights.is_empty() => {
                let item = flight_items.front().expect("guarded").clone();
                let sent = match event {
                    FrontEvent::Delta(delta) => {
                        flights.front_mut().expect("guarded").shown.push_str(&delta);
                        send(&mut socket, out.event("conversation.item.input_audio_transcription.delta",
                            json!({"item_id": item, "content_index": 0, "delta": delta}))).await
                    }
                    FrontEvent::Timeout => {
                        // Dropping the flight cancels its work without waiting for a stalled engine.
                        drop(flights.pop_front());
                        flight_items.pop_front();
                        send(&mut socket, out.event("conversation.item.input_audio_transcription.failed", json!({
                            "item_id": item, "content_index": 0, "error": {"type": "server_error", "code": "timeout",
                            "message": DEADLINE, "param": null}}))).await
                    }
                    FrontEvent::Done(result) => {
                        let mut f = flights.pop_front().expect("guarded");
                        flight_items.pop_front();
                        match result {
                            Ok(Ok(result)) => {
                                while let Ok(delta) = f.deltas.try_recv() {
                                    f.shown.push_str(&delta);
                                    send(&mut socket, out.event("conversation.item.input_audio_transcription.delta",
                                        json!({"item_id": item, "content_index": 0, "delta": delta}))).await;
                                }
                                if let Some(rest) = result.text.strip_prefix(f.shown.as_str()).filter(|r| !r.is_empty()) {
                                    send(&mut socket, out.event("conversation.item.input_audio_transcription.delta",
                                        json!({"item_id": item, "content_index": 0, "delta": rest}))).await;
                                }
                                f.run.first();
                                f.run.done();
                                send(&mut socket, out.event("conversation.item.input_audio_transcription.completed",
                                    json!({"item_id": item, "content_index": 0, "transcript": result.text}))).await
                            }
                            other => {
                                let (code, message) = match other {
                                    Ok(Err(crate::RuntimeError::Overloaded(m))) => ("rate_limit_exceeded", m),
                                    Ok(Err(crate::RuntimeError::Unavailable(m))) => ("unavailable", m),
                                    Ok(Err(e)) => ("server_error", e.to_string()),
                                    _ => ("server_error", "ASR engine response channel closed".into()),
                                };
                                send(&mut socket, out.event("conversation.item.input_audio_transcription.failed", json!({
                                    "item_id": item, "content_index": 0, "error": {"type": "server_error", "code": code,
                                    "message": message, "param": null}}))).await
                            }
                        }
                    }
                };
                if !sent {
                    return;
                }
                continue;
            }
            _ = ping.tick() => {
                if last_pong.elapsed() >= PONG_TIMEOUT {
                    going_away(&mut socket, &mut out, "ping_timeout", "ping timeout").await;
                    return;
                }
                if socket.send(Message::Ping(Vec::new())).await.is_err() {
                    return;
                }
                continue;
            }
            Ok(()) = shutdown.changed() => {
                if *shutdown.borrow() {
                    going_away(&mut socket, &mut out, "unavailable", SHUTTING_DOWN).await;
                    return;
                }
                continue;
            }
        };
        let text = match message {
            Message::Text(text) => text,
            Message::Pong(_) => {
                last_pong = tokio::time::Instant::now();
                continue;
            }
            Message::Ping(_) => continue,
            Message::Close(_) => return,
            Message::Binary(_) => {
                if !send(&mut socket, out.error("invalid_request_error", "binary frames are not part of the protocol; send input_audio_buffer.append", None, None)).await {
                    return;
                }
                continue;
            }
        };
        idle_at = tokio::time::Instant::now() + REALTIME_IDLE;
        let Ok(event) = serde_json::from_str::<serde_json::Value>(&text) else {
            if !send(&mut socket, out.error("invalid_json", "event is not JSON", None, None)).await {
                return;
            }
            continue;
        };
        let client_id = event["event_id"].as_str().map(str::to_owned);
        let client_id = client_id.as_deref();
        let reply = match event["type"].as_str().unwrap_or_default() {
            kind @ ("transcription_session.update" | "session.update") => match config.update(&event["session"]) {
                Ok(next) => {
                    // Route the (new) model now, so a bad name fails the update, not a later turn.
                    let mut rerouted = None;
                    if let Some(model) = next.model.clone().filter(|m| route.as_ref().is_none_or(|r| &r.0 != m)) {
                        match state.route(&model, ids.session.as_deref()).await {
                            Ok((r, finalization)) => rerouted = Some((model, r, finalization)),
                            Err(_) => {
                                if !send(&mut socket, out.error("model_not_found", format!("unknown ASR model {model:?}"),
                                    Some("session.input_audio_transcription.model"), client_id)).await {
                                    return;
                                }
                                continue;
                            }
                        }
                    }
                    if next.format != config.format {
                        resampler = Resampler::new(next.format.rate()).expect("a listed format rate");
                    }
                    if next.vad != config.vad {
                        // Settings change between turns: the open turn (if any) is committed first,
                        // with the settings it was spoken under.
                        if let Some(segment) = endpointer.as_mut().and_then(Endpointer::finish) {
                            let turn = Turn::of(&config, &route);
                            commit_turn(&mut out, &mut socket, &mut speaking, &mut previous, &mut waiting, &mut next_item, epoch, segment, turn).await;
                        }
                        epoch += fed;
                        fed = 0;
                        endpointer = new_endpointer(next.vad, rerouted.as_ref().or(route.as_ref()).map(|r| &r.2));
                    }
                    if let Some((model, r, finalization)) = rerouted {
                        _metrics = Some(AsrSessionMetrics::new(state.metrics(&model)));
                        route = Some((model, r, finalization));
                    }
                    config = next;
                    let updated = if kind == "session.update" { "session.updated" } else { "transcription_session.updated" };
                    let mut session = config.session(&session_id);
                    if kind == "session.update" {
                        session["type"] = "transcription".into();
                    }
                    out.event(updated, json!({"session": session}))
                }
                Err((param, message)) => out.error("invalid_value", message, Some(param), client_id),
            },
            "input_audio_buffer.append" => {
                let Some(audio) = event["audio"].as_str() else {
                    if !send(&mut socket, out.error("missing_required_parameter", "audio is required", Some("audio"), client_id)).await {
                        return;
                    }
                    continue;
                };
                let Some(samples) = base64::engine::general_purpose::STANDARD
                    .decode(audio)
                    .ok()
                    .and_then(|bytes| config.format.decode(&bytes))
                else {
                    if !send(&mut socket, out.error("invalid_value", format!("audio is not base64 {}", config.format.name()),
                        Some("audio"), client_id)).await {
                        return;
                    }
                    continue;
                };
                if route.is_none() {
                    if !send(&mut socket, out.error("missing_required_parameter",
                        "no transcription model: pass ?model= or session.input_audio_transcription.model",
                        Some("session.input_audio_transcription.model"), client_id)).await {
                        return;
                    }
                    continue;
                }
                let pcm = resampler.push(&samples);
                match endpointer.as_mut() {
                    None => {
                        if manual.len() + pcm.len() > MAX_SAMPLES {
                            if !send(&mut socket, out.error("input_audio_buffer_overflow", "the buffer holds at most 30 s; commit first",
                                None, client_id)).await {
                                return;
                            }
                            continue;
                        }
                        manual.extend_from_slice(&pcm);
                        continue;
                    }
                    Some(e) => {
                        fed += pcm.len() as u64;
                        let closed = e.push(&pcm);
                        let open = e.open_audio().map(|(index, audio)| (index, fed - audio.len() as u64));
                        for segment in closed {
                            if !commit_turn(&mut out, &mut socket, &mut speaking, &mut previous, &mut waiting, &mut next_item, epoch, segment, Turn::of(&config, &route)).await {
                                return;
                            }
                        }
                        if let Some((index, start)) = open.filter(|o| speaking.as_ref().is_none_or(|s| s.0 != o.0)) {
                            let item = next_item();
                            let started = out.event("input_audio_buffer.speech_started",
                                json!({"audio_start_ms": (epoch + start) / 16, "item_id": item}));
                            speaking = Some((index, item));
                            if !send(&mut socket, started).await {
                                return;
                            }
                        }
                        continue;
                    }
                }
            }
            "input_audio_buffer.commit" => {
                let tail = resampler.finish();
                resampler = Resampler::new(config.format.rate()).expect("a listed format rate");
                let segment = match endpointer.as_mut() {
                    None => {
                        manual.extend_from_slice(&tail);
                        (manual.len() >= MIN_COMMIT).then(|| {
                            let samples = std::mem::take(&mut manual);
                            let end = fed + samples.len() as u64;
                            let segment = Segment { index: u64::MAX, start: fed, end, samples };
                            fed = end;
                            segment
                        })
                    }
                    Some(e) => {
                        fed += tail.len() as u64;
                        let mut closed = e.push(&tail);
                        closed.extend(e.finish());
                        let last = closed.pop();
                        for segment in closed {
                            if !commit_turn(&mut out, &mut socket, &mut speaking, &mut previous, &mut waiting, &mut next_item, epoch, segment, Turn::of(&config, &route)).await {
                                return;
                            }
                        }
                        last
                    }
                };
                match segment {
                    Some(segment) => {
                        if !commit_turn(&mut out, &mut socket, &mut speaking, &mut previous, &mut waiting, &mut next_item, epoch, segment, Turn::of(&config, &route)).await {
                            return;
                        }
                        continue;
                    }
                    None => out.error("input_audio_buffer_commit_empty", "buffer has less than 100ms of audio", None, client_id),
                }
            }
            "input_audio_buffer.clear" => {
                manual.clear();
                resampler = Resampler::new(config.format.rate()).expect("a listed format rate");
                epoch += fed;
                fed = 0;
                speaking = None;
                endpointer = new_endpointer(config.vad, route.as_ref().map(|r| &r.2));
                out.event("input_audio_buffer.cleared", json!({}))
            }
            other => out.error("invalid_event", format!("unsupported event type {other:?}"), Some("type"), client_id),
        };
        if !send(&mut socket, reply).await {
            return;
        }
    }
}

/// A closed turn: `speech_stopped` (VAD; `speech_started` first if it opened and closed in one
/// append), then `committed`, and the turn joins the transcription queue.
#[allow(clippy::too_many_arguments)]
async fn commit_turn(
    out: &mut Events,
    socket: &mut WebSocket,
    speaking: &mut Option<(u64, String)>,
    previous: &mut Option<String>,
    waiting: &mut VecDeque<(String, Segment, Turn)>,
    next_item: &mut impl FnMut() -> String,
    epoch: u64,
    segment: Segment,
    turn: Option<Turn>,
) -> bool {
    // Audio is refused until the session has a model, so a turn always has one.
    let Some(turn) = turn else { return true };
    let vad = segment.index != u64::MAX;
    let item = match speaking.take() {
        Some((index, item)) if index == segment.index => item,
        other => {
            *speaking = other;
            let item = next_item();
            if vad && !send(socket, out.event("input_audio_buffer.speech_started",
                json!({"audio_start_ms": (epoch + segment.start) / 16, "item_id": item}))).await {
                return false;
            }
            item
        }
    };
    if vad && !send(socket, out.event("input_audio_buffer.speech_stopped",
        json!({"audio_end_ms": (epoch + segment.end) / 16, "item_id": item}))).await {
        return false;
    }
    if !send(socket, out.event("input_audio_buffer.committed", json!({"item_id": item, "previous_item_id": previous}))).await {
        return false;
    }
    *previous = Some(item.clone());
    waiting.push_back((item, segment, turn));
    true
}

/// What a committed turn is transcribed with: the session's settings at its commit, so a later
/// session update changes only later turns.
struct Turn {
    model: String,
    route: Route,
    finalization: FinalizationPolicy,
    language: Option<String>,
    prompt: String,
}

impl Turn {
    fn of(config: &Config, route: &Option<(String, Route, FinalizationPolicy)>) -> Option<Self> {
        let (model, route, finalization) = route.as_ref()?;
        Some(Self {
            model: model.clone(),
            route: route.clone(),
            finalization: *finalization,
            language: config.language.clone(),
            prompt: config.prompt.clone(),
        })
    }
}

fn launch(state: &AsrServer, ids: &RequestIds, turn: &Turn, segment: &Segment) -> Result<Flight, SubmitError> {
    let Turn { model, route, finalization, .. } = turn;
    let mut samples = segment.samples.clone();
    if samples.len() < SAMPLE_RATE as usize / 2 {
        samples.resize(SAMPLE_RATE as usize / 2, 0.0);
    }
    append_final_padding(&mut samples, finalization.final_padding_samples, finalization.final_padding_amplitude);
    let run = crate::serve::turns::StageRun::start(ids, crate::serve::turns::Kind::Asr, model, state.metrics(model), Instant::now(), true);
    let (tx, deltas) = mpsc::unbounded_channel();
    let cancel = Cancellation(Arc::new(AtomicBool::new(false)));
    let opts = AsrOpts {
        final_pass: true,
        ids: Some(RequestIds { turn_key: run.key(), ..ids.with_new_request() }),
        windows: None,
        deltas: Some(tx),
        report: None,
        ..Default::default()
    };
    let work = route.submit(samples, turn.language.clone(), turn.prompt.clone(), cancel.0.clone(), opts)?;
    Ok(Flight {
        segment: segment.index,
        start_ms: segment.start / 16,
        end_ms: segment.end / 16,
        work,
        deltas,
        shown: String::new(),
        deadline: state.request_timeout.map(|t| tokio::time::Instant::now() + t),
        run,
        _cancel: cancel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};

    type ClientSocket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

    /// Grows "hello" -> "hello world" per call, numbering calls: `hello world 0`, `hello world 1`.
    struct Growing(usize);
    impl Transcriber for Growing {
        fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
            Ok(None)
        }
        fn transcribe(&mut self, s: &[f32], l: Option<&str>, c: &str, x: &AtomicBool) -> crate::Result<Transcript> {
            self.transcribe_streaming(s, l, c, x, &mut |_| {})
        }
        fn transcribe_streaming(
            &mut self,
            samples: &[f32],
            _: Option<&str>,
            _: &str,
            _: &AtomicBool,
            on_text: &mut dyn FnMut(&str),
        ) -> crate::Result<Transcript> {
            assert!(samples.len() >= 8_000);
            let text = format!("hello world {}", self.0);
            self.0 += 1;
            on_text("hello");
            on_text(&text);
            Ok(Transcript { text, language: None })
        }
    }

    async fn server(keys: &[crate::config::ApiKey]) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = crate::serve::auth::require(AsrServer::new("test".into(), Growing(0)).router(true), keys);
        (address, tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }))
    }

    async fn connect(address: std::net::SocketAddr, query: &str) -> ClientSocket {
        connect_async(format!("ws://{address}/v1/realtime?intent=transcription{query}")).await.unwrap().0
    }

    async fn next(socket: &mut ClientSocket) -> serde_json::Value {
        loop {
            match tokio::time::timeout(Duration::from_secs(5), socket.next()).await.unwrap().unwrap().unwrap() {
                ClientMessage::Text(text) => return serde_json::from_str(&text).unwrap(),
                ClientMessage::Close(frame) => return json!({"type":"close","code":u16::from(frame.unwrap().code)}),
                _ => {}
            }
        }
    }

    async fn event(socket: &mut ClientSocket, value: serde_json::Value) {
        socket.send(ClientMessage::Text(value.to_string())).await.unwrap();
    }

    /// 24 kHz pcm16 of `spans` (seconds, speech?), base64 in 100 ms appends.
    async fn append_24k(socket: &mut ClientSocket, spans: &[(f32, bool)]) {
        let audio: Vec<i16> = spans
            .iter()
            .flat_map(|&(seconds, tone)| {
                (0..(seconds * 24_000.0) as usize)
                    .map(move |i| if tone { (crate::asr::endpoint::speechlike(i * 2 / 3) * 32767.0) as i16 } else { 0 })
            })
            .collect();
        for chunk in audio.chunks(2_400) {
            let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
            let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
            event(socket, json!({"type":"input_audio_buffer.append","audio":audio})).await;
        }
    }

    #[test]
    fn g711_expands_known_codes() {
        assert_eq!([ulaw(0xFF), ulaw(0x7F), ulaw(0x00), ulaw(0x80)], [0, 0, -32124, 32124]);
        assert_eq!([alaw(0xD5), alaw(0x55), alaw(0xAA), alaw(0x2A)], [8, -8, 32256, -32256]);
        assert!((0..=255u8).all(|b| ulaw(b) == -ulaw(b ^ 0x80) || ulaw(b) == 0));
    }

    #[tokio::test]
    async fn manual_commit_creates_one_item_with_deltas_then_completed() {
        let (address, task) = server(&[]).await;
        let mut socket = connect(address, "&model=test").await;
        let created = next(&mut socket).await;
        assert_eq!(created["type"], "transcription_session.created");
        assert!(created["event_id"].as_str().is_some());
        event(&mut socket, json!({"type":"transcription_session.update","session":{"turn_detection":null,
            "input_audio_transcription":{"model":"test","language":"en"}}})).await;
        let updated = next(&mut socket).await;
        assert_eq!(updated["type"], "transcription_session.updated");
        assert!(updated["session"]["turn_detection"].is_null());
        assert_eq!(updated["session"]["input_audio_transcription"]["language"], "en");
        event(&mut socket, json!({"type":"input_audio_buffer.commit","event_id":"c0"})).await;
        let empty = next(&mut socket).await;
        assert_eq!((empty["type"].as_str(), empty["error"]["code"].as_str(), empty["error"]["event_id"].as_str()),
            (Some("error"), Some("input_audio_buffer_commit_empty"), Some("c0")));
        append_24k(&mut socket, &[(1.0, true)]).await;
        event(&mut socket, json!({"type":"input_audio_buffer.commit"})).await;
        let committed = next(&mut socket).await;
        assert_eq!(committed["type"], "input_audio_buffer.committed");
        assert!(committed["previous_item_id"].is_null());
        let item = committed["item_id"].clone();
        let mut deltas = String::new();
        let completed = loop {
            let e = next(&mut socket).await;
            assert_eq!(e["item_id"], item);
            match e["type"].as_str().unwrap() {
                "conversation.item.input_audio_transcription.delta" => deltas.push_str(e["delta"].as_str().unwrap()),
                "conversation.item.input_audio_transcription.completed" => break e,
                other => panic!("unexpected {other}"),
            }
        };
        assert_eq!(completed["transcript"], "hello world 0");
        assert_eq!(deltas, "hello world 0");
        event(&mut socket, json!({"type":"input_audio_buffer.clear"})).await;
        assert_eq!(next(&mut socket).await["type"], "input_audio_buffer.cleared");
        event(&mut socket, json!({"type":"response.create"})).await;
        let bad = next(&mut socket).await;
        assert_eq!((bad["type"].as_str(), bad["error"]["code"].as_str()), (Some("error"), Some("invalid_event")));
        task.abort();
    }

    #[tokio::test]
    async fn server_vad_turns_become_items_in_order() {
        let (address, task) = server(&[]).await;
        let mut socket = connect(address, "").await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.created");
        event(&mut socket, json!({"type":"input_audio_buffer.append","audio":""})).await;
        assert_eq!(next(&mut socket).await["error"]["code"], "missing_required_parameter");
        event(&mut socket, json!({"type":"transcription_session.update","session":{"input_audio_transcription":{"model":"nope"}}})).await;
        assert_eq!(next(&mut socket).await["error"]["code"], "model_not_found");
        event(&mut socket, json!({"type":"transcription_session.update","session":{"input_audio_transcription":{"model":"test"},
            "turn_detection":{"type":"server_vad","silence_duration_ms":500}}})).await;
        assert_eq!(next(&mut socket).await["session"]["turn_detection"]["silence_duration_ms"], 500);
        append_24k(&mut socket, &[(0.5, false), (2.0, true), (1.5, false), (2.0, true), (1.5, false)]).await;
        let mut kinds = Vec::new();
        let mut transcripts = Vec::new();
        let mut items = Vec::new();
        while transcripts.len() < 2 {
            let e = next(&mut socket).await;
            let kind = e["type"].as_str().unwrap().to_owned();
            if kind == "conversation.item.input_audio_transcription.delta" {
                continue;
            }
            if kind == "conversation.item.input_audio_transcription.completed" {
                transcripts.push(e["transcript"].as_str().unwrap().to_owned());
            }
            if kind == "input_audio_buffer.speech_started" {
                items.push(e["item_id"].clone());
                assert!(e["audio_start_ms"].as_u64().unwrap() >= 300);
            }
            kinds.push(kind);
        }
        assert_eq!(items.len(), 2);
        assert_ne!(items[0], items[1]);
        let order: Vec<&str> = kinds.iter().map(String::as_str).collect();
        let first = &order[..order.iter().position(|k| k.ends_with("completed")).unwrap() + 1];
        assert_eq!(&first[..3], ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]);
        assert_eq!(transcripts, ["hello world 0", "hello world 1"]);
        task.abort();
    }

    type Gate = Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

    fn open(gate: &Gate) {
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
    }

    /// Answers `language|prompt`, but only once `gate` opens (cancellation is ignored: a stall).
    struct Gated(Gate);
    impl Transcriber for Gated {
        fn language(&self, _: Option<&str>) -> crate::Result<Option<String>> {
            Ok(None)
        }
        fn transcribe(&mut self, _: &[f32], language: Option<&str>, prompt: &str, _: &AtomicBool) -> crate::Result<Transcript> {
            let mut open = self.0 .0.lock().unwrap();
            while !*open {
                open = self.0 .1.wait(open).unwrap();
            }
            Ok(Transcript { text: format!("{}|{prompt}", language.unwrap_or("-")), language: None })
        }
    }

    async fn gated_server(timeout: Option<Duration>) -> (std::net::SocketAddr, Gate, tokio::task::JoinHandle<()>) {
        let gate: Gate = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = AsrServer::new("test".into(), Gated(gate.clone())).with_request_timeout(timeout).router(true);
        (address, gate, tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }))
    }

    /// `?model=` alone routes the session; an unknown one is refused at connect.
    #[tokio::test]
    async fn query_model_routes_the_session() {
        let (address, task) = server(&[]).await;
        let mut socket = connect(address, "&model=test").await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.created");
        event(&mut socket, json!({"type":"transcription_session.update","session":{"turn_detection":null}})).await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.updated");
        append_24k(&mut socket, &[(1.0, true)]).await;
        event(&mut socket, json!({"type":"input_audio_buffer.commit"})).await;
        let completed = loop {
            let e = next(&mut socket).await;
            assert_ne!(e["type"], "error", "{e}");
            if e["type"] == "conversation.item.input_audio_transcription.completed" {
                break e;
            }
        };
        assert_eq!(completed["transcript"], "hello world 0");

        let mut socket = connect(address, "&model=nope").await;
        let refused = next(&mut socket).await;
        assert_eq!((refused["type"].as_str(), refused["error"]["code"].as_str()), (Some("error"), Some("model_not_found")));
        assert_eq!(next(&mut socket).await["type"], "close");
        task.abort();
    }

    /// A session update changes only turns committed after it, not ones still waiting to launch.
    #[tokio::test]
    async fn committed_turns_keep_their_settings() {
        let (address, gate, task) = gated_server(None).await;
        let mut socket = connect(address, "&model=test").await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.created");
        event(&mut socket, json!({"type":"transcription_session.update","session":{"turn_detection":null,
            "input_audio_transcription":{"language":"en","prompt":"one"}}})).await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.updated");
        // Two turns in flight on the held engine, the third waits behind them.
        for _ in 0..3 {
            append_24k(&mut socket, &[(0.5, true)]).await;
            event(&mut socket, json!({"type":"input_audio_buffer.commit"})).await;
            assert_eq!(next(&mut socket).await["type"], "input_audio_buffer.committed");
        }
        event(&mut socket, json!({"type":"transcription_session.update","session":{
            "input_audio_transcription":{"language":"fr","prompt":"two"}}})).await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.updated");
        append_24k(&mut socket, &[(0.5, true)]).await;
        event(&mut socket, json!({"type":"input_audio_buffer.commit"})).await;
        assert_eq!(next(&mut socket).await["type"], "input_audio_buffer.committed");
        open(&gate);
        let mut transcripts = Vec::new();
        while transcripts.len() < 4 {
            let e = next(&mut socket).await;
            if e["type"] == "conversation.item.input_audio_transcription.completed" {
                transcripts.push(e["transcript"].as_str().unwrap().to_owned());
            }
        }
        assert_eq!(transcripts, ["en|one", "en|one", "en|one", "fr|two"]);
        task.abort();
    }

    /// The deadline answers while the engine is still stalled on the turn.
    #[tokio::test]
    async fn deadline_fails_the_turn_without_waiting_for_the_engine() {
        let (address, gate, task) = gated_server(Some(Duration::from_millis(100))).await;
        let mut socket = connect(address, "&model=test").await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.created");
        event(&mut socket, json!({"type":"transcription_session.update","session":{"turn_detection":null}})).await;
        assert_eq!(next(&mut socket).await["type"], "transcription_session.updated");
        append_24k(&mut socket, &[(0.5, true)]).await;
        event(&mut socket, json!({"type":"input_audio_buffer.commit"})).await;
        assert_eq!(next(&mut socket).await["type"], "input_audio_buffer.committed");
        let failed = next(&mut socket).await;
        assert_eq!((failed["type"].as_str(), failed["error"]["code"].as_str()),
            (Some("conversation.item.input_audio_transcription.failed"), Some("timeout")));
        open(&gate);
        task.abort();
    }

    #[tokio::test]
    async fn api_key_rides_the_browser_subprotocol() {
        let keys = ["k1".parse().unwrap()];
        let (address, task) = server(&keys).await;
        let url = format!("ws://{address}/v1/realtime?intent=transcription&model=test");
        assert!(connect_async(url.as_str()).await.is_err());
        let mut request = tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(url.as_str()).unwrap();
        request.headers_mut().insert("sec-websocket-protocol", "realtime, openai-insecure-api-key.k1".parse().unwrap());
        let (mut socket, response) = connect_async(request).await.unwrap();
        assert_eq!(response.headers()["sec-websocket-protocol"], "realtime");
        assert_eq!(next(&mut socket).await["type"], "transcription_session.created");
        task.abort();
    }
}
