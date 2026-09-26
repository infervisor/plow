W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


patch('tts/mod.rs', [
    ("#[cfg(feature = \"cuda\")]\npub mod t3;", "#[cfg(feature = \"cuda\")]\npub mod t3;\n#[cfg(feature = \"cuda\")]\npub mod s3gen;\n#[cfg(feature = \"cuda\")]\npub mod chatterbox;"),
])
patch('tts/serving.rs', [
    ("""pub struct SpeechModel {""", """/// Speech pipelines that own their engine (`tts.t3_cfg.v1`), by served model name.
fn workers() -> &'static Mutex<HashMap<String, Arc<super::chatterbox::ChatterboxWorker>>> {
    static W: OnceLock<Mutex<HashMap<String, Arc<super::chatterbox::ChatterboxWorker>>>> = OnceLock::new();
    W.get_or_init(Default::default)
}

/// Split `plowrt serve --assets` into text-engine assets and self-hosted speech pipelines. Each
/// `tts.t3_cfg.v1` asset starts a Chatterbox worker (its own engine on `device`) served under
/// the directory name; the rest go to the text registry unchanged.
pub fn start_speech_workers(assets: Vec<PathBuf>, device: u8) -> crate::Result<Vec<PathBuf>> {
    let mut text = Vec::new();
    for dir in assets {
        if super::t3::T3Contract::load(&dir)?.is_none() {
            text.push(dir);
            continue;
        }
        let name = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let w = super::chatterbox::ChatterboxWorker::start(&dir, device)?;
        tracing::info!(model = %name, dir = %dir.display(), "tts: chatterbox speech pipeline ready");
        workers().lock().insert(name, Arc::new(w));
    }
    Ok(text)
}

async fn speech_on_worker(w: Arc<super::chatterbox::ChatterboxWorker>, req: SpeechRequest, t_arrive: Instant) -> Response {
    let wav = match req.response_format.as_deref().unwrap_or("wav") {
        "wav" => true,
        "pcm" => false,
        f => return bad(format!("response_format {f:?} unsupported; use wav or pcm"), "response_format"),
    };
    if req.input.trim().is_empty() {
        return bad("`input` is empty", "input");
    }
    let seed = req.seed.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });
    match w.synthesize(req.voice.clone(), req.input.clone(), seed).await {
        Err(e) => server_error(e),
        Ok(a) => {
            let audio_s = a.pcm.len() as f64 / f64::from(w.sample_rate);
            let mut out = if wav { wav_header(w.sample_rate, (a.pcm.len() * 2) as u32) } else { Vec::new() };
            pcm16(&a.pcm, &mut out);
            let total = t_arrive.elapsed().as_secs_f64();
            tracing::info!(tokens = a.tokens, audio_s, t3_ms = a.t3_ms, s3gen_ms = a.s3gen_ms, total_ms = total * 1e3, rtf = total / audio_s, "tts: chatterbox speech");
            let ct = if wav { "audio/wav" } else { "audio/pcm" };
            ([(header::CONTENT_TYPE, ct.to_string()), (HeaderName::from_static("x-plow-audio-seconds"), format!("{audio_s:.3}"))], out)
                .into_response()
        }
    }
}

pub struct SpeechModel {"""),
    ("""    let t_arrive = Instant::now();
    if let Some(canonical) = state.registry.resolve(&req.model) {""", """    let t_arrive = Instant::now();
    if let Some(w) = workers().lock().get(&req.model).cloned() {
        return speech_on_worker(w, req, t_arrive).await;
    }
    if let Some(canonical) = state.registry.resolve(&req.model) {"""),
])
patch('main.rs', [
    ("""    mux_cfg: MuxConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = bringup_runtime(assets, executors, trace, mux_cfg, served_model_name).await?;""",
     """    mux_cfg: MuxConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Speech pipelines that own their engine start first; the rest are text-engine models.
    #[cfg(feature = "cuda")]
    let assets = {
        let device = RuntimeConfig::get().devices.first().copied().unwrap_or(0).min(u32::from(u8::MAX)) as u8;
        plowrt::tts::serving::start_speech_workers(assets, device)?
    };
    let state = bringup_runtime(assets, executors, trace, mux_cfg, served_model_name).await?;"""),
])
print("ok")
