//! Speech (TTS) packet contracts: a causal codec-token LM declared as a `tts.codec_lm.v1`
//! pipeline. The programs are the ordinary causal ones; the pipeline adds the numeric contract a
//! backend-neutral speech controller needs (prompt framing, codec frame layout, stop ids,
//! sampling defaults) so plowrt binds a family by driver, never by checkpoint knowledge.

use crate::pipeline::{causal_pipeline_section, CausalPipelineSpec};
use packet::devbuild::{Model, SectionData};
use plow_asset::packet_pipeline::{PacketPipelines, SECTION};

pub const DRIVER: &str = "tts.codec_lm.v1";
/// `prompt.format` values. 1: `<spk_{voice}> {input}` between prefix and suffix ids (Veena).
pub const PROMPT_SPEAKER_TAG: u64 = 1;
/// `codec.kind` values. 1: SNAC 24 kHz, 7 codes per frame in Orpheus order, 2048 samples/frame.
pub const CODEC_SNAC24K_FRAME7: u64 = 1;

/// One speech family's contract. Token ids are the checkpoint tokenizer's.
pub struct SpeechProfile {
    pub prompt_format: u64,
    pub prefix: &'static [u64],
    pub suffix: &'static [u64],
    pub stops: &'static [u64],
    pub codec_kind: u64,
    pub sample_rate: u64,
    pub frame_codes: u64,
    pub codebook: u64,
    pub frame_samples: u64,
    pub audio_token_base: u64,
    /// max_new_tokens = min(floor(chars * per_char_frames) * frame_codes + 21, max_new_tokens_cap)
    pub per_char_frames: f32,
    pub max_new_tokens_cap: u64,
    pub temperature: f32,
    pub top_p: f32,
}

/// maya-research/Veena (model card): [SOH] <spk_v> text [EOH] [SOA] [SOS] -> SNAC codes, stop on
/// END_OF_SPEECH / END_OF_AI.
pub const VEENA: SpeechProfile = SpeechProfile {
    prompt_format: PROMPT_SPEAKER_TAG,
    prefix: &[128259],
    suffix: &[128260, 128261, 128257],
    stops: &[128258, 128262],
    codec_kind: CODEC_SNAC24K_FRAME7,
    sample_rate: 24000,
    frame_codes: 7,
    codebook: 4096,
    frame_samples: 2048,
    audio_token_base: 128266,
    per_char_frames: 1.3,
    max_new_tokens_cap: 700,
    temperature: 0.4,
    top_p: 0.9,
};

/// Chatterbox T3 (`tts.t3_cfg.v1`): a causal LM whose prefill rows are host embeddings (the
/// `overlay` role, every row) and whose decode embedding adds a learned speech position counted
/// from the per-slot `pos_base`. Classifier-free guidance pairs two slots per request; the
/// contract scalars (text/speech control ids, sampling, guidance weight) ride as parameters.
pub const T3_DRIVER: &str = "tts.t3_cfg.v1";

pub fn t3_pipeline_section(
    model: &Model,
    spec: CausalPipelineSpec<'_>,
    pos_base: u32,
    params: &[(String, u64)],
) -> Result<SectionData, String> {
    use plow_asset::packet_pipeline::{PipelineDType, PipelineTensor};
    let causal = causal_pipeline_section(model, spec)?;
    let mut meta: PacketPipelines = serde_json::from_slice(&causal.data).map_err(|e| e.to_string())?;
    let pipe = meta.pipelines.first_mut().ok_or("causal section has no pipeline")?;
    pipe.driver = T3_DRIVER.into();
    let t = model.tensors.get(pos_base as usize).ok_or("T3 pos_base tensor is missing")?;
    pipe.tensors.insert(
        "pos_base".into(),
        PipelineTensor { name: t.name.clone(), dtype: PipelineDType::U32, shape: vec![u64::from(spec.decode_capacity)] },
    );
    for (k, v) in params {
        pipe.parameters.insert(format!("t3.{k}"), *v);
    }
    // Prefill layout: [voice rows][start_text, text..., stop_text: table + position][BOS x2];
    // the unconditional CFG member keeps positions but drops the text table.
    pipe.parameters.insert("prompt.bos_repeat".into(), 2);
    pipe.parameters.insert("cfg.uncond_drops_text".into(), 1);
    pipe.strings.insert("text.rules".into(), T3_TEXT_RULES.into());
    for t in model.tensors.iter().filter(|t| t.name.starts_with("in.prompt.")) {
        pipe.tensors.insert(
            t.name.clone(),
            PipelineTensor { name: t.name.clone(), dtype: PipelineDType::F32, shape: vec![t.bytes / 4] },
        );
    }
    meta.validate(model.progs.len(), |name| model.tensors.iter().find(|t| t.name == name).map(|t| t.bytes))?;
    Ok(SectionData { kind: causal.kind, name: SECTION.into(), data: serde_json::to_vec(&meta).map_err(|e| e.to_string())? })
}

pub fn profile(name: &str) -> Result<&'static SpeechProfile, String> {
    match name {
        "veena" => Ok(&VEENA),
        other => Err(format!("unknown tts profile {other:?} (known: veena)")),
    }
}

pub fn speech_pipeline_section(
    model: &Model,
    spec: CausalPipelineSpec<'_>,
    p: &SpeechProfile,
    vocab: u32,
) -> Result<SectionData, String> {
    if p.audio_token_base + p.frame_codes * p.codebook > u64::from(vocab) {
        return Err("tts profile audio codes exceed the vocabulary".into());
    }
    let causal = causal_pipeline_section(model, spec)?;
    let mut meta: PacketPipelines = serde_json::from_slice(&causal.data).map_err(|e| e.to_string())?;
    let pipe = meta.pipelines.first_mut().ok_or("causal section has no pipeline")?;
    pipe.name = "speech".into();
    pipe.driver = DRIVER.into();
    let params = &mut pipe.parameters;
    let mut put = |k: String, v: u64| {
        params.insert(k, v);
    };
    put("prompt.format".into(), p.prompt_format);
    for (list, key) in [(p.prefix, "prompt.prefix"), (p.suffix, "prompt.suffix"), (p.stops, "stop")] {
        put(format!("{key}.count"), list.len() as u64);
        for (i, &id) in list.iter().enumerate() {
            put(format!("{key}.{i}"), id);
        }
    }
    put("codec.kind".into(), p.codec_kind);
    put("audio.sample_rate".into(), p.sample_rate);
    put("codec.frame_codes".into(), p.frame_codes);
    put("codec.codebook".into(), p.codebook);
    put("codec.frame_samples".into(), p.frame_samples);
    put("audio.token_base".into(), p.audio_token_base);
    put("tokens.per_char_frames_f32".into(), u64::from(p.per_char_frames.to_bits()));
    put("tokens.max_new_cap".into(), p.max_new_tokens_cap);
    put("sampling.temperature_f32".into(), u64::from(p.temperature.to_bits()));
    put("sampling.top_p_f32".into(), u64::from(p.top_p.to_bits()));
    meta.validate(model.progs.len(), |name| {
        model.tensors.iter().find(|t| t.name == name).map(|t| t.bytes)
    })?;
    Ok(SectionData {
        kind: causal.kind,
        name: SECTION.into(),
        data: serde_json::to_vec(&meta).map_err(|e| e.to_string())?,
    })
}

/// Host-side tables of a guided speech LM whose prefill rows are embeddings (T3): text token
/// and text position tables, the speech BOS row, and one conditioning block per voice. Embedded
/// in the packet (`in.prompt.*` tensors) so the runtime assembles prefill rows from packet data only.
pub fn t3_host_tables(dir: &std::path::Path, hidden: u32) -> Result<Vec<packet::devbuild::TensorDecl>, String> {
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    let f32_tensor = |name: &str, out: &str| -> Result<packet::devbuild::TensorDecl, String> {
        let (dtype, bytes) = reader.read(name)?;
        if dtype != "F32" || bytes.len() % (hidden as usize * 4) != 0 {
            return Err(format!("{name}: expected f32 rows of {hidden}"));
        }
        Ok(packet::devbuild::TensorDecl { name: out.into(), bytes: bytes.len() as u64, init: Some(bytes) })
    };
    let mut tables = vec![
        f32_tensor("t3.text_emb.weight", "in.prompt.text_table")?,
        f32_tensor("t3.text_pos_emb.weight", "in.prompt.text_pos")?,
        f32_tensor("t3.speech_bos", "in.prompt.bos_row")?,
    ];
    let voices = dir.join("voices");
    let mut entries: Vec<_> = std::fs::read_dir(&voices)
        .map_err(|e| format!("{}: {e}", voices.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "f32"))
        .collect();
    entries.sort();
    for path in entries {
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.is_empty() || bytes.len() % (hidden as usize * 4) != 0 {
            return Err(format!("{}: not [rows][{hidden}] f32", path.display()));
        }
        let name = format!("in.prompt.voice.{}", path.file_stem().unwrap_or_default().to_string_lossy());
        tables.push(packet::devbuild::TensorDecl { name, bytes: bytes.len() as u64, init: Some(bytes) });
    }
    Ok(tables)
}

/// Chatterbox's `punc_norm` plus its `[SPACE]` tokenizer convention, as generic text rules
/// (one per line, tab-separated: kind, args).
pub const T3_TEXT_RULES: &str = "default_if_empty\tYou need to add some text for me to talk.
capitalize_first
collapse_whitespace
replace\t...\t, 
replace\t\u{2026}\t, 
replace\t:\t,
replace\t - \t, 
replace\t;\t, 
replace\t\u{2014}\t-
replace\t\u{2013}\t-
replace\t ,\t,
replace\t\u{201c}\t\"
replace\t\u{201d}\t\"
replace\t\u{2018}\t'
replace\t\u{2019}\t'
trim_end\t 
ensure_suffix\t.!?-,\t.
replace\t \t[SPACE]";
