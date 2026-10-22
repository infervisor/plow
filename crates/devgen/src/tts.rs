//! Speech (TTS) packet contracts: a causal codec-token LM declared as a `tts.codec_lm.v1`
//! pipeline. The programs are the ordinary causal ones; the pipeline adds the numeric contract a
//! backend-neutral speech controller needs (prompt framing, codec frame layout, stop ids,
//! sampling defaults) so plowrt binds a family by driver, never by checkpoint knowledge.

use crate::pipeline::{causal_pipeline_section, CausalPipelineSpec};
use packet::devbuild::{Model, SectionData};
use plow_asset::packet_pipeline::{PacketPipelines, SECTION};

pub const DRIVER: &str = "tts.codec_lm.v1";

/// One speech family's contract. Token ids are the checkpoint tokenizer's.
pub struct SpeechProfile {
    /// Prompt text between the prefix and suffix ids, with `{voice}` and `{input}`.
    pub prompt_template: &'static str,
    /// A voice is valid when this renders to one vocabulary token, unless `voices` lists them.
    pub voice_token: &'static str,
    /// Named voices (multi-token names); empty = any single-token `voice_token`.
    pub voices: &'static [&'static str],
    pub prefix: &'static [u64],
    pub suffix: &'static [u64],
    pub stops: &'static [u64],
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
    /// Default repetition penalty (1.0 = none).
    pub repetition_penalty: f32,
    /// Default top-k (0 = none).
    pub top_k: u64,
    /// Frames come from the packet's depth stage (`fish_depth.json`): each LM token is one frame
    /// (`audio_token_base + class`) whose `frame_codes` codes ride beside it.
    pub depth: bool,
    /// The depth stage's sampler: 0 = Fish (bf16 parameters, RAS, no penalty), 1 = Breeze
    /// (f32 parameters, backbone repetition penalty).
    pub depth_sampler: u64,
    /// Rendered into the template's `{instructions}` with `{instructions}` = the request's text.
    pub instructions: Option<&'static str>,
    /// Sub-bundle of an encoder-mode packet whose `act.enc_out` rows are the whole prompt's
    /// embeddings (the `overlay` role): the prompt ids only size the prefill.
    pub text_encoder: Option<&'static str>,
    /// Voice cloning through the reference encoder packet.
    pub clone: Option<CloneLayout>,
}

/// A cloning prompt: `head`, the reference transcript's prompt text, `mid`, the reference frames
/// (one frame token each, its codes in `in.frame_codes`), an end frame whose every code is
/// `eos_code` (if any), `tail`, then the text's prompt text and the suffix. `head` / `tail` =
/// `None` keep the prompt prefix.
pub struct CloneLayout {
    pub head: Option<&'static [u64]>,
    pub mid: &'static [u64],
    pub tail: Option<&'static [u64]>,
    pub eos_code: Option<u64>,
}

/// maya-research/Veena (model card): [SOH] <spk_v> text [EOH] [SOA] [SOS] -> SNAC codes, stop on
/// END_OF_SPEECH / END_OF_AI.
pub const VEENA: SpeechProfile = SpeechProfile {
    prompt_template: "<spk_{voice}> {input}",
    voice_token: "<spk_{voice}>",
    voices: &[],
    prefix: &[128259],
    suffix: &[128260, 128261, 128257],
    stops: &[128258, 128262],
    sample_rate: 24000,
    frame_codes: 7,
    codebook: 4096,
    frame_samples: 2048,
    audio_token_base: 128266,
    per_char_frames: 1.3,
    // The model card's 700 (8.3 s) clipped inputs from ~110 characters. At 1400 the cap binds
    // past 151 characters, and the runtime splits longer inputs into segments.
    max_new_tokens_cap: 1400,
    temperature: 0.4,
    top_p: 0.9,
    repetition_penalty: 1.0,
    top_k: 0,
    depth: false,
    depth_sampler: 0,
    instructions: None,
    text_encoder: None,
    clone: None,
};

/// canopylabs/orpheus-3b-0.1-ft (orpheus_tts `_format_prompt`): [SOH] BOS "voice: text" [EOT]
/// [EOH] [SOA] [SOS] -> the same 7-code SNAC frames as Veena, stop on END_OF_SPEECH.
pub const ORPHEUS: SpeechProfile = SpeechProfile {
    prompt_template: "{voice}: {input}",
    voice_token: "{voice}",
    voices: &["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
    prefix: &[128259, 128000],
    suffix: &[128009, 128260, 128261, 128257],
    stops: &[128258],
    sample_rate: 24000,
    frame_codes: 7,
    codebook: 4096,
    frame_samples: 2048,
    audio_token_base: 128266,
    // Orpheus voices speak down to ~8 characters/s; 1.3 frames (9 chars/s) clipped the slow tail.
    // The cap binds past 141 characters, and the runtime splits longer inputs into segments.
    per_char_frames: 1.8,
    max_new_tokens_cap: 1800,
    temperature: 0.6,
    top_p: 0.8,
    // H100 c64, 214 requests: 1.0 ran 32 to the token cap (repetition loops, 5/81 clips CER > 0.3);
    // 1.1, 1.2 and 1.3 ran none, all clips CER <= 0.3.
    repetition_penalty: 1.1,
    top_k: 0,
    depth: false,
    depth_sampler: 0,
    instructions: None,
    text_encoder: None,
    clone: None,
};

/// fishaudio/s2-pro (fish-speech `inference.py`, no reference): system "convert the provided text
/// to speech", user "<|speaker:0|>text", assistant "<|voice|>" -> one semantic token per 21.5 Hz
/// frame (151678 + class) plus 9 depth codebooks, stop on <|im_end|>. Sampling: T 0.7, top-p 0.7,
/// top-k 30 (RAS in the depth stage).
pub const FISH: SpeechProfile = SpeechProfile {
    prompt_template: "<|speaker:0|>{input}",
    voice_token: "{voice}",
    voices: &["default"],
    prefix: &[
        151644, 8948, 198, 14166, 279, 3897, 1467, 311, 8806, 151645, 198, 151644, 872, 198,
    ],
    suffix: &[151645, 198, 151644, 77091, 198, 151673],
    stops: &[151645],
    sample_rate: 44100,
    frame_codes: 10,
    codebook: 4096,
    frame_samples: 2048,
    audio_token_base: 151678,
    // 21.5 frames/s over ~11-15 characters/s English; the official per-chunk budget is 1024 frames.
    per_char_frames: 2.4,
    max_new_tokens_cap: 1024,
    temperature: 0.7,
    top_p: 0.7,
    repetition_penalty: 1.0,
    top_k: 30,
    depth: true,
    depth_sampler: 0,
    instructions: None,
    text_encoder: None,
    // `generate_long` with a reference: system "convert the provided text to speech reference to
    // the following:\n\nText:\n<|speaker:0|>ref_text\n\nSpeech:\n" + the reference frames, then
    // the user turn; each text part is tokenized on its own, no end frame.
    clone: Some(CloneLayout {
        head: Some(&[
            151644, 8948, 198, 14166, 279, 3897, 1467, 311, 8806, 5785, 311, 279, 2701, 1447, 1178,
            510,
        ]),
        mid: &[271, 55625, 510],
        tail: Some(&[151645, 198, 151644, 872, 198]),
        eos_code: None,
    }),
};

/// BreezeBlue/Breeze-TTS-2 (`breeze_infer/templates.py`): each text segment `<bos>[S0]text` (voice
/// design: `[S0]<ins_bos>instruction<ins_eos>text`) runs through the T5Gemma2 text encoder; its
/// projected rows are the backbone prompt. One cb0 class per 12.5 Hz frame (EOS = class 2048) plus
/// 15 depth codebooks. Sampling (`runtime.py`, `infer.py`): T 0.9, top-k 50, top-p 1.0,
/// repetition penalty 1.1 on cb0, max 1500 frames.
pub const BREEZE: SpeechProfile = SpeechProfile {
    prompt_template: "[{voice}]{instructions}{input}",
    voice_token: "[{voice}]",
    voices: &["S0"],
    prefix: &[2],
    suffix: &[],
    stops: &[2048],
    sample_rate: 24000,
    frame_codes: 16,
    codebook: 2048,
    frame_samples: 1920,
    audio_token_base: 0,
    // 12.5 frames/s over ~4 characters/s Mandarin (~2.8 frames per character; English needs
    // ~1). Segments of (1500 - 21) / 3.2 = 462 characters stay inside the encoder's 512 rows.
    per_char_frames: 3.2,
    max_new_tokens_cap: 1500,
    temperature: 0.9,
    top_p: 1.0,
    repetition_penalty: 1.1,
    top_k: 50,
    depth: true,
    depth_sampler: 1,
    instructions: Some("<ins_bos>{instructions}<ins_eos>"),
    text_encoder: Some("text_encoder"),
    // `ref_clone_tata`: [TE(<bos>[S0]ref_text)] [ref frames] [audio_eos: codebook_eos_token_id 0
    // on every codebook] [TE(<bos>[S0]text)].
    clone: Some(CloneLayout {
        head: None,
        mid: &[],
        tail: None,
        eos_code: Some(0),
    }),
};

/// Chatterbox T3 (`tts.t3_cfg.v1`): a causal LM whose prefill rows are host embeddings (the
/// `overlay` role, every row) and whose decode embedding adds a learned speech position counted
/// from the per-slot `pos_base`. Classifier-free guidance pairs two slots per request; the
/// contract scalars (text/speech control ids, sampling, guidance weight) ride as parameters.
pub const GUIDED_LM_DRIVER: &str = "tts.guided_lm.v1";
/// Sidecar stage packets beside `model.pkt`, named in the pipeline so the runtime never
/// hard-codes a file name.
pub const CODEC_PACKET: &str = "codec.pkt";

pub fn t3_pipeline_section(
    model: &Model,
    spec: CausalPipelineSpec<'_>,
    pos_base: u32,
    params: &[(String, u64)],
    dir: &std::path::Path,
) -> Result<SectionData, String> {
    use plow_asset::packet_pipeline::{PipelineDType, PipelineTensor};
    let causal = causal_pipeline_section(model, spec)?;
    let mut meta: PacketPipelines =
        serde_json::from_slice(&causal.data).map_err(|e| e.to_string())?;
    let pipe = meta
        .pipelines
        .first_mut()
        .ok_or("causal section has no pipeline")?;
    pipe.driver = GUIDED_LM_DRIVER.into();
    let t = model
        .tensors
        .get(pos_base as usize)
        .ok_or("T3 pos_base tensor is missing")?;
    pipe.tensors.insert(
        "pos_base".into(),
        PipelineTensor {
            name: t.name.clone(),
            dtype: PipelineDType::U32,
            shape: vec![u64::from(spec.decode_capacity)],
        },
    );
    for (k, v) in params {
        let k = if k == "s3_valid_below" {
            "valid_below"
        } else {
            k.as_str()
        };
        pipe.parameters.insert(format!("lm.{k}"), *v);
    }
    // Prefill layout: [voice rows][start_text, text..., stop_text: table + position][BOS x2];
    // the unconditional CFG member keeps positions but drops the text table.
    pipe.strings
        .insert("vocoder.packet".into(), crate::s3gen::PACKET.into());
    pipe.parameters.insert("prompt.bos_repeat".into(), 2);
    pipe.parameters.insert("cfg.uncond_drops_text".into(), 1);
    match t3_text_frontend(dir)? {
        Some(f) => {
            pipe.strings.insert("text.rules".into(), f.rules);
            for (code, rules) in f.languages {
                pipe.strings
                    .insert(format!("text.rules.lang.{code}"), rules);
            }
            pipe.strings
                .insert("text.default_language".into(), f.default_language);
        }
        None => {
            pipe.strings
                .insert("text.rules".into(), T3_TEXT_RULES.into());
        }
    }
    for t in model
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("in.prompt."))
    {
        pipe.tensors.insert(
            t.name.clone(),
            PipelineTensor {
                name: t.name.clone(),
                dtype: PipelineDType::F32,
                shape: vec![t.bytes / 4],
            },
        );
    }
    meta.validate(model.progs.len(), |name| {
        model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.bytes)
    })?;
    Ok(SectionData {
        kind: causal.kind,
        name: SECTION.into(),
        data: serde_json::to_vec(&meta).map_err(|e| e.to_string())?,
    })
}

pub fn profile(name: &str) -> Result<&'static SpeechProfile, String> {
    match name {
        "veena" => Ok(&VEENA),
        "orpheus" => Ok(&ORPHEUS),
        "fish" => Ok(&FISH),
        "breeze" => Ok(&BREEZE),
        other => Err(format!(
            "unknown tts profile {other:?} (known: veena, orpheus, fish, breeze)"
        )),
    }
}

pub fn speech_pipeline_section(
    model: &Model,
    spec: CausalPipelineSpec<'_>,
    p: &SpeechProfile,
    vocab: u32,
) -> Result<SectionData, String> {
    let audio_rows = if p.depth { 1 } else { p.frame_codes };
    if p.audio_token_base + audio_rows * p.codebook > u64::from(vocab) {
        return Err("tts profile audio codes exceed the vocabulary".into());
    }
    let causal = causal_pipeline_section(model, spec)?;
    let mut meta: PacketPipelines =
        serde_json::from_slice(&causal.data).map_err(|e| e.to_string())?;
    let pipe = meta
        .pipelines
        .first_mut()
        .ok_or("causal section has no pipeline")?;
    pipe.name = "speech".into();
    pipe.driver = DRIVER.into();
    pipe.strings
        .insert("prompt.template".into(), p.prompt_template.into());
    pipe.strings
        .insert("prompt.voice_token".into(), p.voice_token.into());
    if !p.voices.is_empty() {
        pipe.strings
            .insert("prompt.voices".into(), p.voices.join("\n"));
    }
    pipe.strings
        .insert("codec.packet".into(), CODEC_PACKET.into());
    if let Some(t) = p.instructions {
        pipe.strings.insert("prompt.instructions".into(), t.into());
    }
    if let Some(t) = p.text_encoder {
        pipe.strings.insert("text.encoder".into(), t.into());
    }
    if let Some(clone) = &p.clone {
        pipe.strings
            .insert("clone.encoder".into(), crate::mimi_encoder::PACKET.into());
        if let Some(code) = clone.eos_code {
            pipe.parameters.insert("clone.eos_code".into(), code);
        }
        // Space-separated ids: a pipeline holds at most 64 parameters.
        for (list, key) in [
            (clone.head, "clone.head"),
            (Some(clone.mid), "clone.mid"),
            (clone.tail, "clone.tail"),
        ] {
            let Some(list) = list else { continue };
            let ids: Vec<String> = list.iter().map(u64::to_string).collect();
            pipe.strings.insert(key.into(), ids.join(" "));
        }
    }
    let params = &mut pipe.parameters;
    let mut put = |k: String, v: u64| {
        params.insert(k, v);
    };
    for (list, key) in [
        (p.prefix, "prompt.prefix"),
        (p.suffix, "prompt.suffix"),
        (p.stops, "stop"),
    ] {
        put(format!("{key}.count"), list.len() as u64);
        for (i, &id) in list.iter().enumerate() {
            put(format!("{key}.{i}"), id);
        }
    }
    put("audio.sample_rate".into(), p.sample_rate);
    put("codec.frame_codes".into(), p.frame_codes);
    put("codec.codebook".into(), p.codebook);
    put("codec.frame_samples".into(), p.frame_samples);
    put("audio.token_base".into(), p.audio_token_base);
    put(
        "tokens.per_char_frames_f32".into(),
        u64::from(p.per_char_frames.to_bits()),
    );
    put("tokens.max_new_cap".into(), p.max_new_tokens_cap);
    put(
        "sampling.temperature_f32".into(),
        u64::from(p.temperature.to_bits()),
    );
    put("sampling.top_p_f32".into(), u64::from(p.top_p.to_bits()));
    put("sampling.top_k".into(), p.top_k);
    put("frames.depth".into(), u64::from(p.depth));
    put("frames.depth_sampler".into(), p.depth_sampler);
    put(
        "sampling.repetition_penalty_f32".into(),
        u64::from(p.repetition_penalty.to_bits()),
    );
    meta.validate(model.progs.len(), |name| {
        model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.bytes)
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
pub fn t3_host_tables(
    dir: &std::path::Path,
    hidden: u32,
) -> Result<Vec<packet::devbuild::TensorDecl>, String> {
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    let f32_tensor = |name: &str, out: &str| -> Result<packet::devbuild::TensorDecl, String> {
        let (dtype, bytes) = reader.read(name)?;
        if dtype != "F32" || bytes.len() % (hidden as usize * 4) != 0 {
            return Err(format!("{name}: expected f32 rows of {hidden}"));
        }
        Ok(packet::devbuild::TensorDecl {
            name: out.into(),
            bytes: bytes.len() as u64,
            init: Some(bytes),
        })
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
        let name = format!(
            "in.prompt.voice.{}",
            path.file_stem().unwrap_or_default().to_string_lossy()
        );
        tables.push(packet::devbuild::TensorDecl {
            name,
            bytes: bytes.len() as u64,
            init: Some(bytes),
        });
    }
    Ok(tables)
}

/// A prep-written multilingual text frontend (`text_frontend.json`): the rule list, one rule list
/// per language code, the default language, and the tables under `text_tables/`.
struct TextFrontend {
    rules: String,
    languages: Vec<(String, String)>,
    default_language: String,
    tables: Vec<String>,
}

fn t3_text_frontend(dir: &std::path::Path) -> Result<Option<TextFrontend>, String> {
    let p = dir.join("text_frontend.json");
    if !p.exists() {
        return Ok(None);
    }
    let raw = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", p.display()))?;
    let bad = |k: &str| format!("{}: bad or missing {k}", p.display());
    let s = |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| bad(k));
    Ok(Some(TextFrontend {
        rules: s("rules")?,
        languages: v["languages"]
            .as_object()
            .ok_or_else(|| bad("languages"))?
            .iter()
            .map(|(k, r)| {
                r.as_str()
                    .map(|r| (k.clone(), r.to_string()))
                    .ok_or_else(|| bad("languages"))
            })
            .collect::<Result<_, _>>()?,
        default_language: s("default_language")?,
        tables: v["tables"]
            .as_array()
            .ok_or_else(|| bad("tables"))?
            .iter()
            .map(|t| t.as_str().map(str::to_string).ok_or_else(|| bad("tables")))
            .collect::<Result<_, _>>()?,
    }))
}

/// The frontend's tables as the `text_tables.v1` metadata section (host data: repeated
/// `u32 name_len, name, u64 len, data`), or `None` without a frontend.
pub fn t3_text_tables_section(dir: &std::path::Path) -> Result<Option<SectionData>, String> {
    let Some(f) = t3_text_frontend(dir)? else {
        return Ok(None);
    };
    let mut data = Vec::new();
    for name in &f.tables {
        let p = dir.join("text_tables").join(name);
        let t = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        data.extend((name.len() as u32).to_le_bytes());
        data.extend(name.as_bytes());
        data.extend((t.len() as u64).to_le_bytes());
        data.extend(t);
    }
    Ok(Some(SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: "text_tables.v1".into(),
        data,
    }))
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
